#!/usr/bin/env python
"""Collect policy seats from official top-ladder replays.

The default consumes the newest pinned daily export (about 5,000 complete
episodes and up to 10,000 acting-seat sequences).  Conversion is deliberately
bounded and streamed: only ``2 * workers`` episode results can be resident in
the parent at once, which avoids repeating the earlier all-results RAM spike.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from poke_bot import archetypes, config, paths
from poke_bot.episodes_index import (
    EPISODES_RAW_DIR,
    download_daily_dataset,
    ensure_episodes_index,
    iter_episode_paths,
    latest_n_days,
    load_daily_manifest,
)
from poke_bot.ladder_replay import LadderReplayClassifier
from poke_bot.replay_import import (
    convert_episode_to_records,
    episode_id_of,
    load_episode_payload,
)


MIX_PATH = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
REPRESENTATIVES_PATH = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)

_WORKER_CLASSIFIER: Optional[LadderReplayClassifier] = None
_ZIP_HANDLES: dict[str, zipfile.ZipFile] = {}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Newest official daily exports to consume (default: pinned latest day).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive YYYY-MM-DD manifest range start (requires --end-date).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive YYYY-MM-DD manifest range end (requires --start-date).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, int(config.HARDWARE.feature_workers)),
        help="Bounded conversion processes (default min(8, feature workers)).",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=0, help="Canary cap; 0 consumes all."
    )
    parser.add_argument(
        "--min-sequences",
        type=int,
        default=5000,
        help="Fail closed below this many usable acting-seat sequences.",
    )
    parser.add_argument(
        "--min-recognized-seat-frac",
        type=float,
        default=0.90,
        help="Fail closed when deck-family coverage is below this fraction.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.DATA_DIR / "bootstrap" / "top_ladder_all.jsonl",
    )
    parser.add_argument("--mix", type=Path, default=MIX_PATH)
    parser.add_argument(
        "--representatives", type=Path, default=REPRESENTATIVES_PATH
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use only already-downloaded daily archives.",
    )
    parser.add_argument(
        "--recognized-only",
        action="store_true",
        help=(
            "Clone only seats assigned to a pinned ladder family. By default "
            "valid unrecognized decks are retained as deck-agnostic policy data."
        ),
    )
    parser.add_argument(
        "--archetype-filter",
        default="",
        help=(
            "Keep only records whose acting seat has this exact classified "
            "archetype (for example: alakazam)."
        ),
    )
    parser.add_argument(
        "--additive-archetype",
        action="append",
        default=[],
        help=(
            "Registered archetype to include in the pinned classifier and "
            "recognized-only output. Repeat for multiple additive families."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing output after a successful rebuild.",
    )
    return parser.parse_args(argv)


def _init_worker(
    mix: str,
    representatives: str,
    card_csv: str,
    additive_archetypes: tuple[str, ...],
) -> None:
    global _WORKER_CLASSIFIER
    # Prevent each CPU conversion worker from creating a BLAS thread team.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _WORKER_CLASSIFIER = LadderReplayClassifier.from_paths(
        mix,
        representatives,
        card_csv=card_csv,
        additive_registered_ids=additive_archetypes,
    )


def _load_payload(ref: str) -> dict[str, Any]:
    if not ref.startswith("zip:"):
        return load_episode_payload(ref)
    body = ref[len("zip:") :]
    zip_path, separator, member = body.partition("::")
    if not separator or not member:
        raise ValueError(f"malformed zip episode reference: {ref!r}")
    archive = _ZIP_HANDLES.get(zip_path)
    if archive is None:
        archive = zipfile.ZipFile(zip_path, "r")
        _ZIP_HANDLES[zip_path] = archive
    return json.loads(archive.read(member).decode("utf-8"))


def _convert_ref(job: tuple[str, str, bool, str]) -> dict[str, Any]:
    ref, source, recognized_only, archetype_filter = job
    classifier = _WORKER_CLASSIFIER
    if classifier is None:
        raise RuntimeError("ladder replay worker was not initialized")
    try:
        payload = _load_payload(ref)
        _decks, labels = classifier.classify_episode(payload)
        label_ids = [label.deck_id for label in labels]
        recognized_ids = (
            *classifier.active_ids,
            *classifier.additive_registered_ids,
        )
        records = convert_episode_to_records(
            payload,
            source=source,
            archetype_filter=archetype_filter or None,
            seat_archetypes=label_ids,
            allowed_archetypes=(
                recognized_ids
                if recognized_only
                else (*recognized_ids, archetypes.UNKNOWN)
            ),
            require_complete=True,
            strict_info_set=True,
        )
        return {
            "episode_id": episode_id_of(payload, fallback=Path(ref).stem),
            "labels": label_ids,
            "methods": [label.method for label in labels],
            "records": records,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - counted and fail-gated by parent
        return {
            "episode_id": Path(ref).stem,
            "labels": [],
            "methods": [],
            "records": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_result(
    result: dict[str, Any],
    handle: Any,
    *,
    active_ids: frozenset[str],
    seen: set[tuple[str, int]],
    stats: dict[str, Any],
) -> None:
    stats["episodes_processed"] += 1
    if result.get("error"):
        stats["episode_errors"] += 1
        error_name = str(result["error"]).split(":", 1)[0]
        stats["error_types"][error_name] += 1
        return

    labels = [str(value) for value in result.get("labels") or []]
    methods = [str(value) for value in result.get("methods") or []]
    stats["seats_seen"] += len(labels)
    stats["recognized_seats"] += sum(label in active_ids for label in labels)
    stats["seat_labels"].update(labels)
    stats["label_methods"].update(methods)

    for record in result.get("records") or []:
        key = (str(record.get("episode_id") or ""), int(record.get("seat", -1)))
        if key in seen:
            stats["duplicate_records"] += 1
            continue
        seen.add(key)
        if not bool(record.get("info_set_ok", False)):
            stats["info_set_failures"] += 1
            continue
        handle.write(
            json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        )
        stats["records_written"] += 1
        stats["decisions_written"] += int(record.get("n_decisions") or 0)
        stats["dropped_incompatible_action_frames"] += int(
            record.get("dropped_incompatible_action_frames") or 0
        )
        stats["record_archetypes"][str(record.get("archetype"))] += 1


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.days <= 0 or args.workers <= 0:
        raise SystemExit("--days and --workers must be positive")
    if not 0.0 <= args.min_recognized_seat_frac <= 1.0:
        raise SystemExit("--min-recognized-seat-frac must be in [0, 1]")

    additive_archetypes = tuple(
        dict.fromkeys(str(value).strip().casefold() for value in args.additive_archetype)
    )
    classifier = LadderReplayClassifier.from_paths(
        args.mix,
        args.representatives,
        card_csv=paths.en_card_data_path(),
        additive_registered_ids=additive_archetypes,
    )
    active_ids = frozenset(
        (*classifier.active_ids, *classifier.additive_registered_ids)
    )
    archetype_filter = str(args.archetype_filter).strip().casefold()
    if archetype_filter and archetype_filter not in active_ids:
        raise SystemExit(
            f"--archetype-filter {archetype_filter!r} is not in the pinned "
            "ladder classifier"
        )
    manifest = load_daily_manifest(ensure_episodes_index())
    if bool(args.start_date) != bool(args.end_date):
        raise SystemExit("--start-date and --end-date must be supplied together")
    if args.start_date and args.end_date:
        if args.start_date > args.end_date:
            raise SystemExit("--start-date must not be after --end-date")
        days = [
            entry
            for entry in manifest
            if args.start_date <= entry.date <= args.end_date
        ]
        if not days or days[0].date != args.start_date or days[-1].date != args.end_date:
            raise SystemExit(
                f"manifest does not cover requested range "
                f"{args.start_date}..{args.end_date}"
            )
    else:
        days = latest_n_days(manifest, args.days)
        if len(days) != args.days:
            raise SystemExit(
                f"requested {args.days} days but manifest has {len(days)}"
            )

    jobs: list[tuple[str, str, bool, str]] = []
    source_rows: list[dict[str, Any]] = []
    EPISODES_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for entry in days:
        if not args.skip_download:
            download_daily_dataset(entry, root=EPISODES_RAW_DIR, unzip=False)
        refs = list(iter_episode_paths(entry, root=EPISODES_RAW_DIR))
        if not refs:
            raise SystemExit(f"no local episodes for {entry.slug}")
        source_rows.append(
            {
                "date": entry.date,
                "slug": entry.slug,
                "manifest_episode_count": entry.episode_count,
                "refs": len(refs),
            }
        )
        jobs.extend(
            (
                str(ref),
                entry.slug,
                bool(args.recognized_only),
                archetype_filter,
            )
            for _episode_id, ref in refs
        )
    if args.max_episodes > 0:
        jobs = jobs[: args.max_episodes]

    out = Path(args.out)
    if out.exists() and not args.replace:
        raise SystemExit(f"output exists (use --replace): {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial.{os.getpid()}")

    stats: dict[str, Any] = {
        "episodes_scheduled": len(jobs),
        "episodes_processed": 0,
        "episode_errors": 0,
        "error_types": Counter(),
        "seats_seen": 0,
        "recognized_seats": 0,
        "seat_labels": Counter(),
        "label_methods": Counter(),
        "records_written": 0,
        "decisions_written": 0,
        "dropped_incompatible_action_frames": 0,
        "record_archetypes": Counter(),
        "duplicate_records": 0,
        "info_set_failures": 0,
    }
    seen: set[tuple[str, int]] = set()
    inflight = max(1, args.workers * 2)

    try:
        with partial.open("w", encoding="utf-8") as handle:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(
                    str(args.mix),
                    str(args.representatives),
                    str(paths.en_card_data_path()),
                    additive_archetypes,
                ),
            ) as pool:
                pending: deque[Future[dict[str, Any]]] = deque()
                next_job = 0
                while next_job < min(inflight, len(jobs)):
                    pending.append(pool.submit(_convert_ref, jobs[next_job]))
                    next_job += 1
                with tqdm(total=len(jobs), desc="top-ladder convert", unit="ep") as bar:
                    while pending:
                        result = pending.popleft().result()
                        _write_result(
                            result,
                            handle,
                            active_ids=active_ids,
                            seen=seen,
                            stats=stats,
                        )
                        if next_job < len(jobs):
                            pending.append(pool.submit(_convert_ref, jobs[next_job]))
                            next_job += 1
                        bar.update(1)
                        bar.set_postfix(
                            seq=stats["records_written"],
                            errors=stats["episode_errors"],
                        )
            handle.flush()
            os.fsync(handle.fileno())

        recognized_frac = stats["recognized_seats"] / max(1, stats["seats_seen"])
        failures: list[str] = []
        if stats["records_written"] < int(args.min_sequences):
            failures.append(
                f"records {stats['records_written']} < minimum {args.min_sequences}"
            )
        if recognized_frac < float(args.min_recognized_seat_frac):
            failures.append(
                f"recognized seat fraction {recognized_frac:.4f} < "
                f"minimum {args.min_recognized_seat_frac:.4f}"
            )
        if stats["info_set_failures"]:
            failures.append(f"info-set failures={stats['info_set_failures']}")
        if stats["episode_errors"] > max(5, int(0.01 * len(jobs))):
            failures.append(f"episode conversion errors={stats['episode_errors']}")
        if failures:
            raise RuntimeError("; ".join(failures))

        os.replace(partial, out)
        meta = {
            "schema": "poke_bot.top_ladder_bootstrap/v1",
            "output": str(out),
            "output_bytes": out.stat().st_size,
            "output_sha256": _sha256(out),
            "sources": source_rows,
            "classifier": classifier.contract,
            "policy_scope": (
                "acting_seat_archetype_exact"
                if archetype_filter
                else "recognized_families_only"
                if args.recognized_only
                else "all_valid_top_ladder_seats"
            ),
            "selection": (
                {
                    "field": "record.archetype",
                    "operator": "exact_casefold",
                    "value": archetype_filter,
                    "seat_semantics": "acting_seat_only",
                }
                if archetype_filter
                else None
            ),
            "quality_gates": {
                "min_sequences": int(args.min_sequences),
                "min_recognized_seat_frac": float(
                    args.min_recognized_seat_frac
                ),
                "recognized_seat_frac": recognized_frac,
                "passed": True,
            },
            "stats": {
                key: dict(value) if isinstance(value, Counter) else value
                for key, value in stats.items()
            },
        }
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)
        return 0
    finally:
        if partial.exists():
            partial.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
