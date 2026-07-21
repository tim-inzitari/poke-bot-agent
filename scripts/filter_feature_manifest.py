#!/usr/bin/env python3
"""Create a durable, checksummed single-archetype feature manifest.

The source feature shards already contain one causal ``GameSequence`` per
acting seat.  Filtering those streams is therefore lossless and cheap: keep a
sequence only when *that seat's* ``archetype`` equals the requested deck ID.
No opponent-seat decisions leak into the specialist corpus and no replay is
re-simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION
from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot.features import FEATURE_SCHEMA_VERSION


PINNED_CORPUS_SCHEMA = "poke_bot.pinned_expert_corpus/v1"
PINNED_CORPUS_NAME = "PROTECTED_EXPERT_CORPUS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal_filtered_manifest(manifest_path: Path) -> Path:
    """Pin a completed filtered corpus to one immutable manifest digest.

    The resulting file is also a valid input to
    :func:`poke_bot.pure_rl.expert_rehearsal.resolve_expert_manifest`. Periodic
    rehearsal therefore fails closed if ``manifest.json`` is ever replaced,
    while the large feature shards remain reusable without being copied.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = dict(manifest.get("selection") or {})
    quality = dict(manifest.get("quality_gates") or {})
    totals = dict(manifest.get("totals") or {})
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or selection.get("field") != "GameSequence.archetype"
        or selection.get("seat_semantics") != "acting_seat_only"
        or not str(selection.get("value") or "").strip()
        or quality.get("passed") is not True
        or quality.get("acting_seat_archetype_exact") is not True
        or int(totals.get("records_kept") or 0) <= 0
        or int(totals.get("decisions_kept") or 0) <= 0
    ):
        raise ValueError("refusing to seal an invalid filtered feature manifest")
    pointer = {
        "schema": PINNED_CORPUS_SCHEMA,
        "protected": True,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "selection": selection,
        "totals": {
            "bytes": int(totals.get("bytes") or 0),
            "records_kept": int(totals["records_kept"]),
            "decisions_kept": int(totals["decisions_kept"]),
        },
    }
    pointer_path = manifest_path.parent / PINNED_CORPUS_NAME
    if pointer_path.exists():
        existing = json.loads(pointer_path.read_text(encoding="utf-8"))
        if existing != pointer:
            raise RuntimeError(
                "existing protected expert-corpus pointer differs from manifest"
            )
    else:
        atomic_json(pointer_path, pointer)
    return pointer_path


def _source_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = pickle.load(stream)
    if not isinstance(header, dict) or header.get("format") != SHARD_FORMAT:
        raise ValueError(f"invalid source feature shard: {path}")
    return header


def _output_is_valid(
    output: Path,
    sidecar: Path,
    *,
    source_digest: str,
    archetype: str,
) -> dict[str, Any] | None:
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not output.is_file()
        or metadata.get("source_sha256") != source_digest
        or metadata.get("selection_archetype") != archetype
        or metadata.get("sha256") != sha256(output)
    ):
        return None
    # Full stream validation catches a valid hash over a malformed/truncated
    # artifact before it can be admitted into the immutable manifest.
    count = sum(1 for _ in iter_feature_shard(output))
    if count != int((metadata.get("stats") or {}).get("records_kept", -1)):
        return None
    return metadata


def filter_feature_shard(
    source: Path,
    output: Path,
    *,
    expected_source_digest: str,
    archetype: str,
) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    requested = str(archetype).strip().lower()
    if not requested:
        raise ValueError("archetype cannot be empty")
    actual_source_digest = sha256(source)
    if actual_source_digest != str(expected_source_digest):
        raise ValueError(
            f"source feature digest mismatch: {source}: "
            f"expected={expected_source_digest} actual={actual_source_digest}"
        )
    sidecar = output.with_suffix(output.suffix + ".json")
    existing = _output_is_valid(
        output,
        sidecar,
        source_digest=actual_source_digest,
        archetype=requested,
    )
    if existing is not None:
        return existing
    if output.exists() or sidecar.exists():
        raise FileExistsError(
            f"non-idempotent filtered output already exists: {output}"
        )

    source_header = _source_header(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{os.getpid()}.partial")
    scanned = kept = decisions = 0
    seats: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    header = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "compact_mode": COMPACT_MODE,
        "source_feature_shard": source.name,
        "source_sha256": actual_source_digest,
        "source_dates": list(source_header.get("source_dates") or []),
        "max_context": int(source_header.get("max_context") or 0),
        "selection": "sequence.archetype == selection_archetype",
        "selection_archetype": requested,
    }
    try:
        with partial.open("xb") as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for sequence in iter_feature_shard(source):
                scanned += 1
                if str(sequence.archetype).strip().lower() != requested:
                    continue
                if not sequence.info_set_ok:
                    raise ValueError(
                        f"selected sequence failed info-set guard: {sequence.episode_id}"
                    )
                pickle.dump(sequence, stream, protocol=pickle.HIGHEST_PROTOCOL)
                kept += 1
                decisions += len(sequence)
                seats[str(int(sequence.seat))] += 1
                opponents[str(sequence.opp_archetype)] += 1
            stats = {
                # This is a derived selection, so every output record is usable.
                "records_total": kept,
                "records_kept": kept,
                "records_dropped": 0,
                "decisions_kept": decisions,
                "drop_reasons": {},
                "decisions_truncated": 0,
                "policy_targets_padded": 0,
                "policy_targets_truncated": 0,
                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE,
                "source_records_scanned": scanned,
                "source_records_excluded": scanned - kept,
                "seat_counts": dict(sorted(seats.items())),
                "opponent_archetypes": dict(sorted(opponents.items())),
                "selection_archetype": requested,
            }
            footer = {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            }
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if kept <= 0 or decisions <= 0:
        output.unlink(missing_ok=True)
        raise ValueError(f"filter selected no usable {requested} sequences: {source}")
    metadata = {
        **header,
        "path": output.name,
        "bytes": int(output.stat().st_size),
        "sha256": sha256(output),
        "source_records_scanned": scanned,
        "source_sha256": actual_source_digest,
        "selection_archetype": requested,
        "stats": stats,
    }
    atomic_json(sidecar, metadata)
    return metadata


def _filter_one(job: tuple[str, str, str, str]) -> dict[str, Any]:
    source, output, digest, archetype = job
    return filter_feature_shard(
        Path(source),
        Path(output),
        expected_source_digest=digest,
        archetype=archetype,
    )


def filter_manifest(
    source_manifest: Path,
    output_dir: Path,
    *,
    archetype: str,
    workers: int = 1,
) -> Path:
    source_manifest = Path(source_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("source is not a bootstrap feature manifest")
    if int(payload.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported source feature manifest version")
    requested = str(archetype).strip().lower()
    shards = list(payload.get("shards") or [])
    if not shards:
        raise ValueError("source feature manifest has no shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, str, str]] = []
    for row in shards:
        source = (source_manifest.parent / str(row["path"])).resolve()
        stem = source.name[: -len(".features")] if source.name.endswith(".features") else source.stem
        output = output_dir / f"{stem}.{requested}.features"
        jobs.append((str(source), str(output), str(row["sha256"]), requested))
    if int(workers) <= 1:
        filtered = [_filter_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
            filtered = list(pool.map(_filter_one, jobs))
    filtered.sort(key=lambda row: tuple(row.get("source_dates") or ()))
    totals = {
        "bytes": sum(int(row["bytes"]) for row in filtered),
        "records_kept": sum(int(row["stats"]["records_kept"]) for row in filtered),
        "records_total": sum(int(row["stats"]["records_total"]) for row in filtered),
        "decisions_kept": sum(int(row["stats"]["decisions_kept"]) for row in filtered),
        "source_records_scanned": sum(
            int(row["stats"]["source_records_scanned"]) for row in filtered
        ),
    }
    if totals["records_kept"] <= 0 or totals["decisions_kept"] <= 0:
        raise ValueError("filtered manifest would be empty")
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "date_start": payload.get("date_start"),
        "date_end": payload.get("date_end"),
        "dates": list(payload.get("dates") or []),
        "selection": {
            "field": "GameSequence.archetype",
            "operator": "exact_casefold",
            "value": requested,
            "seat_semantics": "acting_seat_only",
        },
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "shards": filtered,
        "totals": totals,
        "quality_gates": {
            "passed": True,
            "nonempty": True,
            "checksummed": True,
            "acting_seat_archetype_exact": True,
        },
    }
    output_manifest = output_dir / "manifest.json"
    if output_manifest.exists():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing filtered manifest differs from rebuilt manifest")
    else:
        atomic_json(output_manifest, manifest)
    seal_filtered_manifest(output_manifest)
    return output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = filter_manifest(
        args.source_manifest,
        args.output_dir,
        archetype=args.archetype,
        workers=max(1, int(args.workers)),
    )
    print(json.dumps(json.loads(manifest.read_text()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
