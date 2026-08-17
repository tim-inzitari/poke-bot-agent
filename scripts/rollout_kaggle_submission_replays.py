#!/usr/bin/env python3
"""Full authoritative visual-trace rollout for archived Kaggle submission replays.

Converts each episode-*-replay.json under the submission-replay archive into
per-episode JSONL decision records via convert_visual_episode (omniscient
visualize timeline → masked obs + exact private aux targets). Failures are
recorded per episode so partial progress stays durable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

from poke_bot import archetypes
from poke_bot.authoritative_visual_trace import (
    VisualTraceError,
    convert_visual_episode,
)
from poke_bot.replay_import import extract_setup_decks

SCHEMA = "poke_bot.kaggle_submission_replay_rollout/v1"
ALL_ARCHETYPES = "*"


class _RuleClassifier:
    """Minimal current-rule classifier accepted by the visual-trace validator."""

    def classify_episode(
        self, payload: dict[str, Any]
    ) -> tuple[list[list[int] | None], list[SimpleNamespace]]:
        decks = extract_setup_decks(payload)
        labels = [
            SimpleNamespace(
                deck_id=(
                    archetypes.classify_deck(deck)
                    if deck is not None
                    else archetypes.UNKNOWN
                ),
                method="current_rule_classifier",
            )
            for deck in decks
        ]
        return decks, labels


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    temporary = Path(raw)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rollout_one(args: tuple[str, str, str, str]) -> dict[str, Any]:
    replay_path_s, output_path_s, source, required_archetype = args
    replay_path = Path(replay_path_s)
    output_path = Path(output_path_s)
    episode_id = replay_path.name
    if episode_id.startswith("episode-") and episode_id.endswith("-replay.json"):
        episode_id = episode_id[len("episode-") : -len("-replay.json")]

    if output_path.is_file() and output_path.stat().st_size > 0:
        return {
            "episode_id": episode_id,
            "status": "reused",
            "output": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }

    try:
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        result = convert_visual_episode(
            payload,
            _RuleClassifier(),
            source=source,
            required_archetype=required_archetype,
        )
    except (VisualTraceError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        return {
            "episode_id": episode_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "replay": str(replay_path),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    os.close(fd)
    temporary = Path(raw)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in result.records:
                stream.write(
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False)
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "episode_id": episode_id,
        "status": "complete",
        "output": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "records": len(result.records),
        "decisions": sum(int(row.get("n_decisions") or 0) for row in result.records),
        "transitions_validated": int(
            result.stats.get("transitions_validated") or 0
        ),
        "exact_target_rows": int(result.stats.get("exact_target_rows") or 0),
        "stats": result.stats,
    }


def _submission_dirs(replay_root: Path) -> list[Path]:
    dirs = []
    for path in sorted(replay_root.iterdir()):
        if path.is_dir() and path.name.isdigit():
            dirs.append(path)
    return dirs


def _load_new_downloads(path: Path) -> dict[int, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: dict[int, set[str]] = {}
    for row in payload.get("submissions") or []:
        submission_id = int(row.get("submission_id") or 0)
        if submission_id <= 0:
            continue
        ids = {
            str(int(value))
            for value in (row.get("episode_ids") or [])
            if str(value).strip()
        }
        if ids:
            selected[submission_id] = ids
    return selected


def _rollout_submission(
    submission_dir: Path,
    output_root: Path,
    *,
    workers: int,
    required_archetype: str,
    force: bool,
    episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    submission_id = submission_dir.name
    out_dir = output_root / submission_id
    out_dir.mkdir(parents=True, exist_ok=True)
    replays = sorted(submission_dir.glob("episode-*-replay.json"))
    if episode_ids is not None:
        replays = [
            replay
            for replay in replays
            if replay.name[len("episode-") : -len("-replay.json")] in episode_ids
        ]
    jobs: list[tuple[str, str, str, str]] = []
    for replay in replays:
        episode_id = replay.name[len("episode-") : -len("-replay.json")]
        output = out_dir / f"episode-{episode_id}.jsonl"
        if force and output.exists():
            output.unlink()
        # Zero-byte stubs from interrupted runs are not durable completions.
        if output.is_file() and output.stat().st_size == 0:
            output.unlink()
        jobs.append(
            (
                str(replay),
                str(output),
                f"kaggle-submission-{submission_id}",
                required_archetype,
            )
        )

    results: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            results.append(_rollout_one(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_rollout_one, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda row: str(row.get("episode_id") or ""))
    complete = [row for row in results if row.get("status") == "complete"]
    reused = [row for row in results if row.get("status") == "reused"]
    failed = [row for row in results if row.get("status") == "failed"]
    status = "complete" if not failed else ("partial" if (complete or reused) else "failed")
    receipt = {
        "schema": SCHEMA,
        "status": status,
        "created_at_utc": _utc_now(),
        "submission_id": int(submission_id),
        "required_archetype": required_archetype,
        "replay_count": len(replays),
        "selected_episode_ids": sorted(episode_ids) if episode_ids is not None else None,
        "new_downloads_only": episode_ids is not None,
        "complete_count": len(complete),
        "reused_count": len(reused),
        "failed_count": len(failed),
        "records": sum(int(row.get("records") or 0) for row in complete),
        "decisions": sum(int(row.get("decisions") or 0) for row in complete),
        "transitions_validated": sum(
            int(row.get("transitions_validated") or 0) for row in complete
        ),
        "exact_target_rows": sum(
            int(row.get("exact_target_rows") or 0) for row in complete
        ),
        "failures": [
            {"episode_id": row.get("episode_id"), "error": row.get("error")}
            for row in failed
        ],
        "episodes": [
            {
                key: row[key]
                for key in (
                    "episode_id",
                    "status",
                    "output",
                    "bytes",
                    "sha256",
                    "records",
                    "decisions",
                    "error",
                )
                if key in row
            }
            for row in results
        ],
    }
    _atomic_json(out_dir / "ROLLOUT_RECEIPT.json", receipt)
    return {
        "submission_id": int(submission_id),
        "status": status,
        "replay_count": len(replays),
        "complete_count": len(complete),
        "reused_count": len(reused),
        "failed_count": len(failed),
        "records": receipt["records"],
        "decisions": receipt["decisions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path(
            os.environ.get(
                "POKEBOT_SUBMISSION_REPLAY_ARCHIVE",
                "/srv/poke-bot-agent/archive/submission-replays",
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "POKEBOT_SUBMISSION_REPLAY_ROLLOUT_ROOT",
                "/srv/poke-bot-agent/archive/submission-replay-rollouts",
            )
        ),
    )
    parser.add_argument(
        "--submission",
        action="append",
        type=int,
        default=None,
        help="Optional submission id filter(s)",
    )
    parser.add_argument(
        "--required-archetype",
        default=os.environ.get("POKEBOT_SUBMISSION_REPLAY_ROLLOUT_ARCHETYPE", ALL_ARCHETYPES),
        help='Archetype filter for acting seats; "*" = all recognized',
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("POKEBOT_SUBMISSION_REPLAY_ROLLOUT_WORKERS", "4")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--new-downloads-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trace only episode ids listed in NEW_DOWNLOADS.json from the latest sync",
    )
    parser.add_argument(
        "--new-downloads-manifest",
        type=Path,
        default=None,
        help="Override path to NEW_DOWNLOADS.json (default: <replay-root>/NEW_DOWNLOADS.json)",
    )
    args = parser.parse_args()

    replay_root = args.replay_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not replay_root.is_dir():
        raise SystemExit(f"replay root missing: {replay_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    selected_by_submission: dict[int, set[str]] | None = None
    if args.new_downloads_only:
        manifest = (
            args.new_downloads_manifest.expanduser().resolve()
            if args.new_downloads_manifest is not None
            else replay_root / "NEW_DOWNLOADS.json"
        )
        if not manifest.is_file():
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "new_downloads_manifest_missing",
                        "manifest": str(manifest),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        selected_by_submission = _load_new_downloads(manifest)
        if not selected_by_submission:
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "no_new_downloads",
                        "manifest": str(manifest),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0

    dirs = _submission_dirs(replay_root)
    if args.submission:
        wanted = {str(value) for value in args.submission}
        dirs = [path for path in dirs if path.name in wanted]
    if selected_by_submission is not None:
        dirs = [
            path
            for path in dirs
            if int(path.name) in selected_by_submission
        ]
    if not dirs:
        print(json.dumps({"status": "empty", "submission_ids": []}, sort_keys=True))
        return 0

    summaries: list[dict[str, Any]] = []
    for submission_dir in dirs:
        episode_ids = (
            selected_by_submission.get(int(submission_dir.name))
            if selected_by_submission is not None
            else None
        )
        print(
            f"[rollout] begin submission_id={submission_dir.name} "
            f"new_downloads_only={args.new_downloads_only} "
            f"episodes={len(episode_ids) if episode_ids is not None else 'all'}",
            flush=True,
        )
        summary = _rollout_submission(
            submission_dir,
            output_root,
            workers=max(1, args.workers),
            required_archetype=str(args.required_archetype),
            force=bool(args.force),
            episode_ids=episode_ids,
        )
        summaries.append(summary)
        print("[rollout] done " + json.dumps(summary, sort_keys=True), flush=True)

    status = "complete"
    if any(row.get("status") == "failed" for row in summaries):
        status = "failed"
    elif any(row.get("status") == "partial" for row in summaries):
        status = "partial"
    registry = {
        "schema": SCHEMA,
        "status": status,
        "updated_at_utc": _utc_now(),
        "replay_root": str(replay_root),
        "output_root": str(output_root),
        "required_archetype": str(args.required_archetype),
        "submission_ids": [int(path.name) for path in dirs],
        "submissions": summaries,
    }
    _atomic_json(output_root / "registry.json", registry)
    print(json.dumps(registry, sort_keys=True), flush=True)
    # Partial is success for the pipeline (durable progress); only hard-fail all-failed.
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
