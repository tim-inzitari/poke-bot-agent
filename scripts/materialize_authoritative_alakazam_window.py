#!/usr/bin/env python3
"""Build a resumable date window of authoritative Alakazam expert shards."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config, paths
from poke_bot.authoritative_visual_trace import materialize_day
from poke_bot.ladder_replay import LadderReplayClassifier


DEFAULT_MIX = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
DEFAULT_REPRESENTATIVES = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)
STATUS_SCHEMA = "pokebot-authoritative-visual-window-status/v1"


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("--end must not precede --start")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last YYYY-MM-DD")
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--mix", type=Path, default=DEFAULT_MIX)
    parser.add_argument(
        "--representatives", type=Path, default=DEFAULT_REPRESENTATIVES
    )
    parser.add_argument("--card-csv", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=config.MODEL.max_context)
    parser.add_argument("--memory-floor-gib", type=float, default=16.0)
    parser.add_argument("--min-records", type=int, default=1)
    args = parser.parse_args()

    days = _dates(args.start, args.end)
    archive_dir = args.archive_dir.resolve()
    out_dir = args.out_dir.resolve()
    status_path = args.status.resolve()
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("authoritative window materializer is already running") from exc

        classifier = LadderReplayClassifier.from_paths(
            args.mix,
            args.representatives,
            card_csv=args.card_csv or paths.en_card_data_path(),
        )
        started = time.time()
        completed: list[dict[str, Any]] = []
        base_status: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "state": "running",
            "pid": os.getpid(),
            "started_at": started,
            "updated_at": started,
            "date_window": {"start": days[0], "end": days[-1], "days": len(days)},
            "completed": completed,
        }
        _atomic_json(status_path, base_status)

        try:
            for day in days:
                archive = archive_dir / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
                output = out_dir / f"alakazam-{day}.features"
                if not archive.is_file():
                    raise FileNotFoundError(archive)
                _atomic_json(
                    status_path,
                    {
                        **base_status,
                        "current_date": day,
                        "updated_at": time.time(),
                        "completed": completed,
                    },
                )
                receipt = materialize_day(
                    archive,
                    output,
                    classifier=classifier,
                    source_date=day,
                    workers=int(args.workers),
                    max_in_flight=int(args.max_in_flight),
                    max_context=int(args.max_context),
                    resume=True,
                    min_available_bytes=int(args.memory_floor_gib * 1024**3),
                    min_records=int(args.min_records),
                )
                completed.append(
                    {
                        "date": day,
                        "output": str(output),
                        "sha256": (receipt.get("output") or {}).get("sha256"),
                        "records": int((receipt.get("stats") or {}).get("records_kept", 0)),
                        "decisions": int((receipt.get("stats") or {}).get("decisions_kept", 0)),
                        "resumed": bool(receipt.get("resumed")),
                    }
                )
                print(json.dumps(completed[-1], sort_keys=True), flush=True)
            finished = time.time()
            final = {
                **base_status,
                "state": "complete",
                "current_date": None,
                "updated_at": finished,
                "completed_at": finished,
                "elapsed_seconds": finished - started,
                "completed": completed,
                "totals": {
                    "days": len(completed),
                    "records": sum(row["records"] for row in completed),
                    "decisions": sum(row["decisions"] for row in completed),
                },
            }
            _atomic_json(status_path, final)
            print(json.dumps(final, indent=2, sort_keys=True), flush=True)
            return 0
        except BaseException as exc:
            _atomic_json(
                status_path,
                {
                    **base_status,
                    "state": "failed",
                    "updated_at": time.time(),
                    "completed": completed,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
