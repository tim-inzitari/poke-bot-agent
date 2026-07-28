#!/usr/bin/env python3
"""Build independent daily guide shards concurrently under one safe receipt.

Each day is an isolated subprocess running the established authoritative day
materializer. The parent owns the window lock and atomically publishes ordered
progress; failed or interrupted runs resume checksum-backed daily artifacts
instead of recomputing completed days.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any


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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _completed_row(
    *,
    day: str,
    output: Path,
    specialist_id: str,
    guide_id: str,
) -> dict[str, Any] | None:
    receipt_path = output.with_name(output.name + ".receipt.json")
    if not receipt_path.is_file() or not output.is_file():
        return None
    receipt = _read(receipt_path)
    selection = dict(receipt.get("selection") or {})
    stats = dict(receipt.get("stats") or {})
    target_coverage = stats.get("target_coverage")
    records = int(stats.get("records_kept", 0))
    decisions = int(stats.get("decisions_kept", 0))
    guide_rows_present = (
        isinstance(target_coverage, dict)
        and "guide_rows" in target_coverage
    )
    guide_rows = (
        int(target_coverage["guide_rows"])
        if guide_rows_present
        else (0 if records == 0 and decisions == 0 else -1)
    )
    if not (
        receipt.get("format") == "pokebot-authoritative-visual-day-receipt"
        and int(receipt.get("format_version", -1)) == 1
        and receipt.get("source_date") == day
        and selection.get("acting_seat_archetype") == specialist_id
        and selection.get("current_deck_guide") == guide_id
        and guide_rows >= 0
    ):
        return None
    return {
        "date": day,
        "output": str(output),
        "sha256": (receipt.get("output") or {}).get("sha256"),
        "records": records,
        "decisions": decisions,
        "guide_rows": guide_rows,
        "zero_guide_rows": guide_rows == 0,
        "resumed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--required-archetype", required=True)
    parser.add_argument("--current-deck-guide", required=True)
    parser.add_argument(
        "--logical-alias",
        action="append",
        default=[],
        help="Forward a checksum-bound SOURCE=TARGET classifier alias.",
    )
    parser.add_argument(
        "--authoritative-deck-catalog",
        type=Path,
        action="append",
        default=[],
        help="Forward a checksum-bound public exact-deck archetype catalog.",
    )
    parser.add_argument("--day-parallelism", type=int, default=4)
    parser.add_argument("--workers-per-day", type=int, default=3)
    parser.add_argument("--max-in-flight-per-day", type=int, default=6)
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--memory-floor-gib", type=float, default=16.0)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument(
        "--day-script",
        type=Path,
        default=Path(__file__).with_name(
            "materialize_authoritative_alakazam_day.py"
        ),
    )
    args = parser.parse_args()
    if args.day_parallelism < 1 or args.workers_per_day < 1:
        raise ValueError("parallelism and workers must be positive")

    days = _dates(args.start, args.end)
    archive_dir = args.archive_dir.resolve()
    out_dir = args.out_dir.resolve()
    status_path = args.status.resolve()
    day_script = args.day_script.resolve()
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("authoritative guide window is already running") from exc

        started = time.time()
        completed: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        for day in days:
            output = out_dir / f"{args.required_archetype}-{day}.features"
            row = _completed_row(
                day=day,
                output=output,
                specialist_id=args.required_archetype,
                guide_id=args.current_deck_guide,
            )
            if row is None:
                pending.append(day)
            else:
                completed[day] = row

        running: dict[str, tuple[subprocess.Popen, Any]] = {}

        def publish(state: str, *, error: str | None = None) -> None:
            ordered = [completed[day] for day in days if day in completed]
            payload: dict[str, Any] = {
                "schema": STATUS_SCHEMA,
                "state": state,
                "pid": os.getpid(),
                "started_at": started,
                "updated_at": time.time(),
                "date_window": {
                    "start": days[0],
                    "end": days[-1],
                    "days": len(days),
                },
                "parallel_contract": {
                    "day_parallelism": int(args.day_parallelism),
                    "workers_per_day": int(args.workers_per_day),
                    "maximum_worker_processes": (
                        int(args.day_parallelism)
                        * int(args.workers_per_day)
                    ),
                    "independent_daily_outputs": True,
                },
                "current_dates": sorted(running),
                "completed": ordered,
                "totals": {
                    "days": len(ordered),
                    "records": sum(row["records"] for row in ordered),
                    "decisions": sum(row["decisions"] for row in ordered),
                    "guide_rows": sum(row["guide_rows"] for row in ordered),
                },
            }
            if error:
                payload["error"] = error
            if state == "complete":
                payload["completed_at"] = time.time()
                payload["elapsed_seconds"] = time.time() - started
            _atomic_json(status_path, payload)

        publish("running")
        try:
            while pending or running:
                while pending and len(running) < int(args.day_parallelism):
                    day = pending.pop(0)
                    archive = (
                        archive_dir
                        / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
                    )
                    if not archive.is_file():
                        raise FileNotFoundError(archive)
                    output = (
                        out_dir
                        / f"{args.required_archetype}-{day}.features"
                    )
                    log_handle = (
                        logs / f"{args.required_archetype}-{day}.log"
                    ).open("ab", buffering=0)
                    command = [
                        sys.executable,
                        str(day_script),
                        "--date",
                        day,
                        "--archive",
                        str(archive),
                        "--out",
                        str(output),
                        "--workers",
                        str(args.workers_per_day),
                        "--max-in-flight",
                        str(args.max_in_flight_per_day),
                        "--max-context",
                        str(args.max_context),
                        "--memory-floor-gib",
                        str(args.memory_floor_gib),
                        "--min-records",
                        str(args.min_records),
                        "--required-archetype",
                        args.required_archetype,
                        "--current-deck-guide",
                        args.current_deck_guide,
                    ]
                    for alias in args.logical_alias:
                        command.extend(["--logical-alias", alias])
                    for catalog in args.authoritative_deck_catalog:
                        command.extend(
                            ["--authoritative-deck-catalog", str(catalog)]
                        )
                    running[day] = (
                        subprocess.Popen(
                            command,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                        ),
                        log_handle,
                    )
                publish("running")
                if not running:
                    continue
                time.sleep(1.0)
                for day, (process, handle) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    del running[day]
                    if code:
                        raise RuntimeError(
                            f"daily guide materializer failed: "
                            f"date={day} exit={code}"
                        )
                    output = (
                        out_dir
                        / f"{args.required_archetype}-{day}.features"
                    )
                    row = _completed_row(
                        day=day,
                        output=output,
                        specialist_id=args.required_archetype,
                        guide_id=args.current_deck_guide,
                    )
                    if row is None:
                        raise RuntimeError(
                            f"daily guide receipt invalid after success: {day}"
                        )
                    row["resumed"] = False
                    completed[day] = row
                    print(json.dumps(row, sort_keys=True), flush=True)
            publish("complete")
            return 0
        except BaseException as exc:
            for process, handle in running.values():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                handle.close()
            publish("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
