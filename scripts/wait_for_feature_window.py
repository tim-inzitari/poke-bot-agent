#!/usr/bin/env python3
"""Wait for exact, non-overlapping feature dates, then exec final assembly."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("--end must not precede --start")
    return [
        date.fromordinal(value).isoformat()
        for value in range(first.toordinal(), last.toordinal() + 1)
    ]


def _available_dates(staging: Path) -> set[str]:
    owners: dict[str, Path] = {}
    for sidecar in staging.glob("*.features.json"):
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shard = staging / str(metadata.get("path") or "")
        if not shard.is_file() or shard.stat().st_size <= 0:
            continue
        for day in metadata.get("source_dates") or []:
            day = str(day)
            previous = owners.get(day)
            if previous is not None and previous != sidecar:
                raise RuntimeError(
                    f"overlapping feature date {day}: {previous} and {sidecar}"
                )
            owners[day] = sidecar
    return set(owners)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument(
        "--append-expected-date-args",
        action="store_true",
        help="Append one --expected-date argument per required day to command.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    expected = _dates(args.start, args.end)
    expected_set = set(expected)
    deadline = time.monotonic() + float(args.timeout_seconds)
    while True:
        available = _available_dates(args.staging_dir)
        unexpected = sorted(available - expected_set)
        if unexpected:
            raise SystemExit(f"unexpected dates in feature staging: {unexpected}")
        missing = sorted(expected_set - available)
        if not missing:
            if args.append_expected_date_args:
                for day in expected:
                    command.extend(("--expected-date", day))
            print(
                json.dumps(
                    {"state": "assembling", "dates": expected, "command": command},
                    sort_keys=True,
                ),
                flush=True,
            )
            os.execv(command[0], command)
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for feature dates: {missing}")
        print(
            json.dumps(
                {
                    "state": "waiting",
                    "ready": len(expected) - len(missing),
                    "required": len(expected),
                    "next_missing": missing[0],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
