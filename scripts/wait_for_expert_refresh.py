#!/usr/bin/env python3
"""Wait for validated expert archives, then atomically exec a worker command.

The refresh status is the authority: observing a filename is insufficient
because Kaggle writes the final ZIP while it is still downloading.  This
wrapper launches downstream featurization only after every required date has
an explicit ``ready`` receipt and a nonempty archive at the recorded path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import date
from pathlib import Path


REFRESH_SCHEMA = "poke_bot.expert_boundary_refresh/v1"


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("--end must not precede --start")
    return [
        date.fromordinal(value).isoformat()
        for value in range(first.toordinal(), last.toordinal() + 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument(
        "--wait-unit-inactive",
        default="",
        help="Also wait until this user-systemd unit is no longer active.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    required = set(_dates(args.start, args.end))
    deadline = time.monotonic() + float(args.timeout_seconds)

    while True:
        try:
            payload = json.loads(args.status.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {}
        schema = str(payload.get("schema") or "")
        if schema and schema != REFRESH_SCHEMA:
            raise SystemExit(f"unexpected refresh status schema: {schema!r}")
        if payload.get("state") == "failed":
            raise SystemExit(f"expert refresh failed: {payload.get('error')}")
        ready: set[str] = set()
        for row in payload.get("ready") or []:
            if row.get("state") != "ready":
                continue
            path = Path(str(row.get("path") or ""))
            if path.is_file() and path.stat().st_size > 0:
                ready.add(str(row.get("date") or ""))
        missing = sorted(required - ready)
        waiting_unit = False
        if args.wait_unit_inactive:
            probe = subprocess.run(
                ["systemctl", "--user", "is-active", args.wait_unit_inactive],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            waiting_unit = probe.returncode == 0
        if not missing and not waiting_unit:
            print(
                json.dumps(
                    {
                        "state": "launching",
                        "validated_dates": sorted(required),
                        "command": command,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            os.execv(command[0], command)
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for validated dates: {missing}")
        print(
            json.dumps(
                {
                    "state": "waiting",
                    "ready": len(required) - len(missing),
                    "required": len(required),
                    "next_missing": missing[0] if missing else None,
                    "wait_unit_active": waiting_unit,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
