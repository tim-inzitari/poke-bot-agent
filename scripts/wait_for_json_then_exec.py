#!/usr/bin/env python3
"""Wait for an atomically published JSON artifact, then exec a command."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    deadline = time.monotonic() + float(args.timeout_seconds)
    while True:
        ready = False
        try:
            payload = json.loads(args.path.read_text(encoding="utf-8"))
            ready = isinstance(payload, dict) and args.path.stat().st_size > 0
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        if ready:
            print(
                json.dumps(
                    {"state": "launching", "artifact": str(args.path), "command": command},
                    sort_keys=True,
                ),
                flush=True,
            )
            os.execv(command[0], command)
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for JSON artifact: {args.path}")
        print(
            json.dumps({"state": "waiting", "artifact": str(args.path)}, sort_keys=True),
            flush=True,
        )
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
