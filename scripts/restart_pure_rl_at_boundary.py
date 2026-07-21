#!/usr/bin/env python3
"""Restart one user service immediately after an immutable RL commit lands."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _last_completed(path: Path) -> int:
    try:
        value = json.loads(path.read_text())
        return int(value.get("last_completed_iteration", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.20)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    old_pid = _systemctl("show", args.service, "-p", "MainPID", "--value").stdout.strip()
    deadline = time.monotonic() + max(1.0, args.timeout_seconds)
    print(
        f"waiting for immutable commit iteration>={args.after_iteration} "
        f"service={args.service} old_pid={old_pid}",
        flush=True,
    )
    while time.monotonic() < deadline:
        completed = _last_completed(args.loop_state)
        if completed >= args.after_iteration:
            print(f"boundary committed iteration={completed}; restarting", flush=True)
            result = _systemctl("restart", args.service)
            print(result.stdout.strip(), flush=True)
            if result.returncode != 0:
                return result.returncode
            for _ in range(150):
                active = _systemctl("is-active", args.service).stdout.strip()
                new_pid = _systemctl(
                    "show", args.service, "-p", "MainPID", "--value"
                ).stdout.strip()
                if active == "active" and new_pid and new_pid != "0" and new_pid != old_pid:
                    print(f"restart healthy new_pid={new_pid}", flush=True)
                    return 0
                time.sleep(0.20)
            print("restart did not become healthy within 30 seconds", flush=True)
            return 1
        time.sleep(max(0.05, args.poll_seconds))
    print("timed out before requested iteration boundary", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
