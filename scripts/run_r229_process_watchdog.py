#!/usr/bin/env python3
"""Bound one exact r229 game child without changing its runtime semantics."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Sequence


def run(argv: Sequence[str], *, timeout_seconds: float, grace_seconds: float) -> int:
    if not argv:
        raise ValueError("watchdog requires a child command")
    child = subprocess.Popen(list(argv), start_new_session=True)
    try:
        return child.wait(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        # This group contains only the exact evaluation child created above.
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=float(grace_seconds))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        print(
            f"R229_GAME_WATCHDOG_TIMEOUT seconds={float(timeout_seconds):g}",
            file=sys.stderr,
            flush=True,
        )
        return 124


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(
        command,
        timeout_seconds=args.timeout_seconds,
        grace_seconds=args.grace_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
