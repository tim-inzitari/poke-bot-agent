#!/usr/bin/env python3
"""Bound one exact r229 game child without changing its runtime semantics."""

from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence


def run(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    grace_seconds: float,
    idle_timeout_seconds: float | None = None,
) -> int:
    if not argv:
        raise ValueError("watchdog requires a child command")
    if idle_timeout_seconds is not None and float(idle_timeout_seconds) <= 0:
        raise ValueError("idle timeout must be positive")
    child = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert child.stdout is not None
    output: queue.Queue[bytes | None] = queue.Queue()

    def pump() -> None:
        try:
            for line in iter(child.stdout.readline, b""):
                output.put(line)
        finally:
            output.put(None)

    reader = threading.Thread(target=pump, name="r229-watchdog-output", daemon=True)
    reader.start()
    started = last_progress = time.monotonic()
    timeout_kind: str | None = None
    while True:
        now = time.monotonic()
        wall_remaining = float(timeout_seconds) - (now - started)
        idle_remaining = (
            float("inf")
            if idle_timeout_seconds is None
            else float(idle_timeout_seconds) - (now - last_progress)
        )
        if wall_remaining <= 0:
            timeout_kind = "TIMEOUT"
            break
        if idle_remaining <= 0:
            timeout_kind = "IDLE_TIMEOUT"
            break
        try:
            line = output.get(timeout=min(wall_remaining, idle_remaining, 1.0))
        except queue.Empty:
            if child.poll() is not None:
                break
            continue
        if line is None:
            break
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        last_progress = time.monotonic()
    if timeout_kind is not None:
        # This group contains only the exact evaluation child created above.
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=float(grace_seconds))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        print(
            f"R229_GAME_WATCHDOG_{timeout_kind} seconds="
            f"{float(idle_timeout_seconds if timeout_kind == 'IDLE_TIMEOUT' else timeout_seconds):g}",
            file=sys.stderr,
            flush=True,
        )
        return 124
    return child.wait()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--idle-timeout-seconds", type=float)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(
        command,
        timeout_seconds=args.timeout_seconds,
        grace_seconds=args.grace_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
