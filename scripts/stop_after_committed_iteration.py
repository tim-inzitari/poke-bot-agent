#!/usr/bin/env python3
"""Stop one process group immediately after an immutable RL commit appears."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument("--process-group", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    args = parser.parse_args()
    loop_path = args.run_dir / "loop_state.json"
    target = int(args.after_iteration)
    print(f"BOUNDARY_STOP_ARMED target={target} pgid={args.process_group}", flush=True)
    while True:
        state = read_json(loop_path)
        completed = int(state.get("last_completed_iteration", -1))
        next_iteration = int(state.get("next_iteration", -1))
        if completed >= target:
            commit_path = args.run_dir / "commits" / f"iter_{target:05d}.json"
            commit = read_json(commit_path)
            if (
                int(commit.get("last_completed_iteration", -1)) != target
                or int(commit.get("next_iteration", -1)) != target + 1
            ):
                raise RuntimeError("loop advanced without the exact immutable commit")
            print(
                f"BOUNDARY_STOP_TRIGGER completed={completed} next={next_iteration}",
                flush=True,
            )
            try:
                os.killpg(int(args.process_group), signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        try:
            os.killpg(int(args.process_group), 0)
        except ProcessLookupError as exc:
            raise RuntimeError("trainer process group exited before target commit") from exc
        time.sleep(max(0.02, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
