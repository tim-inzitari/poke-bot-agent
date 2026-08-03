#!/usr/bin/env python3
"""Activate a checksum-staged exact worker profile at a committed boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _property(unit: str, name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, f"--property={name}", "--value"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-state", required=True, type=Path)
    parser.add_argument("--commit", required=True, type=Path)
    parser.add_argument("--unit-source", required=True, type=Path)
    parser.add_argument("--unit-target", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--target-next-iteration", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    expected = int(args.workers)
    required_lines = (
        f"Environment=PURE_RL_SIM_WORKERS={expected}",
        f"Environment=PURE_RL_GAMES_IN_FLIGHT={expected}",
        f"Environment=PURE_RL_HELDOUT_LOCAL_WORKERS={expected}",
    )
    unit_text = args.unit_source.read_text(encoding="utf-8")
    if any(line not in unit_text for line in required_lines):
        raise RuntimeError("staged unit does not pin the complete exact worker profile")
    source_digest = _sha256(args.unit_source)

    while True:
        try:
            state = json.loads(args.loop_state.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            state = {}
        next_iteration = int(state.get("next_iteration", -1))
        commit_ready = args.commit.is_file()
        _atomic_json(
            args.status,
            {
                "schema": "poke_bot.heldout_worker_profile_boundary/v1",
                "status": "waiting" if not commit_ready else "applying",
                "target_next_iteration": args.target_next_iteration,
                "observed_next_iteration": next_iteration,
                "commit_ready": commit_ready,
                "workers": expected,
                "unit": args.unit,
                "unit_source_sha256": source_digest,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if commit_ready and next_iteration == args.target_next_iteration:
            break
        time.sleep(max(0.5, args.poll_seconds))

    commit = json.loads(args.commit.read_text(encoding="utf-8"))
    completed_iteration = int(
        commit.get("last_completed_iteration", commit.get("iteration", -1))
    )
    if (
        completed_iteration != args.target_next_iteration - 1
        or int(commit.get("next_iteration", -1)) != args.target_next_iteration
    ):
        raise RuntimeError("boundary commit iteration does not match activation target")

    temporary = args.unit_target.with_suffix(args.unit_target.suffix + ".tmp")
    temporary.write_text(unit_text, encoding="utf-8")
    os.replace(temporary, args.unit_target)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "restart", args.unit], check=True)

    deadline = time.monotonic() + 120.0
    pid = 0
    environment = ""
    while time.monotonic() < deadline:
        try:
            pid = int(_property(args.unit, "MainPID") or 0)
        except ValueError:
            pid = 0
        if pid > 0:
            try:
                environment = (
                    Path(f"/proc/{pid}/environ")
                    .read_bytes()
                    .replace(b"\0", b"\n")
                    .decode()
                )
            except OSError:
                environment = ""
            if all(f"{line.removeprefix('Environment=')}\n" in environment + "\n" for line in required_lines):
                _atomic_json(
                    args.status,
                    {
                        "schema": "poke_bot.heldout_worker_profile_boundary/v1",
                        "status": "applied",
                        "target_next_iteration": args.target_next_iteration,
                        "observed_next_iteration": next_iteration,
                        "boundary_commit": str(args.commit.resolve()),
                        "boundary_commit_sha256": _sha256(args.commit),
                        "workers": expected,
                        "main_pid": pid,
                        "active_state": _property(args.unit, "ActiveState"),
                        "unit": args.unit,
                        "unit_source_sha256": source_digest,
                        "verified_environment": [
                            line.removeprefix("Environment=") for line in required_lines
                        ],
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return 0
        time.sleep(1.0)

    raise RuntimeError("restarted trainer did not expose the exact worker profile")


if __name__ == "__main__":
    raise SystemExit(main())
