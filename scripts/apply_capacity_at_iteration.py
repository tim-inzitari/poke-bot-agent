#!/usr/bin/env python3
"""Apply an already-installed systemd profile at a committed RL boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def service_property(unit: str, name: str) -> str:
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
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--target-next-iteration", required=True, type=int)
    parser.add_argument("--expected-workers", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    while True:
        try:
            state = json.loads(args.state.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            state = {}
        next_iteration = int(state.get("next_iteration", -1))
        atomic_json(
            args.status,
            {
                "status": "waiting" if next_iteration < args.target_next_iteration else "applying",
                "observed_next_iteration": next_iteration,
                "target_next_iteration": args.target_next_iteration,
                "expected_workers": args.expected_workers,
                "unit": args.unit,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if next_iteration >= args.target_next_iteration:
            break
        time.sleep(max(0.5, args.poll_seconds))

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "restart", args.unit], check=True)

    deadline = time.monotonic() + 90.0
    pid = 0
    environment = ""
    while time.monotonic() < deadline:
        try:
            pid = int(service_property(args.unit, "MainPID") or 0)
        except ValueError:
            pid = 0
        if pid > 0:
            try:
                environment = Path(f"/proc/{pid}/environ").read_bytes().replace(b"\0", b"\n").decode()
            except OSError:
                environment = ""
            if f"PURE_RL_SIM_WORKERS={args.expected_workers}\n" in environment + "\n":
                atomic_json(
                    args.status,
                    {
                        "status": "applied",
                        "observed_next_iteration": next_iteration,
                        "target_next_iteration": args.target_next_iteration,
                        "workers": args.expected_workers,
                        "main_pid": pid,
                        "active_state": service_property(args.unit, "ActiveState"),
                        "unit": args.unit,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return 0
        time.sleep(1.0)

    atomic_json(
        args.status,
        {
            "status": "failed_verification",
            "observed_next_iteration": next_iteration,
            "target_next_iteration": args.target_next_iteration,
            "expected_workers": args.expected_workers,
            "main_pid": pid,
            "active_state": service_property(args.unit, "ActiveState"),
            "unit": args.unit,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
