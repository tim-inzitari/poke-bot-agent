#!/usr/bin/env python3
"""Switch persistent remote sockets to one-game claims at a committed boundary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def publish(path: Path, **values: Any) -> None:
    payload = {
        "schema": "poke_bot.refill_claim_boundary/v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(argv: list[str], *, cwd: Path | None = None, timeout: float = 90.0) -> str:
    result = subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result.stdout.strip()


def next_iteration(path: Path) -> int:
    try:
        return int(json.loads(path.read_text()).get("next_iteration", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def service_value(unit: str, key: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
        check=False,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-unit", type=Path, required=True)
    parser.add_argument("--active-unit", type=Path, required=True)
    parser.add_argument("--base-unit", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    args = parser.parse_args()

    publish(
        args.status,
        status="waiting_for_committed_boundary",
        observed_next_iteration=next_iteration(args.loop_state),
        target_next_iteration=args.target_next_iteration,
    )
    while next_iteration(args.loop_state) < args.target_next_iteration:
        time.sleep(max(0.05, args.poll_seconds))

    backup_dir = args.status.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    active_backup = backup_dir / "active.service"
    base_backup = backup_dir / "base.service"
    shutil.copy2(args.active_unit, active_backup)
    shutil.copy2(args.base_unit, base_backup)

    publish(
        args.status,
        status="stopping_at_committed_boundary",
        observed_next_iteration=next_iteration(args.loop_state),
    )
    run(["systemctl", "--user", "stop", args.unit], timeout=60)
    try:
        publish(args.status, status="validating_one_game_refill")
        tests = [
            "tests/test_remote_demand_caps.py::test_remote_refill_claim_size_is_configurable",
            "tests/test_training_memory_containment.py::test_continuous_unit_resumes_promotions_and_pins_measurement_decks",
            "tests/test_remote_checkpoint_staging.py::test_low_water_refills_elmo_and_bert_in_the_same_probe_round",
        ]
        run(
            [str(args.python), "-m", "pytest", "-q", *tests, "--maxfail=1"],
            cwd=args.test_root,
            timeout=60,
        )
        unit_text = args.staged_unit.read_text()
        if "Environment=POKEBOT_REMOTE_REFILL_GAMES=1" not in unit_text:
            raise RuntimeError("staged unit does not use one-game refill claims")
        if "--base-checkpoint" in unit_text or "--allow-clean-boundary" in unit_text:
            raise RuntimeError("staged unit contains stale one-time resume authority")

        publish(args.status, status="installing_and_starting")
        shutil.copy2(args.staged_unit, args.active_unit)
        shutil.copy2(args.staged_unit, args.base_unit)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        subprocess.run(
            ["systemctl", "--user", "reset-failed", args.unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        run(["systemctl", "--user", "start", args.unit], timeout=60)
        deadline = time.monotonic() + 30.0
        first_pid = 0
        while time.monotonic() < deadline:
            active = service_value(args.unit, "ActiveState")
            pid = int(service_value(args.unit, "MainPID") or 0)
            if active == "failed":
                raise RuntimeError("one-game refill service failed during startup")
            if active == "active" and pid > 0:
                if first_pid == 0:
                    first_pid = pid
                elif pid == first_pid and time.monotonic() + 10.0 >= deadline:
                    break
            time.sleep(1.0)
        if service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("one-game refill service is not active")
        final_pid = int(service_value(args.unit, "MainPID") or 0)
        if final_pid <= 0 or (first_pid and final_pid != first_pid):
            raise RuntimeError("one-game refill service did not keep a stable PID")
        publish(
            args.status,
            status="complete",
            main_pid=final_pid,
            observed_next_iteration=next_iteration(args.loop_state),
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - rollback must cover all failures
        publish(args.status, status="rolling_back", error=f"{type(exc).__name__}: {exc}")
        shutil.copy2(active_backup, args.active_unit)
        shutil.copy2(base_backup, args.base_unit)
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["systemctl", "--user", "reset-failed", args.unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["systemctl", "--user", "start", args.unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        publish(
            args.status,
            status="rolled_back_and_previous_unit_restarted",
            error=f"{type(exc).__name__}: {exc}",
            active_state=service_value(args.unit, "ActiveState"),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
