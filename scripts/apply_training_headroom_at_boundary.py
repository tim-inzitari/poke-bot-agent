#!/usr/bin/env python3
"""Install a safer RL decision cap immediately after a committed iteration."""

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
        "schema": "poke_bot.training_headroom_boundary/v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 180.0,
    check: bool = True,
) -> str:
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
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result.stdout.strip()


def next_iteration(path: Path) -> int:
    try:
        return int(json.loads(path.read_text()).get("next_iteration", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def service_value(unit: str, key: str) -> str:
    return run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        timeout=15,
        check=False,
    ).strip()


def migration_unit(post_migration_text: str, reason: str) -> str:
    if "--allow-clean-boundary-design-migration" in post_migration_text:
        raise RuntimeError("staged unit already contains one-time migration authority")
    needle = " --resume auto\n"
    replacement = (
        " --resume auto --allow-clean-boundary-design-migration "
        f"--boundary-design-migration-reason {reason}\n"
    )
    if post_migration_text.count(needle) != 1:
        raise RuntimeError("could not identify exactly one ExecStart resume boundary")
    return post_migration_text.replace(needle, replacement)


def wait_for_receipt(
    run_dir: Path,
    *,
    reason: str,
    after_mtime: float,
    unit: str,
) -> Path:
    deadline = time.monotonic() + 240.0
    while time.monotonic() < deadline:
        for receipt in sorted((run_dir / "design_migrations").glob("migration_*.json")):
            try:
                if receipt.stat().st_mtime < after_mtime:
                    continue
                payload = json.loads(receipt.read_text())
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if str(payload.get("reason") or "") == reason:
                return receipt
        if service_value(unit, "ActiveState") == "failed":
            break
        time.sleep(0.5)
    raise RuntimeError("training headroom migration did not record a receipt")


def validate_staged_unit(text: str, expected_decisions: int) -> None:
    expected = f"--train-max-decisions-per-batch {int(expected_decisions)}"
    if text.count(expected) != 1:
        raise RuntimeError(f"staged unit must contain exactly one {expected!r}")
    if "--train-max-decisions-per-batch 20480" in text:
        raise RuntimeError("staged unit still contains the crash-prone 20,480 cap")
    if "--train-games-per-batch 240" not in text:
        raise RuntimeError("staged unit unexpectedly changed the game batch cap")
    if "--base-checkpoint" in text or "--initial-learner-checkpoint" in text:
        raise RuntimeError("staged unit pins a stale bootstrap checkpoint")
    if "--allow-clean-boundary-design-migration" in text:
        raise RuntimeError("staged unit contains persistent one-time authority")
    for guard in ("MemoryHigh=96G", "MemoryMax=112G", "MemorySwapMax=0"):
        if guard not in text:
            raise RuntimeError(f"staged unit lost memory guard {guard}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-unit", type=Path, required=True)
    parser.add_argument("--active-unit", type=Path, required=True)
    parser.add_argument("--base-unit", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--expected-max-decisions", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    args = parser.parse_args()

    publish(
        args.status,
        status="waiting_for_committed_boundary",
        observed_next_iteration=next_iteration(args.loop_state),
        target_next_iteration=args.target_next_iteration,
        expected_max_decisions=args.expected_max_decisions,
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
        expected_max_decisions=args.expected_max_decisions,
    )
    run(["systemctl", "--user", "stop", args.unit], timeout=60)
    try:
        publish(args.status, status="validating_headroom_migration")
        staged_text = args.staged_unit.read_text()
        validate_staged_unit(staged_text, args.expected_max_decisions)
        tests = [
            "tests/test_pure_rl_recovery_and_scheduling.py::test_clean_boundary_migration_audits_operational_batch_update",
            "tests/test_runtime_and_metrics.py::test_oom_split_processes_both_halves_completely",
        ]
        run(
            [str(args.python), "-m", "pytest", "-q", *tests, "--maxfail=1"],
            cwd=args.test_root,
            timeout=90,
        )

        publish(args.status, status="installing_one_time_migration")
        args.active_unit.write_text(migration_unit(staged_text, args.migration_reason))
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        run(["systemctl", "--user", "reset-failed", args.unit], timeout=15, check=False)
        migration_started = time.time() - 1.0
        run(["systemctl", "--user", "start", args.unit], timeout=60)
        receipt = wait_for_receipt(
            args.run_dir,
            reason=args.migration_reason,
            after_mtime=migration_started,
            unit=args.unit,
        )

        # The running process consumed the authority. Revoke it for every
        # future automatic restart while leaving that process uninterrupted.
        shutil.copy2(args.staged_unit, args.active_unit)
        shutil.copy2(args.staged_unit, args.base_unit)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)

        deadline = time.monotonic() + 25.0
        first_pid = 0
        while time.monotonic() < deadline:
            state = service_value(args.unit, "ActiveState")
            pid = int(service_value(args.unit, "MainPID") or 0)
            if state == "failed":
                raise RuntimeError("migrated trainer failed during stability window")
            if state == "active" and pid > 0:
                first_pid = first_pid or pid
                if pid != first_pid:
                    raise RuntimeError("migrated trainer restarted during stability window")
            time.sleep(1.0)
        if service_value(args.unit, "ActiveState") != "active" or first_pid <= 0:
            raise RuntimeError("migrated trainer is not active")

        publish(
            args.status,
            status="complete",
            migration_receipt=str(receipt),
            main_pid=first_pid,
            observed_next_iteration=next_iteration(args.loop_state),
            expected_max_decisions=args.expected_max_decisions,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - rollback covers every failure
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
