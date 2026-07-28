#!/usr/bin/env python3
"""Install the tested low-water scheduler immediately after an RL commit."""

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
        "schema": "poke_bot.queue_refill_boundary/v1",
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
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result.stdout.strip()


def loop_next_iteration(path: Path) -> int:
    try:
        return int(json.loads(path.read_text()).get("next_iteration", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def service_property(unit: str, name: str) -> str:
    return run(
        ["systemctl", "--user", "show", unit, "-p", name, "--value"],
        timeout=15,
        check=False,
    ).strip()


def migration_unit(post_migration_text: str, reason: str) -> str:
    if "--allow-clean-boundary-design-migration" in post_migration_text:
        raise RuntimeError("staged post-migration unit already contains one-time authority")
    needle = " --resume auto\n"
    replacement = (
        " --resume auto --allow-clean-boundary-design-migration "
        f"--boundary-design-migration-reason {reason}\n"
    )
    if post_migration_text.count(needle) != 1:
        raise RuntimeError("could not identify exactly one ExecStart resume boundary")
    return post_migration_text.replace(needle, replacement)


def wait_for_migration(run_dir: Path, reason: str, after_mtime: float) -> Path:
    # Production preflight plus a systemd restart backoff can legitimately
    # exceed 90 seconds. Keep rollback bounded without racing a healthy retry.
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        for receipt in sorted((run_dir / "design_migrations").glob("migration_*.json")):
            try:
                if receipt.stat().st_mtime < after_mtime:
                    continue
                payload = json.loads(receipt.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if str(payload.get("reason") or "") == reason:
                return receipt
        if service_property("pokebot-pure-rl-continuous-rehearsal.service", "ActiveState") == "failed":
            break
        time.sleep(0.5)
    raise RuntimeError("new scheduler did not record its clean-boundary migration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-source", type=Path, required=True)
    parser.add_argument("--deployment-source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--staged-training-source", type=Path, required=True)
    parser.add_argument("--deployment-training-source", type=Path, required=True)
    parser.add_argument("--base-training-source", type=Path, required=True)
    parser.add_argument("--staged-unit", type=Path, required=True)
    parser.add_argument("--active-unit", type=Path, required=True)
    parser.add_argument("--base-unit", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()

    publish(
        args.status,
        status="waiting_for_committed_boundary",
        observed_next_iteration=loop_next_iteration(args.loop_state),
        target_next_iteration=args.target_next_iteration,
    )
    while loop_next_iteration(args.loop_state) < args.target_next_iteration:
        time.sleep(max(0.1, args.poll_seconds))

    backup_dir = args.status.parent / "queue-refill-boundary-backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_paths = {
        "deployment_source": backup_dir / "remote_jobs.deployment.py",
        "base_source": backup_dir / "remote_jobs.base.py",
        "deployment_training_source": backup_dir / "train_pure_rl.deployment.py",
        "base_training_source": backup_dir / "train_pure_rl.base.py",
        "active_unit": backup_dir / "active.service",
        "base_unit": backup_dir / "base.service",
    }
    for source, destination in (
        (args.deployment_source, backup_paths["deployment_source"]),
        (args.base_source, backup_paths["base_source"]),
        (
            args.deployment_training_source,
            backup_paths["deployment_training_source"],
        ),
        (args.base_training_source, backup_paths["base_training_source"]),
        (args.active_unit, backup_paths["active_unit"]),
        (args.base_unit, backup_paths["base_unit"]),
    ):
        shutil.copy2(source, destination)

    publish(
        args.status,
        status="stopping_at_committed_boundary",
        observed_next_iteration=loop_next_iteration(args.loop_state),
    )
    run(["systemctl", "--user", "stop", args.unit], timeout=60)
    try:
        publish(args.status, status="validating_staged_scheduler")
        tests = [
            "tests/test_training_memory_containment.py::test_remote_result_queue_is_two_ram_waves_plus_bounded_disk",
            "tests/test_training_memory_containment.py::test_remote_result_overflow_spills_and_is_removed",
            "tests/test_remote_demand_caps.py::test_remote_low_water_reserve_is_bounded_and_half_full",
            "tests/test_remote_checkpoint_staging.py::test_scheduled_initial_claim_round_reaches_each_endpoint",
            "tests/test_remote_checkpoint_staging.py::test_scheduled_demand_shrink_finishes_claimed_chunks",
            "tests/test_remote_checkpoint_staging.py::test_low_water_refills_elmo_and_bert_in_the_same_probe_round",
            "tests/test_ladder_deck_mix.py::test_measurement_decks_are_a_strict_ordered_subset",
            "tests/test_ladder_deck_mix.py::test_measurement_decks_fail_closed_on_unknown_or_duplicate_ids",
            "tests/test_training_memory_containment.py::test_continuous_unit_resumes_promotions_and_pins_measurement_decks",
            "tests/test_pure_rl_recovery_and_scheduling.py::test_clean_boundary_migration_allows_explicit_measurement_deck_change",
        ]
        run(
            [str(args.python), "-m", "py_compile", "poke_bot/remote_jobs.py"],
            cwd=args.test_root,
            timeout=30,
        )
        run(
            [str(args.python), "-m", "py_compile", "scripts/train_pure_rl.py"],
            cwd=args.test_root,
            timeout=30,
        )
        run(
            [str(args.python), "-m", "pytest", "-q", *tests, "--maxfail=1"],
            cwd=args.test_root,
            timeout=60,
        )

        publish(args.status, status="installing_scheduler_and_one_time_migration")
        shutil.copy2(args.staged_source, args.deployment_source)
        shutil.copy2(args.staged_source, args.base_source)
        shutil.copy2(args.staged_training_source, args.deployment_training_source)
        shutil.copy2(args.staged_training_source, args.base_training_source)
        post_unit = args.staged_unit.read_text()
        args.active_unit.write_text(migration_unit(post_unit, args.migration_reason))
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        run(["systemctl", "--user", "reset-failed", args.unit], timeout=15)
        migration_started = time.time() - 1.0
        run(["systemctl", "--user", "start", args.unit], timeout=60)
        receipt = wait_for_migration(
            args.run_dir, args.migration_reason, migration_started
        )

        # Revoke the one-time authority immediately.  The running process has
        # already consumed it; daemon-reload changes only future starts.
        shutil.copy2(args.staged_unit, args.active_unit)
        shutil.copy2(args.staged_unit, args.base_unit)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        if service_property(args.unit, "ActiveState") != "active":
            raise RuntimeError("new scheduler is not active after migration")
        publish(
            args.status,
            status="complete",
            migration_receipt=str(receipt),
            main_pid=int(service_property(args.unit, "MainPID") or 0),
            observed_next_iteration=loop_next_iteration(args.loop_state),
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - rollback must cover all failures
        publish(
            args.status,
            status="rolling_back",
            error=f"{type(exc).__name__}: {exc}",
        )
        shutil.copy2(backup_paths["deployment_source"], args.deployment_source)
        shutil.copy2(backup_paths["base_source"], args.base_source)
        shutil.copy2(
            backup_paths["deployment_training_source"],
            args.deployment_training_source,
        )
        shutil.copy2(
            backup_paths["base_training_source"], args.base_training_source
        )
        shutil.copy2(backup_paths["active_unit"], args.active_unit)
        shutil.copy2(backup_paths["base_unit"], args.base_unit)
        run(["systemctl", "--user", "daemon-reload"], timeout=30, check=False)
        run(["systemctl", "--user", "reset-failed", args.unit], timeout=15, check=False)
        run(["systemctl", "--user", "start", args.unit], timeout=60, check=False)
        publish(
            args.status,
            status="rolled_back_and_old_scheduler_restarted",
            error=f"{type(exc).__name__}: {exc}",
            active_state=service_property(args.unit, "ActiveState"),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
