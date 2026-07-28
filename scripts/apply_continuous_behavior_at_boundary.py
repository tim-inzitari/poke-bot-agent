#!/usr/bin/env python3
"""Move an active RL lineage to continuous-learner rollouts at a clean boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


def publish(path: Path, **values: Any) -> None:
    payload = {
        "schema": "poke_bot.continuous_behavior_boundary/v1",
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


def service_value(unit: str, key: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    return result.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def next_iteration(path: Path) -> int:
    return int(load_json(path).get("next_iteration", -1))


def migration_unit(stable_text: str, reason: str) -> str:
    if "--allow-clean-boundary-design-migration" in stable_text:
        raise RuntimeError("staged unit already contains one-time migration authority")
    needle = " --resume auto\n"
    replacement = (
        " --resume auto --allow-clean-boundary-design-migration "
        f"--boundary-design-migration-reason {reason}\n"
    )
    if stable_text.count(needle) != 1:
        raise RuntimeError("could not identify exactly one resume boundary")
    return stable_text.replace(needle, replacement)


def rollback_unit(stable_text: str) -> str:
    old = "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v4"
    fallback = "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v2"
    if stable_text.count(old) != 1:
        raise RuntimeError("staged unit does not name exactly one v4 deployment")
    return stable_text.replace(old, fallback)


def validate_staged_unit(text: str, staged_root: Path) -> None:
    required = (
        "WorkingDirectory=/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v4",
        "--resume auto",
        "pure_rl_alakazam_resident_v9_20260720/loop_state.json",
        " --train-device-resident ",
        "--train-max-decisions-per-batch 32768",
    )
    for token in required:
        if text.count(token) != 1:
            raise RuntimeError(f"staged unit must contain exactly one {token!r}")
    forbidden = (
        "--base-checkpoint",
        "--initial-learner-checkpoint",
        "--allow-clean-boundary-design-migration",
    )
    for token in forbidden:
        if token in text:
            raise RuntimeError(f"staged unit contains forbidden persistent token {token!r}")
    source = (staged_root / "scripts" / "train_pure_rl.py").read_text(
        encoding="utf-8"
    )
    for token in (
        '"behavior_policy": "continuous_learner_after_safety_carry"',
        "ckpt=learner_after.path",
        "digest=learner_after.digest",
        "learner_after.path,\n                    learner_after.digest",
    ):
        if token not in source:
            raise RuntimeError(f"staged trainer lacks continuous behavior token {token!r}")


def load_trainer_module(staged_root: Path) -> ModuleType:
    source = staged_root / "scripts" / "train_pure_rl.py"
    root_text = str(staged_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("boundary_train_pure_rl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import staged trainer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover_uncommitted_iteration(staged_root: Path, run_dir: Path) -> str | None:
    module = load_trainer_module(staged_root)
    state = module._load_loop_state(run_dir)
    if state is None:
        raise RuntimeError("loop state vanished at the committed boundary")
    recovered = module._recover_interrupted_iteration(run_dir, state)
    remaining = module._iteration_artifact_paths(
        run_dir, int(state.get("next_iteration", -1))
    )
    if remaining:
        raise RuntimeError(f"next-iteration artifacts remain after recovery: {remaining}")
    return str(recovered) if recovered is not None else None


def wait_for_receipt(
    run_dir: Path,
    *,
    reason: str,
    after_mtime: float,
    unit: str,
    timeout: float = 240.0,
) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for receipt in sorted((run_dir / "design_migrations").glob("migration_*.json")):
            try:
                if receipt.stat().st_mtime < after_mtime:
                    continue
                payload = load_json(receipt)
            except OSError:
                continue
            if str(payload.get("reason") or "") == reason:
                return receipt
        if service_value(unit, "ActiveState") == "failed":
            break
        time.sleep(0.5)
    raise RuntimeError("continuous-behavior migration did not record a receipt")


def wait_for_collection_digest(
    *,
    loop_state: Path,
    runtime: Path,
    unit: str,
    timeout: float = 180.0,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = load_json(loop_state)
        learner = dict(state.get("learner") or {})
        learner_digest = str(learner.get("digest") or "")
        active = load_json(runtime)
        if (
            active.get("phase") == "collect"
            and int(active.get("iteration", -1)) == int(state.get("next_iteration", -2))
            and learner_digest
            and str(active.get("checkpoint_digest") or "") == learner_digest
        ):
            return learner_digest
        if service_value(unit, "ActiveState") == "failed":
            raise RuntimeError("migrated trainer failed before collection started")
        time.sleep(0.5)
    raise RuntimeError("next collection did not publish the ledger learner digest")


def install_unit(path: Path, text: str, unit: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    run(["systemctl", "--user", "daemon-reload"], timeout=30)
    run(["systemctl", "--user", "reset-failed", unit], timeout=15, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--staged-unit", type=Path, required=True)
    parser.add_argument("--active-unit", type=Path, required=True)
    parser.add_argument("--base-unit", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args()

    stable_text = args.staged_unit.read_text(encoding="utf-8")
    validate_staged_unit(stable_text, args.staged_root)
    rollback_text = rollback_unit(stable_text)
    backup_dir = args.status.parent / "continuous-behavior-boundary-backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.active_unit, backup_dir / "active.service")
    if args.base_unit.is_file():
        shutil.copy2(args.base_unit, backup_dir / "base.service")

    publish(
        args.status,
        status="waiting_for_committed_boundary",
        observed_next_iteration=next_iteration(args.loop_state),
        target_next_iteration=args.target_next_iteration,
    )
    last_health_check = 0.0
    while next_iteration(args.loop_state) < args.target_next_iteration:
        now = time.monotonic()
        if now - last_health_check >= 5.0:
            if service_value(args.unit, "ActiveState") not in ("active", "activating"):
                raise RuntimeError("production trainer stopped before the requested boundary")
            last_health_check = now
        time.sleep(max(0.02, args.poll_seconds))

    if (args.run_dir / "SPECIALIST_GATE_PASSED").is_file():
        publish(args.status, status="skipped_terminal_gate_already_passed")
        return 0

    migrated = False
    try:
        publish(
            args.status,
            status="stopping_at_committed_boundary",
            observed_next_iteration=next_iteration(args.loop_state),
        )
        run(["systemctl", "--user", "stop", args.unit], timeout=60)
        recovered = recover_uncommitted_iteration(args.staged_root, args.run_dir)
        publish(
            args.status,
            status="installing_one_time_migration",
            recovered_partial_iteration=recovered,
        )

        tests = (
            "tests/test_pure_rl_deferred_weight_publish.py",
            "tests/test_pure_rl_recovery_and_scheduling.py",
            "tests/test_pure_rl_lineage_retention.py",
            "tests/test_pure_rl_replay_cache.py",
        )
        run(
            [str(args.python), "-m", "pytest", "-q", *tests, "--maxfail=1"],
            cwd=args.staged_root,
            timeout=120,
        )

        install_unit(
            args.active_unit,
            migration_unit(stable_text, args.migration_reason),
            args.unit,
        )
        migration_started = time.time() - 1.0
        run(["systemctl", "--user", "start", args.unit], timeout=60)
        receipt = wait_for_receipt(
            args.run_dir,
            reason=args.migration_reason,
            after_mtime=migration_started,
            unit=args.unit,
        )
        migrated = True

        # Revoke one-time authority while the migrated process keeps running.
        install_unit(args.active_unit, stable_text, args.unit)
        args.base_unit.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.staged_unit, args.base_unit)
        learner_digest = wait_for_collection_digest(
            loop_state=args.loop_state,
            runtime=args.run_dir / "iteration_runtime.json",
            unit=args.unit,
        )

        first_pid = int(service_value(args.unit, "MainPID") or 0)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if service_value(args.unit, "ActiveState") != "active":
                raise RuntimeError("migrated trainer left active state")
            if int(service_value(args.unit, "MainPID") or 0) != first_pid:
                raise RuntimeError("migrated trainer restarted during stability window")
            time.sleep(1.0)

        publish(
            args.status,
            status="complete",
            migration_receipt=str(receipt),
            main_pid=first_pid,
            observed_next_iteration=next_iteration(args.loop_state),
            collection_behavior_digest=learner_digest,
            recovered_partial_iteration=recovered,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - recovery must cover all failures
        error = f"{type(exc).__name__}: {exc}"
        publish(args.status, status="recovering_after_error", error=error)
        if migrated:
            install_unit(args.active_unit, stable_text, args.unit)
        else:
            install_unit(args.active_unit, rollback_text, args.unit)
        run(["systemctl", "--user", "start", args.unit], timeout=60, check=False)
        publish(
            args.status,
            status=("migrated_v4_restarted" if migrated else "rolled_back_to_v2"),
            error=error,
            active_state=service_value(args.unit, "ActiveState"),
            main_pid=int(service_value(args.unit, "MainPID") or 0),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
