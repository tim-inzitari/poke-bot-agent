#!/usr/bin/env python3
"""Activate the cumulative learner policy at one committed RL boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def publish(path: Path, **values: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            {
                "schema": "poke_bot.learner_retention_boundary/v1",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def run(argv: list[str], *, timeout: float = 90.0, check: bool = True) -> str:
    result = subprocess.run(
        argv,
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


def service_value(unit: str, key: str) -> str:
    return run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        timeout=15,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trainer(staged_root: Path) -> ModuleType:
    source = staged_root / "scripts" / "train_pure_rl.py"
    root = str(staged_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        "learner_retention_boundary_trainer", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import staged trainer: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_candidate(
    *,
    staged_root: Path,
    dropin: Path,
    migration_reason: str,
    expected_source_sha256: str,
) -> ModuleType:
    source = staged_root / "scripts" / "train_pure_rl.py"
    actual_digest = sha256(source)
    if actual_digest != expected_source_sha256:
        raise RuntimeError(
            "staged trainer digest mismatch: "
            f"expected={expected_source_sha256} actual={actual_digest}"
        )
    text = dropin.read_text(encoding="utf-8")
    required = (
        f"WorkingDirectory={staged_root}",
        "--allow-clean-boundary-design-migration",
        f"--boundary-design-migration-reason {migration_reason}",
        "--resume auto",
    )
    for token in required:
        if text.count(token) != 1:
            raise RuntimeError(f"migration drop-in must contain one {token!r}")
    module = load_trainer(staged_root)
    decide = module._continuous_learner_carry_decision
    truth_table = (
        (
            dict(
                candidate_safety_ok=True,
                candidate_safety_reason="continuous_learner",
                heldout_audit_ok=True,
                promoted=False,
            ),
            (True, "continuous_learner_safety_carry"),
        ),
        (
            dict(
                candidate_safety_ok=False,
                candidate_safety_reason="head_to_head_collapse",
                heldout_audit_ok=True,
                promoted=False,
            ),
            (False, "head_to_head_collapse"),
        ),
        (
            dict(
                candidate_safety_ok=True,
                candidate_safety_reason="continuous_learner",
                heldout_audit_ok=False,
                promoted=False,
            ),
            (False, "heldout_contract_audit_failed"),
        ),
    )
    for inputs, expected in truth_table:
        actual = decide(**inputs)
        if actual != expected:
            raise RuntimeError(
                f"continuous learner truth table mismatch: {inputs} -> {actual}"
            )
    return module


def next_iteration(loop_state: Path) -> int:
    return int(load_json(loop_state).get("next_iteration", -1))


def migration_receipt(
    run_dir: Path, *, reason: str, after_mtime: float
) -> Path | None:
    for path in sorted((run_dir / "design_migrations").glob("migration_*.json")):
        try:
            if path.stat().st_mtime < after_mtime:
                continue
        except OSError:
            continue
        if str(load_json(path).get("reason") or "") == reason:
            return path
    return None


def revoke_migration_authority(dropin: Path, reason: str) -> None:
    text = dropin.read_text(encoding="utf-8")
    pattern = re.compile(
        r" --allow-clean-boundary-design-migration"
        r" --boundary-design-migration-reason " + re.escape(reason)
    )
    if len(pattern.findall(text)) != 1:
        raise RuntimeError("cannot identify exactly one migration-authority token")
    atomic_text(dropin, pattern.sub("", text))
    run(["systemctl", "--user", "daemon-reload"], timeout=30)


def recover_uncommitted_iteration(
    module: ModuleType, run_dir: Path
) -> str | None:
    state = module._load_loop_state(run_dir)
    if state is None:
        raise RuntimeError("loop state vanished at boundary")
    recovered = module._recover_interrupted_iteration(run_dir, state)
    return str(recovered) if recovered is not None else None


def validate_receipt(path: Path, *, target: int) -> dict[str, Any]:
    receipt = load_json(path)
    changed = set(receipt.get("changed_paths") or [])
    expected = {
        "collection.behavior_policy",
        "source.source_tree_sha256",
    }
    if int(receipt.get("boundary_next_iteration", -1)) != target:
        raise RuntimeError("migration receipt committed at wrong boundary")
    if changed != expected:
        raise RuntimeError(
            f"migration changed unexpected design fields: {sorted(changed)}"
        )
    return receipt


def wait_for_learner_collection(
    *, loop_state: Path, run_dir: Path, target: int, unit: str, timeout: float
) -> tuple[str, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = load_json(loop_state)
        learner_digest = str((state.get("learner") or {}).get("digest") or "")
        runtime = load_json(run_dir / "iteration_runtime.json")
        pid = int(service_value(unit, "MainPID") or 0)
        if (
            pid > 0
            and learner_digest
            and int(runtime.get("iteration", -1)) == target
            and str(runtime.get("phase") or "") == "collect"
            and str(runtime.get("checkpoint_digest") or "") == learner_digest
        ):
            return learner_digest, pid
        if service_value(unit, "ActiveState") == "failed":
            raise RuntimeError("migrated trainer failed before learner collection")
        time.sleep(0.5)
    raise RuntimeError("next collection did not use the ledger learner digest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--dropin", type=Path, required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    module = validate_candidate(
        staged_root=args.staged_root,
        dropin=args.dropin,
        migration_reason=args.migration_reason,
        expected_source_sha256=args.source_sha256,
    )
    observed = next_iteration(args.loop_state)
    if observed > args.target_next_iteration:
        raise RuntimeError(
            f"requested boundary already passed: target={args.target_next_iteration} "
            f"observed={observed}"
        )
    publish(
        args.status,
        status="validated" if args.validate_only else "waiting_for_boundary",
        target_next_iteration=args.target_next_iteration,
        observed_next_iteration=observed,
        staged_source_sha256=args.source_sha256,
    )
    if args.validate_only:
        return 0

    last_health_check = 0.0
    while next_iteration(args.loop_state) < args.target_next_iteration:
        now = time.monotonic()
        if now - last_health_check >= 1.0:
            if service_value(args.unit, "ActiveState") not in (
                "active",
                "activating",
            ):
                raise RuntimeError("production trainer stopped before boundary")
            last_health_check = now
        time.sleep(max(0.02, args.poll_seconds))

    migration_started = time.time() - 1.0
    migrated = False
    recovered: str | None = None
    try:
        publish(
            args.status,
            status="stopping_at_committed_boundary",
            observed_next_iteration=next_iteration(args.loop_state),
        )
        run(["systemctl", "--user", "stop", args.unit], timeout=60)
        recovered = recover_uncommitted_iteration(module, args.run_dir)
        publish(
            args.status,
            status="starting_migrated_source",
            recovered_partial_iteration=recovered,
        )
        run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        run(["systemctl", "--user", "start", args.unit], timeout=60)

        deadline = time.monotonic() + 180.0
        receipt_path: Path | None = None
        while time.monotonic() < deadline:
            receipt_path = migration_receipt(
                args.run_dir,
                reason=args.migration_reason,
                after_mtime=migration_started,
            )
            if receipt_path is not None:
                break
            if service_value(args.unit, "ActiveState") == "failed":
                raise RuntimeError("v12 failed before committing migration receipt")
            time.sleep(0.5)
        if receipt_path is None:
            raise RuntimeError("v12 did not commit migration receipt")
        receipt = validate_receipt(
            receipt_path, target=args.target_next_iteration
        )
        migrated = True
        revoke_migration_authority(args.dropin, args.migration_reason)
        learner_digest, pid = wait_for_learner_collection(
            loop_state=args.loop_state,
            run_dir=args.run_dir,
            target=args.target_next_iteration,
            unit=args.unit,
            timeout=180.0,
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if service_value(args.unit, "ActiveState") != "active":
                raise RuntimeError("migrated trainer left active state")
            if int(service_value(args.unit, "MainPID") or 0) != pid:
                raise RuntimeError("migrated trainer restarted during stability check")
            time.sleep(1.0)
        publish(
            args.status,
            status="complete",
            migration_receipt=str(receipt_path),
            changed_paths=receipt.get("changed_paths"),
            main_pid=pid,
            observed_next_iteration=next_iteration(args.loop_state),
            collection_behavior_digest=learner_digest,
            recovered_partial_iteration=recovered,
            migration_authority_revoked=True,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - fail-safe recovery path
        error = f"{type(exc).__name__}: {exc}"
        publish(args.status, status="recovering_after_error", error=error)
        run(["systemctl", "--user", "stop", args.unit], timeout=60, check=False)
        if migrated:
            try:
                revoke_migration_authority(args.dropin, args.migration_reason)
            except BaseException:  # noqa: BLE001
                pass
        else:
            try:
                args.dropin.unlink()
            except FileNotFoundError:
                pass
            run(["systemctl", "--user", "daemon-reload"], timeout=30)
        run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        run(["systemctl", "--user", "start", args.unit], timeout=60, check=False)
        publish(
            args.status,
            status=("migrated_v12_recovered" if migrated else "rolled_back_to_v10"),
            error=error,
            active_state=service_value(args.unit, "ActiveState"),
            main_pid=int(service_value(args.unit, "MainPID") or 0),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
