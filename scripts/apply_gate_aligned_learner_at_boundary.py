#!/usr/bin/env python3
"""Activate gate-aligned learner rollback at one committed RL boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


OFFICIAL_LIBCG_SHA256 = (
    "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def publish(path: Path, **values: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            {
                "schema": "poke_bot.gate_aligned_learner_boundary/v1",
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
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(
            f"cannot read {key} for {unit}: {result.stdout.strip()}"
        )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trainer(staged_root: Path) -> ModuleType:
    source = staged_root / "scripts/train_pure_rl.py"
    root = str(staged_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        "gate_aligned_boundary_trainer", source
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
    migration_dropin: Path,
    migration_reason: str,
    engine: Path,
    expected_source_sha256: str,
    expected_eval_sha256: str,
    expected_dropin_sha256: str,
    expected_engine_sha256: str,
) -> ModuleType:
    paths = (
        (
            staged_root / "scripts/train_pure_rl.py",
            expected_source_sha256,
            "trainer",
        ),
        (
            staged_root / "poke_bot/pure_rl/eval_public.py",
            expected_eval_sha256,
            "gate ranking module",
        ),
        (migration_dropin, expected_dropin_sha256, "migration drop-in"),
        (engine, expected_engine_sha256, "hidden-state engine"),
    )
    for path, expected, label in paths:
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} digest mismatch: expected={expected} actual={actual}"
            )

    text = migration_dropin.read_text(encoding="utf-8")
    required_once = (
        f"WorkingDirectory={staged_root}",
        "RefuseManualStop=no",
        "--continuous-learner-exact-regression-margin 0.01",
        "--continuous-learner-exact-regression-patience 2",
        "--allow-clean-boundary-design-migration",
        f"--boundary-design-migration-reason {migration_reason}",
        # The launch canary validates the unmodified public engine. The
        # private training engine is independently pinned by the sha256sum
        # ExecStartPre; conflating the two digests rejects a valid build.
        f"Environment=POKEBOT_EXPECTED_LIBCG_SHA256={OFFICIAL_LIBCG_SHA256}",
        "sha256sum --check --status",
        "--resume auto",
    )
    for token in required_once:
        if text.count(token) != 1:
            raise RuntimeError(f"migration drop-in must contain one {token!r}")

    module = load_trainer(staged_root)
    for name in ("_exact_gate_regression_streak", "_recover_interrupted_iteration"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"staged trainer lacks {name}")
    report = module._exact_gate_regression_streak(
        history=[
            {
                "iteration": 1,
                "raw_heldout_gate": {
                    "iteration": 1,
                    "gate_id": "canary",
                    "skill_weighted_wr": 0.47,
                    "audit": {"passed": True},
                },
            }
        ],
        current_gate_result={
            "iteration": 2,
            "gate_id": "canary",
            "skill_weighted_wr": 0.47,
            "audit": {"passed": True},
        },
        anchor_evidence={"gate_id": "canary", "win_rate": 0.50},
        regression_margin=0.01,
    )
    if int(report.get("streak", -1)) != 2:
        raise RuntimeError(f"staged regression guard failed canary: {report}")
    return module


def _unlock(path: Path) -> None:
    run(["sudo", "-n", "chattr", "-i", str(path)], timeout=15, check=False)


def _lock(path: Path) -> None:
    run(["sudo", "-n", "chattr", "+i", str(path)], timeout=15)


def install_dropin(source: Path, destination: Path) -> None:
    _unlock(destination.parent)
    if destination.exists():
        _unlock(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, 0o644)
    os.replace(temporary, destination)
    _lock(destination)
    _lock(destination.parent)


def remove_dropin(destination: Path) -> None:
    _unlock(destination.parent)
    if destination.exists():
        _unlock(destination)
        destination.unlink()
    _lock(destination.parent)


def make_dropin_steady(destination: Path, migration_reason: str) -> None:
    _unlock(destination.parent)
    _unlock(destination)
    text = destination.read_text(encoding="utf-8")
    if text.count("RefuseManualStop=no") != 1:
        raise RuntimeError("active migration drop-in lost its stop override")
    pattern = re.compile(
        r" --allow-clean-boundary-design-migration"
        r" --boundary-design-migration-reason " + re.escape(migration_reason)
    )
    if len(pattern.findall(text)) != 1:
        raise RuntimeError("cannot identify one-shot migration authority")
    text = text.replace("RefuseManualStop=no", "RefuseManualStop=yes")
    atomic_text(destination, pattern.sub("", text))
    _lock(destination)
    _lock(destination.parent)


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


def validate_receipt(path: Path, *, target_next_iteration: int) -> dict[str, Any]:
    receipt = load_json(path)
    changed = {str(value) for value in (receipt.get("changed_paths") or [])}
    required = {
        "learner.exact_gate_regression_margin",
        "learner.exact_gate_regression_patience",
        "collection.behavior_policy",
    }
    # Moving the same pinned contracts into the versioned deployment changes
    # only their absolute path fields. Their content identities remain part of
    # the receipt and are already checked by the trainer's migration chain.
    deployment_path_changes = {
        "collection.research_control_phase.registry.path",
        "gates.active_contract.path",
    }
    unexpected = {
        value
        for value in changed
        if value not in required
        and value not in deployment_path_changes
        and not value.startswith("source.")
    }
    if int(receipt.get("boundary_next_iteration", -1)) != target_next_iteration:
        raise RuntimeError("migration receipt committed at the wrong boundary")
    if not required.issubset(changed) or unexpected:
        raise RuntimeError(
            "migration receipt changed unexpected design paths: "
            f"changed={sorted(changed)} unexpected={sorted(unexpected)}"
        )
    current = receipt.get("current_contract")
    learner = current.get("learner") if isinstance(current, dict) else None
    collection = current.get("collection") if isinstance(current, dict) else None
    if not isinstance(learner, dict) or not isinstance(collection, dict):
        raise RuntimeError("migration receipt lacks its resulting design contract")
    if float(learner.get("exact_gate_regression_margin", -1.0)) != 0.01:
        raise RuntimeError("migration receipt has the wrong regression margin")
    if int(learner.get("exact_gate_regression_patience", -1)) != 2:
        raise RuntimeError("migration receipt has the wrong regression patience")
    if str(collection.get("behavior_policy") or "") != (
        "gate_aligned_continuous_learner_with_exact_regression_rollback_v3"
    ):
        raise RuntimeError("migration receipt has the wrong behavior policy")
    return receipt


def recover_partial(module: ModuleType, run_dir: Path) -> str | None:
    state = module._load_loop_state(run_dir)
    if state is None:
        raise RuntimeError("loop state vanished at the boundary")
    recovered = module._recover_interrupted_iteration(run_dir, state)
    return str(recovered) if recovered is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--migration-dropin", type=Path, required=True)
    parser.add_argument("--active-dropin", type=Path, required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--eval-sha256", required=True)
    parser.add_argument("--dropin-sha256", required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    loop_state = args.run_dir / "loop_state.json"
    pass_marker = args.run_dir / "SPECIALIST_GATE_PASSED"
    target_next = int(args.after_iteration) + 1
    module = validate_candidate(
        staged_root=args.staged_root,
        migration_dropin=args.migration_dropin,
        migration_reason=args.migration_reason,
        engine=args.engine,
        expected_source_sha256=args.source_sha256,
        expected_eval_sha256=args.eval_sha256,
        expected_dropin_sha256=args.dropin_sha256,
        expected_engine_sha256=args.engine_sha256,
    )
    observed = int(load_json(loop_state).get("next_iteration", -1))
    if observed > target_next:
        prior_status = load_json(args.status)
        if str(prior_status.get("status") or "") in {
            "complete",
            "superseded_by_exact_gate_pass",
        }:
            return 0
        raise RuntimeError(
            f"requested boundary passed: target={target_next} observed={observed}"
        )
    publish(
        args.status,
        status="validated" if args.validate_only else "waiting_for_boundary",
        after_iteration=int(args.after_iteration),
        target_next_iteration=target_next,
        observed_next_iteration=observed,
        source_sha256=args.source_sha256,
        eval_sha256=args.eval_sha256,
        engine_sha256=args.engine_sha256,
    )
    if args.validate_only:
        return 0

    last_health_check = 0.0
    while True:
        if pass_marker.is_file():
            publish(
                args.status,
                status="superseded_by_exact_gate_pass",
                pass_marker=str(pass_marker),
            )
            return 0
        state = load_json(loop_state)
        completed = int(state.get("last_completed_iteration", -1))
        next_iteration = int(state.get("next_iteration", -1))
        if completed >= int(args.after_iteration):
            if next_iteration != target_next:
                raise RuntimeError(
                    "boundary advanced unexpectedly: "
                    f"completed={completed} next={next_iteration} target={target_next}"
                )
            commit = args.run_dir / "commits" / f"iter_{completed:05d}.json"
            if not commit.is_file():
                raise RuntimeError("loop state advanced without an immutable commit")
            break
        now = time.monotonic()
        if now - last_health_check >= 1.0:
            if service_value(args.unit, "ActiveState") not in (
                "active",
                "activating",
            ):
                raise RuntimeError("production trainer stopped before the boundary")
            last_health_check = now
        time.sleep(max(0.05, float(args.poll_seconds)))

    boundary_state = load_json(loop_state)
    boundary_history = list(boundary_state.get("history") or [])
    boundary_row = boundary_history[-1] if boundary_history else {}
    boundary_gate = (
        boundary_row.get("stage_gate") if isinstance(boundary_row, dict) else None
    )
    if isinstance(boundary_gate, dict) and bool(boundary_gate.get("passed")):
        # The immutable commit and mutable loop pointer precede terminal-marker
        # publication by a few display/retention operations. Never let the
        # boundary watcher race that small window and suppress a valid pass.
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline and not pass_marker.is_file():
            if service_value(args.unit, "ActiveState") == "failed":
                raise RuntimeError(
                    "trainer failed after committing a passed gate but before marker"
                )
            time.sleep(0.1)
        if not pass_marker.is_file():
            raise RuntimeError(
                "passed immutable gate commit did not publish its terminal marker"
            )
        publish(
            args.status,
            status="superseded_by_exact_gate_pass",
            pass_marker=str(pass_marker),
        )
        return 0
    if pass_marker.is_file():
        publish(args.status, status="superseded_by_exact_gate_pass")
        return 0

    migration_started = time.time() - 1.0
    migration_committed = False
    recovered: str | None = None
    try:
        publish(args.status, status="installing_boundary_source")
        install_dropin(args.migration_dropin, args.active_dropin)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        if service_value(args.unit, "RefuseManualStop") != "no":
            raise RuntimeError("migration drop-in did not unlock the boundary stop")
        run(["systemctl", "--user", "stop", args.unit], timeout=75)
        recovered = recover_partial(module, args.run_dir)
        publish(
            args.status,
            status="starting_gate_aligned_source",
            recovered_partial_iteration=recovered,
        )
        run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        run(["systemctl", "--user", "start", args.unit], timeout=75)

        deadline = time.monotonic() + 240.0
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
                raise RuntimeError("gate-aligned trainer failed before migration receipt")
            time.sleep(0.5)
        if receipt_path is None:
            raise RuntimeError("gate-aligned trainer did not commit migration receipt")
        receipt = validate_receipt(
            receipt_path, target_next_iteration=target_next
        )
        migration_committed = True
        make_dropin_steady(args.active_dropin, args.migration_reason)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        if service_value(args.unit, "RefuseManualStop") != "yes":
            raise RuntimeError("steady drop-in did not restore stop protection")
        if service_value(args.unit, "WorkingDirectory") != str(args.staged_root):
            raise RuntimeError("service did not retain the gate-aligned source root")

        pid = int(service_value(args.unit, "MainPID") or 0)
        if pid <= 0 or service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("gate-aligned trainer is not active")
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if int(service_value(args.unit, "MainPID") or 0) != pid:
                raise RuntimeError("gate-aligned trainer restarted during stability check")
            if service_value(args.unit, "ActiveState") != "active":
                raise RuntimeError("gate-aligned trainer left active state")
            time.sleep(1.0)
        publish(
            args.status,
            status="complete",
            migration_receipt=str(receipt_path),
            changed_paths=receipt.get("changed_paths"),
            main_pid=pid,
            observed_next_iteration=int(load_json(loop_state).get("next_iteration", -1)),
            recovered_partial_iteration=recovered,
            migration_authority_revoked=True,
            stop_protection_restored=True,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - fail-safe boundary recovery
        error = f"{type(exc).__name__}: {exc}"
        publish(args.status, status="recovering_after_error", error=error)
        if migration_committed:
            try:
                make_dropin_steady(args.active_dropin, args.migration_reason)
            except BaseException:  # noqa: BLE001
                pass
        else:
            try:
                remove_dropin(args.active_dropin)
            except BaseException:  # noqa: BLE001
                pass
        run(["systemctl", "--user", "daemon-reload"], timeout=30, check=False)
        run(["systemctl", "--user", "reset-failed", args.unit], check=False)
        if service_value(args.unit, "ActiveState") not in ("active", "activating"):
            run(["systemctl", "--user", "start", args.unit], timeout=75, check=False)
        publish(
            args.status,
            status=(
                "migrated_source_recovered"
                if migration_committed
                else "rolled_back_to_previous_source"
            ),
            error=error,
            active_state=service_value(args.unit, "ActiveState"),
            main_pid=int(service_value(args.unit, "MainPID") or 0),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
