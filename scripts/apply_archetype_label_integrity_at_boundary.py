#!/usr/bin/env python3
"""Activate canonical matchup labels and isolated research controls at a boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
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


_ALLOWED_MIGRATION_PREFIXES = (
    "source.",
    "collection.behavior_policy",
    "collection.auxiliary_targets.hidden_engine.digest",
    "collection.auxiliary_targets.hidden_engine.size",
    "collection.group_games_per_iteration",
    "collection.official_exploit",
    "collection.official_targeting",
    "collection.research_control_phase",
    "collection.strong_public_practice",
    "expert_rehearsal.loss_weights",
    "gates.active_contract",
    "opponents.collect",
    "opponents.official_target_training",
    "opponents.research_controls",
)

_REQUIRED_V17_MIGRATION_PREFIXES = (
    "collection.group_games_per_iteration",
    "collection.research_control_phase",
    "expert_rehearsal.loss_weights",
    "gates.active_contract",
    "opponents.research_controls",
)

_EXPECTED_EXPERT_LOSS_WEIGHTS = {
    "archetype": 0.05,
    "opponent_hand": 0.05,
    "opponent_hidden_remainder": 0.05,
    "lethal_threat": 0.025,
    "prize_race": 0.025,
    "alakazam_guide": 0.05,
}


def _matches_design_prefix(value: str, prefix: str) -> bool:
    if prefix.endswith("."):
        return value.startswith(prefix)
    return value == prefix or value.startswith(prefix + ".")


def validate_v17_migration_receipt(
    module: ModuleType,
    receipt: dict[str, Any],
    *,
    target_next_iteration: int,
    staged_root: Path,
) -> set[str]:
    """Validate one complete migration delta and the resulting v17 contract."""
    return validate_v17_migration_receipt_chain(
        module,
        [receipt],
        target_next_iteration=target_next_iteration,
        staged_root=staged_root,
    )


def validate_v17_migration_receipt_chain(
    module: ModuleType,
    receipts: list[dict[str, Any]],
    *,
    target_next_iteration: int,
    staged_root: Path,
) -> set[str]:
    """Validate a contiguous same-boundary v17 migration receipt chain.

    A boundary restart can legitimately append a second receipt containing only
    a source-tree digest correction after the first receipt installed the full
    v17 design. Required v17 fields therefore apply to the union of the linked
    receipts, while the final contract remains the authoritative contract.
    """
    if not receipts:
        raise RuntimeError("v17 migration receipt chain is empty")

    changed_paths: set[str] = set()
    previous_receipt: dict[str, Any] | None = None
    chain_reason = str(receipts[0].get("reason") or "")
    for index, receipt in enumerate(receipts):
        receipt_paths = {
            str(value) for value in (receipt.get("changed_paths") or []) if value
        }
        unexpected_paths = sorted(
            value
            for value in receipt_paths
            if not any(
                _matches_design_prefix(value, prefix)
                for prefix in _ALLOWED_MIGRATION_PREFIXES
            )
        )
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        previous_fingerprint = str(receipt.get("previous_fingerprint") or "")
        current_fingerprint = str(receipt.get("current_fingerprint") or "")
        actual_paths = (
            module._changed_design_paths(previous, current)
            if isinstance(previous, dict) and isinstance(current, dict)
            else set()
        )
        linked = True
        if previous_receipt is not None:
            linked = bool(
                previous_fingerprint
                == str(previous_receipt.get("current_fingerprint") or "")
                and previous == previous_receipt.get("current_contract")
            )
        if (
            int(receipt.get("schema", -1)) != 1
            or int(receipt.get("boundary_next_iteration", -1))
            != int(target_next_iteration)
            or not chain_reason
            or str(receipt.get("reason") or "") != chain_reason
            or not receipt_paths
            or unexpected_paths
            or not isinstance(previous, dict)
            or not isinstance(current, dict)
            or module._design_fingerprint(previous) != previous_fingerprint
            or module._design_fingerprint(current) != current_fingerprint
            or set(actual_paths) != receipt_paths
            or not linked
        ):
            raise RuntimeError(
                "migration receipt chain is invalid: "
                f"index={index} boundary={receipt.get('boundary_next_iteration')} "
                f"paths={sorted(receipt_paths)} unexpected={unexpected_paths} "
                f"actual={sorted(actual_paths)} linked={linked}"
            )
        changed_paths.update(receipt_paths)
        previous_receipt = receipt

    changed_paths = {
        str(value) for value in changed_paths if value
    }
    missing_required = sorted(
        prefix
        for prefix in _REQUIRED_V17_MIGRATION_PREFIXES
        if not any(
            _matches_design_prefix(value, prefix) for value in changed_paths
        )
    )
    if missing_required:
        raise RuntimeError(
            "migration receipt changed unexpected design fields: "
            f"boundary={target_next_iteration} paths={sorted(changed_paths)} "
            "unexpected=[] "
            f"missing_required={missing_required}"
        )

    current = receipts[-1]["current_contract"]
    games = current.get("games")
    collection = current.get("collection")
    expert = current.get("expert_rehearsal")
    gates = current.get("gates")
    opponents = current.get("opponents")
    if not all(
        isinstance(value, dict)
        for value in (games, collection, expert, gates, opponents)
    ):
        raise RuntimeError("v17 current_contract is missing required sections")
    group_counts = collection.get("group_games_per_iteration")
    research = collection.get("research_control_phase")
    active_contract = gates.get("active_contract")
    research_opponents = opponents.get("research_controls")
    expected_root = Path(staged_root).resolve()
    expected_gate_path = (
        expected_root / "ops" / "alakazam_gate_program_v1.json"
    ).resolve()
    expected_registry_path = (
        expected_root / "ops" / "research_control_registry_v1.json"
    ).resolve()
    research_registry = (
        research.get("registry") if isinstance(research, dict) else None
    )
    research_roster = (
        research.get("roster") if isinstance(research, dict) else None
    )
    roster_ids = {
        str(row.get("id") or "")
        for row in research_roster
        if isinstance(row, dict)
    } if isinstance(research_roster, list) else set()
    opponent_ids = {
        str(row.get("id") or "")
        for row in research_opponents
        if isinstance(row, dict)
    } if isinstance(research_opponents, list) else set()
    exact_research_flags = {
        "enabled": True,
        "stage": "measure:research_controls",
        "games_per_iteration": 1000,
        "games_per_control": 250,
        "seat0_games_per_control": 125,
        "seat1_games_per_control": 125,
        "action_selection": "greedy",
        "sampled_behavior_policy": False,
        "training_eligible": False,
        "replay_eligible": False,
        "diagnostic_only": True,
        "additive_to_training_budget": True,
        "formal_eval": False,
        "included_in_gate_pass": False,
        "gate_weight": 0.0,
        "seed_namespace": "eval/research-controls-fixed-manifest-v1",
        "separate_result_artifact": True,
    }
    if (
        games.get("per_iteration") != 8192
        or games.get("heldout") != 2000
        or group_counts
        != {
            "self_play": 1024,
            "strong_public_practice": 4584,
            "diverse_public": 2584,
        }
        or not isinstance(research, dict)
        or any(research.get(key) != value for key, value in exact_research_flags.items())
        or not isinstance(research_registry, dict)
        or Path(str(research_registry.get("path") or "")).resolve()
        != expected_registry_path
        or not isinstance(active_contract, dict)
        or Path(str(active_contract.get("path") or "")).resolve()
        != expected_gate_path
        or expert.get("loss_weights") != _EXPECTED_EXPERT_LOSS_WEIGHTS
        or len(roster_ids) != 4
        or roster_ids != opponent_ids
    ):
        raise RuntimeError(
            "migration receipt current_contract is not the exact v17 design"
        )
    return changed_paths


def load_v17_migration_receipt_chain(
    run_dir: Path,
    *,
    latest_receipt: Path,
    reason: str,
    target_next_iteration: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Load all linked receipts for this v17 migration through ``latest``."""
    paths: list[Path] = []
    receipts: list[dict[str, Any]] = []
    found_latest = False
    for path in sorted((run_dir / "design_migrations").glob("migration_*.json")):
        receipt = load_json(path)
        if (
            str(receipt.get("reason") or "") == str(reason)
            and int(receipt.get("boundary_next_iteration", -1))
            == int(target_next_iteration)
        ):
            paths.append(path)
            receipts.append(receipt)
        if path.resolve() == latest_receipt.resolve():
            found_latest = True
            break
    if not found_latest or not paths or paths[-1].resolve() != latest_receipt.resolve():
        raise RuntimeError("latest v17 migration receipt is not in its receipt chain")
    return paths, receipts


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
                "schema": "poke_bot.archetype_label_research_boundary/v2",
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
        "archetype_label_boundary_trainer", source
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
    steady_dropin: Path,
    migration_reason: str,
    expected_source_sha256: str,
    expected_source_tree_sha256: str,
    expected_gate_contract_sha256: str,
    expected_migration_dropin_sha256: str,
    expected_steady_dropin_sha256: str,
    expected_research_module_sha256: str,
    expected_research_registry_sha256: str,
) -> ModuleType:
    source = staged_root / "scripts" / "train_pure_rl.py"
    actual_digest = sha256(source)
    if actual_digest != expected_source_sha256:
        raise RuntimeError(
            "staged trainer digest mismatch: "
            f"expected={expected_source_sha256} actual={actual_digest}"
        )
    research_module = staged_root / "poke_bot" / "pure_rl" / "research_controls.py"
    research_registry = staged_root / "ops" / "research_control_registry_v1.json"
    gate_contract = staged_root / "ops" / "alakazam_gate_program_v1.json"
    for path, expected_digest, label in (
        (gate_contract, expected_gate_contract_sha256, "active gate contract"),
        (migration_dropin, expected_migration_dropin_sha256, "migration drop-in"),
        (steady_dropin, expected_steady_dropin_sha256, "steady drop-in"),
        (research_module, expected_research_module_sha256, "research module"),
        (research_registry, expected_research_registry_sha256, "research registry"),
    ):
        actual = sha256(path)
        if actual != expected_digest:
            raise RuntimeError(
                f"staged {label} digest mismatch: expected={expected_digest} actual={actual}"
            )
    migration_text = migration_dropin.read_text(encoding="utf-8")
    steady_text = steady_dropin.read_text(encoding="utf-8")
    for token in (
        f"WorkingDirectory={staged_root}",
        "RefuseManualStop=no",
        "--allow-clean-boundary-design-migration",
        f"--boundary-design-migration-reason {migration_reason}",
        "--resume auto",
    ):
        if migration_text.count(token) != 1:
            raise RuntimeError(f"migration drop-in must contain one {token!r}")
    for token in (
        f"WorkingDirectory={staged_root}",
        "RefuseManualStop=yes",
        "--resume auto",
    ):
        if steady_text.count(token) != 1:
            raise RuntimeError(f"steady drop-in must contain one {token!r}")
    if (
        "--allow-clean-boundary-design-migration" in steady_text
        or "--boundary-design-migration-reason" in steady_text
    ):
        raise RuntimeError("steady drop-in retains one-time migration authority")
    module = load_trainer(staged_root)
    actual_source_tree_sha256 = str(
        module._source_snapshot(staged_root).get("source_tree_sha256") or ""
    )
    if actual_source_tree_sha256 != expected_source_tree_sha256:
        raise RuntimeError(
            "staged source-tree digest mismatch: "
            f"expected={expected_source_tree_sha256} "
            f"actual={actual_source_tree_sha256}"
        )
    parameters = inspect.signature(module._consume_results).parameters
    required = {
        "practice_record_contracts",
        "practice_seen_indices",
        "practice_successful_indices",
        "practice_written_indices",
    }
    if not required.issubset(parameters):
        raise RuntimeError("staged coordinator lacks canonical receipt parameters")
    source_text = source.read_text(encoding="utf-8")
    for token in (
        "strong_public_practice_record_receipt",
        "opponent_archetype_id",
        "stale_records_repaired",
        "_assert_training_jobs_exclude_research_controls",
        "_research_control_measurement",
        "measure:research_controls",
    ):
        if token not in source_text:
            raise RuntimeError(f"staged coordinator lacks {token!r}")
    return module


def set_active_dropin(source: Path, destination: Path) -> None:
    parent = destination.parent
    run(["sudo", "-n", "chattr", "-i", str(parent)], timeout=15)
    if destination.exists():
        run(
            ["sudo", "-n", "chattr", "-i", str(destination)],
            timeout=15,
            check=False,
        )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, 0o644)
    os.replace(temporary, destination)
    run(["sudo", "-n", "chattr", "+i", str(destination)], timeout=15)
    run(["sudo", "-n", "chattr", "+i", str(parent)], timeout=15)


def remove_active_dropin(destination: Path) -> None:
    parent = destination.parent
    run(["sudo", "-n", "chattr", "-i", str(parent)], timeout=15)
    if destination.exists():
        run(
            ["sudo", "-n", "chattr", "-i", str(destination)],
            timeout=15,
            check=False,
        )
        destination.unlink()
    run(["sudo", "-n", "chattr", "+i", str(parent)], timeout=15)


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


def recover_partial(module: ModuleType, run_dir: Path) -> str | None:
    state = module._load_loop_state(run_dir)
    if state is None:
        raise RuntimeError("loop state vanished at the boundary")
    recovered = module._recover_interrupted_iteration(run_dir, state)
    return str(recovered) if recovered is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--after-iteration", type=int, required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--migration-dropin", type=Path, required=True)
    parser.add_argument("--steady-dropin", type=Path, required=True)
    parser.add_argument("--active-dropin", type=Path, required=True)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--gate-contract-sha256", required=True)
    parser.add_argument("--migration-dropin-sha256", required=True)
    parser.add_argument("--steady-dropin-sha256", required=True)
    parser.add_argument("--research-module-sha256", required=True)
    parser.add_argument("--research-registry-sha256", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    target_next_iteration = int(args.after_iteration) + 1
    module = validate_candidate(
        staged_root=args.staged_root,
        migration_dropin=args.migration_dropin,
        steady_dropin=args.steady_dropin,
        migration_reason=args.migration_reason,
        expected_source_sha256=args.source_sha256,
        expected_source_tree_sha256=args.source_tree_sha256,
        expected_gate_contract_sha256=args.gate_contract_sha256,
        expected_migration_dropin_sha256=args.migration_dropin_sha256,
        expected_steady_dropin_sha256=args.steady_dropin_sha256,
        expected_research_module_sha256=args.research_module_sha256,
        expected_research_registry_sha256=args.research_registry_sha256,
    )
    observed = int(load_json(args.loop_state).get("next_iteration", -1))
    if observed > target_next_iteration:
        raise RuntimeError(
            f"requested boundary already passed: target={target_next_iteration} "
            f"observed={observed}"
        )
    publish(
        args.status,
        status="validated" if args.validate_only else "waiting_for_boundary",
        after_iteration=int(args.after_iteration),
        target_next_iteration=target_next_iteration,
        observed_next_iteration=observed,
        staged_source_sha256=args.source_sha256,
        staged_source_tree_sha256=args.source_tree_sha256,
    )
    if args.validate_only:
        return 0

    last_health_check = 0.0
    while True:
        state = load_json(args.loop_state)
        completed = int(state.get("last_completed_iteration", -1))
        next_iteration = int(state.get("next_iteration", -1))
        if completed >= int(args.after_iteration):
            if next_iteration != target_next_iteration:
                raise RuntimeError(
                    "boundary advanced unexpectedly: "
                    f"completed={completed} next={next_iteration} "
                    f"target={target_next_iteration}"
                )
            break
        now = time.monotonic()
        if now - last_health_check >= 1.0:
            if service_value(args.service, "ActiveState") not in (
                "active",
                "activating",
            ):
                raise RuntimeError("production trainer stopped before boundary")
            last_health_check = now
        time.sleep(max(0.02, float(args.poll_seconds)))

    migration_started = time.time() - 1.0
    migration_committed = False
    recovered: str | None = None
    try:
        publish(
            args.status,
            status="installing_boundary_source",
            observed_next_iteration=target_next_iteration,
        )
        set_active_dropin(args.migration_dropin, args.active_dropin)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        if service_value(args.service, "RefuseManualStop") != "no":
            raise RuntimeError("migration drop-in did not unlock the boundary stop")
        run(["systemctl", "--user", "stop", args.service], timeout=75)
        recovered = recover_partial(module, args.run_dir)
        publish(
            args.status,
            status="starting_integrity_source",
            recovered_partial_iteration=recovered,
        )
        run(["systemctl", "--user", "reset-failed", args.service], check=False)
        run(["systemctl", "--user", "start", args.service], timeout=75)

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
            if service_value(args.service, "ActiveState") == "failed":
                raise RuntimeError("integrity source failed before migration receipt")
            time.sleep(0.5)
        if receipt_path is None:
            raise RuntimeError("integrity source did not commit migration receipt")
        # From this point onward the append-only migration is authoritative;
        # recovery must keep the new source even if a post-receipt audit fails.
        migration_committed = True
        receipt_paths, receipts = load_v17_migration_receipt_chain(
            args.run_dir,
            latest_receipt=receipt_path,
            reason=args.migration_reason,
            target_next_iteration=target_next_iteration,
        )
        changed_paths = validate_v17_migration_receipt_chain(
            module,
            receipts,
            target_next_iteration=target_next_iteration,
            staged_root=args.staged_root,
        )
        set_active_dropin(args.steady_dropin, args.active_dropin)
        run(["systemctl", "--user", "daemon-reload"], timeout=30)
        if service_value(args.service, "RefuseManualStop") != "yes":
            raise RuntimeError("steady drop-in did not restore stop protection")

        deadline = time.monotonic() + 240.0
        main_pid = 0
        runtime: dict[str, Any] = {}
        while time.monotonic() < deadline:
            active = service_value(args.service, "ActiveState")
            main_pid = int(service_value(args.service, "MainPID") or 0)
            runtime = load_json(args.run_dir / "iteration_runtime.json")
            cwd = ""
            if main_pid > 0:
                try:
                    cwd = str(Path(f"/proc/{main_pid}/cwd").resolve())
                except OSError:
                    cwd = ""
            if (
                active == "active"
                and main_pid > 0
                and cwd == str(args.staged_root.resolve())
                and int(runtime.get("iteration", -1)) == target_next_iteration
            ):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("integrity source did not resume the next iteration")

        stable_pid = main_pid
        for _ in range(15):
            if (
                service_value(args.service, "ActiveState") != "active"
                or int(service_value(args.service, "MainPID") or 0) != stable_pid
            ):
                raise RuntimeError("integrity source restarted during stability check")
            time.sleep(1.0)
        publish(
            args.status,
            status="complete",
            migration_receipt=str(receipt_path),
            migration_receipts=[str(path) for path in receipt_paths],
            changed_paths=sorted(changed_paths),
            main_pid=stable_pid,
            observed_next_iteration=target_next_iteration,
            runtime_phase=runtime.get("phase"),
            recovered_partial_iteration=recovered,
            migration_authority_revoked=True,
            stop_protection_restored=True,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - preserve production on errors
        error = f"{type(exc).__name__}: {exc}"
        publish(args.status, status="recovering_after_error", error=error)
        if migration_committed:
            try:
                set_active_dropin(args.steady_dropin, args.active_dropin)
                run(["systemctl", "--user", "daemon-reload"], timeout=30)
            except BaseException:
                pass
        else:
            try:
                run(["systemctl", "--user", "stop", args.service], check=False)
                remove_active_dropin(args.active_dropin)
                run(["systemctl", "--user", "daemon-reload"], timeout=30)
            except BaseException:
                pass
        run(["systemctl", "--user", "reset-failed", args.service], check=False)
        if service_value(args.service, "ActiveState") != "active":
            run(["systemctl", "--user", "start", args.service], check=False)
        publish(
            args.status,
            status=(
                "integrity_source_recovered"
                if migration_committed
                else "rolled_back_to_previous_source"
            ),
            error=error,
            active_state=service_value(args.service, "ActiveState"),
            main_pid=int(service_value(args.service, "MainPID") or 0),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
