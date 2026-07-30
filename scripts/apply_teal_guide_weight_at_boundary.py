#!/usr/bin/env python3
"""Apply an owner-authorized Teal guide-weight change at an immutable boundary.

The controller is intended to run as a managed user service. It leaves the
active trainer untouched until the requested iteration is checksum-committed,
briefly stops that managed service during its boundary pause, installs one
checksum-pinned registry migration, and resumes at the next iteration. The
historical revision-42 migration may also install its checksum-pinned wording
update; later registry-only owner overrides preserve the active source tree.
"""

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


SCHEMA = "poke_bot.current_deck_guide_weight_boundary/v1"
MIGRATION_REASON = "receipt_backed_current_deck_guide_weight_curve_v1"
SPECIALIST_ID = "teal-mask-ogerpon-ex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_staged_registry(
    current_path: Path,
    staged_path: Path,
    *,
    old_weight: float,
    new_weight: float,
) -> None:
    """Materialize the one-field registry candidate when none was prebuilt."""
    if staged_path.exists():
        return
    staged = read_json(current_path)
    row = dict((staged.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if not row:
        raise RuntimeError("runtime registry lacks the active Teal specialist")
    if float(row.get("guide_loss_weight", -1.0)) != old_weight:
        raise RuntimeError("active registry guide weight is not the expected source")
    row["guide_loss_weight"] = new_weight
    staged["specialists"][SPECIALIST_ID] = row
    atomic_bytes(
        staged_path,
        (json.dumps(staged, indent=2, sort_keys=True) + "\n").encode(),
        0o600,
    )


def publish(path: Path, **values: Any) -> None:
    payload = {
        "schema": SCHEMA,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        0o600,
    )


def run(
    argv: list[str],
    *,
    timeout: float = 90.0,
    check: bool = True,
    emit_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if emit_output and result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {' '.join(argv)}"
        )
    return result


def systemctl(
    *args: str,
    timeout: float = 90.0,
    emit_output: bool = True,
) -> str:
    return run(
        ["systemctl", "--user", *args],
        timeout=timeout,
        emit_output=emit_output,
    ).stdout.strip()


def service_value(unit: str, key: str) -> str:
    return systemctl(
        "show",
        unit,
        "-p",
        key,
        "--value",
        timeout=15.0,
        emit_output=False,
    )


def selector_value(text: str, key: str) -> str | None:
    values = [
        line.split("=", 1)[1]
        for line in text.splitlines()
        if line.startswith(key + "=")
    ]
    if len(values) > 1:
        raise RuntimeError(f"selector contains duplicate {key}")
    return values[0] if values else None


def set_selector_value(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(key + "=")]
    if len(matches) != 1:
        raise RuntimeError(f"selector must contain exactly one {key}")
    lines[matches[0]] = f"{key}={value}"
    return "\n".join(lines) + "\n"


def validate_registry_pair(
    current_path: Path,
    staged_path: Path,
    *,
    old_weight: float,
    new_weight: float,
) -> None:
    current = read_json(current_path)
    staged = read_json(staged_path)
    current_row = dict((current.get("specialists") or {}).get(SPECIALIST_ID) or {})
    staged_row = dict((staged.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if not current_row or not staged_row:
        raise RuntimeError("runtime registry lacks the active Teal specialist")
    if float(current_row.get("guide_loss_weight", -1.0)) != old_weight:
        raise RuntimeError("active registry guide weight is not the expected source")
    if float(staged_row.get("guide_loss_weight", -1.0)) != new_weight:
        raise RuntimeError("staged registry guide weight is not the requested target")
    current_row["guide_loss_weight"] = new_weight
    if current_row != staged_row:
        raise RuntimeError("staged registry changes fields beyond guide_loss_weight")
    current["specialists"][SPECIALIST_ID] = current_row
    if current != staged:
        raise RuntimeError("staged registry changes non-Teal runtime state")


def validate_current_deck_guide_log_naming_migration(
    current_path: Path,
    staged_path: Path,
) -> None:
    """Allow only the generic current-deck guide user-facing wording fix."""
    current = current_path.read_text(encoding="utf-8")
    expected = current.replace(
        '"Alakazam guide target is not aligned to policy options"',
        '"current-deck guide target is not aligned to policy options"',
        1,
    ).replace(
        "    Alakazam guide scores are collapsed during featurization to a unique-best\n"
        "    index and bounded confidence, then distilled with masked CE. The default\n"
        "    weight is zero, so core and older feature shards preserve exact behavior.",
        "    Current-deck guide scores are collapsed during featurization to a\n"
        "    unique-best index and bounded confidence, then distilled with masked CE.\n"
        "    The default weight is zero, so core and older feature shards preserve\n"
        "    exact behavior.",
        1,
    ).replace(
        '"Alakazam guide loss weight cannot be negative"',
        '"current-deck guide loss weight cannot be negative"',
        1,
    ).replace(
        '"resident Alakazam guide target is outside option row: "',
        '"resident current-deck guide target is outside option row: "',
        2,
    ).replace(
        '"nonzero Alakazam guide weight requires resident guide targets"',
        '"nonzero current-deck guide weight requires resident guide targets"',
        1,
    ).replace(
        '"nonzero Alakazam guide loss has no usable guide rows; "',
        '"nonzero current-deck guide loss has no usable guide rows; "',
        1,
    ).replace(
        '"verify POKEBOT_ALAKAZAM_GUIDE_TARGETS=1 and scorer coverage"',
        '"verify the selected current-deck guide target switch and "\n'
        '                "scorer coverage"',
        1,
    ).replace(
        'f"[rl-train] Alakazam guide rows={guide_rows} "',
        'f"[rl-train] current-deck guide rows={guide_rows} "',
        1,
    )
    if expected == current:
        raise RuntimeError("active training module lacks the exact legacy wording")
    if staged_path.read_text(encoding="utf-8") != expected:
        raise RuntimeError(
            "staged training module changes code beyond current-deck guide wording"
        )


def exact_boundary(
    run_dir: Path,
    completed_iteration: int,
) -> tuple[Path, dict[str, Any], Path, str] | None:
    commit_path = run_dir / "commits" / f"iter_{completed_iteration:05d}.json"
    if not commit_path.is_file():
        return None
    commit = read_json(commit_path)
    loop_state = read_json(run_dir / "loop_state.json")
    next_iteration = completed_iteration + 1
    if (
        int(commit.get("last_completed_iteration", -1)) != completed_iteration
        or int(commit.get("next_iteration", -1)) != next_iteration
        or loop_state != commit
    ):
        return None
    next_commit = run_dir / "commits" / f"iter_{next_iteration:05d}.json"
    if next_commit.exists():
        raise RuntimeError("requested migration boundary has already passed")
    learner = dict(commit.get("learner") or {})
    checkpoint = Path(str(learner.get("path") or "")).expanduser().resolve()
    digest = str(learner.get("digest") or "").lower()
    if not checkpoint.is_file() or sha256(checkpoint) != digest:
        raise RuntimeError("boundary learner checkpoint identity does not verify")
    return commit_path.resolve(), commit, checkpoint, digest


def install_stop_window(unit: str, dropin: Path) -> None:
    atomic_bytes(dropin, b"[Unit]\nRefuseManualStop=no\n", 0o644)
    systemctl("daemon-reload")
    if service_value(unit, "RefuseManualStop").strip().lower() != "no":
        raise RuntimeError("managed boundary stop window did not activate")


def close_stop_window(unit: str, dropin: Path) -> None:
    dropin.unlink(missing_ok=True)
    systemctl("daemon-reload")
    if service_value(unit, "RefuseManualStop").strip().lower() != "yes":
        raise RuntimeError("manual-stop protection was not restored")


def wait_inactive(unit: str, timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if (
            service_value(unit, "ActiveState") in {"inactive", "failed"}
            and int(service_value(unit, "MainPID") or 0) == 0
        ):
            return
        time.sleep(0.10)
    raise RuntimeError("managed trainer did not stop at the boundary")


def wait_migration(
    run_dir: Path,
    *,
    boundary_next_iteration: int,
    timeout_seconds: float = 120.0,
) -> Path:
    required_changed_paths = {
        "learner.alakazam_guide_loss_weight",
        "learner.current_deck_guide_loss_weight",
        "expert_rehearsal.loss_weights.alakazam_guide",
    }
    allowed_changed_paths = required_changed_paths | {
        # Replacing the checksum-authorized trainer at this same boundary
        # legitimately changes the source-tree identity recorded in the
        # immutable design contract.
        "source.source_tree_sha256",
    }
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for path in sorted(
            (run_dir / "design_migrations").glob("migration_*.json"),
            reverse=True,
        ):
            receipt = read_json(path)
            if (
                receipt.get("reason") == MIGRATION_REASON
                and int(receipt.get("boundary_next_iteration", -1))
                == boundary_next_iteration
                and required_changed_paths
                <= set(receipt.get("changed_paths") or ())
                <= allowed_changed_paths
            ):
                return path.resolve()
        time.sleep(0.10)
    raise RuntimeError("trainer did not publish the guide-weight migration receipt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--staged-registry", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--staged-trainer", type=Path, required=True)
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--staged-train-module", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--completed-iteration", type=int, default=5)
    parser.add_argument("--old-weight", type=float, default=0.05)
    parser.add_argument("--new-weight", type=float, default=0.25)
    parser.add_argument("--registry-only", action="store_true")
    parser.add_argument("--owner-decision-revision", type=int, default=42)
    parser.add_argument("--expected-staged-trainer-sha256", required=True)
    parser.add_argument("--expected-active-train-module-sha256", required=True)
    parser.add_argument("--expected-staged-train-module-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    selector = args.selector.expanduser().resolve()
    registry = args.registry.expanduser().resolve()
    staged_registry = args.staged_registry.expanduser().resolve()
    trainer = args.trainer.expanduser().resolve()
    staged_trainer = args.staged_trainer.expanduser().resolve()
    train_module = args.train_module.expanduser().resolve()
    staged_train_module = args.staged_train_module.expanduser().resolve()
    status = args.status.expanduser().resolve()
    dropin = (
        Path.home()
        / ".config/systemd/user"
        / f"{args.unit}.d/90-teal-guide-weight-boundary.conf"
    )

    if args.registry_only:
        if (
            args.completed_iteration != 12
            or args.old_weight != 0.25
            or args.new_weight != 0.05
            or args.owner_decision_revision != 50
        ):
            raise RuntimeError(
                "registry-only receipt contract is pinned to owner revision 50 "
                "and the Teal iter12 0.25 -> 0.05 boundary"
            )
        if sha256(trainer) != args.expected_staged_trainer_sha256:
            raise RuntimeError("active trainer checksum does not match authorization")
        if sha256(train_module) != args.expected_active_train_module_sha256:
            raise RuntimeError("active training module checksum drifted before staging")
        if sha256(train_module) != args.expected_staged_train_module_sha256:
            raise RuntimeError(
                "registry-only migration must preserve the active training module"
            )
        trainer_text = trainer.read_text(encoding="utf-8")
    else:
        if (
            args.completed_iteration != 5
            or args.old_weight != 0.05
            or args.new_weight != 0.25
            or args.owner_decision_revision != 42
        ):
            raise RuntimeError(
                "historical receipt contract is pinned to owner revision 42 "
                "and the Teal iter5 0.05 -> 0.25 boundary"
            )
        if sha256(staged_trainer) != args.expected_staged_trainer_sha256:
            raise RuntimeError("staged trainer checksum does not match authorization")
        if sha256(train_module) != args.expected_active_train_module_sha256:
            raise RuntimeError("active training module checksum drifted before staging")
        if sha256(staged_train_module) != args.expected_staged_train_module_sha256:
            raise RuntimeError(
                "staged training module checksum does not match authorization"
            )
        validate_current_deck_guide_log_naming_migration(
            train_module,
            staged_train_module,
        )
        trainer_text = staged_trainer.read_text(encoding="utf-8")
    for token in (
        "_safe_current_deck_guide_weight_migration",
        MIGRATION_REASON,
        "_CURRENT_DECK_GUIDE_WEIGHT_MIGRATION_PATHS",
    ):
        if token not in trainer_text:
            raise RuntimeError(f"staged trainer lacks {token}")
    create_staged_registry(
        registry,
        staged_registry,
        old_weight=args.old_weight,
        new_weight=args.new_weight,
    )
    validate_registry_pair(
        registry,
        staged_registry,
        old_weight=args.old_weight,
        new_weight=args.new_weight,
    )
    original_selector = selector.read_bytes()
    original_registry = registry.read_bytes()
    original_trainer = trainer.read_bytes()
    original_train_module = train_module.read_bytes()
    selector_text = original_selector.decode()
    if selector_value(selector_text, "POKEBOT_ACTIVE_SPECIALIST") != SPECIALIST_ID:
        raise RuntimeError("live selector is not Teal Mask Ogerpon ex")
    if service_value(args.unit, "ActiveState") != "active":
        raise RuntimeError("managed trainer is not active before staging")
    old_pid = int(service_value(args.unit, "MainPID") or 0)
    publish(
        status,
        status=f"waiting_for_immutable_iteration_{args.completed_iteration}_commit",
        unit=args.unit,
        old_pid=old_pid,
        old_weight=args.old_weight,
        new_weight=args.new_weight,
        owner_decision_revision=args.owner_decision_revision,
        registry_only=args.registry_only,
        active_specialist=SPECIALIST_ID,
        staged_registry_sha256=sha256(staged_registry),
        staged_trainer_sha256=sha256(staged_trainer),
        staged_train_module_sha256=sha256(staged_train_module),
    )
    deadline = time.monotonic() + max(1.0, args.timeout_seconds)
    proof = None
    while time.monotonic() < deadline:
        proof = exact_boundary(run_dir, args.completed_iteration)
        if proof is not None:
            break
        if service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("trainer stopped before the requested boundary")
        time.sleep(max(0.02, args.poll_seconds))
    if proof is None:
        raise RuntimeError(
            "timed out waiting for immutable iteration "
            f"{args.completed_iteration} commit"
        )
    commit_path, commit, checkpoint, checkpoint_digest = proof
    publish(
        status,
        status="boundary_verified_stopping_managed_trainer",
        commit=str(commit_path),
        commit_sha256=sha256(commit_path),
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_digest,
        old_pid=old_pid,
        old_weight=args.old_weight,
        new_weight=args.new_weight,
    )
    try:
        # Keep manual-stop protection enabled throughout the potentially long
        # wait. Open it only after the exact immutable boundary verifies.
        install_stop_window(args.unit, dropin)
        systemctl("stop", args.unit)
        wait_inactive(args.unit)
    except Exception:
        if dropin.exists():
            close_stop_window(args.unit, dropin)
        publish(
            status,
            status="failed_before_managed_boundary_stop",
            unit=args.unit,
            old_weight=args.old_weight,
            requested_weight=args.new_weight,
        )
        raise
    migration_path: Path | None = None
    try:
        if not args.registry_only:
            atomic_bytes(trainer, staged_trainer.read_bytes(), 0o755)
            atomic_bytes(train_module, staged_train_module.read_bytes(), 0o644)
        atomic_bytes(registry, staged_registry.read_bytes(), 0o600)
        migrated_selector = set_selector_value(
            selector_text,
            "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION",
            "1",
        )
        migrated_selector = set_selector_value(
            migrated_selector,
            "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE",
            MIGRATION_REASON,
        )
        atomic_bytes(selector, migrated_selector.encode(), 0o644)
        close_stop_window(args.unit, dropin)
        systemctl("start", args.unit)
        migration_path = wait_migration(
            run_dir,
            boundary_next_iteration=args.completed_iteration + 1,
        )
        if service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("trainer is not active after guide-weight migration")
        new_pid = int(service_value(args.unit, "MainPID") or 0)
        if new_pid <= 0 or new_pid == old_pid:
            raise RuntimeError("trainer PID did not advance after boundary migration")
        # The one-start permission is no longer needed after its append-only
        # receipt exists. Restore the original selector bytes without touching
        # the already-running process.
        atomic_bytes(selector, original_selector, 0o644)
        publish(
            status,
            status="activated",
            unit=args.unit,
            old_pid=old_pid,
            new_pid=new_pid,
            active_specialist=SPECIALIST_ID,
            selector_identity_unchanged=True,
            completed_iteration=args.completed_iteration,
            boundary_next_iteration=args.completed_iteration + 1,
            old_weight=args.old_weight,
            new_weight=args.new_weight,
            owner_decision_revision=args.owner_decision_revision,
            registry_only=args.registry_only,
            commit=str(commit_path),
            commit_sha256=sha256(commit_path),
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_digest,
            registry_sha256=sha256(registry),
            trainer_sha256=sha256(trainer),
            train_module_sha256=sha256(train_module),
            design_migration_receipt=str(migration_path),
            design_migration_receipt_sha256=sha256(migration_path),
            stop_protection_restored=True,
            one_start_migration_permission_restored=True,
        )
    except Exception:
        # Before the append-only migration receipt exists, restore the last
        # known design. After it exists, the new design is authoritative and
        # rolling its files back would break the receipt chain.
        if migration_path is None:
            if not args.registry_only:
                atomic_bytes(trainer, original_trainer, 0o755)
                atomic_bytes(train_module, original_train_module, 0o644)
            atomic_bytes(registry, original_registry, 0o600)
        atomic_bytes(selector, original_selector, 0o644)
        close_stop_window(args.unit, dropin)
        if service_value(args.unit, "ActiveState") != "active":
            systemctl("start", args.unit)
        publish(
            status,
            status=(
                "failed_rolled_back"
                if migration_path is None
                else "failed_after_migration_receipt_preserved"
            ),
            unit=args.unit,
            old_weight=args.old_weight,
            requested_weight=args.new_weight,
            original_registry_sha256="sha256:"
            + hashlib.sha256(original_registry).hexdigest(),
            original_trainer_sha256="sha256:"
            + hashlib.sha256(original_trainer).hexdigest(),
            original_train_module_sha256="sha256:"
            + hashlib.sha256(original_train_module).hexdigest(),
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
