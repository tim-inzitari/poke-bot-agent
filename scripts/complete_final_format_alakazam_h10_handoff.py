#!/usr/bin/env python3
"""Complete the ordinary-Alakazam -> H10-I boundary, fail closed.

This program is intended to run as the ``OnSuccess`` successor of the exact
25-epoch ordinary fallback bootstrap.  It never selects, registers, freezes,
serves, or submits the H10 child.  The handoff receipt is published last and is
therefore the sole authority for a later isolated H10 training stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


READY_SCHEMA = "poke_bot.specialist_expert_bootstrap_ready/v1"
HANDOFF_SCHEMA = "poke_bot.final_format_alakazam_h10_handoff/v1"
MIGRATION_SCHEMA = "poke_bot.final_format_alakazam_migration/v1"
ROLE_SCHEMA = "poke_bot.final_format_learned_role_route_inventory/v1"
EXPECTED_EXPANDED_HEADS = {
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcome",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing or corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def validate_ordinary_parent(
    *,
    ready_path: Path,
    state_path: Path,
    expected_run_name: str,
    expected_family: str,
    expected_core_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate the only parent eligible for the real H10 migration."""

    ready = _read_json(ready_path)
    state = _read_json(state_path)
    checkpoint = Path(str(ready.get("checkpoint") or "")).expanduser().resolve()
    checkpoint_digest = str(ready.get("checkpoint_digest") or "")
    history = list(state.get("history") or [])
    epochs = [int(dict(row).get("epoch") or -1) for row in history]
    expanded = dict(ready.get("expanded_head_training") or {})
    fusion = dict(ready.get("decision_fusion") or {})
    state_ready = dict(state.get("ready") or {})

    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("status") != "ready"
        or ready.get("run_name") != expected_run_name
        or ready.get("family") != expected_family
        or ready.get("acting_seat_archetype") != "alakazam"
        or ready.get("core_checkpoint_digest") != expected_core_sha256
        or ready.get("expert_manifest_sha256") != expected_manifest_sha256
        or int(ready.get("epochs_completed") or 0) != 25
        or int(ready.get("epochs_max") or 0) != 25
        or state.get("status") != "complete"
        or int(state.get("epochs_max") or 0) != 25
        or epochs != list(range(1, 26))
        or str(state.get("best_digest") or "") != checkpoint_digest
        or str(state_ready.get("checkpoint_digest") or "") != checkpoint_digest
        or set(ready.get("expanded_heads_trained") or ())
        != EXPECTED_EXPANDED_HEADS
        or set(expanded.get("trained_heads") or ()) != EXPECTED_EXPANDED_HEADS
        or expanded.get("runtime_enabled_heads") != []
        or fusion.get("runtime_enabled") is not True
        or len(fusion.get("required_heads") or ()) != 17
        or ready.get("flat_policy_remains_authoritative") is not False
    ):
        raise RuntimeError("ordinary Alakazam ready/state contract is incomplete")
    if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_digest:
        raise RuntimeError("ordinary Alakazam checkpoint identity changed")

    best_epoch = next(
        (
            int(dict(row).get("epoch") or 0)
            for row in history
            if str(dict(row).get("checkpoint_digest") or "")
            == checkpoint_digest
        ),
        0,
    )
    if best_epoch < 16 or best_epoch > 25:
        raise RuntimeError("ordinary Alakazam best checkpoint was not fully trained")
    return {
        "ready": ready,
        "state": state,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "ready_sha256": _sha256(ready_path),
        "state_sha256": _sha256(state_path),
        "best_epoch": best_epoch,
        "learned_head_count": len(fusion["required_heads"]),
    }


def validate_migration_outputs(
    *, child: Path, migration_receipt: Path, role_receipt: Path, parent_sha256: str
) -> dict[str, Any]:
    migration = _read_json(migration_receipt)
    roles = _read_json(role_receipt)
    child_sha256 = _sha256(child)
    if (
        migration.get("schema") != MIGRATION_SCHEMA
        or migration.get("status") != "issued_step_zero_passed_training_pending"
        or migration.get("parent_checkpoint_sha256") != parent_sha256
        or migration.get("child_checkpoint_sha256") != child_sha256
        or int(migration.get("learned_head_count") or 0) != 19
        or int(migration.get("learned_route_count") or 0) != 19
        or migration.get(
            "one_distinct_bounded_option_conditioned_route_per_learned_head"
        )
        is not True
        or int(migration.get("guide_runtime_route_count") or 0) != 0
        or int(
            dict(migration.get("parameter_inventory") or {}).get(
                "learned_parameters"
            )
            or 0
        )
        != 10_352_606
        or dict(migration.get("step_zero_parity") or {}).get("passed") is not True
        or migration.get("runtime_authority") != "none"
        or migration.get("selector_eligible") is not False
        or migration.get("kaggle_eligible") is not False
        or roles.get("schema") != ROLE_SCHEMA
        or int(roles.get("learned_head_count") or 0) != 19
        or int(roles.get("learned_route_count") or 0) != 19
        or roles.get("all_learned_heads_influence_actions") is not True
        or roles.get("setup_board_outcome_included") is not True
        or roles.get("validated_slowking_combo_head_included") is not True
        or dict(roles.get("guide") or {}).get("runtime_route_count") != 0
    ):
        raise RuntimeError("H10 migration output contract is incomplete")
    role_sha256 = _sha256(role_receipt)
    if (
        migration.get("learned_head_role_map_sha256") != role_sha256
        or migration.get("learned_route_inventory_sha256") != role_sha256
    ):
        raise RuntimeError("H10 role/route receipt binding changed")
    return {
        "child_sha256": child_sha256,
        "migration_receipt_sha256": _sha256(migration_receipt),
        "role_route_receipt_sha256": role_sha256,
        "learned_parameters": 10_352_606,
        "learned_head_count": 19,
        "learned_route_count": 19,
    }


def _validate_existing_handoff(
    *, handoff_path: Path, child: Path, migration: Path, roles: Path
) -> dict[str, Any]:
    handoff = _read_json(handoff_path)
    if (
        handoff.get("schema") != HANDOFF_SCHEMA
        or handoff.get("status") != "h10_migrated_training_pending"
        or handoff.get("child_checkpoint_sha256") != _sha256(child)
        or handoff.get("migration_receipt_sha256") != _sha256(migration)
        or handoff.get("role_route_receipt_sha256") != _sha256(roles)
        or handoff.get("runtime_authority") != "none"
        or handoff.get("selector_eligible") is not False
        or handoff.get("kaggle_eligible") is not False
    ):
        raise RuntimeError("existing H10 handoff receipt is invalid")
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-run-name", required=True)
    parser.add_argument("--expected-family", required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--direct-migration-failure", type=Path, required=True)
    parser.add_argument("--expected-direct-migration-failure-sha256", required=True)
    parser.add_argument("--migration-script", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--role-route-receipt", type=Path, required=True)
    parser.add_argument("--parent-lock-receipt", type=Path, required=True)
    parser.add_argument("--handoff-receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in (
            "ready",
            "state",
            "direct_migration_failure",
            "migration_script",
            "child",
            "migration_receipt",
            "role_route_receipt",
            "parent_lock_receipt",
            "handoff_receipt",
            "lock",
            "scratch_root",
        )
    }
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if paths["handoff_receipt"].is_file():
            handoff = _validate_existing_handoff(
                handoff_path=paths["handoff_receipt"],
                child=paths["child"],
                migration=paths["migration_receipt"],
                roles=paths["role_route_receipt"],
            )
            print(json.dumps(handoff, sort_keys=True), flush=True)
            return 0
        if any(
            path.exists()
            for path in (
                paths["child"],
                paths["migration_receipt"],
                paths["role_route_receipt"],
                paths["parent_lock_receipt"],
            )
        ):
            raise RuntimeError("partial final H10 handoff artifacts already exist")

        parent = validate_ordinary_parent(
            ready_path=paths["ready"],
            state_path=paths["state"],
            expected_run_name=args.expected_run_name,
            expected_family=args.expected_family,
            expected_core_sha256=args.expected_core_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        direct_failure = _read_json(paths["direct_migration_failure"])
        if (
            _sha256(paths["direct_migration_failure"])
            != args.expected_direct_migration_failure_sha256
            or direct_failure.get("status") != "failed_closed_fallback_required"
            or direct_failure.get("fallback_authorized") is not True
            or dict(direct_failure.get("fallback") or {}).get(
                "accepted_core_checkpoint_sha256"
            )
            != args.expected_core_sha256
        ):
            raise RuntimeError("direct-migration failure/fallback authority changed")

        paths["scratch_root"].mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="ordinary-to-h10-", dir=paths["scratch_root"]
        ) as temporary:
            attempt = Path(temporary)
            staged_child = attempt / "alakazam_r79_h10_i_step_zero.pt"
            staged_migration = attempt / "migration.json"
            staged_roles = attempt / "roles.json"
            subprocess.run(
                [
                    args.python,
                    str(paths["migration_script"]),
                    "--parent",
                    str(parent["checkpoint"]),
                    "--expected-parent-sha256",
                    str(parent["checkpoint_sha256"]),
                    "--child",
                    str(staged_child),
                    "--migration-receipt",
                    str(staged_migration),
                    "--role-route-receipt",
                    str(staged_roles),
                ],
                check=True,
            )
            migrated = validate_migration_outputs(
                child=staged_child,
                migration_receipt=staged_migration,
                role_receipt=staged_roles,
                parent_sha256=str(parent["checkpoint_sha256"]),
            )
            parent_lock = {
                "schema": "poke_bot.final_format_alakazam_parent_lock/v1",
                "template_only": False,
                "status": "locked_ordinary_same_archetype_fallback",
                "runtime_authority": "none",
                "specialist_id": "alakazam",
                "archetype": "alakazam",
                "selected_parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "selected_parent_ready_receipt_sha256": parent["ready_sha256"],
                "selected_parent_state_sha256": parent["state_sha256"],
                "selected_parent_best_epoch": parent["best_epoch"],
                "selected_parent_learned_head_count": parent["learned_head_count"],
                "same_archetype_fallback": {
                    "required": True,
                    "direct_migration_failure_receipt_sha256": _sha256(
                        paths["direct_migration_failure"]
                    ),
                    "accepted_core_checkpoint_sha256": args.expected_core_sha256,
                    "ordinary_alakazam_refresh_checkpoint_sha256": parent[
                        "checkpoint_sha256"
                    ],
                    "ordinary_alakazam_refresh_receipt_sha256": parent[
                        "ready_sha256"
                    ],
                    "partial_old_alakazam_core_overlay_used": False,
                },
                "historical_alakazam_checkpoint_rewritten": False,
                "immutable": True,
                "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            staged_parent_lock = attempt / "parent-lock.json"
            _write_exclusive(staged_parent_lock, parent_lock)
            handoff = {
                "schema": HANDOFF_SCHEMA,
                "status": "h10_migrated_training_pending",
                "goal_revision": 79,
                "specialist_id": "alakazam",
                "capacity_profile": "H10-I/v1",
                "ordinary_parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "ordinary_parent_ready_receipt_sha256": parent["ready_sha256"],
                "ordinary_parent_best_epoch": parent["best_epoch"],
                "parent_lock_receipt_sha256": _sha256(staged_parent_lock),
                "child_checkpoint_sha256": migrated["child_sha256"],
                "migration_receipt_sha256": migrated["migration_receipt_sha256"],
                "role_route_receipt_sha256": migrated[
                    "role_route_receipt_sha256"
                ],
                "learned_parameters": migrated["learned_parameters"],
                "learned_head_count": migrated["learned_head_count"],
                "learned_route_count": migrated["learned_route_count"],
                "training_seat_split": {"first": 0.5, "second": 0.5},
                "package_preference": "first_if_allowed",
                "second_focused_arm_allowed": False,
                "guide_runtime_route_count": 0,
                "runtime_authority": "none",
                "selector_eligible": False,
                "kaggle_eligible": False,
                "next_boundary": "isolated_h10_training_preflight",
                "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            staged_handoff = attempt / "handoff.json"
            _write_exclusive(staged_handoff, handoff)

            _publish_file(staged_child, paths["child"])
            _publish_file(staged_roles, paths["role_route_receipt"])
            _publish_file(staged_migration, paths["migration_receipt"])
            _publish_file(staged_parent_lock, paths["parent_lock_receipt"])
            _publish_file(staged_handoff, paths["handoff_receipt"])
        print(json.dumps(handoff, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
