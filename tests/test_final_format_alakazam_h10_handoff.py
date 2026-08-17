from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.complete_final_format_alakazam_h10_handoff import (
    EXPECTED_EXPANDED_HEADS,
    validate_migration_outputs,
    validate_ordinary_parent,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ordinary_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    checkpoint = tmp_path / "ordinary.pt"
    checkpoint.write_bytes(b"immutable ordinary parent")
    checkpoint_sha = _sha(checkpoint)
    core_sha = "sha256:" + "1" * 64
    manifest_sha = "sha256:" + "2" * 64
    fusion = {
        "runtime_enabled": True,
        "required_heads": [f"head-{index}" for index in range(17)],
    }
    ready = {
        "schema": "poke_bot.specialist_expert_bootstrap_ready/v1",
        "status": "ready",
        "run_name": "ordinary-run",
        "family": "ordinary-family",
        "acting_seat_archetype": "alakazam",
        "checkpoint": str(checkpoint),
        "checkpoint_digest": checkpoint_sha,
        "core_checkpoint_digest": core_sha,
        "expert_manifest_sha256": manifest_sha,
        "epochs_completed": 25,
        "epochs_max": 25,
        "expanded_heads_trained": sorted(EXPECTED_EXPANDED_HEADS),
        "expanded_head_training": {
            "trained_heads": sorted(EXPECTED_EXPANDED_HEADS),
            "runtime_enabled_heads": [],
        },
        "decision_fusion": fusion,
        "flat_policy_remains_authoritative": False,
    }
    ready_path = tmp_path / "ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    state = {
        "status": "complete",
        "epochs_max": 25,
        "best_digest": checkpoint_sha,
        "history": [
            {
                "epoch": epoch,
                "checkpoint_digest": checkpoint_sha if epoch == 21 else f"epoch-{epoch}",
            }
            for epoch in range(1, 26)
        ],
        "ready": ready,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return ready_path, state_path, core_sha, manifest_sha


def test_ordinary_parent_requires_exact_complete_25_epoch_receipt(tmp_path: Path) -> None:
    ready, state, core_sha, manifest_sha = _ordinary_fixture(tmp_path)
    result = validate_ordinary_parent(
        ready_path=ready,
        state_path=state,
        expected_run_name="ordinary-run",
        expected_family="ordinary-family",
        expected_core_sha256=core_sha,
        expected_manifest_sha256=manifest_sha,
    )
    assert result["best_epoch"] == 21
    assert result["learned_head_count"] == 17


def test_ordinary_parent_rejects_intermediate_epoch(tmp_path: Path) -> None:
    ready, state, core_sha, manifest_sha = _ordinary_fixture(tmp_path)
    payload = json.loads(state.read_text())
    payload["history"] = payload["history"][:-1]
    state.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_ordinary_parent(
            ready_path=ready,
            state_path=state,
            expected_run_name="ordinary-run",
            expected_family="ordinary-family",
            expected_core_sha256=core_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_h10_outputs_require_exact_capacity_and_nineteen_routes(tmp_path: Path) -> None:
    child = tmp_path / "h10.pt"
    child.write_bytes(b"h10 child")
    parent_sha = "sha256:" + "3" * 64
    roles = {
        "schema": "poke_bot.final_format_learned_role_route_inventory/v1",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "all_learned_heads_influence_actions": True,
        "setup_board_outcome_included": True,
        "validated_slowking_combo_head_included": True,
        "guide": {"runtime_route_count": 0},
    }
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(json.dumps(roles), encoding="utf-8")
    migration = {
        "schema": "poke_bot.final_format_alakazam_migration/v1",
        "status": "issued_step_zero_passed_training_pending",
        "parent_checkpoint_sha256": parent_sha,
        "child_checkpoint_sha256": _sha(child),
        "learned_head_count": 19,
        "learned_route_count": 19,
        "one_distinct_bounded_option_conditioned_route_per_learned_head": True,
        "guide_runtime_route_count": 0,
        "parameter_inventory": {"learned_parameters": 10_352_606},
        "step_zero_parity": {"passed": True},
        "learned_head_role_map_sha256": _sha(roles_path),
        "learned_route_inventory_sha256": _sha(roles_path),
        "runtime_authority": "none",
        "selector_eligible": False,
        "kaggle_eligible": False,
    }
    migration_path = tmp_path / "migration.json"
    migration_path.write_text(json.dumps(migration), encoding="utf-8")
    result = validate_migration_outputs(
        child=child,
        migration_receipt=migration_path,
        role_receipt=roles_path,
        parent_sha256=parent_sha,
    )
    assert result["learned_parameters"] == 10_352_606

    migration["learned_route_count"] = 18
    migration_path.write_text(json.dumps(migration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_migration_outputs(
            child=child,
            migration_receipt=migration_path,
            role_receipt=roles_path,
            parent_sha256=parent_sha,
        )
