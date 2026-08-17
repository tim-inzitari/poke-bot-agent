from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_r229_mirror_game.py"
spec = importlib.util.spec_from_file_location("r229_game", SCRIPT)
assert spec and spec.loader
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)


def _package(monkeypatch, tmp_path: Path) -> Path:
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    libraries = {}
    for platform_name, (relative, _digest, _size) in game.CANONICAL_NATIVE_LIBRARIES.items():
        payload = (platform_name + "\n").encode()
        path = package / relative
        path.write_bytes(payload)
        libraries[platform_name] = (
            relative,
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            len(payload),
        )
    monkeypatch.setattr(game, "CANONICAL_NATIVE_LIBRARIES", libraries)
    (package / "r253_fleet_evaluation_manifest.json").write_text(json.dumps({
        "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r253_package/v1",
        "status": "sealed_restarting_serial_mcts_leaf_bounded_evaluation_only",
        "owner_goal_revision": 253,
        "canonical_libcg_revision": 236,
        "superseded_two_lane_topology_revision": 239,
        "owner_handle_scoped_search_id_revision": 244,
        "owner_process_lane_recovery_revision": 249,
        "owner_serial_mcts_revision": 250,
        "owner_internal_leaf_boundary_revision": 252,
        "owner_restarting_serial_rollout_revision": 253,
        "native_simulator_worker_process_count": 1,
        "shared_tree_and_frozen_model_remain_in_parent": True,
        "native_search_calls_in_parent_worker_threads": False,
        "concurrent_libcg_search_calls_allowed": False,
        "complete_serial_retry_count_after_fault": 1,
        "failed_partial_tree_reuse_allowed": False,
        "search_seconds_per_attempt": 8.0,
        "exhausted_recovery_direct_fallback_is_degraded": True,
        "clean_full_game_preflight_max_exhausted_recovery_fallbacks": 0,
        "simulator_lane_count": 1,
        "internal_agent_start_arena_count": 1,
        "minimum_search_begin_call_count_per_searched_decision": 2,
        "search_begin_call_count_equals_completed_rollout_count": True,
        "required_handle_identity_count": 1,
        "required_handle_scoped_search_id_chain_count": 1,
        "required_handle_first_search_id_composite_count": 1,
        "handle_scoped_first_search_id_composite_state_array_field": (
            "handle_scoped_first_search_id_composite_states"
        ),
        "handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order": [
            "lane_id",
            "handle_identity",
            "first_search_id",
        ],
        "search_begin_identity_scope": "arena_handle_plus_handle_local_search_id",
        "raw_search_id_global_uniqueness_required": False,
        "logical_frontier_leaf_count_per_frozen_model_batch": 1,
        "partial_frontier_batches_allowed": False,
        "serial_one_lane_continuation_required": False,
        "independent_exact_root_restart_per_rollout_required": True,
        "one_new_leaf_or_value_boundary_maximum_per_rollout": True,
        "rollout_search_id_chain_count_equals_rollout_count": True,
        "bounded_release_and_search_end_per_rollout_required": True,
        "minimum_completed_rollouts_for_mcts_action_authority": 2,
        "maximum_rollouts_per_decision": 1000,
        "one_shared_logical_mcts_tree_required": True,
        "process_parallel_node_evaluation_included": False,
        "complete_ordered_action_ceiling": 65536,
        "internal_ordered_action_expansion_ceiling": 64,
        "every_explicit_chance_context_is_pre_random_value_boundary": True,
        "explicit_chance_probability_distribution_assumed": False,
        "deterministic_internal_fanout_over_64_is_value_only_boundary": True,
        "internal_boundary_representative_action_has_no_tree_authority": True,
        "internal_boundary_has_action_or_child_authority": False,
        "internal_value_boundary_telemetry_required": True,
        "bo_lifecycle_revision": 233,
        "r234_kaggle_broker_or_queue_lifecycle_included": False,
        "kaggle_search_policy_changes_included": False,
        "r249_bo_process_lane_boundary_included": True,
        "r250_serial_process_lane_topology_included": True,
        "r252_internal_leaf_boundary_included": True,
        "r253_restarting_serial_rollout_included": True,
        "continuous_single_trajectory_action_authority_allowed": False,
        "canonical_native_libraries": {
            name: {"path": relative, "sha256": digest, "size_bytes": size}
            for name, (relative, digest, size) in libraries.items()
        },
    }))
    return package


def test_game_preflight_binds_complete_set_and_host_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    monkeypatch.setattr(game.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(game.platform, "machine", lambda: "arm64")
    complete, selected = game._verify_canonical_native_set(package)
    assert set(complete) == set(game.CANONICAL_NATIVE_LIBRARIES)
    assert selected["platform_identity"] == "macos_arm64"
    assert selected["path"] == "cg/libcg.dylib"


def test_game_preflight_rejects_mixed_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    (package / "cg/libcg.so").write_bytes(b"historical")
    with pytest.raises(game.R229GameError, match="drifted"):
        game._verify_canonical_native_set(package)


def test_game_preflight_rejects_extra_native_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    (package / "cg/libcg-old.so").write_bytes(b"historical")
    with pytest.raises(game.R229GameError, match="mixed or incomplete"):
        game._verify_canonical_native_set(package)


def test_game_preflight_rejects_nonserial_manifest(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    manifest = package / "r253_fleet_evaluation_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["simulator_lane_count"] = 8
    manifest.write_text(json.dumps(payload))
    with pytest.raises(game.R229GameError, match="manifest identity drifted"):
        game._verify_canonical_native_set(package)


def _serial_receipt():
    return {
        "mode": "shared_tree_mcts",
        "internal_value_boundary_count": 0,
        "internal_value_boundary_reasons": {},
        "max_internal_ordered_action_count": 64,
        "internal_ordered_action_expansion_ceiling": 64,
        "explicit_chance_probability_distribution_assumed": False,
        "explicit_chance_always_stops_before_random_resolution": True,
        "internal_boundary_has_action_or_child_authority": False,
        "lane_process_recovery": {
            "serial_lane_count": 1,
            "attempt_count": 1,
            "recovered_search": False,
            "exhausted_direct_fallback": False,
            "attempts": [{"attempt": 1, "status": "complete"}],
        },
        "requested_simulator_lane_count": 1,
        "active_simulator_lane_count": 1,
        "arena_count": 1,
        "unique_handle_count": 1,
        "per_lane_handle_identities": [1001],
        "search_begin_calls": 2,
        "search_step_calls": 2,
        "completed_backups": 2,
        "root_visits": 2,
        "search_release_calls": 4,
        "search_end_calls": 2,
        "max_simulator_calls_in_flight": 1,
        "per_lane_depth": [1],
        "per_lane_search_id_chains": [[0, 1]],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
        ],
        "rollout_count": 2,
        "rollout_search_id_chains": [[0, 1], [0, 1]],
        "rollout_search_begin_states": [
            {"rollout_index": 0, "handle_identity": 1001, "first_search_id": 0},
            {"rollout_index": 1, "handle_identity": 1001, "first_search_id": 0},
        ],
        "rollout_root_actions": [[0], [1]],
        "root_action_visit_counts": [1, 1],
        "distinct_root_actions_visited": 2,
        "legal_action_count": 2,
        "max_rollout_depth": 1,
        "rollout_stop_reason": "rollout_ceiling",
        "rollout_ceiling": 1000,
        "microbatch_sizes": [1, 1],
        "outstanding_virtual_loss": 0,
    }


def test_decision_receipt_requires_exactly_one_serial_lane():
    game._validate_serial_receipts([_serial_receipt()])
    bad = _serial_receipt()
    bad["per_lane_handle_identities"] = [1001, 1002]
    with pytest.raises(game.R229GameError, match="restarting-serial"):
        game._validate_serial_receipts([bad])
    bad = _serial_receipt()
    bad["handle_scoped_first_search_id_composite_states"][0] = {
        "handle_identity": 1001,
        "lane_id": 0,
        "first_search_id": 0,
    }
    with pytest.raises(game.R229GameError, match="restarting-serial"):
        game._validate_serial_receipts([bad])
    bad = _serial_receipt()
    bad["microbatch_sizes"] = [2]
    with pytest.raises(game.R229GameError, match="restarting-serial"):
        game._validate_serial_receipts([bad])


def test_exhausted_recovery_fallback_cannot_claim_mcts_change_credit():
    row = {
        "mode": "bounded_lane_recovery_exhausted_direct_fallback",
        "mcts_action_authority": False,
        "action_changed": False,
        "internal_value_boundary_count": 0,
        "internal_value_boundary_reasons": {},
        "max_internal_ordered_action_count": 0,
        "internal_ordered_action_expansion_ceiling": 64,
        "explicit_chance_probability_distribution_assumed": False,
        "explicit_chance_always_stops_before_random_resolution": True,
        "internal_boundary_has_action_or_child_authority": False,
        "lane_process_recovery": {
            "serial_lane_count": 1,
            "attempt_count": 2,
            "recovered_search": False,
            "exhausted_direct_fallback": True,
            "attempts": [
                {"attempt": 1, "status": "failed"},
                {"attempt": 2, "status": "failed"},
            ],
        },
    }
    game._validate_serial_receipts([row])
    row["action_changed"] = True
    with pytest.raises(game.R229GameError, match="gained MCTS authority"):
        game._validate_serial_receipts([row])
