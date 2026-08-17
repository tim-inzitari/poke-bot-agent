"""Static r222 runner and root-authority checks.

These tests deliberately do not require Torch, a GPU, or the archived r195
package.  Stock-libcg and lane-isolation integration are sealed-source
preflight responsibilities.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.r222_multi_search_turn_belief_mcts import (
    r222_plan_result_from_mcts_result,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts/run_alakazam_local_multi_search_turn_belief_mcts_bo1000_r222.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("r222_runner_static", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostics(**overrides):
    values = {
        "selected_action": [0],
        "selected_action_legal": True,
        "selected_action_fully_backed_up": True,
        "selected_action_visit_count": 1,
        "selected_action_completed_backups": 1,
        "completed_backups": 1,
        "chance_samples": 0,
        "sampled_unforceable_chance_nodes": 0,
        "sampled_unforceable_chance_reasons": {},
        "unforceable_chance_boundary_nodes": 0,
        "unforceable_chance_boundary_leaf_evaluations": 0,
        "unforceable_chance_boundary_reasons": {},
        "private_unforceable_chance_samples_prohibited": True,
        "seed_hunting_or_pre_randomization_prohibited": True,
    }
    values.update(overrides)
    return values


def _result(**diagnostic_overrides):
    return SimpleNamespace(
        select=[0],
        sims_run=1,
        target=SimpleNamespace(diagnostics=_diagnostics(**diagnostic_overrides)),
    )


def test_r222_plan_authority_requires_explicit_backed_root_and_safe_random_receipt():
    plan = r222_plan_result_from_mcts_result(_result(), selected_action=[0])

    assert plan.sims_run == 1
    assert plan.diagnostics["r222_selected_root_action_authority"] is True
    assert plan.diagnostics["r222_selected_root_action_authority_reason"] == (
        "selected_root_action_explicitly_backed_up"
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"selected_action_visit_count": 0},
        {"selected_action_completed_backups": 0},
        {"chance_samples": 1},
        {"sampled_unforceable_chance_nodes": 1},
        {"private_unforceable_chance_samples_prohibited": False},
        {"seed_hunting_or_pre_randomization_prohibited": False},
    ),
)
def test_r222_plan_authority_fails_closed_for_unbacked_or_unsafe_results(overrides):
    plan = r222_plan_result_from_mcts_result(_result(**overrides), selected_action=[0])

    assert plan.sims_run == 0
    assert plan.diagnostics["r222_selected_root_action_authority"] is False


def test_r222_runner_is_stock_transport_and_one_whole_bo1000_only():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= {
        "argparse",
        "hashlib",
        "json",
        "os",
        "re",
        "subprocess",
        "sys",
        "time",
        "uuid",
    }
    assert "r215_seeded_mirror_runtime" not in source
    assert "r219_seeded_mirror_runtime" not in source
    assert "BattleStartSeeded" not in source
    assert "B77" not in source
    assert '"independent_unmatched"' in source
    assert '"live_prefix_diagnostic"' in source
    assert "separate_canary" not in source
    assert "one whole --pair-start 0 --pair-count 500" in source


def test_r222_runner_parser_refuses_shard_or_old_prefix_modes(tmp_path: Path):
    runner = _runner_module()
    base = [
        "--source-root",
        str(tmp_path / "source"),
        "--output-root",
        str(tmp_path / "output"),
    ]

    parsed = runner._parse_args(base)
    assert parsed.mode == "bo1000"
    assert parsed.pair_start == 0
    assert parsed.pair_count == 500
    with pytest.raises(SystemExit):
        runner._parse_args([*base, "--pair-count", "5"])
    with pytest.raises(SystemExit):
        runner._parse_args([*base, "--mode", "canary"])


def _valid_shared_tree_lane_receipt() -> dict[str, object]:
    return {
        "requested_lane_count": 8,
        "active_lane_count": 8,
        "isolated_stock_search_state_count": 8,
        "all_lanes_isolated": True,
        "all_lanes_multistep_capable": True,
        "shared_logical_tree": True,
        "independent_root_parallel_forest_or_root_stat_merge": False,
        "shared_frozen_leaf_broker": True,
        "virtual_loss_or_path_and_leaf_reservations_enabled": True,
        "same_world_inflight_model_evaluation_coalescing_enabled": True,
        "native_semantic_state_equivalence_required": True,
        "public_lookalike_cross_world_merges_prevented": True,
        "partial_lane_statistics_used": False,
        "stock_search_state_isolation_preflight_result": "passed",
        "shared_logical_tree_identity_or_equivalent_integrity_receipt": "tree:root-1",
        "decision_fingerprint": "decision:root-1",
        "frozen_model_identity_or_checksum": "sha256:" + "a" * 64,
        "leaf_microbatch_count": 1,
        "leaf_microbatch_size_distribution": [8],
        "lane_trajectory_count": 8,
        "lane_backup_count": 8,
        "virtual_loss_or_path_leaf_reservation_count": 8,
        "in_flight_frozen_eval_coalescing_count": 0,
        "safe_frozen_eval_cache_hit_count": 0,
        "unavoidable_repeat_expansion_count": 0,
        "outstanding_path_or_leaf_reservation_count_at_action_return": 0,
        "outstanding_virtual_loss_count_at_action_return": 0,
        "public_lookalike_cross_world_merge_count": 0,
        "genuine_multistep_mcts": True,
        "max_simulator_search_depth": 2,
        "multi_step_simulations": 1,
        "per_lane_lifecycle": [
            {
                "lane_id": lane_id,
                "search_begin_calls": 1,
                "search_release_calls": 2,
                "search_end_calls": 1,
            }
            for lane_id in range(8)
        ],
    }


def test_r222_runner_requires_a_true_complete_shared_tree_lane_receipt():
    runner = _runner_module()
    lane = _valid_shared_tree_lane_receipt()

    assert runner._shared_tree_lane_receipt_valid(lane, preflight=False) == (True, None)

    lane["outstanding_virtual_loss_count_at_action_return"] = 1
    assert runner._shared_tree_lane_receipt_valid(lane, preflight=False)[0] is False

    lane = _valid_shared_tree_lane_receipt()
    lane["independent_root_parallel_forest_or_root_stat_merge"] = True
    assert runner._shared_tree_lane_receipt_valid(lane, preflight=False)[0] is False

    lane = _valid_shared_tree_lane_receipt()
    lane["public_lookalike_cross_world_merge_count"] = 1
    assert runner._shared_tree_lane_receipt_valid(lane, preflight=False)[0] is False


def test_r222_runner_requires_explicit_zero_private_randomness_counters():
    runner = _runner_module()
    receipt = {
        "complete": True,
        "private_random_outcome_samples": 0,
        "guessed_random_rules_or_successors": 0,
        "unobserved_random_outcome_advances": 0,
        "finite_chance_enumerations": 0,
        "unforceable_random_pre_boundary_leaf_evaluations": 1,
        "unforceable_random_boundary_reasons": {"stock_forceability_unproven": 1},
        "private_unforceable_chance_samples_prohibited": True,
        "seed_hunting_or_pre_randomization_prohibited": True,
    }
    assert runner._randomness_receipt_valid(receipt) == (True, None)
    receipt.pop("private_random_outcome_samples")
    assert runner._randomness_receipt_valid(receipt)[0] is False


def test_r222_runner_requires_actual_selected_action_to_be_backed_up_in_same_tree():
    runner = _runner_module()
    lane = _valid_shared_tree_lane_receipt()
    row = {
        "executed_action": [1, 2],
        "selected_root_action_receipt": {
            "selected_action": [1, 2],
            "complete_root_legal_actions": [[1, 2], [3, 4]],
            "selected_action_legal": True,
            "selected_action_fully_backed_up": True,
            "selected_action_visit_count": 2,
            "selected_action_completed_backups": 2,
            "shared_logical_tree_identity": lane[
                "shared_logical_tree_identity_or_equivalent_integrity_receipt"
            ],
            "decision_fingerprint": lane["decision_fingerprint"],
        },
    }
    assert runner._selected_shared_tree_root_edge_valid(row, lane) == (True, None)

    row["selected_root_action_receipt"]["shared_logical_tree_identity"] = "tree:other"
    assert runner._selected_shared_tree_root_edge_valid(row, lane)[0] is False
