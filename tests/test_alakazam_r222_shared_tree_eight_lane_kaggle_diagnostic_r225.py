"""Focused r225 guard for the one-shot shared-tree eight-lane diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
)
CONTRACT_SHA256 = "7db1b6770bf71623cf0ed48ddeeaa503f5921cbfd7f39c6b8a9322b138a803f4"
R222_PATH = ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json"
R222_SHA256 = "8b5a19e8746b8e5f667683ad6437a2f3506aa0fbdcca8495ec2b8bbd1eebeb7e"
R224_PATH = ROOT / "state/alakazam-phase1-ladder-eight-lane-search-r224.json"
R224_SHA256 = "700b96912b8bb6d871883790dbf300781d9941b65a962416dd288d95619ebb84"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r225_is_exactly_one_bound_shared_tree_eight_lane_diagnostic() -> None:
    contract = _contract()
    local = contract["local_preflight"]
    diagnostic = contract["one_shot_kaggle_diagnostic"]
    authority = contract["authority"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R222_PATH.read_bytes()).hexdigest() == R222_SHA256
    assert hashlib.sha256(R224_PATH.read_bytes()).hexdigest() == R224_SHA256
    assert contract["owner_decision_revision"] == 225
    assert contract["owner_topology_correction_revision"] == 227
    assert contract["owner_gameplay_viability_correction_revision"] == 228
    assert contract["status"] == (
        "conditionally_authorized_one_shot_kaggle_full_gameplay_viability_test_"
        "pending_local_exact_package_full_game_smoke"
    )
    assert contract["relationship_to_existing_work"][
        "r222_contract_must_be_preserved_byte_for_byte"
    ]
    assert contract["relationship_to_existing_work"][
        "r224_contract_must_be_preserved_byte_for_byte"
    ]
    assert contract["relationship_to_existing_work"][
        "r225_must_not_reuse_or_conflate_r224_root_parallel_two_vcpu_no_gpu_candidate"
    ]
    assert contract["relationship_to_existing_work"][
        "r228_rough_one_shot_full_gameplay_viability_test_is_the_active_deliverable_"
        "ahead_of_r222_bo1000"
    ]
    assert contract["relationship_to_existing_work"][
        "r228_does_not_modify_or_authorize_r222_bo1000"
    ]

    assert local["offline_exact_package_build_authorized"]
    assert local["local_child_process_capability_and_throughput_smoke_authorized"]
    assert local["one_kaggle_submission_agent_process_per_child_required"]
    assert local["one_loaded_stock_libcg_dso_per_child_required"]
    assert local[
        "required_internal_agent_start_simulator_search_arena_count_per_child"
    ] == 8
    assert local[
        "one_internal_agent_start_arena_per_persistent_cpu_worker_lane_required"
    ]
    assert local[
        "all_eight_internal_agent_start_arenas_share_the_one_loaded_stock_libcg_dso"
    ]
    assert local["internal_agent_start_arenas_are_not_competition_agents"]
    assert local[
        "one_agent_start_handle_concurrently_shared_across_cpu_worker_lanes_allowed"
    ] is False
    assert local[
        "required_distinct_search_begin_id_count_across_the_eight_internal_arenas"
    ] == 8
    assert local["each_cpu_worker_lane_owns_one_distinct_search_begin_id"]
    assert local["local_exact_package_full_game_smoke_required_before_the_one_shot"]
    assert local[
        "local_exact_package_full_game_smoke_must_exercise_branching_gameplay_decisions"
    ]
    gameplay = local["full_gameplay_shared_tree_contract"]
    assert gameplay[
        "rough_one_shot_gameplay_viability_test_precedes_r222_bo1000"
    ]
    assert gameplay["required_at_every_branching_gameplay_decision"]
    assert gameplay[
        "search_begin_exactly_once_per_lane_per_branching_gameplay_decision"
    ]
    assert gameplay[
        "retain_exact_lane_handle_search_id_tuple_across_repeated_depth_waves"
    ]
    assert gameplay[
        "simulator_advances_after_each_assigned_action_to_discover_next_legal_actions"
    ]
    assert gameplay[
        "returned_frontier_states_are_batched_through_the_frozen_r195_gpu_evaluator"
    ]
    assert gameplay["maximum_returned_frontier_states_per_frozen_gpu_batch"] == 8
    assert gameplay[
        "all_valid_frontier_results_back_up_into_the_one_master_owned_shared_tree"
    ]
    assert gameplay[
        "continue_depth_waves_until_branch_boundary_clean_deadline_or_convergence"
    ]
    assert gameplay[
        "legal_root_action_requires_at_least_one_completed_backup_when_not_using_"
        "clean_deadline_fallback"
    ]
    assert gameplay["only_forced_single_action_prompts_may_bypass_search"]
    assert gameplay[
        "unforceable_randomness_stops_the_branch_before_any_unobserved_outcome"
    ]
    assert gameplay[
        "private_random_sampling_guessing_or_unobserved_outcome_advance_allowed"
    ] is False
    assert gameplay["clean_per_decision_or_turn_deadline_is_nonfatal"]
    assert gameplay[
        "clean_deadline_requires_all_heads_to_be_cleaned_and_all_reservations_released"
    ]
    assert gameplay[
        "clean_deadline_with_a_legal_fully_backed_root_action_returns_the_best_"
        "backed_action"
    ]
    assert gameplay[
        "clean_deadline_with_zero_completed_backups_uses_frozen_direct_policy_"
        "fallback_after_cleanup"
    ]
    assert gameplay[
        "frozen_direct_policy_fallback_for_a_branching_searched_decision_allowed_"
        "only_after_clean_deadline_with_zero_completed_backups"
    ]
    assert gameplay[
        "zero_completed_backups_for_any_non_deadline_or_structural_cause_is_hard_failure"
    ]
    assert gameplay["greedy_serial_or_partial_mcts_action_fallback_allowed"] is False
    sparse_fail_closed = local["sparse_log_structural_fail_closed_behavior"]
    assert sparse_fail_closed["missing_lane_crash_or_unclean_timeout_is_structural_failure"]
    assert sparse_fail_closed[
        "stale_or_duplicate_lane_handle_search_id_or_state_is_structural_failure"
    ]
    assert sparse_fail_closed["illegal_edge_or_illegal_returned_action_is_structural_failure"]
    assert sparse_fail_closed[
        "incomplete_depth_wave_tree_invariant_violation_or_cleanup_leak_is_structural_failure"
    ]
    assert sparse_fail_closed["unbacked_non_deadline_action_is_structural_failure"]
    assert sparse_fail_closed[
        "structural_failure_must_exit_nonzero_without_greedy_serial_or_partial_fallback"
    ]
    assert sparse_fail_closed[
        "exactly_one_explicit_success_marker_may_be_emitted_only_after_the_full_"
        "gameplay_loop_passes"
    ]
    assert sparse_fail_closed[
        "clean_deadline_is_not_structural_failure_when_the_deadline_cleanup_contract_"
        "is_satisfied"
    ]
    assert local["eight_simultaneous_in_decision_trajectory_lanes_required"]
    assert local["one_shared_logical_mcts_tree_required"]
    assert local[
        "master_owns_the_sole_mutable_shared_logical_mcts_tree"
    ]
    assert local[
        "master_selects_reserves_and_backs_up_all_eight_lane_paths_into_the_same_logical_tree"
    ]
    assert local[
        "each_lane_requires_an_isolated_stock_search_begin_id_on_its_own_internal_agent_start_arena"
    ]
    assert local[
        "gather_eight_frontier_leaves_into_one_frozen_gpu_batch_per_round_required"
    ]
    assert local["logical_frontier_leaf_count_per_frozen_gpu_batch"] == 8
    assert local["per_lane_or_one_lane_frozen_gpu_batch_allowed"] is False
    assert local["all_eight_frontier_results_must_be_backed_up_before_the_next_round"]
    assert local["repeat_eight_lane_frontier_batch_cycle_until_budget_ends"]
    assert local["shared_gpu_leaf_microbatch_broker_uses_only_the_frozen_model"]
    assert local["partial_lane_mcts_action_authority_allowed"] is False
    assert (
        local[
            "eight_games_models_root_parallel_forest_literal_beam_or_serial_lane_fallback_allowed"
        ]
        is False
    )
    assert (
        local["public_lookalike_hidden_random_or_future_legality_world_merge_allowed"]
        is False
    )
    assert local["zero_outstanding_reservations_required_at_action_return"]
    assert local["one_lane_baseline_or_ratio_comparison_allowed"] is False
    assert local["one_lane_baseline_or_ratio_comparison_count_must_be_zero"]
    assert local["failed_or_missing_local_preflight_behavior"] == (
        "hard_fail_closed_and_do_not_call_kaggle"
    )

    assert diagnostic[
        "exactly_one_direct_submission_conditionally_authorized_after_all_local_preflight_receipts"
    ]
    assert diagnostic[
        "rough_one_shot_full_gameplay_viability_test_is_the_active_deliverable_"
        "ahead_of_r222_bo1000"
    ]
    assert diagnostic["local_exact_package_full_game_smoke_required_before_submission"]
    assert diagnostic["competition"] == "pokemon-tcg-ai-battle"
    assert diagnostic["submission_message_required_literal"] == (
        "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY"
    )
    assert (
        diagnostic["submission_message_must_begin_with_literal"]
        == "DONT USE FOR REVIEW"
    )
    assert diagnostic[
        "must_hard_fail_the_diagnostic_if_eight_active_lanes_are_not_present"
    ]
    assert diagnostic["automatic_retry_allowed"] is False
    assert diagnostic["automatic_copy_or_resubmission_allowed"] is False
    assert diagnostic["queue_or_batch_submission_allowed"] is False
    assert diagnostic["must_not_modify_or_pause_r222_blackwell_bo1000"]
    assert diagnostic["expected_submission_resources"] == {
        "compute": "AWS p5.4xlarge (or equivalent)",
        "gpu": "NVIDIA H100 GPU (80 GB VRAM)",
        "ram": "256 GiB",
        "vcpus": 16,
        "resource_probe_and_receipt_required": True,
        "unverified_or_mismatched_resource_result_is_not_a_valid_viability_pass": True,
    }
    assert diagnostic["immutable_pre_submission_binding_receipt_required"]
    assert diagnostic["missing_or_mismatched_binding_behavior"] == "do_not_submit"

    assert authority[
        "one_direct_kaggle_diagnostic_submission_authorized_only_after_all_typed_preconditions"
    ]
    assert authority["local_exact_package_full_game_smoke_authorized"]
    assert authority[
        "rough_one_shot_full_gameplay_viability_test_is_active_deliverable_"
        "ahead_of_r222_bo1000"
    ]
    assert authority["kaggle_api_call_permitted_now_before_preconditions"] is False
    assert authority["kaggle_upload_permitted_now_before_preconditions"] is False
    assert authority["kaggle_queue_submission_permitted"] is False
    assert authority["automatic_kaggle_submission_allowed"] is False
    assert authority["promotion_authorized"] is False


def test_r225_projections_and_goal_keep_the_one_shot_noncolliding() -> None:
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[
        "alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225"
    ]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[
        "alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225"
    ]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"][
        "alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225"
    ]
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    for projection in (protocol, specialists, compatibility):
        assert (
            projection.get("owner_decision_revision", projection.get("goal_revision"))
            == 225
        )
        assert projection["owner_topology_correction_revision"] == 227
        assert projection["owner_gameplay_viability_correction_revision"] == 228
        assert projection["status"] == (
            "conditionally_authorized_one_shot_kaggle_full_gameplay_viability_test_"
            "pending_local_exact_package_full_game_smoke"
        )
        assert projection["r222_contract_sha256"] == "sha256:" + R222_SHA256
        assert projection["r222_contract_must_be_preserved_byte_for_byte"]
        assert projection["r224_contract_sha256"] == "sha256:" + R224_SHA256
        assert projection["r224_contract_must_be_preserved_byte_for_byte"]
        assert projection[
            "r225_must_not_reuse_or_conflate_r224_root_parallel_two_vcpu_no_gpu_candidate"
        ]
        local = projection["local_preflight"]
        assert local["one_kaggle_submission_agent_process_per_child_required"]
        assert local["one_loaded_stock_libcg_dso_per_child_required"]
        assert local[
            "required_internal_agent_start_simulator_search_arena_count_per_child"
        ] == 8
        assert local["internal_agent_start_arenas_are_not_competition_agents"]
        assert local[
            "one_agent_start_handle_concurrently_shared_across_cpu_worker_lanes_allowed"
        ] is False
        assert local[
            "required_distinct_search_begin_id_count_across_the_eight_internal_arenas"
        ] == 8
        assert local[
            "local_exact_package_full_game_smoke_required_before_the_one_shot"
        ]
        assert local[
            "local_exact_package_full_game_smoke_must_exercise_branching_gameplay_decisions"
        ]
        gameplay = local["full_gameplay_shared_tree_contract"]
        assert gameplay[
            "rough_one_shot_gameplay_viability_test_precedes_r222_bo1000"
        ]
        assert gameplay["required_at_every_branching_gameplay_decision"]
        assert gameplay[
            "search_begin_exactly_once_per_lane_per_branching_gameplay_decision"
        ]
        assert gameplay[
            "retain_exact_lane_handle_search_id_tuple_across_repeated_depth_waves"
        ]
        assert gameplay[
            "simulator_advances_after_each_assigned_action_to_discover_next_legal_actions"
        ]
        assert gameplay[
            "returned_frontier_states_are_batched_through_the_frozen_r195_gpu_evaluator"
        ]
        assert gameplay["maximum_returned_frontier_states_per_frozen_gpu_batch"] == 8
        assert gameplay[
            "all_valid_frontier_results_back_up_into_the_one_master_owned_shared_tree"
        ]
        assert gameplay[
            "continue_depth_waves_until_branch_boundary_clean_deadline_or_convergence"
        ]
        assert gameplay[
            "legal_root_action_requires_at_least_one_completed_backup_when_not_using_"
            "clean_deadline_fallback"
        ]
        assert gameplay["only_forced_single_action_prompts_may_bypass_search"]
        assert gameplay[
            "unforceable_randomness_stops_the_branch_before_any_unobserved_outcome"
        ]
        assert gameplay[
            "private_random_sampling_guessing_or_unobserved_outcome_advance_allowed"
        ] is False
        assert gameplay["clean_per_decision_or_turn_deadline_is_nonfatal"]
        assert gameplay[
            "clean_deadline_requires_all_heads_to_be_cleaned_and_all_reservations_released"
        ]
        assert gameplay[
            "clean_deadline_with_a_legal_fully_backed_root_action_returns_the_best_"
            "backed_action"
        ]
        assert gameplay[
            "clean_deadline_with_zero_completed_backups_uses_frozen_direct_policy_"
            "fallback_after_cleanup"
        ]
        assert gameplay[
            "frozen_direct_policy_fallback_for_a_branching_searched_decision_allowed_"
            "only_after_clean_deadline_with_zero_completed_backups"
        ]
        assert gameplay[
            "zero_completed_backups_for_any_non_deadline_or_structural_cause_is_hard_failure"
        ]
        assert gameplay["greedy_serial_or_partial_mcts_action_fallback_allowed"] is False
        sparse_fail_closed = local["sparse_log_structural_fail_closed_behavior"]
        assert sparse_fail_closed[
            "missing_lane_crash_or_unclean_timeout_is_structural_failure"
        ]
        assert sparse_fail_closed[
            "stale_or_duplicate_lane_handle_search_id_or_state_is_structural_failure"
        ]
        assert sparse_fail_closed[
            "illegal_edge_or_illegal_returned_action_is_structural_failure"
        ]
        assert sparse_fail_closed[
            "incomplete_depth_wave_tree_invariant_violation_or_cleanup_leak_is_structural_failure"
        ]
        assert sparse_fail_closed["unbacked_non_deadline_action_is_structural_failure"]
        assert sparse_fail_closed[
            "structural_failure_must_exit_nonzero_without_greedy_serial_or_partial_fallback"
        ]
        assert sparse_fail_closed[
            "exactly_one_explicit_success_marker_may_be_emitted_only_after_the_full_"
            "gameplay_loop_passes"
        ]
        assert sparse_fail_closed[
            "clean_deadline_is_not_structural_failure_when_the_deadline_cleanup_contract_"
            "is_satisfied"
        ]
        assert local["eight_simultaneous_in_decision_trajectory_lanes_required"]
        assert local["one_shared_logical_mcts_tree_required"]
        assert local["master_owns_the_sole_mutable_shared_logical_mcts_tree"]
        assert local[
            "gather_eight_frontier_leaves_into_one_frozen_gpu_batch_per_round_required"
        ]
        assert local["logical_frontier_leaf_count_per_frozen_gpu_batch"] == 8
        assert local["per_lane_or_one_lane_frozen_gpu_batch_allowed"] is False
        assert local["all_eight_frontier_results_must_be_backed_up_before_the_next_round"]
        assert local["partial_lane_or_serial_lane_mcts_authority_allowed"] is False
        assert local["zero_outstanding_reservations_required_at_action_return"]
        assert local["one_lane_baseline_or_ratio_comparison_allowed"] is False
        diagnostic = projection["one_shot_kaggle_diagnostic"]
        assert diagnostic[
            "rough_one_shot_full_gameplay_viability_test_is_the_active_deliverable_"
            "ahead_of_r222_bo1000"
        ]
        assert diagnostic[
            "local_exact_package_full_game_smoke_required_before_submission"
        ]
        assert diagnostic["competition"] == "pokemon-tcg-ai-battle"
        assert (
            diagnostic["submission_message_must_begin_with_literal"]
            == "DONT USE FOR REVIEW"
        )
        assert diagnostic["automatic_retry_allowed"] is False
        assert diagnostic["automatic_copy_or_resubmission_allowed"] is False
        assert diagnostic["queue_or_batch_submission_allowed"] is False
        assert diagnostic["must_not_modify_or_pause_r222_blackwell_bo1000"]
        authority = projection["authority"]
        assert authority["local_exact_package_full_game_smoke_authorized"]
        assert authority[
            "rough_one_shot_full_gameplay_viability_test_is_active_deliverable_"
            "ahead_of_r222_bo1000"
        ]
        assert authority[
            "one_direct_kaggle_diagnostic_submission_authorized_only_after_all_typed_preconditions"
        ]
        assert authority["kaggle_api_call_permitted_now_before_preconditions"] is False
        assert authority["kaggle_upload_permitted_now_before_preconditions"] is False
        assert authority["automatic_kaggle_submission_allowed"] is False

    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 228
    assert "Under revision 225 (Alakazam shared-tree eight-lane diagnostic)" in goal
    assert "Under revision 228-MCTS" in goal
    assert "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY" in goal
    assert "r224's distinct root-parallel" in goal
    assert "A clean per-decision/turn deadline is nonfatal only after all heads and" in goal
    assert "With zero completed backups caused solely by that clean deadline" in goal
