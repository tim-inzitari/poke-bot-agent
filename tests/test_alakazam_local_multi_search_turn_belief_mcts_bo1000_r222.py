"""Focused r222 contract tests for single-run stock-libcg BO1000 sequencing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json"
)
CONTRACT_SHA256 = "8b5a19e8746b8e5f667683ad6437a2f3506aa0fbdcca8495ec2b8bbd1eebeb7e"
R221_PATH = ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r221.json"
R221_SHA256 = "48ffe984c71b4177eb8cb3bc1565cdb05597cc6b65150cace148166c222150e0"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r222_preserves_r221_random_boundary_and_requires_stock_libcg() -> None:
    contract = _contract()
    relation = contract["relationship_to_existing_work"]
    transport = contract["runtime_transport"]
    lanes = contract["experimental_arm"]["in_decision_mcts_trajectory_lanes"]
    duplicate_work = contract["experimental_arm"]["shared_tree_duplicate_work_control"]
    chance = contract["experimental_arm"]["chance_and_information_handling"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R221_PATH.read_bytes()).hexdigest() == R221_SHA256
    assert contract["schema"] == (
        "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r222/v1"
    )
    assert contract["owner_decision_revision"] == 222
    assert relation["supersedes_only_r221_launch_sequencing_for_a_new_r222_run_only"]
    assert relation["r221_contract_must_be_preserved_byte_for_byte"]
    assert relation["r221_stochastic_randomness_boundary_semantics_preserved"]
    assert relation[
        "r222_stock_libcg_one_game_transport_requirement_supersedes_r221_execution_transport_only"
    ]

    assert transport["exact_stock_libcg_archived_in_r195_required"]
    assert transport["r195_bundle_sha256"] == (
        "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
    )
    assert transport["stock_libcg_relative_path"] == "cg/libcg.so"
    assert transport["stock_libcg_sha256"] == (
        "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
    )
    assert transport["stock_libcg_size_bytes"] == 1_342_400
    assert transport["required_stock_libcg_exports"] == [
        "AgentStart",
        "BattleStart",
        "SearchBegin",
        "SearchStep",
        "SearchEnd",
    ]
    assert transport["b77_allowed"] is False
    assert transport["seeded_engine_or_battle_start_seeded_allowed"] is False
    assert transport["batch_or_multi_game_custom_engine_allowed"] is False
    assert transport[
        "each_game_uses_a_fresh_os_process_with_one_stock_libcg_environment"
    ]
    assert transport["private_mcts_uses_the_in_process_stock_search_api"]
    assert transport[
        "multiple_fresh_game_processes_may_share_blackwell_gpu_queue_leaf_inference"
    ]
    topology = transport["stock_agent_search_topology"]
    assert topology["one_kaggle_submission_agent_process_per_game_required"]
    assert topology["one_loaded_stock_libcg_dso_per_game_process_required"]
    assert (
        topology[
            "required_internal_agent_start_simulator_search_arena_count_per_game_process"
        ]
        == 8
    )
    assert topology[
        "one_internal_agent_start_arena_per_persistent_cpu_worker_lane_required"
    ]
    assert topology[
        "all_eight_internal_agent_start_arenas_share_the_one_loaded_stock_libcg_dso"
    ]
    assert topology["internal_agent_start_arenas_are_not_competition_agents"]
    assert topology[
        "one_agent_start_handle_concurrently_shared_across_cpu_worker_lanes_allowed"
    ] is False
    assert (
        topology[
            "required_distinct_search_begin_ids_per_meaningful_mcts_decision_segment"
        ]
        == 8
    )
    assert topology["each_cpu_worker_lane_owns_one_distinct_search_begin_id"]
    isolation_preflight = transport["stock_search_state_isolation_preflight"]
    assert isolation_preflight["required_simultaneous_in_decision_search_states"] == 8
    assert isolation_preflight[
        "required_loaded_stock_libcg_dso_count_per_game_process"
    ] == 1
    assert isolation_preflight[
        "required_internal_agent_start_arena_count_per_game_process"
    ] == 8
    assert isolation_preflight["required_agent_handle_count_per_game_process"] == 8
    assert isolation_preflight[
        "required_distinct_search_begin_id_count_across_the_eight_internal_arenas"
    ] == 8
    assert isolation_preflight[
        "must_prove_one_loaded_dso_and_eight_isolated_internal_agent_start_arenas"
    ]
    assert isolation_preflight[
        "must_prove_eight_isolated_stock_libcg_search_ids_or_states"
    ]
    assert isolation_preflight["unisolated_state_sharing_allowed"] is False
    assert isolation_preflight["lane_count_reduction_or_substitution_allowed"] is False
    assert isolation_preflight["failure_behavior"] == (
        "hard_fail_r222_preflight_before_the_single_bo1000_launch"
    )

    assert (
        lanes["requested_trajectory_lanes"]
        == lanes["required_active_trajectory_lanes"]
        == 8
    )
    assert lanes["exactly_eight_concurrent_mcts_simulation_trajectories_required"]
    assert lanes[
        "one_kaggle_submission_agent_process_and_one_loaded_stock_libcg_dso_are_shared_by_all_eight_lanes"
    ]
    assert lanes[
        "one_internal_agent_start_arena_distinct_search_begin_id_and_persistent_cpu_worker_per_trajectory_required"
    ]
    assert lanes["internal_agent_start_arenas_are_not_competition_agents"]
    assert lanes["master_owns_the_sole_mutable_logical_mcts_tree"]
    assert lanes[
        "master_selects_and_reserves_all_eight_lane_paths_from_the_one_logical_mcts_tree"
    ]
    assert lanes["all_lanes_share_one_frozen_r195_model_and_matchup_adapter_path"]
    assert lanes[
        "each_lane_uses_its_distinct_isolated_stock_search_begin_id_on_its_own_internal_agent_start_arena"
    ]
    assert lanes[
        "the_eight_frontier_leaves_are_gathered_into_one_frozen_gpu_batch_per_round"
    ]
    assert lanes["logical_frontier_leaf_count_per_frozen_gpu_batch"] == 8
    assert lanes["per_lane_or_one_lane_frozen_gpu_batch_allowed"] is False
    assert lanes[
        "master_backs_up_all_eight_frontier_results_into_the_one_shared_logical_tree_before_repeat"
    ]
    assert lanes[
        "repeat_the_eight_lane_frontier_batch_cycle_until_the_segment_budget_ends"
    ]
    assert lanes["independent_root_parallel_forest_or_root_stat_merge_allowed"] is False
    assert lanes["all_requested_lanes_must_be_active_for_mcts_action_authority"]
    assert lanes["partial_lane_result_may_authorize_an_action"] is False
    assert lanes["serial_lane_execution_or_sequential_fallback_allowed"] is False
    assert lanes["one_lane_baseline_or_ratio_comparison_allowed"] is False
    assert lanes["is_not_eight_games_models_or_literal_beam_search"]
    assert lanes[
        "cross_game_process_parallelism_is_distinct_from_in_decision_trajectory_parallelism"
    ]
    assert lanes[
        "stock_search_state_isolation_preflight_must_hard_fail_before_launch_if_not_proven"
    ]
    assert duplicate_work["virtual_loss_or_path_and_leaf_reservations_required"]
    assert duplicate_work[
        "in_flight_frozen_model_evaluations_must_be_coalesced_or_cached_when_safe"
    ]
    assert duplicate_work[
        "deduplication_requires_native_semantic_state_equivalence_not_public_observation_lookalike"
    ]
    assert duplicate_work[
        "hidden_random_or_future_legality_worlds_may_not_merge_on_public_lookalike_alone"
    ]
    assert duplicate_work[
        "zero_outstanding_path_or_leaf_reservations_required_at_action_return"
    ]
    assert duplicate_work[
        "all_eight_path_and_leaf_reservations_must_be_released_after_every_complete_batch_backup"
    ]
    assert duplicate_work["unavoidable_repeat_work_must_be_counted_and_reported"]

    assert chance[
        "exact_finite_chance_enumeration_may_use_only_a_stock_libcg_abi_capability_proven_in_r222_preflight"
    ]
    assert chance[
        "b77_seeded_or_custom_engine_may_not_supply_exact_chance_forceability"
    ]
    assert chance[
        "stock_forceability_is_not_assumed_and_unproven_stock_chance_stops_at_the_pre_random_boundary"
    ]
    assert chance[
        "unforceable_or_incompletely_proven_randomness_is_a_pre_random_leaf_evaluation_boundary"
    ]
    assert (
        chance[
            "private_sampling_of_unforceable_coin_die_or_other_random_outcome_allowed"
        ]
        is False
    )
    assert chance["guessing_random_distribution_rules_or_successors_allowed"] is False
    assert chance["advance_through_unobserved_random_outcome_allowed"] is False


def test_r222_uses_one_blackwell_bo1000_with_a_nonblocking_prefix_diagnostic() -> None:
    contract = _contract()
    evaluation = contract["evaluation_design"]
    launch = evaluation["single_run_launch"]
    prefix = evaluation["live_prefix_diagnostic"]
    pairing = evaluation["pairing"]

    assert evaluation["execution_target"]["host"] == "train-blackwell"
    assert evaluation["total_games"] == 1000
    assert evaluation["seat_swapped_pairs"] == 500
    assert evaluation["games_per_pair"] == 2
    assert evaluation["complete_all_1000_games_without_early_best_of_stop"]
    assert launch["one_whole_500_pair_1000_game_launch_required"]
    assert launch["separate_canary_or_prefix_gate_allowed"] is False
    assert launch["all_pairs_authorized_by_the_single_launch_after_fresh_preflight"]
    assert launch["prefix_diagnostic_may_not_be_used_to_authorize_pairs_5_through_499"]

    assert prefix["pair_indices_inclusive"] == [0, 4]
    assert prefix["game_indices_inclusive"] == [0, 9]
    assert prefix["seat_swapped_pairs"] == 5
    assert prefix["games"] == 10
    assert prefix["diagnostic_only"]
    assert prefix["is_a_canary_or_validity_gate"] is False
    assert prefix["must_not_pause_the_already_running_remainder"]
    assert prefix["must_not_restart_the_already_running_remainder"]
    assert prefix["must_not_reauthorize_the_already_running_remainder"]

    assert pairing["pair_rng_streams"] == "independent_unmatched"
    assert pairing["paired_rng_or_identical_seed_claim_allowed"] is False
    assert pairing["independent_unmatched_rng_streams_must_be_reported"]
    assert pairing["seat_swapping_does_not_make_rng_streams_matched"]
    assert "matched_rng_pairs" not in evaluation
    seat_balance = evaluation["seat_balance"]
    assert (
        seat_balance["belief_mcts_as_seat_0"]
        == seat_balance["belief_mcts_as_seat_1"]
        == 500
    )
    actual_turn_order_balance = evaluation["actual_turn_order_balance"]
    assert (
        actual_turn_order_balance["belief_mcts_actual_first"]
        == actual_turn_order_balance["belief_mcts_actual_second"]
        == 500
    )
    assert evaluation["actual_turn_order_balance"][
        "exact_500_500_balance_is_enforced_by_the_wrapper"
    ]


def test_r222_projections_and_goal_keep_kaggle_smoke_non_authoritative() -> None:
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r222"
    ]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r222"
    ]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"][
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r222"
    ]
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    for projection in (protocol, specialists, compatibility):
        revision = projection.get(
            "owner_decision_revision", projection.get("goal_revision")
        )
        assert revision == 222
        assert projection["r221_contract_must_be_preserved_byte_for_byte"]
        assert projection["r221_contract_sha256"] == "sha256:" + R221_SHA256
        transport = projection["runtime_transport"]
        assert transport["exact_stock_libcg_archived_in_r195_required"]
        assert transport["stock_libcg_sha256"] == (
            "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
        )
        assert transport["b77_allowed"] is False
        assert transport["seeded_engine_or_battle_start_seeded_allowed"] is False
        assert transport["batch_or_multi_game_custom_engine_allowed"] is False
        isolation_preflight = transport["stock_search_state_isolation_preflight"]
        assert (
            isolation_preflight["required_simultaneous_in_decision_search_states"] == 8
        )
        assert isolation_preflight[
            "must_prove_eight_isolated_stock_libcg_search_ids_or_states"
        ]
        assert isolation_preflight["unisolated_state_sharing_allowed"] is False
        assert (
            isolation_preflight["lane_count_reduction_or_substitution_allowed"] is False
        )
        chance = projection.get("experimental_arm", projection)[
            "chance_and_information"
        ]
        assert chance[
            "unforceable_or_incompletely_proven_randomness_is_a_pre_random_leaf_evaluation_boundary"
        ]
        assert (
            chance[
                "private_sampling_of_unforceable_coin_die_or_other_random_outcome_allowed"
            ]
            is False
        )
        evaluation = projection["evaluation"]
        assert evaluation["total_games"] == 1000
        assert evaluation["seat_swapped_pairs"] == 500
        assert (
            evaluation["single_run_launch"]["separate_canary_or_prefix_gate_allowed"]
            is False
        )
        assert (
            evaluation["live_prefix_diagnostic"]["is_a_canary_or_validity_gate"]
            is False
        )
        assert evaluation["pairing"]["pair_rng_streams"] == "independent_unmatched"
        smoke = projection["stock_portable_kaggle_runtime_compatibility_smoke"]
        assert smoke["separate_stock_portable_runtime_compatibility_smoke_required"]
        assert smoke["must_not_call_kaggle_api"]
        assert smoke["must_not_queue_upload_or_submit_to_kaggle"]
        assert smoke["must_not_grant_kaggle_runtime_or_submission_authority"]

    lane_projections = (
        protocol["experimental_arm"]["in_decision_mcts_trajectory_lanes"],
        specialists["in_decision_mcts_trajectory_lanes"],
        compatibility["in_decision_mcts_trajectory_lanes"],
    )
    duplicate_work_projections = (
        protocol["experimental_arm"]["shared_tree_duplicate_work_control"],
        specialists["shared_tree_duplicate_work_control"],
        compatibility["shared_tree_duplicate_work_control"],
    )
    for lanes, duplicate_work in zip(
        lane_projections, duplicate_work_projections, strict=True
    ):
        assert (
            lanes["requested_trajectory_lanes"]
            == lanes["required_active_trajectory_lanes"]
            == 8
        )
        assert lanes[
            "one_kaggle_submission_agent_process_and_one_loaded_stock_libcg_dso_are_shared_by_all_eight_lanes"
        ]
        assert lanes[
            "one_internal_agent_start_arena_distinct_search_begin_id_and_persistent_cpu_worker_per_trajectory_required"
        ]
        assert lanes["internal_agent_start_arenas_are_not_competition_agents"]
        assert lanes["master_owns_the_sole_mutable_logical_mcts_tree"]
        assert lanes[
            "master_selects_and_reserves_all_eight_lane_paths_from_the_one_logical_mcts_tree"
        ]
        assert lanes["all_lanes_share_one_frozen_r195_model_and_matchup_adapter_path"]
        assert lanes[
            "each_lane_uses_its_distinct_isolated_stock_search_begin_id_on_its_own_internal_agent_start_arena"
        ]
        assert lanes[
            "the_eight_frontier_leaves_are_gathered_into_one_frozen_gpu_batch_per_round"
        ]
        assert lanes["logical_frontier_leaf_count_per_frozen_gpu_batch"] == 8
        assert lanes["per_lane_or_one_lane_frozen_gpu_batch_allowed"] is False
        assert lanes[
            "master_backs_up_all_eight_frontier_results_into_the_one_shared_logical_tree_before_repeat"
        ]
        assert lanes[
            "repeat_the_eight_lane_frontier_batch_cycle_until_the_segment_budget_ends"
        ]
        assert (
            lanes["independent_root_parallel_forest_or_root_stat_merge_allowed"]
            is False
        )
        assert lanes["partial_lane_result_may_authorize_an_action"] is False
        assert lanes["serial_lane_execution_or_sequential_fallback_allowed"] is False
        assert lanes["one_lane_baseline_or_ratio_comparison_allowed"] is False
        assert lanes["is_not_eight_games_models_or_literal_beam_search"]
        assert lanes[
            "stock_search_state_isolation_preflight_must_hard_fail_before_launch_if_not_proven"
        ]

        assert duplicate_work["virtual_loss_or_path_and_leaf_reservations_required"]
        assert duplicate_work[
            "in_flight_frozen_model_evaluations_must_be_coalesced_or_cached_when_safe"
        ]
        assert duplicate_work[
            "deduplication_requires_native_semantic_state_equivalence_not_public_observation_lookalike"
        ]
        assert duplicate_work[
            "hidden_random_or_future_legality_worlds_may_not_merge_on_public_lookalike_alone"
        ]
        assert duplicate_work[
            "zero_outstanding_path_or_leaf_reservations_required_at_action_return"
        ]
        assert duplicate_work[
            "all_eight_path_and_leaf_reservations_must_be_released_after_every_complete_batch_backup"
        ]
        assert duplicate_work["unavoidable_repeat_work_must_be_counted_and_reported"]

    for projection in (protocol, specialists, compatibility):
        reporting = projection["reporting"]
        assert reporting["in_decision_lane_and_leaf_microbatch_telemetry_required"]
        assert (
            reporting["in_decision_lane_or_microbatch_telemetry_may_not_be_imputed"]
            is False
        )
        assert set(reporting["required_in_decision_lane_telemetry_fields"]) >= {
            "requested_trajectory_lanes",
            "active_trajectory_lanes",
            "loaded_stock_libcg_dso_count",
            "internal_agent_start_arena_count",
            "internal_agent_start_call_count",
            "internal_agent_start_handle_count",
            "distinct_search_begin_id_count",
            "cpu_worker_lane_count",
            "unique_raw_handle_count",
            "search_begin_calls",
            "search_step_calls",
            "search_release_calls",
            "search_end_calls",
            "lane_topology",
            "master_owned_shared_tree_integrity_receipt",
            "isolated_stock_search_state_count",
            "leaf_microbatch_count",
            "leaf_microbatch_size_distribution",
            "eight_frontier_leaf_gpu_batch_count",
            "eight_frontier_leaf_gpu_batch_size_distribution",
            "lane_trajectory_count",
            "lane_backup_count",
            "all_eight_frontier_results_backed_up_before_repeat_count",
            "shared_tree_reservation_or_contention_receipt",
            "virtual_loss_or_path_leaf_reservation_count",
            "in_flight_frozen_eval_coalescing_count",
            "safe_frozen_eval_cache_hit_count",
            "unavoidable_repeat_expansion_count",
            "outstanding_path_or_leaf_reservation_count_at_action_return",
            "stock_search_state_isolation_preflight_result",
            "one_lane_baseline_or_ratio_comparison_count",
        }
        assert reporting["one_lane_baseline_or_ratio_comparison_count_must_be_zero"]

    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 222
    assert "Under revision 222 (Alakazam MCTS sequencing)" in goal
    assert "never B77" in goal
    assert "independent, unmatched RNG streams" in goal
    assert "eight concurrent simulation trajectories" in goal
    assert "not eight games, models, or a literal" in goal
    assert "one shared logical MCTS tree" in goal
    assert "zero outstanding" in goal
    assert "one loaded stock r195 `cg/libcg.so` DSO" in goal
    assert "internal `AgentStart()`" in goal
    assert "one-lane baseline or ratio" in goal
    assert "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json" in goal
