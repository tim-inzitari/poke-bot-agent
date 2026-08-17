from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json"
CONTRACT_SHA256 = "0ba3e67de761eae8c189cf4bf9900ff01574b54941ca42d0dbdc2b9fdb134f3e"
R218_PATH = ROOT / "state/alakazam-local-first-decision-belief-mcts-bo1000-r218.json"
R218_SHA256 = "5ffb63883290d5cc295cb337ceb9fee9ba075356ab88d67a6cf616ec44bb485a"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r219_restores_multi_search_turn_planning_without_rewriting_r218() -> None:
    contract = _contract()
    relationship = contract["relationship_to_existing_work"]
    planner = contract["experimental_arm"]["multi_search_actual_turn_planning"]
    boundaries = contract["experimental_arm"]["meaningful_stop_or_decision_boundaries"]
    deterministic = contract["experimental_arm"]["deterministic_cached_or_obvious_steps"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R218_PATH.read_bytes()).hexdigest() == R218_SHA256
    assert contract["schema"] == (
        "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1"
    )
    assert contract["owner_decision_revision"] == 219
    assert relationship["supersedes_r218_local_execution_semantics_for_a_new_r219_run_only"]
    assert relationship["r218_contract_must_be_preserved_byte_for_byte"]

    assert planner["one_shared_source_backed_planner_pool_per_actual_turn_required"]
    assert planner["planner_is_receding_horizon_multi_search_turn_planner"]
    assert planner["fresh_searches_per_actual_turn_is_fixed"] is False
    assert planner["fresh_searches_may_recur_only_at_meaningful_stop_or_decision_boundaries"]
    assert planner["per_meaningful_search_segment_ceiling_seconds"] == 15.0
    assert planner["first_meaningful_search_has_no_special_lower_ceiling"]
    assert planner["later_meaningful_searches_use_only_residual_shared_turn_pool"]
    assert planner["no_hard_abort_when_pool_is_short_or_exhausted"]

    assert boundaries["realized_chance_or_information_divergence_may_open_fresh_search_if_residual_pool_remains"]
    assert boundaries["validated_cached_plan_endpoint_may_open_fresh_search_if_residual_pool_remains"]
    assert boundaries["actual_turn_end_may_not_open_another_search"]
    assert boundaries["actual_turn_end_closes_and_discards_the_current_turn_pool_and_cache"]
    assert deterministic["consume_only_required_validation_and_dispatch_time"]
    assert deterministic["forced_or_single_legal_action_is_not_a_search_boundary"]
    assert deterministic["cached_deterministic_continuation_is_not_researched_when_valid"]


def test_r219_timing_chance_and_prior_contracts_are_explicit() -> None:
    contract = _contract()
    timing = contract["timing"]
    outer = timing["source_backed_outer_game_clock"]
    pool = timing["shared_actual_turn_planner_pool"]
    segments = timing["meaningful_search_segments"]
    chance = contract["experimental_arm"]["chance_and_information_handling"]
    priors = contract["experimental_arm"]["prior_allocation"]

    assert outer["dynamic_game_allowance_formula"] == (
        "min(45.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
    )
    assert outer["healthy_game_dynamic_allowance_seconds"] == 45.0
    assert outer["dynamic_allowance_shrink_begins_only_when_remaining_game_seconds_below"] == 390.0
    assert pool["default_wall_seconds"] == 45.0
    assert pool["atomic_steps_do_not_reset_the_turn_pool"]
    assert pool["pool_exhaustion_is_a_direct_fallback_condition_not_a_hard_abort"]
    assert segments["per_search_segment_ceiling_seconds"] == 15.0
    assert segments["effective_fresh_search_allowance_formula"] == (
        "min(15.0, remaining_shared_turn_pool)"
    )
    assert segments["first_segment_has_no_special_lower_ceiling"]
    assert timing["later_meaningful_searches"]["fixed_per_step_search_cap"] == 15.0

    assert chance["fully_enumerated_outcome_cap"] == 6
    assert chance["coin_flip_outcome_count"] == 2
    assert chance["standard_die_outcome_count"] == 6
    assert chance["simple_finite_chance_backup_is_exact_probability_weighted_sum_when_that_condition_holds"]
    assert chance["continue_evaluation_beyond_each_enumerated_child_within_current_segment_and_remaining_turn_budget"]
    assert chance["exact_finite_chance_node_requires_force_enumeration_and_probability_receipt"]
    assert chance["local_approximate_run_must_not_be_labelled_r207_exact_chance"]

    assert priors["frozen_policy_priors_feed_puct"]
    assert priors["dominant_prior_lines_are_naturally_prioritized_by_puct"]
    assert priors["positive_legal_low_prior_lines_remain_available_to_search"]
    assert priors["arbitrary_probability_threshold_pruning_allowed"] is False
    assert priors["nonzero_probability_finite_chance_children_receive_bounded_coverage_when_valid_and_budget_permits"]
    assert priors["nonzero_probability_finite_chance_child_threshold_pruning_allowed"] is False

    assert timing["fixed_simulation_target_allowed"] is False
    assert timing["fixed_depth_target_allowed"] is False
    assert timing["emergency_safety_guard_only"]["simulation_count_ceiling"] == 1_000_000
    assert timing["early_search_stop"]["requires_explicit_stable_root_convergence_receipt"]
    assert timing["early_search_stop"]["requires_fully_backed_up_selected_legal_action"]


def test_r219_requires_the_ten_game_canary_before_bo1000() -> None:
    contract = _contract()
    evaluation = contract["evaluation_design"]
    canary = evaluation["required_r219_canary_before_bo1000"]
    launch = contract["launch"]

    assert canary["total_games"] == 10
    assert canary["matched_rng_pairs"] == 5
    assert canary["games_per_pair"] == 2
    assert canary["belief_mcts_as_seat_0"] == canary["belief_mcts_as_seat_1"] == 5
    assert canary["belief_mcts_actual_first"] == canary["belief_mcts_actual_second"] == 5
    assert canary["successful_valid_canary_required_before_bo1000_dispatch"]
    assert {
        "total_mcts_turns",
        "turns_with_exactly_one_search_segment",
        "turns_with_one_or_more_later_research_segments",
        "average_search_segments_per_turn",
        "maximum_search_segments_per_turn",
        "cache_only_later_steps",
        "finite_chance_enumerations",
        "chance_or_information_rebuilds",
        "simulations",
        "depth",
        "convergence",
        "direct_fallbacks",
        "mcts_action_changes_relative_to_frozen_direct_policy",
    } <= set(canary["required_report_facts"])

    assert evaluation["total_games"] == 1000
    assert evaluation["matched_rng_pairs"] == 500
    assert launch["managed_r219_canary_launch_authorized_after_fresh_preflight"]
    assert launch["local_exploratory_bo1000_authorized_after_fresh_r219_preflight_and_valid_canary"]
    assert launch["contract_recording_step_performs_no_launch_runtime_or_service_change"]
    assert contract["authority"][
        "local_exploratory_bo1000_authorized_after_fresh_r219_preflight_and_valid_canary"
    ]
    assert "local_exploratory_bo1000_authorized_after_fresh_r219_preflight" not in contract[
        "authority"
    ]


def test_r219_projections_and_goal_match_typed_contract() -> None:
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r219"
    ]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r219"
    ]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"]["alakazam_local_multi_search_turn_belief_mcts_bo1000_r219"]
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    for projection in (protocol, specialists, compatibility):
        revision = projection.get("owner_decision_revision", projection.get("goal_revision"))
        assert revision == 219
    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert protocol["r218_contract_must_be_preserved_byte_for_byte"]
    assert specialists["r218_contract_must_be_preserved_byte_for_byte"]
    assert compatibility["r218_contract_sha256"] == "sha256:" + R218_SHA256

    for projection in (protocol, specialists, compatibility):
        timing = projection["timing"]
        assert timing["default_actual_turn_planner_pool_seconds"] == 45.0
        assert timing["per_search_segment_ceiling_seconds"] == 15.0
        assert timing["dynamic_game_allowance_formula"] == (
            "min(45.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
        )
    assert protocol["launch"]["managed_r219_canary_launch_authorized_after_fresh_preflight"]
    assert specialists["managed_r219_canary_launch_authorized_after_fresh_preflight"]
    assert compatibility["managed_r219_canary_launch_authorized_after_fresh_preflight"]
    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 220
    assert "Under revision 219" in goal
    assert "45-second planner pool" in goal
    assert "at most 15 seconds" in goal
    assert "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json" in goal
