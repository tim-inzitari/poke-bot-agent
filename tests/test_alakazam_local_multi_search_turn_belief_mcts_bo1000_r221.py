"""Focused r221 contract tests for unforceable-randomness search boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r221.json"
)
CONTRACT_SHA256 = "48ffe984c71b4177eb8cb3bc1565cdb05597cc6b65150cace148166c222150e0"
R219_PATH = ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json"
R219_SHA256 = "0ba3e67de761eae8c189cf4bf9900ff01574b54941ca42d0dbdc2b9fdb134f3e"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r221_preserves_r219_and_changes_only_unforceable_randomness() -> None:
    contract = _contract()
    relation = contract["relationship_to_existing_work"]
    chance = contract["experimental_arm"]["chance_and_information_handling"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R219_PATH.read_bytes()).hexdigest() == R219_SHA256
    assert contract["schema"] == (
        "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r221/v1"
    )
    assert contract["owner_decision_revision"] == 221
    assert relation[
        "supersedes_only_r219_stochastic_fallback_semantics_for_a_new_r221_run_only"
    ]
    assert relation["r219_contract_must_be_preserved_byte_for_byte"]
    assert relation[
        "r219_45_second_pool_15_second_segment_multi_search_canary_bo1000_and_no_kaggle_constraints_preserved"
    ]
    assert relation["all_nonstochastic_r219_execution_semantics_remain_in_force"]

    assert chance[
        "simple_finite_chance_may_be_advanced_past_for_effective_value_only_when_all_outcomes_exact_probabilities_independently_forceable_successors_and_future_legality_are_available"
    ]
    assert chance["fully_enumerated_outcome_cap"] == 6
    assert chance["coin_flip_outcome_count"] == 2
    assert chance["standard_die_outcome_count"] == 6
    assert chance[
        "simple_finite_chance_backup_is_exact_probability_weighted_sum_when_that_condition_holds"
    ]
    assert chance[
        "exact_finite_chance_node_requires_force_enumeration_probability_independent_successor_and_future_legality_receipt"
    ]
    assert chance[
        "exact_finite_chance_children_must_each_be_forced_from_the_same_pre_random_state"
    ]
    assert chance["paired_engine_seed_material_is_for_match_reproducibility_only"]
    assert chance[
        "paired_engine_seeding_may_not_hunt_or_pre_randomize_desired_chance_outcomes"
    ]

    assert chance[
        "unforceable_or_incompletely_proven_randomness_is_a_pre_random_leaf_evaluation_boundary"
    ]
    assert chance["unforceable_randomness_behavior"] == (
        "stop_at_pre_random_boundary_and_leaf_evaluate_without_sampling_guessing_or_unobserved_advancement"
    )
    assert (
        chance[
            "private_sampling_of_unforceable_coin_die_or_other_random_outcome_allowed"
        ]
        is False
    )
    assert chance["guessing_random_distribution_rules_or_successors_allowed"] is False
    assert chance["advance_through_unobserved_random_outcome_allowed"] is False
    assert chance[
        "root_sampled_hidden_particles_do_not_authorize_private_random_outcome_sampling"
    ]
    assert chance[
        "realized_unforceable_random_outcome_may_research_only_after_reality_and_from_residual_turn_pool"
    ]


def test_r221_keeps_multi_search_timing_and_canary_bo1000_gates() -> None:
    contract = _contract()
    r219 = json.loads(R219_PATH.read_text(encoding="utf-8"))
    planner = contract["experimental_arm"]["multi_search_actual_turn_planning"]
    timing = contract["timing"]
    outer = timing["source_backed_outer_game_clock"]
    canary = contract["evaluation_design"]["required_r221_canary_before_bo1000"]

    assert planner["one_shared_source_backed_planner_pool_per_actual_turn_required"]
    assert planner["fresh_searches_per_actual_turn_is_fixed"] is False
    assert planner[
        "fresh_searches_may_recur_only_at_meaningful_stop_or_decision_boundaries"
    ]
    assert planner["per_meaningful_search_segment_ceiling_seconds"] == 15.0
    assert planner["later_meaningful_searches_use_only_residual_shared_turn_pool"]
    assert outer["dynamic_game_allowance_formula"] == (
        "min(45.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
    )
    assert timing["shared_actual_turn_planner_pool"]["default_wall_seconds"] == 45.0
    assert (
        timing["meaningful_search_segments"]["per_search_segment_ceiling_seconds"]
        == 15.0
    )

    assert canary["total_games"] == 10
    assert canary["matched_rng_pairs"] == 5
    assert canary["belief_mcts_as_seat_0"] == canary["belief_mcts_as_seat_1"] == 5
    assert (
        canary["belief_mcts_actual_first"] == canary["belief_mcts_actual_second"] == 5
    )
    assert canary["successful_valid_canary_required_before_bo1000_dispatch"]
    assert {
        "finite_chance_enumerations",
        "unforceable_random_pre_boundary_leaf_evaluations",
        "unforceable_random_boundary_reasons",
        "private_random_outcome_samples",
        "guessed_random_rules_or_successors",
        "unobserved_random_outcome_advances",
    } <= set(canary["required_report_facts"])
    assert contract["evaluation_design"]["total_games"] == 1000
    assert contract["evaluation_design"]["matched_rng_pairs"] == 500
    assert contract["launch"][
        "managed_r221_canary_launch_authorized_after_fresh_preflight"
    ]
    assert contract["launch"][
        "local_exploratory_bo1000_authorized_after_fresh_r221_preflight_and_valid_canary"
    ]
    assert contract["authority"]["training_or_gradient_updates_authorized"] is False
    assert contract["authority"]["kaggle_submission_authorized"] is False

    assert (
        timing["source_backed_outer_game_clock"]
        == r219["timing"]["source_backed_outer_game_clock"]
    )
    assert (
        timing["shared_actual_turn_planner_pool"]
        == r219["timing"]["shared_actual_turn_planner_pool"]
    )
    assert (
        timing["meaningful_search_segments"]
        == r219["timing"]["meaningful_search_segments"]
    )
    assert (
        contract["evaluation_design"]["total_games"]
        == r219["evaluation_design"]["total_games"]
    )
    assert (
        contract["evaluation_design"]["matched_rng_pairs"]
        == r219["evaluation_design"]["matched_rng_pairs"]
    )
    for key in (
        "training_or_gradient_updates_authorized",
        "evaluation_games_training_eligible",
        "serving_eligible",
        "production_action_authority_enabled",
        "selector_change_authorized",
        "checkpoint_publication_authorized",
        "promotion_authorized",
        "kaggle_api_calls_authorized",
        "kaggle_upload_authorized",
        "kaggle_queue_authorized",
        "kaggle_submission_authorized",
    ):
        assert contract["authority"][key] is r219["authority"][key] is False


def test_r221_projections_and_goal_bind_the_same_randomness_boundary() -> None:
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r221"
    ]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r221"
    ]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"][
        "alakazam_local_multi_search_turn_belief_mcts_bo1000_r221"
    ]
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    for projection in (protocol, specialists, compatibility):
        revision = projection.get(
            "owner_decision_revision", projection.get("goal_revision")
        )
        assert revision == 221
        assert projection["r219_contract_must_be_preserved_byte_for_byte"]
        assert projection["r219_contract_sha256"] == "sha256:" + R219_SHA256
        assert projection[
            "r219_45_second_pool_15_second_segment_multi_search_canary_bo1000_and_no_kaggle_constraints_preserved"
        ]
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
        assert (
            chance["guessing_random_distribution_rules_or_successors_allowed"] is False
        )
        assert chance["advance_through_unobserved_random_outcome_allowed"] is False
        assert chance[
            "exact_finite_chance_children_must_each_be_forced_from_the_same_pre_random_state"
        ]
        assert chance["paired_engine_seed_material_is_for_match_reproducibility_only"]
        assert chance[
            "paired_engine_seeding_may_not_hunt_or_pre_randomize_desired_chance_outcomes"
        ]
        assert projection["timing"]["default_actual_turn_planner_pool_seconds"] == 45.0
        assert projection["timing"]["per_search_segment_ceiling_seconds"] == 15.0

    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 221
    assert "Under revision 221" in goal
    assert "pre-random" in goal
    assert "may not hunt or pre-randomize" in goal
    assert "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r221.json" in goal
