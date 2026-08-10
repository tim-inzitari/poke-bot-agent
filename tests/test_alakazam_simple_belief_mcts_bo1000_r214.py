from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-simple-belief-mcts-bo1000-r214.json"
R207_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json"
R210_PATH = ROOT / "state/alakazam-rtp-abandonment-r210.json"
R212_PATH = ROOT / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json"

R214_SHA256 = "88f93de0dc82ea6dd54ebb35f52f1c8637db48b9927e790ce41803fea4918f0c"
R207_SHA256 = "d9cb5f8d15e2bebbcbf943f5a273a4116703c3e8549a3328b7d78d161f7b5dce"
R210_SHA256 = "bb9eaa02398175fc5c9bd8e29ce290f102afff234b6d27bf7588fc1e53f09961"
R212_SHA256 = "aa9c7b8158c91d183c092b92bab3047c7bd7af705d539c68cdd3e9c206c0c2b9"
R195_CHECKPOINT = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R195_BUNDLE = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
R195_TREE = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r214_is_a_separate_frozen_no_rtp_direct_vs_belief_mcts_bo1000() -> None:
    contract = _contract()
    frozen = contract["frozen_r195_package"]
    design = contract["evaluation_design"]

    assert contract["schema"] == "poke_bot.alakazam_simple_belief_mcts_bo1000_r214/v1"
    assert contract["owner_decision_revision"] == 214
    assert contract["status"] == (
        "authorized_implementation_preflight_and_exact_bo1000_shadow_"
        "evaluation_pending_prerequisites"
    )
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == R214_SHA256
    assert frozen["checkpoint_sha256"] == R195_CHECKPOINT
    assert frozen["bundle_sha256"] == R195_BUNDLE
    assert frozen[
        "same_checkpoint_bundle_deck_model_config_and_non_experimental_runtime_required_for_both_arms"
    ] is True
    assert frozen["additional_training_authorized"] is False
    assert frozen["evaluation_games_training_eligible"] is False
    assert frozen["disabled_runtime_components"] == {
        "recursive_turn_planner_rtp": True,
        "legacy_rtp_sidecar_or_executor": True,
        "guide_linear_additive_logit_layer": True,
        "guide_logit_bonus": True,
        "guide2vec": True,
        "new_learned_head_or_adapter_training": True,
    }
    assert design["total_games"] == 1_000
    assert design["matched_rng_pairs"] == 500
    assert design["games_per_pair"] == 2
    assert design["complete_all_1000_games_without_early_best_of_stop"] is True
    assert design["seat_balance"] == {
        "belief_mcts_as_seat_0": 500,
        "belief_mcts_as_seat_1": 500,
        "direct_policy_as_seat_0": 500,
        "direct_policy_as_seat_1": 500,
        "swap_seats_within_every_rng_pair": True,
    }
    assert design["actual_turn_order_balance"] == {
        "belief_mcts_actual_first": 500,
        "belief_mcts_actual_second": 500,
        "direct_policy_actual_first": 500,
        "direct_policy_actual_second": 500,
        "initial_actor_is_explicit_sealed_pair_material_not_inferred_from_seat": True,
        "missing_duplicate_crossed_or_unbalanced_pair_fails_closed": True,
    }
    assert design["pairing"]["only_experimental_runtime_graph_difference_is_belief_mcts_wrapping_action_selection"] is True


def test_r214_wraps_the_whole_frozen_model_with_adapter_on_and_labels_sampling_truthfully() -> None:
    contract = _contract()
    adapter = contract["matchup_adapter"]
    arm = contract["experimental_arm"]

    assert adapter == {
        "required_on_both_arms": True,
        "frozen_trained_adapter_bank_required": True,
        "runtime_enabled_required": True,
        "exact_r195_public_matchup_tree_sha256": R195_TREE,
        "same_tree_bank_runtime_graph_and_route_resolution_required_for_both_arms": True,
        "adapter_disabled_shadow_only_or_different_tree_allowed": False,
        "matchup_adapter_is_inside_the_frozen_whole_model_path": True,
    }
    implementation = arm["implementation"]
    assert implementation["module"] == "poke_bot.belief_mcts"
    assert implementation["class"] == "BeliefMCTS"
    assert implementation["search_semantics"] == (
        "public_history_root_sampled_information_set_mcts"
    )
    assert implementation["tree_is_real_search_tree_not_prebuilt_cache_or_direct_policy_alias"] is True
    assert implementation["policy_visible_information_only"] is True
    assert implementation["root_sampled_hidden_particles_required"] is True
    assert implementation["fresh_libcg_search_world_per_simulation_required"] is True
    assert implementation["particle_identity_may_key_a_tree_node"] is False
    assert implementation["strategy_fusion_by_per_particle_action_selection_allowed"] is False
    assert arm["whole_frozen_model_wrapper"] == {
        "all_policy_value_fusion_and_matchup_adapter_inference_comes_from_exact_frozen_r195_package": True,
        "frozen_model_policy_priors_required": True,
        "frozen_model_leaf_values_required": True,
        "matchup_adapter_route_is_used_in_root_and_simulated_model_forwards": True,
        "standalone_reranker_or_partial_model_substitution_allowed": False,
        "gradient_optimizer_parameter_update_or_calibration_training_allowed": False,
        "direct_arm_uses_the_same_complete_frozen_model_path_without_search": True,
    }
    chance = arm["chance_and_information_label"]
    assert chance["root_sampled_hidden_particles"] is True
    assert chance["explicit_coin_behavior"] == "sampled_uniform_coin_outcome_in_each_simulation"
    assert chance["r207_exact_finite_chance_probability_weighted_expectation_claimed"] is False
    assert chance["r207_exact_terminal_or_future_legality_receipt_reused"] is False
    assert chance["required_report_label"] == "root_sampled_belief_mcts_non_r207_exact_chance"
    assert chance["sampled_hidden_or_coin_outcome_may_be_described_as_exact_chance"] is False


def test_r214_timing_report_and_authority_stay_bounded_and_testing_only() -> None:
    contract = _contract()
    timing = contract["timing"]
    telemetry = contract["required_per_turn_telemetry"]
    report = contract["required_report"]
    authority = contract["authority"]

    assert timing["clock"] == "monotonic_wall_clock"
    assert timing["max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert timing["max_planner_wall_seconds_before_each_atomic_action"] == 5.0
    assert timing["values_owned_by_one_easy_to_change_typed_object"] is True
    assert timing["all_particle_sampling_simulator_prior_leaf_adapter_validation_backup_and_receipt_work_charged"] is True
    assert timing["automatic_extension_or_escalation_allowed"] is False
    assert timing["deadline_or_minimum_trusted_simulation_failure_behavior"] == (
        "execute_exact_frozen_r195_direct_policy_action"
    )
    for field in (
        "sims_run",
        "leaf_evaluations",
        "unique_nodes",
        "particles_sampled",
        "chance_samples",
        "requested_simulation_target_completed_within_budget",
        "matchup_adapter_enabled_and_route_receipt",
    ):
        assert telemetry[field] is True
    assert telemetry["full_finite_tree_completion_metric"] == (
        "not_applicable_root_sampled_stochastic_belief_tree"
    )
    assert report["plain_english_result_table_required"] is True
    assert report["missing_telemetry_may_be_imputed"] is False
    assert authority["offline_implementation_and_tests_authorized"] is True
    assert authority["exact_bo1000_shadow_evaluation_authorized_after_prerequisites"] is True
    assert all(
        authority[field] is False
        for field in (
            "training_service_start_authorized",
            "training_or_gradient_updates_authorized",
            "frozen_model_or_matchup_adapter_updates_authorized",
            "serving_eligible",
            "production_action_authority_enabled",
            "selector_change_authorized",
            "checkpoint_publication_authorized",
            "kaggle_submission_authorized",
            "promotion_authorized",
            "r175_restart_authorized",
            "iteration_21_collection_authorized",
        )
    )


def test_r214_projection_matches_and_preserves_r207_r210_and_r212_bytes() -> None:
    contract = _contract()
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )["alakazam_simple_belief_mcts_bo1000_r214"]
    specialist = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )["alakazam_simple_belief_mcts_bo1000_r214"]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]["alakazam_simple_belief_mcts_bo1000_r214"]

    for projection in (protocol, specialist, compatibility):
        revision = projection.get("owner_decision_revision", projection.get("goal_revision"))
        assert revision == 214
        assert projection["status"] == contract["status"]
        assert projection["strategy_id"] == contract["experimental_arm"]["strategy_id"]
    for projection in (specialist, compatibility):
        assert projection["checkpoint_sha256"] == R195_CHECKPOINT
        assert projection["bundle_sha256"] == R195_BUNDLE
        assert projection["rtp_enabled"] is False
        assert projection["guide_linear_additive_logit_layer_enabled"] is False
        assert projection["guide_logit_bonus_enabled"] is False
        assert projection["guide2vec_enabled"] is False
        assert projection["matchup_adapter"]["required_on_both_arms"] is True
        assert projection["matchup_adapter"]["exact_r195_public_matchup_tree_sha256"] == R195_TREE
    assert protocol["timing"]["max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert protocol["timing"]["max_planner_wall_seconds_before_each_atomic_action"] == 5.0
    assert specialist["selector_eligible"] is False
    assert specialist["serving_eligible"] is False
    assert compatibility["evaluation"]["belief_mcts_as_seat_0"] == 500
    assert compatibility["evaluation"]["belief_mcts_actual_second"] == 500
    assert hashlib.sha256(R207_PATH.read_bytes()).hexdigest() == R207_SHA256
    assert hashlib.sha256(R210_PATH.read_bytes()).hexdigest() == R210_SHA256
    assert hashlib.sha256(R212_PATH.read_bytes()).hexdigest() == R212_SHA256


def test_goal_gateway_records_r213_and_r214_without_replacing_r207() -> None:
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    current_revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert current_revision >= 214
    assert "| 213 |" in goal
    assert "| 214 |" in goal
    assert "state/replay-model-inspector-ptcg-visualizer-link-r213.json" in goal
    assert "state/alakazam-simple-belief-mcts-bo1000-r214.json" in goal
    assert "root_sampled_belief_mcts_non_r207_exact_chance" in goal
    assert "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json" in goal
