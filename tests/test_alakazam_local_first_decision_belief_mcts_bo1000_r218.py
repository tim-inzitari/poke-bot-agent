from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-local-first-decision-belief-mcts-bo1000-r218.json"
CONTRACT_SHA256 = "5ffb63883290d5cc295cb337ceb9fee9ba075356ab88d67a6cf616ec44bb485a"
R216_PATH = ROOT / "state/alakazam-local-approximate-belief-mcts-bo1000-r216.json"
R216_SHA256 = "2e260755c33d9fa8a2f821f7eb5e6edb8cd609112d8e01e7c94937aefbe776f3"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r218_first_decision_only_search_contract_preserves_r216_bytes() -> None:
    contract = _contract()
    first_decision = contract["experimental_arm"]["first_actual_decision_search"]
    later_decisions = contract["experimental_arm"]["later_same_turn_decisions"]
    timing = contract["timing"]
    early_stop = timing["early_search_stop"]

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R216_PATH.read_bytes()).hexdigest() == R216_SHA256
    assert contract["schema"] == "poke_bot.alakazam_local_first_decision_belief_mcts_bo1000_r218/v1"
    assert contract["owner_decision_revision"] == 218
    assert contract["relationship_to_existing_work"][
        "supersedes_r216_local_execution_semantics_for_a_new_r218_run_only"
    ]
    assert contract["relationship_to_existing_work"]["r216_contract_must_be_preserved_byte_for_byte"]

    assert first_decision["fresh_searches_per_actual_turn"] == 1
    assert first_decision["only_the_first_actual_decision_may_launch_fresh_search"]
    assert not first_decision["later_same_turn_fresh_search_allowed"]
    assert first_decision["new_actual_turn_resets_first_decision_search_eligibility"]
    assert later_decisions["only_allowed_action_sources"] == [
        "fingerprint_validated_cached_plan",
        "exact_frozen_r195_direct_policy_fallback",
    ]
    assert not later_decisions["fresh_search_or_tree_rebuild_allowed"]
    assert later_decisions["cached_plan_validation_required"]
    assert later_decisions["invalid_missing_diverged_or_exhausted_cached_plan_behavior"] == (
        "execute_exact_frozen_r195_direct_policy_action_without_launching_search"
    )
    assert later_decisions["same_turn_cache_validation_or_direct_fallback_may_not_open_a_new_search_budget"]

    outer_clock = timing["source_backed_outer_game_clock"]
    assert outer_clock["dynamic_game_allowance_formula"] == (
        "min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))"
    )
    assert timing["maximum_fresh_search_wall_seconds_at_first_actual_decision"] == 10.0
    assert timing["effective_first_decision_fresh_search_allowance_formula"] == (
        "min(10.0, dynamic_game_allowance)"
    )
    assert timing["maximum_first_decision_search_or_fallback_operation_wall_seconds"] == 10.0
    partition = timing["first_decision_private_search_and_direct_fallback_partition"]
    assert partition["private_search_wall_seconds_when_full_first_decision_allowance_available"] == 9.5
    assert partition["exact_frozen_r195_direct_policy_fallback_reserve_wall_seconds"] == 0.5
    assert partition["partition_must_not_extend_the_effective_first_decision_allowance"]
    assert "maximum_model_or_simulator_operation_wall_seconds" not in timing
    calls = timing["individual_model_or_simulator_calls"]
    assert not calls["five_second_outer_call_cap_inherited_from_r216"]
    assert not calls["hard_outer_call_cap_enforced"]
    assert calls["observed_and_telemetrized"]
    assert timing["later_same_turn_cache_validation_uses_no_fresh_search_allowance"]
    assert not timing["fixed_simulation_target_allowed"]
    assert not timing["fixed_depth_target_allowed"]
    assert not timing["fixed_simulation_or_depth_completion_gate_allowed"]
    assert timing["emergency_safety_guard_only"]["simulation_count_ceiling"] == 1_000_000
    assert not timing["emergency_safety_guard_only"]["is_a_requested_simulation_or_depth_target"]
    assert early_stop["requires_explicit_stable_root_convergence_receipt"]
    assert early_stop["requires_fully_backed_up_selected_legal_action"]
    assert early_stop["selected_action_must_be_in_current_complete_legal_action_set"]
    assert not early_stop["partial_tree_time_only_simulation_count_only_or_unbacked_action_early_stop_allowed"]


def test_r218_bo1000_boundary_and_future_h100_target() -> None:
    contract = _contract()
    evaluation = contract["evaluation_design"]
    target = contract["future_separately_authorized_kaggle_runtime_target"]

    assert evaluation["total_games"] == 1000
    assert evaluation["matched_rng_pairs"] == 500
    assert evaluation["games_per_pair"] == 2
    assert evaluation["complete_all_1000_games_without_early_best_of_stop"]
    assert evaluation["seat_balance"]["belief_mcts_as_seat_0"] == 500
    assert evaluation["seat_balance"]["belief_mcts_as_seat_1"] == 500
    assert evaluation["actual_turn_order_balance"]["belief_mcts_actual_first"] == 500
    assert evaluation["actual_turn_order_balance"]["belief_mcts_actual_second"] == 500

    assert target["applies_only_after_a_separate_future_owner_authorization"]
    assert target["aws_instance_equivalent"] == "p5.4xlarge-equivalent"
    assert target["accelerator"] == "H100 80GB"
    assert target["accelerator_memory_gib"] == 80
    assert target["host_memory_gib"] == 256
    assert target["vcpus"] == 16
    assert target["optimization_requirements"] == [
        "batched_frozen_inference",
        "resource_aware_search",
    ]
    assert not target["this_contract_grants_kaggle_runtime_or_submission_authority"]
    assert contract["launch"]["current_task_performs_no_launch_runtime_or_service_change"]
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
        "r175_restart_authorized",
        "iteration_21_collection_authorized",
    ):
        assert contract["authority"][key] is False


def test_r218_projections_and_goal_match_typed_contract() -> None:
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[ 
        "alakazam_local_first_decision_belief_mcts_bo1000_r218"
    ]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[ 
        "alakazam_local_first_decision_belief_mcts_bo1000_r218"
    ]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"]["alakazam_local_first_decision_belief_mcts_bo1000_r218"]
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    for projection in (protocol, specialists, compatibility):
        revision = projection.get("owner_decision_revision", projection.get("goal_revision"))
        assert revision == 218
    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert protocol["r216_contract_must_be_preserved_byte_for_byte"]
    assert specialists["r216_contract_must_be_preserved_byte_for_byte"]
    assert compatibility["r216_contract_sha256"] == "sha256:" + R216_SHA256
    assert protocol["experimental_arm"]["fresh_searches_per_actual_turn"] == 1
    assert not protocol["experimental_arm"]["later_same_turn_fresh_search_or_tree_rebuild_allowed"]
    assert compatibility["timing"]["effective_first_decision_fresh_search_allowance_formula"] == (
        "min(10.0, dynamic_game_allowance)"
    )
    assert compatibility["timing"]["maximum_first_decision_search_or_fallback_operation_wall_seconds"] == 10.0
    assert compatibility["timing"]["first_decision_private_search_and_direct_fallback_partition"][
        "private_search_wall_seconds_when_full_first_decision_allowance_available"
    ] == 9.5
    assert "maximum_model_or_simulator_operation_wall_seconds" not in compatibility["timing"]
    assert not compatibility["timing"]["individual_model_or_simulator_calls"][
        "hard_outer_call_cap_enforced"
    ]
    assert compatibility["timing"]["individual_model_or_simulator_calls"][
        "observed_and_telemetrized"
    ]
    assert not compatibility["timing"]["fixed_simulation_target_allowed"]
    assert not compatibility["timing"]["fixed_depth_target_allowed"]
    assert compatibility["timing"]["emergency_ceiling_is_only_a_safety_guard"]
    assert compatibility["future_separately_authorized_kaggle_runtime_target"]["accelerator"] == (
        "H100 80GB"
    )
    assert not compatibility["future_separately_authorized_kaggle_runtime_target"][
        "kaggle_authority_granted_by_this_contract"
    ]
    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 220
    assert "Under revision 218" in goal
    assert "min(10.0, dynamic_game_allowance)" in goal
    assert "inherited five-second outer call cap" in goal
    assert "state/alakazam-local-first-decision-belief-mcts-bo1000-r218.json" in goal
