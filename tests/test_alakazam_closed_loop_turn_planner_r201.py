from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-closed-loop-turn-planner-r201.json"
R200_PATH = ROOT / "state/alakazam-gpu-turn-planner-r200.json"
R200_SHA256 = "046144920e1e66679f340d2c84ffadb45b8b348766b6f276c1494fc44c603361"
STATUS = "superseded_before_implementation_by_revision_202"
STRATEGY = "closed_loop_receding_horizon_full_turn_gpu_planner"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r201_supersedes_single_step_before_implementation() -> None:
    contract = _contract()

    assert contract["schema"] == "poke_bot.alakazam_closed_loop_turn_planner_r201/v1"
    assert contract["owner_decision_revision"] == 201
    assert contract["status"] == STATUS
    assert contract["implementation_allowed"] is False
    assert contract["superseded_by"]["owner_decision_revision"] == 202
    assert contract["superseded_by"][
        "superseded_before_phase_1_module_candidate_service_or_authority_activation"
    ] is True
    superseded = contract["superseded_design"]
    assert superseded["owner_decision_revision"] == 200
    assert superseded["sha256"] == f"sha256:{R200_SHA256}"
    assert superseded[
        "superseded_before_module_candidate_service_or_authority_activation"
    ] is True
    assert hashlib.sha256(R200_PATH.read_bytes()).hexdigest() == R200_SHA256
    assert contract["frozen_predecessors"][
        "attempt10_must_continue_unchanged_to_terminal_or_fail_closed"
    ] is True


def test_r201_plans_full_turn_but_executes_closed_loop() -> None:
    contract = _contract()
    scope = contract["turn_scope"]

    assert scope["strategy_id"] == STRATEGY
    assert scope["plans_multiple_atomic_actions"] is True
    assert scope["planning_horizon"] == "current_turn_until_typed_end_turn"
    assert scope["cross_turn_planning_allowed"] is False
    assert scope["single_step_reranker_is_sufficient"] is False
    assert scope["receding_horizon_execution"] is True
    assert scope["actions_executed_before_real_observation"] == 1
    assert scope["replan_after_every_real_action_result"] is True
    assert scope["cached_remainder_has_action_authority"] is False

    state = contract["closed_loop_state_contract"]
    assert state["fresh_policy_visible_observation_required_each_step"] is True
    assert state["fresh_complete_legal_action_set_required_each_step"] is True
    assert state["fresh_option_encoding_required_each_step"] is True
    assert state["branch_predicate_observation_required"] is True
    assert state["missing_branch_predicate_default_branch_allowed"] is False
    assert state["reuse_root_legal_actions_after_state_change_allowed"] is False
    assert state["reuse_root_option_hidden_after_state_change_allowed"] is False
    assert state["direct_policy_action_is_mandatory_candidate_each_step"] is True
    assert state["direct_policy_action_is_exact_fallback_each_step"] is True


def test_r201_requires_real_successors_and_trusted_multistep_targets() -> None:
    contract = _contract()
    search = contract["trajectory_search_contract"]

    assert search["valid_successor_state_mechanism_required"] is True
    assert search["exact_future_legality_mechanism_required"] is True
    assert search["information_set_safe_arbitrary_decision_branch_abi_required"] is True
    assert search[
        "battle_start_pairing_snapshot_counts_as_arbitrary_decision_branch_abi"
    ] is False
    assert search["current_r197_latent_rollout_counts_as_exact_simulator_or_beam"] is False
    assert search[
        "beam_mcts_or_rollout_runtime_claim_allowed_before_branch_abi_receipt"
    ] is False
    assert search["hidden_opponent_state_as_planner_input_allowed"] is False

    targets = contract["learning_and_target_contract"]
    for key in (
        "current_r197_should_recurse_target_count",
        "current_r197_root_plan_target_count",
        "current_r197_trusted_candidate_return_target_count",
        "current_r197_trusted_candidate_ranking_pair_count",
        "current_r197_trusted_candidate_calibration_target_count",
    ):
        assert targets[key] == 0
    assert targets["current_r197_targets_may_be_relabelled_or_fabricated"] is False
    assert targets["trusted_multi_step_transition_targets_required"] is True
    assert targets["trusted_trajectory_return_and_ranking_targets_required"] is True
    assert targets["evaluation_or_kaggle_games_training_eligible"] is False
    assert all(value is False for value in contract["authority"].values())


def test_r201_compatibility_projections_match_and_deny_authority() -> None:
    contract = _contract()
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]["alakazam_closed_loop_turn_planner_r201"]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )["alakazam_closed_loop_turn_planner_r201"]
    specialist = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )["alakazam_closed_loop_turn_planner_r201"]

    for projection in (compatibility, protocol, specialist):
        revision = projection.get(
            "goal_revision", projection.get("owner_decision_revision")
        )
        assert revision == contract["owner_decision_revision"]
        assert projection["status"] == contract["status"]
        assert projection["implementation_allowed"] is False
        assert projection["strategy_id"] == STRATEGY

    for key in (
        "training_service_start_authorized",
        "evaluation_service_start_authorized",
        "serving_eligible",
        "action_authority_enabled",
        "selector_change_authorized",
        "checkpoint_publication_authorized",
        "kaggle_submission_authorized",
        "promotion_authorized",
        "r175_restart_authorized",
        "iteration_21_collection_authorized",
    ):
        assert compatibility[key] is False
    assert all(value is False for value in protocol["authority"].values())
    for key in (
        "selector_eligible",
        "serving_eligible",
        "action_authority_enabled",
        "training_service_start_authorized",
        "evaluation_service_start_authorized",
        "checkpoint_publication_authorized",
        "promotion_authorized",
        "r175_restart_authorized",
        "iteration_21_collection_authorized",
        "automatic_kaggle_submission_allowed",
    ):
        assert specialist[key] is False


def test_r201_goal_and_protocol_say_full_turn_not_one_step() -> None:
    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())
    protocol = " ".join(
        (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8").split()
    )

    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 201
    assert "| 201 |" in goal
    assert "one whole current turn with multiple atomic steps" in goal
    assert "executes only the next atomic action" in goal
    assert "state/alakazam-closed-loop-turn-planner-r201.json" in goal
    assert "revisions 197–207" in protocol
    assert "planning one whole current turn over multiple atomic actions" in protocol
    assert "execute exactly one next action" in protocol
