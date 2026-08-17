from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-gpu-turn-planner-r200.json"
R198_PATH = ROOT / "state/alakazam-rtp-realignment-r197.json"
R199_PATH = ROOT / "state/alakazam-rtp-continuation-r199.json"
R198_SHA256 = "ea032624be23341fbae6e0b9b9debf6695a7a3b5a51613cf7294248bdba39c05"
R199_SHA256 = "9de3cce02940bec190dd5d7028036e6943511889c85164d165a40c979d9f7869"
STATUS = "superseded_before_implementation_by_revision_201"
STRATEGY = "conservative_batched_one_turn_complete_action_gpu_reranker"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r200_is_separate_shadow_research_and_freezes_predecessors() -> None:
    contract = _contract()

    assert contract["schema"] == "poke_bot.alakazam_gpu_turn_planner_r200/v1"
    assert contract["owner_decision_revision"] == 200
    assert contract["status"] == STATUS
    assert contract["supersession"]["owner_decision_revision"] == 201
    assert contract["supersession"][
        "module_candidate_service_or_runtime_implemented"
    ] is False
    assert contract["supersession"]["implementation_under_revision_200_allowed"] is False
    predecessors = contract["frozen_predecessors"]
    assert predecessors["r198_contract"]["sha256"] == f"sha256:{R198_SHA256}"
    assert predecessors["r199_continuation"]["sha256"] == f"sha256:{R199_SHA256}"
    assert hashlib.sha256(R198_PATH.read_bytes()).hexdigest() == R198_SHA256
    assert hashlib.sha256(R199_PATH.read_bytes()).hexdigest() == R199_SHA256
    assert predecessors["attempt10"][
        "must_continue_unchanged_to_terminal_or_fail_closed"
    ] is True
    assert predecessors["attempt10"][
        "preemption_restart_retarget_or_retry_allowed"
    ] is False


def test_r200_planner_defaults_to_base_and_cannot_fabricate_targets() -> None:
    contract = _contract()
    strategy = contract["strategy"]

    assert strategy["strategy_id"] == STRATEGY
    assert strategy["decision_scope"] == "current_decision_only"
    assert strategy["complete_ordered_actions_required"] is True
    assert strategy["max_complete_ordered_actions"] == 1024
    assert strategy["stale_multi_action_program_execution_allowed"] is False
    assert strategy["recursive_program_executor_used"] is False
    assert strategy["base_policy_action_is_default"] is True
    assert strategy["planner_override_without_all_gates_allowed"] is False

    targets = contract["counterfactual_target_contract"]
    assert targets["current_r197_trusted_candidate_return_target_count"] == 0
    assert targets["current_r197_trusted_candidate_ranking_pair_count"] == 0
    assert targets["current_r197_trusted_candidate_calibration_target_count"] == 0
    assert targets["current_r197_targets_may_be_relabelled_or_fabricated"] is False
    assert targets["action_space_fingerprint_binding_required"] is True
    assert targets["evaluation_or_kaggle_games_training_eligible"] is False
    assert targets["hidden_opponent_state_as_model_input_allowed"] is False

    gpu = contract["gpu_execution_contract"]
    assert gpu["full_batch_and_microbatch_numerical_equivalence_required"] is True
    assert gpu["candidate_order_invariance_required"] is True
    assert gpu["cpu_reference_parity_required"] is True
    assert gpu["candidate_state_cache_or_rng_mutation_allowed"] is False
    assert gpu["automatic_budget_escalation_allowed"] is False
    assert gpu["attempt10_gpu_or_service_interference_allowed"] is False
    assert all(value is False for value in contract["authority"].values())


def test_r200_compatibility_projections_match_and_deny_authority() -> None:
    contract = _contract()
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]["alakazam_gpu_turn_planner_r200"]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )["alakazam_gpu_turn_planner_r200"]
    specialist = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )["alakazam_gpu_turn_planner_r200"]

    for projection in (compatibility, protocol, specialist):
        revision = projection.get(
            "goal_revision", projection.get("owner_decision_revision")
        )
        assert revision == contract["owner_decision_revision"]
        assert projection["status"] == contract["status"]
        assert projection["strategy_id"] == STRATEGY

    for key in (
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


def test_r200_goal_and_protocol_define_one_turn_gpu_research_only() -> None:
    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())
    protocol = " ".join(
        (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8").split()
    )

    goal_revision = int(goal.split("Revision: `", 1)[1].split("`", 1)[0])
    assert goal_revision >= 200
    assert "| 200 |" in goal
    assert "conservative batched one-turn complete-action GPU reranker" in goal
    assert "must never execute a stale multi-action program" in goal
    assert "state/alakazam-gpu-turn-planner-r200.json" in goal
    assert "Revision 200 authorizes" in protocol
    assert "base policy remains the exact default" in protocol
    assert "never train on evaluation or Kaggle games" in protocol
