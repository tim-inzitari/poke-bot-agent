from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-r202.json"
R201_PATH = ROOT / "state/alakazam-closed-loop-turn-planner-r201.json"
R201_SHA256 = "a7dd427e3222dbcfbdc91572ef35c3d808870a80b2a72f1306f210ba626b8820"
STATUS = (
    "authorized_offline_shadow_design_and_phase_1_implementation_"
    "not_runtime_attached"
)
STRATEGY = "chance_aware_cached_inter_turn_expectimax_mcts"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r202_supersedes_unconditional_r201_replanning_before_implementation() -> None:
    contract = _contract()

    assert contract["schema"] == "poke_bot.alakazam_chance_aware_inter_turn_mcts_r202/v1"
    assert contract["owner_decision_revision"] == 202
    assert contract["status"] == STATUS
    superseded = contract["superseded_design"]
    assert superseded["owner_decision_revision"] == 201
    assert superseded["sha256"] == f"sha256:{R201_SHA256}"
    assert superseded[
        "superseded_before_phase_1_module_candidate_service_or_authority_activation"
    ] is True
    assert hashlib.sha256(R201_PATH.read_bytes()).hexdigest() == R201_SHA256
    assert contract["frozen_predecessors"][
        "attempt10_must_continue_unchanged_to_terminal_or_fail_closed"
    ] is True


def test_r202_reuses_only_exact_deterministic_execution_subtrees() -> None:
    contract = _contract()
    scope = contract["tree_scope"]
    cache = contract["cached_subtree_contract"]

    assert scope["strategy_id"] == STRATEGY
    assert scope["plans_multiple_atomic_actions"] is True
    assert scope["inter_turn_search_allowed"] is True
    assert scope["actions_executed_before_real_observation"] == 1
    assert scope["deterministic_matching_subtree_reused_without_recalculation"] is True
    assert scope["unconditional_replan_after_every_action_required"] is False
    assert scope["turn_key_change_alone_invalidates_tree"] is False
    assert cache["missing_branch_or_chance_outcome_may_default_to_any_child"] is False
    assert cache["direct_policy_action_is_mandatory_candidate_each_decision"] is True
    assert cache["direct_policy_action_is_exact_fallback_each_decision"] is True
    assert "any_realized_chance_outcome_including_a_fully_enumerated_simple_event" in cache[
        "recalculate_or_stop_triggers"
    ]
    assert cache["no_recalculation_conditions"][0] == (
        "the realized transition is attested deterministic and contains no chance outcome"
    )


def test_r202_exact_finite_chance_uses_expectation_but_rebuilds_after_reality() -> None:
    chance = _contract()["finite_chance_contract"]

    assert chance["simple_exact_chance_may_be_expanded"] is True
    assert chance["all_possible_outcomes_must_be_enumerated"] is True
    assert chance["probabilities_must_be_exact_positive_rationals"] is True
    assert chance["probabilities_must_sum_exactly_to_one"] is True
    assert chance["hidden_state_determinization_allowed"] is False
    assert chance["sampling_a_subset_and_calling_it_exact_allowed"] is False
    assert chance["backup_operator"] == (
        "sum_over_all_outcomes_probability_times_child_value"
    )
    assert chance["observed_outcome_may_reuse_precomputed_child_without_recalculation"] is False
    assert chance[
        "realized_chance_is_an_execution_cache_boundary_even_when_its_expectation_was_expanded"
    ] is True


def test_r202_budget_defaults_are_centralized_config_values() -> None:
    budget = _contract()["compute_budget_contract"]

    assert budget["configuration_surface"] == "one_typed_budget_object"
    assert budget["default_max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert budget[
        "default_max_planner_wall_seconds_before_each_atomic_action"
    ] == 5.0
    assert budget["values_must_not_be_scattered_magic_literals"] is True
    assert budget["offline_budget_values_are_easy_to_override_explicitly"] is True
    assert budget["candidate_and_receipts_must_bind_the_effective_budget_values"] is True
    assert budget["partial_tree_has_action_authority_after_timeout"] is False
    assert budget["automatic_budget_extension_or_escalation_allowed"] is False


def test_r202_projections_match_and_deny_every_authority() -> None:
    contract = _contract()
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"]["alakazam_chance_aware_inter_turn_mcts_r202"]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )["alakazam_chance_aware_inter_turn_mcts_r202"]
    specialist = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )["alakazam_chance_aware_inter_turn_mcts_r202"]

    for projection in (compatibility, protocol, specialist):
        revision = projection.get(
            "goal_revision", projection.get("owner_decision_revision")
        )
        assert revision == contract["owner_decision_revision"]
        assert projection["status"] == contract["status"]
        assert projection["strategy_id"] == STRATEGY

    assert compatibility["default_max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert compatibility[
        "default_max_planner_wall_seconds_before_each_atomic_action"
    ] == 5.0
    assert protocol["budget"]["default_max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert specialist["default_max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert all(value is False for value in contract["authority"].values())
    assert all(value is False for value in protocol["authority"].values())
    for key in (
        "active",
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


def test_r202_goal_and_protocol_match_owner_clarifications() -> None:
    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())
    protocol = " ".join(
        (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8").split()
    )

    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 202
    assert "| 202 |" in goal
    assert "reuses the exact matching deterministic subtree without recalculating" in goal
    assert "probability-weighted sum of child values" in goal
    assert "20 seconds total planner wall time" in goal
    assert "5 seconds before any one atomic action" in goal
    assert "state/alakazam-chance-aware-inter-turn-mcts-r202.json" in goal
    assert "revisions 197–210" in protocol
    assert "Matching deterministic children advance without rebuilding" in protocol
    assert "probability-weighted sum of all child values" in protocol
    assert "`20.0` seconds" in protocol
    assert "`5.0` seconds" in protocol
