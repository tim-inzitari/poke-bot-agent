from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json"
R202_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-r202.json"
R205_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json"
R195_PATH = ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
R202_SHA256 = "5df1eadedb342e90c56aa24c5b59f9887c229243411a4112acb6e69562841d32"
R205_SHA256 = "90d1018d67fddc0565adc195f56830aca2f92d60f57d20b6bb1494d956e74a1d"
R195_SHA256 = "e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
STATUS = (
    "authorized_implementation_preflight_and_exact_bo1000_shadow_"
    "evaluation_pending_prerequisites"
)
STRATEGY = (
    "simulator_backed_chance_aware_inter_turn_mcts_with_frozen_policy_priors_"
    "and_batched_frozen_outcome_value_leaf_reranking"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r207_binds_unchanged_r202_r205_and_r195_inputs() -> None:
    contract = _contract()

    assert contract["schema"] == (
        "poke_bot.alakazam_chance_aware_inter_turn_mcts_bo1000_r207/v1"
    )
    assert contract["owner_decision_revision"] == 207
    assert contract["status"] == STATUS
    assert contract["supersedes"]["sha256"] == f"sha256:{R205_SHA256}"
    assert contract["supersedes"]["owner_decision_revision"] == 205
    assert contract["supersedes"]["supersedes_only_experimental_arm_mechanics"] is True
    assert contract["supersedes"][
        "retains_exact_bo1000_pairing_frozen_model_timing_and_authority_limits"
    ] is True
    assert contract["design_source"]["sha256"] == f"sha256:{R202_SHA256}"
    assert contract["frozen_model"]["r195_contract_sha256"] == f"sha256:{R195_SHA256}"
    assert hashlib.sha256(R202_PATH.read_bytes()).hexdigest() == R202_SHA256
    assert hashlib.sha256(R205_PATH.read_bytes()).hexdigest() == R205_SHA256
    assert hashlib.sha256(R195_PATH.read_bytes()).hexdigest() == R195_SHA256
    assert contract["frozen_model"]["checkpoint_sha256"] == (
        "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
    )
    assert contract["frozen_model"]["additional_training_authorized"] is False
    assert contract["frozen_model"]["evaluation_games_training_eligible"] is False


def test_r207_retains_the_exact_1000_game_seat_swapped_mirror() -> None:
    design = _contract()["evaluation_design"]

    assert design["total_games"] == 1000
    assert design["matched_rng_pairs"] == 500
    assert design["games_per_pair"] == 2
    assert design["complete_all_1000_games_without_early_best_of_stop"] is True
    assert design["arms"] == [
        "simulator_backed_chance_aware_inter_turn_mcts",
        "no_rtp_direct_policy",
    ]
    assert design["seat_balance"] == {
        "mcts_as_seat_0": 500,
        "mcts_as_seat_1": 500,
        "no_rtp_as_seat_0": 500,
        "no_rtp_as_seat_1": 500,
        "swap_seats_within_every_rng_pair": True,
    }
    assert design["pairing"]["missing_duplicate_or_crossed_pair_fails_closed"] is True


def test_r207_requires_simulator_exact_terminals_and_frozen_model_only_search() -> None:
    arm = _contract()["experimental_arm"]
    simulator = arm["simulator"]
    priors = arm["frozen_model_policy_priors"]
    reranking = arm["batched_frozen_outcome_value_leaf_reranking"]

    assert arm["strategy_id"] == STRATEGY
    assert simulator["simulator_backed_successor_state_and_exact_future_legality_required"] is True
    assert simulator["simulator_terminal_result_is_exact_leaf_result"] is True
    assert simulator["terminal_result_may_be_replaced_reweighted_or_reranked_by_model"] is False
    assert priors["required"] is True
    assert priors["same_checksum_bound_r195_frozen_model_required"] is True
    assert priors["training_gradient_optimizer_or_parameter_update_allowed"] is False
    assert reranking["required_for_nonterminal_leaf_reranking"] is True
    assert reranking["batched_evaluation_required"] is True
    assert reranking["terminal_exact_results_excluded_from_model_reranking"] is True
    assert reranking["training_gradient_optimizer_or_parameter_update_allowed"] is False


def test_r207_preserves_hard_clocks_and_component_split_telemetry() -> None:
    contract = _contract()
    timing = contract["timing"]
    telemetry = contract["required_per_turn_split_telemetry"]
    report = contract["required_report"]

    assert timing["clock"] == "monotonic_wall_clock"
    assert timing["max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert timing["max_planner_wall_seconds_before_each_atomic_action"] == 5.0
    assert timing["all_simulator_prior_leaf_batch_validation_cache_and_backup_work_charged"] is True
    assert timing["automatic_extension_or_escalation_allowed"] is False
    assert timing["deadline_exhaustion_behavior"] == (
        "discard_unverified_partial_work_and_execute_exact_no_rtp_direct_action"
    )
    for key in (
        "simulator_transitions_seen",
        "terminal_exact_results_seen",
        "frozen_policy_prior_batches",
        "frozen_policy_prior_evaluations",
        "batched_frozen_outcome_value_leaf_reranking_batches",
        "frozen_outcome_leaf_evaluations",
        "frozen_value_leaf_evaluations",
        "terminal_exact_results_not_reranked",
        "split_by_mcts_seat_required",
        "split_by_terminal_exact_result_vs_nonterminal_frozen_leaf_required",
        "split_by_policy_prior_outcome_leaf_and_value_leaf_required",
    ):
        assert telemetry[key] is True
    assert "same_metrics_split_by_mcts_seat" in report["search_throughput"]
    assert "policy_prior_outcome_leaf_value_leaf_and_terminal_exact_result_splits" in report[
        "search_throughput"
    ]


def test_r207_requires_host_noninterference_and_keeps_authority_false() -> None:
    contract = _contract()
    parallelism = contract["evaluation_design"]["game_parallelism"]
    prerequisites = contract["launch_prerequisites"]
    authority = contract["authority"]

    assert parallelism[
        "bert_elmo_and_train_hosts_allowed_only_after_safe_noninterference_preflight"
    ] is True
    assert parallelism["safe_noninterference_receipt_required_per_selected_host"] is True
    assert parallelism["active_attempt10_host_gpu_may_be_used_before_attempt10_terminal"] is False
    assert parallelism["interactive_sessions_may_be_signalled_terminated_or_replaced"] is False
    assert prerequisites["terminal_exact_result_parity_receipt"] is True
    assert prerequisites[
        "same_frozen_model_policy_prior_identity_and_no_training_receipt"
    ] is True
    assert prerequisites[
        "batched_frozen_outcome_value_leaf_reranking_parity_and_no_training_receipt"
    ] is True
    assert prerequisites["safe_noninterference_preflight_for_bert_elmo_and_train"] is True
    assert prerequisites["launch_before_every_prerequisite_is_immutable_and_valid"] is False
    assert authority["offline_implementation_and_tests_authorized"] is True
    assert authority["exact_bo1000_shadow_evaluation_authorized_after_prerequisites"] is True
    assert authority["remote_evaluation_workers_authorized_after_noninterference_preflight"] is True
    assert all(
        authority[key] is False
        for key in (
            "training_service_start_authorized",
            "training_or_gradient_updates_authorized",
            "frozen_prior_or_leaf_model_updates_authorized",
            "attempt10_preemption_restart_or_mutation_authorized",
            "serving_eligible",
            "production_action_authority_enabled",
            "selector_change_authorized",
            "checkpoint_publication_authorized",
            "kaggle_submission_authorized",
            "promotion_authorized",
            "self_promotion_authorized",
            "r175_restart_authorized",
            "iteration_21_collection_authorized",
        )
    )


def test_r207_projections_match_the_typed_contract() -> None:
    key = "alakazam_chance_aware_inter_turn_mcts_bo1000_r207"
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text(encoding="utf-8")
    )["current_owner_overrides"][key]
    protocol = yaml.safe_load(
        (ROOT / "config/rl_protocol.yaml").read_text(encoding="utf-8")
    )[key]
    specialist = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )[key]

    for projection in (compatibility, protocol, specialist):
        revision = projection.get(
            "goal_revision", projection.get("owner_decision_revision")
        )
        assert revision == 207
        assert projection["status"] == STATUS
        assert projection["strategy_id"] == STRATEGY
        assert projection["supersedes_r205_sha256"] == f"sha256:{R205_SHA256}"
    for projection in (compatibility, specialist):
        assert projection["total_games"] == 1000
        assert projection["mcts_games_per_seat"] == 500
        assert projection["max_planner_wall_seconds_per_actual_turn"] == 20.0
        assert projection["max_planner_wall_seconds_before_each_atomic_action"] == 5.0
        assert projection["terminal_simulator_result_is_exact_and_not_model_reranked"] is True
        assert projection["frozen_model_policy_priors_required"] is True
        assert projection[
            "batched_frozen_outcome_value_leaf_reranking_required_for_nonterminal_leaves"
        ] is True
        assert projection["frozen_prior_or_leaf_model_training_allowed"] is False
        assert projection[
            "bert_elmo_and_train_allowed_only_after_safe_noninterference_preflight"
        ] is True
    assert protocol["experimental_arm"]["terminal_simulator_result_is_exact_and_not_model_reranked"] is True
    assert protocol["experimental_arm"]["frozen_prior_or_leaf_model_training_allowed"] is False
    assert protocol["timing"]["max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert protocol["timing"]["max_planner_wall_seconds_before_each_atomic_action"] == 5.0
    assert protocol["authority"]["training_or_gradient_updates_authorized"] is False
    assert specialist["selector_eligible"] is False
    assert specialist["serving_eligible"] is False
    assert specialist["production_action_authority_enabled"] is False


def test_r207_goal_and_protocol_record_the_owner_design_change() -> None:
    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())
    protocol = " ".join(
        (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8").split()
    )

    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 207
    assert "| 207 |" in goal
    assert "simulator-backed chance-aware inter-turn MCTS" in goal
    assert "batched frozen outcome/value reranking" in goal
    assert "exact terminal result" in goal
    assert "Bert, Elmo, and train may participate only after a per-host safe-noninterference" in goal
    assert "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json" in goal
    assert "revisions 197–210" in protocol
    assert "simulator-backed chance-aware inter-turn MCTS" in protocol
    assert "terminal leaf records the exact terminal result" in protocol
    assert "Bert, Elmo, and train may participate only after a per-host" in protocol
