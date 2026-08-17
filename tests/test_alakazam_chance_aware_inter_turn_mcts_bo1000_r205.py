from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json"
R202_PATH = ROOT / "state/alakazam-chance-aware-inter-turn-mcts-r202.json"
R195_PATH = ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
R202_SHA256 = "5df1eadedb342e90c56aa24c5b59f9887c229243411a4112acb6e69562841d32"
R195_SHA256 = "e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
STATUS = (
    "authorized_implementation_preflight_and_exact_bo1000_shadow_"
    "evaluation_pending_prerequisites"
)
STRATEGY = "chance_aware_cached_inter_turn_expectimax_mcts"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r205_binds_exact_r202_design_and_r195_no_rtp_model() -> None:
    contract = _contract()

    assert contract["schema"] == (
        "poke_bot.alakazam_chance_aware_inter_turn_mcts_bo1000_r205/v1"
    )
    assert contract["owner_decision_revision"] == 205
    assert contract["status"] == STATUS
    assert contract["design_source"]["sha256"] == f"sha256:{R202_SHA256}"
    assert hashlib.sha256(R202_PATH.read_bytes()).hexdigest() == R202_SHA256
    assert contract["frozen_model"]["r195_contract_sha256"] == (
        f"sha256:{R195_SHA256}"
    )
    assert hashlib.sha256(R195_PATH.read_bytes()).hexdigest() == R195_SHA256
    assert contract["frozen_model"]["submission_id"] == 55378392
    assert contract["frozen_model"]["checkpoint_sha256"] == (
        "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
    )
    assert contract["frozen_model"][
        "same_checkpoint_deck_model_config_and_nonplanner_runtime_required_for_both_arms"
    ] is True
    assert contract["frozen_model"]["additional_training_authorized"] is False
    assert contract["frozen_model"]["evaluation_games_training_eligible"] is False


def test_r205_is_exactly_500_seat_swapped_pairs_and_1000_games() -> None:
    design = _contract()["evaluation_design"]

    assert design["total_games"] == 1000
    assert design["matched_rng_pairs"] == 500
    assert design["games_per_pair"] == 2
    assert design["complete_all_1000_games_without_early_best_of_stop"] is True
    assert design["arms"] == [
        "chance_aware_inter_turn_mcts",
        "no_rtp_direct_policy",
    ]
    assert design["seat_balance"] == {
        "mcts_as_seat_0": 500,
        "mcts_as_seat_1": 500,
        "no_rtp_as_seat_0": 500,
        "no_rtp_as_seat_1": 500,
        "swap_seats_within_every_rng_pair": True,
    }
    assert design["pairing"][
        "identical_initial_rng_state_within_each_seat_swapped_pair"
    ] is True
    assert design["pairing"]["missing_duplicate_or_crossed_pair_fails_closed"] is True


def test_r205_enforces_owner_timing_and_search_report() -> None:
    contract = _contract()
    timing = contract["timing"]
    telemetry = contract["required_per_turn_search_telemetry"]
    report = contract["required_report"]

    assert timing["clock"] == "monotonic_wall_clock"
    assert timing["max_planner_wall_seconds_per_actual_turn"] == 20.0
    assert timing["max_planner_wall_seconds_before_each_atomic_action"] == 5.0
    assert timing["all_search_validation_cache_and_backup_work_charged"] is True
    assert timing["deadline_exhaustion_behavior"] == (
        "discard_unverified_partial_work_and_execute_exact_no_rtp_direct_action"
    )
    for key in (
        "result_or_leaf_evaluations_seen",
        "unique_tree_nodes_seen",
        "decision_nodes_expanded",
        "terminal_results_seen",
        "boundary_leaf_results_seen",
        "finite_chance_outcomes_evaluated",
        "requested_tree_fully_expanded_and_backed_up_within_budget",
        "tree_incomplete_reason",
        "deadline_hit",
        "direct_fallback_used",
    ):
        assert telemetry[key] is True
    assert "mean_result_or_leaf_evaluations_seen_per_turn" in report[
        "search_throughput"
    ]
    assert "p95_result_or_leaf_evaluations_seen_per_turn" in report[
        "search_throughput"
    ]
    assert "full_tree_completion_rate_over_all_mcts_turns" in report[
        "tree_completion"
    ]
    assert "full_tree_completion_rate_by_mcts_seat" in report["tree_completion"]
    assert report["raw_per_game_and_per_turn_receipts_preserved"] is True
    assert report["missing_telemetry_may_be_imputed"] is False


def test_r205_launch_is_fail_closed_until_real_mcts_prerequisites() -> None:
    contract = _contract()
    prerequisites = contract["launch_prerequisites"]
    parallelism = contract["evaluation_design"]["game_parallelism"]

    assert prerequisites["real_mcts_or_expectimax_tree_builder_exists"] is True
    assert prerequisites["information_set_safe_successor_branch_abi_receipt"] is True
    assert prerequisites["exact_future_legality_receipt"] is True
    assert prerequisites[
        "monotonic_20_second_turn_and_5_second_action_enforcement_tests_pass"
    ] is True
    assert prerequisites[
        "phase_1_prebuilt_tree_cache_scaffold_alone_satisfies_prerequisites"
    ] is False
    assert prerequisites["launch_before_every_prerequisite_is_immutable_and_valid"] is False
    assert parallelism["parallel_execution_required_when_safe_capacity_exists"] is True
    assert parallelism["remote_workers_allowed"] is True
    assert parallelism["active_attempt10_host_gpu_may_be_used_before_attempt10_terminal"] is False
    assert parallelism[
        "interactive_sessions_may_be_signalled_terminated_or_replaced"
    ] is False


def test_r205_projections_match_and_production_authority_stays_false() -> None:
    contract = _contract()
    key = "alakazam_chance_aware_inter_turn_mcts_bo1000_r205"
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
        assert revision == 205
        assert projection["status"] == STATUS
        assert projection["strategy_id"] == STRATEGY
    assert compatibility["total_games"] == 1000
    assert compatibility["matched_rng_pairs"] == 500
    assert protocol["evaluation"]["mcts_games_per_seat"] == 500
    assert specialist["mean_results_seen_per_turn_required"] is True
    assert all(
        contract["authority"][key] is False
        for key in (
            "training_service_start_authorized",
            "training_or_gradient_updates_authorized",
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


def test_r205_goal_and_human_protocol_record_exact_owner_request() -> None:
    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())
    protocol = " ".join(
        (ROOT / "docs/RL_TRAINING_PROTOCOL.md").read_text(encoding="utf-8").split()
    )

    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 205
    assert "| 205 |" in goal
    assert "500 RNG-matched pairs" in goal
    assert "seat 0 exactly 500 times and seat 1 exactly 500 times" in goal
    assert "mean, median, p95, minimum, and maximum results seen per turn" in goal
    assert "state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json" in goal
    assert "revisions 197–210" in protocol
    assert "complete all 1,000 games" in protocol
    assert "mean/median/p95/min/max results seen per turn" in protocol
