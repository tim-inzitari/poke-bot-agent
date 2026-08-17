from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from poke_bot.recursive_turn_planner.bo1000_evaluation import (
    BO1000_REPORT_SCHEMA,
    R207_CANONICAL_PLANNER_CONFIG_SCHEMA,
    R207_CANONICAL_PLANNER_CONFIG_SHA256,
    R207_CONTRACT_SHA256,
    R207_EVALUATION_ID,
    R207_MAX_ACTION_SECONDS,
    R207_MAX_TURN_SECONDS,
    BO1000EvidenceError,
    BO1000GameReceipt,
    MCTSTurnTelemetry,
    build_bo1000_schedule,
    compile_bo1000_report,
)
from poke_bot.recursive_turn_planner.chance_aware_tree import ChanceAwareSearchConfig


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


CHECKPOINT = _digest("checkpoint")
BUNDLE = _digest("bundle")


def _turn(spec, *, complete: bool = True) -> MCTSTurnTelemetry:
    return MCTSTurnTelemetry(
        game_nonce_sha256=spec.game_nonce_sha256,
        pair_id=spec.pair_id,
        mcts_seat=spec.mcts_seat,
        planner_turn_id=f"{spec.pair_id}:{spec.game_index}:turn0",
        turn_key=(spec.mcts_seat, 0),
        actions_dispatched=1,
        simulator_transitions_seen=20,
        result_or_leaf_evaluations_seen=12,
        simulator_leaf_evaluations_seen=2,
        neural_leaf_evaluations_seen=10,
        unique_tree_nodes_seen=20,
        decision_nodes_expanded=6,
        terminal_exact_results_seen=2,
        boundary_leaf_results_seen=10,
        finite_chance_outcomes_evaluated=2,
        frozen_policy_prior_batches=1,
        frozen_policy_prior_evaluations=6,
        batched_frozen_outcome_value_leaf_reranking_batches=1,
        frozen_outcome_leaf_evaluations=10,
        frozen_value_leaf_evaluations=10,
        nonterminal_leaves_reranked=True,
        terminal_exact_results_not_reranked=True,
        cache_hits=1,
        deterministic_subtree_reuses=1,
        tree_rebuilds=0,
        turn_planner_wall_seconds=4.0,
        max_single_action_planner_wall_seconds=4.0,
        requested_tree_fully_expanded_and_backed_up_within_budget=complete,
        tree_incomplete_reason=None if complete else "action_deadline",
        deadline_hit=not complete,
        direct_fallback_used=not complete,
        selected_action_legal=True,
        selected_action_sha256=_digest("action"),
        legal_actions_sha256=_digest("legal"),
        tree_sha256=_digest("tree"),
        config_sha256=R207_CANONICAL_PLANNER_CONFIG_SHA256,
    )


def _receipts(schedule):
    receipts = []
    for spec in schedule:
        receipts.append(
            BO1000GameReceipt(
                game_nonce_sha256=spec.game_nonce_sha256,
                pair_id=spec.pair_id,
                game_index=spec.game_index,
                mcts_seat=spec.mcts_seat,
                no_rtp_seat=spec.no_rtp_seat,
                pair_rng_snapshot_sha256=_digest(f"rng:{spec.pair_id}"),
                deck_order_rng_sha256=_digest(f"deck:{spec.pair_id}"),
                checkpoint_sha256=CHECKPOINT,
                bundle_sha256=BUNDLE,
                terminal_status="completed",
                winner_seat=spec.mcts_seat if spec.pair_index % 2 == 0 else spec.no_rtp_seat,
                illegal_action_count=0,
                forfeit_count=0,
                crash_count=0,
                timeout_count=0,
                mcts_turns=(_turn(spec, complete=spec.pair_index % 2 == 0),),
            )
        )
    return receipts


def test_schedule_is_exactly_500_pairs_1000_games_and_seat_balanced() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    assert len(schedule) == 1000
    assert len({game.pair_id for game in schedule}) == 500
    assert len({game.pair_nonce_sha256 for game in schedule}) == 500
    assert sum(game.mcts_seat == 0 for game in schedule) == 500
    assert sum(game.mcts_seat == 1 for game in schedule) == 500
    for index in range(500):
        pair = [game for game in schedule if game.pair_index == index]
        assert all(game.pair_id.startswith(f"r207-pair-{index:06d}-") for game in pair)
        assert [(game.game_index, game.mcts_seat) for game in pair] == [(0, 0), (1, 1)]
        assert len({game.pair_nonce_sha256 for game in pair}) == 1


def test_compiler_reports_results_neural_split_tree_completion_and_latency() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    report = compile_bo1000_report(
        schedule,
        _receipts(schedule),
        checkpoint_sha256=CHECKPOINT,
        bundle_sha256=BUNDLE,
    )
    assert report["status"] == "complete"
    assert report["schema"] == BO1000_REPORT_SCHEMA
    assert report["support"] == {
        "scheduled_games": 1000,
        "observed_terminal_game_receipts": 1000,
        "rng_matched_pairs": 500,
        "mcts_as_seat_0": 500,
        "mcts_as_seat_1": 500,
        "no_rtp_as_seat_0": 500,
        "no_rtp_as_seat_1": 500,
        "mcts_turns": 1000,
        "paired_analysis_eligible_pairs": 500,
        "paired_analysis_excluded_failed_closed_pairs": 0,
    }
    assert report["game_outcomes"]["mcts_perspective"] == {
        "win": 500,
        "draw": 0,
        "loss": 500,
        "failed_closed": 0,
    }
    assert report["identities"]["evaluation_id"] == R207_EVALUATION_ID
    assert report["identities"]["experimental_arm"] == (
        "simulator_backed_chance_aware_inter_turn_mcts"
    )
    assert report["identities"]["control_arm"] == "no_rtp_direct_policy"
    assert report["identities"]["r207_contract_sha256"] == R207_CONTRACT_SHA256
    assert report["identities"]["canonical_planner_config_schema"] == (
        R207_CANONICAL_PLANNER_CONFIG_SCHEMA
    )
    assert report["identities"]["canonical_planner_config_sha256"] == (
        R207_CANONICAL_PLANNER_CONFIG_SHA256
    )
    assert report["game_outcomes"]["by_arm"]["no_rtp_direct_policy"] == {
        "win": 500,
        "draw": 0,
        "loss": 500,
        "failed_closed": 0,
    }
    assert report["game_outcomes"]["paired_analysis"] == {
        "eligible_completed_pairs": 500,
        "excluded_failed_closed_pairs": 0,
        "excluded_failed_closed_pair_ids_sha256": "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "imputation_used": False,
    }
    throughput = report["search_throughput"]
    assert throughput["result_or_leaf_evaluations_seen_per_turn"]["mean"] == 12.0
    assert throughput["simulator_transitions_seen_per_turn"]["mean"] == 20.0
    assert throughput["terminal_exact_results_seen_per_turn"]["mean"] == 2.0
    assert throughput["boundary_leaf_results_seen_per_turn"]["mean"] == 10.0
    assert throughput["simulator_leaf_evaluations_seen_per_turn"]["mean"] == 2.0
    assert throughput["neural_leaf_evaluations_seen_per_turn"]["mean"] == 10.0
    assert throughput["frozen_policy_prior_evaluations_per_turn"]["mean"] == 6.0
    assert throughput["frozen_outcome_leaf_evaluations_per_turn"]["mean"] == 10.0
    assert throughput["frozen_value_leaf_evaluations_per_turn"]["mean"] == 10.0
    component_splits = throughput["component_splits"]
    assert component_splits["by_mcts_seat"]["0"][
        "terminal_exact_results_seen_per_turn"
    ]["mean"] == 2.0
    assert component_splits["by_tree_completion"]["complete"][
        "frozen_value_leaf_evaluations_per_turn"
    ]["mean"] == 10.0
    assert component_splits["terminal_exact_vs_nonterminal_frozen_leaf"] == {
        "terminal_exact_results_seen_per_turn": throughput[
            "terminal_exact_results_seen_per_turn"
        ],
        "boundary_leaf_results_seen_per_turn": throughput[
            "boundary_leaf_results_seen_per_turn"
        ],
        "nonterminal_frozen_leaf_evaluations_seen_per_turn": throughput[
            "neural_leaf_evaluations_seen_per_turn"
        ],
        "turns_with_nonterminal_leaves_reranked": 1000,
        "turns_with_terminal_exact_results_not_reranked": 1000,
    }
    assert report["tree_completion"]["complete_turns"] == 500
    assert report["tree_completion"]["full_tree_completion_rate"] == 0.5
    assert report["latency"]["turn_budget_breach_count"] == 0
    assert report["latency"]["turn_budget_seconds"] == R207_MAX_TURN_SECONDS
    assert report["latency"]["action_budget_seconds"] == R207_MAX_ACTION_SECONDS
    assert report["canonical_sha256"].startswith("sha256:")
    assert report["authority"]["training_eligible"] is False
    assert report["telemetry_integrity"] == {
        "raw_per_game_and_per_turn_receipts_preserved": True,
        "missing_telemetry_may_be_imputed": False,
        "paired_failed_closed_outcomes_imputed": False,
        "completed_experimental_games_require_material_mcts_telemetry": True,
        "result_or_leaf_count_invariant": (
            "terminal_exact_results_seen + neural_leaf_evaluations_seen"
        ),
        "r207_timing_override_allowed": False,
    }


def test_compiler_rejects_missing_duplicate_crossed_or_rng_mismatched_evidence() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    receipts = _receipts(schedule)
    with pytest.raises(BO1000EvidenceError, match="missing 1"):
        compile_bo1000_report(
            schedule,
            receipts[:-1],
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )
    with pytest.raises(BO1000EvidenceError, match="duplicate game receipt"):
        compile_bo1000_report(
            schedule,
            [*receipts, receipts[0]],
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )
    with pytest.raises(BO1000EvidenceError, match="BO1000GameReceipt"):
        compile_bo1000_report(
            schedule,
            [*receipts[:-1], object()],
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )

    wrong = replace(
        receipts[1],
        pair_rng_snapshot_sha256=_digest("wrong-rng"),
    )
    with pytest.raises(BO1000EvidenceError, match="RNG snapshot mismatch"):
        compile_bo1000_report(
            schedule,
            [receipts[0], wrong, *receipts[2:]],
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )

    malformed_schedule = tuple(
        replace(spec, pair_index=0) if spec.pair_index == 499 else spec
        for spec in schedule
    )
    with pytest.raises(BO1000EvidenceError, match="pair indices must be exactly"):
        compile_bo1000_report(
            malformed_schedule,
            _receipts(malformed_schedule),
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )

    malformed_pair_ids = tuple(
        replace(spec, pair_id="r207-pair-tampered") if spec.pair_index == 499 else spec
        for spec in schedule
    )
    with pytest.raises(BO1000EvidenceError, match="pair_id must bind"):
        compile_bo1000_report(
            malformed_pair_ids,
            _receipts(malformed_pair_ids),
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )


def test_turn_telemetry_rejects_fake_result_count_or_illegal_action() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    spec = schedule[0]
    payload = {name: getattr(_turn(spec), name) for name in MCTSTurnTelemetry.__dataclass_fields__}
    payload["result_or_leaf_evaluations_seen"] = 13
    with pytest.raises(BO1000EvidenceError, match="terminal exact"):
        MCTSTurnTelemetry(**payload)
    payload["result_or_leaf_evaluations_seen"] = 12
    payload["batched_frozen_outcome_value_leaf_reranking_batches"] = 0
    with pytest.raises(BO1000EvidenceError, match="neural leaf evaluations require a batch"):
        MCTSTurnTelemetry(**payload)
    payload["batched_frozen_outcome_value_leaf_reranking_batches"] = 1
    payload["selected_action_legal"] = False
    with pytest.raises(BO1000EvidenceError, match="exactly legal"):
        MCTSTurnTelemetry(**payload)


def test_turn_telemetry_binds_canonical_r207_config_clocks_and_components() -> None:
    spec = build_bo1000_schedule(_digest("seed"))[0]
    assert ChanceAwareSearchConfig().identity_sha256 == (
        R207_CANONICAL_PLANNER_CONFIG_SHA256
    )
    payload = {
        name: getattr(_turn(spec), name)
        for name in MCTSTurnTelemetry.__dataclass_fields__
    }

    payload["config_sha256"] = _digest("non-r207-config")
    with pytest.raises(BO1000EvidenceError, match="canonical r207"):
        MCTSTurnTelemetry(**payload)

    payload["config_sha256"] = R207_CANONICAL_PLANNER_CONFIG_SHA256
    payload["turn_planner_wall_seconds"] = R207_MAX_TURN_SECONDS + 0.001
    with pytest.raises(BO1000EvidenceError, match="20.0-second budget"):
        MCTSTurnTelemetry(**payload)

    payload["turn_planner_wall_seconds"] = R207_MAX_ACTION_SECONDS
    payload["max_single_action_planner_wall_seconds"] = R207_MAX_ACTION_SECONDS + 0.001
    with pytest.raises(BO1000EvidenceError, match="5.0-second budget"):
        MCTSTurnTelemetry(**payload)

    payload["max_single_action_planner_wall_seconds"] = R207_MAX_ACTION_SECONDS
    payload["frozen_outcome_leaf_evaluations"] = 9
    with pytest.raises(BO1000EvidenceError, match="outcome-head leaf count"):
        MCTSTurnTelemetry(**payload)

    payload["frozen_outcome_leaf_evaluations"] = 10
    payload["frozen_value_leaf_evaluations"] = 9
    with pytest.raises(BO1000EvidenceError, match="value-head leaf count"):
        MCTSTurnTelemetry(**payload)


def test_compiler_rejects_budget_overrides_empty_or_empty_work_mcts_receipts() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    receipts = _receipts(schedule)

    with pytest.raises(BO1000EvidenceError, match="max_turn_seconds is fixed"):
        compile_bo1000_report(
            schedule,
            receipts,
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
            max_turn_seconds=999.0,
        )
    with pytest.raises(BO1000EvidenceError, match="max_action_seconds is fixed"):
        compile_bo1000_report(
            schedule,
            receipts,
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
            max_action_seconds=999.0,
        )

    missing_turns = list(receipts)
    missing_turns[0] = replace(missing_turns[0], mcts_turns=())
    with pytest.raises(BO1000EvidenceError, match="missing MCTS turn telemetry"):
        compile_bo1000_report(
            schedule,
            missing_turns,
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )

    zero_work_turn = replace(
        _turn(schedule[0]),
        simulator_transitions_seen=0,
        result_or_leaf_evaluations_seen=0,
        simulator_leaf_evaluations_seen=0,
        neural_leaf_evaluations_seen=0,
        unique_tree_nodes_seen=0,
        decision_nodes_expanded=0,
        terminal_exact_results_seen=0,
        boundary_leaf_results_seen=0,
        finite_chance_outcomes_evaluated=0,
        frozen_policy_prior_batches=0,
        frozen_policy_prior_evaluations=0,
        batched_frozen_outcome_value_leaf_reranking_batches=0,
        frozen_outcome_leaf_evaluations=0,
        frozen_value_leaf_evaluations=0,
        nonterminal_leaves_reranked=False,
    )
    empty_work = list(receipts)
    empty_work[0] = replace(empty_work[0], mcts_turns=(zero_work_turn,))
    with pytest.raises(BO1000EvidenceError, match="no material MCTS telemetry"):
        compile_bo1000_report(
            schedule,
            empty_work,
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )


def test_compiler_fails_closed_on_post_construction_timing_breach() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    receipts = _receipts(schedule)
    breached_turn = _turn(schedule[0])
    object.__setattr__(
        breached_turn,
        "turn_planner_wall_seconds",
        R207_MAX_TURN_SECONDS + 0.001,
    )
    receipts[0] = replace(receipts[0], mcts_turns=(breached_turn,))

    with pytest.raises(BO1000EvidenceError, match="canonical r207 MCTS contract"):
        compile_bo1000_report(
            schedule,
            receipts,
            checkpoint_sha256=CHECKPOINT,
            bundle_sha256=BUNDLE,
        )


def test_compiler_preserves_failed_pair_exclusion_without_imputation() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    receipts = _receipts(schedule)
    receipts[0] = replace(
        receipts[0],
        terminal_status="failed_closed",
        winner_seat=None,
        mcts_turns=(),
    )

    report = compile_bo1000_report(
        schedule,
        receipts,
        checkpoint_sha256=CHECKPOINT,
        bundle_sha256=BUNDLE,
    )

    assert report["status"] == "complete_with_runtime_failures"
    assert report["support"]["paired_analysis_eligible_pairs"] == 499
    assert report["support"]["paired_analysis_excluded_failed_closed_pairs"] == 1
    assert report["game_outcomes"]["paired_analysis"]["imputation_used"] is False
    assert report["game_outcomes"]["paired_mcts_score"] == 249 / 499
    assert report["game_outcomes"]["by_arm"]["simulator_backed_chance_aware_inter_turn_mcts"][
        "failed_closed"
    ] == 1
    assert report["game_outcomes"]["by_arm"]["no_rtp_direct_policy"][
        "failed_closed"
    ] == 1


def test_game_receipt_rejects_duplicate_turn_key() -> None:
    schedule = build_bo1000_schedule(_digest("seed"))
    receipt = _receipts(schedule)[0]
    first_turn = receipt.mcts_turns[0]
    duplicate_key_turn = replace(
        first_turn,
        planner_turn_id=f"{first_turn.planner_turn_id}:duplicate",
    )
    with pytest.raises(BO1000EvidenceError, match="duplicate turn_key"):
        replace(receipt, mcts_turns=(first_turn, duplicate_key_turn))
