from __future__ import annotations

import ast
from pathlib import Path

import pytest

from poke_bot.r215_full_turn_belief_mcts import (
    R215CachedBranchStep,
    R215FullTurnBeliefMCTS,
    R215GameClock,
    R215Observation,
    R215PlanRequest,
    R215PlanResult,
    R215TimingConfig,
    R215TurnIdentity,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "poke_bot/r215_full_turn_belief_mcts.py"


class _ScriptedClock:
    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def __call__(self) -> float:
        if not self._readings:
            raise AssertionError("r215 controller read past the supplied monotonic clock")
        return self._readings.pop(0)


class _Planner:
    def __init__(self, *results: R215PlanResult) -> None:
        self._results = list(results)
        self.requests: list[R215PlanRequest] = []

    def plan_turn(self, request: R215PlanRequest) -> R215PlanResult:
        self.requests.append(request)
        if not self._results:
            raise AssertionError("unexpected extra private planning call")
        return self._results.pop(0)


def _observation(
    label: str,
    actions: tuple[tuple[int, ...], ...] = ((0,), (1,), (2,)),
) -> R215Observation:
    return R215Observation(
        public_observation_fingerprint=f"public:{label}",
        legal_actions=actions,
    )


def _cached_step(observation: R215Observation, action: tuple[int, ...]) -> R215CachedBranchStep:
    return R215CachedBranchStep(
        expected_public_observation_fingerprint=observation.public_observation_fingerprint,
        expected_legal_action_order_fingerprint=observation.legal_action_order_fingerprint,
        selected_action=action,
    )


def _controller(
    planner: _Planner,
    *,
    clock: _ScriptedClock,
    game_clock: R215GameClock | None = None,
    timing: R215TimingConfig | None = None,
) -> R215FullTurnBeliefMCTS:
    return R215FullTurnBeliefMCTS(
        planner,
        direct_policy=lambda observation: observation.legal_actions[0],
        timing=timing,
        game_clock=game_clock,
        monotonic=clock,
    )


def test_healthy_600_second_game_gets_20_second_turn_pool_and_five_second_operation_cap() -> None:
    planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    dispatched: list[list[int]] = []
    controller = _controller(planner, clock=_ScriptedClock(0.0, 0.0, 0.0))

    decision = controller.act(
        R215TurnIdentity(0, "turn-1"),
        _observation("root"),
        dispatch=dispatched.append,
    )

    receipt = decision.receipt
    assert decision.source == "belief_mcts"
    assert decision.selected_action == (1,)
    assert dispatched == [[1]]
    assert receipt["game_clock_remaining_seconds_before"] == 600.0
    assert receipt["game_clock_allocator_turn_pool_seconds"] == 20.0
    assert receipt["effective_actual_turn_planner_pool_seconds"] == 20.0
    assert receipt["effective_operation_allowance_seconds"] == 5.0
    assert planner.requests[0].turn_pool_remaining_seconds == 20.0
    assert planner.requests[0].operation_allowance_seconds == 5.0
    assert planner.requests[0].first_decision_search_allowance_seconds == 10.0


def test_outer_game_clock_shrinks_pool_and_exhaustion_falls_back_nonfatally() -> None:
    near_clock = R215GameClock()
    near_clock.remaining_s = 189.0
    planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    controller = _controller(
        planner, clock=_ScriptedClock(0.0, 0.0, 0.0), game_clock=near_clock
    )

    near = controller.act(R215TurnIdentity(0, "near"), _observation("near"))
    assert near.receipt["game_clock_allocator_turn_pool_seconds"] == pytest.approx(19.875)
    assert near.receipt["effective_actual_turn_planner_pool_seconds"] == pytest.approx(19.875)
    assert near.receipt["effective_operation_allowance_seconds"] == 5.0

    exhausted_clock = R215GameClock()
    exhausted_clock.remaining_s = 30.0
    exhausted_planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    exhausted = _controller(
        exhausted_planner,
        clock=_ScriptedClock(0.0, 0.0, 0.0),
        game_clock=exhausted_clock,
    ).act(R215TurnIdentity(0, "reserve"), _observation("reserve"))
    assert exhausted.source == "direct_policy_fallback"
    assert exhausted.selected_action == (0,)
    assert exhausted.receipt["fallback_reason"] == "turn_or_game_clock_exhausted"
    assert exhausted_planner.requests == []


def test_same_turn_cached_hop_uses_residual_pool_charges_validation_and_dispatches_once() -> None:
    first = _observation("first")
    second = _observation("second")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached_step(second, (2,)),),
        )
    )
    dispatched: list[list[int]] = []
    controller = _controller(
        planner, clock=_ScriptedClock(0.0, 3.0, 3.0, 3.0, 3.5, 3.5)
    )
    identity = R215TurnIdentity(0, "same-turn")

    first_decision = controller.act(identity, first, dispatch=dispatched.append)
    cached_decision = controller.act(identity, second, dispatch=dispatched.append)

    assert first_decision.receipt["planner_wall_seconds_used_after"] == 3.0
    assert cached_decision.source == "deterministic_cached_branch"
    assert cached_decision.selected_action == (2,)
    assert cached_decision.receipt["planner_wall_seconds_used_before"] == 3.0
    assert cached_decision.receipt["effective_operation_allowance_seconds"] == 5.0
    assert cached_decision.receipt["planner_wall_seconds_used_after"] == 3.5
    assert cached_decision.receipt["remaining_turn_planner_pool_seconds_after"] == 16.5
    assert cached_decision.receipt["cached_branch_hops"] == 1
    assert len(planner.requests) == 1
    assert dispatched == [[1], [2]]
    assert first_decision.receipt["real_actions_dispatched"] == 1
    assert cached_decision.receipt["real_actions_dispatched"] == 1


def test_chance_boundary_discards_cached_remainder_and_falls_back_without_second_search() -> None:
    first = _observation("chance-first")
    after_chance = _observation("chance-after")
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            continuation=(_cached_step(after_chance, (2,)),),
        )
    )
    controller = _controller(
        planner, clock=_ScriptedClock(0.0, 2.0, 2.0, 2.0, 4.0, 4.0)
    )
    identity = R215TurnIdentity(0, "chance-turn")

    controller.act(identity, first)
    fallback = controller.act(identity, after_chance, boundary_reason="chance")

    assert fallback.source == "direct_policy_fallback"
    assert fallback.selected_action == (0,)
    assert len(planner.requests) == 1
    assert fallback.receipt["planner_wall_seconds_used_before"] == 2.0
    assert fallback.receipt["sampled_or_opaque_chance_boundaries"] == 1
    assert fallback.receipt["cached_branch_hops"] == 0
    assert fallback.receipt["rebuild_count_and_reasons"] == {}
    assert fallback.receipt["cached_branch_invalidations_and_reasons"] == {"chance": 1}
    assert fallback.receipt["fallback_reason"] == "same_turn_boundary_requires_direct_fallback"


def test_first_root_may_use_ten_seconds_and_report_receipt_backed_stability() -> None:
    planner = _Planner(
        R215PlanResult(
            selected_action=(1,),
            sims_run=1,
            root_action_stable=True,
            root_stability_receipt={"winner": [1], "margin": 0.25},
        )
    )
    controller = _controller(planner, clock=_ScriptedClock(0.0, 8.0, 8.0))

    decision = controller.act(R215TurnIdentity(0, "stable"), _observation("stable"))

    assert decision.source == "belief_mcts"
    assert decision.receipt["effective_first_decision_search_allowance_seconds"] == 10.0
    assert decision.receipt["first_decision_search_budget_breach"] is False
    assert decision.receipt["component_operation_budget_breach"] is False
    assert decision.receipt["root_selected_action_stable"] is True
    assert decision.receipt["root_stability_receipt"] == {
        "winner": [1],
        "margin": 0.25,
    }


def test_new_actual_turn_clears_cache_and_reruns_the_near_clock_allocator() -> None:
    game_clock = R215GameClock()
    game_clock.remaining_s = 189.0
    planner = _Planner(
        R215PlanResult(selected_action=(1,), sims_run=1),
        R215PlanResult(selected_action=(1,), sims_run=1),
    )
    controller = _controller(
        planner,
        clock=_ScriptedClock(0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        game_clock=game_clock,
    )

    controller.act(R215TurnIdentity(0, "turn-a"), _observation("a"))
    next_turn = controller.act(R215TurnIdentity(0, "turn-b"), _observation("b"))

    assert len(planner.requests) == 2
    assert next_turn.source == "belief_mcts"
    assert next_turn.receipt["game_clock_remaining_seconds_before"] == 188.0
    assert next_turn.receipt["effective_actual_turn_planner_pool_seconds"] == pytest.approx(19.75)
    assert next_turn.receipt["cached_branch_hops"] == 0


def test_transposition_is_fail_closed_and_timing_identity_changes_with_budget_values() -> None:
    planner = _Planner(R215PlanResult(selected_action=(1,), sims_run=1))
    controller = _controller(planner, clock=_ScriptedClock(0.0, 0.0, 0.0))

    merge = controller.attempt_transposition_merge(
        "same-public-observation", (1,), (2,), "same-search-id-or-seed"
    )
    assert merge.accepted is False
    assert merge.reason == "native_complete_semantic_state_identity_unavailable"

    decision = controller.act(R215TurnIdentity(0, "merge"), _observation("merge"))
    assert decision.receipt["transposition_merges_attempted"] == 1
    assert decision.receipt["transposition_merges_accepted"] == 0
    assert decision.receipt["transposition_model_evaluation_savings"] == 0
    assert decision.receipt["transposition_merge_rejection_reasons"] == {
        "native_complete_semantic_state_identity_unavailable": 1
    }

    defaults = R215TimingConfig()
    assert defaults.identity_sha256 != R215TimingConfig(
        default_turn_pool_seconds=19.0
    ).identity_sha256
    assert defaults.identity_sha256 != R215TimingConfig(
        per_operation_ceiling_seconds=4.0
    ).identity_sha256


def test_controller_has_no_kaggle_or_process_launch_path() -> None:
    parsed = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert {"kaggle", "requests", "subprocess"}.isdisjoint(imported)
    assert "kaggle" not in MODULE_PATH.read_text(encoding="utf-8").lower()
