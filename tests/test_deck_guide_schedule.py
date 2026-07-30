from __future__ import annotations

import pytest

from poke_bot.pure_rl.deck_guide_schedule import (
    GuideWeightState,
    bootstrap_weight,
    update_after_evaluation,
)


def test_bootstrap_ramps_then_holds() -> None:
    assert [bootstrap_weight(epoch) for epoch in range(1, 6)] == [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]
    assert bootstrap_weight(25) == 0.05
    with pytest.raises(ValueError):
        bootstrap_weight(0)


def test_positive_realized_importance_holds_weight_and_resets_decay() -> None:
    updated = update_after_evaluation(
        GuideWeightState(0.50, 1),
        realized_win_rate_delta_lower_confidence_bound=0.001,
    )
    assert updated == GuideWeightState(0.50, 0)


def test_positive_realized_importance_ramps_fleet_guide_weight() -> None:
    state = GuideWeightState(0.05)
    observed = []
    for _ in range(4):
        state = update_after_evaluation(
            state,
            realized_win_rate_delta_lower_confidence_bound=0.001,
        )
        observed.append(state.weight)
    assert observed == [0.15, 0.25, 0.35, 0.50]


def test_two_nonpositive_evaluations_follow_canonical_decay() -> None:
    first = update_after_evaluation(
        GuideWeightState(0.50),
        realized_win_rate_delta_lower_confidence_bound=0.0,
    )
    second = update_after_evaluation(
        first,
        realized_win_rate_delta_lower_confidence_bound=-0.01,
    )
    assert first == GuideWeightState(0.50, 1)
    assert second == GuideWeightState(0.15, 0)
    state = GuideWeightState(0.075)
    state = update_after_evaluation(
        state, realized_win_rate_delta_lower_confidence_bound=-0.1
    )
    state = update_after_evaluation(
        state, realized_win_rate_delta_lower_confidence_bound=-0.1
    )
    assert state.weight == 0.0


def test_legacy_confidence_bound_alias_remains_readable() -> None:
    assert update_after_evaluation(
        GuideWeightState(0.05),
        agreement_lift_lower_confidence_bound=0.01,
    ).weight == 0.15
