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
        GuideWeightState(0.05, 1),
        agreement_lift_lower_confidence_bound=0.001,
    )
    assert updated == GuideWeightState(0.05, 0)


def test_two_nonpositive_evaluations_decay_without_ever_increasing() -> None:
    first = update_after_evaluation(
        GuideWeightState(0.05),
        agreement_lift_lower_confidence_bound=0.0,
    )
    second = update_after_evaluation(
        first,
        agreement_lift_lower_confidence_bound=-0.01,
    )
    assert first == GuideWeightState(0.05, 1)
    assert second == GuideWeightState(0.04, 0)
    state = GuideWeightState(0.005)
    state = update_after_evaluation(
        state, agreement_lift_lower_confidence_bound=-0.1
    )
    state = update_after_evaluation(
        state, agreement_lift_lower_confidence_bound=-0.1
    )
    assert state.weight == 0.0
