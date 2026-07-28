"""Deterministic current-deck guide weight schedule.

The evaluator supplies a training-ineligible agreement-lift confidence bound.
This module contains no game data access and cannot turn evaluation trajectories
into training examples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideWeightState:
    weight: float
    consecutive_nonpositive_evaluations: int = 0


def bootstrap_weight(epoch: int) -> float:
    """Return the exact 1-indexed 25-epoch bootstrap guide schedule."""
    if epoch < 1 or epoch > 25:
        raise ValueError("bootstrap guide epoch must be in [1, 25]")
    if epoch <= 5:
        return round(0.01 + (epoch - 1) * 0.01, 10)
    return 0.05


def update_after_evaluation(
    state: GuideWeightState,
    *,
    agreement_lift_lower_confidence_bound: float,
) -> GuideWeightState:
    """Hold on positive evidence; decay after two non-positive evaluations."""
    weight = float(state.weight)
    if not 0.0 <= weight <= 0.05:
        raise ValueError("guide weight must be in [0, 0.05]")
    if agreement_lift_lower_confidence_bound > 0.0:
        return GuideWeightState(weight=weight, consecutive_nonpositive_evaluations=0)
    count = int(state.consecutive_nonpositive_evaluations) + 1
    if count < 2 or weight == 0.0:
        return GuideWeightState(
            weight=weight,
            consecutive_nonpositive_evaluations=count,
        )
    decayed = weight * 0.8
    return GuideWeightState(
        weight=0.0 if decayed < 0.005 else round(decayed, 10),
        consecutive_nonpositive_evaluations=0,
    )


__all__ = ["GuideWeightState", "bootstrap_weight", "update_after_evaluation"]
