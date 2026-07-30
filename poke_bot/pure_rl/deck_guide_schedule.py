"""Deterministic fleet-wide current-deck guide weight schedule.

The evaluator supplies a training-ineligible paired realized-win confidence
bound. Positive evidence ramps the auxiliary guide weight, sustained evidence
holds the plateau, and two non-positive reviews trigger decay.
This module contains no game data access and cannot turn evaluation trajectories
into training examples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideWeightState:
    weight: float
    consecutive_nonpositive_evaluations: int = 0


MAXIMUM_AUXILIARY_WEIGHT = 0.50
POSITIVE_RAMP_SCHEDULE = (0.15, 0.25, 0.35, 0.50)
DECAY_SCHEDULE = (0.15, 0.075, 0.0)


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
    realized_win_rate_delta_lower_confidence_bound: float | None = None,
    agreement_lift_lower_confidence_bound: float | None = None,
) -> GuideWeightState:
    """Apply one fleet ramp/hold/decay review from ineligible paired evidence.

    ``agreement_lift_lower_confidence_bound`` is retained as a compatibility
    alias for older receipt readers. New receipts must provide the paired
    realized-win delta lower confidence bound.
    """
    weight = float(state.weight)
    if not 0.0 <= weight <= MAXIMUM_AUXILIARY_WEIGHT:
        raise ValueError("guide weight must be in [0, 0.50]")
    if realized_win_rate_delta_lower_confidence_bound is None:
        realized_win_rate_delta_lower_confidence_bound = (
            agreement_lift_lower_confidence_bound
        )
    elif agreement_lift_lower_confidence_bound is not None:
        raise ValueError("provide exactly one guide contribution confidence bound")
    if realized_win_rate_delta_lower_confidence_bound is None:
        raise ValueError("paired realized-win confidence bound is required")
    if realized_win_rate_delta_lower_confidence_bound > 0.0:
        next_weight = next(
            (step for step in POSITIVE_RAMP_SCHEDULE if step > weight),
            weight,
        )
        return GuideWeightState(
            weight=next_weight,
            consecutive_nonpositive_evaluations=0,
        )
    count = int(state.consecutive_nonpositive_evaluations) + 1
    if count < 2 or weight == 0.0:
        return GuideWeightState(
            weight=weight,
            consecutive_nonpositive_evaluations=count,
        )
    decayed = next((step for step in DECAY_SCHEDULE if step < weight), 0.0)
    return GuideWeightState(
        weight=decayed,
        consecutive_nonpositive_evaluations=0,
    )


__all__ = [
    "DECAY_SCHEDULE",
    "GuideWeightState",
    "MAXIMUM_AUXILIARY_WEIGHT",
    "POSITIVE_RAMP_SCHEDULE",
    "bootstrap_weight",
    "update_after_evaluation",
]
