"""Bound BO serial-MCTS expansion before random or oversized internal states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


INTERNAL_ORDERED_ACTION_EXPANSION_CEILING = 64
CHANCE_SELECTION_CONTEXT = 46


class R252SearchLeafBoundaryError(RuntimeError):
    """An internal simulator selection cannot be classified safely."""


@dataclass(frozen=True)
class R252SearchLeafBoundary:
    is_boundary: bool
    reason: str | None
    ordered_action_count: int
    representative_action: tuple[int, ...] | None


def classify_search_leaf(
    raw: Mapping[str, Any],
    *,
    ordered_action_count: int,
    expansion_ceiling: int = INTERNAL_ORDERED_ACTION_EXPANSION_CEILING,
) -> R252SearchLeafBoundary:
    """Classify an internal state without weakening real-root completeness.

    Every explicit chance context is a pre-random value boundary because the
    stock interface does not provide a trustworthy probability distribution
    for all random transitions.  A deterministic internal choice with more
    than ``expansion_ceiling`` ordered outcomes is also value-only.  The one
    representative selection exists solely to encode the state for the frozen
    value head; it is never installed as a child or given action authority.
    """

    count = int(ordered_action_count)
    ceiling = int(expansion_ceiling)
    if count < 1:
        raise R252SearchLeafBoundaryError(
            "internal ordered action count must be positive"
        )
    if ceiling < 1:
        raise R252SearchLeafBoundaryError(
            "internal expansion ceiling must be positive"
        )
    selection = raw.get("select")
    if not isinstance(selection, Mapping):
        raise R252SearchLeafBoundaryError("internal simulator leaf has no selection")
    context = selection.get("context")
    is_chance = context == CHANCE_SELECTION_CONTEXT
    is_oversized = count > ceiling
    if not is_chance and not is_oversized:
        return R252SearchLeafBoundary(False, None, count, None)

    options = selection.get("option")
    if not isinstance(options, list):
        raise R252SearchLeafBoundaryError(
            "internal selection options are not a list"
        )
    option_count = len(options)
    minimum = max(0, min(int(selection.get("minCount", 0)), option_count))
    representative = tuple(range(minimum))
    reason = (
        "explicit_chance_pre_random"
        if is_chance
        else "deterministic_internal_fanout_over_64"
    )
    return R252SearchLeafBoundary(True, reason, count, representative)


__all__ = [
    "CHANCE_SELECTION_CONTEXT",
    "INTERNAL_ORDERED_ACTION_EXPANSION_CEILING",
    "R252SearchLeafBoundary",
    "R252SearchLeafBoundaryError",
    "classify_search_leaf",
]
