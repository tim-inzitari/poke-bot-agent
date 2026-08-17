from __future__ import annotations

import math

import pytest

from poke_bot.r252_search_leaf_boundary import (
    INTERNAL_ORDERED_ACTION_EXPANSION_CEILING,
    R252SearchLeafBoundaryError,
    classify_search_leaf,
)


def _leaf(*, options: int, minimum: int, context: int = 1) -> dict:
    return {
        "select": {
            "context": context,
            "option": [{} for _ in range(options)],
            "minCount": minimum,
            "maxCount": minimum,
        }
    }


def test_astronomical_internal_ordered_space_is_value_only_boundary():
    count = math.factorial(25) // math.factorial(25 - 22)
    row = classify_search_leaf(
        _leaf(options=25, minimum=22), ordered_action_count=count
    )
    assert count == 2_585_201_673_888_497_664_000_000
    assert row.is_boundary is True
    assert row.reason == "deterministic_internal_fanout_over_64"
    assert row.representative_action == tuple(range(22))


def test_every_explicit_chance_context_is_a_pre_random_boundary():
    row = classify_search_leaf(
        _leaf(options=2, minimum=1, context=46), ordered_action_count=2
    )
    assert row.is_boundary is True
    assert row.reason == "explicit_chance_pre_random"
    assert row.representative_action == (0,)


def test_deterministic_internal_fanout_over_64_is_boundary():
    row = classify_search_leaf(
        _leaf(options=8, minimum=5), ordered_action_count=6_720
    )
    assert INTERNAL_ORDERED_ACTION_EXPANSION_CEILING == 64
    assert row.is_boundary is True
    assert row.reason == "deterministic_internal_fanout_over_64"


def test_deterministic_internal_fanout_at_64_remains_expandable():
    row = classify_search_leaf(
        _leaf(options=8, minimum=2), ordered_action_count=64
    )
    assert row.is_boundary is False
    assert row.reason is None
    assert row.representative_action is None


def test_invalid_internal_contract_fails_closed():
    with pytest.raises(R252SearchLeafBoundaryError, match="no selection"):
        classify_search_leaf({}, ordered_action_count=1)
    with pytest.raises(R252SearchLeafBoundaryError, match="must be positive"):
        classify_search_leaf(_leaf(options=1, minimum=1), ordered_action_count=0)
