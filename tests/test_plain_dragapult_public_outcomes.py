from __future__ import annotations

from collections import Counter

import pytest

from scripts.audit_plain_dragapult_public_outcomes import (
    OutcomeRow,
    _first_player,
    _group,
    _outcome,
    _record,
)


def test_first_player_requires_exact_public_integer() -> None:
    payload = {
        "steps": [
            [
                {"observation": {"current": {"firstPlayer": True}}},
                {"observation": {"current": {"firstPlayer": 1}}},
            ]
        ]
    }
    assert _first_player(payload) == 1
    assert _first_player({"steps": []}) is None


@pytest.mark.parametrize(
    ("reward", "expected"),
    ((1, "win"), (0, "draw"), (-1, "loss")),
)
def test_outcome_uses_only_completed_two_seat_rewards(
    reward: int,
    expected: str,
) -> None:
    payload = {
        "rewards": [reward, -reward],
        "statuses": ["DONE", "DONE"],
    }
    assert _outcome(payload, 0) == expected
    with pytest.raises(RuntimeError):
        _outcome({"rewards": [1, -1], "statuses": ["DONE", "ERROR"]}, 0)


def test_observational_summary_preserves_draws_and_unknown_groups() -> None:
    rows = [
        OutcomeRow("2026-07-01", "1", 0, "sha256:a", "first", "crustle", "x", "win"),
        OutcomeRow("2026-07-01", "2", 1, "sha256:b", "second", "crustle", "x", "draw"),
        OutcomeRow("2026-07-02", "3", 0, "sha256:c", "unknown", "unknown", "y", "loss"),
    ]
    assert _record(Counter(row.outcome for row in rows)) == {
        "games": 3,
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "win_rate": 1 / 3,
        "non_loss_rate": 2 / 3,
    }
    by_order = _group(rows, "play_order")
    assert by_order["first"]["wins"] == 1
    assert by_order["second"]["draws"] == 1
    assert by_order["unknown"]["losses"] == 1
