from __future__ import annotations

import pytest

from poke_bot.tactical_sequence_planner import legal_action_order_fingerprint
from poke_bot.tactical_sequence_supervision import (
    TacticalSequenceSupervisionError,
    tactical_sequence_option_targets,
)


def _receipt(status: str, *, proposed=(1,)) -> dict:
    actions = ((0,), (1,))
    return {
        "schema": "poke_bot.tactical_sequence_shadow/v1",
        "mode": "shadow_only",
        "status": status,
        "root_legal_order_fingerprint": legal_action_order_fingerprint(actions),
        "direct_action": [0],
        "proposed_action": list(proposed),
        "dispatch_authorized": False,
        "tactical_outcome_head_is_proof": False,
        "failure": None,
        "boundary_counts": {},
        "proof": {"terminal_winner": 0},
    }


def test_terminal_shadow_proof_labels_only_the_certified_option() -> None:
    target = tactical_sequence_option_targets(
        _receipt("proven_exact_terminal_win_shadow"),
        legal_actions=((0,), (1,)),
    )
    assert target["label"] == "exact_terminal_win"
    assert target["rows"][0]["mask"] == [False, False, False, False]
    assert target["rows"][1] == {
        "values": [0.0, 1.0, 0.0, 0.0],
        "mask": [True, True, True, True],
    }


def test_deadline_remains_fully_masked() -> None:
    receipt = _receipt("deadline")
    receipt["boundary_counts"] = {"depth_cap": 5}
    target = tactical_sequence_option_targets(
        receipt,
        legal_actions=((0,), (1,)),
    )
    assert target["label"] is None
    assert all(not any(row["mask"]) for row in target["rows"])


def test_dispatch_authority_or_legal_reordering_fails_closed() -> None:
    receipt = _receipt("no_goal_found")
    receipt["dispatch_authorized"] = True
    with pytest.raises(TacticalSequenceSupervisionError, match="dispatch"):
        tactical_sequence_option_targets(receipt, legal_actions=((0,), (1,)))

    receipt["dispatch_authorized"] = False
    with pytest.raises(TacticalSequenceSupervisionError, match="order"):
        tactical_sequence_option_targets(receipt, legal_actions=((1,), (0,)))
