"""Causal target-only labels for the r263 tactical-sequence outcome head.

The bounded planner remains shadow-only.  This module converts only its
auditable public-state receipt into masked option targets; it never exposes a
planner proposal or search score as a model input or direct action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .tactical_sequence_planner import (
    TACTICAL_SEQUENCE_RECEIPT_SCHEMA,
    legal_action_order_fingerprint,
)


TACTICAL_SEQUENCE_OUTCOME_TARGET_SCHEMA = (
    "poke_bot.tactical_sequence_outcome_targets/v1"
)
TACTICAL_SEQUENCE_OUTCOME_LABELS = (
    "no_proof",
    "exact_terminal_win",
    "public_sme_goal",
    "typed_boundary",
)
_TYPED_BOUNDARIES = frozenset(
    {
        "actor_change",
        "turn_change",
        "explicit_chance_pre_random",
        "information_reobservation",
        "deterministic_internal_fanout_over_64",
        "no_legal_action",
        "terminal_non_goal",
        "stochastic_transition",
        "depth_cap",
    }
)


class TacticalSequenceSupervisionError(ValueError):
    """A shadow receipt cannot safely become a training target."""


def _action(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TacticalSequenceSupervisionError(f"{label} must be an action")
    action = tuple(int(item) for item in value)
    if len(action) != len(set(action)) or any(item < 0 for item in action):
        raise TacticalSequenceSupervisionError(f"{label} is malformed")
    return action


def tactical_sequence_option_targets(
    receipt: Mapping[str, Any],
    *,
    legal_actions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Return one masked four-output row per exact root legal action.

    Exhaustive no-goal results supervise only the direct action as
    ``no_proof``. Proven terminal/public goals supervise only the first
    certified proposed action. Typed fail-closed boundaries supervise only the
    direct action as ``typed_boundary``. Deadline, node-cap, backend, policy,
    or malformed-search failures remain fully masked.
    """

    actions = tuple(_action(value, label="legal action") for value in legal_actions)
    if len(actions) != len(set(actions)):
        raise TacticalSequenceSupervisionError("legal actions are not unique")
    if receipt.get("schema") != TACTICAL_SEQUENCE_RECEIPT_SCHEMA:
        raise TacticalSequenceSupervisionError("shadow receipt schema mismatch")
    if receipt.get("mode") != "shadow_only":
        raise TacticalSequenceSupervisionError("planner receipt is not shadow-only")
    if receipt.get("dispatch_authorized") is not False:
        raise TacticalSequenceSupervisionError("planner receipt grants dispatch authority")
    if receipt.get("tactical_outcome_head_is_proof") is not False:
        raise TacticalSequenceSupervisionError("planner receipt treats a learned hint as proof")
    if receipt.get("root_legal_order_fingerprint") != legal_action_order_fingerprint(actions):
        raise TacticalSequenceSupervisionError("root legal-action order changed")

    direct = _action(receipt.get("direct_action"), label="direct action")
    if direct not in actions:
        raise TacticalSequenceSupervisionError("direct action is not root-legal")
    status = str(receipt.get("status") or "")
    label: str | None = None
    target_action: tuple[int, ...] | None = None
    if status == "proven_exact_terminal_win_shadow":
        label = "exact_terminal_win"
        target_action = _action(receipt.get("proposed_action"), label="proposed action")
        proof = receipt.get("proof")
        if not isinstance(proof, Mapping) or proof.get("terminal_winner") not in (0, 1):
            raise TacticalSequenceSupervisionError("terminal proof is incomplete")
    elif status == "public_goal_reached_shadow":
        label = "public_sme_goal"
        target_action = _action(receipt.get("proposed_action"), label="proposed action")
    else:
        boundary_counts = receipt.get("boundary_counts")
        typed_boundary_seen = bool(
            isinstance(boundary_counts, Mapping)
            and any(
                int(boundary_counts.get(name, 0) or 0) > 0
                for name in _TYPED_BOUNDARIES
            )
        )
        if (
            status not in {
                "deadline",
                "node_cap",
                "backend_fault",
                "invalid_policy_candidates",
            }
            and typed_boundary_seen
        ):
            label = "typed_boundary"
            target_action = direct
        elif status == "no_goal_found" and receipt.get("failure") is None:
            label = "no_proof"
            target_action = direct

    rows = [
        {"values": [0.0] * len(TACTICAL_SEQUENCE_OUTCOME_LABELS), "mask": [False] * len(TACTICAL_SEQUENCE_OUTCOME_LABELS)}
        for _ in actions
    ]
    if label is not None and target_action is not None:
        if target_action not in actions:
            raise TacticalSequenceSupervisionError("labeled action is not root-legal")
        option_index = actions.index(target_action)
        label_index = TACTICAL_SEQUENCE_OUTCOME_LABELS.index(label)
        rows[option_index]["values"][label_index] = 1.0
        rows[option_index]["mask"] = [True] * len(TACTICAL_SEQUENCE_OUTCOME_LABELS)

    return {
        "schema": TACTICAL_SEQUENCE_OUTCOME_TARGET_SCHEMA,
        "target_only": True,
        "model_input": False,
        "planner_dispatch_authority": False,
        "root_legal_order_fingerprint": receipt["root_legal_order_fingerprint"],
        "status": status,
        "label": label,
        "rows": rows,
    }


__all__ = [
    "TACTICAL_SEQUENCE_OUTCOME_LABELS",
    "TACTICAL_SEQUENCE_OUTCOME_TARGET_SCHEMA",
    "TacticalSequenceSupervisionError",
    "tactical_sequence_option_targets",
]
