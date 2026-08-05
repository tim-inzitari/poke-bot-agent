"""Deployment observation vs privileged training targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .reasons import ReasonCode

PRIVILEGED_OPPONENT_KEYS: frozenset[str] = frozenset(
    {
        "hand",
        "deck",
        "deckOrder",
        "deck_order",
        "prizeOrder",
        "prize_cards",
        "hiddenPrize",
        "truePrize",
        "privateState",
    }
)


@dataclass(frozen=True)
class DeploymentObservation:
    """Acting-player-visible observation for deployment planning.

    Never contains opponent private fields. ``raw_obs`` is the competition
    dict after visibility scrubbing.
    """

    schema_version: str
    seat: int
    turn: int
    raw_obs: dict[str, Any]
    visibility_mask: tuple[str, ...] = ()
    observation_hash: str = ""
    matchup_route: int = -1
    deck: tuple[int, ...] = ()

    def turn_key(self) -> tuple[int, int]:
        return (int(self.seat), int(self.turn))


@dataclass(frozen=True)
class PrivilegedTargets:
    """Training-only labels. Must never be attached to DeploymentObservation."""

    schema_version: str = "poke_bot.poke_rlm.privileged_targets/v1"
    opponent_hand: Optional[tuple[int, ...]] = None
    opponent_deck_order: Optional[tuple[int, ...]] = None
    teacher_plan_json: Optional[dict[str, Any]] = None
    counterfactual_values: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def observation_hash(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def reject_hidden_fields(obs: Mapping[str, Any]) -> ReasonCode:
    current = obs.get("current")
    if not isinstance(current, dict):
        return ReasonCode.OK
    players = current.get("players")
    if not isinstance(players, list) or len(players) < 2:
        return ReasonCode.OK
    try:
        seat = int(current.get("yourIndex", 0))
    except (TypeError, ValueError):
        seat = 0
    opp = players[1 - seat] if seat in (0, 1) else None
    if not isinstance(opp, dict):
        return ReasonCode.OK
    for key in PRIVILEGED_OPPONENT_KEYS:
        if key == "hand":
            if opp.get("hand") is not None:
                return ReasonCode.HIDDEN_FIELD_REJECTED
            continue
        if key in opp and opp.get(key) is not None:
            return ReasonCode.HIDDEN_FIELD_REJECTED
    return ReasonCode.OK


def scrub_privileged_fields(obs: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied obs safe for deployment typing."""
    import copy

    out = copy.deepcopy(obs)
    current = out.get("current")
    if not isinstance(current, dict):
        return out
    players = current.get("players")
    if not isinstance(players, list) or len(players) < 2:
        return out
    try:
        seat = int(current.get("yourIndex", 0))
    except (TypeError, ValueError):
        return out
    opp = players[1 - seat]
    if not isinstance(opp, dict):
        return out
    opp["hand"] = None
    for key in PRIVILEGED_OPPONENT_KEYS:
        if key == "hand":
            continue
        opp.pop(key, None)
    return out


def build_deployment_observation(
    obs_dict: dict[str, Any],
    *,
    deck: Optional[list[int]] = None,
    matchup_route: int = -1,
    schema_version: str = "poke_bot.poke_rlm.deployment_observation/v1",
) -> tuple[Optional[DeploymentObservation], ReasonCode]:
    reason = reject_hidden_fields(obs_dict)
    if reason is not ReasonCode.OK:
        return None, reason
    scrubbed = scrub_privileged_fields(obs_dict)
    current = scrubbed.get("current") if isinstance(scrubbed, dict) else None
    seat = -1
    turn = -1
    if isinstance(current, dict):
        try:
            seat = int(current.get("yourIndex", -1))
        except (TypeError, ValueError):
            seat = -1
        try:
            turn = int(current.get("turn", -1))
        except (TypeError, ValueError):
            turn = -1
    dep = DeploymentObservation(
        schema_version=schema_version,
        seat=seat,
        turn=turn,
        raw_obs=scrubbed,
        visibility_mask=("public", "own_private", "belief_from_public"),
        observation_hash=observation_hash(
            {
                "seat": seat,
                "turn": turn,
                "select": scrubbed.get("select"),
                "current_turn": turn,
            }
        ),
        matchup_route=int(matchup_route),
        deck=tuple(int(x) for x in (deck or ())),
    )
    return dep, ReasonCode.OK
