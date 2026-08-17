"""Shared masked-observation helpers for combo-state target builders.

Used by Slowking and Slop Box so expert / self-play attach paths share one
card-resolution and zone-walk implementation. Only the acting player's masked
observation, current legal options, and recorded action are read.
"""

from __future__ import annotations

from typing import Any

from . import features

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12
OPTION_PLAY = 7


def context(observation: dict[str, Any]) -> int:
    raw = (observation.get("select") or {}).get("context", -1)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw) if -1 <= int(raw) <= 48 else -1
    try:
        return int(features._validated_context(raw))
    except (features.FeatureContractError, TypeError, ValueError, OverflowError):
        return -1


def card_id(value: Any) -> int | None:
    if isinstance(value, dict):
        raw = value.get("id")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return int(raw)
        for key in ("card", "source", "pokemon", "energy"):
            found = card_id(value.get(key))
            if found is not None:
                return found
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def all_card_ids(value: Any) -> list[int]:
    result: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            found = card_id(item)
            if found is not None:
                result.append(found)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return result


def effect_card(observation: dict[str, Any]) -> int | None:
    select = observation.get("select") or {}
    for key in ("effect", "source", "card"):
        value = select.get(key)
        found = card_id(value)
        if found is not None:
            return found
    return None


def option_card(observation: dict[str, Any], option: Any) -> Any | None:
    """Resolve one legal option from the actor's masked public state."""

    if not isinstance(option, dict):
        return None
    direct = card_id(option)
    if direct is not None:
        return option
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    actor = current.get("yourIndex")
    declared_actor = option.get("playerIndex")
    if (
        isinstance(actor, bool)
        or not isinstance(actor, int)
        or actor not in (0, 1)
        or (
            declared_actor is not None
            and (
                isinstance(declared_actor, bool)
                or not isinstance(declared_actor, int)
                or declared_actor != actor
            )
        )
    ):
        return None
    index = option.get("index")
    area = option.get("area")
    if option.get("type") == OPTION_PLAY:
        area = AREA_HAND
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or isinstance(area, bool)
        or not isinstance(area, int)
    ):
        return None
    players = current.get("players") or []
    if area == AREA_DECK:
        zone = select.get("deck")
    elif area == AREA_STADIUM:
        zone = current.get("stadium")
    elif area == AREA_LOOKING:
        zone = current.get("looking")
    elif len(players) == 2 and isinstance(players[actor], dict):
        zone = players[actor].get(
            {
                AREA_HAND: "hand",
                AREA_DISCARD: "discard",
                AREA_ACTIVE: "active",
                AREA_BENCH: "bench",
                AREA_PRIZE: "prize",
            }.get(area, "")
        )
    else:
        zone = None
    # Active may appear as a lone object in some traces; normalize to a list.
    if isinstance(zone, dict):
        zone = [zone]
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    return zone[index]


def chosen_card_ids(observation: dict[str, Any], action: list[int]) -> list[int]:
    options = list((observation.get("select") or {}).get("option") or [])
    chosen: list[int] = []
    for raw_index in action:
        index = int(raw_index)
        if not 0 <= index < len(options):
            return []
        found = card_id(option_card(observation, options[index]))
        if found is None:
            return []
        chosen.append(found)
    return chosen


def own_public_cards(observation: dict[str, Any]) -> list[int]:
    current = observation.get("current") or {}
    actor = int(current.get("yourIndex", -1))
    players = current.get("players") or []
    if actor not in (0, 1) or len(players) != 2:
        return []
    player = players[actor]
    return all_card_ids(
        {
            "hand": player.get("hand"),
            "active": player.get("active"),
            "bench": player.get("bench"),
            "discard": player.get("discard"),
        }
    )


def own_player(observation: dict[str, Any]) -> dict[str, Any]:
    current = observation.get("current") or {}
    actor = int(current.get("yourIndex", -1))
    players = current.get("players") or []
    if actor not in (0, 1) or len(players) != 2:
        return {}
    player = players[actor]
    return player if isinstance(player, dict) else {}
