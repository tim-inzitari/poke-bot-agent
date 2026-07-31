"""Causal replay targets for the exact Slowking combo/toolbox specialist.

Only the acting player's masked observation, current legal options, and
recorded action are read. No full pre-state, private prize cards, hidden deck
order, or future suffix is accepted by this API.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import features
from .combo_state_contract import (
    COMBO_STATE_KEY,
    COMBO_STATE_TARGET_SCHEMA,
    VECTOR_WIDTH,
    validate_combo_state_labels,
)

SLOWKING_DECK = Counter(
    {
        5: 4, 9: 2, 19: 4, 115: 2, 140: 1, 144: 2, 162: 4, 163: 4,
        183: 2, 184: 2, 224: 2, 756: 2, 1071: 1, 1092: 1, 1097: 3,
        1121: 4, 1146: 3, 1152: 4, 1168: 1, 1188: 4, 1227: 4, 1248: 4,
    }
)

ACADEMY_AT_NIGHT = 1248
CIPHERMANIAC = 1188
SLOWKING = 163
SLOWPOKE = 162
TELEPATH_ENERGY = 19
BOOMERANG_ENERGY = 9
PSYCHIC_ENERGY = 5
WONDROUS_PATCH = 1146
NIGHT_STRETCHER = 1097
POKE_PAD = 1152
ULTRA_BALL = 1121
SECRET_BOX = 1092

PAYLOAD_CLASS = {115: 0, 144: 1, 224: 2}
TOP_SETTERS = {ACADEMY_AT_NIGHT, CIPHERMANIAC}
RECOVERY = {NIGHT_STRETCHER}
SEARCH = {POKE_PAD, ULTRA_BALL, SECRET_BOX}
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12
OPTION_PLAY = 7


def is_exact_slowking_deck(deck: list[int]) -> bool:
    return len(deck) == 60 and Counter(int(card) for card in deck) == SLOWKING_DECK


def _context(observation: dict[str, Any]) -> int:
    raw = (observation.get("select") or {}).get("context", -1)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw) if -1 <= int(raw) <= 48 else -1
    try:
        return int(
            features._validated_context(raw)
        )
    except (features.FeatureContractError, TypeError, ValueError, OverflowError):
        return -1


def _card_id(value: Any) -> int | None:
    if isinstance(value, dict):
        raw = value.get("id")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return int(raw)
        for key in ("card", "source", "pokemon", "energy"):
            found = _card_id(value.get(key))
            if found is not None:
                return found
    return None


def _all_card_ids(value: Any) -> list[int]:
    result: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            card = _card_id(item)
            if card is not None:
                result.append(card)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return result


def _effect_card(observation: dict[str, Any]) -> int | None:
    select = observation.get("select") or {}
    for key in ("effect", "source", "card"):
        value = select.get(key)
        found = (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else _card_id(value)
        )
        if found is not None:
            return found
    return None


def _option_card(
    observation: dict[str, Any],
    option: Any,
) -> Any | None:
    """Resolve one legal option from the actor's masked public state."""

    if not isinstance(option, dict):
        return None
    direct = _card_id(option)
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
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    return zone[index]


def _chosen_card_ids(observation: dict[str, Any], action: list[int]) -> list[int]:
    options = list((observation.get("select") or {}).get("option") or [])
    chosen: list[int] = []
    for raw_index in action:
        index = int(raw_index)
        if not 0 <= index < len(options):
            return []
        card = _card_id(_option_card(observation, options[index]))
        if card is None:
            return []
        chosen.append(card)
    return chosen


def _own_public_cards(observation: dict[str, Any]) -> list[int]:
    current = observation.get("current") or {}
    actor = int(current.get("yourIndex", -1))
    players = current.get("players") or []
    if actor not in (0, 1) or len(players) != 2:
        return []
    player = players[actor]
    return _all_card_ids(
        {
            "hand": player.get("hand"),
            "active": player.get("active"),
            "bench": player.get("bench"),
            "discard": player.get("discard"),
        }
    )


def _visible_vector(observation: dict[str, Any]) -> tuple[list[float | None], list[bool]]:
    cards = set(_own_public_cards(observation))
    select_cards = {
        card
        for card in (
            _card_id(_option_card(observation, option))
            for option in (
                (observation.get("select") or {}).get("option") or []
            )
        )
        if card is not None
    }
    context = _context(observation)
    # Six copied-attack legality channels: the three payload attackers, any
    # other exposed copied attack, a target stage, and any legal copied option.
    copied = [
        float(card in select_cards) if context == 35 else None
        for card in (115, 144, 224)
    ]
    copied += [
        float(bool(select_cards - set(PAYLOAD_CLASS))) if context == 35 else None,
        float(context == 15),
        float(bool(select_cards)) if context == 35 else None,
    ]
    pieces = [
        float(SLOWPOKE in cards),
        float(SLOWKING in cards),
        float(bool(cards & TOP_SETTERS)),
        float(bool(cards & RECOVERY)),
        float(bool(cards & SEARCH)),
    ]
    energy = [
        float(PSYCHIC_ENERGY in cards),
        float(TELEPATH_ENERGY in cards),
        float(BOOMERANG_ENERGY in cards),
        float(WONDROUS_PATCH in cards),
        float(SLOWKING in cards and bool(cards & {PSYCHIC_ENERGY, TELEPATH_ENERGY, BOOMERANG_ENERGY})),
    ]
    current = observation.get("current") or {}
    actor = int(current.get("yourIndex", -1))
    players = current.get("players") or []
    player = players[actor] if actor in (0, 1) and len(players) == 2 else {}
    active_ids = set(_all_card_ids(player.get("active")))
    bench_ids = set(_all_card_ids(player.get("bench")))
    continuity = [
        float(SLOWPOKE in active_ids),
        float(SLOWPOKE in bench_ids),
        float(SLOWKING in active_ids),
        float(SLOWKING in bench_ids),
    ]
    values = copied + pieces + energy + continuity
    if len(values) != VECTOR_WIDTH:
        raise AssertionError("Slowking combo vector width drift")
    return (
        [None if value is None else float(value) for value in values],
        [value is not None for value in values],
    )


def build_combo_state_target(
    observation: dict[str, Any],
    action: list[int],
) -> dict[str, Any]:
    """Build one decision target without accepting hidden/future state."""

    context = _context(observation)
    effect = _effect_card(observation)
    chosen = _chosen_card_ids(observation, action)
    top_target: int | None = None
    if context == 9 and effect in TOP_SETTERS and chosen:
        # Academy selects one exact top card. Cipher exposes an ordered pair;
        # the last demonstrated selection is the resulting top card.
        top_target = PAYLOAD_CLASS.get(chosen[-1], 3 if chosen[-1] in {140, 183, 184, 756, 1071, 162, 163} else 4)
    seek_target: int | None = None
    if context == 9 and effect == ACADEMY_AT_NIGHT:
        seek_target = 0
    elif context == 9 and effect == CIPHERMANIAC:
        seek_target = 1
    elif context == 35:
        seek_target = 2
    elif effect == TELEPATH_ENERGY:
        seek_target = 3
    elif effect == WONDROUS_PATCH:
        seek_target = 4
    elif effect in RECOVERY:
        seek_target = 5
    elif effect is not None and (
        effect == SLOWKING or effect in SEARCH or context == 15
    ):
        seek_target = 6
    vector, vector_mask = _visible_vector(observation)
    target = {
        "schema": COMBO_STATE_TARGET_SCHEMA,
        "top_deck": {
            "target": top_target,
            "mask": top_target is not None,
        },
        "seek_source": {
            "target": seek_target,
            "mask": seek_target is not None,
        },
        "vector": {"target": vector, "mask": vector_mask},
    }
    validate_combo_state_labels(target)
    return target


def attach_slowking_combo_state_labels(
    steps: list[dict[str, Any]],
    *,
    deck: list[int],
) -> dict[str, int]:
    """Attach strict labels to an exact Slowking record in place."""

    if not is_exact_slowking_deck(deck):
        raise ValueError("combo-state labels require the exact Slowking deck")
    coverage = {"decisions": 0, "top_deck": 0, "seek_source": 0, "vector_cells": 0}
    for step in steps:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("Slowking combo target is missing masked observation")
        target = build_combo_state_target(
            observation, [int(value) for value in (step.get("action") or [])]
        )
        clean = validate_combo_state_labels(target)
        aux = dict(step.get("aux_labels") or {})
        aux[COMBO_STATE_KEY] = target
        step["aux_labels"] = aux
        coverage["decisions"] += 1
        coverage["top_deck"] += int(clean["top_deck_mask"])
        coverage["seek_source"] += int(clean["seek_source_mask"])
        coverage["vector_cells"] += sum(clean["vector_mask"])
    return coverage
