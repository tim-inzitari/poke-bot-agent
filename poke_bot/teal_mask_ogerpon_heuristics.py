"""Sparse causal guide teacher for Teal Mask Ogerpon ex / Slop Box.

Slop Box is the public archetype-151 Raging Bolt Ogerpon family built around
Mega Kangaskhan ex, Meowth ex, Teal Mask Ogerpon ex, Raging Bolt ex, Area Zero,
Crispin, Energy Switch, Glass Trumpet, and five Basic Energy types. The learned
17-input fused policy owns attack, Energy routing, target, prize-map, recovery,
and matchup decisions. This auxiliary teacher labels only complete current
legal stages whose exact public inputs support a safe preference: opening
Pokemon placement and currently legal, resolved draw/acceleration abilities.

Coin outcomes, future draws, face-down Prize identities, hidden opponent
information, speculative Energy Switch routes, attack targets, and future
bench-collapse plans are always masked. This module never selects or executes
runtime actions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from . import archetypes


GUIDE_VERSION = "teal-mask-ogerpon-ex-slop-box-north-star-v3"

RAGING_BOLT_EX = 63
TEAL_MASK_OGERPON_EX = 96
WELLSPRING_MASK_OGERPON_EX = 108
MUNKIDORI = 112
FEZANDIPITI_EX = 140
LATIAS_EX = 184
PECHARUNT = 230
LILLIES_CLEFAIRY_EX = 272
MEGA_KANGASKHAN_EX = 756
MEOWTH_EX = 1071

AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5

OPT_PLAY = 7
OPT_ABILITY = 10

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2

CORE_SIGNATURE_MINIMUMS = {
    RAGING_BOLT_EX: 2,
    TEAL_MASK_OGERPON_EX: 3,
    WELLSPRING_MASK_OGERPON_EX: 1,
    LILLIES_CLEFAIRY_EX: 1,
    MEGA_KANGASKHAN_EX: 3,
    MEOWTH_EX: 3,
    archetypes.ENERGY_SWITCH: 4,
    archetypes.AREA_ZERO_UNDERDEPTHS: 4,
}
ABSTENTION_MARGIN = 0.25


def enabled() -> bool:
    """Return whether the generic registry selected this guide."""
    from . import deck_guides

    return (
        deck_guides.enabled()
        and deck_guides.selected_id() == "teal-mask-ogerpon-ex"
    )


def _exact_int(value: Any) -> Optional[int]:
    if isinstance(value, (bool, float, str)):
        return None
    try:
        result = int(value)
        return result if value == result else None
    except (OverflowError, TypeError, ValueError):
        return None


def _card_id(card: Any) -> Optional[int]:
    value = card.get("id") if isinstance(card, dict) else getattr(card, "id", None)
    return _exact_int(value)


def _cards(zone: Any) -> list[Any]:
    return list(zone) if isinstance(zone, (list, tuple)) else []


def is_teal_mask_ogerpon_ex_deck(deck: Iterable[int]) -> bool:
    """Require the exact 60-card Slop Box family signature."""
    try:
        card_ids = [_exact_int(card_id) for card_id in deck]
    except TypeError:
        return False
    if len(card_ids) != 60 or any(card_id is None for card_id in card_ids):
        return False
    counts = Counter(card_ids)
    return all(
        counts[card_id] >= minimum
        for card_id, minimum in CORE_SIGNATURE_MINIMUMS.items()
    ) and archetypes.is_teal_mask_ogerpon_box_signature(card_ids)


def applies(deck_card_ids: Iterable[int]) -> bool:
    """Compatibility predicate for guide dispatch and corpus featurization."""
    return is_teal_mask_ogerpon_ex_deck(deck_card_ids)


def prior_logit_bias(
    obs: Any,
    action_combos: Sequence[Sequence[int]],
    *,
    scale: float = 1.0,
) -> list[float]:
    """Preserve the serving API as an exact neutral bypass."""
    del obs, scale
    return [0.0] * len(action_combos)


def _players(obs: dict[str, Any]) -> tuple[Optional[dict], Optional[dict]]:
    current = obs.get("current") if isinstance(obs, dict) else None
    players = current.get("players") if isinstance(current, dict) else None
    seat = _exact_int(current.get("yourIndex", 0)) if current else None
    if seat not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        return None, None
    if not isinstance(players[seat], dict) or not isinstance(players[1 - seat], dict):
        return None, None
    return players[seat], players[1 - seat]


def _zone_ids(player: dict, key: str) -> list[int]:
    return [
        card_id
        for card_id in (_card_id(card) for card in _cards(player.get(key)))
        if card_id is not None
    ]


def _board_counts(player: dict) -> Counter[int]:
    return Counter(_zone_ids(player, "active") + _zone_ids(player, "bench"))


def _resolve_own_card(
    obs: dict[str, Any],
    *,
    area: Any,
    index: Any,
) -> Optional[Any]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    seat = _exact_int(current.get("yourIndex", 0))
    area_i = _exact_int(area)
    index_i = _exact_int(index)
    if seat not in (0, 1) or index_i is None or index_i < 0:
        return None
    try:
        player = players[seat]
    except (IndexError, TypeError):
        return None
    if not isinstance(player, dict):
        return None
    key = {
        AREA_HAND: "hand",
        AREA_ACTIVE: "active",
        AREA_BENCH: "bench",
    }.get(area_i)
    if key is None:
        return None
    zone = player.get(key) or []
    try:
        return zone[index_i]
    except (IndexError, TypeError):
        return None


def _option_card(obs: dict[str, Any], option: dict[str, Any]) -> Optional[Any]:
    current = obs.get("current") or {}
    seat = _exact_int(current.get("yourIndex", 0))
    declared_seat = option.get("playerIndex")
    if seat not in (0, 1) or (
        declared_seat is not None and _exact_int(declared_seat) != seat
    ):
        return None
    if _exact_int(option.get("type")) == OPT_PLAY:
        return _resolve_own_card(
            obs,
            area=AREA_HAND,
            index=option.get("index"),
        )
    return _resolve_own_card(
        obs,
        area=option.get("area"),
        index=option.get("index"),
    )


def _setup_score(
    card_id: int,
    *,
    context: int,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    if context == CTX_SETUP_ACTIVE:
        return {
            MEGA_KANGASKHAN_EX: 3.0,
            TEAL_MASK_OGERPON_EX: 2.6,
            LILLIES_CLEFAIRY_EX: 2.0,
            WELLSPRING_MASK_OGERPON_EX: 1.5,
            PECHARUNT: 0.8,
        }.get(card_id, 0.0)
    if card_id == TEAL_MASK_OGERPON_EX:
        count = board[card_id] + chosen[card_id]
        return 3.2 if count < 2 else (1.4 if count < 3 else 0.0)
    if card_id == LATIAS_EX and board[card_id] + chosen[card_id] == 0:
        return 2.5
    if card_id == MEOWTH_EX and board[card_id] + chosen[card_id] == 0:
        return 2.2
    if card_id == MEGA_KANGASKHAN_EX and board[card_id] + chosen[card_id] == 0:
        return 1.8
    return 0.0


def _main_score(obs: dict[str, Any], option: dict[str, Any]) -> Optional[float]:
    option_type = _exact_int(option.get("type"))
    if option_type is None:
        return None
    if option_type != OPT_ABILITY:
        return 0.0
    card_id = _card_id(_option_card(obs, option))
    if card_id is None:
        return None
    return {
        TEAL_MASK_OGERPON_EX: 3.0,
        MEGA_KANGASKHAN_EX: 2.7,
        FEZANDIPITI_EX: 2.5,
        MUNKIDORI: 2.0,
    }.get(card_id, 0.0)


def _validated_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
) -> bool:
    select = obs.get("select")
    if not isinstance(select, dict):
        return False
    options = select.get("option")
    if not isinstance(options, list) or not all(
        isinstance(option, dict) for option in options
    ):
        return False
    minimum = _exact_int(select.get("minCount"))
    maximum = _exact_int(select.get("maxCount"))
    if (
        minimum is None
        or maximum is None
        or minimum < 0
        or maximum > len(options)
        or minimum > maximum
    ):
        return False
    for combo in action_combos:
        if not isinstance(combo, (list, tuple)):
            return False
        indices = [_exact_int(index) for index in combo]
        if (
            any(index is None for index in indices)
            or len(indices) < minimum
            or len(indices) > maximum
            or len(indices) != len(set(indices))
            or any(index < 0 or index >= len(options) for index in indices)
        ):
            return False
    return True


def _combo_score(
    obs: dict[str, Any],
    combo: Sequence[int],
    *,
    me: dict,
) -> Optional[float]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    context = _exact_int(select.get("context"))
    if context not in {CTX_MAIN, CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
        return None
    board = _board_counts(me)
    chosen: Counter[int] = Counter()
    total = 0.0
    for index in combo:
        index_i = _exact_int(index)
        if index_i is None:
            return None
        try:
            option = options[index_i]
        except (IndexError, TypeError):
            return None
        if not isinstance(option, dict):
            return None
        if context == CTX_MAIN:
            score = _main_score(obs, option)
            if score is None:
                return None
            total += score
            continue
        card_id = _card_id(_option_card(obs, option))
        if card_id is None:
            return None
        total += _setup_score(
            card_id,
            context=context,
            board=board,
            chosen=chosen,
        )
        chosen[card_id] += 1
    return max(-8.0, min(8.0, float(total)))


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Score one complete legal stage or mask the entire stage."""
    if (
        (not force_enabled and not enabled())
        or not is_teal_mask_ogerpon_ex_deck(deck)
        or not isinstance(obs, dict)
        or not action_combos
    ):
        return None
    me, opponent = _players(obs)
    if me is None or opponent is None or not _validated_stage(obs, action_combos):
        return None
    scores: list[float] = []
    for combo in action_combos:
        score = _combo_score(obs, combo, me=me)
        if score is None or not math.isfinite(score):
            return None
        scores.append(score)
    if len(scores) != len(action_combos):
        return None
    if max(scores) - min(scores) < ABSTENTION_MARGIN:
        return None
    return scores


def describe() -> str:
    return (
        f"TealMaskOgerponGuide(version={GUIDE_VERSION}, "
        f"signature={sorted(CORE_SIGNATURE_MINIMUMS)})"
    )


__all__ = [
    "GUIDE_VERSION",
    "applies",
    "describe",
    "enabled",
    "guide_scores",
    "is_teal_mask_ogerpon_ex_deck",
    "prior_logit_bias",
]
