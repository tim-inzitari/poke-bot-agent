"""Sparse, causal training teacher for Hammer Dragapult.

The human guide is intentionally much broader than this module.  The teacher
scores only current, fully resolved legal stages: opening Budew/Dreepy choices,
Buddy-Buddy Poffin searches, visible Dragapult-line evolutions, Recon
Directive, and playing Crushing Hammer when an opposing Pokémon visibly has
Energy attached.  It never labels the coin flip, future draw, later damage
placement, prize outcome, or a hidden-card inference.

Every returned score list covers the complete supplied legal-combo stage.
Malformed or ambiguous stages, unresolved cards, unsupported prompts, wrong
decks, and low-margin stages are masked with ``None``.  This module supplies
training targets only; it does not select or execute runtime actions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from . import archetypes


GUIDE_VERSION = "hammer-pult-north-star-v1"

FIRE_ENERGY = 2
PSYCHIC_ENERGY = 5
DARKNESS_ENERGY = 7

MUNKIDORI = 112
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
FEZANDIPITI_EX = 140
BUDEW = 235

UNFAIR_STAMP = 1080
BUDDY_BUDDY_POFFIN = 1086
CRUSHING_HAMMER = 1120

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPT_CARD = 3
OPT_PLAY = 7
OPT_EVOLVE = 9
OPT_ABILITY = 10

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_TO_BENCH = 5

CORE_SIGNATURE_MINIMUMS = {
    DREEPY: 4,
    DRAKLOAK: 4,
    DRAGAPULT_EX: 3,
    MUNKIDORI: 1,
    BUDEW: 1,
    CRUSHING_HAMMER: 3,
    UNFAIR_STAMP: 1,
}
ABSTENTION_MARGIN = 0.25


def enabled() -> bool:
    """Return whether the generic guide registry selected this teacher."""
    from . import deck_guides

    return deck_guides.enabled() and deck_guides.selected_id() == "hammer-pult"


def is_hammer_pult_deck(deck: Iterable[int]) -> bool:
    """Require the exact 60-card family signature used by the guide contract."""
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
    )


def applies(deck_card_ids: Iterable[int]) -> bool:
    """Compatibility alias for the historical Hammer-Pult hook."""
    return is_hammer_pult_deck(deck_card_ids) and archetypes.is_hammer_signature(
        deck_card_ids
    )


def prior_logit_bias(
    obs: Any,
    action_combos: Sequence[Sequence[int]],
    *,
    scale: float = 1.0,
) -> list[float]:
    """Preserve the legacy runtime API as an exact neutral bypass."""
    del obs, scale
    return [0.0] * len(action_combos)


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


def _resolve_card(
    obs: dict[str, Any],
    *,
    area: Any,
    index: Any,
    player_index: Any = None,
) -> Optional[Any]:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    players = current.get("players") or []
    area_i = _exact_int(area)
    index_i = _exact_int(index)
    if area_i is None or index_i is None or index_i < 0:
        return None
    if area_i == AREA_DECK:
        zone = select.get("deck") or []
    elif area_i == AREA_STADIUM:
        zone = current.get("stadium") or []
    elif area_i == AREA_LOOKING:
        zone = current.get("looking") or []
    else:
        seat = _exact_int(
            current.get("yourIndex", 0)
            if player_index is None
            else player_index
        )
        if seat not in (0, 1):
            return None
        try:
            player = players[seat]
        except (IndexError, TypeError):
            return None
        if not isinstance(player, dict):
            return None
        key = {
            AREA_HAND: "hand",
            AREA_DISCARD: "discard",
            AREA_ACTIVE: "active",
            AREA_BENCH: "bench",
            AREA_PRIZE: "prize",
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
    option_type = _exact_int(option.get("type"))
    if option_type is None:
        return None
    if option_type == OPT_PLAY:
        return _resolve_card(obs, area=AREA_HAND, index=option.get("index"))
    return _resolve_card(
        obs,
        area=option.get("area"),
        index=option.get("index"),
        player_index=seat,
    )


def _effect_id(select: dict[str, Any]) -> Optional[int]:
    for key in ("effect", "contextCard"):
        value = _card_id(select.get(key))
        if value is not None:
            return value
    return None


def _opponent_has_visible_energy(opponent: dict) -> bool:
    for pokemon in _cards(opponent.get("active")) + _cards(opponent.get("bench")):
        if not isinstance(pokemon, dict):
            return False
        energy = pokemon.get("energyCards")
        if not isinstance(energy, (list, tuple)):
            return False
        if energy:
            return True
    return False


def _setup_score(
    card_id: int,
    *,
    context: int,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    if context == CTX_SETUP_ACTIVE:
        if card_id == BUDEW:
            return 3.0
        if card_id == DREEPY:
            return 2.0
        if card_id == MUNKIDORI:
            return 0.6
        return 0.0
    if card_id == DREEPY:
        count = board[DREEPY] + chosen[DREEPY]
        return 3.2 if count < 2 else (1.3 if count < 3 else 0.0)
    if card_id == BUDEW and board[BUDEW] + chosen[BUDEW] == 0:
        return 1.8
    return 0.0


def _main_score(
    obs: dict[str, Any],
    option: dict[str, Any],
    *,
    opponent: dict,
) -> Optional[float]:
    option_type = _exact_int(option.get("type"))
    if option_type is None:
        return None
    if option_type not in {OPT_PLAY, OPT_EVOLVE, OPT_ABILITY}:
        return 0.0
    card_id = _card_id(_option_card(obs, option))
    if card_id is None:
        return None
    if option_type == OPT_EVOLVE:
        return {DRAKLOAK: 2.8, DRAGAPULT_EX: 3.1}.get(card_id, 0.0)
    if option_type == OPT_ABILITY:
        return 2.6 if card_id == DRAKLOAK else 0.0
    if card_id == CRUSHING_HAMMER:
        return 1.7 if _opponent_has_visible_energy(opponent) else 0.0
    if card_id == BUDDY_BUDDY_POFFIN:
        return 1.4
    return 0.0


def _combo_score(
    obs: dict[str, Any],
    combo: Sequence[int],
    *,
    me: dict,
    opponent: dict,
) -> Optional[float]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    context = _exact_int(select.get("context"))
    if context is None:
        return None
    effect = _effect_id(select)
    if context not in {CTX_MAIN, CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
        if not (context == CTX_TO_BENCH and effect == BUDDY_BUDDY_POFFIN):
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
            score = _main_score(obs, option, opponent=opponent)
            if score is None:
                return None
            total += score
            continue
        card_id = _card_id(_option_card(obs, option))
        if card_id is None:
            return None
        setup_context = (
            CTX_SETUP_BENCH
            if effect == BUDDY_BUDDY_POFFIN
            else context
        )
        total += _setup_score(
            card_id,
            context=setup_context,
            board=board,
            chosen=chosen,
        )
        chosen[card_id] += 1
    return max(-8.0, min(8.0, float(total)))


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


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Score a complete legal stage or mask the complete stage."""
    if (
        (not force_enabled and not enabled())
        or not is_hammer_pult_deck(deck)
        or not isinstance(obs, dict)
        or not action_combos
    ):
        return None
    me, opponent = _players(obs)
    if (
        me is None
        or opponent is None
        or not _validated_stage(obs, action_combos)
    ):
        return None
    scores: list[float] = []
    for combo in action_combos:
        score = _combo_score(obs, combo, me=me, opponent=opponent)
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
        f"HammerGuide(version={GUIDE_VERSION}, "
        f"signature={sorted(CORE_SIGNATURE_MINIMUMS)})"
    )


__all__ = [
    "GUIDE_VERSION",
    "applies",
    "describe",
    "enabled",
    "guide_scores",
    "is_hammer_pult_deck",
    "prior_logit_bias",
]
