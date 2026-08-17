"""Sparse training teacher for Cynthia's Garchomp ex.

This module implements the causally safe subset of the source-cited Garchomp
guide contract: redundant Gible setup, visible evolution-chain development,
and exact Champion's Call choices. Damage-threshold, Tool, Energy, recovery,
and matchup advice remains masked unless the observation exposes every fact
needed to rank the complete legal stage.

The scorer is training-only. It returns one score for every supplied legal
combo or ``None`` for the whole stage; it never chooses a runtime action.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence


GUIDE_VERSION = "cynthias-garchomp-north-star-v1"

CYNTHIAS_GIBLE = 379
CYNTHIAS_GABITE = 380
CYNTHIAS_GARCHOMP_EX = 381
CYNTHIAS_ROSELIA = 341
CYNTHIAS_ROSERADE = 342
CYNTHIAS_SPIRITOMB = 387
CYNTHIAS_POWER_WEIGHT = 1173

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPT_PLAY = 7
OPT_EVOLVE = 9

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2

LINE = frozenset(
    {CYNTHIAS_GIBLE, CYNTHIAS_GABITE, CYNTHIAS_GARCHOMP_EX}
)
ABSTENTION_MARGIN = 0.25


def enabled() -> bool:
    from . import deck_guides

    return (
        deck_guides.enabled()
        and deck_guides.selected_id() == "garchomp"
    )


def is_garchomp_deck(deck: Iterable[int]) -> bool:
    try:
        counts = Counter(int(card_id) for card_id in deck)
    except (TypeError, ValueError):
        return False
    return all(counts[card_id] >= 2 for card_id in LINE)


def _card_id(card: Any) -> Optional[int]:
    value = card.get("id") if isinstance(card, dict) else getattr(card, "id", None)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _cards(zone: Any) -> list[Any]:
    return list(zone) if isinstance(zone, (list, tuple)) else []


def _players(obs: dict[str, Any]) -> tuple[Optional[dict], Optional[dict]]:
    current = obs.get("current") if isinstance(obs, dict) else None
    players = current.get("players") if isinstance(current, dict) else None
    try:
        seat = int(current.get("yourIndex", 0))
    except (AttributeError, TypeError, ValueError):
        return None, None
    if not isinstance(players, list) or len(players) != 2 or seat not in (0, 1):
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


def _hand_counts(player: dict) -> Counter[int]:
    return Counter(_zone_ids(player, "hand"))


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
    try:
        area_i = int(area)
        index_i = int(index)
    except (TypeError, ValueError):
        return None
    if index_i < 0:
        return None
    if area_i == AREA_DECK:
        zone = select.get("deck") or []
    elif area_i == AREA_STADIUM:
        zone = current.get("stadium") or []
    elif area_i == AREA_LOOKING:
        zone = current.get("looking") or []
    else:
        try:
            seat = int(
                current.get("yourIndex", 0)
                if player_index is None
                else player_index
            )
            player = players[seat]
        except (TypeError, ValueError, IndexError):
            return None
        zone_key = {
            AREA_HAND: "hand",
            AREA_DISCARD: "discard",
            AREA_ACTIVE: "active",
            AREA_BENCH: "bench",
            AREA_PRIZE: "prize",
        }.get(area_i)
        if zone_key is None:
            return None
        zone = player.get(zone_key) or []
    try:
        return zone[index_i]
    except (IndexError, TypeError):
        return None


def _option_card(obs: dict[str, Any], option: dict[str, Any]) -> Optional[Any]:
    current = obs.get("current") or {}
    try:
        option_type = int(option.get("type", -1))
    except (TypeError, ValueError):
        return None
    if option_type == OPT_PLAY:
        return _resolve_card(
            obs,
            area=AREA_HAND,
            index=option.get("index"),
            player_index=current.get("yourIndex", 0),
        )
    return _resolve_card(
        obs,
        area=option.get("area"),
        index=option.get("index"),
        player_index=option.get(
            "playerIndex", current.get("yourIndex", 0)
        ),
    )


def _effect_id(select: dict[str, Any]) -> Optional[int]:
    for key in ("effect", "contextCard"):
        value = _card_id(select.get(key))
        if value is not None:
            return value
    return None


def _setup_score(
    card_id: int,
    *,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    if card_id != CYNTHIAS_GIBLE:
        return 0.0
    return 2.4 if board[card_id] + chosen[card_id] < 2 else 0.0


def _champions_call_score(
    card_id: int,
    *,
    board: Counter[int],
    hand: Counter[int],
    chosen: Counter[int],
) -> float:
    """Fill one exact visible Cynthia evolution gap with diminishing value."""
    have = hand + chosen
    if card_id == CYNTHIAS_GARCHOMP_EX:
        return (
            3.0
            if board[CYNTHIAS_GABITE] > have[CYNTHIAS_GARCHOMP_EX]
            else 0.0
        )
    if card_id == CYNTHIAS_GABITE:
        open_lines = max(
            0, board[CYNTHIAS_GIBLE] - board[CYNTHIAS_GABITE]
        )
        return 2.4 if open_lines > have[CYNTHIAS_GABITE] else 0.0
    if card_id == CYNTHIAS_GIBLE:
        return 1.8 if board[card_id] + have[card_id] < 2 else 0.0
    if card_id == CYNTHIAS_ROSERADE:
        return (
            1.2
            if board[CYNTHIAS_ROSELIA] > have[CYNTHIAS_ROSERADE]
            else 0.0
        )
    return 0.0


def _main_score(
    obs: dict[str, Any],
    option: dict[str, Any],
) -> Optional[float]:
    try:
        option_type = int(option.get("type", -1))
    except (TypeError, ValueError):
        return None
    if option_type != OPT_EVOLVE:
        return 0.0
    card_id = _card_id(_option_card(obs, option))
    if card_id is None:
        return None
    return {
        CYNTHIAS_GABITE: 2.4,
        CYNTHIAS_GARCHOMP_EX: 3.0,
        # Roserade is deliberately neutral here. Its value depends on exact
        # current damage, HP, Weakness, Resistance, and attack legality.
        CYNTHIAS_ROSERADE: 0.0,
    }.get(card_id, 0.0)


def _combo_score(
    obs: dict[str, Any],
    combo: Sequence[int],
    *,
    me: dict,
) -> Optional[float]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    try:
        context = int(select.get("context", -1))
    except (TypeError, ValueError):
        return None
    effect = _effect_id(select)
    board = _board_counts(me)
    hand = _hand_counts(me)
    chosen: Counter[int] = Counter()
    total = 0.0

    if context not in {CTX_MAIN, CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
        if effect != CYNTHIAS_GABITE:
            return None
    for raw_index in combo:
        try:
            option = options[int(raw_index)]
        except (TypeError, ValueError, IndexError):
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
        if context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
            total += _setup_score(card_id, board=board, chosen=chosen)
        elif effect == CYNTHIAS_GABITE:
            total += _champions_call_score(
                card_id,
                board=board,
                hand=hand,
                chosen=chosen,
            )
        else:
            return None
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
    try:
        minimum = max(0, int(select.get("minCount", 0)))
        maximum = min(len(options), int(select.get("maxCount", len(options))))
    except (TypeError, ValueError):
        return False
    if minimum > maximum:
        return False
    for combo in action_combos:
        if not isinstance(combo, (list, tuple)):
            return False
        try:
            indices = [int(index) for index in combo]
        except (TypeError, ValueError):
            return False
        if (
            len(indices) < minimum
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
    if (not force_enabled and not enabled()) or not is_garchomp_deck(deck):
        return None
    if not isinstance(obs, dict) or not action_combos:
        return None
    me, opp = _players(obs)
    if me is None or opp is None or not _validated_stage(obs, action_combos):
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


__all__ = [
    "GUIDE_VERSION",
    "enabled",
    "guide_scores",
    "is_garchomp_deck",
]
