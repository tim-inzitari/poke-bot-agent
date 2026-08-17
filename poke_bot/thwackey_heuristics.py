"""Sparse training teacher for the Thwackey / Festival Lead specialist.

The canonical logical specialist identity is ``thwackey``.  Its compatible
physical matchup route and expert-corpus identity remain ``festival-lead``;
this module does not create or address another tensor row.

Only exact, causally observable parts of the researched game plan are scored:
core setup, core evolution, Festival Grounds deployment, legal Boom Boom
Groove activation, and exact Buddy-Buddy Poffin / Boom Boom Groove search
prompts.  Damage modifiers, attack sequencing, recovery, gust targets,
protection packages, and matchup plans are masked until every fact needed to
rank the complete legal stage is available.

The scorer is training-only.  It returns one finite score per supplied legal
combo or ``None`` for the whole stage, and never chooses a runtime action.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence


GUIDE_VERSION = "thwackey-festival-lead-north-star-v1"
PHYSICAL_ROUTE_ID = "festival-lead"

GROOKEY = 89
THWACKEY = 90
APPLIN_TWM = 92
DIPPLIN_TWM = 93
GOLDEEN_TWM = 100
BUDDY_BUDDY_POFFIN = 1086
FESTIVAL_GROUNDS = 1245

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

CORE_SIGNATURE_MINIMUMS = {
    GROOKEY: 2,
    THWACKEY: 2,
    DIPPLIN_TWM: 2,
    FESTIVAL_GROUNDS: 2,
}
FESTIVAL_LEAD_POKEMON = frozenset({DIPPLIN_TWM, GOLDEEN_TWM})
POFFIN_BASICS = frozenset({GROOKEY, APPLIN_TWM, GOLDEEN_TWM})
SUPPORTED_SEARCH_EFFECTS = frozenset({BUDDY_BUDDY_POFFIN, THWACKEY})
ABSTENTION_MARGIN = 0.25


def enabled() -> bool:
    """Return whether the generic guide registry selected this teacher."""
    from . import deck_guides

    return deck_guides.enabled() and deck_guides.selected_id() == "thwackey"


def is_thwackey_deck(deck: Iterable[int]) -> bool:
    """Require the contract's exact minimum Festival Lead engine signature."""
    try:
        counts = Counter(int(card_id) for card_id in deck)
    except (TypeError, ValueError):
        return False
    return all(
        counts[card_id] >= minimum
        for card_id, minimum in CORE_SIGNATURE_MINIMUMS.items()
    )


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
    """Resolve one option using only the current observation."""
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


def _active_has_festival_lead(me: dict) -> bool:
    return any(
        card_id in FESTIVAL_LEAD_POKEMON
        for card_id in _zone_ids(me, "active")
    )


def _festival_grounds_in_play(obs: dict[str, Any]) -> bool:
    current = obs.get("current") or {}
    return any(
        _card_id(card) == FESTIVAL_GROUNDS
        for card in _cards(current.get("stadium"))
    )


def _setup_score(
    card_id: int,
    *,
    context: int,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    """Build a visible Festival Lead active and two redundant engine lines."""
    if card_id not in POFFIN_BASICS:
        return 0.0
    if context == CTX_SETUP_ACTIVE:
        if card_id == GOLDEEN_TWM:
            return 3.0
        if card_id == APPLIN_TWM:
            return 1.0
        if card_id == GROOKEY:
            return 0.6
        return 0.0
    if card_id == GROOKEY:
        return 2.5 if board[card_id] + chosen[card_id] < 2 else 0.0
    if card_id == APPLIN_TWM:
        return 2.4 if board[card_id] + chosen[card_id] < 2 else 0.0
    if card_id == GOLDEEN_TWM:
        visible_leads = sum(
            board[identity] + chosen[identity]
            for identity in FESTIVAL_LEAD_POKEMON
        )
        return 1.8 if visible_leads == 0 else 0.0
    return 0.0


def _boom_search_score(
    card_id: int,
    *,
    obs: dict[str, Any],
    board: Counter[int],
    hand: Counter[int],
    chosen: Counter[int],
) -> float:
    """Fill only exact, current visible engine gaps from a Boom search."""
    if card_id == FESTIVAL_GROUNDS:
        return 3.2 if not _festival_grounds_in_play(obs) else 0.0
    if card_id == DIPPLIN_TWM:
        open_lines = max(0, board[APPLIN_TWM] - board[DIPPLIN_TWM])
        return 2.8 if open_lines > hand[card_id] + chosen[card_id] else 0.0
    if card_id == THWACKEY:
        open_lines = max(0, board[GROOKEY] - board[THWACKEY])
        return 2.6 if open_lines > hand[card_id] + chosen[card_id] else 0.0
    return 0.0


def _main_score(
    obs: dict[str, Any],
    option: dict[str, Any],
    *,
    me: dict,
) -> Optional[float]:
    try:
        option_type = int(option.get("type", -1))
    except (TypeError, ValueError):
        return None
    if option_type not in {OPT_PLAY, OPT_EVOLVE, OPT_ABILITY}:
        return 0.0
    card_id = _card_id(_option_card(obs, option))
    if card_id is None:
        return None
    if option_type == OPT_EVOLVE:
        return {DIPPLIN_TWM: 2.8, THWACKEY: 2.6}.get(card_id, 0.0)
    if option_type == OPT_ABILITY:
        return (
            3.0
            if card_id == THWACKEY and _active_has_festival_lead(me)
            else 0.0
        )
    if card_id == FESTIVAL_GROUNDS:
        return (
            2.8
            if _active_has_festival_lead(me)
            and not _festival_grounds_in_play(obs)
            else 0.0
        )
    if card_id == BUDDY_BUDDY_POFFIN:
        board = _board_counts(me)
        return (
            1.4
            if board[GROOKEY] < 2
            or board[APPLIN_TWM] < 2
            or not _active_has_festival_lead(me)
            else 0.0
        )
    return 0.0


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
        if effect not in SUPPORTED_SEARCH_EFFECTS:
            return None
    for raw_index in combo:
        try:
            option = options[int(raw_index)]
        except (TypeError, ValueError, IndexError):
            return None
        if not isinstance(option, dict):
            return None
        if context == CTX_MAIN:
            score = _main_score(obs, option, me=me)
            if score is None:
                return None
            total += score
            continue
        card_id = _card_id(_option_card(obs, option))
        if card_id is None:
            return None
        if context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
            total += _setup_score(
                card_id,
                context=context,
                board=board,
                chosen=chosen,
            )
        elif effect == BUDDY_BUDDY_POFFIN:
            total += _setup_score(
                card_id,
                context=CTX_SETUP_BENCH,
                board=board,
                chosen=chosen,
            )
        elif effect == THWACKEY:
            total += _boom_search_score(
                card_id,
                obs=obs,
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
    """Score a complete legal stage or mask it in full."""
    if (
        (not force_enabled and not enabled())
        or not is_thwackey_deck(deck)
        or not isinstance(obs, dict)
        or not action_combos
    ):
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
    "PHYSICAL_ROUTE_ID",
    "enabled",
    "guide_scores",
    "is_thwackey_deck",
]
