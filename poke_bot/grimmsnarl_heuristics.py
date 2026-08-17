"""Sparse training teacher for Marnie's Grimmsnarl ex.

The teacher implements only a small, auditable subset of the research guide:
core-line setup, core evolution, and exact Spikemuth Gym / Buddy-Buddy Poffin
search prompts.  It intentionally does *not* guess at Punk Up energy targets,
Shadow Bullet targets, Froslass/Munkidori damage placement, prize maps, or
matchup plans.  Those choices need more state or expert-validated predicates.

Every returned list is aligned with the complete supplied legal action stage.
If a combo is malformed, a required option cannot be resolved, the prompt
origin is ambiguous, or the preference margin is small, the whole stage is
masked by returning ``None``.  This module is a training-only target provider;
selection and authorization live in :mod:`poke_bot.deck_guides`, and it has no
runtime action authority.

Reviewed strategy sources (2026-07-25):
* https://www.tcgplayer.com/content/article/Marnie-s-Grimmsnarl-ex-Deck-Guide-Pokemon-TCG/bc476f13-b374-4602-ae4a-ff8513490c46
* https://limitlesstcg.com/decks/329
* https://www.pokebeach.com/2026/02/grimms-green-glow-up-marnies-grimmsnarl-for-euic
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence


GUIDE_VERSION = "marnie-grimmsnarl-north-star-v1"

SNORUNT = 103
FROSLASS = 104
MUNKIDORI = 112
MARNIES_IMPIDIMP = 646
MARNIES_MORGREM = 647
MARNIES_GRIMMSNARL_EX = 648
BUDDY_BUDDY_POFFIN = 1086
SPIKEMUTH_GYM = 1259

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
OPT_ATTACK = 13
OPT_END = 14

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2

LINE = frozenset({MARNIES_IMPIDIMP, MARNIES_MORGREM, MARNIES_GRIMMSNARL_EX})
SUPPORTED_SEARCH_EFFECTS = frozenset({BUDDY_BUDDY_POFFIN, SPIKEMUTH_GYM})
ABSTENTION_MARGIN = 0.25


def enabled() -> bool:
    """Return whether the generic registry selected this training teacher."""
    from . import deck_guides

    return (
        deck_guides.enabled()
        and deck_guides.selected_id() == "marnie-s-grimmsnarl-ex"
    )


def is_grimmsnarl_deck(deck: Iterable[int]) -> bool:
    """Require the checksum-contract's exact minimum core-line signature."""
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
    """Resolve one current-observation option without consulting future state."""
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
    your_index = current.get("yourIndex", 0)
    try:
        option_type = int(option.get("type", -1))
    except (TypeError, ValueError):
        return None
    if option_type == OPT_PLAY:
        return _resolve_card(
            obs,
            area=AREA_HAND,
            index=option.get("index"),
            player_index=your_index,
        )
    return _resolve_card(
        obs,
        area=option.get("area"),
        index=option.get("index"),
        player_index=option.get("playerIndex", your_index),
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
    """Prefer at most two visible Marnie's Impidimp setup commitments."""
    if card_id != MARNIES_IMPIDIMP:
        return 0.0
    return 2.4 if board[card_id] + chosen[card_id] < 2 else 0.0


def _spikemuth_score(
    card_id: int,
    *,
    board: Counter[int],
    hand: Counter[int],
    chosen: Counter[int],
) -> float:
    """Fill an exact, currently visible core evolution gap with diminishing value."""
    have = hand + chosen
    if card_id == MARNIES_GRIMMSNARL_EX:
        ready = board[MARNIES_MORGREM] + board[MARNIES_IMPIDIMP]
        return 3.0 if ready > have[card_id] else 0.0
    if card_id == MARNIES_MORGREM:
        open_lines = max(0, board[MARNIES_IMPIDIMP] - board[MARNIES_MORGREM])
        return 2.4 if open_lines > have[card_id] else 0.0
    if card_id == MARNIES_IMPIDIMP:
        return 1.8 if board[card_id] + have[card_id] < 2 else 0.0
    return 0.0


def _main_score(
    obs: dict[str, Any],
    option: dict[str, Any],
    *,
    me: dict,
) -> Optional[float]:
    """Score only exact core evolution/setup actions; leave all others neutral."""
    try:
        option_type = int(option.get("type", -1))
    except (TypeError, ValueError):
        return None
    if option_type not in {OPT_PLAY, OPT_EVOLVE}:
        return 0.0
    card = _option_card(obs, option)
    card_id = _card_id(card)
    if card_id is None:
        return None
    if option_type == OPT_EVOLVE:
        return {
            MARNIES_GRIMMSNARL_EX: 3.0,
            MARNIES_MORGREM: 2.0,
        }.get(card_id, 0.0)
    if card_id == SPIKEMUTH_GYM:
        board = _board_counts(me)
        hand = _hand_counts(me)
        has_visible_gap = (
            board[MARNIES_IMPIDIMP] < 2
            or (
                board[MARNIES_IMPIDIMP] > 0
                and hand[MARNIES_MORGREM] + hand[MARNIES_GRIMMSNARL_EX] == 0
            )
        )
        return 1.5 if has_visible_gap else 0.0
    if card_id == BUDDY_BUDDY_POFFIN:
        return 1.2 if _board_counts(me)[MARNIES_IMPIDIMP] < 2 else 0.0
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
            total += _setup_score(card_id, board=board, chosen=chosen)
        elif effect == BUDDY_BUDDY_POFFIN:
            total += _setup_score(card_id, board=board, chosen=chosen)
        elif effect == SPIKEMUTH_GYM:
            total += _spikemuth_score(
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
    """Fail closed unless every supplied legal combo is structurally scoreable."""
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
        if len(indices) < minimum or len(indices) > maximum:
            return False
        if len(indices) != len(set(indices)):
            return False
        if any(index < 0 or index >= len(options) for index in indices):
            return False
    return True


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Return complete aligned scores, or mask the entire ambiguous stage."""
    if (not force_enabled and not enabled()) or not is_grimmsnarl_deck(deck):
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
    "guide_scores",
    "enabled",
    "is_grimmsnarl_deck",
]
