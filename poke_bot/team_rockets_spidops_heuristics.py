"""Sparse training teacher for the Team Rocket's Spidops specialist.

This teacher is intentionally narrower than the complete human pilot guide.
It scores only facts that the current observation resolves exactly: a
single-Prize Team Rocket opening with redundant Tarountula, exact
Buddy-Buddy Poffin or Team Rocket's Proton search prompts, visible Spidops
evolutions, and a legal Charging Up activation when a known Basic Energy is
already in the public discard pile.

Attack selection, Sneasel/Giovanni conversion turns, Tools, recovery,
Energy movement, and matchup plans remain masked until every fact needed to
rank the complete legal stage is exposed.  The full deck is used only as an
identity gate.  This scorer never chooses or executes a runtime action.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence


GUIDE_VERSION = "team-rockets-spidops-north-star-v1"

BASIC_GRASS_ENERGY = 1
BASIC_PSYCHIC_ENERGY = 5
TEAM_ROCKETS_ENERGY = 15

LILLIES_CLEFAIRY_EX = 272
TEAM_ROCKETS_TAROUNTULA = 400
TEAM_ROCKETS_SPIDOPS = 401
TEAM_ROCKETS_ARTICUNO = 414
TEAM_ROCKETS_MEWTWO_EX = 431
TEAM_ROCKETS_MIMIKYU = 434
TEAM_ROCKETS_SNEASEL = 464

BUDDY_BUDDY_POFFIN = 1086
TEAM_ROCKETS_PROTON = 1220

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
CTX_TO_HAND = 7

# Public top-ladder Spidops lists vary substantially in their secondary
# Team Rocket attackers.  The public archetype identity is the invariant 4/4
# Tarountula/Spidops engine with at most two Mewtwo; requiring Articuno,
# Mimikyu, or Sneasel incorrectly masked valid expert rows.
CORE_SIGNATURE_MINIMUMS = {
    TEAM_ROCKETS_TAROUNTULA: 4,
    TEAM_ROCKETS_SPIDOPS: 4,
}
CORE_SIGNATURE_MAXIMUMS = {
    TEAM_ROCKETS_MEWTWO_EX: 2,
}
TEAM_ROCKET_BASICS = frozenset(
    {
        TEAM_ROCKETS_TAROUNTULA,
        TEAM_ROCKETS_ARTICUNO,
        TEAM_ROCKETS_MEWTWO_EX,
        TEAM_ROCKETS_MIMIKYU,
        TEAM_ROCKETS_SNEASEL,
    }
)
SINGLE_PRIZE_TEAM_ROCKET_BASICS = TEAM_ROCKET_BASICS - {
    TEAM_ROCKETS_MEWTWO_EX
}
TEAM_ROCKET_POKEMON = TEAM_ROCKET_BASICS | {TEAM_ROCKETS_SPIDOPS}
KNOWN_BASIC_ENERGIES = frozenset(
    {BASIC_GRASS_ENERGY, BASIC_PSYCHIC_ENERGY}
)
ABSTENTION_MARGIN = 0.25


def enabled() -> bool:
    """Return whether the generic guide registry selected this teacher."""
    from . import deck_guides

    return (
        deck_guides.enabled()
        and deck_guides.selected_id() == "team-rockets-spidops"
    )


def is_team_rockets_spidops_deck(deck: Iterable[int]) -> bool:
    """Require a 60-card public Spidops signature, including <=2 Mewtwo."""
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
    ) and all(
        counts[card_id] <= maximum
        for card_id, maximum in CORE_SIGNATURE_MAXIMUMS.items()
    )


def _exact_int(value: Any) -> Optional[int]:
    """Accept exact integer-like engine values without lossy coercion."""
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
    if seat is None:
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


def _resolve_card(
    obs: dict[str, Any],
    *,
    area: Any,
    index: Any,
    player_index: Any = None,
) -> Optional[Any]:
    """Resolve an option only from zones exposed by the current observation."""
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    players = current.get("players") or []
    area_i = _exact_int(area)
    index_i = _exact_int(index)
    if area_i is None or index_i is None:
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
    your_index = _exact_int(current.get("yourIndex", 0))
    declared_seat = option.get("playerIndex")
    if your_index not in (0, 1) or (
        declared_seat is not None
        and _exact_int(declared_seat) != your_index
    ):
        return None
    option_type = _exact_int(option.get("type"))
    if option_type is None:
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
        player_index=your_index,
    )


def _effect_id(select: dict[str, Any]) -> Optional[int]:
    for key in ("effect", "contextCard"):
        value = _card_id(select.get(key))
        if value is not None:
            return value
    return None


def _discard_has_known_basic_energy(me: dict) -> bool:
    return any(
        card_id in KNOWN_BASIC_ENERGIES
        for card_id in _zone_ids(me, "discard")
    )


def _setup_score(
    card_id: int,
    *,
    context: int,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    """Build a pivot Active and a redundant single-Prize Spidops board."""
    if card_id not in TEAM_ROCKET_BASICS:
        return 0.0
    if context == CTX_SETUP_ACTIVE:
        # Mimikyu's printed zero Retreat Cost makes it the least committal
        # exact setup pivot.  Tarountula remains usable when no pivot is legal.
        if card_id == TEAM_ROCKETS_MIMIKYU:
            return 3.0
        if card_id == TEAM_ROCKETS_TAROUNTULA:
            return 1.0
        if card_id in {
            TEAM_ROCKETS_ARTICUNO,
            TEAM_ROCKETS_SNEASEL,
        }:
            return 0.6
        return 0.0

    if card_id == TEAM_ROCKETS_TAROUNTULA:
        return (
            3.2
            if board[card_id] + chosen[card_id] < 3
            else 0.4
        )
    visible_rockets = sum(
        board[identity] + chosen[identity]
        for identity in TEAM_ROCKET_POKEMON
    )
    if card_id not in SINGLE_PRIZE_TEAM_ROCKET_BASICS:
        # The dedicated guide does not create a default Mewtwo bench label.
        return 0.0
    if card_id == TEAM_ROCKETS_MIMIKYU and board[card_id] + chosen[card_id] == 0:
        return 1.6
    if card_id == TEAM_ROCKETS_ARTICUNO and board[card_id] + chosen[card_id] == 0:
        return 1.0
    if card_id == TEAM_ROCKETS_SNEASEL and board[card_id] + chosen[card_id] == 0:
        return 0.8
    return 0.5 if visible_rockets < 6 else 0.0


def _main_score(
    obs: dict[str, Any],
    option: dict[str, Any],
    *,
    me: dict,
) -> Optional[float]:
    option_type = _exact_int(option.get("type"))
    if option_type is None:
        return None
    if option_type not in {OPT_EVOLVE, OPT_ABILITY}:
        return 0.0
    if option_type == OPT_EVOLVE:
        if _exact_int(option.get("area")) != AREA_HAND:
            return None
        card_id = _card_id(_option_card(obs, option))
        target_area = _exact_int(option.get("inPlayArea"))
        if target_area not in {AREA_ACTIVE, AREA_BENCH}:
            return None
        target = _resolve_card(
            obs,
            area=target_area,
            index=option.get("inPlayIndex"),
            player_index=(obs.get("current") or {}).get("yourIndex", 0),
        )
        if card_id is None or _card_id(target) is None:
            return None
        return (
            3.2
            if card_id == TEAM_ROCKETS_SPIDOPS
            and _card_id(target) == TEAM_ROCKETS_TAROUNTULA
            else 0.0
        )
    if _exact_int(option.get("area")) not in {AREA_ACTIVE, AREA_BENCH}:
        return None
    card_id = _card_id(_option_card(obs, option))
    if card_id is None:
        return None
    return (
        2.8
        if card_id == TEAM_ROCKETS_SPIDOPS
        and _discard_has_known_basic_energy(me)
        else 0.0
    )


def _combo_score(
    obs: dict[str, Any],
    combo: Sequence[int],
    *,
    me: dict,
) -> Optional[float]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    context = _exact_int(select.get("context"))
    if context is None:
        return None
    effect = _effect_id(select)
    board = _board_counts(me)
    chosen: Counter[int] = Counter()
    total = 0.0

    search_context = (
        effect == BUDDY_BUDDY_POFFIN and context == CTX_TO_BENCH
    ) or (
        effect == TEAM_ROCKETS_PROTON and context == CTX_TO_HAND
    )
    if (
        context not in {CTX_MAIN, CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}
        and not search_context
    ):
        return None
    for raw_index in combo:
        index = _exact_int(raw_index)
        if index is None:
            return None
        try:
            option = options[index]
        except (IndexError, TypeError):
            return None
        if not isinstance(option, dict):
            return None
        if context == CTX_MAIN:
            score = _main_score(obs, option, me=me)
            if score is None:
                return None
            total += score
            continue
        option_type = _exact_int(option.get("type"))
        if option_type != OPT_CARD:
            return None
        if search_context and _exact_int(option.get("area")) != AREA_DECK:
            return None
        if (
            context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}
            and _exact_int(option.get("area")) != AREA_HAND
        ):
            return None
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
        elif search_context:
            total += _setup_score(
                card_id,
                context=CTX_SETUP_BENCH,
                board=board,
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
    """Require one complete canonical factorized legal-action stage."""
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
        or not 0 <= minimum <= maximum <= len(options)
    ):
        return False
    normalized: list[tuple[int, ...]] = []
    for combo in action_combos:
        if not isinstance(combo, (list, tuple)):
            return False
        indices = [_exact_int(index) for index in combo]
        if any(index is None for index in indices):
            return False
        exact_indices = [int(index) for index in indices if index is not None]
        if (
            len(exact_indices) > maximum
            or len(exact_indices) != len(set(exact_indices))
            or any(
                index < 0 or index >= len(options)
                for index in exact_indices
            )
        ):
            return False
        normalized.append(tuple(exact_indices))
    if len(normalized) != len(set(normalized)):
        return False

    # ``features.factorized_teacher_forcing_stages`` supplies every possible
    # next choice for one shared ordered prefix, plus STOP once ``minCount`` is
    # met.  Infer that prefix and accept only an exact candidate-set match.
    supplied = set(normalized)
    first = normalized[0]
    for prefix_length in range(len(first) + 1):
        prefix = first[:prefix_length]
        if (
            len(prefix) > maximum
            or len(prefix) != len(set(prefix))
            or any(index < 0 or index >= len(options) for index in prefix)
        ):
            continue
        if len(prefix) >= maximum:
            expected = {prefix}
        else:
            expected = {
                prefix + (index,)
                for index in range(len(options))
                if index not in prefix
            }
            if len(prefix) >= minimum:
                expected.add(prefix)
        if supplied == expected and len(normalized) == len(expected):
            return True
    return False


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Score every supplied legal combo or mask the complete stage."""
    if (
        (not force_enabled and not enabled())
        or not is_team_rockets_spidops_deck(deck)
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
    if len(scores) < 2:
        return None
    ordered_scores = sorted(scores, reverse=True)
    if ordered_scores[0] - ordered_scores[1] < ABSTENTION_MARGIN:
        return None
    return scores


__all__ = [
    "GUIDE_VERSION",
    "enabled",
    "guide_scores",
    "is_team_rockets_spidops_deck",
]
