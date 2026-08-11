"""Training-only causal teacher for the owner's r241 Alakazam list.

The attached owner guide is the strategy authority.  This module turns only
high-confidence, publicly observable parts of that guide into sparse rankings
over a complete legal stage.  It never chooses a runtime action and it masks a
stage whenever it cannot make a safe distinction.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from . import alakazam_heuristics as _legacy


GUIDE_VERSION = "powerful-hand-new-list-r241-v1"
CANONICAL_MULTISET_SHA256 = (
    "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
)

PSYCHIC_ENERGY = 5
MIST_ENERGY = 11
ENRICHING_ENERGY = 13
TELEPATH_PSYCHIC_ENERGY = 19
ROCK_FIGHTING_ENERGY = 20
DUDUNSPARCE = 66
FROSLASS = 104
MUNKIDORI = frozenset({112, 139})
FEZANDIPITI_EX = 140
DUNSPARCE = 305
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
RARE_CANDY = 1079
ENHANCED_HAMMER = 1081
BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
SACRED_ASH = 1129
POKE_PAD = 1152
BOSS_ORDERS = 1182
LANA_AID = 1184
XEROSIC = 1197
HILDA = 1225
DAWN = 1231
BATTLE_CAGE = 1264

POWERFUL_HAND_ATTACK = 1072
BASIC_ENERGY_IDS = frozenset(range(1, 9))
SPECIAL_PROTECTION_ENERGY = frozenset({MIST_ENERGY, ROCK_FIGHTING_ENERGY})
ALAKAZAM_LINE = frozenset({ABRA, KADABRA, ALAKAZAM})
DUDUNSPARCE_LINE = frozenset({DUNSPARCE, DUDUNSPARCE})
RECOVERY_CARDS = frozenset({NIGHT_STRETCHER, SACRED_ASH, LANA_AID})

EXACT_DECK: tuple[int, ...] = (
    741, 741, 741, 741,
    742, 742, 742, 742,
    743, 743, 743,
    305, 305, 305,
    66, 66,
    140,
    1264, 1264, 1264, 1264,
    1086, 1086, 1086, 1086,
    1231, 1231, 1231, 1231,
    1081, 1081, 1081, 1081,
    1225, 1225, 1225, 1225,
    1152, 1152, 1152, 1152,
    1079, 1079, 1079,
    1097, 1097,
    1182, 1182, 1182,
    1197, 1197,
    1184,
    1129,
    19, 19, 19, 19,
    5, 5,
    13,
)
_EXACT_MULTISET = tuple(sorted(EXACT_DECK))


def enabled() -> bool:
    value = os.environ.get("POKEBOT_ALAKAZAM_NEW_LIST_GUIDE_TARGETS", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_alakazam_new_list_deck(deck: Iterable[int]) -> bool:
    """Require full equality with the owner's exact 60-card multiset."""

    return tuple(sorted(int(card_id) for card_id in deck)) == _EXACT_MULTISET


def _protected_from_powerful_hand(card: Any) -> bool:
    return bool(
        SPECIAL_PROTECTION_ENERGY.intersection(_legacy._energy_ids(card))
    )


def _powerful_hand_lethal(me: dict, card: Any, *, hand_delta: int = 0) -> bool:
    if card is None or _protected_from_powerful_hand(card):
        return False
    remaining_hp = _legacy._remaining_hp(card)
    hand = max(0, _legacy._hand_count(me) + int(hand_delta))
    return remaining_hp > 0 and 20 * hand >= remaining_hp


def _has_powerful_hand(obs: dict[str, Any]) -> bool:
    return any(
        int(option.get("type", -1) or -1) == _legacy.OPT_ATTACK
        and int(option.get("attackId", -1) or -1) == POWERFUL_HAND_ATTACK
        for option in ((obs.get("select") or {}).get("option") or [])
        if isinstance(option, dict)
    )


def _setup_need(card_id: int, board: Counter[int], chosen: Counter[int]) -> float:
    have = board[card_id] + chosen[card_id]
    if card_id == ABRA:
        return 2.5 if have < 2 else (1.0 if have < 3 else -0.4)
    if card_id == DUNSPARCE:
        return 2.0 if have < 1 else (0.8 if have < 2 else -0.3)
    if card_id == KADABRA:
        live = max(1, board[ABRA] - board[KADABRA] - board[ALAKAZAM])
        return 2.2 if have < live else 0.2
    if card_id == ALAKAZAM:
        live = max(1, board[KADABRA])
        return 2.8 if have < live else 0.3
    if card_id == DUDUNSPARCE:
        live = max(1, min(2, board[DUNSPARCE]))
        return 2.0 if have < live else 0.1
    if card_id == FEZANDIPITI_EX:
        return -0.8
    return 0.0


def _recovery_needed(me: dict) -> bool:
    board = _legacy._board_counts(me)
    hand = Counter(_legacy._hand_ids(me))
    discard = Counter(_legacy._discard_ids(me))
    missing_line = any(
        board[card_id] + hand[card_id] == 0 for card_id in ALAKAZAM_LINE
    )
    return missing_line and any(discard[card_id] for card_id in ALAKAZAM_LINE)


def _battle_cage_score(obs: dict[str, Any], me: dict, opp: dict) -> float:
    opponent_ids = {
        card_id
        for card_id in (
            _legacy._card_id(card) for card in _legacy._board_cards(opp)
        )
        if card_id is not None
    }
    threatened = bool(opponent_ids.intersection({FROSLASS, *MUNKIDORI}))
    has_bench_to_protect = bool(_legacy._cards(me.get("bench")))
    stadium = _legacy._cards((obs.get("current") or {}).get("stadium"))
    replaces_opponent_stadium = bool(
        stadium and _legacy._card_id(stadium[0]) != BATTLE_CAGE
    )
    if threatened and has_bench_to_protect:
        return 3.4
    if replaces_opponent_stadium:
        return 1.2
    return -0.8


def _main_score(
    obs: dict[str, Any], option: dict[str, Any], *, me: dict, opp: dict
) -> float:
    option_type = int(option.get("type", -1) or -1)
    own_active = _legacy._first(me.get("active"))
    opp_active = _legacy._first(opp.get("active"))
    current_lethal = bool(
        _legacy._card_id(own_active) == ALAKAZAM
        and _has_powerful_hand(obs)
        and _powerful_hand_lethal(me, opp_active)
    )

    if option_type == _legacy.OPT_ATTACK:
        if int(option.get("attackId", -1) or -1) != POWERFUL_HAND_ATTACK:
            return 0.0
        if _protected_from_powerful_hand(opp_active):
            return -4.5
        return 5.0 if _powerful_hand_lethal(me, opp_active) else 0.6
    if option_type == _legacy.OPT_END:
        return -6.0 if current_lethal else -0.2

    source = _legacy._option_card(obs, option)
    source_id = _legacy._card_id(source)
    if source_id is None:
        return 0.0

    if option_type == _legacy.OPT_EVOLVE:
        gain = {KADABRA: 1, ALAKAZAM: 2, DUDUNSPARCE: 2}.get(source_id, 0)
        score = {KADABRA: 1.7, ALAKAZAM: 2.6, DUDUNSPARCE: 1.8}.get(
            source_id, 0.0
        )
        if source_id == ALAKAZAM and _powerful_hand_lethal(
            me, opp_active, hand_delta=gain
        ):
            score += 1.2
        return score

    if option_type == _legacy.OPT_ABILITY:
        if source_id == DUDUNSPARCE:
            if len(_legacy._board_cards(me)) <= 1 or _legacy._deck_count(me) <= 1:
                return -7.0
            return -1.2 if current_lethal and _legacy._deck_count(me) <= 6 else 2.3
        if source_id == FEZANDIPITI_EX:
            return -2.0 if _legacy._deck_count(me) <= 2 else (0.2 if current_lethal else 1.0)
        return 0.0

    if option_type == _legacy.OPT_ATTACH:
        target = _legacy._resolve_card(
            obs,
            area=option.get("inPlayArea"),
            index=option.get("inPlayIndex"),
            player_index=(obs.get("current") or {}).get("yourIndex", 0),
        )
        target_id = _legacy._card_id(target)
        if source_id == ENRICHING_ENERGY:
            return 3.0 if target_id in DUDUNSPARCE_LINE else 0.6
        if source_id == TELEPATH_PSYCHIC_ENERGY:
            return 1.8 if target_id in ALAKAZAM_LINE else -0.5
        if source_id == PSYCHIC_ENERGY and target_id in ALAKAZAM_LINE:
            powered = any(
                energy in {PSYCHIC_ENERGY, TELEPATH_PSYCHIC_ENERGY}
                for energy in _legacy._energy_ids(target)
            )
            return -0.8 if powered else 1.4
        return 0.0

    if option_type != _legacy.OPT_PLAY:
        return 0.0

    board = _legacy._board_counts(me)
    if source_id == BUDDY_BUDDY_POFFIN:
        return 1.9 if board[ABRA] < 2 or board[DUNSPARCE] < 1 else 0.1
    if source_id == POKE_PAD:
        return 1.4 if board[ALAKAZAM] < 2 or board[DUDUNSPARCE] < 1 else 0.2
    if source_id == RARE_CANDY:
        return 2.3 if board[ABRA] and ALAKAZAM in _legacy._hand_ids(me) else 0.5
    if source_id in {DAWN, HILDA}:
        return -0.4 if current_lethal else (1.8 if source_id == DAWN else 1.4)
    if source_id == BOSS_ORDERS:
        lethal = [
            card
            for card in _legacy._cards(opp.get("bench"))
            if _powerful_hand_lethal(me, card, hand_delta=-1)
        ]
        if not lethal:
            return -0.6
        return 3.3 if any(_legacy._rule_box(card) for card in lethal) else 2.4
    if source_id == ENHANCED_HAMMER:
        attached = [
            energy
            for card in _legacy._board_cards(opp)
            for energy in _legacy._energy_ids(card)
        ]
        if SPECIAL_PROTECTION_ENERGY.intersection(attached):
            return 4.2
        return 0.9 if any(card not in BASIC_ENERGY_IDS for card in attached) else -0.5
    if source_id == XEROSIC:
        if current_lethal:
            return -1.5
        try:
            opp_hand = int(opp.get("handCount", 0) or 0)
        except (TypeError, ValueError):
            opp_hand = 0
        return min(1.6, 0.3 * max(0, opp_hand - 3))
    if source_id in RECOVERY_CARDS:
        return 1.7 if _recovery_needed(me) else -0.2
    if source_id == BATTLE_CAGE:
        return _battle_cage_score(obs, me, opp)
    if source_id == FEZANDIPITI_EX:
        # Flip the Script is useful only after a publicly observed prior-turn
        # knockout. The main-stage observation used here does not expose a
        # checksum-bound causal flag for that event, so deck depth alone must
        # never invent a positive bench label.
        return -1.0 if current_lethal or _legacy._deck_count(me) <= 4 else -0.4
    return 0.0


def _yes_no_score(option_type: int, effect_id: Optional[int], me: dict) -> float:
    draw = {KADABRA: 2, ALAKAZAM: 3, DUDUNSPARCE: 3, FEZANDIPITI_EX: 3}.get(
        effect_id
    )
    if draw is None:
        return 0.0
    unsafe = _legacy._deck_count(me) <= draw or (
        effect_id == DUDUNSPARCE and len(_legacy._board_cards(me)) <= 1
    )
    if option_type == _legacy.OPT_YES:
        return -6.0 if unsafe else 2.0
    if option_type == _legacy.OPT_NO:
        return 4.0 if unsafe else -0.8
    return 0.0


def _selection_score(
    card: Any,
    *,
    effect_id: Optional[int],
    me: dict,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    card_id = _legacy._card_id(card)
    if card_id is None:
        return 0.0
    if effect_id in {BUDDY_BUDDY_POFFIN, TELEPATH_PSYCHIC_ENERGY}:
        return _setup_need(card_id, board, chosen)
    if effect_id in {POKE_PAD, DAWN, HILDA}:
        if card_id in ALAKAZAM_LINE or card_id in DUDUNSPARCE_LINE:
            return _setup_need(card_id, board, chosen)
        if effect_id == HILDA and card_id in {
            PSYCHIC_ENERGY,
            TELEPATH_PSYCHIC_ENERGY,
            ENRICHING_ENERGY,
        }:
            return 2.0 if card_id == ENRICHING_ENERGY and board[DUDUNSPARCE] else 1.0
        return 0.0
    if effect_id == BOSS_ORDERS:
        lethal = _powerful_hand_lethal(me, card, hand_delta=-1)
        return (4.6 if _legacy._rule_box(card) else 3.5) if lethal else -0.4
    if effect_id == XEROSIC:
        if card_id in ALAKAZAM_LINE:
            return -1.9
        if card_id in DUDUNSPARCE_LINE:
            return -1.4
        if card_id in {PSYCHIC_ENERGY, TELEPATH_PSYCHIC_ENERGY, ENRICHING_ENERGY}:
            return -1.2
        if card_id in RECOVERY_CARDS or card_id == BOSS_ORDERS:
            return -0.9
        if card_id in {ENHANCED_HAMMER, XEROSIC, BATTLE_CAGE}:
            return 0.5
        return 0.1
    if effect_id in RECOVERY_CARDS:
        return {
            ALAKAZAM: 2.7,
            KADABRA: 2.2,
            ABRA: 1.7,
            DUDUNSPARCE: 1.5,
            DUNSPARCE: 1.0,
            PSYCHIC_ENERGY: 0.8,
            TELEPATH_PSYCHIC_ENERGY: 0.8,
        }.get(card_id, 0.0)
    if effect_id == RARE_CANDY:
        return 2.5 if card_id == ABRA else 0.0
    return _setup_need(card_id, board, chosen)


def _combo_score(
    obs: dict[str, Any], combo: Sequence[int], *, me: dict, opp: dict
) -> float:
    select = obs.get("select") or {}
    options = select.get("option") or []
    try:
        context = int(select.get("context", -1))
    except (TypeError, ValueError):
        context = -1
    effect_id = _legacy._effect_id(select)
    board = _legacy._board_counts(me)
    chosen: Counter[int] = Counter()
    total = 0.0

    for raw_index in combo:
        try:
            option = options[int(raw_index)]
        except (TypeError, ValueError, IndexError):
            return -8.0
        if not isinstance(option, dict):
            return -8.0
        option_type = int(option.get("type", -1) or -1)
        if context == _legacy.CTX_MAIN:
            total += _main_score(obs, option, me=me, opp=opp)
            continue
        if option_type in {_legacy.OPT_YES, _legacy.OPT_NO}:
            total += _yes_no_score(option_type, effect_id, me)
            continue
        if option_type in {_legacy.OPT_ENERGY_CARD, _legacy.OPT_ENERGY} and effect_id == ENHANCED_HAMMER:
            host = _legacy._option_card(obs, option)
            try:
                energy = (host.get("energyCards") or [])[int(option.get("energyIndex", -1))]
            except (AttributeError, IndexError, TypeError, ValueError):
                energy = None
            energy_id = _legacy._card_id(energy)
            total += 4.8 if energy_id in SPECIAL_PROTECTION_ENERGY else (1.0 if energy_id else 0.0)
            continue

        card = _legacy._option_card(obs, option)
        card_id = _legacy._card_id(card)
        if context == _legacy.CTX_SETUP_ACTIVE:
            total += {ABRA: 2.3, DUNSPARCE: 1.3, FEZANDIPITI_EX: -1.5}.get(
                card_id, 0.0
            )
            continue
        if effect_id is None and context != _legacy.CTX_SETUP_BENCH:
            continue
        total += _selection_score(
            card,
            effect_id=effect_id,
            me=me,
            board=board,
            chosen=chosen,
        )
        if card_id is not None:
            chosen[card_id] += 1
    return max(-8.0, min(8.0, float(total)))


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Return sparse aligned scores or mask the entire unsafe stage."""

    if (not force_enabled and not enabled()) or not is_alakazam_new_list_deck(deck):
        return None
    if not isinstance(obs, dict) or not action_combos:
        return None
    me, opp = _legacy._players(obs)
    options = (obs.get("select") or {}).get("option") or []
    if me is None or opp is None or not isinstance(options, list):
        return None
    for combo in action_combos:
        try:
            indices = [int(index) for index in combo]
        except (TypeError, ValueError):
            return None
        if any(index < 0 or index >= len(options) for index in indices):
            return None
    scores = [_combo_score(obs, combo, me=me, opp=opp) for combo in action_combos]
    if not scores or any(not math.isfinite(value) for value in scores):
        return None
    return None if max(scores) - min(scores) < 0.25 else scores


__all__ = [
    "CANONICAL_MULTISET_SHA256",
    "EXACT_DECK",
    "GUIDE_VERSION",
    "enabled",
    "guide_scores",
    "is_alakazam_new_list_deck",
]
