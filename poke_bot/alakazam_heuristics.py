"""Guide-derived action targets for the Powerful Hand Alakazam specialist.

This module is deliberately a *teacher*, not a rule bot.  It scores only
high-confidence, publicly observable Alakazam choices.  The specialist learner
distils those scores into its existing policy logits with a small masked loss;
serving and formal evaluation remain neural-only.

The strategy contract comes from current Powerful Hand / Dudunsparce guides:

* Powerful Hand places two damage counters per card in hand, so exact KO math
  and hand-preserving sequencing matter.
* Kadabra, Alakazam, Dudunsparce, Dawn, Hilda, and Enriching Energy are a draw
  chain; stop spending/drawing once the required KO hand is reached.
* Keep a replacement Alakazam line and a Dudunsparce recovery route, and never
  draw the deck to zero.
* Boss, Enhanced Hammer, and Xerosic's Machinations are conditional tools, not
  cards to fire merely because they are legal. Matchup notes in the pilot
  brief cover Rocky Fighting / Mist Energy Hammer, mirror Xerosic, Shaymin,
  and Battle Cage vs Froslass / Munkidori.

References (reviewed 2026-07-20):
  https://www.tcgplayer.com/content/article/Alakazam-Deck-Guide-Pok%C3%A9mon-TCG/7eb46b82-9dc5-40d8-adf9-28cca05f070f/
  https://ptcgonews.com/tips/ptcgl-alakazam-deck-guide/
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717697

The scorer uses raw observation dictionaries so it can see ``select.effect``
and ``select.contextCard``.  Those fields identify the card which created a
sub-selection (Dawn vs Hilda vs Poke Pad, for example) but are intentionally
not present in the deployed schema-v5 option token.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence


GUIDE_VERSION = "powerful-hand-v1"

# Exact cards in the pinned top-ladder representative.
PSYCHIC_ENERGY = 5
MIST_ENERGY = 11
ENRICHING_ENERGY = 13
TELEPATH_PSYCHIC_ENERGY = 19
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
DUNSPARCE = 305
SHAYMIN = 343
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
NIGHTTIME_MINE = 1266

POWERFUL_HAND_ATTACK = 1072
BASIC_ENERGY_IDS = frozenset(range(1, 9))

# Raw competition enum values.  Keeping them local makes the teacher usable in
# featurization workers and unit tests without importing the native cg runtime.
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_LOOKING = 12

OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2

ALAKAZAM_LINE = frozenset({ABRA, KADABRA, ALAKAZAM})
DUDUNSPARCE_LINE = frozenset({DUNSPARCE, DUDUNSPARCE})
DRAW_ENGINE = frozenset({KADABRA, ALAKAZAM, DUDUNSPARCE})
RECOVERY_CARDS = frozenset({NIGHT_STRETCHER, SACRED_ASH, LANA_AID})


def enabled() -> bool:
    """Return whether fresh replay featurization should build guide targets."""
    value = os.environ.get("POKEBOT_ALAKAZAM_GUIDE_TARGETS", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_alakazam_deck(deck: Iterable[int]) -> bool:
    """Match the Powerful Hand line without relying on a transient board."""
    counts = Counter(int(card_id) for card_id in deck)
    return all(counts[card_id] >= 2 for card_id in ALAKAZAM_LINE)


def _card_id(card: Any) -> Optional[int]:
    if isinstance(card, dict) and card.get("id") is not None:
        try:
            return int(card["id"])
        except (TypeError, ValueError):
            return None
    if card is not None and getattr(card, "id", None) is not None:
        try:
            return int(card.id)
        except (TypeError, ValueError):
            return None
    return None


def _cards(zone: Any) -> list[Any]:
    return list(zone) if isinstance(zone, (list, tuple)) else []


def _players(obs: dict[str, Any]) -> tuple[Optional[dict], Optional[dict]]:
    current = obs.get("current") if isinstance(obs, dict) else None
    if not isinstance(current, dict):
        return None, None
    players = current.get("players") or []
    if not isinstance(players, list) or len(players) < 2:
        return None, None
    try:
        your_index = int(current.get("yourIndex", 0))
    except (TypeError, ValueError):
        return None, None
    if your_index not in (0, 1):
        return None, None
    if not isinstance(players[your_index], dict) or not isinstance(
        players[1 - your_index], dict
    ):
        return None, None
    return players[your_index], players[1 - your_index]


def _first(zone: Any) -> Optional[Any]:
    values = _cards(zone)
    return values[0] if values else None


def _remaining_hp(card: Any) -> int:
    if not isinstance(card, dict):
        return 0
    try:
        return max(0, int(card.get("hp", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _energy_ids(card: Any) -> list[int]:
    if not isinstance(card, dict):
        return []
    values = card.get("energyCards") or []
    result: list[int] = []
    for value in values:
        card_id = _card_id(value)
        if card_id is not None:
            result.append(card_id)
    return result


def _board_cards(player: dict) -> list[Any]:
    return _cards(player.get("active")) + _cards(player.get("bench"))


def _board_counts(player: dict) -> Counter[int]:
    return Counter(
        card_id
        for card_id in (_card_id(card) for card in _board_cards(player))
        if card_id is not None
    )


def _hand_ids(player: dict) -> list[int]:
    return [
        card_id
        for card_id in (_card_id(card) for card in _cards(player.get("hand")))
        if card_id is not None
    ]


def _discard_ids(player: dict) -> list[int]:
    return [
        card_id
        for card_id in (_card_id(card) for card in _cards(player.get("discard")))
        if card_id is not None
    ]


def _hand_count(player: dict) -> int:
    try:
        return max(0, int(player.get("handCount", len(_hand_ids(player))) or 0))
    except (TypeError, ValueError):
        return len(_hand_ids(player))


def _deck_count(player: dict) -> int:
    try:
        return max(0, int(player.get("deckCount", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _prizes(player: dict) -> int:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    for key in ("prizeCount", "remainingPrizes"):
        try:
            if player.get(key) is not None:
                return max(0, int(player[key]))
        except (TypeError, ValueError):
            pass
    return 6


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
    your_index = current.get("yourIndex", 0)
    option_type = int(option.get("type", -1) or -1)
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


@lru_cache(maxsize=2048)
def _card_flags(card_id: int) -> tuple[bool, bool, bool]:
    """Return ``(rule_box, tera, psychic)`` from the installed card table."""
    try:
        from . import features

        card = features.card_table().get(int(card_id))
    except Exception:
        card = None
    if card is None:
        # Known exact-list fallbacks keep unit tests/native-less diagnostics sane.
        return card_id == FEZANDIPITI_EX, card_id == 121, card_id in ALAKAZAM_LINE
    rule_box = bool(
        getattr(card, "ex", False)
        or getattr(card, "megaEx", False)
        or getattr(card, "aceSpec", False)
    )
    tera = bool(getattr(card, "tera", False))
    try:
        psychic = int(getattr(card, "energyType", -1)) == 5
    except (TypeError, ValueError):
        psychic = False
    return rule_box, tera, psychic


@lru_cache(maxsize=2048)
def _minimum_attack_cost(card_id: int) -> Optional[int]:
    """Return the fewest printed energy symbols on any attack."""
    try:
        from . import cg_env, features

        card = features.card_table().get(int(card_id))
        attack_ids = set(getattr(card, "attacks", ()) or ())
        costs = [
            len(getattr(attack, "energies", ()) or ())
            for attack in cg_env.all_attack()
            if int(getattr(attack, "attackId", -1)) in attack_ids
        ]
        costs = [cost for cost in costs if cost > 0]
        return min(costs) if costs else None
    except Exception:
        # Dragapult ex fallback attack-cost probe in the installed pool.
        return 3 if int(card_id) == 121 else None


def _effect_id(select: dict[str, Any]) -> Optional[int]:
    for key in ("effect", "contextCard"):
        card_id = _card_id(select.get(key))
        if card_id is not None:
            return card_id
    return None


def _powerful_hand_lethal(me: dict, opp_card: Any, *, hand_delta: int = 0) -> bool:
    if opp_card is None or MIST_ENERGY in _energy_ids(opp_card):
        return False
    cards = max(0, _hand_count(me) + int(hand_delta))
    return 20 * cards >= _remaining_hp(opp_card) > 0


def _has_powerful_hand_attack(obs: dict[str, Any]) -> bool:
    select = obs.get("select") or {}
    return any(
        int(option.get("type", -1) or -1) == OPT_ATTACK
        and int(option.get("attackId", -1) or -1) == POWERFUL_HAND_ATTACK
        for option in (select.get("option") or [])
        if isinstance(option, dict)
    )


def _setup_need(card_id: int, board: Counter[int], chosen: Counter[int]) -> float:
    """Score one searched/setup Pokémon with diminishing quota value."""
    have = board[card_id] + chosen[card_id]
    if card_id == ABRA:
        return 2.2 if have < 3 else -0.4
    if card_id == DUNSPARCE:
        return 2.0 if have < 2 else -0.3
    if card_id == KADABRA:
        open_lines = max(0, board[ABRA] - board[KADABRA] - board[ALAKAZAM])
        return 2.2 if have < max(1, open_lines) else 0.2
    if card_id == ALAKAZAM:
        ready = board[KADABRA]
        return 2.8 if have < max(1, ready) else 0.4
    if card_id == DUDUNSPARCE:
        return 2.1 if have < max(1, board[DUNSPARCE]) else 0.1
    if card_id == SHAYMIN:
        return 0.3
    if card_id == FEZANDIPITI_EX:
        return -0.8
    return 0.0


def _recovery_is_needed(me: dict) -> bool:
    board = _board_counts(me)
    hand = Counter(_hand_ids(me))
    discard = Counter(_discard_ids(me))
    missing_live_line = (
        board[ALAKAZAM] + hand[ALAKAZAM] == 0
        or board[KADABRA] + hand[KADABRA] == 0
        or board[ABRA] + hand[ABRA] == 0
    )
    return missing_live_line and any(discard[cid] for cid in ALAKAZAM_LINE)


def _attached_energy_units(card: Any) -> int:
    if not isinstance(card, dict):
        return 0
    raw = card.get("energies")
    if isinstance(raw, (list, tuple)):
        try:
            total = sum(max(0, int(value or 0)) for value in raw)
            if total > 0:
                return total
        except (TypeError, ValueError):
            pass
    return len(_energy_ids(card))


def _nighttime_mine_denies_attack(opp: dict) -> bool:
    """True only when +{C} makes a currently powered Tera attack illegal."""
    for card in _board_cards(opp):
        card_id = _card_id(card)
        if card_id is None or not _card_flags(card_id)[1]:
            continue
        printed = _minimum_attack_cost(card_id)
        if printed is None:
            continue
        # Exactly printed cost is ready before Mine and one short afterward.
        if _attached_energy_units(card) == printed:
            return True
    return False


def _rule_box(card: Any) -> bool:
    card_id = _card_id(card)
    return bool(card_id is not None and _card_flags(card_id)[0])


def _main_option_score(
    obs: dict[str, Any],
    option: dict[str, Any],
    *,
    me: dict,
    opp: dict,
) -> float:
    option_type = int(option.get("type", -1) or -1)
    own_active = _first(me.get("active"))
    opp_active = _first(opp.get("active"))
    current_lethal = (
        _card_id(own_active) == ALAKAZAM
        and _has_powerful_hand_attack(obs)
        and _powerful_hand_lethal(me, opp_active)
    )

    if option_type == OPT_ATTACK:
        if int(option.get("attackId", -1) or -1) != POWERFUL_HAND_ATTACK:
            return 0.0
        if MIST_ENERGY in _energy_ids(opp_active):
            return -3.5
        return 5.0 if _powerful_hand_lethal(me, opp_active) else 0.6

    if option_type == OPT_END:
        return -6.0 if current_lethal else -0.25

    source = _option_card(obs, option)
    source_id = _card_id(source)
    if source_id is None:
        return 0.0

    if option_type == OPT_EVOLVE:
        gain = {KADABRA: 1, ALAKAZAM: 2, DUDUNSPARCE: 2}.get(source_id, 0)
        score = {KADABRA: 1.7, ALAKAZAM: 2.5, DUDUNSPARCE: 1.8}.get(
            source_id, 0.0
        )
        if source_id == ALAKAZAM and _powerful_hand_lethal(
            me, opp_active, hand_delta=gain
        ):
            score += 1.2
        return score

    if option_type == OPT_ABILITY:
        if source_id == DUDUNSPARCE:
            if len(_board_cards(me)) <= 1:
                return -8.0  # shuffling the final Pokémon loses immediately
            if _deck_count(me) <= 1:
                return -6.0
            if current_lethal and _deck_count(me) <= 6:
                return -1.5
            return 2.3
        if source_id == FEZANDIPITI_EX:
            return -2.0 if _deck_count(me) <= 2 else (0.2 if current_lethal else 1.1)
        return 0.0

    if option_type == OPT_ATTACH:
        target = _resolve_card(
            obs,
            area=option.get("inPlayArea"),
            index=option.get("inPlayIndex"),
            player_index=(obs.get("current") or {}).get("yourIndex", 0),
        )
        target_id = _card_id(target)
        if source_id == ENRICHING_ENERGY:
            return 3.2 if target_id in DUDUNSPARCE_LINE else 2.0
        if source_id == TELEPATH_PSYCHIC_ENERGY:
            _, _, psychic = _card_flags(target_id or -1)
            score = 1.8 if psychic else -1.0
            if target_id == ALAKAZAM and any(
                energy in (PSYCHIC_ENERGY, TELEPATH_PSYCHIC_ENERGY)
                for energy in _energy_ids(target)
            ):
                score -= 1.5
            return score
        if source_id == PSYCHIC_ENERGY and target_id in ALAKAZAM_LINE:
            already_powered = any(
                energy in (PSYCHIC_ENERGY, TELEPATH_PSYCHIC_ENERGY)
                for energy in _energy_ids(target)
            )
            return -1.0 if already_powered else 1.2
        return 0.0

    if option_type != OPT_PLAY:
        return 0.0

    board = _board_counts(me)
    if source_id == BUDDY_BUDDY_POFFIN:
        return 1.8 if board[ABRA] < 3 or board[DUNSPARCE] < 2 else 0.1
    if source_id == POKE_PAD:
        return 1.4 if any(
            _setup_need(cid, board, Counter()) > 1.0 for cid in DRAW_ENGINE
        ) else 0.2
    if source_id == RARE_CANDY:
        return 2.2 if board[ABRA] and ALAKAZAM in _hand_ids(me) else 0.7
    if source_id == DAWN:
        return -0.4 if current_lethal else 1.8
    if source_id == HILDA:
        return -0.4 if current_lethal else 1.3
    if source_id == BOSS_ORDERS:
        hand_after = -1
        targets = _cards(opp.get("bench"))
        lethal_targets = [
            card for card in targets if _powerful_hand_lethal(me, card, hand_delta=hand_after)
        ]
        if not lethal_targets:
            return -0.6
        return 3.2 if any(_rule_box(card) for card in lethal_targets) else 2.3
    if source_id == ENHANCED_HAMMER:
        attached = [
            energy
            for card in _board_cards(opp)
            for energy in _energy_ids(card)
        ]
        if MIST_ENERGY in attached:
            return 3.8
        return 0.8 if any(energy not in BASIC_ENERGY_IDS for energy in attached) else -0.4
    if source_id == NIGHTTIME_MINE:
        return 2.2 if _nighttime_mine_denies_attack(opp) else -1.7
    if source_id == XEROSIC:
        if current_lethal:
            return -1.5
        try:
            opp_hand = int(opp.get("handCount", 0) or 0)
        except (TypeError, ValueError):
            opp_hand = 0
        return min(1.4, 0.25 * max(0, opp_hand - 3))
    if source_id in RECOVERY_CARDS:
        return 1.6 if _recovery_is_needed(me) else -0.2
    if source_id == FEZANDIPITI_EX:
        # Its draw can be decisive, but an unnecessary two-prize bench liability
        # should not be a default setup action.
        return 0.5 if not current_lethal and _deck_count(me) > 4 else -1.0
    return 0.0


def _yes_no_score(
    option_type: int,
    *,
    effect_id: Optional[int],
    me: dict,
    opp: dict,
) -> float:
    if effect_id not in {KADABRA, ALAKAZAM, DUDUNSPARCE, FEZANDIPITI_EX}:
        return 0.0
    draw = {KADABRA: 2, ALAKAZAM: 3, DUDUNSPARCE: 3, FEZANDIPITI_EX: 3}[effect_id]
    own_active = _first(me.get("active"))
    opp_active = _first(opp.get("active"))
    already_lethal = (
        _card_id(own_active) == ALAKAZAM
        and _powerful_hand_lethal(me, opp_active)
    )
    unsafe = _deck_count(me) <= draw or (
        effect_id == DUDUNSPARCE and len(_board_cards(me)) <= 1
    )
    conserve = already_lethal and _deck_count(me) <= 6
    if option_type == OPT_YES:
        return -6.0 if unsafe else (-1.2 if conserve else 2.2)
    if option_type == OPT_NO:
        return 4.0 if unsafe else (1.4 if conserve else -1.0)
    return 0.0


def _selection_card_score(
    obs: dict[str, Any],
    card: Any,
    *,
    effect_id: Optional[int],
    me: dict,
    opp: dict,
    board: Counter[int],
    chosen: Counter[int],
) -> float:
    card_id = _card_id(card)
    if card_id is None:
        return 0.0

    if effect_id in {BUDDY_BUDDY_POFFIN, TELEPATH_PSYCHIC_ENERGY}:
        if effect_id == TELEPATH_PSYCHIC_ENERGY:
            return 2.5 if card_id == ABRA and board[ABRA] + chosen[ABRA] < 3 else -0.5
        return _setup_need(card_id, board, chosen)

    if effect_id in {POKE_PAD, DAWN, HILDA}:
        if card_id in ALAKAZAM_LINE or card_id in DUDUNSPARCE_LINE:
            return _setup_need(card_id, board, chosen)
        if effect_id == HILDA and card_id in {
            PSYCHIC_ENERGY,
            TELEPATH_PSYCHIC_ENERGY,
            ENRICHING_ENERGY,
        }:
            if card_id == ENRICHING_ENERGY and board[DUDUNSPARCE]:
                return 2.2
            return 1.0
        return 0.0

    if effect_id == BOSS_ORDERS:
        lethal = _powerful_hand_lethal(me, card)
        score = 3.5 if lethal else -0.4
        if lethal and _rule_box(card):
            score += 1.2
        # Prefer the line which shortens the remaining prize map.
        if lethal and _prizes(me) <= (3 if _rule_box(card) else 1):
            score += 0.8
        return score

    if effect_id == ENHANCED_HAMMER:
        # ENERGY_CARD options resolve to the host Pokémon; inspect the selected
        # energy itself in ``_combo_score`` where energyIndex is available.
        return 0.0

    if effect_id == XEROSIC:
        # We are the player being forced down to three cards.  Higher scores
        # identify safer discards; keep the attacker/draw chain and its scarce
        # energy/recovery.  This remains deliberately low-amplitude because
        # exact future needs are strategic, not mechanically provable.
        if card_id in ALAKAZAM_LINE:
            return -1.8
        if card_id in DUDUNSPARCE_LINE:
            return -1.4
        if card_id in {
            PSYCHIC_ENERGY,
            TELEPATH_PSYCHIC_ENERGY,
            ENRICHING_ENERGY,
        }:
            return -1.2
        if card_id in RECOVERY_CARDS or card_id == BOSS_ORDERS:
            return -0.8
        if card_id in {NIGHTTIME_MINE, ENHANCED_HAMMER, XEROSIC}:
            return 0.7
        if card_id in {BUDDY_BUDDY_POFFIN, POKE_PAD}:
            return 0.3
        return 0.0

    if effect_id in RECOVERY_CARDS:
        if card_id == ALAKAZAM:
            return 2.7
        if card_id == KADABRA:
            return 2.2
        if card_id == ABRA:
            return 1.7
        if card_id == DUDUNSPARCE:
            return 1.5
        if card_id in {PSYCHIC_ENERGY, TELEPATH_PSYCHIC_ENERGY}:
            return 0.8
        return 0.0

    if effect_id == RARE_CANDY:
        return 2.5 if card_id == ABRA else 0.0

    return _setup_need(card_id, board, chosen)


def _combo_score(
    obs: dict[str, Any],
    combo: Sequence[int],
    *,
    me: dict,
    opp: dict,
) -> float:
    select = obs.get("select") or {}
    options = select.get("option") or []
    try:
        context = int(select.get("context", -1))
    except (TypeError, ValueError):
        context = -1
    effect_id = _effect_id(select)
    board = _board_counts(me)
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

        if context == CTX_MAIN:
            total += _main_option_score(obs, option, me=me, opp=opp)
            continue

        if option_type in (OPT_YES, OPT_NO):
            total += _yes_no_score(
                option_type, effect_id=effect_id, me=me, opp=opp
            )
            continue

        if option_type in (OPT_ENERGY_CARD, OPT_ENERGY) and effect_id == ENHANCED_HAMMER:
            host = _option_card(obs, option)
            try:
                energy = (host.get("energyCards") or [])[int(option.get("energyIndex", -1))]
            except (AttributeError, IndexError, TypeError, ValueError):
                energy = None
            energy_id = _card_id(energy)
            total += 4.5 if energy_id == MIST_ENERGY else (1.0 if energy_id else 0.0)
            continue

        card = _option_card(obs, option)
        card_id = _card_id(card)
        if context == CTX_SETUP_ACTIVE:
            if card_id == DUNSPARCE:
                total += 2.5
            elif card_id == ABRA:
                total += 0.8
            elif card_id == SHAYMIN:
                total += 0.4
            elif card_id == FEZANDIPITI_EX:
                total -= 1.5
            continue

        # Setup-bench is intrinsically identified by its context.  All other
        # non-main card selections require an exact originating effect; a bare
        # TO_HAND prompt can be prize-taking and must not be mislabeled as a
        # search action.
        if effect_id is None and context != CTX_SETUP_BENCH:
            continue

        total += _selection_card_score(
            obs,
            card,
            effect_id=effect_id,
            me=me,
            opp=opp,
            board=board,
            chosen=chosen,
        )
        if card_id is not None:
            chosen[card_id] += 1

    # Optional setup/selection STOP stays neutral.  Useful choices receive a
    # positive score; unsafe or redundant choices fall below it.
    return max(-8.0, min(8.0, float(total)))


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Return aligned guide scores, or ``None`` when no safe ranking exists.

    The output is intentionally sparse/masked.  Equal or near-equal rows do not
    create a loss, preventing generic choices from being labelled by guesswork.
    """
    if (not force_enabled and not enabled()) or not is_alakazam_deck(deck):
        return None
    if not isinstance(obs, dict) or not action_combos:
        return None
    me, opp = _players(obs)
    if me is None or opp is None:
        return None
    options = (obs.get("select") or {}).get("option") or []
    for combo in action_combos:
        try:
            indices = [int(index) for index in combo]
        except (TypeError, ValueError):
            return None
        if any(index < 0 or index >= len(options) for index in indices):
            return None
    scores = [
        _combo_score(obs, combo, me=me, opp=opp)
        for combo in action_combos
    ]
    if not scores or any(not math.isfinite(score) for score in scores):
        return None
    if max(scores) - min(scores) < 0.25:
        return None
    return scores


__all__ = [
    "GUIDE_VERSION",
    "guide_scores",
    "enabled",
    "is_alakazam_deck",
]
