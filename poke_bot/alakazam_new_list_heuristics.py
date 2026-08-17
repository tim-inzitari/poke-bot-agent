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
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence

from . import alakazam_heuristics as _legacy


GUIDE_VERSION = "powerful-hand-new-list-r241-v1"
CANONICAL_MULTISET_SHA256 = (
    "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
)
# The owner supplied this corrected pilot guide on 2026-08-12.  It is a
# training-only source: this module remains absent from serving action routes.
OWNER_GUIDE_SHA256 = (
    "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b"
)

PSYCHIC_ENERGY = 5
MIST_ENERGY = 11
ENRICHING_ENERGY = 13
TELEPATH_PSYCHIC_ENERGY = 19
ROCK_FIGHTING_ENERGY = 20
DUDUNSPARCE = 66
FROSLASS = 104
# Only the ordinary Munkidori has Adrena-Brain's counter-moving ability.
MUNKIDORI = frozenset({112})
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
POWERFUL_HAND_EFFECT_BLOCKERS = frozenset({203, 835, 1136})
ALAKAZAM_LINE = frozenset({ABRA, KADABRA, ALAKAZAM})
DUDUNSPARCE_LINE = frozenset({DUNSPARCE, DUDUNSPARCE})
RECOVERY_CARDS = frozenset({NIGHT_STRETCHER, SACRED_ASH, LANA_AID})

_OPTION_TYPE_NAMES = {
    "yes": _legacy.OPT_YES,
    "no": _legacy.OPT_NO,
    "card": _legacy.OPT_CARD,
    "toolcard": _legacy.OPT_TOOL_CARD,
    "energycard": _legacy.OPT_ENERGY_CARD,
    "energy": _legacy.OPT_ENERGY,
    "play": _legacy.OPT_PLAY,
    "attach": _legacy.OPT_ATTACH,
    "evolve": _legacy.OPT_EVOLVE,
    "ability": _legacy.OPT_ABILITY,
    "skill": _legacy.OPT_ABILITY,
    "discard": _legacy.OPT_DISCARD,
    "retreat": _legacy.OPT_RETREAT,
    "attack": _legacy.OPT_ATTACK,
    "end": _legacy.OPT_END,
}
_CONTEXT_NAMES = {
    "main": _legacy.CTX_MAIN,
    "setupactive": _legacy.CTX_SETUP_ACTIVE,
    "setupbench": _legacy.CTX_SETUP_BENCH,
    # Target-select contexts are intentionally routed by their publicly named
    # effect, not treated as a main-stage action.  Their numeric enum values
    # vary across native/runtime diagnostic surfaces.
    "tobench": -2,
    "tohand": -2,
    "discardenergy": -2,
    "recoverpokemon": -2,
    "recoverenergy": -2,
}
_AREA_NAMES = {
    "deck": _legacy.AREA_DECK,
    "hand": _legacy.AREA_HAND,
    "discard": _legacy.AREA_DISCARD,
    "active": _legacy.AREA_ACTIVE,
    "bench": _legacy.AREA_BENCH,
    "prize": _legacy.AREA_PRIZE,
    "stadium": _legacy.AREA_STADIUM,
    "energy": _legacy.AREA_ENERGY,
    "tool": _legacy.AREA_TOOL,
    "looking": _legacy.AREA_LOOKING,
}

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


def _normalised_name(value: Any) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _option_type(option: Mapping[str, Any]) -> Optional[int]:
    """Read either the live numeric enum or a diagnostic string enum."""

    value = option.get("type")
    if isinstance(value, str):
        name = _normalised_name(value)
        return _OPTION_TYPE_NAMES.get(name) or next(
            (
                enum
                for label, enum in _OPTION_TYPE_NAMES.items()
                if name.endswith(label)
            ),
            None,
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _select_context(select: Mapping[str, Any]) -> Optional[int]:
    value = select.get("context")
    if isinstance(value, str):
        name = _normalised_name(value)
        return _CONTEXT_NAMES.get(name) or next(
            (
                enum
                for label, enum in _CONTEXT_NAMES.items()
                if name.endswith(label)
            ),
            None,
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _area(value: Any) -> Optional[int]:
    if isinstance(value, str):
        name = _normalised_name(value)
        return _AREA_NAMES.get(name) or next(
            (enum for label, enum in _AREA_NAMES.items() if name.endswith(label)),
            None,
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _source_card(obs: dict[str, Any], option: Mapping[str, Any]) -> Optional[Any]:
    """Resolve an option's source while accepting string enum diagnostics."""

    option_type = _option_type(option)
    if option_type is None:
        return None
    normalised = dict(option)
    normalised["type"] = option_type
    for key in ("area", "inPlayArea"):
        if key in normalised:
            area = _area(normalised.get(key))
            if area is None:
                return None
            normalised[key] = area
    return _legacy._option_card(obs, normalised)


def _known_count(player: Mapping[str, Any], key: str) -> Optional[int]:
    if key not in player:
        return None
    try:
        value = int(player.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, value)


def _fighting_type(card: Any) -> Optional[bool]:
    """Return the public Fighting-type fact, or ``None`` when it is absent.

    Rock Fighting Energy protects only a Fighting Pokémon.  Raw observations
    often omit types, so an unknown host is deliberately not treated as either
    protected or unprotected.
    """

    if isinstance(card, Mapping):
        raw_types = card.get("types", card.get("type"))
        if raw_types is not None:
            values = raw_types if isinstance(raw_types, (list, tuple, set)) else [raw_types]
            names = {_normalised_name(value) for value in values}
            if any(name in {"f", "fighting"} or name.endswith("fighting") for name in names):
                return True
            if names:
                return False
    card_id = _legacy._card_id(card)
    if card_id is None:
        return None
    try:
        from . import features

        metadata = features.card_table().get(card_id)
        energy_type = getattr(metadata, "energyType", None)
        value = getattr(energy_type, "value", energy_type)
        return int(value) == 6  # competition EnergyType.FIGHTING
    except (AttributeError, TypeError, ValueError, ImportError, OSError):
        return None
    except Exception:
        # The native card table is intentionally optional in guide workers.
        return None


def _powerful_hand_protection(card: Any) -> tuple[bool, bool]:
    """Return ``(prevented, conditional_protection_unknown)``.

    Mist protects every attached host from attack effects.  Rock Fighting has
    the same relevant text only for a Fighting host.  Never collapse Rock's
    missing type into a generic immunity.
    """

    if card is None:
        return False, False
    if isinstance(card, Mapping) and bool(
        card.get("preventsAttackEffects", False)
        or card.get("attackEffectsPrevented", False)
    ):
        return True, False
    if _legacy._card_id(card) in POWERFUL_HAND_EFFECT_BLOCKERS:
        return True, False
    energies = set(_legacy._energy_ids(card))
    if MIST_ENERGY in energies:
        return True, False
    if ROCK_FIGHTING_ENERGY not in energies:
        return False, False
    fighting = _fighting_type(card)
    if fighting is True:
        return True, False
    if fighting is False:
        return False, False
    return False, True


def _protected_from_powerful_hand(card: Any) -> bool:
    prevented, conditional_unknown = _powerful_hand_protection(card)
    return prevented or conditional_unknown


def _powerful_hand_lethal(me: Mapping[str, Any], card: Any) -> bool:
    if card is None or _protected_from_powerful_hand(card):
        return False
    return _powerful_hand_damage_reaches(me, card)


def _powerful_hand_damage_reaches(me: Mapping[str, Any], card: Any) -> bool:
    """Apply only Powerful Hand's public hand-to-counter damage equation."""

    if card is None:
        return False
    remaining_hp = _legacy._remaining_hp(card)
    hand = _legacy._hand_count(dict(me))
    return remaining_hp > 0 and 20 * hand >= remaining_hp


def _has_powerful_hand(obs: Mapping[str, Any]) -> bool:
    select = obs.get("select")
    options = select.get("option") if isinstance(select, Mapping) else None
    return any(
        _option_type(option) == _legacy.OPT_ATTACK
        and _exact_int(option.get("attackId")) == POWERFUL_HAND_ATTACK
        for option in (options or [])
        if isinstance(option, Mapping)
    )


def _exact_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _has_psychic_energy(card: Any) -> bool:
    return bool(
        {PSYCHIC_ENERGY, TELEPATH_PSYCHIC_ENERGY}.intersection(
            _legacy._energy_ids(card)
        )
    )


def _active_can_use_powerful_hand(me: Mapping[str, Any]) -> bool:
    active = _legacy._first(me.get("active"))
    return _legacy._card_id(active) == ALAKAZAM and _has_psychic_energy(active)


def _current_powerful_hand_lethal(
    obs: Mapping[str, Any], me: Mapping[str, Any], opp: Mapping[str, Any]
) -> bool:
    return bool(
        _active_can_use_powerful_hand(me)
        and _has_powerful_hand(obs)
        and _powerful_hand_lethal(me, _legacy._first(opp.get("active")))
    )


def _own_in_play_target(
    obs: dict[str, Any], option: Mapping[str, Any]
) -> tuple[Optional[Any], Optional[int]]:
    """Resolve only an explicitly named own Active/Bench attachment target."""

    current = obs.get("current") or {}
    your_index = _exact_int(current.get("yourIndex", 0))
    area = _area(option.get("inPlayArea"))
    index = _exact_int(option.get("inPlayIndex"))
    owner = _exact_int(option.get("inPlayPlayerIndex", your_index))
    if (
        your_index not in (0, 1)
        or owner != your_index
        or area not in {_legacy.AREA_ACTIVE, _legacy.AREA_BENCH}
        or index is None
        or index < 0
    ):
        return None, None
    return (
        _legacy._resolve_card(
            obs, area=area, index=index, player_index=your_index
        ),
        area,
    )


def _is_opponent_zone_target(
    obs: Mapping[str, Any], option: Mapping[str, Any], area: int
) -> bool:
    current = obs.get("current") or {}
    your_index = _exact_int(current.get("yourIndex", 0))
    owner = _exact_int(option.get("playerIndex"))
    index = _exact_int(option.get("index"))
    return (
        your_index in (0, 1)
        and owner == 1 - your_index
        and _area(option.get("area")) == area
        and index is not None
        and index >= 0
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


def _recovery_needed(me: Mapping[str, Any]) -> bool:
    board = _legacy._board_counts(dict(me))
    hand = Counter(_legacy._hand_ids(dict(me)))
    discard = Counter(_legacy._discard_ids(dict(me)))
    missing_line = any(
        board[card_id] + hand[card_id] == 0 for card_id in ALAKAZAM_LINE
    )
    return missing_line and any(discard[card_id] for card_id in ALAKAZAM_LINE)


def _prize_count(player: Mapping[str, Any]) -> Optional[int]:
    prize = player.get("prize")
    if isinstance(prize, (list, tuple)):
        return len(prize)
    for key in ("prizeCount", "remainingPrizes"):
        count = _exact_int(player.get(key))
        if count is not None and count >= 0:
            return count
    return None


def _prize_yield(card: Any) -> Optional[int]:
    """Use public card class only; unknown metadata remains unknown."""

    if not isinstance(card, Mapping):
        return None
    for key in ("prizeYield", "prizeCount"):
        value = _exact_int(card.get(key))
        if value is not None and value in {1, 2, 3}:
            return value
    if bool(card.get("megaEx", False)):
        return 3
    if _legacy._card_id(card) == FEZANDIPITI_EX:
        return 2
    if bool(card.get("ruleBox", False) or card.get("ex", False)):
        return 2
    card_id = _legacy._card_id(card)
    if card_id is None:
        return None
    try:
        from . import features

        metadata = features.card_table().get(card_id)
        if metadata is None:
            return None
        if bool(getattr(metadata, "megaEx", False)):
            return 3
        if bool(getattr(metadata, "ex", False)):
            return 2
        return 1
    except (AttributeError, ImportError, OSError):
        return None
    except Exception:
        return None


def _visible_ability(card: Any) -> bool:
    """Read an ability only from public card identity or explicit payload."""

    if isinstance(card, Mapping):
        if bool(card.get("hasAbility", False) or card.get("ability", False)):
            return True
        abilities = card.get("abilities", card.get("abilityIds"))
        if isinstance(abilities, (list, tuple, set)) and bool(abilities):
            return True
    # These exact-list identities publicly have a relevant Ability.  Do not
    # guess abilities for opaque opposing IDs when native metadata is absent.
    return _legacy._card_id(card) in {
        KADABRA,
        ALAKAZAM,
        DUDUNSPARCE,
        FEZANDIPITI_EX,
    }


def _battle_cage_score(
    obs: Mapping[str, Any], me: Mapping[str, Any], opp: Mapping[str, Any]
) -> float:
    opponent_ids = {
        card_id
        for card_id in (
            _legacy._card_id(card) for card in _legacy._board_cards(dict(opp))
        )
        if card_id is not None
    }
    bench = _legacy._cards(me.get("bench"))
    froslass_live = FROSLASS in opponent_ids and any(
        _visible_ability(card) for card in bench
    )
    opponent_has_damaged_pokemon = any(
        _legacy._remaining_hp(card) < _exact_int(card.get("maxHp"))
        for card in _legacy._board_cards(dict(opp))
        if isinstance(card, Mapping)
        and _exact_int(card.get("maxHp")) is not None
    )
    munkidori_live = any(
        _legacy._card_id(card) in MUNKIDORI
        and 7 in _legacy._energy_ids(card)
        for card in _legacy._board_cards(dict(opp))
    ) and bool(bench) and opponent_has_damaged_pokemon
    current = obs.get("current") or {}
    own_index = _exact_int(current.get("yourIndex", 0))
    stadium = _legacy._cards(current.get("stadium"))
    stadium_owner = (
        _exact_int(stadium[0].get("playerIndex", stadium[0].get("ownerIndex")))
        if stadium and isinstance(stadium[0], Mapping)
        else None
    )
    replaces_opponent_stadium = bool(
        stadium
        and _legacy._card_id(stadium[0]) != BATTLE_CAGE
        and own_index in (0, 1)
        and stadium_owner == 1 - own_index
    )
    if froslass_live or munkidori_live:
        return 3.4
    if replaces_opponent_stadium:
        return 0.8
    # A Stadium with no public immediate defensive or denial value is neutral;
    # it is not a generic early-play label.
    return 0.0


def _main_score(
    obs: dict[str, Any],
    option: Mapping[str, Any],
    *,
    me: Mapping[str, Any],
    opp: Mapping[str, Any],
) -> Optional[float]:
    """Score only a complete, public main-stage consequence.

    Search, gust, recovery, and Hammer source choices are factorised prefixes:
    their value depends on a later legal target.  They stay neutral here and
    receive a sparse label only at the target stage.
    """

    option_type = _option_type(option)
    if option_type is None:
        return None
    opp_active = _legacy._first(opp.get("active"))
    current_lethal = _current_powerful_hand_lethal(obs, me, opp)

    if option_type == _legacy.OPT_ATTACK:
        if _exact_int(option.get("attackId")) != POWERFUL_HAND_ATTACK:
            return 0.0
        prevented, uncertain = _powerful_hand_protection(opp_active)
        if prevented:
            return -4.5
        if uncertain:
            return 0.0
        return 5.0 if _powerful_hand_lethal(me, opp_active) else 0.0
    if option_type == _legacy.OPT_END:
        return -6.0 if current_lethal else 0.0

    source = _source_card(obs, option)
    source_id = _legacy._card_id(source)
    if option_type in {
        _legacy.OPT_PLAY,
        _legacy.OPT_ATTACH,
        _legacy.OPT_EVOLVE,
        _legacy.OPT_ABILITY,
    } and source_id is None:
        return None
    if source_id is None:
        return 0.0

    if option_type == _legacy.OPT_EVOLVE:
        target, _area = _own_in_play_target(obs, option)
        target_id = _legacy._card_id(target)
        # Psychic Draw is a later optional choice.  Do not smuggle its 2/3
        # cards into this evolution prefix or predict a later attack from it.
        if source_id == KADABRA and target_id == ABRA:
            return 1.4
        if source_id == ALAKAZAM and target_id == KADABRA:
            return 2.0
        if source_id == DUDUNSPARCE and target_id == DUNSPARCE:
            return 1.0
        return 0.0

    if option_type == _legacy.OPT_ABILITY:
        deck_count = _known_count(me, "deckCount")
        if source_id == DUDUNSPARCE:
            # Run Away Draw itself is the draw decision.  It draws three and
            # shuffles the user back only after drawing at least one card.
            if len(_legacy._board_cards(dict(me))) <= 1:
                return -7.0
            if deck_count is None:
                return 0.0
            if deck_count == 0:
                return -5.0
            if deck_count < 3:
                return 0.0
            return -1.0 if current_lethal else 1.4
        if source_id == FEZANDIPITI_EX:
            # A legal ability option proves Flip the Script's prior-KO
            # condition; no hidden hand/deck identity is inferred.
            if deck_count is None:
                return 0.0
            return -2.0 if deck_count <= 3 or current_lethal else 1.0
        return 0.0

    if option_type == _legacy.OPT_ATTACH:
        target, _area = _own_in_play_target(obs, option)
        target_id = _legacy._card_id(target)
        if target is None:
            return None
        if source_id == ENRICHING_ENERGY:
            # Enriching's draw-four text is only for an attachment from hand.
            # A future effect that attaches the same card from another zone
            # must not inherit this forced-draw label.
            if not _is_own_zone_target(obs, option, _legacy.AREA_HAND):
                return 0.0
            deck_count = _known_count(me, "deckCount")
            if deck_count is None:
                return 0.0
            # This attachment is C-only and has a forced draw four.  Leaving
            # exactly zero cards is unsafe unless an already-visible terminal
            # line exists, which attachment itself does not establish.
            if deck_count <= 4:
                return -3.5
            psychic_on_line = any(
                _has_psychic_energy(card)
                for card in _legacy._board_cards(dict(me))
                if _legacy._card_id(card) in ALAKAZAM_LINE
            )
            if not psychic_on_line:
                return -2.0
            if current_lethal:
                return -1.0
            return 1.3 if target_id in DUDUNSPARCE_LINE else 0.2
        if source_id == TELEPATH_PSYCHIC_ENERGY:
            # Telepath searches Basics directly to the Bench; it is not a
            # draw-four effect and therefore carries no Enriching draw debit.
            if not _is_own_zone_target(obs, option, _legacy.AREA_HAND):
                return 0.0
            return 1.2 if target_id in ALAKAZAM_LINE else 0.0
        if source_id == PSYCHIC_ENERGY and target_id in ALAKAZAM_LINE:
            return -0.5 if _has_psychic_energy(target) else 1.4
        return 0.0

    if option_type != _legacy.OPT_PLAY:
        return 0.0

    if source_id in {
        BUDDY_BUDDY_POFFIN,
        POKE_PAD,
        RARE_CANDY,
        DAWN,
        HILDA,
        BOSS_ORDERS,
        ENHANCED_HAMMER,
        *RECOVERY_CARDS,
    }:
        return 0.0
    if source_id == XEROSIC:
        # The effect is immediate, but its strategic value is only a public
        # mirror hand compression.  Do not infer an opponent's hidden cards.
        if current_lethal:
            return -1.0
        opponent_ids = {
            _legacy._card_id(card)
            for card in _legacy._board_cards(dict(opp))
        }
        hand_count = _known_count(opp, "handCount")
        if ALAKAZAM in opponent_ids and hand_count is not None and hand_count > 7:
            return min(1.2, 0.2 * (hand_count - 7))
        return 0.0
    if source_id == BATTLE_CAGE:
        return _battle_cage_score(obs, me, opp)
    if source_id == FEZANDIPITI_EX:
        # A bench-only two-prize support card is not a generic setup target.
        # The main stage lacks the public prior-turn KO / exact repair proof.
        return -0.8
    return 0.0


def _effect_belongs_to_me(
    select: Mapping[str, Any], obs: Mapping[str, Any]
) -> bool:
    """Fail only when public context explicitly identifies the opponent."""

    current = obs.get("current") or {}
    your_index = _exact_int(current.get("yourIndex", 0))
    if your_index not in (0, 1):
        return False
    for key in ("effect", "contextCard"):
        effect = select.get(key)
        if isinstance(effect, Mapping) and "playerIndex" in effect:
            owner = _exact_int(effect.get("playerIndex"))
            if owner is not None:
                return owner == your_index
    return True


def _yes_no_score(
    option_type: int, effect_id: Optional[int], me: Mapping[str, Any]
) -> float:
    """Score a visible optional draw only at its actual yes/no stage."""

    draw = {KADABRA: 2, ALAKAZAM: 3, DUDUNSPARCE: 3, FEZANDIPITI_EX: 3}.get(
        effect_id
    )
    if draw is None:
        return 0.0
    deck_count = _known_count(me, "deckCount")
    if deck_count is None:
        return 0.0
    if effect_id == DUDUNSPARCE:
        # Drawing all three and shuffling Dudunsparce back is not a normal
        # draw-to-zero. A one-Pokémon board, however, loses immediately.
        if len(_legacy._board_cards(dict(me))) <= 1 or deck_count == 0:
            unsafe = True
        elif deck_count < draw:
            return 0.0
        else:
            unsafe = False
    else:
        # These draws leave no recycler in the deck. Equal is unsafe because
        # the next mandatory draw has not been discharged.
        unsafe = deck_count <= draw
    if option_type == _legacy.OPT_YES:
        return -6.0 if unsafe else 1.5
    if option_type == _legacy.OPT_NO:
        return 4.0 if unsafe else -0.5
    return 0.0


def _is_own_zone_target(
    obs: Mapping[str, Any], option: Mapping[str, Any], area: int
) -> bool:
    current = obs.get("current") or {}
    your_index = _exact_int(current.get("yourIndex", 0))
    owner = _exact_int(option.get("playerIndex", your_index))
    index = _exact_int(option.get("index"))
    return (
        your_index in (0, 1)
        and owner == your_index
        and _area(option.get("area")) == area
        and index is not None
        and index >= 0
    )


def _boss_target_score(
    obs: dict[str, Any],
    option: Mapping[str, Any],
    target: Any,
    *,
    me: Mapping[str, Any],
    opp: Mapping[str, Any],
) -> float:
    """Reward only a public, target-complete prize-map improvement."""

    if not _is_opponent_zone_target(obs, option, _legacy.AREA_BENCH):
        return 0.0
    if not _active_can_use_powerful_hand(me) or not _powerful_hand_lethal(me, target):
        return 0.0
    target_yield = _prize_yield(target)
    own_prizes = _prize_count(me)
    if target_yield is None or own_prizes is None:
        return 0.0
    if target_yield >= own_prizes:
        return 5.0
    active = _legacy._first(opp.get("active"))
    if not _powerful_hand_lethal(me, active):
        return 3.3
    active_yield = _prize_yield(active)
    if active_yield is not None and target_yield > active_yield:
        return 2.5
    # A same-yield, already-lethal Active is not a prize shortcut.  Do not
    # invent a retreat or future-engine claim from a bare Boss target.
    return 0.0


def _selected_energy(
    host: Any, option: Mapping[str, Any]
) -> tuple[Optional[int], Optional[list[Any]]]:
    if not isinstance(host, Mapping):
        return None, None
    index = _exact_int(option.get("energyIndex"))
    energies = host.get("energyCards")
    if (
        not isinstance(energies, list)
        or index is None
        or index < 0
        or index >= len(energies)
    ):
        return None, None
    return _legacy._card_id(energies[index]), list(energies)


def _hammer_target_score(
    obs: dict[str, Any], option: Mapping[str, Any], *, me: Mapping[str, Any]
) -> float:
    """Score Hammer only when its chosen energy changes this attack now."""

    if not _is_opponent_zone_target(obs, option, _legacy.AREA_ACTIVE):
        return 0.0
    host = _source_card(obs, option)
    energy_id, energies = _selected_energy(host, option)
    if energy_id not in {MIST_ENERGY, ROCK_FIGHTING_ENERGY} or energies is None:
        return 0.0
    protected_before, uncertain_before = _powerful_hand_protection(host)
    if (
        uncertain_before
        or not protected_before
        or not _active_can_use_powerful_hand(me)
        or not _powerful_hand_damage_reaches(me, host)
    ):
        return 0.0
    energy_index = _exact_int(option.get("energyIndex"))
    assert energy_index is not None
    after = dict(host)
    after["energyCards"] = energies[:energy_index] + energies[energy_index + 1 :]
    protected_after, uncertain_after = _powerful_hand_protection(after)
    if protected_after or uncertain_after:
        return 0.0
    return 4.8


def _recovery_target_score(
    obs: dict[str, Any],
    option: Mapping[str, Any],
    card: Any,
    *,
    effect_id: int,
    me: Mapping[str, Any],
) -> float:
    if not _is_own_zone_target(obs, option, _legacy.AREA_DISCARD):
        return 0.0
    card_id = _legacy._card_id(card)
    if card_id not in ALAKAZAM_LINE | {PSYCHIC_ENERGY} or not _recovery_needed(me):
        return 0.0
    board = _legacy._board_counts(dict(me))
    hand = Counter(_legacy._hand_ids(dict(me)))
    if card_id in ALAKAZAM_LINE and board[card_id] + hand[card_id] != 0:
        return 0.0
    if card_id == PSYCHIC_ENERGY and any(
        _has_psychic_energy(candidate)
        for candidate in _legacy._board_cards(dict(me))
        if _legacy._card_id(candidate) in ALAKAZAM_LINE
    ):
        return 0.0
    direct = effect_id in {NIGHT_STRETCHER, LANA_AID}
    base = {ALAKAZAM: 2.1, KADABRA: 1.7, ABRA: 1.3, PSYCHIC_ENERGY: 0.8}.get(
        card_id, 0.0
    )
    # Sacred Ash restores a deck route, not a current-hand line.
    return base if direct else 0.35 * base


def _selection_score(
    obs: dict[str, Any],
    option: Mapping[str, Any],
    card: Any,
    *,
    effect_id: Optional[int],
    me: Mapping[str, Any],
    opp: Mapping[str, Any],
    board: Counter[int],
    chosen: Counter[int],
) -> Optional[float]:
    """Selection-stage targets whose source action already resolved."""

    card_id = _legacy._card_id(card)
    if card_id is None:
        return None
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
            return 1.0 if card_id != ENRICHING_ENERGY else 0.2
        return 0.0
    if effect_id == BOSS_ORDERS:
        return _boss_target_score(obs, option, card, me=me, opp=opp)
    if effect_id == XEROSIC:
        if not _is_own_zone_target(obs, option, _legacy.AREA_HAND):
            return 0.0
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
        return 0.0
    if effect_id in RECOVERY_CARDS:
        return _recovery_target_score(
            obs, option, card, effect_id=effect_id, me=me
        )
    # Rare Candy's selected evolution target is still incomplete until the
    # compulsory Stage-2 selection is observed.
    if effect_id == RARE_CANDY:
        return 0.0
    return _setup_need(card_id, board, chosen)


def _combo_score(
    obs: dict[str, Any],
    combo: Sequence[int],
    *,
    me: Mapping[str, Any],
    opp: Mapping[str, Any],
) -> Optional[float]:
    select = obs.get("select") or {}
    if not isinstance(select, Mapping):
        return None
    options = select.get("option") or []
    if not isinstance(options, list):
        return None
    context = _select_context(select)
    if context is None:
        return None
    effect_id = _legacy._effect_id(dict(select))
    board = _legacy._board_counts(dict(me))
    chosen: Counter[int] = Counter()
    total = 0.0

    for raw_index in combo:
        index = _exact_int(raw_index)
        if index is None or index < 0 or index >= len(options):
            return None
        option = options[index]
        if not isinstance(option, Mapping):
            return None
        option_type = _option_type(option)
        if option_type is None:
            return None
        if context == _legacy.CTX_MAIN:
            score = _main_score(obs, option, me=me, opp=opp)
            if score is None:
                return None
            total += score
            continue
        if option_type in {_legacy.OPT_YES, _legacy.OPT_NO}:
            if not _effect_belongs_to_me(select, obs):
                return None
            total += _yes_no_score(option_type, effect_id, me)
            continue
        if (
            option_type in {_legacy.OPT_ENERGY_CARD, _legacy.OPT_ENERGY}
            and effect_id == ENHANCED_HAMMER
        ):
            total += _hammer_target_score(obs, option, me=me)
            continue

        card = _source_card(obs, option)
        card_id = _legacy._card_id(card)
        if card_id is None:
            return None
        if context == _legacy.CTX_SETUP_ACTIVE:
            total += {ABRA: 2.3, DUNSPARCE: 1.3, FEZANDIPITI_EX: -1.5}.get(
                card_id, 0.0
            )
            continue
        if effect_id is None and context != _legacy.CTX_SETUP_BENCH:
            return None
        score = _selection_score(
            obs,
            option,
            card,
            effect_id=effect_id,
            me=me,
            opp=opp,
            board=board,
            chosen=chosen,
        )
        if score is None:
            return None
        total += score
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
    if (
        not scores
        or any(value is None for value in scores)
        or any(not math.isfinite(float(value)) for value in scores)
    ):
        return None
    resolved_scores = [float(value) for value in scores if value is not None]
    return (
        None
        if max(resolved_scores) - min(resolved_scores) < 0.25
        else resolved_scores
    )


__all__ = [
    "CANONICAL_MULTISET_SHA256",
    "EXACT_DECK",
    "GUIDE_VERSION",
    "OWNER_GUIDE_SHA256",
    "enabled",
    "guide_scores",
    "is_alakazam_new_list_deck",
]
