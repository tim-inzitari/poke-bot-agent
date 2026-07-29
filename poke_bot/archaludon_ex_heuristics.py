"""Sparse, exact-causal guide teacher for Archaludon ex.

The current public shell is Archaludon ex with a Dunsparce/Dudunsparce draw
engine.  The human pilot guide is intentionally much broader than this module.
This teacher scores only complete current legal stages whose prompt origin and
every option can be resolved:

* initial Basic-Pokemon placement;
* exact Fan Call, Buddy-Buddy Poffin, Poke Pad, Ultra Ball, Hilda, and
  Night Stretcher selections; and
* exact Yes/No prompts for Run Away Draw or Flip the Script when the acting
  player's current deck leaves a conservative post-draw reserve and Run Away
  Draw cannot remove their final Pokemon.

Attack selection, Assemble Alloy Energy routing, manual attachments, Prism
Tower and Ultra Ball discards, Scoop Up Cyclone targets, gusts, damage-based
Raging Hammer lines, Mega Mawile prize math, and matchup decisions remain
masked.  They require more state-dependent planning than a sparse heuristic
can safely supply.

Every scored result covers the complete supplied legal-combo stage and carries
stable audit labels.  A malformed, partial, unresolved, unsupported, or
low-margin stage returns ``None``.  No score uses the opponent's hidden hand,
either player's face-down Prizes, future draws, future actions, or eventual
outcomes.  This module creates auxiliary training labels only: it never
selects, executes, or biases a runtime action, and it cannot override the
authoritative fused policy.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional, Sequence


GUIDE_VERSION = "archaludon-ex-north-star-v1"
AUDIT_SCHEMA_VERSION = "poke_bot.causal_deck_guide_audit/v1"

# Internal card identities from cards/EN_Card_Data.csv.  Multiple Duraludon,
# Archaludon, and Dunsparce prints are accepted where they share the relevant
# family role; the current 2026 anchor uses SCR 106 / SSP 130 / JTG 120.
BASIC_METAL_ENERGY = 8

RELICANTH = 57
DUNSPARCE_TEF = 65
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
DURALUDON_SCR = 169
ARCHALUDON_SINGLE_PRIZE_SCR = 170
FAN_ROTOM = 174
ARCHALUDON_EX = 190
DUNSPARCE_JTG = 305
MEGA_MAWILE_EX = 695
DURALUDON_PFL = 839
ARCHALUDON_SINGLE_PRIZE_PFL = 840
DURALUDON_OTHER = 992
MEOWTH_EX = 1071

BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
POKE_PAD = 1152
HILDA = 1225

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_TO_BENCH = 5
CTX_TO_HAND = 7

DURALUDON_IDS = frozenset(
    {DURALUDON_SCR, DURALUDON_PFL, DURALUDON_OTHER}
)
SINGLE_PRIZE_ARCHALUDON_IDS = frozenset(
    {ARCHALUDON_SINGLE_PRIZE_SCR, ARCHALUDON_SINGLE_PRIZE_PFL}
)
DUNSPARCE_IDS = frozenset({DUNSPARCE_TEF, DUNSPARCE_JTG})
ARCHALUDON_LINE = frozenset(
    set(DURALUDON_IDS) | set(SINGLE_PRIZE_ARCHALUDON_IDS) | {ARCHALUDON_EX}
)
DUDUNSPARCE_LINE = frozenset(set(DUNSPARCE_IDS) | {DUDUNSPARCE})

# This is a family gate, not a fuzzy name match.  It accepts current lean and
# Dudunsparce variants while requiring the full Archaludon engine and a real
# Basic Metal allocation.  A concrete guide contract may bind a stricter exact
# 60-card representative checksum on top of this predicate.
ARCHALUDON_EX_MINIMUM = 3
DURALUDON_FAMILY_MINIMUM = 4
BASIC_METAL_MINIMUM = 8

ABSTENTION_MARGIN = 0.25

INITIAL_BASIC_IDS = frozenset(
    set(DURALUDON_IDS)
    | set(DUNSPARCE_IDS)
    | {
        FAN_ROTOM,
        MEOWTH_EX,
        RELICANTH,
        MEGA_MAWILE_EX,
        FEZANDIPITI_EX,
    }
)

SEARCHABLE_CARD_IDS = {
    FAN_ROTOM: frozenset(set(DUNSPARCE_IDS) | {FAN_ROTOM}),
    BUDDY_BUDDY_POFFIN: frozenset(set(DUNSPARCE_IDS) | {FAN_ROTOM}),
    POKE_PAD: frozenset(
        set(DURALUDON_IDS)
        | set(SINGLE_PRIZE_ARCHALUDON_IDS)
        | set(DUNSPARCE_IDS)
        | {DUDUNSPARCE, FAN_ROTOM, RELICANTH}
    ),
    ULTRA_BALL: frozenset(
        set(DURALUDON_IDS)
        | set(SINGLE_PRIZE_ARCHALUDON_IDS)
        | set(DUNSPARCE_IDS)
        | {
            ARCHALUDON_EX,
            DUDUNSPARCE,
            FAN_ROTOM,
            MEOWTH_EX,
            RELICANTH,
            MEGA_MAWILE_EX,
            FEZANDIPITI_EX,
        }
    ),
    HILDA: frozenset(
        set(SINGLE_PRIZE_ARCHALUDON_IDS)
        | {ARCHALUDON_EX, DUDUNSPARCE, BASIC_METAL_ENERGY}
    ),
    NIGHT_STRETCHER: frozenset(
        set(DURALUDON_IDS)
        | set(SINGLE_PRIZE_ARCHALUDON_IDS)
        | set(DUNSPARCE_IDS)
        | {
            ARCHALUDON_EX,
            DUDUNSPARCE,
            FAN_ROTOM,
            MEOWTH_EX,
            RELICANTH,
            MEGA_MAWILE_EX,
            FEZANDIPITI_EX,
            BASIC_METAL_ENERGY,
        }
    ),
}

SEARCH_CONTEXTS = {
    FAN_ROTOM: CTX_TO_HAND,
    BUDDY_BUDDY_POFFIN: CTX_TO_BENCH,
    POKE_PAD: CTX_TO_HAND,
    ULTRA_BALL: CTX_TO_HAND,
    HILDA: CTX_TO_HAND,
    NIGHT_STRETCHER: CTX_TO_HAND,
}

SEARCH_AREAS = {
    FAN_ROTOM: AREA_DECK,
    BUDDY_BUDDY_POFFIN: AREA_DECK,
    POKE_PAD: AREA_DECK,
    ULTRA_BALL: AREA_DECK,
    HILDA: AREA_DECK,
    NIGHT_STRETCHER: AREA_DISCARD,
}

LABEL_DESCRIPTIONS = {
    "opening_active_duraludon": (
        "Prefer a current Duraludon as the evolvable main attacker."
    ),
    "opening_active_dunsparce_pivot": (
        "Use a single-Prize Dunsparce as the secondary safe opener."
    ),
    "opening_active_fan_rotom": (
        "Fan Rotom is a functional first-turn setup opener, below the main line."
    ),
    "opening_active_support_liability": (
        "Avoid volunteering a dedicated support or multi-Prize Basic Active."
    ),
    "bench_redundant_duraludon": (
        "Establish two visible Duraludon before adding a redundant third."
    ),
    "bench_dudunsparce_engine": (
        "Establish up to two visible Dunsparce for the Run Away Draw loop."
    ),
    "bench_first_fan_rotom": (
        "A first Fan Rotom supplies the exact first-turn Fan Call setup role."
    ),
    "bench_relicanth_optional": (
        "Relicanth is useful only after the attacker and draw engine exist."
    ),
    "bench_multi_prize_support_liability": (
        "Do not default-bench an optional two- or three-Prize support attacker."
    ),
    "search_archaludon_attacker": (
        "Fill a visible Duraludon-to-Archaludon ex evolution gap."
    ),
    "search_replacement_duraludon": (
        "Preserve a second visible Duraludon attacker lane."
    ),
    "search_dudunsparce_draw": (
        "Fill a visible Dunsparce-to-Dudunsparce draw-engine gap."
    ),
    "search_dunsparce_engine": (
        "Establish the first two Dunsparce engine bodies."
    ),
    "search_single_prize_archaludon": (
        "Single-Prize Archaludon is a lower-priority wall matchup branch."
    ),
    "search_relicanth_route": (
        "Relicanth unlocks previous-Evolution attacks after the core is set."
    ),
    "search_late_mega_mawile": (
        "Mega Mawile becomes a search branch only after taking at least 3 Prizes."
    ),
    "search_optional_rule_box_support": (
        "Optional Rule Box support is neutral without a proven immediate need."
    ),
    "search_basic_metal": (
        "Hilda or Night Stretcher can recover a visible Basic Metal requirement."
    ),
    "run_away_draw_safe_yes": (
        "Use Run Away Draw with another Pokemon in play and a post-draw reserve."
    ),
    "run_away_draw_unsafe_yes": (
        "Penalize Run Away Draw with no other Pokemon or no conservative reserve."
    ),
    "run_away_draw_safe_no": (
        "Declining a currently safe Run Away Draw gives up exact visible draw value."
    ),
    "run_away_draw_unsafe_no": (
        "Decline Run Away Draw with no other Pokemon or no conservative reserve."
    ),
    "flip_the_script_safe_yes": (
        "Use the already-legal draw-three ability with a post-draw reserve."
    ),
    "flip_the_script_unsafe_yes": (
        "Penalize Flip the Script when three safely drawable cards are unavailable."
    ),
    "flip_the_script_safe_no": (
        "Declining a currently safe Flip the Script gives up exact visible draw value."
    ),
    "flip_the_script_unsafe_no": (
        "Decline the draw-three ability when three cards are not safely available."
    ),
    "supported_neutral": (
        "The option is fully resolved but has no high-confidence sparse preference."
    ),
}


def enabled() -> bool:
    """Return whether the generic current-deck registry selected this guide."""
    from . import deck_guides

    return (
        deck_guides.enabled()
        and deck_guides.selected_id() == "archaludon-ex"
    )


def _exact_int(value: Any) -> Optional[int]:
    """Accept actual integral values but reject coercive strings/floats/bools."""
    if isinstance(value, (bool, float, str)):
        return None
    try:
        result = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if value == result else None


def _card_id(card: Any) -> Optional[int]:
    value = card.get("id") if isinstance(card, dict) else getattr(card, "id", None)
    return _exact_int(value)


def _cards(zone: Any) -> list[Any]:
    return list(zone) if isinstance(zone, (list, tuple)) else []


def is_archaludon_ex_deck(deck: Iterable[int]) -> bool:
    """Require an exact 60-card list with a real Archaludon ex engine."""
    try:
        card_ids = [_exact_int(card_id) for card_id in deck]
    except TypeError:
        return False
    if len(card_ids) != 60 or any(card_id is None for card_id in card_ids):
        return False
    counts = Counter(card_ids)
    duraludon_count = sum(counts[card_id] for card_id in DURALUDON_IDS)
    return (
        counts[ARCHALUDON_EX] >= ARCHALUDON_EX_MINIMUM
        and duraludon_count >= DURALUDON_FAMILY_MINIMUM
        and counts[BASIC_METAL_ENERGY] >= BASIC_METAL_MINIMUM
    )


def applies(deck_card_ids: Iterable[int]) -> bool:
    """Compatibility predicate for guide dispatch and replay featurization."""
    return is_archaludon_ex_deck(deck_card_ids)


def prior_logit_bias(
    obs: Any,
    action_combos: Sequence[Sequence[int]],
    *,
    scale: float = 1.0,
) -> list[float]:
    """Remain an exact runtime bypass; the teacher has no serving authority."""
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


def _board_cards(player: dict) -> list[Any]:
    return _cards(player.get("active")) + _cards(player.get("bench"))


def _board_counts(player: dict) -> Counter[int]:
    return Counter(
        card_id
        for card_id in (_card_id(card) for card in _board_cards(player))
        if card_id is not None
    )


def _hand_counts(player: dict) -> Counter[int]:
    return Counter(_zone_ids(player, "hand"))


def _deck_count(player: dict) -> Optional[int]:
    value = _exact_int(player.get("deckCount"))
    if value is not None and value >= 0:
        return value
    deck = player.get("deck")
    return len(deck) if isinstance(deck, (list, tuple)) else None


def _remaining_prizes(player: dict) -> Optional[int]:
    prize = player.get("prize")
    if isinstance(prize, (list, tuple)):
        count = len(prize)
        return count if 0 <= count <= 6 else None
    for key in ("prizeCount", "remainingPrizes"):
        value = _exact_int(player.get(key))
        if value is not None and 0 <= value <= 6:
            return value
    return None


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
        zone = select.get("deck")
    elif area_i == AREA_STADIUM:
        zone = current.get("stadium")
    elif area_i == AREA_LOOKING:
        zone = current.get("looking")
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
        zone = player.get(key)
    if not isinstance(zone, (list, tuple)):
        return None
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
        return _resolve_card(
            obs,
            area=AREA_HAND,
            index=option.get("index"),
            player_index=seat,
        )
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


def _validated_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
) -> bool:
    """Require one exact factorized legal-action stage, including STOP."""
    select = obs.get("select")
    if not isinstance(select, dict):
        return False
    options = select.get("option")
    if not isinstance(options, list) or not options or not all(
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
    normalized: list[tuple[int, ...]] = []
    for combo in action_combos:
        if not isinstance(combo, (list, tuple)):
            return False
        indices = [_exact_int(index) for index in combo]
        if any(index is None for index in indices):
            return False
        exact_indices = [int(index) for index in indices if index is not None]
        canonical = tuple(exact_indices)
        if (
            len(exact_indices) > maximum
            or len(exact_indices) != len(set(exact_indices))
            or any(index < 0 or index >= len(options) for index in exact_indices)
        ):
            return False
        normalized.append(canonical)
    if len(normalized) != len(set(normalized)):
        return False

    # ``features.factorized_teacher_forcing_stages`` presents every possible
    # next choice for one shared ordered prefix, plus STOP once ``minCount`` is
    # met.  Reconstruct that exact candidate set instead of accepting a
    # caller-selected subset.
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


def _family_count(counts: Counter[int], family: frozenset[int]) -> int:
    return sum(counts[card_id] for card_id in family)


def _opening_score(
    card_id: int,
    *,
    context: int,
    board: Counter[int],
    chosen: Counter[int],
) -> tuple[float, tuple[str, ...]]:
    if context == CTX_SETUP_ACTIVE:
        if card_id in DURALUDON_IDS:
            return 3.2, ("opening_active_duraludon",)
        if card_id in DUNSPARCE_IDS:
            return 2.4, ("opening_active_dunsparce_pivot",)
        if card_id == FAN_ROTOM:
            return 1.4, ("opening_active_fan_rotom",)
        return -1.2, ("opening_active_support_liability",)

    if card_id in DURALUDON_IDS:
        have = _family_count(board + chosen, DURALUDON_IDS)
        return (
            (3.4 if have < 2 else 0.2),
            ("bench_redundant_duraludon",),
        )
    if card_id in DUNSPARCE_IDS:
        have = _family_count(board + chosen, DUNSPARCE_IDS)
        return (
            (2.8 if have < 2 else 0.3),
            ("bench_dudunsparce_engine",),
        )
    if card_id == FAN_ROTOM:
        have = board[FAN_ROTOM] + chosen[FAN_ROTOM]
        return (
            (1.8 if have == 0 else -0.2),
            ("bench_first_fan_rotom",),
        )
    if card_id == RELICANTH:
        return 0.7, ("bench_relicanth_optional",)
    return -1.0, ("bench_multi_prize_support_liability",)


def _search_score(
    card_id: int,
    *,
    effect_id: int,
    board: Counter[int],
    hand: Counter[int],
    chosen: Counter[int],
    remaining_prizes: Optional[int],
) -> tuple[float, tuple[str, ...]]:
    visible = board + hand + chosen

    if effect_id in {FAN_ROTOM, BUDDY_BUDDY_POFFIN}:
        if card_id in DUNSPARCE_IDS:
            have = _family_count(visible, DUNSPARCE_IDS)
            return (
                (3.0 if have < 2 else 0.3),
                ("search_dunsparce_engine",),
            )
        if card_id == FAN_ROTOM:
            have = board[FAN_ROTOM] + chosen[FAN_ROTOM]
            return (
                (1.2 if have == 0 else -0.3),
                ("bench_first_fan_rotom",),
            )

    if card_id == ARCHALUDON_EX:
        open_lines = _family_count(board, DURALUDON_IDS)
        have = hand[ARCHALUDON_EX] + chosen[ARCHALUDON_EX]
        return (
            (3.6 if open_lines > have else 0.8),
            ("search_archaludon_attacker",),
        )

    if card_id in DURALUDON_IDS:
        have = _family_count(visible, DURALUDON_IDS)
        return (
            (3.1 if have < 2 else 0.4),
            ("search_replacement_duraludon",),
        )

    if card_id == DUDUNSPARCE:
        open_lines = _family_count(board, DUNSPARCE_IDS)
        have = hand[DUDUNSPARCE] + chosen[DUDUNSPARCE]
        return (
            (2.8 if open_lines > have else 0.4),
            ("search_dudunsparce_draw",),
        )

    if card_id in DUNSPARCE_IDS:
        have = _family_count(visible, DUNSPARCE_IDS)
        return (
            (2.5 if have < 2 else 0.3),
            ("search_dunsparce_engine",),
        )

    if card_id in SINGLE_PRIZE_ARCHALUDON_IDS:
        open_line = _family_count(board, DURALUDON_IDS) > 0
        return (
            (1.4 if open_line else 0.2),
            ("search_single_prize_archaludon",),
        )

    if card_id == RELICANTH:
        useful = board[ARCHALUDON_EX] > 0 and board[RELICANTH] == 0
        return (
            (1.5 if useful else 0.3),
            ("search_relicanth_route",),
        )

    if card_id == MEGA_MAWILE_EX:
        prizes_taken = (
            None if remaining_prizes is None else 6 - remaining_prizes
        )
        return (
            (2.0 if prizes_taken is not None and prizes_taken >= 3 else -0.8),
            ("search_late_mega_mawile",),
        )

    if card_id == BASIC_METAL_ENERGY:
        return 1.8, ("search_basic_metal",)

    if card_id in {MEOWTH_EX, FEZANDIPITI_EX, FAN_ROTOM}:
        return 0.0, ("search_optional_rule_box_support",)

    return 0.0, ("supported_neutral",)


def _yes_no_score(
    option_type: int,
    *,
    effect_id: int,
    me: dict,
) -> Optional[tuple[float, tuple[str, ...]]]:
    if option_type not in {OPT_YES, OPT_NO}:
        return None
    deck_count = _deck_count(me)
    if deck_count is None:
        return None

    if effect_id == DUDUNSPARCE:
        safe = deck_count > 3 and len(_board_cards(me)) > 1
        if option_type == OPT_YES:
            label = (
                "run_away_draw_safe_yes"
                if safe
                else "run_away_draw_unsafe_yes"
            )
            return (2.6 if safe else -7.0), (label,)
        label = (
            "run_away_draw_safe_no"
            if safe
            else "run_away_draw_unsafe_no"
        )
        return (-1.0 if safe else 4.5), (label,)

    if effect_id == FEZANDIPITI_EX:
        safe = deck_count > 3
        if option_type == OPT_YES:
            label = (
                "flip_the_script_safe_yes"
                if safe
                else "flip_the_script_unsafe_yes"
            )
            return (2.2 if safe else -7.0), (label,)
        label = (
            "flip_the_script_safe_no"
            if safe
            else "flip_the_script_unsafe_no"
        )
        return (-0.8 if safe else 4.5), (label,)

    return None


def _score_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool,
) -> Optional[dict[str, Any]]:
    if (
        (not force_enabled and not enabled())
        or not is_archaludon_ex_deck(deck)
        or not isinstance(obs, dict)
        or not action_combos
        or not _validated_stage(obs, action_combos)
    ):
        return None
    me, opponent = _players(obs)
    if me is None or opponent is None:
        return None

    select = obs.get("select") or {}
    options = select.get("option") or []
    context = _exact_int(select.get("context"))
    effect_id = _effect_id(select)
    board = _board_counts(me)
    hand = _hand_counts(me)
    remaining_prizes = _remaining_prizes(me)

    if context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
        if effect_id is not None:
            return None
        stage_class = (
            "initial_setup_active"
            if context == CTX_SETUP_ACTIVE
            else "initial_setup_bench"
        )
        allowed_ids = INITIAL_BASIC_IDS
        causal_inputs = (
            "select.context",
            "complete_current_legal_combos",
            "resolved_own_hand_card_ids",
            "own_public_active_and_bench_card_ids",
        )
    elif (
        effect_id in SEARCHABLE_CARD_IDS
        and context == SEARCH_CONTEXTS[effect_id]
    ):
        stage_class = {
            FAN_ROTOM: "fan_call_search",
            BUDDY_BUDDY_POFFIN: "buddy_buddy_poffin_search",
            POKE_PAD: "poke_pad_search",
            ULTRA_BALL: "ultra_ball_search",
            HILDA: "hilda_search",
            NIGHT_STRETCHER: "night_stretcher_recovery",
        }[effect_id]
        allowed_ids = SEARCHABLE_CARD_IDS[effect_id]
        causal_inputs = (
            "select.effect_or_contextCard",
            "complete_current_legal_combos",
            "resolved_current_prompt_card_ids",
            "own_current_active_bench_hand_counts",
            "own_remaining_prize_count_when_exposed",
        )
    elif effect_id in {DUDUNSPARCE, FEZANDIPITI_EX}:
        stage_class = (
            "run_away_draw_yes_no"
            if effect_id == DUDUNSPARCE
            else "flip_the_script_yes_no"
        )
        allowed_ids = frozenset()
        causal_inputs = (
            "select.effect_or_contextCard",
            "complete_current_yes_no_stage",
            "own_current_deck_count",
            "own_current_active_and_bench_count",
        )
    else:
        return None

    scores: list[float] = []
    audited_combos: list[dict[str, Any]] = []
    for combo in action_combos:
        chosen: Counter[int] = Counter()
        score = 0.0
        labels: list[str] = []
        resolved_ids: list[int] = []
        for raw_index in combo:
            index = _exact_int(raw_index)
            if index is None:
                return None
            option = options[index]
            option_type = _exact_int(option.get("type"))
            if option_type is None:
                return None

            if effect_id in {DUDUNSPARCE, FEZANDIPITI_EX}:
                row = _yes_no_score(
                    option_type,
                    effect_id=effect_id,
                    me=me,
                )
                if row is None:
                    return None
                value, row_labels = row
                score += value
                labels.extend(row_labels)
                continue

            expected_area = (
                AREA_HAND
                if context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}
                else SEARCH_AREAS[effect_id]
            )
            allowed_types = (
                {OPT_CARD}
                if context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}
                else {OPT_CARD, OPT_ENERGY_CARD}
            )
            if (
                option_type not in allowed_types
                or _exact_int(option.get("area")) != expected_area
            ):
                return None
            card_id = _card_id(_option_card(obs, option))
            if card_id is None or card_id not in allowed_ids:
                return None
            resolved_ids.append(card_id)
            if context in {CTX_SETUP_ACTIVE, CTX_SETUP_BENCH}:
                value, row_labels = _opening_score(
                    card_id,
                    context=context,
                    board=board,
                    chosen=chosen,
                )
            else:
                assert effect_id is not None
                value, row_labels = _search_score(
                    card_id,
                    effect_id=effect_id,
                    board=board,
                    hand=hand,
                    chosen=chosen,
                    remaining_prizes=remaining_prizes,
                )
            score += value
            labels.extend(row_labels)
            chosen[card_id] += 1

        bounded_score = max(-8.0, min(8.0, float(score)))
        if not math.isfinite(bounded_score):
            return None
        scores.append(bounded_score)
        audited_combos.append(
            {
                "combo": [int(index) for index in combo],
                "resolved_card_ids": resolved_ids,
                "score": bounded_score,
                "label_ids": sorted(set(labels or ["supported_neutral"])),
            }
        )

    if (
        len(scores) != len(action_combos)
        or max(scores) - min(scores) < ABSTENTION_MARGIN
    ):
        return None
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "guide_version": GUIDE_VERSION,
        "specialist_id": "archaludon-ex",
        "stage_class": stage_class,
        "causal_inputs": list(causal_inputs),
        "scores": scores,
        "combo_labels": audited_combos,
        "abstention_margin": ABSTENTION_MARGIN,
        "runtime_authority": "none",
    }


def guide_audit(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[dict[str, Any]]:
    """Return scores plus stable causal labels, or mask the complete stage."""
    return _score_stage(
        obs,
        action_combos,
        deck=deck,
        force_enabled=force_enabled,
    )


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> Optional[list[float]]:
    """Return aligned scores for a complete legal stage, else ``None``."""
    audit = guide_audit(
        obs,
        action_combos,
        deck=deck,
        force_enabled=force_enabled,
    )
    return None if audit is None else list(audit["scores"])


def describe() -> str:
    return (
        f"ArchaludonExGuide(version={GUIDE_VERSION}, "
        f"duraludon_min={DURALUDON_FAMILY_MINIMUM}, "
        f"archaludon_ex_min={ARCHALUDON_EX_MINIMUM}, "
        f"metal_min={BASIC_METAL_MINIMUM})"
    )


__all__ = [
    "ABSTENTION_MARGIN",
    "ARCHALUDON_EX",
    "AUDIT_SCHEMA_VERSION",
    "BASIC_METAL_ENERGY",
    "BUDDY_BUDDY_POFFIN",
    "CTX_MAIN",
    "CTX_SETUP_ACTIVE",
    "CTX_SETUP_BENCH",
    "CTX_TO_BENCH",
    "CTX_TO_HAND",
    "DUDUNSPARCE",
    "DUNSPARCE_JTG",
    "DURALUDON_SCR",
    "FAN_ROTOM",
    "FEZANDIPITI_EX",
    "GUIDE_VERSION",
    "HILDA",
    "LABEL_DESCRIPTIONS",
    "MEGA_MAWILE_EX",
    "MEOWTH_EX",
    "NIGHT_STRETCHER",
    "OPT_CARD",
    "OPT_NO",
    "OPT_YES",
    "POKE_PAD",
    "RELICANTH",
    "ULTRA_BALL",
    "applies",
    "describe",
    "enabled",
    "guide_audit",
    "guide_scores",
    "is_archaludon_ex_deck",
    "prior_logit_bias",
]
