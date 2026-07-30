"""Sparse exact-causal training guide for straight Dragapult ex.

This inactive future-specialist guide deliberately excludes Hammer Pult,
Dragapult/Blaziken, Dragapult/Dudunsparce, and Dragapult/Dusknoir. It can label
only:

* complete initial Basic-Pokemon placement stages;
* complete, origin-resolved Buddy-Buddy Poffin, Poke Pad, Ultra Ball, and
  Night Stretcher selection stages; and
* complete, origin-resolved Flip the Script Yes/No stages with a conservative
  current deck-count reserve.

Typed-Energy routing, manual attachment, attack choice, Phantom Dive counter
placement, Adrena-Brain routing, Recon Directive, Ultra Ball discards, gust,
Meowth Supporter search, stadium/bench collapse, matchup lines, and every
hidden- or future-dependent choice are unsupported.  An unsupported,
ambiguous, malformed, partial, or low-margin stage returns ``None``: missing
labels are masked, never converted to zero targets.

The module never selects an action and ``prior_logit_bias`` is an exact zero
bypass. Production activation still requires replay-schema validation,
checksum-bound corpus construction, pre-stage validation, and a safe handoff.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

GUIDE_VERSION = "dragapult-north-star-v1"
AUDIT_SCHEMA_VERSION = "poke_bot.causal_deck_guide_audit/v1"

BASIC_FIRE_ENERGY = 2
BASIC_PSYCHIC_ENERGY = 5
BASIC_DARKNESS_ENERGY = 7

MUNKIDORI = 112
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
FEZANDIPITI_EX = 140
PECHARUNT_EX = 141
BUDEW = 235
LILLIES_CLEFAIRY_EX = 272
YVELTAL = 689
MOLTRES = 791
MEOWTH_EX = 1071

BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
CRUSHING_HAMMER = 1120
ULTRA_BALL = 1121
POKE_PAD = 1152

DUNSPARCE_FAMILY = frozenset({65, 66, 305, 306})
DUSKNOIR_FAMILY = frozenset({131, 132, 133})
BLAZIKEN_FAMILY = frozenset({324, 325, 326, 410, 411, 412})

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

ABSTENTION_MARGIN = 0.25

PLAIN_DRAGAPULT_MINIMUM = 2
DREEPY_MINIMUM = 4
DRAKLOAK_MINIMUM = 4
CRUSHING_HAMMER_FORBIDDEN_MINIMUM = 3

INITIAL_BASIC_IDS = frozenset(
    {
        DREEPY,
        MUNKIDORI,
        FEZANDIPITI_EX,
        PECHARUNT_EX,
        BUDEW,
        LILLIES_CLEFAIRY_EX,
        YVELTAL,
        MOLTRES,
        MEOWTH_EX,
    }
)

SEARCHABLE_CARD_IDS = {
    BUDDY_BUDDY_POFFIN: frozenset({DREEPY, BUDEW}),
    POKE_PAD: frozenset({DREEPY, DRAKLOAK, MUNKIDORI, BUDEW, YVELTAL, MOLTRES}),
    ULTRA_BALL: frozenset(
        {
            DREEPY,
            DRAKLOAK,
            DRAGAPULT_EX,
            MUNKIDORI,
            FEZANDIPITI_EX,
            PECHARUNT_EX,
            BUDEW,
            LILLIES_CLEFAIRY_EX,
            YVELTAL,
            MOLTRES,
            MEOWTH_EX,
        }
    ),
    NIGHT_STRETCHER: frozenset(
        {
            DREEPY,
            DRAKLOAK,
            DRAGAPULT_EX,
            MUNKIDORI,
            FEZANDIPITI_EX,
            PECHARUNT_EX,
            BUDEW,
            LILLIES_CLEFAIRY_EX,
            YVELTAL,
            MOLTRES,
            MEOWTH_EX,
            BASIC_FIRE_ENERGY,
            BASIC_PSYCHIC_ENERGY,
            BASIC_DARKNESS_ENERGY,
        }
    ),
}

SEARCH_CONTEXTS = {
    BUDDY_BUDDY_POFFIN: CTX_TO_BENCH,
    POKE_PAD: CTX_TO_HAND,
    ULTRA_BALL: CTX_TO_HAND,
    NIGHT_STRETCHER: CTX_TO_HAND,
}

SEARCH_AREAS = {
    BUDDY_BUDDY_POFFIN: AREA_DECK,
    POKE_PAD: AREA_DECK,
    ULTRA_BALL: AREA_DECK,
    NIGHT_STRETCHER: AREA_DISCARD,
}

LABEL_DESCRIPTIONS = {
    "opening_active_dreepy": "Prefer the evolvable main line as a functional opener.",
    "opening_active_budew": "Budew is the preferred alternate single-Prize opener.",
    "opening_active_single_prize_pivot": "A single-Prize toolbox Basic is safer than a Rule Box liability.",
    "opening_active_rule_box_liability": "Avoid volunteering an optional two-Prize support Basic.",
    "bench_redundant_dreepy": "Establish two Dreepy before optional toolbox support.",
    "bench_first_budew": "A first Budew is a bounded setup option after the Dreepy core.",
    "bench_single_prize_toolbox": "A single-Prize toolbox Basic is secondary to two Dreepy.",
    "bench_rule_box_liability": "Do not default-bench optional Rule Box support.",
    "search_dreepy_redundancy": "Fill the visible Dreepy redundancy gap.",
    "search_drakloak_bridge": "Fill a visible Dreepy-to-Drakloak evolution gap.",
    "search_dragapult_attacker": "Fill a visible Drakloak-to-Dragapult ex evolution gap.",
    "search_first_budew": "A first Budew is useful only after core Dreepy coverage.",
    "search_single_prize_toolbox": "Single-Prize toolbox options are lower-priority without a proved immediate role.",
    "search_rule_box_support": "Rule Box support remains neutral without a current exact ability need.",
    "search_typed_energy_unsupported": "Typed Energy recovery requires attachment and attacker routing and is masked.",
    "flip_the_script_safe_yes": "Use the already-legal draw-three ability with a post-draw reserve.",
    "flip_the_script_unsafe_yes": "Penalize drawing when three cards cannot be safely drawn.",
    "flip_the_script_safe_no": "Declining a currently safe draw gives up exact visible value.",
    "flip_the_script_unsafe_no": "Decline when a conservative three-card reserve is unavailable.",
    "supported_neutral": "Resolved option with no high-confidence sparse preference.",
}


def enabled() -> bool:
    """Return true only if the training-only guide registry selects Dragapult."""
    try:
        from . import deck_guides
    except ImportError:
        return False
    return deck_guides.enabled() and deck_guides.selected_id() == "dragapult"


def _exact_int(value: Any) -> int | None:
    if isinstance(value, (bool, float, str)):
        return None
    try:
        result = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if value == result else None


def _card_id(card: Any) -> int | None:
    value = card.get("id") if isinstance(card, dict) else getattr(card, "id", None)
    return _exact_int(value)


def _cards(zone: Any) -> list[Any]:
    return list(zone) if isinstance(zone, (list, tuple)) else []


def is_plain_dragapult_deck(deck: Iterable[int]) -> bool:
    """Require the plain Dragapult engine and reject every named variant."""
    try:
        card_ids = [_exact_int(card_id) for card_id in deck]
    except TypeError:
        return False
    if len(card_ids) != 60 or any(card_id is None for card_id in card_ids):
        return False
    counts = Counter(card_ids)
    if (
        counts[DREEPY] < DREEPY_MINIMUM
        or counts[DRAKLOAK] < DRAKLOAK_MINIMUM
        or counts[DRAGAPULT_EX] < PLAIN_DRAGAPULT_MINIMUM
    ):
        return False
    if counts[CRUSHING_HAMMER] >= CRUSHING_HAMMER_FORBIDDEN_MINIMUM:
        return False
    return not any(
        counts[card_id]
        for family in (DUNSPARCE_FAMILY, DUSKNOIR_FAMILY, BLAZIKEN_FAMILY)
        for card_id in family
    )


def applies(deck_card_ids: Iterable[int]) -> bool:
    return is_plain_dragapult_deck(deck_card_ids)


def prior_logit_bias(
    obs: Any,
    action_combos: Sequence[Sequence[int]],
    *,
    scale: float = 1.0,
) -> list[float]:
    """Exact serving bypass: this research proposal has no action authority."""
    del obs, scale
    return [0.0] * len(action_combos)


def _players(obs: dict[str, Any]) -> tuple[dict | None, dict | None]:
    current = obs.get("current") if isinstance(obs, dict) else None
    players = current.get("players") if isinstance(current, dict) else None
    seat = _exact_int(current.get("yourIndex", 0)) if current else None
    if seat not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        return None, None
    if not isinstance(players[seat], dict) or not isinstance(players[1 - seat], dict):
        return None, None
    return players[seat], players[1 - seat]


def _board_cards(player: dict) -> list[Any]:
    return _cards(player.get("active")) + _cards(player.get("bench"))


def _board_counts(player: dict) -> Counter[int]:
    return Counter(
        card_id
        for card_id in (_card_id(card) for card in _board_cards(player))
        if card_id is not None
    )


def _deck_count(player: dict) -> int | None:
    value = _exact_int(player.get("deckCount"))
    if value is not None and value >= 0:
        return value
    deck = player.get("deck")
    return len(deck) if isinstance(deck, (list, tuple)) else None


def _resolve_card(
    obs: dict[str, Any],
    *,
    area: Any,
    index: Any,
    player_index: Any = None,
) -> Any | None:
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


def _option_card(obs: dict[str, Any], option: dict[str, Any]) -> Any | None:
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


def _effect_id(select: dict[str, Any]) -> int | None:
    for key in ("effect", "contextCard"):
        value = _card_id(select.get(key))
        if value is not None:
            return value
    return None


def _validated_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
) -> bool:
    """Require the complete factorized candidate set for one legal stage."""
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
        exact = [int(index) for index in indices if index is not None]
        canonical = tuple(exact)
        if (
            len(exact) > maximum
            or len(exact) != len(set(exact))
            or any(index < 0 or index >= len(options) for index in exact)
        ):
            return False
        normalized.append(canonical)
    if len(normalized) != len(set(normalized)):
        return False

    supplied = set(normalized)
    first = normalized[0]
    for prefix_length in range(len(first) + 1):
        prefix = first[:prefix_length]
        if len(prefix) > maximum or len(prefix) != len(set(prefix)):
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


def _opening_score(
    card_id: int,
    *,
    context: int,
    board: Counter[int],
    chosen: Counter[int],
) -> tuple[float, tuple[str, ...]]:
    if context == CTX_SETUP_ACTIVE:
        if card_id == DREEPY:
            return 3.0, ("opening_active_dreepy",)
        if card_id == BUDEW:
            return 2.4, ("opening_active_budew",)
        if card_id in {MUNKIDORI, YVELTAL, MOLTRES}:
            return 1.0, ("opening_active_single_prize_pivot",)
        return -1.2, ("opening_active_rule_box_liability",)

    if card_id == DREEPY:
        have = board[DREEPY] + chosen[DREEPY]
        return (3.5 if have < 2 else 0.4), ("bench_redundant_dreepy",)
    if card_id == BUDEW:
        have = board[BUDEW] + chosen[BUDEW]
        return (1.4 if have == 0 else -0.2), ("bench_first_budew",)
    if card_id in {MUNKIDORI, YVELTAL, MOLTRES}:
        return 0.3, ("bench_single_prize_toolbox",)
    return -1.0, ("bench_rule_box_liability",)


def _search_score(
    card_id: int,
    *,
    effect_id: int,
    board: Counter[int],
    chosen: Counter[int],
) -> tuple[float, tuple[str, ...]] | None:
    visible = board + chosen
    dreepy_count = visible[DREEPY]
    drakloak_count = visible[DRAKLOAK]

    if card_id in {
        BASIC_FIRE_ENERGY,
        BASIC_PSYCHIC_ENERGY,
        BASIC_DARKNESS_ENERGY,
    }:
        # A correct Energy recovery decision needs exact attachment identities,
        # turn attachment status, attacker/pivot route, and typed Energy plan.
        return None
    if card_id == DREEPY:
        return (
            (3.5 if dreepy_count < 2 else 0.4),
            ("search_dreepy_redundancy",),
        )
    if card_id == DRAKLOAK:
        # Board zones expose only the current top card of each evolution
        # stack.  Every visible Dreepy is therefore its own open Stage-1 line;
        # a Drakloak elsewhere does not consume it.
        open_lines = dreepy_count
        return (
            (3.2 if open_lines > 0 else 0.3),
            ("search_drakloak_bridge",),
        )
    if card_id == DRAGAPULT_EX:
        # For the same reason, every visible Drakloak is an open Stage-2 line.
        open_lines = drakloak_count
        return (
            (3.4 if open_lines > 0 else 0.4),
            ("search_dragapult_attacker",),
        )
    if card_id == BUDEW:
        return (
            (1.0 if board[BUDEW] + chosen[BUDEW] == 0 else -0.2),
            ("search_first_budew",),
        )
    if card_id in {MUNKIDORI, YVELTAL, MOLTRES}:
        return 0.0, ("search_single_prize_toolbox",)
    if card_id in {
        FEZANDIPITI_EX,
        PECHARUNT_EX,
        LILLIES_CLEFAIRY_EX,
        MEOWTH_EX,
    }:
        return 0.0, ("search_rule_box_support",)
    return 0.0, ("supported_neutral",)


def _yes_no_score(
    option_type: int,
    *,
    me: dict,
) -> tuple[float, tuple[str, ...]] | None:
    if option_type not in {OPT_YES, OPT_NO}:
        return None
    deck_count = _deck_count(me)
    if deck_count is None:
        return None
    safe = deck_count > 3
    if option_type == OPT_YES:
        return (
            (2.2 if safe else -7.0),
            (
                "flip_the_script_safe_yes"
                if safe
                else "flip_the_script_unsafe_yes",
            ),
        )
    return (
        (-0.8 if safe else 4.5),
        (
            "flip_the_script_safe_no"
            if safe
            else "flip_the_script_unsafe_no",
        ),
    )


def _score_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool,
) -> dict[str, Any] | None:
    if (
        (not force_enabled and not enabled())
        or not is_plain_dragapult_deck(deck)
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
            "resolved_current_option_card_ids",
            "own_current_active_and_bench_card_ids",
        )
    elif effect_id in SEARCHABLE_CARD_IDS and context == SEARCH_CONTEXTS[effect_id]:
        stage_class = {
            BUDDY_BUDDY_POFFIN: "buddy_buddy_poffin_search",
            POKE_PAD: "poke_pad_search",
            ULTRA_BALL: "ultra_ball_search",
            NIGHT_STRETCHER: "night_stretcher_recovery",
        }[effect_id]
        allowed_ids = SEARCHABLE_CARD_IDS[effect_id]
        causal_inputs = (
            "select.effect_or_contextCard",
            "complete_current_legal_combos",
            "resolved_current_prompt_card_ids",
            "own_current_active_and_bench_card_ids",
        )
    elif effect_id == FEZANDIPITI_EX:
        stage_class = "flip_the_script_yes_no"
        allowed_ids = frozenset()
        causal_inputs = (
            "select.effect_or_contextCard",
            "complete_current_yes_no_stage",
            "own_current_deck_count",
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

            if effect_id == FEZANDIPITI_EX:
                row = _yes_no_score(option_type, me=me)
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
                row = _search_score(
                    card_id,
                    effect_id=effect_id,
                    board=board,
                    chosen=chosen,
                )
                # A single unsupported option masks the complete stage.  This
                # is particularly important for typed-Energy recovery.
                if row is None:
                    return None
                value, row_labels = row
            score += value
            labels.extend(row_labels)
            chosen[card_id] += 1

        bounded = max(-8.0, min(8.0, float(score)))
        if not math.isfinite(bounded):
            return None
        scores.append(bounded)
        audited_combos.append(
            {
                "combo": [int(index) for index in combo],
                "resolved_card_ids": resolved_ids,
                "score": bounded,
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
        "specialist_id": "dragapult",
        "stage_class": stage_class,
        "causal_inputs": list(causal_inputs),
        "scores": scores,
        "combo_labels": audited_combos,
        "abstention_margin": ABSTENTION_MARGIN,
        "missing_label_behavior": "mask_not_zero",
        "runtime_authority": "none",
    }


def guide_audit(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> dict[str, Any] | None:
    """Return complete-stage scores and labels, otherwise mask with ``None``."""
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
) -> list[float] | None:
    audit = guide_audit(
        obs,
        action_combos,
        deck=deck,
        force_enabled=force_enabled,
    )
    return None if audit is None else list(audit["scores"])


def describe() -> str:
    return (
        f"PlainDragapultGuide(version={GUIDE_VERSION}, "
        f"dreepy_min={DREEPY_MINIMUM}, drakloak_min={DRAKLOAK_MINIMUM}, "
        f"dragapult_ex_min={PLAIN_DRAGAPULT_MINIMUM}, "
        f"hammer_max={CRUSHING_HAMMER_FORBIDDEN_MINIMUM - 1})"
    )


__all__ = [
    "ABSTENTION_MARGIN",
    "AUDIT_SCHEMA_VERSION",
    "BUDDY_BUDDY_POFFIN",
    "CTX_MAIN",
    "CTX_SETUP_ACTIVE",
    "CTX_SETUP_BENCH",
    "CTX_TO_BENCH",
    "CTX_TO_HAND",
    "DRAGAPULT_EX",
    "DRAKLOAK",
    "DREEPY",
    "FEZANDIPITI_EX",
    "GUIDE_VERSION",
    "LABEL_DESCRIPTIONS",
    "NIGHT_STRETCHER",
    "OPT_CARD",
    "OPT_NO",
    "OPT_YES",
    "POKE_PAD",
    "ULTRA_BALL",
    "applies",
    "describe",
    "enabled",
    "guide_audit",
    "guide_scores",
    "is_plain_dragapult_deck",
    "prior_logit_bias",
]
