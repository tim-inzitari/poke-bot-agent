"""Sparse exact-causal current-deck curriculum for Crustle.

The guide is bound to Cooper Kull's exact 55th-place NAIC 2026 list in the
canonical representative registry.  It produces training-only curriculum
metadata for observed causal strategic-head targets and selected typed-route
directions.  Final policy logits are never guide targets: ``prior_logit_bias``
is an exact zero bypass, and the generic guide registry must explicitly select
Crustle before target generation is enabled.

Only complete, origin-resolved stages with fully resolved legal options can be
labeled:

* initial Bench setup prefers establishing two Dwebble;
* Buddy-Buddy Poffin, Ultra Ball, and an already-activated Lumiose City fill an
  exact visible Dwebble/Crustle gap while resolving every legal search option;
* Dwebble's Ascension prompt prefers the exact Crustle evolution; and
* Mega Kangaskhan ex's Run Errand Yes/No prompt prefers drawing only when the
  current deck count leaves a conservative post-draw reserve.

Opening Active choice, activating Lumiose City, Energy attachment, Hilda,
Petrel, attack choice, healing, Hammer, gust, stadium play, retreat,
disruption, prizes, matchup plans, and every hidden- or future-dependent
choice are unsupported.  Unsupported, ambiguous, malformed, partial, or
low-margin stages return ``None``. Missing labels are masked, never converted
to zero targets.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

GUIDE_VERSION = "crustle-north-star-v2"
AUDIT_SCHEMA_VERSION = "poke_bot.causal_deck_guide_audit/v1"
GUIDE_TRAINING_MODE = "strategic_directional_v2"

BASIC_GRASS_ENERGY = 1
MIST_ENERGY = 11
SPIKY_ENERGY = 14
GROW_GRASS_ENERGY = 18
# Backward-compatible spelling for older corpus and test helpers.  The exact
# card name in cards/EN_Card_Data.csv is "Grow Grass Energy".
GROWING_GRASS_ENERGY = GROW_GRASS_ENERGY

DWEBBLE = 344
CRUSTLE = 345
MEGA_KANGASKHAN_EX = 756
PSYDUCK = 858

BUDDY_BUDDY_POFFIN = 1086
SUPER_POTION = 1112
CRUSHING_HAMMER = 1120
ULTRA_BALL = 1121
POKEGEAR_30 = 1122
SWITCH = 1123
JUMBO_ICE_CREAM = 1147
HEROS_CAPE = 1159
BOSSS_ORDERS = 1182
XEROSICS_MACHINATIONS = 1197
LISIAS_APPEAL = 1204
TEAM_ROCKETS_PETREL = 1219
HILDA = 1225
LILLIES_DETERMINATION = 1227
FESTIVAL_GROUNDS = 1245
TEAM_ROCKETS_FACTORY = 1257
LUMIOSE_CITY = 1267

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
OPT_PLAY = 7

CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_TO_BENCH = 5
CTX_TO_HAND = 7

ABSTENTION_MARGIN = 0.25

CANONICAL_DECK_COUNTS = {
    MIST_ENERGY: 4,
    SPIKY_ENERGY: 4,
    GROW_GRASS_ENERGY: 4,
    DWEBBLE: 3,
    CRUSTLE: 3,
    MEGA_KANGASKHAN_EX: 4,
    PSYDUCK: 1,
    BUDDY_BUDDY_POFFIN: 2,
    SUPER_POTION: 1,
    CRUSHING_HAMMER: 4,
    ULTRA_BALL: 1,
    POKEGEAR_30: 3,
    SWITCH: 1,
    JUMBO_ICE_CREAM: 4,
    HEROS_CAPE: 1,
    BOSSS_ORDERS: 4,
    XEROSICS_MACHINATIONS: 1,
    LISIAS_APPEAL: 2,
    TEAM_ROCKETS_PETREL: 4,
    HILDA: 2,
    LILLIES_DETERMINATION: 4,
    FESTIVAL_GROUNDS: 1,
    TEAM_ROCKETS_FACTORY: 1,
    LUMIOSE_CITY: 1,
}

INITIAL_BASIC_IDS = frozenset({DWEBBLE, MEGA_KANGASKHAN_EX, PSYDUCK})
SEARCHABLE_CARD_IDS = {
    # Poffin can legally reveal both 70-HP Basics in the exact representative.
    # Psyduck stays neutral unless a Dwebble gap makes Dwebble preferable, but
    # it must still resolve or the complete stage is not causally auditable.
    BUDDY_BUDDY_POFFIN: frozenset({DWEBBLE, PSYDUCK}),
    ULTRA_BALL: frozenset({DWEBBLE, CRUSTLE, MEGA_KANGASKHAN_EX, PSYDUCK}),
    DWEBBLE: frozenset({CRUSTLE}),
    LUMIOSE_CITY: frozenset({DWEBBLE, MEGA_KANGASKHAN_EX, PSYDUCK}),
}
SEARCH_CONTEXTS = {
    BUDDY_BUDDY_POFFIN: CTX_TO_BENCH,
    ULTRA_BALL: CTX_TO_HAND,
    DWEBBLE: CTX_TO_HAND,
    LUMIOSE_CITY: CTX_TO_BENCH,
}
SEARCH_AREAS = {
    BUDDY_BUDDY_POFFIN: AREA_DECK,
    ULTRA_BALL: AREA_DECK,
    DWEBBLE: AREA_DECK,
    LUMIOSE_CITY: AREA_DECK,
}

LABEL_DESCRIPTIONS = {
    "bench_dwebble_redundancy": (
        "Establish two visible Dwebble before optional support Pokemon."
    ),
    "bench_psyduck_neutral": (
        "Psyduck's Damp role is matchup-dependent and receives no blind bonus."
    ),
    "bench_kangaskhan_prize_liability": (
        "Do not default-bench an optional three-Prize Mega Kangaskhan ex."
    ),
    "search_dwebble_redundancy": "Fill the visible two-Dwebble setup gap.",
    "search_crustle_evolution_gap": (
        "Fill a visible Dwebble-to-Crustle evolution gap."
    ),
    "search_psyduck_neutral": (
        "Resolve the legal 70-HP Psyduck option without inferring a matchup."
    ),
    "search_kangaskhan_neutral": (
        "Mega Kangaskhan ex remains neutral without a proved current role."
    ),
    "ascension_crustle": "Complete Dwebble's exact Ascension evolution.",
    "run_errand_safe_yes": (
        "Draw two only with a conservative public post-draw deck reserve."
    ),
    "run_errand_unsafe_yes": (
        "Penalize drawing two without the conservative public reserve."
    ),
    "run_errand_safe_no": "Declining a conservatively safe draw gives up value.",
    "run_errand_unsafe_no": (
        "Decline the draw when the public deck count lacks that reserve."
    ),
}


def enabled() -> bool:
    """Return true only when the training-only guide registry selects Crustle."""
    try:
        from . import deck_guides
    except ImportError:
        return False
    return deck_guides.enabled() and deck_guides.selected_id() == "crustle"


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


def is_crustle_deck(deck: Iterable[int]) -> bool:
    """Match only the checksum-bound Cooper Kull NAIC representative."""
    try:
        card_ids = [_exact_int(card_id) for card_id in deck]
    except TypeError:
        return False
    if len(card_ids) != 60 or any(card_id is None for card_id in card_ids):
        return False
    return Counter(card_ids) == Counter(CANONICAL_DECK_COUNTS)


def is_crustle_family_deck(deck: Iterable[int]) -> bool:
    """Match the collision-safe public Crustle engine across list variants.

    The acting specialist remains bound to the exact Cooper Kull 60-card
    representative above. Public causal guide evidence, however, may come from
    current lists sharing the distinctive Dwebble/Crustle wall-and-healing
    engine. These five simultaneously required markers are present in every
    reviewed 2026 competitive Crustle list and reject unrelated Kangaskhan,
    control, and generic Grass decks.
    """
    try:
        card_ids = [_exact_int(card_id) for card_id in deck]
    except TypeError:
        return False
    if len(card_ids) != 60 or any(card_id is None for card_id in card_ids):
        return False
    counts = Counter(card_ids)
    return (
        counts[DWEBBLE] >= 3
        and counts[CRUSTLE] >= 2
        and counts[MIST_ENERGY] == 4
        and counts[JUMBO_ICE_CREAM] == 4
        and counts[HEROS_CAPE] == 1
    )


def applies(deck_card_ids: Iterable[int]) -> bool:
    return is_crustle_family_deck(deck_card_ids)


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


def _deck_count(player: dict) -> int | None:
    value = _exact_int(player.get("deckCount"))
    if value is not None and value >= 0:
        return value
    return None


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
            current.get("yourIndex", 0) if player_index is None else player_index
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
    if _exact_int(option.get("type")) == OPT_PLAY:
        return _resolve_card(
            obs, area=AREA_HAND, index=option.get("index"), player_index=seat
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
    """Require one complete factorized candidate set and no partial options."""
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
        values = [_exact_int(index) for index in combo]
        if any(index is None for index in values):
            return False
        exact = tuple(int(index) for index in values if index is not None)
        if (
            len(exact) > maximum
            or len(exact) != len(set(exact))
            or any(index < 0 or index >= len(options) for index in exact)
        ):
            return False
        normalized.append(exact)
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


def _score_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool,
) -> dict[str, Any] | None:
    if (
        (not force_enabled and not enabled())
        or not is_crustle_family_deck(deck)
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

    if context == CTX_SETUP_ACTIVE:
        # Correct opener depends on known turn order and attack/retreat plan.
        return None
    if context == CTX_SETUP_BENCH and effect_id is None:
        stage_class = "initial_setup_bench"
        allowed_ids = INITIAL_BASIC_IDS
    elif effect_id in SEARCHABLE_CARD_IDS and context == SEARCH_CONTEXTS[effect_id]:
        stage_class = {
            BUDDY_BUDDY_POFFIN: "buddy_buddy_poffin_search",
            ULTRA_BALL: "ultra_ball_search",
            DWEBBLE: "ascension_search",
            LUMIOSE_CITY: "lumiose_city_search",
        }[effect_id]
        allowed_ids = SEARCHABLE_CARD_IDS[effect_id]
    elif effect_id == MEGA_KANGASKHAN_EX and context == CTX_TO_HAND:
        stage_class = "run_errand_yes_no"
        allowed_ids = frozenset()
    else:
        return None

    scores: list[float] = []
    audited_combos: list[dict[str, Any]] = []
    for combo in action_combos:
        chosen: Counter[int] = Counter()
        score = 0.0
        resolved_ids: list[int] = []
        labels: list[str] = []
        for raw_index in combo:
            index = _exact_int(raw_index)
            if index is None:
                return None
            option = options[index]
            option_type = _exact_int(option.get("type"))
            if stage_class == "run_errand_yes_no":
                if option_type not in {OPT_YES, OPT_NO}:
                    return None
                count = _deck_count(me)
                if count is None:
                    return None
                safe = count > 3
                if option_type == OPT_YES:
                    delta = 2.0 if safe else -6.0
                    labels.append("run_errand_safe_yes" if safe else "run_errand_unsafe_yes")
                else:
                    delta = -0.8 if safe else 4.0
                    labels.append("run_errand_safe_no" if safe else "run_errand_unsafe_no")
                score += delta
                continue

            expected_area = (
                AREA_HAND
                if stage_class == "initial_setup_bench"
                else SEARCH_AREAS[effect_id]
            )
            if (
                option_type != OPT_CARD
                or _exact_int(option.get("area")) != expected_area
            ):
                return None
            card_id = _card_id(_option_card(obs, option))
            if card_id is None or card_id not in allowed_ids:
                return None
            resolved_ids.append(card_id)
            visible = board + chosen
            if stage_class == "initial_setup_bench":
                if card_id == DWEBBLE:
                    delta = 3.5 if visible[DWEBBLE] < 2 else -0.4
                    labels.append("bench_dwebble_redundancy")
                elif card_id == PSYDUCK:
                    delta = 0.0
                    labels.append("bench_psyduck_neutral")
                else:
                    delta = -0.8
                    labels.append("bench_kangaskhan_prize_liability")
            elif stage_class == "ascension_search":
                delta = 4.0
                labels.append("ascension_crustle")
            elif card_id == DWEBBLE:
                search_visible = board + hand + chosen
                delta = 3.5 if search_visible[DWEBBLE] < 2 else -0.4
                labels.append("search_dwebble_redundancy")
            elif card_id == CRUSTLE:
                open_lines = board[DWEBBLE]
                held_evolutions = hand[CRUSTLE] + chosen[CRUSTLE]
                delta = 3.6 if open_lines > held_evolutions else -0.3
                labels.append("search_crustle_evolution_gap")
            elif card_id == PSYDUCK:
                delta = 0.0
                labels.append("search_psyduck_neutral")
            else:
                delta = -0.2
                labels.append("search_kangaskhan_neutral")
            score += delta
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
                "label_ids": sorted(set(labels)),
            }
        )

    if (
        len(scores) != len(action_combos)
        or len(scores) < 2
        or max(scores) - min(scores) < ABSTENTION_MARGIN
    ):
        return None
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "guide_version": GUIDE_VERSION,
        "specialist_id": "crustle",
        "research_only": False,
        "training_mode": GUIDE_TRAINING_MODE,
        "guide_preference_index_role": (
            "selected_causal_route_pairwise_direction_only"
        ),
        "guide_pairwise_route_heads": [
            "action_q",
            "action_resource",
            "action_utility",
            "setup_board_outcome",
            "combo_state",
        ],
        "stage_class": stage_class,
        "causal_inputs": [
            "select.context",
            "select.effect_or_contextCard",
            "complete_current_legal_combos",
            "resolved_current_prompt_card_ids",
            "own_current_active_bench_hand_counts",
            "own_current_deck_count_for_run_errand_only",
        ],
        "scores": scores,
        "combo_labels": audited_combos,
        "abstention_margin": ABSTENTION_MARGIN,
        "missing_label_behavior": "mask_not_zero",
        "mask_not_zero": True,
        "causal_inputs_only": True,
        "target_logits": "none",
        "direct_policy_cross_entropy_allowed": False,
        "final_policy_logits_are_guide_targets": False,
        "runtime_authority": "none",
        "runtime_input_allowed": False,
        "runtime_action_logit_route_allowed": False,
    }


def guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> list[float] | None:
    """Return sparse preference scores or ``None`` for a fully masked stage."""
    audit = _score_stage(
        obs, action_combos, deck=deck, force_enabled=force_enabled
    )
    return None if audit is None else list(audit["scores"])


def audit_guide_scores(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> dict[str, Any] | None:
    """Return the causal audit record for a labeled stage."""
    return _score_stage(
        obs, action_combos, deck=deck, force_enabled=force_enabled
    )


def guide_audit(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool = False,
) -> dict[str, Any] | None:
    """Modern generic-dispatch alias for :func:`audit_guide_scores`."""
    return audit_guide_scores(
        obs,
        action_combos,
        deck=deck,
        force_enabled=force_enabled,
    )
