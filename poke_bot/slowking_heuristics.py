"""Sparse causal curriculum guide for the exact Slowking combo toolbox.

This module qualifies high-confidence training rows for the owner's exact
60-card Slowking list. It is registered only in the generic training-only
guide dispatcher; it has no serving input, action-logit route, selector, or
runtime authority. Tests and offline audits may pass ``force_enabled=True``.

The deck's defining decisions are nonlinear: Academy at Night and
Ciphermaniac's Codebreaking stage a non-Rule-Box Pokémon for Slowking's Seek
Inspiration, while Conkeldurr, Kyurem, Annihilape, and Slowpoke expose attacks
with very different prize and board consequences.  Those top-deck, attack,
target, Energy, recovery, discard, and matchup decisions remain fully masked.

The only prepared labels are complete, origin-resolved stages where current
public state supports a narrow preference:

* establish two Slowpoke without blindly filling the Bench;
* use Poké Pad or Ultra Ball to fill an exact visible Slowpoke/Slowking gap;
* use Telepath Psychic Energy's resolved search to fill that same setup gap;
* recover a visible Slowpoke/Slowking gap with Night Stretcher, provided the
  prompt does not include Basic Energy.

Unsupported, ambiguous, malformed, partial, or low-margin stages return
``None``.  Missing labels are masked, never converted to zero targets.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

GUIDE_VERSION = "slowking-north-star-v1"
AUDIT_SCHEMA_VERSION = "poke_bot.causal_deck_guide_audit/v1"
RESEARCH_ONLY = False

BASIC_PSYCHIC_ENERGY = 5
BOOMERANG_ENERGY = 9
TELEPATH_PSYCHIC_ENERGY = 19

CONKELDURR = 115
FEZANDIPITI_EX = 140
KYUREM = 144
SLOWPOKE = 162
SLOWKING = 163
SMOOCHUM = 183
LATIAS_EX = 184
ANNIHILAPE = 224
MEGA_KANGASKHAN_EX = 756
MEOWTH_EX = 1071

SECRET_BOX = 1092
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
WONDROUS_PATCH = 1146
POKE_PAD = 1152
COUNTER_GAIN = 1168
CIPHERMANIACS_CODEBREAKING = 1188
LILLIES_DETERMINATION = 1227
ACADEMY_AT_NIGHT = 1248

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPT_CARD = 3
OPT_ENERGY_CARD = 5
OPT_PLAY = 7

CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_TO_BENCH = 5
CTX_TO_HAND = 7

ABSTENTION_MARGIN = 0.25
NORMAL_BENCH_LIMIT = 5
RESERVED_BENCH_SLOTS = 1

CANONICAL_DECK_COUNTS = {
    BASIC_PSYCHIC_ENERGY: 4,
    BOOMERANG_ENERGY: 2,
    TELEPATH_PSYCHIC_ENERGY: 4,
    CONKELDURR: 2,
    FEZANDIPITI_EX: 1,
    KYUREM: 2,
    SLOWPOKE: 4,
    SLOWKING: 4,
    SMOOCHUM: 2,
    LATIAS_EX: 2,
    ANNIHILAPE: 2,
    MEGA_KANGASKHAN_EX: 2,
    MEOWTH_EX: 1,
    SECRET_BOX: 1,
    NIGHT_STRETCHER: 3,
    ULTRA_BALL: 4,
    WONDROUS_PATCH: 3,
    POKE_PAD: 4,
    COUNTER_GAIN: 1,
    CIPHERMANIACS_CODEBREAKING: 4,
    LILLIES_DETERMINATION: 4,
    ACADEMY_AT_NIGHT: 4,
}

INITIAL_BASIC_IDS = frozenset(
    {
        FEZANDIPITI_EX,
        KYUREM,
        SLOWPOKE,
        SMOOCHUM,
        LATIAS_EX,
        MEGA_KANGASKHAN_EX,
        MEOWTH_EX,
    }
)
NON_RULE_BOX_POKEMON_IDS = frozenset(
    {CONKELDURR, KYUREM, SLOWPOKE, SLOWKING, SMOOCHUM, ANNIHILAPE}
)
ALL_POKEMON_IDS = frozenset(
    set(NON_RULE_BOX_POKEMON_IDS)
    | {FEZANDIPITI_EX, LATIAS_EX, MEGA_KANGASKHAN_EX, MEOWTH_EX}
)
TELEPATH_BASIC_PSYCHIC_IDS = frozenset({SLOWPOKE, SMOOCHUM, LATIAS_EX})

SEARCHABLE_CARD_IDS = {
    POKE_PAD: NON_RULE_BOX_POKEMON_IDS,
    ULTRA_BALL: ALL_POKEMON_IDS,
    NIGHT_STRETCHER: frozenset(set(ALL_POKEMON_IDS) | {BASIC_PSYCHIC_ENERGY}),
    TELEPATH_PSYCHIC_ENERGY: TELEPATH_BASIC_PSYCHIC_IDS,
}
SEARCH_CONTEXTS = {
    POKE_PAD: CTX_TO_HAND,
    ULTRA_BALL: CTX_TO_HAND,
    NIGHT_STRETCHER: CTX_TO_HAND,
    TELEPATH_PSYCHIC_ENERGY: CTX_TO_BENCH,
}
SEARCH_AREAS = {
    POKE_PAD: AREA_DECK,
    ULTRA_BALL: AREA_DECK,
    NIGHT_STRETCHER: AREA_DISCARD,
    TELEPATH_PSYCHIC_ENERGY: AREA_DECK,
}

LABEL_DESCRIPTIONS = {
    "bench_slowpoke_redundancy": (
        "Establish two visible Slowpoke before optional support Pokémon."
    ),
    "bench_reserve_slot": (
        "Avoid blindly consuming the last normal Bench slot during setup."
    ),
    "bench_latias_for_kangaskhan_pivot": (
        "A first Latias ex is bounded support when Kangaskhan is Active."
    ),
    "bench_optional_support": (
        "Optional support remains neutral without a current exact use."
    ),
    "search_slowpoke_redundancy": "Fill a visible two-Slowpoke setup gap.",
    "search_slowking_evolution": (
        "Fill a visible Slowpoke-to-Slowking evolution gap."
    ),
    "search_latias_for_kangaskhan_pivot": (
        "Fill the exact Latias mobility gap for an Active Kangaskhan."
    ),
    "search_payload_route_unsupported": (
        "Copied-attack payload selection needs exact prize and board context."
    ),
    "night_stretcher_energy_unsupported": (
        "Basic Energy recovery needs exact attachment and attack routing."
    ),
    "supported_neutral": "Resolved option with no safe sparse preference.",
}


def enabled() -> bool:
    """Return true only when the generic training-only registry selects Slowking."""
    try:
        from . import deck_guides
    except ImportError:
        return False
    return deck_guides.enabled() and deck_guides.selected_id() == "slowking"


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


def is_slowking_deck(deck: Iterable[int]) -> bool:
    """Match only the owner's exact 60-card specialist representative."""
    try:
        card_ids = [_exact_int(card_id) for card_id in deck]
    except TypeError:
        return False
    if len(card_ids) != 60 or any(card_id is None for card_id in card_ids):
        return False
    return Counter(card_ids) == Counter(CANONICAL_DECK_COUNTS)


def applies(deck_card_ids: Iterable[int]) -> bool:
    """Exact-deck predicate used by the generic guide dispatcher."""
    return is_slowking_deck(deck_card_ids)


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


def _active_id(player: dict) -> int | None:
    active = _zone_ids(player, "active")
    return active[0] if len(active) == 1 else None


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
    if _exact_int(option.get("type")) == OPT_PLAY:
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
        value = select.get(key)
        card_id = _card_id(value)
        if card_id is not None:
            return card_id
        value_id = _exact_int(value)
        if value_id is not None:
            return value_id
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


def _reserve_slot_penalty(me: dict, chosen: Counter[int]) -> float:
    selected = sum(chosen.values())
    bench_count = len(_cards(me.get("bench")))
    usable_without_last_slot = NORMAL_BENCH_LIMIT - RESERVED_BENCH_SLOTS
    return -3.0 if bench_count + selected >= usable_without_last_slot else 0.0


def _setup_bench_score(
    card_id: int,
    *,
    me: dict,
    board: Counter[int],
    chosen: Counter[int],
) -> tuple[float, tuple[str, ...]]:
    penalty = _reserve_slot_penalty(me, chosen)
    labels: list[str] = []
    if penalty:
        labels.append("bench_reserve_slot")

    if card_id == SLOWPOKE:
        have = board[SLOWPOKE] + chosen[SLOWPOKE]
        value = 4.0 if have < 2 else (0.4 if have < 3 else -0.8)
        labels.append("bench_slowpoke_redundancy")
        return value + penalty, tuple(labels)
    if (
        card_id == LATIAS_EX
        and _active_id(me) == MEGA_KANGASKHAN_EX
        and board[LATIAS_EX] + chosen[LATIAS_EX] == 0
    ):
        labels.append("bench_latias_for_kangaskhan_pivot")
        return 2.0 + penalty, tuple(labels)
    labels.append("bench_optional_support")
    return penalty, tuple(labels)


def _search_score(
    card_id: int,
    *,
    effect_id: int,
    me: dict,
    board: Counter[int],
    chosen: Counter[int],
) -> tuple[float, tuple[str, ...]] | None:
    if effect_id == NIGHT_STRETCHER and card_id == BASIC_PSYCHIC_ENERGY:
        return None

    if card_id == SLOWPOKE:
        have = board[SLOWPOKE] + chosen[SLOWPOKE]
        value = 4.0 if have < 2 else (0.4 if have < 3 else -0.8)
        if effect_id == TELEPATH_PSYCHIC_ENERGY:
            value += _reserve_slot_penalty(me, chosen)
        return value, ("search_slowpoke_redundancy",)

    if card_id == SLOWKING:
        open_lines = max(0, board[SLOWPOKE] - board[SLOWKING])
        value = 4.2 if open_lines > chosen[SLOWKING] else 0.3
        return value, ("search_slowking_evolution",)

    if (
        card_id == LATIAS_EX
        and _active_id(me) == MEGA_KANGASKHAN_EX
        and board[LATIAS_EX] + chosen[LATIAS_EX] == 0
    ):
        value = 2.0
        if effect_id == TELEPATH_PSYCHIC_ENERGY:
            value += _reserve_slot_penalty(me, chosen)
        return value, ("search_latias_for_kangaskhan_pivot",)

    if card_id in {CONKELDURR, KYUREM, ANNIHILAPE}:
        return 0.0, ("search_payload_route_unsupported",)

    return 0.0, ("supported_neutral",)


def _score_stage(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    force_enabled: bool,
) -> dict[str, Any] | None:
    if (
        (not force_enabled and not enabled())
        or not is_slowking_deck(deck)
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

    if context == CTX_SETUP_ACTIVE:
        # The correct opener depends on the unmodeled go-first/go-second plan:
        # Smoochum supports Delightful Kiss going second, while Kangaskhan
        # requires a proved Run Errand plus Latias/pivot line.
        return None
    if context == CTX_SETUP_BENCH and effect_id is None:
        stage_class = "initial_setup_bench"
        allowed_ids = INITIAL_BASIC_IDS
        expected_area = AREA_HAND
        causal_inputs = (
            "select.context",
            "complete_current_legal_combos",
            "resolved_current_option_card_ids",
            "own_current_active_and_bench_card_ids",
        )
    elif effect_id in SEARCHABLE_CARD_IDS and context == SEARCH_CONTEXTS[effect_id]:
        stage_class = {
            POKE_PAD: "poke_pad_search",
            ULTRA_BALL: "ultra_ball_search",
            NIGHT_STRETCHER: "night_stretcher_recovery",
            TELEPATH_PSYCHIC_ENERGY: "telepath_psychic_energy_search",
        }[effect_id]
        allowed_ids = SEARCHABLE_CARD_IDS[effect_id]
        expected_area = SEARCH_AREAS[effect_id]
        causal_inputs = (
            "select.effect_or_contextCard",
            "complete_current_legal_combos",
            "resolved_current_prompt_card_ids",
            "own_current_active_and_bench_card_ids",
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
            if (
                option_type not in {OPT_CARD, OPT_ENERGY_CARD}
                or _exact_int(option.get("area")) != expected_area
            ):
                return None
            card_id = _card_id(_option_card(obs, option))
            if card_id is None or card_id not in allowed_ids:
                return None
            resolved_ids.append(card_id)
            if context == CTX_SETUP_BENCH:
                value, row_labels = _setup_bench_score(
                    card_id,
                    me=me,
                    board=board,
                    chosen=chosen,
                )
            else:
                assert effect_id is not None
                row = _search_score(
                    card_id,
                    effect_id=effect_id,
                    me=me,
                    board=board,
                    chosen=chosen,
                )
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
        "specialist_id": "slowking",
        "research_only": False,
        "training_mode": "strategic_curriculum_v1",
        "guide_preference_index_role": "diagnostics_and_row_qualification_only",
        "direct_policy_cross_entropy_allowed": False,
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
    """Return a complete-stage curriculum audit, otherwise mask with ``None``."""
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
    """Return sparse row-qualification scores for the exact deck."""
    audit = guide_audit(
        obs,
        action_combos,
        deck=deck,
        force_enabled=force_enabled,
    )
    return None if audit is None else list(audit["scores"])


def describe() -> str:
    return (
        f"SlowkingCurriculumGuide(version={GUIDE_VERSION}, "
        "runtime_authority=none, exact_representative_only=True)"
    )


__all__ = [
    "ABSTENTION_MARGIN",
    "ACADEMY_AT_NIGHT",
    "AUDIT_SCHEMA_VERSION",
    "CANONICAL_DECK_COUNTS",
    "CTX_SETUP_ACTIVE",
    "CTX_SETUP_BENCH",
    "CTX_TO_BENCH",
    "CTX_TO_HAND",
    "GUIDE_VERSION",
    "LABEL_DESCRIPTIONS",
    "LATIAS_EX",
    "MEGA_KANGASKHAN_EX",
    "NIGHT_STRETCHER",
    "OPT_CARD",
    "OPT_ENERGY_CARD",
    "POKE_PAD",
    "RESEARCH_ONLY",
    "SLOWKING",
    "SLOWPOKE",
    "SMOOCHUM",
    "TELEPATH_PSYCHIC_ENERGY",
    "ULTRA_BALL",
    "applies",
    "describe",
    "enabled",
    "guide_audit",
    "guide_scores",
    "is_slowking_deck",
    "prior_logit_bias",
]
