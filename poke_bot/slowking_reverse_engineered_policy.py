"""Causal, abstaining Slowking heuristic surrogate for offline research.

The policy is intentionally separate from the serving guide and runtime action
path.  It converts the strongest repeatable patterns in the recovered public
Slowking replays into option-conditioned scores.  Unsupported or ambiguous
prompts return ``None`` instead of manufacturing a target.

Only the current observation, the current legal actions, and the acting deck
are consumed.  Game result, later frames, opponent hidden cards, and exact-list
identity are never inputs.  Exact list fingerprints belong in evaluation
strata, not in the decision rule.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

SCHEMA_VERSION = "poke_bot.slowking_reverse_engineered_policy/v1"
POLICY_VERSION = "slowking-public-replay-surrogate-v1"
RESEARCH_ONLY = True
RUNTIME_AUTHORITY = "none"

# Card ids.
BASIC_PSYCHIC_ENERGY = 5
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
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
WONDROUS_PATCH = 1146
POKE_PAD = 1152
CIPHERMANIAC = 1188
ACADEMY_AT_NIGHT = 1248

# Option, area, and context ids from the simulator schema.
OPT_CARD = 3
OPT_ENERGY_CARD = 5
OPT_PLAY = 7
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_STADIUM = 7
AREA_LOOKING = 12
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_TO_BENCH = 5
CTX_TO_HAND = 7
CTX_TOP_DECK = 9

ABSTENTION_MARGIN = 0.50
NORMAL_BENCH_LIMIT = 5

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
PAYLOAD_IDS = frozenset({KYUREM, CONKELDURR, ANNIHILAPE})
SEARCHABLE = {
    POKE_PAD: frozenset({SLOWKING, SLOWPOKE, KYUREM, CONKELDURR, ANNIHILAPE, SMOOCHUM}),
    ULTRA_BALL: frozenset(INITIAL_BASIC_IDS | {SLOWKING, CONKELDURR, ANNIHILAPE}),
    NIGHT_STRETCHER: frozenset(
        INITIAL_BASIC_IDS | {SLOWKING, CONKELDURR, ANNIHILAPE, BASIC_PSYCHIC_ENERGY}
    ),
    TELEPATH_PSYCHIC_ENERGY: frozenset({SLOWPOKE, SMOOCHUM, LATIAS_EX}),
}


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _card_id(card: Any) -> int | None:
    if not isinstance(card, dict):
        return None
    return _int(card.get("id"))


def is_slowking_archetype(deck: Iterable[int]) -> bool:
    """Accept any legal-length observed list containing Slowking.

    This is deliberately broader than the historical exact representative.
    Capability checks below decide which individual rules apply.
    """
    try:
        ids = list(deck)
    except TypeError:
        return False
    return len(ids) == 60 and all(_int(card) is not None for card in ids) and SLOWKING in ids


def _players(obs: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    current = obs.get("current") if isinstance(obs, dict) else None
    players = current.get("players") if isinstance(current, dict) else None
    seat = _int(current.get("yourIndex")) if isinstance(current, dict) else None
    if seat not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        return None, None
    if not isinstance(players[seat], dict) or not isinstance(players[1 - seat], dict):
        return None, None
    return players[seat], players[1 - seat]


def _zone(player: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = player.get(key)
    return [card for card in value if isinstance(card, dict)] if isinstance(value, list) else []


def _zone_ids(player: dict[str, Any], key: str) -> list[int]:
    return [value for value in (_card_id(card) for card in _zone(player, key)) if value is not None]


def _board_counts(player: dict[str, Any]) -> Counter[int]:
    return Counter(_zone_ids(player, "active") + _zone_ids(player, "bench"))


def _active_id(player: dict[str, Any]) -> int | None:
    active = _zone_ids(player, "active")
    return active[0] if len(active) == 1 else None


def _resolve_card(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    players = current.get("players") or []
    option_type = _int(option.get("type"))
    area = AREA_HAND if option_type == OPT_PLAY else _int(option.get("area"))
    index = _int(option.get("index"))
    if index is None or index < 0:
        return None
    if area == AREA_DECK:
        zone = select.get("deck") or []
    elif area == AREA_STADIUM:
        zone = current.get("stadium") or []
    elif area == AREA_LOOKING:
        zone = current.get("looking") or []
    else:
        seat = _int(option.get("playerIndex"))
        if seat is None:
            seat = _int(current.get("yourIndex"))
        if seat not in (0, 1) or len(players) != 2:
            return None
        key = {
            AREA_HAND: "hand",
            AREA_DISCARD: "discard",
            AREA_ACTIVE: "active",
            AREA_BENCH: "bench",
        }.get(area)
        if key is None or not isinstance(players[seat], dict):
            return None
        zone = players[seat].get(key) or []
    return zone[index] if index < len(zone) and isinstance(zone[index], dict) else None


def _effect_id(select: dict[str, Any]) -> int | None:
    for key in ("effect", "contextCard"):
        value = select.get(key)
        result = _card_id(value)
        if result is not None:
            return result
        result = _int(value)
        if result is not None:
            return result
    return None


def _validate_combos(select: dict[str, Any], action_combos: Sequence[Sequence[int]]) -> bool:
    options = select.get("option")
    if not isinstance(options, list) or not options or not action_combos:
        return False
    for combo in action_combos:
        if not isinstance(combo, (list, tuple)) or not combo:
            return False
        if len(combo) != len(set(combo)):
            return False
        if any(_int(index) is None or index < 0 or index >= len(options) for index in combo):
            return False
    return len({tuple(combo) for combo in action_combos}) == len(action_combos)


def _opening_score(card: int, *, going_first: bool) -> tuple[float, str]:
    # Confirmed opening prompts expose a stable priority ordering in both turn
    # orders.  Turn order remains an explicit causal input and audit stratum,
    # but it does not change the v1 ranking.
    del going_first
    values = {
        MEGA_KANGASKHAN_EX: 5.0,
        SMOOCHUM: 4.0,
        LATIAS_EX: 3.0,
        SLOWPOKE: 2.0,
        FEZANDIPITI_EX: 0.5,
        MEOWTH_EX: 0.25,
        KYUREM: -1.0,
    }
    return values.get(card, -2.0), "opening_kangaskhan_smoochum_priority"


def _bench_score(card: int, *, me: dict[str, Any], chosen: Counter[int]) -> tuple[float, str]:
    board = _board_counts(me)
    bench_after = len(_zone(me, "bench")) + sum(chosen.values()) + 1
    reserve = -2.5 if bench_after >= NORMAL_BENCH_LIMIT else 0.0
    if card == SLOWPOKE:
        have = board[SLOWPOKE] + chosen[SLOWPOKE]
        return (5.0 if have < 2 else (1.0 if have < 3 else -1.0)) + reserve, "two_slowpoke_continuity"
    if (
        card == LATIAS_EX
        and _active_id(me) == MEGA_KANGASKHAN_EX
        and not board[LATIAS_EX]
        and not chosen[LATIAS_EX]
    ):
        return 3.0 + reserve, "kangaskhan_latias_pivot"
    if card == FEZANDIPITI_EX and not board[FEZANDIPITI_EX] and not chosen[FEZANDIPITI_EX]:
        return 0.75 + reserve, "single_draw_support"
    return reserve, "optional_bench_support"


def _search_score(
    card: int, *, effect: int, me: dict[str, Any], chosen: Counter[int]
) -> tuple[float, str] | None:
    board = _board_counts(me)
    if card == SLOWKING:
        open_lines = max(0, board[SLOWPOKE] - board[SLOWKING] - chosen[SLOWKING])
        return (5.0 if open_lines else 0.5), "evolution_continuity"
    if card == SLOWPOKE:
        have = board[SLOWPOKE] + chosen[SLOWPOKE]
        return (4.75 if have < 2 else (1.0 if have < 3 else -1.0)), "two_slowpoke_continuity"
    if card == LATIAS_EX and _active_id(me) == MEGA_KANGASKHAN_EX and not board[LATIAS_EX]:
        return 3.0, "kangaskhan_latias_pivot"
    if effect == NIGHT_STRETCHER and card == BASIC_PSYCHIC_ENERGY:
        # Energy versus Pokémon recovery depends on attack/retreat commitments
        # not safely reconstructed by a small rule.
        return None
    if card == KYUREM:
        return 2.25, "default_spread_payload"
    if card == CONKELDURR:
        return 1.25, "single_target_payload"
    if card == ANNIHILAPE:
        return 1.0, "trade_payload"
    if card == SMOOCHUM:
        return 0.5, "tempo_pivot"
    return 0.0, "supported_neutral"


def _topdeck_score(card: int, *, me: dict[str, Any]) -> tuple[float, str] | None:
    # Only infer the copied-attack payload plan when Slowking is already
    # Active.  Otherwise Academy is often banking a next-turn resource.
    if _active_id(me) != SLOWKING:
        return None
    values = {
        KYUREM: (5.0, "seek_spread_payload"),
        CONKELDURR: (3.0, "seek_single_target_payload"),
        ANNIHILAPE: (2.5, "seek_trade_payload"),
        SLOWKING: (0.75, "continuity_topdeck"),
    }
    return values.get(card, (0.0, "nonpayload_topdeck"))


def audit_decision(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    margin: float = ABSTENTION_MARGIN,
) -> dict[str, Any] | None:
    """Score a complete legal stage or abstain.

    The returned preference is diagnostics-only.  It is suitable as a
    teacher feature or confidence mask, not as unreviewed runtime authority.
    """
    if not is_slowking_archetype(deck) or not isinstance(obs, dict):
        return None
    select = obs.get("select")
    if not isinstance(select, dict) or not _validate_combos(select, action_combos):
        return None
    me, opponent = _players(obs)
    if me is None or opponent is None:
        return None
    options = select.get("option") or []
    context = _int(select.get("context"))
    effect = _effect_id(select)
    current = obs.get("current") or {}
    seat = _int(current.get("yourIndex"))
    first = _int(current.get("firstPlayer"))

    if context == CTX_SETUP_ACTIVE and effect is None and seat in (0, 1) and first in (0, 1):
        stage = "opening_active"
    elif context == CTX_SETUP_BENCH and effect is None:
        stage = "opening_bench"
    elif context in (CTX_TO_HAND, CTX_TO_BENCH) and effect in SEARCHABLE:
        stage = {
            POKE_PAD: "poke_pad_search",
            ULTRA_BALL: "ultra_ball_search",
            NIGHT_STRETCHER: "night_stretcher_recovery",
            TELEPATH_PSYCHIC_ENERGY: "telepath_basic_search",
        }[effect]
    elif context == CTX_TOP_DECK and effect == ACADEMY_AT_NIGHT:
        stage = "academy_seek_topdeck"
    else:
        return None

    scores: list[float] = []
    labels: list[list[str]] = []
    resolved: list[list[int]] = []
    for combo in action_combos:
        chosen: Counter[int] = Counter()
        combo_score = 0.0
        combo_labels: list[str] = []
        combo_ids: list[int] = []
        for index in combo:
            option = options[index]
            card = _card_id(_resolve_card(obs, option))
            if card is None:
                return None
            if stage == "opening_active":
                if card not in INITIAL_BASIC_IDS:
                    return None
                row = _opening_score(card, going_first=seat == first)
            elif stage == "opening_bench":
                if card not in INITIAL_BASIC_IDS:
                    return None
                row = _bench_score(card, me=me, chosen=chosen)
            elif stage == "academy_seek_topdeck":
                row = _topdeck_score(card, me=me)
                if row is None:
                    return None
            else:
                assert effect is not None
                if card not in SEARCHABLE[effect]:
                    return None
                row = _search_score(card, effect=effect, me=me, chosen=chosen)
                if row is None:
                    return None
            value, label = row
            combo_score += value
            combo_labels.append(label)
            combo_ids.append(card)
            chosen[card] += 1
        bounded = max(-8.0, min(8.0, float(combo_score)))
        if not math.isfinite(bounded):
            return None
        scores.append(bounded)
        labels.append(sorted(set(combo_labels)))
        resolved.append(combo_ids)

    ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    if stage == "night_stretcher_recovery" and resolved[ranked[0]] != [SLOWKING]:
        return None
    if len(ranked) < 2 or scores[ranked[0]] - scores[ranked[1]] < margin:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "research_only": RESEARCH_ONLY,
        "runtime_authority": RUNTIME_AUTHORITY,
        "stage_class": stage,
        "scores": scores,
        "preferred_combo_index": ranked[0],
        "margin": scores[ranked[0]] - scores[ranked[1]],
        "combo_resolved_card_ids": resolved,
        "combo_rule_ids": labels,
        "causal_inputs": [
            "current_public_observation",
            "current_legal_options",
            "acting_deck_capabilities",
        ],
        "future_or_result_inputs": [],
    }


def choose_action(
    obs: dict[str, Any],
    action_combos: Sequence[Sequence[int]],
    *,
    deck: Iterable[int],
    margin: float = ABSTENTION_MARGIN,
) -> tuple[int, ...] | None:
    """Return the preferred legal combo, or ``None`` on abstention."""
    audit = audit_decision(obs, action_combos, deck=deck, margin=margin)
    if audit is None:
        return None
    return tuple(action_combos[int(audit["preferred_combo_index"])])


def prior_logit_bias(
    obs: Any, action_combos: Sequence[Sequence[int]], *, scale: float = 1.0
) -> list[float]:
    """Hard runtime bypass; offline policy scores never enter serving logits."""
    del obs, scale
    return [0.0] * len(action_combos)


__all__ = [
    "ABSTENTION_MARGIN",
    "POLICY_VERSION",
    "RESEARCH_ONLY",
    "RUNTIME_AUTHORITY",
    "SCHEMA_VERSION",
    "audit_decision",
    "choose_action",
    "is_slowking_archetype",
    "prior_logit_bias",
]
