"""Archetype game-plan heuristics (SME-guided).

These heuristics encode top-level competitive game plans for the two archetypes
we specialize in, and feed two model heads:

* **value head** — bounded shaping bonuses nudge value targets toward states the
  archetype considers strong (prize-trade structure for Lucario, board spread and
  multi-KO turns for Dragapult).
* **policy head** — per-action scores bias root action ranking during search and
  blend into soft policy targets.

Design asymmetry (per former World Champion SME):

* **Mega Lucario ex** is a *linear* deck — one engine, one prize-trade thesis,
  predictable attacker rotation. It gets strong, phase-ordered scores.
* **Dragapult ex / Dusknoir** is *non-linear* — top players branch by matchup and
  board texture. It gets softer, context-conditioned scores so multiple lines stay
  viable and the policy is not collapsed onto a single script.

Everything here reads the raw observation dict (the same structure the baseline
agents and feature encoder consume), so it works both at collection time and at
tensor-build time without a ``cg.api`` dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from poke_agent.archetypes import HEURISTIC_ARCHETYPE_SLUG_PATTERNS, load_archetype_registry
from poke_agent.deck_pool import deck_matches_archetype_patterns
from poke_agent.rewards import seat_prize_counts

# --- Archetype identities ---
ARCHETYPE_DRAGAPULT = "dragapult-ex"
ARCHETYPE_LUCARIO = "mega-lucario-ex"
ARCHETYPE_ABOMASNOW = "mega-abomasnow-ex"
ARCHETYPE_IONO = "iono"
ARCHETYPE_STARMIE = "starmie"
ARCHETYPE_CRUSTLE = "crustle"
ARCHETYPE_UNKNOWN = "unknown"

ALL_HEURISTIC_ARCHETYPES = (
    ARCHETYPE_LUCARIO,
    ARCHETYPE_DRAGAPULT,
    ARCHETYPE_ABOMASNOW,
    ARCHETYPE_IONO,
    ARCHETYPE_STARMIE,
    ARCHETYPE_CRUSTLE,
)

# --- Signature card IDs (global, from cg.api all_card_data) ---
DREEPY, DRAKLOAK, DRAGAPULT_EX = 119, 120, 121
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
BUDEW = 235
UNFAIR_STAMP = 1080
BOSS_ORDERS = 1182
MAKUHITA, HARIYAMA, LUNATONE, SOLROCK, RIOLU, MEGA_LUCARIO_EX = 673, 674, 675, 676, 677, 678

DRAGAPULT_SIGNATURE = frozenset({DREEPY, DRAKLOAK, DRAGAPULT_EX})
DUSKNOIR_LINE = frozenset({DUSKULL, DUSCLOPS, DUSKNOIR})
LUCARIO_SIGNATURE = frozenset({RIOLU, MEGA_LUCARIO_EX})
LUCARIO_ENGINE = frozenset({SOLROCK, LUNATONE})

KYOGRE, SNOVER, MEGA_ABOMASNOW_EX = 721, 722, 723
BASIC_WATER_ENERGY = 3
ABOMASNOW_SIGNATURE = frozenset({SNOVER, MEGA_ABOMASNOW_EX, KYOGRE})

IONO_VOLTORB, IONO_TADBULB = 265, 268
IONO_BELLIBOLT_EX, IONO_WATTREL, IONO_KILOWATTREL = 269, 270, 271
IONO_SIGNATURE = frozenset(
    {IONO_VOLTORB, IONO_TADBULB, IONO_BELLIBOLT_EX, IONO_WATTREL, IONO_KILOWATTREL}
)

STARYU, STARMIE, MEGA_STARMIE_EX = 860, 861, 104
SNORUNT, FROSLASS, MEGA_FROSLASS_EX = 1030, 1031, 112
STARMIE_SIGNATURE = frozenset({STARYU, STARMIE, MEGA_STARMIE_EX, SNORUNT, FROSLASS, MEGA_FROSLASS_EX})

MEGA_KANGASKHAN_EX = 344
CRUSTLE_CARD = 756
CRUSTLE_SIGNATURE = frozenset({MEGA_KANGASKHAN_EX, CRUSTLE_CARD})

# Fallback when the deck registry is unavailable (e.g. trimmed submission env).
OPPONENT_ARCHETYPE_SIGNATURES: dict[str, frozenset[int]] = {
    ARCHETYPE_DRAGAPULT: DRAGAPULT_SIGNATURE | DUSKNOIR_LINE,
    ARCHETYPE_LUCARIO: LUCARIO_SIGNATURE | LUCARIO_ENGINE | frozenset({MAKUHITA, HARIYAMA}),
}

_VISIBLE_MIN_SCORE = 0.15
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Attack IDs ---
PHANTOM_DIVE_ATTACK = 154
ITCHY_POLLEN_ATTACK = 323
MEGA_BRAVE_ATTACK = 983
HAMMER_LANCHE_ATTACK = 1046

# --- OptionType enum values (cg.api.OptionType) ---
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_ABILITY = 10
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14

_VALUE_BONUS_CLAMP = 0.05


def classify_archetype(deck_cards: list[int] | tuple[int, ...] | None) -> str:
    """Identify our archetype from the 60-card deck by signature Pokémon lines."""
    if not deck_cards:
        return ARCHETYPE_UNKNOWN
    counts = Counter(int(card) for card in deck_cards)
    scores: list[tuple[int, str]] = []
    dragapult = sum(counts[card] for card in DRAGAPULT_SIGNATURE)
    lucario = sum(counts[card] for card in LUCARIO_SIGNATURE)
    abomasnow = sum(counts[card] for card in ABOMASNOW_SIGNATURE)
    iono = sum(counts[card] for card in IONO_SIGNATURE)
    starmie = sum(counts[card] for card in STARMIE_SIGNATURE)
    if dragapult >= 3:
        scores.append((dragapult, ARCHETYPE_DRAGAPULT))
    if lucario >= 2:
        scores.append((lucario, ARCHETYPE_LUCARIO))
    if counts[MEGA_ABOMASNOW_EX] >= 2 or (counts[SNOVER] >= 2 and counts[BASIC_WATER_ENERGY] >= 20):
        scores.append((max(abomasnow, counts[MEGA_ABOMASNOW_EX] * 2), ARCHETYPE_ABOMASNOW))
    if counts[IONO_BELLIBOLT_EX] >= 2 or iono >= 6:
        scores.append((max(iono, counts[IONO_BELLIBOLT_EX] * 2), ARCHETYPE_IONO))
    if starmie >= 4:
        scores.append((starmie, ARCHETYPE_STARMIE))
    if counts[MEGA_KANGASKHAN_EX] >= 2:
        scores.append((counts[MEGA_KANGASKHAN_EX] * 2 + counts[CRUSTLE_CARD], ARCHETYPE_CRUSTLE))
    if not scores:
        return ARCHETYPE_UNKNOWN
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[0][1]


def starmie_variant(deck_cards: list[int] | tuple[int, ...] | None) -> str:
    """Branch key for Starmie family lists."""
    if not deck_cards:
        return "unknown"
    counts = Counter(int(card) for card in deck_cards)
    if counts[MEGA_STARMIE_EX] >= 2 and counts[FROSLASS] < 2:
        return "mega-starmie"
    if counts[DUSKULL] >= 2 or counts[DUSCLOPS] >= 1:
        return "starmie-dusknoir"
    return "starmie-froslass"


def _current(obs: dict[str, Any]) -> dict[str, Any]:
    return obs.get("current") or {}


def _player(obs: dict[str, Any], seat: int) -> dict[str, Any]:
    players = _current(obs).get("players") or []
    if 0 <= seat < len(players) and isinstance(players[seat], dict):
        return players[seat]
    return {}


def _card_id(card: dict[str, Any] | None) -> int | None:
    if not isinstance(card, dict):
        return None
    raw = card.get("id")
    if raw is None:
        raw = card.get("cardId")
    return int(raw) if raw is not None else None


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    active = player.get("active") or []
    if active and isinstance(active[0], dict):
        return active[0]
    return None


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [card for card in (player.get("bench") or []) if isinstance(card, dict)]


def _cards_in_play(player: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    active = _active(player)
    if active is not None:
        card_id = _card_id(active)
        if card_id is not None:
            ids.append(card_id)
    for card in _bench(player):
        card_id = _card_id(card)
        if card_id is not None:
            ids.append(card_id)
    return ids


def _hand_card_id(player: dict[str, Any], hand_index: int | None) -> int | None:
    if hand_index is None:
        return None
    hand = player.get("hand") or []
    if 0 <= hand_index < len(hand):
        return _card_id(hand[hand_index])
    return None


def _is_damaged(card: dict[str, Any]) -> bool:
    hp = int(card.get("hp", 0) or 0)
    max_hp = int(card.get("maxHp", card.get("hp", 0)) or 0)
    return max_hp > 0 and hp < max_hp


def _damaged_opp_bench_count(obs: dict[str, Any], seat: int) -> int:
    opp = _player(obs, 1 - seat)
    return sum(1 for card in _bench(opp) if _is_damaged(card))


def _opp_bench_count(obs: dict[str, Any], seat: int) -> int:
    return len(_bench(_player(obs, 1 - seat)))


def _benched_ready_mega(player: dict[str, Any]) -> bool:
    for card in _bench(player):
        if _card_id(card) == MEGA_LUCARIO_EX and len(card.get("energies") or []) >= 2:
            return True
    return False


def game_phase(obs: dict[str, Any], seat: int) -> str:
    """Coarse game phase from turn count and our remaining prizes."""
    turn = int(_current(obs).get("turn", 0) or 0)
    own_prizes, _ = seat_prize_counts(obs, seat=seat)
    if turn <= 2 or own_prizes >= 6:
        return "early"
    if own_prizes <= 2:
        return "late"
    return "mid"


def _visible_card_counter(player: dict[str, Any]) -> Counter[int]:
    return Counter(_cards_in_play(player))


@lru_cache(maxsize=1)
def _opponent_prediction_registry():
    return load_archetype_registry(_PROJECT_ROOT)


def _predict_opponent_from_signatures(visible: Counter[int]) -> str:
    best = ARCHETYPE_UNKNOWN
    best_overlap = 0
    for candidate, signature in OPPONENT_ARCHETYPE_SIGNATURES.items():
        overlap = sum(1 for card in visible if card in signature)
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best if best_overlap > 0 else ARCHETYPE_UNKNOWN


@dataclass(frozen=True)
class OpponentPrediction:
    slug: str
    confidence: float


def opponent_matches_family(opponent_slug: str, family: str) -> bool:
    """Whether a predicted meta slug belongs to a trained heuristic family."""
    if opponent_slug == ARCHETYPE_UNKNOWN:
        return False
    patterns = HEURISTIC_ARCHETYPE_SLUG_PATTERNS.get(family)
    if not patterns:
        return False
    return deck_matches_archetype_patterns(opponent_slug, patterns)


def _our_family_patterns(our_archetype: str) -> list[str]:
    return list(HEURISTIC_ARCHETYPE_SLUG_PATTERNS.get(our_archetype) or [])


def predict_opponent(obs: dict[str, Any], seat: int) -> OpponentPrediction:
    """Best-guess opponent meta archetype from visible Pokémon (active + bench)."""
    visible = _visible_card_counter(_player(obs, 1 - seat))
    if not visible:
        return OpponentPrediction(slug=ARCHETYPE_UNKNOWN, confidence=0.0)
    try:
        registry = _opponent_prediction_registry()
        slug, score = registry.classify_visible_archetype(visible, min_score=_VISIBLE_MIN_SCORE)
        if slug != "unknown":
            return OpponentPrediction(slug=slug, confidence=score)
    except Exception:
        pass
    fallback = _predict_opponent_from_signatures(visible)
    return OpponentPrediction(
        slug=fallback,
        confidence=0.2 if fallback != ARCHETYPE_UNKNOWN else 0.0,
    )


def predict_opponent_archetype(obs: dict[str, Any], seat: int) -> str:
    """Competitive meta slug (e.g. ``dragapult-dusknoir``) or ``unknown``."""
    return predict_opponent(obs, seat).slug


def matchup_context(obs: dict[str, Any], seat: int, archetype: str) -> str:
    """Opponent-shape context from predicted meta slug and prize pace."""
    opponent_slug = predict_opponent_archetype(obs, seat)
    our_patterns = _our_family_patterns(archetype)
    if our_patterns and deck_matches_archetype_patterns(opponent_slug, our_patterns):
        return "mirror"
    own_prizes, opp_prizes = seat_prize_counts(obs, seat=seat)
    turn = int(_current(obs).get("turn", 0) or 0)
    if opp_prizes < own_prizes and turn <= 4:
        return "aggro"
    return "default"


def _select_options(obs: dict[str, Any]) -> list[dict[str, Any]]:
    select = obs.get("select") or {}
    return [opt for opt in (select.get("option") or []) if isinstance(opt, dict)]


def _score_lucario_option(
    option: dict[str, Any],
    player: dict[str, Any],
    *,
    phase: str,
    opponent_archetype: str,
) -> float:
    """Linear Lucario plan: engine setup, Aura Jab accel, Mega Brave / Hariyama finish."""
    otype = int(option.get("type", -1))
    vs_dragapult = opponent_matches_family(opponent_archetype, ARCHETYPE_DRAGAPULT)
    if otype == OPTION_ATTACK:
        if int(option.get("attackId", -1)) == MEGA_BRAVE_ATTACK:
            return 1.0 if phase in ("mid", "late") else 0.5
        return 0.6  # Aura Jab — damage plus bench energy acceleration
    if otype == OPTION_EVOLVE:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == HARIYAMA:
            # Heave-Ho Catcher gust is especially strong vs Dragapult (snipe Drakloak).
            return 0.95 if vs_dragapult else 0.9
        if card_id == MEGA_LUCARIO_EX:
            return 0.9
        return 0.6
    if otype == OPTION_ABILITY:
        return 0.7  # Lunar Cycle draw engine
    if otype == OPTION_PLAY:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == BOSS_ORDERS:
            # Pull up the Drakloak draw engine before it sets up.
            return 0.7 if vs_dragapult else 0.5
        if card_id in (SOLROCK, LUNATONE, RIOLU):
            return 0.7 if phase == "early" else 0.4
        return 0.4
    if otype == OPTION_ATTACH:
        return 0.55
    if otype == OPTION_END:
        return 0.1 if phase == "early" else 0.3
    return 0.3


def _score_dragapult_option(
    option: dict[str, Any],
    obs: dict[str, Any],
    player: dict[str, Any],
    seat: int,
    *,
    phase: str,
    context: str,
    opponent_archetype: str,
) -> float:
    """Soft, branching Dragapult plan: stall, spread, convert — context-conditioned."""
    otype = int(option.get("type", -1))
    vs_lucario = opponent_matches_family(opponent_archetype, ARCHETYPE_LUCARIO)
    if otype == OPTION_ATTACK:
        attack_id = int(option.get("attackId", -1))
        if attack_id == PHANTOM_DIVE_ATTACK:
            base = 0.8 if _opp_bench_count(obs, seat) >= 2 else 0.7
            # vs Lucario the win condition lives on the bench (Solrock/Lunatone/Riolu
            # engine + Hariyama), so spread that pressures the bench is worth more.
            if vs_lucario and _opp_bench_count(obs, seat) >= 2:
                base = 0.85
            if context == "mirror" and phase == "early":
                base *= 0.4  # SME: avoid early aggression in the mirror
            return base
        if attack_id == ITCHY_POLLEN_ATTACK:
            return 0.7 if phase == "early" else 0.4  # Budew stall window
        return 0.55
    if otype == OPTION_ABILITY:
        return 0.65  # Recon Directive draw / Dusknoir Special Process
    if otype == OPTION_EVOLVE:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == DUSKNOIR:
            return 0.75 if context == "mirror" else 0.6
        if card_id in (DRAKLOAK, DRAGAPULT_EX):
            return 0.6
        return 0.5
    if otype == OPTION_PLAY:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == BUDEW:
            return 0.65 if phase == "early" else 0.3
        if card_id == UNFAIR_STAMP:
            return 0.7 if phase in ("mid", "late") else 0.3
        if card_id == BOSS_ORDERS:
            if phase in ("mid", "late"):
                return 0.7 if vs_lucario else 0.6  # gust the bench engine vs Lucario
            return 0.35
        return 0.45
    if otype == OPTION_ATTACH:
        return 0.5
    if otype == OPTION_END:
        return 0.25
    return 0.35


def _active_energy_count(player: dict[str, Any]) -> int:
    total = 0
    active = _active(player)
    if active is not None:
        total += len(active.get("energies") or [])
    for card in _bench(player):
        total += len(card.get("energies") or [])
    return total


def _score_abomasnow_option(
    option: dict[str, Any],
    player: dict[str, Any],
    *,
    phase: str,
) -> float:
    otype = int(option.get("type", -1))
    if otype == OPTION_ATTACK:
        attack_id = int(option.get("attackId", -1))
        if attack_id == HAMMER_LANCHE_ATTACK:
            return 0.95 if phase in ("mid", "late") else 0.7
        return 0.65
    if otype == OPTION_PLAY:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == BOSS_ORDERS:
            return 0.85 if phase in ("mid", "late") else 0.5
        return 0.45
    if otype == OPTION_ATTACH:
        return 0.7 if phase == "early" else 0.55
    if otype == OPTION_EVOLVE:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id in (MEGA_ABOMASNOW_EX, KYOGRE):
            return 0.85
        return 0.55
    if otype == OPTION_END:
        return 0.15 if phase == "early" else 0.25
    return 0.35


def _score_iono_option(
    option: dict[str, Any],
    player: dict[str, Any],
    *,
    phase: str,
) -> float:
    otype = int(option.get("type", -1))
    energy = _active_energy_count(player)
    if otype == OPTION_ABILITY:
        return 0.85  # Kilowattrel Flashing Draw / Bellibolt streamer
    if otype == OPTION_ATTACK:
        card_id = _card_id(_active(player))
        if card_id == IONO_VOLTORB:
            return 0.8 if energy >= 4 else 0.45
        if card_id == IONO_BELLIBOLT_EX:
            return 0.75 if phase in ("mid", "late") else 0.5
        return 0.55
    if otype == OPTION_ATTACH:
        return 0.75 if phase == "early" else 0.6
    if otype == OPTION_EVOLVE:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id in (IONO_BELLIBOLT_EX, IONO_KILOWATTREL):
            return 0.8
        return 0.55
    if otype == OPTION_PLAY:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id in (IONO_WATTREL, IONO_TADBULB, IONO_VOLTORB):
            return 0.7 if phase == "early" else 0.45
        return 0.45
    if otype == OPTION_END:
        return 0.2
    return 0.35


def _score_starmie_option(
    option: dict[str, Any],
    obs: dict[str, Any],
    player: dict[str, Any],
    seat: int,
    *,
    phase: str,
    context: str,
    variant: str,
) -> float:
    otype = int(option.get("type", -1))
    if otype == OPTION_ATTACK:
        base = 0.75 if _opp_bench_count(obs, seat) >= 1 else 0.6
        if context == "mirror" and phase == "early":
            base *= 0.45
        if variant == "mega-starmie":
            base += 0.05
        return base
    if otype == OPTION_ABILITY:
        return 0.7 if variant in ("starmie-froslass", "starmie-dusknoir") else 0.55
    if otype == OPTION_EVOLVE:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id in (FROSLASS, MEGA_FROSLASS_EX, STARMIE, MEGA_STARMIE_EX):
            return 0.65
        return 0.5
    if otype == OPTION_PLAY:
        if phase == "early":
            return 0.55
        return 0.45
    if otype == OPTION_ATTACH:
        return 0.5
    if otype == OPTION_END:
        return 0.25
    return 0.35


def _score_crustle_option(
    option: dict[str, Any],
    player: dict[str, Any],
    *,
    phase: str,
    opponent_archetype: str,
) -> float:
    vs_dragapult = opponent_matches_family(opponent_archetype, ARCHETYPE_DRAGAPULT)
    otype = int(option.get("type", -1))
    if otype == OPTION_ATTACK:
        return 0.72 if phase in ("mid", "late") else 0.48
    if otype == OPTION_PLAY:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == BOSS_ORDERS:
            return 0.75 if phase in ("mid", "late") else 0.4
        return 0.58 if not (phase == "early" and vs_dragapult) else 0.52
    if otype == OPTION_ATTACH:
        return 0.5
    if otype == OPTION_END:
        return 0.3 if phase == "early" else 0.25
    return 0.4


def score_action(
    action: list[int],
    obs: dict[str, Any],
    seat: int,
    *,
    archetype: str,
    phase: str,
    context: str,
    opponent_archetype: str = ARCHETYPE_UNKNOWN,
    starmie_variant_key: str = "",
) -> float:
    """Mean per-option game-plan score for one concrete action (list of option indices)."""
    options = _select_options(obs)
    if not options or archetype == ARCHETYPE_UNKNOWN:
        return 0.0
    player = _player(obs, seat)
    total = 0.0
    counted = 0
    for index in action:
        if not 0 <= int(index) < len(options):
            continue
        option = options[int(index)]
        if archetype == ARCHETYPE_LUCARIO:
            total += _score_lucario_option(
                option, player, phase=phase, opponent_archetype=opponent_archetype
            )
        elif archetype == ARCHETYPE_DRAGAPULT:
            total += _score_dragapult_option(
                option, obs, player, seat, phase=phase, context=context, opponent_archetype=opponent_archetype
            )
        elif archetype == ARCHETYPE_ABOMASNOW:
            total += _score_abomasnow_option(option, player, phase=phase)
        elif archetype == ARCHETYPE_IONO:
            total += _score_iono_option(option, player, phase=phase)
        elif archetype == ARCHETYPE_STARMIE:
            total += _score_starmie_option(
                option, obs, player, seat, phase=phase, context=context, variant=starmie_variant_key
            )
        elif archetype == ARCHETYPE_CRUSTLE:
            total += _score_crustle_option(
                option, player, phase=phase, opponent_archetype=opponent_archetype
            )
        counted += 1
    if counted == 0:
        return 0.0
    return total / counted


def value_shaping_bonus(
    obs_before: dict[str, Any] | None,
    obs_after: dict[str, Any] | None,
    seat: int,
    *,
    archetype: str,
    phase: str,
    context: str,
) -> float:
    """Bounded value-target nudge toward archetype-strong states (clamped to ±0.05)."""
    if archetype == ARCHETYPE_UNKNOWN or obs_before is None or obs_after is None:
        return 0.0
    own_before, opp_before = seat_prize_counts(obs_before, seat=seat)
    own_after, opp_after = seat_prize_counts(obs_after, seat=seat)
    own_taken = max(0, own_before - own_after)
    opp_taken = max(0, opp_before - opp_after)

    bonus = 0.0
    if archetype == ARCHETYPE_LUCARIO:
        # Prize-trade thesis: taking prizes is good; keeping a ready Mega Lucario
        # behind a single-prize wall is the winning board structure.
        bonus += 0.02 * own_taken
        if _benched_ready_mega(_player(obs_after, seat)):
            bonus += 0.01
        bonus -= 0.015 * opp_taken
    elif archetype == ARCHETYPE_DRAGAPULT:
        # Spread then convert: reward board spread and multi-KO turns, not early aggression.
        if phase == "mid" and _damaged_opp_bench_count(obs_after, seat) >= 2:
            bonus += 0.03
        if own_taken >= 2:
            bonus += 0.04  # multi-KO turn proxy
        elif own_taken == 1:
            bonus += 0.01
        bonus -= 0.015 * opp_taken
    elif archetype == ARCHETYPE_ABOMASNOW:
        bonus += 0.02 * own_taken
        if phase in ("mid", "late") and own_taken >= 1:
            bonus += 0.01
        bonus -= 0.015 * opp_taken
    elif archetype == ARCHETYPE_IONO:
        bonus += 0.018 * own_taken
        if _active_energy_count(_player(obs_after, seat)) >= 6:
            bonus += 0.01
        bonus -= 0.012 * opp_taken
    elif archetype == ARCHETYPE_STARMIE:
        if _opp_bench_count(obs_after, seat) >= 2:
            bonus += 0.02
        bonus += 0.015 * own_taken
        bonus -= 0.012 * opp_taken
    elif archetype == ARCHETYPE_CRUSTLE:
        bonus += 0.018 * own_taken
        if phase in ("mid", "late"):
            bonus += 0.01
        bonus -= 0.014 * opp_taken
    return float(max(-_VALUE_BONUS_CLAMP, min(_VALUE_BONUS_CLAMP, bonus)))


@dataclass(frozen=True)
class HeuristicKnobs:
    """Per-archetype policy prior, value shaping, and search-target mix strengths."""

    policy_beta: dict[str, float]
    value_shaping: dict[str, float]
    target_mix: dict[str, float]

    def policy_beta_for(self, archetype: str) -> float:
        return float(self.policy_beta.get(archetype, 0.0))

    def value_shaping_for(self, archetype: str) -> float:
        return float(self.value_shaping.get(archetype, 0.0))

    def target_mix_for(self, archetype: str) -> float:
        return float(self.target_mix.get(archetype, 0.0))

    def any_policy_beta(self) -> bool:
        return any(value > 0.0 for value in self.policy_beta.values())

    def any_value_shaping(self) -> bool:
        return any(value > 0.0 for value in self.value_shaping.values())

    def any_target_mix(self) -> bool:
        return any(value > 0.0 for value in self.target_mix.values())


def heuristic_knobs_from_settings(settings: dict[str, float]) -> HeuristicKnobs:
    """Build knobs from flat config settings (module defaults or env overrides)."""
    return HeuristicKnobs(
        policy_beta={
            ARCHETYPE_LUCARIO: float(settings.get("heuristic_policy_beta_lucario", 0.0)),
            ARCHETYPE_DRAGAPULT: float(settings.get("heuristic_policy_beta_dragapult", 0.0)),
            ARCHETYPE_ABOMASNOW: float(settings.get("heuristic_policy_beta_abomasnow", 0.0)),
            ARCHETYPE_IONO: float(settings.get("heuristic_policy_beta_iono", 0.0)),
            ARCHETYPE_STARMIE: float(settings.get("heuristic_policy_beta_starmie", 0.0)),
            ARCHETYPE_CRUSTLE: float(settings.get("heuristic_policy_beta_crustle", 0.0)),
        },
        value_shaping={
            ARCHETYPE_LUCARIO: float(settings.get("value_archetype_shaping_weight_lucario", 0.0)),
            ARCHETYPE_DRAGAPULT: float(settings.get("value_archetype_shaping_weight_dragapult", 0.0)),
            ARCHETYPE_ABOMASNOW: float(settings.get("value_archetype_shaping_weight_abomasnow", 0.0)),
            ARCHETYPE_IONO: float(settings.get("value_archetype_shaping_weight_iono", 0.0)),
            ARCHETYPE_STARMIE: float(settings.get("value_archetype_shaping_weight_starmie", 0.0)),
            ARCHETYPE_CRUSTLE: float(settings.get("value_archetype_shaping_weight_crustle", 0.0)),
        },
        target_mix={
            ARCHETYPE_LUCARIO: float(settings.get("heuristic_target_mix_lucario", 0.0)),
            ARCHETYPE_DRAGAPULT: float(settings.get("heuristic_target_mix_dragapult", 0.0)),
            ARCHETYPE_ABOMASNOW: float(settings.get("heuristic_target_mix_abomasnow", 0.0)),
            ARCHETYPE_IONO: float(settings.get("heuristic_target_mix_iono", 0.0)),
            ARCHETYPE_STARMIE: float(settings.get("heuristic_target_mix_starmie", 0.0)),
            ARCHETYPE_CRUSTLE: float(settings.get("heuristic_target_mix_crustle", 0.0)),
        },
    )


@dataclass(frozen=True)
class ArchetypeHeuristic:
    """Bound archetype + beta for one deck; builds per-observation action scorers."""

    archetype: str
    beta: float
    deck_cards: tuple[int, ...] = ()

    @property
    def active(self) -> bool:
        return self.archetype != ARCHETYPE_UNKNOWN and self.beta > 0.0

    def make_action_scorer(
        self,
        obs: dict[str, Any],
        seat: int,
    ) -> Callable[[list[int]], float]:
        phase = game_phase(obs, seat)
        opponent_archetype = predict_opponent_archetype(obs, seat)
        context = matchup_context(obs, seat, self.archetype)
        starmie_variant_key = (
            starmie_variant(self.deck_cards) if self.archetype == ARCHETYPE_STARMIE else ""
        )

        def scorer(action: list[int]) -> float:
            return score_action(
                action,
                obs,
                seat,
                archetype=self.archetype,
                phase=phase,
                context=context,
                opponent_archetype=opponent_archetype,
                starmie_variant_key=starmie_variant_key,
            )

        return scorer


def resolve_policy_beta(
    archetype: str,
    *,
    knobs: HeuristicKnobs | None = None,
    lucario_beta: float = 0.0,
    dragapult_beta: float = 0.0,
) -> float:
    """Per-archetype policy-prior strength."""
    if knobs is not None:
        return knobs.policy_beta_for(archetype)
    if archetype == ARCHETYPE_LUCARIO:
        return float(lucario_beta)
    if archetype == ARCHETYPE_DRAGAPULT:
        return float(dragapult_beta)
    return 0.0


def heuristic_for_deck(
    deck_cards: list[int] | tuple[int, ...] | None,
    *,
    knobs: HeuristicKnobs | None = None,
    lucario_beta: float = 0.0,
    dragapult_beta: float = 0.0,
) -> ArchetypeHeuristic:
    """Factory: classify the deck and bind the matching policy-prior strength."""
    archetype = classify_archetype(deck_cards)
    beta = resolve_policy_beta(
        archetype,
        knobs=knobs,
        lucario_beta=lucario_beta,
        dragapult_beta=dragapult_beta,
    )
    cards = tuple(int(card) for card in deck_cards) if deck_cards else ()
    return ArchetypeHeuristic(archetype=archetype, beta=beta, deck_cards=cards)


def resolve_value_shaping_weight(
    archetype: str,
    *,
    knobs: HeuristicKnobs | None = None,
    lucario_weight: float = 0.0,
    dragapult_weight: float = 0.0,
) -> float:
    """Per-archetype value-shaping blend weight."""
    if knobs is not None:
        return knobs.value_shaping_for(archetype)
    if archetype == ARCHETYPE_LUCARIO:
        return float(lucario_weight)
    if archetype == ARCHETYPE_DRAGAPULT:
        return float(dragapult_weight)
    return 0.0


def heuristic_knobs_from_config(config: dict[str, Any]) -> HeuristicKnobs:
    """Merge objective + self_play heuristic settings from a built config dict."""
    objective = config.get("objective") or {}
    self_play = config.get("self_play") or {}

    def pick(name: str) -> float:
        if name in objective:
            return float(objective[name])
        if name in self_play:
            return float(self_play[name])
        return 0.0

    return heuristic_knobs_from_settings({name: pick(name) for name in (
        "heuristic_policy_beta_lucario",
        "heuristic_policy_beta_dragapult",
        "heuristic_policy_beta_abomasnow",
        "heuristic_policy_beta_iono",
        "heuristic_policy_beta_starmie",
        "heuristic_policy_beta_crustle",
        "value_archetype_shaping_weight_lucario",
        "value_archetype_shaping_weight_dragapult",
        "value_archetype_shaping_weight_abomasnow",
        "value_archetype_shaping_weight_iono",
        "value_archetype_shaping_weight_starmie",
        "value_archetype_shaping_weight_crustle",
        "heuristic_target_mix_lucario",
        "heuristic_target_mix_dragapult",
        "heuristic_target_mix_abomasnow",
        "heuristic_target_mix_iono",
        "heuristic_target_mix_starmie",
        "heuristic_target_mix_crustle",
    )})


def resolve_target_mix(
    archetype: str,
    *,
    knobs: HeuristicKnobs | None = None,
) -> float:
    """Per-archetype heuristic mass blend into soft search-policy target."""
    if knobs is None:
        return 0.0
    return knobs.target_mix_for(archetype)
