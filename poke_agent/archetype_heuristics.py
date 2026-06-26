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
from typing import Any, Callable

from poke_agent.rewards import seat_prize_counts

# --- Archetype identities ---
ARCHETYPE_DRAGAPULT = "dragapult-ex"
ARCHETYPE_LUCARIO = "mega-lucario-ex"
ARCHETYPE_UNKNOWN = "unknown"

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

# Opponent archetypes we can recognize from visible board card IDs. The gameplan
# branches on the *predicted* opponent so we adapt before their deck is fully revealed.
OPPONENT_ARCHETYPE_SIGNATURES: dict[str, frozenset[int]] = {
    ARCHETYPE_DRAGAPULT: DRAGAPULT_SIGNATURE | DUSKNOIR_LINE,
    ARCHETYPE_LUCARIO: LUCARIO_SIGNATURE | LUCARIO_ENGINE | frozenset({MAKUHITA, HARIYAMA}),
}

# --- Attack IDs ---
PHANTOM_DIVE_ATTACK = 154
ITCHY_POLLEN_ATTACK = 323
MEGA_BRAVE_ATTACK = 983

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
    dragapult = sum(counts[card] for card in DRAGAPULT_SIGNATURE)
    lucario = sum(counts[card] for card in LUCARIO_SIGNATURE)
    if dragapult >= 3 and dragapult >= lucario:
        return ARCHETYPE_DRAGAPULT
    if lucario >= 2 and lucario > dragapult:
        return ARCHETYPE_LUCARIO
    return ARCHETYPE_UNKNOWN


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


def predict_opponent_archetype(obs: dict[str, Any], seat: int) -> str:
    """Best-guess opponent archetype from their visible Pokémon (active + bench)."""
    opp_ids = set(_cards_in_play(_player(obs, 1 - seat)))
    if not opp_ids:
        return ARCHETYPE_UNKNOWN
    best = ARCHETYPE_UNKNOWN
    best_overlap = 0
    for candidate, signature in OPPONENT_ARCHETYPE_SIGNATURES.items():
        overlap = len(opp_ids & signature)
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best


def matchup_context(obs: dict[str, Any], seat: int, archetype: str) -> str:
    """Opponent-shape context from predicted archetype and prize pace."""
    opponent = predict_opponent_archetype(obs, seat)
    if archetype != ARCHETYPE_UNKNOWN and opponent == archetype:
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
    vs_dragapult = opponent_archetype == ARCHETYPE_DRAGAPULT
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
    vs_lucario = opponent_archetype == ARCHETYPE_LUCARIO
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


def score_action(
    action: list[int],
    obs: dict[str, Any],
    seat: int,
    *,
    archetype: str,
    phase: str,
    context: str,
    opponent_archetype: str = ARCHETYPE_UNKNOWN,
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
        else:
            total += _score_dragapult_option(
                option, obs, player, seat, phase=phase, context=context, opponent_archetype=opponent_archetype
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
    else:  # Dragapult
        # Spread then convert: reward board spread and multi-KO turns, not early aggression.
        if phase == "mid" and _damaged_opp_bench_count(obs_after, seat) >= 2:
            bonus += 0.03
        if own_taken >= 2:
            bonus += 0.04  # multi-KO turn proxy
        elif own_taken == 1:
            bonus += 0.01
        bonus -= 0.015 * opp_taken
    return float(max(-_VALUE_BONUS_CLAMP, min(_VALUE_BONUS_CLAMP, bonus)))


@dataclass(frozen=True)
class ArchetypeHeuristic:
    """Bound archetype + beta for one deck; builds per-observation action scorers."""

    archetype: str
    beta: float

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

        def scorer(action: list[int]) -> float:
            return score_action(
                action,
                obs,
                seat,
                archetype=self.archetype,
                phase=phase,
                context=context,
                opponent_archetype=opponent_archetype,
            )

        return scorer


def resolve_policy_beta(
    archetype: str,
    *,
    lucario_beta: float,
    dragapult_beta: float,
) -> float:
    """Per-archetype policy-prior strength (Lucario tolerates a stronger prior)."""
    if archetype == ARCHETYPE_LUCARIO:
        return float(lucario_beta)
    if archetype == ARCHETYPE_DRAGAPULT:
        return float(dragapult_beta)
    return 0.0


def heuristic_for_deck(
    deck_cards: list[int] | tuple[int, ...] | None,
    *,
    lucario_beta: float,
    dragapult_beta: float,
) -> ArchetypeHeuristic:
    """Factory: classify the deck and bind the matching policy-prior strength."""
    archetype = classify_archetype(deck_cards)
    beta = resolve_policy_beta(
        archetype,
        lucario_beta=lucario_beta,
        dragapult_beta=dragapult_beta,
    )
    return ArchetypeHeuristic(archetype=archetype, beta=beta)


def resolve_value_shaping_weight(
    archetype: str,
    *,
    lucario_weight: float,
    dragapult_weight: float,
) -> float:
    """Per-archetype value-shaping blend weight."""
    if archetype == ARCHETYPE_LUCARIO:
        return float(lucario_weight)
    if archetype == ARCHETYPE_DRAGAPULT:
        return float(dragapult_weight)
    return 0.0
