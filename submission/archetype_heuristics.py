"""Archetype game-plan policy prior for Kaggle inference (self-contained).

Mirror of ``poke_agent/archetype_heuristics.py`` (policy-prior subset) with flat
imports for the trimmed submission package. Only the inference-relevant pieces are
included — value-target shaping is a training-time concern and is omitted here.
Keep the scoring logic in sync with the training module so search behaves identically.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from rewards import prize_count

try:
    from archetype_signatures_data import VISIBLE_ARCHETYPE_SIGNATURES, _WEIGHT_SCALE
except ImportError:
    VISIBLE_ARCHETYPE_SIGNATURES: dict[str, dict[int, int]] = {}
    _WEIGHT_SCALE = 10_000

ARCHETYPE_DRAGAPULT = "dragapult-ex"
ARCHETYPE_LUCARIO = "mega-lucario-ex"
ARCHETYPE_UNKNOWN = "unknown"

HEURISTIC_ARCHETYPE_SLUG_PATTERNS = {
    ARCHETYPE_DRAGAPULT: ["dragapult"],
    ARCHETYPE_LUCARIO: ["mega-lucario", "lucario-hariyama"],
}

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

OPPONENT_ARCHETYPE_SIGNATURES = {
    ARCHETYPE_DRAGAPULT: DRAGAPULT_SIGNATURE | DUSKNOIR_LINE,
    ARCHETYPE_LUCARIO: LUCARIO_SIGNATURE | LUCARIO_ENGINE | frozenset({MAKUHITA, HARIYAMA}),
}

_VISIBLE_MIN_SCORE = 0.15

def _deck_matches_archetype_patterns(slug: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if slug == pattern or slug.startswith(f"{pattern}-"):
            return True
    return False


def opponent_matches_family(opponent_slug: str, family: str) -> bool:
    if opponent_slug == ARCHETYPE_UNKNOWN:
        return False
    patterns = HEURISTIC_ARCHETYPE_SLUG_PATTERNS.get(family)
    if not patterns:
        return False
    return _deck_matches_archetype_patterns(opponent_slug, patterns)


def _our_family_patterns(our_archetype: str) -> list[str]:
    return list(HEURISTIC_ARCHETYPE_SLUG_PATTERNS.get(our_archetype) or [])

PHANTOM_DIVE_ATTACK = 154
ITCHY_POLLEN_ATTACK = 323
MEGA_BRAVE_ATTACK = 983

OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_ABILITY = 10
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14


def _seat_prize_counts(obs: dict[str, Any] | None, seat: int) -> tuple[int, int]:
    return prize_count(obs, seat), prize_count(obs, 1 - seat)


def classify_archetype(deck_cards: list[int] | tuple[int, ...] | None) -> str:
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


def _opp_bench_count(obs: dict[str, Any], seat: int) -> int:
    return len(_bench(_player(obs, 1 - seat)))


def game_phase(obs: dict[str, Any], seat: int) -> str:
    turn = int(_current(obs).get("turn", 0) or 0)
    own_prizes, _ = _seat_prize_counts(obs, seat)
    if turn <= 2 or own_prizes >= 6:
        return "early"
    if own_prizes <= 2:
        return "late"
    return "mid"


def _visible_card_counter(player: dict[str, Any]) -> Counter[int]:
    return Counter(_cards_in_play(player))


def _weighted_visible_score(visible: Counter[int], weights: dict[int, int]) -> float:
    if not visible or not weights:
        return 0.0
    matched = [(card, weights.get(card, 0)) for card in visible if weights.get(card, 0) > 0]
    if not matched:
        return 0.0
    hit = sum(visible[card] * weight for card, weight in matched)
    match_count = len(matched)
    coverage = match_count / len(visible)
    strength = (hit / match_count) / _WEIGHT_SCALE
    return coverage * strength


def _predict_opponent_from_deck_signatures(visible: Counter[int]) -> tuple[str, float]:
    best = ARCHETYPE_UNKNOWN
    best_score = 0.0
    for slug, raw_weights in VISIBLE_ARCHETYPE_SIGNATURES.items():
        score = _weighted_visible_score(visible, raw_weights)
        if score > best_score:
            best = slug
            best_score = score
    return best, best_score


def _predict_opponent_from_hardcoded(visible: Counter[int]) -> str:
    best = ARCHETYPE_UNKNOWN
    best_overlap = 0
    for candidate, signature in OPPONENT_ARCHETYPE_SIGNATURES.items():
        overlap = sum(1 for card in visible if card in signature)
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best if best_overlap > 0 else ARCHETYPE_UNKNOWN


def predict_opponent_archetype(obs: dict[str, Any], seat: int) -> str:
    visible = _visible_card_counter(_player(obs, 1 - seat))
    if not visible:
        return ARCHETYPE_UNKNOWN
    slug, score = _predict_opponent_from_deck_signatures(visible)
    if slug != ARCHETYPE_UNKNOWN and score >= _VISIBLE_MIN_SCORE:
        return slug
    return _predict_opponent_from_hardcoded(visible)


def matchup_context(obs: dict[str, Any], seat: int, archetype: str) -> str:
    opponent_slug = predict_opponent_archetype(obs, seat)
    our_patterns = _our_family_patterns(archetype)
    if our_patterns and _deck_matches_archetype_patterns(opponent_slug, our_patterns):
        return "mirror"
    own_prizes, opp_prizes = _seat_prize_counts(obs, seat)
    turn = int(_current(obs).get("turn", 0) or 0)
    if opp_prizes < own_prizes and turn <= 4:
        return "aggro"
    return "default"


def _select_options(obs: dict[str, Any]) -> list[dict[str, Any]]:
    select = obs.get("select") or {}
    return [opt for opt in (select.get("option") or []) if isinstance(opt, dict)]


def _score_lucario_option(
    option: dict[str, Any], player: dict[str, Any], *, phase: str, opponent_archetype: str
) -> float:
    otype = int(option.get("type", -1))
    vs_dragapult = opponent_matches_family(opponent_archetype, ARCHETYPE_DRAGAPULT)
    if otype == OPTION_ATTACK:
        if int(option.get("attackId", -1)) == MEGA_BRAVE_ATTACK:
            return 1.0 if phase in ("mid", "late") else 0.5
        return 0.6
    if otype == OPTION_EVOLVE:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == HARIYAMA:
            return 0.95 if vs_dragapult else 0.9
        if card_id == MEGA_LUCARIO_EX:
            return 0.9
        return 0.6
    if otype == OPTION_ABILITY:
        return 0.7
    if otype == OPTION_PLAY:
        card_id = _hand_card_id(player, option.get("index"))
        if card_id == BOSS_ORDERS:
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
    otype = int(option.get("type", -1))
    vs_lucario = opponent_matches_family(opponent_archetype, ARCHETYPE_LUCARIO)
    if otype == OPTION_ATTACK:
        attack_id = int(option.get("attackId", -1))
        if attack_id == PHANTOM_DIVE_ATTACK:
            base = 0.8 if _opp_bench_count(obs, seat) >= 2 else 0.7
            if vs_lucario and _opp_bench_count(obs, seat) >= 2:
                base = 0.85
            if context == "mirror" and phase == "early":
                base *= 0.4
            return base
        if attack_id == ITCHY_POLLEN_ATTACK:
            return 0.7 if phase == "early" else 0.4
        return 0.55
    if otype == OPTION_ABILITY:
        return 0.65
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
                return 0.7 if vs_lucario else 0.6
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
                option,
                obs,
                player,
                seat,
                phase=phase,
                context=context,
                opponent_archetype=opponent_archetype,
            )
        counted += 1
    if counted == 0:
        return 0.0
    return total / counted


def resolve_policy_beta(archetype: str, *, lucario_beta: float, dragapult_beta: float) -> float:
    if archetype == ARCHETYPE_LUCARIO:
        return float(lucario_beta)
    if archetype == ARCHETYPE_DRAGAPULT:
        return float(dragapult_beta)
    return 0.0


@dataclass(frozen=True)
class ArchetypeHeuristic:
    archetype: str
    beta: float

    @property
    def active(self) -> bool:
        return self.archetype != ARCHETYPE_UNKNOWN and self.beta > 0.0

    def make_action_scorer(self, obs: dict[str, Any], seat: int) -> Callable[[list[int]], float]:
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


def heuristic_for_deck(
    deck_cards: list[int] | tuple[int, ...] | None,
    *,
    lucario_beta: float,
    dragapult_beta: float,
) -> ArchetypeHeuristic:
    archetype = classify_archetype(deck_cards)
    beta = resolve_policy_beta(archetype, lucario_beta=lucario_beta, dragapult_beta=dragapult_beta)
    return ArchetypeHeuristic(archetype=archetype, beta=beta)
