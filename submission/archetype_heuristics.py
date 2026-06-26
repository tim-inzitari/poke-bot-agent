"""Archetype game-plan policy prior for Kaggle inference (self-contained).

Mirror of ``poke_agent/archetype_heuristics.py`` (policy-prior subset) with flat
imports for the trimmed submission package. Value-target shaping is training-only.
Keep scoring logic in sync with the training module.
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

HEURISTIC_ARCHETYPE_SLUG_PATTERNS: dict[str, list[str]] = {
    "dragapult-ex": ["dragapult"],
    "mega-lucario-ex": ["mega-lucario", "lucario-hariyama"],
    "mega-abomasnow-ex": ["mega-abomasnow", "abomasnow"],
    "iono": ["iono", "bellibolt"],
    "starmie": ["starmie"],
    "crustle": ["crustle"],
}


def _deck_matches_archetype_patterns(slug: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if slug == pattern or slug.startswith(f"{pattern}-"):
            return True
    return False


def _seat_prize_counts(obs: dict[str, Any] | None, seat: int) -> tuple[int, int]:
    return prize_count(obs, seat), prize_count(obs, 1 - seat)




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
    own_prizes, _ = _seat_prize_counts(obs, seat=seat)
    if turn <= 2 or own_prizes >= 6:
        return "early"
    if own_prizes <= 2:
        return "late"
    return "mid"


def _visible_card_counter(player: dict[str, Any]) -> Counter[int]:
    return Counter(_cards_in_play(player))



def _predict_opponent_from_signatures(visible: Counter[int]) -> str:
    best = ARCHETYPE_UNKNOWN
    best_overlap = 0
    for candidate, signature in OPPONENT_ARCHETYPE_SIGNATURES.items():
        overlap = sum(1 for card in visible if card in signature)
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best if best_overlap > 0 else ARCHETYPE_UNKNOWN



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


def predict_opponent_archetype(obs: dict[str, Any], seat: int) -> str:
    visible = _visible_card_counter(_player(obs, 1 - seat))
    if not visible:
        return ARCHETYPE_UNKNOWN
    slug, score = _predict_opponent_from_deck_signatures(visible)
    if slug != ARCHETYPE_UNKNOWN and score >= _VISIBLE_MIN_SCORE:
        return slug
    return _predict_opponent_from_signatures(visible)


def matchup_context(obs: dict[str, Any], seat: int, archetype: str) -> str:
    """Opponent-shape context from predicted meta slug and prize pace."""
    opponent_slug = predict_opponent_archetype(obs, seat)
    our_patterns = _our_family_patterns(archetype)
    if our_patterns and _deck_matches_archetype_patterns(opponent_slug, our_patterns):
        return "mirror"
    own_prizes, opp_prizes = _seat_prize_counts(obs, seat=seat)
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


def resolve_policy_beta(
    archetype: str,
    *,
    knobs: HeuristicKnobs | None = None,
    lucario_beta: float = 0.0,
    dragapult_beta: float = 0.0,
) -> float:
    if knobs is not None:
        return knobs.policy_beta_for(archetype)
    if archetype == ARCHETYPE_LUCARIO:
        return float(lucario_beta)
    if archetype == ARCHETYPE_DRAGAPULT:
        return float(dragapult_beta)
    return 0.0


@dataclass(frozen=True)
class HeuristicKnobs:
    policy_beta: dict[str, float]
    value_shaping: dict[str, float]
    target_mix: dict[str, float]

    def policy_beta_for(self, archetype: str) -> float:
        return float(self.policy_beta.get(archetype, 0.0))

    def any_policy_beta(self) -> bool:
        return any(value > 0.0 for value in self.policy_beta.values())


@dataclass(frozen=True)
class ArchetypeHeuristic:
    archetype: str
    beta: float
    deck_cards: tuple[int, ...] = ()

    @property
    def active(self) -> bool:
        return self.archetype != ARCHETYPE_UNKNOWN and self.beta > 0.0

    def make_action_scorer(self, obs: dict[str, Any], seat: int) -> Callable[[list[int]], float]:
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


def heuristic_for_deck(
    deck_cards: list[int] | tuple[int, ...] | None,
    *,
    knobs: HeuristicKnobs | None = None,
    lucario_beta: float = 0.0,
    dragapult_beta: float = 0.0,
    abomasnow_beta: float = 0.0,
    iono_beta: float = 0.0,
    starmie_beta: float = 0.0,
    crustle_beta: float = 0.0,
) -> ArchetypeHeuristic:
    archetype = classify_archetype(deck_cards)
    if knobs is not None:
        beta = knobs.policy_beta_for(archetype)
    else:
        beta = resolve_policy_beta(
            archetype,
            knobs=HeuristicKnobs(
                policy_beta={
                    ARCHETYPE_LUCARIO: lucario_beta,
                    ARCHETYPE_DRAGAPULT: dragapult_beta,
                    ARCHETYPE_ABOMASNOW: abomasnow_beta,
                    ARCHETYPE_IONO: iono_beta,
                    ARCHETYPE_STARMIE: starmie_beta,
                    ARCHETYPE_CRUSTLE: crustle_beta,
                },
                value_shaping={},
                target_mix={},
            ),
        )
    cards = tuple(int(card) for card in deck_cards) if deck_cards else ()
    return ArchetypeHeuristic(archetype=archetype, beta=beta, deck_cards=cards)
