"""Observation featurization.

Ports the *featurization approach* from the official RL+MCTS sample
(``kiyotah/reinforcement-learning-and-mcts-sample-code``) onto clean, stable
functions. It does **not** contain the model — only the token builders the
model phase will consume:

  - :func:`build_board_tokens` -> spatial board tokens (24 EmbeddingBag "words"):
    self/opp bench x8, self/opp active, self/opp player summary, hand, deck,
    stadium, global.
  - :func:`build_option_tokens` -> per-candidate-action tokens keyed by
    OptionType / attackId / cardId / SelectContext.
  - :func:`enumerate_action_combos` -> the multi-select combos to score.

Both builders return a :class:`SparseVector` (index/value/offset lists), which
is exactly the input an ``EmbeddingBag`` expects and is trivially convertible to
dense tensors later. No torch/numpy dependency here.

Vocabulary sizes come from the live engine (``all_card_data``/``all_attack``):
card vocab = max(cardId)+1 (currently 1268), attack vocab = max(attackId)+1.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional, Union

from . import cg_env

# ---------------------------------------------------------------------------
# Layout constants (match the sample's encoder/decoder word structure)
# ---------------------------------------------------------------------------

#: Number of spatial board tokens ("words") produced per observation.
NUM_BOARD_TOKENS: int = 24

#: Number of SelectContext.MAIN option sub-features (PLAY, ATTACH-from/to,
#: EVOLVE-from/to, ABILITY, DISCARD, RETREAT).
DECODER_MAIN_FEATURE: int = 8

#: First decoder index reserved for attack-id features (0..13 are typed flags:
#: none / END / YES / NO / 5x SpecialCondition / 5x Number bucket).
DECODER_ATTACK_OFFSET: int = 14

#: Hard materialization ceiling. Trusted callers fail if the complete ordered
#: action space exceeds this bound; they never train/evaluate on a truncated set.
MAX_ACTION_COMBOS: int = 4096

#: Bump whenever feature indices or action enumeration semantics change. It is
#: included in dataset cache keys so incompatible pickles are never reused.
FEATURE_SCHEMA_VERSION: int = 4


class ActionSpaceTooLarge(RuntimeError):
    """Complete ordered legal action space exceeds the safe decoder ceiling."""


def _selection_bounds(obs) -> tuple[object, int, int, int]:
    if isinstance(obs, dict):
        sel = obs.get("select") or {}
        n_opt = len(sel.get("option") or [])
        lo = max(0, min(int(sel.get("minCount", 0)), n_opt))
        hi = max(lo, min(int(sel.get("maxCount", 0)), n_opt))
        return obs, n_opt, lo, hi
    sel = obs.select
    n_opt = len(sel.option)
    lo = max(0, min(int(sel.minCount), n_opt))
    hi = max(lo, min(int(sel.maxCount), n_opt))
    return obs, n_opt, lo, hi


def factorized_action_candidates(obs, prefix: list[int]) -> list[list[int]]:
    """Return one autoregressive stage with complete legal support.

    Candidates append any remaining option in order. Once ``minCount`` is met,
    the unchanged prefix is an explicit STOP candidate. Repeated application
    assigns support to every legal ordered selection without materializing all
    permutations.
    """
    _obs, n_opt, lo, hi = _selection_bounds(obs)
    prefix = [int(i) for i in prefix]
    if len(prefix) != len(set(prefix)) or any(i < 0 or i >= n_opt for i in prefix):
        raise ValueError("invalid factorized action prefix")
    if len(prefix) > hi:
        raise ValueError("factorized action prefix exceeds maxCount")
    if len(prefix) >= hi:
        return [prefix]
    candidates = [prefix + [i] for i in range(n_opt) if i not in prefix]
    if len(prefix) >= lo:
        candidates.append(prefix)  # explicit STOP
    if not candidates:
        return [prefix]
    return candidates


def ordered_action_count(obs) -> int:
    """Count complete ordered legal actions without materializing permutations."""
    _obs, n_opt, lo, hi = _selection_bounds(obs)
    return sum(math.perm(n_opt, k) for k in range(lo, hi + 1))


def factorized_teacher_forcing_stages(
    obs, action: list[int]
) -> list[tuple[list[list[int]], int]]:
    """Canonical autoregressive candidate/target stages for a legal action."""
    _obs, n_opt, lo, hi = _selection_bounds(obs)
    target = [int(i) for i in action]
    if (
        len(target) < lo
        or len(target) > hi
        or len(target) != len(set(target))
        or any(i < 0 or i >= n_opt for i in target)
    ):
        raise ValueError(
            f"illegal ordered action length/content: action={target}, "
            f"n_options={n_opt}, bounds=[{lo}, {hi}]"
        )
    stages: list[tuple[list[list[int]], int]] = []
    prefix: list[int] = []
    for next_option in target:
        candidates = factorized_action_candidates(obs, prefix)
        chosen = prefix + [next_option]
        stages.append((candidates, candidates.index(chosen)))
        prefix = chosen
    if len(prefix) < hi:
        candidates = factorized_action_candidates(obs, prefix)
        stages.append((candidates, candidates.index(prefix)))
    if not stages:
        stages.append(([[]], 0))
    return stages


class ActionCombos(list):
    """List-compatible legal actions with explicit truncation diagnostics."""

    def __init__(
        self,
        values=(),
        *,
        total_count: int,
        min_count: int,
        max_count: int,
    ) -> None:
        super().__init__(values)
        self.total_count = int(total_count)
        self.min_count = int(min_count)
        self.max_count = int(max_count)
        self.truncated = len(self) < self.total_count

# ---------------------------------------------------------------------------
# Vocabulary (cached from the live cg library)
# ---------------------------------------------------------------------------

_CARD_COUNT: Optional[int] = None
_ATTACK_COUNT: Optional[int] = None
_CARD_TABLE: Optional[dict] = None


def _ensure_vocab() -> None:
    global _CARD_COUNT, _ATTACK_COUNT, _CARD_TABLE
    if _CARD_COUNT is None:
        cards = cg_env.all_card_data()
        _CARD_TABLE = {c.cardId: c for c in cards}
        _CARD_COUNT = max((c.cardId for c in cards), default=0) + 1
        _ATTACK_COUNT = max((a.attackId for a in cg_env.all_attack()), default=0) + 1


def card_vocab_size() -> int:
    """Card-embedding table size = max(cardId) + 1 from ``all_card_data()``."""
    _ensure_vocab()
    assert _CARD_COUNT is not None
    return _CARD_COUNT


def attack_vocab_size() -> int:
    """Attack-embedding table size = max(attackId) + 1 from ``all_attack()``."""
    _ensure_vocab()
    assert _ATTACK_COUNT is not None
    return _ATTACK_COUNT


def card_table() -> dict:
    """Return ``{cardId: CardData}`` for all cards."""
    _ensure_vocab()
    assert _CARD_TABLE is not None
    return _CARD_TABLE


def _pokemon_feature_width(card_count: int) -> int:
    # add_single(empty flag) + [hp] + card + tools-bag + energy-bag
    return 2 + 3 * card_count


def encoder_vocab_size(headroom: int = 512) -> int:
    """Exact EmbeddingBag vocab needed by :func:`build_board_tokens` (+headroom).

    The board encoding advances a running feature index; this returns the total
    span plus a little headroom so future card additions don't overflow.
    """
    cc = card_vocab_size()
    poke = _pokemon_feature_width(cc)
    # bench: pos advances once per player (slot-0..6 share via pos reset trick)
    span = 2 * poke          # bench (2 players, 1 poke width each)
    span += 2 * poke         # active (2 players)
    player_width = 7 + 5 + cc  # counts(7) + status(5) + discard bag(card_count)
    span += 2 * player_width  # player summaries
    span += cc               # hand bag
    span += cc               # deck bag
    span += cc               # stadium bag
    span += 5                # global (bias, turn, first-player flag, seat one-hot)
    return span + headroom


def decoder_vocab_size() -> int:
    """EmbeddingBag vocab needed by :func:`build_option_tokens`.

    14 typed flags + attack ids + (1 + 8 main features + 49 SelectContexts)
    card-id blocks. SelectContext currently tops out at 48
    (RECOVER_SPECIAL_CONDITION), so 49 contexts are reserved.
    """
    cc = card_vocab_size()
    ac = attack_vocab_size()
    card_offset = DECODER_ATTACK_OFFSET + ac
    n_contexts = int(cg_env.SelectContext.RECOVER_SPECIAL_CONDITION) + 1
    return card_offset + (1 + DECODER_MAIN_FEATURE + n_contexts) * cc


# ---------------------------------------------------------------------------
# SparseVector: EmbeddingBag input (index / value / offset)
# ---------------------------------------------------------------------------

class SparseVector:
    """Sparse multi-hot bag-of-features spanning one or more "words".

    ``index``/``value`` are parallel arrays of active (feature-index, weight)
    pairs; ``offset`` marks where each word begins in ``index``. ``pos`` is the
    running base index used while building a word.
    """

    __slots__ = ("index", "value", "offset", "pos")

    def __init__(self) -> None:
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []
        self.pos: int = 0

    def add(self, index: int, value: Union[float, int, bool]) -> None:
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int) -> None:
        self.pos += pos

    def add_single(self, value: Union[float, int, bool]) -> None:
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self) -> None:
        self.offset.append(len(self.index))

    @property
    def num_words(self) -> int:
        return len(self.offset)


# ---------------------------------------------------------------------------
# Encoder (spatial board) feature builders
# ---------------------------------------------------------------------------

def _add_card(sv: SparseVector, card, card_count: int) -> None:
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def _add_cards(sv: SparseVector, cards, value: float, card_count: int) -> None:
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


def _add_pokemon(sv: SparseVector, poke, card_count: int) -> None:
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        _add_card(sv, poke, card_count)
        _add_cards(sv, poke.tools, 1.0, card_count)
        _add_cards(sv, poke.energyCards, 0.5, card_count)


def _add_player(sv: SparseVector, ps, card_count: int) -> None:
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    _add_cards(sv, ps.discard, 0.25, card_count)


def build_board_tokens(obs, your_deck: list[int]) -> SparseVector:
    """Build the 24 spatial board tokens for observation ``obs``.

    ``obs`` may be a raw observation dict or a :class:`cg.api.Observation`.
    ``your_deck`` is the flat 60-card id list of the agent's own deck (used as a
    bag feature so the network knows our archetype's card pool).
    """
    if isinstance(obs, dict):
        obs = cg_env.to_observation(obs)
    cc = card_vocab_size()
    state = obs.current
    your_index = state.yourIndex

    sv = SparseVector()

    # Bench: 8 slots per player. Slots 0..6 reuse the same feature span (pos
    # reset trick), only slot 7 advances pos -> one poke width per player.
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                _add_pokemon(sv, ps.bench[j], cc)
            else:
                _add_pokemon(sv, None, cc)
            if j != 7:
                sv.pos = pos

    # Active (self then opp).
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if len(ps.active) > 0:
            _add_pokemon(sv, ps.active[0], cc)
        else:
            _add_pokemon(sv, None, cc)

    # Player summaries.
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        _add_player(sv, ps, cc)

    # Own hand (bag).
    sv.word_start()
    _add_cards(sv, state.players[your_index].hand, 0.25, cc)

    # Own deck pool (bag).
    sv.word_start()
    for cid in your_deck:
        sv.add(cid, 0.25)
    sv.add_pos(cc)

    # Stadium.
    sv.word_start()
    _add_cards(sv, state.stadium, 1.0, cc)

    # Global scalars.
    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    sv.add(your_index, 1)
    sv.add_pos(2)
    return sv


# ---------------------------------------------------------------------------
# Decoder (option / action) feature builders
# ---------------------------------------------------------------------------

def get_card(obs, area, index: int, player_index: int):
    """Resolve the Card/Pokemon referenced by (area, index, player_index)."""
    AreaType = cg_env.AreaType
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def _decoder_card_offset() -> int:
    return DECODER_ATTACK_OFFSET + attack_vocab_size()


def _decoder_main(
    sv: SparseVector, feature_index: int, card, cc: int, weight: float = 1.0
) -> None:
    if card is not None:
        sv.add(_decoder_card_offset() + feature_index * cc + card.id, weight)


def _decoder_card_id(
    sv: SparseVector, context, card_id: int, cc: int, weight: float = 1.0
) -> None:
    sv.add(
        _decoder_card_offset() + (DECODER_MAIN_FEATURE + int(context)) * cc + card_id,
        weight,
    )


def _decoder_card(
    sv: SparseVector, context, card, cc: int, weight: float = 1.0
) -> None:
    if card is not None:
        _decoder_card_id(sv, context, card.id, cc, weight)


def build_option_tokens(obs, action_combos: list[list[int]]) -> SparseVector:
    """Build one option token per candidate action combo.

    ``action_combos`` is a list of option-index lists (see
    :func:`enumerate_action_combos`). Each combo becomes one bag-word encoding
    the involved OptionType(s), attack/card ids, and SelectContext.
    """
    if isinstance(obs, dict):
        obs = cg_env.to_observation(obs)
    OptionType = cg_env.OptionType
    cc = card_vocab_size()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context

    sv = SparseVector()
    for action in action_combos:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1)
            continue
        for rank, i in enumerate(action):
            o = obs.select.option[i]
            t = o.type
            # A weighted positional code keeps ordered selections distinct
            # without changing checkpoint embedding-table dimensions.
            weight = float(rank + 1)
            if t == OptionType.END:
                sv.add(1, weight)
            elif t == OptionType.YES:
                sv.add(2, weight)
            elif t == OptionType.NO:
                sv.add(3, weight)
            elif t == OptionType.SPECIAL_CONDITION:
                sv.add(4 + int(o.specialConditionType), weight)
            elif t == OptionType.NUMBER:
                sv.add(9 + min(o.number, 4), weight)
            elif t == OptionType.ATTACK:
                sv.add(DECODER_ATTACK_OFFSET + o.attackId, weight)
            elif t == OptionType.PLAY:
                _decoder_main(sv, 0, ps.hand[o.index], cc, weight)
            elif t == OptionType.ATTACH:
                _decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index), cc, weight)
                _decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index), cc, weight)
            elif t == OptionType.EVOLVE:
                _decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index), cc, weight)
                _decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index), cc, weight)
            elif t == OptionType.ABILITY:
                _decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index), cc, weight)
            elif t == OptionType.DISCARD:
                _decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index), cc, weight)
            elif t == OptionType.RETREAT:
                _decoder_main(sv, 7, ps.active[0], cc, weight)
            elif t == OptionType.CARD:
                _decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex), cc, weight)
            elif t == OptionType.TOOL_CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.tools[o.toolIndex], cc, weight)
            elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                card = get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.energyCards[o.energyIndex], cc, weight)
            elif t == OptionType.SKILL:
                _decoder_card_id(sv, context, o.cardId, cc, weight)
    return sv


def enumerate_action_combos(obs, max_combos: int = MAX_ACTION_COMBOS) -> ActionCombos:
    """Enumerate the complete ordered legal action set or fail.

    The engine consumes a sequence of option indices, so permutations are
    distinct legal actions. If materializing all sizes in ``[minCount,
    maxCount]`` exceeds ``max_combos``, :class:`ActionSpaceTooLarge` is raised.
    Trusted play/training therefore cannot silently omit strategic actions.
    """
    obs, n_opt, lo, hi = _selection_bounds(obs)
    counts = list(range(lo, hi + 1))
    total = ordered_action_count(obs)
    cap = max(0, int(max_combos))
    if total > cap:
        raise ActionSpaceTooLarge(
            f"complete ordered action space has {total} actions "
            f"(n_options={n_opt}, counts={counts}), cap={cap}"
        )
    combos = [
        list(action)
        for k in counts
        for action in itertools.permutations(range(n_opt), k)
    ]
    return ActionCombos(
        combos,
        total_count=total,
        min_count=lo,
        max_count=hi,
    )


# ---------------------------------------------------------------------------
# Information-set integrity
# ---------------------------------------------------------------------------

def assert_info_set(obs) -> None:
    """Assert ``obs`` is a valid acting-seat information set.

    Opponent hand must be hidden (``None``); face-down prizes are allowed as
    ``None`` entries. Raises ``AssertionError`` if private opponent fields leak
    into the observation used for policy / in-search value inputs.
    """
    if isinstance(obs, dict):
        obs = cg_env.to_observation(obs)
    state = obs.current
    if state is None:
        return
    yi = state.yourIndex
    opp = state.players[1 - yi]
    assert opp.hand is None, (
        "info-set violation: opponent hand is visible (must be None for "
        "policy/value inputs)"
    )
