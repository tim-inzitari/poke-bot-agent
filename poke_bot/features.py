"""Observation featurization.

Ports the *featurization approach* from the official RL+MCTS sample
(``kiyotah/reinforcement-learning-and-mcts-sample-code``) onto clean, stable
functions. It does **not** contain the model — only the token builders the
model phase will consume:

  - :func:`build_board_tokens` -> spatial board tokens (24 EmbeddingBag "words"):
    self/opp bench x8, self/opp active, self/opp player summary, hand, deck,
    stadium, global.
  - :func:`build_option_tokens` -> per-candidate-action tokens keyed by
    OptionType / attackId / cardId / SelectContext plus explicit source/target
    owner, area, and slot bindings.
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
from typing import Any, Mapping, Optional, Sequence, Union

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

#: Option bindings are encoded after the legacy card/context blocks.  Every
#: role-specific (owner, area, index) tuple gets one composite row.  A composite
#: row is necessary: separate additive marginals can collapse two multi-select
#: actions that pair the same owners/areas/indices differently.
DECODER_BINDING_ROLE_COUNT: int = 4
DECODER_BINDING_OWNER_COUNT: int = 3  # acting seat / opponent / unspecified
DECODER_BINDING_AREA_COUNT: int = 13  # unknown=0, AreaType values are 1..12
DECODER_BINDING_INDEX_COUNT: int = 65  # unknown=0, exact engine indices 0..63
DECODER_BINDING_TUPLES_PER_ROLE: int = (
    DECODER_BINDING_OWNER_COUNT
    * DECODER_BINDING_AREA_COUNT
    * DECODER_BINDING_INDEX_COUNT
)
DECODER_BINDING_VOCAB_SIZE: int = (
    DECODER_BINDING_ROLE_COUNT * DECODER_BINDING_TUPLES_PER_ROLE
)

# Binding roles.  These are plain ints so importing this module does not force
# the optional competition ``cg`` runtime to load.
DECODER_BINDING_SOURCE: int = 0
DECODER_BINDING_TARGET: int = 1
DECODER_BINDING_TOOL: int = 2
DECODER_BINDING_ENERGY: int = 3

#: Hard materialization ceiling. Trusted callers fail if the complete ordered
#: action space exceeds this bound; they never train/evaluate on a truncated set.
MAX_ACTION_COMBOS: int = 65536

#: Bump whenever feature indices or action enumeration semantics change. It is
#: included in dataset cache keys so incompatible pickles are never reused.
FEATURE_SCHEMA_VERSION: int = 5


class ActionSpaceTooLarge(RuntimeError):
    """Complete ordered legal action space exceeds the safe decoder ceiling."""


class FeatureContractError(ValueError):
    """An observation or option cannot be represented without ambiguity."""


def _enum_token(value: object) -> str:
    """Normalize a competition enum without importing the model runtime."""

    if hasattr(value, "name"):
        value = getattr(value, "name")
    return "".join(character for character in str(value).lower() if character.isalnum())


def forced_go_first_action(obs_dict: dict) -> Optional[list[int]]:
    """Return the legal ``Yes`` action for the external turn-order prompt.

    This small parser deliberately lives in the torch-free feature module: the
    submission boundary and isolated evaluation runner must resolve the
    engine-mandated prompt without importing a candidate model.  A recognized
    prompt fails closed unless it exposes exactly one legal ``Yes`` option.
    """

    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(select, dict):
        return None
    raw_context = select.get("context")
    try:
        is_first = int(raw_context) == 41
    except (TypeError, ValueError):
        is_first = _enum_token(raw_context) == "isfirst"
    if not is_first:
        return None
    options = select.get("option") or []
    yes_indices: list[int] = []
    for index, option in enumerate(options):
        raw_type = (
            option.get("type")
            if isinstance(option, dict)
            else getattr(option, "type", None)
        )
        try:
            is_yes = int(raw_type) == 1
        except (TypeError, ValueError):
            is_yes = _enum_token(raw_type) == "yes"
        if is_yes:
            yes_indices.append(index)
    lo = int(select.get("minCount", 0) or 0)
    hi = min(int(select.get("maxCount", 0) or 0), len(options))
    if len(yes_indices) != 1 or not (lo <= 1 <= hi):
        raise RuntimeError(
            "IsFirst prompt does not expose one legal Yes option; refusing "
            "to choose turn order ambiguously"
        )
    return [yes_indices[0]]


def _exact_int(value, *, field: str) -> int:
    """Convert enum/integer-like engine values without lossy coercion."""
    if isinstance(value, bool):
        raise FeatureContractError(f"{field} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureContractError(f"{field} is not an integer: {value!r}") from exc
    try:
        exact = bool(value == result)
    except Exception:  # pragma: no cover - defensive for foreign scalar types
        exact = False
    if not exact:
        raise FeatureContractError(f"{field} is not an exact integer: {value!r}")
    return result


def _enum_int(value, enum_type, *, field: str) -> int:
    """Resolve either IntEnum values or official JSON enum-name strings.

    The competition ``to_observation_class`` recursively creates dataclasses
    but intentionally leaves enum scalar fields in their JSON form (for
    example ``"IsFirst"``, ``"ToolCard"`` and ``"Hand"``).  Normalizing by
    alphanumeric enum name keeps live integer observations and Kaggle visual
    traces on the exact same feature rows.
    """
    if isinstance(value, str):
        normalized = "".join(character for character in value if character.isalnum())
        normalized = normalized.casefold()
        for member in enum_type:
            member_name = "".join(
                character for character in member.name if character.isalnum()
            ).casefold()
            if normalized == member_name:
                return int(member)
        raise FeatureContractError(f"unsupported {field} name: {value!r}")
    return _exact_int(value, field=field)


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


def complete_ordered_action_space_summary(
    obs: Any,
    *,
    max_combos: int,
) -> dict[str, Any]:
    """Describe an action space without enumerating any complete action.

    This is intentionally stricter than :func:`_selection_bounds`.  The
    latter predates the r198 audit boundary and accepts coercible engine
    values before clamping them.  An immutable evaluator trace must instead
    bind the *raw* JSON-like selection scalars: floats, strings, and bools
    cannot silently become valid counts in an over-cap attestation.

    The returned shape is deliberately small and JSON-native so a torch-free
    compiler/promotion consumer can independently recompute the cardinality.
    No complete ordered action is constructed here.
    """

    if type(max_combos) is not int or max_combos < 0:
        raise FeatureContractError("max_combos must be a nonnegative exact integer")

    if isinstance(obs, Mapping):
        select = obs.get("select")
        if not isinstance(select, Mapping):
            raise FeatureContractError("observation select must be an object")
        options = select.get("option")
        raw_min = select.get("minCount")
        raw_max = select.get("maxCount")
    else:
        select = getattr(obs, "select", None)
        options = getattr(select, "option", None)
        raw_min = getattr(select, "minCount", None)
        raw_max = getattr(select, "maxCount", None)

    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise FeatureContractError("observation select.option must be a sequence")
    if type(raw_min) is not int or type(raw_max) is not int:
        raise FeatureContractError(
            "selection minCount/maxCount must be exact integers, not coercible values"
        )
    n_options = len(options)
    if not (0 <= raw_min <= raw_max <= n_options):
        raise FeatureContractError(
            "selection minCount/maxCount are outside the exact option bounds"
        )
    counts = list(range(raw_min, raw_max + 1))
    cardinality = sum(math.perm(n_options, count) for count in counts)
    return {
        "n_options": n_options,
        "min_count": raw_min,
        "max_count": raw_max,
        "counts": counts,
        "complete_ordered_action_cardinality": cardinality,
        "complete_ordered_action_cap": max_combos,
        "over_cap": cardinality > max_combos,
        # These attest implementation behavior, rather than substituting for
        # the independently recomputed cardinality above.  They are useful to
        # a sealed evaluator because truncated enumeration would otherwise
        # resemble a successful factorized selection.
        "complete_ordered_actions_materialized": False,
        "complete_ordered_action_truncated": False,
    }


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


def _decoder_legacy_vocab_size() -> int:
    """End of the v4 decoder layout (kept stable for append-only migration)."""
    cc = card_vocab_size()
    ac = attack_vocab_size()
    card_offset = DECODER_ATTACK_OFFSET + ac
    n_contexts = int(cg_env.SelectContext.RECOVER_SPECIAL_CONDITION) + 1
    return card_offset + (1 + DECODER_MAIN_FEATURE + n_contexts) * cc


def decoder_binding_offset() -> int:
    """First decoder feature reserved for explicit option bindings."""
    return _decoder_legacy_vocab_size()


def decoder_vocab_size() -> int:
    """EmbeddingBag vocab needed by :func:`build_option_tokens`.

    14 typed flags + attack ids + (1 + 8 main features + 49 SelectContexts)
    card-id blocks, followed by compact role-specific owner/area/index binding
    features. SelectContext currently tops out at 48
    (RECOVER_SPECIAL_CONDITION), so 49 contexts are reserved.
    """
    return decoder_binding_offset() + DECODER_BINDING_VOCAB_SIZE


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
        index = _exact_int(index, field="sparse feature index")
        if index < 0:
            raise FeatureContractError(f"negative sparse feature index: {index}")
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int) -> None:
        pos = _exact_int(pos, field="sparse feature width")
        if pos < 0:
            raise FeatureContractError(f"negative sparse feature width: {pos}")
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

def _validated_card_id(
    card_id,
    card_count: int,
    *,
    field: str,
    allow_zero: bool = False,
) -> int:
    value = _exact_int(card_id, field=field)
    lower = 0 if allow_zero else 1
    if not lower <= value < int(card_count):
        raise FeatureContractError(
            f"{field} {value} is outside {lower}..{int(card_count) - 1}"
        )
    return value


def _validated_attack_id(attack_id) -> int:
    value = _exact_int(attack_id, field="option attackId")
    count = attack_vocab_size()
    if not 1 <= value < count:
        raise FeatureContractError(
            f"option attackId {value} is outside 1..{count - 1}"
        )
    return value


def _validated_context(context) -> int:
    value = _enum_int(context, cg_env.SelectContext, field="select context")
    maximum = int(cg_env.SelectContext.RECOVER_SPECIAL_CONDITION)
    if not 0 <= value <= maximum:
        raise FeatureContractError(
            f"select context {value} is outside the feature schema 0..{maximum}"
        )
    return value

def _add_card(sv: SparseVector, card, card_count: int) -> None:
    if card is not None:
        sv.add(
            _validated_card_id(
                card.id,
                card_count,
                field="board card id",
            ),
            1,
        )
    sv.add_pos(card_count)


def _add_cards(sv: SparseVector, cards, value: float, card_count: int) -> None:
    if cards is not None:
        for card in cards:
            sv.add(
                _validated_card_id(
                    card.id,
                    card_count,
                    field="board card id",
                ),
                value,
            )
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
        sv.add(
            _validated_card_id(
                cid,
                cc,
                field="own deck card id",
            ),
            0.25,
        )
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
    player_index = _exact_int(player_index, field="option playerIndex")
    if player_index not in (0, 1):
        raise FeatureContractError(f"invalid option playerIndex: {player_index}")
    index = _exact_int(index, field="option index")
    if index < 0:
        raise FeatureContractError(f"negative option index: {index}")
    area_code = _enum_int(area, AreaType, field="option area")
    if not 1 <= area_code < DECODER_BINDING_AREA_COUNT:
        raise FeatureContractError(
            f"option area is outside the feature schema: {area_code}"
        )
    ps = obs.current.players[player_index]
    if area_code == int(AreaType.DECK):
        return obs.select.deck[index]
    if area_code == int(AreaType.HAND):
        return ps.hand[index]
    if area_code == int(AreaType.DISCARD):
        return ps.discard[index]
    if area_code == int(AreaType.ACTIVE):
        return ps.active[index]
    if area_code == int(AreaType.BENCH):
        return ps.bench[index]
    if area_code == int(AreaType.PRIZE):
        return ps.prize[index]
    if area_code == int(AreaType.STADIUM):
        return obs.current.stadium[index]
    if area_code == int(AreaType.LOOKING):
        return obs.current.looking[index]
    return None


def _decoder_card_offset() -> int:
    return DECODER_ATTACK_OFFSET + attack_vocab_size()


def _decoder_main(
    sv: SparseVector, feature_index: int, card, cc: int, weight: float = 1.0
) -> None:
    feature_index = _exact_int(feature_index, field="decoder main feature")
    if not 0 <= feature_index < DECODER_MAIN_FEATURE:
        raise FeatureContractError(
            f"invalid decoder main feature index: {feature_index}"
        )
    if card is not None:
        card_id = _validated_card_id(
            card.id,
            cc,
            field="option card id",
        )
        sv.add(_decoder_card_offset() + feature_index * cc + card_id, weight)


def _decoder_card_id(
    sv: SparseVector,
    context,
    card_id: int,
    cc: int,
    weight: float = 1.0,
    *,
    allow_zero: bool = False,
) -> None:
    context = _validated_context(context)
    card_id = _validated_card_id(
        card_id,
        cc,
        field="option card id",
        allow_zero=allow_zero,
    )
    sv.add(
        _decoder_card_offset() + (DECODER_MAIN_FEATURE + context) * cc + card_id,
        weight,
    )


def _decoder_card(
    sv: SparseVector, context, card, cc: int, weight: float = 1.0
) -> None:
    if card is not None:
        _decoder_card_id(sv, context, card.id, cc, weight)


def _decoder_binding(
    sv: SparseVector,
    role: int,
    *,
    player_index: Optional[int],
    area,
    index: Optional[int],
    your_index: int,
    weight: float = 1.0,
) -> None:
    """Add an exact, role-specific (owner, area, index) option binding.

    The engine has two seats, AreaType values 1..12, and at most 60 physical
    cards.  Invalid values fail closed instead of folding into an overflow
    bucket, because folding would make two legal bindings indistinguishable.
    """
    role = int(role)
    if not 0 <= role < DECODER_BINDING_ROLE_COUNT:
        raise ValueError(f"invalid decoder binding role: {role}")

    if player_index is None:
        owner_code = 2
    elif int(player_index) == int(your_index):
        owner_code = 0
    elif int(player_index) == 1 - int(your_index):
        owner_code = 1
    else:
        raise ValueError(f"invalid option playerIndex: {player_index}")

    area_code = (
        0
        if area is None
        else _enum_int(area, cg_env.AreaType, field="option area")
    )
    if not 0 <= area_code < DECODER_BINDING_AREA_COUNT:
        raise ValueError(f"option area is outside the feature schema: {area_code}")

    if index is None:
        index_code = 0
    else:
        raw_index = int(index)
        if not 0 <= raw_index < DECODER_BINDING_INDEX_COUNT - 1:
            raise ValueError(
                "option index is outside the exact feature schema: "
                f"{raw_index} (expected 0..{DECODER_BINDING_INDEX_COUNT - 2})"
            )
        index_code = raw_index + 1

    tuple_index = (
        (
            role * DECODER_BINDING_OWNER_COUNT
            + owner_code
        )
        * DECODER_BINDING_AREA_COUNT
        + area_code
    )
    tuple_index = tuple_index * DECODER_BINDING_INDEX_COUNT + index_code
    sv.add(decoder_binding_offset() + tuple_index, weight)


def _decoder_binding_if_present(
    sv: SparseVector,
    role: int,
    option,
    *,
    your_index: int,
    weight: float,
) -> None:
    """Encode optional source fields used by SKILL and future option types."""
    area = getattr(option, "area", None)
    index = getattr(option, "index", None)
    player_index = getattr(option, "playerIndex", None)
    if area is None and index is None and player_index is None:
        return
    _decoder_binding(
        sv,
        role,
        player_index=player_index,
        area=area,
        index=index,
        your_index=your_index,
        weight=weight,
    )


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
    _validated_context(context)

    sv = SparseVector()
    for action in action_combos:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1)
            continue
        action_indices = [
            _exact_int(index, field="action option index") for index in action
        ]
        if len(action_indices) != len(set(action_indices)):
            raise FeatureContractError(
                f"action repeats an option index: {action_indices}"
            )
        if any(
            index < 0 or index >= len(obs.select.option)
            for index in action_indices
        ):
            raise FeatureContractError(
                f"action option index is outside 0..{len(obs.select.option) - 1}: "
                f"{action_indices}"
            )
        for rank, i in enumerate(action_indices):
            o = obs.select.option[i]
            t = _enum_int(o.type, OptionType, field="option type")
            # A weighted positional code keeps ordered selections distinct
            # without changing checkpoint embedding-table dimensions.
            weight = float(rank + 1)
            if t == int(OptionType.END):
                sv.add(1, weight)
            elif t == int(OptionType.YES):
                sv.add(2, weight)
            elif t == int(OptionType.NO):
                sv.add(3, weight)
            elif t == int(OptionType.SPECIAL_CONDITION):
                condition = _enum_int(
                    o.specialConditionType,
                    cg_env.SpecialConditionType,
                    field="option specialConditionType",
                )
                if not 0 <= condition < 5:
                    raise FeatureContractError(
                        "option specialConditionType is outside the feature "
                        f"schema: {condition}"
                    )
                sv.add(4 + condition, weight)
            elif t == int(OptionType.NUMBER):
                number = _exact_int(o.number, field="option number")
                if number < 0:
                    raise FeatureContractError(f"negative option number: {number}")
                sv.add(9 + min(number, 4), weight)
            elif t == int(OptionType.ATTACK):
                sv.add(
                    DECODER_ATTACK_OFFSET + _validated_attack_id(o.attackId),
                    weight,
                )
            elif t == int(OptionType.PLAY):
                _decoder_main(sv, 0, ps.hand[o.index], cc, weight)
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=your_index,
                    area=cg_env.AreaType.HAND,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.ATTACH):
                _decoder_main(
                    sv,
                    1,
                    get_card(obs, o.area, o.index, your_index),
                    cc,
                    weight,
                )
                _decoder_main(
                    sv,
                    2,
                    get_card(obs, o.inPlayArea, o.inPlayIndex, your_index),
                    cc,
                    weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=your_index,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_TARGET,
                    player_index=your_index,
                    area=o.inPlayArea,
                    index=o.inPlayIndex,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.EVOLVE):
                _decoder_main(
                    sv,
                    3,
                    get_card(obs, o.area, o.index, your_index),
                    cc,
                    weight,
                )
                _decoder_main(
                    sv,
                    4,
                    get_card(obs, o.inPlayArea, o.inPlayIndex, your_index),
                    cc,
                    weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=your_index,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_TARGET,
                    player_index=your_index,
                    area=o.inPlayArea,
                    index=o.inPlayIndex,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.ABILITY):
                _decoder_main(
                    sv,
                    5,
                    get_card(obs, o.area, o.index, your_index),
                    cc,
                    weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=your_index,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.DISCARD):
                _decoder_main(
                    sv,
                    6,
                    get_card(obs, o.area, o.index, your_index),
                    cc,
                    weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=your_index,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.RETREAT):
                _decoder_main(sv, 7, ps.active[0], cc, weight)
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=your_index,
                    area=cg_env.AreaType.ACTIVE,
                    index=0,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.CARD):
                _decoder_card(
                    sv,
                    context,
                    get_card(obs, o.area, o.index, o.playerIndex),
                    cc,
                    weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=o.playerIndex,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.TOOL_CARD):
                card = get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.tools[o.toolIndex], cc, weight)
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=o.playerIndex,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_TOOL,
                    player_index=o.playerIndex,
                    area=cg_env.AreaType.TOOL,
                    index=o.toolIndex,
                    your_index=your_index,
                    weight=weight,
                )
            elif t in (int(OptionType.ENERGY_CARD), int(OptionType.ENERGY)):
                card = get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.energyCards[o.energyIndex], cc, weight)
                _decoder_binding(
                    sv,
                    DECODER_BINDING_SOURCE,
                    player_index=o.playerIndex,
                    area=o.area,
                    index=o.index,
                    your_index=your_index,
                    weight=weight,
                )
                _decoder_binding(
                    sv,
                    DECODER_BINDING_ENERGY,
                    player_index=o.playerIndex,
                    area=cg_env.AreaType.ENERGY,
                    index=o.energyIndex,
                    your_index=your_index,
                    weight=weight,
                )
            elif t == int(OptionType.SKILL):
                # Official API reserves cardId=0 for special-condition skills.
                _decoder_card_id(
                    sv,
                    context,
                    o.cardId,
                    cc,
                    weight,
                    allow_zero=True,
                )
                _decoder_binding_if_present(
                    sv,
                    DECODER_BINDING_SOURCE,
                    o,
                    your_index=your_index,
                    weight=weight,
                )
            else:
                raise FeatureContractError(
                    f"unsupported option type {o.type!r}; "
                    "feature schema must be updated"
                )
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
