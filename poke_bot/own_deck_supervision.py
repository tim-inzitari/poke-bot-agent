"""Dormant causal supervision for own-deck tutor and closeout research.

This module deliberately has no model, corpus, trainer, feature, simulator, or
runtime imports.  It is a post-Alakazam-refresh successor primitive only:
callers may materialize its returned *targets*, but must never put
``transition_after`` or continuation observations into serving features.

The two target families are intentionally factual rather than counterfactual:

* visible tutor labels describe an actually selected card from an actually
  exposed ``select.deck`` list and, when provable, its observed same-actor
  follow-through;
* terminal-conversion labels describe only the immediate transition of the
  selected complete action.

No function infers whether an unselected legal action would have won, whether a
filtered tutor list represents the complete deck, or what a hidden prize/deck
card might be.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

OWN_DECK_SUPERVISION_SCHEMA = "poke_bot.own_deck_supervision/v1"
OWN_DECK_SUPERVISION_VERSION = 1

# Public competition wire representation.  Textual enum renderings are also
# accepted below so archived JSON need not import a live ``cg`` runtime.
DECK_AREA_WIRE_VALUE = 1

TERMINAL_CONVERSION_CLASSES: tuple[str, ...] = (
    "nonterminal",
    "own_win",
    "own_loss",
    "draw",
)
TERMINAL_CONVERSION_CLASS_COUNT = len(TERMINAL_CONVERSION_CLASSES)
TERMINAL_CONVERSION_SCALAR_TARGET_NAMES: tuple[str, ...] = (
    "prize_closeout",
    "opponent_knockout",
)
TERMINAL_CONVERSION_CLASS_SLICE = slice(0, TERMINAL_CONVERSION_CLASS_COUNT)
TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX = TERMINAL_CONVERSION_CLASS_SLICE.stop
TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX = (
    TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX + 1
)
TERMINAL_CONVERSION_OUTPUT_LAYOUT: tuple[str, ...] = (
    "terminal_class.nonterminal",
    "terminal_class.own_win",
    "terminal_class.own_loss",
    "terminal_class.draw",
    "prize_closeout",
    "opponent_knockout",
)
TERMINAL_CONVERSION_OUTPUT_DIM = 6
VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES: tuple[str, ...] = (
    "selected_from_visible_deck",
    "selected_target_observed_after_action",
    "same_actor_followup",
)
VISIBLE_TUTOR_COMPLETION_SCALAR_SLICE = slice(
    0, len(VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES)
)
VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX = 0
VISIBLE_TUTOR_COMPLETION_SELECTED_TARGET_OBSERVED_AFTER_ACTION_INDEX = 1
VISIBLE_TUTOR_COMPLETION_SAME_ACTOR_FOLLOWUP_INDEX = 2
VISIBLE_TUTOR_COMPLETION_TERMINAL_CLASS_SLICE = slice(
    VISIBLE_TUTOR_COMPLETION_SCALAR_SLICE.stop,
    VISIBLE_TUTOR_COMPLETION_SCALAR_SLICE.stop + TERMINAL_CONVERSION_CLASS_COUNT,
)
VISIBLE_TUTOR_COMPLETION_OUTPUT_LAYOUT: tuple[str, ...] = (
    "selected_from_visible_deck",
    "selected_target_observed_after_action",
    "same_actor_followup",
    "same_actor_terminal_class.nonterminal",
    "same_actor_terminal_class.own_win",
    "same_actor_terminal_class.own_loss",
    "same_actor_terminal_class.draw",
)
VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM = 7
# Tutor selection is deliberately a dynamic candidate target: one masked
# binary label per card in the observed select.deck menu.  It is not a global
# card-ID classifier, because its legal output width is the visible menu size.
VISIBLE_TUTOR_SELECTION_TARGET_KIND = "visible_deck_menu_candidate"
VISIBLE_TUTOR_SELECTION_AUXILIARY_OUTPUT_DIM = 0
TerminalConversionClass = Literal[
    "nonterminal", "own_win", "own_loss", "draw"
]
_TERMINAL_CLASS_INDEX = {
    name: index for index, name in enumerate(TERMINAL_CONVERSION_CLASSES)
}


class ScalarTarget(TypedDict):
    """A tensor-ready scalar target whose zero is inert when masked."""

    value: float
    mask: bool


class CategoricalTarget(TypedDict):
    """A tensor-ready categorical target whose zero is inert when masked."""

    value: int
    mask: bool


class TerminalConversionLabels(TypedDict):
    """Immediate, selected-complete-action labels only."""

    terminal_class: CategoricalTarget
    prize_closeout: ScalarTarget
    opponent_knockout: ScalarTarget
    target_only: bool
    provenance: str


class VisibleTutorCompletionLabels(TypedDict):
    """Visible deck-selection labels with no hidden-deck inference."""

    selected_card_id: CategoricalTarget
    selected_card_ids: list[int]
    selected_card_serials: list[int]
    selected_from_visible_deck: ScalarTarget
    selected_target_observed_after_action: ScalarTarget
    same_actor_followup: ScalarTarget
    same_actor_terminal_class: CategoricalTarget
    target_only: bool
    provenance: str
    mask_reason: str | None


class OwnDeckSupervisionLabels(TypedDict):
    """One dormant target record aligned to one complete decision step."""

    schema: str
    version: int
    terminal_conversion: TerminalConversionLabels
    visible_tutor_completion: VisibleTutorCompletionLabels


TargetVector = tuple[float, ...]
TargetMask = tuple[bool, ...]


def _masked_scalar() -> ScalarTarget:
    return {"value": 0.0, "mask": False}


def _masked_category() -> CategoricalTarget:
    return {"value": 0, "mask": False}


def _scalar(value: bool | float) -> ScalarTarget:
    return {"value": float(value), "mask": True}


def _category(name: TerminalConversionClass) -> CategoricalTarget:
    return {"value": _TERMINAL_CLASS_INDEX[name], "mask": True}


def _masked_scalar_component(value: Any) -> tuple[float, bool]:
    """Return a finite scalar only when its target mask is exactly true."""

    row = _mapping(value)
    if row is None or row.get("mask") is not True:
        return 0.0, False
    raw = row.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0, False
    parsed = float(raw)
    return (parsed, True) if math.isfinite(parsed) else (0.0, False)


def _masked_one_hot_component(
    value: Any,
    *,
    width: int,
) -> tuple[TargetVector, TargetMask]:
    """One-hot an exact masked category; malformed categories fail closed."""

    row = _mapping(value)
    index = None if row is None or row.get("mask") is not True else _exact_int(row.get("value"))
    if index is None or index < 0 or index >= width:
        return (0.0,) * width, (False,) * width
    target = [0.0] * width
    target[index] = 1.0
    return tuple(target), (True,) * width


def _terminal_conversion_vector_and_mask(
    labels: Mapping[str, Any],
) -> tuple[TargetVector, TargetMask]:
    class_vector, class_mask = _masked_one_hot_component(
        labels.get("terminal_class"),
        width=TERMINAL_CONVERSION_CLASS_COUNT,
    )
    prize_value, prize_mask = _masked_scalar_component(
        labels.get("prize_closeout")
    )
    knockout_value, knockout_mask = _masked_scalar_component(
        labels.get("opponent_knockout")
    )
    target = (
        *class_vector,
        prize_value,
        knockout_value,
    )
    mask = (
        *class_mask,
        prize_mask,
        knockout_mask,
    )
    assert len(target) == TERMINAL_CONVERSION_OUTPUT_DIM
    assert len(mask) == TERMINAL_CONVERSION_OUTPUT_DIM
    return target, mask


def terminal_conversion_target_vector(
    labels: TerminalConversionLabels | Mapping[str, Any],
) -> TargetVector:
    """Return the fixed six-wide target vector for the terminal head.

    Layout is :data:`TERMINAL_CONVERSION_OUTPUT_LAYOUT`.  The first four
    positions are a one-hot categorical target for a cross-entropy-style loss;
    the final two are independently masked binary/scalar targets.  This helper
    intentionally emits no observation or transition data.
    """

    row = _mapping(labels)
    if row is None:
        return (0.0,) * TERMINAL_CONVERSION_OUTPUT_DIM
    return _terminal_conversion_vector_and_mask(row)[0]


def terminal_conversion_target_mask(
    labels: TerminalConversionLabels | Mapping[str, Any],
) -> TargetMask:
    """Return the fixed six-wide mask aligned to the terminal target vector."""

    row = _mapping(labels)
    if row is None:
        return (False,) * TERMINAL_CONVERSION_OUTPUT_DIM
    return _terminal_conversion_vector_and_mask(row)[1]


def _visible_tutor_completion_vector_and_mask(
    labels: Mapping[str, Any],
) -> tuple[TargetVector, TargetMask]:
    scalar_values: list[float] = []
    scalar_masks: list[bool] = []
    for name in VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES:
        value, mask = _masked_scalar_component(labels.get(name))
        scalar_values.append(value)
        scalar_masks.append(mask)
    class_vector, class_mask = _masked_one_hot_component(
        labels.get("same_actor_terminal_class"),
        width=TERMINAL_CONVERSION_CLASS_COUNT,
    )
    target = (*scalar_values, *class_vector)
    mask = (*scalar_masks, *class_mask)
    assert len(target) == VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
    assert len(mask) == VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
    return target, mask


def visible_tutor_completion_target_vector(
    labels: VisibleTutorCompletionLabels | Mapping[str, Any],
) -> TargetVector:
    """Return the fixed seven-wide tutor-completion auxiliary target.

    Tutor *selection* is deliberately absent: it is trained against the
    current legal ``select.deck`` candidates through the policy/option path,
    whose width is the actually observed menu size.  The first three positions
    are completion scalars and the final four are a one-hot same-actor terminal
    class, according to :data:`VISIBLE_TUTOR_COMPLETION_OUTPUT_LAYOUT`.
    """

    row = _mapping(labels)
    if row is None:
        return (0.0,) * VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
    return _visible_tutor_completion_vector_and_mask(row)[0]


def visible_tutor_completion_target_mask(
    labels: VisibleTutorCompletionLabels | Mapping[str, Any],
) -> TargetMask:
    """Return the fixed seven-wide mask aligned to tutor-completion targets."""

    row = _mapping(labels)
    if row is None:
        return (False,) * VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
    return _visible_tutor_completion_vector_and_mask(row)[1]


def _exact_int(value: Any) -> int | None:
    """Return only non-bool exact integer-like JSON values."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _zone(value: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    rows = list(value)
    if not all(isinstance(row, Mapping) for row in rows):
        return None
    return rows


def _normalized_name(value: Any) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _is_deck_area(value: Any) -> bool:
    numeric = _exact_int(value)
    if numeric is not None:
        return numeric == DECK_AREA_WIRE_VALUE
    name = _normalized_name(value)
    return name in {"deck", "areatypedeck"}


def _card_id(card: Any) -> int | None:
    row = _mapping(card)
    return None if row is None else _exact_int(row.get("id"))


def _card_serial(card: Any) -> int | None:
    row = _mapping(card)
    return None if row is None else _exact_int(row.get("serial"))


_VISIBLE_CARD_CHILDREN: tuple[tuple[str, str | None], ...] = (
    ("tools", "tool_serials"),
    ("toolCards", "tool_serials"),
    ("energyCards", "energy_serials"),
    ("preEvolution", "pre_evolution_serials"),
    ("preEvolutions", "pre_evolution_serials"),
)
_EVOLUTION_STACK_CHILDREN: tuple[str, ...] = (
    "preEvolution",
    "preEvolutions",
)


def _nested_card_rows(value: Any) -> list[Mapping[str, Any]] | None:
    """Accept a singleton nested card or an array of nested cards."""

    if value is None:
        return []
    row = _mapping(value)
    if row is not None:
        return [row]
    return _zone(value)


def _append_packed_serials(value: Any, *, serials: set[int]) -> bool:
    """Append a normalized serial list without treating malformed data as fact."""

    if not isinstance(value, (list, tuple)):
        return False
    for raw in value:
        serial = _exact_int(raw)
        if serial is None or serial in serials:
            return False
        serials.add(serial)
    return True


def _append_card_serials(
    card: Mapping[str, Any],
    *,
    serials: set[int],
    children: Sequence[tuple[str, str | None]],
) -> bool:
    """Recursively collect physical serials from public card attachments."""

    serial = _card_serial(card)
    if serial is None or serial in serials:
        return False
    serials.add(serial)
    for name, packed_name in children:
        raw = card.get(name)
        if raw is not None:
            rows = _nested_card_rows(raw)
            if rows is None:
                return False
            for child in rows:
                if not _append_card_serials(
                    child,
                    serials=serials,
                    children=children,
                ):
                    return False
        elif packed_name is not None and packed_name in card:
            if not _append_packed_serials(card.get(packed_name), serials=serials):
                return False
    return True


def _serials_from_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    children: Sequence[tuple[str, str | None]],
) -> set[int] | None:
    serials: set[int] = set()
    for card in cards:
        if not _append_card_serials(card, serials=serials, children=children):
            return None
    return serials


def _evolution_stack_serials(card: Mapping[str, Any], *, serials: set[int]) -> bool:
    """Collect only Pokémon roots and their recursive pre-evolution stack."""

    return _append_card_serials(
        card,
        serials=serials,
        children=tuple((name, None) for name in _EVOLUTION_STACK_CHILDREN),
    )


def _unwrap_transition(value: Any) -> Mapping[str, Any] | None:
    """Accept a raw observation, a wrapped observation, or a public snapshot."""

    row = _mapping(value)
    if row is None:
        return None
    nested = _mapping(row.get("observation"))
    if nested is not None:
        return nested
    return row


def _current(value: Any) -> Mapping[str, Any] | None:
    row = _unwrap_transition(value)
    return None if row is None else _mapping(row.get("current"))


def _raw_players(value: Any) -> list[Mapping[str, Any]] | None:
    row = _unwrap_transition(value)
    if row is None:
        return None
    raw = row.get("players")
    if raw is None:
        current = _mapping(row.get("current"))
        raw = None if current is None else current.get("players")
    rows = _zone(raw)
    return rows if rows is not None and len(rows) == 2 else None


def _actor_from_step(step: Mapping[str, Any]) -> int | None:
    observation = _mapping(step.get("observation"))
    current = None if observation is None else _mapping(observation.get("current"))
    actor = None if current is None else _exact_int(current.get("yourIndex"))
    if actor is None:
        actor = _exact_int(step.get("actor_seat"))
    return actor if actor in (0, 1) else None


def _turn(value: Any) -> int | None:
    row = _unwrap_transition(value)
    current = None if row is None else _mapping(row.get("current"))
    raw = row.get("turn") if current is None and row is not None else (
        None if current is None else current.get("turn")
    )
    return _exact_int(raw)


def _validated_status_next_actor(row: Mapping[str, Any]) -> int | None:
    """Read an explicitly validated active-status provenance, if supplied."""

    candidates: list[Mapping[str, Any]] = []
    if row.get("active_status_validated") is True:
        candidates.append(row)
    for key in ("active_status", "active_status_provenance"):
        proof = _mapping(row.get(key))
        if proof is not None and proof.get("validated") is True:
            candidates.append(proof)
    for proof in candidates:
        for field in (
            "next_actor_seat",
            "active_actor_seat",
            "active_seat",
            "actor_seat",
        ):
            if field not in proof:
                continue
            actor = _exact_int(proof.get(field))
            return actor if actor in (0, 1) else None
    return None


def _next_actor(value: Any) -> int | None:
    """Return only independently attested next-actor provenance.

    ``current.yourIndex`` is the replay observer's perspective seat, not a
    reliable turn-owner signal.  Inspect an outer transition wrapper before an
    embedded observation so that a collection-side ``next_actor_seat`` proof
    survives unwrapping.  With no explicit or validated provenance, callers
    must mask same-actor continuation labels.
    """

    outer = _mapping(value)
    if outer is None:
        return None
    rows = [outer]
    nested = _mapping(outer.get("observation"))
    if nested is not None:
        rows.append(nested)
    for row in rows:
        if "next_actor_seat" in row:
            actor = _exact_int(row.get("next_actor_seat"))
            return actor if actor in (0, 1) else None
        actor = _validated_status_next_actor(row)
        if actor is not None:
            return actor
    return None


def _transition_result(value: Any) -> int | None:
    row = _unwrap_transition(value)
    if row is None or row.get("valid") is False:
        return None
    current = _mapping(row.get("current"))
    raw = row.get("result") if current is None else current.get("result")
    result = _exact_int(raw)
    return result if result in (-1, 0, 1, 2) else None


def _terminal_class(value: Any, *, actor: int) -> TerminalConversionClass | None:
    result = _transition_result(value)
    if result is None:
        return None
    if result == -1:
        return "nonterminal"
    if result == actor:
        return "own_win"
    if result == 1 - actor:
        return "own_loss"
    return "draw"


def _prize_count(value: Any, *, seat: int) -> int | None:
    players = _raw_players(value)
    if players is None or seat not in (0, 1):
        return None
    player = players[seat]
    for field in ("prize_count", "prizeCount", "remainingPrizes"):
        count = _exact_int(player.get(field))
        if count is not None and count >= 0:
            return count
    prize = player.get("prize")
    if isinstance(prize, (list, tuple)):
        return len(prize)
    return None


def _board_serials(value: Any, *, seat: int) -> set[int] | None:
    """Return public Pokémon roots plus recursive pre-evolution stacks.

    A root-card-only comparison mislabels an evolution as a KO when the old
    physical card becomes a nested ``preEvolution``.  Attachments are excluded
    here because their ordinary discard is not Pokémon removal evidence.
    """

    players = _raw_players(value)
    if players is None or seat not in (0, 1):
        return None
    player = players[seat]
    active = _zone(player.get("active"))
    bench = _zone(player.get("bench"))
    if active is None or bench is None:
        return None
    serials: set[int] = set()
    for card in [*active, *bench]:
        if not _evolution_stack_serials(card, serials=serials):
            return None
    return serials


def _discard_serials(value: Any, *, seat: int) -> set[int] | None:
    players = _raw_players(value)
    if players is None or seat not in (0, 1):
        return None
    player = players[seat]
    # Normalized public snapshots use discard_serials; raw observations use a
    # card zone.  Both are target-only evidence.
    packed = player.get("discard_serials")
    if isinstance(packed, list):
        result: set[int] = set()
        return result if _append_packed_serials(packed, serials=result) else None
    discard = _zone(player.get("discard"))
    if discard is None:
        return None
    return _serials_from_cards(discard, children=_VISIBLE_CARD_CHILDREN)


def _public_prize_knockout_evidence(
    before: Any,
    after: Any,
    *,
    actor: int,
) -> bool:
    """Use an observed own-prize decrease as public KO evidence only."""

    before_prizes = _prize_count(before, seat=actor)
    after_prizes = _prize_count(after, seat=actor)
    return (
        before_prizes is not None
        and after_prizes is not None
        and after_prizes < before_prizes
    )


def _opponent_knockout(
    before: Any,
    after: Any,
    *,
    actor: int,
) -> ScalarTarget:
    """Prove a KO from stack-aware removal plus observed prize evidence."""

    victim = 1 - actor
    before_board = _board_serials(before, seat=victim)
    after_board = _board_serials(after, seat=victim)
    discarded = _discard_serials(after, seat=victim)
    if before_board is None or after_board is None or discarded is None:
        return _masked_scalar()
    disappeared = before_board - after_board
    if not disappeared:
        return _scalar(False)
    if not disappeared.issubset(discarded):
        return _masked_scalar()
    if not _public_prize_knockout_evidence(before, after, actor=actor):
        return _masked_scalar()
    return _scalar(True)


def _is_final_selected_action(
    step: Mapping[str, Any],
    explicit: bool | None,
) -> bool:
    """Recognize known prefixes while treating ordinary trajectory rows as full actions."""

    if explicit is not None:
        return bool(explicit)
    for key in ("is_prefix", "prefix_only"):
        if step.get(key) is True:
            return False
    for key in ("is_final_selected_action", "is_final_action_stage", "stage_is_final"):
        value = step.get(key)
        if isinstance(value, bool):
            return value
    for index_key, count_key in (
        ("raw_stage_index", "raw_stage_count"),
        ("stage_index", "stage_count"),
    ):
        index = _exact_int(step.get(index_key))
        count = _exact_int(step.get(count_key))
        if index is not None or count is not None:
            return bool(index is not None and count is not None and count > 0 and index == count - 1)
    # A DecisionSample-style trajectory row stores the completed ordered action
    # once.  There is no prefix object unless one of the explicit fields above
    # says otherwise.
    return True


def _immediate_transition_allowed(step: Mapping[str, Any]) -> bool:
    """Reject boundary evidence on the step, wrapper, or unwrapped post-state."""

    if step.get("transition_after_immediate") is False:
        return False
    transition = step.get("transition_after")
    post = _unwrap_transition(transition)
    return not (
        _has_explicit_boundary(step)
        or _has_explicit_boundary(transition)
        or _has_explicit_boundary(post)
    )


def _has_explicit_boundary(value: Any) -> bool:
    """Honor explicit provenance markers without guessing a hidden boundary."""

    row = _mapping(value)
    if row is None:
        return False
    truthy_keys = (
        "chance",
        "is_chance",
        "chance_boundary",
        "stochastic",
        "unresolved_randomness",
        "opponent_intervened",
        "actor_boundary",
        "turn_boundary",
    )
    if any(row.get(key) is True for key in truthy_keys):
        return True
    boundary = row.get("boundary")
    if isinstance(boundary, str):
        name = _normalized_name(boundary)
        if any(token in name for token in ("chance", "stochastic", "opponent", "actor", "turn")):
            return True
    diagnostics = _mapping(row.get("diagnostics"))
    return diagnostics is not None and _has_explicit_boundary(diagnostics)


def terminal_conversion_labels(
    step: Mapping[str, Any],
    *,
    is_final_selected_action: bool | None = None,
) -> TerminalConversionLabels:
    """Build immediate factual terminal labels for one selected complete action.

    ``transition_after`` is read strictly as target construction evidence.  It
    is neither returned nor suitable as a model feature.  Prefix rows,
    incomplete transitions, and explicitly non-immediate/boundary records are
    fully masked.
    """

    empty: TerminalConversionLabels = {
        "terminal_class": _masked_category(),
        "prize_closeout": _masked_scalar(),
        "opponent_knockout": _masked_scalar(),
        "target_only": True,
        "provenance": "masked",
    }
    if not _is_final_selected_action(step, is_final_selected_action):
        empty["provenance"] = "masked_prefix"
        return empty
    if not _immediate_transition_allowed(step):
        empty["provenance"] = "masked_non_immediate_or_boundary"
        return empty
    actor = _actor_from_step(step)
    after = _unwrap_transition(step.get("transition_after"))
    if actor is None or after is None:
        empty["provenance"] = "masked_missing_actor_or_transition"
        return empty
    outcome = _terminal_class(after, actor=actor)
    if outcome is None:
        empty["provenance"] = "masked_unproven_terminal_result"
        return empty

    result: TerminalConversionLabels = {
        "terminal_class": _category(outcome),
        "prize_closeout": _masked_scalar(),
        "opponent_knockout": _masked_scalar(),
        "target_only": True,
        "provenance": "immediate_selected_public_transition",
    }
    before = _mapping(step.get("observation"))
    before_prizes = _prize_count(before, seat=actor)
    after_prizes = _prize_count(after, seat=actor)
    if before_prizes is not None and after_prizes is not None:
        result["prize_closeout"] = _scalar(
            before_prizes > 0 and after_prizes == 0
        )
    result["opponent_knockout"] = _opponent_knockout(
        before, after, actor=actor
    )
    return result


def _selected_option_indices(step: Mapping[str, Any]) -> list[int] | None:
    raw = step.get("action")
    if not isinstance(raw, (list, tuple)):
        raw = step.get("selected_action")
    if not isinstance(raw, (list, tuple)):
        return None
    values: list[int] = []
    for item in raw:
        index = _exact_int(item)
        if index is None or index < 0:
            return None
        values.append(index)
    return values


def _visible_tutor_targets(
    step: Mapping[str, Any],
    *,
    actor: int,
) -> tuple[list[Mapping[str, Any]], str | None]:
    """Resolve selected cards only from the currently visible own deck list."""

    observation = _mapping(step.get("observation"))
    select = None if observation is None else _mapping(observation.get("select"))
    if select is None:
        return [], "missing_select"
    deck = _zone(select.get("deck"))
    options = _zone(select.get("option"))
    indices = _selected_option_indices(step)
    if deck is None:
        return [], "deck_not_visible"
    if options is None or indices is None:
        return [], "missing_options_or_selected_action"

    selected: list[Mapping[str, Any]] = []
    for option_index in indices:
        if option_index >= len(options):
            return [], "selected_option_out_of_range"
        option = options[option_index]
        if not _is_deck_area(option.get("area")):
            continue
        owner = _exact_int(option.get("playerIndex"))
        if owner is not None and owner != actor:
            return [], "non_actor_deck_reference"
        deck_index = _exact_int(option.get("index"))
        if deck_index is None or deck_index < 0 or deck_index >= len(deck):
            return [], "visible_deck_index_out_of_range"
        selected.append(deck[deck_index])
    if not selected:
        return [], "no_selected_visible_deck_card"
    return selected, None


def _actor_visible_serials(value: Any, *, actor: int) -> set[int] | None:
    """Return recursively visible actor-card serials after the action.

    Tutor targets can become an attached Tool/Energy or an evolution-stack
    card rather than appearing as a zone root, so all public physical-card
    children are included.  Ambiguous aliases fail closed.
    """

    players = _raw_players(value)
    if players is None or actor not in (0, 1):
        return None
    player = players[actor]
    serials: set[int] = set()
    for name in ("hand", "discard", "active", "bench"):
        rows = _zone(player.get(name))
        if rows is None:
            # A normalized public transition snapshot stores own hand/discard
            # identities as serial lists rather than raw card objects.
            packed_name = {
                "hand": "hand_serials",
                "discard": "discard_serials",
            }.get(name)
            if packed_name is not None and player.get(packed_name) is not None:
                raw = player.get(packed_name)
                if not isinstance(raw, (list, tuple)):
                    return None
                if not _append_packed_serials(raw, serials=serials):
                    return None
                continue
            return None
        for card in rows:
            if not _append_card_serials(
                card,
                serials=serials,
                children=_VISIBLE_CARD_CHILDREN,
            ):
                return None
    return serials


def _option_attachment_index(
    option: Mapping[str, Any],
    *fields: str,
) -> tuple[bool, int | None]:
    """Return a nonnegative explicit attachment index or a malformed marker."""

    present = False
    resolved: int | None = None
    for field in fields:
        if field not in option or option.get(field) is None:
            continue
        present = True
        index = _exact_int(option.get(field))
        if index is None or index < 0:
            return True, None
        if resolved is not None and resolved != index:
            return True, None
        resolved = index
    return present, resolved


def _single_child_zone(
    card: Mapping[str, Any],
    *fields: str,
) -> list[Mapping[str, Any]] | None:
    """Return one unambiguous child zone, accepting singleton pre-evolutions."""

    candidates: list[list[Mapping[str, Any]]] = []
    for field in fields:
        if field not in card or card.get(field) is None:
            continue
        rows = _nested_card_rows(card.get(field))
        if rows is None:
            return None
        if rows:
            candidates.append(rows)
    if len(candidates) > 1:
        return None
    return candidates[0] if candidates else []


def _option_source_serials(step: Mapping[str, Any], *, actor: int) -> set[int] | None:
    """Resolve exact selected sources, including nested public card objects.

    The selected physical source is a Tool/Energy child when its corresponding
    index is present.  An unresolved option is not treated as evidence that a
    tutor target was unused: the whole continuation label is masked instead.
    """

    observation = _mapping(step.get("observation"))
    if observation is None:
        return None
    select = _mapping(observation.get("select"))
    current = _mapping(observation.get("current"))
    if select is None or current is None:
        return None
    options = _zone(select.get("option"))
    selected = _selected_option_indices(step)
    players = _raw_players(observation)
    if options is None or selected is None or players is None:
        return None

    def resolve(option: Mapping[str, Any]) -> int | None:
        owner = _exact_int(option.get("playerIndex"))
        if owner is None:
            owner = actor
        if owner != actor:
            return None
        area = option.get("area")
        index = _exact_int(option.get("index"))
        if index is None or index < 0:
            return None
        if _is_deck_area(area):
            zone = _zone(select.get("deck"))
        else:
            normalized = _normalized_name(area)
            player = players[actor]
            option_type = _normalized_name(option.get("type"))
            if (
                normalized in {"hand", "areatypehand"}
                or _exact_int(area) == 2
                or option_type in {"play", "optiontypeplay"}
            ):
                zone = _zone(player.get("hand"))
            elif normalized in {"discard", "areatypediscard"} or _exact_int(area) == 3:
                zone = _zone(player.get("discard"))
            elif normalized in {"active", "areatypeactive"} or _exact_int(area) == 4:
                zone = _zone(player.get("active"))
            elif normalized in {"bench", "areatypebench"} or _exact_int(area) == 5:
                zone = _zone(player.get("bench"))
            elif normalized in {"looking", "areatypelooking"} or _exact_int(area) == 12:
                zone = _zone(current.get("looking"))
            else:
                return None
        if zone is None or index >= len(zone):
            return None
        host = zone[index]

        tool_present, tool_index = _option_attachment_index(
            option, "toolIndex", "tool_index"
        )
        energy_present, energy_index = _option_attachment_index(
            option, "energyIndex", "energy_index"
        )
        evolution_present, evolution_index = _option_attachment_index(
            option, "preEvolutionIndex", "pre_evolution_index"
        )
        selected_child_count = sum(
            int(present)
            for present in (tool_present, energy_present, evolution_present)
        )
        if selected_child_count > 1:
            return None
        if tool_present:
            if tool_index is None:
                return None
            children = _single_child_zone(host, "tools", "toolCards")
            if children is None or tool_index >= len(children):
                return None
            return _card_serial(children[tool_index])
        if energy_present:
            if energy_index is None:
                return None
            children = _single_child_zone(host, "energyCards")
            if children is None or energy_index >= len(children):
                return None
            return _card_serial(children[energy_index])
        if evolution_present:
            if evolution_index is None:
                return None
            children = _single_child_zone(host, "preEvolution", "preEvolutions")
            if children is None or evolution_index >= len(children):
                return None
            return _card_serial(children[evolution_index])
        return _card_serial(host)

    result: set[int] = set()
    for option_index in selected:
        if option_index >= len(options):
            return None
        serial = resolve(options[option_index])
        if serial is None or serial in result:
            return None
        result.add(serial)
    return result


def _same_actor_continuation_labels(
    steps: Sequence[Mapping[str, Any]],
    index: int,
    *,
    actor: int,
    target_serials: set[int],
) -> tuple[ScalarTarget, CategoricalTarget, str | None]:
    """Trace only a contiguous, same-turn, same-actor observed segment."""

    immediate_source = steps[index].get("transition_after")
    immediate = _unwrap_transition(immediate_source)
    immediate_class = _terminal_class(immediate, actor=actor)
    if immediate_class is None:
        return _masked_scalar(), _masked_category(), "unproven_immediate_transition"
    if immediate_class != "nonterminal":
        return _masked_scalar(), _category(immediate_class), None
    if (
        _has_explicit_boundary(steps[index])
        or _has_explicit_boundary(immediate_source)
        or _has_explicit_boundary(immediate)
    ):
        return _masked_scalar(), _masked_category(), "explicit_boundary"

    start_turn = _turn(steps[index].get("observation"))
    post_turn = _turn(immediate)
    if start_turn is None or post_turn is None or post_turn != start_turn:
        return _masked_scalar(), _masked_category(), "turn_boundary_or_missing_turn"
    immediate_actor = _next_actor(immediate_source)
    if immediate_actor is None:
        return _masked_scalar(), _masked_category(), "missing_next_actor_provenance"
    if immediate_actor != actor:
        return _masked_scalar(), _masked_category(), "actor_boundary"

    used = False
    cursor = index + 1
    while cursor < len(steps):
        step = steps[cursor]
        if _has_explicit_boundary(step):
            return _masked_scalar(), _masked_category(), "explicit_boundary"
        if _actor_from_step(step) != actor or _turn(step.get("observation")) != start_turn:
            return _masked_scalar(), _masked_category(), "actor_or_turn_mismatch"
        source_serials = _option_source_serials(step, actor=actor)
        if source_serials is None:
            return _masked_scalar(), _masked_category(), "unproven_followup_source"
        used = used or bool(target_serials.intersection(source_serials))

        after_source = step.get("transition_after")
        after = _unwrap_transition(after_source)
        outcome = _terminal_class(after, actor=actor)
        if outcome is None:
            return _masked_scalar(), _masked_category(), "missing_followup_transition"
        if outcome != "nonterminal":
            return _scalar(used), _category(outcome), None
        if _has_explicit_boundary(after_source) or _has_explicit_boundary(after):
            return _masked_scalar(), _masked_category(), "explicit_boundary"
        next_actor = _next_actor(after_source)
        if next_actor is None:
            return _masked_scalar(), _masked_category(), "missing_next_actor_provenance"
        if next_actor != actor or _turn(after) != start_turn:
            # The same-actor turn segment has completed and the action did not
            # itself end the game.  This is a factual nonterminal label.
            return _scalar(used), _category("nonterminal"), None
        cursor += 1
    return _masked_scalar(), _masked_category(), "truncated_same_actor_continuation"


def visible_tutor_completion_labels(
    steps: Sequence[Mapping[str, Any]],
    index: int,
    *,
    is_final_selected_action: bool | None = None,
) -> VisibleTutorCompletionLabels:
    """Build tutor selection/follow-through labels for one trajectory row.

    The selected target must be a card visibly exposed at this exact prompt.
    Follow-through is labeled only when card serials and a contiguous same-actor
    same-turn continuation prove it.  A later opponent decision, chance marker,
    missing transition, or truncated trajectory masks continuation labels.
    """

    empty: VisibleTutorCompletionLabels = {
        "selected_card_id": _masked_category(),
        "selected_card_ids": [],
        "selected_card_serials": [],
        "selected_from_visible_deck": _masked_scalar(),
        "selected_target_observed_after_action": _masked_scalar(),
        "same_actor_followup": _masked_scalar(),
        "same_actor_terminal_class": _masked_category(),
        "target_only": True,
        "provenance": "masked",
        "mask_reason": None,
    }
    if index < 0 or index >= len(steps):
        empty["mask_reason"] = "step_index_out_of_range"
        return empty
    step = steps[index]
    if not _is_final_selected_action(step, is_final_selected_action):
        empty["mask_reason"] = "prefix"
        return empty
    if not _immediate_transition_allowed(step):
        empty["mask_reason"] = "non_immediate_or_boundary"
        return empty
    actor = _actor_from_step(step)
    if actor is None:
        empty["mask_reason"] = "missing_actor"
        return empty
    targets, reason = _visible_tutor_targets(step, actor=actor)
    if reason is not None:
        empty["mask_reason"] = reason
        return empty

    card_ids = [_card_id(card) for card in targets]
    serials = [_card_serial(card) for card in targets]
    if any(value is None for value in card_ids) or any(value is None for value in serials):
        empty["mask_reason"] = "visible_target_missing_id_or_serial"
        return empty
    parsed_ids = [int(value) for value in card_ids if value is not None]
    parsed_serials = [int(value) for value in serials if value is not None]
    result: VisibleTutorCompletionLabels = {
        "selected_card_id": (
            {"value": parsed_ids[0], "mask": True}
            if len(parsed_ids) == 1
            else _masked_category()
        ),
        "selected_card_ids": parsed_ids,
        "selected_card_serials": parsed_serials,
        "selected_from_visible_deck": _scalar(True),
        "selected_target_observed_after_action": _masked_scalar(),
        "same_actor_followup": _masked_scalar(),
        "same_actor_terminal_class": _masked_category(),
        "target_only": True,
        "provenance": "selected_visible_deck_card",
        "mask_reason": None,
    }
    after = _unwrap_transition(step.get("transition_after"))
    post_serials = _actor_visible_serials(after, actor=actor)
    if post_serials is not None:
        result["selected_target_observed_after_action"] = _scalar(
            set(parsed_serials).issubset(post_serials)
        )
    followup, terminal, continuation_reason = _same_actor_continuation_labels(
        steps,
        index,
        actor=actor,
        target_serials=set(parsed_serials),
    )
    result["same_actor_followup"] = followup
    result["same_actor_terminal_class"] = terminal
    if continuation_reason is not None:
        result["mask_reason"] = continuation_reason
    return result


def build_own_deck_supervision_targets(
    steps: Sequence[Mapping[str, Any]],
) -> list[OwnDeckSupervisionLabels]:
    """Return immutable-input, target-only labels for complete trajectory rows.

    This helper assumes each supplied row is a complete executed action, as in
    a normal decision trajectory.  Prefix-shaped rows remain masked because
    :func:`terminal_conversion_labels` and
    :func:`visible_tutor_completion_labels` independently inspect their stage
    metadata.
    """

    result: list[OwnDeckSupervisionLabels] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"step {index} must be a mapping")
        result.append(
            {
                "schema": OWN_DECK_SUPERVISION_SCHEMA,
                "version": OWN_DECK_SUPERVISION_VERSION,
                "terminal_conversion": terminal_conversion_labels(step),
                "visible_tutor_completion": visible_tutor_completion_labels(
                    steps, index
                ),
            }
        )
    return result


# Intentional descriptive aliases for future corpus code.  They preserve this
# module's target-only semantics while making call sites read naturally.
build_terminal_conversion_labels = terminal_conversion_labels
build_visible_tutor_completion_labels = visible_tutor_completion_labels


__all__ = [
    "DECK_AREA_WIRE_VALUE",
    "OWN_DECK_SUPERVISION_SCHEMA",
    "OWN_DECK_SUPERVISION_VERSION",
    "TERMINAL_CONVERSION_CLASSES",
    "TERMINAL_CONVERSION_CLASS_COUNT",
    "TERMINAL_CONVERSION_CLASS_SLICE",
    "TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX",
    "TERMINAL_CONVERSION_OUTPUT_DIM",
    "TERMINAL_CONVERSION_OUTPUT_LAYOUT",
    "TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX",
    "TERMINAL_CONVERSION_SCALAR_TARGET_NAMES",
    "VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM",
    "VISIBLE_TUTOR_COMPLETION_OUTPUT_LAYOUT",
    "VISIBLE_TUTOR_COMPLETION_SAME_ACTOR_FOLLOWUP_INDEX",
    "VISIBLE_TUTOR_COMPLETION_SCALAR_SLICE",
    "VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES",
    "VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX",
    "VISIBLE_TUTOR_COMPLETION_SELECTED_TARGET_OBSERVED_AFTER_ACTION_INDEX",
    "VISIBLE_TUTOR_COMPLETION_TERMINAL_CLASS_SLICE",
    "VISIBLE_TUTOR_SELECTION_AUXILIARY_OUTPUT_DIM",
    "VISIBLE_TUTOR_SELECTION_TARGET_KIND",
    "CategoricalTarget",
    "OwnDeckSupervisionLabels",
    "ScalarTarget",
    "TargetMask",
    "TargetVector",
    "TerminalConversionLabels",
    "VisibleTutorCompletionLabels",
    "build_own_deck_supervision_targets",
    "build_terminal_conversion_labels",
    "build_visible_tutor_completion_labels",
    "terminal_conversion_labels",
    "terminal_conversion_target_mask",
    "terminal_conversion_target_vector",
    "visible_tutor_completion_labels",
    "visible_tutor_completion_target_mask",
    "visible_tutor_completion_target_vector",
]
