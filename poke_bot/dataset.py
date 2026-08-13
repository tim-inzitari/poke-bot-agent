"""Bootstrap dataset conversion with explicit accounting.

Loads records produced by :mod:`poke_bot.replay_import` and featurizes each
decision step with :func:`poke_bot.features.build_board_tokens` /
:func:`poke_bot.features.build_option_tokens`. Decisions remain grouped as each
acting seat's causal, deployment-visible history. ``max_context`` truncation is
applied identically to steps and policy targets and reported.

Optional RAM-resident cache under ``config.HARDWARE.cache_dir`` (default
``data/cache/``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from . import config, deck_guides, features
from .own_deck_ledger import (
    OPTION_FEATURE_DIM,
    OWN_DECK_LEDGER_SCHEMA,
    OWN_DECK_LEDGER_SCHEMA_VERSION,
    OwnDeckLedger,
    OwnDeckLedgerError,
    OwnDeckLedgerSnapshot,
)
from .own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    OWN_DECK_SUPERVISION_VERSION,
    TERMINAL_CONVERSION_OUTPUT_DIM,
    VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM,
    build_own_deck_supervision_targets,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)
from .replay_import import InfoSetViolation, assert_info_set

# v5 invalidated cached/sharded decisions built before official JSON enum-name
# scalars (for example "IsFirst" / "Yes" / "Hand") were normalized to the
# same feature rows as their IntEnum values. v6 attaches expanded strategic
# targets from the complete trajectory before context truncation. v7 preserves
# the exact select context and demonstrated STOP status on every policy stage
# for future setup-active/setup-bench objectives. v8 preserves strict causal
# Slowking combo-state labels in immutable temporal feature shards.
DATASET_CACHE_SCHEMA_VERSION = 8
OWN_DECK_SIDECAR_JOIN_SCHEMA = "poke_bot.own_deck_rollout_sidecar_join/v1"
# The compact provenance travels with the feature shard.  A detached, sealed
# receipt is a distinct next-train artifact that binds that provenance to a
# model/code/migration identity; accepting the former as the latter would
# bypass the immutable preparation boundary.
OWN_DECK_SIDECAR_JOIN_RECEIPT_SCHEMA = (
    "poke_bot.own_deck_rollout_store_join_receipt/v1"
)


class OwnDeckSidecarJoinError(ValueError):
    """An immutable r259 sidecar cannot be attached to these feature rows."""


@dataclass
class PolicyStage:
    """One teacher-forced autoregressive option/STOP decision."""

    options: features.SparseVector
    action_combos: list[list[int]]
    target_index: int
    #: Training-only current-deck guide target collapsed from transient scorer
    #: output. ``-1`` masks the stage. Keeping two scalars rather than a Python
    #: score list per legal action prevents large rollout windows from growing
    #: by multiple GiB.
    guide_target_index: int = -1
    #: Unique-best margin, clamped to [0, 1], used to weight guide CE.
    guide_confidence: float = 0.0
    #: Exact public SelectContext for stage-gated future auxiliary objectives.
    select_context: int = -1
    #: Whether the recorded candidate stopped the autoregressive selection at
    #: this stage. This is diagnostic/stratification metadata, never an input.
    selected_is_stop: bool = False
    #: Causal own-deck facts for the current legal candidates.  This remains
    #: ``None`` for legacy corpora and is deliberately separate from targets.
    ledger_option_features: Optional[tuple[tuple[float, ...], ...]] = None


@dataclass
class DecisionSample:
    """One decision timestep inside a game sequence."""

    board: features.SparseVector
    options: features.SparseVector
    action: list[int]
    #: Index of the chosen combo in ``action_combos`` (-1 if not found).
    action_combo_index: int
    action_combos: list[list[int]]
    env_step: int
    aux_labels: dict[str, Any] = field(default_factory=dict)
    #: One-word decoder feature for the action actually taken. Shifted by one
    #: timestep when building causal history so no current-target leakage occurs.
    action_token: Optional[features.SparseVector] = None
    policy_stages: list[PolicyStage] = field(default_factory=list)
    #: Offline-only oracle route attached by the package/full-deck audited
    #: corpus preparer.  It may train an adapter but must never be an inference
    #: feature.
    matchup_adapter_oracle_route: int = -1
    #: Runtime-candidate route derived only from the legally observable public
    #: prefix.  This remains ``-1`` for unrecognized/ambiguous states and for
    #: every route whose independent recognizer gate has not passed.
    matchup_adapter_public_route: int = -1
    #: Immutable actor-visible ledger snapshot reconstructed before context
    #: truncation.  It is a model input, never an auxiliary target.
    ledger_snapshot: Optional[OwnDeckLedgerSnapshot] = None
    #: Target-only successor supervision.  Keep it out of ``aux_labels`` so
    #: neither continuation observations nor transition fields can become
    #: ordinary model inputs.
    own_deck_supervision: Optional[dict[str, Any]] = None
    #: Typed target-family → factorized selected-stage mapping.  Tutor facts
    #: belong to the actual exposed deck-card selection (not a later STOP
    #: candidate); terminal conversion remains the completed final action.
    own_deck_supervision_stage_indices: dict[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    #: Target-only r263 shadow-planner labels aligned to one exact complete
    #: legal-action stage.  Search receipts and proposals are never model
    #: inputs; only the masked four-output target rows are retained here.
    tactical_sequence_supervision: Optional[dict[str, Any]] = None
    tactical_sequence_supervision_stage_index: int = -1
    #: Exact sidecar action-menu digests used only to bridge legacy compact
    #: shards whose integer action combos were deliberately omitted.  These
    #: are provenance, never model inputs.
    sidecar_action_combos_fingerprints: tuple[str, ...] = ()
    #: Immutable public-observation provenance.  This is not featurized or
    #: supplied to the model; it is the fourth canonical r259 join-key field.
    observation_fingerprint: Optional[str] = None


@dataclass
class GameSequence:
    """Whole-game (per-seat) training sequence."""

    episode_id: str
    seat: int
    archetype: str
    opp_archetype: str
    deck: list[int]
    value: float
    decisions: list[DecisionSample]
    info_set_ok: bool = True
    source: str = ""
    policy_targets: Optional[list[Optional[list[float]]]] = None
    factorized_policy_targets: Optional[list[Optional[list[dict[str, Any]]]]] = None
    target_provenance: dict[str, Any] = field(default_factory=dict)
    #: Offline-only, checksummed routing authorization.  This is deliberately a
    #: plain mapping so it survives dataclass replacement, pickle feature shards,
    #: and process boundaries without importing the activation module here.  It
    #: is never embedded into board/action features or used by serving.
    matchup_adapter_training_ticket: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.decisions)


def _combo_index(action: list[int], combos: list[list[int]]) -> int:
    target = list(action)
    for i, c in enumerate(combos):
        if c == target:
            return i
    # Fallback: single-select match on first index.
    if len(target) == 1:
        for i, c in enumerate(combos):
            if c == target:
                return i
    return -1


def _resolve_own_deck_ledger_enabled(value: Optional[bool]) -> bool:
    """Resolve an explicit corpus contract without breaking legacy callers."""

    if value is None:
        return bool(getattr(config.MODEL, "own_deck_ledger_enabled", False))
    if type(value) is not bool:
        raise TypeError("own_deck_ledger_enabled must be bool or None")
    return bool(value)


def _actor_visible_observation(
    step: Mapping[str, Any],
    *,
    verify_info_set: bool,
) -> dict[str, Any]:
    """Return the same remasked observation used for normal featurization.

    Keeping this operation in one helper is important for the causal ledger:
    its snapshots must be derived from exactly the actor-visible input that the
    board/options see, not from a full replay row or future transition.
    """

    obs = step.get("observation") or {}
    if verify_info_set:
        report = assert_info_set(obs, strict=False)
        if not report.ok:
            raise InfoSetViolation("; ".join(report.violations))
        # Prefer the same independently remasked copy used by the historical
        # feature builder.  The ledger itself also excludes opponent fields,
        # but this makes the corpus contract explicit and auditable.
        from .replay_import import _strip_opp_private

        obs, _aux, _ = _strip_opp_private(obs)
    return obs


def _ledger_option_features(
    snapshot: OwnDeckLedgerSnapshot,
    observation: Mapping[str, Any],
    action_combos: list[list[int]],
) -> tuple[tuple[float, ...], ...]:
    """Validate the sparse menu-aligned ledger projection before storage."""

    rows = tuple(
        tuple(float(value) for value in row)
        for row in snapshot.option_features(observation, action_combos)
    )
    if len(rows) != len(action_combos) or any(
        len(row) != OPTION_FEATURE_DIM
        or not all(math.isfinite(value) for value in row)
        for row in rows
    ):
        raise OwnDeckLedgerError("ledger option projection is malformed")
    return rows


def _exact_nonnegative_int(value: Any) -> Optional[int]:
    """Return a JSON integer without accepting bool/coercive values."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _deck_area(value: Any) -> bool:
    """Recognize only the public wire/name forms for a select.deck option."""

    if _exact_nonnegative_int(value) == 1:
        return True
    if hasattr(value, "name"):
        value = getattr(value, "name")
    return "".join(character for character in str(value).lower() if character.isalnum()) == "deck"


def _selected_visible_tutor_stage_indices(
    observation: Mapping[str, Any],
    action: list[int],
    stages: list[PolicyStage],
    supervision: Optional[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Map selected visible-deck cards to their factorized selection stages.

    The target primitive is intentionally card/menu factual rather than a
    generic end-of-action label.  Match both the public menu card identity and
    serial where present, then verify the factorized stage selected exactly the
    corresponding action prefix.  Any ambiguity is masked rather than moved
    to a STOP stage.
    """

    if not isinstance(supervision, Mapping):
        return ()
    labels = supervision.get("visible_tutor_completion")
    if not isinstance(labels, Mapping):
        return ()
    raw_ids = labels.get("selected_card_ids")
    raw_serials = labels.get("selected_card_serials")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        return ()
    if not isinstance(raw_serials, (list, tuple)) or len(raw_serials) != len(raw_ids):
        return ()
    selected_pairs: list[tuple[int, int]] = []
    for raw_id, raw_serial in zip(raw_ids, raw_serials):
        card_id = _exact_nonnegative_int(raw_id)
        serial = _exact_nonnegative_int(raw_serial)
        if card_id is None or serial is None:
            return ()
        selected_pairs.append((card_id, serial))
    # A physical serial cannot be selected twice.  Treat duplicate or partial
    # labels as malformed rather than attaching a factual target to an
    # arbitrary candidate with the same card id.
    if not selected_pairs or len(set(selected_pairs)) != len(selected_pairs):
        return ()
    selected_pair_set = set(selected_pairs)
    current = observation.get("current")
    select = observation.get("select")
    if not isinstance(current, Mapping) or not isinstance(select, Mapping):
        return ()
    actor = _exact_nonnegative_int(current.get("yourIndex"))
    options = select.get("option")
    deck = select.get("deck")
    if (
        actor not in (0, 1)
        or not isinstance(options, (list, tuple))
        or not isinstance(deck, (list, tuple))
    ):
        return ()
    result: list[int] = []
    observed_pairs: set[tuple[int, int]] = set()
    for stage_index, option_index in enumerate(action):
        if stage_index >= len(stages):
            break
        raw_option_index = _exact_nonnegative_int(option_index)
        if raw_option_index is None or raw_option_index >= len(options):
            return ()
        option = options[raw_option_index]
        if not isinstance(option, Mapping) or not _deck_area(option.get("area")):
            continue
        owner = option.get("playerIndex")
        if owner is not None and _exact_nonnegative_int(owner) != actor:
            return ()
        deck_index = _exact_nonnegative_int(option.get("index"))
        if deck_index is None or deck_index >= len(deck):
            return ()
        card = deck[deck_index]
        if not isinstance(card, Mapping):
            return ()
        card_id = _exact_nonnegative_int(card.get("id"))
        serial = _exact_nonnegative_int(card.get("serial"))
        pair = None if card_id is None or serial is None else (card_id, serial)
        if pair not in selected_pair_set:
            continue
        stage = stages[stage_index]
        # A repeated prefix is a STOP stage, not a fresh menu-card selection.
        # Its option feature may still inherit the earlier card-bearing prefix,
        # so it is never a valid tutor supervision location.
        if stage.selected_is_stop:
            return ()
        selected_index = int(stage.target_index)
        if not 0 <= selected_index < len(stage.action_combos):
            return ()
        expected_prefix = [int(value) for value in action[: stage_index + 1]]
        if stage.action_combos[selected_index] != expected_prefix:
            return ()
        if pair in observed_pairs:
            return ()
        observed_pairs.add(pair)
        result.append(stage_index)
    if observed_pairs != selected_pair_set:
        return ()
    return tuple(result)


def _tutor_stage_indices_from_option_features(
    stages: list[PolicyStage],
    option_features: Optional[Sequence[Sequence[Sequence[float]]]] = None,
) -> tuple[int, ...]:
    """Recover exact menu-selection positions from sidecar stage rows only.

    An already-featurized shard has no raw observation.  Its sidecar carries a
    menu-aligned 8-wide ledger row for every candidate; the selected candidate
    can be a visible ``select.deck`` card only when its explicit ``select_count``
    field is positive.  Do not infer a stage from generic exposure/looking.
    """

    if option_features is not None and len(option_features) != len(stages):
        raise OwnDeckSidecarJoinError(
            "sidecar tutor option-feature stages do not align"
        )
    result: list[int] = []
    for stage_index, stage in enumerate(stages):
        rows = (
            stage.ledger_option_features
            if option_features is None
            else option_features[stage_index]
        )
        selected_index = int(stage.target_index)
        if (
            stage.selected_is_stop
            or rows is None
            or not 0 <= selected_index < len(rows)
            or len(rows[selected_index]) != OPTION_FEATURE_DIM
        ):
            continue
        # OPTION_FEATURE_NAMES[5] is select_count.  This positional contract
        # is fixed by the ledger input schema and rejects looking-only rows.
        if float(rows[selected_index][5]) > 0.0:
            result.append(stage_index)
    return tuple(result)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _normalized_sidecar_meta_sha256s(value: Mapping[str, str]) -> dict[str, str]:
    """Validate the exact daily receipt identities used by a sidecar join."""

    if not isinstance(value, Mapping) or not value:
        raise OwnDeckSidecarJoinError(
            "sidecar join requires a nonempty daily-meta SHA-256 mapping"
        )
    output: dict[str, str] = {}
    for raw_day, raw_digest in value.items():
        if not isinstance(raw_day, str) or not raw_day:
            raise OwnDeckSidecarJoinError("sidecar metadata day is invalid")
        if not _is_sha256(raw_digest):
            raise OwnDeckSidecarJoinError(
                f"sidecar metadata digest is invalid for {raw_day}"
            )
        if raw_day in output:
            raise OwnDeckSidecarJoinError("sidecar metadata day is duplicated")
        output[raw_day] = raw_digest
    return dict(sorted(output.items()))


def _sidecar_meta_identity(
    *,
    source_manifest_sha256: str,
    daily_meta_sha256s: Mapping[str, str],
) -> str:
    """Return a stable cache/provenance identity for one immutable side store."""

    if not _is_sha256(source_manifest_sha256):
        raise OwnDeckSidecarJoinError("sidecar source manifest digest is invalid")
    payload = {
        "schema": OWN_DECK_SIDECAR_JOIN_SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "daily_meta_sha256s": dict(daily_meta_sha256s),
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - prevalidated map
        raise OwnDeckSidecarJoinError("sidecar metadata is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _r259_public_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    """Use the side-store's exact public sanitizer and canonical hash ABI.

    This is deliberately delegated to the r259 materializer rather than a
    lookalike local masking rule.  The resulting fingerprint is provenance
    only, but it is part of the canonical four-field join key and must match
    the one written by Elmo byte-for-byte.
    """

    try:
        from .own_deck_rollout_store import (
            _masked_public_observation,
            canonical_json_bytes,
            sha256_bytes,
        )

        current = observation.get("current")
        actor = (
            _exact_nonnegative_int(current.get("yourIndex"))
            if isinstance(current, Mapping)
            else None
        )
        if actor not in (0, 1):
            raise OwnDeckLedgerError("observation has no exact acting seat")
        masked = _masked_public_observation(
            observation,
            expected_seat=actor,
        )
        return sha256_bytes(canonical_json_bytes(masked))
    except OwnDeckLedgerError:
        raise
    except Exception as exc:
        raise OwnDeckLedgerError(
            "could not reproduce r259 public observation fingerprint"
        ) from exc


def _sparse_board_feature_fingerprint(vector: features.SparseVector) -> str:
    """Recreate the r259 board join fingerprint from an existing feature row."""

    try:
        index = [_exact_nonnegative_int(value) for value in vector.index]
        offset = [_exact_nonnegative_int(value) for value in vector.offset]
        values = [float(value) for value in vector.value]
        num_words = _exact_nonnegative_int(vector.num_words)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise OwnDeckSidecarJoinError("feature row has no valid SparseVector board") from exc
    if (
        num_words is None
        or any(value is None for value in index)
        or any(value is None for value in offset)
        or len(index) != len(values)
        or not all(math.isfinite(value) for value in values)
    ):
        raise OwnDeckSidecarJoinError("feature row board SparseVector is malformed")
    payload = {
        "index": [int(value) for value in index if value is not None],
        "value": values,
        "offset": [int(value) for value in offset if value is not None],
        "num_words": num_words,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_action_combos(action_combos: Any) -> list[list[int]]:
    """Return the exact typed candidate matrix used by r259 sidecar hashes."""

    if not isinstance(action_combos, (list, tuple)) or not action_combos:
        raise OwnDeckSidecarJoinError("feature policy stage has no action combinations")
    result: list[list[int]] = []
    for raw_combo in action_combos:
        if not isinstance(raw_combo, (list, tuple)):
            raise OwnDeckSidecarJoinError("feature action combination is malformed")
        combo: list[int] = []
        for raw_option in raw_combo:
            option = _exact_nonnegative_int(raw_option)
            if option is None:
                raise OwnDeckSidecarJoinError("feature action option index is invalid")
            combo.append(option)
        if len(combo) != len(set(combo)):
            raise OwnDeckSidecarJoinError("feature action combination repeats an option")
        result.append(combo)
    return result


def _action_combos_fingerprint(action_combos: Any) -> str:
    combos = _canonical_action_combos(action_combos)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            combos,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _validated_sidecar_option_features(
    value: Any,
    *,
    candidate_count: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != candidate_count:
        raise OwnDeckSidecarJoinError(
            "sidecar option feature candidate count does not match policy stage"
        )
    output: list[tuple[float, ...]] = []
    for raw_row in value:
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != OPTION_FEATURE_DIM:
            raise OwnDeckSidecarJoinError("sidecar option feature width is invalid")
        try:
            row = tuple(float(item) for item in raw_row)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OwnDeckSidecarJoinError("sidecar option feature is nonnumeric") from exc
        if not all(math.isfinite(item) for item in row):
            raise OwnDeckSidecarJoinError("sidecar option feature is nonfinite")
        output.append(row)
    return tuple(output)


def _sidecar_supervision_labels(value: Any) -> dict[str, Any]:
    """Validate target-only vectors/masks against their canonical labels."""

    if not isinstance(value, Mapping):
        raise OwnDeckSidecarJoinError("sidecar supervision is missing")
    if (
        value.get("schema") != OWN_DECK_SUPERVISION_SCHEMA
        or value.get("version") != OWN_DECK_SUPERVISION_VERSION
        or value.get("target_only") is not True
    ):
        raise OwnDeckSidecarJoinError("sidecar supervision contract is invalid")
    families = (
        (
            "terminal_conversion",
            TERMINAL_CONVERSION_OUTPUT_DIM,
            terminal_conversion_target_vector,
            terminal_conversion_target_mask,
        ),
        (
            "visible_tutor_completion",
            VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM,
            visible_tutor_completion_target_vector,
            visible_tutor_completion_target_mask,
        ),
    )
    result: dict[str, Any] = {
        "schema": OWN_DECK_SUPERVISION_SCHEMA,
        "version": OWN_DECK_SUPERVISION_VERSION,
    }
    for name, width, vector_builder, mask_builder in families:
        family = value.get(name)
        if not isinstance(family, Mapping):
            raise OwnDeckSidecarJoinError(f"sidecar supervision lacks {name}")
        labels = family.get("labels")
        raw_vector = family.get("vector")
        raw_mask = family.get("mask")
        if not isinstance(labels, Mapping) or not isinstance(
            raw_vector, (list, tuple)
        ) or not isinstance(raw_mask, (list, tuple)):
            raise OwnDeckSidecarJoinError(f"sidecar {name} target payload is malformed")
        if len(raw_vector) != width or len(raw_mask) != width:
            raise OwnDeckSidecarJoinError(f"sidecar {name} target width is invalid")
        try:
            vector = [float(item) for item in raw_vector]
            expected_vector = [float(item) for item in vector_builder(labels)]
            expected_mask = [bool(item) for item in mask_builder(labels)]
        except (TypeError, ValueError, OverflowError) as exc:
            raise OwnDeckSidecarJoinError(f"sidecar {name} labels are invalid") from exc
        if (
            any(
                isinstance(item, bool) or not math.isfinite(float(item))
                for item in raw_vector
            )
            or any(type(item) is not bool for item in raw_mask)
            or vector != expected_vector
            or list(raw_mask) != expected_mask
        ):
            raise OwnDeckSidecarJoinError(
                f"sidecar {name} labels/vector/mask parity failed"
            )
        result[name] = copy.deepcopy(dict(labels))
    return result


def _local_masked_target_agrees(local: Any, sidecar: Any) -> bool:
    """Compare a local target only when it was actually observable locally."""

    if not isinstance(local, Mapping) or not isinstance(sidecar, Mapping):
        return False
    local_mask = local.get("mask")
    if type(local_mask) is not bool:
        return False
    if local_mask is not True:
        # A raw r241 record often has no post-action transition.  Its masked
        # field is an unavailable oracle, not a factual negative that may
        # contradict the r259 target-only transition observation.
        return True
    return sidecar.get("mask") is True and local.get("value") == sidecar.get("value")


def _local_supervision_agrees_when_observed(
    local: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> bool:
    """Require no contradiction while permitting locally unavailable futures.

    Current selected-card facts and every locally unmasked target must agree.
    A local masked continuation/terminal field is intentionally not compared:
    r241 source rows lack the bounded public ``transition_after`` envelope that
    the r259 materializer reconstructs for target-only supervision.
    """

    if (
        local.get("schema") != OWN_DECK_SUPERVISION_SCHEMA
        or local.get("version") != OWN_DECK_SUPERVISION_VERSION
    ):
        return False
    local_terminal = local.get("terminal_conversion")
    sidecar_terminal = sidecar.get("terminal_conversion")
    local_tutor = local.get("visible_tutor_completion")
    sidecar_tutor = sidecar.get("visible_tutor_completion")
    if not all(
        isinstance(item, Mapping)
        for item in (local_terminal, sidecar_terminal, local_tutor, sidecar_tutor)
    ):
        return False
    for name in ("terminal_class", "prize_closeout", "opponent_knockout"):
        if not _local_masked_target_agrees(
            local_terminal.get(name), sidecar_terminal.get(name)
        ):
            return False
    for name in (
        "selected_card_id",
        "selected_from_visible_deck",
        "selected_target_observed_after_action",
        "same_actor_followup",
        "same_actor_terminal_class",
    ):
        if not _local_masked_target_agrees(local_tutor.get(name), sidecar_tutor.get(name)):
            return False
    local_selection = local_tutor.get("selected_from_visible_deck")
    if isinstance(local_selection, Mapping) and local_selection.get("mask") is True:
        for name in ("selected_card_ids", "selected_card_serials"):
            if local_tutor.get(name) != sidecar_tutor.get(name):
                return False
    return True


def _sidecar_visible_tutor_stage_indices(
    stages: list[PolicyStage],
    supervision: Mapping[str, Any],
    *,
    option_features: Sequence[Sequence[Sequence[float]]],
    legacy_compact_action_chain: bool = False,
) -> tuple[int, ...]:
    """Attach tutor facts only to selected, non-STOP exposed-card stages."""

    labels = supervision.get("visible_tutor_completion")
    if not isinstance(labels, Mapping):
        raise OwnDeckSidecarJoinError("sidecar tutor labels are missing")
    selected = labels.get("selected_from_visible_deck")
    selected_value = selected.get("value") if isinstance(selected, Mapping) else None
    selected_visible = (
        isinstance(selected, Mapping)
        and selected.get("mask") is True
        and not isinstance(selected_value, bool)
        and isinstance(selected_value, (int, float))
        and math.isfinite(float(selected_value))
        and float(selected_value) == 1.0
    )
    stages_for_tutor = _tutor_stage_indices_from_option_features(
        stages,
        option_features,
    )
    if not selected_visible:
        return ()
    raw_ids = labels.get("selected_card_ids")
    raw_serials = labels.get("selected_card_serials")
    if (
        not isinstance(raw_ids, (list, tuple))
        or not isinstance(raw_serials, (list, tuple))
        or not raw_ids
        or len(raw_ids) != len(raw_serials)
    ):
        raise OwnDeckSidecarJoinError("sidecar tutor selected-card labels are malformed")
    selected_pairs: set[tuple[int, int]] = set()
    for raw_id, raw_serial in zip(raw_ids, raw_serials):
        card_id = _exact_nonnegative_int(raw_id)
        serial = _exact_nonnegative_int(raw_serial)
        if card_id is None or serial is None:
            raise OwnDeckSidecarJoinError("sidecar tutor card identity is invalid")
        selected_pairs.add((card_id, serial))
    if (
        legacy_compact_action_chain
        and len(stages_for_tutor) > len(raw_ids)
    ):
        # Compact autoregressive shards repeat the same completed action's
        # menu-aligned ledger facts at each token prefix.  Once the selected
        # prefix chain has been proven against the immutable action token,
        # supervise only the final N completed exposed-card choices.  This
        # preserves the sidecar's factual selected-card cardinality instead
        # of treating tokenizer prefixes as additional tutor selections.
        stages_for_tutor = stages_for_tutor[-len(raw_ids) :]
    if len(selected_pairs) != len(raw_ids) or len(stages_for_tutor) != len(raw_ids):
        raise OwnDeckSidecarJoinError(
            "sidecar tutor labels do not match selected exposed-card stages"
        )
    return stages_for_tutor


def _validate_sidecar_row(
    row: Mapping[str, Any],
    *,
    sequence: GameSequence,
    decision: DecisionSample,
    expected_source_manifest_sha256: str,
    allow_legacy_board_abi: bool = False,
) -> tuple[
    OwnDeckLedgerSnapshot,
    dict[str, Any],
    tuple[tuple[tuple[float, ...], ...], ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    bool,
    bool,
    tuple[str, ...],
]:
    """Validate one row without mutating the in-memory feature shard."""

    from .own_deck_rollout_store import (
        OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
        OWN_DECK_ROLLOUT_SIDECAR_VERSION,
    )

    if (
        row.get("schema") != OWN_DECK_ROLLOUT_SIDECAR_SCHEMA
        or row.get("version") != OWN_DECK_ROLLOUT_SIDECAR_VERSION
        or row.get("source_manifest_sha256") != expected_source_manifest_sha256
    ):
        raise OwnDeckSidecarJoinError("sidecar row schema or source binding mismatches")
    if not _is_sha256(row.get("observation_fingerprint")) or not _is_sha256(
        row.get("ledger_observation_fingerprint")
    ):
        raise OwnDeckSidecarJoinError("sidecar row observation fingerprint is invalid")
    if decision.observation_fingerprint != row.get("observation_fingerprint"):
        raise OwnDeckSidecarJoinError(
            "sidecar row observation fingerprint mismatches feature shard"
        )
    eligibility = row.get("training_eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("active_r241") is not False:
        raise OwnDeckSidecarJoinError("sidecar row is unsafe for active r241")
    snapshot_payload = row.get("ledger_snapshot")
    if not isinstance(snapshot_payload, Mapping):
        raise OwnDeckSidecarJoinError("sidecar row ledger snapshot is missing")
    try:
        snapshot = OwnDeckLedgerSnapshot.from_dict(dict(snapshot_payload))
    except (OwnDeckLedgerError, TypeError, ValueError, KeyError) as exc:
        raise OwnDeckSidecarJoinError("sidecar ledger snapshot is invalid") from exc
    if (
        snapshot.to_dict() != dict(snapshot_payload)
        or snapshot.fail_closed
        or not snapshot.integrity_ok
        or snapshot.observation_fingerprint != row.get("ledger_observation_fingerprint")
        or snapshot.deck_fingerprint != row.get("deck_fingerprint")
    ):
        raise OwnDeckSidecarJoinError("sidecar ledger snapshot parity failed")
    try:
        local_deck_fingerprint = OwnDeckLedger(list(sequence.deck)).deck_fingerprint
    except (OwnDeckLedgerError, TypeError, ValueError, OverflowError) as exc:
        raise OwnDeckSidecarJoinError("feature shard deck cannot bind a ledger") from exc
    if snapshot.deck_fingerprint != local_deck_fingerprint:
        raise OwnDeckSidecarJoinError("sidecar row deck fingerprint mismatches shard")
    board_parity = row.get("board_feature_fingerprint") == _sparse_board_feature_fingerprint(
        decision.board
    )
    if not board_parity and not allow_legacy_board_abi:
        raise OwnDeckSidecarJoinError("sidecar row board fingerprint mismatches shard")

    raw_stages = row.get("policy_stage_option_features")
    if not isinstance(raw_stages, list) or len(raw_stages) != len(decision.policy_stages):
        raise OwnDeckSidecarJoinError("sidecar policy-stage count mismatches shard")

    def _selected_word(stage: PolicyStage, selected: int) -> tuple[tuple[int, float], ...] | None:
        action_token = decision.action_token
        option_count = int(stage.options.num_words)
        if (
            action_token is None
            or int(action_token.num_words) != 1
            or option_count <= 0
            or int(stage.options.pos) != int(action_token.pos) * option_count
            or not 0 <= selected < option_count
        ):
            return None
        start = int(stage.options.offset[selected])
        end = (
            int(stage.options.offset[selected + 1])
            if selected + 1 < option_count
            else len(stage.options.index)
        )
        base = selected * int(action_token.pos)
        return tuple(
            (int(stage.options.index[pos]) - base, float(stage.options.value[pos]))
            for pos in range(start, end)
        )

    def _selected_chain_matches_action_token() -> bool:
        action_token = decision.action_token
        if action_token is None or int(action_token.num_words) != 1:
            return False
        action_word = tuple(
            (int(index), float(value))
            for index, value in zip(action_token.index, action_token.value, strict=True)
        )
        selected_words: list[tuple[tuple[int, float], ...]] = []
        for stage in decision.policy_stages:
            selected_word = _selected_word(stage, int(stage.target_index))
            if (
                selected_word is None
                or not selected_word
                or len(selected_word) > len(action_word)
                or selected_word != action_word[: len(selected_word)]
            ):
                return False
            selected_words.append(selected_word)
        return bool(selected_words) and selected_words[-1] == action_word

    compact_decision_action_parity = False
    if allow_legacy_board_abi and all(
        not stage.action_combos for stage in decision.policy_stages
    ):
        compact_decision_action_parity = _selected_chain_matches_action_token()
    projected_selected_index = False
    option_features: list[tuple[tuple[float, ...], ...]] = []
    selected_indices: list[int] = []
    action_combos_fingerprints: list[str] = []
    for index, (raw_stage, stage) in enumerate(
        zip(raw_stages, decision.policy_stages, strict=True)
    ):
        if not isinstance(raw_stage, Mapping) or _exact_nonnegative_int(
            raw_stage.get("stage_index")
        ) != index:
            raise OwnDeckSidecarJoinError("sidecar policy-stage order mismatches shard")
        legacy_compact_stage = bool(
            allow_legacy_board_abi and not stage.action_combos
        )
        combos = (
            None
            if legacy_compact_stage
            else _canonical_action_combos(stage.action_combos)
        )
        selected_index = _exact_nonnegative_int(raw_stage.get("selected_index"))
        if (
            selected_index is None
            or (
                selected_index != int(stage.target_index)
                and not (
                    legacy_compact_stage and compact_decision_action_parity
                )
            )
            or raw_stage.get("candidate_count")
            != (int(stage.options.num_words) if combos is None else len(combos))
            or (
                combos is not None
                and raw_stage.get("action_combos_fingerprint")
                != _action_combos_fingerprint(combos)
            )
        ):
            raise OwnDeckSidecarJoinError(
                "sidecar policy candidate order mismatches shard "
                f"(episode={sequence.episode_id}, seat={sequence.seat}, "
                f"env_step={decision.env_step}, stage={index}, "
                f"sidecar_selected={selected_index}, "
                f"shard_selected={int(stage.target_index)}, "
                f"sidecar_candidates={raw_stage.get('candidate_count')}, "
                f"shard_candidates={int(stage.options.num_words)}, "
                f"compact={legacy_compact_stage}, "
                f"stages={len(decision.policy_stages)}, "
                f"final_action_parity={compact_decision_action_parity}, "
                f"stage_pos={int(stage.options.pos)}, "
                f"action_pos={int(decision.action_token.pos) if decision.action_token is not None else None})"
            )
        action_combos_fingerprint = raw_stage.get("action_combos_fingerprint")
        if not _is_sha256(action_combos_fingerprint):
            raise OwnDeckSidecarJoinError(
                "sidecar policy action-combo fingerprint is invalid"
            )
        effective_selected_index = int(selected_index)
        if legacy_compact_stage and compact_decision_action_parity:
            effective_selected_index = int(stage.target_index)
            projected_selected_index = (
                projected_selected_index
                or effective_selected_index != int(selected_index)
            )
        rows = _validated_sidecar_option_features(
            raw_stage.get("ledger_option_features"),
            candidate_count=(
                int(stage.options.num_words) if combos is None else len(combos)
            ),
        )
        existing = stage.ledger_option_features
        if existing is not None and tuple(existing) != rows:
            raise OwnDeckSidecarJoinError(
                "raw/sidecar ledger option feature parity failed"
            )
        option_features.append(rows)
        selected_indices.append(effective_selected_index)
        action_combos_fingerprints.append(str(action_combos_fingerprint))

    supervision = _sidecar_supervision_labels(row.get("supervision"))
    existing_supervision = decision.own_deck_supervision
    raw_parity = False
    if existing_supervision is not None:
        if (
            not isinstance(existing_supervision, Mapping)
            or not _local_supervision_agrees_when_observed(
                existing_supervision,
                supervision,
            )
        ):
            raise OwnDeckSidecarJoinError(
                "raw/sidecar observed supervision parity failed"
            )
        raw_parity = True
    existing_snapshot = decision.ledger_snapshot
    if existing_snapshot is not None:
        if not isinstance(existing_snapshot, OwnDeckLedgerSnapshot) or (
            existing_snapshot.to_dict() != snapshot.to_dict()
        ):
            raise OwnDeckSidecarJoinError("raw/sidecar ledger snapshot parity failed")
        raw_parity = True

    stage_indices: dict[str, tuple[int, ...]] = {
        "terminal_conversion": (len(decision.policy_stages) - 1,)
    }
    tutor_indices = _sidecar_visible_tutor_stage_indices(
        decision.policy_stages,
        supervision,
        option_features=option_features,
        legacy_compact_action_chain=compact_decision_action_parity,
    )
    if tutor_indices:
        stage_indices["visible_tutor_completion"] = tutor_indices
    existing_indices = getattr(decision, "own_deck_supervision_stage_indices", {})
    if existing_indices:
        if not isinstance(existing_indices, Mapping):
            raise OwnDeckSidecarJoinError(
                "raw/sidecar target-stage mapping is malformed"
            )
        # A local r241 target can be masked solely because its source row has
        # no post-action transition.  In that case it has no tutor-stage fact
        # to compare with a valid r259 target.  Conversely, every locally
        # present mapping is a factual assertion and must agree exactly.
        for family, indices in existing_indices.items():
            if family not in stage_indices or tuple(indices) != stage_indices[family]:
                raise OwnDeckSidecarJoinError(
                    "raw/sidecar target-stage mapping parity failed"
                )
    return (
        snapshot,
        supervision,
        tuple(option_features),
        stage_indices,
        tuple(selected_indices),
        raw_parity,
        projected_selected_index,
        tuple(action_combos_fingerprints),
    )


def attach_own_deck_sidecar(
    sequences: Sequence[GameSequence],
    *,
    expected_source_manifest_sha256: str,
    daily_meta_sha256s: Mapping[str, str],
    sidecar_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    output_root: Optional[Union[str, Path]] = None,
    test_only_sidecar_rows: bool = False,
    allow_legacy_board_abi: bool = False,
    verified_sidecar_index: Optional[object] = None,
) -> dict[str, Any]:
    """Attach one immutable r259 side store to already-featurized sequences.

    Production accepts only verified daily files via ``output_root``.  Direct
    rows are deliberately test-only and require an explicit opt-in, even though
    they remain bound to daily metadata digests.  Validation completes for every
    row before any decision is mutated, including exact keys, deck/board/
    candidate hashes, all stage matrices, canonical snapshot fingerprints, and
    target-only labels.
    """

    if not _is_sha256(expected_source_manifest_sha256):
        raise OwnDeckSidecarJoinError("expected sidecar source manifest is invalid")
    meta_sha256s = _normalized_sidecar_meta_sha256s(daily_meta_sha256s)
    if (sidecar_rows is None) == (output_root is None):
        raise OwnDeckSidecarJoinError(
            "provide exactly one of sidecar_rows or output_root for sidecar join"
        )
    if type(test_only_sidecar_rows) is not bool:
        raise TypeError("test_only_sidecar_rows must be bool")
    if sidecar_rows is not None and test_only_sidecar_rows is not True:
        verifier = getattr(verified_sidecar_index, "assert_verified", None)
        if not callable(verifier):
            raise OwnDeckSidecarJoinError("direct sidecar rows are test-only; use verified output_root")
        try:
            verifier(expected_source_manifest_sha256=expected_source_manifest_sha256, daily_meta_sha256s=meta_sha256s)
        except Exception as exc:
            raise OwnDeckSidecarJoinError("disk-backed sidecar index provenance failed") from exc
    if output_root is not None and test_only_sidecar_rows:
        raise OwnDeckSidecarJoinError(
            "test-only direct-row flag cannot be used with output_root"
        )
    if output_root is not None:
        from .own_deck_rollout_store import (
            OwnDeckRolloutStoreError,
            iter_daily_sidecar_rows,
            read_daily_meta,
        )

        rows: list[Mapping[str, Any]] = []
        try:
            for day_text, expected_meta_sha in meta_sha256s.items():
                meta = read_daily_meta(output_root, day_text)
                source = ((meta.get("source") or {}).get("manifest") or {}).get(
                    "sha256"
                )
                if (
                    meta.get("meta_sha256") != expected_meta_sha
                    or source != expected_source_manifest_sha256
                ):
                    raise OwnDeckSidecarJoinError(
                        "daily sidecar metadata source/checksum mismatches join"
                    )
                rows.extend(
                    iter_daily_sidecar_rows(
                        output_root,
                        day_text,
                        expected_meta_sha256=expected_meta_sha,
                    )
                )
        except OwnDeckRolloutStoreError as exc:
            raise OwnDeckSidecarJoinError("immutable sidecar read failed") from exc
    else:
        rows = list(sidecar_rows or ())

    expected: dict[
        tuple[str, int, int, str], tuple[GameSequence, DecisionSample]
    ] = {}
    for sequence in sequences:
        episode_id = str(sequence.episode_id)
        seat = _exact_nonnegative_int(sequence.seat)
        if not episode_id or seat not in (0, 1):
            raise OwnDeckSidecarJoinError("feature shard sequence identity is invalid")
        for decision in sequence.decisions:
            env_step = _exact_nonnegative_int(decision.env_step)
            observation_fingerprint = decision.observation_fingerprint
            if env_step is None or not _is_sha256(observation_fingerprint):
                raise OwnDeckSidecarJoinError(
                    "feature shard decision lacks canonical observation fingerprint"
                )
            key = (episode_id, seat, env_step, observation_fingerprint)
            if key in expected:
                raise OwnDeckSidecarJoinError("feature shard has duplicate sidecar key")
            expected[key] = (sequence, decision)
    if not expected:
        raise OwnDeckSidecarJoinError("cannot join an empty feature shard")

    sidecar: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise OwnDeckSidecarJoinError("sidecar iterator emitted a non-object row")
        source_date = row.get("source_date")
        episode_id = row.get("episode_id")
        seat = _exact_nonnegative_int(row.get("seat"))
        env_step = _exact_nonnegative_int(row.get("env_step"))
        observation_fingerprint = row.get("observation_fingerprint")
        if (
            not isinstance(source_date, str)
            or source_date not in meta_sha256s
            or not isinstance(episode_id, str)
            or not episode_id
            or seat not in (0, 1)
            or env_step is None
            or not _is_sha256(observation_fingerprint)
        ):
            raise OwnDeckSidecarJoinError("sidecar row identity is invalid")
        key = (episode_id, seat, env_step, observation_fingerprint)
        if key in sidecar:
            raise OwnDeckSidecarJoinError("sidecar has duplicate record key")
        sidecar[key] = row
    if set(sidecar) != set(expected):
        missing = len(set(expected) - set(sidecar))
        extra = len(set(sidecar) - set(expected))
        raise OwnDeckSidecarJoinError(
            f"sidecar coverage is not one-to-one (missing={missing}, extra={extra})"
        )

    pending: list[
        tuple[
            GameSequence,
            DecisionSample,
            OwnDeckLedgerSnapshot,
            dict[str, Any],
            tuple[tuple[tuple[float, ...], ...], ...],
            dict[str, tuple[int, ...]],
            tuple[int, ...],
            bool,
            bool,
            tuple[str, ...],
        ]
    ] = []
    for key, (sequence, decision) in expected.items():
        (
            snapshot,
            supervision,
            option_features,
            stage_indices,
            selected_indices,
            raw_parity,
            projected_selected_index,
            action_combos_fingerprints,
        ) = (
            _validate_sidecar_row(
                sidecar[key],
                sequence=sequence,
                decision=decision,
                expected_source_manifest_sha256=expected_source_manifest_sha256,
                allow_legacy_board_abi=allow_legacy_board_abi,
            )
        )
        pending.append(
            (
                sequence,
                decision,
                snapshot,
                supervision,
                option_features,
                stage_indices,
                selected_indices,
                raw_parity,
                projected_selected_index,
                action_combos_fingerprints,
            )
        )

    sidecar_identity = _sidecar_meta_identity(
        source_manifest_sha256=expected_source_manifest_sha256,
        daily_meta_sha256s=meta_sha256s,
    )
    provenance = {
        "schema": OWN_DECK_SIDECAR_JOIN_SCHEMA,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "daily_meta_sha256s": dict(meta_sha256s),
        "sidecar_meta_identity": sidecar_identity,
        "record_key": [
            "episode_id",
            "seat",
            "env_step",
            "observation_fingerprint",
        ],
        "joined_decision_count": len(pending),
        "sidecar_record_count": len(sidecar),
        "unmatched_record_count": 0,
        "duplicate_key_count": 0,
        "observation_fingerprint_parity_count": len(pending),
        "canonical_record_key_coverage": True,
        "raw_reconstruction_parity_count": sum(
            int(row[-3]) for row in pending
        ),
        "legacy_compact_selected_index_projection_count": sum(
            int(row[-2]) for row in pending
        ),
        "sidecar_action_menu_digest_count": sum(len(row[-1]) for row in pending),
        "one_to_one_coverage": True,
        "active_r241_training_eligible": False,
        "legacy_board_abi_compatibility": bool(allow_legacy_board_abi),
    }
    for sequence in sequences:
        target_provenance = dict(sequence.target_provenance or {})
        target_provenance["own_deck_sidecar"] = copy.deepcopy(provenance)
        sequence.target_provenance = target_provenance
    for (
        _sequence,
        decision,
        snapshot,
        supervision,
        option_features,
        stage_indices,
        selected_indices,
        _raw_parity,
        _projected_selected_index,
        action_combos_fingerprints,
    ) in pending:
        decision.ledger_snapshot = snapshot
        decision.own_deck_supervision = supervision
        decision.own_deck_supervision_stage_indices = stage_indices
        decision.sidecar_action_combos_fingerprints = action_combos_fingerprints
        for stage, rows, selected_index in zip(
            decision.policy_stages,
            option_features,
            selected_indices,
            strict=True,
        ):
            stage.target_index = int(selected_index)
            stage.ledger_option_features = rows
        if selected_indices:
            decision.action_combo_index = int(selected_indices[0])
    return provenance


def _strict_receipt_count(value: Any, *, label: str) -> int:
    """Accept only a canonical JSON nonnegative integer in a sealed receipt."""

    if type(value) is not int or value < 0:
        raise OwnDeckSidecarJoinError(f"{label} must be a nonnegative integer")
    return value


def _validated_sidecar_join_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the complete exact-key provenance before sealing a receipt."""

    if not isinstance(value, Mapping):
        raise OwnDeckSidecarJoinError("sidecar join provenance is not an object")
    if value.get("schema") != OWN_DECK_SIDECAR_JOIN_SCHEMA:
        raise OwnDeckSidecarJoinError("sidecar join provenance schema is invalid")
    source_manifest_sha256 = value.get("source_manifest_sha256")
    if not _is_sha256(source_manifest_sha256):
        raise OwnDeckSidecarJoinError("sidecar join source manifest digest is invalid")
    daily_meta_sha256s = _normalized_sidecar_meta_sha256s(
        value.get("daily_meta_sha256s")
    )
    expected_identity = _sidecar_meta_identity(
        source_manifest_sha256=source_manifest_sha256,
        daily_meta_sha256s=daily_meta_sha256s,
    )
    if value.get("sidecar_meta_identity") != expected_identity:
        raise OwnDeckSidecarJoinError("sidecar join metadata identity mismatches")
    record_key = value.get("record_key")
    expected_key = [
        "episode_id",
        "seat",
        "env_step",
        "observation_fingerprint",
    ]
    if not isinstance(record_key, list) or record_key != expected_key:
        raise OwnDeckSidecarJoinError("sidecar join record key is not canonical")
    counts = {
        name: _strict_receipt_count(value.get(name), label=f"sidecar join {name}")
        for name in (
            "sidecar_record_count",
            "joined_decision_count",
            "unmatched_record_count",
            "duplicate_key_count",
            "observation_fingerprint_parity_count",
            "raw_reconstruction_parity_count",
        )
    }
    if (
        counts["sidecar_record_count"] != counts["joined_decision_count"]
        or counts["unmatched_record_count"] != 0
        or counts["duplicate_key_count"] != 0
        or counts["observation_fingerprint_parity_count"]
        != counts["joined_decision_count"]
        or counts["raw_reconstruction_parity_count"]
        > counts["joined_decision_count"]
    ):
        raise OwnDeckSidecarJoinError("sidecar join coverage/parity counts are invalid")
    if value.get("one_to_one_coverage") is not True:
        raise OwnDeckSidecarJoinError("sidecar join is not one-to-one")
    if value.get("canonical_record_key_coverage") is not True:
        raise OwnDeckSidecarJoinError("sidecar join lacks canonical key coverage")
    if value.get("active_r241_training_eligible") is not False:
        raise OwnDeckSidecarJoinError("sidecar join is unsafe for active r241")
    return {
        "schema": OWN_DECK_SIDECAR_JOIN_SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "daily_meta_sha256s": daily_meta_sha256s,
        "sidecar_meta_identity": expected_identity,
        "record_key": expected_key,
        **counts,
        "one_to_one_coverage": True,
        "canonical_record_key_coverage": True,
        "active_r241_training_eligible": False,
    }


def _sealed_sidecar_join_receipt(
    provenance: Mapping[str, Any],
    *,
    manifest_sha256: str,
    code_sha256: str,
    sidecar_dataset_sha256: str,
    model_sha256: str,
    migration_receipt_sha256: str,
) -> dict[str, Any]:
    """Build a self-checking receipt without granting any training authority."""

    from .own_deck_training_contract import (
        SIDE_STORE_JOIN_RECEIPT_SCHEMA,
        canonical_json_bytes,
        receipt_digest,
        sha256_bytes,
    )

    if SIDE_STORE_JOIN_RECEIPT_SCHEMA != OWN_DECK_SIDECAR_JOIN_RECEIPT_SCHEMA:
        raise OwnDeckSidecarJoinError("next-train join receipt schema drifted")
    for label, digest in (
        ("manifest", manifest_sha256),
        ("code", code_sha256),
        ("sidecar dataset", sidecar_dataset_sha256),
        ("model", model_sha256),
        ("migration receipt", migration_receipt_sha256),
    ):
        if not _is_sha256(digest):
            raise OwnDeckSidecarJoinError(f"sidecar join {label} digest is invalid")
    normalized = _validated_sidecar_join_provenance(provenance)
    receipt = {
        **normalized,
        "schema": OWN_DECK_SIDECAR_JOIN_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": normalized["source_manifest_sha256"],
        "code_sha256": code_sha256,
        # Keep the future parity contract's spelling too.  It is redundant by
        # design, and makes a code-identity mismatch locally detectable.
        "training_code_sha256": code_sha256,
        "sidecar_dataset_sha256": sidecar_dataset_sha256,
        "model_sha256": model_sha256,
        "migration_receipt_sha256": migration_receipt_sha256,
        "join_provenance_schema": OWN_DECK_SIDECAR_JOIN_SCHEMA,
        "join_provenance_sha256": sha256_bytes(canonical_json_bytes(normalized)),
    }
    receipt["receipt_sha256"] = receipt_digest(receipt)
    return receipt


def validate_own_deck_sidecar_join_receipt(
    value: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate a detached immutable sidecar-join receipt, fail closed.

    Mapping inputs are useful for unit tests and a pipeline's in-memory handoff.
    Path inputs additionally require a regular, non-writable file so a receipt
    that has already been published cannot be silently replaced in place.
    """

    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        path = Path(value)
        try:
            info = path.lstat()
        except OSError as exc:
            raise OwnDeckSidecarJoinError("sidecar join receipt is unreadable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OwnDeckSidecarJoinError(
                "sidecar join receipt must be a regular non-symlink file"
            )
        if info.st_mode & 0o222:
            raise OwnDeckSidecarJoinError("sidecar join receipt is not immutable")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OwnDeckSidecarJoinError("sidecar join receipt is not JSON") from exc
        if not isinstance(loaded, Mapping):
            raise OwnDeckSidecarJoinError("sidecar join receipt is not an object")
        raw = dict(loaded)

    from .own_deck_training_contract import (
        SIDE_STORE_JOIN_RECEIPT_SCHEMA,
        canonical_json_bytes,
        receipt_digest,
        sha256_bytes,
    )

    if (
        SIDE_STORE_JOIN_RECEIPT_SCHEMA != OWN_DECK_SIDECAR_JOIN_RECEIPT_SCHEMA
        or raw.get("schema") != OWN_DECK_SIDECAR_JOIN_RECEIPT_SCHEMA
    ):
        raise OwnDeckSidecarJoinError("sidecar join receipt schema is invalid")
    if raw.get("status") not in {"passed", "complete", "completed"}:
        raise OwnDeckSidecarJoinError("sidecar join receipt is not complete")
    if not _is_sha256(raw.get("receipt_sha256")) or raw.get("receipt_sha256") != receipt_digest(raw):
        raise OwnDeckSidecarJoinError("sidecar join receipt digest mismatches")
    for label, key in (
        ("manifest", "manifest_sha256"),
        ("code", "code_sha256"),
        ("training code", "training_code_sha256"),
        ("sidecar dataset", "sidecar_dataset_sha256"),
        ("model", "model_sha256"),
        ("migration receipt", "migration_receipt_sha256"),
    ):
        if not _is_sha256(raw.get(key)):
            raise OwnDeckSidecarJoinError(
                f"sidecar join receipt {label} digest is invalid"
            )
    if raw.get("code_sha256") != raw.get("training_code_sha256"):
        raise OwnDeckSidecarJoinError("sidecar join receipt code identities differ")
    if raw.get("join_provenance_schema") != OWN_DECK_SIDECAR_JOIN_SCHEMA:
        raise OwnDeckSidecarJoinError("sidecar join receipt provenance schema is invalid")
    # The receipt intentionally has a distinct top-level schema.  Reconstruct
    # only the embedded compact provenance view before validating its exact
    # four-field key/count attestations.
    provenance_view = dict(raw)
    provenance_view["schema"] = raw.get("join_provenance_schema")
    normalized = _validated_sidecar_join_provenance(provenance_view)
    expected_provenance_sha = sha256_bytes(canonical_json_bytes(normalized))
    if raw.get("join_provenance_sha256") != expected_provenance_sha:
        raise OwnDeckSidecarJoinError("sidecar join provenance digest mismatches")
    return copy.deepcopy(raw)


def write_own_deck_sidecar_join_receipt(
    path: str | Path,
    provenance: Mapping[str, Any],
    *,
    manifest_sha256: str,
    code_sha256: str,
    sidecar_dataset_sha256: str,
    model_sha256: str,
    migration_receipt_sha256: str,
) -> Path:
    """Atomically publish one 0444 sealed next-train join receipt.

    Existing paths are never replaced.  The caller still needs the separate
    local/remote parity receipt before a successor canary can be prepared.
    """

    receipt = _sealed_sidecar_join_receipt(
        provenance,
        manifest_sha256=manifest_sha256,
        code_sha256=code_sha256,
        sidecar_dataset_sha256=sidecar_dataset_sha256,
        model_sha256=model_sha256,
        migration_receipt_sha256=migration_receipt_sha256,
    )
    # Validate the exact mapping before it reaches disk.  This also protects
    # against a future receipt-schema change that leaves the writer stale.
    validate_own_deck_sidecar_join_receipt(receipt)
    target = Path(path)
    if target.name in {"", ".", ".."}:
        raise OwnDeckSidecarJoinError("sidecar join receipt path is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = target.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise OwnDeckSidecarJoinError("sidecar join receipt target is unreadable") from exc
    if existing is not None:
        raise OwnDeckSidecarJoinError("sidecar join receipt target already exists")

    from .own_deck_training_contract import canonical_json_bytes

    payload = canonical_json_bytes(receipt)
    temporary_path: Optional[Path] = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        # `link` is an atomic create-if-absent publication primitive.  It
        # cannot replace a receipt another process sealed first.
        os.link(temporary_path, target)
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise OwnDeckSidecarJoinError(
            "sidecar join receipt target already exists"
        ) from exc
    except OSError as exc:
        raise OwnDeckSidecarJoinError("could not publish sidecar join receipt") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return target


def featurize_step(
    step: dict[str, Any],
    deck: list[int],
    *,
    verify_info_set: bool = True,
    ledger_snapshot: Optional[OwnDeckLedgerSnapshot] = None,
    own_deck_supervision: Optional[dict[str, Any]] = None,
    tactical_sequence_supervision: Optional[dict[str, Any]] = None,
) -> DecisionSample:
    """Build board/option tokens for one JSONL step dict."""
    obs = _actor_visible_observation(step, verify_info_set=verify_info_set)
    # Only successor reconstruction needs this extra provenance field.  Keep
    # legacy r241 conversion byte-for-byte feature-compatible and avoid making
    # the r259 side-store a dependency for ordinary corpora.
    observation_fingerprint = (
        # Start from the original wire observation so this calls the side
        # store's masker exactly once, in the same place and order as r259.
        # `obs` can already be remasked for normal feature construction.
        _r259_public_observation_fingerprint(step.get("observation") or {})
        if ledger_snapshot is not None
        else None
    )

    action = [int(x) for x in (step.get("action") or [])]
    stage_defs = features.factorized_teacher_forcing_stages(obs, action)
    policy_stages: list[PolicyStage] = []
    raw_context = (obs.get("select") or {}).get("context", -1)
    try:
        # Use the same enum-name normalization as the option feature builder.
        # Public traces may encode this scalar as either ``1`` or
        # ``"SetupActivePokemon"``; treating the latter as unknown would
        # silently remove the exact setup rows from the future curriculum.
        select_context = features._validated_context(raw_context)
    except (features.FeatureContractError, TypeError, ValueError, OverflowError):
        select_context = -1
    prefix: list[int] = []
    for combos, target_index in stage_defs:
        stage_combos = [list(combo) for combo in combos]
        raw_guide = (
            deck_guides.guide_scores(
                obs,
                stage_combos,
                deck=deck,
            )
            if deck_guides.enabled()
            else None
        )
        guide_target_index = -1
        guide_confidence = 0.0
        if raw_guide is not None:
            candidate = [float(value) for value in raw_guide]
            # Scoring must cover the full legal stage: otherwise a collapsed
            # CE target would silently treat unknown actions as inferior.
            if len(candidate) == len(stage_combos) and all(
                math.isfinite(value) for value in candidate
            ):
                order = sorted(
                    range(len(candidate)),
                    key=lambda index: candidate[index],
                    reverse=True,
                )
                if len(order) >= 2:
                    margin = candidate[order[0]] - candidate[order[1]]
                    if margin > 1e-8:
                        guide_target_index = int(order[0])
                        guide_confidence = min(1.0, max(0.0, float(margin)))
        policy_stages.append(
            PolicyStage(
                options=features.build_option_tokens(obs, stage_combos),
                action_combos=stage_combos,
                target_index=target_index,
                guide_target_index=guide_target_index,
                guide_confidence=guide_confidence,
                select_context=select_context,
                selected_is_stop=(
                    0 <= int(target_index) < len(stage_combos)
                    and stage_combos[int(target_index)] == prefix
                ),
                ledger_option_features=(
                    _ledger_option_features(
                        ledger_snapshot,
                        obs,
                        stage_combos,
                    )
                    if ledger_snapshot is not None
                    else None
                ),
            )
        )
        if 0 <= int(target_index) < len(stage_combos):
            prefix = list(stage_combos[int(target_index)])
    combos = policy_stages[0].action_combos
    board = features.build_board_tokens(obs, deck)
    option_sv = policy_stages[0].options
    action_token = features.build_option_tokens(obs, [action])
    supervision_stage_indices: dict[str, tuple[int, ...]] = {}
    if isinstance(own_deck_supervision, Mapping):
        # Immediate terminal facts describe the complete selected action.
        supervision_stage_indices["terminal_conversion"] = (
            len(policy_stages) - 1,
        )
        tutor_indices = _selected_visible_tutor_stage_indices(
            obs,
            action,
            policy_stages,
            own_deck_supervision,
        )
        if tutor_indices:
            supervision_stage_indices["visible_tutor_completion"] = tutor_indices
    tactical_stage_index = -1
    if isinstance(tactical_sequence_supervision, Mapping):
        rows = tactical_sequence_supervision.get("rows")
        if (
            len(policy_stages) != 1
            or not isinstance(rows, list)
            or len(rows) != len(policy_stages[0].action_combos)
        ):
            raise ValueError(
                "tactical supervision requires exact one-stage legal alignment"
            )
        tactical_stage_index = 0
    return DecisionSample(
        board=board,
        options=option_sv,
        action=action,
        action_combo_index=policy_stages[0].target_index,
        action_combos=combos,
        env_step=int(step.get("env_step", -1)),
        aux_labels=dict(step.get("aux_labels") or {}),
        action_token=action_token,
        policy_stages=policy_stages,
        ledger_snapshot=ledger_snapshot,
        own_deck_supervision=(
            dict(own_deck_supervision)
            if isinstance(own_deck_supervision, Mapping)
            else None
        ),
        own_deck_supervision_stage_indices=supervision_stage_indices,
        tactical_sequence_supervision=(
            dict(tactical_sequence_supervision)
            if isinstance(tactical_sequence_supervision, Mapping)
            else None
        ),
        tactical_sequence_supervision_stage_index=tactical_stage_index,
        observation_fingerprint=observation_fingerprint,
    )


def record_to_sequence(
    record: dict[str, Any],
    *,
    max_context: Optional[int] = None,
    verify_info_set: bool = True,
    own_deck_ledger_enabled: Optional[bool] = None,
) -> Optional[GameSequence]:
    """Convert one record; use :func:`convert_record` for drop diagnostics."""
    seq, _reason, _details = convert_record(
        record,
        max_context=max_context,
        verify_info_set=verify_info_set,
        own_deck_ledger_enabled=own_deck_ledger_enabled,
    )
    return seq


def convert_record(
    record: dict[str, Any],
    *,
    max_context: Optional[int] = None,
    verify_info_set: bool = True,
    own_deck_ledger_enabled: Optional[bool] = None,
) -> tuple[Optional[GameSequence], Optional[str], dict[str, int]]:
    """Convert a record and return ``(sequence, drop_reason, accounting)``."""
    details = {
        "decisions_truncated": 0,
        "policy_targets_padded": 0,
        "policy_targets_truncated": 0,
    }
    if not record.get("info_set_ok", True):
        return None, "info_set_marked_invalid", details
    deck = list(record.get("deck") or [])
    if len(deck) != 60:
        return None, "invalid_deck_length", details
    original_steps = list(record.get("steps") or [])
    if not original_steps:
        return None, "no_steps", details

    max_ctx = int(max_context if max_context is not None else config.MODEL.max_context)
    if max_ctx <= 0:
        return None, "invalid_max_context", details

    # Successor-only input reconstruction deliberately occurs across the
    # complete original trajectory, before both future-label construction and
    # context trimming.  A retained suffix can therefore still carry facts
    # genuinely exposed earlier in the same match, while never consulting
    # ``transition_after`` or any continuation observation.
    ledger_enabled = _resolve_own_deck_ledger_enabled(own_deck_ledger_enabled)
    full_ledger_snapshots: Optional[list[OwnDeckLedgerSnapshot]] = None
    full_own_deck_supervision: Optional[list[dict[str, Any]]] = None
    if ledger_enabled:
        try:
            ledger = OwnDeckLedger(deck)
            full_ledger_snapshots = [
                ledger.observe(
                    _actor_visible_observation(
                        step,
                        verify_info_set=verify_info_set,
                    )
                )
                for step in original_steps
            ]
            raw_supervision = build_own_deck_supervision_targets(original_steps)
            if len(raw_supervision) != len(original_steps):
                raise OwnDeckLedgerError("own-deck supervision lost step alignment")
            full_own_deck_supervision = [
                dict(row) for row in raw_supervision
            ]
        except InfoSetViolation:
            return None, "info_set_violation", details
        except OwnDeckLedgerError:
            return None, "malformed_own_deck_ledger", details
        except (AttributeError, TypeError, ValueError, KeyError, OverflowError):
            # Both successor primitives are intentionally fail-closed.  Do not
            # allow a malformed full-history row to degrade into a suffix-only
            # reconstruction whose omission could look like a valid zero.
            return None, "malformed_own_deck_ledger", details

    raw_targets = record.get("policy_targets")
    policy_targets: Optional[list] = None
    if isinstance(raw_targets, list):
        policy_targets = list(raw_targets)
        if len(policy_targets) < len(original_steps):
            missing = len(original_steps) - len(policy_targets)
            policy_targets.extend([None] * missing)
            details["policy_targets_padded"] = missing
        elif len(policy_targets) > len(original_steps):
            details["policy_targets_truncated"] = len(policy_targets) - len(original_steps)
            policy_targets = policy_targets[: len(original_steps)]

    raw_factorized = record.get("factorized_policy_targets")
    factorized_policy_targets: Optional[list] = None
    if isinstance(raw_factorized, list):
        factorized_policy_targets = list(raw_factorized)
        if len(factorized_policy_targets) < len(original_steps):
            factorized_policy_targets.extend(
                [None] * (len(original_steps) - len(factorized_policy_targets))
            )
        elif len(factorized_policy_targets) > len(original_steps):
            factorized_policy_targets = factorized_policy_targets[
                : len(original_steps)
            ]

    # Future-derived labels must see the complete acting-seat trajectory.
    # Construct them before selecting the model context window; doing this
    # afterwards silently turns valid events beyond the window into negatives.
    from .blackwell_heads import attach_blackwell_strategy_labels
    from .slop_box_combo_targets import (
        attach_slop_box_combo_state_labels,
        is_slop_box_combo_deck,
    )
    from .slowking_combo_targets import (
        attach_slowking_combo_state_labels,
        is_exact_slowking_deck,
    )
    from .strategic_heads import (
        StrategicTargetContractError,
        attach_expanded_strategic_labels,
    )

    attach_blackwell_strategy_labels(original_steps)
    terminal_complete = not bool(
        record.get("incomplete")
        or (record.get("target_provenance") or {}).get(
            "terminal_policy_failure"
        )
    )
    try:
        strategic_contract = attach_expanded_strategic_labels(
            original_steps,
            game_value=float(record.get("value", 0.0)),
            terminal_complete=terminal_complete,
        )
    except StrategicTargetContractError:
        return None, "malformed_strategic_targets", details
    combo_coverage = None
    if is_exact_slowking_deck(deck):
        combo_coverage = attach_slowking_combo_state_labels(
            original_steps,
            deck=deck,
        )
    elif is_slop_box_combo_deck(deck):
        combo_coverage = attach_slop_box_combo_state_labels(
            original_steps,
            deck=deck,
        )

    start = max(0, len(original_steps) - max_ctx)
    steps = original_steps[start:]
    details["decisions_truncated"] = start
    ledger_snapshots = (
        None
        if full_ledger_snapshots is None
        else full_ledger_snapshots[start:]
    )
    own_deck_supervision = (
        None
        if full_own_deck_supervision is None
        else full_own_deck_supervision[start:]
    )
    if ledger_snapshots is not None and len(ledger_snapshots) != len(steps):
        raise AssertionError("ledger snapshot truncation lost step alignment")
    if own_deck_supervision is not None and len(own_deck_supervision) != len(steps):
        raise AssertionError("own-deck supervision truncation lost step alignment")
    if policy_targets is not None:
        policy_targets = policy_targets[start:]
        if len(policy_targets) != len(steps):
            raise AssertionError("step/policy target truncation lost alignment")
    if factorized_policy_targets is not None:
        factorized_policy_targets = factorized_policy_targets[start:]
        if len(factorized_policy_targets) != len(steps):
            raise AssertionError("step/factorized target truncation lost alignment")

    decisions: list[DecisionSample] = []
    for index, step in enumerate(steps):
        try:
            decisions.append(
                featurize_step(
                    step,
                    deck,
                    verify_info_set=verify_info_set,
                    ledger_snapshot=(
                        None if ledger_snapshots is None else ledger_snapshots[index]
                    ),
                    own_deck_supervision=(
                        None
                        if own_deck_supervision is None
                        else own_deck_supervision[index]
                    ),
                    tactical_sequence_supervision=(
                        dict(step.get("tactical_sequence_supervision"))
                        if isinstance(
                            step.get("tactical_sequence_supervision"), Mapping
                        )
                        else None
                    ),
                )
            )
        except InfoSetViolation:
            return None, "info_set_violation", details
        except features.ActionSpaceTooLarge:
            return None, "action_space_too_large", details
        except OwnDeckLedgerError:
            return None, "malformed_own_deck_ledger", details
        except Exception:
            return None, "malformed_step", details

    if not decisions:
        return None, "no_decisions", details

    target_provenance = dict(record.get("target_provenance") or {})
    target_provenance["expanded_strategic_targets"] = strategic_contract
    if ledger_enabled:
        target_provenance["own_deck_ledger"] = {
            "schema": OWN_DECK_LEDGER_SCHEMA,
            "version": OWN_DECK_LEDGER_SCHEMA_VERSION,
            "enabled": True,
            "reconstruction": "full_original_actor_visible_history_before_trimming",
            "transition_after_used_as_input": False,
        }
        target_provenance["own_deck_supervision"] = {
            "schema": OWN_DECK_SUPERVISION_SCHEMA,
            "version": OWN_DECK_SUPERVISION_VERSION,
            "target_only": True,
        }
    if combo_coverage is not None:
        if is_exact_slowking_deck(deck):
            target_provenance["slowking_combo_state_targets"] = combo_coverage
        else:
            target_provenance["slop_box_combo_state_targets"] = combo_coverage
    return GameSequence(
        episode_id=str(record.get("episode_id", "")),
        seat=int(record.get("seat", 0)),
        archetype=str(record.get("archetype", "")),
        opp_archetype=str(record.get("opp_archetype", "")),
        deck=deck,
        value=float(record.get("value", 0.0)),
        decisions=decisions,
        info_set_ok=True,
        source=str(record.get("source", "")),
        policy_targets=policy_targets,
        factorized_policy_targets=factorized_policy_targets,
        target_provenance=target_provenance,
    ), None, details


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cache_key(
    jsonl_path: Path,
    max_context: int,
    verify: bool,
    max_games: int = 0,
    own_deck_ledger_enabled: Optional[bool] = None,
    own_deck_sidecar_meta_identity: Optional[str] = None,
) -> str:
    st = jsonl_path.stat()
    raw = (
        f"{jsonl_path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{max_context}|"
        f"{int(verify)}|{int(max_games)}|dataset={DATASET_CACHE_SCHEMA_VERSION}|"
        f"features={features.FEATURE_SCHEMA_VERSION}|contract=history"
    )
    # Preserve the exact legacy key while disabled so pre-successor caches
    # remain usable.  Enabled corpus construction receives a distinct key and
    # must also pass the explicit payload provenance check below.
    if _resolve_own_deck_ledger_enabled(own_deck_ledger_enabled):
        raw += (
            f"|own_deck_ledger={OWN_DECK_LEDGER_SCHEMA}|"
            f"version={OWN_DECK_LEDGER_SCHEMA_VERSION}"
        )
    if own_deck_sidecar_meta_identity is not None:
        if not _is_sha256(own_deck_sidecar_meta_identity):
            raise OwnDeckSidecarJoinError("sidecar cache identity is invalid")
        # This is intentionally separate from FEATURE_SCHEMA_VERSION: r241 and
        # all legacy shards retain their exact cache identity while a successor
        # cache is bound to the immutable daily metadata receipts it joined.
        raw += f"|own_deck_sidecar_meta={own_deck_sidecar_meta_identity}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class BootstrapDataset:
    """In-memory (optionally disk-cached) list of :class:`GameSequence`.

    Compatible with a future torch Dataset wrapper — exposes ``__len__`` /
    ``__getitem__`` over game sequences. Training loops can flatten decisions
    or pack whole games up to ``MAX_CONTEXT``.
    """

    def __init__(
        self,
        sequences: list[GameSequence],
        *,
        jsonl_path: Optional[Path] = None,
        conversion_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        self.sequences = sequences
        self.jsonl_path = jsonl_path
        self.conversion_stats = dict(conversion_stats or {})

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> GameSequence:
        return self.sequences[idx]

    @property
    def n_decisions(self) -> int:
        return sum(len(s) for s in self.sequences)

    @property
    def info_set_ok_all(self) -> bool:
        return all(s.info_set_ok for s in self.sequences)

    def summary(self) -> dict[str, Any]:
        arches: dict[str, int] = {}
        opp: dict[str, int] = {}
        for s in self.sequences:
            arches[s.archetype] = arches.get(s.archetype, 0) + 1
            opp[s.opp_archetype] = opp.get(s.opp_archetype, 0) + 1
        lengths = [len(s) for s in self.sequences]
        return {
            "n_games": len(self.sequences),
            "n_decisions": self.n_decisions,
            "info_set_ok_all": self.info_set_ok_all,
            "archetypes": arches,
            "opp_archetypes": opp,
            "max_decisions": max(lengths) if lengths else 0,
            "mean_decisions": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "conversion": dict(self.conversion_stats),
        }

    def attach_own_deck_sidecar(
        self,
        *,
        expected_source_manifest_sha256: str,
        daily_meta_sha256s: Mapping[str, str],
        sidecar_rows: Optional[Iterable[Mapping[str, Any]]] = None,
        output_root: Optional[Union[str, Path]] = None,
        test_only_sidecar_rows: bool = False,
    ) -> dict[str, Any]:
        """Fail-closed mutation boundary for a pre-featurized successor corpus."""

        provenance = attach_own_deck_sidecar(
            self.sequences,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            daily_meta_sha256s=daily_meta_sha256s,
            sidecar_rows=sidecar_rows,
            output_root=output_root,
            test_only_sidecar_rows=test_only_sidecar_rows,
        )
        self.conversion_stats.update(
            {
                "own_deck_ledger_enabled": True,
                "own_deck_ledger_schema": OWN_DECK_LEDGER_SCHEMA,
                "own_deck_ledger_schema_version": OWN_DECK_LEDGER_SCHEMA_VERSION,
                "own_deck_sidecar": copy.deepcopy(provenance),
                "own_deck_sidecar_meta_identity": provenance[
                    "sidecar_meta_identity"
                ],
            }
        )
        return provenance

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        *,
        max_context: Optional[int] = None,
        verify_info_set: bool = True,
        max_games: int = 0,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        own_deck_ledger_enabled: Optional[bool] = None,
        own_deck_sidecar_output_root: Optional[Union[str, Path]] = None,
        own_deck_sidecar_meta_sha256s: Optional[Mapping[str, str]] = None,
        own_deck_sidecar_source_manifest_sha256: Optional[str] = None,
    ) -> "BootstrapDataset":
        path = Path(path)
        max_ctx = int(max_context if max_context is not None else config.MODEL.max_context)
        sidecar_requested = any(
            value is not None
            for value in (
                own_deck_sidecar_output_root,
                own_deck_sidecar_meta_sha256s,
                own_deck_sidecar_source_manifest_sha256,
            )
        )
        sidecar_meta_sha256s: Optional[dict[str, str]] = None
        sidecar_meta_identity: Optional[str] = None
        if sidecar_requested:
            if (
                own_deck_sidecar_output_root is None
                or own_deck_sidecar_meta_sha256s is None
                or own_deck_sidecar_source_manifest_sha256 is None
            ):
                raise OwnDeckSidecarJoinError(
                    "sidecar corpus conversion requires output root, source manifest, "
                    "and every daily metadata SHA-256"
                )
            if own_deck_ledger_enabled is False:
                raise OwnDeckSidecarJoinError(
                    "sidecar corpus conversion cannot disable its ledger oracle"
                )
            sidecar_meta_sha256s = _normalized_sidecar_meta_sha256s(
                own_deck_sidecar_meta_sha256s
            )
            sidecar_meta_identity = _sidecar_meta_identity(
                source_manifest_sha256=own_deck_sidecar_source_manifest_sha256,
                daily_meta_sha256s=sidecar_meta_sha256s,
            )
            # Reconstruct raw rows locally before joining so a stale sidecar is
            # compared, never silently preferred.  This does not expose a
            # transition as a model input: snapshots remain actor-visible and
            # supervision stays target-only.
            ledger_enabled = True
        else:
            ledger_enabled = _resolve_own_deck_ledger_enabled(
                own_deck_ledger_enabled
            )
        cache_dir = Path(cache_dir) if cache_dir else Path(config.HARDWARE.cache_dir)
        cache_path: Optional[Path] = None

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = _cache_key(
                path,
                max_ctx,
                verify_info_set,
                max_games,
                own_deck_ledger_enabled=ledger_enabled,
                own_deck_sidecar_meta_identity=sidecar_meta_identity,
            )
            cache_path = cache_dir / f"bootstrap_{path.stem}_{key}.pkl"
            if cache_path.is_file():
                try:
                    with cache_path.open("rb") as fh:
                        payload = pickle.load(fh)
                    ledger_cache_compatible = isinstance(payload, dict) and (
                        not ledger_enabled
                        or (
                            payload.get("own_deck_ledger_enabled") is True
                            and payload.get("own_deck_ledger_schema")
                            == OWN_DECK_LEDGER_SCHEMA
                            and payload.get("own_deck_ledger_schema_version")
                            == OWN_DECK_LEDGER_SCHEMA_VERSION
                        )
                    )
                    sidecar_cache_compatible = isinstance(payload, dict) and (
                        not sidecar_requested
                        or (
                            payload.get("own_deck_sidecar_meta_identity")
                            == sidecar_meta_identity
                            and payload.get(
                                "own_deck_sidecar_source_manifest_sha256"
                            )
                            == own_deck_sidecar_source_manifest_sha256
                            and payload.get("own_deck_sidecar_meta_sha256s")
                            == sidecar_meta_sha256s
                        )
                    )
                    if (
                        isinstance(payload, dict)
                        and payload.get("dataset_schema") == DATASET_CACHE_SCHEMA_VERSION
                        and payload.get("feature_schema") == features.FEATURE_SCHEMA_VERSION
                        and ledger_cache_compatible
                        and sidecar_cache_compatible
                    ):
                        cached = cls(
                            list(payload["sequences"]),
                            jsonl_path=path,
                            conversion_stats=dict(payload.get("conversion_stats") or {}),
                        )
                        if sidecar_requested:
                            # Re-read/re-verify the immutable shards even on a
                            # cache hit; an absent or swapped sidecar may never
                            # be masked by a previously serialized feature set.
                            cached.attach_own_deck_sidecar(
                                expected_source_manifest_sha256=(
                                    own_deck_sidecar_source_manifest_sha256
                                ),
                                daily_meta_sha256s=sidecar_meta_sha256s,
                                output_root=own_deck_sidecar_output_root,
                            )
                        return cached
                except Exception:
                    # Rebuild below. A corrupt/incompatible cache is never
                    # accepted as training data.
                    pass

        sequences: list[GameSequence] = []
        from tqdm.auto import tqdm

        stats: dict[str, Any] = {
            "records_total": 0,
            "records_kept": 0,
            "records_dropped": 0,
            "drop_reasons": {},
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
            "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
            "feature_schema": features.FEATURE_SCHEMA_VERSION,
            "own_deck_ledger_enabled": ledger_enabled,
            "own_deck_ledger_schema": (
                OWN_DECK_LEDGER_SCHEMA
                if ledger_enabled
                else None
            ),
            "own_deck_ledger_schema_version": (
                OWN_DECK_LEDGER_SCHEMA_VERSION
                if ledger_enabled
                else None
            ),
        }
        with path.open("r", encoding="utf-8") as fh:
            for raw in tqdm(fh, desc=f"featurize {path.name}", unit="seq"):
                if not raw.strip():
                    continue
                if max_games > 0 and stats["records_total"] >= max_games:
                    break
                stats["records_total"] += 1
                try:
                    obj = json.loads(raw)
                    if not isinstance(obj, dict):
                        raise TypeError("record is not an object")
                except Exception:
                    reasons = stats["drop_reasons"]
                    reasons["invalid_json"] = reasons.get("invalid_json", 0) + 1
                    stats["records_dropped"] += 1
                    continue
                seq, reason, details = convert_record(
                    obj,
                    max_context=max_ctx,
                    verify_info_set=verify_info_set,
                    own_deck_ledger_enabled=ledger_enabled,
                )
                for key in (
                    "decisions_truncated",
                    "policy_targets_padded",
                    "policy_targets_truncated",
                ):
                    stats[key] += int(details.get(key, 0))
                if seq is None:
                    reason = reason or "unknown"
                    reasons = stats["drop_reasons"]
                    reasons[reason] = reasons.get(reason, 0) + 1
                    stats["records_dropped"] += 1
                    continue
                sequences.append(seq)
                stats["records_kept"] += 1

        if sidecar_requested:
            attached = attach_own_deck_sidecar(
                sequences,
                expected_source_manifest_sha256=(
                    own_deck_sidecar_source_manifest_sha256
                ),
                daily_meta_sha256s=sidecar_meta_sha256s,
                output_root=own_deck_sidecar_output_root,
            )
            stats["own_deck_sidecar"] = copy.deepcopy(attached)
            stats["own_deck_sidecar_meta_identity"] = attached[
                "sidecar_meta_identity"
            ]

        if use_cache and cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(
                    {
                        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                        "feature_schema": features.FEATURE_SCHEMA_VERSION,
                        "own_deck_ledger_enabled": ledger_enabled,
                        "own_deck_ledger_schema": (
                            OWN_DECK_LEDGER_SCHEMA
                            if ledger_enabled
                            else None
                        ),
                        "own_deck_ledger_schema_version": (
                            OWN_DECK_LEDGER_SCHEMA_VERSION
                            if ledger_enabled
                            else None
                        ),
                        "own_deck_sidecar_meta_identity": sidecar_meta_identity,
                        "own_deck_sidecar_source_manifest_sha256": (
                            own_deck_sidecar_source_manifest_sha256
                            if sidecar_requested
                            else None
                        ),
                        "own_deck_sidecar_meta_sha256s": (
                            sidecar_meta_sha256s if sidecar_requested else None
                        ),
                        "sequences": sequences,
                        "conversion_stats": stats,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(cache_path)

        return cls(sequences, jsonl_path=path, conversion_stats=stats)


def load_bootstrap_dataset(
    path: Union[str, Path],
    **kwargs: Any,
) -> BootstrapDataset:
    """Convenience wrapper around :meth:`BootstrapDataset.from_jsonl`."""
    return BootstrapDataset.from_jsonl(path, **kwargs)


def sparse_to_dict(sv: features.SparseVector) -> dict[str, Any]:
    """Serialize a SparseVector (for debugging / light caches)."""
    return {
        "index": list(sv.index),
        "value": list(sv.value),
        "offset": list(sv.offset),
        "num_words": sv.num_words,
    }
