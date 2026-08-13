#!/usr/bin/env python3
"""Fit only eligible r288 Alakazam checklist gates from sealed JSONL rows.

This is intentionally an offline, validation-only compiler.  It imports no
model, simulator, remote worker, service manager, or optimizer state.  The
only fitted values are six bounded checklist scalar gates used by the
parameter-free r288 residual.  All eight checklist answers remain traced, but
the Q5 board-exposure gate and aggregate Q6 disruption gate are initially
trace-only under the r293 overlap audit.  The ninth, separate guide-support
gate is also traced and emitted, while r293--r295 require it to remain exact
zero and calibration-ineligible until a later, separately validated owner
contract.
For every option, the forward pass keeps the greatest-absolute gated
contribution in the closure group and in the continuity group (canonical
channel-order tie break), sums those two winners, and applies the global cap.
The artifact records the winning gate/channel and post-cap residual for every
input row and option.
Input rows carry *already runtime-normalized*, action-order-aligned vectors,
frozen base logits, an expert/recorded chosen option (or an equivalent option
group), and source-disjoint train/validation identifiers.

The command creates a new output directory containing a self-digesting gate
artifact and a self-digesting receipt.  Those files are inert: runtime remains
off until a separate receipt-backed activation explicitly selects them.

Canonical JSONL row shape (extra provenance fields are allowed)::

    {
      "schema": "poke_bot.alakazam_turn_checklist_calibration_row/v1",
      "row_id": "unique-stage-id",
      "source_id": "immutable-source-game-or-record-id",
      "source_group_id": "source-disjoint-group-id",
      "source_split_id": "train",
      "seed_identity": "immutable-calibration-seed-id",
      "exact_new_list_multiset_sha256": "sha256:...",
      "corrected_guide_attachment_sha256": "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b",
      "runtime_layer_schema": "poke_bot.alakazam_turn_checklist_heuristic_logit_layer/v1",
      "runtime_config_sha256": "sha256:<this-config-file-bytes>",
      "runtime_vector_stage": "centered_linf_normalized_pre_gate",
      "frozen_neural_policy": {
        "checkpoint_sha256": "sha256:...",
        "all_neural_model_and_checkpoint_tensors_frozen": true
      },
      "causal_public_evidence_only": true,
      "training_eligible": false,
      "production_eligible": false,
      "replay_eligible": false,
      "existing_route_overlap_deduplication": {
        "post_deduplication_vectors": true,
        "existing_learned_logic_modified": false,
        "overlap_groups": {
          "closure": ["ko_hand_threshold", "immediate_disruption_outcome", "terminal_before_forced_draw"],
          "continuity": ["safe_spend_above_threshold", "replacement_alakazam_line", "unavoidable_draws_before_attack", "unknown_prize_robust_line"],
          "board": ["bench_prize_exposure"]
        },
        "per_channel_overlap_or_distinct_reason": {
          "ko_hand_threshold": "post-audit example reason",
          "safe_spend_above_threshold": "post-audit example reason",
          "replacement_alakazam_line": "post-audit example reason",
          "unavoidable_draws_before_attack": "post-audit example reason",
          "bench_prize_exposure": "board group remains trace-only",
          "immediate_disruption_outcome": "aggregate Q6 remains trace-only",
          "unknown_prize_robust_line": "post-audit example reason",
          "terminal_before_forced_draw": "post-audit example reason"
        },
        "new_layer_gate_or_attenuation_decision": {
          "ko_hand_threshold": "retained",
          "safe_spend_above_threshold": "retained",
          "replacement_alakazam_line": "retained",
          "unavoidable_draws_before_attack": "retained",
          "bench_prize_exposure": "trace_only_fixed_zero",
          "immediate_disruption_outcome": "trace_only_fixed_zero",
          "unknown_prize_robust_line": "retained",
          "terminal_before_forced_draw": "retained"
        }
      },
      "legal_option_order_sha256": "sha256:...",
      "base_logits": [0.1, -0.2],
      "normalized_channel_vectors": {
        "ko_hand_threshold": [1.0, -1.0],
        "safe_spend_above_threshold": [0.0, 0.0],
        "replacement_alakazam_line": [0.0, 0.0],
        "unavoidable_draws_before_attack": [0.0, 0.0],
        "bench_prize_exposure": [0.0, 0.0],
        "immediate_disruption_outcome": [0.0, 0.0],
        "unknown_prize_robust_line": [0.0, 0.0],
        "terminal_before_forced_draw": [0.0, 0.0]
      },
      "normalized_guide_support_vector": [0.0, 0.0],
      "channel_status": {
        "ko_hand_threshold": {"available": true},
        "safe_spend_above_threshold": {
          "available": false,
          "reason": "no_public_threshold_fact"
        },
        "replacement_alakazam_line": {"available": false, "reason": "flat_stage"},
        "unavoidable_draws_before_attack": {"available": false, "reason": "flat_stage"},
        "bench_prize_exposure": {"available": false, "reason": "flat_stage"},
        "immediate_disruption_outcome": {"available": false, "reason": "flat_stage"},
        "unknown_prize_robust_line": {"available": false, "reason": "flat_stage"},
        "terminal_before_forced_draw": {"available": false, "reason": "flat_stage"},
        "guide_support": {"available": false, "reason": "trace_only_flat_stage"}
      },
      "channel_option_availability": {
        "ko_hand_threshold": [true, true],
        "safe_spend_above_threshold": [false, false],
        "replacement_alakazam_line": [true, true],
        "unavoidable_draws_before_attack": [true, true],
        "bench_prize_exposure": [true, true],
        "immediate_disruption_outcome": [true, true],
        "unknown_prize_robust_line": [true, true],
        "terminal_before_forced_draw": [true, true],
        "guide_support": [true, true]
      },
      "target_chosen_option": 0
    }

``target_chosen_option_group`` may replace ``target_chosen_option`` when
several action-order indices are equally acceptable.  The objective is the
negative log probability of that whole group, so it never invents a preference
within an explicitly tied target group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "policy_layers" / "alakazam-turn-checklist-r288.json"

CONFIG_SCHEMA = "poke_bot.alakazam_turn_checklist_heuristic_logit_layer_config/v1"
RUNTIME_SCHEMA = "poke_bot.alakazam_turn_checklist_heuristic_logit_layer/v1"
ROW_SCHEMA = "poke_bot.alakazam_turn_checklist_calibration_row/v1"
ARTIFACT_SCHEMA = "poke_bot.alakazam_turn_checklist_gate_calibration_artifact/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_turn_checklist_gate_calibration_receipt/v1"

CHANNELS = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "replacement_alakazam_line",
    "unavoidable_draws_before_attack",
    "bench_prize_exposure",
    "immediate_disruption_outcome",
    "unknown_prize_robust_line",
    "terminal_before_forced_draw",
)
GUIDE_CHANNEL = "guide_support"
CHECKLIST_GATES = tuple(f"{name}_gate" for name in CHANNELS)
GUIDE_GATE = "separate_guide_gate"
GATES = CHECKLIST_GATES + (GUIDE_GATE,)
TRAINABLE_GATES = (
    "ko_hand_threshold_gate",
    "safe_spend_above_threshold_gate",
    "replacement_alakazam_line_gate",
    "unavoidable_draws_before_attack_gate",
    "unknown_prize_robust_line_gate",
    "terminal_before_forced_draw_gate",
)
FIXED_TRACE_ONLY_GATES = {
    "bench_prize_exposure_gate": 0.0,
    "immediate_disruption_outcome_gate": 0.0,
    GUIDE_GATE: 0.0,
}
OVERLAP_GROUPS = {
    "closure": (
        "ko_hand_threshold",
        "immediate_disruption_outcome",
        "terminal_before_forced_draw",
    ),
    "continuity": (
        "safe_spend_above_threshold",
        "replacement_alakazam_line",
        "unavoidable_draws_before_attack",
        "unknown_prize_robust_line",
    ),
    "board": ("bench_prize_exposure",),
}
RESIDUAL_GROUPS = {
    "closure": (
        "ko_hand_threshold_gate",
        "immediate_disruption_outcome_gate",
        "terminal_before_forced_draw_gate",
    ),
    "continuity": (
        "safe_spend_above_threshold_gate",
        "replacement_alakazam_line_gate",
        "unavoidable_draws_before_attack_gate",
        "unknown_prize_robust_line_gate",
    ),
}
CORRECTED_GUIDE_ATTACHMENT_SHA256 = (
    "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b"
)
_ZERO_TOLERANCE = 1e-6
_PROBABILITY_FLOOR = 1e-300


class ChecklistCalibrationError(ValueError):
    """A source row or calibration contract is not safe to use."""


@dataclass(frozen=True)
class CalibrationConfig:
    path: Path
    file_sha256: str
    exact_new_list_sha256: str
    corrected_guide_attachment_sha256: str
    cap: float
    gate_minimum: float
    gate_maximum: float
    default_gates: tuple[float, ...]
    trainable_gate_names: tuple[str, ...]
    fixed_gates: tuple[tuple[str, float], ...]
    post_deduplication_vectors_required: bool
    max_steps: int
    learning_rate: float
    l2_anchor: float
    early_stopping_patience: int
    train_split_ids: frozenset[str]
    validation_split_ids: frozenset[str]
    minimum_nonflat_rows_per_gate_per_split: int
    minimum_distinct_source_ids_per_split: int
    minimum_distinct_source_groups_per_split: int


@dataclass(frozen=True)
class CalibrationRow:
    row_id: str
    source_id: str
    source_group_id: str
    split_id: str
    seed_identity: str
    base_logits: tuple[float, ...]
    vectors: tuple[tuple[float, ...], ...]
    option_availability: tuple[tuple[bool, ...], ...]
    target_indices: tuple[int, ...]
    frozen_neural_policy: dict[str, Any]
    overlap_deduplication: dict[str, Any]
    legal_option_order_sha256: str
    input_path: str
    input_line: int

    @property
    def width(self) -> int:
        return len(self.base_logits)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _semantic_sha256(payload: Mapping[str, Any], field: str) -> str:
    detached = dict(payload)
    detached.pop(field, None)
    return _sha256_bytes(_canonical_json_bytes(detached))


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ChecklistCalibrationError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChecklistCalibrationError(f"{label} must be an array")
    return list(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChecklistCalibrationError(f"{label} must be a nonempty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label).lower()
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ChecklistCalibrationError(f"{label} must be a sha256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ChecklistCalibrationError(f"{label} must be a sha256 digest") from exc
    return digest


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChecklistCalibrationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ChecklistCalibrationError(f"{label} must be at least {minimum}")
    return int(value)


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChecklistCalibrationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ChecklistCalibrationError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ChecklistCalibrationError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ChecklistCalibrationError(f"{label} must be at most {maximum}")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ChecklistCalibrationError(f"{label} must be a boolean")
    return value


def _sorted_nonempty_text_set(value: Any, label: str) -> frozenset[str]:
    items = [_text(item, f"{label}[]") for item in _sequence(value, label)]
    if not items or len(set(items)) != len(items):
        raise ChecklistCalibrationError(f"{label} must contain unique nonempty IDs")
    return frozenset(items)


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ChecklistCalibrationError(f"cannot read {path}: {exc}") from exc
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }


def _load_config(path: Path) -> CalibrationConfig:
    path = path.expanduser().resolve()
    identity = _file_identity(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChecklistCalibrationError(f"cannot parse config {path}: {exc}") from exc
    payload = _mapping(raw, "config")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ChecklistCalibrationError("config schema is not the r288 checklist config")
    if payload.get("owner_decision_revision") != 288:
        raise ChecklistCalibrationError("config owner_decision_revision must be 288")
    if payload.get("latest_owner_clarification_revision") != 295:
        raise ChecklistCalibrationError("config must include the r295 corrected-guide boundary")
    if payload.get("active") is not False:
        raise ChecklistCalibrationError("r288 calibration config must remain default-off")
    if payload.get("selected_by_runtime_without_explicit_receipt") is not False:
        raise ChecklistCalibrationError("config must not self-select at runtime")
    if (
        payload.get("selected_by_submission_builder") is not False
        or payload.get("selected_by_submission_entrypoint") is not False
    ):
        raise ChecklistCalibrationError("config must not self-select in submission paths")

    exact_new_list = _mapping(payload.get("exact_new_list"), "config.exact_new_list")
    exact_new_list_sha256 = _sha256(
        exact_new_list.get("canonical_multiset_sha256"),
        "config.exact_new_list.canonical_multiset_sha256",
    )
    if exact_new_list.get("require_exact_match_for_nonzero_residual") is not True:
        raise ChecklistCalibrationError("config must fail closed for a non-exact deck")

    guide_attachment = _mapping(payload.get("guide_attachment"), "config.guide_attachment")
    corrected_guide_attachment_sha256 = _sha256(
        guide_attachment.get("sha256"), "config.guide_attachment.sha256"
    )
    if corrected_guide_attachment_sha256 != CORRECTED_GUIDE_ATTACHMENT_SHA256:
        raise ChecklistCalibrationError("config is not bound to the corrected r295 guide bytes")
    corrected_inventory = _mapping(
        guide_attachment.get("corrected_exact_inventory"),
        "config.guide_attachment.corrected_exact_inventory",
    )
    if (
        guide_attachment.get("owner_revision") != 295
        or corrected_inventory
        != {"pokemon": 17, "trainers": 36, "energy": 7, "alakazam": 3}
        or guide_attachment.get("matches_exact_new_list_canonical_multiset") is not True
        or guide_attachment.get("runtime_action_authority") is not False
        or guide_attachment.get("guide_selected_action_override") is not False
    ):
        raise ChecklistCalibrationError("corrected r295 guide authority or exact inventory drifted")

    runtime = _mapping(payload.get("runtime"), "config.runtime")
    if runtime.get("kind") != "parameter_free_deterministic_post_neural_logit_residual":
        raise ChecklistCalibrationError("config runtime must remain parameter-free")
    if runtime.get("enabled_default") is not False:
        raise ChecklistCalibrationError("config runtime must remain disabled by default")
    if (
        runtime.get("checkpoint_tensor_change") is not False
        or runtime.get("legal_action_set_change") is not False
    ):
        raise ChecklistCalibrationError("runtime cannot mutate tensors or legal actions")
    if tuple(_sequence(runtime.get("channel_order"), "config.runtime.channel_order")) != CHANNELS:
        raise ChecklistCalibrationError("runtime channel order drifted")
    if runtime.get("guide_support_channel") != GUIDE_CHANNEL:
        raise ChecklistCalibrationError("runtime guide channel drifted")
    if tuple(_sequence(runtime.get("gate_order"), "config.runtime.gate_order")) != GATES:
        raise ChecklistCalibrationError("runtime gate order drifted")
    if tuple(
        _sequence(
            runtime.get("residual_enabled_gate_order"),
            "config.runtime.residual_enabled_gate_order",
        )
    ) != TRAINABLE_GATES:
        raise ChecklistCalibrationError("runtime residual-enabled gate order drifted")

    guide_policy = _mapping(
        runtime.get("guide_support_policy"), "config.runtime.guide_support_policy"
    )
    if (
        guide_policy.get("mode") != "trace_only"
        or guide_policy.get("runtime_residual_authority") is not False
        or _number(
            guide_policy.get("runtime_gate_required"),
            "config.runtime.guide_support_policy.runtime_gate_required",
        )
        != 0.0
        or guide_policy.get("calibration_eligible") is not False
    ):
        raise ChecklistCalibrationError("r293 guide trace-only boundary drifted")
    expected_gate_channel_map = {
        **{f"{channel}_gate": channel for channel in CHANNELS},
        "separate_guide_gate": GUIDE_CHANNEL,
    }
    gate_channel_map = _mapping(runtime.get("gate_channel_map"), "config.runtime.gate_channel_map")
    if gate_channel_map != expected_gate_channel_map:
        raise ChecklistCalibrationError("runtime gate-to-channel map drifted")
    vector_contract = _mapping(
        runtime.get("per_stage_vector_contract"), "config.runtime.per_stage_vector_contract"
    )
    if (
        vector_contract.get("kind") != "centered_linf_normalized_signed_evidence"
        or vector_contract.get("center_over")
        != "legal_options_with_per_option_causal_evidence"
        or vector_contract.get("normalizer") != "maximum_absolute_centered_value"
        or vector_contract.get("zero_or_unavailable_vector") != "all_zero"
        or vector_contract.get("requires_action_order_alignment") is not True
        or vector_contract.get("requires_per_option_availability_mask") is not True
        or vector_contract.get("channel_available_means")
        != "has_a_nonflat_safe_action_distinction"
        or vector_contract.get("channel_unavailable_with_nonempty_mask")
        != "permitted_only_for_exact_flat_zero_vector"
    ):
        raise ChecklistCalibrationError("runtime vector-normalization contract drifted")
    if _number(
        vector_contract.get("maximum_absolute_component"),
        "config.runtime.per_stage_vector_contract.maximum_absolute_component",
    ) != 1.0:
        raise ChecklistCalibrationError("normalized components must be bounded by one")

    cap = _number(runtime.get("total_residual_cap"), "config.runtime.total_residual_cap", minimum=0.0)
    formula = _mapping(runtime.get("residual_formula"), "config.runtime.residual_formula")
    group_gate_members = _mapping(
        formula.get("group_gate_members"),
        "config.runtime.residual_formula.group_gate_members",
    )
    if (
        formula.get("guide_support_is_separate_gate") is not True
        or formula.get("neutral_when_all_vectors_zero") is not True
        or formula.get("kind")
        != "strongest_absolute_gated_contribution_per_option_per_group"
        or formula.get("grouped_aggregation")
        != "for each option and each residual group, retain the nonzero gated channel contribution with greatest absolute magnitude; break equal-magnitude ties by runtime.channel_order; an exactly-zero group has no winner and contributes zero"
        or tuple(
            _sequence(
                formula.get("group_order"),
                "config.runtime.residual_formula.group_order",
            )
        )
        != tuple(RESIDUAL_GROUPS)
        or {
            name: tuple(
                _sequence(
                    group_gate_members.get(name),
                    f"config.runtime.residual_formula.group_gate_members.{name}",
                )
            )
            for name in RESIDUAL_GROUPS
        }
        != RESIDUAL_GROUPS
        or set(group_gate_members) != set(RESIDUAL_GROUPS)
        or formula.get("tie_break") != "runtime.channel_order_first"
        or formula.get("exact_zero_group_winner") is not None
        or formula.get("pre_clip")
        != "closure_winner_contribution[option_index] + continuity_winner_contribution[option_index]"
        or formula.get("post_clip_trace_required") is not True
        or _number(formula.get("clip_min"), "config.runtime.residual_formula.clip_min") != -cap
        or _number(formula.get("clip_max"), "config.runtime.residual_formula.clip_max") != cap
    ):
        raise ChecklistCalibrationError("runtime grouped residual contract drifted")
    scalar_gates = _mapping(runtime.get("scalar_gates"), "config.runtime.scalar_gates")
    if set(scalar_gates) != set(GATES):
        raise ChecklistCalibrationError("runtime scalar-gate inventory drifted")
    overlap = _mapping(
        runtime.get("existing_route_overlap_deduplication"),
        "config.runtime.existing_route_overlap_deduplication",
    )
    overlap_groups = _mapping(
        overlap.get("overlap_groups"),
        "config.runtime.existing_route_overlap_deduplication.overlap_groups",
    )
    if {
        name: tuple(
            _sequence(
                overlap_groups.get(name),
                f"config.runtime.existing_route_overlap_deduplication.overlap_groups.{name}",
            )
        )
        for name in OVERLAP_GROUPS
    } != OVERLAP_GROUPS:
        raise ChecklistCalibrationError("r293 overlap groups drifted")
    expected_eligibility = {
        "ko_hand_threshold": "calibration_eligible",
        "safe_spend_above_threshold": "calibration_eligible",
        "replacement_alakazam_line": "calibration_eligible",
        "unavoidable_draws_before_attack": "calibration_eligible",
        "bench_prize_exposure": "trace_only_fixed_zero",
        "immediate_disruption_outcome": "trace_only_fixed_zero_until_predicate_separation",
        "unknown_prize_robust_line": "calibration_eligible",
        "terminal_before_forced_draw": "calibration_eligible",
    }
    disruption_boundary = _mapping(
        overlap.get("immediate_disruption_predicate_boundary"),
        "config.runtime.existing_route_overlap_deduplication.immediate_disruption_predicate_boundary",
    )
    if (
        overlap.get("existing_learned_logic_modification_allowed") is not False
        or overlap.get("resolution_owner")
        != "new_checklist_gate_attenuation_or_suppression_only"
        or overlap.get("calibration_vectors_must_be_post_deduplication") is not True
        or overlap.get("calibration_rows_must_carry_post_deduplication_provenance")
        is not True
        or overlap.get("missing_receipt_must_not_be_represented_by_placeholder_sha256")
        is not True
        or _mapping(
            overlap.get("initial_channel_eligibility"),
            "config.runtime.existing_route_overlap_deduplication.initial_channel_eligibility",
        )
        != expected_eligibility
        or disruption_boundary.get("generic_xerosic_or_cage_branch") != "trace_only"
        or disruption_boundary.get("aggregate_channel_gate_currently") != "fixed_zero"
    ):
        raise ChecklistCalibrationError("r293 overlap de-duplication boundary drifted")

    offline = _mapping(payload.get("offline_calibration"), "config.offline_calibration")
    if (
        offline.get("permitted") is not True
        or offline.get("role") != "isolated_validation_only"
        or offline.get("host_scope") != "elmo_isolated_only"
        or offline.get("all_neural_model_and_checkpoint_tensors_frozen") is not True
        or offline.get("trainable_scalars_exact") != len(TRAINABLE_GATES)
        or offline.get("runtime_activation_authority") is not False
        or offline.get("production_loop_authority") is not False
        or offline.get("learner_or_checkpoint_training_authority") is not False
        or offline.get("selector_or_submission_authority") is not False
        or offline.get("elmo_production_readmission_authority") is not False
    ):
        raise ChecklistCalibrationError("offline calibration authority contract drifted")
    trainable_gate_names = tuple(
        _sequence(
            offline.get("trainable_gate_order"),
            "config.offline_calibration.trainable_gate_order",
        )
    )
    if trainable_gate_names != TRAINABLE_GATES:
        raise ChecklistCalibrationError("offline trainable gate order drifted")
    fixed_gates_mapping = _mapping(
        offline.get("fixed_trace_only_gates"),
        "config.offline_calibration.fixed_trace_only_gates",
    )
    if set(fixed_gates_mapping) != set(FIXED_TRACE_ONLY_GATES):
        raise ChecklistCalibrationError("offline fixed trace-only gate inventory drifted")
    fixed_gates = tuple(
        (
            name,
            _number(
                fixed_gates_mapping[name],
                f"config.offline_calibration.fixed_trace_only_gates.{name}",
            ),
        )
        for name in FIXED_TRACE_ONLY_GATES
    )
    if dict(fixed_gates) != FIXED_TRACE_ONLY_GATES:
        raise ChecklistCalibrationError("all initially ineligible gates must remain exact zero")
    guide_calibration = _mapping(
        offline.get("guide_support_calibration"),
        "config.offline_calibration.guide_support_calibration",
    )
    if (
        guide_calibration.get("eligible") is not False
        or guide_calibration.get("trace_collection_allowed") is not True
        or guide_calibration.get("fit_or_nonzero_artifact_allowed") is not False
        or guide_calibration.get("current_artifact_must_emit_exact_zero_gate") is not True
    ):
        raise ChecklistCalibrationError("guide-support calibration boundary drifted")
    gate_bounds = _mapping(offline.get("gate_bounds"), "config.offline_calibration.gate_bounds")
    gate_minimum = _number(gate_bounds.get("minimum"), "config.offline_calibration.gate_bounds.minimum")
    gate_maximum = _number(gate_bounds.get("maximum"), "config.offline_calibration.gate_bounds.maximum")
    if gate_minimum < 0.0 or gate_maximum > cap or gate_minimum > gate_maximum:
        raise ChecklistCalibrationError("scalar-gate bounds must lie within the residual cap")
    default_gates = tuple(
        _number(
            scalar_gates[name],
            f"config.runtime.scalar_gates.{name}",
            minimum=gate_minimum,
            maximum=gate_maximum,
        )
        for name in GATES
    )
    for gate, fixed_value in fixed_gates:
        if default_gates[GATES.index(gate)] != fixed_value:
            raise ChecklistCalibrationError(
                f"runtime default for trace-only {gate} must remain {fixed_value}"
            )

    optimizer = _mapping(offline.get("optimizer"), "config.offline_calibration.optimizer")
    if optimizer.get("kind") != "deterministic_full_batch_projected_gradient_descent":
        raise ChecklistCalibrationError("only deterministic projected-gradient calibration is allowed")
    if optimizer.get("no_model_or_checkpoint_optimizer_state") is not True:
        raise ChecklistCalibrationError("calibration cannot own model/checkpoint optimizer state")
    if (
        optimizer.get("piecewise_group_subgradient")
        != "likelihood_differentiates_only_the_deterministic_current_group_winner_and_zeroes_globally_clipped_options_then_adds_the_declared_l2_anchor_to_all_trainable_gates"
    ):
        raise ChecklistCalibrationError("group-winner subgradient contract drifted")
    max_steps = _integer(optimizer.get("maximum_steps"), "config.offline_calibration.optimizer.maximum_steps", minimum=1)
    learning_rate = _number(
        optimizer.get("learning_rate"),
        "config.offline_calibration.optimizer.learning_rate",
        minimum=0.0,
    )
    if learning_rate <= 0.0:
        raise ChecklistCalibrationError("calibration learning rate must be positive")
    l2_anchor = _number(
        optimizer.get("l2_anchor_to_default"),
        "config.offline_calibration.optimizer.l2_anchor_to_default",
        minimum=0.0,
    )
    early_stopping_patience = _integer(
        optimizer.get("early_stopping_patience"),
        "config.offline_calibration.optimizer.early_stopping_patience",
        minimum=1,
    )
    if optimizer.get("selection_metric") != "source_disjoint_validation_group_cross_entropy":
        raise ChecklistCalibrationError("calibration must select only on source-disjoint validation loss")

    input_contract = _mapping(offline.get("input"), "config.offline_calibration.input")
    if (
        input_contract.get("schema") != ROW_SCHEMA
        or input_contract.get("format") != "jsonl"
        or input_contract.get("precomputed_runtime_vector_stage")
        != "centered_linf_normalized_pre_gate"
        or input_contract.get("requires_base_logits") is not True
        or input_contract.get("requires_exact_new_list_multiset_sha256") is not True
        or input_contract.get("requires_corrected_guide_attachment_sha256") is not True
        or input_contract.get(
            "requires_existing_route_overlap_deduplication_provenance"
        )
        is not True
        or input_contract.get("requires_frozen_neural_policy_identity") is not True
        or input_contract.get("requires_channel_availability_or_unavailable_reason") is not True
        or input_contract.get("requires_channel_option_availability") is not True
        or input_contract.get("requires_target_chosen_option_or_target_chosen_option_group")
        is not True
        or input_contract.get("requires_source_id") is not True
        or input_contract.get("requires_source_group_id") is not True
        or input_contract.get("requires_source_split_id") is not True
        or input_contract.get("requires_seed_identity") is not True
        or input_contract.get("input_rows_must_be_training_ineligible") is not True
        or input_contract.get("input_rows_must_be_production_ineligible") is not True
        or input_contract.get("input_rows_must_be_replay_ineligible") is not True
    ):
        raise ChecklistCalibrationError("calibration input contract drifted")
    train_split_ids = _sorted_nonempty_text_set(
        input_contract.get("training_split_ids"), "config.offline_calibration.input.training_split_ids"
    )
    validation_split_ids = _sorted_nonempty_text_set(
        input_contract.get("validation_split_ids"),
        "config.offline_calibration.input.validation_split_ids",
    )
    if train_split_ids.intersection(validation_split_ids):
        raise ChecklistCalibrationError("train and validation split IDs must be disjoint")
    if input_contract.get("source_id_overlap_between_train_and_validation") != "fail_closed":
        raise ChecklistCalibrationError("source-id overlap must fail closed")
    if input_contract.get("source_group_id_overlap_between_train_and_validation") != "fail_closed":
        raise ChecklistCalibrationError("source-group overlap must fail closed")
    if input_contract.get("seed_identity_overlap_between_train_and_validation") != "fail_closed":
        raise ChecklistCalibrationError("seed-identity overlap must fail closed")

    future_bo250_seed_disjointness = _mapping(
        offline.get("future_bo250_seed_disjointness"),
        "config.offline_calibration.future_bo250_seed_disjointness",
    )
    if (
        future_bo250_seed_disjointness.get("calibration_rows_must_carry_seed_identity")
        is not True
        or future_bo250_seed_disjointness.get(
            "receipt_must_expose_excluded_seed_identities_when_bound"
        )
        is not True
        or future_bo250_seed_disjointness.get("overlap_behavior") != "fail_closed"
        or future_bo250_seed_disjointness.get(
            "unbound_calibration_artifact_may_not_authorize_bo250"
        )
        is not True
        or future_bo250_seed_disjointness.get(
            "bo250_bound_calibration_receipt_required_before_launch"
        )
        is not True
        or future_bo250_seed_disjointness.get(
            "isolated_unit_test_or_shadow_trace_may_use_unbound_artifact"
        )
        is not True
    ):
        raise ChecklistCalibrationError("future BO250 seed-disjointness contract drifted")

    minimum_nonflat_rows_per_gate_per_split = _integer(
        offline.get("minimum_nonflat_rows_per_gate_per_split", 1),
        "config.offline_calibration.minimum_nonflat_rows_per_gate_per_split",
        minimum=1,
    )
    minimum_distinct_source_ids_per_split = _integer(
        offline.get("minimum_distinct_source_ids_per_split", 1),
        "config.offline_calibration.minimum_distinct_source_ids_per_split",
        minimum=1,
    )
    minimum_distinct_source_groups_per_split = _integer(
        offline.get("minimum_distinct_source_groups_per_split", 1),
        "config.offline_calibration.minimum_distinct_source_groups_per_split",
        minimum=1,
    )

    artifact = _mapping(offline.get("artifact"), "config.offline_calibration.artifact")
    if (
        artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("receipt_schema") != RECEIPT_SCHEMA
        or artifact.get("runtime_active") is not False
        or artifact.get("requires_separate_receipt_backed_activation") is not True
        or artifact.get("write_mode") != "create_only"
        or artifact.get("canonical_json") != "utf8_sorted_keys_compact_newline"
    ):
        raise ChecklistCalibrationError("calibration artifact contract drifted")
    calibrated_loading = _mapping(
        offline.get("future_calibrated_artifact_loading"),
        "config.offline_calibration.future_calibrated_artifact_loading",
    )
    expected_artifact_binding = [
        "artifact_sha256_self_digest",
        "runtime_contract.layer_schema",
        "runtime_contract.config_file_sha256",
        "runtime_contract.exact_new_list_multiset_sha256",
        "runtime_contract.corrected_guide_attachment_sha256",
        "runtime_contract.corrected_guide_exact_inventory",
        "runtime_contract.channel_order",
        "runtime_contract.guide_support_channel",
        "runtime_contract.gate_order",
        "runtime_contract.fitted_gate_order",
        "runtime_contract.fixed_trace_only_gates",
        "runtime_contract.gate_channel_map",
        "runtime_contract.grouped_aggregation",
        "runtime_contract.vector_stage",
        "runtime_contract.post_deduplication_vectors_required",
        "runtime_contract.total_residual_cap",
        "runtime_contract.gate_bounds",
    ]
    if (
        calibrated_loading.get("default") != "not_loaded"
        or calibrated_loading.get("requires_separate_receipt_backed_activation") is not True
        or calibrated_loading.get("artifact_schema") != ARTIFACT_SCHEMA
        or calibrated_loading.get("permitted_runtime_override")
        != "runtime.scalar_gates_for_currently_calibration_eligible_six_checklist_gates_only"
        or calibrated_loading.get("fixed_trace_only_gate_override_allowed") is not False
        or calibrated_loading.get("separate_guide_gate_override_allowed") is not False
        or list(
            _sequence(
                calibrated_loading.get("must_match"),
                "config.offline_calibration.future_calibrated_artifact_loading.must_match",
            )
        )
        != expected_artifact_binding
        or calibrated_loading.get("artifact_runtime_active_must_be_false") is not True
        or calibrated_loading.get("ambient_or_implicit_artifact_discovery_allowed")
        is not False
    ):
        raise ChecklistCalibrationError("future calibrated-artifact loading contract drifted")

    forbidden = _mapping(payload.get("forbidden"), "config.forbidden")
    if any(
        forbidden.get(name) is not False
        for name in (
            "hidden_information_inference",
            "search",
            "rollout",
            "mcts",
            "rtp",
            "managed_service_changes",
            "service_start",
            "service_stop",
            "selector_change",
            "kaggle_submission",
        )
    ):
        raise ChecklistCalibrationError("config permits a forbidden runtime or service side effect")

    return CalibrationConfig(
        path=path,
        file_sha256=str(identity["sha256"]),
        exact_new_list_sha256=exact_new_list_sha256,
        corrected_guide_attachment_sha256=corrected_guide_attachment_sha256,
        cap=cap,
        gate_minimum=gate_minimum,
        gate_maximum=gate_maximum,
        default_gates=default_gates,
        trainable_gate_names=trainable_gate_names,
        fixed_gates=fixed_gates,
        post_deduplication_vectors_required=True,
        max_steps=max_steps,
        learning_rate=learning_rate,
        l2_anchor=l2_anchor,
        early_stopping_patience=early_stopping_patience,
        train_split_ids=train_split_ids,
        validation_split_ids=validation_split_ids,
        minimum_nonflat_rows_per_gate_per_split=minimum_nonflat_rows_per_gate_per_split,
        minimum_distinct_source_ids_per_split=minimum_distinct_source_ids_per_split,
        minimum_distinct_source_groups_per_split=minimum_distinct_source_groups_per_split,
    )


def _vector(value: Any, label: str, *, width: int) -> tuple[float, ...]:
    items = _sequence(value, label)
    if len(items) != width:
        raise ChecklistCalibrationError(f"{label} must have exactly {width} action-order values")
    vector = tuple(_number(item, f"{label}[]", minimum=-1.0 - _ZERO_TOLERANCE, maximum=1.0 + _ZERO_TOLERANCE) for item in items)
    return tuple(0.0 if abs(item) <= _ZERO_TOLERANCE else max(-1.0, min(1.0, item)) for item in vector)


def _validate_runtime_normalized_vector(
    vector: tuple[float, ...],
    *,
    option_availability: tuple[bool, ...],
    channel_available: bool,
    label: str,
) -> None:
    """Verify the exact runtime centering domain, not an inferred global mean.

    A checklist fact may be causal for only some candidates in a factorized
    stage.  The runtime leaves the other legal candidates at exact zero.  The
    export therefore carries a same-width boolean mask, and fitting validates
    centering/max-abs scaling only over that public-evidence subset.
    """

    if len(vector) != len(option_availability):
        raise ChecklistCalibrationError(f"{label} availability mask width drifted")
    indices = [index for index, available in enumerate(option_availability) if available]
    # The runtime distinguishes “a fact was exactly available for these
    # options” from “that fact makes a non-flat safe action distinction.”  A
    # fully answerable but flat stage is therefore `available=false` with a
    # nonempty option mask and an all-zero vector.  Preserve that useful audit
    # distinction rather than forcing a false positive availability flag.
    if channel_available and not indices:
        raise ChecklistCalibrationError(f"{label} is marked action-distinguishing without a causal option mask")
    if not indices:
        if any(abs(value) > _ZERO_TOLERANCE for value in vector):
            raise ChecklistCalibrationError(f"{label} is unavailable but has nonzero evidence")
        return
    if any(
        abs(value) > _ZERO_TOLERANCE
        for index, value in enumerate(vector)
        if index not in indices
    ):
        raise ChecklistCalibrationError(f"{label} has evidence outside its causal option mask")
    max_abs = max(abs(vector[index]) for index in indices)
    if not channel_available:
        if max_abs > _ZERO_TOLERANCE:
            raise ChecklistCalibrationError(f"{label} is not action-distinguishing but has nonzero evidence")
        return
    if max_abs <= _ZERO_TOLERANCE:
        raise ChecklistCalibrationError(
            f"{label} is marked action-distinguishing but is exactly flat"
        )
    if abs(sum(vector[index] for index in indices)) > _ZERO_TOLERANCE:
        raise ChecklistCalibrationError(f"{label} is not mean-centered over its causal option mask")
    if abs(max_abs - 1.0) > _ZERO_TOLERANCE:
        raise ChecklistCalibrationError(f"{label} is not max-absolute normalized")


def _channel_statuses(value: Any, *, row_label: str) -> dict[str, bool]:
    statuses = _mapping(value, f"{row_label}.channel_status")
    expected = set(CHANNELS) | {GUIDE_CHANNEL}
    if set(statuses) != expected:
        raise ChecklistCalibrationError(f"{row_label}.channel_status inventory drifted")
    result: dict[str, bool] = {}
    for channel in (*CHANNELS, GUIDE_CHANNEL):
        status = _mapping(statuses[channel], f"{row_label}.channel_status.{channel}")
        available = _strict_bool(
            status.get("available"), f"{row_label}.channel_status.{channel}.available"
        )
        # Runtime traces use ``reason`` for both available and unavailable
        # channels.  ``unavailable_reason`` remains accepted for hand-built
        # offline rows, but capture/export code need not rename the trace.
        reason = status.get("unavailable_reason", status.get("reason"))
        if not available and not isinstance(reason, str):
            raise ChecklistCalibrationError(
                f"{row_label}.channel_status.{channel} needs a reason"
            )
        if not available and not reason.strip():
            raise ChecklistCalibrationError(
                f"{row_label}.channel_status.{channel} needs a nonempty reason"
            )
        result[channel] = available
    return result


def _channel_option_availability(
    value: Any,
    *,
    row_label: str,
    width: int,
) -> dict[str, tuple[bool, ...]]:
    """Parse the action-order-aligned causal mask emitted by the runtime."""

    masks = _mapping(value, f"{row_label}.channel_option_availability")
    expected = set(CHANNELS) | {GUIDE_CHANNEL}
    if set(masks) != expected:
        raise ChecklistCalibrationError(
            f"{row_label}.channel_option_availability inventory drifted"
        )
    result: dict[str, tuple[bool, ...]] = {}
    for channel in (*CHANNELS, GUIDE_CHANNEL):
        raw_mask = _sequence(
            masks[channel], f"{row_label}.channel_option_availability.{channel}"
        )
        if len(raw_mask) != width:
            raise ChecklistCalibrationError(
                f"{row_label}.channel_option_availability.{channel} must have exactly {width} action-order values"
            )
        result[channel] = tuple(
            _strict_bool(
                item,
                f"{row_label}.channel_option_availability.{channel}[]",
            )
            for item in raw_mask
        )
    return result


def _overlap_deduplication_provenance(value: Any, *, row_label: str) -> dict[str, Any]:
    """Require auditable post-overlap vectors without inventing a receipt.

    A real overlap-audit receipt may be attached once one exists.  Until then,
    the row carries exact local provenance and simply omits that receipt field;
    placeholder or all-zero digests are rejected.
    """

    provenance = _mapping(
        value, f"{row_label}.existing_route_overlap_deduplication"
    )
    if provenance.get("post_deduplication_vectors") is not True:
        raise ChecklistCalibrationError(
            f"{row_label} does not attest post-de-duplication calibration vectors"
        )
    if provenance.get("existing_learned_logic_modified") is not False:
        raise ChecklistCalibrationError(
            f"{row_label} may not modify learned heads/Fusion/OwnDeck/Adapter logic"
        )
    groups = _mapping(
        provenance.get("overlap_groups"),
        f"{row_label}.existing_route_overlap_deduplication.overlap_groups",
    )
    parsed_groups = {
        name: tuple(
            _sequence(
                groups.get(name),
                f"{row_label}.existing_route_overlap_deduplication.overlap_groups.{name}",
            )
        )
        for name in OVERLAP_GROUPS
    }
    if parsed_groups != OVERLAP_GROUPS or set(groups) != set(OVERLAP_GROUPS):
        raise ChecklistCalibrationError(f"{row_label} overlap-group contract drifted")

    reasons = _mapping(
        provenance.get("per_channel_overlap_or_distinct_reason"),
        f"{row_label}.existing_route_overlap_deduplication.per_channel_overlap_or_distinct_reason",
    )
    decisions = _mapping(
        provenance.get("new_layer_gate_or_attenuation_decision"),
        f"{row_label}.existing_route_overlap_deduplication.new_layer_gate_or_attenuation_decision",
    )
    if set(reasons) != set(CHANNELS) or set(decisions) != set(CHANNELS):
        raise ChecklistCalibrationError(
            f"{row_label} needs overlap rationale and a new-layer decision for all eight channels"
        )
    normalized_reasons = {
        channel: _text(
            reasons[channel],
            f"{row_label}.existing_route_overlap_deduplication.per_channel_overlap_or_distinct_reason.{channel}",
        )
        for channel in CHANNELS
    }
    eligible_decisions = {"retained", "attenuated", "suppressed"}
    normalized_decisions: dict[str, str] = {}
    for channel in CHANNELS:
        decision = _text(
            decisions[channel],
            f"{row_label}.existing_route_overlap_deduplication.new_layer_gate_or_attenuation_decision.{channel}",
        )
        gate = f"{channel}_gate"
        if gate in FIXED_TRACE_ONLY_GATES:
            if decision != "trace_only_fixed_zero":
                raise ChecklistCalibrationError(
                    f"{row_label} {channel} must remain trace_only_fixed_zero"
                )
        elif decision not in eligible_decisions:
            raise ChecklistCalibrationError(
                f"{row_label} {channel} decision must be retained, attenuated, or suppressed"
            )
        normalized_decisions[channel] = decision

    normalized: dict[str, Any] = {
        "post_deduplication_vectors": True,
        "existing_learned_logic_modified": False,
        "overlap_groups": {name: list(values) for name, values in OVERLAP_GROUPS.items()},
        "runtime_grouped_aggregation": {
            "group_order": list(RESIDUAL_GROUPS),
            "group_gate_members": {
                name: list(gates) for name, gates in RESIDUAL_GROUPS.items()
            },
            "winner": "greatest absolute gated contribution per option and group",
            "tie_break": "runtime.channel_order_first",
            "exact_zero_group_winner": None,
            "final": "sum group winners then clamp to total_residual_cap",
        },
        "per_channel_overlap_or_distinct_reason": normalized_reasons,
        "new_layer_gate_or_attenuation_decision": normalized_decisions,
    }
    if "overlap_audit_receipt" in provenance:
        receipt = _mapping(
            provenance["overlap_audit_receipt"],
            f"{row_label}.existing_route_overlap_deduplication.overlap_audit_receipt",
        )
        receipt_sha256 = _sha256(
            receipt.get("sha256"),
            f"{row_label}.existing_route_overlap_deduplication.overlap_audit_receipt.sha256",
        )
        if receipt_sha256 == "sha256:" + ("0" * 64):
            raise ChecklistCalibrationError(
                f"{row_label} overlap-audit receipt cannot use a placeholder digest"
            )
        normalized["overlap_audit_receipt"] = {
            "identity": _text(
                receipt.get("identity"),
                f"{row_label}.existing_route_overlap_deduplication.overlap_audit_receipt.identity",
            ),
            "sha256": receipt_sha256,
        }
    return normalized


def _target_indices(row: Mapping[str, Any], *, width: int, row_label: str) -> tuple[int, ...]:
    singular_present = "target_chosen_option" in row
    group_present = "target_chosen_option_group" in row
    if singular_present == group_present:
        raise ChecklistCalibrationError(
            f"{row_label} must provide exactly one target_chosen_option or target_chosen_option_group"
        )
    if singular_present:
        target = (_integer(row["target_chosen_option"], f"{row_label}.target_chosen_option", minimum=0),)
    else:
        target = tuple(
            _integer(value, f"{row_label}.target_chosen_option_group[]", minimum=0)
            for value in _sequence(row["target_chosen_option_group"], f"{row_label}.target_chosen_option_group")
        )
        if not target or len(set(target)) != len(target):
            raise ChecklistCalibrationError(f"{row_label}.target_chosen_option_group must contain unique indices")
    if any(index >= width for index in target):
        raise ChecklistCalibrationError(f"{row_label} target index is outside its legal stage")
    return tuple(sorted(target))


def _parse_row(
    value: Any,
    *,
    config: CalibrationConfig,
    input_path: Path,
    input_line: int,
) -> CalibrationRow:
    row = _mapping(value, f"{input_path}:{input_line}")
    row_label = f"{input_path}:{input_line}"
    if row.get("schema") != ROW_SCHEMA:
        raise ChecklistCalibrationError(f"{row_label} has the wrong row schema")
    if _sha256(
        row.get("exact_new_list_multiset_sha256"),
        f"{row_label}.exact_new_list_multiset_sha256",
    ) != config.exact_new_list_sha256:
        raise ChecklistCalibrationError(f"{row_label} is not bound to the exact r241 new list")
    if _sha256(
        row.get("corrected_guide_attachment_sha256"),
        f"{row_label}.corrected_guide_attachment_sha256",
    ) != config.corrected_guide_attachment_sha256:
        raise ChecklistCalibrationError(f"{row_label} is not bound to the corrected r295 guide")
    if row.get("runtime_layer_schema") != RUNTIME_SCHEMA:
        raise ChecklistCalibrationError(f"{row_label} is not an r288 layer trace")
    if _sha256(row.get("runtime_config_sha256"), f"{row_label}.runtime_config_sha256") != config.file_sha256:
        raise ChecklistCalibrationError(f"{row_label} uses a different runtime config")
    if row.get("runtime_vector_stage") != "centered_linf_normalized_pre_gate":
        raise ChecklistCalibrationError(f"{row_label} does not contain pre-gate normalized vectors")
    if row.get("causal_public_evidence_only") is not True:
        raise ChecklistCalibrationError(f"{row_label} lacks causal public-evidence attestation")
    if row.get("training_eligible") is not False:
        raise ChecklistCalibrationError(f"{row_label} must be training-ineligible")
    if row.get("production_eligible") is not False:
        raise ChecklistCalibrationError(f"{row_label} must be production-ineligible")
    if row.get("replay_eligible") is not False:
        raise ChecklistCalibrationError(f"{row_label} must be replay-ineligible")

    row_id = _text(row.get("row_id"), f"{row_label}.row_id")
    source_id = _text(row.get("source_id"), f"{row_label}.source_id")
    source_group_id = _text(row.get("source_group_id"), f"{row_label}.source_group_id")
    split_id = _text(row.get("source_split_id"), f"{row_label}.source_split_id")
    seed_identity = _text(row.get("seed_identity"), f"{row_label}.seed_identity")
    if split_id not in config.train_split_ids | config.validation_split_ids:
        raise ChecklistCalibrationError(f"{row_label} has an unrecognized source split {split_id!r}")
    legal_option_order_sha256 = _sha256(
        row.get("legal_option_order_sha256"), f"{row_label}.legal_option_order_sha256"
    )
    frozen_neural_policy = _mapping(row.get("frozen_neural_policy"), f"{row_label}.frozen_neural_policy")
    _sha256(
        frozen_neural_policy.get("checkpoint_sha256"),
        f"{row_label}.frozen_neural_policy.checkpoint_sha256",
    )
    if frozen_neural_policy.get("all_neural_model_and_checkpoint_tensors_frozen") is not True:
        raise ChecklistCalibrationError(f"{row_label} does not attest frozen neural tensors")
    overlap_deduplication = _overlap_deduplication_provenance(
        row.get("existing_route_overlap_deduplication"), row_label=row_label
    )

    base_logits = tuple(
        _number(item, f"{row_label}.base_logits[]")
        for item in _sequence(row.get("base_logits"), f"{row_label}.base_logits")
    )
    if len(base_logits) < 2:
        raise ChecklistCalibrationError(f"{row_label}.base_logits needs at least two legal options")
    statuses = _channel_statuses(row.get("channel_status"), row_label=row_label)
    option_availability = _channel_option_availability(
        row.get("channel_option_availability"),
        row_label=row_label,
        width=len(base_logits),
    )
    vectors_payload = _mapping(
        row.get("normalized_channel_vectors"), f"{row_label}.normalized_channel_vectors"
    )
    if set(vectors_payload) != set(CHANNELS):
        raise ChecklistCalibrationError(f"{row_label}.normalized_channel_vectors inventory drifted")
    vectors: list[tuple[float, ...]] = []
    for channel in CHANNELS:
        vector = _vector(
            vectors_payload[channel],
            f"{row_label}.normalized_channel_vectors.{channel}",
            width=len(base_logits),
        )
        _validate_runtime_normalized_vector(
            vector,
            option_availability=option_availability[channel],
            channel_available=statuses[channel],
            label=f"{row_label}.normalized_channel_vectors.{channel}",
        )
        vectors.append(vector)
    guide_vector = _vector(
        row.get("normalized_guide_support_vector"),
        f"{row_label}.normalized_guide_support_vector",
        width=len(base_logits),
    )
    _validate_runtime_normalized_vector(
        guide_vector,
        option_availability=option_availability[GUIDE_CHANNEL],
        channel_available=statuses[GUIDE_CHANNEL],
        label=f"{row_label}.normalized_guide_support_vector",
    )
    vectors.append(guide_vector)
    target = _target_indices(row, width=len(base_logits), row_label=row_label)
    return CalibrationRow(
        row_id=row_id,
        source_id=source_id,
        source_group_id=source_group_id,
        split_id=split_id,
        seed_identity=seed_identity,
        base_logits=base_logits,
        vectors=tuple(vectors),
        option_availability=tuple(
            option_availability[channel] for channel in (*CHANNELS, GUIDE_CHANNEL)
        ),
        target_indices=target,
        frozen_neural_policy=json.loads(_canonical_json_bytes(frozen_neural_policy).decode("utf-8")),
        overlap_deduplication=overlap_deduplication,
        legal_option_order_sha256=legal_option_order_sha256,
        input_path=str(input_path),
        input_line=input_line,
    )


def _load_rows(paths: Sequence[Path], config: CalibrationConfig) -> tuple[list[CalibrationRow], list[dict[str, Any]]]:
    rows: list[CalibrationRow] = []
    identities: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        identity = _file_identity(path)
        identities.append(identity)
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ChecklistCalibrationError(f"cannot open JSONL input {path}: {exc}") from exc
        with stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ChecklistCalibrationError(
                        f"{path}:{line_number} is not valid JSONL: {exc}"
                    ) from exc
                parsed = _parse_row(
                    payload,
                    config=config,
                    input_path=path,
                    input_line=line_number,
                )
                if parsed.row_id in row_ids:
                    raise ChecklistCalibrationError(f"duplicate calibration row_id {parsed.row_id!r}")
                row_ids.add(parsed.row_id)
                rows.append(parsed)
        final_identity = _file_identity(path)
        if (
            final_identity["sha256"] != identity["sha256"]
            or final_identity["bytes"] != identity["bytes"]
        ):
            raise ChecklistCalibrationError(
                f"calibration JSONL changed while it was being read: {path}"
            )
    if not rows:
        raise ChecklistCalibrationError("calibration JSONL inputs contain no rows")
    return rows, identities


def _source_disjoint_split(
    rows: Sequence[CalibrationRow], config: CalibrationConfig
) -> tuple[list[CalibrationRow], list[CalibrationRow], dict[str, Any]]:
    train = [row for row in rows if row.split_id in config.train_split_ids]
    validation = [row for row in rows if row.split_id in config.validation_split_ids]
    if not train or not validation:
        raise ChecklistCalibrationError("both source-disjoint train and validation rows are required")

    def ids(items: Sequence[CalibrationRow], attribute: str) -> set[str]:
        return {str(getattr(item, attribute)) for item in items}

    train_source_ids = ids(train, "source_id")
    validation_source_ids = ids(validation, "source_id")
    train_group_ids = ids(train, "source_group_id")
    validation_group_ids = ids(validation, "source_group_id")
    train_seed_ids = ids(train, "seed_identity")
    validation_seed_ids = ids(validation, "seed_identity")
    overlapping_sources = sorted(train_source_ids.intersection(validation_source_ids))
    overlapping_groups = sorted(train_group_ids.intersection(validation_group_ids))
    overlapping_seeds = sorted(train_seed_ids.intersection(validation_seed_ids))
    if overlapping_sources:
        raise ChecklistCalibrationError(
            "source IDs overlap between train and validation: " + ", ".join(overlapping_sources[:8])
        )
    if overlapping_groups:
        raise ChecklistCalibrationError(
            "source group IDs overlap between train and validation: " + ", ".join(overlapping_groups[:8])
        )
    if overlapping_seeds:
        raise ChecklistCalibrationError(
            "seed identities overlap between train and validation: " + ", ".join(overlapping_seeds[:8])
        )
    if len(train_source_ids) < config.minimum_distinct_source_ids_per_split:
        raise ChecklistCalibrationError("train split lacks required distinct source IDs")
    if len(validation_source_ids) < config.minimum_distinct_source_ids_per_split:
        raise ChecklistCalibrationError("validation split lacks required distinct source IDs")
    if len(train_group_ids) < config.minimum_distinct_source_groups_per_split:
        raise ChecklistCalibrationError("train split lacks required distinct source-group IDs")
    if len(validation_group_ids) < config.minimum_distinct_source_groups_per_split:
        raise ChecklistCalibrationError("validation split lacks required distinct source-group IDs")

    def summary(items: Sequence[CalibrationRow], source_ids: set[str], group_ids: set[str]) -> dict[str, Any]:
        return {
            "rows": len(items),
            "row_ids_sha256": _sha256_bytes(_canonical_json_bytes(sorted(row.row_id for row in items))),
            "source_ids": len(source_ids),
            "source_ids_sha256": _sha256_bytes(_canonical_json_bytes(sorted(source_ids))),
            "source_group_ids": len(group_ids),
            "source_group_ids_sha256": _sha256_bytes(_canonical_json_bytes(sorted(group_ids))),
            "seed_identities": len({row.seed_identity for row in items}),
            "seed_identities_sha256": _sha256_bytes(
                _canonical_json_bytes(sorted({row.seed_identity for row in items}))
            ),
            "split_ids": sorted({row.split_id for row in items}),
        }

    return train, validation, {
        "source_disjoint": True,
        "source_id_overlap": [],
        "source_group_id_overlap": [],
        "seed_identity_overlap": [],
        "train": summary(train, train_source_ids, train_group_ids),
        "validation": summary(validation, validation_source_ids, validation_group_ids),
    }


def _bo250_seed_disjointness(
    rows: Sequence[CalibrationRow], excluded_seed_identities: Sequence[str]
) -> dict[str, Any]:
    """Bind calibration seed identities away from a later r289 BO250 schedule.

    The r288 calibration can exist before r289 seeds are assigned, so an empty
    exclusion list is represented explicitly as *unbound*, never as proof of
    future seed disjointness.  Once r289 gives this compiler its exact seed
    identities, even one overlap fails before an artifact is created.
    """

    observed = {row.seed_identity for row in rows}
    excluded = {
        _sha256(value, "bo250 excluded seed identity")
        for value in excluded_seed_identities
    }
    overlap = sorted(observed.intersection(excluded))
    if overlap:
        raise ChecklistCalibrationError(
            "calibration seed identities overlap future BO250 seeds: " + ", ".join(overlap[:8])
        )
    bound = bool(excluded)
    return {
        "bo250_seed_disjoint": bound,
        "bo250_seed_disjointness_bound": bound,
        "bo250_seed_identities_excluded": sorted(excluded),
        "bo250_seed_identities_excluded_sha256": _sha256_bytes(
            _canonical_json_bytes(sorted(excluded))
        ),
        "calibration_seed_identities_observed": len(observed),
        "calibration_seed_identities_observed_sha256": _sha256_bytes(
            _canonical_json_bytes(sorted(observed))
        ),
        "overlap": [],
        "unbound_artifact_may_not_authorize_bo250": not bound,
    }


def _frozen_policy_identity(rows: Sequence[CalibrationRow]) -> dict[str, Any]:
    encoded = {_canonical_json_bytes(row.frozen_neural_policy) for row in rows}
    if len(encoded) != 1:
        raise ChecklistCalibrationError("all calibration rows must use one frozen neural policy identity")
    value = json.loads(next(iter(encoded)).decode("utf-8"))
    if value.get("all_neural_model_and_checkpoint_tensors_frozen") is not True:
        raise ChecklistCalibrationError("frozen neural policy identity is not immutable")
    return value


def _overlap_provenance_summary(rows: Sequence[CalibrationRow]) -> dict[str, Any]:
    bound_receipts: dict[str, dict[str, str]] = {}
    row_bindings: list[dict[str, Any]] = []
    for row in rows:
        provenance = row.overlap_deduplication
        if provenance.get("post_deduplication_vectors") is not True:
            raise ChecklistCalibrationError("post-de-duplication row provenance drifted")
        receipt = provenance.get("overlap_audit_receipt")
        if receipt is not None:
            receipt_mapping = _mapping(receipt, "row overlap_audit_receipt")
            digest = _sha256(receipt_mapping.get("sha256"), "row overlap_audit_receipt.sha256")
            identity = _text(
                receipt_mapping.get("identity"), "row overlap_audit_receipt.identity"
            )
            bound_receipts[f"{identity}|{digest}"] = {
                "identity": identity,
                "sha256": digest,
            }
        row_bindings.append({"row_id": row.row_id, "provenance": provenance})
    return {
        "post_deduplication_vectors_required": True,
        "post_deduplication_vectors_attested_for_all_rows": True,
        "existing_learned_logic_modified": False,
        "overlap_groups": {name: list(values) for name, values in OVERLAP_GROUPS.items()},
        "row_provenance_sha256": _sha256_bytes(
            _canonical_json_bytes(sorted(row_bindings, key=lambda item: item["row_id"]))
        ),
        "actual_overlap_audit_receipts": [bound_receipts[key] for key in sorted(bound_receipts)],
        "missing_receipt_represented_by_placeholder_sha256": False,
    }


def _nonflat(vector: Sequence[float]) -> bool:
    return max((abs(value) for value in vector), default=0.0) > _ZERO_TOLERANCE


def _support_by_gate(rows: Sequence[CalibrationRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, gate in enumerate(GATES):
        supported = [row for row in rows if _nonflat(row.vectors[index])]
        result[gate] = {
            "nonflat_rows": len(supported),
            "source_ids": len({row.source_id for row in supported}),
            "source_group_ids": len({row.source_group_id for row in supported}),
        }
    return result


def _validate_gate_support(
    train: Sequence[CalibrationRow], validation: Sequence[CalibrationRow], config: CalibrationConfig
) -> dict[str, Any]:
    train_support = _support_by_gate(train)
    validation_support = _support_by_gate(validation)
    minimum = config.minimum_nonflat_rows_per_gate_per_split
    missing = [
        gate
        for gate in config.trainable_gate_names
        if train_support[gate]["nonflat_rows"] < minimum
        or validation_support[gate]["nonflat_rows"] < minimum
    ]
    if missing:
        raise ChecklistCalibrationError(
            "each eligible fitted gate requires nonflat source-disjoint support; missing "
            + ", ".join(missing)
        )
    return {
        "minimum_nonflat_rows_per_gate_per_split": minimum,
        "fitted_gate_order": list(config.trainable_gate_names),
        "fixed_trace_only_gates": dict(config.fixed_gates),
        "train": train_support,
        "validation": validation_support,
    }


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ChecklistCalibrationError("non-finite policy normalizer during calibration")
    return [value / total for value in weights]


def _residual(
    row: CalibrationRow, gates: Sequence[float], cap: float
) -> tuple[list[float], list[bool], list[tuple[int | None, ...]]]:
    """Apply the exact runtime strongest-per-group aggregation.

    Group members are ordered by the canonical channel order.  Updating only
    on a strict increase in absolute magnitude therefore gives the required
    deterministic first-channel tie break.  An all-zero group has no winner.
    """

    winner_indices: list[tuple[int | None, ...]] = []
    preclip: list[float] = []
    for option_index in range(row.width):
        option_winners: list[int | None] = []
        option_total = 0.0
        for gate_names in RESIDUAL_GROUPS.values():
            winner_index: int | None = None
            winner_contribution = 0.0
            winner_magnitude = 0.0
            for gate_name in gate_names:
                gate_index = GATES.index(gate_name)
                contribution = gates[gate_index] * row.vectors[gate_index][option_index]
                magnitude = abs(contribution)
                if magnitude > winner_magnitude:
                    winner_index = gate_index
                    winner_contribution = contribution
                    winner_magnitude = magnitude
            option_winners.append(winner_index)
            option_total += winner_contribution
        winner_indices.append(tuple(option_winners))
        preclip.append(option_total)
    residual = [max(-cap, min(cap, value)) for value in preclip]
    unclipped = [(-cap <= value <= cap) for value in preclip]
    return residual, unclipped, winner_indices


def _loss_and_gradient(
    rows: Sequence[CalibrationRow],
    gates: Sequence[float],
    *,
    cap: float,
    anchor: Sequence[float],
    l2_anchor: float,
) -> tuple[float, list[float]]:
    if not rows:
        raise ChecklistCalibrationError("cannot fit scalar gates without rows")
    loss = 0.0
    gradient = [0.0 for _ in GATES]
    for row in rows:
        residual, unclipped, winner_indices = _residual(row, gates, cap)
        policy = _softmax([base + delta for base, delta in zip(row.base_logits, residual)])
        target_probability = sum(policy[index] for index in row.target_indices)
        loss -= math.log(max(target_probability, _PROBABILITY_FLOOR))
        for option_index, probability in enumerate(policy):
            derivative = probability
            if option_index in row.target_indices:
                derivative -= probability / max(target_probability, _PROBABILITY_FLOOR)
            if not unclipped[option_index]:
                continue
            for gate_index in winner_indices[option_index]:
                if gate_index is None:
                    continue
                gradient[gate_index] += derivative * row.vectors[gate_index][option_index]
    scale = 1.0 / len(rows)
    loss *= scale
    gradient = [item * scale for item in gradient]
    if l2_anchor:
        for index in range(len(GATES)):
            distance = gates[index] - anchor[index]
            loss += 0.5 * l2_anchor * distance * distance
            gradient[index] += l2_anchor * distance
    return loss, gradient


def _metrics(rows: Sequence[CalibrationRow], gates: Sequence[float], cap: float) -> dict[str, Any]:
    if not rows:
        raise ChecklistCalibrationError("cannot score an empty split")
    nll = 0.0
    target_probability = 0.0
    correct = 0
    residual_total = 0.0
    residual_count = 0
    clipped_count = 0
    max_abs_residual = 0.0
    winner_counts = {
        group_name: {**{gate_name: 0 for gate_name in gate_names}, "none": 0}
        for group_name, gate_names in RESIDUAL_GROUPS.items()
    }
    for row in rows:
        residual, unclipped, winner_indices = _residual(row, gates, cap)
        policy = _softmax([base + delta for base, delta in zip(row.base_logits, residual)])
        group_probability = sum(policy[index] for index in row.target_indices)
        nll -= math.log(max(group_probability, _PROBABILITY_FLOOR))
        target_probability += group_probability
        best = max(policy)
        if any(abs(policy[index] - best) <= 1e-12 for index in row.target_indices):
            correct += 1
        residual_total += sum(abs(value) for value in residual)
        residual_count += len(residual)
        clipped_count += sum(1 for value in unclipped if not value)
        max_abs_residual = max(max_abs_residual, *(abs(value) for value in residual))
        for option_winners in winner_indices:
            for group_index, (group_name, _) in enumerate(RESIDUAL_GROUPS.items()):
                winner_index = option_winners[group_index]
                winner_name = "none" if winner_index is None else GATES[winner_index]
                winner_counts[group_name][winner_name] += 1
    count = len(rows)
    return {
        "rows": count,
        "group_cross_entropy": nll / count,
        "mean_target_group_probability": target_probability / count,
        "top1_target_group_accuracy": correct / count,
        "mean_absolute_residual": residual_total / max(1, residual_count),
        "maximum_absolute_residual": max_abs_residual,
        "clipped_option_fraction": clipped_count / max(1, residual_count),
        "group_winner_identity_counts": winner_counts,
    }


def _winner_identity_trace(
    rows: Sequence[CalibrationRow], gates: Sequence[float], cap: float
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        residual, unclipped, winner_indices = _residual(row, gates, cap)
        option_records: list[dict[str, Any]] = []
        for option_index, option_winners in enumerate(winner_indices):
            groups: dict[str, Any] = {}
            pre_cap = 0.0
            for group_index, group_name in enumerate(RESIDUAL_GROUPS):
                winner_index = option_winners[group_index]
                if winner_index is None:
                    groups[group_name] = {
                        "winner_gate": None,
                        "winner_channel": None,
                        "gated_contribution": 0.0,
                    }
                    continue
                gate_name = GATES[winner_index]
                contribution = gates[winner_index] * row.vectors[winner_index][option_index]
                pre_cap += contribution
                groups[group_name] = {
                    "winner_gate": gate_name,
                    "winner_channel": CHANNELS[winner_index],
                    "gated_contribution": contribution,
                }
            option_records.append(
                {
                    "option_index": option_index,
                    "groups": groups,
                    "pre_cap_residual": pre_cap,
                    "post_cap_residual": residual[option_index],
                    "global_cap_applied": not unclipped[option_index],
                }
            )
        records.append(
            {
                "row_id": row.row_id,
                "source_split_id": row.split_id,
                "options": option_records,
            }
        )
    return {
        "group_order": list(RESIDUAL_GROUPS),
        "tie_break": "runtime.channel_order_first",
        "exact_zero_group_winner": None,
        "rows": records,
        "rows_sha256": _sha256_bytes(_canonical_json_bytes(records)),
    }


def _project(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _fit_gates(
    train: Sequence[CalibrationRow],
    validation: Sequence[CalibrationRow],
    config: CalibrationConfig,
) -> dict[str, Any]:
    anchor = tuple(config.default_gates)
    gates = list(anchor)
    trainable_indices = {GATES.index(name) for name in config.trainable_gate_names}
    fixed_gates = dict(config.fixed_gates)
    for gate_name, fixed_value in fixed_gates.items():
        gates[GATES.index(gate_name)] = fixed_value
    initial_validation = _metrics(validation, gates, config.cap)["group_cross_entropy"]
    best_gates = tuple(gates)
    best_validation = float(initial_validation)
    best_step = 0
    no_improvement = 0
    last_train_loss = _loss_and_gradient(
        train,
        gates,
        cap=config.cap,
        anchor=anchor,
        l2_anchor=config.l2_anchor,
    )[0]
    completed_steps = 0
    for step in range(1, config.max_steps + 1):
        train_loss, gradient = _loss_and_gradient(
            train,
            gates,
            cap=config.cap,
            anchor=anchor,
            l2_anchor=config.l2_anchor,
        )
        learning_rate = config.learning_rate / math.sqrt(step)
        proposed = []
        for index, (gate, grad) in enumerate(zip(gates, gradient)):
            if index in trainable_indices:
                proposed.append(
                    _project(
                        gate - learning_rate * grad,
                        minimum=config.gate_minimum,
                        maximum=config.gate_maximum,
                    )
                )
            else:
                proposed.append(fixed_gates[GATES[index]])
        gates = proposed
        completed_steps = step
        last_train_loss = train_loss
        validation_loss = float(_metrics(validation, gates, config.cap)["group_cross_entropy"])
        if validation_loss < best_validation - 1e-15:
            best_validation = validation_loss
            best_gates = tuple(gates)
            best_step = step
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.early_stopping_patience:
            break
    fitted_train_loss, _ = _loss_and_gradient(
        train,
        best_gates,
        cap=config.cap,
        anchor=anchor,
        l2_anchor=config.l2_anchor,
    )
    return {
        "gates": {name: best_gates[index] for index, name in enumerate(GATES)},
        "fitted_gate_order": list(config.trainable_gate_names),
        "fixed_trace_only_gates": fixed_gates,
        "optimizer": {
            "kind": "deterministic_full_batch_projected_gradient_descent",
            "completed_steps": completed_steps,
            "best_validation_step": best_step,
            "maximum_steps": config.max_steps,
            "base_learning_rate": config.learning_rate,
            "learning_rate_schedule": "base_learning_rate/sqrt(step)",
            "piecewise_group_subgradient": "likelihood_differentiates_only_the_deterministic_current_group_winner_and_zeroes_globally_clipped_options_then_adds_the_declared_l2_anchor_to_all_trainable_gates",
            "l2_anchor_to_default": config.l2_anchor,
            "early_stopping_patience": config.early_stopping_patience,
            "last_preupdate_train_objective": last_train_loss,
            "best_train_objective": fitted_train_loss,
            "best_validation_group_cross_entropy": best_validation,
        },
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> tuple[str, int]:
    encoded = _canonical_json_bytes(dict(payload))
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise ChecklistCalibrationError(f"cannot create-only write {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # The caller gets an explicit failure and the partial create-only path
        # remains for audit rather than being silently overwritten or removed.
        raise
    return _sha256_bytes(encoded), len(encoded)


def _artifact_payload(
    *,
    config: CalibrationConfig,
    frozen_policy: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    support: Mapping[str, Any],
    input_identities: Sequence[Mapping[str, Any]],
    bo250_seed_disjointness: Mapping[str, Any],
    fit: Mapping[str, Any],
    train: Sequence[CalibrationRow],
    validation: Sequence[CalibrationRow],
) -> dict[str, Any]:
    zero_gates = tuple(0.0 for _ in GATES)
    default_gates = tuple(config.default_gates)
    fitted_gate_mapping = _mapping(fit.get("gates"), "fit.gates")
    if set(fitted_gate_mapping) != set(GATES):
        raise ChecklistCalibrationError("fitted scalar-gate inventory drifted")
    fitted_gates = tuple(
        _number(
            fitted_gate_mapping[name],
            f"fit.gates.{name}",
            minimum=config.gate_minimum,
            maximum=config.gate_maximum,
        )
        for name in GATES
    )
    for gate_name, fixed_value in config.fixed_gates:
        if fitted_gates[GATES.index(gate_name)] != fixed_value:
            raise ChecklistCalibrationError(
                f"calibration attempted to activate trace-only gate {gate_name}"
            )
    overlap_summary = _overlap_provenance_summary((*train, *validation))
    return {
        "schema": ARTIFACT_SCHEMA,
        "owner_decision_revision": 288,
        "latest_owner_clarification_revision": 295,
        "status": "offline_validation_only_not_runtime_active",
        "runtime_active": False,
        "requires_separate_receipt_backed_activation": True,
        "runtime_contract": {
            "layer_schema": RUNTIME_SCHEMA,
            "config_path": str(config.path),
            "config_file_sha256": config.file_sha256,
            "exact_new_list_multiset_sha256": config.exact_new_list_sha256,
            "corrected_guide_attachment_sha256": config.corrected_guide_attachment_sha256,
            "corrected_guide_exact_inventory": {
                "pokemon": 17,
                "trainers": 36,
                "energy": 7,
                "alakazam": 3,
            },
            "channel_order": list(CHANNELS),
            "guide_support_channel": GUIDE_CHANNEL,
            "gate_order": list(GATES),
            "fitted_gate_order": list(config.trainable_gate_names),
            "fixed_trace_only_gates": dict(config.fixed_gates),
            "gate_channel_map": {
                **{f"{channel}_gate": channel for channel in CHANNELS},
                "separate_guide_gate": GUIDE_CHANNEL,
            },
            "grouped_aggregation": {
                "kind": "strongest_absolute_gated_contribution_per_option_per_group",
                "group_order": list(RESIDUAL_GROUPS),
                "group_gate_members": {
                    name: list(gates) for name, gates in RESIDUAL_GROUPS.items()
                },
                "tie_break": "runtime.channel_order_first",
                "exact_zero_group_winner": None,
                "final_aggregation": "sum_group_winner_contributions_then_global_clamp",
            },
            "vector_stage": "centered_linf_normalized_pre_gate",
            "per_option_availability_mask_required": True,
            "aggregation": "strongest absolute gated contribution in closure plus strongest absolute gated contribution in continuity, then clamp per option",
            "total_residual_cap": config.cap,
            "gate_bounds": {"minimum": config.gate_minimum, "maximum": config.gate_maximum},
            "guide_support_is_separate_gate": True,
            "guide_support_trace_only": True,
            "guide_support_calibration_eligible": False,
            "post_deduplication_vectors_required": True,
        },
        "frozen_neural_policy": dict(frozen_policy),
        "neural_checkpoint_or_tensor_mutation": False,
        "gates": {name: fitted_gates[index] for index, name in enumerate(GATES)},
        "default_gates": {name: default_gates[index] for index, name in enumerate(GATES)},
        "zero_layer_gates": {name: zero_gates[index] for index, name in enumerate(GATES)},
        "fitted_group_winner_identity_trace": _winner_identity_trace(
            (*train, *validation), fitted_gates, config.cap
        ),
        "overlap_deduplication": overlap_summary,
        "source_disjoint_split": dict(split_summary),
        "source_disjoint_exact_new_list_data": True,
        "bo250_seed_disjointness": dict(bo250_seed_disjointness),
        "gate_support": dict(support),
        "input_files": [dict(identity) for identity in input_identities],
        "input_rows_total": len(train) + len(validation),
        "optimization": dict(_mapping(fit.get("optimizer"), "fit.optimizer")),
        "activation_requirements": {
            "runtime_activation_authority": False,
            "bo250_bound_calibration_receipt_required_before_launch": True,
            "current_receipt_is_bo250_bound": bool(
                bo250_seed_disjointness.get("bo250_seed_disjointness_bound")
            ),
            "unbound_artifact_allowed_for": ["isolated_unit_test", "shadow_trace"],
            "unbound_artifact_may_launch_bo250": False,
        },
        "metrics": {
            "train": {
                "fitted": _metrics(train, fitted_gates, config.cap),
                "default_heuristic": _metrics(train, default_gates, config.cap),
                "zero_layer": _metrics(train, zero_gates, config.cap),
            },
            "validation": {
                "fitted": _metrics(validation, fitted_gates, config.cap),
                "default_heuristic": _metrics(validation, default_gates, config.cap),
                "zero_layer": _metrics(validation, zero_gates, config.cap),
            },
        },
        "authority": {
            "production_loop": False,
            "learner_or_checkpoint_training": False,
            "selector": False,
            "submission": False,
            "elmo_production_readmission": False,
            "managed_service_change": False,
            "search": False,
            "rollout": False,
            "mcts": False,
            "rtp": False,
        },
        "artifact_sha256": None,
    }


def _receipt_payload(
    *,
    config: CalibrationConfig,
    artifact_file_name: str,
    artifact_file_sha256: str,
    artifact_file_bytes: int,
    artifact_semantic_sha256: str,
    split_summary: Mapping[str, Any],
    frozen_policy: Mapping[str, Any],
    bo250_seed_disjointness: Mapping[str, Any],
    overlap_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "owner_decision_revision": 288,
        "latest_owner_clarification_revision": 295,
        "status": "completed",
        "completion_scope": "offline_calibration_only_not_runtime_active",
        "runtime_active": False,
        "all_neural_model_and_checkpoint_tensors_frozen": True,
        "training_eligible": False,
        "production_authority": False,
        "replay_eligible": False,
        "config": {
            "path": str(config.path),
            "file_sha256": config.file_sha256,
            "exact_new_list_multiset_sha256": config.exact_new_list_sha256,
            "corrected_guide_attachment_sha256": config.corrected_guide_attachment_sha256,
        },
        "artifact": {
            "path": artifact_file_name,
            "file_sha256": artifact_file_sha256,
            "bytes": artifact_file_bytes,
            "semantic_sha256": artifact_semantic_sha256,
        },
        "frozen_neural_policy": dict(frozen_policy),
        "source_disjoint_split": dict(split_summary),
        "source_disjoint_exact_new_list_data": True,
        "overlap_deduplication": dict(overlap_summary),
        "bo250_seed_disjointness": dict(bo250_seed_disjointness),
        "bo250_seed_identities_excluded": list(
            _sequence(
                bo250_seed_disjointness.get("bo250_seed_identities_excluded"),
                "bo250_seed_disjointness.bo250_seed_identities_excluded",
            )
        ),
        "bo250_seed_disjoint": bool(bo250_seed_disjointness.get("bo250_seed_disjoint")),
        "bo250_bound_calibration_receipt_required_before_launch": True,
        "bo250_binding_satisfied": bool(
            bo250_seed_disjointness.get("bo250_seed_disjointness_bound")
        ),
        "launch_authority": False,
        "attestations": {
            "all_neural_model_and_checkpoint_tensors_frozen": True,
            "exactly_six_checklist_scalar_gates_fitted": True,
            "all_eight_checklist_channels_traced": True,
            "bench_prize_exposure_gate_trace_only_exact_zero": True,
            "immediate_disruption_outcome_gate_trace_only_exact_zero": True,
            "guide_gate_separate_trace_only_exact_zero": True,
            "guide_support_calibration_eligible": False,
            "precomputed_runtime_normalized_vectors_only": True,
            "strongest_per_group_aggregation_exact": True,
            "winner_identity_trace_bound_in_artifact": True,
            "post_deduplication_vectors_only": True,
            "existing_learned_logic_modified": False,
            "corrected_r295_guide_attachment_bound": True,
            "source_disjoint_train_validation_verified": True,
            "exact_new_list_only": True,
            "input_training_eligible": False,
            "input_production_eligible": False,
            "input_replay_eligible": False,
            "runtime_activation_authority": False,
            "production_loop_authority": False,
            "learner_or_checkpoint_training_authority": False,
            "selector_or_submission_authority": False,
            "elmo_production_readmission_authority": False,
            "managed_service_change": False,
        },
        "receipt_sha256": None,
    }


def calibrate(
    *,
    config_path: Path,
    input_paths: Sequence[Path],
    output_dir: Path,
    bo250_excluded_seed_identities: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile an inert, create-only r288 calibration artifact and receipt."""

    config = _load_config(config_path)
    rows, identities = _load_rows(input_paths, config)
    train, validation, split_summary = _source_disjoint_split(rows, config)
    frozen_policy = _frozen_policy_identity(rows)
    bo250_seed_disjointness = _bo250_seed_disjointness(rows, bo250_excluded_seed_identities)
    support = _validate_gate_support(train, validation, config)
    fit = _fit_gates(train, validation, config)
    final_config_identity = _file_identity(config.path)
    if final_config_identity["sha256"] != config.file_sha256:
        raise ChecklistCalibrationError("r288 config changed during calibration")
    artifact = _artifact_payload(
        config=config,
        frozen_policy=frozen_policy,
        split_summary=split_summary,
        support=support,
        input_identities=identities,
        bo250_seed_disjointness=bo250_seed_disjointness,
        fit=fit,
        train=train,
        validation=validation,
    )
    overlap_summary = _mapping(
        artifact.get("overlap_deduplication"), "artifact.overlap_deduplication"
    )
    artifact["artifact_sha256"] = _semantic_sha256(artifact, "artifact_sha256")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ChecklistCalibrationError(
            f"output directory already exists (create-only): {output_dir}"
        )
    if not output_dir.parent.is_dir():
        raise ChecklistCalibrationError(f"output parent directory does not exist: {output_dir.parent}")
    try:
        output_dir.mkdir(mode=0o755)
    except OSError as exc:
        raise ChecklistCalibrationError(f"cannot create output directory {output_dir}: {exc}") from exc
    artifact_path = output_dir / "alakazam-turn-checklist-r288-gates.json"
    artifact_file_sha256, artifact_file_bytes = _write_create_only(artifact_path, artifact)
    receipt = _receipt_payload(
        config=config,
        artifact_file_name=artifact_path.name,
        artifact_file_sha256=artifact_file_sha256,
        artifact_file_bytes=artifact_file_bytes,
        artifact_semantic_sha256=str(artifact["artifact_sha256"]),
        split_summary=split_summary,
        frozen_policy=frozen_policy,
        bo250_seed_disjointness=bo250_seed_disjointness,
        overlap_summary=overlap_summary,
    )
    receipt["receipt_sha256"] = _semantic_sha256(receipt, "receipt_sha256")
    receipt_path = output_dir / "alakazam-turn-checklist-r288-calibration-receipt.json"
    receipt_file_sha256, receipt_file_bytes = _write_create_only(receipt_path, receipt)
    return {
        "status": "offline_calibration_compiled_not_runtime_active",
        "output_dir": str(output_dir),
        "artifact": {
            "path": str(artifact_path),
            "file_sha256": artifact_file_sha256,
            "bytes": artifact_file_bytes,
            "semantic_sha256": artifact["artifact_sha256"],
        },
        "receipt": {
            "path": str(receipt_path),
            "file_sha256": receipt_file_sha256,
            "bytes": receipt_file_bytes,
            "semantic_sha256": receipt["receipt_sha256"],
        },
        "gates": _mapping(fit.get("gates"), "fit.gates"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="r288 default-off layer config; scalar formula and bounds are read only from it",
    )
    parser.add_argument(
        "--bo250-excluded-seed-id",
        action="append",
        default=[],
        metavar="SEED_ID",
        help=(
            "exact future r289 BO250 seed identity to exclude from calibration; "
            "repeat once per seed. Omit only for an explicitly unbound r288 artifact."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="precomputed r288 post-dedup JSONL vector capture; repeat for multiple immutable inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new create-only directory for inert artifact and receipt",
    )
    parser.add_argument(
        "--print-row-schema",
        action="store_true",
        help="print the strict JSONL field contract and exit without reading or writing artifacts",
    )
    return parser


def _row_schema_description() -> dict[str, Any]:
    return {
        "schema": ROW_SCHEMA,
        "required_fields": {
            "identity": [
                "row_id",
                "source_id",
                "source_group_id",
                "source_split_id",
                "seed_identity",
                "exact_new_list_multiset_sha256",
                "corrected_guide_attachment_sha256",
                "runtime_layer_schema",
                "runtime_config_sha256",
                "runtime_vector_stage",
                "frozen_neural_policy",
                "legal_option_order_sha256",
                "existing_route_overlap_deduplication",
            ],
            "causal_and_authority_attestations": [
                "causal_public_evidence_only=true",
                "training_eligible=false",
                "production_eligible=false",
                "replay_eligible=false",
            ],
            "policy_and_vectors": [
                "base_logits",
                "normalized_channel_vectors",
                "normalized_guide_support_vector",
                "channel_status",
                "channel_option_availability",
            ],
            "label": ["target_chosen_option XOR target_chosen_option_group"],
        },
        "channel_order": list(CHANNELS),
        "guide_support_channel": GUIDE_CHANNEL,
        "gate_order": list(GATES),
        "fitted_gate_order": list(TRAINABLE_GATES),
        "fixed_trace_only_gates": FIXED_TRACE_ONLY_GATES,
        "corrected_guide_attachment_sha256": CORRECTED_GUIDE_ATTACHMENT_SHA256,
        "overlap_groups": {name: list(values) for name, values in OVERLAP_GROUPS.items()},
        "runtime_grouped_aggregation": {
            "group_order": list(RESIDUAL_GROUPS),
            "group_gate_members": {
                name: list(gates) for name, gates in RESIDUAL_GROUPS.items()
            },
            "winner": "greatest absolute gated contribution per option and group",
            "tie_break": "runtime.channel_order_first",
            "exact_zero_group_winner": None,
            "final": "sum group winners then clamp to total_residual_cap",
        },
        "overlap_receipt": "omit until a real checksum-bound receipt exists; placeholders fail closed",
        "normalized_vector_requirements": {
            "action_order_aligned": True,
            "range": [-1.0, 1.0],
            "available_nonflat": "mean zero and max absolute value one over the true option availability mask",
            "unavailable": "all zero plus nonempty reason; option mask may be nonempty for an exactly flat answer",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_row_schema:
        print(json.dumps(_row_schema_description(), sort_keys=True, separators=(",", ":")))
        return 0
    if not args.input:
        _parser().error("at least one --input JSONL capture is required")
    if args.output_dir is None:
        _parser().error("--output-dir is required")
    try:
        result = calibrate(
            config_path=args.config,
            input_paths=list(args.input),
            output_dir=args.output_dir,
            bo250_excluded_seed_identities=list(args.bo250_excluded_seed_id),
        )
    except ChecklistCalibrationError as exc:
        print(f"r288 turn-checklist calibration failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
