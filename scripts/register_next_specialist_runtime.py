#!/usr/bin/env python3
"""Atomically register one bootstrapped specialist in the canonical runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
from poke_bot.pure_rl.deck_guide_schedule import (
    DECAY_SCHEDULE,
    MAXIMUM_AUXILIARY_WEIGHT,
    POSITIVE_RAMP_SCHEDULE,
)


AUTH_SCHEMA = "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1"
REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
GUIDE_WEIGHT_POLICY_SCHEMA = "poke_bot.current_deck_guide_weight_policy/v1"
GUIDE_WEIGHT_OWNER_DECISION_REVISION = 43
GUIDE_WEIGHT_PROSPECTIVE_SCOPE_REVISION = 44
GUIDE_WEIGHT_LEARNING_SEMANTICS_REVISION = 46
GUIDE_CURRICULUM_REVISION = 51
GUIDE_BRANCH_SCOPE_REVISION = 56
GUIDE_ACTION_INFLUENCE_REVISION = 56
DECISION_FUSION_V2_SCHEMA = "poke_bot.causal_decision_fusion/v2"
CANONICAL_LEARNED_DECISION_SOURCES = tuple(
    dict.fromkeys(
        (*DECISION_FUSION_REQUIRED_HEADS, "setup_board_outcome")
    )
)
GUIDE_TRAINING_MODE_LEGACY = "legacy_policy_ce_v1"
GUIDE_TRAINING_MODE_STRATEGIC = "strategic_curriculum_v1"
GUIDE_TRAINING_MODES = {
    GUIDE_TRAINING_MODE_LEGACY,
    GUIDE_TRAINING_MODE_STRATEGIC,
}
STRATEGIC_CURRICULUM_SCHEMA = (
    "poke_bot.future_specialist_strategic_curriculum/v1"
)
STRATEGIC_HEAD_ROLE_MAP_SCHEMA = (
    "poke_bot.future_specialist_strategic_head_roles/v1"
)
STRATEGIC_VALIDATION_SCHEMA = (
    "poke_bot.future_specialist_strategic_curriculum_validation/v1"
)
STRATEGIC_BUNDLE_SCHEMA = "poke_bot.specialist_guide_training_contract/v1"
STRATEGIC_REQUIRED_TRAINING_PATHS = [
    "supervised_bootstrap",
    "pure_rl",
    "resident_expert_rehearsal",
]
STRATEGIC_REQUIRED_CHECKS = (
    "guide_supervision_terminates_at_strategic_heads",
    "fused_policy_remains_outcome_and_win_trained",
    "direct_policy_cross_entropy_absent",
    "observed_outcome_targets_not_replaced",
    "all_curriculum_heads_have_valid_fusion_roles",
    "every_curriculum_head_has_bounded_action_scoring_route",
    "per_head_action_influence_ablation_passed",
    "exact_parent_parity_at_initialization",
    "one_option_conditioned_route_per_learned_head",
    "causal_suffix_invariance",
    "legal_option_dependence",
    "bounded_aggregate_residual",
    "all_training_paths_use_the_declared_mode",
    "active_and_historical_specialists_unchanged",
)


def _guide_weight_policy(runtime_root: Path) -> dict[str, Any]:
    """Bind a newly registered future run to the realized-win schedule."""

    module = (
        runtime_root.expanduser().resolve()
        / "poke_bot/pure_rl/deck_guide_schedule.py"
    )
    evidence_module = (
        runtime_root.expanduser().resolve()
        / "poke_bot/pure_rl/guide_weight_evidence.py"
    )
    review_module = (
        runtime_root.expanduser().resolve()
        / "poke_bot/pure_rl/guide_weight_review.py"
    )
    boundary_controller = (
        runtime_root.expanduser().resolve()
        / "scripts/apply_future_guide_weight_at_boundary.py"
    )
    shadow_pair_module = (
        runtime_root.expanduser().resolve()
        / "poke_bot/pure_rl/guide_weight_shadow_pair.py"
    )
    shadow_pair_runner = (
        runtime_root.expanduser().resolve()
        / "scripts/run_future_guide_weight_shadow_pair.py"
    )
    shadow_queue_processor = (
        runtime_root.expanduser().resolve()
        / "scripts/process_future_guide_weight_review_queue.py"
    )
    if (
        not module.is_file()
        or not evidence_module.is_file()
        or not review_module.is_file()
        or not boundary_controller.is_file()
        or not shadow_pair_module.is_file()
        or not shadow_pair_runner.is_file()
        or not shadow_queue_processor.is_file()
    ):
        raise RuntimeError("fleet guide-weight policy module is missing")
    return {
        "schema": GUIDE_WEIGHT_POLICY_SCHEMA,
        "owner_decision_revision": GUIDE_WEIGHT_OWNER_DECISION_REVISION,
        "prospective_scope_revision": GUIDE_WEIGHT_PROSPECTIVE_SCOPE_REVISION,
        "learning_semantics_revision": (
            GUIDE_WEIGHT_LEARNING_SEMANTICS_REVISION
        ),
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": GUIDE_BRANCH_SCOPE_REVISION,
        "action_influence_revision": GUIDE_ACTION_INFLUENCE_REVISION,
        "scope": "future_specialist_training_runs_only",
        "prospective_effective_specialist": "archaludon-ex",
        "retroactive_application_to_completed_frozen_or_started_runs": False,
        "historical_weight_or_receipt_rewrite_allowed": False,
        "active_teal_revision42_exception_preserved": True,
        "schedule_module": str(module),
        "schedule_module_sha256": sha256(module).removeprefix("sha256:"),
        "evidence_module": str(evidence_module),
        "evidence_module_sha256": sha256(evidence_module).removeprefix("sha256:"),
        "review_request_module": str(review_module),
        "review_request_module_sha256": sha256(review_module).removeprefix(
            "sha256:"
        ),
        "boundary_controller": str(boundary_controller),
        "boundary_controller_sha256": sha256(
            boundary_controller
        ).removeprefix("sha256:"),
        "boundary_receipt_schema": (
            "poke_bot.future_specialist_guide_weight_boundary/v1"
        ),
        "shadow_pair_module": str(shadow_pair_module),
        "shadow_pair_module_sha256": sha256(shadow_pair_module).removeprefix(
            "sha256:"
        ),
        "shadow_pair_runner": str(shadow_pair_runner),
        "shadow_pair_runner_sha256": sha256(shadow_pair_runner).removeprefix(
            "sha256:"
        ),
        "shadow_queue_processor": str(shadow_queue_processor),
        "shadow_queue_processor_sha256": sha256(
            shadow_queue_processor
        ).removeprefix("sha256:"),
        "shadow_pair_manifest_schema": (
            "poke_bot.future_guide_weight_shadow_pair/v1"
        ),
        "review_request_schema": (
            "poke_bot.current_deck_guide_weight_review_request/v1"
        ),
        "automatic_review_after_each_five_iteration_commit": True,
        "learning_effect": (
            "literal_multiplier_on_bounded_guide_conditioned_"
            "strategic_head_curriculum"
        ),
        "multiplier_applied_before_backpropagation": True,
        "gradient_effect": (
            "scales_guide_conditioned_strategic_head_gradient_contribution"
        ),
        "training_target_mode": "bounded_strategic_head_curriculum",
        "direct_policy_cross_entropy_allowed": False,
        "activation_requires_prestage_validation_receipt": True,
        "allowed_fusion_roles": ["fused_input"],
        "every_head_declares_fusion_role": True,
        "allowed_computation_roles": ["independent_head"],
        "every_head_declares_computation_role": True,
        "independent_pre_fusion_branches_trainable": True,
        "action_influence": "bounded_option_conditioned_route",
        "causal_input": "board_state_and_legal_option",
        "direct_action_selection_authority": False,
        "runtime_activation_requirement": "receipt_backed_validation",
        "per_head_action_logit_ablation_required": True,
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "preserve_v1_additive_residual": True,
        "canonical_learned_decision_sources": list(
            CANONICAL_LEARNED_DECISION_SOURCES
        ),
        "one_route_per_learned_source": True,
        "route_input": "option_hidden_plus_typed_output",
        "route_reduction": "fixed_mean",
        "aggregate_absolute_logit_cap": 1.0,
        "zero_safe_final_projection": True,
        "guide_is_only_action_route_exception": True,
        "elapsed_time_only_progression_allowed": False,
        "guide_head_only_bookkeeping_allowed": False,
        "dashboard_only_or_runtime_action_bias_allowed": False,
        "evidence_receipt_schema": (
            "poke_bot.current_deck_guide_paired_evaluation/v1"
        ),
        "schedule_receipt_schema": (
            "poke_bot.current_deck_guide_weight_schedule/v1"
        ),
        "bootstrap_ramp": [0.01, 0.05],
        "post_bootstrap_positive_ramp_steps": list(POSITIVE_RAMP_SCHEDULE),
        "maximum_auxiliary_loss_weight": MAXIMUM_AUXILIARY_WEIGHT,
        "review_every_completed_iterations": 5,
        "minimum_paired_games_per_review": 1000,
        "minimum_paired_games_per_matchup": 50,
        "realized_win_delta_confidence_level": 0.90,
        "positive_ramp_evidence": (
            "training_ineligible_paired_guide_on_guide_off_"
            "realized_win_delta_lower_confidence_bound_above_zero"
        ),
        "decay_after_consecutive_nonpositive_reviews": 2,
        "decay_steps": list(DECAY_SCHEDULE),
        "clean_boundary_schedule_receipt_required": True,
        "schedule_records_earliest_eligible_next_iteration": True,
        "application_boundary": (
            "first_available_future_five_iteration_hard_pause"
        ),
        "actual_application_iteration_recorded_only_by_boundary_receipt": True,
        "consecutive_nonpositive_evaluations": 0,
        "training_replay_and_formal_gate_tuning_allowed": False,
        "counterfactual_checkpoint_serving_allowed": False,
        "counterfactual_checkpoint_promotion_allowed": False,
        "runtime_action_override_allowed": False,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _normalized_sha256(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw.removeprefix("sha256:")
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise RuntimeError("invalid sha256 digest")
    return raw


def _checked_json_file(
    path: Path | None,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    if path is None:
        raise RuntimeError(f"strategic curriculum lacks {label}")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"strategic curriculum input is missing: {resolved}")
    expected = _normalized_sha256(expected_sha256)
    actual = sha256(resolved).removeprefix("sha256:")
    if actual != expected:
        raise RuntimeError(
            f"strategic curriculum digest mismatch: {label} "
            f"expected={expected} actual={actual}"
        )
    return resolved, _read(resolved), actual


def _validate_head_role_map(
    payload: dict[str, Any],
    *,
    specialist_id: str,
    guide_contract_sha256: str,
) -> tuple[str, ...]:
    heads = payload.get("heads")
    try:
        aggregate_cap = float(payload.get("aggregate_absolute_logit_cap"))
    except (TypeError, ValueError):
        aggregate_cap = float("nan")
    if (
        payload.get("schema") != STRATEGIC_HEAD_ROLE_MAP_SCHEMA
        or payload.get("specialist_id") != specialist_id
        or payload.get("training_mode") != GUIDE_TRAINING_MODE_STRATEGIC
        or int(payload.get("guide_curriculum_revision") or 0)
        != GUIDE_CURRICULUM_REVISION
        or int(payload.get("strategic_branch_scope_revision") or 0)
        != GUIDE_BRANCH_SCOPE_REVISION
        or int(payload.get("action_influence_revision") or 0)
        != GUIDE_ACTION_INFLUENCE_REVISION
        or payload.get("guide_contract_sha256")
        != f"sha256:{guide_contract_sha256}"
        or payload.get("decision_fusion_schema")
        != DECISION_FUSION_V2_SCHEMA
        or payload.get("preserve_v1_additive_residual") is not True
        or payload.get("canonical_learned_decision_sources")
        != list(CANONICAL_LEARNED_DECISION_SOURCES)
        or payload.get("one_route_per_learned_source") is not True
        or payload.get("route_input")
        != "option_hidden_plus_typed_output"
        or payload.get("route_reduction") != "fixed_mean"
        or aggregate_cap != 1.0
        or payload.get("zero_safe_final_projection") is not True
        or payload.get("guide_is_only_action_route_exception") is not True
        or not isinstance(heads, dict)
        or set(heads) != set(CANONICAL_LEARNED_DECISION_SOURCES)
    ):
        raise RuntimeError("strategic curriculum head-role map is invalid")

    seen_action_route = False
    route_ids: set[str] = set()
    for head_id, raw in heads.items():
        if (
            not isinstance(head_id, str)
            or not head_id
            or not all(
                char.islower()
                or char.isdigit()
                or char in {"_", "-"}
                for char in head_id
            )
            or not isinstance(raw, dict)
        ):
            raise RuntimeError("strategic curriculum head-role map is invalid")
        role = raw.get("fusion_role")
        try:
            maximum_contribution = float(
                raw.get("maximum_absolute_logit_contribution")
            )
        except (TypeError, ValueError):
            maximum_contribution = float("nan")
        if (
            role != "fused_input"
            or raw.get("computation_role") != "independent_head"
            or raw.get("trainable") is not True
            or raw.get("causal_training_targets_only") is not True
            or raw.get("guide_action_target_allowed") is not False
            or raw.get("action_influence")
            != "bounded_option_conditioned_route"
            or raw.get("causal_input") != "board_state_and_legal_option"
            or raw.get("direct_action_selection_authority") is not False
            or raw.get("runtime_activation_requirement")
            != "receipt_backed_validation"
            or raw.get("route_input")
            != "option_hidden_plus_typed_output"
            or raw.get("route_reduction") != "fixed_mean"
            or raw.get("zero_safe_final_projection") is not True
            or not math.isfinite(maximum_contribution)
            or maximum_contribution <= 0.0
            or maximum_contribution > 1.0
        ):
            raise RuntimeError(
                f"strategic curriculum head role is invalid: {head_id}"
            )
        if (
            head_id not in CANONICAL_LEARNED_DECISION_SOURCES
            or raw.get("enters_decision_fusion") is not True
        ):
            raise RuntimeError(
                f"strategic fused-input head is invalid: {head_id}"
            )
        route_id = str(raw.get("route_id") or "")
        if (
            not route_id
            or not all(
                char.islower()
                or char.isdigit()
                or char in {"_", "-"}
                for char in route_id
            )
            or route_id in route_ids
        ):
            raise RuntimeError(
                f"strategic action route identity is invalid: {head_id}"
            )
        route_ids.add(route_id)
        seen_action_route = True
    if not seen_action_route:
        raise RuntimeError("strategic curriculum lacks an action-scoring route")
    return tuple(sorted(heads))


def _validate_strategic_curriculum_bundle(
    *,
    specialist_id: str,
    guide_contract_sha256: str,
    curriculum_spec: Path | None,
    curriculum_spec_sha256: str,
    head_role_map: Path | None,
    head_role_map_sha256: str,
    validation_receipt: Path | None,
    validation_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate the future-only curriculum before a runtime row can exist."""

    role_path, role_payload, role_digest = _checked_json_file(
        head_role_map,
        head_role_map_sha256,
        label="head-role map",
    )
    head_ids = _validate_head_role_map(
        role_payload,
        specialist_id=specialist_id,
        guide_contract_sha256=guide_contract_sha256,
    )
    spec_path, spec, spec_digest = _checked_json_file(
        curriculum_spec,
        curriculum_spec_sha256,
        label="curriculum spec",
    )
    exact_spec = {
        "schema": STRATEGIC_CURRICULUM_SCHEMA,
        "specialist_id": specialist_id,
        "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": GUIDE_BRANCH_SCOPE_REVISION,
        "action_influence_revision": GUIDE_ACTION_INFLUENCE_REVISION,
        "guide_contract_sha256": f"sha256:{guide_contract_sha256}",
        "head_role_map_sha256": f"sha256:{role_digest}",
        "curriculum_heads": list(head_ids),
        "guide_targets": "observed_causal_strategic_heads_only",
        "direct_policy_cross_entropy_allowed": False,
        "guide_runtime_input_allowed": False,
        "guide_action_selection_allowed": False,
        "replace_observed_outcome_targets_allowed": False,
        "all_curriculum_heads_must_influence_actions": True,
        "computation_role": "independent_head",
        "fusion_role": "fused_input",
        "action_influence": "bounded_option_conditioned_route",
        "causal_input": "board_state_and_legal_option",
        "direct_action_selection_authority": False,
        "runtime_activation_requirement": "receipt_backed_validation",
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "preserve_v1_additive_residual": True,
        "canonical_learned_decision_sources": list(
            CANONICAL_LEARNED_DECISION_SOURCES
        ),
        "one_route_per_learned_source": True,
        "route_input": "option_hidden_plus_typed_output",
        "route_reduction": "fixed_mean",
        "aggregate_absolute_logit_cap": 1.0,
        "zero_safe_final_projection": True,
        "guide_is_only_action_route_exception": True,
        "pre_fleet_h10_compute_allowed": False,
    }
    if any(spec.get(key) != value for key, value in exact_spec.items()):
        raise RuntimeError("strategic curriculum spec is invalid")

    receipt_path, receipt, receipt_digest = _checked_json_file(
        validation_receipt,
        validation_receipt_sha256,
        label="validation receipt",
    )
    exact_receipt = {
        "schema": STRATEGIC_VALIDATION_SCHEMA,
        "status": "validated",
        "specialist_id": specialist_id,
        "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": GUIDE_BRANCH_SCOPE_REVISION,
        "action_influence_revision": GUIDE_ACTION_INFLUENCE_REVISION,
        "guide_contract_sha256": f"sha256:{guide_contract_sha256}",
        "curriculum_spec_sha256": f"sha256:{spec_digest}",
        "head_role_map_sha256": f"sha256:{role_digest}",
        "required_training_paths": STRATEGIC_REQUIRED_TRAINING_PATHS,
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "validated_route_ids": [
            str(role_payload["heads"][head_id]["route_id"])
            for head_id in head_ids
        ],
    }
    checks = receipt.get("checks")
    measurements = receipt.get("measurements")
    action_ablations = receipt.get("action_influence_ablations")
    route_validation = receipt.get("route_validation")
    implementation_artifacts = receipt.get("implementation_artifacts")
    try:
        strategic_gradient_norm = float(
            dict(measurements or {}).get(
                "guide_to_strategic_head_gradient_norm",
                float("nan"),
            )
        )
        guide_labeled_rows = int(
            dict(measurements or {}).get("guide_labeled_rows") or 0
        )
        aggregate_residual = float(
            dict(measurements or {}).get(
                "maximum_absolute_aggregate_residual_observed",
                float("nan"),
            )
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            "strategic curriculum validation receipt is invalid"
        ) from None
    if (
        any(receipt.get(key) != value for key, value in exact_receipt.items())
        or not isinstance(checks, dict)
        or any(checks.get(key) is not True for key in STRATEGIC_REQUIRED_CHECKS)
        or not isinstance(measurements, dict)
        or not math.isfinite(strategic_gradient_norm)
        or strategic_gradient_norm <= 0.0
        or guide_labeled_rows <= 0
        or not math.isfinite(aggregate_residual)
        or aggregate_residual < 0.0
        or aggregate_residual > 1.0
        or not isinstance(action_ablations, dict)
        or set(action_ablations) != set(head_ids)
        or not isinstance(route_validation, dict)
        or set(route_validation) != set(head_ids)
        or not isinstance(implementation_artifacts, list)
        or len(implementation_artifacts) < 2
    ):
        raise RuntimeError("strategic curriculum validation receipt is invalid")
    for head_id in head_ids:
        ablation = action_ablations.get(head_id)
        route = route_validation.get(head_id)
        try:
            decisions = int(dict(ablation or {}).get("decisions_evaluated") or 0)
            mean_delta = float(
                dict(ablation or {}).get(
                    "mean_absolute_action_logit_delta",
                    float("nan"),
                )
            )
            observed_maximum = float(
                dict(ablation or {}).get(
                    "maximum_absolute_logit_contribution_observed",
                    float("nan"),
                )
            )
            selection_change_rate = float(
                dict(ablation or {}).get(
                    "selection_change_rate",
                    float("nan"),
                )
            )
            declared_maximum = float(
                dict(role_payload["heads"][head_id])[
                    "maximum_absolute_logit_contribution"
                ]
            )
            route_gradient_norm = float(
                dict(route or {}).get(
                    "post_training_route_gradient_norm",
                    float("nan"),
                )
            )
            legal_option_delta = float(
                dict(route or {}).get(
                    "legal_option_dependence_delta",
                    float("nan"),
                )
            )
            suffix_delta = float(
                dict(route or {}).get(
                    "causal_suffix_max_logit_delta",
                    float("nan"),
                )
            )
            step_zero_delta = float(
                dict(route or {}).get(
                    "zero_safe_initial_max_logit_delta",
                    float("nan"),
                )
            )
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                "strategic curriculum action-influence ablation is invalid"
            ) from None
        if (
            decisions <= 0
            or not math.isfinite(mean_delta)
            or mean_delta <= 0.0
            or not math.isfinite(observed_maximum)
            or observed_maximum <= 0.0
            or observed_maximum > declared_maximum
            or not math.isfinite(selection_change_rate)
            or selection_change_rate < 0.0
            or selection_change_rate > 1.0
            or dict(route or {}).get("route_id")
            != role_payload["heads"][head_id]["route_id"]
            or dict(route or {}).get("route_input")
            != "option_hidden_plus_typed_output"
            or not math.isfinite(route_gradient_norm)
            or route_gradient_norm <= 0.0
            or not math.isfinite(legal_option_delta)
            or legal_option_delta <= 0.0
            or suffix_delta != 0.0
            or step_zero_delta != 0.0
        ):
            raise RuntimeError(
                "strategic curriculum action-influence ablation is invalid"
            )

    artifact_roles: set[str] = set()
    for artifact in implementation_artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError(
                "strategic curriculum implementation artifact is invalid"
            )
        artifact_path = Path(str(artifact.get("path") or "")).expanduser()
        artifact_digest = str(artifact.get("sha256") or "")
        role = str(artifact.get("role") or "")
        if (
            role not in {"training_implementation", "validation_test"}
            or not artifact_path.is_absolute()
        ):
            raise RuntimeError(
                "strategic curriculum implementation artifact role is invalid"
            )
        _checked_json_file_or_binary(
            artifact_path,
            artifact_digest,
            label=f"implementation artifact {role}",
        )
        artifact_roles.add(role)
    if artifact_roles != {"training_implementation", "validation_test"}:
        raise RuntimeError(
            "strategic curriculum implementation evidence is incomplete"
        )

    return {
        "schema": STRATEGIC_BUNDLE_SCHEMA,
        "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": GUIDE_BRANCH_SCOPE_REVISION,
        "action_influence_revision": GUIDE_ACTION_INFLUENCE_REVISION,
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "curriculum_spec": str(spec_path),
        "curriculum_spec_sha256": spec_digest,
        "head_role_map": str(role_path),
        "head_role_map_sha256": role_digest,
        "validation_receipt": str(receipt_path),
        "validation_receipt_sha256": receipt_digest,
    }


def _checked_json_file_or_binary(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"strategic curriculum input is missing: {resolved}")
    expected = _normalized_sha256(expected_sha256)
    actual = sha256(resolved).removeprefix("sha256:")
    if actual != expected:
        raise RuntimeError(
            f"strategic curriculum digest mismatch: {label} "
            f"expected={expected} actual={actual}"
        )
    return resolved


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_selector(
    path: Path,
    specialist_id: str,
    runtime_root: Path,
    required_fusion_heads: tuple[str, ...],
) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "POKEBOT_ACTIVE_SPECIALIST=": specialist_id,
        "POKEBOT_SPECIALIST_RUNTIME_ROOT=": str(runtime_root),
        "PYTHONPATH=": str(runtime_root),
        "POKEBOT_EXPANDED_HEADS_ENABLED=": "1",
        "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=": (
            "1"
            if "setup_board_outcome" in required_fusion_heads
            else "0"
        ),
        "POKEBOT_COMBO_STATE_HEAD_ENABLED=": (
            "1" if "combo_state" in required_fusion_heads else "0"
        ),
        "POKEBOT_DECISION_FUSION_ENABLED=": "1",
        "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=": "1",
        "POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION=": "44",
        "POKEBOT_GUIDE_LEARNING_SEMANTICS_REVISION=": "46",
        "POKEBOT_GUIDE_CONSECUTIVE_NONPOSITIVE_EVALUATIONS=": "0",
    }
    replaced = {key: False for key in replacements}
    output: list[str] = []
    for row in rows:
        if row.startswith("EXPANDED_HEADS_ENABLED="):
            # Retire the unscoped spelling; project config reads the
            # POKEBOT_-prefixed variable.
            continue
        matched = next(
            (key for key in replacements if row.startswith(key)),
            None,
        )
        if matched is None:
            output.append(row)
            continue
        if replaced[matched]:
            continue
        output.append(matched + replacements[matched])
        replaced[matched] = True
    for key, value in replacements.items():
        if not replaced[key]:
            output.append(key + value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def register(
    *,
    specialist_id: str,
    family: Path,
    expert: Path,
    runtime_tree: Path,
    runtime_registry: Path,
    selector_env: Path,
    state_root: Path,
    run_name: str,
    handoff_service: str,
    minimum_decisions: int = 20_000,
    required_target_coverage: tuple[str, ...] = (),
    guide_id: str | None = None,
    guide_loss_weight: float = 0.0,
    guide_contract: Path | None = None,
    guide_contract_sha256: str = "",
    guide_version: str = "",
    guide_training_mode: str = "",
    strategic_curriculum_spec: Path | None = None,
    strategic_curriculum_spec_sha256: str = "",
    strategic_head_role_map: Path | None = None,
    strategic_head_role_map_sha256: str = "",
    strategic_validation_receipt: Path | None = None,
    strategic_validation_receipt_sha256: str = "",
    authorization_name: str = "",
    replace_unpassed: bool = False,
    matchup_target_ids: tuple[str, ...] | None = None,
    decision_fusion_required_heads: tuple[str, ...] | None = None,
    receipt_suffix: str = "",
) -> dict[str, Any]:
    specialist_id = specialist_id.strip().lower()
    if not specialist_id or not all(
        char.islower() or char.isdigit() or char == "-" for char in specialist_id
    ):
        raise ValueError("unsafe specialist id")
    frozen = verify_frozen_model(family.expanduser().resolve())
    checkpoint_path = Path(str(frozen["model_path"])).resolve()
    checkpoint_digest = str(frozen["checkpoint_digest"])
    expert = expert.expanduser().resolve()
    runtime_tree = runtime_tree.expanduser().resolve()
    runtime_registry = runtime_registry.expanduser().resolve()
    selector_env = selector_env.expanduser().resolve()
    state_root = state_root.expanduser().resolve()
    registry = _read(runtime_registry)
    specialists = dict(registry.get("specialists") or {})
    existing_row = specialists.get(specialist_id)
    existing_legacy_row = bool(
        isinstance(existing_row, dict)
        and existing_row.get("guide_weight_policy") is None
        and existing_row.get("guide_training_mode") is None
    )
    guide_id = str(guide_id or "").strip().casefold() or None
    guide_loss_weight = float(guide_loss_weight)
    resolved_guide_contract = (
        guide_contract.expanduser().resolve()
        if guide_contract is not None
        else None
    )
    if guide_loss_weight > 0.0 and (
        guide_id != specialist_id
        or resolved_guide_contract is None
        or not resolved_guide_contract.is_file()
        or sha256(resolved_guide_contract) != str(guide_contract_sha256)
        or not str(guide_version).strip()
    ):
        raise RuntimeError("next specialist guide registration contract failed")
    if guide_loss_weight < 0.0 or guide_loss_weight > 0.05:
        raise RuntimeError("next specialist guide loss weight is invalid")
    requested_training_mode = str(guide_training_mode or "").strip()
    if requested_training_mode:
        if requested_training_mode not in GUIDE_TRAINING_MODES:
            raise RuntimeError("unknown guide training mode")
        resolved_training_mode = requested_training_mode
    elif existing_legacy_row or guide_loss_weight == 0.0:
        resolved_training_mode = GUIDE_TRAINING_MODE_LEGACY
    else:
        raise RuntimeError(
            "new guided specialist requires an explicit guide training mode"
        )

    supplied_strategic_inputs = any(
        (
            strategic_curriculum_spec is not None,
            bool(strategic_curriculum_spec_sha256),
            strategic_head_role_map is not None,
            bool(strategic_head_role_map_sha256),
            strategic_validation_receipt is not None,
            bool(strategic_validation_receipt_sha256),
        )
    )
    guide_contract_digest = (
        _normalized_sha256(guide_contract_sha256)
        if resolved_guide_contract is not None
        else ""
    )
    if resolved_training_mode == GUIDE_TRAINING_MODE_STRATEGIC:
        if guide_loss_weight <= 0.0 or resolved_guide_contract is None:
            raise RuntimeError(
                "strategic curriculum requires a positive checksum-bound guide"
            )
        strategic_curriculum = _validate_strategic_curriculum_bundle(
            specialist_id=specialist_id,
            guide_contract_sha256=guide_contract_digest,
            curriculum_spec=strategic_curriculum_spec,
            curriculum_spec_sha256=strategic_curriculum_spec_sha256,
            head_role_map=strategic_head_role_map,
            head_role_map_sha256=strategic_head_role_map_sha256,
            validation_receipt=strategic_validation_receipt,
            validation_receipt_sha256=strategic_validation_receipt_sha256,
        )
    else:
        if supplied_strategic_inputs:
            raise RuntimeError(
                "legacy guide training cannot bind strategic curriculum inputs"
            )
        if (
            guide_loss_weight > 0.0
            and specialist_id != "teal-mask-ogerpon-ex"
            and not existing_legacy_row
        ):
            raise RuntimeError(
                "new future specialists cannot use legacy policy imitation"
            )
        strategic_curriculum = None

    runtime_root = Path(str(registry.get("runtime_root") or "")).expanduser()
    tree = _read(runtime_tree)
    pointer = _read(expert)
    targets = tuple(str(value) for value in tree.get("targets") or ())
    expected_targets = (
        tuple(str(value) for value in matchup_target_ids)
        if matchup_target_ids is not None
        else tuple(EXPERT_IDS)
    )
    accepted = {
        str(value)
        for value in (tree.get("runtime_contract") or {}).get(
            "accepted_archetype_ids", ()
        )
    }
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or not runtime_root.is_absolute()
        or not runtime_root.is_dir()
        or pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or int((pointer.get("totals") or {}).get("decisions_kept") or 0)
        < int(minimum_decisions)
        or tree.get("runtime_enabled") is not True
        or not expected_targets
        or len(set(expected_targets)) != len(expected_targets)
        or targets != expected_targets
        or len(set(targets)) != len(expected_targets)
        or specialist_id not in targets
        or specialist_id not in accepted
        or (tree.get("runtime_contract") or {}).get("one_route_per_decision")
        is not True
        or (tree.get("runtime_contract") or {}).get("unknown_route_exact_bypass")
        is not True
        or not selector_env.is_file()
        or not handoff_service.startswith("pokebot-")
        or not handoff_service.endswith(".service")
    ):
        raise RuntimeError("next specialist runtime input contract failed")

    manifest = family.resolve() / "manifest.json"
    timestamp = datetime.now(timezone.utc).isoformat()
    authorization_filename = (
        authorization_name.strip()
        or f"{specialist_id}-matchup-adapter-bootstrap-v1.json"
    )
    if (
        Path(authorization_filename).name != authorization_filename
        or not authorization_filename.endswith(".json")
    ):
        raise ValueError("unsafe adapter authorization name")
    authorization_path = state_root / authorization_filename
    authorization = {
        "schema": AUTH_SCHEMA,
        "specialist_id": specialist_id,
        "completed_iteration": -1,
        "first_eligible_iteration": 0,
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_digest": checkpoint_digest,
        "protected_manifest": str(manifest),
        "protected_manifest_digest": sha256(manifest),
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "parent_untouched": True,
        "purpose": "specialist-bootstrap-causal-router-aligned-adapter-fitting",
        "required_target_coverage": list(required_target_coverage),
        "created_at_utc": timestamp,
    }
    if authorization_path.is_file():
        if _read(authorization_path) != authorization:
            # The timestamp is evidence, not identity. Permit an idempotent
            # replay only when every substantive field is unchanged.
            existing = _read(authorization_path)
            existing_semantic = {
                key: value
                for key, value in existing.items()
                if key != "created_at_utc"
            }
            expected_semantic = {
                key: value
                for key, value in authorization.items()
                if key != "created_at_utc"
            }
            legacy_expected = dict(expected_semantic)
            legacy_expected.pop("required_target_coverage")
            if existing_semantic == legacy_expected:
                authorization["created_at_utc"] = str(
                    existing.get("created_at_utc") or timestamp
                )
                _atomic_json(authorization_path, authorization)
            elif existing_semantic != expected_semantic:
                raise RuntimeError(
                    "existing adapter authorization identity changed"
                )
            else:
                authorization = existing
    else:
        _atomic_json(authorization_path, authorization)

    guide_weight_policy = (
        _guide_weight_policy(runtime_root)
        if resolved_training_mode == GUIDE_TRAINING_MODE_STRATEGIC
        else None
    )
    strategic_decision_fusion = (
        resolved_training_mode == GUIDE_TRAINING_MODE_STRATEGIC
    )
    resolved_fusion_heads = tuple(
        decision_fusion_required_heads
        or (
            CANONICAL_LEARNED_DECISION_SOURCES
            if strategic_decision_fusion
            else DECISION_FUSION_REQUIRED_HEADS
        )
    )
    canonical_fusion_prefix = tuple(
        CANONICAL_LEARNED_DECISION_SOURCES
        if strategic_decision_fusion
        else DECISION_FUSION_REQUIRED_HEADS
    )
    if (
        resolved_fusion_heads[: len(canonical_fusion_prefix)]
        != canonical_fusion_prefix
        or resolved_fusion_heads[len(canonical_fusion_prefix) :]
        not in {(), ("combo_state",)}
        or len(resolved_fusion_heads) != len(set(resolved_fusion_heads))
    ):
        raise RuntimeError("decision-fusion runtime head contract changed")
    if receipt_suffix not in {"", "-combo-v2"}:
        raise RuntimeError("runtime registration receipt suffix changed")
    row = {
        "status": "ready",
        "reason": None,
        "run_name": run_name,
        "log": f"/home/inzi/poke-bot-agent/outputs/logs/{run_name}.log",
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_sha256": checkpoint_digest.removeprefix("sha256:"),
        "expert_manifest": str(expert),
        "expert_manifest_sha256": sha256(expert).removeprefix("sha256:"),
        "expert_minimum_decisions": int(minimum_decisions),
        "expert_required_target_coverage": list(required_target_coverage),
        "matchup_runtime_tree": str(runtime_tree),
        "matchup_runtime_tree_sha256": sha256(runtime_tree).removeprefix("sha256:"),
        "matchup_adapter_authorization": str(authorization_path),
        "matchup_adapter_authorization_sha256": sha256(
            authorization_path
        ).removeprefix("sha256:"),
        "matchup_adapter_epochs_per_rl_iteration": 1,
        "measurement_decks": specialist_id,
        "decision_fusion": {
            "schema": (
                DECISION_FUSION_V2_SCHEMA
                if strategic_decision_fusion
                else DECISION_FUSION_SCHEMA
            ),
            "required": True,
            "runtime_enabled": True,
            "required_heads": list(resolved_fusion_heads),
        },
        "guide_id": guide_id,
        "guide_loss_weight": guide_loss_weight,
        "guide_contract": (
            str(resolved_guide_contract)
            if resolved_guide_contract is not None
            else None
        ),
        "guide_contract_sha256": (
            str(guide_contract_sha256).removeprefix("sha256:")
            if resolved_guide_contract
            else None
        ),
        "guide_version": str(guide_version).strip() or None,
        "guide_training_mode": resolved_training_mode,
        "strategic_curriculum": strategic_curriculum,
        "guide_weight_policy": guide_weight_policy,
        "terminal_gate_marker": f"SPECIALIST_GATE_PASSED.{specialist_id}-splus-v1",
        "pass_handler": {
            "family": f"{specialist_id}-protocol-gate-pass-v1",
            "display_name": f"{specialist_id} Exact Protocol Gate Champion",
            "submission_root": (
                "/home/inzi/poke-bot-agent/outputs/submissions/"
                f"{specialist_id}-protocol-gate-pass-v1"
            ),
            "state": (
                "/home/inzi/poke-bot-agent/outputs/state/"
                f"{specialist_id}-passed-gate-handler-v1.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                f"{specialist_id}-passed-gate-handler-v1.lock"
            ),
            "handoff_service": handoff_service,
        },
    }
    if existing_row is not None and existing_row != row:
        legacy_row = dict(row)
        legacy_row.pop("expert_minimum_decisions")
        legacy_row.pop("expert_required_target_coverage")
        legacy_guide_policy_row = dict(row)
        legacy_guide_policy_row.pop("guide_weight_policy")
        immutable_legacy_mode_row = dict(row)
        immutable_legacy_mode_row.pop("guide_training_mode")
        immutable_legacy_mode_row.pop("strategic_curriculum")
        immutable_pre_policy_row = dict(immutable_legacy_mode_row)
        immutable_pre_policy_row.pop("guide_weight_policy")
        authorization_migration_row = dict(row)
        authorization_migration_row[
            "matchup_adapter_authorization_sha256"
        ] = str(
            dict(existing_row).get(
                "matchup_adapter_authorization_sha256"
            )
            or ""
        )
        if (
            existing_row != legacy_row
            and existing_row != legacy_guide_policy_row
            and existing_row != immutable_legacy_mode_row
            and existing_row != immutable_pre_policy_row
            and existing_row != authorization_migration_row
        ):
            if not replace_unpassed:
                raise RuntimeError(
                    "refusing to replace a different specialist runtime row"
                )
            pass_state = Path(
                str(
                    dict(dict(existing_row).get("pass_handler") or {}).get(
                        "state"
                    )
                    or ""
                )
            )
            if (
                dict(existing_row).get("status") not in {"ready", "failed"}
                or (
                    pass_state.is_file()
                    and str(_read(pass_state).get("status") or "").casefold()
                    in {"passed", "complete", "frozen", "submitted"}
                )
            ):
                raise RuntimeError(
                    "refusing to replace a passing or frozen specialist runtime row"
                )
        elif existing_row in (
            immutable_legacy_mode_row,
            immutable_pre_policy_row,
        ):
            # Do not retrofit the future curriculum contract into an immutable
            # Teal/historical runtime row.
            row = dict(existing_row)
    specialists[specialist_id] = row
    registry["specialists"] = specialists
    _atomic_json(runtime_registry, registry)
    _atomic_selector(
        selector_env,
        specialist_id,
        runtime_root.resolve(),
        resolved_fusion_heads,
    )
    receipt = {
        "schema": "poke_bot.specialist_runtime_registration/v1",
        "created_at_utc": timestamp,
        "specialist_id": specialist_id,
        "runtime_row": row,
        "runtime_registry": str(runtime_registry),
        "runtime_registry_sha256": sha256(runtime_registry),
        "selector_env": str(selector_env),
        "selector_env_sha256": sha256(selector_env),
    }
    encoded = json.dumps(
        {key: value for key, value in receipt.items() if key != "created_at_utc"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt["identity_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    receipt_path = state_root / (
        f"{specialist_id}-runtime-registration-v1{receipt_suffix}.json"
    )
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--runtime-tree", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--selector-env", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--minimum-decisions", type=int, default=20_000)
    parser.add_argument("--required-target", action="append", default=[])
    parser.add_argument(
        "--handoff-service",
        default="pokebot-specialist-cycle-handoff.service",
    )
    parser.add_argument("--authorization-name", default="")
    parser.add_argument("--guide-id", default="")
    parser.add_argument("--guide-loss-weight", type=float, default=0.0)
    parser.add_argument("--guide-contract", type=Path)
    parser.add_argument("--guide-contract-sha256", default="")
    parser.add_argument("--guide-version", default="")
    parser.add_argument("--guide-training-mode", default="")
    parser.add_argument("--strategic-curriculum-spec", type=Path)
    parser.add_argument("--strategic-curriculum-spec-sha256", default="")
    parser.add_argument("--strategic-head-role-map", type=Path)
    parser.add_argument("--strategic-head-role-map-sha256", default="")
    parser.add_argument("--strategic-validation-receipt", type=Path)
    parser.add_argument("--strategic-validation-receipt-sha256", default="")
    parser.add_argument(
        "--replace-unpassed",
        action="store_true",
        help="Replace only an existing ready/failed row that has no passing state.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            register(
                specialist_id=args.specialist_id,
                family=args.family,
                expert=args.expert,
                runtime_tree=args.runtime_tree,
                runtime_registry=args.runtime_registry,
                selector_env=args.selector_env,
                state_root=args.state_root,
                run_name=args.run_name,
                handoff_service=args.handoff_service,
                minimum_decisions=args.minimum_decisions,
                required_target_coverage=tuple(args.required_target),
                guide_id=args.guide_id,
                guide_loss_weight=args.guide_loss_weight,
                guide_contract=args.guide_contract,
                guide_contract_sha256=args.guide_contract_sha256,
                guide_version=args.guide_version,
                guide_training_mode=args.guide_training_mode,
                strategic_curriculum_spec=args.strategic_curriculum_spec,
                strategic_curriculum_spec_sha256=(
                    args.strategic_curriculum_spec_sha256
                ),
                strategic_head_role_map=args.strategic_head_role_map,
                strategic_head_role_map_sha256=(
                    args.strategic_head_role_map_sha256
                ),
                strategic_validation_receipt=(
                    args.strategic_validation_receipt
                ),
                strategic_validation_receipt_sha256=(
                    args.strategic_validation_receipt_sha256
                ),
                authorization_name=args.authorization_name,
                replace_unpassed=args.replace_unpassed,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
