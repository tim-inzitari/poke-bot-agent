#!/usr/bin/env python3
"""Resolve one specialist selector into a fully validated Pure-RL launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.holdout_supersession import (
    superseded_external_archetypes,
)
from poke_bot.matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as V5_ADAPTER_CHECKPOINT_FORMAT,
)
from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
    load_slot_registry,
)
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "ops/specialist_runtime_registry_v1.json"
SELECTOR_ENV = "POKEBOT_ACTIVE_SPECIALIST"
GUIDE_WEIGHT_POLICY_SCHEMA = "poke_bot.current_deck_guide_weight_policy/v1"
CRUSTLE_PERSISTENT_GUIDE_POLICY_SCHEMA = (
    "poke_bot.crustle_persistent_guide_hold/v1"
)
GUIDE_TRAINING_MODE_LEGACY = "legacy_policy_ce_v1"
GUIDE_TRAINING_MODE_STRATEGIC = "strategic_curriculum_v1"
GUIDE_TRAINING_MODE_DIRECTIONAL = "strategic_directional_v2"
GUIDE_STRATEGIC_TRAINING_MODES = {
    GUIDE_TRAINING_MODE_STRATEGIC,
    GUIDE_TRAINING_MODE_DIRECTIONAL,
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
GUIDE_ACTION_INFLUENCE_REVISION = 56
DECISION_FUSION_V2_SCHEMA = "poke_bot.causal_decision_fusion/v2"
CANONICAL_LEARNED_DECISION_SOURCES = tuple(
    dict.fromkeys(
        (*DECISION_FUSION_REQUIRED_HEADS, "setup_board_outcome")
    )
)
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


# Revision 192 is deliberately scoped to this exact additional H10 package.
# A generic S++ row must not become launchable by accident.
R192_MARNIE_H10_OPPONENT_ID = (
    "specialist-marnie-final-format-h10-f20efb20f5c3"
)
R192_MARNIE_H10_CHECKPOINT_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
R192_MARNIE_H10_CONTENT_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
R192_HISTORICAL_MARNIE_OPPONENT_ID = (
    "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"
)
R192_SPLUSPLUS_SEMANTICS_KEY = "exact_additional_splusplus_specialist"
_STANDARD_OPPONENT_TIER_POLICY = {
    "eligible_non_active_h10_specialist": {"tier": "S", "weight": 2.0},
    "other_frozen_specialist": {"tier": "A", "weight": 1.0},
    "remaining_public_opponent": {"tier": "A", "weight": 1.0},
}
_R192_MARNIE_SPLUSPLUS_SEMANTICS = {
    "opponent_id": R192_MARNIE_H10_OPPONENT_ID,
    "checkpoint_digest": R192_MARNIE_H10_CHECKPOINT_DIGEST,
    "content_digest": R192_MARNIE_H10_CONTENT_DIGEST,
    "tier": "S++",
    "weight": 4.0,
    "strong_public_practice_floor_games": 1024,
}


def _canonical_expert_ids() -> tuple[str, ...]:
    registry = load_slot_registry(ROOT / "state/matchup_adapter_roster.json")
    return tuple(str(value) for value in registry["active_expert_ids"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "poke_bot.specialist_runtime_registry/v1"
        or int(payload.get("version") or 0) < 1
        or payload.get("selector_environment_variable") != SELECTOR_ENV
        or not isinstance(payload.get("specialists"), dict)
    ):
        raise RuntimeError("invalid specialist runtime registry")
    payload["_path"] = str(source)
    return payload


def _required_runtime_fusion_heads(row: dict[str, Any]) -> list[str]:
    """Resolve the exact registered head inventory without weakening its base."""

    strategic = row.get("guide_training_mode") in GUIDE_STRATEGIC_TRAINING_MODES
    mandatory = list(
        CANONICAL_LEARNED_DECISION_SOURCES
        if strategic
        else DECISION_FUSION_REQUIRED_HEADS
    )
    fusion_contract = row.get("decision_fusion")
    if not isinstance(fusion_contract, dict) or not fusion_contract:
        return mandatory
    declared = fusion_contract.get("required_heads")
    if (
        not isinstance(declared, list)
        or declared[: len(mandatory)] != mandatory
        or len(declared) != len(set(declared))
        or any(
            not isinstance(head, str)
            or not head
            or not all(
                char.islower()
                or char.isdigit()
                or char in {"_", "-"}
                for char in head
            )
            for head in declared
        )
    ):
        raise RuntimeError(
            "registered decision-fusion heads weaken or reorder the "
            "mandatory learned-head inventory"
        )
    return list(declared)


def _required_file(row: dict[str, Any], field: str, digest_field: str) -> Path:
    value = row.get(field)
    expected = str(row.get(digest_field) or "").lower()
    if not value or len(expected) != 64:
        raise RuntimeError(f"ready specialist lacks {field}/{digest_field}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"registered specialist input is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"registered specialist input digest mismatch: {field} "
            f"expected={expected} actual={actual}"
        )
    return path


def _normalized_sha256(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw.removeprefix("sha256:")
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise RuntimeError("invalid sha256 digest")
    return raw


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _validate_strategic_head_roles(
    payload: dict[str, Any],
    *,
    specialist_id: str,
    guide_contract_sha256: str,
    registered_learned_sources: tuple[str, ...],
    require_all_registered_sources: bool,
) -> tuple[str, ...]:
    heads = payload.get("heads")
    declared_sources = payload.get("canonical_learned_decision_sources")
    canonical_prefix = list(CANONICAL_LEARNED_DECISION_SOURCES)
    sources_are_valid = (
        isinstance(declared_sources, list)
        and declared_sources[: len(canonical_prefix)] == canonical_prefix
        and len(declared_sources) == len(set(declared_sources))
        and all(source in registered_learned_sources for source in declared_sources)
        and (
            not require_all_registered_sources
            or declared_sources == list(registered_learned_sources)
        )
    )
    try:
        aggregate_cap = float(payload.get("aggregate_absolute_logit_cap"))
    except (TypeError, ValueError):
        aggregate_cap = float("nan")
    if (
        payload.get("schema") != STRATEGIC_HEAD_ROLE_MAP_SCHEMA
        or payload.get("specialist_id") != specialist_id
        or payload.get("training_mode") != GUIDE_TRAINING_MODE_STRATEGIC
        or int(payload.get("guide_curriculum_revision") or 0) != 51
        or int(payload.get("strategic_branch_scope_revision") or 0) != 56
        or int(payload.get("action_influence_revision") or 0) != 56
        or payload.get("guide_contract_sha256")
        != f"sha256:{guide_contract_sha256}"
        or payload.get("decision_fusion_schema")
        != DECISION_FUSION_V2_SCHEMA
        or payload.get("preserve_v1_additive_residual") is not True
        or not sources_are_valid
        or payload.get("one_route_per_learned_source") is not True
        or payload.get("route_input")
        != "option_hidden_plus_typed_output"
        or payload.get("route_reduction") != "fixed_mean"
        or aggregate_cap != 1.0
        or payload.get("zero_safe_final_projection") is not True
        or payload.get("guide_is_only_action_route_exception") is not True
        or not isinstance(heads, dict)
        or set(heads) != set(declared_sources or ())
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
            head_id not in declared_sources
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


def _validate_implementation_artifacts(receipt: dict[str, Any]) -> None:
    artifacts = receipt.get("implementation_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        raise RuntimeError("strategic curriculum implementation evidence is incomplete")
    roles: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError(
                "strategic curriculum implementation artifact is invalid"
            )
        path = Path(str(artifact.get("path") or "")).expanduser()
        role = str(artifact.get("role") or "")
        if (
            role not in {"training_implementation", "validation_test"}
            or not path.is_absolute()
            or not path.is_file()
        ):
            raise RuntimeError(
                "strategic curriculum implementation artifact is invalid"
            )
        expected = _normalized_sha256(artifact.get("sha256"))
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                "strategic curriculum implementation artifact digest mismatch"
            )
        roles.add(role)
    if roles != {"training_implementation", "validation_test"}:
        raise RuntimeError("strategic curriculum implementation evidence is incomplete")


def _validate_crustle_persistent_guide_policy(policy: dict[str, Any]) -> None:
    """Fail closed unless Crustle retains its owner-pinned guide exception."""

    exact = {
        "schema": CRUSTLE_PERSISTENT_GUIDE_POLICY_SCHEMA,
        "owner_decision_revision": 165,
        "scope": "crustle_persistent_training_only",
        "held_weight": 0.05,
        "automatic_review_after_each_five_iteration_commit": False,
        "automatic_ramp_allowed": False,
        "automatic_decay_allowed": False,
        "change_requires_explicit_owner_decision": True,
        "change_requires_checksum_bound_boundary_receipt": True,
        "direct_policy_cross_entropy_allowed": False,
        "runtime_action_override_allowed": False,
        "serving_authority": False,
        "gate_authority": False,
    }
    if policy != exact:
        raise RuntimeError("Crustle persistent guide policy is incomplete")


def _validate_guide_training_contract(
    row: dict[str, Any],
    specialist_id: str,
    registered_learned_sources: tuple[str, ...] = (
        CANONICAL_LEARNED_DECISION_SOURCES
    ),
) -> str:
    """Return the validated mode, preserving immutable pre-policy rows."""

    raw_mode = str(row.get("guide_training_mode") or "").strip()
    policy = row.get("guide_weight_policy")
    bundle = row.get("strategic_curriculum")
    if row.get("guide_retired") is True:
        permitted_retirement = (
            (
                specialist_id == "marnie-s-grimmsnarl-ex"
                and int(row.get("guide_retirement_revision") or 0) == 140
            )
            or (
                specialist_id == "crustle"
                and int(row.get("guide_retirement_revision") or 0) == 161
            )
        )
        if (
            not permitted_retirement
            or float(row.get("guide_loss_weight") or 0.0) != 0.0
            or row.get("guide_target_generation_required") is not False
            or row.get("guide_conditioned_losses_enabled") is not False
            or row.get("guide_action_influence") is not False
            or raw_mode != GUIDE_TRAINING_MODE_DIRECTIONAL
        ):
            raise RuntimeError("retired specialist guide contract is incomplete")
        # Keep the checksum-bound historical curriculum metadata so the full
        # 19-head Fusion-v3 inventory remains validated. Weight zero and the
        # cleared launch environment make the guide an exact training bypass.
        return raw_mode
    if not raw_mode:
        if policy is not None or bundle is not None:
            raise RuntimeError(
                "guided future specialist lacks an explicit training mode"
            )
        return GUIDE_TRAINING_MODE_LEGACY
    if raw_mode not in {
        GUIDE_TRAINING_MODE_LEGACY,
        *GUIDE_STRATEGIC_TRAINING_MODES,
    }:
        raise RuntimeError("unknown guide training mode")
    if raw_mode == GUIDE_TRAINING_MODE_LEGACY:
        if policy is not None or bundle is not None:
            raise RuntimeError(
                "legacy guide training cannot bind future curriculum policy"
            )
        return raw_mode
    if raw_mode == GUIDE_TRAINING_MODE_DIRECTIONAL:
        if (
            float(row.get("guide_loss_weight") or 0.0) <= 0.0
            or not isinstance(policy, dict)
            or not isinstance(bundle, dict)
            or bundle.get("schema") != STRATEGIC_BUNDLE_SCHEMA
            or bundle.get("training_mode") != raw_mode
            or int(bundle.get("guide_curriculum_revision") or 0) != 104
            or bundle.get("decision_fusion_schema")
            != "poke_bot.causal_decision_fusion/v3"
            or bundle.get("typed_output_centered_routes") is not True
            or bundle.get("route_reliability_bounds") != [0.25, 4.0]
            or float(bundle.get("action_type_reliability_cap") or 0.0) != 0.25
            or bundle.get("final_policy_logits_are_guide_targets") is not False
        ):
            raise RuntimeError("directional guide training contract is incomplete")
        if specialist_id == "crustle":
            _validate_crustle_persistent_guide_policy(policy)
        role_path = _required_file(
            bundle, "head_role_map", "head_role_map_sha256"
        )
        spec_path = _required_file(
            bundle, "curriculum_spec", "curriculum_spec_sha256"
        )
        receipt_path = _required_file(
            bundle, "validation_receipt", "validation_receipt_sha256"
        )
        from poke_bot.train import assert_strategic_curriculum_receipt_contract

        assert_strategic_curriculum_receipt_contract(
            specialist_id=specialist_id,
            curriculum_spec=str(spec_path),
            head_role_map=str(role_path),
            validation_receipt=str(receipt_path),
            expected_training_mode=raw_mode,
        )
        return raw_mode
    if (
        float(row.get("guide_loss_weight") or 0.0) <= 0.0
        or not isinstance(policy, dict)
        or not isinstance(bundle, dict)
        or bundle.get("schema") != STRATEGIC_BUNDLE_SCHEMA
        or bundle.get("training_mode") != GUIDE_TRAINING_MODE_STRATEGIC
        or int(bundle.get("guide_curriculum_revision") or 0) != 51
        or int(bundle.get("strategic_branch_scope_revision") or 0) != 56
        or int(bundle.get("action_influence_revision") or 0) != 56
        or bundle.get("decision_fusion_schema")
        != DECISION_FUSION_V2_SCHEMA
    ):
        raise RuntimeError("strategic guide training contract is incomplete")

    guide_contract_sha256 = _normalized_sha256(
        row.get("guide_contract_sha256")
    )
    role_path = _required_file(
        bundle, "head_role_map", "head_role_map_sha256"
    )
    role_payload = _read_json_object(
        role_path,
        label="strategic curriculum head-role map",
    )
    head_ids = _validate_strategic_head_roles(
        role_payload,
        specialist_id=specialist_id,
        guide_contract_sha256=guide_contract_sha256,
        registered_learned_sources=registered_learned_sources,
        require_all_registered_sources=(
            bundle.get("require_all_registered_learned_sources") is True
        ),
    )
    curriculum_sources = list(
        role_payload["canonical_learned_decision_sources"]
    )
    spec_path = _required_file(
        bundle, "curriculum_spec", "curriculum_spec_sha256"
    )
    spec = _read_json_object(spec_path, label="strategic curriculum spec")
    exact_spec = {
        "schema": STRATEGIC_CURRICULUM_SCHEMA,
        "specialist_id": specialist_id,
        "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
        "guide_curriculum_revision": 51,
        "strategic_branch_scope_revision": 56,
        "action_influence_revision": 56,
        "guide_contract_sha256": f"sha256:{guide_contract_sha256}",
        "head_role_map_sha256": f"sha256:{_sha256(role_path)}",
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
        "canonical_learned_decision_sources": curriculum_sources,
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

    receipt_path = _required_file(
        bundle, "validation_receipt", "validation_receipt_sha256"
    )
    receipt = _read_json_object(
        receipt_path,
        label="strategic curriculum validation receipt",
    )
    exact_receipt = {
        "schema": STRATEGIC_VALIDATION_SCHEMA,
        "status": "validated",
        "specialist_id": specialist_id,
        "training_mode": GUIDE_TRAINING_MODE_STRATEGIC,
        "guide_curriculum_revision": 51,
        "strategic_branch_scope_revision": 56,
        "action_influence_revision": 56,
        "guide_contract_sha256": f"sha256:{guide_contract_sha256}",
        "curriculum_spec_sha256": f"sha256:{_sha256(spec_path)}",
        "head_role_map_sha256": f"sha256:{_sha256(role_path)}",
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
    try:
        strategic_norm = float(
            dict(measurements or {}).get(
                "guide_to_strategic_head_gradient_norm",
                float("nan"),
            )
        )
        guide_rows = int(
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
        or not math.isfinite(strategic_norm)
        or strategic_norm <= 0.0
        or guide_rows <= 0
        or not math.isfinite(aggregate_residual)
        or aggregate_residual < 0.0
        or aggregate_residual > 1.0
        or not isinstance(action_ablations, dict)
        or set(action_ablations) != set(head_ids)
        or not isinstance(route_validation, dict)
        or set(route_validation) != set(head_ids)
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
    _validate_implementation_artifacts(receipt)
    return raw_mode


def _validate_guide_weight_policy(
    row: dict[str, Any],
    learned_decision_sources: tuple[str, ...] = (
        CANONICAL_LEARNED_DECISION_SOURCES
    ),
) -> None:
    """Validate the prospective policy when a future successor declares it."""

    policy = row.get("guide_weight_policy")
    if policy is None:
        # Existing registered lineages predate the prospective policy. Their immutable
        # runtime rows remain readable; every new registration writes policy.
        return
    if not isinstance(policy, dict):
        raise RuntimeError("current-deck guide weight policy is malformed")
    module = _required_file(
        policy, "schedule_module", "schedule_module_sha256"
    )
    evidence_module = _required_file(
        policy, "evidence_module", "evidence_module_sha256"
    )
    review_module = _required_file(
        policy, "review_request_module", "review_request_module_sha256"
    )
    boundary_controller = _required_file(
        policy, "boundary_controller", "boundary_controller_sha256"
    )
    shadow_pair_module = _required_file(
        policy, "shadow_pair_module", "shadow_pair_module_sha256"
    )
    shadow_pair_runner = _required_file(
        policy, "shadow_pair_runner", "shadow_pair_runner_sha256"
    )
    shadow_queue_processor = _required_file(
        policy, "shadow_queue_processor", "shadow_queue_processor_sha256"
    )
    exact = {
        "schema": GUIDE_WEIGHT_POLICY_SCHEMA,
        "owner_decision_revision": 43,
        "prospective_scope_revision": 44,
        "learning_semantics_revision": 46,
        "guide_curriculum_revision": 51,
        "strategic_branch_scope_revision": 56,
        "action_influence_revision": 56,
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "scope": "future_specialist_training_runs_only",
        "prospective_effective_specialist": "archaludon-ex",
        "retroactive_application_to_completed_frozen_or_started_runs": False,
        "historical_weight_or_receipt_rewrite_allowed": False,
        "active_teal_revision42_exception_preserved": True,
        "bootstrap_ramp": [0.01, 0.05],
        "post_bootstrap_positive_ramp_steps": [0.15, 0.25, 0.35, 0.50],
        "maximum_auxiliary_loss_weight": 0.50,
        "review_every_completed_iterations": 5,
        "minimum_paired_games_per_review": 1000,
        "minimum_paired_games_per_matchup": 50,
        "realized_win_delta_confidence_level": 0.90,
        "evidence_receipt_schema": (
            "poke_bot.current_deck_guide_paired_evaluation/v1"
        ),
        "schedule_receipt_schema": (
            "poke_bot.current_deck_guide_weight_schedule/v1"
        ),
        "review_request_schema": (
            "poke_bot.current_deck_guide_weight_review_request/v1"
        ),
        "boundary_receipt_schema": (
            "poke_bot.future_specialist_guide_weight_boundary/v1"
        ),
        "shadow_pair_manifest_schema": (
            "poke_bot.future_guide_weight_shadow_pair/v1"
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
        "preserve_v1_additive_residual": True,
        "canonical_learned_decision_sources": list(learned_decision_sources),
        "one_route_per_learned_source": True,
        "route_input": "option_hidden_plus_typed_output",
        "route_reduction": "fixed_mean",
        "aggregate_absolute_logit_cap": 1.0,
        "zero_safe_final_projection": True,
        "guide_is_only_action_route_exception": True,
        "elapsed_time_only_progression_allowed": False,
        "guide_head_only_bookkeeping_allowed": False,
        "dashboard_only_or_runtime_action_bias_allowed": False,
        "decay_after_consecutive_nonpositive_reviews": 2,
        "decay_steps": [0.15, 0.075, 0.0],
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
    for key, expected in exact.items():
        if policy.get(key) != expected:
            raise RuntimeError(
                f"current-deck guide weight policy mismatch: {key}"
            )
    if (
        policy.get("positive_ramp_evidence")
        != "training_ineligible_paired_guide_on_guide_off_"
        "realized_win_delta_lower_confidence_bound_above_zero"
        or module.name != "deck_guide_schedule.py"
        or evidence_module.name != "guide_weight_evidence.py"
        or review_module.name != "guide_weight_review.py"
        or boundary_controller.name
        != "apply_future_guide_weight_at_boundary.py"
        or shadow_pair_module.name != "guide_weight_shadow_pair.py"
        or shadow_pair_runner.name
        != "run_future_guide_weight_shadow_pair.py"
        or shadow_queue_processor.name
        != "process_future_guide_weight_review_queue.py"
        or len(
            {
                module,
                evidence_module,
                review_module,
                boundary_controller,
                shadow_pair_module,
                shadow_pair_runner,
                shadow_queue_processor,
            }
        )
        != 7
    ):
        raise RuntimeError("current-deck guide evidence policy is invalid")


def _resolve(
    registry: dict[str, Any], specialist_id: str
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    selected = str(specialist_id or "").strip().lower()
    rows = dict(registry["specialists"])
    if selected not in rows:
        raise RuntimeError(
            f"unknown {SELECTOR_ENV}={selected!r}; "
            f"registered={sorted(rows)}"
        )
    row = dict(rows[selected])
    if row.get("status") != "ready":
        raise RuntimeError(
            f"specialist {selected!r} is not ready: "
            f"{row.get('reason') or row.get('status') or 'unknown reason'}"
        )
    checkpoint = _required_file(
        row, "initial_checkpoint", "initial_checkpoint_sha256"
    )
    expert = _required_file(
        row, "expert_manifest", "expert_manifest_sha256"
    )
    runtime_tree = _required_file(
        row, "matchup_runtime_tree", "matchup_runtime_tree_sha256"
    )
    adapter_authorization = _required_file(
        row,
        "matchup_adapter_authorization",
        "matchup_adapter_authorization_sha256",
    )
    tree = json.loads(runtime_tree.read_text(encoding="utf-8"))
    runtime = dict(tree.get("runtime_contract") or {})
    targets = tuple(str(value) for value in tree.get("targets") or ())
    canonical_expert_ids = _canonical_expert_ids()
    accepted = {
        str(value) for value in runtime.get("accepted_archetype_ids") or ()
    }
    if (
        tree.get("runtime_enabled") is not True
        or targets != canonical_expert_ids
        or len(set(targets)) != len(canonical_expert_ids)
        or selected not in targets
        or selected not in accepted
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise RuntimeError(
            f"specialist {selected!r} lacks an activated canonical mirror route"
        )
    authorization = json.loads(
        adapter_authorization.read_text(encoding="utf-8")
    )
    if (
        authorization.get("schema")
        not in {
            "poke_bot.matchup_adapter_rehearsal_authorization/v1",
            "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1",
        }
        or authorization.get("optimizer_scope") != "matchup_adapter_bank_only"
        or authorization.get("runtime_enabled") is not False
        or int(row.get("matchup_adapter_epochs_per_rl_iteration") or 0) < 1
    ):
        raise RuntimeError(
            f"specialist {selected!r} lacks adapter-only training authorization"
        )
    required_fusion_heads = tuple(_required_runtime_fusion_heads(row))
    guide_weight = float(row.get("guide_loss_weight") or 0.0)
    guide_id = str(row.get("guide_id") or "").strip().casefold()
    guide_training_mode = _validate_guide_training_contract(
        row,
        selected,
        required_fusion_heads,
    )
    if guide_weight > 0.0:
        guide_contract = _required_file(
            row, "guide_contract", "guide_contract_sha256"
        )
        if (
            guide_id != selected
            or not str(row.get("guide_version") or "").strip()
            or guide_contract.suffix not in {".yaml", ".yml"}
        ):
            raise RuntimeError(
                f"specialist {selected!r} lacks its checksum-bound deck guide"
            )
        if guide_training_mode == GUIDE_TRAINING_MODE_STRATEGIC:
            role_map = _read_json_object(
                _required_file(
                    dict(row["strategic_curriculum"]),
                    "head_role_map",
                    "head_role_map_sha256",
                ),
                label="strategic curriculum head-role map",
            )
            _validate_guide_weight_policy(
                row,
                tuple(role_map["canonical_learned_decision_sources"]),
            )
    tactical_override = row.get("tactical_outcome_loss_weight_override")
    if tactical_override is not None and (
        selected != "teal-mask-ogerpon-ex"
        or float(tactical_override) != 0.01
    ):
        raise RuntimeError(
            "tactical-outcome weight override is authorized only for Teal at 0.01"
        )
    import torch

    validation_checkpoint = checkpoint
    loop_state_path = (
        ROOT / "outputs" / "pure_rl" / str(row["run_name"]) / "loop_state.json"
    )
    if loop_state_path.is_file():
        loop_state = _read_json_object(loop_state_path, label="specialist loop state")
        learner = dict(loop_state.get("learner") or {})
        learner_path = Path(str(learner.get("path") or "")).expanduser()
        learner_digest = str(learner.get("digest") or "").removeprefix("sha256:")
        if (
            learner_path.is_file()
            and len(learner_digest) == 64
            and _sha256(learner_path) == learner_digest
        ):
            validation_checkpoint = learner_path
    checkpoint_payload = torch.load(
        validation_checkpoint, map_location="cpu", weights_only=False
    )
    fusion_contract = dict(row.get("decision_fusion") or {})
    model_config = dict(checkpoint_payload.get("model_config") or {})
    required_fusion_heads = list(required_fusion_heads)
    required_fusion_schema = (
        "poke_bot.causal_decision_fusion/v3"
        if row.get("guide_training_mode") == GUIDE_TRAINING_MODE_DIRECTIONAL
        else (
            DECISION_FUSION_V2_SCHEMA
            if row.get("guide_training_mode") == GUIDE_TRAINING_MODE_STRATEGIC
            else DECISION_FUSION_SCHEMA
        )
    )
    if (
        fusion_contract.get("required") is True
        or model_config.get("expanded_heads_enabled") is True
    ):
        provenance = dict(checkpoint_payload.get("provenance") or {})
        fusion_inventory = dict(provenance.get("decision_fusion") or {})
        fusion_tensors = {
            name: tensor
            for name, tensor in dict(
                checkpoint_payload.get("model_state_dict") or {}
            ).items()
            if str(name).startswith("decision_fusion.")
        }
        if (
            (
                fusion_contract
                and (
                    fusion_contract.get("schema") != required_fusion_schema
                    or fusion_contract.get("runtime_enabled") is not True
                    or fusion_contract.get("required_heads")
                    != required_fusion_heads
                )
            )
            or model_config.get("expanded_heads_enabled") is not True
            or model_config.get("decision_fusion_enabled") is not True
            or model_config.get("decision_fusion_runtime_enabled") is not True
            or fusion_inventory.get("schema") != required_fusion_schema
            or fusion_inventory.get("enabled") is not True
            or fusion_inventory.get("runtime_enabled") is not True
            or fusion_inventory.get("required_heads")
            != required_fusion_heads
            or not fusion_tensors
        ):
            raise RuntimeError(
                f"specialist {selected!r} lacks its mandatory all-head "
                "runtime decision-fusion checkpoint"
            )
    checkpoint_extra = dict(checkpoint_payload.get("extra") or {})
    checkpoint_adapter_config = dict(
        checkpoint_extra.get("matchup_adapter_config") or {}
    )
    checkpoint_format = str(checkpoint_adapter_config.get("format") or "")
    # V5 checkpoints created before the format selector was serialized contain
    # only the ordered expert_ids list.  Preserve that established wire format
    # while requiring every V6 checkpoint to identify itself explicitly.
    if not checkpoint_format and checkpoint_adapter_config.get("expert_ids"):
        checkpoint_format = V5_ADAPTER_CHECKPOINT_FORMAT
    if checkpoint_format == V6_ADAPTER_CHECKPOINT_FORMAT:
        checkpoint_registry = dict(
            checkpoint_adapter_config.get("slot_registry") or {}
        )
        canonical_registry = load_slot_registry(
            ROOT / "state/matchup_adapter_roster.json"
        )
        if checkpoint_registry != canonical_registry:
            raise RuntimeError(
                f"specialist {selected!r} V6 checkpoint registry is not the "
                "exact runtime registry"
            )
        checkpoint_routes = tuple(
            str(value)
            for value in checkpoint_registry.get("active_expert_ids") or ()
        )
        physical_route_count = int(
            checkpoint_adapter_config.get("slot_capacity") or 0
        )
    elif checkpoint_format == V5_ADAPTER_CHECKPOINT_FORMAT:
        checkpoint_routes = tuple(
            str(value)
            for value in checkpoint_adapter_config.get("expert_ids") or ()
        )
        physical_route_count = len(checkpoint_routes)
    else:
        raise RuntimeError(
            f"specialist {selected!r} has an unsupported adapter format"
        )
    adapter_tensor_routes = {
        int(name.split(".")[2])
        for name in dict(checkpoint_payload.get("model_state_dict") or {})
        if name.startswith("matchup_adapter_bank.experts.")
        and len(name.split(".")) >= 4
    }
    if (
        checkpoint_routes != targets
        or adapter_tensor_routes != set(range(physical_route_count))
    ):
        raise RuntimeError(
            f"specialist {selected!r} checkpoint lacks the canonical "
            f"{len(canonical_expert_ids)}-route logical bank with "
            f"{physical_route_count} physical slots"
        )
    for field in (
        "run_name",
        "log",
        "measurement_decks",
        "terminal_gate_marker",
    ):
        if not str(row.get(field) or "").strip():
            raise RuntimeError(f"ready specialist lacks {field}")
    return row, checkpoint, expert, runtime_tree, adapter_authorization


def _r192_marnie_splusplus_semantics(
    semantics: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the one authorized additive S++ binding, if present."""

    override = semantics.get(R192_SPLUSPLUS_SEMANTICS_KEY)
    if override is None:
        return None
    if (
        override != _R192_MARNIE_SPLUSPLUS_SEMANTICS
        or type(override.get("strong_public_practice_floor_games")) is not int
    ):
        raise RuntimeError("active specialist r192 Marnie S++ semantics changed")
    return dict(override)


def _validate_r192_marnie_splusplus_binding(
    *,
    roster: list[dict[str, Any]],
    gate_by_id: dict[str, dict[str, Any]],
    frozen_by_id: dict[str, dict[str, Any]],
) -> None:
    """Fail closed unless r192 retains two distinct checksum-bound Marnies."""

    roster_ids = [str(row.get("opponent_id") or "") for row in roster]
    content_digests = [str(row.get("content_digest") or "") for row in roster]
    if (
        not all(roster_ids)
        or len(set(roster_ids)) != len(roster_ids)
        or not all(content_digests)
        or len(set(content_digests)) != len(content_digests)
    ):
        raise RuntimeError("active specialist r192 Marnie roster has an alias")

    h10_gate = gate_by_id.get(R192_MARNIE_H10_OPPONENT_ID)
    historical_gate = gate_by_id.get(R192_HISTORICAL_MARNIE_OPPONENT_ID)
    h10_frozen = frozen_by_id.get(R192_MARNIE_H10_OPPONENT_ID)
    historical_frozen = frozen_by_id.get(R192_HISTORICAL_MARNIE_OPPONENT_ID)
    if not all((h10_gate, historical_gate, h10_frozen, historical_frozen)):
        raise RuntimeError(
            "active specialist r192 Marnie requires distinct H10 and historical rows"
        )
    if (
        h10_gate.get("frozen_checkpoint_digest")
        != R192_MARNIE_H10_CHECKPOINT_DIGEST
        or h10_gate.get("content_digest") != R192_MARNIE_H10_CONTENT_DIGEST
        or h10_gate.get("strong_public_practice_floor_games") != 1024
        or type(h10_gate.get("strong_public_practice_floor_games")) is not int
        or h10_frozen.get("checkpoint_digest")
        != R192_MARNIE_H10_CHECKPOINT_DIGEST
        or h10_frozen.get("content_digest") != R192_MARNIE_H10_CONTENT_DIGEST
        or h10_frozen.get("final_format_h10_refresh") is not True
        or not str(historical_gate.get("content_digest") or "")
        or historical_gate.get("content_digest") == R192_MARNIE_H10_CONTENT_DIGEST
        or historical_gate.get("content_digest")
        != historical_frozen.get("content_digest")
    ):
        raise RuntimeError("active specialist r192 Marnie checksum binding changed")


def _gate_runtime(active_gate: Path, frozen_registry: Path) -> tuple[str, int]:
    gate_contract = json.loads(active_gate.read_text(encoding="utf-8"))
    registry = json.loads(frozen_registry.read_text(encoding="utf-8"))
    gate = dict(gate_contract.get("next_gate") or {})
    evaluation = dict(gate.get("evaluation") or {})
    roster = [dict(row) for row in (gate.get("roster") or [])]
    frozen = [
        dict(row)
        for row in (registry.get("specialists") or [])
        if row.get("frozen") is True and row.get("public_mix_eligible") is True
    ]
    gate_frozen = [
        row for row in roster if row.get("frozen_specialist") is True
    ]
    gate_by_id = {
        str(row.get("opponent_id") or ""): row for row in gate_frozen
    }
    frozen_by_id = {
        str(row.get("opponent_id") or ""): row for row in frozen
    }
    gate_id = str(gate.get("id") or "")
    games_total = int(evaluation.get("games_total", -1))
    semantics = dict(gate_contract.get("active_gate_semantics") or {})
    tier_policy = dict(semantics.get("opponent_tier_policy") or {})
    r192_marnie_semantics = _r192_marnie_splusplus_semantics(semantics)
    superseded_archetypes = superseded_external_archetypes(registry)
    base_premium_agents = int(
        semantics.get(
            "base_premium_agents",
            -1 if superseded_archetypes else 8,
        )
    )
    if (
        registry.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or not gate_id
        or base_premium_agents < 0
        or len(roster) != base_premium_agents + len(frozen)
        or len(gate_by_id) != len(gate_frozen)
        or len(frozen_by_id) != len(frozen)
        or set(gate_by_id) != set(frozen_by_id)
        or games_total != 250 * len(roster)
        or int(evaluation.get("games_per_opponent", -1)) != 250
        or int(evaluation.get("seat0_games_per_opponent", -1)) != 125
        or int(evaluation.get("seat1_games_per_opponent", -1)) != 125
    ):
        raise RuntimeError("active specialist gate/registry contract changed")
    if r192_marnie_semantics is not None:
        _validate_r192_marnie_splusplus_binding(
            roster=roster,
            gate_by_id=gate_by_id,
            frozen_by_id=frozen_by_id,
        )
    for opponent_id, frozen_row in frozen_by_id.items():
        gate_row = gate_by_id[opponent_id]
        if gate_row.get("frozen_checkpoint_digest") != frozen_row.get(
            "checkpoint_digest"
        ):
            raise RuntimeError(
                f"active specialist checkpoint mismatch: {opponent_id}"
            )
        if (
            r192_marnie_semantics is not None
            and opponent_id == R192_MARNIE_H10_OPPONENT_ID
        ):
            if (gate_row.get("tier"), float(gate_row.get("weight", -1))) != (
                "S++",
                4.0,
            ):
                raise RuntimeError(
                    f"active specialist r192 Marnie S++ tier mismatch: {opponent_id}"
                )
        elif tier_policy:
            is_h10 = bool(frozen_row.get("final_format_h10_refresh")) or (
                opponent_id
                == "specialist-alakazam-final-format-h10-02c014ad7c33"
                and gate_row.get("frozen_checkpoint_digest")
                == "sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92"
            )
            expected = ("S", 2.0) if is_h10 else ("A", 1.0)
            if (gate_row.get("tier"), float(gate_row.get("weight", -1))) != expected:
                raise RuntimeError(
                    f"active specialist tier-policy mismatch: {opponent_id}"
                )
        elif r192_marnie_semantics is not None:
            if (gate_row.get("tier"), float(gate_row.get("weight", -1))) != (
                "S+",
                2.0,
            ):
                raise RuntimeError(
                    f"active specialist S+ tier mismatch: {opponent_id}"
                )
        elif gate_row.get("tier") != "S+":
            raise RuntimeError(
                f"active specialist S+ tier mismatch: {opponent_id}"
            )
    if tier_policy:
        expected_policy = _STANDARD_OPPONENT_TIER_POLICY
        public_rows = [row for row in roster if row.get("frozen_specialist") is not True]
        if tier_policy != expected_policy or any(
            (row.get("tier"), float(row.get("weight", -1))) != ("A", 1.0)
            for row in public_rows
        ):
            raise RuntimeError("active specialist public tier policy changed")
    return gate_id, games_total


def _validate_expert_practice_anchor(
    row: dict[str, Any],
    frozen_registry: Path,
) -> None:
    anchor = row.get("expert_practice_anchor")
    if anchor is None:
        return
    if not isinstance(anchor, dict):
        raise RuntimeError("expert/practice anchor is malformed")
    checkpoint_digest = str(anchor.get("checkpoint_digest") or "")
    completion = Path(str(anchor.get("completion_receipt") or "")).resolve()
    bundle = Path(str(anchor.get("submission_bundle") or "")).resolve()
    if (
        anchor.get("schema")
        != "poke_bot.crustle_marnie_expert_practice_anchor/v1"
        or anchor.get("owner_decision_revision") != 161
        or anchor.get("specialist_id") != "marnie-s-grimmsnarl-ex"
        or anchor.get("actions_relabelled_as_crustle_expert_targets") is not False
        or not completion.is_file()
        or _sha256(completion) != str(anchor.get("completion_receipt_sha256") or "")
        or not bundle.is_file()
        or _sha256(bundle) != str(anchor.get("submission_bundle_sha256") or "")
        or not checkpoint_digest.startswith("sha256:")
    ):
        raise RuntimeError("Crustle Marnie expert/practice anchor changed")
    frozen = _read_json_object(frozen_registry, label="frozen specialist registry")
    matches = [
        dict(candidate)
        for candidate in frozen.get("specialists") or ()
        if candidate.get("specialist_id") == "marnie-s-grimmsnarl-ex"
    ]
    if (
        len(matches) != 1
        or matches[0].get("checkpoint_digest") != checkpoint_digest
        or matches[0].get("public_mix_eligible") is not True
        or matches[0].get("final_format_h10_refresh") is not True
    ):
        raise RuntimeError(
            "completed Marnie H10 is absent from Crustle practice/holdout"
        )


def _without_valued_options(
    values: list[str], options: set[str]
) -> list[str]:
    """Remove repeatable ``--option value`` pairs before row overrides."""

    output: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in options:
            if index + 1 >= len(values):
                raise RuntimeError(f"trainer option lacks a value: {value}")
            index += 2
            continue
        output.append(value)
        index += 1
    return output


def _build_command(
    registry: dict[str, Any],
    specialist_id: str,
    row: dict[str, Any],
    checkpoint: Path,
    expert: Path,
    runtime_tree: Path,
    adapter_authorization: Path,
) -> list[str]:
    runtime_root = Path(str(registry["runtime_root"])).expanduser().resolve()
    python = str(registry["python"])
    launcher = runtime_root / str(registry["launcher"])
    active_gate = runtime_root / str(registry["active_gate_contract"])
    research = runtime_root / str(registry["research_control_registry"])
    frozen = runtime_root / str(registry["frozen_specialist_registry"])
    for path in (launcher, active_gate, research, frozen):
        if not path.is_file():
            raise RuntimeError(f"runtime contract is missing: {path}")
    gate_id, heldout_games = _gate_runtime(active_gate, frozen)
    _validate_expert_practice_anchor(row, frozen)
    common_trainer_args = [
        str(value) for value in registry.get("common_trainer_args") or []
    ]
    common_trainer_args = _without_valued_options(
        common_trainer_args,
        {"--expert-min-decisions", "--expert-required-target"},
    )
    forbidden = {
        "--heldout-games",
        "--terminal-active-gate-id",
        "--minimum-terminal-iteration",
        "--iterations",
    }
    if any(value in forbidden for value in common_trainer_args):
        raise RuntimeError(
            "dynamic gate arguments cannot be duplicated in common_trainer_args"
        )
    minimum_terminal_iteration = int(
        row.get(
            "minimum_terminal_iteration",
            registry["minimum_terminal_iteration"],
        )
    )
    iteration_ceiling = int(
        row.get("iteration_ceiling", registry["iteration_ceiling"])
    )
    if minimum_terminal_iteration < 0 or iteration_ceiling < minimum_terminal_iteration:
        raise RuntimeError("invalid specialist iteration window")
    expert_required_targets = [
        str(target)
        for target in row.get("expert_required_target_coverage") or ()
    ]
    if specialist_id in {"slowking", "teal-mask-ogerpon-ex"} and (
        "combo_state_rows" not in expert_required_targets
    ):
        expert_required_targets.append("combo_state_rows")
    run_root = (
        runtime_root
        / "outputs"
        / "pure_rl"
        / Path(str(row["run_name"])).name
    )
    loop_state_path = run_root / "loop_state.json"
    initial_checkpoint_args = [
        "--initial-learner-checkpoint",
        str(checkpoint),
    ]
    if loop_state_path.is_file():
        try:
            loop_state = json.loads(loop_state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"existing specialist loop state failed to parse: {loop_state_path}"
            ) from exc
        learner = (
            dict(loop_state.get("learner") or {})
            if isinstance(loop_state, dict)
            else {}
        )
        learner_path = Path(str(learner.get("path") or "")).expanduser()
        learner_digest = str(learner.get("digest") or "").removeprefix("sha256:")
        if (
            not isinstance(loop_state, dict)
            or int(loop_state.get("next_iteration", -1)) < 0
            or not learner_path.is_file()
            or len(learner_digest) != 64
            or _sha256(learner_path) != learner_digest
        ):
            raise RuntimeError(
                f"existing specialist loop state is not resumable: {loop_state_path}"
            )
        # This option establishes a new run's immutable seed lineage. Passing
        # a later checkpoint here during resume conflicts with that lineage;
        # the checksum-validated loop state is authoritative instead.
        initial_checkpoint_args = []
    _combo_default_specialists = {
        "slowking",
        "marnie-s-grimmsnarl-ex",
        "teal-mask-ogerpon-ex",
    }
    combo_state_loss_weight = float(
        row.get(
            "combo_state_loss_weight",
            0.025 if specialist_id in _combo_default_specialists else 0.0,
        )
    )
    setup_board_outcome_loss_weight = float(
        row.get("setup_board_outcome_loss_weight", 0.025)
    )
    command = [
        python,
        "-u",
        str(launcher),
        "--run-name",
        str(row["run_name"]),
        *[str(value) for value in registry.get("common_launcher_args") or []],
        "--python",
        python,
        "--log",
        str(row["log"]),
        "--",
        "--specialist-archetype",
        specialist_id,
        *initial_checkpoint_args,
        "--active-gate-contract",
        str(active_gate),
        "--research-control-registry",
        str(research),
        "--frozen-specialist-registry",
        str(frozen),
        "--measurement-decks",
        str(row["measurement_decks"]),
        "--expert-manifest",
        str(expert),
        "--dormant-matchup-adapter-epochs",
        str(int(row["matchup_adapter_epochs_per_rl_iteration"])),
        "--dormant-matchup-adapter-activation-receipt",
        str(adapter_authorization),
        "--current-deck-guide-loss-weight",
        str(float(row.get("guide_loss_weight") or 0.0)),
        "--setup-board-outcome-loss-weight",
        str(setup_board_outcome_loss_weight),
        "--combo-state-loss-weight",
        str(combo_state_loss_weight),
        *(
            [
                "--current-deck-guide-training-mode",
                str(row["guide_training_mode"]),
                "--current-deck-guide-curriculum-spec",
                str(dict(row["strategic_curriculum"])["curriculum_spec"]),
                "--current-deck-guide-head-role-map",
                str(dict(row["strategic_curriculum"])["head_role_map"]),
                "--current-deck-guide-curriculum-validation-receipt",
                str(
                    dict(row["strategic_curriculum"])[
                        "validation_receipt"
                    ]
                ),
            ]
            if (
                row.get("guide_training_mode") in GUIDE_STRATEGIC_TRAINING_MODES
                and row.get("guide_retired") is not True
            )
            else []
        ),
        *(
            [
                "--tactical-outcome-loss-weight-override",
                str(float(row["tactical_outcome_loss_weight_override"])),
            ]
            if row.get("tactical_outcome_loss_weight_override") is not None
            else []
        ),
        "--terminal-active-gate-id",
        gate_id,
        "--terminal-gate-marker-name",
        str(row["terminal_gate_marker"]),
        "--minimum-terminal-iteration",
        str(minimum_terminal_iteration),
        "--iterations",
        str(iteration_ceiling + 1),
        "--heldout-games",
        str(heldout_games),
        "--expert-min-decisions",
        str(int(row.get("expert_minimum_decisions") or 20_000)),
        *[
            value
            for target in expert_required_targets
            for value in ("--expert-required-target", str(target))
        ],
        *common_trainer_args,
    ]
    return command


def _validate_exact_runtime_identity(
    specialist_id: str,
    row: dict[str, Any],
) -> None:
    """Exercise exact deck and guide dispatch before the service starts."""

    from scripts.train_pure_rl import _our_decks

    decks = _our_decks("specialist", specialist_id)
    if len(decks) != 1 or decks[0][0] != specialist_id or len(decks[0][1]) != 60:
        raise RuntimeError(
            "selected specialist has no unique exact 60-card runtime identity"
        )

    guide_weight = float(row.get("guide_loss_weight") or 0.0)
    if guide_weight <= 0.0:
        return
    guide_id = str(row.get("guide_id") or "").strip().lower()
    if guide_id != specialist_id:
        raise RuntimeError(
            "nonzero guide weight requires guide_id equal to active specialist"
        )
    from poke_bot import deck_guides

    if guide_id not in deck_guides.supported_ids():
        raise RuntimeError(
            f"current-deck guide is not registered for runtime: {guide_id!r}"
        )
    previous_id = os.environ.get("POKEBOT_CURRENT_DECK_GUIDE")
    previous_targets = os.environ.get("POKEBOT_CURRENT_DECK_GUIDE_TARGETS")
    try:
        os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = guide_id
        os.environ["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = "1"
        if not deck_guides.enabled():
            raise RuntimeError("selected current-deck guide is not enabled")
        if deck_guides.guide_version() != str(row.get("guide_version") or ""):
            raise RuntimeError(
                "selected current-deck guide version differs from runtime registry"
            )
    finally:
        if previous_id is None:
            os.environ.pop("POKEBOT_CURRENT_DECK_GUIDE", None)
        else:
            os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = previous_id
        if previous_targets is None:
            os.environ.pop("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", None)
        else:
            os.environ["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = previous_targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print the selected command without executing it.",
    )
    args = parser.parse_args(argv)
    registry = _load_registry(args.registry)
    selected = str(os.environ.get(SELECTOR_ENV) or "").strip().lower()
    if not selected:
        raise RuntimeError(f"{SELECTOR_ENV} is required")
    row, checkpoint, expert, runtime_tree, adapter_authorization = _resolve(
        registry, selected
    )
    command = _build_command(
        registry,
        selected,
        row,
        checkpoint,
        expert,
        runtime_tree,
        adapter_authorization,
    )
    _validate_exact_runtime_identity(selected, row)
    print(
        "SPECIALIST_SELECTOR_OK "
        f"id={selected} run={row['run_name']} "
        f"checkpoint=sha256:{_sha256(checkpoint)} "
        f"expert=sha256:{_sha256(expert)} "
        f"runtime_tree=sha256:{_sha256(runtime_tree)} "
        f"adapter_authorization=sha256:{_sha256(adapter_authorization)}",
        flush=True,
    )
    print("SPECIALIST_COMMAND " + shlex.join(command), flush=True)
    if args.check:
        return 0
    os.chdir(Path(str(registry["runtime_root"])).expanduser().resolve())
    environment = os.environ.copy()
    guide_id = str(row.get("guide_id") or "").strip().lower()
    guide_weight = float(row.get("guide_loss_weight") or 0.0)
    if guide_weight > 0.0:
        if guide_id != selected:
            raise RuntimeError(
                "nonzero guide weight requires guide_id equal to active specialist"
            )
        environment["POKEBOT_CURRENT_DECK_GUIDE"] = guide_id
        environment["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = "1"
        environment["POKEBOT_CURRENT_DECK_GUIDE_CONTRACT"] = str(
            Path(str(row["guide_contract"])).expanduser().resolve()
        )
    else:
        environment.pop("POKEBOT_CURRENT_DECK_GUIDE", None)
        environment.pop("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", None)
        environment.pop("POKEBOT_CURRENT_DECK_GUIDE_CONTRACT", None)
    environment["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    environment["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(runtime_tree)
    environment["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] = "runtime"
    import torch

    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_adapter_config = dict(
        (checkpoint_payload.get("extra") or {}).get("matchup_adapter_config")
        or {}
    )
    checkpoint_format = str(checkpoint_adapter_config.get("format") or "")
    if not checkpoint_format and checkpoint_adapter_config.get("expert_ids"):
        checkpoint_format = V5_ADAPTER_CHECKPOINT_FORMAT
    environment["POKEBOT_MATCHUP_ADAPTER_FORMAT"] = checkpoint_format
    if checkpoint_format == V6_ADAPTER_CHECKPOINT_FORMAT:
        environment["POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH"] = str(
            ROOT / "state/matchup_adapter_roster.json"
        )
    else:
        environment.pop("POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH", None)
    os.execvpe(command[0], command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
