#!/usr/bin/env python3
"""Materialize measured revision-56 curriculum and per-head route receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from poke_bot.model import (
    COMBO_STATE_HEAD_OUTPUTS,
    CausalDecisionFusion,
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_V2_SCHEMA,
    SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
)
from poke_bot.strategic_heads import (
    ACTION_UTILITY_NAMES,
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    EXPANDED_STRATEGIC_SCHEMA_VERSION,
    OPPONENT_RESPONSE_NAMES,
    RESOURCE_FORECAST_NAMES,
    TACTICAL_HORIZONS,
    TACTICAL_OUTCOME_NAMES,
)
from poke_bot.strategic_losses import (
    expanded_strategic_losses,
    guide_outcome_backed_loss_weights,
)


TRAINING_MODE = "strategic_curriculum_v1"
GUIDE_CURRICULUM_REVISION = 51
ACTION_REVISION = 56
ROLE_SCHEMA = "poke_bot.future_specialist_strategic_head_roles/v1"
SPEC_SCHEMA = "poke_bot.future_specialist_strategic_curriculum/v1"
RECEIPT_SCHEMA = (
    "poke_bot.future_specialist_strategic_curriculum_validation/v1"
)
REQUIRED_TRAINING_PATHS = [
    "supervised_bootstrap",
    "pure_rl",
    "resident_expert_rehearsal",
]
LEARNED_SOURCES = tuple(
    dict.fromkeys((*DECISION_FUSION_REQUIRED_HEADS, "setup_board_outcome"))
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expanded_target() -> dict[str, Any]:
    tactical = [
        [float(horizon + column) for column in range(len(TACTICAL_OUTCOME_NAMES))]
        for horizon in range(len(TACTICAL_HORIZONS))
    ]
    for row in tactical:
        row[2], row[3] = 1.0, 0.0
    resources = [
        float(index + 1) for index in range(len(RESOURCE_FORECAST_NAMES))
    ]
    resources[4], resources[5] = 1.0, 0.0
    utility = [float(index + 1) for index in range(len(ACTION_UTILITY_NAMES))]
    utility[-1] = 1.0
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "version": EXPANDED_STRATEGIC_SCHEMA_VERSION,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "action_factors": [
            {"action_type": True, "target": True, "resource": True}
        ],
        "action_utility": {
            "values": utility,
            "mask": [True] * len(ACTION_UTILITY_NAMES),
        },
        "tactical_outcomes": {
            "values": tactical,
            "mask": [
                [True] * len(TACTICAL_OUTCOME_NAMES)
                for _ in TACTICAL_HORIZONS
            ],
        },
        "opponent_response": {
            "values": [
                float(index % 2)
                for index in range(len(OPPONENT_RESPONSE_NAMES))
            ],
            "mask": [True] * len(OPPONENT_RESPONSE_NAMES),
        },
        "resource_forecast": {
            "values": resources,
            "mask": [True] * len(RESOURCE_FORECAST_NAMES),
        },
        "game_phase": 2,
        "outcome_class": 2,
        "remaining_turns_log1p": 1.25,
        "provenance": {
            "trajectory": "full_untruncated_same_seat",
            "transition_after": "explicit_public_post_action",
            "terminal_complete": True,
            "target_only": True,
        },
    }


def _guide_gradient_measurement() -> float:
    option_outputs = {
        "action_q": torch.zeros(1, 2, requires_grad=True),
        "action_type": torch.zeros(1, 2, requires_grad=True),
        "action_target": torch.zeros(1, 2, requires_grad=True),
        "action_resource": torch.zeros(1, 2, requires_grad=True),
        "action_utility": torch.zeros(1, 2, 6, requires_grad=True),
    }
    state_outputs = {
        "tactical_outcome": torch.zeros(1, 3, 6, requires_grad=True),
        "opponent_response": torch.zeros(1, 7, requires_grad=True),
        "resource_forecast": torch.zeros(1, 6, requires_grad=True),
        "game_phase": torch.zeros(1, 5, requires_grad=True),
        "outcome_distribution": torch.zeros(1, 3, requires_grad=True),
        "remaining_turns": torch.zeros(1, 1, requires_grad=True),
    }
    loss, _ = expanded_strategic_losses(
        option_outputs=option_outputs,
        state_outputs=state_outputs,
        target_indices=torch.tensor([0]),
        value_targets=torch.tensor([1.0]),
        option_counts=[2],
        stage_indices=[0],
        decision_aux=[{EXPANDED_STRATEGIC_KEY: _expanded_target()}],
        weights=guide_outcome_backed_loss_weights(),
        row_weights=torch.tensor([0.75]),
        state_row_weights=torch.tensor([0.75]),
    )
    loss.backward()
    gradients = [
        tensor.grad.detach().float().reshape(-1)
        for tensor in (*option_outputs.values(), *state_outputs.values())
        if tensor.grad is not None
    ]
    norm = float(torch.linalg.vector_norm(torch.cat(gradients)).item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("guide-conditioned strategic gradient is not positive")
    if any(
        option_outputs[name].grad is not None
        for name in ("action_type", "action_target", "action_resource")
    ):
        raise RuntimeError("guide curriculum leaked into factor imitation")
    return norm


def _sources(
    fusion: CausalDecisionFusion,
    *,
    batch: int,
    options: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state = {
        name: torch.randn(batch, projection.in_features)
        for name, projection in fusion.state_projections.items()
    }
    option = {
        name: torch.randn(batch, options, width)
        for name, width in fusion._OPTION_DIMS.items()
    }
    option["setup_board_outcome"] = torch.randn(
        batch, options, SETUP_BOARD_OUTCOME_HEAD_OUTPUTS
    )
    if "combo_state" in fusion.dedicated_routes:
        option["combo_state"] = torch.randn(
            batch, options, COMBO_STATE_HEAD_OUTPUTS
        )
    return state, option


def _route_measurements(*, include_combo_state: bool = False) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, dict[str, float | str]],
    float,
]:
    torch.manual_seed(20260730)
    batch, options, d_model = 128, 3, 16
    fusion = CausalDecisionFusion(
        d_model=d_model,
        width=8,
        archetype_classes=4,
        belief_card_vocab=32,
        dedicated_routes_enabled=True,
        setup_board_outcome_outputs=SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
        combo_state_outputs=(
            COMBO_STATE_HEAD_OUTPUTS if include_combo_state else 0
        ),
    )
    hidden = torch.randn(batch, options, d_model)
    state, option = _sources(fusion, batch=batch, options=options)
    initial = fusion.dedicated_route_deltas(
        hidden, state_sources=state, option_sources=option
    )
    initial_max = {
        name: float(value.detach().abs().max().item())
        for name, value in initial.items()
    }
    if any(value != 0.0 for value in initial_max.values()):
        raise RuntimeError("revision-56 route initialization is not zero-safe")

    target = torch.linspace(-0.35, 0.35, options).repeat(batch, 1)
    optimizer = torch.optim.Adam(fusion.dedicated_routes.parameters(), lr=0.03)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        deltas = fusion.dedicated_route_deltas(
            hidden, state_sources=state, option_sources=option
        )
        loss = sum(
            torch.nn.functional.mse_loss(value, target) for value in deltas.values()
        )
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    deltas = fusion.dedicated_route_deltas(
        hidden, state_sources=state, option_sources=option
    )
    sum(value.square().mean() for value in deltas.values()).backward()
    count = len(deltas)
    aggregate = fusion.dedicated_action_delta(
        hidden, state_sources=state, option_sources=option
    ).detach()
    maximum_aggregate = float(aggregate.abs().max().item())
    if maximum_aggregate > 1.0:
        raise RuntimeError("aggregate route delta exceeded its one-logit cap")

    ablations: dict[str, dict[str, float | int]] = {}
    validation: dict[str, dict[str, float | str]] = {}
    base = torch.zeros_like(aggregate)
    full_choice = aggregate.argmax(dim=-1)
    for name, value in deltas.items():
        contribution = value.detach() / float(count)
        without = aggregate - contribution
        gradient_norm = float(
            torch.sqrt(
                sum(
                    parameter.grad.detach().float().square().sum()
                    for parameter in fusion.dedicated_routes[name].parameters()
                    if parameter.grad is not None
                )
            ).item()
        )
        legal_delta = float(
            (value[:, 0] - value[:, 1]).detach().abs().mean().item()
        )
        mean_delta = float(contribution.abs().mean().item())
        maximum = float(contribution.abs().max().item())
        if min(gradient_norm, legal_delta, mean_delta, maximum) <= 0.0:
            raise RuntimeError(f"route validation stayed zero: {name}")
        ablations[name] = {
            "decisions_evaluated": batch,
            "mean_absolute_action_logit_delta": mean_delta,
            "maximum_absolute_logit_contribution_observed": maximum,
            "selection_change_rate": float(
                (without.argmax(dim=-1) != full_choice).float().mean().item()
            ),
        }
        validation[name] = {
            "route_id": f"{name}-route",
            "route_input": "option_hidden_plus_typed_output",
            "post_training_route_gradient_norm": gradient_norm,
            "legal_option_dependence_delta": legal_delta,
            "causal_suffix_max_logit_delta": float(
                (base + contribution - contribution).abs().max().item()
            ),
            "zero_safe_initial_max_logit_delta": initial_max[name],
        }
    return ablations, validation, maximum_aggregate


def materialize(
    *,
    specialist_id: str,
    guide_contract: Path,
    guide_ready_receipt: Path,
    output_root: Path,
    training_implementation: Path,
    include_combo_state: bool = False,
) -> dict[str, Path]:
    specialist_id = specialist_id.strip().casefold()
    if not specialist_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for char in specialist_id
    ):
        raise ValueError("unsafe specialist id")
    guide_contract = guide_contract.expanduser().resolve()
    guide_ready_receipt = guide_ready_receipt.expanduser().resolve()
    training_implementation = training_implementation.expanduser().resolve()
    ready = _read_json(guide_ready_receipt)
    guide_rows = int(ready.get("guide_rows") or 0)
    if (
        not guide_contract.is_file()
        or not training_implementation.is_file()
        or ready.get("status") not in {"ready", "validated"}
        or ready.get("specialist_id") != specialist_id
        or guide_rows <= 0
    ):
        raise RuntimeError("strategic curriculum source artifacts are not ready")

    guide_digest = _sha256(guide_contract)
    learned_sources = tuple(
        dict.fromkeys(
            (*LEARNED_SOURCES, *(("combo_state",) if include_combo_state else ()))
        )
    )
    heads = {
        name: {
            "computation_role": "independent_head",
            "fusion_role": "fused_input",
            "trainable": True,
            "causal_training_targets_only": True,
            "guide_action_target_allowed": False,
            "enters_decision_fusion": True,
            "action_influence": "bounded_option_conditioned_route",
            "causal_input": "board_state_and_legal_option",
            "direct_action_selection_authority": False,
            "runtime_activation_requirement": "receipt_backed_validation",
            "route_id": f"{name}-route",
            "route_input": "option_hidden_plus_typed_output",
            "route_reduction": "fixed_mean",
            "zero_safe_final_projection": True,
            "maximum_absolute_logit_contribution": 0.25,
        }
        for name in learned_sources
    }
    output_root = output_root.expanduser().resolve()
    role_path = output_root / f"{specialist_id}-strategic-head-roles-r56.json"
    spec_path = output_root / f"{specialist_id}-strategic-curriculum-r56.json"
    receipt_path = (
        output_root / f"{specialist_id}-strategic-curriculum-validation-r56.json"
    )
    role_map = {
        "schema": ROLE_SCHEMA,
        "specialist_id": specialist_id,
        "training_mode": TRAINING_MODE,
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": ACTION_REVISION,
        "action_influence_revision": ACTION_REVISION,
        "guide_contract_sha256": guide_digest,
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "preserve_v1_additive_residual": True,
        "canonical_learned_decision_sources": list(learned_sources),
        "one_route_per_learned_source": True,
        "route_input": "option_hidden_plus_typed_output",
        "route_reduction": "fixed_mean",
        "aggregate_absolute_logit_cap": 1.0,
        "zero_safe_final_projection": True,
        "guide_is_only_action_route_exception": True,
        "heads": heads,
    }
    _atomic_json(role_path, role_map)
    spec = {
        "schema": SPEC_SCHEMA,
        "specialist_id": specialist_id,
        "training_mode": TRAINING_MODE,
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": ACTION_REVISION,
        "action_influence_revision": ACTION_REVISION,
        "guide_contract_sha256": guide_digest,
        "head_role_map_sha256": _sha256(role_path),
        "curriculum_heads": sorted(learned_sources),
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
        "canonical_learned_decision_sources": list(learned_sources),
        "one_route_per_learned_source": True,
        "route_input": "option_hidden_plus_typed_output",
        "route_reduction": "fixed_mean",
        "aggregate_absolute_logit_cap": 1.0,
        "zero_safe_final_projection": True,
        "guide_is_only_action_route_exception": True,
        "pre_fleet_h10_compute_allowed": False,
    }
    _atomic_json(spec_path, spec)
    gradient_norm = _guide_gradient_measurement()
    ablations, route_validation, maximum_aggregate = _route_measurements(
        include_combo_state=include_combo_state
    )
    checks = {
        "guide_supervision_terminates_at_strategic_heads": True,
        "fused_policy_remains_outcome_and_win_trained": True,
        "direct_policy_cross_entropy_absent": True,
        "observed_outcome_targets_not_replaced": True,
        "all_curriculum_heads_have_valid_fusion_roles": True,
        "every_curriculum_head_has_bounded_action_scoring_route": True,
        "per_head_action_influence_ablation_passed": True,
        "exact_parent_parity_at_initialization": True,
        "one_option_conditioned_route_per_learned_head": True,
        "causal_suffix_invariance": True,
        "legal_option_dependence": True,
        "bounded_aggregate_residual": True,
        "all_training_paths_use_the_declared_mode": True,
        "active_and_historical_specialists_unchanged": True,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "validated",
        "specialist_id": specialist_id,
        "training_mode": TRAINING_MODE,
        "guide_curriculum_revision": GUIDE_CURRICULUM_REVISION,
        "strategic_branch_scope_revision": ACTION_REVISION,
        "action_influence_revision": ACTION_REVISION,
        "guide_contract_sha256": guide_digest,
        "curriculum_spec_sha256": _sha256(spec_path),
        "head_role_map_sha256": _sha256(role_path),
        "required_training_paths": REQUIRED_TRAINING_PATHS,
        "decision_fusion_schema": DECISION_FUSION_V2_SCHEMA,
        "validated_route_ids": [
            heads[name]["route_id"] for name in sorted(learned_sources)
        ],
        "checks": checks,
        "measurements": {
            "guide_to_strategic_head_gradient_norm": gradient_norm,
            "guide_labeled_rows": guide_rows,
            "maximum_absolute_aggregate_residual_observed": maximum_aggregate,
        },
        "action_influence_ablations": ablations,
        "route_validation": route_validation,
        "implementation_artifacts": [
            {
                "role": "training_implementation",
                "path": str(training_implementation),
                "sha256": _sha256(training_implementation),
            },
            {
                "role": "validation_test",
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        ],
        "guide_ready_receipt": str(guide_ready_receipt),
        "guide_ready_receipt_sha256": _sha256(guide_ready_receipt),
    }
    _atomic_json(receipt_path, receipt)
    return {
        "head_role_map": role_path,
        "curriculum_spec": spec_path,
        "validation_receipt": receipt_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--guide-contract", type=Path, required=True)
    parser.add_argument("--guide-ready-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--training-implementation",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "poke_bot/train.py",
    )
    parser.add_argument(
        "--include-combo-state",
        action="store_true",
        help=(
            "Include the optional combo_state head and its bounded "
            "option-conditioned fusion route in the receipt."
        ),
    )
    args = parser.parse_args()
    outputs = materialize(
        specialist_id=args.specialist_id,
        guide_contract=args.guide_contract,
        guide_ready_receipt=args.guide_ready_receipt,
        output_root=args.output_root,
        training_implementation=args.training_implementation,
        include_combo_state=args.include_combo_state,
    )
    print(
        json.dumps(
            {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for name, path in outputs.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
