#!/usr/bin/env python3
"""Validate the Slowking combo-head implementation without authorizing training.

This is an architecture/gradient receipt only. Exact Slowking corpus coverage,
typed calibration, resource measurements, package compatibility, and ablation
gain remain separate fail-closed requirements before the final validation
receipt can become ``validated``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from poke_bot.combo_state import VECTOR_WIDTH, combo_state_loss
from poke_bot.model import (
    COMBO_STATE_HEAD_OUTPUTS,
    CausalDecisionFusion,
    ComboStateHead,
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_V2_TOTAL_DELTA_CAP,
    SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(
    fusion: CausalDecisionFusion,
    *,
    batch: int,
    options: int,
    combo: torch.Tensor,
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
    option["combo_state"] = combo
    return state, option


def validate() -> dict[str, Any]:
    torch.manual_seed(20260730)
    d_model = 96
    batch = 8
    options = 3
    head = ComboStateHead(d_model)
    fusion = CausalDecisionFusion(
        d_model=d_model,
        width=16,
        archetype_classes=20,
        belief_card_vocab=64,
        dedicated_routes_enabled=True,
        setup_board_outcome_outputs=SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
        combo_state_outputs=COMBO_STATE_HEAD_OUTPUTS,
    )
    route = fusion.dedicated_routes["combo_state"]
    option_hidden = torch.randn(batch, options, d_model)
    base_logits = torch.randn(batch, options)
    combo = head(option_hidden)
    state_sources, option_sources = _sources(
        fusion, batch=batch, options=options, combo=combo
    )
    step_zero = fusion(
        option_hidden,
        base_logits,
        state_sources=state_sources,
        option_sources=option_sources,
        dedicated_routes_active=True,
    )
    step_zero_parity = torch.equal(step_zero, base_logits)

    selected = torch.arange(batch) % options
    combo_loss, combo_metrics = combo_state_loss(
        predictions=combo,
        selected_indices=selected,
        option_counts=torch.full((batch,), options),
        top_deck_targets=torch.arange(batch) % 5,
        top_deck_masks=torch.ones(batch, dtype=torch.bool),
        seek_source_targets=torch.arange(batch) % 7,
        seek_source_masks=torch.ones(batch, dtype=torch.bool),
        vector_targets=torch.randint(0, 2, (batch, VECTOR_WIDTH)).float(),
        vector_masks=torch.ones(batch, VECTOR_WIDTH, dtype=torch.bool),
        guide_confidences=torch.linspace(0.0, 1.0, batch),
        base_loss_weight=0.025,
        guide_loss_weight=0.05,
    )
    optimizer = torch.optim.Adam(
        [*head.parameters(), *fusion.dedicated_routes.parameters()],
        lr=0.02,
    )
    policy_target = (selected + 1) % options
    optimizer.zero_grad(set_to_none=True)
    first_policy = F.cross_entropy(step_zero, policy_target)
    (first_policy + combo_loss).backward()
    head_gradient = sum(
        float(parameter.grad.norm())
        for parameter in head.parameters()
        if parameter.grad is not None
    )
    route_gradient = sum(
        float(parameter.grad.norm())
        for parameter in route.parameters()
        if parameter.grad is not None
    )
    before_head = {
        name: tensor.detach().clone() for name, tensor in head.state_dict().items()
    }
    before_route = {
        name: tensor.detach().clone() for name, tensor in route.state_dict().items()
    }
    optimizer.step()

    # A second step allows policy gradients to flow through the now-nonzero
    # zero-safe final projection while retaining the typed observed loss.
    optimizer.zero_grad(set_to_none=True)
    combo = head(option_hidden)
    state_sources, option_sources = _sources(
        fusion, batch=batch, options=options, combo=combo
    )
    fused = fusion(
        option_hidden,
        base_logits,
        state_sources=state_sources,
        option_sources=option_sources,
        dedicated_routes_active=True,
    )
    second_policy = F.cross_entropy(fused, policy_target)
    second_policy.backward()
    routed_head_gradient = sum(
        float(parameter.grad.norm())
        for parameter in head.parameters()
        if parameter.grad is not None
    )
    routed_route_gradient = sum(
        float(parameter.grad.norm())
        for parameter in route.parameters()
        if parameter.grad is not None
    )

    deltas = fusion.dedicated_route_deltas(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    combo_delta = deltas["combo_state"]
    all_delta = fusion.dedicated_action_delta(
        option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
    )
    without_combo = {
        name: value for name, value in deltas.items() if name != "combo_state"
    }
    without_combo_delta = (
        DECISION_FUSION_V2_TOTAL_DELTA_CAP
        * torch.stack(tuple(without_combo.values())).mean(dim=0)
    )
    parameter_inventory = {
        "combo_state_head.network": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "decision_fusion.dedicated_routes.combo_state": sum(
            parameter.numel() for parameter in route.parameters()
        ),
    }
    checks = {
        "implementation_importable": True,
        "exact_output_width": COMBO_STATE_HEAD_OUTPUTS == 32,
        "exact_module_parameter_inventory": parameter_inventory
        == {
            "combo_state_head.network": 24800,
            "decision_fusion.dedicated_routes.combo_state": 2081,
        },
        "step_zero_parent_parity": step_zero_parity,
        "finite_nonzero_head_gradient": head_gradient > 0.0,
        "finite_nonzero_route_gradient": route_gradient > 0.0,
        "finite_nonzero_routed_head_gradient": routed_head_gradient > 0.0,
        "finite_nonzero_routed_route_gradient": routed_route_gradient > 0.0,
        "head_parameters_updated": any(
            not torch.equal(before_head[name], tensor)
            for name, tensor in head.state_dict().items()
        ),
        "route_parameters_updated": any(
            not torch.equal(before_route[name], tensor)
            for name, tensor in route.state_dict().items()
        ),
        "legal_option_dependence": bool(
            torch.any(combo[:, 0] != combo[:, 1]).item()
        ),
        "bounded_route_output": bool(
            torch.isfinite(all_delta).all()
            and all_delta.abs().max() <= DECISION_FUSION_V2_TOTAL_DELTA_CAP
        ),
        "nonzero_combo_leave_one_out_logit_attribution": bool(
            torch.any(
                (all_delta - without_combo_delta).abs() > 0.0
            ).item()
        ),
        "guide_preferred_action_not_consumed": (
            "guide_preferred_indices"
            not in combo_state_loss.__code__.co_varnames
        ),
    }
    source_files = [
        Path("poke_bot/model.py"),
        Path("poke_bot/combo_state.py"),
        Path("poke_bot/config.py"),
        Path("scripts/run_starmie_expert_bootstrap.py"),
    ]
    return {
        "schema": "poke_bot.slowking_combo_head_implementation_validation/v1",
        "specialist_id": "slowking",
        "status": (
            "implementation_validated_data_pending"
            if all(checks.values())
            else "implementation_validation_failed"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": [d_model, 192, 32],
        "fusion_route": "decision_fusion.dedicated_routes.combo_state",
        "required_heads": [
            *DECISION_FUSION_REQUIRED_HEADS,
            "setup_board_outcome",
            "combo_state",
        ],
        "parameter_inventory": parameter_inventory,
        "metrics": {
            "typed_loss": float(combo_loss.detach()),
            "typed_loss_coverage": combo_metrics.as_dict(),
            "head_gradient_norm": head_gradient,
            "route_gradient_norm": route_gradient,
            "routed_head_gradient_norm": routed_head_gradient,
            "routed_route_gradient_norm": routed_route_gradient,
            "combo_route_max_abs_delta": float(combo_delta.abs().max().detach()),
            "aggregate_route_max_abs_delta": float(all_delta.abs().max().detach()),
        },
        "checks": checks,
        "source_files": [
            {"path": str(path), "sha256": _digest(path)}
            for path in source_files
        ],
        "not_yet_proved": [
            "exact_deck_binding",
            "labeled_row_coverage_by_requirement",
            "loss_and_calibration_by_typed_output_on_slowking_rows",
            "causal_suffix_invariance_on_replay_rows",
            "leave_one_head_out_action_attribution_on_slowking_states",
            "complete_candidate_parameter_count",
            "peak_memory_and_latency",
            "package_size_and_final_submission_compatibility",
            "unused_capacity_rejected_by_ablation",
        ],
        "training_ready": False,
        "launch_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "implementation_validated_data_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
