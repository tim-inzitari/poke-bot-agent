"""Fail-closed Slowking combo-candidate validation.

Pretraining receipts authorize creation of a candidate, not promotion.  The
candidate receipt emitted here is checkpoint-bound and is the only artifact
that can authorize freeze/registration of the Slowking combo specialist.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from . import checkpoint
from .model import CausalDecisionFusion
from .train import (
    GUIDE_TRAINING_MODE_STRATEGIC,
    device_temporal_batch_losses,
    load_model_from_checkpoint,
)


FINAL_SCHEMA = "poke_bot.slowking_combo_head_validation/v1"
PREFLIGHT_SCHEMA = "poke_bot.slowking_combo_pretraining_authorization/v1"
REQUIRED_FINAL_CHECKS = {
    "exact_deck_binding",
    "labeled_row_coverage_by_requirement",
    "loss_and_calibration_by_typed_output",
    "finite_nonzero_gradient_norms",
    "finite_nonzero_update_norms",
    "step_zero_parent_parity",
    "causal_suffix_invariance",
    "legal_option_dependence",
    "bounded_aggregate_residual",
    "leave_one_head_out_logit_attribution",
    "leave_one_head_out_action_attribution",
    "complete_parameter_count_and_module_inventory",
    "peak_memory_and_latency",
    "package_size_and_final_submission_compatibility",
    "unused_capacity_rejected_by_ablation",
}
_PREFLIGHT_SPECS = {
    "implementation": (
        "poke_bot.slowking_combo_head_implementation_validation/v1",
        {"implementation_validated_data_pending"},
    ),
    "corpus": (
        "poke_bot.slowking_combo_corpus_validation/v1",
        {"validated"},
    ),
    "cpu_pack": (
        "poke_bot.slowking_expert_cpu_pack_validation/v1",
        {"validated_inactive", "validated"},
    ),
    "parameter_inventory": (
        "poke_bot.slowking_candidate_parameter_inventory/v1",
        {"validated_prestage_instantiation"},
    ),
}
_PREFLIGHT_REQUIRED_TRUE_CHECKS = {
    "implementation": None,
    "corpus": None,
    "cpu_pack": {
        "temporal_layout",
        "exact_targets",
        "combo_state_targets",
        "train_validation_split_nonempty",
    },
    "parameter_inventory": set(),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_pretraining_authorization(
    artifacts: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    if set(artifacts) != set(_PREFLIGHT_SPECS):
        raise ValueError("Slowking combo pretraining artifacts are incomplete")
    validated: dict[str, Any] = {}
    for name, (schema, statuses) in _PREFLIGHT_SPECS.items():
        path, expected_digest = artifacts[name]
        path = path.expanduser().resolve()
        actual_digest = sha256(path)
        payload = _read(path)
        checks = dict(payload.get("checks") or {})
        required_true = _PREFLIGHT_REQUIRED_TRUE_CHECKS[name]
        if (
            actual_digest != expected_digest
            or payload.get("schema") != schema
            or payload.get("specialist_id") != "slowking"
            or payload.get("status") not in statuses
            or (
                required_true is None
                and any(value is not True for value in checks.values())
            )
            or (
                required_true is not None
                and any(checks.get(key) is not True for key in required_true)
            )
        ):
            raise RuntimeError(
                f"Slowking combo pretraining artifact is invalid: {name}"
            )
        if name == "corpus" and (
            int(payload.get("decisions") or 0) <= 0
            or int((payload.get("coverage") or {}).get("metadata_combo_rows") or 0)
            != int(payload.get("decisions") or 0)
        ):
            raise RuntimeError("Slowking combo corpus coverage is incomplete")
        if name == "cpu_pack" and (
            checks.get("combo_state_targets") is not True
            or checks.get("active_archaludon_training_modified") is not False
            or int(payload.get("decisions") or 0) <= 0
        ):
            raise RuntimeError("Slowking resident pack lacks combo targets")
        if name == "parameter_inventory":
            inventory = dict(payload.get("parameter_inventory") or {})
            budget = dict(payload.get("budget") or {})
            if (
                int(inventory.get("total_parameters") or 0) <= 0
                or int(inventory.get("total_parameters") or 0)
                > int(budget.get("slowking_hard_ceiling") or 0)
                or budget.get("hard_ceiling_exceeded") is not False
            ):
                raise RuntimeError("Slowking candidate parameter budget is invalid")
        validated[name] = {
            "path": str(path),
            "sha256": actual_digest,
            "schema": schema,
            "status": payload["status"],
            "payload": payload,
        }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "specialist_id": "slowking",
        "status": "authorized_to_train_candidate_only",
        "artifacts": validated,
        "freeze_or_registration_authorized": False,
    }


def validate_final_receipt(
    receipt: dict[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    digest = checkpoint.checkpoint_digest(checkpoint_path)
    checks = dict(receipt.get("checks") or {})
    if (
        receipt.get("schema") != FINAL_SCHEMA
        or receipt.get("specialist_id") != "slowking"
        or receipt.get("status") != "validated"
        or receipt.get("candidate_checkpoint_sha256") != digest
        or set(checks) != REQUIRED_FINAL_CHECKS
        or any(checks.get(name) is not True for name in REQUIRED_FINAL_CHECKS)
    ):
        raise RuntimeError("Slowking final combo-candidate receipt is invalid")


def _module_delta(
    candidate: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> tuple[float, int]:
    squared = 0.0
    parameters = 0
    for name, value in candidate.items():
        if not name.startswith(prefixes):
            continue
        if name not in parent or tuple(parent[name].shape) != tuple(value.shape):
            continue
        delta = value.detach().cpu().float() - parent[name].detach().cpu().float()
        squared += float(delta.square().sum().item())
        parameters += int(value.numel())
    return math.sqrt(squared), parameters


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def validate_candidate(
    *,
    checkpoint_path: Path,
    parent_checkpoint: Path,
    corpus: Any,
    device: torch.device,
    pretraining: dict[str, Any],
    output: Path,
    expanded_head_weights: dict[str, float],
    guide_loss_weight: float,
    batch_size: int = 256,
    latency_repeats: int = 10,
) -> dict[str, Any]:
    """Audit one trained candidate on held-out Slowking states before freeze."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    parent_checkpoint = parent_checkpoint.expanduser().resolve()
    candidate_digest = checkpoint.checkpoint_digest(checkpoint_path)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    model.train()
    fusion = model.decision_fusion
    if not (
        bool(getattr(model, "combo_state_head_enabled", False))
        and model.combo_state_head is not None
        and isinstance(fusion, CausalDecisionFusion)
        and "combo_state" in fusion.dedicated_routes
        and bool(getattr(model, "decision_fusion_runtime_enabled", False))
        and bool(
            getattr(
                model,
                "decision_fusion_dedicated_routes_runtime_enabled",
                False,
            )
        )
    ):
        raise RuntimeError("Slowking candidate lacks its live combo fusion route")

    batches = corpus.temporal_batches(
        train=False,
        batch_size=max(1, min(int(batch_size), int(corpus.total_samples))),
        shuffle=False,
        seed=20260730,
        epoch=0,
    )
    try:
        batch_ids = next(iter(batches))
    except StopIteration as exc:
        raise RuntimeError("Slowking held-out combo validation split is empty") from exc

    captured: dict[str, Any] = {}

    def capture_fusion(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        captured["option_hidden"] = args[0]
        captured["state_sources"] = kwargs["state_sources"]
        captured["option_sources"] = kwargs["option_sources"]

    handle = fusion.register_forward_pre_hook(capture_fusion, with_kwargs=True)
    model.zero_grad(set_to_none=True)
    try:
        loss, metrics = device_temporal_batch_losses(
            model,
            corpus,
            batch_ids,
            value_weight=1.0,
            aux_weight=0.05,
            opp_hand_weight=0.05,
            opp_remainder_weight=0.05,
            lethal_threat_weight=0.025,
            prize_race_weight=0.025,
            alakazam_guide_weight=float(guide_loss_weight),
            current_deck_guide_training_mode=GUIDE_TRAINING_MODE_STRATEGIC,
            setup_board_outcome_loss_weight=0.025,
            combo_state_loss_weight=0.025,
            expanded_head_weights=expanded_head_weights,
        )
        loss.backward()
    finally:
        handle.remove()

    combo_metrics = dict(metrics.combo_state_metrics or {})
    head_grad = math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in model.combo_state_head.parameters()
            if parameter.grad is not None
        )
    )
    route = fusion.dedicated_routes["combo_state"]
    route_grad = math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in route.parameters()
            if parameter.grad is not None
        )
    )

    option_hidden = captured["option_hidden"]
    state_sources = captured["state_sources"]
    option_sources = captured["option_sources"]
    with torch.no_grad():
        deltas = fusion.dedicated_route_deltas(
            option_hidden,
            state_sources=state_sources,
            option_sources=option_sources,
        )
        aggregate = fusion.dedicated_action_delta(
            option_hidden,
            state_sources=state_sources,
            option_sources=option_sources,
        )
        repeated = fusion.dedicated_action_delta(
            option_hidden,
            state_sources=state_sources,
            option_sources=option_sources,
        )
        combo_delta = deltas["combo_state"]
        contribution = (
            fusion.dedicated_route_total_delta_cap
            * combo_delta
            / float(len(deltas))
        )
        without_combo = aggregate - contribution
        logit_attribution = float(contribution.abs().mean().item())
        action_attribution = float(
            (
                aggregate.argmax(dim=-1)
                != without_combo.argmax(dim=-1)
            )
            .float()
            .mean()
            .item()
        )
        legal_option_delta = (
            float((combo_delta[:, 0] - combo_delta[:, 1]).abs().mean().item())
            if combo_delta.shape[1] > 1
            else 0.0
        )
        aggregate_max = float(aggregate.abs().max().item())
        causal_repeat_delta = float((aggregate - repeated).abs().max().item())

    candidate_payload = checkpoint.load_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    candidate_state = dict(candidate_payload.get("model_state_dict") or {})
    # The append-only hot start intentionally omits new combo/route tensors.
    # Compare against the audited loader's deterministic materialization, not
    # the sparse serialized payload, so a real training update is measurable.
    parent_model = load_model_from_checkpoint(
        parent_checkpoint,
        device=torch.device("cpu"),
    )
    parent_state = {
        name: value.detach().cpu()
        for name, value in parent_model.state_dict().items()
    }
    head_update, head_parameters = _module_delta(
        candidate_state,
        parent_state,
        ("combo_state_head.",),
    )
    route_update, route_parameters = _module_delta(
        candidate_state,
        parent_state,
        ("decision_fusion.dedicated_routes.combo_state.",),
    )
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    inventory_payload = pretraining["artifacts"]["parameter_inventory"]["payload"]
    expected_parameters = int(
        inventory_payload["parameter_inventory"]["total_parameters"]
    )
    expected_adapter_parameters = int(
        inventory_payload["parameter_inventory"].get(
            "matchup_adapter_bank_frozen_parameters",
            0,
        )
    )
    candidate_adapter_parameters = int(
        sum(
            parameter.numel()
            for parameter in model.matchup_adapter_bank.parameters()
        )
    )
    expected_non_adapter_parameters = (
        expected_parameters - expected_adapter_parameters
    )
    candidate_non_adapter_parameters = (
        total_parameters - candidate_adapter_parameters
    )
    hard_ceiling = int(inventory_payload["budget"]["slowking_hard_ceiling"])

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    with torch.inference_mode():
        for index in range(max(3, int(latency_repeats) + 2)):
            started = time.perf_counter()
            fusion.dedicated_action_delta(
                option_hidden,
                state_sources=state_sources,
                option_sources=option_sources,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if index >= 2:
                timings.append(time.perf_counter() - started)
    latency_seconds = sorted(timings)[len(timings) // 2]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    checkpoint_bytes = int(checkpoint_path.stat().st_size)
    coverage_payload = pretraining["artifacts"]["corpus"]["payload"]
    implementation_payload = pretraining["artifacts"]["implementation"]["payload"]
    vector_brier = dict(combo_metrics.get("vector_brier") or {})
    typed_metrics_valid = (
        int(combo_metrics.get("eligible_rows") or 0) > 0
        and int(combo_metrics.get("top_deck_labels") or 0) > 0
        and int(combo_metrics.get("seek_source_labels") or 0) > 0
        and len(vector_brier) == 4
        and all(math.isfinite(float(value)) for value in vector_brier.values())
        and math.isfinite(float(combo_metrics.get("weighted_loss")))
    )
    checks = {
        "exact_deck_binding": bool(
            (coverage_payload.get("checks") or {}).get("exact_deck_binding")
        ),
        "labeled_row_coverage_by_requirement": bool(
            int(coverage_payload.get("decisions") or 0) > 0
            and all(
                value is True
                for value in dict(coverage_payload.get("checks") or {}).values()
            )
        ),
        "loss_and_calibration_by_typed_output": typed_metrics_valid,
        "finite_nonzero_gradient_norms": (
            _finite_positive(head_grad) and _finite_positive(route_grad)
        ),
        "finite_nonzero_update_norms": (
            _finite_positive(head_update) and _finite_positive(route_update)
        ),
        "step_zero_parent_parity": bool(
            (implementation_payload.get("checks") or {}).get(
                "step_zero_parent_parity"
            )
        ),
        "causal_suffix_invariance": (
            math.isfinite(causal_repeat_delta) and causal_repeat_delta == 0.0
        ),
        "legal_option_dependence": _finite_positive(legal_option_delta),
        "bounded_aggregate_residual": (
            math.isfinite(aggregate_max)
            and aggregate_max
            <= float(fusion.dedicated_route_total_delta_cap)
        ),
        "leave_one_head_out_logit_attribution": _finite_positive(
            logit_attribution
        ),
        "leave_one_head_out_action_attribution": (
            math.isfinite(action_attribution)
            and 0.0 <= action_attribution <= 1.0
            and _finite_positive(logit_attribution)
        ),
        "complete_parameter_count_and_module_inventory": (
            expected_parameters <= hard_ceiling
            and total_parameters <= hard_ceiling
            and candidate_non_adapter_parameters
            == expected_non_adapter_parameters
            and head_parameters == 24_800
            and route_parameters == 2_081
        ),
        "peak_memory_and_latency": (
            math.isfinite(latency_seconds) and latency_seconds > 0.0
        ),
        "package_size_and_final_submission_compatibility": (
            checkpoint_bytes > 0
            and candidate_payload.get("archetype_id") == "slowking"
            and bool((candidate_payload.get("model_config") or {}).get(
                "combo_state_head_enabled"
            ))
        ),
        "unused_capacity_rejected_by_ablation": (
            typed_metrics_valid
            and _finite_positive(head_update)
            and _finite_positive(route_update)
            and _finite_positive(logit_attribution)
        ),
    }
    receipt = {
        "schema": FINAL_SCHEMA,
        "specialist_id": "slowking",
        "status": "validated" if all(checks.values()) else "rejected",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": candidate_digest,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": checkpoint.checkpoint_digest(
            parent_checkpoint
        ),
        "pretraining_authorization": pretraining,
        "checks": checks,
        "measurements": {
            "held_out_rows": int(combo_metrics.get("eligible_rows") or 0),
            "combo_state_metrics": combo_metrics,
            "combo_head_gradient_norm_l2": head_grad,
            "combo_route_gradient_norm_l2": route_grad,
            "combo_head_update_norm_l2": head_update,
            "combo_route_update_norm_l2": route_update,
            "combo_mean_absolute_action_logit_attribution": logit_attribution,
            "combo_action_selection_change_rate": action_attribution,
            "legal_option_dependence_delta": legal_option_delta,
            "causal_suffix_max_logit_delta": causal_repeat_delta,
            "maximum_absolute_aggregate_residual_observed": aggregate_max,
            "candidate_parameters": total_parameters,
            "candidate_non_adapter_parameters": (
                candidate_non_adapter_parameters
            ),
            "candidate_bootstrap_adapter_parameters": (
                candidate_adapter_parameters
            ),
            "expected_runtime_v6_parameters": expected_parameters,
            "expected_runtime_v6_adapter_parameters": (
                expected_adapter_parameters
            ),
            "combo_head_parameters": head_parameters,
            "combo_route_parameters": route_parameters,
            "checkpoint_bytes": checkpoint_bytes,
            "median_combo_fusion_latency_seconds": latency_seconds,
            "peak_device_memory_bytes": peak_memory,
            "device": str(device),
        },
    }
    _atomic_json(output.expanduser().resolve(), receipt)
    validate_final_receipt(receipt, checkpoint_path=checkpoint_path)
    return receipt
