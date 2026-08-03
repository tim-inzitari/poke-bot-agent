"""Masked losses for the additive V6 strategic-head targets.

This layer is intentionally independent from target derivation and model
construction.  It consumes the validated, training-only target mapping stored
under ``DecisionSample.aux_labels["expanded_strategic"]`` and never treats a
missing value as a numerical zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .strategic_schedule import EXPANDED_HEAD_IDS, canonical_head_id


GUIDE_OUTCOME_BACKED_HEAD_IDS: tuple[str, ...] = (
    "action_q",
    "action_utility",
    "tactical_outcome",
    "opponent_response",
    "resource_forecast",
    "outcome_distribution",
    "remaining_turns",
)

GUIDE_DIRECTIONAL_ROUTE_HEAD_IDS: tuple[str, ...] = (
    "action_q",
    "action_resource",
    "action_utility",
    "setup_board_outcome",
    "combo_state",
)


def guide_pairwise_route_ranking_loss(
    *,
    route_deltas: Mapping[str, torch.Tensor],
    guide_target_indices: torch.Tensor,
    guide_confidences: torch.Tensor,
    option_counts: torch.Tensor,
    margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Guide selected causal routes without supervising final policy logits.

    Each eligible typed route is asked only to rank the guide-preferred legal
    option above the mean alternative by ``margin``. The guide remains absent
    from serving and does not replace observed strategic targets.
    """

    available = [
        name for name in GUIDE_DIRECTIONAL_ROUTE_HEAD_IDS if name in route_deltas
    ]
    if not available:
        if route_deltas:
            zero = next(iter(route_deltas.values())).sum() * 0.0
        else:
            zero = guide_confidences.sum() * 0.0
        return zero, {"eligible_rows": 0, "heads": {}, "margin": float(margin)}
    reference = route_deltas[available[0]]
    if reference.dim() != 2:
        raise ValueError("guide route delta must be [rows, options]")
    rows, max_options = reference.shape
    targets = guide_target_indices.reshape(-1).to(device=reference.device, dtype=torch.long)
    confidence = guide_confidences.reshape(-1).to(device=reference.device, dtype=reference.dtype)
    counts = option_counts.reshape(-1).to(device=reference.device, dtype=torch.long)
    if not (targets.numel() == confidence.numel() == counts.numel() == rows):
        raise ValueError("guide route-ranking rows do not align")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("guide route-ranking margin must be finite and nonnegative")
    if bool((counts <= 0).any()) or bool((counts > max_options).any()):
        raise ValueError("guide route-ranking option count is invalid")
    if not bool(torch.isfinite(confidence).all()) or bool(
        ((confidence < 0.0) | (confidence > 1.0)).any()
    ):
        raise ValueError("guide route-ranking confidence is outside [0, 1]")
    # A legal state can expose only one action (including forced/no-choice
    # transitions).  It has no alternative to rank, so it is a valid masked
    # row rather than malformed training data.
    eligible = (
        (counts > 1)
        & (targets >= 0)
        & (targets < counts)
        & confidence.gt(0.0)
    )
    if not bool(eligible.any()):
        return reference.sum() * 0.0, {
            "eligible_rows": 0,
            "heads": {name: 0.0 for name in available},
            "margin": float(margin),
        }
    row_ids = torch.nonzero(eligible, as_tuple=False).flatten()
    local_targets = targets.index_select(0, row_ids)
    local_counts = counts.index_select(0, row_ids)
    local_confidence = confidence.index_select(0, row_ids)
    option_ids = torch.arange(max_options, device=reference.device).unsqueeze(0)
    legal = option_ids < local_counts.unsqueeze(1)
    preferred_mask = option_ids.eq(local_targets.unsqueeze(1))
    alternative_mask = legal & ~preferred_mask
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for name in available:
        scores = route_deltas[name]
        if scores.shape != reference.shape:
            raise ValueError(f"guide route delta shape mismatch: {name}")
        selected_scores = scores.index_select(0, row_ids)
        preferred = selected_scores.gather(1, local_targets.unsqueeze(1)).squeeze(1)
        alternative = (
            selected_scores.masked_fill(~alternative_mask, 0.0).sum(dim=1)
            / alternative_mask.sum(dim=1).clamp_min(1)
        )
        row_loss = F.softplus(float(margin) - (preferred - alternative))
        loss = (row_loss * local_confidence).sum() / local_confidence.sum().clamp_min(1e-12)
        losses.append(loss)
        metrics[name] = float(loss.detach().item())
    total = torch.stack(losses).mean()
    return total, {
        "eligible_rows": int(row_ids.numel()),
        "heads": metrics,
        "margin": float(margin),
    }


def guide_outcome_backed_loss_weights() -> dict[str, float]:
    """Equal-weight the outcome-backed heads used by future guide curricula."""

    weight = 1.0 / len(GUIDE_OUTCOME_BACKED_HEAD_IDS)
    return {
        name: weight if name in GUIDE_OUTCOME_BACKED_HEAD_IDS else 0.0
        for name in EXPANDED_HEAD_IDS
    }


@dataclass
class ExpandedStrategicLossMetrics:
    """Per-head loss/coverage plus outcome calibration diagnostics."""

    losses: dict[str, float] = field(default_factory=dict)
    labeled: dict[str, int] = field(default_factory=dict)
    total: dict[str, int] = field(default_factory=dict)
    masked: dict[str, int] = field(default_factory=dict)
    outcome_brier: float | None = None
    outcome_ece: float | None = None
    outcome_entropy: float | None = None

    def as_dict(self) -> dict[str, Any]:
        coverage = {
            name: (
                float(self.labeled.get(name, 0))
                / max(int(self.total.get(name, 0)), 1)
            )
            for name in EXPANDED_HEAD_IDS
        }
        return {
            "losses": dict(self.losses),
            "labeled": dict(self.labeled),
            "masked": dict(self.masked),
            "total": dict(self.total),
            "coverage": coverage,
            "calibration": {
                "outcome_brier": self.outcome_brier,
                "outcome_ece": self.outcome_ece,
                "outcome_entropy": self.outcome_entropy,
            },
        }


def canonical_expanded_loss_weights(
    weights: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return all eleven weights, rejecting unknown/invalid entries."""

    result = {name: 0.0 for name in EXPANDED_HEAD_IDS}
    for raw_name, raw_value in dict(weights or {}).items():
        name = canonical_head_id(raw_name)
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"expanded strategic loss weight is invalid: {name}")
        result[name] = value
    return result


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"present expanded target {field_name} must be a mapping")
    return dict(value)


def _vector(
    row: Mapping[str, Any],
    *,
    field_name: str,
    width: int,
) -> tuple[list[float], list[bool]]:
    values = row.get("values")
    mask = row.get("mask")
    if not isinstance(values, list) or not isinstance(mask, list):
        raise ValueError(f"present expanded target {field_name} lacks values/mask")
    if len(values) != width or len(mask) != width:
        raise ValueError(
            f"expanded target {field_name} width mismatch: "
            f"values={len(values)} mask={len(mask)} expected={width}"
        )
    parsed_values: list[float] = []
    parsed_mask: list[bool] = []
    for index, (raw_value, raw_mask) in enumerate(zip(values, mask)):
        present = bool(raw_mask)
        parsed_mask.append(present)
        if not present:
            parsed_values.append(0.0)
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"expanded target {field_name}[{index}] is non-finite"
            )
        parsed_values.append(value)
    return parsed_values, parsed_mask


def _matrix(
    row: Mapping[str, Any],
    *,
    field_name: str,
    height: int,
    width: int,
) -> tuple[list[list[float]], list[list[bool]]]:
    values = row.get("values")
    masks = row.get("mask")
    if (
        not isinstance(values, list)
        or not isinstance(masks, list)
        or len(values) != height
        or len(masks) != height
    ):
        raise ValueError(f"expanded target {field_name} matrix shape mismatch")
    parsed_values: list[list[float]] = []
    parsed_masks: list[list[bool]] = []
    for index in range(height):
        value_row, mask_row = _vector(
            {"values": values[index], "mask": masks[index]},
            field_name=f"{field_name}[{index}]",
            width=width,
        )
        parsed_values.append(value_row)
        parsed_masks.append(mask_row)
    return parsed_values, parsed_masks


def _masked_mixed_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    binary_columns: Sequence[int],
    count_event_columns: Sequence[int] = (),
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    binary = set(int(value) for value in binary_columns)
    count_events = set(int(value) for value in count_event_columns)
    if not count_events.issubset(binary):
        raise ValueError("count-event columns must also be binary columns")
    expanded_row_weights: torch.Tensor | None = None
    if row_weights is not None:
        expanded_row_weights = row_weights.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        while expanded_row_weights.dim() < prediction.dim() - 1:
            expanded_row_weights = expanded_row_weights.unsqueeze(-1)
        try:
            expanded_row_weights = expanded_row_weights.expand(
                prediction.shape[:-1]
            )
        except RuntimeError as exc:
            raise ValueError(
                "expanded strategic row weights do not broadcast to predictions"
            ) from exc
        if not bool(torch.isfinite(expanded_row_weights).all()) or bool(
            (expanded_row_weights < 0.0).any()
        ):
            raise ValueError(
                "expanded strategic row weights must be finite and nonnegative"
            )
    for column in range(int(prediction.shape[-1])):
        selected = mask[..., column]
        if not bool(selected.any()):
            continue
        pred = prediction[..., column][selected]
        truth = target[..., column][selected]
        if column in binary:
            if column in count_events:
                # Corrected early V2 shards retained exact prize/KO counts.
                # The corresponding head predicts event occurrence, so the
                # exact nonnegative integer count has a lossless binary
                # projection. Fractional or negative values remain malformed.
                if bool(
                    (
                        (truth < 0.0)
                        | (truth != torch.round(truth))
                    ).any()
                ):
                    raise ValueError(
                        "binary event-count target is not a nonnegative integer"
                    )
                truth = truth.gt(0.0).to(dtype=pred.dtype)
            elif bool(((truth < 0.0) | (truth > 1.0)).any()):
                raise ValueError("binary expanded target is outside [0, 1]")
            per_value = F.binary_cross_entropy_with_logits(
                pred, truth, reduction="none"
            )
        else:
            per_value = F.smooth_l1_loss(pred, truth, reduction="none")
        if expanded_row_weights is not None:
            per_value = per_value * expanded_row_weights[selected]
        parts.append(per_value.mean())
    if not parts:
        return prediction.sum() * 0.0
    return torch.stack(parts).mean()


def _weighted_mean(
    values: torch.Tensor,
    row_weights: torch.Tensor | None,
) -> torch.Tensor:
    if values.dim() != 1:
        raise ValueError("weighted expanded loss expects one value per row")
    if row_weights is None:
        return values.mean()
    weights = row_weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if int(weights.numel()) != int(values.numel()):
        raise ValueError("expanded loss row weights do not align")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError(
            "expanded strategic row weights must be finite and nonnegative"
        )
    return (values * weights).mean()


def _outcome_calibration(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    probs = torch.softmax(logits.detach().float(), dim=-1)
    one_hot = F.one_hot(target.detach(), num_classes=3).float()
    brier = float((probs - one_hot).square().sum(dim=-1).mean().item())
    confidence, prediction = probs.max(dim=-1)
    correct = prediction.eq(target.detach()).float()
    # Five fixed bins are stable for small validation slices and cheap enough
    # to record after every stage.
    ece = torch.zeros((), device=probs.device)
    for lower in torch.linspace(0.0, 0.8, 5, device=probs.device):
        upper = lower + 0.2
        selected = (confidence > lower) & (
            confidence <= upper if float(upper) >= 1.0 else confidence < upper
        )
        if bool(selected.any()):
            ece = ece + selected.float().mean() * (
                confidence[selected].mean() - correct[selected].mean()
            ).abs()
    entropy = float(
        (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()).item()
    )
    return brier, float(ece.item()), entropy


def _resident_tensor(
    targets: Mapping[str, torch.Tensor],
    name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    value = targets.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"resident expanded target tensor is missing: {name}")
    if value.device != device:
        raise ValueError(
            f"resident expanded target tensor is not on {device}: "
            f"{name} is on {value.device}"
        )
    return value


def _resident_mask(
    targets: Mapping[str, torch.Tensor],
    name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    value = _resident_tensor(targets, name, device=device)
    if bool(torch.any((value != 0) & (value != 1))):
        raise ValueError(f"resident expanded target mask is not binary: {name}")
    return value.to(dtype=torch.bool)


def resident_expanded_strategic_losses(
    *,
    option_outputs: Mapping[str, torch.Tensor],
    state_outputs: Mapping[str, torch.Tensor],
    target_indices: torch.Tensor,
    option_counts: torch.Tensor,
    target_tensors: Mapping[str, torch.Tensor],
    sample_ids: torch.Tensor,
    decision_ids: torch.Tensor,
    weights: Mapping[str, Any] | None,
    sample_row_weights: torch.Tensor | None = None,
    decision_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ExpandedStrategicLossMetrics]:
    """Compute exact masked V6 losses without moving targets off the device.

    ``option_outputs`` and ``target_indices`` are aligned with ``sample_ids``.
    Each sample id addresses the sample-aligned tensors in the resident corpus.
    ``state_outputs`` is aligned one-to-one with the unique ``decision_ids``;
    each decision id addresses a decision-aligned resident target tensor.  This
    explicit two-space contract prevents factorized policy stages from
    multiplying state-head loss.

    The resident pack owns the final-stage rule for action utility: only final
    canonical action stages may have a nonzero utility mask.  This function
    consumes that exact mask and never infers a target from a numerical zero.
    """

    canonical_weights = canonical_expanded_loss_weights(weights)
    enabled = {
        name for name, weight in canonical_weights.items() if float(weight) > 0.0
    }
    reference = next(iter(option_outputs.values()), None)
    if reference is None:
        reference = next(iter(state_outputs.values()), None)
    if reference is None:
        raise ValueError("resident expanded strategic outputs are empty")
    device = reference.device
    zero = reference.sum() * 0.0

    sample_ids = sample_ids.to(dtype=torch.long).reshape(-1)
    decision_ids = decision_ids.to(dtype=torch.long).reshape(-1)
    indices = target_indices.to(dtype=torch.long).reshape(-1)
    counts = option_counts.to(dtype=torch.long).reshape(-1)
    sample_rows = int(sample_ids.numel())
    decision_rows = int(decision_ids.numel())
    if sample_row_weights is not None:
        sample_row_weights = sample_row_weights.to(
            device=device, dtype=reference.dtype
        ).reshape(-1)
        if int(sample_row_weights.numel()) != sample_rows:
            raise ValueError(
                "resident expanded sample row weights do not align"
            )
    if decision_row_weights is not None:
        decision_row_weights = decision_row_weights.to(
            device=device, dtype=reference.dtype
        ).reshape(-1)
        if int(decision_row_weights.numel()) != decision_rows:
            raise ValueError(
                "resident expanded decision row weights do not align"
            )
    metrics = ExpandedStrategicLossMetrics(
        losses={name: 0.0 for name in EXPANDED_HEAD_IDS},
        labeled={name: 0 for name in EXPANDED_HEAD_IDS},
        total={
            name: (
                sample_rows
                if name
                in {
                    "action_q",
                    "action_type",
                    "action_target",
                    "action_resource",
                    "action_utility",
                }
                else decision_rows
            )
            for name in EXPANDED_HEAD_IDS
        },
    )
    metrics.masked = dict(metrics.total)

    # The all-zero schedule is the V5 compatibility path. Do not touch target
    # tensors or require a V6 resident pack when no strategic head is active.
    if not enabled:
        return zero, metrics

    for name, value in (
        ("sample_ids", sample_ids),
        ("decision_ids", decision_ids),
        ("target_indices", indices),
        ("option_counts", counts),
    ):
        if value.device != device:
            raise ValueError(
                f"resident expanded {name} is not on {device}: {value.device}"
            )
    if not (
        int(indices.numel()) == int(counts.numel()) == sample_rows
    ):
        raise ValueError("resident expanded option-row alignment mismatch")
    if sample_rows and int(torch.unique(sample_ids).numel()) != sample_rows:
        raise ValueError("resident expanded sample_ids contain duplicates")
    if decision_rows and int(torch.unique(decision_ids).numel()) != decision_rows:
        raise ValueError("resident expanded decision_ids contain duplicates")
    if bool((sample_ids < 0).any()) or bool((decision_ids < 0).any()):
        raise ValueError("resident expanded target index is negative")
    if bool((counts <= 0).any()):
        raise ValueError("resident expanded option count is not positive")
    if bool(((indices < 0) | (indices >= counts)).any()):
        raise ValueError("resident expanded selected option is outside row")

    required_option_shapes = {
        "action_q": 2,
        "action_type": 2,
        "action_target": 2,
        "action_resource": 2,
        "action_utility": 3,
    }
    max_options: int | None = None
    for name, dimensions in required_option_shapes.items():
        output = option_outputs.get(name)
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"resident expanded option output is missing: {name}")
        if output.device != device:
            raise ValueError(
                f"resident expanded option output is not on {device}: {name}"
            )
        if output.dim() != dimensions or int(output.shape[0]) != sample_rows:
            raise ValueError(
                f"resident expanded option output shape mismatch: "
                f"{name}={list(output.shape)} rows={sample_rows}"
            )
        if max_options is None:
            max_options = int(output.shape[1])
        elif int(output.shape[1]) != max_options:
            raise ValueError("resident expanded option widths do not align")
    if int(option_outputs["action_utility"].shape[-1]) != 6:
        raise ValueError("resident expanded action utility width is not 6")
    if max_options is None or bool((counts > max_options).any()):
        raise ValueError("resident expanded option count exceeds output width")

    required_state_shapes = {
        "tactical_outcome": (3, 6),
        "opponent_response": (7,),
        "resource_forecast": (6,),
        "game_phase": (5,),
        "outcome_distribution": (3,),
        "remaining_turns": (1,),
    }
    for name, suffix in required_state_shapes.items():
        output = state_outputs.get(name)
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"resident expanded state output is missing: {name}")
        if output.device != device:
            raise ValueError(
                f"resident expanded state output is not on {device}: {name}"
            )
        expected = (decision_rows, *suffix)
        if tuple(output.shape) != expected:
            raise ValueError(
                f"resident expanded state output shape mismatch: "
                f"{name}={list(output.shape)} expected={list(expected)}"
            )

    selected_rows = torch.arange(sample_rows, device=device)
    selected_option_outputs = {
        name: option_outputs[name][selected_rows, indices]
        for name in required_option_shapes
    }
    losses: dict[str, torch.Tensor] = {}

    action_q_target = _resident_tensor(
        target_tensors, "strategic_action_q_target", device=device
    )
    action_q_mask_all = _resident_mask(
        target_tensors, "strategic_action_q_mask", device=device
    )
    action_q_mask = action_q_mask_all.index_select(0, sample_ids)
    if bool(action_q_mask.any()):
        target = action_q_target.index_select(0, sample_ids)[action_q_mask]
        if not bool(torch.isfinite(target).all()):
            raise ValueError("resident expanded action-Q target is non-finite")
        pred = selected_option_outputs["action_q"][action_q_mask]
        losses["action_q"] = _weighted_mean(
            F.smooth_l1_loss(
                pred, target.to(dtype=pred.dtype), reduction="none"
            ),
            (
                None
                if sample_row_weights is None
                else sample_row_weights[action_q_mask]
            ),
        )
    else:
        losses["action_q"] = selected_option_outputs["action_q"].sum() * 0.0
    metrics.labeled["action_q"] = int(action_q_mask.sum().item())

    factor_mask_all = _resident_mask(
        target_tensors, "strategic_action_factor_mask", device=device
    )
    if factor_mask_all.dim() != 2 or int(factor_mask_all.shape[1]) != 3:
        raise ValueError("resident expanded action-factor mask shape mismatch")
    factor_mask = factor_mask_all.index_select(0, sample_ids)
    for column, name in enumerate(
        ("action_type", "action_target", "action_resource")
    ):
        usable = factor_mask[:, column]
        if bool(usable.any()):
            logits = option_outputs[name][usable]
            local_counts = counts[usable]
            padding = (
                torch.arange(logits.shape[1], device=device).unsqueeze(0)
                >= local_counts.unsqueeze(1)
            )
            logits = logits.masked_fill(padding, float("-inf"))
            losses[name] = _weighted_mean(
                F.cross_entropy(
                    logits, indices[usable], reduction="none"
                ),
                (
                    None
                    if sample_row_weights is None
                    else sample_row_weights[usable]
                ),
            )
        else:
            losses[name] = option_outputs[name].sum() * 0.0
        metrics.labeled[name] = int(usable.sum().item())

    utility_target_all = _resident_tensor(
        target_tensors, "strategic_action_utility_target", device=device
    )
    utility_mask_all = _resident_mask(
        target_tensors, "strategic_action_utility_mask", device=device
    )
    utility_target = utility_target_all.index_select(0, sample_ids)
    utility_mask = utility_mask_all.index_select(0, sample_ids)
    if (
        utility_target.dim() != 2
        or tuple(utility_target.shape) != (sample_rows, 6)
        or tuple(utility_mask.shape) != (sample_rows, 6)
    ):
        raise ValueError("resident expanded action-utility shape mismatch")
    utility_rows = utility_mask.any(dim=-1)
    if bool(utility_rows.any()):
        target = utility_target[utility_rows]
        if not bool(torch.isfinite(target[utility_mask[utility_rows]]).all()):
            raise ValueError("resident expanded action-utility target is non-finite")
        losses["action_utility"] = _masked_mixed_loss(
            selected_option_outputs["action_utility"][utility_rows],
            target.to(dtype=selected_option_outputs["action_utility"].dtype),
            utility_mask[utility_rows],
            binary_columns=(5,),
            count_event_columns=(5,),
            row_weights=(
                None
                if sample_row_weights is None
                else sample_row_weights[utility_rows]
            ),
        )
    else:
        losses["action_utility"] = (
            selected_option_outputs["action_utility"].sum() * 0.0
        )
    metrics.labeled["action_utility"] = int(utility_rows.sum().item())

    tactical_target_all = _resident_tensor(
        target_tensors, "strategic_tactical_outcome_target", device=device
    )
    tactical_mask_all = _resident_mask(
        target_tensors, "strategic_tactical_outcome_mask", device=device
    )
    tactical_target = tactical_target_all.index_select(0, decision_ids)
    tactical_mask = tactical_mask_all.index_select(0, decision_ids)
    if (
        tuple(tactical_target.shape) != (decision_rows, 3, 6)
        or tuple(tactical_mask.shape) != (decision_rows, 3, 6)
    ):
        raise ValueError("resident expanded tactical-outcome shape mismatch")
    tactical_rows = tactical_mask.reshape(decision_rows, -1).any(dim=-1)
    if bool(tactical_rows.any()):
        target = tactical_target[tactical_rows]
        if not bool(torch.isfinite(target[tactical_mask[tactical_rows]]).all()):
            raise ValueError(
                "resident expanded tactical-outcome target is non-finite"
            )
        losses["tactical_outcome"] = _masked_mixed_loss(
            state_outputs["tactical_outcome"][tactical_rows],
            target.to(dtype=state_outputs["tactical_outcome"].dtype),
            tactical_mask[tactical_rows],
            binary_columns=(2, 3),
            row_weights=(
                None
                if decision_row_weights is None
                else decision_row_weights[tactical_rows]
            ),
        )
    else:
        losses["tactical_outcome"] = (
            state_outputs["tactical_outcome"].sum() * 0.0
        )
    metrics.labeled["tactical_outcome"] = int(tactical_rows.sum().item())

    def resident_state_vector_loss(
        name: str,
        target_name: str,
        mask_name: str,
        width: int,
        *,
        binary_columns: Sequence[int],
        count_event_columns: Sequence[int] = (),
    ) -> tuple[torch.Tensor, int]:
        target_all = _resident_tensor(target_tensors, target_name, device=device)
        mask_all = _resident_mask(target_tensors, mask_name, device=device)
        target = target_all.index_select(0, decision_ids)
        mask = mask_all.index_select(0, decision_ids)
        expected = (decision_rows, width)
        if tuple(target.shape) != expected or tuple(mask.shape) != expected:
            raise ValueError(
                f"resident expanded {name} target shape mismatch"
            )
        usable = mask.any(dim=-1)
        if not bool(usable.any()):
            return state_outputs[name].sum() * 0.0, 0
        selected_target = target[usable]
        selected_mask = mask[usable]
        if not bool(torch.isfinite(selected_target[selected_mask]).all()):
            raise ValueError(f"resident expanded {name} target is non-finite")
        return (
            _masked_mixed_loss(
                state_outputs[name][usable],
                selected_target.to(dtype=state_outputs[name].dtype),
                selected_mask,
                binary_columns=binary_columns,
                count_event_columns=count_event_columns,
                row_weights=(
                    None
                    if decision_row_weights is None
                    else decision_row_weights[usable]
                ),
            ),
            int(usable.sum().item()),
        )

    losses["opponent_response"], metrics.labeled["opponent_response"] = (
        resident_state_vector_loss(
            "opponent_response",
            "strategic_opponent_response_target",
            "strategic_opponent_response_mask",
            7,
            binary_columns=tuple(range(7)),
            count_event_columns=(1, 2),
        )
    )
    losses["resource_forecast"], metrics.labeled["resource_forecast"] = (
        resident_state_vector_loss(
            "resource_forecast",
            "strategic_resource_forecast_target",
            "strategic_resource_forecast_mask",
            6,
            binary_columns=(4, 5),
        )
    )

    phase_target_all = _resident_tensor(
        target_tensors, "strategic_game_phase_target", device=device
    )
    phase_mask_all = _resident_mask(
        target_tensors, "strategic_game_phase_mask", device=device
    )
    phase_target = phase_target_all.index_select(0, decision_ids)
    phase_mask = phase_mask_all.index_select(0, decision_ids)
    if tuple(phase_target.shape) != (decision_rows,):
        raise ValueError("resident expanded game-phase target shape mismatch")
    if bool(phase_mask.any()):
        target = phase_target[phase_mask].to(dtype=torch.long)
        if bool(((target < 0) | (target >= 5)).any()):
            raise ValueError("resident expanded game-phase target is outside [0, 5)")
        losses["game_phase"] = _weighted_mean(
            F.cross_entropy(
                state_outputs["game_phase"][phase_mask],
                target,
                reduction="none",
            ),
            (
                None
                if decision_row_weights is None
                else decision_row_weights[phase_mask]
            ),
        )
    else:
        losses["game_phase"] = state_outputs["game_phase"].sum() * 0.0
    metrics.labeled["game_phase"] = int(phase_mask.sum().item())

    outcome_target_all = _resident_tensor(
        target_tensors, "strategic_outcome_class_target", device=device
    )
    outcome_mask_all = _resident_mask(
        target_tensors, "strategic_outcome_class_mask", device=device
    )
    outcome_target = outcome_target_all.index_select(0, decision_ids)
    outcome_mask = outcome_mask_all.index_select(0, decision_ids)
    if tuple(outcome_target.shape) != (decision_rows,):
        raise ValueError("resident expanded outcome target shape mismatch")
    if bool(outcome_mask.any()):
        target = outcome_target[outcome_mask].to(dtype=torch.long)
        if bool(((target < 0) | (target >= 3)).any()):
            raise ValueError("resident expanded outcome target is outside [0, 3)")
        logits = state_outputs["outcome_distribution"][outcome_mask]
        losses["outcome_distribution"] = _weighted_mean(
            F.cross_entropy(logits, target, reduction="none"),
            (
                None
                if decision_row_weights is None
                else decision_row_weights[outcome_mask]
            ),
        )
        (
            metrics.outcome_brier,
            metrics.outcome_ece,
            metrics.outcome_entropy,
        ) = _outcome_calibration(logits, target)
    else:
        losses["outcome_distribution"] = (
            state_outputs["outcome_distribution"].sum() * 0.0
        )
    metrics.labeled["outcome_distribution"] = int(outcome_mask.sum().item())

    remaining_target_all = _resident_tensor(
        target_tensors, "strategic_remaining_turns_target", device=device
    )
    remaining_mask_all = _resident_mask(
        target_tensors, "strategic_remaining_turns_mask", device=device
    )
    remaining_target = remaining_target_all.index_select(0, decision_ids)
    remaining_mask = remaining_mask_all.index_select(0, decision_ids)
    if tuple(remaining_target.shape) != (decision_rows,):
        raise ValueError("resident expanded remaining-turns target shape mismatch")
    if bool(remaining_mask.any()):
        target = remaining_target[remaining_mask]
        if (
            not bool(torch.isfinite(target).all())
            or bool((target < 0.0).any())
        ):
            raise ValueError("resident expanded remaining-turns target is invalid")
        pred = state_outputs["remaining_turns"][remaining_mask].squeeze(-1)
        losses["remaining_turns"] = _weighted_mean(
            F.smooth_l1_loss(
                pred, target.to(dtype=pred.dtype), reduction="none"
            ),
            (
                None
                if decision_row_weights is None
                else decision_row_weights[remaining_mask]
            ),
        )
    else:
        losses["remaining_turns"] = (
            state_outputs["remaining_turns"].sum() * 0.0
        )
    metrics.labeled["remaining_turns"] = int(remaining_mask.sum().item())

    for name in EXPANDED_HEAD_IDS:
        metrics.masked[name] = max(
            0, int(metrics.total[name]) - int(metrics.labeled[name])
        )
        metrics.losses[name] = float(losses[name].detach().item())
    weighted = torch.stack(
        [
            losses[name] * float(canonical_weights[name])
            for name in EXPANDED_HEAD_IDS
            if float(canonical_weights[name]) > 0.0
        ]
    ).sum()
    return weighted, metrics


def expanded_strategic_losses(
    *,
    option_outputs: Mapping[str, torch.Tensor],
    state_outputs: Mapping[str, torch.Tensor],
    target_indices: torch.Tensor,
    value_targets: torch.Tensor,
    option_counts: Sequence[int] | torch.Tensor,
    stage_indices: Sequence[int],
    decision_aux: Sequence[Mapping[str, Any]],
    weights: Mapping[str, Any] | None,
    row_weights: torch.Tensor | None = None,
    state_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ExpandedStrategicLossMetrics]:
    """Compute exact masked V6 head losses for a flattened policy-stage batch."""

    canonical_weights = canonical_expanded_loss_weights(weights)
    enabled = {
        name for name, weight in canonical_weights.items() if float(weight) > 0.0
    }
    reference = next(iter(option_outputs.values()), None)
    if reference is None:
        reference = next(iter(state_outputs.values()), None)
    if reference is None:
        raise ValueError("expanded strategic outputs are empty")
    device = reference.device
    rows = int(target_indices.numel())
    counts = torch.as_tensor(
        option_counts, device=device, dtype=torch.long
    ).reshape(-1)
    indices = target_indices.to(device=device, dtype=torch.long).reshape(-1)
    values = value_targets.to(device=device).reshape(-1)
    if not (
        rows
        == int(counts.numel())
        == len(stage_indices)
        == len(decision_aux)
        == int(values.numel())
    ):
        raise ValueError("expanded strategic flattened row alignment mismatch")
    if bool(((indices < 0) | (indices >= counts)).any()):
        raise ValueError("expanded strategic selected option is outside row")
    if row_weights is not None:
        row_weights = row_weights.to(
            device=device, dtype=reference.dtype
        ).reshape(-1)
        if int(row_weights.numel()) != rows:
            raise ValueError("expanded strategic row weights do not align")
        if not bool(torch.isfinite(row_weights).all()) or bool(
            (row_weights < 0.0).any()
        ):
            raise ValueError(
                "expanded strategic row weights must be finite and nonnegative"
            )
    if state_row_weights is not None:
        state_row_weights = state_row_weights.to(
            device=device, dtype=reference.dtype
        ).reshape(-1)
        if int(state_row_weights.numel()) != rows:
            raise ValueError(
                "expanded strategic state row weights do not align"
            )
        if not bool(torch.isfinite(state_row_weights).all()) or bool(
            (state_row_weights < 0.0).any()
        ):
            raise ValueError(
                "expanded strategic state row weights must be finite and "
                "nonnegative"
            )
    effective_state_row_weights = (
        row_weights if state_row_weights is None else state_row_weights
    )

    metrics = ExpandedStrategicLossMetrics(
        losses={name: 0.0 for name in EXPANDED_HEAD_IDS},
        labeled={name: 0 for name in EXPANDED_HEAD_IDS},
        total={name: rows for name in EXPANDED_HEAD_IDS},
        masked={name: rows for name in EXPANDED_HEAD_IDS},
    )
    zero = reference.sum() * 0.0
    if not enabled:
        return zero, metrics

    contracts: list[dict[str, Any] | None] = []
    for aux in decision_aux:
        raw = dict(aux or {}).get("expanded_strategic")
        contracts.append(
            None if raw is None else _mapping(raw, field_name="expanded_strategic")
        )

    row_ids = torch.arange(rows, device=device)
    selected_option_outputs = {
        name: tensor[row_ids, indices]
        for name, tensor in option_outputs.items()
    }
    losses: dict[str, torch.Tensor] = {}

    # Action-Q is a selected-action Monte-Carlo return. Incomplete games have
    # outcome_class=None and are masked instead of treated as draws.
    action_q_rows = [
        row
        for row, contract in enumerate(contracts)
        if contract is not None and contract.get("outcome_class") is not None
    ]
    if action_q_rows:
        selected = torch.tensor(action_q_rows, device=device, dtype=torch.long)
        prediction = selected_option_outputs["action_q"].index_select(
            0, selected
        )
        losses["action_q"] = _weighted_mean(
            F.smooth_l1_loss(
                prediction,
                values.index_select(0, selected).to(
                    dtype=prediction.dtype
                ),
                reduction="none",
            ),
            (
                None
                if row_weights is None
                else row_weights.index_select(0, selected)
            ),
        )
    else:
        losses["action_q"] = selected_option_outputs["action_q"].sum() * 0.0
    metrics.labeled["action_q"] = len(action_q_rows)

    for name in ("action_type", "action_target", "action_resource"):
        factor_key = name.removeprefix("action_")
        usable: list[int] = []
        for row, (contract, stage_index) in enumerate(
            zip(contracts, stage_indices)
        ):
            if contract is None:
                continue
            factors = contract.get("action_factors")
            if not isinstance(factors, list) or not 0 <= int(stage_index) < len(
                factors
            ):
                continue
            factor_row = _mapping(
                factors[int(stage_index)],
                field_name=f"action_factors[{int(stage_index)}]",
            )
            if bool(factor_row.get(factor_key)):
                usable.append(row)
        if usable:
            selected = torch.tensor(usable, device=device, dtype=torch.long)
            logits = option_outputs[name].index_select(0, selected)
            local_counts = counts.index_select(0, selected)
            padding = (
                torch.arange(logits.shape[1], device=device).unsqueeze(0)
                >= local_counts.unsqueeze(1)
            )
            logits = logits.masked_fill(padding, float("-inf"))
            losses[name] = _weighted_mean(
                F.cross_entropy(
                    logits,
                    indices.index_select(0, selected),
                    reduction="none",
                ),
                (
                    None
                    if row_weights is None
                    else row_weights.index_select(0, selected)
                ),
            )
        else:
            losses[name] = option_outputs[name].sum() * 0.0
        metrics.labeled[name] = len(usable)

    utility_rows: list[int] = []
    utility_values: list[list[float]] = []
    utility_masks: list[list[bool]] = []
    for row, (contract, stage_index) in enumerate(
        zip(contracts, stage_indices)
    ):
        if contract is None or contract.get("action_utility") is None:
            continue
        factors = contract.get("action_factors")
        if (
            not isinstance(factors, list)
            or not factors
            or int(stage_index) != len(factors) - 1
        ):
            continue
        target, mask = _vector(
            _mapping(
                contract["action_utility"], field_name="action_utility"
            ),
            field_name="action_utility",
            width=6,
        )
        if any(mask):
            utility_rows.append(row)
            utility_values.append(target)
            utility_masks.append(mask)
    if utility_rows:
        selected = torch.tensor(utility_rows, device=device, dtype=torch.long)
        pred = selected_option_outputs["action_utility"].index_select(0, selected)
        target = torch.tensor(
            utility_values, device=device, dtype=pred.dtype
        )
        mask = torch.tensor(utility_masks, device=device, dtype=torch.bool)
        losses["action_utility"] = _masked_mixed_loss(
            pred,
            target,
            mask,
            binary_columns=(5,),
            count_event_columns=(5,),
            row_weights=(
                None
                if row_weights is None
                else row_weights.index_select(0, selected)
            ),
        )
    else:
        losses["action_utility"] = (
            selected_option_outputs["action_utility"].sum() * 0.0
        )
    metrics.labeled["action_utility"] = len(utility_rows)

    # State heads train only once per real decision. Later autoregressive
    # policy stages share the same state and would otherwise overweight it.
    decision_rows = [
        row
        for row, stage_index in enumerate(stage_indices)
        if int(stage_index) == 0
    ]
    for name in (
        "tactical_outcome",
        "opponent_response",
        "resource_forecast",
        "game_phase",
        "outcome_distribution",
        "remaining_turns",
    ):
        metrics.total[name] = len(decision_rows)

    tactical_rows: list[int] = []
    tactical_values: list[list[list[float]]] = []
    tactical_masks: list[list[list[bool]]] = []
    for row in decision_rows:
        contract = contracts[row]
        if contract is None:
            continue
        raw = contract.get("tactical_outcomes")
        if raw is None:
            continue
        target, mask = _matrix(
            _mapping(raw, field_name="tactical_outcomes"),
            field_name="tactical_outcomes",
            height=3,
            width=6,
        )
        if any(any(item) for item in mask):
            tactical_rows.append(row)
            tactical_values.append(target)
            tactical_masks.append(mask)
    if tactical_rows:
        selected = torch.tensor(tactical_rows, device=device, dtype=torch.long)
        pred = state_outputs["tactical_outcome"].index_select(0, selected)
        target = torch.tensor(
            tactical_values, device=device, dtype=pred.dtype
        )
        mask = torch.tensor(tactical_masks, device=device, dtype=torch.bool)
        losses["tactical_outcome"] = _masked_mixed_loss(
            pred,
            target,
            mask,
            binary_columns=(2, 3),
            row_weights=(
                None
                if effective_state_row_weights is None
                else effective_state_row_weights.index_select(0, selected)
            ),
        )
    else:
        losses["tactical_outcome"] = (
            state_outputs["tactical_outcome"].sum() * 0.0
        )
    metrics.labeled["tactical_outcome"] = len(tactical_rows)

    def state_vector_loss(
        name: str,
        target_name: str,
        width: int,
        *,
        binary_columns: Sequence[int],
        count_event_columns: Sequence[int] = (),
    ) -> tuple[torch.Tensor, int]:
        usable: list[int] = []
        targets: list[list[float]] = []
        masks: list[list[bool]] = []
        for row in decision_rows:
            contract = contracts[row]
            if contract is None:
                continue
            raw = contract.get(target_name)
            if raw is None:
                continue
            target, mask = _vector(
                _mapping(raw, field_name=target_name),
                field_name=target_name,
                width=width,
            )
            if any(mask):
                usable.append(row)
                targets.append(target)
                masks.append(mask)
        pred_all = state_outputs[name]
        if not usable:
            return pred_all.sum() * 0.0, 0
        selected = torch.tensor(usable, device=device, dtype=torch.long)
        pred = pred_all.index_select(0, selected)
        target = torch.tensor(targets, device=device, dtype=pred.dtype)
        mask = torch.tensor(masks, device=device, dtype=torch.bool)
        return (
            _masked_mixed_loss(
                pred,
                target,
                mask,
                binary_columns=binary_columns,
                count_event_columns=count_event_columns,
                row_weights=(
                    None
                    if effective_state_row_weights is None
                    else effective_state_row_weights.index_select(0, selected)
                ),
            ),
            len(usable),
        )

    losses["opponent_response"], metrics.labeled["opponent_response"] = (
        state_vector_loss(
            "opponent_response",
            "opponent_response",
            7,
            binary_columns=tuple(range(7)),
            count_event_columns=(1, 2),
        )
    )
    losses["resource_forecast"], metrics.labeled["resource_forecast"] = (
        state_vector_loss(
            "resource_forecast",
            "resource_forecast",
            6,
            binary_columns=(4, 5),
        )
    )

    phase_rows: list[int] = []
    phase_targets: list[int] = []
    outcome_rows: list[int] = []
    outcome_targets: list[int] = []
    remaining_rows: list[int] = []
    remaining_targets: list[float] = []
    for row in decision_rows:
        contract = contracts[row]
        if contract is None:
            continue
        phase = contract.get("game_phase")
        if phase is not None:
            phase = int(phase)
            if not 0 <= phase < 5:
                raise ValueError("expanded game_phase target is outside [0, 5)")
            phase_rows.append(row)
            phase_targets.append(phase)
        outcome = contract.get("outcome_class")
        if outcome is not None:
            outcome = int(outcome)
            if not 0 <= outcome < 3:
                raise ValueError("expanded outcome_class target is outside [0, 3)")
            outcome_rows.append(row)
            outcome_targets.append(outcome)
        remaining = contract.get("remaining_turns_log1p")
        if remaining is not None:
            remaining = float(remaining)
            if not math.isfinite(remaining) or remaining < 0.0:
                raise ValueError("expanded remaining-turns target is invalid")
            remaining_rows.append(row)
            remaining_targets.append(remaining)

    if phase_rows:
        selected = torch.tensor(phase_rows, device=device, dtype=torch.long)
        losses["game_phase"] = _weighted_mean(
            F.cross_entropy(
                state_outputs["game_phase"].index_select(0, selected),
                torch.tensor(phase_targets, device=device, dtype=torch.long),
                reduction="none",
            ),
            (
                None
                if effective_state_row_weights is None
                else effective_state_row_weights.index_select(0, selected)
            ),
        )
    else:
        losses["game_phase"] = state_outputs["game_phase"].sum() * 0.0
    metrics.labeled["game_phase"] = len(phase_rows)

    if outcome_rows:
        selected = torch.tensor(outcome_rows, device=device, dtype=torch.long)
        outcome_logits = state_outputs["outcome_distribution"].index_select(
            0, selected
        )
        outcome_target = torch.tensor(
            outcome_targets, device=device, dtype=torch.long
        )
        losses["outcome_distribution"] = _weighted_mean(
            F.cross_entropy(
                outcome_logits, outcome_target, reduction="none"
            ),
            (
                None
                if effective_state_row_weights is None
                else effective_state_row_weights.index_select(0, selected)
            ),
        )
        (
            metrics.outcome_brier,
            metrics.outcome_ece,
            metrics.outcome_entropy,
        ) = _outcome_calibration(outcome_logits, outcome_target)
    else:
        losses["outcome_distribution"] = (
            state_outputs["outcome_distribution"].sum() * 0.0
        )
    metrics.labeled["outcome_distribution"] = len(outcome_rows)

    if remaining_rows:
        selected = torch.tensor(remaining_rows, device=device, dtype=torch.long)
        pred = state_outputs["remaining_turns"].index_select(0, selected).squeeze(
            -1
        )
        losses["remaining_turns"] = _weighted_mean(
            F.smooth_l1_loss(
                pred,
                torch.tensor(
                    remaining_targets, device=device, dtype=pred.dtype
                ),
                reduction="none",
            ),
            (
                None
                if effective_state_row_weights is None
                else effective_state_row_weights.index_select(0, selected)
            ),
        )
    else:
        losses["remaining_turns"] = (
            state_outputs["remaining_turns"].sum() * 0.0
        )
    metrics.labeled["remaining_turns"] = len(remaining_rows)

    for name in EXPANDED_HEAD_IDS:
        metrics.masked[name] = max(
            0, int(metrics.total[name]) - int(metrics.labeled[name])
        )
        metrics.losses[name] = float(losses[name].detach().item())
    weighted = torch.stack(
        [
            losses[name] * float(canonical_weights[name])
            for name in EXPANDED_HEAD_IDS
            if float(canonical_weights[name]) > 0.0
        ]
    ).sum()
    return weighted, metrics
