"""Observed-target curriculum for setup-active and setup-bench choices.

The current-deck guide may focus learning pressure on these rows, but it never
supplies a prediction target.  Direction comes only from the already-causal
next-own-decision resource labels and the complete-game outcome.  Deliberately,
this module has no guide-option-index argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


SETUP_ACTIVE_CONTEXT = 1
SETUP_BENCH_CONTEXT = 2
SETUP_CONTEXT_NAMES = {
    SETUP_ACTIVE_CONTEXT: "setup_active",
    SETUP_BENCH_CONTEXT: "setup_bench",
}
SETUP_BOARD_RESOURCE_WIDTH = 6
SETUP_BOARD_OUTCOME_CLASSES = 3
SETUP_BOARD_OUTPUT_WIDTH = (
    SETUP_BOARD_RESOURCE_WIDTH + SETUP_BOARD_OUTCOME_CLASSES
)
RESOURCE_BINARY_COLUMNS: tuple[int, ...] = (4, 5)


@dataclass(frozen=True)
class SetupBoardOutcomeMetrics:
    """Coverage and calibration evidence for one loss computation."""

    total_rows: int
    eligible_rows: int
    context_rows: dict[str, int]
    stop_rows: int
    non_stop_rows: int
    resource_labels: int
    outcome_labels: int
    guide_rows: int
    base_observed_loss: float
    guide_observed_loss: float
    weighted_loss: float
    outcome_brier: float | None
    resource_mae: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rows": int(self.total_rows),
            "eligible_rows": int(self.eligible_rows),
            "context_rows": dict(self.context_rows),
            "stop_rows": int(self.stop_rows),
            "non_stop_rows": int(self.non_stop_rows),
            "resource_labels": int(self.resource_labels),
            "outcome_labels": int(self.outcome_labels),
            "guide_rows": int(self.guide_rows),
            "base_observed_loss": float(self.base_observed_loss),
            "guide_observed_loss": float(self.guide_observed_loss),
            "weighted_loss": float(self.weighted_loss),
            "outcome_brier": self.outcome_brier,
            "resource_mae": self.resource_mae,
        }


def _as_bool_mask(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dtype == torch.bool:
        return value
    if bool(torch.any((value != 0) & (value != 1))):
        raise ValueError(f"{name} must be binary")
    return value.to(dtype=torch.bool)


def _mean_or_zero(parts: Sequence[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    if not parts:
        return zero
    return torch.stack(list(parts)).mean()


def _balanced_observed_loss(
    *,
    selected: torch.Tensor,
    eligible: torch.Tensor,
    contexts: torch.Tensor,
    resource_target: torch.Tensor,
    resource_mask: torch.Tensor,
    outcome_target: torch.Tensor,
    outcome_mask: torch.Tensor,
    row_scale: torch.Tensor,
) -> torch.Tensor:
    """Average targets within context, then average setup contexts.

    ``row_scale`` is all ones for the ordinary loss.  For the guide curriculum
    it is guide confidence, so confidence genuinely changes gradient magnitude
    instead of being normalized away.
    """

    zero = selected.sum() * 0.0
    context_losses: list[torch.Tensor] = []
    binary_columns = set(RESOURCE_BINARY_COLUMNS)
    for context in (SETUP_ACTIVE_CONTEXT, SETUP_BENCH_CONTEXT):
        in_context = eligible & contexts.eq(context)
        if not bool(in_context.any()):
            continue
        target_branches: list[torch.Tensor] = []
        resource_columns: list[torch.Tensor] = []
        for column in range(SETUP_BOARD_RESOURCE_WIDTH):
            usable = in_context & resource_mask[:, column]
            if not bool(usable.any()):
                continue
            prediction = selected[:, column][usable]
            truth = resource_target[:, column][usable].to(
                dtype=prediction.dtype
            )
            if column in binary_columns:
                per_row = F.binary_cross_entropy_with_logits(
                    prediction,
                    truth,
                    reduction="none",
                )
            else:
                per_row = F.smooth_l1_loss(
                    prediction,
                    truth,
                    reduction="none",
                )
            resource_columns.append(
                (per_row * row_scale[usable].to(dtype=per_row.dtype)).mean()
            )
        if resource_columns:
            target_branches.append(torch.stack(resource_columns).mean())

        usable_outcome = in_context & outcome_mask
        if bool(usable_outcome.any()):
            per_row = F.cross_entropy(
                selected[:, SETUP_BOARD_RESOURCE_WIDTH :][usable_outcome],
                outcome_target[usable_outcome].to(dtype=torch.long),
                reduction="none",
            )
            target_branches.append(
                (
                    per_row
                    * row_scale[usable_outcome].to(dtype=per_row.dtype)
                ).mean()
            )
        if target_branches:
            context_losses.append(torch.stack(target_branches).mean())
    return _mean_or_zero(context_losses, zero)


def setup_board_outcome_loss(
    *,
    predictions: torch.Tensor,
    selected_indices: torch.Tensor,
    option_counts: torch.Tensor,
    select_contexts: torch.Tensor,
    resource_targets: torch.Tensor,
    resource_masks: torch.Tensor,
    outcome_targets: torch.Tensor,
    outcome_masks: torch.Tensor,
    guide_confidences: torch.Tensor,
    selected_is_stop: torch.Tensor | None = None,
    base_loss_weight: float = 0.025,
    guide_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, SetupBoardOutcomeMetrics]:
    """Train selected setup choices from observed targets, never imitation.

    Predictions are ``[rows, max_options, 9]``.  Only the demonstrated option
    is gathered, which keeps every unchosen option exactly masked.  The guide
    can only scale the same observed-target loss through ``guide_confidences``;
    no preferred guide action is accepted by this API.
    """

    if predictions.dim() != 3 or int(predictions.shape[-1]) != SETUP_BOARD_OUTPUT_WIDTH:
        raise ValueError(
            "setup-board prediction shape must be [rows, max_options, 9]"
        )
    rows, max_options, _ = predictions.shape
    device = predictions.device
    if rows <= 0 or max_options <= 0:
        raise ValueError("setup-board predictions must contain rows and options")

    vectors = {
        "selected_indices": selected_indices,
        "option_counts": option_counts,
        "select_contexts": select_contexts,
        "outcome_targets": outcome_targets,
        "outcome_masks": outcome_masks,
        "guide_confidences": guide_confidences,
    }
    for name, value in vectors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.device != device:
            raise ValueError(f"{name} is on {value.device}, expected {device}")
        if value.reshape(-1).numel() != rows:
            raise ValueError(f"{name} row count does not match predictions")
    if resource_targets.device != device or resource_masks.device != device:
        raise ValueError("setup-board resource targets must share prediction device")
    if tuple(resource_targets.shape) != (rows, SETUP_BOARD_RESOURCE_WIDTH):
        raise ValueError("setup-board resource target shape mismatch")
    if tuple(resource_masks.shape) != (rows, SETUP_BOARD_RESOURCE_WIDTH):
        raise ValueError("setup-board resource mask shape mismatch")

    indices = selected_indices.reshape(-1).to(dtype=torch.long)
    counts = option_counts.reshape(-1).to(dtype=torch.long)
    contexts = select_contexts.reshape(-1).to(dtype=torch.long)
    outcome = outcome_targets.reshape(-1).to(dtype=torch.long)
    resource_mask = _as_bool_mask(resource_masks, name="resource_masks")
    outcome_mask = _as_bool_mask(
        outcome_masks.reshape(-1), name="outcome_masks"
    )
    confidence = guide_confidences.reshape(-1).to(dtype=torch.float32)

    if bool((counts <= 0).any()) or bool((counts > max_options).any()):
        raise ValueError("setup-board option count is outside prediction width")
    if bool(((indices < 0) | (indices >= counts)).any()):
        raise ValueError("setup-board selected option is outside legal row")
    if not bool(torch.isfinite(confidence).all()) or bool(
        ((confidence < 0.0) | (confidence > 1.0)).any()
    ):
        raise ValueError("setup-board guide confidence is outside [0, 1]")
    if not bool(
        torch.isfinite(resource_targets[resource_mask]).all()
    ):
        raise ValueError("setup-board resource target is non-finite")
    for column in RESOURCE_BINARY_COLUMNS:
        values = resource_targets[:, column][resource_mask[:, column]]
        if bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(
                f"setup-board binary resource target {column} is outside [0, 1]"
            )
    if bool(
        (
            outcome_mask
            & ((outcome < 0) | (outcome >= SETUP_BOARD_OUTCOME_CLASSES))
        ).any()
    ):
        raise ValueError("setup-board outcome target is outside [0, 3)")

    for name, weight in (
        ("base_loss_weight", base_loss_weight),
        ("guide_loss_weight", guide_loss_weight),
    ):
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    row_ids = torch.arange(rows, device=device)
    selected = predictions[row_ids, indices]
    # Terminal outcome presence is the exact existing proof that the source
    # trajectory completed. Resource labels from a censored/truncated game
    # must not make a setup row eligible by themselves.
    eligible = (
        contexts.eq(SETUP_ACTIVE_CONTEXT)
        | contexts.eq(SETUP_BENCH_CONTEXT)
    ) & outcome_mask
    observed = resource_mask.any(dim=-1) | outcome_mask
    eligible_observed = eligible & observed

    base = _balanced_observed_loss(
        selected=selected,
        eligible=eligible,
        contexts=contexts,
        resource_target=resource_targets,
        resource_mask=resource_mask,
        outcome_target=outcome,
        outcome_mask=outcome_mask,
        row_scale=torch.ones_like(confidence),
    )
    guide = _balanced_observed_loss(
        selected=selected,
        eligible=eligible,
        contexts=contexts,
        resource_target=resource_targets,
        resource_mask=resource_mask,
        outcome_target=outcome,
        outcome_mask=outcome_mask,
        row_scale=confidence,
    )
    total = float(base_loss_weight) * base + float(guide_loss_weight) * guide

    if selected_is_stop is not None:
        if selected_is_stop.device != device or selected_is_stop.numel() != rows:
            raise ValueError("selected_is_stop row/device mismatch")
    stop = (
        torch.zeros(rows, dtype=torch.bool, device=device)
        if selected_is_stop is None
        else _as_bool_mask(
            selected_is_stop.reshape(-1), name="selected_is_stop"
        )
    )

    usable_resource = eligible.unsqueeze(-1) & resource_mask
    resource_mae: float | None = None
    if bool(usable_resource.any()):
        resource_prediction = selected[:, :SETUP_BOARD_RESOURCE_WIDTH].detach()
        resource_prediction = resource_prediction.clone()
        resource_prediction[:, list(RESOURCE_BINARY_COLUMNS)] = torch.sigmoid(
            resource_prediction[:, list(RESOURCE_BINARY_COLUMNS)]
        )
        resource_mae = float(
            (
                resource_prediction[usable_resource].float()
                - resource_targets[usable_resource].detach().float()
            )
            .abs()
            .mean()
            .item()
        )
    usable_outcome = eligible & outcome_mask
    outcome_brier: float | None = None
    if bool(usable_outcome.any()):
        probabilities = torch.softmax(
            selected[
                usable_outcome,
                SETUP_BOARD_RESOURCE_WIDTH :,
            ]
            .detach()
            .float(),
            dim=-1,
        )
        truth = F.one_hot(
            outcome[usable_outcome], num_classes=SETUP_BOARD_OUTCOME_CLASSES
        ).float()
        outcome_brier = float(
            (probabilities - truth).square().sum(dim=-1).mean().item()
        )

    metrics = SetupBoardOutcomeMetrics(
        total_rows=int(rows),
        eligible_rows=int(eligible_observed.sum().item()),
        context_rows={
            name: int((eligible_observed & contexts.eq(context)).sum().item())
            for context, name in SETUP_CONTEXT_NAMES.items()
        },
        stop_rows=int((eligible_observed & stop).sum().item()),
        non_stop_rows=int((eligible_observed & ~stop).sum().item()),
        resource_labels=int(usable_resource.sum().item()),
        outcome_labels=int(usable_outcome.sum().item()),
        guide_rows=int(
            (eligible_observed & confidence.gt(0.0)).sum().item()
        ),
        base_observed_loss=float(base.detach().item()),
        guide_observed_loss=float(guide.detach().item()),
        weighted_loss=float(total.detach().item()),
        outcome_brier=outcome_brier,
        resource_mae=resource_mae,
    )
    return total, metrics
