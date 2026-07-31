"""Typed observed-target loss for the Slowking combo-state head.

The head is option-conditioned because each legal action can preserve or
consume a different combo line. Targets come only from exact causal state or an
observed transition. The current-deck guide may scale an already-observed row,
but this API deliberately has no guide-preferred action-index input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .combo_state_contract import (
    BENCH_CONTINUITY_WIDTH,
    COMBO_STATE_KEY,
    COMBO_STATE_OUTPUT_WIDTH,
    COMBO_STATE_TARGET_SCHEMA,
    COPIED_ATTACK_WIDTH,
    ENERGY_ROUTE_WIDTH,
    SEEK_SOURCE_CLASSES,
    TOP_DECK_CLASSES,
    VECTOR_WIDTH,
    VISIBLE_PIECE_WIDTH,
    validate_combo_state_labels,
)

TOP_DECK_SLICE = slice(0, TOP_DECK_CLASSES)
SEEK_SOURCE_SLICE = slice(
    TOP_DECK_CLASSES,
    TOP_DECK_CLASSES + SEEK_SOURCE_CLASSES,
)
VECTOR_SLICE = slice(TOP_DECK_CLASSES + SEEK_SOURCE_CLASSES, COMBO_STATE_OUTPUT_WIDTH)

VECTOR_GROUPS: tuple[tuple[str, slice], ...] = (
    ("copied_attack_legality", slice(0, COPIED_ATTACK_WIDTH)),
    (
        "visible_combo_piece_availability",
        slice(COPIED_ATTACK_WIDTH, COPIED_ATTACK_WIDTH + VISIBLE_PIECE_WIDTH),
    ),
    (
        "energy_route_readiness",
        slice(
            COPIED_ATTACK_WIDTH + VISIBLE_PIECE_WIDTH,
            COPIED_ATTACK_WIDTH + VISIBLE_PIECE_WIDTH + ENERGY_ROUTE_WIDTH,
        ),
    ),
    (
        "bench_continuity",
        slice(
            COPIED_ATTACK_WIDTH + VISIBLE_PIECE_WIDTH + ENERGY_ROUTE_WIDTH,
            VECTOR_WIDTH,
        ),
    ),
)

@dataclass(frozen=True)
class ComboStateMetrics:
    total_rows: int
    eligible_rows: int
    top_deck_labels: int
    seek_source_labels: int
    vector_labels: dict[str, int]
    guide_rows: int
    base_observed_loss: float
    guide_observed_loss: float
    weighted_loss: float
    top_deck_accuracy: float | None
    seek_source_accuracy: float | None
    vector_brier: dict[str, float | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "eligible_rows": self.eligible_rows,
            "top_deck_labels": self.top_deck_labels,
            "seek_source_labels": self.seek_source_labels,
            "vector_labels": dict(self.vector_labels),
            "guide_rows": self.guide_rows,
            "base_observed_loss": self.base_observed_loss,
            "guide_observed_loss": self.guide_observed_loss,
            "weighted_loss": self.weighted_loss,
            "top_deck_accuracy": self.top_deck_accuracy,
            "seek_source_accuracy": self.seek_source_accuracy,
            "vector_brier": dict(self.vector_brier),
        }


def _mask(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dtype == torch.bool:
        return value
    if bool(torch.any((value != 0) & (value != 1))):
        raise ValueError(f"{name} must be binary")
    return value.to(dtype=torch.bool)


def combo_state_loss(
    *,
    predictions: torch.Tensor,
    selected_indices: torch.Tensor,
    option_counts: torch.Tensor,
    top_deck_targets: torch.Tensor,
    top_deck_masks: torch.Tensor,
    seek_source_targets: torch.Tensor,
    seek_source_masks: torch.Tensor,
    vector_targets: torch.Tensor,
    vector_masks: torch.Tensor,
    guide_confidences: torch.Tensor,
    base_loss_weight: float = 0.025,
    guide_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, ComboStateMetrics]:
    """Train the selected legal option from masked causal combo-state labels."""

    if predictions.dim() != 3 or predictions.size(-1) != COMBO_STATE_OUTPUT_WIDTH:
        raise ValueError("combo-state predictions must be [rows, options, 32]")
    rows, max_options, _ = predictions.shape
    if rows <= 0 or max_options <= 0:
        raise ValueError("combo-state predictions must contain rows and options")
    device = predictions.device
    vectors = {
        "selected_indices": selected_indices,
        "option_counts": option_counts,
        "top_deck_targets": top_deck_targets,
        "top_deck_masks": top_deck_masks,
        "seek_source_targets": seek_source_targets,
        "seek_source_masks": seek_source_masks,
        "guide_confidences": guide_confidences,
    }
    for name, value in vectors.items():
        if value.device != device or value.reshape(-1).numel() != rows:
            raise ValueError(f"{name} must contain one row on {device}")
    if vector_targets.device != device or vector_masks.device != device:
        raise ValueError("combo-state vector targets must share prediction device")
    if vector_targets.shape != (rows, VECTOR_WIDTH):
        raise ValueError("combo-state vector target shape mismatch")
    if vector_masks.shape != (rows, VECTOR_WIDTH):
        raise ValueError("combo-state vector mask shape mismatch")

    selected_index = selected_indices.reshape(-1).to(dtype=torch.long)
    counts = option_counts.reshape(-1).to(dtype=torch.long)
    if bool(((counts <= 0) | (counts > max_options)).any()):
        raise ValueError("combo-state option count is outside prediction width")
    if bool(((selected_index < 0) | (selected_index >= counts)).any()):
        raise ValueError("combo-state selected option is outside legal row")
    top_target = top_deck_targets.reshape(-1).to(dtype=torch.long)
    seek_target = seek_source_targets.reshape(-1).to(dtype=torch.long)
    top_mask = _mask(top_deck_masks.reshape(-1), name="top_deck_masks")
    seek_mask = _mask(seek_source_masks.reshape(-1), name="seek_source_masks")
    binary_mask = _mask(vector_masks, name="vector_masks")
    if bool((top_mask & ((top_target < 0) | (top_target >= TOP_DECK_CLASSES))).any()):
        raise ValueError("top-deck class target is outside range")
    if bool(
        (
            seek_mask
            & ((seek_target < 0) | (seek_target >= SEEK_SOURCE_CLASSES))
        ).any()
    ):
        raise ValueError("seek-source class target is outside range")
    if bool(not torch.isfinite(vector_targets[binary_mask]).all()):
        raise ValueError("combo-state vector target is non-finite")
    values = vector_targets[binary_mask]
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("combo-state vector target is outside [0, 1]")
    confidence = guide_confidences.reshape(-1).float()
    if bool(not torch.isfinite(confidence).all()) or bool(
        ((confidence < 0.0) | (confidence > 1.0)).any()
    ):
        raise ValueError("combo-state guide confidence is outside [0, 1]")
    for name, weight in (
        ("base_loss_weight", base_loss_weight),
        ("guide_loss_weight", guide_loss_weight),
    ):
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    row_ids = torch.arange(rows, device=device)
    selected = predictions[row_ids, selected_index]
    zero = selected.sum() * 0.0

    def observed_loss(row_scale: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if bool(top_mask.any()):
            per_row = F.cross_entropy(
                selected[:, TOP_DECK_SLICE][top_mask],
                top_target[top_mask],
                reduction="none",
            )
            parts.append((per_row * row_scale[top_mask]).mean())
        if bool(seek_mask.any()):
            per_row = F.cross_entropy(
                selected[:, SEEK_SOURCE_SLICE][seek_mask],
                seek_target[seek_mask],
                reduction="none",
            )
            parts.append((per_row * row_scale[seek_mask]).mean())
        for _name, group in VECTOR_GROUPS:
            usable = binary_mask[:, group]
            if not bool(usable.any()):
                continue
            logits = selected[:, VECTOR_SLICE][:, group]
            target = vector_targets[:, group].to(dtype=logits.dtype)
            per_cell = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
            scaled = per_cell * row_scale[:, None].to(dtype=per_cell.dtype)
            parts.append(scaled[usable].mean())
        return torch.stack(parts).mean() if parts else zero

    base = observed_loss(torch.ones_like(confidence))
    guide = observed_loss(confidence)
    weighted = float(base_loss_weight) * base + float(guide_loss_weight) * guide
    eligible = top_mask | seek_mask | binary_mask.any(dim=-1)

    def class_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if not bool(mask.any()):
            return None
        return float(logits[mask].argmax(dim=-1).eq(target[mask]).float().mean())

    vector_brier: dict[str, float | None] = {}
    vector_labels: dict[str, int] = {}
    for name, group in VECTOR_GROUPS:
        usable = binary_mask[:, group]
        vector_labels[name] = int(usable.sum().item())
        if bool(usable.any()):
            probability = selected[:, VECTOR_SLICE][:, group].sigmoid()
            truth = vector_targets[:, group].to(dtype=probability.dtype)
            vector_brier[name] = float(
                ((probability - truth) ** 2)[usable].mean().detach()
            )
        else:
            vector_brier[name] = None

    metrics = ComboStateMetrics(
        total_rows=rows,
        eligible_rows=int(eligible.sum().item()),
        top_deck_labels=int(top_mask.sum().item()),
        seek_source_labels=int(seek_mask.sum().item()),
        vector_labels=vector_labels,
        guide_rows=int((eligible & confidence.gt(0.0)).sum().item()),
        base_observed_loss=float(base.detach()),
        guide_observed_loss=float(guide.detach()),
        weighted_loss=float(weighted.detach()),
        top_deck_accuracy=class_accuracy(
            selected[:, TOP_DECK_SLICE], top_target, top_mask
        ),
        seek_source_accuracy=class_accuracy(
            selected[:, SEEK_SOURCE_SLICE], seek_target, seek_mask
        ),
        vector_brier=vector_brier,
    )
    return weighted, metrics
