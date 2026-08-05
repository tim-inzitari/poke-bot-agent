"""Supervised losses for Recursive Turn Planner multi-turn training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RTPLossBundle:
    total: Tensor
    action: Tensor
    complexity: Tensor
    dynamics: Tensor
    root_plan: Tensor
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "action": float(self.action.detach().item()),
            "complexity": float(self.complexity.detach().item()),
            "dynamics": float(self.dynamics.detach().item()),
            "root_plan": float(self.root_plan.detach().item()),
        }


def compute_rtp_losses(
    *,
    action_scores: Tensor,
    chosen_action_index: int,
    complexity_logit: Tensor,
    should_recurse: bool,
    predicted_next_latent: Optional[Tensor] = None,
    target_next_latent: Optional[Tensor] = None,
    root_plan_logits: Optional[Tensor] = None,
    root_plan_target: Optional[int] = None,
    action_weight: float = 1.0,
    complexity_weight: float = 0.25,
    dynamics_weight: float = 0.5,
    root_plan_weight: float = 0.15,
) -> RTPLossBundle:
    """Multi-turn planner losses for one decision (batch dim optional)."""
    if action_scores.dim() == 1:
        action_scores = action_scores.unsqueeze(0)
    if complexity_logit.dim() == 0:
        complexity_logit = complexity_logit.unsqueeze(0)
    elif complexity_logit.dim() == 2 and complexity_logit.size(-1) == 1:
        complexity_logit = complexity_logit.squeeze(-1)

    device = action_scores.device
    action_t = torch.tensor([int(chosen_action_index)], device=device, dtype=torch.long)
    # Clamp illegal indices fail-closed to 0 for CE stability.
    if int(chosen_action_index) < 0 or int(chosen_action_index) >= action_scores.size(-1):
        action_t = torch.zeros(1, device=device, dtype=torch.long)
    action_loss = F.cross_entropy(action_scores, action_t)

    recurse_t = torch.tensor(
        [1.0 if should_recurse else 0.0], device=device, dtype=torch.float32
    )
    complexity_loss = F.binary_cross_entropy_with_logits(
        complexity_logit.view(-1), recurse_t
    )

    if predicted_next_latent is not None and target_next_latent is not None:
        dynamics_loss = F.mse_loss(
            predicted_next_latent, target_next_latent.detach()
        )
    else:
        dynamics_loss = action_scores.new_zeros(())

    if (
        root_plan_logits is not None
        and root_plan_target is not None
        and int(root_plan_target) >= 0
    ):
        if root_plan_logits.dim() == 1:
            root_plan_logits = root_plan_logits.unsqueeze(0)
        root_t = torch.tensor(
            [int(root_plan_target) % root_plan_logits.size(-1)],
            device=device,
            dtype=torch.long,
        )
        root_loss = F.cross_entropy(root_plan_logits, root_t)
    else:
        root_loss = action_scores.new_zeros(())

    total = (
        float(action_weight) * action_loss
        + float(complexity_weight) * complexity_loss
        + float(dynamics_weight) * dynamics_loss
        + float(root_plan_weight) * root_loss
    )
    return RTPLossBundle(
        total=total,
        action=action_loss,
        complexity=complexity_loss,
        dynamics=dynamics_loss,
        root_plan=root_loss,
        metadata={"should_recurse": bool(should_recurse)},
    )
