"""Composite losses for PokeRLM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from poke_bot.poke_rlm.training.labels import PlanSupervisionLabels


@dataclass
class PokeRLMLossBundle:
    """Named loss terms + total."""

    total: torch.Tensor
    action: torch.Tensor
    route: torch.Tensor
    recurse: torch.Tensor
    dynamics: torch.Tensor
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "action": float(self.action.detach().item()),
            "route": float(self.route.detach().item()),
            "recurse": float(self.recurse.detach().item()),
            "dynamics": float(self.dynamics.detach().item()),
        }


# Router emits "root"; older labels may say "root_plan".
_ROUTE_TO_IDX = {
    "direct": 0,
    "root": 1,
    "root_plan": 1,
    "recursive": 2,
}


def compute_poke_rlm_losses(
    *,
    action_logits: torch.Tensor,
    route_logits: torch.Tensor,
    recurse_logits: torch.Tensor,
    labels: PlanSupervisionLabels,
    predicted_next_latent: torch.Tensor | None = None,
    target_next_latent: torch.Tensor | None = None,
    action_weight: float = 1.0,
    route_weight: float = 0.25,
    recurse_weight: float = 0.25,
    dynamics_weight: float = 0.5,
) -> PokeRLMLossBundle:
    """Compute supervised losses for one example (batch dim optional)."""
    if action_logits.dim() == 1:
        action_logits = action_logits.unsqueeze(0)
        route_logits = route_logits.unsqueeze(0)
        recurse_logits = recurse_logits.unsqueeze(0)

    device = action_logits.device
    action_t = torch.tensor([labels.chosen_action_index], device=device, dtype=torch.long)
    route_t = torch.tensor([_ROUTE_TO_IDX.get(labels.route_target, 0)], device=device, dtype=torch.long)
    recurse_t = torch.tensor(
        [1.0 if labels.should_recurse else 0.0],
        device=device,
        dtype=torch.float32,
    )

    action_loss = F.cross_entropy(action_logits, action_t)
    route_loss = F.cross_entropy(route_logits, route_t)
    recurse_loss = F.binary_cross_entropy_with_logits(recurse_logits.view(-1), recurse_t)

    if predicted_next_latent is not None and target_next_latent is not None:
        dynamics_loss = F.mse_loss(predicted_next_latent, target_next_latent.detach())
    else:
        dynamics_loss = action_logits.new_zeros(())

    total = (
        float(action_weight) * action_loss
        + float(route_weight) * route_loss
        + float(recurse_weight) * recurse_loss
        + float(dynamics_weight) * dynamics_loss
    )
    return PokeRLMLossBundle(
        total=total,
        action=action_loss,
        route=route_loss,
        recurse=recurse_loss,
        dynamics=dynamics_loss,
        metadata={"stop_reason": labels.stop_reason},
    )
