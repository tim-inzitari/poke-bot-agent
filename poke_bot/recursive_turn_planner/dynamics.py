"""Learned latent transition evaluator for recursive planning.

Predicts decision-relevant successor representations after an action or short
action chunk. Exact legality stays outside this module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class LatentTransitionDynamics(nn.Module):
    """z_{t+1} = D(z_t, a_t) with value and uncertainty heads.

    Lightweight MLP over concatenated state and action embeddings. Compatible
    with the existing action-conditioned latent-lookahead role: internal plan
    evaluator, not a simulator replacement.
    """

    def __init__(
        self,
        d_model: int,
        *,
        width: int = 256,
        action_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or width <= 0:
            raise ValueError("dynamics dimensions must be positive")
        self.d_model = int(d_model)
        self.width = int(width)
        self.action_embed_dim = int(
            action_embed_dim if action_embed_dim is not None else d_model
        )
        in_dim = self.d_model + self.action_embed_dim
        self.input_norm = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
        )
        self.next_latent = nn.Linear(self.width, self.d_model)
        self.value_head = nn.Linear(self.width, 1)
        self.uncertainty_head = nn.Linear(self.width, 1)
        # Start uncertainty near zero so untrained dynamics do not dominate.
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def embed_action_ids(
        self,
        actions: tuple[tuple[int, ...], ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Deterministic bag embedding for typed action index tuples.

        This is a lightweight stand-in until option-hidden states from the
        shared encoder are supplied directly.
        """
        rows: list[Tensor] = []
        for action in actions:
            vec = torch.zeros(self.action_embed_dim, device=device, dtype=dtype)
            if not action:
                rows.append(vec)
                continue
            for rank, idx in enumerate(action):
                slot = int(idx) % self.action_embed_dim
                vec[slot] += 1.0 / float(rank + 1)
            rows.append(vec)
        return torch.stack(rows, dim=0)

    def forward(
        self,
        latent_state: Tensor,
        action_embed: Tensor,
    ) -> dict[str, Tensor]:
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        if action_embed.dim() == 1:
            action_embed = action_embed.unsqueeze(0)
        if latent_state.dim() != 2 or action_embed.dim() != 2:
            raise ValueError("dynamics expects [B,D] latent and action embeds")
        if latent_state.size(-1) != self.d_model:
            raise ValueError("latent width mismatch")
        if action_embed.size(-1) != self.action_embed_dim:
            raise ValueError("action embed width mismatch")
        # Broadcast a shared latent across a batch of candidate actions.
        if latent_state.size(0) == 1 and action_embed.size(0) > 1:
            latent_state = latent_state.expand(action_embed.size(0), -1)
        elif action_embed.size(0) == 1 and latent_state.size(0) > 1:
            action_embed = action_embed.expand(latent_state.size(0), -1)
        if latent_state.size(0) != action_embed.size(0):
            raise ValueError("dynamics batch mismatch")
        hidden = self.trunk(
            self.input_norm(torch.cat((latent_state, action_embed), dim=-1))
        )
        return {
            "next_latent": self.next_latent(hidden),
            "value": torch.tanh(self.value_head(hidden).squeeze(-1)),
            "uncertainty": torch.sigmoid(self.uncertainty_head(hidden).squeeze(-1)),
        }

    @torch.no_grad()
    def rollout_program_value(
        self,
        latent_state: Tensor,
        actions: tuple[tuple[int, ...], ...],
        *,
        max_horizon: int,
    ) -> dict[str, float]:
        """Short latent rollout over primitive actions in a plan."""
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        device = latent_state.device
        dtype = latent_state.dtype
        z = latent_state.detach().clone()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        values: list[float] = []
        uncertainties: list[float] = []
        for step, action in enumerate(actions[:max_horizon]):
            embed = self.embed_action_ids((action,), device=device, dtype=dtype)
            out = self.forward(z, embed)
            z = out["next_latent"]
            values.append(float(out["value"][0].item()))
            uncertainties.append(float(out["uncertainty"][0].item()))
            if step + 1 >= max_horizon:
                break
        if not values:
            return {"value": 0.0, "uncertainty": 1.0, "horizon": 0.0}
        # Prefer terminal chunk value; average uncertainty over the short path.
        return {
            "value": values[-1],
            "uncertainty": sum(uncertainties) / len(uncertainties),
            "horizon": float(len(values)),
        }
