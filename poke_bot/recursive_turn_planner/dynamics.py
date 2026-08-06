"""Learned latent transition evaluator for recursive planning.

Predicts decision-relevant successor representations after an action or short
action chunk. Exact legality stays outside this module.

Preferred production path: feed ``option_hidden`` from
``TemporalCabtTransformer.decode_options(..., return_hidden=True)``.
Deterministic action-id embeddings remain only as a lightweight stand-in.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class _LookaheadModule(Protocol):
    width: int

    def __call__(
        self, option_hidden: Tensor, state_vec: Tensor
    ) -> dict[str, Tensor]: ...

    def inventory(self, *, action_authority_enabled: bool) -> dict[str, object]: ...


class LatentTransitionDynamics(nn.Module):
    """z_{t+1} = D(z_t, a_t) with value and uncertainty heads.

    Shape contract mirrors ``ActionConditionedLatentLookahead``:
    concat(state, action_embed) → width trunk → next latent / value.
    Uncertainty is RTP-specific for plan selection under compute penalty.
    """

    def __init__(
        self,
        d_model: int,
        *,
        width: int = 512,
        action_embed_dim: Optional[int] = None,
        prefer_option_hidden: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or width <= 0:
            raise ValueError("dynamics dimensions must be positive")
        self.d_model = int(d_model)
        self.width = int(width)
        self.action_embed_dim = int(
            action_embed_dim if action_embed_dim is not None else d_model
        )
        if self.action_embed_dim != self.d_model and prefer_option_hidden:
            raise ValueError(
                "prefer_option_hidden requires action_embed_dim == d_model"
            )
        self.prefer_option_hidden = bool(prefer_option_hidden)
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

    def inventory(self) -> dict[str, object]:
        return {
            "schema": "poke_bot.recursive_turn_planner.dynamics/v1",
            "d_model": self.d_model,
            "width": self.width,
            "action_embed_dim": self.action_embed_dim,
            "prefer_option_hidden": self.prefer_option_hidden,
            "neural_only": True,
            "mcts_allowed": False,
            "beam_search_allowed": False,
            "competition_time_simulator_search_allowed": False,
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "reuse_note": (
                "Prefer option_hidden from decode_options; optional future "
                "adapter over ActionConditionedLatentLookahead."
            ),
        }

    def embed_action_ids(
        self,
        actions: tuple[tuple[int, ...], ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Deterministic bag embedding for typed action index tuples.

        Stand-in only. Production-shaped scoring should pass option_hidden.
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

    def resolve_action_embeds(
        self,
        actions: Sequence[Sequence[int]],
        *,
        option_hidden: Optional[Tensor] = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, str]:
        """Choose option_hidden when available and correctly shaped."""
        action_tuple = tuple(tuple(int(x) for x in a) for a in actions)
        if (
            self.prefer_option_hidden
            and option_hidden is not None
            and option_hidden.dim() == 2
            and option_hidden.size(0) == len(action_tuple)
            and option_hidden.size(1) == self.action_embed_dim
        ):
            return option_hidden.to(device=device, dtype=dtype), "option_hidden"
        return (
            self.embed_action_ids(action_tuple, device=device, dtype=dtype),
            "action_id_hash",
        )

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
    def score_actions(
        self,
        latent_state: Tensor,
        actions: Sequence[Sequence[int]],
        *,
        option_hidden: Optional[Tensor] = None,
    ) -> dict[str, Tensor | str]:
        """Batched one-step scores for a legal action set."""
        device = latent_state.device
        dtype = latent_state.dtype
        embeds, source = self.resolve_action_embeds(
            actions,
            option_hidden=option_hidden,
            device=device,
            dtype=dtype,
        )
        out = self.forward(latent_state, embeds)
        return {
            "next_latent": out["next_latent"],
            "value": out["value"],
            "uncertainty": out["uncertainty"],
            "action_embed_source": source,
        }

    @torch.no_grad()
    def rollout_program_value(
        self,
        latent_state: Tensor,
        actions: tuple[tuple[int, ...], ...],
        *,
        max_horizon: int,
        option_hidden_by_action: Optional[Mapping[tuple[int, ...], Tensor]] = None,
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
        sources: list[str] = []
        for step, action in enumerate(actions[:max_horizon]):
            option_hidden = None
            if option_hidden_by_action is not None and action in option_hidden_by_action:
                option_hidden = option_hidden_by_action[action].unsqueeze(0)
            embeds, source = self.resolve_action_embeds(
                (action,),
                option_hidden=option_hidden,
                device=device,
                dtype=dtype,
            )
            sources.append(source)
            out = self.forward(z, embeds)
            z = out["next_latent"]
            values.append(float(out["value"][0].item()))
            uncertainties.append(float(out["uncertainty"][0].item()))
            if step + 1 >= max_horizon:
                break
        if not values:
            return {
                "value": 0.0,
                "uncertainty": 1.0,
                "horizon": 0.0,
                "option_hidden_fraction": 0.0,
            }
        hidden_frac = (
            sum(1 for s in sources if s == "option_hidden") / float(len(sources))
        )
        return {
            "value": values[-1],
            "uncertainty": sum(uncertainties) / len(uncertainties),
            "horizon": float(len(values)),
            "option_hidden_fraction": float(hidden_frac),
        }


class LookaheadBackedDynamics(nn.Module):
    """Adapter that reuses ``ActionConditionedLatentLookahead`` as D(·).

    Keeps RTP's uncertainty/compute-penalty interface while avoiding a second
    incompatible latent evaluator when the parent model already has lookahead.
    Action-id hashing falls back only when option_hidden is unavailable.
    """

    def __init__(
        self,
        lookahead: _LookaheadModule,
        *,
        d_model: int,
        uncertainty_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.lookahead = lookahead  # type: ignore[assignment]
        self.d_model = int(d_model)
        self.prefer_option_hidden = True
        self.action_embed_dim = int(d_model)
        self.width = int(getattr(lookahead, "width", 2 * d_model))
        self.uncertainty_bias = float(uncertainty_bias)
        self._id_embedder = LatentTransitionDynamics(
            d_model,
            width=max(32, self.width),
            prefer_option_hidden=False,
        )
        self._resolver = LatentTransitionDynamics(
            d_model,
            width=max(32, self.width),
            prefer_option_hidden=True,
        )

    def inventory(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "schema": "poke_bot.recursive_turn_planner.lookahead_dynamics/v1",
            "backed_by": "ActionConditionedLatentLookahead",
            "d_model": self.d_model,
            "width": self.width,
            "prefer_option_hidden": True,
            "parameters": int(sum(p.numel() for p in self.parameters())),
        }
        inv = getattr(self.lookahead, "inventory", None)
        if callable(inv):
            base["lookahead_inventory"] = inv(action_authority_enabled=False)
        return base

    def embed_action_ids(self, *args: Any, **kwargs: Any) -> Tensor:
        return self._id_embedder.embed_action_ids(*args, **kwargs)

    def resolve_action_embeds(self, *args: Any, **kwargs: Any) -> tuple[Tensor, str]:
        return self._resolver.resolve_action_embeds(*args, **kwargs)

    def forward(
        self,
        latent_state: Tensor,
        action_embed: Tensor,
    ) -> dict[str, Tensor]:
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        if action_embed.dim() == 1:
            action_embed = action_embed.unsqueeze(0)
        if action_embed.dim() != 2:
            raise ValueError("lookahead dynamics expects [N,D] action embeds")
        option_hidden = action_embed.unsqueeze(0)  # [1,N,D]
        out = self.lookahead(option_hidden, latent_state)
        value = out["continuation_value"].squeeze(0)
        next_latent = out["predicted_next_state_latent"].squeeze(0)
        uncertainty = torch.sigmoid(
            torch.full_like(value, self.uncertainty_bias)
        )
        return {
            "next_latent": next_latent,
            "value": value,
            "uncertainty": uncertainty,
        }

    @torch.no_grad()
    def score_actions(
        self,
        latent_state: Tensor,
        actions: Sequence[Sequence[int]],
        *,
        option_hidden: Optional[Tensor] = None,
    ) -> dict[str, Tensor | str]:
        device = latent_state.device
        dtype = latent_state.dtype
        embeds, source = self.resolve_action_embeds(
            actions,
            option_hidden=option_hidden,
            device=device,
            dtype=dtype,
        )
        if source != "option_hidden":
            # Without encoder option states, fall back to local MLP dynamics.
            return self._id_embedder.score_actions(
                latent_state, actions, option_hidden=None
            )
        out = self.forward(latent_state, embeds)
        return {
            "next_latent": out["next_latent"],
            "value": out["value"],
            "uncertainty": out["uncertainty"],
            "action_embed_source": source,
        }

    @torch.no_grad()
    def rollout_program_value(
        self,
        latent_state: Tensor,
        actions: tuple[tuple[int, ...], ...],
        *,
        max_horizon: int,
        option_hidden_by_action: Optional[Mapping[tuple[int, ...], Tensor]] = None,
    ) -> dict[str, float]:
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        z = latent_state.detach().clone()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        values: list[float] = []
        uncertainties: list[float] = []
        sources: list[str] = []
        for step, action in enumerate(actions[:max_horizon]):
            option_hidden = None
            if option_hidden_by_action is not None and action in option_hidden_by_action:
                option_hidden = option_hidden_by_action[action].unsqueeze(0)
            scored = self.score_actions(z, (action,), option_hidden=option_hidden)
            source = str(scored["action_embed_source"])
            sources.append(source)
            next_latent = scored["next_latent"]
            assert isinstance(next_latent, Tensor)
            value = scored["value"]
            uncertainty = scored["uncertainty"]
            assert isinstance(value, Tensor) and isinstance(uncertainty, Tensor)
            z = next_latent if next_latent.dim() == 2 else next_latent.unsqueeze(0)
            values.append(float(value.reshape(-1)[0].item()))
            uncertainties.append(float(uncertainty.reshape(-1)[0].item()))
            if step + 1 >= max_horizon:
                break
        if not values:
            return {
                "value": 0.0,
                "uncertainty": 1.0,
                "horizon": 0.0,
                "option_hidden_fraction": 0.0,
            }
        hidden_frac = (
            sum(1 for s in sources if s == "option_hidden") / float(len(sources))
        )
        return {
            "value": values[-1],
            "uncertainty": sum(uncertainties) / len(uncertainties),
            "horizon": float(len(values)),
            "option_hidden_fraction": float(hidden_frac),
        }
