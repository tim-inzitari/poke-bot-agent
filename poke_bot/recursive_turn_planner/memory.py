"""Persistent turn memory: encode once, update incrementally."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class PersistentTurnMemory:
    """Cached turn-start encoding and working state for plan persistence.

    ``H`` is the shared state memory produced near the start of the turn. Later
    atomic decisions should reuse it and apply only incremental updates from
    real observations, not full backbone re-encodes.

    When available, ``option_hidden`` should be the packed legal-option states
    from ``decode_options(..., return_hidden=True)``, shape ``[N, D]``.
    """

    H: Tensor
    spatial_memory: Optional[Tensor] = None
    option_hidden: Optional[Tensor] = None
    value_estimate: float = 0.0
    legal_actions: tuple[tuple[int, ...], ...] = ()
    remaining_resources: dict[str, Any] = field(default_factory=dict)
    matchup_route: int = -1
    turn_objective: str = ""
    latent_state: Optional[Tensor] = None
    observations: dict[str, Any] = field(default_factory=dict)
    encode_pass_count: int = 1
    sizing_profile: str = ""

    def __post_init__(self) -> None:
        if self.H.dim() != 1:
            raise ValueError("turn memory H must be a rank-1 state vector")
        if self.latent_state is None:
            self.latent_state = self.H.detach().clone()
        else:
            if self.latent_state.shape != self.H.shape:
                raise ValueError("latent_state must match H shape")
        if self.option_hidden is not None:
            if self.option_hidden.dim() != 2:
                raise ValueError("option_hidden must be [N, D]")
            if self.option_hidden.size(-1) != self.H.numel():
                raise ValueError("option_hidden width must match H")
            if self.legal_actions and self.option_hidden.size(0) != len(
                self.legal_actions
            ):
                raise ValueError("option_hidden rows must match legal_actions")

    @property
    def d_model(self) -> int:
        return int(self.H.numel())

    def option_hidden_for(
        self,
        action: tuple[int, ...],
    ) -> Optional[Tensor]:
        if self.option_hidden is None or not self.legal_actions:
            return None
        try:
            idx = self.legal_actions.index(action)
        except ValueError:
            return None
        return self.option_hidden[idx]

    def option_hidden_map(self) -> dict[tuple[int, ...], Tensor]:
        if self.option_hidden is None:
            return {}
        return {
            action: self.option_hidden[i]
            for i, action in enumerate(self.legal_actions)
        }

    def update_observation(
        self,
        *,
        observations: Optional[dict[str, Any]] = None,
        legal_actions: Optional[tuple[tuple[int, ...], ...]] = None,
        remaining_resources: Optional[dict[str, Any]] = None,
        latent_delta: Optional[Tensor] = None,
        option_hidden: Optional[Tensor] = None,
    ) -> None:
        """Incrementally update working memory after a real observation."""
        if observations:
            self.observations.update(observations)
        if legal_actions is not None:
            self.legal_actions = tuple(legal_actions)
        if remaining_resources is not None:
            self.remaining_resources.update(remaining_resources)
        if option_hidden is not None:
            if option_hidden.dim() != 2:
                raise ValueError("option_hidden must be [N, D]")
            if option_hidden.size(-1) != self.d_model:
                raise ValueError("option_hidden width must match H")
            if self.legal_actions and option_hidden.size(0) != len(self.legal_actions):
                raise ValueError("option_hidden rows must match legal_actions")
            self.option_hidden = option_hidden.detach().clone()
        if latent_delta is not None:
            if self.latent_state is None:
                raise RuntimeError("latent_state missing from turn memory")
            if latent_delta.shape != self.latent_state.shape:
                raise ValueError("latent_delta shape mismatch")
            self.latent_state = self.latent_state + latent_delta

    def snapshot_latent(self) -> Tensor:
        if self.latent_state is None:
            raise RuntimeError("latent_state missing from turn memory")
        return self.latent_state.detach().clone()
