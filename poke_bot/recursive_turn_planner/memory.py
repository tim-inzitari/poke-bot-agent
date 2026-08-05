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
    """

    H: Tensor
    spatial_memory: Optional[Tensor] = None
    value_estimate: float = 0.0
    legal_actions: tuple[tuple[int, ...], ...] = ()
    remaining_resources: dict[str, Any] = field(default_factory=dict)
    matchup_route: int = -1
    turn_objective: str = ""
    latent_state: Optional[Tensor] = None
    observations: dict[str, Any] = field(default_factory=dict)
    encode_pass_count: int = 1

    def __post_init__(self) -> None:
        if self.H.dim() != 1:
            raise ValueError("turn memory H must be a rank-1 state vector")
        if self.latent_state is None:
            self.latent_state = self.H.detach().clone()
        else:
            if self.latent_state.shape != self.H.shape:
                raise ValueError("latent_state must match H shape")

    @property
    def d_model(self) -> int:
        return int(self.H.numel())

    def update_observation(
        self,
        *,
        observations: Optional[dict[str, Any]] = None,
        legal_actions: Optional[tuple[tuple[int, ...], ...]] = None,
        remaining_resources: Optional[dict[str, Any]] = None,
        latent_delta: Optional[Tensor] = None,
    ) -> None:
        """Incrementally update working memory after a real observation."""
        if observations:
            self.observations.update(observations)
        if legal_actions is not None:
            self.legal_actions = tuple(legal_actions)
        if remaining_resources is not None:
            self.remaining_resources.update(remaining_resources)
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
