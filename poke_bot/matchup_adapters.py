"""Dormant, exact-match residual adapters for policy/value state.

The routing order and adapter dimensions are a checkpoint contract.  Offline
bootstrap routes come only from audited oracle tickets. Runtime routes, if a
later gate enables them, must come from a separate public-prefix recognizer;
unknown identifiers use ``UNKNOWN_ROUTE`` and bypass the bank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


HIDDEN_DIM = 96
BOTTLENECK_DIM = 8
RESIDUAL_SCALE = 0.25
UNKNOWN_ROUTE = -1

# Order is part of the adapter checkpoint/routing contract.  Do not derive it
# from the mutable archetype registry or corpus frequency.  New routes append
# only so all seven v1 route indices remain stable.
LEGACY_EXPERT_IDS_V1: tuple[str, ...] = (
    "crustle",
    "marnie-s-grimmsnarl-ex",
    "garchomp",
    "cornerstone-ogerpon",
    "rockets-mewtwo",
    "starmie",
    "hammer-pult",
)
EXPERT_IDS_V2: tuple[str, ...] = LEGACY_EXPERT_IDS_V1 + (
    "alakazam",
    "lucario",
    "archaludon-ex",
)
EXPERT_IDS_V3: tuple[str, ...] = EXPERT_IDS_V2 + (
    "dragapult-dudunsparce",
    "dragapult",
    "dudunsparce",
    "hops-trevenant",
    "walrein",
)
# Historical v4 route order. This remains available only for explicit,
# checksum-audited checkpoint migration; it is not a live routing roster.
LEGACY_EXPERT_IDS_V4: tuple[str, ...] = EXPERT_IDS_V3 + (
    "dragapult-dusknoir",
    "dragapult-blaziken",
    "lopunny",
    "gardevoir",
    "ns-zoroark",
    "raging-bolt",
    "festival-lead",
)

RETIRED_EXPERT_IDS_V5: frozenset[str] = frozenset(
    {
        "cornerstone-ogerpon",
        "lopunny",
        "gardevoir",
        "ns-zoroark",
        "raging-bolt",
    }
)

# Stable canonical v5 roster. The five retired rows are physically absent,
# Festival Lead's retained weights are addressed as Thwackey, and Spidops is a
# new exact-zero row. Frozen v4 checkpoints remain immutable and load only
# through the explicit identity-preserving migration tool.
EXPERT_IDS: tuple[str, ...] = (
    "crustle",
    "marnie-s-grimmsnarl-ex",
    "garchomp",
    "rockets-mewtwo",
    "starmie",
    "hammer-pult",
    "alakazam",
    "lucario",
    "archaludon-ex",
    "dragapult-dudunsparce",
    "dragapult",
    "dudunsparce",
    "hops-trevenant",
    "walrein",
    "dragapult-dusknoir",
    "dragapult-blaziken",
    "thwackey",
    "team-rockets-spidops",
)
STAGED_EXPERT_IDS_V5: tuple[str, ...] = EXPERT_IDS
ACTIVE_EXPERT_IDS_V5: tuple[str, ...] = EXPERT_IDS
LOGICAL_EXPERT_ALIASES_V5: Mapping[str, str] = MappingProxyType(
    {"festival-lead": "thwackey"}
)
EXPERT_ID_TO_ROUTE: Mapping[str, int] = MappingProxyType(
    {expert_id: route for route, expert_id in enumerate(EXPERT_IDS)}
)

ADAPTER_CHECKPOINT_FORMAT = "poke-bot-matchup-adapter-bank-v5-roster18"
V4_ADAPTER_CHECKPOINT_FORMAT = "alakazam-matchup-adapter-bank-v4"
V3_ADAPTER_CHECKPOINT_FORMAT = "alakazam-matchup-adapter-bank-v3"
V2_ADAPTER_CHECKPOINT_FORMAT = "alakazam-matchup-adapter-bank-v2"
LEGACY_ADAPTER_CHECKPOINT_FORMAT = "alakazam-matchup-adapter-bank-v1"
ZERO_DORMANT_CHECKPOINT_SCHEMA = "poke_bot.zero_dormant_matchup_adapter/v1"


def route_for_archetype(archetype_id: str | None) -> int:
    """Return the pinned route, or ``UNKNOWN_ROUTE`` for an exact non-match."""

    if archetype_id is None:
        return UNKNOWN_ROUTE
    return EXPERT_ID_TO_ROUTE.get(archetype_id, UNKNOWN_ROUTE)


def archetype_for_route(route: int) -> str | None:
    """Return the pinned matchup identifier for a route; ``-1`` is unknown."""

    route = int(route)
    if route == UNKNOWN_ROUTE:
        return None
    if route < 0 or route >= len(EXPERT_IDS):
        raise ValueError(
            f"route must be -1 or in [0, {len(EXPERT_IDS) - 1}], got {route}"
        )
    return EXPERT_IDS[route]


def routes_for_archetypes(
    archetype_ids: Sequence[str | None],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Map exact matchup identifiers to a one-dimensional route tensor."""

    return torch.tensor(
        [route_for_archetype(archetype_id) for archetype_id in archetype_ids],
        dtype=torch.long,
        device=device,
    )


class _MatchupExpert(nn.Module):
    """One fixed-width ``96 -> 8 -> 96`` residual expert."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM, bias=True)
        self.up = nn.Linear(BOTTLENECK_DIM, HIDDEN_DIM, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, state: Tensor) -> Tensor:
        normalized = F.layer_norm(
            state,
            normalized_shape=(HIDDEN_DIM,),
            weight=None,
            bias=None,
        )
        return self.up(F.gelu(self.down(normalized)))


class MatchupAdapterBank(nn.Module):
    """Append-only sparsely executed residual experts over 96-D states.

    Disabled banks and route tensors containing only ``UNKNOWN_ROUTE`` return
    the original tensor object.  Sparse dispatch also leaves unselected expert
    gradients as ``None``, which prevents AdamW weight decay from changing an
    unrelated expert.
    """

    hidden_dim = HIDDEN_DIM
    bottleneck_dim = BOTTLENECK_DIM
    residual_scale = RESIDUAL_SCALE
    expert_ids = EXPERT_IDS

    def __init__(self, *, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.experts = nn.ModuleList(_MatchupExpert() for _ in EXPERT_IDS)

    @staticmethod
    def config_dict() -> dict[str, Any]:
        """Return the immutable architecture and routing contract."""

        return {
            "format": ADAPTER_CHECKPOINT_FORMAT,
            "hidden_dim": HIDDEN_DIM,
            "bottleneck_dim": BOTTLENECK_DIM,
            "residual_scale": RESIDUAL_SCALE,
            "expert_ids": list(EXPERT_IDS),
            "unknown_route": UNKNOWN_ROUTE,
            "normalization": "parameter-free-layer-norm",
            "activation": "gelu",
            "residual_activation": "tanh",
        }

    @staticmethod
    def legacy_config_dict_v1() -> dict[str, Any]:
        """Return the exact append-compatible seven-route contract."""

        return {
            **MatchupAdapterBank.config_dict(),
            "format": LEGACY_ADAPTER_CHECKPOINT_FORMAT,
            "expert_ids": list(LEGACY_EXPERT_IDS_V1),
        }

    @staticmethod
    def legacy_config_dict_v2() -> dict[str, Any]:
        """Return the exact append-compatible ten-route contract."""

        return {
            **MatchupAdapterBank.config_dict(),
            "format": V2_ADAPTER_CHECKPOINT_FORMAT,
            "expert_ids": list(EXPERT_IDS_V2),
        }

    @staticmethod
    def legacy_config_dict_v3() -> dict[str, Any]:
        """Return the exact append-compatible fifteen-route contract."""

        return {
            **MatchupAdapterBank.config_dict(),
            "format": V3_ADAPTER_CHECKPOINT_FORMAT,
            "expert_ids": list(EXPERT_IDS_V3),
        }

    @staticmethod
    def legacy_config_dict_v4() -> dict[str, Any]:
        """Return the historical 22-route contract for migration audits."""

        return {
            **MatchupAdapterBank.config_dict(),
            "format": V4_ADAPTER_CHECKPOINT_FORMAT,
            "expert_ids": list(LEGACY_EXPERT_IDS_V4),
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Treat a wholly absent bank as a legacy zero-init checkpoint.

        Partial adapter state remains an error.  Filling only a wholly absent
        bank makes old model checkpoints strict-loadable while preserving the
        freshly constructed zero-output initialization.
        """

        has_adapter_state = any(key.startswith(prefix) for key in state_dict)
        if not has_adapter_state:
            for key, value in self.state_dict().items():
                state_dict[prefix + key] = value.detach().clone()
        # A partially present bank is intentionally not expanded. The v5
        # roster deletes and renames rows, so index-prefix migration would
        # silently attach trained weights to the wrong archetype. Historical
        # banks must pass through the checksum-audited identity migration tool.
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def adapter_checkpoint(self) -> dict[str, Any]:
        """Return an adapter-only payload with the pinned route contract."""

        return {
            "config": self.config_dict(),
            "adapter_state_dict": self.state_dict(),
        }

    def load_adapter_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        strict: bool = True,
    ):
        """Validate route order and load an adapter-only payload."""

        config = payload.get("config")
        expected = self.config_dict()
        if config != expected:
            raise ValueError(
                "adapter checkpoint config does not match the pinned v2 contract: "
                f"expected {expected!r}, got {config!r}"
            )
        state_dict = payload.get("adapter_state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("adapter checkpoint lacks an adapter_state_dict mapping")
        return self.load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        state: Tensor,
        routes: Tensor | Sequence[int],
        *,
        enabled: bool | None = None,
    ) -> Tensor:
        """Apply only experts selected by ``routes``; ``-1`` bypasses."""

        active = self.enabled if enabled is None else bool(enabled)
        if not active:
            return state
        if state.ndim < 1 or state.shape[-1] != HIDDEN_DIM:
            raise ValueError(
                f"state must end in hidden dimension {HIDDEN_DIM}, got {tuple(state.shape)}"
            )

        route_tensor = torch.as_tensor(routes, dtype=torch.long, device=state.device)
        expected_shape = state.shape[:-1]
        if route_tensor.shape != expected_shape:
            raise ValueError(
                f"routes shape must equal state.shape[:-1] ({tuple(expected_shape)}), "
                f"got {tuple(route_tensor.shape)}"
            )

        flat_routes = route_tensor.reshape(-1)
        if flat_routes.numel() == 0:
            return state
        invalid = (flat_routes < UNKNOWN_ROUTE) | (flat_routes >= len(EXPERT_IDS))
        if bool(torch.any(invalid).item()):
            invalid_values = torch.unique(flat_routes[invalid]).detach().cpu().tolist()
            raise ValueError(
                f"routes contain invalid values {invalid_values}; expected -1 or "
                f"[0, {len(EXPERT_IDS) - 1}]"
            )

        selected_routes = torch.unique(flat_routes[flat_routes >= 0])
        if selected_routes.numel() == 0:
            return state

        flat_state = state.reshape(-1, HIDDEN_DIM)
        adapted = flat_state
        for route in selected_routes.detach().cpu().tolist():
            rows = torch.nonzero(flat_routes == route, as_tuple=False).flatten()
            selected = flat_state.index_select(0, rows)
            residual = self.experts[route](selected)
            updated = selected + RESIDUAL_SCALE * torch.tanh(residual)
            adapted = adapted.index_copy(0, rows, updated)
        return adapted.reshape_as(state)
