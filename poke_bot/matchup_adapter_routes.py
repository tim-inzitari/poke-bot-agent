"""Shared logical-route and physical-slot validation for adapter checkpoints.

The public matchup tree predicts logical archetype identities.  V5 stores those
identities in one contiguous bank, while V6 keeps a fixed 64-slot physical bank
whose logical roster is carried by an embedded registry.  Runtime activation,
marker creation, and remote reloads must all resolve that distinction in the
same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as V5_ADAPTER_CHECKPOINT_FORMAT,
)
from .matchup_adapters import EXPERT_IDS


@dataclass(frozen=True)
class MatchupAdapterRouteContract:
    """Validated route identities and their checkpoint tensor slots."""

    adapter_format: str
    target_ids: tuple[str, ...]
    physical_slots: tuple[int, ...]
    slot_capacity: int
    slot_registry_digest: str | None

    @property
    def physical_slot_by_target(self) -> dict[str, int]:
        return dict(zip(self.target_ids, self.physical_slots, strict=True))

    def runtime_binding(self) -> dict[str, Any]:
        """Return the JSON-safe fields pinned into trees and remote markers."""

        result: dict[str, Any] = {
            "adapter_format": self.adapter_format,
            "route_target_ids": list(self.target_ids),
            "route_physical_slots": list(self.physical_slots),
            "physical_slot_capacity": int(self.slot_capacity),
        }
        if self.slot_registry_digest is not None:
            result["slot_registry_digest"] = self.slot_registry_digest
        return result


def resolve_matchup_adapter_route_contract(
    adapter_config: Mapping[str, Any],
) -> MatchupAdapterRouteContract:
    """Resolve one exact V5 or V6 serialized checkpoint route contract.

    V5 inference without an explicit ``format`` is retained only for the
    established roster-18 wire format.  V6 must checksum-bind its complete
    embedded registry and fixed physical capacity.
    """

    config = dict(adapter_config)
    adapter_format = str(config.get("format") or "")
    if not adapter_format and config.get("expert_ids"):
        adapter_format = V5_ADAPTER_CHECKPOINT_FORMAT

    if adapter_format == V5_ADAPTER_CHECKPOINT_FORMAT:
        targets = tuple(str(value) for value in config.get("expert_ids") or ())
        if targets != tuple(EXPERT_IDS):
            raise ValueError("checkpoint changed the canonical V5 route order")
        return MatchupAdapterRouteContract(
            adapter_format=adapter_format,
            target_ids=targets,
            physical_slots=tuple(range(len(targets))),
            slot_capacity=len(targets),
            slot_registry_digest=None,
        )

    from .matchup_adapters_v6 import (
        ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
    )
    from .matchup_adapters_v6 import (
        LEGACY_V5_PREFIX_LENGTH,
        SLOT_CAPACITY,
        load_slot_registry_dict,
        registry_digest,
        slot_map,
    )

    if adapter_format == V6_ADAPTER_CHECKPOINT_FORMAT:
        registry = load_slot_registry_dict(
            dict(config.get("slot_registry") or {})
        )
        if (
            registry.get("checkpoint_format") != V6_ADAPTER_CHECKPOINT_FORMAT
            or int(registry.get("legacy_v5_prefix_length") or -1)
            != LEGACY_V5_PREFIX_LENGTH
            or int(config.get("slot_capacity") or -1) != SLOT_CAPACITY
        ):
            raise ValueError("checkpoint V6 slot registry metadata changed")
        expected_digest = registry_digest(registry)
        if str(config.get("slot_registry_digest") or "") != expected_digest:
            raise ValueError("checkpoint V6 slot registry digest changed")
        targets = tuple(
            str(value) for value in registry.get("active_expert_ids") or ()
        )
        mapping = slot_map(registry)
        physical_slots = tuple(int(mapping[target]) for target in targets)
        if (
            not targets
            or len(targets) != len(set(targets))
            or len(physical_slots) != len(set(physical_slots))
            or any(slot < 0 or slot >= SLOT_CAPACITY for slot in physical_slots)
        ):
            raise ValueError("checkpoint V6 logical route mapping is invalid")
        return MatchupAdapterRouteContract(
            adapter_format=adapter_format,
            target_ids=targets,
            physical_slots=physical_slots,
            slot_capacity=SLOT_CAPACITY,
            slot_registry_digest=expected_digest,
        )

    raise ValueError("checkpoint has no supported matchup adapter contract")


def require_runtime_route_binding(
    payload: Mapping[str, Any],
    contract: MatchupAdapterRouteContract,
    *,
    allow_legacy_v5: bool = False,
) -> None:
    """Fail closed unless a tree/marker names the checkpoint's exact routes."""

    expected = contract.runtime_binding()
    if (
        allow_legacy_v5
        and contract.adapter_format == V5_ADAPTER_CHECKPOINT_FORMAT
        and not any(key in payload for key in expected)
    ):
        return
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"runtime matchup route binding changed: {key}")
    if (
        contract.slot_registry_digest is None
        and payload.get("slot_registry_digest") not in {None, ""}
    ):
        raise ValueError("V5 runtime cannot declare a V6 slot registry digest")


__all__ = [
    "MatchupAdapterRouteContract",
    "require_runtime_route_binding",
    "resolve_matchup_adapter_route_contract",
]
