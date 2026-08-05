"""Phase-8 specialist adapter scaffolding (inactive by default)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SpecialistAdapterSpec:
    """Checksum-bound specialist adapter descriptor."""

    adapter_id: str
    archetype: str
    checksum: str
    rank: int = 8
    enabled: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class SpecialistAdapterRegistry:
    """Registry for future archetype adapters; never auto-activates."""

    adapters: dict[str, SpecialistAdapterSpec] = field(default_factory=dict)

    def register(self, spec: SpecialistAdapterSpec) -> None:
        self.adapters[spec.adapter_id] = spec

    def get(self, adapter_id: str) -> Optional[SpecialistAdapterSpec]:
        return self.adapters.get(adapter_id)

    def active(self) -> list[SpecialistAdapterSpec]:
        return [a for a in self.adapters.values() if a.enabled]

    def inventory(self) -> dict[str, Any]:
        return {
            "schema": "poke_bot.poke_rlm.specialists/v1",
            "count": len(self.adapters),
            "active": [a.adapter_id for a in self.active()],
            "adapters": {
                k: {
                    "archetype": v.archetype,
                    "checksum": v.checksum,
                    "rank": v.rank,
                    "enabled": v.enabled,
                }
                for k, v in self.adapters.items()
            },
        }
