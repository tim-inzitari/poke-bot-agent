"""Shadow traces and turn-level metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .budget import TurnComputeBudget
from .reasons import ReasonCode


@dataclass
class ShadowTrace:
    enabled: bool = False
    mode: str = "disabled"
    turn_key: tuple[int, int] = (-1, -1)
    route: str = ""
    reason: ReasonCode = ReasonCode.DISABLED
    selected_action: tuple[int, ...] = ()
    planner_action: tuple[int, ...] = ()
    used_for_selection: bool = False
    observation_hash: str = ""
    plan_id: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "turn_key": list(self.turn_key),
            "route": self.route,
            "reason": self.reason.value,
            "selected_action": list(self.selected_action),
            "planner_action": list(self.planner_action),
            "used_for_selection": self.used_for_selection,
            "observation_hash": self.observation_hash,
            "plan_id": self.plan_id,
            "budget": dict(self.budget),
            "extras": dict(self.extras),
        }


def budget_to_trace(budget: Optional[TurnComputeBudget]) -> dict[str, Any]:
    if budget is None:
        return {}
    return budget.snapshot()
