"""Turn-plan traces for offline distillation and audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from poke_bot.poke_rlm.plan_ir import TurnPlan, turn_plan_to_json


@dataclass
class TurnPlanTrace:
    """One turn of planner/executor behavior."""

    battle_id: str
    turn_index: int
    side_id: str
    observation_fingerprint: str
    legal_action_fingerprints: list[str]
    chosen_action_fingerprint: str
    plan: Optional[TurnPlan]
    mode: str
    route: str
    stop_reason: str
    budget_spent: dict[str, int] = field(default_factory=dict)
    executor_events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "turn_index": self.turn_index,
            "side_id": self.side_id,
            "observation_fingerprint": self.observation_fingerprint,
            "legal_action_fingerprints": list(self.legal_action_fingerprints),
            "chosen_action_fingerprint": self.chosen_action_fingerprint,
            "plan": turn_plan_to_json(self.plan) if self.plan is not None else None,
            "mode": self.mode,
            "route": self.route,
            "stop_reason": self.stop_reason,
            "budget_spent": dict(self.budget_spent),
            "executor_events": list(self.executor_events),
            "metadata": dict(self.metadata),
        }


def serialize_trace(trace: TurnPlanTrace) -> dict[str, Any]:
    """JSON-friendly serialization."""
    return trace.to_dict()
