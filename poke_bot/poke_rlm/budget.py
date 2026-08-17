"""Whole-turn compute budget shared across micro-decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .reasons import ReasonCode


@dataclass
class TurnComputeBudget:
    """Hard caps for one acting-seat turn.

    All counters are fail-closed: exceeding a cap returns a reason code and
    must stop optional compute.
    """

    max_depth: int = 2
    max_nodes: int = 32
    max_subgoals: int = 8
    max_model_calls: int = 4
    max_simulator_calls: int = 16
    legacy_hard_cap_simulator_calls: int = 75
    deadline_ns: Optional[int] = None
    repair_reserve_calls: int = 4

    model_calls: int = 0
    simulator_calls: int = 0
    nodes_emitted: int = 0
    subgoals_refined: int = 0
    depth_reached: int = 0
    repairs: int = 0
    started_ns: int = field(default_factory=time.monotonic_ns)
    extras: dict[str, Any] = field(default_factory=dict)

    def remaining_model_calls(self) -> int:
        return max(0, self.max_model_calls - self.model_calls)

    def remaining_simulator_calls(self) -> int:
        return max(0, self.max_simulator_calls - self.simulator_calls)

    def consume_model_call(self, n: int = 1) -> ReasonCode:
        if self.model_calls + n > self.max_model_calls:
            return ReasonCode.BUDGET_MODEL_CALLS
        self.model_calls += n
        return ReasonCode.OK

    def consume_simulator_call(self, n: int = 1) -> ReasonCode:
        if self.simulator_calls + n > self.max_simulator_calls:
            return ReasonCode.BUDGET_SIMULATOR_CALLS
        if self.simulator_calls + n > self.legacy_hard_cap_simulator_calls:
            return ReasonCode.BUDGET_SIMULATOR_CALLS
        self.simulator_calls += n
        return ReasonCode.OK

    def emit_nodes(self, n: int = 1) -> ReasonCode:
        if self.nodes_emitted + n > self.max_nodes:
            return ReasonCode.BUDGET_NODES
        self.nodes_emitted += n
        return ReasonCode.OK

    def note_depth(self, depth: int) -> ReasonCode:
        if depth > self.max_depth:
            return ReasonCode.BUDGET_DEPTH
        self.depth_reached = max(self.depth_reached, depth)
        return ReasonCode.OK

    def check_deadline(self) -> ReasonCode:
        if self.deadline_ns is None:
            return ReasonCode.OK
        if time.monotonic_ns() >= int(self.deadline_ns):
            return ReasonCode.BUDGET_DEADLINE
        return ReasonCode.OK

    def snapshot(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "simulator_calls": self.simulator_calls,
            "nodes_emitted": self.nodes_emitted,
            "subgoals_refined": self.subgoals_refined,
            "depth_reached": self.depth_reached,
            "repairs": self.repairs,
            "max_model_calls": self.max_model_calls,
            "max_simulator_calls": self.max_simulator_calls,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "elapsed_ms": (time.monotonic_ns() - self.started_ns) / 1e6,
        }
