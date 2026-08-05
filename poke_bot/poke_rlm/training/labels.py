"""Supervision labels derived from turn-plan traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from poke_bot.poke_rlm.plan_ir import PrimitiveAction, SubgoalNode, TurnPlan
from poke_bot.poke_rlm.training.traces import TurnPlanTrace


@dataclass
class PlanSupervisionLabels:
    """Dense labels for root / recursive / executor heads."""

    chosen_action_index: int
    route_target: str
    should_recurse: bool
    stop_reason: str
    node_kinds: list[str] = field(default_factory=list)
    commit_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def labels_from_plan(
    *,
    plan: Optional[TurnPlan],
    chosen_action_index: int,
    route_target: str,
    stop_reason: str,
    metadata: Optional[dict[str, Any]] = None,
) -> PlanSupervisionLabels:
    node_kinds: list[str] = []
    commit_indices: list[int] = []
    should_recurse = False
    if plan is not None:
        should_recurse = plan.recursion_depth > 0
        for node in plan.iter_nodes():
            kind = getattr(node, "kind", type(node).__name__)
            node_kinds.append(str(kind))
            if isinstance(node, SubgoalNode):
                should_recurse = True
            if isinstance(node, PrimitiveAction) and node.action_ref is not None:
                commit_indices.append(int(node.action_ref))
    return PlanSupervisionLabels(
        chosen_action_index=int(chosen_action_index),
        route_target=str(route_target),
        should_recurse=bool(should_recurse),
        stop_reason=str(stop_reason),
        node_kinds=node_kinds,
        commit_indices=commit_indices,
        metadata=dict(metadata or {}),
    )


def build_labels_from_trace(
    *,
    trace: TurnPlanTrace,
    chosen_action_index: int,
) -> PlanSupervisionLabels:
    """Build training labels from an executed plan trace."""
    return labels_from_plan(
        plan=trace.plan,
        chosen_action_index=chosen_action_index,
        route_target=trace.route,
        stop_reason=trace.stop_reason,
        metadata={
            "battle_id": trace.battle_id,
            "turn_index": trace.turn_index,
            "mode": trace.mode,
        },
    )
