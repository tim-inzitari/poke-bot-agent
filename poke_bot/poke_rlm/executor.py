"""Stateful plan execution, resource ledger, and bounded repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .budget import TurnComputeBudget
from .legal_action import LegalAction, resolve_selector
from .plan_ir import (
    ConditionalNode,
    PlanNode,
    PrimitiveAction,
    SequenceNode,
    StopNode,
    SubgoalNode,
    TurnPlan,
)
from .reasons import ReasonCode


@dataclass
class ResourceLedger:
    supporter_available: bool = True
    energy_attached_this_turn: bool = False
    retreated_this_turn: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def apply_flags(self, flags: tuple[str, ...]) -> ReasonCode:
        for flag in flags:
            if flag == "supporter" and not self.supporter_available:
                return ReasonCode.RESOURCE_LEDGER_VIOLATION
            if flag == "supporter":
                self.supporter_available = False
            if flag == "energy_attach":
                self.energy_attached_this_turn = True
            if flag == "retreat":
                if self.retreated_this_turn:
                    return ReasonCode.RESOURCE_LEDGER_VIOLATION
                self.retreated_this_turn = True
        return ReasonCode.OK


@dataclass
class ExecutionStep:
    action: Optional[tuple[int, ...]]
    done: bool
    reason: ReasonCode
    repaired: bool = False
    remaining: Optional[PlanNode] = None


RepairFn = Callable[[TurnPlan, tuple[LegalAction, ...]], TurnPlan]


class PlanExecutor:
    def __init__(
        self,
        *,
        budget: Optional[TurnComputeBudget] = None,
        repair_fn: Optional[RepairFn] = None,
        repair_budget: int = 1,
    ) -> None:
        self.budget = budget or TurnComputeBudget()
        self.repair_fn = repair_fn
        self.repair_budget = int(repair_budget)
        self.plan: Optional[TurnPlan] = None
        self.cursor: Optional[PlanNode] = None
        self.ledger = ResourceLedger()
        self.repairs_used = 0

    def load(self, plan: TurnPlan) -> None:
        self.plan = plan
        self.cursor = plan.root
        self.ledger = ResourceLedger()
        self.repairs_used = 0

    def clear(self) -> None:
        self.plan = None
        self.cursor = None
        self.repairs_used = 0

    def _select_branch(
        self, node: ConditionalNode, observations: dict[str, Any]
    ) -> tuple[PlanNode, ReasonCode]:
        if not node.predicate.visibility_ok(observations):
            return node.if_false, ReasonCode.BRANCH_PREDICATE_HIDDEN
        if node.predicate.matches(observations):
            return node.if_true, ReasonCode.OK
        return node.if_false, ReasonCode.OK

    def apply_observation(self, observations: dict[str, Any]) -> ReasonCode:
        if self.cursor is None:
            return ReasonCode.OK
        node = self.cursor
        if isinstance(node, SequenceNode) and node.children:
            first = node.children[0]
            if isinstance(first, ConditionalNode):
                chosen, reason = self._select_branch(first, observations)
                self.cursor = SequenceNode(children=(chosen,) + node.children[1:])
                return reason
        if isinstance(node, ConditionalNode):
            chosen, reason = self._select_branch(node, observations)
            self.cursor = chosen
            return reason
        return ReasonCode.OK

    def _advance(
        self, node: PlanNode, legal: tuple[LegalAction, ...]
    ) -> ExecutionStep:
        if isinstance(node, StopNode):
            return ExecutionStep(None, True, ReasonCode.OK, remaining=node)
        if isinstance(node, SubgoalNode):
            # Unresolved subgoal should already be refined; use fallback.
            return self._advance(node.fallback, legal)
        if isinstance(node, PrimitiveAction):
            resolved, reason = resolve_selector(node.selector, legal)
            if resolved is None:
                if node.on_unresolvable is not None:
                    return self._advance(node.on_unresolvable, legal)
                return ExecutionStep(None, True, reason)
            if reason is ReasonCode.STALE_OPTION_INDEX:
                # Still legal by path; allow but report.
                pass
            elif reason is not ReasonCode.OK:
                if node.on_unresolvable is not None:
                    return self._advance(node.on_unresolvable, legal)
                return ExecutionStep(None, True, reason)
            ledger_reason = self.ledger.apply_flags(node.expected_delta.once_per_turn_flags)
            if ledger_reason is not ReasonCode.OK:
                if node.on_unresolvable is not None:
                    return self._advance(node.on_unresolvable, legal)
                return ExecutionStep(None, True, ledger_reason)
            return ExecutionStep(
                resolved.option_index_path,
                False,
                ReasonCode.OK,
                remaining=StopNode(),
            )
        if isinstance(node, ConditionalNode):
            # Prefer then-branch until observation applied.
            return self._advance(node.if_true, legal)
        if isinstance(node, SequenceNode):
            if not node.children:
                return ExecutionStep(None, True, ReasonCode.INVALID_PLAN_STATIC)
            head = node.children[0]
            step = self._advance(head, legal)
            if step.done and step.action is None:
                if len(node.children) == 1:
                    return step
                return self._advance(
                    SequenceNode(children=node.children[1:]), legal
                )
            if step.remaining is None or isinstance(step.remaining, StopNode):
                rem = node.children[1:]
            else:
                rem = (step.remaining,) + node.children[1:]
            remaining: Optional[PlanNode]
            if not rem:
                remaining = StopNode()
            else:
                remaining = SequenceNode(children=rem)
            return ExecutionStep(
                step.action,
                step.action is None,
                step.reason,
                repaired=step.repaired,
                remaining=remaining,
            )
        return ExecutionStep(None, True, ReasonCode.INVALID_PLAN_STATIC)

    def next_action(
        self,
        legal: tuple[LegalAction, ...],
        *,
        observations: Optional[dict[str, Any]] = None,
    ) -> ExecutionStep:
        if self.plan is None or self.cursor is None:
            return ExecutionStep(None, True, ReasonCode.FALLBACK_POLICY)
        if observations:
            self.apply_observation(observations)
        assert self.cursor is not None
        step = self._advance(self.cursor, legal)
        if step.action is None and step.reason not in {ReasonCode.OK}:
            if (
                self.repairs_used < self.repair_budget
                and self.repair_fn is not None
                and self.plan is not None
            ):
                self.repairs_used += 1
                self.budget.repairs += 1
                repaired = self.repair_fn(self.plan, legal)
                self.load(repaired)
                step2 = self._advance(self.cursor, legal)  # type: ignore[arg-type]
                step2.repaired = True
                self.cursor = step2.remaining
                return step2
            return step
        self.cursor = step.remaining
        if step.action is not None and isinstance(self.cursor, StopNode):
            step.done = True
        return step
