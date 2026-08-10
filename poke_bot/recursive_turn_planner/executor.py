"""Conditional plan execution with sparse repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import RTPConfig
from .legality import TypedLegalityVerifier
from .memory import PersistentTurnMemory
from .types import NodeKind, PlanNode, TurnProgram


@dataclass
class PlanStepResult:
    action: Optional[tuple[int, ...]]
    done: bool
    repaired: bool = False
    reason: str = ""
    observations_applied: dict[str, Any] = field(default_factory=dict)
    remaining_root: Optional[PlanNode] = None


class PlanExecutor:
    """Persist and execute a conditional plan across atomic decisions.

    At turn start the planner produces a conditional program. Subsequent atomic
    steps follow matching branches and only run a small repair pass when the
    observation was not covered or the plan became illegal.
    """

    def __init__(
        self,
        config: Optional[RTPConfig] = None,
        *,
        legality: Optional[TypedLegalityVerifier] = None,
        repair_fn: Optional[
            Callable[[PersistentTurnMemory, TurnProgram], TurnProgram]
        ] = None,
    ) -> None:
        self.config = config or RTPConfig()
        self.legality = legality or TypedLegalityVerifier()
        self.repair_fn = repair_fn
        self.active_program: Optional[TurnProgram] = None
        self.cursor: Optional[PlanNode] = None
        self.repairs_used = 0
        self.steps_executed = 0

    def load(self, program: TurnProgram) -> None:
        self.active_program = program
        self.cursor = program.root
        self.repairs_used = 0
        self.steps_executed = 0

    def clear(self) -> None:
        self.active_program = None
        self.cursor = None
        self.repairs_used = 0
        self.steps_executed = 0

    def _advance_sequence(
        self,
        node: PlanNode,
    ) -> tuple[Optional[tuple[int, ...]], Optional[PlanNode], bool]:
        """Return (action, remaining_node, done)."""
        if node.kind is NodeKind.PRIMITIVE:
            return node.action, PlanNode(kind=NodeKind.STOP), False
        if node.kind in {NodeKind.STOP, NodeKind.END_TURN}:
            return None, node, True
        if node.kind is NodeKind.SUBGOAL:
            # Unresolved subgoal should have been refined; treat as stop.
            return None, PlanNode(kind=NodeKind.STOP), True
        if node.kind is NodeKind.BRANCH:
            # Without a fresh observation, prefer then-branch.
            return self._advance_sequence(node.children[0])
        if node.kind is NodeKind.SEQUENCE:
            if not node.children:
                return None, PlanNode(kind=NodeKind.STOP), True
            head = node.children[0]
            action, head_remaining, done = self._advance_sequence(head)
            if done and action is None:
                # Head was terminal; continue into the tail.
                if len(node.children) == 1:
                    return None, PlanNode(kind=NodeKind.STOP), True
                return self._advance_sequence(
                    PlanNode(kind=NodeKind.SEQUENCE, children=node.children[1:])
                )
            if head_remaining is None or head_remaining.is_terminal:
                remaining_children = node.children[1:]
            else:
                remaining_children = (head_remaining,) + node.children[1:]
            if not remaining_children:
                return action, PlanNode(kind=NodeKind.STOP), action is None
            return (
                action,
                PlanNode(kind=NodeKind.SEQUENCE, children=remaining_children),
                False,
            )
        return None, PlanNode(kind=NodeKind.STOP), True

    def _select_branch(
        self,
        node: PlanNode,
        observations: dict[str, Any],
    ) -> PlanNode:
        assert node.kind is NodeKind.BRANCH
        assert node.predicate is not None
        if node.predicate.matches(observations):
            return node.children[0]
        return node.children[1]

    def apply_observation(self, observations: dict[str, Any]) -> None:
        """Resolve a leading BRANCH against a real observation."""
        if self.cursor is None:
            return
        node = self.cursor
        if node.kind is NodeKind.SEQUENCE and node.children:
            first = node.children[0]
            if first.kind is NodeKind.BRANCH:
                chosen = self._select_branch(first, observations)
                self.cursor = PlanNode(
                    kind=NodeKind.SEQUENCE,
                    children=(chosen,) + node.children[1:],
                )
                return
        if node.kind is NodeKind.BRANCH:
            self.cursor = self._select_branch(node, observations)

    def next_action(
        self,
        memory: PersistentTurnMemory,
        *,
        observations: Optional[dict[str, Any]] = None,
    ) -> PlanStepResult:
        if self.active_program is None or self.cursor is None:
            return PlanStepResult(
                action=None,
                done=True,
                reason="no_active_program",
            )

        if observations:
            self.apply_observation(observations)
            memory.update_observation(observations=observations)

        # Validate remaining plan against current legal actions.
        remaining_program = TurnProgram(
            root=self.cursor,
            score=self.active_program.score,
            plan_id=self.active_program.plan_id,
        )
        report = self.legality.validate_program(
            remaining_program, memory.legal_actions
        )
        repaired = False
        if not report.valid:
            if (
                self.repairs_used < self.config.repair_budget
                and self.repair_fn is not None
            ):
                repaired_program = self.repair_fn(memory, self.active_program)
                self.repairs_used += 1
                self.active_program = repaired_program
                self.cursor = repaired_program.root
                repaired = True
                remaining_program = repaired_program
                report = self.legality.validate_program(
                    remaining_program, memory.legal_actions
                )
            if not report.valid:
                return PlanStepResult(
                    action=None,
                    done=True,
                    repaired=repaired,
                    reason=f"plan_invalid:{report.reason}",
                    remaining_root=self.cursor,
                )

        action, remaining, done = self._advance_sequence(self.cursor)
        self.cursor = remaining
        if action is not None:
            self.steps_executed += 1
            if not self.legality.is_action_legal(action, memory.legal_actions):
                # Immediate repair / abort if extracted action is illegal.
                if (
                    self.repairs_used < self.config.repair_budget
                    and self.repair_fn is not None
                ):
                    repaired_program = self.repair_fn(memory, self.active_program)
                    self.repairs_used += 1
                    self.active_program = repaired_program
                    self.cursor = repaired_program.root
                    repaired_step = self.next_action(memory, observations=None)
                    return PlanStepResult(
                        action=repaired_step.action,
                        done=repaired_step.done,
                        repaired=True,
                        reason=repaired_step.reason,
                        observations_applied=repaired_step.observations_applied,
                        remaining_root=repaired_step.remaining_root,
                    )
                return PlanStepResult(
                    action=None,
                    done=True,
                    # An illegal extraction can follow sequence validation
                    # that kept a later legal child while leaving the cursor
                    # unchanged.  With no callback invocation this is an
                    # abort, not a repair; callers use the distinction to
                    # enter their explicit, telemetry-visible replan path.
                    repaired=repaired,
                    reason="extracted_action_illegal",
                )
        return PlanStepResult(
            action=action,
            done=done or action is None,
            repaired=repaired,
            reason="ok" if action is not None else "plan_exhausted",
            observations_applied=dict(observations or {}),
            remaining_root=self.cursor,
        )
