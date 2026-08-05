"""Typed turn-program AST for the Recursive Turn Planner.

Programs are compact structured objects (card/effect/target/resource/predicate
ids), never natural-language reasoning traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeKind(str, Enum):
    PRIMITIVE = "primitive"
    SEQUENCE = "sequence"
    SUBGOAL = "subgoal"
    BRANCH = "branch"
    STOP = "stop"
    END_TURN = "end_turn"


class SubgoalKind(str, Enum):
    ESTABLISH_ATTACKER = "establish_attacker"
    FIND_RESOURCE = "find_resource"
    MAXIMIZE_DRAW = "maximize_draw"
    PRESERVE_ESCAPE = "preserve_escape"
    REACH_DAMAGE_THRESHOLD = "reach_damage_threshold"
    SETUP_NEXT_TURN = "setup_next_turn"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ObservationPredicate:
    """Typed observation predicate for conditional branches."""

    name: str
    expected: Any = True
    constraints: tuple[str, ...] = ()

    def matches(self, observations: dict[str, Any]) -> bool:
        if self.name not in observations:
            return False
        return observations[self.name] == self.expected


@dataclass(frozen=True)
class PlanNode:
    """One node in a typed conditional turn program."""

    kind: NodeKind
    action: Optional[tuple[int, ...]] = None
    subgoal: Optional[SubgoalKind] = None
    subgoal_label: str = ""
    constraints: tuple[str, ...] = ()
    children: tuple["PlanNode", ...] = ()
    predicate: Optional[ObservationPredicate] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is NodeKind.PRIMITIVE and self.action is None:
            raise ValueError("primitive nodes require an action")
        if self.kind is NodeKind.SUBGOAL and self.subgoal is None:
            raise ValueError("subgoal nodes require a subgoal kind")
        if self.kind is NodeKind.BRANCH and len(self.children) != 2:
            raise ValueError("branch nodes require exactly two children")
        if self.kind is NodeKind.BRANCH and self.predicate is None:
            raise ValueError("branch nodes require a predicate")
        if self.kind is NodeKind.SEQUENCE and not self.children:
            raise ValueError("sequence nodes require children")

    @property
    def is_terminal(self) -> bool:
        return self.kind in {NodeKind.STOP, NodeKind.END_TURN}

    def first_action(self) -> Optional[tuple[int, ...]]:
        """Return the first primitive action reachable by left-to-right walk."""
        if self.kind is NodeKind.PRIMITIVE:
            return self.action
        if self.kind is NodeKind.SEQUENCE:
            for child in self.children:
                found = child.first_action()
                if found is not None:
                    return found
            return None
        if self.kind is NodeKind.BRANCH:
            # Prefer then-branch for speculative first-step extraction.
            return self.children[0].first_action()
        return None

    def flatten_primitives(self) -> tuple[tuple[int, ...], ...]:
        if self.kind is NodeKind.PRIMITIVE and self.action is not None:
            return (self.action,)
        if self.kind in {NodeKind.SEQUENCE, NodeKind.BRANCH}:
            out: list[tuple[int, ...]] = []
            for child in self.children:
                out.extend(child.flatten_primitives())
            return tuple(out)
        return ()

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def replace_children(self, children: tuple["PlanNode", ...]) -> "PlanNode":
        return PlanNode(
            kind=self.kind,
            action=self.action,
            subgoal=self.subgoal,
            subgoal_label=self.subgoal_label,
            constraints=self.constraints,
            children=children,
            predicate=self.predicate,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class TurnProgram:
    """A complete conditional turn program with score metadata."""

    root: PlanNode
    score: float = 0.0
    uncertainty: float = 0.0
    compute_cost: float = 0.0
    plan_id: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.root.flatten_primitives())

    def first_action(self) -> Optional[tuple[int, ...]]:
        return self.root.first_action()
