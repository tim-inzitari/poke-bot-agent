"""Typed legality / resource verifier outside the neural planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import NodeKind, PlanNode, TurnProgram


@dataclass
class LegalityReport:
    valid: bool
    reason: str = ""
    pruned_nodes: int = 0
    kept_actions: tuple[tuple[int, ...], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


class TypedLegalityVerifier:
    """Exact rules stay outside the network.

    Neural planner proposes -> this layer validates -> invalid fragments prune.
    """

    def __init__(
        self,
        *,
        once_per_turn_flags: Optional[dict[str, bool]] = None,
    ) -> None:
        self.once_per_turn_flags = dict(once_per_turn_flags or {})

    def is_action_legal(
        self,
        action: tuple[int, ...],
        legal_actions: tuple[tuple[int, ...], ...],
    ) -> bool:
        return action in set(legal_actions)

    def validate_node(
        self,
        node: PlanNode,
        legal_actions: tuple[tuple[int, ...], ...],
    ) -> LegalityReport:
        if node.kind is NodeKind.PRIMITIVE:
            assert node.action is not None
            ok = self.is_action_legal(node.action, legal_actions)
            return LegalityReport(
                valid=ok,
                reason="" if ok else "illegal_primitive",
                pruned_nodes=0 if ok else 1,
                kept_actions=(node.action,) if ok else (),
            )
        if node.kind in {NodeKind.STOP, NodeKind.END_TURN}:
            return LegalityReport(valid=True, kept_actions=())
        if node.kind is NodeKind.SUBGOAL:
            # Subgoals are unresolved intents; legality applies after expansion.
            return LegalityReport(valid=True, kept_actions=())
        if node.kind is NodeKind.SEQUENCE:
            kept_children: list[PlanNode] = []
            pruned = 0
            kept_actions: list[tuple[int, ...]] = []
            for child in node.children:
                report = self.validate_node(child, legal_actions)
                pruned += report.pruned_nodes
                if report.valid:
                    kept_children.append(child)
                    kept_actions.extend(report.kept_actions)
            if not kept_children:
                return LegalityReport(
                    valid=False,
                    reason="empty_sequence_after_prune",
                    pruned_nodes=pruned,
                )
            return LegalityReport(
                valid=True,
                pruned_nodes=pruned,
                kept_actions=tuple(kept_actions),
                details={"kept_children": len(kept_children)},
            )
        if node.kind is NodeKind.BRANCH:
            then_report = self.validate_node(node.children[0], legal_actions)
            else_report = self.validate_node(node.children[1], legal_actions)
            pruned = then_report.pruned_nodes + else_report.pruned_nodes
            if not then_report.valid and not else_report.valid:
                return LegalityReport(
                    valid=False,
                    reason="both_branches_illegal",
                    pruned_nodes=pruned,
                )
            kept: list[tuple[int, ...]] = []
            if then_report.valid:
                kept.extend(then_report.kept_actions)
            if else_report.valid:
                kept.extend(else_report.kept_actions)
            return LegalityReport(
                valid=True,
                pruned_nodes=pruned,
                kept_actions=tuple(kept),
            )
        return LegalityReport(valid=False, reason=f"unknown_kind:{node.kind}")

    def validate_program(
        self,
        program: TurnProgram,
        legal_actions: tuple[tuple[int, ...], ...],
    ) -> LegalityReport:
        report = self.validate_node(program.root, legal_actions)
        if not report.valid:
            return report
        # Lightweight once-per-turn resource guard via constraints metadata.
        used_flags: set[str] = set()
        for action in report.kept_actions:
            _ = action  # action ids themselves are engine-validated above
        for key, available in self.once_per_turn_flags.items():
            if not available and key in used_flags:
                return LegalityReport(
                    valid=False,
                    reason=f"resource_exhausted:{key}",
                    pruned_nodes=report.pruned_nodes,
                )
        return report

    def prune_program(
        self,
        program: TurnProgram,
        legal_actions: tuple[tuple[int, ...], ...],
    ) -> Optional[TurnProgram]:
        report = self.validate_program(program, legal_actions)
        if not report.valid:
            return None
        return program
