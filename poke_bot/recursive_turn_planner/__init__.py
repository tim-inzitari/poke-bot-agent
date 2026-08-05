"""Lightweight Recursive Turn Planner (experimental).

Constructs typed, conditional full-turn programs from a shared state memory,
refines unresolved subgoals with shallow batched recursion, evaluates
candidates through learned latent transitions, and persists a conditional plan
across atomic actions with sparse repair.

This package is intentionally isolated from production PolicyAgent defaults and
online MCTS. It does not rewrite owner-contract authority.
"""

from .config import RTPConfig
from .dynamics import LatentTransitionDynamics
from .executor import PlanExecutor, PlanStepResult
from .legality import TypedLegalityVerifier
from .memory import PersistentTurnMemory
from .planner import PlanProposal, RecursiveTurnPlanner, TurnDecision
from .types import (
    NodeKind,
    ObservationPredicate,
    PlanNode,
    SubgoalKind,
    TurnProgram,
)

__all__ = [
    "LatentTransitionDynamics",
    "NodeKind",
    "ObservationPredicate",
    "PersistentTurnMemory",
    "PlanExecutor",
    "PlanNode",
    "PlanProposal",
    "PlanStepResult",
    "RTPConfig",
    "RecursiveTurnPlanner",
    "SubgoalKind",
    "TurnDecision",
    "TurnProgram",
    "TypedLegalityVerifier",
]
