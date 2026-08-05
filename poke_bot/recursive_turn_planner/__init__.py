"""Lightweight Recursive Turn Planner (experimental).

Constructs typed, conditional full-turn programs from a shared state memory,
refines unresolved subgoals with shallow batched recursion, evaluates
candidates through learned latent transitions, and persists a conditional plan
across atomic actions with sparse repair.

This package is intentionally isolated from production PolicyAgent defaults and
online MCTS. It does not rewrite owner-contract authority.

Sizing is profile-bound:
- ``global_transformer`` → d_model=256, dynamics_width=512
- ``pure_rl`` → d_model=96, dynamics_width=192
"""

from .agent_bridge import (
    RTPAgentBridge,
    RTPBridgeDiagnostics,
    resolve_rtp_config_for_model,
    turn_key_from_obs,
)
from .config import RTPConfig
from .dynamics import LatentTransitionDynamics, LookaheadBackedDynamics
from .executor import PlanExecutor, PlanStepResult
from .legality import TypedLegalityVerifier
from .memory import PersistentTurnMemory
from .planner import PlanProposal, RecursiveTurnPlanner, TurnDecision
from .profiles import (
    GLOBAL_TRANSFORMER,
    PURE_RL,
    UNIT_TEST,
    VERIFY_ABLATONS,
    get_profile,
    profile_inventory,
)
from .types import (
    NodeKind,
    ObservationPredicate,
    PlanNode,
    SubgoalKind,
    TurnProgram,
)

__all__ = [
    "GLOBAL_TRANSFORMER",
    "LatentTransitionDynamics",
    "LookaheadBackedDynamics",
    "NodeKind",
    "ObservationPredicate",
    "PURE_RL",
    "PersistentTurnMemory",
    "PlanExecutor",
    "PlanNode",
    "PlanProposal",
    "PlanStepResult",
    "RTPAgentBridge",
    "RTPBridgeDiagnostics",
    "RTPConfig",
    "RecursiveTurnPlanner",
    "SubgoalKind",
    "TurnDecision",
    "TurnProgram",
    "TypedLegalityVerifier",
    "UNIT_TEST",
    "VERIFY_ABLATONS",
    "get_profile",
    "profile_inventory",
    "resolve_rtp_config_for_model",
    "turn_key_from_obs",
]
