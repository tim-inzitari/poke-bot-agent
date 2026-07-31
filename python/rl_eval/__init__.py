"""Draw-aware eval metrics, promotion gates, and abort heuristics."""

from .aborts import AbortDecision, evaluate_aborts
from .metrics import (
    FieldReport,
    MatchupStats,
    stratified_bootstrap_interval,
    wilson_interval,
    wilson_lower,
)
from .promotion import (
    CheckpointIdentity,
    PromotionGateConfig,
    evaluate_candidate_gate,
    next_iteration,
)

__version__ = "0.1.1"

__all__ = [
    "AbortDecision",
    "CheckpointIdentity",
    "FieldReport",
    "MatchupStats",
    "PromotionGateConfig",
    "evaluate_aborts",
    "evaluate_candidate_gate",
    "next_iteration",
    "stratified_bootstrap_interval",
    "wilson_interval",
    "wilson_lower",
    "__version__",
]
