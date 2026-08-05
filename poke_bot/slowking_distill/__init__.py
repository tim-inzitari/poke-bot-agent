"""Slowking neural distillation research pipeline.

Research-only. Creates no runtime, selector, checkpoint, training, serving,
freeze, registration, or submission authority for the failed Slowking
specialist. Enable stages explicitly via CLI flags / env; defaults are off.
"""

from .authority import RESEARCH_ONLY, RUNTIME_AUTHORITY, TRAINING_AUTHORITY
from .config import (
    SlowkingDistillMode,
    SlowkingDistillRuntimeConfig,
    load_runtime_config_from_env,
)
from .policy_bridge import SlowkingDistillAgentBridge
from .promotion import PromotionRequest, evaluate_promotion
from .runtime import SlowkingDistillRuntime, build_runtime_from_env

__all__ = [
    "RESEARCH_ONLY",
    "RUNTIME_AUTHORITY",
    "TRAINING_AUTHORITY",
    "SlowkingDistillMode",
    "SlowkingDistillRuntimeConfig",
    "SlowkingDistillRuntime",
    "SlowkingDistillAgentBridge",
    "PromotionRequest",
    "build_runtime_from_env",
    "evaluate_promotion",
    "load_runtime_config_from_env",
]
