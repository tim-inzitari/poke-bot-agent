"""Slowking neural distillation research pipeline.

Research-only. Creates no runtime, selector, checkpoint, training, serving,
freeze, registration, or submission authority for the failed Slowking
specialist. Enable stages explicitly via CLI flags / env; defaults are off.
"""

from .authority import RESEARCH_ONLY, RUNTIME_AUTHORITY, TRAINING_AUTHORITY

__all__ = [
    "RESEARCH_ONLY",
    "RUNTIME_AUTHORITY",
    "TRAINING_AUTHORITY",
]
