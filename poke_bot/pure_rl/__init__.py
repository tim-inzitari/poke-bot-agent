"""Single-model pure-RL track: AWR train, compact shards, win-rate gates."""

from __future__ import annotations

from poke_bot.pure_rl.aborts import AbortDecision, evaluate_aborts
from poke_bot.pure_rl.curriculum import CurriculumStage, stage_for_iteration, stage_to_dict
from poke_bot.pure_rl.eval_public import HeldOutGateResult, aggregate_heldout_wr
from poke_bot.pure_rl.hardware import FullHardwareProfile, full_hardware_profile
from poke_bot.pure_rl.metrics import IterationMetrics, metrics_to_dict
from poke_bot.pure_rl.shards import CompactShardWriter, compact_decision_from_step

__all__ = [
    "AbortDecision",
    "CompactShardWriter",
    "CurriculumStage",
    "FullHardwareProfile",
    "HeldOutGateResult",
    "IterationMetrics",
    "aggregate_heldout_wr",
    "compact_decision_from_step",
    "evaluate_aborts",
    "full_hardware_profile",
    "metrics_to_dict",
    "stage_for_iteration",
    "stage_to_dict",
]
