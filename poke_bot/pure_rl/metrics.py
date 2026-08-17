"""Iteration metrics schema for pure-RL loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class IterationMetrics:
    iteration: int = 0
    stage: str = "core"
    games: int = 0
    decisions: int = 0
    games_per_sec: float = 0.0
    decisions_per_sec: float = 0.0
    games_per_hour: float = 0.0
    parent_queue_wait_s: float = 0.0
    gpu0_util: float = 0.0
    gpu1_util: float = 0.0
    mean_return: float = 0.0
    mean_advantage: float = 0.0
    raw_advantage_mean: float = 0.0
    raw_advantage_std: float = 0.0
    raw_advantage_mean_abs: float = 0.0
    normalized_advantage_mean: float = 0.0
    normalized_advantage_std: float = 0.0
    awr_weight_mean: float = 0.0
    awr_weight_p50: float = 0.0
    awr_weight_p95: float = 0.0
    awr_weight_clip_frac: float = 0.0
    awr_effective_sample_size: float = 0.0
    awr_effective_sample_fraction: float = 0.0
    policy_selected_nll: float = 0.0
    policy_accuracy: float = 0.0
    #: Exact candidate-vs-parent argmax agreement on the iteration replay rows.
    policy_prev_agreement: float = 0.0
    self_distill_flag: bool = False
    heldout_wr: Optional[float] = None
    heldout_games: int = 0
    gate_passed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def metrics_to_dict(m: IterationMetrics) -> dict[str, Any]:
    d = asdict(m)
    return d
