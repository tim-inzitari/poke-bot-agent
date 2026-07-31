"""Fail-closed promote aborts for self-distillation / dead advantages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class AbortDecision:
    abort: bool
    reason: str
    self_distill_flag: bool
    mean_advantage: float
    advantage_signal: float


def evaluate_aborts(
    *,
    mean_advantages: Sequence[float],
    policy_prev_agreements: Sequence[float],
    advantage_mean_abs: Optional[Sequence[float]] = None,
    k: int = 3,
    adv_eps: float = 1e-3,
    agreement_hi: float = 0.95,
) -> AbortDecision:
    if k < 1:
        raise ValueError("k must be >= 1")
    source = advantage_mean_abs if advantage_mean_abs is not None else mean_advantages
    signals = [abs(float(x)) for x in source][-k:]
    ags = [float(x) for x in policy_prev_agreements][-k:]
    if len(signals) < k or len(ags) < k:
        last_signal = signals[-1] if signals else 0.0
        return AbortDecision(
            abort=False,
            reason="insufficient_history",
            self_distill_flag=False,
            mean_advantage=last_signal,
            advantage_signal=last_signal,
        )
    mean_signal = sum(signals) / len(signals)
    mean_ag = sum(ags) / len(ags)
    dead_adv = all(signal <= adv_eps for signal in signals)
    self_distill = dead_adv and mean_ag >= agreement_hi
    if self_distill:
        return AbortDecision(
            abort=True,
            reason="self_distill",
            self_distill_flag=True,
            mean_advantage=mean_signal,
            advantage_signal=mean_signal,
        )
    if dead_adv:
        return AbortDecision(
            abort=True,
            reason="zero_advantage",
            self_distill_flag=False,
            mean_advantage=mean_signal,
            advantage_signal=mean_signal,
        )
    return AbortDecision(
        abort=False,
        reason="ok",
        self_distill_flag=False,
        mean_advantage=mean_signal,
        advantage_signal=mean_signal,
    )
