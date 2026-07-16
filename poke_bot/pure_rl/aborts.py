"""Fail-closed promote aborts for self-distillation / dead advantages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AbortDecision:
    abort: bool
    reason: str
    self_distill_flag: bool
    mean_advantage: float


def evaluate_aborts(
    *,
    mean_advantages: Sequence[float],
    policy_prev_agreements: Sequence[float],
    k: int = 3,
    adv_eps: float = 1e-3,
    agreement_hi: float = 0.95,
) -> AbortDecision:
    """Abort if the last ``k`` iters look like self-distill or ~zero advantage."""
    if k < 1:
        raise ValueError("k must be >= 1")
    advs = [float(x) for x in mean_advantages][-k:]
    ags = [float(x) for x in policy_prev_agreements][-k:]
    if len(advs) < k or len(ags) < k:
        last_adv = advs[-1] if advs else 0.0
        return AbortDecision(
            abort=False,
            reason="insufficient_history",
            self_distill_flag=False,
            mean_advantage=last_adv,
        )
    mean_adv = sum(advs) / len(advs)
    mean_ag = sum(ags) / len(ags)
    dead_adv = all(abs(a) <= adv_eps for a in advs)
    self_distill = dead_adv and mean_ag >= agreement_hi
    if self_distill:
        return AbortDecision(
            abort=True,
            reason="self_distill",
            self_distill_flag=True,
            mean_advantage=mean_adv,
        )
    if dead_adv:
        return AbortDecision(
            abort=True,
            reason="zero_advantage",
            self_distill_flag=False,
            mean_advantage=mean_adv,
        )
    return AbortDecision(
        abort=False,
        reason="ok",
        self_distill_flag=False,
        mean_advantage=mean_adv,
    )
