"""Pure-RL self-play leaf wiring (CPU sim + optional coalesced GPU leaves).

Official ``libcg`` battles stay per-process on CPU. Network eval can either:
  - load the policy in each sim worker (CPU-local, status quo bug for pure-RL), or
  - offload forwards to the persistent leaf servers that already coalesce batches
    across workers (same path round-robin ``_worker_play`` uses).

This module only decides *which* path to take; it does not touch overnight
launch scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SelfPlayLeafPlan:
    """How one self-play job should bind models / remote leaf backends."""

    use_leaf_for_us: bool
    use_leaf_for_them: bool
    load_us_local: bool
    load_them_local: bool
    same_checkpoint: bool
    mode: str  # "gpu-leaf-both" | "gpu-leaf-us-only" | "cpu-local"


def plan_self_play_leaf_wiring(
    *,
    us_checkpoint: str,
    them_checkpoint: str,
    leaf_channel_active: bool,
) -> SelfPlayLeafPlan:
    """Choose leaf vs local-CPU inference for a pure-RL self-play job.

    Leaf servers hold the *current* champion weights only. When the opponent
    checkpoint differs (recent-self pool), only our seat can use the remote
    leaf; the opponent stays CPU-local.
    """
    same = str(us_checkpoint) == str(them_checkpoint)
    if not leaf_channel_active:
        return SelfPlayLeafPlan(
            use_leaf_for_us=False,
            use_leaf_for_them=False,
            load_us_local=True,
            load_them_local=True,
            same_checkpoint=same,
            mode="cpu-local",
        )
    if same:
        return SelfPlayLeafPlan(
            use_leaf_for_us=True,
            use_leaf_for_them=True,
            load_us_local=False,
            load_them_local=False,
            same_checkpoint=True,
            mode="gpu-leaf-both",
        )
    return SelfPlayLeafPlan(
        use_leaf_for_us=True,
        use_leaf_for_them=False,
        load_us_local=False,
        load_them_local=True,
        same_checkpoint=False,
        mode="gpu-leaf-us-only",
    )


def expected_leaf_share(plan: SelfPlayLeafPlan) -> float:
    """Fraction of policy seats that will hit the GPU leaf path (0, 0.5, or 1)."""
    n = int(plan.use_leaf_for_us) + int(plan.use_leaf_for_them)
    return n / 2.0
