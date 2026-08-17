"""Pure-RL self-play leaf wiring (CPU sim + optional coalesced GPU leaves).

Official ``libcg`` battles stay on CPU (one process, or multi-handle via
``LibcgMultiEnv``). Network eval can either:
  - load the policy in each sim worker (CPU-local), or
  - offload forwards to the persistent leaf servers that already coalesce batches
    across workers (same path round-robin ``_worker_play`` uses).

:func:`plan_self_play_leaf_wiring` is consumed by ``remote_self_play_job`` and
the multi-env collect path. Coalesce defaults for the tiny pure-RL net live in
:func:`poke_bot.pure_rl.multi_env_self_play.pure_rl_leaf_coalesce_ms`.

Recursive Turn Planner (RTP) is a local-model swap-in: ``PolicyAgent`` only
initializes the RTP bridge when ``model is not None``, and
``RTPAgentBridge.encode`` runs parent ``TemporalCabtTransformer`` forwards.
When ``POKEBOT_USE_RECURSIVE_TURN_PLANNER`` is armed with a readable sidecar
(or explicit untrained allow), collect must force the CPU-local path even if
GPU leaves are up — leaf-only ``model=None`` workers never stamp RTP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelfPlayLeafPlan:
    """How one self-play job should bind models / remote leaf backends."""

    use_leaf_for_us: bool
    use_leaf_for_them: bool
    load_us_local: bool
    load_them_local: bool
    same_checkpoint: bool
    mode: str  # "gpu-leaf-both" | "gpu-leaf-us-only" | "cpu-local" | "rtp-cpu-local"


def rtp_requires_local_model() -> bool:
    """True when env arms RTP and a usable planner sidecar (or untrained allow).

    Mirrors ``PolicyAgent.__post_init__`` arming: without a local model the
    bridge never initializes, so GPU-leaf-only collect silently falls back to
    greedy ``history_policy``.
    """
    env_rtp = os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER", "").strip().lower()
    if env_rtp not in {"1", "true", "yes", "on"}:
        return False
    ckpt = os.environ.get("POKEBOT_RTP_CHECKPOINT", "").strip()
    allow_untrained = os.environ.get(
        "POKEBOT_RTP_ALLOW_UNTRAINED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    return bool((ckpt and Path(ckpt).is_file()) or allow_untrained)


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

    Armed RTP overrides leaf offload: the planner needs a local parent encoder
    (see ``experiments/recursive_turn_planner/SWAP_IN.md``).
    """
    same = str(us_checkpoint) == str(them_checkpoint)
    if rtp_requires_local_model():
        return SelfPlayLeafPlan(
            use_leaf_for_us=False,
            use_leaf_for_them=False,
            load_us_local=True,
            load_them_local=True,
            same_checkpoint=same,
            mode="rtp-cpu-local",
        )
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
