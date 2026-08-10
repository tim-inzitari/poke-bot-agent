"""Apply ``live_pool_plan.json`` knobs to pure-RL hardware at iter boundaries."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from poke_bot.live_pool import (
    LivePoolPlan,
    _MAX_LEAF_GPU0,
    _MAX_LEAF_GPU1,
    _MAX_LEAF_SERVERS,
    _MAX_WORKERS,
    live_pool_enabled,
    max_local_workers_for_ram,
    read_live_pool_plan,
    should_apply_plan,
)
from poke_bot.pure_rl.hardware import FullHardwareProfile
from poke_bot.pure_rl.multi_env_self_play import process_worker_count


# Revision 70 / GOAL's final-format profile is an explicit fixed-capacity
# contract, not an adaptive scheduling preference.  Keep this narrow: generic
# profiles continue to use the no-swap RAM clamp below.
_EXACT_FIXED_LOCAL_WORKERS = 96
_EXACT_FIXED_WORKER_ENV_KEYS = (
    "PURE_RL_SIM_WORKERS",
    "PURE_RL_GAMES_IN_FLIGHT",
    "PURE_RL_REBALANCE_MIN_WORKERS",
    "PURE_RL_REBALANCE_MAX_WORKERS",
    "POKEBOT_LIVE_POOL_MAX_WORKERS",
)


def _exact_fixed_local_worker_target() -> Optional[int]:
    """Return the hard 96-worker target only for the fully pinned profile.

    A live-pool plan is normally advisory and may be RAM-capped.  The
    final-format 96/96 profile is different: its worker floor, target,
    ceiling, and games-in-flight are all owner-pinned.  Do not infer this
    from one setting alone; all five runtime settings must explicitly agree.
    """
    values: list[int] = []
    for key in _EXACT_FIXED_WORKER_ENV_KEYS:
        raw = os.environ.get(key)
        if raw is None or not str(raw).strip():
            return None
        try:
            values.append(int(raw))
        except ValueError:
            return None
    if all(value == _EXACT_FIXED_LOCAL_WORKERS for value in values):
        return _EXACT_FIXED_LOCAL_WORKERS
    return None


def split_leaf_replicas(total: int, *, prefer_gpu1: bool = True) -> tuple[int, int]:
    """Split total leaf servers across GPU0 (3080 Ti) / GPU1 (Blackwell).

    3080 Ti is VRAM-capped (≤``_MAX_LEAF_GPU0``); all headroom goes to Blackwell.
    """
    total = max(1, int(total))
    if total == 1:
        return (0, 1) if prefer_gpu1 else (1, 0)
    # Prefer pinning GPU0 at its conservative cap when the total is large
    # enough; keep Ti a minority on small smoke totals.
    gpu0 = min(_MAX_LEAF_GPU0, total - 1)
    if prefer_gpu1 and gpu0 * 2 > total:
        gpu0 = max(1, min(_MAX_LEAF_GPU0, total // 3))
    gpu1 = min(_MAX_LEAF_GPU1, total - gpu0)
    gpu0 = total - gpu1
    if gpu0 > _MAX_LEAF_GPU0:
        gpu0 = _MAX_LEAF_GPU0
        gpu1 = min(_MAX_LEAF_GPU1, total - gpu0)
    return int(gpu0), int(gpu1)


def apply_live_pool_plan(
    *,
    hw: FullHardwareProfile,
    last_seq: int,
    multi_env_per_worker: int,
    visible_gpu_count: int,
) -> tuple[FullHardwareProfile, int, int, Optional[LivePoolPlan], bool]:
    """Return (hw, proc_workers, new_last_seq, plan_or_None, leaf_topology_changed).

    No-op when live pool disabled or no newer plan.
    """
    if not live_pool_enabled():
        proc = process_worker_count(hw.sim_workers, multi_env_per_worker)
        return hw, proc, last_seq, None, False

    plan = read_live_pool_plan()
    if plan is None or not should_apply_plan(plan, last_seq):
        proc = process_worker_count(hw.sim_workers, multi_env_per_worker)
        return hw, proc, last_seq, None, False

    # Use module hard-max headroom (defaults stay moderate in hardware.py /
    # launch env; watcher grows plans toward these ceilings).
    plan = plan.clamped(
        max_workers=_MAX_WORKERS, max_leaf_servers=_MAX_LEAF_SERVERS
    )
    want_w = int(plan.workers) if plan.workers is not None else int(hw.sim_workers)
    want_w = max(1, want_w)
    fixed_worker_target = _exact_fixed_local_worker_target()
    if fixed_worker_target is not None:
        # Never let a stale watcher plan or its generic RAM model reduce the
        # explicitly hard-stuck final-format collection profile.
        want_w = fixed_worker_target

    if plan.leaf_gpu0 is not None and plan.leaf_gpu1 is not None:
        gpu0, gpu1 = int(plan.leaf_gpu0), int(plan.leaf_gpu1)
        if gpu0 + gpu1 < 1:
            gpu0, gpu1 = split_leaf_replicas(
                int(hw.leaf_replicas_total), prefer_gpu1=True
            )
    else:
        want_leaf_total = (
            int(plan.leaf_servers)
            if plan.leaf_servers is not None
            else int(hw.leaf_replicas_total)
        )
        gpu0, gpu1 = split_leaf_replicas(want_leaf_total, prefer_gpu1=True)

    # Enforce per-GPU caps even if an older plan pre-dates the 3080 clamp.
    gpu0 = max(0, min(int(gpu0), _MAX_LEAF_GPU0))
    gpu1 = max(1 if visible_gpu_count >= 2 else 0, min(int(gpu1), _MAX_LEAF_GPU1))

    if visible_gpu_count < 2:
        total = max(1, gpu0 + gpu1)
        gpu0, gpu1 = total, 0

    if fixed_worker_target is None:
        # No-swap RAM ceiling: a plan (hand-written or watcher-emitted) must
        # never raise local workers above what physically fits without swap,
        # even if it is inside the module's headroom-only _MAX_WORKERS sanity
        # clamp above. Leaf server RSS is accounted for since it grows/shrinks
        # with this same plan.
        ram_cap = max_local_workers_for_ram(leaf_count=gpu0 + gpu1, min_workers=1)
        if want_w > ram_cap:
            want_w = max(1, ram_cap)

    leaf_changed = gpu0 != hw.leaf_gpu0_replicas or gpu1 != hw.leaf_gpu1_replicas
    workers_changed = want_w != hw.sim_workers or want_w != hw.games_in_flight
    new_hw = replace(
        hw,
        sim_workers=want_w,
        games_in_flight=max(
            want_w, hw.games_in_flight if not workers_changed else want_w
        ),
        leaf_gpu0_replicas=gpu0,
        leaf_gpu1_replicas=gpu1,
    )
    # games_in_flight tracks workers for pure-RL saturation
    new_hw = replace(new_hw, games_in_flight=max(want_w, new_hw.games_in_flight))
    proc = process_worker_count(new_hw.sim_workers, multi_env_per_worker)
    return new_hw, proc, int(plan.seq), plan, leaf_changed
