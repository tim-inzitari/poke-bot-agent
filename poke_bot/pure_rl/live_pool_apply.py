"""Apply ``live_pool_plan.json`` knobs to pure-RL hardware at iter boundaries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from poke_bot.live_pool import (
    LivePoolPlan,
    live_pool_enabled,
    read_live_pool_plan,
    should_apply_plan,
)
from poke_bot.pure_rl.hardware import FullHardwareProfile
from poke_bot.pure_rl.multi_env_self_play import process_worker_count


def split_leaf_replicas(total: int, *, prefer_gpu1: bool = True) -> tuple[int, int]:
    """Split total leaf servers across GPU1 (Blackwell) / GPU0 (3080 Ti).

    Default ratio matches ``full_hardware_profile`` (~2:1 gpu1:gpu0).
    """
    total = max(1, int(total))
    if total == 1:
        return (0, 1) if prefer_gpu1 else (1, 0)
    # ~2/3 on GPU1, ~1/3 on GPU0
    gpu1 = max(1, (total * 2) // 3)
    gpu0 = max(1, total - gpu1)
    if gpu0 + gpu1 > total:
        gpu1 = total - gpu0
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

    plan = plan.clamped(max_workers=128, max_leaf_servers=16)
    want_w = int(plan.workers) if plan.workers is not None else int(hw.sim_workers)
    want_leaf_total = (
        int(plan.leaf_servers)
        if plan.leaf_servers is not None
        else int(hw.leaf_replicas_total)
    )
    want_w = max(1, want_w)
    gpu0, gpu1 = split_leaf_replicas(want_leaf_total, prefer_gpu1=True)
    if visible_gpu_count < 2 and not hw.allow_single_gpu:
        # Keep both sides if profile requires; otherwise collapse to gpu0.
        pass
    if visible_gpu_count < 2:
        gpu0, gpu1 = max(1, want_leaf_total), 0

    leaf_changed = gpu0 != hw.leaf_gpu0_replicas or gpu1 != hw.leaf_gpu1_replicas
    workers_changed = want_w != hw.sim_workers or want_w != hw.games_in_flight
    new_hw = replace(
        hw,
        sim_workers=want_w,
        games_in_flight=max(want_w, hw.games_in_flight if not workers_changed else want_w),
        leaf_gpu0_replicas=gpu0,
        leaf_gpu1_replicas=gpu1,
    )
    # games_in_flight tracks workers for pure-RL saturation
    new_hw = replace(new_hw, games_in_flight=max(want_w, new_hw.games_in_flight))
    proc = process_worker_count(new_hw.sim_workers, multi_env_per_worker)
    return new_hw, proc, int(plan.seq), plan, leaf_changed
