"""Unit tests for pure-RL live-pool apply helpers (no torch / libcg)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, rel: str):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Load without importing poke_bot.pure_rl package (avoids torch).
live_pool = _load("poke_bot.live_pool", "poke_bot/live_pool.py")
# hardware imports poke_bot.config — ok without torch
hardware = _load("poke_bot.pure_rl.hardware", "poke_bot/pure_rl/hardware.py")
# multi_env_self_play has no torch
multi = _load(
    "poke_bot.pure_rl.multi_env_self_play",
    "poke_bot/pure_rl/multi_env_self_play.py",
)
sys.modules["poke_bot.pure_rl.multi_env_self_play"] = multi
sys.modules["poke_bot.pure_rl.hardware"] = hardware
sys.modules["poke_bot.live_pool"] = live_pool
apply_mod = _load(
    "poke_bot.pure_rl.live_pool_apply",
    "poke_bot/pure_rl/live_pool_apply.py",
)

FullHardwareProfile = hardware.FullHardwareProfile
LivePoolPlan = live_pool.LivePoolPlan
write_live_pool_plan = live_pool.write_live_pool_plan
apply_live_pool_plan = apply_mod.apply_live_pool_plan
split_leaf_replicas = apply_mod.split_leaf_replicas


def test_split_leaf_replicas_ratio() -> None:
    from poke_bot.live_pool import _MAX_LEAF_GPU0

    g0, g1 = split_leaf_replicas(9)
    assert g0 + g1 == 9
    assert g1 >= g0
    # Headroom on Blackwell only; 3080 Ti pinned at conservative cap.
    g0, g1 = split_leaf_replicas(42)
    assert g0 == _MAX_LEAF_GPU0
    assert g0 <= 12
    assert g1 == 42 - g0
    assert g1 > g0


def test_leaf_cuda_devices_stripes_both_gpus() -> None:
    from poke_bot.pure_rl.hardware import (
        pick_leaf_server_index,
        sticky_leaf_server_index,
    )

    hw = FullHardwareProfile(
        sim_workers=96,
        games_in_flight=96,
        train_cuda_device=1,
        leaf_gpu1_replicas=30,
        leaf_gpu0_replicas=12,
        torch_threads=8,
    )
    devices = hw.leaf_cuda_devices()
    assert len(devices) == 42
    assert devices.count(0) == 12
    assert devices.count(1) == 30
    # Contiguous GPU1-then-GPU0 must not return; GPU0 is spread across the map.
    assert set(devices[:24]) == {0, 1}
    g0_idx = [i for i, d in enumerate(devices) if d == 0]
    assert g0_idx[0] == 0
    assert g0_idx[-1] >= 30  # not bunched only in the first 2*n0 slots
    # Sticky bias oversubscribes GPU0 clients vs leaf fraction (12/42≈0.29).
    binds = [sticky_leaf_server_index(s, devices, gpu0_client_frac=0.38) for s in range(96)]
    g0_clients = sum(1 for b in binds if devices[b] == 0)
    assert 30 <= g0_clients <= 45
    # Least-queue prefers an empty GPU0 server over a busy GPU1.

    class _Q:
        def __init__(self, n: int) -> None:
            self._n = n

        def qsize(self) -> int:
            return self._n

    qs = [_Q(3) for _ in devices]
    qs[g0_idx[1]] = _Q(0)
    assert pick_leaf_server_index(req_qs=qs, devices=devices) == g0_idx[1]


def test_apply_honors_explicit_leaf_gpu_split(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_LIVE_POOL", "1")
    monkeypatch.setattr(
        apply_mod,
        "read_live_pool_plan",
        lambda path=None: LivePoolPlan(
            seq=4,
            workers=96,
            leaf_servers=42,
            leaf_gpu0=18,  # over 3080 cap — must clamp to ≤12
            leaf_gpu1=24,
            reason="gpu0 feed",
        ).clamped(),
    )
    # Isolate leaf-split behavior from the real host's /proc/meminfo: the
    # no-swap RAM ceiling (see test_apply_ram_cap_clamps_workers_below_plan_
    # request below) is a separate concern with its own dedicated coverage.
    monkeypatch.setattr(apply_mod, "max_local_workers_for_ram", lambda **_: 1_000_000)
    hw = FullHardwareProfile(
        sim_workers=72,
        games_in_flight=72,
        train_cuda_device=1,
        leaf_gpu1_replicas=24,
        leaf_gpu0_replicas=6,
        torch_threads=8,
    )
    new_hw, procs, seq, plan, leaf_changed = apply_live_pool_plan(
        hw=hw,
        last_seq=0,
        multi_env_per_worker=4,
        visible_gpu_count=2,
    )
    assert plan is not None
    assert seq == 4
    assert new_hw.sim_workers == 96
    assert new_hw.leaf_gpu0_replicas <= 12
    assert new_hw.leaf_gpu1_replicas == 24
    assert leaf_changed
    assert procs == 96 // 4


def test_apply_ram_cap_clamps_workers_below_plan_request(monkeypatch) -> None:
    """No-swap RAM ceiling wins even when a plan requests more workers.

    A hand-written or watcher-emitted plan must never push local workers
    above what physically fits without swap, even though it is inside the
    module's headroom-only ``_MAX_WORKERS`` sanity clamp (160).
    """
    monkeypatch.setenv("POKEBOT_LIVE_POOL", "1")
    monkeypatch.setattr(
        apply_mod,
        "read_live_pool_plan",
        lambda path=None: LivePoolPlan(
            seq=5,
            workers=96,
            leaf_servers=12,
            reason="watcher requested growth",
        ).clamped(),
    )
    monkeypatch.setattr(apply_mod, "max_local_workers_for_ram", lambda **_: 20)
    hw = FullHardwareProfile(
        sim_workers=32,
        games_in_flight=32,
        train_cuda_device=1,
        leaf_gpu1_replicas=8,
        leaf_gpu0_replicas=4,
        torch_threads=8,
    )
    new_hw, procs, seq, plan, leaf_changed = apply_live_pool_plan(
        hw=hw,
        last_seq=0,
        multi_env_per_worker=4,
        visible_gpu_count=2,
    )
    assert plan is not None
    assert seq == 5
    # RAM ceiling (20) beats the requested/clamped worker count (96).
    assert new_hw.sim_workers == 20
    assert procs == 20 // 4


def test_apply_live_pool_plan(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_LIVE_POOL", "1")
    monkeypatch.setattr(
        apply_mod,
        "read_live_pool_plan",
        lambda path=None: LivePoolPlan(
            seq=3, workers=48, leaf_servers=12, reason="test bump"
        ).clamped(),
    )
    hw = FullHardwareProfile(
        sim_workers=32,
        games_in_flight=32,
        train_cuda_device=1,
        leaf_gpu1_replicas=6,
        leaf_gpu0_replicas=3,
        torch_threads=8,
    )
    new_hw, procs, seq, plan, leaf_changed = apply_live_pool_plan(
        hw=hw,
        last_seq=0,
        multi_env_per_worker=4,
        visible_gpu_count=2,
    )
    assert plan is not None
    assert seq == 3
    assert new_hw.sim_workers == 48
    assert new_hw.leaf_replicas_total == 12
    assert leaf_changed
    assert procs == 48 // 4


def test_apply_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_LIVE_POOL", "0")
    hw = FullHardwareProfile(
        sim_workers=32,
        games_in_flight=32,
        train_cuda_device=1,
        leaf_gpu1_replicas=6,
        leaf_gpu0_replicas=3,
        torch_threads=8,
    )
    new_hw, procs, seq, plan, changed = apply_live_pool_plan(
        hw=hw,
        last_seq=0,
        multi_env_per_worker=1,
        visible_gpu_count=2,
    )
    assert plan is None
    assert new_hw.sim_workers == 32
    assert not changed
    assert procs == 32
