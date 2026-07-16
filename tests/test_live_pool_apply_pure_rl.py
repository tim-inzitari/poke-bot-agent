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
    g0, g1 = split_leaf_replicas(9)
    assert g0 + g1 == 9
    assert g1 >= g0


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
