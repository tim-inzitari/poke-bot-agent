"""Tests for live pool plan parsing/clamping and WorkerPool resize semantics."""

from __future__ import annotations

import json
from pathlib import Path

from poke_bot.live_pool import (
    LivePoolPlan,
    read_live_pool_plan,
    should_apply_plan,
    write_live_pool_plan,
)
from poke_bot.worker_pool import WorkerPool


def _double(x: int) -> int:
    return x * 2


def test_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "live_pool_plan.json"
    written = write_live_pool_plan(
        seq=3,
        workers=48,
        leaf_servers=6,
        promotion_workers=12,
        reason="unit-test",
        path=path,
    )
    assert written.seq == 3
    assert written.workers == 48
    assert written.leaf_servers == 6
    assert path.is_file()
    loaded = read_live_pool_plan(path)
    assert loaded is not None
    assert loaded.seq == 3
    assert loaded.workers == 48
    assert loaded.leaf_servers == 6
    assert loaded.promotion_workers == 12
    assert loaded.reason == "unit-test"


def test_clamp_leaf_servers_independent_of_workers() -> None:
    """Leaf farms may exceed CPU worker counts (multi-env + shared leaf RPC)."""
    from poke_bot.live_pool import _MAX_LEAF_SERVERS

    plan = LivePoolPlan(seq=1, workers=4, leaf_servers=99).clamped()
    assert plan.workers == 4
    assert plan.leaf_servers == _MAX_LEAF_SERVERS


def test_clamp_hard_caps() -> None:
    plan = LivePoolPlan(
        seq=1, workers=10_000, leaf_servers=100, promotion_workers=999
    ).clamped(max_workers=64, max_leaf_servers=8)
    assert plan.workers == 64
    assert plan.leaf_servers == 8
    assert plan.promotion_workers == 64


def test_full_hw_leaf_gpu0_hard_capped() -> None:
    """3080 Ti leaf count cannot exceed the conservative 12GB cap."""
    from poke_bot.live_pool import _MAX_LEAF_GPU0

    plan = LivePoolPlan(
        seq=2, workers=96, leaf_servers=42, leaf_gpu0=18, leaf_gpu1=24
    ).clamped()
    assert plan.leaf_gpu0 == _MAX_LEAF_GPU0
    assert plan.leaf_gpu0 <= 12
    assert plan.leaf_gpu1 == 24
    assert plan.leaf_servers == plan.leaf_gpu0 + plan.leaf_gpu1


def test_max_ceilings_exceed_steady_defaults(monkeypatch) -> None:
    """Headroom on workers + Blackwell only; 3080 Ti has no leaf headroom."""
    from poke_bot.live_pool import (
        _MAX_LEAF_GPU0,
        _MAX_LEAF_GPU1,
        _MAX_LEAF_SERVERS,
        _MAX_WORKERS,
    )
    from poke_bot.pure_rl.hardware import full_hardware_profile
    from poke_bot.pure_rl.mid_iter_scheduler import MidIterScheduler

    for key in (
        "PURE_RL_REBALANCE_MAX_WORKERS",
        "PURE_RL_SIM_WORKERS",
        "PURE_RL_LEAF_GPU0_REPLICAS",
        "PURE_RL_LEAF_GPU1_REPLICAS",
        "POKEBOT_SIM_WORKERS",
        "POKEBOT_LEAF_GPU0_REPLICAS",
        "POKEBOT_LEAF_GPU1_REPLICAS",
    ):
        monkeypatch.delenv(key, raising=False)

    hw = full_hardware_profile()
    sched = MidIterScheduler.from_env(baseline_workers=40)
    assert _MAX_WORKERS >= 160
    assert _MAX_LEAF_GPU0 <= 12
    assert _MAX_LEAF_GPU1 >= 48
    assert _MAX_LEAF_SERVERS >= _MAX_LEAF_GPU0 + _MAX_LEAF_GPU1
    assert sched.max_workers >= 160
    assert sched.target_workers < sched.max_workers
    assert hw.sim_workers < _MAX_WORKERS
    assert hw.leaf_gpu0_replicas <= _MAX_LEAF_GPU0
    assert hw.leaf_gpu1_replicas < _MAX_LEAF_GPU1  # BW has headroom
    # Fat plan: GPU0 crushed to cap; Blackwell keeps headroom.
    fat = LivePoolPlan(
        seq=9, workers=150, leaf_servers=72, leaf_gpu0=28, leaf_gpu1=44
    ).clamped()
    assert fat.workers == 150
    assert fat.leaf_gpu0 == _MAX_LEAF_GPU0
    assert fat.leaf_gpu1 == 44
    assert fat.leaf_servers == fat.leaf_gpu0 + fat.leaf_gpu1


def test_should_apply_plan_seq_and_apply_field() -> None:
    plan = LivePoolPlan(seq=5, workers=16, apply="next_iter")
    assert should_apply_plan(plan, last_seq=4)
    assert not should_apply_plan(plan, last_seq=5)
    assert not should_apply_plan(plan, last_seq=6)
    mid = LivePoolPlan(seq=7, apply="mid_wave")
    assert not should_apply_plan(mid, last_seq=0)


def test_corrupt_plan_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    assert read_live_pool_plan(path) is None


def test_successive_worker_pools_different_sizes_without_remote() -> None:
    """Wave-boundary resize: new pools with different sizes are valid."""

    with WorkerPool(num_workers=2) as pool:
        assert sorted(pool.imap_unordered(_double, [1, 2, 3])) == [2, 4, 6]
    with WorkerPool(num_workers=4) as pool:
        assert sorted(pool.imap_unordered(_double, [1, 2, 3, 4])) == [2, 4, 6, 8]


def test_write_plan_omits_null_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    write_live_pool_plan(seq=1, workers=32, path=path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["workers"] == 32
    assert raw["leaf_servers"] is None
    assert raw["apply"] == "next_iter"
