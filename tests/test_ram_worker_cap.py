"""Unit tests for the no-swap RAM worker ceiling (mocked mem signals).

Covers :func:`poke_bot.live_pool.max_local_workers_for_ram` directly (pure
math, deterministic via explicit kwargs) plus the mid-iter scheduler's
low-RAM shrink path and the ``LIVE_POOL_MAX_WORKERS`` / ``POKEBOT_`` env
alias. No torch import required.
"""

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
sys.modules["poke_bot.live_pool"] = live_pool

max_local_workers_for_ram = live_pool.max_local_workers_for_ram
sample_mem_gb = live_pool.sample_mem_gb


def test_ram_cap_matches_measured_box_math() -> None:
    """~124 GiB box, 60 leaves, 1.3 GiB/worker: cap comfortably below 96/160."""
    cap = max_local_workers_for_ram(
        mem_total_gb=131.0,
        mem_available_gb=30.0,
        per_worker_rss_gb=1.3,
        leaf_count=60,
        leaf_rss_gb=0.33,
        reserve_gb=6.0,
        free_ram_floor_gb=8.0,
    )
    # budget = 131 - 6 - 60*0.33 - 8 = 97.2; floor(97.2/1.3) = 74
    assert cap == 74
    assert cap < 96
    assert cap < 160


def test_ram_cap_scales_down_with_more_leaves() -> None:
    fewer_leaves = max_local_workers_for_ram(
        mem_total_gb=131.0, per_worker_rss_gb=1.3, leaf_count=12
    )
    more_leaves = max_local_workers_for_ram(
        mem_total_gb=131.0, per_worker_rss_gb=1.3, leaf_count=60
    )
    assert more_leaves < fewer_leaves


def test_ram_cap_never_below_min_workers() -> None:
    cap = max_local_workers_for_ram(
        mem_total_gb=4.0,  # tiny box — budget goes negative
        per_worker_rss_gb=1.3,
        leaf_count=60,
        min_workers=1,
    )
    assert cap == 1


def test_ram_cap_uses_available_when_total_unknown() -> None:
    cap_total_unset = max_local_workers_for_ram(
        mem_total_gb=0.0,
        mem_available_gb=40.0,
        per_worker_rss_gb=1.3,
        leaf_count=0,
        reserve_gb=0.0,
        free_ram_floor_gb=8.0,
    )
    # (40 - 0 - 8) / 1.3 = 24.6 -> 24
    assert cap_total_unset == 24


def test_ram_cap_no_signal_never_clamps() -> None:
    """No /proc/meminfo signal at all -> caller must not be clamped."""
    cap = max_local_workers_for_ram(mem_total_gb=0.0, mem_available_gb=0.0)
    assert cap >= 1_000_000


def test_sample_mem_gb_reads_real_proc_meminfo() -> None:
    total, avail = sample_mem_gb()
    # Real Linux host in CI/dev: sane positive readings.
    assert total > 0.0
    assert avail >= 0.0
    assert avail <= total


def test_live_pool_max_workers_env_alias_pokebot_prefix(monkeypatch) -> None:
    """live_pool._env_max must honor POKEBOT_-prefixed launch env too."""
    monkeypatch.delenv("LIVE_POOL_MAX_WORKERS", raising=False)
    monkeypatch.setenv("POKEBOT_LIVE_POOL_MAX_WORKERS", "48")
    assert live_pool._env_max("LIVE_POOL_MAX_WORKERS", 160) == 48


def test_live_pool_max_workers_env_alias_unprefixed_wins(monkeypatch) -> None:
    monkeypatch.setenv("LIVE_POOL_MAX_WORKERS", "72")
    monkeypatch.setenv("POKEBOT_LIVE_POOL_MAX_WORKERS", "48")
    assert live_pool._env_max("LIVE_POOL_MAX_WORKERS", 160) == 72


def test_apply_ram_cap_clamps_plan_via_helper_reference(monkeypatch) -> None:
    """apply_live_pool_plan resolves the RAM cap through a mockable module
    attribute (poke_bot.pure_rl.live_pool_apply.max_local_workers_for_ram),
    matching the read_live_pool_plan monkeypatch pattern used elsewhere.
    """
    hardware = _load("poke_bot.pure_rl.hardware", "poke_bot/pure_rl/hardware.py")
    multi = _load(
        "poke_bot.pure_rl.multi_env_self_play",
        "poke_bot/pure_rl/multi_env_self_play.py",
    )
    sys.modules["poke_bot.pure_rl.multi_env_self_play"] = multi
    sys.modules["poke_bot.pure_rl.hardware"] = hardware
    apply_mod = _load(
        "poke_bot.pure_rl.live_pool_apply", "poke_bot/pure_rl/live_pool_apply.py"
    )

    monkeypatch.setenv("POKEBOT_LIVE_POOL", "1")
    for key in (
        "PURE_RL_SIM_WORKERS",
        "PURE_RL_GAMES_IN_FLIGHT",
        "PURE_RL_REBALANCE_MIN_WORKERS",
        "PURE_RL_REBALANCE_MAX_WORKERS",
        "POKEBOT_LIVE_POOL_MAX_WORKERS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        apply_mod,
        "read_live_pool_plan",
        lambda path=None: live_pool.LivePoolPlan(
            seq=1, workers=160, leaf_servers=12, reason="stale watcher plan"
        ).clamped(),
    )
    monkeypatch.setattr(apply_mod, "max_local_workers_for_ram", lambda **_: 48)

    hw = hardware.FullHardwareProfile(
        sim_workers=32,
        games_in_flight=32,
        train_cuda_device=1,
        leaf_gpu1_replicas=8,
        leaf_gpu0_replicas=4,
        torch_threads=8,
    )
    new_hw, procs, seq, plan, _leaf_changed = apply_mod.apply_live_pool_plan(
        hw=hw, last_seq=0, multi_env_per_worker=1, visible_gpu_count=2
    )
    assert plan is not None
    assert seq == 1
    assert new_hw.sim_workers == 48
    assert procs == 48


def test_apply_exact_fixed_96_profile_ignores_live_plan_and_ram_clamp(
    monkeypatch,
) -> None:
    """The owner-pinned 96/96 profile must never silently downscale."""
    hardware = _load("poke_bot.pure_rl.hardware", "poke_bot/pure_rl/hardware.py")
    multi = _load(
        "poke_bot.pure_rl.multi_env_self_play",
        "poke_bot/pure_rl/multi_env_self_play.py",
    )
    sys.modules["poke_bot.pure_rl.multi_env_self_play"] = multi
    sys.modules["poke_bot.pure_rl.hardware"] = hardware
    apply_mod = _load(
        "poke_bot.pure_rl.live_pool_apply", "poke_bot/pure_rl/live_pool_apply.py"
    )

    monkeypatch.setenv("POKEBOT_LIVE_POOL", "1")
    for key in (
        "PURE_RL_SIM_WORKERS",
        "PURE_RL_GAMES_IN_FLIGHT",
        "PURE_RL_REBALANCE_MIN_WORKERS",
        "PURE_RL_REBALANCE_MAX_WORKERS",
    ):
        monkeypatch.setenv(key, "96")
    monkeypatch.delenv("POKEBOT_LIVE_POOL_MAX_WORKERS", raising=False)
    assert apply_mod._exact_fixed_local_worker_target() is None
    monkeypatch.setenv("POKEBOT_LIVE_POOL_MAX_WORKERS", "96")
    monkeypatch.setattr(
        apply_mod,
        "read_live_pool_plan",
        lambda path=None: live_pool.LivePoolPlan(
            seq=1, workers=74, leaf_servers=12, reason="generic ram backoff"
        ).clamped(),
    )

    def ram_cap_must_not_run(**_kwargs):
        raise AssertionError("fixed 96 profile must bypass generic RAM clamping")

    monkeypatch.setattr(apply_mod, "max_local_workers_for_ram", ram_cap_must_not_run)

    hw = hardware.FullHardwareProfile(
        sim_workers=96,
        games_in_flight=96,
        train_cuda_device=1,
        leaf_gpu1_replicas=8,
        leaf_gpu0_replicas=4,
        torch_threads=8,
    )
    new_hw, procs, seq, plan, _leaf_changed = apply_mod.apply_live_pool_plan(
        hw=hw, last_seq=0, multi_env_per_worker=1, visible_gpu_count=2
    )
    assert plan is not None
    assert seq == 1
    assert new_hw.sim_workers == 96
    assert new_hw.games_in_flight == 96
    assert procs == 96
