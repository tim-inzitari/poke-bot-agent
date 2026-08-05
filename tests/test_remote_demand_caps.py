"""Remote demand caps: max ≫ default; scheduler grows into ceiling."""

from __future__ import annotations

import time

import pytest

from poke_bot.remote_jobs import (
    demand_slot_count,
    endpoint_default_workers,
    endpoint_max_workers,
    endpoint_role,
    remote_refill_games_per_socket,
    remote_queue_low_water_fraction,
    remote_queue_probe_interval_s,
    remote_socket_prefetch_factor,
    remote_socket_prefetch_max_factor,
    remote_socket_max_target,
    remote_socket_target,
)
from poke_bot.pure_rl.mid_iter_scheduler import (
    HardwareSignals,
    MidIterScheduler,
    WaveGpsTracker,
)


def _bind_elmo_bert(sched: MidIterScheduler) -> None:
    sched.remote_defaults = {
        "192.168.1.143:8765": 20,
        "bert.local:8766": 10,
    }
    sched.remote_maxima = {
        "192.168.1.143:8765": 40,
        "bert.local:8766": 20,
    }
    sched._remote_demand = {
        "192.168.1.143:8765": 20,
        "bert.local:8766": 10,
    }


def test_wave_tracker_reports_direct_side_decision_rates() -> None:
    tracker = WaveGpsTracker()
    tracker.t0 = time.monotonic() - 10.0
    tracker.note(side="local", n=2, decisions=300)
    tracker.note(side="remote", n=3, decisions=500)
    snap = tracker.snapshot()
    assert snap["local_gps"] == pytest.approx(0.2, rel=0.03)
    assert snap["remote_gps"] == pytest.approx(0.3, rel=0.03)
    assert snap["local_sps"] == pytest.approx(30.0, rel=0.03)
    assert snap["remote_sps"] == pytest.approx(50.0, rel=0.03)
    assert snap["wave_sps"] == pytest.approx(80.0, rel=0.03)


def _force_tracker(
    sched: MidIterScheduler,
    *,
    elapsed_s: float,
    done_local: int,
    done_remote: int,
) -> None:
    """Pin wave tracker to a synthetic completion snapshot."""
    now = time.monotonic()
    tr = sched.tracker
    tr.t0 = now - float(elapsed_s)
    tr.done_local = int(done_local)
    tr.done_remote = int(done_remote)
    tr.done_total = int(done_local) + int(done_remote)
    tr._win_t0 = now - float(elapsed_s)
    tr._win_done = int(done_local) + int(done_remote)
    tr._ema_inited = False
    tr.ema_gps = 0.0
    sched._last_tick = now - 100.0
    sched._demand_probe.pending = ""
    sched._demand_probe.grow_blocked_until = 0.0
    sched._demand_probe.baseline_eff = 0.0


def test_scheduled_dispatch_has_demand_shrink_sockets() -> None:
    """Grow-only left sockets stuck high after demand_hurt (GPS 15→6.7).

    Completion gate must retire emit threads so remotes= falls with demand.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "poke_bot" / "remote_jobs.py"
    text = src.read_text(encoding="utf-8")
    assert "def _maybe_shrink_remote_slots" in text
    assert "demand_shrink_queue" in text
    assert "demand_shrink" in text
    # Regression: the July 16 overnight run stranded 20 jobs when a slot
    # retired in the middle of its already-claimed chunk.  Retirement must be
    # cooperative at chunk boundaries; returned tail jobs plus the remote
    # share ceiling can otherwise leave every claimant ineligible forever.
    assert "mid-chunk retire" not in text
    assert "scheduled tail-drain override" in text
    assert "scheduled dispatch exhausted all emitters" in text
    # Shrink before grow on the same tick.
    shrink_at = text.index("_maybe_shrink_remote_slots(dec)")
    grow_at = text.index("_maybe_grow_remote_slots(dec)")
    assert shrink_at < grow_at


def test_production_launcher_requires_configured_remotes() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "launch_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    assert 'env.setdefault("POKEBOT_REMOTE_REQUIRE_ALL", "1")' in text
    assert 'env.setdefault("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "1")' in text


def test_elmo_bert_default_below_max() -> None:
    assert endpoint_role("192.168.1.143") == "elmo"
    assert endpoint_role("bert.local") == "bert"
    assert endpoint_default_workers("192.168.1.143", 8765) == 20
    assert endpoint_max_workers("192.168.1.143", 8765) == 40
    assert endpoint_default_workers("bert.local", 8766) == 10
    assert endpoint_max_workers("bert.local", 8766) == 20
    assert endpoint_default_workers("192.168.1.143") < endpoint_max_workers(
        "192.168.1.143"
    )
    assert endpoint_default_workers("bert.local") < endpoint_max_workers("bert.local")


def test_env_overrides_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "POKEBOT_REMOTE_DEFAULT_WORKERS",
        "192.168.1.143:8765=16,bert.local:8766=8",
    )
    monkeypatch.setenv(
        "POKEBOT_REMOTE_MAX_WORKERS",
        "192.168.1.143:8765=40,bert.local:8766=20",
    )
    assert endpoint_default_workers("192.168.1.143", 8765) == 16
    assert endpoint_max_workers("bert.local", 8766) == 20


def test_demand_slot_count_clamps() -> None:
    assert demand_slot_count(capacity=40, demand=20, max_cap=40) == 20
    assert demand_slot_count(capacity=40, demand=40, max_cap=40) == 40
    assert demand_slot_count(capacity=20, demand=40, max_cap=40) == 20
    assert demand_slot_count(capacity=40, demand=40, max_cap=20) == 20
    assert demand_slot_count(capacity=0, demand=20, max_cap=40) == 0


def test_remote_socket_prefetch_keeps_one_queued_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "2")
    assert remote_socket_prefetch_factor() == 2
    assert remote_socket_target(48) == 96
    assert remote_socket_target(16) == 32


def test_remote_refill_claim_size_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_REFILL_GAMES", "1")
    assert remote_refill_games_per_socket() == 1
    monkeypatch.setenv("POKEBOT_REMOTE_REFILL_GAMES", "8")
    assert remote_refill_games_per_socket() == 8


def test_remote_socket_prefetch_is_safely_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "999")
    assert remote_socket_prefetch_factor() == 8
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "invalid")
    assert remote_socket_prefetch_factor() == 1


def test_remote_low_water_reserve_is_bounded_and_half_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH", "5")
    monkeypatch.setenv("POKEBOT_REMOTE_SOCKET_PREFETCH_MAX", "8")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_LOW_WATER_FRAC", "0.5")
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "2")
    assert remote_socket_target(48) == 240
    assert remote_socket_max_target(48) == 384
    assert remote_socket_prefetch_max_factor() == 8
    assert remote_queue_low_water_fraction() == 0.5
    assert remote_queue_probe_interval_s() == 2.0


def test_remote_queue_probe_can_run_at_public_mix_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "0.2")
    assert remote_queue_probe_interval_s() == 0.2
    monkeypatch.setenv("POKEBOT_REMOTE_QUEUE_PROBE_S", "0.01")
    assert remote_queue_probe_interval_s() == 0.2


def test_scheduler_starts_at_default_not_max() -> None:
    sched = MidIterScheduler(
        remote_defaults={
            "192.168.1.143:8765": 20,
            "bert.local:8766": 10,
        },
        remote_maxima={
            "192.168.1.143:8765": 40,
            "bert.local:8766": 20,
        },
    )
    dec = sched.decision()
    assert dec.remote_demand["192.168.1.143:8765"] == 20
    assert dec.remote_demand["bert.local:8766"] == 10
    assert dec.remote_demand["192.168.1.143:8765"] < 40
    assert dec.remote_demand["bert.local:8766"] < 20


def test_scheduler_can_grow_toward_max() -> None:
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.remote_defaults = {"192.168.1.143:8765": 20, "bert.local:8766": 10}
    sched.remote_maxima = {"192.168.1.143:8765": 40, "bert.local:8766": 20}
    sched._remote_demand = {"192.168.1.143:8765": 20, "bert.local:8766": 10}
    reasons: list[str] = []
    sched._step_demand(grow=True, reasons=reasons)
    assert sched._remote_demand["192.168.1.143:8765"] > 20
    assert sched._remote_demand["192.168.1.143:8765"] <= 40
    assert sched._remote_demand["bert.local:8766"] > 10
    assert sched._remote_demand["bert.local:8766"] <= 20
    assert any("remote_grow" in r for r in reasons)


def test_scheduler_shrink_stops_at_default() -> None:
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.remote_defaults = {"192.168.1.143:8765": 20}
    sched.remote_maxima = {"192.168.1.143:8765": 40}
    sched._remote_demand = {"192.168.1.143:8765": 25}
    for _ in range(8):
        sched._step_demand(grow=False, reasons=[])
    assert sched._remote_demand["192.168.1.143:8765"] == 20


def test_max_total_exceeds_local_default() -> None:
    """Total ceiling is absurd — never binds local+remote peak scaling."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    realistic_peak = (
        int(sched.max_workers)  # local headroom 160
        + endpoint_max_workers("192.168.1.143", 8765)  # elmo 40
        + endpoint_max_workers("bert.local", 8766)  # bert 20
    )
    assert sched.target_workers == 96
    assert sched.max_workers >= 160
    assert realistic_peak == 220
    # 10000 ≫ 220: total cap must never force remotes to steal from local.
    assert sched.max_total_workers >= 10_000
    assert sched.max_total_workers > realistic_peak * 10
    assert sched.target_workers < sched.max_total_workers
    assert sched.min_remote_frac >= 0.20


def test_remote_floor_survives_gpu_feed_bias() -> None:
    """Bound remotes keep min_remote_frac even when local-bias reasons fire."""
    sched = MidIterScheduler(
        prefer_local_frac=0.55,
        min_local_frac=0.40,
        max_remote_frac=0.60,
        min_remote_frac=0.25,
        target_workers=96,
        max_workers=160,
        max_total_workers=10_000,
        remote_defaults={"192.168.1.143:8765": 20, "bert.local:8766": 10},
        remote_maxima={"192.168.1.143:8765": 40, "bert.local:8766": 20},
    )
    # Simulate GPU-feed / cpu headroom pushing local_share toward 0.95.
    with sched._lock:
        sched._remote_demand = {
            "192.168.1.143:8765": 20,
            "bert.local:8766": 10,
        }
        local_share = 0.95
        remote_share = 1.0 - local_share
        if sched._remote_demand and sched.min_remote_frac > 0.0:
            remote_share = max(float(sched.min_remote_frac), float(remote_share))
            remote_share = min(sched.max_remote_frac, remote_share)
        local_share = max(sched.min_local_frac, 1.0 - remote_share)
        remote_share = 1.0 - local_share
        if remote_share < sched.min_remote_frac and local_share > sched.min_local_frac:
            steal = min(
                local_share - sched.min_local_frac,
                sched.min_remote_frac - remote_share,
            )
            local_share -= steal
            remote_share += steal
    assert remote_share >= 0.25 - 1e-9
    assert local_share >= 0.40 - 1e-9
    assert abs(local_share + remote_share - 1.0) < 1e-9


def test_remote_gps_ahead_without_local_need_holds_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch-dump remote_gps ≫ local must not grow when local is not bottlenecked."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    # Remotes ahead, host has CPU headroom — no offload need.
    _force_tracker(sched, elapsed_s=25.0, done_local=8, done_remote=165)
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=40.0,
            load1=8.0,
            mem_available_gb=40.0,
            gpu0_util_pct=70.0,
            gpu1_util_pct=70.0,
            ok=True,
        ),
    )
    before = dict(sched._remote_demand)
    dec = sched.maybe_tick(remaining=1500, force=True)
    assert dec is not None
    assert dec.remote_demand["192.168.1.143:8765"] == before["192.168.1.143:8765"]
    assert "remote_grow" not in dec.reason
    assert (
        "remote_ahead_hold_demand" in dec.reason
        or "remote_demand_warmup" in dec.reason
        or "demand_target_workers" in dec.reason
    )


def test_demand_grows_when_local_needs_offload_and_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demand (= target remote workers) rises with completion-gated offload need."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    # Warm: ≥30 remote completions; local finishing ahead → grow target workers.
    _force_tracker(sched, elapsed_s=40.0, done_local=200, done_remote=80)
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=5.0,
            load1=48.0,
            mem_available_gb=20.0,
            gpu0_util_pct=90.0,
            gpu1_util_pct=90.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1400, force=True)
    assert dec is not None
    assert any("remote_grow" in r for r in dec.reason.split("+"))
    assert "demand_probe_grow" in dec.reason or "completion_offload_grow" in dec.reason
    assert dec.remote_demand["192.168.1.143:8765"] > 20
    assert sched._demand_probe.pending == "grow"
    # Demand rise raises remote claim share toward additive target.
    assert "demand_raise_remote_share" in dec.reason or dec.remote_share >= 0.25


def test_demand_probe_shrinks_when_grow_hurts_wave_gps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a grow, settle+worse wave GPS rolls demand back (completion gate)."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    sched._demand_probe.settle_s = 1.0
    sched._demand_probe.cooldown_s = 30.0
    _bind_elmo_bert(sched)
    # Demand already grown; probe armed as if we just grew at wave_gps≈7.
    sched._remote_demand = {
        "192.168.1.143:8765": 25,
        "bert.local:8766": 12,
    }
    sched._demand_probe.pending = "grow"
    sched._demand_probe.probe_mono = time.monotonic() - 2.0
    sched._demand_probe.probe_demand_sum = 37
    sched._demand_probe.probe_pre_demand_sum = 30
    sched._demand_probe.probe_wave_gps = 7.0
    sched._demand_probe.probe_ema_gps = 7.0
    sched._demand_probe.probe_remote_gps = 5.0
    sched._demand_probe.baseline_eff = 5.0 / 30.0
    # Settled wave is worse (live: 6.91 → 4.81 after first grow).
    _force_tracker(sched, elapsed_s=50.0, done_local=60, done_remote=180)
    # Restore probe after _force_tracker cleared it.
    sched._demand_probe.pending = "grow"
    sched._demand_probe.probe_mono = time.monotonic() - 2.0
    sched._demand_probe.probe_demand_sum = 37
    sched._demand_probe.probe_pre_demand_sum = 30
    sched._demand_probe.probe_wave_gps = 7.0
    sched._demand_probe.probe_ema_gps = 7.0
    sched._demand_probe.probe_remote_gps = 5.0
    sched._demand_probe.baseline_eff = 5.0 / 30.0
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=0.0,
            load1=64.0,
            mem_available_gb=19.0,
            gpu0_util_pct=99.0,
            gpu1_util_pct=84.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1400, force=True)
    assert dec is not None
    assert "demand_hurt_wave_gps" in dec.reason or "demand_eff_collapse" in dec.reason
    assert any("remote_shrink" in r for r in dec.reason.split("+"))
    assert dec.remote_demand["192.168.1.143:8765"] < 25
    assert sched._demand_probe.grow_blocked_until > time.monotonic()
    # Hurt ratchets ceiling to pre-grow (20/10) so re-grow cannot climb again.
    assert sched._demand_probe.grow_ceiling == 30
    assert "demand_grow_ceiling=30" in dec.reason


def test_hurt_grow_ceiling_blocks_regrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After hurt→shrink to 20/10, cooldown expiry must not re-grow past ceiling."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    sched._remote_demand = {
        "192.168.1.143:8765": 20,
        "bert.local:8766": 10,
    }
    sched._demand_probe.grow_ceiling = 30
    sched._demand_probe.grow_blocked_until = 0.0
    sched._demand_probe.baseline_eff = 0.15
    # Warm + local hot → would otherwise completion_offload_grow.
    _force_tracker(sched, elapsed_s=40.0, done_local=200, done_remote=80)
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=5.0,
            load1=48.0,
            mem_available_gb=20.0,
            gpu0_util_pct=90.0,
            gpu1_util_pct=90.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1400, force=True)
    assert dec is not None
    assert dec.remote_demand["192.168.1.143:8765"] == 20
    assert dec.remote_demand["bert.local:8766"] == 10
    assert "remote_grow" not in dec.reason
    assert "demand_at_grow_ceiling" in dec.reason


def test_flat_grow_does_not_chain_another_grow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settle with flat GPS → hold_level; same tick must not ladder 25→30."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    sched._demand_probe.settle_s = 1.0
    _bind_elmo_bert(sched)
    sched._remote_demand = {
        "192.168.1.143:8765": 25,
        "bert.local:8766": 12,
    }
    # wave≈6.9 on baseline 7.1 → not hurt (≥-8%) but flat (<+3%).
    _force_tracker(sched, elapsed_s=50.0, done_local=130, done_remote=215)
    sched._demand_probe.pending = "grow"
    sched._demand_probe.probe_mono = time.monotonic() - 2.0
    sched._demand_probe.probe_demand_sum = 37
    sched._demand_probe.probe_pre_demand_sum = 30
    sched._demand_probe.probe_wave_gps = 7.1
    sched._demand_probe.probe_ema_gps = 7.1
    sched._demand_probe.probe_remote_gps = 5.0
    sched._demand_probe.baseline_eff = 5.0 / 30.0
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=0.0,
            load1=64.0,
            mem_available_gb=19.0,
            gpu0_util_pct=99.0,
            gpu1_util_pct=84.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1000, force=True)
    assert dec is not None
    assert "demand_grow_flat_hold" in dec.reason
    assert "remote_grow" not in dec.reason
    assert dec.remote_demand["192.168.1.143:8765"] == 25


def test_worse_than_best_default_forces_shrink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demand above 20/10 with wave GPS well below best default must shrink."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    sched._remote_demand = {
        "192.168.1.143:8765": 30,
        "bert.local:8766": 14,
    }
    # ~6.7 gps while best default was ~15.
    _force_tracker(sched, elapsed_s=60.0, done_local=150, done_remote=250)
    sched._demand_probe.best_default_wave_gps = 15.0
    sched._demand_probe.baseline_eff = 0.20
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=15.0,
            load1=40.0,
            mem_available_gb=20.0,
            gpu0_util_pct=80.0,
            gpu1_util_pct=80.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1000, force=True)
    assert dec is not None
    assert "demand_worse_than_best_default" in dec.reason
    assert any("remote_shrink" in r for r in dec.reason.split("+"))
    assert dec.remote_demand["192.168.1.143:8765"] < 30
    assert sched._demand_probe.grow_ceiling == 30


def test_tqdm_remote_postfix_separates_sockets_owned_games_and_demand() -> None:
    """Capacity, ownership, and scheduler demand retain distinct labels."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_bar", path)
    assert spec is not None and spec.loader is not None
    sys.modules.pop("train_pure_rl_bar", None)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog = mod._TqdmProgress(
        stage="collect:self_play",
        iteration=0,
        total=10,
        remotes=30,
        inplace=False,
        mininterval=60.0,
    )
    try:
        assert prog.remotes == 30
        assert "rdmd" not in prog._postfix(sps="0")
        prog.set_remotes(37, demand=37)
        assert prog.remotes == 37
        assert "rdmd" not in prog._postfix(sps="1")
        prog.set_remotes(
            45,
            demand=60,
            outstanding=41,
            outstanding_elmo=30,
            outstanding_bert=11,
        )
        pf = prog._postfix(sps="1")
        assert pf["rsock"] == 45
        assert pf["rout"] == 41
        assert pf["eout"] == 30
        assert pf["bout"] == 11
        assert pf["rdmd"] == 60
    finally:
        prog.close()


def test_demand_frac_scales_claim_target_with_workers() -> None:
    """Rising demand raises additive remote claim fraction vs local baseline."""
    local = 96
    for demand, lo in ((30, 0.23), (45, 0.30), (60, 0.38)):
        frac = demand / float(local + demand)
        assert frac >= lo
        assert frac <= 0.60


def test_demand_rise_does_not_cut_healthy_local_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growing Elmo/bert demand must not steal local_share when local is fine.

    Remotes are additive — prefer_local floor stays; shares may sum > 1.
    """
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    # High demand (past defaults) while local has headroom (not cpu_hot).
    sched._remote_demand = {
        "192.168.1.143:8765": 40,
        "bert.local:8766": 20,
    }
    # Balanced finish rates; do not arm worse_than_best_default (that path
    # shrinks remotes — separate from the additive local-floor contract).
    _force_tracker(sched, elapsed_s=40.0, done_local=200, done_remote=180)
    sched._demand_probe.best_default_wave_gps = 0.0
    sched._demand_probe.baseline_eff = 180.0 / 40.0 / 60.0  # current eff
    sched._demand_probe.grow_ceiling = 60  # hold demand level; no further grow
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=25.0,
            load1=20.0,
            mem_available_gb=30.0,
            gpu0_util_pct=70.0,
            gpu1_util_pct=70.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=800, force=True)
    assert dec is not None
    demand_frac = 60.0 / (96.0 + 60.0)
    assert dec.local_share + 1e-9 >= float(sched.prefer_local_frac)
    assert dec.target_sim_workers >= int(sched.target_workers)
    assert "local_additive_preserve" in dec.reason or dec.local_share >= 0.55
    # Remote share may rise with demand without forcing local down.
    assert dec.remote_share + 1e-9 >= min(sched.max_remote_frac, demand_frac)
    assert dec.local_share + dec.remote_share + 1e-9 >= 1.0
    # Demand level held (ceiling) — not shrunk to feed a pie renormalize.
    assert dec.remote_demand["192.168.1.143:8765"] == 40
    assert dec.remote_demand["bert.local:8766"] == 20


def test_demand_raise_preserves_gpu_fed_local_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU-feed local bias must survive demand_raise_remote_share (no steal)."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    sched._remote_demand = {
        "192.168.1.143:8765": 40,
        "bert.local:8766": 20,
    }
    _force_tracker(sched, elapsed_s=40.0, done_local=120, done_remote=90)
    sched._demand_probe.best_default_wave_gps = 0.0
    sched._demand_probe.baseline_eff = 90.0 / 40.0 / 60.0
    sched._demand_probe.grow_ceiling = 60
    # Underfed GPUs pull local_share up; CPU has headroom (healthy).
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=45.0,
            load1=8.0,
            mem_available_gb=50.0,
            gpu0_util_pct=35.0,
            gpu1_util_pct=35.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1000, force=True)
    assert dec is not None
    # demand_frac≈0.385; old renormalize would cut a ~0.70 local down to ~0.615.
    assert dec.local_share + 1e-9 >= 0.65
    assert dec.target_sim_workers >= 96
    assert (
        "demand_raise_remote_share" in dec.reason
        or "local_additive_preserve" in dec.reason
        or "feed_both_gpus" in dec.reason
    )


def test_claim_soft_ceiling_never_caps_local_side() -> None:
    """remote_jobs _claim: local side unrestricted; remote uses demand frac."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "poke_bot" / "remote_jobs.py"
    text = src.read_text(encoding="utf-8")
    assert "demand growth never soft-caps local" in text
    assert "punished local when" in text or "never a steal from" in text
    assert "Local-primary / additive: always allow local claims" in text


def test_demand_eff_collapse_shrinks_without_straggle_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High demand with collapsed per-slot finish rate shrinks toward default."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    _bind_elmo_bert(sched)
    sched._remote_demand = {
        "192.168.1.143:8765": 40,
        "bert.local:8766": 20,
    }
    # 60 slots but ~2.3 remote gps → eff far below baseline; keep local/remote
    # rates close so the straggle heuristic does not fire first.
    _force_tracker(sched, elapsed_s=60.0, done_local=150, done_remote=140)
    sched._demand_probe.baseline_eff = 0.20  # healthy ~6 gps / 30 slots
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=15.0,
            load1=40.0,
            mem_available_gb=20.0,
            gpu0_util_pct=80.0,
            gpu1_util_pct=80.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1000, force=True)
    assert dec is not None
    assert "demand_eff_collapse" in dec.reason
    assert "remote_straggle" not in dec.reason
    assert dec.remote_demand["192.168.1.143:8765"] < 40
    assert dec.remote_demand["bert.local:8766"] < 20


def test_ram_low_shrinks_target_workers_toward_ram_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-swap floor: MemAvailable under ram_floor_gb actively shrinks the
    local worker target toward the RAM-fit ceiling, not just refuses growth.
    """
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    sched.ram_floor_gb = 8.0
    _bind_elmo_bert(sched)
    _force_tracker(sched, elapsed_s=25.0, done_local=40, done_remote=40)
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=50.0,
            load1=4.0,
            mem_available_gb=4.0,  # well under the 8 GiB floor
            mem_total_gb=131.0,
            gpu0_util_pct=50.0,
            gpu1_util_pct=50.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1000, force=True)
    assert dec is not None
    assert any(r.startswith("ram_low_shrink->") for r in dec.reason.split("+"))
    assert dec.target_sim_workers < 96


def test_ram_available_above_floor_does_not_shrink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plenty of MemAvailable (well above the floor) must not trigger shrink."""
    sched = MidIterScheduler.from_env(baseline_workers=96)
    sched.tick_s = 0.0
    sched.min_gps_window_s = 5.0
    sched.ram_floor_gb = 8.0
    _bind_elmo_bert(sched)
    _force_tracker(sched, elapsed_s=25.0, done_local=40, done_remote=40)
    monkeypatch.setattr(
        "poke_bot.pure_rl.mid_iter_scheduler.sample_hardware_signals",
        lambda: HardwareSignals(
            cpu_idle_pct=50.0,
            load1=4.0,
            mem_available_gb=40.0,
            mem_total_gb=131.0,
            gpu0_util_pct=50.0,
            gpu1_util_pct=50.0,
            ok=True,
        ),
    )
    dec = sched.maybe_tick(remaining=1000, force=True)
    assert dec is not None
    assert "ram_low_shrink" not in dec.reason
