"""Mid-iteration collect scheduler (hot rebalance without killing in-flight games).

Metrics contract
----------------
* **Wave wall-clock GPS / ETA only.** ``gps = completed / (now - wave_t0)``
  (and side-specific cumulative rates). Optional window EMA requires a minimum
  window (default 20s) so a remote *batch completion burst* cannot look like
  sustained GPS.
* **Never** use inter-arrival / batch-dump instantaneous rates for rebalance.

Remote dispatch contract
------------------------
* Remotes keep **chunked** job lists (large ``remote_dispatch_chunk``) so each
  LAN socket streams many games before asking the scheduler again — amortizes
  Elmo/bert RTT. Do **not** switch to chatty per-game claim/RPC for remotes.
* Rebalance only affects *which side claims the next chunk*, chunk sizing
  hints, and **remote demand** (in-flight sockets per endpoint) — in-flight
  games drain normally.
* Remote demand: **max ≫ default** (Elmo 20→40, Bert 10→20). Grow/shrink is
  gated on **live wave completion** (wall-clock GPS / settle probe), not hello
  slots or early batch-dump ``remote_gps``. Grow only when a prior level proved
  it did not hurt wave ETA; shrink when a grow degrades wave GPS or per-slot
  finish rate collapses. Never stick the steady default at the ceiling.
* **Additive vs local:** growing remote demand must not cut healthy local
  workers/slots. Remotes grow on top; ``max_total_workers`` is non-binding
  (default 10000). Soft local floor (``prefer_local_frac``) stays when local
  util/GPS is fine. Never shrink local to feed remotes — only shrink remotes
  on completion-time hurt.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from poke_bot.live_pool import max_local_workers_for_ram


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class WaveGpsTracker:
    """Wall-clock wave GPS — immune to batch-arrival spikes.

    Batch dumps (many remote results in one poll) must not inflate the rate
    used for local/remote rebalance. Cumulative ``done/elapsed`` from wave
    start is the primary signal; window EMA only updates after ``min_window_s``.
    """

    min_window_s: float = 20.0
    ema_alpha: float = 0.35
    t0: float = field(default_factory=time.monotonic)
    done_total: int = 0
    done_local: int = 0
    done_remote: int = 0
    decisions_total: int = 0
    decisions_local: int = 0
    decisions_remote: int = 0
    _win_t0: float = field(default_factory=time.monotonic)
    _win_done: int = 0
    ema_gps: float = 0.0
    _ema_inited: bool = False

    def note(self, *, side: str, n: int = 1, decisions: int = 0) -> None:
        n = max(0, int(n))
        decisions = max(0, int(decisions))
        if n <= 0:
            return
        self.done_total += n
        self.decisions_total += decisions
        if side == "remote":
            self.done_remote += n
            self.decisions_remote += decisions
        else:
            self.done_local += n
            self.decisions_local += decisions
        self._win_done += n
        self._maybe_roll_window()

    def _maybe_roll_window(self) -> None:
        now = time.monotonic()
        dt = now - self._win_t0
        if dt < float(self.min_window_s):
            return
        inst = self._win_done / max(dt, 1e-6)
        if not self._ema_inited:
            self.ema_gps = inst
            self._ema_inited = True
        else:
            a = float(self.ema_alpha)
            self.ema_gps = a * inst + (1.0 - a) * self.ema_gps
        self._win_t0 = now
        self._win_done = 0

    def elapsed(self) -> float:
        return max(time.monotonic() - self.t0, 1e-6)

    def wave_gps(self) -> float:
        """Primary rebalance metric: cumulative wall-clock games/s."""
        return float(self.done_total) / self.elapsed()

    def local_gps(self) -> float:
        return float(self.done_local) / self.elapsed()

    def remote_gps(self) -> float:
        return float(self.done_remote) / self.elapsed()

    def wave_sps(self) -> float:
        return float(self.decisions_total) / self.elapsed()

    def local_sps(self) -> float:
        return float(self.decisions_local) / self.elapsed()

    def remote_sps(self) -> float:
        return float(self.decisions_remote) / self.elapsed()

    def eta_s(self, remaining: int) -> float:
        gps = self.wave_gps()
        if gps <= 1e-9:
            return float("inf")
        return float(max(0, int(remaining))) / gps

    def snapshot(self) -> dict[str, float]:
        self._maybe_roll_window()
        return {
            "wave_gps": self.wave_gps(),
            "local_gps": self.local_gps(),
            "remote_gps": self.remote_gps(),
            "wave_sps": self.wave_sps(),
            "local_sps": self.local_sps(),
            "remote_sps": self.remote_sps(),
            "ema_gps": float(self.ema_gps) if self._ema_inited else self.wave_gps(),
            "elapsed_s": self.elapsed(),
            "done_total": float(self.done_total),
            "done_local": float(self.done_local),
            "done_remote": float(self.done_remote),
        }


@dataclass
class HardwareSignals:
    cpu_idle_pct: float = 50.0
    load1: float = 0.0
    mem_available_gb: float = 0.0
    mem_total_gb: float = 0.0
    gpu0_util_pct: float = 0.0
    gpu1_util_pct: float = 0.0
    gpu0_mem_used_mb: float = 0.0
    gpu1_mem_used_mb: float = 0.0
    ok: bool = False


def sample_hardware_signals() -> HardwareSignals:
    """Best-effort host signals (never raises into the collect loop)."""
    sig = HardwareSignals()
    try:
        load1, _, _ = os.getloadavg()
        sig.load1 = float(load1)
        ncpu = max(1, os.cpu_count() or 1)
        # Rough idle from load: 100% when load<<ncpu.
        sig.cpu_idle_pct = float(
            max(0.0, min(100.0, 100.0 * (1.0 - (load1 / float(ncpu)))))
        )
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    sig.mem_available_gb = float(parts[1]) / (1024.0 * 1024.0)
                elif line.startswith("MemTotal:"):
                    parts = line.split()
                    sig.mem_total_gb = float(parts[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        )
        rows = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if rows:
            u0, m0 = [x.strip() for x in rows[0].split(",")]
            sig.gpu0_util_pct = float(u0)
            sig.gpu0_mem_used_mb = float(m0)
        if len(rows) > 1:
            u1, m1 = [x.strip() for x in rows[1].split(",")]
            sig.gpu1_util_pct = float(u1)
            sig.gpu1_mem_used_mb = float(m1)
        sig.ok = True
    except Exception:
        sig.ok = False
    return sig


def _demand_bits(demand: dict[str, int]) -> str:
    if not demand:
        return "remote_demand=-"
    parts = []
    for ep, n in sorted(demand.items()):
        short = ep.split(":")[0]
        if short.startswith("192.168.1.143"):
            short = "elmo"
        elif "bert" in short:
            short = "bert"
        parts.append(f"{short}={n}")
    return "remote_demand=" + ",".join(parts)


@dataclass
class SchedulerDecision:
    local_share: float
    remote_share: float
    target_sim_workers: int
    leaf_gpu0_frac: float
    remote_chunk: int
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    hardware: dict[str, float] = field(default_factory=dict)
    remote_demand: dict[str, int] = field(default_factory=dict)

    def as_log(self) -> str:
        m = self.metrics
        h = self.hardware
        return (
            f"scheduler=mid_iter local_share={self.local_share:.2f} "
            f"remote_share={self.remote_share:.2f} "
            f"target_workers={self.target_sim_workers} "
            f"leaf_gpu0_frac={self.leaf_gpu0_frac:.2f} "
            f"remote_chunk={self.remote_chunk} "
            f"{_demand_bits(self.remote_demand)} "
            f"wave_gps={m.get('wave_gps', 0):.2f} "
            f"local_gps={m.get('local_gps', 0):.2f} "
            f"remote_gps={m.get('remote_gps', 0):.2f} "
            f"wave_sps={m.get('wave_sps', 0):.1f} "
            f"local_sps={m.get('local_sps', 0):.1f} "
            f"remote_sps={m.get('remote_sps', 0):.1f} "
            f"ema_gps={m.get('ema_gps', 0):.2f} "
            f"cpu_idle={h.get('cpu_idle_pct', 0):.0f}% "
            f"gpu0={h.get('gpu0_util_pct', 0):.0f}% "
            f"gpu1={h.get('gpu1_util_pct', 0):.0f}% "
            f"mem_avail_gb={h.get('mem_available_gb', 0):.1f} "
            f"metrics=wave_wall_clock_gps "
            f"remote_batch=chunked(rtt_amortize) "
            f"demand_gate=completion "
            f"reason={self.reason}"
        )


@dataclass
class DemandCompletionProbe:
    """Gate remote demand on measured wave completion, not advertised slots.

    After a grow/shrink, wait ``settle_s`` then compare wall-clock wave GPS.
    A grow that worsens wave GPS (or collapses per-slot remote finish rate)
    is rolled back and further grows are cooled down.

    Hard rules for wave-wall-time minimization:
    * A hurtful grow ratchets ``grow_ceiling`` so the failed level cannot be
      re-attempted (persists across waves/iters — prevents 20/10 → 25/12 hurt
      → re-grow → 30/14 every new collect).
    * Probe resolve never chains into another grow on the same tick.
    * Track best wave GPS observed at/under default demand; if demand is above
      default and wave GPS falls well below that best, force shrink even without
      a pending grow probe.
    * Smart-loading intelligence (demand, ceilings, best GPS, shares) is
      snapshotted to disk so a new ``_collect_wave`` does not forget.
    """

    settle_s: float = 20.0
    degrade_frac: float = 0.08  # >8% wave GPS drop after grow → shrink
    improve_frac: float = 0.03  # need ≥3% GPS gain to call grow "helped"
    eff_collapse_frac: float = 0.55  # per-slot remote gps vs baseline
    # After hurt: long cool-down. 45s was shorter than settle+settle and the
    # controller re-grew into the same bad level within one minute.
    cooldown_s: float = 180.0
    pending: str = ""  # "grow" | "shrink" | ""
    probe_mono: float = 0.0
    probe_demand_sum: int = 0  # demand after the action being probed
    probe_pre_demand_sum: int = 0  # demand before a grow (ratchet target)
    probe_wave_gps: float = 0.0
    probe_ema_gps: float = 0.0
    probe_remote_gps: float = 0.0
    baseline_eff: float = 0.0  # remote_gps / demand_sum at defaults
    grow_blocked_until: float = 0.0
    # Max demand_sum allowed to grow toward this wave (ratchet on hurt).
    # 0 = no ceiling yet. After hurt, set to pre-grow sum (usually defaults).
    grow_ceiling: int = 0
    # Best wall-clock wave GPS seen while demand ≤ defaults (20/10 operating point).
    best_default_wave_gps: float = 0.0
    last_action: str = ""
    # Set when a probe resolves this tick — blocks immediate re-grow.
    just_resolved: bool = False


@dataclass
class MidIterScheduler:
    """Live local/remote + leaf-bias + remote-demand controller for one collect wave."""

    min_local_frac: float = 0.40
    prefer_local_frac: float = 0.55
    max_remote_frac: float = 0.60
    # Floor remote share when remotes are bound — GPU-feed must not crush
    # additive Elmo/bert capacity down to ~0 while local stays at 96.
    min_remote_frac: float = 0.25
    target_workers: int = 96  # steady *local* default; max_workers is local headroom
    min_workers: int = 64
    max_workers: int = 160  # ≫ default so underfed local CPU/GPU can scale up
    # Absurd total ceiling — never a binding constraint. Realistic peak is
    # local_max+elmo_max+bert_max ≈ 160+40+20=220 ≪ 10000; remotes stay
    # additive on top of local without the total cap stealing local slots.
    max_total_workers: int = 10_000
    # Soft RAM floor (GiB): below this MemAvailable, actively shrink the
    # local worker target toward the RAM-fit ceiling (no-swap policy).
    ram_floor_gb: float = 8.0
    leaf_gpu0_frac: float = 12.0 / 42.0  # 12/30 map (3080 Ti 12GB hard-cap)
    remote_chunk: int = 128
    tick_s: float = 15.0
    min_gps_window_s: float = 20.0
    remote_defaults: dict[str, int] = field(default_factory=dict)
    remote_maxima: dict[str, int] = field(default_factory=dict)
    tracker: WaveGpsTracker = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _last_tick: float = field(default_factory=time.monotonic, init=False)
    _decision: SchedulerDecision = field(init=False)
    _ticks: int = 0
    _remote_demand: dict[str, int] = field(default_factory=dict, init=False)
    _demand_probe: DemandCompletionProbe = field(
        default_factory=DemandCompletionProbe, init=False
    )

    def __post_init__(self) -> None:
        self.min_local_frac = min(0.95, max(0.05, float(self.min_local_frac)))
        self.prefer_local_frac = min(
            0.95, max(self.min_local_frac, float(self.prefer_local_frac))
        )
        self.max_remote_frac = min(
            1.0 - self.min_local_frac, max(0.05, float(self.max_remote_frac))
        )
        self.min_remote_frac = min(
            self.max_remote_frac, max(0.0, float(self.min_remote_frac))
        )
        self.max_total_workers = max(
            int(self.max_workers), int(self.max_total_workers)
        )
        self.remote_chunk = max(8, int(self.remote_chunk))
        self.tracker = WaveGpsTracker(
            min_window_s=float(self.min_gps_window_s),
            ema_alpha=_env_float("PURE_RL_REBALANCE_EMA_ALPHA", 0.35),
        )
        self._demand_probe = DemandCompletionProbe(
            settle_s=_env_float(
                "PURE_RL_DEMAND_SETTLE_S", float(self.min_gps_window_s)
            ),
            degrade_frac=_env_float("PURE_RL_DEMAND_DEGRADE_FRAC", 0.08),
            improve_frac=_env_float("PURE_RL_DEMAND_IMPROVE_FRAC", 0.03),
            eff_collapse_frac=_env_float("PURE_RL_DEMAND_EFF_COLLAPSE_FRAC", 0.55),
            cooldown_s=_env_float("PURE_RL_DEMAND_GROW_COOLDOWN_S", 180.0),
        )
        # Start at defaults — never at the ceiling.
        self._remote_demand = {
            ep: int(self.remote_defaults.get(ep, n))
            for ep, n in {**self.remote_maxima, **self.remote_defaults}.items()
        }
        for ep, dflt in self.remote_defaults.items():
            mx = int(self.remote_maxima.get(ep, dflt))
            self._remote_demand[ep] = max(1, min(int(dflt), mx))
        self._decision = SchedulerDecision(
            local_share=self.prefer_local_frac,
            remote_share=1.0 - self.prefer_local_frac,
            target_sim_workers=int(self.target_workers),
            leaf_gpu0_frac=float(self.leaf_gpu0_frac),
            remote_chunk=int(self.remote_chunk),
            reason="init",
            metrics=self.tracker.snapshot(),
            remote_demand=dict(self._remote_demand),
        )

    @classmethod
    def from_env(cls, *, baseline_workers: int) -> "MidIterScheduler":
        return cls(
            min_local_frac=_env_float("PURE_RL_REBALANCE_MIN_LOCAL_FRAC", 0.40),
            prefer_local_frac=_env_float(
                "PURE_RL_REBALANCE_PREFER_LOCAL_FRAC", 0.55
            ),
            max_remote_frac=_env_float("PURE_RL_REBALANCE_MAX_REMOTE_FRAC", 0.60),
            min_remote_frac=_env_float("PURE_RL_REBALANCE_MIN_REMOTE_FRAC", 0.25),
            target_workers=_env_int(
                "PURE_RL_SIM_WORKERS", max(96, int(baseline_workers))
            ),
            min_workers=_env_int("PURE_RL_REBALANCE_MIN_WORKERS", 64),
            max_workers=_env_int("PURE_RL_REBALANCE_MAX_WORKERS", 160),
            # Default 10000 ≫ realistic peak (160+40+20); never binds scaling.
            max_total_workers=_env_int(
                "PURE_RL_REBALANCE_MAX_TOTAL_WORKERS", 10_000
            ),
            ram_floor_gb=_env_float("PURE_RL_REBALANCE_RAM_FLOOR_GB", 8.0),
            leaf_gpu0_frac=_env_float("PURE_RL_LEAF_GPU0_FRAC", 12.0 / 42.0),
            remote_chunk=_env_int("PURE_RL_REMOTE_DISPATCH_CHUNK", 128),
            tick_s=_env_float("PURE_RL_SCHEDULER_TICK_S", 15.0),
            min_gps_window_s=_env_float("PURE_RL_WAVE_GPS_MIN_WINDOW_S", 20.0),
        )

    def bind_remote_endpoints(self, clients: list[Any]) -> None:
        """Attach per-endpoint default/max demand from hello + config caps."""
        from poke_bot.remote_jobs import (
            endpoint_default_workers,
            endpoint_max_workers,
            remote_capacity_workers,
        )

        with self._lock:
            for client in clients:
                ep = str(getattr(client, "endpoint", "") or "")
                if not ep:
                    continue
                host = str(getattr(client, "host", "") or "")
                port = getattr(client, "port", None)
                info = getattr(client, "info", None)
                dflt = endpoint_default_workers(host, port)
                mx = endpoint_max_workers(host, port)
                if info is not None:
                    capacity = remote_capacity_workers(info)
                    # Cannot demand above real process-pool capacity.
                    mx = min(mx, max(1, int(capacity)))
                    # Prefer remote-advertised default when present and sane.
                    adv_default = int(getattr(info, "default_workers", 0) or 0)
                    if adv_default > 0:
                        dflt = min(max(1, adv_default), mx)
                    else:
                        # Legacy hello: workers field is the advertise/default.
                        adv = int(getattr(info, "workers", 0) or 0)
                        if adv > 0:
                            dflt = min(max(1, adv), mx)
                # Ensure default stays strictly below max when headroom exists.
                if mx > dflt:
                    pass
                else:
                    # Capacity not yet redeployed — demand capped at capacity.
                    dflt = min(dflt, mx)
                self.remote_defaults[ep] = int(dflt)
                self.remote_maxima[ep] = int(mx)
                # Preserve prior demand if already bound; else start at default.
                if ep not in self._remote_demand:
                    self._remote_demand[ep] = int(dflt)
                else:
                    self._remote_demand[ep] = max(
                        1, min(int(self._remote_demand[ep]), int(mx))
                    )
            self._decision = SchedulerDecision(
                local_share=self._decision.local_share,
                remote_share=self._decision.remote_share,
                target_sim_workers=self._decision.target_sim_workers,
                leaf_gpu0_frac=self._decision.leaf_gpu0_frac,
                remote_chunk=self._decision.remote_chunk,
                reason=self._decision.reason + "+bind_remotes",
                metrics=self.tracker.snapshot(),
                hardware=dict(self._decision.hardware),
                remote_demand=dict(self._remote_demand),
            )

    def remote_demand(self) -> dict[str, int]:
        with self._lock:
            return dict(self._remote_demand)

    def note_completed(
        self, *, side: str, n: int = 1, decisions: int = 0
    ) -> None:
        with self._lock:
            self.tracker.note(side=side, n=n, decisions=decisions)

    def decision(self) -> SchedulerDecision:
        with self._lock:
            return self._decision

    def _demand_sum(self) -> int:
        return int(sum(int(v) for v in self._remote_demand.values()))

    def _default_demand_sum(self) -> int:
        return int(
            sum(
                int(self.remote_defaults.get(ep, n))
                for ep, n in self._remote_demand.items()
            )
        )

    def _arm_demand_probe(
        self,
        *,
        action: str,
        metrics: dict[str, float],
        reasons: list[str],
        pre_demand_sum: int = 0,
    ) -> None:
        probe = self._demand_probe
        probe.pending = action
        probe.probe_mono = time.monotonic()
        probe.probe_demand_sum = self._demand_sum()
        probe.probe_pre_demand_sum = int(pre_demand_sum) or int(
            probe.probe_demand_sum
        )
        probe.probe_wave_gps = float(metrics.get("wave_gps", 0.0))
        probe.probe_ema_gps = float(metrics.get("ema_gps", probe.probe_wave_gps))
        probe.probe_remote_gps = float(metrics.get("remote_gps", 0.0))
        probe.last_action = action
        reasons.append(f"demand_probe_{action}")

    def _step_demand(self, *, grow: bool, reasons: list[str]) -> bool:
        """Nudge each endpoint toward max (grow) or default (shrink).

        Returns True if any endpoint demand changed.
        Grow respects ``probe.grow_ceiling`` (0 = uncapped).
        """
        changed = False
        ceiling = int(self._demand_probe.grow_ceiling)
        for ep, cur in list(self._remote_demand.items()):
            dflt = int(self.remote_defaults.get(ep, cur))
            mx = int(self.remote_maxima.get(ep, cur))
            if mx < 1:
                continue
            span = max(1, mx - dflt)
            step = max(1, span // 4)
            short = "elmo" if "143" in ep else ("bert" if "bert" in ep else ep)
            if grow:
                if cur < mx:
                    nxt = min(mx, cur + step)
                    # Ratchet: refuse steps that would push total above a
                    # previously failed demand_sum this wave.
                    if ceiling > 0:
                        other = self._demand_sum() - int(cur)
                        if other + nxt > ceiling:
                            nxt = max(int(cur), ceiling - other)
                            nxt = min(nxt, mx)
                    if nxt != cur:
                        self._remote_demand[ep] = nxt
                        reasons.append(f"remote_grow_{short}->{nxt}")
                        changed = True
            else:
                if cur > dflt:
                    nxt = max(dflt, cur - step)
                    if nxt != cur:
                        self._remote_demand[ep] = nxt
                        reasons.append(f"remote_shrink_{short}->{nxt}")
                        changed = True
        return changed

    def _resolve_demand_probe(
        self, *, metrics: dict[str, float], reasons: list[str]
    ) -> Optional[str]:
        """Return 'shrink' / 'hold' / 'ok' / 'hold_level' once settled."""
        probe = self._demand_probe
        probe.just_resolved = False
        if not probe.pending:
            return None
        now = time.monotonic()
        if (now - float(probe.probe_mono)) < float(probe.settle_s):
            reasons.append("demand_probe_settling")
            return "hold"
        wave = float(metrics.get("wave_gps", 0.0))
        ema = float(metrics.get("ema_gps", wave))
        # Prefer the more conservative of EMA / cumulative once warm so a
        # sticky cumulative climb cannot mask a live GPS collapse (and so a
        # noisy EMA spike cannot declare "helped" into another grow).
        if float(metrics.get("elapsed_s", 0.0)) >= 2.0 * float(self.min_gps_window_s):
            cur = min(ema, wave) if ema > 1e-9 else wave
        else:
            cur = wave
        baseline = max(
            float(probe.probe_ema_gps), float(probe.probe_wave_gps), 1e-9
        )
        demand_sum = max(1, self._demand_sum())
        rg = float(metrics.get("remote_gps", 0.0))
        eff = rg / float(demand_sum)
        if probe.baseline_eff <= 1e-12 and probe.probe_demand_sum > 0:
            probe.baseline_eff = float(probe.probe_remote_gps) / max(
                1.0, float(probe.probe_demand_sum)
            )
        pending = probe.pending
        probe.pending = ""
        probe.just_resolved = True
        if pending == "grow":
            degraded = cur < baseline * (1.0 - float(probe.degrade_frac))
            # Also treat "no real improvement" after grow as failure when the
            # pre-grow operating point was already healthy — prevents laddering
            # 25→30→35 on flat/noisy GPS (live: demand_helped then remotes=44@~5gps).
            flat = cur < baseline * (1.0 + float(probe.improve_frac))
            eff_bad = (
                probe.baseline_eff > 1e-9
                and eff < probe.baseline_eff * float(probe.eff_collapse_frac)
            )
            if degraded or eff_bad:
                probe.grow_blocked_until = now + float(probe.cooldown_s)
                # Ceiling at the pre-grow demand so we cannot re-enter the
                # failed level this wave (typically defaults 20/10 = 30).
                pre = max(
                    1,
                    int(probe.probe_pre_demand_sum)
                    or (int(probe.probe_demand_sum) - 1),
                )
                if probe.grow_ceiling <= 0:
                    probe.grow_ceiling = pre
                else:
                    probe.grow_ceiling = min(int(probe.grow_ceiling), pre)
                reasons.append(
                    "demand_hurt_wave_gps"
                    if degraded
                    else "demand_eff_collapse"
                )
                reasons.append(f"demand_grow_ceiling={probe.grow_ceiling}")
                return "shrink"
            if flat:
                # Hold the new level but do not celebrate / chain-grow.
                probe.grow_blocked_until = max(
                    float(probe.grow_blocked_until),
                    now + float(probe.settle_s),
                )
                reasons.append("demand_grow_flat_hold")
                return "hold_level"
            reasons.append("demand_helped_wave_gps")
            # Refresh baseline efficiency at the new sustainable level.
            if eff > probe.baseline_eff:
                probe.baseline_eff = eff
            return "ok"
        reasons.append("demand_shrink_settled")
        return "ok"

    def maybe_tick(self, *, remaining: int, force: bool = False) -> Optional[SchedulerDecision]:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_tick) < float(self.tick_s):
                return None
            self._last_tick = now
            prev = self._decision
            hw = sample_hardware_signals()
            metrics = self.tracker.snapshot()
            local_share = float(self.prefer_local_frac)
            leaf_g0 = float(self.leaf_gpu0_frac)
            workers = int(self.target_workers)
            reasons: list[str] = []

            # Wave GPS sides (cumulative) — never batch-burst instants.
            lg = float(metrics["local_gps"])
            rg = float(metrics["remote_gps"])
            elapsed = float(metrics["elapsed_s"])
            if elapsed >= float(self.min_gps_window_s) and (lg + rg) > 1e-6:
                # Remotes ahead is informational for demand grow/hold — do NOT
                # cut local_share (remotes are additive; punishing healthy local
                # to "make room" is forbidden).
                if rg > lg * 1.25 and rg > 0.5:
                    reasons.append("remote_wave_gps_ahead")
                elif lg > rg * 1.25 and lg > 0.5:
                    local_share = min(1.0 - 0.05, local_share + 0.05)
                    reasons.append("local_wave_gps_ahead")

            # Fill freed CPU/RAM: bias local when idle; ease when saturated.
            local_bottleneck = False
            if hw.cpu_idle_pct >= 35.0 and hw.mem_available_gb >= 20.0:
                local_share = min(1.0 - 0.05, local_share + 0.05)
                workers = min(self.max_workers, max(workers, self.target_workers))
                reasons.append("cpu_ram_headroom")
            elif hw.cpu_idle_pct <= 10.0:
                local_share = max(self.min_local_frac, local_share - 0.05)
                local_bottleneck = True
                reasons.append("cpu_hot")

            # No-swap RAM floor: MemAvailable dropping under the soft floor
            # (or swap already in use) means the host cannot sustain the
            # current worker count without paging. Actively SHRINK the
            # target toward the RAM-fit ceiling rather than only refusing
            # to grow — a stale high target_workers (e.g. left over from a
            # prior headroom bump) must not keep planning 96/160 once RAM
            # cannot fit it. Overrides the additive target_workers floor
            # below (that floor exists to stop *demand* from cutting local
            # workers, not to defeat a real no-swap safety shrink).
            ram_shrink_active = False
            if hw.mem_available_gb > 0.0 and hw.mem_available_gb < float(
                self.ram_floor_gb
            ):
                ram_cap = max_local_workers_for_ram(
                    mem_total_gb=hw.mem_total_gb or None,
                    mem_available_gb=hw.mem_available_gb,
                    min_workers=int(self.min_workers),
                )
                if ram_cap < workers:
                    workers = max(int(self.min_workers), ram_cap)
                    local_bottleneck = True
                    ram_shrink_active = True
                    reasons.append(f"ram_low_shrink->{workers}")

            # Dual-GPU underfed: pull work home so striped leaf farms stay busy.
            if hw.ok and hw.gpu0_util_pct < 55.0 and hw.gpu1_util_pct < 55.0:
                local_share = min(1.0 - 0.05, local_share + 0.10)
                reasons.append("feed_both_gpus")
            elif hw.ok and min(hw.gpu0_util_pct, hw.gpu1_util_pct) < 40.0:
                local_share = min(1.0 - 0.05, local_share + 0.05)
                reasons.append("feed_starved_gpu")

            # Leaf bias: starve neither GPU; feed underused 3080.
            # (Routing uses striped leaf_cuda_devices; frac is for live_pool hints.)
            gpu0_starved = bool(
                hw.ok
                and hw.gpu0_util_pct < 50.0
                and hw.gpu0_util_pct + 20.0 < hw.gpu1_util_pct
            )
            if hw.ok:
                if hw.gpu0_util_pct + 15.0 < hw.gpu1_util_pct:
                    leaf_g0 = min(0.55, leaf_g0 + 0.05)
                    reasons.append("feed_gpu0")
                elif hw.gpu1_util_pct + 15.0 < hw.gpu0_util_pct:
                    leaf_g0 = max(0.30, leaf_g0 - 0.05)
                    reasons.append("feed_gpu1")

            # Hard override AFTER cpu_hot/remote_wave nudges: those often cancel
            # feed_starved_gpu (+0.05 vs -0.05/-0.05) while the 3080 sits at
            # ~10% between MCTS waves. Pull local claim share back up and do
            # not grow remote demand further this tick.
            if gpu0_starved:
                floor = min(1.0 - self.min_remote_frac, max(local_share, self.prefer_local_frac))
                local_share = min(1.0 - self.min_remote_frac, floor + 0.10)
                reasons.append("gpu0_starve_pull_local")

            # Core loop: demand ↔ actual wave completion time.
            # Demand (= target remote workers) may rise toward max; each grow is
            # settle-probed on wall-clock wave GPS. Grow when the system needs
            # more remote finish capacity; shrink when a grow hurts wave ETA /
            # per-slot finish rate. Hello slots alone never drive demand.
            if self._remote_demand and elapsed >= float(self.min_gps_window_s):
                remote_headroom = any(
                    self._remote_demand[ep] < int(self.remote_maxima.get(ep, 0))
                    for ep in self._remote_demand
                )
                demand_sum = max(1, self._demand_sum())
                default_sum = max(1, self._default_demand_sum())
                done_remote = float(metrics["done_remote"])
                remotes_idle = rg < 0.25 and done_remote < 3
                remotes_straggle = (
                    lg > 0.5 and rg > 1e-9 and lg > rg * 1.5 and not local_bottleneck
                )
                # Warmup: ≥1 completion per current demand slot before growing.
                remote_warm = done_remote >= float(demand_sum)
                eff = rg / float(demand_sum)
                wave_gps = float(metrics["wave_gps"])
                probe = self._demand_probe
                if (
                    probe.baseline_eff <= 1e-12
                    and remote_warm
                    and demand_sum <= default_sum
                    and rg >= 0.3
                ):
                    probe.baseline_eff = eff
                # Remember the best 20/10 (default) operating point this wave.
                if demand_sum <= default_sum and wave_gps > 0.5:
                    if wave_gps > float(probe.best_default_wave_gps):
                        probe.best_default_wave_gps = wave_gps
                now_mono = time.monotonic()
                grow_cooled = now_mono < float(probe.grow_blocked_until)
                at_ceiling = (
                    int(probe.grow_ceiling) > 0
                    and demand_sum >= int(probe.grow_ceiling)
                )
                probe_verdict = self._resolve_demand_probe(
                    metrics=metrics, reasons=reasons
                )
                # Above-default demand with wave GPS well below the measured
                # best default point → force shrink (15gps@20/10 beats 6.7@30+).
                worse_than_best_default = bool(
                    demand_sum > default_sum
                    and float(probe.best_default_wave_gps) > 1.0
                    and wave_gps
                    < float(probe.best_default_wave_gps)
                    * (1.0 - float(probe.degrade_frac))
                    and remote_warm
                    and elapsed >= 2.0 * float(self.min_gps_window_s)
                )
                if probe_verdict == "shrink":
                    if self._step_demand(grow=False, reasons=reasons):
                        self._arm_demand_probe(
                            action="shrink", metrics=metrics, reasons=reasons
                        )
                elif probe_verdict in ("hold", "hold_level"):
                    pass  # settling / flat — never chain-grow this tick
                elif probe.just_resolved:
                    # Probe just cleared as ok/helped — re-measure before next grow.
                    reasons.append("demand_probe_just_resolved")
                elif gpu0_starved:
                    reasons.append("gpu0_hold_remote_demand")
                elif remotes_idle or remotes_straggle or worse_than_best_default:
                    if self._step_demand(grow=False, reasons=reasons):
                        self._arm_demand_probe(
                            action="shrink", metrics=metrics, reasons=reasons
                        )
                    if remotes_idle:
                        reasons.append("remote_idle")
                    if remotes_straggle:
                        reasons.append("remote_straggle")
                    if worse_than_best_default:
                        reasons.append("demand_worse_than_best_default")
                        # Ratchet ceiling to defaults so we stay at 20/10.
                        if probe.grow_ceiling <= 0:
                            probe.grow_ceiling = int(default_sum)
                        else:
                            probe.grow_ceiling = min(
                                int(probe.grow_ceiling), int(default_sum)
                            )
                        probe.grow_blocked_until = max(
                            float(probe.grow_blocked_until),
                            now_mono + float(probe.cooldown_s),
                        )
                        reasons.append(
                            f"demand_grow_ceiling={probe.grow_ceiling}"
                        )
                elif (
                    demand_sum > default_sum
                    and probe.baseline_eff > 1e-9
                    and eff
                    < probe.baseline_eff * float(probe.eff_collapse_frac)
                    and remote_warm
                ):
                    if self._step_demand(grow=False, reasons=reasons):
                        self._arm_demand_probe(
                            action="shrink", metrics=metrics, reasons=reasons
                        )
                    reasons.append("demand_eff_collapse")
                elif grow_cooled:
                    reasons.append("demand_grow_cooldown")
                elif at_ceiling:
                    reasons.append(
                        f"demand_at_grow_ceiling={probe.grow_ceiling}"
                    )
                elif not remote_warm:
                    reasons.append("remote_demand_warmup")
                elif remote_headroom and not remotes_idle and rg >= 0.3:
                    # Demand may rise: local needs offload and/or remotes are
                    # sustaining finish rate. Never grow on batch-dump
                    # remote_gps alone without a local need — but DO grow when
                    # local is hot/ahead (target workers should rise).
                    local_needs_offload = bool(
                        local_bottleneck or "local_wave_gps_ahead" in reasons
                    )
                    remotes_keeping_up = bool(
                        eff >= max(1e-9, probe.baseline_eff) * 0.70
                        or probe.baseline_eff <= 1e-12
                    )
                    if local_needs_offload and remotes_keeping_up:
                        pre = int(demand_sum)
                        if self._step_demand(grow=True, reasons=reasons):
                            self._arm_demand_probe(
                                action="grow",
                                metrics=metrics,
                                reasons=reasons,
                                pre_demand_sum=pre,
                            )
                            reasons.append("completion_offload_grow")
                        elif at_ceiling or (
                            int(probe.grow_ceiling) > 0
                            and self._demand_sum() >= int(probe.grow_ceiling)
                        ):
                            reasons.append(
                                f"demand_at_grow_ceiling={probe.grow_ceiling}"
                            )
                    elif (
                        "remote_wave_gps_ahead" in reasons
                        and not local_needs_offload
                    ):
                        # Remotes already ahead, local fine — hold level; probe
                        # will shrink later if wave GPS degrades.
                        reasons.append("remote_ahead_hold_demand")
                    elif local_needs_offload and not remotes_keeping_up:
                        reasons.append("remote_eff_not_ready")
                # Stash demand→target-worker fraction for share clamp below.
                demand_sum = max(1, self._demand_sum())
                metrics["demand_sum"] = float(demand_sum)
                metrics["demand_frac"] = float(demand_sum) / float(
                    max(1, int(self.target_workers)) + demand_sum
                )
                if int(probe.grow_ceiling) > 0:
                    metrics["demand_grow_ceiling"] = float(probe.grow_ceiling)
                reasons.append(f"demand_target_workers={demand_sum}")

            local_share = min(
                1.0 - 0.05, max(self.min_local_frac, float(local_share))
            )
            remote_share = min(self.max_remote_frac, max(0.0, 1.0 - local_share))
            # Couple demand → remote_share so remote claim soft-ceiling rises
            # with completion-gated demand. Remotes are additive on top of
            # local — raising demand must NOT steal local_share when local is
            # healthy (prefer_local floor). Shares may sum > 1 (dual-full).
            local_before_demand = float(local_share)
            local_healthy = not local_bottleneck
            if self._remote_demand:
                demand_frac = float(metrics.get("demand_frac", 0.0))
                if demand_frac > remote_share + 1e-9:
                    remote_share = min(self.max_remote_frac, demand_frac)
                    reasons.append("demand_raise_remote_share")
            if self._remote_demand and self.min_remote_frac > 0.0:
                before = float(remote_share)
                remote_share = max(float(self.min_remote_frac), float(remote_share))
                remote_share = min(self.max_remote_frac, remote_share)
                if remote_share > before + 1e-9:
                    reasons.append("remote_floor")
            # Additive local preserve: never punish healthy local to free pie
            # room for Elmo/bert demand. Only allow local below prefer when
            # local is actually hot/bottlenecked (true offload need).
            local_floor = (
                max(float(self.min_local_frac), float(self.prefer_local_frac))
                if local_healthy
                else float(self.min_local_frac)
            )
            if local_share < local_floor - 1e-9 or (
                local_healthy and local_share + 1e-9 < local_before_demand
            ):
                pinned = max(local_floor, local_before_demand)
                if pinned > local_share + 1e-9:
                    local_share = min(0.95, pinned)
                    reasons.append("local_additive_preserve")
            # Do NOT renormalize to sum=1 — that was the steal path
            # (demand↑ → remote_share↑ → local_share = 1-remote). Keep both.
            local_share = min(0.95, max(self.min_local_frac, float(local_share)))
            remote_share = min(
                self.max_remote_frac,
                max(float(self.min_remote_frac), float(remote_share)),
            )
            # target_sim_workers stays local baseline — demand never shrinks
            # local worker count (remotes are additive; max_total non-binding).
            # Exception: an active no-swap RAM shrink above must stick.
            if not ram_shrink_active:
                workers = max(int(workers), int(self.target_workers))

            eta = self.tracker.eta_s(remaining)
            if eta < 1e18:
                reasons.append(f"eta={eta:.0f}s")
            if not reasons:
                reasons.append("hold")

            dec = SchedulerDecision(
                local_share=float(local_share),
                remote_share=float(remote_share),
                target_sim_workers=int(workers),
                leaf_gpu0_frac=float(leaf_g0),
                remote_chunk=int(self.remote_chunk),
                reason="+".join(reasons),
                metrics=metrics,
                hardware={
                    "cpu_idle_pct": hw.cpu_idle_pct,
                    "load1": hw.load1,
                    "mem_available_gb": hw.mem_available_gb,
                    "gpu0_util_pct": hw.gpu0_util_pct,
                    "gpu1_util_pct": hw.gpu1_util_pct,
                },
                remote_demand=dict(self._remote_demand),
            )
            self._decision = dec
            self._ticks += 1
            prev_demand = prev.remote_demand or {}
            demand_changed = prev_demand != dec.remote_demand
            changed = (
                abs(dec.local_share - prev.local_share) >= 0.049
                or abs(dec.leaf_gpu0_frac - prev.leaf_gpu0_frac) >= 0.049
                or dec.target_sim_workers != prev.target_sim_workers
                or demand_changed
                or self._ticks <= 1
            )
            return dec if (changed or force or self._ticks <= 1) else dec

    def intelligence_dict(self) -> dict[str, Any]:
        """Export cross-iter smart-loading state (demand, ceilings, best GPS, shares)."""
        with self._lock:
            cooldown_left = max(
                0.0, float(self._demand_probe.grow_blocked_until) - time.monotonic()
            )
            return {
                "version": INTEL_VERSION,
                "remote_demand": {
                    str(k): int(v) for k, v in self._remote_demand.items()
                },
                "grow_ceiling": int(self._demand_probe.grow_ceiling),
                "best_default_wave_gps": float(
                    self._demand_probe.best_default_wave_gps
                ),
                "baseline_eff": float(self._demand_probe.baseline_eff),
                "grow_cooldown_remaining_s": float(cooldown_left),
                "local_share": float(self._decision.local_share),
                "remote_share": float(self._decision.remote_share),
                "leaf_gpu0_frac": float(self._decision.leaf_gpu0_frac),
                "target_sim_workers": int(self._decision.target_sim_workers),
                "ema_gps": float(getattr(self.tracker, "ema_gps", 0.0) or 0.0),
            }

    def apply_intelligence(self, data: dict[str, Any] | None) -> list[str]:
        """Restore cross-iter intelligence. Returns human-readable applied bits."""
        if not data or not isinstance(data, dict):
            return []
        applied: list[str] = []
        with self._lock:
            demand_in = data.get("remote_demand") or {}
            if isinstance(demand_in, dict) and demand_in:
                for ep, n in demand_in.items():
                    ep_s = str(ep)
                    try:
                        want = int(n)
                    except (TypeError, ValueError):
                        continue
                    dflt = int(self.remote_defaults.get(ep_s, want))
                    mx = int(self.remote_maxima.get(ep_s, max(dflt, want)))
                    self._remote_demand[ep_s] = max(
                        1, min(want, mx if mx > 0 else want)
                    )
                applied.append(
                    "demand="
                    + ",".join(
                        f"{k.split(':')[0]}={v}"
                        for k, v in sorted(self._remote_demand.items())
                    )
                )
            try:
                ceil = int(data.get("grow_ceiling") or 0)
            except (TypeError, ValueError):
                ceil = 0
            if ceil > 0:
                self._demand_probe.grow_ceiling = ceil
                applied.append(f"grow_ceiling={ceil}")
            try:
                best = float(data.get("best_default_wave_gps") or 0.0)
            except (TypeError, ValueError):
                best = 0.0
            if best > 0.0:
                self._demand_probe.best_default_wave_gps = best
                applied.append(f"best_default_gps={best:.2f}")
            try:
                beff = float(data.get("baseline_eff") or 0.0)
            except (TypeError, ValueError):
                beff = 0.0
            if beff > 0.0:
                self._demand_probe.baseline_eff = beff
            try:
                cool = float(data.get("grow_cooldown_remaining_s") or 0.0)
            except (TypeError, ValueError):
                cool = 0.0
            if cool > 1.0:
                self._demand_probe.grow_blocked_until = time.monotonic() + cool
                applied.append(f"grow_cooldown={cool:.0f}s")
            # Clear in-flight probe — new wave starts clean settle windows.
            self._demand_probe.pending = ""
            self._demand_probe.just_resolved = False
            try:
                ls = float(data.get("local_share", self._decision.local_share))
                rs = float(data.get("remote_share", self._decision.remote_share))
                g0 = float(data.get("leaf_gpu0_frac", self._decision.leaf_gpu0_frac))
                tw = int(
                    data.get("target_sim_workers", self._decision.target_sim_workers)
                )
            except (TypeError, ValueError):
                ls = float(self._decision.local_share)
                rs = float(self._decision.remote_share)
                g0 = float(self._decision.leaf_gpu0_frac)
                tw = int(self._decision.target_sim_workers)
            ls = min(0.95, max(self.min_local_frac, ls))
            rs = min(self.max_remote_frac, max(self.min_remote_frac, rs))
            tw = max(self.min_workers, min(self.max_workers, tw))
            g0 = min(0.95, max(0.05, g0))
            self._decision = SchedulerDecision(
                local_share=ls,
                remote_share=rs,
                target_sim_workers=tw,
                leaf_gpu0_frac=g0,
                remote_chunk=int(self.remote_chunk),
                reason="restored_intelligence",
                metrics=self.tracker.snapshot(),
                hardware=dict(self._decision.hardware),
                remote_demand=dict(self._remote_demand),
            )
            try:
                ema = float(data.get("ema_gps") or 0.0)
            except (TypeError, ValueError):
                ema = 0.0
            if ema > 0.0:
                self.tracker.ema_gps = ema
                self.tracker._ema_inited = True
                applied.append(f"ema_gps={ema:.2f}")
            applied.append(f"shares=L{ls:.2f}/R{rs:.2f}/g0={g0:.2f}")
        return applied


INTEL_VERSION = 1


def mid_iter_scheduler_enabled() -> bool:
    return _env_bool("PURE_RL_MID_ITER_SCHEDULER", True)


def default_scheduler_intelligence_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "scheduler_intelligence.json"


def load_scheduler_intelligence(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def save_scheduler_intelligence(
    path: Path | str | None, sched: MidIterScheduler
) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = sched.intelligence_dict()  # type: ignore[attr-defined]
    payload["saved_at_unix"] = time.time()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def log_scheduler_banner(sched: MidIterScheduler) -> None:
    dec = sched.decision()
    caps = []
    for ep in sorted(sched.remote_defaults):
        short = "elmo" if "143" in ep else ("bert" if "bert" in ep else ep)
        caps.append(
            f"{short}:default={sched.remote_defaults[ep]}"
            f"/max={sched.remote_maxima.get(ep, '?')}"
        )
    cap_s = " ".join(caps) if caps else "remote_caps=unbound"
    remote_max_sum = sum(int(v) for v in sched.remote_maxima.values())
    max_total = max(int(sched.max_total_workers), int(sched.max_workers) + remote_max_sum)
    print(
        f"[pure_rl] mid_iter_rebalance=on {dec.as_log()} "
        f"({cap_s}; local_target={sched.target_workers} "
        f"local_max={sched.max_workers} max_total={max_total} "
        f"min_remote_frac={sched.min_remote_frac:.2f}; "
        f"remote keeps chunked dispatch for LAN RTT; "
        f"rebalance metrics=wave_wall_clock_gps not batch-burst; "
        f"remote_demand max≫default completion-gated; "
        f"remotes additive on top of local)",
        flush=True,
    )
