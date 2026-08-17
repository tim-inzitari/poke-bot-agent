#!/usr/bin/env python
"""Adaptive dual-GPU resource watcher (separate, additive, non-destructive).

Periodically samples **CPU / RAM / per-GPU VRAM** and *ratchets a recommended
knob plan upward* whenever there is sustained headroom (so both the Blackwell
hammer RL and the 3080 Ti core-kernel jobs can be pushed harder), while staying
inside OOM-safe ceilings (VRAM ~85%, RAM budget ~100-110 GB, CPU oversub bound).

Design goals (per the dual-GPU plan)
------------------------------------
* **Separate process**: never imported by the training loops; it only observes
  and writes an advisory plan. Safe to start/stop independently.
* **Anti-oscillation**: a knob only bumps up after ``--hysteresis`` consecutive
  headroom samples AND at least ``--min-bump-interval`` seconds since the last
  bump. Under pressure it backs off immediately (fast down, slow up).
* **OOM-safe ceilings**: hard maxima per knob; VRAM/RAM/CPU pressure gates any
  further increase. It ratchets — it does not thrash between two values.
* **Advisory by default**: writes ``outputs/state/resource_plan.json`` (env
  overrides consumed on the *next* (re)launch of a job) and logs every sample.
  It does **not** kill or restart anything unless ``--manage-core-kernel`` is
  set (opt-in), and it never touches the hammer RL process.

Knobs it plans (env-overridable; consumed by config.py on relaunch)
-------------------------------------------------------------------
  POKEBOT_RL_GAMES_IN_FLIGHT   hammer in-flight self-play games (Blackwell side)
  POKEBOT_LEAF_SERVER_REPLICAS hammer GPU leaf-eval server replicas (Blackwell)
  POKEBOT_SIM_WORKERS          CPU sim workers (shared CPU pool)
  POKEBOT_TI_GAMES_PER_BATCH   3080 Ti core-kernel games/optimizer-batch
  POKEBOT_TI_MAX_DECISIONS     3080 Ti core-kernel decision budget/micro-batch
  POKEBOT_TORCH_THREADS        core-kernel intra-op threads (leave room for sims)

Usage
-----
    python scripts/resource_watcher.py --interval 30 \
        --log outputs/logs/resource_watcher.log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# An r241 source snapshot is intentionally read-only and code-only.  Direct
# invocations of this helper must honor the same durable artifact boundary as
# the parent launcher, rather than defaulting to ``<snapshot>/outputs``.
_outputs_override = os.environ.get("POKEBOT_OUTPUTS_DIR", "").strip()
OUTPUTS_ROOT = (
    Path(_outputs_override).expanduser().resolve()
    if _outputs_override
    else ROOT / "outputs"
)


# --------------------------------------------------------------------------
# Sampling helpers (no torch import — cheap + isolated)
# --------------------------------------------------------------------------

def _cpu_count() -> int:
    return os.cpu_count() or 32


def sample_ram() -> dict:
    """Return {total_gb, available_gb, used_gb, used_pct} from /proc/meminfo."""
    meta: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                meta[k.strip()] = int(rest.strip().split()[0])  # kB
    except Exception:
        return {"total_gb": 0.0, "available_gb": 0.0, "used_gb": 0.0, "used_pct": 0.0}
    total = meta.get("MemTotal", 0) / 1e6
    avail = meta.get("MemAvailable", meta.get("MemFree", 0)) / 1e6
    used = max(0.0, total - avail)
    return {
        "total_gb": round(total, 1),
        "available_gb": round(avail, 1),
        "used_gb": round(used, 1),
        "used_pct": round((used / total * 100.0) if total else 0.0, 1),
    }


def sample_cpu(interval: float = 0.5) -> float:
    """System-wide CPU utilisation percent over a short window (/proc/stat)."""
    def _read():
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            parts = fh.readline().split()[1:]
        vals = list(map(int, parts))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle
    try:
        t1, i1 = _read()
        time.sleep(interval)
        t2, i2 = _read()
        dt = t2 - t1
        di = i2 - i1
        return round((1.0 - di / dt) * 100.0, 1) if dt > 0 else 0.0
    except Exception:
        return 0.0


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout


def sample_gpus() -> list[dict]:
    """Per-GPU {index, name, mem_used_gb, mem_total_gb, mem_pct, util} via nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = _run([
            smi,
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
    except Exception:
        return []
    gpus: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            used = float(parts[2]) / 1024.0
            total = float(parts[3]) / 1024.0
            util = float(parts[4])
        except ValueError:
            continue
        gpus.append({
            "index": idx,
            "name": parts[1],
            "mem_used_gb": round(used, 2),
            "mem_total_gb": round(total, 2),
            "mem_pct": round((used / total * 100.0) if total else 0.0, 1),
            "util": util,
        })
    return gpus


def _classify(gpus: list[dict]) -> tuple[dict | None, dict | None]:
    """Return (blackwell_gpu, ti_gpu) by name match."""
    bw = ti = None
    for g in gpus:
        n = g["name"].lower()
        if "blackwell" in n or "pro 5000" in n or "rtx pro" in n:
            bw = g
        elif "3080" in n:
            ti = g
    return bw, ti


# --------------------------------------------------------------------------
# Knob plan with ratchet + ceilings
# --------------------------------------------------------------------------

@dataclass
class Knob:
    env: str
    value: int
    step: int
    ceiling: int
    floor: int

    def bump(self) -> bool:
        if self.value + self.step <= self.ceiling:
            self.value += self.step
            return True
        return False

    def backoff(self) -> bool:
        if self.value - self.step >= self.floor:
            self.value -= self.step
            return True
        return False


def _ram_worker_ceiling(default_leaf_count: int = 60) -> int:
    """No-swap RAM ceiling for local worker knobs (env/`/proc/meminfo`-aware).

    Torch-free import (``poke_bot.live_pool`` has no heavy deps) so this
    watcher keeps its "cheap + isolated" design goal. Falls back to the
    caller's fixed ceiling if the helper is unavailable for any reason.
    """
    try:
        from poke_bot.live_pool import max_local_workers_for_ram

        return max_local_workers_for_ram(leaf_count=default_leaf_count, min_workers=8)
    except Exception:
        return 160


def default_knobs(cores: int) -> dict[str, Knob]:
    # Steady defaults stay moderate; ceilings are headroom (max ≫ default) —
    # but never above what physical RAM can sustain without swap. The old
    # fixed 160 ceiling let the watcher ratchet local workers past the
    # no-swap RAM budget overnight; clamp it to the RAM-fit ceiling instead.
    # Leaves: 3080 Ti hard-capped (no headroom); all leaf growth → Blackwell.
    ram_ceiling = min(160, _ram_worker_ceiling())
    worker_ceiling = max(32, ram_ceiling)
    worker_floor = min(32, worker_ceiling)
    in_flight_floor = min(24, worker_ceiling)
    return {
        # Blackwell hammer side / pure-RL local collect
        "rl_games_in_flight": Knob(
            "POKEBOT_RL_GAMES_IN_FLIGHT",
            min(40, worker_ceiling),
            4,
            worker_ceiling,
            in_flight_floor,
        ),
        # Total leaf replicas; GPU0 pinned ≤12, growth assigned to GPU1 only.
        "leaf_server_replicas": Knob("POKEBOT_LEAF_SERVER_REPLICAS", 30, 2, 60, 16),
        # shared CPU sim pool
        "sim_workers": Knob(
            "POKEBOT_SIM_WORKERS",
            min(cores, worker_ceiling),
            4,
            worker_ceiling,
            worker_floor,
        ),
        # 3080 Ti core-kernel side
        "ti_games_per_batch": Knob("POKEBOT_TI_GAMES_PER_BATCH", 24, 4, 48, 12),
        "ti_max_decisions": Knob("POKEBOT_TI_MAX_DECISIONS", 2048, 512, 6144, 1024),
        "torch_threads": Knob("POKEBOT_TORCH_THREADS", 8, 2, 16, 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=30.0, help="Sample period (s).")
    ap.add_argument("--log", type=Path,
                    default=OUTPUTS_ROOT / "logs" / "resource_watcher.log")
    ap.add_argument("--plan", type=Path,
                    default=OUTPUTS_ROOT / "state" / "resource_plan.json")
    ap.add_argument(
        "--live-pool-plan",
        type=Path,
        default=OUTPUTS_ROOT / "state" / "live_pool_plan.json",
        help="Iteration-boundary plan for running trainers "
             "(workers / leaf_servers). Pure-RL + RR consume at next iter.",
    )
    ap.add_argument(
        "--emit-live-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write --live-pool-plan on bumps/backoff (default ON for max "
             "throughput). Disable with --no-emit-live-pool.",
    )
    ap.add_argument("--hysteresis", type=int, default=4,
                    help="Consecutive headroom samples required before a bump.")
    ap.add_argument("--min-bump-interval", type=float, default=180.0,
                    help="Min seconds between successive up-bumps (anti-oscillation).")
    # OOM-safe ceilings / gates
    ap.add_argument("--vram-max-pct", type=float, default=85.0)
    ap.add_argument("--ram-max-gb", type=float, default=110.0)
    ap.add_argument("--ram-cushion-gb", type=float, default=10.0,
                    help="Require this much free RAM before bumping RAM-heavy knobs.")
    ap.add_argument("--cpu-max-pct", type=float, default=92.0,
                    help="Above this, back off CPU-heavy knobs; below headroom, bump.")
    ap.add_argument("--cpu-headroom-pct", type=float, default=70.0,
                    help="Below this CPU util is considered 'has headroom'.")
    ap.add_argument("--manage-core-kernel", action="store_true",
                    help="OPT-IN: restart the core-kernel with the new plan on a bump "
                         "(resumes from checkpoint). Off by default = advisory only.")
    ap.add_argument(
        "--promotion-workers",
        type=int,
        default=8,
        help="Promotion worker count stamped into live_pool_plan (RR).",
    )
    args = ap.parse_args(argv)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    cores = _cpu_count()
    knobs = default_knobs(cores)
    live_seq = 0

    # Resume from prior plans so a watcher restart does not snap back to floor.
    try:
        if args.plan.is_file():
            prev = json.loads(args.plan.read_text(encoding="utf-8"))
            prev_knobs = prev.get("knobs") or {}
            for name, knob in knobs.items():
                raw = prev_knobs.get(name) or {}
                if "value" in raw:
                    knob.value = max(knob.floor, min(knob.ceiling, int(raw["value"])))
        if args.live_pool_plan.is_file():
            live_prev = json.loads(args.live_pool_plan.read_text(encoding="utf-8"))
            live_seq = max(0, int(live_prev.get("seq") or 0))
            lw = live_prev.get("workers")
            ll = live_prev.get("leaf_servers")
            if lw is not None:
                knobs["rl_games_in_flight"].value = max(
                    knobs["rl_games_in_flight"].floor,
                    min(knobs["rl_games_in_flight"].ceiling, int(lw)),
                )
            if ll is not None:
                knobs["leaf_server_replicas"].value = max(
                    knobs["leaf_server_replicas"].floor,
                    min(knobs["leaf_server_replicas"].ceiling, int(ll)),
                )
    except Exception:
        pass

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with args.log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def write_plan(reason: str) -> None:
        nonlocal live_seq
        plan = {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "env": {k.env: k.value for k in knobs.values()},
            "knobs": {name: asdict(k) for name, k in knobs.items()},
        }
        tmp = args.plan.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(plan, indent=2) + "\n")
        tmp.replace(args.plan)
        if args.emit_live_pool:
            live_seq += 1
            try:
                from poke_bot.live_pool import (
                    _MAX_LEAF_GPU0,
                    _MAX_LEAF_GPU1,
                    write_live_pool_plan,
                )

                # Prefer in-flight games for workers; fall back to sim_workers.
                workers = int(knobs["rl_games_in_flight"].value)
                leaf_servers = int(knobs["leaf_server_replicas"].value)
                env_l0 = os.environ.get("PURE_RL_LEAF_GPU0_REPLICAS")
                env_l1 = os.environ.get("PURE_RL_LEAF_GPU1_REPLICAS")
                # 3080 Ti: pin at min(env/plan, hard max). Never grow GPU0.
                # Sibling agents may shrink GPU0 further; we only enforce the cap.
                prev_g0 = None
                try:
                    if args.live_pool_plan.is_file():
                        prev_live = json.loads(
                            args.live_pool_plan.read_text(encoding="utf-8")
                        )
                        if prev_live.get("leaf_gpu0") is not None:
                            prev_g0 = int(prev_live["leaf_gpu0"])
                except Exception:
                    prev_g0 = None
                if env_l0 and str(env_l0).strip():
                    leaf_gpu0 = max(1, min(int(env_l0), _MAX_LEAF_GPU0))
                elif prev_g0 is not None:
                    leaf_gpu0 = max(1, min(prev_g0, _MAX_LEAF_GPU0))
                else:
                    leaf_gpu0 = _MAX_LEAF_GPU0
                if env_l1 and str(env_l1).strip():
                    leaf_gpu1 = max(1, min(int(env_l1), _MAX_LEAF_GPU1))
                else:
                    leaf_gpu1 = max(1, leaf_servers - leaf_gpu0)
                # Floor workers at PURE_RL_SIM_WORKERS on every emit — bump/backoff
                # of rl_games_in_flight alone was shrinking the self-play pool to
                # ~44 and starving dual-GPU leaf feed (sticky bind coverage).
                env_w = os.environ.get("PURE_RL_SIM_WORKERS") or os.environ.get(
                    "PURE_RL_GAMES_IN_FLIGHT"
                )
                if env_w is not None and str(env_w).strip():
                    workers = max(workers, int(env_w))
                    knobs["rl_games_in_flight"].value = workers
                if reason == "init":
                    leaf_servers = max(leaf_servers, int(leaf_gpu0) + int(leaf_gpu1))
                    knobs["leaf_server_replicas"].value = leaf_servers
                # All total-leaf growth/backoff beyond the GPU0 pin goes to BW.
                leaf_gpu0 = max(1, min(int(leaf_gpu0), _MAX_LEAF_GPU0))
                leaf_gpu1 = max(1, min(_MAX_LEAF_GPU1, int(leaf_servers) - leaf_gpu0))
                leaf_servers = int(leaf_gpu0) + int(leaf_gpu1)
                knobs["leaf_server_replicas"].value = leaf_servers
                write_live_pool_plan(
                    seq=live_seq,
                    workers=workers,
                    leaf_servers=leaf_servers,
                    leaf_gpu0=leaf_gpu0,
                    leaf_gpu1=leaf_gpu1,
                    promotion_workers=int(args.promotion_workers),
                    reason=reason,
                    path=args.live_pool_plan,
                    apply="next_iter",
                )
            except Exception as exc:
                log(f"[watcher] live_pool_plan write failed: {exc!r}")

    knob_headroom = ", ".join(
        f"{k.env} default/start={k.value} ceiling={k.ceiling} (max>default={k.ceiling > k.value})"
        for k in knobs.values()
    )
    log(f"[watcher] start pid={os.getpid()} cores={cores} interval={args.interval}s "
        f"hysteresis={args.hysteresis} min_bump={args.min_bump_interval}s "
        f"ceilings: vram<={args.vram_max_pct}% ram<={args.ram_max_gb}GB "
        f"cpu<={args.cpu_max_pct}% emit_live_pool={args.emit_live_pool}")
    log(f"[watcher] knob headroom (max≫steady): {knob_headroom}")
    write_plan("init")

    ok_streak = 0
    last_bump = 0.0
    while True:
        try:
            cpu = sample_cpu(0.5)
            ram = sample_ram()
            gpus = sample_gpus()
            bw, ti = _classify(gpus)
            bw_pct = bw["mem_pct"] if bw else 0.0
            ti_pct = ti["mem_pct"] if ti else 0.0
            bw_util = bw["util"] if bw else 0.0
            ti_util = ti["util"] if ti else 0.0

            # Pressure = any resource above its safe ceiling.
            vram_pressure = bw_pct >= args.vram_max_pct or ti_pct >= args.vram_max_pct
            ram_pressure = ram["used_gb"] >= args.ram_max_gb or \
                ram["available_gb"] <= args.ram_cushion_gb
            cpu_pressure = cpu >= args.cpu_max_pct

            status = (f"[sample] cpu={cpu:.0f}% ram={ram['used_gb']:.0f}/"
                      f"{ram['total_gb']:.0f}GB(avail {ram['available_gb']:.0f}) "
                      f"bw_vram={bw_pct:.0f}%(util {bw_util:.0f}) "
                      f"ti_vram={ti_pct:.0f}%(util {ti_util:.0f})")

            if vram_pressure or ram_pressure or cpu_pressure:
                ok_streak = 0
                changed = []
                # Fast backoff: relieve whichever resource is stressed.
                if cpu_pressure or ram_pressure:
                    for name in ("rl_games_in_flight", "sim_workers", "ti_games_per_batch"):
                        if knobs[name].backoff():
                            changed.append(name)
                if vram_pressure:
                    for name in ("ti_max_decisions", "ti_games_per_batch",
                                 "leaf_server_replicas"):
                        if knobs[name].backoff():
                            changed.append(name)
                if changed:
                    write_plan("backoff: " + ",".join(sorted(set(changed))))
                    log(f"{status} PRESSURE -> backoff {sorted(set(changed))}")
                else:
                    log(f"{status} PRESSURE (already at floor)")
            else:
                # Headroom on all resources.
                ok_streak += 1
                cpu_headroom = cpu < args.cpu_headroom_pct
                ram_headroom = ram["available_gb"] > args.ram_cushion_gb * 2
                now = time.time()
                can_bump = (ok_streak >= args.hysteresis
                            and now - last_bump >= args.min_bump_interval)
                if can_bump:
                    changed = []
                    # Push GPU-cheap / util-bound knobs first; gate CPU/RAM knobs.
                    if ti_pct < args.vram_max_pct - 10:
                        if knobs["ti_max_decisions"].bump():
                            changed.append("ti_max_decisions")
                    if cpu_headroom and ram_headroom:
                        for name in ("ti_games_per_batch", "rl_games_in_flight",
                                     "sim_workers"):
                            if knobs[name].bump():
                                changed.append(name)
                                break  # one CPU-knob bump per interval (gentle)
                    # Grow leaves only on Blackwell VRAM/util headroom.
                    # Never use 3080 Ti underfeed as a reason to add replicas
                    # (GPU0 is hard-capped; growth is assigned to GPU1 only).
                    if (
                        ram_headroom
                        and bw_pct < args.vram_max_pct - 15
                        and ti_pct < args.vram_max_pct  # Ti already safe
                        and bw_util < 90.0
                    ):
                        steps = 2 if bw_util < 60.0 else 1
                        for _ in range(steps):
                            if knobs["leaf_server_replicas"].bump():
                                if "leaf_server_replicas" not in changed:
                                    changed.append("leaf_server_replicas")
                            else:
                                break
                    if changed:
                        last_bump = now
                        ok_streak = 0
                        write_plan("bump: " + ",".join(changed))
                        log(f"{status} HEADROOM streak={args.hysteresis} -> BUMP {changed} "
                            f"plan={{ {', '.join(f'{k.env}={k.value}' for k in knobs.values())} }}")
                        if args.manage_core_kernel:
                            log("[watcher] --manage-core-kernel set: core-kernel should be "
                                "relaunched to consume plan (delegated to launch wrapper).")
                    else:
                        log(f"{status} HEADROOM (all knobs at ceiling)")
                else:
                    log(f"{status} headroom streak={ok_streak}/{args.hysteresis}")
        except Exception as exc:  # never die on a transient sampling error
            log(f"[watcher] sample error: {exc!r}")
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
