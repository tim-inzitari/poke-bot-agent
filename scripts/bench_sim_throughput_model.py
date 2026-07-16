#!/usr/bin/env python3
"""Synthetic microbench: multi-game leaf coalesce vs CPU-local policy evals.

No ``libcg`` / CUDA required. Uses **wave-based** accounting: ``workers``
sim processes advance roughly in lockstep, so coalesce wait is paid once per
wave (not once per request serially).

Also sweeps coalesce_ms × model size proxies to show when GPU leaves win for
the pure-RL ~1.6M policy vs a larger Hope-sized net.

Run::

    python3 scripts/bench_sim_throughput_model.py
    python3 scripts/bench_sim_throughput_model.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-operation costs in milliseconds."""

    cpu_forward_ms: float = 1.8
    featurize_ms: float = 0.35
    ipc_overhead_ms: float = 0.25
    gpu_forward_base_ms: float = 0.40
    gpu_forward_per_leaf_ms: float = 0.012
    sim_step_ms: float = 0.15


def gpu_batch_forward_ms(n_leaves: int, costs: CostModel) -> float:
    return costs.gpu_forward_base_ms + costs.gpu_forward_per_leaf_ms * max(1, n_leaves)


def simulate_cpu_local(
    *,
    workers: int,
    decisions_per_game: int,
    games: int,
    costs: CostModel,
) -> dict:
    """Wave model: each wave completes ``workers`` seat-decisions in parallel."""
    total_decisions = games * decisions_per_game
    policy_evals = total_decisions * 2  # both seats
    per_wave = costs.featurize_ms + costs.cpu_forward_ms + costs.sim_step_ms
    n_waves = math.ceil(policy_evals / max(1, workers))
    wall_ms = n_waves * per_wave
    return {
        "mode": "cpu-local",
        "wall_ms": wall_ms,
        "policy_evals": policy_evals,
        "n_waves": n_waves,
        "sps_proxy": (policy_evals / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
    }


def simulate_gpu_leaf(
    *,
    workers: int,
    decisions_per_game: int,
    games: int,
    coalesce_ms: float,
    max_batch: int,
    leaf_replicas: int,
    costs: CostModel,
    leaf_share: float = 1.0,
) -> dict:
    """Wave model with coalesced GPU forwards.

    Per wave of ``workers`` concurrent requests:
      wall ≈ featurize + ipc + coalesce + gpu_forward(batch) / replicas + sim_step
    CPU-local fraction (1-leaf_share) uses cpu forward instead of IPC/GPU.
    """
    total_decisions = games * decisions_per_game
    policy_evals = total_decisions * 2
    n_waves = math.ceil(policy_evals / max(1, workers))

    # Concurrent requests hitting leaves this wave.
    concurrent = workers
    gpu_concurrent = max(0, int(round(concurrent * leaf_share)))
    cpu_concurrent = concurrent - gpu_concurrent

    per_replica = max(1, leaf_replicas)
    # Spread GPU requests across replicas; each replica's batch size:
    batch = min(max_batch, max(1, math.ceil(gpu_concurrent / per_replica))) if gpu_concurrent else 0
    forward_ms = gpu_batch_forward_ms(batch, costs) if batch else 0.0

    gpu_wave_ms = (
        costs.featurize_ms
        + costs.ipc_overhead_ms
        + coalesce_ms
        + forward_ms
        + costs.sim_step_ms
    )
    cpu_wave_ms = costs.featurize_ms + costs.cpu_forward_ms + costs.sim_step_ms
    # Mixed wave: both paths run concurrently → wall is max.
    if gpu_concurrent and cpu_concurrent:
        wave_ms = max(gpu_wave_ms, cpu_wave_ms)
    elif gpu_concurrent:
        wave_ms = gpu_wave_ms
    else:
        wave_ms = cpu_wave_ms

    wall_ms = n_waves * wave_ms
    return {
        "mode": "gpu-leaf",
        "leaf_share": leaf_share,
        "batch_per_replica": batch,
        "wave_ms": wave_ms,
        "wall_ms": wall_ms,
        "policy_evals": policy_evals,
        "n_waves": n_waves,
        "sps_proxy": (policy_evals / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
    }


def sweep(workers: int, costs_small: CostModel, costs_large: CostModel) -> list[dict]:
    rows = []
    for label, costs in (("pure_rl_1p6M", costs_small), ("hope_large", costs_large)):
        for coalesce in (0.0, 1.0, 4.0):
            cpu = simulate_cpu_local(
                workers=workers, decisions_per_game=40, games=256, costs=costs
            )
            gpu = simulate_gpu_leaf(
                workers=workers,
                decisions_per_game=40,
                games=256,
                coalesce_ms=coalesce,
                max_batch=1024,
                leaf_replicas=9,
                costs=costs,
                leaf_share=1.0,
            )
            rows.append(
                {
                    "model": label,
                    "coalesce_ms": coalesce,
                    "cpu_wall_ms": cpu["wall_ms"],
                    "gpu_wall_ms": gpu["wall_ms"],
                    "speedup": cpu["wall_ms"] / gpu["wall_ms"] if gpu["wall_ms"] else None,
                    "batch_per_replica": gpu["batch_per_replica"],
                }
            )
    return rows


def run_timer_sanity(iters: int = 50_000) -> dict:
    t0 = time.perf_counter()
    x = 0
    for i in range(iters):
        x += i % 7
    elapsed = time.perf_counter() - t0
    return {
        "iters": iters,
        "elapsed_s": elapsed,
        "ns_per_iter": (elapsed / iters) * 1e9,
        "checksum": x,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--decisions", type=int, default=40)
    p.add_argument("--coalesce-ms", type=float, default=4.0)
    p.add_argument("--max-batch", type=int, default=1024)
    p.add_argument("--leaf-replicas", type=int, default=9)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    # Small pure-RL net: CPU forward is cheap → coalesce dominates unless tuned.
    costs_small = CostModel(cpu_forward_ms=1.8, gpu_forward_base_ms=0.35, gpu_forward_per_leaf_ms=0.01)
    # Larger Hope-class net: CPU forward expensive → GPU coalesce wins easily.
    costs_large = CostModel(cpu_forward_ms=12.0, gpu_forward_base_ms=0.8, gpu_forward_per_leaf_ms=0.04)

    cpu = simulate_cpu_local(
        workers=args.workers,
        decisions_per_game=args.decisions,
        games=args.games,
        costs=costs_small,
    )
    both = simulate_gpu_leaf(
        workers=args.workers,
        decisions_per_game=args.decisions,
        games=args.games,
        coalesce_ms=args.coalesce_ms,
        max_batch=args.max_batch,
        leaf_replicas=args.leaf_replicas,
        costs=costs_small,
        leaf_share=1.0,
    )
    both_tuned = simulate_gpu_leaf(
        workers=args.workers,
        decisions_per_game=args.decisions,
        games=args.games,
        coalesce_ms=0.0,
        max_batch=args.max_batch,
        leaf_replicas=args.leaf_replicas,
        costs=costs_small,
        leaf_share=1.0,
    )
    half = simulate_gpu_leaf(
        workers=args.workers,
        decisions_per_game=args.decisions,
        games=args.games,
        coalesce_ms=0.0,
        max_batch=args.max_batch,
        leaf_replicas=args.leaf_replicas,
        costs=costs_small,
        leaf_share=0.5,
    )
    rows = sweep(args.workers, costs_small, costs_large)
    sanity = run_timer_sanity()

    payload = {
        "costs_small_ms": asdict(costs_small),
        "costs_large_ms": asdict(costs_large),
        "config": {
            "workers": args.workers,
            "games": args.games,
            "decisions_per_game": args.decisions,
            "coalesce_ms": args.coalesce_ms,
            "max_batch": args.max_batch,
            "leaf_replicas": args.leaf_replicas,
        },
        "cpu_local_small": cpu,
        "gpu_leaf_both_coalesce_default": both,
        "gpu_leaf_both_coalesce_0": both_tuned,
        "gpu_leaf_us_only_coalesce_0": half,
        "speedup_small_model": {
            "leaf_coalesce_default": cpu["wall_ms"] / both["wall_ms"],
            "leaf_coalesce_0": cpu["wall_ms"] / both_tuned["wall_ms"],
            "leaf_us_only_coalesce_0": cpu["wall_ms"] / half["wall_ms"],
        },
        "sweep": rows,
        "timer_sanity": sanity,
        "note": (
            "Wave-based synthetic model. For pure-RL ~1.6M, default coalesce_ms=4 "
            "can lose to CPU-local; set LEAF_SERVER_COALESCE_MS=0 (or ~1) when "
            "wiring self-play to leaves. Larger nets still prefer GPU leaves."
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("=== sim throughput model (wave-based, synthetic) ===")
    print(
        f"workers={args.workers} games={args.games} decisions/game={args.decisions} "
        f"leaves={args.leaf_replicas}"
    )
    print(
        f"cpu-local (1.6M proxy):           wall={cpu['wall_ms']:.1f}ms  "
        f"sps_proxy={cpu['sps_proxy']:.0f}"
    )
    print(
        f"gpu-leaf both coalesce={args.coalesce_ms}ms: wall={both['wall_ms']:.1f}ms  "
        f"speedup={cpu['wall_ms']/both['wall_ms']:.2f}x  batch/replica≈{both['batch_per_replica']}"
    )
    print(
        f"gpu-leaf both coalesce=0:         wall={both_tuned['wall_ms']:.1f}ms  "
        f"speedup={cpu['wall_ms']/both_tuned['wall_ms']:.2f}x"
    )
    print(
        f"gpu-leaf 50% coalesce=0:          wall={half['wall_ms']:.1f}ms  "
        f"speedup={cpu['wall_ms']/half['wall_ms']:.2f}x"
    )
    print("\nSweep (speedup >1 ⇒ GPU leaf wins):")
    for r in rows:
        print(
            f"  {r['model']:12s} coalesce={r['coalesce_ms']:.0f}ms  "
            f"speedup={r['speedup']:.2f}x  batch/rep={r['batch_per_replica']}"
        )
    print(
        f"\ntimer sanity: {sanity['ns_per_iter']:.1f} ns/iter over {sanity['iters']} loops"
    )


if __name__ == "__main__":
    main()
