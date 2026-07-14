#!/usr/bin/env python
"""Benchmark cross-game trusted belief-MCTS leaf batching."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.worker_pool import WorkerPool

_POSTERIOR = None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--servers", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=128)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--coalesce-ms", type=float, default=2.0)
    parser.add_argument("--move-time", type=float, default=8.0)
    parser.add_argument("--sims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def _worker(job: dict) -> dict:
    global _POSTERIOR
    import torch

    from poke_bot import batched_infer, cg_env, deck_pool
    from poke_bot.agent import PolicyAgent
    from poke_bot.baselines_runtime import load_manifest
    from poke_bot.belief import EmpiricalDeckPosterior

    if _POSTERIOR is None:
        _POSTERIOR = EmpiricalDeckPosterior.from_manifest()
    specs = load_manifest()
    own = deck_pool.primary_deck()
    from poke_bot.deck_pool import read_deck

    opponent = read_deck(specs[int(job["opponent_index"]) % len(specs)].deck_csv)
    backend = batched_infer.remote_leaf_backend_from_worker()
    if backend is None:
        raise RuntimeError("benchmark worker lacks remote leaf backend")
    obs, _ = cg_env.battle_start(own, opponent)
    try:
        agent = PolicyAgent(
            model=None,
            deck=own,
            use_mcts=True,
            belief_mcts=True,
            belief_posterior=_POSTERIOR,
            checkpoint_digest=job["digest"],
            model_generation=int(job["generation"]),
            max_sims=int(job["sims"]),
            move_time_s=float(job["move_time"]),
            collect_targets=True,
            strict_runtime=True,
            rng=random.Random(int(job["seed"])),
            device=torch.device("cpu"),
            leaf_backend=backend,
        )
        started = time.perf_counter()
        action = agent(obs)
        elapsed = time.perf_counter() - started
        diagnostics = dict(agent.targets[-1]["diagnostics"])
        return {
            "ok": True,
            "action": action,
            "elapsed_s": elapsed,
            "diagnostics": diagnostics,
        }
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        cg_env.battle_finish()


def _gpu_sampler(index: int, stop: threading.Event, rows: list[float]) -> None:
    while not stop.wait(0.2):
        try:
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(index),
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            rows.append(float(value.strip()))
        except Exception:
            pass


def main() -> int:
    args = _args()
    digest = checkpoint.checkpoint_digest(args.checkpoint)
    ctx = mp.get_context("spawn")
    response_queues = [ctx.Queue(maxsize=2) for _ in range(args.workers)]
    request_queues = [
        ctx.Queue(maxsize=args.queue_depth) for _ in range(args.servers)
    ]
    control_queues = [ctx.Queue(maxsize=8) for _ in range(args.servers)]
    status_queues = [ctx.Queue(maxsize=16) for _ in range(args.servers)]
    alive_events = [ctx.Event() for _ in range(args.servers)]
    ready_events = [ctx.Event() for _ in range(args.servers)]
    servers = []
    from poke_bot.batched_infer import run_leaf_server

    for index in range(args.servers):
        process = ctx.Process(
            target=run_leaf_server,
            args=(
                str(args.checkpoint),
                args.gpu,
                request_queues[index],
                response_queues,
            ),
            kwargs={
                "ready_evt": ready_events[index],
                "alive_evt": alive_events[index],
                "ctrl_q": control_queues[index],
                "status_q": status_queues[index],
                "expected_digest": digest,
                "initial_version": 0,
                "max_batch": args.max_batch,
                "coalesce_ms": args.coalesce_ms,
            },
            daemon=True,
        )
        process.start()
        servers.append(process)
    for event, status_queue in zip(ready_events, status_queues):
        if not event.wait(120):
            raise RuntimeError("leaf server startup timeout")
        status = status_queue.get(timeout=5)
        if not status.get("ok"):
            raise RuntimeError(f"leaf server startup failed: {status}")
    slot_counter = ctx.Value("i", 0)
    remote = {
        "req_qs": request_queues,
        "resp_qs": response_queues,
        "slot_counter": slot_counter,
        "ctrl_qs": control_queues,
        "generation": 1,
        "alive_evts": alive_events,
        "expected_digest": digest,
        "expected_version": 0,
        "timeout_s": max(60.0, args.move_time * 2),
    }
    reports = []
    try:
        with WorkerPool(
            num_workers=args.workers, remote_channel=remote
        ) as pool:
            for budget in args.sims:
                jobs = [
                    {
                        "digest": digest,
                        "generation": 1,
                        "sims": budget,
                        "move_time": args.move_time,
                        "seed": 1000 + index,
                        "opponent_index": index,
                    }
                    for index in range(args.workers * args.repeats)
                ]
                gpu_rows: list[float] = []
                stop = threading.Event()
                sampler = threading.Thread(
                    target=_gpu_sampler,
                    args=(args.physical_gpu_index, stop, gpu_rows),
                    daemon=True,
                )
                sampler.start()
                started = time.perf_counter()
                results = list(pool.imap_unordered(_worker, jobs, chunksize=1))
                wall = time.perf_counter() - started
                stop.set()
                sampler.join(timeout=2)
                failures = [row for row in results if not row.get("ok")]
                if failures:
                    raise RuntimeError(f"benchmark failures: {failures[:3]}")
                diagnostics = [row["diagnostics"] for row in results]
                report = {
                    "sims_planned": budget,
                    "games": len(results),
                    "wall_s": wall,
                    "decisions_per_s": len(results) / max(wall, 1e-9),
                    "completed_sims": sum(row["sims_run"] for row in diagnostics),
                    "sims_per_s": sum(row["sims_run"] for row in diagnostics)
                    / max(wall, 1e-9),
                    "max_depth": max(row["max_depth"] for row in diagnostics),
                    "max_complete_ordered_action_count": max(
                        row.get("complete_ordered_action_count", 0)
                        for row in diagnostics
                    ),
                    "factorized_decisions": sum(
                        row.get("action_space_mode")
                        == "exact_autoregressive_hierarchical"
                        for row in diagnostics
                    ),
                    "mean_depth": statistics.fmean(
                        row["mean_depth"] for row in diagnostics
                    ),
                    "batch_mean": statistics.fmean(
                        row["inference_batch_size_mean"] for row in diagnostics
                    ),
                    "queue_wait_ms_mean": statistics.fmean(
                        row["queue_wait_ms_mean"] for row in diagnostics
                    ),
                    "gpu_util_mean": (
                        statistics.fmean(gpu_rows) if gpu_rows else None
                    ),
                    "gpu_util_max": max(gpu_rows) if gpu_rows else None,
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
    finally:
        for control, request in zip(control_queues, request_queues):
            try:
                control.put_nowait({"cmd": "stop"})
            except Exception:
                pass
            try:
                request.put_nowait(None)
            except Exception:
                pass
        for process in servers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for queue_obj in (
            *request_queues,
            *control_queues,
            *status_queues,
            *response_queues,
        ):
            try:
                queue_obj.cancel_join_thread()
            except Exception:
                pass
            try:
                queue_obj.close()
            except Exception:
                pass
    print(json.dumps({"config": vars(args), "reports": reports}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
