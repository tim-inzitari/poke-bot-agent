#!/usr/bin/env python3
"""Sweep Bert CPU worker/leaf topology and run a long stability validation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from typing import Any

from run_apple_optimization import (
    CG_LIB,
    CHECKPOINT,
    DECKS,
    PORT,
    PYTHON,
    ROOT,
    STATUS,
    _publish,
    _stop_worker,
    _wait_for_worker,
)


def _worker_command(*, workers: int, leaves: int, threads: int) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "run_remote_worker.py"),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--workers", str(workers),
        "--default-workers", str(workers),
        "--leaf-servers", str(leaves),
        "--leaf-gpu", "cpu",
        "--leaf-max-batch", "32",
        "--leaf-queue-depth", "32",
        "--leaf-coalesce-ms", "1",
        "--max-connections", str(workers + 4),
        "--tree-rss-limit-gb", "18",
        "--min-free-ram-gb", "8",
        "--checkpoint", str(CHECKPOINT),
        "--cg-lib-path", str(CG_LIB),
    ]


def _benchmark_command(
    *, name: str, games: int, concurrency: int, seed: int
) -> tuple[list[str], Any]:
    output = STATUS.parent / f"m4_whole_game_{name}.json"
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "benchmark_remote_model_gps.py"),
        "--endpoint", f"127.0.0.1:{PORT}",
        "--checkpoint", str(CHECKPOINT),
        "--games", str(games),
        "--concurrency", str(concurrency),
        "--timeout", "600",
        "--seed", str(seed),
        "--no-reload",
        "--json-out", str(output),
    ]
    for deck in DECKS:
        command.extend(("--deck", str(deck)))
    return command, output


def main() -> int:
    report: dict[str, Any] = json.loads(STATUS.read_text())
    report.update(
        {
            "status": "running",
            "stage": "cpu-scaling",
            "active_variant": None,
            "worker_pid": None,
            "scaling_started_at": time.time(),
        }
    )
    report.pop("error", None)
    report.setdefault("scaling_results", {})
    _publish(report)

    child: subprocess.Popen[Any] | None = None

    def _shutdown(signum: int, _frame: object) -> None:
        if child is not None:
            _stop_worker(child)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    baseline_hashes = (
        report.get("results", {}).get("cpu-2t", {}).get("game_fingerprints", {})
    )
    variants = (
        {"name": "cpu-w6-l1-1t", "workers": 6, "leaves": 1, "threads": 1},
        {"name": "cpu-w8-l1-1t", "workers": 8, "leaves": 1, "threads": 1},
        {"name": "cpu-w8-l2-1t", "workers": 8, "leaves": 2, "threads": 1},
        {"name": "cpu-w10-l2-1t", "workers": 10, "leaves": 2, "threads": 1},
        {"name": "cpu-w8-l2-2t", "workers": 8, "leaves": 2, "threads": 2},
    )
    try:
        for variant in variants:
            name = str(variant["name"])
            workers = int(variant["workers"])
            leaves = int(variant["leaves"])
            threads = int(variant["threads"])
            prior = report["scaling_results"].get(name) or {}
            if (
                int(prior.get("games_completed") or 0) == 16
                and not prior.get("errors")
            ):
                print(f"[apple-scale] reuse completed {name}", flush=True)
                continue
            report["stage"] = f"cpu-scaling:{name}"
            report["active_variant"] = name
            _publish(report)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(ROOT),
                    "PYTHONUNBUFFERED": "1",
                    "CG_LIB_PATH": str(CG_LIB),
                    "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": "20260717",
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                }
            )
            print(
                f"[apple-scale] start {name} workers={workers} "
                f"leaves={leaves} threads={threads}",
                flush=True,
            )
            child = subprocess.Popen(
                _worker_command(workers=workers, leaves=leaves, threads=threads),
                cwd=ROOT,
                env=env,
            )
            report["worker_pid"] = child.pid
            _publish(report)
            _wait_for_worker(child)
            command, output = _benchmark_command(
                name=name, games=16, concurrency=workers, seed=946000
            )
            completed = subprocess.run(command, cwd=ROOT, env=env)
            if completed.returncode != 0:
                raise RuntimeError(f"{name} benchmark failed: rc={completed.returncode}")
            result = json.loads(output.read_text())
            hashes = result.get("game_fingerprints") or {}
            common = set(hashes) & set(baseline_hashes)
            matches = sum(hashes[key] == baseline_hashes[key] for key in common)
            result["gameplay_reproducibility_vs_cpu_w4_l1_2t"] = {
                "compared_games": len(common),
                "matching_gameplay_fingerprints": matches,
                "passed": bool(common) and matches == len(common),
            }
            result["simulator_accuracy"] = {
                "engine": "official libcg",
                "same_engine_as_production": True,
                "policy_leaf_numeric_parity_reference": "m4_leaf_sweep.json",
            }
            report["scaling_results"][name] = result
            _publish(report)
            _stop_worker(child)
            child = None

        eligible = [
            (name, result)
            for name, result in report["scaling_results"].items()
            if int(result.get("games_completed") or 0) == 16
            and not result.get("errors")
        ]
        if not eligible:
            raise RuntimeError("no CPU scaling topology completed cleanly")
        best_name, best_result = max(
            eligible, key=lambda item: float(item[1].get("games_per_s") or 0.0)
        )
        best = next(item for item in variants if item["name"] == best_name)
        report["recommended_cpu_topology"] = {
            **best,
            "sweep_games_per_s": best_result.get("games_per_s"),
            "sweep_decisions_per_s": best_result.get("decisions_per_s"),
        }
        report["stage"] = f"stability:{best_name}"
        report["active_variant"] = best_name
        _publish(report)

        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(ROOT),
                "PYTHONUNBUFFERED": "1",
                "CG_LIB_PATH": str(CG_LIB),
                "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": "20260717",
                "OMP_NUM_THREADS": str(best["threads"]),
                "MKL_NUM_THREADS": str(best["threads"]),
            }
        )
        child = subprocess.Popen(
            _worker_command(
                workers=int(best["workers"]),
                leaves=int(best["leaves"]),
                threads=int(best["threads"]),
            ),
            cwd=ROOT,
            env=env,
        )
        report["worker_pid"] = child.pid
        _publish(report)
        _wait_for_worker(child)
        command, output = _benchmark_command(
            name=f"{best_name}-stability",
            games=256,
            concurrency=int(best["workers"]),
            seed=947000,
        )
        completed = subprocess.run(command, cwd=ROOT, env=env)
        if completed.returncode != 0:
            raise RuntimeError(f"{best_name} stability failed: rc={completed.returncode}")
        report["stability_result"] = json.loads(output.read_text())
        _stop_worker(child)
        child = None
        report["status"] = "complete"
        report["stage"] = "complete"
        report["active_variant"] = None
        report["worker_pid"] = None
        report["scaling_completed_at"] = time.time()
        _publish(report)
        print(
            f"[apple-scale] complete recommend={best_name} "
            f"gps={report['stability_result'].get('games_per_s')} "
            f"games={report['stability_result'].get('games_completed')}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["stage"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_at"] = time.time()
        _publish(report)
        raise
    finally:
        if child is not None:
            _stop_worker(child)


if __name__ == "__main__":
    raise SystemExit(main())
