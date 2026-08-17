#!/usr/bin/env python3
"""Bounded 2,048-game endurance test for Bert's selected CPU topology."""

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


ROUNDS = 8
GAMES_PER_ROUND = 256


def main() -> int:
    report: dict[str, Any] = json.loads(STATUS.read_text())
    endurance = {
        "status": "running",
        "round": 0,
        "rounds": ROUNDS,
        "games_completed": 0,
        "games_target": ROUNDS * GAMES_PER_ROUND,
        "decisions_completed": 0,
        "elapsed_s": 0.0,
        "errors": [],
        "peak_tree_rss_gb": 0.0,
        "minimum_free_ram_gb": None,
        "round_results": [],
        "started_at": time.time(),
    }
    report.update(
        {
            "status": "running",
            "stage": f"cpu-endurance:0/{ROUNDS}",
            "active_variant": "cpu-w8-l1-1t",
            "apple_cpu_endurance": endurance,
        }
    )
    _publish(report)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
            "CG_LIB_PATH": str(CG_LIB),
            "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": "20260717",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    worker_command = [
        str(PYTHON),
        str(ROOT / "scripts" / "run_remote_worker.py"),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--workers", "8",
        "--default-workers", "8",
        "--leaf-servers", "1",
        "--leaf-gpu", "cpu",
        "--leaf-max-batch", "32",
        "--leaf-queue-depth", "32",
        "--leaf-coalesce-ms", "1",
        "--max-connections", "12",
        "--tree-rss-limit-gb", "18",
        "--min-free-ram-gb", "8",
        "--checkpoint", str(CHECKPOINT),
        "--cg-lib-path", str(CG_LIB),
    ]
    child: subprocess.Popen[Any] | None = None

    def _shutdown(signum: int, _frame: object) -> None:
        if child is not None:
            _stop_worker(child)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        child = subprocess.Popen(worker_command, cwd=ROOT, env=env)
        report["worker_pid"] = child.pid
        _publish(report)
        _wait_for_worker(child)
        for round_index in range(ROUNDS):
            report["stage"] = f"cpu-endurance:{round_index + 1}/{ROUNDS}"
            endurance["round"] = round_index + 1
            _publish(report)
            output = STATUS.parent / f"m4_cpu_endurance_round_{round_index + 1:02d}.json"
            command = [
                str(PYTHON),
                str(ROOT / "scripts" / "benchmark_remote_model_gps.py"),
                "--endpoint", f"127.0.0.1:{PORT}",
                "--checkpoint", str(CHECKPOINT),
                "--games", str(GAMES_PER_ROUND),
                "--concurrency", "8",
                "--timeout", "600",
                "--seed", str(950000 + round_index * GAMES_PER_ROUND),
                "--no-reload",
                "--quiet",
                "--json-out", str(output),
            ]
            for deck in DECKS:
                command.extend(("--deck", str(deck)))
            completed = subprocess.run(command, cwd=ROOT, env=env)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"CPU endurance round {round_index + 1} failed: rc={completed.returncode}"
                )
            result = json.loads(output.read_text())
            summary = {
                key: result.get(key)
                for key in (
                    "games_completed",
                    "games_per_s",
                    "decisions_per_s",
                    "elapsed_s",
                    "usable_game_fraction",
                    "tree_rss_gb",
                    "free_ram_gb",
                )
            }
            summary["error_count"] = len(result.get("errors") or [])
            endurance["round_results"].append(summary)
            endurance["games_completed"] += int(result.get("games_completed") or 0)
            endurance["decisions_completed"] += int(
                round(
                    float(result.get("decisions_per_s") or 0.0)
                    * float(result.get("elapsed_s") or 0.0)
                )
            )
            endurance["elapsed_s"] += float(result.get("elapsed_s") or 0.0)
            endurance["errors"].extend(result.get("errors") or [])
            endurance["peak_tree_rss_gb"] = max(
                float(endurance["peak_tree_rss_gb"] or 0.0),
                float(result.get("tree_rss_gb") or 0.0),
            )
            free_ram = result.get("free_ram_gb")
            if free_ram is not None:
                endurance["minimum_free_ram_gb"] = min(
                    float(endurance["minimum_free_ram_gb"])
                    if endurance["minimum_free_ram_gb"] is not None
                    else float(free_ram),
                    float(free_ram),
                )
            endurance["games_per_s"] = endurance["games_completed"] / max(
                float(endurance["elapsed_s"]), 1e-9
            )
            endurance["decisions_per_s"] = endurance["decisions_completed"] / max(
                float(endurance["elapsed_s"]), 1e-9
            )
            _publish(report)
        endurance["status"] = "complete"
        endurance["completed_at"] = time.time()
        _stop_worker(child)
        child = None
        report["status"] = "complete"
        report["stage"] = "cpu-endurance:complete"
        report["active_variant"] = None
        report["worker_pid"] = None
        _publish(report)
        return 0
    except BaseException as exc:
        endurance["status"] = "failed"
        endurance["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        report["stage"] = "cpu-endurance:failed"
        report["error"] = endurance["error"]
        _publish(report)
        raise
    finally:
        if child is not None:
            _stop_worker(child)


if __name__ == "__main__":
    raise SystemExit(main())
