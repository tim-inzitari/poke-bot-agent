#!/usr/bin/env python3
"""Validate bulk MPS readback and cache policy in whole seeded games."""

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


def main() -> int:
    report: dict[str, Any] = json.loads(STATUS.read_text())
    report.update(
        {
            "status": "running",
            "stage": "mps-optimized:leaf-parity",
            "active_variant": "mps-bulk-readback-cache0",
            "mps_optimized_started_at": time.time(),
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
            "POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES": "0",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.50",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.40",
        }
    )
    child: subprocess.Popen[Any] | None = None

    def _shutdown(signum: int, _frame: object) -> None:
        if child is not None:
            _stop_worker(child)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        leaf_output = STATUS.parent / "m4_leaf_bulk_readback.json"
        leaf_command = [
            str(PYTHON),
            str(ROOT / "scripts" / "benchmark_apple_leaf.py"),
            "--checkpoint", str(CHECKPOINT),
            "--json-out", str(leaf_output),
            "--batches", "1,4,16,32",
            "--cpu-threads", "2",
            "--target-decisions", "1024",
        ]
        leaf_completed = subprocess.run(leaf_command, cwd=ROOT, env=env)
        if leaf_completed.returncode != 0:
            raise RuntimeError(f"optimized leaf parity failed: rc={leaf_completed.returncode}")
        leaf = json.loads(leaf_output.read_text())
        report["optimized_mps_leaf"] = {
            "status": leaf.get("status"),
            "rows": leaf.get("rows"),
            "best_eligible": leaf.get("best_eligible"),
        }
        report["stage"] = "mps-optimized:whole-game"
        _publish(report)

        worker_command = [
            str(PYTHON),
            str(ROOT / "scripts" / "run_remote_worker.py"),
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--workers", "8",
            "--default-workers", "8",
            "--leaf-servers", "1",
            "--leaf-gpu", "mps",
            "--leaf-max-batch", "32",
            "--leaf-queue-depth", "32",
            "--leaf-coalesce-ms", "2",
            "--max-connections", "12",
            "--tree-rss-limit-gb", "18",
            "--min-free-ram-gb", "8",
            "--checkpoint", str(CHECKPOINT),
            "--cg-lib-path", str(CG_LIB),
        ]
        child = subprocess.Popen(worker_command, cwd=ROOT, env=env)
        report["worker_pid"] = child.pid
        _publish(report)
        _wait_for_worker(child)
        output = STATUS.parent / "m4_whole_game_mps_bulk_cache0.json"
        benchmark = [
            str(PYTHON),
            str(ROOT / "scripts" / "benchmark_remote_model_gps.py"),
            "--endpoint", f"127.0.0.1:{PORT}",
            "--checkpoint", str(CHECKPOINT),
            "--games", "64",
            "--concurrency", "8",
            "--timeout", "600",
            "--seed", "946000",
            "--no-reload",
            "--json-out", str(output),
        ]
        for deck in DECKS:
            benchmark.extend(("--deck", str(deck)))
        completed = subprocess.run(benchmark, cwd=ROOT, env=env)
        if completed.returncode != 0:
            raise RuntimeError(f"optimized MPS whole-game failed: rc={completed.returncode}")
        result = json.loads(output.read_text())
        report["optimized_mps_result"] = result
        old = (report.get("results") or {}).get("mps") or {}
        old_gps = float(old.get("games_per_s") or 0.0)
        new_gps = float(result.get("games_per_s") or 0.0)
        report["optimized_mps_speedup_vs_original"] = (
            new_gps / old_gps if old_gps > 0 else None
        )
        report["optimized_mps_runtime_valid"] = bool(
            int(result.get("games_completed") or 0) == 64
            and not result.get("errors")
            and all(
                bool(row.get("parity_passed"))
                for row in report["optimized_mps_leaf"]["rows"]
                if row.get("device") == "mps"
            )
        )
        cpu_gps = float(
            (report.get("stability_result") or {}).get("games_per_s") or 0.0
        )
        report["optimized_mps_production_eligible"] = bool(
            report["optimized_mps_runtime_valid"]
            and new_gps >= cpu_gps
            and (report.get("whole_game_parity") or {}).get("passed")
        )
        report["optimized_mps_blockers"] = []
        if cpu_gps > new_gps:
            report["optimized_mps_blockers"].append(
                f"{new_gps:.3f} GPS is {cpu_gps / max(new_gps, 1e-9):.2f}x "
                f"slower than the validated {cpu_gps:.3f} GPS CPU topology"
            )
        if not (report.get("whole_game_parity") or {}).get("passed"):
            report["optimized_mps_blockers"].append(
                "MPS and CPU policy trajectories are not bit-identical despite "
                "numeric leaf parity"
            )
        _stop_worker(child)
        child = None
        report["status"] = "complete"
        report["stage"] = "mps-optimized:complete"
        report["active_variant"] = None
        report["worker_pid"] = None
        report["mps_optimized_completed_at"] = time.time()
        _publish(report)
        print(
            f"[apple-mps-opt] complete gps={new_gps:.3f} "
            f"speedup={report['optimized_mps_speedup_vs_original']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["stage"] = "mps-optimized:failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_at"] = time.time()
        _publish(report)
        raise
    finally:
        if child is not None:
            _stop_worker(child)


if __name__ == "__main__":
    raise SystemExit(main())
