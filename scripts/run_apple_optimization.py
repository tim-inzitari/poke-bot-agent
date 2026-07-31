#!/usr/bin/env python3
"""Run isolated Apple CPU-vs-MPS whole-game benchmarks on Bert.

This job never advertises itself to the production trainer. It starts a worker
on loopback port 8776, benchmarks identical seeded games with CPU and MPS policy
leaves, and publishes an atomic status report after every phase.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = Path(
    os.environ.get(
        "POKEBOT_APPLE_BENCHMARK_PYTHON",
        "/Users/tsinzitari/workspace/poke-bot-agent-deployments/"
        "safety-20260717-8a71861e9984/.venv/bin/python",
    )
)
CHECKPOINT = Path(
    os.environ.get(
        "POKEBOT_APPLE_BENCHMARK_CHECKPOINT",
        "/Users/tsinzitari/workspace/poke-bot-agent/outputs/pure_rl/"
        "pure_rl_core_exact20k_resident_v1_20260719/checkpoints/iter_00001.pt",
    )
)
CG_LIB = Path(
    "/Users/tsinzitari/workspace/poke-bot-agent/kaggle/input/"
    "pokemon-tcg-ai-battle/sample_submission/sample_submission"
)
DECKS = (
    Path("/Users/tsinzitari/workspace/poke-bot-agent/baselines/decks/ryota-alakazam-best5/deck.csv"),
    Path("/Users/tsinzitari/workspace/poke-bot-agent/baselines/decks/kokinn-lucario-search-915/deck.csv"),
)
STATUS = Path(
    os.environ.get(
        "POKEBOT_APPLE_BENCHMARK_STATUS",
        str(ROOT / "outputs" / "benchmarks" / "m4_optimization_status.json"),
    )
)
STATUS_MIRROR = Path(
    os.environ.get(
        "POKEBOT_APPLE_BENCHMARK_STATUS_MIRROR",
        "/Users/tsinzitari/pokebot-dashboard/v1/m4_optimization_status.json",
    )
)
LEAF_REPORT = ROOT / "outputs" / "benchmarks" / "m4_leaf_sweep.json"
PORT = int(os.environ.get("POKEBOT_APPLE_BENCHMARK_PORT", "8776"))
GAMES = int(os.environ.get("POKEBOT_APPLE_BENCHMARK_GAMES", "16"))
WORKERS = int(os.environ.get("POKEBOT_APPLE_BENCHMARK_WORKERS", "4"))
LEAF_SERVERS = int(os.environ.get("POKEBOT_APPLE_BENCHMARK_LEAF_SERVERS", "1"))
CONCURRENCY = int(os.environ.get("POKEBOT_APPLE_BENCHMARK_CONCURRENCY", str(WORKERS)))
TREE_RSS_LIMIT_GB = float(
    os.environ.get("POKEBOT_APPLE_BENCHMARK_TREE_RSS_LIMIT_GB", "18")
)
MIN_FREE_RAM_GB = float(
    os.environ.get("POKEBOT_APPLE_BENCHMARK_MIN_FREE_RAM_GB", "8")
)
LEAF_MAX_BATCH = int(
    os.environ.get("POKEBOT_APPLE_BENCHMARK_LEAF_MAX_BATCH", "32")
)
LEAF_COALESCE_MS = float(
    os.environ.get("POKEBOT_APPLE_BENCHMARK_LEAF_COALESCE_MS", "2")
)


def _publish(report: dict[str, Any]) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = time.time()
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)
    try:
        STATUS_MIRROR.parent.mkdir(parents=True, exist_ok=True)
        mirror_temporary = STATUS_MIRROR.with_suffix(STATUS_MIRROR.suffix + ".tmp")
        mirror_temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        mirror_temporary.replace(STATUS_MIRROR)
    except OSError:
        pass


def _wait_for_worker(proc: subprocess.Popen[Any], timeout_s: float = 180.0) -> dict:
    from poke_bot.remote_jobs import RemoteJobClient

    deadline = time.monotonic() + timeout_s
    last_error = "worker did not start"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"worker exited during startup: rc={proc.returncode}")
        client = RemoteJobClient("127.0.0.1", PORT, timeout_s=10.0)
        try:
            client.connect()
            health = client.health()
            client.close()
            if health.get("leaf_alive") and health.get("worker_capacity_healthy", True):
                return health
            last_error = f"worker not healthy: {health}"
        except Exception as exc:  # startup polling
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)
    raise TimeoutError(last_error)


def _stop_worker(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10.0)


def main() -> int:
    report: dict[str, Any] = {
        "schema": "poke_bot.apple_optimization/v1",
        "status": "running",
        "stage": "initializing",
        "started_at": time.time(),
        "production_active": False,
        "role": "inactive from production · Apple optimization testing",
        "worker_endpoint": f"127.0.0.1:{PORT}",
        "games_per_variant": GAMES,
        "workers": WORKERS,
        "leaf_servers": LEAF_SERVERS,
        "concurrency": CONCURRENCY,
        "tree_rss_limit_gb": TREE_RSS_LIMIT_GB,
        "min_free_ram_gb": MIN_FREE_RAM_GB,
        "mps_empty_cache_every_batches": os.environ.get(
            "POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES", "1"
        ),
        "mps_autocast_dtype": os.environ.get(
            "POKEBOT_MPS_AUTOCAST_DTYPE", "float32"
        ),
        "leaf_max_batch": LEAF_MAX_BATCH,
        "leaf_coalesce_ms": LEAF_COALESCE_MS,
        "results": {},
    }
    if LEAF_REPORT.is_file():
        leaf = json.loads(LEAF_REPORT.read_text())
        report["leaf_sweep"] = {
            "status": leaf.get("status"),
            "rows": len(leaf.get("rows") or []),
            "best_eligible": leaf.get("best_eligible"),
            "mps_available": leaf.get("mps_available"),
        }
    _publish(report)

    child: subprocess.Popen[Any] | None = None

    def _shutdown(_signum: int, _frame: object) -> None:
        if child is not None:
            _stop_worker(child)
        raise SystemExit(128 + int(_signum))

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    available_variants = (
        ("cpu-2t", "cpu", {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}),
        ("mps", "mps", {}),
    )
    requested_variants = {
        item.strip()
        for item in os.environ.get(
            "POKEBOT_APPLE_BENCHMARK_VARIANTS", "cpu-2t,mps"
        ).split(",")
        if item.strip()
    }
    variants = tuple(row for row in available_variants if row[0] in requested_variants)
    if not variants:
        raise ValueError("POKEBOT_APPLE_BENCHMARK_VARIANTS selected no variants")
    try:
        for name, device, tuning in variants:
            report["stage"] = f"starting-{name}-worker"
            report["active_variant"] = name
            _publish(report)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(ROOT),
                    "PYTHONUNBUFFERED": "1",
                    "CG_LIB_PATH": str(CG_LIB),
                    "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": "20260717",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.50",
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.40",
                    **tuning,
                }
            )
            worker_command = [
                str(PYTHON),
                str(ROOT / "scripts" / "run_remote_worker.py"),
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--workers", str(WORKERS),
                "--default-workers", str(WORKERS),
                "--leaf-servers", str(LEAF_SERVERS),
                "--leaf-gpu", device,
                "--leaf-max-batch", str(LEAF_MAX_BATCH),
                "--leaf-queue-depth", "32",
                "--leaf-coalesce-ms", str(LEAF_COALESCE_MS),
                "--max-connections", str(max(8, CONCURRENCY * 2)),
                "--tree-rss-limit-gb", str(TREE_RSS_LIMIT_GB),
                "--min-free-ram-gb", str(MIN_FREE_RAM_GB),
                "--checkpoint", str(CHECKPOINT),
                "--cg-lib-path", str(CG_LIB),
            ]
            print(f"[apple-opt] start variant={name} device={device}", flush=True)
            child = subprocess.Popen(worker_command, cwd=ROOT, env=env)
            report["worker_pid"] = child.pid
            _publish(report)
            health = _wait_for_worker(child)
            report["stage"] = f"benchmarking-{name}"
            report["worker_health"] = {
                "leaf_alive": health.get("leaf_alive"),
                "workers": health.get("workers"),
                "tree_rss_gb": health.get("tree_rss_gb"),
            }
            _publish(report)

            output_path = STATUS.parent / f"h10_whole_game_{GAMES}_{name}.json"
            benchmark_command = [
                str(PYTHON),
                str(ROOT / "scripts" / "benchmark_remote_model_gps.py"),
                "--endpoint", f"127.0.0.1:{PORT}",
                "--checkpoint", str(CHECKPOINT),
                "--games", str(GAMES),
                "--concurrency", str(CONCURRENCY),
                "--timeout", "600",
                "--seed", "946000",
                "--no-reload",
                "--json-out", str(output_path),
            ]
            for deck in DECKS:
                benchmark_command.extend(("--deck", str(deck)))
            completed = subprocess.run(benchmark_command, cwd=ROOT, env=env)
            if completed.returncode != 0:
                raise RuntimeError(f"{name} benchmark failed: rc={completed.returncode}")
            result = json.loads(output_path.read_text())
            report["results"][name] = result
            _publish(report)
            _stop_worker(child)
            child = None

        cpu = report["results"].get("cpu-2t")
        mps = report["results"].get("mps")
        matches = 0
        common: list[str] = []
        if cpu is not None and mps is not None:
            cpu_hashes = cpu.get("game_fingerprints") or {}
            mps_hashes = mps.get("game_fingerprints") or {}
            common = sorted(set(cpu_hashes) & set(mps_hashes), key=int)
            matches = sum(cpu_hashes[key] == mps_hashes[key] for key in common)
            report["whole_game_parity"] = {
                "compared_games": len(common),
                "matching_gameplay_fingerprints": matches,
                "passed": bool(common) and matches == len(common),
            }
            report["recommendation"] = (
                "cpu-2t"
                if float(cpu.get("games_per_s") or 0.0)
                >= float(mps.get("games_per_s") or 0.0)
                else "mps"
            )
        else:
            report["recommendation"] = "compare_with_existing_cpu_baseline"
        report["status"] = "complete"
        report["stage"] = "complete"
        report["active_variant"] = None
        report["worker_pid"] = None
        report["completed_at"] = time.time()
        _publish(report)
        print(
            "[apple-opt] complete "
            f"cpu_gps={cpu.get('games_per_s') if cpu else None} "
            f"mps_gps={mps.get('games_per_s') if mps else None} "
            f"parity={matches}/{len(common)} recommend={report['recommendation']}",
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
