#!/usr/bin/env python3
"""Launch one full-hardware pure-RL trainee (core or specialist) + monitor.

Saturates CPU workers and dual-GPU leaf servers for a single active lineage.
Refuses to start when two GPUs are visible but leaf replicas omit GPU0 or GPU1
unless ``PURE_RL_ALLOW_SINGLE_GPU=1``.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "outputs/logs/pure_rl.log"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--mode", choices=("core", "specialist"), default="core")
    p.add_argument("--log", type=Path, default=Path(os.environ.get("POKEBOT_PURE_RL_LOG", DEFAULT_LOG)))
    p.add_argument("--python", default=os.environ.get("POKEBOT_PYTHON", sys.executable))
    p.add_argument("--preflight-profile", choices=("canary", "quick", "none"), default="quick")
    p.add_argument("--stall-minutes", type=float, default=20.0)
    p.add_argument("--oom-limit", type=int, default=2)
    p.add_argument("--report-minutes", type=float, default=5.0)
    p.add_argument("--monitor-interval", type=float, default=30.0)
    p.add_argument("--log-threshold-mb", type=float, default=_env_float("POKEBOT_LOG_THRESHOLD_MB", 256.0))
    p.add_argument("--log-keep-mb", type=float, default=_env_float("POKEBOT_LOG_KEEP_MB", 16.0))
    p.add_argument("--allow-single-gpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--multi-env-per-worker",
        type=int,
        default=None,
        help=(
            "Forward to train_pure_rl: LibcgMultiEnv battles per OS worker. "
            "Also honour POKEBOT_MULTI_ENV=1 in the child env."
        ),
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=None,
        help="Forward to train_pure_rl (default via PURE_RL_LEAF_COALESCE_MS=0).",
    )
    p.add_argument(
        "--remote-worker-endpoints",
        default=None,
        help=(
            "Whole-game farms (comma-separated). Default production: "
            "192.168.1.143:8765,bert.local:8766. Empty string disables."
        ),
    )
    p.add_argument(
        "--no-remote-workers",
        action="store_true",
        help="Disable Elmo/bert whole-game farms",
    )
    p.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Args after '--' forwarded to train_pure_rl.py",
    )
    args = p.parse_args(argv)
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    if args.run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = f"pure_rl_{args.mode}_{stamp}"
    return args


def open_stable_log(path: Path):
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o644)
    return os.fdopen(fd, "wb", buffering=0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from poke_bot.pure_rl.hardware import full_hardware_profile
    from dataclasses import replace

    hw = full_hardware_profile()
    if args.allow_single_gpu or args.smoke:
        hw = replace(hw, allow_single_gpu=True)
    try:
        import torch

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        visible = 0
    if args.smoke:
        visible = max(visible, 1)
    try:
        hw.validate_or_raise(visible_gpu_count=visible if visible else (1 if hw.allow_single_gpu else 0))
    except ValueError as exc:
        # If no CUDA in this environment, require explicit smoke/single-gpu.
        if visible < 2 and not hw.allow_single_gpu:
            print(
                f"error: {exc}; pass --allow-single-gpu or --smoke on non-dual-GPU hosts",
                file=sys.stderr,
            )
            return 2
        if not hw.allow_single_gpu:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Both GPUs visible so leaf servers can bind 0 and 1; train pins device 1.
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    env["POKEBOT_WORKER_CPU_ONLY"] = "1"
    env["PURE_RL_SIM_WORKERS"] = str(hw.sim_workers)
    env["PURE_RL_LEAF_GPU0_REPLICAS"] = str(hw.leaf_gpu0_replicas)
    env["PURE_RL_LEAF_GPU1_REPLICAS"] = str(hw.leaf_gpu1_replicas)
    env["PURE_RL_TORCH_THREADS"] = str(hw.torch_threads)
    # Tiny ~1.6M pure-RL policy: coalesce≈0 beats the RR Hope-large default (4ms).
    # Do not set LEAF_SERVER_COALESCE_MS globally here if already exported (ops override).
    env.setdefault("PURE_RL_LEAF_COALESCE_MS", "0")
    if hw.allow_single_gpu:
        env["PURE_RL_ALLOW_SINGLE_GPU"] = "1"

    if args.preflight_profile != "none" and not args.smoke:
        preflight = subprocess.run(
            [
                args.python,
                str(ROOT / "scripts/run_test_profile.py"),
                args.preflight_profile,
                "--python",
                args.python,
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if preflight.returncode != 0:
            print(
                f"error: {args.preflight_profile} preflight failed",
                file=sys.stderr,
            )
            return preflight.returncode

    train_cmd = [
        args.python,
        "-u",
        str(ROOT / "scripts/train_pure_rl.py"),
        "--run-name",
        args.run_name,
        "--mode",
        args.mode,
        *args.train_args,
    ]
    if args.smoke and "--smoke" not in train_cmd:
        train_cmd.append("--smoke")
    if args.allow_single_gpu and "--allow-single-gpu" not in train_cmd:
        train_cmd.append("--allow-single-gpu")
    if args.multi_env_per_worker is not None and not any(
        a == "--multi-env-per-worker" or a.startswith("--multi-env-per-worker=")
        for a in train_cmd
    ):
        train_cmd.extend(["--multi-env-per-worker", str(args.multi_env_per_worker)])
    if args.leaf_coalesce_ms is not None and not any(
        a == "--leaf-coalesce-ms" or a.startswith("--leaf-coalesce-ms=")
        for a in train_cmd
    ):
        train_cmd.extend(["--leaf-coalesce-ms", str(args.leaf_coalesce_ms)])
    # Production: remotes ON by default (canary/smoke skips).
    has_remote_flag = any(
        a == "--remote-worker-endpoints" or a.startswith("--remote-worker-endpoints=")
        for a in train_cmd
    )
    if args.no_remote_workers and "--no-remote-workers" not in train_cmd:
        train_cmd.append("--no-remote-workers")
    elif (
        not args.smoke
        and not args.no_remote_workers
        and not has_remote_flag
    ):
        endpoints = args.remote_worker_endpoints
        if endpoints is None:
            endpoints = os.environ.get(
                "PURE_RL_REMOTE_WORKER_ENDPOINTS",
                os.environ.get(
                    "POKEBOT_REMOTE_WORKER_ENDPOINTS",
                    "192.168.1.143:8765,bert.local:8766",
                ),
            )
        if str(endpoints).strip():
            train_cmd.extend(["--remote-worker-endpoints", str(endpoints)])

    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    log_stream = open_stable_log(log_path)
    print(
        f"PURE_RL_RUN name={args.run_name} mode={args.mode} "
        f"workers={hw.sim_workers} leaves0={hw.leaf_gpu0_replicas} "
        f"leaves1={hw.leaf_gpu1_replicas} log={log_path}",
        flush=True,
    )
    training = subprocess.Popen(
        train_cmd,
        cwd=ROOT,
        env=env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    monitor = None
    monitor_script = ROOT / "scripts/unattended_monitor.py"
    if monitor_script.is_file() and not args.smoke:
        run_dir = ROOT / "outputs/pure_rl" / args.run_name
        monitor_cmd = [
            args.python,
            "-u",
            str(monitor_script),
            "--pid",
            str(training.pid),
            "--log",
            str(log_path),
            "--run-dir",
            str(run_dir),
            "--interval",
            str(args.monitor_interval),
            "--stall-minutes",
            str(args.stall_minutes),
            "--oom-limit",
            str(args.oom_limit),
            "--report-minutes",
            str(args.report_minutes),
            "--log-threshold-mb",
            str(args.log_threshold_mb),
            "--log-keep-mb",
            str(args.log_keep_mb),
            "--process-group",
        ]
        monitor = subprocess.Popen(monitor_cmd, cwd=ROOT, env=env, start_new_session=True)
        print(f"PURE_RL_PIDS training={training.pid} monitor={monitor.pid}", flush=True)
    else:
        print(f"PURE_RL_PIDS training={training.pid}", flush=True)

    def _stop(_signum: int, _frame: object) -> None:
        try:
            os.killpg(training.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if monitor and monitor.poll() is None:
            monitor.terminate()

    prev = {sig: signal.signal(sig, _stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return training.wait()
    finally:
        log_stream.close()
        if monitor and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                monitor.kill()
        for sig, handler in prev.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
