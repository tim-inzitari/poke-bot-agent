#!/usr/bin/env python3
"""Launch the isolated GPU0 core pipeline after quick/canary preflight."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.launch_blackwell import open_stable_log


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--log", type=Path, default=ROOT / "outputs/logs/core_kernel.log"
    )
    parser.add_argument(
        "--preflight-profile",
        choices=("canary", "quick", "none"),
        default="canary",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--monitor-interval", type=float, default=30.0)
    parser.add_argument("--stall-minutes", type=float, default=20.0)
    parser.add_argument("--report-minutes", type=float, default=5.0)
    parser.add_argument("--log-threshold-mb", type=float, default=256.0)
    parser.add_argument("--log-keep-mb", type=float, default=16.0)
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.pipeline_args[:1] == ["--"]:
        args.pipeline_args = args.pipeline_args[1:]
    if any(
        value == "--run-name" or value.startswith("--run-name=")
        for value in args.pipeline_args
    ):
        parser.error("pass --run-name before '--'")
    return args


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "POKEBOT_GPU_PROFILE": "3080ti",
            "POKEBOT_PRIMARY_ARCHETYPE": "core-canonical",
            "POKEBOT_WORKER_CPU_ONLY": "1",
            "POKEBOT_ALLOW_ORACLE_DECK": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Opening time cut (0.5x) + min_trusted=128 caused Core RemoteLeafTimeout storms
    # (deadline leftovers <50ms). Keep full move budget unless operator overrides.
    # Config reads POKEBOT_<NAME> via _env_float — unprefixed OPENING_MOVE_TIME_MULT is ignored.
    env.setdefault("POKEBOT_OPENING_MOVE_TIME_MULT", "1.0")
    # Core = driver only; Elmo+Bert run nearly all sims. Never dump onto local CPU.
    env.setdefault("REMOTE_REQUEST_TIMEOUT_S", "120")
    env.setdefault("POKEBOT_REMOTE_CONNECT_TIMEOUT_S", "60")
    env.setdefault("POKEBOT_REMOTE_CONTROL_TIMEOUT_S", "300")
    env.setdefault("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", "600")
    env.setdefault("POKEBOT_REMOTE_PRIMARY", "1")
    env.setdefault("POKEBOT_REMOTE_ONLY", "1")
    env.setdefault("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "1")
    env.setdefault("POKEBOT_REMOTE_JOB_RETRIES", "8")
    if args.preflight_profile != "none":
        completed = subprocess.run(
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
        if completed.returncode != 0:
            print(
                f"error: {args.preflight_profile} preflight failed; "
                "core pipeline was not started",
                file=sys.stderr,
            )
            return completed.returncode

    log_path = args.log.expanduser()
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    stream = open_stable_log(log_path)
    train_command = [
        args.python,
        "-u",
        str(ROOT / "scripts/train_core_pipeline.py"),
        "--run-name",
        args.run_name,
        *args.pipeline_args,
    ]
    trainer = subprocess.Popen(
        train_command,
        cwd=ROOT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    run_dir = ROOT / "outputs/runs" / args.run_name
    monitor = subprocess.Popen(
        [
            args.python,
            "-u",
            str(ROOT / "scripts/unattended_monitor.py"),
            "--pid",
            str(trainer.pid),
            "--log",
            str(log_path),
            "--run-dir",
            str(run_dir),
            "--interval",
            str(args.monitor_interval),
            "--stall-minutes",
            str(args.stall_minutes),
            "--report-minutes",
            str(args.report_minutes),
            "--log-threshold-mb",
            str(args.log_threshold_mb),
            "--log-keep-mb",
            str(args.log_keep_mb),
            "--process-group",
            "--forbid-gpu-index",
            "1",
        ],
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    print(
        f"CORE_PIPELINE_PIDS training={trainer.pid} monitor={monitor.pid} "
        f"log={log_path}",
        flush=True,
    )

    def request_stop(_signum: int, _frame: object) -> None:
        _stop_group(trainer)
        if monitor.poll() is None:
            monitor.terminate()

    handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        time.sleep(0.1)
        if monitor.poll() is not None:
            _stop_group(trainer)
            trainer.wait()
            return monitor.returncode or 1
        return trainer.wait()
    finally:
        stream.close()
        if monitor.poll() is None:
            try:
                monitor.wait(timeout=max(15.0, args.monitor_interval * 2))
            except subprocess.TimeoutExpired:
                monitor.terminate()
                monitor.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
