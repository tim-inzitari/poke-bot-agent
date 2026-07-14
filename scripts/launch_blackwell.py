#!/usr/bin/env python3
"""Launch one Blackwell round-robin run with a bounded stable console log.

Checkpoint, replay, and run metadata retain a unique ``--run-name``. Console
output always goes directly to ``outputs/logs/blackwell.log`` (or ``--log``),
which is truncated at launch and bounded by the accompanying health monitor.
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
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "outputs/logs/blackwell.log"
DEFAULT_THRESHOLD_MB = 256.0
DEFAULT_KEEP_MB = 16.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"blackwell_{stamp}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(os.environ.get("POKEBOT_BLACKWELL_LOG", DEFAULT_LOG)),
        help="Stable console path (default/env POKEBOT_BLACKWELL_LOG: "
        "outputs/logs/blackwell.log).",
    )
    parser.add_argument(
        "--log-threshold-mb",
        type=float,
        default=_env_float("POKEBOT_LOG_THRESHOLD_MB", DEFAULT_THRESHOLD_MB),
        help="Trim threshold (default/env POKEBOT_LOG_THRESHOLD_MB: 256).",
    )
    parser.add_argument(
        "--log-keep-mb",
        type=float,
        default=_env_float("POKEBOT_LOG_KEEP_MB", DEFAULT_KEEP_MB),
        help="Newest MiB retained (default/env POKEBOT_LOG_KEEP_MB: 16).",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=_env_float("POKEBOT_LOG_TRIM_INTERVAL", 30.0),
    )
    parser.add_argument("--stall-minutes", type=float, default=20.0)
    parser.add_argument("--oom-limit", type=int, default=2)
    parser.add_argument("--report-minutes", type=float, default=5.0)
    parser.add_argument(
        "--preflight-profile",
        choices=("canary", "quick", "none"),
        default="canary",
        help="Run quick→canary before launch by default; full remains manual.",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("POKEBOT_PYTHON", sys.executable),
        help="Python executable for training and monitoring.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to train_round_robin.py.",
    )
    args = parser.parse_args(argv)
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    if any(
        arg == "--run-name" or arg.startswith("--run-name=")
        for arg in args.train_args
    ):
        parser.error("pass --run-name to launch_blackwell.py, before '--'")
    args.run_name = args.run_name or _default_run_name()
    if (
        args.log_threshold_mb <= 0
        or args.log_keep_mb <= 0
        or args.log_keep_mb >= args.log_threshold_mb
    ):
        parser.error("log retention must be positive and below the threshold")
    if args.monitor_interval <= 0:
        parser.error("monitor interval must be positive")
    return args


def _absolute_from_root(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def open_stable_log(path: Path) -> BinaryIO:
    """Return an O_APPEND stream freshly truncated at a real stable path.

    A symlink used to expose an already-running legacy log is removed without
    touching its target. This makes every future launch write to the stable
    path's own inode rather than extending a timestamped historical log.
    """
    path = _absolute_from_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND,
        0o644,
    )
    return os.fdopen(fd, "wb", buffering=0)


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    log = _absolute_from_root(args.log)
    run_dir = ROOT / "outputs/runs" / args.run_name
    train = [
        args.python,
        "-u",
        str(ROOT / "scripts/train_round_robin.py"),
        "--run-name",
        args.run_name,
        *args.train_args,
    ]
    monitor = [
        args.python,
        "-u",
        str(ROOT / "scripts/unattended_monitor.py"),
        "--pid",
        "{TRAIN_PID}",
        "--log",
        str(log),
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
        "--forbid-gpu-index",
        "0",
    ]
    return train, monitor


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    train_command, monitor_template = build_commands(args)
    log_path = _absolute_from_root(args.log)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env.setdefault("POKEBOT_WORKER_CPU_ONLY", "1")
    env.setdefault("POKEBOT_ALLOW_ORACLE_DECK", "0")
    # Scope B (Hammer Blackwell strategy heads): deploy/search gate + profile.
    # Core/3080ti launches must not set these. Training loss weights are set by
    # train_round_robin CLI defaults for hammer specialists.
    env.setdefault("POKEBOT_GPU_PROFILE", "blackwell")
    env.setdefault("POKEBOT_BLACKWELL_STRATEGY_HEADS", "1")
    env.setdefault("POKEBOT_PRIMARY_ARCHETYPE", "hammer-pult")

    if args.preflight_profile != "none":
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
                f"error: {args.preflight_profile} preflight failed; "
                "training was not started",
                file=sys.stderr,
            )
            return preflight.returncode

    log_stream = open_stable_log(log_path)

    print(f"BLACKWELL_RUN name={args.run_name} log={log_path}", flush=True)
    training = subprocess.Popen(
        train_command,
        cwd=ROOT,
        env=env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor_command = [
        str(training.pid) if value == "{TRAIN_PID}" else value
        for value in monitor_template
    ]
    monitor = subprocess.Popen(
        monitor_command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    print(
        f"BLACKWELL_PIDS training={training.pid} monitor={monitor.pid}",
        flush=True,
    )

    def request_stop(_signum: int, _frame: object) -> None:
        _terminate_process_group(training)
        if monitor.poll() is None:
            monitor.terminate()

    previous_handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Catch a duplicate-owner/configuration failure before settling into a
        # long run. The monitor itself holds the log's advisory trim lock.
        time.sleep(0.1)
        if monitor.poll() is not None:
            _terminate_process_group(training)
            training.wait()
            print(
                f"error: log monitor exited early with {monitor.returncode}",
                file=sys.stderr,
            )
            return monitor.returncode or 1
        return training.wait()
    finally:
        log_stream.close()
        if monitor.poll() is None:
            try:
                monitor.wait(timeout=max(15.0, args.monitor_interval * 2))
            except subprocess.TimeoutExpired:
                monitor.terminate()
                monitor.wait()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
