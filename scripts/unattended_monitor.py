#!/usr/bin/env python
"""Fail-safe monitor for one unattended round-robin process.

The monitor never deletes or rewrites run artifacts. It records health snapshots
and requests a graceful process-group stop when a trusted-path corruption signal
is observed.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from scripts.log_trimmer import (
        DEFAULT_KEEP_BYTES,
        DEFAULT_THRESHOLD_BYTES,
        MB,
        acquire_trim_lock,
        trim_lock_matches,
        trim_file_if_needed,
    )
except ModuleNotFoundError:  # direct ``python scripts/unattended_monitor.py``
    from log_trimmer import (  # type: ignore[no-redef]
        DEFAULT_KEEP_BYTES,
        DEFAULT_THRESHOLD_BYTES,
        MB,
        acquire_trim_lock,
        trim_lock_matches,
        trim_file_if_needed,
    )


FATAL_PATTERNS = {
    "fail_closed": re.compile(r"FAIL-CLOSED|fail[_ -]closed(?:_games)?=[1-9]", re.I),
    "zero_target": re.compile(r"zero[_ -]target(?:_games)?=[1-9]", re.I),
    "incomplete": re.compile(
        r"game (?:is )?incomplete|incomplete after|reached max_steps", re.I
    ),
    "dead_server": re.compile(r"leaf server.*(?:dead|died)|server is not alive", re.I),
    "digest_mismatch": re.compile(r"(?:checkpoint|reload).*digest mismatch", re.I),
    "non_finite": re.compile(r"\b(?:nan|inf)\b|non-finite|loss explosion", re.I),
    "missing_opponent": re.compile(r"expected (?:baseline|opponent) unavailable", re.I),
    "broken_pipe": re.compile(r"broken pipe|connection reset", re.I),
    "fatal_gate": re.compile(r"FATAL HEALTH GATE|ABORT:", re.I),
    "hidden_info": re.compile(r"hidden-state leakage|info-set violation", re.I),
    "stale_generation": re.compile(r"stale.*generation|generation.*mismatch", re.I),
    "insufficient_sims": re.compile(r"insufficient trusted belief simulations", re.I),
    "untrusted_target": re.compile(r"untrusted target|oracle.*cannot generate", re.I),
    "writer_failure": re.compile(
        r"writer (?:failed|committed .*?/|thread did not stop)|ordering gaps", re.I
    ),
    "large_action_failure": re.compile(r"ActionSpaceTooLarge", re.I),
    "response_slot_overflow": re.compile(
        r"remote response slot overflow", re.I
    ),
    "belief_support": re.compile(r"BeliefSupportError", re.I),
    "trust_failure": re.compile(
        r"trust(?:ed)?[_ -](?:search[_ -])?failures?=[1-9]", re.I
    ),
    "game_timeout": re.compile(
        r"game_timeouts?=[1-9]|game exceeded \d+s", re.I
    ),
}
OOM_PATTERN = re.compile(r"out of memory|CUDA OOM|OutOfMemory", re.I)
PROGRESS_PATTERN = re.compile(
    r"iter(?:ation)?[ =:]+\d+|games?[ =:]+\d+|core search games|core bc games|"
    r"core_deep_search|rl-train|PROMOTED|REJECTED|checkpoint",
    re.I,
)
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
GATE_PATTERN = re.compile(r"\b(?:PROMOTED|REJECTED)\b", re.I)
METRIC_PATTERN = re.compile(
    r"\b(?:wr=|draw_aware_lo=|wins?=|draws?=|losses?=|policy_loss=|value_loss=)",
    re.I,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--stall-minutes", type=float, default=20.0)
    p.add_argument("--oom-limit", type=int, default=2)
    p.add_argument("--report-minutes", type=float, default=5.0)
    p.add_argument(
        "--log-threshold-mb",
        type=float,
        default=_env_float(
            "POKEBOT_LOG_THRESHOLD_MB", DEFAULT_THRESHOLD_BYTES / MB
        ),
        help="Inode-preserving trim threshold (default/env: 256 MiB).",
    )
    p.add_argument(
        "--log-keep-mb",
        type=float,
        default=_env_float("POKEBOT_LOG_KEEP_MB", DEFAULT_KEEP_BYTES / MB),
        help="Newest log tail retained after trimming (default/env: 16 MiB).",
    )
    p.add_argument("--process-group", action="store_true")
    p.add_argument(
        "--start-at-end",
        action="store_true",
        help="Ignore pre-existing log content and monitor only new writes.",
    )
    p.add_argument(
        "--forbid-gpu-index",
        type=int,
        default=None,
        help="Request a safe stop if this process group allocates on the GPU.",
    )
    return p.parse_args(argv)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        if len(stat) > 2 and stat[2] == "Z":
            return False
        return True
    except (OSError, IndexError):
        return False


def _gpu_snapshot() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                }
            )
    return rows


def _gpu_compute_processes() -> list[dict[str, Any]]:
    try:
        gpu_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        uuid_to_index = {
            parts[1].strip(): int(parts[0].strip())
            for line in gpu_raw.splitlines()
            if len(parts := line.split(",")) == 2
        }
        process_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in process_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "gpu_index": uuid_to_index.get(parts[1]),
                    "memory_used_mb": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def _memory_snapshot() -> dict[str, float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    except Exception:
        return {}
    return {
        "available_gb": values.get("MemAvailable", 0) / 1024**2,
        "total_gb": values.get("MemTotal", 0) / 1024**2,
    }


def _process_snapshot(pid: int, process_group: bool) -> dict[str, Any]:
    """Return aggregate CPU/RSS for the monitored process or its POSIX group."""
    try:
        raw = subprocess.check_output(
            ["ps", "-eo", "pid=,pgid=,pcpu=,rss=,stat="],
            text=True,
            timeout=10,
        )
    except Exception:
        return {}
    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            row_pid, pgid = int(parts[0]), int(parts[1])
            if (process_group and pgid != pid) or (
                not process_group and row_pid != pid
            ):
                continue
            rows.append(
                {
                    "pid": row_pid,
                    "cpu_pct": float(parts[2]),
                    "rss_mb": int(parts[3]) / 1024,
                    "state": parts[4],
                }
            )
        except ValueError:
            continue
    return {
        "process_count": len(rows),
        "cpu_pct": sum(row["cpu_pct"] for row in rows),
        "rss_mb": sum(row["rss_mb"] for row in rows),
        "members": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _request_stop(pid: int, process_group: bool, reason: str) -> None:
    print(f"MONITOR_ALERT stop_requested reason={reason}", flush=True)
    try:
        if process_group:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def main() -> int:
    args = _args()
    threshold_bytes = int(args.log_threshold_mb * MB)
    keep_bytes = int(args.log_keep_mb * MB)
    if threshold_bytes <= 0 or keep_bytes <= 0 or keep_bytes >= threshold_bytes:
        print(
            "error: log threshold and retention must be positive, "
            "with retention below threshold",
            flush=True,
        )
        return 2
    trim_lock_fd = acquire_trim_lock(args.log)
    if trim_lock_fd is None:
        print(
            f"error: another truncation monitor owns {args.log}",
            flush=True,
        )
        return 3
    atexit.register(os.close, trim_lock_fd)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / "monitor_state.json"
    alert_path = args.run_dir / "MONITOR_STOP_REQUESTED.json"
    offset = (
        args.log.stat().st_size
        if args.start_at_end and args.log.exists()
        else 0
    )
    oom_count = 0
    last_progress = time.monotonic()
    last_size = -1
    stop_requested = False
    last_progress_line = ""
    last_gate_line = ""
    last_metric_line = ""
    last_report = 0.0
    trim_count = 0

    while True:
        now = time.time()
        alive = _alive(args.pid)
        try:
            size = args.log.stat().st_size
            log_age = now - args.log.stat().st_mtime
        except OSError:
            size = 0
            log_age = float("inf")
        if size < offset:
            offset = 0
        chunk = ""
        if size > offset:
            try:
                with args.log.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                    offset = fh.tell()
            except OSError:
                chunk = ""

        clean_lines = [
            ANSI_PATTERN.sub("", line).strip()
            for line in chunk.replace("\r", "\n").splitlines()
            if line.strip()
        ]
        progress_lines = [line for line in clean_lines if PROGRESS_PATTERN.search(line)]
        gate_lines = [line for line in clean_lines if GATE_PATTERN.search(line)]
        metric_lines = [line for line in clean_lines if METRIC_PATTERN.search(line)]
        if progress_lines:
            last_progress = time.monotonic()
            last_progress_line = progress_lines[-1]
        if gate_lines:
            last_gate_line = gate_lines[-1]
        if metric_lines:
            last_metric_line = metric_lines[-1]
        log_growth = max(0, size - max(last_size, 0))
        fatal = [
            name for name, pattern in FATAL_PATTERNS.items() if pattern.search(chunk)
        ]
        oom_count += len(OOM_PATTERN.findall(chunk))
        if oom_count >= args.oom_limit:
            fatal.append(f"repeated_oom_{oom_count}")
        stalled_for = time.monotonic() - last_progress
        if alive and stalled_for > args.stall_minutes * 60:
            fatal.append(f"no_progress_{stalled_for / 60:.1f}m")
        process_snapshot = _process_snapshot(args.pid, args.process_group)
        compute_processes = _gpu_compute_processes()
        if args.forbid_gpu_index is not None:
            group_pids = {
                int(row["pid"]) for row in process_snapshot.get("members", [])
            }
            offenders = [
                row
                for row in compute_processes
                if row.get("gpu_index") == args.forbid_gpu_index
                and int(row["pid"]) in group_pids
            ]
            if offenders:
                fatal.append(
                    f"forbidden_gpu_{args.forbid_gpu_index}_allocation"
                )

        if not trim_lock_matches(trim_lock_fd, args.log):
            print(
                f"error: monitored log changed inode; relinquishing {args.log}",
                flush=True,
            )
            return 3
        trimmed = trim_file_if_needed(
            args.log,
            threshold_bytes=threshold_bytes,
            keep_bytes=keep_bytes,
            log=lambda message: print(f"MONITOR_LOG_TRIM {message}", flush=True),
        )
        if trimmed:
            trim_count += 1
            try:
                size = args.log.stat().st_size
            except OSError:
                size = 0
            # The retained tail was already inspected before the trim. Resume
            # after it so old fatal text and OOMs are not counted twice.
            offset = size
        last_size = size

        disk = shutil.disk_usage(args.run_dir)
        snapshot = {
            "timestamp": now,
            "monitor_pid": os.getpid(),
            "pid": args.pid,
            "process_alive": alive,
            "stop_requested": stop_requested,
            "log": {
                "path": str(args.log),
                "bytes": size,
                "growth_bytes": log_growth,
                "age_seconds": log_age,
                "stalled_seconds": stalled_for,
                "trim_threshold_bytes": threshold_bytes,
                "trim_keep_bytes": keep_bytes,
                "trim_count": trim_count,
            },
            "progress": {
                "last_line": last_progress_line,
                "last_gate": last_gate_line,
                "last_metrics": last_metric_line,
            },
            "oom_count": oom_count,
            "fatal_signals": fatal,
            "process": process_snapshot,
            "gpu": _gpu_snapshot(),
            "gpu_compute_processes": compute_processes,
            "memory": _memory_snapshot(),
            "disk": {
                "free_gb": disk.free / 1024**3,
                "used_gb": disk.used / 1024**3,
            },
        }
        _write_json(state_path, snapshot)
        monotonic_now = time.monotonic()
        if monotonic_now - last_report >= max(60.0, args.report_minutes * 60):
            gpu_summary = ",".join(
                f"gpu{row['index']}={row['utilization_pct']}%/"
                f"{row['memory_used_mb']}MiB"
                for row in snapshot["gpu"]
            )
            process = snapshot["process"]
            progress = last_progress_line[-180:] if last_progress_line else "starting"
            print(
                "MONITOR_HEALTH "
                f"alive={alive} processes={process.get('process_count', 0)} "
                f"cpu={process.get('cpu_pct', 0):.1f}% "
                f"rss={process.get('rss_mb', 0):.0f}MiB "
                f"log={size}B stall={stalled_for:.0f}s oom={oom_count} "
                f"{gpu_summary} progress={progress}",
                flush=True,
            )
            last_report = monotonic_now

        if fatal and alive and not stop_requested:
            stop_requested = True
            reason = ",".join(sorted(set(fatal)))
            _write_json(
                alert_path,
                {"timestamp": now, "pid": args.pid, "reason": reason},
            )
            _request_stop(args.pid, args.process_group, reason)
        if not alive:
            print(
                f"MONITOR_EXIT process_alive=false stop_requested={stop_requested}",
                flush=True,
            )
            return 2 if stop_requested else 0
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
