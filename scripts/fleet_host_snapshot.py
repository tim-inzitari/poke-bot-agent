#!/usr/bin/env python3
"""Emit cross-platform host, GPU, and simulator-worker telemetry as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import re
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


_FRAME_HEADER = struct.Struct("!I")
_MAX_HEALTH_FRAME = 8 * 1024 * 1024


def run(argv: list[str], timeout: float = 2.0) -> str:
    try:
        return subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def memory_state() -> tuple[int | None, int | None]:
    if platform.system() == "Linux":
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    values[key] = int(value.strip().split()[0]) * 1024
        except (OSError, ValueError):
            pass
        return values.get("MemTotal"), values.get("MemAvailable")

    total = number(run(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
    raw = run(["/usr/bin/vm_stat"])
    page_match = re.search(r"page size of (\d+) bytes", raw)
    page_size = int(page_match.group(1)) if page_match else 4096
    pages: dict[str, int] = {}
    for line in raw.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        try:
            pages[key.strip()] = int(value.strip().rstrip(".").replace(".", ""))
        except ValueError:
            pass
    available_pages = sum(
        pages.get(key, 0)
        for key in (
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        )
    )
    return int(total) if total is not None else None, available_pages * page_size


def cpu_utilization_percent(sample_seconds: float = 0.15) -> float | None:
    """Return host-wide busy CPU over a short interval, not process lifetime CPU."""
    if platform.system() != "Linux":
        return None

    def sample() -> tuple[int, int] | None:
        try:
            fields = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        except (OSError, ValueError, IndexError):
            return None
        if len(fields) < 5:
            return None
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return sum(fields), idle

    first = sample()
    if first is None:
        return None
    time.sleep(max(0.05, float(sample_seconds)))
    second = sample()
    if second is None:
        return None
    total_delta = second[0] - first[0]
    idle_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))


def gpu_state() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        if platform.system() != "Darwin":
            return []
        try:
            payload = subprocess.run(
                [
                    "/usr/sbin/ioreg",
                    "-r",
                    "-c",
                    "AGXAccelerator",
                    "-d",
                    "1",
                    "-k",
                    "PerformanceStatistics",
                    "-a",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).stdout
            objects = plistlib.loads(payload)
            stats = next(
                obj.get("PerformanceStatistics")
                for obj in objects
                if isinstance(obj, dict) and isinstance(obj.get("PerformanceStatistics"), dict)
            )
            total = number(run(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
            name = run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
            return [
                {
                    "index": 0,
                    "name": f"{name} GPU",
                    "utilization": number(stats.get("Device Utilization %")),
                    "renderer_utilization": number(stats.get("Renderer Utilization %")),
                    "tiler_utilization": number(stats.get("Tiler Utilization %")),
                    "memory_used_mib": (number(stats.get("In use system memory")) or 0) / 1048576,
                    "memory_allocated_mib": (number(stats.get("Alloc system memory")) or 0) / 1048576,
                    "memory_total_mib": total / 1048576 if total is not None else None,
                    "power_w": None,
                    "power_limit_w": None,
                    "temperature_c": None,
                    "unified_memory": True,
                    "telemetry_source": "AGXAccelerator",
                }
            ]
        except (OSError, subprocess.TimeoutExpired, plistlib.InvalidFileException, StopIteration):
            return []
    raw = run(
        [
            binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    result: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        result.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else None,
                "name": parts[1],
                "utilization": number(parts[2]),
                "memory_used_mib": number(parts[3]),
                "memory_total_mib": number(parts[4]),
                "power_w": number(parts[5]),
                "power_limit_w": number(parts[6]),
                "temperature_c": number(parts[7]),
            }
        )
    return result


def process_rows() -> list[tuple[int, int, float, int, str]]:
    if platform.system() == "Darwin":
        raw = run(["/bin/ps", "-axo", "pid=,ppid=,pcpu=,rss=,command="], timeout=3)
    else:
        raw = run(["/bin/ps", "-eo", "pid=,ppid=,pcpu=,rss=,args="], timeout=3)
    rows: list[tuple[int, int, float, int, str]] = []
    for line in raw.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3]), parts[4]))
        except ValueError:
            pass
    return rows


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise OSError("remote worker closed health connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(_FRAME_HEADER.pack(len(body)) + body)


def _read_frame(sock: socket.socket) -> dict[str, Any]:
    (size,) = _FRAME_HEADER.unpack(_recv_exact(sock, _FRAME_HEADER.size))
    if size > _MAX_HEALTH_FRAME:
        raise OSError(f"health frame is unexpectedly large: {size}")
    value = json.loads(_recv_exact(sock, size).decode("utf-8"))
    return value if isinstance(value, dict) else {}


def remote_worker_health(port: int) -> dict[str, Any]:
    """Read monotonic host counters without importing the training package."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0) as sock:
            sock.settimeout(1.5)
            _send_frame(sock, {"type": "hello", "proto": 1, "client": "dashboard"})
            hello = _read_frame(sock)
            if hello.get("type") != "hello_ok":
                return {}
            _send_frame(sock, {"type": "health"})
            health = _read_frame(sock)
            try:
                _send_frame(sock, {"type": "bye"})
            except OSError:
                pass
            return health if health.get("type") == "health_ok" else hello
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return {}


def worker_state() -> dict[str, Any]:
    rows = process_rows()
    roots = {
        pid
        for pid, _ppid, _cpu, _rss, command in rows
        if "run_remote_worker.py" in command and "fleet_host_snapshot.py" not in command
    }
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid, _cpu, _rss, _command in rows:
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    commands = [command for pid, _ppid, _cpu, _rss, command in rows if pid in roots]
    command = commands[0] if commands else ""
    workers_match = re.search(r"--workers\s+(\d+)", command)
    leaf_match = re.search(r"--leaf-servers\s+(\d+)", command)
    port_match = re.search(r"--port\s+(\d+)", command)
    port = int(port_match.group(1)) if port_match else 8765
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            listening = True
    except OSError:
        listening = False
    health = remote_worker_health(port) if listening else {}
    return {
        "active": bool(roots) and listening,
        "listening": listening,
        "controller_pids": sorted(roots),
        "processes": len(selected),
        "workers": (
            int(health.get("workers"))
            if health.get("workers") is not None
            else (int(workers_match.group(1)) if workers_match else None)
        ),
        "max_workers": (
            int(health.get("max_workers"))
            if health.get("max_workers") is not None
            else None
        ),
        "leaf_servers": (
            int(health.get("leaf_servers"))
            if health.get("leaf_servers") is not None
            else (int(leaf_match.group(1)) if leaf_match else None)
        ),
        "port": port,
        "cpu_percent": sum(cpu for pid, _ppid, cpu, _rss, _command in rows if pid in selected),
        "rss_bytes": sum(rss for pid, _ppid, _cpu, rss, _command in rows if pid in selected) * 1024,
        "command": command,
        "jobs_completed": health.get("jobs_completed"),
        "jobs_failed": health.get("jobs_failed"),
        "decisions_completed": health.get("decisions_completed"),
        "trajectories_completed": health.get("trajectories_completed"),
        "active_jobs": health.get("active_jobs"),
        "uptime_s": health.get("uptime_s"),
        "counter_sampled_at": time.time() if health else None,
        "health_current": bool(health),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="worker")
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    total, available = memory_state()
    loads = os.getloadavg()
    cpu_percent = cpu_utilization_percent()
    print(
        json.dumps(
            {
                "reachable": True,
                "observed_at": time.time(),
                "name": args.name or socket.gethostname(),
                "role": args.role,
                "platform": platform.system().lower(),
                "system": {
                    "cpu_count": os.cpu_count(),
                    "cpu_utilization_percent": cpu_percent,
                    "load_1m": loads[0],
                    "load_5m": loads[1],
                    "load_15m": loads[2],
                    "memory_total_bytes": total,
                    "memory_available_bytes": available,
                },
                "gpus": gpu_state(),
                "worker": worker_state(),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
