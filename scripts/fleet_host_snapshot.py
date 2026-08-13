#!/usr/bin/env python3
"""Emit cross-platform host, GPU, and simulator-worker telemetry as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


_FRAME_HEADER = struct.Struct("!I")
_MAX_HEALTH_FRAME = 8 * 1024 * 1024
APPLE_OPTIMIZATION_STATUS = Path(
    "/Users/tsinzitari/pokebot-dashboard/v1/m4_optimization_status.json"
)
APPLE_OPTIMIZATION_STATUS_ROOTS = (
    Path(
        "/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/"
        "outputs/benchmarks"
    ),
)
APPLE_GPU_POWERMETRICS = Path(
    "/Users/tsinzitari/pokebot-dashboard/v1/apple_gpu_powermetrics.json"
)


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


def _bytes_from_sized_token(value: str, unit: str | None) -> int | None:
    amount = number(value)
    if amount is None:
        return None
    scale = {
        None: 1,
        "": 1,
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }.get((unit or "").upper())
    if scale is None:
        return None
    return int(amount * scale)


def darwin_swap_state() -> tuple[int | None, int | None, int | None]:
    """Return (total, used, free) bytes from macOS vm.swapusage."""
    raw = run(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    if not raw:
        raw = run(["/usr/sbin/sysctl", "vm.swapusage"])
    matches = {
        key.lower(): _bytes_from_sized_token(amount, unit)
        for key, amount, unit in re.findall(
            r"(total|used|free)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]B?)?",
            raw,
            flags=re.IGNORECASE,
        )
    }
    total = matches.get("total")
    used = matches.get("used")
    free = matches.get("free")
    if used is None and total is not None and free is not None:
        used = max(0, total - free)
    if free is None and total is not None and used is not None:
        free = max(0, total - used)
    return total, used, free


def memory_state() -> dict[str, int | None]:
    if platform.system() == "Linux":
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    values[key] = int(value.strip().split()[0]) * 1024
        except (OSError, ValueError):
            pass
        swap_total = values.get("SwapTotal")
        swap_free = values.get("SwapFree")
        swap_used = (
            max(0, swap_total - swap_free)
            if swap_total is not None and swap_free is not None
            else None
        )
        return {
            "memory_total_bytes": values.get("MemTotal"),
            "memory_available_bytes": values.get("MemAvailable"),
            "swap_total_bytes": swap_total,
            "swap_used_bytes": swap_used,
            "swap_free_bytes": swap_free,
        }

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
    swap_total, swap_used, swap_free = darwin_swap_state()
    return {
        "memory_total_bytes": int(total) if total is not None else None,
        "memory_available_bytes": available_pages * page_size,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_used,
        "swap_free_bytes": swap_free,
    }


def _default_network_interfaces() -> list[str]:
    """Return the host interfaces carrying its default route."""

    if platform.system() == "Linux":
        binary = shutil.which("ip") or "/usr/sbin/ip"
        raw = run([binary, "route", "show", "default"])
        return sorted(set(re.findall(r"(?:^|\s)dev\s+(\S+)", raw)))
    raw = run(["/sbin/route", "-n", "get", "default"])
    return sorted(
        set(re.findall(r"(?m)^\s*interface:\s*(\S+)\s*$", raw))
    )


def _linux_network_counters(interfaces: set[str]) -> tuple[int, int]:
    received = 0
    transmitted = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, separator, counters = line.partition(":")
        if not separator or name.strip() not in interfaces:
            continue
        fields = counters.split()
        if len(fields) >= 9:
            received += int(fields[0])
            transmitted += int(fields[8])
    return received, transmitted


def _darwin_network_counters(interfaces: set[str]) -> tuple[int, int]:
    raw = run(["/usr/sbin/netstat", "-ibdn"], timeout=3)
    lines = raw.splitlines()
    if not lines:
        raise ValueError("netstat returned no interface counters")
    headings = lines[0].split()
    ibytes_index = headings.index("Ibytes")
    obytes_index = headings.index("Obytes")
    # macOS repeats one interface's cumulative counters for every bound
    # address. Retain the maximum per interface so those rows are not summed.
    per_interface: dict[str, tuple[int, int]] = {}
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0].rstrip("*") not in interfaces:
            continue
        if len(fields) <= max(ibytes_index, obytes_index):
            continue
        name = fields[0].rstrip("*")
        try:
            counters = (int(fields[ibytes_index]), int(fields[obytes_index]))
        except ValueError:
            continue
        previous = per_interface.get(name, (0, 0))
        per_interface[name] = (
            max(previous[0], counters[0]),
            max(previous[1], counters[1]),
        )
    return (
        sum(row[0] for row in per_interface.values()),
        sum(row[1] for row in per_interface.values()),
    )


def network_state() -> dict[str, Any]:
    """Read cumulative hardware-interface counters for central rate sampling."""

    sampled_at = time.time()
    interfaces = _default_network_interfaces()
    if not interfaces:
        return {
            "available": False,
            "interfaces": [],
            "sampled_at": sampled_at,
            "reason": "default network interface unavailable",
        }
    try:
        if platform.system() == "Linux":
            received, transmitted = _linux_network_counters(set(interfaces))
            source = "/proc/net/dev"
        else:
            received, transmitted = _darwin_network_counters(set(interfaces))
            source = "netstat -ibdn"
    except (OSError, ValueError, IndexError):
        return {
            "available": False,
            "interfaces": interfaces,
            "sampled_at": sampled_at,
            "reason": "network counters unavailable",
        }
    return {
        "available": True,
        "interfaces": interfaces,
        "rx_bytes": received,
        "tx_bytes": transmitted,
        "sampled_at": sampled_at,
        "counter_source": source,
    }


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
            power: dict[str, Any] = {}
            try:
                candidate = json.loads(APPLE_GPU_POWERMETRICS.read_text())
                if (
                    isinstance(candidate, dict)
                    and time.time() - float(candidate.get("sampled_at") or 0.0) < 8.0
                    and not candidate.get("error")
                ):
                    power = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            utilization = power.get("gpu_active_residency_percent")
            if utilization is None:
                utilization = number(stats.get("Device Utilization %"))
            return [
                {
                    "index": 0,
                    "name": f"{name} GPU",
                    "utilization": utilization,
                    "renderer_utilization": number(stats.get("Renderer Utilization %")),
                    "tiler_utilization": number(stats.get("Tiler Utilization %")),
                    "memory_used_mib": (number(stats.get("In use system memory")) or 0) / 1048576,
                    "memory_allocated_mib": (number(stats.get("Alloc system memory")) or 0) / 1048576,
                    "memory_total_mib": total / 1048576 if total is not None else None,
                    "power_w": (
                        float(power["gpu_power_mw"]) / 1000.0
                        if power.get("gpu_power_mw") is not None
                        else None
                    ),
                    "power_limit_w": None,
                    "temperature_c": None,
                    "unified_memory": True,
                    "active_frequency_mhz": power.get("gpu_active_frequency_mhz"),
                    "idle_residency_percent": power.get("gpu_idle_residency_percent"),
                    "telemetry_source": (
                        "powermetrics gpu_power + AGXAccelerator"
                        if power
                        else "AGXAccelerator"
                    ),
                }
            ]
        except (OSError, subprocess.TimeoutExpired, plistlib.InvalidFileException, StopIteration):
            return []
    compute_pids_by_uuid: dict[str, list[int]] = {}
    compute_raw = run(
        [
            binary,
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    for line in compute_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_uuid, pid_text = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        compute_pids_by_uuid.setdefault(gpu_uuid, []).append(pid)

    raw = run(
        [
            binary,
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    result: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        gpu_uuid = parts[1]
        result.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else None,
                "uuid": gpu_uuid,
                "name": parts[2],
                "utilization": number(parts[3]),
                "memory_used_mib": number(parts[4]),
                "memory_total_mib": number(parts[5]),
                "power_w": number(parts[6]),
                "power_limit_w": number(parts[7]),
                "temperature_c": number(parts[8]),
                "compute_process_pids": sorted(
                    set(compute_pids_by_uuid.get(gpu_uuid, []))
                ),
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


def _python_script(command: str, names: set[str]) -> bool:
    """Match a real Python script argv, never a shell/rg diagnostic command."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or not Path(tokens[0]).name.lower().startswith("python"):
        return False
    return any(Path(token).name in names for token in tokens[1:4])


def _tree_pids(rows: list[tuple[int, int, float, int, str]], roots: set[int]) -> set[int]:
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid, _cpu, _rss, _command in rows:
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def apple_optimization_state(
    rows: list[tuple[int, int, float, int, str]],
) -> dict[str, Any]:
    roots = {
        pid
        for pid, _ppid, _cpu, _rss, command in rows
        if _python_script(
            command,
            {
                "run_apple_optimization.py",
                "run_apple_cpu_scaling.py",
                "profile_apple_mps_pipeline.py",
                "run_apple_mps_optimized_validation.py",
                "run_apple_cpu_endurance.py",
            },
        )
    }
    selected = _tree_pids(rows, roots)
    status_candidates = [APPLE_OPTIMIZATION_STATUS]
    for root in APPLE_OPTIMIZATION_STATUS_ROOTS:
        try:
            status_candidates.extend(root.glob("*status.json"))
        except OSError:
            pass
    statuses: list[tuple[float, Path, dict[str, Any]]] = []
    for path in status_candidates:
        try:
            candidate = json.loads(path.read_text())
            if not isinstance(candidate, dict):
                continue
            sort_time = float(
                candidate.get("started_at")
                or candidate.get("updated_at")
                or path.stat().st_mtime
            )
            statuses.append((sort_time, path, candidate))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    # A running controller owns the newest running receipt. Completed and
    # failed historical sweeps remain available only when no optimization is
    # active, preventing an old dashboard-local status file from relabeling
    # today's isolated H10 benchmark.
    eligible = (
        [item for item in statuses if item[2].get("status") == "running"]
        if roots
        else statuses
    )
    selected_status = max(eligible or statuses, default=None, key=lambda item: item[0])
    status_path = selected_status[1] if selected_status else APPLE_OPTIMIZATION_STATUS
    status = selected_status[2] if selected_status else {}

    active_variant = str(status.get("active_variant") or "")
    endurance = status.get("apple_cpu_endurance") or {}
    rate = (
        {
            "games_per_s": endurance.get("games_per_s"),
            "decisions_per_s": endurance.get("decisions_per_s"),
        }
        if int(endurance.get("games_completed") or 0) > 0
        else status.get("stability_result") or {}
    )
    if not rate and active_variant:
        rate = (status.get("scaling_results") or {}).get(active_variant) or {}
    if not rate:
        recommended = status.get("recommended_cpu_topology") or {}
        rate = {
            "games_per_s": recommended.get("sweep_games_per_s"),
            "decisions_per_s": recommended.get("sweep_decisions_per_s"),
        }
    if not rate and active_variant:
        rate = (status.get("results") or {}).get(active_variant) or {}
    return {
        "active": bool(roots),
        "status": status.get("status"),
        "stage": status.get("stage"),
        "active_variant": status.get("active_variant"),
        "device": (
            "mps" if "mps" in active_variant.lower() else "cpu"
            if active_variant
            else None
        ),
        "controller_pids": sorted(roots),
        "processes": len(selected),
        "tree_cpu_percent": sum(
            cpu for pid, _ppid, cpu, _rss, _command in rows if pid in selected
        ),
        "tree_rss_bytes": sum(
            rss for pid, _ppid, _cpu, rss, _command in rows if pid in selected
        ) * 1024,
        "gps": rate.get("games_per_s"),
        "sps": rate.get("decisions_per_s"),
        "rate_source": (
            "Apple optimization active benchmark status"
            if status.get("status") == "running"
            else "Apple optimization latest completed topology"
        ),
        "status_source": str(status_path),
        "status_updated_at": status.get("updated_at"),
        "recommended_cpu_topology": status.get("recommended_cpu_topology"),
        "whole_game_parity": status.get("whole_game_parity"),
    }


def worker_state(rows: list[tuple[int, int, float, int, str]]) -> dict[str, Any]:
    roots = {
        pid
        for pid, _ppid, _cpu, _rss, command in rows
        if _python_script(command, {"run_remote_worker.py"})
    }
    selected = _tree_pids(rows, roots)
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
    memory = memory_state()
    network = network_state()
    loads = os.getloadavg()
    rows = process_rows()
    optimization = apple_optimization_state(rows)
    worker = worker_state(rows)
    # A completed Apple benchmark is historical metadata, not an active test.
    # Never let it relabel a live production worker on :8766 as TEST.
    if optimization.get("status") and (optimization.get("active") or not worker.get("active")):
        worker.update(
            testing=True,
            production_active=False,
            optimization_stage=optimization.get("stage"),
            optimization_variant=optimization.get("active_variant"),
            optimization_device=optimization.get("device"),
            gps=optimization.get("gps"),
            sps=optimization.get("sps"),
            rate_source=optimization.get("rate_source"),
            sps_estimated=False,
        )
    if optimization.get("active"):
        worker.update(
            active=True,
            testing=True,
            production_active=False,
            optimization_controller_pids=optimization.get("controller_pids"),
            optimization_processes=optimization.get("processes"),
            processes=optimization.get("processes"),
            cpu_percent=optimization.get("tree_cpu_percent"),
            rss_bytes=optimization.get("tree_rss_bytes"),
            gps=optimization.get("gps"),
            sps=optimization.get("sps"),
            rate_source=optimization.get("rate_source"),
            sps_estimated=False,
        )
    gpus = gpu_state()
    for gpu in gpus:
        if optimization.get("active"):
            gpu["assignment"] = (
                "Apple optimization · "
                f"{optimization.get('device') or 'CPU/MPS'} · "
                f"{optimization.get('stage') or 'running'}"
            )
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
                    "load_1m": loads[0],
                    "load_5m": loads[1],
                    "load_15m": loads[2],
                    "memory_total_bytes": memory.get("memory_total_bytes"),
                    "memory_available_bytes": memory.get("memory_available_bytes"),
                    "swap_total_bytes": memory.get("swap_total_bytes"),
                    "swap_used_bytes": memory.get("swap_used_bytes"),
                    "swap_free_bytes": memory.get("swap_free_bytes"),
                    "network": network,
                },
                "gpus": gpus,
                "worker": worker,
                "optimization": optimization,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
