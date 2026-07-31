"""Cheap host/GPU sampling (no torch import)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any


def sample_ram() -> dict[str, float]:
    meta: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                meta[key.strip()] = int(rest.strip().split()[0])
    except Exception:
        return {"total_gb": 0.0, "available_gb": 0.0, "used_gb": 0.0, "used_pct": 0.0}
    total = meta.get("MemTotal", 0) / 1e6
    avail = meta.get("MemAvailable", meta.get("MemFree", 0)) / 1e6
    used = max(0.0, total - avail)
    return {
        "total_gb": round(total, 1),
        "available_gb": round(avail, 1),
        "used_gb": round(used, 1),
        "used_pct": round((used / total * 100.0) if total else 0.0, 1),
    }


def sample_cpu(interval: float = 0.05) -> float:
    def _read() -> tuple[int, int]:
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            parts = fh.readline().split()[1:]
        vals = list(map(int, parts))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle

    try:
        t1, i1 = _read()
        time.sleep(interval)
        t2, i2 = _read()
        dt = t2 - t1
        di = i2 - i1
        return round((1.0 - di / dt) * 100.0, 1) if dt > 0 else 0.0
    except Exception:
        return 0.0


def sample_gpus() -> list[dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            used = float(parts[2]) / 1024.0
            total = float(parts[3]) / 1024.0
            util = float(parts[4])
            idx = int(parts[0])
        except ValueError:
            continue
        gpus.append(
            {
                "index": idx,
                "name": parts[1],
                "mem_used_gb": round(used, 2),
                "mem_total_gb": round(total, 2),
                "mem_pct": round((used / total * 100.0) if total else 0.0, 1),
                "util": util,
            }
        )
    return gpus


def cpu_count() -> int:
    return os.cpu_count() or 1
