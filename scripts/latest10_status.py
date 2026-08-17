#!/usr/bin/env python3
"""Report exact July 9–18 bootstrap preparation from files and systemd units."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _tail(path: Path, size: int = 160_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return ANSI_RE.sub("", handle.read().decode("utf-8", "replace")).replace(
                "\r", "\n"
            )
    except OSError:
        return ""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _unit_state(name: str) -> dict[str, Any]:
    raw = _run(
        [
            "systemctl",
            "show",
            name,
            "--property=ActiveState,SubState,MainPID,MemoryCurrent,MemoryPeak",
        ]
    )
    values: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    number = lambda key: int(values[key]) if values.get(key, "").isdigit() else None
    return {
        "active": values.get("ActiveState") == "active",
        "active_state": values.get("ActiveState", "not-found"),
        "sub_state": values.get("SubState", "dead"),
        "pid": number("MainPID"),
        "memory_bytes": number("MemoryCurrent"),
        "memory_peak_bytes": number("MemoryPeak"),
    }


def _find(root: Path, name: str) -> Path | None:
    candidates = [
        root / "data/bootstrap/latest10-20260709-20260718" / name,
        root / "data/bootstrap" / name,
    ]
    return next((path for path in candidates if path.is_file()), None)


def _last_progress_line(clean: str) -> str:
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    for line in reversed(lines):
        if "feature " in line or "top-ladder convert:" in line:
            return line
    return lines[-1] if lines else ""


def day_state(root: Path, host: str, day: str, expected_episodes: int) -> dict[str, Any]:
    suffix = day[-2:]
    archive = root / f"data/episodes/raw/pokemon-tcg-ai-battle-episodes-{day}.zip"
    jsonl = _find(root, f"top_ladder_all_{day}.jsonl")
    features = _find(root, f"top_ladder_all_{day}.features")
    sidecar = _find(root, f"top_ladder_all_{day}.features.json")
    prep_unit = _unit_state(f"pokemon-latest10-day-{suffix}.service")
    download_unit = _unit_state(f"pokemon-latest10-download-07{suffix}.service")
    log_path = root / f"outputs/logs/latest10-day-{suffix}.log"
    download_log = root / f"outputs/logs/latest10-download-07{suffix}.log"
    clean = _tail(log_path)
    latest = _last_progress_line(clean)
    metadata = _json(sidecar) if sidecar else {}
    stats = dict(metadata.get("stats") or {})
    expected_records = max(1, expected_episodes * 2)
    current: int | None = None
    total: int | None = None
    rate: float | None = None
    unit = ""
    stage = "waiting"
    percent = 0.0

    if features and sidecar and int(stats.get("records_kept", 0)) > 0:
        stage = "ready"
        percent = 100.0
        current = int(stats.get("records_kept", 0))
        total = current
        unit = "sequences"
    elif prep_unit["active"]:
        if "[stage feature]" in clean or latest.startswith("feature "):
            stage = "featurizing"
            matches = re.findall(
                r"feature\s+.*?:\s*(\d+)seq\s+\[[^\]]*?([0-9.]+)seq/s",
                clean,
            )
            if matches:
                current = int(matches[-1][0])
                total = expected_records
                rate = float(matches[-1][1])
            percent = 55.0 + 45.0 * min(1.0, (current or 0) / expected_records)
            unit = "sequences"
        else:
            stage = "converting"
            matches = re.findall(r"top-ladder convert:\s*\d+%.*?(\d+)/(\d+)", clean)
            if matches:
                current, total = map(int, matches[-1])
            percent = 10.0 + 45.0 * min(1.0, (current or 0) / max(1, total or expected_episodes))
            unit = "episodes"
    elif jsonl:
        stage = "converted"
        percent = 55.0
    elif download_unit["active"]:
        stage = "downloading"
        percent = 5.0
        latest = _last_progress_line(_tail(download_log)) or f"Downloading {day}"
    elif archive.is_file():
        stage = "downloaded"
        percent = 10.0

    service = prep_unit if prep_unit["active"] else download_unit
    return {
        "day": day,
        "host": host,
        "stage": stage,
        "percent": round(percent, 2),
        "current": current,
        "total": total,
        "unit": unit,
        "rate": rate,
        "latest_line": latest,
        "archive_bytes": archive.stat().st_size if archive.is_file() else None,
        "jsonl_bytes": jsonl.stat().st_size if jsonl else None,
        "feature_bytes": features.stat().st_size if features else None,
        "records_kept": stats.get("records_kept"),
        "decisions_kept": stats.get("decisions_kept"),
        "service": service,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = args.manifest or (
        args.root / "kaggle/input/pokemon-tcg-ai-battle-episodes-index/manifest.csv"
    )
    episode_counts: dict[str, int] = {}
    try:
        import csv

        for row in csv.DictReader(manifest.open(newline="", encoding="utf-8")):
            episode_counts[str(row["date"])] = int(row["episode_count"])
    except (OSError, KeyError, TypeError, ValueError):
        pass
    days = [f"2026-07-{value:02d}" for value in range(9, 19)]
    rows = [
        day_state(args.root, args.host, day, episode_counts.get(day, 5000))
        for day in days
    ]
    print(
        json.dumps(
            {
                "ok": True,
                "host": args.host,
                "observed_at": time.time(),
                "days": rows,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
