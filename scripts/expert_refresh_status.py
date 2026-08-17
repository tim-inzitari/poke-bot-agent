#!/usr/bin/env python3
"""Emit compact dashboard telemetry for the versioned expert replay refresh."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DOWNLOAD_RE = re.compile(
    r"(?P<pct>\d+)%\|[^\r\n]*?\|\s*"
    r"(?P<current>[0-9.]+)(?P<current_unit>[kMGT]?)\/"
    r"(?P<total>[0-9.]+)(?P<total_unit>[kMGT]?)\s*"
    r"\[[^\]]*?,\s*(?P<rate>[0-9.]+)(?P<rate_unit>[kMGT]?)B\/s\]"
)
UNIT = {"": 1.0, "k": 1024.0, "M": 1024.0**2, "G": 1024.0**3, "T": 1024.0**4}


def _tail(path: Path, size: int = 1_000_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _tsv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("\t")
            if separator:
                values[key] = value
    except OSError:
        pass
    return values


def _unit_state(name: str) -> dict[str, Any]:
    try:
        raw = subprocess.run(
            [
                "systemctl",
                "show",
                name,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "MainPID",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        raw = ""
    values = dict(
        line.split("=", 1) for line in raw.splitlines() if "=" in line
    )
    return {
        "active": values.get("ActiveState") == "active",
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "pid": int(values.get("MainPID") or 0),
    }


def _bytes(value: str, unit: str) -> float:
    return float(value) * UNIT.get(unit, 1.0)


def build_status(root: Path, *, host: str, unit: str) -> dict[str, Any]:
    status_path = root / "refresh-status.tsv"
    manifest_path = (
        root / "kaggle/input/pokemon-tcg-ai-battle-episodes-index/manifest.csv"
    )
    raw_dir = root / "data/episodes/raw"
    feature_dir = root / "data/bootstrap"
    log_path = root / "logs/refresh.log"
    status = _tsv(status_path)
    manifest_rows: list[dict[str, str]] = []
    try:
        manifest_rows = list(
            csv.DictReader(manifest_path.open(newline="", encoding="utf-8"))
        )[-10:]
    except OSError:
        pass
    days = [str(row.get("date") or "") for row in manifest_rows if row.get("date")]
    if len(days) != 10:
        return {
            "available": False,
            "active": False,
            "host": host,
            "reason": "refreshed Kaggle index does not contain ten complete days",
            "source": str(manifest_path),
        }

    current_day = status.get("day") or ""
    stage = status.get("stage") or "waiting"
    completed = max(0, min(10, int(status.get("completed") or 0)))
    log = _tail(log_path).replace("\r", "\n")
    matches = list(DOWNLOAD_RE.finditer(log))
    download = matches[-1].groupdict() if matches else {}
    current_bytes = total_bytes = rate_bps = None
    day_percent = None
    if stage == "downloading" and download:
        day_percent = float(download["pct"])
        current_bytes = _bytes(download["current"], download["current_unit"])
        total_bytes = _bytes(download["total"], download["total_unit"])
        rate_bps = _bytes(download["rate"], download["rate_unit"])
    elif stage in {"validating", "ready_day"}:
        day_percent = 99.0
    elif stage == "ready":
        day_percent = 100.0

    rows: list[dict[str, Any]] = []
    for index, day in enumerate(days):
        archive = raw_dir / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        jsonl = feature_dir / f"top_ladder_all_{day}.jsonl"
        features = feature_dir / f"top_ladder_all_{day}.features"
        sidecar = features.with_suffix(features.suffix + ".json")
        if features.is_file() and sidecar.is_file():
            row_stage, percent = "feature_ready", 100.0
        elif jsonl.is_file():
            row_stage, percent = "converted", 70.0
        elif index < completed:
            row_stage, percent = "archive_ready", 50.0
        elif day == current_day:
            row_stage = stage
            percent = 50.0 * float(day_percent or 0.0) / 100.0
        elif archive.is_file() and stage == "ready":
            row_stage, percent = "archive_ready", 50.0
        else:
            row_stage, percent = "waiting", 0.0
        rows.append(
            {
                "day": day,
                "host": host,
                "stage": row_stage,
                "percent": round(percent, 2),
                "archive_bytes": archive.stat().st_size if archive.is_file() else None,
                "jsonl_bytes": jsonl.stat().st_size if jsonl.is_file() else None,
                "feature_bytes": features.stat().st_size if features.is_file() else None,
                "service": {"active": day == current_day and stage != "ready"},
            }
        )

    archive_ready = sum(row["stage"] in {"archive_ready", "converted", "feature_ready"} for row in rows)
    converted = sum(row["stage"] in {"converted", "feature_ready"} for row in rows)
    feature_ready = sum(row["stage"] == "feature_ready" for row in rows)
    # Download/validation is the first half of the expert-refresh pipeline;
    # conversion + featurization is the second half.
    percent = sum(float(row["percent"]) for row in rows) / 10.0
    service = _unit_state(unit)
    eta_s = (
        max(0.0, (float(total_bytes) - float(current_bytes)) / float(rate_bps))
        if current_bytes is not None and total_bytes is not None and rate_bps
        else None
    )
    detail = status.get("detail") or "waiting for expert replay refresh"
    if stage == "downloading" and current_bytes is not None and total_bytes is not None:
        detail = (
            f"{current_day}: {current_bytes / 1024**2:.0f}/{total_bytes / 1024**2:.0f} MiB "
            f"at {float(rate_bps or 0.0) / 1024**2:.2f} MiB/s"
        )
    return {
        "available": True,
        "active": bool(service["active"]),
        "complete": feature_ready == 10,
        "archive_window_ready": archive_ready == 10,
        "host": host,
        "stage": stage,
        "phase": (
            "download_validate"
            if archive_ready < 10
            else "convert_featurize"
            if feature_ready < 10
            else "ready_for_expert_rehearsal"
        ),
        "window_start": days[0],
        "window_end": days[-1],
        "current_day": current_day or None,
        "completed_days": archive_ready,
        "archive_ready_days": archive_ready,
        "converted_days": converted,
        "feature_ready_days": feature_ready,
        "total_days": 10,
        "percent": round(percent, 2),
        "day_percent": day_percent,
        "current": current_bytes,
        "total": total_bytes,
        "unit": "bytes",
        "rate": rate_bps,
        "eta_s": eta_s,
        "latest_line": detail,
        "days": rows,
        "service": service,
        "source": str(status_path),
        "updated_at": float(status.get("updated_epoch") or status_path.stat().st_mtime),
        "metric_definition": (
            "overall percent allocates 50% to archive download/validation and "
            "50% to conversion/feature readiness across the latest ten complete days"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="Elmo")
    parser.add_argument("--unit", default="pokebot-ladder-refresh-20260721.service")
    args = parser.parse_args()
    print(json.dumps(build_status(args.root, host=args.host, unit=args.unit), separators=(",", ":")))


if __name__ == "__main__":
    main()
