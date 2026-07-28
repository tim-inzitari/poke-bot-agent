#!/usr/bin/env python3
"""Emit the current Elmo current-deck-guide preparation windows as JSON."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "expert-latest20-derived/daily/current-deck-guides-v1"
)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _unit_state(specialist_id: str) -> dict[str, Any]:
    unit = f"pokebot-{specialist_id}-guide-window-v1.service"
    try:
        raw = subprocess.run(
            [
                "systemctl",
                "show",
                unit,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "Result",
                "-p",
                "MainPID",
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        raw = ""
    fields = dict(
        line.split("=", 1)
        for line in raw.splitlines()
        if "=" in line
    )
    return {
        "name": unit,
        "active": fields.get("ActiveState") in {"active", "activating"},
        "active_state": fields.get("ActiveState"),
        "sub_state": fields.get("SubState"),
        "result": fields.get("Result"),
        "pid": int(fields.get("MainPID") or 0),
    }


def snapshot(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    for status_path in sorted(root.glob("*/status/window.json")):
        specialist_id = status_path.parent.parent.name
        status = _read(status_path)
        if not status:
            continue
        completed = [
            row
            for row in (status.get("completed") or [])
            if isinstance(row, dict)
        ]
        date_window = dict(status.get("date_window") or {})
        totals = dict(status.get("totals") or {})
        ready_path = status_path.parent.parent / (
            "CURRENT_DECK_GUIDE_CORPUS_READY.json"
        )
        ready = _read(ready_path)
        service = _unit_state(specialist_id)
        expected_days = int(date_window.get("days") or 0)
        complete_days = len(completed)
        windows.append(
            {
                "specialist_id": specialist_id,
                "state": status.get("state"),
                "date_start": date_window.get("start"),
                "date_end": date_window.get("end"),
                "expected_days": expected_days,
                "completed_days": complete_days,
                "percent": (
                    100.0 * complete_days / expected_days
                    if expected_days > 0
                    else None
                ),
                "current_dates": list(status.get("current_dates") or []),
                "records": int(totals.get("records") or 0),
                "decisions": int(totals.get("decisions") or 0),
                "guide_rows": int(totals.get("guide_rows") or 0),
                "updated_at": status.get("updated_at"),
                "ready": (
                    ready.get("schema")
                    == "poke_bot.current_deck_guide_corpus_ready/v1"
                    and ready.get("status") == "ready"
                    and ready.get("specialist_id") == specialist_id
                ),
                "ready_receipt": str(ready_path) if ready_path.is_file() else None,
                "service": service,
                "status_source": str(status_path),
            }
        )
    windows.sort(
        key=lambda row: (
            row["state"] == "running" or row["service"]["active"],
            float(row.get("updated_at") or 0),
        ),
        reverse=True,
    )
    return {
        "schema": "poke_bot.current_deck_guide_prestage_snapshot/v1",
        "available": bool(windows),
        "observed_at": time.time(),
        "active": windows[0] if windows else None,
        "windows": windows,
    }


if __name__ == "__main__":
    print(json.dumps(snapshot(), separators=(",", ":"), sort_keys=True))
