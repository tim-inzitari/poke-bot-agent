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
DEFAULT_ADDITIONAL_ROOTS = (
    Path(
        "/mnt/Main/main/poke-bot-agent/archive/"
        "teal-mask-ogerpon-ex-guide-corpus-full-v4-slop-box"
    ),
    Path(
        "/mnt/Main/main/poke-bot-agent/archive/"
        "teal-mask-ogerpon-ex-guide-corpus-full-v3"
    ),
    Path(
        "/mnt/Main/main/poke-bot-agent/archive/"
        "archaludon-ex-guide-full-public-schema7-r56-v1"
    ),
)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _unit_state(
    specialist_id: str,
    *,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    if (
        specialist_id == "teal-mask-ogerpon-ex"
        and corpus_root is not None
        and corpus_root.name
        == "teal-mask-ogerpon-ex-guide-corpus-full-v4-slop-box"
    ):
        unit = "pokebot-teal-mask-slop-box-full33-guide-v4.service"
    elif specialist_id == "teal-mask-ogerpon-ex":
        unit = "pokebot-teal-mask-full32-guide-v3-clean.service"
    elif specialist_id == "archaludon-ex":
        unit = (
            "pokebot-archaludon-ex-guide-full-public-"
            "schema7-r56-v1.service"
        )
    else:
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


def snapshot(
    root: Path = DEFAULT_ROOT,
    *,
    additional_roots: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    if additional_roots is None:
        additional_roots = (
            DEFAULT_ADDITIONAL_ROOTS if root == DEFAULT_ROOT else ()
        )
    status_paths = list(root.glob("*/status/window.json"))
    status_paths.extend(
        candidate / "status/window.json"
        for candidate in additional_roots
        if (candidate / "status/window.json").is_file()
    )
    for status_path in sorted(set(status_paths)):
        corpus_root = status_path.parent.parent
        ready_path = corpus_root / (
            "CURRENT_DECK_GUIDE_CORPUS_READY.json"
        )
        ready = _read(ready_path)
        inferred_specialist_id = corpus_root.name
        if inferred_specialist_id.startswith(
            "teal-mask-ogerpon-ex-guide-corpus-full"
        ):
            inferred_specialist_id = "teal-mask-ogerpon-ex"
        specialist_id = str(
            ready.get("specialist_id") or inferred_specialist_id
        )
        ready_valid = (
            ready.get("schema")
            == "poke_bot.current_deck_guide_corpus_ready/v1"
            and ready.get("status") == "ready"
            and ready.get("specialist_id") == specialist_id
        )
        final_ready_path: Path | None = None
        if specialist_id == "archaludon-ex":
            final_ready_path = (
                corpus_root / "ARCHALUDON_EX_GUIDE_CORPUS_READY.json"
            )
            final_ready = _read(final_ready_path)
            ready_valid = ready_valid and (
                final_ready.get("schema")
                == "poke_bot.archaludon_ex_guide_corpus_validation/v2"
                and final_ready.get("status") == "ready_checksum_validated"
                and int(final_ready.get("records") or 0) >= 16_639
                and int(final_ready.get("dataset_schema") or -1) == 7
                and int(final_ready.get("feature_schema") or -1) == 5
                and int(
                    (final_ready.get("source_window") or {}).get("days")
                    or 0
                )
                == 44
                and final_ready.get("schema6_feature_reuse_allowed")
                is False
            )
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
        service = _unit_state(
            specialist_id,
            corpus_root=corpus_root,
        )
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
                "ready": ready_valid,
                "ready_receipt": str(ready_path) if ready_path.is_file() else None,
                "final_validation_receipt": (
                    str(final_ready_path)
                    if final_ready_path is not None
                    and final_ready_path.is_file()
                    else None
                ),
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
