#!/usr/bin/env python3
"""Bert dash: treat stopped receipt-backed final-format refresh as current stage/progress."""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

PATH = Path("/Users/tsinzitari/pokebot-dashboard/v1/server.py")


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    changed: list[str] = []

    old_stage = """                "current": bool(
                    handoff_progress_current
                    or handoff_transition_current
                    or managed_boundary_current
                    or managed_boundary_paused_current
                    or (
                        service_active
                        and int(service.get("restart_count") or 0) >= 0
                    )
                ),
"""
    new_stage = """                "current": bool(
                    handoff_progress_current
                    or handoff_transition_current
                    or managed_boundary_current
                    or managed_boundary_paused_current
                    or final_refresh_current
                    or (
                        service_active
                        and int(service.get("restart_count") or 0) >= 0
                    )
                ),
"""
    if "or final_refresh_current\n                    or (\n                        service_active" not in text:
        if old_stage not in text:
            raise SystemExit("stage anchor missing")
        text = text.replace(old_stage, new_stage, 1)
        changed.append("stage")

    old_prog = """                "current": bool(
                    handoff_progress_current
                    or handoff_transition_current
                    or managed_boundary_paused_current
                    or (
                        curriculum.get("active")
                        and curriculum.get("source_current") is True
                        and str(curriculum.get("run") or "")
                    )
                ),
"""
    new_prog = """                "current": bool(
                    handoff_progress_current
                    or handoff_transition_current
                    or managed_boundary_paused_current
                    or final_refresh_current
                    or (
                        curriculum.get("source_current") is True
                        and str(curriculum.get("run") or "")
                        and (
                            curriculum.get("active")
                            or training.get("status")
                            in {"stopped", "complete"}
                        )
                    )
                ),
"""
    if "or final_refresh_current\n                    or (\n                        curriculum.get(\"source_current\")" not in text:
        if old_prog not in text:
            raise SystemExit("progress anchor missing")
        text = text.replace(old_prog, new_prog, 1)
        changed.append("progress")

    if not changed:
        print("already patched")
        return 0

    bak = PATH.with_name(PATH.name + f".pre-progress-stopped-{stamp}")
    shutil.copy2(PATH, bak)
    PATH.write_text(text, encoding="utf-8")
    print(f"patched {changed} bak={bak.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
