#!/usr/bin/env python3
"""Patch Bert dashboard server integrity for completed Slop Box bootstrap."""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

PATH = Path("/Users/tsinzitari/pokebot-dashboard/v1/server.py")


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if "receipt_backed_slop_box_bootstrap" in text:
        print("already patched")
        return 0

    needle = """        receipt_backed_stopped_final_refresh = bool(
            not service_active
            and training.get("status") == "stopped"
            and active_runtime_refresh.get("active") is False
            and str(training.get("mode") or "").startswith("final_format_")
            and runtime_specialist
            == str(active_runtime_refresh.get("specialist_id") or "")
            == str(protocol.get("canonical_active_refresh_specialist") or "")
            and str(active_runtime_refresh.get("run_name") or "")
            == str(training.get("run") or "")
        )
"""
    insert = """        receipt_backed_stopped_final_refresh = bool(
            not service_active
            and training.get("status") in {"stopped", "complete"}
            and active_runtime_refresh.get("active") is False
            and str(training.get("mode") or "").startswith("final_format_")
            and runtime_specialist
            == str(active_runtime_refresh.get("specialist_id") or "")
            == str(protocol.get("canonical_active_refresh_specialist") or "")
            and str(active_runtime_refresh.get("run_name") or "")
            == str(training.get("run") or "")
        )
        receipt_backed_slop_box_bootstrap = bool(
            str(training.get("mode") or "")
            == "final_format_slop_box_h10_rtp_bootstrap"
            and runtime_specialist == "teal-mask-ogerpon-ex"
            and str(training.get("run") or curriculum.get("run") or "")
            == "final_format_slop_box_h10_rtp_bootstrap"
            and (
                training.get("status") == "running"
                or (
                    training.get("status") in {"stopped", "complete"}
                    and str(training.get("phase") or "").endswith(
                        "bootstrap:complete"
                    )
                )
            )
        )
"""
    if needle not in text:
        # maybe stopped already accepts complete
        needle2 = needle.replace(
            'training.get("status") == "stopped"',
            'training.get("status") in {"stopped", "complete"}',
        )
        if needle2 in text:
            needle = needle2
        else:
            raise SystemExit("stopped-refresh anchor missing")
    old_fr = """        final_refresh_current = (
            receipt_backed_final_refresh
            or receipt_backed_stopped_final_refresh
"""
    new_fr = """        final_refresh_current = (
            receipt_backed_final_refresh
            or receipt_backed_stopped_final_refresh
            or receipt_backed_slop_box_bootstrap
"""
    if old_fr not in text:
        raise SystemExit("final_refresh_current anchor missing")
    bak = PATH.with_name(PATH.name + f".pre-slop-box-integrity-{stamp}")
    shutil.copy2(PATH, bak)
    text = text.replace(needle, insert, 1).replace(old_fr, new_fr, 1)
    PATH.write_text(text, encoding="utf-8")
    print(f"patched {PATH} bak={bak.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
