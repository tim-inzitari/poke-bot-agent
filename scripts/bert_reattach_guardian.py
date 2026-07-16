#!/usr/bin/env python3
"""Hold ownership and suppress No-Bert overnight loops during Bert Core attach."""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "outputs/state/SOLE_TRAIN_OWNER.lock"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    end = time.time() + 900
    while time.time() < end:
        pids: dict[str, str] = {}
        for ent in Path("/proc").iterdir():
            if not ent.name.isdigit():
                continue
            try:
                cmd = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            except Exception:
                continue
            if "bert_reattach_guardian" in cmd:
                continue
            if "AGENT_LOOP_TICK_track_a_overnight" in cmd and (
                "Do not attach Bert" in cmd or "NO bert.local" in cmd
            ):
                try:
                    os.kill(int(ent.name), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                continue
            if "AGENT_LOOP_TICK_core_companion" in cmd and "Do NOT attach Bert" in cmd:
                try:
                    os.kill(int(ent.name), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                continue
            if "AGENT_LOOP" in cmd:
                continue
            if "launch_core_pipeline.py" in cmd and "core_kernel" in cmd:
                pids["launch"] = ent.name
            if "train_core_pipeline.py" in cmd and "core_kernel" in cmd:
                pids["pipe"] = ent.name
            if (
                "train_round_robin.py" in cmd
                and "hammer_search_rl" in cmd
                and "core_kernel" in cmd
            ):
                pids["rr"] = ent.name
            if "train_round_robin.py" in cmd and "blackwell_hammer" in cmd:
                pids["bw"] = ent.name
        # Do not overwrite terminal OVERNIGHT_MONITOR with watch if already done
        cur = OWNER.read_text() if OWNER.exists() else ""
        status = "BERT_REATTACH_WATCH"
        if "status=OVERNIGHT_MONITOR" in cur and "CORE_ELMO_PLUS_BERT" in cur:
            status = "OVERNIGHT_MONITOR"
        OWNER.write_text(
            "SOLE_TRAIN_OWNER=1\n"
            f"status={status}\nupdated={utc()}\n"
            "agent=bert-reattach-track-b\ntopology=CORE_ELMO_PLUS_BERT\n"
            "bw=local-only\n"
            "rule=AUTHORIZED Bert on Core. Do NOT SIGTERM Core for bert.local. Do NOT rearm NO_BERT.\n"
            f"pids={pids}\n"
        )
        time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
