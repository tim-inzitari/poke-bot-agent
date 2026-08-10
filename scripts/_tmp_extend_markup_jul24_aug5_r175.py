#!/usr/bin/env python3
"""Extend self-play-equivalent matchup markup over Jul24-Aug5 schema-8 corpus.

Waits for the combined protected expert corpus/manifest, then runs offline-oracle
markup. Does not start bootstrap.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/inzi/poke-bot-agent")
EXPERT = ROOT / "data/bootstrap/expert-slop-box-schema8-jul24-aug5-r175/teal-mask-ogerpon-ex"
MANIFEST = EXPERT / "manifest.json"
PROTECTED = EXPERT / "PROTECTED_EXPERT_CORPUS.json"
SEAL = ROOT / "outputs/state/slop-box-schema8-jul24-aug5-sealed-r175.json"
CPU_READY = ROOT / "outputs/state/slop-box-schema8-jul24-aug5-cpu-pack-ready-r175.json"
ARCHIVE = Path("/tmp/truenas_main/poke-bot-agent/archive/episode-days")
OUT = ROOT / "outputs/bootstrap/slop-box-h10-rtp/expert-matchup-marked-selfplay-equiv-jul24-aug5-r175"
WORK = ROOT / "outputs/bootstrap/slop-box-h10-rtp/.expert-matchup-marked-selfplay-equiv-jul24-aug5-r175.work"
RECEIPT = ROOT / "outputs/state/slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-r175.json"
ROSTER = ROOT / "state/matchup_adapter_roster.json"
GATE = ROOT / "outputs/state/alakazam_gate_program_v1.json"
LOG = ROOT / "outputs/bootstrap/slop-box-h10-rtp/logs/expert-matchup-markup-jul24-aug5-r175.log"
PYBIN = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
SCRIPT = ROOT / "scripts/markup_slop_box_expert_matchup_selfplay_equiv_r171.py"


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    print("[markup-r175] waiting for Jul24-Aug5 manifest + seal", flush=True)
    while True:
        ready = (
            SEAL.is_file()
            and MANIFEST.is_file()
            and PROTECTED.is_file()
            and ARCHIVE.is_dir()
        )
        if ready:
            break
        print(
            f"[markup-r175] seal={SEAL.is_file()} manifest={MANIFEST.is_file()} "
            f"protected={PROTECTED.is_file()} archive={ARCHIVE.is_dir()}",
            flush=True,
        )
        time.sleep(30)

    # Prefer waiting for CPU pack too so markup consumes the same sealed corpus
    # the pack binder published; do not block forever if pack is still building.
    deadline = time.time() + 3600
    while not CPU_READY.is_file() and time.time() < deadline:
        print("[markup-r175] waiting briefly for cpu pack ready...", flush=True)
        time.sleep(30)

    # The markup worker assembles into a sibling temporary directory and
    # atomically renames it to OUT.  Creating OUT here makes every launch fail
    # its deliberate no-overwrite guard before resumable work can begin.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYBIN,
        "-u",
        str(SCRIPT),
        "--feature-manifest",
        str(MANIFEST),
        "--archive-dir",
        str(ARCHIVE),
        "--output-dir",
        str(OUT),
        "--roster",
        str(ROSTER),
        "--gate-contract",
        str(GATE),
        "--receipt-out",
        str(RECEIPT),
        "--jobs",
        "7",
        "--thread-workers",
        "4",
        "--work-dir",
        str(WORK),
    ]
    print("RUN", " ".join(cmd), flush=True)
    _atomic(
        ROOT / "outputs/state/slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-started-r175.json",
        {
            "schema": "poke_bot.slop_box_markup_jul24_aug5_started_r175/v1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "cmd": cmd,
            "window": {"start": "2026-07-24", "end": "2026-08-05"},
            "offline_oracle_only": True,
            "bootstrap_held": True,
            "do_not_start_bootstrap": True,
        },
    )
    with LOG.open("a", encoding="utf-8") as log:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=log, stderr=log)
    if rc != 0:
        raise SystemExit(rc)
    receipt = json.loads(RECEIPT.read_text())
    summary = {
        "schema": "poke_bot.slop_box_markup_jul24_aug5_summary_r175/v1",
        "status": "ready",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "receipt": str(RECEIPT),
        "output_dir": str(OUT),
        "marked_games": receipt.get("marked_games"),
        "marked_decisions": receipt.get("marked_decisions"),
        "source_games": receipt.get("source_games"),
        "source_decisions": receipt.get("source_decisions"),
        "window": {"start": "2026-07-24", "end": "2026-08-05"},
        "offline_oracle_only": True,
        "bootstrap_held": True,
        "do_not_start_bootstrap": True,
    }
    _atomic(
        ROOT / "outputs/state/slop-box-h10-rtp-expert-matchup-markup-jul24-aug5-summary-r175.json",
        summary,
    )
    print("MARKUP_READY", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
