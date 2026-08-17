#!/usr/bin/env python3
"""Promote elmo v5 teal corpus into local bootstrap pointer for CE fine-tune."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/tmp/truenas_main/poke-bot-agent/archive/teal-mask-ogerpon-ex-guide-corpus-full-v5-r170")
DST = Path("/home/pokebot/poke-bot-agent/data/bootstrap/expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex")

def main() -> int:
    if not (SRC / "CURRENT_DECK_GUIDE_CORPUS_READY.json").is_file():
        raise SystemExit(f"source not ready: {SRC}")
    if not (SRC / "PROTECTED_EXPERT_CORPUS.json").is_file():
        raise SystemExit("missing protected pointer on source")
    DST.mkdir(parents=True, exist_ok=True)
    # rsync-like copy with hardlink where possible
    cmd = [
        "rsync", "-a", "--info=progress2",
        str(SRC) + "/",
        str(DST) + "/",
    ]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ready = json.loads((DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json").read_text())
    pointer = DST / "PROTECTED_EXPERT_CORPUS.json"
    receipt = {
        "schema": "poke_bot.slop_box_expanded_corpus_promoted_r170/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC),
        "destination": str(DST),
        "protected_pointer": str(pointer),
        "records": ready.get("records"),
        "decisions": ready.get("decisions"),
        "days": ready.get("days"),
        "date_end": ready.get("date_end"),
    }
    out = Path("/home/pokebot/poke-bot-agent/outputs/state/slop-box-expanded-corpus-promoted-r170.json")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
