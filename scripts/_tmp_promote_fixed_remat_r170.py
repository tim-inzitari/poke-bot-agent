#!/usr/bin/env python3
"""Promote fixed-catalog Jul31-Aug5 days into main v5-r170 when records_kept>0."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

MAIN = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "teal-mask-ogerpon-ex-guide-corpus-full-v5-r170"
)
SIDE = Path(str(MAIN) + "-fixed-remat-jul31-aug5")
DAYS = [
    "2026-07-31",
    "2026-08-01",
    "2026-08-02",
    "2026-08-03",
    "2026-08-04",
    "2026-08-05",
]
POLL = 60
EXPECTED = {"2026-07-31": 135, "2026-08-01": 173, "2026-08-02": 88,
            "2026-08-03": 106, "2026-08-04": 146, "2026-08-05": 54}


def day_kept(day: str, root: Path) -> int | None:
    receipt = root / f"teal-mask-ogerpon-ex-{day}.features.receipt.json"
    feat = root / f"teal-mask-ogerpon-ex-{day}.features"
    if not receipt.is_file() or not feat.is_file():
        return None
    rec = json.loads(receipt.read_text())
    return int((rec.get("stats") or {}).get("records_kept") or 0)


def promote(day: str) -> dict:
    moved = []
    for suffix in [".features", ".features.json", ".features.receipt.json"]:
        src = SIDE / f"teal-mask-ogerpon-ex-{day}{suffix}"
        dst = MAIN / src.name
        if not src.is_file():
            continue
        if dst.exists():
            q = MAIN / "quarantine_before_fixed_promote_r170"
            q.mkdir(exist_ok=True)
            shutil.move(str(dst), str(q / f"{dst.name}.{int(time.time())}"))
        shutil.copy2(src, dst)
        moved.append(dst.name)
    return {"day": day, "moved": moved, "kept": day_kept(day, MAIN)}


def write_status(payload: dict) -> None:
    path = MAIN / "REMATERIALIZE_FIXED_PROMOTE_STATUS_r170.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    promoted: dict[str, dict] = {}
    print(f"[promote] watching {SIDE}", flush=True)
    while True:
        snap = {}
        for day in DAYS:
            kept = day_kept(day, SIDE)
            snap[day] = kept
            if kept is not None and kept > 0 and day not in promoted:
                print(f"[promote] promoting {day} kept={kept} expected~{EXPECTED.get(day)}", flush=True)
                promoted[day] = promote(day)
            elif kept == 0:
                print(f"[promote] WARN {day} sealed with 0 records (fixed catalog?)", flush=True)
        payload = {
            "schema": "poke_bot.slop_box_fixed_remat_promote_status_r170/v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "side_kept": snap,
            "promoted": promoted,
            "expected_by_day": EXPECTED,
            "all_positive": all(
                isinstance(snap.get(d), int) and snap[d] > 0 for d in DAYS
            ),
            "total_new_games": sum(v for v in snap.values() if isinstance(v, int)),
        }
        write_status(payload)
        if payload["all_positive"]:
            print("[promote] all six days positive; done", flush=True)
            print(json.dumps(payload, indent=2), flush=True)
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
