#!/usr/bin/env python3
"""Finalize expanded teal corpus + start Chao-hard CE once remat days exist.

Owner r170: wait for Jul24–Aug5 feature shards under the elmo/v5 archive
(or local TrueNAS mount), assemble CURRENT_DECK_GUIDE_CORPUS_READY, promote
into expert-slop-box-teal-mask-full41-r170, stage Chao-hard maps, and start
pokebot-slop-box-chao-hard-ce-r170.service via systemd.

Safe to poll: exits 0 with status=waiting while required days are incomplete.
Does not touch Crustle. Does not restart healthy unrelated trainers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.expert_pilot_importance import file_digest  # noqa: E402

SPECIALIST = "teal-mask-ogerpon-ex"
PRODUCTIVE_START = "2026-07-24"
PRODUCTIVE_END = "2026-08-05"
# Immediate-learning floor: Jul24-30 already materialized (1997 games).
# Full Aug tail continues on elmo; pass --end to raise the floor later.
DEFAULT_ACTIVATE_END = "2026-07-30"
DEFAULT_SRC_CANDIDATES = [
    Path("/tmp/truenas_main/poke-bot-agent/archive/teal-mask-ogerpon-ex-guide-corpus-full-v5-r170"),
    Path("/mnt/Main/main/poke-bot-agent/archive/teal-mask-ogerpon-ex-guide-corpus-full-v5-r170"),
]
DST = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex"
)
STATE = Path("/home/inzi/poke-bot-agent/outputs/state")
UNIT = "pokebot-slop-box-chao-hard-ce-r170.service"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def resolve_src(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for candidate in DEFAULT_SRC_CANDIDATES:
        try:
            if candidate.is_dir() and os.access(candidate, os.R_OK | os.X_OK):
                # Probe one known day file; ACL can allow dir list but deny files.
                probe = candidate / f"{SPECIALIST}-2026-07-25.features"
                if probe.is_file() and os.access(probe, os.R_OK):
                    return candidate.resolve()
        except OSError:
            continue
    raise SystemExit(
        "no readable v5-r170 archive mount found "
        "(fix NFS ACLs on elmo or pass --src)"
    )


def day_ready(src: Path, day: str) -> dict[str, Any] | None:
    feat = src / f"{SPECIALIST}-{day}.features"
    receipt = src / f"{SPECIALIST}-{day}.features.receipt.json"
    try:
        if not feat.is_file() or not receipt.is_file():
            return None
        size = feat.stat().st_size
        if size <= 0:
            return None
        receipt_obj = json.loads(receipt.read_text(encoding="utf-8"))
    except PermissionError:
        return None
    except OSError:
        return None
    stats = receipt_obj.get("stats") or {}
    receipt_digest = (
        receipt_obj.get("sha256")
        or (receipt_obj.get("output") or {}).get("sha256")
        or stats.get("sha256")
    )
    return {
        "date": day,
        "decisions": int(stats.get("decisions_kept") or 0),
        "guide_rows": int(stats.get("guide_rows") or stats.get("guide_labels") or 0),
        "output": str(feat),
        "records": int(stats.get("records_kept") or 0),
        "resumed": True,
        # Prefer receipt digest during poll; avoid multi-GB NFS rehash every timer tick.
        "sha256": receipt_digest or f"size:{size}",
        "zero_guide_rows": int(stats.get("guide_rows") or 0) == 0,
    }


def inventory(src: Path, *, start: str, end: str) -> dict[str, Any]:
    days = _dates(start, end)
    completed: list[dict[str, Any]] = []
    missing: list[str] = []
    for day in days:
        row = day_ready(src, day)
        if row is None:
            missing.append(day)
        else:
            completed.append(row)
    return {
        "expected_days": days,
        "completed": completed,
        "missing": missing,
        "records": sum(int(r["records"]) for r in completed),
        "decisions": sum(int(r["decisions"]) for r in completed),
        "start": start,
        "end": end,
    }


def write_complete_status(
    src: Path, completed: list[dict[str, Any]], *, start: str, end: str
) -> Path:
    status_path = src / "status" / f"window_productive_{start}_{end}_r170.json"
    payload = {
        "schema": "poke_bot.authoritative_guide_window_status/v1",
        "state": "complete",
        "pid": os.getpid(),
        "started_at": time.time(),
        "updated_at": time.time(),
        "date_window": {
            "start": start,
            "end": end,
            "days": len(completed),
        },
        "current_dates": [],
        "completed": completed,
        "totals": {
            "days": len(completed),
            "records": sum(int(r["records"]) for r in completed),
            "decisions": sum(int(r["decisions"]) for r in completed),
            "guide_rows": sum(int(r["guide_rows"]) for r in completed),
        },
    }
    _atomic_json(status_path, payload)
    return status_path


def finalize(src: Path, status_path: Path, *, start: str, end: str) -> None:
    catalog = src / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "finalize_current_deck_guide_window.py"),
        "--status",
        str(status_path),
        "--out-dir",
        str(src),
        "--start",
        start,
        "--end",
        end,
        "--specialist-id",
        SPECIALIST,
        "--guide-version",
        "teal-mask-ogerpon-ex-slop-box-north-star-v3",
        "--minimum-records",
        "1442",
        "--public-deck-catalog",
        str(catalog),
    ]
    print("[activate] finalize", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def promote(src: Path) -> dict[str, Any]:
    DST.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--info=stats2", str(src) + "/", str(DST) + "/"]
    print("[activate] promote", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ready = json.loads((DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json").read_text())
    receipt = {
        "schema": "poke_bot.slop_box_expanded_corpus_promoted_r170/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "destination": str(DST),
        "protected_pointer": str(DST / "PROTECTED_EXPERT_CORPUS.json"),
        "records": ready.get("records"),
        "decisions": ready.get("decisions"),
        "days": ready.get("days"),
        "date_end": ready.get("date_end"),
        "date_start": ready.get("date_start"),
    }
    _atomic_json(STATE / "slop-box-expanded-corpus-promoted-r170.json", receipt)
    return receipt


def stage_and_start(*, epochs: int) -> dict[str, Any]:
    expert = DST / "PROTECTED_EXPERT_CORPUS.json"
    stage_cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "stage_slop_box_chao_hard_ce_r170.py"),
        "--expert-pointer",
        str(expert),
        "--cox-chao-train-weight",
        "25",
        "--james-cox-train-weight",
        "5",
        "--epochs",
        str(epochs),
        "--install-systemd",
    ]
    print("[activate] stage", " ".join(stage_cmd), flush=True)
    subprocess.run(stage_cmd, check=True, cwd=str(ROOT))
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT], check=False)
    print("[activate] start", UNIT, flush=True)
    subprocess.run(["systemctl", "--user", "start", UNIT], check=True)
    return {"unit": UNIT, "started": True}


def update_dash(inv: dict[str, Any], *, phase: str, line: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    proj = {
        "schema": "poke_bot.slop_box_dashboard_bootstrap_projection_r170/v1",
        "specialist": SPECIALIST,
        "display_name": "Slop Box H10 RTP",
        "phase": phase,
        "status": phase,
        "ready": False,
        "rl": False,
        "cox_chao_gate_threshold": 0.9,
        "latest_line": line,
        "corpus_records": inv.get("records"),
        "corpus_decisions": inv.get("decisions"),
        "missing_days": inv.get("missing"),
        "updated_at_utc": now,
    }
    _atomic_json(STATE / "slop-box-dashboard-bootstrap-projection-r170.json", proj)
    sync = {
        "schema": "poke_bot.slop_box_dashboard_active_sync/v1",
        "goal_revision": 170,
        "phase": phase,
        "latest_line": line,
        "cox_chao_gate_passed": False,
        "cox_chao_gate_threshold": 0.9,
        "crustle_data_deleted": False,
        "bootstrap_restarted": False,
        "updated_at_utc": now,
        "created_at_utc": now,
    }
    _atomic_json(STATE / "slop-box-dashboard-active-sync-r170.json", sync)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--start", default=PRODUCTIVE_START)
    parser.add_argument(
        "--end",
        default=DEFAULT_ACTIVATE_END,
        help=(
            f"Inclusive end date to require before activate "
            f"(default {DEFAULT_ACTIVATE_END}; full tail is {PRODUCTIVE_END})"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=0)
    parser.add_argument("--force-finalize", action="store_true")
    args = parser.parse_args()

    src = resolve_src(args.src)
    start = str(args.start)
    end = str(args.end)
    deadline = None
    if args.poll_seconds > 0:
        deadline = time.time() + float(args.poll_seconds)

    while True:
        inv = inventory(src, start=start, end=end)
        if inv["missing"]:
            update_dash(
                inv,
                phase="bootstrap:full_archive_rematerialize",
                line=(
                    f"WAITING remat {start}..{end} days {len(inv['completed'])}/"
                    f"{len(inv['expected_days'])} · missing {','.join(inv['missing'][:4])}"
                    + ("…" if len(inv["missing"]) > 4 else "")
                    + f" · records_so_far={inv['records']}"
                ),
            )
            receipt = {
                "schema": "poke_bot.slop_box_chao_hard_activate_r170/v1",
                "status": "waiting_rematerialize",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "src": str(src),
                "window": {"start": start, "end": end},
                "inventory": {
                    "completed_days": len(inv["completed"]),
                    "expected_days": len(inv["expected_days"]),
                    "missing": inv["missing"],
                    "records": inv["records"],
                    "decisions": inv["decisions"],
                },
            }
            _atomic_json(STATE / "slop-box-chao-hard-activate-r170.json", receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
            if deadline is None or time.time() >= deadline:
                return 0
            time.sleep(min(60, max(5, deadline - time.time())))
            continue

        ready_marker = src / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
        if args.force_finalize or not ready_marker.is_file():
            status_path = write_complete_status(
                src, inv["completed"], start=start, end=end
            )
            finalize(src, status_path, start=start, end=end)
        promoted = promote(src)
        update_dash(
            inv,
            phase="bootstrap:chao_hard_starting",
            line=(
                f"EXPANDED corpus ready {start}..{end} "
                f"records={promoted.get('records')} "
                f"decisions={promoted.get('decisions')} · starting Chao-hard CE"
            ),
        )
        started = stage_and_start(epochs=int(args.epochs))
        receipt = {
            "schema": "poke_bot.slop_box_chao_hard_activate_r170/v1",
            "status": "chao_hard_started",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "src": str(src),
            "window": {"start": start, "end": end},
            "promoted": promoted,
            "started": started,
            "inventory": {
                "completed_days": len(inv["completed"]),
                "expected_days": len(inv["expected_days"]),
                "records": inv["records"],
                "decisions": inv["decisions"],
            },
            "note": (
                "Aug1-5 remat may still be running; re-run with "
                f"--end {PRODUCTIVE_END} --force-finalize after those days land"
            ),
        }
        _atomic_json(STATE / "slop-box-chao-hard-activate-r170.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
