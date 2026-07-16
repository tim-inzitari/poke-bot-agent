#!/usr/bin/env python3
"""Post-attach watch for Core Elmo+Bert; write result artifacts."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "outputs/state/BERT_REATTACH_STATUS.txt"
OWNER = ROOT / "outputs/state/SOLE_TRAIN_OWNER.lock"
TOPOLOGY = ROOT / "outputs/state/TRACK_A_TOPOLOGY.txt"
RESULT = ROOT / "outputs/state/BERT_REATTACH_RESULT.json"
ELMO = "192.168.1.143:8765"
BERT = "bert.local:8766"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with STATUS.open("a") as fh:
        fh.write(line + "\n")


def find() -> dict:
    out: dict = {}
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit():
            continue
        try:
            cmd = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except Exception:
            continue
        if "AGENT_LOOP" in cmd or "bert_reattach_watch" in cmd:
            continue
        if "launch_core_pipeline.py" in cmd and "core_kernel" in cmd:
            out["launch"] = int(ent.name)
        if "train_core_pipeline.py" in cmd and "core_kernel" in cmd:
            out["pipe"] = int(ent.name)
        if (
            "train_round_robin.py" in cmd
            and "hammer_search_rl" in cmd
            and "core_kernel" in cmd
        ):
            out["rr"] = int(ent.name)
            out["rr_cmd"] = cmd
        if "train_round_robin.py" in cmd and "blackwell_hammer" in cmd:
            out["bw"] = int(ent.name)
            out["bw_cmd"] = cmd
    return out


def main() -> int:
    p = find()
    log(f"PIDS launch={p.get('launch')} pipe={p.get('pipe')} rr={p.get('rr')} bw={p.get('bw')}")
    if not p.get("rr"):
        log("FAIL no core rr")
        return 2
    if ELMO not in p["rr_cmd"] or BERT not in p["rr_cmd"]:
        log("FAIL cmdline missing endpoints")
        return 3
    if "remote-worker" in p.get("bw_cmd", ""):
        log("FAIL bw has remote")
        return 5
    log("PROOF cmdline elmo+bert OK; bw local OK")

    raw = (ROOT / "outputs/logs/core_kernel.log").read_text(errors="replace").replace(
        "\r", "\n"
    )
    idx = raw.rfind("[core-pipeline][start]")
    chunk = raw[idx:]
    remotes = [l.strip() for l in chunk.splitlines() if "remote-worker=" in l]
    has_both = any(BERT in l for l in remotes) and any(ELMO in l for l in remotes)
    log(f"PROOF log_remotes_both={has_both} lines={remotes[-4:]}")
    if not has_both:
        return 3

    t0 = time.time()
    samples = []
    storm = False
    rr = p["rr"]
    while time.time() - t0 < 300:
        if not Path(f"/proc/{rr}").exists():
            p2 = find()
            if not p2.get("rr"):
                log("CORE_RR_DIED")
                return 6
            rr = p2["rr"]
            if BERT not in p2.get("rr_cmd", ""):
                log("CORE_LOST_BERT")
                return 7
        raw = (ROOT / "outputs/logs/core_kernel.log").read_text(errors="replace").replace(
            "\r", "\n"
        )
        idx = raw.rfind("[core-pipeline][start]")
        chunk = raw[idx:]
        g = re.findall(r"iter(\d+) games:\s+\d+%\|[^\|]*\|\s+(\d+)/(\d+)", chunk)
        to = chunk.count("RemoteLeafTimeout")
        slot = len(re.findall(r"bert\.local:8766 slot failed", chunk))
        rerr = len(re.findall(r"\[remote\] bert\.local", chunk))
        ss = subprocess.check_output(["ss", "-tn"], text=True)
        elmo_e = sum(
            1 for l in ss.splitlines() if "ESTAB" in l and "192.168.1.143:8765" in l
        )
        bert_e = sum(1 for l in ss.splitlines() if "ESTAB" in l and ":8766" in l)
        sample = {
            "t": int(time.time() - t0),
            "games": g[-1] if g else None,
            "timeouts": to,
            "bert_slot": slot,
            "remote_bert_err": rerr,
            "elmo_estab": elmo_e,
            "bert_estab": bert_e,
        }
        samples.append(sample)
        log(f"WATCH {sample}")
        OWNER.write_text(
            "SOLE_TRAIN_OWNER=1\n"
            f"status=BERT_REATTACH_WATCH\nupdated={utc()}\n"
            "agent=bert-reattach-track-b\ntopology=CORE_ELMO_PLUS_BERT\n"
            "bw=local-only\n"
            "rule=AUTHORIZED Bert on Core. Do NOT SIGTERM Core for bert.local.\n"
            f"pids=launch={p.get('launch')} pipe={p.get('pipe')} rr={rr} bw={p.get('bw')}\n"
            f"last={sample}\n"
        )
        if slot >= 8 or rerr >= 8:
            storm = True
            log(f"STORM {sample}")
            break
        time.sleep(20)

    p = find()
    pids = {
        "core_launch": p.get("launch"),
        "core_pipe": p.get("pipe"),
        "core_rr": p.get("rr"),
        "bw_rr": p.get("bw"),
        "bw_launch": 2598057,
    }
    RESULT.write_text(
        json.dumps(
            {
                "updated": utc(),
                "ok": not storm,
                "storm": storm,
                "pids": pids,
                "samples": samples,
                "remotes": remotes[-4:],
            },
            indent=2,
        )
    )
    if storm:
        OWNER.write_text(
            "SOLE_TRAIN_OWNER=1\n"
            f"status=BERT_STORM_DETECTED\nupdated={utc()}\n"
            "agent=bert-reattach-track-b\ntopology=CORE_ELMO_PLUS_BERT\n"
            "bw=local-only\n"
            f"pids={pids}\nlast={samples[-1]}\n"
        )
        log("DONE storm detected (Bert still attached; report)")
        return 4

    TOPOLOGY.write_text(
        f"""# Track B topology (Bert on Core)

updated={utc()}

## Production
- **Core**: Elmo `{ELMO}` + Bert `{BERT}`, `POKEBOT_REMOTE_PRIMARY=1`, no SLOT_DIVISOR
- **Blackwell**: local-only (no Bert)
- Post-attach 5m watch: no Bert slot-fail storm

## PIDs
- core_launch={pids['core_launch']} core_pipe={pids['core_pipe']} core_rr={pids['core_rr']}
- bw_launch=2598057 bw_rr={pids['bw_rr']}

## Proof
- cmdline Elmo+Bert
- remote-worker hello both
- bw local
- last={samples[-1]}
"""
    )
    OWNER.write_text(
        "SOLE_TRAIN_OWNER=1\n"
        f"status=OVERNIGHT_MONITOR\nupdated={utc()}\n"
        "agent=bert-reattach-track-b\ntopology=CORE_ELMO_PLUS_BERT\n"
        "bw=local-only\n"
        "rule=Bert on Core authorized (Track B GO). BW local-only. Do NOT SIGTERM Core for bert.local.\n"
        f"pids={pids}\nlast={samples[-1]}\n"
    )
    log(f"DONE Bert attached pids={pids} last={samples[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
