#!/usr/bin/env python3
"""Supervise Chao-hard CE: refresh dash, wait for 60 epochs or unit exit."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/inzi/poke-bot-agent")
STATE = ROOT / "outputs/state"
RUN = ROOT / "outputs/bootstrap/final_format_slop_box_chao_hard_r170"
UNIT = "pokebot-slop-box-chao-hard-ce-r170.service"
TARGET = 60
POLL = 120
MAX_WAIT_S = 3 * 3600


def epochs_done() -> int:
    cks = sorted(
        RUN.glob("checkpoints/epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return int(cks[-1].stem.split("_")[1]) if cks else 0


def unit_state() -> dict[str, str]:
    out = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            UNIT,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "NRestarts",
            "-p",
            "Result",
            "--no-pager",
        ],
        text=True,
    )
    data: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def refresh_dash(done: int, us: dict[str, str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    pid = us.get("MainPID")
    line = (
        f"LIVE Chao-hard CE epoch {done}/{TARGET} · hot-start epoch_152 · "
        f"corpus Jul24-30 1997g/135774d · Chao×25 James×5 · select by Chao held · "
        f"gate 0.90 · PID {pid} · remat Jul31-Aug5 continuing · no RL"
    )
    active = us.get("ActiveState") == "activating"
    proj = {
        "schema": "poke_bot.slop_box_dashboard_bootstrap_projection_r170/v1",
        "specialist": "teal-mask-ogerpon-ex",
        "display_name": "Slop Box H10 RTP",
        "phase": "bootstrap:chao_hard_ce",
        "status": "running" if active else us.get("ActiveState"),
        "ready": False,
        "rl": False,
        "cox_chao_gate_threshold": 0.9,
        "latest_line": line,
        "corpus_records": 1997,
        "corpus_decisions": 135774,
        "chao_hard": {
            "unit": UNIT,
            "epochs_completed": done,
            "epochs_planned": TARGET,
            "main_pid": pid,
            "n_restarts": us.get("NRestarts"),
        },
        "updated_at_utc": now,
    }
    sync = {
        "schema": "poke_bot.slop_box_dashboard_active_sync/v1",
        "goal_revision": 170,
        "phase": "bootstrap:chao_hard_ce",
        "latest_line": line,
        "cox_chao_gate_passed": False,
        "cox_chao_gate_threshold": 0.9,
        "crustle_data_deleted": False,
        "bootstrap_restarted": False,
        "chao_hard_active": True,
        "chao_hard_epochs_completed": done,
        "chao_hard_epochs_planned": TARGET,
        "updated_at_utc": now,
        "created_at_utc": now,
    }
    hb = {
        "schema": "poke_bot.goalmd_loop_heartbeat_slop_box_r170/v1",
        "goal_revision": 170,
        "active_work": "chao_hard_ce_r170",
        "unit": UNIT,
        "epochs_completed": done,
        "epochs_planned": TARGET,
        "hot_start": "epoch_152",
        "main_pid": pid,
        "n_restarts": us.get("NRestarts"),
        "corpus": {
            "games": 1997,
            "decisions": 135774,
            "window": "2026-07-24..2026-07-30",
        },
        "upweight": {"chao": 25, "james_cox_train_only": 5},
        "cox_chao": {
            "status": "chao_hard_training_pending_held_select",
            "threshold": 0.9,
            "passed": False,
            "prior_measured_policy_acc": 0.7652127086698977,
        },
        "remat": {
            "status": "running_on_elmo",
            "expected_extra_games_approx": 702,
        },
        "crustle_abandoned_preserve": True,
        "no_ready": True,
        "no_rl": True,
        "ready_status": "failed_cox_chao_held_policy_acc_gate",
        "deep_ce_stopped": True,
        "updated_at_utc": now,
    }
    atomic(STATE / "slop-box-dashboard-bootstrap-projection-r170.json", proj)
    atomic(STATE / "slop-box-dashboard-active-sync-r170.json", sync)
    atomic(STATE / "goalmd-loop-heartbeat-slop-box-r170.json", hb)
    atomic(ROOT / "state" / "goalmd_loop_heartbeat_slop_box_r170.json", hb)
    rp = STATE / "slop-box-chao-hard-ce-run-r170.json"
    if rp.is_file():
        receipt = json.loads(rp.read_text())
        receipt["status"] = "training"
        receipt["epochs_completed"] = done
        receipt["updated_at_utc"] = now
        atomic(rp, receipt)


def main() -> int:
    start = time.time()
    last_done = -1
    print(
        f"[monitor] start {datetime.now().isoformat()} target={TARGET}",
        flush=True,
    )
    while True:
        us = unit_state()
        done = epochs_done()
        elapsed = time.time() - start
        if done != last_done:
            print(
                f"[monitor] t+{elapsed/60:.1f}m epoch={done}/{TARGET} "
                f"state={us.get('ActiveState')}/{us.get('SubState')} "
                f"pid={us.get('MainPID')} restarts={us.get('NRestarts')}",
                flush=True,
            )
            refresh_dash(done, us)
            last_done = done
        if us.get("ActiveState") == "failed" or us.get("Result") == "failed":
            print(f"[monitor] BLOCKER unit failed: {us}", flush=True)
            refresh_dash(done, us)
            break
        if us.get("ActiveState") == "inactive" and us.get("SubState") == "dead":
            print(
                f"[monitor] unit inactive/dead epochs={done} "
                f"result={us.get('Result')}",
                flush=True,
            )
            refresh_dash(done, us)
            break
        if elapsed > MAX_WAIT_S:
            print(f"[monitor] slice timeout at epoch={done}", flush=True)
            refresh_dash(done, us)
            break
        time.sleep(POLL)

    print(
        "FINAL",
        json.dumps({"epochs": epochs_done(), "unit": unit_state()}, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
