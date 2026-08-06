#!/usr/bin/env python3
"""Dynamic wake watcher for GOAL.md Crustle monitor."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

PROMPT = (
    "Monitor GOAL.md Crustle: read revision/objective; audit crustle systemd "
    "units on train including pokebot-crustle-rtp-r167; keep dashboard aligned "
    "to runtime truth (RTP override supersedes ordinary RL when its receipt "
    "exists); leave healthy training alone; coordinate with sibling recovery "
    "via live state; do not rewrite GOAL.md or disrupt git/PR work."
)
STATE_DIR = Path("/Users/tsinzitari/Documents/poke-agent-codex/.cursor-loop")
GOAL_LOCAL = Path("/Users/tsinzitari/Documents/poke-agent-codex/GOAL.md")
PREV_FILE = STATE_DIR / "goalmd_watcher_prev.fp"
INTERVAL = 20

REMOTE_FP_PY = r'''
import glob, json, os, re, subprocess, time
from pathlib import Path

def unit_fp(name):
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "show", name,
             "-p", "ActiveState", "-p", "SubState", "-p", "Result",
             "-p", "NRestarts", "-p", "MainPID", "-p", "ExecMainStatus",
             "--no-pager"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return f"{name}=err"
    kv = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k] = v
    # Coarse service fingerprint: ignore MainPID churn during auto-restart.
    sub = kv.get("SubState", "?")
    if sub in {"auto-restart", "start-pre", "start", "stop-sigterm", "final-sigterm"}:
        sub_bucket = "starting_or_stopping"
    else:
        sub_bucket = sub
    return (
        f"{name}={kv.get('ActiveState','?')}/{sub_bucket}"
        f":r{kv.get('NRestarts','?')}"
        f":res{kv.get('Result','?')}:ems{kv.get('ExecMainStatus','?')}"
    )

units = [
    "pokebot-crustle-rtp-r167.service",
    "pokebot-final-format-crustle-r113-h10-rl.service",
    "pokebot-final-format-crustle-r113-h10-bootstrap.service",
    "pokebot-final-format-crustle-r113-h10-register.service",
    "pokebot-final-format-crustle-r113-h10-gate-handler.service",
    "pokebot-final-format-crustle-r113-completion.service",
]
parts = [unit_fp(u) for u in units]

cands = []
# Gate handler state is polled every ~15s; phase/pid are tracked separately.
# Exclude it so receipt mtime churn does not wake-storm the resident loop.
_skip_receipts = {"crustle-passed-gate-handler-v1.json"}
for path in glob.glob("/home/inzi/poke-bot-agent/outputs/state/*crustle*"):
    try:
        base = os.path.basename(path)
        if base in _skip_receipts:
            continue
        st = os.stat(path)
        if os.path.isfile(path):
            cands.append((st.st_mtime, st.st_size, base))
    except OSError:
        pass
cands.sort(reverse=True)
top = ",".join(f"{n}:{int(m)}:{s}" for m, s, n in cands[:8])
parts.append("receipts=" + (top or "none"))

gate = Path("/home/inzi/poke-bot-agent/outputs/state/crustle-passed-gate-handler-v1.json")
if gate.exists():
    try:
        d = json.loads(gate.read_text())
        # Omit volatile updated_at; wake on phase / observed trainer PID only.
        parts.append(
            f"gate_phase={d.get('phase')}|gate_pid={d.get('observed_training_pid')}"
        )
    except Exception:
        parts.append("gate_phase=unreadable")

dash_mtime = 0
dash_name = "none"
for path in glob.glob("/home/inzi/poke-bot-agent/outputs/state/*dashboard*"):
    try:
        st = os.stat(path)
        if st.st_mtime > dash_mtime:
            dash_mtime = st.st_mtime
            dash_name = os.path.basename(path)
    except OSError:
        pass
# Coarse freshness only: wake on new/replaced dashboard artifacts, not clock drift.
parts.append(f"dash={dash_name}:mt{int(dash_mtime)}")

# Prefer RTP override fingerprint when that managed unit owns training.
rtp_svc = unit_fp("pokebot-crustle-rtp-r167.service")
rtp_active = "active/running" in rtp_svc
rtp_override = Path(
    "/home/inzi/poke-bot-agent/outputs/state/crustle-rtp-training-override-r167.json"
)
rtp_receipt = Path(
    "/home/inzi/poke-bot-agent/outputs/rtp_fleet/crustle/rtp/"
    "rtp_shadow_planner.pt.receipt.json"
)
if rtp_override.exists():
    epoch = "?"
    if rtp_receipt.exists():
        try:
            rd = json.loads(rtp_receipt.read_text())
            epoch = (rd.get("metrics") or {}).get("epoch", "?")
        except Exception:
            epoch = "unreadable"
    parts.append(
        f"rtp_alive={rtp_active}|rtp_epoch={epoch}|rtp_override=1"
    )
    if rtp_active:
        parts.append("run_alive=True|run_fail=ok_or_progress")
    else:
        parts.append("run_alive=False|run_fail=rtp_override_inactive")
else:
    # Ordinary RL run-dir progress marker (ignore volatile monitor mtime / pid churn)
    run = Path(
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "final_format_crustle_r113_h10_i_v6_8k"
    )
    mon = run / "monitor_state.json"
    svc = unit_fp("pokebot-final-format-crustle-r113-h10-rl.service")
    svc_active = "active/running" in svc
    if mon.exists():
        try:
            d = json.loads(mon.read_text())
            alive = bool(d.get("process_alive")) and svc_active
            line = ((d.get("progress") or {}).get("last_line") or "")[:160]
            stable = line
            if "frozen specialist packages are unavailable" in line:
                stable = "frozen_specialist_packages_unavailable"
            elif "strong-public package digest mismatch" in line:
                stable = "strong_public_digest_mismatch"
            elif "checkpoint does not match exact intended pure-RL model profile" in line:
                stable = "checkpoint_profile_mismatch"
            elif "BetweenIterSyncError" in line or "between-iter hard-gate remote sync failed" in line:
                stable = "between_iter_remote_sync_failed"
            elif "design migration requires a completed N+1 boundary" in line:
                stable = "design_migration_boundary"
            elif "free-disk guard refused" in line:
                stable = "free_disk_guard"
            elif "baseline id is absent from local manifest" in line or "baseline_failed" in line:
                stable = "remote_baseline_missing"
            elif "model has" in line and "params >" in line:
                stable = "param_budget_fail_closed"
            elif line.startswith("[pure_rl]"):
                # informational seed/write lines are not fail classes
                stable = "info:" + line.split(" ", 1)[-1][:80]
            parts.append(
                f"run_alive={alive}|run_fail={stable if not alive else 'ok_or_progress'}"
            )
        except Exception:
            parts.append("run_mon=unreadable")
    else:
        parts.append(f"run_exists={run.exists()}|run_mon=missing")

g = Path("/home/inzi/poke-bot-agent/GOAL.md")
if g.exists():
    text = g.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^Revision:\s*`([^`]+)`", text, re.M)
    parts.append(
        f"remote_goal_rev={m.group(1) if m else '?'}|remote_goal_mt={int(g.stat().st_mtime)}"
    )

print("|".join(parts))
'''


def local_fp() -> str:
    text = GOAL_LOCAL.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^Revision:\s*`([^`]+)`", text, re.M)
    rev = m.group(1) if m else "?"
    st = GOAL_LOCAL.stat()
    return f"goal_rev={rev}|goal_mtime={int(st.st_mtime)}|goal_size={st.st_size}"


def remote_fp() -> str:
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=12",
                "train",
                "python3",
                "-",
            ],
            input=REMOTE_FP_PY,
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    except Exception as exc:
        return f"remote_err={type(exc).__name__}:{exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:200]
        return f"remote_err=rc{proc.returncode}:{err}"
    return (proc.stdout or "").strip() or "remote_empty"


def fingerprint() -> str:
    return f"{local_fp()}|{remote_fp()}"


def emit_wake(prev: str, cur: str, reason: str) -> None:
    payload = {
        "prompt": PROMPT,
        "reason": reason,
        "prev_tail": prev[-240:],
        "cur_tail": cur[-240:],
    }
    print("AGENT_LOOP_WAKE_goalmd " + json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    prev = fingerprint()
    PREV_FILE.write_text(prev, encoding="utf-8")
    print("WATCHER_ARMED goalmd_crustle baseline_ready", flush=True)
    while True:
        time.sleep(INTERVAL)
        cur = fingerprint()
        if cur != prev:
            emit_wake(prev, cur, "material_fingerprint_change")
            prev = cur
            PREV_FILE.write_text(prev, encoding="utf-8")


if __name__ == "__main__":
    main()
