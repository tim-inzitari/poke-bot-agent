#!/usr/bin/env python3
"""Repeatedly keep the LAN dash aligned to sibling Crustle runtime truth."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

STATE_DIR = Path("/Users/tsinzitari/Documents/poke-agent-codex/.cursor-loop")
STATE_FILE = STATE_DIR / "dash_sync_loop_state.json"
LOG_FILE = STATE_DIR / "dash_sync_loop.log"
STOP_FILE = STATE_DIR / "dash_sync_loop.stop"
INTERVAL_S = 45

REMOTE_PY = r'''
import json, importlib.util, glob, os, subprocess
from pathlib import Path

def unit(name):
    out = subprocess.check_output(
        ["systemctl", "--user", "show", name,
         "-p", "ActiveState", "-p", "SubState", "-p", "Result",
         "-p", "MainPID", "-p", "NRestarts", "--no-pager"],
        text=True,
    )
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)

rtp = unit("pokebot-crustle-rtp-r167.service")
cotrain = unit("pokebot-crustle-rtp-cotrain-r167.service")
rl = unit("pokebot-final-format-crustle-r113-h10-rl.service")
override = Path("/home/inzi/poke-bot-agent/outputs/state/crustle-rtp-training-override-r167.json")
cotrain_receipt = Path("/home/inzi/poke-bot-agent/outputs/state/crustle-rtp-rl-cotrain-r167.json")
summary = Path("/home/inzi/poke-bot-agent/outputs/rtp_fleet/crustle/pipeline_summary.json")
receipt = Path("/home/inzi/poke-bot-agent/outputs/rtp_fleet/crustle/rtp/rtp_shadow_planner.pt.receipt.json")
log = Path("/home/inzi/poke-bot-agent/outputs/rtp_fleet/crustle_rtp_r167.log")
snap = Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py")
text = snap.read_text(encoding="utf-8", errors="replace")
has_rtp = "def final_format_crustle_rtp_progress(" in text
has_cotrain = "def crustle_rtp_cotrain_sidecar(" in text
spec = importlib.util.spec_from_file_location("ds", snap)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
prog = mod.final_format_crustle_progress()
# newest crustle/dashboard receipt names
cands = []
for path in glob.glob("/home/inzi/poke-bot-agent/outputs/state/*crustle*") + glob.glob(
    "/home/inzi/poke-bot-agent/outputs/state/*dashboard*"
):
    try:
        st = os.stat(path)
        if os.path.isfile(path):
            cands.append((st.st_mtime, os.path.basename(path)))
    except OSError:
        pass
cands.sort(reverse=True)
payload = {
    "rtp": rtp,
    "cotrain": cotrain,
    "rl": rl,
    "override_exists": override.exists(),
    "override_mtime": int(override.stat().st_mtime) if override.exists() else None,
    "cotrain_receipt_exists": cotrain_receipt.exists(),
    "cotrain_receipt_mtime": int(cotrain_receipt.stat().st_mtime) if cotrain_receipt.exists() else None,
    "summary_exists": summary.exists(),
    "summary_mtime": int(summary.stat().st_mtime) if summary.exists() else None,
    "receipt_exists": receipt.exists(),
    "receipt_mtime": int(receipt.stat().st_mtime) if receipt.exists() else None,
    "log_size": log.stat().st_size if log.exists() else 0,
    "has_rtp_helper": has_rtp,
    "has_cotrain_helper": has_cotrain,
    "proj": {
        "status": prog.get("status"),
        "mode": prog.get("mode"),
        "phase": prog.get("phase"),
        "iteration": prog.get("iteration"),
        "percent": prog.get("percent"),
        "latest_line": (prog.get("latest_line") or "")[:180],
        "run": prog.get("run"),
        "svc_name": (prog.get("service") or {}).get("name"),
        "svc_active": (prog.get("service") or {}).get("active"),
        "svc_pid": (prog.get("service") or {}).get("pid"),
        "rtp_cotrain": ((prog.get("metrics") or {}).get("rtp_cotrain") or {}),
    },
    "top_receipts": [{"mtime": int(m), "name": n} for m, n in cands[:6]],
}
print(json.dumps(payload, separators=(",", ":")))
'''


def remote_truth() -> dict:
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "train",
            "python3",
            "-",
        ],
        input=REMOTE_PY,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "error": f"rc{proc.returncode}",
            "stderr": (proc.stderr or proc.stdout or "")[:300],
        }
    return json.loads(proc.stdout)


def dash_status() -> dict:
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            "60",
            "http://127.0.0.1:8780/api/status",
        ],
        text=True,
        capture_output=True,
        timeout=70,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"error": (proc.stderr or "empty")[:200]}
    d = json.loads(proc.stdout)
    cur = d.get("curriculum") or {}
    ff = cur.get("final_format_refresh") or {}
    return {
        "active": cur.get("active"),
        "stage": cur.get("stage"),
        "run": cur.get("run"),
        "units": cur.get("active_units"),
        "pids": cur.get("active_pids"),
        "status": ff.get("status"),
        "mode": ff.get("mode"),
        "phase": ff.get("phase"),
        "line": (ff.get("latest_line") or "")[:180],
        "svc_name": (ff.get("service") or {}).get("name"),
        "svc_pid": (ff.get("service") or {}).get("pid"),
        "svc_active": (ff.get("service") or {}).get("active"),
    }


def fingerprint(truth: dict, dash: dict) -> str:
    p = truth.get("proj") or {}
    rtp = truth.get("rtp") or {}
    rl = truth.get("rl") or {}
    ct = truth.get("cotrain") or {}
    return "|".join(
        [
            f"rl={rl.get('ActiveState')}/{rl.get('SubState')}:{rl.get('MainPID')}",
            f"rtp={rtp.get('ActiveState')}/{rtp.get('SubState')}:{rtp.get('MainPID')}",
            f"ct={ct.get('ActiveState')}/{ct.get('SubState')}:{ct.get('MainPID')}",
            f"ov={truth.get('override_mtime')}",
            f"ctr={truth.get('cotrain_receipt_mtime')}",
            f"sum={truth.get('summary_mtime')}",
            f"rec={truth.get('receipt_mtime')}",
            f"helper={truth.get('has_rtp_helper')}/{truth.get('has_cotrain_helper')}",
            f"proj={p.get('status')}/{p.get('mode')}/{p.get('phase')}/{p.get('svc_pid')}",
            f"dash={dash.get('status')}/{dash.get('mode')}/{dash.get('stage')}/{dash.get('svc_pid')}",
        ]
    )


def needs_repatch(truth: dict) -> bool:
    if truth.get("error"):
        return False
    if not truth.get("has_rtp_helper") and truth.get("override_exists"):
        return True
    if truth.get("cotrain_receipt_exists") and not truth.get("has_cotrain_helper"):
        return True
    mode = (truth.get("proj") or {}).get("mode")
    if truth.get("override_exists") and mode != "crustle_rtp_r167":
        return True
    return False


def repatch() -> str:
    # Prefer cotrain sidecar patch when concurrent receipt exists; else exclusive RTP.
    cotrain = Path(
        "/Users/tsinzitari/Documents/poke-agent-codex/scripts/"
        "_tmp_patch_crustle_dashboard_cotrain_r167.py"
    )
    rtp = Path(
        "/Users/tsinzitari/Documents/poke-agent-codex/scripts/"
        "_tmp_patch_crustle_dashboard_rtp_r167.py"
    )
    local = cotrain if cotrain.exists() else rtp
    remote = f"/tmp/{local.name}"
    subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            str(local),
            f"train:{remote}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            "train",
            f"/home/inzi/miniconda3/envs/poke-bot-agent/bin/python {remote}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (proc.stdout or proc.stderr or "")[-400:]

def write_state(payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload.get("last_pass") or {}, separators=(",", ":")) + "\n")


def one_pass(prev_fp: str | None) -> tuple[str, dict]:
    truth = remote_truth()
    dash = dash_status()
    action = "noop"
    detail = ""
    if needs_repatch(truth):
        action = "repatch"
        detail = repatch()
        truth = remote_truth()
        time.sleep(2)
        dash = dash_status()
    fp = fingerprint(truth, dash)
    drift = False
    if not truth.get("error") and not dash.get("error"):
        p = truth.get("proj") or {}
        drift = (
            dash.get("mode") != p.get("mode")
            or dash.get("status") != p.get("status")
            or str(dash.get("svc_pid") or 0) != str(p.get("svc_pid") or 0)
        )
        if drift and action == "noop":
            # Projection is correct on train; wait for Mac cache poll (~1s).
            time.sleep(2)
            dash = dash_status()
            fp = fingerprint(truth, dash)
            drift = (
                dash.get("mode") != p.get("mode")
                or dash.get("status") != p.get("status")
                or str(dash.get("svc_pid") or 0) != str(p.get("svc_pid") or 0)
            )
            if drift:
                action = "dash_lag"
    changed = prev_fp is not None and fp != prev_fp
    summary = {
        "ts": int(time.time()),
        "fp": fp,
        "changed": changed,
        "action": action,
        "drift": drift,
        "rl": (truth.get("rl") or {}),
        "rtp": (truth.get("rtp") or {}),
        "cotrain": (truth.get("cotrain") or {}),
        "proj": (truth.get("proj") or {}),
        "dash": dash,
        "detail": detail[:200],
    }
    if changed:
        print(
            "AGENT_LOOP_WAKE_dash "
            + json.dumps(
                {
                    "prompt": (
                        "Keep dash aligned to sibling Crustle/RTP runtime truth; "
                        "leave healthy training alone; do not rewrite GOAL.md."
                    ),
                    "reason": "material_fingerprint_change",
                    "summary": summary,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    print(
        f"DASH_SYNC pass action={action} drift={drift} "
        f"rtp={summary['rtp'].get('ActiveState')}/{summary['rtp'].get('MainPID')} "
        f"dash={dash.get('status')}/{dash.get('mode')}/{dash.get('svc_pid')}",
        flush=True,
    )
    return fp, summary


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    prev = None
    if STATE_FILE.exists():
        try:
            prev = (json.loads(STATE_FILE.read_text()).get("fingerprint"))
        except Exception:
            prev = None
    print("DASH_SYNC_ARMED interval=%ss" % INTERVAL_S, flush=True)
    while True:
        if STOP_FILE.exists():
            print("DASH_SYNC_STOP requested", flush=True)
            STOP_FILE.unlink(missing_ok=True)
            break
        try:
            fp, summary = one_pass(prev)
            prev = fp
            write_state(
                {
                    "schema": "poke_bot.cursor_loop_dash_sync/v1",
                    "fingerprint": fp,
                    "updated_at_unix": int(time.time()),
                    "interval_s": INTERVAL_S,
                    "last_pass": summary,
                }
            )
            # Also mirror into goalmd loop state briefly.
            goal_state = STATE_DIR / "goalmd_loop_state.json"
            if goal_state.exists():
                try:
                    gs = json.loads(goal_state.read_text())
                    gs["dash_sync"] = {
                        "updated_at_unix": int(time.time()),
                        "fingerprint": fp,
                        "last_pass": {
                            k: summary.get(k)
                            for k in (
                                "action",
                                "drift",
                                "rtp",
                                "rl",
                                "cotrain",
                                "proj",
                                "dash",
                            )
                        },
                    }
                    goal_state.write_text(json.dumps(gs, indent=2) + "\n")
                except Exception:
                    pass
        except Exception as exc:
            print(f"DASH_SYNC_ERR {type(exc).__name__}:{exc}", flush=True)
        for _ in range(INTERVAL_S):
            if STOP_FILE.exists():
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
