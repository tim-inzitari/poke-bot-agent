#!/usr/bin/env python3
"""Surface Cox/Chao gate-fail + RTP-ready Slop Box status on the dashboard.

Dashboard-only. Does not restart trainers or delete Crustle artifacts.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

PATH = Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py")
RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/slop-box-dashboard-gate-status-r170.json"
)

PATHS_OLD = """FINAL_FORMAT_SLOP_BOX_H10_RTP_CUT = (
    ROOT / \"outputs/state/slop-box-h10-rtp-bootstrap-cut-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE = (
    ROOT / \"outputs/rtp_fleet/slop-box-h10.live/rtp_shadow_planner.pt\"
)
"""

PATHS_NEW = """FINAL_FORMAT_SLOP_BOX_H10_RTP_CUT = (
    ROOT / \"outputs/state/slop-box-h10-rtp-bootstrap-cut-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE = (
    ROOT / \"outputs/rtp_fleet/slop-box-h10.live/rtp_shadow_planner.pt\"
)
FINAL_FORMAT_SLOP_BOX_H10_GATE = (
    ROOT / \"outputs/state/slop-box-cox-chao-held-policy-acc-gate-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_GATE_ARMED = (
    ROOT / \"outputs/state/slop-box-cox-chao-gate-armed-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_HEARTBEAT = (
    ROOT / \"outputs/state/goalmd-loop-heartbeat-slop-box-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_CE_SERVICE = (
    \"pokebot-final-format-slop-box-h10-rtp-ce-intensify.service\"
)
"""

OLD_BLOCK = '''    rtp_live = FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE.is_file()
    rtp_cut_ready = bool(rtp_cut) or rtp_live
    if active and expert_complete and not rtp_cut_ready:
        phase = "bootstrap:rtp_initial_cut"
        status = "running"
        latest_line = (
            "Slop Box expert bootstrap complete · initial RTP cut in progress"
        )
    elif active and not expert_complete:
        phase = "bootstrap:expert"
        status = "running"
        latest_line = (
            f"Slop Box expert bootstrap epoch {epochs_completed}/{epochs_max}"
        )
    elif expert_complete and rtp_cut_ready:
        # Oneshoot finished: report stopped (honest), not a fake live run.
        phase = "stopped:bootstrap:complete" if not active else "bootstrap:complete"
        status = "stopped" if not active else "running"
        latest_line = (
            "STOPPED · Slop Box expert bootstrap + initial RTP cut complete"
            if not active
            else "Slop Box bootstrap finishing managed boundary"
        )
    elif active:
        phase = "bootstrap:starting"
        status = "running"
        latest_line = "Slop Box H10 RTP bootstrap starting"
    else:
        phase = "bootstrap:waiting"
        status = "stopped"
        latest_line = "Slop Box H10 RTP bootstrap idle"
'''

NEW_BLOCK = '''    rtp_live = FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE.is_file()
    rtp_cut_ready = bool(rtp_cut) or rtp_live
    gate = read_json(FINAL_FORMAT_SLOP_BOX_H10_GATE) or {}
    gate_armed = read_json(FINAL_FORMAT_SLOP_BOX_H10_GATE_ARMED) or {}
    heartbeat = read_json(FINAL_FORMAT_SLOP_BOX_H10_HEARTBEAT) or {}
    ready_status = str(ready.get("status") or "").strip().lower()
    gate_measured = gate.get("cox_chao_held_policy_acc")
    if gate_measured is None:
        gate_measured = (ready.get("cox_chao_policy_accuracy_gate") or {}).get(
            "measured"
        )
    if gate_measured is None:
        gate_measured = (gate_armed.get("measured_cox_chao_held_policy_acc"))
    if gate_measured is None:
        gate_measured = ((heartbeat.get("cox_chao") or {}).get("measured_policy_acc"))
    try:
        gate_measured_f = float(gate_measured) if gate_measured is not None else None
    except (TypeError, ValueError):
        gate_measured_f = None
    gate_threshold = (
        gate.get("threshold")
        or (ready.get("cox_chao_policy_accuracy_gate") or {}).get("threshold")
        or gate_armed.get("threshold")
        or 0.9
    )
    try:
        gate_threshold_f = float(gate_threshold)
    except (TypeError, ValueError):
        gate_threshold_f = 0.9
    gate_failed = bool(
        ready_status == "failed_cox_chao_held_policy_acc_gate"
        or gate.get("passed") is False
        or gate_armed.get("passed") is False
        or (heartbeat.get("cox_chao") or {}).get("passed") is False
    )
    gate_passed = bool(
        gate.get("passed") is True
        or gate_armed.get("passed") is True
        or (heartbeat.get("cox_chao") or {}).get("passed") is True
    )
    ce_service = unit_state(FINAL_FORMAT_SLOP_BOX_H10_CE_SERVICE, user=True)
    ce_active = bool(
        (
            ce_service.get("active")
            or ce_service.get("active_state") == "activating"
        )
        and (
            int(ce_service.get("pid") or 0) > 0
            or ce_service.get("sub_state") in {"running", "start"}
        )
    )
    # Prefer a live CE-intensify / continue unit when present.
    if ce_active and not active:
        service = ce_service
        active = True
    measured_txt = (
        f"{gate_measured_f:.3f}" if gate_measured_f is not None else "n/a"
    )
    threshold_txt = f"{gate_threshold_f:.2f}"
    if active and expert_complete and not rtp_cut_ready:
        phase = "bootstrap:rtp_initial_cut"
        status = "running"
        latest_line = (
            "Slop Box expert bootstrap complete · initial RTP cut in progress"
        )
    elif active and not expert_complete and not gate_failed:
        phase = "bootstrap:expert"
        status = "running"
        latest_line = (
            f"Slop Box expert bootstrap epoch {epochs_completed}/{epochs_max}"
        )
    elif active and gate_failed:
        phase = "bootstrap:ce_intensify"
        status = "running"
        latest_line = (
            "RUNNING · Slop Box CE intensify toward Cox/Chao gate "
            f"{measured_txt}/{threshold_txt} · RTP cut ready · continuing"
        )
    elif (expert_complete or gate_failed) and rtp_cut_ready:
        # Bootstrap oneshot finished (possibly fail-closed on Cox/Chao).
        if gate_failed and not gate_passed:
            phase = "stopped:bootstrap:gate_failed"
            status = "stopped" if not active else "running"
            latest_line = (
                "STOPPED · Slop Box bootstrap complete · Cox/Chao gate FAILED "
                f"{measured_txt} < {threshold_txt} · RTP cut ready · "
                "awaiting CE intensify / continue"
            )
        else:
            phase = (
                "stopped:bootstrap:complete" if not active else "bootstrap:complete"
            )
            status = "stopped" if not active else "running"
            latest_line = (
                "STOPPED · Slop Box expert bootstrap + initial RTP cut complete"
                if not active
                else "Slop Box bootstrap finishing managed boundary"
            )
    elif active:
        phase = "bootstrap:starting"
        status = "running"
        latest_line = "Slop Box H10 RTP bootstrap starting"
    else:
        phase = "bootstrap:waiting"
        status = "stopped"
        latest_line = "Slop Box H10 RTP bootstrap idle"
'''

METRICS_OLD = '''        "metrics": {
            "best_metric": ready.get("best_metric") or state.get("best_metric"),
            "expert_complete": expert_complete,
            "rtp_cut_ready": rtp_cut_ready,
            "rtp_live_checkpoint": str(FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE)
            if rtp_live
            else None,
        },
'''

METRICS_NEW = '''        "metrics": {
            "best_metric": ready.get("best_metric") or state.get("best_metric"),
            "expert_complete": expert_complete,
            "rtp_cut_ready": rtp_cut_ready,
            "rtp_live_checkpoint": str(FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE)
            if rtp_live
            else None,
            "cox_chao_gate_failed": gate_failed and not gate_passed,
            "cox_chao_gate_passed": gate_passed,
            "cox_chao_held_policy_acc": gate_measured_f,
            "cox_chao_gate_threshold": gate_threshold_f,
            "ready_status": ready.get("status") or None,
            "ce_intensify_active": ce_active,
        },
'''

UPDATED_OLD = '''    updated_candidates = [
        path.stat().st_mtime
        for path in (
            FINAL_FORMAT_SLOP_BOX_H10_READY,
            FINAL_FORMAT_SLOP_BOX_H10_STARTED,
            FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json",
            FINAL_FORMAT_SLOP_BOX_H10_RTP_CUT,
            FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE,
        )
        if path.is_file()
    ]
'''

UPDATED_NEW = '''    updated_candidates = [
        path.stat().st_mtime
        for path in (
            FINAL_FORMAT_SLOP_BOX_H10_READY,
            FINAL_FORMAT_SLOP_BOX_H10_STARTED,
            FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json",
            FINAL_FORMAT_SLOP_BOX_H10_RTP_CUT,
            FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE,
            FINAL_FORMAT_SLOP_BOX_H10_GATE,
            FINAL_FORMAT_SLOP_BOX_H10_GATE_ARMED,
            FINAL_FORMAT_SLOP_BOX_H10_HEARTBEAT,
        )
        if path.is_file()
    ]
'''


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if "FINAL_FORMAT_SLOP_BOX_H10_GATE =" in text and "cox_chao_gate_failed" in text:
        print("already patched")
        return 0
    bak = PATH.with_name(PATH.name + f".pre-slop-box-gate-status-{stamp}")
    shutil.copy2(PATH, bak)
    if "FINAL_FORMAT_SLOP_BOX_H10_GATE =" not in text:
        if PATHS_OLD not in text:
            raise SystemExit("paths anchor missing")
        text = text.replace(PATHS_OLD, PATHS_NEW, 1)
    if OLD_BLOCK not in text:
        raise SystemExit("phase/status block missing")
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    if METRICS_OLD not in text:
        raise SystemExit("metrics anchor missing")
    text = text.replace(METRICS_OLD, METRICS_NEW, 1)
    if UPDATED_OLD not in text:
        raise SystemExit("updated_candidates anchor missing")
    text = text.replace(UPDATED_OLD, UPDATED_NEW, 1)
    PATH.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(PATH.read_bytes()).hexdigest()
    receipt = {
        "schema": "poke_bot.slop_box_dashboard_gate_status/v1",
        "goal_revision": 170,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "dashboard-only Slop Box Cox/Chao gate + RTP status projection",
        "trainer_mutated": False,
        "bootstrap_restarted": False,
        "crustle_data_deleted": False,
        "backup": str(bak),
        "deployed_source": str(PATH),
        "deployed_sha256": f"sha256:{digest}",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
