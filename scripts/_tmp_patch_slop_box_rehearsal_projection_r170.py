#!/usr/bin/env python3
"""Update Slop Box dash projection for CE-intensify continue-to-90% rehearsal."""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

TARGETS = [
    Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py"),
    Path(
        "/home/inzi/poke-bot-agent-deployments/final-format-marnie-postupload-r136/"
        "scripts/dashboard_snapshot.py"
    ),
    Path(
        "/home/inzi/poke-bot-agent-deployments/state-core-v1/scripts/"
        "dashboard_snapshot.py"
    ),
    Path(
        "/home/inzi/poke-bot-agent-deployments/"
        "pure-rl-resident-v41-specialist-matchup-runtime/scripts/"
        "dashboard_snapshot.py"
    ),
]

OLD = '''    epochs_max = int(
        ready.get("epochs_max")
        or state.get("epochs_max")
        or 25
    )
    epochs_completed = int(
        ready.get("epochs_completed")
        or len(state.get("history") or [])
        or 0
    )
    expert_complete = bool(
        str(ready.get("status") or "").lower() in {"ready", "complete"}
        or str(state.get("status") or "").lower() == "complete"
        or epochs_completed >= epochs_max > 0
    )
    rtp_live = FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE.is_file()
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

NEW = '''    epochs_max = int(
        ready.get("epochs_max")
        or state.get("epochs_max")
        or 25
    )
    epochs_completed = int(
        ready.get("epochs_completed")
        or len(state.get("history") or [])
        or 0
    )
    gate = dict(
        ready.get("cox_chao_policy_accuracy_gate")
        or {}
    )
    gate_receipt = read_json(
        ROOT / "outputs/state/slop-box-cox-chao-held-policy-acc-gate-r170.json"
    ) or {}
    gate_armed = read_json(
        ROOT / "outputs/state/slop-box-cox-chao-gate-armed-r170.json"
    ) or {}
    if not gate and gate_receipt:
        gate = {
            "measured": gate_receipt.get("cox_chao_held_policy_acc"),
            "threshold": 0.9,
            "passed": gate_receipt.get("passed"),
            "fail_closed": gate_receipt.get("fail_closed"),
        }
    gate_measured = gate.get("measured")
    if gate_measured is None:
        gate_measured = gate_armed.get("measured_cox_chao_held_policy_acc")
    try:
        gate_measured_f = float(gate_measured) if gate_measured is not None else None
    except (TypeError, ValueError):
        gate_measured_f = None
    gate_passed = bool(gate.get("passed") is True or gate_armed.get("passed") is True)
    gate_failed = bool(
        gate_passed is False
        and (
            gate.get("passed") is False
            or gate_armed.get("passed") is False
            or str(ready.get("status") or "").startswith("failed_cox_chao")
            or (
                ROOT
                / "outputs/state/"
                "final-format-slop-box-h10-rtp-bootstrap-ready-FAILED-cox-chao-gate-r170.json"
            ).is_file()
        )
    )
    continue_rehearsal = bool(
        active
        and (
            gate_failed
            or epochs_max > 25
            or epochs_completed > 25
            or str(state.get("status") or "").lower()
            in {"running", "training", "continue", "rehearsal"}
        )
    )
    expert_complete = bool(
        str(ready.get("status") or "").lower() in {"ready", "complete"}
        or str(state.get("status") or "").lower() == "complete"
        or (epochs_completed >= 25 and epochs_max >= 25 and not continue_rehearsal)
    )
    rtp_live = FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE.is_file()
    rtp_cut_ready = bool(rtp_cut) or rtp_live
    gate_txt = (
        f"Cox/Chao held policy_acc={gate_measured_f:.3f}/0.900"
        if gate_measured_f is not None
        else "Cox/Chao held policy_acc gate"
    )
    if continue_rehearsal:
        phase = "rehearsal:ce_intensify_to_90"
        status = "running"
        latest_line = (
            f"Slop Box CE intensify rehearsal · epoch "
            f"{epochs_completed}/{epochs_max} · {gate_txt} · target ≥0.90"
        )
    elif active and expert_complete and not rtp_cut_ready:
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
        if gate_failed and not active:
            phase = "stopped:bootstrap:gate_failed_continue_pending"
            status = "stopped"
            latest_line = (
                f"STOPPED · Slop Box bootstrap+RTP done · {gate_txt} FAILED · "
                "continue-to-90% rehearsal pending"
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


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in TARGETS:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if "rehearsal:ce_intensify_to_90" in text:
            print(f"already {path}")
            continue
        if OLD not in text:
            raise SystemExit(f"anchor missing in {path}")
        bak = path.with_name(path.name + f".pre-rehearsal-proj-{stamp}")
        shutil.copy2(path, bak)
        text = text.replace(OLD, NEW, 1)
        # Prefer rehearsal mode label when CE intensify is live.
        mode_old = '''        "mode": "final_format_slop_box_h10_rtp_bootstrap",
        "phase": phase,
        "run": "final_format_slop_box_h10_rtp_bootstrap",
'''
        mode_new = '''        "mode": (
            "final_format_slop_box_h10_rtp_ce_intensify"
            if continue_rehearsal
            else "final_format_slop_box_h10_rtp_bootstrap"
        ),
        "phase": phase,
        "run": (
            "final_format_slop_box_h10_rtp_ce_intensify"
            if continue_rehearsal
            else "final_format_slop_box_h10_rtp_bootstrap"
        ),
'''
        if "final_format_slop_box_h10_rtp_ce_intensify" not in text:
            if mode_old not in text:
                raise SystemExit(f"mode anchor missing in {path}")
            text = text.replace(mode_old, mode_new, 1)
        path.write_text(text, encoding="utf-8")
        print(f"patched {path} bak={bak.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
