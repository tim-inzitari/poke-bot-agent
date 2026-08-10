#!/usr/bin/env python3
"""Patch train dashboard_snapshot.py so Slop Box H10 RTP bootstrap is live.

Dashboard-only: never restarts bootstrap/training units; preserves Crustle data.
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
    "/home/inzi/poke-bot-agent/outputs/state/slop-box-dashboard-bootstrap-projection-r170.json"
)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = PATH.with_name(PATH.name + f".pre-slop-box-dashboard-{stamp}")
    shutil.copy2(PATH, bak)

    const_old = """FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-bootstrap.service\"
)
"""
    const_new = """FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-bootstrap.service\"
)
FINAL_FORMAT_SLOP_BOX_H10_RTP_BOOTSTRAP_SERVICE = (
    \"pokebot-final-format-slop-box-h10-rtp-bootstrap.service\"
)
"""
    if "FINAL_FORMAT_SLOP_BOX_H10_RTP_BOOTSTRAP_SERVICE" not in text:
        if const_old not in text:
            raise SystemExit("const anchor missing")
        text = text.replace(const_old, const_new, 1)

    paths_old = """FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / \"outputs/state/marnie-canonical-training-freeze-r163.json\"
)
"""
    paths_new = """FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / \"outputs/state/marnie-canonical-training-freeze-r163.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR = (
    ROOT / \"outputs/bootstrap/final_format_slop_box_h10_rtp\"
)
FINAL_FORMAT_SLOP_BOX_H10_READY = (
    ROOT / \"outputs/state/final-format-slop-box-h10-rtp-bootstrap-ready.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_STARTED = (
    ROOT / \"outputs/state/slop-box-h10-rtp-bootstrap-started-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_IDENTITY = (
    ROOT / \"state/slop_box_h10_rtp_prestage_identity_r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_RTP_CUT = (
    ROOT / \"outputs/state/slop-box-h10-rtp-bootstrap-cut-r170.json\"
)
FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE = (
    ROOT / \"outputs/rtp_fleet/slop-box-h10.live/rtp_shadow_planner.pt\"
)
CRUSTLE_OWNER_ABANDON_R170 = (
    ROOT / \"outputs/state/crustle-owner-abandon-r170.json\"
)
"""
    if "FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR" not in text:
        if paths_old not in text:
            raise SystemExit("paths anchor missing")
        text = text.replace(paths_old, paths_new, 1)

    fn_marker = "def final_format_slop_box_progress() -> dict[str, Any]:"
    fn_insert_after = "def final_format_crustle_progress() -> dict[str, Any]:"
    fn_body = '''def final_format_slop_box_progress() -> dict[str, Any]:
    """Project managed Slop Box H10 expert bootstrap + initial RTP cut."""

    service = unit_state(
        FINAL_FORMAT_SLOP_BOX_H10_RTP_BOOTSTRAP_SERVICE, user=True
    )
    active = bool(
        (
            service.get("active")
            or service.get("active_state") == "activating"
        )
        and (
            int(service.get("pid") or 0) > 0
            or service.get("sub_state") in {"running", "start"}
        )
    )
    ready = read_json(FINAL_FORMAT_SLOP_BOX_H10_READY) or {}
    started = read_json(FINAL_FORMAT_SLOP_BOX_H10_STARTED) or {}
    identity = read_json(FINAL_FORMAT_SLOP_BOX_H10_IDENTITY) or {}
    state = read_json(FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json") or {}
    rtp_cut = read_json(FINAL_FORMAT_SLOP_BOX_H10_RTP_CUT) or {}
    if not (active or ready or started or state):
        return {"status": "waiting", "available": False}

    epochs_max = int(
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
        phase = "bootstrap:complete"
        status = "complete" if not active else "running"
        latest_line = (
            "Slop Box expert bootstrap + initial RTP cut complete"
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

    checkpoint = (
        ready.get("checkpoint")
        or state.get("best_path")
        or None
    )
    checkpoint_digest = (
        ready.get("checkpoint_digest")
        or state.get("best_digest")
        or None
    )
    if checkpoint_digest and not str(checkpoint_digest).startswith("sha256:"):
        checkpoint_digest = f"sha256:{checkpoint_digest}"
    structure = checkpoint_structure_telemetry(
        checkpoint,
        checkpoint_digest,
        cache_path=(
            ROOT
            / "outputs/state/dashboard-slop-box-h10-live-structure-cache.json"
        ),
    ) if checkpoint else {
        "available": False,
        "verified": False,
        "checkpoint": checkpoint,
        "checkpoint_digest": checkpoint_digest,
    }
    updated_candidates = [
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
    updated = max(updated_candidates) if updated_candidates else None
    specialist = dict((identity.get("specialist") or {}))
    specialist_id = str(
        specialist.get("logical_id") or "teal-mask-ogerpon-ex"
    ).strip().lower()
    percent = (
        100.0
        if expert_complete and rtp_cut_ready
        else (
            min(99.0, (float(epochs_completed) / float(epochs_max)) * 90.0 + 5.0)
            if expert_complete
            else (
                (float(epochs_completed) / float(epochs_max)) * 100.0
                if epochs_max
                else None
            )
        )
    )
    return {
        "available": True,
        "authoritative": True,
        "source": str(
            FINAL_FORMAT_SLOP_BOX_H10_READY
            if ready
            else FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"
        ),
        "log": str(FINAL_FORMAT_SLOP_BOX_H10_RUN_DIR / "state.json"),
        "latest_line": latest_line,
        "raw_latest_line": latest_line,
        "updated_at": updated,
        "fresh": bool(active),
        "status": status,
        "mode": "final_format_slop_box_h10_rtp_bootstrap",
        "phase": phase,
        "run": "final_format_slop_box_h10_rtp_bootstrap",
        "specialist_id": specialist_id,
        "display_name": specialist.get("display_name") or "Slop Box",
        "epoch": epochs_completed,
        "epochs_target": epochs_max,
        "iteration": epochs_completed,
        "iterations_target": epochs_max,
        "current": epochs_completed,
        "total": epochs_max,
        "percent": percent,
        "rate": None,
        "rate_unit": None,
        "games_per_second": 0.0,
        "samples_per_second": 0.0,
        "eta": None,
        "metrics": {
            "best_metric": ready.get("best_metric") or state.get("best_metric"),
            "expert_complete": expert_complete,
            "rtp_cut_ready": rtp_cut_ready,
            "rtp_live_checkpoint": str(FINAL_FORMAT_SLOP_BOX_H10_RTP_LIVE)
            if rtp_live
            else None,
        },
        "remote_workers": 0,
        "remote_endpoints": [],
        "scheduler_queues": {
            "available": False,
            "mode": "bootstrap" if active else "stopped",
            "local": {"active_or_claimed": 0},
            "endpoints": {},
            "unassigned": 0,
            "results": {"waiting_ingest": 0},
        },
        "checkpoint": checkpoint,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_structure": structure,
        "structure": structure,
        "service": service,
        "identity_source": str(FINAL_FORMAT_SLOP_BOX_H10_IDENTITY),
        "crustle_abandon_honored": CRUSTLE_OWNER_ABANDON_R170.is_file(),
    }


'''
    if fn_marker not in text:
        if fn_insert_after not in text:
            raise SystemExit("function insert anchor missing")
        text = text.replace(fn_insert_after, fn_body + fn_insert_after, 1)

    # Abandoned Crustle must not mask the live Slop Box selector as "stopped".
    crustle_guard_old = """    if not (bootstrap_active or rl_active or rl_history or ready):
        return {\"status\": \"waiting\", \"available\": False}

    # Live RL owns the display. Historical Marnie RL must never preempt an
    # active Crustle service, and bootstrap keeps ownership until its own
    # managed boundary completes.
    if rl_active or (rl_history and not bootstrap_active):
"""
    crustle_guard_new = """    if not (bootstrap_active or rl_active or rl_history or ready):
        return {\"status\": \"waiting\", \"available\": False}

    # Owner abandon (r170): keep Crustle artifacts immutable, but do not let a
    # stopped historical Crustle RL projection own the live dashboard selector.
    if (
        CRUSTLE_OWNER_ABANDON_R170.is_file()
        and not bootstrap_active
        and not rl_active
    ):
        return {\"status\": \"waiting\", \"available\": False, \"abandoned\": True}

    # Live RL owns the display. Historical Marnie RL must never preempt an
    # active Crustle service, and bootstrap keeps ownership until its own
    # managed boundary completes.
    if rl_active or (rl_history and not bootstrap_active):
"""
    if "CRUSTLE_OWNER_ABANDON_R170.is_file()" not in text:
        if crustle_guard_old not in text:
            raise SystemExit("crustle abandon guard anchor missing")
        text = text.replace(crustle_guard_old, crustle_guard_new, 1)

    sel_old = """        final_alakazam = final_format_alakazam_progress()
        final_marnie = final_format_marnie_progress()
        final_crustle = final_format_crustle_progress()
        postupload_bootstrap = marnie_postupload_bootstrap_state()
        postupload_family = marnie_postupload_family_study_state()
        postupload_boundary = (
            postupload_bootstrap
            if postupload_bootstrap.get(\"current\") is True
            else postupload_family
        )
        # Prefer an authoritative Crustle projection whenever it is available.
        # Historical stopped Marnie RL must not mask the active Crustle selector
        # during brief systemd transitions or after Marnie completion.
        if final_crustle.get(\"status\") in {\"running\", \"stopped\", \"complete\"}:
            active_final_refresh = final_crustle
        elif final_marnie.get(\"status\") in {\"running\", \"complete\", \"stopped\"}:
            active_final_refresh = final_marnie
        else:
            active_final_refresh = final_alakazam
"""
    sel_new = """        final_alakazam = final_format_alakazam_progress()
        final_marnie = final_format_marnie_progress()
        final_crustle = final_format_crustle_progress()
        final_slop_box = final_format_slop_box_progress()
        postupload_bootstrap = marnie_postupload_bootstrap_state()
        postupload_family = marnie_postupload_family_study_state()
        postupload_boundary = (
            postupload_bootstrap
            if postupload_bootstrap.get(\"current\") is True
            else postupload_family
        )
        # Slop Box H10 RTP bootstrap (r170) owns the live selector while active.
        # Abandoned/stopped Crustle must not mask it; do not fake Crustle green.
        if final_slop_box.get(\"status\") == \"running\":
            active_final_refresh = final_slop_box
        elif final_crustle.get(\"status\") == \"running\":
            active_final_refresh = final_crustle
        elif final_marnie.get(\"status\") == \"running\":
            active_final_refresh = final_marnie
        elif final_slop_box.get(\"status\") in {\"complete\", \"stopped\"}:
            active_final_refresh = final_slop_box
        elif final_crustle.get(\"status\") in {\"complete\", \"stopped\"}:
            active_final_refresh = final_crustle
        elif final_marnie.get(\"status\") in {\"complete\", \"stopped\"}:
            active_final_refresh = final_marnie
        else:
            active_final_refresh = final_alakazam
"""
    if "final_slop_box = final_format_slop_box_progress()" not in text:
        if sel_old not in text:
            raise SystemExit("selector anchor missing")
        text = text.replace(sel_old, sel_new, 1)

    PATH.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(PATH.read_bytes()).hexdigest()
    receipt = {
        "schema": "poke_bot.slop_box_dashboard_bootstrap_projection/v1",
        "goal_revision": 170,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "dashboard-only Slop Box bootstrap projection + abandon-aware selector",
        "trainer_mutated": False,
        "bootstrap_restarted": False,
        "crustle_data_deleted": False,
        "backup": str(bak),
        "deployed_source": str(PATH),
        "deployed_sha256": f"sha256:{digest}",
        "behavior": [
            "project pokebot-final-format-slop-box-h10-rtp-bootstrap.service",
            "prefer live Slop Box over abandoned/stopped Crustle",
            "keep Crustle artifacts when abandon receipt is present",
        ],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
