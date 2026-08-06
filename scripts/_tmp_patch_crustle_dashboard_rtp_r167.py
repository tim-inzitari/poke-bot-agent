#!/usr/bin/env python3
"""Patch train dashboard_snapshot.py so Crustle RTP owns the live projection."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

PATHS = [
    Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py"),
    Path(
        "/home/inzi/poke-bot-agent-deployments/"
        "final-format-marnie-postupload-r136/scripts/dashboard_snapshot.py"
    ),
]
CGR = Path("/home/inzi/poke-bot-agent/ops/current_goal_requirements.json")
RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/crustle-dashboard-rtp-projection-r167.json"
)
LOCAL_RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "crustle_dashboard_rtp_projection_r167.json"
)

CONST_OLD = """FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-rl.service\"
)
"""
CONST_NEW = """FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-rl.service\"
)
FINAL_FORMAT_CRUSTLE_RTP_SERVICE = \"pokebot-crustle-rtp-r167.service\"
"""

PATHS_OLD = """FINAL_FORMAT_CRUSTLE_H10_READY = (
    ROOT / \"outputs/state/final-format-crustle-r113-h10-bootstrap-ready.json\"
)
FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / \"outputs/state/marnie-canonical-training-freeze-r163.json\"
)
"""
PATHS_NEW = """FINAL_FORMAT_CRUSTLE_H10_READY = (
    ROOT / \"outputs/state/final-format-crustle-r113-h10-bootstrap-ready.json\"
)
FINAL_FORMAT_CRUSTLE_RTP_OVERRIDE = (
    ROOT / \"outputs/state/crustle-rtp-training-override-r167.json\"
)
FINAL_FORMAT_CRUSTLE_RTP_LOG = ROOT / \"outputs/rtp_fleet/crustle_rtp_r167.log\"
FINAL_FORMAT_CRUSTLE_RTP_RECEIPT = (
    ROOT / \"outputs/rtp_fleet/crustle/rtp/rtp_shadow_planner.pt.receipt.json\"
)
FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / \"outputs/state/marnie-canonical-training-freeze-r163.json\"
)
"""

FN_OLD = """def final_format_crustle_progress() -> dict[str, Any]:
    \"\"\"Project the managed Crustle H10 bootstrap or active RL run.\"\"\"

    bootstrap_service = unit_state(
        FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE,
        user=True,
    )
    rl_service = unit_state(FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE, user=True)
"""

# Also accept the docstring variant already present on some hosts.
FN_OLD_ALT = """def final_format_crustle_progress() -> dict[str, Any]:
    \"\"\"Project managed Crustle H10 bootstrap or active RL collection.\"\"\"

    bootstrap_service = unit_state(
        FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE, user=True
    )
    rl_service = unit_state(FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE, user=True)
"""

FN_HELPER = '''def final_format_crustle_rtp_progress() -> dict[str, Any] | None:
    """Project owner-authorized Crustle RTP when its override receipt exists."""

    override = read_json(FINAL_FORMAT_CRUSTLE_RTP_OVERRIDE)
    if not override:
        return None
    rtp_service = unit_state(FINAL_FORMAT_CRUSTLE_RTP_SERVICE, user=True)
    rtp_active = bool(
        rtp_service.get("active")
        and (
            int(rtp_service.get("pid") or 0) > 0
            or rtp_service.get("sub_state") in {"running", "start"}
        )
    )
    receipt = read_json(FINAL_FORMAT_CRUSTLE_RTP_RECEIPT) or {}
    metrics = dict(receipt.get("metrics") or {})
    log_tail = read_tail(FINAL_FORMAT_CRUSTLE_RTP_LOG, 20_000).strip()
    log_line = ""
    if log_tail:
        for line in reversed(log_tail.splitlines()):
            text = line.strip()
            if text:
                log_line = text[:240]
                break
    epoch = metrics.get("epoch")
    mean_loss = metrics.get("mean_loss")
    n_steps = metrics.get("n_steps")
    parameters = int(receipt.get("parameters") or 0)
    if log_line:
        latest_line = log_line
    elif metrics:
        latest_line = (
            f"RTP shadow train epoch={epoch} mean_loss={mean_loss} "
            f"n_steps={n_steps} params={parameters}"
        )
    else:
        latest_line = str(
            override.get("reason") or "Crustle RTP training override active"
        )
    if not rtp_active:
        latest_line = f"STOPPED · last progress: {latest_line}"
    updated_candidates = [
        path.stat().st_mtime
        for path in (
            FINAL_FORMAT_CRUSTLE_RTP_OVERRIDE,
            FINAL_FORMAT_CRUSTLE_RTP_RECEIPT,
            FINAL_FORMAT_CRUSTLE_RTP_LOG,
        )
        if path.is_file()
    ]
    updated = max(updated_candidates) if updated_candidates else None
    checkpoint = receipt.get("checkpoint_path") or override.get("parent_checkpoint")
    checkpoint_digest = override.get("parent_digest")
    phase = "train:rtp_shadow" if rtp_active else "stopped:train:rtp_shadow"
    percent = None
    if epoch is not None:
        try:
            percent = min(100.0, max(0.0, (float(epoch) / 4.0) * 100.0))
        except (TypeError, ValueError):
            percent = None
    return {
        "available": True,
        "authoritative": True,
        "source": str(FINAL_FORMAT_CRUSTLE_RTP_OVERRIDE),
        "log": str(FINAL_FORMAT_CRUSTLE_RTP_LOG),
        "latest_line": latest_line,
        "raw_latest_line": log_line or latest_line,
        "updated_at": updated,
        "fresh": bool(rtp_active and updated and time.time() - updated < 15 * 60),
        "status": "running" if rtp_active else "stopped",
        "mode": "crustle_rtp_r167",
        "phase": phase,
        "run": "crustle_rtp_r167",
        "specialist_id": "crustle",
        "iteration": int(epoch) if epoch is not None else 0,
        "epoch": epoch,
        "iterations_target": 4,
        "current": int(n_steps) if n_steps is not None else None,
        "total": None,
        "percent": percent,
        "rate": None,
        "rate_unit": None,
        "games_per_second": 0.0,
        "samples_per_second": 0.0,
        "eta": None,
        "metrics": {
            "mean_loss": mean_loss,
            "n_steps": n_steps,
            "parameters": parameters,
            "serving_eligible": receipt.get("serving_eligible"),
            "override_active_training": override.get("active_training"),
            "baseline_fleet_sync": (
                "outputs/state/crustle-h10-marnie-baseline-fleet-sync-r167.json"
            ),
        },
        "remote_workers": 0,
        "remote_endpoints": [],
        "scheduler_queues": {
            "available": False,
            "mode": "rtp_local_cuda",
            "local": {"active_or_claimed": 1 if rtp_active else 0},
            "endpoints": {},
            "unassigned": 0,
            "results": {"waiting_ingest": 0},
        },
        "model_parameters": parameters,
        "checkpoint": checkpoint,
        "checkpoint_digest": checkpoint_digest,
        "structure": {},
        "service": rtp_service,
        "rtp_override": override,
    }


'''

FN_NEW = (
    FN_HELPER
    + """def final_format_crustle_progress() -> dict[str, Any]:
    \"\"\"Project managed Crustle H10 bootstrap, RTP override, or RL run.\"\"\"

    rtp_progress = final_format_crustle_rtp_progress()
    if rtp_progress is not None:
        return rtp_progress

    bootstrap_service = unit_state(
        FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE,
        user=True,
    )
    rl_service = unit_state(FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE, user=True)
"""
)

FN_NEW_ALT = (
    FN_HELPER
    + """def final_format_crustle_progress() -> dict[str, Any]:
    \"\"\"Project managed Crustle H10 bootstrap, RTP override, or RL collection.\"\"\"

    rtp_progress = final_format_crustle_rtp_progress()
    if rtp_progress is not None:
        return rtp_progress

    bootstrap_service = unit_state(
        FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE, user=True
    )
    rl_service = unit_state(FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE, user=True)
"""
)


def patch_text(text: str) -> str:
    if "def final_format_crustle_rtp_progress(" in text:
        return text
    if "FINAL_FORMAT_CRUSTLE_RTP_SERVICE" not in text:
        if CONST_OLD not in text:
            raise SystemExit("const anchor missing")
        text = text.replace(CONST_OLD, CONST_NEW, 1)
    if "FINAL_FORMAT_CRUSTLE_RTP_OVERRIDE" not in text:
        if PATHS_OLD not in text:
            raise SystemExit("paths anchor missing")
        text = text.replace(PATHS_OLD, PATHS_NEW, 1)
    if FN_OLD in text:
        text = text.replace(FN_OLD, FN_NEW, 1)
    elif FN_OLD_ALT in text:
        text = text.replace(FN_OLD_ALT, FN_NEW_ALT, 1)
    else:
        raise SystemExit("function anchor missing")
    return text


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digests = {}
    for path in PATHS:
        if not path.is_file():
            print(f"skip_missing {path}", file=sys.stderr)
            continue
        bak = path.with_name(path.name + f".pre-crustle-rtp-dashboard-{stamp}")
        shutil.copy2(path, bak)
        text = patch_text(path.read_text(encoding="utf-8"))
        path.write_text(text, encoding="utf-8")
        digests[str(path)] = {
            "backup": str(bak),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    data = json.loads(CGR.read_text(encoding="utf-8"))
    overrides = data.setdefault("current_owner_overrides", {})
    overrides["crustle_dashboard_rtp_projection"] = {
        "goal_revision": 167,
        "status": "activated_display_only",
        "specialist_id": "crustle",
        "service": "pokebot-crustle-rtp-r167.service",
        "run_name": "crustle_rtp_r167",
        "mode": "crustle_rtp_r167",
        "phase": "train:rtp_shadow",
        "ordinary_rl_overridden": True,
        "training_restarted": False,
        "runtime_or_selector_changed": False,
        "receipt": "state/crustle_dashboard_rtp_projection_r167.json",
    }
    # Keep the prior RL projection, but mark it superseded so UI helpers
    # cannot treat stopped ordinary RL as the live training truth.
    prior = overrides.get("crustle_dashboard_active_rl_projection")
    if isinstance(prior, dict):
        prior = dict(prior)
        prior["status"] = "superseded_by_rtp_override"
        prior["superseded_by"] = "crustle_dashboard_rtp_projection"
        overrides["crustle_dashboard_active_rl_projection"] = prior
    CGR.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "schema": "poke_bot.crustle_dashboard_rtp_projection/v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "paths": digests,
        "active_service": "pokebot-crustle-rtp-r167.service",
        "cgr": str(CGR),
    }
    payload = json.dumps(receipt, indent=2) + "\n"
    RECEIPT.write_text(payload, encoding="utf-8")
    LOCAL_RECEIPT.write_text(
        json.dumps(
            {
                "schema": "poke_bot.crustle_dashboard_rtp_projection/v1",
                "status": "activated_display_only",
                "goal_revision": 167,
                "recorded_at_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "specialist_id": "crustle",
                "repair": {
                    "active_service": "pokebot-crustle-rtp-r167.service",
                    "run_name": "crustle_rtp_r167",
                    "mode": "crustle_rtp_r167",
                    "phase": "train:rtp_shadow",
                    "ordinary_rl_overridden": True,
                },
                "validation_paths": digests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(payload)

    os.chdir("/home/inzi/poke-bot-agent")
    primary = PATHS[0]
    spec = importlib.util.spec_from_file_location("dashboard_snapshot", primary)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog = mod.final_format_crustle_progress()
    print(
        "crustle_progress_status",
        prog.get("status"),
        "mode",
        prog.get("mode"),
        "service",
        (prog.get("service") or {}).get("name"),
        "pid",
        (prog.get("service") or {}).get("pid"),
        "phase",
        prog.get("phase"),
        "iteration",
        prog.get("iteration"),
        "percent",
        prog.get("percent"),
        "line",
        str(prog.get("latest_line") or "")[:160],
    )
    if prog.get("mode") != "crustle_rtp_r167":
        raise SystemExit("expected crustle_rtp_r167 mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
