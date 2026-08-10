#!/usr/bin/env python3
"""Patch train dashboard_snapshot.py so Crustle H10 RL is the live projection."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

PATH = Path("/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py")
CGR = Path("/home/inzi/poke-bot-agent/ops/current_goal_requirements.json")
RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/crustle-dashboard-rl-projection-r167.json"
)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = PATH.with_name(PATH.name + f".pre-crustle-rl-dashboard-{stamp}")
    shutil.copy2(PATH, bak)

    const_old = """FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-bootstrap.service\"
)
"""
    const_new = """FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-bootstrap.service\"
)
FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE = (
    \"pokebot-final-format-crustle-r113-h10-rl.service\"
)
"""
    if "FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE" not in text:
        if const_old not in text:
            raise SystemExit("const anchor missing")
        text = text.replace(const_old, const_new, 1)

    paths_old = """FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG = (
    FINAL_FORMAT_CRUSTLE_ROOT / \"logs/bootstrap.log\"
)
FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / \"outputs/state/marnie-canonical-training-freeze-r163.json\"
)
"""
    paths_new = """FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG = (
    FINAL_FORMAT_CRUSTLE_ROOT / \"logs/bootstrap.log\"
)
FINAL_FORMAT_CRUSTLE_H10_LOG = (
    ROOT / \"outputs/logs/final_format_crustle_r113_h10_i_v6_8k.log\"
)
FINAL_FORMAT_CRUSTLE_H10_PROGRESS_LOG = (
    ROOT / \"outputs/logs/final_format_crustle_r113_h10_i_v6_8k.progress.log\"
)
FINAL_FORMAT_CRUSTLE_H10_PROGRESS_STATUS = (
    ROOT / \"outputs/logs/final_format_crustle_r113_h10_i_v6_8k.progress.status\"
)
FINAL_FORMAT_CRUSTLE_H10_RUN_DIR = (
    ROOT / \"outputs/pure_rl/final_format_crustle_r113_h10_i_v6_8k\"
)
FINAL_FORMAT_CRUSTLE_H10_REGISTRY = (
    FINAL_FORMAT_CRUSTLE_ROOT
    / \"runtime/specialist_runtime_registry_h10_r113.json\"
)
FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / \"outputs/state/marnie-canonical-training-freeze-r163.json\"
)
"""
    if "FINAL_FORMAT_CRUSTLE_H10_LOG =" not in text:
        if paths_old not in text:
            raise SystemExit("paths anchor missing")
        text = text.replace(paths_old, paths_new, 1)

    fn_old = """def final_format_crustle_progress() -> dict[str, Any]:
    \"\"\"Project the managed all-guide Crustle H10 bootstrap.\"\"\"

    service = unit_state(FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE, user=True)
    active = bool(
        (
            service.get(\"active\")
            or service.get(\"active_state\") == \"activating\"
        )
        and (
            int(service.get(\"pid\") or 0) > 0
            or service.get(\"sub_state\") in {\"running\", \"start\"}
        )
    )
    if not active:
        return {\"status\": \"waiting\", \"available\": False}
"""
    fn_new_text = r'''def final_format_crustle_progress() -> dict[str, Any]:
    """Project managed Crustle H10 bootstrap or active RL collection."""

    bootstrap_service = unit_state(
        FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE, user=True
    )
    rl_service = unit_state(FINAL_FORMAT_CRUSTLE_H10_RL_SERVICE, user=True)
    bootstrap_active = bool(
        (
            bootstrap_service.get("active")
            or bootstrap_service.get("active_state") == "activating"
        )
        and (
            int(bootstrap_service.get("pid") or 0) > 0
            or bootstrap_service.get("sub_state") in {"running", "start"}
        )
    )
    rl_active = bool(
        rl_service.get("active")
        and (
            int(rl_service.get("pid") or 0) > 0
            or rl_service.get("sub_state") in {"running", "start"}
        )
    )
    status_text = read_tail(FINAL_FORMAT_CRUSTLE_H10_PROGRESS_STATUS, 20_000)
    progress_log = read_tail(FINAL_FORMAT_CRUSTLE_H10_PROGRESS_LOG, 2_000_000)
    loop_state = read_json(FINAL_FORMAT_CRUSTLE_H10_RUN_DIR / "loop_state.json")
    registry = read_json(FINAL_FORMAT_CRUSTLE_H10_REGISTRY)
    rl_history = bool(
        loop_state
        and registry
        and (
            status_text.strip()
            or FINAL_FORMAT_CRUSTLE_H10_PROGRESS_LOG.is_file()
            or FINAL_FORMAT_CRUSTLE_H10_LOG.is_file()
        )
    )
    if rl_active or (rl_history and not bootstrap_active):
        progress = parse_curriculum_progress(
            status_text,
            progress_log,
            iteration_hint=int((loop_state or {}).get("next_iteration") or 0),
        )
        progress = infer_post_train_gate_progress(
            progress,
            read_tail(FINAL_FORMAT_CRUSTLE_H10_LOG, 500_000),
            iteration_hint=int((loop_state or {}).get("next_iteration") or 0),
        )
        updated = max(
            (
                path.stat().st_mtime
                for path in (
                    FINAL_FORMAT_CRUSTLE_H10_LOG,
                    FINAL_FORMAT_CRUSTLE_H10_PROGRESS_LOG,
                    FINAL_FORMAT_CRUSTLE_H10_PROGRESS_STATUS,
                )
                if path.is_file()
            ),
            default=None,
        )
        specialist = dict(
            ((registry or {}).get("specialists") or {}).get("crustle") or {}
        )
        learner = dict((loop_state or {}).get("learner") or {})
        checkpoint = learner.get("path") or specialist.get("initial_checkpoint")
        checkpoint_digest = (
            learner.get("digest") or specialist.get("initial_checkpoint_sha256")
        )
        if checkpoint_digest and not str(checkpoint_digest).startswith("sha256:"):
            checkpoint_digest = f"sha256:{checkpoint_digest}"
        structure = checkpoint_structure_telemetry(
            checkpoint,
            checkpoint_digest,
            cache_path=(
                ROOT
                / "outputs/state/dashboard-final-crustle-h10-live-structure-cache.json"
            ),
        )
        model_parameters = int(structure.get("model_parameters") or 0)
        if model_parameters <= 0:
            matches = re.findall(
                r"(?:model_params=|loaded checkpoint params=)(\d+)",
                read_tail(FINAL_FORMAT_CRUSTLE_H10_LOG, 200_000),
            )
            model_parameters = int(matches[-1]) if matches else 0
        trainer_args = [
            str(item) for item in (registry or {}).get("common_trainer_args") or []
        ]
        remote_endpoints: list[str] = []
        try:
            endpoint_arg = trainer_args.index("--remote-worker-endpoints")
            remote_endpoints = [
                endpoint.strip()
                for endpoint in trainer_args[endpoint_arg + 1].split(",")
                if endpoint.strip()
            ]
        except (ValueError, IndexError):
            pass
        if rl_active:
            scheduler_queues = scheduler_queue_state(
                specialist.get("run_name")
                or "final_format_crustle_r113_h10_i_v6_8k",
                log_path=FINAL_FORMAT_CRUSTLE_H10_LOG,
            )
            scheduler_queues = scope_scheduler_queues_to_progress(
                progress,
                scheduler_queues,
            )
            drain_projection = result_drain_projection(progress, scheduler_queues)
        else:
            scheduler_queues = {
                "available": False,
                "mode": "stopped",
                "local": {"active_or_claimed": 0},
                "endpoints": {},
                "unassigned": 0,
                "results": {"waiting_ingest": 0},
            }
            drain_projection = {}
        progress_metrics = dict(progress.get("metrics") or {})
        progress_metrics.update(drain_projection.get("metrics") or {})
        last_phase = drain_projection.get(
            "phase", progress.get("stage") or "collect"
        )
        latest_line = drain_projection.get(
            "latest_line", progress.get("line")
        )
        if not rl_active:
            last_phase = f"stopped:{last_phase}"
            latest_line = f"STOPPED · last progress: {latest_line or '—'}"
        phase_fresh_window_s = (
            20 * 60
            if progress.get("stage") == "heldout:checkpoint_staging"
            else 30
        )
        return {
            "available": True,
            "authoritative": True,
            "source": str(FINAL_FORMAT_CRUSTLE_H10_PROGRESS_STATUS),
            "log": str(FINAL_FORMAT_CRUSTLE_H10_LOG),
            "latest_line": latest_line,
            "raw_latest_line": progress.get("line"),
            "updated_at": updated,
            "fresh": bool(
                rl_active
                and updated
                and time.time() - updated < phase_fresh_window_s
            ),
            "status": "running" if rl_active else "stopped",
            "mode": "final_format_crustle_h10_rl",
            "phase": last_phase,
            "run": specialist.get("run_name")
            or "final_format_crustle_r113_h10_i_v6_8k",
            "specialist_id": "crustle",
            "iteration": progress.get(
                "iteration", (loop_state or {}).get("next_iteration")
            ),
            "iterations_target": 16,
            "current": progress.get("current"),
            "total": progress.get("total"),
            "percent": progress.get("percent"),
            "rate": progress.get("rate"),
            "rate_unit": progress.get("rate_unit"),
            "games_per_second": (
                drain_projection.get("games_per_second", progress.get("gps"))
                if rl_active
                else 0.0
            ),
            "samples_per_second": progress.get("sps") if rl_active else 0.0,
            "eta": progress.get("eta") if rl_active else None,
            "metrics": progress_metrics,
            "remote_workers": (
                scheduler_queues.get("remote_workers") if rl_active else 0
            ),
            "remote_endpoints": remote_endpoints,
            "scheduler_queues": scheduler_queues,
            "model_parameters": model_parameters,
            "checkpoint": checkpoint,
            "checkpoint_digest": checkpoint_digest,
            "structure": structure,
            "service": rl_service,
        }
    service = bootstrap_service
    active = bootstrap_active
    if not active:
        return {"status": "waiting", "available": False}
'''
    if 'mode": "final_format_crustle_h10_rl"' not in text:
        if fn_old not in text:
            raise SystemExit("function anchor missing")
        text = text.replace(fn_old, fn_new_text, 1)

    sel_old = """        active_final_refresh = (
            final_crustle
            if final_crustle.get(\"status\") == \"running\"
            else (
                final_marnie
                if final_marnie.get(\"status\")
                in {\"running\", \"complete\", \"stopped\"}
                else final_alakazam
            )
        )
"""
    sel_new = """        active_final_refresh = (
            final_crustle
            if final_crustle.get(\"status\") == \"running\"
            else (
                final_marnie
                if final_marnie.get(\"status\") == \"running\"
                else (
                    final_crustle
                    if final_crustle.get(\"status\")
                    in {\"complete\", \"stopped\"}
                    else (
                        final_marnie
                        if final_marnie.get(\"status\")
                        in {\"complete\", \"stopped\"}
                        else final_alakazam
                    )
                )
            )
        )
"""
    if sel_old in text:
        text = text.replace(sel_old, sel_new, 1)

    PATH.write_text(text, encoding="utf-8")

    data = json.loads(CGR.read_text(encoding="utf-8"))
    auth = data.get("authoritative_sources")
    if isinstance(auth, dict):
        auth["live_selector"] = (
            "/home/inzi/poke-bot-agent/outputs/final_format_crustle_r113/runtime/"
            "specialist_runtime_h10_r113.env#POKEBOT_ACTIVE_SPECIALIST"
        )
        data["authoritative_sources"] = auth
        data["live_selector"] = auth["live_selector"]
        CGR.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "schema": "poke_bot.crustle_dashboard_rl_projection/v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backup": str(bak),
        "dashboard_snapshot_sha256": "sha256:"
        + hashlib.sha256(PATH.read_bytes()).hexdigest(),
        "active_service": "pokebot-final-format-crustle-r113-h10-rl.service",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    os.chdir("/home/inzi/poke-bot-agent")
    spec = importlib.util.spec_from_file_location("dashboard_snapshot", PATH)
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
        "iteration",
        prog.get("iteration"),
        "phase",
        prog.get("phase"),
        "percent",
        prog.get("percent"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
