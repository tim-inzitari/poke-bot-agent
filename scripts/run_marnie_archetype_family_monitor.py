#!/usr/bin/env python3
"""Run the fresh, training-ineligible post-activation Marnie monitor.

The managed trainer must already be stopped at the first clean commit after
family activation. This script neither controls the service nor writes replay;
it only materializes a checksum-bound monitor and, when required, an immutable
rollback request for the managed rollback installer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    MIGRATION_SCHEMA,
    PAUSE_SCHEMA,
    ROLLBACK_REQUEST_SCHEMA,
    sha256,
    validate_activation_request,
    validate_migration_receipt,
)
from poke_bot.archetype_family_study import (  # noqa: E402
    MONITOR_SCHEMA,
    compile_post_activation_monitor,
    family_shadow_plan,
    validate_post_activation_monitor,
)
from poke_bot.specialist_archetype_family import validate_manifest  # noqa: E402
from scripts.run_marnie_archetype_family_study import (  # noqa: E402
    _identity,
    _load_gate_specs,
    _read,
    _run_panel,
    _write_immutable,
)


def _require_managed_trainer_inactive(service: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", str(service)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    state = result.stdout.strip() or "unknown"
    if state in {"active", "activating", "reloading"}:
        raise RuntimeError(f"managed trainer is still {state}")
    return state


def run(args: argparse.Namespace) -> dict[str, Any]:
    service_state = _require_managed_trainer_inactive(args.service)
    pause = _read(args.pause)
    if (
        pause.get("schema") != PAUSE_SCHEMA
        or pause.get("next_collection_started") is not False
        or int(pause.get("restart_prevent_status", -1)) != 78
        or pause.get("pause_reason")
        != "fresh_post_activation_family_monitor_required"
    ):
        raise RuntimeError("monitor lacks its exact clean-boundary status-78 pause")
    migration = _read(args.migration)
    if migration.get("schema") != MIGRATION_SCHEMA:
        raise RuntimeError("monitor lacks an activation migration")
    request_path = Path(str(migration.get("request", ""))).resolve()
    validate_migration_receipt(migration, request_path=request_path)
    request = _read(request_path)
    validate_activation_request(
        request,
        expected_learner_digest=str(
            (request.get("bindings") or {}).get("learner_sha256", "")
        ),
    )
    if (
        pause.get("request_sha256") != sha256(request_path)
        or pause.get("migration_sha256") != sha256(args.migration)
    ):
        raise RuntimeError("monitor pause binds another activation")
    activation_pause = _read(Path(str(migration.get("pause_receipt", ""))))
    if int(pause["committed_iteration"]) < int(activation_pause["target_iteration"]):
        raise RuntimeError("monitor boundary precedes the first activated iteration")
    loop = _read(args.loop_state)
    learner = dict(loop.get("learner") or {})
    candidate_path = Path(str(learner.get("path", ""))).resolve()
    candidate_digest = str(learner.get("digest", ""))
    if (
        candidate_digest != str(pause.get("learner_sha256", ""))
        or not candidate_path.is_file()
        or sha256(candidate_path) != candidate_digest
    ):
        raise RuntimeError("monitor pause and current learner differ")
    parent = dict((request.get("bindings") or {}).get("checkpoint") or {})
    parent_path = Path(str(parent.get("path", ""))).resolve()
    if not parent_path.is_file() or sha256(parent_path) != parent.get("sha256"):
        raise RuntimeError("sealed pre-activation checkpoint changed")
    manifest = validate_manifest(
        _read(args.manifest), require_activation_ready=True
    )
    specs, opponent_ids = _load_gate_specs(args.active_gate_contract)
    plan = family_shadow_plan(
        manifest,
        opponent_ids,
        seed_book=f"{args.seed_book}:{sha256(args.pause)}",
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _write_immutable(output_dir / "monitor-plan.json", plan)
    treatments = {
        "candidate": {"path": str(candidate_path), "sha256": candidate_digest},
        "parent": {"path": str(parent_path), "sha256": str(parent["sha256"])},
    }
    locked_path = output_dir / "monitor-locked.json"
    package_path = output_dir / "monitor-package.json"
    locked_rows = _run_panel(
        output_path=locked_path,
        panel=plan["panels"]["locked"],
        treatments=treatments,
        manifest=manifest,
        specs=specs,
        workers=int(args.workers),
        game_timeout_seconds=int(args.game_timeout_seconds),
    )
    package_rows = _run_panel(
        output_path=package_path,
        panel=plan["panels"]["package"],
        treatments=treatments,
        manifest=manifest,
        specs=specs,
        workers=int(args.workers),
        game_timeout_seconds=int(args.game_timeout_seconds),
    )
    metrics = compile_post_activation_monitor(
        locked_rows=locked_rows,
        package_rows=package_rows,
    )
    monitor = {
        "schema": MONITOR_SCHEMA,
        "status": (
            "rollback_required"
            if metrics["rollback_required"]
            else "passed_continue_family_design"
        ),
        "rollback_required": bool(metrics["rollback_required"]),
        "metrics": metrics,
        "pause_receipt": str(args.pause.resolve()),
        "pause_receipt_sha256": sha256(args.pause),
        "migration": _identity(args.migration),
        "request": _identity(request_path),
        "candidate_checkpoint": _identity(candidate_path),
        "parent_checkpoint": _identity(parent_path),
        "plan": _identity(plan_path),
        "locked_rows": _identity(locked_path),
        "package_rows": _identity(package_path),
        "managed_trainer_state": service_state,
        "training_eligible": False,
        "replay_eligible": False,
        "kaggle_evidence_used_for_training_or_tuning": False,
    }
    monitor_path = _write_immutable(args.monitor_receipt, monitor)
    validate_post_activation_monitor(_read(monitor_path))
    rollback_path: Path | None = None
    if metrics["rollback_required"]:
        rollback = {
            "schema": ROLLBACK_REQUEST_SCHEMA,
            "status": "required_at_current_clean_boundary",
            "migration": _identity(args.migration),
            "monitor": _identity(monitor_path),
            "pause": _identity(args.pause),
            "restore_registry": dict(
                (request.get("bindings") or {}).get("registry") or {}
            ),
            "restore_selector": dict(
                (request.get("bindings") or {}).get("selector") or {}
            ),
            "restore_checkpoint": dict(parent),
            "sealed_pre_activation": dict(
                request.get("sealed_pre_activation") or {}
            ),
            "next_collection_started": False,
        }
        rollback_path = _write_immutable(args.rollback_request, rollback)
    return {
        "status": str(monitor["status"]),
        "monitor": str(monitor_path),
        "rollback_request": str(rollback_path) if rollback_path else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--pause", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--active-gate-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monitor-receipt", type=Path, required=True)
    parser.add_argument("--rollback-request", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--game-timeout-seconds", type=int, default=600)
    parser.add_argument("--seed-book", default="marnie-family-monitor-r133-v1")
    return parser


def main() -> int:
    result = run(_parser().parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
