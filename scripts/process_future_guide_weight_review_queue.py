#!/usr/bin/env python3
"""Process the oldest pending future specialist guide-weight review."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.guide_weight_shadow_pair import (  # noqa: E402
    finalize,
    prepare_manifest,
    run_evaluation,
    run_training,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _selector(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate selector key: {key}")
        values[key] = value
    return values


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--shadow-device", default="cpu")
    parser.add_argument("--production-device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--game-timeout-seconds", type=int, default=600)
    parser.add_argument("--boundary-timeout-seconds", type=float, default=43200)
    args = parser.parse_args()

    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock = args.lock.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"ok": True, "status": "already_running"}))
        return 0

    selector = _selector(args.selector.expanduser().resolve())
    registry_path = args.registry.expanduser().resolve()
    registry = _read(registry_path)
    specialist = selector.get("POKEBOT_ACTIVE_SPECIALIST", "")
    row = dict((registry.get("specialists") or {}).get(specialist) or {})
    policy = dict(row.get("guide_weight_policy") or {})
    if (
        not specialist
        or specialist == "teal-mask-ogerpon-ex"
        or policy.get("scope") != "future_specialist_training_runs_only"
        or int(policy.get("prospective_scope_revision") or 0) != 44
        or int(policy.get("learning_semantics_revision") or 0) != 46
        or selector.get("POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION") != "44"
        or selector.get("POKEBOT_GUIDE_LEARNING_SEMANTICS_REVISION") != "46"
    ):
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "not_an_eligible_future_specialist",
                    "specialist_id": specialist or None,
                },
                sort_keys=True,
            )
        )
        return 0

    runtime_root = Path(str(registry.get("runtime_root") or "")).resolve()
    run_dir = runtime_root / "outputs/pure_rl" / Path(row["run_name"]).name
    requests = sorted((run_dir / "guide_weight_reviews").glob("review_*.request.json"))
    for request in requests:
        request_row = _read(request)
        iteration = int(request_row["completed_iteration"])
        output = (
            args.output_root.expanduser().resolve()
            / specialist
            / f"iter_{iteration:05d}_{_sha(request)[:12]}"
        )
        boundary_status = output / "boundary_receipt.json"
        if boundary_status.exists() and _read(boundary_status).get("status") in {
            "activated",
            "state_updated_weight_held",
        }:
            continue
        manifest = prepare_manifest(
            request_path=request,
            output_dir=output,
            shadow_device=args.shadow_device,
            production_device=args.production_device,
            baseline_manifest=args.baseline_manifest,
        )
        training = run_training(manifest, device=args.shadow_device)
        rows = run_evaluation(
            manifest,
            training,
            workers=args.workers,
            game_timeout_seconds=args.game_timeout_seconds,
        )
        _evidence, schedule = finalize(manifest, training, rows)
        boundary = Path(str(policy["boundary_controller"])).resolve()
        command = [
            sys.executable,
            str(boundary),
            "--schedule",
            str(schedule),
            "--run-dir",
            str(run_dir),
            "--log",
            str(row["log"]),
            "--unit",
            args.training_unit,
            "--selector",
            str(args.selector.resolve()),
            "--registry",
            str(registry_path),
            "--status",
            str(boundary_status),
            "--timeout-seconds",
            str(args.boundary_timeout_seconds),
        ]
        subprocess.run(command, check=True)
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "processed",
                    "specialist_id": specialist,
                    "completed_iteration": iteration,
                    "schedule": str(schedule),
                    "boundary_receipt": str(boundary_status),
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "ok": True,
                "status": "no_pending_review",
                "specialist_id": specialist,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
