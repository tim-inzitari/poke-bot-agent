#!/usr/bin/env python3
"""Derive the checksum-bound Marnie iteration-20 registry for Revision 113."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    args = parser.parse_args()

    parent = read_json(args.parent_registry)
    row = dict(dict(parent.get("specialists") or {}).get(SPECIALIST_ID) or {})
    if (
        int(parent.get("iteration_ceiling", -1)) != 15
        or int(row.get("iteration_ceiling", -1)) != 15
        or int(parent.get("minimum_terminal_iteration", -1)) != 5
    ):
        raise RuntimeError("parent registry is not the expected Marnie r111 contract")

    candidate = json.loads(json.dumps(parent))
    candidate["owner_decision_revision"] = 113
    candidate["minimum_terminal_iteration"] = 20
    candidate["iteration_ceiling"] = 20
    isolated = dict(candidate.get("isolated_refresh_contract") or {})
    isolated["owner_decision_revision"] = 113
    isolated["games_per_iteration"] = 8192
    isolated["self_play_fraction"] = 0.125
    isolated["self_play_games_per_iteration"] = 1024
    isolated["public_opponent_games_per_iteration"] = 7168
    isolated["premium_skill_weighted_win_rate"] = 0.80
    isolated["premium_skill_weighted_confidence_lower"] = 0.50
    isolated["minimum_terminal_iteration"] = 20
    isolated["maximum_iterations"] = 21
    isolated["maximum_training_games"] = 172032
    candidate["isolated_refresh_contract"] = isolated
    candidate["specialists"][SPECIALIST_ID]["minimum_terminal_iteration"] = 20
    candidate["specialists"][SPECIALIST_ID]["iteration_ceiling"] = 20
    candidate["revision_113_terminal_contract"] = {
        "minimum_terminal_iteration": 20,
        "terminal_ceiling_completed_iteration": 20,
        "maximum_iterations": 21,
        "collect_after_terminal_ceiling": False,
        "activation_boundary": "durable_iteration_5_commit_before_iteration_6_collection",
    }
    atomic_json(args.candidate_registry, candidate)

    receipt = {
        "schema": "poke_bot.marnie_iteration20_stage/v1",
        "status": "ready_for_iteration_5_to_6_boundary",
        "owner_decision_revision": 113,
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_registry": str(args.parent_registry),
        "parent_registry_sha256": sha256(args.parent_registry),
        "candidate_registry": str(args.candidate_registry),
        "candidate_registry_sha256": sha256(args.candidate_registry),
        "minimum_terminal_iteration": 20,
        "iteration_ceiling": 20,
        "maximum_iterations": 21,
        "collect_iteration_21": False,
        "current_iteration_5_mutated": False,
    }
    atomic_json(args.stage_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
