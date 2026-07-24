#!/usr/bin/env python3
"""Fail closed unless the deterministic V31 production-ready receipt verifies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.matchup_runtime_production_ready/v1"
BOUNDARY_SCHEMA = "poke_bot.matchup_runtime_boundary_activation/v1"
MARKER_SCHEMA = "poke_bot.remote_matchup_runtime_activation/v1"

EXPECTED_PATHS = {
    "boundary_receipt": Path(
        "/home/inzi/poke-bot-agent/outputs/state/"
        "alakazam-matchup-runtime-iter26-v31.json"
    ),
    "merged_checkpoint": Path(
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "pure_rl_alakazam_temporal1_8k_teacher_v16_20260721/"
        "checkpoints/iter_00026_matchup_v31.pt"
    ),
    "runtime_tree": Path(
        "/home/inzi/poke-bot-agent/outputs/state/"
        "public-matchup-tree-runtime-v31.json"
    ),
    "remote_marker": Path(
        "/home/inzi/poke-bot-agent/outputs/state/"
        "matchup-runtime-activation-v31.json"
    ),
    "production_dropin": Path(
        "/home/inzi/.config/systemd/user/"
        "pokebot-pure-rl-alakazam.service.d/"
        "zzzzzzzzzzzzzzzzzz-v31-matchup-runtime.conf"
    ),
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def validate(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _json(receipt_path)
    artifacts = receipt.get("artifacts")
    accepted = sorted(str(value) for value in receipt.get("accepted_archetype_ids") or ())
    if not (
        receipt.get("schema") == SCHEMA
        and receipt.get("runtime_enabled") is True
        and int(receipt.get("iteration", -1)) == 27
        and accepted
        and isinstance(artifacts, dict)
        and set(artifacts) == set(EXPECTED_PATHS)
    ):
        raise ValueError("production-ready receipt contract is invalid")
    for name, expected in EXPECTED_PATHS.items():
        row = artifacts.get(name)
        path = expected.expanduser().resolve()
        if not (
            isinstance(row, dict)
            and Path(str(row.get("path") or "")).expanduser().resolve() == path
            and path.is_file()
            and path.stat().st_size > 0
            and str(row.get("digest") or "") == _digest(path)
        ):
            raise ValueError(f"production-ready artifact failed: {name}")

    boundary = _json(EXPECTED_PATHS["boundary_receipt"])
    tree = _json(EXPECTED_PATHS["runtime_tree"])
    marker = _json(EXPECTED_PATHS["remote_marker"])
    runtime = dict(tree.get("runtime_contract") or {})
    boundary_tree = dict(boundary.get("runtime_tree") or {})
    activated = dict(boundary.get("activated_learner") or {})
    dropin = EXPECTED_PATHS["production_dropin"].read_text(encoding="utf-8")
    if not (
        boundary.get("schema") == BOUNDARY_SCHEMA
        and int((boundary.get("boundary") or {}).get("next_iteration", -1)) == 27
        and str(activated.get("digest") or "")
        == _digest(EXPECTED_PATHS["merged_checkpoint"])
        and sorted(boundary_tree.get("accepted_archetype_ids") or ()) == accepted
        and str(boundary_tree.get("digest") or "")
        == _digest(EXPECTED_PATHS["runtime_tree"])
        and boundary_tree.get("continuous_re_evaluation") is True
        and boundary_tree.get("one_route_per_decision") is True
        and boundary_tree.get("unknown_route_exact_bypass") is True
        and marker.get("schema") == MARKER_SCHEMA
        and marker.get("runtime_enabled") is True
        and sorted(marker.get("accepted_archetype_ids") or ()) == accepted
        and str(marker.get("tree_digest") or "")
        == _digest(EXPECTED_PATHS["runtime_tree"])
        and marker.get("continuous_reevaluation") is True
        and marker.get("one_route_per_decision") is True
        and sorted(runtime.get("accepted_archetype_ids") or ()) == accepted
        and runtime.get("one_route_per_decision") is True
        and runtime.get("unknown_route_exact_bypass") is True
        and "WorkingDirectory=/home/inzi/poke-bot-agent-deployments/"
        "pure-rl-resident-v31-matchup-runtime" in dropin
        and "Environment=POKEBOT_MATCHUP_ADAPTER_RUNTIME=1" in dropin
        and "public-matchup-tree-runtime-v31.json" in dropin
    ):
        raise ValueError("production-ready linked runtime contract is invalid")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validated = validate(args.receipt)
    print(
        "matchup_runtime_production_ready_ok "
        f"routes={len(validated['accepted_archetype_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
