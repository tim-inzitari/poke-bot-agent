#!/usr/bin/env python3
"""Build the immutable revision-111 Marnie opponent-tier derivative."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


H10_OPPONENT_ID = "specialist-alakazam-final-format-h10-02c014ad7c33"
H10_CHECKPOINT = "sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92"
NEW_GATE_ID = "specialist-strong-public-roster-sw80-at-iter5-v1+h10-s-other-a-r111"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def derive_gate(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    gate = copy.deepcopy(source)
    next_gate = gate.get("next_gate")
    if not isinstance(next_gate, dict) or next_gate.get("id") != gate.get("active_gate_id"):
        raise RuntimeError("source gate identity is inconsistent")
    evaluation = dict(next_gate.get("evaluation") or {})
    criteria = dict(next_gate.get("pass_criteria") or {})
    roster = next_gate.get("roster")
    if (
        not isinstance(roster, list)
        or len(roster) != 17
        or int(evaluation.get("games_per_opponent", -1)) != 250
        or int(evaluation.get("games_total", -1)) != 4250
        or int(evaluation.get("seat0_games_per_opponent", -1)) != 125
        or int(evaluation.get("seat1_games_per_opponent", -1)) != 125
        or float(criteria.get("skill_weighted_win_rate", -1)) != 0.80
        or float(criteria.get("skill_weighted_confidence_lower", -1)) != 0.50
    ):
        raise RuntimeError("source gate does not match the protected 17x250 contract")

    counts = {"h10_s": 0, "other_frozen_a": 0, "public_a": 0}
    for row in roster:
        if not isinstance(row, dict):
            raise RuntimeError("malformed opponent row")
        if row.get("opponent_id") == H10_OPPONENT_ID:
            if row.get("frozen_checkpoint_digest") != H10_CHECKPOINT or row.get("frozen_specialist") is not True:
                raise RuntimeError("H10 Alakazam identity/checkpoint mismatch")
            row["tier"], row["weight"] = "S", 2.0
            counts["h10_s"] += 1
        elif row.get("frozen_specialist") is True:
            row["tier"], row["weight"] = "A", 1.0
            counts["other_frozen_a"] += 1
        else:
            row["tier"], row["weight"] = "A", 1.0
            counts["public_a"] += 1
    if counts != {"h10_s": 1, "other_frozen_a": 13, "public_a": 3}:
        raise RuntimeError(f"unexpected tier populations: {counts}")

    gate["active_gate_id"] = NEW_GATE_ID
    next_gate["id"] = NEW_GATE_ID
    next_gate["label"] = "Strong public/frozen A roster + non-active H10 S (revision 111)"
    gate["owner_decision_revision"] = 111
    semantics = gate.setdefault("active_gate_semantics", {})
    semantics.pop("frozen_specialist_tier", None)
    semantics["opponent_tier_policy"] = {
        "eligible_non_active_h10_specialist": {"tier": "S", "weight": 2.0},
        "other_frozen_specialist": {"tier": "A", "weight": 1.0},
        "remaining_public_opponent": {"tier": "A", "weight": 1.0},
    }
    semantics["invariant"] = (
        "The exact 17-opponent checksum roster remains fixed at 250 greedy games "
        "per opponent and 125/125 seats; eligible non-active H10 specialists are "
        "S/2.0 and all other frozen/public opponents are A/1.0."
    )
    gate["derivation"] = {
        **dict(gate.get("derivation") or {}),
        "owner_decision_revision": 111,
        "source_gate_id": str(source.get("active_gate_id") or ""),
        "tier_only_derivative": True,
        "opponent_identity_content_digest_game_and_seat_contract_preserved": True,
        "terminal_thresholds_preserved": True,
        "current_iteration_5_evidence_reinterpreted": False,
        "first_affected_iteration": 6,
    }
    return gate, counts


def derive_registry(source: dict[str, Any], *, gate_relative_path: str) -> dict[str, Any]:
    if source.get("schema") != "poke_bot.specialist_runtime_registry/v1":
        raise RuntimeError("invalid source runtime registry")
    registry = copy.deepcopy(source)
    registry["version"] = max(5, int(registry.get("version") or 0) + 1)
    registry["owner_decision_revision"] = 111
    registry["active_gate_contract"] = gate_relative_path
    registry["terminal_active_gate_id"] = NEW_GATE_ID
    command = list(registry.get("common_trainer_args") or [])
    if "--boundary-design-migration-reason" not in command:
        raise RuntimeError("source registry lacks boundary migration authority")
    reason_index = command.index("--boundary-design-migration-reason") + 1
    if reason_index >= len(command):
        raise RuntimeError("source registry has malformed boundary migration reason")
    command[reason_index] = "receipt_backed_opponent_tiers_r111"
    registry["common_trainer_args"] = command
    registry["opponent_tier_policy"] = {
        "owner_decision_revision": 111,
        "scope": ["public_mix", "formal_premium_holdout"],
        "first_affected_iteration": 6,
        "eligible_non_active_h10_specialist": {"tier": "S", "weight": 2.0},
        "other_frozen_specialist": {"tier": "A", "weight": 1.0},
        "remaining_public_opponent": {"tier": "A", "weight": 1.0},
    }
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--gate-relative-path", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source_gate = read_json(args.source_gate)
    source_registry = read_json(args.source_registry)
    gate, counts = derive_gate(source_gate)
    registry = derive_registry(source_registry, gate_relative_path=args.gate_relative_path)
    write_once(args.output_gate, gate)
    write_once(args.output_registry, registry)
    receipt = {
        "schema": "poke_bot.marnie_opponent_tier_stage/v1",
        "status": "ready_for_iteration_5_to_6_boundary",
        "owner_decision_revision": 111,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_gate": str(args.source_gate),
        "source_gate_sha256": sha256(args.source_gate),
        "candidate_gate": str(args.output_gate),
        "candidate_gate_sha256": sha256(args.output_gate),
        "source_registry": str(args.source_registry),
        "source_registry_sha256": sha256(args.source_registry),
        "candidate_registry": str(args.output_registry),
        "candidate_registry_sha256": sha256(args.output_registry),
        "tier_counts": counts,
        "exact_roster_size": 17,
        "games_per_opponent": 250,
        "formal_games_total": 4250,
        "activation_after_completed_iteration": 5,
        "first_affected_iteration": 6,
        "current_iteration_5_mutated": False,
        "training_interrupted": False,
    }
    write_once(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
