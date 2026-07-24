#!/usr/bin/env python3
"""Audit an inactive causal matchup-tree candidate without activating it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from poke_bot.matchup_adapters import EXPERT_IDS

SCHEMA = "poke_bot.public_matchup_tree_candidate_audit/v1"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    artifact: Path,
    *,
    minimum_precision: float,
    minimum_weighted_support: int,
) -> dict[str, Any]:
    artifact = artifact.expanduser().resolve()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    targets = tuple(str(value) for value in payload.get("targets") or ())
    contract = dict(payload.get("calibration_contract") or {})
    validation = dict((payload.get("validation") or {}).get("classes") or {})
    calibration = dict(
        (payload.get("runtime_calibration") or {}).get("per_archetype") or {}
    )
    if (
        payload.get("schema") != "poke_bot.public_matchup_decision_tree/v1"
        or payload.get("runtime_enabled") is not False
        or targets != EXPERT_IDS
        or len(set(targets)) != len(EXPERT_IDS)
        or contract.get("probability_columns")
        != "expanded_to_canonical_class_indexes"
        or int(contract.get("canonical_class_count") or 0)
        != len(EXPERT_IDS) + 1
    ):
        raise RuntimeError("inactive canonical calibration contract failed")

    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for specialist_id in targets:
        metrics = dict(validation.get(specialist_id) or {})
        calibrated = dict(calibration.get(specialist_id) or {})
        precision = float(calibrated.get("precision") or 0.0)
        support = int(metrics.get("weighted_support") or 0)
        threshold = calibrated.get("min_leaf_confidence")
        reasons: list[str] = []
        if calibrated.get("available") is not True or threshold is None:
            reasons.append("no_calibrated_precision_threshold")
        elif not 0.0 < float(threshold) <= 1.0:
            reasons.append("invalid_calibrated_threshold")
        if precision < float(minimum_precision):
            reasons.append("precision_below_floor")
        if support < int(minimum_weighted_support):
            reasons.append("support_below_floor")
        rows[specialist_id] = {
            "precision": precision,
            "weighted_support": support,
            "min_leaf_confidence": threshold,
            "recall": float(calibrated.get("recall") or 0.0),
            "accepted": not reasons,
            "reasons": reasons,
        }
        if reasons:
            rejected[specialist_id] = reasons
        else:
            accepted.append(specialist_id)
    return {
        "schema": SCHEMA,
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "runtime_enabled": False,
        "minimum_precision": float(minimum_precision),
        "minimum_weighted_support": int(minimum_weighted_support),
        "target_count": len(targets),
        "accepted_count": len(accepted),
        "accepted_specialist_ids": accepted,
        "rejected_specialists": rejected,
        "per_specialist": rows,
        "safe_to_activate_automatically": False,
        "activation_requires_checkpoint_binding_and_safe_iteration_boundary": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.93)
    parser.add_argument("--minimum-weighted-support", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.artifact,
        minimum_precision=args.minimum_precision,
        minimum_weighted_support=args.minimum_weighted_support,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to replace audit: {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
