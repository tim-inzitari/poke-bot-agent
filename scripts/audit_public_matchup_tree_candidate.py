#!/usr/bin/env python3
"""Audit an inactive causal matchup-tree candidate without activating it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from poke_bot.matchup_adapters import EXPERT_IDS

SCHEMA = "poke_bot.public_matchup_tree_candidate_audit/v1"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_public_state_audit(
    payload: dict[str, Any],
    *,
    targets: tuple[str, ...],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Prove that an all-zero public feature row cannot select an adapter."""

    input_contract = dict(payload.get("input_contract") or {})
    if input_contract.get("empty_public_state_exact_bypass") is not True:
        raise RuntimeError(
            "empty public state exact-bypass declaration is missing"
        )

    tree = dict(payload.get("tree") or {})
    class_names = tuple(str(value) for value in tree.get("class_names") or ())
    expected_class_names = targets + ("unknown",)
    try:
        node_count = int(tree.get("node_count", -1))
        children_left = tuple(
            int(value) for value in tree.get("children_left") or ()
        )
        children_right = tuple(
            int(value) for value in tree.get("children_right") or ()
        )
        feature_card_id = tuple(
            int(value) for value in tree.get("feature_card_id") or ()
        )
        thresholds = tuple(
            float(value) for value in tree.get("threshold") or ()
        )
        weighted_class_counts = tuple(
            tuple(float(value) for value in row)
            for row in tree.get("weighted_class_counts") or ()
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            "empty public state tree contract is malformed"
        ) from None
    if (
        node_count <= 0
        or class_names != expected_class_names
        or any(
            len(values) != node_count
            for values in (
                children_left,
                children_right,
                feature_card_id,
                thresholds,
                weighted_class_counts,
            )
        )
        or any(not math.isfinite(value) for value in thresholds)
        or any(
            len(row) != len(expected_class_names)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in row
            )
            for row in weighted_class_counts
        )
    ):
        raise RuntimeError("empty public state tree contract is malformed")

    node = 0
    visited: set[int] = set()
    while children_left[node] >= 0 or children_right[node] >= 0:
        if node in visited:
            raise RuntimeError("empty public state tree path contains a cycle")
        visited.add(node)
        # An all-zero sparse feature row has value 0.0 for every card ID.
        node = (
            children_right[node]
            if 0.0 > thresholds[node]
            else children_left[node]
        )
        if node < 0 or node >= node_count:
            raise RuntimeError("empty public state tree path is invalid")

    counts = weighted_class_counts[node]
    total = float(sum(counts))
    if total <= 0.0:
        return {
            "declared_exact_bypass": True,
            "verified_exact_bypass_after_calibration": True,
            "tree_leaf": node,
            "raw_prediction": "unknown",
            "raw_confidence": 0.0,
            "calibrated_min_leaf_confidence": None,
            "calibrated_route": None,
        }

    class_index = max(range(len(counts)), key=counts.__getitem__)
    raw_prediction = class_names[class_index]
    confidence = float(counts[class_index] / total)
    threshold: float | None = None
    calibrated_route: str | None = None
    if raw_prediction != "unknown":
        calibrated = dict(calibration.get(raw_prediction) or {})
        raw_threshold = calibrated.get("min_leaf_confidence")
        try:
            threshold = (
                float(raw_threshold) if raw_threshold is not None else None
            )
        except (TypeError, ValueError):
            threshold = None
        if (
            calibrated.get("available") is True
            and threshold is not None
            and math.isfinite(threshold)
            and 0.0 < threshold <= 1.0
            and confidence >= threshold
        ):
            calibrated_route = raw_prediction
    if calibrated_route is not None:
        raise RuntimeError(
            "empty public state routes after calibration: "
            f"{calibrated_route} at confidence {confidence:.12g}"
        )
    return {
        "declared_exact_bypass": True,
        "verified_exact_bypass_after_calibration": True,
        "tree_leaf": node,
        "raw_prediction": raw_prediction,
        "raw_confidence": confidence,
        "calibrated_min_leaf_confidence": threshold,
        "calibrated_route": None,
    }


def audit(
    artifact: Path,
    *,
    minimum_precision: float,
    minimum_weighted_support: int,
    expected_targets: Iterable[str] | None = None,
    target_registry: Path | None = None,
) -> dict[str, Any]:
    artifact = artifact.expanduser().resolve()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    targets = tuple(str(value) for value in payload.get("targets") or ())
    contract = dict(payload.get("calibration_contract") or {})
    validation = dict((payload.get("validation") or {}).get("classes") or {})
    calibration = dict(
        (payload.get("runtime_calibration") or {}).get("per_archetype") or {}
    )
    canonical_targets = tuple(
        str(value)
        for value in (
            expected_targets if expected_targets is not None else EXPERT_IDS
        )
    )
    if (
        payload.get("schema") != "poke_bot.public_matchup_decision_tree/v1"
        or payload.get("runtime_enabled") is not False
        or not canonical_targets
        or len(set(canonical_targets)) != len(canonical_targets)
        or targets != canonical_targets
        or contract.get("probability_columns")
        != "expanded_to_canonical_class_indexes"
        or int(contract.get("canonical_class_count") or 0)
        != len(canonical_targets) + 1
    ):
        raise RuntimeError("inactive canonical calibration contract failed")
    empty_public_state = _empty_public_state_audit(
        payload,
        targets=targets,
        calibration=calibration,
    )

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
        "canonical_target_ids": list(canonical_targets),
        "target_registry": (
            str(target_registry.expanduser().resolve())
            if target_registry is not None
            else None
        ),
        "target_registry_sha256": (
            _sha256(target_registry.expanduser().resolve())
            if target_registry is not None
            else None
        ),
        "accepted_count": len(accepted),
        "accepted_specialist_ids": accepted,
        "rejected_specialists": rejected,
        "per_specialist": rows,
        "empty_public_state": empty_public_state,
        "safe_to_activate_automatically": False,
        "activation_requires_checkpoint_binding_and_safe_iteration_boundary": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.93)
    parser.add_argument("--minimum-weighted-support", type=int, default=10_000)
    parser.add_argument(
        "--roster",
        type=Path,
        help=(
            "Optional staged V6 registry. Its ordered expert_ids replace the "
            "physical V5 default without changing checkpoint rows."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected_targets = None
    if args.roster is not None:
        roster = json.loads(args.roster.read_text(encoding="utf-8"))
        expected_targets = tuple(
            str(value) for value in roster.get("expert_ids") or ()
        )
        slot_targets = {
            str(row.get("archetype_id") or "")
            for row in roster.get("slots") or ()
            if isinstance(row, dict)
            and str(row.get("status") or "") in {"active", "dormant"}
            and str(row.get("archetype_id") or "")
        }
        if (
            roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
            or not expected_targets
            or len(expected_targets) != len(set(expected_targets))
            or slot_targets != set(expected_targets)
            or int(roster.get("required_specialist_count") or 0)
            != len(expected_targets)
        ):
            raise RuntimeError("staged V6 target registry is invalid")
    result = audit(
        args.artifact,
        minimum_precision=args.minimum_precision,
        minimum_weighted_support=args.minimum_weighted_support,
        expected_targets=expected_targets,
        target_registry=args.roster,
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
