#!/usr/bin/env python3
"""Create an immutable, precision-gated runtime copy of a public-card tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.public_matchup_router import PublicMatchupDecisionTree


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def activate(
    source: Path,
    checkpoint: Path,
    output: Path,
    *,
    min_precision: float,
    min_support: int,
    min_leaf_confidence: float,
    consecutive_required: int,
    allow_zero_materialized_adapters: bool = False,
) -> dict:
    raw = source.read_bytes()
    payload = json.loads(raw)
    PublicMatchupDecisionTree(payload, digest=_digest(raw))
    if payload.get("runtime_enabled") is not False:
        raise ValueError("source tree must be an inactive validation artifact")
    checkpoint_raw = checkpoint.read_bytes()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fit = dict((saved.get("extra") or {}).get("dormant_matchup_adapter_fit") or {})
    extra = dict(saved.get("extra") or {})
    dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
    adapter_config = dict(extra.get("matchup_adapter_config") or {})
    zero_materialized = bool(
        allow_zero_materialized_adapters
        and dormant.get("schema") == "poke_bot.zero_dormant_matchup_adapter/v1"
        and dormant.get("zero_output") is True
        and dormant.get("runtime_enabled") is False
        and tuple(adapter_config.get("expert_ids") or ()) == tuple(EXPERT_IDS)
    )
    route_decisions = {
        str(key): int(value) for key, value in (fit.get("route_decisions") or {}).items()
    }
    classes = dict((payload.get("validation") or {}).get("classes") or {})
    calibrated = dict(
        (payload.get("runtime_calibration") or {}).get("per_archetype") or {}
    )
    accepted = []
    thresholds = {}
    rejected = {}
    for archetype_id in EXPERT_IDS:
        metrics = dict(classes.get(archetype_id) or {})
        reasons = []
        if route_decisions.get(archetype_id, 0) <= 0 and not zero_materialized:
            reasons.append("no_trained_adapter_rows")
        calibration = dict(calibrated.get(archetype_id) or {})
        calibrated_precision = float(calibration.get("precision") or 0.0)
        threshold = calibration.get("min_leaf_confidence")
        if calibrated:
            if (
                calibration.get("available") is not True
                or threshold is None
                or not 0.0 < float(threshold) <= 1.0
                or calibrated_precision < float(min_precision)
            ):
                reasons.append("no_calibrated_precision_threshold")
        elif float(metrics.get("precision") or 0.0) < float(min_precision):
            reasons.append("precision_below_floor")
        if int(metrics.get("weighted_support") or 0) < int(min_support):
            reasons.append("support_below_floor")
        if reasons:
            rejected[archetype_id] = reasons
        else:
            accepted.append(archetype_id)
            thresholds[archetype_id] = float(
                threshold if threshold is not None else min_leaf_confidence
            )
    if not accepted:
        raise ValueError("no trained route passes the runtime precision gate")
    payload["runtime_enabled"] = True
    payload["runtime_contract"] = {
        "schema": "poke_bot.public_matchup_tree_runtime_activation/v1",
        "source_tree_digest": _digest(raw),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_digest": _digest(checkpoint_raw),
        "accepted_archetype_ids": accepted,
        "per_archetype_min_leaf_confidence": thresholds,
        "rejected_archetype_ids": rejected,
        "min_validation_precision": float(min_precision),
        "min_validation_weighted_support": int(min_support),
        "min_leaf_confidence": float(min_leaf_confidence),
        "consecutive_required": int(consecutive_required),
        "unknown_route_exact_bypass": True,
        "one_route_per_decision": True,
        "oracle_or_package_identity_forbidden": True,
        "zero_materialized_adapters_allowed": zero_materialized,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace runtime tree: {output}")
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    PublicMatchupDecisionTree.from_path(output, require_runtime_enabled=True)
    return {
        "output": str(output),
        "digest": _digest(encoded),
        "accepted_archetype_ids": accepted,
        "rejected_count": len(rejected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-precision", type=float, default=0.93)
    parser.add_argument("--min-support", type=int, default=10_000)
    parser.add_argument("--min-leaf-confidence", type=float, default=0.90)
    parser.add_argument("--consecutive-required", type=int, default=2)
    parser.add_argument(
        "--allow-zero-materialized-adapters",
        action="store_true",
        help=(
            "Allow precision-gated routing into a checksum-bound complete "
            "zero-output bank before its first isolated specialist fit."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(activate(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
