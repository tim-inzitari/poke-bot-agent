#!/usr/bin/env python3
"""Seal the 19-day AWR diagnostic plus the explicit Aug-11 blocker.

This is deliberately a non-promotion receipt.  The exact recent-20 finalizer
remains strict and is not weakened to accept a target-authority reconstruction
as an actor-policy shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any


DAY_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_awr_gate_receipt/v1"
CACHE_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_policy_additive_cache_receipt/v1"
OVERRIDE_SCHEMA = "poke_bot.alakazam_critic_aug11_target_authority_override/v1"
SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_awr_diagnostic_blocked/v1"
AUTHORITY = "revision_23_prize_plan_v2_h3_actor_canary"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def expected_days() -> list[str]:
    first = date.fromisoformat("2026-07-23")
    last = date.fromisoformat("2026-08-10")
    result: list[str] = []
    current = first
    while current <= last:
        result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, float | int]:
    metrics = [dict(row[arm]) for row in rows]
    decisions = sum(int(row["decisions"]) for row in metrics)
    if decisions <= 0:
        raise ValueError("AWR diagnostic has no decision-stage rows")
    weight_sum = sum(float(row["awr_weight_sum"]) for row in metrics)
    weight_sq_sum = sum(float(row["awr_weight_sq_sum"]) for row in metrics)
    ess = min(float(decisions), weight_sum * weight_sum / max(weight_sq_sum, 1e-12))
    clip = sum(
        float(row["awr_weight_clip_frac"]) * int(row["decisions"])
        for row in metrics
    ) / decisions
    return {
        "decisions": decisions,
        "awr_weight_sum": weight_sum,
        "awr_weight_sq_sum": weight_sq_sum,
        "awr_effective_sample_size": ess,
        "awr_effective_sample_fraction": ess / decisions,
        "awr_weight_clip_frac": clip,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day-receipt", type=Path, action="append", required=True)
    parser.add_argument("--exact20-cache-receipt", type=Path, required=True)
    parser.add_argument("--aug11-cache-receipt", type=Path, required=True)
    parser.add_argument("--aug11-override-receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("output is create-only")
    contract_path = args.contract.expanduser().resolve()
    if sha256_file(contract_path) != args.contract_sha256:
        raise ValueError("canonical contract digest mismatch")
    contract = load_object(contract_path, "canonical contract")
    authority = contract.get(AUTHORITY)
    if (
        not isinstance(authority, dict)
        or authority.get("owner_goal_revision") != 23
        or authority.get("activation_and_failure_boundary", {}).get(
            "current_activation_allowed"
        )
        is not False
    ):
        raise ValueError("canonical r23 authority is absent or already active")

    exact20_cache_path = args.exact20_cache_receipt.expanduser().resolve()
    exact20_cache = load_object(exact20_cache_path, "exact-20 cache receipt")
    if (
        exact20_cache.get("schema") != CACHE_SCHEMA
        or exact20_cache.get("actor_activation") is not False
        or exact20_cache.get("activation_eligible") is not False
        or len(exact20_cache.get("day_shards") or ()) != 20
    ):
        raise ValueError("exact-20 cache receipt is not the sealed inactive cache")
    exact20_cache_sha = sha256_file(exact20_cache_path)

    loaded: list[dict[str, Any]] = []
    identities: list[dict[str, str]] = []
    for path_arg in args.day_receipt:
        path = path_arg.expanduser().resolve()
        row = load_object(path, "day AWR receipt")
        if (
            row.get("schema") != DAY_SCHEMA
            or row.get("cache_receipt_sha256") != exact20_cache_sha
            or row.get("finite") is not True
            or row.get("actor_activation") is not False
            or row.get("activation_eligible") is not False
            or row.get("comparison", {}).get(
                "same_replay_membership_and_weights"
            )
            is not True
        ):
            raise ValueError(f"day AWR receipt is not a sealed inactive diagnostic: {path}")
        loaded.append(row)
        identities.append(
            {
                "utc_day": str(row["utc_day"]),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    days = sorted(str(row["utc_day"]) for row in loaded)
    if days != expected_days() or len(set(days)) != 19:
        raise ValueError("day receipts are not exactly 2026-07-23 through 2026-08-10")

    aug11_cache_path = args.aug11_cache_receipt.expanduser().resolve()
    aug11_cache = load_object(aug11_cache_path, "Aug-11 cache receipt")
    aug11_rows = list(aug11_cache.get("day_shards") or ())
    if (
        aug11_cache.get("schema") != CACHE_SCHEMA
        or aug11_cache.get("actor_activation") is not False
        or aug11_cache.get("activation_eligible") is not False
        or len(aug11_rows) != 1
        or aug11_rows[0].get("utc_day") != "2026-08-11"
    ):
        raise ValueError("Aug-11 cache receipt is not the target-authority canary")
    aug11 = dict(aug11_rows[0])

    override_path = args.aug11_override_receipt.expanduser().resolve()
    override = load_object(override_path, "Aug-11 override receipt")
    if (
        override.get("schema") != OVERRIDE_SCHEMA
        or override.get("utc_day") != "2026-08-11"
        or override.get("critic_cache_only") is not True
        or override.get("expert_training_eligible") is not False
        or override.get("actor_activation") is not False
        or override.get("sealed_target_cache_canary", {}).get("receipt_sha256")
        != sha256_file(aug11_cache_path)
    ):
        raise ValueError("Aug-11 override does not bind the cache canary")

    legacy = aggregate(loaded, "legacy")
    h3 = aggregate(loaded, "h3")
    if legacy["decisions"] != h3["decisions"]:
        raise ValueError("legacy/H3 historical decision membership changed")
    result = {
        "schema": SCHEMA,
        "contract": {
            "path": str(contract_path),
            "sha256": args.contract_sha256,
            "goal_revision": int(contract["goal_revision"]),
            "semantic_owner_goal_revision": 23,
            "required_authority": AUTHORITY,
        },
        "exact20_cache_receipt": {
            "path": str(exact20_cache_path),
            "sha256": exact20_cache_sha,
        },
        "historical_exact_policy_days": {
            "utc_days": days,
            "day_receipts": sorted(identities, key=lambda row: row["utc_day"]),
            "legacy": legacy,
            "h3": h3,
            "comparison": {
                "ess_fraction_ratio_h3_over_legacy": float(
                    h3["awr_effective_sample_fraction"]
                )
                / float(legacy["awr_effective_sample_fraction"]),
                "clip_fraction_absolute_delta": float(h3["awr_weight_clip_frac"])
                - float(legacy["awr_weight_clip_frac"]),
                "same_exact_replay_membership_and_weights": True,
            },
        },
        "aug11_blocker": {
            "utc_day": "2026-08-11",
            "cache_receipt_path": str(aug11_cache_path),
            "cache_receipt_sha256": sha256_file(aug11_cache_path),
            "override_receipt_path": str(override_path),
            "override_receipt_sha256": sha256_file(override_path),
            "target_authority_rows": int(aug11["rows"]),
            "missing_policy_rows_reconstructed_for_critic_cache_only": int(
                aug11["policy_cache_only_missing_reconstructed_from_sealed_target"]
            ),
            "mismatched_policy_rows_replaced_for_critic_cache_only": int(
                aug11["policy_cache_only_mismatched_replaced_by_sealed_target"]
            ),
            "reason": "no_exact_same_membership_legacy_actor_policy_corpus",
            "eligible_for_awr_actor_comparison": False,
        },
        "promotion_gate_passed": False,
        "activation_eligible": False,
        "actor_activation": False,
        "production_branch": "exact_legacy_z_minus_V_existing",
        "blocked_reasons": [
            "aug11_lacks_exact_same_membership_legacy_actor_policy_corpus",
            "H1_vs_H3_shadow_policy_paired_game_evaluation_receipt_missing",
            "safe_boundary_activation_receipt_missing",
        ],
        "finite": all(
            math.isfinite(float(value))
            for metrics in (legacy, h3)
            for value in metrics.values()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "activation_eligible": False,
                **result["historical_exact_policy_days"]["comparison"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
