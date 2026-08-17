#!/usr/bin/env python3
"""Seal an exact recent-20 ESS/clip comparison from per-day AWR receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


DAY_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_awr_gate_receipt/v1"
SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_recent20_awr_gate_receipt/v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def aggregate(rows: list[dict], arm: str) -> dict[str, float | int]:
    metrics = [dict(row[arm]) for row in rows]
    decisions = sum(int(row["decisions"]) for row in metrics)
    weight_sum = sum(float(row["awr_weight_sum"]) for row in metrics)
    weight_sq_sum = sum(float(row["awr_weight_sq_sum"]) for row in metrics)
    ess = min(float(decisions), weight_sum * weight_sum / max(weight_sq_sum, 1e-12))
    clip = sum(float(row["awr_weight_clip_frac"]) * int(row["decisions"]) for row in metrics) / decisions
    return {
        "decisions": decisions,
        "awr_weight_sum": weight_sum,
        "awr_weight_sq_sum": weight_sq_sum,
        "awr_effective_sample_size": ess,
        "awr_effective_sample_fraction": ess / decisions,
        "awr_weight_clip_frac": clip,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-receipt", type=Path, action="append", required=True)
    p.add_argument("--cache-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("output is create-only")
    cache_sha = sha(args.cache_receipt.resolve())
    loaded: list[dict] = []
    identities: list[dict] = []
    for path in args.day_receipt:
        path = path.resolve()
        value = json.loads(path.read_text())
        if value.get("schema") != DAY_SCHEMA or value.get("cache_receipt_sha256") != cache_sha:
            raise SystemExit(f"day receipt identity mismatch: {path}")
        if value.get("finite") is not True or value.get("actor_activation") is not False:
            raise SystemExit(f"day receipt is not an inactive finite diagnostic: {path}")
        loaded.append(value)
        identities.append({"utc_day": value["utc_day"], "path": str(path), "sha256": sha(path)})
    days = sorted(str(row["utc_day"]) for row in loaded)
    if len(days) != 20 or len(set(days)) != 20 or days[0] != "2026-07-23" or days[-1] != "2026-08-11":
        raise SystemExit("AWR receipts are not the exact recent-20 window")
    legacy = aggregate(loaded, "legacy")
    h3 = aggregate(loaded, "h3")
    if legacy["decisions"] != h3["decisions"]:
        raise SystemExit("legacy/H3 decision membership differs")
    result = {
        "schema": SCHEMA,
        "cache_receipt_sha256": cache_sha,
        "day_receipts": sorted(identities, key=lambda row: row["utc_day"]),
        "legacy": legacy,
        "h3": h3,
        "comparison": {
            "ess_fraction_ratio_h3_over_legacy": float(h3["awr_effective_sample_fraction"]) / float(legacy["awr_effective_sample_fraction"]),
            "clip_fraction_absolute_delta": float(h3["awr_weight_clip_frac"]) - float(legacy["awr_weight_clip_frac"]),
            "same_exact_recent20_replay_membership_and_weights": True,
        },
        "finite": all(math.isfinite(float(v)) for arm in (legacy, h3) for v in arm.values()),
        "actor_activation": False,
        "activation_eligible": False,
    }
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data); os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), **result["comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
