#!/usr/bin/env python3
"""Report causal-oracle/feature decision-count mismatches without staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke_bot.feature_shards import iter_feature_shard
from poke_bot.pure_rl.matchup_adapter_corpus import ORACLE_INDEX_ROW_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    feature_manifest = json.loads(args.feature_manifest.read_text())
    oracle_manifest = json.loads(args.oracle_manifest.read_text())
    features_by_digest = {
        str(row["sha256"]): args.feature_manifest.parent / str(row["path"])
        for row in feature_manifest["shards"]
    }
    mismatches = 0
    checked = 0
    for oracle_row in oracle_manifest["shards"]:
        feature = features_by_digest[str(oracle_row["source_feature_digest"])]
        oracle = args.oracle_manifest.parent / str(oracle_row["path"])
        rows = (
            json.loads(line)
            for line in oracle.read_text().splitlines()
            if line.strip()
        )
        oracle_rows = (
            row for row in rows if row.get("schema") == ORACLE_INDEX_ROW_SCHEMA
        )
        for sequence, row in zip(iter_feature_shard(feature), oracle_rows, strict=True):
            checked += 1
            oracle_count = int(row.get("public_decisions") or 0)
            compact_count = len(sequence.decisions)
            if oracle_count == compact_count:
                continue
            print(
                json.dumps(
                    {
                        "episode_id": sequence.episode_id,
                        "seat": sequence.seat,
                        "feature": feature.name,
                        "oracle_decisions": oracle_count,
                        "compact_decisions": compact_count,
                        "first_compact_env_step": sequence.decisions[0].env_step,
                        "last_compact_env_step": sequence.decisions[-1].env_step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            mismatches += 1
            if mismatches >= args.limit:
                print(json.dumps({"checked": checked, "mismatches": mismatches}))
                return 1
    print(json.dumps({"checked": checked, "mismatches": mismatches}))
    return int(mismatches > 0)


if __name__ == "__main__":
    raise SystemExit(main())
