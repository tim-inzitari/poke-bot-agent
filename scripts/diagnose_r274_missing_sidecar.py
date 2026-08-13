#!/usr/bin/env python3
"""Report the first compact expert decision absent from an r274 sidecar index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT
from poke_bot.pure_rl.expert_feature_stream import EpisodeGroupedFeatureManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    digest = "sha256:" + hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    plan = EpisodeGroupedFeatureManifest.open(
        args.manifest,
        expected_manifest_digest=digest,
        val_frac=0.10,
        seed=274,
        max_context=320,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        workers=1,
    )
    connection = sqlite3.connect(f"file:{args.index.resolve()}?mode=ro", uri=True)
    checked = 0
    try:
        train, validation = plan.splits()
        for split_name, split in (("train", train), ("validation", validation)):
            for sequence in split:
                for decision in sequence.decisions:
                    checked += 1
                    rows = connection.execute(
                        "SELECT observation_fingerprint FROM rows "
                        "WHERE episode_id=? AND seat=? AND env_step=?",
                        (str(sequence.episode_id), int(sequence.seat), int(decision.env_step)),
                    ).fetchall()
                    if len(rows) != 1:
                        print(json.dumps({
                            "checked": checked,
                            "split": split_name,
                            "episode_id": str(sequence.episode_id),
                            "seat": int(sequence.seat),
                            "env_step": int(decision.env_step),
                            "compact_observation_fingerprint": decision.observation_fingerprint,
                            "sidecar_match_count": len(rows),
                        }, sort_keys=True), flush=True)
                        return 1
    finally:
        connection.close()
        del plan
    print(json.dumps({"checked": checked, "status": "complete_exact_coverage"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
