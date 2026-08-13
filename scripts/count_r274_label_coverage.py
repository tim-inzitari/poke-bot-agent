#!/usr/bin/env python3
"""Count target-only label keys in one compact expert manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT
from poke_bot.pure_rl.expert_feature_stream import EpisodeGroupedFeatureManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
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
    keys: Counter[str] = Counter()
    combo_rows = 0
    decisions = 0
    examples: list[dict[str, object]] = []
    try:
        train, validation = plan.splits()
        for split_name, split in (("train", train), ("validation", validation)):
            for sequence in split:
                for decision in sequence.decisions:
                    decisions += 1
                    labels = dict(decision.aux_labels or {})
                    keys.update(str(key) for key in labels)
                    combo = labels.get("combo_state")
                    if combo is not None:
                        combo_rows += 1
                        if len(examples) < 8:
                            examples.append(
                                {
                                    "split": split_name,
                                    "episode_id": str(sequence.episode_id),
                                    "seat": int(sequence.seat),
                                    "env_step": int(decision.env_step),
                                    "combo_state": combo,
                                }
                            )
    finally:
        del plan
    print(
        json.dumps(
            {
                "decisions": decisions,
                "aux_key_counts": dict(keys),
                "combo_rows": combo_rows,
                "combo_examples": examples,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
