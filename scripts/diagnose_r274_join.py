#!/usr/bin/env python3
"""Print one compact expert decision without mutating its join inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT
from poke_bot.pure_rl.expert_feature_stream import EpisodeGroupedFeatureManifest


def sparse_words(value: object) -> list[list[list[float | int]]]:
    words: list[list[list[float | int]]] = []
    offsets = list(value.offset)
    for word in range(int(value.num_words)):
        start = int(offsets[word])
        end = int(offsets[word + 1]) if word + 1 < len(offsets) else len(value.index)
        words.append(
            [
                [int(index) - word * int(value.pos), float(weight)]
                for index, weight in zip(
                    value.index[start:end], value.value[start:end], strict=True
                )
            ]
        )
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--seat", required=True, type=int)
    parser.add_argument("--env-step", required=True, type=int)
    args = parser.parse_args()
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    plan = EpisodeGroupedFeatureManifest.open(
        args.manifest,
        expected_manifest_digest="sha256:" + digest,
        val_frac=0.10,
        seed=274,
        max_context=320,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        workers=1,
    )
    try:
        train, validation = plan.splits()
        for split_name, split in (("train", train), ("validation", validation)):
            for sequence in split:
                if str(sequence.episode_id) != args.episode or int(sequence.seat) != args.seat:
                    continue
                for decision in sequence.decisions:
                    if int(decision.env_step) != args.env_step:
                        continue
                    print(
                        json.dumps(
                            {
                                "split": split_name,
                                "episode_id": str(sequence.episode_id),
                                "seat": int(sequence.seat),
                                "env_step": int(decision.env_step),
                                "observation_fingerprint": decision.observation_fingerprint,
                                "action_token": sparse_words(decision.action_token),
                                "stages": [
                                    {
                                        "target_index": int(stage.target_index),
                                        "action_combos": stage.action_combos,
                                        "option_words": sparse_words(stage.options),
                                    }
                                    for stage in decision.policy_stages
                                ],
                            },
                            sort_keys=True,
                        )
                    )
                    return 0
    finally:
        del plan
    raise RuntimeError("decision not found")


if __name__ == "__main__":
    raise SystemExit(main())
