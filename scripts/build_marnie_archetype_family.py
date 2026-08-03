#!/usr/bin/env python3
"""Build the observed exact-list Marnie family from immutable feature shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from poke_bot.archetypes import classify_deck
from poke_bot.deck_pool import read_deck
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.specialist_archetype_family import (
    SCHEMA,
    SPECIALIST_ID,
    canonical_counts,
    cluster_variants,
    digest_json,
    multiset_digest,
    ordered_digest,
    split_clusters,
    validate_manifest,
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def build(feature_manifest: Path, package_csv: Path) -> dict[str, Any]:
    root = feature_manifest.parent
    source = json.loads(feature_manifest.read_text(encoding="utf-8"))
    package_cards = [int(value) for value in read_deck(package_csv)]
    package_multiset = multiset_digest(package_cards)
    observed: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in source.get("shards") or []:
        shard_path = root / str(shard["path"])
        expected = str(shard["sha256"])
        if _sha256(shard_path) != expected:
            raise RuntimeError(f"immutable shard digest mismatch: {shard_path}")
        for sequence in iter_feature_shard(shard_path):
            cards = [int(card) for card in sequence.deck]
            if len(cards) != 60 or classify_deck(cards) != SPECIALIST_ID:
                continue
            digest = multiset_digest(cards)
            observed.setdefault(
                digest,
                {
                    "family_id": SPECIALIST_ID,
                    "variant_id": "marnie-" + digest.split(":", 1)[1][:16],
                    "card_ids": cards,
                    "card_counts": [[card, count] for card, count in canonical_counts(cards)],
                    "ordered_digest": ordered_digest(cards),
                    "multiset_digest": digest,
                    "legality": {
                        "legal": True,
                        "validator": "existing_replay_import_exact_60_positive_integer_contract",
                    },
                    "classification": {
                        "archetype_id": SPECIALIST_ID,
                        "classifier": "poke_bot.archetypes.classify_deck",
                    },
                    "training_weight": 0.0,
                    "capability_mask": {
                        "core_setup_continuity": True,
                        "resource_attack_readiness": True,
                        "long_horizon_prize_pressure": True,
                    },
                    "package": digest == package_multiset,
                    "measurement": digest == package_multiset,
                },
            )
            sources[digest].append(
                {
                    "feature_shard": str(shard["path"]),
                    "feature_shard_sha256": expected,
                    "episode_id": str(sequence.episode_id),
                    "seat": int(sequence.seat),
                    "source": str(sequence.source),
                }
            )
    if package_multiset not in observed:
        raise RuntimeError("current package list was not observed in the sealed corpus")
    rows = list(observed.values())
    for row in rows:
        provenance_rows = sorted(
            sources[row["multiset_digest"]],
            key=lambda item: (item["feature_shard"], item["episode_id"], item["seat"]),
        )
        per_shard: dict[tuple[str, str], int] = defaultdict(int)
        for item in provenance_rows:
            per_shard[(item["feature_shard"], item["feature_shard_sha256"])] += 1
        row["provenance"] = {
            "observed_only": True,
            "occurrences": len(provenance_rows),
            "occurrence_identity_sha256": digest_json(provenance_rows),
            "source_shards": [
                {
                    "feature_shard": path,
                    "feature_shard_sha256": digest,
                    "occurrences": count,
                }
                for (path, digest), count in sorted(per_shard.items())
            ],
            "first_observation": provenance_rows[0],
            "last_observation": provenance_rows[-1],
            "source_manifest": str(feature_manifest),
            "source_manifest_sha256": _sha256(feature_manifest),
        }
    clusters = cluster_variants(rows)
    for row in rows:
        row["cluster_id"] = clusters[row["variant_id"]]
    package = next(row for row in rows if row["package"])
    splits = split_clusters(
        clusters.values(),
        package_cluster_id=package["cluster_id"],
        seed=_sha256(feature_manifest),
    )
    for row in rows:
        row["split"] = splits[row["cluster_id"]]
    rows.sort(key=lambda row: row["variant_id"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "family_id": SPECIALIST_ID,
        "status": "staged_runtime_inert",
        "owner_revision": 120,
        "source_manifest": str(feature_manifest),
        "source_manifest_sha256": _sha256(feature_manifest),
        "cluster_contract": {"metric": "card_swap_distance", "maximum": 2},
        "split_contract": {
            "cluster_stable": True,
            "package_forced_train": True,
            "other_cluster_targets": {"train": 0.60, "dev": 0.20, "locked": 0.20},
        },
        "minimum_non_package_clusters_for_activation": 12,
        "package_probability": 0.20,
        "variants": rows,
    }
    result["artifact_sha256"] = digest_json(result)
    return validate_manifest(result, require_activation_ready=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--package-deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.feature_manifest.resolve(), args.package_deck.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clusters = {row["cluster_id"] for row in result["variants"]}
    splits = defaultdict(set)
    for row in result["variants"]:
        splits[row["split"]].add(row["cluster_id"])
    print(json.dumps({"variants": len(result["variants"]), "clusters": len(clusters), "split_clusters": {k: len(v) for k, v in splits.items()}, "artifact_sha256": result["artifact_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
