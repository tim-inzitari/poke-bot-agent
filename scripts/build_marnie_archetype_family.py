#!/usr/bin/env python3
"""Build the observed exact-list Marnie family from immutable feature shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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
    family_probabilities,
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


def _observe_package_replay(
    receipt_path: Path,
    package_cards: list[int],
) -> dict[str, Any]:
    """Prove the singular package list was actually observed in sealed replay."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "poke_bot.completed_collection/v1":
        raise RuntimeError("package observation is not a completed collection")
    shard = receipt.get("shard") or {}
    shard_path = Path(str(shard.get("path") or "")).expanduser().resolve()
    expected = str(shard.get("sha256") or "")
    if not shard_path.is_file() or not expected.startswith("sha256:"):
        raise RuntimeError("package observation shard identity is incomplete")

    package_digest = multiset_digest(package_cards)
    digest = hashlib.sha256()
    observation: dict[str, Any] | None = None
    with shard_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if observation is not None:
                continue
            try:
                row = json.loads(raw)
                cards = [int(card) for card in row.get("deck") or ()]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                len(cards) == 60
                and multiset_digest(cards) == package_digest
                and str(row.get("archetype") or "") == SPECIALIST_ID
                and classify_deck(cards) == SPECIALIST_ID
            ):
                observation = {
                    "source_kind": "sealed_pure_rl_replay",
                    "source_artifact": str(shard_path),
                    "source_artifact_sha256": expected,
                    "episode_id": str(row.get("episode_id") or ""),
                    "seat": int(row.get("seat", -1)),
                    "source": str(row.get("source") or ""),
                    "line_number": line_number,
                    "collection_receipt": str(receipt_path),
                    "collection_receipt_sha256": _sha256(receipt_path),
                }
    actual = "sha256:" + digest.hexdigest()
    if actual != expected:
        raise RuntimeError("package observation replay digest mismatch")
    if observation is None:
        raise RuntimeError("current package list was not observed in sealed replay")
    if not observation["episode_id"] or observation["seat"] not in {0, 1}:
        raise RuntimeError("package replay observation lacks exact provenance")
    return observation


def build(
    feature_manifest: Path,
    package_csv: Path,
    *,
    package_observation_receipt: Path | None = None,
    observed_deck_dir: Path | None = None,
) -> dict[str, Any]:
    root = feature_manifest.parent
    source = json.loads(feature_manifest.read_text(encoding="utf-8"))
    resolved_feature_manifest = feature_manifest
    if not source.get("shards") and source.get("manifest"):
        resolved_feature_manifest = (
            feature_manifest.parent / str(source["manifest"])
        ).resolve()
        expected_manifest_sha256 = str(source.get("manifest_sha256") or "")
        if _sha256(resolved_feature_manifest) != expected_manifest_sha256:
            raise RuntimeError("protected feature-manifest pointer digest mismatch")
        source = json.loads(resolved_feature_manifest.read_text(encoding="utf-8"))
        root = resolved_feature_manifest.parent
    if not source.get("shards"):
        raise RuntimeError("resolved feature manifest contains no immutable shards")
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
                    "source_kind": "sealed_public_feature_shard",
                    "source_artifact": str(shard["path"]),
                    "source_artifact_sha256": expected,
                    "episode_id": str(sequence.episode_id),
                    "seat": int(sequence.seat),
                    "source": str(sequence.source),
                }
            )
    observed_deck_files: list[Path] = []
    if observed_deck_dir is not None:
        observed_deck_files = sorted(
            observed_deck_dir.glob("*grimmsnarl-froslass.csv")
        )
        if not observed_deck_files:
            raise RuntimeError("observed Marnie tournament deck catalog is empty")
        for deck_path in observed_deck_files:
            cards = [int(card) for card in read_deck(deck_path)]
            if len(cards) != 60 or classify_deck(cards) != SPECIALIST_ID:
                raise RuntimeError(
                    f"observed tournament list is not an exact legal family list: "
                    f"{deck_path}"
                )
            digest = multiset_digest(cards)
            observed.setdefault(
                digest,
                {
                    "family_id": SPECIALIST_ID,
                    "variant_id": "marnie-" + digest.split(":", 1)[1][:16],
                    "card_ids": cards,
                    "card_counts": [
                        [card, count] for card, count in canonical_counts(cards)
                    ],
                    "ordered_digest": ordered_digest(cards),
                    "multiset_digest": digest,
                    "legality": {
                        "legal": True,
                        "validator": "existing_exact_60_positive_integer_and_family_contract",
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
                    "source_kind": "observed_public_tournament_deck",
                    "source_artifact": str(deck_path.resolve()),
                    "source_artifact_sha256": _sha256(deck_path),
                    "episode_id": deck_path.stem,
                    "seat": -1,
                    "source": "repository_tournament_public_catalog",
                }
            )
    if package_multiset not in observed:
        if package_observation_receipt is None:
            raise RuntimeError(
                "current package list was not observed in the sealed public corpus "
                "and no receipt-backed replay observation was supplied"
            )
        package_observation = _observe_package_replay(
            package_observation_receipt,
            package_cards,
        )
        observed[package_multiset] = {
            "family_id": SPECIALIST_ID,
            "variant_id": "marnie-" + package_multiset.split(":", 1)[1][:16],
            "card_ids": package_cards,
            "card_counts": [
                [card, count] for card, count in canonical_counts(package_cards)
            ],
            "ordered_digest": ordered_digest(package_cards),
            "multiset_digest": package_multiset,
            "legality": {
                "legal": True,
                "validator": "existing_exact_60_positive_integer_and_family_contract",
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
            "package": True,
            "measurement": True,
        }
        sources[package_multiset].append(package_observation)
    rows = list(observed.values())
    for row in rows:
        provenance_rows = sorted(
            sources[row["multiset_digest"]],
            key=lambda item: (
                item["source_kind"],
                item["source_artifact"],
                item["episode_id"],
                item["seat"],
            ),
        )
        per_shard: dict[tuple[str, str, str], int] = defaultdict(int)
        for item in provenance_rows:
            per_shard[
                (
                    item["source_kind"],
                    item["source_artifact"],
                    item["source_artifact_sha256"],
                )
            ] += 1
        row["provenance"] = {
            "observed_only": True,
            "occurrences": len(provenance_rows),
            "occurrence_identity_sha256": digest_json(provenance_rows),
            "source_shards": [
                {
                    "source_kind": kind,
                    "source_artifact": path,
                    "source_artifact_sha256": digest,
                    "occurrences": count,
                }
                for (kind, path, digest), count in sorted(per_shard.items())
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
        "resolved_feature_manifest": str(resolved_feature_manifest),
        "resolved_feature_manifest_sha256": _sha256(resolved_feature_manifest),
        "observed_deck_catalog": {
            "directory": str(observed_deck_dir) if observed_deck_dir else None,
            "file_count": len(observed_deck_files),
            "files_sha256": {
                str(path.resolve()): _sha256(path) for path in observed_deck_files
            },
        },
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
    for row in rows:
        row["training_weight"] = family_probabilities(result).get(
            row["variant_id"], 0.0
        )
    result["artifact_sha256"] = digest_json(result)
    return validate_manifest(result, require_activation_ready=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--package-deck", type=Path, required=True)
    parser.add_argument("--package-observation-receipt", type=Path)
    parser.add_argument("--observed-deck-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.feature_manifest.resolve(),
        args.package_deck.resolve(),
        package_observation_receipt=(
            args.package_observation_receipt.resolve()
            if args.package_observation_receipt is not None
            else None
        ),
        observed_deck_dir=(
            args.observed_deck_dir.resolve()
            if args.observed_deck_dir is not None
            else None
        ),
    )
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
