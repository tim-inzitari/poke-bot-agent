#!/usr/bin/env python3
"""Retrain the canonical router with independently simulated Spidops evidence.

The validated v35 row shards contain the expensive historical public-prefix
extraction.  Their integer labels are remapped by name into the current
canonical roster (including logical aliases), while a new independently
generated calibration archive supplies Team Rocket's Spidops observations.
No hidden state, future observation, or recorded archetype is used as input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier

from poke_bot.public_matchup_router import visible_opponent_card_ids
from scripts.train_public_matchup_tree_v32 import (
    SCHEMA,
    UNKNOWN,
    _calibrate_runtime_thresholds,
    _episode_split,
    _expanded_predict_proba,
    _load_shard,
    _load_shard_metadata,
    _safe_card_ids,
    _tree_payload,
    _weighted_metrics,
    sha256_file,
)


def _canonical_targets(roster_path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    payload = json.loads(roster_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "poke_bot.matchup_adapter_roster/v1":
        raise RuntimeError("canonical matchup-adapter roster schema changed")
    targets = tuple(
        str(value) for value in payload.get("active_expert_ids") or ()
    )
    if len(targets) != 18 or len(set(targets)) != len(targets):
        raise RuntimeError("canonical matchup-adapter roster is not exact roster-18")
    aliases = {
        str(source): str(target)
        for source, target in (payload.get("logical_aliases") or {}).items()
    }
    return targets, aliases


def _remap_historical_shards(
    row_root: Path,
    *,
    old_targets: tuple[str, ...],
    targets: tuple[str, ...],
    aliases: dict[str, str],
) -> tuple[list, list, list, list, list, list, list[dict]]:
    new_index = {name: index for index, name in enumerate(targets)}
    old_names = old_targets + (UNKNOWN,)
    unknown_index = len(targets)
    label_map = np.asarray(
        [
            new_index.get(aliases.get(name, name), unknown_index)
            if name != UNKNOWN
            else unknown_index
            for name in old_names
        ],
        dtype=np.int32,
    )
    train_x, train_y, train_w = [], [], []
    val_x, val_y, val_w = [], [], []
    sources = []
    for path in sorted(row_root.glob("*.npz")):
        metadata = _load_shard_metadata(path)
        matrix, labels, weights, splits = _load_shard(path)
        if labels.size and int(labels.max()) >= len(label_map):
            raise RuntimeError(f"historical label outside source roster: {path}")
        labels = label_map[labels]
        train_mask = splits == 0
        val_mask = splits == 1
        train_x.append(matrix[train_mask])
        train_y.append(labels[train_mask])
        train_w.append(weights[train_mask])
        val_x.append(matrix[val_mask])
        val_y.append(labels[val_mask])
        val_w.append(weights[val_mask])
        sources.append(
            {
                "archive": str(metadata["archive"]),
                "archive_digest": str(metadata["archive_digest"]),
                "row_shard": str(path),
                "row_shard_sha256": sha256_file(path),
                "public_states": int(labels.size),
                "label_remap": "source-name-to-canonical-roster-v1",
            }
        )
    if not sources:
        raise RuntimeError("no validated historical row shards found")
    return train_x, train_y, train_w, val_x, val_y, val_w, sources


def _spidops_calibration_rows(
    archive_path: Path,
    *,
    label: int,
    max_card_id: int,
    split_seed: int,
    validation_percent: int,
):
    indices: list[int] = []
    indptr = [0]
    labels: list[int] = []
    weights: list[float] = []
    splits: list[int] = []
    episodes = 0
    observations = 0
    with zipfile.ZipFile(archive_path, "r") as stream:
        members = sorted(
            row.filename
            for row in stream.infolist()
            if not row.is_dir() and row.filename.endswith(".json")
        )
        for member in members:
            payload = json.loads(stream.read(member))
            proof = payload.get("router_calibration") or {}
            if (
                proof.get("schema") != "poke_bot.router_calibration_episode/v1"
                or proof.get("target_archetype") != "team-rockets-spidops"
                or proof.get("causal_public_zones_only") is not True
                or proof.get("hidden_fields_written") is not False
            ):
                raise RuntimeError(f"invalid Spidops calibration proof: {member}")
            target_seat = int(proof["target_seat"])
            facing_seat = 1 - target_seat
            cumulative: set[int] = set()
            state_counts: dict[tuple[int, ...], int] = {}
            for step in payload.get("steps") or ():
                if not isinstance(step, list) or len(step) != 2:
                    continue
                row = step[facing_seat]
                observation = row.get("observation") if isinstance(row, dict) else None
                if not isinstance(observation, dict):
                    continue
                cumulative.update(visible_opponent_card_ids(observation))
                state = _safe_card_ids(cumulative, max_card_id)
                state_counts[state] = state_counts.get(state, 0) + 1
                observations += 1
            episode_id = str(payload.get("id") or member)
            split = _episode_split(
                episode_id, facing_seat, split_seed, validation_percent
            )
            for state, count in state_counts.items():
                indices.extend(state)
                indptr.append(len(indices))
                labels.append(label)
                weights.append(float(count))
                splits.append(split)
            episodes += 1
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.uint8),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(labels), max_card_id + 1),
        dtype=np.uint8,
    )
    labels_array = np.asarray(labels, dtype=np.int32)
    weights_array = np.asarray(weights, dtype=np.float64)
    splits_array = np.asarray(splits, dtype=np.uint8)
    return matrix, labels_array, weights_array, splits_array, {
        "archive": archive_path.name,
        "archive_digest": sha256_file(archive_path),
        "episodes": episodes,
        "public_states": len(labels),
        "observations": observations,
        "input": "cumulative_opponent_public_card_ids_only",
        "independent_simulation_label": "team-rockets-spidops",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-row-root", type=Path, required=True)
    parser.add_argument("--calibration-archive", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--max-card-id", type=int, default=4095)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--runtime-precision-floor", type=float, default=0.93)
    args = parser.parse_args()

    source_tree = json.loads(args.source_tree.read_text(encoding="utf-8"))
    if source_tree.get("schema") != SCHEMA:
        raise RuntimeError("source public matchup tree schema changed")
    old_targets = tuple(str(value) for value in source_tree.get("targets") or ())
    targets, aliases = _canonical_targets(args.roster)
    if "team-rockets-spidops" not in targets:
        raise RuntimeError("Spidops absent from canonical target roster")

    started = time.time()
    train_x, train_y, train_w, val_x, val_y, val_w, sources = (
        _remap_historical_shards(
            args.source_row_root,
            old_targets=old_targets,
            targets=targets,
            aliases=aliases,
        )
    )
    matrix, labels, weights, splits, calibration_source = (
        _spidops_calibration_rows(
            args.calibration_archive,
            label=targets.index("team-rockets-spidops"),
            max_card_id=args.max_card_id,
            split_seed=args.seed,
            validation_percent=args.validation_percent,
        )
    )
    train_mask = splits == 0
    val_mask = splits == 1
    train_x.append(matrix[train_mask])
    train_y.append(labels[train_mask])
    train_w.append(weights[train_mask])
    val_x.append(matrix[val_mask])
    val_y.append(labels[val_mask])
    val_w.append(weights[val_mask])
    sources.append(calibration_source)

    x_train = sparse.vstack(train_x, format="csr")
    y_train = np.concatenate(train_y)
    w_train = np.concatenate(train_w)
    x_val = sparse.vstack(val_x, format="csr")
    y_val = np.concatenate(val_y)
    w_val = np.concatenate(val_w)
    class_names = targets + (UNKNOWN,)
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    prediction = model.predict(x_val)
    probabilities = _expanded_predict_proba(model, x_val, len(class_names))
    validation = _weighted_metrics(y_val, prediction, w_val, class_names)
    runtime_calibration = _calibrate_runtime_thresholds(
        y_val,
        prediction,
        probabilities,
        w_val,
        class_names,
        precision_floor=args.runtime_precision_floor,
    )
    payload = {
        "schema": SCHEMA,
        "created_at_unix": time.time(),
        "runtime_enabled": False,
        "activation_status": "validation_required",
        "input_contract": {
            "features": "cumulative binary opponent-public card IDs",
            "public_zones": [
                "active",
                "bench",
                "discard",
                "attached_tools",
                "attached_energy",
                "public_pre_evolution",
            ],
            "uses_our_deck": False,
            "uses_our_archetype": False,
            "prohibited": [
                "opponent_hand",
                "opponent_deck",
                "opponent_prizes",
                "agent_identity",
                "recorded_archetype",
                "future_observations",
            ],
            "oracle_labels_are_inputs": False,
            "unknown_falls_back_to_base_policy": True,
            "one_route_maximum_per_observation": True,
        },
        "targets": list(targets),
        "unknown_class": UNKNOWN,
        "prediction_contract": {
            "route_output_width": len(targets),
            "route_class_names": list(targets),
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(targets),
            "adapter_count": len(targets),
        },
        "split": {
            "kind": "episode-and-seat-disjoint-deterministic-hash",
            "seed": args.seed,
            "validation_percent": args.validation_percent,
        },
        "training": {
            "criterion": "entropy",
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "train_public_states": int(x_train.shape[0]),
            "validation_public_states": int(x_val.shape[0]),
            "train_weighted_observations": float(w_train.sum()),
            "validation_weighted_observations": float(w_val.sum()),
            "elapsed_seconds": time.time() - started,
            "historical_label_remap": aliases,
        },
        "sources": sources,
        "validation": validation,
        "runtime_calibration": runtime_calibration,
        "calibration_contract": {
            "probability_columns": "expanded_to_canonical_class_indexes",
            "canonical_class_count": len(class_names),
            "fitted_class_indexes": [int(value) for value in model.classes_],
            "missing_fitted_classes_receive_zero_probability": True,
        },
        "tree": _tree_payload(model, class_names),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "public-matchup-tree.json"
    if artifact.exists():
        raise FileExistsError(f"immutable output already exists: {artifact}")
    descriptor, name = tempfile.mkstemp(prefix=".public-matchup-tree.", dir=output_dir)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, artifact)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema": "poke_bot.public_matchup_decision_tree_receipt/v1",
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "runtime_enabled": False,
        "validation": validation,
        "runtime_calibration": runtime_calibration,
    }
    (output_dir / "PUBLIC_MATCHUP_TREE_READY.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
