#!/usr/bin/env python3
"""Train a deck-agnostic matchup router from causal public card evidence.

Full archived decks are used only by the pinned ladder classifier to create
offline labels.  Model inputs contain only cumulative opponent card IDs that
were public to the acting seat at that exact point in the game.  The resulting
tree is exported as plain JSON and is always marked runtime-disabled; enabling
it requires a separate activation/precision gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.public_matchup_decision_tree/v1"
SHARD_SCHEMA = "poke_bot.public_matchup_tree_rows/v1"
UNKNOWN = "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _episode_split(episode_id: str, seat: int, seed: int, val_percent: int) -> int:
    raw = f"{seed}:{episode_id}:{seat}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 100
    return int(bucket < val_percent)


def _safe_card_ids(values: Sequence[int], max_card_id: int) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values if 0 < int(value) <= max_card_id}))


def _build_archive_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    # Heavy imports stay inside the worker so --help and provenance inspection
    # remain usable without the scientific stack.
    import numpy as np
    from scipy import sparse

    from poke_bot.ladder_replay import LadderReplayClassifier
    from poke_bot.public_matchup_router import visible_opponent_card_ids
    from poke_bot.replay_import import episode_id_of

    archive = Path(str(task["archive"]))
    output = Path(str(task["output"]))
    expected_digest = str(task["archive_digest"])
    targets = tuple(str(value) for value in task["targets"])
    target_index = {value: index for index, value in enumerate(targets)}
    unknown_index = len(targets)
    max_card_id = int(task["max_card_id"])
    classifier = LadderReplayClassifier.from_paths(
        Path(str(task["mix"])),
        Path(str(task["representatives"])),
        card_csv=Path(str(task["card_csv"])),
        additive_registered_ids=tuple(str(value) for value in task["additive"]),
    )

    indices: list[int] = []
    indptr = [0]
    labels: list[int] = []
    weights: list[float] = []
    splits: list[int] = []
    episodes = 0
    acting_seats = 0
    observations = 0
    hidden_field_guard = True

    with zipfile.ZipFile(archive, "r") as stream:
        members = sorted(
            row.filename
            for row in stream.infolist()
            if not row.is_dir() and row.filename.endswith(".json")
        )
        for member in members:
            try:
                payload = json.loads(stream.read(member))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            decks, classified = classifier.classify_episode(payload)
            if len(decks) != 2 or len(classified) != 2:
                continue
            episode_id = str(episode_id_of(payload))
            steps = payload.get("steps")
            if not episode_id or not isinstance(steps, list):
                continue
            episodes += 1
            for seat in (0, 1):
                opponent_label = str(classified[1 - seat].deck_id or UNKNOWN)
                label = target_index.get(opponent_label, unknown_index)
                cumulative: set[int] = set()
                state_counts: dict[tuple[int, ...], int] = {}
                for step in steps:
                    if not isinstance(step, list) or len(step) != 2:
                        continue
                    seat_row = step[seat]
                    if not isinstance(seat_row, Mapping):
                        continue
                    obs = seat_row.get("observation")
                    if not isinstance(obs, Mapping):
                        continue
                    # This function walks only explicit opponent public zones.
                    # Hand/deck/prize, metadata, labels, and future steps cannot
                    # enter the feature vector.
                    cumulative.update(visible_opponent_card_ids(obs))
                    public_state = _safe_card_ids(cumulative, max_card_id)
                    state_counts[public_state] = state_counts.get(public_state, 0) + 1
                    observations += 1
                if not state_counts:
                    continue
                acting_seats += 1
                split = _episode_split(
                    episode_id,
                    seat,
                    int(task["seed"]),
                    int(task["val_percent"]),
                )
                for public_state, count in state_counts.items():
                    indices.extend(public_state)
                    indptr.append(len(indices))
                    labels.append(label)
                    weights.append(float(count))
                    splits.append(split)

    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.uint8),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(labels), max_card_id + 1),
        dtype=np.uint8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=np.asarray(matrix.shape, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int16),
            weights=np.asarray(weights, dtype=np.float32),
            splits=np.asarray(splits, dtype=np.uint8),
            metadata=np.asarray(
                json.dumps(
                    {
                        "schema": SHARD_SCHEMA,
                        "archive": archive.name,
                        "archive_digest": expected_digest,
                        "episodes": episodes,
                        "acting_seats": acting_seats,
                        "public_states": len(labels),
                        "observations": observations,
                        "hidden_field_guard": hidden_field_guard,
                        "input": "cumulative_opponent_public_card_ids_only",
                    },
                    sort_keys=True,
                )
            ),
        )
    os.replace(temporary, output)
    return {
        "archive": archive.name,
        "archive_digest": expected_digest,
        "output": str(output),
        "output_digest": sha256_file(output),
        "episodes": episodes,
        "acting_seats": acting_seats,
        "public_states": len(labels),
        "observations": observations,
    }


def _load_shard(path: Path):
    import numpy as np
    from scipy import sparse

    with np.load(path, allow_pickle=False) as payload:
        shape = tuple(int(value) for value in payload["shape"])
        matrix = sparse.csr_matrix(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=shape,
        )
        return (
            matrix,
            payload["labels"].astype("int32", copy=False),
            payload["weights"].astype("float64", copy=False),
            payload["splits"].astype("uint8", copy=False),
        )


def _load_shard_metadata(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
    if metadata.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"invalid cached public-tree row shard: {path}")
    return metadata


def _weighted_metrics(y_true, y_pred, weights, class_names: Sequence[str]) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        sample_weight=weights,
        zero_division=0.0,
    )
    correct = float(weights[y_true == y_pred].sum())
    total = float(weights.sum())
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        sample_weight=weights,
    )
    return {
        "weighted_accuracy": correct / total if total else 0.0,
        "weighted_observations": total,
        "classes": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "weighted_support": float(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.tolist(),
    }


def _tree_payload(model, class_names: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    tree = model.tree_
    values = np.asarray(tree.value)
    if values.ndim == 3:
        values = values[:, 0, :]
    expanded = np.zeros((values.shape[0], len(class_names)), dtype=np.float64)
    model_classes = [int(value) for value in model.classes_]
    for source_index, class_index in enumerate(model_classes):
        expanded[:, class_index] = values[:, source_index]
    return {
        "class_names": list(class_names),
        "fitted_class_indexes": model_classes,
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature_card_id": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "weighted_class_counts": expanded.tolist(),
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
    }


def _expanded_predict_proba(model, matrix, class_count: int):
    """Return probabilities indexed by canonical class index.

    ``DecisionTreeClassifier.predict_proba`` emits one column per class that
    occurred in the training split. Sparse specialist corpora can omit one or
    more canonical classes, so using the canonical class index directly
    against that compressed matrix silently calibrates the wrong route.
    """

    import numpy as np

    compressed = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if compressed.ndim != 2:
        raise ValueError("predict_proba must return a two-dimensional matrix")
    expanded = np.zeros((compressed.shape[0], int(class_count)), dtype=np.float64)
    for source_index, raw_class_index in enumerate(model.classes_):
        class_index = int(raw_class_index)
        if not 0 <= class_index < int(class_count):
            raise ValueError(
                f"fitted class index {class_index} outside canonical width "
                f"{class_count}"
            )
        expanded[:, class_index] = compressed[:, source_index]
    return expanded


def _calibrate_runtime_thresholds(
    y_true,
    y_pred,
    probabilities,
    weights,
    class_names: Sequence[str],
    *,
    precision_floor: float,
) -> dict[str, Any]:
    """Choose the maximum-recall per-class threshold meeting precision floor."""

    import numpy as np

    rows: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names[:-1]):
        predicted = np.flatnonzero(y_pred == class_index)
        true_support = float(weights[y_true == class_index].sum())
        if predicted.size == 0 or true_support <= 0.0:
            rows[class_name] = {
                "available": False,
                "min_leaf_confidence": None,
                "precision": 0.0,
                "recall": 0.0,
                "accepted_weighted_observations": 0.0,
                "reason": "no_validation_predictions_or_support",
            }
            continue
        scores = probabilities[predicted, class_index]
        order = np.argsort(-scores, kind="stable")
        predicted = predicted[order]
        scores = scores[order]
        accepted_weight = np.cumsum(weights[predicted])
        true_positive_weight = np.cumsum(
            weights[predicted] * (y_true[predicted] == class_index)
        )
        precision = np.divide(
            true_positive_weight,
            accepted_weight,
            out=np.zeros_like(true_positive_weight),
            where=accepted_weight > 0,
        )
        # Only evaluate the final row for each distinct score, matching the
        # runtime's inclusive confidence threshold exactly.
        group_ends = np.flatnonzero(
            np.r_[scores[1:] != scores[:-1], True]
        )
        eligible = group_ends[precision[group_ends] >= float(precision_floor)]
        if eligible.size == 0:
            rows[class_name] = {
                "available": False,
                "min_leaf_confidence": None,
                "precision": float(precision[group_ends[0]]),
                "recall": 0.0,
                "accepted_weighted_observations": 0.0,
                "reason": "no_threshold_meets_precision_floor",
            }
            continue
        chosen = int(eligible[-1])
        rows[class_name] = {
            "available": True,
            "min_leaf_confidence": float(scores[chosen]),
            "precision": float(precision[chosen]),
            "recall": float(true_positive_weight[chosen] / true_support),
            "accepted_weighted_observations": float(accepted_weight[chosen]),
            "reason": None,
        }
    return {
        "schema": "poke_bot.public_matchup_tree_calibration/v1",
        "precision_floor": float(precision_floor),
        "selection": "maximum_recall_subject_to_weighted_precision_floor",
        "per_archetype": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--card-csv", type=Path, required=True)
    parser.add_argument("--target-archetype", action="append", required=True)
    parser.add_argument("--additive-archetype", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--max-card-id", type=int, default=4095)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--runtime-precision-floor", type=float, default=0.93)
    parser.add_argument(
        "--reuse-row-shards",
        action="store_true",
        help="reuse and verify completed per-archive row shards in output-dir",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.validation_percent <= 50:
        raise ValueError("validation-percent must be in [1, 50]")
    targets = tuple(dict.fromkeys(str(value).strip() for value in args.target_archetype))
    if not targets or UNKNOWN in targets:
        raise ValueError("targets must be nonempty and must not include unknown")
    archives = sorted({path.resolve() for path in args.archive})
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "row-shards"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    tasks = []
    for archive in archives:
        digest = sha256_file(archive)
        output = cache_dir / f"{archive.stem}.public-tree-rows.npz"
        task = {
            "archive": str(archive),
            "archive_digest": digest,
            "output": str(output),
            "mix": str(args.mix.resolve()),
            "representatives": str(args.representatives.resolve()),
            "card_csv": str(args.card_csv.resolve()),
            "targets": targets,
            "additive": tuple(args.additive_archetype),
            "max_card_id": args.max_card_id,
            "seed": args.seed,
            "val_percent": args.validation_percent,
        }
        if args.reuse_row_shards and output.is_file():
            metadata = _load_shard_metadata(output)
            if (
                metadata.get("archive") != archive.name
                or metadata.get("archive_digest") != digest
                or metadata.get("input") != "cumulative_opponent_public_card_ids_only"
                or metadata.get("hidden_field_guard") is not True
            ):
                raise ValueError(f"cached public-tree row shard drifted: {output}")
            source_rows.append(
                {
                    "archive": archive.name,
                    "archive_digest": digest,
                    "output": str(output),
                    "output_digest": sha256_file(output),
                    "episodes": int(metadata["episodes"]),
                    "acting_seats": int(metadata["acting_seats"]),
                    "public_states": int(metadata["public_states"]),
                    "observations": int(metadata["observations"]),
                }
            )
        else:
            tasks.append(task)

    started = time.time()
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_build_archive_shard, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                source_rows.append(row)
                print(json.dumps({"stage": "public_rows", **row}, sort_keys=True), flush=True)

    import numpy as np
    from scipy import sparse
    from sklearn.tree import DecisionTreeClassifier

    train_x = []
    train_y = []
    train_w = []
    val_x = []
    val_y = []
    val_w = []
    for row in sorted(source_rows, key=lambda value: value["archive"]):
        matrix, labels, weights, splits = _load_shard(Path(row["output"]))
        train_mask = splits == 0
        val_mask = splits == 1
        train_x.append(matrix[train_mask])
        train_y.append(labels[train_mask])
        train_w.append(weights[train_mask])
        val_x.append(matrix[val_mask])
        val_y.append(labels[val_mask])
        val_w.append(weights[val_mask])

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
        precision_floor=float(args.runtime_precision_floor),
    )
    payload = {
        "schema": SCHEMA,
        "created_at_unix": time.time(),
        "runtime_enabled": False,
        "activation_status": "validation_required",
        "input_contract": {
            "features": "cumulative binary opponent-public card IDs",
            "public_zones": ["active", "bench", "discard", "attached_tools", "attached_energy", "public_pre_evolution"],
            "uses_our_deck": False,
            "uses_our_archetype": False,
            "prohibited": ["opponent_hand", "opponent_deck", "opponent_prizes", "agent_identity", "recorded_archetype", "future_observations"],
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
        },
        "sources": sorted(source_rows, key=lambda value: value["archive"]),
        "validation": validation,
        "runtime_calibration": runtime_calibration,
        "calibration_contract": {
            "probability_columns": "expanded_to_canonical_class_indexes",
            "canonical_class_count": len(class_names),
            "fitted_class_indexes": [
                int(value) for value in model.classes_
            ],
            "missing_fitted_classes_receive_zero_probability": True,
        },
        "tree": _tree_payload(model, class_names),
    }
    artifact = output_dir / "public-matchup-tree.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=artifact.name + ".", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
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
    receipt_path = output_dir / "PUBLIC_MATCHUP_TREE_READY.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
