#!/usr/bin/env python3
"""Pack every recent-20 semantic decision occurrence without deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


RECORD_SCHEMA = "poke_bot.alakazam_collision_census_r298_option_record/v1"
HEADER_SCHEMA = "poke_bot.alakazam_recent20_intraday_refeature_shard/v1"
PACK_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not (result == result and abs(result) != float("inf")):
        return 0.0
    return max(-65535.0, min(65535.0, result)) / 65535.0


def candidate_features(row: dict[str, Any]) -> tuple[float, ...]:
    key = row["complete_semantic_option_key"]
    stage = key["factorized_stage"]
    if stage["candidate_semantics"]:
        candidate = stage["candidate_semantics"][-1]
        option = candidate["option"]
        selection = candidate["selection"]
    elif stage.get("kind") == "stop":
        option = {"option_type": "pass"}
        selection = key["normalized_selection"]
    else:
        raise ValueError("empty candidate semantics outside STOP stage")
    option_kinds = ("attack", "skill", "number", "card", "pass", "retreat", "target", "unknown")
    selection_kinds = ("card", "pokemon", "energy", "target", "number", "unknown")
    option_type = str(option.get("option_type", "unknown")).split(":")[-1]
    selection_type = str(selection.get("selection_type", "unknown")).split(":")[-1]
    values = [1.0 if option_type == name else 0.0 for name in option_kinds]
    values += [1.0 if selection_type == name else 0.0 for name in selection_kinds]
    values += [
        finite(option.get("number")), finite(option.get("count")),
        finite(selection.get("min_count")), finite(selection.get("max_count")),
        finite(selection.get("remain_damage_counter")),
        finite(selection.get("remain_energy_cost")),
    ]
    values += [0.0] * 16
    values += [
        1.0 if option.get("source") is True else 0.0,
        1.0 if option.get("target") is True else 0.0,
        1.0 if option.get("stable_simulator_discriminator") is True else 0.0,
        finite(len(row.get("referenced_card_ids", ()))),
    ]
    if len(values) != 40:
        raise ValueError("semantic feature width drifted")
    return tuple(values)


def group_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    source = row["source"]
    return (
        source["source_archive_date"], source["episode_id"], source["acting_seat"],
        source["env_step"], source["factorized_stage"],
        json.dumps(row.get("factorized_stage_prefix", []), sort_keys=True, separators=(",", ":")),
    )


def write_group(rows: list[dict[str, Any]], features, offsets, selected, keys, counts: list[int]) -> None:
    chosen = [i for i, row in enumerate(rows) if row.get("selected_candidate") is True]
    if len(chosen) != 1:
        raise ValueError("factorized stage does not have exactly one selected candidate")
    for row in rows:
        features.write(struct.pack("<40f", *candidate_features(row)))
    counts[0] += len(rows)
    counts[1] += 1
    offsets.write(struct.pack("<Q", counts[0]))
    selected.write(struct.pack("<I", chosen[0]))
    keys.write(hashlib.sha256(canonical_bytes(group_identity(rows[0]))).digest())


def pack_one(task: tuple[str, str, str]) -> dict[str, Any]:
    source_raw, expected_sha, output_raw = task
    source, output = Path(source_raw), Path(output_raw)
    if sha256_file(source) != expected_sha:
        raise ValueError(f"source digest mismatch: {source}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "features_f32": output / "features.f32",
        "decision_offsets_u64": output / "decision_offsets.u64",
        "selected_option_u32": output / "selected_option.u32",
        "decision_key_sha256": output / "decision_keys.sha256",
    }
    counts = [0, 0]
    with (
        source.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as stream,
        paths["features_f32"].open("xb", buffering=16 * 1024 * 1024) as features,
        paths["decision_offsets_u64"].open("xb", buffering=4 * 1024 * 1024) as offsets,
        paths["selected_option_u32"].open("xb", buffering=4 * 1024 * 1024) as selected,
        paths["decision_key_sha256"].open("xb", buffering=4 * 1024 * 1024) as keys,
    ):
        header = json.loads(stream.readline())
        if header.get("schema") != HEADER_SCHEMA:
            raise ValueError("source header schema drifted")
        offsets.write(struct.pack("<Q", 0))
        current_key = None
        rows: list[dict[str, Any]] = []
        for line in stream:
            row = json.loads(line)
            if row.get("schema") != RECORD_SCHEMA:
                raise ValueError("record schema drifted")
            key = group_identity(row)
            if current_key is not None and key != current_key:
                write_group(rows, features, offsets, selected, keys, counts)
                rows = []
            current_key = key
            rows.append(row)
        if rows:
            write_group(rows, features, offsets, selected, keys, counts)
        for handle in (features, offsets, selected, keys):
            handle.flush()
            os.fsync(handle.fileno())
    files = {
        name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    receipt = {
        "schema": PACK_SCHEMA,
        "source_path": str(source), "source_sha256": expected_sha,
        "feature_width": 40, "feature_dtype": "float32_le",
        "option_occurrence_count": counts[0], "decision_occurrence_count": counts[1],
        "deduplicated": False, "files": files,
    }
    receipt_path = output / "receipt.json"
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, canonical_bytes(receipt)); os.fsync(fd)
    finally:
        os.close(fd)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.corpus_manifest.read_text())
    parity = json.loads(args.parity_receipt.read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    tasks = []
    for row in manifest["shards"]:
        path = parity["destination_object_paths_by_sha256"][row["sha256"]]
        tasks.append((path, row["sha256"], str(args.output_root / row["utc_day"])))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(pack_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps({"packed": result["source_sha256"], "decisions": result["decision_occurrence_count"]}), flush=True)
    results.sort(key=lambda row: row["source_path"])
    completion = {
        "schema": "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1",
        "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
        "source_shard_count": len(results),
        "option_occurrence_count": sum(x["option_occurrence_count"] for x in results),
        "decision_occurrence_count": sum(x["decision_occurrence_count"] for x in results),
        "deduplicated": False, "packs": results,
    }
    path = args.output_root / "COMPLETE.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, canonical_bytes(completion)); os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"complete": str(path), "decisions": completion["decision_occurrence_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
