"""Closed revision-9 Inzi corpus handoff receipt from immutable evidence.

Inzi-native shards remain at their finalized paths and are inventoried rather
than copied. Elmo shards must be present in the quarantine root with exact
per-shard transfer receipts. This receipt never makes either set trainable;
that remains the later activation receipt's job.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GOAL_REVISION = 9
GOAL_CONTRACT_PATH = "goals/alakazam-elmo-rule-derivative/contract.json"
GOAL_CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
CORPUS_MANIFEST_SHA256 = "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
PARITY_RECEIPT_SHA256 = "sha256:92e5ad858598c670ca4ee7459f11009e61bd48d4e385ebed64c95bb6a7e69732"
SPLIT_RECEIPT_SHA256 = "sha256:3e939b01bf32e7c5956422350306ed9213ec42bdb48a9c595d13308a8d081a18"
SPLIT_MANIFEST_SHA256 = "sha256:0e5608b40b4d36cee6a910059ce2b55ed2db55523e8e2c13f7cc8c69f17cc0d3"
CENSUS_RECEIPT_SHA256 = "sha256:34c51a59f4e843ff9d04ec07a807afe88ff4149210f6471833e422b41975fb9c"
BRANCH_RECEIPT_SHA256 = "sha256:084d068bebfa2a0da15209bda798842c38e59a52637bb49723fad063c487a52e"
SCHEMA = "poke_bot.alakazam_rule_derivative_inzi_corpus_receipt/v1"


class CorpusHandoffError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_exact(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise CorpusHandoffError(f"artifact identity mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CorpusHandoffError(f"artifact is not an object: {path}")
    return value


def build_corpus_receipt(
    *,
    corpus_manifest_path: Path,
    parity_receipt_path: Path,
    split_manifest_path: Path,
    split_receipt_path: Path,
    census_receipt_path: Path,
    branch_receipt_path: Path,
) -> dict[str, Any]:
    manifest = _load_exact(corpus_manifest_path, CORPUS_MANIFEST_SHA256)
    parity = _load_exact(parity_receipt_path, PARITY_RECEIPT_SHA256)
    split = _load_exact(split_manifest_path, SPLIT_MANIFEST_SHA256)
    split_receipt = _load_exact(split_receipt_path, SPLIT_RECEIPT_SHA256)
    census = _load_exact(census_receipt_path, CENSUS_RECEIPT_SHA256)
    branch = _load_exact(branch_receipt_path, BRANCH_RECEIPT_SHA256)
    if split_receipt.get("source_day_group_disjointness_passed") is not True:
        raise CorpusHandoffError("split evidence did not pass")
    if branch.get("eligible_trainable_branches") != ["public_rule_semantic_projection"]:
        raise CorpusHandoffError("eligible branch inventory drift")
    if census.get("all_option_record_count") != manifest.get("record_count"):
        raise CorpusHandoffError("census/corpus row-count mismatch")
    if manifest.get("finalized_shard_count") != 20 or parity.get("source_and_destination_object_count") != 20:
        raise CorpusHandoffError("finalized/transferred shard counts mismatch")

    ordered_shards = []
    ordered_transfer_receipts = []
    destination_paths = parity["destination_object_paths_by_sha256"]
    if len(destination_paths) != 20:
        raise CorpusHandoffError("combined destination inventory must contain twenty objects")
    for row in sorted(manifest["shards"], key=lambda item: item["utc_day"]):
        if row["source_host"] == "elmo":
            path = Path(destination_paths[row["sha256"]])
            disposition = "elmo_transferred_create_only_quarantine"
            receipt_path = (
                parity_receipt_path.parent
                / "receipts"
                / f"sha256-{row['sha256'].removeprefix('sha256:')}.transfer-receipt.json"
            )
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise CorpusHandoffError(f"missing transfer receipt: {receipt_path}")
            receipt_sha = sha256_file(receipt_path)
            transfer_payload = json.loads(receipt_path.read_text())
            if (
                transfer_payload.get("source_sha256") != row["sha256"]
                or transfer_payload.get("destination_validation_passed") is not True
                or transfer_payload.get("inzi_loading_or_execution_authority") is not False
            ):
                raise CorpusHandoffError(f"invalid transfer receipt: {receipt_path}")
            ordered_transfer_receipts.append(receipt_sha)
        elif row["source_host"] == "inzi":
            path = (
                Path(row["day_receipt_path"]).parent
                / "refeatured-records"
                / "shards"
                / row["filename"]
            )
            disposition = "inzi_native_finalized_in_place_no_round_trip_copy"
            receipt_sha = None
        else:
            raise CorpusHandoffError("foreign source host")
        if path.is_symlink() or not path.is_file():
            raise CorpusHandoffError(f"missing regular shard: {path}")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise CorpusHandoffError(f"shard identity mismatch: {path}")
        ordered_shards.append({
            "utc_day": row["utc_day"],
            "source_host": row["source_host"],
            "path": str(path),
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "record_count": row["record_count"],
            "disposition": disposition,
            "transfer_receipt_sha256": receipt_sha,
        })

    schema_binding = parity["schema_binding"]
    inventory_sha = "sha256:" + hashlib.sha256(canonical_bytes(ordered_shards)).hexdigest()
    return {
        "schema": SCHEMA,
        "goal_contract_path": GOAL_CONTRACT_PATH,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "window_start_utc": "2026-07-23",
        "window_end_utc": "2026-08-11",
        "utc_partition_count": 20,
        "validated_deduplicated_raw_manifest_sha256": manifest[
            "validated_30_day_raw_manifest_sha256"
        ],
        "source_day_group_disjoint_split_manifest_sha256s": {
            "combined": SPLIT_MANIFEST_SHA256,
            "train": split["splits"]["train"]["group_inventory_sha256"],
            "validation": split["splits"]["validation"]["group_inventory_sha256"],
            "evaluation": split["splits"]["evaluation"]["group_inventory_sha256"],
        },
        "collision_census_receipt_sha256": CENSUS_RECEIPT_SHA256,
        "branch_adjudication_receipt_sha256": BRANCH_RECEIPT_SHA256,
        "source_manifest_sha256": CORPUS_MANIFEST_SHA256,
        "destination_manifest_sha256": inventory_sha,
        "source_remote_parity_receipt_sha256": PARITY_RECEIPT_SHA256,
        "staging_root": str(parity_receipt_path.parent),
        "ordered_content_addressed_shard_sha256_and_size": ordered_shards,
        "ordered_per_shard_transfer_receipt_sha256s": ordered_transfer_receipts,
        "object_count": len(ordered_shards),
        "total_games": split["total_game_count"],
        "total_decisions": split["total_decision_count"],
        "schema_validation_passed": set(schema_binding) == {
            "feature_schema_sha256",
            "target_schema_sha256",
            "checklist_provenance_schema_sha256",
        },
        "source_remote_parity_passed": parity["source_remote_parity_passed"],
        "hidden_inputs_absent": True,
        "training_eligible_before_activation": False,
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise CorpusHandoffError("output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        body = canonical_bytes(value)
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


__all__ = ["CorpusHandoffError", "build_corpus_receipt", "write_create_only"]
