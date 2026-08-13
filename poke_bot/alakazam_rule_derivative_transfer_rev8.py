"""Revision-8 finalized-shard transfer and parity receipts.

This is deliberately a small post-publication boundary.  It never reads worker
spools or ``*.partial`` files.  The Inzi-produced half remains local; only the
validated Elmo half is planned for create-only publication into the revision-8
quarantine root.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .alakazam_rule_derivative_shard_limit_rev8 import (
    MAXIMUM_FINAL_SHARD_BYTES,
    REV8_CONTRACT_SHA256,
    Rev8ShardLimitError,
    load_revision_8_contract,
    sha256_file,
)


MAX_TRANSFER_LANES = 4
INZI_QUARANTINE_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/quarantine/"
    "alakazam-elmo-rule-derivative/g8-one-gib-refeaturized-census-shards"
)
PER_SHARD_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_elmo_refeaturization_shard_transfer_receipt/v2"
)
PARITY_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_elmo_refeaturization_transfer_parity_receipt/v2"
)
PLAN_SCHEMA = "poke_bot.alakazam_elmo_refeaturization_transfer_plan/v2"


class Rev8TransferError(Rev8ShardLimitError):
    """A completed shard or its remote observation is not transferable."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _shards(run: Mapping[str, Any], *, host_index: int) -> list[Mapping[str, Any]]:
    if run.get("host_index") != host_index:
        raise Rev8TransferError(f"expected host index {host_index}")
    rows = run.get("shards")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise Rev8TransferError("validated half-run has no shard inventory")
    normalized: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise Rev8TransferError("validated shard row is malformed")
        filename = row.get("filename")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not isinstance(filename, str)
            or filename.startswith(".")
            or ".partial" in filename
            or not isinstance(digest, str)
            or filename != f"sha256-{digest.removeprefix('sha256:')}.refeaturization-census.shard"
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 < size <= MAXIMUM_FINAL_SHARD_BYTES
        ):
            raise Rev8TransferError("validated shard is not a final one-GiB content address")
        if digest in seen:
            raise Rev8TransferError("duplicate shard digest")
        seen.add(digest)
        normalized.append(row)
    return normalized


def build_transfer_plan(
    *,
    inzi_half: Mapping[str, Any],
    elmo_half: Mapping[str, Any],
    raw_manifest_sha256: str,
    migration_receipt_sha256: str,
    lanes: int = MAX_TRANSFER_LANES,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic plan containing only finalized Elmo objects."""

    load_revision_8_contract(repo_root=repo_root)
    if isinstance(lanes, bool) or not isinstance(lanes, int) or not 1 <= lanes <= 4:
        raise Rev8TransferError("transfer lanes must be between one and four")
    if not raw_manifest_sha256.startswith("sha256:") or not migration_receipt_sha256.startswith(
        "sha256:"
    ):
        raise Rev8TransferError("plan evidence digest is malformed")
    local = _shards(inzi_half, host_index=0)
    remote = _shards(elmo_half, host_index=1)
    entries = []
    for index, row in enumerate(sorted(remote, key=lambda item: str(item["sha256"]))):
        filename = str(row["filename"])
        entries.append(
            {
                "source_host": "elmo",
                "source_filename": filename,
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "record_count": row["record_count"],
                "destination_host": "inzi",
                "destination_path": str(INZI_QUARANTINE_ROOT / filename),
                "lane": index % lanes,
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "goal_contract_sha256": REV8_CONTRACT_SHA256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "revision_8_migration_receipt_sha256": migration_receipt_sha256,
        "maximum_object_bytes": MAXIMUM_FINAL_SHARD_BYTES,
        "requested_parallel_lanes": lanes,
        "destination_root": str(INZI_QUARANTINE_ROOT),
        "private_partials_transferable": False,
        "inzi_local_shard_count": len(local),
        "inzi_local_shard_bytes": sum(int(row["size_bytes"]) for row in local),
        "elmo_transfer_shard_count": len(entries),
        "elmo_transfer_shard_bytes": sum(int(row["size_bytes"]) for row in remote),
        "entries": entries,
    }


def build_shard_transfer_receipt(
    *, plan: Mapping[str, Any], entry: Mapping[str, Any], remote_path: Path
) -> dict[str, Any]:
    """Re-open one final remote object and issue its exact parity observation."""

    if entry not in plan.get("entries", []):
        raise Rev8TransferError("entry is not part of the validated transfer plan")
    if remote_path.is_symlink() or not remote_path.is_file():
        raise Rev8TransferError("remote final object is absent or not a regular file")
    size = remote_path.stat().st_size
    digest = sha256_file(remote_path)
    if size != entry.get("size_bytes") or digest != entry.get("sha256"):
        raise Rev8TransferError("remote final object does not match the Elmo source")
    receipt = {
        "schema": PER_SHARD_RECEIPT_SCHEMA,
        "goal_contract_sha256": REV8_CONTRACT_SHA256,
        "transfer_plan_sha256": canonical_sha256(plan),
        "source_host": "elmo",
        "source_sha256": digest,
        "source_size_bytes": size,
        "destination_host": "inzi",
        "destination_path": str(remote_path),
        "destination_sha256": digest,
        "destination_size_bytes": size,
        "content_addressed_final_object": True,
        "private_partial_transferred": False,
        "validation_passed": True,
    }
    return receipt


def build_parity_receipt(
    *,
    plan: Mapping[str, Any],
    inzi_half: Mapping[str, Any],
    elmo_half: Mapping[str, Any],
    transfer_receipts: Sequence[Mapping[str, Any]],
    sealed_at_utc: str,
) -> dict[str, Any]:
    """Close the combined local-plus-transferred 30-day shard inventory."""

    local = _shards(inzi_half, host_index=0)
    remote = _shards(elmo_half, host_index=1)
    expected = {str(row["sha256"]): row for row in remote}
    observed: dict[str, Mapping[str, Any]] = {}
    for receipt in transfer_receipts:
        if receipt.get("schema") != PER_SHARD_RECEIPT_SCHEMA or receipt.get(
            "transfer_plan_sha256"
        ) != canonical_sha256(plan):
            raise Rev8TransferError("per-shard transfer receipt is foreign")
        digest = receipt.get("source_sha256")
        if digest not in expected or receipt.get("validation_passed") is not True:
            raise Rev8TransferError("per-shard transfer receipt does not close an Elmo shard")
        if digest in observed:
            raise Rev8TransferError("duplicate per-shard transfer receipt")
        if (
            receipt.get("destination_sha256") != digest
            or receipt.get("destination_size_bytes") != expected[digest]["size_bytes"]
            or receipt.get("private_partial_transferred") is not False
        ):
            raise Rev8TransferError("per-shard transfer receipt parity failed")
        observed[str(digest)] = receipt
    if set(observed) != set(expected):
        raise Rev8TransferError("Elmo transfer receipt inventory is incomplete")
    combined = sorted(
        [{"storage_host": "inzi", **dict(row)} for row in local]
        + [{"storage_host": "inzi", **dict(row)} for row in remote],
        key=lambda row: str(row["sha256"]),
    )
    return {
        "schema": PARITY_RECEIPT_SCHEMA,
        "goal_contract_sha256": REV8_CONTRACT_SHA256,
        "transfer_plan_sha256": canonical_sha256(plan),
        "raw_manifest_sha256": plan["raw_manifest_sha256"],
        "revision_8_migration_receipt_sha256": plan[
            "revision_8_migration_receipt_sha256"
        ],
        "inzi_local_shards_reused_without_round_trip": len(local),
        "elmo_shards_transferred_create_only": len(remote),
        "combined_shard_count": len(combined),
        "combined_shard_bytes": sum(int(row["size_bytes"]) for row in combined),
        "combined_inventory_sha256": canonical_sha256(combined),
        "all_objects_at_or_below_one_gib": all(
            int(row["size_bytes"]) <= MAXIMUM_FINAL_SHARD_BYTES for row in combined
        ),
        "private_partials_transferred": False,
        "parity_validation_passed": True,
        "sealed_at_utc": sealed_at_utc,
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    """Write one immutable canonical receipt without replacing existing data."""

    if path.exists() or path.is_symlink():
        raise Rev8TransferError(f"receipt path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, _canonical_bytes(value))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


__all__ = [
    "INZI_QUARANTINE_ROOT",
    "MAX_TRANSFER_LANES",
    "PARITY_RECEIPT_SCHEMA",
    "PER_SHARD_RECEIPT_SCHEMA",
    "PLAN_SCHEMA",
    "Rev8TransferError",
    "build_parity_receipt",
    "build_shard_transfer_receipt",
    "build_transfer_plan",
    "canonical_sha256",
    "write_json_create_only",
]
