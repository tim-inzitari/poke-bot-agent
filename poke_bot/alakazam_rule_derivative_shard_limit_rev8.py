"""Revision-8 acceptance for completed revision-7 one-GiB feature shards.

The active re-featurizers were launched under revision 7 after the owner had
already selected one-GiB final shards.  Revision 8 records that correction.
This module validates the two immutable half-run receipts and their published
objects without restarting, retagging, or recomputing either run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


REV8_GOAL_REVISION = 8
REV8_GATEWAY_SHA256 = "sha256:3e710a6f474e096e2562c8a42d6c886e78009baf622c9fd6cb68901657c7ced4"
REV8_CONTRACT_SHA256 = "sha256:b522af1617f02a49522302947f1a4841ef24db7213f0e2ea8abeaba1332fb2cc"
REV7_GATEWAY_SHA256 = "sha256:d275696bb8322c1741463d63b4506d9b04f1252c3c6cdbe7559c999b55b83da7"
REV7_CONTRACT_SHA256 = "sha256:98e39d771569bdf778885848fc61732275c89c517e057a34bd3006144c23bdd1"
MAXIMUM_FINAL_SHARD_BYTES = 1024**3
CARD_ID_FILTER = 743
RUN_SCHEMA = "poke_bot.alakazam_fast_distributed_refeature/v1"
MIGRATION_SCHEMA = "poke_bot.alakazam_rule_derivative_revision_8_shard_limit_migration_receipt/v1"
SHARD_SUFFIX = ".refeaturization-census.shard"
WINDOW_DAYS = tuple(f"2026-07-{day:02d}" for day in range(13, 32)) + tuple(
    f"2026-08-{day:02d}" for day in range(1, 12)
)


class Rev8ShardLimitError(RuntimeError):
    """Raised when a revision-7 output cannot be accepted under revision 8."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Rev8ShardLimitError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise Rev8ShardLimitError(f"JSON artifact is not an object: {path}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_revision_8_contract(*, repo_root: Path | None = None) -> Mapping[str, Any]:
    root = (repo_root or _repo_root()).resolve()
    gateway = root / "goals/alakazam-elmo-rule-derivative/GOAL.md"
    contract_path = root / "goals/alakazam-elmo-rule-derivative/contract.json"
    if gateway.is_symlink() or contract_path.is_symlink():
        raise Rev8ShardLimitError("revision-8 authority files may not be symlinks")
    if sha256_file(gateway) != REV8_GATEWAY_SHA256 or sha256_file(contract_path) != REV8_CONTRACT_SHA256:
        raise Rev8ShardLimitError("revision-8 authority identity drifted")
    contract = _read_json(contract_path)
    revision = contract.get("revision_8_one_gib_content_addressed_shards")
    if contract.get("goal_revision") != REV8_GOAL_REVISION or not isinstance(revision, Mapping):
        raise Rev8ShardLimitError("revision-8 contract is missing its shard-limit authority")
    if (
        revision.get("predecessor_gateway_sha256"),
        revision.get("predecessor_contract_sha256"),
        revision.get("maximum_final_content_addressed_shard_bytes"),
        revision.get("maximum_transfer_object_bytes"),
        revision.get("in_flight_revision_7_runs_may_complete_without_restart"),
    ) != (
        REV7_GATEWAY_SHA256,
        REV7_CONTRACT_SHA256,
        MAXIMUM_FINAL_SHARD_BYTES,
        MAXIMUM_FINAL_SHARD_BYTES,
        True,
    ):
        raise Rev8ShardLimitError("revision-8 shard-limit contract drifted")
    return contract


def _validate_content_addressed_shards(
    receipt: Mapping[str, Any], *, output_root: Path
) -> list[dict[str, Any]]:
    manifest = receipt.get("shard_manifest")
    if not isinstance(manifest, Mapping):
        raise Rev8ShardLimitError("half-run receipt lacks a shard manifest")
    rows = manifest.get("shards")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or not rows:
        raise Rev8ShardLimitError("half-run shard inventory is empty or malformed")
    if (
        manifest.get("maximum_shard_size_bytes") != MAXIMUM_FINAL_SHARD_BYTES
        or manifest.get("record_scope") != "acting_seat_card_743_materialized_rows"
        or manifest.get("all_shards_individually_schema_sha256_size_validated") is not True
    ):
        raise Rev8ShardLimitError("half-run shard manifest does not bind the revision-8 limit/scope")
    normalized: list[dict[str, Any]] = []
    total_records = 0
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise Rev8ShardLimitError("half-run shard row is malformed")
        filename = raw.get("filename")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        records = raw.get("record_count")
        if not isinstance(filename, str) or not filename.endswith(SHARD_SUFFIX):
            raise Rev8ShardLimitError("half-run shard filename is malformed")
        if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
            raise Rev8ShardLimitError("half-run shard digest is malformed")
        if filename != f"sha256-{digest[7:]}{SHARD_SUFFIX}":
            raise Rev8ShardLimitError("half-run shard filename is not content addressed")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAXIMUM_FINAL_SHARD_BYTES:
            raise Rev8ShardLimitError("half-run shard exceeds one GiB or has an invalid size")
        if not isinstance(records, int) or isinstance(records, bool) or records < 0:
            raise Rev8ShardLimitError("half-run shard record count is malformed")
        path = output_root / "refeatured-records/shards" / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise Rev8ShardLimitError(f"half-run shard bytes failed SHA/size validation: {path}")
        normalized.append({"filename": filename, "sha256": digest, "size_bytes": size, "record_count": records})
        total_records += records
    if manifest.get("shard_count") != len(normalized) or manifest.get("record_count") != total_records:
        raise Rev8ShardLimitError("half-run shard counts do not close")
    return normalized


def validate_revision_7_half_run_under_revision_8(
    receipt_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    receipt = _read_json(receipt_path)
    root = (output_root or receipt_path.parent).resolve()
    if receipt.get("schema") != RUN_SCHEMA or receipt.get("status") != "complete":
        raise Rev8ShardLimitError("half-run completion receipt schema/status drifted")
    host_index = receipt.get("host_index")
    days = receipt.get("days")
    if host_index not in (0, 1) or not isinstance(days, list):
        raise Rev8ShardLimitError("half-run host/day partition is malformed")
    expected_days = list(WINDOW_DAYS[host_index::2])
    if days != expected_days or receipt.get("day_count") != 15:
        raise Rev8ShardLimitError("half-run does not cover its exact 15-day partition")
    if (
        receipt.get("card_id_filter"),
        receipt.get("acting_seat_only"),
        receipt.get("maximum_final_shard_bytes"),
    ) != (CARD_ID_FILTER, True, MAXIMUM_FINAL_SHARD_BYTES):
        raise Rev8ShardLimitError("half-run row scope or final shard limit drifted")
    shards = _validate_content_addressed_shards(receipt, output_root=root)
    return {
        "host_index": host_index,
        "days": days,
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_canonical_sha256": canonical_sha256(receipt),
        "record_count": receipt.get("record_count"),
        "shard_count": len(shards),
        "shards": shards,
    }


def build_revision_8_shard_limit_migration_receipt(
    *,
    half_run_receipt_paths: Sequence[Path],
    raw_manifest_sha256: str,
    repo_root: Path | None = None,
    sealed_at_utc: str,
) -> dict[str, Any]:
    load_revision_8_contract(repo_root=repo_root)
    if len(half_run_receipt_paths) != 2:
        raise Rev8ShardLimitError("revision-8 migration requires exactly two half-run receipts")
    runs = sorted(
        (validate_revision_7_half_run_under_revision_8(path) for path in half_run_receipt_paths),
        key=lambda row: int(row["host_index"]),
    )
    if [row["host_index"] for row in runs] != [0, 1]:
        raise Rev8ShardLimitError("revision-8 migration requires complementary host partitions")
    if sorted(day for row in runs for day in row["days"]) != sorted(WINDOW_DAYS):
        raise Rev8ShardLimitError("revision-8 migration does not close the exact 30-day window")
    if not isinstance(raw_manifest_sha256, str) or not raw_manifest_sha256.startswith("sha256:"):
        raise Rev8ShardLimitError("raw manifest digest is malformed")
    return {
        "schema": MIGRATION_SCHEMA,
        "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
        "goal_contract_sha256": REV8_CONTRACT_SHA256,
        "goal_revision": REV8_GOAL_REVISION,
        "predecessor_gateway_sha256": REV7_GATEWAY_SHA256,
        "predecessor_contract_sha256": REV7_CONTRACT_SHA256,
        "validated_30_day_raw_manifest_sha256": raw_manifest_sha256,
        "revision_7_run_receipt_sha256s": [row["receipt_file_sha256"] for row in runs],
        "logical_routing_unchanged": True,
        "all_final_objects_content_addressed": True,
        "all_final_objects_complete_and_immutable": True,
        "all_final_objects_maximum_size_bytes": MAXIMUM_FINAL_SHARD_BYTES,
        "all_final_objects_at_or_below_one_gib": True,
        "private_partials_transferred": False,
        "restart_or_recompute_due_only_to_revision_8": False,
        "migration_validation_passed": True,
        "sealed_at_utc": sealed_at_utc,
    }


def write_revision_8_shard_limit_migration_receipt_create_only(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Rev8ShardLimitError(f"revision-8 receipt path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, _canonical_bytes(receipt))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MAXIMUM_FINAL_SHARD_BYTES",
    "MIGRATION_SCHEMA",
    "Rev8ShardLimitError",
    "build_revision_8_shard_limit_migration_receipt",
    "canonical_sha256",
    "load_revision_8_contract",
    "sha256_file",
    "validate_revision_7_half_run_under_revision_8",
    "write_revision_8_shard_limit_migration_receipt_create_only",
]
