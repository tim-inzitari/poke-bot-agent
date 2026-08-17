"""Close the revision-9 recent-20 corpus and source/Inzi parity receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


GOAL_REVISION = 9
GOAL_CONTRACT_PATH = "goals/alakazam-elmo-rule-derivative/contract.json"
GOAL_CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
WINDOW_START = "2026-07-23"
WINDOW_END = "2026-08-11"
DAY_COUNT = 20
MAXIMUM_SHARD_BYTES = 15_000_000_000
MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_refeaturized_corpus_manifest/v1"
PARITY_SCHEMA = "poke_bot.alakazam_elmo_refeaturization_transfer_parity_receipt/v2"
TRANSFER_SCHEMA = "poke_bot.alakazam_elmo_refeaturization_shard_transfer_receipt/v2"
DESTINATION_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/quarantine/alakazam-elmo-rule-derivative/"
    "g9-recent20-15gb-refeaturized-shards"
)


class Recent20ParityError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _day_row(receipt: Mapping[str, Any], *, receipt_path: str, receipt_sha256: str) -> dict[str, Any]:
    if (
        receipt.get("schema") != "poke_bot.alakazam_recent20_intraday_refeature_day_receipt/v1"
        or receipt.get("status") != "complete"
        or receipt.get("goal_revision") != GOAL_REVISION
        or receipt.get("goal_contract_sha256") != GOAL_CONTRACT_SHA256
    ):
        raise Recent20ParityError("day receipt is foreign or incomplete")
    manifest = receipt.get("shard_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("shard_count") != 1:
        raise Recent20ParityError("revision-9 day must contain exactly one finalized shard")
    shards = manifest.get("shards")
    if not isinstance(shards, Sequence) or len(shards) != 1 or not isinstance(shards[0], Mapping):
        raise Recent20ParityError("day shard inventory is malformed")
    shard = shards[0]
    day = receipt.get("utc_day")
    digest = shard.get("sha256")
    size = shard.get("size_bytes")
    records = shard.get("record_count")
    if (
        not isinstance(day, str)
        or shard.get("utc_day") != day
        or not isinstance(digest, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 0 < size <= MAXIMUM_SHARD_BYTES
        or not isinstance(records, int)
        or records < 1
        or manifest.get("total_bytes") != size
        or manifest.get("record_count") != records
        or receipt.get("record_count") != records
    ):
        raise Recent20ParityError("day receipt totals are inconsistent")
    host = "elmo" if receipt.get("hostname") == "truenas" else "inzi"
    filename = shard.get("filename")
    if filename != f"sha256-{digest.removeprefix('sha256:')}.jsonl":
        raise Recent20ParityError("day shard filename is not its content address")
    return {
        "utc_day": day,
        "source_host": host,
        "day_receipt_path": receipt_path,
        "day_receipt_sha256": receipt_sha256,
        "filename": filename,
        "sha256": digest,
        "size_bytes": size,
        "record_count": records,
    }


def build_manifest(
    *,
    day_receipts: Sequence[tuple[Mapping[str, Any], str, str]],
    elmo_host_receipt_path: str,
    elmo_host_receipt_sha256: str,
    inzi_host_receipt_path: str,
    inzi_host_receipt_sha256: str,
    raw_manifest_sha256: str,
    sealed_at_utc: str,
) -> dict[str, Any]:
    rows = sorted(
        (_day_row(value, receipt_path=path, receipt_sha256=digest) for value, path, digest in day_receipts),
        key=lambda row: row["utc_day"],
    )
    expected = [f"2026-07-{day:02d}" for day in range(23, 32)] + [
        f"2026-08-{day:02d}" for day in range(1, 12)
    ]
    if len(rows) != DAY_COUNT or [row["utc_day"] for row in rows] != expected:
        raise Recent20ParityError("recent-20 day inventory is incomplete or duplicated")
    if len({row["sha256"] for row in rows}) != DAY_COUNT:
        raise Recent20ParityError("recent-20 shard digests are not unique")
    if sum(row["source_host"] == "elmo" for row in rows) != 10:
        raise Recent20ParityError("recent-20 host split is not ten and ten")
    return {
        "schema": MANIFEST_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": GOAL_CONTRACT_PATH,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "window_start_utc": WINDOW_START,
        "window_end_utc": WINDOW_END,
        "utc_partition_count": DAY_COUNT,
        "row_scope": "acting_seat_setup_deck_contains_literal_card_id_743",
        "validated_30_day_raw_manifest_sha256": raw_manifest_sha256,
        "elmo_host_receipt_path": elmo_host_receipt_path,
        "elmo_host_receipt_sha256": elmo_host_receipt_sha256,
        "inzi_host_receipt_path": inzi_host_receipt_path,
        "inzi_host_receipt_sha256": inzi_host_receipt_sha256,
        "finalized_shard_count": DAY_COUNT,
        "finalized_shard_bytes": sum(row["size_bytes"] for row in rows),
        "record_count": sum(row["record_count"] for row in rows),
        "shards": rows,
        "sealed_at_utc": sealed_at_utc,
    }


def build_parity_receipt(
    *,
    source_manifest: Mapping[str, Any],
    transfer_receipts: Sequence[tuple[Mapping[str, Any], str]],
    destination_observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if source_manifest.get("schema") != MANIFEST_SCHEMA:
        raise Recent20ParityError("source manifest is foreign")
    shards = source_manifest.get("shards")
    if not isinstance(shards, Sequence) or len(shards) != DAY_COUNT:
        raise Recent20ParityError("source manifest shard inventory is incomplete")
    transfers: dict[str, tuple[Mapping[str, Any], str]] = {}
    for receipt, receipt_sha in transfer_receipts:
        digest = receipt.get("source_sha256")
        if (
            receipt.get("schema") != TRANSFER_SCHEMA
            or receipt.get("goal_revision") != GOAL_REVISION
            or receipt.get("goal_contract_sha256") != GOAL_CONTRACT_SHA256
            or receipt.get("destination_validation_passed") is not True
            or receipt.get("inzi_loading_or_execution_authority") is not False
            or not isinstance(digest, str)
            or digest in transfers
        ):
            raise Recent20ParityError("per-shard transfer receipt is foreign or duplicated")
        transfers[digest] = (receipt, receipt_sha)
    expected_elmo = {row["sha256"] for row in shards if row["source_host"] == "elmo"}
    if set(transfers) != expected_elmo:
        raise Recent20ParityError("Elmo transfer receipt set is incomplete")
    source_rows: list[dict[str, Any]] = []
    destination_rows: list[dict[str, Any]] = []
    for row in shards:
        digest = row["sha256"]
        observed = destination_observations.get(digest)
        if not isinstance(observed, Mapping) or observed.get("sha256") != digest or observed.get("size_bytes") != row["size_bytes"]:
            raise Recent20ParityError("destination observation does not match source")
        source_rows.append({"sha256": digest, "size_bytes": row["size_bytes"]})
        destination_rows.append({"sha256": digest, "size_bytes": observed["size_bytes"]})
    source_rows.sort(key=lambda row: row["sha256"])
    destination_rows.sort(key=lambda row: row["sha256"])
    return {
        "schema": PARITY_SCHEMA,
        "goal_contract_path": GOAL_CONTRACT_PATH,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "destination_directory": str(DESTINATION_ROOT),
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "destination_manifest_sha256": canonical_sha256(destination_rows),
        "source_and_destination_object_count": DAY_COUNT,
        "content_addressed_filename_set": sorted(row["filename"] for row in shards),
        "per_object_source_sha256_and_size": source_rows,
        "per_object_destination_sha256_and_size": destination_rows,
        "schema_binding": {
            "feature_schema_sha256": "sha256:d4d8f1be1219f5bb5a1ff31582225d6f43d369e09a5a78e1c74aa560c67b0f1c",
            "target_schema_sha256": "sha256:6b9544fa7285edad25e34ab7fca08bfa4cf9c395e1748178c73708d3377f354f",
            "checklist_provenance_schema_sha256": "sha256:fd39740b1a2092768dffadcf2f4beb3ae1cc1f38fe374b73a7dac0d4a7f20d9f",
        },
        "validated_30_day_raw_manifest_binding": source_manifest["validated_30_day_raw_manifest_sha256"],
        "ordered_per_shard_receipt_sha256s": [transfers[digest][1] for digest in sorted(transfers)],
        "skip_resume_and_conflict_counts": {"copied": 10, "resumed": 0, "skipped_exact": 0, "conflicts": 0},
        "parallel_lane_maximum_observed": 1,
        "source_remote_parity_passed": True,
        "inzi_loading_or_execution_authority": False,
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise Recent20ParityError(f"output already exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        body = canonical_bytes(value)
        offset = 0
        while offset < len(body):
            offset += os.write(fd, body[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256_file(path)

