"""Create-only publication receipts for revision-9 finalized rollout shards.

The transport may write only a private partial.  This boundary reopens the
destination bytes, verifies their full content address and exact size, creates
the final name without replacement, and then writes the closed revision-9
per-shard receipt.  It never loads or decodes feature rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


GOAL_REVISION = 9
GOAL_CONTRACT_PATH = "goals/alakazam-elmo-rule-derivative/contract.json"
GOAL_CONTRACT_SHA256 = (
    "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
)
MAXIMUM_FINAL_SHARD_BYTES = 15_000_000_000
PER_SHARD_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_elmo_refeaturization_shard_transfer_receipt/v2"
)
DESTINATION_ROOT = Path(
    "/home/pokebot/poke-bot-agent/outputs/quarantine/"
    "alakazam-elmo-rule-derivative/g9-recent20-15gb-refeaturized-shards"
)
_SHA256 = re.compile(r"^sha256:([0-9a-f]{64})$")


class Rev9TransferError(RuntimeError):
    """A private transfer object cannot be safely published."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise Rev9TransferError(f"receipt already exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        body = canonical_bytes(payload)
        written = 0
        while written < len(body):
            written += os.write(descriptor, body[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


def _validate_contract(repo_root: Path) -> None:
    contract = repo_root / GOAL_CONTRACT_PATH
    if not contract.is_file() or contract.is_symlink():
        raise Rev9TransferError("revision-9 contract is absent or not a regular file")
    if sha256_file(contract) != GOAL_CONTRACT_SHA256:
        raise Rev9TransferError("revision-9 contract identity drift")
    payload = json.loads(contract.read_text(encoding="utf-8"))
    if payload.get("goal_revision") != GOAL_REVISION:
        raise Rev9TransferError("revision-9 contract revision mismatch")


def finalize_private_partial(
    *,
    repo_root: Path,
    destination_root: Path,
    partial_path: Path,
    content_addressed_filename: str,
    expected_sha256: str,
    expected_size_bytes: int,
    source_shard_logical_id: str,
    validated_30_day_raw_manifest_sha256: str,
    feature_schema_sha256: str,
    target_schema_sha256: str,
    checklist_provenance_schema_sha256: str,
    transfer_disposition: str,
    parallel_lane_id: str,
    started_at_utc: str,
    validated_at_utc: str,
) -> tuple[dict[str, Any], Path, Path, str]:
    """Validate, publish, and receipt one complete private destination object."""

    _validate_contract(repo_root)
    match = _SHA256.fullmatch(expected_sha256)
    if match is None:
        raise Rev9TransferError("expected SHA-256 is malformed")
    if not 0 < expected_size_bytes <= MAXIMUM_FINAL_SHARD_BYTES:
        raise Rev9TransferError("expected size is outside the revision-9 boundary")
    if content_addressed_filename != f"sha256-{match.group(1)}.jsonl":
        raise Rev9TransferError("final filename does not bind the full content digest")
    if transfer_disposition not in {"copied", "resumed", "skipped_exact"}:
        raise Rev9TransferError("invalid transfer disposition")
    for label, digest in (
        ("raw manifest", validated_30_day_raw_manifest_sha256),
        ("feature schema", feature_schema_sha256),
        ("target schema", target_schema_sha256),
        ("checklist schema", checklist_provenance_schema_sha256),
    ):
        if _SHA256.fullmatch(digest) is None:
            raise Rev9TransferError(f"{label} digest is malformed")
    if destination_root.resolve() != DESTINATION_ROOT:
        raise Rev9TransferError("foreign revision-9 destination root")
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise Rev9TransferError("destination root is absent, foreign, or a symlink")
    partial_root = destination_root / ".partial"
    if partial_path.parent.resolve() != partial_root.resolve():
        raise Rev9TransferError("partial is outside the private partial root")
    if partial_path.is_symlink() or not partial_path.is_file():
        raise Rev9TransferError("private partial is absent or not a regular file")
    observed_size = partial_path.stat().st_size
    observed_sha256 = sha256_file(partial_path)
    if observed_size != expected_size_bytes or observed_sha256 != expected_sha256:
        raise Rev9TransferError("private destination bytes fail SHA/size validation")

    final_path = destination_root / content_addressed_filename
    disposition = transfer_disposition
    if final_path.exists() or final_path.is_symlink():
        if final_path.is_symlink() or not final_path.is_file():
            raise Rev9TransferError("conflicting final destination object")
        if final_path.stat().st_size != expected_size_bytes or sha256_file(final_path) != expected_sha256:
            raise Rev9TransferError("conflicting final destination content")
        disposition = "skipped_exact"
    else:
        os.link(partial_path, final_path, follow_symlinks=False)
        os.chmod(final_path, 0o444, follow_symlinks=False)
        directory_fd = os.open(destination_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if partial_path.exists():
        partial_path.unlink()

    receipt = {
        "schema": PER_SHARD_RECEIPT_SCHEMA,
        "goal_contract_path": GOAL_CONTRACT_PATH,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "validated_30_day_raw_manifest_sha256": validated_30_day_raw_manifest_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "target_schema_sha256": target_schema_sha256,
        "checklist_provenance_schema_sha256": checklist_provenance_schema_sha256,
        "source_shard_logical_id": source_shard_logical_id,
        "content_addressed_filename": content_addressed_filename,
        "source_sha256": expected_sha256,
        "source_size_bytes": expected_size_bytes,
        "source_schema_validation_passed": True,
        "destination_host": "inzi",
        "destination_directory": str(destination_root),
        "destination_path": str(final_path),
        "destination_sha256": observed_sha256,
        "destination_size_bytes": observed_size,
        "destination_validation_passed": True,
        "transfer_disposition_copied_resumed_or_skipped_exact": disposition,
        "parallel_lane_id": parallel_lane_id,
        "started_at_utc": started_at_utc,
        "validated_at_utc": validated_at_utc,
        "conflict_refused": False,
        "inzi_loading_or_execution_authority": False,
    }
    receipt_path = (
        destination_root
        / "receipts"
        / f"sha256-{match.group(1)}.transfer-receipt.json"
    )
    receipt_sha256 = _write_json_create_only(receipt_path, receipt)
    return receipt, final_path, receipt_path, receipt_sha256


__all__ = [
    "DESTINATION_ROOT",
    "GOAL_CONTRACT_SHA256",
    "GOAL_REVISION",
    "MAXIMUM_FINAL_SHARD_BYTES",
    "PER_SHARD_RECEIPT_SCHEMA",
    "Rev9TransferError",
    "canonical_bytes",
    "finalize_private_partial",
    "sha256_file",
]
