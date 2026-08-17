"""Register an Inzi-native finalized shard in rev9 quarantine without copying bytes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from poke_bot import alakazam_rule_derivative_transfer_rev9 as transfer


class LocalInventoryError(RuntimeError):
    pass


def register_local_finalized_shard(
    *,
    repo_root: Path,
    source_path: Path,
    destination_root: Path,
    content_addressed_filename: str,
    expected_sha256: str,
    expected_size_bytes: int,
    source_shard_logical_id: str,
    validated_30_day_raw_manifest_sha256: str,
    feature_schema_sha256: str,
    target_schema_sha256: str,
    checklist_provenance_schema_sha256: str,
    parallel_lane_id: str,
    started_at_utc: str,
    validated_at_utc: str,
) -> tuple[dict[str, Any], Path, Path, str]:
    if source_path.is_symlink() or not source_path.is_file():
        raise LocalInventoryError("native source is absent or not a regular file")
    if source_path.stat().st_dev != destination_root.stat().st_dev:
        raise LocalInventoryError("native source and quarantine are on different filesystems")
    final_path = destination_root / content_addressed_filename
    receipt_path = (
        destination_root
        / "receipts"
        / f"sha256-{expected_sha256.removeprefix('sha256:')}.transfer-receipt.json"
    )
    if final_path.exists() or final_path.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise LocalInventoryError("native destination or receipt already exists")
    partial_root = destination_root / ".partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    partial_path = partial_root / f"{content_addressed_filename}.native-hardlink.partial"
    if partial_path.exists() or partial_path.is_symlink():
        raise LocalInventoryError("native registration partial already exists")
    os.link(source_path, partial_path, follow_symlinks=False)
    return transfer.finalize_private_partial(
        repo_root=repo_root,
        destination_root=destination_root,
        partial_path=partial_path,
        content_addressed_filename=content_addressed_filename,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        source_shard_logical_id=source_shard_logical_id,
        validated_30_day_raw_manifest_sha256=validated_30_day_raw_manifest_sha256,
        feature_schema_sha256=feature_schema_sha256,
        target_schema_sha256=target_schema_sha256,
        checklist_provenance_schema_sha256=checklist_provenance_schema_sha256,
        transfer_disposition="copied",
        parallel_lane_id=parallel_lane_id,
        started_at_utc=started_at_utc,
        validated_at_utc=validated_at_utc,
    )


__all__ = ["LocalInventoryError", "register_local_finalized_shard"]
