from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import poke_bot.alakazam_rule_derivative_transfer_rev9 as transfer


def _digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def test_finalize_private_partial_is_create_only_and_receipted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / transfer.GOAL_CONTRACT_PATH
    contract.parent.mkdir(parents=True)
    body = json.dumps({"goal_revision": 9}).encode()
    contract.write_bytes(body)
    monkeypatch.setattr(transfer, "GOAL_CONTRACT_SHA256", _digest(body))
    destination = tmp_path / "quarantine"
    monkeypatch.setattr(transfer, "DESTINATION_ROOT", destination.resolve())
    partial_root = destination / ".partial"
    partial_root.mkdir(parents=True)
    payload = b'{"row":1}\n'
    digest = _digest(payload)
    filename = f"sha256-{digest.removeprefix('sha256:')}.jsonl"
    partial = partial_root / f"{filename}.partial"
    partial.write_bytes(payload)

    receipt, final_path, receipt_path, receipt_sha = transfer.finalize_private_partial(
        repo_root=tmp_path,
        destination_root=destination,
        partial_path=partial,
        content_addressed_filename=filename,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
        source_shard_logical_id="2026-07-24/0",
        validated_30_day_raw_manifest_sha256="sha256:" + "1" * 64,
        feature_schema_sha256="sha256:" + "2" * 64,
        target_schema_sha256="sha256:" + "3" * 64,
        checklist_provenance_schema_sha256="sha256:" + "4" * 64,
        transfer_disposition="resumed",
        parallel_lane_id="four_range_direct_lan",
        started_at_utc="2026-08-13T06:43:00Z",
        validated_at_utc="2026-08-13T07:10:00Z",
    )

    assert not partial.exists()
    assert final_path.read_bytes() == payload
    assert receipt["destination_validation_passed"] is True
    assert receipt["inzi_loading_or_execution_authority"] is False
    assert transfer.sha256_file(receipt_path) == receipt_sha


def test_mismatched_partial_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / transfer.GOAL_CONTRACT_PATH
    contract.parent.mkdir(parents=True)
    body = json.dumps({"goal_revision": 9}).encode()
    contract.write_bytes(body)
    monkeypatch.setattr(transfer, "GOAL_CONTRACT_SHA256", _digest(body))
    destination = tmp_path / "quarantine"
    monkeypatch.setattr(transfer, "DESTINATION_ROOT", destination.resolve())
    partial_root = destination / ".partial"
    partial_root.mkdir(parents=True)
    expected = b"expected\n"
    digest = _digest(expected)
    filename = f"sha256-{digest.removeprefix('sha256:')}.jsonl"
    partial = partial_root / f"{filename}.partial"
    partial.write_bytes(b"wrong\n")

    with pytest.raises(transfer.Rev9TransferError, match="SHA/size"):
        transfer.finalize_private_partial(
            repo_root=tmp_path,
            destination_root=destination,
            partial_path=partial,
            content_addressed_filename=filename,
            expected_sha256=digest,
            expected_size_bytes=len(expected),
            source_shard_logical_id="2026-07-24/0",
            validated_30_day_raw_manifest_sha256="sha256:" + "1" * 64,
            feature_schema_sha256="sha256:" + "2" * 64,
            target_schema_sha256="sha256:" + "3" * 64,
            checklist_provenance_schema_sha256="sha256:" + "4" * 64,
            transfer_disposition="copied",
            parallel_lane_id="lane0",
            started_at_utc="2026-08-13T06:43:00Z",
            validated_at_utc="2026-08-13T07:10:00Z",
        )
    assert not (destination / filename).exists()
