from __future__ import annotations

import hashlib
import json
from pathlib import Path

import poke_bot.alakazam_rule_derivative_transfer_rev9 as transfer
from poke_bot.alakazam_rule_derivative_local_inventory_rev9 import (
    register_local_finalized_shard,
)


def _digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def test_native_shard_is_registered_by_hardlink_and_receipted(tmp_path: Path, monkeypatch) -> None:
    contract = tmp_path / transfer.GOAL_CONTRACT_PATH
    contract.parent.mkdir(parents=True)
    contract_body = json.dumps({"goal_revision": 9}).encode()
    contract.write_bytes(contract_body)
    monkeypatch.setattr(transfer, "GOAL_CONTRACT_SHA256", _digest(contract_body))
    destination = tmp_path / "quarantine"
    destination.mkdir()
    monkeypatch.setattr(transfer, "DESTINATION_ROOT", destination.resolve())
    source = tmp_path / "native.jsonl"
    body = b'{"header":1}\n{"row":1}\n'
    source.write_bytes(body)
    digest = _digest(body)
    filename = f"sha256-{digest.removeprefix('sha256:')}.jsonl"
    receipt, final, receipt_path, _ = register_local_finalized_shard(
        repo_root=tmp_path,
        source_path=source,
        destination_root=destination,
        content_addressed_filename=filename,
        expected_sha256=digest,
        expected_size_bytes=len(body),
        source_shard_logical_id="2026-07-23/0",
        validated_30_day_raw_manifest_sha256="sha256:" + "1" * 64,
        feature_schema_sha256="sha256:" + "2" * 64,
        target_schema_sha256="sha256:" + "3" * 64,
        checklist_provenance_schema_sha256="sha256:" + "4" * 64,
        parallel_lane_id="inzi_native_0",
        started_at_utc="2026-08-13T00:00:00Z",
        validated_at_utc="2026-08-13T00:00:01Z",
    )
    assert source.stat().st_ino == final.stat().st_ino
    assert final.read_bytes() == body
    assert receipt_path.is_file()
    assert receipt["destination_validation_passed"] is True
    assert receipt["inzi_loading_or_execution_authority"] is False
