from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from poke_bot import alakazam_rule_derivative_transfer_rev8 as transfer


def _row(payload: bytes) -> dict[str, object]:
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {
        "filename": f"sha256-{digest[7:]}.refeaturization-census.shard",
        "sha256": digest,
        "size_bytes": len(payload),
        "record_count": 1,
    }


def _run(host_index: int, row: dict[str, object]) -> dict[str, object]:
    return {"host_index": host_index, "shards": [row]}


def test_plan_transfers_only_elmo_final_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer, "load_revision_8_contract", lambda **_: {})
    local = _row(b"local")
    remote = _row(b"remote")
    plan = transfer.build_transfer_plan(
        inzi_half=_run(0, local),
        elmo_half=_run(1, remote),
        raw_manifest_sha256="sha256:" + "1" * 64,
        migration_receipt_sha256="sha256:" + "2" * 64,
    )
    assert plan["inzi_local_shard_count"] == 1
    assert plan["elmo_transfer_shard_count"] == 1
    assert plan["entries"][0]["sha256"] == remote["sha256"]
    assert plan["private_partials_transferable"] is False


def test_plan_rejects_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer, "load_revision_8_contract", lambda **_: {})
    local = _row(b"local")
    remote = _row(b"remote")
    remote["filename"] = ".bad.partial"
    with pytest.raises(transfer.Rev8TransferError):
        transfer.build_transfer_plan(
            inzi_half=_run(0, local),
            elmo_half=_run(1, remote),
            raw_manifest_sha256="sha256:" + "1" * 64,
            migration_receipt_sha256="sha256:" + "2" * 64,
        )


def test_remote_receipt_and_combined_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfer, "load_revision_8_contract", lambda **_: {})
    local = _row(b"local")
    remote_payload = b"remote"
    remote = _row(remote_payload)
    plan = transfer.build_transfer_plan(
        inzi_half=_run(0, local),
        elmo_half=_run(1, remote),
        raw_manifest_sha256="sha256:" + "1" * 64,
        migration_receipt_sha256="sha256:" + "2" * 64,
    )
    final = tmp_path / str(remote["filename"])
    final.write_bytes(remote_payload)
    receipt = transfer.build_shard_transfer_receipt(
        plan=plan, entry=plan["entries"][0], remote_path=final
    )
    parity = transfer.build_parity_receipt(
        plan=plan,
        inzi_half=_run(0, local),
        elmo_half=_run(1, remote),
        transfer_receipts=[receipt],
        sealed_at_utc="2026-08-13T00:00:00Z",
    )
    assert parity["combined_shard_count"] == 2
    assert parity["elmo_shards_transferred_create_only"] == 1
    assert parity["private_partials_transferred"] is False


def test_remote_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer, "load_revision_8_contract", lambda **_: {})
    local = _row(b"local")
    remote = _row(b"remote")
    plan = transfer.build_transfer_plan(
        inzi_half=_run(0, local),
        elmo_half=_run(1, remote),
        raw_manifest_sha256="sha256:" + "1" * 64,
        migration_receipt_sha256="sha256:" + "2" * 64,
    )
    final = tmp_path / str(remote["filename"])
    final.write_bytes(b"wrong")
    with pytest.raises(transfer.Rev8TransferError):
        transfer.build_shard_transfer_receipt(
            plan=plan, entry=plan["entries"][0], remote_path=final
        )
