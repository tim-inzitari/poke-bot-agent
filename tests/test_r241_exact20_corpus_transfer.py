"""Focused coverage for the inert, checksum-first r241 corpus handoff."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys

import pytest

from poke_bot import r241_checkpoint_receipts as checkpoint_receipts
from scripts import transfer_r241_exact20_alakazam_corpus as transfer


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, transfer.Exact20Contract]:
    """Build a compact but semantically complete 20-day source tree."""

    source = tmp_path / "elmo-source"
    source.mkdir()
    archive = tmp_path / "archive-current.json"
    start = date.fromisoformat(transfer.WINDOW_START)
    dates = tuple((start + timedelta(days=index)).isoformat() for index in range(20))
    archives = [
        {
            "date": day,
            "validated": True,
            "sha256": "sha256:" + hashlib.sha256(day.encode("utf-8")).hexdigest(),
        }
        for day in dates
    ]
    _write_json(
        archive,
        {
            "schema": transfer.ARCHIVE_SCHEMA,
            "status": "ready",
            "window_policy": "exact_20_consecutive_calendar_days",
            "window_start": transfer.WINDOW_START,
            "window_end": transfer.WINDOW_END,
            "days": 20,
            "all_dates_represented": True,
            "total_episodes": transfer.WINDOW_TOTAL_EPISODES,
            "archives": archives,
        },
    )
    archive_sha = _sha256(archive)
    archive_by_date = {row["date"]: row["sha256"] for row in archives}

    rows: list[dict[str, object]] = []
    total_bytes = 0
    for index, day in enumerate(dates):
        relative = f"all-recognized-{day}.alakazam.features"
        feature = source / relative
        header = {
            "format": "pokebot-bootstrap-feature-shard",
            "source_dates": [day],
            "source_archive_sha256": archive_by_date[day],
            "required_archetype": "alakazam",
            "compact_mode": "temporal-expert-v1",
            "dataset_schema": 6,
            "feature_schema": 5,
            "max_context": 320,
        }
        with feature.open("wb") as stream:
            pickle.dump(header, stream)
            stream.write(bytes([index]) * (index + 1))
        _write_json(
            source / f"{relative}.json",
            {
                "source_dates": [day],
                "source_archive_sha256": archive_by_date[day],
                "dataset_schema": 6,
                "feature_schema": 5,
            },
        )
        total_bytes += feature.stat().st_size
        rows.append(
            {
                "path": relative,
                "sha256": _sha256(feature),
                "bytes": feature.stat().st_size,
                "source_dates": [day],
                "source_archive_sha256": archive_by_date[day],
                "required_archetype": "alakazam",
                "selection_archetype": "alakazam",
                "compact_mode": "temporal-expert-v1",
                "dataset_schema": 6,
                "feature_schema": 5,
                "max_context": 320,
                "stats": {"records_kept": 1, "decisions_kept": 1},
            }
        )

    selection = {
        "field": "GameSequence.archetype",
        "operator": "exact_casefold",
        "opponent_routes_only": False,
        "seat_semantics": "acting_seat_only",
        "value": "alakazam",
    }
    manifest = source / transfer.MANIFEST_NAME
    _write_json(
        manifest,
        {
            "format": "pokebot-bootstrap-feature-manifest",
            "format_version": 1,
            "compact_mode": "temporal-expert-v1",
            "date_start": transfer.WINDOW_START,
            "date_end": transfer.WINDOW_END,
            "dates": list(dates),
            "selection": selection,
            "shards": rows,
            "totals": {
                "bytes": total_bytes,
                "records_kept": len(rows),
                "decisions_kept": len(rows),
            },
        },
    )
    pointer = source / transfer.SOURCE_POINTER_NAME
    _write_json(
        pointer,
        {
            "schema": transfer.POINTER_SCHEMA,
            "protected": True,
            "manifest": transfer.MANIFEST_NAME,
            "manifest_sha256": _sha256(manifest),
            "selection": selection,
            "totals": {
                "bytes": total_bytes,
                "records_kept": len(rows),
                "decisions_kept": len(rows),
            },
        },
    )
    ready = source / transfer.SOURCE_READY_COPY_NAME
    _write_json(
        ready,
        {
            "schema": transfer.FINAL_READY_SCHEMA,
            "status": "ready",
            "archive_receipt_sha256": archive_sha,
            "dates": list(dates),
            "results": [
                {
                    "archetype": "alakazam",
                    "status": "ready",
                    "protected_corpus": "alakazam/PROTECTED_EXPERT_CORPUS.json",
                    "manifest_sha256": _sha256(manifest),
                    "records": len(rows),
                    "decisions": len(rows),
                }
            ],
        },
    )
    contract = transfer.Exact20Contract(
        ready_sha256=_sha256(ready),
        ready_size_bytes=ready.stat().st_size,
        source_pointer_sha256=_sha256(pointer),
        source_pointer_size_bytes=pointer.stat().st_size,
        manifest_sha256=_sha256(manifest),
        manifest_size_bytes=manifest.stat().st_size,
        archive_sha256=archive_sha,
        archive_size_bytes=archive.stat().st_size,
        shard_bytes=total_bytes,
        records=len(rows),
        decisions=len(rows),
        dates=dates,
    )
    return source, archive, contract


def test_local_finalizer_preserves_source_pointer_and_adds_exact_archive_binding(
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    source_before = _tree_digest(source)
    destination = tmp_path / "inzi-runtime" / "expert"
    archive_destination = tmp_path / "inzi-state" / "expert-current.json"

    identity = transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=destination,
        archive_destination=archive_destination,
        contract=contract,
        source_host="test-elmo",
        source_root_label="/elmo/r241/window",
        source_archive_label="/elmo/r241/archive-current.json",
    )

    assert identity.manifest.sha256 == contract.manifest_sha256
    assert _tree_digest(source) == source_before
    assert _sha256(destination / transfer.SOURCE_POINTER_NAME) == contract.source_pointer_sha256
    assert _sha256(destination / transfer.MANIFEST_NAME) == contract.manifest_sha256
    assert _sha256(archive_destination) == contract.archive_sha256
    pointer = json.loads((destination / transfer.TOP_LEVEL_POINTER_NAME).read_text(encoding="utf-8"))
    assert pointer["r241_source_finalization"]["source_pointer"]["sha256"] == contract.source_pointer_sha256
    assert pointer["r241_archive_binding"] == {
        "archive_receipt_path": str(archive_destination.resolve()),
        "archive_receipt_sha256": contract.archive_sha256,
        "archive_receipt_size_bytes": contract.archive_size_bytes,
        "copied_archive_receipt": {
            "path": transfer.ARCHIVE_COPY_NAME,
            "sha256": contract.archive_sha256,
            "size_bytes": contract.archive_size_bytes,
        },
        "source_host": "test-elmo",
        "source_archive_receipt_path": "/elmo/r241/archive-current.json",
        "source_archive_receipt_sha256": contract.archive_sha256,
        "source_archive_receipt_size_bytes": contract.archive_size_bytes,
    }
    assert len(pointer["r241_exact20_transfer"]["receipt"]["sha256"]) == 71
    assert transfer._validate_finalized_destination(
        destination, archive_destination=archive_destination, contract=contract
    ).shard_bytes == contract.shard_bytes


def test_local_finalizer_is_idempotent_only_for_the_same_complete_identity(
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    destination = tmp_path / "runtime" / "expert"
    archive_destination = tmp_path / "state" / "archive.json"
    first = transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=destination,
        archive_destination=archive_destination,
        contract=contract,
    )
    pointer_before = (destination / transfer.TOP_LEVEL_POINTER_NAME).read_bytes()
    receipt_before = (destination / transfer.TRANSFER_RECEIPT_NAME).read_bytes()
    second = transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=destination,
        archive_destination=archive_destination,
        contract=contract,
    )
    assert second == first
    assert (destination / transfer.TOP_LEVEL_POINTER_NAME).read_bytes() == pointer_before
    assert (destination / transfer.TRANSFER_RECEIPT_NAME).read_bytes() == receipt_before


def test_local_finalizer_resumes_the_durable_nonruntime_partial_root(
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    destination = tmp_path / "runtime" / "expert"
    archive_destination = tmp_path / "state" / "archive.json"
    partial = transfer._make_stage(destination.parent, destination_name=destination.name)
    # Model an interrupted metadata copy.  This root has no runtime pointer,
    # and the next explicit finalizer invocation must reuse rather than delete
    # it before downloading/copying the remaining shards.
    (partial / transfer.SOURCE_READY_COPY_NAME).write_bytes(
        (source / transfer.SOURCE_READY_COPY_NAME).read_bytes()
    )

    identity = transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=destination,
        archive_destination=archive_destination,
        contract=contract,
    )

    assert identity.manifest.path == destination / transfer.MANIFEST_NAME
    assert destination.is_dir()
    assert not partial.exists()


def test_checkpoint_validator_rehashes_exact_archive_binding_and_source_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    destination = tmp_path / "runtime" / "expert"
    archive_destination = tmp_path / "state" / "archive.json"
    transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=destination,
        archive_destination=archive_destination,
        contract=contract,
    )
    for name, value in {
        "R241_EXACT20_SOURCE_READY_SHA256": contract.ready_sha256,
        "R241_EXACT20_SOURCE_READY_SIZE_BYTES": contract.ready_size_bytes,
        "R241_EXACT20_SOURCE_POINTER_SHA256": contract.source_pointer_sha256,
        "R241_EXACT20_SOURCE_POINTER_SIZE_BYTES": contract.source_pointer_size_bytes,
        "R241_EXACT20_MANIFEST_SHA256": contract.manifest_sha256,
        "R241_EXACT20_MANIFEST_SIZE_BYTES": contract.manifest_size_bytes,
        "R241_EXACT20_ARCHIVE_SHA256": contract.archive_sha256,
        "R241_EXACT20_ARCHIVE_SIZE_BYTES": contract.archive_size_bytes,
        "R241_EXACT20_SHARD_BYTES": contract.shard_bytes,
        "R241_EXACT20_RECORDS": contract.records,
        "R241_EXACT20_DECISIONS": contract.decisions,
    }.items():
        monkeypatch.setattr(checkpoint_receipts, name, value)

    pointer_path = destination / transfer.TOP_LEVEL_POINTER_NAME
    observed = checkpoint_receipts.validate_r241_protected_expert_pointer(
        pointer_path,
        archive_receipt_path=archive_destination,
    )
    assert observed.path == pointer_path

    # The archive itself still hashes correctly; only the declared binding is
    # forged.  A prefix-only check would accept this, while the shared
    # launcher/checkpoint validator must reject it.
    destination.chmod(0o755)
    pointer_path.chmod(0o644)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["r241_archive_binding"]["archive_receipt_sha256"] = "sha256:" + "0" * 64
    _write_json(pointer_path, pointer)
    with pytest.raises(
        checkpoint_receipts.R241CheckpointReceiptError,
        match="actual archive bytes",
    ):
        checkpoint_receipts.validate_r241_protected_expert_pointer(
            pointer_path,
            archive_receipt_path=archive_destination,
        )


def test_elmo_metadata_handoff_is_small_create_only_and_uses_the_shared_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    source_before = _tree_digest(source)
    inzi_destination = tmp_path / "inzi-runtime" / "expert"
    inzi_archive = tmp_path / "inzi-state" / "archive.json"
    transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=inzi_destination,
        archive_destination=inzi_archive,
        contract=contract,
    )
    inzi_body = (inzi_destination / transfer.TRANSFER_RECEIPT_NAME).read_bytes()
    inzi_sha = "sha256:" + hashlib.sha256(inzi_body).hexdigest()
    elmo_destination = tmp_path / "elmo-runtime" / "expert"

    handoff = transfer.finalize_elmo_metadata_handoff_local_copy(
        source_ready=source / transfer.SOURCE_READY_COPY_NAME,
        source_pointer=source / transfer.SOURCE_POINTER_NAME,
        source_manifest=source / transfer.MANIFEST_NAME,
        source_archive_receipt=archive,
        inzi_transfer_receipt_body=inzi_body,
        destination=elmo_destination,
        contract=contract,
        expected_inzi_receipt_sha256=inzi_sha,
        expected_inzi_receipt_size_bytes=len(inzi_body),
        expected_inzi_archive_destination=inzi_archive,
        inzi_receipt_path="/inzi/sealed/R241_EXACT20_CORPUS_TRANSFER_READY.json",
    )

    assert handoff.root == elmo_destination
    assert _tree_digest(source) == source_before
    assert {path.name for path in elmo_destination.iterdir()} == {
        transfer.SOURCE_READY_COPY_NAME,
        transfer.SOURCE_POINTER_NAME,
        transfer.MANIFEST_NAME,
        transfer.ARCHIVE_COPY_NAME,
        transfer.INZI_TRANSFER_RECEIPT_COPY_NAME,
        transfer.TRANSFER_RECEIPT_NAME,
        transfer.TOP_LEVEL_POINTER_NAME,
    }
    assert not list(elmo_destination.glob("*.features"))
    assert not list(elmo_destination.glob("*.features.json"))
    assert _sha256(elmo_destination / transfer.SOURCE_POINTER_NAME) == contract.source_pointer_sha256
    assert (elmo_destination / transfer.INZI_TRANSFER_RECEIPT_COPY_NAME).read_bytes() == inzi_body
    pointer = json.loads((elmo_destination / transfer.TOP_LEVEL_POINTER_NAME).read_text(encoding="utf-8"))
    assert pointer["r241_archive_binding"]["archive_receipt_path"] == str(archive.resolve())
    local_receipt = json.loads(
        (elmo_destination / transfer.TRANSFER_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert local_receipt["r241_elmo_metadata_handoff"] == {
        "schema": transfer.ELMO_METADATA_HANDOFF_SCHEMA,
        "metadata_only": True,
        "feature_shards_copied": False,
        "feature_sidecars_copied": False,
        "source_archive_reused_without_mutation": True,
        "inzi_transfer_receipt": {
            "host": transfer.INZI_HOST,
            "remote_path": "/inzi/sealed/R241_EXACT20_CORPUS_TRANSFER_READY.json",
            "local_copy": {
                "path": transfer.INZI_TRANSFER_RECEIPT_COPY_NAME,
                "sha256": inzi_sha,
                "size_bytes": len(inzi_body),
            },
        },
    }
    for name, value in {
        "R241_EXACT20_SOURCE_READY_SHA256": contract.ready_sha256,
        "R241_EXACT20_SOURCE_READY_SIZE_BYTES": contract.ready_size_bytes,
        "R241_EXACT20_SOURCE_POINTER_SHA256": contract.source_pointer_sha256,
        "R241_EXACT20_SOURCE_POINTER_SIZE_BYTES": contract.source_pointer_size_bytes,
        "R241_EXACT20_MANIFEST_SHA256": contract.manifest_sha256,
        "R241_EXACT20_MANIFEST_SIZE_BYTES": contract.manifest_size_bytes,
        "R241_EXACT20_ARCHIVE_SHA256": contract.archive_sha256,
        "R241_EXACT20_ARCHIVE_SIZE_BYTES": contract.archive_size_bytes,
        "R241_EXACT20_SHARD_BYTES": contract.shard_bytes,
        "R241_EXACT20_RECORDS": contract.records,
        "R241_EXACT20_DECISIONS": contract.decisions,
        "R241_EXACT20_INZI_TRANSFER_RECEIPT_SHA256": inzi_sha,
        "R241_EXACT20_INZI_TRANSFER_RECEIPT_SIZE_BYTES": len(inzi_body),
        "R241_EXACT20_INZI_TRANSFER_RECEIPT_PATH": "/inzi/sealed/R241_EXACT20_CORPUS_TRANSFER_READY.json",
    }.items():
        monkeypatch.setattr(checkpoint_receipts, name, value)
    observed = checkpoint_receipts.validate_r241_protected_expert_pointer(
        elmo_destination / transfer.TOP_LEVEL_POINTER_NAME,
        archive_receipt_path=archive,
    )
    assert observed.path == elmo_destination / transfer.TOP_LEVEL_POINTER_NAME

    # A receipt-only projection must remain exactly that: neither a shard nor
    # an unrelated payload can be slipped under an otherwise valid pointer.
    elmo_destination.chmod(0o755)
    (elmo_destination / "unexpected.features").write_bytes(b"not a corpus shard")
    with pytest.raises(
        checkpoint_receipts.R241CheckpointReceiptError,
        match="exactly seven receipt files",
    ):
        checkpoint_receipts.validate_r241_protected_expert_pointer(
            elmo_destination / transfer.TOP_LEVEL_POINTER_NAME,
            archive_receipt_path=archive,
        )


def test_elmo_metadata_handoff_is_idempotent_and_does_not_project_shards(
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    inzi_destination = tmp_path / "inzi-runtime" / "expert"
    inzi_archive = tmp_path / "inzi-state" / "archive.json"
    transfer.finalize_local_copy(
        source_root=source,
        source_archive_receipt=archive,
        destination=inzi_destination,
        archive_destination=inzi_archive,
        contract=contract,
    )
    inzi_body = (inzi_destination / transfer.TRANSFER_RECEIPT_NAME).read_bytes()
    inzi_sha = "sha256:" + hashlib.sha256(inzi_body).hexdigest()
    destination = tmp_path / "elmo-runtime" / "expert"
    kwargs = {
        "source_ready": source / transfer.SOURCE_READY_COPY_NAME,
        "source_pointer": source / transfer.SOURCE_POINTER_NAME,
        "source_manifest": source / transfer.MANIFEST_NAME,
        "source_archive_receipt": archive,
        "inzi_transfer_receipt_body": inzi_body,
        "destination": destination,
        "contract": contract,
        "expected_inzi_receipt_sha256": inzi_sha,
        "expected_inzi_receipt_size_bytes": len(inzi_body),
        "expected_inzi_archive_destination": inzi_archive,
        "inzi_receipt_path": "/inzi/sealed/R241_EXACT20_CORPUS_TRANSFER_READY.json",
    }
    first = transfer.finalize_elmo_metadata_handoff_local_copy(**kwargs)
    before = _tree_digest(destination)
    second = transfer.finalize_elmo_metadata_handoff_local_copy(**kwargs)
    assert second == first
    assert _tree_digest(destination) == before
    assert not (destination.parent / ".expert.r241-elmo-metadata-handoff.partial").exists()


def test_source_plan_is_hash_first_and_has_no_rsync_or_destination_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    source_map: dict[str, Path] = {
        f"{transfer.SOURCE_WINDOW_ROOT}/{transfer.SOURCE_READY_NAME}": (
            source / transfer.SOURCE_READY_COPY_NAME
        ),
        (
            f"{transfer.SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/"
            "PROTECTED_EXPERT_CORPUS.json"
        ): source / transfer.SOURCE_POINTER_NAME,
        (
            f"{transfer.SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/"
            f"{transfer.MANIFEST_NAME}"
        ): source / transfer.MANIFEST_NAME,
        transfer.SOURCE_ARCHIVE_RECEIPT: archive,
    }
    manifest = json.loads((source / transfer.MANIFEST_NAME).read_text(encoding="utf-8"))
    for row in manifest["shards"]:
        relative = row["path"]
        remote = (
            f"{transfer.SOURCE_WINDOW_ROOT}/specialist-corpora/alakazam/{relative}"
        )
        source_map[remote] = source / relative
        source_map[f"{remote}.json"] = source / f"{relative}.json"

    def remote_identity(_host: str, remote: str, *, label: str) -> transfer.FileIdentity:
        local = source_map[remote]
        return transfer.FileIdentity(local, _sha256(local), local.stat().st_size)

    def remote_json(
        _host: str, remote: str, *, label: str
    ) -> tuple[transfer.FileIdentity, dict[str, object], bytes]:
        local = source_map[remote]
        body = local.read_bytes()
        return remote_identity(_host, remote, label=label), json.loads(body), body

    def remote_bytes(
        _host: str,
        remote: str,
        *,
        label: str,
        limit_bytes: int | None = None,
    ) -> bytes:
        body = source_map[remote].read_bytes()
        return body if limit_bytes is None else body[:limit_bytes]

    monkeypatch.setattr(transfer, "_remote_file_identity", remote_identity)
    monkeypatch.setattr(transfer, "_remote_json", remote_json)
    monkeypatch.setattr(transfer, "_remote_readonly_bytes", remote_bytes)

    plan = transfer.plan_from_elmo(contract=contract)

    assert plan["status"] == "source_validated_plan_only"
    assert plan["source_read_only"] is True
    assert plan["rsync_invoked"] is False
    assert plan["transfer_executed"] is False
    assert plan["planned_member_count"] == 44
    assert plan["corpus"] == {
        "records": contract.records,
        "decisions": contract.decisions,
        "shard_bytes": contract.shard_bytes,
        "feature_shard_count": 20,
        "sidecar_count": 20,
    }
    assert not transfer.DESTINATION_ROOT.exists()
    assert not transfer.DESTINATION_ARCHIVE_RECEIPT.exists()


def test_execute_streams_to_inzi_without_local_destination_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = {
        "schema": transfer.TRANSFER_SCHEMA,
        "status": "ready",
        "execution_host": transfer.INZI_HOST,
        "destination": str(transfer.DESTINATION_ROOT),
        "archive_destination": str(transfer.DESTINATION_ARCHIVE_RECEIPT),
        "manifest_sha256": transfer.DEFAULT_CONTRACT.manifest_sha256,
        "source_pointer_sha256": transfer.DEFAULT_CONTRACT.source_pointer_sha256,
        "archive_receipt_sha256": transfer.DEFAULT_CONTRACT.archive_sha256,
        "records": transfer.DEFAULT_CONTRACT.records,
        "decisions": transfer.DEFAULT_CONTRACT.decisions,
        "shard_bytes": transfer.DEFAULT_CONTRACT.shard_bytes,
        "source_mutated": False,
        "service_action": "none",
    }
    observed: dict[str, object] = {}

    def forbidden_destination(path: Path) -> bool:
        return str(path).startswith("/home/pokebot/")

    for method_name in ("mkdir", "open", "stat"):
        original = getattr(Path, method_name)

        def guarded(self: Path, *args: object, _original=original, **kwargs: object) -> object:
            if forbidden_destination(self):
                raise AssertionError(f"controller touched Inzi destination locally: {self}")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(completion).encode("utf-8") + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)

    assert transfer.transfer_from_elmo() == completion
    command = list(observed["command"])
    assert command[:7] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        transfer.INZI_HOST,
        "env",
    ]
    assert "R241_EXACT20_INZI_FINALIZER=1" in command
    assert transfer.REMOTE_INZI_FINALIZE_FLAG in command
    assert isinstance(observed["input"], bytes)


def test_elmo_metadata_execute_streams_without_local_elmo_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = {
        "schema": transfer.ELMO_METADATA_HANDOFF_SCHEMA,
        "status": "ready",
        "execution_host": transfer.SOURCE_HOST,
        "destination": str(transfer.ELMO_METADATA_HANDOFF_ROOT),
        "archive_destination": transfer.SOURCE_ARCHIVE_RECEIPT,
        "inzi_transfer_receipt_path": transfer.INZI_TRANSFER_RECEIPT_PATH,
        "pointer_sha256": "sha256:" + "1" * 64,
        "transfer_receipt_sha256": "sha256:" + "2" * 64,
        "inzi_transfer_receipt_sha256": transfer.INZI_TRANSFER_RECEIPT_SHA256,
        "archive_receipt_sha256": transfer.DEFAULT_CONTRACT.archive_sha256,
        "source_ready_sha256": transfer.DEFAULT_CONTRACT.ready_sha256,
        "source_pointer_sha256": transfer.DEFAULT_CONTRACT.source_pointer_sha256,
        "manifest_sha256": transfer.DEFAULT_CONTRACT.manifest_sha256,
        "metadata_only": True,
        "feature_shards_copied": False,
        "source_mutated": False,
        "service_action": "none",
    }
    observed: dict[str, object] = {}

    for method_name in ("mkdir", "open", "stat"):
        original = getattr(Path, method_name)

        def guarded(self: Path, *args: object, _original=original, **kwargs: object) -> object:
            if str(self).startswith("/srv/poke-bot-agent/"):
                raise AssertionError(f"controller touched Elmo destination locally: {self}")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded)

    monkeypatch.setattr(
        transfer,
        "_read_sealed_inzi_transfer_receipt",
        lambda: b"sealed-inzi-receipt-bytes",
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(completion).encode("utf-8") + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)

    assert transfer._run_elmo_metadata_handoff(plan_only=False) == completion
    command = list(observed["command"])
    assert command[:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        transfer.SOURCE_HOST,
    ]
    assert len(command) == 7
    remote_command = str(command[6])
    assert remote_command.startswith("sudo -n python3 -c ")
    assert shlex.quote(transfer._ELMO_METADATA_REMOTE_BOOTSTRAP) in remote_command
    assert f"{transfer.ELMO_METADATA_INZI_RECEIPT_ENV}=" not in remote_command
    assert transfer.REMOTE_ELMO_METADATA_HANDOFF_FLAG in remote_command
    stream = json.loads(bytes(observed["input"]).decode("utf-8"))
    assert stream["schema"] == transfer.ELMO_METADATA_STREAM_SCHEMA
    assert stream["inzi_transfer_receipt_b64"] == "c2VhbGVkLWluemktcmVjZWlwdC1ieXRlcw=="
    assert stream["inzi_transfer_receipt_sha256"] == (
        "sha256:" + hashlib.sha256(b"sealed-inzi-receipt-bytes").hexdigest()
    )


def test_local_finalizer_fails_before_publish_when_a_shard_changes(
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    manifest = json.loads((source / transfer.MANIFEST_NAME).read_text(encoding="utf-8"))
    shard = source / manifest["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    destination = tmp_path / "runtime" / "expert"
    archive_destination = tmp_path / "state" / "archive.json"

    with pytest.raises(transfer.R241Exact20TransferError, match="checksum mismatch"):
        transfer.finalize_local_copy(
            source_root=source,
            source_archive_receipt=archive,
            destination=destination,
            archive_destination=archive_destination,
            contract=contract,
        )
    assert not destination.exists()
    assert not archive_destination.exists()


def test_existing_destination_or_archive_with_another_identity_fails_closed(
    tmp_path: Path,
) -> None:
    source, archive, contract = _fixture(tmp_path)
    destination = tmp_path / "runtime" / "expert"
    destination.mkdir(parents=True)
    (destination / transfer.TOP_LEVEL_POINTER_NAME).write_text("{}\n", encoding="utf-8")
    archive_destination = tmp_path / "state" / "archive.json"

    with pytest.raises(transfer.R241Exact20TransferError):
        transfer.finalize_local_copy(
            source_root=source,
            source_archive_receipt=archive,
            destination=destination,
            archive_destination=archive_destination,
            contract=contract,
        )
    assert _sha256(source / transfer.SOURCE_POINTER_NAME) == contract.source_pointer_sha256


def test_cli_is_plan_only_without_execute() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(transfer.__file__))],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["schema"] == transfer.TRANSFER_SCHEMA
    assert payload["status"] == "inert_plan_only"
    assert payload["execute_required"] is True
    assert payload["source_mutated"] is False
    assert payload["service_action"] == "none"
