import json
from pathlib import Path

import pytest

from scripts.import_prestaged_specialist_corpus import (
    _superseded_existing_corpus,
    _validate,
    _validated_existing_receipt,
)


ROOT = Path(__file__).parents[1]


def _digest(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_prestaged_import_is_atomic_checksum_bound_and_non_runtime() -> None:
    source = (ROOT / "scripts/import_prestaged_specialist_corpus.py").read_text(
        encoding="utf-8"
    )
    assert "os.replace(staging, destination)" in source
    assert "pre-staged shard checksum failed" in source
    assert "CURRENT_DECK_GUIDE_CORPUS_READY.json" in source
    assert "active_training_modified" in source
    assert "--bwlimit=" in source
    assert "--minimum-records" in source
    assert "replaced_unavailable_placeholder" in source
    assert "superseded_existing_corpus" in source
    assert "_validated_existing_receipt" in source
    assert "except (RuntimeError, FileNotFoundError):" in source


def test_optional_finalization_receipt_is_checksum_bound(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "pokebot-bootstrap-feature-manifest",
                "totals": {
                    "records_kept": 1458,
                    "decisions_kept": 83980,
                    "target_coverage": {"guide_rows": 1445},
                },
                "shards": [],
            }
        ),
        encoding="utf-8",
    )
    pointer_path = tmp_path / "PROTECTED_EXPERT_CORPUS.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "selection": {"value": "archaludon-ex"},
                "manifest_sha256": _digest(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    ready_path = tmp_path / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    ready_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": "archaludon-ex",
                "guide_version": "guide-v1",
                "manifest_sha256": _digest(manifest_path),
                "protected_pointer_sha256": _digest(pointer_path),
                "records": 1458,
                "decisions": 83980,
                "guide_rows": 1445,
            }
        ),
        encoding="utf-8",
    )
    finalization_path = tmp_path / "ARCHALUDON_EX_GUIDE_CORPUS_READY.json"
    finalization_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.archaludon_ex_guide_corpus_validation/v1",
                "status": "ready_checksum_validated",
                "specialist_id": "archaludon-ex",
                "guide_version": "guide-v1",
                "guide_ready_receipt_sha256": _digest(ready_path),
                "records": 1458,
                "decisions": 83980,
                "guide_rows": 1445,
                "active_training_modified": False,
            }
        ),
        encoding="utf-8",
    )

    identity = _validate(
        tmp_path,
        specialist_id="archaludon-ex",
        guide_version="guide-v1",
        minimum_records=1458,
        finalization_receipt_name=finalization_path.name,
        finalization_receipt_schema=(
            "poke_bot.archaludon_ex_guide_corpus_validation/v1"
        ),
    )

    assert identity["finalization_receipt_name"] == finalization_path.name
    assert identity["finalization_receipt_sha256"] == _digest(
        finalization_path
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["guide_rows"] = 1444
    finalization_path.write_text(json.dumps(finalization), encoding="utf-8")
    with pytest.raises(RuntimeError, match="finalization receipt failed"):
        _validate(
            tmp_path,
            specialist_id="archaludon-ex",
            guide_version="guide-v1",
            minimum_records=1458,
            finalization_receipt_name=finalization_path.name,
            finalization_receipt_schema=(
                "poke_bot.archaludon_ex_guide_corpus_validation/v1"
            ),
        )


def test_superseded_existing_corpus_requires_exact_ready_checksum(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "teal-mask-ogerpon-ex"
    destination.mkdir()
    ready = destination / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    ready.write_text('{"status":"ready"}\n', encoding="utf-8")
    import hashlib

    digest = "sha256:" + hashlib.sha256(ready.read_bytes()).hexdigest()
    assert _superseded_existing_corpus(
        destination, expected_ready_sha256=digest
    ) == {
        "identity_name": "CURRENT_DECK_GUIDE_CORPUS_READY.json",
        "identity_path": str(ready),
        "identity_sha256": digest,
        "ready_path": str(ready),
        "ready_sha256": digest,
    }
    with pytest.raises(RuntimeError, match="not the authorized"):
        _superseded_existing_corpus(
            destination,
            expected_ready_sha256="sha256:" + "0" * 64,
        )


def test_superseded_unguided_corpus_can_bind_exact_protected_pointer(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "archaludon-ex"
    destination.mkdir()
    pointer = destination / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text('{"protected":true}\n', encoding="utf-8")
    import hashlib

    digest = "sha256:" + hashlib.sha256(pointer.read_bytes()).hexdigest()
    assert _superseded_existing_corpus(
        destination,
        expected_pointer_sha256=digest,
    ) == {
        "identity_name": "PROTECTED_EXPERT_CORPUS.json",
        "identity_path": str(pointer),
        "identity_sha256": digest,
        "ready_path": None,
        "ready_sha256": digest,
    }


def test_prestaged_import_receipt_is_idempotent_after_placeholder_replacement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "team-rockets-spidops"
    destination.mkdir()
    archive = tmp_path / ".team-rockets-spidops.unavailable-deadbeef"
    archive.mkdir()
    identity = {
        "specialist_id": "team-rockets-spidops",
        "guide_version": "guide-v1",
        "records": 16650,
        "decisions": 1186370,
        "guide_rows": 113013,
        "manifest_sha256": "sha256:" + "1" * 64,
        "protected_pointer_sha256": "sha256:" + "2" * 64,
        "ready_receipt_sha256": "sha256:" + "3" * 64,
    }
    receipt = {
        "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
        "status": "ready",
        "created_at_utc": "2026-07-28T19:20:30+00:00",
        "source_host": "admin@example",
        "source_root": "/remote/corpus",
        "destination": str(destination),
        **identity,
        "replaced_unavailable_placeholder": True,
        "unavailable_placeholder_archive": str(archive),
        "unavailable_placeholder_sha256": "sha256:" + "4" * 64,
        "active_training_modified": False,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    observed = _validated_existing_receipt(
        path,
        source_host="admin@example",
        source_root="/remote/corpus",
        destination=destination,
        identity=identity,
    )

    assert observed == receipt


def test_prestaged_import_receipt_rejects_changed_corpus_identity(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "team-rockets-spidops"
    destination.mkdir()
    identity = {
        "specialist_id": "team-rockets-spidops",
        "guide_version": "guide-v1",
        "records": 16650,
    }
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
                "status": "ready",
                "created_at_utc": "2026-07-28T19:20:30+00:00",
                "source_host": "admin@example",
                "source_root": "/remote/corpus",
                "destination": str(destination),
                **identity,
                "records": 16639,
                "replaced_unavailable_placeholder": False,
                "unavailable_placeholder_archive": None,
                "unavailable_placeholder_sha256": None,
                "active_training_modified": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="identity differs"):
        _validated_existing_receipt(
            path,
            source_host="admin@example",
            source_root="/remote/corpus",
            destination=destination,
            identity=identity,
        )
