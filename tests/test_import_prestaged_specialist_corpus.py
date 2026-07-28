import json
from pathlib import Path

import pytest

from scripts.import_prestaged_specialist_corpus import (
    _validated_existing_receipt,
)


ROOT = Path(__file__).parents[1]


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
    assert "_validated_existing_receipt" in source


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
