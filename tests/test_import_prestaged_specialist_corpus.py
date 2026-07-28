from pathlib import Path


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
    assert 'existing_receipt.get("created_at_utc")' in source
