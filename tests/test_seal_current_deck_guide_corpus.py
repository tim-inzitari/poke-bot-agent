from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_generic_guide_sealer_is_checksum_bound_and_roster_agnostic() -> None:
    source = (ROOT / "scripts/seal_current_deck_guide_corpus.py").read_text(
        encoding="utf-8"
    )
    assert "poke_bot.current_deck_guide_corpus_ready/v1" in source
    assert "guide_rows <= 0" in source
    assert "pointer.get(\"manifest_sha256\")" in source
    assert "active_training_modified" in source
    assert "immutable guide receipt differs" in source
    assert 'existing_receipt.get("created_at_utc")' in source
