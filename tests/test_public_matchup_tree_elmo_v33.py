from pathlib import Path


def test_elmo_v33_adds_two_days_without_touching_live_tree() -> None:
    source = Path("scripts/retrain_public_matchup_tree_elmo_v33.py").read_text()
    assert 'EXTRA_DAYS = ("2026-07-22", "2026-07-23")' in source
    assert "public-matchup-tree-calibration-v32/row-shards" in source
    assert "public-matchup-tree-calibration-v33" in source
    assert "trevenant-public-matchup-tree-iter5-v1.json" not in source
    assert 'STAGED_ARCHETYPES = ROOT / "src/poke_bot/archetypes_v33.py"' in source
    assert "/work/src/poke_bot/archetypes.py:ro" in source
    assert "--cpus" in source
    assert '"4"' in source
    assert "runtime_enabled" not in source
