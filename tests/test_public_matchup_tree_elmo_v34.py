from pathlib import Path


def test_elmo_v34_adds_sparse_route_history_without_touching_live_tree() -> None:
    source = Path("scripts/retrain_public_matchup_tree_elmo_v34.py").read_text()
    assert "public-matchup-tree-calibration-v33/row-shards" in source
    assert "public-matchup-tree-calibration-v34" in source
    for day in (
        "2026-06-26",
        "2026-06-27",
        "2026-06-28",
        "2026-06-29",
        "2026-06-30",
        "2026-07-01",
    ):
        assert day in source
    assert "trevenant-public-matchup-tree-iter5-v1.json" not in source
    assert 'STAGED_ARCHETYPES = ROOT / "src/poke_bot/archetypes_v33.py"' in source
    assert "/work/src/poke_bot/archetypes.py:ro" in source
    assert "--cpus" in source
    assert '"4"' in source
    assert "runtime_enabled" not in source
