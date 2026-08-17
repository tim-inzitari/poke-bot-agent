from pathlib import Path


def test_elmo_v32_stages_without_replacing_live_tree() -> None:
    source = Path("scripts/retrain_public_matchup_tree_elmo_v32.py").read_text()
    assert "public-matchup-tree-calibration-v32" in source
    assert "trevenant-public-matchup-tree-iter5-v1.json" not in source
    assert "--cpus" in source
    assert '"4"' in source
    assert "--memory" in source
    assert '"24g"' in source
    assert "refusing replacement" in source
