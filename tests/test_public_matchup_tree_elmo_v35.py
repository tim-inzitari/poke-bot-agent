from pathlib import Path


def test_v35_adds_simulator_rows_without_weakening_audit() -> None:
    source = Path(
        "scripts/retrain_public_matchup_tree_elmo_v35.py"
    ).read_text()
    assert "public-matchup-tree-calibration-v34/row-shards" in source
    assert "router-calibration-v35.zip" in source
    assert '"0.93"' in source
    assert '"--max-depth", "24"' in source
    assert '"--min-samples-leaf", "20"' in source
    unit = Path(
        "deploy/systemd/pokebot-public-tree-v35-after-sim.service"
    ).read_text()
    assert "--minimum-precision 0.93" in unit
    assert "--minimum-weighted-support 10000" in unit


def test_v35_simulator_is_bounded_and_public_only() -> None:
    unit = Path(
        "deploy/systemd/pokebot-router-calibration-v35-sim.service"
    ).read_text()
    assert "--cpus 8" in unit
    assert "--memory 12g" in unit
    assert "--games-per-target 1500" in unit
    source = Path(
        "scripts/generate_router_calibration_episodes.py"
    ).read_text()
    assert "causal_public_zones_only" in source
    assert "hidden_fields_written" in source
