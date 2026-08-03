from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_heldout_inherits_canonical_local_worker_count_by_default() -> None:
    source = (ROOT / "scripts" / "train_pure_rl.py").read_text(encoding="utf-8")

    assert '"PURE_RL_HELDOUT_LOCAL_WORKERS",\n                            str(hw.sim_workers),' in source
    assert 'os.environ.get("PURE_RL_HELDOUT_LOCAL_WORKERS", "64")' not in source


def test_final_format_units_pin_blackwell_collection_and_heldout_to_96() -> None:
    for relative in (
        "deploy/systemd/pokebot-final-format-alakazam-r79-h10.service",
        "deploy/systemd/pokebot-final-format-marnie-r104-h10-rl.service",
    ):
        unit = (ROOT / relative).read_text(encoding="utf-8")
        assert "Environment=PURE_RL_SIM_WORKERS=96" in unit
        assert "Environment=PURE_RL_GAMES_IN_FLIGHT=96" in unit
        assert "Environment=PURE_RL_HELDOUT_LOCAL_WORKERS=96" in unit
