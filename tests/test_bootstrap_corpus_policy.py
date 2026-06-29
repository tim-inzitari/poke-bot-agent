"""Integration coverage for the consolidated bootstrap data gate.

Exercises poke_agent.training_diversity.assert_bootstrap_training_data — the single
entry point dataset.prepare_training_tensors now calls — to lock in two behaviors
introduced by the bloat-audit cleanup:

  1. The top-of-ladder gate auto-bypasses a CABT self-play corpus (archetype bootstrap),
     so archetype runs no longer need REQUIRE_TOP_OF_LADDER_DATA=0.
  2. A corpus of unknown provenance still trips the gate.
"""

from __future__ import annotations

import pytest

from poke_agent.training_diversity import TrainingDiversityError, assert_bootstrap_training_data


def _rows(source: str, *, matchups: int = 2) -> list[dict]:
    rows = []
    for episode in range(matchups):
        # Distinct deck0/deck1 per episode so matchup-diversity (when enabled) passes.
        rows.append({
            "episode": episode,
            "step": 0,
            "features": [0.0],
            "value": 0.0,
            "source": source,
            "deck0": f"dragapult-{episode}",
            "deck1": f"lucario-{episode}",
        })
    return rows


def test_ladder_gate_auto_bypasses_cabt_bootstrap():
    config = {
        "require_training_matchup_diversity": False,
        "require_top_of_ladder_data": True,
        "min_top_of_ladder_fraction": 0.0,
        "state_hash_dim": 32,
    }
    # CABT self-play corpus has zero ladder games — must NOT raise (auto-bypassed).
    assert_bootstrap_training_data(config, _rows("multideck-cabt"), data_path="lucario_bootstrap.jsonl")


def test_ladder_gate_still_fires_for_unknown_provenance():
    config = {
        "require_training_matchup_diversity": False,
        "require_top_of_ladder_data": True,
        "min_top_of_ladder_fraction": 0.0,
        "state_hash_dim": 32,
    }
    # Empty/unknown source is not a recognized CABT corpus → gate enforced.
    with pytest.raises(TrainingDiversityError, match="episodes-index|top-of-ladder|competition"):
        assert_bootstrap_training_data(config, _rows(""), data_path="mystery.jsonl")


def test_ladder_gate_disabled_skips_entirely():
    config = {
        "require_training_matchup_diversity": False,
        "require_top_of_ladder_data": False,
        "state_hash_dim": 32,
    }
    # Gate off → unknown provenance is fine.
    assert_bootstrap_training_data(config, _rows(""), data_path="anything.jsonl")
