import numpy as np
import pytest

from poke_agent.training_diversity import (
    TrainingDiversityError,
    assert_deck_metadata_not_in_features,
    assert_submission_deck_separate_from_training,
    assert_training_matchup_diversity,
    training_matchup_stats,
)


def _row(episode: int, deck0: str, deck1: str, step: int = 0) -> dict:
    obs = {
        "current": {
            "turn": step,
            "yourIndex": 0,
            "firstPlayer": 0,
            "result": -1,
            "players": [
                {"deckCount": 40, "handCount": 5, "bench": []},
                {"deckCount": 40, "handCount": 5, "bench": []},
            ],
        },
        "select": {"option": [0, 1], "minCount": 1, "maxCount": 1},
        "logs": [],
    }
    return {
        "episode": episode,
        "step": step,
        "deck0": deck0,
        "deck1": deck1,
        "features": [0.0] * 27,
        "next_features": [0.0] * 27,
        "observation": obs,
        "next_observation": obs,
        "action": [0],
        "value": 0.0,
        "terminal": False,
    }


def test_training_matchup_stats_counts_diverse_matchups():
    rows = [
        _row(0, "dragapult", "lucario"),
        _row(1, "gardevoir", "charizard"),
    ]
    stats = training_matchup_stats(rows)
    assert stats["games"] == 2
    assert stats["unique_matchups"] == 2
    assert stats["unique_deck_slugs"] == 4
    assert stats["mirror_only"] is False


def test_mirror_only_training_rejected():
    rows = [_row(0, "lucario", "lucario"), _row(1, "lucario", "lucario")]
    with pytest.raises(TrainingDiversityError, match="mirror"):
        assert_training_matchup_diversity(rows)


def test_single_matchup_rejected():
    rows = [_row(0, "dragapult", "lucario"), _row(1, "dragapult", "lucario")]
    with pytest.raises(TrainingDiversityError, match="distinct matchup"):
        assert_training_matchup_diversity(rows)


def test_allow_single_matchup_for_smoke():
    rows = [_row(0, "dragapult", "lucario")]
    stats = assert_training_matchup_diversity(rows, allow_single_matchup=True)
    assert stats["games"] == 1


def test_deck_metadata_not_in_features():
    rows = [_row(0, "dragapult", "lucario")]
    assert_deck_metadata_not_in_features(rows, state_hash_dim=8)


def test_submission_mirror_only_rejected(tmp_path):
    deck_path = tmp_path / "mega-lucario.csv"
    deck_path.write_text("\n".join(str(i) for i in range(60)))
    rows = [_row(0, "mega-lucario", "mega-lucario"), _row(1, "mega-lucario", "mega-lucario")]
    config = {"agent_deck_path": deck_path}
    with pytest.raises(TrainingDiversityError, match="submission deck"):
        assert_submission_deck_separate_from_training(config, rows)
