from __future__ import annotations

import pytest

from poke_agent.episodes_index import is_top_of_ladder_source
from poke_agent.training_diversity import (
    TrainingDiversityError,
    assert_top_of_ladder_data,
    top_of_ladder_stats,
)


def _rows(sources: list[str]) -> list[dict]:
    rows = []
    for episode, source in enumerate(sources):
        rows.append({
            "episode": episode,
            "step": 0,
            "features": [0.0],
            "value": 0.0,
            "source": source,
        })
    return rows


def test_is_top_of_ladder_source():
    assert is_top_of_ladder_source("pokemon-tcg-ai-battle-episodes-2026-06-20")
    assert is_top_of_ladder_source("episode-ids-file")
    assert not is_top_of_ladder_source("multideck-cabt")
    assert not is_top_of_ladder_source("self_play")
    assert not is_top_of_ladder_source("ladder-scrape")
    assert not is_top_of_ladder_source("")
    assert not is_top_of_ladder_source(None)


def test_assert_top_of_ladder_data_passes_with_index_games():
    rows = _rows(["multideck-cabt", "pokemon-tcg-ai-battle-episodes-2026-06-20", "multideck-cabt"])
    stats = assert_top_of_ladder_data(rows)
    assert stats["ladder_games"] == 1
    assert stats["games"] == 3


def test_assert_top_of_ladder_data_rejects_synthetic_only():
    rows = _rows(["multideck-cabt", "multideck-cabt"])
    with pytest.raises(TrainingDiversityError, match="episodes-index|top-of-ladder|competition"):
        assert_top_of_ladder_data(rows)


def test_assert_top_of_ladder_data_enforces_min_fraction():
    rows = _rows(["multideck-cabt"] * 9 + ["pokemon-tcg-ai-battle-episodes-2026-06-20"])
    # 10% ladder; requiring 50% should fail.
    with pytest.raises(TrainingDiversityError, match="only"):
        assert_top_of_ladder_data(rows, min_fraction=0.5)
    # ...but the default (>=1 ladder game) passes.
    assert assert_top_of_ladder_data(rows)["ladder_games"] == 1


def test_top_of_ladder_stats_counts_sources():
    rows = _rows([
        "pokemon-tcg-ai-battle-episodes-2026-06-20",
        "pokemon-tcg-ai-battle-episodes-2026-06-20",
        "multideck-cabt",
    ])
    stats = top_of_ladder_stats(rows)
    assert stats["ladder_games"] == 2
    assert stats["ladder_fraction"] == pytest.approx(2 / 3)
