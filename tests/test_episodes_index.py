from pathlib import Path

import pytest

from poke_agent.episodes_index import (
    DailyDatasetEntry,
    EpisodeRecord,
    daily_slugs_for_top_games,
    default_index_path,
    filter_top_percent,
    load_daily_manifest,
    load_episode_pool,
)


def test_load_daily_manifest():
    root = Path(__file__).resolve().parents[1]
    path = default_index_path(root)
    if not path.is_file():
        pytest.skip(f"Kaggle episodes index manifest not present: {path}")
    entries = load_daily_manifest(path)
    assert entries
    assert entries[0].slug.startswith("pokemon-tcg-ai-battle-episodes-")


def test_filter_top_percent():
    records = [
        EpisodeRecord(episode_id="a", score=10),
        EpisodeRecord(episode_id="b", score=5),
        EpisodeRecord(episode_id="c", score=1),
    ]
    top = filter_top_percent(records, 34.0)
    assert len(top) == 1
    assert top[0].episode_id == "a"


def test_daily_slugs_for_top_games_latest_day_at_100_percent():
    entries = [
        DailyDatasetEntry(
            date="2026-06-01",
            slug="pokemon-tcg-ai-battle-episodes-2026-06-01",
            url="",
            episode_count=100,
            total_bytes=0,
            top_avg_score=0.5,
            median_avg_score=0.3,
        ),
        DailyDatasetEntry(
            date="2026-06-02",
            slug="pokemon-tcg-ai-battle-episodes-2026-06-02",
            url="",
            episode_count=100,
            total_bytes=0,
            top_avg_score=0.9,
            median_avg_score=0.4,
        ),
    ]
    slugs = daily_slugs_for_top_games(entries, top_percent=100.0)
    assert slugs == ["pokemon-tcg-ai-battle-episodes-2026-06-02"]


def test_load_episode_pool_includes_sample_replay():
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/sample-replays/80560722.json"
    if not sample.is_file():
        pytest.skip(f"sample replay not present: {sample}")
    pool = load_episode_pool(root, scrape_index_csv=root / "data/ladder-replays/missing.csv")
    assert isinstance(pool, list)
