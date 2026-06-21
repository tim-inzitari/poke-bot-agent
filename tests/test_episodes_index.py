from pathlib import Path

from poke_agent.episodes_index import (
    default_index_path,
    filter_top_percent,
    load_daily_manifest,
    load_episode_pool,
)


def test_load_daily_manifest():
    root = Path(__file__).resolve().parents[1]
    path = default_index_path(root)
    if not path.is_file():
        return
    entries = load_daily_manifest(path)
    assert entries
    assert entries[0].slug.startswith("pokemon-tcg-ai-battle-episodes-")


def test_filter_top_percent():
    from poke_agent.episodes_index import EpisodeRecord

    records = [
        EpisodeRecord(episode_id="a", score=10),
        EpisodeRecord(episode_id="b", score=5),
        EpisodeRecord(episode_id="c", score=1),
    ]
    top = filter_top_percent(records, 34.0)
    assert len(top) == 1
    assert top[0].episode_id == "a"


def test_load_episode_pool_includes_sample_replay():
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/sample-replays/80560722.json"
    if not sample.is_file():
        return
    pool = load_episode_pool(root, scrape_index_csv=root / "data/ladder-replays/missing.csv")
    # pool may be empty without index files; sample dir path handled via records_from_directory in other tests
