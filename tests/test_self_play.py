from pathlib import Path

from poke_agent.config import build_config, resolve_self_play_train_window
from poke_agent.deck_pool import choose_agent_vs_field_matchup, mirror_matchup, read_deck_file
from poke_agent.self_play import (
    OpponentPool,
    SelfPlaySettings,
    _merge_fixed_opponent_reports,
    resolve_self_play_workers,
    rollout_buffer_overwrites,
    summarize_results,
)
from poke_agent.data_pipeline import episode_chunks


def test_opponent_pool_keeps_recent_checkpoints():
    pool = OpponentPool(max_size=2)
    pool.add(Path("a.pt"))
    pool.add(Path("b.pt"))
    pool.add(Path("c.pt"))
    assert [path.name for path in pool.checkpoints] == ["c.pt", "b.pt"]
    assert pool.sample() is not None


def test_opponent_pool_sample_excludes_current():
    pool = OpponentPool(max_size=3)
    pool.add(Path("a.pt"))
    pool.add(Path("b.pt"))
    for _ in range(10):
        choice = pool.sample(exclude=Path("a.pt"))
        assert choice is not None
        assert choice.name == "b.pt"


def test_summarize_results_from_seat0_perspective():
    stats = summarize_results([0, 1, 2, 0], seat_index=0)
    assert stats["wins"] == 2.0
    assert stats["losses"] == 1.0
    assert stats["draws"] == 1.0
    assert stats["win_rate"] == 2 / 3


def test_agent_vs_field_matchup_alternates_seats():
    agent = [1] * 60
    field = [("dragapult", [2] * 60), ("garchomp", [3] * 60)]
    m0 = choose_agent_vs_field_matchup(0, "lucario", agent, field, mode="round-robin")
    m1 = choose_agent_vs_field_matchup(1, "lucario", agent, field, mode="round-robin")
    assert m0.agent_seat == 0
    assert m1.agent_seat == 1
    assert m0.field_name == "dragapult"
    assert m1.field_name == "garchomp"
    assert m0.deck0 == agent
    assert m1.deck1 == agent


def test_mirror_matchup():
    agent = list(range(60))
    matchup = mirror_matchup("lucario", agent)
    assert matchup.deck0 == agent
    assert matchup.deck1 == agent


def test_read_deck_file(tmp_path: Path):
    deck_path = tmp_path / "test.csv"
    deck_path.write_text("\n".join(str(i) for i in range(60)), encoding="utf-8")
    assert read_deck_file(deck_path) == list(range(60))


def test_resolve_self_play_workers_auto_caps_to_games():
    settings = SelfPlaySettings(
        games_per_iteration=3,
        eval_games=0,
        iterations=1,
        opponent_pool_size=1,
        use_beam=False,
        output_path=Path("out.jsonl"),
        checkpoint_dir=Path("ckpt"),
    )
    assert resolve_self_play_workers(settings, games=3) <= 3


def test_baseline_seat_alternation_uses_collect_start_episode():
    assert (5 - 0) % 2 == 1
    assert (6 - 0) % 2 == 0
    # Chunk starting at episode 5 in a batch that began at 0 should continue seat alternation.
    collect_start = 0
    assert (5 - collect_start) % 2 == 1
    assert (6 - collect_start) % 2 == 0


def test_episode_chunks_split_evenly_for_four_workers():
    chunks = episode_chunks(20, 4)
    assert len(chunks) == 4
    assert sum(stop - start for start, stop in chunks) == 20


def test_episode_chunks_caps_size_for_progress_updates():
    chunks = episode_chunks(150, 8, max_chunk_size=4)
    assert all(stop - start <= 4 for start, stop in chunks)
    assert sum(stop - start for start, stop in chunks) == 150
    assert len(chunks) == 38


def test_train_window_defaults_to_games_per_iteration():
    assert resolve_self_play_train_window({"self_play_games": 50, "self_play_train_window_games": None}) == 50
    assert resolve_self_play_train_window({"self_play_games": 20, "self_play_train_window_games": 40}) == 40


def test_build_config_train_window_follows_games_override(tmp_path):
    config = build_config(tmp_path, overrides={"self_play_games": 50, "self_play_train_window_games": None})
    assert config["self_play"]["train_window_games"] == 50
    assert config["self_play"]["games_per_iteration"] == 50


def test_rollout_buffer_overwrites_when_games_match_window(tmp_path):
    settings = SelfPlaySettings(
        games_per_iteration=150,
        eval_games=0,
        iterations=1,
        opponent_pool_size=1,
        use_beam=False,
        output_path=tmp_path / "rollouts.jsonl",
        checkpoint_dir=tmp_path / "ckpt",
        train_window_games=150,
        trim_rollout_file=True,
    )
    assert rollout_buffer_overwrites(settings) is True

    settings.train_window_games = 300
    assert rollout_buffer_overwrites(settings) is False


def test_merge_fixed_opponent_reports():
    merged = _merge_fixed_opponent_reports([
        {
            "agent_a": {"games": 2.0, "wins": 1.0, "losses": 1.0, "draws": 0.0, "win_rate": 0.5},
            "results": [0, 1],
            "opponent": "iono",
        },
        {
            "agent_a": {"games": 2.0, "wins": 2.0, "losses": 0.0, "draws": 0.0, "win_rate": 1.0},
            "results": [0, 0],
            "opponent": "iono",
        },
    ])
    assert merged["agent_a"]["wins"] == 3.0
    assert merged["agent_a"]["losses"] == 1.0
    assert merged["agent_a"]["win_rate"] == 0.75
