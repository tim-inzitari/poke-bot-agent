from pathlib import Path

from poke_agent.deck_pool import choose_agent_vs_field_matchup, mirror_matchup, read_deck_file
from poke_agent.self_play import OpponentPool, SelfPlaySettings, resolve_self_play_workers, summarize_results
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


def test_episode_chunks_cover_all_games():
    chunks = episode_chunks(20, 6)
    assert chunks[0][0] == 0
    assert chunks[-1][1] == 20
    assert sum(stop - start for start, stop in chunks) == 20
