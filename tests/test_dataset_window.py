from poke_agent.dataset import limit_dataset_games, limit_recent_dataset_games, trim_rollout_jsonl


def _rows_for_episodes(episode_ids: list[int]) -> list[dict]:
    rows: list[dict] = []
    for episode in episode_ids:
        rows.append({"episode": episode, "step": 0})
        rows.append({"episode": episode, "step": 1})
    return rows


def test_limit_recent_dataset_games_keeps_highest_episode_ids():
    rows = _rows_for_episodes([1, 2, 3, 4, 5])
    limited, source_rows, source_games = limit_recent_dataset_games(rows, 2)
    kept = sorted({int(row["episode"]) for row in limited})
    assert kept == [4, 5]
    assert source_rows == 10
    assert source_games == 5


def test_limit_dataset_games_keeps_lowest_episode_ids():
    rows = _rows_for_episodes([1, 2, 3, 4, 5])
    limited, _, _ = limit_dataset_games(rows, 2)
    kept = sorted({int(row["episode"]) for row in limited})
    assert kept == [1, 2]


def test_trim_rollout_jsonl_rewrites_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    rows = _rows_for_episodes([10, 11, 12, 13])
    path.write_text("\n".join(
        '{"episode": %d, "step": %d}' % (row["episode"], row["step"])
        for row in rows
    ) + "\n", encoding="utf-8")

    before, after = trim_rollout_jsonl(path, 2)
    assert before == 4
    assert after == 2

    trimmed = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(trimmed) == 4
    episodes = sorted({int(__import__("json").loads(line)["episode"]) for line in trimmed})
    assert episodes == [12, 13]
