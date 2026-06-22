from pathlib import Path

from poke_agent.replay_import import convert_replay_file, is_replay_complete, load_replay_payload


def test_sample_replay_is_complete():
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/sample-replays/80560722.json"
    if not sample.is_file():
        return
    payload = load_replay_payload(sample)
    assert is_replay_complete(payload)


def test_convert_sample_replay():
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/sample-replays/80560722.json"
    if not sample.is_file():
        return
    rows = convert_replay_file(sample, episode=0, root=root, source="sample")
    assert rows
    assert rows[0]["observation"]
    assert rows[0]["action"]
    assert rows[0].get("complete") is True
    assert "deck0" in rows[0]
    assert -1.0 <= float(rows[-1].get("value", 0.0)) <= 1.0
    assert all(-1.0 <= float(row.get("value", 0.0)) <= 1.0 for row in rows)
