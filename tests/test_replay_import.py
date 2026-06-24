import json
from pathlib import Path

import pytest

from poke_agent.replay_import import convert_replay_file, is_replay_complete, load_replay_payload


def _find_sample_replay(root: Path) -> Path | None:
    sample = root / "data/sample-replays/80560722.json"
    if sample.is_file():
        return sample
    index_dir = root / "kaggle/input"
    if index_dir.is_dir():
        for bundle in sorted(index_dir.glob("pokemon-tcg-ai-battle-episodes-*")):
            for replay in sorted(bundle.glob("*.json")):
                return replay
    return None


def test_sample_replay_is_complete():
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/sample-replays/80560722.json"
    if not sample.is_file():
        pytest.skip(f"sample replay not present: {sample}")
    payload = load_replay_payload(sample)
    assert is_replay_complete(payload)


def test_convert_sample_replay():
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/sample-replays/80560722.json"
    if not sample.is_file():
        pytest.skip(f"sample replay not present: {sample}")
    rows = convert_replay_file(sample, episode=0, root=root, source="sample")
    assert rows
    assert rows[0]["observation"]
    assert rows[0]["action"]
    assert rows[0].get("complete") is True
    assert "deck0" in rows[0]
    assert -1.0 <= float(rows[-1].get("value", 0.0)) <= 1.0
    assert all(-1.0 <= float(row.get("value", 0.0)) <= 1.0 for row in rows)


def test_converted_rows_are_json_serializable():
    """Rollout rows must be plain JSON (features = list[float], not numpy arrays)."""
    root = Path(__file__).resolve().parents[1]
    sample = _find_sample_replay(root)
    if sample is None:
        pytest.skip("no replay available (data/sample-replays or kaggle/input)")
    rows = convert_replay_file(sample, episode=0, root=root, source="episodes-index")
    assert rows
    features = rows[0]["features"]
    assert isinstance(features, list)
    assert all(isinstance(v, float) for v in features)
    assert isinstance(rows[0]["next_features"], list)
    # Must not raise TypeError: Object of type ndarray is not JSON serializable.
    json.dumps(rows, separators=(",", ":"))
