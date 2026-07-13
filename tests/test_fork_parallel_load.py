import json

import numpy as np
import pytest

from poke_agent.dataset import load_jsonl
from poke_agent.features import build_training_arrays


def _minimal_rows() -> list[dict]:
    feat = [0.1] * 32
    rows: list[dict] = []
    for episode in range(4):
        for player in (0, 1):
            for step in range(3):
                rows.append(
                    {
                        "episode": episode,
                        "player": player,
                        "step": step,
                        "action": 0,
                        "features": feat,
                        "next_features": feat,
                        "result": 0 if player == 0 else 1,
                        "deck0": "deck-a",
                        "deck1": "deck-b",
                        "value": 0.0,
                    }
                )
    return rows


def test_load_jsonl_single_pass(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = _minimal_rows()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    loaded = load_jsonl(path)
    assert len(loaded) == len(rows)
    assert loaded == rows


def test_build_training_arrays_parallel_matches_single_worker():
    rows = _minimal_rows()
    kwargs = dict(
        transition_classes=16,
        state_hash_dim=8,
        window_size=4,
        card_vocab_size=64,
    )
    single = build_training_arrays(rows, workers=1, **kwargs)
    parallel = build_training_arrays(rows, workers=2, **kwargs)
    for left, right in zip(single, parallel):
        if isinstance(left, np.ndarray) and left.dtype.kind in {"f", "i"}:
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
