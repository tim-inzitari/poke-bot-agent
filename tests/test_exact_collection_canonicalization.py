from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.canonicalize_overcollected_iteration import canonicalize


def _row(index: int, *, self_play: bool, replacement: bool = False) -> str:
    return json.dumps(
        {
            "episode_id": f"episode-{index}",
            "target_provenance": {
                "self_play": self_play,
                "replacement_capacity": replacement,
            },
            "decisions": [{"env_step": 0, "selected_index": 0, "n_options": 1}],
        },
        separators=(",", ":"),
    )


def test_canonicalizer_atomically_drops_only_spare_attempts(tmp_path: Path) -> None:
    shard = tmp_path / "iter_00027.jsonl"
    shard.write_text(
        "\n".join(
            [
                _row(0, self_play=True),
                _row(1, self_play=True),
                _row(2, self_play=True, replacement=True),
                _row(3, self_play=False),
                _row(4, self_play=False),
                _row(5, self_play=False),
            ]
        )
        + "\n"
    )
    audit = tmp_path / "audit.json"

    result = canonicalize(
        shard,
        self_play_games=2,
        public_mix_games=3,
        audit_path=audit,
    )

    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    assert [row["episode_id"] for row in rows] == [
        "episode-0",
        "episode-1",
        "episode-3",
        "episode-4",
        "episode-5",
    ]
    assert result["discarded_replacement_capacity_games"] == 1
    assert result["canonical"]["source_games"] == 5
    assert json.loads(audit.read_text()) == result


def test_canonicalizer_refuses_to_hide_missing_primary_games(tmp_path: Path) -> None:
    shard = tmp_path / "iter.jsonl"
    original = (
        _row(0, self_play=True)
        + "\n"
        + _row(1, self_play=True, replacement=True)
        + "\n"
        + _row(2, self_play=False)
        + "\n"
    )
    shard.write_text(original)

    with pytest.raises(RuntimeError, match="refusing arbitrary replacement"):
        canonicalize(
            shard,
            self_play_games=2,
            public_mix_games=1,
            audit_path=tmp_path / "audit.json",
        )

    assert shard.read_text() == original
    assert not (tmp_path / "audit.json").exists()
