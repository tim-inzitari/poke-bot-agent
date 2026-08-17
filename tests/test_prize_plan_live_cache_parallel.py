from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from poke_bot.prize_plan_live_cache import (
    _iter_raw_sequences_range,
    _scan_tasks,
)


def _write_rows(path: Path, count: int) -> list[dict[str, object]]:
    rows = [
        {
            "episode_id": f"episode-{index:04d}",
            "seat": index % 2,
            "decisions": [],
            "padding": "x" * (index % 17),
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def test_byte_ranges_cover_every_sequence_once(tmp_path: Path) -> None:
    shard = tmp_path / "replay.jsonl"
    expected = _write_rows(shard, 113)
    tasks = _scan_tasks([shard], workers=24)
    observed = [
        row
        for path, start, end in tasks
        for row in _iter_raw_sequences_range(path, start, end)
    ]
    assert sorted(row["episode_id"] for row in observed) == sorted(
        row["episode_id"] for row in expected
    )
    assert len(observed) == len(expected)


def test_multi_shard_plan_preserves_each_shard_and_uses_inner_ranges(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_rows(first, 200)
    _write_rows(second, 100)
    tasks = _scan_tasks([first, second], workers=24)
    assert len(tasks) >= 24
    assert {path for path, _start, _end in tasks} == {
        first.resolve(),
        second.resolve(),
    }
    for source in (first.resolve(), second.resolve()):
        ranges = sorted((start, end) for path, start, end in tasks if path == source)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == source.stat().st_size
        assert all(left[1] == right[0] for left, right in pairwise(ranges))
