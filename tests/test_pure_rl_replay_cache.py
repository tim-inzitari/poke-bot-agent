"""Correctness and crash-safety checks for compact pure-RL replay caching."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from poke_bot import config
from poke_bot.pure_rl.dataset_bridge import (
    StreamingReplayCache,
    _range_worker,
    dataset_from_shard,
)
from poke_bot.pure_rl.shards import CompactGame, CompactShardWriter


def _write_empty_games(path: Path, count: int) -> None:
    writer = CompactShardWriter(path)
    for index in range(count):
        writer.write_game(
            CompactGame(
                episode_id=f"episode-{index:04d}",
                seat=index % 2,
                archetype="core",
                opp_archetype="core",
                deck=[1] * 60,
                value=float(index % 2),
                decisions=[],
            )
        )


def test_byte_ranges_cover_each_json_row_once(tmp_path: Path) -> None:
    shard = tmp_path / "range.jsonl"
    _write_empty_games(shard, 37)
    split = shard.stat().st_size // 2 + 17  # intentionally inside a JSON row
    outputs = [tmp_path / "a.pkl", tmp_path / "b.pkl"]
    first = _range_worker(str(shard), 0, split, str(outputs[0]), False, 64)
    second = _range_worker(
        str(shard), split, shard.stat().st_size, str(outputs[1]), False, 64
    )
    assert first["records"] + second["records"] == 37
    assert first["bytes"] + second["bytes"] == shard.stat().st_size
    assert first["dropped"] + second["dropped"] == 37


def test_range_worker_fails_closed_on_malformed_json(tmp_path: Path) -> None:
    shard = tmp_path / "bad.jsonl"
    shard.write_text("{not-json}\n")
    with pytest.raises(json.JSONDecodeError):
        _range_worker(
            str(shard), 0, shard.stat().st_size, str(tmp_path / "bad.pkl"), False, 64
        )


def test_completed_cache_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shard = tmp_path / "run" / "shards" / "iter_00000.jsonl"
    _write_empty_games(shard, 41)
    monkeypatch.setattr(config.HARDWARE, "cache_dir", tmp_path / "cache")
    monkeypatch.setenv("PURE_RL_REPLAY_CACHE_PARALLEL_MIN_MIB", "0")
    monkeypatch.setenv("PURE_RL_REPLAY_FEATURIZE_WORKERS", "2")

    first = dataset_from_shard(shard, verify_info_set=False, max_context=64)
    assert len(first.sequences) == 0
    manifests = list((tmp_path / "cache").rglob("manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text())
    assert payload["records"] == 41
    assert payload["dropped"] == 41

    def _must_not_rebuild(*_args, **_kwargs):
        raise AssertionError("valid replay cache was rebuilt")

    monkeypatch.setattr(
        "poke_bot.pure_rl.dataset_bridge._build_parallel_cache", _must_not_rebuild
    )
    second = dataset_from_shard(shard, verify_info_set=False, max_context=64)
    assert len(second.sequences) == 0


def test_stream_cache_publishes_only_complete_source(tmp_path: Path, monkeypatch) -> None:
    shard = tmp_path / "run" / "shards" / "iter_00003.jsonl"
    monkeypatch.setattr(config.HARDWARE, "cache_dir", tmp_path / "cache")
    cache = StreamingReplayCache(
        shard,
        verify_info_set=False,
        max_context=64,
        workers=1,
        chunk_mib=1,
    )
    _write_empty_games(shard, 29)
    cache.note_append()
    manifest = cache.finish()
    assert manifest is not None
    assert manifest["stream_built"] is True
    assert manifest["records"] == 29
    assert manifest["dropped"] == 29
    assert sum(int(part["bytes"]) for part in manifest["parts"]) == shard.stat().st_size
    for part in manifest["parts"]:
        with Path(part["path"]).open("rb") as handle:
            assert len(pickle.load(handle)["sequences"]) == 0
