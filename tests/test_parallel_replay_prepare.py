from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from poke_bot.pure_rl.replay_parallel_prepare import (
    PackedAdapterRouting,
    ParallelReplayUnavailable,
    RangeIdentity,
    _validate_ranges,
    deterministic_ranges,
    prepare_with_serial_fallback,
    scan_source,
)
import torch


def _source(tmp_path: Path, rows: int = 97) -> Path:
    path = tmp_path / "replay.jsonl"
    path.write_bytes(
        b"".join(
            (f'{{"episode_id":"e-{index}","decisions":[]}}\n').encode()
            for index in range(rows)
        )
    )
    return path


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 16, 32])
def test_ranges_scale_beyond_partition_count_and_cover_once(
    tmp_path: Path, workers: int
) -> None:
    source, offsets = scan_source(_source(tmp_path))
    ranges = deterministic_ranges(source, offsets, target_ranges=workers * 4)
    assert ranges[0].row_start == 0
    assert ranges[-1].row_end == source.rows
    assert ranges[0].byte_start == 0
    assert ranges[-1].byte_end == source.size_bytes
    assert len({item.identity for item in ranges}) == len(ranges)
    assert sum(item.row_end - item.row_start for item in ranges) == source.rows
    for left, right in zip(ranges, ranges[1:]):
        assert left.row_end == right.row_start
        assert left.byte_end == right.byte_start


def test_scan_rejects_truncated_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"episode_id":"partial"}')
    with pytest.raises(RuntimeError, match="partial row"):
        scan_source(path)


def test_duplicate_or_missing_ranges_fail_closed(tmp_path: Path) -> None:
    source, offsets = scan_source(_source(tmp_path, rows=8))
    ranges = deterministic_ranges(source, offsets, target_ranges=4)
    with pytest.raises(RuntimeError, match="duplicate, missing, or unordered"):
        _validate_ranges(source, [ranges[0], ranges[0], *ranges[2:]])
    missing = [ranges[0], *ranges[2:]]
    with pytest.raises(RuntimeError):
        _validate_ranges(source, missing)


def test_range_identity_binds_source_digest(tmp_path: Path) -> None:
    path = _source(tmp_path, rows=8)
    first, first_offsets = scan_source(path)
    first_ranges = deterministic_ranges(first, first_offsets, target_ranges=4)
    with path.open("ab") as stream:
        stream.write(b'{"episode_id":"late","decisions":[]}\n')
    second, second_offsets = scan_source(path)
    second_ranges = deterministic_ranges(second, second_offsets, target_ranges=4)
    assert first.sha256 != second.sha256
    assert [row.identity for row in first_ranges] != [row.identity for row in second_ranges]


def test_serial_fallback_and_strict_mode() -> None:
    def unavailable():
        raise ParallelReplayUnavailable("not installed")

    assert prepare_with_serial_fallback(
        unavailable, lambda: "serial", strict_parallel=False
    ) == ("serial", "serial_fallback")
    with pytest.raises(ParallelReplayUnavailable):
        prepare_with_serial_fallback(
            unavailable, lambda: "serial", strict_parallel=True
        )


def test_unexpected_parallel_failure_never_silently_falls_back() -> None:
    def corrupt():
        raise RuntimeError("fragment checksum mismatch")

    with pytest.raises(RuntimeError, match="checksum"):
        prepare_with_serial_fallback(corrupt, lambda: "serial", strict_parallel=False)


def test_packed_adapter_batches_preserve_legacy_order_and_caps() -> None:
    packed = PackedAdapterRouting(
        game_route=torch.tensor([-1, 3, 4, 3, -1, 4], dtype=torch.int16),
        game_seat=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int8),
        game_source_row=torch.arange(6, dtype=torch.int64),
        game_decisions=torch.tensor([2, 5, 4, 3, 9, 2], dtype=torch.int32),
        episode_utf8=torch.tensor(list(b"abcdef"), dtype=torch.uint8),
        episode_offset=torch.arange(7, dtype=torch.int64),
        ticket_utf8=torch.empty(0, dtype=torch.uint8),
        ticket_offset=torch.zeros(7, dtype=torch.int64),
    )
    batches = packed.routed_batches(
        games_per_batch=2,
        max_decisions=7,
        shuffle=False,
        seed=9,
        epoch=0,
        device=torch.device("cpu"),
    )
    assert [batch.tolist() for batch in batches] == [[1], [2, 3], [5]]
    first = packed.routed_batches(
        games_per_batch=2,
        max_decisions=7,
        shuffle=True,
        seed=91,
        epoch=2,
        device=torch.device("cpu"),
    )
    second = packed.routed_batches(
        games_per_batch=2,
        max_decisions=7,
        shuffle=True,
        seed=91,
        epoch=2,
        device=torch.device("cpu"),
    )
    assert [row.tolist() for row in first] == [row.tolist() for row in second]
