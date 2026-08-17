import json

import pytest

from poke_bot.replay_writer import OrderedReplayWriter, ReplayWriterError
from poke_bot.worker_pool import _cap_worker_native_threads


def _record(index: int) -> str:
    return json.dumps(
        {
            "episode_id": f"game-{index}",
            "steps": [{"observation": {}, "action": [0]}],
        }
    )


def test_ordered_writer_writes_each_game_once_in_job_order(tmp_path) -> None:
    partial = tmp_path / "iter000.jsonl.partial"
    writer = OrderedReplayWriter(
        partial, expected_jobs=3, queue_depth=2, fsync_batch=2
    )
    writer.submit(2, _record(2), {"winner": 0})
    writer.submit(0, _record(0), {"winner": 1})
    writer.submit(1, _record(1), {"winner": 2})
    telemetry = writer.close()
    assert telemetry["next_index"] == 3
    assert telemetry["written_records"] == 3
    rows = [json.loads(line) for line in partial.read_text().splitlines()]
    assert [row["episode_id"] for row in rows] == [
        "game-0",
        "game-1",
        "game-2",
    ]


def test_ordered_writer_recovers_at_fsync_boundary_without_duplicates(
    tmp_path,
) -> None:
    partial = tmp_path / "iter001.jsonl.partial"
    first = OrderedReplayWriter(
        partial, expected_jobs=3, queue_depth=2, fsync_batch=1
    )
    first.submit(0, _record(0), {"winner": 0})
    with pytest.raises(ReplayWriterError, match="1/3"):
        first.close()

    resumed = OrderedReplayWriter(
        partial, expected_jobs=3, queue_depth=2, fsync_batch=1
    )
    assert resumed.resume_index == 1
    assert resumed.submit(0, _record(0), {"winner": 0}) is False
    resumed.submit(2, _record(2), {"winner": 0})
    resumed.submit(1, _record(1), {"winner": 0})
    resumed.close()
    rows = [json.loads(line) for line in partial.read_text().splitlines()]
    assert [row["episode_id"] for row in rows] == [
        "game-0",
        "game-1",
        "game-2",
    ]


def test_ordered_writer_abort_preserves_resumable_contiguous_prefix(
    tmp_path,
) -> None:
    partial = tmp_path / "iter002.jsonl.partial"
    first = OrderedReplayWriter(
        partial, expected_jobs=4, queue_depth=4, fsync_batch=1
    )
    first.submit(0, _record(0), {"winner": 0})
    first.submit(2, _record(2), {"winner": 0})
    telemetry = first.abort("mid-iteration fatal health")
    assert telemetry["aborted"] is True
    assert telemetry["next_index"] == 1

    resumed = OrderedReplayWriter(
        partial, expected_jobs=4, queue_depth=4, fsync_batch=1
    )
    assert resumed.resume_index == 1
    for index in range(1, 4):
        resumed.submit(index, _record(index), {"winner": 0})
    resumed.close()
    rows = [json.loads(line) for line in partial.read_text().splitlines()]
    assert [row["episode_id"] for row in rows] == [
        "game-0",
        "game-1",
        "game-2",
        "game-3",
    ]


def test_ordered_writer_rejects_sentinel_job_indices(tmp_path) -> None:
    writer = OrderedReplayWriter(
        tmp_path / "iter003.jsonl.partial",
        expected_jobs=1,
    )
    with pytest.raises(ReplayWriterError, match="out of range"):
        writer.submit(-1, None, {})
    writer.submit(0, None, {"accounted": True})
    writer.close()


def test_sim_worker_thread_caps_override_inherited_values(monkeypatch) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "32")
    _cap_worker_native_threads()
    assert all(
        __import__("os").environ[name] == "1"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    )
