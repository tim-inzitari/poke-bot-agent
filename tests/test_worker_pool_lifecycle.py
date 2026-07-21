import multiprocessing as mp
import os
import signal
import time

import pytest

from poke_bot.worker_pool import WorkerPool, WorkerPoolStopped


def _remote_slot_task(payload) -> tuple[int, int]:
    from poke_bot.batched_infer import _REMOTE

    seen, ready = payload
    seen[os.getpid()] = int(_REMOTE["slot"])
    if len(seen) >= 2:
        ready.set()
    if not ready.wait(10):
        raise TimeoutError("both worker slots were not scheduled")
    return int(_REMOTE["slot"]), int(_REMOTE["generation"])


def _single_apply_task(value: int) -> tuple[int, int]:
    return os.getpid(), value * 2


def _recycled_remote_slot_task(value: int) -> tuple[int, int, int, int]:
    from poke_bot.batched_infer import _REMOTE

    return (
        os.getpid(),
        int(_REMOTE["slot"]),
        int(_REMOTE["generation"]),
        value,
    )


def _remote_channel(workers: int) -> dict:
    ctx = mp.get_context("spawn")
    return {
        "req_qs": [ctx.Queue(maxsize=8)],
        "resp_qs": [ctx.Queue(maxsize=2) for _ in range(workers)],
        "generation": 7,
    }


def test_remote_slots_reset_safely_across_repeated_pools() -> None:
    remote = _remote_channel(2)
    generations = []
    with mp.get_context("spawn").Manager() as manager:
        for _ in range(3):
            seen = manager.dict()
            ready = manager.Event()
            with WorkerPool(num_workers=2, remote_channel=remote) as pool:
                rows = list(
                    pool.imap_unordered(
                        _remote_slot_task,
                        [(seen, ready), (seen, ready)],
                    )
                )
                assert sorted(slot for slot, _generation in rows) == [0, 1]
                generations.append(rows[0][1])
    assert len(set(generations)) == 3


def test_mid_iteration_stop_terminates_once_without_slot_overflow() -> None:
    remote = _remote_channel(2)
    for _ in range(3):
        pool = WorkerPool(num_workers=2, remote_channel=remote)
        pool.__enter__()
        pool.request_stop("fatal health test")
        pool.request_stop("duplicate stop must be idempotent")
        pool.__exit__(None, None, None)
        assert pool.stopped is True


def test_apply_runs_single_job_in_pool_child() -> None:
    with WorkerPool(num_workers=1) as pool:
        child_pid, value = pool.apply(_single_apply_task, 21)
    assert child_pid != os.getpid()
    assert value == 42


def test_imap_unordered_surfaces_latched_stop_while_waiting_for_result() -> None:
    """A lost Pool result must not leave the scheduler emitter blocked forever."""

    class _NeverResult:
        def next(self, timeout):
            assert timeout == 1.0
            raise mp.TimeoutError

    class _FakePool:
        def imap_unordered(self, _fn, _jobs, chunksize=1):
            assert chunksize == 1
            return _NeverResult()

    pool = WorkerPool(num_workers=1)
    pool._pool = _FakePool()  # type: ignore[assignment]
    results = pool.imap_unordered(_single_apply_task, [1])
    pool._terminated = True
    pool._stop_reason = "synthetic abrupt worker loss"

    with pytest.raises(WorkerPoolStopped, match="synthetic abrupt worker loss"):
        next(results)


def test_imap_unordered_stops_after_continuous_capacity_loss(monkeypatch) -> None:
    """A lost local result must eventually terminate its unmonitored pool."""

    class _DeadProcess:
        pid = 123

        @staticmethod
        def is_alive() -> bool:
            return False

    class _NeverResult:
        calls = 0

        def next(self, timeout):
            assert timeout == 1.0
            self.calls += 1
            raise mp.TimeoutError

    class _FakePool:
        def __init__(self):
            self._pool = [_DeadProcess()]
            self.results = _NeverResult()
            self.terminate_calls = 0

        def imap_unordered(self, _fn, _jobs, chunksize=1):
            assert chunksize == 1
            return self.results

        def terminate(self):
            self.terminate_calls += 1

    ticks = iter((10.0, 12.0, 15.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    fake_pool = _FakePool()
    pool = WorkerPool(num_workers=1, capacity_recovery_grace_s=5.0)
    pool._pool = fake_pool  # type: ignore[assignment]

    with pytest.raises(
        WorkerPoolStopped,
        match=r"live=0 expected=1 unhealthy_for=5\.0s grace=5\.0s",
    ):
        next(pool.imap_unordered(_single_apply_task, [1]))

    assert pool.stopped is True
    assert fake_pool.results.calls == 3
    assert fake_pool.terminate_calls == 1


def test_imap_unordered_capacity_recovery_resets_grace(monkeypatch) -> None:
    """A normal recycle may recover and later receive a fresh grace window."""

    class _Process:
        pid = 123

        def __init__(self):
            self._states = iter((False, False, True, False, False))

        def is_alive(self) -> bool:
            return next(self._states)

    class _FiniteTimeouts:
        calls = 0

        def next(self, timeout):
            assert timeout == 1.0
            self.calls += 1
            if self.calls <= 5:
                raise mp.TimeoutError
            raise StopIteration

    class _FakePool:
        def __init__(self):
            self._pool = [_Process()]
            self.results = _FiniteTimeouts()
            self.terminate_calls = 0

        def imap_unordered(self, _fn, _jobs, chunksize=1):
            return self.results

        def terminate(self):
            self.terminate_calls += 1

    # The two capacity-loss windows are each four seconds.  Without the full
    # recovery resetting the first window, the second would fail immediately.
    ticks = iter((0.0, 4.0, 10.0, 14.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    fake_pool = _FakePool()
    pool = WorkerPool(num_workers=1, capacity_recovery_grace_s=5.0)
    pool._pool = fake_pool  # type: ignore[assignment]

    assert list(pool.imap_unordered(_single_apply_task, [1])) == []
    assert pool.stopped is False
    assert fake_pool.results.calls == 6
    assert fake_pool.terminate_calls == 0


def test_remote_worker_recycling_releases_slot_and_rotates_generation() -> None:
    """Remote services must recycle libcg workers without exhausting slots."""
    remote = _remote_channel(1)
    with WorkerPool(
        num_workers=1,
        recycle_games=1,
        remote_channel=remote,
    ) as pool:
        assert pool.worker_capacity_healthy
        assert len(pool.ready_worker_pids) == 1
        rows = [pool.apply(_recycled_remote_slot_task, value) for value in range(4)]

    pids = [pid for pid, _slot, _generation, _value in rows]
    slots = [slot for _pid, slot, _generation, _value in rows]
    generations = [generation for _pid, _slot, generation, _value in rows]
    values = [value for _pid, _slot, _generation, value in rows]
    assert len(set(pids)) == len(rows)
    assert slots == [0, 0, 0, 0]
    assert len(set(generations)) == len(rows)
    assert values == [0, 1, 2, 3]


def test_remote_slot_monitor_tolerates_live_retiring_owner() -> None:
    """Pool-list absence alone must not fail a clean retiring child."""

    class _EmptyPool:
        _pool = ()

        def terminate(self) -> None:
            raise AssertionError("live retiring owner must not terminate the pool")

    ctx = mp.get_context("spawn")
    owner = os.getpid()
    pool = WorkerPool(num_workers=1)
    pool._pool = _EmptyPool()  # type: ignore[assignment]
    pool._slot_condition = ctx.Condition(ctx.RLock())
    pool._slot_owners = ctx.Array("q", [owner], lock=False)
    pool._slot_ready = ctx.Array("q", [owner], lock=False)
    pool._worker_failure_event = ctx.Event()
    pool._start_worker_monitor()
    try:
        # Three monitor periods cover the dead-owner confirmation window. The
        # pre-fix monitor stopped on its first sample because _pool is empty.
        time.sleep(0.20)
        assert pool.stopped is False
        assert int(pool._slot_owners[0]) == owner
        assert int(pool._slot_ready[0]) == owner
        assert pool._worker_failure_event.is_set() is False
    finally:
        pool._finish_worker_monitor()


def test_remote_worker_recycling_stress_keeps_pool_healthy() -> None:
    """Concurrent recycle waves must not trip the remote slot reaper."""
    workers = 3
    values = list(range(24))
    remote = _remote_channel(workers)
    with WorkerPool(
        num_workers=workers,
        recycle_games=1,
        remote_channel=remote,
    ) as pool:
        rows = list(
            pool.imap_unordered(
                _recycled_remote_slot_task,
                values,
                chunksize=1,
            )
        )
        pool.wait_until_ready(timeout_s=30)
        assert pool.stopped is False
        assert pool.worker_capacity_healthy

    assert sorted(value for _pid, _slot, _generation, value in rows) == values
    assert len({pid for pid, _slot, _generation, _value in rows}) == len(values)
    assert len({generation for _pid, _slot, generation, _value in rows}) == len(
        values
    )
    assert {slot for _pid, slot, _generation, _value in rows} <= set(
        range(workers)
    )


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires POSIX SIGKILL")
def test_abrupt_remote_child_death_fails_closed_without_replacement_thrash() -> None:
    """A crash bypasses Finalize, but must not lose a lease or spin forever."""
    remote = _remote_channel(1)
    pool = WorkerPool(
        num_workers=1,
        recycle_games=100,
        remote_channel=remote,
    )
    pool.__enter__()
    victim = pool.ready_worker_pids[0]
    os.kill(victim, signal.SIGKILL)

    deadline = time.monotonic() + 5.0
    while not pool.stopped and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert pool.stopped is True
        assert pool.worker_capacity_healthy is False
        # One replacement may begin before the 50ms parent reaper observes the
        # dead owner. The shared failure latch prevents an unbounded spawn loop.
        assert pool.initializer_attempts <= 3
        with pytest.raises(WorkerPoolStopped):
            pool.apply(_single_apply_task, 1)
    finally:
        pool.__exit__(None, None, None)
