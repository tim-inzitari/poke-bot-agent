import multiprocessing as mp
import os

from poke_bot.worker_pool import WorkerPool


def _remote_slot_task(payload) -> tuple[int, int]:
    from poke_bot.batched_infer import _REMOTE

    seen, ready = payload
    seen[os.getpid()] = int(_REMOTE["slot"])
    if len(seen) >= 2:
        ready.set()
    if not ready.wait(10):
        raise TimeoutError("both worker slots were not scheduled")
    return int(_REMOTE["slot"]), int(_REMOTE["generation"])


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
