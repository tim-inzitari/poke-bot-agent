from __future__ import annotations

import multiprocessing as mp
import threading
from queue import Empty
from typing import Any, Callable, Iterable, Iterator, TypeVar

R = TypeVar("R")

_PROGRESS_QUEUE: mp.Queue | None = None


def bind_progress_queue(queue: mp.Queue | None) -> None:
    """Attach a shared queue for per-game progress pings from worker processes."""
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = queue


def emit_game_progress(*, result: int | None = None, our_seat: int | None = None) -> None:
    """Notify the parent collector that one game finished (no-op outside worker pools)."""
    queue = _PROGRESS_QUEUE
    if queue is None:
        return
    if result is None or our_seat is None:
        queue.put(1)
    else:
        queue.put({"result": int(result), "our_seat": int(our_seat)})


def _pool_worker_init(
    progress_queue: mp.Queue | None,
    user_initializer: Callable[..., None],
    user_initargs: tuple[Any, ...],
) -> None:
    """Spawn-safe pool initializer: bind progress queue, then run user setup."""
    bind_progress_queue(progress_queue)
    user_initializer(*user_initargs)


def imap_persistent(
    *,
    workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    task_fn: Callable[..., R],
    tasks: Iterable[Any],
    progress_queue: mp.Queue | None = None,
) -> Iterator[R]:
    """Run tasks on a spawn pool whose workers initialize once and stay loaded."""
    ctx = mp.get_context("spawn")

    if progress_queue is not None:
        pool_initializer: Callable[..., None] = _pool_worker_init
        pool_initargs = (progress_queue, initializer, initargs)
    else:
        pool_initializer = initializer
        pool_initargs = initargs

    with ctx.Pool(processes=workers, initializer=pool_initializer, initargs=pool_initargs) as pool:
        yield from pool.imap_unordered(task_fn, tasks)


def iter_with_live_progress(
    iterator: Iterable[R],
    progress_queue: mp.Queue,
    progress: Any,
    *,
    on_game: Callable[[Any], None] | None = None,
) -> Iterator[R]:
    """Yield pool results while a side thread drains per-game progress from workers."""
    stop = threading.Event()

    def drain() -> None:
        while True:
            if stop.is_set():
                try:
                    message = progress_queue.get_nowait()
                except Empty:
                    break
            else:
                try:
                    message = progress_queue.get(timeout=0.05)
                except Empty:
                    continue
            progress.update(1)
            if on_game is not None:
                on_game(message)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        yield from iterator
    finally:
        stop.set()
        thread.join(timeout=5.0)
        while True:
            try:
                message = progress_queue.get_nowait()
            except Empty:
                break
            progress.update(1)
            if on_game is not None:
                on_game(message)
