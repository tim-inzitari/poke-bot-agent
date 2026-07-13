"""Parallel sim-worker pool with process recycling.

``libcg.so`` is sequential CPU and holds per-process state that grows slowly
over long runs (reported leaks on the RL/environment thread). We therefore run
games across a pool of worker processes and **recycle** each worker after a
bounded number of tasks (``maxtasksperchild``).

The pool uses the ``spawn`` start method (clean interpreter per worker, safe
alongside CUDA in the parent) and an initializer that puts the correct ``cg``
runtime on ``sys.path`` in every worker.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from . import config

T = TypeVar("T")
R = TypeVar("R")


def _worker_init(cg_lib_path: Optional[str]) -> None:
    """Runs once per worker process: make ``import cg`` resolve."""
    if cg_lib_path:
        os.environ["CG_LIB_PATH"] = cg_lib_path
    # Import here so each spawned worker binds its own cg runtime.
    from . import cg_env

    cg_env.ensure_cg_importable()


class WorkerPool:
    """Recycling process pool for simulator work.

    Example::

        with WorkerPool() as pool:
            for result in pool.imap_unordered(run_one_game, jobs):
                ...
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        recycle_games: Optional[int] = None,
        cg_lib_path: Optional[str] = None,
    ) -> None:
        self.num_workers = num_workers or config.HARDWARE.sim_workers
        self.recycle_games = recycle_games or config.HARDWARE.worker_recycle_games
        self.cg_lib_path = cg_lib_path or os.environ.get("CG_LIB_PATH")
        self._pool: Optional[mp.pool.Pool] = None

    def __enter__(self) -> "WorkerPool":
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(self.cg_lib_path,),
            maxtasksperchild=self.recycle_games,
        )
        return self

    def __exit__(self, *exc) -> None:
        assert self._pool is not None
        self._pool.close()
        self._pool.join()
        self._pool = None

    def imap_unordered(
        self, fn: Callable[[T], R], jobs: Iterable[T], chunksize: int = 1
    ) -> Iterator[R]:
        assert self._pool is not None, "Use WorkerPool as a context manager."
        return self._pool.imap_unordered(fn, jobs, chunksize=chunksize)

    def map(self, fn: Callable[[T], R], jobs: Iterable[T], chunksize: int = 1) -> list[R]:
        assert self._pool is not None, "Use WorkerPool as a context manager."
        return self._pool.map(fn, jobs, chunksize=chunksize)
