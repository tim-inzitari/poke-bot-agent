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

import atexit
import itertools
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from . import config

T = TypeVar("T")
R = TypeVar("R")
_POOL_GENERATIONS = itertools.count(1)


class WorkerPoolStopped(RuntimeError):
    """The parent cancelled an active simulator pool."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def _cap_worker_native_threads() -> None:
    """Override inherited BLAS/OpenMP caps for one simulator process."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"


def _release_slot(slot_queue, slot: int) -> None:
    """Best-effort lease return so a recycled/dead worker cannot starve the pool."""
    if slot_queue is None:
        return
    try:
        slot_queue.put_nowait(int(slot))
    except Exception:
        pass


def _worker_init(cg_lib_path: Optional[str], remote_channel=None) -> None:
    """Runs once per worker process: make ``import cg`` resolve.

    Two safety knobs (must be applied BEFORE torch is first imported in the
    worker, hence here at the very top of the child process):

    * ``POKEBOT_WORKER_CPU_ONLY`` → set ``CUDA_VISIBLE_DEVICES=""`` so this sim
      worker can NEVER create a CUDA context. This prevents GPU oversubscription
      (dozens of workers × ~400 MiB CUDA context each → false ``CUDA out of
      memory``). Sim/game rollouts (incl. rule-based baselines) then run purely
      on CPU; only the parent process touches the GPU.
    * Thread caps → avoid CPU thread oversubscription when many workers each
      spin up BLAS/torch intra-op pools on the shared box.

    ``remote_channel`` (optional) = ``(req_qs|req_q, resp_qs, slot_counter)`` for
    the persistent GPU leaf-eval server(s). ``req_qs`` may be one Queue or a list
    of Queues (one per replica). Each worker claims a stable slot and registers
    ``(req_qs[slot % n], resp_qs[slot])`` so its MCTS can offload leaf forwards
    to the GPU server while staying CPU-only itself.
    """
    if cg_lib_path:
        os.environ["CG_LIB_PATH"] = cg_lib_path
    if _env_truthy("POKEBOT_WORKER_CPU_ONLY"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Override inherited parent values (the live run inherited 32). These are
    # simulator processes; one native thread each is intentional.
    _cap_worker_native_threads()
    # Import here so each spawned worker binds its own cg runtime.
    from . import cg_env

    cg_env.ensure_cg_importable()
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    if remote_channel is not None:
        # Preferred dict shape carries a per-pool generation and server
        # identity. A legacy 3-tuple remains accepted for local callers.
        # req_qs may be a single Queue OR a list of Queues (one per GPU replica);
        # workers are sharded across replicas by slot % n_servers.
        if isinstance(remote_channel, dict):
            req_qs = remote_channel["req_qs"]
            resp_qs = remote_channel["resp_qs"]
            slot_queue = remote_channel.get("slot_queue")
            slot_counter = remote_channel.get("slot_counter")
            generation = int(remote_channel.get("generation", 0))
            alive_evts = remote_channel.get("alive_evts")
            expected_digest = remote_channel.get("expected_digest")
            expected_version = remote_channel.get("expected_version")
            timeout_s = remote_channel.get("timeout_s")
            stop_event = remote_channel.get("stop_event")
        else:
            req_qs, resp_qs, slot_counter = remote_channel
            slot_queue = None
            generation = 0
            alive_evts = None
            expected_digest = None
            expected_version = None
            timeout_s = None
            stop_event = None
        if stop_event is not None and stop_event.is_set():
            # Exit quietly during teardown so Pool.terminate() does not get a
            # traceback flood / replacement death spiral.
            os._exit(0)
        if slot_queue is not None:
            deadline = time.monotonic() + 30.0
            slot = None
            while slot is None:
                if stop_event is not None and stop_event.is_set():
                    os._exit(0)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Parent likely terminated workers without reclaiming leases.
                    # Exit quietly; request_stop/terminate must stop replacements.
                    if stop_event is not None and stop_event.is_set():
                        os._exit(0)
                    raise WorkerPoolStopped(
                        "no response slot available for this pool generation"
                    )
                try:
                    slot = int(slot_queue.get(timeout=min(1.0, remaining)))
                except queue.Empty:
                    continue
            atexit.register(_release_slot, slot_queue, slot)
        else:
            with slot_counter.get_lock():
                slot = int(slot_counter.value)
                slot_counter.value = slot + 1
        if not 0 <= slot < len(resp_qs):
            raise RuntimeError(
                f"remote response slot overflow: slot={slot}, queues={len(resp_qs)}"
            )
        if isinstance(req_qs, (list, tuple)):
            server_idx = slot % len(req_qs)
            req_q = req_qs[server_idx]
        else:
            server_idx = 0
            req_q = req_qs
        from .batched_infer import set_remote_leaf_channel

        set_remote_leaf_channel(
            slot,
            req_q,
            resp_qs[slot],
            generation=generation,
            alive_evt=(alive_evts[server_idx] if alive_evts else None),
            expected_digest=expected_digest,
            expected_version=expected_version,
            timeout_s=timeout_s,
            stop_event=stop_event,
        )


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
        remote_channel=None,
    ) -> None:
        # Default worker count comes from the central hardware profile (all 32
        # CPU threads on the dedicated box); recycle_games bounds libcg's slow
        # per-process leak so long unattended runs have no unbounded RAM growth.
        self.num_workers = num_workers or config.HARDWARE.sim_workers
        self.recycle_games = recycle_games or config.HARDWARE.worker_recycle_games
        self.cg_lib_path = cg_lib_path or os.environ.get("CG_LIB_PATH")
        # Remote GPU leaf-eval channel = (req_q, resp_qs, slot_counter). When
        # present, recycling is DISABLED so each worker keeps its slot for the
        # lifetime of the pool (the per-iteration pool teardown still bounds the
        # libcg leak, since the pool is recreated each RL iteration).
        self.remote_channel = remote_channel
        self._pool: Optional[mp.pool.Pool] = None
        self._pool_remote_channel = None
        self._stop_event = None
        self._slot_queue = None
        self._terminated = False
        self._stop_reason: Optional[str] = None

    def __enter__(self) -> "WorkerPool":
        ctx = mp.get_context("spawn")
        maxtasks = None if self.remote_channel is not None else self.recycle_games
        remote_channel = self.remote_channel
        if isinstance(remote_channel, dict):
            resp_qs = list(remote_channel.get("resp_qs") or [])
            if len(resp_qs) < self.num_workers:
                raise ValueError(
                    "remote response queues must cover every active worker: "
                    f"queues={len(resp_qs)} workers={self.num_workers}"
                )
            # Every pool generation gets a fresh bounded lease set. A worker
            # replacement can wait for a lease, but can never consume an
            # unbounded monotonic slot or collide with another active worker.
            slot_queue = ctx.Queue(maxsize=self.num_workers)
            for slot in range(self.num_workers):
                slot_queue.put(slot)
            self._slot_queue = slot_queue
            self._stop_event = ctx.Event()
            generation = (
                int(remote_channel.get("generation", 0)) * 1_000_000
                + next(_POOL_GENERATIONS)
            )
            remote_channel = {
                **remote_channel,
                "generation": generation,
                "slot_queue": slot_queue,
                "stop_event": self._stop_event,
            }
            # A prior generation may have been cancelled after its caller died.
            # New workers must never consume those stale responses.
            for response_queue in resp_qs[: self.num_workers]:
                while True:
                    try:
                        response_queue.get_nowait()
                    except queue.Empty:
                        break
                    except (AttributeError, OSError):
                        break
            self._pool_remote_channel = remote_channel
        self._pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(self.cg_lib_path, remote_channel),
            maxtasksperchild=maxtasks,
        )
        return self

    def __exit__(self, *exc) -> None:
        assert self._pool is not None
        if exc and exc[0] is not None:
            self.request_stop(f"{exc[0].__name__}: context exit")
        if not self._terminated:
            self._pool.close()
        # Overnight SIGTERM previously hung join() for hours while replacement
        # workers spun on an empty slot_queue. Bound teardown and force-kill.
        join_thread = threading.Thread(target=self._pool.join, daemon=True)
        join_thread.start()
        join_thread.join(timeout=45.0)
        if join_thread.is_alive():
            try:
                self._pool.terminate()
            except Exception:
                pass
            join_thread.join(timeout=10.0)
        self._pool = None
        if self._slot_queue is not None:
            try:
                self._slot_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                self._slot_queue.close()
            except Exception:
                pass
            self._slot_queue = None

    @property
    def stopped(self) -> bool:
        return self._terminated

    @property
    def generation(self) -> Optional[int]:
        if isinstance(self._pool_remote_channel, dict):
            return int(self._pool_remote_channel["generation"])
        return None

    def request_stop(self, reason: str = "parent requested stop") -> None:
        """Cancel one pool generation and prevent multiprocessing respawns."""
        if self._terminated:
            return
        self._terminated = True
        self._stop_reason = str(reason)
        if self._stop_event is not None:
            self._stop_event.set()
        remote = self._pool_remote_channel
        if isinstance(remote, dict):
            generation = int(remote.get("generation", -1))
            for control_queue in remote.get("ctrl_qs") or []:
                try:
                    control_queue.put_nowait(
                        {
                            "cmd": "cancel_generation",
                            "generation": generation,
                            "reason": self._stop_reason,
                        }
                    )
                except (AttributeError, OSError, queue.Full):
                    pass
        if self._pool is not None:
            # terminate() changes the pool state before killing children, so
            # its worker-maintenance thread cannot respawn replacements.
            self._pool.terminate()

    def imap_unordered(
        self, fn: Callable[[T], R], jobs: Iterable[T], chunksize: int = 1
    ) -> Iterator[R]:
        assert self._pool is not None, "Use WorkerPool as a context manager."
        if self._terminated:
            raise WorkerPoolStopped(self._stop_reason or "pool is stopped")
        return self._pool.imap_unordered(fn, jobs, chunksize=chunksize)

    def map(self, fn: Callable[[T], R], jobs: Iterable[T], chunksize: int = 1) -> list[R]:
        assert self._pool is not None, "Use WorkerPool as a context manager."
        return self._pool.map(fn, jobs, chunksize=chunksize)
