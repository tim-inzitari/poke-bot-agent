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

import itertools
import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing import util as mp_util
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from . import config

T = TypeVar("T")
R = TypeVar("R")
_POOL_GENERATIONS = itertools.count(1)
_GENERATION_POOL_SCALE = 1_000_000_000
_GENERATION_SLOT_SCALE = 1_000_000


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


def _worker_incarnation_generation(
    pool_generation: int, slot: int, incarnation: int
) -> int:
    """Return a wire-stable id that changes whenever a slot is recycled."""
    return (
        int(pool_generation) * _GENERATION_POOL_SCALE
        + int(slot) * _GENERATION_SLOT_SCALE
        + int(incarnation)
    )


def _pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` still names a live process (POSIX hosts)."""
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _release_remote_slot(
    slot_condition,
    slot_owners,
    slot_ready,
    slot: int,
    owner_pid: int,
) -> None:
    """Release a normal-exit worker's response-queue lease.

    ``multiprocessing.util.Finalize`` runs for an ordinary
    ``maxtasksperchild`` retirement, but not after ``SIGKILL``/``os._exit``.
    Keeping the owner PID in shared memory lets the parent distinguish those
    cases and fail closed instead of losing a queue token forever.
    """
    try:
        with slot_condition:
            slot_i = int(slot)
            if int(slot_owners[slot_i]) == int(owner_pid):
                slot_ready[slot_i] = 0
                slot_owners[slot_i] = 0
                slot_condition.notify_all()
    except Exception:
        # Pool teardown may destroy synchronization primitives after children
        # have already been asked to exit. The parent reaper treats a live-pool
        # stale owner as fatal, so this must never be the sole crash path.
        pass


def _claim_remote_slot(remote_channel: dict) -> tuple[int, int]:
    """Atomically lease one response slot and return ``(slot, incarnation)``."""
    slot_condition = remote_channel["slot_condition"]
    slot_owners = remote_channel["slot_owners"]
    slot_ready = remote_channel["slot_ready"]
    slot_incarnations = remote_channel["slot_incarnations"]
    stop_event = remote_channel.get("stop_event")
    worker_failure_event = remote_channel.get("worker_failure_event")
    pid = os.getpid()
    deadline = time.monotonic() + 30.0

    with slot_condition:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise WorkerPoolStopped("pool stopped before slot acquisition")
            if worker_failure_event is not None and worker_failure_event.is_set():
                raise WorkerPoolStopped("pool worker failure was latched")

            for slot, raw_owner in enumerate(slot_owners):
                if int(raw_owner) != 0:
                    continue
                slot_owners[slot] = int(pid)
                slot_ready[slot] = 0
                incarnation = int(slot_incarnations[slot]) + 1
                slot_incarnations[slot] = incarnation
                return int(slot), incarnation

            # A clean maxtasks retirement clears its owner in Finalize before
            # its replacement starts. An occupied slot whose PID is already
            # dead therefore proves an abnormal exit. Latch it immediately;
            # silently reclaiming here would hide a crash and allow Pool's
            # maintenance thread to spin replacements indefinitely.
            dead_owners = [
                (slot, int(owner))
                for slot, owner in enumerate(slot_owners)
                if int(owner) > 0 and not _pid_is_alive(int(owner))
            ]
            if dead_owners:
                for slot, owner in dead_owners:
                    if int(slot_owners[slot]) == owner:
                        slot_ready[slot] = 0
                        slot_owners[slot] = 0
                if worker_failure_event is not None:
                    worker_failure_event.set()
                slot_condition.notify_all()
                raise WorkerPoolStopped(
                    f"remote worker died without releasing slot(s): {dead_owners}"
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if worker_failure_event is not None:
                    worker_failure_event.set()
                slot_condition.notify_all()
                raise WorkerPoolStopped(
                    "no response slot became available within 30 seconds"
                )
            slot_condition.wait(timeout=min(0.25, remaining))


def _mark_remote_slot_ready(remote_channel: dict, slot: int) -> None:
    """Publish that the worker completed its entire initializer."""
    slot_condition = remote_channel["slot_condition"]
    slot_owners = remote_channel["slot_owners"]
    slot_ready = remote_channel["slot_ready"]
    pid = os.getpid()
    with slot_condition:
        if int(slot_owners[int(slot)]) != int(pid):
            raise WorkerPoolStopped(
                f"worker {pid} lost response slot {int(slot)} during initialization"
            )
        slot_ready[int(slot)] = int(pid)
        slot_condition.notify_all()


def _worker_init_impl(cg_lib_path: Optional[str], remote_channel=None) -> None:
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
    # Acquire the response-slot lease before cg/torch imports. If the OS kills
    # a child during native-library initialization, its shared owner PID then
    # remains visible to the parent reaper; acquiring after those imports let
    # Pool respawn an unowned initializer forever.
    early_slot: Optional[int] = None
    early_incarnation: Optional[int] = None
    if isinstance(remote_channel, dict) and remote_channel.get("slot_condition") is not None:
        early_slot, early_incarnation = _claim_remote_slot(remote_channel)
        owner_pid = os.getpid()
        mp_util.Finalize(
            None,
            _release_remote_slot,
            args=(
                remote_channel["slot_condition"],
                remote_channel["slot_owners"],
                remote_channel["slot_ready"],
                early_slot,
                owner_pid,
            ),
            exitpriority=100,
        )

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
            slot_condition = remote_channel.get("slot_condition")
            slot_counter = remote_channel.get("slot_counter")
            generation = int(remote_channel.get("generation", 0))
            alive_evts = remote_channel.get("alive_evts")
            expected_digest = remote_channel.get("expected_digest")
            expected_version = remote_channel.get("expected_version")
            timeout_s = remote_channel.get("timeout_s")
            stop_event = remote_channel.get("stop_event")
            slot_incarnations = remote_channel.get("slot_incarnations")
        else:
            req_qs, resp_qs, slot_counter = remote_channel
            slot_condition = None
            generation = 0
            alive_evts = None
            expected_digest = None
            expected_version = None
            timeout_s = None
            stop_event = None
            slot_incarnations = None
        if stop_event is not None and stop_event.is_set():
            raise WorkerPoolStopped("pool stopped before worker initialization")
        if slot_condition is not None:
            if early_slot is None or early_incarnation is None:
                raise WorkerPoolStopped("remote slot was not leased before imports")
            slot = int(early_slot)
            incarnation = int(early_incarnation)
            if slot_incarnations is not None:
                generation = _worker_incarnation_generation(
                    generation, slot, incarnation
                )
        else:
            with slot_counter.get_lock():
                slot = int(slot_counter.value)
                slot_counter.value = slot + 1
        if not 0 <= slot < len(resp_qs):
            raise RuntimeError(
                f"remote response slot overflow: slot={slot}, queues={len(resp_qs)}"
            )
        # A timed-out request from the prior slot owner may finish after that
        # owner retires. Drain anything already queued; the per-incarnation
        # generation above rejects any response that arrives later.
        try:
            while True:
                resp_qs[slot].get_nowait()
        except queue.Empty:
            pass
        except (AttributeError, OSError):
            pass
        leaf_devices = None
        gpu0_client_frac = 0.38
        if isinstance(remote_channel, dict):
            leaf_devices = remote_channel.get("leaf_devices")
            raw_frac = remote_channel.get("gpu0_client_frac")
            if raw_frac is not None:
                gpu0_client_frac = float(raw_frac)
        if isinstance(req_qs, (list, tuple)):
            if leaf_devices is not None and len(leaf_devices) == len(req_qs):
                from poke_bot.pure_rl.hardware import sticky_leaf_server_index

                server_idx = sticky_leaf_server_index(
                    slot,
                    list(leaf_devices),
                    gpu0_client_frac=gpu0_client_frac,
                )
            else:
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
            req_qs=(list(req_qs) if isinstance(req_qs, (list, tuple)) else None),
            leaf_devices=(list(leaf_devices) if leaf_devices is not None else None),
            alive_evts=(list(alive_evts) if alive_evts is not None else None),
        )
        if slot_condition is not None:
            _mark_remote_slot_ready(remote_channel, slot)


def _worker_init(cg_lib_path: Optional[str], remote_channel=None) -> None:
    """Pool initializer with cross-process failure publication.

    ``multiprocessing.Pool`` otherwise replaces an initializer-crashing child
    forever. Publishing the first failure lets the parent terminate the pool
    before that maintenance loop becomes a spawn storm.
    """
    attempts = (
        remote_channel.get("init_attempts")
        if isinstance(remote_channel, dict)
        else None
    )
    successes = (
        remote_channel.get("init_successes")
        if isinstance(remote_channel, dict)
        else None
    )
    failures = (
        remote_channel.get("init_failures")
        if isinstance(remote_channel, dict)
        else None
    )
    failure_event = (
        remote_channel.get("worker_failure_event")
        if isinstance(remote_channel, dict)
        else None
    )
    condition = (
        remote_channel.get("slot_condition")
        if isinstance(remote_channel, dict)
        else None
    )
    stop_event = (
        remote_channel.get("stop_event")
        if isinstance(remote_channel, dict)
        else None
    )
    if attempts is not None:
        with attempts.get_lock():
            attempts.value += 1
    try:
        _worker_init_impl(cg_lib_path, remote_channel)
    except BaseException:
        intentional_stop = stop_event is not None and stop_event.is_set()
        already_failed = failure_event is not None and failure_event.is_set()
        if not intentional_stop and not already_failed:
            if failures is not None:
                with failures.get_lock():
                    failures.value += 1
            if failure_event is not None:
                failure_event.set()
            if condition is not None:
                with condition:
                    condition.notify_all()
        raise
    else:
        if successes is not None:
            with successes.get_lock():
                successes.value += 1


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
        capacity_recovery_grace_s: Optional[float] = None,
    ) -> None:
        # Default worker count comes from the central hardware profile (all 32
        # CPU threads on the dedicated box); recycle_games bounds libcg's slow
        # per-process leak so long unattended runs have no unbounded RAM growth.
        self.num_workers = num_workers or config.HARDWARE.sim_workers
        self.recycle_games = recycle_games or config.HARDWARE.worker_recycle_games
        self.cg_lib_path = cg_lib_path or os.environ.get("CG_LIB_PATH")
        # Pool briefly retires children at ``maxtasksperchild`` before their
        # replacements are alive.  That is expected, but a capacity hole that
        # never heals can strand an accepted imap task forever because stdlib
        # Pool does not requeue a result lost with its child.  Keep the grace
        # comfortably longer than a normal spawn/import cycle while giving the
        # parent a deterministic escape from that otherwise infinite wait.
        self.capacity_recovery_grace_s = max(
            0.0,
            float(
                capacity_recovery_grace_s
                if capacity_recovery_grace_s is not None
                else os.environ.get(
                    "POKEBOT_WORKER_CAPACITY_RECOVERY_GRACE_S", "60"
                )
            ),
        )
        # Remote GPU leaf-eval channel = (req_q, resp_qs, slot_counter). Remote
        # worker services keep this pool alive across many waves, so recycling
        # must stay enabled there too; bounded slot leases make replacement safe.
        self.remote_channel = remote_channel
        self._pool: Optional[mp.pool.Pool] = None
        self._pool_remote_channel = None
        self._stop_event = None
        self._slot_condition = None
        self._slot_owners = None
        self._slot_ready = None
        self._worker_failure_event = None
        self._init_attempts = None
        self._init_successes = None
        self._init_failures = None
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._terminated = False
        self._stop_reason: Optional[str] = None

    def __enter__(self) -> "WorkerPool":
        ctx = mp.get_context("spawn")
        maxtasks = self.recycle_games
        remote_channel = self.remote_channel
        if isinstance(remote_channel, dict):
            resp_qs = list(remote_channel.get("resp_qs") or [])
            if len(resp_qs) < self.num_workers:
                raise ValueError(
                    "remote response queues must cover every active worker: "
                    f"queues={len(resp_qs)} workers={self.num_workers}"
                )
            # Every pool generation gets fresh, parent-observable response-slot
            # leases. A shared owner PID survives abrupt child death, unlike a
            # queue token that only Finalize can return. ``slot_ready`` is set
            # only after the complete child initializer succeeds.
            slot_condition = ctx.Condition(ctx.RLock())
            slot_owners = ctx.Array(
                "q", [0 for _ in range(self.num_workers)], lock=False
            )
            slot_ready = ctx.Array(
                "q", [0 for _ in range(self.num_workers)], lock=False
            )
            self._stop_event = ctx.Event()
            slot_incarnations = ctx.Array(
                "q", [0 for _ in range(self.num_workers)], lock=False
            )
            worker_failure_event = ctx.Event()
            init_attempts = ctx.Value("q", 0)
            init_successes = ctx.Value("q", 0)
            init_failures = ctx.Value("q", 0)
            self._slot_condition = slot_condition
            self._slot_owners = slot_owners
            self._slot_ready = slot_ready
            self._worker_failure_event = worker_failure_event
            self._init_attempts = init_attempts
            self._init_successes = init_successes
            self._init_failures = init_failures
            generation = (
                int(remote_channel.get("generation", 0)) * 1_000_000
                + next(_POOL_GENERATIONS)
            )
            remote_channel = {
                **remote_channel,
                "generation": generation,
                "slot_condition": slot_condition,
                "slot_owners": slot_owners,
                "slot_ready": slot_ready,
                "slot_incarnations": slot_incarnations,
                "stop_event": self._stop_event,
                "worker_failure_event": worker_failure_event,
                "init_attempts": init_attempts,
                "init_successes": init_successes,
                "init_failures": init_failures,
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
        try:
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(self.cg_lib_path, remote_channel),
                maxtasksperchild=maxtasks,
            )
            if isinstance(remote_channel, dict):
                self._start_worker_monitor()
                self.wait_until_ready()
        except BaseException as exc:
            if self._pool is not None:
                self.request_stop(f"worker pool startup failed: {exc}")
                self._pool.join()
                self._pool = None
            self._finish_worker_monitor()
            self._clear_remote_runtime()
            raise
        return self

    def __exit__(self, *exc) -> None:
        assert self._pool is not None
        if exc and exc[0] is not None:
            self.request_stop(f"{exc[0].__name__}: context exit")
        if not self._terminated:
            self._pool.close()
        self._pool.join()
        self._pool = None
        self._finish_worker_monitor()
        self._clear_remote_runtime()

    def _clear_remote_runtime(self) -> None:
        self._pool_remote_channel = None
        self._stop_event = None
        self._slot_condition = None
        self._slot_owners = None
        self._slot_ready = None
        self._worker_failure_event = None
        self._init_attempts = None
        self._init_successes = None
        self._init_failures = None

    def _start_worker_monitor(self) -> None:
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_remote_workers,
            name="worker-pool-slot-reaper",
            daemon=True,
        )
        self._monitor_thread.start()

    def _finish_worker_monitor(self) -> None:
        self._monitor_stop.set()
        condition = self._slot_condition
        if condition is not None:
            try:
                with condition:
                    condition.notify_all()
            except Exception:
                pass
        thread = self._monitor_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._monitor_thread = None

    def _monitor_remote_workers(self) -> None:
        """Latch abrupt exits/initializer failures before Pool can churn."""
        while not self._monitor_stop.wait(0.05):
            failure_event = self._worker_failure_event
            if failure_event is not None and failure_event.is_set():
                self.request_stop(
                    "remote worker initializer failed or slot owner died"
                )
                return

            condition = self._slot_condition
            owners = self._slot_owners
            ready = self._slot_ready
            if condition is None or owners is None or ready is None:
                return
            pool_pids = set(self.live_worker_pids)
            stale: list[tuple[int, int]] = []
            with condition:
                for slot, raw_owner in enumerate(owners):
                    owner = int(raw_owner)
                    if owner > 0 and (
                        owner not in pool_pids or not _pid_is_alive(owner)
                    ):
                        stale.append((slot, owner))
                if stale:
                    for slot, owner in stale:
                        if int(owners[slot]) == owner:
                            ready[slot] = 0
                            owners[slot] = 0
                    if failure_event is not None:
                        failure_event.set()
                    condition.notify_all()
            if stale:
                self.request_stop(
                    f"remote worker(s) died without clean slot release: {stale}"
                )
                return

    def wait_until_ready(self, timeout_s: Optional[float] = None) -> None:
        """Wait for every remote worker to finish its complete initializer."""
        if self._slot_condition is None:
            return
        timeout = float(
            timeout_s
            if timeout_s is not None
            else os.environ.get("POKEBOT_WORKER_INIT_TIMEOUT_S", "120")
        )
        deadline = time.monotonic() + max(0.1, timeout)
        condition = self._slot_condition
        with condition:
            while True:
                if self._terminated:
                    raise WorkerPoolStopped(
                        self._stop_reason or "worker pool stopped during startup"
                    )
                failure_event = self._worker_failure_event
                if failure_event is not None and failure_event.is_set():
                    raise WorkerPoolStopped(
                        "remote worker initializer failed during startup"
                    )
                owners = [int(pid) for pid in (self._slot_owners or [])]
                ready = [int(pid) for pid in (self._slot_ready or [])]
                if (
                    len(owners) == self.num_workers
                    and all(pid > 0 for pid in owners)
                    and ready == owners
                ):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerPoolStopped(
                        "timed out waiting for remote workers to initialize: "
                        f"ready={sum(pid > 0 for pid in ready)}/{self.num_workers} "
                        f"attempts={self.initializer_attempts} "
                        f"failures={self.initializer_failures}"
                    )
                condition.wait(timeout=min(0.25, remaining))

    @property
    def stopped(self) -> bool:
        return self._terminated

    @property
    def generation(self) -> Optional[int]:
        if isinstance(self._pool_remote_channel, dict):
            return int(self._pool_remote_channel["generation"])
        return None

    @property
    def live_worker_pids(self) -> tuple[int, ...]:
        """Return the currently alive pool children for service health checks."""
        if self._pool is None:
            return ()
        processes = getattr(self._pool, "_pool", ())
        return tuple(
            int(proc.pid)
            for proc in processes
            if proc.pid is not None and proc.is_alive()
        )

    @property
    def ready_worker_pids(self) -> tuple[int, ...]:
        """Return children that own a slot and completed initialization."""
        condition = self._slot_condition
        owners = self._slot_owners
        ready = self._slot_ready
        if condition is None or owners is None or ready is None:
            return self.live_worker_pids
        pool_pids = set(self.live_worker_pids)
        with condition:
            return tuple(
                int(owner)
                for owner, ready_pid in zip(owners, ready)
                if int(owner) > 0
                and int(ready_pid) == int(owner)
                and int(owner) in pool_pids
                and _pid_is_alive(int(owner))
            )

    @property
    def initializer_attempts(self) -> int:
        value = self._init_attempts
        if value is None:
            return 0
        with value.get_lock():
            return int(value.value)

    @property
    def initializer_failures(self) -> int:
        value = self._init_failures
        if value is None:
            return 0
        with value.get_lock():
            return int(value.value)

    @property
    def worker_capacity_healthy(self) -> bool:
        """Whether all configured children are initialized and non-failed."""
        failure_event = self._worker_failure_event
        return bool(
            not self._terminated
            and (failure_event is None or not failure_event.is_set())
            and len(self.ready_worker_pids) == self.num_workers
        )

    def request_stop(self, reason: str = "parent requested stop") -> None:
        """Cancel one pool generation and prevent multiprocessing respawns."""
        if self._terminated:
            return
        self._terminated = True
        self._stop_reason = str(reason)
        self._monitor_stop.set()
        if self._stop_event is not None:
            self._stop_event.set()
        condition = self._slot_condition
        if condition is not None:
            try:
                with condition:
                    condition.notify_all()
            except Exception:
                pass
        remote = self._pool_remote_channel
        if isinstance(remote, dict):
            pool_generation = int(remote.get("generation", -1))
            generations: list[int] = []
            slot_incarnations = remote.get("slot_incarnations")
            if slot_incarnations is not None:
                if condition is not None:
                    with condition:
                        incarnations = [int(value) for value in slot_incarnations]
                else:
                    incarnations = [int(value) for value in slot_incarnations]
                generations.extend(
                    _worker_incarnation_generation(
                        pool_generation, slot, incarnation
                    )
                    for slot, incarnation in enumerate(incarnations)
                    if incarnation > 0
                )
            if not generations:
                generations.append(pool_generation)
            for control_queue in remote.get("ctrl_qs") or []:
                for generation in generations:
                    try:
                        control_queue.put_nowait(
                            {
                                "cmd": "cancel_generation",
                                "generation": generation,
                                "reason": self._stop_reason,
                            }
                        )
                    except (AttributeError, OSError, queue.Full):
                        break
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
        result_iter = self._pool.imap_unordered(fn, jobs, chunksize=chunksize)

        def _checked_results() -> Iterator[R]:
            # ``multiprocessing.pool.IMapUnorderedIterator.__next__`` has no
            # timeout.  If a child dies after accepting a task, Pool may spawn
            # a replacement while that lost result's iterator waits forever.
            # The remote-slot monitor already latches/terminates that failure;
            # poll so its stop reaches the scheduler emitter within one second.
            capacity_loss_since: Optional[float] = None
            while True:
                try:
                    result = result_iter.next(timeout=1.0)
                    # Notice a recovery even when queued results keep arriving
                    # without a timeout.  Otherwise a later, unrelated recycle
                    # could inherit an obsolete capacity-loss deadline.
                    if len(self.live_worker_pids) >= self.num_workers:
                        capacity_loss_since = None
                    elif capacity_loss_since is None:
                        capacity_loss_since = time.monotonic()
                    yield result
                except mp.TimeoutError:
                    if self._terminated:
                        raise WorkerPoolStopped(
                            self._stop_reason
                            or "pool stopped while waiting for a worker result"
                        )
                    live_workers = len(self.live_worker_pids)
                    if live_workers >= self.num_workers:
                        capacity_loss_since = None
                        continue
                    now = time.monotonic()
                    if capacity_loss_since is None:
                        capacity_loss_since = now
                    unhealthy_for = max(0.0, now - capacity_loss_since)
                    if unhealthy_for >= self.capacity_recovery_grace_s:
                        reason = (
                            "sim worker capacity did not recover within grace: "
                            f"live={live_workers} expected={self.num_workers} "
                            f"unhealthy_for={unhealthy_for:.1f}s "
                            f"grace={self.capacity_recovery_grace_s:.1f}s"
                        )
                        self.request_stop(reason)
                        raise WorkerPoolStopped(reason)
                    continue
                except StopIteration:
                    return

        return _checked_results()

    def apply(self, fn: Callable[[T], R], job: T) -> R:
        """Run one pickle-safe job in a pool child and return its result.

        Remote TCP handlers and additive fallback threads must not invoke game
        functions in their own thread (the workers use SIGALRM and process-local
        simulator state).  Keep this small wrapper alongside ``map``/``imap`` so
        every checkout uses the same safe multiprocessing path.
        """
        assert self._pool is not None, "Use WorkerPool as a context manager."
        if self._terminated:
            raise WorkerPoolStopped(self._stop_reason or "pool is stopped")
        return self._pool.apply(fn, (job,))

    def map(self, fn: Callable[[T], R], jobs: Iterable[T], chunksize: int = 1) -> list[R]:
        assert self._pool is not None, "Use WorkerPool as a context manager."
        if self._terminated:
            raise WorkerPoolStopped(self._stop_reason or "pool is stopped")
        return self._pool.map(fn, jobs, chunksize=chunksize)
