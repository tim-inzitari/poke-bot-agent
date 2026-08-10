"""Stock-libcg eight-trajectory microbatch primitive for the r222 experiment.

This module intentionally owns *neither* a policy/model nor a mutable MCTS
tree.  It is the small transport seam required to turn eight selected paths
from **one** shared tree into eight isolated stock ``SearchBegin`` worlds,
collect their terminal/nonterminal leaf observations into one frozen-model
microbatch, and only then hand the rows back to the shared-tree owner for
backup.

Why this is a separate module
-----------------------------

``BeliefMCTS`` currently owns selection, native expansion, evaluation and
backup in one sequential routine.  Replacing that routine in place would make
it very easy to accidentally claim that eight independent forests are one
tree.  This file instead exposes a transaction-shaped primitive:

* the caller performs shared-tree selection (including virtual loss) on its
  coordinator thread and supplies exactly eight trajectories;
* each lane owns one ``cg.sim.lib.AgentStart()`` raw handle on its persistent
  thread and touches no other lane's handle;
* the coordinator gathers all eight completed leaf observations, sends one
  request to a queue-owned frozen leaf broker, and invokes shared-tree backup
  serially only after the complete batch validates;
* any lane/model/backup failure rolls back completed backups, informs the
  caller to remove virtual loss, and returns no partial action statistic.

It calls only the stock Search ABI through :class:`cg_env.NativeSearchLane`:
``AgentStart``, ``SearchBegin``, ``SearchStep``, ``SearchRelease`` and
``SearchEnd``.  It deliberately contains no B77, seeded start, batch-engine,
RTP, training, Kaggle, or game-battle transport path.

The actual r222 integration belongs at the narrow seam where BeliefMCTS has
already selected a speculative path but has not yet evaluated/updated its
shared tree.  Until that integration has transactional selection/rollback
coverage, this module is a preflight/test component, not action authority.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from . import cg_env


R222_STOCK_SHARED_TREE_BATCH_SCHEMA = "poke_bot.r222_stock_shared_tree_batch/v1"
R222_STOCK_LIBCG_SHA256 = (
    "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
)
R222_STOCK_LIBCG_SIZE_BYTES = 1_342_400
R222_STOCK_LANE_COUNT = 8
R222_REQUIRED_STOCK_EXPORTS = (
    "AgentStart",
    "BattleStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
)


class R222StockBatchError(RuntimeError):
    """The stock eight-lane batch has no safe shared-tree result."""


class R222StockBatchDeadline(R222StockBatchError, TimeoutError):
    """All lane work was joined, but its common decision deadline elapsed."""


class R222StockBatchIntegrityError(R222StockBatchError):
    """A lane, root identity, leaf batch, or transactional backup was invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class R222StockLibraryReceipt:
    """Measured identity/capability facts for the exact archived stock ABI."""

    path: str
    sha256: str
    size_bytes: int
    required_exports: tuple[str, ...]
    custom_engine_symbols_checked: tuple[str, ...]
    custom_engine_symbols_present: tuple[str, ...]
    exact_stock_r195: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def attest_stock_r195_library(
    path: str | os.PathLike[str],
    *,
    library: Any | None = None,
    expected_sha256: str = R222_STOCK_LIBCG_SHA256,
    expected_size_bytes: int = R222_STOCK_LIBCG_SIZE_BYTES,
) -> R222StockLibraryReceipt:
    """Fail closed unless ``path`` is the exact stock r195 ``libcg.so``.

    The content digest is the primary proof.  The export check gives a useful
    failure reason before a threaded search preflight starts; it is not used to
    bless a different binary.  We only *inspect* custom symbols and never call
    them.  Their presence is itself a rejection because the r222 contract
    allows neither seeded nor custom/batched engine transport.
    """

    candidate = Path(path).expanduser()
    try:
        stat = candidate.lstat()
    except OSError as exc:
        raise R222StockBatchIntegrityError(
            f"cannot stat stock libcg: {candidate}"
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise R222StockBatchIntegrityError(
            "stock libcg must be a physical regular file, not a symlink"
        )
    resolved = candidate.resolve()
    observed_size = int(stat.st_size)
    observed_sha = _sha256_file(resolved)
    if observed_size != int(expected_size_bytes):
        raise R222StockBatchIntegrityError(
            f"stock libcg size mismatch: {observed_size} != {expected_size_bytes}"
        )
    if observed_sha != str(expected_sha256):
        raise R222StockBatchIntegrityError(
            "stock libcg sha256 mismatch; refusing non-r195 or custom engine"
        )
    if library is None:
        # ``ctypes.CDLL`` is kept local so importing this preflight module does
        # not load an engine or initialize global card tables.
        import ctypes

        library = ctypes.CDLL(str(resolved))
    missing = [name for name in R222_REQUIRED_STOCK_EXPORTS if not hasattr(library, name)]
    if missing:
        raise R222StockBatchIntegrityError(
            "stock libcg misses required Search ABI exports: " + ", ".join(missing)
        )
    custom_symbols = (
        "BattleStartSeeded",
        "BattleStartBatchSeeded",
        "BatchAbiVersion",
        "GetBattleDataBatch",
        "StepBatch",
        "GetHiddenSnapshot",
        "RtpPairingBattleStartSeededOut",
    )
    present = tuple(name for name in custom_symbols if hasattr(library, name))
    if present:
        raise R222StockBatchIntegrityError(
            "exact-stock digest unexpectedly exposes forbidden custom symbols: "
            + ", ".join(present)
        )
    return R222StockLibraryReceipt(
        path=str(resolved),
        sha256=observed_sha,
        size_bytes=observed_size,
        required_exports=R222_REQUIRED_STOCK_EXPORTS,
        custom_engine_symbols_checked=custom_symbols,
        custom_engine_symbols_present=present,
        exact_stock_r195=True,
    )


def attest_loaded_stock_r195_library(
    library: Any,
    *,
    expected_path: str | os.PathLike[str],
) -> R222StockLibraryReceipt:
    """Prove the already-loaded ``cg.sim.lib`` is the package-local stock file.

    Opening a second ``ctypes`` handle merely proves that a good file exists;
    this companion check binds the raw library actually used by ``AgentStart``
    to the same physical r195 path before any lane is created.
    """

    expected = Path(expected_path).expanduser().resolve()
    loaded_name = getattr(library, "_name", None)
    if not isinstance(loaded_name, (str, os.PathLike)):
        raise R222StockBatchIntegrityError("loaded cg.sim.lib has no inspectable library path")
    loaded = Path(loaded_name).expanduser().resolve()
    if loaded != expected:
        raise R222StockBatchIntegrityError(
            f"cg.sim.lib path mismatch: loaded {loaded}, expected {expected}"
        )
    return attest_stock_r195_library(expected, library=library)


def canonical_observation_fingerprint(observation: Any) -> str:
    """Stable diagnostic fingerprint for a search observation.

    This is intentionally a transport/parity fingerprint, not an information
    set key used by MCTS.  A caller may provide its stricter policy-visible
    root fingerprint in :class:`R222SharedTreeBatchRequest`; this helper simply
    verifies that the eight search lanes were supplied the same raw root.
    """

    def normalize(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return normalize(dataclasses.asdict(value))
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # cg API objects in the official Python binding are dataclasses.  Do
        # not silently stringify an unknown object because that can hide a
        # lane mismatch behind a process-address repr.
        raise TypeError(f"cannot canonically fingerprint {type(value).__name__}")

    encoded = json.dumps(
        normalize(observation), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _explicit_chance_observation(observation: Any) -> bool:
    """Return whether stock manual-coin search exposed a chance prompt.

    The official runtime identifies its manual coin prompt as SelectContext 46.
    This narrow transport knows no force/enumeration ABI, so it never advances
    such a state.  The caller must finish the trajectory at this *pre-random*
    leaf and let the frozen model evaluate it, exactly as r221/r222 require.
    """

    if not isinstance(observation, Mapping):
        if dataclasses.is_dataclass(observation):
            observation = dataclasses.asdict(observation)
        else:
            return False
    select = observation.get("select")
    return isinstance(select, Mapping) and int(select.get("context", -1)) == 46


def _max_overlap(executions: Sequence["R222LaneExecution"]) -> int:
    """Maximum simultaneous private-lane intervals from recorded timestamps."""

    events: list[tuple[float, int]] = []
    for row in executions:
        events.append((float(row.task_started_monotonic), 1))
        events.append((float(row.cleanup_completed_monotonic), -1))
    active = maximum = 0
    # Starts precede ends at an equal timestamp to avoid undercounting a
    # handoff that was concurrent at the measurement precision.
    for _when, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


@dataclass(frozen=True)
class R222StockTrajectory:
    """One selected but not-yet-evaluated path from the shared tree.

    ``tree_token`` is opaque to this transport.  It is returned to the caller
    during backup and is normally the shared-tree edge/path transaction token
    containing virtual-loss bookkeeping.  Search inputs may differ between
    root-sampled particles, while ``root_observation`` must be the same actual
    decision observation in all eight lanes.
    """

    lane_id: int
    root_observation: Mapping[str, Any]
    search_inputs: Mapping[str, Sequence[int]]
    action_path: tuple[tuple[int, ...], ...]
    tree_token: Any


@dataclass(frozen=True)
class R222SharedTreeBatchRequest:
    """Exactly eight selected trajectories for one real MCTS decision."""

    decision_fingerprint: str
    complete_root_legal_actions: tuple[tuple[int, ...], ...]
    direct_policy_action: tuple[int, ...]
    trajectories: tuple[R222StockTrajectory, ...]

    def validate(self, *, lane_count: int = R222_STOCK_LANE_COUNT) -> None:
        if not isinstance(self.decision_fingerprint, str) or not self.decision_fingerprint:
            raise R222StockBatchIntegrityError("shared-tree decision fingerprint is missing")
        canonical = tuple(tuple(int(item) for item in action) for action in self.complete_root_legal_actions)
        if not canonical or len(set(canonical)) != len(canonical):
            raise R222StockBatchIntegrityError("root legal action order is empty or duplicated")
        if tuple(int(item) for item in self.direct_policy_action) not in canonical:
            raise R222StockBatchIntegrityError("frozen direct-policy action is not root legal")
        if len(self.trajectories) != int(lane_count):
            raise R222StockBatchIntegrityError(
                f"shared-tree batch requires exactly {lane_count} trajectories"
            )
        lane_ids = [int(row.lane_id) for row in self.trajectories]
        if sorted(lane_ids) != list(range(int(lane_count))):
            raise R222StockBatchIntegrityError(
                "trajectory lane ids must be exactly 0 through seven"
            )
        roots = [canonical_observation_fingerprint(row.root_observation) for row in self.trajectories]
        if len(set(roots)) != 1:
            raise R222StockBatchIntegrityError(
                "lanes were not given the same actual root observation"
            )
        for row in self.trajectories:
            if not isinstance(row.search_inputs, Mapping):
                raise R222StockBatchIntegrityError("trajectory search inputs are malformed")
            for action in row.action_path:
                values = tuple(int(item) for item in action)
                if len(values) != len(action):
                    raise R222StockBatchIntegrityError("trajectory action path is malformed")


@dataclass(frozen=True)
class R222LaneExecution:
    """A private stock-search outcome, returned only after cleanup."""

    lane_id: int
    root_search_id: int
    final_search_id: int
    final_observation: Any
    root_observation_fingerprint: str
    action_path: tuple[tuple[int, ...], ...]
    search_begin_calls: int
    search_step_calls: int
    search_release_calls: int
    search_end_calls: int
    owner_thread_id: int
    elapsed_seconds: float
    task_started_monotonic: float
    search_begin_started_monotonic: float
    search_begin_completed_monotonic: float
    first_search_step_started_monotonic: float | None
    first_search_step_completed_monotonic: float | None
    cleanup_completed_monotonic: float


@dataclass(frozen=True)
class R222StockMicrobatchReceipt:
    """Complete, truthful accounting for one eight-trajectory transaction."""

    schema: str
    requested_lane_count: int
    completed_lane_count: int
    lane_ids: tuple[int, ...]
    unique_raw_handle_count: int
    root_observation_fingerprint: str
    decision_fingerprint: str
    complete_root_legal_actions: tuple[tuple[int, ...], ...]
    direct_policy_action: tuple[int, ...]
    search_begin_calls: int
    search_step_calls: int
    search_release_calls: int
    search_end_calls: int
    leaf_request_count: int
    leaf_batch_rows: int
    frozen_model_forwards: int
    completed_backed_simulations: int
    elapsed_seconds: float
    backed_simulations_per_second: float
    all_lanes_cleaned_before_return: bool
    partial_lane_statistics_used: bool
    lane_topology: tuple[tuple[int, int, int | str], ...]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _LaneTask:
    trajectory: R222StockTrajectory
    phase_barrier: threading.Barrier
    cancel: threading.Event
    deadline: float
    max_action_depth: int
    done: threading.Event = field(default_factory=threading.Event)
    passed_begin_barrier: bool = False
    result: R222LaneExecution | None = None
    error: BaseException | None = None


class _StockLaneWorker:
    """One persistent thread and one native handle; no model/tree state."""

    def __init__(
        self,
        lane_id: int,
        backend_factory: Callable[[int], cg_env.SearchBackend],
    ) -> None:
        self.lane_id = int(lane_id)
        self._backend_factory = backend_factory
        self._queue: queue.Queue[_LaneTask | None] = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.initialization_error: BaseException | None = None
        self.owner_thread_id: int | None = None
        self.handle_identity: int | str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"r222-stock-search-lane-{self.lane_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def topology(self) -> tuple[int, int, int | str]:
        if self.owner_thread_id is None or self.handle_identity is None:
            raise R222StockBatchIntegrityError("stock lane has no initialized native handle")
        return self.lane_id, self.owner_thread_id, self.handle_identity

    def submit(self, task: _LaneTask) -> None:
        self._queue.put_nowait(task)

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    @staticmethod
    def _wait_phase(task: _LaneTask) -> None:
        if task.cancel.is_set():
            raise R222StockBatchDeadline("batch was cancelled before a stock search phase")
        remaining = task.deadline - time.monotonic()
        if remaining <= 0.0:
            raise R222StockBatchDeadline("stock search phase reached its deadline")
        try:
            task.phase_barrier.wait(timeout=remaining)
        except threading.BrokenBarrierError as exc:
            raise R222StockBatchIntegrityError("one stock search lane failed its phase") from exc

    def _execute(self, backend: cg_env.SearchBackend, task: _LaneTask) -> R222LaneExecution:
        started = time.monotonic()
        live_ids: list[int] = []
        begin_calls = step_calls = release_calls = end_calls = 0
        root_id = final_id = -1
        final_observation: Any = None
        cleanup_error: BaseException | None = None
        operation_error: BaseException | None = None
        begin_started = begin_completed = started
        first_step_started: float | None = None
        first_step_completed: float | None = None
        trajectory = task.trajectory
        try:
            begin_started = time.monotonic()
            root = backend.search_begin(
                dict(trajectory.root_observation),
                {key: list(value) for key, value in trajectory.search_inputs.items()},
                # Never let stock libcg privately resolve an unobserved coin.
                # The only exact-force ABI allowed by r222 would be a separately
                # attested capability; this stock transport intentionally has
                # none, so all exposed chance is a pre-random leaf boundary.
                manual_coin=True,
            )
            begin_completed = time.monotonic()
            begin_calls += 1
            root_id = int(root.searchId)
            final_id = root_id
            live_ids.append(root_id)
            final_observation = root.observation
            # Every lane has begun before any can make its first SearchStep.
            self._wait_phase(task)
            task.passed_begin_barrier = True
            # Paths in a genuine MCTS batch naturally reach different frontier
            # depths.  After the begin barrier they advance independently; the
            # only batch boundary is the all-eight leaf gather below.  Padding
            # a short trajectory with fake SearchStep work would change its
            # simulator history and is therefore forbidden.
            for action in trajectory.action_path:
                if task.cancel.is_set():
                    raise R222StockBatchDeadline("stock batch cancelled during SearchStep")
                if _explicit_chance_observation(final_observation):
                    raise R222StockBatchIntegrityError(
                        "stock shared-tree lane tried to advance an unforceable manual chance prompt"
                    )
                remaining = task.deadline - time.monotonic()
                if remaining <= 0.0:
                    raise R222StockBatchDeadline("stock SearchStep reached deadline")
                if first_step_started is None:
                    first_step_started = time.monotonic()
                child = backend.search_step(final_id, list(action))
                if first_step_completed is None:
                    first_step_completed = time.monotonic()
                step_calls += 1
                final_id = int(child.searchId)
                live_ids.append(final_id)
                final_observation = child.observation
        except BaseException as exc:
            operation_error = exc
        finally:
            # Release descendants first, then the root, while staying on the
            # raw handle's owning thread.  SearchEnd runs even after a failed
            # begin/step; an official backend that rejects that lifecycle is a
            # failed capability preflight rather than a reason to leak work.
            for search_id in reversed(live_ids):
                try:
                    backend.search_release(search_id)
                    release_calls += 1
                except BaseException as exc:  # keep attempting remaining cleanup
                    cleanup_error = cleanup_error or exc
            try:
                backend.search_end()
                end_calls += 1
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise R222StockBatchIntegrityError(
                    f"lane {self.lane_id} stock search cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                ) from cleanup_error
        if operation_error is not None:
            raise operation_error
        if root_id < 0 or final_id < 0 or final_observation is None:
            raise R222StockBatchIntegrityError("stock lane completed without a search state")
        cleanup_completed = time.monotonic()
        return R222LaneExecution(
            lane_id=self.lane_id,
            root_search_id=root_id,
            final_search_id=final_id,
            final_observation=final_observation,
            root_observation_fingerprint=canonical_observation_fingerprint(
                trajectory.root_observation
            ),
            action_path=tuple(tuple(int(item) for item in action) for action in trajectory.action_path),
            search_begin_calls=begin_calls,
            search_step_calls=step_calls,
            search_release_calls=release_calls,
            search_end_calls=end_calls,
            owner_thread_id=threading.get_ident(),
            elapsed_seconds=max(0.0, cleanup_completed - started),
            task_started_monotonic=started,
            search_begin_started_monotonic=begin_started,
            search_begin_completed_monotonic=begin_completed,
            first_search_step_started_monotonic=first_step_started,
            first_search_step_completed_monotonic=first_step_completed,
            cleanup_completed_monotonic=cleanup_completed,
        )

    def _run(self) -> None:
        try:
            backend = self._backend_factory(self.lane_id)
            self.owner_thread_id = threading.get_ident()
            self.handle_identity = getattr(backend, "handle_identity", None)
            if self.handle_identity is None:
                raise R222StockBatchIntegrityError(
                    "stock lane backend does not expose its raw handle identity"
                )
        except BaseException as exc:
            self.initialization_error = exc
            self.ready.set()
            return
        self.ready.set()
        while True:
            task = self._queue.get()
            if task is None:
                return
            try:
                task.result = self._execute(backend, task)
            except BaseException as exc:
                task.error = exc
                task.cancel.set()
                # A failure before/inside the common SearchBegin barrier must
                # wake peers.  Once a lane has returned through that barrier,
                # aborting it can race a peer still leaving ``wait()`` and
                # suppress that peer's required finally/SearchEnd cleanup.
                if not task.passed_begin_barrier:
                    try:
                        task.phase_barrier.abort()
                    except Exception:
                        pass
            finally:
                task.done.set()


class R222StockSearchLanePool:
    """Persistent exact-eight stock Search ABI workers for one game process.

    The pool can be constructed with one lane by unit/parity probes, but the
    r222 shared-tree coordinator below always requires exactly eight.  It does
    not create a game/BattleStart or use a seeded/custom engine; callers pass a
    current stock-engine observation to every trajectory.
    """

    def __init__(
        self,
        backend_factory: Callable[[int], cg_env.SearchBackend] | None = None,
        *,
        lane_count: int = R222_STOCK_LANE_COUNT,
    ) -> None:
        if int(lane_count) < 1:
            raise ValueError("stock search lane count must be positive")
        self.lane_count = int(lane_count)
        self._closed = False
        self._run_lock = threading.Lock()
        if backend_factory is None:
            # Finish the process-global cg metadata/card initialization before
            # any worker invokes AgentStart.  This is the only supported
            # stock-libcg default path; injected factories are unit/preflight
            # harnesses that own their own initialization discipline.
            cg_env.prewarm_native_search_runtime()
            factory = lambda lane_id: cg_env.NativeSearchLane(lane_id)
        else:
            factory = backend_factory
        self._workers: list[_StockLaneWorker] = []
        # Import/global init is intentionally completed before this constructor
        # is called by the physical preflight.  Worker initialization is
        # serial to avoid treating AgentStart as a concurrency benchmark.
        for lane_id in range(self.lane_count):
            worker = _StockLaneWorker(lane_id, factory)
            worker.ready.wait()
            if worker.initialization_error is not None:
                for prior in self._workers:
                    prior.close()
                raise R222StockBatchIntegrityError(
                    f"lane {lane_id} AgentStart failed: "
                    f"{type(worker.initialization_error).__name__}: {worker.initialization_error}"
                ) from worker.initialization_error
            self._workers.append(worker)
        handles = [row.topology[2] for row in self._workers]
        if len(set(handles)) != self.lane_count:
            self.close()
            raise R222StockBatchIntegrityError("stock AgentStart handles are not unique")

    @property
    def lane_topology(self) -> tuple[tuple[int, int, int | str], ...]:
        return tuple(worker.topology for worker in self._workers)

    def execute(
        self,
        trajectories: Sequence[R222StockTrajectory],
        *,
        deadline_monotonic: float,
    ) -> tuple[R222LaneExecution, ...]:
        """Run all selected paths in interleaved SearchBegin/SearchStep waves.

        A failed/late lane aborts the phase barrier, waits for every worker to
        finish cleanup, then raises.  Thus a caller never receives a real-game
        decision while stock search work is still running in the background.
        """

        if self._closed:
            raise R222StockBatchIntegrityError("stock search lane pool is closed")
        if len(trajectories) != self.lane_count:
            raise R222StockBatchIntegrityError(
                f"lane pool expected {self.lane_count} trajectories"
            )
        ordered = tuple(sorted(trajectories, key=lambda row: int(row.lane_id)))
        if [row.lane_id for row in ordered] != list(range(self.lane_count)):
            raise R222StockBatchIntegrityError("trajectory ids do not match lane pool")
        with self._run_lock:
            barrier = threading.Barrier(self.lane_count)
            cancel = threading.Event()
            tasks = [
                _LaneTask(
                    trajectory=row,
                    phase_barrier=barrier,
                    cancel=cancel,
                    deadline=float(deadline_monotonic),
                    max_action_depth=max((len(item.action_path) for item in ordered), default=0),
                )
                for row in ordered
            ]
            for worker, task in zip(self._workers, tasks):
                worker.submit(task)
            deadline_hit = False
            while not all(task.done.is_set() for task in tasks):
                remaining = float(deadline_monotonic) - time.monotonic()
                if remaining <= 0.0:
                    deadline_hit = True
                    cancel.set()
                    try:
                        barrier.abort()
                    except Exception:
                        pass
                    break
                if cancel.is_set():
                    break
                next(task for task in tasks if not task.done.is_set()).done.wait(
                    timeout=min(0.005, remaining)
                )
            # Native calls cannot be safely force-cancelled.  The important
            # invariant is that action selection does not return until all
            # worker tasks reached their cleanup finally path.
            cancel.set()
            for task in tasks:
                task.done.wait()
            errors = [task.error for task in tasks if task.error is not None]
            if deadline_hit:
                raise R222StockBatchDeadline(
                    "stock eight-lane search exceeded deadline after joining cleanup"
                )
            if errors:
                first = errors[0]
                raise R222StockBatchIntegrityError(
                    "stock eight-lane search rejected: "
                    + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
                ) from first
            results = tuple(task.result for task in tasks)
            if any(row is None for row in results):
                raise R222StockBatchIntegrityError("stock lane completed without a result")
            concrete = tuple(row for row in results if row is not None)
            if any(row.search_end_calls != 1 for row in concrete):
                raise R222StockBatchIntegrityError("a stock lane did not cleanly SearchEnd")
            return concrete

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.close()


PacketT = TypeVar("PacketT")
LeafT = TypeVar("LeafT")


@dataclass
class _LeafRequest(Generic[PacketT, LeafT]):
    packets: tuple[PacketT, ...]
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    result: tuple[LeafT, ...] | None = None
    error: BaseException | None = None
    cancelled: bool = False


class R222FrozenLeafMicrobatchBroker(Generic[PacketT, LeafT]):
    """Queue-owned frozen inference broker; it owns no stock handle or tree.

    ``forward`` must be a frozen model forward (typically a wrapper around
    ``forward_leaf_batch``).  It receives flattened rows exactly once per
    broker batch and must return the same row count/order.  The broker is
    deliberately thread-only: it is an in-process seam for one game worker;
    a future multi-process GPU server may expose the same callable protocol
    without granting its process access to simulator handles.
    """

    source = "trained_checkpoint_policy_value_head"

    def __init__(
        self,
        forward: Callable[[Sequence[PacketT]], Sequence[LeafT]],
        *,
        checkpoint_digest: str,
        max_batch_rows: int = R222_STOCK_LANE_COUNT,
        coalesce_ms: float = 0.5,
    ) -> None:
        if not checkpoint_digest.startswith("sha256:"):
            raise ValueError("frozen leaf broker needs an immutable checkpoint digest")
        if int(max_batch_rows) < R222_STOCK_LANE_COUNT:
            raise ValueError("frozen leaf broker must admit one full eight-lane batch")
        if float(coalesce_ms) < 0.0:
            raise ValueError("leaf coalesce window must be non-negative")
        self.checkpoint_digest = checkpoint_digest
        self._forward = forward
        self.max_batch_rows = int(max_batch_rows)
        self.coalesce_seconds = float(coalesce_ms) / 1000.0
        self._queue: queue.Queue[_LeafRequest[PacketT, LeafT] | None] = queue.Queue()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._request_count = 0
        self._forward_count = 0
        self._rows: list[int] = []
        self._thread = threading.Thread(
            target=self._serve, name="r222-frozen-leaf-microbatch", daemon=True
        )
        self._thread.start()

    def telemetry_mark(self) -> tuple[int, int, int]:
        with self._stats_lock:
            return self._request_count, self._forward_count, len(self._rows)

    def telemetry_since(self, mark: tuple[int, int, int]) -> dict[str, Any]:
        with self._stats_lock:
            rows = self._rows[int(mark[2]) :]
            return {
                "queue_owned_frozen_leaf_requests": self._request_count - int(mark[0]),
                "queue_owned_frozen_leaf_forwards": self._forward_count - int(mark[1]),
                "queue_owned_frozen_leaf_rows": sum(rows),
                "queue_owned_frozen_leaf_batch_rows": tuple(rows),
                "inference_batch_size_mean": (
                    math.fsum(rows) / len(rows) if rows else 0.0
                ),
            }

    def __call__(self, packets: Sequence[PacketT], *, deadline_monotonic: float | None = None) -> tuple[LeafT, ...]:
        if self._closed:
            raise R222StockBatchIntegrityError("frozen leaf broker is closed")
        payload = tuple(packets)
        if not payload:
            return ()
        deadline = float(deadline_monotonic) if deadline_monotonic is not None else float("inf")
        if time.monotonic() >= deadline:
            raise R222StockBatchDeadline("leaf broker request reached its deadline")
        request: _LeafRequest[PacketT, LeafT] = _LeafRequest(payload, deadline)
        self._queue.put(request)
        remaining = None if math.isinf(deadline) else max(0.0, deadline - time.monotonic())
        if not request.done.wait(timeout=remaining):
            request.cancelled = True
            # Do not leave GPU work behind a real action.  We wait until the
            # broker has observed cancellation or completed its current forward.
            request.done.wait()
            raise R222StockBatchDeadline("leaf broker exceeded its deadline")
        if request.error is not None:
            raise request.error
        if request.result is None or len(request.result) != len(payload):
            raise R222StockBatchIntegrityError("frozen leaf broker changed row count")
        return request.result

    def _serve(self) -> None:
        stop_after_batch = False
        while True:
            first = self._queue.get()
            if first is None:
                return
            pending = [first]
            rows = len(first.packets)
            until = time.monotonic() + self.coalesce_seconds
            while rows < self.max_batch_rows:
                remaining = until - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                if rows + len(item.packets) > self.max_batch_rows:
                    # Keep a too-large next request ordered by putting it back;
                    # no request is split across independent tree transactions.
                    self._queue.put(item)
                    break
                pending.append(item)
                rows += len(item.packets)
            active = [
                request
                for request in pending
                if not request.cancelled and time.monotonic() < request.deadline
            ]
            for request in pending:
                if request not in active:
                    request.error = R222StockBatchDeadline(
                        "leaf broker request expired before frozen forward"
                    )
                    request.done.set()
            if active:
                flat = tuple(packet for request in active for packet in request.packets)
                try:
                    evaluated = tuple(self._forward(flat))
                    if len(evaluated) != len(flat):
                        raise R222StockBatchIntegrityError(
                            "frozen leaf forward returned wrong row count"
                        )
                    offset = 0
                    for request in active:
                        count = len(request.packets)
                        request.result = evaluated[offset : offset + count]
                        offset += count
                except BaseException as exc:
                    for request in active:
                        request.error = exc
                finally:
                    with self._stats_lock:
                        self._request_count += len(active)
                        self._forward_count += 1
                        self._rows.append(len(flat))
                    for request in active:
                        request.done.set()
            if stop_after_batch:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


BackupUndo = Callable[[], None]


class R222SharedTreeMicrobatchCoordinator(Generic[PacketT, LeafT]):
    """Commit exactly eight isolated stock leaves into one caller-owned tree.

    ``build_packet`` and ``backup`` run only on the coordinator/caller thread.
    The latter must return an undo closure; this gives the primitive a genuine
    all-or-nothing backup guarantee if a later row is rejected.  ``abort`` is
    where the caller removes selection-time virtual loss or any other selected
    but uncommitted tree state.
    """

    def __init__(
        self,
        lane_pool: R222StockSearchLanePool,
        leaf_broker: Callable[..., Sequence[LeafT]],
        *,
        lane_count: int = R222_STOCK_LANE_COUNT,
    ) -> None:
        if int(lane_count) != R222_STOCK_LANE_COUNT:
            raise ValueError("r222 shared-tree coordinator is exactly eight lanes")
        if lane_pool.lane_count != R222_STOCK_LANE_COUNT:
            raise ValueError("r222 shared-tree coordinator needs an exact-eight lane pool")
        self.lane_pool = lane_pool
        self.leaf_broker = leaf_broker
        self.lane_count = R222_STOCK_LANE_COUNT

    def execute(
        self,
        request: R222SharedTreeBatchRequest,
        *,
        deadline_monotonic: float,
        build_packet: Callable[[R222StockTrajectory, R222LaneExecution], PacketT],
        backup: Callable[[R222StockTrajectory, R222LaneExecution, LeafT], BackupUndo],
        abort: Callable[[BaseException], None],
    ) -> tuple[tuple[LeafT, ...], R222StockMicrobatchReceipt]:
        """Execute one complete eight-row batch or raise after transactional abort."""

        request.validate(lane_count=self.lane_count)
        started = time.monotonic()
        marker = getattr(self.leaf_broker, "telemetry_mark", lambda: None)()
        undos: list[BackupUndo] = []
        try:
            executions = self.lane_pool.execute(
                request.trajectories, deadline_monotonic=float(deadline_monotonic)
            )
            if len(executions) != self.lane_count:
                raise R222StockBatchIntegrityError("stock lane pool returned a partial batch")
            expected_root = canonical_observation_fingerprint(request.trajectories[0].root_observation)
            if any(row.root_observation_fingerprint != expected_root for row in executions):
                raise R222StockBatchIntegrityError("stock lane root fingerprints diverged")
            packets = tuple(
                build_packet(trajectory, execution)
                for trajectory, execution in zip(request.trajectories, executions)
            )
            # The broker may queue/coalesce across callers, but this request is
            # never split: all eight leaves must be returned in the same order.
            try:
                evaluated = tuple(
                    self.leaf_broker(packets, deadline_monotonic=float(deadline_monotonic))
                )
            except TypeError:
                # Small synchronous frozen-forward adapters used by existing
                # tests may not accept the optional deadline keyword.
                evaluated = tuple(self.leaf_broker(packets))
            if len(evaluated) != self.lane_count:
                raise R222StockBatchIntegrityError("frozen leaf batch was partial")
            for trajectory, execution, leaf in zip(request.trajectories, executions, evaluated):
                undo = backup(trajectory, execution, leaf)
                if not callable(undo):
                    raise R222StockBatchIntegrityError(
                        "shared-tree backup must return an undo closure"
                    )
                undos.append(undo)
            elapsed = max(0.0, time.monotonic() - started)
            telemetry = (
                dict(getattr(self.leaf_broker, "telemetry_since", lambda _mark: {})(marker))
                if marker is not None
                else {}
            )
            forwards = int(telemetry.get("queue_owned_frozen_leaf_forwards", 1) or 0)
            rows = int(telemetry.get("queue_owned_frozen_leaf_rows", self.lane_count) or 0)
            if forwards < 1 or rows < self.lane_count:
                raise R222StockBatchIntegrityError("frozen leaf broker omitted batch telemetry")
            receipt = R222StockMicrobatchReceipt(
                schema=R222_STOCK_SHARED_TREE_BATCH_SCHEMA,
                requested_lane_count=self.lane_count,
                completed_lane_count=self.lane_count,
                lane_ids=tuple(row.lane_id for row in executions),
                unique_raw_handle_count=len({row[2] for row in self.lane_pool.lane_topology}),
                root_observation_fingerprint=expected_root,
                decision_fingerprint=request.decision_fingerprint,
                complete_root_legal_actions=request.complete_root_legal_actions,
                direct_policy_action=request.direct_policy_action,
                search_begin_calls=sum(row.search_begin_calls for row in executions),
                search_step_calls=sum(row.search_step_calls for row in executions),
                search_release_calls=sum(row.search_release_calls for row in executions),
                search_end_calls=sum(row.search_end_calls for row in executions),
                leaf_request_count=int(telemetry.get("queue_owned_frozen_leaf_requests", 1) or 0),
                leaf_batch_rows=rows,
                frozen_model_forwards=forwards,
                completed_backed_simulations=self.lane_count,
                elapsed_seconds=elapsed,
                backed_simulations_per_second=(self.lane_count / elapsed if elapsed > 0.0 else 0.0),
                all_lanes_cleaned_before_return=True,
                partial_lane_statistics_used=False,
                lane_topology=self.lane_pool.lane_topology,
            )
            return evaluated, receipt
        except BaseException as exc:
            undo_errors: list[BaseException] = []
            for undo in reversed(undos):
                try:
                    undo()
                except BaseException as undo_error:
                    undo_errors.append(undo_error)
            try:
                abort(exc)
            except BaseException as abort_error:
                undo_errors.append(abort_error)
            if undo_errors:
                raise R222StockBatchIntegrityError(
                    "shared-tree batch failed and rollback/abort was incomplete"
                ) from undo_errors[0]
            if isinstance(exc, R222StockBatchError):
                raise
            raise R222StockBatchIntegrityError(
                f"shared-tree stock batch failed: {type(exc).__name__}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Minimal real shared-logical-tree viability core
# ---------------------------------------------------------------------------


@dataclass
class _SharedTreeEdge:
    action: tuple[int, ...]
    prior: float
    visits: int = 0
    total: float = 0.0
    virtual_loss: int = 0
    # A post-step node is keyed by a caller-attested complete semantic/world
    # state key, never by a public observation lookalike.
    outcomes: dict[str, "_SharedTreeNode"] = field(default_factory=dict)
    successor_by_parent_world: dict[str, str] = field(default_factory=dict)

    def q(self) -> float:
        return self.total / self.visits if self.visits else 0.0


@dataclass
class _SharedTreeNode:
    actor: int | None
    state_key: str
    edges: list[_SharedTreeEdge] = field(default_factory=list)
    visits: int = 0
    total: float = 0.0
    terminal_or_boundary: bool = False


@dataclass(frozen=True)
class R222SharedTreeLeaf:
    """One complete leaf result for the coordinator-owned logical tree.

    ``semantic_state_key`` must be an attested complete semantic/world key.  A
    public observation hash is specifically insufficient.  For a terminal or
    pre-random boundary, set ``expandable`` false; its value may be backed up,
    but no future trajectory is allowed to select an unobserved chance action.
    """

    value: float
    semantic_state_key: str
    actor: int | None
    legal_actions: tuple[tuple[int, ...], ...] = ()
    priors: tuple[float, ...] = ()
    expandable: bool = True
    boundary_kind: str = "model_leaf"

    def validate(self) -> None:
        if not math.isfinite(float(self.value)):
            raise R222StockBatchIntegrityError("shared-tree leaf value is non-finite")
        if not isinstance(self.semantic_state_key, str) or not self.semantic_state_key:
            raise R222StockBatchIntegrityError(
                "shared-tree leaf lacks an attested complete semantic/world key"
            )
        if self.actor is not None and int(self.actor) not in (0, 1):
            raise R222StockBatchIntegrityError("shared-tree leaf actor is invalid")
        if not self.expandable:
            return
        actions = tuple(tuple(int(item) for item in action) for action in self.legal_actions)
        if not actions or len(set(actions)) != len(actions):
            raise R222StockBatchIntegrityError("expandable shared-tree leaf has invalid legal actions")
        if len(self.priors) != len(actions) or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.priors
        ):
            raise R222StockBatchIntegrityError("shared-tree leaf priors are invalid")
        if math.fsum(float(value) for value in self.priors) <= 0.0:
            raise R222StockBatchIntegrityError("shared-tree leaf priors have zero mass")


@dataclass(frozen=True)
class R222SharedTreeLaneSeed:
    """One root-sampled world for one lane, supplied by the coordinator."""

    lane_id: int
    search_inputs: Mapping[str, Sequence[int]]
    root_world_key: str


@dataclass(frozen=True)
class R222SharedTreeReservation:
    lane_id: int
    root_world_key: str
    action_path: tuple[tuple[int, ...], ...]
    selected_edges: tuple[tuple[int, ...], ...]
    parent_world_keys: tuple[str, ...]


@dataclass(frozen=True)
class R222SharedTreeLeafWork(Generic[PacketT]):
    """Model work for a completed private trajectory.

    A terminal leaf supplies ``terminal_leaf`` and no model packet.  A
    pre-random chance boundary is *not* terminal, but it remains a frozen model
    leaf and therefore supplies a packet with ``boundary_kind`` recorded by the
    caller's decoder.
    """

    model_packet: PacketT | None
    safe_model_input_key: str | None
    terminal_leaf: R222SharedTreeLeaf | None = None

    def validate(self) -> None:
        if self.model_packet is None and self.terminal_leaf is None:
            raise R222StockBatchIntegrityError("leaf work is neither terminal nor model-evaluated")
        if self.model_packet is not None and self.terminal_leaf is not None:
            raise R222StockBatchIntegrityError("leaf work cannot be terminal and model-evaluated")
        if self.model_packet is not None and (
            not isinstance(self.safe_model_input_key, str)
            or not self.safe_model_input_key
        ):
            raise R222StockBatchIntegrityError(
                "model leaf needs a complete-world-safe model-input key"
            )
        if self.terminal_leaf is not None:
            self.terminal_leaf.validate()


@dataclass(frozen=True)
class R222SharedTreeDecisionReceipt:
    """Receipt for one real eight-lane shared-tree microbatch decision."""

    schema: str
    transaction_id: str
    shared_logical_tree: bool
    shared_logical_tree_id: str
    requested_lane_count: int
    active_lane_count: int
    lane_ids: tuple[int, ...]
    unique_raw_handle_count: int
    max_concurrent_active_lanes: int
    all_eight_began_before_first_step: bool
    root_visits_before: int
    root_visits_after: int
    root_visit_delta: int
    completed_backed_simulations: int
    selected_action: tuple[int, ...]
    selected_action_legal: bool
    selected_action_fully_backed_up: bool
    selected_action_visit_count: int
    selected_action_completed_backups: int
    deterministic_backup_order: tuple[int, ...]
    virtual_loss_reserved: int
    virtual_loss_after: int
    outstanding_reservations: int
    outstanding_virtual_loss: int
    duplicate_path_avoided: int
    inflight_eval_coalesced: int
    eval_cache_hits: int
    unavoidable_distinct_world_repeats: int
    same_world_model_eval_dedup: bool
    # Stock SearchIds are lane-handle scoped and deliberately released before
    # a batch returns.  This diagnostic core therefore replays the sealed
    # descriptor (root observation + exact particle/world + selected path) on
    # the owning lane instead of caching an ID across lanes or turns.
    state_cache_hits: int
    replayed_steps: int
    native_search_id_cross_lane_reuse: int
    leaf_microbatch_count: int
    leaf_microbatch_sizes: tuple[int, ...]
    leaf_microbatch_mean: float
    leaf_microbatch_p95: float
    leaf_microbatch_max: int
    terminal_leaf_count: int
    pre_random_boundary_leaf_count: int
    private_random_outcome_samples: int
    guessed_random_rules_or_successors: int
    unobserved_random_outcome_advances: int
    per_lane: tuple[dict[str, Any], ...]
    all_lane_work_finished_before_return: bool
    partial_lane_statistics_used: bool
    forest_merge_used: bool
    elapsed_seconds: float
    backed_simulations_per_second: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class R222SharedLogicalMCTSTree:
    """One coordinator-owned PUCT tree with virtual-loss reservations.

    Lane threads never receive this object.  ``reserve`` and ``backup`` hold a
    lock so a future caller may issue reservation waves safely, while all
    actual mutation remains on the coordinator.  A node follows an edge into a
    child only when the same parent-world key has an attested successor mapping;
    therefore neither a different hidden particle nor an unresolved random
    world can borrow a public-lookalike continuation.
    """

    def __init__(
        self,
        *,
        decision_fingerprint: str,
        root_actions: Sequence[Sequence[int]],
        root_priors: Sequence[float],
        root_actor: int,
        puct_c: float = 1.25,
        max_depth: int = 8,
    ) -> None:
        actions = tuple(tuple(int(item) for item in action) for action in root_actions)
        priors = tuple(float(value) for value in root_priors)
        if not isinstance(decision_fingerprint, str) or not decision_fingerprint:
            raise ValueError("shared tree needs a decision fingerprint")
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("shared tree root actions are invalid")
        if len(priors) != len(actions) or any(
            not math.isfinite(value) or value < 0.0 for value in priors
        ) or math.fsum(priors) <= 0.0:
            raise ValueError("shared tree root priors are invalid")
        if int(root_actor) not in (0, 1):
            raise ValueError("shared tree root actor is invalid")
        if int(max_depth) < 1:
            raise ValueError("shared tree max depth must be positive")
        self.decision_fingerprint = decision_fingerprint
        self.tree_id = "r222-shared-tree:" + hashlib.sha256(
            decision_fingerprint.encode("utf-8")
        ).hexdigest()
        total_prior = math.fsum(priors)
        self._root = _SharedTreeNode(
            actor=int(root_actor),
            state_key="root:" + decision_fingerprint,
            edges=[
                _SharedTreeEdge(action=action, prior=prior / total_prior)
                for action, prior in zip(actions, priors)
            ],
        )
        self.puct_c = float(puct_c)
        self.max_depth = int(max_depth)
        self._lock = threading.RLock()
        self._reservations: dict[int, R222SharedTreeReservation] = {}
        self._reservation_nodes: dict[int, tuple[_SharedTreeNode, ...]] = {}
        self._reservation_edges: dict[int, tuple[_SharedTreeEdge, ...]] = {}
        self._duplicate_path_avoided = 0

    @property
    def root_visits(self) -> int:
        with self._lock:
            return self._root.visits

    @property
    def outstanding_reservations(self) -> int:
        with self._lock:
            return len(self._reservations)

    @property
    def outstanding_virtual_loss(self) -> int:
        with self._lock:
            return self._count_virtual_loss(self._root)

    def transaction_snapshot(self) -> tuple[Any, ...]:
        """Capture one internally consistent coordinator-owned tree snapshot."""

        with self._lock:
            # Deep-copy the tuple in one operation so references from the
            # reservation maps still point into the copied root.  Copying the
            # members separately would create unrelated edge objects and make
            # rollback leak virtual loss.
            return copy.deepcopy(
                (
                    self._root,
                    self._reservations,
                    self._reservation_nodes,
                    self._reservation_edges,
                    self._duplicate_path_avoided,
                )
            )

    def restore_transaction_snapshot(self, snapshot: tuple[Any, ...]) -> None:
        """Atomically restore a snapshot after any partial lane failure."""

        if len(snapshot) != 5:
            raise R222StockBatchIntegrityError("invalid shared-tree transaction snapshot")
        with self._lock:
            (
                self._root,
                self._reservations,
                self._reservation_nodes,
                self._reservation_edges,
                self._duplicate_path_avoided,
            ) = snapshot

    @staticmethod
    def _count_virtual_loss(node: _SharedTreeNode) -> int:
        count = 0
        for edge in node.edges:
            count += edge.virtual_loss
            for child in edge.outcomes.values():
                count += R222SharedLogicalMCTSTree._count_virtual_loss(child)
        return count

    @staticmethod
    def _edge_score(node: _SharedTreeNode, edge: _SharedTreeEdge, *, actor_sign: float) -> float:
        return (
            actor_sign * edge.q()
            + math.sqrt(max(1, node.visits)) * edge.prior / (1 + edge.visits + edge.virtual_loss)
            - float(edge.virtual_loss)
        )

    def _select_edge(self, node: _SharedTreeNode, root_actor: int) -> _SharedTreeEdge:
        if not node.edges:
            raise R222StockBatchIntegrityError("cannot select from an unexpanded shared node")
        sign = 1.0 if node.actor == root_actor else -1.0
        # Record a reservation-induced diversion only when virtual loss changes
        # the deterministic PUCT winner.  It is evidence, not a claim that
        # every lane necessarily chose a distinct legal action.
        baseline = max(
            node.edges,
            key=lambda edge: (
                sign * edge.q() + math.sqrt(max(1, node.visits)) * edge.prior / (1 + edge.visits),
                edge.prior,
                tuple(-item for item in edge.action),
            ),
        )
        selected = max(
            node.edges,
            key=lambda edge: (
                self._edge_score(node, edge, actor_sign=sign),
                edge.prior,
                tuple(-item for item in edge.action),
            ),
        )
        if selected is not baseline:
            self._duplicate_path_avoided += 1
        return selected

    def reserve(self, seed: R222SharedTreeLaneSeed, *, root_actor: int) -> R222SharedTreeReservation:
        """Select and reserve one path under lock before its private rollout."""

        if int(seed.lane_id) in self._reservations:
            raise R222StockBatchIntegrityError("lane already has a shared-tree reservation")
        if not isinstance(seed.root_world_key, str) or not seed.root_world_key:
            raise R222StockBatchIntegrityError("lane lacks a complete root-world key")
        with self._lock:
            node = self._root
            world_key = seed.root_world_key
            nodes = [node]
            edges: list[_SharedTreeEdge] = []
            actions: list[tuple[int, ...]] = []
            parent_worlds: list[str] = []
            for _ in range(self.max_depth):
                if node.terminal_or_boundary or not node.edges:
                    break
                edge = self._select_edge(node, int(root_actor))
                edge.virtual_loss += 1
                edges.append(edge)
                actions.append(edge.action)
                parent_worlds.append(world_key)
                successor_state_key = edge.successor_by_parent_world.get(world_key)
                if successor_state_key is None:
                    break
                child = edge.outcomes.get(successor_state_key)
                if child is None:
                    raise R222StockBatchIntegrityError("shared-tree successor mapping lost its node")
                node = child
                nodes.append(node)
                world_key = successor_state_key
            if not actions:
                raise R222StockBatchIntegrityError("shared tree selected no simulatable action")
            reservation = R222SharedTreeReservation(
                lane_id=int(seed.lane_id),
                root_world_key=seed.root_world_key,
                action_path=tuple(actions),
                selected_edges=tuple(edge.action for edge in edges),
                parent_world_keys=tuple(parent_worlds),
            )
            self._reservations[reservation.lane_id] = reservation
            self._reservation_nodes[reservation.lane_id] = tuple(nodes)
            self._reservation_edges[reservation.lane_id] = tuple(edges)
            return reservation

    def reserve_eight(
        self,
        seeds: Sequence[R222SharedTreeLaneSeed],
        *,
        root_actor: int,
    ) -> tuple[R222SharedTreeReservation, ...]:
        if len(seeds) != R222_STOCK_LANE_COUNT:
            raise R222StockBatchIntegrityError("shared tree must reserve exactly eight lanes")
        ordered = tuple(sorted(seeds, key=lambda row: int(row.lane_id)))
        if [row.lane_id for row in ordered] != list(range(R222_STOCK_LANE_COUNT)):
            raise R222StockBatchIntegrityError("shared-tree lane ids must be 0 through seven")
        reserved: list[R222SharedTreeReservation] = []
        try:
            for seed in ordered:
                reserved.append(self.reserve(seed, root_actor=root_actor))
            return tuple(reserved)
        except BaseException:
            self.abort(tuple(reserved))
            raise

    def reserve_next_from_prefix(
        self,
        seed: R222SharedTreeLaneSeed,
        *,
        root_actor: int,
        action_prefix: Sequence[Sequence[int]],
        world_prefix: Sequence[str],
    ) -> R222SharedTreeReservation:
        """Reserve exactly one next edge below a materialized lane frontier.

        The persistent-session transport retains a native ``SearchId`` per
        lane.  It must therefore *not* replay a root path merely because the
        tree has learned a deeper child.  This method validates the exact
        lane/world prefix already materialized by that native handle, reserves
        every prefix edge for truthful virtual-loss accounting, then selects
        exactly one new legal child edge.  The returned action path is full for
        backup, but its final action is the only one sent to ``SearchStep``.
        """

        lane_id = int(seed.lane_id)
        prefix = tuple(tuple(int(item) for item in action) for action in action_prefix)
        worlds = tuple(str(world) for world in world_prefix)
        if len(worlds) != len(prefix) + 1 or not worlds or worlds[0] != seed.root_world_key:
            raise R222StockBatchIntegrityError(
                "persistent lane prefix lacks exact root/successor world keys"
            )
        with self._lock:
            if lane_id in self._reservations:
                raise R222StockBatchIntegrityError("lane already has a shared-tree reservation")
            if len(prefix) >= self.max_depth:
                raise R222StockBatchIntegrityError("persistent lane reached configured tree depth")
            node = self._root
            nodes = [node]
            edges: list[_SharedTreeEdge] = []
            parent_worlds: list[str] = []
            actions: list[tuple[int, ...]] = []
            for action, parent_world, expected_world in zip(prefix, worlds, worlds[1:]):
                edge = next((row for row in node.edges if row.action == action), None)
                if edge is None:
                    raise R222StockBatchIntegrityError(
                        "persistent lane selected an action absent from its materialized tree node"
                    )
                actual_world = edge.successor_by_parent_world.get(parent_world)
                if actual_world != expected_world:
                    raise R222StockBatchIntegrityError(
                        "persistent lane native frontier does not match its exact tree world"
                    )
                child = edge.outcomes.get(expected_world)
                if child is None:
                    raise R222StockBatchIntegrityError(
                        "persistent lane successor mapping has no materialized tree node"
                    )
                edge.virtual_loss += 1
                edges.append(edge)
                actions.append(edge.action)
                parent_worlds.append(parent_world)
                node = child
                nodes.append(node)
            if node.terminal_or_boundary or not node.edges:
                raise R222StockBatchIntegrityError(
                    "persistent lane frontier cannot safely select another edge"
                )
            edge = self._select_edge(node, int(root_actor))
            edge.virtual_loss += 1
            edges.append(edge)
            actions.append(edge.action)
            parent_worlds.append(worlds[-1])
            reservation = R222SharedTreeReservation(
                lane_id=lane_id,
                root_world_key=seed.root_world_key,
                action_path=tuple(actions),
                selected_edges=tuple(row.action for row in edges),
                parent_world_keys=tuple(parent_worlds),
            )
            self._reservations[lane_id] = reservation
            self._reservation_nodes[lane_id] = tuple(nodes)
            self._reservation_edges[lane_id] = tuple(edges)
            return reservation

    @staticmethod
    def _expand_node(node: _SharedTreeNode, leaf: R222SharedTreeLeaf) -> None:
        leaf.validate()
        if node.edges or node.terminal_or_boundary:
            # Re-evaluating an already materialized same-world node is allowed
            # only if it has the exact same semantic identity/shape.  Values
            # are backed up independently; this never conflates worlds.
            if node.state_key != leaf.semantic_state_key:
                raise R222StockBatchIntegrityError("leaf semantic key conflicts with shared node")
            return
        node.actor = leaf.actor
        node.state_key = leaf.semantic_state_key
        node.terminal_or_boundary = not leaf.expandable
        if leaf.expandable:
            total = math.fsum(float(value) for value in leaf.priors)
            node.edges = [
                _SharedTreeEdge(action=tuple(action), prior=float(prior) / total)
                for action, prior in zip(leaf.legal_actions, leaf.priors)
            ]

    def backup(self, reservation: R222SharedTreeReservation, leaf: R222SharedTreeLeaf) -> BackupUndo:
        """Materialize one safe successor and back it up into this *same* tree."""

        leaf.validate()
        with self._lock:
            active = self._reservations.get(reservation.lane_id)
            if active != reservation:
                raise R222StockBatchIntegrityError("shared-tree reservation is not active")
            # A compact deep snapshot makes this diagnostic core genuinely
            # transactional: a later lane backup failure restores both visits
            # and virtual-loss reservations before the caller's abort clears
            # them.  It deliberately avoids a partial-tree fallback.
            root_snapshot = copy.deepcopy(self._root)
            reservations_snapshot = copy.deepcopy(self._reservations)
            nodes_snapshot = copy.deepcopy(self._reservation_nodes)
            edges_snapshot = copy.deepcopy(self._reservation_edges)
            duplicate_snapshot = self._duplicate_path_avoided
            edges = self._reservation_edges[reservation.lane_id]
            nodes = list(self._reservation_nodes[reservation.lane_id])
            parent_worlds = reservation.parent_world_keys
            terminal_node = nodes[-1]
            final_edge = edges[-1]
            parent_world = parent_worlds[-1]
            existing_key = final_edge.successor_by_parent_world.get(parent_world)
            if existing_key is None:
                final_edge.successor_by_parent_world[parent_world] = leaf.semantic_state_key
                terminal_node = _SharedTreeNode(actor=leaf.actor, state_key=leaf.semantic_state_key)
                final_edge.outcomes[leaf.semantic_state_key] = terminal_node
                nodes.append(terminal_node)
            elif existing_key != leaf.semantic_state_key:
                raise R222StockBatchIntegrityError(
                    "same complete parent-world/action produced a different semantic successor"
                )
            else:
                terminal_node = final_edge.outcomes.get(existing_key)
                if terminal_node is None:
                    raise R222StockBatchIntegrityError("shared-tree mapped successor is missing")
                nodes[-1] = terminal_node
            self._expand_node(terminal_node, leaf)
            value = float(leaf.value)
            for node in nodes:
                node.visits += 1
                node.total += value
            for edge in edges:
                if edge.virtual_loss < 1:
                    raise R222StockBatchIntegrityError("shared-tree virtual loss underflow")
                edge.virtual_loss -= 1
                edge.visits += 1
                edge.total += value
            self._reservations.pop(reservation.lane_id, None)
            self._reservation_nodes.pop(reservation.lane_id, None)
            self._reservation_edges.pop(reservation.lane_id, None)

            def undo() -> None:
                with self._lock:
                    self._root = root_snapshot
                    self._reservations = reservations_snapshot
                    self._reservation_nodes = nodes_snapshot
                    self._reservation_edges = edges_snapshot
                    self._duplicate_path_avoided = duplicate_snapshot

            return undo

    def abort(self, reservations: Sequence[R222SharedTreeReservation]) -> None:
        """Remove every still-live virtual loss; called on any failed batch."""

        with self._lock:
            for reservation in reservations:
                active = self._reservations.pop(reservation.lane_id, None)
                edges = self._reservation_edges.pop(reservation.lane_id, ())
                self._reservation_nodes.pop(reservation.lane_id, None)
                if active is None:
                    continue
                for edge in edges:
                    if edge.virtual_loss < 1:
                        raise R222StockBatchIntegrityError("virtual-loss underflow on abort")
                    edge.virtual_loss -= 1
            if self._reservations:
                raise R222StockBatchIntegrityError("shared-tree abort left unknown reservations")

    def selected_root_action(self) -> tuple[tuple[int, ...], int]:
        with self._lock:
            if self._reservations or self.outstanding_virtual_loss:
                raise R222StockBatchIntegrityError("cannot select root while reservations remain")
            if self._root.visits < 1:
                raise R222StockBatchIntegrityError("shared tree has no completed backup")
            edge = max(
                self._root.edges,
                key=lambda row: (row.visits, tuple(-item for item in row.action)),
            )
            if edge.visits < 1:
                raise R222StockBatchIntegrityError("selected root edge has no backup")
            return edge.action, edge.visits


class R222StockSharedTreeMCTS(Generic[PacketT, LeafT]):
    """Coordinator-only true shared-tree eight-lane diagnostic execution.

    This is the concrete r222 viability API.  It performs no forest merge:
    every successful lane backs up directly into ``self.tree`` in lane-id order.
    The frozen model is invoked only on unique caller-attested model-input keys;
    cached outputs are reused only for the same safe key, never by public board
    equality alone.
    """

    def __init__(
        self,
        *,
        tree: R222SharedLogicalMCTSTree,
        lane_pool: R222StockSearchLanePool,
        leaf_broker: Callable[..., Sequence[LeafT]],
        root_observation: Mapping[str, Any],
        root_actor: int,
        direct_policy_action: Sequence[int],
    ) -> None:
        if lane_pool.lane_count != R222_STOCK_LANE_COUNT:
            raise ValueError("shared-tree r222 core requires eight active stock lanes")
        self.tree = tree
        self.lane_pool = lane_pool
        self.leaf_broker = leaf_broker
        self.root_observation = dict(root_observation)
        self.root_actor = int(root_actor)
        self.direct_policy_action = tuple(int(item) for item in direct_policy_action)
        self._completed_model_cache: dict[str, LeafT] = {}
        self._transaction_counter = 0

    def run_eight(
        self,
        seeds: Sequence[R222SharedTreeLaneSeed],
        *,
        deadline_monotonic: float,
        make_leaf_work: Callable[[R222SharedTreeReservation, R222LaneExecution], R222SharedTreeLeafWork[PacketT]],
        decode_model_leaf: Callable[[R222SharedTreeReservation, R222LaneExecution, LeafT], R222SharedTreeLeaf],
    ) -> R222SharedTreeDecisionReceipt:
        """Reserve, simulate, batch-evaluate and back up exactly eight lanes."""

        started = time.monotonic()
        self._transaction_counter += 1
        transaction_id = f"{self.tree.tree_id}:tx:{self._transaction_counter:08d}"
        root_before = self.tree.root_visits
        duplicate_before = self.tree._duplicate_path_avoided
        transaction_snapshot = self.tree.transaction_snapshot()
        reservations = self.tree.reserve_eight(seeds, root_actor=self.root_actor)
        virtual_reserved = self.tree.outstanding_virtual_loss
        undos: list[BackupUndo] = []
        try:
            trajectories = tuple(
                R222StockTrajectory(
                    lane_id=seed.lane_id,
                    root_observation=self.root_observation,
                    search_inputs=seed.search_inputs,
                    action_path=reservation.action_path,
                    tree_token=reservation,
                )
                for seed, reservation in zip(
                    sorted(seeds, key=lambda row: int(row.lane_id)), reservations
                )
            )
            executions = self.lane_pool.execute(
                trajectories, deadline_monotonic=float(deadline_monotonic)
            )
            if len(executions) != R222_STOCK_LANE_COUNT:
                raise R222StockBatchIntegrityError("shared-tree core received a partial lane set")
            works = tuple(
                make_leaf_work(reservation, execution)
                for reservation, execution in zip(reservations, executions)
            )
            for work in works:
                work.validate()
            # Coalesce only caller-attested complete-world model-input keys.
            groups: dict[str, list[int]] = {}
            unique_packets: list[PacketT] = []
            unique_keys: list[str] = []
            cache_hits = 0
            for index, work in enumerate(works):
                if work.model_packet is None:
                    continue
                key = str(work.safe_model_input_key)
                if key in self._completed_model_cache:
                    groups.setdefault(key, []).append(index)
                    cache_hits += 1
                    continue
                if key in groups:
                    groups[key].append(index)
                    continue
                groups[key] = [index]
                unique_keys.append(key)
                unique_packets.append(work.model_packet)
            evaluated_by_key: dict[str, LeafT] = {
                key: self._completed_model_cache[key]
                for key in groups
                if key in self._completed_model_cache
            }
            if unique_packets:
                try:
                    outputs = tuple(
                        self.leaf_broker(unique_packets, deadline_monotonic=float(deadline_monotonic))
                    )
                except TypeError:
                    outputs = tuple(self.leaf_broker(unique_packets))
                if len(outputs) != len(unique_packets):
                    raise R222StockBatchIntegrityError("shared-tree leaf broker returned partial rows")
                for key, output in zip(unique_keys, outputs):
                    self._completed_model_cache[key] = output
                    evaluated_by_key[key] = output
            leaves: list[R222SharedTreeLeaf] = []
            for reservation, execution, work in zip(reservations, executions, works):
                if work.terminal_leaf is not None:
                    leaf = work.terminal_leaf
                else:
                    leaf = decode_model_leaf(
                        reservation,
                        execution,
                        evaluated_by_key[str(work.safe_model_input_key)],
                    )
                leaf.validate()
                leaves.append(leaf)
            for reservation, leaf in zip(reservations, leaves):
                undos.append(self.tree.backup(reservation, leaf))
            if self.tree.outstanding_reservations or self.tree.outstanding_virtual_loss:
                raise R222StockBatchIntegrityError("successful shared-tree transaction leaked reservations")
            selected, selected_visits = self.tree.selected_root_action()
            root_after = self.tree.root_visits
            paths = [reservation.action_path for reservation in reservations]
            path_counts: dict[tuple[tuple[int, ...], ...], int] = {}
            for path in paths:
                path_counts[path] = path_counts.get(path, 0) + 1
            distinct_world_repeats = sum(
                count - 1 for count in path_counts.values() if count > 1
            ) - sum(len(indices) - 1 for indices in groups.values() if len(indices) > 1)
            leaf_sizes = (len(unique_packets),) if unique_packets else ()
            elapsed = max(0.0, time.monotonic() - started)
            pre_random = sum(leaf.boundary_kind == "pre_random_frozen_model_leaf" for leaf in leaves)
            return R222SharedTreeDecisionReceipt(
                schema=R222_STOCK_SHARED_TREE_BATCH_SCHEMA,
                transaction_id=transaction_id,
                shared_logical_tree=True,
                shared_logical_tree_id=self.tree.tree_id,
                requested_lane_count=R222_STOCK_LANE_COUNT,
                active_lane_count=len(executions),
                lane_ids=tuple(row.lane_id for row in executions),
                unique_raw_handle_count=len({row[2] for row in self.lane_pool.lane_topology}),
                max_concurrent_active_lanes=_max_overlap(executions),
                all_eight_began_before_first_step=(
                    bool(executions)
                    and all(row.first_search_step_started_monotonic is not None for row in executions)
                    and max(row.search_begin_completed_monotonic for row in executions)
                    <= min(
                        float(row.first_search_step_started_monotonic)
                        for row in executions
                        if row.first_search_step_started_monotonic is not None
                    )
                ),
                root_visits_before=root_before,
                root_visits_after=root_after,
                root_visit_delta=root_after - root_before,
                completed_backed_simulations=len(leaves),
                selected_action=selected,
                selected_action_legal=selected in tuple(edge.action for edge in self.tree._root.edges),
                selected_action_fully_backed_up=selected_visits >= 1,
                selected_action_visit_count=selected_visits,
                selected_action_completed_backups=selected_visits,
                deterministic_backup_order=tuple(reservation.lane_id for reservation in reservations),
                virtual_loss_reserved=virtual_reserved,
                virtual_loss_after=self.tree.outstanding_virtual_loss,
                outstanding_reservations=self.tree.outstanding_reservations,
                outstanding_virtual_loss=self.tree.outstanding_virtual_loss,
                duplicate_path_avoided=self.tree._duplicate_path_avoided - duplicate_before,
                inflight_eval_coalesced=sum(len(indices) - 1 for indices in groups.values() if len(indices) > 1),
                eval_cache_hits=cache_hits,
                unavoidable_distinct_world_repeats=max(0, distinct_world_repeats),
                same_world_model_eval_dedup=(
                    sum(len(indices) - 1 for indices in groups.values() if len(indices) > 1) > 0
                    or cache_hits > 0
                ),
                state_cache_hits=0,
                replayed_steps=sum(len(reservation.action_path) for reservation in reservations),
                native_search_id_cross_lane_reuse=0,
                leaf_microbatch_count=1 if unique_packets else 0,
                leaf_microbatch_sizes=leaf_sizes,
                leaf_microbatch_mean=float(leaf_sizes[0]) if leaf_sizes else 0.0,
                leaf_microbatch_p95=float(leaf_sizes[0]) if leaf_sizes else 0.0,
                leaf_microbatch_max=max(leaf_sizes, default=0),
                terminal_leaf_count=sum(work.terminal_leaf is not None for work in works),
                pre_random_boundary_leaf_count=pre_random,
                private_random_outcome_samples=0,
                guessed_random_rules_or_successors=0,
                unobserved_random_outcome_advances=0,
                per_lane=tuple(
                    {
                        "lane_id": execution.lane_id,
                        "trajectory_depth": len(reservation.action_path),
                        "search_begin_calls": execution.search_begin_calls,
                        "search_step_calls": execution.search_step_calls,
                        "search_release_calls": execution.search_release_calls,
                        "search_end_calls": execution.search_end_calls,
                        "task_started_monotonic": execution.task_started_monotonic,
                        "search_begin_started_monotonic": execution.search_begin_started_monotonic,
                        "search_begin_completed_monotonic": execution.search_begin_completed_monotonic,
                        "first_search_step_started_monotonic": execution.first_search_step_started_monotonic,
                        "first_search_step_completed_monotonic": execution.first_search_step_completed_monotonic,
                        "cleanup_completed_monotonic": execution.cleanup_completed_monotonic,
                        "backups": 1,
                        "root_world_key": reservation.root_world_key,
                        "state_cache_hit": False,
                        "replayed_steps": len(reservation.action_path),
                        "boundary_kind": leaf.boundary_kind,
                    }
                    for reservation, execution, leaf in zip(reservations, executions, leaves)
                ),
                all_lane_work_finished_before_return=True,
                partial_lane_statistics_used=False,
                forest_merge_used=False,
                elapsed_seconds=elapsed,
                backed_simulations_per_second=(len(leaves) / elapsed if elapsed else 0.0),
            )
        except BaseException as exc:
            # The lane pool has already joined every native worker before it
            # raises.  This abort only clears coordinator-side virtual loss.
            self.tree.restore_transaction_snapshot(transaction_snapshot)
            if self.tree.outstanding_reservations or self.tree.outstanding_virtual_loss:
                raise R222StockBatchIntegrityError(
                    "failed shared-tree transaction leaked reservations"
                ) from exc
            if isinstance(exc, R222StockBatchError):
                raise
            raise R222StockBatchIntegrityError(
                f"shared-tree transaction failed: {type(exc).__name__}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Persistent per-decision retained-SearchId transport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R222PersistentFrontierIdentity:
    """Attested identity and step permission for one retained native frontier.

    ``world_key`` is a complete simulator/world key, never a public-observation
    hash.  ``legal_fingerprint`` binds the exact current legal action order.
    The caller must set ``deterministic_transition_permitted`` only after its
    stock capability layer has ruled out opaque RNG progression for this step.
    """

    world_key: str
    legal_fingerprint: str
    legal_actions: tuple[tuple[int, ...], ...]
    deterministic_transition_permitted: bool

    def validate(self) -> None:
        if not isinstance(self.world_key, str) or not self.world_key:
            raise R222StockBatchIntegrityError("retained search frontier has no exact world key")
        if not isinstance(self.legal_fingerprint, str) or not self.legal_fingerprint:
            raise R222StockBatchIntegrityError("retained search frontier has no legal fingerprint")
        actions = tuple(tuple(int(item) for item in action) for action in self.legal_actions)
        if actions and len(actions) != len(set(actions)):
            raise R222StockBatchIntegrityError("retained search frontier legal actions are invalid")
        if not actions and self.deterministic_transition_permitted:
            raise R222StockBatchIntegrityError(
                "deterministically expandable retained frontier has no legal actions"
            )


@dataclass(frozen=True)
class R222PersistentLaneFrontier:
    """One lane-local, still-live SearchId after open or one SearchStep."""

    lane_id: int
    handle_identity: int | str
    root_search_id: int
    current_search_id: int
    observation: Any
    identity: R222PersistentFrontierIdentity
    action_path: tuple[tuple[int, ...], ...]
    search_begin_calls: int
    search_step_calls: int
    search_release_calls: int
    search_end_calls: int
    owner_thread_id: int
    search_begin_started_monotonic: float
    search_begin_completed_monotonic: float
    first_search_step_started_monotonic: float | None
    first_search_step_completed_monotonic: float | None
    last_search_step_completed_monotonic: float | None


@dataclass(frozen=True)
class R222PersistentLaneExecution:
    """Lifecycle accounting emitted only after the owner-worker cleanup."""

    lane_id: int
    handle_identity: int | str
    root_search_id: int
    final_search_id: int
    search_id_chain: tuple[int, ...]
    action_path: tuple[tuple[int, ...], ...]
    search_begin_calls: int
    search_step_calls: int
    search_release_calls: int
    search_end_calls: int
    cleanup_completed_monotonic: float


@dataclass(frozen=True)
class R222PersistentStepCommand:
    """Master-selected one-edge command for exactly one retained lane state."""

    lane_id: int
    parent_search_id: int
    expected_parent_world_key: str
    expected_legal_fingerprint: str
    selected_action: tuple[int, ...]
    reservation: R222SharedTreeReservation


@dataclass(frozen=True)
class R222PersistentDecisionReceipt:
    """Receipt for a complete retained-ID eight-head decision session."""

    schema: str
    transaction_id: str
    status: str
    success_marker: str | None
    requested_lane_count: int
    active_lane_count: int
    shared_logical_tree: bool
    shared_logical_tree_id: str
    unique_raw_handle_count: int
    search_begin_calls: int
    search_step_calls: int
    search_release_calls: int
    search_end_calls: int
    root_reopen_count: int
    root_replay_count: int
    retained_search_id_across_waves: bool
    wave_count: int
    wave_backups: tuple[int, ...]
    wave_step_overlap: tuple[int, ...]
    leaf_microbatch_sizes: tuple[int, ...]
    completed_backed_simulations: int
    root_visits_before: int
    root_visits_after: int
    selected_action: tuple[int, ...]
    selected_action_legal: bool
    selected_action_fully_backed_up: bool
    selected_action_visit_count: int
    deadline_exhausted: bool
    clean_deadline_zero_backup_fallback: bool
    direct_policy_fallback: bool
    opened_eight_sessions: bool
    outstanding_reservations: int
    outstanding_virtual_loss: int
    private_random_outcome_samples: int
    guessed_random_rules_or_successors: int
    unobserved_random_outcome_advances: int
    all_lane_work_finished_before_return: bool
    per_lane: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class R222AsyncPersistentDecisionReceipt:
    """Small receipt for the non-barrier retained-ID async probe.

    This is deliberately narrower than the synchronous diagnostic receipt: it
    records only the facts needed to prove that one master tree kept eight
    independent native arenas moving without an all-eight step barrier.
    """

    transaction_id: str
    shared_logical_tree: bool
    shared_logical_tree_id: str
    requested_lane_count: int
    active_lane_count: int
    unique_raw_handle_count: int
    completed_backed_simulations: int
    leaf_microbatch_sizes: tuple[int, ...]
    selected_action: tuple[int, ...]
    selected_action_legal: bool
    selected_action_fully_backed_up: bool
    outstanding_reservations: int
    outstanding_virtual_loss: int
    all_lane_work_finished_before_return: bool
    per_lane: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _PersistentState:
    root_search_id: int
    current_search_id: int
    observation: Any
    identity: R222PersistentFrontierIdentity
    action_path: list[tuple[int, ...]]
    live_ids: list[int]
    search_id_chain: list[int]
    search_begin_started_monotonic: float
    search_begin_completed_monotonic: float
    first_search_step_started_monotonic: float | None = None
    first_search_step_completed_monotonic: float | None = None
    last_search_step_completed_monotonic: float | None = None
    search_begin_calls: int = 1
    search_step_calls: int = 0
    search_release_calls: int = 0
    search_end_calls: int = 0


@dataclass
class _PersistentOpenTask:
    root_observation: Mapping[str, Any]
    search_inputs: Mapping[str, Sequence[int]]
    phase_barrier: threading.Barrier
    cancel: threading.Event
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    passed_barrier: bool = False
    result: R222PersistentLaneFrontier | None = None
    error: BaseException | None = None


@dataclass
class _PersistentStepTask:
    command: R222PersistentStepCommand
    phase_barrier: threading.Barrier
    cancel: threading.Event
    deadline: float
    wave_index: int
    done: threading.Event = field(default_factory=threading.Event)
    passed_barrier: bool = False
    result: R222PersistentLaneFrontier | None = None
    error: BaseException | None = None
    completion_queue: queue.Queue["_PersistentStepTask"] | None = None


@dataclass
class _PersistentCloseTask:
    done: threading.Event = field(default_factory=threading.Event)
    result: R222PersistentLaneExecution | None = None
    error: BaseException | None = None


class _PersistentStepOverlap:
    """Measured native SearchStep overlap; fake/physical probes must reach 8."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._maximum: dict[int, int] = {}

    def enter(self, wave_index: int) -> None:
        with self._lock:
            self._active += 1
            self._maximum[wave_index] = max(
                self._maximum.get(wave_index, 0), self._active
            )

    def exit(self) -> None:
        with self._lock:
            self._active -= 1
            if self._active < 0:
                raise R222StockBatchIntegrityError("persistent SearchStep overlap underflow")

    def maximum(self, wave_index: int) -> int:
        with self._lock:
            return self._maximum.get(wave_index, 0)


class _PersistentStockLaneWorker:
    """A thread-affine AgentStart arena retaining exactly one decision session."""

    def __init__(
        self,
        lane_id: int,
        backend_factory: Callable[[int], cg_env.SearchBackend],
        frontier_identity: Callable[[int, Any], R222PersistentFrontierIdentity],
        overlap: _PersistentStepOverlap,
    ) -> None:
        self.lane_id = int(lane_id)
        self._backend_factory = backend_factory
        self._frontier_identity = frontier_identity
        self._overlap = overlap
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.initialization_error: BaseException | None = None
        self.owner_thread_id: int | None = None
        self.handle_identity: int | str | None = None
        self._state: _PersistentState | None = None
        self._closed_execution: R222PersistentLaneExecution | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"r222-persistent-stock-search-lane-{self.lane_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def topology(self) -> tuple[int, int, int | str]:
        if self.owner_thread_id is None or self.handle_identity is None:
            raise R222StockBatchIntegrityError("persistent stock lane has no native handle")
        return self.lane_id, self.owner_thread_id, self.handle_identity

    def submit(self, task: Any) -> None:
        self._queue.put_nowait(task)

    def close_thread(self) -> None:
        self._queue.put(None)
        self._thread.join()

    @staticmethod
    def _wait_phase(task: _PersistentOpenTask | _PersistentStepTask) -> None:
        if task.cancel.is_set():
            raise R222StockBatchDeadline("persistent stock session was cancelled")
        remaining = task.deadline - time.monotonic()
        if remaining <= 0.0:
            raise R222StockBatchDeadline("persistent stock phase reached its deadline")
        try:
            task.phase_barrier.wait(timeout=remaining)
        except threading.BrokenBarrierError as exc:
            if task.cancel.is_set():
                raise R222StockBatchDeadline("persistent stock phase cancelled at deadline") from exc
            raise R222StockBatchIntegrityError("persistent stock lane phase failed") from exc

    def _frontier(self) -> R222PersistentLaneFrontier:
        state = self._state
        if state is None or self.handle_identity is None or self.owner_thread_id is None:
            raise R222StockBatchIntegrityError("persistent stock lane has no live frontier")
        return R222PersistentLaneFrontier(
            lane_id=self.lane_id,
            handle_identity=self.handle_identity,
            root_search_id=state.root_search_id,
            current_search_id=state.current_search_id,
            observation=state.observation,
            identity=state.identity,
            action_path=tuple(state.action_path),
            search_begin_calls=state.search_begin_calls,
            search_step_calls=state.search_step_calls,
            search_release_calls=state.search_release_calls,
            search_end_calls=state.search_end_calls,
            owner_thread_id=self.owner_thread_id,
            search_begin_started_monotonic=state.search_begin_started_monotonic,
            search_begin_completed_monotonic=state.search_begin_completed_monotonic,
            first_search_step_started_monotonic=state.first_search_step_started_monotonic,
            first_search_step_completed_monotonic=state.first_search_step_completed_monotonic,
            last_search_step_completed_monotonic=state.last_search_step_completed_monotonic,
        )

    def _cleanup(self, backend: cg_env.SearchBackend) -> R222PersistentLaneExecution | None:
        state = self._state
        if state is None:
            return self._closed_execution
        cleanup_error: BaseException | None = None
        for search_id in reversed(state.live_ids):
            try:
                backend.search_release(search_id)
                state.search_release_calls += 1
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        try:
            backend.search_end()
            state.search_end_calls += 1
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        completed = time.monotonic()
        result = R222PersistentLaneExecution(
            lane_id=self.lane_id,
            handle_identity=self.topology[2],
            root_search_id=state.root_search_id,
            final_search_id=state.current_search_id,
            search_id_chain=tuple(state.search_id_chain),
            action_path=tuple(state.action_path),
            search_begin_calls=state.search_begin_calls,
            search_step_calls=state.search_step_calls,
            search_release_calls=state.search_release_calls,
            search_end_calls=state.search_end_calls,
            cleanup_completed_monotonic=completed,
        )
        self._state = None
        self._closed_execution = result
        if cleanup_error is not None:
            raise R222StockBatchIntegrityError(
                f"persistent lane {self.lane_id} cleanup failed: {cleanup_error}"
            ) from cleanup_error
        return result

    def _open(
        self, backend: cg_env.SearchBackend, task: _PersistentOpenTask
    ) -> R222PersistentLaneFrontier:
        if self._state is not None:
            raise R222StockBatchIntegrityError("persistent lane attempted a second SearchBegin")
        begin_started = time.monotonic()
        root = backend.search_begin(
            dict(task.root_observation),
            {key: list(value) for key, value in task.search_inputs.items()},
            manual_coin=True,
        )
        begin_completed = time.monotonic()
        root_id = int(root.searchId)
        identity = self._frontier_identity(self.lane_id, root.observation)
        identity.validate()
        self._state = _PersistentState(
            root_search_id=root_id,
            current_search_id=root_id,
            observation=root.observation,
            identity=identity,
            action_path=[],
            live_ids=[root_id],
            search_id_chain=[root_id],
            search_begin_started_monotonic=begin_started,
            search_begin_completed_monotonic=begin_completed,
        )
        self._closed_execution = None
        self._wait_phase(task)
        task.passed_barrier = True
        return self._frontier()

    def _step(
        self, backend: cg_env.SearchBackend, task: _PersistentStepTask
    ) -> R222PersistentLaneFrontier:
        state = self._state
        command = task.command
        if state is None:
            raise R222StockBatchIntegrityError("persistent lane has no open SearchBegin session")
        if int(command.lane_id) != self.lane_id:
            raise R222StockBatchIntegrityError("persistent command was routed to the wrong lane")
        if int(command.parent_search_id) != state.current_search_id:
            raise R222StockBatchIntegrityError("persistent command uses a stale SearchId")
        if int(command.parent_search_id) not in state.live_ids:
            raise R222StockBatchIntegrityError("persistent command uses a released SearchId")
        if _explicit_chance_observation(state.observation):
            raise R222StockBatchIntegrityError("persistent session attempted to advance context 46")
        identity = state.identity
        identity.validate()
        if command.expected_parent_world_key != identity.world_key:
            raise R222StockBatchIntegrityError("persistent command world key is stale")
        if command.expected_legal_fingerprint != identity.legal_fingerprint:
            raise R222StockBatchIntegrityError("persistent command legal fingerprint is stale")
        if not identity.deterministic_transition_permitted:
            raise R222StockBatchIntegrityError("stock transition lacks deterministic capability receipt")
        action = tuple(int(item) for item in command.selected_action)
        if action not in identity.legal_actions:
            raise R222StockBatchIntegrityError("persistent command selected an illegal tree edge")
        self._wait_phase(task)
        task.passed_barrier = True
        if state.first_search_step_started_monotonic is None:
            state.first_search_step_started_monotonic = time.monotonic()
        self._overlap.enter(task.wave_index)
        try:
            child = backend.search_step(state.current_search_id, list(action))
        finally:
            self._overlap.exit()
        now = time.monotonic()
        if state.first_search_step_completed_monotonic is None:
            state.first_search_step_completed_monotonic = now
        state.last_search_step_completed_monotonic = now
        child_id = int(child.searchId)
        if child_id in state.live_ids:
            raise R222StockBatchIntegrityError("stock SearchStep reused a live lane-local SearchId")
        child_identity = self._frontier_identity(self.lane_id, child.observation)
        child_identity.validate()
        state.current_search_id = child_id
        state.observation = child.observation
        state.identity = child_identity
        state.action_path.append(action)
        state.live_ids.append(child_id)
        state.search_id_chain.append(child_id)
        state.search_step_calls += 1
        return self._frontier()

    def _run(self) -> None:
        try:
            backend = self._backend_factory(self.lane_id)
            self.owner_thread_id = threading.get_ident()
            self.handle_identity = getattr(backend, "handle_identity", None)
            if self.handle_identity is None:
                raise R222StockBatchIntegrityError("persistent backend has no handle identity")
        except BaseException as exc:
            self.initialization_error = exc
            self.ready.set()
            return
        self.ready.set()
        while True:
            task = self._queue.get()
            if task is None:
                return
            try:
                if isinstance(task, _PersistentOpenTask):
                    task.result = self._open(backend, task)
                elif isinstance(task, _PersistentStepTask):
                    task.result = self._step(backend, task)
                elif isinstance(task, _PersistentCloseTask):
                    task.result = self._cleanup(backend)
                else:
                    raise R222StockBatchIntegrityError("unknown persistent stock lane task")
            except BaseException as exc:
                task.error = exc
                if isinstance(task, (_PersistentOpenTask, _PersistentStepTask)):
                    task.cancel.set()
                    if not task.passed_barrier:
                        try:
                            task.phase_barrier.abort()
                        except Exception:
                            pass
                try:
                    self._cleanup(backend)
                except BaseException as cleanup_exc:
                    task.error = R222StockBatchIntegrityError(
                        f"persistent lane failure and cleanup failure: {exc}; {cleanup_exc}"
                    )
            finally:
                task.done.set()
                if isinstance(task, _PersistentStepTask) and task.completion_queue is not None:
                    task.completion_queue.put(task)


class R222PersistentStockSessionPool:
    """Eight owner-worker arenas with one retained SearchBegin session each."""

    def __init__(
        self,
        backend_factory: Callable[[int], cg_env.SearchBackend],
        *,
        frontier_identity: Callable[[int, Any], R222PersistentFrontierIdentity],
        lane_count: int = R222_STOCK_LANE_COUNT,
        require_full_step_overlap: bool = False,
    ) -> None:
        if int(lane_count) != R222_STOCK_LANE_COUNT:
            raise ValueError("persistent r222 session pool requires exactly eight lanes")
        self.lane_count = R222_STOCK_LANE_COUNT
        self._run_lock = threading.RLock()
        self._session_open = False
        self._closed = False
        self._overlap = _PersistentStepOverlap()
        self.require_full_step_overlap = bool(require_full_step_overlap)
        self._workers: list[_PersistentStockLaneWorker] = []
        for lane_id in range(self.lane_count):
            worker = _PersistentStockLaneWorker(
                lane_id, backend_factory, frontier_identity, self._overlap
            )
            worker.ready.wait()
            if worker.initialization_error is not None:
                for prior in self._workers:
                    prior.close_thread()
                raise R222StockBatchIntegrityError(
                    f"persistent lane {lane_id} AgentStart failed: {worker.initialization_error}"
                ) from worker.initialization_error
            self._workers.append(worker)
        handles = [row.topology[2] for row in self._workers]
        if len(set(handles)) != self.lane_count:
            self.close()
            raise R222StockBatchIntegrityError("persistent AgentStart handles are not unique")

    @property
    def lane_topology(self) -> tuple[tuple[int, int, int | str], ...]:
        return tuple(worker.topology for worker in self._workers)

    @staticmethod
    def _wait_tasks(
        tasks: Sequence[_PersistentOpenTask | _PersistentStepTask],
        *,
        deadline: float,
        barrier: threading.Barrier,
        cancel: threading.Event,
    ) -> None:
        deadline_hit = False
        while not all(task.done.is_set() for task in tasks):
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0.0:
                deadline_hit = True
                cancel.set()
                try:
                    barrier.abort()
                except Exception:
                    pass
                break
            next(task for task in tasks if not task.done.is_set()).done.wait(
                timeout=min(0.005, remaining)
            )
        for task in tasks:
            task.done.wait()
        errors = [task.error for task in tasks if task.error is not None]
        non_deadline = [error for error in errors if not isinstance(error, R222StockBatchDeadline)]
        if non_deadline:
            raise R222StockBatchIntegrityError(
                "persistent eight-lane worker failure: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in non_deadline)
            ) from non_deadline[0]
        if deadline_hit or errors:
            raise R222StockBatchDeadline("persistent eight-lane phase reached clean deadline")

    def open_session(
        self,
        seeds: Sequence[R222SharedTreeLaneSeed],
        *,
        root_observation: Mapping[str, Any],
        deadline_monotonic: float,
    ) -> tuple[R222PersistentLaneFrontier, ...]:
        if len(seeds) != self.lane_count:
            raise R222StockBatchIntegrityError("persistent session requires eight lane seeds")
        ordered = tuple(sorted(seeds, key=lambda row: int(row.lane_id)))
        if [row.lane_id for row in ordered] != list(range(self.lane_count)):
            raise R222StockBatchIntegrityError("persistent lane ids must be zero through seven")
        with self._run_lock:
            if self._closed or self._session_open:
                raise R222StockBatchIntegrityError("persistent stock session is unavailable")
            barrier = threading.Barrier(self.lane_count)
            cancel = threading.Event()
            tasks = [
                _PersistentOpenTask(
                    root_observation=root_observation,
                    search_inputs=seed.search_inputs,
                    phase_barrier=barrier,
                    cancel=cancel,
                    deadline=float(deadline_monotonic),
                )
                for seed in ordered
            ]
            for worker, task in zip(self._workers, tasks):
                worker.submit(task)
            try:
                self._wait_tasks(tasks, deadline=float(deadline_monotonic), barrier=barrier, cancel=cancel)
                result = tuple(task.result for task in tasks)
                if any(row is None for row in result):
                    raise R222StockBatchIntegrityError("persistent SearchBegin returned no frontier")
                self._session_open = True
                return tuple(row for row in result if row is not None)
            except BaseException:
                # Workers self-clean after an aborted phase; ``close_session``
                # joins any surviving owner-worker state before propagation.
                self._session_open = True
                try:
                    self.close_session()
                finally:
                    self._session_open = False
                raise

    def step_wave(
        self,
        commands: Sequence[R222PersistentStepCommand],
        *,
        deadline_monotonic: float,
        wave_index: int,
    ) -> tuple[R222PersistentLaneFrontier, ...]:
        if len(commands) != self.lane_count:
            raise R222StockBatchIntegrityError("persistent step wave requires exactly eight commands")
        ordered = tuple(sorted(commands, key=lambda row: int(row.lane_id)))
        if [row.lane_id for row in ordered] != list(range(self.lane_count)):
            raise R222StockBatchIntegrityError("persistent command lane ids are incomplete")
        with self._run_lock:
            if self._closed or not self._session_open:
                raise R222StockBatchIntegrityError("persistent step wave has no open session")
            barrier = threading.Barrier(self.lane_count)
            cancel = threading.Event()
            tasks = [
                _PersistentStepTask(
                    command=command,
                    phase_barrier=barrier,
                    cancel=cancel,
                    deadline=float(deadline_monotonic),
                    wave_index=int(wave_index),
                )
                for command in ordered
            ]
            for worker, task in zip(self._workers, tasks):
                worker.submit(task)
            self._wait_tasks(tasks, deadline=float(deadline_monotonic), barrier=barrier, cancel=cancel)
            overlap = self._overlap.maximum(int(wave_index))
            if self.require_full_step_overlap and overlap != self.lane_count:
                raise R222StockBatchIntegrityError(
                    f"persistent wave {wave_index} lacked eight-way native SearchStep overlap: {overlap}"
                )
            result = tuple(task.result for task in tasks)
            if any(row is None for row in result):
                raise R222StockBatchIntegrityError("persistent SearchStep returned no frontier")
            return tuple(row for row in result if row is not None)

    def submit_async_step(
        self,
        command: R222PersistentStepCommand,
        *,
        deadline_monotonic: float,
        ticket: int,
        completion_queue: queue.Queue[_PersistentStepTask],
    ) -> _PersistentStepTask:
        """Queue one lane-local retained-ID SearchStep without a wave barrier."""

        lane_id = int(command.lane_id)
        if lane_id < 0 or lane_id >= self.lane_count:
            raise R222StockBatchIntegrityError("async persistent command has invalid lane")
        with self._run_lock:
            if self._closed or not self._session_open:
                raise R222StockBatchIntegrityError("async persistent command has no open session")
            task = _PersistentStepTask(
                command=command,
                phase_barrier=threading.Barrier(1),
                cancel=threading.Event(),
                deadline=float(deadline_monotonic),
                wave_index=int(ticket),
                completion_queue=completion_queue,
            )
            self._workers[lane_id].submit(task)
            return task

    def wave_overlap(self, wave_index: int) -> int:
        return self._overlap.maximum(int(wave_index))

    def close_session(self) -> tuple[R222PersistentLaneExecution, ...]:
        with self._run_lock:
            if not self._session_open:
                return ()
            tasks = [_PersistentCloseTask() for _ in self._workers]
            for worker, task in zip(self._workers, tasks):
                worker.submit(task)
            for task in tasks:
                task.done.wait()
            self._session_open = False
            errors = [task.error for task in tasks if task.error is not None]
            if errors:
                raise R222StockBatchIntegrityError(
                    "persistent session cleanup failed: "
                    + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
                ) from errors[0]
            result = tuple(task.result for task in tasks)
            if any(row is None for row in result):
                raise R222StockBatchIntegrityError("persistent session cleanup omitted a lane")
            concrete = tuple(row for row in result if row is not None)
            if any(row.search_end_calls != 1 for row in concrete):
                raise R222StockBatchIntegrityError("persistent session omitted SearchEnd")
            return concrete

    def close(self) -> None:
        with self._run_lock:
            if self._closed:
                return
            if self._session_open:
                raise R222StockBatchIntegrityError(
                    "persistent pool close requires explicit close_session first"
                )
            self._closed = True
            for worker in self._workers:
                worker.close_thread()


class R222PersistentSharedTreeMCTS(Generic[PacketT, LeafT]):
    """Minimal two+ wave retained-ID shared-tree MCTS diagnostic.

    It is intentionally not a production BeliefMCTS replacement.  It proves
    the critical topology: each branching decision opens eight owner-thread
    SearchBegin sessions once, advances one master-reserved edge per lane per
    wave, forwards all eight leaves together, backs every leaf into one tree,
    and only then advances the retained lane-local SearchIds again.
    """

    def __init__(
        self,
        *,
        tree: R222SharedLogicalMCTSTree,
        session_pool: R222PersistentStockSessionPool,
        leaf_broker: Callable[..., Sequence[LeafT]],
        root_observation: Mapping[str, Any],
        root_actor: int,
        direct_policy_action: Sequence[int],
    ) -> None:
        self.tree = tree
        self.session_pool = session_pool
        self.leaf_broker = leaf_broker
        self.root_observation = dict(root_observation)
        self.root_actor = int(root_actor)
        self.direct_policy_action = tuple(int(item) for item in direct_policy_action)
        self._transaction_counter = 0

    def run_persistent(
        self,
        seeds: Sequence[R222SharedTreeLaneSeed],
        *,
        deadline_monotonic: float,
        max_waves: int = 2,
        make_leaf_work: Callable[[R222SharedTreeReservation, R222PersistentLaneFrontier], R222SharedTreeLeafWork[PacketT]],
        decode_model_leaf: Callable[[R222SharedTreeReservation, R222PersistentLaneFrontier, LeafT], R222SharedTreeLeaf],
    ) -> R222PersistentDecisionReceipt:
        if int(max_waves) < 2:
            raise ValueError("persistent r222 diagnostic requires at least two waves")
        if len(seeds) != R222_STOCK_LANE_COUNT:
            raise R222StockBatchIntegrityError("persistent r222 decision requires eight seeds")
        ordered_seeds = tuple(sorted(seeds, key=lambda row: int(row.lane_id)))
        if [row.lane_id for row in ordered_seeds] != list(range(R222_STOCK_LANE_COUNT)):
            raise R222StockBatchIntegrityError("persistent r222 seed ids are incomplete")
        self._transaction_counter += 1
        transaction_id = f"{self.tree.tree_id}:persistent:{self._transaction_counter:08d}"
        root_before = self.tree.root_visits
        decision_snapshot = self.tree.transaction_snapshot()
        deadline_exhausted = False
        opened_eight_sessions = False
        completed = 0
        wave_backups: list[int] = []
        wave_overlap: list[int] = []
        batch_sizes: list[int] = []
        contexts: dict[int, tuple[list[tuple[int, ...]], list[str], R222PersistentLaneFrontier]] = {}
        executions: tuple[R222PersistentLaneExecution, ...] = ()
        structural_error: BaseException | None = None
        try:
            try:
                opened = self.session_pool.open_session(
                    ordered_seeds,
                    root_observation=self.root_observation,
                    deadline_monotonic=float(deadline_monotonic),
                )
                opened_eight_sessions = len(opened) == R222_STOCK_LANE_COUNT
            except R222StockBatchDeadline:
                if not opened_eight_sessions:
                    raise R222StockBatchIntegrityError(
                        "deadline exhausted before all eight persistent sessions opened"
                    )
                deadline_exhausted = True
                opened = ()
            if opened:
                for seed, frontier in zip(ordered_seeds, opened):
                    if frontier.identity.world_key != seed.root_world_key:
                        raise R222StockBatchIntegrityError(
                            "SearchBegin frontier world key does not match lane seed"
                        )
                    contexts[seed.lane_id] = ([], [seed.root_world_key], frontier)
                for wave_index in range(int(max_waves)):
                    if time.monotonic() >= float(deadline_monotonic):
                        deadline_exhausted = True
                        break
                    wave_snapshot = self.tree.transaction_snapshot()
                    reservations: list[R222SharedTreeReservation] = []
                    try:
                        for seed in ordered_seeds:
                            actions, worlds, _frontier = contexts[seed.lane_id]
                            reservations.append(
                                self.tree.reserve_next_from_prefix(
                                    seed,
                                    root_actor=self.root_actor,
                                    action_prefix=actions,
                                    world_prefix=worlds,
                                )
                            )
                        commands = tuple(
                            R222PersistentStepCommand(
                                lane_id=reservation.lane_id,
                                parent_search_id=contexts[reservation.lane_id][2].current_search_id,
                                expected_parent_world_key=contexts[reservation.lane_id][2].identity.world_key,
                                expected_legal_fingerprint=contexts[reservation.lane_id][2].identity.legal_fingerprint,
                                selected_action=reservation.action_path[-1],
                                reservation=reservation,
                            )
                            for reservation in reservations
                        )
                        frontiers = self.session_pool.step_wave(
                            commands,
                            deadline_monotonic=float(deadline_monotonic),
                            wave_index=wave_index,
                        )
                        if len(frontiers) != R222_STOCK_LANE_COUNT:
                            raise R222StockBatchIntegrityError("persistent wave returned partial lanes")
                        works = tuple(
                            make_leaf_work(reservation, frontier)
                            for reservation, frontier in zip(reservations, frontiers)
                        )
                        for work in works:
                            work.validate()
                            if work.model_packet is None:
                                raise R222StockBatchIntegrityError(
                                    "persistent eight-head wave requires all eight frozen leaf rows"
                                )
                        packets = tuple(work.model_packet for work in works)
                        try:
                            outputs = tuple(
                                self.leaf_broker(packets, deadline_monotonic=float(deadline_monotonic))
                            )
                        except TypeError:
                            outputs = tuple(self.leaf_broker(packets))
                        if len(outputs) != R222_STOCK_LANE_COUNT:
                            raise R222StockBatchIntegrityError("persistent frozen leaf batch was incomplete")
                        leaves: list[R222SharedTreeLeaf] = []
                        for reservation, frontier, output in zip(reservations, frontiers, outputs):
                            leaf = decode_model_leaf(reservation, frontier, output)
                            leaf.validate()
                            if leaf.semantic_state_key != frontier.identity.world_key:
                                raise R222StockBatchIntegrityError(
                                    "decoded leaf semantic key differs from retained native frontier"
                                )
                            leaves.append(leaf)
                        for reservation, leaf in zip(reservations, leaves):
                            self.tree.backup(reservation, leaf)
                        if self.tree.outstanding_reservations or self.tree.outstanding_virtual_loss:
                            raise R222StockBatchIntegrityError("persistent wave leaked reservations")
                        completed += R222_STOCK_LANE_COUNT
                        wave_backups.append(R222_STOCK_LANE_COUNT)
                        wave_overlap.append(self.session_pool.wave_overlap(wave_index))
                        batch_sizes.append(R222_STOCK_LANE_COUNT)
                        for reservation, frontier, leaf in zip(reservations, frontiers, leaves):
                            actions, worlds, _previous = contexts[reservation.lane_id]
                            contexts[reservation.lane_id] = (
                                actions + [reservation.action_path[-1]],
                                worlds + [leaf.semantic_state_key],
                                frontier,
                            )
                        if any(not leaf.expandable for leaf in leaves):
                            break
                    except R222StockBatchDeadline:
                        self.tree.restore_transaction_snapshot(wave_snapshot)
                        deadline_exhausted = True
                        break
                    except BaseException:
                        self.tree.restore_transaction_snapshot(wave_snapshot)
                        raise
        except BaseException as exc:
            structural_error = exc
        finally:
            try:
                executions = self.session_pool.close_session()
            except BaseException as cleanup_exc:
                structural_error = structural_error or cleanup_exc
        if structural_error is not None:
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError(
                f"persistent shared-tree session failed structurally: {type(structural_error).__name__}: {structural_error}"
            ) from structural_error
        if self.tree.outstanding_reservations or self.tree.outstanding_virtual_loss:
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError("persistent shared-tree session leaked reservations")
        if any(
            len(row.search_id_chain) != row.search_step_calls + 1
            or len(set(row.search_id_chain)) != len(row.search_id_chain)
            for row in executions
        ):
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError(
                "persistent session SearchId chain proves reopen or lane-local reuse"
            )
        root_actions = tuple(edge.action for edge in self.tree._root.edges)
        if self.direct_policy_action not in root_actions:
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError("persistent direct-policy action is not root legal")
        if completed:
            selected, selected_visits = self.tree.selected_root_action()
            direct_fallback = False
            selected_fully_backed = True
        elif deadline_exhausted:
            selected = self.direct_policy_action
            selected_visits = 0
            direct_fallback = True
            selected_fully_backed = False
        else:
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError("persistent shared-tree session ended without backup")
        status = (
            "clean_deadline_zero_backup_fallback"
            if deadline_exhausted and not completed
            else "clean_deadline_with_backed_action"
            if deadline_exhausted
            else "persistent_eight_lane_complete"
        )
        per_lane = tuple(
            {
                "lane_id": row.lane_id,
                "handle_identity": row.handle_identity,
                "root_search_id": row.root_search_id,
                "final_search_id": row.final_search_id,
                "search_id_chain": row.search_id_chain,
                "search_begin_calls": row.search_begin_calls,
                "search_step_calls": row.search_step_calls,
                "search_release_calls": row.search_release_calls,
                "search_end_calls": row.search_end_calls,
                "action_depth": len(row.action_path),
            }
            for row in executions
        )
        return R222PersistentDecisionReceipt(
            schema=R222_STOCK_SHARED_TREE_BATCH_SCHEMA,
            transaction_id=transaction_id,
            status=status,
            success_marker=(
                "R222_PERSISTENT_EIGHT_LANE_DECISION_OK"
                if status == "persistent_eight_lane_complete"
                else None
            ),
            requested_lane_count=R222_STOCK_LANE_COUNT,
            active_lane_count=len(executions),
            shared_logical_tree=True,
            shared_logical_tree_id=self.tree.tree_id,
            unique_raw_handle_count=len({row.handle_identity for row in executions}),
            search_begin_calls=sum(row.search_begin_calls for row in executions),
            search_step_calls=sum(row.search_step_calls for row in executions),
            search_release_calls=sum(row.search_release_calls for row in executions),
            search_end_calls=sum(row.search_end_calls for row in executions),
            root_reopen_count=0,
            root_replay_count=0,
            retained_search_id_across_waves=all(
                row.search_begin_calls == 1 and row.search_step_calls >= len(wave_backups)
                for row in executions
            ),
            wave_count=len(wave_backups),
            wave_backups=tuple(wave_backups),
            wave_step_overlap=tuple(wave_overlap),
            leaf_microbatch_sizes=tuple(batch_sizes),
            completed_backed_simulations=completed,
            root_visits_before=root_before,
            root_visits_after=self.tree.root_visits,
            selected_action=selected,
            selected_action_legal=selected in root_actions,
            selected_action_fully_backed_up=selected_fully_backed,
            selected_action_visit_count=selected_visits,
            deadline_exhausted=deadline_exhausted,
            clean_deadline_zero_backup_fallback=deadline_exhausted and not completed,
            direct_policy_fallback=direct_fallback,
            opened_eight_sessions=opened_eight_sessions,
            outstanding_reservations=self.tree.outstanding_reservations,
            outstanding_virtual_loss=self.tree.outstanding_virtual_loss,
            private_random_outcome_samples=0,
            guessed_random_rules_or_successors=0,
            unobserved_random_outcome_advances=0,
            all_lane_work_finished_before_return=True,
            per_lane=per_lane,
        )


class R222AsyncPersistentSharedTreeMCTS(Generic[PacketT, LeafT]):
    """Eight retained search heads driven by a completion queue, not waves.

    The master is the only tree owner.  A worker owns its arena/SearchIds and
    publishes one completed frontier at a time.  The master evaluates whatever
    frontiers are ready (up to eight), backs them into the same tree in lane
    order, then immediately schedules those lanes again while slower native
    calls remain in flight.
    """

    def __init__(
        self,
        *,
        tree: R222SharedLogicalMCTSTree,
        session_pool: R222PersistentStockSessionPool,
        leaf_broker: Callable[..., Sequence[LeafT]],
        root_observation: Mapping[str, Any],
        root_actor: int,
    ) -> None:
        self.tree = tree
        self.session_pool = session_pool
        self.leaf_broker = leaf_broker
        self.root_observation = dict(root_observation)
        self.root_actor = int(root_actor)
        self._transaction_counter = 0

    def run_async(
        self,
        seeds: Sequence[R222SharedTreeLaneSeed],
        *,
        deadline_monotonic: float,
        steps_per_lane: int = 2,
        make_leaf_work: Callable[[R222SharedTreeReservation, R222PersistentLaneFrontier], R222SharedTreeLeafWork[PacketT]],
        decode_model_leaf: Callable[[R222SharedTreeReservation, R222PersistentLaneFrontier, LeafT], R222SharedTreeLeaf],
    ) -> R222AsyncPersistentDecisionReceipt:
        """Run a bounded async retained-ID probe for one logical decision.

        ``steps_per_lane`` is intentionally small: this is a local viability
        loop, not a replacement for production MCTS control.  It requires two
        or more actual retained-ID descents per lane so root reopen/replay
        cannot masquerade as asynchronous search.
        """

        if int(steps_per_lane) < 2:
            raise ValueError("async persistent probe requires at least two steps per lane")
        if len(seeds) != R222_STOCK_LANE_COUNT:
            raise R222StockBatchIntegrityError("async persistent probe requires eight seeds")
        ordered_seeds = tuple(sorted(seeds, key=lambda row: int(row.lane_id)))
        if [row.lane_id for row in ordered_seeds] != list(range(R222_STOCK_LANE_COUNT)):
            raise R222StockBatchIntegrityError("async persistent seed ids are incomplete")

        self._transaction_counter += 1
        transaction_id = f"{self.tree.tree_id}:async:{self._transaction_counter:08d}"
        decision_snapshot = self.tree.transaction_snapshot()
        contexts: dict[int, tuple[list[tuple[int, ...]], list[str], R222PersistentLaneFrontier]] = {}
        seeds_by_lane = {int(seed.lane_id): seed for seed in ordered_seeds}
        steps_by_lane = {lane_id: 0 for lane_id in range(R222_STOCK_LANE_COUNT)}
        completion_queue: queue.Queue[_PersistentStepTask] = queue.Queue()
        inflight: dict[int, tuple[_PersistentStepTask, R222SharedTreeReservation]] = {}
        batch_sizes: list[int] = []
        completed = 0
        executions: tuple[R222PersistentLaneExecution, ...] = ()
        structural_error: BaseException | None = None

        def schedule(lane_id: int, ticket: int) -> None:
            if lane_id in inflight:
                raise R222StockBatchIntegrityError("async lane already has an in-flight SearchStep")
            seed = seeds_by_lane[lane_id]
            actions, worlds, frontier = contexts[lane_id]
            reservation = self.tree.reserve_next_from_prefix(
                seed,
                root_actor=self.root_actor,
                action_prefix=actions,
                world_prefix=worlds,
            )
            command = R222PersistentStepCommand(
                lane_id=lane_id,
                parent_search_id=frontier.current_search_id,
                expected_parent_world_key=frontier.identity.world_key,
                expected_legal_fingerprint=frontier.identity.legal_fingerprint,
                selected_action=reservation.action_path[-1],
                reservation=reservation,
            )
            try:
                task = self.session_pool.submit_async_step(
                    command,
                    deadline_monotonic=float(deadline_monotonic),
                    ticket=ticket,
                    completion_queue=completion_queue,
                )
            except BaseException:
                self.tree.abort((reservation,))
                raise
            inflight[lane_id] = (task, reservation)

        try:
            opened = self.session_pool.open_session(
                ordered_seeds,
                root_observation=self.root_observation,
                deadline_monotonic=float(deadline_monotonic),
            )
            if len(opened) != R222_STOCK_LANE_COUNT:
                raise R222StockBatchIntegrityError("async persistent probe did not open eight sessions")
            for seed, frontier in zip(ordered_seeds, opened):
                if frontier.identity.world_key != seed.root_world_key:
                    raise R222StockBatchIntegrityError(
                        "async SearchBegin frontier world key does not match lane seed"
                    )
                contexts[int(seed.lane_id)] = ([], [seed.root_world_key], frontier)

            ticket = 0
            for lane_id in range(R222_STOCK_LANE_COUNT):
                schedule(lane_id, ticket)
                ticket += 1

            while inflight:
                remaining = float(deadline_monotonic) - time.monotonic()
                if remaining <= 0.0:
                    raise R222StockBatchDeadline("async persistent probe reached its deadline")
                try:
                    first = completion_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise R222StockBatchDeadline("async persistent probe timed out waiting for a frontier") from exc
                ready = [first]
                while len(ready) < R222_STOCK_LANE_COUNT:
                    try:
                        ready.append(completion_queue.get_nowait())
                    except queue.Empty:
                        break
                ready.sort(key=lambda task: int(task.command.lane_id))

                completed_rows: list[
                    tuple[
                        R222SharedTreeReservation,
                        R222PersistentLaneFrontier,
                        _PersistentStepTask,
                    ]
                ] = []
                for task in ready:
                    lane_id = int(task.command.lane_id)
                    expected = inflight.get(lane_id)
                    if expected is None or expected[0] is not task:
                        raise R222StockBatchIntegrityError("async completion does not match an in-flight lane")
                    inflight.pop(lane_id)
                    if task.error is not None:
                        raise R222StockBatchIntegrityError(
                            f"async persistent lane {lane_id} failed: {type(task.error).__name__}: {task.error}"
                        ) from task.error
                    if task.result is None:
                        raise R222StockBatchIntegrityError("async SearchStep returned no retained frontier")
                    completed_rows.append((expected[1], task.result, task))

                works = tuple(make_leaf_work(reservation, frontier) for reservation, frontier, _ in completed_rows)
                for work in works:
                    work.validate()
                    if work.model_packet is None:
                        raise R222StockBatchIntegrityError(
                            "async persistent probe requires a model packet for every ready frontier"
                        )
                packets = tuple(work.model_packet for work in works)
                outputs = tuple(
                    self.leaf_broker(packets, deadline_monotonic=float(deadline_monotonic))
                )
                if len(outputs) != len(completed_rows):
                    raise R222StockBatchIntegrityError("async frozen leaf microbatch was incomplete")

                leaves: list[R222SharedTreeLeaf] = []
                for (reservation, frontier, _), output in zip(completed_rows, outputs):
                    leaf = decode_model_leaf(reservation, frontier, output)
                    leaf.validate()
                    if leaf.semantic_state_key != frontier.identity.world_key:
                        raise R222StockBatchIntegrityError(
                            "async decoded leaf differs from retained native frontier"
                        )
                    leaves.append(leaf)
                for (reservation, _frontier, _), leaf in zip(completed_rows, leaves):
                    self.tree.backup(reservation, leaf)
                if self.tree.outstanding_reservations != len(inflight) or self.tree.outstanding_virtual_loss < len(inflight):
                    raise R222StockBatchIntegrityError("async shared-tree reservation accounting drifted")
                completed += len(completed_rows)
                batch_sizes.append(len(completed_rows))

                for (reservation, frontier, _), leaf in zip(completed_rows, leaves):
                    lane_id = reservation.lane_id
                    actions, worlds, _previous = contexts[lane_id]
                    contexts[lane_id] = (
                        actions + [reservation.action_path[-1]],
                        worlds + [leaf.semantic_state_key],
                        frontier,
                    )
                    steps_by_lane[lane_id] += 1
                    if steps_by_lane[lane_id] < int(steps_per_lane):
                        if not leaf.expandable:
                            raise R222StockBatchIntegrityError(
                                "async persistent lane reached a non-expandable frontier before its second step"
                            )
                        schedule(lane_id, ticket)
                        ticket += 1
        except BaseException as exc:
            structural_error = exc
        finally:
            # An outstanding worker may still be inside SearchStep when a
            # sibling/model error is discovered.  Join it before sending close
            # work; ``SearchEnd`` must never race a lane-owned native call.
            for task, _reservation in tuple(inflight.values()):
                task.done.wait()
            try:
                executions = self.session_pool.close_session()
            except BaseException as cleanup_exc:
                structural_error = structural_error or cleanup_exc

        if structural_error is not None:
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError(
                f"async persistent shared-tree session failed: {type(structural_error).__name__}: {structural_error}"
            ) from structural_error
        if self.tree.outstanding_reservations or self.tree.outstanding_virtual_loss:
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError("async persistent session leaked reservations")
        if any(
            row.search_begin_calls != 1
            or row.search_step_calls < int(steps_per_lane)
            or len(row.search_id_chain) != row.search_step_calls + 1
            or len(set(row.search_id_chain)) != len(row.search_id_chain)
            for row in executions
        ):
            self.tree.restore_transaction_snapshot(decision_snapshot)
            raise R222StockBatchIntegrityError("async persistent session did not retain valid SearchId chains")
        selected, _visits = self.tree.selected_root_action()
        root_actions = tuple(edge.action for edge in self.tree._root.edges)
        return R222AsyncPersistentDecisionReceipt(
            transaction_id=transaction_id,
            shared_logical_tree=True,
            shared_logical_tree_id=self.tree.tree_id,
            requested_lane_count=R222_STOCK_LANE_COUNT,
            active_lane_count=len(executions),
            unique_raw_handle_count=len({row.handle_identity for row in executions}),
            completed_backed_simulations=completed,
            leaf_microbatch_sizes=tuple(batch_sizes),
            selected_action=selected,
            selected_action_legal=selected in root_actions,
            selected_action_fully_backed_up=True,
            outstanding_reservations=self.tree.outstanding_reservations,
            outstanding_virtual_loss=self.tree.outstanding_virtual_loss,
            all_lane_work_finished_before_return=True,
            per_lane=tuple(
                {
                    "lane_id": row.lane_id,
                    "handle_identity": row.handle_identity,
                    "search_id_chain": row.search_id_chain,
                    "search_begin_calls": row.search_begin_calls,
                    "search_step_calls": row.search_step_calls,
                    "search_release_calls": row.search_release_calls,
                    "search_end_calls": row.search_end_calls,
                }
                for row in executions
            ),
        )


__all__ = [
    "R222_REQUIRED_STOCK_EXPORTS",
    "R222_STOCK_LANE_COUNT",
    "R222_STOCK_LIBCG_SHA256",
    "R222_STOCK_LIBCG_SIZE_BYTES",
    "R222_STOCK_SHARED_TREE_BATCH_SCHEMA",
    "R222FrozenLeafMicrobatchBroker",
    "R222AsyncPersistentDecisionReceipt",
    "R222AsyncPersistentSharedTreeMCTS",
    "R222LaneExecution",
    "R222PersistentDecisionReceipt",
    "R222PersistentFrontierIdentity",
    "R222PersistentLaneExecution",
    "R222PersistentLaneFrontier",
    "R222PersistentSharedTreeMCTS",
    "R222PersistentStepCommand",
    "R222PersistentStockSessionPool",
    "R222SharedLogicalMCTSTree",
    "R222SharedTreeBatchRequest",
    "R222SharedTreeDecisionReceipt",
    "R222SharedTreeLaneSeed",
    "R222SharedTreeLeaf",
    "R222SharedTreeLeafWork",
    "R222SharedTreeMicrobatchCoordinator",
    "R222StockBatchDeadline",
    "R222StockBatchError",
    "R222StockBatchIntegrityError",
    "R222StockLibraryReceipt",
    "R222StockSearchLanePool",
    "R222StockSharedTreeMCTS",
    "R222StockMicrobatchReceipt",
    "R222StockTrajectory",
    "attest_loaded_stock_r195_library",
    "attest_stock_r195_library",
    "canonical_observation_fingerprint",
]
