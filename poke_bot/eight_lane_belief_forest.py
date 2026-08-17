"""Eight isolated native search lanes for the staged Phase-1 candidate.

This module does not enable submission search.  It supplies the runtime that a
separately authorized profile can select after native isolation, memory,
deadline, and target-hardware receipts pass.

The important boundary is ownership: every persistent worker creates exactly
one raw ``AgentStart()`` handle on its own thread and no other thread touches
that handle.  Independent BeliefMCTS trees share only immutable request data
and a queue-owned frozen-model inference broker.  The coordinator accepts a
forest result only when all eight complete and expose the same complete ordered
root action space; otherwise the caller must use its exact direct policy.
"""

from __future__ import annotations

import copy
import hashlib
import math
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from . import cg_env, features
from .batched_infer import LeafPacket, forward_leaf_batch
from .belief import EmpiricalDeckPosterior, PublicBeliefHistory
from .belief_mcts import BeliefMCTS, NeuralBeliefPriors, information_state_fingerprint
from .matchup_adapter_activation import ShadowMatchupAdapterRouter
from .mcts import GameClock, MCTSResult
from .model import TemporalCabtTransformer
from .search_targets import build_search_target, select_by_visits


EIGHT_LANE_COUNT = 8
EIGHT_LANE_SCHEMA = "poke_bot.eight_lane_belief_forest/v1"


class EightLaneSearchError(RuntimeError):
    """The speculative forest has no action authority."""


class EightLaneDeadlineExceeded(EightLaneSearchError, TimeoutError):
    """The common forest deadline expired; all partial statistics are rejected."""


@dataclass
class _LeafRequest:
    packets: tuple[LeafPacket, ...]
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[list[LeafPacket]] = None
    error: Optional[BaseException] = None
    cancelled: bool = False


class ThreadBatchingLeafBackend:
    """One queue-owned frozen-model forward path shared by all eight lanes.

    Lane calls remain synchronous, but requests arriving in the small coalesce
    window are flattened into one model batch.  Only this broker thread invokes
    the model while a forest search is active.
    """

    source = "trained_checkpoint_policy_value_head"

    def __init__(
        self,
        model: TemporalCabtTransformer,
        *,
        checkpoint_digest: str,
        max_batch_rows: int = 32,
        coalesce_ms: float = 0.5,
    ) -> None:
        if not checkpoint_digest.startswith("sha256:"):
            raise ValueError("leaf broker requires an immutable checkpoint digest")
        if int(max_batch_rows) < EIGHT_LANE_COUNT:
            raise ValueError("eight-lane leaf broker batch must hold at least eight rows")
        if float(coalesce_ms) < 0.0:
            raise ValueError("leaf broker coalesce window must be non-negative")
        self.model = model
        self.model.eval()
        self.checkpoint_digest = checkpoint_digest
        self.max_batch_rows = int(max_batch_rows)
        self.coalesce_s = float(coalesce_ms) / 1000.0
        self._requests: queue.Queue[_LeafRequest | None] = queue.Queue()
        self._deadline = threading.local()
        self._closed = False
        self._telemetry_lock = threading.Lock()
        self._request_count = 0
        self._leaf_count = 0
        self._batch_sizes: list[int] = []
        self._thread = threading.Thread(
            target=self._serve,
            name="pokebot-eight-lane-leaf-broker",
            daemon=True,
        )
        self._thread.start()

    def set_deadline(self, deadline: Optional[float]) -> None:
        self._deadline.value = deadline

    def telemetry_mark(self) -> tuple[int, int, int]:
        with self._telemetry_lock:
            return self._request_count, self._leaf_count, len(self._batch_sizes)

    def telemetry_since(self, marker: tuple[int, int, int]) -> dict[str, Any]:
        with self._telemetry_lock:
            requests = self._request_count - int(marker[0])
            leaves = self._leaf_count - int(marker[1])
            batches = self._batch_sizes[int(marker[2]) :]
        return {
            "remote_requests": 0,
            "remote_leaves": 0,
            "queue_wait_ms_mean": 0.0,
            "queue_wait_ms_p95": 0.0,
            "inference_batch_size_mean": (
                math.fsum(batches) / len(batches) if batches else 0.0
            ),
            "inference_batch_size_p95": (
                float(sorted(batches)[min(len(batches) - 1, int(0.95 * len(batches)))])
                if batches
                else 0.0
            ),
            "server_inference_ms_mean": 0.0,
            "client_roundtrip_ms_mean": 0.0,
            "thread_batched_requests": requests,
            "thread_batched_leaves": leaves,
            "thread_batched_forwards": len(batches),
        }

    def __call__(self, packets: Sequence[LeafPacket]) -> list[LeafPacket]:
        if self._closed:
            raise EightLaneSearchError("leaf broker is closed")
        packet_tuple = tuple(packets)
        if not packet_tuple:
            return []
        deadline_value = getattr(self._deadline, "value", None)
        deadline = (
            float(deadline_value)
            if deadline_value is not None
            else float("inf")
        )
        if time.monotonic() >= deadline:
            raise EightLaneDeadlineExceeded("leaf request reached its deadline")
        request = _LeafRequest(packet_tuple, deadline)
        self._requests.put(request)
        timeout = None if math.isinf(deadline) else max(0.0, deadline - time.monotonic())
        if not request.done.wait(timeout=timeout):
            request.cancelled = True
            # The native/model forward cannot be preempted safely.  Wait for
            # the broker to observe cancellation or finish the in-flight
            # forward before unwinding the lane.  This may consume reserve
            # time, but it prevents speculative inference from continuing
            # behind the direct-policy fallback.
            request.done.wait()
            raise EightLaneDeadlineExceeded("leaf request exceeded its deadline")
        if request.error is not None:
            raise request.error
        if request.result is None or len(request.result) != len(packet_tuple):
            raise EightLaneSearchError("leaf broker returned the wrong row count")
        return request.result

    def _serve(self) -> None:
        stop_after_batch = False
        while True:
            first = self._requests.get()
            if first is None:
                return
            pending = [first]
            rows = len(first.packets)
            coalesce_deadline = time.monotonic() + self.coalesce_s
            while rows < self.max_batch_rows:
                remaining = coalesce_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    item = self._requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                pending.append(item)
                rows += len(item.packets)
            active: list[_LeafRequest] = []
            now = time.monotonic()
            for request in pending:
                if request.cancelled or now >= request.deadline:
                    request.error = EightLaneDeadlineExceeded(
                        "leaf request expired before inference"
                    )
                    request.done.set()
                else:
                    active.append(request)
            if active:
                flat = [packet for request in active for packet in request.packets]
                try:
                    evaluated = forward_leaf_batch(self.model, flat)
                    if len(evaluated) != len(flat):
                        raise EightLaneSearchError(
                            "frozen model returned the wrong batched leaf count"
                        )
                    offset = 0
                    for request in active:
                        count = len(request.packets)
                        request.result = list(evaluated[offset : offset + count])
                        offset += count
                except BaseException as exc:  # propagate the exact broker failure
                    for request in active:
                        request.error = exc
                finally:
                    with self._telemetry_lock:
                        self._request_count += len(active)
                        self._leaf_count += len(flat)
                        self._batch_sizes.append(len(flat))
                    for request in active:
                        request.done.set()
            if stop_after_batch:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join()


@dataclass
class _LaneTask:
    operation: Callable[[int, cg_env.SearchBackend, threading.Event], Any]
    cancellation: threading.Event
    barrier: threading.Barrier
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class _PersistentLaneWorker:
    def __init__(
        self,
        lane_id: int,
        backend_factory: Callable[[int], cg_env.SearchBackend],
    ) -> None:
        self.lane_id = int(lane_id)
        self._factory = backend_factory
        self._queue: queue.Queue[_LaneTask | None] = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.initialization_error: Optional[BaseException] = None
        self.backend_info: dict[str, Any] = {}
        self.thread = threading.Thread(
            target=self._run,
            name=f"pokebot-native-search-lane-{self.lane_id}",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            backend = self._factory(self.lane_id)
            self.backend_info = {
                "lane_id": self.lane_id,
                "owner_thread_id": threading.get_ident(),
                "handle_identity": getattr(backend, "handle_identity", None),
            }
        except BaseException as exc:
            self.initialization_error = exc
            self.ready.set()
            return
        self.ready.set()
        while True:
            task = self._queue.get()
            if task is None:
                return
            entered_operation = False
            try:
                remaining = task.deadline - time.monotonic()
                if remaining <= 0.0:
                    raise EightLaneDeadlineExceeded(
                        f"lane {self.lane_id} missed the common start deadline"
                    )
                task.barrier.wait(timeout=remaining)
                # Do not abort this one-shot start barrier once a lane has
                # entered its operation.  A peer may still be unwinding from
                # ``wait()``; aborting then turns a real native operation (and
                # its finally/SearchEnd cleanup) into a skipped task.  The
                # coordinator already joins every submitted task before it
                # returns a fallback, so cancellation alone is sufficient
                # after the common start gate has opened.
                entered_operation = True
                task.result = task.operation(
                    self.lane_id, backend, task.cancellation
                )
            except BaseException as exc:
                task.error = exc
                task.cancellation.set()
                # A worker can fail before reaching the common start barrier.
                # Abort it so peers wake immediately instead of waiting until
                # the decision deadline before they can clean up.
                if not entered_operation:
                    try:
                        task.barrier.abort()
                    except threading.BrokenBarrierError:
                        pass
            finally:
                task.done.set()

    def submit(self, task: _LaneTask) -> None:
        self._queue.put_nowait(task)

    def close(self) -> None:
        if self.thread.is_alive():
            self._queue.put(None)
            self.thread.join()


class PersistentEightLanePool:
    """Exactly eight dedicated workers with stable handle/thread ownership."""

    def __init__(
        self,
        backend_factory: Callable[[int], cg_env.SearchBackend],
        *,
        lane_count: int = EIGHT_LANE_COUNT,
    ) -> None:
        if int(lane_count) != EIGHT_LANE_COUNT:
            raise ValueError("the staged candidate requires exactly eight lanes")
        self.lane_count = EIGHT_LANE_COUNT
        self._closed = False
        self._run_lock = threading.Lock()
        self._workers: list[_PersistentLaneWorker] = []
        # Start and attest one worker at a time so AgentStart itself is not a
        # concurrent initialization experiment.  Parallelism begins only after
        # all eight owner handles exist.
        for lane_id in range(self.lane_count):
            worker = _PersistentLaneWorker(lane_id, backend_factory)
            worker.ready.wait()
            if worker.initialization_error is not None:
                for prior in self._workers:
                    prior.close()
                raise EightLaneSearchError(
                    f"lane {lane_id} failed to initialize: "
                    f"{type(worker.initialization_error).__name__}: "
                    f"{worker.initialization_error}"
                ) from worker.initialization_error
            self._workers.append(worker)
        handles = [row.get("handle_identity") for row in self.lane_topology]
        if None not in handles and len(set(handles)) != self.lane_count:
            self.close()
            raise EightLaneSearchError("native search handles are not unique")

    @property
    def lane_topology(self) -> list[dict[str, Any]]:
        return [dict(worker.backend_info) for worker in self._workers]

    def run_all(
        self,
        operation: Callable[[int, cg_env.SearchBackend, threading.Event], Any],
        *,
        deadline_monotonic: float,
    ) -> list[Any]:
        if self._closed:
            raise EightLaneSearchError("eight-lane pool is closed")
        with self._run_lock:
            cancellation = threading.Event()
            barrier = threading.Barrier(self.lane_count)
            tasks = [
                _LaneTask(operation, cancellation, barrier, float(deadline_monotonic))
                for _ in self._workers
            ]
            for worker, task in zip(self._workers, tasks):
                worker.submit(task)
            deadline_hit = False
            while not all(task.done.is_set() for task in tasks):
                remaining = float(deadline_monotonic) - time.monotonic()
                if remaining <= 0.0:
                    deadline_hit = True
                    cancellation.set()
                    break
                if cancellation.is_set():
                    break
                # Wake promptly for failures without spinning.
                unfinished = [task for task in tasks if not task.done.is_set()]
                if not unfinished:
                    break
                unfinished[0].done.wait(timeout=min(0.01, remaining))

            if deadline_hit or cancellation.is_set():
                cancellation.set()
                # A worker that failed before the start barrier has already
                # aborted it.  Do *not* abort merely because an operation
                # failed after the gate: peers may still be returning from
                # ``wait()`` and must enter their operation/finally so their
                # native SearchEnd cleanup runs.  A real deadline remains the
                # one coordinator-side reason to break a still-waiting gate.
                if deadline_hit:
                    try:
                        barrier.abort()
                    except threading.BrokenBarrierError:
                        pass
            # Never return an action while a speculative native call is still
            # running.  A target-hardware preflight must prove this cooperative
            # cleanup is bounded before the staged path can be activated.
            for task in tasks:
                task.done.wait()
            errors = [
                (lane_id, task.error)
                for lane_id, task in enumerate(tasks)
                if task.error is not None
            ]
            if deadline_hit:
                raise EightLaneDeadlineExceeded(
                    "eight-lane deadline expired; all partial forests were discarded"
                )
            if errors:
                details = "; ".join(
                    f"lane={lane_id} {type(error).__name__}: {error}"
                    for lane_id, error in errors
                )
                raise EightLaneSearchError(
                    "one or more required search lanes failed: " + details
                ) from errors[0][1]
            return [task.result for task in tasks]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.close()


def merge_complete_root_statistics(
    lane_results: Sequence[tuple[int, MCTSResult]],
    *,
    canonical_legal_actions: Sequence[Sequence[int]],
    canonical_root_fingerprint: str,
    elapsed_s: float,
    leaf_telemetry: Optional[dict[str, Any]] = None,
) -> MCTSResult:
    """Deterministically reduce eight complete independent root trees."""

    if len(lane_results) != EIGHT_LANE_COUNT:
        raise EightLaneSearchError("forest reduction requires all eight lanes")
    lane_ids = [int(lane_id) for lane_id, _ in lane_results]
    if sorted(lane_ids) != list(range(EIGHT_LANE_COUNT)):
        raise EightLaneSearchError("forest lane ids must be exactly 0 through 7")
    canonical = [list(map(int, action)) for action in canonical_legal_actions]
    if not canonical or len({tuple(action) for action in canonical}) != len(canonical):
        raise EightLaneSearchError("canonical legal actions are empty or duplicated")
    if not str(canonical_root_fingerprint):
        raise EightLaneSearchError("canonical root fingerprint is empty")

    visits_by_lane: list[list[int]] = []
    priors_by_lane: list[list[float]] = []
    values: list[tuple[int, float, int]] = []
    lane_diagnostics: list[dict[str, Any]] = []
    for lane_id, result in sorted(lane_results, key=lambda item: int(item[0])):
        target = getattr(result, "target", None)
        if target is None or target.action_combos != canonical:
            raise EightLaneSearchError(
                f"lane {lane_id} root legal action order does not match reality"
            )
        mode = target.diagnostics.get("action_space_mode")
        if mode is not None and mode != "complete_materialized":
            raise EightLaneSearchError(
                f"lane {lane_id} returned a factorized or unknown root action space"
            )
        if (
            target.diagnostics.get("root_information_state_fingerprint")
            != str(canonical_root_fingerprint)
        ):
            raise EightLaneSearchError(
                f"lane {lane_id} root fingerprint does not match reality"
            )
        if int(target.diagnostics.get("complete_ordered_action_count", -1)) != len(
            canonical
        ):
            raise EightLaneSearchError(
                f"lane {lane_id} did not attest the complete root action count"
            )
        visits = list(target.visits)
        if (
            len(visits) != len(canonical)
            or any(type(value) is not int or value < 0 for value in visits)
            or sum(visits) <= 0
        ):
            raise EightLaneSearchError(f"lane {lane_id} returned invalid root visits")
        if list(result.select) not in canonical:
            raise EightLaneSearchError(f"lane {lane_id} selected an illegal action")
        selected_index = canonical.index(list(result.select))
        if (
            visits[selected_index] <= 0
            or target.diagnostics.get("selected_action_fully_backed_up") is not True
        ):
            raise EightLaneSearchError(
                f"lane {lane_id} selected an unbacked root action"
            )
        priors = list(target.prior or ())
        if len(priors) != len(canonical) or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in priors
        ):
            raise EightLaneSearchError(f"lane {lane_id} returned invalid root priors")
        weight = int(result.sims_run)
        if (
            weight <= 0
            or sum(visits) != weight
            or not math.isfinite(float(target.value))
            or not -1.0 <= float(target.value) <= 1.0
        ):
            raise EightLaneSearchError(f"lane {lane_id} returned no completed search")
        visits_by_lane.append(visits)
        priors_by_lane.append([float(value) for value in priors])
        values.append((int(lane_id), float(target.value), weight))
        lane_diagnostics.append(
            {
                "lane_id": int(lane_id),
                "sims_run": weight,
                "elapsed_s": float(result.elapsed_s),
                "selected_action": list(result.select),
                "root_visits": visits,
            }
        )

    aggregate_visits = [
        sum(row[action_index] for row in visits_by_lane)
        for action_index in range(len(canonical))
    ]
    aggregate_priors = [
        math.fsum(row[action_index] for row in priors_by_lane)
        / EIGHT_LANE_COUNT
        for action_index in range(len(canonical))
    ]
    total_weight = sum(weight for _, _, weight in values)
    aggregate_value = math.fsum(
        value * weight for _, value, weight in values
    ) / total_weight
    selected = select_by_visits(canonical, aggregate_visits)
    diagnostics = {
        "schema": EIGHT_LANE_SCHEMA,
        "search_semantics": "independent_root_parallel_belief_forest",
        "requested_lanes": EIGHT_LANE_COUNT,
        "completed_lanes": EIGHT_LANE_COUNT,
        "aggregate_method": "canonical_complete_root_visit_sum",
        "aggregate_tie_break": "earliest_canonical_legal_action",
        "root_action_stable": False,
        "root_stability_receipt": None,
        "lane_stability_receipts_have_aggregate_authority": False,
        "all_required_lanes_completed": True,
        "partial_lane_statistics_used": False,
        "elapsed_s": float(elapsed_s),
        "sims_run": total_weight,
        "lane_results": lane_diagnostics,
        "action_space_mode": "complete_materialized",
    }
    if leaf_telemetry:
        diagnostics["leaf_broker"] = dict(leaf_telemetry)
    target = build_search_target(
        canonical,
        aggregate_visits,
        aggregate_value,
        prior=aggregate_priors,
        diagnostics=diagnostics,
    )
    return MCTSResult(
        select=selected,
        target=target,
        sims_run=total_weight,
        elapsed_s=float(elapsed_s),
    )


class EightLaneBeliefForest:
    """Root-parallel BeliefMCTS using exactly eight native owner lanes."""

    def __init__(
        self,
        model: TemporalCabtTransformer,
        own_deck: Sequence[int],
        posterior: EmpiricalDeckPosterior,
        *,
        checkpoint_digest: str,
        model_generation: int,
        device: Any,
        min_trusted_sims: int,
        particle_count: int,
        max_context: int,
        rng: random.Random,
        backend_factory: Optional[Callable[[int], cg_env.SearchBackend]] = None,
        leaf_backend: Optional[ThreadBatchingLeafBackend] = None,
        leaf_batch_rows: int = 32,
        leaf_coalesce_ms: float = 0.5,
    ) -> None:
        if len(tuple(own_deck)) != 60:
            raise ValueError("eight-lane forest requires a 60-card deck")
        self.model = model
        self.own_deck = tuple(int(card) for card in own_deck)
        self.posterior = posterior
        self.checkpoint_digest = checkpoint_digest
        self.model_generation = int(model_generation)
        self.device = device
        self.min_trusted_sims = int(min_trusted_sims)
        self.particle_count = int(particle_count)
        self.max_context = int(max_context)
        self.rng = rng
        if backend_factory is None:
            api, sim = cg_env.prewarm_native_search_runtime()
            backend_factory = lambda lane_id: cg_env.NativeSearchLane(
                lane_id, lib=sim.lib, api_module=api
            )
        self.leaf_backend = leaf_backend or ThreadBatchingLeafBackend(
            model,
            checkpoint_digest=checkpoint_digest,
            max_batch_rows=leaf_batch_rows,
            coalesce_ms=leaf_coalesce_ms,
        )
        self._owns_leaf_backend = leaf_backend is None
        try:
            self.pool = PersistentEightLanePool(backend_factory)
        except BaseException:
            if self._owns_leaf_backend:
                self.leaf_backend.close()
            raise
        self._disabled_reason: Optional[str] = None

    @staticmethod
    def _lane_seed(nonce: int, root_fingerprint: str, lane_id: int) -> int:
        payload = f"{nonce}:{root_fingerprint}:{int(lane_id)}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def search(
        self,
        obs_dict: dict[str, Any],
        *,
        belief_history: PublicBeliefHistory,
        root_history_boards: Sequence[features.SparseVector],
        root_history_previous_actions: Sequence[Optional[features.SparseVector]],
        matchup_shadow_router: Optional[ShadowMatchupAdapterRouter],
        matchup_model_route: int,
        clock: Optional[GameClock],
        max_sims: int,
        move_time_s: float,
        temperature: float = 1.0,
    ) -> MCTSResult:
        if self._disabled_reason is not None:
            raise EightLaneSearchError(
                "eight-lane forest was permanently disabled after a prior "
                f"integrity failure: {self._disabled_reason}"
            )
        try:
            canonical_actions = features.enumerate_action_combos(obs_dict)
        except features.ActionSpaceTooLarge as exc:
            raise EightLaneSearchError(
                "eight-lane aggregate requires a complete materialized root action space"
            ) from exc
        if not canonical_actions:
            raise EightLaneSearchError("eight-lane root has no legal actions")
        configured_budget = max(0.05, float(move_time_s))
        # PolicyAgent already requested this decision's fair-share allocation
        # before entering the forest.  The forest owns only consumption; asking
        # GameClock for another allocation here would double-budget the move.
        move_budget = configured_budget
        started = time.monotonic()
        deadline = started + move_budget
        root_fingerprint = information_state_fingerprint(obs_dict)
        nonce = self.rng.getrandbits(128)

        # This one root-only auxiliary forward happens before native workers are
        # released.  Every lane receives the immutable result, avoiding eight
        # concurrent direct calls into the shared model outside the broker.
        root_probe = BeliefMCTS(
            self.model,
            self.own_deck,
            self.posterior,
            checkpoint_digest=self.checkpoint_digest,
            model_generation=self.model_generation,
            device=self.device,
            leaf_backend=self.leaf_backend,
            min_trusted_sims=self.min_trusted_sims,
            particle_count=self.particle_count,
            max_context=self.max_context,
            matchup_model_route=int(matchup_model_route),
        )
        root_neural_priors: NeuralBeliefPriors = root_probe._root_neural_priors(
            root_history_boards=root_history_boards,
            root_history_previous_actions=root_history_previous_actions,
        )
        history_snapshot = copy.deepcopy(belief_history)
        boards_snapshot = list(root_history_boards)
        actions_snapshot = list(root_history_previous_actions)
        leaf_marker = self.leaf_backend.telemetry_mark()

        def run_lane(
            lane_id: int,
            backend: cg_env.SearchBackend,
            cancellation: threading.Event,
        ) -> MCTSResult:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise EightLaneDeadlineExceeded(
                    f"lane {lane_id} had no search budget after synchronized start"
                )
            lane_router = (
                matchup_shadow_router.fork()
                if matchup_shadow_router is not None
                else None
            )
            engine = BeliefMCTS(
                None,
                self.own_deck,
                self.posterior,
                checkpoint_digest=self.checkpoint_digest,
                model_generation=self.model_generation,
                device=self.device,
                leaf_backend=self.leaf_backend,
                rng=random.Random(
                    self._lane_seed(nonce, root_fingerprint, lane_id)
                ),
                min_trusted_sims=self.min_trusted_sims,
                particle_count=self.particle_count,
                max_context=self.max_context,
                matchup_shadow_router=lane_router,
                matchup_model_route=int(matchup_model_route),
                search_backend=backend,
                stop_requested=cancellation.is_set,
            )
            return engine.search(
                obs_dict,
                belief_history=copy.deepcopy(history_snapshot),
                root_history_boards=list(boards_snapshot),
                root_history_previous_actions=list(actions_snapshot),
                clock=None,
                max_sims=int(max_sims),
                move_time_s=max(0.05, remaining),
                temperature=float(temperature),
                root_neural_priors=root_neural_priors,
            )

        try:
            raw_results = self.pool.run_all(
                run_lane,
                deadline_monotonic=deadline,
            )
            elapsed = time.monotonic() - started
            if elapsed > move_budget:
                raise EightLaneDeadlineExceeded(
                    "eight-lane forest completed after its common deadline"
                )
            result = merge_complete_root_statistics(
                list(enumerate(raw_results)),
                canonical_legal_actions=canonical_actions,
                canonical_root_fingerprint=root_fingerprint,
                elapsed_s=elapsed,
                leaf_telemetry=self.leaf_backend.telemetry_since(leaf_marker),
            )
            result.target.diagnostics["lane_topology"] = self.pool.lane_topology
            result.target.diagnostics["requested_lane_count"] = EIGHT_LANE_COUNT
            result.target.diagnostics["automatic_lane_reduction_allowed"] = False
            return result
        except EightLaneDeadlineExceeded:
            # Cooperative deadline cleanup has completed before this point.
            # A later decision may try again, but no partial tree survives.
            raise
        except EightLaneSearchError as exc:
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            raise
        except BaseException as exc:
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            raise EightLaneSearchError(
                "unexpected eight-lane integrity failure"
            ) from exc
        finally:
            if clock is not None:
                clock.consume(time.monotonic() - started)

    def close(self) -> None:
        self.pool.close()
        if self._owns_leaf_backend:
            self.leaf_backend.close()


__all__ = [
    "EIGHT_LANE_COUNT",
    "EIGHT_LANE_SCHEMA",
    "EightLaneBeliefForest",
    "EightLaneDeadlineExceeded",
    "EightLaneSearchError",
    "PersistentEightLanePool",
    "ThreadBatchingLeafBackend",
    "merge_complete_root_statistics",
]
