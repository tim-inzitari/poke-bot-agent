"""Minimal persistent asynchronous multi-lane shared-tree search.

This module is deliberately small.  Thread-affine simulator arenas keep
their native search states alive while a coordinator repeatedly:

1. reserves a legal edge from one shared tree;
2. lets the owning simulator worker advance exactly one step;
3. microbatches whichever frontier states are ready;
4. backs those values into the same tree; and
5. immediately queues the next edge for those lanes.

It is a viability implementation, not a production-strength MCTS variant.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

DEFAULT_LANE_COUNT = 2
MAX_PRINCIPAL_VARIATION_DEPTH = 8
PROVEN_TERMINAL_WIN_REVISION = 246
PROVEN_TERMINAL_WIN_STOP_REASON = (
    "proven_deterministic_terminal_win_this_turn"
)
PROVEN_TERMINAL_WIN_PROOF_KIND = (
    "exact_deterministic_simulator_terminal_win_this_turn"
)
# Compatibility name for package code which imported the former fixed width.
LANES = DEFAULT_LANE_COUNT
PacketT = TypeVar("PacketT")


class AsyncLanePoolError(RuntimeError):
    """The multi-lane search could not return a trustworthy decision."""


class AsyncDirectFallbackRequired(AsyncLanePoolError):
    """A clean decision deadline elapsed before any root backup completed."""

    def __init__(
        self,
        message: str,
        *,
        cleanup_receipt: AsyncDirectFallbackReceipt | None = None,
    ) -> None:
        super().__init__(message)
        # ``receipt`` is the concise runtime-facing spelling; retain the more
        # explicit alias for callers which distinguish cleanup from action
        # authority.  A production queue zero-backup path always supplies it.
        self.receipt = cleanup_receipt
        self.cleanup_receipt = cleanup_receipt


# Backward-compatible import name used by the r228 package entrypoint.
AsyncEightWorkerError = AsyncLanePoolError


class SimulatorArena(Protocol):
    @property
    def handle_identity(self) -> int | str: ...

    def search_begin(
        self,
        obs_dict: Mapping[str, Any],
        search_inputs: Mapping[str, Sequence[int]],
        manual_coin: bool = True,
    ) -> Any: ...

    def search_step(self, search_id: int, select: list[int]) -> Any: ...

    def search_release(self, search_id: int) -> None: ...

    def search_end(self) -> None: ...


@dataclass(frozen=True)
class DecodedLeaf:
    """One frozen-model result for a simulator frontier."""

    state_key: str
    value: float
    legal_actions: tuple[tuple[int, ...], ...]
    priors: tuple[float, ...]
    boundary: bool = False
    actor_seat: int | None = None
    # This fingerprint is deliberately separate from ``state_key``.  The
    # latter remains lane-private search identity; this optional digest is a
    # lane-independent, exact public-observation identity used only to prove a
    # deterministic continuation after search has completed.
    observation_fingerprint: str | None = None
    # Terminal authority is deliberately explicit and redundant.  A value of
    # +1 alone is still only evaluator evidence; it cannot prove a win.  The
    # queue grants the r246 override only to an exact terminal result returned
    # by the simulator, for the current root actor, before any actor/chance
    # boundary or unresolved randomness.
    terminal_result: str | None = None
    terminal_winner_seat: int | None = None
    terminal_leaf_reached: bool = False
    chance_boundary: bool = False
    actor_change_boundary: bool = False
    unresolved_randomness: bool = False

    def validate(self) -> None:
        if not self.state_key:
            raise AsyncEightWorkerError("leaf has no state key")
        if not math.isfinite(float(self.value)):
            raise AsyncEightWorkerError("leaf value is not finite")
        if self.actor_seat not in (None, 0, 1):
            raise AsyncEightWorkerError("leaf actor seat is invalid")
        if self.observation_fingerprint is not None and (
            not isinstance(self.observation_fingerprint, str)
            or not self.observation_fingerprint.strip()
        ):
            raise AsyncEightWorkerError("leaf observation fingerprint is invalid")
        for field_name, field_value in (
            ("terminal_leaf_reached", self.terminal_leaf_reached),
            ("chance_boundary", self.chance_boundary),
            ("actor_change_boundary", self.actor_change_boundary),
            ("unresolved_randomness", self.unresolved_randomness),
        ):
            if not isinstance(field_value, bool):
                raise AsyncEightWorkerError(f"leaf {field_name} is not boolean")
        if self.terminal_result not in (None, "win", "loss", "draw"):
            raise AsyncEightWorkerError("leaf terminal result is invalid")
        if isinstance(self.terminal_winner_seat, bool) or (
            self.terminal_winner_seat not in (None, 0, 1, 2)
        ):
            raise AsyncEightWorkerError("leaf terminal winner seat is invalid")
        if self.terminal_leaf_reached:
            if not self.boundary:
                raise AsyncEightWorkerError("terminal leaf is not a boundary")
            if self.terminal_result is None:
                raise AsyncEightWorkerError("terminal leaf omitted its result")
            if self.terminal_result in ("win", "loss") and (
                self.terminal_winner_seat not in (0, 1)
            ):
                raise AsyncEightWorkerError("decisive terminal leaf omitted its winner")
            if self.terminal_result == "draw" and self.terminal_winner_seat not in (
                None,
                2,
            ):
                raise AsyncEightWorkerError("draw terminal leaf named a winning seat")
        elif self.terminal_result is not None or self.terminal_winner_seat is not None:
            raise AsyncEightWorkerError("nonterminal leaf claimed a terminal result")
        if (
            self.chance_boundary
            or self.actor_change_boundary
            or self.unresolved_randomness
        ) and not self.boundary:
            raise AsyncEightWorkerError("classified leaf boundary is not closed")
        if self.boundary:
            return
        if not self.legal_actions or len(self.legal_actions) != len(self.priors):
            raise AsyncEightWorkerError("leaf legal/prior shape is invalid")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise AsyncEightWorkerError("leaf legal actions are duplicated")
        if any((not math.isfinite(float(p)) or float(p) < 0.0) for p in self.priors):
            raise AsyncEightWorkerError("leaf prior is invalid")
        if math.fsum(float(p) for p in self.priors) <= 0.0:
            raise AsyncEightWorkerError("leaf prior has no mass")


@dataclass
class _Edge:
    action: tuple[int, ...]
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    virtual_loss: int = 0
    # Simulator worlds are never merged across lanes, even if a caller
    # accidentally supplies colliding state keys.  Public fingerprints live on
    # the child node and are used only for post-search agreement checks.
    children: dict[tuple[int, str], _Node] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class _Node:
    state_key: str
    edges: list[_Edge]
    actor_seat: int | None = None
    observation_fingerprint: str | None = None
    boundary: bool = False
    lane_id: int | None = None
    visits: int = 0


@dataclass(frozen=True)
class _StepCommand:
    lane_id: int
    parent_search_id: int
    action: tuple[int, ...]
    sequence: int


@dataclass(frozen=True)
class _OpenCommand:
    observation: Mapping[str, Any]
    search_inputs: Mapping[str, Sequence[int]]


@dataclass(frozen=True)
class _CloseCommand:
    pass


@dataclass
class _WorkerResult:
    lane_id: int
    kind: str
    command_kind: str | None = None
    sequence: int = 0
    search_id: int | None = None
    observation: Any = None
    error: Exception | None = None
    started: float = 0.0
    completed: float = 0.0
    search_id_chain: tuple[int, ...] = ()
    releases: int = 0
    search_end_calls: int = 0


class _ArenaWorker:
    def __init__(
        self,
        lane_id: int,
        factory: Callable[[int], SimulatorArena],
        completions: queue.Queue[_WorkerResult],
    ) -> None:
        self.lane_id = lane_id
        self._factory = factory
        self._completions = completions
        self._commands: queue.Queue[object] = queue.Queue()
        self._ready = threading.Event()
        self._arena: SimulatorArena | None = None
        self._init_error: Exception | None = None
        self._live_ids: list[int] = []
        self._current_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"r228-async-simulator-{lane_id}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            raise AsyncEightWorkerError(
                f"lane {lane_id} arena initialization failed: {self._init_error}"
            ) from self._init_error

    @property
    def handle_identity(self) -> int | str:
        if self._arena is None:
            raise AsyncEightWorkerError("arena worker is not initialized")
        return self._arena.handle_identity

    def submit(self, command: object) -> None:
        self._commands.put(command)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def request_shutdown(self) -> None:
        self._commands.put(None)

    def wait_for_shutdown(self, *, timeout_seconds: float) -> bool:
        self._thread.join(timeout=max(0.0, float(timeout_seconds)))
        return not self._thread.is_alive()

    def shutdown(self, *, timeout_seconds: float) -> bool:
        self.request_shutdown()
        return self.wait_for_shutdown(timeout_seconds=timeout_seconds)

    def _open(self, command: _OpenCommand) -> _WorkerResult:
        if self._arena is None or self._live_ids:
            raise AsyncEightWorkerError("lane opened a second live decision")
        started = time.monotonic()
        state = self._arena.search_begin(
            command.observation,
            command.search_inputs,
            manual_coin=True,
        )
        search_id = int(state.searchId)
        self._live_ids = [search_id]
        self._current_id = search_id
        return _WorkerResult(
            lane_id=self.lane_id,
            kind="open",
            search_id=search_id,
            observation=state.observation,
            started=started,
            completed=time.monotonic(),
            search_id_chain=(search_id,),
        )

    def _step(self, command: _StepCommand) -> _WorkerResult:
        if self._arena is None or self._current_id is None:
            raise AsyncEightWorkerError("lane has no open simulator state")
        if command.parent_search_id != self._current_id:
            raise AsyncEightWorkerError("lane received a stale simulator state")
        started = time.monotonic()
        state = self._arena.search_step(self._current_id, list(command.action))
        child_id = int(state.searchId)
        if child_id in self._live_ids:
            raise AsyncEightWorkerError("lane reused a live SearchId")
        self._live_ids.append(child_id)
        self._current_id = child_id
        return _WorkerResult(
            lane_id=self.lane_id,
            kind="step",
            sequence=command.sequence,
            search_id=child_id,
            observation=state.observation,
            started=started,
            completed=time.monotonic(),
            search_id_chain=tuple(self._live_ids),
        )

    def _close(self) -> _WorkerResult:
        if self._arena is None:
            raise AsyncEightWorkerError("lane arena is absent")
        error: Exception | None = None
        releases = 0
        chain = tuple(self._live_ids)
        for search_id in reversed(self._live_ids):
            try:
                self._arena.search_release(search_id)
                releases += 1
            except Exception as exc:  # noqa: BLE001 - native cleanup must continue
                error = error or exc
        try:
            self._arena.search_end()
        except Exception as exc:  # noqa: BLE001 - native cleanup must continue
            error = error or exc
        self._live_ids = []
        self._current_id = None
        if error is not None:
            raise AsyncEightWorkerError(
                f"lane {self.lane_id} simulator cleanup failed: {error}"
            ) from error
        return _WorkerResult(
            lane_id=self.lane_id,
            kind="close",
            completed=time.monotonic(),
            search_id_chain=chain,
            releases=releases,
            search_end_calls=1,
        )

    def _run(self) -> None:
        try:
            self._arena = self._factory(self.lane_id)
        except Exception as exc:  # noqa: BLE001 - report worker initialization
            self._init_error = exc
        finally:
            self._ready.set()
        if self._init_error is not None:
            return
        while True:
            try:
                command = self._commands.get(timeout=0.100)
            except queue.Empty:
                continue
            if command is None:
                return
            result: _WorkerResult
            command_kind: str | None = None
            try:
                if isinstance(command, _OpenCommand):
                    command_kind = "open"
                    result = self._open(command)
                elif isinstance(command, _StepCommand):
                    command_kind = "step"
                    result = self._step(command)
                elif isinstance(command, _CloseCommand):
                    command_kind = "close"
                    result = self._close()
                else:
                    raise AsyncEightWorkerError("unknown simulator command")
            except Exception as exc:  # noqa: BLE001 - marshal worker failures
                result = _WorkerResult(
                    lane_id=self.lane_id,
                    kind="error",
                    command_kind=command_kind,
                    sequence=getattr(command, "sequence", 0),
                    error=exc,
                    completed=time.monotonic(),
                    search_id_chain=tuple(self._live_ids),
                )
            else:
                result.command_kind = command_kind
            self._completions.put(result)


@dataclass
class _LaneContext:
    lane_id: int
    search_id: int
    node: _Node
    path: list[_Edge]
    action_path: list[tuple[int, ...]]
    actor_path: list[int | None]
    search_id_chain: tuple[int, ...]
    in_flight: bool = False
    stopped: bool = False


@dataclass(frozen=True)
class AsyncDecisionReceipt:
    selected_action: tuple[int, ...]
    selected_action_visits: int
    selected_action_value: float
    selected_action_prior: float
    root_visits: int
    arena_count: int
    unique_handle_count: int
    search_begin_calls: int
    search_step_calls: int
    completed_backups: int
    microbatch_sizes: tuple[int, ...]
    max_simulator_calls_in_flight: int
    completion_order: tuple[int, ...]
    per_lane_depth: tuple[int, ...]
    per_lane_search_id_chains: tuple[tuple[int, ...], ...]
    per_lane_handle_identities: tuple[int | str, ...]
    distinct_search_begin_composite_count: int
    search_release_calls: int
    search_end_calls: int
    outstanding_virtual_loss: int
    stop_reason: str
    minimum_backups_before_stability: int
    stable_root_leader_observations: int
    maximum_backups_per_decision: int
    leader_stability_count: int
    elapsed_seconds: float
    root_seat: int
    principal_variation: tuple[dict[str, object], ...]
    root_actor_seat: int | None = None
    root_observation_fingerprint: str | None = None
    root_legal_order_fingerprint: str | None = None
    terminal_win_proof: dict[str, object] | None = None
    owner_proven_deterministic_terminal_win_this_turn_revision: int = (
        PROVEN_TERMINAL_WIN_REVISION
    )


@dataclass(frozen=True)
class AsyncDirectFallbackReceipt:
    """Proof that a zero-backup deadline was cleanly and fully contained."""

    arena_count: int
    unique_handle_count: int
    search_begin_calls: int
    search_step_calls: int
    completed_backups: int
    microbatch_sizes: tuple[int, ...]
    max_simulator_calls_in_flight: int
    completion_order: tuple[int, ...]
    per_lane_depth: tuple[int, ...]
    per_lane_search_id_chains: tuple[tuple[int, ...], ...]
    per_lane_handle_identities: tuple[int | str, ...]
    distinct_search_begin_composite_count: int
    search_release_calls: int
    search_end_calls: int
    outstanding_virtual_loss: int
    stop_reason: str
    minimum_backups_before_stability: int
    stable_root_leader_observations: int
    maximum_backups_per_decision: int
    leader_stability_count: int
    elapsed_seconds: float
    root_seat: int


class PersistentAsyncSharedTreeMCTS:
    """Persistent simulator lanes feeding one coordinator-owned shared tree."""

    def __init__(
        self,
        *,
        arena_factory: Callable[[int], SimulatorArena],
        make_packet: Callable[[int, Any], PacketT],
        evaluate_batch: Callable[[Sequence[PacketT]], Sequence[DecodedLeaf]],
        puct_c: float = 1.25,
        coalesce_seconds: float = 0.001,
        lane_count: int = DEFAULT_LANE_COUNT,
        minimum_backups_before_stability: int = 8,
        stable_root_leader_observations: int = 3,
        maximum_backups_per_decision: int = 32,
        cleanup_timeout_seconds: float = 1.0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        lane_count = int(lane_count)
        if lane_count < 1:
            raise ValueError("lane_count must be positive")
        minimum_backups_before_stability = int(minimum_backups_before_stability)
        stable_root_leader_observations = int(stable_root_leader_observations)
        maximum_backups_per_decision = int(maximum_backups_per_decision)
        if minimum_backups_before_stability < 1:
            raise ValueError("minimum_backups_before_stability must be positive")
        if stable_root_leader_observations < 1:
            raise ValueError("stable_root_leader_observations must be positive")
        if maximum_backups_per_decision < minimum_backups_before_stability:
            raise ValueError(
                "maximum_backups_per_decision must be at least "
                "minimum_backups_before_stability"
            )
        cleanup_timeout_seconds = float(cleanup_timeout_seconds)
        if not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0.0:
            raise ValueError("cleanup_timeout_seconds must be finite and positive")
        self._completions: queue.Queue[_WorkerResult] = queue.Queue()
        self._lane_count = lane_count
        self._minimum_backups_before_stability = minimum_backups_before_stability
        self._stable_root_leader_observations = stable_root_leader_observations
        self._maximum_backups_per_decision = maximum_backups_per_decision
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._progress_callback = progress_callback
        self._closed = False
        self._poisoned = False
        self._poison_reason: str | None = None
        self._workers = [
            _ArenaWorker(lane, arena_factory, self._completions)
            for lane in range(self._lane_count)
        ]
        self._handle_identities = tuple(
            worker.handle_identity for worker in self._workers
        )
        if len(set(self._handle_identities)) != self._lane_count:
            try:
                self.close()
            except AsyncEightWorkerError:
                pass
            raise AsyncEightWorkerError("simulator arena handles are not unique")
        self._make_packet = make_packet
        self._evaluate_batch = evaluate_batch
        self._puct_c = float(puct_c)
        self._coalesce_seconds = max(0.0, float(coalesce_seconds))

    @property
    def lane_count(self) -> int:
        return self._lane_count

    @property
    def minimum_backups_before_stability(self) -> int:
        return self._minimum_backups_before_stability

    @property
    def stable_root_leader_observations(self) -> int:
        return self._stable_root_leader_observations

    @property
    def maximum_backups_per_decision(self) -> int:
        return self._maximum_backups_per_decision

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + self._cleanup_timeout_seconds
        failures: list[str] = []
        self._emit_progress(
            "worker_shutdown_begin",
            pending_lanes=tuple(range(self._lane_count)),
        )
        for worker in self._workers:
            worker.request_shutdown()
        pending = {worker.lane_id: worker for worker in self._workers}
        while pending and time.monotonic() < deadline:
            for lane_id, worker in tuple(pending.items()):
                if not worker.is_alive:
                    pending.pop(lane_id, None)
                    self._emit_progress(
                        "worker_shutdown_result",
                        lane_id=lane_id,
                        pending_lanes=tuple(pending),
                        status="stopped",
                    )
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    stopped = worker.wait_for_shutdown(
                        timeout_seconds=remaining / max(1, len(pending))
                    )
                except Exception as exc:  # noqa: BLE001 - bounded shutdown must report failure
                    reason = f"lane {lane_id} shutdown raised: {exc}"
                    failures.append(reason)
                    self._mark_poisoned(reason)
                    self._emit_progress(
                        "worker_shutdown_error",
                        lane_id=lane_id,
                        pending_lanes=tuple(pending),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    pending.pop(lane_id, None)
                    continue
                if stopped:
                    pending.pop(lane_id, None)
                    self._emit_progress(
                        "worker_shutdown_result",
                        lane_id=lane_id,
                        pending_lanes=tuple(pending),
                        status="stopped",
                    )
        for lane_id in sorted(pending):
            reason = f"lane {lane_id} worker shutdown exceeded cleanup timeout"
            failures.append(reason)
            self._mark_poisoned(reason)
            self._emit_progress(
                "worker_shutdown_timeout",
                lane_id=lane_id,
                pending_lanes=tuple(pending),
            )
        if failures:
            raise AsyncEightWorkerError("; ".join(failures))

    def _emit_progress(
        self,
        phase: str,
        *,
        lane_id: int | None = None,
        pending_lanes: Sequence[int] = (),
        **fields: Any,
    ) -> None:
        """Best-effort diagnostics; telemetry must not alter action authority."""

        callback = self._progress_callback
        if callback is None:
            return
        payload: dict[str, Any] = {
            "schema": "poke_bot.r238_two_lane_kaggle_viability/v1",
            "record_type": "queue_progress",
            "phase": str(phase),
            "monotonic_seconds": time.monotonic(),
            "pending_lanes": [int(lane) for lane in sorted(set(pending_lanes))],
        }
        if lane_id is not None:
            payload["lane_id"] = int(lane_id)
        payload.update(fields)
        try:
            callback(payload)
        except Exception:
            # A diagnostics sink must never block or change the submitted action.
            return

    def _mark_poisoned(self, reason: str) -> None:
        if not self._poisoned:
            self._poison_reason = str(reason)
        self._poisoned = True
        self._emit_progress("pool_poisoned", reason=str(reason))

    @staticmethod
    def _node(
        state_key: str,
        actions: Sequence[Sequence[int]],
        priors: Sequence[float],
        *,
        actor_seat: int | None,
        observation_fingerprint: str | None = None,
        boundary: bool = False,
        lane_id: int | None = None,
    ) -> _Node:
        normalized_actions = tuple(tuple(int(i) for i in action) for action in actions)
        normalized_priors = tuple(float(p) for p in priors)
        if not normalized_actions or len(normalized_actions) != len(normalized_priors):
            raise AsyncEightWorkerError("tree node action/prior shape is invalid")
        total = math.fsum(normalized_priors)
        if total <= 0.0:
            raise AsyncEightWorkerError("tree node prior has no mass")
        return _Node(
            state_key=state_key,
            actor_seat=actor_seat,
            observation_fingerprint=observation_fingerprint,
            boundary=bool(boundary),
            lane_id=lane_id,
            edges=[
                _Edge(action=action, prior=prior / total)
                for action, prior in zip(normalized_actions, normalized_priors)
            ],
        )

    def _reserve(self, node: _Node, *, root_seat: int) -> _Edge:
        parent_scale = math.sqrt(max(1, node.visits + sum(e.virtual_loss for e in node.edges)))
        direction = 1.0 if node.actor_seat in (None, root_seat) else -1.0
        edge = max(
            node.edges,
            key=lambda row: (
                direction * row.q
                + self._puct_c
                * row.prior
                * parent_scale
                / (1 + row.visits + row.virtual_loss),
                row.prior,
                tuple(-item for item in row.action),
            ),
        )
        edge.virtual_loss += 1
        return edge

    @staticmethod
    def _root_leader(root: _Node) -> _Edge | None:
        backed = [edge for edge in root.edges if edge.visits > 0]
        if not backed:
            return None
        return max(
            backed,
            key=lambda edge: (
                edge.visits,
                edge.q,
                edge.prior,
                tuple(-item for item in edge.action),
            ),
        )

    def _principal_variation(
        self,
        *,
        selected_root_edge: _Edge,
        root_seat: int,
        maximum_depth: int = MAX_PRINCIPAL_VARIATION_DEPTH,
    ) -> tuple[dict[str, object], ...]:
        """Return only the continuation independently proved by both lanes.

        Search nodes stay lane-private.  This method merely walks the two
        independent backed paths after search and emits a subsequent action
        when the exact public observation and deterministic backed leader agree
        in both worlds.  Any uncertainty truncates the continuation.
        """

        if self._lane_count != DEFAULT_LANE_COUNT or maximum_depth < 1:
            return ()
        lane_edges = {
            lane_id: selected_root_edge for lane_id in range(self._lane_count)
        }
        continuation: list[dict[str, object]] = []
        for _depth in range(int(maximum_depth)):
            lane_nodes: dict[int, _Node] = {}
            for lane_id, edge in lane_edges.items():
                matches = [
                    child
                    for child in edge.children.values()
                    if child.lane_id == lane_id and child.visits > 0
                ]
                if len(matches) != 1:
                    return tuple(continuation)
                lane_nodes[lane_id] = matches[0]

            fingerprints = {
                node.observation_fingerprint for node in lane_nodes.values()
            }
            if None in fingerprints or len(fingerprints) != 1:
                return tuple(continuation)
            if any(
                node.boundary or node.actor_seat != root_seat
                for node in lane_nodes.values()
            ):
                return tuple(continuation)

            lane_leaders = {
                lane_id: self._root_leader(node)
                for lane_id, node in lane_nodes.items()
            }
            if any(leader is None for leader in lane_leaders.values()):
                return tuple(continuation)
            actions = {
                leader.action
                for leader in lane_leaders.values()
                if leader is not None
            }
            if len(actions) != 1:
                return tuple(continuation)
            action = next(iter(actions))
            fingerprint = next(iter(fingerprints))
            if not isinstance(fingerprint, str) or not fingerprint:
                return tuple(continuation)
            continuation.append(
                {
                    "observation_fingerprint": fingerprint,
                    # This is an edge from the exact decoded legal set and has
                    # at least one backup by construction of ``_root_leader``.
                    "action": [int(item) for item in action],
                }
            )
            lane_edges = {
                lane_id: leader
                for lane_id, leader in lane_leaders.items()
                if leader is not None
            }
        return tuple(continuation)

    @staticmethod
    def _backup(context: _LaneContext, edge: _Edge, leaf: DecodedLeaf) -> _Node:
        if edge.virtual_loss < 1:
            raise AsyncEightWorkerError("shared-tree virtual loss underflow")
        edge.virtual_loss -= 1
        for visited in (*context.path, edge):
            visited.visits += 1
            visited.value_sum += float(leaf.value)
        # The nodes are coordinator-only.  The composite key enforces lane
        # privacy even if a malformed caller reuses a state key across worlds.
        child_key = (context.lane_id, leaf.state_key)
        child = edge.children.get(child_key)
        if child is None:
            child = (
                _Node(
                    state_key=leaf.state_key,
                    edges=[],
                    actor_seat=leaf.actor_seat,
                    observation_fingerprint=leaf.observation_fingerprint,
                    boundary=True,
                    lane_id=context.lane_id,
                )
                if leaf.boundary
                else PersistentAsyncSharedTreeMCTS._node(
                    leaf.state_key, leaf.legal_actions, leaf.priors,
                    actor_seat=leaf.actor_seat,
                    observation_fingerprint=leaf.observation_fingerprint,
                    lane_id=context.lane_id,
                )
            )
            edge.children[child_key] = child
        elif (
            child.lane_id != context.lane_id
            or child.actor_seat != leaf.actor_seat
            or child.boundary != bool(leaf.boundary)
            or child.observation_fingerprint != leaf.observation_fingerprint
        ):
            raise AsyncEightWorkerError(
                f"lane {context.lane_id} returned conflicting leaf identity"
            )
        child.visits += 1
        return child

    @staticmethod
    def _proven_terminal_win_this_turn(
        *,
        context: _LaneContext,
        leaf: DecodedLeaf,
        root: _Node,
        root_actor_seat: int,
        root_observation_fingerprint: str | None,
        root_legal_order_fingerprint: str | None,
    ) -> dict[str, object] | None:
        """Build one exact r246 proof, or return no selection authority.

        This is intentionally opportunistic: it examines only a terminal leaf
        reached by ordinary two-lane search.  It neither scans unvisited root
        actions nor treats a model value as a terminal result.
        """

        if (
            not leaf.terminal_leaf_reached
            or leaf.terminal_result != "win"
            or leaf.terminal_winner_seat != root_actor_seat
            or leaf.chance_boundary
            or leaf.actor_change_boundary
            or leaf.unresolved_randomness
            or not leaf.boundary
            or not context.action_path
            or len(context.actor_path) != len(context.action_path)
            or any(actor != root_actor_seat for actor in context.actor_path)
            or not isinstance(root_observation_fingerprint, str)
            or not root_observation_fingerprint
            or not isinstance(root_legal_order_fingerprint, str)
            or not root_legal_order_fingerprint
        ):
            return None
        root_action = context.action_path[0]
        matching_root_edges = [edge for edge in root.edges if edge.action == root_action]
        if (
            len(matching_root_edges) != 1
            or matching_root_edges[0].visits < 1
            or root_actor_seat != root.actor_seat
        ):
            return None
        serialized_action = [int(item) for item in root_action]
        return {
            "proof_kind": PROVEN_TERMINAL_WIN_PROOF_KIND,
            "root_observation_fingerprint": root_observation_fingerprint,
            "root_legal_order_fingerprint": root_legal_order_fingerprint,
            "root_actor_seat": root_actor_seat,
            "root_action": serialized_action,
            "selected_action": list(serialized_action),
            "terminal_result": "win",
            "terminal_winner_seat": root_actor_seat,
            "terminal_leaf_reached": True,
            "proof_path_action_count": len(context.action_path),
            "path_actor_seats": [int(actor) for actor in context.actor_path],
            "path_no_actor_change_boundary": True,
            "path_no_opponent_boundary_crossing": True,
            "path_no_chance_boundary": True,
            "path_no_unresolved_randomness": True,
            "proof_is_deterministic": True,
            "discovering_lane_id": context.lane_id,
        }

    def run_decision(
        self,
        *,
        root_observation: Mapping[str, Any],
        search_inputs: Sequence[Mapping[str, Sequence[int]]],
        root_state_key: str,
        root_actions: Sequence[Sequence[int]],
        root_priors: Sequence[float],
        root_seat: int,
        deadline_monotonic: float,
        root_observation_fingerprint: str | None = None,
        root_legal_order_fingerprint: str | None = None,
        root_actor_seat: int | None = None,
        smoke_min_depth_per_lane: int | None = None,
    ) -> AsyncDecisionReceipt:
        """Search one decision until deadline or the optional smoke criterion."""

        if self._closed:
            raise AsyncEightWorkerError("asynchronous worker pool is closed")
        if self._poisoned:
            raise AsyncEightWorkerError(
                "asynchronous worker pool is poisoned; process recycle required: "
                f"{self._poison_reason or 'bounded cleanup failed'}"
            )
        if len(search_inputs) != self._lane_count:
            raise AsyncEightWorkerError(
                f"exactly {self._lane_count} search-input rows are required"
            )
        started = time.monotonic()
        if root_seat not in (0, 1):
            raise AsyncEightWorkerError("root seat must be 0 or 1")
        if root_actor_seat is None:
            root_actor_seat = root_seat
        if (
            isinstance(root_actor_seat, bool)
            or not isinstance(root_actor_seat, int)
            or root_actor_seat not in (0, 1)
            or root_actor_seat != root_seat
        ):
            raise AsyncEightWorkerError("root actor seat does not match root seat")
        for field_name, fingerprint in (
            ("root observation", root_observation_fingerprint),
            ("root legal order", root_legal_order_fingerprint),
        ):
            if fingerprint is not None and (
                not isinstance(fingerprint, str) or not fingerprint.strip()
            ):
                raise AsyncEightWorkerError(f"{field_name} fingerprint is invalid")
        if (root_observation_fingerprint is None) != (
            root_legal_order_fingerprint is None
        ):
            raise AsyncEightWorkerError(
                "root observation/legal fingerprints must be supplied together"
            )
        root = self._node(
            root_state_key, root_actions, root_priors, actor_seat=root_seat
        )
        self._emit_progress(
            "decision_begin",
            pending_lanes=tuple(range(self._lane_count)),
            deadline_monotonic=float(deadline_monotonic),
            cleanup_timeout_seconds=self._cleanup_timeout_seconds,
            lane_count=self._lane_count,
            minimum_backups_before_stability=(
                self._minimum_backups_before_stability
            ),
            stable_root_leader_observations=(
                self._stable_root_leader_observations
            ),
            maximum_backups_per_decision=self._maximum_backups_per_decision,
            per_lane_handle_identities=list(self._handle_identities),
        )
        for worker, inputs in zip(self._workers, search_inputs):
            worker.submit(_OpenCommand(root_observation, inputs))
            self._emit_progress(
                "open_submit",
                lane_id=worker.lane_id,
                pending_lanes=tuple(range(self._lane_count)),
            )
        open_rows: dict[int, _WorkerResult] = {}
        while len(open_rows) < self._lane_count:
            remaining = float(deadline_monotonic) - time.monotonic()
            if remaining <= 0.0:
                raise AsyncEightWorkerError(
                    "decision deadline expired before all simulator lanes opened"
                )
            row = self._completions.get(timeout=remaining)
            self._emit_progress(
                "open_result",
                lane_id=row.lane_id,
                pending_lanes=tuple(
                    lane for lane in range(self._lane_count) if lane not in open_rows
                ),
                result_kind=row.kind,
                command_kind=row.command_kind,
                error_type=(type(row.error).__name__ if row.error is not None else None),
            )
            if row.error is not None or row.kind != "open" or row.search_id is None:
                raise AsyncEightWorkerError(
                    f"lane {row.lane_id} SearchBegin failed: {row.error or row.kind}"
                )
            if row.lane_id in open_rows:
                raise AsyncEightWorkerError(
                    f"duplicate SearchBegin completion for lane {row.lane_id}"
                )
            open_rows[row.lane_id] = row
        if sorted(open_rows) != list(range(self._lane_count)):
            raise AsyncEightWorkerError("SearchBegin lane set is incomplete")
        search_begin_composites = tuple(
            (self._handle_identities[lane], int(open_rows[lane].search_id))
            for lane in range(self._lane_count)
        )
        distinct_search_begin_composite_count = len(set(search_begin_composites))
        if distinct_search_begin_composite_count != self._lane_count:
            raise AsyncEightWorkerError(
                "SearchBegin handle/id composite set is incomplete"
            )

        contexts = {
            lane: _LaneContext(
                lane_id=lane,
                search_id=int(open_rows[lane].search_id),
                node=root,
                path=[],
                action_path=[],
                actor_path=[],
                search_id_chain=open_rows[lane].search_id_chain,
            )
            for lane in range(self._lane_count)
        }
        in_flight: dict[int, tuple[_LaneContext, _Edge, int]] = {}
        sequence = 0
        completed_backups = 0
        microbatches: list[int] = []
        completion_order: list[int] = []
        peak_in_flight = 0
        stop_reason: str | None = None
        last_leader_action: tuple[int, ...] | None = None
        leader_stability_count = 0
        terminal_win_proof: dict[str, object] | None = None

        def dispatch(context: _LaneContext) -> None:
            nonlocal sequence, peak_in_flight
            if context.stopped or context.in_flight or not context.node.edges:
                return
            if (
                completed_backups + len(in_flight)
                >= self._maximum_backups_per_decision
            ):
                return
            edge = self._reserve(context.node, root_seat=root_seat)
            sequence += 1
            context.in_flight = True
            in_flight[context.lane_id] = (context, edge, sequence)
            self._workers[context.lane_id].submit(
                _StepCommand(
                    lane_id=context.lane_id,
                    parent_search_id=context.search_id,
                    action=edge.action,
                    sequence=sequence,
                )
            )
            self._emit_progress(
                "step_submit",
                lane_id=context.lane_id,
                pending_lanes=tuple(in_flight),
                sequence=sequence,
                search_id=context.search_id,
            )
            peak_in_flight = max(peak_in_flight, len(in_flight))

        for context in contexts.values():
            dispatch(context)

        structural_error: Exception | None = None
        close_rows: dict[int, _WorkerResult] = {}

        def remember_error(error: Exception) -> None:
            nonlocal structural_error
            structural_error = structural_error or error

        def release_reservation(*, lane_id: int, edge: _Edge, phase: str) -> None:
            if edge.virtual_loss < 1:
                remember_error(
                    AsyncEightWorkerError(
                        f"lane {lane_id} lost its shared-tree reservation during {phase}"
                    )
                )
                return
            edge.virtual_loss -= 1
            self._emit_progress(
                "reservation_released",
                lane_id=lane_id,
                pending_lanes=tuple(in_flight),
                cleanup_phase=phase,
            )

        try:
            while in_flight and time.monotonic() < float(deadline_monotonic):
                remaining = float(deadline_monotonic) - time.monotonic()
                try:
                    first = self._completions.get(timeout=max(0.0, remaining))
                except queue.Empty:
                    break
                ready = [first]
                coalesce_until = min(
                    float(deadline_monotonic), time.monotonic() + self._coalesce_seconds
                )
                while len(ready) < self._lane_count:
                    wait = coalesce_until - time.monotonic()
                    if wait <= 0.0:
                        break
                    try:
                        ready.append(self._completions.get(timeout=wait))
                    except queue.Empty:
                        break
                consumed_rows: list[tuple[_WorkerResult, _LaneContext, _Edge]] = []
                try:
                    # A completion has already been consumed from the queue.  Remove its
                    # reservation before packet construction/evaluation so an evaluator
                    # failure can never wait for that same completion during cleanup.
                    for row in ready:
                        if row.lane_id not in in_flight:
                            raise AsyncEightWorkerError("received an untracked simulator result")
                        context, edge, expected_sequence = in_flight.pop(row.lane_id)
                        context.in_flight = False
                        consumed_rows.append((row, context, edge))
                        self._emit_progress(
                            "step_result",
                            lane_id=row.lane_id,
                            pending_lanes=tuple(in_flight),
                            sequence=row.sequence,
                            result_kind=row.kind,
                            command_kind=row.command_kind,
                            error_type=(
                                type(row.error).__name__
                                if row.error is not None
                                else None
                            ),
                        )
                        if row.sequence != expected_sequence:
                            raise AsyncEightWorkerError(
                                f"lane {row.lane_id} returned stale SearchStep sequence "
                                f"{row.sequence}, expected {expected_sequence}"
                            )
                    for row, _context, _edge in consumed_rows:
                        if row.error is not None or row.kind != "step" or row.search_id is None:
                            raise AsyncEightWorkerError(
                                f"lane {row.lane_id} SearchStep failed: {row.error or row.kind}"
                            )
                    packets = [
                        self._make_packet(row.lane_id, row.observation)
                        for row, _context, _edge in consumed_rows
                    ]
                    self._emit_progress(
                        "eval_begin",
                        pending_lanes=tuple(in_flight),
                        lanes=[row.lane_id for row, _context, _edge in consumed_rows],
                    )
                    leaves = tuple(self._evaluate_batch(packets))
                    if len(leaves) != len(consumed_rows):
                        raise AsyncEightWorkerError("GPU evaluator returned a partial microbatch")
                    for leaf in leaves:
                        leaf.validate()
                except Exception as exc:
                    for row, _context, edge in consumed_rows:
                        release_reservation(
                            lane_id=row.lane_id,
                            edge=edge,
                            phase="eval_failure",
                        )
                    self._emit_progress(
                        "eval_error",
                        pending_lanes=tuple(in_flight),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                self._emit_progress(
                    "eval_complete",
                    pending_lanes=tuple(in_flight),
                    lanes=[row.lane_id for row, _context, _edge in consumed_rows],
                )
                microbatches.append(len(leaves))
                requeue_contexts: list[_LaneContext] = []
                for (row, context, edge), leaf in zip(consumed_rows, leaves):
                    context.search_id = int(row.search_id)
                    context.search_id_chain = row.search_id_chain
                    context.action_path.append(edge.action)
                    context.actor_path.append(context.node.actor_seat)
                    child = self._backup(context, edge, leaf)
                    context.path.append(edge)
                    context.node = child
                    context.stopped = bool(leaf.boundary or not child.edges)
                    root.visits += 1
                    completed_backups += 1
                    completion_order.append(row.lane_id)
                    candidate = self._proven_terminal_win_this_turn(
                        context=context,
                        leaf=leaf,
                        root=root,
                        root_actor_seat=root_actor_seat,
                        root_observation_fingerprint=(
                            root_observation_fingerprint
                        ),
                        root_legal_order_fingerprint=(
                            root_legal_order_fingerprint
                        ),
                    )
                    if candidate is not None and terminal_win_proof is None:
                        terminal_win_proof = candidate
                    reached_smoke_depth = (
                        smoke_min_depth_per_lane is not None
                        and len(context.action_path) >= int(smoke_min_depth_per_lane)
                    )
                    if not context.stopped and not reached_smoke_depth:
                        requeue_contexts.append(context)

                leader = self._root_leader(root)
                if leader is not None:
                    if leader.action == last_leader_action:
                        leader_stability_count += 1
                    else:
                        last_leader_action = leader.action
                        leader_stability_count = 1
                    self._emit_progress(
                        "leader_observation",
                        pending_lanes=tuple(in_flight),
                        completed_backups=completed_backups,
                        leader_action=list(leader.action),
                        leader_stability_count=leader_stability_count,
                    )

                all_lanes_progressed = all(
                    bool(context.action_path) for context in contexts.values()
                )
                smoke_depth_reached = (
                    smoke_min_depth_per_lane is not None
                    and all(
                        len(context.action_path) >= int(smoke_min_depth_per_lane)
                        for context in contexts.values()
                    )
                )
                # The monotonic deadline is the highest-priority stop.  A
                # batch which finishes at/after it may still be backed, but it
                # cannot be relabelled as an early convergence observation.
                if time.monotonic() >= float(deadline_monotonic):
                    # A result that missed the hard decision deadline remains
                    # useful ordinary backup evidence, but cannot acquire the
                    # exceptional r246 early-stop/selection authority.
                    terminal_win_proof = None
                    stop_reason = "decision_deadline"
                elif terminal_win_proof is not None:
                    stop_reason = PROVEN_TERMINAL_WIN_STOP_REASON
                elif completed_backups >= self._maximum_backups_per_decision:
                    stop_reason = "maximum_backups"
                elif smoke_depth_reached:
                    stop_reason = "smoke_min_depth"
                elif (
                    all_lanes_progressed
                    and completed_backups >= self._minimum_backups_before_stability
                    and leader_stability_count
                    >= self._stable_root_leader_observations
                ):
                    stop_reason = "stable_root_leader"
                if stop_reason is not None:
                    self._emit_progress(
                        "search_stop",
                        pending_lanes=tuple(in_flight),
                        stop_reason=stop_reason,
                        completed_backups=completed_backups,
                        leader_stability_count=leader_stability_count,
                        all_lanes_progressed=all_lanes_progressed,
                        terminal_win_proof=terminal_win_proof is not None,
                    )
                    break
                for context in requeue_contexts:
                    dispatch(context)

            if stop_reason is None:
                stop_reason = (
                    "decision_deadline"
                    if time.monotonic() >= float(deadline_monotonic)
                    else "tree_exhausted"
                )
                self._emit_progress(
                    "search_stop",
                    pending_lanes=tuple(in_flight),
                    stop_reason=stop_reason,
                    completed_backups=completed_backups,
                    leader_stability_count=leader_stability_count,
                    all_lanes_progressed=all(
                        bool(context.action_path) for context in contexts.values()
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - cleanup every arena before raising
            remember_error(exc)
            self._emit_progress(
                "decision_error",
                pending_lanes=tuple(in_flight),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            cleanup_deadline = time.monotonic() + self._cleanup_timeout_seconds
            # Finish any already-issued native calls before cleanup.  Their
            # results are intentionally not used after the decision stop.
            self._emit_progress("drain_begin", pending_lanes=tuple(in_flight))
            while in_flight:
                remaining = cleanup_deadline - time.monotonic()
                if remaining <= 0.0:
                    pending = tuple(sorted(in_flight))
                    error = AsyncEightWorkerError(
                        f"native in-flight drain timed out for lanes {list(pending)}"
                    )
                    remember_error(error)
                    self._mark_poisoned(str(error))
                    self._emit_progress(
                        "drain_timeout",
                        pending_lanes=pending,
                        error=str(error),
                    )
                    for lane_id, (context, edge, _expected_sequence) in tuple(
                        in_flight.items()
                    ):
                        context.in_flight = False
                        release_reservation(
                            lane_id=lane_id,
                            edge=edge,
                            phase="drain_timeout",
                        )
                    in_flight.clear()
                    break
                try:
                    row = self._completions.get(timeout=remaining)
                except queue.Empty:
                    pending = tuple(sorted(in_flight))
                    error = AsyncEightWorkerError(
                        f"native in-flight drain timed out for lanes {list(pending)}"
                    )
                    remember_error(error)
                    self._mark_poisoned(str(error))
                    self._emit_progress(
                        "drain_timeout",
                        pending_lanes=pending,
                        error=str(error),
                    )
                    for lane_id, (context, edge, _expected_sequence) in tuple(
                        in_flight.items()
                    ):
                        context.in_flight = False
                        release_reservation(
                            lane_id=lane_id,
                            edge=edge,
                            phase="drain_timeout",
                        )
                    in_flight.clear()
                    break
                self._emit_progress(
                    "drain_result",
                    lane_id=row.lane_id,
                    pending_lanes=tuple(in_flight),
                    sequence=row.sequence,
                    result_kind=row.kind,
                    command_kind=row.command_kind,
                    error_type=(type(row.error).__name__ if row.error is not None else None),
                )
                if row.lane_id not in in_flight:
                    remember_error(
                        AsyncEightWorkerError(
                            f"unexpected completion while draining lane {row.lane_id}"
                        )
                    )
                    continue
                context, edge, expected_sequence = in_flight.pop(row.lane_id)
                context.in_flight = False
                release_reservation(
                    lane_id=row.lane_id,
                    edge=edge,
                    phase="drain_result",
                )
                if row.sequence != expected_sequence:
                    remember_error(
                        AsyncEightWorkerError(
                            f"lane {row.lane_id} returned stale SearchStep sequence "
                            f"{row.sequence} during drain, expected {expected_sequence}"
                        )
                    )
                if row.error is not None or row.kind != "step" or row.search_id is None:
                    remember_error(
                        AsyncEightWorkerError(
                            f"lane {row.lane_id} SearchStep failed during drain: "
                            f"{row.error or row.kind}"
                        )
                    )
            for worker in self._workers:
                worker.submit(_CloseCommand())
                self._emit_progress(
                    "close_submit",
                    lane_id=worker.lane_id,
                    pending_lanes=tuple(
                        lane
                        for lane in range(self._lane_count)
                        if lane not in close_rows
                    ),
                )
            close_terminal_lanes: set[int] = set()
            while len(close_terminal_lanes) < self._lane_count:
                remaining = cleanup_deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = tuple(
                        sorted(set(range(self._lane_count)) - close_terminal_lanes)
                    )
                    error = AsyncEightWorkerError(
                        f"simulator close timed out for lanes {list(missing)}"
                    )
                    remember_error(error)
                    self._mark_poisoned(str(error))
                    self._emit_progress(
                        "close_timeout",
                        pending_lanes=missing,
                        error=str(error),
                    )
                    break
                try:
                    row = self._completions.get(timeout=remaining)
                except queue.Empty:
                    missing = tuple(
                        sorted(set(range(self._lane_count)) - close_terminal_lanes)
                    )
                    error = AsyncEightWorkerError(
                        f"simulator close timed out for lanes {list(missing)}"
                    )
                    remember_error(error)
                    self._mark_poisoned(str(error))
                    self._emit_progress(
                        "close_timeout",
                        pending_lanes=missing,
                        error=str(error),
                    )
                    break
                self._emit_progress(
                    "close_result",
                    lane_id=row.lane_id,
                    pending_lanes=tuple(
                        sorted(set(range(self._lane_count)) - close_terminal_lanes)
                    ),
                    sequence=row.sequence,
                    result_kind=row.kind,
                    command_kind=row.command_kind,
                    error_type=(type(row.error).__name__ if row.error is not None else None),
                )
                if row.lane_id not in range(self._lane_count):
                    remember_error(
                        AsyncEightWorkerError(
                            f"cleanup returned invalid lane {row.lane_id}"
                        )
                    )
                    continue
                if row.command_kind != "close":
                    remember_error(
                        AsyncEightWorkerError(
                            "unexpected result during cleanup: "
                            f"lane {row.lane_id} {row.command_kind or row.kind}"
                        )
                    )
                    continue
                if row.lane_id in close_terminal_lanes:
                    remember_error(
                        AsyncEightWorkerError(
                            f"duplicate close completion for lane {row.lane_id}"
                        )
                    )
                    continue
                close_terminal_lanes.add(row.lane_id)
                close_rows[row.lane_id] = row
                if row.kind != "close" or row.error is not None:
                    remember_error(
                        AsyncEightWorkerError(
                            f"lane {row.lane_id} simulator cleanup failed: "
                            f"{row.error or row.kind}"
                        )
                    )
            self._emit_progress(
                "cleanup_complete",
                pending_lanes=tuple(
                    sorted(set(range(self._lane_count)) - close_terminal_lanes)
                ),
                status=("error" if structural_error is not None else "ok"),
            )
        if structural_error is not None:
            raise AsyncEightWorkerError(
                f"asynchronous {self._lane_count}-lane decision failed: {structural_error}"
            ) from structural_error
        if len(close_rows) != self._lane_count or any(
            row.error is not None for row in close_rows.values()
        ):
            raise AsyncEightWorkerError("asynchronous simulator cleanup was incomplete")
        seen_nodes: set[int] = set()

        def virtual_loss_total(node: _Node) -> int:
            identity = id(node)
            if identity in seen_nodes:
                return 0
            seen_nodes.add(identity)
            total = 0
            for edge in node.edges:
                total += edge.virtual_loss
                for child in edge.children.values():
                    total += virtual_loss_total(child)
            return total

        outstanding = virtual_loss_total(root)
        if outstanding:
            raise AsyncEightWorkerError("shared tree leaked virtual loss")
        per_lane_depth = tuple(
            len(contexts[lane].action_path) for lane in range(self._lane_count)
        )
        per_lane_search_id_chains = tuple(
            contexts[lane].search_id_chain for lane in range(self._lane_count)
        )
        search_release_calls = sum(row.releases for row in close_rows.values())
        search_end_calls = sum(row.search_end_calls for row in close_rows.values())
        elapsed_seconds = max(0.0, time.monotonic() - started)
        if completed_backups < 1:
            if stop_reason != "decision_deadline":
                raise AsyncEightWorkerError(
                    "zero-backup search ended without a clean decision deadline"
                )
            cleanup_receipt = AsyncDirectFallbackReceipt(
                arena_count=self._lane_count,
                unique_handle_count=len(set(self._handle_identities)),
                search_begin_calls=self._lane_count,
                # These calls were issued and boundedly drained even though no
                # result was admitted into the shared tree.
                search_step_calls=sequence,
                completed_backups=0,
                microbatch_sizes=tuple(microbatches),
                max_simulator_calls_in_flight=peak_in_flight,
                completion_order=tuple(completion_order),
                per_lane_depth=per_lane_depth,
                per_lane_search_id_chains=per_lane_search_id_chains,
                per_lane_handle_identities=self._handle_identities,
                distinct_search_begin_composite_count=(
                    distinct_search_begin_composite_count
                ),
                search_release_calls=search_release_calls,
                search_end_calls=search_end_calls,
                outstanding_virtual_loss=outstanding,
                stop_reason="decision_deadline",
                minimum_backups_before_stability=(
                    self._minimum_backups_before_stability
                ),
                stable_root_leader_observations=(
                    self._stable_root_leader_observations
                ),
                maximum_backups_per_decision=(
                    self._maximum_backups_per_decision
                ),
                leader_stability_count=leader_stability_count,
                elapsed_seconds=elapsed_seconds,
                root_seat=root_seat,
            )
            raise AsyncDirectFallbackRequired(
                "clean decision stop completed no backups; typed direct fallback "
                f"required (stop_reason={stop_reason})",
                cleanup_receipt=cleanup_receipt,
            )
        selected: _Edge | None = None
        if terminal_win_proof is not None:
            if stop_reason != PROVEN_TERMINAL_WIN_STOP_REASON:
                raise AsyncEightWorkerError(
                    "terminal-win proof survived without its terminal stop reason"
                )
            proof_action = terminal_win_proof.get("selected_action")
            if not isinstance(proof_action, list):
                raise AsyncEightWorkerError("terminal-win proof omitted selected action")
            normalized_proof_action = tuple(int(item) for item in proof_action)
            matches = [
                edge for edge in root.edges if edge.action == normalized_proof_action
            ]
            if len(matches) != 1 or matches[0].visits < 1:
                raise AsyncEightWorkerError(
                    "terminal-win proof does not identify one backed root edge"
                )
            selected = matches[0]
        elif stop_reason == PROVEN_TERMINAL_WIN_STOP_REASON:
            raise AsyncEightWorkerError(
                "terminal-win stop omitted its deterministic terminal proof"
            )
        else:
            selected = self._root_leader(root)
        if selected is None:
            raise AsyncEightWorkerError("shared root has no backed action")
        principal_variation = (
            ()
            if terminal_win_proof is not None
            else self._principal_variation(
                selected_root_edge=selected,
                root_seat=root_seat,
            )
        )
        return AsyncDecisionReceipt(
            selected_action=selected.action,
            selected_action_visits=selected.visits,
            selected_action_value=selected.q,
            selected_action_prior=selected.prior,
            root_visits=root.visits,
            arena_count=self._lane_count,
            unique_handle_count=len({worker.handle_identity for worker in self._workers}),
            search_begin_calls=self._lane_count,
            search_step_calls=sequence,
            completed_backups=completed_backups,
            microbatch_sizes=tuple(microbatches),
            max_simulator_calls_in_flight=peak_in_flight,
            completion_order=tuple(completion_order),
            per_lane_depth=per_lane_depth,
            per_lane_search_id_chains=per_lane_search_id_chains,
            per_lane_handle_identities=self._handle_identities,
            distinct_search_begin_composite_count=(
                distinct_search_begin_composite_count
            ),
            search_release_calls=search_release_calls,
            search_end_calls=search_end_calls,
            outstanding_virtual_loss=outstanding,
            stop_reason=str(stop_reason),
            minimum_backups_before_stability=(
                self._minimum_backups_before_stability
            ),
            stable_root_leader_observations=(
                self._stable_root_leader_observations
            ),
            maximum_backups_per_decision=self._maximum_backups_per_decision,
            leader_stability_count=leader_stability_count,
            elapsed_seconds=elapsed_seconds,
            root_seat=root_seat,
            principal_variation=principal_variation,
            root_actor_seat=root_actor_seat,
            root_observation_fingerprint=root_observation_fingerprint,
            root_legal_order_fingerprint=root_legal_order_fingerprint,
            terminal_win_proof=terminal_win_proof,
            owner_proven_deterministic_terminal_win_this_turn_revision=(
                PROVEN_TERMINAL_WIN_REVISION
            ),
        )


# Backward-compatible import name used by the r228 package entrypoint.
PersistentAsyncEightWorkerMCTS = PersistentAsyncSharedTreeMCTS


__all__ = [
    "AsyncDecisionReceipt",
    "AsyncDirectFallbackReceipt",
    "AsyncDirectFallbackRequired",
    "AsyncEightWorkerError",
    "AsyncLanePoolError",
    "DEFAULT_LANE_COUNT",
    "DecodedLeaf",
    "LANES",
    "MAX_PRINCIPAL_VARIATION_DEPTH",
    "PROVEN_TERMINAL_WIN_PROOF_KIND",
    "PROVEN_TERMINAL_WIN_REVISION",
    "PROVEN_TERMINAL_WIN_STOP_REASON",
    "PersistentAsyncEightWorkerMCTS",
    "PersistentAsyncSharedTreeMCTS",
]
