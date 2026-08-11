"""True serial MCTS as repeated independent native root rollouts.

The frozen model and logical tree live in the authoritative parent process.
Exactly one process-owned native simulator handle is used, with no concurrent
native calls.  Each rollout starts again from the exact physical root, expands
at most one new tree node (or reaches a value-only boundary), backs one value,
and is completely released before the next rollout begins.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .r228_async_shared_tree_queue import AsyncEightWorkerError, DecodedLeaf


SCHEMA = "poke_bot.r253_restarting_serial_mcts/v1"
DEFAULT_MAX_ROLLOUTS = 1000
DEFAULT_MAX_ROLLOUT_DEPTH = 256
PacketT = TypeVar("PacketT")


@dataclass
class _Edge:
    action: tuple[int, ...]
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[str, "_Node"] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class _Node:
    state_key: str
    actor_seat: int | None
    edges: list[_Edge]
    visits: int = 0


@dataclass(frozen=True)
class R253SerialDecisionReceipt:
    selected_action: tuple[int, ...]
    selected_action_visits: int
    selected_action_value: float
    selected_action_prior: float
    root_visits: int
    arena_count: int
    unique_handle_count: int
    per_lane_handle_identities: tuple[int | str, ...]
    search_begin_calls: int
    search_step_calls: int
    completed_backups: int
    microbatch_sizes: tuple[int, ...]
    max_simulator_calls_in_flight: int
    completion_order: tuple[int, ...]
    per_lane_depth: tuple[int, ...]
    per_lane_search_id_chains: tuple[tuple[int, ...], ...]
    search_release_calls: int
    search_end_calls: int
    outstanding_virtual_loss: int
    elapsed_seconds: float
    rollout_count: int
    rollout_search_id_chains: tuple[tuple[int, ...], ...]
    rollout_root_actions: tuple[tuple[int, ...], ...]
    root_action_visit_counts: tuple[int, ...]
    distinct_root_actions_visited: int
    max_rollout_depth: int
    stop_reason: str
    rollout_ceiling: int


def _normalized_node(
    state_key: str,
    actions: Sequence[Sequence[int]],
    priors: Sequence[float],
    *,
    actor_seat: int | None,
) -> _Node:
    normalized_actions = tuple(tuple(int(item) for item in row) for row in actions)
    normalized_priors = tuple(float(value) for value in priors)
    if (
        not state_key
        or actor_seat not in (None, 0, 1)
        or not normalized_actions
        or len(normalized_actions) != len(normalized_priors)
        or len(set(normalized_actions)) != len(normalized_actions)
        or any(not math.isfinite(value) or value < 0.0 for value in normalized_priors)
    ):
        raise AsyncEightWorkerError("r253 tree node action/prior shape is invalid")
    total = math.fsum(normalized_priors)
    if total <= 0.0:
        raise AsyncEightWorkerError("r253 tree node prior has no mass")
    return _Node(
        state_key=str(state_key),
        actor_seat=actor_seat,
        edges=[
            _Edge(action=action, prior=prior / total)
            for action, prior in zip(normalized_actions, normalized_priors)
        ],
    )


class R253RestartingSerialMCTS:
    """One parent tree fed by repeated, fully cleaned native root rollouts."""

    def __init__(
        self,
        *,
        arena_factory: Callable[[int], Any],
        make_packet: Callable[[int, Any], PacketT],
        evaluate_batch: Callable[[Sequence[PacketT]], Sequence[DecodedLeaf]],
        puct_c: float = 1.25,
        coalesce_seconds: float = 0.0,
        max_rollouts: int = DEFAULT_MAX_ROLLOUTS,
        max_rollout_depth: int = DEFAULT_MAX_ROLLOUT_DEPTH,
    ) -> None:
        del coalesce_seconds  # retained only for the pre-r253 constructor ABI
        self._arena = arena_factory(0)
        self._make_packet = make_packet
        self._evaluate_batch = evaluate_batch
        self._puct_c = float(puct_c)
        self._max_rollouts = int(max_rollouts)
        self._max_rollout_depth = int(max_rollout_depth)
        self._closed = False
        if (
            not math.isfinite(self._puct_c)
            or self._puct_c <= 0.0
            or self._max_rollouts < 2
            or self._max_rollout_depth < 1
        ):
            self.close()
            raise ValueError("r253 serial rollout limits are invalid")
        self.lanes = [self._arena]

    @property
    def handle_identity(self) -> int | str:
        return self._arena.handle_identity

    @staticmethod
    def _fault_count(arena: Any) -> int:
        faults = getattr(arena, "faults", ())
        try:
            return len(faults)
        except TypeError:
            return 0

    def _reserve(self, node: _Node, *, root_seat: int) -> _Edge:
        direction = 1.0 if node.actor_seat in (None, root_seat) else -1.0
        parent_scale = math.sqrt(max(1, node.visits))
        return max(
            node.edges,
            key=lambda edge: (
                direction * edge.q
                + self._puct_c
                * edge.prior
                * parent_scale
                / (1 + edge.visits),
                edge.prior,
                tuple(-item for item in edge.action),
            ),
        )

    @staticmethod
    def _backup(nodes: Sequence[_Node], edges: Sequence[_Edge], value: float) -> None:
        if len(nodes) != len(edges) + 1:
            raise AsyncEightWorkerError("r253 backup path shape drifted")
        for node in nodes:
            node.visits += 1
        for edge in edges:
            edge.visits += 1
            edge.value_sum += float(value)

    @staticmethod
    def _assert_existing_child(child: _Node, leaf: DecodedLeaf) -> None:
        expected_actions = tuple(edge.action for edge in child.edges)
        if (
            child.actor_seat != leaf.actor_seat
            or bool(child.edges) == bool(leaf.boundary)
            or (not leaf.boundary and expected_actions != tuple(leaf.legal_actions))
        ):
            raise AsyncEightWorkerError(
                "r253 repeated rollout reached a tree-inconsistent simulator state"
            )

    def _cleanup_rollout(
        self,
        search_ids: Sequence[int],
        *,
        fault_offset: int,
    ) -> tuple[int, int]:
        cleanup_error: Exception | None = None
        releases = 0
        for search_id in reversed(search_ids):
            try:
                self._arena.search_release(int(search_id))
                releases += 1
            except Exception as exc:  # noqa: BLE001 - cleanup every live id
                cleanup_error = cleanup_error or exc
        try:
            self._arena.search_end()
        except Exception as exc:  # noqa: BLE001 - preserve the first failure
            cleanup_error = cleanup_error or exc
        if self._fault_count(self._arena) > fault_offset:
            cleanup_error = cleanup_error or AsyncEightWorkerError(
                "contained_native_lane_fault during r253 rollout cleanup"
            )
        if cleanup_error is not None:
            raise AsyncEightWorkerError(
                f"r253 bounded rollout cleanup failed: {cleanup_error}"
            ) from cleanup_error
        return releases, 1

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
        smoke_min_depth_per_lane: int | None = None,
    ) -> R253SerialDecisionReceipt:
        if self._closed:
            raise AsyncEightWorkerError("r253 serial MCTS is closed")
        if root_seat not in (0, 1):
            raise AsyncEightWorkerError("r253 root seat must be zero or one")
        if len(search_inputs) != 1:
            raise AsyncEightWorkerError("r253 requires exactly one search-input row")
        if smoke_min_depth_per_lane is not None:
            raise AsyncEightWorkerError("r253 does not use continuous-trajectory smoke depth")

        started = time.monotonic()
        root = _normalized_node(
            root_state_key, root_actions, root_priors, actor_seat=root_seat
        )
        original_handle = self.handle_identity
        rollout_chains: list[tuple[int, ...]] = []
        rollout_root_actions: list[tuple[int, ...]] = []
        microbatches: list[int] = []
        search_begin_calls = search_step_calls = completed_backups = 0
        search_release_calls = search_end_calls = 0
        max_depth = 0

        while (
            completed_backups < self._max_rollouts
            and time.monotonic() < float(deadline_monotonic)
        ):
            fault_offset = self._fault_count(self._arena)
            search_ids: list[int] = []
            rollout_error: Exception | None = None
            leaf_value: float | None = None
            nodes = [root]
            edges: list[_Edge] = []
            try:
                if self.handle_identity != original_handle:
                    raise AsyncEightWorkerError(
                        "r253 native handle changed between successful rollouts"
                    )
                state = self._arena.search_begin(
                    root_observation, search_inputs[0], manual_coin=True
                )
                search_begin_calls += 1
                search_ids.append(int(state.searchId))
                node = root
                for _depth in range(self._max_rollout_depth):
                    edge = self._reserve(node, root_seat=root_seat)
                    if not edges:
                        rollout_root_actions.append(edge.action)
                    state = self._arena.search_step(
                        search_ids[-1], list(edge.action)
                    )
                    search_step_calls += 1
                    child_id = int(state.searchId)
                    if child_id in search_ids:
                        raise AsyncEightWorkerError(
                            "r253 rollout reused a live SearchId"
                        )
                    search_ids.append(child_id)
                    packets = [self._make_packet(0, state.observation)]
                    leaves = tuple(self._evaluate_batch(packets))
                    if len(leaves) != 1:
                        raise AsyncEightWorkerError(
                            "r253 frozen evaluator returned a partial serial batch"
                        )
                    leaf = leaves[0]
                    leaf.validate()
                    microbatches.append(1)
                    edges.append(edge)
                    child = edge.children.get(leaf.state_key)
                    newly_expanded = child is None
                    if child is None:
                        child = (
                            _Node(
                                state_key=leaf.state_key,
                                actor_seat=leaf.actor_seat,
                                edges=[],
                            )
                            if leaf.boundary
                            else _normalized_node(
                                leaf.state_key,
                                leaf.legal_actions,
                                leaf.priors,
                                actor_seat=leaf.actor_seat,
                            )
                        )
                        edge.children[leaf.state_key] = child
                    else:
                        self._assert_existing_child(child, leaf)
                    nodes.append(child)
                    leaf_value = float(leaf.value)
                    if (
                        newly_expanded
                        or leaf.boundary
                        or not child.edges
                        or time.monotonic() >= float(deadline_monotonic)
                    ):
                        break
                    node = child
                else:
                    if leaf_value is None:
                        raise AsyncEightWorkerError(
                            "r253 rollout depth ceiling produced no evaluated leaf"
                        )
                if leaf_value is None:
                    raise AsyncEightWorkerError("r253 rollout produced no frozen value")
                self._backup(nodes, edges, leaf_value)
                completed_backups += 1
                max_depth = max(max_depth, len(edges))
            except Exception as exc:  # noqa: BLE001 - cleanup before propagation
                rollout_error = exc

            cleanup_error: Exception | None = None
            try:
                releases, ends = self._cleanup_rollout(
                    search_ids, fault_offset=fault_offset
                )
                search_release_calls += releases
                search_end_calls += ends
            except Exception as exc:  # noqa: BLE001 - cleanup failure invalidates attempt
                cleanup_error = exc
            rollout_chains.append(tuple(search_ids))
            if rollout_error is not None:
                if cleanup_error is not None:
                    raise cleanup_error from rollout_error
                raise rollout_error
            if cleanup_error is not None:
                raise cleanup_error

        if completed_backups < 2:
            raise AsyncEightWorkerError(
                "r253 completed fewer than two independent root rollouts"
            )
        visited = [edge for edge in root.edges if edge.visits > 0]
        if not visited:
            raise AsyncEightWorkerError("r253 root has no backed action")
        selected = max(
            visited,
            key=lambda edge: (
                edge.visits,
                edge.q,
                edge.prior,
                tuple(-item for item in edge.action),
            ),
        )
        stop_reason = (
            "rollout_ceiling"
            if completed_backups >= self._max_rollouts
            else "decision_deadline"
        )
        return R253SerialDecisionReceipt(
            selected_action=selected.action,
            selected_action_visits=selected.visits,
            selected_action_value=selected.q,
            selected_action_prior=selected.prior,
            root_visits=root.visits,
            arena_count=1,
            unique_handle_count=1,
            per_lane_handle_identities=(original_handle,),
            search_begin_calls=search_begin_calls,
            search_step_calls=search_step_calls,
            completed_backups=completed_backups,
            microbatch_sizes=tuple(microbatches),
            max_simulator_calls_in_flight=1,
            completion_order=tuple(0 for _ in range(completed_backups)),
            per_lane_depth=(max_depth,),
            per_lane_search_id_chains=(rollout_chains[0],),
            search_release_calls=search_release_calls,
            search_end_calls=search_end_calls,
            outstanding_virtual_loss=0,
            elapsed_seconds=max(0.0, time.monotonic() - started),
            rollout_count=completed_backups,
            rollout_search_id_chains=tuple(rollout_chains),
            rollout_root_actions=tuple(rollout_root_actions),
            root_action_visit_counts=tuple(edge.visits for edge in root.edges),
            distinct_root_actions_visited=len(visited),
            max_rollout_depth=max_depth,
            stop_reason=stop_reason,
            rollout_ceiling=self._max_rollouts,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._arena, "close", None)
        if callable(close):
            close()


__all__ = [
    "DEFAULT_MAX_ROLLOUTS",
    "DEFAULT_MAX_ROLLOUT_DEPTH",
    "R253RestartingSerialMCTS",
    "R253SerialDecisionReceipt",
    "SCHEMA",
]
