from __future__ import annotations

import multiprocessing
import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from poke_bot.r228_async_shared_tree_queue import (
    AsyncDirectFallbackRequired,
    AsyncEightWorkerError,
    DEFAULT_LANE_COUNT,
    DecodedLeaf,
    PersistentAsyncEightWorkerMCTS,
    PROVEN_TERMINAL_WIN_PROOF_KIND,
    PROVEN_TERMINAL_WIN_REVISION,
    PROVEN_TERMINAL_WIN_STOP_REASON,
)


class _Trace:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.next_handle = 100
        self.events: list[tuple[str, int, int, float]] = []
        self.live: dict[int, set[int]] = defaultdict(set)

    def event(self, kind: str, lane: int, depth: int) -> None:
        with self.lock:
            self.events.append((kind, lane, depth, time.monotonic()))


class _Arena:
    def __init__(
        self,
        lane: int,
        trace: _Trace,
        *,
        step_seconds: float = 0.001,
        slow_lane_one_first_seconds: float = 0.030,
    ) -> None:
        self.lane = lane
        self.trace = trace
        self.step_seconds = float(step_seconds)
        self.slow_lane_one_first_seconds = float(slow_lane_one_first_seconds)
        self.owner = threading.get_ident()
        with trace.lock:
            self.handle = trace.next_handle
            trace.next_handle += 1

    @property
    def handle_identity(self) -> int:
        return self.handle

    def _owner(self) -> None:
        assert threading.get_ident() == self.owner

    def search_begin(self, _obs, _inputs, manual_coin=True):
        self._owner()
        assert manual_coin is True
        self.trace.live[self.handle].add(0)
        return SimpleNamespace(
            searchId=0,
            observation={"lane": self.lane, "depth": 0, "last_action": None},
        )

    def search_step(self, search_id, select):
        self._owner()
        depth = int(search_id)
        self.trace.event("start", self.lane, depth)
        # Lane 1's first simulator call is deliberately slow.  A genuinely
        # asynchronous coordinator should evaluate/requeue faster lanes before
        # this call returns, rather than wait at an all-lane wave barrier.
        time.sleep(
            self.slow_lane_one_first_seconds
            if self.lane == 1 and depth == 0
            else self.step_seconds
        )
        child = depth + 1
        self.trace.live[self.handle].add(child)
        self.trace.event("finish", self.lane, depth)
        return SimpleNamespace(
            searchId=child,
            observation={
                "lane": self.lane,
                "depth": child,
                "last_action": tuple(int(item) for item in select),
            },
        )

    def search_release(self, search_id):
        self._owner()
        self.trace.live[self.handle].remove(int(search_id))

    def search_end(self):
        self._owner()
        self.trace.live[self.handle].clear()


class _FaultArena(_Arena):
    """CPU-only native-lifecycle fault injection for bounded queue tests."""

    def __init__(
        self,
        lane: int,
        trace: _Trace,
        *,
        fault: str,
        release_stall: threading.Event,
    ) -> None:
        super().__init__(lane, trace)
        self.fault = fault
        self.release_stall = release_stall

    def search_step(self, search_id, select):
        if self.fault == "step_stall" and self.lane == 0:
            self._owner()
            self.trace.event("stall", self.lane, int(search_id))
            self.release_stall.wait()
        return super().search_step(search_id, select)

    def search_end(self):
        super().search_end()
        if self.fault == "close_error" and self.lane == 0:
            raise RuntimeError("injected SearchEnd failure")


def _normal_leaf_batch(rows):
    return tuple(
        DecodedLeaf(
            state_key=(
                f"fault-lane={lane};depth={int(observation['depth'])}"
            ),
            value=0.25,
            legal_actions=((0,), (1,)),
            priors=(0.5, 0.5),
            actor_seat=0,
        )
        for lane, observation in rows
    )


def _run_fault_case_in_child(send, fault: str) -> None:
    """Run an intentionally wedged native stand-in outside the pytest process.

    A regression in cleanup must not leave the test runner with a live daemon
    worker.  This child represents only the exact mock simulator process it
    creates; it never interacts with a managed or interactive process.
    """

    trace = _Trace()
    release_stall = threading.Event()
    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _FaultArena(
            lane,
            trace,
            fault=fault,
            release_stall=release_stall,
        ),
        make_packet=lambda lane, observation: (lane, observation),
        evaluate_batch=(
            lambda rows: (
                (_ for _ in ()).throw(RuntimeError("injected evaluator failure"))
                # The coordinator may correctly form a partial ready microbatch.
                # What matters is that these rows were already consumed from the
                # completion queue before the evaluator raised.
                if fault == "evaluator_raises" and rows
                else _normal_leaf_batch(rows)
            )
        ),
        # The queue's hard cleanup bound is the behavior under test.
        cleanup_timeout_seconds=0.05,
        coalesce_seconds=0.03,
    )
    started = time.monotonic()
    run_error: BaseException | None = None
    try:
        search.run_decision(
            root_observation={"current": {"yourIndex": 0}},
            search_inputs=tuple({} for _ in range(2)),
            root_state_key=f"fault-{fault}",
            root_actions=((0,), (1,)),
            root_priors=(0.5, 0.5),
            root_seat=0,
            deadline_monotonic=time.monotonic() + 0.15,
            smoke_min_depth_per_lane=1,
        )
    except BaseException as exc:  # noqa: BLE001 - serialise the exact failure
        run_error = exc
    run_elapsed = time.monotonic() - started

    close_started = time.monotonic()
    close_error: BaseException | None = None
    try:
        search.close()
    except BaseException as exc:  # noqa: BLE001 - bounded poisoned shutdown is expected
        close_error = exc
    close_elapsed = time.monotonic() - close_started

    # Let the deliberately stalled mock unwind before this exact child exits.
    # A fixed queue returns before this point; an unfixed queue never reaches it
    # and is bounded by the parent test harness below.
    release_stall.set()
    for worker in search._workers:  # noqa: SLF001 - verifies exact child cleanup
        worker._thread.join(timeout=0.20)  # noqa: SLF001
    payload = {
        "run_error_type": type(run_error).__name__ if run_error else None,
        "run_error": str(run_error) if run_error else "",
        "run_elapsed": run_elapsed,
        "close_error_type": type(close_error).__name__ if close_error else None,
        "close_error": str(close_error) if close_error else "",
        "close_elapsed": close_elapsed,
        "workers_alive_after_release": [
            worker.lane_id
            for worker in search._workers  # noqa: SLF001
            if worker._thread.is_alive()  # noqa: SLF001
        ],
    }
    send.send(payload)
    send.close()


def _fault_case(fault: str) -> dict[str, object]:
    """Collect one fault result with a hard parent-side test bound."""

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("bounded r228 native mock needs the fork start method")
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_run_fault_case_in_child, args=(send, fault))
    child.start()
    send.close()
    child.join(timeout=1.5)
    if child.is_alive():
        # This is the exact disposable test child above, not a user session.
        child.terminate()
        child.join(timeout=1.0)
        pytest.fail(f"r228 {fault} case exceeded its parent test bound")
    assert child.exitcode == 0
    assert receive.poll(0.1), f"r228 {fault} case exited without a receipt"
    result = receive.recv()
    receive.close()
    assert isinstance(result, dict)
    return result


def test_one_local_decision_uses_default_two_lane_model_guided_tree() -> None:
    assert DEFAULT_LANE_COUNT == 2
    trace = _Trace()

    def packet(lane: int, observation):
        return lane, int(observation["depth"]), observation["last_action"]

    model_batches: list[tuple[tuple[int, int, object], ...]] = []

    def frozen_model(rows):
        batch = tuple(rows)
        model_batches.append(batch)
        return [
            DecodedLeaf(
                state_key=f"lane={lane};depth={depth}",
                value=0.8 if last_action == (1,) else 0.2,
                legal_actions=((0,), (1,)),
                priors=(0.15, 0.85),
                actor_seat=0,
            )
            for lane, depth, last_action in batch
        ]

    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(lane, trace),
        make_packet=packet,
        evaluate_batch=frozen_model,
        coalesce_seconds=0.002,
    )
    try:
        receipt = search.run_decision(
            root_observation={"current": {"yourIndex": 0}},
            search_inputs=tuple({} for _ in range(2)),
            root_state_key="one-local-decision",
            root_actions=((0,), (1,)),
            root_priors=(0.2, 0.8),
            root_seat=0,
            deadline_monotonic=time.monotonic() + 2.0,
            smoke_min_depth_per_lane=2,
        )
    finally:
        search.close()

    assert search.lane_count == 2
    assert receipt.arena_count == receipt.unique_handle_count == 2
    assert receipt.search_begin_calls == 2
    assert receipt.search_step_calls == receipt.completed_backups == 4
    assert receipt.max_simulator_calls_in_flight == 2
    assert receipt.root_visits == 4
    assert receipt.selected_action in ((0,), (1,))
    assert receipt.selected_action_visits >= 1
    assert receipt.selected_action_prior > 0.0
    assert 0.0 <= receipt.selected_action_value <= 1.0
    assert receipt.per_lane_depth == (2,) * 2
    assert all(len(chain) == 3 for chain in receipt.per_lane_search_id_chains)
    # SearchId is scoped to its AgentStart handle.  Identical raw root IDs are
    # valid because the lane-aligned handle/id composites are distinct.
    assert tuple(chain[0] for chain in receipt.per_lane_search_id_chains) == (0, 0)
    assert receipt.per_lane_handle_identities == (100, 101)
    assert receipt.distinct_search_begin_composite_count == 2
    assert receipt.search_release_calls == 6
    assert receipt.search_end_calls == 2
    assert receipt.outstanding_virtual_loss == 0
    assert receipt.stop_reason == "smoke_min_depth"
    assert receipt.minimum_backups_before_stability == 8
    assert receipt.stable_root_leader_observations == 3
    assert receipt.maximum_backups_per_decision == 32
    assert receipt.leader_stability_count >= 1
    assert receipt.elapsed_seconds > 0.0
    assert receipt.root_seat == 0
    assert receipt.principal_variation == ()
    assert sum(receipt.microbatch_sizes) == 4
    assert model_batches
    assert all(not ids for ids in trace.live.values())

    starts = {(lane, depth): when for kind, lane, depth, when in trace.events if kind == "start"}
    finishes = {(lane, depth): when for kind, lane, depth, when in trace.events if kind == "finish"}
    assert starts[(0, 1)] < finishes[(1, 0)]


def test_exact_terminal_win_overrides_higher_prior_root_leader_and_stops() -> None:
    """One backed deterministic win has r246 authority without a root scan."""

    trace = _Trace()

    def evaluate(rows):
        leaves = []
        for lane, observation in rows:
            action = tuple(observation["last_action"])
            if action == (1,):
                leaves.append(
                    DecodedLeaf(
                        state_key=f"terminal-win-lane={lane}",
                        value=1.0,
                        legal_actions=(),
                        priors=(),
                        boundary=True,
                        actor_seat=None,
                        observation_fingerprint=f"terminal-win-{lane}",
                        terminal_result="win",
                        terminal_winner_seat=0,
                        terminal_leaf_reached=True,
                    )
                )
            else:
                # This ordinary backed alternative ties the terminal edge's
                # visits/value and has the higher prior.  The normal leader is
                # therefore action (0,), proving the terminal selection is an
                # explicit override rather than a coincidental visit winner.
                leaves.append(
                    DecodedLeaf(
                        state_key=f"ordinary-lane={lane}",
                        value=1.0,
                        legal_actions=((0,), (1,)),
                        priors=(0.6, 0.4),
                        actor_seat=0,
                        observation_fingerprint=f"ordinary-{lane}",
                    )
                )
        return tuple(leaves)

    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(
            lane,
            trace,
            step_seconds=0.001,
            slow_lane_one_first_seconds=0.001,
        ),
        make_packet=lambda lane, observation: (lane, observation),
        evaluate_batch=evaluate,
        coalesce_seconds=0.020,
    )
    try:
        receipt = search.run_decision(
            root_observation={"current": {"yourIndex": 0}},
            search_inputs=({}, {}),
            root_state_key="terminal-win-root",
            root_actions=((0,), (1,)),
            root_priors=(0.6, 0.4),
            root_seat=0,
            root_actor_seat=0,
            root_observation_fingerprint="sha256:current-root",
            root_legal_order_fingerprint="sha256:ordered-legal",
            deadline_monotonic=time.monotonic() + 1.0,
        )
    finally:
        search.close()

    assert receipt.stop_reason == PROVEN_TERMINAL_WIN_STOP_REASON
    assert receipt.selected_action == (1,)
    assert receipt.completed_backups >= 1
    assert receipt.selected_action_visits >= 1
    assert receipt.owner_proven_deterministic_terminal_win_this_turn_revision == 246
    proof = receipt.terminal_win_proof
    assert proof is not None
    assert proof["proof_kind"] == PROVEN_TERMINAL_WIN_PROOF_KIND
    assert proof["root_action"] == proof["selected_action"] == [1]
    assert proof["terminal_result"] == "win"
    assert proof["terminal_winner_seat"] == 0
    assert proof["path_actor_seats"] == [0]
    assert proof["path_no_chance_boundary"] is True
    assert proof["path_no_actor_change_boundary"] is True
    assert receipt.principal_variation == ()


@pytest.mark.parametrize("boundary_kind", ["chance", "actor", "loss"])
def test_possible_or_opponent_or_losing_leaf_has_no_terminal_override(
    boundary_kind: str,
) -> None:
    trace = _Trace()

    def evaluate(rows):
        leaves = []
        for lane, _observation in rows:
            terminal = boundary_kind == "loss"
            leaves.append(
                DecodedLeaf(
                    state_key=f"{boundary_kind}-lane={lane}",
                    value=1.0 if not terminal else -1.0,
                    legal_actions=(),
                    priors=(),
                    boundary=True,
                    actor_seat=1 if boundary_kind == "actor" else 0,
                    observation_fingerprint=f"{boundary_kind}-{lane}",
                    terminal_result="loss" if terminal else None,
                    terminal_winner_seat=1 if terminal else None,
                    terminal_leaf_reached=terminal,
                    chance_boundary=boundary_kind == "chance",
                    actor_change_boundary=boundary_kind == "actor",
                    unresolved_randomness=boundary_kind == "chance",
                )
            )
        return tuple(leaves)

    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(
            lane,
            trace,
            step_seconds=0.001,
            slow_lane_one_first_seconds=0.001,
        ),
        make_packet=lambda lane, observation: (lane, observation),
        evaluate_batch=evaluate,
        coalesce_seconds=0.020,
    )
    try:
        receipt = search.run_decision(
            root_observation={"current": {"yourIndex": 0}},
            search_inputs=({}, {}),
            root_state_key=f"no-proof-{boundary_kind}",
            root_actions=((0,), (1,)),
            root_priors=(0.6, 0.4),
            root_seat=0,
            root_actor_seat=0,
            root_observation_fingerprint="sha256:current-root",
            root_legal_order_fingerprint="sha256:ordered-legal",
            deadline_monotonic=time.monotonic() + 1.0,
        )
    finally:
        search.close()

    assert receipt.stop_reason != PROVEN_TERMINAL_WIN_STOP_REASON
    assert receipt.terminal_win_proof is None
    assert receipt.owner_proven_deterministic_terminal_win_this_turn_revision == (
        PROVEN_TERMINAL_WIN_REVISION
    )


def _adaptive_receipt(
    *,
    minimum_backups: int,
    stable_observations: int,
    maximum_backups: int,
    deadline_seconds: float,
    step_seconds: float = 0.001,
    lane_one_first_seconds: float | None = None,
    coalesce_seconds: float = 0.010,
):
    trace = _Trace()
    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(
            lane,
            trace,
            step_seconds=step_seconds,
            slow_lane_one_first_seconds=(
                step_seconds
                if lane_one_first_seconds is None
                else lane_one_first_seconds
            ),
        ),
        make_packet=lambda lane, observation: (lane, observation),
        evaluate_batch=_normal_leaf_batch,
        coalesce_seconds=coalesce_seconds,
        minimum_backups_before_stability=minimum_backups,
        stable_root_leader_observations=stable_observations,
        maximum_backups_per_decision=maximum_backups,
        cleanup_timeout_seconds=0.25,
    )
    try:
        return search.run_decision(
            root_observation={"current": {"yourIndex": 0}},
            search_inputs=({}, {}),
            root_state_key="adaptive-root",
            root_actions=((0,), (1,)),
            root_priors=(0.0, 1.0),
            root_seat=0,
            deadline_monotonic=time.monotonic() + deadline_seconds,
        )
    finally:
        search.close()


def test_adaptive_search_stops_after_stable_root_leader() -> None:
    receipt = _adaptive_receipt(
        minimum_backups=8,
        stable_observations=3,
        maximum_backups=32,
        deadline_seconds=1.0,
    )

    assert receipt.stop_reason == "stable_root_leader"
    assert 8 <= receipt.completed_backups < 32
    assert receipt.leader_stability_count >= 3
    assert all(depth > 0 for depth in receipt.per_lane_depth)


def test_stable_leader_cannot_stop_until_both_lanes_have_progressed() -> None:
    receipt = _adaptive_receipt(
        minimum_backups=8,
        stable_observations=3,
        maximum_backups=32,
        deadline_seconds=0.030,
        step_seconds=0.002,
        lane_one_first_seconds=0.080,
        coalesce_seconds=0.0,
    )

    assert receipt.completed_backups >= 8
    assert receipt.leader_stability_count >= 3
    assert receipt.per_lane_depth[0] >= 8
    assert receipt.per_lane_depth[1] == 0
    assert receipt.stop_reason == "decision_deadline"


def test_adaptive_search_stops_at_exact_backup_cap() -> None:
    receipt = _adaptive_receipt(
        minimum_backups=6,
        stable_observations=100,
        maximum_backups=6,
        deadline_seconds=1.0,
    )

    assert receipt.stop_reason == "maximum_backups"
    assert receipt.completed_backups == 6
    assert receipt.search_step_calls == 6
    assert sum(receipt.per_lane_depth) == 6
    assert all(depth > 0 for depth in receipt.per_lane_depth)


def test_adaptive_search_deadline_is_a_hard_normal_stop_after_backups() -> None:
    receipt = _adaptive_receipt(
        minimum_backups=8,
        stable_observations=100,
        maximum_backups=32,
        deadline_seconds=0.035,
        step_seconds=0.010,
    )

    assert receipt.stop_reason == "decision_deadline"
    assert 0 < receipt.completed_backups < 32
    assert receipt.search_step_calls >= receipt.completed_backups
    assert receipt.elapsed_seconds < 0.30


def test_clean_deadline_with_zero_backups_uses_typed_direct_fallback() -> None:
    trace = _Trace()
    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(
            lane,
            trace,
            step_seconds=0.050,
            slow_lane_one_first_seconds=0.050,
        ),
        make_packet=lambda lane, observation: (lane, observation),
        evaluate_batch=_normal_leaf_batch,
        coalesce_seconds=0.0,
        cleanup_timeout_seconds=0.20,
    )
    try:
        with pytest.raises(
            AsyncDirectFallbackRequired, match="completed no backups"
        ) as raised:
            search.run_decision(
                root_observation={"current": {"yourIndex": 0}},
                search_inputs=({}, {}),
                root_state_key="zero-backup-deadline",
                root_actions=((0,), (1,)),
                root_priors=(0.0, 1.0),
                root_seat=0,
                deadline_monotonic=time.monotonic() + 0.005,
            )
        cleanup = raised.value.cleanup_receipt
        assert cleanup is raised.value.receipt
        assert cleanup is not None
        assert cleanup.arena_count == cleanup.unique_handle_count == 2
        assert cleanup.search_begin_calls == 2
        assert cleanup.search_step_calls == 2
        assert cleanup.completed_backups == 0
        assert cleanup.microbatch_sizes == ()
        assert cleanup.max_simulator_calls_in_flight == 2
        assert cleanup.completion_order == ()
        assert cleanup.per_lane_depth == (0, 0)
        assert tuple(chain[0] for chain in cleanup.per_lane_search_id_chains) == (
            0,
            0,
        )
        assert cleanup.per_lane_handle_identities == (100, 101)
        assert cleanup.distinct_search_begin_composite_count == 2
        assert cleanup.search_release_calls >= 2
        assert cleanup.search_end_calls == 2
        assert cleanup.outstanding_virtual_loss == 0
        assert cleanup.stop_reason == "decision_deadline"
        assert cleanup.minimum_backups_before_stability == 8
        assert cleanup.stable_root_leader_observations == 3
        assert cleanup.maximum_backups_per_decision == 32
        assert cleanup.root_seat == 0
    finally:
        search.close()


def _continuation_receipt(
    *,
    fingerprint_disagreement: bool = False,
    action_disagreement: bool = False,
    chance_boundary: bool = False,
    actor_change: bool = False,
    smoke_depth: int = 2,
    colliding_state_keys: bool = False,
):
    trace = _Trace()

    def evaluate(rows):
        leaves = []
        for lane, observation in rows:
            depth = int(observation["depth"])
            fingerprint = f"public-depth={depth}"
            if fingerprint_disagreement:
                fingerprint += f";lane={lane}"
            priors = (
                ((1.0, 0.0) if lane == 0 else (0.0, 1.0))
                if action_disagreement and depth == 1
                else (0.0, 1.0)
            )
            leaves.append(
                DecodedLeaf(
                    state_key=(
                        f"shared-collision-depth={depth}"
                        if colliding_state_keys
                        else f"private-lane={lane};depth={depth}"
                    ),
                    value=0.25,
                    legal_actions=() if chance_boundary and depth == 1 else ((0,), (1,)),
                    priors=() if chance_boundary and depth == 1 else priors,
                    boundary=bool(chance_boundary and depth == 1),
                    actor_seat=(1 if actor_change and depth == 1 else 0),
                    observation_fingerprint=fingerprint,
                )
            )
        return tuple(leaves)

    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(
            lane,
            trace,
            step_seconds=0.001,
            slow_lane_one_first_seconds=0.001,
        ),
        make_packet=lambda lane, observation: (lane, observation),
        evaluate_batch=evaluate,
        coalesce_seconds=0.010,
        minimum_backups_before_stability=32,
        stable_root_leader_observations=100,
        maximum_backups_per_decision=32,
    )
    try:
        return search.run_decision(
            root_observation={"current": {"yourIndex": 0}},
            search_inputs=({}, {}),
            root_state_key="continuation-root",
            root_actions=((0,), (1,)),
            root_priors=(0.0, 1.0),
            root_seat=0,
            deadline_monotonic=time.monotonic() + 1.0,
            smoke_min_depth_per_lane=(1 if chance_boundary else smoke_depth),
        )
    finally:
        search.close()


def test_two_lane_exact_agreement_emits_only_backed_subsequent_continuation() -> None:
    receipt = _continuation_receipt()

    assert receipt.selected_action == (1,)
    assert receipt.principal_variation == (
        {"observation_fingerprint": "public-depth=1", "action": [1]},
    )


def test_continuation_is_bounded_to_eight_subsequent_actions() -> None:
    receipt = _continuation_receipt(smoke_depth=10)

    assert len(receipt.principal_variation) == 8
    assert [
        row["observation_fingerprint"] for row in receipt.principal_variation
    ] == [f"public-depth={depth}" for depth in range(1, 9)]
    assert all(row["action"] == [1] for row in receipt.principal_variation)


def test_colliding_caller_state_keys_never_merge_lane_private_nodes() -> None:
    receipt = _continuation_receipt(colliding_state_keys=True)

    assert receipt.completed_backups == 4
    assert receipt.per_lane_depth == (2, 2)
    assert receipt.principal_variation == (
        {"observation_fingerprint": "public-depth=1", "action": [1]},
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"fingerprint_disagreement": True},
        {"action_disagreement": True},
        {"chance_boundary": True},
        {"actor_change": True},
    ),
)
def test_continuation_stops_before_any_unproved_next_action(kwargs) -> None:
    receipt = _continuation_receipt(**kwargs)

    assert receipt.principal_variation == ()


def test_opponent_nodes_minimize_root_value_instead_of_cooperating() -> None:
    trace = _Trace()
    search = PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda lane: _Arena(lane, trace),
        make_packet=lambda lane, obs: (lane, obs),
        evaluate_batch=lambda rows: (),
    )
    try:
        opponent = search._node(
            "opponent", ((0,), (1,)), (0.5, 0.5), actor_seat=1
        )
        opponent.edges[0].visits = opponent.edges[1].visits = 10
        opponent.edges[0].value_sum = 8.0
        opponent.edges[1].value_sum = -6.0
        assert search._reserve(opponent, root_seat=0).action == (1,)

        root = search._node("root", ((0,), (1,)), (0.5, 0.5), actor_seat=0)
        root.edges[0].visits = root.edges[1].visits = 10
        root.edges[0].value_sum = 8.0
        root.edges[1].value_sum = -6.0
        assert search._reserve(root, root_seat=0).action == (0,)
    finally:
        search.close()


def test_evaluator_failure_after_ready_completions_is_bounded_and_does_not_hang() -> None:
    """A ready microbatch has already been consumed when evaluation fails.

    This is the dangerous ordering: a coordinator must unwind their reservations
    itself, rather than wait for completions which can never be emitted again.
    """

    result = _fault_case("evaluator_raises")
    assert result["run_error_type"] == AsyncEightWorkerError.__name__
    assert "injected evaluator failure" in str(result["run_error"])
    assert float(result["run_elapsed"]) < 0.60
    assert float(result["close_elapsed"]) < 0.60
    assert result["workers_alive_after_release"] == []


def test_stalled_search_step_and_shutdown_are_bounded_by_cleanup_timeout() -> None:
    """A single wedged native call may poison, but never indefinitely retain, a slot."""

    result = _fault_case("step_stall")
    assert result["run_error_type"] == AsyncEightWorkerError.__name__
    assert float(result["run_elapsed"]) < 0.60
    # The pool may report a poisoned/shutdown error while its native stand-in is
    # blocked, but ``close`` itself must honor the configured 50 ms bound.
    assert float(result["close_elapsed"]) < 0.60
    assert result["close_error_type"] in {None, AsyncEightWorkerError.__name__}
    assert result["workers_alive_after_release"] == []


def test_search_end_error_is_a_terminal_cleanup_result_not_an_unbounded_wait() -> None:
    """A close command that reports ``kind=error`` still satisfies its terminal slot."""

    result = _fault_case("close_error")
    assert result["run_error_type"] == AsyncEightWorkerError.__name__
    assert "SearchEnd failure" in str(result["run_error"])
    assert float(result["run_elapsed"]) < 0.60
    assert float(result["close_elapsed"]) < 0.60
    assert result["workers_alive_after_release"] == []
