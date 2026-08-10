from __future__ import annotations

import multiprocessing
import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from poke_bot.r228_async_shared_tree_queue import (
    AsyncEightWorkerError,
    DecodedLeaf,
    PersistentAsyncEightWorkerMCTS,
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
    def __init__(self, lane: int, trace: _Trace) -> None:
        self.lane = lane
        self.trace = trace
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
        # Lane 7's first simulator call is deliberately slow.  A genuinely
        # asynchronous coordinator should evaluate/requeue faster lanes before
        # this call returns, rather than wait at an all-eight wave barrier.
        time.sleep(0.030 if self.lane == 7 and depth == 0 else 0.001)
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
                if fault == "evaluator_raises" and len(rows) == 8
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
            search_inputs=tuple({} for _ in range(8)),
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


def test_one_local_decision_uses_async_eight_worker_model_guided_tree() -> None:
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
            search_inputs=tuple({} for _ in range(8)),
            root_state_key="one-local-decision",
            root_actions=((0,), (1,)),
            root_priors=(0.2, 0.8),
            root_seat=0,
            deadline_monotonic=time.monotonic() + 2.0,
            smoke_min_depth_per_lane=2,
        )
    finally:
        search.close()

    assert receipt.arena_count == receipt.unique_handle_count == 8
    assert receipt.search_begin_calls == 8
    assert receipt.search_step_calls == receipt.completed_backups == 16
    assert receipt.max_simulator_calls_in_flight == 8
    assert receipt.root_visits == 16
    assert receipt.selected_action in ((0,), (1,))
    assert receipt.selected_action_visits >= 1
    assert receipt.selected_action_prior > 0.0
    assert 0.0 <= receipt.selected_action_value <= 1.0
    assert receipt.per_lane_depth == (2,) * 8
    assert all(len(chain) == 3 for chain in receipt.per_lane_search_id_chains)
    assert receipt.search_release_calls == 24
    assert receipt.search_end_calls == 8
    assert receipt.outstanding_virtual_loss == 0
    assert sum(receipt.microbatch_sizes) == 16
    assert model_batches
    assert all(not ids for ids in trace.live.values())

    starts = {(lane, depth): when for kind, lane, depth, when in trace.events if kind == "start"}
    finishes = {(lane, depth): when for kind, lane, depth, when in trace.events if kind == "finish"}
    assert starts[(0, 1)] < finishes[(7, 0)]


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
    """All eight ready rows have already been consumed when evaluation fails.

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
