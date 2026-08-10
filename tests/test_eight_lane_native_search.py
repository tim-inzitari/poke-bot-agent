from __future__ import annotations

import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from poke_bot import cg_env
from poke_bot.eight_lane_belief_forest import (
    EIGHT_LANE_COUNT,
    EightLaneDeadlineExceeded,
    EightLaneSearchError,
    PersistentEightLanePool,
    merge_complete_root_statistics,
)
from poke_bot.mcts import MCTSResult
from poke_bot.search_targets import build_search_target


class _FakeApi:
    ApiResult = object

    def __init__(self) -> None:
        players = [
            SimpleNamespace(deckCount=0, prize=[], handCount=0, active=[]),
            SimpleNamespace(deckCount=0, prize=[], handCount=0, active=[]),
        ]
        self.observation = SimpleNamespace(
            search_begin_input="sealed-search-input",
            current=SimpleNamespace(yourIndex=0, players=players),
            select=SimpleNamespace(deck=None),
        )

    def to_observation_class(self, _raw):
        return self.observation

    @staticmethod
    def json_to_dataclass(payload, _kind):
        return payload


class _FakeNativeLib:
    """Handle-local fake that deliberately reuses numeric search IDs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_handle = 1
        self.owner_by_handle: dict[int, int] = {}
        self.live_by_handle: dict[int, set[int]] = defaultdict(set)
        self.calls: list[tuple[str, int, int]] = []

    def AgentStart(self):
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self.owner_by_handle[handle] = threading.get_ident()
            self.calls.append(("AgentStart", handle, threading.get_ident()))
            return handle

    def _owned(self, handle: int) -> None:
        assert self.owner_by_handle[handle] == threading.get_ident()

    def SearchBegin(self, handle, *_args):
        handle = int(handle)
        with self._lock:
            self._owned(handle)
            # The same numeric ID exists independently in all eight arenas.
            self.live_by_handle[handle].add(1)
            self.calls.append(("SearchBegin", handle, threading.get_ident()))
        return SimpleNamespace(
            error=0,
            state=SimpleNamespace(searchId=1, observation={"lane": handle}),
        )

    def SearchStep(self, handle, search_id, _select, _count):
        handle = int(handle)
        search_id = int(search_id)
        with self._lock:
            self._owned(handle)
            if search_id not in self.live_by_handle[handle]:
                return SimpleNamespace(error=1, state=None)
            self.live_by_handle[handle].add(2)
            self.calls.append(("SearchStep", handle, threading.get_ident()))
        return SimpleNamespace(
            error=0,
            state=SimpleNamespace(searchId=2, observation={"lane": handle}),
        )

    def SearchRelease(self, handle, search_id):
        handle = int(handle)
        search_id = int(search_id)
        with self._lock:
            self._owned(handle)
            self.live_by_handle[handle].remove(search_id)
            self.calls.append(("SearchRelease", handle, threading.get_ident()))

    def SearchEnd(self, handle):
        handle = int(handle)
        with self._lock:
            self._owned(handle)
            self.live_by_handle[handle].clear()
            self.calls.append(("SearchEnd", handle, threading.get_ident()))


def _search_inputs() -> dict[str, list[int]]:
    return {
        "your_deck": [],
        "your_prize": [],
        "opponent_deck": [],
        "opponent_prize": [],
        "opponent_hand": [],
        "opponent_active": [],
    }


def test_eight_lanes_keep_exactly_eight_persistent_thread_owned_handles() -> None:
    lib = _FakeNativeLib()
    api = _FakeApi()
    created: dict[int, cg_env.NativeSearchLane] = {}

    def factory(lane_id: int) -> cg_env.NativeSearchLane:
        lane = cg_env.NativeSearchLane(lane_id, lib=lib, api_module=api)
        created[lane_id] = lane
        return lane

    pool = PersistentEightLanePool(factory)
    try:
        first_topology = pool.lane_topology

        def one_cycle(lane_id, backend, _cancel):
            root = backend.search_begin({}, _search_inputs(), manual_coin=True)
            child = backend.search_step(root.searchId, [0])
            backend.search_release(child.searchId)
            backend.search_release(root.searchId)
            backend.search_end()
            return lane_id

        deadline = time.monotonic() + 2.0
        assert pool.run_all(one_cycle, deadline_monotonic=deadline) == list(
            range(EIGHT_LANE_COUNT)
        )
        assert pool.run_all(
            one_cycle, deadline_monotonic=time.monotonic() + 2.0
        ) == list(range(EIGHT_LANE_COUNT))

        assert len(created) == EIGHT_LANE_COUNT
        assert len(lib.owner_by_handle) == EIGHT_LANE_COUNT
        assert len({row["handle_identity"] for row in first_topology}) == 8
        assert pool.lane_topology == first_topology
        assert sum(name == "AgentStart" for name, *_ in lib.calls) == 8
        for name, handle, thread_id in lib.calls:
            assert lib.owner_by_handle[handle] == thread_id, name
        assert all(not live for live in lib.live_by_handle.values())
    finally:
        pool.close()


def test_native_lane_refuses_cross_thread_use_and_end_is_handle_scoped() -> None:
    lib = _FakeNativeLib()
    api = _FakeApi()
    created: dict[int, cg_env.NativeSearchLane] = {}

    def factory(lane_id: int) -> cg_env.NativeSearchLane:
        lane = cg_env.NativeSearchLane(lane_id, lib=lib, api_module=api)
        created[lane_id] = lane
        return lane

    pool = PersistentEightLanePool(factory)
    try:
        with pytest.raises(RuntimeError, match="belongs to thread"):
            created[0].search_end()

        def scoped_end(lane_id, backend, _cancel):
            root = backend.search_begin({}, _search_inputs())
            if lane_id == 0:
                backend.search_end()
                return "ended"
            child = backend.search_step(root.searchId, [0])
            backend.search_end()
            return child.searchId

        result = pool.run_all(
            scoped_end, deadline_monotonic=time.monotonic() + 2.0
        )
        assert result[0] == "ended"
        assert result[1:] == [2] * 7
        # Clearing lane 0's ID=1 did not invalidate the colliding ID=1 in any
        # other native arena.
        assert sum(name == "SearchStep" for name, *_ in lib.calls) == 7
    finally:
        pool.close()


def _result(
    action_combos: list[list[int]],
    visits: list[int],
    *,
    selected: list[int],
    value: float,
    lane: int,
) -> MCTSResult:
    target = build_search_target(
        action_combos,
        visits,
        value,
        prior=[0.5, 0.5],
        diagnostics={
            "action_space_mode": "complete_materialized",
            "complete_ordered_action_count": len(action_combos),
            "root_information_state_fingerprint": "sealed-root",
            "selected_action_fully_backed_up": True,
            "lane": lane,
        },
    )
    return MCTSResult(
        select=selected,
        target=target,
        sims_run=sum(visits),
        elapsed_s=0.01 * (lane + 1),
    )


def test_eight_lane_root_merge_is_completion_order_independent_and_canonical() -> None:
    actions = [[7], [3]]
    rows = []
    for lane in range(8):
        visits = [3, 1] if lane % 2 == 0 else [1, 3]
        selected = actions[0] if lane % 2 == 0 else actions[1]
        rows.append((lane, _result(actions, visits, selected=selected, value=0.1, lane=lane)))
    forward = merge_complete_root_statistics(
        rows,
        canonical_legal_actions=actions,
        canonical_root_fingerprint="sealed-root",
        elapsed_s=1.0,
    )
    reverse = merge_complete_root_statistics(
        list(reversed(rows)),
        canonical_legal_actions=actions,
        canonical_root_fingerprint="sealed-root",
        elapsed_s=1.0,
    )
    assert forward.select == reverse.select == [7]
    assert forward.target.visits == reverse.target.visits == [16, 16]
    assert forward.target.diagnostics["aggregate_tie_break"] == (
        "earliest_canonical_legal_action"
    )
    assert forward.target.diagnostics["root_action_stable"] is False

    malformed = list(rows)
    malformed[3] = (
        3,
        _result(list(reversed(actions)), [1, 3], selected=[3], value=0.1, lane=3),
    )
    with pytest.raises(EightLaneSearchError, match="legal action order"):
        merge_complete_root_statistics(
            malformed,
            canonical_legal_actions=actions,
            canonical_root_fingerprint="sealed-root",
            elapsed_s=1.0,
        )


def test_one_lane_failure_rejects_every_partial_forest_result() -> None:
    lib = _FakeNativeLib()
    api = _FakeApi()
    pool = PersistentEightLanePool(
        lambda lane_id: cg_env.NativeSearchLane(
            lane_id, lib=lib, api_module=api
        )
    )
    cleaned: list[int] = []
    try:
        def operation(lane_id, backend, _cancel):
            backend.search_begin({}, _search_inputs())
            try:
                if lane_id == 5:
                    raise ValueError("injected lane failure")
                return lane_id
            finally:
                backend.search_end()
                cleaned.append(lane_id)

        with pytest.raises(EightLaneSearchError, match="lane=5"):
            pool.run_all(
                operation, deadline_monotonic=time.monotonic() + 2.0
            )
        assert sorted(cleaned) == list(range(8))
        assert all(not live for live in lib.live_by_handle.values())
    finally:
        pool.close()


def test_deadline_cancels_and_joins_all_lanes_before_returning() -> None:
    lib = _FakeNativeLib()
    api = _FakeApi()
    pool = PersistentEightLanePool(
        lambda lane_id: cg_env.NativeSearchLane(
            lane_id, lib=lib, api_module=api
        )
    )
    finished: list[int] = []
    try:
        def operation(lane_id, backend, cancel):
            backend.search_begin({}, _search_inputs())
            try:
                while not cancel.is_set():
                    time.sleep(0.001)
                return lane_id
            finally:
                backend.search_end()
                finished.append(lane_id)

        with pytest.raises(EightLaneDeadlineExceeded):
            pool.run_all(
                operation, deadline_monotonic=time.monotonic() + 0.03
            )
        assert sorted(finished) == list(range(8))
        calls_after_return = len(lib.calls)
        time.sleep(0.02)
        assert len(lib.calls) == calls_after_return
    finally:
        pool.close()


def test_belief_mcts_runtime_search_calls_use_only_the_injected_backend() -> None:
    source = __import__("inspect").getsource(
        __import__("poke_bot.belief_mcts", fromlist=["BeliefMCTS"]).BeliefMCTS
    )
    assert 'search_backend = getattr(self, "search_backend", cg_env)' in source
    assert "search_backend.search_begin(" in source
    assert "search_backend.search_step(" in source
    assert "search_backend.search_end()" in source
    assert "cg_env.search_begin(" not in source
    assert "cg_env.search_step(" not in source
    assert "cg_env.search_end()" not in source
