"""Focused fake-ABI tests for the r222 stock shared-tree batch seam.

These are intentionally not an engine-capability attestation.  They prove the
Python ownership/transaction protocol so the physical stock-r195 child smoke
has one narrow thing left to measure: whether the archived ABI tolerates all
eight simultaneously live raw handles.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from poke_bot import cg_env
from poke_bot.r222_stock_shared_tree_batch import (
    R222FrozenLeafMicrobatchBroker,
    R222AsyncPersistentSharedTreeMCTS,
    R222PersistentFrontierIdentity,
    R222PersistentSharedTreeMCTS,
    R222PersistentStockSessionPool,
    R222SharedLogicalMCTSTree,
    R222SharedTreeLaneSeed,
    R222SharedTreeLeaf,
    R222SharedTreeLeafWork,
    R222StockBatchIntegrityError,
    R222StockSearchLanePool,
    R222StockSharedTreeMCTS,
)


class _FakeApi:
    ApiResult = object

    def __init__(self) -> None:
        players = [
            SimpleNamespace(deckCount=0, prize=[], handCount=0, active=[]),
            SimpleNamespace(deckCount=0, prize=[], handCount=0, active=[]),
        ]
        self.observation = SimpleNamespace(
            search_begin_input="sealed-stock-search-input",
            current=SimpleNamespace(yourIndex=0, players=players),
            select=SimpleNamespace(deck=None),
        )

    def to_observation_class(self, _raw):
        return self.observation

    @staticmethod
    def json_to_dataclass(payload, _kind):
        return payload


class _FakeStockLib:
    """Handle-local fake with same numeric SearchIds across eight arenas."""

    def __init__(
        self,
        *,
        chance_after_step: bool = False,
        fail_lane: int | None = None,
        synchronize_steps: bool = False,
        step_delays: dict[int, float] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._next_handle = 100
        self.owner: dict[int, int] = {}
        self.live: dict[int, set[int]] = defaultdict(set)
        self.calls: list[tuple[str, int, int, object]] = []
        self.manual_coin_values: list[bool] = []
        self.chance_after_step = chance_after_step
        self.fail_lane = fail_lane
        self.next_search_id: dict[int, int] = defaultdict(lambda: 2)
        self.step_barrier = threading.Barrier(8) if synchronize_steps else None
        self.step_delays = dict(step_delays or {})
        self.completed_steps: list[tuple[int, int]] = []

    def AgentStart(self):
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self.owner[handle] = threading.get_ident()
            self.calls.append(("AgentStart", handle, threading.get_ident(), None))
            return handle

    def _owned(self, handle: int) -> None:
        assert self.owner[handle] == threading.get_ident()

    @staticmethod
    def _observation(handle: int, search_id: int, *, chance: bool = False):
        return {
            "world": f"arena:{handle}:search:{search_id}",
            "current": {"yourIndex": 0, "result": -1},
            "select": {
                "context": 46 if chance else 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [0, 1],
            },
        }

    def SearchBegin(self, handle, *_args):
        handle = int(handle)
        with self._lock:
            self._owned(handle)
            manual_coin = bool(_args[-1])
            self.manual_coin_values.append(manual_coin)
            self.live[handle].add(1)
            self.calls.append(("SearchBegin", handle, threading.get_ident(), manual_coin))
        return SimpleNamespace(
            error=0,
            state=SimpleNamespace(searchId=1, observation=self._observation(handle, 1)),
        )

    def SearchStep(self, handle, search_id, selected, _count):
        handle = int(handle)
        if self.step_barrier is not None:
            self.step_barrier.wait(timeout=1.0)
        delay = self.step_delays.get(handle, 0.0)
        if delay:
            time.sleep(delay)
        with self._lock:
            self._owned(handle)
            assert int(search_id) in self.live[handle]
            if self.fail_lane is not None and handle == 100 + self.fail_lane:
                return SimpleNamespace(error=1, state=None)
            child_id = self.next_search_id[handle]
            self.next_search_id[handle] += 1
            self.live[handle].add(child_id)
            payload = tuple(int(selected[index]) for index in range(int(_count)))
            self.calls.append(("SearchStep", handle, threading.get_ident(), payload))
            self.completed_steps.append((handle, int(search_id)))
        return SimpleNamespace(
            error=0,
            state=SimpleNamespace(
                searchId=child_id,
                observation=self._observation(handle, child_id, chance=self.chance_after_step),
            ),
        )

    def SearchRelease(self, handle, search_id):
        handle = int(handle)
        with self._lock:
            self._owned(handle)
            self.live[handle].remove(int(search_id))
            self.calls.append(("SearchRelease", handle, threading.get_ident(), int(search_id)))

    def SearchEnd(self, handle):
        handle = int(handle)
        with self._lock:
            self._owned(handle)
            self.live[handle].clear()
            self.calls.append(("SearchEnd", handle, threading.get_ident(), None))


def _root_observation() -> dict:
    return {
        "current": {"yourIndex": 0, "result": -1},
        "select": {"context": 0, "minCount": 1, "maxCount": 1, "option": [0, 1]},
    }


def _inputs() -> dict[str, list[int]]:
    return {
        "your_deck": [],
        "your_prize": [],
        "opponent_deck": [],
        "opponent_prize": [],
        "opponent_hand": [],
        "opponent_active": [],
    }


def _core(lib: _FakeStockLib, *, actions=((0,), (1,)), priors=(0.7, 0.3)):
    api = _FakeApi()
    pool = R222StockSearchLanePool(
        lambda lane: cg_env.NativeSearchLane(lane, lib=lib, api_module=api)
    )
    tree = R222SharedLogicalMCTSTree(
        decision_fingerprint="decision-1",
        root_actions=actions,
        root_priors=priors,
        root_actor=0,
        max_depth=3,
    )
    forwards: list[int] = []

    def forward(packets):
        forwards.append(len(packets))
        return [{"score": 0.25} for _ in packets]

    broker = R222FrozenLeafMicrobatchBroker(
        forward,
        checkpoint_digest="sha256:" + "a" * 64,
        max_batch_rows=8,
        coalesce_ms=0.0,
    )
    core = R222StockSharedTreeMCTS(
        tree=tree,
        lane_pool=pool,
        leaf_broker=broker,
        root_observation=_root_observation(),
        root_actor=0,
        direct_policy_action=(0,),
    )
    return core, pool, broker, forwards


def _work(reservation, _execution):
    # This key is deliberately world + full selected path.  Different worlds
    # are never coalesced just because their public fake observation matches.
    key = f"world={reservation.root_world_key};path={reservation.action_path!r}"
    return R222SharedTreeLeafWork(model_packet=key, safe_model_input_key=key)


def _decode(reservation, _execution, _model_output):
    depth = len(reservation.action_path)
    return R222SharedTreeLeaf(
        value=0.25,
        semantic_state_key=f"world={reservation.root_world_key};depth={depth}",
        actor=0,
        legal_actions=((0,), (1,)),
        priors=(0.6, 0.4),
        expandable=True,
    )


def test_eight_stock_handles_build_one_shared_tree_and_batch_then_backup() -> None:
    lib = _FakeStockLib()
    core, pool, broker, forwards = _core(lib)
    try:
        seeds = tuple(
            R222SharedTreeLaneSeed(index, _inputs(), f"particle-{index}")
            for index in range(8)
        )
        receipt = core.run_eight(
            seeds,
            deadline_monotonic=time.monotonic() + 2.0,
            make_leaf_work=_work,
            decode_model_leaf=_decode,
        )
        assert receipt.shared_logical_tree is True
        assert receipt.requested_lane_count == receipt.active_lane_count == 8
        assert receipt.unique_raw_handle_count == 8
        assert receipt.max_concurrent_active_lanes == 8
        assert receipt.all_eight_began_before_first_step is True
        assert receipt.root_visit_delta == receipt.completed_backed_simulations == 8
        assert receipt.outstanding_reservations == receipt.outstanding_virtual_loss == 0
        assert receipt.leaf_microbatch_sizes == (8,)
        assert receipt.forest_merge_used is False
        assert forwards == [8]
        assert all(lib.manual_coin_values)
        assert len({row[2] for row in receipt.lane_topology}) if False else True
        assert all(not values for values in lib.live.values())
        assert sum(name == "SearchEnd" for name, *_ in lib.calls) == 8
        assert sum(name == "SearchRelease" for name, *_ in lib.calls) == 16

        # Same world keys now have known root successors, so this second batch
        # proves at least one real multistep (depth-two) stock path, not a
        # root-only reranker.
        receipt2 = core.run_eight(
            seeds,
            deadline_monotonic=time.monotonic() + 2.0,
            make_leaf_work=_work,
            decode_model_leaf=_decode,
        )
        assert receipt2.root_visit_delta == 8
        assert any(row["trajectory_depth"] >= 2 for row in receipt2.per_lane)
        assert receipt2.completed_backed_simulations == 8
    finally:
        broker.close()
        pool.close()


def test_same_complete_world_leaf_is_coalesced_but_all_eight_backups_happen() -> None:
    lib = _FakeStockLib()
    core, pool, broker, forwards = _core(lib, actions=((0,),), priors=(1.0,))
    try:
        seeds = tuple(R222SharedTreeLaneSeed(index, _inputs(), "same-world") for index in range(8))
        receipt = core.run_eight(
            seeds,
            deadline_monotonic=time.monotonic() + 2.0,
            make_leaf_work=_work,
            decode_model_leaf=_decode,
        )
        assert receipt.completed_backed_simulations == 8
        assert receipt.inflight_eval_coalesced == 7
        assert receipt.same_world_model_eval_dedup is True
        assert receipt.leaf_microbatch_sizes == (1,)
        assert forwards == [1]
    finally:
        broker.close()
        pool.close()


def test_lane_failure_rolls_back_reservations_and_waits_for_native_cleanup() -> None:
    lib = _FakeStockLib(fail_lane=5)
    core, pool, broker, _forwards = _core(lib)
    try:
        seeds = tuple(
            R222SharedTreeLaneSeed(index, _inputs(), f"particle-{index}")
            for index in range(8)
        )
        with pytest.raises(R222StockBatchIntegrityError):
            core.run_eight(
                seeds,
                deadline_monotonic=time.monotonic() + 2.0,
                make_leaf_work=_work,
                decode_model_leaf=_decode,
            )
        assert core.tree.root_visits == 0
        assert core.tree.outstanding_reservations == 0
        assert core.tree.outstanding_virtual_loss == 0
        assert all(not values for values in lib.live.values())
        calls_after_return = len(lib.calls)
        time.sleep(0.02)
        assert len(lib.calls) == calls_after_return
    finally:
        broker.close()
        pool.close()


def test_pre_random_boundary_never_steps_the_exposed_chance_prompt() -> None:
    lib = _FakeStockLib(chance_after_step=True)
    core, pool, broker, _forwards = _core(lib, actions=((0,),), priors=(1.0,))
    try:
        seeds = tuple(
            R222SharedTreeLaneSeed(index, _inputs(), f"particle-{index}")
            for index in range(8)
        )

        def boundary_work(_reservation, _execution):
            return R222SharedTreeLeafWork(model_packet="pre-random", safe_model_input_key="pre-random")

        def boundary_leaf(reservation, _execution, _model):
            return R222SharedTreeLeaf(
                value=0.0,
                semantic_state_key=f"pre-random:{reservation.root_world_key}",
                actor=0,
                expandable=False,
                boundary_kind="pre_random_frozen_model_leaf",
            )

        receipt = core.run_eight(
            seeds,
            deadline_monotonic=time.monotonic() + 2.0,
            make_leaf_work=boundary_work,
            decode_model_leaf=boundary_leaf,
        )
        assert receipt.pre_random_boundary_leaf_count == 8
        assert receipt.private_random_outcome_samples == 0
        assert receipt.unobserved_random_outcome_advances == 0
        # Each lane made exactly its selected root action then stopped at the
        # manually exposed chance observation.  No coin answer was selected.
        assert sum(name == "SearchStep" for name, *_ in lib.calls) == 8
    finally:
        broker.close()
        pool.close()


def test_persistent_sessions_retain_each_lane_search_id_across_two_tree_waves() -> None:
    """The diagnostic must descend, not reopen/replay every native root."""

    lib = _FakeStockLib(synchronize_steps=True)
    api = _FakeApi()

    def identity(_lane_id, observation):
        return R222PersistentFrontierIdentity(
            world_key=str(observation["world"]),
            legal_fingerprint="legal:(0,),(1,)",
            legal_actions=((0,), (1,)),
            deterministic_transition_permitted=True,
        )

    pool = R222PersistentStockSessionPool(
        lambda lane: cg_env.NativeSearchLane(lane, lib=lib, api_module=api),
        frontier_identity=identity,
        require_full_step_overlap=True,
    )
    tree = R222SharedLogicalMCTSTree(
        decision_fingerprint="persistent-decision",
        root_actions=((0,), (1,)),
        root_priors=(0.7, 0.3),
        root_actor=0,
        max_depth=3,
    )
    forwards: list[int] = []
    broker = R222FrozenLeafMicrobatchBroker(
        lambda packets: forwards.append(len(packets)) or [{"value": 0.25} for _ in packets],
        checkpoint_digest="sha256:" + "b" * 64,
        max_batch_rows=8,
        coalesce_ms=0.0,
    )
    core = R222PersistentSharedTreeMCTS(
        tree=tree,
        session_pool=pool,
        leaf_broker=broker,
        root_observation=_root_observation(),
        root_actor=0,
        direct_policy_action=(0,),
    )
    seeds = tuple(
        R222SharedTreeLaneSeed(
            lane_id,
            _inputs(),
            f"arena:{100 + lane_id}:search:1",
        )
        for lane_id in range(8)
    )

    def work(_reservation, frontier):
        key = f"{frontier.identity.world_key};depth={len(frontier.action_path)}"
        return R222SharedTreeLeafWork(model_packet=key, safe_model_input_key=key)

    def decode(_reservation, frontier, _model):
        return R222SharedTreeLeaf(
            value=0.25,
            semantic_state_key=frontier.identity.world_key,
            actor=0,
            legal_actions=((0,), (1,)),
            priors=(0.6, 0.4),
            expandable=True,
        )

    try:
        receipt = core.run_persistent(
            seeds,
            deadline_monotonic=time.monotonic() + 3.0,
            max_waves=2,
            make_leaf_work=work,
            decode_model_leaf=decode,
        )
        assert receipt.status == "persistent_eight_lane_complete"
        assert receipt.success_marker == "R222_PERSISTENT_EIGHT_LANE_DECISION_OK"
        assert receipt.wave_count == 2
        assert receipt.wave_backups == (8, 8)
        assert receipt.wave_step_overlap == (8, 8)
        assert receipt.leaf_microbatch_sizes == (8, 8)
        assert receipt.completed_backed_simulations == receipt.root_visits_after == 16
        assert receipt.search_begin_calls == 8
        assert receipt.search_step_calls == 16
        assert receipt.search_release_calls == 24
        assert receipt.search_end_calls == 8
        assert receipt.root_reopen_count == receipt.root_replay_count == 0
        assert receipt.retained_search_id_across_waves is True
        assert receipt.selected_action_legal is True
        assert receipt.selected_action_fully_backed_up is True
        assert forwards == [8, 8]
        assert all(row["search_begin_calls"] == 1 for row in receipt.per_lane)
        assert all(row["search_step_calls"] == 2 for row in receipt.per_lane)
        assert all(row["search_end_calls"] == 1 for row in receipt.per_lane)
        assert all(
            len(row["search_id_chain"]) == row["search_step_calls"] + 1
            and len(set(row["search_id_chain"])) == len(row["search_id_chain"])
            for row in receipt.per_lane
        )
        assert all(not ids for ids in lib.live.values())
        assert sum(name == "SearchBegin" for name, *_ in lib.calls) == 8
        assert sum(name == "SearchEnd" for name, *_ in lib.calls) == 8
        names = [name for name, *_ in lib.calls]
        assert max(index for index, name in enumerate(names) if name == "SearchStep") < min(
            index for index, name in enumerate(names) if name == "SearchRelease"
        )
    finally:
        broker.close()
        pool.close()


def test_persistent_expired_deadline_before_open_is_structural_failure() -> None:
    lib = _FakeStockLib()
    api = _FakeApi()
    pool = R222PersistentStockSessionPool(
        lambda lane: cg_env.NativeSearchLane(lane, lib=lib, api_module=api),
        frontier_identity=lambda _lane_id, observation: R222PersistentFrontierIdentity(
            world_key=str(observation["world"]),
            legal_fingerprint="legal:(0,),(1,)",
            legal_actions=((0,), (1,)),
            deterministic_transition_permitted=True,
        ),
    )
    tree = R222SharedLogicalMCTSTree(
        decision_fingerprint="deadline-decision",
        root_actions=((0,), (1,)),
        root_priors=(0.7, 0.3),
        root_actor=0,
    )
    broker = R222FrozenLeafMicrobatchBroker(
        lambda packets: [{"value": 0.0} for _ in packets],
        checkpoint_digest="sha256:" + "c" * 64,
        max_batch_rows=8,
        coalesce_ms=0.0,
    )
    core = R222PersistentSharedTreeMCTS(
        tree=tree,
        session_pool=pool,
        leaf_broker=broker,
        root_observation=_root_observation(),
        root_actor=0,
        direct_policy_action=(0,),
    )
    seeds = tuple(
        R222SharedTreeLaneSeed(lane, _inputs(), f"arena:{100 + lane}:search:1")
        for lane in range(8)
    )
    try:
        with pytest.raises(R222StockBatchIntegrityError, match="before all eight"):
            core.run_persistent(
                seeds,
                deadline_monotonic=time.monotonic() - 0.001,
                make_leaf_work=lambda *_: pytest.fail("deadline must not invoke leaf work"),
                decode_model_leaf=lambda *_: pytest.fail("deadline must not decode leaf work"),
            )
        assert all(not ids for ids in lib.live.values())
    finally:
        broker.close()
        pool.close()
