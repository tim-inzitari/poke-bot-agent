from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from poke_bot.r228_async_shared_tree_queue import AsyncEightWorkerError, DecodedLeaf
from poke_bot.r253_restarting_serial_mcts import R253RestartingSerialMCTS


ROOT = {"root": True, "current": {"yourIndex": 0}}


class FakeLane:
    def __init__(self, lane_id: int, *, cleanup_fault: bool = False) -> None:
        assert lane_id == 0
        self.lane_id = lane_id
        self.handle_identity = "process:10:handle:20"
        self.cleanup_fault = cleanup_fault
        self.begins: list[tuple[dict, dict, bool]] = []
        self.steps: list[tuple[int, tuple[int, ...]]] = []
        self.releases: list[int] = []
        self.ends = 0
        self.closed = False
        self._faults: list[dict] = []

    @property
    def faults(self):
        return tuple(self._faults)

    def search_begin(self, observation, search_inputs, *, manual_coin=True):
        self.begins.append((dict(observation), dict(search_inputs), manual_coin))
        # The official handle may reuse raw SearchId 0 after SearchEnd.
        return SimpleNamespace(searchId=0, observation=dict(observation))

    def search_step(self, search_id, action):
        action = tuple(action)
        self.steps.append((int(search_id), action))
        return SimpleNamespace(
            searchId=1,
            observation={"leaf": action[0], "current": {"yourIndex": 1}},
        )

    def search_release(self, search_id):
        self.releases.append(int(search_id))
        if self.cleanup_fault and not self._faults:
            self._faults.append({"code": "response_timeout"})

    def search_end(self):
        self.ends += 1

    def close(self):
        self.closed = True


def _leaf(_lane, observation):
    return observation


def _evaluate(rows):
    result = []
    for row in rows:
        action = int(row["leaf"])
        result.append(
            DecodedLeaf(
                state_key=f"leaf:{action}",
                value=-1.0 if action == 0 else 1.0,
                legal_actions=(),
                priors=(),
                boundary=True,
                actor_seat=1,
            )
        )
    return result


def _run(tree):
    return tree.run_decision(
        root_observation=ROOT,
        search_inputs=({"opponent_deck": [1] * 60},),
        root_state_key="root",
        root_actions=((0,), (1,)),
        root_priors=(0.9, 0.1),
        root_seat=0,
        deadline_monotonic=time.monotonic() + 5.0,
    )


def test_restarts_exact_root_and_backed_value_can_change_root_selection():
    lanes = []

    def factory(lane_id):
        lane = FakeLane(lane_id)
        lanes.append(lane)
        return lane

    tree = R253RestartingSerialMCTS(
        arena_factory=factory,
        make_packet=_leaf,
        evaluate_batch=_evaluate,
        max_rollouts=2,
    )
    receipt = _run(tree)
    lane = lanes[0]

    assert receipt.selected_action == (1,)
    assert receipt.rollout_count == 2
    assert receipt.search_begin_calls == 2
    assert receipt.completed_backups == 2
    assert receipt.root_visits == 2
    assert receipt.rollout_root_actions == ((0,), (1,))
    assert receipt.root_action_visit_counts == (1, 1)
    assert receipt.distinct_root_actions_visited == 2
    assert receipt.rollout_search_id_chains == ((0, 1), (0, 1))
    assert receipt.search_release_calls == 4
    assert receipt.search_end_calls == 2
    assert receipt.outstanding_virtual_loss == 0
    assert lane.begins == [
        (ROOT, {"opponent_deck": [1] * 60}, True),
        (ROOT, {"opponent_deck": [1] * 60}, True),
    ]
    assert lane.releases == [1, 0, 1, 0]
    assert lane.ends == 2


def test_cleanup_fault_invalidates_complete_attempt():
    lane = FakeLane(0, cleanup_fault=True)
    tree = R253RestartingSerialMCTS(
        arena_factory=lambda _lane_id: lane,
        make_packet=_leaf,
        evaluate_batch=_evaluate,
        max_rollouts=2,
    )
    with pytest.raises(AsyncEightWorkerError, match="cleanup failed"):
        _run(tree)


def test_rollout_ceiling_must_allow_two_independent_roots():
    lane = FakeLane(0)
    with pytest.raises(ValueError, match="limits are invalid"):
        R253RestartingSerialMCTS(
            arena_factory=lambda _lane_id: lane,
            make_packet=_leaf,
            evaluate_batch=_evaluate,
            max_rollouts=1,
        )
    assert lane.closed is True
