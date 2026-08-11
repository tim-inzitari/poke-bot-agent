from __future__ import annotations

from types import SimpleNamespace

import pytest

import poke_bot.r250_recovering_serial_tree as recovery


ROOT = {"select": {"option": [1, 2]}, "current": {"yourIndex": 0}}


class FakeLane:
    def __init__(self, lane_id: int, fault: bool) -> None:
        self.lane_id = lane_id
        self._fault = fault

    @property
    def faults(self):
        return (
            ({"lane_id": self.lane_id, "code": "response_timeout"},)
            if self._fault
            else ()
        )

    def telemetry_snapshot(self):
        return {
            "lane_id": self.lane_id,
            "handle_identity": f"process:{self.lane_id}:handle:x",
            "faults": list(self.faults),
        }


class FakeCore:
    outcomes = []
    instances = []

    def __init__(self, *, arena_factory, **_kwargs):
        self.outcome = self.outcomes.pop(0)
        self.lanes = [arena_factory(0)]
        self.closed = False
        self.deadlines = []
        self.instances.append(self)

    def run_decision(self, **kwargs):
        self.deadlines.append(kwargs["deadline_monotonic"])
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(selected_action=(0,))

    def close(self):
        self.closed = True


def _tree(monkeypatch, outcomes):
    FakeCore.outcomes = list(outcomes)
    FakeCore.instances = []
    monkeypatch.setattr(recovery, "PersistentAsyncEightWorkerMCTS", FakeCore)
    attempt_flags = iter(isinstance(row, Exception) for row in outcomes)
    next_faults = [next(attempt_flags)]

    def serial_factory(lane_id):
        assert lane_id == 0
        lane = FakeLane(lane_id, next_faults[0])
        try:
            next_faults[0] = next(attempt_flags)
        except StopIteration:
            pass
        return lane

    return recovery.R250RecoveringSerialTree(
        arena_factory=serial_factory,
        make_packet=lambda lane, obs: (lane, obs),
        evaluate_batch=lambda rows: rows,
    )


def _kwargs():
    import time

    return {
        "root_observation": ROOT,
        "search_inputs": ({},),
        "root_state_key": "root",
        "root_actions": ((0,), (1,)),
        "root_priors": (0.5, 0.5),
        "root_seat": 0,
        "deadline_monotonic": time.monotonic() + 0.5,
    }


def test_lane_fault_discards_partial_tree_and_retries_fresh_serial_lane(monkeypatch):
    tree = _tree(
        monkeypatch,
        [RuntimeError("contained_native_lane_fault"), object()],
    )
    result = tree.run_decision(**_kwargs())
    assert result.selected_action == (0,)
    assert len(FakeCore.instances) == 2
    assert FakeCore.instances[0].closed is True
    telemetry = tree.last_decision_recovery
    assert telemetry["attempt_count"] == 2
    assert telemetry["serial_lane_count"] == 1
    assert telemetry["recovered_search"] is True
    assert telemetry["exhausted_direct_fallback"] is False
    assert telemetry["attempts"][0]["status"] == "failed"
    assert telemetry["attempts"][1]["status"] == "complete"


def test_two_serial_attempt_faults_raise_explicit_recovery_exhaustion(monkeypatch):
    tree = _tree(
        monkeypatch,
        [
            RuntimeError("contained_native_lane_fault"),
            RuntimeError("contained_native_lane_fault"),
            object(),
        ],
    )
    with pytest.raises(recovery.R250SerialRecoveryExhausted) as caught:
        tree.run_decision(**_kwargs())
    telemetry = caught.value.telemetry
    assert telemetry["attempt_count"] == 2
    assert telemetry["recovered_search"] is False
    assert telemetry["exhausted_direct_fallback"] is True
    # A fresh serial lane is already available for the next decision.
    assert len(FakeCore.instances) == 3


def test_non_lane_tree_error_is_not_retried(monkeypatch):
    tree = _tree(monkeypatch, [ValueError("model leaf shape drift"), object()])
    # Remove injected lane faults so this is provably not a native boundary.
    for lane in tree._lanes:
        lane._fault = False
    with pytest.raises(ValueError, match="model leaf shape drift"):
        tree.run_decision(**_kwargs())
    assert len(FakeCore.instances) == 1
