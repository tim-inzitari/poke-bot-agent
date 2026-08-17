"""Fast parent-controller tests for the r228 Kaggle entrypoint."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from poke_bot import features
from poke_bot.r228_kaggle_broker import R228BrokerError
from poke_bot.r228_kaggle_async_runtime import (
    R228GameplayError,
    canonical_observation_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "submission" / "r228_async_eight_worker_main.py"
_MODULE_IDS = itertools.count()


def _load_entrypoint() -> Any:
    """Load a fresh controller instance so game globals never cross tests."""

    name = f"r228_async_entrypoint_test_{next(_MODULE_IDS)}"
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _branch_observation() -> dict[str, object]:
    return {
        "current": {"yourIndex": 0},
        "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
    }


def _deck_observation() -> dict[str, object]:
    return {"current": None, "select": None}


class _Policy:
    def __init__(self) -> None:
        self._previous_action_token: object = None
        self.history_append_count = 0
        self.board_history: list[object] = []
        self.previous_action_history: list[object] = []
        self.targets: list[dict[str, Any]] = []
        self.collect_targets = False

    def _history_context_limit(self) -> int:
        return 16


class _Direct:
    def __init__(
        self,
        *,
        branch_action: Sequence[int] = (0,),
        deck_action: Sequence[int] = (741,) * 60,
        turn_order_choice: Sequence[int] | None = None,
        selected_stage_probability: float = 0.50,
        emit_target: bool = True,
        trace: list[str] | None = None,
    ) -> None:
        self.branch_action = list(branch_action)
        self.deck_action = list(deck_action)
        self.turn_order_choice = (
            None if turn_order_choice is None else list(turn_order_choice)
        )
        self.selected_stage_probability = float(selected_stage_probability)
        self.emit_target = bool(emit_target)
        self.trace = trace if trace is not None else []
        self.policy = _Policy()
        self.branch_calls = 0
        self.deck_calls = 0

    def agent(self, observation: Mapping[str, Any]) -> list[int]:
        if observation.get("select") is None:
            self.deck_calls += 1
            self.trace.append("direct_deck")
            return list(self.deck_action)
        self.branch_calls += 1
        self.trace.append("direct_branch")
        self.policy.board_history.append(("board", self.branch_calls))
        self.policy.previous_action_history.append(self.policy._previous_action_token)
        limit = self.policy._history_context_limit()
        self.policy.board_history = self.policy.board_history[-limit:]
        self.policy.previous_action_history = self.policy.previous_action_history[-limit:]
        self.policy.history_append_count += 1
        self.policy._previous_action_token = ("direct", list(self.branch_action))
        if self.policy.collect_targets and self.emit_target:
            candidates = features.factorized_action_candidates(dict(observation), [])
            selected_index = candidates.index(list(self.branch_action))
            if len(candidates) == 1:
                probabilities = [1.0]
            else:
                remainder = (1.0 - self.selected_stage_probability) / (
                    len(candidates) - 1
                )
                probabilities = [remainder] * len(candidates)
                probabilities[selected_index] = self.selected_stage_probability
            self.policy.targets.append(
                {
                    "observation": dict(observation),
                    "action": list(self.branch_action),
                    "factorized_stages": [
                        {
                            "action_combos": [list(row) for row in candidates],
                            "policy": probabilities,
                            "selected_index": selected_index,
                        }
                    ],
                    "diagnostics": {
                        "target_source": "history_policy",
                        "trusted": True,
                        "history_length": len(self.policy.board_history),
                    },
                }
            )
        return list(self.branch_action)

    def _ensure_runtime(self) -> tuple[list[int], object, _Policy]:
        self.trace.append("ensure_runtime")
        return [741] * 60, object(), self.policy

    def _turn_order_choice(self, _observation: Mapping[str, Any]) -> list[int] | None:
        return None if self.turn_order_choice is None else list(self.turn_order_choice)


class _FakeBroker:
    """No-model broker surface with deterministic child outcome scripts."""

    def __init__(
        self,
        *,
        outcome: str,
        selected: Sequence[int] = (1,),
        fault: Mapping[str, Any] | None = None,
        receipt_overrides: Mapping[str, Any] | None = None,
        trace: list[str] | None = None,
        close_disables: bool = True,
        **kwargs: Any,
    ) -> None:
        self.outcome = outcome
        self.selected = list(selected)
        self.fault = None if fault is None else dict(fault)
        self.receipt_overrides = (
            None if receipt_overrides is None else dict(receipt_overrides)
        )
        self.trace = trace if trace is not None else []
        self.close_disables = bool(close_disables)
        self.kwargs = dict(kwargs)
        self.begin_game_calls = 0
        self.select_calls: list[tuple[dict[str, Any], list[int]]] = []
        self.note_calls: list[tuple[dict[str, Any], list[int]]] = []
        self.close_calls = 0
        self.degraded = False
        self.disabled = False
        self.last_fault: dict[str, Any] | None = None

    def begin_game(self, *, start_child: bool = True) -> None:
        self.begin_game_calls += 1
        self.trace.append(f"broker_begin:{start_child}")

    def select(
        self, observation: Mapping[str, Any], direct_action: Sequence[int]
    ) -> tuple[list[int], dict[str, Any] | None, dict[str, Any] | None]:
        self.trace.append("broker_select")
        self.select_calls.append((dict(observation), list(direct_action)))
        if self.outcome == "raise":
            raise R228BrokerError("fake child response timed out", code="response_timeout")
        if self.outcome == "fault":
            self.degraded = True
            self.disabled = True
            self.last_fault = dict(
                self.fault
                or {
                    "schema": "poke_bot.r228_kaggle_subprocess_broker/v1",
                    "code": "response_timeout",
                    "message": "fake child stalled",
                    "progress_by_lane": {"1": {"phase": "native_step"}},
                }
            )
            return list(direct_action), None, dict(self.last_fault)
        receipt = {
            "selected_action": list(self.selected),
            "mcts_action_authority": True,
            "mode": "shared_tree_mcts",
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "search_begin_calls": 2,
            "search_release_calls": 2,
            "search_end_calls": 2,
            "per_lane_depth": [1, 1],
            "completed_backups": 2,
            "per_lane_handle_identities": [1001, 1002],
            "per_lane_search_id_chains": [[0], [0]],
            "per_lane_first_search_ids": [0, 0],
            "distinct_search_begin_id_count": 1,
            "distinct_search_begin_composite_count": 2,
            "handle_scoped_first_search_id_composite_states": [
                {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
                {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
            ],
            "microbatch_sizes": [2],
            "max_simulator_calls_in_flight": 2,
            "outstanding_virtual_loss": 0,
            "completed_backups": 2,
            "root_visits": 2,
            "stop_reason": "stable_root_leader",
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations_required": 3,
            "maximum_backups_per_decision": 32,
            "observed_stable_root_leader_observations": 3,
            "root_seat": 0,
            "principal_variation": [],
            "terminal_win_proof": None,
            "proven_deterministic_terminal_win_this_turn": False,
        }
        if self.receipt_overrides is not None:
            receipt.update(self.receipt_overrides)
        return list(self.selected), receipt, None

    def note_direct_action(
        self, observation: Mapping[str, Any], action: Sequence[int]
    ) -> None:
        self.note_calls.append((dict(observation), list(action)))

    def marker_payload(self) -> dict[str, Any]:
        return {
            "schema": "poke_bot.r228_kaggle_subprocess_broker/v1",
            "disabled": self.disabled,
            "degraded": self.degraded,
            "decision_count": len(self.select_calls),
            "child_pid": None,
            "child_identity": None,
            "last_fault": self.last_fault,
            "progress_by_lane": (
                {} if self.last_fault is None else self.last_fault.get("progress_by_lane", {})
            ),
        }

    def close(self) -> None:
        self.close_calls += 1
        if self.close_disables:
            self.disabled = True


class _BrokerFactory:
    def __init__(
        self,
        outcomes: Sequence[str],
        *,
        selected: Sequence[int] = (1,),
        trace: list[str] | None = None,
        close_disables: bool = True,
        receipt_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.selected = list(selected)
        self.trace = trace if trace is not None else []
        self.close_disables = bool(close_disables)
        self.receipt_overrides = (
            None if receipt_overrides is None else dict(receipt_overrides)
        )
        self.instances: list[_FakeBroker] = []

    def __call__(self, **kwargs: Any) -> _FakeBroker:
        self.trace.append("broker_construct")
        outcome = self.outcomes.pop(0) if self.outcomes else "success"
        broker = _FakeBroker(
            outcome=outcome,
            selected=self.selected,
            trace=self.trace,
            close_disables=self.close_disables,
            receipt_overrides=self.receipt_overrides,
            **kwargs,
        )
        self.instances.append(broker)
        return broker


def _install_common_mocks(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    direct: _Direct,
    factory: _BrokerFactory,
    legal: Sequence[Sequence[int]] = ((0,), (1,)),
    token_builder: Callable[..., object] | None = None,
) -> None:
    monkeypatch.setattr(module, "_direct", lambda: direct)
    monkeypatch.setattr(module, "IsolatedR228SearchBroker", factory)
    monkeypatch.setattr(
        module, "_validate_parent_staged_stock_library_identity", lambda: None
    )
    monkeypatch.setattr(
        features,
        "enumerate_action_combos",
        lambda _obs, *, max_combos: [list(action) for action in legal],
    )
    monkeypatch.setattr(
        features,
        "build_option_tokens",
        token_builder
        or (lambda _obs, actions: ("mcts-token", [list(action) for action in actions])),
    )


def _terminal_win_receipt_overrides(
    module: Any,
    observation: Mapping[str, Any],
    *,
    legal: Sequence[Sequence[int]] = ((0,), (1,)),
    action: Sequence[int] = (1,),
    proof_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root_fingerprint = canonical_observation_fingerprint(observation)
    legal_fingerprint = module._legal_order_fingerprint(legal)
    proof: dict[str, Any] = {
        "proof_kind": module.PROVEN_TERMINAL_WIN_PROOF_KIND,
        "root_observation_fingerprint": root_fingerprint,
        "root_legal_order_fingerprint": legal_fingerprint,
        "root_actor_seat": 0,
        "root_action": list(action),
        "selected_action": list(action),
        "terminal_result": "win",
        "terminal_winner_seat": 0,
        "terminal_leaf_reached": True,
        "proof_path_action_count": 1,
        "path_actor_seats": [0],
        "path_no_actor_change_boundary": True,
        "path_no_opponent_boundary_crossing": True,
        "path_no_chance_boundary": True,
        "path_no_unresolved_randomness": True,
        "proof_is_deterministic": True,
        "discovering_lane_id": 0,
    }
    if proof_overrides:
        proof.update(dict(proof_overrides))
    return {
        "stop_reason": module.PROVEN_TERMINAL_WIN_STOP_REASON,
        "completed_backups": 1,
        "root_visits": 1,
        "per_lane_depth": [1, 0],
        "per_lane_search_id_chains": [[0, 1], [0]],
        "microbatch_sizes": [1],
        "root_seat": 0,
        "root_actor_seat": 0,
        "root_observation_fingerprint": root_fingerprint,
        "root_legal_order_fingerprint": legal_fingerprint,
        "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
        "principal_variation": [],
        "terminal_win_proof": proof,
        "proven_deterministic_terminal_win_this_turn": True,
        # Literal child and broker facts.  The parent must reject the terminal
        # action if any of them is absent instead of filling them from a
        # normal receipt/count alias.
        "elapsed_seconds": 0.001,
        "child_search_elapsed_seconds": 0.001,
        "child_search_budget_seconds": module.R234_BROKER_SEARCH_SECONDS,
        "completed_root_backup_count": 1,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "broker_started": True,
        "mcts_child_started": True,
        "mcts_child_called": True,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
    }


def test_branch_precomputes_direct_before_broker_and_rewrites_action_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_entrypoint()
    trace: list[str] = []
    direct = _Direct(branch_action=(0,), trace=trace)
    factory = _BrokerFactory(["success"], selected=(1,), trace=trace)
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    result = module.agent(_branch_observation())

    assert result == [1]
    assert direct.branch_calls == 1
    assert factory.instances[0].select_calls == [(_branch_observation(), [0])]
    assert trace.index("direct_branch") < trace.index("broker_select")
    # ``direct.agent`` appended history; a successful searched action must
    # replace only its prior action token with the actual selected action.
    assert direct.policy._previous_action_token == ("mcts-token", [[1]])
    assert factory.instances[0].kwargs == {
        "stage": module._agent_dir(),
        "action_timeout_seconds": 4.0,
        "search_seconds": 2.0,
        "startup_timeout_seconds": 30.0,
        "reap_grace_seconds": 0.25,
    }


def test_parent_accepts_exact_current_terminal_win_over_direct_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    observation = _branch_observation()
    overrides = _terminal_win_receipt_overrides(module, observation)
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.50)
    factory = _BrokerFactory(
        ["success"], selected=(1,), receipt_overrides=overrides
    )
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(observation) == [1]
    output = capsys.readouterr().out
    assert module.PROVEN_TERMINAL_WIN_STOP_REASON in output
    assert '"terminal_win_action_authority": true' in output
    assert '"proven_deterministic_terminal_win_this_turn": true' in output
    for field in (
        "direct_action_precomputed_and_validated",
        "broker_started",
        "mcts_child_started",
        "mcts_child_called",
        "two_lane_topology_initialized_before_terminal_win_override",
        "terminal_win_proof_backed_up_into_shared_root_tree",
        "terminal_leaf_returned_by_exact_stock_simulator",
        "parent_validated_current_root_observation_legal_fingerprint_and_actor",
        "all_owned_lane_resources_reservations_and_child_cleanup_complete",
    ):
        assert f'"{field}": true' in output
    assert '"completed_root_backup_count": 1' in output
    assert '"terminal_win_proof_count": 1' in output
    assert '"proven_deterministic_terminal_win_this_turn_stop_count": 1' in output
    assert '"child_search_elapsed_seconds": 0.001' in output
    assert '"parent_action_elapsed_seconds":' in output
    assert direct.policy._previous_action_token == ("mcts-token", [[1]])


def test_parent_contains_missing_literal_terminal_marker_fact_to_direct(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_entrypoint()
    observation = _branch_observation()
    overrides = _terminal_win_receipt_overrides(module, observation)
    overrides.pop("mcts_child_called")
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.50)
    factory = _BrokerFactory(
        ["success"], selected=(1,), receipt_overrides=overrides
    )
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(observation) == [0]
    output = capsys.readouterr().out
    assert module.DEGRADED_PREFIX in output
    assert '"selected_action": [0]' in output
    assert '"mcts_child_called": true' not in output


@pytest.mark.parametrize(
    "proof_overrides",
    [
        {"root_observation_fingerprint": "sha256:stale"},
        {"root_legal_order_fingerprint": "sha256:stale"},
        {"path_no_chance_boundary": False},
        {"path_no_unresolved_randomness": False},
        {"path_actor_seats": [1]},
        {"terminal_result": "loss"},
    ],
    ids=[
        "stale_observation",
        "stale_legal_order",
        "chance_boundary",
        "unresolved_randomness",
        "opponent_boundary",
        "loss",
    ],
)
def test_parent_rejects_stale_or_uncertain_terminal_claim_and_contains_to_direct(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    proof_overrides: Mapping[str, Any],
) -> None:
    module = _load_entrypoint()
    observation = _branch_observation()
    overrides = _terminal_win_receipt_overrides(
        module, observation, proof_overrides=proof_overrides
    )
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.50)
    factory = _BrokerFactory(
        ["success"], selected=(1,), receipt_overrides=overrides
    )
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(observation) == [0]
    output = capsys.readouterr().out
    assert module.DEGRADED_PREFIX in output
    assert '"selected_action": [0]' in output
    assert direct.policy._previous_action_token == ("direct", [0])


def test_contained_fault_returns_exact_precomputed_direct_and_emits_degraded_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,))
    factory = _BrokerFactory(["fault"])
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    result = module.agent(_branch_observation())
    output = capsys.readouterr().out

    assert result == [0]
    assert direct.policy._previous_action_token == ("direct", [0])
    assert module.DEGRADED_PREFIX in output
    assert module.HARD_FAILURE_PREFIX not in output
    assert module.FULL_GAMEPLAY_SUCCESS_PREFIX not in output
    assert '"selected_action": [0]' in output
    assert factory.instances[0].disabled


def test_raised_broker_timeout_is_contained_to_direct_action_with_r234_marker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A public broker exception is a contained fault, not parent failure."""

    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,))
    factory = _BrokerFactory(["raise"])
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    result = module.agent(_branch_observation())
    output = capsys.readouterr().out

    assert result == [0]
    assert len(factory.instances[0].select_calls) == 1
    assert factory.instances[0].close_calls == 1
    assert module.DEGRADED_PREFIX in output
    assert "response_timeout" in output
    assert module.HARD_FAILURE_PREFIX not in output


def test_parent_latch_skips_second_select_after_raised_fault_without_broker_latch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The wrapper itself must become direct-only after a protocol fault."""

    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,))
    # This deliberately broken double leaves ``disabled`` false even after
    # close; only the parent-side `_GAME_MCTS_DISABLED` latch can protect the
    # second physical prompt.
    factory = _BrokerFactory(["raise"], close_disables=False)
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(_branch_observation()) == [0]
    first_output = capsys.readouterr().out
    assert module.DEGRADED_PREFIX in first_output
    broker = factory.instances[0]
    assert not broker.disabled

    assert module.agent(_branch_observation()) == [0]
    second_output = capsys.readouterr().out
    assert len(broker.select_calls) == 1
    assert len(broker.note_calls) == 1
    assert module.HARD_FAILURE_PREFIX not in second_output


def test_forced_isfirst_is_journaled_once_without_second_parent_history_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn-order choice stays direct while the child receives one journal row."""

    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), turn_order_choice=(0,))
    factory = _BrokerFactory(["success"])
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    observation = _branch_observation()
    assert module.agent(observation) == [0]

    assert factory.instances == []
    assert direct.branch_calls == 0
    assert direct.policy.history_append_count == 0
    assert direct.policy._previous_action_token is None
    assert module._GAME_DIRECT_JOURNAL == [
        {"observation": observation, "action": [0]}
    ]
    assert "ensure_runtime" not in direct.trace


def test_root_over_65536_hard_fails_before_broker_creation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,))
    factory = _BrokerFactory(["success"])
    monkeypatch.setattr(module, "_direct", lambda: direct)
    monkeypatch.setattr(module, "IsolatedR228SearchBroker", factory)
    monkeypatch.setattr(
        module, "_validate_parent_staged_stock_library_identity", lambda: None
    )
    seen_caps: list[int] = []

    def over_cap(_obs: Mapping[str, Any], *, max_combos: int) -> list[list[int]]:
        seen_caps.append(max_combos)
        raise OverflowError("ordered legal set exceeds cap")

    monkeypatch.setattr(features, "enumerate_action_combos", over_cap)

    with pytest.raises(RuntimeError, match="complete_root_legal_order_invalid_or_over_cap"):
        module.agent(_branch_observation())

    assert direct.branch_calls == 1
    assert seen_caps == [65_536]
    assert factory.instances == []
    assert module.HARD_FAILURE_PREFIX in capsys.readouterr().out


def test_next_deck_request_emits_success_after_nondegraded_searched_game(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), deck_action=(741,) * 60)
    factory = _BrokerFactory(["success", "success"], selected=(1,))
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(_branch_observation()) == [1]
    capsys.readouterr()
    assert module.agent(_deck_observation()) == [741] * 60
    output = capsys.readouterr().out

    assert output.count(module.FULL_GAMEPLAY_SUCCESS_PREFIX) == 1
    assert module.DEGRADED_PREFIX not in output
    assert factory.instances[0].close_calls == 1
    assert len(factory.instances) == 1
    assert factory.instances[0].begin_game_calls == 1


def test_next_deck_request_never_emits_success_after_contained_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), deck_action=(741,) * 60)
    factory = _BrokerFactory(["fault", "success"])
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(_branch_observation()) == [0]
    capsys.readouterr()
    assert module.agent(_deck_observation()) == [741] * 60
    output = capsys.readouterr().out

    assert module.FULL_GAMEPLAY_SUCCESS_PREFIX not in output
    assert module.HARD_FAILURE_PREFIX not in output
    assert factory.instances[0].close_calls == 1
    assert len(factory.instances) == 1
    assert factory.instances[0].begin_game_calls == 1


def test_high_confidence_at_exact_inclusive_threshold_skips_broker_start_and_select(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.80)
    factory = _BrokerFactory(["success"])
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(_branch_observation()) == [0]
    output = capsys.readouterr().out

    assert factory.instances == []
    assert direct.branch_calls == 1
    assert direct.policy.history_append_count == 1
    assert '"mode": "high_confidence_frozen_direct"' in output
    assert '"mcts_child_started_for_this_decision": false' in output
    assert '"mcts_select_call_count": 0' in output
    assert '"history_only_existing_child_journal_count": 0' in output
    assert '"selected_factorized_stage_probability_threshold": 0.8' in output
    assert '"selected_factorized_stage_probabilities": [0.8]' in output
    assert '"all_selected_factorized_stages_meet_threshold": true' in output
    assert module.DEGRADED_PREFIX not in output


def test_parent_records_loaded_model_cuda_observation_before_direct_only_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The observational receipt exists even when high confidence skips MCTS."""

    module = _load_entrypoint()
    trace: list[str] = []
    direct = _Direct(
        branch_action=(0,), selected_stage_probability=0.80, trace=trace
    )
    factory = _BrokerFactory(["success"], trace=trace)
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)
    expected = {
        "schema": module.CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": module.CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "cpu",
        "telemetry_complete": True,
        "error_types": [],
    }
    capture_models: list[object] = []

    def capture(model: object) -> dict[str, object]:
        capture_models.append(model)
        trace.append("capture_cuda")
        return dict(expected)

    monkeypatch.setattr(module, "capture_cuda_runtime_before_search", capture)

    assert module.agent(_branch_observation()) == [0]
    output = capsys.readouterr().out

    marker_line = next(
        line for line in output.splitlines() if line.startswith(module.DECISION_PREFIX)
    )
    marker = json.loads(marker_line[len(module.DECISION_PREFIX) :])
    assert marker["parent_cuda_runtime_before_search"] == expected
    assert len(capture_models) == 1
    assert trace.index("ensure_runtime") < trace.index("capture_cuda") < trace.index(
        "direct_branch"
    )
    assert factory.instances == []


def test_high_confidence_just_below_threshold_routes_to_bounded_mcts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.799)
    factory = _BrokerFactory(["success"], selected=(1,))
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(_branch_observation()) == [1]
    assert len(factory.instances) == 1
    assert len(factory.instances[0].select_calls) == 1


def test_high_confidence_after_prior_mcts_only_notes_existing_child_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.50)
    factory = _BrokerFactory(["success"], selected=(1,))
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(_branch_observation()) == [1]
    capsys.readouterr()
    direct.selected_stage_probability = 0.80
    assert module.agent(_branch_observation()) == [0]
    output = capsys.readouterr().out

    broker = factory.instances[0]
    assert len(broker.select_calls) == 1
    assert broker.note_calls == [(_branch_observation(), [0])]
    assert '"mode": "high_confidence_frozen_direct"' in output
    assert '"mcts_child_started": false' in output
    assert '"mcts_child_call_count": 0' in output
    assert '"history_only_existing_child_journal_count": 1' in output


def test_missing_direct_policy_target_hard_fails_before_broker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    direct = _Direct(branch_action=(0,), emit_target=False)
    factory = _BrokerFactory(["success"])
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    with pytest.raises(RuntimeError, match="validated_direct_policy_receipt_missing"):
        module.agent(_branch_observation())

    assert factory.instances == []
    assert module.HARD_FAILURE_PREFIX in capsys.readouterr().out


def test_two_continuation_prompts_reuse_backed_plan_without_second_select(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_entrypoint()
    observation = _branch_observation()
    fingerprint = canonical_observation_fingerprint(observation)
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.50)
    factory = _BrokerFactory(
        ["success"],
        selected=(1,),
        receipt_overrides={
            "root_seat": 0,
            "principal_variation": [
                {"observation_fingerprint": fingerprint, "action": [1]},
                {"observation_fingerprint": fingerprint, "action": [1]},
            ],
        },
    )
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(observation) == [1]
    capsys.readouterr()
    assert module.agent(observation) == [1]
    second_output = capsys.readouterr().out
    assert module.agent(observation) == [1]
    third_output = capsys.readouterr().out

    broker = factory.instances[0]
    assert len(broker.select_calls) == 1
    assert broker.note_calls == [(observation, [1]), (observation, [1])]
    assert direct.policy.history_append_count == 3
    assert direct.policy._previous_action_token == ("mcts-token", [[1]])
    assert '"mode": "deterministic_mcts_continuation"' in second_output
    assert '"mode": "deterministic_mcts_continuation"' in third_output
    assert '"continuation_observation_fingerprint"' in second_output
    assert '"continuation_actor_seat": 0' in second_output
    assert '"continuation_both_lanes_same_fingerprint": true' in second_output
    assert '"continuation_backed_leader_agreement": true' in second_output
    assert '"planned_vs_direct_action_changed": true' in second_output
    assert '"history_rewritten_to_actual_action": true' in second_output
    assert '"history_only_existing_child_journal_count": 1' in second_output


@pytest.mark.parametrize(
    "planned_action,planned_fingerprint",
    [([1], "sha256:not-the-current-observation"), ([9], None)],
    ids=["fingerprint_mismatch", "illegal_planned_action"],
)
def test_invalid_continuation_clears_entire_plan_and_returns_to_mcts(
    monkeypatch: pytest.MonkeyPatch,
    planned_action: list[int],
    planned_fingerprint: str | None,
) -> None:
    module = _load_entrypoint()
    observation = _branch_observation()
    fingerprint = canonical_observation_fingerprint(observation)
    direct = _Direct(branch_action=(0,), selected_stage_probability=0.50)
    factory = _BrokerFactory(
        ["success"],
        selected=(1,),
        receipt_overrides={
            "root_seat": 0,
            "principal_variation": [
                {
                    "observation_fingerprint": planned_fingerprint or fingerprint,
                    "action": planned_action,
                }
            ],
        },
    )
    _install_common_mocks(monkeypatch, module, direct=direct, factory=factory)

    assert module.agent(observation) == [1]
    # The first plan is deliberately invalid at the next prompt.  A later
    # normal MCTS receipt carries no replacement plan, so we can observe that
    # the stale one was cleared rather than consumed.
    factory.instances[0].receipt_overrides = {"principal_variation": []}
    assert module.agent(observation) == [1]
    assert len(factory.instances[0].select_calls) == 2
    assert module._GAME_PRINCIPAL_VARIATION == []


@pytest.mark.parametrize("problem", ["missing", "tampered"])
def test_parent_rejects_staged_dso_before_loading_direct_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    problem: str,
) -> None:
    module = _load_entrypoint()
    direct_calls = 0

    def no_direct_load() -> _Direct:
        nonlocal direct_calls
        direct_calls += 1
        return _Direct()

    def reject_stage(_stage: Path) -> dict[str, Any]:
        raise R228GameplayError(f"{problem} staged cg/libcg.so")

    monkeypatch.setattr(module, "_direct", no_direct_load)
    import poke_bot.r228_kaggle_async_runtime as runtime

    monkeypatch.setattr(runtime, "validate_staged_stock_library_identity", reject_stage)

    with pytest.raises(RuntimeError, match="parent_stock_library_identity_invalid"):
        module.agent(_branch_observation())

    assert direct_calls == 0
    output = capsys.readouterr().out
    assert module.HARD_FAILURE_PREFIX in output
    assert problem in output
