"""CPU-only contracts for the r228 Kaggle async gameplay wrapper."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from poke_bot import r228_kaggle_async_runtime as runtime
from poke_bot.r228_async_shared_tree_queue import (
    AsyncDecisionReceipt,
    AsyncDirectFallbackReceipt,
    AsyncDirectFallbackRequired,
)


class _LeafPacket:
    """Small stand-in which preserves every inference packet field for assertions."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


@dataclass(frozen=True)
class _ModelLeaf:
    priors: Sequence[float]
    value: float = 0.0


class _Policy:
    def __init__(self) -> None:
        self.board_history: list[object] = []
        self.previous_action_history: list[object | None] = []
        self._previous_action_token: object | None = None
        self.factorized_calls = 0

    def _append_decision_history(self, _obs: dict[str, Any]) -> None:
        self.board_history.append(("board", len(self.board_history)))
        self.previous_action_history.append(None)

    @staticmethod
    def _matchup_model_route() -> int:
        return 3

    @staticmethod
    def _history_context_limit() -> int:
        return 4

    def _factorized_greedy_prepared(self, *args: Any, **kwargs: Any) -> list[int]:
        self.factorized_calls += 1
        raise AssertionError("clean-deadline fallback must reuse the root direct action")


def _bare_gameplay(*, policy: _Policy) -> runtime.R228AsyncGameplay:
    """Construct only the pure wrapper state; native arena construction is excluded."""

    gameplay = object.__new__(runtime.R228AsyncGameplay)
    gameplay.model = object()
    gameplay.policy = policy
    gameplay.deck = tuple(range(60))
    gameplay._decision = None
    gameplay.cleanup_timeout_seconds = 0.05
    gameplay.stock_library_receipt = {"member": "cg/libcg.so"}
    gameplay.decision_count = 0
    gameplay.decision_receipts = []
    return gameplay


def _install_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enumerate_actions: Callable[[dict[str, Any], int], Sequence[Sequence[int]]],
    forward: Callable[[Any, Sequence[_LeafPacket]], Sequence[_ModelLeaf]],
) -> tuple[list[int], list[_LeafPacket], list[tuple[dict[str, Any], list[list[int]]]]]:
    """Patch only r228's method-local imports with deterministic CPU fakes."""

    caps: list[int] = []
    packets: list[_LeafPacket] = []
    option_token_calls: list[tuple[dict[str, Any], list[list[int]]]] = []
    fake_features = types.ModuleType("poke_bot.features")

    def enumerate_action_combos(raw: dict[str, Any], *, max_combos: int):
        caps.append(max_combos)
        return enumerate_actions(raw, max_combos)

    fake_features.enumerate_action_combos = enumerate_action_combos
    fake_features.build_board_tokens = lambda raw, deck: (
        int(raw["current"]["yourIndex"]),
        tuple(deck),
    )

    def build_option_tokens(raw: dict[str, Any], selected: list[list[int]]) -> str:
        option_token_calls.append((raw, selected))
        return "selected-option-token"

    fake_features.build_option_tokens = build_option_tokens

    fake_cg_env = types.ModuleType("poke_bot.cg_env")
    fake_cg_env.is_finished = lambda _raw: False
    fake_cg_env.result_winner = lambda _raw: None
    fake_cg_env.build_search_inputs = lambda _obs, _deck, **_kwargs: {
        "opponent_deck": [61] * 60,
    }

    fake_batched_infer = types.ModuleType("poke_bot.batched_infer")
    fake_batched_infer.LeafPacket = _LeafPacket

    def forward_leaf_batch(model: Any, rows: Sequence[_LeafPacket]):
        assert model is not None
        packets.extend(rows)
        return list(forward(model, rows))

    fake_batched_infer.forward_leaf_batch = forward_leaf_batch

    package = importlib.import_module("poke_bot")
    for name, module in {
        "poke_bot.features": fake_features,
        "poke_bot.cg_env": fake_cg_env,
        "poke_bot.batched_infer": fake_batched_infer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(package, "features", fake_features, raising=False)
    monkeypatch.setattr(package, "cg_env", fake_cg_env, raising=False)
    monkeypatch.setattr(package, "batched_infer", fake_batched_infer, raising=False)
    return caps, packets, option_token_calls


def _observation(*, seat: int = 0) -> dict[str, Any]:
    return {
        "current": {"yourIndex": seat},
        "select": {"context": 0, "option": list(range(8))},
    }


def _receipt(action: Sequence[int], prior: float) -> AsyncDecisionReceipt:
    lane_count = runtime.R228_SIMULATOR_LANE_COUNT
    return AsyncDecisionReceipt(
        selected_action=tuple(action),
        selected_action_visits=1,
        selected_action_value=0.25,
        selected_action_prior=float(prior),
        root_visits=lane_count,
        arena_count=lane_count,
        unique_handle_count=lane_count,
        search_begin_calls=lane_count,
        search_step_calls=lane_count,
        completed_backups=lane_count,
        microbatch_sizes=(lane_count,),
        max_simulator_calls_in_flight=lane_count,
        completion_order=tuple(range(lane_count)),
        per_lane_depth=(1,) * lane_count,
        # SearchIds are handle-local in stock libcg: both valid AgentStart
        # handles may begin at id zero.
        per_lane_search_id_chains=((0, 1),) * lane_count,
        per_lane_handle_identities=tuple(100 + lane for lane in range(lane_count)),
        distinct_search_begin_composite_count=lane_count,
        search_release_calls=lane_count,
        search_end_calls=lane_count,
        outstanding_virtual_loss=0,
        stop_reason="decision_deadline",
        minimum_backups_before_stability=(
            runtime.R238_MINIMUM_BACKUPS_BEFORE_STABILITY
        ),
        stable_root_leader_observations=(
            runtime.R238_STABLE_ROOT_LEADER_OBSERVATIONS
        ),
        maximum_backups_per_decision=(
            runtime.R238_MAXIMUM_BACKUPS_PER_DECISION
        ),
        leader_stability_count=1,
        elapsed_seconds=0.001,
        root_seat=0,
        principal_variation=(),
    )


def _clean_zero_receipt() -> AsyncDirectFallbackReceipt:
    lane_count = runtime.R228_SIMULATOR_LANE_COUNT
    return AsyncDirectFallbackReceipt(
        arena_count=lane_count,
        unique_handle_count=lane_count,
        search_begin_calls=lane_count,
        search_step_calls=lane_count,
        completed_backups=0,
        microbatch_sizes=(),
        max_simulator_calls_in_flight=lane_count,
        completion_order=(),
        per_lane_depth=(0,) * lane_count,
        per_lane_search_id_chains=((0,),) * lane_count,
        per_lane_handle_identities=tuple(100 + lane for lane in range(lane_count)),
        distinct_search_begin_composite_count=lane_count,
        search_release_calls=lane_count,
        search_end_calls=lane_count,
        outstanding_virtual_loss=0,
        stop_reason="decision_deadline",
        minimum_backups_before_stability=(
            runtime.R238_MINIMUM_BACKUPS_BEFORE_STABILITY
        ),
        stable_root_leader_observations=(
            runtime.R238_STABLE_ROOT_LEADER_OBSERVATIONS
        ),
        maximum_backups_per_decision=(
            runtime.R238_MAXIMUM_BACKUPS_PER_DECISION
        ),
        leader_stability_count=0,
        elapsed_seconds=0.001,
        root_seat=0,
    )


def _terminal_win_receipt(
    observation: dict[str, Any],
    legal: Sequence[Sequence[int]],
    *,
    action: Sequence[int] = (1,),
    proof_overrides: dict[str, Any] | None = None,
) -> AsyncDecisionReceipt:
    root_fingerprint = runtime.canonical_observation_fingerprint(observation)
    legal_fingerprint = runtime.legal_order_fingerprint(legal)
    proof: dict[str, Any] = {
        "proof_kind": runtime.PROVEN_TERMINAL_WIN_PROOF_KIND,
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
        proof.update(proof_overrides)
    return replace(
        _receipt(action, 0.1),
        root_visits=1,
        completed_backups=1,
        microbatch_sizes=(1,),
        completion_order=(0,),
        per_lane_depth=(1, 0),
        per_lane_search_id_chains=((0, 1), (0,)),
        stop_reason=runtime.PROVEN_TERMINAL_WIN_STOP_REASON,
        root_actor_seat=0,
        root_observation_fingerprint=root_fingerprint,
        root_legal_order_fingerprint=legal_fingerprint,
        terminal_win_proof=proof,
    )


def test_canonical_observation_fingerprint_is_full_json_and_lane_independent() -> None:
    """Continuation identity is the exact full raw JSON, not a lane/tree key."""

    first = {
        "select": {"option": [2, 1], "context": 0},
        "current": {"yourIndex": 0},
        "opaque": {"nested": [True, None, "x"]},
    }
    reordered = {
        "opaque": {"nested": [True, None, "x"]},
        "current": {"yourIndex": 0},
        "select": {"context": 0, "option": [2, 1]},
    }
    canonical = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    expected = "sha256:" + hashlib.sha256(canonical).hexdigest()

    assert runtime.canonical_observation_fingerprint(first) == expected
    assert runtime.canonical_observation_fingerprint(reordered) == expected
    assert runtime.canonical_observation_fingerprint(
        {**first, "new_field": 1}
    ) != expected
    with pytest.raises(runtime.R228GameplayError, match="canonical JSON"):
        runtime.canonical_observation_fingerprint({"not_json": object()})


def test_terminal_leaf_decode_distinguishes_exact_win_from_model_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gameplay = _bare_gameplay(policy=_Policy())
    gameplay._decision = {
        "root_seat": 0,
        "route": 3,
        "opponent_deck": tuple(range(60, 120)),
        "history_boards": [],
        "history_previous_actions": [],
        "boundary_leaf_count": 0,
        "chance_boundary_leaf_count": 0,
        "actor_change_boundary_leaf_count": 0,
    }
    _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: ((0,), (1,)),
        forward=lambda _model, _rows: [],
    )
    cg_env = sys.modules["poke_bot.cg_env"]
    cg_env.is_finished = lambda raw: bool(raw.get("finished"))
    cg_env.result_winner = lambda raw: raw.get("winner")
    won = {**_observation(), "finished": True, "winner": 0}
    lost = {**_observation(), "finished": True, "winner": 1}

    win_leaf, loss_leaf = gameplay._evaluate_batch(
        (
            runtime._Frontier(lane_id=0, raw=won),
            runtime._Frontier(lane_id=1, raw=lost),
        )
    )

    assert win_leaf.terminal_leaf_reached is True
    assert win_leaf.terminal_result == "win"
    assert win_leaf.terminal_winner_seat == 0
    assert win_leaf.value == 1.0
    assert loss_leaf.terminal_leaf_reached is True
    assert loss_leaf.terminal_result == "loss"
    assert loss_leaf.terminal_winner_seat == 1
    assert loss_leaf.value == -1.0


def test_terminal_win_receipt_binds_current_root_and_rejects_uncertain_paths() -> None:
    observation = _observation()
    legal = ((0,), (1,))
    receipt = _terminal_win_receipt(observation, legal)

    runtime._validate_search_receipt_lanes(receipt)
    proof = runtime._validate_terminal_win_proof(
        receipt,
        root_observation_fingerprint=runtime.canonical_observation_fingerprint(
            observation
        ),
        root_legal_order_fingerprint=runtime.legal_order_fingerprint(legal),
        root_actor_seat=0,
        legal_actions=legal,
    )
    assert proof is not None and proof["selected_action"] == [1]

    stale = replace(
        receipt,
        terminal_win_proof={
            **receipt.terminal_win_proof,
            "root_observation_fingerprint": "sha256:stale",
        },
    )
    with pytest.raises(runtime.R228GameplayError, match="stale"):
        runtime._validate_terminal_win_proof(
            stale,
            root_observation_fingerprint=runtime.canonical_observation_fingerprint(
                observation
            ),
            root_legal_order_fingerprint=runtime.legal_order_fingerprint(legal),
            root_actor_seat=0,
            legal_actions=legal,
        )

    for field, value, match in (
        ("path_no_chance_boundary", False, "path_no_chance_boundary"),
        ("path_no_unresolved_randomness", False, "unresolved_randomness"),
        ("path_actor_seats", [1], "root-actor-only"),
        ("terminal_result", "loss", "not a terminal win"),
    ):
        malformed = replace(
            receipt,
            terminal_win_proof={**receipt.terminal_win_proof, field: value},
        )
        with pytest.raises(runtime.R228GameplayError, match=match):
            runtime._validate_terminal_win_proof(
                malformed,
                root_observation_fingerprint=(
                    runtime.canonical_observation_fingerprint(observation)
                ),
                root_legal_order_fingerprint=runtime.legal_order_fingerprint(legal),
                root_actor_seat=0,
                legal_actions=legal,
            )


def test_evaluate_batch_carries_validated_actor_seat_and_leaf_action_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the root actor can expand; an opponent leaf is value-only."""

    policy = _Policy()
    gameplay = _bare_gameplay(policy=policy)
    gameplay._decision = {
        "root_seat": 0,
        "route": 3,
        "opponent_deck": tuple(range(60, 120)),
        "history_boards": [],
        "history_previous_actions": [],
    }
    caps, packets, _ = _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: ((0,), (1,))
        if max_combos == runtime.R228_COMPLETE_ACTION_CAP
        else (),
        forward=lambda _model, rows: [
            _ModelLeaf(
                priors=(1.0,)
                if row.action_combos_override == [[]]
                else (0.25, 0.75),
                value=-0.25 if row.action_combos_override == [[]] else 0.5,
            )
            for row in rows
        ],
    )

    leaves = gameplay._evaluate_batch(
        (
            runtime._Frontier(lane_id=0, raw=_observation(seat=0)),
            runtime._Frontier(lane_id=1, raw=_observation(seat=1)),
        )
    )

    assert runtime.R228_COMPLETE_ACTION_CAP == 65_536
    # The root actor materializes its complete ordered actions.  The opponent
    # boundary uses the deterministic value-only option token instead.
    assert caps == [65_536]
    assert [leaf.actor_seat for leaf in leaves] == [0, 1]
    assert [leaf.observation_fingerprint for leaf in leaves] == [
        runtime.canonical_observation_fingerprint(_observation(seat=0)),
        runtime.canonical_observation_fingerprint(_observation(seat=1)),
    ]
    assert [packet.your_deck for packet in packets] == [
        list(gameplay.deck),
        list(gameplay._decision["opponent_deck"]),
    ]
    # The model receives the opponent's state/acting deck but keeps the root
    # seat, so its value head remains a root-perspective backup value.
    assert [packet.root_seat for packet in packets] == [0, 0]
    assert [packet.action_combos_override for packet in packets] == [
        [[0], [1]],
        [[]],
    ]
    root_leaf, opponent_leaf = leaves
    assert root_leaf.boundary is False
    assert root_leaf.legal_actions == ((0,), (1,))
    assert root_leaf.priors == (0.25, 0.75)
    assert root_leaf.value == 0.5

    # An actor change is the real end-turn/opponent boundary.  It is still
    # model-evaluated for the root value backup, but cannot create a child
    # action or deterministic continuation beyond our own turn.
    assert opponent_leaf.boundary is True
    assert opponent_leaf.legal_actions == ()
    assert opponent_leaf.priors == ()
    # The dedicated model value is retained even though its synthetic one
    # option's prior is discarded.  This proves an opponent boundary remains
    # value/state-head evidence rather than an unevaluated terminal.
    assert opponent_leaf.value == -0.25
    assert gameplay._decision["actor_change_boundary_leaf_count"] == 1
    assert gameplay._decision["chance_boundary_leaf_count"] == 0
    assert gameplay._decision["boundary_leaf_count"] == 1


def test_runtime_receipt_counts_chance_and_actor_change_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary telemetry distinguishes chance from end-turn/opponent leaves."""

    policy = _Policy()
    gameplay = _bare_gameplay(policy=policy)
    _caps, _packets, _tokens = _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: ((0,), (1,))
        if max_combos == runtime.R228_COMPLETE_ACTION_CAP
        else (),
        forward=lambda _model, rows: [
            _ModelLeaf(
                priors=(1.0,)
                if row.action_combos_override == [[]]
                else (0.25, 0.75),
                value=0.5,
            )
            for row in rows
        ],
    )

    actor_change = _observation(seat=1)
    chance = _observation(seat=0)
    chance["select"]["context"] = 46

    class _BoundarySearch:
        def run_decision(self, **kwargs: Any) -> AsyncDecisionReceipt:
            # The queue turns a boundary leaf into a no-edge node, so neither
            # row can extend the receipt-carried deterministic continuation.
            leaves = gameplay._evaluate_batch(
                (
                    runtime._Frontier(lane_id=0, raw=actor_change),
                    runtime._Frontier(lane_id=1, raw=chance),
                )
            )
            assert all(leaf.boundary for leaf in leaves)
            assert all(not leaf.legal_actions and not leaf.priors for leaf in leaves)
            return _receipt(kwargs["root_actions"][-1], kwargs["root_priors"][-1])

    gameplay._search = _BoundarySearch()
    assert gameplay.select(_observation()) == [1]

    payload = gameplay.decision_receipts[-1]
    assert payload["boundary_leaf_count"] == 2
    assert payload["chance_boundary_leaf_count"] == 1
    assert payload["actor_change_boundary_leaf_count"] == 1
    # The only full enumeration was the actual root.  Both value-only
    # boundaries deliberately bypass the complete-action enumerator.
    assert _caps == [65_536]


def test_runtime_exact_terminal_win_receipt_has_absolute_selected_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation()
    legal = ((0,), (1,))
    gameplay = _bare_gameplay(policy=_Policy())
    _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: legal
        if max_combos == runtime.R228_COMPLETE_ACTION_CAP
        else (),
        # The direct root strongly prefers action 0.  The returned exact
        # terminal proof must nevertheless select action 1.
        forward=lambda _model, rows: [
            _ModelLeaf(priors=(0.95, 0.05)) for _row in rows
        ],
    )

    class _TerminalSearch:
        def run_decision(self, **kwargs: Any) -> AsyncDecisionReceipt:
            assert kwargs["root_observation_fingerprint"] == (
                runtime.canonical_observation_fingerprint(observation)
            )
            assert kwargs["root_legal_order_fingerprint"] == (
                runtime.legal_order_fingerprint(legal)
            )
            assert kwargs["root_actor_seat"] == 0
            return _terminal_win_receipt(observation, legal)

    gameplay._search = _TerminalSearch()
    assert gameplay.select(observation) == [1]
    payload = gameplay.decision_receipts[-1]
    assert payload["selected_action"] == [1]
    assert payload["direct_action"] == [0]
    assert payload["action_changed"] is True
    assert payload["stop_reason"] == runtime.PROVEN_TERMINAL_WIN_STOP_REASON
    assert payload["proven_deterministic_terminal_win_this_turn"] is True
    assert payload["terminal_win_proof"]["terminal_result"] == "win"
    # These are emitted by the child only after the real terminal proof,
    # shared-root backup, and owned two-lane cleanup validations complete.
    assert payload["completed_root_backup_count"] == 1
    assert payload["terminal_win_proof_count"] == 1
    assert payload["proven_deterministic_terminal_win_this_turn_stop_count"] == 1
    assert payload["child_search_elapsed_seconds"] == payload["elapsed_seconds"]
    for field in (
        "two_lane_topology_initialized_before_terminal_win_override",
        "terminal_win_proof_backed_up_into_shared_root_tree",
        "terminal_leaf_returned_by_exact_stock_simulator",
        "all_owned_lane_resources_reservations_and_child_cleanup_complete",
    ):
        assert payload[field] is True


def test_terminal_marker_facts_reject_unfinished_lane_cleanup() -> None:
    observation = _observation()
    legal = ((0,), (1,))
    receipt = _terminal_win_receipt(observation, legal)
    proof = runtime._validate_terminal_win_proof(
        receipt,
        root_observation_fingerprint=runtime.canonical_observation_fingerprint(
            observation
        ),
        root_legal_order_fingerprint=runtime.legal_order_fingerprint(legal),
        root_actor_seat=0,
        legal_actions=legal,
    )
    unfinished = replace(receipt, outstanding_virtual_loss=1)

    with pytest.raises(runtime.R228GameplayError, match="resource cleanup"):
        runtime._terminal_win_execution_marker_facts(
            unfinished,
            terminal_win_proof=proof,
        )


def test_root_6720_ordered_actions_are_kept_under_explicit_65536_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete root enumeration must not reject a legal 6,720-action shape."""

    monkeypatch.delenv(runtime.R238_SEARCH_SECONDS_ENV, raising=False)
    actions = tuple(tuple(row) for row in itertools.permutations(range(8), 5))
    assert len(actions) == 6_720
    policy = _Policy()
    gameplay = _bare_gameplay(policy=policy)
    caps, packets, option_token_calls = _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: actions
        if max_combos == 65_536
        else (),
        forward=lambda _model, rows: [
            _ModelLeaf(priors=(0.1,) * (len(actions) - 1) + (0.9,))
            for _ in rows
        ],
    )

    class _Search:
        call: dict[str, Any] | None = None

        def run_decision(self, **kwargs: Any) -> AsyncDecisionReceipt:
            self.call = kwargs
            return _receipt(kwargs["root_actions"][-1], kwargs["root_priors"][-1])

    search = _Search()
    gameplay._search = search
    selected = gameplay.select(_observation())

    assert selected == list(actions[-1])
    assert caps == [65_536]
    assert packets and len(packets[0].action_combos_override) == 6_720
    assert search.call is not None
    assert len(search.call["root_actions"]) == 6_720
    assert len(search.call["search_inputs"]) == runtime.R228_SIMULATOR_LANE_COUNT
    assert gameplay.decision_receipts[-1]["complete_ordered_action_cap"] == 65_536
    assert gameplay.decision_receipts[-1]["requested_simulator_lane_count"] == 2
    assert gameplay.decision_receipts[-1]["active_simulator_lane_count"] == 2
    assert gameplay.decision_receipts[-1]["per_lane_search_id_chains"] == [
        [0, 1],
        [0, 1],
    ]
    assert gameplay.decision_receipts[-1]["per_lane_handle_identities"] == [100, 101]
    assert gameplay.decision_receipts[-1]["per_lane_first_search_ids"] == [0, 0]
    assert gameplay.decision_receipts[-1][
        "handle_scoped_first_search_id_composite_states"
    ] == [
        {"lane_id": 0, "handle_identity": 100, "first_search_id": 0},
        {"lane_id": 1, "handle_identity": 101, "first_search_id": 0},
    ]
    assert (
        gameplay.decision_receipts[-1]["distinct_search_begin_composite_count"]
        == 2
    )
    assert gameplay.decision_receipts[-1]["configured_search_seconds"] == 2.0
    assert gameplay.decision_receipts[-1]["minimum_backups_before_stability"] == 8
    assert gameplay.decision_receipts[-1]["stable_root_leader_observations"] == 3
    assert gameplay.decision_receipts[-1]["stable_root_leader_observations_required"] == 3
    assert gameplay.decision_receipts[-1]["maximum_backups_per_decision"] == 32
    assert gameplay.decision_receipts[-1]["stop_reason"] == "decision_deadline"
    assert gameplay.decision_receipts[-1]["leader_stability_count"] == 1
    assert (
        gameplay.decision_receipts[-1][
            "observed_stable_root_leader_observations"
        ]
        == gameplay.decision_receipts[-1]["leader_stability_count"]
    )
    assert (
        gameplay.decision_receipts[-1]["deterministic_root_leader_observations"]
        == gameplay.decision_receipts[-1]["leader_stability_count"]
    )
    assert gameplay.decision_receipts[-1]["root_seat"] == 0
    assert gameplay.decision_receipts[-1]["principal_variation"] == []
    assert option_token_calls[-1][1] == [list(actions[-1])]


def test_clean_deadline_fallback_reuses_precomputed_direct_root_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-backup fallback must not invoke a second policy/greedy action path."""

    actions = ((3,), (9,))
    policy = _Policy()
    gameplay = _bare_gameplay(policy=policy)
    caps, _packets, _option_token_calls = _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: actions
        if max_combos == 65_536
        else (),
        forward=lambda _model, rows: [
            _ModelLeaf(priors=(0.1, 0.9)) for _ in rows
        ],
    )

    class _DeadlineSearch:
        def run_decision(self, **_kwargs: Any) -> AsyncDecisionReceipt:
            raise AsyncDirectFallbackRequired(
                "clean decision stop completed no backups",
                cleanup_receipt=_clean_zero_receipt(),
            )

    gameplay._search = _DeadlineSearch()
    # The child model prefers [9], but a clean zero-backup outcome has to
    # return the parent-provided, already precomputed legal [3] exactly.
    selected = gameplay.select(
        _observation(), precomputed_direct_action=actions[0]
    )

    assert selected == [3]
    assert caps == [65_536]
    assert policy.factorized_calls == 0
    assert gameplay._decision is None
    payload = gameplay.decision_receipts[-1]
    assert payload["mode"] == "clean_deadline_zero_backup_frozen_model_fallback"
    assert payload["mcts_action_authority"] is False
    assert payload["selected_action"] == [3]
    assert payload["direct_action"] == [3]
    assert payload["completed_backups"] == 0
    assert payload["stop_reason"] == "decision_deadline"
    assert payload["per_lane_search_id_chains"] == [[0], [0]]
    assert payload["per_lane_handle_identities"] == [100, 101]
    assert payload["per_lane_first_search_ids"] == [0, 0]
    assert payload["handle_scoped_first_search_id_composite_states"] == [
        {"lane_id": 0, "handle_identity": 100, "first_search_id": 0},
        {"lane_id": 1, "handle_identity": 101, "first_search_id": 0},
    ]
    assert payload["distinct_search_begin_composite_count"] == 2
    assert payload["outstanding_virtual_loss"] == 0
    assert payload["clean_deadline_cleanup_complete"] is True


def test_precomputed_parent_direct_mismatch_hard_fails_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An IPC action outside the complete root legal order never becomes fallback."""

    actions = ((3,), (9,))
    gameplay = _bare_gameplay(policy=_Policy())
    _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: actions
        if max_combos == 65_536
        else (),
        forward=lambda _model, rows: [_ModelLeaf(priors=(0.1, 0.9)) for _ in rows],
    )

    class _Search:
        calls = 0

        def run_decision(self, **_kwargs: Any) -> AsyncDecisionReceipt:
            self.calls += 1
            raise AssertionError("invalid parent direct must fail before native search")

    search = _Search()
    gameplay._search = search
    with pytest.raises(runtime.R228GameplayError, match="outside the complete legal order"):
        gameplay.select(_observation(), precomputed_direct_action=(42,))
    assert search.calls == 0


def test_precomputed_parent_direct_does_not_override_backed_mcts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied parent direct is fallback-only, never normal MCTS authority."""

    actions = ((3,), (9,))
    gameplay = _bare_gameplay(policy=_Policy())
    _install_runtime_modules(
        monkeypatch,
        enumerate_actions=lambda _raw, max_combos: actions
        if max_combos == 65_536
        else (),
        forward=lambda _model, rows: [_ModelLeaf(priors=(0.1, 0.9)) for _ in rows],
    )

    class _Search:
        def run_decision(self, **kwargs: Any) -> AsyncDecisionReceipt:
            return _receipt(kwargs["root_actions"][-1], kwargs["root_priors"][-1])

    gameplay._search = _Search()
    assert gameplay.select(_observation(), precomputed_direct_action=(3,)) == [9]
    payload = gameplay.decision_receipts[-1]
    assert payload["mode"] == "shared_tree_mcts"
    assert payload["mcts_action_authority"] is True
    assert payload["direct_action"] == [9]


def test_search_receipt_requires_exactly_two_complete_simulator_lanes() -> None:
    """A successful action cannot carry an old eight-lane or partial receipt."""

    receipt = _receipt((1,), 0.9)
    runtime._validate_search_receipt_lanes(receipt)

    with pytest.raises(runtime.R228GameplayError, match="arena_count"):
        runtime._validate_search_receipt_lanes(replace(receipt, arena_count=8))
    with pytest.raises(runtime.R228GameplayError, match="incomplete lane vector"):
        runtime._validate_search_receipt_lanes(
            replace(receipt, per_lane_search_id_chains=((0,),))
        )
    # The same raw id is valid across distinct AgentStart handles.  A repeated
    # handle/id composite is not.
    with pytest.raises(
        runtime.R228GameplayError,
        match="duplicate SearchBegin handle/id composites",
    ):
        runtime._validate_search_receipt_lanes(
            replace(receipt, per_lane_handle_identities=(100, 100))
        )
    with pytest.raises(runtime.R228GameplayError, match="composite SearchBegin count"):
        runtime._validate_search_receipt_lanes(
            replace(receipt, distinct_search_begin_composite_count=1)
        )
    with pytest.raises(runtime.R228GameplayError, match="invalid normal stop reason"):
        runtime._validate_search_receipt_lanes(
            replace(receipt, stop_reason="smoke_min_depth")
        )
    with pytest.raises(runtime.R228GameplayError, match="maximum backups"):
        runtime._validate_search_receipt_lanes(
            replace(receipt, stop_reason="maximum_backups")
        )
    with pytest.raises(runtime.R228GameplayError, match="minimum backup"):
        runtime._validate_search_receipt_lanes(
            replace(receipt, stop_reason="stable_root_leader")
        )


def test_clean_zero_backup_receipt_requires_exact_cleanup_and_composites() -> None:
    """Only a post-cleanup two-lane deadline can authorize parent fallback."""

    receipt = _clean_zero_receipt()
    assert runtime._validate_clean_zero_backup_receipt(receipt) == 2
    with pytest.raises(runtime.R228GameplayError, match="exactly one step"):
        runtime._validate_clean_zero_backup_receipt(
            replace(receipt, search_step_calls=1)
        )
    with pytest.raises(runtime.R228GameplayError, match="duplicate SearchBegin handle/id"):
        runtime._validate_clean_zero_backup_receipt(
            replace(receipt, per_lane_handle_identities=(100, 100))
        )
    with pytest.raises(runtime.R228GameplayError, match="decision deadline"):
        runtime._validate_clean_zero_backup_receipt(
            replace(receipt, stop_reason="tree_exhausted")
        )


def test_principal_variation_requires_root_bound_json_entries() -> None:
    """Only the queue's narrow two-lane continuation schema crosses IPC."""

    receipt = _receipt((1,), 0.9)
    fingerprint = runtime.canonical_observation_fingerprint(_observation())
    valid = replace(
        receipt,
        principal_variation=(
            {"observation_fingerprint": fingerprint, "action": [1, 2]},
        ),
    )
    assert runtime._validate_principal_variation(valid, root_seat=0) == (
        {"observation_fingerprint": fingerprint, "action": [1, 2]},
    )
    with pytest.raises(runtime.R228GameplayError, match="root seat"):
        runtime._validate_principal_variation(
            replace(valid, root_seat=1), root_seat=0
        )
    with pytest.raises(runtime.R228GameplayError, match="invalid schema"):
        runtime._validate_principal_variation(
            replace(valid, principal_variation=({"action": [1]},)), root_seat=0
        )
    with pytest.raises(runtime.R228GameplayError, match="invalid principal variation"):
        runtime._validate_principal_variation(
            replace(valid, principal_variation=valid.principal_variation * 9),
            root_seat=0,
        )


def test_r238_search_budget_uses_only_its_new_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inherited eight-second r228 setting cannot expand the r238 child window."""

    monkeypatch.setenv("POKEBOT_R228_DECISION_SECONDS", "8.0")
    monkeypatch.delenv(runtime.R238_SEARCH_SECONDS_ENV, raising=False)
    assert runtime._r238_search_seconds() == 2.0

    monkeypatch.setenv(runtime.R238_SEARCH_SECONDS_ENV, "1.25")
    assert runtime._r238_search_seconds() == 1.25


def test_r236_native_manifest_is_exact_and_has_no_historical_linux_pin() -> None:
    """Every current runtime member must come from the one r236 native set."""

    assert runtime.SCHEMA == "poke_bot.r238_two_lane_kaggle_viability/v1"
    assert runtime.DECISION_PREFIX == "R238_TWO_LANE_BOUNDED_MCTS_DECISION"
    assert "EIGHT_WORKER" not in runtime.DECISION_PREFIX
    assert runtime.R228_SIMULATOR_LANE_COUNT == 2
    assert runtime.R238_DEFAULT_SEARCH_SECONDS == 2.0
    assert runtime.R238_MINIMUM_BACKUPS_BEFORE_STABILITY == 8
    assert runtime.R238_STABLE_ROOT_LEADER_OBSERVATIONS == 3
    assert runtime.R238_MAXIMUM_BACKUPS_PER_DECISION == 32
    assert runtime.R238_NORMAL_STOP_REASONS == {
        "stable_root_leader",
        "maximum_backups",
        "decision_deadline",
        "tree_exhausted",
        "proven_deterministic_terminal_win_this_turn",
    }
    assert runtime.KAGGLE_ENVIRONMENTS_VERSION == "1.32.6"
    assert runtime.KAGGLE_ENVIRONMENTS_NATIVE_LIBRARY_UPDATE_COMMIT.startswith(
        "03ab2cc"
    )
    assert runtime.STOCK_LIBRARY_SHA256 == {
        "libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "libcg-arm64.so": "1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "libcg.dylib": "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "cg.dll": "eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
    }
    assert runtime.STOCK_LIBRARY_BYTES == {
        "libcg.so": 1_342_400,
        "libcg-arm64.so": 1_296_464,
        "libcg.dylib": 1_245_544,
        "cg.dll": 1_525_248,
    }
    assert "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c" not in (
        runtime.STOCK_LIBRARY_SHA256.values()
    )
    assert runtime._stock_library_member_for_host(
        system="Linux", machine="x86_64"
    ) == "libcg.so"
    assert runtime.REQUIRED_NATIVE_EXPORTS == (
        "GameInitialize",
        "BattleStart",
        "BattleFinish",
        "GetBattleData",
        "Select",
        "VisualizeData",
        "AgentStart",
        "SearchBegin",
        "SearchStep",
        "SearchRelease",
        "SearchEnd",
        "AllCard",
        "AllAttack",
    )
    assert runtime.REQUIRED_R225_API_CALLABLES == (
        "json_to_dataclass",
        "to_observation_class",
    )
    assert runtime.REQUIRED_R225_API_TYPES == ("ApiResult",)


def test_staged_stock_library_requires_resolved_member_size_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrong-size and historical/mismatched hashes fail before native import."""

    stage = tmp_path / "stage"
    library = stage / "cg" / "libcg.so"
    library.parent.mkdir(parents=True)
    good = b"r236-test-member"
    library.write_bytes(good)
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setitem(runtime.STOCK_LIBRARY_SHA256, "libcg.so", digest)
    monkeypatch.setitem(runtime.STOCK_LIBRARY_BYTES, "libcg.so", len(good))

    resolved, observed_digest, observed_size = runtime._validate_staged_stock_library(
        stage, member="libcg.so"
    )
    assert resolved == library.resolve()
    assert observed_digest == digest
    assert observed_size == len(good)

    monkeypatch.setitem(runtime.STOCK_LIBRARY_BYTES, "libcg.so", len(good) + 1)
    with pytest.raises(runtime.R228GameplayError, match="size mismatch"):
        runtime._validate_staged_stock_library(stage, member="libcg.so")

    monkeypatch.setitem(runtime.STOCK_LIBRARY_BYTES, "libcg.so", len(good))
    monkeypatch.setitem(
        runtime.STOCK_LIBRARY_SHA256,
        "libcg.so",
        "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
    )
    with pytest.raises(runtime.R228GameplayError, match="digest mismatch"):
        runtime._validate_staged_stock_library(stage, member="libcg.so")


def test_public_preimport_stock_identity_check_is_host_bound_and_no_native_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent can bind the selected r236 member before loading r195/CG."""

    stage = tmp_path / "stage"
    library = stage / "cg" / "libcg.so"
    library.parent.mkdir(parents=True)
    payload = b"r236-preimport-identity"
    library.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(runtime.STOCK_LIBRARY_SHA256, "libcg.so", digest)
    monkeypatch.setitem(runtime.STOCK_LIBRARY_BYTES, "libcg.so", len(payload))
    monkeypatch.setattr(runtime, "_stock_library_member_for_host", lambda: "libcg.so")

    assert runtime.validate_staged_stock_library_identity(stage) == {
        "path": str(library.resolve()),
        "member": "cg/libcg.so",
        "sha256": f"sha256:{digest}",
        "size_bytes": len(payload),
        "expected_sha256": f"sha256:{digest}",
        "expected_size_bytes": len(payload),
        "kaggle_environments_version": "1.32.6",
        "kaggle_environments_wheel_sha256": (
            "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
        ),
        "native_library_update_commit": "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303",
        "platform": runtime.platform.system().lower(),
        "machine": runtime.platform.machine().lower(),
    }


def test_r236_native_export_surface_fails_closed_when_incomplete() -> None:
    """A loaded DSO is unusable without the frozen search ABI surface."""

    incomplete = types.SimpleNamespace(AgentStart=lambda: None)
    with pytest.raises(runtime.R228GameplayError, match="SearchEnd"):
        runtime._validate_required_native_exports(incomplete)


def test_frozen_r195_api_compatibility_fails_closed_when_overlay_is_incomplete() -> None:
    """The new DSO cannot replace the Python r195 compatibility wrapper."""

    incomplete = types.SimpleNamespace(
        json_to_dataclass=lambda payload, _type: payload,
    )
    with pytest.raises(runtime.R228GameplayError, match="to_observation_class"):
        runtime._validate_r225_api_compatibility(incomplete)


def _install_fake_native_binding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loaded_path: Path,
    on_prewarm: Callable[[], None] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Install a no-native binding so constructor ordering is testable on CPU."""

    calls: list[str] = []
    queue_kwargs: list[dict[str, Any]] = []
    fake_lane = types.ModuleType("poke_bot.r225_stock_native_lane")

    class _Lane:
        pass

    def prewarm_stock_cg() -> tuple[object, object]:
        calls.append("prewarm")
        if on_prewarm is not None:
            on_prewarm()
        lib = types.SimpleNamespace(_name=str(loaded_path))
        for export in runtime.REQUIRED_NATIVE_EXPORTS:
            setattr(lib, export, lambda: None)
        api = types.SimpleNamespace(
            json_to_dataclass=lambda payload, _type: payload,
            to_observation_class=lambda observation: observation,
            ApiResult=object,
        )
        return api, types.SimpleNamespace(lib=lib)

    fake_lane.R225StockNativeSearchLane = _Lane
    fake_lane.prewarm_stock_cg = prewarm_stock_cg
    package = importlib.import_module("poke_bot")
    monkeypatch.setitem(sys.modules, "poke_bot.r225_stock_native_lane", fake_lane)
    monkeypatch.setattr(package, "r225_stock_native_lane", fake_lane, raising=False)

    class _Queue:
        def __init__(self, **kwargs: Any) -> None:
            queue_kwargs.append(dict(kwargs))

        def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "PersistentAsyncSharedTreeMCTS", _Queue)
    monkeypatch.setattr(
        runtime, "_stock_library_member_for_host", lambda: "libcg.so"
    )
    return calls, queue_kwargs


def test_runtime_validates_staged_dso_before_prewarm_and_revalidates_after_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad staged DSO is never prewarmed; a changed DSO is rejected after it."""

    stage = tmp_path / "stage"
    library = stage / "cg" / "libcg.so"
    library.parent.mkdir(parents=True)
    expected = b"r236-good-bytes"
    library.write_bytes(b"r236-bad--bytes")
    monkeypatch.setitem(
        runtime.STOCK_LIBRARY_SHA256, "libcg.so", hashlib.sha256(expected).hexdigest()
    )
    monkeypatch.setitem(runtime.STOCK_LIBRARY_BYTES, "libcg.so", len(expected))
    calls, _queue_kwargs = _install_fake_native_binding(
        monkeypatch, loaded_path=library
    )

    with pytest.raises(runtime.R228GameplayError, match="digest mismatch"):
        runtime.R228AsyncGameplay(
            stage=stage, model=object(), policy=_Policy(), deck=tuple(range(60))
        )
    assert calls == []

    library.write_bytes(expected)
    calls, _queue_kwargs = _install_fake_native_binding(
        monkeypatch,
        loaded_path=library,
        on_prewarm=lambda: library.write_bytes(b"r236-evil-bytes"),
    )
    with pytest.raises(runtime.R228GameplayError, match="digest mismatch"):
        runtime.R228AsyncGameplay(
            stage=stage, model=object(), policy=_Policy(), deck=tuple(range(60))
        )
    assert calls == ["prewarm"]

    library.write_bytes(expected)
    unexpected = stage / "cg" / "other-libcg.so"
    unexpected.write_bytes(expected)
    calls, _queue_kwargs = _install_fake_native_binding(
        monkeypatch, loaded_path=unexpected
    )
    with pytest.raises(runtime.R228GameplayError, match="different member"):
        runtime.R228AsyncGameplay(
            stage=stage, model=object(), policy=_Policy(), deck=tuple(range(60))
        )
    assert calls == ["prewarm"]

    calls, queue_kwargs = _install_fake_native_binding(monkeypatch, loaded_path=library)
    gameplay = runtime.R228AsyncGameplay(
        stage=stage, model=object(), policy=_Policy(), deck=tuple(range(60))
    )
    assert calls == ["prewarm"]
    assert queue_kwargs[-1]["lane_count"] == runtime.R228_SIMULATOR_LANE_COUNT
    assert queue_kwargs[-1]["minimum_backups_before_stability"] == 8
    assert queue_kwargs[-1]["stable_root_leader_observations"] == 3
    assert queue_kwargs[-1]["maximum_backups_per_decision"] == 32
    assert gameplay.stock_library_receipt == {
        "path": str(library.resolve()),
        "member": "cg/libcg.so",
        "sha256": f"sha256:{hashlib.sha256(expected).hexdigest()}",
        "size_bytes": len(expected),
        "expected_sha256": f"sha256:{hashlib.sha256(expected).hexdigest()}",
        "expected_size_bytes": len(expected),
        "kaggle_environments_version": "1.32.6",
        "kaggle_environments_wheel_sha256": (
            "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
        ),
        "native_library_update_commit": "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303",
        "required_native_exports": list(runtime.REQUIRED_NATIVE_EXPORTS),
        "required_r225_api_callables": list(runtime.REQUIRED_R225_API_CALLABLES),
        "required_r225_api_types": list(runtime.REQUIRED_R225_API_TYPES),
        "platform": runtime.platform.system().lower(),
        "machine": runtime.platform.machine().lower(),
    }
