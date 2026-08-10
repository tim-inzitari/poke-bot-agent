from __future__ import annotations

import hashlib
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from poke_bot.recursive_turn_planner.bo1000_evaluation import (
    R207_CANONICAL_PLANNER_CONFIG_SHA256,
    R207_CONTRACT_SHA256,
    build_bo1000_schedule,
)
from poke_bot.recursive_turn_planner.bo1000_pair_runner import (
    HostLocalBO1000PairController,
    build_bo1000_pair_envelope,
)
from poke_bot.recursive_turn_planner.bo1000_remote_pair_protocol import (
    FROZEN_PAIR_RUNNER_SOURCE_SHA256,
    R207_DECK_CARDS_SHA256,
    R207_FROZEN_BUNDLE_SHA256,
    R207_FROZEN_CHECKPOINT_SHA256,
    R207BoundedHostLocalSubprocessGameRunner,
    R207HostCapabilityBinding,
    R207HostHelloPreflight,
    R207RemoteExecutionLimits,
    R207RemotePairProtocolError,
    R207RemotePairRequest,
    R207RemotePairStore,
    R207RemotePairWorkerBoundary,
)

ROOT = Path(__file__).resolve().parents[1]
HOST = "r207-test-host"


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _binding(*, host: str = HOST) -> R207HostCapabilityBinding:
    return R207HostCapabilityBinding(
        host_id=host,
        source_tree_sha256=_digest("r207-content-addressed-source"),
        evaluation_contract_sha256=R207_CONTRACT_SHA256,
        pair_runner_source_sha256=FROZEN_PAIR_RUNNER_SOURCE_SHA256,
        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
        deck_cards_sha256=R207_DECK_CARDS_SHA256,
        planner_config_sha256=R207_CANONICAL_PLANNER_CONFIG_SHA256,
        host_capability_receipt_sha256=_digest(f"{host}:capability"),
        safe_noninterference_receipt_sha256=_digest(f"{host}:noninterference"),
    )


def _limits() -> R207RemoteExecutionLimits:
    return R207RemoteExecutionLimits(
        max_game_mcts_turns=200,
        child_process_grace_seconds=30,
    )


def _request(*, pair_index: int = 7, host: str = HOST) -> R207RemotePairRequest:
    schedule = build_bo1000_schedule(_digest("r207-bo1000-schedule"))
    envelope = build_bo1000_pair_envelope(
        schedule,
        pair_index=pair_index,
        checkpoint_sha256=R207_FROZEN_CHECKPOINT_SHA256,
        bundle_sha256=R207_FROZEN_BUNDLE_SHA256,
        pair_rng_snapshot_sha256=_digest(f"pair-rng:{pair_index}"),
        deck_order_rng_sha256=_digest(f"deck-rng:{pair_index}"),
        execution_host=host,
    )
    hello = R207HostHelloPreflight(
        hello_nonce_sha256=_digest(f"{host}:hello"),
        host_binding=_binding(host=host),
        execution_limits=_limits(),
    )
    return R207RemotePairRequest(
        dispatch_nonce_sha256=_digest(f"dispatch:{pair_index}"),
        pair_envelope=envelope,
        host_preflight=hello,
    )


_CHILD_RECEIPT = f'''
import json
import os
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
marker = Path(sys.argv[2])
marker.parent.mkdir(parents=True, exist_ok=True)
with marker.open("a", encoding="utf-8") as stream:
    stream.write(f"{{os.getpid()}}:{{request['game']['game_index']}}\\n")
game = request["game"]
nonce = game["game_nonce_sha256"]
turn = {{
    "game_nonce_sha256": nonce,
    "pair_id": game["pair_id"],
    "mcts_seat": game["mcts_seat"],
    "planner_turn_id": f"material-turn-{{game['game_index']}}",
    "turn_key": [0, game["game_index"]],
    "actions_dispatched": 1,
    "simulator_transitions_seen": 2,
    "result_or_leaf_evaluations_seen": 2,
    "simulator_leaf_evaluations_seen": 1,
    "neural_leaf_evaluations_seen": 1,
    "unique_tree_nodes_seen": 3,
    "decision_nodes_expanded": 1,
    "terminal_exact_results_seen": 1,
    "boundary_leaf_results_seen": 1,
    "finite_chance_outcomes_evaluated": 0,
    "frozen_policy_prior_batches": 1,
    "frozen_policy_prior_evaluations": 1,
    "batched_frozen_outcome_value_leaf_reranking_batches": 1,
    "frozen_outcome_leaf_evaluations": 1,
    "frozen_value_leaf_evaluations": 1,
    "nonterminal_leaves_reranked": True,
    "terminal_exact_results_not_reranked": True,
    "cache_hits": 0,
    "deterministic_subtree_reuses": 0,
    "tree_rebuilds": 1,
    "turn_planner_wall_seconds": 1.0,
    "max_single_action_planner_wall_seconds": 0.5,
    "requested_tree_fully_expanded_and_backed_up_within_budget": True,
    "tree_incomplete_reason": None,
    "deadline_hit": False,
    "direct_fallback_used": False,
    "selected_action_legal": True,
    "selected_action_sha256": nonce,
    "legal_actions_sha256": nonce,
    "tree_sha256": nonce,
    "config_sha256": "{R207_CANONICAL_PLANNER_CONFIG_SHA256}",
}}
receipt = {{
    "game_nonce_sha256": nonce,
    "pair_id": game["pair_id"],
    "game_index": game["game_index"],
    "mcts_seat": game["mcts_seat"],
    "no_rtp_seat": game["no_rtp_seat"],
    "pair_rng_snapshot_sha256": request["rng"]["pair_rng_snapshot_sha256"],
    "deck_order_rng_sha256": request["rng"]["deck_order_rng_sha256"],
    "checkpoint_sha256": request["frozen_model"]["checkpoint_sha256"],
    "bundle_sha256": request["frozen_model"]["bundle_sha256"],
    "terminal_status": "completed",
    "winner_seat": game["mcts_seat"],
    "illegal_action_count": 0,
    "forfeit_count": 0,
    "crash_count": 0,
    "timeout_count": 0,
    "mcts_turns": [turn],
}}
print(json.dumps(receipt, sort_keys=True))
'''


def _runner(request: R207RemotePairRequest, marker: Path):
    def command(request_path: Path, _spec) -> list[str]:
        return [sys.executable, "-c", _CHILD_RECEIPT, str(request_path), str(marker)]

    return R207BoundedHostLocalSubprocessGameRunner(
        command_builder=command,
        execution_limits=request.host_preflight.execution_limits,
    )


def test_hello_binds_exact_r207_source_model_deck_config_and_timeout() -> None:
    request = _request()
    binding = request.host_binding
    limits = request.host_preflight.execution_limits

    assert binding.pair_runner_source_sha256 == (
        "sha256:50c064756befeb741c22861b1745ba3f3c84e33b9b00977e0003f54d339e5ae1"
    )
    assert hashlib.sha256(
        (ROOT / "poke_bot/recursive_turn_planner/bo1000_pair_runner.py").read_bytes()
    ).hexdigest() == binding.pair_runner_source_sha256.removeprefix("sha256:")
    assert binding.checkpoint_sha256 == R207_FROZEN_CHECKPOINT_SHA256
    assert binding.evaluation_contract_sha256 == R207_CONTRACT_SHA256
    assert binding.bundle_sha256 == R207_FROZEN_BUNDLE_SHA256
    assert binding.deck_cards_sha256 == R207_DECK_CARDS_SHA256
    assert binding.planner_config_sha256 == R207_CANONICAL_PLANNER_CONFIG_SHA256
    assert limits.hard_child_timeout_seconds == 200 * 20.0 + 30
    capabilities = request.host_preflight.as_payload()["capabilities"]
    assert capabilities["host_local_search_and_inference"] is True
    assert capabilities["hard_child_timeout_enforced"] is True
    assert capabilities["material_mcts_telemetry_required_for_completed_games"] is True
    assert capabilities["per_turn_lan_rpc_authorized"] is False
    assert capabilities["per_leaf_lan_rpc_authorized"] is False
    assert capabilities["coordinator_local_fallback_authorized"] is False

    payload = request.as_payload()
    assert R207RemotePairRequest.from_payload(payload) == request
    payload["execution_contract"]["per_leaf_lan_rpc_authorized"] = True
    with pytest.raises(R207RemotePairProtocolError, match="contract"):
        R207RemotePairRequest.from_payload(payload)


def test_exact_contract_identities_are_not_caller_substitutable() -> None:
    with pytest.raises(R207RemotePairProtocolError, match="frozen r195 checkpoint"):
        replace(_binding(), checkpoint_sha256=_digest("other-checkpoint"))
    with pytest.raises(R207RemotePairProtocolError, match="exact r207 deck"):
        replace(_binding(), deck_cards_sha256=_digest("other-deck"))
    with pytest.raises(R207RemotePairProtocolError, match="canonical r207 planner"):
        replace(_binding(), planner_config_sha256=_digest("other-config"))
    with pytest.raises(R207RemotePairProtocolError, match="20s/5s"):
        replace(_limits(), max_planner_wall_seconds_per_actual_turn=19.0)


def test_submit_lookup_and_nonce_are_create_once_and_durable(tmp_path: Path) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)

    assert store.lookup(request, observed_at_unix_ns=0).state == "unsubmitted"
    request_path = store.submit(request)
    assert store.submit(request) == request_path
    assert store.lookup(request, observed_at_unix_ns=0).state == "pending"
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o444
    assert (
        stat.S_IMODE(store.preflight_path(request.host_preflight).stat().st_mode)
        == 0o444
    )

    changed_nonce = replace(request, dispatch_nonce_sha256=_digest("changed-dispatch"))
    with pytest.raises(R207RemotePairProtocolError, match="differs from create-once"):
        store.submit(changed_nonce)


def test_expired_unstarted_lease_requires_exact_ack_and_never_crosses_owner(
    tmp_path: Path,
) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)
    store.submit(request)
    first = store.acquire_lease(
        request,
        owner_id="worker-a",
        lease_nonce_sha256=_digest("lease-a"),
        issued_at_unix_ns=100,
        lease_duration_ns=100,
    )

    assert store.lookup(request, observed_at_unix_ns=150).state == "leased"
    with pytest.raises(R207RemotePairProtocolError, match="still active"):
        store.acquire_lease(
            request,
            owner_id="worker-b",
            lease_nonce_sha256=_digest("lease-b"),
            issued_at_unix_ns=150,
            lease_duration_ns=100,
        )
    assert store.lookup(request, observed_at_unix_ns=200).state == "lease_expired"
    with pytest.raises(R207RemotePairProtocolError, match="exact lease identity"):
        store.acquire_lease(
            request,
            owner_id="worker-b",
            lease_nonce_sha256=_digest("lease-b"),
            issued_at_unix_ns=200,
            lease_duration_ns=100,
        )
    second = store.acquire_lease(
        request,
        owner_id="worker-b",
        lease_nonce_sha256=_digest("lease-b"),
        issued_at_unix_ns=200,
        lease_duration_ns=100,
        expected_expired_lease_id_sha256=first.lease_id_sha256,
    )
    assert second.previous_lease_id_sha256 == first.lease_id_sha256
    assert second.lease_sequence == 1
    assert store.lookup(request, observed_at_unix_ns=201).lease_owner_id == "worker-b"

    controller = HostLocalBO1000PairController(tmp_path / "pairs", execution_host=HOST)
    worker = R207RemotePairWorkerBoundary(store, controller)
    with pytest.raises(R207RemotePairProtocolError, match="latest"):
        worker.execute_lease(
            request,
            first,
            _runner(request, tmp_path / "old-owner.txt"),
            observed_at_unix_ns=201,
        )


class _WorkerLost(BaseException):
    pass


class _LostAfterStartRunner:
    def __init__(self, limits_sha256: str):
        self.execution_limits_sha256 = limits_sha256

    def run_game(self, **_kwargs):
        raise _WorkerLost


def test_started_execution_expiry_is_unresolved_not_blindly_released(
    tmp_path: Path,
) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)
    store.submit(request)
    lease = store.acquire_lease(
        request,
        owner_id="worker-a",
        lease_nonce_sha256=_digest("lease-a"),
        issued_at_unix_ns=100,
        lease_duration_ns=100,
    )
    controller = HostLocalBO1000PairController(tmp_path / "pairs", execution_host=HOST)
    worker = R207RemotePairWorkerBoundary(store, controller)
    lost = _LostAfterStartRunner(request.host_preflight.execution_limits.limits_sha256)

    with pytest.raises(_WorkerLost):
        worker.execute_lease(request, lease, lost, observed_at_unix_ns=101)
    status = store.lookup(request, observed_at_unix_ns=200)
    assert status.state == "execution_lease_expired_unresolved"
    assert status.execution_start_sha256 is not None
    with pytest.raises(R207RemotePairProtocolError, match="execution has started"):
        store.acquire_lease(
            request,
            owner_id="worker-b",
            lease_nonce_sha256=_digest("lease-b"),
            issued_at_unix_ns=200,
            lease_duration_ns=100,
            expected_expired_lease_id_sha256=lease.lease_id_sha256,
        )


def test_worker_runs_two_fresh_local_children_and_retrieves_idempotently(
    tmp_path: Path,
) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)
    controller = HostLocalBO1000PairController(tmp_path / "pairs", execution_host=HOST)
    worker = R207RemotePairWorkerBoundary(store, controller)
    marker = tmp_path / "child-pids.txt"
    runner = _runner(request, marker)
    store.submit(request)
    lease = store.acquire_lease(
        request,
        owner_id="worker-a",
        lease_nonce_sha256=_digest("lease-a"),
        issued_at_unix_ns=100,
        lease_duration_ns=10_000,
    )

    retrieved = worker.execute_lease(request, lease, runner, observed_at_unix_ns=101)
    assert retrieved.pair_result.status.status == "complete"
    assert len(retrieved.pair_result.game_receipts) == 2
    assert all(receipt.mcts_turns for receipt in retrieved.pair_result.game_receipts)
    assert store.lookup(request, observed_at_unix_ns=102).state == "complete"
    assert stat.S_IMODE(store.result_path(request).stat().st_mode) == 0o444
    children = marker.read_text(encoding="utf-8").splitlines()
    assert len(children) == 2
    assert len({line.split(":", 1)[0] for line in children}) == 2

    retry = worker.execute_lease(request, lease, runner, observed_at_unix_ns=20_000)
    assert retry.result_receipt == retrieved.result_receipt
    assert marker.read_text(encoding="utf-8").splitlines() == children


class _EmptyTelemetryRunner:
    def __init__(self, limits_sha256: str):
        self.execution_limits_sha256 = limits_sha256

    def run_game(self, *, request, **_kwargs):
        game = request["game"]
        return {
            "game_nonce_sha256": game["game_nonce_sha256"],
            "pair_id": game["pair_id"],
            "game_index": game["game_index"],
            "mcts_seat": game["mcts_seat"],
            "no_rtp_seat": game["no_rtp_seat"],
            "pair_rng_snapshot_sha256": request["rng"]["pair_rng_snapshot_sha256"],
            "deck_order_rng_sha256": request["rng"]["deck_order_rng_sha256"],
            "checkpoint_sha256": request["frozen_model"]["checkpoint_sha256"],
            "bundle_sha256": request["frozen_model"]["bundle_sha256"],
            "terminal_status": "completed",
            "winner_seat": game["mcts_seat"],
            "illegal_action_count": 0,
            "forfeit_count": 0,
            "crash_count": 0,
            "timeout_count": 0,
            "mcts_turns": [],
        }


def test_completed_pair_without_material_mcts_telemetry_cannot_publish(
    tmp_path: Path,
) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)
    controller = HostLocalBO1000PairController(tmp_path / "pairs", execution_host=HOST)
    worker = R207RemotePairWorkerBoundary(store, controller)
    store.submit(request)
    lease = store.acquire_lease(
        request,
        owner_id="worker-a",
        lease_nonce_sha256=_digest("lease-a"),
        issued_at_unix_ns=100,
        lease_duration_ns=10_000,
    )
    runner = _EmptyTelemetryRunner(
        request.host_preflight.execution_limits.limits_sha256
    )

    with pytest.raises(R207RemotePairProtocolError, match="material"):
        worker.execute_lease(request, lease, runner, observed_at_unix_ns=101)
    assert store.lookup(request, observed_at_unix_ns=102).state == "executing"
    with pytest.raises(R207RemotePairProtocolError, match="incomplete"):
        store.retrieve(request, controller)


def test_retrieval_fails_closed_if_terminal_evidence_becomes_mutable(
    tmp_path: Path,
) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)
    controller = HostLocalBO1000PairController(tmp_path / "pairs", execution_host=HOST)
    worker = R207RemotePairWorkerBoundary(store, controller)
    store.submit(request)
    lease = store.acquire_lease(
        request,
        owner_id="worker-a",
        lease_nonce_sha256=_digest("lease-a"),
        issued_at_unix_ns=100,
        lease_duration_ns=10_000,
    )
    worker.execute_lease(
        request, lease, _runner(request, tmp_path / "pids.txt"), observed_at_unix_ns=101
    )

    store.result_path(request).chmod(0o644)
    with pytest.raises(R207RemotePairProtocolError, match="mutable"):
        store.retrieve(request, controller)


def test_bounded_runner_passes_exact_hard_timeout_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request()
    runner = R207BoundedHostLocalSubprocessGameRunner(
        command_builder=lambda _path, _spec: [sys.executable, "game.py"],
        execution_limits=request.host_preflight.execution_limits,
    )
    spec = request.pair_envelope.game_specs[0]
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run_game(
        request_path=tmp_path / "request.json",
        request={},
        envelope=request.pair_envelope,
        spec=spec,
    )
    assert result["terminal_status"] == "failed_closed"
    assert result["timeout_count"] == 1
    assert result["crash_count"] == 0
    assert observed["timeout"] == (
        request.host_preflight.execution_limits.hard_child_timeout_seconds
    )
    env = observed["env"]
    assert env["BO1000_REMOTE_DISPATCH_AUTHORIZED"] == "0"
    assert env["BO1000_HARD_CHILD_TIMEOUT_SECONDS"] == "4030.0"
    assert env["BO1000_GUIDE2VEC_ENABLED"] == "0"
    assert env["BO1000_GUIDE_LOGIT_TRANSFORM_ENABLED"] == "0"
    assert env["BO1000_GUIDE_LINEAR_TRANSFORM_ENABLED"] == "0"
    assert env["BO1000_BASE_POLICY_TRANSFORM"] == "frozen_r195_identity"


def test_hard_child_timeouts_seal_a_terminal_pair_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request()
    store = R207RemotePairStore(tmp_path / "protocol", execution_host=HOST)
    controller = HostLocalBO1000PairController(tmp_path / "pairs", execution_host=HOST)
    worker = R207RemotePairWorkerBoundary(store, controller)
    store.submit(request)
    lease = store.acquire_lease(
        request,
        owner_id="worker-a",
        lease_nonce_sha256=_digest("lease-a"),
        issued_at_unix_ns=100,
        lease_duration_ns=10_000,
    )
    runner = R207BoundedHostLocalSubprocessGameRunner(
        command_builder=lambda _path, _spec: [sys.executable, "game.py"],
        execution_limits=request.host_preflight.execution_limits,
    )
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    retrieved = worker.execute_lease(request, lease, runner, observed_at_unix_ns=101)

    assert calls == 2
    assert retrieved.pair_result.status.status == "failed_closed"
    assert [
        receipt.timeout_count for receipt in retrieved.pair_result.game_receipts
    ] == [
        1,
        1,
    ]
    assert all(
        receipt.crash_count == 0 for receipt in retrieved.pair_result.game_receipts
    )
    assert store.lookup(request, observed_at_unix_ns=102).state == "failed_closed"
