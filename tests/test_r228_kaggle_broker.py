"""CPU-only containment tests for the r228 Kaggle subprocess broker."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from poke_bot import r228_kaggle_broker as broker_module
from poke_bot import r228_kaggle_async_runtime as runtime_module


def _observation() -> dict[str, object]:
    """A small select wire payload; complete legality is stubbed below."""

    return {
        "current": {"yourIndex": 0},
        "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
    }


def _two_lane_receipt(action: list[int]) -> dict[str, object]:
    """A minimal parent-visible r238/r240 authoritative receipt."""

    return {
        "selected_action": list(action),
        "mode": "shared_tree_mcts",
        "fake": True,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_release_calls": 2,
        "search_end_calls": 2,
        "per_lane_depth": [1, 1],
        # The ordinary two-lane path returns one backed result per lane.
        "completed_backups": 2,
        # Stock SearchId allocation is handle-local: both fresh raw handles
        # can legitimately return SearchId 0.  The composite is distinct.
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
    }


def _clean_zero_receipt(action: list[int], *, valid: bool = True) -> dict[str, object]:
    """Typed child proof for a clean zero-backup deadline."""

    receipt = _two_lane_receipt(action)
    receipt.update(
        {
            "mode": "clean_deadline_zero_backup_frozen_model_fallback",
            "mcts_action_authority": False,
            "stop_reason": "decision_deadline",
            "completed_backups": 0,
            "per_lane_depth": [0, 0],
            "search_step_calls": 0,
            "microbatch_sizes": [],
            "max_simulator_calls_in_flight": 0,
            "clean_deadline_cleanup_complete": True,
        }
    )
    if not valid:
        receipt["clean_deadline_cleanup_complete"] = False
    return receipt


def _terminal_win_receipt(
    action: list[int], *, proof_overrides: dict[str, object] | None = None
) -> dict[str, object]:
    observation = _observation()
    legal = ((0,), (1,))
    root_fingerprint = broker_module._canonical_observation_fingerprint(observation)
    legal_fingerprint = broker_module._legal_order_fingerprint(legal)
    receipt = _two_lane_receipt(action)
    proof: dict[str, object] = {
        "proof_kind": broker_module.PROVEN_TERMINAL_WIN_PROOF_KIND,
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
    receipt.update(
        {
            "stop_reason": broker_module.PROVEN_TERMINAL_WIN_STOP_REASON,
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
            # Literal child-runtime R246 facts, emitted before the broker
            # adds its own process/IPC observations.
            "elapsed_seconds": 0.001,
            "child_search_elapsed_seconds": 0.001,
            "completed_root_backup_count": 1,
            "terminal_win_proof_count": 1,
            "proven_deterministic_terminal_win_this_turn_stop_count": 1,
            "two_lane_topology_initialized_before_terminal_win_override": True,
            "terminal_win_proof_backed_up_into_shared_root_tree": True,
            "terminal_leaf_returned_by_exact_stock_simulator": True,
            "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
        }
    )
    return receipt


def _preload_stock_library_receipt() -> dict[str, object]:
    """Minimal child pre-load identity carried in the ready handshake."""

    return {
        "member": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
    }


def _cpu_cuda_runtime_observation() -> dict[str, object]:
    """One exact ready-handshake observation for a CPU-only scripted child."""

    return {
        "schema": broker_module.CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": broker_module.CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "cpu",
        "telemetry_complete": True,
        "error_types": [],
    }


class _MockParameter:
    def __init__(self, device: str) -> None:
        self.device = device


class _MockModel:
    def __init__(self, device: str) -> None:
        self._parameter = _MockParameter(device)

    def parameters(self):
        yield self._parameter


class _MockCuda:
    def __init__(
        self,
        *,
        available: bool,
        initialized: bool,
        devices: list[tuple[str, int, int]],
    ) -> None:
        self.available = bool(available)
        self.initialized = bool(initialized)
        self.devices = list(devices)

    def is_available(self) -> bool:
        return self.available

    def is_initialized(self) -> bool:
        return self.initialized

    def device_count(self) -> int:
        return len(self.devices)

    def get_device_name(self, index: int) -> str:
        return self.devices[index][0]

    def mem_get_info(self, index: int) -> tuple[int, int]:
        _name, free, total = self.devices[index]
        return free, total


class _MockTorch:
    def __init__(self, cuda: _MockCuda) -> None:
        self.cuda = cuda


class _ScriptedChild:
    """Tiny socket child standing in for exactly one ``Popen`` invocation."""

    _PID = 70_000

    def __init__(self, argv: list[str], *, mode: str, **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        self.mode = mode
        self.pid = self._PID
        type(self)._PID += 1
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        fd = int(self.kwargs["pass_fds"][0])
        # The broker closes its inherited descriptor after Popen returns.  A
        # duplicate is how a real child retains the endpoint after exec.
        self._socket = socket.socket(fileno=os.dup(fd))
        self._socket.settimeout(0.02)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll(self) -> int | None:
        with self._lock:
            return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return int(self.returncode or 0)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._finish(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self._finish(-9)

    def close(self) -> None:
        self._finish(-15)
        self._thread.join(timeout=0.10)

    def _finish(self, code: int) -> None:
        with self._lock:
            if self.returncode is None:
                self.returncode = int(code)
        self._stopped.set()
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._socket.close()
        except OSError:
            pass

    def _send(self, payload: object) -> None:
        if isinstance(payload, bytes):
            encoded = payload
        else:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._socket.sendall(encoded + b"\n")

    def _messages(self):
        buffer = bytearray()
        while not self._stopped.is_set():
            try:
                chunk = self._socket.recv(64 * 1024)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer.extend(chunk)
            while b"\n" in buffer:
                line, _newline, remainder = buffer.partition(b"\n")
                buffer[:] = remainder
                if line:
                    yield json.loads(line.decode("utf-8"))

    def _run(self) -> None:
        try:
            self._send(
                {
                    "schema": broker_module.SCHEMA,
                    "type": "ready",
                    "preload_stock_library": _preload_stock_library_receipt(),
                    "cuda_runtime_before_search": _cpu_cuda_runtime_observation(),
                }
            )
            for request in self._messages():
                kind = request.get("type")
                if kind == "close":
                    self._send(
                        {
                            "schema": broker_module.SCHEMA,
                            "type": "closed",
                            "request_id": request.get("request_id"),
                        }
                    )
                    return
                if kind != "select":
                    continue
                request_id = request.get("request_id")
                if self.mode == "success":
                    self._send(
                        {
                            "schema": broker_module.SCHEMA,
                            "type": "result",
                            "request_id": request_id,
                            "action": [1],
                            "receipt": _two_lane_receipt([1]),
                        }
                    )
                elif self.mode == "progress_hang":
                    self._send(
                        {
                            "schema": broker_module.SCHEMA,
                            "type": "progress",
                            "request_id": request_id,
                            "payload": {"lane_id": 1, "phase": "native_step"},
                        }
                    )
                    self._stopped.wait()
                    return
                elif self.mode == "crash":
                    self._finish(23)
                    return
                elif self.mode == "malformed":
                    self._send(b"not-json")
                    self._stopped.wait()
                    return
                elif self.mode == "illegal":
                    self._send(
                        {
                            "schema": broker_module.SCHEMA,
                            "type": "result",
                            "request_id": request_id,
                            "action": [9],
                            "receipt": _two_lane_receipt([9]),
                        }
                    )
                elif self.mode in {"clean_zero", "malformed_clean_zero"}:
                    self._send(
                        {
                            "schema": broker_module.SCHEMA,
                            "type": "result",
                            "request_id": request_id,
                            "action": [0],
                            "receipt": _clean_zero_receipt(
                                [0], valid=self.mode == "clean_zero"
                            ),
                        }
                    )
                else:  # pragma: no cover - test authoring guard
                    raise AssertionError(f"unsupported child mode {self.mode!r}")
        except OSError:
            pass
        finally:
            self._finish(self.returncode if self.returncode is not None else 0)


class _PopenFactory:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.children: list[_ScriptedChild] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> _ScriptedChild:
        child = _ScriptedChild(argv, mode=self.mode, **kwargs)
        self.children.append(child)
        return child

    def close(self) -> None:
        for child in self.children:
            child.close()


class _NonReapingChild:
    """A fake whose exact Popen child ignores both bounded reaping attempts."""

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        self.pid = 80_001
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []
        fd = int(self.kwargs["pass_fds"][0])
        self._socket = socket.socket(fileno=os.dup(fd))
        self._socket.sendall(
            (
                json.dumps(
                    {
                        "schema": broker_module.SCHEMA,
                        "type": "ready",
                        "preload_stock_library": _preload_stock_library_receipt(),
                    }
                )
                + "\n"
            ).encode("utf-8")
        )

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(self.argv, timeout)

    def close_peer(self) -> None:
        self.returncode = -9
        self._socket.close()


def _broker(tmp_path: Path, **kwargs: float) -> broker_module.IsolatedR228SearchBroker:
    stage = tmp_path / "stage"
    stage.mkdir()
    return broker_module.IsolatedR228SearchBroker(stage, **kwargs)


def test_capture_cuda_runtime_before_search_records_cpu_without_environment_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only unit coverage: the probe is observational and JSON-safe."""

    torch = _MockTorch(
        _MockCuda(available=False, initialized=False, devices=[])
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    observed = broker_module.capture_cuda_runtime_before_search(_MockModel("cpu"))

    assert observed == _cpu_cuda_runtime_observation()
    serialized = json.dumps(observed, sort_keys=True)
    assert "CUDA_VISIBLE_DEVICES" not in serialized
    assert "uuid" not in serialized.lower()
    assert "serial" not in serialized.lower()


def test_capture_cuda_runtime_before_search_records_mock_hidden_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible CUDA device is recorded, never selected or otherwise used."""

    torch = _MockTorch(
        _MockCuda(
            available=True,
            initialized=True,
            devices=[("Mock Kaggle GPU", 12 * 1024**3, 16 * 1024**3)],
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    observed = broker_module.capture_cuda_runtime_before_search(_MockModel("cuda:0"))

    assert observed == {
        "schema": broker_module.CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": broker_module.CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": True,
        "cuda_initialized": True,
        "device_count": 1,
        "devices": [
            {
                "device_index": 0,
                "device_name": "Mock Kaggle GPU",
                "total_memory_bytes": 16 * 1024**3,
                "free_memory_bytes": 12 * 1024**3,
            }
        ],
        "model_device": "cuda:0",
        "telemetry_complete": True,
        "error_types": [],
    }


def test_capture_cuda_runtime_before_search_does_not_initialize_cuda_for_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anomalous uninitialized CUDA runtime stays diagnostic-only."""

    torch = _MockTorch(
        _MockCuda(
            available=True,
            initialized=False,
            devices=[("Mock Kaggle GPU", 12 * 1024**3, 16 * 1024**3)],
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    observed = broker_module.capture_cuda_runtime_before_search(_MockModel("cpu"))

    assert observed["cuda_available"] is True
    assert observed["cuda_initialized"] is False
    assert observed["devices"] == []
    assert observed["telemetry_complete"] is False
    assert observed["error_types"] == ["cuda_status:AvailableButNotInitialized"]


def test_broker_binds_terminal_win_proof_to_exact_current_request() -> None:
    observation = _observation()
    legal = ((0,), (1,))
    receipt = _terminal_win_receipt([1])

    validated = broker_module._validate_two_lane_receipt(
        receipt,
        observation=observation,
        legal_order=legal,
        selected_action=[1],
    )
    assert validated["terminal_win_proof"]["selected_action"] == [1]
    assert validated["completed_root_backup_count"] == 1
    assert validated["child_search_elapsed_seconds"] == 0.001

    for overrides in (
        {"root_observation_fingerprint": "sha256:stale"},
        {"path_no_chance_boundary": False},
        {"path_actor_seats": [1]},
        {"terminal_result": "draw"},
    ):
        with pytest.raises(broker_module.R228BrokerError):
            broker_module._validate_two_lane_receipt(
                _terminal_win_receipt([1], proof_overrides=overrides),
                observation=observation,
                legal_order=legal,
                selected_action=[1],
            )


def test_broker_rejects_terminal_receipt_missing_literal_child_fact() -> None:
    observation = _observation()
    legal = ((0,), (1,))
    receipt = _terminal_win_receipt([1])
    receipt.pop("all_owned_lane_resources_reservations_and_child_cleanup_complete")

    with pytest.raises(
        broker_module.R228BrokerError,
        match="all_owned_lane_resources_reservations_and_child_cleanup_complete",
    ):
        broker_module._validate_two_lane_receipt(
            receipt,
            observation=observation,
            legal_order=legal,
            selected_action=[1],
        )


@pytest.fixture
def legal_actions(monkeypatch: pytest.MonkeyPatch) -> set[tuple[int, ...]]:
    legal = {(0,), (1,)}
    monkeypatch.setattr(
        broker_module, "_complete_legal_order", lambda _obs: tuple(sorted(legal))
    )
    # Keep the r240 child budget explicit while fake-child action deadlines
    # stay short.  The old eight-second environment key is intentionally gone.
    monkeypatch.setenv("POKEBOT_R238_SEARCH_SECONDS", "0.25")
    return legal


def test_success_uses_socket_child_and_returns_matching_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
) -> None:
    factory = _PopenFactory("success")
    monkeypatch.setattr(broker_module.subprocess, "Popen", factory)
    broker = _broker(
        tmp_path,
        action_timeout_seconds=0.35,
        startup_timeout_seconds=0.20,
        reap_grace_seconds=0.01,
    )
    try:
        selected, receipt, fault = broker.select(_observation(), [0])

        assert selected == [1]
        assert receipt is not None
        assert receipt["selected_action"] == [1]
        assert receipt["mode"] == "shared_tree_mcts"
        assert receipt["fake"] is True
        assert receipt["configured_simulator_lane_count"] == 2
        # These originate in `IsolatedR228SearchBroker.select` after exactly
        # one live child IPC request/reply, not in the parent marker emitter.
        assert receipt["broker_started"] is True
        assert receipt["mcts_child_started"] is True
        assert receipt["mcts_child_started_for_this_decision"] is True
        assert receipt["mcts_child_called"] is True
        assert receipt["mcts_child_call_count"] == 1
        assert receipt["mcts_select_call_count"] == 1
        assert receipt["child_search_budget_seconds"] == broker.search_seconds
        assert receipt["child_preload_stock_library"] == _preload_stock_library_receipt()
        assert broker.marker_payload()["child_identity"][
            "cuda_runtime_before_search"
        ] == _cpu_cuda_runtime_observation()
        assert fault is None
        assert broker.decision_count == 1
        assert not broker.disabled
        assert len(factory.children) == 1
        child = factory.children[0]
        assert child.argv[:4] == [
            sys.executable,
            "-u",
            "-m",
            "poke_bot.r228_kaggle_broker",
        ]
        child_fd_index = child.argv.index("--child-fd") + 1
        assert int(child.argv[child_fd_index]) == child.kwargs["pass_fds"][0]
        assert child.kwargs["cwd"] == str((tmp_path / "stage").resolve())
    finally:
        broker.close()
        factory.close()


def test_progress_hang_reaps_exact_child_then_disables_remaining_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
) -> None:
    factory = _PopenFactory("progress_hang")
    monkeypatch.setattr(broker_module.subprocess, "Popen", factory)
    broker = _broker(
        tmp_path,
        action_timeout_seconds=0.30,
        startup_timeout_seconds=0.20,
        reap_grace_seconds=0.01,
    )
    try:
        selected, receipt, fault = broker.select(_observation(), [0])

        assert selected == [0]
        assert receipt is None
        assert fault is not None
        assert fault["code"] == "response_timeout"
        assert fault["direct_fallback_action"] == [0]
        assert fault["progress_by_lane"]["1"]["phase"] == "native_step"
        assert broker.disabled and broker.degraded
        child = factory.children[0]
        assert child.terminate_calls == 1
        assert child.kill_calls == 0
        # The parent may shorten TERM's wait slightly to preserve the outer
        # action deadline, but it may never stretch beyond the reap grace.
        assert len(child.wait_timeouts) == 1
        term_wait = child.wait_timeouts[0]
        assert term_wait is not None
        assert 0.0 < term_wait <= 0.01

        # Once degraded, no second native child may be launched for this game.
        remainder, remainder_receipt, remainder_fault = broker.select(_observation(), [1])
        assert remainder == [1]
        assert remainder_receipt is None
        assert remainder_fault is not None
        # The causal timeout receipt is retained rather than replaced with a
        # synthetic second failure.
        assert remainder_fault["code"] == "response_timeout"
        assert len(factory.children) == 1
    finally:
        broker.close()
        factory.close()


def test_clean_zero_backup_deadline_returns_only_parent_direct_after_reap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
) -> None:
    factory = _PopenFactory("clean_zero")
    monkeypatch.setattr(broker_module.subprocess, "Popen", factory)
    broker = _broker(
        tmp_path,
        action_timeout_seconds=0.35,
        startup_timeout_seconds=0.20,
        reap_grace_seconds=0.01,
    )
    try:
        selected, receipt, fault = broker.select(_observation(), [0])

        assert selected == [0]
        assert receipt is not None
        assert receipt["mode"] == "zero_backup_precomputed_direct_fallback"
        assert receipt["child_mode"] == "clean_deadline_zero_backup_frozen_model_fallback"
        assert receipt["selected_action"] == [0]
        assert receipt["mcts_action_authority"] is False
        assert receipt["zero_backup_precomputed_direct_fallback"] is True
        assert receipt["exact_child_cleanup_and_reap"]["reap"]["reaped"] is True
        assert fault is None
        assert not broker.degraded
        assert not broker.disabled
        assert not broker.has_live_child
        assert factory.children[0].poll() is not None
    finally:
        broker.close()
        factory.close()


def test_malformed_clean_zero_receipt_is_contained_as_a_child_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
) -> None:
    factory = _PopenFactory("malformed_clean_zero")
    monkeypatch.setattr(broker_module.subprocess, "Popen", factory)
    broker = _broker(
        tmp_path,
        action_timeout_seconds=0.35,
        startup_timeout_seconds=0.20,
        reap_grace_seconds=0.01,
    )
    try:
        selected, receipt, fault = broker.select(_observation(), [0])

        assert selected == [0]
        assert receipt is None
        assert fault is not None
        assert fault["code"] == "clean_zero_receipt_invalid"
        assert broker.degraded and broker.disabled
    finally:
        broker.close()
        factory.close()


@pytest.mark.parametrize(
    ("mode", "expected_codes"),
    [
        ("crash", {"child_exited", "child_socket_closed"}),
        ("malformed", {"malformed_child_json"}),
        ("illegal", {"illegal_child_action"}),
    ],
)
def test_child_faults_fall_back_to_precomputed_direct_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
    mode: str,
    expected_codes: set[str],
) -> None:
    factory = _PopenFactory(mode)
    monkeypatch.setattr(broker_module.subprocess, "Popen", factory)
    broker = _broker(
        tmp_path,
        action_timeout_seconds=0.35,
        startup_timeout_seconds=0.20,
        reap_grace_seconds=0.01,
    )
    try:
        selected, receipt, fault = broker.select(_observation(), [0])

        assert selected == [0]
        assert receipt is None
        assert fault is not None
        assert fault["code"] in expected_codes
        assert fault["direct_fallback_action"] == [0]
        assert broker.disabled
        # A crashing child has already exited.  A malformed reply leaves the
        # child alive and must be reaped; the scripted illegal reply may have
        # already exited by the time the parent validates its action.
        if mode == "crash":
            assert factory.children[0].poll() == 23
        elif mode == "malformed":
            assert factory.children[0].terminate_calls == 1
        else:
            assert factory.children[0].terminate_calls in (0, 1)
    finally:
        broker.close()
        factory.close()


def test_invalid_supplied_direct_action_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
) -> None:
    # Invalid parent input cannot be returned as a supposed safe fallback.
    broker = _broker(tmp_path, action_timeout_seconds=0.05)
    with pytest.raises(broker_module.R228BrokerError, match="direct action"):
        broker.select(_observation(), [99])


def test_tampered_child_stock_library_fails_before_direct_runtime_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The child DSO identity fence precedes frozen direct model loading."""

    stage = tmp_path / "tampered-stage"
    cg_dir = stage / "cg"
    cg_dir.mkdir(parents=True)
    # Populate every known member with deliberately wrong bytes so this test
    # is host-independent while the public helper chooses the current host's
    # one resolved member.
    for member in runtime_module.STOCK_LIBRARY_SHA256:
        (cg_dir / member).write_bytes(b"tampered-r236-stock-library")

    class _Direct:
        ensure_calls = 0

        def _ensure_runtime(self) -> tuple[object, object, object]:
            self.ensure_calls += 1
            raise AssertionError("tampered stage must fail before direct runtime load")

    direct = _Direct()
    monkeypatch.setattr(broker_module, "_child_load_direct", lambda _stage: direct)

    with pytest.raises(runtime_module.R228GameplayError, match="size mismatch"):
        broker_module._child_new_runtime(stage, object())

    assert direct.ensure_calls == 0


def test_child_select_passes_the_parent_precomputed_direct_action_to_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The child must not recompute clean-zero fallback authority locally."""

    class _Runtime:
        child_preload_stock_library_receipt = _preload_stock_library_receipt()
        child_cuda_runtime_before_search = _cpu_cuda_runtime_observation()

        def __init__(self) -> None:
            self.decision_receipts: list[dict[str, object]] = []
            self.calls: list[tuple[dict[str, object], list[int] | None]] = []
            self.closed = False

        def select(
            self,
            observation: dict[str, object],
            *,
            precomputed_direct_action: list[int] | None = None,
        ) -> list[int]:
            self.calls.append((dict(observation), precomputed_direct_action))
            self.decision_receipts.append({"selected_action": [0]})
            return [0]

        def close(self) -> None:
            self.closed = True

    runtime = _Runtime()
    monkeypatch.setattr(broker_module, "_child_new_runtime", lambda _stage, _channel: runtime)

    parent_sock, child_sock = socket.socketpair()
    child_fd = child_sock.detach()
    exit_codes: list[int] = []
    child_thread = threading.Thread(
        target=lambda: exit_codes.append(
            broker_module._child_main(child_fd=child_fd, stage=tmp_path)
        ),
        daemon=True,
    )
    child_thread.start()
    channel = broker_module._JsonSocket(parent_sock, nonblocking=False)
    try:
        ready = channel.recv_blocking_child()
        assert ready is not None and ready["type"] == "ready"
        assert ready["preload_stock_library"] == _preload_stock_library_receipt()
        assert ready["cuda_runtime_before_search"] == _cpu_cuda_runtime_observation()

        channel.send(
            {
                "schema": broker_module.SCHEMA,
                "type": "select",
                "request_id": 1,
                "observation": _observation(),
                "direct_action": [0],
                "timeout_seconds": 0.10,
            },
            deadline=time.monotonic() + 1.0,
        )
        progress = channel.recv_blocking_child()
        result = channel.recv_blocking_child()
        assert progress is not None and progress["type"] == "progress"
        assert result is not None and result["type"] == "result"
        assert result["action"] == [0]
        assert runtime.calls == [(_observation(), [0])]

        channel.send(
            {"schema": broker_module.SCHEMA, "type": "close", "request_id": 2},
            deadline=time.monotonic() + 1.0,
        )
        closed = channel.recv_blocking_child()
        assert closed is not None and closed["type"] == "closed"
    finally:
        channel.close()
        child_thread.join(timeout=1.0)

    assert not child_thread.is_alive()
    assert exit_codes == [0]
    assert runtime.closed


def test_unreapable_exact_child_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_actions: set[tuple[int, ...]],
) -> None:
    child_box: list[_NonReapingChild] = []

    def popen(argv: list[str], **kwargs: Any) -> _NonReapingChild:
        child = _NonReapingChild(argv, **kwargs)
        child_box.append(child)
        return child

    monkeypatch.setattr(broker_module.subprocess, "Popen", popen)
    broker = _broker(
        tmp_path,
        action_timeout_seconds=0.30,
        startup_timeout_seconds=0.10,
        reap_grace_seconds=0.005,
    )
    try:
        with pytest.raises(broker_module.R228BrokerError) as exc_info:
            broker.select(_observation(), [0])
        assert exc_info.value.code == "child_unreaped"
        child = child_box[0]
        assert child.terminate_calls == 1
        assert child.kill_calls == 1
        assert len(child.wait_timeouts) == 2
    finally:
        # This is only a duplicated in-process socket, never a real process.
        for child in child_box:
            child.close_peer()
        broker.close()
