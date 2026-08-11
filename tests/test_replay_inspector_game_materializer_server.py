"""Server-level singleflight tests for physical-game trace materialization."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import replay_inspector.server as inspector_server
from poke_bot import public_matchup_router
from replay_inspector.config import InspectorConfig
from replay_inspector.server import InspectorApplication


def _application(tmp_path: Path) -> InspectorApplication:
    return InspectorApplication(
        InspectorConfig(
            replay_root=tmp_path / "archive",
            rollout_root=tmp_path / "rollouts",
            artifact_roots=(tmp_path / "artifacts",),
            web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
            game_trace_cache_enabled=False,
        )
    )


def _cacheable_identity() -> dict[str, Any]:
    digest = "sha256:" + "a" * 64
    return {
        "schema": "poke_bot.replay_model_inspector.physical_game_materialization/v1",
        "trace_semantics_identity": "replay_inspector.server.trace_payload/v1",
        "submission_id": 77,
        "episode_id": 88,
        "own_seat": 0,
        "replay_sha256": digest,
        "submitted_bundle_sha256": digest,
        "runtime_package_sha256": digest,
        "runtime_source_tree_sha256": digest,
        "checkpoint_sha256": digest,
        "matchup_tree_sha256": None,
        "runtime_parity_receipt_sha256": digest,
        "baseline_trace_request": {
            "head_scales": {},
            "include_setup_model_forward": True,
        },
    }


def _wait_for_game_completion(application: InspectorApplication, key: str) -> None:
    with application._game_materialization_ready:
        job = application._game_materialization_jobs[key]
        while not job.complete:
            application._game_materialization_ready.wait(timeout=5)


def test_same_game_step_requests_join_one_baseline_materialization(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Changing step only waits for the existing game job, never a new forward."""

    application = _application(tmp_path)
    addresses = ((10, 0), (11, 0), (11, 1))
    identity = {"submission_id": 77, "episode_id": 88}
    monkeypatch.setattr(
        application,
        "_prepare_baseline_game_materialization",
        lambda *_args, **_kwargs: ("sha256:fixture-game", identity, addresses),
    )
    monkeypatch.setattr(
        application,
        "_load_materialized_game_trace",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        application,
        "_materialize_baseline_game_backend_traces",
        lambda *_args, **_kwargs: None,
    )

    first_forward_started = threading.Event()
    release_first_forward = threading.Event()
    calls: list[tuple[int, int]] = []
    calls_lock = threading.Lock()

    def raw_trace(
        _submission_id: int,
        _episode_id: int,
        step_index: int,
        factorized_stage: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        with calls_lock:
            calls.append((step_index, factorized_stage))
        if (step_index, factorized_stage) == (10, 0):
            first_forward_started.set()
            assert release_first_forward.wait(timeout=5)
        return {"address": [step_index, factorized_stage]}

    monkeypatch.setattr(application, "_trace_payload_uncached", raw_trace)

    first_result: dict[str, Any] = {}
    second_result: dict[str, Any] = {}

    first = threading.Thread(
        target=lambda: first_result.update(application.trace_payload(77, 88, 10, 0)),
        daemon=True,
    )
    first.start()
    assert first_forward_started.wait(timeout=5)

    second = threading.Thread(
        target=lambda: second_result.update(application.trace_payload(77, 88, 11, 1)),
        daemon=True,
    )
    second.start()
    release_first_forward.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert first_result == {"address": [10, 0]}
    assert second_result == {"address": [11, 1]}

    _wait_for_game_completion(application, "sha256:fixture-game")
    assert sorted(calls) == sorted(addresses)
    assert len(calls) == len(addresses)
    counters = application.health_payload()["game_materialization"]["counters"]
    assert counters["jobs_started"] == 1
    assert counters["backend_game_runs"] == 1
    assert counters["jobs_completed"] == 1
    assert counters["l1_joins"] >= 1


def test_custom_head_scales_bypass_baseline_game_materialization(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A playground counterfactual must not reuse a submitted baseline trace."""

    application = _application(tmp_path)
    prepared_calls = 0
    raw_calls: list[dict[str, Any]] = []

    def prepared(*_args: Any, **_kwargs: Any) -> None:
        nonlocal prepared_calls
        prepared_calls += 1

    def raw_trace(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        raw_calls.append(kwargs)
        return {"counterfactual": True}

    monkeypatch.setattr(application, "_prepare_baseline_game_materialization", prepared)
    monkeypatch.setattr(application, "_trace_payload_uncached", raw_trace)

    assert application.trace_payload(77, 88, 10, 0, head_scales={"value": 1.5}) == {
        "counterfactual": True
    }
    assert prepared_calls == 0
    assert raw_calls == [{"head_scales": {"value": 1.5}, "include_setup_model_forward": True}]


def test_completed_game_manifest_is_reused_after_application_restart(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A complete /tmp game cache, not browser state, serves later steps."""

    addresses = ((10, 0), (11, 0))
    identity = _cacheable_identity()
    with tempfile.TemporaryDirectory(dir="/tmp") as cache_root_text:
        cache_root = Path(cache_root_text)
        config = InspectorConfig(
            replay_root=tmp_path / "archive",
            rollout_root=tmp_path / "rollouts",
            artifact_roots=(tmp_path / "artifacts",),
            web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
            game_trace_cache_root=cache_root,
            game_trace_cache_enabled=True,
            game_trace_cache_max_bytes=4 * 1024 * 1024,
            game_trace_cache_max_game_bytes=2 * 1024 * 1024,
            game_trace_cache_max_entry_bytes=512 * 1024,
            game_trace_cache_min_free_bytes=0,
        )
        first = InspectorApplication(config)
        monkeypatch.setattr(
            first,
            "_prepare_baseline_game_materialization",
            lambda *_args, **_kwargs: ("sha256:manifest-game", identity, addresses),
        )
        monkeypatch.setattr(
            first,
            "_materialize_baseline_game_backend_traces",
            lambda *_args, **_kwargs: None,
        )
        materialized: list[tuple[int, int]] = []

        def raw_trace(
            _submission_id: int,
            _episode_id: int,
            step_index: int,
            factorized_stage: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            materialized.append((step_index, factorized_stage))
            return {
                "availability": {"available": True},
                "model": {"availability": {"available": True}},
                "reproduction_status": "recomputed_not_historical",
                "address": [step_index, factorized_stage],
            }

        monkeypatch.setattr(first, "_trace_payload_uncached", raw_trace)
        assert first.trace_payload(77, 88, 10, 0)["address"] == [10, 0]
        _wait_for_game_completion(first, "sha256:manifest-game")
        assert sorted(materialized) == sorted(addresses)

        second = InspectorApplication(config)
        monkeypatch.setattr(
            second,
            "_prepare_baseline_game_materialization",
            lambda *_args, **_kwargs: ("sha256:manifest-game", identity, addresses),
        )
        monkeypatch.setattr(
            second,
            "_trace_payload_uncached",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("complete physical-game cache must avoid a forward")
            ),
        )
        assert second.trace_payload(77, 88, 11, 0)["address"] == [11, 0]
        counters = second.health_payload()["game_materialization"]["counters"]
        assert counters["l2_hits"] == 1
        assert counters["jobs_started"] == 0
        assert counters["backend_game_runs"] == 0


def test_isolated_game_worker_streams_raw_rows_without_cache_or_deadline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A foreign runtime uses one owned stream worker even with L2 disabled."""

    application = _application(tmp_path)
    calls: list[dict[str, Any]] = []
    streamed = {
        (10, 0): {"availability": {"available": True}, "raw": "first"},
        (11, 1): {"availability": {"available": True}, "raw": "second"},
    }

    class FakeInput:
        def __init__(self) -> None:
            self.payload = b""
            self.closed = False

        def write(self, value: bytes) -> int:
            self.payload += value
            return len(value)

        def close(self) -> None:
            self.closed = True

    class FakeOutput:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakeChild:
        def __init__(self) -> None:
            self.stdin = FakeInput()
            self.stdout = FakeOutput()

    def fake_popen(*args: Any, **kwargs: Any) -> FakeChild:
        calls.append(kwargs)
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is inspector_server.subprocess.PIPE
        assert kwargs["stdout"] is inspector_server.subprocess.PIPE
        assert kwargs["stderr"] is inspector_server.subprocess.DEVNULL
        return FakeChild()

    read_calls: list[tuple[tuple[int, int], ...]] = []

    def read_stream(_child: Any, *, addresses: tuple[tuple[int, int], ...]):
        read_calls.append(addresses)
        return streamed

    cleared: list[bool] = []
    application._model_cache = SimpleNamespace(clear=lambda: cleared.append(True))
    monkeypatch.setattr(inspector_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(application, "_read_isolated_game_worker_stream", read_stream)
    contexts = (
        SimpleNamespace(
            submission=SimpleNamespace(submission_id=77),
            replay_entry=SimpleNamespace(episode_id=88),
            stage=SimpleNamespace(step_index=10, factorized_stage=0),
        ),
        SimpleNamespace(
            submission=SimpleNamespace(submission_id=77),
            replay_entry=SimpleNamespace(episode_id=88),
            stage=SimpleNamespace(step_index=11, factorized_stage=1),
        ),
    )
    runtime_package = tmp_path / "submitted-runtime" / "poke_bot"

    assert application._inspect_exact_game_isolated(
        contexts,
        runtime_root=runtime_package,
        allow_setup_prompt_model_forward=True,
    ) == streamed
    assert read_calls == [((10, 0), (11, 1))]
    assert cleared == [True]
    assert len(calls) == 1
    assert calls[0]["env"]["PYTHONPATH"].split(":")[0] == str(
        runtime_package.parent
    )


def test_streamed_backend_rows_finish_game_when_temporary_cache_is_disabled(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """L2 retention failure never causes a second per-address reconstruction."""

    application = _application(tmp_path)
    addresses = ((10, 0), (11, 0))
    identity = {"submission_id": 77, "episode_id": 88}
    monkeypatch.setattr(
        application,
        "_prepare_baseline_game_materialization",
        lambda *_args, **_kwargs: ("sha256:raw-worker-game", identity, addresses),
    )
    raw_backend = {
        address: {"raw": [address[0], address[1]]} for address in addresses
    }
    monkeypatch.setattr(
        application,
        "_materialize_baseline_game_backend_traces",
        lambda *_args, **_kwargs: dict(raw_backend),
    )
    assembled: list[tuple[int, int]] = []

    def assemble(
        _submission_id: int,
        _episode_id: int,
        step_index: int,
        factorized_stage: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert kwargs["precomputed_trace"] == {
            "raw": [step_index, factorized_stage]
        }
        assert kwargs["prevalidated_game"] is True
        assembled.append((step_index, factorized_stage))
        return {"address": [step_index, factorized_stage]}

    monkeypatch.setattr(application, "_trace_payload_uncached", assemble)
    assert application.trace_payload(77, 88, 10, 0) == {"address": [10, 0]}
    _wait_for_game_completion(application, "sha256:raw-worker-game")
    assert sorted(assembled) == sorted(addresses)


def test_game_batch_matches_individual_setup_and_ordinary_activation_semantics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Hypothetical IsFirst bypasses while ordinary rows retain activation."""

    application = _application(tmp_path)
    digest = "sha256:" + "a" * 64
    tree_path = tmp_path / "matchup-tree.json"
    tree_path.write_text("{}", encoding="utf-8")
    tree = SimpleNamespace(
        available=True,
        resolved_path=tree_path,
        expected_sha256=digest,
    )
    provenance = SimpleNamespace(
        matchup_tree=tree,
        checkpoint=SimpleNamespace(expected_sha256=digest, resolved_path=tmp_path / "m.pt"),
        runtime_parity_receipt=SimpleNamespace(runtime_source_tree_sha256=digest),
        to_dict=dict,
    )
    submission = SimpleNamespace(provenance=provenance)

    class FakeDecisionTree:
        @staticmethod
        def from_path(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(digest=digest)

    class FakeRuntimeRouter:
        def __init__(self, _tree: object) -> None:
            pass

    monkeypatch.setattr(
        public_matchup_router, "PublicMatchupDecisionTree", FakeDecisionTree
    )
    monkeypatch.setattr(
        public_matchup_router, "RuntimePublicMatchupRouter", FakeRuntimeRouter
    )
    monkeypatch.setattr(inspector_server, "sha256_file", lambda _path: digest)
    monkeypatch.setattr(application, "_load_exact_model", lambda *_args: (object(), None))

    calls: list[dict[str, Any]] = []

    def fake_inspect_game(**kwargs: Any) -> dict[tuple[int, int], dict[str, Any]]:
        calls.append(kwargs)
        return {
            address: {
                "availability": {"available": True},
                "policy": {},
                "value": {},
                "adapter": {},
                "heads": [],
                "warnings": [],
            }
            for address in kwargs["addresses"]
        }

    monkeypatch.setattr(
        application,
        "_inference_backend",
        lambda: SimpleNamespace(inspect_replay_game=fake_inspect_game),
    )

    def context(step_index: int, *, model_forward_expected: bool) -> SimpleNamespace:
        return SimpleNamespace(
            submission=submission,
            replay_entry=SimpleNamespace(episode_id=88),
            replay={},
            replay_sha256=digest,
            seat=0,
            stage=SimpleNamespace(
                step_index=step_index,
                factorized_stage=0,
                model_forward_expected=model_forward_expected,
                candidates=((0,), (1,)),
                target_index=1,
                recorded_action=(1,),
            ),
        )

    output = application._inspect_exact_game_locked(
        (context(0, model_forward_expected=False), context(2, model_forward_expected=True)),
        allow_setup_prompt_model_forward=True,
    )

    assert output is not None
    assert list(output) == [(0, 0), (2, 0)]
    assert [call["addresses"] for call in calls] == [[(0, 0)], [(2, 0)]]
    assert calls[0]["submitted_runtime_activation"] is None
    assert calls[0]["allow_setup_prompt_model_forward"] is True
    activation = calls[1]["submitted_runtime_activation"]
    assert activation["basis"] == "checksum_bound_submitted_startup"
    assert calls[1]["allow_setup_prompt_model_forward"] is False


def test_isolated_game_ndjson_protocol_returns_raw_rows(
    tmp_path: Path,
) -> None:
    """The parent consumes frames incrementally instead of a cache manifest."""

    application = _application(tmp_path)
    addresses = ((10, 0), (11, 1))
    records = (
        {
            "protocol": inspector_server.GAME_MATERIALIZATION_WORKER_PROTOCOL,
            "kind": "start",
            "address_count": 2,
        },
        {
            "protocol": inspector_server.GAME_MATERIALIZATION_WORKER_PROTOCOL,
            "kind": "heartbeat",
        },
        {
            "protocol": inspector_server.GAME_MATERIALIZATION_WORKER_PROTOCOL,
            "kind": "trace",
            "step_index": 10,
            "factorized_stage": 0,
            "payload": {"raw": "first"},
        },
        {
            "protocol": inspector_server.GAME_MATERIALIZATION_WORKER_PROTOCOL,
            "kind": "trace",
            "step_index": 11,
            "factorized_stage": 1,
            "payload": {"raw": "second"},
        },
        {
            "protocol": inspector_server.GAME_MATERIALIZATION_WORKER_PROTOCOL,
            "kind": "complete",
            "address_count": 2,
        },
    )
    with tempfile.TemporaryFile(mode="w+b") as stream:
        for record in records:
            stream.write(json.dumps(record).encode("utf-8") + b"\n")
        stream.seek(0)

        class CompletedChild:
            stdout = stream

            def wait(self, *, timeout: float) -> int:
                assert timeout <= 1
                return 0

            def poll(self) -> int:
                return 0

        assert application._read_isolated_game_worker_stream(
            CompletedChild(), addresses=addresses
        ) == {
            (10, 0): {"raw": "first"},
            (11, 1): {"raw": "second"},
        }


def test_isolated_game_worker_stall_watchdog_only_stops_its_owned_child(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """No total deadline exists, but a silent owned child is contained."""

    application = _application(tmp_path)
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    stopped: list[str] = []

    class SilentChild:
        stdout = stream

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            stopped.append("term")

        def wait(self, *, timeout: float) -> int:
            stopped.append("wait")
            return 0

        def kill(self) -> None:
            stopped.append("kill")

    monkeypatch.setattr(inspector_server, "_ISOLATED_GAME_WORKER_POLL_SECONDS", 0.01)
    monkeypatch.setattr(inspector_server, "_ISOLATED_GAME_WORKER_STALL_SECONDS", 0.0)
    try:
        assert application._read_isolated_game_worker_stream(
            SilentChild(), addresses=((10, 0),)
        ) is None
        assert stopped[:2] == ["term", "wait"]
    finally:
        os.close(write_fd)
        stream.close()


def test_isolated_game_worker_failure_is_a_game_result_not_trace_fanout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A malformed/stalled worker leaves each row unavailable without retries."""

    application = _application(tmp_path)

    class FakeInput:
        closed = False

        def write(self, value: bytes) -> int:
            return len(value)

        def close(self) -> None:
            self.closed = True

    class FakeOutput:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakeChild:
        stdin = FakeInput()
        stdout = FakeOutput()

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(inspector_server.subprocess, "Popen", lambda *_a, **_k: FakeChild())
    monkeypatch.setattr(application, "_read_isolated_game_worker_stream", lambda *_a, **_k: None)
    contexts = tuple(
        SimpleNamespace(
            submission=SimpleNamespace(submission_id=77),
            replay_entry=SimpleNamespace(episode_id=88),
            stage=SimpleNamespace(step_index=step, factorized_stage=0),
        )
        for step in (10, 11)
    )
    failed_rows = application._inspect_exact_game_isolated(
        contexts,
        runtime_root=tmp_path / "foreign-runtime",
        allow_setup_prompt_model_forward=True,
    )
    assert failed_rows is not None
    assert set(failed_rows) == {(10, 0), (11, 0)}
    assert {
        row["model"]["availability"]["reason"] for row in failed_rows.values()
    } == {"isolated_game_worker_protocol_failed"}

    addresses = ((10, 0), (11, 0))
    identity = {"submission_id": 77, "episode_id": 88}
    monkeypatch.setattr(
        application,
        "_prepare_baseline_game_materialization",
        lambda *_args, **_kwargs: ("sha256:worker-failure", identity, addresses),
    )
    monkeypatch.setattr(
        application,
        "_materialize_baseline_game_backend_traces",
        lambda *_args, **_kwargs: failed_rows,
    )
    assembled: list[tuple[int, int]] = []

    def assemble(
        _submission_id: int,
        _episode_id: int,
        step_index: int,
        factorized_stage: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # A missing precomputed row would be the old N-worker fallback.
        assert isinstance(kwargs["precomputed_trace"], dict)
        assert kwargs["prevalidated_game"] is True
        assembled.append((step_index, factorized_stage))
        return {"address": [step_index, factorized_stage]}

    monkeypatch.setattr(application, "_trace_payload_uncached", assemble)
    assert application.trace_payload(77, 88, 10, 0) == {"address": [10, 0]}
    _wait_for_game_completion(application, "sha256:worker-failure")
    assert sorted(assembled) == sorted(addresses)


def test_game_worker_refuses_nested_isolation_when_runtime_import_mismatches(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The child guard exits before a game worker could recursively spawn."""

    worker_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "replay_inspector_trace_worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "replay_inspector_trace_worker_nested_guard_test", worker_path
    )
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    class FakeApplication:
        def __init__(self, _config: object) -> None:
            pass

        @staticmethod
        def _imported_runtime_is(_root: Path) -> bool:
            return False

    monkeypatch.setattr(worker, "InspectorApplication", FakeApplication)
    request = {
        "config": {
            "bind_host": "127.0.0.1",
            "port": 8787,
            "replay_root": str(tmp_path / "archive"),
            "rollout_root": str(tmp_path / "rollouts"),
            "artifact_roots": [str(tmp_path / "artifacts")],
            "runtime_source_root": str(tmp_path / "foreign-runtime" / "poke_bot"),
            "web_root": str(tmp_path / "web"),
            "torch_threads": 1,
            "max_parameter_slice": 64,
            "max_tensor_values": 64,
            "verify_digests": True,
        },
        "mode": "game",
        "submission_id": 77,
        "episode_id": 88,
        "step_index": 10,
        "factorized_stage": 0,
        "addresses": [[10, 0]],
    }
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(sys, "stdout", stdout)

    assert worker.main() == 2
    record = json.loads(stdout.getvalue())
    assert record["kind"] == "error"
    assert record["code"] == "isolated_game_worker_runtime_import_mismatch"
