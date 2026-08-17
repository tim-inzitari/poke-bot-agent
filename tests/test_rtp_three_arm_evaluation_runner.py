"""Focused fail-closed checks for the isolated r198 arm executor.

These intentionally exercise the runner's own narrow boundaries without
requiring a native CG installation.  The hermetic end-to-end test owns the
real sealed 1,000-cell material; this file protects the local invariants that
would otherwise be easy to regress while wiring a production factory.
"""

from __future__ import annotations

import json
import io
import os
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from poke_bot import rtp_three_arm_evaluation_runner as runner


def _sealed_bytes(tmp_path: Path, name: str, payload: bytes) -> dict[str, Any]:
    path = tmp_path / name
    path.write_bytes(payload)
    os.chmod(path, 0o444)
    return {
        **runner._identity(path, name),
        "mode": 0o444,
    }


def _sealed_json(tmp_path: Path, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _sealed_bytes(
        tmp_path,
        name,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _captured_worker_process(
    *, stdout: str | bytes, stderr: str | bytes, returncode: int, pid: int = 4242
) -> runner._CapturedWorkerProcess:
    stdout_capture = runner._WorkerStreamCapture(
        parse_limit_bytes=runner._WORKER_RESPONSE_PARSE_MAX_BYTES
    )
    stderr_capture = runner._WorkerStreamCapture()
    stdout_capture.feed(stdout.encode("utf-8") if isinstance(stdout, str) else stdout)
    stderr_capture.feed(stderr.encode("utf-8") if isinstance(stderr, str) else stderr)
    stdout_capture.finish()
    stderr_capture.finish()
    return runner._CapturedWorkerProcess(
        returncode=returncode,
        child_pid=pid,
        stdout=stdout_capture,
        stderr=stderr_capture,
    )


def _evaluation_authority_payload(manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema": runner.AUTHORITY_SCHEMA,
        "status": "authorized_evaluation_only",
        "manifest_sha256": manifest_sha256,
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }


def _worker_environment_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], Path, Path]:
    """Create the small sealed path boundary needed by the child verifier."""

    source_root = tmp_path / "source-snapshot"
    runner_path = source_root / "poke_bot" / "rtp_three_arm_evaluation_runner.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# hermetic test source\n", encoding="utf-8")
    monkeypatch.setattr(runner, "__file__", str(runner_path))

    cg_runtime = tmp_path / "cg-runtime"
    cg_package = cg_runtime / "cg"
    cg_package.mkdir(parents=True)
    engine = _sealed_bytes(cg_package, "libcg.so", b"sealed-test-engine")
    os.chmod(cg_package, 0o555)
    os.chmod(cg_runtime, 0o555)

    closure_record = _sealed_json(
        tmp_path,
        "closure-record.json",
        {"schema": runner.EVALUATION_CG_CLOSURE_SCHEMA},
    )
    metadata_parity = _sealed_json(tmp_path, "metadata-parity.json", {"schema": "test"})
    source_manifest = _sealed_json(
        source_root,
        runner.R198_SOURCE_SNAPSHOT_MANIFEST,
        {
            "schema": runner.R198_SOURCE_SNAPSHOT_SCHEMA,
            "source_tree_sha256": "sha256:" + "a" * 64,
            "eval_cg_closure": {
                "library": engine,
                "runtime_root": str(cg_runtime),
                "runtime_path": str(cg_package),
                "physical_read_only_copy": True,
                "library_mode": 0o444,
                "closure_manifest": closure_record,
            },
        },
    )
    runtime_contract = _sealed_json(
        tmp_path,
        "candidate-runtime-contract.json",
        {
            "schema": runner.R198_CANDIDATE_SNAPSHOT_SCHEMA,
            "status": "sealed",
            "no_symlinks": True,
            "all_paths_read_only": True,
        },
    )
    closure = {
        "receipt": {"sha256": "sha256:" + "b" * 64},
        "engine_artifact": engine,
        "runtime_library": engine,
        "evidence": {
            "closure_manifest": {"identity": closure_record},
            "metadata_parity": {"identity": metadata_parity},
        },
    }
    environment = {
        "CG_LIB_PATH": str(cg_runtime),
        "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT": str(source_root),
        "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256": "sha256:" + "a" * 64,
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT": runtime_contract["path"],
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256": runtime_contract["sha256"],
        "CUDA_VISIBLE_DEVICES": runner.R198_BLACKWELL_UUID,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
    }
    return (
        environment,
        {"engine_artifact": engine},
        closure,
        cg_runtime,
        source_manifest["path"],
    )


def _patch_runner_job_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cell_count: int = 4,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    manifest_path = input_root / "bounded-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    cells: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    for index in range(cell_count):
        cell_id = f"cell-{index:06d}"
        rng_id = f"rng-{index:06d}"
        rng_identity = {
            "id": rng_id,
            "kind": "snapshot",
            "sha256": f"sha256:snapshot-{index}",
            "bytes": index + 1,
            "seal_sha256": f"sha256:seal-{index}",
            "capture_boundary": "test-boundary",
            "boundary_tag": 1,
        }
        cells.append(
            {
                "cell_id": cell_id,
                "evaluation_case_id": f"case-{index:06d}",
                "evaluation_case_bindings_sha256": f"sha256:case-{index}",
                "opponent_id": "iono",
                "candidate_seat": index % 2,
                "rng_identity": rng_identity,
            }
        )
        snapshots[rng_id] = {
            **rng_identity,
            "seal": {"path": f"/sealed/{rng_id}.json"},
        }
    manifest = {
        "schedule": cells,
        "opponents": [{"id": "iono"}],
    }
    manifest_identity = {
        "path": str(manifest_path),
        "sha256": "sha256:bounded-manifest",
        "bytes": manifest_path.stat().st_size,
    }
    monkeypatch.setattr(
        runner, "_read_manifest", lambda _path: (manifest, manifest_identity)
    )
    monkeypatch.setattr(runner, "_verify_pairing_capability", lambda _manifest: {})
    monkeypatch.setattr(runner, "_verify_evaluation_cg_closure", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "_verify_evaluation_authority",
        lambda *_args, **_kwargs: {"identity": {}},
    )
    monkeypatch.setattr(
        runner,
        "_verify_cohort_and_source_exclusion",
        lambda _manifest: {"cohort": {"sha256": "sha256:cohort"}},
    )
    monkeypatch.setattr(runner, "_verify_planner_preflight", lambda _manifest: None)
    monkeypatch.setattr(runner, "_sealed_opponents", lambda _manifest: {"iono": {}})
    monkeypatch.setattr(runner, "_verify_snapshot_package", lambda *_args: None)
    monkeypatch.setattr(runner, "_require_physical_readonly_tree", lambda _entry: {})
    monkeypatch.setattr(
        runner, "_sealed_snapshot_materials", lambda *_args: snapshots
    )
    monkeypatch.setattr(runner, "_verify_snapshot_for_cell", lambda **_kwargs: None)
    written_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        runner,
        "_write_row_evidence",
        lambda **kwargs: written_rows.append(dict(kwargs)) or {},
    )
    return manifest_path, tmp_path / "runner-output" / "bounded-results.json", written_rows


def _failed_worker_spawn_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completed: SimpleNamespace,
) -> dict[str, Any]:
    """Patch only the child boundary for failed-worker evidence tests."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_identity = runner._identity(manifest_path, "test manifest")
    output_root = tmp_path / "failed-output"
    output_root.mkdir()
    source_root = tmp_path / "source-snapshot"
    source_root.mkdir()
    _sealed_json(
        source_root,
        runner.R198_SOURCE_SNAPSHOT_MANIFEST,
        {"schema": runner.R198_SOURCE_SNAPSHOT_SCHEMA},
    )
    runtime_contract = _sealed_json(
        tmp_path,
        "candidate-runtime-contract.json",
        {"schema": runner.R198_CANDIDATE_SNAPSHOT_SCHEMA},
    )
    seal = _sealed_json(tmp_path, "snapshot-seal.json", {"schema": "test"})
    authority = _sealed_json(tmp_path, "authority.json", {"schema": "test"})

    class Factory:
        def worker_environment(self, **_kwargs: Any) -> dict[str, str]:
            return {}

    environment = {
        "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT": str(source_root),
        "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256": "sha256:" + "a" * 64,
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT": runtime_contract["path"],
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256": runtime_contract["sha256"],
    }
    environment_identity = {
        "common_sanitized_environment_sha256": "sha256:common",
        "arm_environment_sha256": "sha256:arm",
    }
    monkeypatch.setattr(runner, "_load_factory", lambda _reference: Factory())
    monkeypatch.setattr(
        runner,
        "_sanitize_environment",
        lambda _raw, _scratch, **_kwargs: (environment, environment_identity),
    )
    monkeypatch.setattr(
        runner,
        "_write_action_fence",
        lambda **_kwargs: {"identity": authority, "payload": {}},
    )
    monkeypatch.setattr(
        runner,
        "_capture_worker_process",
        lambda **_kwargs: _captured_worker_process(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        ),
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_identity": manifest_identity,
        "factory_ref": "test:Factory",
        "cell": {
            "cell_id": "cell-000000",
            "evaluation_case_id": "case-000000",
            "evaluation_case_bindings_sha256": "sha256:case",
            "opponent_id": "iono",
            "candidate_seat": 0,
        },
        "arm": runner.CANONICAL_DIRECT_ARM,
        "snapshot": {"seal": seal},
        "capability": {},
        "closure": {},
        "authority": {"identity": authority},
        "output_root": output_root,
        "max_steps": 1,
    }


@pytest.mark.unit
def test_complexity_probe_is_nonmutating_and_excluded_from_candidate_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    probe_state = {"value": 0}

    def state_digest() -> str:
        events.append("state")
        return f"state:{probe_state['value']}"

    def complexity_intent(_observation: dict[str, Any]) -> dict[str, Any]:
        events.append("probe")
        return {"intended_complex": True, "planner_reason": "shared_gate"}

    def candidate(_observation: dict[str, Any]) -> list[int]:
        events.append("candidate")
        return [7]

    ticks = iter((10.0, 10.25))

    def clock() -> float:
        events.append("timer")
        return next(ticks)

    monkeypatch.setattr(runner.time, "perf_counter", clock)
    runtime = runner.ArmRuntime(
        candidate=candidate,
        opponent=lambda _observation: [0],
        runtime_identity={},
        isolation={},
        complexity_intent=complexity_intent,
        complexity_probe_state_digest=state_digest,
    )

    action, latency, intent = runner._timed_candidate_selection(
        runtime, candidate, {"current": {"yourIndex": 0}}
    )

    assert action == [7]
    assert latency == pytest.approx(0.25)
    assert intent == {"intended_complex": True, "planner_reason": "shared_gate"}
    # The first timer tick happens only after the probe returned and its
    # before/after digest matched.  The probe's work cannot enter latency.
    assert events == ["state", "probe", "state", "timer", "candidate", "timer"]


@pytest.mark.unit
def test_complexity_probe_mutation_fails_before_candidate_selection() -> None:
    calls: list[str] = []
    state = {"value": 0}

    def mutating_probe(_observation: dict[str, Any]) -> dict[str, Any]:
        calls.append("probe")
        state["value"] += 1
        return {"intended_complex": False, "planner_reason": "shared_gate"}

    def candidate(_observation: dict[str, Any]) -> list[int]:
        calls.append("candidate")
        return [1]

    runtime = runner.ArmRuntime(
        candidate=candidate,
        opponent=lambda _observation: [0],
        runtime_identity={},
        isolation={},
        complexity_intent=mutating_probe,
        complexity_probe_state_digest=lambda: f"state:{state['value']}",
    )

    with pytest.raises(runner.RTPThreeArmRunnerError, match="mutated candidate state"):
        runner._timed_candidate_selection(runtime, candidate, {"current": {"yourIndex": 0}})
    assert calls == ["probe"]


@pytest.mark.unit
def test_runtime_profile_uses_explicit_probe_and_action_sidecar_fields(
    tmp_path: Path,
) -> None:
    sidecar = _sealed_bytes(tmp_path, "sidecar.pt", b"sidecar")
    artifact = _sealed_bytes(tmp_path, "runtime.json", b"runtime")
    profile_payload = {
        "rtp": {
            "recursive_turn_planner_enabled": True,
            "direct_bridge_enabled": True,
            "force_direct_bridge_only": True,
            "max_neural_passes": 256,
            "max_action_combos": 1024,
        }
    }
    direct_profile = _sealed_json(tmp_path, "direct-profile.json", profile_payload)
    recursive_profile = _sealed_json(
        tmp_path,
        "recursive-profile.json",
        {
            "rtp": {
                **profile_payload["rtp"],
                "force_direct_bridge_only": False,
            }
        },
    )
    no_rtp_profile = _sealed_json(
        tmp_path,
        "no-rtp-profile.json",
        {
            "rtp": {
                "recursive_turn_planner_enabled": False,
                "direct_bridge_enabled": False,
                "force_direct_bridge_only": False,
                "max_neural_passes": None,
                "max_action_combos": None,
            }
        },
    )
    manifest = {
        "arms": {
            "no_rtp": {"runtime_artifact": artifact, "runtime_profile": no_rtp_profile, "rtp_sidecar": None},
            runner.CANONICAL_DIRECT_ARM: {
                "runtime_artifact": artifact,
                "runtime_profile": direct_profile,
                "rtp_sidecar": sidecar,
            },
            "recursive_rtp": {
                "runtime_artifact": artifact,
                "runtime_profile": recursive_profile,
                "rtp_sidecar": sidecar,
            },
        },
        "shared_artifacts": {},
    }
    runtime = {
        "arm": runner.CANONICAL_DIRECT_ARM,
        "runtime_artifact_sha256": artifact["sha256"],
        "runtime_profile_sha256": direct_profile["sha256"],
        "action_attached_rtp_sidecar_sha256": sidecar["sha256"],
        "complexity_probe_sidecar_sha256": sidecar["sha256"],
        "complexity_probe_sidecar_instrumentation_only": True,
        "complexity_probe_latency_excluded": True,
        "rtp_action_attachment_enabled": True,
        "rtp_action_authority_enabled": False,
        **profile_payload["rtp"],
    }

    checked = runner._runtime_profile_contract(
        manifest, runner.CANONICAL_DIRECT_ARM, runtime
    )
    assert checked["action_attached_rtp_sidecar_sha256"] == sidecar["sha256"]
    assert checked["complexity_probe_sidecar_sha256"] == sidecar["sha256"]

    with pytest.raises(runner.RTPThreeArmRunnerError, match="explicit probe/action"):
        runner._runtime_profile_contract(
            manifest,
            runner.CANONICAL_DIRECT_ARM,
            {**runtime, "rtp_sidecar_sha256": sidecar["sha256"]},
        )


@pytest.mark.unit
def test_action_execution_context_is_cell_seat_and_process_bound(tmp_path: Path) -> None:
    manifest = _sealed_json(tmp_path, "manifest.json", {"schema": "test"})
    authority_identity = _sealed_json(tmp_path, "authority.json", {"schema": "test"})
    runtime_contract = _sealed_json(tmp_path, "runtime.json", {"schema": "test"})
    fence_identity = _sealed_json(tmp_path, "fence.json", {"schema": "test"})
    cell = {
        "cell_id": "cell-000042",
        "evaluation_case_id": "case-000042",
        "opponent_id": "iono",
        "candidate_seat": 1,
    }
    context = runner._evaluation_action_execution_context(
        manifest_identity=manifest,
        authority={"identity": authority_identity},
        runtime_contract=runtime_contract,
        action_fence={"identity": fence_identity},
        cell=cell,
        arm="recursive_rtp",
        launch_nonce="a" * 48,
    )

    assert context["opponent_id"] == "iono"
    assert context["candidate_seat"] == 1
    assert context["process"] == runner._process_identity()
    assert context["execution_kind"] == "evaluation_action_execution"
    assert context["evaluation_only"] is True
    for field in (
        "training_eligible",
        "replay_eligible",
        "serving_change_authorized",
        "selector_change_authorized",
        "action_authority_authorized",
        "kaggle_submission_authorized",
        "serving_eligible",
        "action_authority_enabled",
        "submission_eligible",
        "promotion_eligible",
    ):
        assert context[field] is False


@pytest.mark.unit
def test_action_fence_creation_rehashes_the_new_immutable_file(tmp_path: Path) -> None:
    parent = _sealed_bytes(tmp_path, "parent.pt", b"parent")
    sidecar = _sealed_bytes(tmp_path, "sidecar.pt", b"sidecar")
    manifest_identity = _sealed_json(tmp_path, "manifest.json", {"schema": "test"})
    authority_identity = _sealed_json(tmp_path, "authority.json", {"schema": "test"})
    cell = {
        "cell_id": "cell-000042",
        "evaluation_case_id": "case-000042",
        "opponent_id": "iono",
        "candidate_seat": 1,
    }
    manifest = {
        "shared_artifacts": {"parent_checkpoint": parent},
        "arms": {
            runner.CANONICAL_DIRECT_ARM: {"rtp_sidecar": sidecar},
            "recursive_rtp": {"rtp_sidecar": sidecar},
        },
    }
    fence = runner._write_action_fence(
        scratch=tmp_path / "scratch",
        manifest_identity=manifest_identity,
        authority={"identity": authority_identity},
        manifest=manifest,
        cell=cell,
        arm="recursive_rtp",
        launch_nonce="b" * 48,
    )

    identity = fence["identity"]
    assert identity["mode"] == 0o444
    assert Path(identity["path"]).stat().st_mode & 0o777 == 0o444
    checked = runner._verify_action_fence(
        identity,
        manifest_identity=manifest_identity,
        authority={"identity": authority_identity},
        cell=cell,
        arm="recursive_rtp",
        launch_nonce="b" * 48,
    )
    assert checked["identity"] == identity
    assert checked["payload"]["opponent_id"] == "iono"
    assert checked["payload"]["candidate_seat"] == 1


@pytest.mark.unit
def test_engine_must_use_the_snapshot_local_closure_dso(tmp_path: Path) -> None:
    capability_engine = _sealed_bytes(tmp_path, "private-engine.so", b"same-engine")
    closure_engine = _sealed_bytes(tmp_path, "snapshot-libcg.so", b"same-engine")
    other_equal_bytes = _sealed_bytes(tmp_path, "other-libcg.so", b"same-engine")
    capability = {"engine_artifact": capability_engine, "abi": {"version": 2}}

    class Engine:
        identity = capability_engine
        library_identity = closure_engine
        abi = {"version": 2}

    verified = runner._verify_engine_identity(
        Engine(), capability, {}, loaded_closure_engine=closure_engine
    )
    assert verified["loaded_path"] == closure_engine["path"]

    class SplitHandleEngine:
        identity = capability_engine
        library_identity = other_equal_bytes
        abi = {"version": 2}

    with pytest.raises(runner.RTPThreeArmRunnerError, match="closure libcg.so"):
        runner._verify_engine_identity(
            SplitHandleEngine(), capability, {}, loaded_closure_engine=closure_engine
        )


@pytest.mark.unit
def test_telemetry_rejects_recursion_in_direct_arm_and_records_recursive_trace() -> None:
    direct = runner.DecisionTelemetry(arm=runner.CANONICAL_DIRECT_ARM)
    direct.observe(
        {"mode": "direct_bridge", "extras": {"force_direct_bridge_only": True}},
        0.01,
        complexity_intent={"intended_complex": True, "planner_reason": "gate"},
    )
    direct_row = direct.as_dict()
    assert direct_row["direct_bridge_decisions"] == 1
    assert direct_row["recursive_decisions"] == 0
    assert direct_row["decision_latency_trace"][0]["mode"] == runner.CANONICAL_DIRECT_ARM

    with pytest.raises(runner.RTPThreeArmRunnerError, match="non-recursive arm"):
        direct.observe(
            {"mode": "recursive_plan", "neural_passes": 6},
            0.01,
            complexity_intent={"intended_complex": True, "planner_reason": "gate"},
        )

    recursive = runner.DecisionTelemetry(arm="recursive_rtp")
    recursive.observe(
        {
            "mode": "recursive_plan",
            "neural_passes": 6,
            "extras": {
                "pre_forcing_complexity_intent": {
                    "inherited": False,
                    "new_turn": True,
                    "would_recurse": True,
                }
            },
        },
        0.2,
        complexity_intent={"intended_complex": True, "planner_reason": "gate"},
    )
    recursive_row = recursive.as_dict()
    assert recursive_row["recursive_decisions"] == 1
    assert recursive_row["successful_recursive_intended_complex_decisions"] == 1
    assert recursive_row["recursive_mode_counts"]["recursive_plan"] == 1
    assert recursive_row["decision_latency_trace"][0] == {
        "decision_index": 0,
        "mode": "recursive_rtp",
        "planner_mode": "recursive_plan",
        "planner_reason": "recursive_plan",
        "intended_complex": True,
        "fallback_classification": None,
        "latency_seconds": 0.2,
    }


@pytest.mark.unit
def test_telemetry_accounts_inherited_legal_empty_forced_replan() -> None:
    """A zero-selection replan remains a visible five-pass recursive action."""

    telemetry = runner.DecisionTelemetry(arm="recursive_rtp")
    telemetry.observe(
        {
            "mode": "replan_with_program",
            "neural_passes": 5,
            "required_neural_passes": 5,
            "fallback_code": "",
            "decision_mode": "recursive_plan",
            "extras": {
                "pre_forcing_complexity_intent": {
                    "inherited": True,
                    "new_turn": False,
                    "would_recurse": True,
                },
                "loaded_program_first_step": {
                    "phase": "replan",
                    "expected_action": [],
                    "executor_action": [],
                    "repaired": False,
                },
            },
        },
        0.2,
        complexity_intent={
            "intended_complex": False,
            "planner_reason": "not_new_turn_complexity_gate",
        },
        returned_action=[],
    )

    row = telemetry.as_dict()
    assert row["recursive_decisions"] == 1
    assert row["fallback_decisions"] == 0
    assert row["forced_replan_passes"] == [5]
    assert row["normal_recursive_plan_passes"] == []
    assert row["decision_diagnostics"] == [
        {
            "planner_mode": "replan_with_program",
            "planner_reason": "replan_with_program",
            "complexity_intent": {
                "intended_complex": False,
                "planner_reason": "not_new_turn_complexity_gate",
            },
            "fallback_code": "",
            "neural_passes": 5,
            "required_neural_passes": 5,
        }
    ]


@pytest.mark.unit
def test_forced_turn_order_control_bypasses_probe_and_policy_telemetry() -> None:
    events: list[str] = []
    observation = {
        "current": {"yourIndex": 0, "turn": 0},
        "select": {
            "context": 41,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 2}, {"type": 1}],
        },
    }

    runtime = runner.ArmRuntime(
        candidate=lambda _observation: [1],
        opponent=lambda _observation: [0],
        runtime_identity={},
        isolation={},
        complexity_intent=lambda _observation: pytest.fail(
            "forced turn-order control invoked complexity probe"
        ),
        complexity_probe_state_digest=lambda: pytest.fail(
            "forced turn-order control requested probe state"
        ),
    )

    def candidate(_observation: dict[str, Any]) -> list[int]:
        events.append("candidate")
        return [1]

    action, latency, intent, control, over_cap = runner._select_candidate_action(
        runtime, candidate, observation
    )
    assert action == [1]
    assert latency is None
    assert intent is None
    assert control == ([1], 41, "numeric_41")
    assert over_cap is None
    assert events == ["candidate"]

    telemetry = runner.DecisionTelemetry(arm=runner.CANONICAL_DIRECT_ARM)
    runner._record_forced_turn_order_control(
        telemetry,
        SimpleNamespace(last_rtp_diagnostics=None),
        expected_action=control[0],
        returned_action=action,
        prompt_context=control[1],
        prompt_context_encoding=control[2],
    )
    row = telemetry.as_dict()
    assert row["forced_turn_order_controls"] == 1
    assert row["candidate_decisions"] == 0
    assert row["intended_complex_decisions"] == 0
    assert row["direct_bridge_decisions"] == 0
    assert row["recursive_decisions"] == 0
    assert row["latency_seconds"] == 0.0
    assert row["decision_latency_trace"] == []
    assert row["forced_turn_order_control_trace"] == [
        {
            "control_index": 0,
            "control": "forced_go_first_contract",
            "prompt_context": 41,
            "prompt_context_encoding": "numeric_41",
            "expected_action": [1],
            "returned_action": [1],
            "verified_observation_action_contract": True,
            "rtp_diagnostics_absent": True,
            "complexity_probe_not_invoked": True,
            "excluded_from_candidate_decisions": True,
            "excluded_from_intended_complex_denominator": True,
            "excluded_from_latency": True,
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "arm",
    ("no_rtp", runner.CANONICAL_DIRECT_ARM, "recursive_rtp"),
)
def test_over_cap_factorized_fallback_bypasses_probe_and_excludes_planner_metrics(
    arm: str,
) -> None:
    """The n=9, counts=1..5 decision is 18,729 without enumeration."""

    from poke_bot import features

    observation = {
        "select": {
            "minCount": 1,
            "maxCount": 5,
            "option": [{"type": 2} for _ in range(9)],
        }
    }
    action_space = features.complete_ordered_action_space_summary(
        observation, max_combos=1024
    )
    assert action_space["complete_ordered_action_cardinality"] == 18_729
    selected_calls: list[str] = []
    runtime = runner.ArmRuntime(
        candidate=lambda _observation: [0, 2, 3, 6, 5],
        opponent=lambda _observation: [0],
        runtime_identity={},
        isolation={},
        complexity_intent=lambda _observation: pytest.fail("over-cap invoked complexity probe"),
        complexity_probe_state_digest=lambda: pytest.fail("over-cap requested probe digest"),
        candidate_policy_input_fingerprint=lambda _observation: "sha256:" + "a" * 64,
    )

    def candidate(_observation: dict[str, Any]) -> list[int]:
        selected_calls.append("candidate")
        return [0, 2, 3, 6, 5]

    action, latency, intent, control, special = runner._select_candidate_action(
        runtime, candidate, observation
    )
    assert action == [0, 2, 3, 6, 5]
    assert latency is not None and latency >= 0.0
    assert intent is None
    assert control is None
    assert selected_calls == ["candidate"]
    assert special is not None
    assert special["action_space"] == action_space

    with pytest.raises(runner.RTPThreeArmRunnerError, match="cannot carry complexity intent"):
        runner.DecisionTelemetry(arm=arm).observe(
            None if arm == "no_rtp" else {},
            float(latency),
            complexity_intent={"intended_complex": False, "planner_reason": "late"},
            returned_action=action,
            over_cap_factorized_fallback=special,
        )

    diagnostic: dict[str, Any] | None
    if arm == "no_rtp":
        diagnostic = None
    else:
        diagnostic = {
            "mode": "fallback",
            "fallback_code": "action_space_too_large",
            "neural_passes": 0,
            "required_neural_passes": 0,
            "legal_count": 0,
            "decision_mode": "",
            "extras": {
                runner.OVER_CAP_FACTORIZED_FALLBACK_MODE: {
                    "classification": runner.OVER_CAP_FACTORIZED_FALLBACK_REASON,
                    "action_space": action_space,
                    "factorized_greedy_fallback": True,
                }
            },
        }
    telemetry = runner.DecisionTelemetry(arm=arm)
    telemetry.observe(
        diagnostic,
        float(latency),
        complexity_intent=intent,
        returned_action=action,
        over_cap_factorized_fallback=special,
    )
    row = telemetry.as_dict()
    assert row["candidate_decisions"] == 1
    assert row["planner_eligible_candidate_decisions"] == 0
    assert row["over_cap_factorized_fallback_decisions"] == 1
    assert row["intended_complex_decisions"] == 0
    assert row["direct_bridge_decisions"] == 0
    assert row["recursive_decisions"] == 0
    assert row["fallback_decisions"] == 0
    assert row["normal_recursive_plan_passes"] == []
    assert row["forced_replan_passes"] == []
    assert row["decision_latency_trace"][0]["intended_complex"] is None
    assert row["over_cap_factorized_fallback_trace"][0]["returned_action"] == action
    assert row["over_cap_factorized_fallback_trace"][0][
        "factorized_teacher_forcing_stage_count"
    ] == 5
    if arm == "no_rtp":
        assert row["over_cap_factorized_fallback_trace"][0]["rtp_diagnostic"] is None
    else:
        assert row["over_cap_factorized_fallback_trace"][0]["rtp_diagnostic"] == {
            "mode": "fallback",
            "fallback_code": "action_space_too_large",
            "neural_passes": 0,
            "required_neural_passes": 0,
            "legal_count": 0,
            "decision_mode": "",
        }
        hidden_probe = dict(diagnostic or {})
        hidden_probe["extras"] = {
            **dict(hidden_probe["extras"]),
            "pre_forcing_complexity_intent": {"would_recurse": False},
        }
        with pytest.raises(runner.RTPThreeArmRunnerError, match="planner work"):
            runner.DecisionTelemetry(arm=arm).observe(
                hidden_probe,
                float(latency),
                complexity_intent=None,
                returned_action=action,
                over_cap_factorized_fallback=special,
            )


@pytest.mark.unit
@pytest.mark.parametrize("bad_bound", (1.0, "1", True))
def test_over_cap_selection_rejects_coercible_bounds_before_policy_or_probe(
    bad_bound: object,
) -> None:
    from poke_bot import features

    observation = {
        "select": {
            "minCount": bad_bound,
            "maxCount": 5,
            "option": [{"type": 2} for _ in range(9)],
        }
    }
    runtime = runner.ArmRuntime(
        candidate=lambda _observation: pytest.fail("coercible over-cap reached candidate"),
        opponent=lambda _observation: [0],
        runtime_identity={},
        isolation={},
        complexity_intent=lambda _observation: pytest.fail("coercible over-cap reached probe"),
        complexity_probe_state_digest=lambda: pytest.fail("coercible over-cap read probe state"),
        candidate_policy_input_fingerprint=lambda _observation: pytest.fail(
            "coercible over-cap reached fingerprint"
        ),
    )
    with pytest.raises(features.FeatureContractError, match="exact integers"):
        runner._select_candidate_action(runtime, runtime.candidate, observation)


@pytest.mark.unit
def test_forced_turn_order_control_rejects_tampering_but_normal_missing_diag_stays_fatal() -> None:
    telemetry = runner.DecisionTelemetry(arm=runner.CANONICAL_DIRECT_ARM)
    with pytest.raises(runner.RTPThreeArmRunnerError, match="action_mismatch"):
        runner._record_forced_turn_order_control(
            telemetry,
            SimpleNamespace(last_rtp_diagnostics=None),
            expected_action=[0],
            returned_action=[1],
            prompt_context=41,
            prompt_context_encoding="numeric_41",
        )
    with pytest.raises(runner.RTPThreeArmRunnerError, match="emitted_rtp_diagnostics"):
        runner._record_forced_turn_order_control(
            telemetry,
            SimpleNamespace(last_rtp_diagnostics={"mode": "direct_bridge"}),
            expected_action=[0],
            returned_action=[0],
            prompt_context=41,
            prompt_context_encoding="numeric_41",
        )
    with pytest.raises(runner.RTPThreeArmRunnerError, match="omitted per-decision diagnostics"):
        telemetry.observe(
            None,
            0.01,
        complexity_intent={"intended_complex": False, "planner_reason": "normal"},
    )


@pytest.mark.unit
@pytest.mark.parametrize("raw_context", (41.0, "41"))
def test_forced_turn_order_control_rejects_noncanonical_numeric_context(
    raw_context: object,
) -> None:
    observation = {
        "select": {
            "context": raw_context,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 1}, {"type": 2}],
        }
    }
    with pytest.raises(runner.RTPThreeArmRunnerError, match="unrecognized prompt context"):
        runner._forced_turn_order_control(observation)

    telemetry = runner.DecisionTelemetry(arm=runner.CANONICAL_DIRECT_ARM)
    with pytest.raises(runner.RTPThreeArmRunnerError, match="invalid prompt context"):
        telemetry.observe_forced_turn_order_control(
            expected_action=[0],
            returned_action=[0],
            prompt_context=raw_context,  # type: ignore[arg-type]
            prompt_context_encoding="numeric_41",
        )


@pytest.mark.unit
@pytest.mark.parametrize("action", ([0.0], ["0"], [True]))
def test_forced_turn_order_control_rejects_noninteger_action_indices(
    action: list[object],
) -> None:
    telemetry = runner.DecisionTelemetry(arm=runner.CANONICAL_DIRECT_ARM)
    with pytest.raises(runner.RTPThreeArmRunnerError, match="does not exactly match Yes"):
        telemetry.observe_forced_turn_order_control(
            expected_action=action,  # type: ignore[arg-type]
            returned_action=action,  # type: ignore[arg-type]
            prompt_context=41,
            prompt_context_encoding="numeric_41",
        )


@pytest.mark.unit
def test_spawn_reuses_the_same_sealed_snapshot_for_all_three_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"
    output_root.mkdir()
    seal = _sealed_json(tmp_path, "cell-seal.json", {"schema": "seal"})
    fence = _sealed_json(tmp_path, "fence.json", {"schema": "fence"})
    observed_requests: list[dict[str, Any]] = []
    observed_envs: list[dict[str, str]] = []

    class Factory:
        def worker_environment(self, **_kwargs: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(runner, "_load_factory", lambda _reference: Factory())
    monkeypatch.setattr(
        runner,
        "_sanitize_environment",
        lambda _raw, _scratch, **_kwargs: (
            {"PYTHONPATH": str(tmp_path)},
            {
                "common_sanitized_environment_sha256": "sha256:common",
                "arm_environment_sha256": "sha256:arm",
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_action_fence",
        lambda **_kwargs: {"identity": fence, "payload": {}},
    )

    def fake_capture(**kwargs: Any) -> runner._CapturedWorkerProcess:
        observed_requests.append(dict(kwargs["request"]))
        observed_envs.append(dict(kwargs["environment"]))
        return _captured_worker_process(
            returncode=0,
            stdout=json.dumps({"schema": "test-worker-response"}),
            stderr="",
        )

    monkeypatch.setattr(runner, "_capture_worker_process", fake_capture)
    cell = {
        "cell_id": "cell-000000",
        "evaluation_case_id": "case-000000",
        "opponent_id": "iono",
        "candidate_seat": 0,
    }
    for arm in ("no_rtp", runner.CANONICAL_DIRECT_ARM, "recursive_rtp"):
        runner._spawn_worker(
            manifest_path=str(manifest_path),
            manifest_identity={"path": str(manifest_path), "sha256": "sha256:manifest", "bytes": 3},
            factory_ref="test:Factory",
            cell=cell,
            arm=arm,
            snapshot={"seal": seal},
            capability={},
            closure={},
            authority={"identity": fence},
            output_root=output_root,
            max_steps=10,
        )

    assert len(observed_requests) == 3
    assert {request["snapshot_seal"]["path"] for request in observed_requests} == {
        seal["path"]
    }
    assert all("requested_seed" not in request for request in observed_requests)
    assert observed_requests[0]["action_fence"] is None
    assert all(request["action_fence"] is not None for request in observed_requests[1:])
    assert all(
        not set(environment).intersection(runner._RUNNER_ACTION_FENCE_ENV)
        for environment in observed_envs
    )


@pytest.mark.unit
def test_nonzero_worker_seals_no_clobber_failed_evidence_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _failed_worker_spawn_inputs(
        tmp_path,
        monkeypatch,
        completed=SimpleNamespace(
            returncode=2,
            stdout="x" * 9000 + "\x00tail",
            stderr="child failed\x1b[31m",
        ),
    )
    monkeypatch.setattr(runner.secrets, "token_hex", lambda _size: "fixed-nonce")

    with pytest.raises(runner.RTPThreeArmRunnerError, match="isolated worker failed"):
        runner._spawn_worker(**inputs)

    evidence_paths = sorted((inputs["output_root"] / "failed-worker-evidence").glob("*.json"))
    assert len(evidence_paths) == 1
    evidence_path = evidence_paths[0]
    original_bytes = evidence_path.read_bytes()
    evidence = json.loads(original_bytes)
    assert evidence["schema"] == runner.FAILED_WORKER_EVIDENCE_SCHEMA
    assert evidence["status"] == "failed_closed_not_an_evaluation_result"
    assert evidence["not_an_evaluation_result"] is True
    assert evidence["not_a_result_row"] is True
    assert evidence["not_an_execution_receipt"] is True
    assert evidence["not_a_transcript"] is True
    assert evidence["evaluation_only"] is True
    for key, value in runner._FAILED_WORKER_AUTHORITY_DENIALS.items():
        assert evidence[key] is value
    assert evidence["returncode"] == 2
    assert evidence["evaluation"]["cell_id"] == inputs["cell"]["cell_id"]
    assert evidence["evaluation"]["arm"] == runner.CANONICAL_DIRECT_ARM
    assert evidence["worker_request"]["launch_nonce"] == "fixed-nonce"
    assert evidence["action_fence_identity"] == inputs["authority"]["identity"]
    assert (
        evidence["runtime_identity"]["candidate_runtime_contract"]["observation"]
        == "matched"
    )
    assert (
        evidence["source_snapshot_identity"]["manifest"]["observation"]
        == "observed_without_claimed_digest"
    )
    # The source-tree identity is distinct from the source manifest's file
    # digest, so the diagnostic retains both without conflating the two.
    assert evidence["stdout"]["truncated"] is True
    assert evidence["stdout"]["captured_bytes"] <= runner._FAILED_WORKER_CAPTURE_BYTES
    assert evidence["stdout"]["sha256"] == runner._sha256_text("x" * 9000 + "\x00tail")
    assert "\\u001b" in evidence["stderr"]["captured_text"]
    assert stat.S_IMODE(os.lstat(evidence_path).st_mode) == 0o444
    assert not (inputs["output_root"] / "execution-receipts").exists()
    assert not (inputs["output_root"] / "transcripts").exists()

    # Reusing a launch nonce must never overwrite or silently accept a prior
    # failed child diagnostic, even if the child output is byte-identical.
    with pytest.raises(
        runner.RTPThreeArmRunnerError, match="unable to seal failed-worker evidence"
    ) as excinfo:
        runner._spawn_worker(**inputs)
    assert excinfo.value.__cause__ is not None
    assert "reuse is forbidden" in str(excinfo.value.__cause__)
    assert evidence_path.read_bytes() == original_bytes
    assert sorted((inputs["output_root"] / "failed-worker-evidence").glob("*.json")) == [
        evidence_path
    ]


@pytest.mark.unit
def test_invalid_worker_stdout_seals_failed_evidence_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _failed_worker_spawn_inputs(
        tmp_path,
        monkeypatch,
        completed=SimpleNamespace(
            returncode=0,
            stdout="not-json\x00",
            stderr="worker diagnostic",
        ),
    )

    with pytest.raises(
        runner.RTPThreeArmRunnerError, match="did not emit one JSON response"
    ):
        runner._spawn_worker(**inputs)

    evidence_paths = list((inputs["output_root"] / "failed-worker-evidence").glob("*.json"))
    assert len(evidence_paths) == 1
    evidence = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
    assert evidence["failure"]["stage"] == "worker_stdout_invalid_json"
    assert evidence["returncode"] == 0
    assert evidence["stdout"]["captured_text"] == "not-json\\u0000"
    assert evidence["stdout"]["sha256"] == runner._sha256_text("not-json\x00")
    assert evidence["not_an_evaluation_result"] is True
    assert stat.S_IMODE(os.lstat(evidence_paths[0]).st_mode) == 0o444


@pytest.mark.unit
def test_failed_worker_capture_is_raw_bounded_and_never_parses_oversize_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A noisy zero-exit child is diagnostic-only, without unbounded capture."""

    real_capture = runner._capture_worker_process
    inputs = _failed_worker_spawn_inputs(
        tmp_path,
        monkeypatch,
        completed=SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(runner, "_capture_worker_process", real_capture)
    stdout_bytes = runner._WORKER_RESPONSE_PARSE_MAX_BYTES + 1
    stderr_bytes = runner._FAILED_WORKER_CAPTURE_BYTES * 3
    monkeypatch.setattr(
        runner,
        "_worker_subprocess_argv",
        lambda _request: [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdin.buffer.read(); "
                f"sys.stdout.buffer.write(b'x' * {stdout_bytes}); "
                f"sys.stderr.buffer.write(b'y' * {stderr_bytes})"
            ),
        ],
    )

    with pytest.raises(runner.RTPThreeArmRunnerError, match="response parse limit"):
        runner._spawn_worker(**inputs)

    evidence_path = next((inputs["output_root"] / "failed-worker-evidence").glob("*.json"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["failure"]["stage"] == "worker_stdout_parse_limit_exceeded"
    assert evidence["stdout"]["bytes"] == stdout_bytes
    assert evidence["stdout"]["raw_bytes"] == stdout_bytes
    assert evidence["stdout"]["response_parse_limit_exceeded"] is True
    assert evidence["stdout"]["capture_limit_bytes"] == runner._FAILED_WORKER_CAPTURE_BYTES
    assert evidence["stdout"]["captured_bytes"] <= runner._FAILED_WORKER_CAPTURE_BYTES
    assert evidence["stderr"]["bytes"] == stderr_bytes
    assert evidence["stderr"]["captured_bytes"] <= runner._FAILED_WORKER_CAPTURE_BYTES
    assert not (inputs["output_root"] / "execution-receipts").exists()
    assert not (inputs["output_root"] / "transcripts").exists()


@pytest.mark.unit
def test_worker_response_parse_cap_accommodates_four_thousand_decision_records() -> None:
    records = [
        {"decision": index, "trace": "x" * 1024}
        for index in range(4_000)
    ]
    payload = json.dumps({"decision_latency_trace": records}).encode("utf-8")
    assert len(payload) < runner._WORKER_RESPONSE_PARSE_MAX_BYTES
    capture = runner._WorkerStreamCapture(
        parse_limit_bytes=runner._WORKER_RESPONSE_PARSE_MAX_BYTES
    )
    for offset in range(0, len(payload), 4093):
        capture.feed(payload[offset : offset + 4093])
    capture.finish()
    assert capture.parse_limit_exceeded is False
    assert json.loads(capture.parse_text())["decision_latency_trace"] == records


@pytest.mark.unit
def test_failed_worker_capture_escapes_c1_controls_and_seals_invalid_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _failed_worker_spawn_inputs(
        tmp_path,
        monkeypatch,
        completed=SimpleNamespace(
            returncode=2,
            stdout=b"{}",
            stderr="bad\u009b[31m",
        ),
    )

    with pytest.raises(runner.RTPThreeArmRunnerError, match="isolated worker failed"):
        runner._spawn_worker(**inputs)
    evidence_path = next((inputs["output_root"] / "failed-worker-evidence").glob("*.json"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert "\\u009b" in evidence["stderr"]["captured_text"]
    assert "\u009b" not in evidence["stderr"]["captured_text"]

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    malformed_inputs = {**inputs, "output_root": malformed_root}
    monkeypatch.setattr(
        runner,
        "_capture_worker_process",
        lambda **_kwargs: _captured_worker_process(
            returncode=0, stdout=b"\xffnot-json", stderr=b"\xfe"
        ),
    )
    with pytest.raises(runner.RTPThreeArmRunnerError, match="not strict UTF-8"):
        runner._spawn_worker(**malformed_inputs)
    malformed_path = next((malformed_root / "failed-worker-evidence").glob("*.json"))
    malformed = json.loads(malformed_path.read_text(encoding="utf-8"))
    assert malformed["failure"]["stage"] == "worker_output_decode_failed"
    assert malformed["stdout"]["strict_utf8"] is False
    assert malformed["stderr"]["strict_utf8"] is False
    assert malformed["stdout"]["sha256"] == runner._sha256_bytes(b"\xffnot-json")


@pytest.mark.unit
def test_popen_launch_and_pipe_errors_seal_distinct_failed_worker_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_capture = runner._capture_worker_process
    inputs = _failed_worker_spawn_inputs(
        tmp_path,
        monkeypatch,
        completed=SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )
    monkeypatch.setattr(runner, "_capture_worker_process", real_capture)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic launch")),
    )
    with pytest.raises(runner.RTPThreeArmRunnerError, match="launch/capture failed"):
        runner._spawn_worker(**inputs)
    launch_path = next((inputs["output_root"] / "failed-worker-evidence").glob("*.json"))
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    assert launch["failure"]["stage"] == "worker_subprocess_launch_failed"
    assert launch["returncode"] is None
    assert launch["child"] == {"launched": False, "pid": None}

    io_root = tmp_path / "io-failure"
    io_root.mkdir()
    io_inputs = {**inputs, "output_root": io_root}

    class BrokenRead:
        def read(self, _size: int) -> bytes:
            raise OSError("synthetic pipe read")

        def close(self) -> None:
            return None

    class FakeProcess:
        pid = 777
        returncode: int | None = None
        stdin = io.BytesIO()
        stdout = BrokenRead()
        stderr = io.BytesIO(b"partial stderr")

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    with pytest.raises(runner.RTPThreeArmRunnerError, match="launch/capture failed"):
        runner._spawn_worker(**io_inputs)
    io_path = next((io_root / "failed-worker-evidence").glob("*.json"))
    io_evidence = json.loads(io_path.read_text(encoding="utf-8"))
    assert io_evidence["failure"]["stage"] == "worker_subprocess_io_failed"
    assert io_evidence["returncode"] == -15
    assert io_evidence["child"] == {"launched": True, "pid": 777}


@pytest.mark.unit
def test_preexec_factory_failure_seals_no_request_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _failed_worker_spawn_inputs(
        tmp_path,
        monkeypatch,
        completed=SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )
    monkeypatch.setattr(
        runner,
        "_load_factory",
        lambda _reference: (_ for _ in ()).throw(RuntimeError("factory unavailable")),
    )

    with pytest.raises(runner.RTPThreeArmRunnerError, match="pre-exec setup failed"):
        runner._spawn_worker(**inputs)
    evidence_path = next((inputs["output_root"] / "failed-worker-evidence").glob("*.json"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["failure"]["stage"] == "worker_pre_exec_setup_failed"
    assert evidence["worker_request"] is None
    assert evidence["worker_request_sha256"] is None
    assert evidence["child_launched"] is False
    assert evidence["child_pid"] is None
    assert evidence["child"] == {"launched": False, "pid": None}
    assert "runtime_identity" in evidence["unavailable_fields"]
    assert "source_snapshot_identity" in evidence["unavailable_fields"]


@pytest.mark.unit
@pytest.mark.parametrize("tamper", ("cell", "arm", "manifest", "runtime", "fence"))
def test_failed_worker_evidence_rejects_tampered_parent_request_binding(
    tmp_path: Path, tamper: str
) -> None:
    manifest = _sealed_json(tmp_path, "manifest.json", {"schema": "test"})
    authority = _sealed_json(tmp_path, "authority.json", {"schema": "test"})
    seal = _sealed_json(tmp_path, "seal.json", {"schema": "test"})
    output_root = tmp_path / "output"
    output_root.mkdir()
    cell = {
        "cell_id": "cell-000000",
        "evaluation_case_id": "case-000000",
        "evaluation_case_bindings_sha256": "sha256:case",
        "opponent_id": "iono",
        "candidate_seat": 0,
    }
    request = {
        "schema": runner.WORKER_REQUEST_SCHEMA,
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "factory": "test:Factory",
        "cell": dict(cell),
        "arm": runner.CANONICAL_DIRECT_ARM,
        "snapshot_seal": seal,
        "evaluation_authority": authority,
        "action_fence": None,
        "launch_nonce": "n" * 48,
        "max_steps": 1,
        "environment_identity": {"sha256": "sha256:environment"},
    }
    expected_request = dict(request)
    if tamper == "cell":
        request["cell"] = {**cell, "cell_id": "another-cell"}
    elif tamper == "arm":
        request["arm"] = "recursive_rtp"
    elif tamper == "manifest":
        request["manifest_sha256"] = "sha256:another-manifest"
    elif tamper == "runtime":
        request["environment_identity"] = {"sha256": "sha256:other-environment"}
    else:
        request["action_fence"] = {"sha256": "sha256:other-fence"}
    context = runner._failed_worker_context(
        manifest_identity=manifest,
        factory_ref="test:Factory",
        cell=cell,
        arm=runner.CANONICAL_DIRECT_ARM,
        request=request,
        environment={},
        environment_identity={"sha256": "sha256:environment"},
        capability={},
        closure={},
    )
    with pytest.raises(runner.RTPThreeArmRunnerError, match="failed-worker request"):
        runner._write_failed_worker_evidence(
            output_root=output_root,
            manifest_identity=manifest,
            factory_ref="test:Factory",
            cell=cell,
            arm=runner.CANONICAL_DIRECT_ARM,
            authority_identity=authority,
            snapshot_seal_identity=seal,
            failure_context=context,
            failure_stage="worker_exit_nonzero",
            returncode=2,
            stdout="",
            stderr="",
            error=RuntimeError("synthetic"),
            expected_worker_request=expected_request,
        )
    assert not (output_root / "failed-worker-evidence").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "expected_stage"),
    (
        ({"schema": "wrong-worker-schema"}, "worker_response_schema_invalid"),
        (
            {"schema": runner.WORKER_RESPONSE_SCHEMA, "status": "incomplete"},
            "worker_response_validation_failed",
        ),
    ),
)
def test_invalid_or_failed_worker_response_seals_diagnostic_not_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
    expected_stage: str,
) -> None:
    manifest_path, output, written_rows = _patch_runner_job_inputs(
        tmp_path, monkeypatch
    )
    calls: list[tuple[str, str]] = []

    def spawn(**kwargs: Any) -> tuple[dict[str, Any], str, str]:
        calls.append((kwargs["cell"]["cell_id"], kwargs["arm"]))
        return dict(response), "child stdout", "child stderr"

    monkeypatch.setattr(runner, "_spawn_worker", spawn)

    with pytest.raises(runner.RTPThreeArmRunnerError):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=1,
            max_steps=1,
        )

    evidence_paths = list((output.parent / "failed-worker-evidence").glob("*.json"))
    assert len(evidence_paths) == 1
    evidence = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
    assert evidence["failure"]["stage"] == expected_stage
    assert evidence["returncode"] == 0
    assert evidence["not_an_evaluation_result"] is True
    assert evidence["not_a_result_row"] is True
    assert evidence["not_an_execution_receipt"] is True
    assert evidence["not_a_transcript"] is True
    assert evidence["context_origin"] == "parent_fallback_for_injected_or_legacy_worker_result"
    assert stat.S_IMODE(os.lstat(evidence_paths[0]).st_mode) == 0o444
    assert calls == [("cell-000000", "no_rtp")]
    assert written_rows == []
    assert not output.exists()
    assert not (output.parent / "execution-receipts").exists()
    assert not (output.parent / "transcripts").exists()


@pytest.mark.unit
def test_runner_refuses_to_reuse_a_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results.json"
    output.write_text(json.dumps({"schema": runner.RESULTS_SCHEMA}), encoding="utf-8")
    manifest = {
        "schedule": [{"cell_id": "cell-000000", "opponent_id": "iono"}],
        "opponents": [{"id": "iono"}],
    }
    identity = {"path": str(tmp_path / "manifest.json"), "sha256": "sha256:manifest", "bytes": 1}
    monkeypatch.setattr(runner, "_read_manifest", lambda _path: (manifest, identity))
    monkeypatch.setattr(runner, "_verify_pairing_capability", lambda _manifest: {})
    monkeypatch.setattr(runner, "_verify_evaluation_cg_closure", lambda *_args: {})
    monkeypatch.setattr(runner, "_verify_evaluation_authority", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_verify_cohort_and_source_exclusion", lambda _manifest: {})
    monkeypatch.setattr(runner, "_verify_planner_preflight", lambda _manifest: None)
    monkeypatch.setattr(runner, "_sealed_opponents", lambda _manifest: {"iono": {}})
    monkeypatch.setattr(runner, "_verify_snapshot_package", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_require_physical_readonly_tree", lambda _entry: {})

    with pytest.raises(runner.RTPThreeArmRunnerError, match="fresh no-clobber path"):
        runner.run_three_arm_evaluation(
            manifest_path=tmp_path / "manifest.json",
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
        )


@pytest.mark.unit
def test_execution_attempt_sentinel_blocks_retry_before_any_new_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, output, _written_rows = _patch_runner_job_inputs(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []

    def fail(**kwargs: Any) -> tuple[dict[str, Any], str, str]:
        calls.append((str(kwargs["cell"]["cell_id"]), str(kwargs["arm"])))
        raise runner.RTPThreeArmRunnerError("synthetic first-attempt failure")

    monkeypatch.setattr(runner, "_spawn_worker", fail)
    with pytest.raises(runner.RTPThreeArmRunnerError, match="first-attempt"):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=1,
            max_steps=1,
        )
    attempt = output.parent / "execution-attempt.json"
    payload = json.loads(attempt.read_text(encoding="utf-8"))
    assert payload["schema"] == runner.EXECUTION_ATTEMPT_SCHEMA
    assert payload["schedule"]["job_count"] == 12
    assert stat.S_IMODE(attempt.stat().st_mode) == 0o444
    assert calls == [("cell-000000", "no_rtp")]

    calls.clear()
    with pytest.raises(runner.RTPThreeArmRunnerError, match="execution artifact"):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=1,
            max_steps=1,
        )
    assert calls == []
    assert not output.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_name",
    ("worker-scratch", "transcripts", "execution-receipts", "failed-worker-evidence"),
)
def test_execution_attempt_rejects_any_preexisting_execution_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_name: str
) -> None:
    manifest_path, output, _written_rows = _patch_runner_job_inputs(tmp_path, monkeypatch)
    output.parent.mkdir()
    (output.parent / artifact_name).mkdir()
    calls: list[object] = []
    monkeypatch.setattr(
        runner,
        "_spawn_worker",
        lambda **_kwargs: calls.append(object()) or ({}, "", ""),
    )

    with pytest.raises(runner.RTPThreeArmRunnerError, match="execution artifact"):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=1,
            max_steps=1,
        )
    assert calls == []
    assert not (output.parent / "execution-attempt.json").exists()


@pytest.mark.unit
def test_execution_attempt_permits_only_the_canonical_precreated_input_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, output, _written_rows = _patch_runner_job_inputs(tmp_path, monkeypatch)
    output.parent.mkdir()
    contract = {
        "schema": "poke_bot.alakazam_rtp_r198_three_arm_evaluation_stage/v1",
        "stage_kind": "three_arm_true_rng_evaluation",
        "evaluation_inputs": {
            "prepared_evaluator_manifest": {
                "path": str(manifest_path),
                "sha256": "sha256:bounded-manifest",
                "bytes": manifest_path.stat().st_size,
            },
            "evaluation_only_authority": {},
        },
        "authority": {
            "training_eligible": False,
            "replay_eligible": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "selector_authority": False,
            "kaggle_submission_authorized": False,
            "promotion_authority": False,
            "self_promotion_allowed": False,
        },
    }
    contract_path = output.parent / "evaluation-input-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_path.chmod(0o444)
    calls: list[tuple[str, str]] = []

    def fail_after_sentinel(**kwargs: Any) -> tuple[dict[str, Any], str, str]:
        calls.append((str(kwargs["cell"]["cell_id"]), str(kwargs["arm"])))
        raise runner.RTPThreeArmRunnerError("synthetic post-sentinel failure")

    monkeypatch.setattr(runner, "_spawn_worker", fail_after_sentinel)
    with pytest.raises(runner.RTPThreeArmRunnerError, match="post-sentinel"):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=1,
            max_steps=1,
        )
    assert calls == [("cell-000000", "no_rtp")]
    assert (output.parent / "execution-attempt.json").is_file()


@pytest.mark.unit
def test_evaluation_authority_path_is_rehashed_as_a_strict_sealed_identity(
    tmp_path: Path,
) -> None:
    manifest = _sealed_json(tmp_path, "manifest.json", {"schema": "test"})
    authority = _sealed_json(
        tmp_path,
        "authority.json",
        _evaluation_authority_payload(manifest["sha256"]),
    )

    checked = runner._verify_evaluation_authority(
        Path(authority["path"]), manifest_identity=manifest
    )

    assert checked["identity"] == authority
    assert checked["payload"]["manifest_sha256"] == manifest["sha256"]

    with pytest.raises(
        runner.RTPThreeArmRunnerError, match="bound to this manifest"
    ):
        runner._verify_evaluation_authority(
            authority["path"],
            manifest_identity={"sha256": "sha256:" + "f" * 64},
        )

    writable = tmp_path / "writable-authority.json"
    writable.write_text(
        json.dumps(_evaluation_authority_payload(manifest["sha256"])), encoding="utf-8"
    )
    os.chmod(writable, 0o644)
    with pytest.raises(
        runner.RTPThreeArmRunnerError, match="read-only immutable evidence"
    ):
        runner._verify_evaluation_authority(writable, manifest_identity=manifest)

    owner_only = tmp_path / "owner-only-authority.json"
    owner_only.write_text(
        json.dumps(_evaluation_authority_payload(manifest["sha256"])), encoding="utf-8"
    )
    os.chmod(owner_only, 0o400)
    with pytest.raises(runner.RTPThreeArmRunnerError, match="immutable mode 0444"):
        runner._verify_evaluation_authority(owner_only, manifest_identity=manifest)

    linked = tmp_path / "linked-authority.json"
    linked.symlink_to(authority["path"])
    with pytest.raises(runner.RTPThreeArmRunnerError, match="symbolic link"):
        runner._verify_evaluation_authority(linked, manifest_identity=manifest)


@pytest.mark.unit
def test_worker_environment_rehashes_its_sealed_path_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        environment,
        capability,
        closure,
        cg_runtime,
        source_manifest,
    ) = _worker_environment_fixture(tmp_path, monkeypatch)
    try:
        checked = runner._verify_worker_environment_bindings(
            environment, capability=capability, closure=closure
        )

        assert checked["candidate_runtime_contract_sha256"] == environment[
            "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256"
        ]
        assert checked["source_snapshot_manifest_sha256"] == runner._identity(
            source_manifest, "test source manifest"
        )["sha256"]

        linked_contract = tmp_path / "linked-runtime-contract.json"
        linked_contract.symlink_to(environment["POKEBOT_R198_EVAL_RUNTIME_CONTRACT"])
        with pytest.raises(runner.RTPThreeArmRunnerError, match="symbolic link"):
            runner._verify_worker_environment_bindings(
                {
                    **environment,
                    "POKEBOT_R198_EVAL_RUNTIME_CONTRACT": str(linked_contract),
                },
                capability=capability,
                closure=closure,
            )
    finally:
        # Restore directory permissions so the temporary fixture can be
        # removed normally even if an assertion above fails.
        os.chmod(cg_runtime / "cg", 0o755)
        os.chmod(cg_runtime, 0o755)


@pytest.mark.unit
def test_worker_rejects_fresh_authority_identity_that_differs_from_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _sealed_json(tmp_path, "manifest.json", {"schema": "test"})
    authority = _sealed_json(
        tmp_path,
        "authority.json",
        _evaluation_authority_payload(manifest["sha256"]),
    )
    monkeypatch.setattr(runner, "_read_manifest", lambda _path: ({}, manifest))

    def mismatched_fresh_identity(path: str | Path, label: str) -> dict[str, Any]:
        assert path == authority["path"]
        assert label == "evaluation-only authority"
        return {**authority, "sha256": "sha256:" + "0" * 64}

    monkeypatch.setattr(runner, "_sealed_path_identity", mismatched_fresh_identity)
    request = {
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "factory": "test:Factory",
        "arm": "no_rtp",
        "cell": {},
        "snapshot_seal": {},
        "evaluation_authority": authority,
        "action_fence": None,
        "launch_nonce": "n" * 48,
    }

    with pytest.raises(
        runner.RTPThreeArmRunnerError,
        match="worker authority identity differs from its request",
    ):
        runner._worker_response(request=request)


@pytest.mark.unit
@pytest.mark.parametrize("failure_stage", ("worker", "validation"))
def test_runner_max_workers_one_stops_after_exactly_one_failed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    manifest_path, output, written_rows = _patch_runner_job_inputs(
        tmp_path, monkeypatch
    )
    calls: list[tuple[str, str]] = []

    def spawn(**kwargs: Any) -> tuple[dict[str, Any], str, str]:
        calls.append((kwargs["cell"]["cell_id"], kwargs["arm"]))
        if failure_stage == "worker":
            raise runner.RTPThreeArmRunnerError("diagnostic worker failure")
        return {}, "", ""

    monkeypatch.setattr(runner, "_spawn_worker", spawn)
    if failure_stage == "validation":
        monkeypatch.setattr(
            runner,
            "_validate_worker_response",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                runner.RTPThreeArmRunnerError("diagnostic validation failure")
            ),
        )

    with pytest.raises(runner.RTPThreeArmRunnerError, match="diagnostic"):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=1,
            max_steps=1,
        )

    assert calls == [("cell-000000", "no_rtp")]
    assert written_rows == []
    assert not output.exists()


@pytest.mark.unit
def test_runner_initial_failed_wave_never_exceeds_max_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, output, written_rows = _patch_runner_job_inputs(
        tmp_path, monkeypatch
    )
    calls: list[tuple[str, str]] = []

    def fail(**kwargs: Any) -> tuple[dict[str, Any], str, str]:
        calls.append((kwargs["cell"]["cell_id"], kwargs["arm"]))
        raise runner.RTPThreeArmRunnerError("bounded worker failure")

    monkeypatch.setattr(runner, "_spawn_worker", fail)

    with pytest.raises(runner.RTPThreeArmRunnerError, match="bounded worker failure"):
        runner.run_three_arm_evaluation(
            manifest_path=manifest_path,
            evaluation_authority_path=tmp_path / "authority.json",
            factory="test:Factory",
            output_path=output,
            max_workers=3,
            max_steps=1,
        )

    assert 1 <= len(calls) <= 3
    assert written_rows == []
    assert not output.exists()


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", (1, 2))
def test_runner_bounded_scheduler_refills_until_every_job_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_workers: int
) -> None:
    manifest_path, output, written_rows = _patch_runner_job_inputs(
        tmp_path, monkeypatch, cell_count=2
    )
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    initial_wave = threading.Barrier(max_workers)
    active_workers = 0
    peak_workers = 0

    def spawn(**kwargs: Any) -> tuple[dict[str, Any], str, str]:
        nonlocal active_workers, peak_workers
        cell_id = str(kwargs["cell"]["cell_id"])
        arm = str(kwargs["arm"])
        with lock:
            initial_job = len(calls) < max_workers
            calls.append((cell_id, arm))
            active_workers += 1
            peak_workers = max(peak_workers, active_workers)
        try:
            if initial_job:
                initial_wave.wait(timeout=2)
            return {"cell_id": cell_id, "arm": arm}, "", ""
        finally:
            with lock:
                active_workers -= 1

    def validate(
        _response: dict[str, Any],
        *,
        cell: dict[str, Any],
        arm: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        cell_id = str(cell["cell_id"])
        return {
            "cell_id": cell_id,
            "arm": arm,
            "opponent_id": cell["opponent_id"],
            "candidate_seat": cell["candidate_seat"],
            "candidate_score": 0.5,
            "terminal_outcome": {"winner": "draw"},
            "runtime_identity": {},
            "telemetry": {},
            "isolation": {
                "process_id": f"pid-{cell_id}-{arm}",
                "launch_nonce": f"nonce-{cell_id}-{arm}",
            },
        }

    def write_row(**kwargs: Any) -> dict[str, Any]:
        row = dict(kwargs["row"])
        written_rows.append(row)
        return row

    monkeypatch.setattr(runner, "_spawn_worker", spawn)
    monkeypatch.setattr(runner, "_validate_worker_response", validate)
    monkeypatch.setattr(runner, "_write_row_evidence", write_row)
    monkeypatch.setattr(
        runner,
        "_verify_cohort_and_source_exclusion",
        lambda _manifest: {
            "cohort": {"sha256": "sha256:cohort"},
            "source_exclusion_proof": {"sha256": "sha256:exclusion"},
        },
    )

    result = runner.run_three_arm_evaluation(
        manifest_path=manifest_path,
        evaluation_authority_path=tmp_path / "authority.json",
        factory="test:Factory",
        output_path=output,
        max_workers=max_workers,
        max_steps=1,
    )

    expected_jobs = {
        (f"cell-{index:06d}", arm)
        for index in range(2)
        for arm in ("no_rtp", runner.CANONICAL_DIRECT_ARM, "recursive_rtp")
    }
    assert result == output
    assert output.is_file()
    attempt = output.parent / "execution-attempt.json"
    assert attempt.is_file()
    assert stat.S_IMODE(attempt.stat().st_mode) == 0o444
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == len(expected_jobs)
    assert set(calls) == expected_jobs
    assert len(written_rows) == len(expected_jobs)
    assert len(payload["rows"]) == len(expected_jobs)
    assert {
        (str(row["cell_id"]), str(row["arm"])) for row in payload["rows"]
    } == expected_jobs
    assert peak_workers == max_workers
