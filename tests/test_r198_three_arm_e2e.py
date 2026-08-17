"""Hermetic process-boundary smoke coverage for the r198 evaluator.

The neighboring evaluator-contract test materializes and compiles the full
1,000-cell panel.  This test owns the missing boundary: one selected material
is sent through the real runner ``_spawn_worker`` protocol for all three arms.
The worker executable is deliberately a tiny sealed-engine stand-in, so the
test exercises fresh ``exec`` processes and immutable evidence without a GPU,
private engine, production source snapshot, selector, or Kaggle access.
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import poke_bot.rtp_three_arm_evaluation_runner as runner
from poke_bot.rtp_three_arm_evaluation import compile_three_arm_receipt
from poke_bot.rtp_r198_evaluation_input_materializer import R198EvaluationInputError

# This is intentionally test-only reuse of the file-backed, full-size fixture.
# It keeps this process-boundary test narrow while still starting from the
# materializer's canonical 1,000 snapshot/case-binding panel.
import test_rtp_three_arm_evaluation as contract_fixture


_ARMS = ("no_rtp", runner.CANONICAL_DIRECT_ARM, "recursive_rtp")


def _snapshot_local_materialized_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path]:
    """Run the full test materializer using a 0444 snapshot-local receipt.

    Production's six-artifact candidate boundary includes the r197 completion
    receipt inside the immutable source snapshot.  The shared full-panel
    fixture intentionally stays lightweight, so wrap only its materializer
    call here to ensure this end-to-end boundary never quietly regresses to a
    live or writable receipt path.
    """

    real_materialize = contract_fixture.materialize_r198_evaluation_inputs
    captured_call: dict[str, Any] = {}
    snapshot_receipt = (
        tmp_path
        / "source-snapshot"
        / "evaluation-artifacts"
        / "r197-candidate"
        / "r197-completion-receipt.json"
    )

    def materialize_from_snapshot(**kwargs: Any) -> dict[str, Any]:
        source = Path(kwargs["completion_receipt"])
        snapshot_receipt.parent.mkdir(parents=True, exist_ok=True)
        snapshot_receipt.write_bytes(source.read_bytes())
        snapshot_receipt.chmod(0o444)
        rewritten = {**kwargs, "completion_receipt": snapshot_receipt}
        captured_call.update(rewritten)
        return real_materialize(**rewritten)

    monkeypatch.setattr(
        contract_fixture,
        "materialize_r198_evaluation_inputs",
        materialize_from_snapshot,
    )
    fixture = contract_fixture._materialized_fixture(tmp_path, monkeypatch)
    assert stat.S_IMODE(snapshot_receipt.stat().st_mode) == 0o444
    assert fixture["manifest"]["shared_artifacts"]["r197_completion_receipt"][
        "path"
    ] == str(snapshot_receipt)

    # The content-addressed target is intentionally a fresh, no-clobber root.
    # Reusing the exact request must fail before another cell snapshot can be
    # captured or an existing immutable record can be overwritten.
    with pytest.raises(R198EvaluationInputError, match="refusing to reuse"):
        real_materialize(**captured_call)
    return fixture, snapshot_receipt


def _runtime_identity(manifest: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arm_spec = manifest["arms"][arm]
    profile = arm_spec["profile"]
    action_sidecar = arm_spec["rtp_sidecar"]
    return {
        "arm": arm,
        "runtime_artifact_sha256": arm_spec["runtime_artifact"]["sha256"],
        "runtime_profile_sha256": arm_spec["runtime_profile"]["sha256"],
        "action_attached_rtp_sidecar_sha256": (
            None if action_sidecar is None else action_sidecar["sha256"]
        ),
        "complexity_probe_sidecar_sha256": manifest["arms"][
            runner.CANONICAL_DIRECT_ARM
        ]["rtp_sidecar"]["sha256"],
        "complexity_probe_sidecar_instrumentation_only": True,
        "complexity_probe_latency_excluded": True,
        "rtp_action_attachment_enabled": arm != "no_rtp",
        "rtp_action_authority_enabled": False,
        **{
            f"{name}_sha256": identity["sha256"]
            for name, identity in manifest["shared_artifacts"].items()
        },
        "recursive_turn_planner_enabled": profile["recursive_turn_planner_enabled"],
        "direct_bridge_enabled": profile["direct_bridge_enabled"],
        "force_direct_bridge_only": profile["force_direct_bridge_only"],
        "max_neural_passes": profile["max_neural_passes"],
        "max_action_combos": profile["max_action_combos"],
    }


def _child_template(
    *, manifest: Mapping[str, Any], cell: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    """Build the response shape emitted by a one-turn fake sealed engine."""

    opponent = next(
        item for item in manifest["opponents"] if item["id"] == cell["opponent_id"]
    )
    closure = manifest["evaluation_cg_closure"]
    runtime_library = closure["runtime_library"]
    recursive = arm == "recursive_rtp"
    if arm == "no_rtp":
        mode, planner_reason, latency = "no_rtp", "baseline", 0.001
    elif arm == runner.CANONICAL_DIRECT_ARM:
        mode, planner_reason, latency = "direct_bridge", "bridge_only", 0.002
    else:
        mode, planner_reason, latency = "recursive_plan", "complex", 0.003
    winner = "candidate" if recursive else "opponent"
    candidate_seat = int(cell["candidate_seat"])
    result_code = candidate_seat if winner == "candidate" else 1 - candidate_seat
    tag = f"{cell['cell_id']}:{arm}"
    common = contract_fixture._sha(f"e2e-common-rng:{cell['cell_id']}")
    return {
        "schema": runner.WORKER_RESPONSE_SCHEMA,
        "status": "completed",
        "cell_id": cell["cell_id"],
        "arm": arm,
        "opponent_id": cell["opponent_id"],
        "candidate_seat": candidate_seat,
        "candidate_score": 1.0 if winner == "candidate" else 0.0,
        "terminal_outcome": {
            "winner": winner,
            "engine_result_code": result_code,
            "candidate_forfeit": False,
            "termination": "completed",
            "failed_seat": None,
            "engine_error": None,
            "candidate_error": None,
            "opponent_error": None,
        },
        # Exercise the exact n=9/counts=1..5 special stratum through real
        # child transport, immutable transcript evidence, and full-panel
        # compiler aggregation.  The remaining 999 cells retain the normal
        # fixture telemetry, so ordinary bridge/recursive gates stay covered.
        "telemetry": contract_fixture._over_cap_telemetry(
            arm, candidate_seat=candidate_seat
        ),
        "runtime_identity": _runtime_identity(manifest, arm),
        "native_transcript_events": [
            {
                "event": "restore_sealed_snapshot_manifest",
                "snapshot_seal_sha256": "__REQUEST_SEAL_SHA256__",
                "loaded_dso_path": runtime_library["path"],
            },
            {"event": "first_select", "actor": "candidate"},
        ],
        "isolation": {
            "launch_mode": "subprocess_exec",
            "fresh_process_per_arm": True,
            "one_cell_one_arm": True,
            "pool_reuse": False,
            "forked_from_evaluator": False,
            "process_model_load": True,
            "fresh_candidate_agent": True,
            "candidate_reset_called": True,
            "fresh_opponent_module": True,
            "engine_restore_before_first_select": True,
            "engine_restore_count": 1,
            "battle_start_after_restore_count": 0,
            "no_remote_leaf_sampling_mcts": True,
            "complexity_probe_latency_excluded": True,
            "package_snapshot_verified_before_import": True,
            "baseline_content_digest": opponent["content_digest"],
            "baseline_package_root": opponent["package_root"],
            "baseline_tree_entries_sha256": opponent["tree_entries_sha256"],
            "baseline_package_manifest_sha256": opponent["artifact"]["sha256"],
            "baseline_deck_sha256": opponent["deck_sha256"],
            "candidate_rng_initial_state_sha256": common,
            "opponent_rng_initial_state_sha256": common,
            "opponent_rng_deterministic_or_no_rng": False,
            "python_rng_initial_state_sha256": common,
            "numpy_rng_initial_state_sha256": common,
            "torch_cpu_rng_initial_state_sha256": common,
            "torch_cuda_rng_initial_state_sha256": common,
            "common_sanitized_environment_sha256": common,
            "arm_environment_sha256": contract_fixture._sha(f"e2e-arm-env:{tag}"),
            "evaluation_cg_closure_receipt_sha256": closure["receipt"]["sha256"],
            "evaluation_cg_engine_sha256": runtime_library["sha256"],
            "evaluation_cg_engine_path": runtime_library["path"],
            "evaluation_cg_engine_bytes": str(runtime_library["bytes"]),
            "evaluation_cg_closure_manifest_sha256": closure["closure_manifest"][
                "sha256"
            ],
            "evaluation_cg_metadata_parity_sha256": closure["metadata_parity"][
                "sha256"
            ],
            "engine_loaded_path": runtime_library["path"],
            "candidate_runtime_contract_sha256": contract_fixture._sha(
                f"e2e-candidate-runtime:{tag}"
            ),
            "action_fence_sha256": "__ACTION_FENCE_SHA256__",
            "evaluation_action_execution_sha256": "__ACTION_CONTEXT_SHA256__",
            "process_id": "__PID__",
            "boot_id": "hermetic-e2e",
            "process_start_ticks": "__START_TICKS__",
            "launch_nonce": "__LAUNCH_NONCE__",
        },
    }


_FAKE_WORKER = r"""
import hashlib
import json
import os
import sys

template = json.loads(sys.argv[1])
request = json.load(sys.stdin)
if request.get("schema") != "poke_bot.recursive_turn_planner.three_arm_evaluation_worker_request/v1":
    raise SystemExit("unexpected worker request schema")
if "requested_seed" in request or "snapshot_artifact" in request:
    raise SystemExit("runner sent seed or raw snapshot instead of a sealed manifest")
seal = request.get("snapshot_seal")
if not isinstance(seal, dict) or seal.get("sha256") != template.pop("_expected_seal_sha256"):
    raise SystemExit("worker received another sealed snapshot")
if request.get("arm") != template.get("arm"):
    raise SystemExit("worker arm changed")
isolation = template["isolation"]
isolation["process_id"] = str(os.getpid())
isolation["process_start_ticks"] = str(os.getpid())
isolation["launch_nonce"] = request["launch_nonce"]
if request["arm"] == "no_rtp":
    if request.get("action_fence") is not None:
        raise SystemExit("no-RTP child received an evaluator action fence")
    isolation["action_fence_sha256"] = None
    isolation["evaluation_action_execution_sha256"] = None
else:
    fence = request.get("action_fence")
    if not isinstance(fence, dict) or not fence.get("sha256"):
        raise SystemExit("RTP child lacks its evaluator action fence")
    isolation["action_fence_sha256"] = fence["sha256"]
    isolation["evaluation_action_execution_sha256"] = "sha256:" + hashlib.sha256(
        (request["launch_nonce"] + request["arm"]).encode("utf-8")
    ).hexdigest()
template["native_transcript_events"][0]["snapshot_seal_sha256"] = seal["sha256"]
print(json.dumps(template, sort_keys=True))
"""


@pytest.mark.integration
def test_r198_materialized_seal_runs_three_fresh_children_and_compiles_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise materializer -> sealed child requests -> immutable hold receipt.

    A full canonical panel is produced by the real materializer.  Only the
    first cell uses the fake child engine; its three evidence rows replace the
    equivalent rows in the full, file-backed hermetic result fixture before
    the real compiler verifies and emits a hold.
    """

    fixture, snapshot_completion_receipt = _snapshot_local_materialized_fixture(
        tmp_path, monkeypatch
    )
    manifest = fixture["manifest"]
    manifest_path = fixture["manifest_path"]
    materialized = fixture["materialized"]
    assert len(manifest["schedule"]) == 1_000
    assert manifest["evaluation_cg_closure"]["runtime_library"]["path"] != manifest[
        "evaluation_cg_closure"
    ]["engine_artifact"]["path"]
    assert (
        manifest["evaluation_cg_closure"]["runtime_library"]["sha256"]
        == manifest["evaluation_cg_closure"]["engine_artifact"]["sha256"]
    )

    authority_identity = materialized["evaluation_only_authority"]
    assert stat.S_IMODE(snapshot_completion_receipt.stat().st_mode) == 0o444
    authority_payload = json.loads(Path(authority_identity["path"]).read_text())
    for field in (
        "training_eligible",
        "replay_eligible",
        "serving_change_authorized",
        "selector_change_authorized",
        "action_authority_authorized",
        "kaggle_submission_authorized",
    ):
        assert authority_payload[field] is False

    cell = manifest["schedule"][0]
    material = next(
        item
        for item in manifest["rng_materials"]
        if item["id"] == cell["rng_identity"]["id"]
    )
    seal = json.loads(Path(material["seal"]["path"]).read_text())
    binding = json.loads(Path(seal["case_binding_artifact"]["path"]).read_text())
    assert seal["snapshot_id"] == material["id"]
    assert seal["snapshot_artifact_sha256"] == material["snapshot_artifact"]["sha256"]
    assert seal["requested_seed_is_pairing_proof"] is False
    for field, expected in (
        ("cell_id", cell["cell_id"]),
        ("case_id", cell["evaluation_case_id"]),
        ("opponent_id", cell["opponent_id"]),
        ("candidate_seat", cell["candidate_seat"]),
        ("replicate", cell["replicate"]),
    ):
        assert binding[field] == expected
    assert binding["action_authority_enabled"] is False
    assert binding["promotion_eligible"] is False
    assert binding["kaggle_submission_authorized"] is False

    observed_requests: list[dict[str, Any]] = []
    observed_factory_calls: list[dict[str, Any]] = []

    class Factory:
        def worker_environment(self, **kwargs: Any) -> dict[str, str]:
            observed_factory_calls.append(dict(kwargs))
            return {}

    real_popen = subprocess.Popen

    def fake_argv(request: Mapping[str, Any]) -> list[str]:
        observed_requests.append(dict(request))
        assert request["snapshot_seal"] == material["seal"]
        assert "requested_seed" not in request
        assert "snapshot_artifact" not in request
        template = _child_template(manifest=manifest, cell=cell, arm=str(request["arm"]))
        template["_expected_seal_sha256"] = material["seal"]["sha256"]
        return [
            sys.executable,
            "-c",
            _FAKE_WORKER,
            json.dumps(template, sort_keys=True),
        ]

    def inspected_popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert argv
        assert not {
            "POKEBOT_RTP_SERVING_QUALIFIED",
            "POKEBOT_RTP_PROMOTION_RECEIPT",
            "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256",
        }.intersection(kwargs["env"])
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(runner, "_load_factory", lambda _reference: Factory())
    monkeypatch.setattr(
        runner,
        "_sanitize_environment",
        lambda _raw, _scratch, **_kwargs: (
            {},
            {
                "common_sanitized_environment_sha256": contract_fixture._sha(
                    f"e2e-common-rng:{cell['cell_id']}"
                ),
                "arm_environment_sha256": contract_fixture._sha("e2e-parent-env"),
            },
        ),
    )
    monkeypatch.setattr(runner, "_worker_subprocess_argv", fake_argv)
    monkeypatch.setattr(runner.subprocess, "Popen", inspected_popen)

    output_root = tmp_path / "fresh-worker-evidence"
    responses: list[dict[str, Any]] = []
    child_rows: dict[str, dict[str, Any]] = {}
    manifest_identity = runner._identity(manifest_path, "e2e manifest")
    cohort_identity = manifest["r197_source_exclusion_binding"][
        "evaluation_only_cohort"
    ]
    for arm in _ARMS:
        response, stdout, stderr = runner._spawn_worker(
            manifest_path=str(manifest_path),
            manifest_identity=manifest_identity,
            factory_ref="hermetic:Factory",
            cell=cell,
            arm=arm,
            snapshot={"seal": material["seal"]},
            capability={},
            closure={},
            authority={"identity": authority_identity},
            output_root=output_root,
            max_steps=1,
        )
        responses.append(response)
        assert response["isolation"]["engine_loaded_path"] == manifest[
            "evaluation_cg_closure"
        ]["runtime_library"]["path"]
        assert response["isolation"]["engine_loaded_path"] != manifest[
            "evaluation_cg_closure"
        ]["engine_artifact"]["path"]
        events = response["native_transcript_events"]
        assert [event["event"] for event in events] == [
            "restore_sealed_snapshot_manifest",
            "first_select",
        ]
        assert events[0]["snapshot_seal_sha256"] == material["seal"]["sha256"]
        assert events[0]["loaded_dso_path"] == manifest["evaluation_cg_closure"][
            "runtime_library"
        ]["path"]
        row = {
            "cell_id": response["cell_id"],
            "arm": response["arm"],
            "opponent_id": response["opponent_id"],
            "candidate_seat": response["candidate_seat"],
            "evaluation_case_id": cell["evaluation_case_id"],
            "evaluation_case_bindings_sha256": cell[
                "evaluation_case_bindings_sha256"
            ],
            "completed": True,
            "invalid": False,
            "error": None,
            "candidate_score": response["candidate_score"],
            "terminal_outcome": response["terminal_outcome"],
            "runtime_identity": response["runtime_identity"],
            "rng_identity": {**cell["rng_identity"], "restored_or_replayed": True},
            "telemetry": response["telemetry"],
            "isolation": response["isolation"],
        }
        child_rows[arm] = runner._write_row_evidence(
            output_root=output_root,
            manifest_identity=manifest_identity,
            row=row,
            stdout=stdout,
            stderr=stderr,
            cohort_identity=cohort_identity,
        )

    assert len(observed_factory_calls) == 3
    assert len(observed_requests) == 3
    assert {item["snapshot_seal"]["sha256"] for item in observed_requests} == {
        material["seal"]["sha256"]
    }
    assert observed_requests[0]["action_fence"] is None
    assert all(request["action_fence"] is not None for request in observed_requests[1:])
    assert len({response["isolation"]["process_id"] for response in responses}) == 3
    assert len({response["isolation"]["launch_nonce"] for response in responses}) == 3
    expected_forced_trace = (
        contract_fixture._forced_turn_order_trace()
        if int(cell["candidate_seat"]) == 0
        else []
    )
    assert {
        response["telemetry"]["forced_turn_order_controls"] for response in responses
    } == {int(int(cell["candidate_seat"]) == 0)}
    assert all(
        response["telemetry"]["forced_turn_order_control_trace"]
        == expected_forced_trace
        for response in responses
    )
    assert all(
        len(response["telemetry"]["decision_latency_trace"])
        == response["telemetry"]["candidate_decisions"]
        for response in responses
    )
    assert all(
        response["telemetry"]["planner_eligible_candidate_decisions"] == 0
        and response["telemetry"]["over_cap_factorized_fallback_decisions"] == 1
        and response["telemetry"]["over_cap_factorized_fallback_trace"][0][
            "action_space"
        ]["complete_ordered_action_cardinality"]
        == 18_729
        and response["telemetry"]["over_cap_factorized_fallback_trace"][0][
            "returned_action"
        ]
        == [0, 2, 3, 6, 5]
        for response in responses
    )
    for arm, child_row in child_rows.items():
        transcript = json.loads(Path(child_row["transcript"]["path"]).read_text())
        assert transcript["over_cap_factorized_fallback_trace"] == child_row[
            "telemetry"
        ]["over_cap_factorized_fallback_trace"]
    runner._cross_arm_isolation(responses)

    for request in observed_requests[1:]:
        fence_path = Path(request["action_fence"]["path"])
        fence = json.loads(fence_path.read_text())
        assert fence["evaluation_only"] is True
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
            assert fence[field] is False

    # Compile a full-size hold with the real three subprocess rows substituted
    # for the selected cell.  The companion fixture supplies the remaining
    # file-backed rows so the compiler can enforce exact 3,000-row coverage.
    shaped_results_path = contract_fixture._run_shaped_results(tmp_path, manifest)
    shaped = json.loads(shaped_results_path.read_text())
    shaped["rows"] = [
        child_rows[row["arm"]]
        if row["cell_id"] == cell["cell_id"]
        else row
        for row in shaped["rows"]
    ]
    results_path = contract_fixture._write(
        tmp_path / "e2e-combined-results.json", {"rows": shaped["rows"]}
    )
    receipt_path = compile_three_arm_receipt(
        manifest_path=manifest_path,
        results=results_path,
        output_path=tmp_path / "e2e-hold-receipt.json",
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "hold"
    assert receipt["promotion_decision"]["self_promotion_performed"] is False
    assert receipt["promotion_decision"]["serving_change_authorized"] is False
    assert receipt["promotion_gates"]["immutable_file_backed_execution_evidence"][
        "passed"
    ] is True
    assert receipt["promotion_gates"]["trusted_counterfactual_candidate_targets"][
        "passed"
    ] is False
