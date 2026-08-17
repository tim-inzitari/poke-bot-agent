"""Focused CPU-only fail-closed coverage for the R240/R246 route converter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_r235_r246_exact_stage_probe.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "r235_r246_exact_stage_probe_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module()


def _complete_outcome(payload: object) -> SimpleNamespace:
    """A completed exact-child result; no process, GPU, or network is used."""

    return SimpleNamespace(
        completed=True,
        status="completed",
        stdout=json.dumps(payload, sort_keys=True).encode("utf-8"),
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    stage = tmp_path / "sealed-stage"
    stage.mkdir()
    manifest = stage / "member-manifest.json"
    manifest.write_text('{"members": {}}\n', encoding="utf-8")
    archive = tmp_path / "candidate.tar.gz"
    archive.write_bytes(b"candidate")
    r225 = tmp_path / "r225.json"
    r225.write_text("{}\n", encoding="utf-8")
    r236 = tmp_path / "r236.json"
    r236.write_text("{}\n", encoding="utf-8")
    return {
        "stage": stage.resolve(),
        "archive": archive.resolve(),
        "manifest": manifest.resolve(),
        "r225": r225.resolve(),
        "r236": r236.resolve(),
    }


def _binding_identity(paths: dict[str, Path]) -> dict[str, object]:
    """A typed identity fixture, deliberately avoiding the real archive loader."""

    return {
        "common_identity": {
            "candidate_archive_sha256": "sha256:fixture-archive",
            "r225_contract_sha256": "sha256:fixture-r225",
            "linux_x86_64_libcg_sha256": "sha256:fixture-r236-linux",
        },
        "exact_package": {
            "stage": str(paths["stage"]),
            "archive": str(paths["archive"]),
            "member_manifest": str(paths["manifest"]),
            "r225_contract": str(paths["r225"]),
            "r236_contract": str(paths["r236"]),
        },
        "stage_contract": {
            "stage_tree_sha256": "sha256:fixture-stage",
            "member_count": 1,
        },
    }


def _raw_smoke(
    paths: dict[str, Path],
    binding_identity: dict[str, object],
    *,
    decision_markers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    markers = list(decision_markers or [])
    callbacks: list[dict[str, object]] = [
        {
            "stock_action_accepted": True,
            "callback_elapsed_seconds": 0.01,
        }
    ]
    callbacks.extend(
        {
            "stock_action_accepted": True,
            "callback_elapsed_seconds": 0.01,
            "decision_marker_or_containment": marker,
            "action": marker.get("selected_action"),
        }
        for marker in markers
    )
    return {
        "schema": RUNNER.RAW_SMOKE_SCHEMA,
        "status": "passed",
        "failure": None,
        "package_mutation_check": {"unchanged": True},
        "stock_game": {"terminal": {"physical_terminal_confirmed": True}},
        "full_game_success_markers": [{"full_game": True}],
        "hard_failure_markers": [],
        "degraded_fallback_markers": [],
        "callbacks": callbacks,
        "decision_markers": markers,
        "exact_package_identity": dict(binding_identity["exact_package"]),
        "stage_contract": dict(binding_identity["stage_contract"]),
        "cuda_observations": {"parent": [], "child": []},
        "elapsed_seconds": 1.0,
        "process_observation": {},
    }


def _smoke_result(raw: dict[str, object]) -> Any:
    return RUNNER.ScenarioResult(
        name="physical_stock_full_game",
        payload=raw,
        outcome=_complete_outcome(raw),
    )


def _high_marker() -> dict[str, object]:
    return {
        "mode": "high_confidence_frozen_direct",
        "selected_action": [0],
        "direct_action_precomputed_and_validated": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "mcts_search_call_count": 0,
        "mcts_model_call_count": 0,
        "mcts_simulator_call_count": 0,
        "history_only_existing_child_journal_count": 0,
        "degraded": False,
        "parent_action_elapsed_seconds": 0.01,
    }


def _ordinary_mcts_marker() -> dict[str, object]:
    return {
        "mode": "shared_tree_mcts",
        "stop_reason": "adaptive_early_stop",
        "selected_action": [0],
        "broker_started": True,
        "mcts_child_started": True,
        "mcts_child_called": True,
        "mcts_action_authority": True,
        "both_lanes_progressed": True,
        "deterministic_root_leader_observations": 3,
        "child_search_elapsed_seconds": 0.01,
        "parent_action_elapsed_seconds": 0.02,
    }


def _build_probe(
    paths: dict[str, Path],
    raw: dict[str, object],
    *,
    scenarios: list[Any] | None = None,
) -> dict[str, object]:
    binding_identity = _binding_identity(paths)
    return RUNNER.build_probe_from_actual_scenarios(
        smoke_result=_smoke_result(raw),
        scenario_results=list(scenarios or []),
        stage=paths["stage"],
        archive=paths["archive"],
        manifest=paths["manifest"],
        r225=paths["r225"],
        r236=paths["r236"],
        phase1_full_game_budget_seconds=60.0,
        binding_identity=binding_identity,
    )


def test_raw_smoke_without_high_direct_route_cannot_build_a_probe(tmp_path: Path) -> None:
    """A physical full game cannot be relabelled as missing direct-route evidence."""

    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="no high-confidence direct witness was observed",
    ):
        _build_probe(paths, _raw_smoke(paths, binding_identity))


def test_raw_smoke_without_r246_terminal_route_cannot_build_a_probe(tmp_path: Path) -> None:
    """Ordinary high/MCTS telemetry cannot stand in for a terminal-win proof."""

    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    raw = _raw_smoke(
        paths,
        binding_identity,
        decision_markers=[_high_marker(), _ordinary_mcts_marker()],
    )
    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="no R246 stock terminal-win witness was observed",
    ):
        _build_probe(paths, raw)


def test_actual_terminal_projection_never_invents_marker_facts_or_elapsed_aliases() -> None:
    """A literal marker may change route spelling, never gain evidence."""

    projected = RUNNER._normalize_actual_stock_mcts_witness(
        {
            "mode": "shared_tree_mcts",
            "stop_reason": "proven_deterministic_terminal_win_this_turn",
            "mcts_child_call_count": 1,
            "elapsed_seconds": 0.01,
            "completed_backups": 1,
        },
        label="literal terminal marker",
    )

    # The actual-only preflight must receive every one of these literally from
    # the staged runtime/broker/parent.  A converter cannot create evidence
    # merely because a marker claims the R246 stop reason or an old count/time
    # spelling is present.
    for field in (
        "mcts_child_called",
        "child_search_elapsed_seconds",
        "child_search_budget_seconds",
        "parent_action_deadline_seconds",
        "broker_started",
        "direct_action_precomputed_and_validated",
        "two_lane_topology_initialized_before_terminal_win_override",
        "terminal_win_proof_backed_up_into_shared_root_tree",
        "terminal_leaf_returned_by_exact_stock_simulator",
        "parent_validated_current_root_observation_legal_fingerprint_and_actor",
        "all_owned_lane_resources_reservations_and_child_cleanup_complete",
        "terminal_win_proof_count",
        "proven_deterministic_terminal_win_this_turn_stop_count",
    ):
        assert field not in projected


def test_raw_smoke_identity_drift_is_rejected_before_route_conversion(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    raw = _raw_smoke(paths, binding_identity)
    raw["exact_package_identity"] = {
        **raw["exact_package_identity"],  # type: ignore[arg-type]
        "archive": str(tmp_path / "other-candidate.tar.gz"),
    }
    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="physical smoke exact package identity does not match this run",
    ):
        RUNNER._validate_raw_smoke(
            raw,
            stage=paths["stage"],
            binding_identity=binding_identity,
        )


def test_worker_binding_identity_requires_all_exact_identity_layers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    expected = _binding_identity(paths)
    payload = {
        "exact_package_identity": dict(expected["exact_package"]),
        "stage_contract": dict(expected["stage_contract"]),
        "common_identity": dict(expected["common_identity"]),
    }

    RUNNER._require_binding_identity(payload, expected=expected, label="stock worker")

    payload["common_identity"] = {"candidate_archive_sha256": "sha256:wrong"}
    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="stock worker common identity does not match this run",
    ):
        RUNNER._require_binding_identity(payload, expected=expected, label="stock worker")


def _cpu_cuda_observation() -> dict[str, object]:
    return {
        "schema": RUNNER.preflight.CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": RUNNER.preflight.CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "cpu",
        "telemetry_complete": True,
        "error_types": [],
    }


def _actual_runtime_boundary_payload(
    paths: dict[str, Path], binding_identity: dict[str, object]
) -> dict[str, object]:
    """One typed, actual-only runtime observation without a live simulator."""

    cuda = _cpu_cuda_observation()
    phase1_target = dict(RUNNER.preflight.PHASE1_MANIFEST_RESOURCE_BOUNDS)
    runtime_topology = {
        "configured_vcpus": 2,
        "configured_simulator_lane_count": 2,
        "maximum_simulator_lanes": 2,
        "observed_active_simulator_lane_count": 2,
        "receipt_lane_count": 2,
        "receipt_schema": RUNNER.preflight.R238_MANIFEST_SCHEMA,
        "maximum_simulator_calls_in_flight": 2,
        "worker_thread_count": 1,
        "observed_peak_worker_threads": 1,
    }
    marker = {
        "mode": "shared_tree_mcts",
        "mcts_action_authority": True,
        "degraded": False,
        "selected_action": [0],
    }
    return {
        "schema": RUNNER.SCENARIO_EVIDENCE_SCHEMA,
        "status": "passed",
        "passed": True,
        "witness_origin": RUNNER.ACTUAL_STOCK_RUNTIME_OBSERVATION_ORIGIN,
        "evidence_kind": RUNNER.ACTUAL_STOCK_RUNTIME_EVIDENCE_KIND,
        "common_identity": dict(binding_identity["common_identity"]),
        "exact_package_identity": dict(binding_identity["exact_package"]),
        "stage_contract": dict(binding_identity["stage_contract"]),
        "stage_mutation_check": {"unchanged": True},
        "actual_stock_runtime_observation": {
            "schema": RUNNER.ACTUAL_STOCK_RUNTIME_OBSERVATION_SCHEMA,
            "observation_origin": RUNNER.ACTUAL_STOCK_RUNTIME_SEARCH_STEP_ORIGIN,
            "sealed_stage_runtime_module": "poke_bot.r228_kaggle_async_runtime",
            "sealed_runtime_evaluator_method": "R228AsyncGameplay._evaluate_batch",
            "official_r236_search_step_succeeded": True,
            "model_value_evaluated": True,
            "stage_mutation_unchanged": True,
            "action_authority_granted": False,
            "opponent_action_selected_or_planned": False,
            "opponent_action_cached": False,
            "frozen_evaluator_value_call_count": 1,
            "expanded_legal_action_count": 0,
            "expanded_child_count": 0,
            "search_steps_beyond_boundary": 0,
            "root_actor_seat": 0,
            "leaf_actor_seat": 1,
            "root_observation_fingerprint": "sha256:root",
            "successor_observation_fingerprint": "sha256:successor",
            "official_r236_search_step": {
                "search_begin_succeeded": True,
                "search_step_succeeded": True,
                "lane_handle_identity": "fresh-handle-0",
                "root_search_id": 0,
                "selected_action": [0],
                "root_actor_seat": 0,
                "successor_actor_seat": 1,
                "official_linux_x86_64_libcg_sha256": "sha256:fixture-r236-linux",
            },
        },
        "literal_staged_marker": marker,
        "literal_staged_marker_sha256": RUNNER._sha256_bytes(marker),
        "physical_stock_callback": {
            "stock_action_accepted": True,
            "action": [0],
            "callback_elapsed_seconds": 0.01,
        },
        "observed_resource_probe": {
            "runtime_disk_bytes": 123,
            "child_peak_rss_bytes": 30,
            "phase1_target": phase1_target,
            "runtime": runtime_topology,
            "cuda_runtime_before_search": cuda,
        },
        "actual_parent_broker_resource_startup_observation": {
            "measurement_origin": "fresh_sealed_parent_and_exact_broker_child",
            "startup_ready_before_first_search": True,
            "broker_child_observed_while_alive": True,
            "parent_peak_rss_bytes": 10,
            "broker_child_peak_rss_bytes": 20,
            "combined_nested_parent_broker_peak_rss_bytes": 30,
            "parent_worker_thread_count_peak": 1,
            "broker_child_worker_thread_count_peak": 1,
            "startup_seconds": 0.02,
            "runtime_disk_bytes": 123,
            "phase1_target": phase1_target,
            **runtime_topology,
            "parent_cuda_runtime_before_search": cuda,
            "broker_child_cuda_runtime_before_search": cuda,
        },
        "startup_seconds": 0.02,
    }


def test_actual_runtime_observation_derives_only_a_value_only_actor_boundary(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    payload = _actual_runtime_boundary_payload(paths, binding_identity)

    witnesses = RUNNER._scenario_witness_bundle(
        payload,
        name="actual-boundary",
        binding_identity=binding_identity,
    )

    assert witnesses is not None
    boundary = witnesses["actor_change_end_turn_boundary"]
    assert boundary["declared_opponent_actor_leaf_count"] == 1
    assert boundary["opponent_actor_leaves"] == [
        {
            "model_value_evaluated": True,
            "expanded_legal_action_count": 0,
            "expanded_child_count": 0,
            "search_steps_beyond_boundary": 0,
            "opponent_action_selected_or_planned": False,
            "opponent_action_cached": False,
        }
    ]
    assert witnesses["startup_seconds"] == 0.02
    assert witnesses["observed_resource_probe"]["child_peak_rss_bytes"] == 30


def test_actual_runtime_observation_cannot_claim_a_boundary_after_expansion(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    payload = _actual_runtime_boundary_payload(paths, binding_identity)
    payload["actual_stock_runtime_observation"]["expanded_child_count"] = 1  # type: ignore[index]

    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="expanded beyond actor boundary",
    ):
        RUNNER._scenario_witness_bundle(
            payload,
            name="actual-boundary",
            binding_identity=binding_identity,
        )


def test_actual_runtime_observation_requires_a_bound_official_search_step(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    payload = _actual_runtime_boundary_payload(paths, binding_identity)
    runtime = payload["actual_stock_runtime_observation"]
    assert isinstance(runtime, dict)
    search_step = runtime["official_r236_search_step"]
    assert isinstance(search_step, dict)
    search_step["successor_actor_seat"] = 0

    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="official SearchStep successor actor drifted",
    ):
        RUNNER._scenario_witness_bundle(
            payload,
            name="actual-boundary",
            binding_identity=binding_identity,
        )


def _actual_r244_witness(
    payload: dict[str, object], binding_identity: dict[str, object]
) -> dict[str, object]:
    marker = payload["literal_staged_marker"]
    assert isinstance(marker, dict)
    marker.update(
        {
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "search_begin_calls": 2,
            "per_lane_handle_identities": ["handle-0", "handle-1"],
            "per_lane_search_id_chains": [[0], [0]],
            "per_lane_first_search_ids": [0, 0],
            "handle_scoped_first_search_id_composite_states": [
                {"lane_id": 0, "handle_identity": "handle-0", "first_search_id": 0},
                {"lane_id": 1, "handle_identity": "handle-1", "first_search_id": 0},
            ],
        }
    )
    payload["literal_staged_marker_sha256"] = RUNNER._sha256_bytes(marker)
    return {
        "schema": RUNNER.R244_WITNESS_SCHEMA,
        "witness_origin": (
            "actual_staged_mcts_marker_topology_with_r225_contract_namespace_projection"
        ),
        "common_identity": dict(binding_identity["common_identity"]),
        "exact_package_identity": dict(binding_identity["exact_package"]),
        "stage_contract": dict(binding_identity["stage_contract"]),
        "literal_staged_marker_sha256": payload["literal_staged_marker_sha256"],
        "semantic_contract_source": {
            "kind": RUNNER.R244_CONTRACT_PROJECTION_KIND,
            "r225_contract_sha256": binding_identity["common_identity"]["r225_contract_sha256"],
            "owner_handle_scoped_search_id_revision": 244,
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        },
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_id_numeric_namespace": "per_distinct_agent_start_handle",
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        "per_lane_handle_identities": ["handle-0", "handle-1"],
        "per_lane_search_id_chains": [[0], [0]],
        "per_lane_first_search_ids": [0, 0],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": "handle-0", "first_search_id": 0},
            {"lane_id": 1, "handle_identity": "handle-1", "first_search_id": 0},
        ],
    }


def test_actual_r244_witness_requires_literal_marker_and_contract_provenance(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    scenario = _actual_runtime_boundary_payload(paths, binding_identity)
    witness = _actual_r244_witness(scenario, binding_identity)
    output = tmp_path / "r244-actual-witness.json"
    output.write_text(json.dumps(witness, sort_keys=True), encoding="utf-8")

    checked = RUNNER._validate_written_r244_actual_witness(
        witness_path=output,
        scenario_payload=scenario,
        binding_identity=binding_identity,
    )

    assert checked["per_lane_first_search_ids"] == [0, 0]


def test_actual_r244_witness_rejects_topology_not_in_the_literal_marker(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    binding_identity = _binding_identity(paths)
    scenario = _actual_runtime_boundary_payload(paths, binding_identity)
    witness = _actual_r244_witness(scenario, binding_identity)
    witness["arena_count"] = 1
    output = tmp_path / "r244-actual-witness.json"
    output.write_text(json.dumps(witness, sort_keys=True), encoding="utf-8")

    with pytest.raises(RUNNER.ExactStageProbeError, match="arena_count drifted"):
        RUNNER._validate_written_r244_actual_witness(
            witness_path=output,
            scenario_payload=scenario,
            binding_identity=binding_identity,
        )


def _controlled_parent_payload(stage: Path, witnesses: dict[str, object]) -> dict[str, object]:
    """A reviewed-worker shaped payload without launching its parent process."""

    return {
        "schema": RUNNER.CONTROLLED_PARENT_ROUTE_SCHEMA,
        "status": "passed",
        "controlled": True,
        "evidence_kind": "controlled_parent_route",
        "controlled_parent_route": True,
        "evidence_class": (
            "controlled_in_memory_parent_route_regression_not_physical_game_"
            "not_preflight_eligible"
        ),
        "network_accessed": False,
        "kaggle_api_called": False,
        "kaggle_upload_used": False,
        "gpu_used": False,
        "simulator_started": False,
        "model_loaded": False,
        "stage_mutation_check": {"unchanged": True},
        "normalized_parent_route_evidence": {
            "controlled_only": True,
            "nonphysical": True,
            "not_r240_final_schema": True,
            **witnesses,
        },
        "route_results": [
            {
                "status": "passed",
                "result": {
                    "stage_import": {
                        "main": str(stage / "main.py"),
                        "features": str(stage / "poke_bot/features.py"),
                    }
                },
            }
        ],
    }


def test_controlled_parent_evidence_cannot_supply_stock_only_witnesses(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    payload = _controlled_parent_payload(
        paths["stage"],
        {"actor_change_end_turn_boundary": {"fabricated": False}},
    )
    scenario = RUNNER.ScenarioResult(
        name="controlled",
        payload=payload,
        outcome=_complete_outcome(payload),
    )

    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="controlled_parent_route may not contribute stock-only witnesses: "
        "actor_change_end_turn_boundary",
    ):
        RUNNER._collect_witnesses(
            smoke={"decision_markers": [], "callbacks": []},
            scenario_results=[scenario],
            binding_identity=_binding_identity(paths),
            stage=paths["stage"],
        )


def _controlled_full_game_payload() -> tuple[dict[str, object], dict[str, object]]:
    """Smallest reviewed parent-only source accepted by the legacy mapper."""

    high = {
        **_high_marker(),
        "selected_factorized_stage_probabilities": [0.91],
        "selected_factorized_stage_probability_threshold": 0.8,
        "all_selected_factorized_stages_meet_threshold": True,
    }
    source_marker = {
        "mode": "shared_tree_mcts",
        "mcts_child_started": True,
        "mcts_child_call_count": 1,
        "mcts_action_authority": True,
        "selected_action": [1],
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "per_lane_handle_identities": ["controlled-handle-0", "controlled-handle-1"],
        "per_lane_search_id_chains": [[0], [0]],
        "per_lane_first_search_ids": [0, 0],
        "handle_scoped_first_search_id_composite_states": [
            {
                "lane_id": 0,
                "handle_identity": "controlled-handle-0",
                "first_search_id": 0,
            },
            {
                "lane_id": 1,
                "handle_identity": "controlled-handle-1",
                "first_search_id": 0,
            },
        ],
    }
    fingerprint_source = "staged_main._canonical_observation_fingerprint"
    payload = {
        "controlled_only": True,
        "nonphysical": True,
        "decision_events": [
            {
                "mode": "high_confidence_frozen_direct",
                "controlled_only": True,
                "controlled_root_observation_fingerprint": "sha256:controlled-high",
                "controlled_root_observation_fingerprint_source": fingerprint_source,
                "parent_action_elapsed_seconds": 0.01,
            },
            {
                "mode": "new_adaptive_two_lane_mcts",
                "route_case": "continuation_consume",
                "controlled_only": True,
                "controlled_root_observation_fingerprint": "sha256:controlled-mcts",
                "controlled_root_observation_fingerprint_source": fingerprint_source,
                "selected_action": [1],
                "parent_action_elapsed_seconds": 0.02,
            },
            {
                "mode": "cached_deterministic_continuation",
                "route_case": "continuation_consume",
                "controlled_only": True,
                "controlled_root_observation_fingerprint": "sha256:controlled-consume",
                "controlled_root_observation_fingerprint_source": fingerprint_source,
                "selected_action": [1],
                "parent_action_elapsed_seconds": 0.01,
            },
        ],
        "controlled_high_confidence_root_observation_fingerprint": "sha256:controlled-high",
        "controlled_plan_extraction_root_observation_fingerprint": "sha256:controlled-mcts",
        "controlled_plan_extraction_marker": source_marker,
        "controlled_plan_extraction_marker_is_verbatim_staged_parent_marker": True,
        "deterministic_continuation_plans": [
            {
                "plan_id": "controlled-plan-1",
                "actual_turn_id": "controlled-turn-1",
                "steps": [
                    {
                        "canonical_observation_fingerprint": "sha256:controlled-consume",
                        "planned_action": [1],
                    }
                ],
            }
        ],
        "deterministic_continuation_regression": {
            "chance_disagreement_clears_entire_plan": True,
            "fingerprint_disagreement_clears_entire_plan": True,
            "action_disagreement_clears_entire_plan": True,
            "actor_disagreement_clears_entire_plan": True,
            "precomputed_direct_action_and_history_correction_retained": True,
        },
    }
    return payload, high


def test_normalize_controlled_full_game_emits_three_controlled_legacy_events() -> None:
    payload, high = _controlled_full_game_payload()

    normalized = RUNNER._normalize_controlled_full_game(payload, high=high)

    events = normalized["decision_events"]
    assert isinstance(events, list)
    assert len(events) == 3
    assert [event["mode"] for event in events] == [
        "high_confidence_frozen_direct",
        "new_adaptive_two_lane_mcts",
        "cached_deterministic_continuation",
    ]
    assert all(event["controlled_only"] is True for event in events)
    assert all(event["nonphysical"] is True for event in events)
    assert [event["controlled_evidence_origin"] for event in events] == [
        "staged_parent_direct_route",
        "staged_parent_controlled_broker_route",
        "staged_parent_continuation_consume_route",
    ]


def test_normalize_controlled_full_game_rejects_missing_verbatim_source_marker() -> None:
    payload, high = _controlled_full_game_payload()
    del payload["controlled_plan_extraction_marker"]

    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="controlled verbatim plan-extraction marker must be an object",
    ):
        RUNNER._normalize_controlled_full_game(payload, high=high)


def test_parse_json_output_rejects_multiple_json_values() -> None:
    outcome = _complete_outcome({"first": True})
    outcome.stdout = b'{"first": true}\n{"second": true}\n'
    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="did not emit exactly one JSON object",
    ):
        RUNNER._parse_json_output(outcome, label="controlled scenario")


def test_controlled_parser_requires_its_single_reviewed_evidence_prefix() -> None:
    outcome = _complete_outcome({"status": "passed"})
    with pytest.raises(
        RUNNER.ExactStageProbeError,
        match="did not emit exactly one required controlled-evidence row",
    ):
        RUNNER._parse_json_output(
            outcome,
            label="controlled scenario",
            required_prefix=RUNNER.CONTROLLED_PARENT_ROUTE_PREFIX,
        )


def test_fresh_scenario_parses_only_a_stage_pinned_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = (tmp_path / "stage").resolve()
    stage.mkdir()
    calls: list[dict[str, object]] = []

    class FakeWatchdog:
        def __init__(self, **kwargs: object) -> None:
            calls.append({"init": kwargs})

        def run(self, argv: list[str], *, cwd: Path) -> SimpleNamespace:
            calls.append({"argv": argv, "cwd": cwd})
            return _complete_outcome({"schema": "example/v1", "status": "passed"})

    result = RUNNER._run_fresh_scenario(
        name="controlled",
        argv=[sys.executable, "worker.py", "--stage", str(stage)],
        stage=stage,
        timeout_seconds=1.0,
        term_grace_seconds=0.1,
        kill_grace_seconds=0.1,
        watchdog_factory=FakeWatchdog,
    )

    assert result.payload == {"schema": "example/v1", "status": "passed"}
    assert calls[1]["cwd"] == stage
    assert calls[1]["argv"] == [sys.executable, "worker.py", "--stage", str(stage)]


def test_fresh_scenario_rejects_a_child_pinned_to_another_stage(tmp_path: Path) -> None:
    stage = (tmp_path / "stage").resolve()
    stage.mkdir()
    other = (tmp_path / "other-stage").resolve()
    other.mkdir()

    class MustNotLaunch:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("mismatched stage must be rejected before launch")

    with pytest.raises(RUNNER.ExactStageProbeError, match="not pinned to the requested stage"):
        RUNNER._run_fresh_scenario(
            name="wrong-stage",
            argv=[sys.executable, "worker.py", "--stage", str(other)],
            stage=stage,
            timeout_seconds=1.0,
            term_grace_seconds=0.1,
            kill_grace_seconds=0.1,
            watchdog_factory=MustNotLaunch,
        )
