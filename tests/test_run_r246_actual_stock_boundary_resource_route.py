"""CPU-only contract tests for the actual R246 boundary/resource worker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = ROOT / "scripts/run_r246_actual_stock_boundary_resource_route.py"
OUTER_SCRIPT = ROOT / "scripts/run_r235_r246_exact_stage_probe.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_module("r246_actual_boundary_resource_worker_test", WORKER_SCRIPT)
OUTER = _load_module("r235_r246_outer_contract_for_worker_test", OUTER_SCRIPT)


def _cuda() -> dict[str, object]:
    return {
        "schema": "poke_bot.r238_cuda_runtime_observation/v1",
        "phase": "before_search",
        "torch_imported": True,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "cpu",
        "telemetry_complete": True,
        "error_types": [],
    }


def _binding() -> dict[str, object]:
    return {
        "common_identity": {
            "r225_contract_sha256": "sha256:r225",
            "linux_x86_64_libcg_sha256": "sha256:r236-linux",
            "phase1_submission_environment": {
                "hdd_space_gib": 11.8,
                "ram_gib": 12.2,
                "vcpus": 2,
                "submission_archive_limit_mib": 197.7,
            },
        },
        "exact_package": {"archive": "sha256:candidate"},
        "stage_contract": {"tree": "sha256:stage"},
    }


def _marker() -> dict[str, object]:
    handles = [101, 202]
    chains = [[0, 3], [0, 4]]
    return {
        "mode": "shared_tree_mcts",
        "degraded": False,
        "mcts_action_authority": True,
        "selected_action": [2],
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "configured_simulator_lane_count": 2,
        "max_simulator_calls_in_flight": 2,
        "per_lane_handle_identities": handles,
        "per_lane_search_id_chains": chains,
        "per_lane_first_search_ids": [0, 0],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": 101, "first_search_id": 0},
            {"lane_id": 1, "handle_identity": 202, "first_search_id": 0},
        ],
        "parent_cuda_runtime_before_search": _cuda(),
        "broker": {
            "child_pid": 73,
            "child_identity": {
                "pid": 73,
                "started_monotonic": 1.0,
                "cuda_runtime_before_search": _cuda(),
            },
        },
    }


def _statuses() -> tuple[Any, Any]:
    return (
        WORKER._ProcessStatus(
            pid=71,
            vm_rss_bytes=100,
            vm_hwm_bytes=140,
            thread_count=2,
            source="test-proc",
        ),
        WORKER._ProcessStatus(
            pid=73,
            vm_rss_bytes=60,
            vm_hwm_bytes=90,
            thread_count=1,
            source="test-proc",
        ),
    )


def _runtime_observation() -> dict[str, object]:
    return {
        "schema": WORKER.ACTUAL_RUNTIME_OBSERVATION_SCHEMA,
        "observation_origin": "fresh_official_r236_search_step_actor_change_successor",
        "sealed_stage_runtime_module": "poke_bot.r228_kaggle_async_runtime",
        "sealed_runtime_evaluator_method": "R228AsyncGameplay._evaluate_batch",
        "root_actor_seat": 0,
        "leaf_actor_seat": 1,
        "root_observation_fingerprint": "sha256:root",
        "successor_observation_fingerprint": "sha256:successor",
        "official_r236_search_step_succeeded": True,
        "official_r236_search_step": {
            "search_begin_succeeded": True,
            "search_step_succeeded": True,
            "lane_handle_identity": 999,
            "root_search_id": 0,
            "selected_action": [2],
            "root_actor_seat": 0,
            "successor_actor_seat": 1,
            "official_linux_x86_64_libcg_sha256": "sha256:r236-linux",
        },
        "frozen_evaluator_value_call_count": 1,
        "model_value_evaluated": True,
        "expanded_legal_action_count": 0,
        "expanded_child_count": 0,
        "search_steps_beyond_boundary": 0,
        "opponent_action_selected_or_planned": False,
        "opponent_action_cached": False,
        "stage_mutation_unchanged": True,
        "action_authority_granted": False,
    }


def test_resource_probe_uses_measured_parent_and_broker_hwm_not_lane_count() -> None:
    marker = _marker()
    binding = _binding()
    parent, child = _statuses()

    probe, raw = WORKER._resource_probe_from_actual_measurements(
        stage_disk_bytes=1234,
        marker=marker,
        binding=binding,
        parent=parent,
        broker_child=child,
        startup_seconds=1.25,
    )

    assert raw["measurement_origin"] == "fresh_sealed_parent_and_exact_broker_child"
    assert raw["parent_peak_rss_bytes"] == 140
    assert raw["broker_child_peak_rss_bytes"] == 90
    assert raw["combined_nested_parent_broker_peak_rss_bytes"] == 230
    assert raw["parent_worker_thread_count_peak"] == 2
    assert raw["broker_child_worker_thread_count_peak"] == 1
    assert raw["parent_cuda_runtime_before_search"] == _cuda()
    assert raw["broker_child_cuda_runtime_before_search"] == _cuda()
    assert probe["child_peak_rss_bytes"] == 230
    assert probe["runtime"]["worker_thread_count"] == 2
    assert probe["runtime"]["observed_peak_worker_threads"] == 2
    assert probe["resource_observation_source"] == raw


def test_resource_probe_rejects_marker_child_pid_that_is_not_measured_child() -> None:
    marker = _marker()
    marker["broker"] = {**marker["broker"], "child_pid": 74}  # type: ignore[index]
    parent, child = _statuses()

    with pytest.raises(WORKER.ActualStockBoundaryResourceError, match="measured exact child"):
        WORKER._resource_probe_from_actual_measurements(
            stage_disk_bytes=1,
            marker=marker,
            binding=_binding(),
            parent=parent,
            broker_child=child,
            startup_seconds=0.1,
        )


def test_r244_witness_copies_live_topology_and_labels_only_static_projection() -> None:
    marker = _marker()
    witness = WORKER._r244_witness_from_literal_marker(marker=marker, binding=_binding())

    assert witness["schema"] == WORKER.R244_WITNESS_SCHEMA
    assert witness["witness_origin"] == WORKER.R244_WITNESS_ORIGIN
    assert witness["per_lane_handle_identities"] == [101, 202]
    assert witness["per_lane_search_id_chains"] == [[0, 3], [0, 4]]
    assert witness["handle_scoped_first_search_id_composite_states"] == marker[
        "handle_scoped_first_search_id_composite_states"
    ]
    assert witness["semantic_contract_source"] == {
        "kind": "r225_r244_static_handle_namespace_contract_projection",
        "r225_contract_sha256": "sha256:r225",
        "owner_handle_scoped_search_id_revision": 244,
        "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    }
    marker["per_lane_handle_identities"] = ["mutated", "marker"]
    assert witness["per_lane_handle_identities"] == [101, 202]


def test_write_once_r244_output_rejects_the_immutable_sealed_stage(tmp_path: Path) -> None:
    stage = tmp_path / "sealed-stage"
    stage.mkdir()

    with pytest.raises(WORKER.ActualStockBoundaryResourceError, match="outside the sealed stage"):
        WORKER._write_json_once(
            stage / "r244-actual-witness.json",
            {"schema": "test"},
            sealed_stage=stage,
        )


def test_write_once_r244_output_requires_an_unambiguous_absolute_path() -> None:
    with pytest.raises(WORKER.ActualStockBoundaryResourceError, match="absolute path"):
        WORKER._write_json_once(Path("r244-actual-witness.json"), {"schema": "test"})


def test_outer_converter_accepts_only_the_raw_actual_runtime_shape() -> None:
    binding = _binding()
    marker = _marker()
    parent, child = _statuses()
    probe, raw_resource = WORKER._resource_probe_from_actual_measurements(
        stage_disk_bytes=1234,
        marker=marker,
        binding=binding,
        parent=parent,
        broker_child=child,
        startup_seconds=1.25,
    )
    runtime = _runtime_observation()
    payload = {
        "schema": WORKER.SCHEMA,
        "status": "passed",
        "passed": True,
        "witness_origin": WORKER.WITNESS_ORIGIN,
        "evidence_kind": WORKER.EVIDENCE_KIND,
        "common_identity": binding["common_identity"],
        "exact_package_identity": binding["exact_package"],
        "stage_contract": binding["stage_contract"],
        "literal_staged_marker": marker,
        "literal_staged_marker_sha256": WORKER._canonical_sha256(marker),
        "physical_stock_callback": {
            "stock_action_accepted": True,
            "action": [2],
            "callback_elapsed_seconds": 0.2,
        },
        "actual_stock_runtime_observation": runtime,
        "actual_parent_broker_resource_startup_observation": raw_resource,
        "observed_resource_probe": probe,
        "startup_seconds": 1.25,
        "stage_mutation_check": {"unchanged": True},
        "action_authority_granted": False,
    }

    normalized = OUTER._actual_stock_runtime_observation_witnesses(
        payload,
        name="unit-actual-stock-worker",
        binding_identity=binding,
    )

    assert normalized["actor_change_end_turn_boundary"]["opponent_actor_leaves"] == [
        {
            "model_value_evaluated": True,
            "expanded_legal_action_count": 0,
            "expanded_child_count": 0,
            "search_steps_beyond_boundary": 0,
            "opponent_action_selected_or_planned": False,
            "opponent_action_cached": False,
        }
    ]
    assert "r240_witnesses" not in payload


def test_run_passes_the_exact_stage_and_binding_to_real_search_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    archive = tmp_path / "candidate.tar.gz"
    manifest = tmp_path / "manifest.json"
    r225 = tmp_path / "r225.json"
    r236 = tmp_path / "r236.json"
    for path in (archive, manifest, r225, r236):
        path.write_text("{}", encoding="utf-8")

    binding = _binding()
    marker = _marker()
    root = WORKER._PhysicalBoundaryRoot(
        observation={"current": {"result": -1, "yourIndex": 0}},
        action=[2],
        root_actor_seat=0,
        physical_successor={"current": {"result": -1, "yourIndex": 1}},
        callback_elapsed_seconds=0.1,
        callback_index=3,
    )
    captured: dict[str, object] = {}
    parent, child = _statuses()

    monkeypatch.setattr(WORKER, "stage_snapshot", lambda _stage: {"tree_sha256": "sha256:stage"})
    monkeypatch.setattr(WORKER, "load_binding_identity", lambda **_kwargs: binding)
    monkeypatch.setattr(WORKER, "_stage_disk_bytes", lambda _stage: 1234)
    monkeypatch.setattr(
        WORKER,
        "_load_exact_stage",
        lambda _stage: (SimpleNamespace(_BROKER=None), object(), object()),
    )
    monkeypatch.setattr(WORKER, "_deck", lambda _stage: [1] * 60)
    monkeypatch.setattr(
        WORKER,
        "_capture_mcts_marker_and_physical_actor_change",
        lambda **_kwargs: (marker, root),
    )
    monkeypatch.setattr(WORKER, "_read_proc_status", lambda pid: parent if pid != 73 else child)
    monkeypatch.setattr(WORKER, "_close_exact_broker", lambda _main: None)

    leaf_source = {
        "observation_origin": "fresh_official_r236_search_step_actor_change_successor",
        "official_r236_search_step_succeeded": True,
        "official_r236_search_step": _runtime_observation()["official_r236_search_step"],
    }

    def fake_search(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(leaf_source)

    monkeypatch.setattr(WORKER, "_search_step_actor_change_observation", fake_search)
    monkeypatch.setattr(WORKER, "_evaluate_actual_boundary_leaf", lambda **_kwargs: _runtime_observation())
    monkeypatch.setattr(
        WORKER,
        "_resource_probe_from_actual_measurements",
        lambda **_kwargs: ({"probe": True}, {"raw": True}),
    )
    monkeypatch.setattr(
        WORKER,
        "_r244_witness_from_literal_marker",
        lambda **_kwargs: {"semantic_contract_source": {"kind": "test"}},
    )

    result = WORKER._run(
        stage=stage,
        candidate_archive=archive,
        member_manifest=manifest,
        r225_contract=r225,
        r236_contract=r236,
        max_physical_actions=4,
        r244_witness_output=None,
    )

    assert captured["stage"] == stage.resolve()
    assert captured["binding"] == binding
    assert captured["root"] == root
    assert result["witness_origin"] == WORKER.WITNESS_ORIGIN
    assert result["actual_stock_runtime_observation"]["stage_mutation_unchanged"] is True
    assert "r240_witnesses" not in result
