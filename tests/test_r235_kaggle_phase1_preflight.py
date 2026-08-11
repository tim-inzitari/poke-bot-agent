"""Offline, no-GPU coverage for the R235/R240/R242 package preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile
from typing import Any

import pytest

from poke_bot import r235_kaggle_phase1_preflight as preflight


REQUIRED_EXPORTS = [
    "AgentStart",
    "BattleStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_archive(stage: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted((item for item in stage.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            output.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)


def _probe(stage: Path, *, worker_threads: int = 2, lane_count: int = 2) -> dict[str, Any]:
    runtime_disk = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
    return {
        "schema": preflight.R240_PROBE_SCHEMA,
        "observed_resource_probe": {
            "runtime_disk_bytes": runtime_disk,
            "child_peak_rss_bytes": 4_096,
            "phase1_target": dict(preflight.PHASE1_MANIFEST_RESOURCE_BOUNDS),
            "runtime": {
                "configured_vcpus": 2,
                "configured_simulator_lane_count": lane_count,
                "maximum_simulator_lanes": lane_count,
                "observed_active_simulator_lane_count": lane_count,
                "receipt_lane_count": lane_count,
                "receipt_schema": preflight.R238_MANIFEST_SCHEMA,
                "worker_thread_count": worker_threads,
                "observed_peak_worker_threads": worker_threads,
                "maximum_simulator_calls_in_flight": lane_count,
            },
            # The published CPU/RAM/disk envelope cannot prove CUDA absence.
            # A CPU observation is one valid measured runtime result; separate
            # coverage below proves a hidden CUDA device is accepted too.
            "cuda_runtime_before_search": {
                "schema": preflight.CUDA_RUNTIME_OBSERVATION_SCHEMA,
                "phase": preflight.CUDA_RUNTIME_OBSERVATION_PHASE,
                "torch_imported": True,
                "cuda_available": False,
                "cuda_initialized": False,
                "device_count": 0,
                "devices": [],
                "model_device": "cpu",
                "telemetry_complete": True,
                "error_types": [],
            },
        },
        "startup_seconds": 0.02,
        "decision_latency_seconds": {"sample_count": 8, "p50": 0.002, "p95": 0.004, "max": 0.005},
        "throughput": {"decision_count": 16, "elapsed_seconds": 0.08, "decisions_per_second": 200.0},
        preflight.R240_HYBRID_PROBE_KEY: {
            "owner_decision_revision": 246,
            "configuration": {
                "high_confidence_threshold_owner_revision": 242,
                "high_confidence_threshold": 0.80,
                "child_search_hard_seconds": 2.0,
                "parent_action_hard_seconds": 4.0,
                "minimum_backups_before_stability": 8,
                "stable_root_leader_observations": 3,
                "maximum_backups_per_decision": 32,
                "maximum_deterministic_continuation_actions": 8,
                "legacy_fixed_eight_second_branching_windows_rejected": True,
                "historical_r240_0_90_threshold_draft_and_preflight_rejected": True,
                "proven_deterministic_terminal_win_this_turn_owner_revision": 246,
            },
            "synthetic_high_confidence_direct": {
                "selected_factorized_stage_probabilities": [0.80, 0.97],
                "selected_factorized_stage_probability_threshold": 0.80,
                "all_selected_factorized_stages_meet_threshold": True,
                "mode": "high_confidence_frozen_direct",
                "direct_action_precomputed_and_validated": True,
                "mcts_child_started_for_this_decision": False,
                "mcts_select_call_count": 0,
                "mcts_search_call_count": 0,
                "mcts_model_call_count": 0,
                "mcts_simulator_call_count": 0,
                "history_only_existing_child_journal_count": 1,
                "degraded": False,
                "parent_action_elapsed_seconds": 0.004,
            },
            "synthetic_ambiguous_two_lane_mcts": {
                "selected_factorized_stage_probabilities": [0.799, 0.97],
                "mode": "adaptive_two_lane_mcts",
                "direct_action_precomputed_and_validated": True,
                "broker_started": True,
                "mcts_child_started": True,
                "mcts_child_called": True,
                "mcts_action_authority": True,
                "both_lanes_progressed": True,
                "degraded": False,
                "stop_reason": "adaptive_early_stop",
                "terminal_win_proof": None,
                "requested_simulator_lane_count": 2,
                "active_simulator_lane_count": 2,
                "arena_count": 2,
                "unique_handle_count": 2,
                "maximum_simulator_calls_in_flight": 2,
                "receipt_schema_regression_passed": True,
                "one_shared_logical_mcts_tree_proved": True,
                "historical_eight_lane_manifest_or_receipt_accepted": False,
                "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
                "lane_ids": [0, 1],
                "search_begin_calls": 2,
                "search_release_calls": 2,
                "search_end_calls": 2,
                "per_lane_depth": [4, 4],
                "microbatch_sizes": [2, 2, 2, 2],
                "per_lane_handle_identities": [1001, 1002],
                "per_lane_search_id_chains": [[0], [0]],
                "per_lane_first_search_ids": [0, 0],
                "handle_scoped_first_search_id_composite_states": [
                    {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
                    {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
                ],
                "completed_backups": 8,
                "minimum_backups_before_stability": 8,
                "stable_root_leader_required_observations": 3,
                "maximum_backups_per_decision": 32,
                "deterministic_root_leader_observations": ["a", "a", "a"],
                "actor_change_boundary_leaf_count": 1,
                "chance_boundary_leaf_count": 0,
                "boundary_leaf_count": 1,
                "child_search_budget_seconds": 2.0,
                "child_search_elapsed_seconds": 0.02,
                "parent_action_deadline_seconds": 4.0,
                "parent_action_elapsed_seconds": 0.03,
            },
            "synthetic_proven_deterministic_terminal_win_this_turn": {
                "selected_factorized_stage_probabilities": [0.799, 0.97],
                "mode": "adaptive_two_lane_mcts",
                "stop_reason": "proven_deterministic_terminal_win_this_turn",
                "direct_action_precomputed_and_validated": True,
                "broker_started": True,
                "mcts_child_started": True,
                "mcts_child_called": True,
                "mcts_action_authority": True,
                "degraded": False,
                "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
                "proven_deterministic_terminal_win_this_turn": True,
                "two_lane_topology_initialized_before_terminal_win_override": True,
                "terminal_win_proof_backed_up_into_shared_root_tree": True,
                "terminal_leaf_returned_by_exact_stock_simulator": True,
                "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
                "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
                "two_independent_lane_proofs_required": False,
                "exhaustive_legal_action_scan_required": False,
                "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
                "requested_simulator_lane_count": 2,
                "active_simulator_lane_count": 2,
                "arena_count": 2,
                "unique_handle_count": 2,
                "search_begin_calls": 2,
                "search_release_calls": 2,
                "search_end_calls": 2,
                "outstanding_virtual_loss": 0,
                "per_lane_handle_identities": [2001, 2002],
                "per_lane_search_id_chains": [[0], [0]],
                "per_lane_first_search_ids": [0, 0],
                "handle_scoped_first_search_id_composite_states": [
                    {"lane_id": 0, "handle_identity": 2001, "first_search_id": 0},
                    {"lane_id": 1, "handle_identity": 2002, "first_search_id": 0},
                ],
                "per_lane_depth": [1, 0],
                "completed_backups": 1,
                "completed_root_backup_count": 1,
                "terminal_win_proof_count": 1,
                "proven_deterministic_terminal_win_this_turn_stop_count": 1,
                "minimum_backups_before_stability": 8,
                "stable_root_leader_required_observations": 3,
                "maximum_backups_per_decision": 32,
                "child_search_budget_seconds": 2.0,
                "child_search_elapsed_seconds": 0.01,
                "parent_action_deadline_seconds": 4.0,
                "parent_action_elapsed_seconds": 0.02,
                "terminal_win_proof": {
                    "proof_kind": "exact_deterministic_simulator_terminal_win_this_turn",
                    "root_observation_fingerprint": "terminal-root",
                    "root_legal_order_fingerprint": "terminal-legal",
                    "root_actor_seat": 0,
                    "root_action": [1],
                    "selected_action": [1],
                    "terminal_result": "win",
                    "terminal_winner_seat": 0,
                    "terminal_leaf_reached": True,
                    "proof_path_action_count": 1,
                    "discovering_lane_id": 0,
                    "path_actor_seats": [0],
                    "path_no_chance_boundary": True,
                    "path_no_actor_change_boundary": True,
                    "path_no_opponent_boundary_crossing": True,
                    "path_no_unresolved_randomness": True,
                    "proof_is_deterministic": True,
                },
            },
            "actor_change_end_turn_boundary": {
                "actor_change_end_turn_boundary_regression_passed": True,
                "declared_opponent_actor_leaf_count": 1,
                "value_evaluated_opponent_actor_leaf_count": 1,
                "expanded_legal_action_count": 0,
                "expanded_child_count": 0,
                "search_steps_beyond_boundary": 0,
                "opponent_action_selected_or_planned_count": 0,
                "opponent_action_cached_count": 0,
                "opponent_actor_leaves": [
                    {
                        "model_value_evaluated": True,
                        "expanded_legal_action_count": 0,
                        "expanded_child_count": 0,
                        "search_steps_beyond_boundary": 0,
                        "opponent_action_selected_or_planned": False,
                        "opponent_action_cached": False,
                    }
                ],
            },
            "actor_change_boundary_leaf_count": 1,
            "chance_boundary_leaf_count": 0,
            "boundary_leaf_count": 1,
            "full_game_cumulative": {
                "phase1_full_game_budget_seconds": 1.0,
                "cumulative_parent_wall_seconds": 0.05,
                "cumulative_child_search_seconds": 0.02,
                "new_mcts_search_count": 1,
                "cached_deterministic_continuation_count": 1,
                "high_confidence_frozen_direct_count": 1,
                "deterministic_continuation_plans": [
                    {
                        "plan_id": "plan-1",
                        "actual_turn_id": "turn-1",
                        "extracted_from_mode": "adaptive_two_lane_mcts",
                        "exact_fingerprint_proven": True,
                        "two_lane_agreed_backed_leader": True,
                        "no_chance_boundary_or_opponent_transition": True,
                        "crossed_actor_change_end_turn_boundary": False,
                        "steps": [
                            {
                                "canonical_observation_fingerprint": "next-fingerprint",
                                "planned_action": [1],
                            }
                        ],
                    }
                ],
                "decision_events": [
                    {
                        "mode": "new_adaptive_two_lane_mcts",
                        "actual_turn_id": "turn-1",
                        "canonical_observation_fingerprint": "root-fingerprint",
                        "broker_started": True,
                        "mcts_child_started": True,
                        "mcts_child_called": True,
                        "new_mcts_search_started": True,
                        "requested_simulator_lane_count": 2,
                        "active_simulator_lane_count": 2,
                        "per_lane_handle_identities": [1001, 1002],
                        "per_lane_search_id_chains": [[0], [0]],
                        "per_lane_first_search_ids": [0, 0],
                        "handle_scoped_first_search_id_composite_states": [
                            {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
                            {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
                        ],
                        "child_search_budget_seconds": 2.0,
                        "child_search_elapsed_seconds": 0.02,
                        "parent_action_deadline_seconds": 4.0,
                        "parent_action_elapsed_seconds": 0.03,
                    },
                    {
                        "mode": "cached_deterministic_continuation",
                        "actual_turn_id": "turn-1",
                        "canonical_observation_fingerprint": "next-fingerprint",
                        "plan_id": "plan-1",
                        "selected_action": [1],
                        "exact_fingerprint_match": True,
                        "same_actor": True,
                        "action_in_complete_legal_order": True,
                        "two_lane_agreed_backed_leader": True,
                        "no_chance_boundary_or_opponent_transition": True,
                        "crossed_actor_change_end_turn_boundary": False,
                        "new_mcts_search_started": False,
                        "mcts_child_called": False,
                        "mcts_child_started_for_this_decision": False,
                        "mcts_select_call_count": 0,
                        "mcts_search_call_count": 0,
                        "mcts_model_call_count": 0,
                        "mcts_simulator_call_count": 0,
                        "history_only_existing_child_journal_count": 1,
                        "degraded": False,
                        "parent_action_elapsed_seconds": 0.005,
                    },
                    {
                        "mode": "high_confidence_frozen_direct",
                        "actual_turn_id": "turn-2",
                        "canonical_observation_fingerprint": "direct-fingerprint",
                        "selected_factorized_stage_probabilities": [0.80],
                        "selected_factorized_stage_probability_threshold": 0.80,
                        "all_selected_factorized_stages_meet_threshold": True,
                        "mcts_child_started_for_this_decision": False,
                        "mcts_select_call_count": 0,
                        "mcts_search_call_count": 0,
                        "mcts_model_call_count": 0,
                        "mcts_simulator_call_count": 0,
                        "history_only_existing_child_journal_count": 0,
                        "degraded": False,
                        "parent_action_elapsed_seconds": 0.004,
                    },
                ],
                "deterministic_continuation_regression": {
                    "chance_disagreement_clears_entire_plan": True,
                    "fingerprint_disagreement_clears_entire_plan": True,
                    "action_disagreement_clears_entire_plan": True,
                    "actor_disagreement_clears_entire_plan": True,
                    "precomputed_direct_action_and_history_correction_retained": True,
                },
            },
        },
    }


def _fixture(
    tmp_path: Path, *, manifest_lane_count: int = 2
) -> tuple[preflight.R235PreflightInputs, preflight.R235PreflightLimits, dict[str, Any], Path]:
    stage = tmp_path / "stage"
    (stage / "cg").mkdir(parents=True)
    (stage / "main.py").write_text("def agent(_obs):\n    return [0]\n", encoding="utf-8")
    libcg = stage / "cg" / "libcg.so"
    libcg.write_bytes(b"synthetic-r236-libcg")
    entrypoint_sha = _sha(stage / "main.py")
    libcg_sha = _sha(libcg)
    r236 = {
        "schema": preflight.R236_SCHEMA,
        "owner_decision_revision": 236,
        "canonical_native_libraries": {
            "linux_x86_64": {
                "package_relative_path": "cg/libcg.so",
                "sha256": libcg_sha,
                "size_bytes": libcg.stat().st_size,
            }
        },
        "required_native_exports": REQUIRED_EXPORTS,
        "upstream_provenance": {"wheel_sha256": "sha256:" + "a" * 64, "package_version": "1.32.6"},
    }
    r236_path = tmp_path / "r236.json"
    r236_path.write_text(json.dumps(r236), encoding="utf-8")
    r225 = {
        "schema": preflight.R225_SCHEMA,
        "owner_decision_revision": 242,
        "owner_phase1_submission_resources_and_two_lane_revision": 238,
        "owner_hybrid_confidence_bounded_mcts_revision": 240,
        "owner_high_confidence_frozen_direct_threshold_revision": 242,
        "relationship_to_existing_work": {
            "r240_supersedes_only_the_new_r235_replacement_package_decision_scheduling_and_does_not_modify_r229_or_bo1000": True,
            "r240_deterministic_continuation_is_parent_owned_and_does_not_change_the_r234_containment_boundary": True,
            "r242_supersedes_only_the_r240_high_confidence_frozen_direct_threshold_for_the_new_r235_replacement_package": True,
            "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible": True,
            "r242_does_not_modify_r234_r236_r238_r235_continuation_or_r229_bo1000": True,
        },
        "phase1_submission_environment": {
            "owner_decision_revision": 238,
            "hdd_space_gib": 11.8,
            "ram_gib": 12.2,
            "vcpus": 2,
            "submission_archive_limit_mib": 197.7,
            "resource_probe_and_archive_size_receipt_required": True,
            "resource_mismatch_or_archive_over_limit_behavior": "hard_fail_closed_and_do_not_upload",
        },
        "replacement_kaggle_diagnostic": {
            "revision": 235,
            "phase1_two_lane_resource_revision": 238,
            "phase1_simulator_search": {
                "required_simulator_search_lane_count": 2,
                "required_active_simulator_search_lane_count": 2,
                "required_internal_agent_start_simulator_search_arena_count": 2,
                "required_distinct_search_begin_id_count": 2,
                "required_unique_raw_handle_count": 2,
                "one_lane_serial_fallback_eight_lane_topology_or_partial_lane_mcts_authority_allowed": False,
            },
        },
        "local_preflight": {
            "required_simulator_search_lane_count": 2,
            "required_internal_agent_start_simulator_search_arena_count_per_child": 2,
            "required_distinct_search_begin_id_count": 2,
            "exactly_two_simultaneous_in_decision_simulator_search_lanes_required": True,
            "r240_hybrid_scheduler": {
                "high_confidence_frozen_direct_threshold_owner_revision": 242,
                "selected_factorized_stage_probability_threshold": 0.80,
                "all_selected_factorized_stages_must_meet_threshold": True,
                "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
                "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
                "high_confidence_frozen_direct_mode": "high_confidence_frozen_direct",
                "high_confidence_mcts_child_started_for_this_decision": False,
                "high_confidence_mcts_select_search_model_or_simulator_call_allowed": False,
                "high_confidence_existing_child_history_only_note_direct_action_ipc_allowed_and_required_when_child_exists": True,
                "high_confidence_existing_child_history_only_note_direct_action_ipc_count_range": [0, 1],
                "high_confidence_history_only_note_direct_action_ipc_must_not_invoke_mcts_select_search_model_or_simulator": True,
                "child_search_hard_seconds": 2.0,
                "parent_action_hard_seconds": 4.0,
                "adaptive_early_stop_min_completed_backups": 8,
                "adaptive_early_stop_stable_deterministic_root_leader_observations": 3,
                "adaptive_early_stop_both_lanes_progressed_required": True,
                "hard_completed_backup_stop": 32,
                "mcts_simulated_rollout_expansion_stops_at_terminal_chance_boundary_or_actor_change_away_from_root_seat": True,
                "root_actor_change_away_from_our_seat_leaf_is_value_evaluated_without_expanded_legal_actions_or_children": True,
                "mcts_opponent_action_selection_or_planning_allowed": False,
                "boundary_leaf_receipt_required_fields": [
                    "actor_change_boundary_leaf_count",
                    "chance_boundary_leaf_count",
                    "boundary_leaf_count",
                ],
                "high_confidence_receipt_required_values": {
                    "selected_factorized_stage_probability_threshold": 0.80,
                    "all_selected_factorized_stages_meet_threshold": True,
                    "mcts_child_started_for_this_decision": False,
                    "mcts_select_call_count": 0,
                    "history_only_existing_child_journal_count_range": [0, 1],
                    "degraded": False,
                },
                "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
            },
            "deterministic_continuation": {
                "maximum_depth": 8,
                "current_canonical_observation_fingerprint_must_exactly_match": True,
                "current_actor_must_remain_our_seat": True,
                "next_planned_action_must_be_in_current_complete_ordered_legal_actions": True,
                "plan_extraction_requires_both_lanes_saw_the_same_fingerprint": True,
                "plan_extraction_requires_both_lanes_agreed_on_a_backed_leader": True,
                "chance_boundary_or_opponent_transition_since_plan_extraction_allowed": False,
                "deterministic_continuation_stops_at_terminal_chance_boundary_or_actor_change_away_from_our_seat": True,
                "valid_plan_starts_or_calls_new_mcts_search": False,
                "valid_plan_mcts_child_started_for_this_decision": False,
                "valid_plan_mcts_select_call_count": 0,
                "valid_plan_history_only_existing_child_journal_count_range": [0, 1],
                "valid_plan_receipt_required_values": {
                    "mcts_child_started_for_this_decision": False,
                    "mcts_select_call_count": 0,
                    "history_only_existing_child_journal_count_range": [0, 1],
                    "degraded": False,
                },
                "rewrite_history_to_actual_planned_action": True,
                "journal_exactly_once": True,
            },
        },
        "canonical_libcg_revision": {"typed_source": "state/canonical-libcg-r236.json"},
    }
    # Production preflight pins the final immutable canonical R244 bytes even
    # in dry mode; the local mapping above documents the fixture semantics.
    r225_path = preflight.ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
    manifest = {
        "schema": preflight.R238_MANIFEST_SCHEMA,
        "role": preflight.R238_MANIFEST_ROLE,
        "required_label": preflight.R235_LABEL,
        "complete_action_cap": preflight.COMPLETE_ACTION_CAP,
        "lane_count": manifest_lane_count,
        "phase1_kaggle_resource_bounds": {
            **preflight.PHASE1_MANIFEST_RESOURCE_BOUNDS,
            "archive_max_bytes": preflight.PHASE1_ARCHIVE_MAX_BYTES,
        },
        "entrypoint_sha256": entrypoint_sha,
        "canonical_native_members": {
            "cg/libcg.so": {"sha256": libcg_sha, "size_bytes": libcg.stat().st_size}
        },
        "broker_contract": {
            "complete_action_cap": preflight.COMPLETE_ACTION_CAP,
            "search_seconds": 2.0,
            "action_timeout_seconds": 4.0,
            "subprocess_containment": {
                "signals_exact_owned_child_only": True,
                "process_group_or_session_signalling": True,
                "bounded_reap_required": True,
            },
        },
        "r240_hybrid_scheduler": dict(preflight.R240_MANIFEST_HYBRID_SCHEDULER),
        "deterministic_continuation": dict(
            preflight.R240_MANIFEST_DETERMINISTIC_CONTINUATION
        ),
    }
    manifest_path = stage / "truthful-r238-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    archive = tmp_path / "truthful-r238-two-lane.tar.gz"
    _write_archive(stage, archive)
    inputs = preflight.R235PreflightInputs(
        stage_dir=stage,
        archive_path=archive,
        manifest_path=manifest_path,
        expected_archive_sha256=_sha(archive),
        expected_manifest_sha256=_sha(manifest_path),
        receipt_path=tmp_path / "resource-preflight.receipt.json",
        r225_contract_path=r225_path,
        r236_contract_path=r236_path,
    )
    limits = preflight.R235PreflightLimits(
        probe_timeout_seconds=1.0,
        term_grace_seconds=0.1,
        kill_grace_seconds=0.1,
        max_startup_seconds=0.5,
        max_decision_latency_seconds=0.1,
        max_full_game_cumulative_seconds=1.0,
        min_throughput_decisions_per_second=10.0,
        min_throughput_decision_count=8,
    )
    return inputs, limits, _probe(stage, lane_count=manifest_lane_count), stage


def test_dry_preflight_binds_r238_resources_and_r240_schedule_without_launching_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)

    class MustNotLaunch:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("dry preflight must not launch an exact child")

    monkeypatch.setattr(preflight, "R235ExactChildWatchdog", MustNotLaunch)
    receipt = preflight.run_r235_phase1_preflight(
        inputs=inputs,
        limits=limits,
        dry_run=True,
        offline_probe_payload=probe,
        offline_exports=REQUIRED_EXPORTS,
    )

    assert receipt["status"] == "dry_run_not_execution_eligible"
    assert receipt["passed"] is False
    assert receipt["simulator_search_lane_count"] == 2
    assert receipt["phase1_resources"] == preflight.PHASE1_RESOURCES
    assert receipt["phase1_submission_environment"] == preflight.PHASE1_SUBMISSION_ENVIRONMENT
    assert receipt["r240_hybrid_scheduler"] == preflight.R240_HYBRID_SCHEDULER
    assert receipt["deterministic_continuation"] == preflight.DETERMINISTIC_CONTINUATION
    assert receipt["phase1_enforcement_bytes"]["archive_max_bytes"] == 207_303_475
    assert receipt["r240_hybrid_decision_preflight"]["full_game_cumulative"][
        "decision_mode_counts"
    ] == {
        "new_adaptive_two_lane_mcts": 1,
        "cached_deterministic_continuation": 1,
        "high_confidence_frozen_direct": 1,
    }
    assert receipt["resource_memory_startup_and_throughput_preflight_passed"] is False
    assert receipt["cuda_runtime_before_search"] == {
        "schema": preflight.CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": preflight.CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "cpu",
        "telemetry_complete": True,
        "error_types": [],
    }
    high_direct = receipt["r240_hybrid_decision_preflight"]["synthetic_high_confidence_direct"]
    assert high_direct["selected_factorized_stage_probabilities"][0] == 0.80
    assert high_direct["mcts_child_started_for_this_decision"] is False
    assert high_direct["mcts_search_call_count"] == 0
    assert high_direct["mcts_model_call_count"] == 0
    assert high_direct["mcts_simulator_call_count"] == 0
    assert high_direct["history_only_existing_child_journal_count"] == 1
    ambiguous = receipt["r240_hybrid_decision_preflight"]["synthetic_ambiguous_two_lane_mcts"]
    assert ambiguous["selected_factorized_stage_probabilities"][0] == 0.799
    assert ambiguous["per_lane_search_id_chains"] == [[0], [0]]
    assert len(ambiguous["per_lane_handle_first_search_id_composites"]) == 2
    boundary = receipt["r240_hybrid_decision_preflight"]["actor_change_end_turn_boundary"]
    assert boundary["declared_opponent_actor_leaf_count"] == 1
    assert boundary["search_steps_beyond_boundary"] == 0
    assert boundary["opponent_actor_leaves"][0]["expanded_child_count"] == 0
    assert receipt["r240_hybrid_decision_preflight"]["full_game_cumulative"][
        "decision_events"
    ][1]["crossed_actor_change_end_turn_boundary"] is False
    assert receipt["kaggle_api_called"] is False
    assert inputs.receipt_path.read_bytes() == preflight.canonical_json(receipt)


def test_preflight_accepts_a_measured_hidden_cuda_device_without_resource_inference(
    tmp_path: Path,
) -> None:
    """GPU visibility is runtime evidence, not a contradiction of Phase-1 limits."""

    inputs, limits, probe, _stage = _fixture(tmp_path)
    observation = probe["observed_resource_probe"]["cuda_runtime_before_search"]
    observation.update(
        {
            "cuda_available": True,
            "cuda_initialized": True,
            "device_count": 1,
            "devices": [
                {
                    "device_index": 0,
                    "device_name": "Mock Kaggle CUDA Device",
                    "total_memory_bytes": 16 * 1024**3,
                    "free_memory_bytes": 12 * 1024**3,
                }
            ],
            "model_device": "cuda:0",
        }
    )

    receipt = preflight.run_r235_phase1_preflight(
        inputs=inputs,
        limits=limits,
        dry_run=True,
        offline_probe_payload=probe,
        offline_exports=REQUIRED_EXPORTS,
    )

    cuda = receipt["cuda_runtime_before_search"]
    assert cuda["cuda_available"] is True
    assert cuda["device_count"] == 1
    assert cuda["devices"][0]["device_name"] == "Mock Kaggle CUDA Device"
    assert receipt["cuda_available_on_probe_host"] is True
    assert receipt["cuda_device_count_on_probe_host"] == 1
    assert receipt["observed_resource_probe"]["phase1_target"] == (
        preflight.PHASE1_MANIFEST_RESOURCE_BOUNDS
    )


def test_preflight_rejects_incomplete_or_inconsistent_cuda_observation(
    tmp_path: Path,
) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    observation = probe["observed_resource_probe"]["cuda_runtime_before_search"]
    observation.update(
        {
            "cuda_available": True,
            "cuda_initialized": True,
            "device_count": 1,
            "devices": [
                {
                    "device_index": 0,
                    "device_name": "Mock CUDA Device",
                    "total_memory_bytes": 1024,
                    "free_memory_bytes": 1025,
                }
            ],
            "model_device": "cuda:0",
        }
    )

    with pytest.raises(preflight.R235PreflightFailure, match="memory range"):
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )


def test_legacy_eight_lane_manifest_fails_closed_and_preserves_failure_receipt(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path, manifest_lane_count=8)
    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "lane count" in str(exc_info.value)
    failure = json.loads(inputs.receipt_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["passed"] is False
    assert failure["phase1_resources"]["vcpus"] == 2


def test_thread_oversubscription_is_a_hard_preflight_failure(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    probe["observed_resource_probe"]["runtime"]["worker_thread_count"] = 3
    probe["observed_resource_probe"]["runtime"]["observed_peak_worker_threads"] = 3

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "oversubscribed" in str(exc_info.value)
    assert json.loads(inputs.receipt_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_r240_rejects_legacy_fixed_eight_second_child_window(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    hybrid = probe[preflight.R240_HYBRID_PROBE_KEY]
    hybrid["synthetic_ambiguous_two_lane_mcts"]["child_search_budget_seconds"] = 8.0

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "child search budget" in str(exc_info.value)
    assert json.loads(inputs.receipt_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_r242_rejects_stale_or_relaxed_threshold_configurations(tmp_path: Path) -> None:
    for threshold in (0.90, 0.75):
        inputs, limits, probe, _stage = _fixture(tmp_path / str(threshold))
        probe[preflight.R240_HYBRID_PROBE_KEY]["configuration"]["high_confidence_threshold"] = threshold

        with pytest.raises(preflight.R235PreflightFailure) as exc_info:
            preflight.run_r235_phase1_preflight(
                inputs=inputs,
                limits=limits,
                dry_run=True,
                offline_probe_payload=probe,
                offline_exports=REQUIRED_EXPORTS,
            )

        assert "configuration" in str(exc_info.value)
        assert json.loads(inputs.receipt_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_r242_rejects_direct_path_search_or_opponent_boundary_expansion(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path / "direct-search")
    probe[preflight.R240_HYBRID_PROBE_KEY]["synthetic_high_confidence_direct"][
        "mcts_model_call_count"
    ] = 1

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "mcts_model_call_count" in str(exc_info.value)

    inputs, limits, probe, _stage = _fixture(tmp_path / "actor-boundary")
    probe[preflight.R240_HYBRID_PROBE_KEY]["actor_change_end_turn_boundary"][
        "opponent_actor_leaves"
    ][0]["expanded_legal_action_count"] = 1

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "expanded_legal_action_count" in str(exc_info.value)


def test_r244_rejects_duplicate_handle_scoped_search_id_composite(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    probe[preflight.R240_HYBRID_PROBE_KEY]["synthetic_ambiguous_two_lane_mcts"][
        "per_lane_handle_identities"
    ] = [1001, 1001]

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "distinct raw handles" in str(exc_info.value)


def test_r242_rejects_continuation_crossing_actor_change_end_turn_boundary(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    continuation = probe[preflight.R240_HYBRID_PROBE_KEY]["full_game_cumulative"][
        "decision_events"
    ][1]
    continuation["crossed_actor_change_end_turn_boundary"] = True

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "actor-change/end-turn boundary crossing" in str(exc_info.value)


def test_r240_rejects_repeated_search_for_an_exact_planned_same_turn_prompt(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    events = probe[preflight.R240_HYBRID_PROBE_KEY]["full_game_cumulative"]["decision_events"]
    events[0]["canonical_observation_fingerprint"] = "next-fingerprint"

    with pytest.raises(preflight.R235PreflightFailure) as exc_info:
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert "repeated a full MCTS search" in str(exc_info.value)
    assert json.loads(inputs.receipt_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_r246_rejects_a_nonwinning_or_stale_terminal_proof(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    terminal = probe[preflight.R240_HYBRID_PROBE_KEY][
        preflight.R246_TERMINAL_WIN_PROBE_KEY
    ]
    terminal["terminal_win_proof"]["terminal_winner_seat"] = 1

    with pytest.raises(preflight.R235PreflightFailure, match="terminal_winner_seat"):
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="production exact-child RSS evidence requires Linux /proc",
)
def test_actual_exact_child_preflight_publishes_all_derived_gate_receipts(
    tmp_path: Path,
) -> None:
    """Exercise immutable fan-out from one real, short exact child on Linux.

    The child emits a test payload solely to exercise publication plumbing;
    the receipt is still bound to the fresh child the watchdog created and is
    never a substitute for the package-external real-stage probe CLI.
    """

    inputs, limits, probe, _stage = _fixture(tmp_path)
    probe["observed_resource_probe"]["child_peak_rss_bytes"] = 512 * 1024**2
    gate_dir = tmp_path / "actual-gates"
    gate_dir.mkdir()
    inputs = preflight.R235PreflightInputs(
        stage_dir=inputs.stage_dir,
        archive_path=inputs.archive_path,
        manifest_path=inputs.manifest_path,
        expected_archive_sha256=inputs.expected_archive_sha256,
        expected_manifest_sha256=inputs.expected_manifest_sha256,
        receipt_path=inputs.receipt_path,
        r225_contract_path=inputs.r225_contract_path,
        r236_contract_path=inputs.r236_contract_path,
        entrypoint_relative_path=inputs.entrypoint_relative_path,
        actual_gate_receipts_dir=gate_dir,
    )
    command = [
        sys.executable,
        "-c",
        "import json, time; print(json.dumps(" + repr(probe) + ")); time.sleep(0.12)",
    ]

    class ExportHandle:
        pass

    for export in REQUIRED_EXPORTS:
        setattr(ExportHandle, export, object())

    receipt = preflight.run_r235_phase1_preflight(
        inputs=inputs,
        limits=limits,
        probe_command=command,
        native_loader=lambda _path: ExportHandle(),
    )

    assert receipt["status"] == "passed"
    assert receipt["probe_execution"]["execution_mode"] == "exact_child"
    assert receipt["cuda_runtime_before_search"]["telemetry_complete"] is True
    expected_names = {
        "phase1_resource": "phase1_submission_resource_and_archive_limit_receipt",
        "two_lane_topology": "two_lane_shared_tree_topology_and_receipt_schema_regression_receipt",
        "high_confidence": "high_confidence_direct_and_adaptive_bounded_mcts_regression_receipt",
        "terminal_win": "proven_deterministic_terminal_win_this_turn_regression_receipt",
        "deterministic_continuation": "deterministic_continuation_regression_receipt",
    }
    assert set(receipt["actual_gate_receipt_paths"]) == set(expected_names)
    source_sha = _sha(inputs.receipt_path)
    for gate, receipt_name in expected_names.items():
        path = Path(receipt["actual_gate_receipt_paths"][gate])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["receipt_name"] == receipt_name
        assert payload["status"] == "passed"
        assert payload["immutable"] is True
        assert payload["write_once"] is True
        assert payload["source_resource_preflight_receipt_sha256"] == source_sha
        assert path.stat().st_mode & 0o222 == 0
    terminal_gate = json.loads(
        Path(receipt["actual_gate_receipt_paths"]["terminal_win"]).read_text(encoding="utf-8")
    )
    assert terminal_gate[
        "all_owned_lane_resources_reservations_and_child_cleanup_complete"
    ] is True
    assert terminal_gate["search_release_calls"] == 2


def test_receipt_creation_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    inputs, limits, probe, _stage = _fixture(tmp_path)
    first = preflight.run_r235_phase1_preflight(
        inputs=inputs,
        limits=limits,
        dry_run=True,
        offline_probe_payload=probe,
        offline_exports=REQUIRED_EXPORTS,
    )
    body = inputs.receipt_path.read_bytes()

    with pytest.raises(preflight.ImmutableReceiptError):
        preflight.run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            dry_run=True,
            offline_probe_payload=probe,
            offline_exports=REQUIRED_EXPORTS,
        )

    assert inputs.receipt_path.read_bytes() == body == preflight.canonical_json(first)
