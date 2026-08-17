from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import stat
import tarfile
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_r235_r236_immutable_replacement_binding.py"
SUPPORT_SCRIPT_PATH = ROOT / "scripts/r235_direct_submission_tooling.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r235_r236_binding_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_support_module():
    spec = importlib.util.spec_from_file_location(
        "r235_direct_submission_tooling_test", SUPPORT_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # Gate fixtures model immutable files.  A test that intentionally changes
    # their JSON temporarily reopens its own fixture, then restores its prior
    # immutable mode so semantic validation remains independent of the new
    # binding read-side permission gate.
    original_mode: int | None = None
    if path.exists() and not path.is_symlink():
        original_mode = stat.S_IMODE(path.stat().st_mode)
        if not original_mode & 0o222:
            path.chmod(original_mode | stat.S_IWUSR)
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if original_mode is not None and not original_mode & 0o222:
            path.chmod(original_mode)


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in sorted(files.items()):
            info = tarfile.TarInfo(name="./" + name)
            info.size = len(body)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(body))


def _fake_members() -> tuple[dict[str, dict[str, object]], dict[str, bytes]]:
    bodies = {
        "linux_x86_64": b"official-r240-linux-libcg",
        "linux_aarch64": b"official-r240-linux-arm-libcg",
        "macos_arm64": b"official-r240-macos-libcg",
        "windows_x86_64": b"official-r240-windows-libcg",
    }
    paths = {
        "linux_x86_64": ("cg/libcg.so", "ELF test"),
        "linux_aarch64": ("cg/libcg-arm64.so", "ELF arm test"),
        "macos_arm64": ("cg/libcg.dylib", "Mach-O test"),
        "windows_x86_64": ("cg/cg.dll", "PE test"),
    }
    members: dict[str, dict[str, object]] = {}
    for platform, body in bodies.items():
        package_path, fmt = paths[platform]
        members[platform] = {
            "wheel_member": "kaggle_environments/envs/cabt/" + package_path,
            "package_relative_path": package_path,
            "sha256": _sha_bytes(body),
            "size_bytes": len(body),
            "format": fmt,
        }
    return members, bodies


def _r236_contract(module: Any, members: dict[str, dict[str, object]]) -> dict[str, Any]:
    return {
        "schema": module.R236_SCHEMA,
        "typed_source": module.CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH,
        "owner_decision_revision": 236,
        "upstream_provenance": {
            "package_version": module.OFFICIAL_PACKAGE_VERSION,
            "wheel_filename": module.OFFICIAL_WHEEL_FILENAME,
            "wheel_sha256": module.OFFICIAL_WHEEL_SHA256,
        },
        "canonical_native_libraries": members,
        "required_native_exports": list(module.REQUIRED_NATIVE_EXPORTS),
        "scope": {
            "r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary": True,
            "all_four_platform_native_members_must_be_bound": True,
            "mixed_old_and_new_library_sets_allowed": False,
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
        },
        "r225_package_preflight": {
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
            "all_four_canonical_native_members_checksum_and_size_verified": True,
            "old_or_mixed_native_members_rejected": True,
            "required_native_exports": list(module.REQUIRED_NATIVE_EXPORTS),
        },
        "authority": {
            "managed_training_or_evaluation_service_restart_authorized": False,
            "training_or_gradient_updates_authorized": False,
            "additional_kaggle_upload_retry_copy_or_queue_authorized": False,
        },
    }


def _r225_contract(module: Any, linux: dict[str, object]) -> dict[str, Any]:
    required_gate_names = [
        value for key, value in module.GATE_NAMES.items() if key != "go_first"
    ]
    return {
        "schema": module.R225_SCHEMA,
        "owner_decision_revision": module.R246_OWNER_DECISION_REVISION,
        "owner_kaggle_replacement_diagnostic_revision": 235,
        "owner_canonical_libcg_revision": 236,
        "owner_phase1_submission_resources_and_two_lane_revision": 238,
        "owner_hybrid_confidence_bounded_mcts_revision": 240,
        "owner_high_confidence_frozen_direct_threshold_revision": 242,
        "owner_handle_scoped_search_id_revision": module.R244_OWNER_DECISION_REVISION,
        "owner_proven_deterministic_terminal_win_this_turn_revision": (
            module.R246_OWNER_DECISION_REVISION
        ),
        "relationship_to_existing_work": {
            "r240_supersedes_only_the_new_r235_replacement_package_decision_scheduling_and_does_not_modify_r229_or_bo1000": True,
            "r242_supersedes_only_the_r240_high_confidence_frozen_direct_threshold_for_the_new_r235_replacement_package": True,
            "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible": True,
            "r242_does_not_modify_r234_r236_r238_r235_continuation_or_r229_bo1000": True,
            "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229": True,
            "r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries": True,
            "r246_supersedes_only_ambiguous_r235_mcts_root_selection_after_a_valid_deterministic_terminal_win_this_turn_proof": True,
            "r246_does_not_change_r242_high_confidence_direct_before_child_or_any_r229_bo1000_lifecycle": True,
        },
        "canonical_libcg_revision": {
            "typed_source": module.CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH,
            "official_wheel_sha256": module.OFFICIAL_WHEEL_SHA256,
            "linux_x86_64_sha256": linux["sha256"],
            "linux_x86_64_size_bytes": linux["size_bytes"],
            "new_r235_package_must_overlay_the_exact_official_linux_binary": True,
            "new_package_may_retain_the_old_frozen_r195_libcg_member": False,
            "saved_episode_and_full_game_gates_must_be_reissued_on_the_new_binary": True,
        },
        "exact_frozen_base": {
            "r195_bundle_sha256": module.R195_BUNDLE_SHA256,
            "r195_checkpoint_sha256": module.R195_CHECKPOINT_SHA256,
            "r195_matchup_tree_sha256": module.R195_MATCHUP_TREE_SHA256,
            "stock_libcg_sha256": linux["sha256"],
            "stock_libcg_size_bytes": linux["size_bytes"],
        },
        "complete_ordered_action_space_contract": {
            "complete_ordered_legal_action_ceiling": module.COMPLETE_ACTION_CAP,
            "ceiling_applies_at_root_and_every_private_leaf": True,
            "sampling_pruning_or_reinterpretation_of_legal_choices_allowed": False,
            "root_over_cap_behavior": "hard_fail_nonzero",
            "private_leaf_over_cap_behavior": (
                "contained_degraded_parent_direct_fallback_only_after_validated_"
                "precomputed_root_direct_action_and_exact_child_reap"
            ),
        },
        "phase1_submission_environment": {
            "owner_decision_revision": 238,
            **module.PHASE1_RESOURCES,
            "resource_probe_and_archive_size_receipt_required": True,
            "resource_mismatch_or_archive_over_limit_behavior": "hard_fail_closed_and_do_not_upload",
            "gpu_or_os_python_environment_is_not_inferred_from_the_reported_submission_resource_values": True,
        },
        "local_preflight": {
            "parent_precomputes_and_validates_exact_frozen_r195_direct_action_on_complete_ordered_root_legal_set": True,
            "parent_records_direct_action_legal_fingerprint_before_mcts_child_start": True,
            "mcts_child_owns_its_own_checksum_identical_frozen_r195_model_one_stock_dso_one_logical_tree_and_exactly_two_arenas_when_started_for_ambiguous_mcts": True,
            "required_simulator_search_lane_count": 2,
            "required_internal_agent_start_simulator_search_arena_count_per_child": 2,
            "required_search_begin_call_count_per_ambiguous_mcts_decision": 2,
            "required_distinct_internal_agent_start_handle_identity_count": 2,
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
            "per_lane_handle_scoped_search_id_chains_required": True,
            "one_lane_baseline_or_ratio_comparison_allowed": False,
            "r238_two_lane_receipt_contract": {
                "normal_mcts_decision_exact_counts": {
                    "requested_simulator_lane_count": 2,
                    "active_simulator_lane_count": 2,
                    "arena_count": 2,
                    "unique_handle_count": 2,
                    "distinct_handle_identity_count": 2,
                    "distinct_handle_scoped_first_search_id_composite_state_count": 2,
                    "search_begin_calls": 2,
                    "search_end_calls": 2,
                },
                "normal_mcts_decision_minimum_counts": {"search_release_calls": 2},
                "normal_mcts_decision_lane_ids": [0, 1],
                "normal_mcts_decision_required_fields": [
                    "requested_simulator_lane_count",
                    "active_simulator_lane_count",
                    "arena_count",
                    "unique_handle_count",
                    "per_lane_handle_identities",
                    "per_lane_first_search_ids",
                    "handle_scoped_first_search_id_composite_states",
                    "search_begin_calls",
                    "search_release_calls",
                    "search_end_calls",
                    "per_lane_depth",
                    "per_lane_search_id_chains",
                    "microbatch_sizes",
                    "max_simulator_calls_in_flight",
                    "outstanding_virtual_loss",
                ],
                "normal_mcts_decision_per_lane_vectors_exact_length": {
                    "per_lane_depth": 2,
                    "per_lane_search_id_chains": 2,
                    "per_lane_handle_identities": 2,
                    "per_lane_first_search_ids": 2,
                    "handle_scoped_first_search_id_composite_states": 2,
                },
                "normal_mcts_decision_microbatch_size_range": [1, 2],
                "normal_mcts_decision_max_simulator_calls_in_flight_range": [1, 2],
                "search_id_identity_contract": module.R244_HANDLE_SCOPED_SEARCH_IDENTITY,
                "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
            },
            "full_gameplay_shared_tree_contract": {
                "retain_exact_lane_handle_identity_search_id_tuple_across_repeated_depth_waves": True,
                "per_lane_handle_scoped_search_id_chains_required": True,
                "two_distinct_handle_identity_first_search_id_composite_states_required": True,
                "global_raw_search_id_integer_distinctness_required": False,
                "r246_proven_deterministic_terminal_win_this_turn_is_an_in_search_early_stop_not_a_r242_direct_bypass": True,
                "r246_valid_terminal_win_proof_may_bypass_only_the_normal_adaptive_stop_thresholds_after_two_lane_initialization": True,
                "r246_valid_terminal_win_proof_requires_at_least_one_exact_terminal_backup_but_not_the_normal_eight_backup_leader_or_both_lane_progress_threshold": True,
            },
            "r240_hybrid_scheduler": {
                **module.R242_MANIFEST_SCHEDULER,
                "high_confidence_journal_required_fields": [
                    "mode",
                    "direct_action",
                    "legal_order_fingerprint",
                    "selected_factorized_stage_probabilities",
                    "selected_factorized_stage_probability_threshold",
                    "all_selected_factorized_stages_meet_threshold",
                    "mcts_child_started_for_this_decision",
                    "mcts_select_call_count",
                    "history_only_existing_child_journal_count",
                    "degraded",
                ],
                "ambiguous_mcts_receipt_required_fields": [
                    "mode",
                    "confidence_classification",
                    "selected_factorized_stage_probabilities",
                    "selected_factorized_stage_probability_threshold",
                    "mcts_child_started",
                    "mcts_child_call_count",
                    "child_search_hard_seconds",
                    "parent_action_hard_seconds",
                    "completed_backups",
                    "deterministic_root_leader_observations",
                    "both_lanes_progressed",
                    "adaptive_early_stop_qualified",
                    "hard_completed_backup_stop",
                    "actor_change_boundary_leaf_count",
                    "chance_boundary_leaf_count",
                    "boundary_leaf_count",
                    "stop_reason",
                    "zero_backup_precomputed_direct_fallback",
                    "terminal_win_proof",
                ],
            },
            "deterministic_continuation": module.MANIFEST_DETERMINISTIC_CONTINUATION,
        },
        "replacement_kaggle_diagnostic": {
            "revision": 235,
            "phase1_two_lane_resource_revision": 238,
            "hybrid_confidence_bounded_mcts_revision": 240,
            "handle_scoped_search_id_correction_revision": module.R244_OWNER_DECISION_REVISION,
            "proven_deterministic_terminal_win_this_turn_revision": (
                module.R246_OWNER_DECISION_REVISION
            ),
            "replacement_submission_count_limit": 1,
            "replacement_submission_consumed": False,
            "competition": module.COMPETITION,
            "submission_message_required_literal": module.R235_LABEL,
            "submission_message_must_be_unique_and_exact": True,
            "complete_ordered_legal_action_ceiling": module.COMPLETE_ACTION_CAP,
            "all_required_local_gates_and_immutable_binding_must_pass_before_direct_api_upload": True,
            "kaggle_api_call_permitted_now_before_gates": False,
            "kaggle_upload_permitted_now_before_gates": False,
            "queue_or_batch_submission_allowed": False,
            "automatic_retry_allowed": False,
            "automatic_copy_or_resubmission_allowed": False,
            "second_upload_allowed": False,
            "phase1_submission_environment": {
                **module.PHASE1_RESOURCES,
                "resource_probe_and_archive_size_receipt_required": True,
            },
            "phase1_simulator_search": {
                "exactly_two_simulator_search_lanes_required": True,
                "required_simulator_search_lane_count": 2,
                "required_active_simulator_search_lane_count": 2,
                "required_internal_agent_start_simulator_search_arena_count": 2,
                "required_unique_raw_handle_count": 2,
                "required_search_begin_call_count": 2,
                "required_distinct_per_lane_handle_identity_count": 2,
                "per_lane_handle_scoped_search_id_chains_required": True,
                "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
                "globally_distinct_raw_search_id_integers_required": False,
                "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
                "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
                "first_search_id_identity_composite": "(handle_identity, first_search_id)",
                "maximum_frontier_leaves_per_frozen_evaluator_batch": 2,
                "one_shared_logical_mcts_tree_required": True,
                "one_lane_serial_fallback_eight_lane_topology_or_partial_lane_mcts_authority_allowed": False,
            },
                "r240_hybrid_scheduler": {
                "high_confidence_frozen_direct_mode": "high_confidence_frozen_direct",
                "high_confidence_frozen_direct_threshold_owner_revision": 242,
                "selected_factorized_stage_probability_threshold": 0.80,
                "all_selected_factorized_stages_must_be_finite_and_greater_than_or_equal_to_threshold": True,
                "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
                "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
                "complete_legal_precomputed_direct_action_required": True,
                "high_confidence_mcts_child_started_for_this_decision": False,
                "high_confidence_mcts_select_search_model_or_simulator_call_allowed": False,
                "high_confidence_existing_child_history_only_note_direct_action_ipc_allowed_and_required_when_child_exists": True,
                "high_confidence_existing_child_history_only_note_direct_action_ipc_count_range": [0, 1],
                "high_confidence_history_only_note_direct_action_ipc_must_not_invoke_mcts_select_search_model_or_simulator": True,
                "high_confidence_direct_is_a_permitted_new_mcts_search_bypass_for_a_branching_prompt": True,
                "high_confidence_direct_journal_required_and_not_degraded": True,
                "high_confidence_receipt_required_values": {
                    "selected_factorized_stage_probability_threshold": 0.80,
                    "all_selected_factorized_stages_meet_threshold": True,
                    "mcts_child_started_for_this_decision": False,
                    "mcts_select_call_count": 0,
                    "history_only_existing_child_journal_count_range": [0, 1],
                    "degraded": False,
                },
                "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
                    "ambiguous_mcts_exact_lane_count": 2,
                    "two_lane_mcts_topology_backup_and_stop_contract_applies_only_when_confidence_routes_to_ambiguous_mcts": True,
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
                "r246_proven_deterministic_terminal_win_this_turn": (
                    module.R246_REPLACEMENT_TERMINAL_WIN_THIS_TURN
                ),
                "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
                "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
                "zero_backups_use_only_existing_precomputed_direct_fallback_contract": True,
                "historical_fixed_eight_second_r228_branching_window_is_not_current": True,
                "stop_reason_fields": [
                    "high_confidence_frozen_direct",
                    "deterministic_continuation_plan",
                    module.R246_TERMINAL_WIN_STOP_REASON,
                    "adaptive_early_stop",
                    "hard_completed_backup_stop",
                    "child_search_hard_deadline",
                    "parent_action_hard_deadline",
                    "zero_backup_precomputed_direct_fallback",
                    "contained_child_fault",
                ],
            },
                "deterministic_continuation": {
                "maximum_depth": 8,
                "current_canonical_observation_fingerprint_must_exactly_match": True,
                "current_actor_must_remain_our_seat": True,
                "next_planned_action_must_be_in_current_complete_ordered_legal_actions": True,
                "plan_extraction_requires_both_lanes_same_fingerprint_and_backed_leader_agreement": True,
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
                "any_mismatch_clears_entire_plan_and_routes_to_normal_high_confidence_direct_or_adaptive_mcts": True,
                "parent_computes_exact_direct_first_and_rewrites_history_to_actual_planned_action": True,
                    "journal_exactly_once_and_log_planned_vs_direct": True,
                    "journal_required_fields": list(
                        module.MANIFEST_DETERMINISTIC_CONTINUATION[
                            "journal_required_fields"
                        ]
                    ),
                },
            "required_local_gate_receipts": required_gate_names,
        },
        "authority": {
            "kaggle_api_call_permitted_now_before_preconditions": False,
            "kaggle_upload_permitted_now_before_preconditions": False,
            "kaggle_api_call_permitted_now": False,
            "kaggle_upload_permitted_now": False,
            "kaggle_queue_submission_permitted": False,
            "automatic_kaggle_submission_allowed": False,
            "kaggle_retry_or_copy_permitted": False,
            "second_kaggle_upload_permitted": False,
            "training_or_gradient_updates_authorized": False,
            "selector_change_authorized": False,
            "promotion_authorized": False,
        },
    }


def _common_receipt(module: Any, name: str, identity: dict[str, object]) -> dict[str, Any]:
    return {
        "schema": module.PREFLIGHT_RECEIPT_SCHEMA,
        "receipt_name": name,
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        **identity,
    }


def _make_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_go_first_sidecar: bool = True,
) -> tuple[Any, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    module = _load_module()
    members, bodies = _fake_members()
    monkeypatch.setattr(module, "OFFICIAL_LIBCG_MEMBERS", members)
    frozen_bundle = b"synthetic-frozen-r195-bundle"
    frozen_checkpoint = b"synthetic-frozen-r195-checkpoint"
    frozen_matchup_tree = b'{"synthetic":"frozen-r195-matchup-tree"}\n'
    monkeypatch.setattr(module, "R195_BUNDLE_SHA256", _sha_bytes(frozen_bundle))
    monkeypatch.setattr(
        module, "R195_CHECKPOINT_SHA256", _sha_bytes(frozen_checkpoint)
    )
    monkeypatch.setattr(
        module, "R195_MATCHUP_TREE_SHA256", _sha_bytes(frozen_matchup_tree)
    )

    r236 = tmp_path / "canonical-libcg-r236.json"
    _write_json(r236, _r236_contract(module, members))
    r225 = tmp_path / "r225-r246.json"
    _write_json(r225, _r225_contract(module, members["linux_x86_64"]))
    monkeypatch.setattr(module, "CANONICAL_R225_R240_TYPED_CONTRACT_PATH", r225)
    monkeypatch.setattr(
        module, "CANONICAL_R225_R240_TYPED_CONTRACT_SHA256", _sha_file(r225)
    )
    monkeypatch.setattr(module, "CANONICAL_R236_TYPED_CONTRACT_PATH", r236)
    monkeypatch.setattr(module, "CANONICAL_R236_TYPED_CONTRACT_SHA256", _sha_file(r236))

    main = b"def agent(observation):\n    return [0]\n"
    direct_main = b'''def agent(observation):
    select = observation.get("select") if isinstance(observation, dict) else None
    if not isinstance(select, dict) or select.get("context") not in (41, "IS_FIRST", "IsFirst"):
        return [0]
    for index, option in enumerate(select.get("option") or []):
        if isinstance(option, dict) and option.get("type") in (1, "Yes"):
            return [index]
    return [0]
'''
    turn_order_profile = module.canonical_json(
        {
            "schema": "poke_bot.submission_turn_order_profile/v1",
            "turn_order_preference": "first_if_allowed",
        }
    )
    manifest_member = "r238_two_lane_bounded_mcts_manifest.json"
    manifest_path = tmp_path / manifest_member
    manifest = {
        "schema": module.R238_PACKAGE_MANIFEST_SCHEMA,
        "role": module.R238_PACKAGE_MANIFEST_ROLE,
        "r225_typed_contract": {
            "path": module.CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH,
            "schema": module.R225_SCHEMA,
            "sha256": _sha_file(r225),
        },
        "required_label": module.R235_LABEL,
        "complete_action_cap": module.COMPLETE_ACTION_CAP,
        "lane_count": 2,
        "required_search_lifecycle_counts": {
            "search_begin_calls": 2,
            "search_end_calls": 2,
            "search_release_calls": 2,
        },
        "phase1_kaggle_resource_bounds": module.PHASE1_MANIFEST_RESOURCE_BOUNDS,
        "r240_hybrid_scheduler": module.R240_MANIFEST_SCHEDULER,
        "deterministic_continuation": module.MANIFEST_DETERMINISTIC_CONTINUATION,
        "r240_required_preflight_receipts": [
            module.GATE_NAMES["high_confidence"],
            module.GATE_NAMES["deterministic_continuation"],
        ],
        "owner_proven_deterministic_terminal_win_this_turn_revision": (
            module.R246_OWNER_DECISION_REVISION
        ),
        "r246_proven_deterministic_terminal_win_this_turn": (
            module.R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN
        ),
        "r246_required_preflight_receipts": [module.GATE_NAMES["terminal_win"]],
        "entrypoint_sha256": _sha_bytes(main),
        "direct_entrypoint": {
            "path": "r195_direct_main.py",
            "sha256": _sha_bytes(direct_main),
        },
        "broker_contract": {
            "complete_action_cap": module.COMPLETE_ACTION_CAP,
            "degraded_fallback_marker": "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED",
            "search_seconds": 2.0,
            "action_timeout_seconds": 4.0,
        },
        "canonical_libcg_contract": _r236_contract(module, members),
        "canonical_native_members": module._expected_manifest_native_members(),
        "canonical_native_member_sha256": {
            member["package_relative_path"]: member["sha256"]
            for member in members.values()
        },
        "exact_frozen_base": {
            "r195_bundle_sha256": module.R195_BUNDLE_SHA256,
            "r195_checkpoint_sha256": module.R195_CHECKPOINT_SHA256,
            "r195_matchup_tree_sha256": module.R195_MATCHUP_TREE_SHA256,
            "stock_libcg_sha256": members["linux_x86_64"]["sha256"],
            "stock_libcg_size_bytes": members["linux_x86_64"]["size_bytes"],
        },
    }
    _write_json(manifest_path, manifest)
    archive = tmp_path / "candidate.tar.gz"
    archive_files = {
        "main.py": main,
        "r195_direct_main.py": direct_main,
        "turn_order_profile.json": turn_order_profile,
        manifest_member: manifest_path.read_bytes(),
        "model.pt": frozen_checkpoint,
        "matchup_tree.json": frozen_matchup_tree,
        module.CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH: r225.read_bytes(),
        "cg/libcg.so": bodies["linux_x86_64"],
        "cg/libcg-arm64.so": bodies["linux_aarch64"],
        "cg/libcg.dylib": bodies["macos_arm64"],
        "cg/cg.dll": bodies["windows_x86_64"],
    }
    _write_archive(archive, archive_files)

    identity: dict[str, object] = {
        "candidate_archive_sha256": _sha_file(archive),
        "candidate_archive_size_bytes": archive.stat().st_size,
        "member_manifest_sha256": _sha_file(manifest_path),
        "entrypoint_sha256": _sha_bytes(main),
        "r225_contract_sha256": _sha_file(r225),
        "canonical_libcg_contract_sha256": _sha_file(r236),
        "linux_x86_64_libcg_sha256": members["linux_x86_64"]["sha256"],
        "linux_x86_64_libcg_size_bytes": members["linux_x86_64"]["size_bytes"],
        "complete_ordered_action_cap": module.COMPLETE_ACTION_CAP,
        "simulator_search_lane_count": 2,
        "phase1_submission_environment": module.PHASE1_RESOURCES,
        "r240_hybrid_scheduler": module.R240_HYBRID_SCHEDULER,
        "deterministic_continuation": module.DETERMINISTIC_CONTINUATION,
    }

    focused = _common_receipt(module, module.GATE_NAMES["focused_fault"], identity)
    focused.update(
        {
            "focused_fault_suite_passed": True,
            "nonreaped_child_hard_fail_test_passed": True,
            "parent_returned_action_legality_hard_fail_test_passed": True,
            "fault_injected_full_game_degraded_marker_and_no_viability_credit_passed": True,
            "fault_classes_covered": [
                "timeout",
                "crash",
                "protocol",
                "evaluator",
                "native",
                "cleanup",
            ],
        }
    )
    saved = _common_receipt(module, module.GATE_NAMES["saved_episode"], identity)
    saved.update(
        {
            "source_submission_id": 55_416_396,
            "source_episode_id": 91_766_923,
            "seat": 0,
            "final_callback_step": 58,
            "final_callback_ordered_legal_action_count": 2,
            "legal_action_before_hard_deadline": True,
            "fault_injected_broker_child_reap_proved": True,
            "result_path": "contained_precomputed_parent_direct_fallback_after_exact_child_reap",
        }
    )
    full_game = _common_receipt(module, module.GATE_NAMES["full_game"], identity)
    full_game.update(
        {
            "exact_package_full_local_game_passed": True,
            "full_gameplay_loop_completed": True,
            "branching_gameplay_decision_count": 3,
            "explicit_success_marker_count": 1,
            "degraded_game_count": 0,
            "active_simulator_search_lane_count": 2,
        }
    )
    resource = _common_receipt(module, module.GATE_NAMES["resource"], identity)
    resource.update(
        {
            "resource_memory_startup_and_throughput_preflight_passed": True,
            "memory_preflight_passed": True,
            "startup_preflight_passed": True,
            "throughput_preflight_passed": True,
            "observed_resource_probe": {"hdd_gib": 11.8, "ram_gib": 12.2, "vcpus": 2},
            "startup_seconds": 0.2,
            "throughput_decisions_per_second": 1.0,
        }
    )
    phase1 = _common_receipt(module, module.GATE_NAMES["phase1_resource"], identity)
    phase1.update(
        {
            "phase1_submission_resource_and_archive_limit_receipt_passed": True,
            "resource_probe_matches_phase1_submission_environment": True,
            "archive_within_submission_limit": True,
            "observed_phase1_submission_environment": module.PHASE1_RECEIPT_ENVIRONMENT,
            "observed_submission_archive_size_bytes": archive.stat().st_size,
            "observed_submission_archive_size_mib": archive.stat().st_size / (1024 * 1024),
        }
    )
    topology = _common_receipt(module, module.GATE_NAMES["two_lane_topology"], identity)
    topology.update(
        {
            "two_lane_shared_tree_topology_and_receipt_schema_regression_passed": True,
            "receipt_schema_regression_passed": True,
            "one_shared_logical_mcts_tree_proved": True,
            "historical_eight_lane_manifest_or_receipt_accepted": False,
            "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "distinct_handle_identity_count": 2,
            "distinct_handle_scoped_first_search_id_composite_state_count": 2,
            "search_begin_calls": 2,
            "search_end_calls": 2,
            "search_release_calls": 2,
            "search_id_numeric_namespace": "per_distinct_agent_start_handle",
            "lane_ids": [0, 1],
            "per_lane_depth": [2, 2],
            "per_lane_handle_identities": [1001, 1002],
            "per_lane_search_id_chains": [[0], [0]],
            "per_lane_first_search_ids": [0, 0],
            "handle_scoped_first_search_id_composite_states": [
                {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
                {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
            ],
            "per_lane_handle_first_search_id_composites": [
                {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
                {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
            ],
            "microbatch_sizes": [2],
            "max_simulator_calls_in_flight": 2,
        }
    )
    handle_scoped = {
        "schema": module.R244_HANDLE_SCOPED_SEARCH_ID_RECEIPT_SCHEMA,
        "receipt_name": module.GATE_NAMES["handle_scoped_search_id"],
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        "r225_contract_sha256": _sha_file(r225),
        "canonical_libcg_contract_sha256": _sha_file(r236),
        "official_libcg_handle_scoped_search_id_identity_regression_passed": True,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "distinct_handle_identity_count": 2,
        "distinct_handle_scoped_first_search_id_composite_state_count": 2,
        "search_begin_calls": 2,
        "search_id_numeric_namespace": "per_distinct_agent_start_handle",
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        "r244_owner_revision": 244,
        "same_raw_first_search_id_on_distinct_handles_accepted": True,
        "duplicate_handle_identity_rejected": True,
        "duplicate_handle_scoped_first_search_id_composite_rejected": True,
        "per_lane_handle_identities": [1001, 1002],
        "per_lane_search_id_chains": [[0], [0]],
        "per_lane_first_search_ids": [0, 0],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
            {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
        ],
        "per_lane_handle_first_search_id_composites": [
            {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
            {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
        ],
    }
    high_confidence = _common_receipt(module, module.GATE_NAMES["high_confidence"], identity)
    high_confidence.update(
        {
            "high_confidence_direct_and_adaptive_bounded_mcts_regression_passed": True,
            "high_confidence_mode": "high_confidence_frozen_direct",
            "selected_factorized_stage_probabilities": [0.80, 0.99],
            "selected_factorized_stage_probability_threshold": 0.80,
            "all_selected_factorized_stages_meet_threshold": True,
            "high_confidence_path_returned_precomputed_legal_direct_action": True,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "mcts_search_call_count": 0,
            "mcts_model_call_count": 0,
            "mcts_simulator_call_count": 0,
            "history_only_existing_child_journal_count": 1,
            "degraded": False,
            "ambiguous_selected_stage_forced_mcts": True,
            "child_search_seconds": 2.0,
            "parent_action_deadline_seconds": 4.0,
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations": 3,
            "maximum_backups_per_decision": 32,
            "both_lanes_progressed": True,
            "stop_reason": "stable_root_leader",
            "legacy_fixed_eight_second_window_used": False,
            "completed_backups": 8,
            "same_root_leader_observations": 3,
            "actor_change_boundary_leaf_count": 1,
            "chance_boundary_leaf_count": 0,
            "boundary_leaf_count": 1,
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
        }
    )
    terminal_win = _common_receipt(module, module.GATE_NAMES["terminal_win"], identity)
    terminal_win.update(
        {
            "owner_proven_deterministic_terminal_win_this_turn_revision": (
                module.R246_OWNER_DECISION_REVISION
            ),
            "proven_deterministic_terminal_win_this_turn_regression_passed": True,
            "stop_reason": module.R246_TERMINAL_WIN_STOP_REASON,
            "two_lane_topology_initialized_before_terminal_win_override": True,
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "search_begin_calls": 2,
            "search_release_calls": 2,
            "search_end_calls": 2,
            "completed_root_backup_count": 1,
            "terminal_win_proof_count": 1,
            "proven_deterministic_terminal_win_this_turn_stop_count": 1,
            "terminal_win_proof_backed_up_into_shared_root_tree": True,
            "terminal_leaf_returned_by_exact_stock_simulator": True,
            "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
            module.R246_CLEANUP_COMPLETE_FIELD: True,
            "outstanding_virtual_loss": 0,
            "two_independent_lane_proofs_required": False,
            "exhaustive_legal_action_scan_required": False,
            "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
            "terminal_win_proof": {
                "proof_kind": module.R246_TERMINAL_WIN_PROOF_KIND,
                "root_observation_fingerprint": "sha256:" + "a" * 64,
                "root_legal_order_fingerprint": "sha256:" + "b" * 64,
                "root_actor_seat": 0,
                "root_action": [1, 0],
                "selected_action": [1, 0],
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
        }
    )
    continuation = _common_receipt(
        module, module.GATE_NAMES["deterministic_continuation"], identity
    )
    continuation.update(
        {
            "deterministic_continuation_regression_passed": True,
            "two_lane_agreed_exact_fingerprint_path_consumed": True,
            "valid_match_no_new_search": True,
            "valid_match_started_new_search": False,
            "valid_match_backed_action_consumed": True,
            "valid_match_same_root_actor": True,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "history_only_existing_child_journal_count": 1,
            "degraded": False,
            "configured_max_depth": 8,
            "observed_valid_match_depth": 8,
            "chance_disagreement_clears_entire_plan": True,
            "fingerprint_disagreement_clears_entire_plan": True,
            "action_disagreement_clears_entire_plan": True,
            "actor_disagreement_clears_entire_plan": True,
            "crossed_actor_change_end_turn_boundary": False,
            "precomputed_direct_action_and_history_correction_retained": True,
        }
    )
    go_first = {
        "schema": module.GO_FIRST_ATTESTATION_SCHEMA,
        "kind": module.GO_FIRST_VERIFIER_KIND,
        "status": "passed",
        "receipt_name": module.GATE_NAMES["go_first"],
        "passed": True,
        "immutable": True,
        "write_once": True,
        "go_first_contract_passed": True,
        "forced_yes_action_legal": True,
        "turn_order_preference": "first_if_allowed",
        "go_first_if_offered": True,
        "go_second_if_offered": False,
        "verified_cases": module.GO_FIRST_VERIFIED_CASES,
        "case_results": module.GO_FIRST_CASE_RESULTS,
        "file_sha256": identity["candidate_archive_sha256"],
        "file_bytes": identity["candidate_archive_size_bytes"],
        **{
            field: identity[field]
            for field in (
                "candidate_archive_sha256",
                "candidate_archive_size_bytes",
                "member_manifest_sha256",
                "entrypoint_sha256",
                "r225_contract_sha256",
                "canonical_libcg_contract_sha256",
                "linux_x86_64_libcg_sha256",
                "linux_x86_64_libcg_size_bytes",
                "complete_ordered_action_cap",
                "simulator_search_lane_count",
                "phase1_submission_environment",
                "r240_hybrid_scheduler",
                "deterministic_continuation",
            )
        },
        "submission": {"competition": module.COMPETITION, "message": module.R235_LABEL},
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": identity["member_manifest_sha256"],
            "member": manifest_member,
            "schema": module.R238_PACKAGE_MANIFEST_SCHEMA,
            "lane_count": 2,
        },
        "typed_contracts": {
            "r225_sha256": identity["r225_contract_sha256"],
            "r236_sha256": identity["canonical_libcg_contract_sha256"],
            "r225_owner_decision_revision": module.R246_OWNER_DECISION_REVISION,
        },
        "verified_at_utc": "2026-08-11T00:00:00+00:00",
    }
    receipt_paths: dict[str, Path] = {}
    for key, value in {
        "focused": focused,
        "saved": saved,
        "full_game": full_game,
        "resource": resource,
        "phase1": phase1,
        "topology": topology,
        "handle_scoped": handle_scoped,
        "high_confidence": high_confidence,
        "terminal_win": terminal_win,
        "continuation": continuation,
    }.items():
        path = tmp_path / f"{key}.json"
        _write_json(path, value)
        path.chmod(0o444)
        receipt_paths[key] = path
    go_first_path = Path(str(archive) + ".go-first-verified.json")
    if create_go_first_sidecar:
        _write_json(go_first_path, go_first)
        go_first_path.chmod(0o444)
    receipt_paths["go_first"] = go_first_path

    return module, {
        "archive": archive,
        "archive_files": archive_files,
        "manifest": manifest_path,
        "manifest_member": manifest_member,
        "r225": r225,
        "r236": r236,
        "receipts": receipt_paths,
    }


def _build(module: Any, values: dict[str, Any], output: Path, **overrides: Any) -> dict[str, Any]:
    receipts = values["receipts"]
    arguments: dict[str, Any] = {
        "candidate_archive": values["archive"],
        "member_manifest": values["manifest"],
        "member_manifest_member": values["manifest_member"],
        "focused_fault_receipt": receipts["focused"],
        "saved_episode_receipt": receipts["saved"],
        "full_game_receipt": receipts["full_game"],
        "resource_receipt": receipts["resource"],
        "phase1_resource_receipt": receipts["phase1"],
        "two_lane_topology_receipt": receipts["topology"],
        "handle_scoped_search_id_receipt": receipts["handle_scoped"],
        "high_confidence_receipt": receipts["high_confidence"],
        "terminal_win_receipt": receipts["terminal_win"],
        "deterministic_continuation_receipt": receipts["continuation"],
        "go_first_receipt": receipts["go_first"],
        "output": output,
        "r225_contract": values["r225"],
        "canonical_libcg_contract": values["r236"],
    }
    arguments.update(overrides)
    return module.build_binding(**arguments)


@pytest.mark.parametrize(
    "mode",
    (0o644, 0o464, 0o446),
    ids=("owner-writable-0644", "group-writable-0464", "other-writable-0446"),
)
def test_gate_receipt_read_side_rejects_any_legacy_write_bit(
    tmp_path: Path, mode: int
) -> None:
    """A JSON receipt cannot self-attest immutability while still writable."""

    module = _load_module()
    receipt = tmp_path / "legacy-mutable-gate.json"
    _write_json(receipt, {"status": "passed"})
    receipt.chmod(mode)

    with pytest.raises(module.R235R236BindingError, match="must not be writable"):
        module._read_and_validate_gate(
            receipt,
            receipt_name="legacy mutable gate",
            identity={},
            validator=lambda _payload, _identity: None,
            common_identity=False,
        )


def test_gate_receipt_read_side_accepts_mode_0444(tmp_path: Path) -> None:
    """The writers' exact sealed mode remains eligible to the binding reader."""

    module = _load_module()
    receipt = tmp_path / "sealed-gate.json"
    _write_json(receipt, {"status": "passed"})
    receipt.chmod(0o444)

    bound = module._read_and_validate_gate(
        receipt,
        receipt_name="sealed gate",
        identity={},
        validator=lambda _payload, _identity: None,
        common_identity=False,
    )

    assert bound == {
        "receipt_name": "sealed gate",
        "sha256": _sha_file(receipt),
        "passed": True,
    }


def test_binds_exact_r246_two_lane_archive_and_all_required_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    first_output = tmp_path / "binding-one.json"
    second_output = tmp_path / "binding-two.json"

    first = _build(module, values, first_output)
    second = _build(module, values, second_output)

    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()
    assert stat.S_IMODE(first_output.stat().st_mode) == 0o444
    assert first["candidate_package"]["archive_sha256"] == _sha_file(values["archive"])
    assert first["candidate_package"]["member_manifest_sha256"] == _sha_file(
        values["manifest"]
    )
    assert first["typed_contracts"]["r225_historical_r246_typed_contract"]["sha256"] == _sha_file(
        values["r225"]
    )
    assert first["owner_decision_revision"] == 246
    assert first["simulator_search_topology"]["lane_count"] == 2
    assert first["simulator_search_topology"]["historical_eight_lane_manifest_or_receipt_accepted"] is False
    assert first["r240_hybrid_scheduler"]["child_search_seconds"] == 2.0
    assert first["r240_hybrid_scheduler"]["parent_action_deadline_seconds"] == 4.0
    assert first["deterministic_continuation"]["max_depth"] == 8
    r244_topology = first["local_gate_receipts"]["two_lane_topology"][
        "validated_counter_projection"
    ]
    assert r244_topology["per_lane_first_search_ids"] == [0, 0]
    assert r244_topology["handle_scoped_first_search_id_composite_states"] == [
        {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
        {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
    ]
    assert first["r244_handle_scoped_search_id_identity"][
        "globally_distinct_raw_search_id_integers_required"
    ] is False
    boundary = first["local_gate_receipts"][
        "high_confidence_and_adaptive_bounded_mcts"
    ]["validated_counter_projection"]
    assert boundary["opponent_actor_leaf_count"] == 1
    assert boundary["expanded_child_count"] == 0
    assert boundary["opponent_action_cached_count"] == 0
    terminal_projection = first["local_gate_receipts"][
        "proven_deterministic_terminal_win_this_turn"
    ]["validated_counter_projection"]
    assert terminal_projection["completed_root_backup_count"] == 1
    assert (
        terminal_projection[module.R246_CLEANUP_COMPLETE_FIELD]
        is True
    )
    assert (
        module.R246_LEGACY_CLEANUP_COMPLETED_FIELD
        not in terminal_projection
    )
    assert terminal_projection["stop_reason"] == module.R246_TERMINAL_WIN_STOP_REASON
    assert terminal_projection["terminal_win_proof"]["discovering_lane_id"] == 0
    assert terminal_projection["terminal_win_proof"]["selected_action"] == [1, 0]
    assert set(terminal_projection["terminal_win_proof"]) == set(
        module.R246_TERMINAL_WIN_PROOF_FIELDS
    )
    assert first["r246_proven_deterministic_terminal_win_this_turn"] == (
        module.R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN
    )
    assert set(first["canonical_libcg_r236"]["members"]) == {
        "linux_x86_64",
        "linux_aarch64",
        "macos_arm64",
        "windows_x86_64",
    }
    assert set(first["local_gate_receipts"]) == {
        "focused_native_child_fault_suite",
        "saved_episode_91766923_step58",
        "exact_repaired_package_full_local_game",
        "resource_memory_startup_throughput",
        "phase1_resource_and_archive",
        "two_lane_topology",
        "official_libcg_handle_scoped_search_id_identity",
        "high_confidence_and_adaptive_bounded_mcts",
        "proven_deterministic_terminal_win_this_turn",
        "deterministic_continuation",
        "go_first",
    }
    assert first["authorization"]["kaggle_upload_permitted_now"] is False
    assert first["authorization"]["automatic_retry_allowed"] is False
    assert first["builder"] == {
        "network_accessed": False,
        "kaggle_api_called": False,
        "kaggle_queue_used": False,
        "kaggle_upload_used": False,
        "gpu_used": False,
        "service_modified": False,
        "bo1000_modified": False,
    }


def test_candidate_then_real_prebinding_go_first_sidecar_then_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the required order without letting the sidecar see a binding."""

    module, values = _make_inputs(
        tmp_path, monkeypatch, create_go_first_sidecar=False
    )
    support = _load_support_module()
    monkeypatch.setattr(support, "CANONICAL_R225_PATH", values["r225"])
    monkeypatch.setattr(
        support, "CANONICAL_R225_R246_SHA256", _sha_file(values["r225"])
    )
    monkeypatch.setattr(support, "CANONICAL_R236_PATH", values["r236"])
    monkeypatch.setattr(support, "CANONICAL_R236_SHA256", _sha_file(values["r236"]))

    sidecar = values["receipts"]["go_first"]
    assert not sidecar.exists()
    emitted = support.verify_go_first(
        archive_path=values["archive"],
        manifest_path=values["manifest"],
        manifest_member=values["manifest_member"],
        r225_contract=values["r225"],
        r236_contract=values["r236"],
        output_path=sidecar,
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert emitted["path"] == str(sidecar.resolve())
    assert emitted["sha256"] == _sha_file(sidecar)
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    assert set(payload) == {
        "schema",
        "kind",
        "status",
        "receipt_name",
        "passed",
        "immutable",
        "write_once",
        "go_first_contract_passed",
        "forced_yes_action_legal",
        "turn_order_preference",
        "go_first_if_offered",
        "go_second_if_offered",
        "verified_cases",
        "case_results",
        "file_sha256",
        "file_bytes",
        "candidate_archive_sha256",
        "candidate_archive_size_bytes",
        "member_manifest_sha256",
        "entrypoint_sha256",
        "r225_contract_sha256",
        "canonical_libcg_contract_sha256",
        "linux_x86_64_libcg_sha256",
        "linux_x86_64_libcg_size_bytes",
        "complete_ordered_action_cap",
        "simulator_search_lane_count",
        "phase1_submission_environment",
        "r240_hybrid_scheduler",
        "deterministic_continuation",
        "submission",
        "manifest",
        "typed_contracts",
        "verified_at_utc",
    }
    assert "r235_binding" not in payload

    binding = _build(module, values, tmp_path / "binding-after-go-first.json")
    assert binding["local_gate_receipts"]["go_first"] == {
        "receipt_name": module.GATE_NAMES["go_first"],
        "sha256": _sha_file(sidecar),
        "passed": True,
    }


def test_rejects_circular_or_unbound_prebinding_go_first_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    sidecar = values["receipts"]["go_first"]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.chmod(0o644)
    payload["r235_binding"] = {"sha256": "sha256:" + "0" * 64}
    _write_json(sidecar, payload)
    sidecar.chmod(0o444)
    with pytest.raises(module.R235R236BindingError, match="unexpected or missing field"):
        _build(module, values, tmp_path / "circular-sidecar.json")

    module, values = _make_inputs(tmp_path / "stale", monkeypatch)
    sidecar = values["receipts"]["go_first"]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.chmod(0o644)
    payload["candidate_archive_sha256"] = "sha256:" + "f" * 64
    _write_json(sidecar, payload)
    sidecar.chmod(0o444)
    with pytest.raises(module.R235R236BindingError, match="candidate_archive_sha256"):
        _build(module, values, tmp_path / "stale-sidecar.json")


def test_rejects_old_eight_lane_manifest_and_topology_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["lane_count"] = 8
    _write_json(values["manifest"], manifest)
    values["archive_files"][values["manifest_member"]] = values["manifest"].read_bytes()
    _write_archive(values["archive"], values["archive_files"])
    with pytest.raises(module.R235R236BindingError, match="lane_count"):
        _build(module, values, tmp_path / "manifest-rejected.json")

    module, values = _make_inputs(tmp_path / "receipt", monkeypatch)
    topology = json.loads(values["receipts"]["topology"].read_text(encoding="utf-8"))
    topology["requested_simulator_lane_count"] = 8
    _write_json(values["receipts"]["topology"], topology)
    output = tmp_path / "receipt-rejected.json"
    with pytest.raises(module.R235R236BindingError, match="requested_simulator_lane_count"):
        _build(module, values, output)
    assert not output.exists()

    module, values = _make_inputs(tmp_path / "composite", monkeypatch)
    topology = json.loads(values["receipts"]["topology"].read_text(encoding="utf-8"))
    topology["per_lane_handle_identities"] = [1001, 1001]
    _write_json(values["receipts"]["topology"], topology)
    with pytest.raises(module.R235R236BindingError, match="distinct handles"):
        _build(module, values, tmp_path / "composite-rejected.json")

    module, values = _make_inputs(tmp_path / "composite-projection", monkeypatch)
    topology = json.loads(values["receipts"]["topology"].read_text(encoding="utf-8"))
    topology["handle_scoped_first_search_id_composite_states"][1]["first_search_id"] = 1
    _write_json(values["receipts"]["topology"], topology)
    with pytest.raises(module.R235R236BindingError, match="handle-scoped first SearchId composites"):
        _build(module, values, tmp_path / "composite-projection-rejected.json")

    module, values = _make_inputs(tmp_path / "r244-receipt", monkeypatch)
    handle_scoped = json.loads(
        values["receipts"]["handle_scoped"].read_text(encoding="utf-8")
    )
    handle_scoped["handle_scoped_first_search_id_composite_states"][1][
        "handle_identity"
    ] = 1001
    _write_json(values["receipts"]["handle_scoped"], handle_scoped)
    with pytest.raises(module.R235R236BindingError, match="handle-scoped first SearchId composites"):
        _build(module, values, tmp_path / "r244-composite-rejected.json")

    module, values = _make_inputs(tmp_path / "flat-r240", monkeypatch)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["r240_hybrid_scheduler"] = module.R240_HYBRID_SCHEDULER
    _write_json(values["manifest"], manifest)
    values["archive_files"][values["manifest_member"]] = values["manifest"].read_bytes()
    _write_archive(values["archive"], values["archive_files"])
    with pytest.raises(module.R235R236BindingError, match="r240 scheduler"):
        _build(module, values, tmp_path / "flat-r240-rejected.json")


def test_rejects_saved_step_and_r240_fixed_eight_second_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    saved = json.loads(values["receipts"]["saved"].read_text(encoding="utf-8"))
    saved["final_callback_step"] = 57
    _write_json(values["receipts"]["saved"], saved)
    output = tmp_path / "saved-step-drift.json"
    with pytest.raises(module.R235R236BindingError, match="final_callback_step"):
        _build(module, values, output)
    assert not output.exists()

    module, values = _make_inputs(tmp_path / "r240", monkeypatch)
    high = json.loads(values["receipts"]["high_confidence"].read_text(encoding="utf-8"))
    high["child_search_seconds"] = 8.0
    _write_json(values["receipts"]["high_confidence"], high)
    with pytest.raises(module.R235R236BindingError, match="child_search_seconds"):
        _build(module, values, tmp_path / "fixed-eight-rejected.json")


def test_rejects_r242_legacy_threshold_and_actor_boundary_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["r240_hybrid_scheduler"]["selected_factorized_stage_probability_threshold"] = 0.90
    _write_json(values["manifest"], manifest)
    values["archive_files"][values["manifest_member"]] = values["manifest"].read_bytes()
    _write_archive(values["archive"], values["archive_files"])
    with pytest.raises(module.R235R236BindingError, match="selected_factorized_stage_probability_threshold"):
        _build(module, values, tmp_path / "legacy-threshold-rejected.json")

    module, values = _make_inputs(tmp_path / "actor-boundary", monkeypatch)
    high = json.loads(values["receipts"]["high_confidence"].read_text(encoding="utf-8"))
    boundary = high["actor_change_end_turn_boundary"]
    boundary["expanded_child_count"] = 1
    _write_json(values["receipts"]["high_confidence"], high)
    with pytest.raises(module.R235R236BindingError, match="expanded_child_count"):
        _build(module, values, tmp_path / "actor-boundary-rejected.json")


def test_rejects_terminal_win_proof_without_exact_boundary_free_two_lane_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    terminal = json.loads(
        values["receipts"]["terminal_win"].read_text(encoding="utf-8")
    )
    terminal["terminal_win_proof"]["path_no_chance_boundary"] = False
    _write_json(values["receipts"]["terminal_win"], terminal)
    output = tmp_path / "terminal-chance-boundary-rejected.json"
    with pytest.raises(module.R235R236BindingError, match="path_no_chance_boundary"):
        _build(module, values, output)
    assert not output.exists()

    module, values = _make_inputs(tmp_path / "topology", monkeypatch)
    terminal = json.loads(
        values["receipts"]["terminal_win"].read_text(encoding="utf-8")
    )
    terminal["active_simulator_lane_count"] = 1
    _write_json(values["receipts"]["terminal_win"], terminal)
    with pytest.raises(module.R235R236BindingError, match="active_simulator_lane_count"):
        _build(module, values, tmp_path / "terminal-one-lane-rejected.json")


def test_rejects_legacy_completed_cleanup_field_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    terminal = json.loads(
        values["receipts"]["terminal_win"].read_text(encoding="utf-8")
    )
    assert terminal[module.R246_CLEANUP_COMPLETE_FIELD] is True
    terminal[module.R246_LEGACY_CLEANUP_COMPLETED_FIELD] = True
    _write_json(values["receipts"]["terminal_win"], terminal)

    output = tmp_path / "terminal-legacy-cleanup-alias-rejected.json"
    with pytest.raises(
        module.R235R236BindingError,
        match=module.R246_LEGACY_CLEANUP_COMPLETED_FIELD,
    ):
        _build(module, values, output)
    assert not output.exists()


def test_rejects_bad_continuation_and_noncanonical_contract_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    continuation = json.loads(values["receipts"]["continuation"].read_text(encoding="utf-8"))
    continuation["valid_match_started_new_search"] = True
    _write_json(values["receipts"]["continuation"], continuation)
    with pytest.raises(module.R235R236BindingError, match="valid_match_started_new_search"):
        _build(module, values, tmp_path / "continuation-rejected.json")

    module, values = _make_inputs(tmp_path / "path", monkeypatch)
    alternate = tmp_path / "path" / "alternate-r225.json"
    alternate.write_bytes(values["r225"].read_bytes())
    with pytest.raises(module.R235R236BindingError, match="canonical typed source"):
        _build(module, values, tmp_path / "path-rejected.json", r225_contract=alternate)


def test_output_is_write_once_and_builder_has_no_network_or_execution_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    output = tmp_path / "binding.json"
    _build(module, values, output)
    prior = output.read_bytes()
    with pytest.raises(module.R235R236BindingError, match="already exists"):
        _build(module, values, output)
    assert output.read_bytes() == prior

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported & {"requests", "socket", "subprocess", "urllib", "torch", "kaggle"}


def test_refuses_to_bind_before_the_r246_contract_digest_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, values = _make_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "CANONICAL_R225_R240_TYPED_CONTRACT_SHA256", None)
    output = tmp_path / "unfrozen-r246.json"
    with pytest.raises(module.R235R236BindingError, match="final r246 canonical r225 contract digest"):
        _build(module, values, output)
    assert not output.exists()
