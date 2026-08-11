"""CPU-only contract checks for the canonical revision-236 libcg set.

These tests never download a wheel or load a native library.  An optional
local wheel audit is enabled only by setting
``POKEBOT_R236_OFFICIAL_WHEEL_PATH`` to an already-downloaded official wheel.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "state/canonical-libcg-r236.json"
R225_PATH = ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
R229_PATH = ROOT / "state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json"
OFFICIAL_WHEEL_ENV = "POKEBOT_R236_OFFICIAL_WHEEL_PATH"
R225_R246_SHA256 = "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
R229_R254_SHA256 = "sha256:7ba7860906c5a4322d78c565bc9c31de4c3d803995a28a2f6ed78f9add2cf4f1"

OFFICIAL_WHEEL = {
    "package_version": "1.32.6",
    "filename": "kaggle_environments-1.32.6-py3-none-any.whl",
    "sha256": "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab",
    "size_bytes": 60_677_343,
    "native_library_update_commit": "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303",
}

CANONICAL_MEMBERS = {
    "linux_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
    },
    "linux_aarch64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
    },
    "macos_arm64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "package_relative_path": "cg/libcg.dylib",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
    },
    "windows_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/cg.dll",
        "package_relative_path": "cg/cg.dll",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
    },
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical() -> dict[str, Any]:
    return _json(CANONICAL_PATH)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_exact_canonical_members(canonical: dict[str, Any]) -> None:
    libraries = canonical["canonical_native_libraries"]
    assert set(libraries) == set(CANONICAL_MEMBERS)
    for platform_name, expected in CANONICAL_MEMBERS.items():
        observed = libraries[platform_name]
        for key, value in expected.items():
            assert observed[key] == value


def test_r236_canonical_official_wheel_and_complete_native_set() -> None:
    canonical = _canonical()
    upstream = canonical["upstream_provenance"]
    scope = canonical["scope"]
    preflight = canonical["required_preflight"]

    assert canonical["schema"] == "poke_bot.canonical_libcg_r236/v1"
    assert canonical["owner_decision_revision"] == 236
    assert canonical["status"] == "canonical_for_all_new_simulator_packages_and_preflights"
    assert upstream["package_version"] == OFFICIAL_WHEEL["package_version"]
    assert upstream["wheel_filename"] == OFFICIAL_WHEEL["filename"]
    assert upstream["wheel_sha256"] == OFFICIAL_WHEEL["sha256"]
    assert upstream["wheel_size_bytes"] == OFFICIAL_WHEEL["size_bytes"]
    assert (
        upstream["native_library_update_commit"]
        == OFFICIAL_WHEEL["native_library_update_commit"]
    )
    _assert_exact_canonical_members(canonical)

    assert scope["r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary"]
    assert scope["r229_cross_host_package_must_overlay_and_bind_the_complete_per_platform_set"]
    assert scope["r229_both_arms_must_use_the_same_canonical_library_identity"]
    assert (
        scope[
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid"
        ]
    )
    assert scope["mixed_old_and_new_library_sets_allowed"] is False
    assert canonical["historical_identity_preservation"]["old_linux_x86_64_sha256"] != (
        CANONICAL_MEMBERS["linux_x86_64"]["sha256"]
    )
    assert preflight["wheel_sha256_and_all_four_member_sha256_and_size_checks"]
    assert preflight["same_library_set_manifest_on_every_r229_host"]
    assert preflight["no_prior_old_binary_receipt_may_satisfy_a_new_binary_gate"]


def test_r225_linux_projection_agrees_and_retains_the_wrapper_overlay_rule() -> None:
    canonical = _canonical()
    r225 = _json(R225_PATH)
    r225_libcg = r225["canonical_libcg_revision"]
    exact_base = r225["exact_frozen_base"]
    relationship = r225["relationship_to_existing_work"]
    linux = CANONICAL_MEMBERS["linux_x86_64"]

    assert r225["owner_canonical_libcg_revision"] == canonical["owner_decision_revision"]
    assert r225_libcg["typed_source"] == "state/canonical-libcg-r236.json"
    assert r225_libcg["package_version"] == (
        f"kaggle-environments=={OFFICIAL_WHEEL['package_version']}"
    )
    assert r225_libcg["official_wheel_sha256"] == OFFICIAL_WHEEL["sha256"]
    assert r225_libcg["native_library_update_commit"] == OFFICIAL_WHEEL[
        "native_library_update_commit"
    ]
    assert r225_libcg["linux_x86_64_sha256"] == linux["sha256"]
    assert r225_libcg["linux_x86_64_size_bytes"] == linux["size_bytes"]
    assert r225_libcg["new_r235_package_must_overlay_the_exact_official_linux_binary"]
    assert r225_libcg["new_package_may_retain_the_old_frozen_r195_libcg_member"] is False

    assert exact_base["stock_libcg_relative_path"] == linux["package_relative_path"]
    assert exact_base["stock_libcg_sha256"] == linux["sha256"]
    assert exact_base["stock_libcg_size_bytes"] == linux["size_bytes"]
    assert exact_base["stock_libcg_official_wheel_sha256"] == OFFICIAL_WHEEL["sha256"]
    assert exact_base["stock_libcg_update_commit"] == OFFICIAL_WHEEL[
        "native_library_update_commit"
    ]
    assert exact_base[
        "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid"
    ]
    assert "SearchRelease" in exact_base["required_stock_libcg_exports"]

    assert relationship["r229_adopts_only_the_r236_library_identity_and_retains_its_r233_outer_watchdog"]
    assert relationship["r234_kaggle_broker_or_queue_lifecycle_may_be_imported_into_r229"] is False
    assert relationship["r229_bo1000_outer_watchdog_and_all_bo_files_remain_unchanged"]


def test_r229_projects_all_four_members_and_preserves_its_lifecycle_boundary() -> None:
    canonical = _canonical()
    r229 = _json(R229_PATH)
    r229_libcg = r229["canonical_libcg_r236"]
    scope = canonical["scope"]

    assert r229["owner_canonical_libcg_revision"] == canonical["owner_decision_revision"]
    assert r229["outer_game_watchdog_revision"] == 233
    assert r229["frozen_identity"]["canonical_libcg_contract_path"] == (
        "state/canonical-libcg-r236.json"
    )
    assert r229["frozen_identity"]["canonical_libcg_official_wheel_sha256"] == (
        OFFICIAL_WHEEL["sha256"]
    )
    assert r229_libcg["package_version"] == OFFICIAL_WHEEL["package_version"]
    assert r229_libcg["native_library_update_commit"] == OFFICIAL_WHEEL[
        "native_library_update_commit"
    ]

    for platform_name, expected in CANONICAL_MEMBERS.items():
        assert r229_libcg[platform_name] == {
            "path": expected["package_relative_path"],
            "sha256": expected["sha256"],
            "size_bytes": expected["size_bytes"],
        }

    assert r229_libcg["complete_per_platform_set_must_be_resealed_before_any_new_game"]
    assert r229_libcg["same_exact_platform_appropriate_identity_required_across_both_arms"]
    assert r229_libcg["old_and_new_member_mixing_allowed"] is False
    assert r229_libcg["previous_r229_sealed_package_is_historical_and_not_execution_eligible"]
    assert r229_libcg["revision_233_outer_game_watchdog_remains_authoritative"]
    assert r229_libcg["r234_kaggle_parent_broker_direct_fallback_in_r229_allowed"] is False
    assert r229_libcg["r234_kaggle_queue_cleanup_lifecycle_in_r229_allowed"] is False
    assert scope["r229_cross_host_package_must_overlay_and_bind_the_complete_per_platform_set"]
    assert r229["safety"]["canonical_libcg_r236_platform_identity_and_no_mixed_member_preflight_required"]
    assert r229["safety"]["kaggle_specific_r228_lifecycle_may_be_changed_by_bo1000"] is False


def test_r238_r235_replacement_is_exactly_two_lanes_for_the_observed_phase1_envelope() -> None:
    """The r238 replacement scope must not retain the historic eight-lane target."""

    r225 = _json(R225_PATH)
    environment = r225["phase1_submission_environment"]
    local = r225["local_preflight"]
    shared_tree = local["full_gameplay_shared_tree_contract"]

    assert r225["owner_phase1_submission_resources_and_two_lane_revision"] == 238
    assert environment["owner_decision_revision"] == 238
    assert environment["scope"] == "new_r235_replacement_package_and_its_fresh_local_preflight_only"
    assert {
        "hdd_space_gib": environment["hdd_space_gib"],
        "ram_gib": environment["ram_gib"],
        "vcpus": environment["vcpus"],
        "submission_archive_limit_mib": environment["submission_archive_limit_mib"],
    } == {
        "hdd_space_gib": 11.8,
        "ram_gib": 12.2,
        "vcpus": 2,
        "submission_archive_limit_mib": 197.7,
    }
    assert environment["resource_probe_and_archive_size_receipt_required"]
    assert environment[
        "old_p5_h100_256gib_16vcpu_viability_expectation_is_historical_and_ineligible_for_the_r235_replacement"
    ]
    assert environment["resource_mismatch_or_archive_over_limit_behavior"] == (
        "hard_fail_closed_and_do_not_upload"
    )

    assert local["required_simulator_search_lane_count"] == 2
    assert local["required_internal_agent_start_simulator_search_arena_count_per_child"] == 2
    assert local["required_search_begin_call_count_per_ambiguous_mcts_decision"] == 2
    assert shared_tree["maximum_returned_frontier_states_per_frozen_gpu_batch"] == 2
    assert shared_tree["exactly_two_frontier_states_per_full_lane_round_required"]


def test_r244_search_id_rule_is_retained_by_r246_and_r252() -> None:
    """The current Kaggle and BO projections retain r244 handle identity."""

    r225 = _json(R225_PATH)
    r229 = _json(R229_PATH)
    local = r225["local_preflight"]
    shared_tree = local["full_gameplay_shared_tree_contract"]
    kaggle_search = r225["replacement_kaggle_diagnostic"]["phase1_simulator_search"]
    relationship = r225["relationship_to_existing_work"]
    bo_search = r229["r244_official_libcg_handle_scoped_search_id_correction"]

    assert _sha256(R225_PATH) == R225_R246_SHA256
    assert r225["owner_decision_revision"] == 246
    assert r225["owner_handle_scoped_search_id_revision"] == 244
    assert r225["owner_proven_deterministic_terminal_win_this_turn_revision"] == 246
    assert r229["owner_decision_revision"] == 252
    assert r229["owner_handle_scoped_search_id_revision"] == 244
    assert r229["owner_process_lane_recovery_revision"] == 249
    assert r229["owner_serial_mcts_revision"] == 250
    assert relationship[
        "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229"
    ]
    assert relationship[
        "r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries"
    ]

    assert local["required_simulator_search_lane_count"] == 2
    assert local["required_internal_agent_start_simulator_search_arena_count_per_child"] == 2
    assert local["required_search_begin_call_count_per_ambiguous_mcts_decision"] == 2
    assert local["required_distinct_internal_agent_start_handle_identity_count"] == 2
    assert local["search_id_numeric_namespace_is_per_distinct_agent_start_handle"]
    assert local["globally_distinct_raw_search_id_integers_required"] is False
    assert local["first_raw_search_id_may_be_zero_on_each_distinct_handle"]
    assert local["required_distinct_handle_identity_first_search_id_composite_state_count"] == 2
    assert local["per_lane_handle_scoped_search_id_chains_required"]
    assert shared_tree["retain_exact_lane_handle_identity_search_id_tuple_across_repeated_depth_waves"]
    assert shared_tree["per_lane_handle_scoped_search_id_chains_required"]
    assert shared_tree["two_distinct_handle_identity_first_search_id_composite_states_required"]
    assert shared_tree["global_raw_search_id_integer_distinctness_required"] is False

    assert kaggle_search["required_simulator_search_lane_count"] == 2
    assert kaggle_search["required_active_simulator_search_lane_count"] == 2
    assert kaggle_search["required_internal_agent_start_simulator_search_arena_count"] == 2
    assert kaggle_search["required_unique_raw_handle_count"] == 2
    assert kaggle_search["required_search_begin_call_count"] == 2
    assert kaggle_search["required_distinct_per_lane_handle_identity_count"] == 2
    assert kaggle_search["per_lane_handle_scoped_search_id_chains_required"]
    assert kaggle_search["search_id_numeric_namespace_is_per_distinct_agent_start_handle"]
    assert kaggle_search["globally_distinct_raw_search_id_integers_required"] is False
    assert kaggle_search["first_raw_search_id_may_be_zero_on_each_distinct_handle"]
    assert kaggle_search["required_distinct_handle_identity_first_search_id_composite_state_count"] == 2
    assert kaggle_search["first_search_id_identity_composite"] == "(handle_identity, first_search_id)"
    assert kaggle_search["handle_scoped_first_search_id_composite_state_public_shape"] == {
        "array_field": "handle_scoped_first_search_id_composite_states",
        "entry_exact_keys_in_order": ["lane_id", "handle_identity", "first_search_id"],
        "lane_id_values": [0, 1],
        "handle_identity": "opaque AgentStart handle identity; exactly two distinct values",
        "first_search_id": (
            "nonnegative native SearchId scoped to the entry handle; raw values may "
            "repeat across distinct handles"
        ),
        "state_identity_composite": "(handle_identity, first_search_id)",
    }

    assert bo_search["scope"] == "r229_next_two_lane_package_and_evaluation_only"
    assert bo_search["supersedes_only"] == "global_raw_search_id_integer_distinctness"
    assert bo_search["two_arenas_and_two_search_begin_calls_required"]
    assert bo_search["two_distinct_per_lane_handle_identities_required"]
    assert bo_search["both_per_lane_handle_scoped_search_id_chains_required"]
    assert bo_search["two_distinct_handle_identity_first_search_id_composite_states_required"]
    assert bo_search["search_id_numeric_namespace"] == "per_distinct_agent_start_handle"
    assert bo_search["first_raw_search_id_may_be_zero_on_each_distinct_handle"]
    assert bo_search["globally_distinct_raw_search_id_integers_required"] is False
    assert bo_search["public_composite_state_array_field"] == (
        "handle_scoped_first_search_id_composite_states"
    )
    assert bo_search["public_composite_state_entry_exact_keys_in_order"] == [
        "lane_id",
        "handle_identity",
        "first_search_id",
    ]
    assert bo_search["public_composite_state_entry_lane_id_values"] == [0, 1]
    assert bo_search["missing_handle_or_chain_or_duplicate_composite_state_behavior"] == (
        "fail_closed_before_game"
    )
    assert bo_search["r239_outer_watchdog_and_pre_r234_lifecycle_boundary_preserved"]
    assert r229["safety"][
        "r244_official_libcg_handle_scoped_search_id_identity_receipt_adapted_to_exactly_one_handle_required_before_fleet_execution"
    ]


def test_r254_abandons_full_mcts_and_preserves_r253_diagnostic_evidence() -> None:
    """R254 stops full MCTS and gates a conservative selective proof search."""

    r229 = _json(R229_PATH)
    recovery = r229["r249_two_process_native_lane_recovery"]
    serial = r229["r250_serial_process_native_lane"]
    boundaries = r229["r252_internal_simulated_leaf_boundaries"]
    rollouts = r229["r253_restarting_serial_root_rollouts"]
    abandonment = r229[
        "r254_full_mcts_abandonment_and_selective_tactical_proof_search"
    ]
    parallel = r229["post_serial_process_parallel_node_evaluation_investigation"]
    fleet = r229["fleet"]
    lifecycle = r229["r229_lifecycle_boundary"]
    safety = r229["safety"]

    assert _sha256(R229_PATH) == R229_R254_SHA256
    assert r229["owner_decision_revision"] == 254
    assert r229["owner_full_mcts_abandonment_revision"] == 254
    assert r229["owner_process_lane_recovery_revision"] == 249
    assert r229["owner_canonical_libcg_revision"] == 236
    assert r229["owner_handle_scoped_search_id_revision"] == 244
    assert r229["owner_serial_mcts_revision"] == 250
    assert r229["owner_internal_leaf_boundary_revision"] == 252
    assert r229["owner_restarting_serial_rollout_revision"] == 253

    assert recovery["scope"] == "r229_bo1000_only"
    assert recovery["authoritative_game_process_owns_frozen_model"]
    assert recovery["authoritative_game_process_owns_one_logical_shared_mcts_tree"]
    assert recovery["exact_persistent_native_simulator_worker_process_count"] == 2
    assert recovery["one_official_r236_agent_start_handle_per_worker_process"]
    assert recovery["native_search_calls_in_uncancellable_parent_python_threads_allowed"] is False
    assert recovery["every_native_request_and_cleanup_has_finite_parent_deadline"]
    assert recovery["fault_scope"] == "reap_only_the_exact_owned_lane_child_process"
    assert recovery["failed_partial_tree_reuse_allowed"] is False
    assert recovery["full_two_lane_retry_count_after_first_fault"] == 1
    assert recovery["retry_root_binding"] == [
        "same_exact_root_observation",
        "same_complete_ordered_legal_action_space",
        "same_frozen_model_and_policy_history",
        "fresh_two_lane_agent_start_handles",
    ]
    assert recovery["search_seconds_per_attempt"] == 8.0
    assert recovery["successful_retry_action_authority"] == (
        "ordinary_fully_backed_two_lane_shared_tree_mcts"
    )
    assert recovery["direct_fallback_allowed_only_after_retry_exhaustion"]
    assert recovery["retry_exhaustion_fallback"] == (
        "already_computed_legal_same_state_frozen_r195_direct_counterfactual"
    )
    assert recovery["retry_exhaustion_fallback_is_explicitly_degraded"]
    assert recovery["retry_exhaustion_fallback_receives_mcts_change_credit"] is False
    assert recovery["required_recovery_telemetry"] == [
        "attempt_count",
        "lane_process_identities",
        "fault_codes",
        "fault_phases",
        "bounded_reap_outcomes",
        "recovered_search",
        "exhausted_direct_fallback",
        "recovery_elapsed_seconds",
    ]
    assert recovery["clean_full_game_launch_preflight_max_exhausted_fallbacks"] == 0
    assert recovery["kaggle_broker_or_search_policy_import_allowed"] is False
    assert recovery["two_process_topology_superseded_by_owner_revision"] == 250
    assert recovery[
        "r249_two_process_packages_preflights_and_results_are_historical_and_execution_ineligible_for_r250"
    ]

    assert serial["exact_persistent_native_simulator_worker_process_count"] == 1
    assert serial["exact_official_r236_agent_start_handle_count"] == 1
    assert serial["exact_search_begin_call_count_per_searched_decision"] == 1
    assert serial["exact_handle_scoped_search_id_chain_count_per_searched_decision"] == 1
    assert serial[
        "exact_handle_identity_first_search_id_composite_state_count_per_searched_decision"
    ] == 1
    assert serial["concurrent_libcg_search_calls_allowed"] is False
    assert serial["native_search_calls_in_uncancellable_parent_python_threads_allowed"] is False
    assert serial["every_native_request_and_cleanup_has_finite_parent_deadline"]
    assert serial["failed_partial_tree_reuse_allowed"] is False
    assert serial["complete_root_retry_count_after_first_fault"] == 1
    assert serial["search_seconds_per_attempt"] == 8.0
    assert serial["successful_retry_action_authority"] == (
        "ordinary_fully_backed_serial_mcts"
    )
    assert serial["direct_fallback_allowed_only_after_retry_exhaustion"]
    assert serial[
        "retry_exhaustion_fallback_receives_mcts_change_or_meaningful_change_credit"
    ] is False
    assert serial["clean_full_game_launch_preflight_max_exhausted_fallbacks"] == 0
    assert serial["r239_and_r249_topology_artifacts_execution_eligible"] is False
    assert serial["kaggle_broker_queue_runtime_main_or_search_policy_import_allowed"] is False
    assert serial["continuous_single_trajectory_mechanics_superseded_by_owner_revision"] == 253
    assert serial["continuous_single_trajectory_packages_and_results_execution_eligible"] is False

    assert boundaries["root_complete_ordered_action_ceiling"] == 65536
    assert boundaries["root_action_enumeration_or_authority_changed"] is False
    assert boundaries["real_root_at_or_below_65536_is_completely_enumerated"] is True
    assert boundaries["bo_real_root_over_65536_behavior"] == (
        "legal_same_state_frozen_r195_factorized_direct_policy_action"
    )
    assert boundaries["bo_real_root_over_65536_telemetry"] == (
        "oversized_direct_fallback"
    )
    assert boundaries["bo_real_root_over_65536_counts_as_mcts_search_or_change"] is False
    assert boundaries[
        "bo_real_root_over_65536_is_retry_or_retry_exhaustion_fallback"
    ] is False
    assert boundaries["kaggle_real_root_over_65536_behavior"] == "hard_fail"
    assert boundaries[
        "deterministic_internal_complete_ordered_action_expansion_ceiling"
    ] == 64
    assert boundaries["every_explicit_chance_context_stops_before_random_resolution"]
    assert boundaries[
        "stock_simulator_probability_distribution_may_be_assumed_or_invented"
    ] is False
    assert boundaries["private_random_outcome_sampling_allowed"] is False
    assert boundaries[
        "boundary_representative_may_be_installed_as_tree_child_or_backed_action"
    ] is False
    assert boundaries["boundary_may_supply_root_action_or_continuation_plan_authority"] is False
    assert boundaries["r250_package_preflights_and_partial_results_execution_eligible"] is False
    assert boundaries["kaggle_r246_semantics_changed"] is False

    assert rollouts["scope"] == "r229_bo1000_only"
    assert rollouts["owned_native_child_process_count"] == 1
    assert rollouts["official_r236_agent_start_handle_count"] == 1
    assert rollouts["concurrent_native_calls_allowed"] is False
    assert rollouts["same_exact_physical_root_search_begin_required_per_rollout"]
    assert rollouts["independent_root_rollouts_required"]
    assert rollouts["one_new_leaf_or_r252_value_boundary_maximum_per_rollout"]
    assert rollouts["parent_tree_visit_accumulation_across_successful_rollouts_required"]
    assert rollouts["rollout_search_ids_carry_tree_or_action_authority_between_rollouts"] is False
    assert rollouts["bounded_release_of_every_rollout_search_id_required"]
    assert rollouts["bounded_search_end_per_rollout_required"]
    assert rollouts["decision_rollout_ceiling"] == 1000
    assert rollouts["minimum_completed_independent_rollouts_for_mcts_action_authority"] == 2
    assert rollouts["complete_fresh_root_retry_count"] == 1
    assert rollouts["stopped_r252_launch"]["completed_games"] == 13
    assert rollouts["stopped_r252_launch"][
        "bo_result_or_mcts_rate_or_search_effect_or_model_comparison_authority"
    ] is False
    assert rollouts["kaggle_runtime_package_preflight_or_submission_authority_changed"] is False

    assert fleet["execution_data_plane"] == "host_local_sealed_package"
    assert fleet["lan_control_plane"] == "claims_heartbeats_and_json_receipts_only"
    assert fleet[
        "each_host_pre_stages_and_checksum_verifies_its_platform_appropriate_package_model_and_libcg"
    ]
    assert fleet[
        "per_game_package_model_or_simulator_state_transfer_over_lan_allowed"
    ] is False
    assert fleet["host_local_result_then_small_json_receipt_return_required"]

    assert parallel["status"] == "owner_abandoned_with_full_mcts_under_revision_254"
    assert parallel["separately_versioned_design_required"]
    assert parallel["authoritative_parent_owned_logical_tree_coordination_required"]
    assert parallel["one_distinct_native_handle_per_worker_process_required"]
    assert parallel["concurrent_native_calls_within_one_worker_process_allowed"] is False
    assert parallel["current_package_preflight_game_fleet_or_action_authority"] is False
    assert parallel["may_change_kaggle_specific_r228_lifecycle"] is False

    assert abandonment["r253_managed_run_disposition"] == (
        "owner_abandoned_cleanly_stopped"
    )
    assert abandonment["completed_games"] == 45
    assert abandonment["unplayed_games"] == 955
    assert abandonment["observed_mcts_wins"] == 5
    assert abandonment["observed_direct_policy_wins"] == 40
    assert abandonment["observed_meaningful_changes"] == 388
    assert abandonment["observed_searched_decisions"] == 2014
    assert abandonment["incomplete_prefix_final_1000_game_efficacy_authority"] is False
    assert abandonment["full_mcts_resume_refill_or_replacement_allowed"] is False
    assert abandonment["process_parallel_full_mcts_follow_on_allowed"] is False
    assert abandonment["frozen_r195_direct_policy_is_default_action"]
    assert abandonment["learned_leaf_value_override_authority"] is False
    assert abandonment["visit_or_prior_or_partial_tree_override_authority"] is False
    assert abandonment["simulated_action_tokens_in_temporal_previous_action_history_required"]
    assert abandonment["r253_missing_simulated_action_token_behavior_may_be_copied"] is False
    assert abandonment[
        "bo1000_kaggle_training_serving_selector_promotion_checkpoint_or_model_update_authority"
    ] is False

    assert lifecycle[
        "pre_r234_search_policy_semantics_must_be_preserved_except_for_r252_internal_value_boundaries_and_content_verified_at_package_staging"
    ]
    assert lifecycle["revision_233_outer_one_game_process_watchdog_remains_authoritative"]
    assert lifecycle["r234_kaggle_parent_broker_direct_fallback_may_be_imported"] is False
    assert lifecycle["r234_kaggle_queue_cleanup_lifecycle_may_be_imported"] is False
    assert lifecycle["r249_bo_only_bounded_process_ownership_reap_and_retry_boundary_required"]
    assert lifecycle["r250_exactly_one_serial_process_lane_required"]
    assert lifecycle["r252_internal_leaf_boundary_contract_required"]
    assert lifecycle["r253_independent_exact_root_restart_per_rollout_required"]
    assert lifecycle["r250_continuous_single_trajectory_action_authority_allowed"] is False
    assert lifecycle["concurrent_libcg_calls_allowed"] is False
    assert lifecycle["failed_partial_tree_may_not_supply_action_authority"]
    assert lifecycle["r250_bounded_retry_exhaustion_may_use_only_same_state_r195_direct"]
    assert safety["canonical_libcg_r236_platform_identity_and_no_mixed_member_preflight_required"]
    assert safety["r250_serial_process_lane_fault_injection_receipt_required_before_fleet_execution"]
    assert safety["r250_clean_full_game_preflight_requires_zero_exhausted_recovery_fallbacks"]
    assert safety["r252_internal_leaf_boundary_receipt_required_before_fleet_execution"]
    assert safety[
        "r253_multi_root_edge_and_changed_selection_regressions_required_before_fleet_execution"
    ]
    assert safety[
        "stopped_r252_thirteen_game_launch_may_supply_bo_result_or_mcts_effect_authority"
    ] is False
    assert safety["kaggle_specific_r228_lifecycle_may_be_changed_by_bo1000"] is False


def test_optional_local_official_wheel_matches_every_bound_member() -> None:
    """Audit a supplied wheel locally; this test never fetches one."""

    configured_path = os.environ.get(OFFICIAL_WHEEL_ENV)
    if not configured_path:
        pytest.skip(f"set {OFFICIAL_WHEEL_ENV} to audit an already-downloaded wheel")

    wheel_path = Path(configured_path)
    assert wheel_path.is_file(), f"official r236 wheel is not a file: {wheel_path}"
    assert wheel_path.name == OFFICIAL_WHEEL["filename"]
    assert wheel_path.stat().st_size == OFFICIAL_WHEEL["size_bytes"]
    assert "sha256:" + hashlib.sha256(wheel_path.read_bytes()).hexdigest() == (
        OFFICIAL_WHEEL["sha256"]
    )

    with ZipFile(wheel_path) as wheel:
        for expected in CANONICAL_MEMBERS.values():
            info = wheel.getinfo(expected["wheel_member"])
            contents = wheel.read(info)
            assert info.file_size == expected["size_bytes"]
            assert len(contents) == expected["size_bytes"]
            assert "sha256:" + hashlib.sha256(contents).hexdigest() == expected["sha256"]
