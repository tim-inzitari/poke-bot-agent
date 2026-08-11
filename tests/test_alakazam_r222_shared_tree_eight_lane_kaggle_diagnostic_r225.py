"""Focused historical-r225 identity and frozen-r246 replacement guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
)
CONTRACT_SHA256 = "3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
R222_PATH = ROOT / "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json"
R222_SHA256 = "8b5a19e8746b8e5f667683ad6437a2f3506aa0fbdcca8495ec2b8bbd1eebeb7e"
R224_PATH = ROOT / "state/alakazam-phase1-ladder-eight-lane-search-r224.json"
R224_SHA256 = "700b96912b8bb6d871883790dbf300781d9941b65a962416dd288d95619ebb84"
CONTRACT_KEY = "alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225"
COMPOSITE_KEYS = ["lane_id", "handle_identity", "first_search_id"]


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _assert_handle_scoped_identity(identity: dict) -> None:
    """Check the r244 handle-scoped identity rule, not obsolete raw-ID uniqueness."""

    assert identity.get(
        "numeric_namespace", identity.get("search_id_numeric_namespace")
    ) == "per_distinct_agent_start_handle"
    assert identity["globally_distinct_raw_search_id_integers_required"] is False
    assert identity["first_raw_search_id_may_be_zero_on_each_distinct_handle"]
    assert (
        identity.get(
            "required_distinct_handle_identity_first_search_id_composite_state_count",
            identity.get(
                "distinct_handle_identity_first_search_id_composite_state_count"
            ),
        )
        == 2
    )
    assert identity["first_search_id_identity_composite"] == (
        "(handle_identity, first_search_id)"
    )
    assert identity["public_composite_state_array_field"] == (
        "handle_scoped_first_search_id_composite_states"
    )
    assert identity["public_composite_state_entry_exact_keys_in_order"] == COMPOSITE_KEYS
    assert identity["public_composite_state_entry_lane_id_values"] == [0, 1]


def _assert_two_lane_local_preflight(local: dict, *, projection: bool) -> None:
    if projection:
        assert local["one_kaggle_submission_agent_process_per_child_required"]
        assert local["one_loaded_stock_libcg_dso_per_child_required"]
    else:
        assert local["top_level_submission_parent_required"]
        assert local["one_stock_libcg_dso_mapping_per_runtime_process_required"]
    assert local["required_simulator_search_lane_count"] == 2
    assert (
        local["required_internal_agent_start_simulator_search_arena_count_per_child"]
        == 2
    )
    assert local["one_internal_agent_start_arena_per_persistent_cpu_worker_lane_required"]
    assert (
        local["one_agent_start_handle_concurrently_shared_across_cpu_worker_lanes_allowed"]
        is False
    )
    assert local["required_search_begin_call_count_per_ambiguous_mcts_decision"] == 2
    assert local["required_distinct_internal_agent_start_handle_identity_count"] == 2
    assert local["search_id_numeric_namespace_is_per_distinct_agent_start_handle"]
    assert local["globally_distinct_raw_search_id_integers_required"] is False
    assert local["first_raw_search_id_may_be_zero_on_each_distinct_handle"]
    assert (
        local[
            "required_distinct_handle_identity_first_search_id_composite_state_count"
        ]
        == 2
    )
    assert local["per_lane_handle_scoped_search_id_chains_required"]
    assert local["exactly_two_simultaneous_in_decision_simulator_search_lanes_required"]
    assert local["one_shared_logical_mcts_tree_required"]
    assert local["per_lane_or_one_lane_frozen_gpu_batch_allowed"] is False
    assert (
        local.get(
            "partial_lane_or_serial_lane_mcts_authority_allowed",
            local.get("partial_lane_mcts_action_authority_allowed"),
        )
        is False
    )
    assert local["zero_outstanding_reservations_required_at_action_return"]

    if projection:
        _assert_handle_scoped_identity(
            local["official_libcg_handle_scoped_search_id_contract"]
        )
        return

    receipt = local["r238_two_lane_receipt_contract"]
    assert receipt["normal_mcts_decision_exact_counts"] == {
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "distinct_handle_identity_count": 2,
        "distinct_handle_scoped_first_search_id_composite_state_count": 2,
        "search_begin_calls": 2,
        "search_end_calls": 2,
    }
    assert receipt["normal_mcts_decision_lane_ids"] == [0, 1]
    assert receipt["normal_mcts_decision_microbatch_size_range"] == [1, 2]
    assert receipt["normal_mcts_decision_max_simulator_calls_in_flight_range"] == [1, 2]
    _assert_handle_scoped_identity(receipt["search_id_identity_contract"])


def _assert_replacement_diagnostic(replacement: dict) -> None:
    assert replacement["revision"] == 235
    assert replacement["canonical_libcg_revision"] == 236
    assert replacement["phase1_two_lane_resource_revision"] == 238
    assert replacement["hybrid_confidence_bounded_mcts_revision"] == 240
    assert replacement["handle_scoped_search_id_correction_revision"] == 244
    assert replacement["competition"] == "pokemon-tcg-ai-battle"
    assert replacement["exactly_one_new_direct_replacement_submission_conditionally_authorized"]
    assert replacement["replacement_submission_count_limit"] == 1
    assert replacement["replacement_submission_consumed"] is False
    assert replacement["submission_message_required_literal"] == (
        "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"
    )
    assert replacement["submission_message_must_be_unique_and_exact"]
    assert replacement["automatic_retry_allowed"] is False
    assert replacement["automatic_copy_or_resubmission_allowed"] is False
    assert replacement["queue_or_batch_submission_allowed"] is False
    assert replacement["second_upload_allowed"] is False
    assert replacement["kaggle_api_call_permitted_now_before_gates"] is False
    assert replacement["kaggle_upload_permitted_now_before_gates"] is False
    assert replacement["complete_ordered_legal_action_ceiling"] == 65536

    environment = replacement["phase1_submission_environment"]
    assert environment == {
        "hdd_space_gib": 11.8,
        "ram_gib": 12.2,
        "vcpus": 2,
        "submission_archive_limit_mib": 197.7,
        "resource_probe_and_archive_size_receipt_required": True,
        "unverified_or_mismatched_resource_or_archive_result_is_not_a_valid_r235_replacement_pass": True,
    }

    search = replacement["phase1_simulator_search"]
    assert search["required_simulator_search_lane_count"] == 2
    assert search["required_active_simulator_search_lane_count"] == 2
    assert search["required_internal_agent_start_simulator_search_arena_count"] == 2
    assert search["required_unique_raw_handle_count"] == 2
    assert search["required_search_begin_call_count"] == 2
    assert search["required_distinct_per_lane_handle_identity_count"] == 2
    assert search["per_lane_handle_scoped_search_id_chains_required"]
    assert search["search_id_numeric_namespace_is_per_distinct_agent_start_handle"]
    assert search["globally_distinct_raw_search_id_integers_required"] is False
    assert search["first_raw_search_id_may_be_zero_on_each_distinct_handle"]
    assert (
        search[
            "required_distinct_handle_identity_first_search_id_composite_state_count"
        ]
        == 2
    )
    assert search["first_search_id_identity_composite"] == (
        "(handle_identity, first_search_id)"
    )
    assert search["maximum_frontier_leaves_per_frozen_evaluator_batch"] == 2
    assert search["one_shared_logical_mcts_tree_required"]
    assert (
        search[
            "one_lane_serial_fallback_eight_lane_topology_or_partial_lane_mcts_authority_allowed"
        ]
        is False
    )
    public_shape = search["handle_scoped_first_search_id_composite_state_public_shape"]
    assert public_shape["array_field"] == "handle_scoped_first_search_id_composite_states"
    assert public_shape["entry_exact_keys_in_order"] == COMPOSITE_KEYS
    assert public_shape["lane_id_values"] == [0, 1]
    assert public_shape["state_identity_composite"] == "(handle_identity, first_search_id)"

    scheduler = replacement["r240_hybrid_scheduler"]
    assert scheduler["selected_factorized_stage_probability_threshold"] == 0.8
    assert scheduler["high_confidence_frozen_direct_threshold_owner_revision"] == 242
    assert scheduler["historical_r240_0_90_threshold_draft_and_preflight_are_ineligible"]
    assert scheduler["child_search_hard_seconds"] == 2.0
    assert scheduler["parent_action_hard_seconds"] == 4.0
    assert scheduler["adaptive_early_stop_min_completed_backups"] == 8
    assert scheduler["adaptive_early_stop_stable_deterministic_root_leader_observations"] == 3
    assert scheduler["adaptive_early_stop_both_lanes_progressed_required"]
    assert scheduler["hard_completed_backup_stop"] == 32
    assert scheduler["mcts_opponent_action_selection_or_planning_allowed"] is False
    assert (
        scheduler[
            "mcts_simulated_rollout_expansion_stops_at_terminal_chance_boundary_or_actor_change_away_from_root_seat"
        ]
    )

    continuation = replacement["deterministic_continuation"]
    assert continuation["maximum_depth"] == 8
    assert continuation["current_canonical_observation_fingerprint_must_exactly_match"]
    assert continuation["current_actor_must_remain_our_seat"]
    assert continuation["next_planned_action_must_be_in_current_complete_ordered_legal_actions"]
    assert continuation[
        "plan_extraction_requires_both_lanes_same_fingerprint_and_backed_leader_agreement"
    ]
    assert continuation["chance_boundary_or_opponent_transition_since_plan_extraction_allowed"] is False
    assert continuation["valid_plan_starts_or_calls_new_mcts_search"] is False
    assert continuation["valid_plan_mcts_child_started_for_this_decision"] is False
    assert continuation["valid_plan_mcts_select_call_count"] == 0


def test_historical_r225_schema_is_bound_to_the_frozen_r246_replacement() -> None:
    contract = _contract()

    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(R222_PATH.read_bytes()).hexdigest() == R222_SHA256
    assert hashlib.sha256(R224_PATH.read_bytes()).hexdigest() == R224_SHA256
    # The historical filename/schema and original decision stay stable while its
    # active replacement semantics are owned by revision 246.
    assert CONTRACT_PATH.name == "alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
    assert contract["schema"] == (
        "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
    )
    assert contract["originating_owner_decision_revision"] == 225
    assert contract["owner_decision_revision"] == 246
    assert contract["owner_topology_correction_revision"] == 227
    assert contract["owner_gameplay_viability_correction_revision"] == 228
    assert contract["owner_kaggle_native_containment_revision"] == 234
    assert contract["owner_kaggle_replacement_diagnostic_revision"] == 235
    assert contract["owner_canonical_libcg_revision"] == 236
    assert contract["owner_phase1_submission_resources_and_two_lane_revision"] == 238
    assert contract["owner_hybrid_confidence_bounded_mcts_revision"] == 240
    assert contract["owner_high_confidence_frozen_direct_threshold_revision"] == 242
    assert contract["owner_handle_scoped_search_id_revision"] == 244
    assert contract["owner_proven_deterministic_terminal_win_this_turn_revision"] == 246
    assert "r246_proven_deterministic_terminal_win" in contract["status"]

    relationship = contract["relationship_to_existing_work"]
    assert relationship["r222_contract_must_be_preserved_byte_for_byte"]
    assert relationship["r224_contract_must_be_preserved_byte_for_byte"]
    assert relationship["r228_rough_one_shot_full_gameplay_viability_test_is_consumed_and_failed"]
    assert relationship[
        "r238_supersedes_only_the_new_r235_replacement_package_lane_count_and_phase1_submission_resource_envelope"
    ]
    assert relationship[
        "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible"
    ]
    assert relationship[
        "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229"
    ]

    action_space = contract["complete_ordered_action_space_contract"]
    assert action_space["complete_ordered_legal_action_ceiling"] == 65536
    assert action_space["ceiling_applies_at_root_and_every_private_leaf"]
    assert action_space["observed_6720_action_private_leaf_must_be_fully_enumerated"]
    assert action_space["sampling_pruning_or_reinterpretation_of_legal_choices_allowed"] is False

    _assert_two_lane_local_preflight(contract["local_preflight"], projection=False)
    _assert_replacement_diagnostic(contract["replacement_kaggle_diagnostic"])

    historical = contract["one_shot_kaggle_diagnostic"]
    assert historical["consumed_submission_id"] == 55416396
    assert historical[
        "historical_eight_lane_requirements_apply_only_to_the_consumed_one_shot_and_not_to_the_r235_r238_replacement"
    ]
    assert historical[
        "exactly_one_direct_submission_conditionally_authorized_after_all_local_preflight_receipts"
    ] is False
    assert historical["replacement_submission_authorized"] is False

    authority = contract["authority"]
    assert authority["offline_package_build_authorized"]
    assert authority["local_child_process_preflight_authorized"]
    assert authority["local_fault_injection_preflight_authorized"]
    assert authority[
        "one_direct_r235_replacement_kaggle_submission_authorized_only_after_all_local_gates_and_immutable_binding"
    ]
    assert authority["kaggle_api_call_permitted_now"] is False
    assert authority["kaggle_upload_permitted_now"] is False
    assert authority["kaggle_queue_submission_permitted"] is False
    assert authority["kaggle_retry_or_copy_permitted"] is False
    assert authority["second_kaggle_upload_permitted"] is False
    assert authority["production_action_authority_enabled"] is False


def test_r246_projections_keep_the_historical_key_and_current_replacement() -> None:
    protocol = yaml.safe_load((ROOT / "config/rl_protocol.yaml").read_text())[CONTRACT_KEY]
    specialists = yaml.safe_load((ROOT / "state/specialists.yaml").read_text())[CONTRACT_KEY]
    compatibility = json.loads(
        (ROOT / "ops/current_goal_requirements.json").read_text()
    )["current_owner_overrides"][CONTRACT_KEY]
    goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")

    for projection in (protocol, specialists, compatibility):
        assert projection.get("owner_decision_revision", projection.get("goal_revision")) == 246
        assert projection["owner_handle_scoped_search_id_revision"] == 244
        assert projection["owner_proven_deterministic_terminal_win_this_turn_revision"] == 246
        assert projection["owner_high_confidence_frozen_direct_threshold_revision"] == 242
        assert projection["owner_phase1_submission_resources_and_two_lane_revision"] == 238
        assert projection["r222_contract_sha256"] == "sha256:" + R222_SHA256
        assert projection["r224_contract_sha256"] == "sha256:" + R224_SHA256
        _assert_two_lane_local_preflight(projection["local_preflight"], projection=True)

        replacement = projection["replacement_kaggle_diagnostic"]
        assert replacement["handle_scoped_search_id_correction_revision"] == 244
        assert replacement["submission_message_required_literal"] == (
            "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"
        )
        assert replacement["replacement_submission_count_limit"] == 1
        assert replacement["replacement_submission_consumed"] is False
        assert replacement["complete_ordered_legal_action_ceiling"] == 65536
        assert replacement["phase1_simulator_search"][
            "globally_distinct_raw_search_id_integers_required"
        ] is False
        assert replacement["phase1_simulator_search"][
            "first_raw_search_id_may_be_zero_on_each_distinct_handle"
        ]
        assert replacement["phase1_simulator_search"][
            "handle_scoped_first_search_id_composite_state_public_shape"
        ]["entry_exact_keys_in_order"] == COMPOSITE_KEYS
        assert replacement["r240_hybrid_scheduler"][
            "selected_factorized_stage_probability_threshold"
        ] == 0.8
        assert replacement["r240_hybrid_scheduler"]["hard_completed_backup_stop"] == 32

        historical = projection["one_shot_kaggle_diagnostic"]
        assert historical[
            "historical_eight_lane_requirements_apply_only_to_consumed_submission_55416396"
        ]
        assert historical[
            "exactly_one_direct_submission_conditionally_authorized_after_all_local_preflight_receipts"
        ] is False

        authority = projection["authority"]
        assert authority[
            "one_direct_r235_replacement_kaggle_submission_authorized_only_after_all_typed_preconditions"
        ]
        assert authority["kaggle_queue_submission_permitted"] is False
        assert authority["automatic_kaggle_submission_allowed"] is False
        assert authority["production_action_authority_enabled"] is False

    assert protocol["values_owned_by_sha256"] == "sha256:" + CONTRACT_SHA256
    assert specialists["owner_contract_sha256"] == "sha256:" + CONTRACT_SHA256
    assert compatibility["typed_source_sha256"] == "sha256:" + CONTRACT_SHA256
    assert "Under revision 244-LIBCG" in goal
    assert "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST" in goal
    assert "Global raw-SearchId integer uniqueness is not an isolation proof" in goal
