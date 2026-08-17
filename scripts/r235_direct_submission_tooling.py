#!/usr/bin/env python3
"""Offline, fail-closed support for the sole R235 direct Kaggle upload.

This tool never imports a Kaggle client and contains no upload, queue, retry,
or copy operation.  It validates the pre-binding candidate and the later
immutable binding made by ``build_r235_r236_immutable_replacement_binding.py``
and can only create:

* a digest-bound, guard-compatible pre-binding go-first receipt;
* one immutable authorization for the exact archive; and
* a later immutable submission-ID resolution receipt from saved local output.

Every command takes the candidate archive, its external manifest, and binding
explicitly.  Nothing is inferred from an old eight-lane filename.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_R225_PATH = ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
CANONICAL_R236_PATH = ROOT / "state/canonical-libcg-r236.json"
# This is the owner-frozen r246 typed source.  Any later source edit blocks
# validation until it receives a new explicit owner digest.
CANONICAL_R225_R246_SHA256: str | None = (
    "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
)
CANONICAL_R236_SHA256 = "sha256:d75ff752808ead08f3ae20f7f2f8a034c9e6163109188a46d3b877bf1910ae2d"

R225_SCHEMA = "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
R236_SCHEMA = "poke_bot.canonical_libcg_r236/v1"
R238_MANIFEST_SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
R235_BINDING_SCHEMA = "poke_bot.r235_r236_immutable_replacement_binding/v1"
AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
GO_FIRST_SCHEMA = "poke_bot.submission_turn_order_attestation/v1"
R235_AUTHORITY_KIND = "r235_direct_single_upload"
R235_CONSUMPTION_SCHEMA = "poke_bot.r235_direct_submission_consumption/v1"
R235_ID_RECEIPT_SCHEMA = "poke_bot.r235_direct_submission_id_receipt/v1"
# R246 owns the canonical source revision.  R242 still owns the high-confidence
# scheduler projection whose historical field names retain ``r240``.
R225_OWNER_DECISION_REVISION = 246
R242_OWNER_DECISION_REVISION = 242
R244_HANDLE_SCOPED_SEARCH_ID_REVISION = 244
R246_PROVEN_TERMINAL_WIN_REVISION = 246
R238_MANIFEST_ROLE = "isolated_r238_two_lane_bounded_mcts_fallback_diagnostic"

COMPETITION = "pokemon-tcg-ai-battle"
LABEL = "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"
LANE_COUNT = 2
ACTION_CAP = 65_536
PHASE1_RESOURCES = {
    "hdd_space_gib": 11.8,
    "ram_gib": 12.2,
    "vcpus": 2,
    "submission_archive_limit_mib": 197.7,
}
PHASE1_ARCHIVE_MAX_BYTES = int(PHASE1_RESOURCES["submission_archive_limit_mib"] * 1024 * 1024)
# Phase-1 resource declarations do not establish CUDA visibility.  The staged
# package must record its own runtime CUDA observation before search; accepting
# the old ``gpu_available: false`` field would falsely turn an envelope into a
# runtime hardware claim.
PHASE1_MANIFEST_RESOURCE_BOUNDS = {
    "vcpus": 2,
    "ram_gib": 12.2,
    "hdd_gib": 11.8,
    "archive_mib": 197.7,
    "gpu_environment_inferred_from_resource_envelope": False,
    "runtime_cuda_observation_required_before_search": True,
    "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
}
R240_REQUIRED_REGRESSION_RECEIPTS = (
    "high_confidence_direct_and_adaptive_bounded_mcts_regression_receipt",
    "deterministic_continuation_regression_receipt",
)
R246_REQUIRED_REGRESSION_RECEIPTS = (
    "proven_deterministic_terminal_win_this_turn_regression_receipt",
)
R242_HIGH_CONFIDENCE_THRESHOLD = 0.80
R240_STOP_REASONS = (
    "high_confidence_frozen_direct",
    "deterministic_continuation_plan",
    "proven_deterministic_terminal_win_this_turn",
    "adaptive_early_stop",
    "hard_completed_backup_stop",
    "child_search_hard_deadline",
    "parent_action_hard_deadline",
    "zero_backup_precomputed_direct_fallback",
    "contained_child_fault",
)
R240_NORMALIZED_SCHEDULER = {
    "high_confidence_threshold": R242_HIGH_CONFIDENCE_THRESHOLD,
    "all_selected_stages_finite": True,
    "immediate_no_child": True,
    "no_mcts_select_search_model_or_simulator_calls": True,
    "history_only_existing_child_journal_count_range": [0, 1],
    "high_confidence_degraded": False,
    "child_search_seconds": 2.0,
    "parent_action_deadline_seconds": 4.0,
    "minimum_backups_before_stability": 8,
    "stable_root_leader_observations": 3,
    "maximum_backups_per_decision": 32,
    "early_stop_requires_both_lanes_progressed": True,
    "stop_reason_required": True,
}
# This is the compact, pre-binding continuation projection carried by every
# binder gate receipt.  It is deliberately distinct from the fuller package
# manifest continuation contract below.
R235_GATE_DETERMINISTIC_CONTINUATION = {
    "max_depth": 8,
    "exact_observation_fingerprint_required": True,
    "both_lanes_same_fingerprint_and_backed_action_required": True,
    "same_root_actor_required": True,
    "chance_or_boundary_forbidden": True,
    "no_new_search_on_valid_match": True,
    "mismatch_clears_entire_plan": True,
}
R242_MANIFEST_SCHEDULER = {
    "scope": "new_r235_replacement_package_only",
    "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_DECISION_REVISION,
    "selected_factorized_stage_probability_threshold": R242_HIGH_CONFIDENCE_THRESHOLD,
    "threshold_comparison": (
        "every selected factorized-stage probability is finite and "
        "greater_than_or_equal_to_0.80"
    ),
    "all_selected_factorized_stages_must_meet_threshold": True,
    "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
    "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_"
    "inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
    "high_confidence_frozen_direct_mode": "high_confidence_frozen_direct",
    "high_confidence_requires_precomputed_complete_legal_frozen_r195_direct_action": True,
    "high_confidence_requires_precomputed_direct_action_match_the_complete_ordered_root_legal_set_and_legal_fingerprint": True,
    "high_confidence_mcts_child_started_for_this_decision": False,
    "high_confidence_mcts_select_search_model_or_simulator_call_allowed": False,
    "high_confidence_existing_child_history_only_note_direct_action_ipc_allowed_and_required_when_child_exists": True,
    "high_confidence_existing_child_history_only_note_direct_action_ipc_max_count": 1,
    "high_confidence_existing_child_history_only_note_direct_action_ipc_count_range": [0, 1],
    "high_confidence_history_only_note_direct_action_ipc_must_not_invoke_mcts_select_search_model_or_simulator": True,
    "high_confidence_direct_is_a_permitted_new_mcts_search_bypass_for_a_branching_prompt": True,
    "high_confidence_journaling_required": True,
    "high_confidence_degraded": False,
    "high_confidence_receipt_required_values": {
        "selected_factorized_stage_probability_threshold": R242_HIGH_CONFIDENCE_THRESHOLD,
        "all_selected_factorized_stages_meet_threshold": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "history_only_existing_child_journal_count_range": [0, 1],
        "degraded": False,
    },
    "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
    "two_lane_mcts_topology_backup_and_stop_contract_applies_only_when_confidence_routes_to_ambiguous_mcts": True,
    "ambiguous_mcts_exact_simulator_search_lane_count": LANE_COUNT,
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
    "stop_reason_fields": list(R240_STOP_REASONS),
    "zero_completed_backups_returns_only_the_precomputed_legal_direct_action_under_existing_clean_deadline_or_containment_rules": True,
    "partial_lane_serial_or_unbounded_search_authority_allowed": False,
    "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
}
# The serialized key retains its historical r240 name.  Its current exact
# r246 projection is defined after the revision-specific terminal-win contract.
MANIFEST_DETERMINISTIC_CONTINUATION = {
    "scope": "optional_receipt_carried_plan_for_new_r235_replacement_package_only",
    "maximum_depth": 8,
    "parent_computes_exact_precomputed_direct_action_before_plan_validation": True,
    "valid_plan_has_precedence_over_normal_high_confidence_or_adaptive_mcts_only_at_the_proven_continuation_step": True,
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
    "on_any_fingerprint_actor_legality_lane_disagreement_randomness_boundary_or_opponent_transition_mismatch_clear_entire_plan": True,
    "after_clear_route_to_normal_high_confidence_direct_or_adaptive_mcts": True,
    "rewrite_history_to_actual_planned_action": True,
    "journal_exactly_once": True,
    "log_planned_action_and_precomputed_direct_action": True,
    "journal_required_fields": [
        "continuation_plan_depth_remaining",
        "continuation_observation_fingerprint",
        "continuation_actor_seat",
        "continuation_both_lanes_same_fingerprint",
        "continuation_backed_leader_agreement",
        "direct_action",
        "selected_action",
        "planned_vs_direct_action_changed",
        "mcts_child_started_for_this_decision",
        "mcts_select_call_count",
        "history_only_existing_child_journal_count",
        "degraded",
    ],
}
R242_LOCAL_SCHEDULER = {
    **R242_MANIFEST_SCHEDULER,
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
}
R242_REPLACEMENT_SCHEDULER = {
    "high_confidence_frozen_direct_mode": "high_confidence_frozen_direct",
    "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_DECISION_REVISION,
    "selected_factorized_stage_probability_threshold": R242_HIGH_CONFIDENCE_THRESHOLD,
    "all_selected_factorized_stages_must_be_finite_and_greater_than_or_equal_to_threshold": True,
    "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
    "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_"
    "inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
    "complete_legal_precomputed_direct_action_required": True,
    "high_confidence_mcts_child_started_for_this_decision": False,
    "high_confidence_mcts_select_search_model_or_simulator_call_allowed": False,
    "high_confidence_existing_child_history_only_note_direct_action_ipc_allowed_and_required_when_child_exists": True,
    "high_confidence_existing_child_history_only_note_direct_action_ipc_count_range": [0, 1],
    "high_confidence_history_only_note_direct_action_ipc_must_not_invoke_mcts_select_search_model_or_simulator": True,
    "high_confidence_direct_is_a_permitted_new_mcts_search_bypass_for_a_branching_prompt": True,
    "high_confidence_direct_journal_required_and_not_degraded": True,
    "high_confidence_receipt_required_values": {
        "selected_factorized_stage_probability_threshold": R242_HIGH_CONFIDENCE_THRESHOLD,
        "all_selected_factorized_stages_meet_threshold": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "history_only_existing_child_journal_count_range": [0, 1],
        "degraded": False,
    },
    "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
    "ambiguous_mcts_exact_lane_count": LANE_COUNT,
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
    "stop_reason_fields": list(R240_STOP_REASONS),
    "zero_backups_use_only_existing_precomputed_direct_fallback_contract": True,
    "historical_fixed_eight_second_r228_branching_window_is_not_current": True,
}
R242_REPLACEMENT_DETERMINISTIC_CONTINUATION = {
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
    "journal_required_fields": list(MANIFEST_DETERMINISTIC_CONTINUATION["journal_required_fields"]),
}
R246_TERMINAL_WIN_CONTRACT_KEY = "r246_proven_deterministic_terminal_win_this_turn"
R246_TERMINAL_WIN_STOP_REASON = "proven_deterministic_terminal_win_this_turn"
R246_TERMINAL_WIN_GATE = "proven_deterministic_terminal_win_this_turn"
R246_TERMINAL_WIN_RECEIPT_NAME = (
    "proven_deterministic_terminal_win_this_turn_regression_receipt"
)
R246_TERMINAL_WIN_PROOF_KIND = "exact_deterministic_simulator_terminal_win_this_turn"
R246_CLEANUP_COMPLETE_FIELD = (
    "all_owned_lane_resources_reservations_and_child_cleanup_complete"
)
R246_LEGACY_CLEANUP_COMPLETED_FIELD = (
    "all_owned_lane_resources_reservations_and_child_cleanup_completed"
)
R246_TERMINAL_WIN_PROOF_FIELDS = [
    "proof_kind",
    "root_observation_fingerprint",
    "root_legal_order_fingerprint",
    "root_actor_seat",
    "root_action",
    "selected_action",
    "terminal_result",
    "terminal_winner_seat",
    "terminal_leaf_reached",
    "proof_path_action_count",
    "discovering_lane_id",
    "path_actor_seats",
    "path_no_chance_boundary",
    "path_no_actor_change_boundary",
    "path_no_opponent_boundary_crossing",
    "path_no_unresolved_randomness",
    "proof_is_deterministic",
]
R246_LOCAL_TERMINAL_WIN_CONTRACT = {
    "scope": "ambiguous_two_lane_mcts_for_new_r235_replacement_package_only",
    "owner_decision_revision": R246_PROVEN_TERMINAL_WIN_REVISION,
    "r242_high_confidence_frozen_direct_before_child_is_unchanged": True,
    "requires_exact_two_lane_topology_initialized_before_override": True,
    "one_valid_terminal_win_proof_from_either_lane_is_sufficient": True,
    "two_independent_lane_proofs_required": False,
    "exhaustive_legal_action_scan_required": False,
    "terminal_leaf_must_be_returned_by_exact_stock_simulator": True,
    "terminal_leaf_must_be_backed_up_into_shared_root_tree": True,
    "minimum_completed_backups_for_valid_proof": 1,
    "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
    "root_action_has_absolute_selection_and_early_stop_authority_over_visits_priors_and_nonterminal_actions": True,
    "proof_kind_required_literal": R246_TERMINAL_WIN_PROOF_KIND,
    "terminal_result_required_literal": "win",
    "terminal_winner_seat_must_equal_root_actor_seat": True,
    "root_action_must_be_currently_legal_and_equal_selected_action": True,
    "parent_must_validate_current_root_observation_legal_fingerprint_and_actor_before_action": True,
    "proof_path_actor_seats_must_all_equal_root_actor_seat": True,
    "proof_path_actor_change_or_opponent_boundary_allowed": False,
    "proof_path_chance_or_unresolved_randomness_allowed": False,
    "model_value_policy_confidence_or_heuristic_may_substitute_for_terminal_simulator_result": False,
    "loss_draw_nonterminal_stale_or_malformed_claim_has_terminal_win_override_authority": False,
    "stale_or_malformed_claim_marked_as_terminal_win_is_contained_child_protocol_fault": True,
    "all_owned_lane_resources_reservations_and_child_cleanup_required_before_parent_return": True,
    "stop_reason": R246_TERMINAL_WIN_STOP_REASON,
    "required_queue_run_decision_inputs": [
        "root_observation_fingerprint",
        "root_legal_order_fingerprint",
        "root_actor_seat",
    ],
    "required_receipt_fields": list(R246_TERMINAL_WIN_PROOF_FIELDS),
}
R246_REPLACEMENT_TERMINAL_WIN_CONTRACT = {
    key: value
    for key, value in R246_LOCAL_TERMINAL_WIN_CONTRACT.items()
    if key != "scope"
}
R246_MANIFEST_SCHEDULER = {
    **R242_MANIFEST_SCHEDULER,
    R246_TERMINAL_WIN_CONTRACT_KEY: R246_LOCAL_TERMINAL_WIN_CONTRACT,
    "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
    "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
}
R246_LOCAL_SCHEDULER = {
    **R242_LOCAL_SCHEDULER,
    R246_TERMINAL_WIN_CONTRACT_KEY: R246_LOCAL_TERMINAL_WIN_CONTRACT,
    "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
    "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
}
R246_REPLACEMENT_SCHEDULER = {
    **R242_REPLACEMENT_SCHEDULER,
    R246_TERMINAL_WIN_CONTRACT_KEY: R246_REPLACEMENT_TERMINAL_WIN_CONTRACT,
    "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
    "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
}
# The serialized key retains its historical r240 name, while R246 is now its
# exact current projection in both the candidate manifest and typed source.
R240_MANIFEST_SCHEDULER = R246_MANIFEST_SCHEDULER
R244_SEARCH_ID_IDENTITY_CONTRACT = {
    "numeric_namespace": "per_distinct_agent_start_handle",
    "globally_distinct_raw_search_id_integers_required": False,
    "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    "first_search_id_identity_composite": "(handle_identity, first_search_id)",
    "public_composite_state_array_field": "handle_scoped_first_search_id_composite_states",
    "public_composite_state_entry_exact_keys_in_order": [
        "lane_id",
        "handle_identity",
        "first_search_id",
    ],
    "public_composite_state_entry_lane_id_values": [0, 1],
    "public_composite_state_entry_handle_identity": (
        "opaque AgentStart handle identity; exactly two distinct values"
    ),
    "public_composite_state_entry_first_search_id": (
        "nonnegative native SearchId scoped to the entry handle; raw values may repeat "
        "across distinct handles"
    ),
    "required_distinct_handle_identity_first_search_id_composite_state_count": LANE_COUNT,
}
R244_TWO_LANE_RECEIPT_REQUIRED_FIELDS = [
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
]
R235_BINDING_GATE_NAMES = frozenset(
    {
        "focused_native_child_fault_suite",
        "saved_episode_91766923_step58",
        "exact_repaired_package_full_local_game",
        "resource_memory_startup_throughput",
        "phase1_resource_and_archive",
        "two_lane_topology",
        "official_libcg_handle_scoped_search_id_identity",
        "high_confidence_and_adaptive_bounded_mcts",
        R246_TERMINAL_WIN_GATE,
        "deterministic_continuation",
        "go_first",
    }
)
R235_BINDING_GATE_RECEIPT_NAMES = {
    "focused_native_child_fault_suite": "focused_native_child_fault_suite_receipt",
    "saved_episode_91766923_step58": (
        "saved_episode_91766923_seat_0_step_58_two_choice_callback_"
        "legal_hard_deadline_regression_receipt"
    ),
    "exact_repaired_package_full_local_game": "exact_repaired_package_full_local_game_receipt",
    "resource_memory_startup_throughput": "resource_memory_startup_and_throughput_preflight_receipt",
    "phase1_resource_and_archive": "phase1_submission_resource_and_archive_limit_receipt",
    "two_lane_topology": "two_lane_shared_tree_topology_and_receipt_schema_regression_receipt",
    "official_libcg_handle_scoped_search_id_identity": (
        "official_libcg_handle_scoped_search_id_identity_regression_receipt"
    ),
    "high_confidence_and_adaptive_bounded_mcts": (
        "high_confidence_direct_and_adaptive_bounded_mcts_regression_receipt"
    ),
    R246_TERMINAL_WIN_GATE: R246_TERMINAL_WIN_RECEIPT_NAME,
    "deterministic_continuation": "deterministic_continuation_regression_receipt",
    "go_first": "go_first_receipt",
}
GO_FIRST_CASES = frozenset(
    {"integer_enum", "string_enum_reversed_options", "live_engine_prompt"}
)
GO_FIRST_PREBINDING_RECEIPT_FIELDS = frozenset(
    {
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
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NONCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,191}$")
_SUBMISSION_ID = re.compile(r"^[1-9][0-9]*$")
_OUTPUT_ID = re.compile(
    r"(?i)\bsubmission(?:[\s_-]*id)?\s*[:=#]\s*[\"']?([1-9][0-9]*)"
)


class R235SupportError(RuntimeError):
    """A required R235 package, binding, or receipt invariant failed."""


def _canonical_json(payload: object) -> bytes:
    try:
        return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise R235SupportError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R235SupportError(f"{label} must be a regular non-symlink file")
    return raw.resolve()


def _required_path(value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise R235SupportError(f"{label} is missing")
    return Path(text).expanduser()


def _read_json(path: Path | str, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235SupportError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R235SupportError(f"{label} must be a JSON object")
    return value


def _digest(value: object, *, label: str) -> str:
    result = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(result):
        raise R235SupportError(f"{label} is not a SHA-256 digest")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise R235SupportError(f"{label} must be a positive integer")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R235SupportError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise R235SupportError(f"{label} must be finite")
    return result


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a new receipt once, with no replacement or symlink target."""

    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        raise R235SupportError(f"immutable receipt already exists: {target}")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise R235SupportError("immutable receipt parent must already be a real directory")
    encoded = _canonical_json(dict(payload))
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        created = True
        position = 0
        while position < len(encoded):
            amount = os.write(descriptor, encoded[position:])
            if amount <= 0:
                raise OSError("short immutable receipt write")
            position += amount
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(target, 0o444)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                target.unlink()
            except OSError:
                pass
        raise R235SupportError(f"cannot write immutable receipt: {target}") from exc
    return target.resolve()


def _safe_member_name(name: str, *, label: str) -> str:
    result = str(name).removeprefix("./").strip("/")
    candidate = PurePosixPath(result)
    if (
        not result
        or "\\" in result
        or candidate.is_absolute()
        or "." in candidate.parts
        or ".." in candidate.parts
    ):
        raise R235SupportError(f"{label} is not a safe archive member path")
    return result


def _archive_members(archive_path: Path) -> dict[str, dict[str, Any]]:
    """Hash every safe regular tar member without extracting it."""

    archive = _regular_file(archive_path, label="R235 archive")
    result: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                name = _safe_member_name(member.name, label="R235 archive member")
                if name in result:
                    raise R235SupportError("R235 archive has duplicate member paths")
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                    raise R235SupportError("R235 archive has a non-regular or linked member")
                stream = source.extractfile(member)
                if stream is None:
                    raise R235SupportError("R235 archive member cannot be read")
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                if size != member.size:
                    raise R235SupportError("R235 archive member size is truncated")
                result[name] = {
                    "sha256": "sha256:" + digest.hexdigest(),
                    "size_bytes": size,
                    "mode": member.mode & 0o777,
                }
    except (OSError, tarfile.TarError) as exc:
        raise R235SupportError("R235 archive is unreadable") from exc
    if not result:
        raise R235SupportError("R235 archive has no regular members")
    return result


def _archive_member_bytes(archive_path: Path, member_name: str) -> bytes:
    archive = _regular_file(archive_path, label="R235 archive")
    name = _safe_member_name(member_name, label="R235 selected member")
    try:
        with tarfile.open(archive, "r:*") as source:
            member = next(
                (
                    candidate
                    for candidate in source.getmembers()
                    if _safe_member_name(candidate.name, label="R235 archive member") == name
                ),
                None,
            )
            if member is None:
                raise KeyError(name)
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise R235SupportError(f"R235 member is not regular: {name}")
            stream = source.extractfile(member)
            if stream is None:
                raise R235SupportError(f"R235 member cannot be read: {name}")
            return stream.read()
    except KeyError as exc:
        raise R235SupportError(f"R235 archive lacks member: {name}") from exc
    except (OSError, tarfile.TarError) as exc:
        raise R235SupportError(f"R235 member cannot be read: {name}") from exc


def _read_archive_json(archive: Path, member: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_archive_member_bytes(archive, member).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235SupportError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R235SupportError(f"{label} must be a JSON object")
    return value


def _canonical_source(path: Path, expected: Path, *, label: str) -> Path:
    source = _regular_file(path, label=label)
    if source != expected.resolve():
        raise R235SupportError(f"{label} must be the exact canonical typed source")
    return source


def _load_contracts(*, r225_contract: Path, r236_contract: Path) -> dict[str, Any]:
    """Validate the one canonical r225/r246 and r236 typed-source pair.

    The r238 Phase-1 envelope is now projected into the r225 typed source;
    there is deliberately no second mutable ``r238 contract`` selector.
    """

    r225_path = _canonical_source(r225_contract, CANONICAL_R225_PATH, label="R225 contract")
    r236_path = _canonical_source(r236_contract, CANONICAL_R236_PATH, label="R236 contract")
    if CANONICAL_R225_R246_SHA256 is None:
        raise R235SupportError("final r246 canonical R225 digest is not configured")
    if sha256_file(r225_path) != CANONICAL_R225_R246_SHA256:
        raise R235SupportError("R225 canonical r246 source digest changed")
    if sha256_file(r236_path) != CANONICAL_R236_SHA256:
        raise R235SupportError("R236 canonical source digest changed")
    r225 = _read_json(r225_path, label="R225 contract")
    r236 = _read_json(r236_path, label="R236 contract")
    if (
        r225.get("schema") != R225_SCHEMA
        or r225.get("owner_decision_revision") != R225_OWNER_DECISION_REVISION
    ):
        raise R235SupportError("R225 typed source is not the final r246 owner decision")
    for field, expected in {
        "owner_kaggle_replacement_diagnostic_revision": 235,
        "owner_canonical_libcg_revision": 236,
        "owner_phase1_submission_resources_and_two_lane_revision": 238,
        "owner_hybrid_confidence_bounded_mcts_revision": 240,
        "owner_high_confidence_frozen_direct_threshold_revision": R242_OWNER_DECISION_REVISION,
        "owner_handle_scoped_search_id_revision": R244_HANDLE_SCOPED_SEARCH_ID_REVISION,
        "owner_proven_deterministic_terminal_win_this_turn_revision": (
            R246_PROVEN_TERMINAL_WIN_REVISION
        ),
    }.items():
        if r225.get(field) != expected:
            raise R235SupportError(f"R225 typed source changed: {field}")
    relationship = r225.get("relationship_to_existing_work")
    if not isinstance(relationship, Mapping) or any(
        relationship.get(field) is not True
        for field in (
            "r242_supersedes_only_the_r240_high_confidence_frozen_direct_threshold_for_the_new_r235_replacement_package",
            "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible",
            "r242_does_not_modify_r234_r236_r238_r235_continuation_or_r229_bo1000",
            "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229",
            "r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries",
            "r246_supersedes_only_ambiguous_r235_mcts_root_selection_after_a_valid_deterministic_terminal_win_this_turn_proof",
            "r246_does_not_change_r242_high_confidence_direct_before_child_or_any_r229_bo1000_lifecycle",
        )
    ):
        raise R235SupportError("R225 r242/r244/r246 replacement boundary changed")
    replacement = r225.get("replacement_kaggle_diagnostic")
    if not isinstance(replacement, Mapping):
        raise R235SupportError("R225 replacement diagnostic is absent")
    for field, expected in {
        "revision": 235,
        "replacement_submission_count_limit": 1,
        "replacement_submission_consumed": False,
        "competition": COMPETITION,
        "submission_message_required_literal": LABEL,
        "submission_message_must_be_unique_and_exact": True,
        "queue_or_batch_submission_allowed": False,
        "automatic_retry_allowed": False,
        "automatic_copy_or_resubmission_allowed": False,
        "second_upload_allowed": False,
        "hybrid_confidence_bounded_mcts_revision": 240,
        "proven_deterministic_terminal_win_this_turn_revision": (
            R246_PROVEN_TERMINAL_WIN_REVISION
        ),
        "complete_ordered_legal_action_ceiling": ACTION_CAP,
        "all_required_local_gates_and_immutable_binding_must_pass_before_direct_api_upload": True,
        "kaggle_api_call_permitted_now_before_gates": False,
        "kaggle_upload_permitted_now_before_gates": False,
    }.items():
        if replacement.get(field) != expected:
            raise R235SupportError(f"R225 replacement authority changed: {field}")
    required_receipts = replacement.get("required_local_gate_receipts")
    if (
        not isinstance(required_receipts, list)
        or not set(R240_REQUIRED_REGRESSION_RECEIPTS) <= set(required_receipts)
        or not set(R246_REQUIRED_REGRESSION_RECEIPTS) <= set(required_receipts)
    ):
        raise R235SupportError("R225 replacement lacks required R240/R246 regression receipts")
    phase1 = r225.get("phase1_submission_environment")
    if not isinstance(phase1, Mapping) or any(
        phase1.get(key) != value
        for key, value in {
            "owner_decision_revision": 238,
            **PHASE1_RESOURCES,
            "resource_probe_and_archive_size_receipt_required": True,
            "resource_mismatch_or_archive_over_limit_behavior": "hard_fail_closed_and_do_not_upload",
            "gpu_or_os_python_environment_is_not_inferred_from_the_reported_submission_resource_values": True,
        }.items()
    ) or "gpu_available" in phase1:
        raise R235SupportError("R225 Phase-1 resource envelope changed")
    local = r225.get("local_preflight")
    if not isinstance(local, Mapping) or any(
        local.get(key) != LANE_COUNT
        for key in (
            "required_simulator_search_lane_count",
            "required_internal_agent_start_simulator_search_arena_count_per_child",
            "required_search_begin_call_count_per_ambiguous_mcts_decision",
            "required_distinct_internal_agent_start_handle_identity_count",
            "required_distinct_handle_identity_first_search_id_composite_state_count",
        )
    ):
        raise R235SupportError("R225 local preflight is not exactly two-lane")
    if any(
        local.get(field) is not expected
        for field, expected in {
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "per_lane_handle_scoped_search_id_chains_required": True,
        }.items()
    ):
        raise R235SupportError("R225 does not retain the r244 handle-scoped SearchId contract")
    r238_receipt = local.get("r238_two_lane_receipt_contract")
    if not isinstance(r238_receipt, Mapping):
        raise R235SupportError("R225 lacks the r238 two-lane receipt contract")
    exact_counts = r238_receipt.get("normal_mcts_decision_exact_counts")
    if not isinstance(exact_counts, Mapping) or any(
        exact_counts.get(field) != LANE_COUNT
        for field in (
            "requested_simulator_lane_count",
            "active_simulator_lane_count",
            "arena_count",
            "unique_handle_count",
            "distinct_handle_identity_count",
            "distinct_handle_scoped_first_search_id_composite_state_count",
            "search_begin_calls",
            "search_end_calls",
        )
    ):
        raise R235SupportError("R225 r238 receipt contract is not exactly two-lane")
    if any(
        r238_receipt.get(field) != expected
        for field, expected in {
            "normal_mcts_decision_required_fields": R244_TWO_LANE_RECEIPT_REQUIRED_FIELDS,
            "normal_mcts_decision_lane_ids": [0, 1],
            "normal_mcts_decision_microbatch_size_range": [1, 2],
            "normal_mcts_decision_max_simulator_calls_in_flight_range": [1, 2],
            "normal_mcts_decision_minimum_counts": {"search_release_calls": LANE_COUNT},
            "normal_mcts_decision_per_lane_vectors_exact_length": {
                "per_lane_depth": LANE_COUNT,
                "per_lane_search_id_chains": LANE_COUNT,
                "per_lane_handle_identities": LANE_COUNT,
                "per_lane_first_search_ids": LANE_COUNT,
                "handle_scoped_first_search_id_composite_states": LANE_COUNT,
            },
            "search_id_identity_contract": R244_SEARCH_ID_IDENTITY_CONTRACT,
            "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
        }.items()
    ):
        raise R235SupportError("R225 r238 two-lane receipt details changed")
    gameplay = local.get("full_gameplay_shared_tree_contract")
    if not isinstance(gameplay, Mapping) or any(
        gameplay.get(field) is not expected
        for field, expected in {
            "retain_exact_lane_handle_identity_search_id_tuple_across_repeated_depth_waves": True,
            "per_lane_handle_scoped_search_id_chains_required": True,
            "two_distinct_handle_identity_first_search_id_composite_states_required": True,
            "global_raw_search_id_integer_distinctness_required": False,
            "r246_proven_deterministic_terminal_win_this_turn_is_an_in_search_early_stop_not_a_r242_direct_bypass": True,
            "r246_valid_terminal_win_proof_may_bypass_only_the_normal_adaptive_stop_thresholds_after_two_lane_initialization": True,
            "r246_valid_terminal_win_proof_requires_at_least_one_exact_terminal_backup_but_not_the_normal_eight_backup_leader_or_both_lane_progress_threshold": True,
        }.items()
    ):
        raise R235SupportError("R225 full-game r244/r246 topology changed")
    hybrid = local.get("r240_hybrid_scheduler")
    if not isinstance(hybrid, Mapping) or dict(hybrid) != R246_LOCAL_SCHEDULER:
        raise R235SupportError("R225 r242/r246 high-confidence/adaptive scheduler changed")
    local_continuation = local.get("deterministic_continuation")
    if not isinstance(local_continuation, Mapping) or dict(local_continuation) != MANIFEST_DETERMINISTIC_CONTINUATION:
        raise R235SupportError("R225 deterministic-continuation preflight changed")
    replacement_scheduler = replacement.get("r240_hybrid_scheduler")
    if not isinstance(replacement_scheduler, Mapping) or dict(replacement_scheduler) != R246_REPLACEMENT_SCHEDULER:
        raise R235SupportError("R225 replacement r242/r246 scheduler changed")
    continuation = replacement.get("deterministic_continuation")
    if not isinstance(continuation, Mapping) or dict(continuation) != R242_REPLACEMENT_DETERMINISTIC_CONTINUATION:
        raise R235SupportError("R225 deterministic-continuation contract changed")

    action_space = r225.get("complete_ordered_action_space_contract")
    if not isinstance(action_space, Mapping) or any(
        action_space.get(field) != expected
        for field, expected in {
            "complete_ordered_legal_action_ceiling": ACTION_CAP,
            "ceiling_applies_at_root_and_every_private_leaf": True,
            "sampling_pruning_or_reinterpretation_of_legal_choices_allowed": False,
            "root_over_cap_behavior": "hard_fail_nonzero",
            "private_leaf_over_cap_behavior": (
                "contained_degraded_parent_direct_fallback_only_after_validated_"
                "precomputed_root_direct_action_and_exact_child_reap"
            ),
        }.items()
    ):
        raise R235SupportError("R225 complete-action contract changed")
    replacement_topology = replacement.get("phase1_simulator_search")
    if not isinstance(replacement_topology, Mapping) or any(
        replacement_topology.get(field) != expected
        for field, expected in {
            "exactly_two_simulator_search_lanes_required": True,
            "required_simulator_search_lane_count": LANE_COUNT,
            "required_active_simulator_search_lane_count": LANE_COUNT,
            "required_internal_agent_start_simulator_search_arena_count": LANE_COUNT,
            "required_unique_raw_handle_count": LANE_COUNT,
            "required_distinct_per_lane_handle_identity_count": LANE_COUNT,
            "required_search_begin_call_count": LANE_COUNT,
            "required_distinct_handle_identity_first_search_id_composite_state_count": LANE_COUNT,
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "first_search_id_identity_composite": "(handle_identity, first_search_id)",
            "per_lane_handle_scoped_search_id_chains_required": True,
            "maximum_frontier_leaves_per_frozen_evaluator_batch": LANE_COUNT,
            "one_shared_logical_mcts_tree_required": True,
            "one_lane_serial_fallback_eight_lane_topology_or_partial_lane_mcts_authority_allowed": False,
        }.items()
    ):
        raise R235SupportError("R225 replacement is not the exact r238 two-lane topology")
    authority = r225.get("authority")
    if not isinstance(authority, Mapping) or any(
        authority.get(field) is not False
        for field in (
            "kaggle_api_call_permitted_now_before_preconditions",
            "kaggle_upload_permitted_now_before_preconditions",
            "kaggle_api_call_permitted_now",
            "kaggle_upload_permitted_now",
            "kaggle_queue_submission_permitted",
            "automatic_kaggle_submission_allowed",
            "kaggle_retry_or_copy_permitted",
            "second_kaggle_upload_permitted",
            "training_or_gradient_updates_authorized",
            "selector_change_authorized",
            "promotion_authorized",
        )
    ):
        raise R235SupportError("R225 authority is not fail-closed")

    if r236.get("schema") != R236_SCHEMA or r236.get("owner_decision_revision") != 236:
        raise R235SupportError("R236 canonical libcg source changed")
    native_libraries = r236.get("canonical_native_libraries")
    if not isinstance(native_libraries, Mapping) or set(native_libraries) != {
        "linux_x86_64", "linux_aarch64", "macos_arm64", "windows_x86_64"
    }:
        raise R235SupportError("R236 does not bind the complete canonical native set")
    native_members: dict[str, dict[str, Any]] = {}
    for platform, payload in native_libraries.items():
        if not isinstance(payload, Mapping):
            raise R235SupportError(f"R236 {platform} native member is malformed")
        member_path = _safe_member_name(
            str(payload.get("package_relative_path") or ""), label=f"R236 {platform} member"
        )
        if member_path in native_members:
            raise R235SupportError("R236 repeats a native package member")
        native_members[member_path] = {
            "platform": platform,
            "sha256": _digest(payload.get("sha256"), label=f"R236 {platform} digest"),
            "size_bytes": _positive_int(payload.get("size_bytes"), label=f"R236 {platform} size"),
        }
    linux = native_libraries.get("linux_x86_64")
    if not isinstance(linux, Mapping):
        raise R235SupportError("R236 Linux libcg identity is absent")
    linux_digest = _digest(linux.get("sha256"), label="R236 Linux libcg digest")
    linux_size = _positive_int(linux.get("size_bytes"), label="R236 Linux libcg size")
    base = r225.get("exact_frozen_base")
    if not isinstance(base, Mapping):
        raise R235SupportError("R225 frozen base is absent")
    model_digest = _digest(base.get("r195_checkpoint_sha256"), label="R195 checkpoint")
    tree_digest = _digest(base.get("r195_matchup_tree_sha256"), label="R195 tree")
    if base.get("stock_libcg_sha256") != linux_digest or base.get("stock_libcg_size_bytes") != linux_size:
        raise R235SupportError("R225 frozen libcg does not match R236")
    return {
        "r225": {"path": str(r225_path), "sha256": sha256_file(r225_path), "payload": r225},
        "r236": {
            "path": str(r236_path),
            "sha256": sha256_file(r236_path),
            "payload": r236,
            "linux_sha256": linux_digest,
            "linux_size_bytes": linux_size,
            "native_members": native_members,
        },
        "model_sha256": model_digest,
        "tree_sha256": tree_digest,
    }


def _inspect_package(
    *, archive_path: Path, manifest_path: Path, manifest_member: str, contracts: Mapping[str, Any]
) -> dict[str, Any]:
    archive = _regular_file(archive_path, label="R235 archive")
    manifest_file = _regular_file(manifest_path, label="R238 external manifest")
    if archive.stat().st_size > PHASE1_ARCHIVE_MAX_BYTES:
        raise R235SupportError("R235 archive exceeds the Phase-1 archive cap")
    member_name = _safe_member_name(manifest_member, label="R238 manifest selector")
    external_bytes = manifest_file.read_bytes()
    try:
        manifest = json.loads(external_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235SupportError("R238 external manifest is unreadable JSON") from exc
    if not isinstance(manifest, dict):
        raise R235SupportError("R238 external manifest must be a JSON object")
    members = _archive_members(archive)
    required = {
        member_name,
        "main.py",
        "r195_direct_main.py",
        "turn_order_profile.json",
        "model.pt",
        "matchup_tree.json",
        "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
        *contracts["r236"]["native_members"],
    }
    absent = sorted(required - set(members))
    if absent:
        raise R235SupportError("R235 archive lacks required member(s): " + ", ".join(absent))
    if _archive_member_bytes(archive, member_name) != external_bytes:
        raise R235SupportError("external manifest does not byte-match the archive member")
    r225_member = "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
    if (
        members[r225_member]["sha256"] != contracts["r225"]["sha256"]
        or members[r225_member]["size_bytes"]
        != Path(str(contracts["r225"]["path"])).stat().st_size
    ):
        raise R235SupportError("R235 archive does not carry the exact canonical r225/r246 source")

    for field, expected in {
        "schema": R238_MANIFEST_SCHEMA,
        "role": R238_MANIFEST_ROLE,
        "required_label": LABEL,
        "complete_action_cap": ACTION_CAP,
        "lane_count": LANE_COUNT,
    }.items():
        if manifest.get(field) != expected:
            raise R235SupportError(f"R238 manifest changed: {field}")
    if manifest.get("r225_typed_contract") != {
        "path": "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
        "schema": R225_SCHEMA,
        "sha256": contracts["r225"]["sha256"],
    }:
        raise R235SupportError("R238 manifest does not bind the exact final R225 typed source")
    # The manifest carries the r236 projection before a binding exists.  Check
    # it independently here rather than trusting a later binding to repair a
    # stale/mixed native-library declaration.
    r236_payload = contracts["r236"]["payload"]
    upstream = r236_payload.get("upstream_provenance") if isinstance(r236_payload, Mapping) else None
    native_libraries = (
        r236_payload.get("canonical_native_libraries") if isinstance(r236_payload, Mapping) else None
    )
    required_exports = r236_payload.get("required_native_exports") if isinstance(r236_payload, Mapping) else None
    embedded_r236 = manifest.get("canonical_libcg_contract")
    if not isinstance(embedded_r236, Mapping) or any(
        embedded_r236.get(field) != expected
        for field, expected in {
            "schema": R236_SCHEMA,
            "typed_source": "state/canonical-libcg-r236.json",
            "owner_decision_revision": 236,
            "upstream_provenance": upstream,
            "canonical_native_libraries": native_libraries,
            "required_native_exports": required_exports,
        }.items()
    ):
        raise R235SupportError("R238 manifest does not embed the exact canonical R236 contract")
    embedded_scope = embedded_r236.get("scope")
    if not isinstance(embedded_scope, Mapping) or any(
        embedded_scope.get(field) is not expected
        for field, expected in {
            "r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary": True,
            "all_four_platform_native_members_must_be_bound": True,
            "mixed_old_and_new_library_sets_allowed": False,
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
        }.items()
    ):
        raise R235SupportError("R238 manifest R236 scope changed")
    embedded_preflight = embedded_r236.get("r225_package_preflight")
    if not isinstance(embedded_preflight, Mapping) or any(
        embedded_preflight.get(field) != expected
        for field, expected in {
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
            "all_four_canonical_native_members_checksum_and_size_verified": True,
            "old_or_mixed_native_members_rejected": True,
            "required_native_exports": required_exports,
        }.items()
    ):
        raise R235SupportError("R238 manifest R236 preflight changed")
    if not isinstance(native_libraries, Mapping):
        raise R235SupportError("canonical R236 native set is malformed")
    expected_native_members = {
        str(member_path): {
            "platform": platform,
            "wheel_member": payload.get("wheel_member"),
            "sha256": payload.get("sha256"),
            "size_bytes": payload.get("size_bytes"),
        }
        for platform, payload in native_libraries.items()
        if isinstance(payload, Mapping)
        for member_path in (payload.get("package_relative_path"),)
    }
    expected_native_hashes = {
        str(member_path): payload.get("sha256")
        for payload in native_libraries.values()
        if isinstance(payload, Mapping)
        for member_path in (payload.get("package_relative_path"),)
    }
    if (
        manifest.get("canonical_native_members") != expected_native_members
        or manifest.get("canonical_native_member_sha256") != expected_native_hashes
    ):
        raise R235SupportError("R238 manifest R236 native member projection changed")
    frozen_base = contracts["r225"]["payload"].get("exact_frozen_base")
    expected_manifest_base = {
        "r195_bundle_sha256": frozen_base.get("r195_bundle_sha256") if isinstance(frozen_base, Mapping) else None,
        "r195_checkpoint_sha256": contracts["model_sha256"],
        "r195_matchup_tree_sha256": contracts["tree_sha256"],
        "stock_libcg_sha256": contracts["r236"]["linux_sha256"],
        "stock_libcg_size_bytes": contracts["r236"]["linux_size_bytes"],
    }
    manifest_base = manifest.get("exact_frozen_base")
    if not isinstance(manifest_base, Mapping) or any(
        manifest_base.get(field) != expected for field, expected in expected_manifest_base.items()
    ):
        raise R235SupportError("R238 manifest frozen R195/R236 base changed")
    # The current r238 schema owns the exact operational lane count.  Accept
    # no legacy field and require any duplicate spelling to agree with it.
    for field in ("simulator_search_lane_count", "active_lane_count", "requested_lane_count"):
        if field in manifest and manifest.get(field) != LANE_COUNT:
            raise R235SupportError(f"R238 manifest is not exactly two-lane: {field}")
    resource_bounds = manifest.get("phase1_kaggle_resource_bounds")
    if (
        not isinstance(resource_bounds, Mapping)
        or dict(resource_bounds) != PHASE1_MANIFEST_RESOURCE_BOUNDS
    ):
        raise R235SupportError("R238 manifest Phase-1 resource receipt is absent or mismatched")
    if manifest.get("required_search_lifecycle_counts") != {
        "search_begin_calls": LANE_COUNT,
        "search_end_calls": LANE_COUNT,
        "search_release_calls": LANE_COUNT,
    }:
        raise R235SupportError("R238 manifest does not require the exact two-lane lifecycle")
    if manifest.get("r240_required_preflight_receipts") != list(R240_REQUIRED_REGRESSION_RECEIPTS):
        raise R235SupportError("R238 manifest lacks the exact R240 regression receipt contract")
    if (
        manifest.get("owner_proven_deterministic_terminal_win_this_turn_revision")
        != R246_PROVEN_TERMINAL_WIN_REVISION
        or manifest.get(R246_TERMINAL_WIN_CONTRACT_KEY) != R246_LOCAL_TERMINAL_WIN_CONTRACT
        or manifest.get("r246_required_preflight_receipts")
        != list(R246_REQUIRED_REGRESSION_RECEIPTS)
    ):
        raise R235SupportError("R238 manifest lacks the exact R246 terminal-win receipt contract")
    r240 = manifest.get("r240_hybrid_scheduler")
    if not isinstance(r240, Mapping) or dict(r240) != R240_MANIFEST_SCHEDULER:
        raise R235SupportError("R238 manifest r242/r246 high-confidence/adaptive scheduler mismatch")
    continuation = manifest.get("deterministic_continuation")
    if not isinstance(continuation, Mapping) or dict(continuation) != MANIFEST_DETERMINISTIC_CONTINUATION:
        raise R235SupportError("R238 manifest deterministic-continuation mismatch")
    broker = manifest.get("broker_contract")
    if not isinstance(broker, Mapping) or any(
        broker.get(field) != expected
        for field, expected in {
            "complete_action_cap": ACTION_CAP,
            "degraded_fallback_marker": "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED",
            "search_seconds": 2.0,
            "action_timeout_seconds": 4.0,
        }.items()
    ):
        raise R235SupportError("R238 manifest bounded broker contract mismatch")
    if manifest.get("entrypoint_sha256") != members["main.py"]["sha256"]:
        raise R235SupportError("R238 manifest main.py digest mismatch")
    direct = manifest.get("direct_entrypoint")
    if not isinstance(direct, Mapping) or direct.get("path") != "r195_direct_main.py" or direct.get(
        "sha256"
    ) != members["r195_direct_main.py"]["sha256"]:
        raise R235SupportError("R238 manifest direct-entrypoint digest mismatch")
    profile = _read_archive_json(archive, "turn_order_profile.json", label="turn-order profile")
    if profile != {
        "schema": "poke_bot.submission_turn_order_profile/v1",
        "turn_order_preference": "first_if_allowed",
    }:
        raise R235SupportError("R235 package is not the exact first-preferring profile")
    if members["model.pt"]["sha256"] != contracts["model_sha256"]:
        raise R235SupportError("R235 package checkpoint digest mismatch")
    if members["matchup_tree.json"]["sha256"] != contracts["tree_sha256"]:
        raise R235SupportError("R235 package matchup-tree digest mismatch")
    for member_path, expected in contracts["r236"]["native_members"].items():
        observed = members[member_path]
        if (
            observed["sha256"] != expected["sha256"]
            or observed["size_bytes"] != expected["size_bytes"]
        ):
            raise R235SupportError(
                f"R235 package does not use the exact R236 native member: {member_path}"
            )
    return {
        "path": str(archive),
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "manifest_member": member_name,
        "manifest": manifest,
        "members": members,
    }


def _immutable(path: Path, *, label: str) -> Path:
    source = _regular_file(path, label=label)
    if source.stat().st_mode & 0o222:
        raise R235SupportError(f"{label} is writable, not immutable")
    return source


def _validate_r244_handle_scoped_projection(value: object, *, label: str) -> None:
    """Accept SearchIds only in their owning AgentStart-handle namespace.

    Official libcg may return raw first IDs ``[0, 0]`` from two distinct
    handles.  The receipt must therefore prove two distinct composite states,
    rather than incorrectly requiring globally distinct raw integers.
    """

    if not isinstance(value, Mapping):
        raise R235SupportError(f"{label} lacks an R244 handle-scoped projection")
    expected_keys = {
        "per_lane_handle_identities",
        "per_lane_search_id_chains",
        "per_lane_first_search_ids",
        "handle_scoped_first_search_id_composite_states",
        "distinct_handle_identity_count",
        "distinct_handle_scoped_first_search_id_composite_state_count",
        "globally_distinct_raw_search_id_integers_required",
    }
    if set(value) != expected_keys:
        raise R235SupportError(f"{label} handle-scoped projection fields changed")
    handles = value.get("per_lane_handle_identities")
    chains = value.get("per_lane_search_id_chains")
    first_ids = value.get("per_lane_first_search_ids")
    if not isinstance(handles, list) or not isinstance(chains, list) or not isinstance(first_ids, list):
        raise R235SupportError(f"{label} handle-scoped vectors are malformed")
    if not (len(handles) == len(chains) == len(first_ids) == LANE_COUNT):
        raise R235SupportError(f"{label} must retain exactly two handle/search chains")
    checked_handles: list[int | str] = []
    checked_chains: list[list[int]] = []
    for lane, (handle, chain) in enumerate(zip(handles, chains, strict=True)):
        if (
            isinstance(handle, bool)
            or not isinstance(handle, (int, str))
            or (isinstance(handle, str) and not handle)
        ):
            raise R235SupportError(f"{label} has an invalid handle identity at lane {lane}")
        if not isinstance(chain, list) or not chain:
            raise R235SupportError(f"{label} has an empty SearchId chain at lane {lane}")
        checked_chain: list[int] = []
        for index, search_id in enumerate(chain):
            if (
                isinstance(search_id, bool)
                or not isinstance(search_id, int)
                or search_id < 0
            ):
                raise R235SupportError(
                    f"{label} has an invalid SearchId at lane {lane}, index {index}"
                )
            checked_chain.append(search_id)
        checked_handles.append(handle)
        checked_chains.append(checked_chain)
    if len(set(checked_handles)) != LANE_COUNT:
        raise R235SupportError(f"{label} does not retain two distinct handle identities")
    expected_first_ids = [chain[0] for chain in checked_chains]
    if first_ids != expected_first_ids:
        raise R235SupportError(f"{label} first SearchIds do not match the chains")
    expected_composites = [
        {
            "lane_id": lane,
            "handle_identity": checked_handles[lane],
            "first_search_id": expected_first_ids[lane],
        }
        for lane in range(LANE_COUNT)
    ]
    if value.get("handle_scoped_first_search_id_composite_states") != expected_composites:
        raise R235SupportError(f"{label} handle-scoped first SearchId composites changed")
    if len({(handle, first_id) for handle, first_id in zip(checked_handles, expected_first_ids, strict=True)}) != LANE_COUNT:
        raise R235SupportError(f"{label} does not retain two distinct handle/SearchId composites")
    if (
        value.get("distinct_handle_identity_count") != LANE_COUNT
        or value.get("distinct_handle_scoped_first_search_id_composite_state_count") != LANE_COUNT
        or value.get("globally_distinct_raw_search_id_integers_required") is not False
    ):
        raise R235SupportError(f"{label} r244 SearchId identity counters changed")


def _r246_action(value: object, *, label: str) -> list[int]:
    """Validate one exact nonempty factorized root action projection."""

    if not isinstance(value, list) or not value:
        raise R235SupportError(f"{label} must be a nonempty factorized action list")
    action: list[int] = []
    for index, part in enumerate(value):
        if isinstance(part, bool) or not isinstance(part, int) or part < 0:
            raise R235SupportError(
                f"{label} has an invalid factorized action part at index {index}"
            )
        action.append(part)
    return action


def _r246_actor_seat(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise R235SupportError(f"{label} must be seat 0 or 1")
    return value


def _r246_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R235SupportError(f"{label} must be a nonnegative integer")
    return value


def _validate_r246_terminal_win_projection(value: object, *, label: str) -> None:
    """Require the binder's exact one-proof terminal-win projection.

    The binding intentionally stores a normalized projection, not a raw
    mutable preflight receipt.  Rechecking it here ensures the guard and
    later evidence collection cannot treat an arbitrary passed gate digest as
    terminal-win authority.
    """

    if not isinstance(value, Mapping):
        raise R235SupportError(f"{label} lacks an R246 terminal-win projection")
    if R246_LEGACY_CLEANUP_COMPLETED_FIELD in value:
        raise R235SupportError(
            f"{label} uses the legacy cleanup field "
            f"{R246_LEGACY_CLEANUP_COMPLETED_FIELD}; expected "
            f"{R246_CLEANUP_COMPLETE_FIELD}"
        )
    expected_keys = {
        "owner_proven_deterministic_terminal_win_this_turn_revision",
        "proven_deterministic_terminal_win_this_turn_regression_passed",
        "stop_reason",
        "two_lane_topology_initialized_before_terminal_win_override",
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_release_calls",
        "search_end_calls",
        "completed_root_backup_count",
        "terminal_win_proof_count",
        "proven_deterministic_terminal_win_this_turn_stop_count",
        "terminal_win_proof_backed_up_into_shared_root_tree",
        "terminal_leaf_returned_by_exact_stock_simulator",
        "parent_validated_current_root_observation_legal_fingerprint_and_actor",
        R246_CLEANUP_COMPLETE_FIELD,
        "outstanding_virtual_loss",
        "two_independent_lane_proofs_required",
        "exhaustive_legal_action_scan_required",
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof",
        "terminal_win_proof",
    }
    if set(value) != expected_keys:
        raise R235SupportError(f"{label} terminal-win projection fields changed")
    required = {
        "owner_proven_deterministic_terminal_win_this_turn_revision": (
            R246_PROVEN_TERMINAL_WIN_REVISION
        ),
        "proven_deterministic_terminal_win_this_turn_regression_passed": True,
        "stop_reason": R246_TERMINAL_WIN_STOP_REASON,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "requested_simulator_lane_count": LANE_COUNT,
        "active_simulator_lane_count": LANE_COUNT,
        "arena_count": LANE_COUNT,
        "unique_handle_count": LANE_COUNT,
        "search_begin_calls": LANE_COUNT,
        "search_release_calls": LANE_COUNT,
        "search_end_calls": LANE_COUNT,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
        R246_CLEANUP_COMPLETE_FIELD: True,
        "outstanding_virtual_loss": 0,
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
    }
    if any(value.get(field) != expected for field, expected in required.items()):
        raise R235SupportError(f"{label} terminal-win projection values changed")
    completed_backup_count = _r246_nonnegative_int(
        value.get("completed_root_backup_count"),
        label=f"{label}.completed_root_backup_count",
    )
    if not 1 <= completed_backup_count <= R240_NORMALIZED_SCHEDULER[
        "maximum_backups_per_decision"
    ]:
        raise R235SupportError(f"{label} terminal backup count is outside the R240 bounds")
    proof = value.get("terminal_win_proof")
    if not isinstance(proof, Mapping) or set(proof) != set(R246_TERMINAL_WIN_PROOF_FIELDS):
        raise R235SupportError(f"{label} terminal-win proof fields changed")
    expected_proof = {
        "proof_kind": R246_TERMINAL_WIN_PROOF_KIND,
        "terminal_result": "win",
        "terminal_leaf_reached": True,
        "path_no_chance_boundary": True,
        "path_no_actor_change_boundary": True,
        "path_no_opponent_boundary_crossing": True,
        "path_no_unresolved_randomness": True,
        "proof_is_deterministic": True,
    }
    if any(proof.get(field) != expected for field, expected in expected_proof.items()):
        raise R235SupportError(f"{label} terminal-win proof values changed")
    for field in ("root_observation_fingerprint", "root_legal_order_fingerprint"):
        if not isinstance(proof.get(field), str) or not proof[field]:
            raise R235SupportError(f"{label} terminal-win proof has invalid {field}")
    root_actor = _r246_actor_seat(
        proof.get("root_actor_seat"), label=f"{label}.terminal_win_proof.root_actor_seat"
    )
    root_action = _r246_action(
        proof.get("root_action"), label=f"{label}.terminal_win_proof.root_action"
    )
    selected_action = _r246_action(
        proof.get("selected_action"), label=f"{label}.terminal_win_proof.selected_action"
    )
    if selected_action != root_action:
        raise R235SupportError(f"{label} terminal-win selected action differs from root action")
    terminal_winner = _r246_actor_seat(
        proof.get("terminal_winner_seat"),
        label=f"{label}.terminal_win_proof.terminal_winner_seat",
    )
    if terminal_winner != root_actor:
        raise R235SupportError(f"{label} terminal-win winner differs from root actor")
    path_count = _r246_nonnegative_int(
        proof.get("proof_path_action_count"),
        label=f"{label}.terminal_win_proof.proof_path_action_count",
    )
    if path_count < 1:
        raise R235SupportError(f"{label} terminal-win proof has an empty action path")
    if path_count > completed_backup_count:
        raise R235SupportError(
            f"{label} terminal-win proof path exceeds completed root backups"
        )
    discovering_lane = _r246_nonnegative_int(
        proof.get("discovering_lane_id"),
        label=f"{label}.terminal_win_proof.discovering_lane_id",
    )
    if discovering_lane not in range(LANE_COUNT):
        raise R235SupportError(f"{label} terminal-win proof has an invalid discovering lane")
    path_actor_seats = proof.get("path_actor_seats")
    if not isinstance(path_actor_seats, list) or len(path_actor_seats) != path_count:
        raise R235SupportError(f"{label} terminal-win proof path length changed")
    for index, actor in enumerate(path_actor_seats):
        if _r246_actor_seat(
            actor, label=f"{label}.terminal_win_proof.path_actor_seats[{index}]"
        ) != root_actor:
            raise R235SupportError(f"{label} terminal-win proof crosses an actor boundary")


def _binding_gate_rows(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != R235_BINDING_GATE_NAMES:
        raise R235SupportError("immutable binding has a changed local-gate set")
    for name, row in value.items():
        if (
            not isinstance(row, Mapping)
            or row.get("receipt_name") != R235_BINDING_GATE_RECEIPT_NAMES[name]
            or row.get("passed") is not True
        ):
            raise R235SupportError(f"immutable binding gate is not passed: {name}")
        _digest(row.get("sha256"), label=f"immutable binding gate digest {name}")
    for name in (
        "two_lane_topology",
        "official_libcg_handle_scoped_search_id_identity",
    ):
        row = value[name]
        _validate_r244_handle_scoped_projection(
            row.get("validated_counter_projection"), label=f"immutable binding gate {name}"
        )
    _validate_r246_terminal_win_projection(
        value[R246_TERMINAL_WIN_GATE].get("validated_counter_projection"),
        label=f"immutable binding gate {R246_TERMINAL_WIN_GATE}",
    )
    return value


def validate_r235_candidate(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    r225_contract: Path,
    r236_contract: Path,
) -> dict[str, Any]:
    """Validate the pre-binding candidate package and canonical contracts.

    This intentionally stops before any immutable binding exists.  It is the
    only validation needed to create the archive's digest-bound go-first
    sidecar, which is itself one of the binder's required input receipts.
    """

    contracts = _load_contracts(r225_contract=r225_contract, r236_contract=r236_contract)
    package = _inspect_package(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        contracts=contracts,
    )
    return {"package": package, "contracts": contracts}


def validate_r235_binding(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    binding_path: Path,
    r225_contract: Path,
    r236_contract: Path,
) -> dict[str, Any]:
    """Revalidate the binder-owned immutable receipt against exact current bytes."""

    candidate_context = validate_r235_candidate(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        r225_contract=r225_contract,
        r236_contract=r236_contract,
    )
    contracts = candidate_context["contracts"]
    package = candidate_context["package"]
    binding_file = _immutable(binding_path, label="R235 immutable binding")
    binding = _read_json(binding_file, label="R235 immutable binding")
    if (
        binding.get("schema") != R235_BINDING_SCHEMA
        or binding.get("status") != "local_gates_bound_not_submitted"
        or binding.get("immutable") is not True
        or binding.get("write_once") is not True
        or binding.get("owner_decision_revision") != R225_OWNER_DECISION_REVISION
    ):
        raise R235SupportError("immutable binding schema/status changed")
    candidate = binding.get("candidate_package")
    if not isinstance(candidate, Mapping):
        raise R235SupportError("immutable binding lacks candidate package")
    expected_manifest = {
        "schema": "poke_bot.r235_r236_archive_member_manifest/v1",
        "archive_sha256": package["sha256"],
        "members": package["members"],
    }
    expected_member_manifest_sha = _sha256_bytes(_canonical_json(expected_manifest))
    for field, expected in {
        "archive_sha256": package["sha256"],
        "archive_size_bytes": package["size_bytes"],
        "member_manifest_member": package["manifest_member"],
        "member_manifest_sha256": package["manifest_sha256"],
        "manifest_schema": R238_MANIFEST_SCHEMA,
        "manifest_role": R238_MANIFEST_ROLE,
        "entrypoint_member": "main.py",
        "entrypoint_sha256": package["members"]["main.py"]["sha256"],
        "computed_archive_member_manifest_sha256": expected_member_manifest_sha,
    }.items():
        if candidate.get(field) != expected:
            raise R235SupportError(f"immutable binding candidate package mismatch: {field}")
    if candidate.get("computed_archive_member_manifest") != expected_manifest:
        raise R235SupportError("immutable binding archive member map mismatch")
    typed = binding.get("typed_contracts")
    if not isinstance(typed, Mapping):
        raise R235SupportError("immutable binding typed contracts are absent")
    expected_typed_keys = {
        "r225_historical_r246_typed_contract",
        "r236_canonical_libcg_typed_contract",
        "r225_r246_replacement_summary",
    }
    if set(typed) != expected_typed_keys:
        raise R235SupportError("immutable binding typed-contract field set changed")
    r225_typed = typed.get("r225_historical_r246_typed_contract")
    r236_typed = typed.get("r236_canonical_libcg_typed_contract")
    if not isinstance(r225_typed, Mapping) or dict(r225_typed) != {
        "canonical_relative_path": "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
        "resolved_path": contracts["r225"]["path"],
        "schema": R225_SCHEMA,
        "sha256": contracts["r225"]["sha256"],
        "expected_sha256": contracts["r225"]["sha256"],
        "archive_member_bound": True,
    }:
        raise R235SupportError("immutable binding r225/r246 typed-contract mismatch")
    if not isinstance(r236_typed, Mapping) or dict(r236_typed) != {
        "canonical_relative_path": "state/canonical-libcg-r236.json",
        "resolved_path": contracts["r236"]["path"],
        "schema": R236_SCHEMA,
        "sha256": contracts["r236"]["sha256"],
        "expected_sha256": contracts["r236"]["sha256"],
    }:
        raise R235SupportError("immutable binding r236 typed-contract mismatch")
    r225_summary = typed.get("r225_r246_replacement_summary")
    expected_summary = {
        "owner_decision_revision": R225_OWNER_DECISION_REVISION,
        "replacement_submission_count_limit": 1,
        "replacement_submission_consumed": False,
        "competition": COMPETITION,
        "submission_message_required_literal": LABEL,
        "complete_ordered_action_cap": ACTION_CAP,
        "canonical_libcg_contract_sha256": contracts["r236"]["sha256"],
    }
    if not isinstance(r225_summary, Mapping) or dict(r225_summary) != expected_summary:
        raise R235SupportError("immutable binding r225/r246 replacement summary mismatch")
    canonical = binding.get("canonical_libcg_r236")
    r236_payload = contracts["r236"]["payload"]
    upstream = r236_payload.get("upstream_provenance") if isinstance(r236_payload, Mapping) else None
    native_libraries = r236_payload.get("canonical_native_libraries") if isinstance(r236_payload, Mapping) else None
    expected_canonical = {
        "package_version": upstream.get("package_version") if isinstance(upstream, Mapping) else None,
        "wheel_filename": upstream.get("wheel_filename") if isinstance(upstream, Mapping) else None,
        "official_wheel_sha256": upstream.get("wheel_sha256") if isinstance(upstream, Mapping) else None,
        "required_native_exports": r236_payload.get("required_native_exports")
        if isinstance(r236_payload, Mapping)
        else None,
        "members": native_libraries,
    }
    if not isinstance(canonical, Mapping) or dict(canonical) != expected_canonical:
        raise R235SupportError("immutable binding canonical R236 native set mismatch")
    linux = (canonical.get("members") or {}).get("linux_x86_64")
    if not isinstance(linux, Mapping) or (
        linux.get("sha256") != contracts["r236"]["linux_sha256"]
        or linux.get("size_bytes") != contracts["r236"]["linux_size_bytes"]
    ):
        raise R235SupportError("immutable binding canonical R236 Linux libcg mismatch")
    frozen = binding.get("frozen_r195_identity")
    base = contracts["r225"]["payload"].get("exact_frozen_base")
    if not isinstance(frozen, Mapping) or not isinstance(base, Mapping) or dict(frozen) != {
        "bundle_sha256": base.get("r195_bundle_sha256"),
        "checkpoint_sha256": contracts["model_sha256"],
        "matchup_tree_sha256": contracts["tree_sha256"],
    }:
        raise R235SupportError("immutable binding frozen r195 identity mismatch")
    action = binding.get("action_space")
    if not isinstance(action, Mapping) or any(
        action.get(field) != expected
        for field, expected in {
            "complete_ordered_action_cap": ACTION_CAP,
            "applies_at_root_and_every_private_leaf": True,
            "sampling_pruning_or_reinterpretation_allowed": False,
            "root_over_cap_behavior": "hard_fail_nonzero",
            "private_leaf_over_cap_behavior": (
                "contained_degraded_parent_direct_fallback_only_after_validated_"
                "precomputed_root_direct_action_and_exact_child_reap"
            ),
        }.items()
    ):
        raise R235SupportError("immutable binding action-space contract changed")
    topology = binding.get("simulator_search_topology")
    if not isinstance(topology, Mapping) or (
        topology.get("lane_count") != LANE_COUNT
        or topology.get("historical_eight_lane_manifest_or_receipt_accepted") is not False
    ):
        raise R235SupportError("immutable binding is not exactly two-lane")
    resources = topology.get("phase1_submission_environment")
    if not isinstance(resources, Mapping) or dict(resources) != {
        **PHASE1_RESOURCES,
        "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
    }:
        raise R235SupportError("immutable binding lacks the exact Phase-1 resource receipt")
    if topology.get("phase1_manifest_resource_bounds") != PHASE1_MANIFEST_RESOURCE_BOUNDS:
        raise R235SupportError("immutable binding Phase-1 manifest resource receipt mismatch")
    expected_normalized_scheduler = {
        **R240_NORMALIZED_SCHEDULER,
        "high_confidence_mode": "high_confidence_frozen_direct",
        "legacy_fixed_eight_second_window_accepted": False,
    }
    if binding.get("r240_hybrid_scheduler") != expected_normalized_scheduler:
        raise R235SupportError("immutable binding r242 normalized scheduler mismatch")
    if binding.get("r240_manifest_scheduler") != R240_MANIFEST_SCHEDULER:
        raise R235SupportError("immutable binding r242 manifest scheduler mismatch")
    if binding.get("r244_handle_scoped_search_id_identity") != R244_SEARCH_ID_IDENTITY_CONTRACT:
        raise R235SupportError("immutable binding r244 handle-scoped SearchId identity mismatch")
    if binding.get(R246_TERMINAL_WIN_CONTRACT_KEY) != R246_LOCAL_TERMINAL_WIN_CONTRACT:
        raise R235SupportError("immutable binding r246 terminal-win contract mismatch")
    if binding.get("deterministic_continuation") != R235_GATE_DETERMINISTIC_CONTINUATION:
        raise R235SupportError("immutable binding deterministic continuation mismatch")
    if binding.get("manifest_deterministic_continuation") != MANIFEST_DETERMINISTIC_CONTINUATION:
        raise R235SupportError("immutable binding manifest deterministic continuation mismatch")
    required_submission = binding.get("required_submission")
    if not isinstance(required_submission, Mapping) or required_submission != {
        "competition": COMPETITION,
        "label": LABEL,
        "go_first_preference": "first_if_allowed",
    }:
        raise R235SupportError("immutable binding submission identity changed")
    _binding_gate_rows(binding.get("local_gate_receipts"))
    authorization = binding.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(field) != expected
        for field, expected in {
            "replacement_submission_count_limit": 1,
            "replacement_submission_consumed": False,
            "kaggle_api_call_permitted_now": False,
            "kaggle_upload_permitted_now": False,
            "operator_directed_direct_upload_permitted_only_after_all_gates_and_binding": True,
            "queue_or_batch_submission_allowed": False,
            "automatic_retry_allowed": False,
            "automatic_copy_or_resubmission_allowed": False,
            "second_upload_allowed": False,
            "validation_failure_behavior": (
                "preserve_and_download_logs_do_not_retry_queue_copy_or_upload_"
                "without_new_owner_order"
            ),
            "training_authority": False,
            "gpu_authority": False,
            "service_authority": False,
            "bo1000_authority": False,
        }.items()
    ):
        raise R235SupportError("immutable binding authority changed")
    builder = binding.get("builder")
    if not isinstance(builder, Mapping) or any(builder.get(field) is not False for field in (
        "network_accessed", "kaggle_api_called", "kaggle_queue_used", "kaggle_upload_used",
        "gpu_used", "service_modified", "bo1000_modified",
    )):
        raise R235SupportError("immutable binding builder provenance changed")
    return {
        "path": str(binding_file),
        "sha256": sha256_file(binding_file),
        "payload": binding,
        **candidate_context,
    }


_GO_FIRST_PROBE = r'''
import builtins
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

def _network_disabled(*_args, **_kwargs):
    raise RuntimeError("network is disabled in the R235 go-first verifier")

# The verifier deliberately executes only a tiny, digest-bound turn-order
# branch.  It is not a runtime sandbox, but block the ordinary Python network
# and process-launch surfaces before archive code is imported.  A package that
# needs one of them fails its local gate rather than making an unexpected call.
for _name in (
    "socket", "create_connection", "getaddrinfo", "gethostbyname",
    "gethostbyname_ex", "gethostbyaddr", "getnameinfo", "getfqdn",
):
    if hasattr(socket, _name):
        setattr(socket, _name, _network_disabled)

for _name in (
    "system", "popen", "execl", "execle", "execlp", "execlpe", "execv",
    "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp",
    "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "posix_spawn",
    "posix_spawnp",
):
    if hasattr(os, _name):
        setattr(os, _name, _network_disabled)

_blocked_import_roots = {
    "asyncio", "ctypes", "ftplib", "http", "imaplib", "multiprocessing",
    "nntplib", "poplib", "requests", "smtplib", "socket", "socketserver",
    "ssl", "subprocess", "telnetlib", "urllib", "webbrowser", "xmlrpc",
}
_original_import = builtins.__import__
def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in _blocked_import_roots:
        raise RuntimeError("network/process imports are disabled in the R235 go-first verifier")
    return _original_import(name, globals, locals, fromlist, level)
builtins.__import__ = _guarded_import

stage = Path(sys.argv[1]).resolve()
entrypoint = stage / "r195_direct_main.py"
os.chdir(stage)
spec = importlib.util.spec_from_file_location("r235_go_first_probe", entrypoint)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load packaged direct entrypoint")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = {
    "integer_enum": ({"select": {"context": 41, "minCount": 1, "maxCount": 1,
        "option": [{"type": 1}, {"type": 2}]}}, [0]),
    "string_enum_reversed_options": ({"select": {"context": "IS_FIRST", "minCount": 1, "maxCount": 1,
        "option": [{"type": "No"}, {"type": "Yes"}]}}, [1]),
    "live_engine_prompt": ({"select": {"context": "IsFirst", "type": 9, "minCount": 1, "maxCount": 1,
        "option": [{"type": 2}, {"type": 1}]}}, [1]),
}
observed = {}
for name, (prompt, expected) in cases.items():
    action = module.agent(prompt)
    if action != expected:
        raise SystemExit("%s expected %r, got %r" % (name, expected, action))
    observed[name] = {"selected_action": action}
print(json.dumps({"verified_cases": observed}, sort_keys=True))
'''


def _probe_go_first(archive: Path) -> dict[str, Any]:
    """Run only the direct entrypoint's early IsFirst path in isolated Python."""

    members = _archive_members(archive)
    with tempfile.TemporaryDirectory(prefix="r235-go-first-") as temporary:
        stage = Path(temporary)
        # Stage every validated regular member so the exact direct entrypoint
        # may import its package-local dependencies.  The probe disables normal
        # Python sockets before that entrypoint is imported.
        for member_name in members:
            target = stage / member_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_archive_member_bytes(archive, member_name))
            os.chmod(target, 0o400)
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _GO_FIRST_PROBE, str(stage)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
        )
    if completed.returncode != 0:
        raise R235SupportError("go-first probe failed: " + (completed.stderr or completed.stdout).strip())
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise R235SupportError("go-first probe emitted invalid JSON") from exc
    cases = result.get("verified_cases") if isinstance(result, Mapping) else None
    if not isinstance(cases, Mapping) or set(cases) != GO_FIRST_CASES:
        raise R235SupportError("go-first probe did not verify every required case")
    return dict(cases)


def _go_first_gate_identity(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return the builder-compatible identity derivable before binding exists."""

    package = context["package"]
    contracts = context["contracts"]
    return {
        "candidate_archive_sha256": package["sha256"],
        "candidate_archive_size_bytes": package["size_bytes"],
        "member_manifest_sha256": package["manifest_sha256"],
        "entrypoint_sha256": package["members"]["main.py"]["sha256"],
        "r225_contract_sha256": contracts["r225"]["sha256"],
        "canonical_libcg_contract_sha256": contracts["r236"]["sha256"],
        "linux_x86_64_libcg_sha256": contracts["r236"]["linux_sha256"],
        "linux_x86_64_libcg_size_bytes": contracts["r236"]["linux_size_bytes"],
        "complete_ordered_action_cap": ACTION_CAP,
        "simulator_search_lane_count": LANE_COUNT,
        "phase1_submission_environment": dict(PHASE1_RESOURCES),
        "r240_hybrid_scheduler": dict(R240_NORMALIZED_SCHEDULER),
        "deterministic_continuation": dict(R235_GATE_DETERMINISTIC_CONTINUATION),
    }


def verify_go_first(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    r225_contract: Path,
    r236_contract: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish the pre-binding archive sidecar required by the guard/binder.

    The candidate is validated against the exact manifest and canonical r225 /
    r236 sources, but no immutable binding is required or consulted here.  The
    builder consumes this sealed sidecar to create that binding afterwards.
    """

    context = validate_r235_candidate(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        r225_contract=r225_contract,
        r236_contract=r236_contract,
    )
    package = context["package"]
    expected_output = Path(str(package["path"]) + ".go-first-verified.json").resolve()
    if Path(output_path).expanduser().resolve() != expected_output:
        raise R235SupportError("go-first receipt must be the exact archive sidecar used by the guard")
    cases = _probe_go_first(Path(package["path"]))
    receipt = {
        "schema": GO_FIRST_SCHEMA,
        "kind": "r235_digest_bound_go_first_verifier",
        # These fields make the exact same sidecar a pre-binding builder gate
        # receipt while preserving the guard's turn-order attestation schema.
        "status": "passed",
        "receipt_name": R235_BINDING_GATE_RECEIPT_NAMES["go_first"],
        "passed": True,
        "immutable": True,
        "write_once": True,
        "go_first_contract_passed": True,
        "forced_yes_action_legal": True,
        "file_sha256": package["sha256"],
        "file_bytes": package["size_bytes"],
        "turn_order_preference": "first_if_allowed",
        "go_first_if_offered": True,
        "go_second_if_offered": False,
        "verified_cases": sorted(GO_FIRST_CASES),
        "case_results": cases,
        "submission": {"competition": COMPETITION, "message": LABEL},
        **_go_first_gate_identity(context),
        "manifest": {
            "path": package["manifest_path"],
            "sha256": package["manifest_sha256"],
            "member": package["manifest_member"],
            "schema": R238_MANIFEST_SCHEMA,
            "lane_count": LANE_COUNT,
        },
        "typed_contracts": {
            "r225_sha256": context["contracts"]["r225"]["sha256"],
            "r236_sha256": context["contracts"]["r236"]["sha256"],
            "r225_owner_decision_revision": R225_OWNER_DECISION_REVISION,
        },
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    written = _write_immutable_json(expected_output, receipt)
    return {**receipt, "path": str(written), "sha256": sha256_file(written)}


def _validate_go_first_receipt(context: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    package = context["package"]
    expected_path = Path(str(package["path"]) + ".go-first-verified.json").resolve()
    receipt_file = _immutable(receipt_path, label="R235 go-first receipt")
    if receipt_file != expected_path:
        raise R235SupportError("R235 go-first receipt is not the archive guard sidecar")
    receipt = _read_json(receipt_file, label="R235 go-first receipt")
    contracts = context["contracts"]
    expected_gate_identity = _go_first_gate_identity(context)
    binding_gates = (context.get("payload") or {}).get("local_gate_receipts")
    expected_binding_digest = (
        binding_gates.get("go_first", {}).get("sha256")
        if isinstance(binding_gates, Mapping)
        else None
    )
    expected_actions = {
        "integer_enum": [0],
        "string_enum_reversed_options": [1],
        "live_engine_prompt": [1],
    }
    observed_cases = receipt.get("case_results")
    checks = {
        "field_set": set(receipt) == GO_FIRST_PREBINDING_RECEIPT_FIELDS,
        "schema": receipt.get("schema") == GO_FIRST_SCHEMA,
        "kind": receipt.get("kind") == "r235_digest_bound_go_first_verifier",
        "gate_receipt": receipt.get("status") == "passed"
        and receipt.get("receipt_name") == R235_BINDING_GATE_RECEIPT_NAMES["go_first"]
        and receipt.get("passed") is True
        and receipt.get("immutable") is True
        and receipt.get("write_once") is True
        and receipt.get("go_first_contract_passed") is True
        and receipt.get("forced_yes_action_legal") is True,
        "archive": receipt.get("file_sha256") == package["sha256"]
        and receipt.get("file_bytes") == package["size_bytes"]
        and receipt.get("candidate_archive_sha256") == package["sha256"]
        and receipt.get("candidate_archive_size_bytes") == package["size_bytes"],
        "first": receipt.get("turn_order_preference") == "first_if_allowed"
        and receipt.get("go_first_if_offered") is True
        and receipt.get("go_second_if_offered") is False,
        "cases": set(receipt.get("verified_cases") or []) == GO_FIRST_CASES
        and isinstance(observed_cases, Mapping)
        and set(observed_cases) == GO_FIRST_CASES
        and {
            name: (observed_cases.get(name) or {}).get("selected_action")
            for name in GO_FIRST_CASES
        }
        == expected_actions,
        "label": (receipt.get("submission") or {}).get("competition") == COMPETITION
        and (receipt.get("submission") or {}).get("message") == LABEL,
        "prebinding": "r235_binding" not in receipt,
        "binding_gate_digest": expected_binding_digest == sha256_file(receipt_file),
        "gate_identity": all(receipt.get(field) == expected for field, expected in expected_gate_identity.items()),
        "manifest": (receipt.get("manifest") or {}).get("sha256") == package["manifest_sha256"]
        and (receipt.get("manifest") or {}).get("path") == package["manifest_path"]
        and (receipt.get("manifest") or {}).get("member") == package["manifest_member"]
        and (receipt.get("manifest") or {}).get("schema") == R238_MANIFEST_SCHEMA
        and (receipt.get("manifest") or {}).get("lane_count") == LANE_COUNT,
        "contracts": (receipt.get("typed_contracts") or {}).get("r225_sha256") == contracts["r225"]["sha256"]
        and (receipt.get("typed_contracts") or {}).get("r236_sha256") == contracts["r236"]["sha256"]
        and (receipt.get("typed_contracts") or {}).get("r225_owner_decision_revision")
        == R225_OWNER_DECISION_REVISION,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise R235SupportError("R235 go-first receipt mismatch: " + ", ".join(failed))
    return {"path": str(receipt_file), "sha256": sha256_file(receipt_file), "payload": receipt}


def _validate_consumption(
    *, context: Mapping[str, Any], consumption_path: Path, require_unconsumed: bool
) -> Path:
    target = Path(consumption_path).expanduser()
    if target.is_symlink():
        raise R235SupportError("R235 authority-consumption target may not be a symlink")
    if not target.exists():
        if require_unconsumed:
            return target.resolve()
        raise R235SupportError("R235 authority-consumption receipt is absent")
    if require_unconsumed:
        raise R235SupportError("R235 direct-upload authority is already consumed")
    receipt_path = _immutable(target, label="R235 authority-consumption receipt")
    receipt = _read_json(receipt_path, label="R235 authority-consumption receipt")
    if (
        receipt.get("schema") != R235_CONSUMPTION_SCHEMA
        or receipt.get("authority_kind") != R235_AUTHORITY_KIND
        or receipt.get("binding_sha256") != context["sha256"]
        or receipt.get("consumed_before_upload") is not True
    ):
        raise R235SupportError("R235 authority-consumption receipt does not bind this authority")
    return receipt_path


def build_one_use_authorization(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    binding_path: Path,
    r225_contract: Path,
    r236_contract: Path,
    go_first_receipt_path: Path,
    consumption_path: Path,
    output_path: Path,
    nonce: str,
    expires_at_epoch: float,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Create one guard authorization only while the immutable binding is unused."""

    if not _NONCE.fullmatch(nonce):
        raise R235SupportError("R235 authorization nonce is malformed")
    now = float(dt.datetime.now(dt.timezone.utc).timestamp() if now_epoch is None else now_epoch)
    expiry = _finite(expires_at_epoch, label="R235 authorization expiry")
    if expiry <= now or expiry > now + 3600:
        raise R235SupportError("R235 authorization expiry must be within one hour")
    context = validate_r235_binding(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        binding_path=binding_path,
        r225_contract=r225_contract,
        r236_contract=r236_contract,
    )
    go_first = _validate_go_first_receipt(context, go_first_receipt_path)
    consumption = _validate_consumption(
        context=context, consumption_path=consumption_path, require_unconsumed=True
    )
    output = Path(output_path).expanduser()
    if output.resolve() == consumption.resolve():
        raise R235SupportError("authorization and authority-consumption receipts must differ")
    package = context["package"]
    contracts = context["contracts"]
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "approval_source": "GOAL.md#/revision-235-direct-replacement",
        "authority_kind": R235_AUTHORITY_KIND,
        "remaining_uses": 1,
        "nonce": nonce,
        "expires_at_epoch": expiry,
        "competition": COMPETITION,
        "file_sha256": package["sha256"],
        "message": LABEL,
        "turn_order_preference": "first_if_allowed",
        "r235_binding_path": context["path"],
        "r235_binding_sha256": context["sha256"],
        "r235_manifest_path": package["manifest_path"],
        "r235_manifest_sha256": package["manifest_sha256"],
        "r235_manifest_member": package["manifest_member"],
        "r235_go_first_receipt_path": go_first["path"],
        "r235_go_first_receipt_sha256": go_first["sha256"],
        "r225_contract_path": contracts["r225"]["path"],
        "r225_contract_sha256": contracts["r225"]["sha256"],
        "r236_contract_path": contracts["r236"]["path"],
        "r236_contract_sha256": contracts["r236"]["sha256"],
        "r225_owner_decision_revision": R225_OWNER_DECISION_REVISION,
        "r244_handle_scoped_search_id_revision": R244_HANDLE_SCOPED_SEARCH_ID_REVISION,
        "r246_proven_deterministic_terminal_win_this_turn_revision": (
            R246_PROVEN_TERMINAL_WIN_REVISION
        ),
        "r235_lane_count": LANE_COUNT,
        "r235_authority_consumption_path": str(consumption),
        "queue_allowed": False,
        "automatic_retry_allowed": False,
        "copy_or_resubmission_allowed": False,
        "second_upload_allowed": False,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    written = _write_immutable_json(output, authorization)
    return {**authorization, "path": str(written), "sha256": sha256_file(written)}


def _r235_marker(authorization: Mapping[str, Any]) -> bool:
    return authorization.get("authority_kind") == R235_AUTHORITY_KIND or any(
        str(key).startswith(
            (
                "r225_",
                "r235_",
                "r236_",
                "r238_",
                "r240_",
                "r242_",
                "r244_",
                "r246_",
            )
        )
        for key in authorization
    )


def validate_r235_authorization(
    authorization: Mapping[str, Any], *, file_path: Path, file_sha256: str, competition: str, message: str
) -> tuple[bool, str, dict[str, Any]]:
    """Validate R235-only fields for the generic Kaggle CLI guard."""

    if not _r235_marker(authorization):
        return True, "not_r235", {}
    try:
        if authorization.get("authority_kind") != R235_AUTHORITY_KIND:
            raise R235SupportError("R235 authority kind is missing or mismatched")
        if competition != COMPETITION or message != LABEL:
            raise R235SupportError("R235 competition or exact label mismatch")
        paths = {
            key: _required_path(authorization.get(key), label=key)
            for key in (
                "r235_manifest_path", "r235_binding_path", "r225_contract_path",
                "r236_contract_path", "r235_go_first_receipt_path",
                "r235_authority_consumption_path",
            )
        }
        context = validate_r235_binding(
            archive_path=file_path,
            manifest_path=paths["r235_manifest_path"],
            manifest_member=str(authorization.get("r235_manifest_member") or ""),
            binding_path=paths["r235_binding_path"],
            r225_contract=paths["r225_contract_path"],
            r236_contract=paths["r236_contract_path"],
        )
        go_first = _validate_go_first_receipt(context, paths["r235_go_first_receipt_path"])
        consumption = _validate_consumption(
            context=context, consumption_path=paths["r235_authority_consumption_path"], require_unconsumed=True
        )
        package = context["package"]
        contracts = context["contracts"]
        checks = {
            "archive": file_sha256 == package["sha256"],
            "binding": authorization.get("r235_binding_sha256") == context["sha256"],
            "manifest": authorization.get("r235_manifest_sha256") == package["manifest_sha256"],
            "go_first": authorization.get("r235_go_first_receipt_sha256") == go_first["sha256"],
            "r225": authorization.get("r225_contract_sha256") == contracts["r225"]["sha256"],
            "r236": authorization.get("r236_contract_sha256") == contracts["r236"]["sha256"],
            "r240": authorization.get("r225_owner_decision_revision")
            == R225_OWNER_DECISION_REVISION,
            "r244": authorization.get("r244_handle_scoped_search_id_revision")
            == R244_HANDLE_SCOPED_SEARCH_ID_REVISION,
            "r246": authorization.get(
                "r246_proven_deterministic_terminal_win_this_turn_revision"
            )
            == R246_PROVEN_TERMINAL_WIN_REVISION,
            "two_lane": authorization.get("r235_lane_count") == LANE_COUNT,
            "queue": authorization.get("queue_allowed") is False,
            "retry": authorization.get("automatic_retry_allowed") is False,
            "copy": authorization.get("copy_or_resubmission_allowed") is False,
            "second_upload": authorization.get("second_upload_allowed") is False,
        }
        failed = sorted(key for key, value in checks.items() if not value)
        if failed:
            return False, "R235 authorization mismatch: " + ", ".join(failed), {"checks": checks}
        return True, "R235 binding verified", {
            "checks": checks,
            "binding_path": context["path"],
            "binding_sha256": context["sha256"],
            "consumption_path": str(consumption),
        }
    except (OSError, R235SupportError, ValueError) as exc:
        return False, f"R235 binding validation failed: {exc}", {}


def consume_r235_authority(authorization: Mapping[str, Any], *, identity: Mapping[str, Any]) -> Path | None:
    """Atomically consume R235's separate authority receipt before upload starts."""

    if not _r235_marker(authorization):
        return None
    file_path = _required_path(identity.get("file"), label="guard file identity")
    valid, reason, details = validate_r235_authorization(
        authorization,
        file_path=file_path,
        file_sha256=str(identity.get("file_sha256") or ""),
        competition=str(identity.get("competition") or ""),
        message=str(identity.get("message") or ""),
    )
    if not valid:
        raise R235SupportError("R235 authority cannot be consumed: " + reason)
    target = _required_path(authorization.get("r235_authority_consumption_path"), label="consumption path")
    binding = _required_path(authorization.get("r235_binding_path"), label="binding path")
    payload = {
        "schema": R235_CONSUMPTION_SCHEMA,
        "authority_kind": R235_AUTHORITY_KIND,
        "binding_path": str(_regular_file(binding, label="R235 immutable binding")),
        "binding_sha256": str(details["binding_sha256"]),
        "authorization_nonce": str(authorization.get("nonce") or ""),
        "file_sha256": str(identity.get("file_sha256") or ""),
        "competition": str(identity.get("competition") or ""),
        "message": str(identity.get("message") or ""),
        "consumed_before_upload": True,
        "consumed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return _write_immutable_json(target, payload)


def _decimal_submission_id(value: object, *, label: str) -> tuple[int, str]:
    if isinstance(value, bool):
        raise R235SupportError(f"{label} is not an exact decimal submission ID")
    text = str(value) if isinstance(value, (str, int)) else ""
    if not _SUBMISSION_ID.fullmatch(text):
        raise R235SupportError(f"{label} is not an exact decimal submission ID")
    return int(text), text


def _walk_ids(value: object) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    if isinstance(value, Mapping):
        for key in ("submission_id", "submissionId", "submission_id_text", "submissionIdText"):
            if key in value:
                try:
                    result.add(_decimal_submission_id(value[key], label=key))
                except R235SupportError:
                    pass
        for child in value.values():
            result.update(_walk_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_walk_ids(child))
    return result


def _guard_output_ids(path: Path) -> tuple[set[tuple[int, str]], dict[str, str]]:
    source = _regular_file(path, label="guard output")
    text = source.read_text(encoding="utf-8")
    result = {_decimal_submission_id(match.group(1), label="guard output") for match in _OUTPUT_ID.finditer(text)}
    for line in text.splitlines():
        try:
            result.update(_walk_ids(json.loads(line)))
        except json.JSONDecodeError:
            pass
    return result, {"path": str(source), "sha256": sha256_file(source)}


def _api_snapshot_ids(path: Path) -> tuple[set[tuple[int, str]], dict[str, Any]]:
    source = _regular_file(path, label="Kaggle API snapshot")
    payload = _read_json(source, label="Kaggle API snapshot")
    if payload.get("competition") not in {None, COMPETITION}:
        raise R235SupportError("Kaggle API snapshot competition mismatch")
    rows: list[Mapping[str, Any]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            label = value.get("description", value.get("label", value.get("message")))
            if label == LABEL and value.get("competition") in {None, COMPETITION}:
                rows.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    ids = {
        _decimal_submission_id(
            row.get("ref", row.get("submission_id", row.get("submissionId", row.get("id")))),
            label="Kaggle API submission ID",
        )
        for row in rows
    }
    if len(ids) != 1:
        raise R235SupportError("Kaggle API snapshot does not resolve one exact R235-label ID")
    return ids, {"path": str(source), "sha256": sha256_file(source), "matched_rows": len(rows)}


def capture_submission_id(
    *,
    archive_path: Path,
    manifest_path: Path,
    manifest_member: str,
    binding_path: Path,
    r225_contract: Path,
    r236_contract: Path,
    authorization_consumed_path: Path,
    attempt_path: Path,
    output_path: Path,
    guard_output_path: Path | None = None,
    api_snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Seal the one exact R235 ID from saved guard output and/or API JSON."""

    if guard_output_path is None and api_snapshot_path is None:
        raise R235SupportError("provide saved guard output and/or API snapshot")
    consumed_file = _regular_file(authorization_consumed_path, label="guard authorization receipt")
    consumed = _read_json(consumed_file, label="guard authorization receipt")
    consumption_path = _required_path(
        consumed.get("r235_authority_consumption_path"), label="consumed R235 authority path"
    )
    context = validate_r235_binding(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest_member=manifest_member,
        binding_path=binding_path,
        r225_contract=r225_contract,
        r236_contract=r236_contract,
    )
    consumption = _validate_consumption(
        context=context, consumption_path=consumption_path, require_unconsumed=False
    )
    consumption_payload = _read_json(consumption, label="R235 authority-consumption receipt")
    attempt_file = _regular_file(attempt_path, label="guard attempt receipt")
    attempt = _read_json(attempt_file, label="guard attempt receipt")
    identity = attempt.get("identity")
    attempt_consumed = _required_path(
        attempt.get("authorization_consumed"), label="guard attempt authorization receipt"
    )
    if _regular_file(attempt_consumed, label="guard attempt authorization receipt") != consumed_file:
        raise R235SupportError("guard attempt does not point to the supplied authorization receipt")
    if (
        consumed.get("schema") != AUTH_SCHEMA
        or consumed.get("authority_kind") != R235_AUTHORITY_KIND
        or consumed.get("remaining_uses") != 0
        or consumed.get("consumed_before_upload") is not True
        or consumed.get("r235_binding_sha256") != context["sha256"]
        or attempt.get("schema") != "poke_bot.kaggle_submission_attempt/v1"
        or attempt.get("returncode") != 0
        or attempt.get("nonce") != consumed.get("nonce")
        or not isinstance(identity, Mapping)
        or identity.get("competition") != COMPETITION
        or identity.get("message") != LABEL
        or identity.get("file_sha256") != context["package"]["sha256"]
        or identity.get("r235_authority_consumption") != str(consumption)
        or consumption_payload.get("authorization_nonce") != consumed.get("nonce")
        or consumption_payload.get("file_sha256") != context["package"]["sha256"]
        or consumption_payload.get("competition") != COMPETITION
        or consumption_payload.get("message") != LABEL
    ):
        raise R235SupportError("guard receipts do not match the consumed R235 authority")
    sources: dict[str, Any] = {
        "authorization_consumed": {"path": str(consumed_file), "sha256": sha256_file(consumed_file)},
        "guard_attempt": {"path": str(attempt_file), "sha256": sha256_file(attempt_file)},
    }
    candidates: list[set[tuple[int, str]]] = []
    if guard_output_path is not None:
        ids, source = _guard_output_ids(guard_output_path)
        if len(ids) != 1:
            raise R235SupportError("guard output does not contain exactly one submission ID")
        candidates.append(ids)
        sources["guard_output"] = source
    if api_snapshot_path is not None:
        ids, source = _api_snapshot_ids(api_snapshot_path)
        candidates.append(ids)
        sources["api_snapshot"] = source
    resolved = set.intersection(*candidates)
    if len(resolved) != 1:
        raise R235SupportError("guard output/API do not resolve one identical submission ID")
    submission_id, submission_id_text = next(iter(resolved))
    receipt = {
        "schema": R235_ID_RECEIPT_SCHEMA,
        "status": "resolved",
        "submission": {"id": submission_id, "id_text": submission_id_text, "competition": COMPETITION, "message": LABEL},
        "r235_binding": {"path": context["path"], "sha256": context["sha256"]},
        "authority_consumption": {"path": str(consumption), "sha256": sha256_file(consumption)},
        "archive": {
            "path": context["package"]["path"],
            "sha256": context["package"]["sha256"],
            "manifest_path": context["package"]["manifest_path"],
            "manifest_sha256": context["package"]["manifest_sha256"],
            "manifest_member": context["package"]["manifest_member"],
            "lane_count": LANE_COUNT,
        },
        "sources": sources,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    written = _write_immutable_json(output_path, receipt)
    return {**receipt, "path": str(written), "sha256": sha256_file(written)}


def validate_submission_id_receipt(context: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    """Validate an ID receipt before post-upload evidence collection."""

    source = _immutable(receipt_path, label="R235 submission-ID receipt")
    receipt = _read_json(source, label="R235 submission-ID receipt")
    submission = receipt.get("submission")
    archive = receipt.get("archive")
    consumed = receipt.get("authority_consumption")
    if not isinstance(submission, Mapping) or not isinstance(archive, Mapping) or not isinstance(consumed, Mapping):
        raise R235SupportError("R235 submission-ID receipt is malformed")
    submission_id, submission_id_text = _decimal_submission_id(
        submission.get("id_text", submission.get("id")), label="submission-ID receipt"
    )
    if submission.get("id") != submission_id or submission.get("id_text") != submission_id_text:
        raise R235SupportError("R235 submission-ID receipt does not preserve one exact decimal ID")
    if (
        receipt.get("schema") != R235_ID_RECEIPT_SCHEMA
        or receipt.get("status") != "resolved"
        or submission.get("competition") != COMPETITION
        or submission.get("message") != LABEL
        or (receipt.get("r235_binding") or {}).get("sha256") != context["sha256"]
        or archive.get("sha256") != context["package"]["sha256"]
        or archive.get("manifest_sha256") != context["package"]["manifest_sha256"]
        or archive.get("lane_count") != LANE_COUNT
    ):
        raise R235SupportError("R235 submission-ID receipt does not bind the exact package")
    consumption_path = _required_path(consumed.get("path"), label="submission-ID consumption path")
    validated = _validate_consumption(context=context, consumption_path=consumption_path, require_unconsumed=False)
    if consumed.get("sha256") != sha256_file(validated):
        raise R235SupportError("R235 submission-ID receipt consumption digest drifted")
    return {"path": str(source), "sha256": sha256_file(source), "payload": receipt, "consumption_path": str(validated)}


def _args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def context_arguments(command: argparse.ArgumentParser, *, include_binding: bool = True) -> None:
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-member", required=True)
        if include_binding:
            command.add_argument("--binding", type=Path, required=True)
        command.add_argument("--r225-contract", type=Path, default=CANONICAL_R225_PATH)
        command.add_argument("--r236-contract", type=Path, default=CANONICAL_R236_PATH)

    verify = commands.add_parser("verify-go-first")
    # This is deliberately pre-binding: its immutable sidecar is one of the
    # inputs used to create the later binding.
    context_arguments(verify, include_binding=False)
    verify.add_argument("--output", type=Path, required=True)

    authorize = commands.add_parser("build-authorization")
    context_arguments(authorize)
    authorize.add_argument("--go-first-receipt", type=Path, required=True)
    authorize.add_argument("--consumption-receipt", type=Path, required=True)
    authorize.add_argument("--nonce", required=True)
    authorize.add_argument("--expires-at-epoch", type=float, required=True)
    authorize.add_argument("--output", type=Path, required=True)

    capture = commands.add_parser("capture-submission-id")
    context_arguments(capture)
    capture.add_argument("--authorization-consumed", type=Path, required=True)
    capture.add_argument("--attempt", type=Path, required=True)
    capture.add_argument("--guard-output", type=Path)
    capture.add_argument("--api-snapshot", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _args().parse_args()
    candidate_shared = {
        "archive_path": args.archive,
        "manifest_path": args.manifest,
        "manifest_member": args.manifest_member,
        "r225_contract": args.r225_contract,
        "r236_contract": args.r236_contract,
    }
    try:
        if args.command == "verify-go-first":
            result = verify_go_first(**candidate_shared, output_path=args.output)
        elif args.command == "build-authorization":
            shared = {**candidate_shared, "binding_path": args.binding}
            result = build_one_use_authorization(
                **shared,
                go_first_receipt_path=args.go_first_receipt,
                consumption_path=args.consumption_receipt,
                output_path=args.output,
                nonce=args.nonce,
                expires_at_epoch=args.expires_at_epoch,
            )
        else:
            shared = {**candidate_shared, "binding_path": args.binding}
            result = capture_submission_id(
                **shared,
                authorization_consumed_path=args.authorization_consumed,
                attempt_path=args.attempt,
                guard_output_path=args.guard_output,
                api_snapshot_path=args.api_snapshot,
                output_path=args.output,
            )
    except (OSError, R235SupportError, ValueError) as exc:
        print(f"R235 tooling BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result.get(key) for key in ("path", "sha256", "status", "submission")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
