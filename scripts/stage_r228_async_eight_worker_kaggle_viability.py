#!/usr/bin/env python3
"""Build, but never submit, the r238 two-lane bounded-MCTS diagnostic.

The input must be the immutable r195 NO-RTP archive.  This script relocates
its original entrypoint to ``r195_direct_main.py`` and overlays only the
bounded-MCTS entrypoint and its minimal shared-tree runtime sources.  It does
not start a game, import a Kaggle client, make a network request, or submit.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
R225_TYPED_CONTRACT_PATH = (
    "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
)
R225_TYPED_CONTRACT_SCHEMA = (
    "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
)
R225_TYPED_CONTRACT_OWNER_DECISION_REVISION = 246
R225_TYPED_CONTRACT_SHA256 = (
    "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
)
R195_BUNDLE_SHA256 = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
R195_MODEL_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
R195_SEARCH_CONFIG_SHA256 = "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
STOCK_LIBCG_SHA256 = "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
STOCK_LIBCG_BYTES = 1_342_400
CANONICAL_LIBCG_SCHEMA = "poke_bot.canonical_libcg_r236/v1"
CANONICAL_LIBCG_OWNER_DECISION_REVISION = 236
CANONICAL_LIBCG_TYPED_SOURCE = "state/canonical-libcg-r236.json"
CANONICAL_LIBCG_WHEEL_FILENAME = "kaggle_environments-1.32.6-py3-none-any.whl"
CANONICAL_LIBCG_WHEEL_SHA256 = (
    "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
)
CANONICAL_LIBCG_WHEEL_BYTES = 60_677_343
CANONICAL_LIBCG_UPSTREAM_PROVENANCE = {
    "repository": "https://github.com/Kaggle/kaggle-environments",
    "package_index": "https://pypi.org/project/kaggle-environments/1.32.6/",
    "package_version": "1.32.6",
    "package_published_at_utc": "2026-08-07T17:20:19.634609Z",
    "native_library_update_commit": "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303",
    "native_library_update_commit_date_utc": "2026-07-23T16:03:22Z",
    "native_library_update_commit_message": (
        "Cabt update library (#1356): Fixed a crash caused by a specific combination "
        "of cards."
    ),
    "online_master_checked_at_utc": "2026-08-10T23:26:00Z",
    "online_master_commit": "bded87b0d7879078c726a93a4884d044f79c4eed",
    "online_master_version": "1.32.6",
    "online_master_linux_x86_64_sha256_matches_wheel": True,
}
CANONICAL_LIBCG_MEMBERS = (
    {
        "platform": "linux_x86_64",
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
        "format": "ELF 64-bit LSB shared object x86-64",
    },
    {
        "platform": "linux_aarch64",
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
        "format": "ELF 64-bit LSB shared object ARM aarch64",
    },
    {
        "platform": "macos_arm64",
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "package_relative_path": "cg/libcg.dylib",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
        "format": "Mach-O 64-bit dynamically linked shared library arm64",
    },
    {
        "platform": "windows_x86_64",
        "wheel_member": "kaggle_environments/envs/cabt/cg/cg.dll",
        "package_relative_path": "cg/cg.dll",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
        "format": "PE32+ x86-64 DLL",
    },
)
REQUIRED_STOCK_LIBCG_EXPORTS = (
    "AgentStart",
    "BattleStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
)
R225_CANONICAL_LIBCG_PACKAGE_PREFLIGHT = {
    "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
    "all_four_canonical_native_members_checksum_and_size_verified": True,
    "old_or_mixed_native_members_rejected": True,
    "required_native_exports": list(REQUIRED_STOCK_LIBCG_EXPORTS),
}
SIMULATOR_SEARCH_LANE_COUNT = 2
REQUIRED_SEARCH_BEGIN_COUNT = SIMULATOR_SEARCH_LANE_COUNT
REQUIRED_SEARCH_END_COUNT = SIMULATOR_SEARCH_LANE_COUNT
REQUIRED_SEARCH_RELEASE_COUNT = SIMULATOR_SEARCH_LANE_COUNT
PHASE1_KAGGLE_RESOURCE_BOUNDS = {
    "vcpus": 2,
    "ram_gib": 12.2,
    "hdd_gib": 11.8,
    "archive_mib": 197.7,
    # The reported Phase-1 envelope says nothing about CUDA visibility.  The
    # packaged runtime must observe that fact in its own process immediately
    # before search rather than treating this resource report as a GPU claim.
    "gpu_environment_inferred_from_resource_envelope": False,
    "runtime_cuda_observation_required_before_search": True,
}
PHASE1_ARCHIVE_MAX_BYTES = 207_303_475
REQUIRED_LABEL = "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"
DECISION_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_DECISION"
FULL_GAMEPLAY_SUCCESS_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_FULL_GAMEPLAY_SUCCESS"
HARD_FAILURE_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_HARD_FAILURE"
BROKER_MODULE = "poke_bot.r228_kaggle_broker"
BROKER_SCHEMA = "poke_bot.r228_kaggle_subprocess_broker/v1"
COMPLETE_ACTION_CAP = 65_536
BROKER_ACTION_TIMEOUT_SECONDS = 4.0
BROKER_SEARCH_SECONDS = 2.0
BROKER_STARTUP_TIMEOUT_SECONDS = 30.0
BROKER_REAP_GRACE_SECONDS = 0.25
DEGRADED_FALLBACK_MARKER = "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED"
SUBPROCESS_CONTAINMENT_IDENTITY = (
    "exact_owned_popen_child_pid_bounded_reap_no_process_group_signal/v1"
)
R240_HIGH_CONFIDENCE_DIRECT_MODE = "high_confidence_frozen_direct"
R240_HIGH_CONFIDENCE_DIRECT_THRESHOLD = 0.80
R242_HIGH_CONFIDENCE_THRESHOLD_OWNER_REVISION = 242
R240_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R240_STABLE_ROOT_LEADER_OBSERVATIONS = 3
R240_MAXIMUM_BACKUPS_PER_DECISION = 32
R240_MAX_PRINCIPAL_VARIATION_DEPTH = 8
R240_REQUIRED_REGRESSION_RECEIPTS = (
    "high_confidence_direct_and_adaptive_bounded_mcts_regression_receipt",
    "deterministic_continuation_regression_receipt",
)
R244_HANDLE_SCOPED_SEARCH_IDENTITY_REGRESSION_RECEIPT = (
    "official_libcg_handle_scoped_search_id_identity_regression_receipt"
)
PROVEN_TERMINAL_WIN_REVISION = 246
PROVEN_TERMINAL_WIN_STOP_REASON = "proven_deterministic_terminal_win_this_turn"
PROVEN_TERMINAL_WIN_PROOF_KIND = (
    "exact_deterministic_simulator_terminal_win_this_turn"
)
PROVEN_TERMINAL_WIN_REGRESSION_RECEIPT = (
    "proven_deterministic_terminal_win_this_turn_regression_receipt"
)
PROVEN_TERMINAL_WIN_PROOF_FIELDS = (
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
)
CUDA_RUNTIME_OBSERVATION_SCHEMA = "poke_bot.r238_cuda_runtime_observation/v1"
CUDA_RUNTIME_OBSERVATION_PHASE = "before_search"
CUDA_RUNTIME_OBSERVATION_MARKER = "cuda_runtime_before_search"

# The failed one-shot is immutable diagnostic evidence, not a new submission
# target.  Keep only facts established by the recorded Kaggle episode.
FAILED_KAGGLE_VALIDATION_EVIDENCE = {
    "submission_id": 55_416_396,
    "episode_id": 91_766_923,
    "submission_message": "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY",
    "submission_status": "SubmissionStatus.ERROR",
    "episode_terminal_status": "TIMEOUT",
    "final_root_ordered_legal_action_count": 2,
    "final_unreturned_callback_elapsed_seconds": 438.994125,
    "failed_submission_must_be_preserved": True,
}

ARCHIVE_FILENAME = "r238-two-lane-bounded-mcts-diagnostic.tar.gz"
RECEIPT_FILENAME = "r238-two-lane-bounded-mcts-diagnostic.receipt.json"
MANIFEST_FILENAME = "r238_two_lane_bounded_mcts_manifest.json"

SOURCE_MEMBERS = {
    "main.py": "submission/r228_async_eight_worker_main.py",
    "poke_bot/r228_kaggle_broker.py": "poke_bot/r228_kaggle_broker.py",
    "poke_bot/r228_kaggle_async_runtime.py": "poke_bot/r228_kaggle_async_runtime.py",
    "poke_bot/r228_async_shared_tree_queue.py": "poke_bot/r228_async_shared_tree_queue.py",
    "poke_bot/r225_stock_native_lane.py": "poke_bot/r225_stock_native_lane.py",
    R225_TYPED_CONTRACT_PATH: R225_TYPED_CONTRACT_PATH,
}


class R228StageError(RuntimeError):
    """The proposed package did not preserve its frozen r195 inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def phase1_kaggle_resource_bounds() -> dict[str, Any]:
    """Return Phase-1 limits without inferring a CUDA environment.

    A fixed ``gpu_available`` value was a stale inference from an incomplete
    resource report.  It is deliberately rejected rather than copied into a
    package manifest or receipt: the contained runtime records its own CUDA
    observation before it can begin a search.
    """

    expected = {
        "vcpus": 2,
        "ram_gib": 12.2,
        "hdd_gib": 11.8,
        "archive_mib": 197.7,
        "gpu_environment_inferred_from_resource_envelope": False,
        "runtime_cuda_observation_required_before_search": True,
    }
    if "gpu_available" in PHASE1_KAGGLE_RESOURCE_BOUNDS:
        raise R228StageError(
            "stale gpu_available resource claim is forbidden for r238"
        )
    if PHASE1_KAGGLE_RESOURCE_BOUNDS != expected:
        raise R228StageError("r238 Phase-1 resource envelope is malformed")
    return {
        **PHASE1_KAGGLE_RESOURCE_BOUNDS,
        "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
    }


def r246_proven_deterministic_terminal_win_contract() -> dict[str, Any]:
    """Return the exceptional exact-simulator terminal-win authority contract.

    This deliberately grants no authority to evaluator confidence, unresolved
    randomness, a chance boundary, or an opponent boundary.  It is an
    opportunistic proof returned by the ordinary two-lane search, not an
    exhaustive root-action scan and not a change to the r242 direct shortcut.
    """

    return {
        "scope": "ambiguous_two_lane_mcts_for_new_r235_replacement_package_only",
        "owner_decision_revision": PROVEN_TERMINAL_WIN_REVISION,
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
        "proof_kind_required_literal": PROVEN_TERMINAL_WIN_PROOF_KIND,
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
        "stop_reason": PROVEN_TERMINAL_WIN_STOP_REASON,
        "required_queue_run_decision_inputs": [
            "root_observation_fingerprint",
            "root_legal_order_fingerprint",
            "root_actor_seat",
        ],
        "required_receipt_fields": list(PROVEN_TERMINAL_WIN_PROOF_FIELDS),
    }


def r240_hybrid_scheduler_contract() -> dict[str, Any]:
    """Return the exact R242 direct-or-bounded-MCTS scheduling contract."""

    return {
        "scope": "new_r235_replacement_package_only",
        "high_confidence_frozen_direct_threshold_owner_revision": (
            R242_HIGH_CONFIDENCE_THRESHOLD_OWNER_REVISION
        ),
        "selected_factorized_stage_probability_threshold": (
            R240_HIGH_CONFIDENCE_DIRECT_THRESHOLD
        ),
        "threshold_comparison": (
            "every selected factorized-stage probability is finite and "
            "greater_than_or_equal_to_0.80"
        ),
        "all_selected_factorized_stages_must_meet_threshold": True,
        "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
        "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
        "high_confidence_frozen_direct_mode": R240_HIGH_CONFIDENCE_DIRECT_MODE,
        "high_confidence_requires_precomputed_complete_legal_frozen_r195_direct_action": True,
        "high_confidence_requires_precomputed_direct_action_match_the_complete_ordered_root_legal_set_and_legal_fingerprint": True,
        "high_confidence_mcts_child_started_for_this_decision": False,
        "high_confidence_mcts_select_search_model_or_simulator_call_allowed": False,
        "high_confidence_existing_child_history_only_note_direct_action_ipc_allowed_and_required_when_child_exists": True,
        "high_confidence_existing_child_history_only_note_direct_action_ipc_max_count": 1,
        "high_confidence_existing_child_history_only_note_direct_action_ipc_count_range": [
            0,
            1,
        ],
        "high_confidence_history_only_note_direct_action_ipc_must_not_invoke_mcts_select_search_model_or_simulator": True,
        "high_confidence_direct_is_a_permitted_new_mcts_search_bypass_for_a_branching_prompt": True,
        "high_confidence_journaling_required": True,
        "high_confidence_degraded": False,
        "high_confidence_receipt_required_values": {
            "selected_factorized_stage_probability_threshold": (
                R240_HIGH_CONFIDENCE_DIRECT_THRESHOLD
            ),
            "all_selected_factorized_stages_meet_threshold": True,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "history_only_existing_child_journal_count_range": [0, 1],
            "degraded": False,
        },
        "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
        "two_lane_mcts_topology_backup_and_stop_contract_applies_only_when_confidence_routes_to_ambiguous_mcts": True,
        "ambiguous_mcts_exact_simulator_search_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        "child_search_hard_seconds": BROKER_SEARCH_SECONDS,
        "parent_action_hard_seconds": BROKER_ACTION_TIMEOUT_SECONDS,
        "adaptive_early_stop_min_completed_backups": (
            R240_MINIMUM_BACKUPS_BEFORE_STABILITY
        ),
        "adaptive_early_stop_stable_deterministic_root_leader_observations": (
            R240_STABLE_ROOT_LEADER_OBSERVATIONS
        ),
        "adaptive_early_stop_both_lanes_progressed_required": True,
        "hard_completed_backup_stop": R240_MAXIMUM_BACKUPS_PER_DECISION,
        "mcts_simulated_rollout_expansion_stops_at_terminal_chance_boundary_or_actor_change_away_from_root_seat": True,
        "root_actor_change_away_from_our_seat_leaf_is_value_evaluated_without_expanded_legal_actions_or_children": True,
        "mcts_opponent_action_selection_or_planning_allowed": False,
        "r246_proven_deterministic_terminal_win_this_turn": (
            r246_proven_deterministic_terminal_win_contract()
        ),
        "stop_reason_fields": [
            R240_HIGH_CONFIDENCE_DIRECT_MODE,
            "deterministic_continuation_plan",
            PROVEN_TERMINAL_WIN_STOP_REASON,
            "adaptive_early_stop",
            "hard_completed_backup_stop",
            "child_search_hard_deadline",
            "parent_action_hard_deadline",
            "zero_backup_precomputed_direct_fallback",
            "contained_child_fault",
        ],
        "zero_completed_backups_returns_only_the_precomputed_legal_direct_action_under_existing_clean_deadline_or_containment_rules": True,
        "partial_lane_serial_or_unbounded_search_authority_allowed": False,
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
        "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
        "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
        "boundary_leaf_receipt_required_fields": [
            "actor_change_boundary_leaf_count",
            "chance_boundary_leaf_count",
            "boundary_leaf_count",
        ],
        "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
    }


def deterministic_continuation_contract() -> dict[str, Any]:
    """Return the receipt-bound R242 principal-variation reuse constraints."""

    return {
        "scope": "optional_receipt_carried_plan_for_new_r235_replacement_package_only",
        "maximum_depth": R240_MAX_PRINCIPAL_VARIATION_DEPTH,
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


def r244_handle_scoped_search_identity_contract() -> dict[str, Any]:
    """Return the public r244 two-lane SearchId identity projection."""

    return {
        "simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        "search_id_namespace": "agent_start_handle_local",
        "per_lane_handle_identities_required": True,
        "per_lane_search_id_chains_required": True,
        "composite_state_fields": [
            "lane_id",
            "handle_identity",
            "first_search_id",
        ],
        "distinct_composite_count_required": SIMULATOR_SEARCH_LANE_COUNT,
        "globally_distinct_raw_first_search_ids_required": False,
        "duplicate_raw_first_search_ids_allowed_when_handles_differ": True,
        "gate_receipt": R244_HANDLE_SCOPED_SEARCH_IDENTITY_REGRESSION_RECEIPT,
    }


def _r244_search_identity_matches_canonical(value: object) -> bool:
    """Check the canonical r244 handle-scoped SearchId projection."""

    if not isinstance(value, dict):
        return False
    expected = r244_handle_scoped_search_identity_contract()
    counts = value.get("normal_mcts_decision_exact_counts")
    vectors = value.get("normal_mcts_decision_per_lane_vectors_exact_length")
    required_fields = value.get("normal_mcts_decision_required_fields")
    identity = value.get("search_id_identity_contract")
    if not all(
        isinstance(item, dict)
        for item in (counts, vectors, identity)
    ) or not isinstance(required_fields, list):
        return False
    if counts.get("distinct_handle_identity_count") != expected[
        "simulator_lane_count"
    ] or counts.get(
        "distinct_handle_scoped_first_search_id_composite_state_count"
    ) != expected["distinct_composite_count_required"]:
        return False
    if not {
        "per_lane_handle_identities",
        "per_lane_search_id_chains",
        "per_lane_first_search_ids",
        "handle_scoped_first_search_id_composite_states",
    } <= set(required_fields):
        return False
    if any(
        vectors.get(field) != expected["simulator_lane_count"]
        for field in (
            "per_lane_handle_identities",
            "per_lane_search_id_chains",
            "per_lane_first_search_ids",
            "handle_scoped_first_search_id_composite_states",
        )
    ):
        return False
    return (
        identity.get("numeric_namespace") == "per_distinct_agent_start_handle"
        and identity.get("globally_distinct_raw_search_id_integers_required")
        == expected["globally_distinct_raw_first_search_ids_required"]
        and identity.get("first_raw_search_id_may_be_zero_on_each_distinct_handle")
        is True
        and identity.get("required_distinct_handle_identity_first_search_id_composite_state_count")
        == expected["distinct_composite_count_required"]
        and identity.get("public_composite_state_array_field")
        == "handle_scoped_first_search_id_composite_states"
        and identity.get("public_composite_state_entry_exact_keys_in_order")
        == expected["composite_state_fields"]
        and identity.get("public_composite_state_entry_lane_id_values")
        == list(range(expected["simulator_lane_count"]))
    )


def r225_typed_contract_identity(contract_path: Path) -> dict[str, str]:
    """Read the staged canonical R225 contract and bind its immutable bytes."""

    if not contract_path.is_file() or contract_path.is_symlink():
        raise R228StageError("canonical r225 typed contract is missing or unsafe")
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R228StageError("canonical r225 typed contract is unreadable") from exc
    if not isinstance(payload, dict):
        raise R228StageError("canonical r225 typed contract is not an object")
    if payload.get("schema") != R225_TYPED_CONTRACT_SCHEMA:
        raise R228StageError("canonical r225 typed contract schema mismatch")
    if payload.get("owner_decision_revision") != R225_TYPED_CONTRACT_OWNER_DECISION_REVISION:
        raise R228StageError("canonical r225 typed contract is not the final r246 revision")
    if (
        payload.get("owner_proven_deterministic_terminal_win_this_turn_revision")
        != PROVEN_TERMINAL_WIN_REVISION
    ):
        raise R228StageError(
            "canonical r225 typed contract lacks the final r246 terminal-win owner revision"
        )
    if sha256_file(contract_path) != R225_TYPED_CONTRACT_SHA256:
        raise R228StageError("canonical r225 typed contract digest mismatch")
    phase1_environment = payload.get("phase1_submission_environment")
    if (
        not isinstance(phase1_environment, dict)
        or phase1_environment.get(
            "gpu_or_os_python_environment_is_not_inferred_from_the_reported_submission_resource_values"
        )
        is not True
        or "gpu_available" in phase1_environment
    ):
        raise R228StageError(
            "canonical r225 typed contract makes a stale GPU resource inference"
        )
    local_preflight = payload.get("local_preflight")
    if not isinstance(local_preflight, dict):
        raise R228StageError("canonical r225 typed contract lacks local preflight")
    if local_preflight.get("r240_hybrid_scheduler") != r240_hybrid_scheduler_contract():
        raise R228StageError("canonical r225 typed contract r240 scheduler mismatch")
    if local_preflight.get("deterministic_continuation") != deterministic_continuation_contract():
        raise R228StageError("canonical r225 typed contract continuation mismatch")
    if not _r244_search_identity_matches_canonical(
        local_preflight.get("r238_two_lane_receipt_contract")
    ):
        raise R228StageError("canonical r225 typed contract r244 search identity mismatch")
    required_gates = local_preflight.get(
        "all_required_local_preflight_receipts_must_pass_before_any_future_owner_authorized_kaggle_submission"
    )
    expected_gates = {
        *R240_REQUIRED_REGRESSION_RECEIPTS,
        R244_HANDLE_SCOPED_SEARCH_IDENTITY_REGRESSION_RECEIPT,
        PROVEN_TERMINAL_WIN_REGRESSION_RECEIPT,
    }
    if not isinstance(required_gates, list) or not expected_gates <= set(required_gates):
        raise R228StageError(
            "canonical r225 typed contract lacks r242/r244/r246 regression gates"
        )
    return {
        "path": R225_TYPED_CONTRACT_PATH,
        "schema": R225_TYPED_CONTRACT_SCHEMA,
        "sha256": R225_TYPED_CONTRACT_SHA256,
    }


def _canonical_native_members_by_path() -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    platforms: set[str] = set()
    wheel_members: set[str] = set()
    for raw_member in CANONICAL_LIBCG_MEMBERS:
        member = dict(raw_member)
        platform = member.get("platform")
        package_relative_path = member.get("package_relative_path")
        wheel_member = member.get("wheel_member")
        if not all(
            isinstance(value, str) and value
            for value in (platform, package_relative_path, wheel_member)
        ):
            raise R228StageError("canonical libcg member identity is malformed")
        if (
            package_relative_path in members
            or platform in platforms
            or wheel_member in wheel_members
        ):
            raise R228StageError("canonical libcg member identity is not unique")
        members[package_relative_path] = member
        platforms.add(platform)
        wheel_members.add(wheel_member)
    if len(members) != 4:
        raise R228StageError("canonical libcg contract must bind exactly four members")
    return members


def canonical_libcg_contract() -> dict[str, Any]:
    """Return the complete offline R236 native-library identity contract."""

    native_libraries: dict[str, dict[str, Any]] = {}
    for member in _canonical_native_members_by_path().values():
        platform = str(member["platform"])
        native_libraries[platform] = {
            key: value for key, value in member.items() if key != "platform"
        }
    provenance = dict(CANONICAL_LIBCG_UPSTREAM_PROVENANCE)
    provenance.update(
        {
            "wheel_filename": CANONICAL_LIBCG_WHEEL_FILENAME,
            "wheel_sha256": CANONICAL_LIBCG_WHEEL_SHA256,
            "wheel_size_bytes": CANONICAL_LIBCG_WHEEL_BYTES,
        }
    )
    return {
        "schema": CANONICAL_LIBCG_SCHEMA,
        "typed_source": CANONICAL_LIBCG_TYPED_SOURCE,
        "owner_decision_revision": CANONICAL_LIBCG_OWNER_DECISION_REVISION,
        "upstream_provenance": provenance,
        "canonical_native_libraries": native_libraries,
        "required_native_exports": list(REQUIRED_STOCK_LIBCG_EXPORTS),
        "scope": {
            "r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary": True,
            "all_four_platform_native_members_must_be_bound": True,
            "mixed_old_and_new_library_sets_allowed": False,
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
        },
        "r225_package_preflight": dict(R225_CANONICAL_LIBCG_PACKAGE_PREFLIGHT),
    }


def exact_frozen_base_contract() -> dict[str, Any]:
    """Bind frozen r195 assets while naming the replacement native identity."""

    canonical_linux = _canonical_native_members_by_path()["cg/libcg.so"]
    return {
        "r195_bundle_sha256": R195_BUNDLE_SHA256,
        "r195_checkpoint_sha256": R195_MODEL_SHA256,
        "r195_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "r195_search_config_sha256": R195_SEARCH_CONFIG_SHA256,
        "stock_libcg_relative_path": "cg/libcg.so",
        "stock_libcg_sha256": canonical_linux["sha256"],
        "stock_libcg_size_bytes": canonical_linux["size_bytes"],
        "stock_libcg_source": "official kaggle-environments 1.32.6 wheel overlay",
        "stock_libcg_official_wheel_sha256": CANONICAL_LIBCG_WHEEL_SHA256,
        "stock_libcg_update_commit": CANONICAL_LIBCG_UPSTREAM_PROVENANCE[
            "native_library_update_commit"
        ],
        "frozen_r195_bundle_model_tree_policy_and_deck_are_preserved_but_its_old_libcg_member_is_superseded": True,
        "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
        "required_stock_libcg_exports": list(REQUIRED_STOCK_LIBCG_EXPORTS),
    }


def broker_contract() -> dict[str, Any]:
    """Return the immutable parent/child containment contract for this package.

    Keep this as data rather than importing the broker: staging is deliberately
    build-only and must not instantiate a child, a native runtime, or a Kaggle
    client merely to create its receipt.
    """

    return {
        "module": BROKER_MODULE,
        "schema": BROKER_SCHEMA,
        "complete_action_cap": COMPLETE_ACTION_CAP,
        "action_timeout_seconds": BROKER_ACTION_TIMEOUT_SECONDS,
        "search_seconds": BROKER_SEARCH_SECONDS,
        "startup_timeout_seconds": BROKER_STARTUP_TIMEOUT_SECONDS,
        "reap_grace_seconds": BROKER_REAP_GRACE_SECONDS,
        "degraded_fallback_marker": DEGRADED_FALLBACK_MARKER,
        "subprocess_containment_identity": SUBPROCESS_CONTAINMENT_IDENTITY,
        "subprocess_containment": {
            "child_entrypoint": f"python -m {BROKER_MODULE}",
            "signals_exact_owned_child_only": True,
            "process_group_or_session_signalling": False,
            "bounded_reap_required": True,
        },
    }


def _member_name(member: tarfile.TarInfo) -> str:
    return member.name.removeprefix("./").strip("/")


def safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract only unique regular r195 members below the staging directory."""

    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        seen: set[str] = set()
        for member in members:
            name = _member_name(member)
            if not name:
                continue
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise R228StageError("r195 archive contains an unsafe member path")
            if member.issym() or member.islnk() or member.isdev():
                raise R228StageError("r195 archive contains an unsafe linked/device member")
            if not (member.isfile() or member.isdir()):
                raise R228StageError("r195 archive contains an unsupported member type")
            if name in seen:
                raise R228StageError("r195 archive contains a duplicate member")
            seen.add(name)
        source.extractall(destination, members=members, filter="data")


def verify_canonical_libcg_wheel(wheel: Path) -> dict[str, bytes]:
    """Read only the four checksum-bound native members from the supplied wheel."""

    if not wheel.is_file() or wheel.is_symlink():
        raise R228StageError(f"canonical libcg wheel is missing or not regular: {wheel}")
    if sha256_file(wheel) != CANONICAL_LIBCG_WHEEL_SHA256:
        raise R228StageError("canonical libcg wheel digest mismatch")
    if wheel.stat().st_size != CANONICAL_LIBCG_WHEEL_BYTES:
        raise R228StageError("canonical libcg wheel size mismatch")

    expected = _canonical_native_members_by_path()
    by_wheel_member = {
        str(member["wheel_member"]): member for member in expected.values()
    }
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                if info.filename not in by_wheel_member:
                    continue
                if info.filename in infos:
                    raise R228StageError(
                        f"canonical libcg wheel has duplicate member: {info.filename}"
                    )
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise R228StageError(
                        f"canonical libcg wheel member is not a regular file: {info.filename}"
                    )
                infos[info.filename] = info

            payloads: dict[str, bytes] = {}
            for wheel_member, member in by_wheel_member.items():
                info = infos.get(wheel_member)
                if info is None:
                    raise R228StageError(
                        f"canonical libcg wheel lacks required member: {wheel_member}"
                    )
                payload = archive.read(info)
                relative = str(member["package_relative_path"])
                if len(payload) != member["size_bytes"]:
                    raise R228StageError(
                        f"canonical libcg member size mismatch: {relative}"
                    )
                if sha256_bytes(payload) != member["sha256"]:
                    raise R228StageError(
                        f"canonical libcg member digest mismatch: {relative}"
                    )
                payloads[relative] = payload
    except (OSError, zipfile.BadZipFile) as exc:
        raise R228StageError(f"cannot read canonical libcg wheel: {wheel}") from exc

    return payloads


def _frozen_cg_wrapper_members(stage: Path) -> dict[str, str]:
    """Snapshot non-native r195 ``cg`` files that the overlay must retain."""

    cg_root = stage / "cg"
    if not cg_root.is_dir() or cg_root.is_symlink():
        raise R228StageError("r195 archive lacks a regular cg wrapper directory")
    native_paths = set(_canonical_native_members_by_path())
    members: dict[str, str] = {}
    for path in sorted(cg_root.rglob("*")):
        if path.is_symlink():
            raise R228StageError("r195 cg wrapper contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        if relative not in native_paths:
            members[relative] = sha256_file(path)
    return members


def overlay_canonical_libcg_members(stage: Path, payloads: dict[str, bytes]) -> None:
    """Overlay only canonical native siblings; never replace the ``cg`` tree."""

    expected = _canonical_native_members_by_path()
    if set(payloads) != set(expected):
        raise R228StageError("canonical libcg overlay does not contain exactly four members")
    cg_root = stage / "cg"
    if not cg_root.is_dir() or cg_root.is_symlink():
        raise R228StageError("r195 archive lacks a regular cg wrapper directory")
    for relative, payload in payloads.items():
        target = stage / relative
        if target.exists() and (not target.is_file() or target.is_symlink()):
            raise R228StageError(f"cannot safely overlay canonical native member: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        os.chmod(target, 0o644)


def verify_canonical_libcg_stage(stage: Path) -> dict[str, str]:
    """Reject missing, stale, or mixed native files after the narrow overlay."""

    expected = _canonical_native_members_by_path()
    cg_root = stage / "cg"
    if not cg_root.is_dir() or cg_root.is_symlink():
        raise R228StageError("staged cg wrapper directory is unavailable")
    observed_native_paths: set[str] = set()
    for path in sorted(cg_root.rglob("*")):
        if path.is_symlink():
            raise R228StageError("staged cg wrapper contains a symlink")
        name = path.name.lower()
        is_native_library = ".so" in name or ".dylib" in name or ".dll" in name
        if path.is_file() and is_native_library:
            observed_native_paths.add(path.relative_to(stage).as_posix())
    if observed_native_paths != set(expected):
        raise R228StageError("staged cg tree has old, missing, or mixed native members")

    observed: dict[str, str] = {}
    for relative, member in expected.items():
        path = stage / relative
        if not path.is_file() or path.is_symlink():
            raise R228StageError(f"staged canonical native member is unavailable: {relative}")
        if path.stat().st_size != member["size_bytes"]:
            raise R228StageError(f"staged canonical native size mismatch: {relative}")
        digest = sha256_file(path)
        if digest != member["sha256"]:
            raise R228StageError(f"staged canonical native digest mismatch: {relative}")
        observed[relative] = digest
    return observed


def verify_frozen_cg_wrapper_retained(stage: Path, before: dict[str, str]) -> None:
    if _frozen_cg_wrapper_members(stage) != before:
        raise R228StageError("canonical native overlay modified the frozen r195 cg wrapper")


def _require_regular(stage: Path, relative: str) -> Path:
    path = stage / relative
    if not path.is_file() or path.is_symlink():
        raise R228StageError(f"r195 archive lacks required regular file: {relative}")
    return path


def verify_r195_stage(stage: Path) -> dict[str, str]:
    paths = {
        "main.py": _require_regular(stage, "main.py"),
        "model.pt": _require_regular(stage, "model.pt"),
        "matchup_tree.json": _require_regular(stage, "matchup_tree.json"),
        "search_config.json": _require_regular(stage, "search_config.json"),
        "cg/libcg.so": _require_regular(stage, "cg/libcg.so"),
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "model.pt": R195_MODEL_SHA256,
        "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
        "search_config.json": R195_SEARCH_CONFIG_SHA256,
        "cg/libcg.so": STOCK_LIBCG_SHA256,
    }
    for name, digest in expected.items():
        if observed[name] != digest:
            raise R228StageError(f"r195 member digest mismatch: {name}")
    if paths["cg/libcg.so"].stat().st_size != STOCK_LIBCG_BYTES:
        raise R228StageError("r195 stock cg/libcg.so size changed")
    return observed


def _copy_source(source: Path, destination: Path) -> str:
    if not source.is_file() or source.is_symlink():
        raise R228StageError(f"required r238 source is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)
    return sha256_file(destination)


def _contains_selected_action(node: ast.AST | None) -> bool:
    return node is not None and any(
        isinstance(item, ast.Attribute) and item.attr == "selected_action"
        for item in ast.walk(node)
    )


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


def _referenced_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _literal_module_constant(
    tree: ast.Module, name: str, expected: object
) -> bool:
    """Return whether a module-level named constant has one exact literal."""

    for node in tree.body:
        targets: set[str] = set()
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                targets.update(_target_names(target))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets.update(_target_names(node.target))
            value = node.value
        if (
            name in targets
            and isinstance(value, ast.Constant)
            and value.value == expected
        ):
            return True
    return False


def _module_function(tree: ast.Module, name: str) -> ast.AST | None:
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def _named_call_lines(function: ast.AST, name: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _attribute_call_lines(function: ast.AST, attribute: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def _matching_dict_values(tree: ast.AST, first: str, second: str) -> bool:
    """Require two public payload aliases to originate from one exact value."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        values = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        first_value = values.get(first)
        second_value = values.get(second)
        if first_value is None or second_value is None:
            continue
        if ast.dump(first_value, include_attributes=False) == ast.dump(
            second_value, include_attributes=False
        ):
            return True
    return False


def _contains_direct_agent_call(node: ast.AST | None) -> bool:
    """Identify a frozen direct-action computation without executing it."""

    if node is None:
        return False
    for item in ast.walk(node):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr != "agent":
            continue
        if any("direct" in name.lower() for name in _referenced_names(item.func.value)):
            return True
    return False


def _contains_direct_precompute_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    names = _referenced_names(node.func)
    return any("direct" in name.lower() and "precompute" in name.lower() for name in names)


def _is_broker_select(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "select":
        return False
    return any("broker" in name.lower() for name in _referenced_names(call.func.value))


def _main_delegates_to_broker_with_direct_fallback(wrapper_tree: ast.Module) -> bool:
    """Require parent-held direct authority to be supplied to broker.select()."""

    agent = next(
        (
            function
            for function in wrapper_tree.body
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            and function.name == "agent"
        ),
        None,
    )
    if agent is None:
        return False

    direct_action_lines: dict[str, int] = {}
    for node in ast.walk(agent):
        if isinstance(node, ast.Assign):
            targets = set().union(*(_target_names(target) for target in node.targets))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = _target_names(node.target)
            value = node.value
        else:
            continue
        if _contains_direct_agent_call(value) or _contains_direct_precompute_call(value):
            for target in targets:
                direct_action_lines[target] = getattr(node, "lineno", 0)

    for node in ast.walk(agent):
        if not isinstance(node, ast.Call) or not _is_broker_select(node):
            continue
        supplied = set().union(
            *(_referenced_names(argument) for argument in node.args),
            *(_referenced_names(keyword.value) for keyword in node.keywords),
        )
        if any(
            name in supplied and line < getattr(node, "lineno", 0)
            for name, line in direct_action_lines.items()
        ):
            return True
    return False


def _main_has_explicit_broker_settings(wrapper_tree: ast.Module) -> bool:
    expected = {
        "R234_BROKER_ACTION_TIMEOUT_SECONDS": BROKER_ACTION_TIMEOUT_SECONDS,
        "R234_BROKER_SEARCH_SECONDS": BROKER_SEARCH_SECONDS,
        "R234_BROKER_STARTUP_TIMEOUT_SECONDS": BROKER_STARTUP_TIMEOUT_SECONDS,
        "R234_BROKER_REAP_GRACE_SECONDS": BROKER_REAP_GRACE_SECONDS,
    }
    if not all(
        _literal_module_constant(wrapper_tree, name, value)
        for name, value in expected.items()
    ):
        return False
    for node in ast.walk(wrapper_tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "IsolatedR228SearchBroker":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if all(
            isinstance(keywords.get(argument), ast.Name)
            and keywords[argument].id == constant
            for argument, constant in {
                "action_timeout_seconds": "R234_BROKER_ACTION_TIMEOUT_SECONDS",
                "search_seconds": "R234_BROKER_SEARCH_SECONDS",
                "startup_timeout_seconds": "R234_BROKER_STARTUP_TIMEOUT_SECONDS",
                "reap_grace_seconds": "R234_BROKER_REAP_GRACE_SECONDS",
            }.items()
        ):
            return True
    return False


def _main_has_r240_scheduler_constants(wrapper_tree: ast.Module) -> bool:
    """Reject historical eight-second or unbounded R240 parent scheduling."""

    expected = {
        "R234_BROKER_ACTION_TIMEOUT_SECONDS": BROKER_ACTION_TIMEOUT_SECONDS,
        "R234_BROKER_SEARCH_SECONDS": BROKER_SEARCH_SECONDS,
        "R234_BROKER_STARTUP_TIMEOUT_SECONDS": BROKER_STARTUP_TIMEOUT_SECONDS,
        "R234_BROKER_REAP_GRACE_SECONDS": BROKER_REAP_GRACE_SECONDS,
        "R238_MINIMUM_BACKUPS_BEFORE_STABILITY": (
            R240_MINIMUM_BACKUPS_BEFORE_STABILITY
        ),
        "R238_STABLE_ROOT_LEADER_OBSERVATIONS_REQUIRED": (
            R240_STABLE_ROOT_LEADER_OBSERVATIONS
        ),
        "R238_MAXIMUM_BACKUPS_PER_DECISION": R240_MAXIMUM_BACKUPS_PER_DECISION,
        "R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD": (
            R240_HIGH_CONFIDENCE_DIRECT_THRESHOLD
        ),
        "R240_MAX_PRINCIPAL_VARIATION_DEPTH": (
            R240_MAX_PRINCIPAL_VARIATION_DEPTH
        ),
    }
    return all(
        _literal_module_constant(wrapper_tree, name, value)
        for name, value in expected.items()
    )


def _main_has_r242_hybrid_shortcuts(wrapper_tree: ast.Module) -> bool:
    """Require direct/continuation shortcuts before any new broker selection."""

    required_functions = {
        name: _module_function(wrapper_tree, name)
        for name in (
            "agent",
            "_precompute_validated_direct_action",
            "_is_high_confidence_direct",
            "_consume_principal_variation",
            "_replace_principal_variation",
            "_canonical_observation_fingerprint",
            "_emit_parent_direct_or_continuation_decision",
            "_journal_real_action",
        )
    }
    if any(function is None for function in required_functions.values()):
        return False
    agent = required_functions["agent"]
    assert agent is not None
    precompute_lines = _named_call_lines(agent, "_precompute_validated_direct_action")
    continuation_lines = _named_call_lines(agent, "_consume_principal_variation")
    high_confidence_lines = _named_call_lines(agent, "_is_high_confidence_direct")
    broker_start_lines = _named_call_lines(agent, "_ensure_broker_for_selection")
    broker_select_lines = _attribute_call_lines(agent, "select")
    if not all(
        (precompute_lines, continuation_lines, high_confidence_lines, broker_start_lines)
    ):
        return False
    first_broker_line = min((*broker_start_lines, *broker_select_lines))
    if max(
        max(precompute_lines), max(continuation_lines), max(high_confidence_lines)
    ) >= first_broker_line:
        return False

    high_confidence = required_functions["_is_high_confidence_direct"]
    continuation = required_functions["_consume_principal_variation"]
    replacement = required_functions["_replace_principal_variation"]
    emitter = required_functions["_emit_parent_direct_or_continuation_decision"]
    journal = required_functions["_journal_real_action"]
    assert all(
        function is not None
        for function in (high_confidence, continuation, replacement, emitter, journal)
    )
    high_source = ast.unparse(high_confidence)
    continuation_source = ast.unparse(continuation)
    replacement_source = ast.unparse(replacement)
    emitter_source = ast.unparse(emitter)
    journal_source = ast.unparse(journal)
    agent_source = ast.unparse(agent)
    high_fields = r240_hybrid_scheduler_contract()[
        "high_confidence_journal_required_fields"
    ]
    continuation_fields = deterministic_continuation_contract()[
        "journal_required_fields"
    ]
    return (
        "factorized_selected_stage_probabilities" in high_source
        and "math.isfinite" in high_source
        and ">= R238_HIGH_CONFIDENCE_DIRECT_THRESHOLD" in high_source
        and "return all" in high_source
        and "_clear_principal_variation" in continuation_source
        and "_canonical_observation_fingerprint" in continuation_source
        and "_current_actor_seat" in continuation_source
        and "planned_action not in legal" in continuation_source
        and "del _GAME_PRINCIPAL_VARIATION[0]" in continuation_source
        and "R240_MAX_PRINCIPAL_VARIATION_DEPTH" in replacement_source
        and "root_seat" in replacement_source
        and "observation_fingerprint" in replacement_source
        # The reusable emitter receives the mode as an argument.  Authority
        # therefore lives at the two call sites in ``agent`` rather than in
        # the emitter body itself.
        and R240_HIGH_CONFIDENCE_DIRECT_MODE in agent_source
        and "deterministic_mcts_continuation" in agent_source
        and all(field in emitter_source for field in high_fields)
        and all(field in emitter_source for field in continuation_fields)
        and "'mcts_child_started_for_this_decision': False" in emitter_source
        and "'mcts_select_call_count': 0" in emitter_source
        and "'degraded': False" in emitter_source
        and "_note_direct_action" in journal_source
        and "return 1" in journal_source
        and "return 0" in journal_source
        and ".select" not in high_source
        and ".select" not in continuation_source
    )


def _runtime_has_r240_adaptive_schedule(
    runtime_tree: ast.Module, runtime_source: str
) -> bool:
    """Require the child to instantiate its two-lane adaptive stop contract."""

    expected_constants = {
        "R238_DEFAULT_SEARCH_SECONDS": BROKER_SEARCH_SECONDS,
        "R238_MINIMUM_BACKUPS_BEFORE_STABILITY": (
            R240_MINIMUM_BACKUPS_BEFORE_STABILITY
        ),
        "R238_STABLE_ROOT_LEADER_OBSERVATIONS": (
            R240_STABLE_ROOT_LEADER_OBSERVATIONS
        ),
        "R238_MAXIMUM_BACKUPS_PER_DECISION": R240_MAXIMUM_BACKUPS_PER_DECISION,
    }
    if not all(
        _literal_module_constant(runtime_tree, name, value)
        for name, value in expected_constants.items()
    ):
        return False
    if any(
        legacy in runtime_source
        for legacy in (
            "POKEBOT_R228_DECISION_SECONDS",
            "R228_DEFAULT_DECISION_SECONDS",
            "R228_DECISION_SECONDS_ENV",
        )
    ):
        return False

    expected_queue_keywords = {
        "lane_count": "R228_SIMULATOR_LANE_COUNT",
        "minimum_backups_before_stability": "R238_MINIMUM_BACKUPS_BEFORE_STABILITY",
        "stable_root_leader_observations": "R238_STABLE_ROOT_LEADER_OBSERVATIONS",
        "maximum_backups_per_decision": "R238_MAXIMUM_BACKUPS_PER_DECISION",
    }
    queue_configured = False
    deadline_configured = False
    for node in ast.walk(runtime_tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        else:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if callee == "PersistentAsyncSharedTreeMCTS" and all(
            isinstance(keywords.get(argument), ast.Name)
            and keywords[argument].id == expected_name
            for argument, expected_name in expected_queue_keywords.items()
        ):
            queue_configured = True
        if callee == "run_decision" and "deadline_monotonic" in keywords:
            deadline_configured = True
    required_receipt_fields = {
        "stop_reason",
        "minimum_backups_before_stability",
        "stable_root_leader_observations_required",
        "maximum_backups_per_decision",
        "observed_stable_root_leader_observations",
        "root_seat",
        "principal_variation",
    }
    literals = {
        literal.value
        for literal in ast.walk(runtime_tree)
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
    }
    fingerprint = next(
        (
            node
            for node in runtime_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "canonical_observation_fingerprint"
        ),
        None,
    )
    if fingerprint is None:
        return False
    fingerprint_source = ast.unparse(fingerprint)
    has_canonical_fingerprint = (
        "sort_keys=True" in fingerprint_source
        and "separators=(',', ':')" in fingerprint_source
        and "ensure_ascii=True" in fingerprint_source
        and "allow_nan=False" in fingerprint_source
        and "sha256:" in fingerprint_source
        and "lane_id" not in fingerprint_source
    )
    has_principal_variation_validation = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_validate_principal_variation"
        for node in runtime_tree.body
    )
    return (
        queue_configured
        and deadline_configured
        and required_receipt_fields <= literals
        and has_canonical_fingerprint
        and has_principal_variation_validation
        and "smoke_min_depth" not in runtime_source
    )


def _runtime_has_r242_actor_change_boundary(
    runtime_tree: ast.Module, runtime_source: str
) -> bool:
    """Require value-only actor-change/chance boundaries in the child runtime."""

    gameplay = next(
        (
            node
            for node in runtime_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "R228AsyncGameplay"
        ),
        None,
    )
    if gameplay is None:
        return False
    evaluate_batch = next(
        (
            node
            for node in gameplay.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_evaluate_batch"
        ),
        None,
    )
    if evaluate_batch is None:
        return False
    source = ast.unparse(evaluate_batch)
    required_fragments = (
        "actor_change_boundary = actor != root_seat",
        "boundary = chance_boundary or actor_change_boundary",
        "combos = ((),)",
        "forward_leaf_batch(self.model, packets)",
        "legal_actions=() if boundary else combos",
        "priors=() if boundary else priors",
        "boundary=boundary",
        "actor_seat=actor",
        "observation_fingerprint=canonical_observation_fingerprint",
        "value=float(leaf.value)",
    )
    receipt_fields = {
        "actor_change_boundary_leaf_count",
        "chance_boundary_leaf_count",
        "boundary_leaf_count",
    }
    literals = {
        literal.value
        for literal in ast.walk(runtime_tree)
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
    }
    return (
        all(fragment in source for fragment in required_fragments)
        and receipt_fields <= literals
        and "root_actor_boundary_leaf_count" not in runtime_source
    )


def _runtime_has_r244_handle_scoped_search_identity(
    runtime_tree: ast.Module, runtime_source: str
) -> bool:
    """Require public lane/handle/SearchId composites, never raw-ID authority."""

    validator = _module_function(runtime_tree, "_validate_lane_search_composites")
    if validator is None:
        return False
    validator_source = ast.unparse(validator)
    module_source = ast.unparse(runtime_tree)
    required_fields = {
        "per_lane_handle_identities",
        "per_lane_search_id_chains",
        "per_lane_first_search_ids",
        "handle_scoped_first_search_id_composite_states",
        "distinct_search_begin_composite_count",
    }
    literals = {
        literal.value
        for literal in ast.walk(runtime_tree)
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
    }
    required_fragments = (
        "zip(per_lane_handles, per_lane_chains, per_lane_depth)",
        "composites = tuple",
        "len(set(composites))",
        "reported_count",
        "distinct_search_begin_composite_count",
        "for lane_id in range(R228_SIMULATOR_LANE_COUNT)",
        "'lane_id': lane_id",
        "'handle_identity': per_lane_handle_identities[lane_id]",
        "'first_search_id': per_lane_first_search_ids[lane_id]",
    )
    return (
        required_fields <= literals
        and all(fragment in module_source for fragment in required_fragments)
        and all(fragment in validator_source for fragment in required_fragments[:5])
        and "distinct_search_begin_id_count" not in runtime_source
    )


def _queue_has_r240_deterministic_continuation(queue_tree: ast.Module) -> bool:
    """Require the child queue to derive only a two-lane-backed PV suffix."""

    if not _literal_module_constant(
        queue_tree, "DEFAULT_LANE_COUNT", SIMULATOR_SEARCH_LANE_COUNT
    ) or not _literal_module_constant(
        queue_tree,
        "MAX_PRINCIPAL_VARIATION_DEPTH",
        R240_MAX_PRINCIPAL_VARIATION_DEPTH,
    ):
        return False
    queue_class = next(
        (
            node
            for node in queue_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PersistentAsyncSharedTreeMCTS"
        ),
        None,
    )
    if queue_class is None:
        return False
    principal_variation = next(
        (
            node
            for node in queue_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_principal_variation"
        ),
        None,
    )
    if principal_variation is None:
        return False
    arguments = {
        argument.arg
        for argument in (
            *principal_variation.args.args,
            *principal_variation.args.kwonlyargs,
        )
    }
    if "root_seat" not in arguments:
        return False
    source = ast.unparse(principal_variation)
    required_fragments = (
        "self._lane_count != DEFAULT_LANE_COUNT",
        "range(self._lane_count)",
        "node.observation_fingerprint",
        "node.boundary",
        "node.actor_seat != root_seat",
        "self._root_leader(node)",
        "child.visits > 0",
        "observation_fingerprint",
        "action",
    )
    return all(fragment in source for fragment in required_fragments)


def _queue_has_r244_handle_scoped_search_identity(queue_tree: ast.Module) -> bool:
    """Require two handle-scoped SearchBegin composites in queue receipts."""

    source = ast.unparse(queue_tree)
    required_fragments = (
        "per_lane_handle_identities",
        "distinct_search_begin_composite_count",
        "search_begin_composites = tuple",
        "(self._handle_identities[lane], int(open_rows[lane].search_id))",
        "len(set(search_begin_composites))",
        "distinct_search_begin_composite_count != self._lane_count",
        "per_lane_search_id_chains=per_lane_search_id_chains",
        "per_lane_handle_identities=self._handle_identities",
    )
    return all(fragment in source for fragment in required_fragments)


def _string_literals(tree: ast.AST) -> set[str]:
    """Return literal protocol labels without executing staged package code."""

    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _module_imports_names(tree: ast.Module, names: set[str]) -> bool:
    """Return whether one explicit staged import binds every requested symbol."""

    return any(
        names <= {alias.name for alias in node.names}
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    )


def _has_r246_terminal_win_constants(tree: ast.Module) -> bool:
    return all(
        (
            _literal_module_constant(
                tree, "PROVEN_TERMINAL_WIN_REVISION", PROVEN_TERMINAL_WIN_REVISION
            ),
            _literal_module_constant(
                tree,
                "PROVEN_TERMINAL_WIN_STOP_REASON",
                PROVEN_TERMINAL_WIN_STOP_REASON,
            ),
            _literal_module_constant(
                tree,
                "PROVEN_TERMINAL_WIN_PROOF_KIND",
                PROVEN_TERMINAL_WIN_PROOF_KIND,
            ),
        )
    )


def _terminal_win_proof_schema_is_present(tree: ast.AST) -> bool:
    return set(PROVEN_TERMINAL_WIN_PROOF_FIELDS) <= _string_literals(tree)


def _function_argument_names(function: ast.AST | None) -> set[str]:
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    return {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }


def _class_method(class_node: ast.ClassDef | None, name: str) -> ast.AST | None:
    if class_node is None:
        return None
    return next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def _queue_has_r246_terminal_win_proof(queue_tree: ast.Module) -> bool:
    """Require the queue to construct one backed exact simulator proof only."""

    queue_class = next(
        (
            node
            for node in queue_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PersistentAsyncSharedTreeMCTS"
        ),
        None,
    )
    proof_builder = _class_method(queue_class, "_proven_terminal_win_this_turn")
    run_decision = _class_method(queue_class, "run_decision")
    if proof_builder is None or run_decision is None:
        return False
    if not _has_r246_terminal_win_constants(queue_tree):
        return False
    if not _terminal_win_proof_schema_is_present(proof_builder):
        return False
    if not {
        "root_observation_fingerprint",
        "root_legal_order_fingerprint",
        "root_actor_seat",
    } <= _function_argument_names(run_decision):
        return False

    proof_names = _referenced_names(proof_builder)
    proof_attributes = {
        node.attr for node in ast.walk(proof_builder) if isinstance(node, ast.Attribute)
    }
    proof_source = ast.unparse(proof_builder)
    run_source = ast.unparse(run_decision)
    queue_source = ast.unparse(queue_tree)
    return (
        {
            "PROVEN_TERMINAL_WIN_PROOF_KIND",
            "root_actor_seat",
            "root_observation_fingerprint",
            "root_legal_order_fingerprint",
        }
        <= proof_names
        and {
            "terminal_leaf_reached",
            "terminal_result",
            "terminal_winner_seat",
            "chance_boundary",
            "actor_change_boundary",
            "unresolved_randomness",
            "action_path",
            "actor_path",
        }
        <= proof_attributes
        and "leaf.terminal_result != 'win'" in proof_source
        and "leaf.terminal_winner_seat != root_actor_seat" in proof_source
        and "leaf.chance_boundary" in proof_source
        and "leaf.actor_change_boundary" in proof_source
        and "leaf.unresolved_randomness" in proof_source
        and "any((actor != root_actor_seat for actor in context.actor_path))"
        in proof_source
        and "matching_root_edges[0].visits < 1" in proof_source
        and "terminal_win_proof" in run_source
        and "stop_reason = PROVEN_TERMINAL_WIN_STOP_REASON" in queue_source
        and "proof_action" in run_source
        and "selected_action_visits" in queue_source
    )


def _runtime_has_r246_terminal_win_validation(
    runtime_tree: ast.Module, runtime_source: str
) -> bool:
    """Require the child to bind a terminal proof to its exact current root."""

    validator = _module_function(runtime_tree, "_validate_terminal_win_proof")
    if validator is None:
        return False
    if not _module_imports_names(
        runtime_tree,
        {
            "PROVEN_TERMINAL_WIN_REVISION",
            "PROVEN_TERMINAL_WIN_STOP_REASON",
            "PROVEN_TERMINAL_WIN_PROOF_KIND",
        },
    ) or not _terminal_win_proof_schema_is_present(validator):
        return False
    validator_names = _referenced_names(validator)
    validator_source = ast.unparse(validator)
    return (
        {
            "PROVEN_TERMINAL_WIN_REVISION",
            "PROVEN_TERMINAL_WIN_STOP_REASON",
            "PROVEN_TERMINAL_WIN_PROOF_KIND",
        }
        <= validator_names
        and {
            "root_observation_fingerprint",
            "root_legal_order_fingerprint",
            "root_actor_seat",
            "legal_actions",
        }
        <= _function_argument_names(validator)
        and "terminal_win_proof" in validator_source
        and "stop_reason != PROVEN_TERMINAL_WIN_STOP_REASON" in validator_source
        and "set(raw_proof) != set(_TERMINAL_WIN_PROOF_KEYS)" in validator_source
        and "raw_proof.get('terminal_result') != 'win'" in validator_source
        and "selected_action not in normalized_legal" in validator_source
        and "terminal_result" in validator_source
        and "path_no_chance_boundary" in validator_source
        and "path_no_actor_change_boundary" in validator_source
        and "path_no_opponent_boundary_crossing" in validator_source
        and "path_no_unresolved_randomness" in validator_source
        and "root_observation_fingerprint=root_observation_fingerprint"
        in runtime_source
        and "root_legal_order_fingerprint=root_legal_order_fingerprint"
        in runtime_source
        and "root_actor_seat=root_seat" in runtime_source
        and "proven_deterministic_terminal_win_this_turn" in runtime_source
    )


def _broker_has_r246_terminal_win_validation(
    broker_tree: ast.Module, broker_source: str
) -> bool:
    """Require the IPC parent to reject stale, chance, or losing proof claims."""

    validator = _module_function(broker_tree, "_validate_terminal_win_proof")
    receipt_validator = _module_function(broker_tree, "_validate_two_lane_receipt")
    if validator is None or receipt_validator is None:
        return False
    if not _has_r246_terminal_win_constants(broker_tree):
        return False
    if not _terminal_win_proof_schema_is_present(validator):
        return False
    validator_source = ast.unparse(validator)
    receipt_source = ast.unparse(receipt_validator)
    return (
        {
            "PROVEN_TERMINAL_WIN_REVISION",
            "PROVEN_TERMINAL_WIN_PROOF_KIND",
        }
        <= _referenced_names(validator)
        and "PROVEN_TERMINAL_WIN_STOP_REASON" in receipt_source
        and "terminal_win_proof" in validator_source
        and "set(proof) != set(_TERMINAL_WIN_PROOF_KEYS)" in validator_source
        and "proof.get('terminal_result') != 'win'" in validator_source
        and "tuple(proof_selected) not in normalized_legal" in validator_source
        and "terminal_result" in validator_source
        and "path_no_chance_boundary" in validator_source
        and "path_no_actor_change_boundary" in validator_source
        and "path_no_opponent_boundary_crossing" in validator_source
        and "path_no_unresolved_randomness" in validator_source
        and "non-terminal stop claimed terminal-win authority" in broker_source
        and "_validate_terminal_win_proof" in receipt_source
    )


def _main_has_r246_terminal_win_parent_validation(
    wrapper_tree: ast.Module, wrapper_source: str
) -> bool:
    """Require the submission parent to independently validate proof authority."""

    validator = _module_function(wrapper_tree, "_validate_terminal_win_proof")
    receipt_validator = _module_function(wrapper_tree, "_validate_phase1_two_lane_receipt")
    if validator is None or receipt_validator is None:
        return False
    if not _has_r246_terminal_win_constants(wrapper_tree):
        return False
    if not _terminal_win_proof_schema_is_present(validator):
        return False
    validator_source = ast.unparse(validator)
    receipt_source = ast.unparse(receipt_validator)
    return (
        {
            "PROVEN_TERMINAL_WIN_REVISION",
            "PROVEN_TERMINAL_WIN_PROOF_KIND",
        }
        <= _referenced_names(validator)
        and "PROVEN_TERMINAL_WIN_STOP_REASON" in receipt_source
        and "terminal_win_proof" in validator_source
        and "set(proof) != set(_TERMINAL_WIN_PROOF_KEYS)" in validator_source
        and "proof.get('terminal_result') != 'win'" in validator_source
        and "proof_selected not in legal" in validator_source
        and "terminal_result" in validator_source
        and "path_no_chance_boundary" in validator_source
        and "path_no_actor_change_boundary" in validator_source
        and "path_no_opponent_boundary_crossing" in validator_source
        and "path_no_unresolved_randomness" in validator_source
        and "non-terminal stop claimed terminal-win authority" in wrapper_source
        and "_validate_terminal_win_proof" in receipt_source
    )


def _broker_has_pre_search_cuda_runtime_observation(
    broker_tree: ast.Module, broker_source: str
) -> bool:
    """Require an observational child CUDA receipt before any MCTS work.

    The Phase-1 resource envelope is deliberately silent about GPU visibility.
    The only acceptable evidence is the child process's own post-model-load,
    pre-search observation, carried through its ready identity to the parent.
    """

    if not _literal_module_constant(
        broker_tree, "CUDA_RUNTIME_OBSERVATION_SCHEMA", CUDA_RUNTIME_OBSERVATION_SCHEMA
    ) or not _literal_module_constant(
        broker_tree, "CUDA_RUNTIME_OBSERVATION_PHASE", CUDA_RUNTIME_OBSERVATION_PHASE
    ):
        return False
    capture = _module_function(broker_tree, "capture_cuda_runtime_before_search")
    child_new_runtime = _module_function(broker_tree, "_child_new_runtime")
    child_main = _module_function(broker_tree, "_child_main")
    if any(item is None for item in (capture, child_new_runtime, child_main)):
        return False
    assert capture is not None
    assert child_new_runtime is not None
    assert child_main is not None

    model_load_lines = _attribute_call_lines(child_new_runtime, "_ensure_runtime")
    capture_lines = _named_call_lines(
        child_new_runtime, "capture_cuda_runtime_before_search"
    )
    gameplay_lines = _named_call_lines(child_new_runtime, "R228AsyncGameplay")
    child_start_lines = _named_call_lines(child_main, "_child_new_runtime")
    child_select_lines = _attribute_call_lines(child_main, "select")
    if not all(
        (
            model_load_lines,
            capture_lines,
            gameplay_lines,
            child_start_lines,
            child_select_lines,
        )
    ):
        return False
    if not (
        max(model_load_lines) < min(capture_lines) < min(gameplay_lines)
        and max(child_start_lines) < min(child_select_lines)
    ):
        return False

    capture_source = ast.unparse(capture)
    child_main_source = ast.unparse(child_main)
    capture_literals = {
        literal.value
        for literal in ast.walk(capture)
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
    }
    required_payload_fields = {
        "torch_imported",
        "cuda_available",
        "cuda_initialized",
        "device_count",
        "devices",
        "model_device",
        "telemetry_complete",
        "error_types",
    }
    return (
        required_payload_fields <= capture_literals
        and "sys.modules.get('torch')" in capture_source
        and "cuda.is_available()" in capture_source
        and "cuda.device_count()" in capture_source
        and "cuda.get_device_name(index)" in capture_source
        and "cuda.mem_get_info(index)" in capture_source
        and CUDA_RUNTIME_OBSERVATION_MARKER in child_main_source
        and "type': 'ready'" in child_main_source
        and CUDA_RUNTIME_OBSERVATION_MARKER in broker_source
        and f"self._child_identity['{CUDA_RUNTIME_OBSERVATION_MARKER}']"
        in ast.unparse(broker_tree)
    )


def _main_has_pre_search_cuda_runtime_observation(
    wrapper_tree: ast.Module, wrapper_source: str
) -> bool:
    """Require the parent to record its loaded frozen-model CUDA state too."""

    direct_policy = _module_function(wrapper_tree, "_direct_policy_and_targets")
    capture = _module_function(
        wrapper_tree, "_capture_parent_cuda_runtime_before_search"
    )
    agent = _module_function(wrapper_tree, "agent")
    if any(item is None for item in (direct_policy, capture, agent)):
        return False
    assert direct_policy is not None
    assert capture is not None
    assert agent is not None
    model_load_lines = _named_call_lines(direct_policy, "ensure_runtime")
    capture_lines = _named_call_lines(
        direct_policy, "_capture_parent_cuda_runtime_before_search"
    )
    direct_lines = _named_call_lines(agent, "_precompute_validated_direct_action")
    broker_start_lines = _named_call_lines(agent, "_ensure_broker_for_selection")
    if not all((model_load_lines, capture_lines, direct_lines, broker_start_lines)):
        return False
    if not (
        max(model_load_lines) < min(capture_lines)
        and min(direct_lines) < min(broker_start_lines)
    ):
        return False
    capture_source = ast.unparse(capture)
    return (
        "capture_cuda_runtime_before_search" in wrapper_source
        and "CUDA_RUNTIME_OBSERVATION_SCHEMA" in wrapper_source
        and "CUDA_RUNTIME_OBSERVATION_PHASE" in wrapper_source
        and "parent_cuda_runtime_before_search" in wrapper_source
        and "capture_cuda_runtime_before_search(model)" in capture_source
    )


def _broker_has_r240_timeout_defaults(broker_tree: ast.Module) -> bool:
    """Ensure the broker cannot silently restore the historical 12/8 defaults."""

    broker_class = next(
        (
            node
            for node in broker_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "IsolatedR228SearchBroker"
        ),
        None,
    )
    if broker_class is None:
        return False
    init = next(
        (
            node
            for node in broker_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        ),
        None,
    )
    if init is None:
        return False
    fallback_values = {
        keyword.value.value
        for call in ast.walk(init)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_positive_seconds"
        for keyword in call.keywords
        if keyword.arg == "fallback" and isinstance(keyword.value, ast.Constant)
    }
    return {
        BROKER_ACTION_TIMEOUT_SECONDS,
        BROKER_SEARCH_SECONDS,
        BROKER_STARTUP_TIMEOUT_SECONDS,
        BROKER_REAP_GRACE_SECONDS,
    } <= fallback_values


def _broker_has_required_containment(broker_tree: ast.Module, broker_source: str) -> bool:
    """Check the packaged broker exposes the bounded exact-child contract."""

    if not _literal_module_constant(broker_tree, "SCHEMA", BROKER_SCHEMA):
        return False
    if not _literal_module_constant(broker_tree, "COMPLETE_ACTION_CAP", COMPLETE_ACTION_CAP):
        return False
    broker_class = next(
        (
            node
            for node in broker_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "IsolatedR228SearchBroker"
        ),
        None,
    )
    if broker_class is None:
        return False
    methods = {
        node.name: node
        for node in broker_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_methods = {"begin_game", "note_direct_action", "select", "close", "_dispose_child"}
    if not required_methods <= set(methods):
        return False
    init = methods.get("__init__")
    if init is None:
        return False
    init_args = {argument.arg for argument in init.args.args}
    if not {
        "action_timeout_seconds",
        "search_seconds",
        "startup_timeout_seconds",
        "reap_grace_seconds",
    } <= init_args:
        return False
    select_args = {argument.arg for argument in methods["select"].args.args}
    if "direct_action" not in select_args:
        return False
    calls = [node for node in ast.walk(broker_tree) if isinstance(node, ast.Call)]
    has_popen = any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        and call.func.attr == "Popen"
        for call in calls
    )
    has_terminate = any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "terminate"
        for call in calls
    )
    has_kill = any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "kill"
        for call in calls
    )
    has_bounded_wait = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "wait"
        and any(keyword.arg == "timeout" for keyword in call.keywords)
        for call in calls
    )
    return (
        has_popen
        and has_terminate
        and has_kill
        and has_bounded_wait
        and "max_combos=COMPLETE_ACTION_CAP" in broker_source
        and "os.killpg" not in broker_source
        and "start_new_session=True" not in broker_source
    )


def _function_returns_async_selected_action(function: ast.AST) -> bool:
    """Statically require the runtime to use its MCTS receipt as authority."""

    calls_run_decision = any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "run_decision"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_decision"
        )
        for node in ast.walk(function)
    )
    if not calls_run_decision:
        return False
    selected_names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and _contains_selected_action(node.value):
            for target in node.targets:
                selected_names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign) and _contains_selected_action(node.value):
            selected_names.update(_target_names(node.target))
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        if _contains_selected_action(node.value):
            return True
        if isinstance(node.value, ast.Call):
            values = node.value.args
        elif isinstance(node.value, (ast.Tuple, ast.List)):
            values = node.value.elts
        else:
            values = (node.value,)
        if any(isinstance(value, ast.Name) and value.id in selected_names for value in values):
            return True
    return False


def validate_async_action_authority(
    wrapper: Path,
    runtime: Path,
    broker: Path,
    queue: Path | None = None,
) -> None:
    """Reject uncontained or side-probe branching authority in the package."""

    try:
        wrapper_source = wrapper.read_text(encoding="utf-8")
        runtime_source = runtime.read_text(encoding="utf-8")
        broker_source = broker.read_text(encoding="utf-8")
        wrapper_tree = ast.parse(wrapper_source, filename=str(wrapper))
        runtime_tree = ast.parse(runtime_source, filename=str(runtime))
        broker_tree = ast.parse(broker_source, filename=str(broker))
        queue_tree = (
            None
            if queue is None
            else ast.parse(queue.read_text(encoding="utf-8"), filename=str(queue))
        )
    except (OSError, SyntaxError) as exc:
        raise R228StageError("cannot parse r238 wrapper/runtime/broker source") from exc

    constant_found = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "R228_ASYNC_SELECTED_ACTION_AUTHORITY"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == "receipt.selected_action"
        for node in wrapper_tree.body
    )
    if not constant_found:
        raise R228StageError("main.py does not declare receipt.selected_action authority")
    if not _literal_module_constant(wrapper_tree, "SCHEMA", SCHEMA):
        raise R228StageError("main.py does not bind the r238 package schema")
    if not _literal_module_constant(runtime_tree, "SCHEMA", SCHEMA):
        raise R228StageError("contained child runtime does not bind the r238 package schema")
    if not _literal_module_constant(
        runtime_tree, "R228_SIMULATOR_LANE_COUNT", SIMULATOR_SEARCH_LANE_COUNT
    ):
        raise R228StageError("contained child runtime does not bind exactly two lanes")
    if not _runtime_has_r240_adaptive_schedule(runtime_tree, runtime_source):
        raise R228StageError(
            "contained child runtime lacks the r240 two-second adaptive stop contract"
        )
    if queue_tree is not None:
        if not _main_has_r246_terminal_win_parent_validation(
            wrapper_tree, wrapper_source
        ):
            raise R228StageError(
                "main.py lacks r246 deterministic terminal-win parent validation"
            )
        if not _runtime_has_r246_terminal_win_validation(runtime_tree, runtime_source):
            raise R228StageError(
                "contained child runtime lacks r246 deterministic terminal-win validation"
            )
        if not _broker_has_r246_terminal_win_validation(broker_tree, broker_source):
            raise R228StageError(
                "r238 broker lacks r246 deterministic terminal-win validation"
            )
        if not _queue_has_r246_terminal_win_proof(queue_tree):
            raise R228StageError(
                "contained child queue lacks r246 deterministic terminal-win proof authority"
            )
        if not _main_has_pre_search_cuda_runtime_observation(
            wrapper_tree, wrapper_source
        ):
            raise R228StageError(
                "main.py lacks the required pre-search CUDA runtime observation"
            )
        if not _broker_has_pre_search_cuda_runtime_observation(
            broker_tree, broker_source
        ):
            raise R228StageError(
                "r238 broker lacks the required pre-search CUDA runtime observation"
            )
        if not _runtime_has_r242_actor_change_boundary(runtime_tree, runtime_source):
            raise R228StageError(
                "contained child runtime lacks the r242 value-only actor-change boundary"
            )
        if not _runtime_has_r244_handle_scoped_search_identity(
            runtime_tree, runtime_source
        ):
            raise R228StageError(
                "contained child runtime lacks r244 handle-scoped SearchId identity"
            )
        if not _queue_has_r240_deterministic_continuation(queue_tree):
            raise R228StageError(
                "contained child queue lacks the r240 two-lane deterministic continuation guards"
            )
        if not _queue_has_r244_handle_scoped_search_identity(queue_tree):
            raise R228StageError(
                "contained child queue lacks r244 handle-scoped SearchId identity"
            )
    if DECISION_PREFIX not in runtime_source:
        raise R228StageError("contained child runtime lacks its branching-decision marker")
    functions = (
        node
        for node in ast.walk(runtime_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    if not any(_function_returns_async_selected_action(node) for node in functions):
        raise R228StageError("contained child runtime does not return a run_decision selected action")
    if BROKER_MODULE not in wrapper_source:
        raise R228StageError(
            "main.py does not delegate branching actions through r228_kaggle_broker"
        )
    expected_parent_markers = {
        "DECISION_PREFIX": DECISION_PREFIX,
        "FULL_GAMEPLAY_SUCCESS_PREFIX": FULL_GAMEPLAY_SUCCESS_PREFIX,
        "HARD_FAILURE_PREFIX": HARD_FAILURE_PREFIX,
    }
    if not all(
        _literal_module_constant(wrapper_tree, name, value)
        for name, value in expected_parent_markers.items()
    ):
        raise R228StageError("main.py does not bind the r238 two-lane markers")
    if DEGRADED_FALLBACK_MARKER not in wrapper_source:
        raise R228StageError("main.py does not declare the contained degraded marker")
    if "COMPLETE_ACTION_CAP" not in wrapper_source:
        raise R228StageError("main.py does not bind the complete action cap")
    if not _main_has_explicit_broker_settings(wrapper_tree):
        raise R228StageError("main.py does not bind bounded broker settings")
    if not _main_has_r240_scheduler_constants(wrapper_tree):
        raise R228StageError("main.py lacks the r240 four-second hybrid scheduler")
    if queue_tree is not None and not _main_has_r242_hybrid_shortcuts(wrapper_tree):
        raise R228StageError(
            "main.py lacks the r242 high-confidence/direct-continuation shortcuts"
        )
    if not _main_delegates_to_broker_with_direct_fallback(wrapper_tree):
        raise R228StageError(
            "main.py does not precompute and retain direct fallback authority for broker"
        )
    if not _broker_has_required_containment(broker_tree, broker_source):
        raise R228StageError("r238 broker does not provide bounded exact-child containment")
    if not _broker_has_r240_timeout_defaults(broker_tree):
        raise R228StageError("r238 broker does not bind the r240 four/two-second defaults")


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def write_deterministic_tar(source: Path, output: Path) -> None:
    """Write reproducible gzip/tar bytes without host timestamps or ownership."""

    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in _iter_files(source):
            info = tarfile.TarInfo(name=f"./{path.relative_to(source).as_posix()}")
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def stage_bundle(
    *,
    r195_bundle: Path,
    canonical_libcg_wheel: Path,
    output_dir: Path,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    """Build a deterministic archive and receipt.  This function never submits."""

    r195_bundle = r195_bundle.expanduser().resolve()
    canonical_libcg_wheel = canonical_libcg_wheel.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    if not r195_bundle.is_file():
        raise R228StageError(f"r195 input archive is missing: {r195_bundle}")
    if sha256_file(r195_bundle) != R195_BUNDLE_SHA256:
        raise R228StageError("input archive is not the exact frozen r195 bundle")
    canonical_native_payloads = verify_canonical_libcg_wheel(canonical_libcg_wheel)

    sources = {destination: source_root / relative for destination, relative in SOURCE_MEMBERS.items()}
    for source in sources.values():
        if not source.is_file() or source.is_symlink():
            raise R228StageError(f"required r238 source is unavailable: {source}")
    validate_async_action_authority(
        sources["main.py"],
        sources["poke_bot/r228_kaggle_async_runtime.py"],
        sources["poke_bot/r228_kaggle_broker.py"],
        sources["poke_bot/r228_async_shared_tree_queue.py"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_FILENAME
    receipt_path = output_dir / RECEIPT_FILENAME
    if archive_path.exists() or receipt_path.exists():
        raise R228StageError("r238 output identity already exists; refusing overwrite")

    with tempfile.TemporaryDirectory(prefix="r238-stage-", dir=output_dir.parent) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "stage"
        stage.mkdir()
        safe_extract_archive(r195_bundle, stage)
        frozen_input = verify_r195_stage(stage)
        frozen_cg_wrapper_members = _frozen_cg_wrapper_members(stage)

        direct_main = stage / "r195_direct_main.py"
        (stage / "main.py").replace(direct_main)
        source_sha = {
            destination: _copy_source(source, stage / destination)
            for destination, source in sources.items()
        }
        r225_typed_contract = r225_typed_contract_identity(
            stage / R225_TYPED_CONTRACT_PATH
        )
        if source_sha[R225_TYPED_CONTRACT_PATH] != r225_typed_contract["sha256"]:
            raise R228StageError(
                "staged canonical r225 typed contract digest does not match its member"
            )
        overlay_canonical_libcg_members(stage, canonical_native_payloads)
        canonical_native_members = verify_canonical_libcg_stage(stage)
        verify_frozen_cg_wrapper_retained(stage, frozen_cg_wrapper_members)
        for member, expected in {
            "model.pt": R195_MODEL_SHA256,
            "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
            "search_config.json": R195_SEARCH_CONFIG_SHA256,
        }.items():
            if sha256_file(stage / member) != expected:
                raise R228StageError(f"r238 overlay modified frozen {member}")

        canonical_contract = canonical_libcg_contract()
        exact_frozen_base = exact_frozen_base_contract()
        native_member_identity = {
            relative: {
                "platform": member["platform"],
                "wheel_member": member["wheel_member"],
                "sha256": member["sha256"],
                "size_bytes": member["size_bytes"],
            }
            for relative, member in _canonical_native_members_by_path().items()
        }
        historical_source = {
            "stage_script": "scripts/stage_r228_async_eight_worker_kaggle_viability.py",
            "runtime_source_members": [
                "poke_bot/r228_kaggle_async_runtime.py",
                "poke_bot/r228_async_shared_tree_queue.py",
                "poke_bot/r225_stock_native_lane.py",
            ],
            "contained_child_decision_marker": DECISION_PREFIX,
        }

        manifest = {
            "schema": SCHEMA,
            "role": "isolated_r238_two_lane_bounded_mcts_fallback_diagnostic",
            "historical_source": historical_source,
            "input_r195_bundle_sha256": R195_BUNDLE_SHA256,
            "required_label": REQUIRED_LABEL,
            "branching_decision_marker": DECISION_PREFIX,
            "full_gameplay_success_marker": FULL_GAMEPLAY_SUCCESS_PREFIX,
            "hard_failure_marker": HARD_FAILURE_PREFIX,
            "async_selected_action_authority": "receipt.selected_action",
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "broker_contract": broker_contract(),
            "r240_hybrid_scheduler": r240_hybrid_scheduler_contract(),
            "deterministic_continuation": deterministic_continuation_contract(),
            "r240_required_preflight_receipts": list(
                R240_REQUIRED_REGRESSION_RECEIPTS
            ),
            "owner_proven_deterministic_terminal_win_this_turn_revision": (
                PROVEN_TERMINAL_WIN_REVISION
            ),
            "r246_proven_deterministic_terminal_win_this_turn": (
                r246_proven_deterministic_terminal_win_contract()
            ),
            "r246_required_preflight_receipts": [
                PROVEN_TERMINAL_WIN_REGRESSION_RECEIPT
            ],
            "r244_handle_scoped_search_identity": (
                r244_handle_scoped_search_identity_contract()
            ),
            "r244_required_preflight_receipts": [
                R244_HANDLE_SCOPED_SEARCH_IDENTITY_REGRESSION_RECEIPT
            ],
            "r225_typed_contract": r225_typed_contract,
            "degraded_fallback_marker": DEGRADED_FALLBACK_MARKER,
            "subprocess_containment_identity": SUBPROCESS_CONTAINMENT_IDENTITY,
            "failed_kaggle_validation_evidence": FAILED_KAGGLE_VALIDATION_EVIDENCE,
            "lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "required_search_lifecycle_counts": {
                "search_begin_calls": REQUIRED_SEARCH_BEGIN_COUNT,
                "search_end_calls": REQUIRED_SEARCH_END_COUNT,
                "search_release_calls": REQUIRED_SEARCH_RELEASE_COUNT,
            },
            "phase1_kaggle_resource_bounds": phase1_kaggle_resource_bounds(),
            "canonical_libcg_contract": canonical_contract,
            "canonical_native_members": native_member_identity,
            "canonical_native_member_sha256": canonical_native_members,
            "exact_frozen_base": exact_frozen_base,
            "r225_package_preflight": canonical_contract["r225_package_preflight"],
            "frozen_r195_input_members": frozen_input,
            "frozen_r195_cg_wrapper_members": frozen_cg_wrapper_members,
            "frozen_members": {
                name: digest
                for name, digest in frozen_input.items()
                if name != "cg/libcg.so"
            },
            "preserved_members": {
                "model.pt": R195_MODEL_SHA256,
                "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
                "search_config.json": R195_SEARCH_CONFIG_SHA256,
                "frozen_r195_python_cg_wrapper_members": frozen_cg_wrapper_members,
            },
            "input_r195_replaced_native_member": {
                "path": "cg/libcg.so",
                "sha256": STOCK_LIBCG_SHA256,
                "size_bytes": STOCK_LIBCG_BYTES,
                "replaced_by_canonical_r236_linux_x86_64": True,
            },
            "direct_entrypoint": {
                "path": "r195_direct_main.py",
                "sha256": sha256_file(direct_main),
            },
            "historical_source_members": source_sha,
            "entrypoint_sha256": sha256_file(stage / "main.py"),
            "kaggle_client_or_queue_imported_by_stager": False,
            "stager_never_submits": True,
        }
        manifest_path = stage / MANIFEST_FILENAME
        manifest_path.write_bytes(canonical_json(manifest))
        os.chmod(manifest_path, 0o644)

        temporary_archive = temporary_root / ARCHIVE_FILENAME
        write_deterministic_tar(stage, temporary_archive)
        archive_size_bytes = temporary_archive.stat().st_size
        if archive_size_bytes > PHASE1_ARCHIVE_MAX_BYTES:
            raise R228StageError("staged archive exceeds the Phase 1 package-size limit")
        archive_sha = sha256_file(temporary_archive)
        receipt = {
            "schema": SCHEMA,
            "status": "staged_not_submitted",
            "archive_filename": ARCHIVE_FILENAME,
            "archive_sha256": archive_sha,
            "member_manifest_filename": MANIFEST_FILENAME,
            "member_manifest_sha256": sha256_file(manifest_path),
            "input_r195_bundle_sha256": R195_BUNDLE_SHA256,
            "required_label": REQUIRED_LABEL,
            "role": manifest["role"],
            "historical_source": historical_source,
            "entrypoint_sha256": manifest["entrypoint_sha256"],
            "direct_entrypoint_sha256": manifest["direct_entrypoint"]["sha256"],
            "historical_source_members": source_sha,
            "preserved_search_config_sha256": R195_SEARCH_CONFIG_SHA256,
            "input_r195_replaced_native_member": manifest[
                "input_r195_replaced_native_member"
            ],
            "frozen_r195_cg_wrapper_members": frozen_cg_wrapper_members,
            "canonical_libcg_contract": canonical_contract,
            "canonical_native_members": native_member_identity,
            "canonical_native_member_sha256": canonical_native_members,
            "exact_frozen_base": exact_frozen_base,
            "r225_package_preflight": canonical_contract["r225_package_preflight"],
            "async_selected_action_authority": "receipt.selected_action",
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "broker_contract": broker_contract(),
            "r240_hybrid_scheduler": manifest["r240_hybrid_scheduler"],
            "deterministic_continuation": manifest["deterministic_continuation"],
            "r240_required_preflight_receipts": manifest[
                "r240_required_preflight_receipts"
            ],
            "owner_proven_deterministic_terminal_win_this_turn_revision": manifest[
                "owner_proven_deterministic_terminal_win_this_turn_revision"
            ],
            "r246_proven_deterministic_terminal_win_this_turn": manifest[
                "r246_proven_deterministic_terminal_win_this_turn"
            ],
            "r246_required_preflight_receipts": manifest[
                "r246_required_preflight_receipts"
            ],
            "r244_handle_scoped_search_identity": manifest[
                "r244_handle_scoped_search_identity"
            ],
            "r244_required_preflight_receipts": manifest[
                "r244_required_preflight_receipts"
            ],
            "r225_typed_contract": manifest["r225_typed_contract"],
            "degraded_fallback_marker": DEGRADED_FALLBACK_MARKER,
            "subprocess_containment_identity": SUBPROCESS_CONTAINMENT_IDENTITY,
            "failed_kaggle_validation_evidence": FAILED_KAGGLE_VALIDATION_EVIDENCE,
            "branching_decision_marker": DECISION_PREFIX,
            "full_gameplay_success_marker": FULL_GAMEPLAY_SUCCESS_PREFIX,
            "hard_failure_marker": HARD_FAILURE_PREFIX,
            "lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "required_search_lifecycle_counts": manifest[
                "required_search_lifecycle_counts"
            ],
            "phase1_kaggle_resource_bounds": manifest[
                "phase1_kaggle_resource_bounds"
            ],
            "archive_size_bytes": archive_size_bytes,
            "kaggle_api_called": False,
            "kaggle_queue_used": False,
            "kaggle_upload_used": False,
            "kaggle_submission_created": False,
        }
        temporary_receipt = temporary_root / RECEIPT_FILENAME
        temporary_receipt.write_bytes(canonical_json(receipt))
        os.replace(temporary_archive, archive_path)
        os.replace(temporary_receipt, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r195-bundle", type=Path, required=True)
    parser.add_argument("--canonical-libcg-wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            stage_bundle(
                r195_bundle=args.r195_bundle,
                canonical_libcg_wheel=args.canonical_libcg_wheel,
                output_dir=args.output_dir,
                source_root=args.source_root,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
