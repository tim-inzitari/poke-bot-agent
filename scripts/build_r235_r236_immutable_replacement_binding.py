#!/usr/bin/env python3
"""Build an offline, immutable R235/R236 replacement-package binding.

The builder is deliberately verification-only: it neither imports a Kaggle
client nor starts a simulator, GPU, child process, service, queue, upload, or
BO1000 workload.  It binds one exact local archive, its byte-identical member
manifest, the canonical r225/r246 and r236 sources, and every required
write-once local gate into a no-clobber JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "poke_bot.r235_r236_immutable_replacement_binding/v1"
PREFLIGHT_RECEIPT_SCHEMA = "poke_bot.r235_r236_local_preflight_receipt/v1"
GO_FIRST_ATTESTATION_SCHEMA = "poke_bot.submission_turn_order_attestation/v1"
R244_HANDLE_SCOPED_SEARCH_ID_RECEIPT_SCHEMA = (
    "poke_bot.r244_handle_scoped_search_id_identity_regression_receipt/v1"
)
GO_FIRST_VERIFIER_KIND = "r235_digest_bound_go_first_verifier"
GO_FIRST_VERIFIED_CASES = [
    "integer_enum",
    "live_engine_prompt",
    "string_enum_reversed_options",
]
GO_FIRST_CASE_RESULTS: dict[str, dict[str, list[int]]] = {
    "integer_enum": {"selected_action": [0]},
    "live_engine_prompt": {"selected_action": [1]},
    "string_enum_reversed_options": {"selected_action": [1]},
}
R225_SCHEMA = "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
R236_SCHEMA = "poke_bot.canonical_libcg_r236/v1"
R238_OWNER_DECISION_REVISION = 238
R242_OWNER_DECISION_REVISION = 242
R244_OWNER_DECISION_REVISION = 244
R246_OWNER_DECISION_REVISION = 246
R238_PACKAGE_MANIFEST_SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
R238_PACKAGE_MANIFEST_ROLE = "isolated_r238_two_lane_bounded_mcts_fallback_diagnostic"
CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH = (
    "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
)
CANONICAL_R225_R240_TYPED_CONTRACT_PATH = (
    ROOT / CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH
)
# Frozen by the R246 owner decision.  Any byte drift in the historical r225
# path is rejected before the archive or a receipt can be accepted.
CANONICAL_R225_R240_TYPED_CONTRACT_SHA256 = (
    "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
)
CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH = "state/canonical-libcg-r236.json"
CANONICAL_R236_TYPED_CONTRACT_PATH = ROOT / CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH
CANONICAL_R236_TYPED_CONTRACT_SHA256 = (
    "sha256:d75ff752808ead08f3ae20f7f2f8a034c9e6163109188a46d3b877bf1910ae2d"
)

R235_LABEL = "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"
COMPETITION = "pokemon-tcg-ai-battle"
COMPLETE_ACTION_CAP = 65_536
SIMULATOR_SEARCH_LANE_COUNT = 2
PHASE1_RESOURCES: dict[str, object] = {
    "hdd_space_gib": 11.8,
    "ram_gib": 12.2,
    "vcpus": 2,
    "submission_archive_limit_mib": 197.7,
}
PHASE1_ARCHIVE_MAX_BYTES = int(
    float(PHASE1_RESOURCES["submission_archive_limit_mib"]) * 1024 * 1024
)
PHASE1_MANIFEST_RESOURCE_BOUNDS: dict[str, object] = {
    "vcpus": 2,
    "ram_gib": 12.2,
    "hdd_gib": 11.8,
    "archive_mib": 197.7,
    # R238's CPU/RAM/disk envelope does not establish CUDA visibility.  The
    # package must record an actual runtime observation before search instead
    # of treating this static resource report as a GPU claim.
    "gpu_environment_inferred_from_resource_envelope": False,
    "runtime_cuda_observation_required_before_search": True,
    "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
}
PHASE1_RECEIPT_ENVIRONMENT: dict[str, object] = dict(PHASE1_RESOURCES)
R242_HYBRID_SCHEDULER: dict[str, object] = {
    "high_confidence_threshold": 0.80,
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
R244_HANDLE_SCOPED_SEARCH_IDENTITY: dict[str, object] = {
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
    "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
}
R246_TERMINAL_WIN_STOP_REASON = "proven_deterministic_terminal_win_this_turn"
R246_TERMINAL_WIN_PROOF_KIND = "exact_deterministic_simulator_terminal_win_this_turn"
R246_CLEANUP_COMPLETE_FIELD = (
    "all_owned_lane_resources_reservations_and_child_cleanup_complete"
)
R246_LEGACY_CLEANUP_COMPLETED_FIELD = (
    "all_owned_lane_resources_reservations_and_child_cleanup_completed"
)
R246_TERMINAL_WIN_PROOF_FIELDS = (
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
R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN: dict[str, object] = {
    "scope": "ambiguous_two_lane_mcts_for_new_r235_replacement_package_only",
    "owner_decision_revision": R246_OWNER_DECISION_REVISION,
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
R246_REPLACEMENT_TERMINAL_WIN_THIS_TURN: dict[str, object] = {
    key: value
    for key, value in R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN.items()
    if key != "scope"
}
DETERMINISTIC_CONTINUATION: dict[str, object] = {
    "max_depth": 8,
    "exact_observation_fingerprint_required": True,
    "both_lanes_same_fingerprint_and_backed_action_required": True,
    "same_root_actor_required": True,
    "chance_or_boundary_forbidden": True,
    "no_new_search_on_valid_match": True,
    "mismatch_clears_entire_plan": True,
}
# The package manifest carries the canonical r225/r242 scheduler object. Gate
# receipts use the normalized maps above, so independent preflight producers
# can bind the same semantic knobs without copying the package's full schema.
R242_MANIFEST_SCHEDULER: dict[str, object] = {
    "scope": "new_r235_replacement_package_only",
    "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_DECISION_REVISION,
    "selected_factorized_stage_probability_threshold": 0.80,
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
        "selected_factorized_stage_probability_threshold": 0.80,
        "all_selected_factorized_stages_meet_threshold": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "history_only_existing_child_journal_count_range": [0, 1],
        "degraded": False,
    },
    "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
    "two_lane_mcts_topology_backup_and_stop_contract_applies_only_when_confidence_routes_to_ambiguous_mcts": True,
    "ambiguous_mcts_exact_simulator_search_lane_count": 2,
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
        R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN
    ),
    "stop_reason_fields": [
        "high_confidence_frozen_direct",
        "deterministic_continuation_plan",
        R246_TERMINAL_WIN_STOP_REASON,
        "adaptive_early_stop",
        "hard_completed_backup_stop",
        "child_search_hard_deadline",
        "parent_action_hard_deadline",
        "zero_backup_precomputed_direct_fallback",
        "contained_child_fault",
    ],
    "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
    "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
    "zero_completed_backups_returns_only_the_precomputed_legal_direct_action_under_existing_clean_deadline_or_containment_rules": True,
    "partial_lane_serial_or_unbounded_search_authority_allowed": False,
    "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
}
# The serialized package key remains ``r240_hybrid_scheduler`` because R242
# amends that canonical object in place.  Keep these aliases private-source
# compatible for focused callers while every validated value is R242-specific.
R240_HYBRID_SCHEDULER = R242_HYBRID_SCHEDULER
R240_MANIFEST_SCHEDULER = R242_MANIFEST_SCHEDULER

MANIFEST_DETERMINISTIC_CONTINUATION: dict[str, object] = {
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
R195_BUNDLE_SHA256 = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
R195_CHECKPOINT_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"

OFFICIAL_WHEEL_SHA256 = "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
OFFICIAL_WHEEL_FILENAME = "kaggle_environments-1.32.6-py3-none-any.whl"
OFFICIAL_PACKAGE_VERSION = "1.32.6"

# This set remains literal.  Hashing the typed r236 source does not permit a
# local edit to redefine the official Kaggle Environments 1.32.6 bytes.
OFFICIAL_LIBCG_MEMBERS: dict[str, dict[str, object]] = {
    "linux_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
        "format": "ELF 64-bit LSB shared object x86-64",
    },
    "linux_aarch64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
        "format": "ELF 64-bit LSB shared object ARM aarch64",
    },
    "macos_arm64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "package_relative_path": "cg/libcg.dylib",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
        "format": "Mach-O 64-bit dynamically linked shared library arm64",
    },
    "windows_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/cg.dll",
        "package_relative_path": "cg/cg.dll",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
        "format": "PE32+ x86-64 DLL",
    },
}
REQUIRED_NATIVE_EXPORTS = (
    "AgentStart",
    "BattleStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
)

GATE_NAMES = {
    "focused_fault": "focused_native_child_fault_suite_receipt",
    "saved_episode": (
        "saved_episode_91766923_seat_0_step_58_two_choice_callback_"
        "legal_hard_deadline_regression_receipt"
    ),
    "full_game": "exact_repaired_package_full_local_game_receipt",
    "resource": "resource_memory_startup_and_throughput_preflight_receipt",
    "phase1_resource": "phase1_submission_resource_and_archive_limit_receipt",
    "two_lane_topology": (
        "two_lane_shared_tree_topology_and_receipt_schema_regression_receipt"
    ),
    "handle_scoped_search_id": (
        "official_libcg_handle_scoped_search_id_identity_regression_receipt"
    ),
    "high_confidence": (
        "high_confidence_direct_and_adaptive_bounded_mcts_regression_receipt"
    ),
    "terminal_win": "proven_deterministic_terminal_win_this_turn_regression_receipt",
    "deterministic_continuation": "deterministic_continuation_regression_receipt",
    "go_first": "go_first_receipt",
}

R242_ACTOR_BOUNDARY_COUNTER_FIELDS = (
    "declared_opponent_actor_leaf_count",
    "value_evaluated_opponent_actor_leaf_count",
    "expanded_legal_action_count",
    "expanded_child_count",
    "search_steps_beyond_boundary",
    "opponent_action_selected_or_planned_count",
    "opponent_action_cached_count",
)


class R235R236BindingError(RuntimeError):
    """A proposed replacement binding is incomplete, drifted, or unsafe."""


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise R235R236BindingError(f"{label} must not be a symlink: {raw_path}")
    resolved = raw_path.resolve()
    if not resolved.is_file():
        raise R235R236BindingError(f"{label} must be a regular non-symlink file: {resolved}")
    return resolved


def _canonical_typed_file(path: Path, *, canonical_path: Path, label: str) -> Path:
    observed = _regular_file(path, label=label)
    canonical = _regular_file(canonical_path, label=f"canonical {label}")
    if observed != canonical:
        raise R235R236BindingError(
            f"{label} must resolve to the canonical typed source: {canonical}"
        )
    return observed


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235R236BindingError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise R235R236BindingError(f"{label} must contain a JSON object")
    return payload


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R235R236BindingError(f"{label} must be an object")
    return value


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if value != expected:
        raise R235R236BindingError(f"{label} is not the required value")


def _require_expected_fields(
    observed: Mapping[str, Any], expected: Mapping[str, object], *, label: str
) -> None:
    for field, value in expected.items():
        _require_exact(observed.get(field), value, label=f"{label}.{field}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_member_name(raw_name: str, *, label: str) -> str:
    name = raw_name.removeprefix("./").strip("/")
    candidate = PurePosixPath(name)
    if not name or candidate.is_absolute() or ".." in candidate.parts:
        raise R235R236BindingError(f"{label} has an unsafe archive member path")
    return name


def _iter_archive_regular_members(
    archive_path: Path,
) -> Iterable[tuple[str, tarfile.TarInfo, bytes]]:
    """Yield safe unique regular members and reject every other member type."""

    try:
        archive = tarfile.open(archive_path, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise R235R236BindingError("candidate archive is not a readable tar archive") from exc
    with archive:
        seen: set[str] = set()
        for member in archive.getmembers():
            name = _safe_member_name(member.name, label="candidate archive")
            if name in seen:
                raise R235R236BindingError("candidate archive has duplicate member paths")
            seen.add(name)
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise R235R236BindingError(
                    "candidate archive contains a non-regular or linked member"
                )
            source = archive.extractfile(member)
            if source is None:
                raise R235R236BindingError("candidate archive member cannot be read")
            try:
                body = source.read()
            finally:
                source.close()
            if len(body) != member.size:
                raise R235R236BindingError("candidate archive member has a truncated body")
            yield name, member, body


def inspect_archive(
    archive_path: Path,
    *,
    member_manifest_member: str,
    entrypoint_member: str,
) -> tuple[dict[str, dict[str, object]], bytes, str]:
    """Return the full archive map plus the selected manifest and entrypoint."""

    manifest_member = _safe_member_name(
        member_manifest_member, label="member manifest selector"
    )
    entrypoint = _safe_member_name(entrypoint_member, label="entrypoint selector")
    members: dict[str, dict[str, object]] = {}
    manifest_bytes: bytes | None = None
    entrypoint_sha: str | None = None
    for name, member, body in _iter_archive_regular_members(archive_path):
        digest = sha256_bytes(body)
        members[name] = {
            "sha256": digest,
            "size_bytes": len(body),
            "mode": member.mode & 0o777,
        }
        if name == manifest_member:
            manifest_bytes = body
        if name == entrypoint:
            entrypoint_sha = digest
    if manifest_bytes is None:
        raise R235R236BindingError("candidate archive lacks the selected member manifest")
    if entrypoint_sha is None:
        raise R235R236BindingError("candidate archive lacks the selected entrypoint")
    return members, manifest_bytes, entrypoint_sha


def _validate_r236_contract(contract: Mapping[str, Any]) -> dict[str, object]:
    _require_exact(contract.get("schema"), R236_SCHEMA, label="r236 schema")
    _require_exact(contract.get("owner_decision_revision"), 236, label="r236 revision")
    upstream = _require_mapping(contract.get("upstream_provenance"), label="r236 upstream")
    _require_expected_fields(
        upstream,
        {
            "package_version": OFFICIAL_PACKAGE_VERSION,
            "wheel_filename": OFFICIAL_WHEEL_FILENAME,
            "wheel_sha256": OFFICIAL_WHEEL_SHA256,
        },
        label="r236 upstream",
    )
    libraries = _require_mapping(
        contract.get("canonical_native_libraries"), label="r236 native libraries"
    )
    if set(libraries) != set(OFFICIAL_LIBCG_MEMBERS):
        raise R235R236BindingError("r236 must bind exactly the four official library members")
    for platform, expected in OFFICIAL_LIBCG_MEMBERS.items():
        observed = _require_mapping(libraries.get(platform), label=f"r236 {platform}")
        if dict(observed) != expected:
            raise R235R236BindingError(
                f"r236 {platform} identity does not match official 1.32.6"
            )
    _require_exact(
        tuple(contract.get("required_native_exports", ())),
        REQUIRED_NATIVE_EXPORTS,
        label="r236 required native exports",
    )
    scope = _require_mapping(contract.get("scope"), label="r236 scope")
    _require_expected_fields(
        scope,
        {
            "r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary": True,
            "mixed_old_and_new_library_sets_allowed": False,
        },
        label="r236 scope",
    )
    authority = _require_mapping(contract.get("authority"), label="r236 authority")
    for field in (
        "managed_training_or_evaluation_service_restart_authorized",
        "training_or_gradient_updates_authorized",
        "additional_kaggle_upload_retry_copy_or_queue_authorized",
    ):
        _require_exact(authority.get(field), False, label=f"r236 authority.{field}")
    return {
        "package_version": OFFICIAL_PACKAGE_VERSION,
        "wheel_filename": OFFICIAL_WHEEL_FILENAME,
        "official_wheel_sha256": OFFICIAL_WHEEL_SHA256,
        "required_native_exports": list(REQUIRED_NATIVE_EXPORTS),
        "members": OFFICIAL_LIBCG_MEMBERS,
    }


def _validate_embedded_manifest_r236(contract: Mapping[str, Any]) -> None:
    """Validate the stager's r236 projection, which has no authority section."""

    _require_exact(contract.get("schema"), R236_SCHEMA, label="manifest r236 schema")
    _require_exact(
        contract.get("typed_source"),
        CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH,
        label="manifest r236 typed source",
    )
    _require_exact(
        contract.get("owner_decision_revision"), 236, label="manifest r236 revision"
    )
    upstream = _require_mapping(contract.get("upstream_provenance"), label="manifest r236 upstream")
    _require_expected_fields(
        upstream,
        {
            "package_version": OFFICIAL_PACKAGE_VERSION,
            "wheel_filename": OFFICIAL_WHEEL_FILENAME,
            "wheel_sha256": OFFICIAL_WHEEL_SHA256,
        },
        label="manifest r236 upstream",
    )
    libraries = _require_mapping(
        contract.get("canonical_native_libraries"), label="manifest r236 native libraries"
    )
    if set(libraries) != set(OFFICIAL_LIBCG_MEMBERS):
        raise R235R236BindingError("manifest r236 contract lacks an official member")
    for platform, expected in OFFICIAL_LIBCG_MEMBERS.items():
        observed = _require_mapping(libraries.get(platform), label=f"manifest r236 {platform}")
        if dict(observed) != expected:
            raise R235R236BindingError("manifest r236 member identity drifted")
    _require_exact(
        tuple(contract.get("required_native_exports", ())),
        REQUIRED_NATIVE_EXPORTS,
        label="manifest r236 required exports",
    )
    scope = _require_mapping(contract.get("scope"), label="manifest r236 scope")
    _require_expected_fields(
        scope,
        {
            "r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary": True,
            "all_four_platform_native_members_must_be_bound": True,
            "mixed_old_and_new_library_sets_allowed": False,
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
        },
        label="manifest r236 scope",
    )
    preflight = _require_mapping(
        contract.get("r225_package_preflight"), label="manifest r236 package preflight"
    )
    _require_expected_fields(
        preflight,
        {
            "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
            "all_four_canonical_native_members_checksum_and_size_verified": True,
            "old_or_mixed_native_members_rejected": True,
            "required_native_exports": list(REQUIRED_NATIVE_EXPORTS),
        },
        label="manifest r236 package preflight",
    )


def _validate_r225_contract(
    contract: Mapping[str, Any], *, canonical_contract_sha256: str
) -> dict[str, object]:
    _require_exact(contract.get("schema"), R225_SCHEMA, label="r225 schema")
    _require_exact(
        contract.get("owner_decision_revision"),
        R246_OWNER_DECISION_REVISION,
        label="r225/r246 owner decision revision",
    )
    _require_expected_fields(
        contract,
        {
            "owner_kaggle_replacement_diagnostic_revision": 235,
            "owner_canonical_libcg_revision": 236,
            "owner_phase1_submission_resources_and_two_lane_revision": R238_OWNER_DECISION_REVISION,
            "owner_hybrid_confidence_bounded_mcts_revision": 240,
            "owner_high_confidence_frozen_direct_threshold_revision": R242_OWNER_DECISION_REVISION,
            "owner_handle_scoped_search_id_revision": R244_OWNER_DECISION_REVISION,
            "owner_proven_deterministic_terminal_win_this_turn_revision": R246_OWNER_DECISION_REVISION,
        },
        label="r225/r246 revisions",
    )
    relationship = _require_mapping(
        contract.get("relationship_to_existing_work"), label="r225/r246 relationship"
    )
    _require_exact(
        relationship.get(
            "r240_supersedes_only_the_new_r235_replacement_package_decision_scheduling_and_does_not_modify_r229_or_bo1000"
        ),
        True,
        label="r225/r246 BO1000 boundary",
    )
    _require_expected_fields(
        relationship,
        {
            "r242_supersedes_only_the_r240_high_confidence_frozen_direct_threshold_for_the_new_r235_replacement_package": True,
            "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible": True,
            "r242_does_not_modify_r234_r236_r238_r235_continuation_or_r229_bo1000": True,
            "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229": True,
            "r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries": True,
            "r246_supersedes_only_ambiguous_r235_mcts_root_selection_after_a_valid_deterministic_terminal_win_this_turn_proof": True,
            "r246_does_not_change_r242_high_confidence_direct_before_child_or_any_r229_bo1000_lifecycle": True,
        },
        label="r225/r246 relationship",
    )
    canonical = _require_mapping(
        contract.get("canonical_libcg_revision"), label="r225 canonical-libcg binding"
    )
    linux = OFFICIAL_LIBCG_MEMBERS["linux_x86_64"]
    _require_expected_fields(
        canonical,
        {
            "typed_source": CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH,
            "official_wheel_sha256": OFFICIAL_WHEEL_SHA256,
            "linux_x86_64_sha256": linux["sha256"],
            "linux_x86_64_size_bytes": linux["size_bytes"],
            "new_r235_package_must_overlay_the_exact_official_linux_binary": True,
            "new_package_may_retain_the_old_frozen_r195_libcg_member": False,
            "saved_episode_and_full_game_gates_must_be_reissued_on_the_new_binary": True,
        },
        label="r225 canonical-libcg",
    )
    base = _require_mapping(contract.get("exact_frozen_base"), label="r225 frozen base")
    _require_expected_fields(
        base,
        {
            "r195_bundle_sha256": R195_BUNDLE_SHA256,
            "r195_checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "r195_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "stock_libcg_sha256": linux["sha256"],
            "stock_libcg_size_bytes": linux["size_bytes"],
        },
        label="r225 frozen base",
    )
    action_space = _require_mapping(
        contract.get("complete_ordered_action_space_contract"), label="r225 action-space contract"
    )
    _require_expected_fields(
        action_space,
        {
            "complete_ordered_legal_action_ceiling": COMPLETE_ACTION_CAP,
            "ceiling_applies_at_root_and_every_private_leaf": True,
            "sampling_pruning_or_reinterpretation_of_legal_choices_allowed": False,
            "root_over_cap_behavior": "hard_fail_nonzero",
            "private_leaf_over_cap_behavior": (
                "contained_degraded_parent_direct_fallback_only_after_validated_"
                "precomputed_root_direct_action_and_exact_child_reap"
            ),
        },
        label="r225 action-space",
    )
    environment = _require_mapping(
        contract.get("phase1_submission_environment"), label="r225 Phase-1 environment"
    )
    _require_expected_fields(
        environment,
        {
            "owner_decision_revision": R238_OWNER_DECISION_REVISION,
            **PHASE1_RESOURCES,
            "resource_probe_and_archive_size_receipt_required": True,
            "resource_mismatch_or_archive_over_limit_behavior": "hard_fail_closed_and_do_not_upload",
            "gpu_or_os_python_environment_is_not_inferred_from_the_reported_submission_resource_values": True,
        },
        label="r225 Phase-1 environment",
    )
    if "gpu_available" in environment:
        raise R235R236BindingError(
            "r225 Phase-1 environment must not infer a stale GPU availability value"
        )
    local_preflight = _require_mapping(contract.get("local_preflight"), label="r225 preflight")
    _require_expected_fields(
        local_preflight,
        {
            "parent_precomputes_and_validates_exact_frozen_r195_direct_action_on_complete_ordered_root_legal_set": True,
            "parent_records_direct_action_legal_fingerprint_before_mcts_child_start": True,
            "mcts_child_owns_its_own_checksum_identical_frozen_r195_model_one_stock_dso_one_logical_tree_and_exactly_two_arenas_when_started_for_ambiguous_mcts": True,
            "required_simulator_search_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "required_internal_agent_start_simulator_search_arena_count_per_child": SIMULATOR_SEARCH_LANE_COUNT,
            "required_search_begin_call_count_per_ambiguous_mcts_decision": SIMULATOR_SEARCH_LANE_COUNT,
            "required_distinct_internal_agent_start_handle_identity_count": SIMULATOR_SEARCH_LANE_COUNT,
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "required_distinct_handle_identity_first_search_id_composite_state_count": SIMULATOR_SEARCH_LANE_COUNT,
            "per_lane_handle_scoped_search_id_chains_required": True,
            "one_lane_baseline_or_ratio_comparison_allowed": False,
        },
        label="r225 preflight",
    )
    r238_receipt = _require_mapping(
        local_preflight.get("r238_two_lane_receipt_contract"), label="r238 two-lane receipt contract"
    )
    exact_counts = _require_mapping(
        r238_receipt.get("normal_mcts_decision_exact_counts"), label="r238 exact counts"
    )
    _require_expected_fields(
        exact_counts,
        {
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "distinct_handle_identity_count": 2,
            "distinct_handle_scoped_first_search_id_composite_state_count": 2,
            "search_begin_calls": 2,
            "search_end_calls": 2,
        },
        label="r238 exact counts",
    )
    minimum_counts = _require_mapping(
        r238_receipt.get("normal_mcts_decision_minimum_counts"), label="r238 minimum counts"
    )
    _require_exact(
        minimum_counts.get("search_release_calls"), 2, label="r238 release minimum"
    )
    _require_expected_fields(
        r238_receipt,
        {
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
            "search_id_identity_contract": R244_HANDLE_SCOPED_SEARCH_IDENTITY,
            "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
        },
        label="r238 two-lane receipt contract",
    )
    local_r240_scheduler = _require_mapping(
        local_preflight.get("r240_hybrid_scheduler"), label="r225 r240 scheduler"
    )
    _require_expected_fields(
        local_r240_scheduler,
        {
            "scope": "new_r235_replacement_package_only",
            "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_DECISION_REVISION,
            "selected_factorized_stage_probability_threshold": 0.80,
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
                "selected_factorized_stage_probability_threshold": 0.80,
                "all_selected_factorized_stages_meet_threshold": True,
                "mcts_child_started_for_this_decision": False,
                "mcts_select_call_count": 0,
                "history_only_existing_child_journal_count_range": [0, 1],
                "degraded": False,
            },
            "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
            "two_lane_mcts_topology_backup_and_stop_contract_applies_only_when_confidence_routes_to_ambiguous_mcts": True,
            "ambiguous_mcts_exact_simulator_search_lane_count": 2,
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
            "zero_completed_backups_returns_only_the_precomputed_legal_direct_action_under_existing_clean_deadline_or_containment_rules": True,
            "partial_lane_serial_or_unbounded_search_authority_allowed": False,
            "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
        },
        label="r225 r240 scheduler",
    )
    _require_exact(
        local_r240_scheduler.get("stop_reason_fields"),
        [
            "high_confidence_frozen_direct",
            "deterministic_continuation_plan",
            R246_TERMINAL_WIN_STOP_REASON,
            "adaptive_early_stop",
            "hard_completed_backup_stop",
            "child_search_hard_deadline",
            "parent_action_hard_deadline",
            "zero_backup_precomputed_direct_fallback",
            "contained_child_fault",
        ],
        label="r225 r240 stop reasons",
    )
    _require_exact(
        dict(
            _require_mapping(
                local_r240_scheduler.get(
                    "r246_proven_deterministic_terminal_win_this_turn"
                ),
                label="r225 r246 deterministic terminal-win contract",
            )
        ),
        R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN,
        label="r225 r246 deterministic terminal-win contract",
    )
    _require_expected_fields(
        local_r240_scheduler,
        {
            "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
            "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
        },
        label="r225 r246 terminal-win scheduler fields",
    )
    full_gameplay = _require_mapping(
        local_preflight.get("full_gameplay_shared_tree_contract"),
        label="r225 full-gameplay shared-tree contract",
    )
    _require_expected_fields(
        full_gameplay,
        {
            "r246_proven_deterministic_terminal_win_this_turn_is_an_in_search_early_stop_not_a_r242_direct_bypass": True,
            "r246_valid_terminal_win_proof_may_bypass_only_the_normal_adaptive_stop_thresholds_after_two_lane_initialization": True,
            "r246_valid_terminal_win_proof_requires_at_least_one_exact_terminal_backup_but_not_the_normal_eight_backup_leader_or_both_lane_progress_threshold": True,
        },
        label="r225 r246 full-gameplay contract",
    )
    local_continuation = _require_mapping(
        local_preflight.get("deterministic_continuation"),
        label="r225 deterministic continuation",
    )
    _require_expected_fields(
        local_continuation,
        {
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
        },
        label="r225 deterministic continuation",
    )
    _require_exact(
        local_continuation.get("journal_required_fields"),
        MANIFEST_DETERMINISTIC_CONTINUATION["journal_required_fields"],
        label="r225 deterministic continuation journal fields",
    )
    replacement = _require_mapping(
        contract.get("replacement_kaggle_diagnostic"), label="r225 replacement diagnostic"
    )
    _require_expected_fields(
        replacement,
        {
            "revision": 235,
            "phase1_two_lane_resource_revision": R238_OWNER_DECISION_REVISION,
            "hybrid_confidence_bounded_mcts_revision": 240,
            "handle_scoped_search_id_correction_revision": R244_OWNER_DECISION_REVISION,
            "proven_deterministic_terminal_win_this_turn_revision": R246_OWNER_DECISION_REVISION,
            "replacement_submission_count_limit": 1,
            "replacement_submission_consumed": False,
            "competition": COMPETITION,
            "submission_message_required_literal": R235_LABEL,
            "submission_message_must_be_unique_and_exact": True,
            "complete_ordered_legal_action_ceiling": COMPLETE_ACTION_CAP,
            "all_required_local_gates_and_immutable_binding_must_pass_before_direct_api_upload": True,
            "kaggle_api_call_permitted_now_before_gates": False,
            "kaggle_upload_permitted_now_before_gates": False,
            "queue_or_batch_submission_allowed": False,
            "automatic_retry_allowed": False,
            "automatic_copy_or_resubmission_allowed": False,
            "second_upload_allowed": False,
        },
        label="r225 replacement",
    )
    replacement_environment = _require_mapping(
        replacement.get("phase1_submission_environment"), label="replacement Phase-1 environment"
    )
    _require_expected_fields(
        replacement_environment,
        {
            **PHASE1_RESOURCES,
            "resource_probe_and_archive_size_receipt_required": True,
        },
        label="replacement Phase-1 environment",
    )
    replacement_topology = _require_mapping(
        replacement.get("phase1_simulator_search"), label="replacement two-lane topology"
    )
    _require_expected_fields(
        replacement_topology,
        {
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
        label="replacement two-lane topology",
    )
    replacement_r240_scheduler = _require_mapping(
        replacement.get("r240_hybrid_scheduler"), label="replacement r240 scheduler"
    )
    _require_expected_fields(
        replacement_r240_scheduler,
        {
            "high_confidence_frozen_direct_mode": "high_confidence_frozen_direct",
            "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_DECISION_REVISION,
            "selected_factorized_stage_probability_threshold": 0.80,
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
                "selected_factorized_stage_probability_threshold": 0.80,
                "all_selected_factorized_stages_meet_threshold": True,
                "mcts_child_started_for_this_decision": False,
                "mcts_select_call_count": 0,
                "history_only_existing_child_journal_count_range": [0, 1],
                "degraded": False,
            },
            "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
            "ambiguous_mcts_exact_lane_count": 2,
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
            "terminal_win_proof_required_only_when_stop_reason_is_proven_deterministic_terminal_win_this_turn": True,
            "terminal_win_proof_must_be_absent_or_null_for_other_stop_reasons": True,
            "zero_backups_use_only_existing_precomputed_direct_fallback_contract": True,
            "historical_fixed_eight_second_r228_branching_window_is_not_current": True,
        },
        label="replacement r240 scheduler",
    )
    _require_exact(
        replacement_r240_scheduler.get("stop_reason_fields"),
        [
            "high_confidence_frozen_direct",
            "deterministic_continuation_plan",
            R246_TERMINAL_WIN_STOP_REASON,
            "adaptive_early_stop",
            "hard_completed_backup_stop",
            "child_search_hard_deadline",
            "parent_action_hard_deadline",
            "zero_backup_precomputed_direct_fallback",
            "contained_child_fault",
        ],
        label="replacement r240 stop reasons",
    )
    _require_exact(
        dict(
            _require_mapping(
                replacement_r240_scheduler.get(
                    "r246_proven_deterministic_terminal_win_this_turn"
                ),
                label="replacement r246 deterministic terminal-win contract",
            )
        ),
        R246_REPLACEMENT_TERMINAL_WIN_THIS_TURN,
        label="replacement r246 deterministic terminal-win contract",
    )
    replacement_continuation = _require_mapping(
        replacement.get("deterministic_continuation"),
        label="replacement deterministic continuation",
    )
    _require_expected_fields(
        replacement_continuation,
        {
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
        },
        label="replacement deterministic continuation",
    )
    required_gates = replacement.get("required_local_gate_receipts")
    required_names = set(GATE_NAMES.values()) - {GATE_NAMES["go_first"]}
    if not isinstance(required_gates, list) or not required_names <= set(required_gates):
        raise R235R236BindingError("r225/r246 contract lacks required local gate names")
    authority = _require_mapping(contract.get("authority"), label="r225 authority")
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
    ):
        _require_exact(authority.get(field), False, label=f"r225 authority.{field}")
    return {
        "owner_decision_revision": R246_OWNER_DECISION_REVISION,
        "replacement_submission_count_limit": 1,
        "replacement_submission_consumed": False,
        "competition": COMPETITION,
        "submission_message_required_literal": R235_LABEL,
        "complete_ordered_action_cap": COMPLETE_ACTION_CAP,
        "canonical_libcg_contract_sha256": canonical_contract_sha256,
    }


def _manifest_entrypoint_sha256(manifest: Mapping[str, Any]) -> object:
    direct = manifest.get("entrypoint_sha256")
    if direct is not None:
        return direct
    entrypoint = manifest.get("entrypoint")
    if isinstance(entrypoint, Mapping):
        return entrypoint.get("sha256")
    return None


def _expected_manifest_native_members() -> dict[str, dict[str, object]]:
    return {
        str(member["package_relative_path"]): {
            "platform": platform,
            "wheel_member": member["wheel_member"],
            "sha256": member["sha256"],
            "size_bytes": member["size_bytes"],
        }
        for platform, member in OFFICIAL_LIBCG_MEMBERS.items()
    }


def _validate_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    entrypoint_sha256: str,
    r225_contract_sha256: str,
) -> None:
    _require_expected_fields(
        manifest,
        {
            "schema": R238_PACKAGE_MANIFEST_SCHEMA,
            "role": R238_PACKAGE_MANIFEST_ROLE,
            "required_label": R235_LABEL,
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        },
        label="candidate manifest",
    )
    r225_binding = _require_mapping(
        manifest.get("r225_typed_contract"), label="candidate manifest r225 binding"
    )
    if dict(r225_binding) != {
        "path": CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH,
        "schema": R225_SCHEMA,
        "sha256": r225_contract_sha256,
    }:
        raise R235R236BindingError("candidate manifest r225 typed-contract binding drifted")
    lifecycle = _require_mapping(
        manifest.get("required_search_lifecycle_counts"), label="candidate manifest lifecycle"
    )
    if dict(lifecycle) != {
        "search_begin_calls": 2,
        "search_end_calls": 2,
        "search_release_calls": 2,
    }:
        raise R235R236BindingError("candidate manifest lifecycle does not require exactly two lanes")
    resources = _require_mapping(
        manifest.get("phase1_kaggle_resource_bounds"), label="candidate manifest Phase-1 resources"
    )
    if dict(resources) != PHASE1_MANIFEST_RESOURCE_BOUNDS:
        raise R235R236BindingError("candidate manifest Phase-1 resource bounds drifted")
    scheduler = _require_mapping(
        manifest.get("r240_hybrid_scheduler"), label="candidate manifest r240 scheduler"
    )
    _require_expected_fields(
        scheduler, R242_MANIFEST_SCHEDULER, label="candidate manifest r240 scheduler"
    )
    continuation = _require_mapping(
        manifest.get("deterministic_continuation"), label="candidate manifest continuation"
    )
    _require_expected_fields(
        continuation,
        MANIFEST_DETERMINISTIC_CONTINUATION,
        label="candidate manifest deterministic continuation",
    )
    _require_exact(
        manifest.get("r240_required_preflight_receipts"),
        [
            GATE_NAMES["high_confidence"],
            GATE_NAMES["deterministic_continuation"],
        ],
        label="candidate manifest r240 regression receipts",
    )
    _require_expected_fields(
        manifest,
        {
            "owner_proven_deterministic_terminal_win_this_turn_revision": (
                R246_OWNER_DECISION_REVISION
            ),
            "r246_proven_deterministic_terminal_win_this_turn": (
                R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN
            ),
            "r246_required_preflight_receipts": [GATE_NAMES["terminal_win"]],
        },
        label="candidate manifest r246 deterministic terminal-win contract",
    )
    _require_exact(
        _manifest_entrypoint_sha256(manifest),
        entrypoint_sha256,
        label="candidate manifest entrypoint digest",
    )
    broker = _require_mapping(manifest.get("broker_contract"), label="candidate manifest broker")
    _require_expected_fields(
        broker,
        {
            "complete_action_cap": COMPLETE_ACTION_CAP,
            "degraded_fallback_marker": "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED",
            "search_seconds": 2.0,
            "action_timeout_seconds": 4.0,
        },
        label="candidate manifest broker",
    )
    _validate_embedded_manifest_r236(
        _require_mapping(manifest.get("canonical_libcg_contract"), label="manifest r236 contract")
    )
    native_members = _require_mapping(
        manifest.get("canonical_native_members"), label="manifest native members"
    )
    if dict(native_members) != _expected_manifest_native_members():
        raise R235R236BindingError("manifest does not bind the exact four native members")
    native_hashes = _require_mapping(
        manifest.get("canonical_native_member_sha256"), label="manifest native member digests"
    )
    expected_hashes = {
        str(member["package_relative_path"]): member["sha256"]
        for member in OFFICIAL_LIBCG_MEMBERS.values()
    }
    if dict(native_hashes) != expected_hashes:
        raise R235R236BindingError("manifest native member digest map drifted")
    base = _require_mapping(manifest.get("exact_frozen_base"), label="manifest frozen base")
    linux = OFFICIAL_LIBCG_MEMBERS["linux_x86_64"]
    _require_expected_fields(
        base,
        {
            "r195_bundle_sha256": R195_BUNDLE_SHA256,
            "r195_checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "r195_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "stock_libcg_sha256": linux["sha256"],
            "stock_libcg_size_bytes": linux["size_bytes"],
        },
        label="manifest frozen base",
    )


def _validate_archive_native_members(members: Mapping[str, Mapping[str, object]]) -> None:
    for platform, expected in OFFICIAL_LIBCG_MEMBERS.items():
        path = str(expected["package_relative_path"])
        archive_member = members.get(path)
        if archive_member is None:
            raise R235R236BindingError(f"candidate archive lacks canonical {platform} libcg")
        _require_exact(
            archive_member.get("sha256"), expected["sha256"], label=f"candidate {platform} libcg digest"
        )
        _require_exact(
            archive_member.get("size_bytes"), expected["size_bytes"], label=f"candidate {platform} libcg size"
        )


def _validate_archive_r225_contract_member(
    members: Mapping[str, Mapping[str, object]], *, r225_contract_sha256: str, r225_path: Path
) -> None:
    """Require the package to carry the exact canonical r225/r246 source bytes."""

    archive_member = members.get(CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH)
    if archive_member is None:
        raise R235R236BindingError("candidate archive lacks the canonical r225/r246 contract member")
    _require_exact(
        archive_member.get("sha256"),
        r225_contract_sha256,
        label="candidate archive r225/r246 contract digest",
    )
    _require_exact(
        archive_member.get("size_bytes"),
        r225_path.stat().st_size,
        label="candidate archive r225/r246 contract size",
    )


def _receipt_common(
    payload: Mapping[str, Any], *, receipt_name: str, identity: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "schema": PREFLIGHT_RECEIPT_SCHEMA,
            "receipt_name": receipt_name,
            "status": "passed",
            "passed": True,
            "immutable": True,
            "write_once": True,
        },
        label="gate receipt",
    )
    for field in (
        "candidate_archive_sha256",
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
    ):
        _require_exact(payload.get(field), identity[field], label=f"gate receipt {field}")


def _validate_focused_fault_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    for field in (
        "focused_fault_suite_passed",
        "nonreaped_child_hard_fail_test_passed",
        "parent_returned_action_legality_hard_fail_test_passed",
        "fault_injected_full_game_degraded_marker_and_no_viability_credit_passed",
    ):
        _require_exact(payload.get(field), True, label=f"focused fault receipt {field}")
    observed = payload.get("fault_classes_covered")
    if not isinstance(observed, list) or set(observed) != {
        "timeout",
        "crash",
        "protocol",
        "evaluator",
        "native",
        "cleanup",
    }:
        raise R235R236BindingError("focused fault receipt must cover every containment class")


def _validate_saved_episode_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "source_submission_id": 55_416_396,
            "source_episode_id": 91_766_923,
            "seat": 0,
            "final_callback_step": 58,
            "final_callback_ordered_legal_action_count": 2,
            "legal_action_before_hard_deadline": True,
            "fault_injected_broker_child_reap_proved": True,
        },
        label="saved episode receipt",
    )
    if payload.get("result_path") not in {
        "high_confidence_frozen_direct",
        "validated_deterministic_continuation_plan_action",
        "validated_mcts_action",
        "contained_precomputed_parent_direct_fallback_after_exact_child_reap",
    }:
        raise R235R236BindingError("saved episode receipt used an impermissible action path")


def _validate_full_game_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "exact_package_full_local_game_passed": True,
            "full_gameplay_loop_completed": True,
            "explicit_success_marker_count": 1,
            "degraded_game_count": 0,
            "active_simulator_search_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        },
        label="full-game receipt",
    )
    decisions = payload.get("branching_gameplay_decision_count")
    if not isinstance(decisions, int) or isinstance(decisions, bool) or decisions < 1:
        raise R235R236BindingError("full-game receipt did not exercise a branching decision")


def _validate_resource_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    for field in (
        "resource_memory_startup_and_throughput_preflight_passed",
        "memory_preflight_passed",
        "startup_preflight_passed",
        "throughput_preflight_passed",
    ):
        _require_exact(payload.get(field), True, label=f"resource receipt {field}")
    probe = payload.get("observed_resource_probe")
    if not isinstance(probe, Mapping) or not probe:
        raise R235R236BindingError("resource receipt lacks an observed resource probe")
    for field in ("startup_seconds", "throughput_decisions_per_second"):
        value = payload.get(field)
        if not _is_number(value) or float(value) < 0.0:
            raise R235R236BindingError(f"resource receipt has invalid {field}")


def _validate_phase1_resource_receipt(
    payload: Mapping[str, Any], identity: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "phase1_submission_resource_and_archive_limit_receipt_passed": True,
            "resource_probe_matches_phase1_submission_environment": True,
            "archive_within_submission_limit": True,
            "observed_phase1_submission_environment": PHASE1_RECEIPT_ENVIRONMENT,
        },
        label="Phase-1 resource receipt",
    )
    _require_exact(payload.get("observed_submission_archive_size_bytes"), identity["candidate_archive_size_bytes"], label="Phase-1 archive size")
    observed_mib = payload.get("observed_submission_archive_size_mib")
    if not _is_number(observed_mib) or float(observed_mib) < 0.0:
        raise R235R236BindingError("Phase-1 resource receipt has invalid archive MiB")
    if float(observed_mib) > float(PHASE1_RESOURCES["submission_archive_limit_mib"]):
        raise R235R236BindingError("Phase-1 resource receipt exceeds the archive limit")


def _validate_two_lane_topology_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
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
            "lane_ids": [0, 1],
        },
        label="two-lane topology receipt",
    )
    releases = payload.get("search_release_calls")
    if not isinstance(releases, int) or isinstance(releases, bool) or releases < 2:
        raise R235R236BindingError("two-lane topology receipt lacks two releases")
    for field in ("per_lane_depth",):
        value = payload.get(field)
        if not isinstance(value, list) or len(value) != 2:
            raise R235R236BindingError(f"two-lane topology receipt has invalid {field}")
    _validate_handle_scoped_search_identity(payload, label="two-lane topology receipt")
    microbatches = payload.get("microbatch_sizes")
    if not isinstance(microbatches, list) or not microbatches or any(
        not isinstance(size, int) or isinstance(size, bool) or size not in (1, 2)
        for size in microbatches
    ):
        raise R235R236BindingError("two-lane topology receipt has invalid microbatch sizes")
    inflight = payload.get("max_simulator_calls_in_flight")
    if not isinstance(inflight, int) or isinstance(inflight, bool) or inflight not in (1, 2):
        raise R235R236BindingError("two-lane topology receipt has invalid in-flight count")


def _validate_handle_scoped_search_identity(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, object]:
    """Validate two official-libcg SearchId states in their handle namespaces."""

    handles = payload.get("per_lane_handle_identities")
    chains = payload.get("per_lane_search_id_chains")
    if not isinstance(handles, list) or len(handles) != SIMULATOR_SEARCH_LANE_COUNT:
        raise R235R236BindingError(f"{label} lacks exactly two handle identities")
    if not isinstance(chains, list) or len(chains) != SIMULATOR_SEARCH_LANE_COUNT:
        raise R235R236BindingError(f"{label} lacks exactly two SearchId chains")
    checked_handles: list[int | str] = []
    checked_chains: list[list[int]] = []
    for lane, handle in enumerate(handles):
        if (
            isinstance(handle, bool)
            or not isinstance(handle, (int, str))
            or (isinstance(handle, str) and not handle)
        ):
            raise R235R236BindingError(f"{label} has invalid handle identity for lane {lane}")
        checked_handles.append(handle)
        chain = chains[lane]
        if not isinstance(chain, list) or not chain:
            raise R235R236BindingError(f"{label} has an empty SearchId chain for lane {lane}")
        checked_chain: list[int] = []
        for index, search_id in enumerate(chain):
            if (
                not isinstance(search_id, int)
                or isinstance(search_id, bool)
                or search_id < 0
            ):
                raise R235R236BindingError(
                    f"{label} has invalid SearchId for lane {lane} index {index}"
                )
            checked_chain.append(search_id)
        checked_chains.append(checked_chain)
    if len(set(checked_handles)) != SIMULATOR_SEARCH_LANE_COUNT:
        raise R235R236BindingError(f"{label} lacks two distinct handles")
    first_search_ids = [chain[0] for chain in checked_chains]
    composites = {
        (checked_handles[lane], first_search_ids[lane])
        for lane in range(SIMULATOR_SEARCH_LANE_COUNT)
    }
    if len(composites) != SIMULATOR_SEARCH_LANE_COUNT:
        raise R235R236BindingError(
            f"{label} lacks two distinct handle-scoped first SearchId composites"
        )
    expected_composites = [
        {
            "lane_id": lane,
            "handle_identity": checked_handles[lane],
            "first_search_id": first_search_ids[lane],
        }
        for lane in range(SIMULATOR_SEARCH_LANE_COUNT)
    ]
    observed_composites = payload.get("handle_scoped_first_search_id_composite_states")
    expected_composite_keys = set(
        R244_HANDLE_SCOPED_SEARCH_IDENTITY[
            "public_composite_state_entry_exact_keys_in_order"
        ]
    )
    if not isinstance(observed_composites, list) or any(
        not isinstance(record, Mapping) or set(record) != expected_composite_keys
        for record in observed_composites
    ):
        raise R235R236BindingError(
            f"{label} has an invalid handle-scoped first SearchId composite shape"
        )
    _require_exact(
        payload.get("per_lane_first_search_ids"),
        first_search_ids,
        label=f"{label} per-lane first SearchIds",
    )
    _require_exact(
        observed_composites,
        expected_composites,
        label=f"{label} handle-scoped first SearchId composites",
    )
    compatibility_projection = payload.get("per_lane_handle_first_search_id_composites")
    if compatibility_projection is not None:
        _require_exact(
            compatibility_projection,
            expected_composites,
            label=f"{label} compatibility composite projection",
        )
    return {
        "per_lane_handle_identities": checked_handles,
        "per_lane_search_id_chains": checked_chains,
        "per_lane_first_search_ids": first_search_ids,
        "handle_scoped_first_search_id_composite_states": expected_composites,
        "distinct_handle_identity_count": SIMULATOR_SEARCH_LANE_COUNT,
        "distinct_handle_scoped_first_search_id_composite_state_count": SIMULATOR_SEARCH_LANE_COUNT,
        "globally_distinct_raw_search_id_integers_required": False,
    }


def _validate_handle_scoped_search_id_receipt(
    payload: Mapping[str, Any], identity: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "schema": R244_HANDLE_SCOPED_SEARCH_ID_RECEIPT_SCHEMA,
            "receipt_name": GATE_NAMES["handle_scoped_search_id"],
            "status": "passed",
            "passed": True,
            "immutable": True,
            "write_once": True,
            "r225_contract_sha256": identity["r225_contract_sha256"],
            "canonical_libcg_contract_sha256": identity[
                "canonical_libcg_contract_sha256"
            ],
            "official_libcg_handle_scoped_search_id_identity_regression_passed": True,
            "requested_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "active_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "arena_count": SIMULATOR_SEARCH_LANE_COUNT,
            "unique_handle_count": SIMULATOR_SEARCH_LANE_COUNT,
            "distinct_handle_identity_count": SIMULATOR_SEARCH_LANE_COUNT,
            "distinct_handle_scoped_first_search_id_composite_state_count": SIMULATOR_SEARCH_LANE_COUNT,
            "search_begin_calls": SIMULATOR_SEARCH_LANE_COUNT,
            "search_id_numeric_namespace": "per_distinct_agent_start_handle",
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "r244_owner_revision": R244_OWNER_DECISION_REVISION,
            "same_raw_first_search_id_on_distinct_handles_accepted": True,
            "duplicate_handle_identity_rejected": True,
            "duplicate_handle_scoped_first_search_id_composite_rejected": True,
        },
        label="R244 handle-scoped SearchId receipt",
    )
    _validate_handle_scoped_search_identity(
        payload, label="R244 handle-scoped SearchId receipt"
    )


def _validate_high_confidence_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "high_confidence_direct_and_adaptive_bounded_mcts_regression_passed": True,
            "high_confidence_mode": "high_confidence_frozen_direct",
            "high_confidence_path_returned_precomputed_legal_direct_action": True,
            "selected_factorized_stage_probability_threshold": 0.80,
            "all_selected_factorized_stages_meet_threshold": True,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "mcts_search_call_count": 0,
            "mcts_model_call_count": 0,
            "mcts_simulator_call_count": 0,
            "degraded": False,
            "ambiguous_selected_stage_forced_mcts": True,
            "child_search_seconds": 2.0,
            "parent_action_deadline_seconds": 4.0,
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations": 3,
            "maximum_backups_per_decision": 32,
            "both_lanes_progressed": True,
            "legacy_fixed_eight_second_window_used": False,
        },
        label="r242 high-confidence receipt",
    )
    probabilities = payload.get("selected_factorized_stage_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise R235R236BindingError("r242 high-confidence receipt lacks stage probabilities")
    if any(not _is_number(value) or not math.isfinite(float(value)) or float(value) < 0.80 for value in probabilities):
        raise R235R236BindingError("r242 high-confidence receipt did not prove every stage threshold")
    if payload.get("stop_reason") not in {
        "adaptive_early_stop",
        "stable_root_leader",
    }:
        raise R235R236BindingError(
            "r242 high-confidence receipt lacks an allowed adaptive stop reason"
        )
    history_only_count = payload.get("history_only_existing_child_journal_count")
    if (
        not isinstance(history_only_count, int)
        or isinstance(history_only_count, bool)
        or history_only_count not in (0, 1)
    ):
        raise R235R236BindingError(
            "r242 high-confidence receipt has invalid history-only child journal count"
        )
    backups = payload.get("completed_backups")
    if not isinstance(backups, int) or isinstance(backups, bool) or not 8 <= backups <= 32:
        raise R235R236BindingError("r242 high-confidence receipt has invalid adaptive backup count")
    leader_observations = payload.get("same_root_leader_observations")
    if not isinstance(leader_observations, int) or isinstance(leader_observations, bool) or leader_observations < 3:
        raise R235R236BindingError("r242 high-confidence receipt lacks stable leader evidence")
    _actor_change_end_turn_boundary_projection(payload)


def _r246_action(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise R235R236BindingError(f"{label} must be a nonempty factorized action list")
    checked: list[int] = []
    for index, part in enumerate(value):
        if not isinstance(part, int) or isinstance(part, bool) or part < 0:
            raise R235R236BindingError(
                f"{label} has an invalid factorized action part at index {index}"
            )
        checked.append(part)
    return checked


def _r246_actor_seat(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        raise R235R236BindingError(f"{label} must be seat 0 or 1")
    return value


def _validate_r246_terminal_win_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> dict[str, object]:
    """Validate the one-proof R246 exception after exact two-lane initialization."""

    if R246_LEGACY_CLEANUP_COMPLETED_FIELD in payload:
        raise R235R236BindingError(
            "r246 deterministic terminal-win receipt uses the legacy cleanup field "
            f"{R246_LEGACY_CLEANUP_COMPLETED_FIELD}; expected "
            f"{R246_CLEANUP_COMPLETE_FIELD}"
        )
    _require_expected_fields(
        payload,
        {
            "owner_proven_deterministic_terminal_win_this_turn_revision": R246_OWNER_DECISION_REVISION,
            "proven_deterministic_terminal_win_this_turn_regression_passed": True,
            "stop_reason": R246_TERMINAL_WIN_STOP_REASON,
            "two_lane_topology_initialized_before_terminal_win_override": True,
            "requested_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "active_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "arena_count": SIMULATOR_SEARCH_LANE_COUNT,
            "unique_handle_count": SIMULATOR_SEARCH_LANE_COUNT,
            "search_begin_calls": SIMULATOR_SEARCH_LANE_COUNT,
            "search_release_calls": SIMULATOR_SEARCH_LANE_COUNT,
            "search_end_calls": SIMULATOR_SEARCH_LANE_COUNT,
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
        },
        label="r246 deterministic terminal-win receipt",
    )
    backup_count = _nonnegative_int(
        payload.get("completed_root_backup_count"),
        label="r246 deterministic terminal-win receipt.completed_root_backup_count",
    )
    if not 1 <= backup_count <= int(R242_HYBRID_SCHEDULER["maximum_backups_per_decision"]):
        raise R235R236BindingError(
            "r246 deterministic terminal-win receipt has an invalid terminal backup count"
        )
    proof = _require_mapping(
        payload.get("terminal_win_proof"),
        label="r246 deterministic terminal-win proof",
    )
    if set(proof) != set(R246_TERMINAL_WIN_PROOF_FIELDS):
        raise R235R236BindingError(
            "r246 deterministic terminal-win proof has an unexpected or missing field"
        )
    _require_expected_fields(
        proof,
        {
            "proof_kind": R246_TERMINAL_WIN_PROOF_KIND,
            "terminal_result": "win",
            "terminal_leaf_reached": True,
            "path_no_chance_boundary": True,
            "path_no_actor_change_boundary": True,
            "path_no_opponent_boundary_crossing": True,
            "path_no_unresolved_randomness": True,
            "proof_is_deterministic": True,
        },
        label="r246 deterministic terminal-win proof",
    )
    for field in ("root_observation_fingerprint", "root_legal_order_fingerprint"):
        value = proof.get(field)
        if not isinstance(value, str) or not value:
            raise R235R236BindingError(
                f"r246 deterministic terminal-win proof has invalid {field}"
            )
    root_actor = _r246_actor_seat(
        proof.get("root_actor_seat"),
        label="r246 deterministic terminal-win proof.root_actor_seat",
    )
    root_action = _r246_action(
        proof.get("root_action"), label="r246 deterministic terminal-win proof.root_action"
    )
    selected_action = _r246_action(
        proof.get("selected_action"),
        label="r246 deterministic terminal-win proof.selected_action",
    )
    if selected_action != root_action:
        raise R235R236BindingError(
            "r246 deterministic terminal-win proof selected action differs from legal root action"
        )
    terminal_winner = _r246_actor_seat(
        proof.get("terminal_winner_seat"),
        label="r246 deterministic terminal-win proof.terminal_winner_seat",
    )
    if terminal_winner != root_actor:
        raise R235R236BindingError(
            "r246 deterministic terminal-win proof winner differs from root actor"
        )
    path_count = _nonnegative_int(
        proof.get("proof_path_action_count"),
        label="r246 deterministic terminal-win proof.proof_path_action_count",
    )
    if not 1 <= path_count <= backup_count:
        raise R235R236BindingError(
            "r246 deterministic terminal-win proof path is outside its completed backup count"
        )
    discovering_lane = _nonnegative_int(
        proof.get("discovering_lane_id"),
        label="r246 deterministic terminal-win proof.discovering_lane_id",
    )
    if discovering_lane not in (0, 1):
        raise R235R236BindingError(
            "r246 deterministic terminal-win proof has an invalid discovering lane"
        )
    path_actor_seats = proof.get("path_actor_seats")
    if (
        not isinstance(path_actor_seats, list)
        or len(path_actor_seats) != path_count
        or any(
            _r246_actor_seat(
                seat,
                label="r246 deterministic terminal-win proof.path_actor_seats",
            )
            != root_actor
            for seat in path_actor_seats
        )
    ):
        raise R235R236BindingError(
            "r246 deterministic terminal-win proof crosses an actor/opponent boundary"
        )
    normalized_proof = {
        field: (root_action if field == "root_action" else selected_action if field == "selected_action" else proof[field])
        for field in R246_TERMINAL_WIN_PROOF_FIELDS
    }
    return {
        "owner_proven_deterministic_terminal_win_this_turn_revision": R246_OWNER_DECISION_REVISION,
        "proven_deterministic_terminal_win_this_turn_regression_passed": True,
        "stop_reason": R246_TERMINAL_WIN_STOP_REASON,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "requested_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        "active_simulator_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        "arena_count": SIMULATOR_SEARCH_LANE_COUNT,
        "unique_handle_count": SIMULATOR_SEARCH_LANE_COUNT,
        "search_begin_calls": SIMULATOR_SEARCH_LANE_COUNT,
        "search_release_calls": SIMULATOR_SEARCH_LANE_COUNT,
        "search_end_calls": SIMULATOR_SEARCH_LANE_COUNT,
        "completed_root_backup_count": backup_count,
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
        "terminal_win_proof": normalized_proof,
    }


def _nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise R235R236BindingError(f"{label} must be a nonnegative integer")
    return value


def _actor_change_end_turn_boundary_projection(
    payload: Mapping[str, Any],
) -> dict[str, object]:
    """Validate the R242 actor-change leaf boundary and retain its counters."""

    boundary = _require_mapping(
        payload.get("actor_change_end_turn_boundary"),
        label="r242 actor-change/end-turn boundary receipt",
    )
    _require_exact(
        boundary.get("actor_change_end_turn_boundary_regression_passed"),
        True,
        label="r242 actor-change/end-turn boundary receipt passed",
    )
    counters = {
        field: _nonnegative_int(
            boundary.get(field), label=f"r242 actor-change/end-turn boundary.{field}"
        )
        for field in R242_ACTOR_BOUNDARY_COUNTER_FIELDS
    }
    leaf_count = counters["declared_opponent_actor_leaf_count"]
    if leaf_count < 1:
        raise R235R236BindingError(
            "r242 actor-change/end-turn boundary receipt has no opponent actor leaf"
        )
    if counters["value_evaluated_opponent_actor_leaf_count"] != leaf_count:
        raise R235R236BindingError(
            "r242 actor-change/end-turn boundary receipt did not value-evaluate every leaf"
        )
    actor_change_count = _nonnegative_int(
        payload.get("actor_change_boundary_leaf_count"),
        label="r242 high-confidence receipt.actor_change_boundary_leaf_count",
    )
    chance_count = _nonnegative_int(
        payload.get("chance_boundary_leaf_count"),
        label="r242 high-confidence receipt.chance_boundary_leaf_count",
    )
    boundary_count = _nonnegative_int(
        payload.get("boundary_leaf_count"),
        label="r242 high-confidence receipt.boundary_leaf_count",
    )
    if actor_change_count != leaf_count or boundary_count != actor_change_count + chance_count:
        raise R235R236BindingError(
            "r242 actor-change/end-turn boundary receipt aggregate counters drifted"
        )
    for field in (
        "expanded_legal_action_count",
        "expanded_child_count",
        "search_steps_beyond_boundary",
        "opponent_action_selected_or_planned_count",
        "opponent_action_cached_count",
    ):
        if counters[field] != 0:
            raise R235R236BindingError(
                f"r242 actor-change/end-turn boundary receipt has nonzero {field}"
            )
    leaves = boundary.get("opponent_actor_leaves")
    if not isinstance(leaves, list) or len(leaves) != leaf_count:
        raise R235R236BindingError(
            "r242 actor-change/end-turn boundary receipt leaf list does not match count"
        )
    for index, leaf in enumerate(leaves):
        leaf_mapping = _require_mapping(
            leaf, label=f"r242 actor-change/end-turn boundary leaf {index}"
        )
        _require_expected_fields(
            leaf_mapping,
            {
                "model_value_evaluated": True,
                "expanded_legal_action_count": 0,
                "expanded_child_count": 0,
                "search_steps_beyond_boundary": 0,
                "opponent_action_selected_or_planned": False,
                "opponent_action_cached": False,
            },
            label=f"r242 actor-change/end-turn boundary leaf {index}",
        )
    return {
        "actor_change_end_turn_boundary_regression_passed": True,
        **counters,
        "opponent_actor_leaf_count": leaf_count,
        "actor_change_boundary_leaf_count": actor_change_count,
        "chance_boundary_leaf_count": chance_count,
        "boundary_leaf_count": boundary_count,
    }


def _validate_deterministic_continuation_receipt(
    payload: Mapping[str, Any], _: Mapping[str, object]
) -> None:
    _require_expected_fields(
        payload,
        {
            "deterministic_continuation_regression_passed": True,
            "two_lane_agreed_exact_fingerprint_path_consumed": True,
            "valid_match_no_new_search": True,
            "valid_match_started_new_search": False,
            "valid_match_backed_action_consumed": True,
            "valid_match_same_root_actor": True,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "degraded": False,
            "configured_max_depth": 8,
            "chance_disagreement_clears_entire_plan": True,
            "fingerprint_disagreement_clears_entire_plan": True,
            "action_disagreement_clears_entire_plan": True,
            "actor_disagreement_clears_entire_plan": True,
            "crossed_actor_change_end_turn_boundary": False,
            "precomputed_direct_action_and_history_correction_retained": True,
        },
        label="deterministic continuation receipt",
    )
    depth = payload.get("observed_valid_match_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 8:
        raise R235R236BindingError("deterministic continuation receipt has invalid depth")
    history_only_count = payload.get("history_only_existing_child_journal_count")
    if (
        not isinstance(history_only_count, int)
        or isinstance(history_only_count, bool)
        or history_only_count not in (0, 1)
    ):
        raise R235R236BindingError(
            "deterministic continuation receipt has invalid history-only child journal count"
        )


def _validate_prebinding_go_first_sidecar(
    payload: Mapping[str, Any], identity: Mapping[str, object]
) -> None:
    """Validate the immutable archive sidecar before a binding exists.

    This intentionally does not use ``_receipt_common``: the sidecar is the
    guard's turn-order attestation schema and therefore cannot truthfully also
    claim to be a local-preflight receipt.  Its duplicate candidate identity
    fields bind it to the same archive inputs before this builder publishes the
    immutable binding it will later be consumed with.
    """

    allowed_fields = {
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
    if set(payload) != allowed_fields:
        raise R235R236BindingError(
            "pre-binding go-first sidecar has an unexpected or missing field"
        )
    _require_expected_fields(
        payload,
        {
            "schema": GO_FIRST_ATTESTATION_SCHEMA,
            "kind": GO_FIRST_VERIFIER_KIND,
            "status": "passed",
            "receipt_name": GATE_NAMES["go_first"],
            "passed": True,
            "immutable": True,
            "write_once": True,
            "go_first_contract_passed": True,
            "forced_yes_action_legal": True,
            "turn_order_preference": "first_if_allowed",
            "go_first_if_offered": True,
            "go_second_if_offered": False,
            "verified_cases": GO_FIRST_VERIFIED_CASES,
            "case_results": GO_FIRST_CASE_RESULTS,
            "file_sha256": identity["candidate_archive_sha256"],
            "file_bytes": identity["candidate_archive_size_bytes"],
            "submission": {"competition": COMPETITION, "message": R235_LABEL},
            "manifest": {
                "path": identity["member_manifest_path"],
                "sha256": identity["member_manifest_sha256"],
                "member": identity["member_manifest_member"],
                "schema": R238_PACKAGE_MANIFEST_SCHEMA,
                "lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            },
            "typed_contracts": {
                "r225_sha256": identity["r225_contract_sha256"],
                "r236_sha256": identity["canonical_libcg_contract_sha256"],
                "r225_owner_decision_revision": R246_OWNER_DECISION_REVISION,
            },
        },
        label="pre-binding go-first sidecar",
    )
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
    ):
        _require_exact(
            payload.get(field), identity[field], label=f"pre-binding go-first sidecar {field}"
        )
    verified_at = payload.get("verified_at_utc")
    if not isinstance(verified_at, str) or not verified_at.strip():
        raise R235R236BindingError("pre-binding go-first sidecar lacks verification time")


def _validate_go_first_sidecar_path(path: Path, *, archive_path: Path) -> Path:
    sidecar = _regular_file(path, label="pre-binding go-first sidecar")
    expected = Path(str(archive_path) + ".go-first-verified.json").resolve()
    if sidecar != expected:
        raise R235R236BindingError(
            "pre-binding go-first sidecar must be the exact archive sidecar"
        )
    if sidecar.stat().st_mode & 0o222:
        raise R235R236BindingError("pre-binding go-first sidecar must be read-only")
    return sidecar


def _read_and_validate_gate(
    path: Path,
    *,
    receipt_name: str,
    identity: Mapping[str, object],
    validator: Callable[[Mapping[str, Any], Mapping[str, object]], None],
    projection: Callable[[Mapping[str, Any]], Mapping[str, object]] | None = None,
    common_identity: bool = True,
) -> dict[str, object]:
    # A receipt that names itself immutable/write-once is not eligible to
    # bind an upload if any principal can still rewrite it.  Preserve the
    # existing physical-file/symlink checks, then reject all owner/group/other
    # write bits before parsing or trusting the asserted JSON fields.
    receipt_path = _regular_file(path, label=receipt_name)
    if os.lstat(receipt_path).st_mode & 0o222:
        raise R235R236BindingError(
            f"{receipt_name} must not be writable by owner, group, or other"
        )
    payload = _read_json_object(receipt_path, label=receipt_name)
    if common_identity:
        _receipt_common(payload, receipt_name=receipt_name, identity=identity)
    validator(payload, identity)
    bound: dict[str, object] = {
        "receipt_name": receipt_name,
        "sha256": sha256_file(receipt_path),
        "passed": True,
    }
    if projection is not None:
        bound["validated_counter_projection"] = dict(projection(payload))
    return bound


def write_once_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish bytes through a hard link and never overwrite a receipt."""

    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise R235R236BindingError(f"binding output must not be a symlink: {raw_path}")
    path = raw_path.resolve()
    if path.exists():
        raise R235R236BindingError(f"binding output already exists; refusing overwrite: {path}")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise R235R236BindingError("binding output parent must be an existing directory")
    canonical = canonical_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        try:
            written = 0
            while written < len(canonical):
                amount = os.write(descriptor, canonical[written:])
                if amount <= 0:
                    raise OSError("could not write immutable binding payload")
                written += amount
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise R235R236BindingError(
                f"binding output already exists; refusing overwrite: {path}"
            ) from exc
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_binding(
    *,
    candidate_archive: Path,
    member_manifest: Path,
    focused_fault_receipt: Path,
    saved_episode_receipt: Path,
    full_game_receipt: Path,
    resource_receipt: Path,
    phase1_resource_receipt: Path,
    two_lane_topology_receipt: Path,
    handle_scoped_search_id_receipt: Path,
    high_confidence_receipt: Path,
    terminal_win_receipt: Path,
    deterministic_continuation_receipt: Path,
    go_first_receipt: Path,
    output: Path,
    r225_contract: Path | None = None,
    canonical_libcg_contract: Path | None = None,
    member_manifest_member: str = "",
    entrypoint_member: str = "main.py",
) -> dict[str, Any]:
    """Validate all local evidence and atomically publish one immutable binding."""

    archive_path = _regular_file(candidate_archive, label="candidate archive")
    manifest_path = _regular_file(member_manifest, label="member manifest")
    r225_path = _canonical_typed_file(
        r225_contract or CANONICAL_R225_R240_TYPED_CONTRACT_PATH,
        canonical_path=CANONICAL_R225_R240_TYPED_CONTRACT_PATH,
        label="r225/r246 contract",
    )
    r236_path = _canonical_typed_file(
        canonical_libcg_contract or CANONICAL_R236_TYPED_CONTRACT_PATH,
        canonical_path=CANONICAL_R236_TYPED_CONTRACT_PATH,
        label="r236 contract",
    )
    if not member_manifest_member:
        raise R235R236BindingError("member manifest member must be supplied explicitly")
    if archive_path.stat().st_size > PHASE1_ARCHIVE_MAX_BYTES:
        raise R235R236BindingError("candidate archive exceeds the Phase-1 197.7 MiB cap")
    if not CANONICAL_R225_R240_TYPED_CONTRACT_SHA256:
        raise R235R236BindingError(
            "final r246 canonical r225 contract digest is not configured; refusing binding"
        )

    archive_sha = sha256_file(archive_path)
    r225_sha = sha256_file(r225_path)
    _require_exact(
        r225_sha,
        CANONICAL_R225_R240_TYPED_CONTRACT_SHA256,
        label="canonical r225/r246 contract digest",
    )
    r236_sha = sha256_file(r236_path)
    _require_exact(
        r236_sha,
        CANONICAL_R236_TYPED_CONTRACT_SHA256,
        label="canonical r236 contract digest",
    )
    r236_payload = _read_json_object(r236_path, label="r236 contract")
    canonical_set = _validate_r236_contract(r236_payload)
    r225_payload = _read_json_object(r225_path, label="r225/r246 contract")
    r225_summary = _validate_r225_contract(
        r225_payload, canonical_contract_sha256=r236_sha
    )

    members, embedded_manifest, entrypoint_sha = inspect_archive(
        archive_path,
        member_manifest_member=member_manifest_member,
        entrypoint_member=entrypoint_member,
    )
    external_manifest = manifest_path.read_bytes()
    if embedded_manifest != external_manifest:
        raise R235R236BindingError(
            "provided member manifest does not byte-match the archive member"
        )
    try:
        candidate_manifest = json.loads(external_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235R236BindingError("candidate member manifest is not JSON") from exc
    if not isinstance(candidate_manifest, Mapping):
        raise R235R236BindingError("candidate member manifest must be a JSON object")
    _validate_candidate_manifest(
        candidate_manifest,
        entrypoint_sha256=entrypoint_sha,
        r225_contract_sha256=r225_sha,
    )
    _validate_archive_r225_contract_member(
        members, r225_contract_sha256=r225_sha, r225_path=r225_path
    )
    _validate_archive_native_members(members)

    computed_member_manifest = {
        "schema": "poke_bot.r235_r236_archive_member_manifest/v1",
        "archive_sha256": archive_sha,
        "members": members,
    }
    computed_member_manifest_sha = sha256_bytes(canonical_json(computed_member_manifest))
    linux = OFFICIAL_LIBCG_MEMBERS["linux_x86_64"]
    identity: dict[str, object] = {
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_size_bytes": archive_path.stat().st_size,
        "member_manifest_path": str(manifest_path),
        "member_manifest_member": _safe_member_name(
            member_manifest_member, label="member manifest selector"
        ),
        "member_manifest_sha256": sha256_file(manifest_path),
        "entrypoint_sha256": entrypoint_sha,
        "r225_contract_sha256": r225_sha,
        "canonical_libcg_contract_sha256": r236_sha,
        "linux_x86_64_libcg_sha256": linux["sha256"],
        "linux_x86_64_libcg_size_bytes": linux["size_bytes"],
        "complete_ordered_action_cap": COMPLETE_ACTION_CAP,
        "simulator_search_lane_count": SIMULATOR_SEARCH_LANE_COUNT,
        "phase1_submission_environment": PHASE1_RESOURCES,
        "r240_hybrid_scheduler": R242_HYBRID_SCHEDULER,
        "deterministic_continuation": DETERMINISTIC_CONTINUATION,
    }
    prebinding_go_first_sidecar = _validate_go_first_sidecar_path(
        go_first_receipt, archive_path=archive_path
    )
    gates = {
        "focused_native_child_fault_suite": _read_and_validate_gate(
            focused_fault_receipt,
            receipt_name=GATE_NAMES["focused_fault"],
            identity=identity,
            validator=_validate_focused_fault_receipt,
        ),
        "saved_episode_91766923_step58": _read_and_validate_gate(
            saved_episode_receipt,
            receipt_name=GATE_NAMES["saved_episode"],
            identity=identity,
            validator=_validate_saved_episode_receipt,
        ),
        "exact_repaired_package_full_local_game": _read_and_validate_gate(
            full_game_receipt,
            receipt_name=GATE_NAMES["full_game"],
            identity=identity,
            validator=_validate_full_game_receipt,
        ),
        "resource_memory_startup_throughput": _read_and_validate_gate(
            resource_receipt,
            receipt_name=GATE_NAMES["resource"],
            identity=identity,
            validator=_validate_resource_receipt,
        ),
        "phase1_resource_and_archive": _read_and_validate_gate(
            phase1_resource_receipt,
            receipt_name=GATE_NAMES["phase1_resource"],
            identity=identity,
            validator=_validate_phase1_resource_receipt,
        ),
        "two_lane_topology": _read_and_validate_gate(
            two_lane_topology_receipt,
            receipt_name=GATE_NAMES["two_lane_topology"],
            identity=identity,
            validator=_validate_two_lane_topology_receipt,
            projection=lambda payload: _validate_handle_scoped_search_identity(
                payload, label="two-lane topology receipt"
            ),
        ),
        "official_libcg_handle_scoped_search_id_identity": _read_and_validate_gate(
            handle_scoped_search_id_receipt,
            receipt_name=GATE_NAMES["handle_scoped_search_id"],
            identity=identity,
            validator=_validate_handle_scoped_search_id_receipt,
            projection=lambda payload: _validate_handle_scoped_search_identity(
                payload, label="R244 handle-scoped SearchId receipt"
            ),
            common_identity=False,
        ),
        "high_confidence_and_adaptive_bounded_mcts": _read_and_validate_gate(
            high_confidence_receipt,
            receipt_name=GATE_NAMES["high_confidence"],
            identity=identity,
            validator=_validate_high_confidence_receipt,
            projection=_actor_change_end_turn_boundary_projection,
        ),
        "proven_deterministic_terminal_win_this_turn": _read_and_validate_gate(
            terminal_win_receipt,
            receipt_name=GATE_NAMES["terminal_win"],
            identity=identity,
            validator=_validate_r246_terminal_win_receipt,
            projection=lambda payload: _validate_r246_terminal_win_receipt(
                payload, identity
            ),
        ),
        "deterministic_continuation": _read_and_validate_gate(
            deterministic_continuation_receipt,
            receipt_name=GATE_NAMES["deterministic_continuation"],
            identity=identity,
            validator=_validate_deterministic_continuation_receipt,
        ),
        "go_first": _read_and_validate_gate(
            prebinding_go_first_sidecar,
            receipt_name=GATE_NAMES["go_first"],
            identity=identity,
            validator=_validate_prebinding_go_first_sidecar,
            common_identity=False,
        ),
    }

    binding: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "local_gates_bound_not_submitted",
        "immutable": True,
        "write_once": True,
        "owner_decision_revision": R246_OWNER_DECISION_REVISION,
        "candidate_package": {
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_path.stat().st_size,
            "member_manifest_member": _safe_member_name(
                member_manifest_member, label="member manifest selector"
            ),
            "member_manifest_sha256": identity["member_manifest_sha256"],
            "manifest_schema": R238_PACKAGE_MANIFEST_SCHEMA,
            "manifest_role": R238_PACKAGE_MANIFEST_ROLE,
            "entrypoint_member": _safe_member_name(
                entrypoint_member, label="entrypoint selector"
            ),
            "entrypoint_sha256": entrypoint_sha,
            "computed_archive_member_manifest_sha256": computed_member_manifest_sha,
            "computed_archive_member_manifest": computed_member_manifest,
        },
        "typed_contracts": {
            "r225_historical_r246_typed_contract": {
                "canonical_relative_path": CANONICAL_R225_R240_TYPED_CONTRACT_RELATIVE_PATH,
                "resolved_path": str(r225_path),
                "schema": R225_SCHEMA,
                "sha256": r225_sha,
                "expected_sha256": CANONICAL_R225_R240_TYPED_CONTRACT_SHA256,
                "archive_member_bound": True,
            },
            "r236_canonical_libcg_typed_contract": {
                "canonical_relative_path": CANONICAL_R236_TYPED_CONTRACT_RELATIVE_PATH,
                "resolved_path": str(r236_path),
                "schema": R236_SCHEMA,
                "sha256": r236_sha,
                "expected_sha256": CANONICAL_R236_TYPED_CONTRACT_SHA256,
            },
            "r225_r246_replacement_summary": r225_summary,
        },
        "canonical_libcg_r236": canonical_set,
        "frozen_r195_identity": {
            "bundle_sha256": R195_BUNDLE_SHA256,
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        },
        "action_space": {
            "complete_ordered_action_cap": COMPLETE_ACTION_CAP,
            "applies_at_root_and_every_private_leaf": True,
            "sampling_pruning_or_reinterpretation_allowed": False,
            "root_over_cap_behavior": "hard_fail_nonzero",
            "private_leaf_over_cap_behavior": (
                "contained_degraded_parent_direct_fallback_only_after_validated_"
                "precomputed_root_direct_action_and_exact_child_reap"
            ),
        },
        "simulator_search_topology": {
            "lane_count": SIMULATOR_SEARCH_LANE_COUNT,
            "historical_eight_lane_manifest_or_receipt_accepted": False,
            "phase1_submission_environment": {
                **PHASE1_RESOURCES,
                "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
            },
            "phase1_manifest_resource_bounds": PHASE1_MANIFEST_RESOURCE_BOUNDS,
        },
        "r240_hybrid_scheduler": {
            **R242_HYBRID_SCHEDULER,
            "high_confidence_mode": "high_confidence_frozen_direct",
            "legacy_fixed_eight_second_window_accepted": False,
        },
        "r240_manifest_scheduler": R242_MANIFEST_SCHEDULER,
        "r244_handle_scoped_search_id_identity": R244_HANDLE_SCOPED_SEARCH_IDENTITY,
        "r246_proven_deterministic_terminal_win_this_turn": (
            R246_PROVEN_DETERMINISTIC_TERMINAL_WIN_THIS_TURN
        ),
        "deterministic_continuation": DETERMINISTIC_CONTINUATION,
        "manifest_deterministic_continuation": MANIFEST_DETERMINISTIC_CONTINUATION,
        "required_submission": {
            "competition": COMPETITION,
            "label": R235_LABEL,
            "go_first_preference": "first_if_allowed",
        },
        "local_gate_receipts": gates,
        "authorization": {
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
        },
        "builder": {
            "network_accessed": False,
            "kaggle_api_called": False,
            "kaggle_queue_used": False,
            "kaggle_upload_used": False,
            "gpu_used": False,
            "service_modified": False,
            "bo1000_modified": False,
        },
    }
    write_once_atomic(output, binding)
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument("--member-manifest-member", required=True)
    parser.add_argument("--focused-fault-receipt", type=Path, required=True)
    parser.add_argument("--saved-episode-receipt", type=Path, required=True)
    parser.add_argument("--full-game-receipt", type=Path, required=True)
    parser.add_argument("--resource-receipt", type=Path, required=True)
    parser.add_argument("--phase1-resource-receipt", type=Path, required=True)
    parser.add_argument("--two-lane-topology-receipt", type=Path, required=True)
    parser.add_argument("--handle-scoped-search-id-receipt", type=Path, required=True)
    parser.add_argument("--high-confidence-receipt", type=Path, required=True)
    parser.add_argument("--terminal-win-receipt", type=Path, required=True)
    parser.add_argument("--deterministic-continuation-receipt", type=Path, required=True)
    parser.add_argument("--go-first-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--r225-contract", type=Path, default=CANONICAL_R225_R240_TYPED_CONTRACT_PATH
    )
    parser.add_argument(
        "--canonical-libcg-contract", type=Path, default=CANONICAL_R236_TYPED_CONTRACT_PATH
    )
    parser.add_argument("--entrypoint-member", default="main.py")
    args = parser.parse_args()
    result = build_binding(
        candidate_archive=args.candidate_archive,
        member_manifest=args.member_manifest,
        member_manifest_member=args.member_manifest_member,
        focused_fault_receipt=args.focused_fault_receipt,
        saved_episode_receipt=args.saved_episode_receipt,
        full_game_receipt=args.full_game_receipt,
        resource_receipt=args.resource_receipt,
        phase1_resource_receipt=args.phase1_resource_receipt,
        two_lane_topology_receipt=args.two_lane_topology_receipt,
        handle_scoped_search_id_receipt=args.handle_scoped_search_id_receipt,
        high_confidence_receipt=args.high_confidence_receipt,
        terminal_win_receipt=args.terminal_win_receipt,
        deterministic_continuation_receipt=args.deterministic_continuation_receipt,
        go_first_receipt=args.go_first_receipt,
        output=args.output,
        r225_contract=args.r225_contract,
        canonical_libcg_contract=args.canonical_libcg_contract,
        entrypoint_member=args.entrypoint_member,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
