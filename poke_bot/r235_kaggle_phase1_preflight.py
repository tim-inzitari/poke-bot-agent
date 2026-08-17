"""Offline R235/R242 package, resource, startup, and throughput preflight.

The R235 replacement is a two-lane Phase-1 diagnostic.  This module validates
an explicit staged directory, archive, and manifest without guessing historic
archive names.  It is deliberately unable to upload, queue, start a service,
or mutate a selector.  An actual probe runs only through the exact-child
watchdog; dry mode consumes a supplied JSON fixture and is visibly ineligible
as an execution gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import ctypes
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import stat
import tarfile
import tempfile
import time
from typing import Any, Callable

from poke_bot.r235_exact_child_watchdog import (
    ExactChildOutcome,
    R235ExactChildWatchdog,
)
from poke_bot.r235_actual_gate_receipts import (
    R235ActualGateReceiptPaths,
    build_actual_gate_receipts,
    validate_output_paths as validate_actual_gate_receipt_paths,
    write_actual_gate_receipts,
)


ROOT = Path(__file__).resolve().parents[1]

RECEIPT_SCHEMA = "poke_bot.r235_r236_local_preflight_receipt/v1"
RECEIPT_NAME = "resource_memory_startup_and_throughput_preflight_receipt"
R225_SCHEMA = "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
R225_CANONICAL_SHA256 = (
    "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
)
R236_SCHEMA = "poke_bot.canonical_libcg_r236/v1"
R238_MANIFEST_SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
R238_MANIFEST_ROLE = "isolated_r238_two_lane_bounded_mcts_fallback_diagnostic"
R240_PROBE_SCHEMA = "poke_bot.r240_two_lane_resource_startup_throughput_probe/v1"
CUDA_RUNTIME_OBSERVATION_SCHEMA = "poke_bot.r238_cuda_runtime_observation/v1"
CUDA_RUNTIME_OBSERVATION_PHASE = "before_search"

R235_LABEL = "DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST"
COMPLETE_ACTION_CAP = 65_536
SIMULATOR_LANE_COUNT = 2
PHASE1_RESOURCES: dict[str, object] = {
    "hdd_gib": 11.8,
    "ram_gib": 12.2,
    "vcpus": 2,
    "archive_mib": 197.7,
}
PHASE1_MANIFEST_RESOURCE_BOUNDS: dict[str, object] = {
    **PHASE1_RESOURCES,
    # The published Phase-1 CPU/RAM/disk envelope says nothing about CUDA
    # visibility.  A hidden GPU is neither assumed nor rejected: the exact
    # packaged parent/child model observations record it before search.
    "gpu_environment_inferred_from_resource_envelope": False,
    "runtime_cuda_observation_required_before_search": True,
}
PHASE1_SUBMISSION_ENVIRONMENT: dict[str, object] = {
    "hdd_space_gib": 11.8,
    "ram_gib": 12.2,
    "vcpus": 2,
    "submission_archive_limit_mib": 197.7,
}
PHASE1_RESOURCE_SOURCE = {
    "owner_phase1_resource_revision": 238,
    "owner_hybrid_scheduling_revision": 240,
    "owner_high_confidence_threshold_revision": 242,
    "source": "R238 Phase-1 resource envelope retained by R240 scheduling as amended by R242",
    "typed_contract": "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
    "scope": "new_r235_replacement_package_and_its_fresh_local_preflight_only",
}

# R240 owns the bounded two-lane scheduler.  R242 supersedes only its
# high-confidence threshold.  Keep these literals here rather than accepting a
# caller-provided relaxation: the limits are owner contract, not a benchmark
# preference.  The receipt/probe keys retain ``r240`` for compatibility, but
# must bind the R242 owner revision and inclusive 0.80 threshold.
R240_OWNER_REVISION = 240
R242_OWNER_REVISION = 242
R244_OWNER_REVISION = 244
R246_OWNER_REVISION = 246
R242_HIGH_CONFIDENCE_THRESHOLD = 0.80
# Public aliases retain the established import surface while carrying the R242
# value.  New code should use the explicit R242 names when referring to the
# threshold owner.
R240_HIGH_CONFIDENCE_THRESHOLD = R242_HIGH_CONFIDENCE_THRESHOLD
R240_CHILD_SEARCH_HARD_SECONDS = 2.0
R240_PARENT_ACTION_HARD_SECONDS = 4.0
R240_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R240_STABLE_ROOT_LEADER_OBSERVATIONS = 3
R240_MAXIMUM_BACKUPS_PER_DECISION = 32
R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS = 8
R240_HYBRID_PROBE_KEY = "r240_hybrid_decision_preflight"
R242_ACTOR_BOUNDARY_PROBE_KEY = "actor_change_end_turn_boundary"
R246_TERMINAL_WIN_PROBE_KEY = "synthetic_proven_deterministic_terminal_win_this_turn"
R246_TERMINAL_WIN_STOP_REASON = "proven_deterministic_terminal_win_this_turn"
R246_TERMINAL_WIN_PROOF_KIND = "exact_deterministic_simulator_terminal_win_this_turn"
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
R246_TERMINAL_WIN_CONTRACT: dict[str, object] = {
    "scope": "ambiguous_two_lane_mcts_for_new_r235_replacement_package_only",
    "owner_decision_revision": R246_OWNER_REVISION,
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
R240_HYBRID_SCHEDULER: dict[str, object] = {
    "high_confidence_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
    "all_selected_stages_finite": True,
    "immediate_no_child": True,
    "no_mcts_select_search_model_or_simulator_calls": True,
    "history_only_existing_child_journal_count_range": [0, 1],
    "high_confidence_degraded": False,
    "child_search_seconds": R240_CHILD_SEARCH_HARD_SECONDS,
    "parent_action_deadline_seconds": R240_PARENT_ACTION_HARD_SECONDS,
    "minimum_backups_before_stability": R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
    "stable_root_leader_observations": R240_STABLE_ROOT_LEADER_OBSERVATIONS,
    "maximum_backups_per_decision": R240_MAXIMUM_BACKUPS_PER_DECISION,
    "early_stop_requires_both_lanes_progressed": True,
    "stop_reason_required": True,
}
DETERMINISTIC_CONTINUATION: dict[str, object] = {
    "max_depth": R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS,
    "exact_observation_fingerprint_required": True,
    "both_lanes_same_fingerprint_and_backed_action_required": True,
    "same_root_actor_required": True,
    "chance_or_boundary_forbidden": True,
    "no_new_search_on_valid_match": True,
    "mismatch_clears_entire_plan": True,
}
# The staged R238 manifest deliberately carries a richer nested contract than
# the flat receipt identity above.  Bind its full semantic shape directly
# instead of flattening the package declaration and accepting a stale variant.
R240_MANIFEST_HYBRID_SCHEDULER: dict[str, object] = {
    "scope": "new_r235_replacement_package_only",
    "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_REVISION,
    "selected_factorized_stage_probability_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
    "threshold_comparison": (
        "every selected factorized-stage probability is finite and "
        "greater_than_or_equal_to_0.80"
    ),
    "all_selected_factorized_stages_must_meet_threshold": True,
    "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
    "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
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
    "missing_malformed_nonfinite_or_below_threshold_confidence_routes_to_mcts": True,
    "two_lane_mcts_topology_backup_and_stop_contract_applies_only_when_confidence_routes_to_ambiguous_mcts": True,
    "ambiguous_mcts_exact_simulator_search_lane_count": SIMULATOR_LANE_COUNT,
    "child_search_hard_seconds": R240_CHILD_SEARCH_HARD_SECONDS,
    "parent_action_hard_seconds": R240_PARENT_ACTION_HARD_SECONDS,
    "adaptive_early_stop_min_completed_backups": R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
    "adaptive_early_stop_stable_deterministic_root_leader_observations": R240_STABLE_ROOT_LEADER_OBSERVATIONS,
    "adaptive_early_stop_both_lanes_progressed_required": True,
    "hard_completed_backup_stop": R240_MAXIMUM_BACKUPS_PER_DECISION,
    "mcts_simulated_rollout_expansion_stops_at_terminal_chance_boundary_or_actor_change_away_from_root_seat": True,
    "root_actor_change_away_from_our_seat_leaf_is_value_evaluated_without_expanded_legal_actions_or_children": True,
    "mcts_opponent_action_selection_or_planning_allowed": False,
    "r246_proven_deterministic_terminal_win_this_turn": dict(R246_TERMINAL_WIN_CONTRACT),
    "stop_reason_fields": [
        "high_confidence_frozen_direct",
        "deterministic_continuation_plan",
        R246_TERMINAL_WIN_STOP_REASON,
        "adaptive_early_stop",
        "hard_completed_backup_stop",
        "child_search_hard_deadline",
        "parent_action_hard_deadline",
        "zero_backup_precomputed_direct_fallback",
        "terminal_win_proof",
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
    "high_confidence_receipt_required_values": {
        "selected_factorized_stage_probability_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
        "all_selected_factorized_stages_meet_threshold": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "history_only_existing_child_journal_count_range": [0, 1],
        "degraded": False,
    },
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
    ],
    "boundary_leaf_receipt_required_fields": [
        "actor_change_boundary_leaf_count",
        "chance_boundary_leaf_count",
        "boundary_leaf_count",
    ],
    "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
}
R240_MANIFEST_DETERMINISTIC_CONTINUATION: dict[str, object] = {
    "scope": "optional_receipt_carried_plan_for_new_r235_replacement_package_only",
    "maximum_depth": R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS,
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


def _floor_unit_bytes(amount: str, unit_bytes: int) -> int:
    return int((Decimal(amount) * Decimal(unit_bytes)).to_integral_value(rounding=ROUND_FLOOR))


PHASE1_STAGE_RUNTIME_MAX_BYTES = _floor_unit_bytes("11.8", 1024**3)
PHASE1_COMBINED_RSS_MAX_BYTES = _floor_unit_bytes("12.2", 1024**3)
PHASE1_ARCHIVE_MAX_BYTES = _floor_unit_bytes("197.7", 1024**2)

NativeLoader = Callable[[str], object]


class R235PreflightError(RuntimeError):
    """One fail-closed R235/R238 preflight requirement did not validate."""


class ImmutableReceiptError(R235PreflightError):
    """A receipt target is unsafe or already exists."""


class R235PreflightFailure(R235PreflightError):
    """A failure receipt was written before the preflight raised."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any], path: Path) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)
        self.path = path


@dataclass(frozen=True)
class R235PreflightLimits:
    """Caller-declared timing/throughput ceilings; no hidden timing defaults."""

    probe_timeout_seconds: float
    term_grace_seconds: float
    kill_grace_seconds: float
    max_startup_seconds: float
    max_decision_latency_seconds: float
    max_full_game_cumulative_seconds: float
    min_throughput_decisions_per_second: float
    min_throughput_decision_count: int

    def __post_init__(self) -> None:
        for field in (
            "probe_timeout_seconds",
            "term_grace_seconds",
            "kill_grace_seconds",
            "max_startup_seconds",
            "max_decision_latency_seconds",
            "max_full_game_cumulative_seconds",
            "min_throughput_decisions_per_second",
        ):
            value = float(getattr(self, field))
            if not value > 0.0 or not math.isfinite(value):
                raise ValueError(f"{field} must be a positive finite number")
        if (
            not isinstance(self.min_throughput_decision_count, int)
            or self.min_throughput_decision_count <= 0
        ):
            raise ValueError("min_throughput_decision_count must be a positive integer")
        if self.max_decision_latency_seconds > R240_PARENT_ACTION_HARD_SECONDS:
            raise ValueError(
                "max_decision_latency_seconds cannot exceed the R240 4.0-second parent cap"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "probe_timeout_seconds": self.probe_timeout_seconds,
            "term_grace_seconds": self.term_grace_seconds,
            "kill_grace_seconds": self.kill_grace_seconds,
            "max_startup_seconds": self.max_startup_seconds,
            "max_decision_latency_seconds": self.max_decision_latency_seconds,
            "max_full_game_cumulative_seconds": self.max_full_game_cumulative_seconds,
            "min_throughput_decisions_per_second": self.min_throughput_decisions_per_second,
            "min_throughput_decision_count": self.min_throughput_decision_count,
        }


@dataclass(frozen=True)
class R235PreflightInputs:
    """All package paths are explicit; archive and manifest names are opaque."""

    stage_dir: Path
    archive_path: Path
    manifest_path: Path
    expected_archive_sha256: str
    expected_manifest_sha256: str
    receipt_path: Path
    r225_contract_path: Path = ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
    r236_contract_path: Path = ROOT / "state/canonical-libcg-r236.json"
    entrypoint_relative_path: str = "main.py"
    actual_gate_receipts_dir: Path | None = None


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise R235PreflightError(f"{label} must be a sha256: digest")
    hex_value = value.removeprefix("sha256:")
    if len(hex_value) != 64 or any(character not in "0123456789abcdef" for character in hex_value):
        raise R235PreflightError(f"{label} must be lowercase SHA-256")
    return value


def _physical_directory(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise R235PreflightError(f"{label} must not be a symlink")
    resolved = raw.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise R235PreflightError(f"{label} must be an existing physical directory")
    return resolved


def _regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise R235PreflightError(f"{label} must not be a symlink")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise R235PreflightError(f"{label} must be an existing physical regular file")
    return resolved


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235PreflightError(f"{label} is not a readable JSON object") from exc
    if not isinstance(payload, dict):
        raise R235PreflightError(f"{label} must be a JSON object")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R235PreflightError(f"{label} must be an object")
    return value


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if value != expected:
        raise R235PreflightError(f"{label} is not the required value")


def _require_expected_fields(
    observed: Mapping[str, Any], expected: Mapping[str, object], *, label: str
) -> None:
    for field, value in expected.items():
        _require_exact(observed.get(field), value, label=f"{label} {field}")


def _safe_relative_path(value: str, *, label: str) -> str:
    candidate = PurePosixPath(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise R235PreflightError(f"{label} must be a safe relative POSIX path")
    return candidate.as_posix()


def _stage_members(stage_dir: Path) -> tuple[list[dict[str, object]], int, int]:
    """Hash every physical staged member and reject links/special files."""

    members: list[dict[str, object]] = []
    total_logical_bytes = 0
    total_allocated_bytes = 0

    def visit(directory: Path) -> None:
        nonlocal total_logical_bytes, total_allocated_bytes
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                entry_path = Path(entry.path)
                relative = entry_path.relative_to(stage_dir).as_posix()
                if entry.is_symlink():
                    raise R235PreflightError(f"stage contains symlinked member: {relative}")
                if entry.is_dir(follow_symlinks=False):
                    visit(entry_path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise R235PreflightError(f"stage contains non-regular member: {relative}")
                member_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(member_stat.st_mode):
                    raise R235PreflightError(f"stage member is not a regular file: {relative}")
                size = int(member_stat.st_size)
                allocated = int(getattr(member_stat, "st_blocks", 0)) * 512
                total_logical_bytes += size
                total_allocated_bytes += allocated if allocated > 0 else size
                members.append(
                    {
                        "path": relative,
                        "sha256": sha256_file(entry_path),
                        "size_bytes": size,
                        "mode": stat.S_IMODE(member_stat.st_mode),
                    }
                )

    visit(stage_dir)
    return members, total_logical_bytes, total_allocated_bytes


def _archive_members(archive_path: Path) -> list[dict[str, object]]:
    """Read safe archive member hashes without extracting package bytes."""

    observed: list[dict[str, object]] = []
    try:
        archive = tarfile.open(archive_path, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise R235PreflightError("package archive is not a readable tar archive") from exc
    with archive:
        seen: set[str] = set()
        for member in archive.getmembers():
            name = member.name.removeprefix("./").strip("/")
            name = _safe_relative_path(name, label="archive member")
            if name in seen:
                raise R235PreflightError("package archive has duplicate member paths")
            seen.add(name)
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise R235PreflightError("package archive contains a linked or non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise R235PreflightError("package archive member cannot be read")
            digest = hashlib.sha256()
            observed_size = 0
            try:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    observed_size += len(block)
            finally:
                source.close()
            if observed_size != int(member.size):
                raise R235PreflightError("package archive member has a truncated body")
            observed.append(
                {
                    "path": name,
                    "sha256": "sha256:" + digest.hexdigest(),
                    "size_bytes": observed_size,
                    "mode": stat.S_IMODE(member.mode),
                }
            )
    return sorted(observed, key=lambda item: str(item["path"]))


def _member_manifest_sha256(members: Sequence[Mapping[str, object]]) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "schema": "poke_bot.r235_r238_archive_member_manifest/v1",
                "members": [dict(member) for member in members],
            }
        )
    )


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys_platform_is_darwin() else value * 1024


def sys_platform_is_darwin() -> bool:
    # Keep platform detection local and trivial; no process inspection is used.
    return os.uname().sysname == "Darwin"


def _finite_number(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R235PreflightError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise R235PreflightError(f"{label} is outside its allowed range")
    return parsed


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R235PreflightError(f"{label} must be a nonnegative integer")
    return int(value)


def _validate_r236_contract(payload: Mapping[str, Any]) -> dict[str, object]:
    _require_exact(payload.get("schema"), R236_SCHEMA, label="r236 schema")
    _require_exact(payload.get("owner_decision_revision"), 236, label="r236 owner revision")
    native = _mapping(payload.get("canonical_native_libraries"), label="r236 native libraries")
    linux = _mapping(native.get("linux_x86_64"), label="r236 Linux native library")
    required_exports = payload.get("required_native_exports")
    if not isinstance(required_exports, list) or not all(
        isinstance(value, str) for value in required_exports
    ):
        raise R235PreflightError("r236 required native exports are malformed")
    expected_exports = {
        "AgentStart",
        "BattleStart",
        "SearchBegin",
        "SearchStep",
        "SearchRelease",
        "SearchEnd",
    }
    if not expected_exports.issubset(set(required_exports)):
        raise R235PreflightError("r236 omits a required R235 native export")
    _require_exact(linux.get("package_relative_path"), "cg/libcg.so", label="r236 Linux path")
    digest = _require_sha256(linux.get("sha256"), label="r236 Linux digest")
    size = _nonnegative_int(linux.get("size_bytes"), label="r236 Linux size")
    wheel = _mapping(payload.get("upstream_provenance"), label="r236 wheel provenance")
    wheel_sha = _require_sha256(wheel.get("wheel_sha256"), label="r236 wheel digest")
    _require_exact(wheel.get("package_version"), "1.32.6", label="r236 package version")
    return {
        "linux_relative_path": "cg/libcg.so",
        "linux_sha256": digest,
        "linux_size_bytes": size,
        "wheel_sha256": wheel_sha,
        "required_exports": sorted(expected_exports),
    }


def _validate_r225_r238_contract(payload: Mapping[str, Any]) -> dict[str, object]:
    _require_exact(payload.get("schema"), R225_SCHEMA, label="r225 schema")
    _require_exact(
        payload.get("owner_decision_revision"), R246_OWNER_REVISION, label="r225 R246 owner revision"
    )
    _require_exact(
        payload.get("owner_phase1_submission_resources_and_two_lane_revision"),
        238,
        label="r225 R238 two-lane revision",
    )
    _require_exact(
        payload.get("owner_hybrid_confidence_bounded_mcts_revision"),
        R240_OWNER_REVISION,
        label="r225 R240 hybrid scheduling revision",
    )
    _require_exact(
        payload.get("owner_high_confidence_frozen_direct_threshold_revision"),
        R242_OWNER_REVISION,
        label="r225 R242 threshold revision",
    )
    _require_exact(
        payload.get("owner_handle_scoped_search_id_revision"),
        R244_OWNER_REVISION,
        label="r225 R244 handle-scoped SearchId revision",
    )
    _require_exact(
        payload.get("owner_proven_deterministic_terminal_win_this_turn_revision"),
        R246_OWNER_REVISION,
        label="r225 R246 terminal-win revision",
    )
    relationship = _mapping(
        payload.get("relationship_to_existing_work"), label="r225 relationship to existing work"
    )
    _require_exact(
        relationship.get(
            "r240_supersedes_only_the_new_r235_replacement_package_decision_scheduling_and_does_not_modify_r229_or_bo1000"
        ),
        True,
        label="r225 R240 replacement-only scope",
    )
    _require_exact(
        relationship.get(
            "r240_deterministic_continuation_is_parent_owned_and_does_not_change_the_r234_containment_boundary"
        ),
        True,
        label="r225 R240 continuation containment scope",
    )
    _require_exact(
        relationship.get(
            "r242_supersedes_only_the_r240_high_confidence_frozen_direct_threshold_for_the_new_r235_replacement_package"
        ),
        True,
        label="r225 R242 replacement-only threshold scope",
    )
    _require_exact(
        relationship.get(
            "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible"
        ),
        True,
        label="r225 R242 inclusive threshold and stale-preflight rejection",
    )
    _require_exact(
        relationship.get("r242_does_not_modify_r234_r236_r238_r235_continuation_or_r229_bo1000"),
        True,
        label="r225 R242 scope preservation",
    )
    _require_exact(
        relationship.get(
            "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229"
        ),
        True,
        label="r225 R244 handle-scoped SearchId scope",
    )
    _require_exact(
        relationship.get("r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries"),
        True,
        label="r225 R244 scope preservation",
    )
    _require_expected_fields(
        relationship,
        {
            "r246_supersedes_only_ambiguous_r235_mcts_root_selection_after_a_valid_deterministic_terminal_win_this_turn_proof": True,
            "r246_does_not_change_r242_high_confidence_direct_before_child_or_any_r229_bo1000_lifecycle": True,
        },
        label="r225 R246 terminal-win scope preservation",
    )
    phase1 = _mapping(
        payload.get("phase1_submission_environment"), label="r225 R238 Phase-1 environment"
    )
    expected_phase1 = {
        "owner_decision_revision": 238,
        "hdd_space_gib": 11.8,
        "ram_gib": 12.2,
        "vcpus": 2,
        "submission_archive_limit_mib": 197.7,
        "resource_probe_and_archive_size_receipt_required": True,
        "resource_mismatch_or_archive_over_limit_behavior": "hard_fail_closed_and_do_not_upload",
    }
    for field, expected in expected_phase1.items():
        _require_exact(phase1.get(field), expected, label=f"r225 Phase-1 {field}")
    replacement = _mapping(
        payload.get("replacement_kaggle_diagnostic"), label="r225 replacement diagnostic"
    )
    _require_exact(replacement.get("revision"), 235, label="r225 replacement revision")
    _require_exact(
        replacement.get("phase1_two_lane_resource_revision"),
        238,
        label="r225 replacement R238 revision",
    )
    topology = _mapping(
        replacement.get("phase1_simulator_search"), label="r225 replacement two-lane topology"
    )
    for field in (
        "required_simulator_search_lane_count",
        "required_active_simulator_search_lane_count",
        "required_internal_agent_start_simulator_search_arena_count",
        "required_search_begin_call_count",
        "required_unique_raw_handle_count",
        "required_distinct_per_lane_handle_identity_count",
        "required_distinct_handle_identity_first_search_id_composite_state_count",
    ):
        _require_exact(topology.get(field), 2, label=f"r225 replacement {field}")
    _require_expected_fields(
        topology,
        {
            "exactly_two_simulator_search_lanes_required": True,
            "per_lane_handle_scoped_search_id_chains_required": True,
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        },
        label="r225 R244 replacement handle-scoped SearchId contract",
    )
    _require_exact(
        topology.get("handle_scoped_first_search_id_composite_state_public_shape"),
        {
            "array_field": "handle_scoped_first_search_id_composite_states",
            "entry_exact_keys_in_order": [
                "lane_id",
                "handle_identity",
                "first_search_id",
            ],
            "lane_id_values": [0, 1],
            "handle_identity": "opaque AgentStart handle identity; exactly two distinct values",
            "first_search_id": (
                "nonnegative native SearchId scoped to the entry handle; raw values may repeat "
                "across distinct handles"
            ),
            "state_identity_composite": "(handle_identity, first_search_id)",
        },
        label="r225 R244 replacement public composite-state shape",
    )
    _require_exact(
        topology.get("one_lane_serial_fallback_eight_lane_topology_or_partial_lane_mcts_authority_allowed"),
        False,
        label="r225 replacement rejects legacy lane topology",
    )
    local = _mapping(payload.get("local_preflight"), label="r225 local preflight")
    for field in (
        "required_simulator_search_lane_count",
        "required_internal_agent_start_simulator_search_arena_count_per_child",
        "required_search_begin_call_count_per_ambiguous_mcts_decision",
    ):
        _require_exact(local.get(field), 2, label=f"r225 local {field}")
    _require_exact(
        local.get("exactly_two_simultaneous_in_decision_simulator_search_lanes_required"),
        True,
        label="r225 local two-lane requirement",
    )
    _require_expected_fields(
        local,
        {
            "required_search_begin_call_count_per_ambiguous_mcts_decision": 2,
            "required_distinct_internal_agent_start_handle_identity_count": 2,
            "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
            "per_lane_handle_scoped_search_id_chains_required": True,
        },
        label="r225 R244 handle-scoped SearchId contract",
    )
    r238_receipt_contract = _mapping(
        local.get("r238_two_lane_receipt_contract"), label="r225 R238/R244 topology receipt contract"
    )
    _require_exact(
        r238_receipt_contract.get("normal_mcts_decision_exact_counts"),
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
        label="r225 R244 topology exact counts",
    )
    _require_exact(
        r238_receipt_contract.get("normal_mcts_decision_per_lane_vectors_exact_length"),
        {
            "per_lane_depth": 2,
            "per_lane_search_id_chains": 2,
            "per_lane_handle_identities": 2,
            "per_lane_first_search_ids": 2,
            "handle_scoped_first_search_id_composite_states": 2,
        },
        label="r225 R244 topology per-lane vector lengths",
    )
    _require_exact(
        r238_receipt_contract.get("search_id_identity_contract"),
        {
            "numeric_namespace": "per_distinct_agent_start_handle",
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "first_search_id_identity_composite": "(handle_identity, first_search_id)",
            "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
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
        },
        label="r225 R244 SearchId identity contract",
    )
    r240_scheduler = _mapping(local.get("r240_hybrid_scheduler"), label="r225 R240 scheduler")
    _require_expected_fields(
        r240_scheduler,
        {
            "high_confidence_frozen_direct_threshold_owner_revision": R242_OWNER_REVISION,
            "selected_factorized_stage_probability_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
            "all_selected_factorized_stages_must_meet_threshold": True,
            "historical_r240_0_90_threshold_draft_and_preflight_are_ineligible": True,
            "high_confidence_direct_and_adaptive_bounded_mcts_regression_must_prove_r242_inclusive_0_80_threshold_and_reject_historical_0_90_draft_preflight": True,
            "high_confidence_frozen_direct_mode": "high_confidence_frozen_direct",
            "high_confidence_mcts_child_started_for_this_decision": False,
            "high_confidence_mcts_select_search_model_or_simulator_call_allowed": False,
            "high_confidence_existing_child_history_only_note_direct_action_ipc_allowed_and_required_when_child_exists": True,
            "high_confidence_existing_child_history_only_note_direct_action_ipc_count_range": [0, 1],
            "high_confidence_history_only_note_direct_action_ipc_must_not_invoke_mcts_select_search_model_or_simulator": True,
            "child_search_hard_seconds": R240_CHILD_SEARCH_HARD_SECONDS,
            "parent_action_hard_seconds": R240_PARENT_ACTION_HARD_SECONDS,
            "adaptive_early_stop_min_completed_backups": R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
            "adaptive_early_stop_stable_deterministic_root_leader_observations": R240_STABLE_ROOT_LEADER_OBSERVATIONS,
            "adaptive_early_stop_both_lanes_progressed_required": True,
            "hard_completed_backup_stop": R240_MAXIMUM_BACKUPS_PER_DECISION,
            "mcts_simulated_rollout_expansion_stops_at_terminal_chance_boundary_or_actor_change_away_from_root_seat": True,
            "root_actor_change_away_from_our_seat_leaf_is_value_evaluated_without_expanded_legal_actions_or_children": True,
            "mcts_opponent_action_selection_or_planning_allowed": False,
            "historical_r228_fixed_eight_second_branching_window_is_not_the_current_r235_budget": True,
        },
        label="r225 R240 scheduler as amended by R242",
    )
    _require_exact(
        r240_scheduler.get("boundary_leaf_receipt_required_fields"),
        [
            "actor_change_boundary_leaf_count",
            "chance_boundary_leaf_count",
            "boundary_leaf_count",
        ],
        label="r225 R242 boundary-leaf receipt fields",
    )
    _require_exact(
        r240_scheduler.get("high_confidence_receipt_required_values"),
        {
            "selected_factorized_stage_probability_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
            "all_selected_factorized_stages_meet_threshold": True,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "history_only_existing_child_journal_count_range": [0, 1],
            "degraded": False,
        },
        label="r225 R242 high-confidence receipt values",
    )
    _require_exact(
        r240_scheduler.get("r246_proven_deterministic_terminal_win_this_turn"),
        R246_TERMINAL_WIN_CONTRACT,
        label="r225 R246 terminal-win scheduler contract",
    )
    _require_exact(
        r240_scheduler.get("stop_reason_fields"),
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
        label="r225 R246 stop reasons",
    )
    r240_continuation = _mapping(
        local.get("deterministic_continuation"), label="r225 R240 deterministic continuation"
    )
    _require_expected_fields(
        r240_continuation,
        {
            "maximum_depth": R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS,
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
            "rewrite_history_to_actual_planned_action": True,
            "journal_exactly_once": True,
        },
        label="r225 R240 deterministic continuation",
    )
    _require_exact(
        r240_continuation.get("valid_plan_receipt_required_values"),
        {
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "history_only_existing_child_journal_count_range": [0, 1],
            "degraded": False,
        },
        label="r225 R242 deterministic-continuation receipt values",
    )
    canonical = _mapping(payload.get("canonical_libcg_revision"), label="r225 canonical libcg")
    _require_exact(canonical.get("typed_source"), "state/canonical-libcg-r236.json", label="r225 r236 path")
    return {
        "owner_decision_revision": R246_OWNER_REVISION,
        "phase1_resources": dict(PHASE1_RESOURCES),
        "simulator_search_lane_count": SIMULATOR_LANE_COUNT,
        "r240_hybrid_scheduling": {
            "high_confidence_threshold_owner_revision": R242_OWNER_REVISION,
            "high_confidence_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
            "child_search_hard_seconds": R240_CHILD_SEARCH_HARD_SECONDS,
            "parent_action_hard_seconds": R240_PARENT_ACTION_HARD_SECONDS,
            "minimum_backups_before_stability": R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
            "stable_root_leader_observations": R240_STABLE_ROOT_LEADER_OBSERVATIONS,
            "maximum_backups_per_decision": R240_MAXIMUM_BACKUPS_PER_DECISION,
            "maximum_deterministic_continuation_actions": R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS,
        },
        "r244_handle_scoped_search_id": {
            "numeric_namespace": "per_distinct_agent_start_handle",
            "globally_distinct_raw_search_id_integers_required": False,
            "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
            "distinct_handle_identity_first_search_id_composite_state_count": 2,
        },
        "r246_proven_deterministic_terminal_win_this_turn": dict(
            R246_TERMINAL_WIN_CONTRACT
        ),
    }


def _validate_r238_manifest(
    manifest: Mapping[str, Any],
    *,
    entrypoint_sha256: str,
    r236: Mapping[str, object],
) -> None:
    _require_exact(manifest.get("schema"), R238_MANIFEST_SCHEMA, label="stage manifest schema")
    _require_exact(manifest.get("role"), R238_MANIFEST_ROLE, label="stage manifest role")
    _require_exact(manifest.get("required_label"), R235_LABEL, label="stage manifest label")
    _require_exact(manifest.get("complete_action_cap"), COMPLETE_ACTION_CAP, label="stage manifest action cap")
    _require_exact(manifest.get("lane_count"), 2, label="stage manifest lane count")
    _require_exact(
        manifest.get("phase1_kaggle_resource_bounds"),
        {
            **PHASE1_MANIFEST_RESOURCE_BOUNDS,
            "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
        },
        label="stage manifest R238 Phase-1 resources",
    )
    _require_exact(
        manifest.get("entrypoint_sha256"), entrypoint_sha256, label="stage manifest entrypoint digest"
    )
    native = _mapping(manifest.get("canonical_native_members"), label="stage native members")
    linux = _mapping(native.get("cg/libcg.so"), label="stage Linux libcg member")
    _require_exact(linux.get("sha256"), r236["linux_sha256"], label="stage Linux libcg digest")
    _require_exact(linux.get("size_bytes"), r236["linux_size_bytes"], label="stage Linux libcg size")
    broker = _mapping(manifest.get("broker_contract"), label="stage broker contract")
    _require_exact(broker.get("complete_action_cap"), COMPLETE_ACTION_CAP, label="stage broker action cap")
    _require_exact(
        broker.get("search_seconds"),
        R240_CHILD_SEARCH_HARD_SECONDS,
        label="stage broker R240 child search cap",
    )
    _require_exact(
        broker.get("action_timeout_seconds"),
        R240_PARENT_ACTION_HARD_SECONDS,
        label="stage broker R240 parent action cap",
    )
    containment = _mapping(
        broker.get("subprocess_containment"), label="stage exact-child containment declaration"
    )
    _require_exact(
        containment.get("signals_exact_owned_child_only"),
        True,
        label="stage exact-child-only containment",
    )
    _require_exact(
        containment.get("process_group_or_session_signalling"),
        True,
        label="stage exact child session/process-group containment",
    )
    _require_exact(
        containment.get("bounded_reap_required"),
        True,
        label="stage bounded child reap",
    )
    scheduler = _mapping(manifest.get("r240_hybrid_scheduler"), label="stage R240 scheduler")
    _require_exact(
        dict(scheduler), R240_MANIFEST_HYBRID_SCHEDULER, label="stage R240 scheduler"
    )
    continuation = _mapping(
        manifest.get("deterministic_continuation"), label="stage deterministic continuation"
    )
    _require_exact(
        dict(continuation),
        R240_MANIFEST_DETERMINISTIC_CONTINUATION,
        label="stage deterministic continuation",
    )


def _validate_native_exports(
    libcg_path: Path,
    *,
    required_exports: Sequence[str],
    offline_exports: Sequence[str] | None,
    native_loader: NativeLoader,
) -> dict[str, object]:
    if offline_exports is not None:
        if not all(isinstance(name, str) and name for name in offline_exports):
            raise R235PreflightError("offline native exports must be nonempty strings")
        observed = set(offline_exports)
        source = "offline_fixture"
    else:
        try:
            handle = native_loader(str(libcg_path))
        except Exception as exc:
            raise R235PreflightError("cannot load staged Linux libcg for export validation") from exc
        observed = {name for name in required_exports if getattr(handle, name, None) is not None}
        source = "ctypes_loader"
    missing = sorted(set(required_exports) - observed)
    if missing:
        raise R235PreflightError("staged Linux libcg is missing required exports: " + ", ".join(missing))
    return {"source": source, "required_exports": list(required_exports), "observed_exports": sorted(observed)}


def _probe_payload_from_child(outcome: ExactChildOutcome) -> tuple[dict[str, Any], dict[str, object]]:
    if not outcome.completed:
        raise R235PreflightError(
            "R238 probe child did not complete cleanly: " + outcome.status
        )
    if outcome.stdout_truncated or outcome.stderr_truncated:
        raise R235PreflightError("R238 probe child exceeded bounded output capture")
    try:
        payload = json.loads(outcome.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R235PreflightError("R238 probe child did not emit one JSON object") from exc
    if not isinstance(payload, dict):
        raise R235PreflightError("R238 probe child JSON must be an object")
    return payload, {
        "execution_mode": "exact_child",
        "watchdog": outcome.as_dict(),
        "stdout_sha256": sha256_bytes(outcome.stdout),
        "stderr_sha256": sha256_bytes(outcome.stderr),
        "exact_child_peak_rss_bytes": outcome.exact_child_peak_rss_bytes,
        "exact_child_peak_rss_source": outcome.exact_child_peak_rss_source,
    }


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise R235PreflightError(f"{label} must be a boolean")
    return value


def _factorized_probabilities(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise R235PreflightError(f"{label} must be a nonempty probability list")
    probabilities = [
        _finite_number(item, label=f"{label}[{index}]", minimum=0.0)
        for index, item in enumerate(value)
    ]
    if any(probability > 1.0 for probability in probabilities):
        raise R235PreflightError(f"{label} contains a probability greater than one")
    return probabilities


def _action_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise R235PreflightError(f"{label} must be an action-index list")
    action: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise R235PreflightError(f"{label}[{index}] must be an integer action index")
        action.append(int(item))
    return action


def _fingerprint(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise R235PreflightError(f"{label} must be a nonempty exact fingerprint")
    return value


def _validate_handle_scoped_search_id_chains(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, object]:
    """Validate SearchIds in their raw-handle namespace, never globally.

    Official ``libcg`` may restart its raw SearchId sequence for every
    ``AgentStart`` handle.  Therefore ``[[0], [0]]`` is valid for two distinct
    handles, while two copies of the same ``(handle, first SearchId)`` tuple
    are a hard protocol failure.
    """

    raw_handles = payload.get("per_lane_handle_identities")
    if not isinstance(raw_handles, list) or len(raw_handles) != SIMULATOR_LANE_COUNT:
        raise R235PreflightError(f"{label} must expose exactly two per-lane handle identities")
    handles: list[int | str] = []
    for lane, handle in enumerate(raw_handles):
        if isinstance(handle, bool) or not isinstance(handle, (int, str)):
            raise R235PreflightError(f"{label} handle identity for lane {lane} is malformed")
        if isinstance(handle, str) and not handle:
            raise R235PreflightError(f"{label} handle identity for lane {lane} is empty")
        handles.append(handle)
    if len(set(handles)) != SIMULATOR_LANE_COUNT:
        raise R235PreflightError(f"{label} does not expose two distinct raw handles")

    raw_chains = payload.get("per_lane_search_id_chains")
    if not isinstance(raw_chains, list) or len(raw_chains) != SIMULATOR_LANE_COUNT:
        raise R235PreflightError(f"{label} must expose exactly two per-lane SearchId chains")
    chains: list[list[int]] = []
    for lane, raw_chain in enumerate(raw_chains):
        if not isinstance(raw_chain, list) or not raw_chain:
            raise R235PreflightError(f"{label} SearchId chain for lane {lane} is empty")
        chain = [
            _nonnegative_int(search_id, label=f"{label} SearchId lane {lane} index {index}")
            for index, search_id in enumerate(raw_chain)
        ]
        chains.append(chain)
    first_ids = [chain[0] for chain in chains]
    composites = [(handles[lane], first_ids[lane]) for lane in range(SIMULATOR_LANE_COUNT)]
    if len(set(composites)) != SIMULATOR_LANE_COUNT:
        raise R235PreflightError(
            f"{label} lacks two distinct handle-scoped first SearchId composites"
        )
    composite_projection = [
        {
            "lane_id": lane,
            "handle_identity": handles[lane],
            "first_search_id": first_ids[lane],
        }
        for lane in range(SIMULATOR_LANE_COUNT)
    ]
    _require_exact(
        payload.get("per_lane_first_search_ids"),
        first_ids,
        label=f"{label} per-lane first SearchIds",
    )
    _require_exact(
        payload.get("handle_scoped_first_search_id_composite_states"),
        composite_projection,
        label=f"{label} canonical handle-scoped first SearchId composite states",
    )
    return {
        "per_lane_handle_identities": list(handles),
        "per_lane_search_id_chains": chains,
        "per_lane_first_search_ids": first_ids,
        "handle_scoped_first_search_id_composite_states": composite_projection,
        # The immutable binding builder originally named this normalized
        # projection differently.  Retain it alongside the canonical R244
        # names; both are derived from the same exact handle-scoped tuples.
        "per_lane_handle_first_search_id_composites": composite_projection,
    }


def _r240_call_timing(
    payload: Mapping[str, Any],
    *,
    label: str,
    require_mcts: bool,
) -> dict[str, float]:
    """Validate one decision's exact R240 parent/child timing witness."""

    parent_elapsed = _finite_number(
        payload.get("parent_action_elapsed_seconds"),
        label=f"{label} parent action elapsed seconds",
        minimum=0.0,
    )
    if parent_elapsed > R240_PARENT_ACTION_HARD_SECONDS:
        raise R235PreflightError(f"{label} exceeded the R240 4.0-second parent action cap")
    if not require_mcts:
        return {"parent_action_elapsed_seconds": parent_elapsed}

    _require_exact(
        payload.get("child_search_budget_seconds"),
        R240_CHILD_SEARCH_HARD_SECONDS,
        label=f"{label} child search budget",
    )
    _require_exact(
        payload.get("parent_action_deadline_seconds"),
        R240_PARENT_ACTION_HARD_SECONDS,
        label=f"{label} parent action deadline",
    )
    child_elapsed = _finite_number(
        payload.get("child_search_elapsed_seconds"),
        label=f"{label} child search elapsed seconds",
        minimum=0.0,
    )
    if child_elapsed > R240_CHILD_SEARCH_HARD_SECONDS:
        raise R235PreflightError(f"{label} exceeded the R240 2.0-second child search cap")
    return {
        "parent_action_elapsed_seconds": parent_elapsed,
        "child_search_elapsed_seconds": child_elapsed,
    }


def _validate_r242_direct_no_search_calls(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, int]:
    """Bind the R242 direct path's only permitted existing-child interaction.

    A direct decision never starts a child or invokes MCTS/select/search/model/
    simulator work.  It may send one history-only ``note_direct_action`` IPC to
    an already-existing child, which is deliberately represented by a separate
    count instead of being conflated with a search call.
    """

    _require_exact(
        payload.get("mcts_child_started_for_this_decision"),
        False,
        label=f"{label} MCTS child started for this decision",
    )
    for field in (
        "mcts_select_call_count",
        "mcts_search_call_count",
        "mcts_model_call_count",
        "mcts_simulator_call_count",
    ):
        _require_exact(payload.get(field), 0, label=f"{label} {field}")
    history_only_count = _nonnegative_int(
        payload.get("history_only_existing_child_journal_count"),
        label=f"{label} history-only existing-child journal count",
    )
    if history_only_count > 1:
        raise R235PreflightError(
            f"{label} used more than one history-only note_direct_action IPC"
        )
    _require_exact(payload.get("degraded"), False, label=f"{label} degraded")
    return {
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "mcts_search_call_count": 0,
        "mcts_model_call_count": 0,
        "mcts_simulator_call_count": 0,
        "history_only_existing_child_journal_count": history_only_count,
        "degraded": False,
    }


def _validate_r240_two_lane_mcts_witness(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, object]:
    """Prove the ambiguous fixture used bounded, adaptive, exact two-lane MCTS."""

    probabilities = _factorized_probabilities(
        payload.get("selected_factorized_stage_probabilities"),
        label=f"{label} selected factorized stage probabilities",
    )
    if not any(probability < R240_HIGH_CONFIDENCE_THRESHOLD for probability in probabilities):
        raise R235PreflightError(f"{label} is not an R240 below-threshold ambiguous prompt")
    _require_exact(payload.get("mode"), "adaptive_two_lane_mcts", label=f"{label} mode")
    _require_exact(
        payload.get("stop_reason"), "adaptive_early_stop", label=f"{label} adaptive stop reason"
    )
    _require_exact(
        payload.get("terminal_win_proof"), None, label=f"{label} non-terminal proof absence"
    )
    for field in (
        "direct_action_precomputed_and_validated",
        "broker_started",
        "mcts_child_started",
        "mcts_child_called",
        "mcts_action_authority",
        "both_lanes_progressed",
    ):
        _require_exact(payload.get(field), True, label=f"{label} {field}")
    _require_exact(payload.get("degraded"), False, label=f"{label} degraded")
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "maximum_simulator_calls_in_flight",
    ):
        _require_exact(payload.get(field), SIMULATOR_LANE_COUNT, label=f"{label} {field}")
    completed_backups = _nonnegative_int(
        payload.get("completed_backups"), label=f"{label} completed backups"
    )
    if not R240_MINIMUM_BACKUPS_BEFORE_STABILITY <= completed_backups <= R240_MAXIMUM_BACKUPS_PER_DECISION:
        raise R235PreflightError(f"{label} completed backups are outside R240's 8..32 bounds")
    _require_exact(
        payload.get("minimum_backups_before_stability"),
        R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
        label=f"{label} minimum backups before stability",
    )
    _require_exact(
        payload.get("stable_root_leader_required_observations"),
        R240_STABLE_ROOT_LEADER_OBSERVATIONS,
        label=f"{label} stable root leader requirement",
    )
    _require_exact(
        payload.get("maximum_backups_per_decision"),
        R240_MAXIMUM_BACKUPS_PER_DECISION,
        label=f"{label} maximum backups per decision",
    )
    observations = payload.get("deterministic_root_leader_observations")
    if not isinstance(observations, list) or len(observations) < R240_STABLE_ROOT_LEADER_OBSERVATIONS:
        raise R235PreflightError(f"{label} lacks three deterministic root-leader observations")
    if any(not isinstance(item, str) or not item for item in observations):
        raise R235PreflightError(f"{label} root-leader observations must be nonempty strings")
    if len(set(observations[-R240_STABLE_ROOT_LEADER_OBSERVATIONS :])) != 1:
        raise R235PreflightError(f"{label} did not observe the same root leader three times")
    handle_scoped_search_ids = _validate_handle_scoped_search_id_chains(payload, label=label)
    actor_change_boundary_leaf_count = _nonnegative_int(
        payload.get("actor_change_boundary_leaf_count"),
        label=f"{label} actor-change boundary leaf count",
    )
    chance_boundary_leaf_count = _nonnegative_int(
        payload.get("chance_boundary_leaf_count"),
        label=f"{label} chance boundary leaf count",
    )
    boundary_leaf_count = _nonnegative_int(
        payload.get("boundary_leaf_count"),
        label=f"{label} total boundary leaf count",
    )
    if boundary_leaf_count != actor_change_boundary_leaf_count + chance_boundary_leaf_count:
        raise R235PreflightError(
            f"{label} boundary leaf total does not equal actor-change plus chance counts"
        )
    timing = _r240_call_timing(payload, label=label, require_mcts=True)
    return {
        "mode": "adaptive_two_lane_mcts",
        "stop_reason": "adaptive_early_stop",
        "selected_factorized_stage_probabilities": probabilities,
        "completed_backups": completed_backups,
        "deterministic_root_leader_observations": list(observations),
        "actor_change_boundary_leaf_count": actor_change_boundary_leaf_count,
        "chance_boundary_leaf_count": chance_boundary_leaf_count,
        "boundary_leaf_count": boundary_leaf_count,
        **handle_scoped_search_ids,
        **timing,
    }


def _validate_r246_terminal_win_witness(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, object]:
    """Validate one exact-stock terminal-win exception after two-lane startup.

    This intentionally proves a different stop class from R240's ordinary
    adaptive convergence.  Two lane handles/searches still initialize and
    clean, but the first backed deterministic terminal win need not wait for
    both lanes to make progress or for the normal 8/3 convergence threshold.
    """

    probabilities = _factorized_probabilities(
        payload.get("selected_factorized_stage_probabilities"),
        label=f"{label} selected factorized stage probabilities",
    )
    if not any(probability < R240_HIGH_CONFIDENCE_THRESHOLD for probability in probabilities):
        raise R235PreflightError(f"{label} is not an ambiguous R242 prompt")
    _require_exact(payload.get("mode"), "adaptive_two_lane_mcts", label=f"{label} mode")
    _require_exact(
        payload.get("stop_reason"), R246_TERMINAL_WIN_STOP_REASON, label=f"{label} stop reason"
    )
    for field, expected in (
        ("direct_action_precomputed_and_validated", True),
        ("broker_started", True),
        ("mcts_child_started", True),
        ("mcts_child_called", True),
        ("mcts_action_authority", True),
        ("degraded", False),
        ("two_lane_topology_initialized_before_terminal_win_override", True),
        ("terminal_win_proof_backed_up_into_shared_root_tree", True),
        ("terminal_leaf_returned_by_exact_stock_simulator", True),
        ("parent_validated_current_root_observation_legal_fingerprint_and_actor", True),
        ("all_owned_lane_resources_reservations_and_child_cleanup_complete", True),
        ("two_independent_lane_proofs_required", False),
        ("exhaustive_legal_action_scan_required", False),
        (
            "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof",
            False,
        ),
        ("proven_deterministic_terminal_win_this_turn", True),
    ):
        _require_exact(payload.get(field), expected, label=f"{label} {field}")
    _require_exact(
        payload.get("owner_proven_deterministic_terminal_win_this_turn_revision"),
        R246_OWNER_REVISION,
        label=f"{label} terminal-win owner revision",
    )
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_end_calls",
    ):
        _require_exact(payload.get(field), SIMULATOR_LANE_COUNT, label=f"{label} {field}")
    releases = _nonnegative_int(payload.get("search_release_calls"), label=f"{label} releases")
    if releases != SIMULATOR_LANE_COUNT:
        raise R235PreflightError(f"{label} did not release exactly two initialized searches")
    _require_exact(
        payload.get("outstanding_virtual_loss"), 0, label=f"{label} outstanding virtual loss"
    )
    backups = _nonnegative_int(payload.get("completed_backups"), label=f"{label} completed backups")
    if not 1 <= backups <= R240_MAXIMUM_BACKUPS_PER_DECISION:
        raise R235PreflightError(f"{label} terminal-win backup count is outside 1..32")
    _require_exact(
        payload.get("completed_root_backup_count"), backups, label=f"{label} root backup count"
    )
    _require_exact(payload.get("terminal_win_proof_count"), 1, label=f"{label} proof count")
    _require_exact(
        payload.get("proven_deterministic_terminal_win_this_turn_stop_count"),
        1,
        label=f"{label} terminal-win stop count",
    )
    for field, expected in (
        ("minimum_backups_before_stability", R240_MINIMUM_BACKUPS_BEFORE_STABILITY),
        ("stable_root_leader_required_observations", R240_STABLE_ROOT_LEADER_OBSERVATIONS),
        ("maximum_backups_per_decision", R240_MAXIMUM_BACKUPS_PER_DECISION),
    ):
        _require_exact(payload.get(field), expected, label=f"{label} {field}")
    _validate_handle_scoped_search_id_chains(payload, label=label)
    depths_raw = payload.get("per_lane_depth")
    if not isinstance(depths_raw, list) or len(depths_raw) != SIMULATOR_LANE_COUNT:
        raise R235PreflightError(f"{label} lacks two per-lane depths")
    depths = [_nonnegative_int(value, label=f"{label} lane depth") for value in depths_raw]
    if sum(depths) != backups or not any(depth >= 1 for depth in depths):
        raise R235PreflightError(f"{label} terminal-win depths do not bind its backup")
    timing = _r240_call_timing(payload, label=label, require_mcts=True)
    raw_proof = _mapping(payload.get("terminal_win_proof"), label=f"{label} terminal-win proof")
    if set(raw_proof) != set(R246_TERMINAL_WIN_PROOF_FIELDS):
        raise R235PreflightError(f"{label} terminal-win proof has an invalid exact schema")
    root_observation = _fingerprint(
        raw_proof.get("root_observation_fingerprint"), label=f"{label} root observation fingerprint"
    )
    root_legal = _fingerprint(
        raw_proof.get("root_legal_order_fingerprint"), label=f"{label} root legal fingerprint"
    )
    root_actor = raw_proof.get("root_actor_seat")
    if isinstance(root_actor, bool) or not isinstance(root_actor, int) or root_actor not in (0, 1):
        raise R235PreflightError(f"{label} terminal-win root actor is invalid")
    root_action = _action_list(raw_proof.get("root_action"), label=f"{label} root action")
    selected_action = _action_list(raw_proof.get("selected_action"), label=f"{label} selected action")
    if root_action != selected_action:
        raise R235PreflightError(f"{label} terminal-win root action differs from selected action")
    for field, expected in (
        ("proof_kind", R246_TERMINAL_WIN_PROOF_KIND),
        ("terminal_result", "win"),
        ("terminal_winner_seat", root_actor),
        ("terminal_leaf_reached", True),
        ("path_no_chance_boundary", True),
        ("path_no_actor_change_boundary", True),
        ("path_no_opponent_boundary_crossing", True),
        ("path_no_unresolved_randomness", True),
        ("proof_is_deterministic", True),
    ):
        _require_exact(raw_proof.get(field), expected, label=f"{label} terminal-win proof {field}")
    path_count = _nonnegative_int(
        raw_proof.get("proof_path_action_count"), label=f"{label} proof path count"
    )
    path_actors = raw_proof.get("path_actor_seats")
    if (
        path_count < 1
        or path_count > backups
        or not isinstance(path_actors, list)
        or len(path_actors) != path_count
        or any(
            isinstance(actor, bool) or not isinstance(actor, int) or actor != root_actor
            for actor in path_actors
        )
    ):
        raise R235PreflightError(f"{label} terminal-win proof path is not root-actor-only")
    discovering_lane = raw_proof.get("discovering_lane_id")
    if (
        isinstance(discovering_lane, bool)
        or not isinstance(discovering_lane, int)
        or discovering_lane not in (0, 1)
        or depths[discovering_lane] < path_count
    ):
        raise R235PreflightError(f"{label} terminal-win proof discovering lane is invalid")
    return {
        "mode": "adaptive_two_lane_mcts",
        "stop_reason": R246_TERMINAL_WIN_STOP_REASON,
        "selected_factorized_stage_probabilities": probabilities,
        "requested_simulator_lane_count": SIMULATOR_LANE_COUNT,
        "active_simulator_lane_count": SIMULATOR_LANE_COUNT,
        "arena_count": SIMULATOR_LANE_COUNT,
        "unique_handle_count": SIMULATOR_LANE_COUNT,
        "search_begin_calls": SIMULATOR_LANE_COUNT,
        "search_release_calls": releases,
        "search_end_calls": SIMULATOR_LANE_COUNT,
        "per_lane_depth": depths,
        "completed_backups": backups,
        "completed_root_backup_count": backups,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "outstanding_virtual_loss": 0,
        "owner_proven_deterministic_terminal_win_this_turn_revision": R246_OWNER_REVISION,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
        "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
        "terminal_win_proof": {
            **dict(raw_proof),
            "root_observation_fingerprint": root_observation,
            "root_legal_order_fingerprint": root_legal,
            "root_actor_seat": root_actor,
            "root_action": root_action,
            "selected_action": selected_action,
            "proof_path_action_count": path_count,
            "discovering_lane_id": discovering_lane,
            "path_actor_seats": list(path_actors),
        },
        **timing,
    }


def _validate_r240_high_confidence_witness(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, object]:
    """Prove an inclusive >=0.80 R242 prompt returns direct without search."""

    probabilities = _factorized_probabilities(
        payload.get("selected_factorized_stage_probabilities"),
        label=f"{label} selected factorized stage probabilities",
    )
    if any(probability < R240_HIGH_CONFIDENCE_THRESHOLD for probability in probabilities):
        raise R235PreflightError(f"{label} does not meet R242 at-every-stage confidence")
    _require_exact(
        payload.get("mode"), "high_confidence_frozen_direct", label=f"{label} mode"
    )
    _require_exact(
        payload.get("direct_action_precomputed_and_validated"),
        True,
        label=f"{label} direct action validation",
    )
    _require_exact(
        payload.get("selected_factorized_stage_probability_threshold"),
        R240_HIGH_CONFIDENCE_THRESHOLD,
        label=f"{label} selected-stage threshold",
    )
    _require_exact(
        payload.get("all_selected_factorized_stages_meet_threshold"),
        True,
        label=f"{label} all selected stages meet threshold",
    )
    direct_calls = _validate_r242_direct_no_search_calls(payload, label=label)
    timing = _r240_call_timing(payload, label=label, require_mcts=False)
    return {
        "mode": "high_confidence_frozen_direct",
        "selected_factorized_stage_probabilities": probabilities,
        "selected_factorized_stage_probability_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
        "all_selected_factorized_stages_meet_threshold": True,
        "direct_action_precomputed_and_validated": True,
        **direct_calls,
        **timing,
    }


def _event_mode_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "new_adaptive_two_lane_mcts": 0,
        "cached_deterministic_continuation": 0,
        "high_confidence_frozen_direct": 0,
    }
    for event in events:
        mode = event.get("mode")
        if mode not in counts:
            raise R235PreflightError("R240 full-game event has an unknown decision mode")
        counts[str(mode)] += 1
    return counts


def _validate_r242_actor_change_end_turn_boundary(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, int]:
    """Prove actor-change/end-turn leaves are value-only MCTS boundaries.

    The root agent may value an opponent-actor leaf, but cannot enumerate or
    select that opponent's legal action, create children, advance search beyond
    the boundary, or carry a deterministic continuation across it.
    """

    _require_exact(
        payload.get("actor_change_end_turn_boundary_regression_passed"),
        True,
        label=f"{label} regression passed",
    )
    leaves_raw = payload.get("opponent_actor_leaves")
    if not isinstance(leaves_raw, list) or not leaves_raw:
        raise R235PreflightError(f"{label} must contain at least one opponent-actor boundary leaf")
    leaves = [_mapping(item, label=f"{label} opponent-actor boundary leaf") for item in leaves_raw]
    for leaf_index, leaf in enumerate(leaves):
        for field, expected in (
            ("model_value_evaluated", True),
            ("opponent_action_selected_or_planned", False),
            ("opponent_action_cached", False),
        ):
            _require_exact(leaf.get(field), expected, label=f"{label} leaf {leaf_index} {field}")
        for field in (
            "expanded_legal_action_count",
            "expanded_child_count",
            "search_steps_beyond_boundary",
        ):
            _require_exact(leaf.get(field), 0, label=f"{label} leaf {leaf_index} {field}")

    expected_counts = {
        "declared_opponent_actor_leaf_count": len(leaves),
        "value_evaluated_opponent_actor_leaf_count": len(leaves),
        "expanded_legal_action_count": 0,
        "expanded_child_count": 0,
        "search_steps_beyond_boundary": 0,
        "opponent_action_selected_or_planned_count": 0,
        "opponent_action_cached_count": 0,
    }
    for field, expected in expected_counts.items():
        _require_exact(payload.get(field), expected, label=f"{label} {field}")
    return {
        "actor_change_end_turn_boundary_regression_passed": True,
        **expected_counts,
        "opponent_actor_leaves": [dict(leaf) for leaf in leaves],
    }


def _validate_r240_full_game_cumulative(
    payload: Mapping[str, Any], *, limits: R235PreflightLimits
) -> dict[str, object]:
    """Validate cumulative timing and deterministic continuation reuse evidence."""

    _require_exact(
        payload.get("phase1_full_game_budget_seconds"),
        limits.max_full_game_cumulative_seconds,
        label="R240 full-game Phase-1 budget",
    )
    cumulative_parent = _finite_number(
        payload.get("cumulative_parent_wall_seconds"),
        label="R240 cumulative parent wall seconds",
        minimum=0.0,
    )
    if cumulative_parent > limits.max_full_game_cumulative_seconds:
        raise R235PreflightError("R240 full-game cumulative parent time exceeds its Phase-1 budget")
    cumulative_child = _finite_number(
        payload.get("cumulative_child_search_seconds"),
        label="R240 cumulative child search seconds",
        minimum=0.0,
    )
    events_raw = payload.get("decision_events")
    if not isinstance(events_raw, list) or not events_raw:
        raise R235PreflightError("R240 full-game telemetry must contain decision events")
    events = [_mapping(event, label="R240 full-game decision event") for event in events_raw]
    counts = _event_mode_counts(events)
    declared_counts = {
        "new_adaptive_two_lane_mcts": _nonnegative_int(
            payload.get("new_mcts_search_count"), label="R240 new MCTS search count"
        ),
        "cached_deterministic_continuation": _nonnegative_int(
            payload.get("cached_deterministic_continuation_count"),
            label="R240 cached deterministic continuation count",
        ),
        "high_confidence_frozen_direct": _nonnegative_int(
            payload.get("high_confidence_frozen_direct_count"),
            label="R240 high-confidence direct count",
        ),
    }
    if declared_counts != counts:
        raise R235PreflightError("R240 full-game decision-mode counts do not match event telemetry")
    # This is a synthetic full-game throughput probe, so it deliberately covers
    # every R240 route and one reused deterministic plan rather than hoping a
    # random game happens to exercise them.
    if any(count <= 0 for count in counts.values()):
        raise R235PreflightError("R240 full-game probe did not exercise every required decision route")

    plans_raw = payload.get("deterministic_continuation_plans")
    if not isinstance(plans_raw, list) or not plans_raw:
        raise R235PreflightError("R240 full-game probe lacks deterministic continuation plans")
    planned_steps: dict[tuple[str, str, str], list[int]] = {}
    for plan_index, plan_raw in enumerate(plans_raw):
        plan = _mapping(plan_raw, label="R240 deterministic continuation plan")
        plan_id = _fingerprint(plan.get("plan_id"), label="R240 deterministic continuation plan id")
        turn_id = _fingerprint(plan.get("actual_turn_id"), label="R240 plan actual turn id")
        _require_exact(
            plan.get("extracted_from_mode"),
            "adaptive_two_lane_mcts",
            label="R240 plan extraction mode",
        )
        for field in (
            "exact_fingerprint_proven",
            "two_lane_agreed_backed_leader",
            "no_chance_boundary_or_opponent_transition",
        ):
            _require_exact(plan.get(field), True, label=f"R240 plan {field}")
        _require_exact(
            plan.get("crossed_actor_change_end_turn_boundary"),
            False,
            label="R242 plan actor-change/end-turn boundary crossing",
        )
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps or len(steps) > R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS:
            raise R235PreflightError("R240 deterministic continuation plan must contain 1..8 actions")
        for step_index, step_raw in enumerate(steps):
            step = _mapping(step_raw, label="R240 deterministic continuation step")
            fingerprint = _fingerprint(
                step.get("canonical_observation_fingerprint"),
                label="R240 planned observation fingerprint",
            )
            action = _action_list(step.get("planned_action"), label="R240 planned action")
            key = (plan_id, turn_id, fingerprint)
            if key in planned_steps:
                raise R235PreflightError(
                    f"R240 deterministic plan {plan_index} repeats an exact planned prompt"
                )
            planned_steps[key] = action

    observed_continuations: set[tuple[str, str, str]] = set()
    summed_parent_seconds = 0.0
    summed_child_seconds = 0.0
    for event in events:
        mode = str(event.get("mode"))
        turn_id = _fingerprint(event.get("actual_turn_id"), label="R240 event actual turn id")
        fingerprint = _fingerprint(
            event.get("canonical_observation_fingerprint"),
            label="R240 event observation fingerprint",
        )
        timing = _r240_call_timing(
            event,
            label=f"R240 full-game {mode} event",
            require_mcts=mode == "new_adaptive_two_lane_mcts",
        )
        summed_parent_seconds += float(timing["parent_action_elapsed_seconds"])
        if "child_search_elapsed_seconds" in timing:
            summed_child_seconds += float(timing["child_search_elapsed_seconds"])
        if mode == "high_confidence_frozen_direct":
            direct_probabilities = _factorized_probabilities(
                event.get("selected_factorized_stage_probabilities"),
                label="R242 full-game direct selected factorized stage probabilities",
            )
            if any(probability < R240_HIGH_CONFIDENCE_THRESHOLD for probability in direct_probabilities):
                raise R235PreflightError(
                    "R242 full-game direct event does not meet the inclusive 0.80 threshold"
                )
            _require_exact(
                event.get("selected_factorized_stage_probability_threshold"),
                R240_HIGH_CONFIDENCE_THRESHOLD,
                label="R242 full-game direct selected-stage threshold",
            )
            _require_exact(
                event.get("all_selected_factorized_stages_meet_threshold"),
                True,
                label="R242 full-game direct all selected stages meet threshold",
            )
            _validate_r242_direct_no_search_calls(event, label="R242 full-game direct event")
            continue
        if mode == "new_adaptive_two_lane_mcts":
            for field in ("broker_started", "mcts_child_started", "mcts_child_called", "new_mcts_search_started"):
                _require_exact(event.get(field), True, label=f"R240 MCTS event {field}")
            _require_exact(event.get("requested_simulator_lane_count"), 2, label="R240 MCTS lanes")
            _require_exact(event.get("active_simulator_lane_count"), 2, label="R240 MCTS active lanes")
            _validate_handle_scoped_search_id_chains(event, label="R244 full-game MCTS event")
            # A planned same-turn prompt may never trigger another full search.
            if any(
                plan_turn == turn_id and plan_fingerprint == fingerprint
                for _plan_id, plan_turn, plan_fingerprint in planned_steps
            ):
                raise R235PreflightError(
                    "R240 full-game telemetry repeated a full MCTS search for a planned same-turn prompt"
                )
            continue

        plan_id = _fingerprint(event.get("plan_id"), label="R240 continuation plan id")
        key = (plan_id, turn_id, fingerprint)
        planned_action = planned_steps.get(key)
        if planned_action is None:
            raise R235PreflightError("R240 continuation event is not an exact planned same-turn prompt")
        if key in observed_continuations:
            raise R235PreflightError("R240 continuation plan step was consumed more than once")
        for field in (
            "exact_fingerprint_match",
            "same_actor",
            "action_in_complete_legal_order",
            "two_lane_agreed_backed_leader",
            "no_chance_boundary_or_opponent_transition",
        ):
            _require_exact(event.get(field), True, label=f"R240 continuation event {field}")
        _require_exact(
            event.get("crossed_actor_change_end_turn_boundary"),
            False,
            label="R242 continuation actor-change/end-turn boundary crossing",
        )
        for field in ("new_mcts_search_started", "mcts_child_called"):
            _require_exact(event.get(field), False, label=f"R240 continuation event {field}")
        _validate_r242_direct_no_search_calls(event, label="R242 continuation event")
        if _action_list(event.get("selected_action"), label="R240 continuation selected action") != planned_action:
            raise R235PreflightError("R240 continuation did not use its exact planned action")
        observed_continuations.add(key)

    if observed_continuations != set(planned_steps):
        raise R235PreflightError("R240 full-game probe did not consume every declared deterministic plan step")
    if cumulative_parent + 1e-9 < summed_parent_seconds:
        raise R235PreflightError("R240 cumulative parent time is smaller than its decision telemetry")
    if cumulative_child + 1e-9 < summed_child_seconds:
        raise R235PreflightError("R240 cumulative child time is smaller than its MCTS telemetry")
    continuation_regression = _mapping(
        payload.get("deterministic_continuation_regression"),
        label="R240 deterministic-continuation mismatch regression",
    )
    for field in (
        "chance_disagreement_clears_entire_plan",
        "fingerprint_disagreement_clears_entire_plan",
        "action_disagreement_clears_entire_plan",
        "actor_disagreement_clears_entire_plan",
        "precomputed_direct_action_and_history_correction_retained",
    ):
        _require_exact(
            continuation_regression.get(field),
            True,
            label=f"R240 deterministic-continuation mismatch regression {field}",
        )
    return {
        "phase1_full_game_budget_seconds": limits.max_full_game_cumulative_seconds,
        "cumulative_parent_wall_seconds": cumulative_parent,
        "cumulative_child_search_seconds": cumulative_child,
        "decision_mode_counts": declared_counts,
        "deterministic_continuation_plan_count": len(plans_raw),
        "deterministic_continuation_step_count": len(planned_steps),
        # Preserve validated telemetry in the immutable receipt, including the
        # explicit no-boundary-crossing field on every continuation event.
        "deterministic_continuation_plans": [dict(plan) for plan in plans_raw],
        "decision_events": [dict(event) for event in events],
        "deterministic_continuation_regression": dict(continuation_regression),
    }


def _validate_r240_hybrid_probe(
    payload: Mapping[str, Any], *, limits: R235PreflightLimits
) -> dict[str, object]:
    """Validate R240/R242 routes plus the R246 terminal-win exception."""

    hybrid = _mapping(payload.get(R240_HYBRID_PROBE_KEY), label="R240 hybrid preflight")
    _require_exact(
        hybrid.get("owner_decision_revision"), R246_OWNER_REVISION, label="R246 owner revision"
    )
    configuration = _mapping(hybrid.get("configuration"), label="R242 hybrid configuration")
    expected_configuration = {
        "high_confidence_threshold_owner_revision": R242_OWNER_REVISION,
        "high_confidence_threshold": R240_HIGH_CONFIDENCE_THRESHOLD,
        "child_search_hard_seconds": R240_CHILD_SEARCH_HARD_SECONDS,
        "parent_action_hard_seconds": R240_PARENT_ACTION_HARD_SECONDS,
        "minimum_backups_before_stability": R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
        "stable_root_leader_observations": R240_STABLE_ROOT_LEADER_OBSERVATIONS,
        "maximum_backups_per_decision": R240_MAXIMUM_BACKUPS_PER_DECISION,
        "maximum_deterministic_continuation_actions": R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS,
        "legacy_fixed_eight_second_branching_windows_rejected": True,
        "historical_r240_0_90_threshold_draft_and_preflight_rejected": True,
        "proven_deterministic_terminal_win_this_turn_owner_revision": R246_OWNER_REVISION,
    }
    _require_exact(configuration, expected_configuration, label="R242 hybrid configuration")
    high_confidence = _validate_r240_high_confidence_witness(
        _mapping(
            hybrid.get("synthetic_high_confidence_direct"),
            label="R240 high-confidence synthetic prompt",
        ),
        label="R240 high-confidence synthetic prompt",
    )
    ambiguous = _validate_r240_two_lane_mcts_witness(
        _mapping(
            hybrid.get("synthetic_ambiguous_two_lane_mcts"),
            label="R240 ambiguous synthetic prompt",
        ),
        label="R240 ambiguous synthetic prompt",
    )
    terminal_win = _validate_r246_terminal_win_witness(
        _mapping(
            hybrid.get(R246_TERMINAL_WIN_PROBE_KEY),
            label="R246 deterministic-terminal-win synthetic prompt",
        ),
        label="R246 deterministic-terminal-win synthetic prompt",
    )
    actor_boundary = _validate_r242_actor_change_end_turn_boundary(
        _mapping(
            hybrid.get(R242_ACTOR_BOUNDARY_PROBE_KEY),
            label="R242 actor-change/end-turn boundary preflight",
        ),
        label="R242 actor-change/end-turn boundary preflight",
    )
    if (
        ambiguous["actor_change_boundary_leaf_count"]
        != actor_boundary["declared_opponent_actor_leaf_count"]
    ):
        raise R235PreflightError(
            "R242 actor-change boundary leaf count does not match ambiguous-MCTS telemetry"
        )
    if ambiguous["boundary_leaf_count"] < ambiguous["actor_change_boundary_leaf_count"]:
        raise R235PreflightError(
            "R242 ambiguous-MCTS boundary total is smaller than actor-change count"
        )
    for field in (
        "actor_change_boundary_leaf_count",
        "chance_boundary_leaf_count",
        "boundary_leaf_count",
    ):
        _require_exact(
            hybrid.get(field),
            ambiguous[field],
            label=f"R242 top-level {field}",
        )
    full_game = _validate_r240_full_game_cumulative(
        _mapping(hybrid.get("full_game_cumulative"), label="R240 full-game cumulative telemetry"),
        limits=limits,
    )
    return {
        "owner_decision_revision": R246_OWNER_REVISION,
        "configuration": dict(expected_configuration),
        "synthetic_high_confidence_direct": high_confidence,
        "synthetic_ambiguous_two_lane_mcts": ambiguous,
        R246_TERMINAL_WIN_PROBE_KEY: terminal_win,
        "actor_change_end_turn_boundary": actor_boundary,
        "actor_change_boundary_leaf_count": ambiguous["actor_change_boundary_leaf_count"],
        "chance_boundary_leaf_count": ambiguous["chance_boundary_leaf_count"],
        "boundary_leaf_count": ambiguous["boundary_leaf_count"],
        "full_game_cumulative": full_game,
    }


def _validate_cuda_runtime_before_search(value: object) -> dict[str, object]:
    """Validate one observation-only parent/child model CUDA receipt.

    Phase-1's reported CPU/RAM/disk limits cannot establish whether Kaggle
    exposes an accelerator.  This function deliberately accepts both a
    CPU-only and a CUDA-visible execution when their measured fields are
    internally consistent.  It rejects omitted/partial telemetry for a real
    preflight receipt, but it never selects a device or changes runtime action
    authority.
    """

    payload = _mapping(value, label="probe CUDA runtime observation")
    expected_keys = {
        "schema",
        "phase",
        "torch_imported",
        "cuda_available",
        "cuda_initialized",
        "device_count",
        "devices",
        "model_device",
        "telemetry_complete",
        "error_types",
    }
    if set(payload) != expected_keys:
        raise R235PreflightError(
            "probe CUDA runtime observation has unexpected or missing fields"
        )
    _require_exact(
        payload.get("schema"),
        CUDA_RUNTIME_OBSERVATION_SCHEMA,
        label="probe CUDA runtime observation schema",
    )
    _require_exact(
        payload.get("phase"),
        CUDA_RUNTIME_OBSERVATION_PHASE,
        label="probe CUDA runtime observation phase",
    )
    for field in (
        "torch_imported",
        "cuda_available",
        "cuda_initialized",
        "telemetry_complete",
    ):
        if not isinstance(payload.get(field), bool):
            raise R235PreflightError(f"probe CUDA {field} must be a boolean")
    if payload.get("torch_imported") is not True:
        raise R235PreflightError(
            "probe CUDA observation did not run after the frozen model loaded"
        )
    if payload.get("telemetry_complete") is not True:
        raise R235PreflightError("probe CUDA runtime observation is incomplete")
    errors = payload.get("error_types")
    if not isinstance(errors, list) or any(
        not isinstance(item, str) or not item for item in errors
    ):
        raise R235PreflightError("probe CUDA error type list is malformed")
    if errors:
        raise R235PreflightError("probe CUDA runtime observation reported errors")
    model_device = payload.get("model_device")
    if not isinstance(model_device, str) or not model_device:
        raise R235PreflightError("probe model device is missing")
    cuda_available = bool(payload["cuda_available"])
    cuda_initialized = bool(payload["cuda_initialized"])
    device_count = _nonnegative_int(
        payload.get("device_count"), label="probe CUDA device count"
    )
    devices = payload.get("devices")
    if not isinstance(devices, list):
        raise R235PreflightError("probe CUDA device rows must be a list")
    if not cuda_available:
        if device_count != 0 or devices or cuda_initialized:
            raise R235PreflightError(
                "probe reports initialized or enumerated CUDA while CUDA is unavailable"
            )
        if model_device.startswith("cuda"):
            raise R235PreflightError(
                "probe model uses CUDA while CUDA is unavailable"
            )
        return {
            "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
            "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
            "torch_imported": True,
            "cuda_available": False,
            "cuda_initialized": cuda_initialized,
            "device_count": 0,
            "devices": [],
            "model_device": model_device,
            "telemetry_complete": True,
            "error_types": [],
        }

    if device_count <= 0 or len(devices) != device_count:
        raise R235PreflightError(
            "CUDA-visible probe does not contain every visible device row"
        )
    normalized_devices: list[dict[str, object]] = []
    expected_device_keys = {
        "device_index",
        "device_name",
        "total_memory_bytes",
        "free_memory_bytes",
    }
    for expected_index, raw_device in enumerate(devices):
        device = _mapping(raw_device, label="probe CUDA device row")
        if set(device) != expected_device_keys:
            raise R235PreflightError(
                "probe CUDA device row has unexpected or missing fields"
            )
        _require_exact(
            _nonnegative_int(device.get("device_index"), label="probe CUDA device index"),
            expected_index,
            label="probe CUDA device index order",
        )
        name = device.get("device_name")
        if not isinstance(name, str) or not name:
            raise R235PreflightError("probe CUDA device name is missing")
        total = _nonnegative_int(
            device.get("total_memory_bytes"), label="probe CUDA total memory"
        )
        free = _nonnegative_int(
            device.get("free_memory_bytes"), label="probe CUDA free memory"
        )
        if total <= 0 or free > total:
            raise R235PreflightError("probe CUDA device memory range is invalid")
        normalized_devices.append(
            {
                "device_index": expected_index,
                "device_name": name,
                "total_memory_bytes": total,
                "free_memory_bytes": free,
            }
        )
    if model_device.startswith("cuda") and not cuda_initialized:
        raise R235PreflightError(
            "probe CUDA model device is selected but CUDA was not initialized"
        )
    if not model_device.startswith("cuda"):
        raise R235PreflightError(
            "CUDA-visible probe did not load the frozen model onto CUDA"
        )
    return {
        "schema": CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": True,
        "cuda_initialized": cuda_initialized,
        "device_count": device_count,
        "devices": normalized_devices,
        "model_device": model_device,
        "telemetry_complete": True,
        "error_types": [],
    }


def _validate_probe(
    payload: Mapping[str, Any],
    *,
    limits: R235PreflightLimits,
    stage_disk_bytes: int,
    parent_peak_rss_bytes: int,
    exact_child_peak_rss_bytes: int | None,
) -> dict[str, object]:
    _require_exact(payload.get("schema"), R240_PROBE_SCHEMA, label="probe schema")
    r240_hybrid = _validate_r240_hybrid_probe(payload, limits=limits)
    resource_probe = _mapping(payload.get("observed_resource_probe"), label="probe resources")
    runtime_disk_bytes = _nonnegative_int(
        resource_probe.get("runtime_disk_bytes"), label="probe runtime disk bytes"
    )
    if stage_disk_bytes > PHASE1_STAGE_RUNTIME_MAX_BYTES:
        raise R235PreflightError("staged package exceeds the R238 11.8 GiB disk budget")
    if runtime_disk_bytes > PHASE1_STAGE_RUNTIME_MAX_BYTES:
        raise R235PreflightError("runtime package exceeds the R238 11.8 GiB disk budget")
    child_peak_rss = _nonnegative_int(
        resource_probe.get("child_peak_rss_bytes"), label="probe child peak RSS"
    )
    if exact_child_peak_rss_bytes is not None:
        if child_peak_rss < exact_child_peak_rss_bytes:
            raise R235PreflightError(
                "probe child RSS understates exact-child /proc RSS evidence"
            )
        child_peak_rss = max(child_peak_rss, exact_child_peak_rss_bytes)
    combined_peak_rss = parent_peak_rss_bytes + child_peak_rss
    if combined_peak_rss > PHASE1_COMBINED_RSS_MAX_BYTES:
        raise R235PreflightError("combined parent and exact child RSS exceeds 12.2 GiB")

    target = _mapping(resource_probe.get("phase1_target"), label="probe Phase-1 target")
    _require_exact(target, PHASE1_MANIFEST_RESOURCE_BOUNDS, label="probe Phase-1 target")
    runtime = _mapping(resource_probe.get("runtime"), label="probe runtime topology")
    exact_two = {
        "configured_vcpus": 2,
        "configured_simulator_lane_count": 2,
        "maximum_simulator_lanes": 2,
        "observed_active_simulator_lane_count": 2,
        "receipt_lane_count": 2,
    }
    for field, expected in exact_two.items():
        _require_exact(runtime.get(field), expected, label=f"probe runtime {field}")
    _require_exact(runtime.get("receipt_schema"), R238_MANIFEST_SCHEMA, label="probe receipt schema")
    for field in (
        "worker_thread_count",
        "observed_peak_worker_threads",
        "maximum_simulator_calls_in_flight",
    ):
        observed = _nonnegative_int(runtime.get(field), label=f"probe runtime {field}")
        if observed > SIMULATOR_LANE_COUNT:
            raise R235PreflightError(
                f"probe runtime oversubscribed the two-vCPU budget through {field}"
            )
    # A legacy eight-lane receipt can never be accepted by omission: every
    # declared lane count is exact and the manifest schema is R238.
    for field, value in runtime.items():
        if "lane" in str(field).lower() and isinstance(value, int) and value == 8:
            raise R235PreflightError("legacy eight-lane runtime/receipt evidence is forbidden")

    cuda_runtime_before_search = _validate_cuda_runtime_before_search(
        resource_probe.get("cuda_runtime_before_search")
    )

    startup_seconds = _finite_number(
        payload.get("startup_seconds"), label="probe startup seconds", minimum=0.0
    )
    if startup_seconds > limits.max_startup_seconds:
        raise R235PreflightError("probe startup exceeded the declared startup ceiling")
    latency = _mapping(payload.get("decision_latency_seconds"), label="probe decision latency")
    latency_samples = _nonnegative_int(latency.get("sample_count"), label="probe latency samples")
    if latency_samples <= 0:
        raise R235PreflightError("probe decision latency has no samples")
    p50 = _finite_number(latency.get("p50"), label="probe latency p50", minimum=0.0)
    p95 = _finite_number(latency.get("p95"), label="probe latency p95", minimum=0.0)
    maximum = _finite_number(latency.get("max"), label="probe latency max", minimum=0.0)
    if not p50 <= p95 <= maximum:
        raise R235PreflightError("probe decision latency quantiles are inconsistent")
    if maximum > limits.max_decision_latency_seconds:
        raise R235PreflightError("probe decision latency exceeded the declared ceiling")
    throughput = _mapping(payload.get("throughput"), label="probe throughput")
    count = _nonnegative_int(throughput.get("decision_count"), label="probe decision count")
    elapsed = _finite_number(
        throughput.get("elapsed_seconds"), label="probe throughput elapsed", minimum=0.0
    )
    reported_rate = _finite_number(
        throughput.get("decisions_per_second"), label="probe throughput rate", minimum=0.0
    )
    if elapsed <= 0.0 or count < limits.min_throughput_decision_count:
        raise R235PreflightError("probe throughput did not execute the declared bounded decision count")
    calculated_rate = float(count) / elapsed
    if not math.isclose(reported_rate, calculated_rate, rel_tol=1e-6, abs_tol=1e-9):
        raise R235PreflightError("probe throughput rate does not match decision count and elapsed time")
    if elapsed > limits.probe_timeout_seconds:
        raise R235PreflightError("probe throughput exceeded the hard probe deadline")
    if reported_rate < limits.min_throughput_decisions_per_second:
        raise R235PreflightError("probe throughput is below the declared minimum")
    return {
        "observed_resource_probe": dict(resource_probe),
        "startup_seconds": startup_seconds,
        "decision_latency_seconds": {
            "sample_count": latency_samples,
            "p50": p50,
            "p95": p95,
            "max": maximum,
        },
        "throughput_decision_count": count,
        "throughput_elapsed_seconds": elapsed,
        "throughput_decisions_per_second": reported_rate,
        "stage_disk_bytes": stage_disk_bytes,
        "runtime_disk_bytes": runtime_disk_bytes,
        "parent_peak_rss_bytes": parent_peak_rss_bytes,
        "child_peak_rss_bytes": child_peak_rss,
        "exact_child_peak_rss_bytes": exact_child_peak_rss_bytes,
        "combined_parent_child_peak_rss_bytes": combined_peak_rss,
        # This is measured runtime telemetry, never an inference from the
        # R238 CPU/RAM/disk envelope.  GPU and CPU observations are both
        # valid, so callers can diagnose a hidden Kaggle GPU without changing
        # the frozen model's authority or device-selection behavior.
        "cuda_runtime_before_search": cuda_runtime_before_search,
        "cuda_available_on_probe_host": cuda_runtime_before_search["cuda_available"],
        "cuda_device_count_on_probe_host": cuda_runtime_before_search["device_count"],
        "r240_hybrid_decision_preflight": r240_hybrid,
    }


def write_once_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish immutable JSON without replacing an existing receipt."""

    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise ImmutableReceiptError("receipt target already exists; refusing overwrite")
    # Resolve a normal platform-owned path alias (for example macOS ``/var``)
    # once, but never replace a target or accept a final symlinked receipt.
    target = raw.resolve()
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ImmutableReceiptError("receipt parent must be a physical existing directory")
    body = canonical_json(dict(payload))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        try:
            written = 0
            while written < len(body):
                written += os.write(descriptor, body[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ImmutableReceiptError(
                "receipt target appeared during atomic publication; refusing overwrite"
            ) from exc
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _failure_receipt(
    *,
    inputs: R235PreflightInputs,
    limits: R235PreflightLimits,
    error: Exception,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "receipt_name": RECEIPT_NAME,
        "status": "failed",
        "passed": False,
        "immutable": True,
        "write_once": True,
        "execution_mode": "dry_offline" if dry_run else "exact_child",
        "r238_phase1_resource_revision": 238,
        "r240_owner_revision": R240_OWNER_REVISION,
        "r242_high_confidence_threshold_owner_revision": R242_OWNER_REVISION,
        "r244_handle_scoped_search_id_owner_revision": R244_OWNER_REVISION,
        "r246_proven_deterministic_terminal_win_this_turn_owner_revision": R246_OWNER_REVISION,
        "expected_canonical_r225_contract_sha256": R225_CANONICAL_SHA256,
        "phase1_resources": dict(PHASE1_RESOURCES),
        "phase1_submission_environment": dict(PHASE1_SUBMISSION_ENVIRONMENT),
        "phase1_resource_source": dict(PHASE1_RESOURCE_SOURCE),
        "r240_hybrid_scheduler": dict(R240_HYBRID_SCHEDULER),
        "deterministic_continuation": dict(DETERMINISTIC_CONTINUATION),
        "requested_paths": {
            "stage_dir": str(inputs.stage_dir),
            "archive": str(inputs.archive_path),
            "manifest": str(inputs.manifest_path),
        },
        "declared_limits": limits.as_dict(),
        "failure": {"type": type(error).__name__, "message": str(error)},
        "kaggle_api_called": False,
        "kaggle_upload_used": False,
        "kaggle_queue_used": False,
        "upload_readiness_claim": False,
    }


def run_r235_phase1_preflight(
    *,
    inputs: R235PreflightInputs,
    limits: R235PreflightLimits,
    probe_command: Sequence[str | os.PathLike[str]] | None = None,
    dry_run: bool = False,
    offline_probe_payload: Mapping[str, Any] | None = None,
    offline_exports: Sequence[str] | None = None,
    native_loader: NativeLoader = ctypes.CDLL,
) -> dict[str, Any]:
    """Validate one R238 package and write one immutable receipt.

    In dry mode no child, native DSO, GPU, service, or network action is
    started.  The resulting receipt is intentionally marked not eligible as a
    real local gate.  In actual mode the supplied probe command must emit the
    documented R238 probe JSON on stdout and runs under the exact-child
    watchdog.
    """

    try:
        _require_sha256(inputs.expected_archive_sha256, label="expected archive digest")
        _require_sha256(inputs.expected_manifest_sha256, label="expected manifest digest")
        stage_dir = _physical_directory(inputs.stage_dir, label="stage directory")
        archive_path = _regular_file(inputs.archive_path, label="package archive")
        manifest_path = _regular_file(inputs.manifest_path, label="package manifest")
        r225_path = _regular_file(inputs.r225_contract_path, label="r225 contract")
        r236_path = _regular_file(inputs.r236_contract_path, label="r236 contract")
        receipt_raw = Path(inputs.receipt_path).expanduser()
        if receipt_raw.is_symlink():
            raise ImmutableReceiptError("receipt target must not be a symlink")
        receipt_path = receipt_raw.resolve()
        if _inside(receipt_path, stage_dir):
            raise ImmutableReceiptError("receipt must be outside the immutable staged package")
        actual_gate_paths: R235ActualGateReceiptPaths | None = None
        if dry_run:
            if inputs.actual_gate_receipts_dir is not None:
                raise R235PreflightError(
                    "dry preflight cannot emit actual binder gate receipts"
                )
        else:
            if inputs.actual_gate_receipts_dir is None:
                raise R235PreflightError(
                    "actual exact-child preflight requires an actual gate receipt directory"
                )
            actual_gate_paths = validate_actual_gate_receipt_paths(
                R235ActualGateReceiptPaths.from_directory(inputs.actual_gate_receipts_dir),
                primary_resource_receipt=receipt_path,
                stage_dir=stage_dir,
            )
        if not _inside(manifest_path, stage_dir):
            raise R235PreflightError("explicit package manifest must be a staged package member")
        if sha256_file(archive_path) != inputs.expected_archive_sha256:
            raise R235PreflightError("package archive digest does not match its explicit expected digest")
        if archive_path.stat().st_size > PHASE1_ARCHIVE_MAX_BYTES:
            raise R235PreflightError("package archive exceeds the exact 197.7 MiB R238 limit")
        if sha256_file(manifest_path) != inputs.expected_manifest_sha256:
            raise R235PreflightError("package manifest digest does not match its explicit expected digest")
        r225_contract_sha256 = sha256_file(r225_path)
        _require_exact(
            r225_contract_sha256,
            R225_CANONICAL_SHA256,
            label="final canonical r225 contract digest",
        )

        stage_members, stage_logical_bytes, stage_disk_bytes = _stage_members(stage_dir)
        archive_members = _archive_members(archive_path)
        if stage_members != archive_members:
            raise R235PreflightError("staged package members do not exactly match the supplied archive")
        if (
            stage_logical_bytes > PHASE1_STAGE_RUNTIME_MAX_BYTES
            or stage_disk_bytes > PHASE1_STAGE_RUNTIME_MAX_BYTES
        ):
            raise R235PreflightError("staged package exceeds the exact 11.8 GiB R238 disk limit")
        entrypoint_relative = _safe_relative_path(
            inputs.entrypoint_relative_path, label="entrypoint relative path"
        )
        entrypoint = stage_dir / entrypoint_relative
        entrypoint = _regular_file(entrypoint, label="staged entrypoint")
        entrypoint_sha = sha256_file(entrypoint)

        manifest = _read_json_object(manifest_path, label="package manifest")
        r236_payload = _read_json_object(r236_path, label="r236 contract")
        r236 = _validate_r236_contract(r236_payload)
        r225_payload = _read_json_object(r225_path, label="r225 contract")
        r225 = _validate_r225_r238_contract(r225_payload)
        _validate_r238_manifest(manifest, entrypoint_sha256=entrypoint_sha, r236=r236)

        libcg_path = _regular_file(stage_dir / str(r236["linux_relative_path"]), label="staged Linux libcg")
        if libcg_path.stat().st_size != int(r236["linux_size_bytes"]):
            raise R235PreflightError("staged Linux libcg size differs from R236")
        if sha256_file(libcg_path) != r236["linux_sha256"]:
            raise R235PreflightError("staged Linux libcg digest differs from R236")
        export_receipt = _validate_native_exports(
            libcg_path,
            required_exports=list(r236["required_exports"]),
            offline_exports=offline_exports if dry_run else None,
            native_loader=native_loader,
        )

        parent_peak_rss_bytes = _rss_bytes()
        exact_child_peak_rss_bytes: int | None = None
        probe_execution: dict[str, object]
        if dry_run:
            if probe_command is not None:
                raise R235PreflightError("dry preflight must not receive a probe command")
            if offline_probe_payload is None:
                raise R235PreflightError("dry preflight requires an explicit offline probe payload")
            probe_payload = dict(offline_probe_payload)
            probe_execution = {
                "execution_mode": "dry_offline",
                "watchdog": "not_started",
                "offline_probe_sha256": sha256_bytes(canonical_json(probe_payload)),
            }
        else:
            if offline_probe_payload is not None or offline_exports is not None:
                raise R235PreflightError("actual preflight cannot consume offline fixtures")
            if not probe_command:
                raise R235PreflightError("actual preflight requires an explicit exact-child probe command")
            watchdog = R235ExactChildWatchdog(
                timeout_seconds=limits.probe_timeout_seconds,
                term_grace_seconds=limits.term_grace_seconds,
                kill_grace_seconds=limits.kill_grace_seconds,
            )
            outcome = watchdog.run(probe_command, cwd=stage_dir)
            probe_payload, probe_execution = _probe_payload_from_child(outcome)
            exact_child_peak_rss_bytes = outcome.exact_child_peak_rss_bytes
            if exact_child_peak_rss_bytes is None:
                raise R235PreflightError(
                    "actual preflight lacks exact-child Linux peak RSS evidence"
                )

        parent_peak_rss_bytes = max(parent_peak_rss_bytes, _rss_bytes())

        probe = _validate_probe(
            probe_payload,
            limits=limits,
            stage_disk_bytes=stage_disk_bytes,
            parent_peak_rss_bytes=parent_peak_rss_bytes,
            exact_child_peak_rss_bytes=exact_child_peak_rss_bytes,
        )
        archive_member_manifest_sha = _member_manifest_sha256(archive_members)
        actual_mode = not dry_run
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "receipt_name": RECEIPT_NAME,
            "status": "passed" if actual_mode else "dry_run_not_execution_eligible",
            "passed": actual_mode,
            "immutable": True,
            "write_once": True,
            "execution_mode": "exact_child" if actual_mode else "dry_offline",
            "candidate_archive_sha256": sha256_file(archive_path),
            "candidate_archive_size_bytes": archive_path.stat().st_size,
            "member_manifest_sha256": sha256_file(manifest_path),
            "entrypoint_sha256": entrypoint_sha,
            "r225_contract_sha256": r225_contract_sha256,
            "expected_canonical_r225_contract_sha256": R225_CANONICAL_SHA256,
            "canonical_libcg_contract_sha256": sha256_file(r236_path),
            "linux_x86_64_libcg_sha256": r236["linux_sha256"],
            "linux_x86_64_libcg_size_bytes": r236["linux_size_bytes"],
            "complete_ordered_action_cap": COMPLETE_ACTION_CAP,
            "simulator_search_lane_count": SIMULATOR_LANE_COUNT,
            "r238_phase1_resource_revision": 238,
            "r240_owner_revision": R240_OWNER_REVISION,
            "r242_high_confidence_threshold_owner_revision": R242_OWNER_REVISION,
            "r244_handle_scoped_search_id_owner_revision": R244_OWNER_REVISION,
            "r246_proven_deterministic_terminal_win_this_turn_owner_revision": R246_OWNER_REVISION,
            "phase1_resources": dict(PHASE1_RESOURCES),
            "phase1_submission_environment": dict(PHASE1_SUBMISSION_ENVIRONMENT),
            "phase1_resource_source": dict(PHASE1_RESOURCE_SOURCE),
            "r240_hybrid_scheduler": dict(R240_HYBRID_SCHEDULER),
            "deterministic_continuation": dict(DETERMINISTIC_CONTINUATION),
            "phase1_enforcement_bytes": {
                "stage_and_runtime_disk_max_bytes": PHASE1_STAGE_RUNTIME_MAX_BYTES,
                "combined_parent_child_rss_max_bytes": PHASE1_COMBINED_RSS_MAX_BYTES,
                "archive_max_bytes": PHASE1_ARCHIVE_MAX_BYTES,
            },
            "r238_manifest": {
                "schema": manifest["schema"],
                "role": manifest["role"],
                "sha256": sha256_file(manifest_path),
                "archive_member_manifest_sha256": archive_member_manifest_sha,
                "stage_member_manifest_sha256": _member_manifest_sha256(stage_members),
                "stage_member_count": len(stage_members),
                "archive_member_count": len(archive_members),
            },
            "canonical_libcg_r236": {
                "wheel_sha256": r236["wheel_sha256"],
                "linux_x86_64_relative_path": r236["linux_relative_path"],
                "linux_x86_64_sha256": r236["linux_sha256"],
                "linux_x86_64_size_bytes": r236["linux_size_bytes"],
                "export_validation": export_receipt,
            },
            "r225_r238_contract": r225,
            "declared_limits": limits.as_dict(),
            "observed_resource_probe": probe["observed_resource_probe"],
            "cuda_runtime_before_search": probe["cuda_runtime_before_search"],
            "cuda_available_on_probe_host": probe["cuda_available_on_probe_host"],
            "cuda_device_count_on_probe_host": probe["cuda_device_count_on_probe_host"],
            "startup_seconds": probe["startup_seconds"],
            "decision_latency_seconds": probe["decision_latency_seconds"],
            "throughput_decision_count": probe["throughput_decision_count"],
            "throughput_elapsed_seconds": probe["throughput_elapsed_seconds"],
            "throughput_decisions_per_second": probe["throughput_decisions_per_second"],
            "r240_hybrid_decision_preflight": probe[
                "r240_hybrid_decision_preflight"
            ],
            "stage_disk_bytes": probe["stage_disk_bytes"],
            "stage_logical_bytes": stage_logical_bytes,
            "runtime_disk_bytes": probe["runtime_disk_bytes"],
            "parent_peak_rss_bytes": probe["parent_peak_rss_bytes"],
            "child_peak_rss_bytes": probe["child_peak_rss_bytes"],
            "exact_child_peak_rss_bytes": probe["exact_child_peak_rss_bytes"],
            "combined_parent_child_peak_rss_bytes": probe[
                "combined_parent_child_peak_rss_bytes"
            ],
            "cpu_thread_oversubscription_preflight_passed": True,
            "memory_preflight_passed": True,
            "startup_preflight_passed": True,
            "throughput_preflight_passed": True,
            "resource_memory_startup_and_throughput_preflight_passed": actual_mode,
            "probe_execution": probe_execution,
            "kaggle_api_called": False,
            "kaggle_upload_used": False,
            "kaggle_queue_used": False,
            "upload_readiness_claim": False,
            "dry_run_can_satisfy_owner_gate": False,
        }
        gate_payloads: dict[str, dict[str, object]] | None = None
        if actual_gate_paths is not None:
            receipt["actual_gate_receipt_paths"] = actual_gate_paths.as_dict()
            gate_payloads = build_actual_gate_receipts(
                primary_resource_receipt=receipt,
                primary_resource_receipt_sha256=sha256_bytes(canonical_json(receipt)),
                probe_payload=probe_payload,
            )
        write_once_atomic(receipt_path, receipt)
        if actual_gate_paths is not None and gate_payloads is not None:
            try:
                write_actual_gate_receipts(
                    paths=actual_gate_paths,
                    payloads=gate_payloads,
                    write_once=write_once_atomic,
                    sha256_file=sha256_file,
                )
            except Exception as exc:
                # The resource receipt remains a truthful successful resource
                # gate, but a partial derived-gate set can never satisfy the
                # immutable binding.  Do not overwrite that receipt with a
                # fabricated aggregate failure.
                raise R235PreflightFailure(
                    "actual preflight derived gate receipt publication failed; "
                    "immutable binding is ineligible",
                    receipt=receipt,
                    path=receipt_path,
                ) from exc
        return receipt
    except (ImmutableReceiptError, R235PreflightFailure):
        raise
    except Exception as exc:
        receipt_path = Path(inputs.receipt_path).expanduser().resolve()
        failure = _failure_receipt(inputs=inputs, limits=limits, error=exc, dry_run=dry_run)
        write_once_atomic(receipt_path, failure)
        raise R235PreflightFailure(str(exc), receipt=failure, path=receipt_path) from exc
