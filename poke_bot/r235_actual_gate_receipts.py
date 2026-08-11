"""Actual-only R235 binder-gate receipt extraction.

The R235 package preflight has one primary resource receipt.  The immutable
binding gate also needs five narrower receipts that describe the very same
archive, manifest, contracts, exact-child execution, and validated probe.
This module derives those receipts only after a caller has completed the real
exact-child preflight.  It deliberately has no dry-fixture or standalone CLI
entry point, and it does not issue the separately owned fault, saved-episode,
full-game, or R244 SearchId receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "poke_bot.r235_r236_local_preflight_receipt/v1"
PHASE1_RESOURCE_RECEIPT_NAME = "phase1_submission_resource_and_archive_limit_receipt"
TWO_LANE_TOPOLOGY_RECEIPT_NAME = (
    "two_lane_shared_tree_topology_and_receipt_schema_regression_receipt"
)
HIGH_CONFIDENCE_RECEIPT_NAME = (
    "high_confidence_direct_and_adaptive_bounded_mcts_regression_receipt"
)
TERMINAL_WIN_RECEIPT_NAME = (
    "proven_deterministic_terminal_win_this_turn_regression_receipt"
)
DETERMINISTIC_CONTINUATION_RECEIPT_NAME = "deterministic_continuation_regression_receipt"

_RECEIPT_FILENAMES = {
    "phase1_resource": f"{PHASE1_RESOURCE_RECEIPT_NAME}.json",
    "two_lane_topology": f"{TWO_LANE_TOPOLOGY_RECEIPT_NAME}.json",
    "high_confidence": f"{HIGH_CONFIDENCE_RECEIPT_NAME}.json",
    "terminal_win": f"{TERMINAL_WIN_RECEIPT_NAME}.json",
    "deterministic_continuation": f"{DETERMINISTIC_CONTINUATION_RECEIPT_NAME}.json",
}

_COMMON_IDENTITY_FIELDS = (
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
)


class R235ActualGateReceiptError(RuntimeError):
    """Actual preflight telemetry cannot safely produce a binder gate receipt."""


@dataclass(frozen=True)
class R235ActualGateReceiptPaths:
    """Fixed immutable outputs for gates derived from one actual preflight."""

    phase1_resource: Path
    two_lane_topology: Path
    high_confidence: Path
    terminal_win: Path
    deterministic_continuation: Path

    @classmethod
    def from_directory(cls, directory: Path) -> "R235ActualGateReceiptPaths":
        root = Path(directory).expanduser()
        if not root.is_dir() or root.is_symlink():
            raise R235ActualGateReceiptError(
                "actual gate receipt directory must be an existing physical directory"
            )
        return cls(**{name: root / filename for name, filename in _RECEIPT_FILENAMES.items()})

    def as_dict(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name).expanduser().resolve())
            for name in _RECEIPT_FILENAMES
        }


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R235ActualGateReceiptError(f"{label} must be an object")
    return value


def _require_exact(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise R235ActualGateReceiptError(f"{label} is not the required value")


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R235ActualGateReceiptError(f"{label} must be a nonnegative integer")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R235ActualGateReceiptError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise R235ActualGateReceiptError(f"{label} must be finite")
    return result


def _inside(candidate: Path, directory: Path) -> bool:
    try:
        candidate.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_output_paths(
    paths: R235ActualGateReceiptPaths,
    *,
    primary_resource_receipt: Path,
    stage_dir: Path,
) -> R235ActualGateReceiptPaths:
    """Fail before execution if a derived receipt would be unsafe or stale."""

    stage = Path(stage_dir).expanduser().resolve()
    primary = Path(primary_resource_receipt).expanduser().resolve()
    normalized: dict[str, Path] = {}
    for name in _RECEIPT_FILENAMES:
        raw = Path(getattr(paths, name)).expanduser()
        if raw.exists() or raw.is_symlink():
            raise R235ActualGateReceiptError(
                f"derived actual gate receipt already exists: {raw}"
            )
        parent = raw.parent
        if not parent.is_dir() or parent.is_symlink():
            raise R235ActualGateReceiptError(
                f"derived actual gate receipt parent is not physical: {parent}"
            )
        resolved = raw.resolve()
        if resolved == primary:
            raise R235ActualGateReceiptError(
                "derived actual gate receipt collides with primary resource receipt"
            )
        if _inside(resolved, stage):
            raise R235ActualGateReceiptError(
                "derived actual gate receipt must be outside the immutable staged package"
            )
        normalized[name] = resolved
    if len(set(normalized.values())) != len(normalized):
        raise R235ActualGateReceiptError("derived actual gate receipt targets must be distinct")
    return R235ActualGateReceiptPaths(**normalized)


def _common_gate_payload(
    *,
    primary_receipt: Mapping[str, Any],
    primary_receipt_sha256: str,
    receipt_name: str,
) -> dict[str, object]:
    _require_exact(primary_receipt.get("schema"), RECEIPT_SCHEMA, label="primary receipt schema")
    _require_exact(primary_receipt.get("status"), "passed", label="primary receipt status")
    _require_exact(primary_receipt.get("passed"), True, label="primary receipt passed")
    _require_exact(
        primary_receipt.get("execution_mode"), "exact_child", label="primary receipt execution mode"
    )
    _require_exact(
        primary_receipt.get("resource_memory_startup_and_throughput_preflight_passed"),
        True,
        label="primary resource preflight",
    )
    execution = _mapping(primary_receipt.get("probe_execution"), label="primary probe execution")
    _require_exact(execution.get("execution_mode"), "exact_child", label="primary probe execution mode")
    watchdog = _mapping(execution.get("watchdog"), label="primary exact-child watchdog")
    _require_exact(watchdog.get("status"), "completed", label="primary exact-child watchdog status")
    for field in _COMMON_IDENTITY_FIELDS:
        if field not in primary_receipt:
            raise R235ActualGateReceiptError(f"primary receipt lacks common identity {field}")
    return {
        "schema": RECEIPT_SCHEMA,
        "receipt_name": receipt_name,
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        **{field: primary_receipt[field] for field in _COMMON_IDENTITY_FIELDS},
        "derived_from_actual_exact_child_preflight": True,
        "source_resource_preflight_receipt_sha256": primary_receipt_sha256,
        "source_exact_child_probe": {
            "execution_mode": "exact_child",
            "watchdog_identity": watchdog.get("identity"),
            "stdout_sha256": execution.get("stdout_sha256"),
            "stderr_sha256": execution.get("stderr_sha256"),
            "exact_child_peak_rss_bytes": execution.get("exact_child_peak_rss_bytes"),
            "exact_child_peak_rss_source": execution.get("exact_child_peak_rss_source"),
        },
    }


def _handle_scoped_projection(ambiguous: Mapping[str, Any]) -> dict[str, object]:
    raw_handles = ambiguous.get("per_lane_handle_identities")
    raw_chains = ambiguous.get("per_lane_search_id_chains")
    if not isinstance(raw_handles, list) or len(raw_handles) != 2:
        raise R235ActualGateReceiptError("ambiguous probe lacks exactly two handle identities")
    if not isinstance(raw_chains, list) or len(raw_chains) != 2:
        raise R235ActualGateReceiptError("ambiguous probe lacks exactly two SearchId chains")
    handles: list[int | str] = []
    chains: list[list[int]] = []
    for lane, raw_handle in enumerate(raw_handles):
        if isinstance(raw_handle, bool) or not isinstance(raw_handle, (int, str)):
            raise R235ActualGateReceiptError(f"ambiguous probe handle {lane} is malformed")
        if isinstance(raw_handle, str) and not raw_handle:
            raise R235ActualGateReceiptError(f"ambiguous probe handle {lane} is empty")
        handles.append(raw_handle)
        raw_chain = raw_chains[lane]
        if not isinstance(raw_chain, list) or not raw_chain:
            raise R235ActualGateReceiptError(f"ambiguous probe SearchId chain {lane} is empty")
        chains.append(
            [
                _nonnegative_int(value, label=f"ambiguous probe SearchId {lane}:{index}")
                for index, value in enumerate(raw_chain)
            ]
        )
    if len(set(handles)) != 2:
        raise R235ActualGateReceiptError("ambiguous probe does not prove distinct raw handles")
    first_ids = [chain[0] for chain in chains]
    if len({(handles[lane], first_ids[lane]) for lane in range(2)}) != 2:
        raise R235ActualGateReceiptError(
            "ambiguous probe does not prove distinct handle-scoped SearchId composites"
        )
    composites = [
        {"lane_id": lane, "handle_identity": handles[lane], "first_search_id": first_ids[lane]}
        for lane in range(2)
    ]
    _require_exact(
        ambiguous.get("per_lane_first_search_ids"),
        first_ids,
        label="ambiguous probe per-lane first SearchIds",
    )
    _require_exact(
        ambiguous.get("handle_scoped_first_search_id_composite_states"),
        composites,
        label="ambiguous probe handle-scoped SearchId composites",
    )
    return {
        "per_lane_handle_identities": handles,
        "per_lane_search_id_chains": chains,
        "per_lane_first_search_ids": first_ids,
        "handle_scoped_first_search_id_composite_states": composites,
        "per_lane_handle_first_search_id_composites": composites,
        "distinct_handle_identity_count": 2,
        "distinct_handle_scoped_first_search_id_composite_state_count": 2,
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    }


def _require_topology_witness(ambiguous: Mapping[str, Any]) -> dict[str, object]:
    for field, expected in {
        "receipt_schema_regression_passed": True,
        "one_shared_logical_mcts_tree_proved": True,
        "historical_eight_lane_manifest_or_receipt_accepted": False,
        "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_end_calls": 2,
        "lane_ids": [0, 1],
    }.items():
        _require_exact(ambiguous.get(field), expected, label=f"ambiguous topology {field}")
    releases = _nonnegative_int(
        ambiguous.get("search_release_calls"), label="ambiguous topology search releases"
    )
    if releases < 2:
        raise R235ActualGateReceiptError("ambiguous topology has fewer than two SearchRelease calls")
    depths = ambiguous.get("per_lane_depth")
    if not isinstance(depths, list) or len(depths) != 2:
        raise R235ActualGateReceiptError("ambiguous topology lacks exactly two per-lane depths")
    checked_depths = [
        _nonnegative_int(value, label=f"ambiguous topology lane depth {index}")
        for index, value in enumerate(depths)
    ]
    batches = ambiguous.get("microbatch_sizes")
    if not isinstance(batches, list) or not batches:
        raise R235ActualGateReceiptError("ambiguous topology lacks microbatch telemetry")
    checked_batches = [
        _nonnegative_int(value, label=f"ambiguous topology microbatch {index}")
        for index, value in enumerate(batches)
    ]
    if any(value not in (1, 2) for value in checked_batches):
        raise R235ActualGateReceiptError("ambiguous topology has non-two-lane microbatch telemetry")
    inflight = _nonnegative_int(
        ambiguous.get("maximum_simulator_calls_in_flight"),
        label="ambiguous topology maximum in-flight calls",
    )
    if inflight not in (1, 2):
        raise R235ActualGateReceiptError("ambiguous topology has invalid in-flight call count")
    return {
        "two_lane_shared_tree_topology_and_receipt_schema_regression_passed": True,
        "receipt_schema_regression_passed": True,
        "one_shared_logical_mcts_tree_proved": True,
        "historical_eight_lane_manifest_or_receipt_accepted": False,
        "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_end_calls": 2,
        "search_release_calls": releases,
        "lane_ids": [0, 1],
        "per_lane_depth": checked_depths,
        "microbatch_sizes": checked_batches,
        "max_simulator_calls_in_flight": inflight,
        **_handle_scoped_projection(ambiguous),
    }


def _require_actor_boundary(hybrid: Mapping[str, Any]) -> dict[str, object]:
    boundary = _mapping(
        hybrid.get("actor_change_end_turn_boundary"), label="actor-change/end-turn boundary"
    )
    _require_exact(
        boundary.get("actor_change_end_turn_boundary_regression_passed"),
        True,
        label="actor-change/end-turn boundary pass",
    )
    leaves = boundary.get("opponent_actor_leaves")
    if not isinstance(leaves, list) or not leaves:
        raise R235ActualGateReceiptError("actor-change/end-turn boundary has no opponent leaves")
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
        _require_exact(boundary.get(field), expected, label=f"actor boundary {field}")
    checked_leaves: list[dict[str, object]] = []
    for index, leaf in enumerate(leaves):
        leaf_map = _mapping(leaf, label=f"actor boundary leaf {index}")
        expected_leaf = {
            "model_value_evaluated": True,
            "expanded_legal_action_count": 0,
            "expanded_child_count": 0,
            "search_steps_beyond_boundary": 0,
            "opponent_action_selected_or_planned": False,
            "opponent_action_cached": False,
        }
        for field, expected in expected_leaf.items():
            _require_exact(leaf_map.get(field), expected, label=f"actor boundary leaf {index} {field}")
        checked_leaves.append(expected_leaf)
    for field in (
        "actor_change_boundary_leaf_count",
        "chance_boundary_leaf_count",
        "boundary_leaf_count",
    ):
        _nonnegative_int(hybrid.get(field), label=f"actor boundary aggregate {field}")
    _require_exact(
        hybrid.get("actor_change_boundary_leaf_count"),
        len(leaves),
        label="actor boundary aggregate actor-change count",
    )
    if hybrid["boundary_leaf_count"] != hybrid["actor_change_boundary_leaf_count"] + hybrid["chance_boundary_leaf_count"]:
        raise R235ActualGateReceiptError("actor boundary aggregate total drifted")
    return {
        "actor_change_end_turn_boundary_regression_passed": True,
        **expected_counts,
        "opponent_actor_leaves": checked_leaves,
        "actor_change_boundary_leaf_count": hybrid["actor_change_boundary_leaf_count"],
        "chance_boundary_leaf_count": hybrid["chance_boundary_leaf_count"],
        "boundary_leaf_count": hybrid["boundary_leaf_count"],
    }


def _high_confidence_receipt(
    *, primary: Mapping[str, Any], primary_sha256: str, hybrid: Mapping[str, Any]
) -> dict[str, object]:
    high = _mapping(hybrid.get("synthetic_high_confidence_direct"), label="high-confidence witness")
    ambiguous = _mapping(hybrid.get("synthetic_ambiguous_two_lane_mcts"), label="ambiguous witness")
    configuration = _mapping(hybrid.get("configuration"), label="hybrid configuration")
    _require_exact(configuration.get("high_confidence_threshold"), 0.80, label="high-confidence threshold")
    _require_exact(
        configuration.get("legacy_fixed_eight_second_branching_windows_rejected"),
        True,
        label="legacy fixed-eight-second rejection",
    )
    probabilities = high.get("selected_factorized_stage_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise R235ActualGateReceiptError("high-confidence witness lacks stage probabilities")
    checked_probabilities = [
        _finite_number(value, label=f"high-confidence probability {index}")
        for index, value in enumerate(probabilities)
    ]
    if any(value < 0.80 or value > 1.0 for value in checked_probabilities):
        raise R235ActualGateReceiptError("high-confidence witness does not meet inclusive 0.80")
    for field, expected in {
        "mode": "high_confidence_frozen_direct",
        "direct_action_precomputed_and_validated": True,
        "selected_factorized_stage_probability_threshold": 0.80,
        "all_selected_factorized_stages_meet_threshold": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "mcts_search_call_count": 0,
        "mcts_model_call_count": 0,
        "mcts_simulator_call_count": 0,
        "degraded": False,
    }.items():
        _require_exact(high.get(field), expected, label=f"high-confidence witness {field}")
    history_only_count = _nonnegative_int(
        high.get("history_only_existing_child_journal_count"),
        label="high-confidence history-only journal count",
    )
    if history_only_count > 1:
        raise R235ActualGateReceiptError("high-confidence witness exceeded one history-only journal")
    ambiguous_probabilities = ambiguous.get("selected_factorized_stage_probabilities")
    if not isinstance(ambiguous_probabilities, list) or not any(
        _finite_number(value, label=f"ambiguous probability {index}") < 0.80
        for index, value in enumerate(ambiguous_probabilities)
    ):
        raise R235ActualGateReceiptError("ambiguous witness did not prove below-threshold MCTS routing")
    _require_exact(ambiguous.get("both_lanes_progressed"), True, label="ambiguous both lanes")
    _require_exact(ambiguous.get("child_search_budget_seconds"), 2.0, label="ambiguous child cap")
    _require_exact(ambiguous.get("parent_action_deadline_seconds"), 4.0, label="ambiguous parent cap")
    backups = _nonnegative_int(ambiguous.get("completed_backups"), label="ambiguous backups")
    if not 8 <= backups <= 32:
        raise R235ActualGateReceiptError("ambiguous witness backups are outside 8..32")
    observations = ambiguous.get("deterministic_root_leader_observations")
    if not isinstance(observations, list) or len(observations) < 3:
        raise R235ActualGateReceiptError("ambiguous witness lacks stable leader observations")
    if any(not isinstance(value, str) or not value for value in observations):
        raise R235ActualGateReceiptError("ambiguous witness has malformed leader observations")
    if len(set(observations[-3:])) != 1:
        raise R235ActualGateReceiptError("ambiguous witness leader is not stable")
    _require_exact(
        ambiguous.get("stop_reason"), "adaptive_early_stop", label="ambiguous adaptive stop reason"
    )
    return {
        **_common_gate_payload(
            primary_receipt=primary,
            primary_receipt_sha256=primary_sha256,
            receipt_name=HIGH_CONFIDENCE_RECEIPT_NAME,
        ),
        "high_confidence_direct_and_adaptive_bounded_mcts_regression_passed": True,
        "high_confidence_mode": "high_confidence_frozen_direct",
        "high_confidence_path_returned_precomputed_legal_direct_action": True,
        "selected_factorized_stage_probability_threshold": 0.80,
        "selected_factorized_stage_probabilities": checked_probabilities,
        "all_selected_factorized_stages_meet_threshold": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "mcts_search_call_count": 0,
        "mcts_model_call_count": 0,
        "mcts_simulator_call_count": 0,
        "history_only_existing_child_journal_count": history_only_count,
        "degraded": False,
        "ambiguous_selected_stage_forced_mcts": True,
        "child_search_seconds": 2.0,
        "parent_action_deadline_seconds": 4.0,
        "minimum_backups_before_stability": 8,
        "stable_root_leader_observations": 3,
        "maximum_backups_per_decision": 32,
        "both_lanes_progressed": True,
        "legacy_fixed_eight_second_window_used": False,
        "completed_backups": backups,
        "same_root_leader_observations": len(observations),
        "stop_reason": "adaptive_early_stop",
        **_require_actor_boundary(hybrid),
    }


def _terminal_win_receipt(
    *, primary: Mapping[str, Any], primary_sha256: str, hybrid: Mapping[str, Any]
) -> dict[str, object]:
    """Project one actual R246 terminal-win proof into its own binder gate."""

    terminal = _mapping(
        hybrid.get("synthetic_proven_deterministic_terminal_win_this_turn"),
        label="R246 terminal-win witness",
    )
    for field, expected in {
        "stop_reason": "proven_deterministic_terminal_win_this_turn",
        "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_end_calls": 2,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
        # The R240 probe is normalized before it crosses this package
        # boundary; preserve the canonical R246 ``complete`` spelling in both
        # the raw witness and binder-facing receipt.
        "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
        "outstanding_virtual_loss": 0,
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
    }.items():
        _require_exact(terminal.get(field), expected, label=f"R246 terminal-win {field}")
    releases = _nonnegative_int(
        terminal.get("search_release_calls"), label="R246 terminal-win search releases"
    )
    if releases != 2:
        raise R235ActualGateReceiptError("R246 terminal-win witness did not release exactly two searches")
    backups = _nonnegative_int(
        terminal.get("completed_root_backup_count"), label="R246 terminal-win root backups"
    )
    if backups < 1:
        raise R235ActualGateReceiptError("R246 terminal-win witness has no backed root action")
    proof = _mapping(terminal.get("terminal_win_proof"), label="R246 terminal-win proof")
    proof_keys = {
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
    }
    if set(proof) != proof_keys:
        raise R235ActualGateReceiptError("R246 terminal-win proof exact schema drifted")
    for field in ("root_observation_fingerprint", "root_legal_order_fingerprint"):
        value = proof.get(field)
        if not isinstance(value, str) or not value:
            raise R235ActualGateReceiptError(f"R246 terminal-win proof {field} is absent")
    actor = proof.get("root_actor_seat")
    if isinstance(actor, bool) or not isinstance(actor, int) or actor not in (0, 1):
        raise R235ActualGateReceiptError("R246 terminal-win proof root actor is invalid")
    root_action = proof.get("root_action")
    selected_action = proof.get("selected_action")
    if (
        not isinstance(root_action, list)
        or not isinstance(selected_action, list)
        or root_action != selected_action
        or any(isinstance(value, bool) or not isinstance(value, int) for value in root_action)
    ):
        raise R235ActualGateReceiptError("R246 terminal-win proof root action is invalid")
    for field, expected in {
        "proof_kind": "exact_deterministic_simulator_terminal_win_this_turn",
        "terminal_result": "win",
        "terminal_winner_seat": actor,
        "terminal_leaf_reached": True,
        "path_no_chance_boundary": True,
        "path_no_actor_change_boundary": True,
        "path_no_opponent_boundary_crossing": True,
        "path_no_unresolved_randomness": True,
        "proof_is_deterministic": True,
    }.items():
        _require_exact(proof.get(field), expected, label=f"R246 terminal-win proof {field}")
    path_count = _nonnegative_int(
        proof.get("proof_path_action_count"), label="R246 terminal-win proof path count"
    )
    path_actors = proof.get("path_actor_seats")
    if (
        path_count < 1
        or path_count > backups
        or not isinstance(path_actors, list)
        or len(path_actors) != path_count
        or any(isinstance(value, bool) or not isinstance(value, int) or value != actor for value in path_actors)
    ):
        raise R235ActualGateReceiptError("R246 terminal-win proof path is invalid")
    lane = proof.get("discovering_lane_id")
    if isinstance(lane, bool) or not isinstance(lane, int) or lane not in (0, 1):
        raise R235ActualGateReceiptError("R246 terminal-win proof discovering lane is invalid")
    return {
        **_common_gate_payload(
            primary_receipt=primary,
            primary_receipt_sha256=primary_sha256,
            receipt_name=TERMINAL_WIN_RECEIPT_NAME,
        ),
        "proven_deterministic_terminal_win_this_turn_regression_passed": True,
        "regression_passed": True,
        "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
        "stop_reason": "proven_deterministic_terminal_win_this_turn",
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_release_calls": releases,
        "search_end_calls": 2,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "completed_root_backup_count": backups,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
        "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
        "outstanding_virtual_loss": 0,
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
        "terminal_win_proof": dict(proof),
    }


def _continuation_receipt(
    *, primary: Mapping[str, Any], primary_sha256: str, hybrid: Mapping[str, Any]
) -> dict[str, object]:
    full_game = _mapping(hybrid.get("full_game_cumulative"), label="full-game continuation telemetry")
    plans = full_game.get("deterministic_continuation_plans")
    events = full_game.get("decision_events")
    if not isinstance(plans, list) or not plans or not isinstance(events, list):
        raise R235ActualGateReceiptError("full-game continuation telemetry is incomplete")
    continuations = [
        _mapping(event, label="continuation event")
        for event in events
        if isinstance(event, Mapping) and event.get("mode") == "cached_deterministic_continuation"
    ]
    if len(continuations) != 1:
        raise R235ActualGateReceiptError("full-game telemetry must contain exactly one continuation witness")
    event = continuations[0]
    matching_plans = [
        _mapping(plan, label="continuation plan")
        for plan in plans
        if isinstance(plan, Mapping)
        and plan.get("plan_id") == event.get("plan_id")
        and plan.get("actual_turn_id") == event.get("actual_turn_id")
    ]
    if len(matching_plans) != 1:
        raise R235ActualGateReceiptError("continuation event lacks one matching validated plan")
    steps = matching_plans[0].get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise R235ActualGateReceiptError("continuation plan has invalid observed depth")
    for field, expected in {
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
        "degraded": False,
    }.items():
        _require_exact(event.get(field), expected, label=f"continuation event {field}")
    history_only_count = _nonnegative_int(
        event.get("history_only_existing_child_journal_count"),
        label="continuation history-only journal count",
    )
    if history_only_count > 1:
        raise R235ActualGateReceiptError("continuation used more than one history-only journal")
    regression = _mapping(
        full_game.get("deterministic_continuation_regression"),
        label="full-game continuation mismatch regression",
    )
    for field in (
        "chance_disagreement_clears_entire_plan",
        "fingerprint_disagreement_clears_entire_plan",
        "action_disagreement_clears_entire_plan",
        "actor_disagreement_clears_entire_plan",
        "precomputed_direct_action_and_history_correction_retained",
    ):
        _require_exact(regression.get(field), True, label=f"continuation mismatch regression {field}")
    return {
        **_common_gate_payload(
            primary_receipt=primary,
            primary_receipt_sha256=primary_sha256,
            receipt_name=DETERMINISTIC_CONTINUATION_RECEIPT_NAME,
        ),
        "deterministic_continuation_regression_passed": True,
        "two_lane_agreed_exact_fingerprint_path_consumed": True,
        "valid_match_no_new_search": True,
        "valid_match_started_new_search": False,
        "valid_match_backed_action_consumed": True,
        "valid_match_same_root_actor": True,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "history_only_existing_child_journal_count": history_only_count,
        "degraded": False,
        "configured_max_depth": 8,
        "observed_valid_match_depth": len(steps),
        "crossed_actor_change_end_turn_boundary": False,
        "chance_disagreement_clears_entire_plan": True,
        "fingerprint_disagreement_clears_entire_plan": True,
        "action_disagreement_clears_entire_plan": True,
        "actor_disagreement_clears_entire_plan": True,
        "precomputed_direct_action_and_history_correction_retained": True,
    }


def build_actual_gate_receipts(
    *,
    primary_resource_receipt: Mapping[str, Any],
    primary_resource_receipt_sha256: str,
    probe_payload: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Build five binder-compatible receipts from one validated actual probe.

    The caller must have already run the primary validation and only call this
    function for an ``exact_child`` receipt.  This function rechecks every
    additional binder-specific witness it promotes into a separate gate.
    """

    hybrid = _mapping(probe_payload.get("r240_hybrid_decision_preflight"), label="R240 hybrid probe")
    ambiguous = _mapping(hybrid.get("synthetic_ambiguous_two_lane_mcts"), label="ambiguous witness")
    primary = _mapping(primary_resource_receipt, label="primary resource receipt")
    archive_size = _nonnegative_int(
        primary.get("candidate_archive_size_bytes"), label="candidate archive size"
    )
    observed_environment = primary.get("phase1_submission_environment")
    phase1 = {
        **_common_gate_payload(
            primary_receipt=primary,
            primary_receipt_sha256=primary_resource_receipt_sha256,
            receipt_name=PHASE1_RESOURCE_RECEIPT_NAME,
        ),
        "phase1_submission_resource_and_archive_limit_receipt_passed": True,
        "resource_probe_matches_phase1_submission_environment": True,
        "archive_within_submission_limit": True,
        "observed_phase1_submission_environment": observed_environment,
        "observed_submission_archive_size_bytes": archive_size,
        "observed_submission_archive_size_mib": archive_size / float(1024**2),
    }
    topology = {
        **_common_gate_payload(
            primary_receipt=primary,
            primary_receipt_sha256=primary_resource_receipt_sha256,
            receipt_name=TWO_LANE_TOPOLOGY_RECEIPT_NAME,
        ),
        **_require_topology_witness(ambiguous),
    }
    return {
        "phase1_resource": phase1,
        "two_lane_topology": topology,
        "high_confidence": _high_confidence_receipt(
            primary=primary, primary_sha256=primary_resource_receipt_sha256, hybrid=hybrid
        ),
        "terminal_win": _terminal_win_receipt(
            primary=primary, primary_sha256=primary_resource_receipt_sha256, hybrid=hybrid
        ),
        "deterministic_continuation": _continuation_receipt(
            primary=primary, primary_sha256=primary_resource_receipt_sha256, hybrid=hybrid
        ),
    }


def write_actual_gate_receipts(
    *,
    paths: R235ActualGateReceiptPaths,
    payloads: Mapping[str, Mapping[str, object]],
    write_once: Callable[[Path, Mapping[str, object]], None],
    sha256_file: Callable[[Path], str],
) -> dict[str, dict[str, str]]:
    """Publish the already validated derived gate payloads, never overwrite."""

    expected_names = set(_RECEIPT_FILENAMES)
    if set(payloads) != expected_names:
        raise R235ActualGateReceiptError("actual gate receipt payload set is incomplete")
    emitted: dict[str, dict[str, str]] = {}
    for name in _RECEIPT_FILENAMES:
        path = Path(getattr(paths, name))
        write_once(path, payloads[name])
        emitted[name] = {"path": str(path), "sha256": sha256_file(path)}
    return emitted
