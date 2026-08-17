"""Receipt-bound accounting for the isolated r289 Alakazam derivative BO250.

This module deliberately has no simulator, Torch, training, service, Kaggle, or
selector dependency.  It owns the immutable shape of the diagnostic only:

* three separately reported 125-pair, common-seed, seat-swapped BO250 cohorts,
* the frozen r298 public-rule derivative against r274 and r195 controls,
* the exact r241 new list and immutable r195 native operational list where
  the owner contract requires them, and
* create-only, resumable game receipts and a non-promotional report.

The executable runner lives in
``scripts/run_alakazam_turn_checklist_bo250_r289.py``.  Keeping the schedule
and receipt compiler dependency-light makes it practical to validate a staged
Elmo output without loading either model or touching a managed workload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .seeded_mirror_harness import (
    SeededMirrorGameSpec,
    SeededMirrorHarnessError,
    build_seeded_seat_swapped_schedule,
    require_sha256,
)


R289_SCHEMA = "poke_bot.alakazam_turn_checklist_diagnostic_bo250_r289/v1"
R289_CONFIG_SCHEMA = "poke_bot.alakazam_turn_checklist_diagnostic_bo250_r289_config/v1"
R289_GAME_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_diagnostic_bo250_r289_game_receipt/v1"
)
R289_REPORT_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_diagnostic_bo250_r289_report/v1"
)
R289_OUTPUT_INDEX_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_diagnostic_bo250_r289_output_index/v1"
)
R289_SEEDED_ENGINE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_diagnostic_bo250_r289_seeded_engine_receipt/v1"
)
R289_CALIBRATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_gate_calibration_receipt/v1"
)
R289_R293_OVERLAP_AUDIT_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_r293_overlap_audit_receipt/v1"
)
R298_VALIDATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_validation_receipt/v1"
)
DERIVATIVE_GOAL_CONTRACT_SCHEMA = "poke_bot.alakazam_elmo_rule_derivative_goal/v1"

# The outer evaluation is deliberately only an accounting namespace.  Every
# comparison receives its own schedule namespace so common RNG material cannot
# be accidentally shared across differently-defined cohorts.
EVALUATION_ID = "alakazam-r289-rule-derivative-three-cohort-bo250"
COMPARISON_IDS = ("A", "B", "C")
COMPARISON_EVALUATION_IDS = {
    "A": "alakazam-r289-rule-derivative-bo250-a-r274-same-new-list",
    "B": "alakazam-r289-rule-derivative-bo250-b-r195-native-reference",
    "C": "alakazam-r289-rule-derivative-bo250-c-r195-new-list-confound",
}
PAIR_COUNT = 125
GAME_COUNT = 250
TOTAL_GAME_COUNT = GAME_COUNT * len(COMPARISON_IDS)
CANDIDATE_ARM = "r298_frozen_public_rule_derivative"
CONTROL_ARMS = {
    "A": "unchanged_receipt_bound_r241_r274",
    "B": "immutable_r195_native_operational_reference",
    "C": "immutable_r195_weights_exact_new_list_deck_shift_confound",
}
# Kept as an explicit alias only for small single-cohort helpers.  Callers that
# compile or run the full study must select one of ``CONTROL_ARMS``.
CONTROL_ARM = CONTROL_ARMS["A"]

R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
R195_NO_RTP_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R274_DIRECT_POLICY_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R195_SUBMISSION_ID = 55_378_392
R274_ITER0_CHECKPOINT_SHA256 = (
    "sha256:645f8e6a0bc5e0cb98695e5a65151f38eef2e5011ca63f3ea4eded42cf4a11a2"
)
R274_ITER0_CHECKPOINT_BYTES = 134_661_950
EXACT_NEW_LIST_MULTISET_SHA256 = (
    "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
)
R195_NATIVE_DECK_FILE_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R195_NATIVE_DECK_ORDERED_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R195_NATIVE_DECK_MULTISET_SHA256 = (
    "sha256:803d00583b98f81bc3908dab22974a6d65f3d35724e95874bda1eb5ea76ac9ea"
)
DERIVATIVE_OWNER_ATTACHMENT_SHA256 = (
    "sha256:0f440fc71043b4352e6401a3187c9d582c1c5614d76e186095e0eef51017af6f"
)
DERIVATIVE_MECHANICS_ATTACHMENT_SHA256 = (
    "sha256:d3f06071663dde2ae7012da72b407b410c7facd06d09ab723cad05af44ddb2cb"
)
DERIVATIVE_GOAL_GATEWAY_SHA256 = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
DERIVATIVE_GOAL_CONTRACT_SHA256 = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)
R298_RAW_WINDOW_START_UTC = "2026-07-13"
R298_RAW_WINDOW_END_UTC = "2026-08-11"
R298_RAW_PARTITION_COUNT = 30
ELMO_EXPERIMENT_RAM_CEILING_BYTES = 103_079_215_104
R295_CORRECTED_GUIDE_ATTACHMENT_SHA256 = (
    "sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b"
)
CHECKLIST_CONFIG_SCHEMA = (
    "poke_bot.alakazam_turn_checklist_heuristic_logit_layer_config/v1"
)
CHECKLIST_CHANNELS = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "replacement_alakazam_line",
    "unavoidable_draws_before_attack",
    "bench_prize_exposure",
    "immediate_disruption_outcome",
    "unknown_prize_robust_line",
    "terminal_before_forced_draw",
)
CHECKLIST_GATE_NAMES = tuple(f"{name}_gate" for name in CHECKLIST_CHANNELS) + (
    "separate_guide_gate",
)
# r293 leaves the board-exposure and aggregate disruption residuals trace-only
# until their overlap / predicate boundaries are separately demonstrated.  The
# historic broad guide is likewise trace-only under r295.  These are not
# merely default values: r289 must reject a config or calibration artifact
# that makes any of them nonzero.
CALIBRATION_ELIGIBLE_GATE_NAMES = (
    "ko_hand_threshold_gate",
    "safe_spend_above_threshold_gate",
    "replacement_alakazam_line_gate",
    "unavoidable_draws_before_attack_gate",
    "unknown_prize_robust_line_gate",
    "terminal_before_forced_draw_gate",
)
FIXED_TRACE_ONLY_GATES = {
    "bench_prize_exposure_gate": 0.0,
    "immediate_disruption_outcome_gate": 0.0,
    "separate_guide_gate": 0.0,
}
EXPECTED_GATE_CHANNEL_MAP = {
    **{f"{name}_gate": name for name in CHECKLIST_CHANNELS},
    "separate_guide_gate": "guide_support",
}
RESIDUAL_GROUP_GATE_MEMBERS = {
    "closure": (
        "ko_hand_threshold_gate",
        "immediate_disruption_outcome_gate",
        "terminal_before_forced_draw_gate",
    ),
    "continuity": (
        "safe_spend_above_threshold_gate",
        "replacement_alakazam_line_gate",
        "unavoidable_draws_before_attack_gate",
        "unknown_prize_robust_line_gate",
    ),
}
GROUPED_RESIDUAL_AGGREGATION_KIND = (
    "strongest_absolute_gated_contribution_per_option_per_group"
)


class R289BO250Error(ValueError):
    """A r289 diagnostic input, receipt, or output is unsafe to use."""


def canonical_json_bytes(payload: object) -> bytes:
    """Return the one content-addressing representation used by r289."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_digest(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _is_sha256(value: object) -> bool:
    try:
        require_sha256(value, name="digest")
    except SeededMirrorHarnessError:
        return False
    return True


def _require_digest(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise R289BO250Error(f"{label} must be a lowercase sha256 digest")
    return str(value)


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R289BO250Error(f"{label} must be an integer >= {minimum}")
    return int(value)


def _read_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R289BO250Error(f"{label} must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R289BO250Error(f"{label} is unreadable JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise R289BO250Error(f"{label} must contain a JSON object: {resolved}")
    return resolved, payload


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def file_identity(path: Path | str, *, label: str) -> dict[str, Any]:
    """Return a regular-file identity without following a symlink input."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R289BO250Error(f"{label} must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    try:
        info = resolved.stat()
    except OSError as exc:
        raise R289BO250Error(f"{label} is unreadable: {resolved}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise R289BO250Error(f"{label} is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(info.st_size),
    }


def canonical_deck_multiset_sha256(cards: Sequence[int]) -> str:
    """Hash the sorted 60-card multiset using the r241 canonical encoding."""

    if len(cards) != 60:
        raise R289BO250Error("deck must contain exactly 60 cards")
    try:
        normalized = [int(card) for card in cards]
    except (TypeError, ValueError) as exc:
        raise R289BO250Error("deck contains a non-integer card id") from exc
    return "sha256:" + hashlib.sha256(
        json.dumps(sorted(normalized), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_exact_new_list_deck(path: Path | str) -> tuple[list[int], dict[str, Any]]:
    """Read and bind the exact r241 60-card list; order is preserved for play."""

    identity = file_identity(path, label="exact new-list deck")
    cards: list[int] = []
    try:
        rows = Path(identity["path"]).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise R289BO250Error("exact new-list deck is unreadable") from exc
    for raw in rows:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        try:
            if len(parts) == 1:
                cards.append(int(parts[0]))
            elif len(parts) >= 2:
                cards.extend([int(parts[0])] * int(parts[1]))
            else:
                raise ValueError("empty row")
        except ValueError as exc:
            raise R289BO250Error("exact new-list deck has an invalid card row") from exc
    multiset = canonical_deck_multiset_sha256(cards)
    if multiset != EXACT_NEW_LIST_MULTISET_SHA256:
        raise R289BO250Error(
            "deck multiset is not the exact r241 new Alakazam list: "
            f"expected {EXACT_NEW_LIST_MULTISET_SHA256}, got {multiset}"
        )
    return cards, {**identity, "canonical_multiset_sha256": multiset, "card_count": 60}


def read_r195_native_deck(path: Path | str) -> tuple[list[int], dict[str, Any]]:
    """Read the immutable r195 operational deck without normalizing its order.

    Cohort B intentionally compares the new-list derivative against the older
    operational reference in its native configuration.  Its ordered deck list
    therefore matters as well as its multiset: accepting a reordered or merely
    equivalent 60-card file would silently change an operational control.
    """

    identity = file_identity(path, label="r195 native operational deck")
    if identity["sha256"] != R195_NATIVE_DECK_FILE_SHA256:
        raise R289BO250Error(
            "r195 native deck file identity drifted: "
            f"expected {R195_NATIVE_DECK_FILE_SHA256}, got {identity['sha256']}"
        )
    cards: list[int] = []
    try:
        rows = Path(identity["path"]).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise R289BO250Error("r195 native deck is unreadable") from exc
    for raw in rows:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        try:
            if len(parts) == 1:
                cards.append(int(parts[0]))
            elif len(parts) >= 2:
                cards.extend([int(parts[0])] * int(parts[1]))
            else:
                raise ValueError("empty row")
        except ValueError as exc:
            raise R289BO250Error("r195 native deck has an invalid card row") from exc
    if len(cards) != 60:
        raise R289BO250Error("r195 native operational deck must contain exactly 60 cards")
    ordered = "sha256:" + hashlib.sha256(
        json.dumps(cards, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    multiset = canonical_deck_multiset_sha256(cards)
    if ordered != R195_NATIVE_DECK_ORDERED_SHA256:
        raise R289BO250Error("r195 native operational deck order drifted")
    if multiset != R195_NATIVE_DECK_MULTISET_SHA256:
        raise R289BO250Error("r195 native operational deck multiset drifted")
    return cards, {
        **identity,
        "ordered_cards_sha256": ordered,
        "canonical_multiset_sha256": multiset,
        "card_count": 60,
    }


def _require_exact_keys(raw: Mapping[str, Any], *, keys: set[str], label: str) -> None:
    actual = set(raw)
    if actual != keys:
        raise R289BO250Error(
            f"{label} keys differ (unknown={sorted(actual - keys)}, "
            f"missing={sorted(keys - actual)})"
        )


def load_r289_config(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable, three-comparison Elmo diagnostic contract.

    This is intentionally stricter than a general experiment parser.  The
    candidate's actual checkpoint is *not* carried in this static config: it
    must arrive through the r298 validation receipt, so a stale config cannot
    substitute a convenient r274 or r195 artifact.
    """

    resolved, config = _read_object(path, label="r289 BO250 config")
    if (
        config.get("schema") != R289_CONFIG_SCHEMA
        or config.get("evaluation_id") != EVALUATION_ID
        or config.get("diagnostic_revision") != 289
        or config.get("host_scope") != "elmo_only"
        or config.get("status") != "staged_elmo_only_create_only_diagnostic"
    ):
        raise R289BO250Error("r289 BO250 config identity/scope drifted")
    authority_fields = (
        "training_eligible",
        "replay_eligible",
        "promotion_authority",
        "production_authority",
        "production_selector_authority",
        "kaggle_authority",
        "submission_authority",
        "elmo_production_readmission_authority",
        "managed_service_authority",
        "service_start_authority",
        "service_stop_authority",
        "inzi_authority",
    )
    if any(config.get(field) is not False for field in authority_fields):
        raise R289BO250Error("r289 config unexpectedly grants authority")
    goal = config.get("goal_contract")
    if (
        not isinstance(goal, Mapping)
        or goal.get("schema") != DERIVATIVE_GOAL_CONTRACT_SCHEMA
        or goal.get("path") != "goals/alakazam-elmo-rule-derivative/contract.json"
        or goal.get("sha256") != DERIVATIVE_GOAL_CONTRACT_SHA256
        or goal.get("source_attachment_sha256") != DERIVATIVE_OWNER_ATTACHMENT_SHA256
        or goal.get("mechanics_attachment_sha256") != DERIVATIVE_MECHANICS_ATTACHMENT_SHA256
    ):
        raise R289BO250Error("r289 config lacks the canonical derivative goal binding")
    candidate = config.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("arm") != CANDIDATE_ARM
        or candidate.get("identity_source") != "required_r298_validation_receipt"
        or candidate.get("checklist_layer_enabled") is not True
        or candidate.get("strict_r298_stage_wrapper_required") is not True
        or candidate.get("legacy_r288_action_or_logit_authority") is not False
        or candidate.get("direct_policy_only") is not True
        or candidate.get("frozen_backbone_required") is not True
        or candidate.get("zero_gated_layer_off_baseline_parity_required") is not True
    ):
        raise R289BO250Error("r289 derivative candidate boundary drifted")
    direct = config.get("direct_policy_boundary")
    if not isinstance(direct, Mapping) or any(
        direct.get(key) is not False
        for key in ("rtp", "search", "mcts", "rollout", "hidden_information_inference")
    ):
        raise R289BO250Error("r289 config permits a forbidden action authority")
    decks = config.get("decks")
    if (
        not isinstance(decks, Mapping)
        or not isinstance(decks.get("exact_new_list"), Mapping)
        or decks["exact_new_list"].get("canonical_multiset_sha256")
        != EXACT_NEW_LIST_MULTISET_SHA256
        or not isinstance(decks.get("r195_native_operational"), Mapping)
        or decks["r195_native_operational"].get("file_sha256")
        != R195_NATIVE_DECK_FILE_SHA256
        or decks["r195_native_operational"].get("ordered_cards_sha256")
        != R195_NATIVE_DECK_ORDERED_SHA256
        or decks["r195_native_operational"].get("canonical_multiset_sha256")
        != R195_NATIVE_DECK_MULTISET_SHA256
    ):
        raise R289BO250Error("r289 deck identities drifted")
    if config.get("checklist_channels") != list(CHECKLIST_CHANNELS):
        raise R289BO250Error("r289 checklist channel order drifted")
    expected_comparisons = {
        "A": {
            "candidate_arm": CANDIDATE_ARM,
            "control_arm": CONTROL_ARMS["A"],
            "candidate_deck": "exact_new_list",
            "control_deck": "exact_new_list",
            "interpretation": "same_architecture_primary_comparison",
            "deck_shift_confound": False,
        },
        "B": {
            "candidate_arm": CANDIDATE_ARM,
            "control_arm": CONTROL_ARMS["B"],
            "candidate_deck": "exact_new_list",
            "control_deck": "r195_native_operational",
            "interpretation": "operational_reference_comparison",
            "deck_shift_confound": True,
        },
        "C": {
            "candidate_arm": CANDIDATE_ARM,
            "control_arm": CONTROL_ARMS["C"],
            "candidate_deck": "exact_new_list",
            "control_deck": "exact_new_list",
            "interpretation": "exploratory_deck_shift_confound",
            "deck_shift_confound": True,
        },
    }
    raw_comparisons = config.get("comparisons")
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) != len(COMPARISON_IDS):
        raise R289BO250Error("r289 config must define exactly comparisons A/B/C")
    comparisons = {row.get("id"): row for row in raw_comparisons if isinstance(row, Mapping)}
    if set(comparisons) != set(COMPARISON_IDS):
        raise R289BO250Error("r289 comparison identifiers drifted")
    for comparison_id, expected in expected_comparisons.items():
        actual = comparisons[comparison_id]
        if dict(actual) != {"id": comparison_id, **expected}:
            raise R289BO250Error(f"r289 comparison {comparison_id} contract drifted")
    pairing = config.get("pairing")
    expected_pairing = {
        "pair_count_per_comparison": PAIR_COUNT,
        "games_per_pair": 2,
        "game_count_per_comparison": GAME_COUNT,
        "comparison_count": len(COMPARISON_IDS),
        "total_game_count": TOTAL_GAME_COUNT,
        "seed_matched_within_comparison": True,
        "seat_swapped_within_comparison": True,
        "candidate_seat_0_games_per_comparison": PAIR_COUNT,
        "candidate_seat_1_games_per_comparison": PAIR_COUNT,
        "candidate_actual_first_games_per_comparison": PAIR_COUNT,
        "candidate_actual_second_games_per_comparison": PAIR_COUNT,
        "conclusive_by_itself": False,
    }
    if not isinstance(pairing, Mapping) or dict(pairing) != expected_pairing:
        raise R289BO250Error("r289 three-cohort pairing contract drifted")
    seeded_engine = config.get("seeded_engine")
    calibration = config.get("calibration")
    legacy_trace = config.get("legacy_r293_r295_trace_evidence")
    r298 = config.get("r298_validation")
    rev4_evidence = config.get("rev4_evidence")
    output = config.get("output")
    if (
        not isinstance(seeded_engine, Mapping)
        or seeded_engine.get("required") is not True
        or seeded_engine.get("receipt_required") is not True
        or seeded_engine.get("unseeded_fallback_allowed") is not False
        or seeded_engine.get("same_engine_and_deck_order_seed_within_pair") is not True
        or seeded_engine.get("first_player_seal_required_before_pair_credit") is not True
        or not isinstance(calibration, Mapping)
        or calibration.get("required_before_benchmark") is not True
        or calibration.get("all_neural_checkpoint_tensors_frozen") is not True
        or calibration.get("source_disjoint_exact_new_list_data_required") is not True
        or calibration.get("bo250_seed_identity_exclusion_required") is not True
        or calibration.get("calibration_data_training_eligible") is not False
        or tuple(calibration.get("eligible_gate_order") or ())
        != CALIBRATION_ELIGIBLE_GATE_NAMES
        or dict(calibration.get("fixed_trace_only_gates") or {}) != FIXED_TRACE_ONLY_GATES
        or not isinstance(legacy_trace, Mapping)
        or legacy_trace.get("may_be_staged_for_historical_trace_only") is not True
        or legacy_trace.get("runtime_action_or_logit_authority") is not False
        or legacy_trace.get("legacy_broad_guide_runtime_residual") != 0.0
        or legacy_trace.get("r295_corrected_attachment_sha256")
        != R295_CORRECTED_GUIDE_ATTACHMENT_SHA256
        or not isinstance(r298, Mapping)
        or r298.get("required_before_benchmark") is not True
        or r298.get("receipt_schema") != R298_VALIDATION_RECEIPT_SCHEMA
        or r298.get("host") != "elmo"
        or r298.get("source_attachment_sha256") != DERIVATIVE_OWNER_ATTACHMENT_SHA256
        or r298.get("mechanics_attachment_sha256") != DERIVATIVE_MECHANICS_ATTACHMENT_SHA256
        or not isinstance(rev4_evidence, Mapping)
        or rev4_evidence.get("required_before_benchmark") is not True
        or rev4_evidence.get("raw_corpus_receipt_schema")
        != "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_receipt/v1"
        or rev4_evidence.get("collision_census_receipt_schema")
        != "poke_bot.alakazam_collision_census_r298_receipt/v1"
        or rev4_evidence.get("frozen_schema_and_zero_bypass_receipt_required")
        is not True
        or rev4_evidence.get("census_supported_candidate_training_receipt_required")
        is not True
        or rev4_evidence.get("quarantined_create_only_transfer_predecessor_required")
        is not True
        or rev4_evidence.get("transfer_is_not_runtime_or_training_authority")
        is not True
        or rev4_evidence.get("strict_r298_checklist_runtime_receipt_required")
        is not True
        or rev4_evidence.get("legacy_r288_guide_or_csv_derived_runtime_authority")
        is not False
        or not isinstance(output, Mapping)
        or output.get("content_addressed") is not True
        or output.get("game_receipts") != "create_only"
        or output.get("partial_report_credit") is not False
        or output.get("report_requires_all_750_completed_game_receipts") is not True
    ):
        raise R289BO250Error("r289 config receipt/calibration/output boundary drifted")
    return config, {
        **file_identity(resolved, label="r289 BO250 config"),
        "content_sha256": sha256_file(resolved),
    }


def validate_checklist_config(path: Path | str) -> dict[str, Any]:
    """Ensure an explicit r288 config can only be used as the candidate layer."""

    resolved, payload = _read_object(path, label="r288 checklist config")
    runtime = payload.get("runtime")
    exact_list = payload.get("exact_new_list")
    guide_attachment = payload.get("guide_attachment")
    if (
        payload.get("schema") != CHECKLIST_CONFIG_SCHEMA
        or payload.get("owner_decision_revision") != 288
        or payload.get("active") is not False
        or payload.get("selected_by_runtime_without_explicit_receipt") is not False
        or not isinstance(runtime, Mapping)
        or not isinstance(exact_list, Mapping)
        or exact_list.get("canonical_multiset_sha256") != EXACT_NEW_LIST_MULTISET_SHA256
        or tuple(runtime.get("channel_order") or ()) != CHECKLIST_CHANNELS
        or float(runtime.get("total_residual_cap", math.nan)) != 0.1
    ):
        raise R289BO250Error("r288 checklist config identity or bounded route drifted")
    forbidden = payload.get("forbidden")
    if not isinstance(forbidden, Mapping) or any(
        forbidden.get(key) is not False
        for key in ("hidden_information_inference", "search", "rollout", "mcts", "rtp")
    ):
        raise R289BO250Error("r288 checklist config permits forbidden behavior")
    if (
        not isinstance(guide_attachment, Mapping)
        or guide_attachment.get("owner_revision") != 295
        or guide_attachment.get("sha256")
        != R295_CORRECTED_GUIDE_ATTACHMENT_SHA256
        or guide_attachment.get("corrected_exact_inventory")
        != {"pokemon": 17, "trainers": 36, "energy": 7, "alakazam": 3}
        or guide_attachment.get("matches_exact_new_list_canonical_multiset")
        is not True
        or guide_attachment.get("runtime_action_authority") is not False
        or guide_attachment.get("guide_selected_action_override") is not False
    ):
        raise R289BO250Error("r288 checklist config lacks the corrected r295 guide boundary")
    # Only six new checklist scalar gates may be calibrated or become
    # nonzero in this diagnostic.  Q5, Q6 and the historic broad guide are
    # intentionally still collected as trace evidence, but their residuals
    # must be exactly zero in both the base config and every derived overlay.
    gate_order = runtime.get("gate_order")
    residual_enabled = runtime.get("residual_enabled_gate_order")
    gate_map = runtime.get("gate_channel_map")
    scalar_gates = runtime.get("scalar_gates")
    guide_policy = runtime.get("guide_support_policy")
    residual_formula = runtime.get("residual_formula")
    offline = payload.get("offline_calibration")
    if (
        runtime.get("guide_support_channel") != "guide_support"
        or tuple(gate_order or ()) != CHECKLIST_GATE_NAMES
        or tuple(residual_enabled or ()) != CALIBRATION_ELIGIBLE_GATE_NAMES
        or not isinstance(gate_map, Mapping)
        or dict(gate_map) != EXPECTED_GATE_CHANNEL_MAP
        or not isinstance(scalar_gates, Mapping)
        or set(scalar_gates) != set(CHECKLIST_GATE_NAMES)
        or not isinstance(guide_policy, Mapping)
        or guide_policy.get("mode") != "trace_only"
        or guide_policy.get("runtime_residual_authority") is not False
        or guide_policy.get("runtime_gate_required") != 0.0
        or guide_policy.get("calibration_eligible") is not False
        or not isinstance(residual_formula, Mapping)
        or residual_formula.get("grouped_aggregation")
        != (
            "for each option and each residual group, retain the nonzero gated "
            "channel contribution with greatest absolute magnitude; break "
            "equal-magnitude ties by runtime.channel_order; an exactly-zero "
            "group has no winner and contributes zero"
        )
        or tuple(residual_formula.get("group_order") or ())
        != tuple(RESIDUAL_GROUP_GATE_MEMBERS)
        or not isinstance(residual_formula.get("group_gate_members"), Mapping)
        or {
            name: tuple(residual_formula["group_gate_members"].get(name) or ())
            for name in RESIDUAL_GROUP_GATE_MEMBERS
        }
        != RESIDUAL_GROUP_GATE_MEMBERS
        or set(residual_formula["group_gate_members"])
        != set(RESIDUAL_GROUP_GATE_MEMBERS)
        or residual_formula.get("tie_break") != "runtime.channel_order_first"
        or residual_formula.get("exact_zero_group_winner") is not None
        or residual_formula.get("pre_clip")
        != "closure_winner_contribution[option_index] + continuity_winner_contribution[option_index]"
        or residual_formula.get("clip_min") != -0.1
        or residual_formula.get("clip_max") != 0.1
        or residual_formula.get("neutral_when_all_vectors_zero") is not True
        or residual_formula.get("guide_support_is_separate_gate") is not True
        or not isinstance(offline, Mapping)
        or offline.get("trainable_scalars_exact") != len(CALIBRATION_ELIGIBLE_GATE_NAMES)
        or tuple(offline.get("trainable_gate_order") or ())
        != CALIBRATION_ELIGIBLE_GATE_NAMES
        or not isinstance(offline.get("fixed_trace_only_gates"), Mapping)
        or dict(offline["fixed_trace_only_gates"]) != FIXED_TRACE_ONLY_GATES
    ):
        raise R289BO250Error("r288 six-gate / trace-only residual boundary drifted")
    for name in CHECKLIST_GATE_NAMES:
        value = scalar_gates[name]
        if type(value) is bool or not isinstance(value, (int, float)):
            raise R289BO250Error(f"r288 scalar gate {name} is non-numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 0.10:
            raise R289BO250Error(f"r288 scalar gate {name} exceeds bounds")
    for name, expected in FIXED_TRACE_ONLY_GATES.items():
        if scalar_gates[name] != expected:
            raise R289BO250Error(f"r288 trace-only gate {name} must be exact zero")
    return file_identity(resolved, label="r288 checklist config")


def _require_zero_residual(value: object, *, label: str) -> None:
    """Reject a broad-guide route that claims trace-only yet changes logits."""

    if isinstance(value, Mapping):
        if not value:
            raise R289BO250Error(f"{label} may not be empty")
        for item in value.values():
            _require_zero_residual(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            raise R289BO250Error(f"{label} may not be empty")
        for item in value:
            _require_zero_residual(item, label=label)
        return
    if type(value) is bool or not isinstance(value, (int, float)):
        raise R289BO250Error(f"{label} must be a finite numeric zero")
    if not math.isfinite(float(value)) or float(value) != 0.0:
        raise R289BO250Error(f"{label} must be exact zero")


def validate_r293_overlap_audit_receipt(
    path: Path | str,
    *,
    checklist_config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind r292/r293/r295 prerequisites before any r289 game can start.

    Revision 293 deliberately leaves the existing r274 neural, Fusion,
    OwnDeck, and Matchup Adapter paths unchanged.  The independent receipt
    therefore attests that duplicate influence was reviewed in the new layer
    alone and that the historic broad guide remains trace-only / zero.  The
    runner additionally requires the same fields from every live stage trace.
    """

    resolved, payload = _read_object(path, label="r293/r295 checklist audit receipt")
    candidate = payload.get("candidate")
    authority = payload.get("authority")
    direct = payload.get("direct_policy_boundary")
    channels = payload.get("channel_audits")
    guide = payload.get("legacy_broad_guide")
    if (
        payload.get("schema") != R289_R293_OVERLAP_AUDIT_RECEIPT_SCHEMA
        or payload.get("status") != "completed"
        or payload.get("scope") != "elmo_only_nonproduction"
        or not isinstance(candidate, Mapping)
        or candidate.get("checkpoint_sha256") != R274_ITER0_CHECKPOINT_SHA256
        or candidate.get("checkpoint_size_bytes") != R274_ITER0_CHECKPOINT_BYTES
        or candidate.get("matchup_tree_sha256")
        != R274_DIRECT_POLICY_MATCHUP_TREE_SHA256
        or payload.get("exact_new_list_multiset_sha256")
        != EXACT_NEW_LIST_MULTISET_SHA256
        or payload.get("checklist_base_config_sha256")
        != checklist_config_identity.get("sha256")
        or payload.get("r288_candidate_package_parity_passed") is not True
        or payload.get("r292_bench_only_regression_passed") is not True
        or payload.get("r292_local_elmo_parity_passed") is not True
        or payload.get("r293_overlap_audit_passed") is not True
        or payload.get("r293_per_channel_trace_contract_passed") is not True
        or payload.get("r295_corrected_attachment_sha256")
        != R295_CORRECTED_GUIDE_ATTACHMENT_SHA256
        or not isinstance(direct, Mapping)
        or any(
            direct.get(key) is not False
            for key in ("rtp", "search", "mcts", "rollout", "hidden_information_inference")
        )
        or not isinstance(authority, Mapping)
        or any(
            authority.get(key) is not False
            for key in (
                "training_eligible",
                "replay_eligible",
                "production_authority",
                "promotion_authority",
                "selector_authority",
                "kaggle_authority",
                "submission_authority",
                "elmo_production_readmission_authority",
            )
        )
        or not isinstance(guide, Mapping)
        or guide.get("trace_only") is not True
        or not isinstance(channels, Mapping)
        or set(channels) != set(CHECKLIST_CHANNELS)
    ):
        raise R289BO250Error("r293/r295 checklist audit receipt is incomplete")
    _require_zero_residual(
        guide.get("runtime_residual"), label="legacy broad guide runtime residual"
    )
    for name in CHECKLIST_CHANNELS:
        channel = channels[name]
        if (
            not isinstance(channel, Mapping)
            or not isinstance(
                channel.get("existing_route_overlap_or_distinct_reason"), str
            )
            or not channel["existing_route_overlap_or_distinct_reason"].strip()
            or not isinstance(
                channel.get("attenuation_or_suppression_decision"), str
            )
            or not channel["attenuation_or_suppression_decision"].strip()
        ):
            raise R289BO250Error(f"r293 channel audit is incomplete for {name}")
        post_dedup = channel.get("post_deduplication_signed_residual")
        try:
            _assert_r293_residual_cap(post_dedup)
        except R289BO250Error as exc:
            raise R289BO250Error(
                f"r293 channel audit residual is invalid for {name}"
            ) from exc
    return file_identity(resolved, label="r293/r295 checklist audit receipt")


def _assert_r293_residual_cap(value: object) -> None:
    if isinstance(value, Mapping):
        if not value:
            raise R289BO250Error("r293 signed residual may not be empty")
        for item in value.values():
            _assert_r293_residual_cap(item)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            raise R289BO250Error("r293 signed residual may not be empty")
        for item in value:
            _assert_r293_residual_cap(item)
        return
    if type(value) is bool or not isinstance(value, (int, float)):
        raise R289BO250Error("r293 signed residual must be numeric")
    if not math.isfinite(float(value)) or abs(float(value)) > 0.1000001:
        raise R289BO250Error("r293 signed residual exceeds 0.10")


def validate_owner_contract(path: Path | str) -> dict[str, Any]:
    """Bind the runner to the r289 clause without claiming it activates r274."""

    resolved, payload = _read_object(path, label="r274 owner contract")
    design = payload.get("elmo_turn_checklist_diagnostic_bo250")
    if (
        payload.get("schema") != "poke_bot.alakazam_new_list_direct_policy_r274/v1"
        or int(payload.get("latest_owner_clarification_revision", 0) or 0) < 289
        or not isinstance(design, Mapping)
        or design.get("owner_revision") != 289
        or design.get("host") != "elmo"
    ):
        raise R289BO250Error("owner contract lacks the r289 Elmo BO250 authorization")
    pairing = design.get("pairing")
    exact = design.get("exact_new_list_binding")
    boundary = design.get("direct_policy_boundary")
    authority = design.get("authority")
    disjointness = design.get("disjointness")
    runtime_preservation = payload.get("peak_r195_behavior_preservation")
    if (
        not isinstance(pairing, Mapping)
        or pairing.get("pairs_exact") != PAIR_COUNT
        or pairing.get("games_total_exact") != GAME_COUNT
        or pairing.get("seed_matched") is not True
        or pairing.get("seat_swapped") is not True
        or pairing.get("candidate_seat_0_games_exact") != PAIR_COUNT
        or pairing.get("candidate_seat_1_games_exact") != PAIR_COUNT
        or pairing.get("candidate_actual_first_games_exact") != PAIR_COUNT
        or pairing.get("candidate_actual_second_games_exact") != PAIR_COUNT
        or not isinstance(exact, Mapping)
        or exact.get("canonical_multiset_sha256") != EXACT_NEW_LIST_MULTISET_SHA256
        or not isinstance(boundary, Mapping)
        or any(boundary.get(key) is not False for key in ("rtp", "search", "mcts", "rollout", "hidden_information_inference"))
        or not isinstance(authority, Mapping)
        or any(authority.get(key) is not False for key in ("promotion", "production_loop", "production_selector", "kaggle", "submission", "elmo_production_readmission"))
        or design.get("current_active_iteration_mutation_allowed") is not False
        or not isinstance(disjointness, Mapping)
        or disjointness.get("calibration_rows_disjoint_from_bo250") is not True
        or disjointness.get("calibration_seeds_disjoint_from_bo250") is not True
        or disjointness.get("bo250_training_eligible") is not False
        or not isinstance(runtime_preservation, Mapping)
        or runtime_preservation.get("matchup_adapter_runtime_enabled") is not True
        or runtime_preservation.get("learner_public_matchup_tree_sha256")
        != R274_DIRECT_POLICY_MATCHUP_TREE_SHA256
    ):
        raise R289BO250Error("owner contract r289 diagnostic fields drifted")
    return file_identity(resolved, label="r274 owner contract")


def validate_derivative_goal_contract(path: Path | str) -> dict[str, Any]:
    """Bind this evaluator to the sole typed revision-4 derivative contract.

    The goal gateway is intentionally checked as a separate immutable file by
    the r298 receipt validator.  This function validates the typed contract
    that actually defines the three cohorts and rejects an older root-goal
    clause, even if it happens to mention an r289 BO250 by name.
    """

    resolved, payload = _read_object(path, label="r298 derivative goal contract")
    authority = payload.get("authority")
    sources = payload.get("source_attachment")
    historical = payload.get("historical_provenance")
    baseline_and_candidate = payload.get("baselines_and_candidate")
    evaluation = payload.get("evaluation")
    card2vec = payload.get("card2vec_non_regression")
    transfer = payload.get("create_only_inzi_transfer_staging")
    if (
        payload.get("schema") != DERIVATIVE_GOAL_CONTRACT_SCHEMA
        or payload.get("goal_id") != "alakazam-elmo-rule-derivative"
        or payload.get("goal_revision") != 4
        or payload.get("status")
        != "authorized_isolated_elmo_research_no_production_authority"
        or not isinstance(authority, Mapping)
        or authority.get("sole_typed_canonical_source")
        != "goals/alakazam-elmo-rule-derivative/contract.json"
        or authority.get("isolated_elmo_implementation_and_evaluation") is not True
        or authority.get("inzi_execution_or_staged_byte_use_authority") is not False
        or authority.get("production_authority") is not False
        or authority.get("quarantined_create_only_inzi_shard_transfer_authority")
        is not True
        or authority.get(
            "quarantined_create_only_inzi_shard_transfer_authority_is_only_inzi_exception"
        )
        is not True
        or authority.get("selector_runtime_package_submission_or_deployment_authority")
        is not False
        or not isinstance(sources, Mapping)
        or sources.get("sha256") != DERIVATIVE_OWNER_ATTACHMENT_SHA256
        or not isinstance(historical, Mapping)
        or not isinstance(historical.get("source_attachment"), Mapping)
        or historical["source_attachment"].get("sha256")
        != DERIVATIVE_MECHANICS_ATTACHMENT_SHA256
        or not isinstance(baseline_and_candidate, Mapping)
        or not isinstance(evaluation, Mapping)
        or not isinstance(card2vec, Mapping)
        or card2vec.get("preserve_complete_existing_system") is not True
        or card2vec.get("checkpoint_tensors_bit_identical") is not True
        or card2vec.get("runtime_behavior_bit_identical_when_new_path_off")
        is not True
        or card2vec.get("structured_simulator_residual_may_replace_card2vec")
        is not False
        or card2vec.get("legacy_text_hashes_are_exact_mechanics_authority_for_new_residual")
        is not False
        or not isinstance(transfer, Mapping)
        or transfer.get("owner_goal_revision") != 4
        or transfer.get("authority")
        != "transfer_and_quarantined_byte_staging_only"
        or transfer.get("source_host") != "elmo"
        or transfer.get("destination_host") != "inzi"
        or transfer.get("destination_role")
        != "quarantined_non_runtime_create_only_staging"
        or transfer.get("staged_objects_training_or_evaluation_eligible_on_inzi")
        is not False
        or transfer.get("staging_or_final_parity_receipt_grants_later_inzi_use")
        is not False
    ):
        raise R289BO250Error("canonical r298 derivative goal contract drifted")
    baseline = baseline_and_candidate.get("r241_r274_same_architecture_baseline")
    candidate = baseline_and_candidate.get("derivative_candidate")
    r195 = baseline_and_candidate.get("r195_operational_reference")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("candidate_id") != "alakazam-new-list-direct-policy-r274"
        or baseline.get("must_be_unchanged") is not True
        or baseline.get("exact_new_list_canonical_multiset_sha256")
        != EXACT_NEW_LIST_MULTISET_SHA256
        or baseline.get("checkpoint_sha256") is not None
        or baseline.get("checkpoint_size_bytes") is not None
        or not isinstance(candidate, Mapping)
        or candidate.get("candidate_id") != "alakazam-elmo-public-rule-derivative-g1"
        or candidate.get("frozen_backbone") is not True
        or candidate.get("existing_heads_frozen") is not True
        or candidate.get("fusion_frozen") is not True
        or candidate.get("own_deck_routes_frozen") is not True
        or candidate.get("matchup_adapters_frozen") is not True
        or candidate.get("candidate_validation_receipt_required") is not True
        or candidate.get("checkpoint_sha256") is not None
        or candidate.get("checkpoint_size_bytes") is not None
        or not isinstance(r195, Mapping)
        or r195.get("checkpoint_sha256") != R195_CHECKPOINT_SHA256
        or r195.get("checkpoint_size_bytes") != R195_CHECKPOINT_BYTES
        or r195.get("immutable") is not True
    ):
        raise R289BO250Error("canonical r298 candidate/baseline contract drifted")
    raw_corpus = payload.get("raw_corpus_contract")
    resources = payload.get("elmo_resource_envelope")
    sequencing = payload.get("sequencing_contract")
    transfer_shards = transfer.get("shard_eligibility") if isinstance(transfer, Mapping) else None
    transfer_transport = transfer.get("transport") if isinstance(transfer, Mapping) else None
    transfer_parity = transfer.get("final_parity_receipt") if isinstance(transfer, Mapping) else None
    if (
        not isinstance(raw_corpus, Mapping)
        or raw_corpus.get("window_start_utc") != "2026-07-13"
        or raw_corpus.get("window_end_utc") != "2026-08-11"
        or raw_corpus.get("window_inclusive") is not True
        or raw_corpus.get("contiguous_window_required") is not True
        or raw_corpus.get("utc_partition_count_exact") != 30
        or raw_corpus.get("distinct_utc_partitions_required") is not True
        or raw_corpus.get("validated_manifest_required") is not True
        or raw_corpus.get("deduplicated_manifest_required") is not True
        or raw_corpus.get("twenty_day_fallback_allowed") is not False
        or raw_corpus.get("any_silent_subset_or_adjacent_window_fallback_allowed")
        is not False
        or not isinstance(resources, Mapping)
        or resources.get("experiment_ram_hard_ceiling_bytes") != 103079215104
        or resources.get("memory_heavy_phase_concurrency_exact") != 1
        or resources.get("managed_service_create_edit_stop_restart_reload_or_replace_allowed")
        is not False
        or not isinstance(sequencing, Mapping)
        or sequencing.get("owner_goal_revision") != 4
        or sequencing.get("feature_schema_freeze_precedes_refeaturization") is not True
        or sequencing.get("target_schema_freeze_precedes_refeaturization") is not True
        or sequencing.get("checklist_provenance_schema_freeze_precedes_refeaturization")
        is not True
        or sequencing.get("collision_census_may_be_generated_during_or_from_same_refeaturization_pass")
        is not True
        or not isinstance(sequencing.get("audit_gates"), Mapping)
        or sequencing["audit_gates"].get(
            "bo250_before_corpus_validation_census_and_candidate_receipts"
        )
        is not False
        or not isinstance(transfer_shards, Mapping)
        or transfer_shards.get("maximum_size_bytes") != 100_663_296
        or transfer_shards.get("individual_schema_validation_required") is not True
        or transfer_shards.get("individual_sha256_validation_required") is not True
        or transfer_shards.get("individual_exact_size_validation_required") is not True
        or not isinstance(transfer_transport, Mapping)
        or transfer_transport.get("parallel_lanes_maximum") != 4
        or transfer_transport.get("atomic_create_only_final_publication_required")
        is not True
        or transfer_transport.get("existing_conflicting_final_name_or_content")
        != "refuse_without_overwrite_delete_rename_or_replace"
        or not isinstance(transfer_parity, Mapping)
        or transfer_parity.get("required") is not True
        or transfer_parity.get("source_remote_parity_required") is not True
    ):
        raise R289BO250Error("canonical r298 raw-corpus/resource boundary drifted")
    expected_comparisons = {
        "A": ("exact_new_list", "exact_same_new_list", "same_architecture_primary_comparison"),
        "B": ("exact_new_list", "r195_native_operational_configuration", "operational_reference_comparison"),
        "C": ("exact_new_list", "exact_new_list", "exploratory_deck_shift_confound"),
    }
    raw_comparisons = evaluation.get("comparisons")
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) != 3:
        raise R289BO250Error("canonical r298 contract does not define A/B/C")
    comparisons = {row.get("id"): row for row in raw_comparisons if isinstance(row, Mapping)}
    if set(comparisons) != set(expected_comparisons):
        raise R289BO250Error("canonical r298 comparison identities drifted")
    for comparison_id, (candidate_deck, control_deck, interpretation) in expected_comparisons.items():
        row = comparisons[comparison_id]
        if (
            row.get("candidate_deck") != candidate_deck
            or row.get("control_deck") != control_deck
            or row.get("interpretation") != interpretation
        ):
            raise R289BO250Error(
                f"canonical r298 comparison {comparison_id} deck/interpretation drifted"
            )
    pairing = evaluation.get("bo250_per_reported_comparison")
    if (
        not isinstance(pairing, Mapping)
        or dict(pairing) != {
            "pairs": PAIR_COUNT,
            "games": GAME_COUNT,
            "seed_matched": True,
            "seat_swapped": True,
            "conclusive_by_itself": False,
        }
        or evaluation.get(
            "bo250_before_validated_30_day_corpus_complete_census_and_candidate_receipts"
        )
        is not False
    ):
        raise R289BO250Error("canonical r298 BO250 pairing contract drifted")
    identity = file_identity(resolved, label="r298 derivative goal contract")
    if identity["sha256"] != DERIVATIVE_GOAL_CONTRACT_SHA256:
        raise R289BO250Error(
            "canonical r298 derivative contract digest drifted; do not silently adopt it"
        )
    return {**identity, "content_sha256": identity["sha256"]}


def validate_r298_validation_receipt(
    path: Path | str,
    *,
    goal_contract_identity: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    baseline_r274_checkpoint: Mapping[str, Any],
    comparison_seed_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Consume the strict external r298 receipt and bind files/seeds to it.

    The r298 owner intentionally provided a typed receipt validator.  Reusing
    it here keeps the benchmark from accepting a hand-shaped lookalike receipt
    while this adapter remains staged and zero-gated by default.
    """

    try:
        from .alakazam_simulator_rules_r298_validation import (
            R298ValidationError,
            validate_validation_receipt,
        )
    except ImportError as exc:  # pragma: no cover - package corruption is fatal.
        raise R289BO250Error("r298 validation receipt parser is unavailable") from exc
    resolved, raw = _read_object(path, label="r298 validation receipt")
    try:
        receipt = validate_validation_receipt(raw)
    except R298ValidationError as exc:
        raise R289BO250Error("r298 validation receipt did not pass strict validation") from exc
    if receipt.get("schema") != R298_VALIDATION_RECEIPT_SCHEMA:
        raise R289BO250Error("r298 validation receipt schema drifted")
    sources = receipt.get("canonical_sources")
    if not isinstance(sources, Mapping):
        raise R289BO250Error("r298 validation receipt lacks canonical sources")
    contract_source = sources.get("goal_contract")
    gateway_source = sources.get("goal_gateway")
    if (
        not isinstance(contract_source, Mapping)
        or contract_source.get("sha256") != goal_contract_identity.get("sha256")
        or contract_source.get("size_bytes") != goal_contract_identity.get("size_bytes")
        or contract_source.get("sha256") != DERIVATIVE_GOAL_CONTRACT_SHA256
        or not isinstance(gateway_source, Mapping)
        or gateway_source.get("sha256") != DERIVATIVE_GOAL_GATEWAY_SHA256
        or receipt.get("goal_contract") != gateway_source
        or receipt.get("source_attachment")
        != {"sha256": DERIVATIVE_OWNER_ATTACHMENT_SHA256}
        or receipt.get("mechanics_attachment")
        != {"sha256": DERIVATIVE_MECHANICS_ATTACHMENT_SHA256}
    ):
        raise R289BO250Error("r298 validation receipt source bindings drifted")
    candidate = receipt.get("candidate")
    baseline = receipt.get("baseline_r274")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("candidate_id") != "alakazam-elmo-public-rule-derivative-g1"
        or candidate.get("checkpoint_sha256") != candidate_checkpoint.get("sha256")
        or candidate.get("checkpoint_size_bytes") != candidate_checkpoint.get("size_bytes")
        or candidate.get("frozen_backbone") is not True
        or candidate.get("zero_gated_layer_off_baseline_parity") is not True
        or not _is_sha256(candidate.get("derivative_identity_sha256"))
        or not isinstance(baseline, Mapping)
        or baseline.get("checkpoint_sha256") != baseline_r274_checkpoint.get("sha256")
        or baseline.get("checkpoint_size_bytes") != baseline_r274_checkpoint.get("size_bytes")
        or baseline.get("candidate_id") != "alakazam-new-list-direct-policy-r274"
    ):
        raise R289BO250Error("r298 validation receipt checkpoint bindings drifted")
    normalized_seeds = {
        comparison_id: _require_digest(seed, label=f"comparison {comparison_id} seed")
        for comparison_id, seed in comparison_seed_identities.items()
    }
    if set(normalized_seeds) != set(COMPARISON_IDS):
        raise R289BO250Error("r298 validation receipt needs all A/B/C seed identities")
    disjointness = receipt.get("evaluation_disjointness")
    excluded = (
        set(disjointness.get("benchmark_seed_identities_excluded", ()))
        if isinstance(disjointness, Mapping)
        else set()
    )
    if (
        not isinstance(disjointness, Mapping)
        or disjointness.get("candidate_training_source_disjoint") is not True
        or not set(normalized_seeds.values()).issubset(excluded)
    ):
        raise R289BO250Error(
            "r298 candidate training/benchmark seed disjointness is not receipt-bound"
        )
    # Revision 2 makes the complete 30-day archive and measured Elmo resource
    # envelope launch gates, not descriptive provenance.  Keep this check in
    # the BO consumer as well as the r298 compiler so a stale compiler cannot
    # accidentally issue a receipt that this runner treats as sufficient.
    raw_corpus = receipt.get("raw_expert_corpus")
    resource = receipt.get("resource_receipt")
    if (
        not isinstance(raw_corpus, Mapping)
        or raw_corpus.get("window_start_utc") != R298_RAW_WINDOW_START_UTC
        or raw_corpus.get("window_end_utc") != R298_RAW_WINDOW_END_UTC
        or raw_corpus.get("utc_partition_count") != R298_RAW_PARTITION_COUNT
        or raw_corpus.get("distinct_utc_partitions") is not True
        or raw_corpus.get("contiguous_window") is not True
        or raw_corpus.get("validated_deduplicated_manifest_sha256") is None
        or not _is_sha256(raw_corpus.get("validated_deduplicated_manifest_sha256"))
        or raw_corpus.get("twenty_day_or_subset_fallback") is not False
        or raw_corpus.get("source_day_group_disjoint") is not True
        or not isinstance(resource, Mapping)
        or resource.get("host") != "elmo"
        or resource.get("experiment_ram_hard_ceiling_bytes")
        != ELMO_EXPERIMENT_RAM_CEILING_BYTES
        or resource.get("memory_heavy_phase_concurrency") != 1
        or resource.get("measured_peak_receipt") is not True
        or resource.get("managed_service_changed") is not False
    ):
        raise R289BO250Error(
            "r298 validation receipt lacks the exact 30-day/raw-split/resource launch proof"
        )
    # A frozen trace schema is not a live decision-time evaluator.  Rev4
    # therefore requires a separately sealed wrapper evidence lane before a
    # benchmark is allowed to call the candidate's checklist route.  In
    # particular, the legacy r288 guide/CSV implementation can never be
    # smuggled in merely because all of its scalar gates happen to be zero.
    checklist = receipt.get("checklist_provenance")
    checklist_evidence = checklist.get("evidence") if isinstance(checklist, Mapping) else None
    checklist_config = (
        checklist.get("strict_provenance_config") if isinstance(checklist, Mapping) else None
    )
    if (
        not isinstance(checklist, Mapping)
        or checklist.get("schema")
        != "poke_bot.alakazam_simulator_rules_r298_checklist_provenance_evidence/v1"
        or checklist.get("status") != "passed"
        or checklist.get("candidate_action_time_wrapper_sealed") is not True
        or checklist.get("acting_player_public_information_only") is not True
        or checklist.get("strict_q1_q2_q8_provenance_enforced") is not True
        or checklist.get("q5_q6_exact_zero") is not True
        or checklist.get("legacy_r288_runtime_residual") != 0.0
        or not isinstance(checklist_evidence, Mapping)
        or not _is_sha256(checklist_evidence.get("sha256"))
        or type(checklist_evidence.get("size_bytes")) is not int
        or checklist_evidence.get("size_bytes") <= 0
        or checklist.get("evidence_sha256") != checklist_evidence.get("sha256")
        or not isinstance(checklist_config, Mapping)
        or checklist_config.get("canonical_contract_sha256")
        != DERIVATIVE_GOAL_CONTRACT_SHA256
        or checklist_config.get("runtime_wired") is not False
    ):
        raise R289BO250Error(
            "r298 receipt lacks a sealed strict action-time checklist wrapper"
        )
    transfer = receipt.get("transfer_staging")
    if (
        not isinstance(transfer, Mapping)
        or transfer.get("source_remote_parity_passed") is not True
        or transfer.get("inzi_loading_or_execution_authority") is not False
        or transfer.get("transfer_to_inzi_quarantine_only") is not True
        or not isinstance(transfer.get("per_shard_receipt_identities"), list)
        or not transfer["per_shard_receipt_identities"]
        or not isinstance(transfer.get("parity_receipt_identity"), Mapping)
        or not _is_sha256(transfer["parity_receipt_identity"].get("sha256"))
    ):
        raise R289BO250Error(
            "r298 receipt lacks validated quarantined create-only transfer parity"
        )
    return file_identity(resolved, label="r298 validation receipt")


def validate_r298_raw_corpus_receipt(path: Path | str) -> dict[str, Any]:
    """Validate the known typed receipt for the exact revision-4 raw-corpus gate."""

    resolved, payload = _read_object(path, label="r298 raw corpus receipt")
    source_disjointness = payload.get("source_disjointness")
    authority = payload.get("runtime_authority")
    if (
        payload.get("schema")
        != "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_receipt/v1"
        or payload.get("status") != "passed"
        or payload.get("owner_goal_sha256") != DERIVATIVE_OWNER_ATTACHMENT_SHA256
        or payload.get("rule_derivative_contract_sha256")
        != DERIVATIVE_GOAL_CONTRACT_SHA256
        or payload.get("rule_derivative_gateway_sha256")
        != DERIVATIVE_GOAL_GATEWAY_SHA256
        or payload.get("mechanics_attachment_sha256")
        != DERIVATIVE_MECHANICS_ATTACHMENT_SHA256
        or payload.get("window_start_utc") != R298_RAW_WINDOW_START_UTC
        or payload.get("window_end_utc") != R298_RAW_WINDOW_END_UTC
        or payload.get("distinct_utc_day_count") != R298_RAW_PARTITION_COUNT
        or not _is_sha256(payload.get("raw_expert_corpus_manifest_sha256"))
        or payload.get("recollection_authorized") is not False
        or not isinstance(source_disjointness, Mapping)
        or source_disjointness.get("archive_date_source_sha256_unique") is not True
        or source_disjointness.get("episode_identity_unique") is not True
        or source_disjointness.get("episode_id_content_unique") is not True
        or source_disjointness.get("source_window_blending_permitted") is not False
        or not isinstance(authority, Mapping)
        or authority.get("elmo_only") is not True
        or authority.get("create_only") is not True
        or authority.get("production_activation") is not False
        or authority.get("inzi_mutation") is not False
        or authority.get("archive_mutation") is not False
    ):
        raise R289BO250Error("r298 raw corpus receipt is not the exact revision-4 gate")
    return file_identity(resolved, label="r298 raw corpus receipt")


def validate_r298_collision_census_receipt(
    path: Path | str,
    *,
    raw_corpus_receipt_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate full collision-census completion against the exact raw gate."""

    resolved, payload = _read_object(path, label="r298 collision census receipt")
    raw = payload.get("raw_expert_corpus")
    pinned = payload.get("pinned_simulator")
    authority = payload.get("runtime_authority")
    if (
        payload.get("schema") != "poke_bot.alakazam_collision_census_r298_receipt/v1"
        or payload.get("status") != "passed_no_actionable_public_semantic_collision"
        or payload.get("pass") is not True
        or payload.get("owner_goal_sha256") != DERIVATIVE_OWNER_ATTACHMENT_SHA256
        or payload.get("rule_derivative_contract_sha256")
        != DERIVATIVE_GOAL_CONTRACT_SHA256
        or payload.get("rule_derivative_gateway_sha256")
        != DERIVATIVE_GOAL_GATEWAY_SHA256
        or payload.get("mechanics_attachment_sha256")
        != DERIVATIVE_MECHANICS_ATTACHMENT_SHA256
        or payload.get("raw_expert_corpus_receipt_sha256")
        != raw_corpus_receipt_identity.get("sha256")
        or not isinstance(raw, Mapping)
        or raw.get("window_start_utc") != R298_RAW_WINDOW_START_UTC
        or raw.get("window_end_utc") != R298_RAW_WINDOW_END_UTC
        or raw.get("distinct_utc_day_count") != R298_RAW_PARTITION_COUNT
        or not isinstance(raw.get("source_disjointness"), Mapping)
        or raw["source_disjointness"].get("archive_date_source_sha256_unique")
        is not True
        or not isinstance(pinned, Mapping)
        or pinned.get("libcg_sha256")
        != "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
        or pinned.get("complete_evidence_required_for_pass") is not True
        or pinned.get("hidden_branching_permitted") is not False
        or not isinstance(authority, Mapping)
        or authority.get("elmo_only") is not True
        or authority.get("create_only") is not True
        or authority.get("training_eligible") is not False
        or authority.get("production_activation") is not False
        or authority.get("inzi_mutation") is not False
        or authority.get("learned_baseline_mutation") is not False
    ):
        raise R289BO250Error("r298 collision census receipt is incomplete or mismatched")
    return file_identity(resolved, label="r298 collision census receipt")


def validate_candidate_receipt(path: Path | str, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the sealed r284 iter-0 receipt or an explicit r289 parity receipt."""

    resolved, payload = _read_object(path, label="r274 candidate receipt")
    expected_identity = {
        "sha256": R274_ITER0_CHECKPOINT_SHA256,
        "size_bytes": R274_ITER0_CHECKPOINT_BYTES,
    }
    if payload.get("schema") == "poke_bot.alakazam_r274_iteration_1_boundary_r284/v1":
        if (
            payload.get("status") != "authorized"
            or payload.get("run_name") != "alakazam_new_list_direct_policy_r274"
            or payload.get("candidate_checkpoint_sha256") != expected_identity["sha256"]
            or payload.get("candidate_checkpoint_size_bytes") != expected_identity["size_bytes"]
            or payload.get("combo_state_loss_weight") != 0.0
            or payload.get("combo_state_route_enabled") is not False
            or payload.get("rl_updates_exact") != 20
        ):
            raise R289BO250Error("r284 candidate receipt does not bind current r274 iter-0")
    elif payload.get("schema") == "poke_bot.alakazam_turn_checklist_bo250_r289_candidate_parity/v1":
        artifact = payload.get("candidate")
        direct = payload.get("direct_policy_boundary")
        if (
            payload.get("status") != "passed"
            or not isinstance(artifact, Mapping)
            or artifact.get("sha256") != expected_identity["sha256"]
            or artifact.get("size_bytes") != expected_identity["size_bytes"]
            or not isinstance(direct, Mapping)
            or any(direct.get(key) is not False for key in ("rtp", "search", "mcts", "rollout", "hidden_information_inference"))
            or payload.get("checklist_layer_configured") is not True
        ):
            raise R289BO250Error("r289 candidate parity receipt is incomplete")
    else:
        raise R289BO250Error("candidate receipt has an unsupported schema")
    if checkpoint.get("sha256") != expected_identity["sha256"] or checkpoint.get("size_bytes") != expected_identity["size_bytes"]:
        raise R289BO250Error("candidate checkpoint does not match its sealed r274 receipt")
    return file_identity(resolved, label="r274 candidate receipt")


def validate_r195_contract(path: Path | str, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the immutable r195 control checkpoint and explicit NO-RTP bundle."""

    resolved, payload = _read_object(path, label="r195 NO-RTP contract")
    completion = payload.get("completion")
    if (
        payload.get("schema") != "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195/v1"
        or payload.get("owner_decision_revision") != 195
        or not isinstance(completion, Mapping)
        or completion.get("expert_checkpoint_sha256") != R195_CHECKPOINT_SHA256
        or completion.get("expert_checkpoint_bytes") != R195_CHECKPOINT_BYTES
        or completion.get("no_rtp_bundle_sha256") != R195_NO_RTP_BUNDLE_SHA256
        or not isinstance(completion.get("no_rtp_submission"), Mapping)
        or completion["no_rtp_submission"].get("submission_id") != R195_SUBMISSION_ID
    ):
        raise R289BO250Error("r195 contract does not bind the immutable NO-RTP control")
    if checkpoint.get("sha256") != R195_CHECKPOINT_SHA256 or checkpoint.get("size_bytes") != R195_CHECKPOINT_BYTES:
        raise R289BO250Error("control checkpoint is not immutable r195 NO-RTP")
    return file_identity(resolved, label="r195 NO-RTP contract")


def validate_seeded_engine_receipt(
    path: Path | str,
    *,
    engine_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an immutable external proof before using a seeded diagnostic ABI.

    The r289 runner never treats an unseeded official ``BattleStart`` as a
    seed-matched experiment.  A local engine receipt is therefore mandatory;
    this function also makes the receipt bind the exact loaded shared object.
    """

    resolved, payload = _read_object(path, label="r289 seeded-engine receipt")
    engine = payload.get("engine")
    direct = payload.get("direct_policy_boundary")
    if (
        payload.get("schema") != R289_SEEDED_ENGINE_RECEIPT_SCHEMA
        or payload.get("status") != "passed"
        or not isinstance(engine, Mapping)
        or engine.get("sha256") != engine_identity.get("sha256")
        or engine.get("size_bytes") != engine_identity.get("size_bytes")
        or engine.get("battle_start_seeded_available") is not True
        or engine.get("hidden_snapshot_api_available") is not False
        or not isinstance(direct, Mapping)
        or any(direct.get(key) is not False for key in ("rtp", "search", "mcts", "rollout", "hidden_information_inference"))
        or payload.get("training_eligible") is not False
        or payload.get("production_authority") is not False
    ):
        raise R289BO250Error("seeded-engine receipt is missing a fail-closed r289 proof")
    return file_identity(resolved, label="r289 seeded-engine receipt")


def validate_optional_calibration_receipt(
    path: Path | str | None,
    *,
    evaluation_seed_identity_sha256: str,
) -> dict[str, Any] | None:
    """Accept only a source/seed-disjoint scalar-only r288 calibration receipt."""

    _require_digest(evaluation_seed_identity_sha256, label="evaluation seed identity")
    if path is None:
        return None
    resolved, payload = _read_object(path, label="r288 optional calibration receipt")
    excluded_raw = (
        payload.get("bo250_seed_identities_excluded")
        or payload.get("evaluation_seed_identities_excluded")
        or payload.get("reserved_evaluation_seed_identities")
    )
    excluded = set(excluded_raw) if isinstance(excluded_raw, list) else set()
    source_disjoint = (
        payload.get("source_disjoint_exact_new_list_data") is True
        or payload.get("source_disjoint") is True
    )
    frozen = (
        payload.get("all_neural_model_and_checkpoint_tensors_frozen") is True
        or payload.get("neural_tensors_frozen") is True
    )
    attestations = payload.get("attestations")
    calibration_config = payload.get("config")
    overlap = payload.get("overlap_deduplication")
    if (
        payload.get("schema") != R289_CALIBRATION_RECEIPT_SCHEMA
        or payload.get("status") not in {"passed", "completed"}
        or source_disjoint is not True
        or frozen is not True
        or payload.get("bo250_seed_disjoint") is not True
        or payload.get("training_eligible") is not False
        or payload.get("replay_eligible") is not False
        or payload.get("production_authority") is not False
        or evaluation_seed_identity_sha256 not in excluded
        or not isinstance(attestations, Mapping)
        or not isinstance(calibration_config, Mapping)
        or calibration_config.get("corrected_guide_attachment_sha256")
        != R295_CORRECTED_GUIDE_ATTACHMENT_SHA256
        or not isinstance(overlap, Mapping)
        or overlap.get("post_deduplication_vectors_required") is not True
        or overlap.get("post_deduplication_vectors_attested_for_all_rows") is not True
        or overlap.get("existing_learned_logic_modified") is not False
        or attestations.get("exactly_six_checklist_scalar_gates_fitted") is not True
        or attestations.get("all_eight_checklist_channels_traced") is not True
        or attestations.get("bench_prize_exposure_gate_trace_only_exact_zero")
        is not True
        or attestations.get("immediate_disruption_outcome_gate_trace_only_exact_zero")
        is not True
        or attestations.get("guide_gate_separate_trace_only_exact_zero") is not True
        or attestations.get("guide_support_calibration_eligible") is not False
        or attestations.get("strongest_per_group_aggregation_exact") is not True
        or attestations.get("winner_identity_trace_bound_in_artifact") is not True
        or attestations.get("post_deduplication_vectors_only") is not True
        or attestations.get("corrected_r295_guide_attachment_bound") is not True
        or attestations.get("input_training_eligible") is not False
        or attestations.get("input_production_eligible") is not False
        or attestations.get("input_replay_eligible") is not False
        or attestations.get("learner_or_checkpoint_training_authority") is not False
        or attestations.get("runtime_activation_authority") is not False
    ):
        raise R289BO250Error(
            "optional calibration is not proven scalar-only and BO250 seed/source-disjoint"
        )
    return file_identity(resolved, label="r288 optional calibration receipt")


def validate_required_calibration_receipt(
    path: Path | str,
    *,
    comparison_seed_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Require scalar calibration to exclude every scheduled A/B/C seed.

    Parsing a dormant checklist config remains useful for unit and shadow-trace
    work, but a benchmark may not treat calibration as optional.  In
    particular, excluding only a user-facing root seed is insufficient once
    the runner derives independent seed namespaces for all three cohorts.
    """

    normalized = {
        comparison_id: _require_digest(seed, label=f"comparison {comparison_id} seed")
        for comparison_id, seed in comparison_seed_identities.items()
    }
    if set(normalized) != set(COMPARISON_IDS):
        raise R289BO250Error("calibration receipt needs all A/B/C seed identities")
    identity = validate_optional_calibration_receipt(
        path,
        # The existing receipt parser validates its scalar/frozen/overlap
        # guarantees; the loop below extends its historical one-seed contract
        # to all independently scheduled cohorts.
        evaluation_seed_identity_sha256=next(iter(normalized.values())),
    )
    if identity is None:  # defensive: the non-optional path type forbids it.
        raise R289BO250Error("BO250 calibration receipt is mandatory before launch")
    _, payload = _read_object(path, label="r288 required calibration receipt")
    excluded_raw = (
        payload.get("bo250_seed_identities_excluded")
        or payload.get("evaluation_seed_identities_excluded")
        or payload.get("reserved_evaluation_seed_identities")
    )
    excluded = set(excluded_raw) if isinstance(excluded_raw, list) else set()
    if not set(normalized.values()).issubset(excluded):
        raise R289BO250Error(
            "calibration receipt does not exclude every A/B/C BO250 seed identity"
        )
    return identity


def comparison_evaluation_id(comparison_id: str) -> str:
    if comparison_id not in COMPARISON_EVALUATION_IDS:
        raise R289BO250Error(f"unknown r289 comparison {comparison_id!r}")
    return COMPARISON_EVALUATION_IDS[comparison_id]


def derive_comparison_seed_identity(
    benchmark_seed_identity_sha256: str, *, comparison_id: str
) -> str:
    """Derive an independent public seed namespace for one declared cohort."""

    _require_digest(benchmark_seed_identity_sha256, label="benchmark root seed identity")
    return canonical_digest(
        {
            "schema": R289_SCHEMA,
            "evaluation_id": EVALUATION_ID,
            "benchmark_seed_identity_sha256": benchmark_seed_identity_sha256,
            "comparison_id": comparison_id,
            "comparison_evaluation_id": comparison_evaluation_id(comparison_id),
        }
    )


def build_schedule(
    seed_identity_sha256: str, *, comparison_id: str = "A"
) -> tuple[SeededMirrorGameSpec, ...]:
    """Build one exact 125-pair / 250-game cohort schedule.

    ``seed_identity_sha256`` is the already-derived comparison seed identity,
    not the root seed submitted to the runner.  That distinction makes seed
    disjointness auditable in the r298 and calibration receipts.
    """

    _require_digest(seed_identity_sha256, label="r289 comparison seed identity")
    evaluation_id = comparison_evaluation_id(comparison_id)
    try:
        schedule = build_seeded_seat_swapped_schedule(
            evaluation_id=evaluation_id,
            seed_identity_sha256=seed_identity_sha256,
            pair_count=PAIR_COUNT,
        )
    except SeededMirrorHarnessError as exc:
        raise R289BO250Error(f"cannot build r289 seed schedule: {exc}") from exc
    validate_schedule(
        schedule, seed_identity_sha256=seed_identity_sha256, comparison_id=comparison_id
    )
    return schedule


def validate_schedule(
    schedule: Sequence[SeededMirrorGameSpec],
    *,
    seed_identity_sha256: str,
    comparison_id: str = "A",
) -> None:
    """Reject a schedule that cannot prove exact pairs/seats before games start."""

    _require_digest(seed_identity_sha256, label="r289 comparison seed identity")
    evaluation_id = comparison_evaluation_id(comparison_id)
    if len(schedule) != GAME_COUNT:
        raise R289BO250Error("r289 schedule must contain exactly 250 games")
    try:
        expected = build_seeded_seat_swapped_schedule(
            evaluation_id=evaluation_id,
            seed_identity_sha256=seed_identity_sha256,
            pair_count=PAIR_COUNT,
        )
    except SeededMirrorHarnessError as exc:
        raise R289BO250Error("cannot rebuild the deterministic r289 schedule") from exc
    if tuple(schedule) != tuple(expected):
        raise R289BO250Error("r289 schedule drifted from its deterministic seed identity")
    grouped: dict[int, list[SeededMirrorGameSpec]] = defaultdict(list)
    for spec in schedule:
        if spec.evaluation_id != evaluation_id or spec.pair_index not in range(PAIR_COUNT):
            raise R289BO250Error("r289 schedule contains a foreign pair")
        grouped[spec.pair_index].append(spec)
    if sorted(grouped) != list(range(PAIR_COUNT)):
        raise R289BO250Error("r289 schedule pair indices are incomplete")
    for pair_index, pair in grouped.items():
        ordered = sorted(pair, key=lambda row: row.game_index)
        if len(ordered) != 2 or [row.game_index for row in ordered] != [0, 1]:
            raise R289BO250Error(f"r289 pair {pair_index} is not a two-game cell")
        first, second = ordered
        if (
            first.pair_id != second.pair_id
            or first.pair_nonce_sha256 != second.pair_nonce_sha256
            or first.engine_seed_u32 != second.engine_seed_u32
            or first.deck_order_seed_u32 != second.deck_order_seed_u32
            or first.experimental_seat != 0
            or second.experimental_seat != 1
            or first.control_seat != 1
            or second.control_seat != 0
        ):
            raise R289BO250Error(f"r289 pair {pair_index} lost common seed/seat swap")
    if sum(spec.experimental_seat == 0 for spec in schedule) != PAIR_COUNT:
        raise R289BO250Error("candidate seat-0 count is not 125")
    if sum(spec.experimental_seat == 1 for spec in schedule) != PAIR_COUNT:
        raise R289BO250Error("candidate seat-1 count is not 125")


def schedule_identity(
    schedule: Sequence[SeededMirrorGameSpec],
    *,
    seed_identity_sha256: str,
    comparison_id: str = "A",
) -> str:
    validate_schedule(
        schedule, seed_identity_sha256=seed_identity_sha256, comparison_id=comparison_id
    )
    return canonical_digest(
        {
            "schema": R289_SCHEMA,
            "evaluation_id": comparison_evaluation_id(comparison_id),
            "comparison_id": comparison_id,
            "seed_identity_sha256": seed_identity_sha256,
            "games": [spec.as_payload() for spec in schedule],
        }
    )


def build_run_identity(
    *,
    config_identity: Mapping[str, Any],
    owner_contract_identity: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
    control_checkpoint: Mapping[str, Any],
    r195_contract: Mapping[str, Any],
    r195_bundle: Mapping[str, Any],
    exact_deck: Mapping[str, Any],
    checklist_config: Mapping[str, Any],
    candidate_matchup_tree: Mapping[str, Any],
    control_matchup_tree: Mapping[str, Any],
    seeded_engine: Mapping[str, Any],
    seeded_engine_receipt: Mapping[str, Any],
    r293_overlap_audit_receipt: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any] | None,
    runtime_sources: Mapping[str, Mapping[str, Any]],
    seed_identity_sha256: str,
    schedule_sha256: str,
) -> str:
    """Derive the content-addressed diagnostic output identity from all inputs."""

    for label, identity in (
        ("config", config_identity),
        ("owner contract", owner_contract_identity),
        ("candidate checkpoint", candidate_checkpoint),
        ("candidate receipt", candidate_receipt),
        ("control checkpoint", control_checkpoint),
        ("r195 contract", r195_contract),
        ("r195 bundle", r195_bundle),
        ("deck", exact_deck),
        ("checklist config", checklist_config),
        ("candidate matchup tree", candidate_matchup_tree),
        ("control matchup tree", control_matchup_tree),
        ("seeded engine", seeded_engine),
        ("seeded engine receipt", seeded_engine_receipt),
        ("r293 overlap audit receipt", r293_overlap_audit_receipt),
    ):
        if not isinstance(identity, Mapping) or not _is_sha256(identity.get("sha256")):
            raise R289BO250Error(f"{label} lacks a file identity")
    _require_digest(seed_identity_sha256, label="r289 seed identity")
    _require_digest(schedule_sha256, label="r289 schedule identity")
    if calibration_receipt is not None and not _is_sha256(calibration_receipt.get("sha256")):
        raise R289BO250Error("calibration receipt lacks a file identity")
    if not isinstance(runtime_sources, Mapping) or not runtime_sources:
        raise R289BO250Error("runtime source manifest must be nonempty")
    source_identities: dict[str, dict[str, Any]] = {}
    for relative, identity in sorted(runtime_sources.items()):
        if not isinstance(relative, str) or not relative or not isinstance(identity, Mapping):
            raise R289BO250Error("runtime source manifest is malformed")
        if not _is_sha256(identity.get("sha256")) or not isinstance(
            identity.get("size_bytes"), int
        ):
            raise R289BO250Error(f"runtime source {relative} lacks a file identity")
        source_identities[relative] = _content_identity(identity)
    return canonical_digest(
        {
            "schema": R289_SCHEMA,
            "evaluation_id": EVALUATION_ID,
            "config": _content_identity(config_identity),
            "owner_contract": _content_identity(owner_contract_identity),
            "candidate": {
                "checkpoint": _content_identity(candidate_checkpoint),
                "receipt": _content_identity(candidate_receipt),
                "matchup_tree": _content_identity(candidate_matchup_tree),
                "checklist_config": _content_identity(checklist_config),
            },
            "control": {
                "checkpoint": _content_identity(control_checkpoint),
                "r195_contract": _content_identity(r195_contract),
                "no_rtp_bundle": _content_identity(r195_bundle),
                "matchup_tree": _content_identity(control_matchup_tree),
            },
            "exact_new_list": _content_identity(exact_deck),
            "seeded_engine": {
                "engine": _content_identity(seeded_engine),
                "receipt": _content_identity(seeded_engine_receipt),
            },
            "r293_overlap_audit_receipt": _content_identity(
                r293_overlap_audit_receipt
            ),
            "calibration_receipt": (
                None
                if calibration_receipt is None
                else _content_identity(calibration_receipt)
            ),
            "runtime_sources": source_identities,
            "seed_identity_sha256": seed_identity_sha256,
            "schedule_sha256": schedule_sha256,
            "training_eligible": False,
            "promotion_authority": False,
            "kaggle_authority": False,
            "production_authority": False,
        }
    )


def build_three_cohort_run_identity(
    *,
    config_identity: Mapping[str, Any],
    goal_contract_identity: Mapping[str, Any],
    r298_validation_receipt: Mapping[str, Any],
    raw_corpus_receipt: Mapping[str, Any],
    collision_census_receipt: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    baseline_r274_checkpoint: Mapping[str, Any],
    r195_checkpoint: Mapping[str, Any],
    r195_contract: Mapping[str, Any],
    r195_bundle: Mapping[str, Any],
    exact_new_list_deck: Mapping[str, Any],
    r195_native_deck: Mapping[str, Any],
    checklist_config: Mapping[str, Any],
    candidate_matchup_tree: Mapping[str, Any],
    baseline_r274_matchup_tree: Mapping[str, Any],
    r195_matchup_tree: Mapping[str, Any],
    seeded_engine: Mapping[str, Any],
    seeded_engine_receipt: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
    calibration_artifact: Mapping[str, Any],
    runtime_sources: Mapping[str, Mapping[str, Any]],
    benchmark_seed_identity_sha256: str,
    comparison_seed_identities: Mapping[str, str],
    comparison_schedule_sha256s: Mapping[str, str],
) -> str:
    """Content-address all immutable inputs for the complete 3×BO250 study."""

    required_identities = {
        "config": config_identity,
        "goal contract": goal_contract_identity,
        "r298 validation receipt": r298_validation_receipt,
        "raw corpus receipt": raw_corpus_receipt,
        "collision census receipt": collision_census_receipt,
        "candidate checkpoint": candidate_checkpoint,
        "baseline r274 checkpoint": baseline_r274_checkpoint,
        "r195 checkpoint": r195_checkpoint,
        "r195 contract": r195_contract,
        "r195 NO-RTP bundle": r195_bundle,
        "exact new-list deck": exact_new_list_deck,
        "r195 native deck": r195_native_deck,
        "checklist config": checklist_config,
        "candidate matchup tree": candidate_matchup_tree,
        "baseline r274 matchup tree": baseline_r274_matchup_tree,
        "r195 matchup tree": r195_matchup_tree,
        "seeded engine": seeded_engine,
        "seeded engine receipt": seeded_engine_receipt,
        "calibration receipt": calibration_receipt,
        "calibration artifact": calibration_artifact,
    }
    for label, identity in required_identities.items():
        if not isinstance(identity, Mapping) or not _is_sha256(identity.get("sha256")):
            raise R289BO250Error(f"{label} lacks an immutable file identity")
    _require_digest(benchmark_seed_identity_sha256, label="benchmark root seed identity")
    normalized_seeds = {
        comparison_id: _require_digest(seed, label=f"comparison {comparison_id} seed")
        for comparison_id, seed in comparison_seed_identities.items()
    }
    normalized_schedules = {
        comparison_id: _require_digest(
            digest, label=f"comparison {comparison_id} schedule identity"
        )
        for comparison_id, digest in comparison_schedule_sha256s.items()
    }
    if set(normalized_seeds) != set(COMPARISON_IDS) or set(normalized_schedules) != set(
        COMPARISON_IDS
    ):
        raise R289BO250Error("complete r289 run identity needs every A/B/C cohort")
    if not isinstance(runtime_sources, Mapping) or not runtime_sources:
        raise R289BO250Error("runtime source manifest must be nonempty")
    sources: dict[str, dict[str, Any]] = {}
    for relative, identity in sorted(runtime_sources.items()):
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(identity, Mapping)
            or not _is_sha256(identity.get("sha256"))
            or type(identity.get("size_bytes")) is not int
        ):
            raise R289BO250Error("runtime source manifest is malformed")
        sources[relative] = _content_identity(identity)
    return canonical_digest(
        {
            "schema": R289_SCHEMA,
            "evaluation_id": EVALUATION_ID,
            "config": _content_identity(config_identity),
            "goal_contract": _content_identity(goal_contract_identity),
            "r298_validation_receipt": _content_identity(r298_validation_receipt),
            "raw_corpus_receipt": _content_identity(raw_corpus_receipt),
            "collision_census_receipt": _content_identity(collision_census_receipt),
            "candidate": {
                "checkpoint": _content_identity(candidate_checkpoint),
                "checklist_config": _content_identity(checklist_config),
                "matchup_tree": _content_identity(candidate_matchup_tree),
            },
            "baseline_r274": {
                "checkpoint": _content_identity(baseline_r274_checkpoint),
                "matchup_tree": _content_identity(baseline_r274_matchup_tree),
            },
            "r195": {
                "checkpoint": _content_identity(r195_checkpoint),
                "contract": _content_identity(r195_contract),
                "no_rtp_bundle": _content_identity(r195_bundle),
                "matchup_tree": _content_identity(r195_matchup_tree),
            },
            "decks": {
                "exact_new_list": _content_identity(exact_new_list_deck),
                "r195_native": _content_identity(r195_native_deck),
            },
            "seeded_engine": {
                "engine": _content_identity(seeded_engine),
                "receipt": _content_identity(seeded_engine_receipt),
            },
            "calibration_receipt": _content_identity(calibration_receipt),
            "calibration_artifact": _content_identity(calibration_artifact),
            "runtime_sources": sources,
            "benchmark_seed_identity_sha256": benchmark_seed_identity_sha256,
            "comparison_seed_identities": normalized_seeds,
            "comparison_schedule_sha256s": normalized_schedules,
            "training_eligible": False,
            "promotion_authority": False,
            "kaggle_authority": False,
            "production_authority": False,
            "inzi_authority": False,
        }
    )


def _same_identity(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return dict(existing) == dict(expected)


def _content_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Remove mutable local provenance before deriving an output address.

    The source path remains useful audit material in the preflight record, but
    two byte-identical immutable artifacts must map to the same r289 output
    identity even when they arrived through different Elmo staging locations.
    """

    return {key: value for key, value in identity.items() if key != "path"}


def stage_verified_copy(
    source: Path | str,
    target: Path | str,
    *,
    expected_identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Create one immutable Elmo staging copy, never replacing an existing file."""

    source_identity = file_identity(source, label=f"{label} source")
    if not _same_identity(source_identity, expected_identity):
        raise R289BO250Error(f"{label} source identity changed before staging")
    destination = Path(target).expanduser()
    if destination.is_symlink():
        raise R289BO250Error(f"{label} staging target may not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = file_identity(destination, label=f"{label} existing stage")
        if existing["sha256"] != expected_identity["sha256"] or existing["size_bytes"] != expected_identity["size_bytes"]:
            raise R289BO250Error(f"{label} stage exists with different bytes")
        return existing
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.r289-", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as sink, Path(source_identity["path"]).open("rb") as stream:
            descriptor = None
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sink.write(block)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            # A concurrent resumer won the create-only publication.  It must
            # still be exactly our expected content.
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    staged = file_identity(destination, label=f"{label} staged copy")
    if staged["sha256"] != expected_identity["sha256"] or staged["size_bytes"] != expected_identity["size_bytes"]:
        raise R289BO250Error(f"{label} staging digest mismatch")
    return staged


def write_create_only_json(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Write canonical JSON once; a resume can only reuse byte-identical output."""

    target = Path(path).expanduser()
    if target.is_symlink():
        raise R289BO250Error(f"{label} target may not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(payload))
    if target.exists():
        if not target.is_file() or target.read_bytes() != encoded:
            raise R289BO250Error(f"{label} already exists with different bytes")
        return file_identity(target, label=label)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.r289-", dir=target.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if not target.is_file() or target.read_bytes() != encoded:
        raise R289BO250Error(f"{label} create-only publication did not verify")
    return file_identity(target, label=label)


def empty_checklist_telemetry() -> dict[str, Any]:
    return {
        "policy_decisions": 0,
        "factorized_stages": 0,
        "active_stages": 0,
        "residual_nonzero_stages": 0,
        "action_changed_stages": 0,
        "unavailable_stages": 0,
        "channels": {
            name: {
                "available_stages": 0,
                "unavailable_stages": 0,
                "raw_nonzero_stages": 0,
                "normalized_nonzero_stages": 0,
            }
            for name in CHECKLIST_CHANNELS
        },
        "guide_support": {
            "raw_nonzero_stages": 0,
            "normalized_nonzero_stages": 0,
        },
    }


def _valid_checklist_telemetry(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R289BO250Error("candidate checklist telemetry must be an object")
    expected = empty_checklist_telemetry()
    required_top = set(expected)
    if set(value) != required_top:
        raise R289BO250Error("candidate checklist telemetry fields drifted")
    normalized: dict[str, Any] = {}
    for key in (
        "policy_decisions",
        "factorized_stages",
        "active_stages",
        "residual_nonzero_stages",
        "action_changed_stages",
        "unavailable_stages",
    ):
        normalized[key] = _exact_int(value.get(key), label=f"telemetry {key}")
    channels = value.get("channels")
    if not isinstance(channels, Mapping) or set(channels) != set(CHECKLIST_CHANNELS):
        raise R289BO250Error("candidate telemetry must report all eight channels")
    normalized_channels: dict[str, dict[str, int]] = {}
    for name in CHECKLIST_CHANNELS:
        row = channels[name]
        if not isinstance(row, Mapping) or set(row) != {
            "available_stages",
            "unavailable_stages",
            "raw_nonzero_stages",
            "normalized_nonzero_stages",
        }:
            raise R289BO250Error(f"candidate telemetry channel {name} is malformed")
        normalized_channels[name] = {
            key: _exact_int(row[key], label=f"telemetry {name}.{key}")
            for key in row
        }
    normalized["channels"] = normalized_channels
    guide = value.get("guide_support")
    if not isinstance(guide, Mapping) or set(guide) != {
        "raw_nonzero_stages",
        "normalized_nonzero_stages",
    }:
        raise R289BO250Error("candidate guide-support telemetry is malformed")
    normalized["guide_support"] = {
        key: _exact_int(guide[key], label=f"telemetry guide_support.{key}")
        for key in guide
    }
    if normalized["active_stages"] > normalized["factorized_stages"]:
        raise R289BO250Error("active checklist stages exceed factorized stages")
    if normalized["action_changed_stages"] > normalized["factorized_stages"]:
        raise R289BO250Error("action-change stages exceed factorized stages")
    return normalized


def make_game_receipt(
    *,
    run_identity_sha256: str,
    spec: SeededMirrorGameSpec,
    first_player_seat: int,
    winner_seat: int,
    steps: int,
    checklist_telemetry: Mapping[str, Any],
    runtime_preflight_sha256: str,
    direct_policy_flags: Mapping[str, bool],
    pair_first_player_seal_sha256: str,
    stage_trace_digest: str,
) -> dict[str, Any]:
    """Build one outcome-eligible r289 game receipt after a terminal game."""

    _require_digest(run_identity_sha256, label="run identity")
    _require_digest(runtime_preflight_sha256, label="runtime preflight identity")
    _require_digest(
        pair_first_player_seal_sha256, label="pair first-player seal identity"
    )
    _require_digest(stage_trace_digest, label="stage trace digest")
    if first_player_seat not in {0, 1} or winner_seat not in {0, 1, 2}:
        raise R289BO250Error("game receipt has invalid first player or winner")
    telemetry = _valid_checklist_telemetry(checklist_telemetry)
    expected_flags = {
        "rtp": False,
        "search": False,
        "mcts": False,
        "rollout": False,
        "hidden_information_inference": False,
        "candidate_checklist_layer": True,
        "control_checklist_layer": False,
    }
    if dict(direct_policy_flags) != expected_flags:
        raise R289BO250Error("game does not prove the exact r289 direct-policy boundary")
    candidate_is_first = spec.experimental_seat == first_player_seat
    return {
        "schema": R289_GAME_RECEIPT_SCHEMA,
        "evaluation_id": spec.evaluation_id,
        "run_identity_sha256": run_identity_sha256,
        "pair_index": spec.pair_index,
        "pair_id": spec.pair_id,
        "pair_nonce_sha256": spec.pair_nonce_sha256,
        "game_index": spec.game_index,
        "game_nonce_sha256": spec.game_nonce_sha256,
        "engine_seed_u32": spec.engine_seed_u32,
        "deck_order_seed_u32": spec.deck_order_seed_u32,
        "candidate_policy_rng_seed_u32": spec.experimental_rng_seed_u32,
        "control_policy_rng_seed_u32": spec.control_rng_seed_u32,
        "candidate_seat": spec.experimental_seat,
        "control_seat": spec.control_seat,
        "first_player_seat": first_player_seat,
        "candidate_actual_turn_order": "first" if candidate_is_first else "second",
        "pair_first_player_seal_sha256": pair_first_player_seal_sha256,
        "winner_seat": winner_seat,
        "terminal_status": "completed",
        "outcome_eligible": True,
        "steps": _exact_int(steps, label="game steps", minimum=1),
        "candidate_checklist_telemetry": telemetry,
        "candidate_stage_trace_sha256": stage_trace_digest,
        "runtime_preflight_sha256": runtime_preflight_sha256,
        "direct_policy_boundary": expected_flags,
        "training_eligible": False,
        "promotion_authority": False,
        "kaggle_authority": False,
        "production_authority": False,
    }


def validate_game_receipt(
    receipt: Mapping[str, Any],
    *,
    spec: SeededMirrorGameSpec,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
) -> dict[str, Any]:
    """Validate a persisted receipt before resume or reporting can credit it."""

    first_player = receipt.get("first_player_seat")
    winner = receipt.get("winner_seat")
    steps = receipt.get("steps")
    if type(first_player) is not int or first_player not in {0, 1}:
        raise R289BO250Error("persisted game receipt has an invalid first player")
    if type(winner) is not int or winner not in {0, 1, 2}:
        raise R289BO250Error("persisted game receipt has an invalid winner")
    if type(steps) is not int or steps < 1:
        raise R289BO250Error("persisted game receipt has invalid steps")
    expected = make_game_receipt(
        run_identity_sha256=run_identity_sha256,
        spec=spec,
        first_player_seat=first_player,
        winner_seat=winner,
        steps=steps,
        checklist_telemetry=(
            receipt.get("candidate_checklist_telemetry")
            if isinstance(receipt.get("candidate_checklist_telemetry"), Mapping)
            else {}
        ),
        runtime_preflight_sha256=runtime_preflight_sha256,
        direct_policy_flags=(
            receipt.get("direct_policy_boundary")
            if isinstance(receipt.get("direct_policy_boundary"), Mapping)
            else {}
        ),
        pair_first_player_seal_sha256=str(
            receipt.get("pair_first_player_seal_sha256") or ""
        ),
        stage_trace_digest=str(receipt.get("candidate_stage_trace_sha256") or ""),
    )
    if dict(receipt) != expected:
        unknown = sorted(set(receipt) - set(expected))
        missing = sorted(set(expected) - set(receipt))
        raise R289BO250Error(
            "r289 game receipt differs from its exact schedule/runtime binding "
            f"(unknown={unknown}, missing={missing})"
        )
    return expected


def _add_telemetry(total: dict[str, Any], row: Mapping[str, Any]) -> None:
    for key in (
        "policy_decisions",
        "factorized_stages",
        "active_stages",
        "residual_nonzero_stages",
        "action_changed_stages",
        "unavailable_stages",
    ):
        total[key] += int(row[key])
    for name in CHECKLIST_CHANNELS:
        for key in total["channels"][name]:
            total["channels"][name][key] += int(row["channels"][name][key])
    for key in total["guide_support"]:
        total["guide_support"][key] += int(row["guide_support"][key])


def _outcome_for_candidate(receipt: Mapping[str, Any]) -> str:
    winner = int(receipt["winner_seat"])
    candidate = int(receipt["candidate_seat"])
    if winner == 2:
        return "ties"
    return "wins" if winner == candidate else "losses"


def _paired_hoeffding_interval(pair_scores: Sequence[float]) -> dict[str, Any]:
    """A deterministic, valid bounded-pair confidence interval.

    Each matched two-game pair contributes candidate score in ``[0, 1]``
    (win=1, tie=.5, loss=0 per game, averaged across its two games).  The
    distribution-free Hoeffding interval avoids pretending the two games in a
    pair are independent.  It is intentionally descriptive: the contract
    explicitly says a BO250 is not conclusive by itself.
    """

    if len(pair_scores) != PAIR_COUNT or any(
        not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in pair_scores
    ):
        raise R289BO250Error("paired score interval requires exactly 125 bounded pairs")
    mean = sum(pair_scores) / len(pair_scores)
    alpha = 0.05
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * len(pair_scores)))
    return {
        "method": "paired_two_game_score_hoeffding_95",
        "pair_count": len(pair_scores),
        "candidate_score_mean": mean,
        "lower": max(0.0, mean - radius),
        "upper": min(1.0, mean + radius),
        "interpretation": "descriptive_only_not_conclusive_by_itself",
    }


def compile_report(
    *,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
    seed_identity_sha256: str,
    schedule: Sequence[SeededMirrorGameSpec],
    game_receipts: Sequence[Mapping[str, Any]],
    input_identities: Mapping[str, Any],
    comparison_id: str = "A",
) -> dict[str, Any]:
    """Compile the only creditable BO250 report, rejecting partial/double rows."""

    _require_digest(run_identity_sha256, label="run identity")
    _require_digest(runtime_preflight_sha256, label="runtime preflight identity")
    _require_digest(seed_identity_sha256, label="seed identity")
    validate_schedule(
        schedule, seed_identity_sha256=seed_identity_sha256, comparison_id=comparison_id
    )
    if len(game_receipts) != GAME_COUNT:
        raise R289BO250Error("r289 report is incomplete: exactly 250 receipts are required")
    by_nonce = {spec.game_nonce_sha256: spec for spec in schedule}
    if len(by_nonce) != GAME_COUNT:
        raise R289BO250Error("r289 schedule has duplicate game nonces")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in game_receipts:
        if not isinstance(raw, Mapping):
            raise R289BO250Error("game receipt row is not an object")
        nonce = raw.get("game_nonce_sha256")
        if not isinstance(nonce, str) or nonce not in by_nonce or nonce in seen:
            raise R289BO250Error("game receipt is foreign or double-credited")
        seen.add(nonce)
        validated.append(
            validate_game_receipt(
                raw,
                spec=by_nonce[nonce],
                run_identity_sha256=run_identity_sha256,
                runtime_preflight_sha256=runtime_preflight_sha256,
            )
        )
    if set(by_nonce) != seen:
        raise R289BO250Error("r289 report does not cover the complete schedule")

    candidate = {"wins": 0, "losses": 0, "ties": 0}
    by_seat = {
        "seat_0": {"wins": 0, "losses": 0, "ties": 0},
        "seat_1": {"wins": 0, "losses": 0, "ties": 0},
    }
    by_order = {
        "actual_first": {"wins": 0, "losses": 0, "ties": 0},
        "actual_second": {"wins": 0, "losses": 0, "ties": 0},
    }
    telemetry = empty_checklist_telemetry()
    pair_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for receipt in validated:
        result = _outcome_for_candidate(receipt)
        candidate[result] += 1
        by_seat[f"seat_{receipt['candidate_seat']}"][result] += 1
        key = "actual_first" if receipt["candidate_actual_turn_order"] == "first" else "actual_second"
        by_order[key][result] += 1
        _add_telemetry(telemetry, receipt["candidate_checklist_telemetry"])
        pair_rows[int(receipt["pair_index"])].append(receipt)
    if candidate["wins"] + candidate["losses"] + candidate["ties"] != GAME_COUNT:
        raise R289BO250Error("candidate aggregate does not total 250")
    if any(sum(row.values()) != PAIR_COUNT for row in by_seat.values()):
        raise R289BO250Error("candidate seats are not exactly 125/125")
    if any(sum(row.values()) != PAIR_COUNT for row in by_order.values()):
        raise R289BO250Error("candidate actual first/second are not exactly 125/125")
    if sorted(pair_rows) != list(range(PAIR_COUNT)):
        raise R289BO250Error("r289 report pair coverage is incomplete")
    pair_integrity: list[dict[str, Any]] = []
    paired_scores: list[float] = []
    for pair_index in range(PAIR_COUNT):
        pair = sorted(pair_rows[pair_index], key=lambda item: int(item["game_index"]))
        if len(pair) != 2:
            raise R289BO250Error(f"pair {pair_index} is incomplete")
        first, second = pair
        if (
            [row["game_index"] for row in pair] != [0, 1]
            or first["engine_seed_u32"] != second["engine_seed_u32"]
            or first["deck_order_seed_u32"] != second["deck_order_seed_u32"]
            # Cohort B deliberately uses different candidate/control decks.
            # Each game therefore seals its own post-setup observation rather
            # than falsely requiring byte-identical public observations after
            # seats (and decks) are swapped.  The engine/deck-order seed and
            # resolved first-player seat remain matched pair facts.
            or first["first_player_seat"] != second["first_player_seat"]
            or {row["candidate_seat"] for row in pair} != {0, 1}
            or {row["candidate_actual_turn_order"] for row in pair} != {"first", "second"}
        ):
            raise R289BO250Error(f"pair {pair_index} lost seeded/swap/actual-order parity")
        pair_integrity.append(
            {
                "pair_index": pair_index,
                "pair_id": first["pair_id"],
                "engine_seed_u32": first["engine_seed_u32"],
                "deck_order_seed_u32": first["deck_order_seed_u32"],
                "first_player_seat": first["first_player_seat"],
                "game_first_player_seal_sha256s": [
                    first["pair_first_player_seal_sha256"],
                    second["pair_first_player_seal_sha256"],
                ],
                "candidate_seats": [0, 1],
                "candidate_actual_orders": ["first", "second"],
            }
        )
        score = 0.0
        for row in pair:
            outcome = _outcome_for_candidate(row)
            score += 1.0 if outcome == "wins" else 0.5 if outcome == "ties" else 0.0
        paired_scores.append(score / 2.0)
    paired_interval = _paired_hoeffding_interval(paired_scores)
    report = {
        "schema": R289_REPORT_SCHEMA,
        "evaluation_id": comparison_evaluation_id(comparison_id),
        "comparison_id": comparison_id,
        "run_identity_sha256": run_identity_sha256,
        "runtime_preflight_sha256": runtime_preflight_sha256,
        "seed_identity_sha256": seed_identity_sha256,
        "schedule_sha256": schedule_identity(
            schedule,
            seed_identity_sha256=seed_identity_sha256,
            comparison_id=comparison_id,
        ),
        "status": "completed_diagnostic_only",
        "pair_count": PAIR_COUNT,
        "game_count": GAME_COUNT,
        "candidate_arm": CANDIDATE_ARM,
        "control_arm": CONTROL_ARMS[comparison_id],
        "candidate_results": candidate,
        "candidate_results_by_seat": by_seat,
        "candidate_results_by_actual_turn_order": by_order,
        "paired_confidence_interval": paired_interval,
        "action_disagreements": {
            "status": "unavailable_different_trajectory_no_same_state_counterfactual",
            "candidate_checklist_stage_argmax_changes": telemetry[
                "action_changed_stages"
            ],
            "measurement_note": (
                "stage_argmax_changed_at_prefix is checklist-layer telemetry, not "
                "a cross-arm action disagreement over identical public states"
            ),
        },
        "severe_rule_error_rate": {
            "status": "unavailable_no_authoritative_rule_error_classifier",
            "unavailable_decisions": telemetry["policy_decisions"],
            "non_gating": True,
        },
        "per_channel_coverage": telemetry["channels"],
        "residual_that_changed_each_action": {
            "status": "per_decision_trace_bound",
            "trace_field": "candidate_stage_traces[].stages[].residuals",
            "change_field": "candidate_stage_traces[].stages[].stage_argmax_changed_at_prefix",
            "trace_journal_required": True,
        },
        "candidate_checklist_telemetry": telemetry,
        "pair_integrity": pair_integrity,
        "input_identities": dict(input_identities),
        "labels": {
            "training_eligible": False,
            "replay_eligible": False,
            "promotion_authority": False,
            "production_authority": False,
            "production_selector_authority": False,
            "kaggle_authority": False,
            "submission_authority": False,
            "elmo_production_readmission_authority": False,
            "candidate_checklist_layer_enabled": True,
            "control_checklist_layer_enabled": False,
            "direct_policy_only": True,
            "rtp_enabled": False,
            "mcts_enabled": False,
            "search_enabled": False,
            "rollout_enabled": False,
            "hidden_information_inference": False,
            "calibration_rows_and_seeds_disjoint": True,
            "candidate_training_and_benchmark_seeds_disjoint": True,
            "bo250_conclusive_by_itself": False,
        },
    }
    report["report_sha256"] = canonical_digest(report)
    return report


def compile_three_comparison_report(
    *,
    run_identity_sha256: str,
    benchmark_seed_identity_sha256: str,
    comparison_reports: Mapping[str, Mapping[str, Any]],
    input_identities: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the three complete cohort reports into the only study-level report."""

    _require_digest(run_identity_sha256, label="complete r289 run identity")
    _require_digest(benchmark_seed_identity_sha256, label="benchmark root seed identity")
    if set(comparison_reports) != set(COMPARISON_IDS):
        raise R289BO250Error("combined r289 report must contain A/B/C exactly once")
    normalized: dict[str, dict[str, Any]] = {}
    for comparison_id in COMPARISON_IDS:
        report = comparison_reports[comparison_id]
        if not isinstance(report, Mapping):
            raise R289BO250Error(f"comparison {comparison_id} report is not an object")
        if (
            report.get("schema") != R289_REPORT_SCHEMA
            or report.get("comparison_id") != comparison_id
            or report.get("evaluation_id") != comparison_evaluation_id(comparison_id)
            or report.get("run_identity_sha256") != run_identity_sha256
            or report.get("pair_count") != PAIR_COUNT
            or report.get("game_count") != GAME_COUNT
            or report.get("control_arm") != CONTROL_ARMS[comparison_id]
            or report.get("report_sha256")
            != canonical_digest({key: value for key, value in report.items() if key != "report_sha256"})
        ):
            raise R289BO250Error(f"comparison {comparison_id} report binding drifted")
        normalized[comparison_id] = dict(report)
    combined = {
        "schema": R289_REPORT_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "run_identity_sha256": run_identity_sha256,
        "benchmark_seed_identity_sha256": benchmark_seed_identity_sha256,
        "status": "completed_three_comparison_diagnostic_only",
        "comparison_count": len(COMPARISON_IDS),
        "pair_count_per_comparison": PAIR_COUNT,
        "game_count_per_comparison": GAME_COUNT,
        "total_game_count": TOTAL_GAME_COUNT,
        "comparisons": normalized,
        "input_identities": dict(input_identities),
        "limitations": {
            "bo250_conclusive_by_itself": False,
            "cross_arm_action_disagreements": "unavailable_without_same_public_state_counterfactual",
            "severe_rule_error_classifier": "unavailable_no_authoritative_rule_error_classifier",
            "cohort_B_deck_shift_confound": True,
            "cohort_C_deck_shift_confound": True,
        },
        "labels": {
            "training_eligible": False,
            "replay_eligible": False,
            "promotion_authority": False,
            "production_authority": False,
            "inzi_authority": False,
            "kaggle_authority": False,
            "submission_authority": False,
        },
    }
    combined["report_sha256"] = canonical_digest(combined)
    return combined


def output_index_payload(
    *,
    run_identity_sha256: str,
    output_dir: Path | str,
    schedule_sha256: str,
    input_identities: Mapping[str, Any],
    evaluation_id: str = EVALUATION_ID,
    comparison_id: str | None = None,
) -> dict[str, Any]:
    _require_digest(run_identity_sha256, label="run identity")
    _require_digest(schedule_sha256, label="schedule identity")
    if comparison_id is not None and evaluation_id != comparison_evaluation_id(comparison_id):
        raise R289BO250Error("output index comparison/evaluation binding drifted")
    payload: dict[str, Any] = {
        "schema": R289_OUTPUT_INDEX_SCHEMA,
        "evaluation_id": evaluation_id,
        "run_identity_sha256": run_identity_sha256,
        "content_addressed_output_dir": str(Path(output_dir).expanduser().resolve()),
        "schedule_sha256": schedule_sha256,
        "input_identities": dict(input_identities),
        "write_mode": "create_only_resume_without_double_credit",
        "training_eligible": False,
        "promotion_authority": False,
        "production_authority": False,
        "kaggle_authority": False,
    }
    if comparison_id is not None:
        payload["comparison_id"] = comparison_id
    return payload


__all__ = [
    "CANDIDATE_ARM",
    "CALIBRATION_ELIGIBLE_GATE_NAMES",
    "CHECKLIST_CHANNELS",
    "CHECKLIST_GATE_NAMES",
    "COMPARISON_IDS",
    "COMPARISON_EVALUATION_IDS",
    "CONTROL_ARM",
    "CONTROL_ARMS",
    "DERIVATIVE_GOAL_CONTRACT_SHA256",
    "DERIVATIVE_GOAL_GATEWAY_SHA256",
    "EVALUATION_ID",
    "EXACT_NEW_LIST_MULTISET_SHA256",
    "EXPECTED_GATE_CHANNEL_MAP",
    "FIXED_TRACE_ONLY_GATES",
    "GAME_COUNT",
    "GROUPED_RESIDUAL_AGGREGATION_KIND",
    "PAIR_COUNT",
    "TOTAL_GAME_COUNT",
    "R195_CHECKPOINT_BYTES",
    "R195_CHECKPOINT_SHA256",
    "R195_NO_RTP_BUNDLE_SHA256",
    "R274_ITER0_CHECKPOINT_BYTES",
    "R274_ITER0_CHECKPOINT_SHA256",
    "R274_DIRECT_POLICY_MATCHUP_TREE_SHA256",
    "R295_CORRECTED_GUIDE_ATTACHMENT_SHA256",
    "R289BO250Error",
    "R289_CONFIG_SCHEMA",
    "R289_GAME_RECEIPT_SCHEMA",
    "R289_R293_OVERLAP_AUDIT_RECEIPT_SCHEMA",
    "R289_REPORT_SCHEMA",
    "R289_SCHEMA",
    "R289_SEEDED_ENGINE_RECEIPT_SCHEMA",
    "R298_VALIDATION_RECEIPT_SCHEMA",
    "RESIDUAL_GROUP_GATE_MEMBERS",
    "build_run_identity",
    "build_three_cohort_run_identity",
    "build_schedule",
    "canonical_deck_multiset_sha256",
    "canonical_digest",
    "canonical_json_bytes",
    "compile_report",
    "compile_three_comparison_report",
    "comparison_evaluation_id",
    "derive_comparison_seed_identity",
    "empty_checklist_telemetry",
    "file_identity",
    "load_r289_config",
    "make_game_receipt",
    "output_index_payload",
    "read_exact_new_list_deck",
    "read_r195_native_deck",
    "schedule_identity",
    "sha256_file",
    "stage_verified_copy",
    "validate_candidate_receipt",
    "validate_checklist_config",
    "validate_derivative_goal_contract",
    "validate_game_receipt",
    "validate_optional_calibration_receipt",
    "validate_r298_collision_census_receipt",
    "validate_r298_raw_corpus_receipt",
    "validate_r298_validation_receipt",
    "validate_required_calibration_receipt",
    "validate_owner_contract",
    "validate_r293_overlap_audit_receipt",
    "validate_r195_contract",
    "validate_schedule",
    "validate_seeded_engine_receipt",
    "write_create_only_json",
]
