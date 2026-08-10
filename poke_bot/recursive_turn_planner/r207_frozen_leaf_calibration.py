"""Offline, fail-closed frozen leaf calibration for r207 BO1000.

The r207 planner may use the revision-195 model only as a frozen policy prior
and nonterminal outcome/value leaf reranker.  This module deliberately does
not import Torch, a trainer, an engine, or a service.  It consumes three sealed
``0444`` JSON artifacts: the exact r195 training-source manifest,
source-excluded terminal continuations from nonterminal heldout leaf states,
and frozen inference predictions.  It emits a self-digesting preflight receipt
only when the checkpoint, source/game split, state binding, support,
calibration, and uncertainty gates all hold.  Terminal simulator results are
labels for calibration and are never eligible for model reranking.

Attempt10 evidence, r197/r198 RTP partial rows, and either RTP sidecar are
categorically outside this protocol.  The prediction artifact must instead
bind the exact r195 NO-RTP frozen-policy identity used by both r207 priors and
nonterminal leaf reranking.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poke_bot.recursive_turn_planner.neural_leaf_reranker import (
    FrozenPolicyIdentity,
    LeafScoreCalibration,
)
from poke_bot.rtp_evaluation_immutable_io import (
    ImmutableEvidenceIOError,
    read_immutable_file_bytes,
)

R207_HELDOUT_EVIDENCE_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_frozen_terminal_leaf_heldout_evidence/v1"
)
R207_TRAINING_SOURCE_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_r195_training_source_manifest/v1"
)
R207_FROZEN_PREDICTIONS_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_frozen_terminal_leaf_predictions/v1"
)
R207_CALIBRATION_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_frozen_value_outcome_calibration_preflight/v1"
)
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
_OUTCOME_ORDER = ("loss", "draw", "win")
_OUTCOME_TARGETS = {"loss": 0.0, "draw": 0.5, "win": 1.0}
_VALUE_TARGETS = {"loss": -1.0, "draw": 0.0, "win": 1.0}
_OUTCOME_ONE_HOT = {
    outcome: tuple(1.0 if member == outcome else 0.0 for member in _OUTCOME_ORDER)
    for outcome in _OUTCOME_ORDER
}
_NONTERMINAL_LEAF_KINDS = {"successor", "boundary"}
_R207_CALIBRATION_EVIDENCE_ROLE = "r207_source_excluded_terminal_leaf_calibration"
_R195_NO_RTP_MODEL_ROLE = "r195_no_rtp_frozen_policy_and_leaf_model"


class FrozenLeafCalibrationError(ValueError):
    """The calibration preflight cannot safely support frozen reranking."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenLeafCalibrationError(f"{label} must be an object")
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenLeafCalibrationError(f"{label} must be a nonempty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label).lower()
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise FrozenLeafCalibrationError(f"{label} must be a sha256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise FrozenLeafCalibrationError(f"{label} must be a sha256 digest") from exc
    return digest


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenLeafCalibrationError(f"{label} must be an exact integer")
    if value < minimum:
        raise FrozenLeafCalibrationError(f"{label} must be at least {minimum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenLeafCalibrationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FrozenLeafCalibrationError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise FrozenLeafCalibrationError(f"{label} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise FrozenLeafCalibrationError(f"{label} must be at most {maximum}")
    return number


def _require_false_fields(
    payload: Mapping[str, Any],
    label: str,
    fields: Sequence[str],
) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise FrozenLeafCalibrationError(f"{label}.{field} must be exactly false")


def _sorted_unique_texts(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FrozenLeafCalibrationError(f"{label} must be an array")
    result = tuple(sorted(_text(item, f"{label}[]") for item in value))
    if not result or len(set(result)) != len(result):
        raise FrozenLeafCalibrationError(f"{label} must contain unique nonempty IDs")
    return result


def _identity(material: Any) -> dict[str, Any]:
    return {
        "path": str(material.path),
        "sha256": material.sha256,
        "bytes": material.bytes,
        "mode": f"{material.mode:04o}",
    }


def _read_sealed_json(path: str | Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        material = read_immutable_file_bytes(path, label, exact_mode=0o444)
    except ImmutableEvidenceIOError as exc:
        raise FrozenLeafCalibrationError(str(exc)) from exc
    try:
        payload = json.loads(material.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenLeafCalibrationError(f"{label} is not valid UTF-8 JSON") from exc
    return _mapping(payload, label), _identity(material)


@dataclass(frozen=True)
class FrozenLeafCalibrationPolicy:
    """Explicit bounds the later frozen reranker must consume verbatim."""

    min_terminal_leaves: int
    min_distinct_source_ids: int
    min_distinct_game_ids: int
    min_per_outcome: int
    min_per_source_id: int
    max_outcome_brier: float
    max_outcome_ece: float
    max_value_rmse: float
    max_value_mae: float
    value_weight: float
    outcome_weight: float
    uncertainty_quantile: float = 0.95
    ece_bins: int = 10

    def __post_init__(self) -> None:
        for name in (
            "min_terminal_leaves",
            "min_distinct_source_ids",
            "min_distinct_game_ids",
            "min_per_outcome",
            "min_per_source_id",
            "ece_bins",
        ):
            _integer(getattr(self, name), name, minimum=1)
        for name, maximum in (
            ("max_outcome_brier", 1.0),
            ("max_outcome_ece", 1.0),
            ("max_value_rmse", 2.0),
            ("max_value_mae", 2.0),
        ):
            _number(getattr(self, name), name, minimum=0.0, maximum=maximum)
        _number(
            self.uncertainty_quantile,
            "uncertainty_quantile",
            minimum=0.0,
            maximum=1.0,
        )
        if self.uncertainty_quantile <= 0.0:
            raise FrozenLeafCalibrationError("uncertainty_quantile must be positive")
        value_weight = _number(self.value_weight, "value_weight", minimum=0.0, maximum=1.0)
        outcome_weight = _number(
            self.outcome_weight,
            "outcome_weight",
            minimum=0.0,
            maximum=1.0,
        )
        if value_weight <= 0.0 or outcome_weight <= 0.0:
            raise FrozenLeafCalibrationError(
                "r207 requires nonzero fixed value and outcome reranker weights"
            )
        if not math.isclose(value_weight + outcome_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise FrozenLeafCalibrationError("value_weight and outcome_weight must sum exactly to one")
        if self.min_terminal_leaves < self.min_distinct_game_ids:
            raise FrozenLeafCalibrationError(
                "min_terminal_leaves cannot be smaller than min_distinct_game_ids"
            )
        if self.min_terminal_leaves < 3 * self.min_per_outcome:
            raise FrozenLeafCalibrationError(
                "min_terminal_leaves cannot be smaller than three outcome support floors"
            )

    def as_payload(self) -> dict[str, Any]:
        return {
            "min_terminal_leaves": self.min_terminal_leaves,
            "min_distinct_source_ids": self.min_distinct_source_ids,
            "min_distinct_game_ids": self.min_distinct_game_ids,
            "min_per_outcome": self.min_per_outcome,
            "min_per_source_id": self.min_per_source_id,
            "max_outcome_brier": self.max_outcome_brier,
            "max_outcome_ece": self.max_outcome_ece,
            "max_value_rmse": self.max_value_rmse,
            "max_value_mae": self.max_value_mae,
            "value_weight": self.value_weight,
            "outcome_weight": self.outcome_weight,
            "uncertainty_quantile": self.uncertainty_quantile,
            "ece_bins": self.ece_bins,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256({"schema": "poke_bot.r207_frozen_leaf_calibration_policy/v1", **self.as_payload()})


def _r195_training_source_manifest(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    if manifest.get("schema") != R207_TRAINING_SOURCE_MANIFEST_SCHEMA:
        raise FrozenLeafCalibrationError("r195 training source manifest schema is invalid")
    if manifest.get("status") != "sealed":
        raise FrozenLeafCalibrationError("r195 training source manifest must be sealed")
    if (
        _sha256(manifest.get("checkpoint_sha256"), "training_manifest.checkpoint_sha256")
        != R195_CHECKPOINT_SHA256
    ):
        raise FrozenLeafCalibrationError("training source manifest is not bound to exact r195")
    if (
        _integer(manifest.get("checkpoint_bytes"), "training_manifest.checkpoint_bytes", minimum=1)
        != R195_CHECKPOINT_BYTES
    ):
        raise FrozenLeafCalibrationError("training source manifest checkpoint bytes are invalid")
    if manifest.get("source_role") != "r195_training":
        raise FrozenLeafCalibrationError("training source manifest must identify r195 training sources")
    if manifest.get("training_eligible") is not True:
        raise FrozenLeafCalibrationError("training source manifest must identify training-eligible sources")
    _require_false_fields(
        manifest,
        "training_manifest",
        (
            "attempt10_rtp_partial_rows_used",
            "r197_or_r198_rtp_partial_rows_used",
            "r195_rtp_sidecar_used",
            "r197_rtp_sidecar_used",
        ),
    )
    training_sources = _sorted_unique_texts(
        manifest.get("training_source_ids"), "training_manifest.training_source_ids"
    )
    training_games = _sorted_unique_texts(
        manifest.get("training_game_ids"), "training_manifest.training_game_ids"
    )
    expected = {
        "training_source_ids_sha256": canonical_sha256(list(training_sources)),
        "training_game_ids_sha256": canonical_sha256(list(training_games)),
    }
    for key, digest in expected.items():
        if _sha256(manifest.get(key), f"training_manifest.{key}") != digest:
            raise FrozenLeafCalibrationError(
                f"training source manifest {key} does not bind declared IDs"
            )
    return training_sources, training_games, expected


def _source_exclusion(
    evidence: Mapping[str, Any],
    *,
    training_sources: tuple[str, ...],
    training_games: tuple[str, ...],
    training_manifest_identity: Mapping[str, Any],
    training_manifest_digests: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    source = _mapping(evidence.get("source_exclusion"), "heldout.source_exclusion")
    if source.get("source_disjoint") is not True or source.get("game_disjoint") is not True:
        raise FrozenLeafCalibrationError(
            "heldout source_exclusion must be source- and game-disjoint"
        )
    if (
        _sha256(
            source.get("r195_training_source_manifest_sha256"),
            "r195_training_source_manifest_sha256",
        )
        != training_manifest_identity["sha256"]
    ):
        raise FrozenLeafCalibrationError(
            "heldout source exclusion does not bind the inspected r195 training manifest"
        )
    heldout_sources = _sorted_unique_texts(
        source.get("heldout_source_ids"), "heldout_source_ids"
    )
    heldout_games = _sorted_unique_texts(
        source.get("heldout_game_ids"), "heldout_game_ids"
    )
    source_intersection = sorted(set(training_sources).intersection(heldout_sources))
    game_intersection = sorted(set(training_games).intersection(heldout_games))
    if source_intersection:
        raise FrozenLeafCalibrationError("heldout source IDs overlap r195 training source IDs")
    if game_intersection:
        raise FrozenLeafCalibrationError("heldout game IDs overlap r195 training game IDs")
    if _integer(source.get("intersection_source_id_count"), "intersection_source_id_count") != 0:
        raise FrozenLeafCalibrationError("intersection_source_id_count must be zero")
    if _integer(source.get("intersection_game_id_count"), "intersection_game_id_count") != 0:
        raise FrozenLeafCalibrationError("intersection_game_id_count must be zero")
    expected = {
        **training_manifest_digests,
        "heldout_source_ids_sha256": canonical_sha256(list(heldout_sources)),
        "heldout_game_ids_sha256": canonical_sha256(list(heldout_games)),
        "intersection_source_ids_sha256": canonical_sha256(source_intersection),
        "intersection_game_ids_sha256": canonical_sha256(game_intersection),
    }
    for key, digest in expected.items():
        if _sha256(source.get(key), key) != digest:
            raise FrozenLeafCalibrationError(f"{key} does not bind the declared source IDs")
    return heldout_sources, heldout_games, {
        "source_disjoint": True,
        "game_disjoint": True,
        "r195_training_source_manifest_sha256": training_manifest_identity["sha256"],
        "training_source_ids_sha256": expected["training_source_ids_sha256"],
        "heldout_source_ids_sha256": expected["heldout_source_ids_sha256"],
        "intersection_source_ids_sha256": expected["intersection_source_ids_sha256"],
        "intersection_source_id_count": 0,
        "training_game_ids_sha256": expected["training_game_ids_sha256"],
        "heldout_game_ids_sha256": expected["heldout_game_ids_sha256"],
        "intersection_game_ids_sha256": expected["intersection_game_ids_sha256"],
        "intersection_game_id_count": 0,
    }


def _terminal_labels(
    evidence: Mapping[str, Any],
    heldout_sources: set[str],
    heldout_games: set[str],
) -> dict[str, dict[str, Any]]:
    if evidence.get("outcome_perspective") != "candidate_root":
        raise FrozenLeafCalibrationError(
            "heldout outcomes must use the candidate_root perspective"
        )
    if evidence.get("terminal_outcomes_exact") is not True:
        raise FrozenLeafCalibrationError("heldout terminal outcomes must be exact")
    if evidence.get("nonterminal_policy_visible_leaf_predictions_only") is not True:
        raise FrozenLeafCalibrationError(
            "heldout evidence must contain only policy-visible nonterminal leaves"
        )
    rows = evidence.get("terminal_leaves")
    if not isinstance(rows, list) or not rows:
        raise FrozenLeafCalibrationError("heldout terminal_leaves must be a nonempty array")
    labels: dict[str, dict[str, Any]] = {}
    sources: set[str] = set()
    games: set[str] = set()
    decisions: set[tuple[str, int, int]] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"terminal_leaves[{index}]")
        leaf_id = _text(row.get("leaf_id"), f"terminal_leaves[{index}].leaf_id")
        source_id = _text(row.get("source_id"), f"terminal_leaves[{index}].source_id")
        game_id = _text(row.get("game_id"), f"terminal_leaves[{index}].game_id")
        decision_ordinal = _integer(
            row.get("decision_ordinal"),
            f"terminal_leaves[{index}].decision_ordinal",
            minimum=0,
        )
        root_seat = _integer(
            row.get("root_seat"), f"terminal_leaves[{index}].root_seat", minimum=0
        )
        if root_seat not in {0, 1}:
            raise FrozenLeafCalibrationError("terminal leaf root_seat must be 0 or 1")
        horizon_atomic_actions = _integer(
            row.get("horizon_atomic_actions"),
            f"terminal_leaves[{index}].horizon_atomic_actions",
            minimum=1,
        )
        leaf_kind = _text(row.get("leaf_kind"), f"terminal_leaves[{index}].leaf_kind")
        public_observation_sha256 = _sha256(
            row.get("public_observation_sha256"),
            f"terminal_leaves[{index}].public_observation_sha256",
        )
        legal_actions_sha256 = _sha256(
            row.get("legal_actions_sha256"),
            f"terminal_leaves[{index}].legal_actions_sha256",
        )
        nonterminal_leaf_state_sha256 = _sha256(
            row.get("nonterminal_leaf_state_sha256"),
            f"terminal_leaves[{index}].nonterminal_leaf_state_sha256",
        )
        outcome = _text(row.get("terminal_outcome"), f"terminal_leaves[{index}].terminal_outcome")
        result_kind = _text(
            row.get("terminal_result_kind"),
            f"terminal_leaves[{index}].terminal_result_kind",
        )
        result_sha256 = _sha256(
            row.get("terminal_result_sha256"),
            f"terminal_leaves[{index}].terminal_result_sha256",
        )
        if leaf_id in labels:
            raise FrozenLeafCalibrationError("terminal leaf IDs must be unique")
        if source_id not in heldout_sources:
            raise FrozenLeafCalibrationError("terminal leaf is not from the declared heldout split")
        if game_id not in heldout_games:
            raise FrozenLeafCalibrationError("terminal leaf game is not from the declared heldout split")
        if game_id in games:
            raise FrozenLeafCalibrationError(
                "each heldout game may contribute exactly one calibration leaf"
            )
        decision_key = (game_id, decision_ordinal, root_seat)
        if decision_key in decisions:
            raise FrozenLeafCalibrationError("heldout game/decision rows must be unique")
        if outcome not in _OUTCOME_TARGETS:
            raise FrozenLeafCalibrationError("terminal_outcome must be win, draw, or loss")
        if row.get("policy_visible") is not True or row.get("is_terminal") is not False:
            raise FrozenLeafCalibrationError(
                "calibration predictions must come from policy-visible nonterminal leaves"
            )
        if leaf_kind not in _NONTERMINAL_LEAF_KINDS:
            raise FrozenLeafCalibrationError(
                "calibration leaf_kind must be successor or boundary, never terminal"
            )
        if result_kind != "exact_simulator_terminal":
            raise FrozenLeafCalibrationError(
                "terminal_result_kind must be exact_simulator_terminal"
            )
        labels[leaf_id] = {
            "source_id": source_id,
            "game_id": game_id,
            "decision_ordinal": decision_ordinal,
            "root_seat": root_seat,
            "horizon_atomic_actions": horizon_atomic_actions,
            "leaf_kind": leaf_kind,
            "public_observation_sha256": public_observation_sha256,
            "legal_actions_sha256": legal_actions_sha256,
            "nonterminal_leaf_state_sha256": nonterminal_leaf_state_sha256,
            "outcome": outcome,
            "outcome_target": _OUTCOME_TARGETS[outcome],
            "value_target": _VALUE_TARGETS[outcome],
            "one_hot_outcome_target": _OUTCOME_ONE_HOT[outcome],
            "terminal_result_sha256": result_sha256,
        }
        sources.add(source_id)
        games.add(game_id)
        decisions.add(decision_key)
    if sources != heldout_sources:
        raise FrozenLeafCalibrationError(
            "terminal leaves must cover exactly the declared heldout source IDs"
        )
    if games != heldout_games:
        raise FrozenLeafCalibrationError(
            "terminal leaves must cover exactly the declared heldout game IDs"
        )
    declared = evidence.get("terminal_leaf_count")
    if declared is not None and _integer(declared, "terminal_leaf_count", minimum=1) != len(labels):
        raise FrozenLeafCalibrationError("terminal_leaf_count does not match terminal_leaves")
    return labels


def _outcome_probabilities(value: Any, label: str) -> tuple[float, float, float]:
    payload = _mapping(value, label)
    if set(payload) != set(_OUTCOME_ORDER):
        raise FrozenLeafCalibrationError(
            f"{label} must contain exactly loss, draw, and win probabilities"
        )
    probabilities = tuple(
        _number(payload[outcome], f"{label}.{outcome}", minimum=0.0, maximum=1.0)
        for outcome in _OUTCOME_ORDER
    )
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise FrozenLeafCalibrationError(f"{label} must sum exactly to one")
    return probabilities


def _r195_frozen_policy_identity() -> FrozenPolicyIdentity:
    """The only policy identity r207 may use for priors or leaf reranking."""

    return FrozenPolicyIdentity.r205_no_rtp()


def _frozen_policy_identity_payload(identity: FrozenPolicyIdentity) -> dict[str, Any]:
    return {
        "checkpoint_sha256": identity.checkpoint_sha256,
        "checkpoint_bytes": identity.checkpoint_bytes,
        "bundle_sha256": identity.bundle_sha256,
        "deck_cards_sha256": identity.deck_cards_sha256,
        "nonplanner_runtime_sha256": identity.nonplanner_runtime_sha256,
        "identity_sha256": identity.identity_sha256,
    }


def _predictions(
    payload: Mapping[str, Any],
    *,
    evidence_sha256: str,
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if payload.get("schema") != R207_FROZEN_PREDICTIONS_SCHEMA:
        raise FrozenLeafCalibrationError("predictions schema is not r207 frozen terminal leaf predictions")
    if payload.get("status") != "frozen_inference_complete":
        raise FrozenLeafCalibrationError("predictions must be frozen_inference_complete")
    if payload.get("outcome_perspective") != "candidate_root":
        raise FrozenLeafCalibrationError(
            "predictions must use the candidate_root outcome perspective"
        )
    if tuple(payload.get("outcome_class_order", ())) != _OUTCOME_ORDER:
        raise FrozenLeafCalibrationError(
            "predictions outcome_class_order must be loss, draw, win"
        )
    if payload.get("model_role") != _R195_NO_RTP_MODEL_ROLE:
        raise FrozenLeafCalibrationError("predictions must use the exact r195 NO-RTP model role")
    _require_false_fields(
        payload,
        "predictions",
        (
            "attempt10_rtp_partial_rows_used",
            "r197_or_r198_rtp_partial_rows_used",
            "r195_rtp_sidecar_used",
            "r197_rtp_sidecar_used",
        ),
    )
    identity = _r195_frozen_policy_identity()
    if (
        _sha256(
            payload.get("frozen_policy_identity_sha256"),
            "predictions.frozen_policy_identity_sha256",
        )
        != identity.identity_sha256
    ):
        raise FrozenLeafCalibrationError(
            "predictions are not from the exact r195 frozen policy identity"
        )
    for field, expected in (
        ("r195_no_rtp_bundle_sha256", identity.bundle_sha256),
        ("r195_no_rtp_deck_cards_sha256", identity.deck_cards_sha256),
        ("r195_no_rtp_runtime_sha256", identity.nonplanner_runtime_sha256),
    ):
        if _sha256(payload.get(field), f"predictions.{field}") != expected:
            raise FrozenLeafCalibrationError(
                f"predictions {field} does not bind exact r195 NO-RTP model identity"
            )
    if _sha256(payload.get("checkpoint_sha256"), "predictions.checkpoint_sha256") != R195_CHECKPOINT_SHA256:
        raise FrozenLeafCalibrationError("predictions are not from the exact r195 checkpoint")
    if _integer(payload.get("checkpoint_bytes"), "predictions.checkpoint_bytes", minimum=1) != R195_CHECKPOINT_BYTES:
        raise FrozenLeafCalibrationError("predictions checkpoint bytes do not match r195")
    if _sha256(payload.get("heldout_evidence_sha256"), "predictions.heldout_evidence_sha256") != evidence_sha256:
        raise FrozenLeafCalibrationError("predictions do not bind the heldout evidence bytes")
    for key in ("model_frozen", "training_performed"):
        expected = key == "model_frozen"
        if payload.get(key) is not expected:
            raise FrozenLeafCalibrationError(f"predictions.{key} is invalid")
    if _integer(payload.get("optimizer_steps"), "predictions.optimizer_steps") != 0:
        raise FrozenLeafCalibrationError("frozen prediction artifact has optimizer steps")
    if _integer(payload.get("gradient_updates"), "predictions.gradient_updates") != 0:
        raise FrozenLeafCalibrationError("frozen prediction artifact has gradient updates")
    if payload.get("training_eligible") is not False:
        raise FrozenLeafCalibrationError("prediction artifact must be training-ineligible")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise FrozenLeafCalibrationError("predictions must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"predictions[{index}]")
        leaf_id = _text(row.get("leaf_id"), f"predictions[{index}].leaf_id")
        if leaf_id in result:
            raise FrozenLeafCalibrationError("prediction leaf IDs must be unique")
        label = labels.get(leaf_id)
        if label is None:
            raise FrozenLeafCalibrationError("prediction references a leaf absent from heldout evidence")
        source_id = _text(row.get("source_id"), f"predictions[{index}].source_id")
        if source_id != label["source_id"]:
            raise FrozenLeafCalibrationError("prediction source ID does not match heldout leaf")
        bindings = {
            "game_id": _text(row.get("game_id"), f"predictions[{index}].game_id"),
            "decision_ordinal": _integer(
                row.get("decision_ordinal"),
                f"predictions[{index}].decision_ordinal",
                minimum=0,
            ),
            "root_seat": _integer(
                row.get("root_seat"), f"predictions[{index}].root_seat", minimum=0
            ),
            "horizon_atomic_actions": _integer(
                row.get("horizon_atomic_actions"),
                f"predictions[{index}].horizon_atomic_actions",
                minimum=1,
            ),
            "leaf_kind": _text(row.get("leaf_kind"), f"predictions[{index}].leaf_kind"),
            "public_observation_sha256": _sha256(
                row.get("public_observation_sha256"),
                f"predictions[{index}].public_observation_sha256",
            ),
            "legal_actions_sha256": _sha256(
                row.get("legal_actions_sha256"),
                f"predictions[{index}].legal_actions_sha256",
            ),
            "nonterminal_leaf_state_sha256": _sha256(
                row.get("nonterminal_leaf_state_sha256"),
                f"predictions[{index}].nonterminal_leaf_state_sha256",
            ),
        }
        if bindings["root_seat"] not in {0, 1}:
            raise FrozenLeafCalibrationError("prediction root_seat must be 0 or 1")
        if bindings["leaf_kind"] not in _NONTERMINAL_LEAF_KINDS:
            raise FrozenLeafCalibrationError("prediction leaf_kind must remain nonterminal")
        for key, observed in bindings.items():
            if observed != label[key]:
                raise FrozenLeafCalibrationError(
                    f"prediction {key} does not match heldout nonterminal leaf"
                )
        if _sha256(
            row.get("terminal_result_sha256"),
            f"predictions[{index}].terminal_result_sha256",
        ) != label["terminal_result_sha256"]:
            raise FrozenLeafCalibrationError(
                "prediction terminal result identity does not match heldout leaf"
            )
        result[leaf_id] = {
            "outcome_probabilities": _outcome_probabilities(
                row.get("outcome_probabilities"),
                f"predictions[{index}].outcome_probabilities",
            ),
            "value": _number(
                row.get("value"),
                f"predictions[{index}].value",
                minimum=-1.0,
                maximum=1.0,
            ),
        }
    if set(result) != set(labels):
        raise FrozenLeafCalibrationError("predictions must cover exactly every heldout terminal leaf")
    return result


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise FrozenLeafCalibrationError("cannot compute a quantile without values")
    rank = max(0, math.ceil(quantile * len(values)) - 1)
    return sorted(values)[rank]


def _measurement(
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    policy: FrozenLeafCalibrationPolicy,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    outcome_distribution_errors: list[float] = []
    outcome_expected_value_errors: list[float] = []
    value_errors: list[float] = []
    squared_outcome_errors: list[float] = []
    squared_value_errors: list[float] = []
    by_outcome: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_game: Counter[str] = Counter()
    class_bins: dict[str, list[list[tuple[float, float]]]] = {
        outcome: [[] for _ in range(policy.ece_bins)] for outcome in _OUTCOME_ORDER
    }
    for leaf_id, label in labels.items():
        prediction = predictions[leaf_id]
        target_probabilities = label["one_hot_outcome_target"]
        probabilities = prediction["outcome_probabilities"]
        outcome_error = sum(
            (predicted - target) ** 2
            for predicted, target in zip(probabilities, target_probabilities)
        ) / len(_OUTCOME_ORDER)
        outcome_distribution_error = sum(
            abs(predicted - target)
            for predicted, target in zip(probabilities, target_probabilities)
        ) / len(_OUTCOME_ORDER)
        outcome_expected = probabilities[2] - probabilities[0]
        outcome_expected_error = abs(outcome_expected - label["value_target"])
        value_error = abs(prediction["value"] - label["value_target"])
        outcome_distribution_errors.append(outcome_distribution_error)
        outcome_expected_value_errors.append(outcome_expected_error)
        value_errors.append(value_error)
        squared_outcome_errors.append(outcome_error)
        squared_value_errors.append(value_error * value_error)
        by_outcome[label["outcome"]] += 1
        by_source[label["source_id"]] += 1
        by_game[label["game_id"]] += 1
        for index, outcome in enumerate(_OUTCOME_ORDER):
            probability = probabilities[index]
            bin_index = min(int(probability * policy.ece_bins), policy.ece_bins - 1)
            class_bins[outcome][bin_index].append((probability, target_probabilities[index]))
    count = len(labels)
    ece_by_outcome: dict[str, float] = {}
    bin_receipts: list[dict[str, Any]] = []
    for outcome in _OUTCOME_ORDER:
        bins = class_bins[outcome]
        outcome_ece = sum(
            len(items) / count * abs(
                sum(prediction for prediction, _ in items) / len(items)
                - sum(target for _, target in items) / len(items)
            )
            for items in bins
            if items
        )
        ece_by_outcome[outcome] = outcome_ece
        bin_receipts.append(
            {
                "outcome": outcome,
                "ece": outcome_ece,
                "bins": [
                    {
                        "index": index,
                        "count": len(items),
                        "mean_prediction": (
                            sum(prediction for prediction, _ in items) / len(items)
                            if items
                            else None
                        ),
                        "mean_terminal_target": (
                            sum(target for _, target in items) / len(items)
                            if items
                            else None
                        ),
                    }
                    for index, items in enumerate(bins)
                ],
            }
        )
    ece = sum(ece_by_outcome.values()) / len(_OUTCOME_ORDER)
    calibration = {
        "outcome_multiclass_brier": sum(squared_outcome_errors) / count,
        "outcome_classwise_ece": ece,
        "outcome_expected_value_rmse": math.sqrt(
            sum(error * error for error in outcome_expected_value_errors) / count
        ),
        "value_rmse": math.sqrt(sum(squared_value_errors) / count),
        "value_mae": sum(value_errors) / count,
        "outcome_classwise_ece_bins": bin_receipts,
    }
    uncertainty = {
        "method": "empirical_absolute_residual_nearest_rank",
        "quantile": policy.uncertainty_quantile,
        "outcome_distribution_mean_absolute_error_bound": _quantile(
            outcome_distribution_errors, policy.uncertainty_quantile
        ),
        "outcome_expected_value_absolute_error_bound": _quantile(
            outcome_expected_value_errors, policy.uncertainty_quantile
        ),
        "value_absolute_error_bound": _quantile(value_errors, policy.uncertainty_quantile),
    }
    support = {
        "terminal_leaf_count": count,
        "distinct_source_id_count": len(by_source),
        "distinct_game_id_count": len(by_game),
        "per_outcome": dict(sorted(by_outcome.items())),
        "minimum_per_source_id_observed": min(by_source.values()),
        "maximum_leaves_per_game_observed": max(by_game.values()),
        "configured_minimum_terminal_leaves": policy.min_terminal_leaves,
        "configured_minimum_distinct_source_ids": policy.min_distinct_source_ids,
        "configured_minimum_distinct_game_ids": policy.min_distinct_game_ids,
        "configured_minimum_per_outcome": policy.min_per_outcome,
        "configured_minimum_per_source_id": policy.min_per_source_id,
    }
    failures: list[str] = []
    if count < policy.min_terminal_leaves:
        failures.append("terminal_leaf_support")
    if len(by_source) < policy.min_distinct_source_ids:
        failures.append("distinct_source_support")
    if len(by_game) < policy.min_distinct_game_ids:
        failures.append("distinct_game_support")
    if max(by_game.values()) != 1:
        failures.append("multiple_correlated_leaves_per_game")
    if min(by_source.values()) < policy.min_per_source_id:
        failures.append("per_source_support")
    for outcome in sorted(_OUTCOME_TARGETS):
        if by_outcome[outcome] < policy.min_per_outcome:
            failures.append(f"{outcome}_support")
    for metric, threshold in (
        ("outcome_multiclass_brier", policy.max_outcome_brier),
        ("outcome_classwise_ece", policy.max_outcome_ece),
        ("value_rmse", policy.max_value_rmse),
        ("value_mae", policy.max_value_mae),
    ):
        if calibration[metric] > threshold:
            failures.append(metric)
    return calibration, uncertainty, support, failures


def compile_r207_frozen_leaf_calibration_preflight(
    heldout_evidence_path: str | Path,
    frozen_predictions_path: str | Path,
    r195_training_source_manifest_path: str | Path,
    *,
    policy: FrozenLeafCalibrationPolicy,
) -> dict[str, Any]:
    """Compile a receipt or raise before any frozen reranker can be used.

    All input paths must be physical, descriptor-read, exact-``0444`` JSON
    files.  The training manifest independently fixes the r195 source set, so
    heldout exclusion cannot be asserted only by the candidate artifact.  The
    compiler never invokes model inference and cannot train.
    """

    training_manifest, training_manifest_identity = _read_sealed_json(
        r195_training_source_manifest_path, "r195 training source manifest"
    )
    training_sources, training_games, training_manifest_digests = (
        _r195_training_source_manifest(training_manifest)
    )
    evidence, evidence_identity = _read_sealed_json(heldout_evidence_path, "heldout evidence")
    if evidence.get("schema") != R207_HELDOUT_EVIDENCE_SCHEMA:
        raise FrozenLeafCalibrationError("heldout evidence schema is not r207 terminal leaf evidence")
    if evidence.get("status") != "sealed":
        raise FrozenLeafCalibrationError("heldout evidence must be sealed")
    if evidence.get("evaluation_only") is not True or evidence.get("training_eligible") is not False:
        raise FrozenLeafCalibrationError("heldout evidence must be evaluation-only and training-ineligible")
    if evidence.get("replay_eligible") is not False:
        raise FrozenLeafCalibrationError("heldout evidence must be replay-ineligible")
    if evidence.get("evidence_role") != _R207_CALIBRATION_EVIDENCE_ROLE:
        raise FrozenLeafCalibrationError("heldout evidence role is not the isolated r207 calibration")
    _require_false_fields(
        evidence,
        "heldout evidence",
        (
            "attempt10_rtp_partial_rows_used",
            "r197_or_r198_rtp_partial_rows_used",
            "r195_rtp_sidecar_used",
            "r197_rtp_sidecar_used",
        ),
    )
    heldout_sources, heldout_games, source_exclusion = _source_exclusion(
        evidence,
        training_sources=training_sources,
        training_games=training_games,
        training_manifest_identity=training_manifest_identity,
        training_manifest_digests=training_manifest_digests,
    )
    labels = _terminal_labels(evidence, set(heldout_sources), set(heldout_games))
    predictions_payload, predictions_identity = _read_sealed_json(
        frozen_predictions_path, "frozen predictions"
    )
    predictions = _predictions(
        predictions_payload,
        evidence_sha256=evidence_identity["sha256"],
        labels=labels,
    )
    calibration, uncertainty, support, failures = _measurement(labels, predictions, policy)
    if failures:
        raise FrozenLeafCalibrationError(
            "frozen leaf calibration preflight failed closed: " + ", ".join(failures)
        )
    frozen_policy_identity = _r195_frozen_policy_identity()
    leaf_score_calibration = {
        "frozen_policy_identity_sha256": frozen_policy_identity.identity_sha256,
        "value_weight": policy.value_weight,
        "outcome_weight": policy.outcome_weight,
        "value_head_calibrated": True,
        "outcome_distribution_calibrated": True,
        "source_excluded": True,
        "policy_visible_only": True,
        "evaluation_only": True,
        "training_eligible": False,
        "serving_action_authority": False,
        "outcome_class_order": list(_OUTCOME_ORDER),
    }
    receipt: dict[str, Any] = {
        "schema": R207_CALIBRATION_RECEIPT_SCHEMA,
        "status": "passed",
        "checkpoint": {
            "sha256": R195_CHECKPOINT_SHA256,
            "bytes": R195_CHECKPOINT_BYTES,
            "same_frozen_checkpoint_used_for_policy_priors_and_outcome_value_reranking": True,
        },
        "inputs": {
            "r195_training_source_manifest": training_manifest_identity,
            "heldout_evidence": evidence_identity,
            "frozen_predictions": predictions_identity,
        },
        "frozen_policy_identity": _frozen_policy_identity_payload(frozen_policy_identity),
        "source_exclusion": {
            **source_exclusion,
            "training_source_id_count": len(training_sources),
            "heldout_source_id_count": len(heldout_sources),
            "training_game_id_count": len(training_games),
            "heldout_game_id_count": len(heldout_games),
            "evaluation_only": True,
            "training_eligible": False,
            "attempt10_rtp_partial_rows_used": False,
            "r197_or_r198_rtp_partial_rows_used": False,
            "r195_rtp_sidecar_used": False,
            "r197_rtp_sidecar_used": False,
        },
        "frozen_inference": {
            "model_frozen": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "gradient_updates": 0,
            "terminal_exact_results_model_reranked": False,
            "nonterminal_outcome_value_leaf_reranking_only": True,
            "outcome_class_order": list(_OUTCOME_ORDER),
            "nonterminal_policy_visible_leaf_predictions_only": True,
            "model_role": _R195_NO_RTP_MODEL_ROLE,
            "r195_no_rtp_bundle_sha256": frozen_policy_identity.bundle_sha256,
            "r195_no_rtp_deck_cards_sha256": frozen_policy_identity.deck_cards_sha256,
            "r195_no_rtp_runtime_sha256": frozen_policy_identity.nonplanner_runtime_sha256,
            "attempt10_rtp_partial_rows_used": False,
            "r197_or_r198_rtp_partial_rows_used": False,
            "r195_rtp_sidecar_used": False,
            "r197_rtp_sidecar_used": False,
        },
        "policy": {**policy.as_payload(), "sha256": policy.sha256},
        "leaf_score_calibration": leaf_score_calibration,
        "calibration": calibration,
        "uncertainty": uncertainty,
        "support": support,
        "reranker_bounds": {
            "receipt_required_before_nonterminal_frozen_outcome_value_reranking": True,
            "simulator_terminal_results_must_remain_exact": True,
            "outcome_multiclass_brier_upper_bound": policy.max_outcome_brier,
            "outcome_classwise_ece_upper_bound": policy.max_outcome_ece,
            "value_rmse_upper_bound": policy.max_value_rmse,
            "value_mae_upper_bound": policy.max_value_mae,
            "outcome_distribution_mean_absolute_error_bound": uncertainty[
                "outcome_distribution_mean_absolute_error_bound"
            ],
            "outcome_expected_value_absolute_error_bound": uncertainty[
                "outcome_expected_value_absolute_error_bound"
            ],
            "value_absolute_error_bound": uncertainty["value_absolute_error_bound"],
            "minimum_terminal_leaf_support": policy.min_terminal_leaves,
            "observed_terminal_leaf_support": support["terminal_leaf_count"],
            "minimum_distinct_game_support": policy.min_distinct_game_ids,
            "observed_distinct_game_support": support["distinct_game_id_count"],
            "maximum_leaves_per_game": 1,
            "value_weight": policy.value_weight,
            "outcome_weight": policy.outcome_weight,
        },
        "authority": {
            "frozen_nonterminal_leaf_reranking_preflight_eligible": True,
            "training_authorized": False,
            "gradient_or_optimizer_update_authorized": False,
            "serving_authorized": False,
            "action_authority_enabled": False,
            "selector_change_authorized": False,
            "promotion_authorized": False,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _receipt_file_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _mapping(value, label)
    if set(identity) != {"path", "sha256", "bytes", "mode"}:
        raise FrozenLeafCalibrationError(f"{label} has unexpected or missing identity fields")
    path = Path(_text(identity.get("path"), f"{label}.path"))
    if not path.is_absolute() or ".." in path.parts:
        raise FrozenLeafCalibrationError(f"{label}.path must be an absolute lexical path")
    if identity.get("mode") != "0444":
        raise FrozenLeafCalibrationError(f"{label}.mode must be exactly 0444")
    return {
        "path": str(path),
        "sha256": _sha256(identity.get("sha256"), f"{label}.sha256"),
        "bytes": _integer(identity.get("bytes"), f"{label}.bytes", minimum=1),
        "mode": "0444",
    }


def _policy_from_receipt(value: Any) -> FrozenLeafCalibrationPolicy:
    payload = _mapping(value, "receipt.policy")
    expected_keys = {
        "min_terminal_leaves",
        "min_distinct_source_ids",
        "min_distinct_game_ids",
        "min_per_outcome",
        "min_per_source_id",
        "max_outcome_brier",
        "max_outcome_ece",
        "max_value_rmse",
        "max_value_mae",
        "value_weight",
        "outcome_weight",
        "uncertainty_quantile",
        "ece_bins",
        "sha256",
    }
    if set(payload) != expected_keys:
        raise FrozenLeafCalibrationError("receipt.policy has unexpected or missing fields")
    policy = FrozenLeafCalibrationPolicy(
        min_terminal_leaves=_integer(
            payload.get("min_terminal_leaves"), "receipt.policy.min_terminal_leaves", minimum=1
        ),
        min_distinct_source_ids=_integer(
            payload.get("min_distinct_source_ids"),
            "receipt.policy.min_distinct_source_ids",
            minimum=1,
        ),
        min_distinct_game_ids=_integer(
            payload.get("min_distinct_game_ids"),
            "receipt.policy.min_distinct_game_ids",
            minimum=1,
        ),
        min_per_outcome=_integer(
            payload.get("min_per_outcome"), "receipt.policy.min_per_outcome", minimum=1
        ),
        min_per_source_id=_integer(
            payload.get("min_per_source_id"),
            "receipt.policy.min_per_source_id",
            minimum=1,
        ),
        max_outcome_brier=_number(
            payload.get("max_outcome_brier"),
            "receipt.policy.max_outcome_brier",
            minimum=0.0,
            maximum=1.0,
        ),
        max_outcome_ece=_number(
            payload.get("max_outcome_ece"),
            "receipt.policy.max_outcome_ece",
            minimum=0.0,
            maximum=1.0,
        ),
        max_value_rmse=_number(
            payload.get("max_value_rmse"),
            "receipt.policy.max_value_rmse",
            minimum=0.0,
            maximum=2.0,
        ),
        max_value_mae=_number(
            payload.get("max_value_mae"),
            "receipt.policy.max_value_mae",
            minimum=0.0,
            maximum=2.0,
        ),
        value_weight=_number(
            payload.get("value_weight"),
            "receipt.policy.value_weight",
            minimum=0.0,
            maximum=1.0,
        ),
        outcome_weight=_number(
            payload.get("outcome_weight"),
            "receipt.policy.outcome_weight",
            minimum=0.0,
            maximum=1.0,
        ),
        uncertainty_quantile=_number(
            payload.get("uncertainty_quantile"),
            "receipt.policy.uncertainty_quantile",
            minimum=0.0,
            maximum=1.0,
        ),
        ece_bins=_integer(payload.get("ece_bins"), "receipt.policy.ece_bins", minimum=1),
    )
    if _sha256(payload.get("sha256"), "receipt.policy.sha256") != policy.sha256:
        raise FrozenLeafCalibrationError("receipt.policy.sha256 does not bind policy fields")
    return policy


def verify_r207_frozen_leaf_calibration_receipt(
    receipt: Mapping[str, Any],
    *,
    heldout_evidence_path: str | Path | None = None,
    frozen_predictions_path: str | Path | None = None,
    r195_training_source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless a compiled receipt retains its r207 safety meaning.

    Supplying all paths additionally descriptor-reads the exact sealed inputs
    and recompiles the receipt, so a later consumer can bind the receipt to the
    same immutable evidence rather than trusting a stale pathname.
    """

    supplied_paths = (
        heldout_evidence_path,
        frozen_predictions_path,
        r195_training_source_manifest_path,
    )
    if any(path is None for path in supplied_paths) and any(
        path is not None for path in supplied_paths
    ):
        raise FrozenLeafCalibrationError(
            "r195 training manifest, heldout evidence, and frozen predictions paths must be supplied together"
        )
    payload = _mapping(receipt, "receipt")
    digest = _sha256(payload.pop("receipt_sha256", None), "receipt_sha256")
    if canonical_sha256(payload) != digest:
        raise FrozenLeafCalibrationError("receipt digest does not match its canonical content")
    if (
        payload.get("schema") != R207_CALIBRATION_RECEIPT_SCHEMA
        or payload.get("status") != "passed"
    ):
        raise FrozenLeafCalibrationError("receipt schema or status is invalid")
    checkpoint = _mapping(payload.get("checkpoint"), "receipt.checkpoint")
    if (
        _sha256(checkpoint.get("sha256"), "receipt.checkpoint.sha256")
        != R195_CHECKPOINT_SHA256
    ):
        raise FrozenLeafCalibrationError("receipt checkpoint is not exact r195")
    if (
        _integer(checkpoint.get("bytes"), "receipt.checkpoint.bytes", minimum=1)
        != R195_CHECKPOINT_BYTES
    ):
        raise FrozenLeafCalibrationError("receipt checkpoint bytes are not exact r195")
    if (
        checkpoint.get(
            "same_frozen_checkpoint_used_for_policy_priors_and_outcome_value_reranking"
        )
        is not True
    ):
        raise FrozenLeafCalibrationError("receipt does not bind one frozen checkpoint")

    inputs = _mapping(payload.get("inputs"), "receipt.inputs")
    if set(inputs) != {
        "r195_training_source_manifest",
        "heldout_evidence",
        "frozen_predictions",
    }:
        raise FrozenLeafCalibrationError("receipt.inputs has unexpected or missing fields")
    training_manifest_identity = _receipt_file_identity(
        inputs.get("r195_training_source_manifest"),
        "receipt.inputs.r195_training_source_manifest",
    )
    heldout_identity = _receipt_file_identity(
        inputs.get("heldout_evidence"), "receipt.inputs.heldout_evidence"
    )
    prediction_identity = _receipt_file_identity(
        inputs.get("frozen_predictions"), "receipt.inputs.frozen_predictions"
    )

    source = _mapping(payload.get("source_exclusion"), "receipt.source_exclusion")
    if (
        source.get("source_disjoint") is not True
        or source.get("game_disjoint") is not True
        or source.get("training_eligible") is not False
    ):
        raise FrozenLeafCalibrationError("receipt source exclusion is invalid")
    if source.get("evaluation_only") is not True:
        raise FrozenLeafCalibrationError("receipt source exclusion must be evaluation-only")
    _require_false_fields(
        source,
        "receipt.source_exclusion",
        (
            "attempt10_rtp_partial_rows_used",
            "r197_or_r198_rtp_partial_rows_used",
            "r195_rtp_sidecar_used",
            "r197_rtp_sidecar_used",
        ),
    )
    if (
        _integer(source.get("intersection_source_id_count"), "receipt.intersection_source_id_count")
        != 0
    ):
        raise FrozenLeafCalibrationError("receipt has source overlap")
    if (
        _integer(source.get("intersection_game_id_count"), "receipt.intersection_game_id_count")
        != 0
    ):
        raise FrozenLeafCalibrationError("receipt has game overlap")
    if (
        _sha256(
            source.get("r195_training_source_manifest_sha256"),
            "receipt.r195_training_source_manifest_sha256",
        )
        != training_manifest_identity["sha256"]
    ):
        raise FrozenLeafCalibrationError("receipt source exclusion does not bind training manifest")
    for key in (
        "training_source_ids_sha256",
        "heldout_source_ids_sha256",
        "intersection_source_ids_sha256",
        "training_game_ids_sha256",
        "heldout_game_ids_sha256",
        "intersection_game_ids_sha256",
    ):
        _sha256(source.get(key), f"receipt.source_exclusion.{key}")
    _integer(source.get("training_source_id_count"), "receipt.training_source_id_count", minimum=1)
    _integer(source.get("heldout_source_id_count"), "receipt.heldout_source_id_count", minimum=1)
    _integer(source.get("training_game_id_count"), "receipt.training_game_id_count", minimum=1)
    _integer(source.get("heldout_game_id_count"), "receipt.heldout_game_id_count", minimum=1)

    frozen = _mapping(payload.get("frozen_inference"), "receipt.frozen_inference")
    if frozen.get("model_frozen") is not True or frozen.get("training_performed") is not False:
        raise FrozenLeafCalibrationError("receipt does not bind frozen inference")
    if (
        _integer(frozen.get("optimizer_steps"), "receipt.optimizer_steps") != 0
        or _integer(frozen.get("gradient_updates"), "receipt.gradient_updates") != 0
    ):
        raise FrozenLeafCalibrationError("receipt permits model updates")
    if frozen.get("terminal_exact_results_model_reranked") is not False:
        raise FrozenLeafCalibrationError("receipt permits terminal result reranking")
    if frozen.get("nonterminal_outcome_value_leaf_reranking_only") is not True:
        raise FrozenLeafCalibrationError("receipt does not limit reranking to nonterminal leaves")
    if tuple(frozen.get("outcome_class_order", ())) != _OUTCOME_ORDER:
        raise FrozenLeafCalibrationError("receipt outcome class order is invalid")
    if frozen.get("nonterminal_policy_visible_leaf_predictions_only") is not True:
        raise FrozenLeafCalibrationError(
            "receipt does not bind nonterminal policy-visible prediction states"
        )
    expected_frozen_identity = _frozen_policy_identity_payload(
        _r195_frozen_policy_identity()
    )
    if frozen.get("model_role") != _R195_NO_RTP_MODEL_ROLE:
        raise FrozenLeafCalibrationError("receipt frozen model role is not exact r195 NO-RTP")
    for field, identity_key in (
        ("r195_no_rtp_bundle_sha256", "bundle_sha256"),
        ("r195_no_rtp_deck_cards_sha256", "deck_cards_sha256"),
        ("r195_no_rtp_runtime_sha256", "nonplanner_runtime_sha256"),
    ):
        if (
            _sha256(frozen.get(field), f"receipt.frozen_inference.{field}")
            != expected_frozen_identity[identity_key]
        ):
            raise FrozenLeafCalibrationError(
                f"receipt {field} does not bind exact r195 NO-RTP identity"
            )
    _require_false_fields(
        frozen,
        "receipt.frozen_inference",
        (
            "attempt10_rtp_partial_rows_used",
            "r197_or_r198_rtp_partial_rows_used",
            "r195_rtp_sidecar_used",
            "r197_rtp_sidecar_used",
        ),
    )
    if payload.get("frozen_policy_identity") != expected_frozen_identity:
        raise FrozenLeafCalibrationError("receipt frozen policy identity is invalid")

    policy = _policy_from_receipt(payload.get("policy"))
    expected_leaf_score_calibration = {
        "frozen_policy_identity_sha256": expected_frozen_identity["identity_sha256"],
        "value_weight": policy.value_weight,
        "outcome_weight": policy.outcome_weight,
        "value_head_calibrated": True,
        "outcome_distribution_calibrated": True,
        "source_excluded": True,
        "policy_visible_only": True,
        "evaluation_only": True,
        "training_eligible": False,
        "serving_action_authority": False,
        "outcome_class_order": list(_OUTCOME_ORDER),
    }
    if payload.get("leaf_score_calibration") != expected_leaf_score_calibration:
        raise FrozenLeafCalibrationError(
            "receipt LeafScoreCalibration binding does not match frozen policy or weights"
        )
    calibration = _mapping(payload.get("calibration"), "receipt.calibration")
    required_calibration = {
        "outcome_multiclass_brier",
        "outcome_classwise_ece",
        "outcome_expected_value_rmse",
        "value_rmse",
        "value_mae",
        "outcome_classwise_ece_bins",
    }
    if set(calibration) != required_calibration:
        raise FrozenLeafCalibrationError("receipt.calibration has unexpected or missing fields")
    for key in required_calibration - {"outcome_classwise_ece_bins"}:
        _number(calibration.get(key), f"receipt.calibration.{key}", minimum=0.0)
    if not isinstance(calibration.get("outcome_classwise_ece_bins"), list):
        raise FrozenLeafCalibrationError("receipt calibration bins must be an array")
    if calibration["outcome_multiclass_brier"] > policy.max_outcome_brier:
        raise FrozenLeafCalibrationError("receipt outcome brier exceeds policy")
    if calibration["outcome_classwise_ece"] > policy.max_outcome_ece:
        raise FrozenLeafCalibrationError("receipt outcome ECE exceeds policy")
    if calibration["value_rmse"] > policy.max_value_rmse:
        raise FrozenLeafCalibrationError("receipt value RMSE exceeds policy")
    if calibration["value_mae"] > policy.max_value_mae:
        raise FrozenLeafCalibrationError("receipt value MAE exceeds policy")

    uncertainty = _mapping(payload.get("uncertainty"), "receipt.uncertainty")
    if uncertainty.get("method") != "empirical_absolute_residual_nearest_rank":
        raise FrozenLeafCalibrationError("receipt uncertainty method is invalid")
    if _number(uncertainty.get("quantile"), "receipt.uncertainty.quantile") != policy.uncertainty_quantile:
        raise FrozenLeafCalibrationError("receipt uncertainty quantile does not match policy")
    for key in (
        "outcome_distribution_mean_absolute_error_bound",
        "outcome_expected_value_absolute_error_bound",
        "value_absolute_error_bound",
    ):
        _number(uncertainty.get(key), f"receipt.uncertainty.{key}", minimum=0.0)

    support = _mapping(payload.get("support"), "receipt.support")
    if _integer(support.get("terminal_leaf_count"), "receipt.support.terminal_leaf_count", minimum=1) < policy.min_terminal_leaves:
        raise FrozenLeafCalibrationError("receipt terminal support is below policy")
    if _integer(support.get("distinct_source_id_count"), "receipt.support.distinct_source_id_count", minimum=1) < policy.min_distinct_source_ids:
        raise FrozenLeafCalibrationError("receipt source support is below policy")
    if _integer(support.get("distinct_game_id_count"), "receipt.support.distinct_game_id_count", minimum=1) < policy.min_distinct_game_ids:
        raise FrozenLeafCalibrationError("receipt game support is below policy")
    if _integer(support.get("minimum_per_source_id_observed"), "receipt.support.minimum_per_source_id_observed", minimum=1) < policy.min_per_source_id:
        raise FrozenLeafCalibrationError("receipt per-source support is below policy")
    if _integer(support.get("maximum_leaves_per_game_observed"), "receipt.support.maximum_leaves_per_game_observed", minimum=1) != 1:
        raise FrozenLeafCalibrationError("receipt permits correlated leaves from one game")
    per_outcome = _mapping(support.get("per_outcome"), "receipt.support.per_outcome")
    if set(per_outcome) != set(_OUTCOME_ORDER):
        raise FrozenLeafCalibrationError("receipt support outcomes are incomplete")
    for outcome in _OUTCOME_ORDER:
        if _integer(per_outcome[outcome], f"receipt.support.per_outcome.{outcome}", minimum=0) < policy.min_per_outcome:
            raise FrozenLeafCalibrationError(f"receipt {outcome} support is below policy")

    reranker_bounds = _mapping(payload.get("reranker_bounds"), "receipt.reranker_bounds")
    expected_bounds = {
        "receipt_required_before_nonterminal_frozen_outcome_value_reranking": True,
        "simulator_terminal_results_must_remain_exact": True,
        "outcome_multiclass_brier_upper_bound": policy.max_outcome_brier,
        "outcome_classwise_ece_upper_bound": policy.max_outcome_ece,
        "value_rmse_upper_bound": policy.max_value_rmse,
        "value_mae_upper_bound": policy.max_value_mae,
        "outcome_distribution_mean_absolute_error_bound": uncertainty[
            "outcome_distribution_mean_absolute_error_bound"
        ],
        "outcome_expected_value_absolute_error_bound": uncertainty[
            "outcome_expected_value_absolute_error_bound"
        ],
        "value_absolute_error_bound": uncertainty["value_absolute_error_bound"],
        "minimum_terminal_leaf_support": policy.min_terminal_leaves,
        "observed_terminal_leaf_support": support["terminal_leaf_count"],
        "minimum_distinct_game_support": policy.min_distinct_game_ids,
        "observed_distinct_game_support": support["distinct_game_id_count"],
        "maximum_leaves_per_game": 1,
        "value_weight": policy.value_weight,
        "outcome_weight": policy.outcome_weight,
    }
    if reranker_bounds != expected_bounds:
        raise FrozenLeafCalibrationError("receipt reranker bounds do not bind calibration policy")

    authority = _mapping(payload.get("authority"), "receipt.authority")
    if authority.get("frozen_nonterminal_leaf_reranking_preflight_eligible") is not True:
        raise FrozenLeafCalibrationError("receipt does not authorize its narrow preflight use")
    for key in (
        "training_authorized",
        "gradient_or_optimizer_update_authorized",
        "serving_authorized",
        "action_authority_enabled",
        "selector_change_authorized",
        "promotion_authorized",
    ):
        if authority.get(key) is not False:
            raise FrozenLeafCalibrationError(f"receipt authority {key} must remain false")

    verified = {**payload, "receipt_sha256": digest}
    if heldout_evidence_path is not None:
        _, observed_training_manifest_identity = _read_sealed_json(
            r195_training_source_manifest_path,
            "receipt r195 training source manifest",
        )
        _, observed_heldout_identity = _read_sealed_json(
            heldout_evidence_path, "receipt heldout evidence"
        )
        _, observed_prediction_identity = _read_sealed_json(
            frozen_predictions_path, "receipt frozen predictions"
        )
        if observed_training_manifest_identity != training_manifest_identity:
            raise FrozenLeafCalibrationError("receipt r195 training manifest identity has drifted")
        if observed_heldout_identity != heldout_identity:
            raise FrozenLeafCalibrationError("receipt heldout evidence identity has drifted")
        if observed_prediction_identity != prediction_identity:
            raise FrozenLeafCalibrationError("receipt frozen prediction identity has drifted")
        recompiled = compile_r207_frozen_leaf_calibration_preflight(
            heldout_evidence_path,
            frozen_predictions_path,
            r195_training_source_manifest_path,
            policy=policy,
        )
        if recompiled != verified:
            raise FrozenLeafCalibrationError("receipt does not reproduce from immutable inputs")
    return verified


def leaf_score_calibration_from_r207_receipt(
    receipt: Mapping[str, Any],
    *,
    heldout_evidence_path: str | Path | None = None,
    frozen_predictions_path: str | Path | None = None,
    r195_training_source_manifest_path: str | Path | None = None,
) -> LeafScoreCalibration:
    """Build the only calibrated blend admissible from an r207 receipt.

    This is the narrow bridge to :class:`LeafScoreCalibration`; it does not
    start inference, enable action authority, or choose any action.  The
    returned object preserves the receipt's fixed policy-owned outcome/value
    weights and the exact frozen r195 policy identity.
    """

    verified = verify_r207_frozen_leaf_calibration_receipt(
        receipt,
        heldout_evidence_path=heldout_evidence_path,
        frozen_predictions_path=frozen_predictions_path,
        r195_training_source_manifest_path=r195_training_source_manifest_path,
    )
    binding = _mapping(verified.get("leaf_score_calibration"), "receipt.leaf_score_calibration")
    return LeafScoreCalibration(
        receipt_sha256=verified["receipt_sha256"],
        frozen_policy_identity_sha256=binding["frozen_policy_identity_sha256"],
        value_weight=binding["value_weight"],
        outcome_weight=binding["outcome_weight"],
        value_head_calibrated=binding["value_head_calibrated"],
        outcome_distribution_calibrated=binding["outcome_distribution_calibrated"],
        source_excluded=binding["source_excluded"],
        policy_visible_only=binding["policy_visible_only"],
        evaluation_only=binding["evaluation_only"],
        training_eligible=binding["training_eligible"],
        serving_action_authority=binding["serving_action_authority"],
        outcome_class_order=tuple(binding["outcome_class_order"]),
    )


__all__ = [
    "R195_CHECKPOINT_BYTES",
    "R195_CHECKPOINT_SHA256",
    "R207_CALIBRATION_RECEIPT_SCHEMA",
    "R207_FROZEN_PREDICTIONS_SCHEMA",
    "R207_HELDOUT_EVIDENCE_SCHEMA",
    "R207_TRAINING_SOURCE_MANIFEST_SCHEMA",
    "FrozenLeafCalibrationError",
    "FrozenLeafCalibrationPolicy",
    "canonical_sha256",
    "compile_r207_frozen_leaf_calibration_preflight",
    "leaf_score_calibration_from_r207_receipt",
    "verify_r207_frozen_leaf_calibration_receipt",
]
