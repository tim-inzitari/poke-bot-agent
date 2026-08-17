"""Sealed offline materialization for the r207 frozen-leaf calibration.

This is deliberately a *data-boundary* utility, not a game runner, model
loader, simulator, or training path.  It converts three already-sealed inputs
from the exact revision-195 NO-RTP lineage into the three immutable artifacts
consumed by :mod:`r207_frozen_leaf_calibration`:

* an r195 training-source manifest;
* source-excluded, exact-terminal continuation labels for policy-visible
  nonterminal leaves; and
* frozen r195 outcome/value predictions for precisely those leaves.

The command cannot invent a terminal result or a prediction: its captures must
already bind every leaf to a source-excluded r195 NO-RTP game, an exact
simulator terminal continuation, and a zero-update frozen inference result.
It never imports Torch, starts a service, opens a remote connection, or writes
a checkpoint.  It creates a new output directory only, seals every emitted
JSON file at mode ``0444``, and then lets the existing compiler decide whether
the evidence is actually calibrated enough to issue its narrow receipt.

Legacy RTP, r206 guide-logit packages, r212 Guide2Vec, and r205/r207 BO1000
rows are explicitly excluded.  The evaluator therefore cannot turn a nearby
but different Alakazam submission into r207 calibration evidence.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from poke_bot.recursive_turn_planner.neural_leaf_reranker import FrozenPolicyIdentity
from poke_bot.recursive_turn_planner.r207_frozen_leaf_calibration import (
    R195_CHECKPOINT_BYTES,
    R195_CHECKPOINT_SHA256,
    R207_FROZEN_PREDICTIONS_SCHEMA,
    R207_HELDOUT_EVIDENCE_SCHEMA,
    R207_TRAINING_SOURCE_MANIFEST_SCHEMA,
    FrozenLeafCalibrationError,
    FrozenLeafCalibrationPolicy,
    canonical_sha256,
    compile_r207_frozen_leaf_calibration_preflight,
    verify_r207_frozen_leaf_calibration_receipt,
)
from poke_bot.rtp_evaluation_immutable_io import (
    ImmutableEvidenceIOError,
    lexical_absolute_path,
    read_immutable_file_bytes,
)

R207_R195_GAME_INDEX_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_r195_no_rtp_game_index/v1"
)
R207_TERMINAL_CONTINUATION_CAPTURE_SCHEMA = "poke_bot.recursive_turn_planner.r207_source_excluded_terminal_continuation_capture/v1"
R207_FROZEN_INFERENCE_CAPTURE_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_r195_frozen_nonterminal_inference_capture/v1"
)
R207_MATERIALIZATION_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r207_frozen_leaf_calibration_materialization/v1"
)

R195_NO_RTP_SUBMISSION_ID = 55378392
R195_NO_RTP_MODEL_ROLE = "r195_no_rtp_frozen_policy_and_leaf_model"
R207_CALIBRATION_EVIDENCE_ROLE = "r207_source_excluded_terminal_leaf_calibration"

_OUTCOMES = ("loss", "draw", "win")
_LEAF_KINDS = {"successor", "boundary"}
_EXCLUSION_FIELDS = (
    "attempt10_rtp_partial_rows_used",
    "r197_or_r198_rtp_partial_rows_used",
    "r195_rtp_sidecar_used",
    "r197_rtp_sidecar_used",
    "r206_guide_logit_package_used",
    "r212_guide2vec_used",
    "r205_or_r207_bo1000_rows_used",
)


class R207FrozenLeafMaterializationError(ValueError):
    """The proposed offline input cannot truthfully become r207 evidence."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise R207FrozenLeafMaterializationError(f"{label} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise R207FrozenLeafMaterializationError(
            f"{label} must be a sha256 digest"
        ) from exc
    return value.lower()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise R207FrozenLeafMaterializationError(f"{label} must be a nonempty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R207FrozenLeafMaterializationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R207FrozenLeafMaterializationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise R207FrozenLeafMaterializationError(
            f"{label} must be finite and in [{minimum}, {maximum}]"
        )
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R207FrozenLeafMaterializationError(f"{label} must be an object")
    return dict(value)


def _read_sealed_json(
    path: str | Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        material = read_immutable_file_bytes(path, label, exact_mode=0o444)
    except ImmutableEvidenceIOError as exc:
        raise R207FrozenLeafMaterializationError(str(exc)) from exc
    try:
        payload = json.loads(material.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R207FrozenLeafMaterializationError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    return _mapping(payload, label), {
        "path": str(material.path),
        "sha256": material.sha256,
        "bytes": material.bytes,
        "mode": f"{material.mode:04o}",
    }


def _require_false_fields(payload: Mapping[str, Any], label: str) -> None:
    for field in _EXCLUSION_FIELDS:
        if payload.get(field) is not False:
            raise R207FrozenLeafMaterializationError(
                f"{label}.{field} must be exactly false"
            )


def _frozen_identity_payload() -> dict[str, Any]:
    identity = FrozenPolicyIdentity.r205_no_rtp()
    return {
        "checkpoint_sha256": identity.checkpoint_sha256,
        "checkpoint_bytes": identity.checkpoint_bytes,
        "bundle_sha256": identity.bundle_sha256,
        "deck_cards_sha256": identity.deck_cards_sha256,
        "nonplanner_runtime_sha256": identity.nonplanner_runtime_sha256,
        "identity_sha256": identity.identity_sha256,
    }


def _require_r195_identity(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    identity = _frozen_identity_payload()
    if (
        _integer(payload.get("submission_id"), f"{label}.submission_id", minimum=1)
        != R195_NO_RTP_SUBMISSION_ID
    ):
        raise R207FrozenLeafMaterializationError(f"{label} is not submission 55378392")
    if (
        _sha256(payload.get("checkpoint_sha256"), f"{label}.checkpoint_sha256")
        != R195_CHECKPOINT_SHA256
    ):
        raise R207FrozenLeafMaterializationError(
            f"{label} is not exact r195 NO-RTP checkpoint"
        )
    if (
        _integer(
            payload.get("checkpoint_bytes"), f"{label}.checkpoint_bytes", minimum=1
        )
        != R195_CHECKPOINT_BYTES
    ):
        raise R207FrozenLeafMaterializationError(
            f"{label} checkpoint bytes are not exact r195"
        )
    for field in ("bundle_sha256", "deck_cards_sha256", "nonplanner_runtime_sha256"):
        if _sha256(payload.get(field), f"{label}.{field}") != identity[field]:
            raise R207FrozenLeafMaterializationError(
                f"{label}.{field} does not bind exact r195 NO-RTP identity"
            )
    if (
        _sha256(
            payload.get("frozen_policy_identity_sha256"),
            f"{label}.frozen_policy_identity_sha256",
        )
        != identity["identity_sha256"]
    ):
        raise R207FrozenLeafMaterializationError(
            f"{label} does not bind exact r195 frozen policy identity"
        )
    return identity


def _policy(payload: Any) -> FrozenLeafCalibrationPolicy:
    source = _mapping(payload, "source_index.calibration_policy")
    allowed = {
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
    }
    if set(source) != allowed:
        raise R207FrozenLeafMaterializationError(
            "source_index.calibration_policy has unexpected or missing fields"
        )
    try:
        return FrozenLeafCalibrationPolicy(**source)
    except FrozenLeafCalibrationError as exc:
        raise R207FrozenLeafMaterializationError(str(exc)) from exc


def _sorted_source_records(
    value: Any,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not isinstance(value, list) or not value:
        raise R207FrozenLeafMaterializationError(
            "source_index.games must be a nonempty array"
        )
    training: list[dict[str, str]] = []
    heldout: list[dict[str, str]] = []
    seen_games: set[str] = set()
    for index, raw in enumerate(value):
        row = _mapping(raw, f"source_index.games[{index}]")
        if set(row) != {"partition", "source_id", "game_id", "game_sha256"}:
            raise R207FrozenLeafMaterializationError(
                f"source_index.games[{index}] has unexpected or missing fields"
            )
        partition = _text(
            row.get("partition"), f"source_index.games[{index}].partition"
        )
        if partition not in {"training", "heldout"}:
            raise R207FrozenLeafMaterializationError(
                "source_index game partition must be training or heldout"
            )
        game_id = _text(row.get("game_id"), f"source_index.games[{index}].game_id")
        if game_id in seen_games:
            raise R207FrozenLeafMaterializationError(
                "source_index game IDs must be globally unique"
            )
        seen_games.add(game_id)
        parsed = {
            "source_id": _text(
                row.get("source_id"), f"source_index.games[{index}].source_id"
            ),
            "game_id": game_id,
            "game_sha256": _sha256(
                row.get("game_sha256"), f"source_index.games[{index}].game_sha256"
            ),
        }
        (training if partition == "training" else heldout).append(parsed)
    if not training or not heldout:
        raise R207FrozenLeafMaterializationError(
            "source_index must contain both training and heldout games"
        )
    training_sources = {row["source_id"] for row in training}
    heldout_sources = {row["source_id"] for row in heldout}
    if training_sources.intersection(heldout_sources):
        raise R207FrozenLeafMaterializationError(
            "training and heldout source IDs must be source-disjoint"
        )
    return (
        sorted(training, key=lambda row: (row["source_id"], row["game_id"])),
        sorted(heldout, key=lambda row: (row["source_id"], row["game_id"])),
    )


def _parse_source_index(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], FrozenLeafCalibrationPolicy]:
    if (
        payload.get("schema") != R207_R195_GAME_INDEX_SCHEMA
        or payload.get("status") != "sealed"
    ):
        raise R207FrozenLeafMaterializationError(
            "source index schema or status is invalid"
        )
    if payload.get("source_role") != "r195_no_rtp_archived_games":
        raise R207FrozenLeafMaterializationError(
            "source index must be r195 NO-RTP archived games"
        )
    if payload.get("training_eligible") is not True:
        raise R207FrozenLeafMaterializationError(
            "source index must identify r195 training sources"
        )
    _require_false_fields(payload, "source_index")
    _require_r195_identity(payload, "source_index")
    training, heldout = _sorted_source_records(payload.get("games"))
    canonical_games = [{"partition": "training", **row} for row in training] + [
        {"partition": "heldout", **row} for row in heldout
    ]
    if _sha256(
        payload.get("games_sha256"), "source_index.games_sha256"
    ) != canonical_sha256(canonical_games):
        raise R207FrozenLeafMaterializationError(
            "source_index.games_sha256 does not bind the declared games"
        )
    return training, heldout, _policy(payload.get("calibration_policy"))


def _capture_records(
    payload: Mapping[str, Any],
    *,
    source_index_identity: Mapping[str, Any],
    heldout_games: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    if (
        payload.get("schema") != R207_TERMINAL_CONTINUATION_CAPTURE_SCHEMA
        or payload.get("status") != "sealed"
    ):
        raise R207FrozenLeafMaterializationError(
            "terminal capture schema or status is invalid"
        )
    if (
        payload.get("capture_role")
        != "r207_source_excluded_exact_terminal_continuations"
    ):
        raise R207FrozenLeafMaterializationError("terminal capture role is invalid")
    if (
        payload.get("evaluation_only") is not True
        or payload.get("training_eligible") is not False
    ):
        raise R207FrozenLeafMaterializationError(
            "terminal capture must be evaluation-only"
        )
    if payload.get("replay_eligible") is not False:
        raise R207FrozenLeafMaterializationError(
            "terminal capture must be replay-ineligible"
        )
    if (
        payload.get("simulator_backed") is not True
        or payload.get("terminal_outcomes_exact") is not True
    ):
        raise R207FrozenLeafMaterializationError(
            "terminal capture must prove simulator-backed exact terminal outcomes"
        )
    _require_false_fields(payload, "terminal_capture")
    _require_r195_identity(payload, "terminal_capture")
    if (
        _sha256(
            payload.get("source_index_sha256"), "terminal_capture.source_index_sha256"
        )
        != source_index_identity["sha256"]
    ):
        raise R207FrozenLeafMaterializationError(
            "terminal capture does not bind source index bytes"
        )
    expected_games = {(row["source_id"], row["game_id"]) for row in heldout_games}
    rows = payload.get("continuations")
    if not isinstance(rows, list) or not rows:
        raise R207FrozenLeafMaterializationError(
            "terminal_capture.continuations must be nonempty"
        )
    parsed: list[dict[str, Any]] = []
    observed_games: set[tuple[str, str]] = set()
    expected_fields = {
        "source_id",
        "game_id",
        "decision_ordinal",
        "root_seat",
        "horizon_atomic_actions",
        "leaf_kind",
        "policy_visible",
        "is_terminal",
        "public_observation_sha256",
        "legal_actions_sha256",
        "nonterminal_leaf_state_sha256",
        "terminal_outcome",
        "terminal_result_kind",
        "terminal_result_sha256",
    }
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"terminal_capture.continuations[{index}]")
        if set(row) != expected_fields:
            raise R207FrozenLeafMaterializationError(
                f"terminal_capture.continuations[{index}] has unexpected or missing fields"
            )
        source_id = _text(
            row.get("source_id"), f"terminal_capture.continuations[{index}].source_id"
        )
        game_id = _text(
            row.get("game_id"), f"terminal_capture.continuations[{index}].game_id"
        )
        if (source_id, game_id) not in expected_games:
            raise R207FrozenLeafMaterializationError(
                "terminal continuation is not a declared source-excluded heldout game"
            )
        if (source_id, game_id) in observed_games:
            raise R207FrozenLeafMaterializationError(
                "each heldout game may provide exactly one terminal continuation"
            )
        observed_games.add((source_id, game_id))
        root_seat = _integer(
            row.get("root_seat"), f"terminal_capture.continuations[{index}].root_seat"
        )
        if root_seat not in {0, 1}:
            raise R207FrozenLeafMaterializationError(
                "terminal continuation root_seat must be 0 or 1"
            )
        leaf_kind = _text(
            row.get("leaf_kind"), f"terminal_capture.continuations[{index}].leaf_kind"
        )
        if leaf_kind not in _LEAF_KINDS:
            raise R207FrozenLeafMaterializationError(
                "terminal continuation leaf_kind must be successor or boundary"
            )
        if row.get("policy_visible") is not True or row.get("is_terminal") is not False:
            raise R207FrozenLeafMaterializationError(
                "terminal continuation must originate at a policy-visible nonterminal leaf"
            )
        outcome = _text(
            row.get("terminal_outcome"),
            f"terminal_capture.continuations[{index}].terminal_outcome",
        )
        if outcome not in _OUTCOMES:
            raise R207FrozenLeafMaterializationError(
                "terminal continuation outcome is invalid"
            )
        if row.get("terminal_result_kind") != "exact_simulator_terminal":
            raise R207FrozenLeafMaterializationError(
                "terminal continuation must use exact_simulator_terminal"
            )
        parsed.append(
            {
                "source_id": source_id,
                "game_id": game_id,
                "decision_ordinal": _integer(
                    row.get("decision_ordinal"),
                    f"terminal_capture.continuations[{index}].decision_ordinal",
                ),
                "root_seat": root_seat,
                "horizon_atomic_actions": _integer(
                    row.get("horizon_atomic_actions"),
                    f"terminal_capture.continuations[{index}].horizon_atomic_actions",
                    minimum=1,
                ),
                "leaf_kind": leaf_kind,
                "public_observation_sha256": _sha256(
                    row.get("public_observation_sha256"),
                    f"terminal_capture.continuations[{index}].public_observation_sha256",
                ),
                "legal_actions_sha256": _sha256(
                    row.get("legal_actions_sha256"),
                    f"terminal_capture.continuations[{index}].legal_actions_sha256",
                ),
                "nonterminal_leaf_state_sha256": _sha256(
                    row.get("nonterminal_leaf_state_sha256"),
                    f"terminal_capture.continuations[{index}].nonterminal_leaf_state_sha256",
                ),
                "terminal_outcome": outcome,
                "terminal_result_kind": "exact_simulator_terminal",
                "terminal_result_sha256": _sha256(
                    row.get("terminal_result_sha256"),
                    f"terminal_capture.continuations[{index}].terminal_result_sha256",
                ),
            }
        )
    if observed_games != expected_games:
        raise R207FrozenLeafMaterializationError(
            "terminal capture must cover exactly every declared heldout game"
        )
    return sorted(parsed, key=lambda row: (row["source_id"], row["game_id"]))


def _outcome_probabilities(value: Any, label: str) -> dict[str, float]:
    payload = _mapping(value, label)
    if set(payload) != set(_OUTCOMES):
        raise R207FrozenLeafMaterializationError(
            f"{label} must contain exactly loss, draw, and win"
        )
    result = {
        outcome: _number(
            payload[outcome], f"{label}.{outcome}", minimum=0.0, maximum=1.0
        )
        for outcome in _OUTCOMES
    }
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise R207FrozenLeafMaterializationError(f"{label} must sum exactly to one")
    return result


def _inference_records(
    payload: Mapping[str, Any],
    *,
    source_index_identity: Mapping[str, Any],
    capture_identity: Mapping[str, Any],
    continuations: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if (
        payload.get("schema") != R207_FROZEN_INFERENCE_CAPTURE_SCHEMA
        or payload.get("status") != "sealed"
    ):
        raise R207FrozenLeafMaterializationError(
            "frozen inference capture schema or status is invalid"
        )
    if payload.get("capture_role") != "r207_r195_no_rtp_frozen_nonterminal_inference":
        raise R207FrozenLeafMaterializationError(
            "frozen inference capture role is invalid"
        )
    if payload.get("model_role") != R195_NO_RTP_MODEL_ROLE:
        raise R207FrozenLeafMaterializationError(
            "frozen inference capture model role is invalid"
        )
    _require_false_fields(payload, "frozen_inference_capture")
    _require_r195_identity(payload, "frozen_inference_capture")
    if (
        _sha256(
            payload.get("source_index_sha256"),
            "frozen_inference_capture.source_index_sha256",
        )
        != source_index_identity["sha256"]
    ):
        raise R207FrozenLeafMaterializationError(
            "frozen inference capture does not bind source index"
        )
    if (
        _sha256(
            payload.get("terminal_capture_sha256"),
            "frozen_inference_capture.terminal_capture_sha256",
        )
        != capture_identity["sha256"]
    ):
        raise R207FrozenLeafMaterializationError(
            "frozen inference capture does not bind terminal capture"
        )
    for field, expected in (
        ("model_frozen", True),
        ("training_performed", False),
        ("training_eligible", False),
    ):
        if payload.get(field) is not expected:
            raise R207FrozenLeafMaterializationError(
                f"frozen inference capture {field} is invalid"
            )
    for field in ("optimizer_steps", "gradient_updates"):
        if _integer(payload.get(field), f"frozen_inference_capture.{field}") != 0:
            raise R207FrozenLeafMaterializationError(
                "frozen inference capture permits updates"
            )
    expected = {(row["source_id"], row["game_id"]): row for row in continuations}
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise R207FrozenLeafMaterializationError(
            "frozen_inference_capture.predictions must be nonempty"
        )
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    expected_fields = {
        "source_id",
        "game_id",
        "decision_ordinal",
        "root_seat",
        "horizon_atomic_actions",
        "leaf_kind",
        "public_observation_sha256",
        "legal_actions_sha256",
        "nonterminal_leaf_state_sha256",
        "terminal_result_sha256",
        "outcome_probabilities",
        "value",
    }
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"frozen_inference_capture.predictions[{index}]")
        if set(row) != expected_fields:
            raise R207FrozenLeafMaterializationError(
                f"frozen_inference_capture.predictions[{index}] has unexpected or missing fields"
            )
        key = (
            _text(
                row.get("source_id"),
                f"frozen_inference_capture.predictions[{index}].source_id",
            ),
            _text(
                row.get("game_id"),
                f"frozen_inference_capture.predictions[{index}].game_id",
            ),
        )
        if key in parsed or key not in expected:
            raise R207FrozenLeafMaterializationError(
                "frozen prediction must bind exactly one captured heldout game"
            )
        continuation = expected[key]
        bound = {
            "decision_ordinal": _integer(
                row.get("decision_ordinal"),
                f"frozen_inference_capture.predictions[{index}].decision_ordinal",
            ),
            "root_seat": _integer(
                row.get("root_seat"),
                f"frozen_inference_capture.predictions[{index}].root_seat",
            ),
            "horizon_atomic_actions": _integer(
                row.get("horizon_atomic_actions"),
                f"frozen_inference_capture.predictions[{index}].horizon_atomic_actions",
                minimum=1,
            ),
            "leaf_kind": _text(
                row.get("leaf_kind"),
                f"frozen_inference_capture.predictions[{index}].leaf_kind",
            ),
            "public_observation_sha256": _sha256(
                row.get("public_observation_sha256"),
                f"frozen_inference_capture.predictions[{index}].public_observation_sha256",
            ),
            "legal_actions_sha256": _sha256(
                row.get("legal_actions_sha256"),
                f"frozen_inference_capture.predictions[{index}].legal_actions_sha256",
            ),
            "nonterminal_leaf_state_sha256": _sha256(
                row.get("nonterminal_leaf_state_sha256"),
                f"frozen_inference_capture.predictions[{index}].nonterminal_leaf_state_sha256",
            ),
            "terminal_result_sha256": _sha256(
                row.get("terminal_result_sha256"),
                f"frozen_inference_capture.predictions[{index}].terminal_result_sha256",
            ),
        }
        for field, value in bound.items():
            if value != continuation[field]:
                raise R207FrozenLeafMaterializationError(
                    f"frozen prediction {field} does not match terminal continuation"
                )
        if bound["root_seat"] not in {0, 1} or bound["leaf_kind"] not in _LEAF_KINDS:
            raise R207FrozenLeafMaterializationError(
                "frozen prediction leaf metadata is invalid"
            )
        parsed[key] = {
            **bound,
            "outcome_probabilities": _outcome_probabilities(
                row.get("outcome_probabilities"),
                f"frozen_inference_capture.predictions[{index}].outcome_probabilities",
            ),
            "value": _number(
                row.get("value"),
                f"frozen_inference_capture.predictions[{index}].value",
                minimum=-1.0,
                maximum=1.0,
            ),
        }
    if set(parsed) != set(expected):
        raise R207FrozenLeafMaterializationError(
            "frozen inference capture must cover exactly every terminal continuation"
        )
    return parsed


def _leaf_id(source_index_sha256: str, continuation: Mapping[str, Any]) -> str:
    digest = canonical_sha256(
        {
            "schema": "poke_bot.r207_calibration_leaf_id/v1",
            "source_index_sha256": source_index_sha256,
            "source_id": continuation["source_id"],
            "game_id": continuation["game_id"],
            "decision_ordinal": continuation["decision_ordinal"],
            "root_seat": continuation["root_seat"],
            "terminal_result_sha256": continuation["terminal_result_sha256"],
        }
    )
    return f"r207-leaf-{digest[7:31]}"


def _write_new_sealed_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _canonical_json(dict(payload))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise R207FrozenLeafMaterializationError(
            f"cannot create new output {path}"
        ) from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise R207FrozenLeafMaterializationError(f"cannot write output {path}")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    try:
        material = read_immutable_file_bytes(
            path, f"materialized output {path.name}", exact_mode=0o444
        )
    except ImmutableEvidenceIOError as exc:
        raise R207FrozenLeafMaterializationError(str(exc)) from exc
    return {
        "path": str(material.path),
        "sha256": material.sha256,
        "bytes": material.bytes,
        "mode": f"{material.mode:04o}",
    }


def _new_output_directory(value: str | Path) -> Path:
    try:
        path = lexical_absolute_path(value, "output_dir")
    except ImmutableEvidenceIOError as exc:
        raise R207FrozenLeafMaterializationError(str(exc)) from exc
    if path.exists():
        raise R207FrozenLeafMaterializationError("output_dir must not already exist")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise R207FrozenLeafMaterializationError(
            "output_dir parent must be an existing non-symlink directory"
        )
    try:
        os.mkdir(path, 0o755)
    except OSError as exc:
        raise R207FrozenLeafMaterializationError("cannot create output_dir") from exc
    return path


def materialize_r207_frozen_leaf_calibration(
    *,
    source_index_path: str | Path,
    terminal_capture_path: str | Path,
    frozen_inference_capture_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a new, compiled r207 calibration evidence bundle.

    All three inputs must already be sealed physical ``0444`` files.  The
    function intentionally has no mode that overwrites an existing output or
    falls back to an unbound model/simulator result.
    """

    source_index, source_index_identity = _read_sealed_json(
        source_index_path, "r195 game index"
    )
    training_games, heldout_games, policy = _parse_source_index(source_index)
    capture, capture_identity = _read_sealed_json(
        terminal_capture_path, "terminal continuation capture"
    )
    continuations = _capture_records(
        capture,
        source_index_identity=source_index_identity,
        heldout_games=heldout_games,
    )
    inference_capture, inference_capture_identity = _read_sealed_json(
        frozen_inference_capture_path, "frozen inference capture"
    )
    predictions_by_game = _inference_records(
        inference_capture,
        source_index_identity=source_index_identity,
        capture_identity=capture_identity,
        continuations=continuations,
    )

    output = _new_output_directory(output_dir)
    training_sources = sorted({row["source_id"] for row in training_games})
    training_game_ids = sorted(row["game_id"] for row in training_games)
    heldout_sources = sorted({row["source_id"] for row in heldout_games})
    heldout_game_ids = sorted(row["game_id"] for row in heldout_games)
    common = {
        "submission_id": R195_NO_RTP_SUBMISSION_ID,
        **_frozen_identity_payload(),
        **{field: False for field in _EXCLUSION_FIELDS},
    }
    training_manifest = {
        "schema": R207_TRAINING_SOURCE_MANIFEST_SCHEMA,
        "status": "sealed",
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "source_role": "r195_training",
        "training_eligible": True,
        "training_source_ids": training_sources,
        "training_source_ids_sha256": canonical_sha256(training_sources),
        "training_game_ids": training_game_ids,
        "training_game_ids_sha256": canonical_sha256(training_game_ids),
        **{field: False for field in _EXCLUSION_FIELDS},
    }
    training_manifest_identity = _write_new_sealed_json(
        output / "r195-training-source-manifest.json", training_manifest
    )
    source_exclusion = {
        "source_disjoint": True,
        "game_disjoint": True,
        "r195_training_source_manifest_sha256": training_manifest_identity["sha256"],
        "training_source_ids_sha256": canonical_sha256(training_sources),
        "training_game_ids_sha256": canonical_sha256(training_game_ids),
        "heldout_source_ids": heldout_sources,
        "heldout_source_ids_sha256": canonical_sha256(heldout_sources),
        "heldout_game_ids": heldout_game_ids,
        "heldout_game_ids_sha256": canonical_sha256(heldout_game_ids),
        "intersection_source_ids_sha256": canonical_sha256([]),
        "intersection_source_id_count": 0,
        "intersection_game_ids_sha256": canonical_sha256([]),
        "intersection_game_id_count": 0,
    }
    evidence_leaves: list[dict[str, Any]] = []
    for continuation in continuations:
        evidence_leaves.append(
            {
                "leaf_id": _leaf_id(source_index_identity["sha256"], continuation),
                **continuation,
                "policy_visible": True,
                "is_terminal": False,
            }
        )
    evidence = {
        "schema": R207_HELDOUT_EVIDENCE_SCHEMA,
        "status": "sealed",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "evidence_role": R207_CALIBRATION_EVIDENCE_ROLE,
        "outcome_perspective": "candidate_root",
        "terminal_outcomes_exact": True,
        "nonterminal_policy_visible_leaf_predictions_only": True,
        "source_exclusion": source_exclusion,
        "terminal_leaf_count": len(evidence_leaves),
        "terminal_leaves": evidence_leaves,
        **{field: False for field in _EXCLUSION_FIELDS},
    }
    evidence_identity = _write_new_sealed_json(
        output / "r207-heldout-terminal-leaf-evidence.json", evidence
    )
    prediction_rows: list[dict[str, Any]] = []
    for leaf in evidence_leaves:
        inferred = predictions_by_game[(leaf["source_id"], leaf["game_id"])]
        prediction_rows.append(
            {
                "leaf_id": leaf["leaf_id"],
                "source_id": leaf["source_id"],
                "game_id": leaf["game_id"],
                "decision_ordinal": leaf["decision_ordinal"],
                "root_seat": leaf["root_seat"],
                "horizon_atomic_actions": leaf["horizon_atomic_actions"],
                "leaf_kind": leaf["leaf_kind"],
                "public_observation_sha256": leaf["public_observation_sha256"],
                "legal_actions_sha256": leaf["legal_actions_sha256"],
                "nonterminal_leaf_state_sha256": leaf["nonterminal_leaf_state_sha256"],
                "terminal_result_sha256": leaf["terminal_result_sha256"],
                "outcome_probabilities": inferred["outcome_probabilities"],
                "value": inferred["value"],
            }
        )
    predictions = {
        "schema": R207_FROZEN_PREDICTIONS_SCHEMA,
        "status": "frozen_inference_complete",
        "outcome_perspective": "candidate_root",
        "outcome_class_order": list(_OUTCOMES),
        "model_role": R195_NO_RTP_MODEL_ROLE,
        "frozen_policy_identity_sha256": common["identity_sha256"],
        "r195_no_rtp_bundle_sha256": common["bundle_sha256"],
        "r195_no_rtp_deck_cards_sha256": common["deck_cards_sha256"],
        "r195_no_rtp_runtime_sha256": common["nonplanner_runtime_sha256"],
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "heldout_evidence_sha256": evidence_identity["sha256"],
        "model_frozen": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gradient_updates": 0,
        "training_eligible": False,
        "predictions": prediction_rows,
        **{field: False for field in _EXCLUSION_FIELDS},
    }
    predictions_identity = _write_new_sealed_json(
        output / "r207-frozen-predictions.json", predictions
    )
    receipt = compile_r207_frozen_leaf_calibration_preflight(
        output / "r207-heldout-terminal-leaf-evidence.json",
        output / "r207-frozen-predictions.json",
        output / "r195-training-source-manifest.json",
        policy=policy,
    )
    verify_r207_frozen_leaf_calibration_receipt(
        receipt,
        heldout_evidence_path=output / "r207-heldout-terminal-leaf-evidence.json",
        frozen_predictions_path=output / "r207-frozen-predictions.json",
        r195_training_source_manifest_path=output
        / "r195-training-source-manifest.json",
    )
    calibration_receipt_identity = _write_new_sealed_json(
        output / "r207-frozen-leaf-calibration-receipt.json", receipt
    )
    materialization = {
        "schema": R207_MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "materialized_and_compiler_passed",
        "source_inputs": {
            "r195_game_index": source_index_identity,
            "terminal_continuation_capture": capture_identity,
            "frozen_inference_capture": inference_capture_identity,
        },
        "outputs": {
            "r195_training_source_manifest": training_manifest_identity,
            "heldout_terminal_leaf_evidence": evidence_identity,
            "frozen_predictions": predictions_identity,
            "calibration_receipt": calibration_receipt_identity,
        },
        "frozen_policy_identity": _frozen_identity_payload(),
        "policy": {**policy.as_payload(), "sha256": policy.sha256},
        "offline_only": True,
        "external_workloads_started": False,
        "model_inference_executed_by_materializer": False,
        "simulator_execution_performed_by_materializer": False,
        "training_or_gradient_update_performed": False,
        "legacy_rtp_used": False,
        "guide_logit_package_used": False,
        "guide2vec_used": False,
        "bo1000_rows_used": False,
        "serving_or_action_authority_enabled": False,
    }
    materialization["materialization_receipt_sha256"] = canonical_sha256(
        materialization
    )
    materialization_identity = _write_new_sealed_json(
        output / "r207-frozen-leaf-materialization-receipt.json", materialization
    )
    return {
        "output_dir": str(output),
        "r195_training_source_manifest": training_manifest_identity,
        "heldout_terminal_leaf_evidence": evidence_identity,
        "frozen_predictions": predictions_identity,
        "calibration_receipt": calibration_receipt_identity,
        "materialization_receipt": materialization_identity,
        "calibration_receipt_sha256": receipt["receipt_sha256"],
    }


__all__ = [
    "R207_FROZEN_INFERENCE_CAPTURE_SCHEMA",
    "R207_MATERIALIZATION_RECEIPT_SCHEMA",
    "R207_R195_GAME_INDEX_SCHEMA",
    "R207_TERMINAL_CONTINUATION_CAPTURE_SCHEMA",
    "R207FrozenLeafMaterializationError",
    "materialize_r207_frozen_leaf_calibration",
]
