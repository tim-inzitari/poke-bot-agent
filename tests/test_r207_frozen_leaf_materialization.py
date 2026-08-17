from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from poke_bot.recursive_turn_planner.neural_leaf_reranker import FrozenPolicyIdentity
from poke_bot.recursive_turn_planner.r207_frozen_leaf_calibration import (
    R195_CHECKPOINT_BYTES,
    R195_CHECKPOINT_SHA256,
    verify_r207_frozen_leaf_calibration_receipt,
)
from poke_bot.recursive_turn_planner.r207_frozen_leaf_materialization import (
    R207_FROZEN_INFERENCE_CAPTURE_SCHEMA,
    R207_R195_GAME_INDEX_SCHEMA,
    R207_TERMINAL_CONTINUATION_CAPTURE_SCHEMA,
    R207FrozenLeafMaterializationError,
    materialize_r207_frozen_leaf_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
_EXCLUSION_FIELDS = (
    "attempt10_rtp_partial_rows_used",
    "r197_or_r198_rtp_partial_rows_used",
    "r195_rtp_sidecar_used",
    "r197_rtp_sidecar_used",
    "r206_guide_logit_package_used",
    "r212_guide2vec_used",
    "r205_or_r207_bo1000_rows_used",
)


def _sha256(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    os.chmod(path, 0o444)


def _identity_fields() -> dict[str, Any]:
    identity = FrozenPolicyIdentity.r205_no_rtp()
    return {
        "submission_id": 55378392,
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "bundle_sha256": identity.bundle_sha256,
        "deck_cards_sha256": identity.deck_cards_sha256,
        "nonplanner_runtime_sha256": identity.nonplanner_runtime_sha256,
        "frozen_policy_identity_sha256": identity.identity_sha256,
        **{field: False for field in _EXCLUSION_FIELDS},
    }


def _policy() -> dict[str, Any]:
    return {
        "min_terminal_leaves": 6,
        "min_distinct_source_ids": 3,
        "min_distinct_game_ids": 6,
        "min_per_outcome": 2,
        "min_per_source_id": 2,
        "max_outcome_brier": 0.01,
        "max_outcome_ece": 0.08,
        "max_value_rmse": 0.15,
        "max_value_mae": 0.15,
        "value_weight": 0.4,
        "outcome_weight": 0.6,
        "uncertainty_quantile": 0.95,
        "ece_bins": 3,
    }


def _source_index() -> dict[str, Any]:
    training = [
        {
            "partition": "training",
            "source_id": "r195-train-day-a",
            "game_id": "r195-train-game-a",
            "game_sha256": _sha256("r195-train-game-a"),
        },
        {
            "partition": "training",
            "source_id": "r195-train-day-b",
            "game_id": "r195-train-game-b",
            "game_sha256": _sha256("r195-train-game-b"),
        },
    ]
    heldout = [
        {
            "partition": "heldout",
            "source_id": f"r195-heldout-day-{source}",
            "game_id": f"r195-heldout-game-{index:02d}",
            "game_sha256": _sha256(f"r195-heldout-game-{index:02d}"),
        }
        for index, source in (
            (1, "a"),
            (2, "a"),
            (3, "b"),
            (4, "b"),
            (5, "c"),
            (6, "c"),
        )
    ]
    games = training + heldout
    return {
        "schema": R207_R195_GAME_INDEX_SCHEMA,
        "status": "sealed",
        "source_role": "r195_no_rtp_archived_games",
        "training_eligible": True,
        **_identity_fields(),
        "games": games,
        "games_sha256": "",  # populated after deterministic sort below
        "calibration_policy": _policy(),
    }


def _canonical_games(source_index: dict[str, Any]) -> list[dict[str, str]]:
    rows = source_index["games"]
    training = sorted(
        (
            {
                "source_id": row["source_id"],
                "game_id": row["game_id"],
                "game_sha256": row["game_sha256"],
            }
            for row in rows
            if row["partition"] == "training"
        ),
        key=lambda row: (row["source_id"], row["game_id"]),
    )
    heldout = sorted(
        (
            {
                "source_id": row["source_id"],
                "game_id": row["game_id"],
                "game_sha256": row["game_sha256"],
            }
            for row in rows
            if row["partition"] == "heldout"
        ),
        key=lambda row: (row["source_id"], row["game_id"]),
    )
    return [{"partition": "training", **row} for row in training] + [
        {"partition": "heldout", **row} for row in heldout
    ]


def _terminal_capture(
    source_index_sha256: str, index: dict[str, Any]
) -> dict[str, Any]:
    outcomes = ("loss", "win", "draw", "loss", "win", "draw")
    rows = [row for row in index["games"] if row["partition"] == "heldout"]
    continuations = []
    for ordinal, (game, outcome) in enumerate(
        zip(rows, outcomes, strict=True), start=1
    ):
        continuations.append(
            {
                "source_id": game["source_id"],
                "game_id": game["game_id"],
                "decision_ordinal": ordinal,
                "root_seat": ordinal % 2,
                "horizon_atomic_actions": 2,
                "leaf_kind": "successor" if ordinal % 2 else "boundary",
                "policy_visible": True,
                "is_terminal": False,
                "public_observation_sha256": _sha256(f"observation-{ordinal}"),
                "legal_actions_sha256": _sha256(f"legal-actions-{ordinal}"),
                "nonterminal_leaf_state_sha256": _sha256(f"leaf-state-{ordinal}"),
                "terminal_outcome": outcome,
                "terminal_result_kind": "exact_simulator_terminal",
                "terminal_result_sha256": _sha256(f"terminal-{ordinal}"),
            }
        )
    return {
        "schema": R207_TERMINAL_CONTINUATION_CAPTURE_SCHEMA,
        "status": "sealed",
        "capture_role": "r207_source_excluded_exact_terminal_continuations",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "simulator_backed": True,
        "terminal_outcomes_exact": True,
        **_identity_fields(),
        "source_index_sha256": source_index_sha256,
        "continuations": continuations,
    }


def _frozen_inference_capture(
    source_index_sha256: str,
    terminal_capture_sha256: str,
    capture: dict[str, Any],
) -> dict[str, Any]:
    scores = {
        "loss": ({"loss": 0.96, "draw": 0.03, "win": 0.01}, -0.92),
        "win": ({"loss": 0.01, "draw": 0.03, "win": 0.96}, 0.92),
        "draw": ({"loss": 0.03, "draw": 0.94, "win": 0.03}, 0.03),
    }
    predictions = []
    for continuation in capture["continuations"]:
        probabilities, value = scores[continuation["terminal_outcome"]]
        predictions.append(
            {
                key: continuation[key]
                for key in (
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
                )
            }
            | {"outcome_probabilities": probabilities, "value": value}
        )
    return {
        "schema": R207_FROZEN_INFERENCE_CAPTURE_SCHEMA,
        "status": "sealed",
        "capture_role": "r207_r195_no_rtp_frozen_nonterminal_inference",
        "model_role": "r195_no_rtp_frozen_policy_and_leaf_model",
        **_identity_fields(),
        "source_index_sha256": source_index_sha256,
        "terminal_capture_sha256": terminal_capture_sha256,
        "model_frozen": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gradient_updates": 0,
        "training_eligible": False,
        "predictions": predictions,
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    index_path = tmp_path / "r195-index.json"
    index = _source_index()
    index["games_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                _canonical_games(index),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    _write_sealed_json(index_path, index)

    capture_path = tmp_path / "terminal-capture.json"
    capture = _terminal_capture(_file_sha256(index_path), index)
    _write_sealed_json(capture_path, capture)

    inference_path = tmp_path / "frozen-inference.json"
    inference = _frozen_inference_capture(
        _file_sha256(index_path), _file_sha256(capture_path), capture
    )
    _write_sealed_json(inference_path, inference)
    return index_path, capture_path, inference_path


def test_materializer_builds_and_reverifies_real_shape_from_sealed_r195_captures(
    tmp_path: Path,
) -> None:
    index_path, capture_path, inference_path = _inputs(tmp_path)
    result = materialize_r207_frozen_leaf_calibration(
        source_index_path=index_path,
        terminal_capture_path=capture_path,
        frozen_inference_capture_path=inference_path,
        output_dir=tmp_path / "r207-calibration-output",
    )

    output = Path(result["output_dir"])
    assert result["calibration_receipt_sha256"].startswith("sha256:")
    for name in (
        "r195-training-source-manifest.json",
        "r207-heldout-terminal-leaf-evidence.json",
        "r207-frozen-predictions.json",
        "r207-frozen-leaf-calibration-receipt.json",
        "r207-frozen-leaf-materialization-receipt.json",
    ):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o444

    receipt = json.loads(
        (output / "r207-frozen-leaf-calibration-receipt.json").read_text()
    )
    verified = verify_r207_frozen_leaf_calibration_receipt(
        receipt,
        heldout_evidence_path=output / "r207-heldout-terminal-leaf-evidence.json",
        frozen_predictions_path=output / "r207-frozen-predictions.json",
        r195_training_source_manifest_path=output
        / "r195-training-source-manifest.json",
    )
    assert verified == receipt
    assert receipt["support"]["terminal_leaf_count"] == 6
    assert receipt["authority"]["training_authorized"] is False
    assert receipt["frozen_inference"]["r195_rtp_sidecar_used"] is False

    materialization = json.loads(
        (output / "r207-frozen-leaf-materialization-receipt.json").read_text()
    )
    assert materialization["offline_only"] is True
    assert materialization["external_workloads_started"] is False
    assert materialization["model_inference_executed_by_materializer"] is False
    assert materialization["guide2vec_used"] is False
    assert materialization["bo1000_rows_used"] is False


def test_materializer_rejects_guide_or_bo1000_contamination_before_output_creation(
    tmp_path: Path,
) -> None:
    index_path, capture_path, inference_path = _inputs(tmp_path)
    index = json.loads(index_path.read_text())
    os.chmod(index_path, 0o644)
    index["r212_guide2vec_used"] = True
    _write_sealed_json(index_path, index)

    with pytest.raises(R207FrozenLeafMaterializationError, match="r212_guide2vec_used"):
        materialize_r207_frozen_leaf_calibration(
            source_index_path=index_path,
            terminal_capture_path=capture_path,
            frozen_inference_capture_path=inference_path,
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_materializer_rejects_unsealed_or_mismatched_capture_and_never_overwrites(
    tmp_path: Path,
) -> None:
    index_path, capture_path, inference_path = _inputs(tmp_path)
    os.chmod(capture_path, 0o644)
    with pytest.raises(R207FrozenLeafMaterializationError, match="immutable mode 0444"):
        materialize_r207_frozen_leaf_calibration(
            source_index_path=index_path,
            terminal_capture_path=capture_path,
            frozen_inference_capture_path=inference_path,
            output_dir=tmp_path / "unsealed-output",
        )
    assert not (tmp_path / "unsealed-output").exists()

    os.chmod(capture_path, 0o444)
    output = tmp_path / "existing-output"
    output.mkdir()
    with pytest.raises(
        R207FrozenLeafMaterializationError, match="must not already exist"
    ):
        materialize_r207_frozen_leaf_calibration(
            source_index_path=index_path,
            terminal_capture_path=capture_path,
            frozen_inference_capture_path=inference_path,
            output_dir=output,
        )


def test_offline_cli_materializes_without_starting_an_external_workload(
    tmp_path: Path,
) -> None:
    index_path, capture_path, inference_path = _inputs(tmp_path)
    output = tmp_path / "cli-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_r207_frozen_leaf_calibration.py"),
            "--source-index",
            str(index_path),
            "--terminal-capture",
            str(capture_path),
            "--frozen-inference-capture",
            str(inference_path),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["output_dir"] == str(output)
    assert (output / "r207-frozen-leaf-calibration-receipt.json").is_file()
