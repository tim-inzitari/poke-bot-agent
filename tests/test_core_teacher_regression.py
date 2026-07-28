from __future__ import annotations

import json
from pathlib import Path

from poke_bot import checkpoint
from scripts import run_core_teacher_regression as regression


def _model(path: Path, value: float) -> Path:
    checkpoint.immutable_torch_save(
        {"model_state_dict": {"weight": __import__("torch").tensor([value])}},
        path,
    )
    return path


def test_regression_uses_established_gate_for_each_teacher(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = _model(tmp_path / "candidate.pt", 1.0)
    teacher_a = _model(tmp_path / "a.pt", 2.0)
    teacher_b = _model(tmp_path / "b.pt", 3.0)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
                "core_refresh": {
                    "teachers": [
                        {
                            "specialist_id": "alakazam",
                            "mode": "frozen_inference_only",
                            "checkpoint": str(teacher_a),
                            "checksum": checkpoint.checkpoint_digest(teacher_a),
                        },
                        {
                            "specialist_id": "hops-trevenant",
                            "mode": "frozen_inference_only",
                            "checkpoint": str(teacher_b),
                            "checksum": checkpoint.checkpoint_digest(teacher_b),
                        },
                    ]
                },
                "acceptance": {
                    "games_per_teacher": 80,
                    "gate_threshold": 0.35,
                    "aggregate_gate_threshold": 0.4,
                    "confidence": 0.9,
                    "gate_thresholds_authoritative_source": "registry args",
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        regression,
        "_our_decks",
        lambda _mode, specialist_id: [(specialist_id, [1, 2, 3])],
    )

    def evaluate(**kwargs):
        calls.append(kwargs)
        return (
            {
                "valid": True,
                "passed": True,
                "wr": 0.45,
                "games": kwargs["n_games"],
                "config": {"threshold": kwargs["threshold"]},
            },
            [],
        )

    monkeypatch.setattr(regression, "_promotion_eval", evaluate)
    output = tmp_path / "result.json"
    result = regression.run(
        contract_path=contract,
        candidate=candidate,
        output=output,
        workers=12,
    )

    assert result["passed"] is True
    assert [row["specialist_id"] for row in result["results"]] == [
        "alakazam",
        "hops-trevenant",
    ]
    assert [row["n_games"] for row in calls] == [80, 80]
    assert [row["threshold"] for row in calls] == [0.35, 0.35]
    assert [row["confidence"] for row in calls] == [0.9, 0.9]
    assert result["training_eligible"] is False
    assert result["replay_eligible"] is False
    assert result["criteria"]["aggregate_raw_win_rate"] == 0.45
    assert result["criteria"]["confidence_intervals_diagnostic_only"] is True


def test_newly_passing_teacher_checksum_is_resolved_from_frozen_file(
    tmp_path: Path,
) -> None:
    teacher_a = _model(tmp_path / "trevenant.pt", 2.0)
    teacher_b = _model(tmp_path / "starmie.pt", 3.0)
    rows = regression._teacher_rows(
        {
            "core_refresh": {
                "teachers": [
                    {
                        "specialist_id": "hops-trevenant",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(teacher_a),
                        "checksum": checkpoint.checkpoint_digest(teacher_a),
                    },
                    {
                        "specialist_id": "starmie",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(teacher_b),
                        "checksum": None,
                    },
                ]
            }
        }
    )
    assert [row["specialist_id"] for row in rows] == [
        "hops-trevenant",
        "starmie",
    ]
    assert rows[1]["checksum"] == checkpoint.checkpoint_digest(teacher_b)


def test_historical_teacher_uses_checksum_bound_inference_derivative(
    tmp_path: Path,
) -> None:
    source = _model(tmp_path / "source.pt", 2.0)
    derivative = _model(tmp_path / "derivative.pt", 3.0)
    derivative_receipt = tmp_path / "derivative-manifest.json"
    derivative_receipt.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_roster_v5_derivative/v1",
                "specialist_id": "alakazam",
                "source_passing_checkpoint": str(source),
                "source_passing_checkpoint_digest": checkpoint.checkpoint_digest(
                    source
                ),
                "derived_checkpoint": str(derivative),
                "derived_checkpoint_digest": checkpoint.checkpoint_digest(
                    derivative
                ),
                "inference_only": True,
                "kaggle_submission_eligible": False,
                "retained_rows_byte_identical": True,
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "alakazam",
                        "source_passing_checkpoint_digest": (
                            checkpoint.checkpoint_digest(source)
                        ),
                        "checkpoint_digest": checkpoint.checkpoint_digest(
                            derivative
                        ),
                        "v5_derivative_receipt": str(derivative_receipt),
                        "frozen": True,
                        "public_mix_eligible": True,
                        "kaggle_submission_eligible": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = regression._teacher_rows(
        {
            "runtime": {"frozen_specialist_registry": str(registry)},
            "core_refresh": {
                "teachers": [
                    {
                        "specialist_id": "alakazam",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(source),
                        "checksum": checkpoint.checkpoint_digest(source),
                    },
                    {
                        "specialist_id": "dragapult-dusknoir",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(derivative),
                        "checksum": checkpoint.checkpoint_digest(derivative),
                    },
                ]
            },
        }
    )

    assert rows[0]["checkpoint"] == str(source.resolve())
    assert rows[0]["checksum"] == checkpoint.checkpoint_digest(source)
    assert rows[0]["evaluation_checkpoint"] == str(derivative.resolve())
    assert rows[0]["evaluation_checksum"] == checkpoint.checkpoint_digest(
        derivative
    )
    assert rows[1]["evaluation_checkpoint"] == str(derivative.resolve())


def test_current_format_teacher_uses_direct_registered_checkpoint(
    tmp_path: Path,
) -> None:
    teacher = _model(tmp_path / "direct.pt", 2.0)
    other = _model(tmp_path / "other.pt", 3.0)
    digest = checkpoint.checkpoint_digest(teacher)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "dragapult-dusknoir",
                        "checkpoint_digest": digest,
                        "source_passing_checkpoint_digest": None,
                        "frozen": True,
                        "public_mix_eligible": True,
                        "kaggle_submission_eligible": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = regression._teacher_rows(
        {
            "runtime": {"frozen_specialist_registry": str(registry)},
            "core_refresh": {
                "teachers": [
                    {
                        "specialist_id": "dragapult-dusknoir",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(teacher),
                        "checksum": digest,
                    },
                    {
                        "specialist_id": "dudunsparce",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(other),
                        "checksum": checkpoint.checkpoint_digest(other),
                    },
                ]
            },
        }
    )

    assert rows[0]["evaluation_checkpoint"] == str(teacher.resolve())
    assert rows[0]["evaluation_checksum"] == digest
    assert "evaluation_derivative_receipt" not in rows[0]


def test_completed_regression_ignores_downstream_contract_digest_only(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = _model(tmp_path / "candidate.pt", 1.0)
    teacher_a = _model(tmp_path / "a.pt", 2.0)
    teacher_b = _model(tmp_path / "b.pt", 3.0)
    contract = tmp_path / "contract.json"
    payload = {
        "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
        "core_refresh": {
            "teachers": [
                {
                    "specialist_id": "alakazam",
                    "mode": "frozen_inference_only",
                    "checkpoint": str(teacher_a),
                    "checksum": checkpoint.checkpoint_digest(teacher_a),
                },
                {
                    "specialist_id": "hops-trevenant",
                    "mode": "frozen_inference_only",
                    "checkpoint": str(teacher_b),
                    "checksum": checkpoint.checkpoint_digest(teacher_b),
                },
            ]
        },
        "acceptance": {
            "games_per_teacher": 80,
            "gate_threshold": 0.35,
            "aggregate_gate_threshold": 0.4,
            "confidence": 0.9,
            "gate_thresholds_authoritative_source": "registry args",
        },
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        regression,
        "_our_decks",
        lambda _mode, specialist_id: [(specialist_id, [1, 2, 3])],
    )
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return (
            {
                "valid": True,
                "passed": True,
                "wr": 0.45,
                "games": kwargs["n_games"],
                "config": {"threshold": kwargs["threshold"]},
            },
            [],
        )

    monkeypatch.setattr(regression, "_promotion_eval", evaluate)
    output = tmp_path / "result.json"
    first = regression.run(
        contract_path=contract,
        candidate=candidate,
        output=output,
        workers=2,
    )
    original_digest = first["identity"]["contract_digest"]

    payload["next_specialist"] = {
        "strict_priority_prefix": ["dragapult-dusknoir"],
        "minimum_decisions_by_specialist": {"dragapult-dusknoir": 10_000},
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")
    second = regression.run(
        contract_path=contract,
        candidate=candidate,
        output=output,
        workers=2,
    )

    assert len(calls) == 2
    assert second == first
    assert second["identity"]["contract_digest"] == original_digest
