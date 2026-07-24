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
