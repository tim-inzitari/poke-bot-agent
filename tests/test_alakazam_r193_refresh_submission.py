from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage_alakazam_rtp_r175_milestone_submissions import validate_commit


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_iter15(tmp_path: Path, *, epochs: int) -> Path:
    run = tmp_path / "run"
    commits = run / "commits"
    checkpoints = run / "checkpoints"
    commits.mkdir(parents=True)
    checkpoints.mkdir()
    candidate = checkpoints / "iter_00015.pt"
    candidate.write_bytes(b"iteration-15 refreshed learner")
    rehearsal = checkpoints / "expert_before_iter_00015.pt"
    rehearsal.write_bytes(b"owner-r193 large expert refresh")
    refresh = {
        "before_iteration": 15,
        "epochs": epochs,
        "checkpoint": str(rehearsal),
        "checkpoint_digest": _digest(rehearsal),
        "manifest": {
            "dates": [
                "2026-08-01",
                "2026-08-02",
                "2026-08-03",
                "2026-08-04",
                "2026-08-05",
            ]
        },
        "loss_weights": {"alakazam_guide": 0.05},
        "expanded_head_training": {"trained_this_epoch": ["action_q"]},
    }
    (commits / "iter_00015.json").write_text(
        json.dumps(
            {
                "last_completed_iteration": 15,
                "next_iteration": 16,
                "learner": {"digest": _digest(candidate)},
                "history": [
                    {
                        "iteration": 15,
                        "completed": True,
                        "candidate": {"digest": _digest(candidate)},
                        "expert_rehearsal": refresh,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_iter15_submission_requires_exact_r193_large_refresh(
    tmp_path: Path,
) -> None:
    run = _materialize_iter15(tmp_path, epochs=25)
    _, checkpoint, digest, refresh = validate_commit(run, 15)
    assert checkpoint.name == "iter_00015.pt"
    assert digest == _digest(checkpoint)
    assert refresh["epochs"] == 25


def test_iter15_submission_rejects_ordinary_five_epoch_refresh(
    tmp_path: Path,
) -> None:
    run = _materialize_iter15(tmp_path, epochs=5)
    with pytest.raises(RuntimeError, match="owner-r193 25-epoch"):
        validate_commit(run, 15)
