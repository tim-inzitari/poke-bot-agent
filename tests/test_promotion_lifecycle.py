from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint
from poke_bot.promotion import (
    CheckpointIdentity,
    PromotionGateConfig,
    evaluate_candidate_gate,
    next_iteration,
)


def test_immutable_candidate_identity_and_overwrite_guard(tmp_path: Path) -> None:
    path0 = checkpoint.candidate_path("run", 0, root=tmp_path)
    path1 = checkpoint.candidate_path("run", 1, root=tmp_path)
    checkpoint.immutable_torch_save({"weight": torch.tensor([1.0])}, path0)
    checkpoint.immutable_torch_save({"weight": torch.tensor([2.0])}, path1)

    id0 = CheckpointIdentity.from_path(path0)
    id1 = CheckpointIdentity.from_path(path1)
    assert id0.path != id1.path
    assert id0.digest != id1.digest
    with pytest.raises(FileExistsError):
        checkpoint.immutable_torch_save({"weight": torch.tensor([3.0])}, path0)
    assert CheckpointIdentity.from_path(path0) == id0


def test_promotion_acceptance_and_rejection_are_seat_stratified() -> None:
    cfg = PromotionGateConfig(
        min_games=4,
        min_complete_pairs=2,
        threshold=0.5,
        confidence=0.90,
        bootstrap_resamples=200,
    )
    accepted = [
        {"valid": True, "candidate_seat": seat, "winner": seat, "pair_id": pair}
        for pair in ("p0", "p1")
        for seat in (0, 1)
    ]
    rejected = [
        {
            "valid": True,
            "candidate_seat": seat,
            "winner": 1 - seat,
            "pair_id": pair,
        }
        for pair in ("p0", "p1")
        for seat in (0, 1)
    ]
    assert evaluate_candidate_gate(accepted, cfg)["passed"] is True
    assert evaluate_candidate_gate(accepted, cfg)["pairing_claimed"] is False
    assert evaluate_candidate_gate(rejected, cfg)["passed"] is False

    incomplete = accepted[:-1]
    report = evaluate_candidate_gate(incomplete, cfg)
    assert report["valid"] is False
    assert report["passed"] is False


def test_resume_continues_at_next_completed_iteration() -> None:
    state = {
        "iteration": 99,  # in-flight/current index is deliberately ignored
        "history": [
            {"iteration": 2, "completed": True},
            {"iteration": 3, "completed": True},
        ],
        "last_completed_iteration": 3,
    }
    assert next_iteration(state, checkpoint_iteration=3) == 4
    assert next_iteration({"history": []}, checkpoint_iteration=-1) == 0
