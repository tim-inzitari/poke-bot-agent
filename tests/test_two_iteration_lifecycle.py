from pathlib import Path

import torch

from poke_bot import checkpoint
from poke_bot.promotion import (
    CheckpointIdentity,
    PromotionGateConfig,
    evaluate_candidate_gate,
    next_iteration,
)


def _games(candidate_wins: bool) -> list[dict]:
    rows = []
    for game in range(4):
        seat = game % 2
        rows.append(
            {
                "valid": True,
                "candidate_seat": seat,
                "winner": seat if candidate_wins else 1 - seat,
                "pair_id": None,
            }
        )
    return rows


def test_two_iteration_reject_then_reload_promoted_digest(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    checkpoint.immutable_torch_save({"weight": torch.tensor([0.0])}, parent_path)
    incumbent = CheckpointIdentity.from_path(parent_path)
    cfg = PromotionGateConfig(
        min_games=4,
        min_complete_pairs=2,
        threshold=0.5,
        confidence=0.9,
        bootstrap_resamples=200,
    )
    state = {"history": [], "last_completed_iteration": -1}
    version = 0

    for iteration, wins in enumerate((False, True)):
        candidate_path = checkpoint.candidate_path(
            "smoke", iteration, root=tmp_path
        )
        checkpoint.immutable_torch_save(
            {"weight": torch.tensor([float(iteration + 1)])},
            candidate_path,
        )
        candidate = CheckpointIdentity.from_path(candidate_path)
        report = evaluate_candidate_gate(_games(wins), cfg)
        if report["passed"]:
            version += 1
            acknowledgements = [
                {
                    "ok": True,
                    "version": version,
                    "checkpoint_digest": candidate.digest,
                }
                for _ in range(2)
            ]
            assert all(
                ack["checkpoint_digest"] == candidate.digest
                and ack["version"] == version
                for ack in acknowledgements
            )
            incumbent = candidate
        state["history"].append(
            {
                "iteration": iteration,
                "completed": True,
                "promotion": report,
                "incumbent_after": incumbent.as_dict(),
            }
        )
        state["last_completed_iteration"] = iteration

    assert state["history"][0]["promotion"]["passed"] is False
    assert state["history"][0]["incumbent_after"]["path"] == str(
        parent_path.resolve()
    )
    assert state["history"][1]["promotion"]["passed"] is True
    assert incumbent.path.endswith("smoke.candidate.iter000001.pt")
    assert version == 1
    assert next_iteration(state, checkpoint_iteration=1) == 2
