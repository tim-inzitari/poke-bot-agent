from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.handle_passed_gate import queue_submission_copies
from scripts.stage_final_format_alakazam_milestone_submissions import (
    eligible_iterations,
    receipt_schema,
    validate_commit,
)


def test_milestone_cadence_uses_commit_before_each_fifth_iteration(
    tmp_path: Path,
) -> None:
    commits = tmp_path / "commits"
    commits.mkdir()
    for iteration in (0, 4, 5, 9, 10, 14, 180, 184, 185):
        (commits / f"iter_{iteration:05d}.json").touch()
    assert eligible_iterations(tmp_path) == [4, 9, 14, 184]


def test_marnie_milestone_cadence_includes_iteration_zero(
    tmp_path: Path,
) -> None:
    commits = tmp_path / "commits"
    commits.mkdir()
    for iteration in (0, 1, 4, 5, 9, 10, 14, 15, 19):
        (commits / f"iter_{iteration:05d}.json").touch()
    assert eligible_iterations(
        tmp_path,
        maximum_iteration=184,
        include_iteration_zero=True,
    ) == [0, 4, 9, 14, 19]
    assert receipt_schema("marnie-s-grimmsnarl-ex") == (
        "poke_bot.final_format_milestone_submission/v1"
    )


def test_marnie_iteration9_uses_exact_committed_learner_after_gate_rollback(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    commits = run_dir / "commits"
    checkpoints = run_dir / "checkpoints"
    commits.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    candidate = checkpoints / "iter_00009.pt"
    candidate.write_bytes(b"promoted candidate later rejected by exact gate")
    learner = checkpoints / "iter_00007.pt"
    learner.write_bytes(b"stronger exact-gate learner")
    learner_digest = "sha256:" + hashlib.sha256(learner.read_bytes()).hexdigest()
    (commits / "iter_00009.json").write_text(
        json.dumps(
            {
                "last_completed_iteration": 9,
                "next_iteration": 10,
                "learner": {"path": str(learner), "digest": learner_digest},
                "history": [
                    {
                        "iteration": 9,
                        "completed": True,
                        "candidate": {
                            "path": str(candidate),
                            "digest": "sha256:"
                            + hashlib.sha256(candidate.read_bytes()).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _, selected, digest, role = validate_commit(
        run_dir,
        9,
        prefer_committed_learner=True,
    )
    assert selected == learner.resolve()
    assert digest == learner_digest
    assert role == "committed_learner"


def test_marnie_managed_watcher_is_registration_gated_and_nonblocking() -> None:
    root = Path(__file__).resolve().parents[1]
    service = (
        root
        / "deploy/systemd/"
        "pokebot-final-format-marnie-r104-milestone-submissions.service"
    ).read_text(encoding="utf-8")
    timer = (
        root
        / "deploy/systemd/"
        "pokebot-final-format-marnie-r104-milestone-submissions.timer"
    ).read_text(encoding="utf-8")
    protocol = (
        root / "config/rl_protocol.yaml"
    ).read_text(encoding="utf-8")

    assert "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-runtime-registration.json" in service
    assert "--specialist-id marnie-s-grimmsnarl-ex" in service
    assert "--include-iteration-zero" in service
    assert "--maximum-iteration 184" in service
    assert "--owner-decision-revision 107" in service
    assert "Nice=10" in service
    assert "OnUnitActiveSec=60" in timer
    assert "zero_indexed_iterations: [0, 4, 9, 14]" in protocol
    assert "terminal_completion_submission_remains_separate: true" in protocol


def test_milestone_queue_row_is_nonterminal_and_first_preferring(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    copy = {
        "slot": 1,
        "path": str(tmp_path / "submission.tar.gz"),
        "sha256": "sha256:" + "b" * 64,
        "specialist_id": "alakazam",
        "turn_order_preference": "first_if_allowed",
        "model_sha256": digest,
        "deck_sha256": "sha256:" + "c" * 64,
        "deck_cards_sha256": "sha256:" + "d" * 64,
        "representatives_sha256": "sha256:" + "e" * 64,
        "matchup_tree_sha256": "sha256:" + "f" * 64,
        "search_config_sha256": "sha256:" + "1" * 64,
        "belief_decks_sha256": "sha256:" + "2" * 64,
    }
    queue = tmp_path / "queue.json"
    rows = queue_submission_copies(
        queue_path=queue,
        copies=[copy],
        gate_plan={
            "checkpoint_digest": digest,
            "gate_id": "final-format-alakazam-training-milestone-r97",
            "iteration": 4,
            "completion_authority": "training_milestone_snapshot",
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    assert len(rows) == 1
    assert rows[0]["label"].startswith("alakazam training milestone iter 4")
    assert rows[0]["turn_order_preference"] == "first_if_allowed"
    assert rows[0]["queue_status"] == "pending"
    payload = json.loads(queue.read_text(encoding="utf-8"))
    assert payload["queue"][0] == rows[0]
