from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poke_bot.archetype_family_activation import (
    boundary_pause_hook,
    validate_iteration9_upload_trigger,
)
from scripts.materialize_marnie_iteration9_upload_trigger import (
    TriggerMaterializationError,
    materialize_trigger,
    sha256,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _evidence(tmp_path: Path) -> dict[str, Path | dict]:
    commit = _write_json(tmp_path / "run/commits/iter_00009.json", {"iteration": 9})
    checkpoint = tmp_path / "run/checkpoints/iter_00009.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact iteration nine checkpoint")
    package = tmp_path / "submissions/build/submission.tar.gz"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"exact package bundle")
    uploaded = tmp_path / "submissions/copy-1/submission.tar.gz"
    uploaded.parent.mkdir(parents=True)
    uploaded.write_bytes(package.read_bytes())
    deck = tmp_path / "submissions/pinned-marnie.deck.csv"
    deck.write_bytes(b"exact representative deck")

    queue_identity = {
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "copy_number": 1,
        "turn_order_preference": "first_if_allowed",
        "label": "marnie-s-grimmsnarl-ex training milestone iter 9 copy 1/1 first abcdef123456",
        "checkpoint_checksum": sha256(checkpoint),
        "model_checksum": sha256(checkpoint),
        "deck_file_checksum": sha256(deck),
        "deck_cards_checksum": "sha256:" + "c" * 64,
        "representatives_checksum": "sha256:" + "d" * 64,
        "matchup_tree_checksum": "sha256:" + "e" * 64,
        "search_config_checksum": "sha256:" + "f" * 64,
        "belief_decks_checksum": "sha256:" + "1" * 64,
        "gate_id": "final-format-marnie-s-grimmsnarl-ex-training-snapshot-r107",
        "iteration": 9,
        "competition": "pokemon-tcg-ai-battle",
        "file": str(uploaded.resolve()),
        "file_sha256": sha256(uploaded),
    }
    milestone = {
        "schema": "poke_bot.final_format_milestone_submission/v1",
        "status": "queued",
        "owner_decision_revision": 107,
        "iteration": 9,
        "commit": str(commit.resolve()),
        "commit_sha256": sha256(commit),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "bundle": {
            "path": str(package.resolve()),
            "sha256": sha256(package),
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "checkpoint_archetype_id": "marnie-s-grimmsnarl-ex",
            "deck": {
                "path": str(deck.resolve()),
                "file_sha256": sha256(deck),
                "cards_sha256": queue_identity["deck_cards_checksum"],
                "representatives_sha256": queue_identity["representatives_checksum"],
            },
            "contents": {"model_sha256": sha256(checkpoint)},
        },
        "queue_entry": queue_identity,
        "training_stop_or_freeze_authority": False,
    }
    milestone_path = _write_json(
        tmp_path / "receipts/iter_00009.json", milestone
    )

    nonce = "marnie-iter9-copy1-automatic-exact"
    receipts = tmp_path / "attempts"
    auth_path = _write_json(
        receipts / f"stamp-{nonce}.authorization-consumed.json",
        {
            "schema": "poke_bot.kaggle_submission_authorization/v1",
            "nonce": nonce,
            "remaining_uses": 0,
            "consumed_before_upload": True,
            "submission_file_checksum": sha256(uploaded),
            "frozen_checkpoint_checksum": sha256(checkpoint),
        },
    )
    _write_json(
        receipts / f"stamp-{nonce}.json",
        {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": nonce,
            "authorization_consumed": str(auth_path.resolve()),
            "returncode": 0,
            "identity": {
                "file": str(uploaded.resolve()),
                "file_sha256": sha256(uploaded),
                "competition": queue_identity["competition"],
                "message": queue_identity["label"],
            },
        },
    )
    accepted = {
        **queue_identity,
        "queue_status": "accepted",
        "kaggle_status": "COMPLETE",
        "submission_id": 12345,
        "submitted_at": "2026-08-03T12:00:00+00:00",
        "returned_score": 0.75,
        "failure_reason": None,
        "one_shot_authorization_nonce": nonce,
    }
    queue = {
        "schema": "poke_bot.kaggle_submission_queue/v1",
        "queue": [accepted],
    }
    queue_path = _write_json(tmp_path / "state/kaggle-queue.json", queue)
    return {
        "queue": queue_path,
        "milestone": milestone_path,
        "attempts": receipts,
        "output": tmp_path / "state/family/iteration9-upload-trigger.json",
        "queue_payload": queue,
    }


def test_pending_complete_evidence_creates_no_trigger_and_pauses_after_iter9(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    queue = copy.deepcopy(evidence["queue_payload"])
    queue["queue"][0]["queue_status"] = "submitted"
    queue["queue"][0]["kaggle_status"] = "RUNNING"
    _write_json(evidence["queue"], queue)

    result = materialize_trigger(
        queue_path=evidence["queue"],
        milestone_receipt=evidence["milestone"],
        attempt_receipts=evidence["attempts"],
        output=evidence["output"],
    )
    assert result == {"status": "not_ready", "reason": "exact_upload_not_complete"}
    assert not evidence["output"].exists()
    decision = boundary_pause_hook(
        request_path=tmp_path / "state/family/activation-request.json",
        trigger_path=evidence["output"],
        run_dir=tmp_path / "pretrigger-run",
        committed_iteration=9,
        learner_digest="sha256:learner",
        next_collection_started=False,
    )
    assert decision["action"] == "pause_for_required_evidence"
    assert decision["reason"] == "successful_iteration9_upload_pending"
    pause = json.loads(Path(decision["pause_receipt"]).read_text(encoding="utf-8"))
    assert pause["owner_revision"] == 134
    assert pause["committed_iteration"] == 9
    assert pause["target_iteration"] == 10
    assert pause["learner_sha256"] == "sha256:learner"
    assert pause["next_collection_started"] is False
    assert pause["restart_prevent_status"] == 75


def test_pending_upload_does_not_pause_before_iteration9(tmp_path: Path) -> None:
    trigger = tmp_path / "future-trigger.json"
    decision = boundary_pause_hook(
        request_path=None,
        trigger_path=trigger,
        run_dir=tmp_path / "iter8-run",
        committed_iteration=8,
        learner_digest="sha256:learner8",
        next_collection_started=False,
    )
    assert decision["action"] == "continue_unchanged"


def test_exact_complete_evidence_materializes_once_and_stops_old_collection(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    first = materialize_trigger(
        queue_path=evidence["queue"],
        milestone_receipt=evidence["milestone"],
        attempt_receipts=evidence["attempts"],
        output=evidence["output"],
    )
    assert first["status"] == "materialized"
    original_bytes = evidence["output"].read_bytes()
    trigger = json.loads(original_bytes)
    assert validate_iteration9_upload_trigger(trigger)["valid"] is True
    assert trigger["owner_revision"] == 130
    assert trigger["source_evidence"]["accepted_complete_queue_row"]["kaggle_status"] == "COMPLETE"

    second = materialize_trigger(
        queue_path=evidence["queue"],
        milestone_receipt=evidence["milestone"],
        attempt_receipts=evidence["attempts"],
        output=evidence["output"],
    )
    assert second["status"] == "already_materialized"
    assert evidence["output"].read_bytes() == original_bytes

    decision = boundary_pause_hook(
        request_path=tmp_path / "state/family/activation-request.json",
        trigger_path=evidence["output"],
        run_dir=tmp_path / "posttrigger-run",
        committed_iteration=9,
        learner_digest="sha256:learner",
        next_collection_started=False,
    )
    assert decision["action"] == "pause_for_required_evidence"
    pause = json.loads(Path(decision["pause_receipt"]).read_text(encoding="utf-8"))
    assert pause["restart_prevent_status"] == 75


def test_trigger_rejects_milestone_queue_or_consumed_evidence_mismatch(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    queue = copy.deepcopy(evidence["queue_payload"])
    queue["queue"][0]["competition"] = "different-competition"
    _write_json(evidence["queue"], queue)
    with pytest.raises(TriggerMaterializationError, match="milestone identity"):
        materialize_trigger(
            queue_path=evidence["queue"],
            milestone_receipt=evidence["milestone"],
            attempt_receipts=evidence["attempts"],
            output=evidence["output"],
        )
    assert not evidence["output"].exists()

    evidence = _evidence(tmp_path / "missing-auth")
    next(Path(evidence["attempts"]).glob("*.authorization-consumed.json")).unlink()
    with pytest.raises(TriggerMaterializationError, match="consumed one-shot"):
        materialize_trigger(
            queue_path=evidence["queue"],
            milestone_receipt=evidence["milestone"],
            attempt_receipts=evidence["attempts"],
            output=evidence["output"],
        )
    assert not evidence["output"].exists()


def test_local_units_advertise_stable_paths_without_live_activation() -> None:
    root = Path(__file__).resolve().parents[1]
    stable_root = "/home/pokebot/poke-bot-agent/outputs/state/marnie-archetype-family-r130"
    drop_in = (
        root
        / "deploy/systemd/pokebot-final-format-marnie-r104-h10-rl.service.d/zzzzz-family-boundary-r130.conf"
    ).read_text(encoding="utf-8")
    producer = (
        root / "deploy/systemd/pokebot-marnie-family-iteration9-trigger-r130.service"
    ).read_text(encoding="utf-8")
    timer = (
        root / "deploy/systemd/pokebot-marnie-family-iteration9-trigger-r130.timer"
    ).read_text(encoding="utf-8")

    assert f"POKEBOT_MARNIE_ITERATION9_UPLOAD_TRIGGER={stable_root}/iteration9-upload-trigger.json" in drop_in
    assert f"POKEBOT_MARNIE_FAMILY_ACTIVATION_REQUEST={stable_root}/activation-request.json" in drop_in
    assert "materialize_marnie_iteration9_upload_trigger.py" in producer
    assert "final-format-marnie-milestone-submissions-r107/iter_00009.json" in producer
    assert f"--output {stable_root}/iteration9-upload-trigger.json" in producer
    assert "OnUnitActiveSec=60" in timer
