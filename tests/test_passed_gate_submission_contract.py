from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from poke_bot.pure_rl.model_registry import sha256
from scripts import handle_passed_gate as handler
from scripts.archive_passed_system_to_elmo import _validate_handler


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_handler_defaults_to_one_submission_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required_paths = {
        "run-dir": tmp_path / "run",
        "contract": tmp_path / "gate.json",
        "registry-root": tmp_path / "registry",
        "submission-root": tmp_path / "submissions",
        "state": tmp_path / "state.json",
        "lock": tmp_path / "handler.lock",
        "authorization": tmp_path / "authorization.json",
        "submission-receipts": tmp_path / "receipts",
        "continue-drop-in-source": tmp_path / "source.conf",
        "continue-drop-in-target": tmp_path / "target.conf",
    }
    argv = ["handle_passed_gate.py"]
    for name, path in required_paths.items():
        argv.extend((f"--{name}", str(path)))
    monkeypatch.setattr(sys, "argv", argv)

    assert handler._parse_args().submission_count == 1


def test_status_143_recovery_does_not_mask_real_training_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(tuple(command))
        if "show" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Result=exit-code\nExecMainStatus=143\n"
                    "ActiveState=failed\nSubState=failed\n"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)
    result = handler._recover_status_143_stop("trainer.service")
    assert result["recovered"] is True
    assert any("reset-failed" in command for command in calls)
    assert any("start" in command for command in calls)

    calls.clear()

    def real_failure(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(tuple(command))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "Result=exit-code\nExecMainStatus=1\n"
                "ActiveState=failed\nSubState=failed\n"
            ),
        )

    monkeypatch.setattr(handler.subprocess, "run", real_failure)
    result = handler._recover_status_143_stop("trainer.service")
    assert result["recovered"] is False
    assert len(calls) == 1


def test_status_143_recovery_respects_valid_managed_boundary_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = tmp_path / "maintenance.json"
    _write(
        lock,
        {
            "schema": "poke_bot.managed_training_maintenance/v1",
            "training_service": "trainer.service",
            "owner_pid": 123,
            "expires_at_epoch": 9_999_999_999.0,
        },
    )
    monkeypatch.setenv("POKEBOT_MANAGED_MAINTENANCE_LOCK", str(lock))
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(tuple(command))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "Result=exit-code\nExecMainStatus=143\n"
                "ActiveState=failed\nSubState=failed\n"
            ),
        )

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    result = handler._recover_status_143_stop("trainer.service")

    assert result["recovered"] is False
    assert result["managed_maintenance"]["owner_pid"] == 123
    assert len(calls) == 1


def _attempt_record(
    tmp_path: Path,
    *,
    slot: int,
    bundle_bytes: bytes = b"one exact submission bundle",
    returncode: int = 0,
) -> dict:
    copy_path = tmp_path / f"copy-{slot}" / "submission.tar.gz"
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    copy_path.write_bytes(bundle_bytes)
    digest = sha256(copy_path)
    nonce = handler._nonce(digest, slot)
    consumed = (
        tmp_path
        / "receipts"
        / f"20260722T000000-{nonce}.authorization-consumed.json"
    )
    attempt = tmp_path / "receipts" / f"20260722T000001-{nonce}.json"
    _write(
        consumed,
        {
            "schema": handler.AUTH_SCHEMA,
            "nonce": nonce,
            "remaining_uses": 0,
            "consumed_before_upload": True,
        },
    )
    _write(
        attempt,
        {
            "schema": handler.ATTEMPT_SCHEMA,
            "nonce": nonce,
            "authorization_consumed": str(consumed.resolve()),
            "returncode": returncode,
            "identity": {
                "nonce": nonce,
                "file_sha256": digest,
                "message": f"copy {slot}",
            },
            "retry_requires_new_explicit_user_approval": True,
            "started_at_epoch": 1.0,
            "completed_at_epoch": 2.0,
        },
    )
    return handler._submission_attempt_record(
        copy={"slot": slot, "path": str(copy_path), "sha256": digest},
        nonce=nonce,
        consumed_paths=[consumed],
        attempt_paths=[attempt],
        recovered=False,
        process_returncode=returncode,
        timed_out=False,
        message=f"copy {slot}",
        started_at_epoch=1.0,
        completed_at_epoch=2.0,
    )


def test_two_successful_uploads_require_two_unique_durable_receipts(
    tmp_path: Path,
) -> None:
    first = _attempt_record(tmp_path, slot=1)
    second = _attempt_record(tmp_path, slot=2)
    assert first["file_sha256"] == second["file_sha256"]

    validated = handler.validate_two_successful_submission_attempts(
        [first, second], expected_bundle_sha256=first["file_sha256"]
    )
    assert [row["slot"] for row in validated] == [1, 2]
    assert all(row["succeeded"] is True for row in validated)


def test_pending_copies_are_persistent_idempotent_and_checkpoint_bound(
    tmp_path: Path,
) -> None:
    bundle = b"one exact frozen checkpoint bundle"
    digest = "sha256:" + "a" * 64
    copies = []
    for slot in (1, 2):
        path = tmp_path / f"copy-{slot}" / "submission.tar.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(bundle)
        copies.append(
            {
                "slot": slot,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "specialist_id": "alakazam",
                "model_sha256": digest,
                "deck_sha256": "sha256:" + "b" * 64,
                "deck_cards_sha256": "sha256:" + "c" * 64,
                "representatives_sha256": "sha256:" + "d" * 64,
                "matchup_tree_sha256": "sha256:" + "e" * 64,
                "search_config_sha256": "sha256:" + "f" * 64,
                "belief_decks_sha256": "sha256:" + "1" * 64,
            }
        )
    queue_path = tmp_path / "submission-queue.json"
    first = handler.queue_submission_copies(
        queue_path=queue_path,
        copies=copies,
        gate_plan={
            "iteration": 30,
            "gate_id": "alakazam-strong-public-gate",
            "checkpoint_digest": digest,
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    second = handler.queue_submission_copies(
        queue_path=queue_path,
        copies=copies,
        gate_plan={
            "iteration": 30,
            "gate_id": "alakazam-strong-public-gate",
            "checkpoint_digest": digest,
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["schema"] == handler.SUBMISSION_QUEUE_SCHEMA
    assert payload["daily_submission_limit"] == 5
    assert payload["automatic_one_shot_authorization_on_training_complete"] is True
    assert payload["one_shot_authorization_uses"] == 1
    assert (
        payload["standing_owner_decision_source"]
        == "GOAL.md#/decision-ledger/revision-18"
    )
    assert payload["queue_order"] == "oldest_first"
    assert len(payload["queue"]) == 2
    assert first == second
    assert [row["copy_number"] for row in first] == [1, 2]
    assert all(row["checkpoint_checksum"] == digest for row in first)
    assert all(row["model_checksum"] == digest for row in first)
    assert all(row["deck_file_checksum"] == "sha256:" + "b" * 64 for row in first)
    assert all(
        row["matchup_tree_checksum"] == "sha256:" + "e" * 64
        for row in first
    )
    assert all(
        row["search_config_checksum"] == "sha256:" + "f" * 64
        for row in first
    )
    assert all(
        row["belief_decks_checksum"] == "sha256:" + "1" * 64
        for row in first
    )
    assert all(row["gate_id"] == "alakazam-strong-public-gate" for row in first)
    assert all(row["iteration"] == 30 for row in first)
    assert all(row["queue_status"] == "pending" for row in first)


def test_direct_policy_queue_binds_intentional_search_asset_absence(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "submission.tar.gz"
    bundle.write_bytes(b"direct only")
    digest = "sha256:" + "a" * 64
    rows = handler.queue_submission_copies(
        queue_path=tmp_path / "r274-queue.json",
        copies=[
            {
                "slot": 1,
                "path": str(bundle),
                "sha256": sha256(bundle),
                "specialist_id": "alakazam",
                "model_sha256": digest,
                "deck_sha256": "sha256:" + "b" * 64,
                "deck_cards_sha256": "sha256:" + "c" * 64,
                "representatives_sha256": "sha256:" + "d" * 64,
                "matchup_tree_sha256": "sha256:" + "e" * 64,
                "search_config_sha256": "",
                "belief_decks_sha256": "",
                "search_assets_packaged": False,
                "rtp_mode": "off",
            }
        ],
        gate_plan={
            "iteration": 0,
            "gate_id": "alakazam-r274-bootstrap",
            "checkpoint_digest": digest,
            "owner_decision_source": "GOAL.md#/revision-264-TRAINING",
        },
        specialist_id="alakazam",
        competition="pokemon-tcg-ai-battle",
    )
    queue = json.loads((tmp_path / "r274-queue.json").read_text())
    assert queue["standing_owner_decision_source"] == "GOAL.md#/revision-264-TRAINING"
    assert rows[0]["search_assets_packaged"] is False
    assert rows[0]["search_config_checksum"] == ""
    assert rows[0]["belief_decks_checksum"] == ""


def test_failed_or_ambiguous_upload_is_never_success(tmp_path: Path) -> None:
    first = _attempt_record(tmp_path, slot=1, returncode=1)
    second = _attempt_record(tmp_path, slot=2)
    assert first["attempted"] is True
    assert first["succeeded"] is False
    assert first["automatic_retry"] is False

    with pytest.raises(RuntimeError, match="slot 1 is not proven successful"):
        handler.validate_two_successful_submission_attempts(
            [first, second], expected_bundle_sha256=first["file_sha256"]
        )

    recovered_unknown = dict(first)
    recovered_unknown["recovered_from_consumed_authorization"] = True
    recovered_unknown["guard_attempt_receipts"] = []
    recovered_unknown["guard_attempt_receipt_digests"] = {}
    with pytest.raises(RuntimeError, match="one_guard_attempt_receipt"):
        handler.validate_two_successful_submission_attempts(
            [recovered_unknown, second],
            expected_bundle_sha256=first["file_sha256"],
        )


def test_archive_rejects_attempted_but_failed_uploads(tmp_path: Path) -> None:
    failed = _attempt_record(tmp_path, slot=1, returncode=1)
    succeeded = _attempt_record(tmp_path, slot=2)
    digest = "sha256:" + "a" * 64
    state = {
        "phase": "complete_handoff_started",
        "approved_submission_count": 2,
        "automatic_retries": False,
        "all_submissions_succeeded": True,
        "successful_submission_count": 2,
        "gate": {"checkpoint_digest": digest},
        "submission_bundle": {"sha256": succeeded["file_sha256"]},
        "submission_attempts": [failed, succeeded],
    }
    with pytest.raises(RuntimeError, match="slot 1 is not proven successful"):
        _validate_handler(state, {"checkpoint_digest": digest})

    state["submission_attempts"] = [
        _attempt_record(tmp_path / "ok", slot=1),
        _attempt_record(tmp_path / "ok", slot=2),
    ]
    state["submission_bundle"]["sha256"] = state["submission_attempts"][0][
        "file_sha256"
    ]
    _validate_handler(state, {"checkpoint_digest": digest})


def test_failed_two_attempts_stop_without_handoff_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "SPECIALIST_GATE_PASSED").write_text("{}\n", encoding="utf-8")
    submission_root = tmp_path / "submissions"
    bundle_path = submission_root / "build" / "submission.tar.gz"
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_bytes(b"one exact submission bundle")
    bundle_digest = sha256(bundle_path)
    attempts = {
        1: _attempt_record(tmp_path / "evidence", slot=1, returncode=1),
        2: _attempt_record(tmp_path / "evidence", slot=2),
    }
    submit_calls: list[int] = []
    handoffs: list[str] = []

    monkeypatch.setattr(handler, "validate_exact_pass", lambda *_: {"iteration": 15})
    monkeypatch.setattr(handler, "freeze_exact_pass", lambda *_args, **_kw: {})
    monkeypatch.setattr(
        handler, "materialize_pinned_specialist_deck", lambda **_kw: {}
    )
    monkeypatch.setattr(
        handler,
        "build_submission_bundle",
        lambda **_kw: {
            "path": str(bundle_path),
            "sha256": bundle_digest,
            "attestation": str(bundle_path) + ".go-first-verified.json",
        },
    )
    monkeypatch.setattr(
        handler,
        "_copy_submission_slot",
        lambda _bundle, _root, slot: {
            "slot": slot,
            "path": attempts[slot]["file"],
            "sha256": bundle_digest,
        },
    )

    def submit(**kwargs):
        slot = int(kwargs["copy"]["slot"])
        submit_calls.append(slot)
        return attempts[slot]

    monkeypatch.setattr(handler, "submit_one_approved_copy", submit)
    monkeypatch.setattr(handler, "_service_main_pid", lambda _service: 123)
    monkeypatch.setattr(handler, "_start_handoff", handoffs.append)

    args = SimpleNamespace(
        submission_count=2,
        lock=tmp_path / "handler.lock",
        state=tmp_path / "handler-state.json",
        run_dir=run_dir,
        contract=tmp_path / "contract.json",
        training_service="trainer.service",
        once=True,
        poll_seconds=0.01,
        registry_root=tmp_path / "registry",
        family="family",
        display_name="display",
        representatives=tmp_path / "reps.json",
        archetype="alakazam",
        submission_root=submission_root,
        competition="pokemon-tcg-ai-battle",
        kaggle=tmp_path / "kaggle",
        authorization=tmp_path / "authorization.json",
        submission_receipts=tmp_path / "receipts",
        upload_timeout_seconds=30.0,
        handoff_service="archive.service",
        continue_drop_in_source=tmp_path / "continue-source.conf",
        continue_drop_in_target=tmp_path / "continue-target.conf",
        python=Path("/usr/bin/python3"),
    )

    assert handler.run(args) == 0
    saved = json.loads(args.state.read_text())
    assert submit_calls == [1, 2]
    assert handoffs == []
    assert saved["phase"] == "two_submissions_failed_or_unknown"
    assert saved["all_submissions_succeeded"] is False
    assert saved["automatic_retries"] is False

    assert handler.run(args) == 0
    assert submit_calls == [1, 2]
    assert handoffs == []


def test_post_gate_unit_requires_real_registry_artifacts() -> None:
    unit = (
        ROOT / "deploy/systemd/pokebot-post-alakazam-gate-pipeline.service"
    ).read_text(encoding="utf-8")
    assert "/manifest.json" in unit
    assert "/PROTECTED_DO_NOT_PRUNE.json" in unit
    assert "/model.pt" in unit
    assert "FROZEN_MODEL.json" not in unit
