from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import finalize_alakazam_new_list_direct_r241 as finalizer
from scripts import process_alakazam_new_list_direct_r241_submission_queue as queue_processor
from scripts import upload_alakazam_new_list_direct_r241_submission_queue as uploader


NOW = datetime(2026, 8, 10, 18, 30, tzinfo=timezone.utc)


def _write(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": finalizer.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _binding(tmp_path: Path) -> uploader.DirectUploadBinding:
    contract = _write(tmp_path / "contract.json", b"{\"r241\":true}\n")
    queue = _write(tmp_path / "queue.json", b"{\"queue\":true}\n")
    authorization = _write(tmp_path / "authorization.json", b"{\"auth\":true}\n")
    receipt = _write(tmp_path / "finalizer.json", b"{\"receipt\":true}\n")
    archive = _write(tmp_path / "submission.tar.gz", b"receipt-bound-direct-package")
    entry = {
        "single_use_nonce": "sha256:" + "a" * 64,
        "archive": _identity(archive),
    }
    partial = uploader.DirectUploadBinding(
        owner_contract=_identity(contract),
        queue=_identity(queue),
        authorization=_identity(authorization),
        finalizer_receipt=_identity(receipt),
        entry=entry,
        archive=_identity(archive),
        label="",
    )
    return uploader.DirectUploadBinding(
        owner_contract=partial.owner_contract,
        queue=partial.queue,
        authorization=partial.authorization,
        finalizer_receipt=partial.finalizer_receipt,
        entry=entry,
        archive=partial.archive,
        label=uploader._submission_label(entry),
    )


def _patch_binding(
    monkeypatch: pytest.MonkeyPatch, binding: uploader.DirectUploadBinding
) -> None:
    monkeypatch.setattr(
        uploader,
        "validate_current_handoff",
        lambda **_kwargs: binding,
    )


def _process(
    tmp_path: Path,
    *,
    upload: bool,
) -> dict[str, object]:
    return uploader.process_once(
        queue_path=tmp_path / "ignored-queue.json",
        authorization_path=tmp_path / "ignored-authorization.json",
        finalizer_receipt_path=tmp_path / "ignored-finalizer.json",
        receipts_dir=tmp_path / "receipts",
        kaggle=Path("/fake/kaggle"),
        upload=upload,
    )


def test_r241_uploader_preflight_never_queries_or_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    monkeypatch.setattr(
        uploader.guarded_kaggle,
        "_list_submissions",
        lambda *_args, **_kwargs: pytest.fail("preflight must not query Kaggle"),
    )
    monkeypatch.setattr(
        uploader.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("preflight must not invoke Kaggle"),
    )

    result = _process(tmp_path, upload=False)

    assert result["status"] == "local_preflight_passed_upload_not_requested"
    assert result["competition"] == "pokemon-tcg-ai-battle"
    assert result["turn_order_preference"] == "first_if_allowed"
    assert "FIRST IF ALLOWED" in str(result["submission_label"])
    assert not (tmp_path / "receipts").exists()


def test_r241_uploader_commits_immutable_attempt_and_upload_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    monkeypatch.setattr(uploader, "_now", lambda: NOW)
    monkeypatch.setattr(uploader.guarded_kaggle, "_list_submissions", lambda *_: [])
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="accepted", stderr="")

    monkeypatch.setattr(uploader.subprocess, "run", fake_run)

    first = _process(tmp_path, upload=True)

    assert first["status"] == "submitted"
    assert len(calls) == 1
    assert calls[0] == [
        "/fake/kaggle",
        "competitions",
        "submit",
        "-c",
        "pokemon-tcg-ai-battle",
        "-f",
        str(binding.archive["path"]),
        "-m",
        binding.label,
    ]
    attempt_path, upload_path, _lock = uploader._receipt_paths(
        tmp_path / "receipts", binding
    )
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    recorded = json.loads(upload_path.read_text(encoding="utf-8"))
    assert attempt["schema"] == uploader.ATTEMPT_RECEIPT_SCHEMA
    assert attempt["turn_order_preference"] == "first_if_allowed"
    assert recorded["schema"] == uploader.UPLOAD_RECEIPT_SCHEMA
    assert recorded["status"] == "submitted"
    assert recorded["attempt_receipt"]["sha256"] == finalizer.sha256_file(attempt_path)
    assert upload_path.stat().st_mode & 0o222 == 0

    second = _process(tmp_path, upload=True)

    assert second["status"] == "already_recorded_no_second_upload"
    assert len(calls) == 1


def test_r241_uploader_never_retries_an_unresolved_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    receipts = tmp_path / "receipts"
    attempt_path, _upload_path, _lock = uploader._receipt_paths(receipts, binding)
    uploader._write_immutable_json(
        attempt_path,
        uploader._attempt_payload(binding, created_at_utc=NOW.isoformat()),
        label="test r241 attempt receipt",
    )
    monkeypatch.setattr(
        uploader.guarded_kaggle,
        "_list_submissions",
        lambda *_args, **_kwargs: pytest.fail("unresolved attempt must not query Kaggle"),
    )
    monkeypatch.setattr(
        uploader.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("unresolved attempt must not submit"),
    )

    result = _process(tmp_path, upload=True)

    assert result["status"] == "prior_attempt_outcome_unknown_no_second_upload"
    assert uploader.exit_code_for_result(result) == 2


@pytest.mark.parametrize(
    ("submissions", "expected"),
    [
        (
            [
                {
                    "ref": str(55000000 + index),
                    "date": "2026-08-10T12:00:00+00:00",
                    "description": f"other-{index}",
                    "status": "SubmissionStatus.COMPLETE",
                }
                for index in range(5)
            ],
            "quota_wait",
        ),
        (
            [
                {
                    "ref": "55000011",
                    "date": "2026-08-10T18:00:00+00:00",
                    "description": "newest-other",
                    "status": "SubmissionStatus.COMPLETE",
                },
                {
                    "ref": "55000010",
                    "date": "2026-08-10T17:00:00+00:00",
                    "description": "anchor-other",
                    "status": "SubmissionStatus.COMPLETE",
                },
            ],
            "spacing_wait",
        ),
    ],
)
def test_r241_uploader_honors_guarded_quota_and_spacing_before_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submissions: list[dict[str, str]],
    expected: str,
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    monkeypatch.setattr(uploader, "_now", lambda: NOW)
    monkeypatch.setattr(
        uploader.guarded_kaggle, "_list_submissions", lambda *_args: submissions
    )
    monkeypatch.setattr(
        uploader.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("guarded wait must not submit"),
    )

    result = _process(tmp_path, upload=True)

    assert result["status"] == expected
    attempt_path, upload_path, _lock = uploader._receipt_paths(
        tmp_path / "receipts", binding
    )
    assert not attempt_path.exists()
    assert not upload_path.exists()


def test_r241_uploader_reconciles_the_exact_label_without_a_second_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding(tmp_path)
    _patch_binding(monkeypatch, binding)
    monkeypatch.setattr(uploader, "_now", lambda: NOW)
    monkeypatch.setattr(
        uploader.guarded_kaggle,
        "_list_submissions",
        lambda *_args: [
            {
                "ref": "55409999",
                "date": "2026-08-10T17:00:00+00:00",
                "description": binding.label,
                "status": "SubmissionStatus.COMPLETE",
            }
        ],
    )
    monkeypatch.setattr(
        uploader.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("reconciled label must not submit"),
    )

    result = _process(tmp_path, upload=True)

    assert result["status"] == "reconciled_existing_remote_submission"
    upload_path = Path(str(result["upload_receipt"]["path"]))
    receipt = json.loads(upload_path.read_text(encoding="utf-8"))
    assert receipt["remote_submission_id"] == 55409999
    assert receipt["attempt_receipt"] is None


def test_r241_handoff_accepts_only_the_exact_r241_queue_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write(tmp_path / "package.tar.gz", b"direct archive")
    authorization = _write(tmp_path / "authorization.json", b"{\"a\":1}\n")
    finalizer_receipt = _write(tmp_path / "finalizer.json", b"{\"f\":1}\n")
    contract = _write(tmp_path / "contract.json", b"{\"c\":1}\n")
    entry = {
        "sequence": 1,
        "single_use_nonce": "sha256:" + "b" * 64,
        "turn_order_preference": "first_if_allowed",
        "maximum_uses": 1,
        "remaining_uses": 1,
        "direct_policy_only": True,
        "authorization": _identity(authorization),
        "finalizer_receipt": _identity(finalizer_receipt),
        "archive": _identity(archive),
    }
    handoff = {
        "authorization": _identity(authorization),
        "finalizer_receipt": _identity(finalizer_receipt),
        "entry": entry,
    }
    queue = tmp_path / "direct-queue.json"
    queue.write_text(
        json.dumps(queue_processor._queue_payload(entry)), encoding="utf-8"
    )
    monkeypatch.setattr(
        uploader,
        "_validate_canonical_contract",
        lambda *_args, **_kwargs: _identity(contract),
    )
    monkeypatch.setattr(
        queue_processor,
        "validate_authorized_handoff",
        lambda **_kwargs: handoff,
    )

    binding = uploader.validate_current_handoff(
        queue_path=queue,
        authorization_path=authorization,
        finalizer_receipt_path=finalizer_receipt,
        contract_path=contract,
        allow_noncanonical_contract_for_test=True,
    )

    assert binding.label.endswith(str(entry["archive"]["sha256"])[7:19])
    bad = json.loads(queue.read_text(encoding="utf-8"))
    bad["schema"] = "poke_bot.kaggle_submission_queue/v1"
    queue.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(uploader.R241DirectUploaderError, match="only the exact"):
        uploader.validate_current_handoff(
            queue_path=queue,
            authorization_path=authorization,
            finalizer_receipt_path=finalizer_receipt,
            contract_path=contract,
            allow_noncanonical_contract_for_test=True,
        )


def test_r241_uploader_source_does_not_delegate_to_the_legacy_queue_processor() -> None:
    source = Path(uploader.__file__).read_text(encoding="utf-8")
    assert "SUBMISSION_QUEUE_SCHEMA" not in source
    assert "guarded_kaggle.process_once" not in source
    assert "--upload" in source


def test_r241_uploader_oneshot_exit_codes_keep_waits_and_failures_distinct() -> None:
    assert uploader.exit_code_for_result({"status": "submitted"}) == 0
    assert (
        uploader.exit_code_for_result(
            {
                "status": "already_recorded_no_second_upload",
                "recorded_status": "submitted",
            }
        )
        == 0
    )
    assert uploader.exit_code_for_result({"status": "spacing_wait"}) == 75
    assert (
        uploader.exit_code_for_result(
            {"status": "kaggle_timeout_unknown_no_second_attempt"}
        )
        == 2
    )
