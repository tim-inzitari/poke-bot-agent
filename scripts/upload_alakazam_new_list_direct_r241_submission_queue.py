#!/usr/bin/env python3
"""Upload the one receipt-bound r241 direct-policy terminal package.

This is intentionally separate from the generic frozen-specialist uploader.
It accepts only the r241 direct-policy queue emitted by
``process_alakazam_new_list_direct_r241_submission_queue.py`` and reopens the
canonical terminal handoff before doing anything external.  The queue itself
stays immutable: a deterministic attempt receipt is committed before the CLI
can run and a deterministic outcome receipt is committed afterward.  A crash
after the attempt receipt is therefore fail-closed rather than creating a
second possible upload.

No external command is run unless ``--upload`` is explicit.  The default is a
fully local preflight suitable for the terminal-finalizer dependency chain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from scripts import finalize_alakazam_new_list_direct_r241 as finalizer
from scripts import process_alakazam_new_list_direct_r241_submission_queue as queue_processor
from scripts import process_kaggle_submission_queue as guarded_kaggle


QUEUE_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241_submission_queue/v1"
UPLOADER_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241_submission_uploader/v1"
ATTEMPT_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_submission_attempt/v1"
)
UPLOAD_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_submission_upload/v1"
)

COMPETITION = "pokemon-tcg-ai-battle"
TURN_ORDER_PREFERENCE = "first_if_allowed"
DAILY_SUBMISSION_LIMIT = 5
MINIMUM_SUBMISSION_SPACING_HOURS = 4
KAGGLE_TIMEOUT_SECONDS = 900
TEMPORARILY_DEFERRED_EXIT_CODE = 75

_SUCCESSFUL_UPLOAD_STATUSES = {
    "submitted",
    "reconciled_existing_remote_submission",
}
_TERMINAL_FAILURE_STATUSES = {
    "remote_terminal_result_no_second_attempt",
    "kaggle_quota_rejected_no_second_attempt",
    "kaggle_rejected_no_second_attempt",
    "kaggle_timeout_unknown_no_second_attempt",
    "kaggle_start_failure_no_second_attempt",
    "prior_attempt_outcome_unknown_no_second_upload",
}


class R241DirectUploaderError(RuntimeError):
    """The one r241 terminal upload cannot safely proceed."""


@dataclass(frozen=True)
class DirectUploadBinding:
    """All immutable facts that identify the one allowable upload."""

    owner_contract: Mapping[str, object]
    queue: Mapping[str, object]
    authorization: Mapping[str, object]
    finalizer_receipt: Mapping[str, object]
    entry: Mapping[str, object]
    archive: Mapping[str, object]
    label: str

    @property
    def queue_entry_sha256(self) -> str:
        return _canonical_digest(self.entry)

    def as_dict(self) -> dict[str, object]:
        return {
            "owner_contract": dict(self.owner_contract),
            "queue": dict(self.queue),
            "authorization": dict(self.authorization),
            "finalizer_receipt": dict(self.finalizer_receipt),
            "queue_entry_sha256": self.queue_entry_sha256,
            "archive": dict(self.archive),
            "competition": COMPETITION,
            "turn_order_preference": TURN_ORDER_PREFERENCE,
            "submission_label": self.label,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241DirectUploaderError(f"{label} must be a regular non-symlink file")
    return raw.resolve()


def _directory(path: Path | str, *, label: str, create: bool = False) -> Path:
    raw = Path(path).expanduser()
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    if raw.is_symlink() or not raw.is_dir():
        raise R241DirectUploaderError(f"{label} must be a real directory")
    return raw.resolve()


def _read_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241DirectUploaderError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R241DirectUploaderError(f"{label} must be a JSON object")
    return source, value


def _identity(path: Path | str, *, label: str) -> dict[str, object]:
    source = _regular_file(path, label=label)
    return {
        "path": str(source),
        "sha256": finalizer.sha256_file(source),
        "size_bytes": int(source.stat().st_size),
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return finalizer.canonical_json(dict(payload))
    except finalizer.R241FinalizerError as exc:
        raise R241DirectUploaderError("r241 receipt is not canonical JSON") from exc


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return finalizer.sha256_bytes(_canonical_json(payload))


def _identity_matches(
    observed: object,
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise R241DirectUploaderError(f"{label} is not a file identity")
    expected_path = str(expected.get("path") or "")
    expected_sha256 = str(expected.get("sha256") or "")
    try:
        expected_size = int(expected.get("size_bytes", -1))
        observed_size = int(observed.get("size_bytes", -2))
    except (TypeError, ValueError) as exc:
        raise R241DirectUploaderError(f"{label} size is invalid") from exc
    if (
        str(observed.get("sha256") or "") != expected_sha256
        or observed_size != expected_size
        or str(observed.get("path") or "") != expected_path
    ):
        raise R241DirectUploaderError(f"{label} identity drifted")


def _submission_label(entry: Mapping[str, object]) -> str:
    """Return a stable, checksum-distinct label with no caller override."""

    archive = dict(entry.get("archive") or {})
    digest = str(archive.get("sha256") or "")
    if not finalizer._SHA256.fullmatch(digest):
        raise R241DirectUploaderError("r241 archive digest cannot form a submission label")
    return f"ALAKAZAM R241 DIRECT T10 FIRST IF ALLOWED {digest[7:19]}"


def _validate_canonical_contract(
    contract_path: Path,
    *,
    allow_noncanonical_contract_for_test: bool,
) -> dict[str, object]:
    source = _regular_file(contract_path, label="r241 typed contract")
    identity = _identity(source, label="r241 typed contract")
    if not allow_noncanonical_contract_for_test:
        if source != finalizer.CONTRACT_PATH.resolve():
            raise R241DirectUploaderError(
                "production uploader requires the canonical r241 typed contract"
            )
        if identity["sha256"] != finalizer.CONTRACT_SHA256:
            raise R241DirectUploaderError(
                "canonical r241 typed contract changed without uploader revision"
            )
    return identity


def validate_current_handoff(
    *,
    queue_path: Path,
    authorization_path: Path,
    finalizer_receipt_path: Path,
    contract_path: Path = finalizer.CONTRACT_PATH,
    official_libcg_staging_path: Path = finalizer.OFFICIAL_LIBCG_STAGING_PATH,
    allow_noncanonical_contract_for_test: bool = False,
) -> DirectUploadBinding:
    """Revalidate the sole queue item against current terminal evidence."""

    contract_identity = _validate_canonical_contract(
        contract_path,
        allow_noncanonical_contract_for_test=allow_noncanonical_contract_for_test,
    )
    try:
        handoff = queue_processor.validate_authorized_handoff(
            authorization_path=authorization_path,
            finalizer_receipt_path=finalizer_receipt_path,
            contract_path=contract_path,
            official_libcg_staging_path=official_libcg_staging_path,
        )
    except queue_processor.R241QueueProcessorError as exc:
        raise R241DirectUploaderError(
            "r241 terminal authorization/archive revalidation failed"
        ) from exc

    queue_file, queue = _read_object(queue_path, label="r241 direct-policy queue")
    queue_identity = _identity(queue_file, label="r241 direct-policy queue")
    entry = dict(handoff.get("entry") or {})
    expected_queue = queue_processor._queue_payload(entry)
    if queue.get("schema") != QUEUE_SCHEMA or queue != expected_queue:
        raise R241DirectUploaderError(
            "r241 uploader accepts only the exact unopened direct-policy queue"
        )
    if (
        queue.get("competition") != COMPETITION
        or queue.get("turn_order_preference") != TURN_ORDER_PREFERENCE
        or queue.get("submission_count_authorized") != 1
        or queue.get("submission_count_enqueued") != 1
        or queue.get("submission_count_performed") != 0
        or len(queue.get("queue") or []) != 1
    ):
        raise R241DirectUploaderError("r241 queue single-upload boundary drifted")

    authorization_identity = dict(handoff.get("authorization") or {})
    finalizer_identity = dict(handoff.get("finalizer_receipt") or {})
    if not authorization_identity or not finalizer_identity:
        raise R241DirectUploaderError("r241 terminal handoff has no immutable identities")
    _identity_matches(
        entry.get("authorization"),
        authorization_identity,
        label="r241 queue authorization",
    )
    _identity_matches(
        entry.get("finalizer_receipt"),
        finalizer_identity,
        label="r241 queue finalizer receipt",
    )
    archive = dict(entry.get("archive") or {})
    archive_path = _regular_file(
        str(archive.get("path") or ""), label="r241 queued archive"
    )
    _identity_matches(
        archive,
        _identity(archive_path, label="r241 queued archive"),
        label="r241 queued archive",
    )
    if (
        entry.get("sequence") != 1
        or entry.get("single_use_nonce") is None
        or entry.get("turn_order_preference") != TURN_ORDER_PREFERENCE
        or entry.get("maximum_uses") != 1
        or entry.get("remaining_uses") != 1
        or entry.get("direct_policy_only") is not True
    ):
        raise R241DirectUploaderError("r241 queue entry no longer represents one direct upload")
    return DirectUploadBinding(
        owner_contract=contract_identity,
        queue=queue_identity,
        authorization=authorization_identity,
        finalizer_receipt=finalizer_identity,
        entry=entry,
        archive=archive,
        label=_submission_label(entry),
    )


def _receipt_paths(receipts_dir: Path, binding: DirectUploadBinding) -> tuple[Path, Path, Path]:
    nonce = str(binding.entry.get("single_use_nonce") or "")
    if not finalizer._SHA256.fullmatch(nonce):
        raise R241DirectUploaderError("r241 single-use nonce is invalid")
    suffix = nonce.removeprefix("sha256:")
    base = receipts_dir / f"r241-direct-policy-{suffix}"
    return (
        base.with_name(base.name + "-attempt.json"),
        base.with_name(base.name + "-upload.json"),
        base.with_name("." + base.name + ".lock"),
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    parent = _directory(path.parent, label=f"{label} parent", create=True)
    target = parent / path.name
    body = _canonical_json(payload)
    if target.exists() or target.is_symlink():
        existing = _regular_file(target, label=label)
        if existing.read_bytes() != body:
            raise R241DirectUploaderError(
                f"immutable {label} already exists with different bytes"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.r241-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _regular_file(target, label=label)
            if existing.read_bytes() != body:
                raise R241DirectUploaderError(
                    f"immutable {label} raced with different bytes"
                )
        else:
            os.chmod(target, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _binding_matches_receipt(
    receipt: Mapping[str, Any],
    binding: DirectUploadBinding,
    *,
    label: str,
) -> None:
    if (
        int(receipt.get("owner_decision_revision", -1)) != finalizer.OWNER_REVISION
        or receipt.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or receipt.get("competition") != COMPETITION
        or receipt.get("turn_order_preference") != TURN_ORDER_PREFERENCE
        or receipt.get("submission_label") != binding.label
        or receipt.get("queue_entry_sha256") != binding.queue_entry_sha256
    ):
        raise R241DirectUploaderError(f"{label} does not bind the current r241 upload")
    _identity_matches(
        receipt.get("owner_contract"), binding.owner_contract, label=f"{label} contract"
    )
    _identity_matches(receipt.get("queue"), binding.queue, label=f"{label} queue")
    _identity_matches(
        receipt.get("authorization"), binding.authorization, label=f"{label} authorization"
    )
    _identity_matches(
        receipt.get("finalizer_receipt"),
        binding.finalizer_receipt,
        label=f"{label} finalizer receipt",
    )
    _identity_matches(receipt.get("archive"), binding.archive, label=f"{label} archive")


def _attempt_payload(binding: DirectUploadBinding, *, created_at_utc: str) -> dict[str, object]:
    return {
        "schema": ATTEMPT_RECEIPT_SCHEMA,
        "owner_decision_revision": finalizer.OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "upload_command_reserved_no_second_attempt",
        "created_at_utc": created_at_utc,
        **binding.as_dict(),
        "upload_command": {
            "argv_shape": "kaggle competitions submit -c COMPETITION -f ARCHIVE -m LABEL",
            "timeout_seconds": KAGGLE_TIMEOUT_SECONDS,
        },
        "outcome_receipt_present": False,
    }


def _validate_attempt_receipt(path: Path, binding: DirectUploadBinding) -> dict[str, Any]:
    _source, receipt = _read_object(path, label="r241 upload attempt receipt")
    if (
        receipt.get("schema") != ATTEMPT_RECEIPT_SCHEMA
        or receipt.get("status") != "upload_command_reserved_no_second_attempt"
        or receipt.get("outcome_receipt_present") is not False
    ):
        raise R241DirectUploaderError("r241 upload attempt receipt has an invalid schema")
    _binding_matches_receipt(receipt, binding, label="r241 upload attempt receipt")
    return receipt


def _validate_upload_receipt(path: Path, binding: DirectUploadBinding) -> dict[str, Any]:
    _source, receipt = _read_object(path, label="r241 upload receipt")
    outcomes = {
        "submitted",
        "reconciled_existing_remote_submission",
        "remote_terminal_result_no_second_attempt",
        "kaggle_quota_rejected_no_second_attempt",
        "kaggle_rejected_no_second_attempt",
        "kaggle_timeout_unknown_no_second_attempt",
        "kaggle_start_failure_no_second_attempt",
    }
    if receipt.get("schema") != UPLOAD_RECEIPT_SCHEMA or receipt.get("status") not in outcomes:
        raise R241DirectUploaderError("r241 upload receipt has an invalid schema")
    _binding_matches_receipt(receipt, binding, label="r241 upload receipt")
    if receipt.get("no_second_attempt_allowed") is not True:
        raise R241DirectUploaderError("r241 upload receipt permits another attempt")
    remote_only = {
        "reconciled_existing_remote_submission",
        "remote_terminal_result_no_second_attempt",
    }
    attempt_reference = receipt.get("attempt_receipt")
    if receipt["status"] in remote_only:
        if attempt_reference is not None or receipt.get("kaggle_cli_invoked") is not False:
            raise R241DirectUploaderError(
                "reconciled r241 upload receipt claims a local command"
            )
    else:
        if not isinstance(attempt_reference, Mapping):
            raise R241DirectUploaderError("r241 upload receipt lacks its attempt identity")
        attempt_path = _regular_file(
            str(attempt_reference.get("path") or ""),
            label="r241 upload receipt attempt",
        )
        _identity_matches(
            attempt_reference,
            _identity(attempt_path, label="r241 upload receipt attempt"),
            label="r241 upload receipt attempt",
        )
        _validate_attempt_receipt(attempt_path, binding)
        expected_invocation = receipt["status"] != "kaggle_start_failure_no_second_attempt"
        if receipt.get("kaggle_cli_invoked") is not expected_invocation:
            raise R241DirectUploaderError(
                "r241 command-result receipt misstates command invocation"
            )
    return receipt


def _submission_id(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        result = int(text)
    except ValueError:
        return None
    return result if result > 0 else None


def _remote_label_match(
    submissions: Sequence[Mapping[str, object]], label: str
) -> Mapping[str, object] | None:
    matches = [
        row for row in submissions if str(row.get("description") or "") == label
    ]
    if len(matches) > 1:
        raise R241DirectUploaderError(
            "multiple Kaggle rows share the r241 checksum-bound submission label"
        )
    return matches[0] if matches else None


def _quota_and_spacing_guard(
    submissions: Sequence[Mapping[str, object]], *, now: datetime
) -> dict[str, object] | None:
    quota_date = now.date().isoformat()
    used = sum(
        1
        for row in submissions
        if str(row.get("date") or "").strip().startswith(quota_date)
    )
    if used >= DAILY_SUBMISSION_LIMIT:
        return {
            "status": "quota_wait",
            "used": used,
            "limit": DAILY_SUBMISSION_LIMIT,
            "quota_date": quota_date,
        }
    # This exact helper is the established guarded policy: it deduplicates
    # logical remote submissions and anchors spacing at the second newest one.
    times = guarded_kaggle._submission_times(  # noqa: SLF001
        [dict(row) for row in submissions], []
    )
    anchor = times[1] if len(times) >= 2 else None
    if anchor is not None:
        eligible_at = anchor + timedelta(hours=MINIMUM_SUBMISSION_SPACING_HOURS)
        if now < eligible_at:
            return {
                "status": "spacing_wait",
                "next_submission_eligible_at": eligible_at.isoformat(),
                "remaining_seconds": int((eligible_at - now).total_seconds()),
                "spacing_anchor_policy": "second_most_recent_logical_submission",
                "spacing_anchor_submission_at": anchor.isoformat(),
            }
    return None


def _remote_receipt_payload(
    binding: DirectUploadBinding,
    *,
    status: str,
    created_at_utc: str,
    remote: Mapping[str, object] | None,
    attempt_receipt: Mapping[str, object] | None,
    kaggle_cli_invoked: bool,
    returncode: int | None,
    output: str,
) -> dict[str, object]:
    output_bytes = output.encode("utf-8", errors="replace")
    return {
        "schema": UPLOAD_RECEIPT_SCHEMA,
        "owner_decision_revision": finalizer.OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": status,
        "recorded_at_utc": created_at_utc,
        **binding.as_dict(),
        "attempt_receipt": dict(attempt_receipt) if attempt_receipt is not None else None,
        "kaggle_cli_invoked": kaggle_cli_invoked,
        "kaggle_cli_returncode": returncode,
        "remote_submission_id": _submission_id(
            None if remote is None else remote.get("ref")
        ),
        "remote_submission_status": (
            None
            if remote is None
            else guarded_kaggle._status_name(str(remote.get("status") or ""))  # noqa: SLF001
        ),
        "remote_submitted_at": None if remote is None else remote.get("date"),
        "cli_output_sha256": finalizer.sha256_bytes(output_bytes),
        "cli_output_tail": output[-2000:],
        "no_second_attempt_allowed": True,
    }


def _record_existing_remote_submission(
    *,
    upload_path: Path,
    binding: DirectUploadBinding,
    remote: Mapping[str, object],
    now: datetime,
) -> dict[str, object]:
    remote_status = guarded_kaggle._status_name(str(remote.get("status") or ""))  # noqa: SLF001
    status = (
        "reconciled_existing_remote_submission"
        if remote_status not in guarded_kaggle.TERMINAL_KAGGLE_FAILURES
        else "remote_terminal_result_no_second_attempt"
    )
    payload = _remote_receipt_payload(
        binding,
        status=status,
        created_at_utc=now.isoformat(),
        remote=remote,
        attempt_receipt=None,
        kaggle_cli_invoked=False,
        returncode=None,
        output="remote label reconciliation before local command",
    )
    _write_immutable_json(upload_path, payload, label="r241 reconciled upload receipt")
    return {
        "schema": UPLOADER_SCHEMA,
        "status": status,
        "upload_receipt": _identity(upload_path, label="r241 reconciled upload receipt"),
        "submission_label": binding.label,
    }


def _run_upload(
    *,
    binding: DirectUploadBinding,
    kaggle: Path,
    attempt_path: Path,
    upload_path: Path,
    now: datetime,
) -> dict[str, object]:
    attempt = _attempt_payload(binding, created_at_utc=now.isoformat())
    _write_immutable_json(attempt_path, attempt, label="r241 upload attempt receipt")
    attempt_identity = _identity(attempt_path, label="r241 upload attempt receipt")
    command = [
        str(kaggle),
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(binding.archive["path"]),
        "-m",
        binding.label,
    ]
    returncode: int | None = None
    output = ""
    status: str
    cli_started = False
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=KAGGLE_TIMEOUT_SECONDS,
        )
        cli_started = True
        returncode = int(completed.returncode)
        output = "\n".join((completed.stdout or "", completed.stderr or "")).strip()
        if returncode == 0:
            status = "submitted"
        elif guarded_kaggle._quota_error(output):  # noqa: SLF001
            status = "kaggle_quota_rejected_no_second_attempt"
        else:
            status = "kaggle_rejected_no_second_attempt"
    except subprocess.TimeoutExpired as exc:
        cli_started = True
        output = "\n".join(
            part
            for part in (
                str(exc.stdout or ""),
                str(exc.stderr or ""),
                "Kaggle CLI timeout; outcome intentionally treated as unknown.",
            )
            if part
        )
        status = "kaggle_timeout_unknown_no_second_attempt"
    except OSError as exc:
        output = f"Kaggle CLI could not be started: {exc}"
        status = "kaggle_start_failure_no_second_attempt"
    payload = _remote_receipt_payload(
        binding,
        status=status,
        created_at_utc=_now().isoformat(),
        remote=None,
        attempt_receipt=attempt_identity,
        kaggle_cli_invoked=cli_started,
        returncode=returncode,
        output=output,
    )
    _write_immutable_json(upload_path, payload, label="r241 upload receipt")
    return {
        "schema": UPLOADER_SCHEMA,
        "status": status,
        "submission_label": binding.label,
        "attempt_receipt": attempt_identity,
        "upload_receipt": _identity(upload_path, label="r241 upload receipt"),
    }


def process_once(
    *,
    queue_path: Path,
    authorization_path: Path,
    finalizer_receipt_path: Path,
    receipts_dir: Path,
    kaggle: Path,
    contract_path: Path = finalizer.CONTRACT_PATH,
    official_libcg_staging_path: Path = finalizer.OFFICIAL_LIBCG_STAGING_PATH,
    upload: bool = False,
    allow_noncanonical_contract_for_test: bool = False,
) -> dict[str, object]:
    """Preflight or perform the one direct-policy upload exactly once."""

    binding = validate_current_handoff(
        queue_path=queue_path,
        authorization_path=authorization_path,
        finalizer_receipt_path=finalizer_receipt_path,
        contract_path=contract_path,
        official_libcg_staging_path=official_libcg_staging_path,
        allow_noncanonical_contract_for_test=allow_noncanonical_contract_for_test,
    )
    raw_receipts = Path(receipts_dir).expanduser()
    if raw_receipts.exists() or raw_receipts.is_symlink():
        receipt_root = _directory(raw_receipts, label="r241 upload receipts")
    else:
        receipt_root = raw_receipts.absolute()
        if upload:
            receipt_root = _directory(
                receipt_root, label="r241 upload receipts", create=True
            )
    attempt_path, upload_path, lock_path = _receipt_paths(receipt_root, binding)

    if not upload:
        if upload_path.exists() or upload_path.is_symlink():
            receipt = _validate_upload_receipt(upload_path, binding)
            return {
                "schema": UPLOADER_SCHEMA,
                "status": "already_recorded_no_network",
                "submission_label": binding.label,
                "upload_receipt": _identity(upload_path, label="r241 upload receipt"),
                "recorded_status": receipt["status"],
            }
        if attempt_path.exists() or attempt_path.is_symlink():
            _validate_attempt_receipt(attempt_path, binding)
            return {
                "schema": UPLOADER_SCHEMA,
                "status": "prior_attempt_outcome_unknown_no_network",
                "submission_label": binding.label,
                "attempt_receipt": _identity(
                    attempt_path, label="r241 upload attempt receipt"
                ),
            }
        return {
            "schema": UPLOADER_SCHEMA,
            "status": "local_preflight_passed_upload_not_requested",
            "submission_label": binding.label,
            "competition": COMPETITION,
            "turn_order_preference": TURN_ORDER_PREFERENCE,
            "queue_entry_sha256": binding.queue_entry_sha256,
            "network_io_performed": False,
        }

    # A single lock covers receipt observation, remote reconciliation, intent
    # reservation, and the command invocation.  It is intentionally a narrow
    # isolated lock, not the generic queue daemon's shared lock.
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if upload_path.exists() or upload_path.is_symlink():
                receipt = _validate_upload_receipt(upload_path, binding)
                return {
                    "schema": UPLOADER_SCHEMA,
                    "status": "already_recorded_no_second_upload",
                    "submission_label": binding.label,
                    "upload_receipt": _identity(upload_path, label="r241 upload receipt"),
                    "recorded_status": receipt["status"],
                }
            if attempt_path.exists() or attempt_path.is_symlink():
                _validate_attempt_receipt(attempt_path, binding)
                return {
                    "schema": UPLOADER_SCHEMA,
                    "status": "prior_attempt_outcome_unknown_no_second_upload",
                    "submission_label": binding.label,
                    "attempt_receipt": _identity(
                        attempt_path, label="r241 upload attempt receipt"
                    ),
                }
            submissions = guarded_kaggle._list_submissions(kaggle, COMPETITION)  # noqa: SLF001
            remote = _remote_label_match(submissions, binding.label)
            now = _now()
            if remote is not None:
                return _record_existing_remote_submission(
                    upload_path=upload_path,
                    binding=binding,
                    remote=remote,
                    now=now,
                )
            guard = _quota_and_spacing_guard(submissions, now=now)
            if guard is not None:
                return {
                    "schema": UPLOADER_SCHEMA,
                    **guard,
                    "submission_label": binding.label,
                    "network_io_performed": True,
                    "attempt_receipt_written": False,
                }
            return _run_upload(
                binding=binding,
                kaggle=kaggle,
                attempt_path=attempt_path,
                upload_path=upload_path,
                now=now,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def exit_code_for_result(result: Mapping[str, object]) -> int:
    """Keep a receipt-backed terminal failure visible to a oneshot service."""

    status = str(result.get("status") or "")
    if status in {"quota_wait", "spacing_wait"}:
        return TEMPORARILY_DEFERRED_EXIT_CODE
    if status in _TERMINAL_FAILURE_STATUSES:
        return 2
    if status in {"already_recorded_no_second_upload", "already_recorded_no_network"}:
        return (
            0
            if str(result.get("recorded_status") or "") in _SUCCESSFUL_UPLOAD_STATUSES
            else 2
        )
    if status in {
        "local_preflight_passed_upload_not_requested",
        "submitted",
        "reconciled_existing_remote_submission",
    }:
        return 0
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--finalizer-receipt", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=finalizer.CONTRACT_PATH)
    parser.add_argument(
        "--official-libcg-staging",
        type=Path,
        default=finalizer.OFFICIAL_LIBCG_STAGING_PATH,
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="allow the one guarded Kaggle CLI invocation after local preflight",
    )
    args = parser.parse_args(argv)
    try:
        result = process_once(
            queue_path=args.queue,
            authorization_path=args.authorization,
            finalizer_receipt_path=args.finalizer_receipt,
            receipts_dir=args.receipts_dir,
            kaggle=args.kaggle,
            contract_path=args.contract,
            official_libcg_staging_path=args.official_libcg_staging,
            upload=args.upload,
        )
    except R241DirectUploaderError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:  # guarded remote listing errors remain explicit
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
