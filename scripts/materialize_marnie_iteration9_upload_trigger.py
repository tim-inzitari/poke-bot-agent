#!/usr/bin/env python3
"""Materialize revision-130's immutable Marnie iteration-9 upload trigger.

This is intentionally inert until all local, receipt-backed evidence exists.
An absent or non-COMPLETE queue row is a normal not-ready state and never
creates a trigger.  Existing triggers are validated and left byte-for-byte
unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from poke_bot.archetype_family_activation import (
    TRIGGER_SCHEMA,
    FamilyActivationError,
    validate_iteration9_upload_trigger,
)


MILESTONE_SCHEMA = "poke_bot.final_format_milestone_submission/v1"
QUEUE_SCHEMA = "poke_bot.kaggle_submission_queue/v1"
SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
ITERATION = 9
ATTEMPT_SCHEMA = "poke_bot.kaggle_submission_attempt/v1"
AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"

QUEUE_IDENTITY_FIELDS = (
    "specialist_id",
    "copy_number",
    "turn_order_preference",
    "label",
    "checkpoint_checksum",
    "model_checksum",
    "deck_file_checksum",
    "deck_cards_checksum",
    "representatives_checksum",
    "matchup_tree_checksum",
    "search_config_checksum",
    "belief_decks_checksum",
    "gate_id",
    "iteration",
    "competition",
    "file",
    "file_sha256",
)


class TriggerMaterializationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TriggerMaterializationError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise TriggerMaterializationError(f"{label} is not a JSON object")
    return value


def _verified_file(path_value: Any, digest_value: Any, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    expected = str(digest_value or "")
    if not path.is_file() or not expected.startswith("sha256:") or sha256(path) != expected:
        raise TriggerMaterializationError(f"{label} is absent or digest-mismatched")
    return path


def _queue_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in QUEUE_IDENTITY_FIELDS}


def _accepted_row(
    queue: Mapping[str, Any], milestone_row: Mapping[str, Any]
) -> dict[str, Any] | None:
    if queue.get("schema") != QUEUE_SCHEMA:
        raise TriggerMaterializationError("Kaggle queue schema changed")
    expected = _queue_identity(milestone_row)
    base = (
        expected["specialist_id"],
        expected["copy_number"],
        expected["checkpoint_checksum"],
    )
    candidates = [
        dict(row)
        for row in (queue.get("queue") or [])
        if isinstance(row, dict)
        and (
            row.get("specialist_id"),
            row.get("copy_number"),
            row.get("checkpoint_checksum"),
        )
        == base
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise TriggerMaterializationError("iteration-9 queue identity is ambiguous")
    row = candidates[0]
    if _queue_identity(row) != expected:
        raise TriggerMaterializationError(
            "accepted queue row differs from the immutable milestone identity"
        )
    status = str(row.get("kaggle_status") or "").rsplit(".", 1)[-1].upper()
    if row.get("queue_status") != "accepted" or status != "COMPLETE":
        return None
    if row.get("failure_reason") is not None or row.get("submission_id") is None:
        raise TriggerMaterializationError("accepted COMPLETE row lacks terminal success identity")
    return row


def _attempt_evidence(
    receipts_dir: Path,
    *,
    nonce: str,
    uploaded_file: Path,
    uploaded_digest: str,
    checkpoint_digest: str,
    competition: str,
    label: str,
) -> tuple[Path, Path]:
    attempts: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(receipts_dir.glob("*.json")):
        payload = _read_object(path, "Kaggle receipt")
        if payload.get("schema") == ATTEMPT_SCHEMA and payload.get("nonce") == nonce:
            attempts.append((path.resolve(), payload))
    if len(attempts) != 1:
        raise TriggerMaterializationError(
            "one-shot nonce does not have exactly one immutable Kaggle attempt"
        )
    attempt_path, attempt = attempts[0]
    if int(attempt.get("returncode", -1)) != 0:
        raise TriggerMaterializationError("iteration-9 Kaggle attempt did not succeed")
    auth_path = Path(str(attempt.get("authorization_consumed") or "")).expanduser().resolve()
    auth = _read_object(auth_path, "consumed one-shot authorization")
    if (
        auth.get("schema") != AUTH_SCHEMA
        or auth.get("nonce") != nonce
        or auth.get("consumed_before_upload") is not True
        or int(auth.get("remaining_uses", -1)) != 0
    ):
        raise TriggerMaterializationError("one-shot authorization was not consumed before upload")
    identity = attempt.get("identity") or {}
    if (
        Path(str(identity.get("file") or "")).expanduser().resolve() != uploaded_file
        or identity.get("file_sha256") != uploaded_digest
        or identity.get("competition") != competition
        or identity.get("message") != label
        or auth.get("submission_file_checksum") != uploaded_digest
        or auth.get("frozen_checkpoint_checksum") != checkpoint_digest
    ):
        raise TriggerMaterializationError("attempt or authorization identity differs from queue")
    return attempt_path, auth_path


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _existing_result(output: Path, milestone_receipt: Path) -> dict[str, Any]:
    existing = _read_object(output, "existing iteration-9 upload trigger")
    try:
        validate_iteration9_upload_trigger(existing)
    except (FamilyActivationError, OSError, json.JSONDecodeError) as exc:
        raise TriggerMaterializationError("existing iteration-9 trigger is invalid") from exc
    source = existing.get("source_evidence") or {}
    milestone = source.get("milestone_receipt") or {}
    if (
        Path(str(milestone.get("path") or "")).expanduser().resolve()
        != milestone_receipt.resolve()
        or milestone.get("sha256") != sha256(milestone_receipt)
    ):
        raise TriggerMaterializationError("existing trigger binds different milestone evidence")
    return {"status": "already_materialized", "trigger": str(output.resolve())}


def materialize_trigger(
    *,
    queue_path: Path,
    milestone_receipt: Path,
    attempt_receipts: Path,
    output: Path,
) -> dict[str, Any]:
    """Create the trigger once, only after exact local success evidence exists."""
    queue_path = Path(queue_path).expanduser().resolve()
    milestone_receipt = Path(milestone_receipt).expanduser().resolve()
    attempt_receipts = Path(attempt_receipts).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not milestone_receipt.is_file() or not queue_path.is_file():
        return {"status": "not_ready", "reason": "milestone_or_queue_absent"}

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as output_lock:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return _existing_result(output, milestone_receipt)

        milestone = _read_object(milestone_receipt, "iteration-9 milestone receipt")
        if (
            milestone.get("schema") != MILESTONE_SCHEMA
            or milestone.get("status") != "queued"
            or milestone.get("specialist_id", SPECIALIST_ID) != SPECIALIST_ID
            or int(milestone.get("iteration", -1)) != ITERATION
        ):
            raise TriggerMaterializationError("wrong Marnie iteration-9 milestone receipt")
        milestone_queue = milestone.get("queue_entry") or {}
        if (
            milestone_queue.get("specialist_id") != SPECIALIST_ID
            or int(milestone_queue.get("iteration", -1)) != ITERATION
            or int(milestone_queue.get("copy_number", -1)) != 1
        ):
            raise TriggerMaterializationError("milestone queue identity is not exact Marnie iter-9")

        queue_lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
        with queue_lock_path.open("a+", encoding="utf-8") as queue_lock:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_SH)
            queue = _read_object(queue_path, "Kaggle submission queue")
        row = _accepted_row(queue, milestone_queue)
        if row is None:
            return {"status": "not_ready", "reason": "exact_upload_not_complete"}

        commit = _verified_file(
            milestone.get("commit"), milestone.get("commit_sha256"), "milestone commit"
        )
        checkpoint = _verified_file(
            milestone.get("checkpoint"),
            milestone.get("checkpoint_sha256"),
            "iteration-9 checkpoint",
        )
        bundle = milestone.get("bundle") or {}
        package = _verified_file(bundle.get("path"), bundle.get("sha256"), "package bundle")
        deck = bundle.get("deck") or {}
        representative = _verified_file(
            deck.get("path"), deck.get("file_sha256"), "representative deck"
        )
        uploaded = _verified_file(row.get("file"), row.get("file_sha256"), "uploaded file")
        if (
            bundle.get("specialist_id") != SPECIALIST_ID
            or bundle.get("checkpoint_archetype_id") != SPECIALIST_ID
            or bundle.get("sha256") != row.get("file_sha256")
            or (bundle.get("contents") or {}).get("model_sha256")
            != milestone.get("checkpoint_sha256")
            or deck.get("file_sha256") != row.get("deck_file_checksum")
            or deck.get("cards_sha256") != row.get("deck_cards_checksum")
            or deck.get("representatives_sha256") != row.get("representatives_checksum")
        ):
            raise TriggerMaterializationError("milestone bundle identity differs from queue")
        nonce = str(row.get("one_shot_authorization_nonce") or "")
        if not nonce:
            raise TriggerMaterializationError("accepted row lacks its one-shot nonce")
        attempt_path, auth_path = _attempt_evidence(
            attempt_receipts,
            nonce=nonce,
            uploaded_file=uploaded,
            uploaded_digest=str(row["file_sha256"]),
            checkpoint_digest=str(milestone["checkpoint_sha256"]),
            competition=str(row["competition"]),
            label=str(row["label"]),
        )
        trigger = {
            "schema": TRIGGER_SCHEMA,
            "specialist_id": SPECIALIST_ID,
            "iteration": ITERATION,
            "owner_revision": 130,
            "bindings": {
                "commit": {"path": str(commit), "sha256": sha256(commit)},
                "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
                "package_bundle": {"path": str(package), "sha256": sha256(package)},
                "representative_deck": {
                    "path": str(representative),
                    "sha256": sha256(representative),
                },
                "competition": str(row["competition"]),
                "submission_label": str(row["label"]),
                "uploaded_file": {"path": str(uploaded), "sha256": sha256(uploaded)},
            },
            "attempt": {"path": str(attempt_path), "sha256": sha256(attempt_path)},
            "consumed_authorization": {"path": str(auth_path), "sha256": sha256(auth_path)},
            "source_evidence": {
                "milestone_receipt": {
                    "path": str(milestone_receipt),
                    "sha256": sha256(milestone_receipt),
                },
                "accepted_complete_queue_row": row,
                "accepted_complete_queue_row_sha256": canonical_digest(row),
            },
        }
        try:
            validate_iteration9_upload_trigger(trigger)
        except (FamilyActivationError, OSError, json.JSONDecodeError) as exc:
            raise TriggerMaterializationError("constructed trigger failed validation") from exc
        _write_exclusive(output, trigger)
        return {"status": "materialized", "trigger": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--milestone-receipt", type=Path, required=True)
    parser.add_argument("--attempt-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_trigger(
        queue_path=args.queue,
        milestone_receipt=args.milestone_receipt,
        attempt_receipts=args.attempt_receipts,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
