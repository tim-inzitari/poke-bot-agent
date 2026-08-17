#!/usr/bin/env python3
"""Materialize the successful exact epoch-25 bootstrap upload authority."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_marnie_iteration9_upload_trigger import (  # noqa: E402
    TriggerMaterializationError,
    _accepted_row,
    _attempt_evidence,
    _read_object,
    canonical_digest,
    sha256,
)


SCHEMA = "poke_bot.marnie_postupload_bootstrap_upload/v1"
SUBMISSION_SCHEMA = "poke_bot.marnie_postupload_bootstrap_submission/v1"
SPECIALIST_ID = "marnie-s-grimmsnarl-ex"


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
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


def materialize(
    *,
    queue_path: Path,
    submission_receipt: Path,
    attempt_receipts: Path,
    output: Path,
) -> dict[str, Any]:
    queue_path = queue_path.expanduser().resolve()
    submission_receipt = submission_receipt.expanduser().resolve()
    attempt_receipts = attempt_receipts.expanduser().resolve()
    output = output.expanduser().resolve()
    if not submission_receipt.is_file() or not queue_path.is_file():
        return {"status": "not_ready", "reason": "submission_or_queue_absent"}
    lock_path = output.with_suffix(output.suffix + ".lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            existing = _read_object(output, "existing bootstrap upload receipt")
            if existing.get("schema") != SCHEMA or existing.get("status") != "successful_upload":
                raise TriggerMaterializationError(
                    "existing bootstrap upload receipt is invalid"
                )
            return {"status": "already_materialized", "receipt": str(output)}

        staged = _read_object(submission_receipt, "bootstrap submission receipt")
        queue_entry = dict(staged.get("queue_entry") or {})
        if (
            staged.get("schema") != SUBMISSION_SCHEMA
            or staged.get("status") != "queued"
            or queue_entry.get("specialist_id") != SPECIALIST_ID
            or staged.get("checkpoint_sha256")
            != queue_entry.get("checkpoint_checksum")
            or staged.get("checkpoint_sha256")
            != queue_entry.get("model_checksum")
        ):
            raise TriggerMaterializationError(
                "bootstrap submission receipt identity is invalid"
            )
        with queue_path.with_suffix(queue_path.suffix + ".lock").open(
            "a+", encoding="utf-8"
        ) as queue_lock:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_SH)
            queue = _read_object(queue_path, "Kaggle queue")
        row = _accepted_row(queue, queue_entry)
        if row is None:
            return {"status": "not_ready", "reason": "exact_upload_not_complete"}

        checkpoint = Path(str(staged.get("checkpoint") or "")).resolve()
        ready_row = dict(staged.get("ready") or {})
        ready = Path(str(ready_row.get("path") or "")).resolve()
        bundle = dict(staged.get("bundle") or {})
        uploaded = Path(str(row.get("file") or "")).resolve()
        for path, expected, label in (
            (checkpoint, staged.get("checkpoint_sha256"), "bootstrap checkpoint"),
            (ready, ready_row.get("sha256"), "bootstrap ready receipt"),
            (Path(str(bundle.get("path") or "")).resolve(), bundle.get("sha256"), "bundle"),
            (uploaded, row.get("file_sha256"), "uploaded bundle"),
        ):
            if not path.is_file() or sha256(path) != str(expected or ""):
                raise TriggerMaterializationError(f"{label} changed")
        nonce = str(row.get("one_shot_authorization_nonce") or "")
        if not nonce:
            raise TriggerMaterializationError("bootstrap queue row lacks authorization nonce")
        attempt_path, authorization_path = _attempt_evidence(
            attempt_receipts,
            nonce=nonce,
            uploaded_file=uploaded,
            uploaded_digest=str(row["file_sha256"]),
            checkpoint_digest=str(staged["checkpoint_sha256"]),
            competition=str(row["competition"]),
            label=str(row["label"]),
        )
        payload = {
            "schema": SCHEMA,
            "status": "successful_upload",
            "owner_revisions": [130, 134, 135],
            "specialist_id": SPECIALIST_ID,
            "source_submission": {
                "path": str(submission_receipt),
                "sha256": sha256(submission_receipt),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": str(staged["checkpoint_sha256"]),
            },
            "uploaded_file": {"path": str(uploaded), "sha256": sha256(uploaded)},
            "accepted_complete_queue_row": row,
            "accepted_complete_queue_row_sha256": canonical_digest(row),
            "attempt": {"path": str(attempt_path), "sha256": sha256(attempt_path)},
            "consumed_authorization": {
                "path": str(authorization_path),
                "sha256": sha256(authorization_path),
            },
            "first_new_system_self_play_authorized": True,
            "old_system_collection_authorized": False,
        }
        _write_exclusive(output, payload)
        return {"status": "materialized", "receipt": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--attempt-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        queue_path=args.queue,
        submission_receipt=args.submission_receipt,
        attempt_receipts=args.attempt_receipts,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
