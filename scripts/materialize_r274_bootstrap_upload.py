#!/usr/bin/env python3
"""Seal the exact successful r274 bootstrap upload for tactical activation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from poke_bot.r274_bootstrap_handoff import file_identity
from scripts.materialize_marnie_iteration9_upload_trigger import (
    TriggerMaterializationError,
    _accepted_row,
    _attempt_evidence,
    _read_object,
    canonical_digest,
    sha256,
)
from scripts.stage_r274_bootstrap_submission import CANDIDATE_ID, SCHEMA as STAGE_SCHEMA


SCHEMA = "poke_bot.alakazam_r274_bootstrap_upload/v1"
SPECIALIST_ID = "alakazam"


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
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
    stage_receipt: Path,
    attempt_receipts: Path,
    output: Path,
) -> dict[str, Any]:
    queue_path = queue_path.expanduser().resolve()
    stage_receipt = stage_receipt.expanduser().resolve()
    attempt_receipts = attempt_receipts.expanduser().resolve()
    output = output.expanduser().resolve()
    if not stage_receipt.is_file() or not queue_path.is_file():
        return {"status": "not_ready", "reason": "stage_or_queue_absent"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open(
        "a+", encoding="utf-8"
    ) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            existing = _read_object(output, "existing r274 upload receipt")
            if (
                existing.get("schema") != SCHEMA
                or existing.get("status") != "submitted"
                or existing.get("direct_policy_only") is not True
                or existing.get("rtp_enabled") is not False
            ):
                raise TriggerMaterializationError(
                    "existing r274 upload receipt is invalid"
                )
            return {"status": "already_materialized", "receipt": str(output)}

        staged = _read_object(stage_receipt, "r274 bootstrap stage receipt")
        queue_entry = dict(staged.get("queue_entry") or {})
        checkpoint = dict(staged.get("checkpoint") or {})
        if (
            staged.get("schema") != STAGE_SCHEMA
            or staged.get("status") != "queued"
            or staged.get("candidate_id") != CANDIDATE_ID
            or staged.get("boundary") != "bootstrap"
            or staged.get("direct_policy_only") is not True
            or staged.get("rtp_enabled") is not False
            or staged.get("search_assets_packaged") is not False
            or staged.get("tactical_route_enabled") is not False
            or queue_entry.get("specialist_id") != SPECIALIST_ID
            or queue_entry.get("rtp_mode") != "off"
            or queue_entry.get("search_assets_packaged") is not False
            or checkpoint.get("sha256") != queue_entry.get("checkpoint_checksum")
            or checkpoint.get("sha256") != queue_entry.get("model_checksum")
        ):
            raise TriggerMaterializationError(
                "r274 bootstrap stage identity is invalid"
            )
        if file_identity(checkpoint.get("path", "")) != checkpoint:
            raise TriggerMaterializationError("r274 submission checkpoint changed")

        with queue_path.with_suffix(queue_path.suffix + ".lock").open(
            "a+", encoding="utf-8"
        ) as queue_lock:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_SH)
            queue = _read_object(queue_path, "r274 Kaggle queue")
        row = _accepted_row(queue, queue_entry)
        if row is None:
            return {"status": "not_ready", "reason": "exact_upload_not_complete"}
        if (
            row.get("rtp_mode") != "off"
            or row.get("search_assets_packaged") is not False
            or row.get("failure_reason") is not None
        ):
            raise TriggerMaterializationError(
                "accepted r274 row is not direct-policy NO-RTP"
            )

        uploaded = Path(str(row.get("file") or "")).expanduser().resolve()
        if (
            not uploaded.is_file()
            or sha256(uploaded) != row.get("file_sha256")
            or row.get("file_sha256") != queue_entry.get("file_sha256")
        ):
            raise TriggerMaterializationError("accepted r274 upload artifact changed")
        nonce = str(row.get("one_shot_authorization_nonce") or "")
        if not nonce:
            raise TriggerMaterializationError(
                "accepted r274 row lacks one-shot authorization"
            )
        attempt_path, authorization_path = _attempt_evidence(
            attempt_receipts,
            nonce=nonce,
            uploaded_file=uploaded,
            uploaded_digest=str(row["file_sha256"]),
            checkpoint_digest=str(checkpoint["sha256"]),
            competition=str(row["competition"]),
            label=str(row["label"]),
        )
        submission_id = row.get("submission_id")
        if isinstance(submission_id, bool) or not isinstance(submission_id, int):
            raise TriggerMaterializationError(
                "accepted r274 upload lacks exact remote submission id"
            )
        payload = {
            "schema": SCHEMA,
            "status": "submitted",
            "owner_revisions": [264, 268, 274, 275, 277, 279, 280],
            "candidate_id": CANDIDATE_ID,
            "checkpoint": checkpoint,
            "remote_submission_id": submission_id,
            "direct_policy_only": True,
            "rtp_enabled": False,
            "search_assets_packaged": False,
            "source_stage": file_identity(stage_receipt),
            "uploaded_file": file_identity(uploaded),
            "accepted_complete_queue_row": row,
            "accepted_complete_queue_row_sha256": canonical_digest(row),
            "attempt": file_identity(attempt_path),
            "consumed_authorization": file_identity(authorization_path),
            "tactical_activation_authorized": True,
        }
        _write_exclusive(output, payload)
        return {"status": "materialized", "receipt": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--attempt-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(**vars(args)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
