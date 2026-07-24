#!/usr/bin/env python3
"""Authorize one retry that was blocked before Kaggle network I/O."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
QUEUE_SCHEMA = "poke_bot.kaggle_submission_queue/v1"
PRE_NETWORK_BLOCK = "no matching unused one-shot explicit authorization"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--copy-number", type=int, required=True)
    parser.add_argument("--approval-text", required=True)
    parser.add_argument("--expires-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    queue_path = args.queue.expanduser().resolve()
    authorization_path = args.authorization.expanduser().resolve()
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        if queue.get("schema") != QUEUE_SCHEMA:
            raise RuntimeError("submission queue schema changed")
        matches = [
            row
            for row in queue.get("queue") or []
            if row.get("specialist_id") == args.specialist_id
            and int(row.get("copy_number", -1)) == args.copy_number
        ]
        if len(matches) != 1:
            raise RuntimeError("expected exactly one matching queue entry")
        entry = matches[0]
        failure = str(entry.get("failure_reason") or "")
        if (
            entry.get("queue_status") != "failed"
            or PRE_NETWORK_BLOCK not in failure
            or entry.get("submission_id") is not None
            or entry.get("submitted_at") is not None
        ):
            raise RuntimeError("queue entry is not a proven pre-network block")
        upload = Path(str(entry.get("file") or "")).expanduser().resolve()
        digest = _sha256(upload)
        if digest != str(entry.get("file_sha256") or ""):
            raise RuntimeError("queued upload digest changed")
        if authorization_path.exists():
            raise RuntimeError("another Kaggle authorization is already present")

        now = datetime.now(timezone.utc)
        nonce = (
            f"{args.specialist_id}-iter{int(entry.get('iteration', -1))}-"
            f"copy{args.copy_number}-pre-network-retry-"
            f"{now.strftime('%Y%m%dT%H%M%SZ')}"
        )
        authorization = {
            "schema": AUTH_SCHEMA,
            "explicit_user_approval": True,
            "approval_text": args.approval_text,
            "remaining_uses": 1,
            "nonce": nonce,
            "expires_at_epoch": time.time() + float(args.expires_seconds),
            "competition": str(entry.get("competition") or ""),
            "file_sha256": digest,
            "message": str(entry.get("label") or ""),
            "conditional_gate_id": str(entry.get("gate_id") or ""),
            "conditional_checkpoint_digest": str(
                entry.get("checkpoint_checksum") or ""
            ),
            "recovery_reason": (
                "The prior queue invocation was rejected by the local "
                "authorization guard before Kaggle network I/O."
            ),
        }
        _atomic_json(authorization_path, authorization)

        entry["queue_status"] = "pending"
        entry["failure_reason"] = None
        entry["attempt_started_at"] = None
        entry["attempt_quota_date"] = None
        entry["retry_count"] = int(entry.get("retry_count") or 0) + 1
        entry["pre_network_retry_authorized_at"] = now.isoformat()
        entry["pre_network_retry_nonce"] = nonce
        queue["updated_at_utc"] = now.isoformat()
        _atomic_json(queue_path, queue)
        print(
            json.dumps(
                {
                    "status": "authorized",
                    "specialist_id": args.specialist_id,
                    "copy_number": args.copy_number,
                    "file_sha256": digest,
                    "nonce": nonce,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
