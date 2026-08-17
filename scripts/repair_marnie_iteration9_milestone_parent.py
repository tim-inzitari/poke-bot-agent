#!/usr/bin/env python3
"""Supersede an unuploaded iter-9 candidate bundle with the committed learner.

This repair is intentionally narrow.  It is valid only while the managed queue
processor is stopped, no upload attempt has started, and the iteration-9 family
trigger is still absent.  The original receipt, bundle, and queue row remain
immutable audit evidence under explicit superseded identities.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage_final_format_alakazam_milestone_submissions import (
    atomic_json,
    read_json,
    sha256,
    validate_commit,
)


SCHEMA = "poke_bot.marnie_iteration9_milestone_parent_repair/v1"


def _queue_service_inactive(service: str) -> None:
    state = subprocess.run(
        ["systemctl", "--user", "is-active", service],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    if state in {"active", "activating", "reloading"}:
        raise RuntimeError(f"managed queue processor is still {state}")


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reconcile_interrupted_attempt(
    *, row: dict[str, Any], attempts: Path, kaggle: Path
) -> dict[str, Any] | None:
    if not row.get("attempt_started_at"):
        return None
    nonce = str(row.get("one_shot_authorization_nonce") or "")
    if not nonce:
        raise RuntimeError("started attempt lacks its one-shot nonce")
    consumed_matches = sorted(
        attempts.glob(f"*-{nonce}.authorization-consumed.json")
    )
    attempt_matches = sorted(
        path
        for path in attempts.glob(f"*-{nonce}.json")
        if not path.name.endswith(".authorization-consumed.json")
    )
    if len(consumed_matches) != 1 or attempt_matches:
        raise RuntimeError("interrupted attempt receipts are not exact")
    consumed = read_json(consumed_matches[0])
    if (
        consumed.get("schema")
        != "poke_bot.kaggle_submission_authorization/v1"
        or consumed.get("consumed_before_upload") is not True
        or int(consumed.get("remaining_uses", -1)) != 0
        or consumed.get("nonce") != nonce
        or consumed.get("frozen_checkpoint_checksum")
        != row.get("checkpoint_checksum")
        or consumed.get("submission_file_checksum") != row.get("file_sha256")
    ):
        raise RuntimeError("consumed authorization identity changed")
    completed = subprocess.run(
        [
            str(kaggle),
            "competitions",
            "submissions",
            "-c",
            str(row.get("competition") or "pokemon-tcg-ai-battle"),
            "--csv",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot reconcile interrupted attempt against Kaggle")
    submissions = list(csv.DictReader(io.StringIO(completed.stdout)))
    if any(
        str(item.get("description") or "") == str(row.get("label") or "")
        for item in submissions
    ):
        raise RuntimeError("old candidate is already visible on Kaggle")
    return {
        "status": "consumed_authorization_but_no_upload_receipt_or_kaggle_row",
        "attempt_started_at": str(row.get("attempt_started_at")),
        "nonce": nonce,
        "consumed_authorization": str(consumed_matches[0].resolve()),
        "consumed_authorization_sha256": sha256(consumed_matches[0]),
        "kaggle_rows_checked": len(submissions),
        "label_absent": True,
    }


def repair(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = read_json(args.output)
        if existing.get("schema") != SCHEMA:
            raise RuntimeError("existing repair receipt has the wrong schema")
        return existing
    _queue_service_inactive(args.queue_service)
    if args.trigger.exists():
        raise RuntimeError("iteration-9 upload trigger already exists")

    _, learner_path, learner_digest, learner_role = validate_commit(
        args.run_dir,
        9,
        prefer_committed_learner=True,
    )
    if learner_role != "committed_learner":
        raise RuntimeError("iteration-9 learner selection contract changed")

    old = read_json(args.milestone_receipt)
    if (
        old.get("schema") != "poke_bot.final_format_milestone_submission/v1"
        or old.get("status") != "queued"
        or int(old.get("iteration", -1)) != 9
        or old.get("checkpoint_sha256") == learner_digest
    ):
        raise RuntimeError("milestone is not the unuploaded candidate mismatch")
    old_checkpoint = Path(str(old.get("checkpoint") or "")).resolve()
    old_digest = str(old.get("checkpoint_sha256") or "")
    if not old_checkpoint.is_file() or sha256(old_checkpoint) != old_digest:
        raise RuntimeError("old candidate checkpoint identity changed")
    old_row = dict(old.get("queue_entry") or {})
    old_bundle = dict(old.get("bundle") or {})
    old_bundle_path = Path(str(old_bundle.get("path") or "")).resolve()
    if not old_bundle_path.is_file() or sha256(old_bundle_path) != str(
        old_bundle.get("sha256") or ""
    ):
        raise RuntimeError("old candidate bundle identity changed")

    lock_path = args.queue.with_suffix(args.queue.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    repaired_at = datetime.now(timezone.utc).isoformat()
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        queue = read_json(args.queue)
        matches = [
            row
            for row in list(queue.get("queue") or [])
            if isinstance(row, dict)
            and row.get("label") == old_row.get("label")
            and row.get("checkpoint_checksum") == old_digest
            and row.get("file_sha256") == old_row.get("file_sha256")
        ]
        if len(matches) != 1:
            raise RuntimeError("old candidate queue row is not unique")
        row = matches[0]
        already_superseded = row.get("queue_status") == "superseded_before_upload"
        interrupted_attempt = _reconcile_interrupted_attempt(
            row=row,
            attempts=args.attempt_receipts,
            kaggle=args.kaggle,
        )
        if not already_superseded and (
            row.get("queue_status") != "pending"
            or row.get("submitted_at")
            or row.get("submission_id")
            or row.get("kaggle_status")
        ):
            raise RuntimeError("old candidate upload already started")
        if not already_superseded:
            row.update(
                {
                    "queue_status": "superseded_before_upload",
                    "failure_reason": (
                        "iteration-9 exact-gate rollback selected a different "
                        "committed learner"
                    ),
                    "superseded_at_utc": repaired_at,
                    "replacement_checkpoint_checksum": learner_digest,
                }
            )
            queue["updated_at_utc"] = repaired_at
            atomic_json(args.queue, queue)

    archive_receipt = (
        args.milestone_receipt.parent
        / "superseded"
        / f"iter_00009-candidate-{old_digest.removeprefix('sha256:')[:12]}.json"
    )
    archive_receipt.parent.mkdir(parents=True, exist_ok=True)
    if archive_receipt.exists():
        if sha256(archive_receipt) != sha256(args.milestone_receipt):
            raise RuntimeError("archived milestone receipt identity changed")
        args.milestone_receipt.unlink()
    else:
        os.replace(args.milestone_receipt, archive_receipt)

    archived_root = args.submission_root.with_name(
        f"{args.submission_root.name}.superseded-candidate-"
        f"{old_digest.removeprefix('sha256:')[:12]}"
    )
    if archived_root.exists():
        if args.submission_root.exists():
            raise RuntimeError("both active and archived candidate roots exist")
    else:
        os.replace(args.submission_root, archived_root)
    archived_bundle_path = archived_root / old_bundle_path.relative_to(
        args.submission_root
    )
    if (
        not archived_bundle_path.is_file()
        or sha256(archived_bundle_path) != str(old_bundle.get("sha256") or "")
    ):
        raise RuntimeError("archived candidate bundle identity changed")

    payload = {
        "schema": SCHEMA,
        "status": "superseded_before_upload",
        "iteration": 9,
        "old_candidate": {
            "checkpoint": str(old_checkpoint),
            "sha256": old_digest,
            "bundle": str(archived_bundle_path),
            "bundle_sha256": str(old_bundle.get("sha256") or ""),
            "archived_receipt": str(archive_receipt),
            "archived_receipt_sha256": sha256(archive_receipt),
            "archived_submission_root": str(archived_root),
        },
        "replacement_committed_learner": {
            "checkpoint": str(learner_path),
            "sha256": learner_digest,
            "role": learner_role,
        },
        "queue": str(args.queue),
        "queue_sha256_after_supersede": sha256(args.queue),
        "trigger_absent": True,
        "interrupted_attempt_reconciliation": interrupted_attempt,
        "repaired_at_utc": repaired_at,
    }
    _exclusive_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--milestone-receipt", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-receipts", type=Path, required=True)
    parser.add_argument(
        "--kaggle",
        type=Path,
        default=Path("/home/pokebot/miniconda3/envs/poke-bot-agent/bin/kaggle"),
    )
    parser.add_argument(
        "--queue-service",
        default="pokebot-kaggle-submission-queue.service",
    )
    args = parser.parse_args()
    for name in (
        "run_dir",
        "milestone_receipt",
        "submission_root",
        "queue",
        "trigger",
        "output",
        "attempt_receipts",
        "kaggle",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    print(json.dumps(repair(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
