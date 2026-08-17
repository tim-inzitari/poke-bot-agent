#!/usr/bin/env python3
"""Seal the accepted Kaggle identity for the one r317 upload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_SHA = "sha256:deed52e1524870d9cebee8cd044d4a8f3555b74de1cf1f5f248f9898564fd719"
MODEL_SHA = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
DECK_SHA = "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
LABEL = "8b59af9af1d7 new policy on 55468965 r195 deck list NO RTP"
NONCE = "alakazam-r317-8b59af9af1d7-on-55468965-single-network-upload"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_create_only(path: Path, value: Any) -> str:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body); os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--guard-receipts", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    args = parser.parse_args()
    package = args.package.resolve()
    if sha_file(package) != PACKAGE_SHA:
        raise RuntimeError("package identity mismatch")
    if args.authorization.exists():
        raise RuntimeError("one-shot authorization was not consumed")
    guard_files = sorted(args.guard_receipts.glob(f"*{NONCE}*.json"))
    attempt_files = [path for path in guard_files if not path.name.endswith(".authorization-consumed.json")]
    consumed_files = [path for path in guard_files if path.name.endswith(".authorization-consumed.json")]
    if len(attempt_files) != 1 or len(consumed_files) != 1:
        raise RuntimeError("expected exactly one consumed authorization and one guard attempt receipt")
    guard_attempt = json.loads(attempt_files[0].read_text())
    if guard_attempt.get("returncode") != 0:
        raise RuntimeError("the sole actual network upload did not succeed")
    completed = subprocess.run(
        [str(args.kaggle), "competitions", "submissions", args.competition, "-v", "--page-size", "200"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    matches = [row for row in csv.DictReader(io.StringIO(completed.stdout)) if row.get("description") == LABEL]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Kaggle row with the canonical label, got {len(matches)}")
    row = matches[0]
    receipt = {
        "schema": "poke_bot.alakazam_derivative_r195_list_kaggle_submission_receipt/v1",
        "root_goal_revision": 317,
        "dedicated_goal_revision": 15,
        "competition": args.competition,
        "submission_id": int(row["ref"]),
        "submitted_at": row["date"],
        "kaggle_status": row["status"].removeprefix("SubmissionStatus."),
        "public_score": row.get("publicScore") or None,
        "canonical_label": LABEL,
        "package_path": str(package),
        "package_sha256": PACKAGE_SHA,
        "model_checkpoint_sha256": MODEL_SHA,
        "deck_source_submission_id": 55468965,
        "deck_file_sha256": DECK_SHA,
        "rtp_mode": "disabled",
        "search_or_mcts_enabled": False,
        "turn_order_preference": "first_if_allowed",
        "matching_kaggle_submission_count": len(matches),
        "actual_network_upload_attempt_count": 1,
        "retry_count": 0,
        "duplicate_or_second_copy_created": False,
        "guard_attempt_receipt_path": str(attempt_files[0]),
        "guard_attempt_receipt_sha256": sha_file(attempt_files[0]),
        "consumed_authorization_path": str(consumed_files[0]),
        "consumed_authorization_sha256": sha_file(consumed_files[0]),
        "reconciled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_sha = write_create_only(package.parent / "kaggle-submission-receipt.json", receipt)
    print(json.dumps({**receipt, "submission_receipt_sha256": receipt_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
