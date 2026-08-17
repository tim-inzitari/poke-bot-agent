#!/usr/bin/env python3
"""Seal the final Kaggle completion state for submission 55487412."""

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


LABEL = "8b59af9af1d7 new policy on 55468965 r195 deck list NO RTP"
PACKAGE_SHA = "sha256:deed52e1524870d9cebee8cd044d4a8f3555b74de1cf1f5f248f9898564fd719"
SUBMISSION_RECEIPT_SHA = "sha256:40cebb431b699150ad9a265737535c889428c5979b38775674b870564cea628a"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    if sha_file(args.package) != PACKAGE_SHA or sha_file(args.submission_receipt) != SUBMISSION_RECEIPT_SHA:
        raise RuntimeError("package or accepted-submission receipt identity mismatch")
    result = subprocess.run(
        [str(args.kaggle), "competitions", "submissions", "pokemon-tcg-ai-battle", "-v", "--page-size", "200"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    matches = [row for row in csv.DictReader(io.StringIO(result.stdout)) if row.get("description") == LABEL]
    if len(matches) != 1 or matches[0].get("ref") != "55487412":
        raise RuntimeError("exact submission is not unique")
    row = matches[0]
    if row.get("status") != "SubmissionStatus.COMPLETE":
        raise RuntimeError("submission is not complete")
    payload = {
        "schema": "poke_bot.alakazam_derivative_r195_list_kaggle_completion_receipt/v1",
        "submission_id": 55487412,
        "kaggle_status": "COMPLETE",
        "public_score": float(row["publicScore"]),
        "canonical_label": LABEL,
        "package_sha256": PACKAGE_SHA,
        "accepted_submission_receipt_sha256": SUBMISSION_RECEIPT_SHA,
        "matching_kaggle_submission_count": 1,
        "actual_network_upload_attempt_count": 1,
        "retry_count": 0,
        "duplicate_or_second_copy_created": False,
        "completed_at_observed_utc": datetime.now(timezone.utc).isoformat(),
    }
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path = args.package.parent / "kaggle-completion-receipt.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body); os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({**payload, "completion_receipt_sha256": "sha256:" + hashlib.sha256(body).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
