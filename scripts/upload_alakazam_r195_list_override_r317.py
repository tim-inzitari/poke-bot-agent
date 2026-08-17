#!/usr/bin/env python3
"""Perform the one owner-authorized immediate Kaggle upload, without retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_SHA = "sha256:deed52e1524870d9cebee8cd044d4a8f3555b74de1cf1f5f248f9898564fd719"
RECEIPT_SHA = "sha256:2a69ae1041b23dd630e97387ae3a113c9ec737e4d7122078815067ffe55a3b98"
MODEL_SHA = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
DECK_SHA = "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
LABEL = "8b59af9af1d7 new policy on 55468965 r195 deck list NO RTP"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_create_only(path: Path, value: Any) -> str:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    args = parser.parse_args()
    if sha_file(args.package) != PACKAGE_SHA or sha_file(args.package_receipt) != RECEIPT_SHA:
        raise RuntimeError("sealed package or receipt identity mismatch")
    receipt = json.loads(args.package_receipt.read_text())
    if not (
        receipt.get("model_checkpoint_sha256") == MODEL_SHA
        and receipt.get("deck_file_sha256") == DECK_SHA
        and receipt.get("changed_regular_members_from_model_source") == ["deck.csv"]
        and receipt.get("isolated_smoke_passed") is True
        and receipt.get("rtp_mode") == "disabled"
        and receipt.get("search_or_mcts_enabled") is False
    ):
        raise RuntimeError("package receipt is not upload eligible")
    authorization_path = args.package.parent / "immediate-single-upload-authorization.json"
    attempt_path = args.package.parent / "immediate-upload-attempt.json"
    if authorization_path.exists() or attempt_path.exists():
        raise RuntimeError("single upload authorization was already created or consumed")
    authorized_at = datetime.now(timezone.utc).isoformat()
    authorization_sha = write_create_only(authorization_path, {
        "schema": "poke_bot.alakazam_derivative_r195_list_immediate_upload_authorization/v1",
        "root_goal_revision": 317,
        "dedicated_goal_revision": 15,
        "owner_instruction": "upload now dont wait for boundry",
        "competition": args.competition,
        "canonical_label": LABEL,
        "package_sha256": PACKAGE_SHA,
        "package_receipt_sha256": RECEIPT_SHA,
        "model_checkpoint_sha256": MODEL_SHA,
        "deck_source_submission_id": 55468965,
        "deck_file_sha256": DECK_SHA,
        "authorized_network_upload_attempts": 1,
        "spacing_wait_bypassed": True,
        "retry_duplicate_or_second_copy_authorized": False,
        "authorized_at_utc": authorized_at,
    })
    started = datetime.now(timezone.utc).isoformat()
    command = [
        str(args.kaggle), "competitions", "submit", "-c", args.competition,
        "-f", str(args.package.resolve()), "-m", LABEL,
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900)
        returncode: int | None = completed.returncode
        output = "\n".join((completed.stdout, completed.stderr)).strip()
        outcome = "accepted_by_kaggle_cli" if returncode == 0 else "failed_no_retry"
    except subprocess.TimeoutExpired as exc:
        returncode = None
        output = f"timeout after {exc.timeout} seconds; upload outcome unknown; retry forbidden"
        outcome = "unknown_no_retry"
    finished = datetime.now(timezone.utc).isoformat()
    attempt_sha = write_create_only(attempt_path, {
        "schema": "poke_bot.alakazam_derivative_r195_list_immediate_upload_attempt/v1",
        "root_goal_revision": 317,
        "dedicated_goal_revision": 15,
        "authorization_sha256": authorization_sha,
        "competition": args.competition,
        "canonical_label": LABEL,
        "package_path": str(args.package.resolve()),
        "package_sha256": PACKAGE_SHA,
        "network_upload_attempt_count": 1,
        "retry_count": 0,
        "retry_forbidden": True,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "returncode": returncode,
        "outcome": outcome,
        "kaggle_cli_output": output,
    })
    print(json.dumps({
        "outcome": outcome, "returncode": returncode, "output": output,
        "authorization_sha256": authorization_sha, "attempt_receipt_sha256": attempt_sha,
    }, sort_keys=True))
    return 0 if returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
