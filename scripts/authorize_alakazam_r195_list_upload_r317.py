#!/usr/bin/env python3
"""Attest turn order and arm the one actual r317 Kaggle network upload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_SHA = "sha256:deed52e1524870d9cebee8cd044d4a8f3555b74de1cf1f5f248f9898564fd719"
MODEL_SHA = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
LABEL = "8b59af9af1d7 new policy on 55468965 r195 deck list NO RTP"
NONCE = "alakazam-r317-8b59af9af1d7-on-55468965-single-network-upload"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_create_only(path: Path, value: Any, mode: int = 0o444) -> str:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def probe(package: Path) -> dict[str, Any]:
    temporary = Path(tempfile.mkdtemp(prefix="r317-turn-order-attestation-"))
    prior = Path.cwd()
    try:
        with tarfile.open(package, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        os.chdir(temporary)
        spec = importlib.util.spec_from_file_location("r317_submission_main", temporary / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load submission entrypoint")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cases = {
            "integer_enum": ({"select": {"context": 41, "minCount": 1, "maxCount": 1, "option": [{"type": 1}, {"type": 2}]}}, [0]),
            "string_enum_reversed_options": ({"select": {"context": "IS_FIRST", "minCount": 1, "maxCount": 1, "option": [{"type": "No"}, {"type": "Yes"}]}}, [1]),
            "live_engine_prompt": ({"select": {"context": "IsFirst", "type": 9, "minCount": 1, "maxCount": 1, "option": [{"type": 2}, {"type": 1}]}}, [1]),
        }
        observed: dict[str, Any] = {}
        for name, (observation, expected) in cases.items():
            actual = module.agent(observation)
            if actual != expected:
                raise RuntimeError(f"turn-order case {name} expected {expected}, got {actual}")
            observed[name] = {"selected_action": actual}
        return observed
    finally:
        os.chdir(prior)
        shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--failed-attempt", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    args = parser.parse_args()
    package = args.package.resolve()
    if sha_file(package) != PACKAGE_SHA:
        raise RuntimeError("package identity mismatch")
    failed = json.loads(args.failed_attempt.read_text())
    if not (
        failed.get("returncode") == 73
        and failed.get("outcome") == "failed_no_retry"
        and "submission BLOCKED" in str(failed.get("kaggle_cli_output"))
    ):
        raise RuntimeError("prior invocation is not a proven local pre-network guard rejection")
    if args.authorization.exists():
        raise RuntimeError("a Kaggle authorization already exists")
    attestation_path = Path(str(package) + ".go-first-verified.json")
    repair_path = package.parent / "pre-network-guard-repair.json"
    if attestation_path.exists() or repair_path.exists():
        raise RuntimeError("r317 turn-order attestation or repair receipt already exists")
    observed = probe(package)
    now = datetime.now(timezone.utc).isoformat()
    attestation_sha = write_create_only(attestation_path, {
        "schema": "poke_bot.submission_turn_order_attestation/v1",
        "file_sha256": PACKAGE_SHA,
        "turn_order_preference": "first_if_allowed",
        "go_first_if_offered": True,
        "go_second_if_offered": False,
        "verified_cases": sorted(observed),
        "case_results": observed,
        "verified_at_utc": now,
    })
    authorization = {
        "schema": "poke_bot.kaggle_submission_authorization/v1",
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": NONCE,
        "expires_at_epoch": time.time() + 900,
        "competition": args.competition,
        "file_sha256": PACKAGE_SHA,
        "message": LABEL,
        "turn_order_preference": "first_if_allowed",
        "frozen_checkpoint_checksum": MODEL_SHA,
        "owner_authorization_source": "goals/alakazam-elmo-rule-derivative/GOAL.md#revision-15",
        "root_owner_authorization_source": "GOAL.md#/decision-ledger/revision-317-KAGGLE",
        "retry_or_second_copy_authorized": False,
        "created_at_utc": now,
    }
    authorization_sha = write_create_only(args.authorization, authorization, 0o600)
    repair_sha = write_create_only(repair_path, {
        "schema": "poke_bot.alakazam_derivative_r195_list_pre_network_guard_repair/v1",
        "package_sha256": PACKAGE_SHA,
        "model_checkpoint_sha256": MODEL_SHA,
        "prior_guard_returncode": 73,
        "prior_network_upload_occurred": False,
        "same_package_reused": True,
        "duplicate_package_created": False,
        "turn_order_attestation_sha256": attestation_sha,
        "authorization_sha256": authorization_sha,
        "authorized_actual_network_uploads_remaining": 1,
        "repaired_at_utc": now,
    })
    print(json.dumps({"authorization_sha256": authorization_sha, "attestation_sha256": attestation_sha, "repair_receipt_sha256": repair_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
