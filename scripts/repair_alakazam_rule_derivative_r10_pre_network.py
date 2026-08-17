#!/usr/bin/env python3
"""Repair the missing turn-order sidecar for the exact revision-10 package.

The first queue attempt was rejected by the local Kaggle guard before network
upload.  This tool proves the package's three required turn-order cases, writes
the missing digest-bound sidecar create-only, and re-arms the same queue row.
It never creates a second package, queue row, or submission authorization.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone


PACKAGE_SHA256 = "sha256:e241f15408c725ab8b6a01e27c63f4b9e8a604c5f08a525f42acf89fe4e32efe"
CHECKPOINT_SHA256 = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
ENTRY_ID = "alakazam-rule-derivative-g5-r10-e241f15408c7-copy1"
AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
ATTESTATION_SCHEMA = "poke_bot.submission_turn_order_attestation/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_rule_derivative_r10_pre_network_queue_repair/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_create_only(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def probe_package(package: Path) -> dict[str, object]:
    temporary = Path(tempfile.mkdtemp(prefix="r10-go-first-repair-"))
    try:
        with tarfile.open(package, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        entrypoint = temporary / "main.py"
        os.chdir(temporary)
        spec = importlib.util.spec_from_file_location("r10_submission_main", entrypoint)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load packaged main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = {
            "integer_enum": (
                {"select": {"context": 41, "minCount": 1, "maxCount": 1,
                            "option": [{"type": 1}, {"type": 2}]}}, [0]),
            "string_enum_reversed_options": (
                {"select": {"context": "IS_FIRST", "minCount": 1, "maxCount": 1,
                            "option": [{"type": "No"}, {"type": "Yes"}]}}, [1]),
            "live_engine_prompt": (
                {"select": {"context": "IsFirst", "type": 9, "minCount": 1,
                            "maxCount": 1, "option": [{"type": 2}, {"type": 1}]}}, [1]),
        }
        observed: dict[str, object] = {}
        for name, (observation, expected) in cases.items():
            selected = module.agent(observation)
            if selected != expected:
                raise RuntimeError(f"{name} expected {expected!r}, got {selected!r}")
            observed[name] = {"selected_action": selected}
        return observed
    finally:
        os.chdir("/")
        def _make_writable_and_retry(function: object, path: str, _error: object) -> None:
            os.chmod(path, 0o700)
            function(path)  # type: ignore[operator]

        shutil.rmtree(temporary, onerror=_make_writable_and_retry)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()

    package = args.package.resolve()
    if sha256_file(package) != PACKAGE_SHA256:
        raise RuntimeError("package digest mismatch")
    observed = probe_package(package)

    authorization = json.loads(args.authorization.read_text())
    if (
        authorization.get("schema") != AUTH_SCHEMA
        or authorization.get("file_sha256") != PACKAGE_SHA256
        or authorization.get("frozen_checkpoint_checksum") != CHECKPOINT_SHA256
        or authorization.get("remaining_uses") != 1
    ):
        raise RuntimeError("the exact pre-network authorization is not unused")

    attestation_path = Path(str(package) + ".go-first-verified.json")
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "file_sha256": PACKAGE_SHA256,
        "turn_order_preference": "first_if_allowed",
        "go_first_if_offered": True,
        "go_second_if_offered": False,
        "verified_cases": sorted(observed),
        "case_results": observed,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    attestation_sha256 = write_create_only(attestation_path, attestation)

    queue_path = args.queue.resolve()
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        queue = json.loads(queue_path.read_text())
        matches = [row for row in queue.get("queue", []) if row.get("shared_queue_entry_id") == ENTRY_ID]
        if len(matches) != 1:
            raise RuntimeError("exact queue row is not unique")
        row = matches[0]
        if (
            row.get("file_sha256") != PACKAGE_SHA256
            or row.get("checkpoint_checksum") != CHECKPOINT_SHA256
            or row.get("queue_status") != "failed"
            or "go_first_attestation" not in str(row.get("failure_reason") or "")
            or row.get("submission_id")
        ):
            raise RuntimeError("queue row is not the proven pre-network sidecar failure")
        row["queue_status"] = "pending"
        row["attempt_started_at"] = None
        row["attempt_quota_date"] = None
        row["failure_reason"] = None
        row["pre_network_turn_order_repair_sha256"] = attestation_sha256
        row["pre_network_turn_order_repaired_at_utc"] = datetime.now(timezone.utc).isoformat()
        queue["updated_at_utc"] = row["pre_network_turn_order_repaired_at_utc"]
        replace_json(queue_path, queue)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "package_path": str(package),
        "package_sha256": PACKAGE_SHA256,
        "candidate_checkpoint_sha256": CHECKPOINT_SHA256,
        "queue_path": str(queue_path),
        "shared_queue_entry_id": ENTRY_ID,
        "same_queue_row_rearmed": True,
        "duplicate_package_or_queue_row_created": False,
        "network_upload_occurred_before_repair": False,
        "authorization_remaining_uses": 1,
        "turn_order_attestation_path": str(attestation_path),
        "turn_order_attestation_sha256": attestation_sha256,
        "verified_cases": sorted(observed),
        "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_sha256 = write_create_only(args.receipt_root / "pre-network-queue-repair-receipt.json", receipt)
    print(json.dumps({**receipt, "receipt_sha256": receipt_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
