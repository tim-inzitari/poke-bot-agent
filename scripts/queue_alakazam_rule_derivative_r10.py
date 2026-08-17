#!/usr/bin/env python3
"""Durably enqueue the one authorized revision-10 full-model package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any


QUEUE_SCHEMA = "poke_bot.kaggle_submission_queue/v1"
GOAL_REVISION = 11
GOAL_CONTRACT_SHA256 = "sha256:d74152bca415c80e4983172b5fdcd8c03313e0c8a0a18e24e0c68a7bcfe84245"
PACKAGE_SHA256 = "sha256:e241f15408c725ab8b6a01e27c63f4b9e8a604c5f08a525f42acf89fe4e32efe"
PACKAGE_RECEIPT_SHA256 = "sha256:bd7f8d0910538c5c4bb1af9c6dcbd463862c9240c6af39a882ae41c8dff90c8f"
CANDIDATE_SHA256 = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
DECK_FILE_SHA256 = "sha256:d834c66c5a3629dd79c8533a04fde770a22ca8590ac55c9868440121b6df5fba"
DECK_ORDERED_SHA256 = "sha256:e61c0a4ffcfeb730808ac561f39c1efa9de5f80aec577d3be82f3fc790b7dab2"
DECK_MULTISET_SHA256 = "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
SEARCH_CONFIG_SHA256 = "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
BELIEF_DECKS_SHA256 = "sha256:b8a7f709426652fe85c18b6f5c9cdb757dd99abef6fd04d62537805306e29af0"
PUBLIC_CATALOG_SHA256 = "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6"
COMPETITION = "pokemon-tcg-ai-battle"
LABEL = "Alakazam public-rule derivative g5, exact new list, Blackwell bootstrap, no RTP"
AUTHORIZATION_ID = "alakazam-rule-derivative-g5-r10-e241f15408c7-single-use"
ENTRY_ID = "alakazam-rule-derivative-g5-r10-e241f15408c7-copy1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json_create_only(path: Path, payload: Any) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def replace_json_durable(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()

    if args.receipt_root.exists():
        raise RuntimeError(f"receipt root already exists: {args.receipt_root}")
    for path, expected, label in [
        (args.package, PACKAGE_SHA256, "package"),
        (args.package_receipt, PACKAGE_RECEIPT_SHA256, "package receipt"),
        (args.candidate, CANDIDATE_SHA256, "candidate"),
    ]:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"{label} identity mismatch")
    package_receipt = json.loads(args.package_receipt.read_text())
    if (
        package_receipt.get("goal_revision") != GOAL_REVISION
        or package_receipt.get("goal_contract_sha256") != GOAL_CONTRACT_SHA256
        or package_receipt.get("package_sha256") != PACKAGE_SHA256
        or package_receipt.get("candidate_checkpoint_sha256") != CANDIDATE_SHA256
        or package_receipt.get("deck_canonical_multiset_sha256") != DECK_MULTISET_SHA256
        or package_receipt.get("rtp_enabled") is not False
        or package_receipt.get("search_or_mcts_enabled") is not False
        or package_receipt.get("package_smoke_passed") is not True
        or package_receipt.get("package_parity_passed") is not True
    ):
        raise RuntimeError("package receipt is not queue eligible")

    deduplication_key = digest_value({
        "authorization_id": AUTHORIZATION_ID,
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "competition": COMPETITION,
        "deck_canonical_multiset_sha256": DECK_MULTISET_SHA256,
        "goal_revision": GOAL_REVISION,
        "package_sha256": PACKAGE_SHA256,
    })
    queued_at = datetime.now(timezone.utc).isoformat()
    args.receipt_root.mkdir(parents=True, exist_ok=False)
    authorization = {
        "schema": "poke_bot.alakazam_rule_derivative_kaggle_single_use_authorization/v1",
        "authorization_id": AUTHORIZATION_ID,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "owner_bootstrap_revision": 10,
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "package_receipt_sha256": PACKAGE_RECEIPT_SHA256,
        "package_sha256": PACKAGE_SHA256,
        "deck_canonical_multiset_sha256": DECK_MULTISET_SHA256,
        "competition": COMPETITION,
        "canonical_label": LABEL,
        "turn_order_preference": "first_if_allowed",
        "remaining_uses_before_enqueue": 1,
        "authorized_action": "append_exactly_one_existing_shared_queue_entry",
        "authorized_at_utc": queued_at,
    }
    authorization_sha = write_json_create_only(args.receipt_root / "single-use-authorization.json", authorization)
    entry = {
        "shared_queue_entry_id": ENTRY_ID,
        "deduplication_key": deduplication_key,
        "file": str(args.package.resolve()),
        "file_sha256": PACKAGE_SHA256,
        "label": LABEL,
        "gate_id": "alakazam-rule-derivative-g5-r10",
        "iteration": 25,
        "specialist_id": "alakazam-rule-derivative-g5",
        "copy_number": 1,
        "model_checksum": CANDIDATE_SHA256,
        "checkpoint_checksum": CANDIDATE_SHA256,
        "deck_file_checksum": DECK_FILE_SHA256,
        "deck_cards_checksum": DECK_ORDERED_SHA256,
        "deck_canonical_multiset_sha256": DECK_MULTISET_SHA256,
        "representatives_checksum": PUBLIC_CATALOG_SHA256,
        "matchup_tree_checksum": MATCHUP_TREE_SHA256,
        "search_config_checksum": SEARCH_CONFIG_SHA256,
        "belief_decks_checksum": BELIEF_DECKS_SHA256,
        "competition": COMPETITION,
        "turn_order_preference": "first_if_allowed",
        "rtp_mode": "disabled",
        "queue_status": "pending",
        "queued_at": queued_at,
        "owner_authorization_source": "goals/alakazam-elmo-rule-derivative/GOAL.md#revision-10",
        "owner_decision_revision": 10,
        "derivative_single_use_authorization_id": AUTHORIZATION_ID,
        "derivative_single_use_authorization_sha256": authorization_sha,
        "derivative_package_receipt_sha256": PACKAGE_RECEIPT_SHA256,
        "attempt_count": 0,
        "retry_count": 0,
        "failure_reason": None,
    }

    queue_path = args.queue.resolve()
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        queue = json.loads(queue_path.read_text())
        if queue.get("schema") != QUEUE_SCHEMA:
            raise RuntimeError("shared queue schema mismatch")
        rows = list(queue.get("queue") or [])
        preexisting = sum(1 for row in rows if (
            row.get("deduplication_key") == deduplication_key
            or row.get("file_sha256") == PACKAGE_SHA256
            or row.get("checkpoint_checksum") == CANDIDATE_SHA256
            or row.get("shared_queue_entry_id") == ENTRY_ID
        ))
        if preexisting:
            raise RuntimeError("revision-10 full candidate/package is already queued")
        rows.append(entry)
        queue["queue"] = rows
        queue["updated_at_utc"] = queued_at
        replace_json_durable(queue_path, queue)
        queue_sha = sha256_file(queue_path)
        reopened = json.loads(queue_path.read_text())
        exact = [row for row in reopened.get("queue", []) if row.get("shared_queue_entry_id") == ENTRY_ID and row.get("file_sha256") == PACKAGE_SHA256]
        if len(exact) != 1:
            raise RuntimeError("durable queue reopen did not contain one exact entry")

    receipt = {
        "schema": "poke_bot.alakazam_rule_derivative_kaggle_queue_receipt/v1",
        "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "single_use_authorization_id": AUTHORIZATION_ID,
        "authorization_receipt_sha256": authorization_sha,
        "package_receipt_sha256": PACKAGE_RECEIPT_SHA256,
        "package_sha256": PACKAGE_SHA256,
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "deck_canonical_multiset_sha256": DECK_MULTISET_SHA256,
        "canonical_label": LABEL,
        "competition": COMPETITION,
        "turn_order_preference": "first_if_allowed",
        "shared_queue_path": str(queue_path),
        "shared_queue_service": "pokebot-kaggle-submission-queue.service",
        "shared_queue_entry_id": ENTRY_ID,
        "deduplication_key": deduplication_key,
        "preexisting_matching_entry_count": 0,
        "queue_entry_status": "accepted_by_queue",
        "quota_spacing_state": dict(queue.get("quota") or {}),
        "queue_file_sha256_after": queue_sha,
        "single_use_consumed": True,
        "duplicate_or_retry_created": False,
        "durable_fsync_or_equivalent_passed": True,
        "queued_at_utc": queued_at,
    }
    receipt_sha = write_json_create_only(args.receipt_root / "kaggle-queue-receipt.json", receipt)
    print(json.dumps({"queue_receipt_sha256": receipt_sha, "authorization_receipt_sha256": authorization_sha, "shared_queue_entry_id": ENTRY_ID, "package_sha256": PACKAGE_SHA256, "candidate_checkpoint_sha256": CANDIDATE_SHA256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
