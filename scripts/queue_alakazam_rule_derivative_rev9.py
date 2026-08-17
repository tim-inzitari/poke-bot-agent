#!/usr/bin/env python3
"""Durably enqueue the single authorized revision-9 derivative package."""

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
AUTH_SCHEMA = "poke_bot.alakazam_rule_derivative_kaggle_single_use_authorization/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_rule_derivative_kaggle_queue_receipt/v1"
GOAL_REVISION = 9
GOAL_CONTRACT_SHA256 = (
    "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
)
PACKAGE_SHA256 = (
    "sha256:bfc5736373bf5d2b63d30218851f5907148a3ff18e237236d971c8af2eabc7e5"
)
PACKAGE_RECEIPT_SHA256 = (
    "sha256:03f34eccce7955e7eef7472e1c6cb5937a839f36a5285c09288d00ed1650a423"
)
CANDIDATE_SHA256 = (
    "sha256:5c42b99a5eb101c1ea173ae9426326db61d3bdb81a84903468e7bac5e6a30f24"
)
DECK_FILE_SHA256 = (
    "sha256:d834c66c5a3629dd79c8533a04fde770a22ca8590ac55c9868440121b6df5fba"
)
DECK_ORDERED_SHA256 = (
    "sha256:e61c0a4ffcfeb730808ac561f39c1efa9de5f80aec577d3be82f3fc790b7dab2"
)
DECK_MULTISET_SHA256 = (
    "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
)
MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
SEARCH_CONFIG_SHA256 = (
    "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
)
BELIEF_DECKS_SHA256 = (
    "sha256:b8a7f709426652fe85c18b6f5c9cdb757dd99abef6fd04d62537805306e29af0"
)
PUBLIC_CATALOG_SHA256 = (
    "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6"
)
COMPETITION = "pokemon-tcg-ai-battle"
LABEL = "Alakazam public-rule derivative g5, exact new list, Blackwell bootstrap, no RTP"
AUTHORIZATION_ID = "alakazam-rule-derivative-g5-rev9-bfc5736373bf-single-use"
ENTRY_ID = "alakazam-rule-derivative-g5-rev9-bfc5736373bf-copy1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json_durable(path: Path, payload: Any) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()

    queue_path = args.queue.resolve()
    package_path = args.package.resolve()
    package_receipt_path = args.package_receipt.resolve()
    candidate_path = args.candidate.resolve()
    receipt_root = args.receipt_root.resolve()
    if receipt_root.exists():
        raise RuntimeError(f"receipt root already exists: {receipt_root}")
    if _sha256(package_path) != PACKAGE_SHA256:
        raise RuntimeError("package identity mismatch")
    if _sha256(package_receipt_path) != PACKAGE_RECEIPT_SHA256:
        raise RuntimeError("package receipt identity mismatch")
    if _sha256(candidate_path) != CANDIDATE_SHA256:
        raise RuntimeError("candidate identity mismatch")
    package_receipt = json.loads(package_receipt_path.read_text(encoding="utf-8"))
    if (
        package_receipt.get("package_sha256") != PACKAGE_SHA256
        or package_receipt.get("candidate_checkpoint_sha256") != CANDIDATE_SHA256
        or package_receipt.get("deck_canonical_multiset_sha256")
        != DECK_MULTISET_SHA256
        or package_receipt.get("rtp_enabled") is not False
        or package_receipt.get("search_or_mcts_enabled") is not False
        or package_receipt.get("package_smoke_passed") is not True
        or package_receipt.get("package_parity_passed") is not True
    ):
        raise RuntimeError("package receipt is not queue eligible")

    deduplication_key = _digest_value(
        {
            "authorization_id": AUTHORIZATION_ID,
            "candidate_checkpoint_sha256": CANDIDATE_SHA256,
            "competition": COMPETITION,
            "deck_canonical_multiset_sha256": DECK_MULTISET_SHA256,
            "goal_revision": GOAL_REVISION,
            "package_sha256": PACKAGE_SHA256,
        }
    )
    queued_at = datetime.now(timezone.utc).isoformat()
    authorization = {
        "schema": AUTH_SCHEMA,
        "authorization_id": AUTHORIZATION_ID,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
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

    receipt_root.mkdir(parents=True, exist_ok=False)
    authorization_path = receipt_root / "single-use-authorization.json"
    _write_json_create_only(authorization_path, authorization)
    authorization_sha256 = _sha256(authorization_path)

    entry = {
        "shared_queue_entry_id": ENTRY_ID,
        "deduplication_key": deduplication_key,
        "file": str(package_path),
        "file_sha256": PACKAGE_SHA256,
        "label": LABEL,
        "gate_id": "alakazam-rule-derivative-g5-rev9",
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
        "owner_authorization_source": "goals/alakazam-elmo-rule-derivative/GOAL.md#revision-9",
        "owner_decision_revision": GOAL_REVISION,
        "derivative_single_use_authorization_id": AUTHORIZATION_ID,
        "derivative_single_use_authorization_sha256": authorization_sha256,
        "derivative_package_receipt_sha256": PACKAGE_RECEIPT_SHA256,
        "attempt_count": 0,
        "retry_count": 0,
        "failure_reason": None,
    }

    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
        if queue_payload.get("schema") != QUEUE_SCHEMA:
            raise RuntimeError("shared queue schema mismatch")
        rows = list(queue_payload.get("queue") or [])
        preexisting = sum(
            1
            for row in rows
            if row.get("deduplication_key") == deduplication_key
            or row.get("file_sha256") == PACKAGE_SHA256
            or row.get("shared_queue_entry_id") == ENTRY_ID
        )
        if preexisting:
            raise RuntimeError("the derivative package is already queued")
        rows.append(entry)
        queue_payload["queue"] = rows
        queue_payload["updated_at_utc"] = queued_at
        _replace_json_durable(queue_path, queue_payload)
        queue_file_sha256_after = _sha256(queue_path)
        reopened = json.loads(queue_path.read_text(encoding="utf-8"))
        exact_rows = [
            row
            for row in reopened.get("queue", [])
            if row.get("shared_queue_entry_id") == ENTRY_ID
            and row.get("deduplication_key") == deduplication_key
            and row.get("file_sha256") == PACKAGE_SHA256
        ]
        if len(exact_rows) != 1:
            raise RuntimeError("durable queue reopen did not contain one exact entry")

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
            "goal_contract_sha256": GOAL_CONTRACT_SHA256,
            "goal_revision": GOAL_REVISION,
            "single_use_authorization_id": AUTHORIZATION_ID,
            "authorization_receipt_sha256": authorization_sha256,
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
            "preexisting_matching_entry_count": preexisting,
            "queue_entry_status": "accepted_by_queue",
            "quota_spacing_state": dict(queue_payload.get("quota") or {}),
            "queue_file_sha256_after": queue_file_sha256_after,
            "single_use_consumed": True,
            "duplicate_or_retry_created": False,
            "durable_fsync_or_equivalent_passed": True,
            "queued_at_utc": queued_at,
        }
        receipt_path = receipt_root / "kaggle-queue-receipt.json"
        _write_json_create_only(receipt_path, receipt)
        complete = {
            "schema": "poke_bot.alakazam_rule_derivative_kaggle_queue_completion/v1",
            "status": "accepted_by_queue",
            "authorization_receipt_sha256": authorization_sha256,
            "queue_receipt_sha256": _sha256(receipt_path),
            "queue_file_sha256_after": queue_file_sha256_after,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_create_only(receipt_root / "COMPLETE.json", complete)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    for path in receipt_root.iterdir():
        path.chmod(0o444)
    receipt_root.chmod(0o555)
    print(json.dumps(complete, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
