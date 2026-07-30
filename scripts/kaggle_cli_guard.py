#!/usr/bin/env python3
"""Deny Kaggle submissions unless a matching one-shot grant exists.

All non-mutating Kaggle CLI commands continue to work.  A submission grant is
consumed *before* the network upload begins, so a stalled command can never be
retried under the same approval.  Failed uploads also require a new explicit
grant; this is intentional fail-closed behavior.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_AUTHORIZATION = Path(
    "/home/inzi/.config/pokebot/kaggle-submission-authorization.json"
)
DEFAULT_LOCK = Path("/home/inzi/.local/state/pokebot/kaggle-submission.lock")
DEFAULT_RECEIPTS = Path(
    "/home/inzi/.local/state/pokebot/kaggle-submission-attempts"
)
AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
GO_FIRST_ATTESTATION_SCHEMA = "poke_bot.submission_go_first_attestation/v1"
TURN_ORDER_ATTESTATION_SCHEMA = "poke_bot.submission_turn_order_attestation/v1"
GO_FIRST_VERIFIED_CASES = {
    "integer_enum",
    "string_enum_reversed_options",
    "live_engine_prompt",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_go_first_attestation(
    file_path: Path,
    file_digest: str,
    expected_preference: str = "first_if_allowed",
) -> tuple[bool, str, dict[str, Any]]:
    """Require a digest-bound proof for the authorized turn-order profile."""

    if expected_preference not in {
        "first_if_allowed",
        "second_if_allowed",
    }:
        return False, "invalid expected turn-order preference", {}
    receipt_path = Path(str(file_path) + ".go-first-verified.json")
    receipt = _read_json(receipt_path)
    verified_cases = {
        str(item) for item in (receipt.get("verified_cases") or [])
    }
    schema = str(receipt.get("schema") or "")
    legacy_first = schema == GO_FIRST_ATTESTATION_SCHEMA
    declared_preference = str(
        receipt.get("turn_order_preference")
        or ("first_if_allowed" if legacy_first else "")
    )
    checks = {
        "schema": schema
        in {GO_FIRST_ATTESTATION_SCHEMA, TURN_ORDER_ATTESTATION_SCHEMA},
        "file_sha256": str(receipt.get("file_sha256") or "") == file_digest,
        "turn_order_preference": declared_preference == expected_preference,
        "go_first_if_offered": (
            receipt.get("go_first_if_offered")
            is (expected_preference == "first_if_allowed")
        ),
        "go_second_if_offered": (
            legacy_first
            and expected_preference == "first_if_allowed"
            or receipt.get("go_second_if_offered")
            is (expected_preference == "second_if_allowed")
        ),
        "verified_cases": GO_FIRST_VERIFIED_CASES <= verified_cases,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return (
        not failed,
        ", ".join(failed) if failed else "verified",
        {
            "path": str(receipt_path),
            "schema": receipt.get("schema"),
            "file_sha256": receipt.get("file_sha256"),
            "turn_order_preference": declared_preference,
            "go_first_if_offered": receipt.get("go_first_if_offered"),
            "go_second_if_offered": receipt.get("go_second_if_offered"),
            "verified_cases": sorted(verified_cases),
            "checks": checks,
        },
    )


def _value(argv: list[str], short: str, long: str) -> str | None:
    for index, token in enumerate(argv):
        if token in {short, long} and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(long + "="):
            return token.split("=", 1)[1]
    return None


def _is_submit(argv: list[str]) -> bool:
    lowered = [token.lower() for token in argv]
    return any(
        lowered[index] in {"competition", "competitions"}
        and lowered[index + 1] == "submit"
        for index in range(len(lowered) - 1)
    )


def _validate_authorization(
    authorization: dict[str, Any], argv: list[str]
) -> tuple[bool, str, dict[str, Any]]:
    competition = _value(argv, "-c", "--competition")
    file_raw = _value(argv, "-f", "--file")
    message = _value(argv, "-m", "--message")
    if not competition or not file_raw or message is None:
        return False, "submission argv is missing competition, file, or message", {}
    file_path = Path(file_raw).expanduser().resolve()
    try:
        digest = _sha256(file_path)
    except OSError as exc:
        return False, f"submission file cannot be hashed: {exc}", {}
    expected_turn_order = str(
        authorization.get("turn_order_preference") or "first_if_allowed"
    )
    go_first_valid, go_first_reason, go_first_details = (
        _validate_go_first_attestation(
            file_path,
            digest,
            expected_preference=expected_turn_order,
        )
    )

    now = time.time()
    expires_at = authorization.get("expires_at_epoch")
    expected_digest = str(authorization.get("file_sha256") or "")
    expected_message = str(authorization.get("message") or "")
    expected_competition = str(authorization.get("competition") or "")
    nonce = str(authorization.get("nonce") or "")
    checks = {
        "schema": authorization.get("schema") == AUTH_SCHEMA,
        "explicit_user_approval": authorization.get("explicit_user_approval") is True,
        "one_shot": int(authorization.get("remaining_uses") or 0) == 1,
        "nonce": bool(nonce),
        "not_expired": isinstance(expires_at, (int, float)) and now <= float(expires_at),
        "competition": expected_competition == competition,
        "file_sha256": bool(expected_digest) and expected_digest == digest,
        "message": expected_message == message,
        "turn_order_preference": expected_turn_order
        in {"first_if_allowed", "second_if_allowed"},
        "go_first_attestation": go_first_valid,
    }
    failed = [name for name, passed in checks.items() if not passed]
    details = {
        "competition": competition,
        "file": str(file_path),
        "file_sha256": digest,
        "message": message,
        "turn_order_preference": expected_turn_order,
        "nonce": nonce,
        "checks": checks,
        "go_first_attestation": go_first_details,
    }
    reason = ", ".join(failed) if failed else "authorized"
    if not go_first_valid:
        reason += f" ({go_first_reason})"
    return not failed, reason, details


def _receipt_path(receipts: Path, nonce: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_nonce = "".join(ch for ch in nonce if ch.isalnum() or ch in "-_")
    return receipts / f"{stamp}-{safe_nonce or uuid.uuid4().hex}.json"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    argv = list(parsed.remainder)
    if argv and argv[0] == "--":
        argv = argv[1:]
    real = parsed.real.resolve()
    if not _is_submit(argv):
        os.execv(str(real), [str(real), *argv])

    parsed.lock.parent.mkdir(parents=True, exist_ok=True)
    with parsed.lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        authorization = _read_json(parsed.authorization)
        valid, reason, details = _validate_authorization(authorization, argv)
        if not valid:
            print(
                "Kaggle submission BLOCKED: no matching unused one-shot explicit "
                f"authorization ({reason}).",
                file=sys.stderr,
            )
            return 73

        nonce = str(details["nonce"])
        consumed_path = _receipt_path(parsed.receipts, nonce).with_suffix(
            ".authorization-consumed.json"
        )
        consumed = {
            **authorization,
            "remaining_uses": 0,
            "consumed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "consumed_before_upload": True,
            "argv_identity": details,
        }
        _atomic_json(consumed_path, consumed)
        parsed.authorization.unlink()

        started = time.time()
        completed = subprocess.run([str(real), *argv], check=False).returncode
        receipt = {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": nonce,
            "authorization_consumed": str(consumed_path),
            "started_at_epoch": started,
            "completed_at_epoch": time.time(),
            "returncode": completed,
            "identity": details,
            "retry_requires_new_explicit_user_approval": True,
        }
        _atomic_json(_receipt_path(parsed.receipts, nonce), receipt)
        return int(completed)


if __name__ == "__main__":
    raise SystemExit(main())
