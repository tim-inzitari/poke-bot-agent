from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from scripts.kaggle_cli_guard import (
    AUTH_SCHEMA,
    GO_FIRST_ATTESTATION_SCHEMA,
    GO_FIRST_VERIFIED_CASES,
    TURN_ORDER_ATTESTATION_SCHEMA,
    _is_submit,
    _validate_authorization,
)


def _write_go_first_attestation(bundle: Path, digest: str) -> None:
    Path(str(bundle) + ".go-first-verified.json").write_text(
        json.dumps(
            {
                "schema": GO_FIRST_ATTESTATION_SCHEMA,
                "file_sha256": digest,
                "go_first_if_offered": True,
                "verified_cases": sorted(GO_FIRST_VERIFIED_CASES),
            }
        ),
        encoding="utf-8",
    )


def test_only_submission_command_is_guarded() -> None:
    assert _is_submit(["competitions", "submit", "-c", "x"])
    assert _is_submit(["--quiet", "competition", "submit", "-c", "x"])
    assert not _is_submit(["competitions", "submissions", "-c", "x"])
    assert not _is_submit(["datasets", "download", "-d", "x"])


def test_authorization_is_bound_to_exact_upload_identity(tmp_path: Path) -> None:
    bundle = tmp_path / "submission.tar.gz"
    bundle.write_bytes(b"exact bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    _write_go_first_attestation(bundle, digest)
    argv = [
        "competitions",
        "submit",
        "-c",
        "pokemon-tcg-ai-battle",
        "-f",
        str(bundle),
        "-m",
        "one exact submission",
    ]
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": "approval-1",
        "expires_at_epoch": time.time() + 60,
        "competition": "pokemon-tcg-ai-battle",
        "file_sha256": digest,
        "message": "one exact submission",
    }

    valid, reason, _details = _validate_authorization(authorization, argv)
    assert valid is True
    assert reason == "authorized"

    authorization["remaining_uses"] = 0
    assert _validate_authorization(authorization, argv)[0] is False


def test_second_preference_requires_matching_digest_bound_attestation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "submission.tar.gz"
    bundle.write_bytes(b"second-preferring bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    Path(str(bundle) + ".go-first-verified.json").write_text(
        json.dumps(
            {
                "schema": TURN_ORDER_ATTESTATION_SCHEMA,
                "file_sha256": digest,
                "turn_order_preference": "second_if_allowed",
                "go_first_if_offered": False,
                "go_second_if_offered": True,
                "verified_cases": sorted(GO_FIRST_VERIFIED_CASES),
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "competitions",
        "submit",
        "-c",
        "pokemon-tcg-ai-battle",
        "-f",
        str(bundle),
        "-m",
        "teal mask copy 2 second",
    ]
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": "teal-second",
        "expires_at_epoch": time.time() + 60,
        "competition": "pokemon-tcg-ai-battle",
        "file_sha256": digest,
        "message": "teal mask copy 2 second",
        "turn_order_preference": "second_if_allowed",
    }

    assert _validate_authorization(authorization, argv)[0] is True
    authorization["turn_order_preference"] = "first_if_allowed"
    assert _validate_authorization(authorization, argv)[0] is False
    authorization["remaining_uses"] = 1
    authorization["message"] = "different"
    assert _validate_authorization(authorization, argv)[0] is False


def test_submission_without_digest_bound_go_first_attestation_is_blocked(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "submission.tar.gz"
    bundle.write_bytes(b"exact bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    argv = [
        "competitions",
        "submit",
        "-c",
        "pokemon-tcg-ai-battle",
        "-f",
        str(bundle),
        "-m",
        "one exact submission",
    ]
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": "approval-1",
        "expires_at_epoch": time.time() + 60,
        "competition": "pokemon-tcg-ai-battle",
        "file_sha256": digest,
        "message": "one exact submission",
    }
    valid, reason, details = _validate_authorization(authorization, argv)
    assert valid is False
    assert "go_first_attestation" in reason
    assert details["checks"]["go_first_attestation"] is False

    _write_go_first_attestation(bundle, "sha256:" + "0" * 64)
    assert _validate_authorization(authorization, argv)[0] is False

    _write_go_first_attestation(bundle, digest)
    assert _validate_authorization(authorization, argv)[0] is True


def test_authorization_file_shape_is_json_serializable() -> None:
    payload = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": "single-use",
    }
    assert json.loads(json.dumps(payload)) == payload
