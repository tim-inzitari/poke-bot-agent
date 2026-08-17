from __future__ import annotations

import json
from pathlib import Path

from poke_bot.r274_bootstrap_handoff import file_identity
from scripts.materialize_marnie_iteration9_upload_trigger import (
    QUEUE_IDENTITY_FIELDS,
    sha256,
)
from scripts.materialize_r274_bootstrap_upload import SCHEMA, materialize
from scripts.stage_r274_bootstrap_submission import CANDIDATE_ID, SCHEMA as STAGE_SCHEMA


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_accepted_direct_policy_upload_materializes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "r274.pt"
    checkpoint.write_bytes(b"r274")
    bundle = tmp_path / "submission.tar.gz"
    bundle.write_bytes(b"bundle")
    identity = {field: None for field in QUEUE_IDENTITY_FIELDS}
    identity.update(
        {
            "specialist_id": "alakazam",
            "copy_number": 1,
            "turn_order_preference": "first_if_allowed",
            "label": "alakazam bootstrap",
            "checkpoint_checksum": sha256(checkpoint),
            "model_checksum": sha256(checkpoint),
            "competition": "pokemon-tcg-ai-battle",
            "file": str(bundle.resolve()),
            "file_sha256": sha256(bundle),
            "iteration": 0,
            "rtp_mode": "off",
            "search_assets_packaged": False,
        }
    )
    stage = _json(
        tmp_path / "stage.json",
        {
            "schema": STAGE_SCHEMA,
            "status": "queued",
            "candidate_id": CANDIDATE_ID,
            "boundary": "bootstrap",
            "checkpoint": file_identity(checkpoint),
            "queue_entry": dict(identity),
            "direct_policy_only": True,
            "rtp_enabled": False,
            "search_assets_packaged": False,
            "tactical_route_enabled": False,
        },
    )
    nonce = "r274-test"
    auth = _json(
        tmp_path / "attempts/auth.authorization-consumed.json",
        {
            "schema": "poke_bot.kaggle_submission_authorization/v1",
            "nonce": nonce,
            "consumed_before_upload": True,
            "remaining_uses": 0,
            "submission_file_checksum": sha256(bundle),
            "frozen_checkpoint_checksum": sha256(checkpoint),
        },
    )
    _json(
        tmp_path / "attempts/attempt.json",
        {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": nonce,
            "returncode": 0,
            "authorization_consumed": str(auth),
            "identity": {
                "file": str(bundle.resolve()),
                "file_sha256": sha256(bundle),
                "competition": "pokemon-tcg-ai-battle",
                "message": "alakazam bootstrap",
            },
        },
    )
    queue_row = {
        **identity,
        "queue_status": "accepted",
        "kaggle_status": "COMPLETE",
        "failure_reason": None,
        "submission_id": 123456,
        "one_shot_authorization_nonce": nonce,
    }
    queue = _json(
        tmp_path / "queue.json",
        {"schema": "poke_bot.kaggle_submission_queue/v1", "queue": [queue_row]},
    )
    output = tmp_path / "upload.json"

    result = materialize(
        queue_path=queue,
        stage_receipt=stage,
        attempt_receipts=tmp_path / "attempts",
        output=output,
    )

    assert result["status"] == "materialized"
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "submitted"
    assert receipt["remote_submission_id"] == 123456
    assert receipt["direct_policy_only"] is True
    assert receipt["rtp_enabled"] is False
