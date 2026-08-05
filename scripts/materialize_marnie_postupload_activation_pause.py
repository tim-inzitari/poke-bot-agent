#!/usr/bin/env python3
"""Bind the iter-9 upload wait to the exact family activation request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    PAUSE_SCHEMA,
    _write_exclusive,
    sha256,
    validate_activation_request,
    validate_iteration9_upload_trigger,
)


def materialize(
    *,
    service: str,
    wait_pause: Path,
    trigger: Path,
    request: Path,
    output: Path,
) -> dict:
    state = subprocess.run(
        ["systemctl", "--user", "is-active", service],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    if state in {"active", "activating", "reloading"}:
        raise RuntimeError(f"managed trainer is still {state}")
    waiting = json.loads(wait_pause.read_text(encoding="utf-8"))
    trigger_payload = json.loads(trigger.read_text(encoding="utf-8"))
    validate_iteration9_upload_trigger(trigger_payload)
    trigger_checkpoint = dict(
        (trigger_payload.get("bindings") or {}).get("checkpoint") or {}
    )
    if (
        waiting.get("schema") != PAUSE_SCHEMA
        or int(waiting.get("owner_revision", -1)) != 134
        or int(waiting.get("committed_iteration", -1)) != 9
        or int(waiting.get("target_iteration", -1)) != 10
        or waiting.get("next_collection_started") is not False
        or int(waiting.get("restart_prevent_status", -1)) != 75
        or waiting.get("pause_reason")
        != "awaiting_successful_checksum_exact_iteration9_upload"
        or Path(str(waiting.get("expected_trigger_path") or "")).resolve()
        != trigger.resolve()
        or waiting.get("learner_sha256")
        != str(trigger_checkpoint.get("sha256") or "")
    ):
        raise RuntimeError("iteration-9 upload-wait pause identity changed")
    request_payload = json.loads(request.read_text(encoding="utf-8"))
    validate_activation_request(
        request_payload,
        expected_learner_digest=str(waiting["learner_sha256"]),
    )
    request_trigger = dict(request_payload.get("trigger") or {})
    if (
        Path(str(request_trigger.get("path") or "")).resolve()
        != trigger.resolve()
        or str(request_trigger.get("sha256") or "") != sha256(trigger)
    ):
        raise RuntimeError("activation request binds another iter-9 upload")
    payload = {
        "schema": PAUSE_SCHEMA,
        "owner_revision": 134,
        "committed_iteration": 9,
        "target_iteration": 10,
        "learner_sha256": str(waiting["learner_sha256"]),
        "trigger_path": str(trigger.resolve()),
        "trigger_sha256": sha256(trigger),
        "request_path": str(request.resolve()),
        "request_sha256": sha256(request),
        "activation_evidence_complete": True,
        "pause_reason": "ready_for_atomic_activation_and_weighted_bootstrap",
        "next_collection_started": False,
        "restart_prevent_status": 75,
        "managed_training_active": False,
        "derived_from_pause": {
            "path": str(wait_pause.resolve()),
            "sha256": sha256(wait_pause),
        },
    }
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("post-upload activation pause identity changed")
        return existing
    _write_exclusive(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--wait-pause", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        service=args.service,
        wait_pause=args.wait_pause.expanduser().resolve(),
        trigger=args.trigger.expanduser().resolve(),
        request=args.request.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
