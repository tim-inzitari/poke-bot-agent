#!/usr/bin/env python3
"""Clear stale pre-bootstrap heldout evidence at the exact iter-10 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    loop_path = args.loop_state.expanduser().resolve()
    activation_path = args.activation_receipt.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    if receipt_path.exists():
        existing = read(receipt_path)
        if existing.get("status") != "repaired":
            raise RuntimeError("existing repair receipt is invalid")
        print(json.dumps(existing, sort_keys=True))
        return 0

    activation = read(activation_path)
    loop = read(loop_path)
    activated = dict(activation.get("activated_bootstrap") or {})
    heldout = dict(loop.get("heldout_champion") or {})
    evidence = dict(loop.get("heldout_champion_evidence") or {})
    prior_digest = str(
        (activation.get("source_iteration9_learner") or {}).get("digest") or ""
    )
    if (
        activation.get("schema")
        != "poke_bot.marnie_postupload_bootstrap_activation/v1"
        or activation.get("status") != "activated"
        or int(loop.get("last_completed_iteration", -1)) != 9
        or int(loop.get("next_iteration", -1)) != 10
        or heldout.get("digest") != activated.get("digest")
        or evidence.get("checkpoint_digest") != prior_digest
        or heldout.get("digest") == prior_digest
    ):
        raise RuntimeError("stale heldout evidence repair boundary is invalid")

    before_sha = sha256(loop_path)
    evidence_sha = canonical_sha256(evidence)
    loop["heldout_champion_evidence"] = {}
    loop["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(loop_path, loop)
    receipt = {
        "schema": "poke_bot.marnie_postupload_heldout_evidence_repair/v1",
        "status": "repaired",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_revision": 153,
        "loop_state": str(loop_path),
        "loop_state_before_sha256": before_sha,
        "loop_state_after_sha256": sha256(loop_path),
        "activation_receipt": str(activation_path),
        "activation_receipt_sha256": sha256(activation_path),
        "cleared_evidence_sha256": evidence_sha,
        "activated_checkpoint_sha256": heldout["digest"],
        "prior_evaluated_checkpoint_sha256": prior_digest,
        "proof": {
            "no_evidence_transferred_between_checkpoints": True,
            "new_exact_gate_required": True,
            "guide_runtime_authority_changed": False,
        },
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
