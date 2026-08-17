#!/usr/bin/env python3
"""Bind the revision-118 activated learner as the resumable champion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.loop_state.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    state = json.loads(state_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "poke_bot.marnie_latent_lookahead_owner_activation/v1":
        raise RuntimeError("activation receipt schema mismatch")
    if receipt.get("status") != "activated_for_restarted_iteration_6":
        raise RuntimeError("activation receipt is not complete")
    activated = dict(receipt.get("activated_checkpoint") or {})
    path = Path(str(activated.get("path") or "")).resolve()
    if digest(path) != activated.get("digest"):
        raise RuntimeError("activated checkpoint digest mismatch")
    if int(state.get("last_completed_iteration", -1)) != 5 or int(state.get("next_iteration", -1)) != 6:
        raise RuntimeError("resume boundary moved")
    if dict(state.get("learner") or {}) != activated:
        raise RuntimeError("loop learner is not the activated checkpoint")
    champion = dict(state.get("champion") or {})
    if champion == activated:
        print(json.dumps({"status": "already_finalized", "champion": activated}, sort_keys=True))
        return 0
    expected_parent = dict(receipt.get("protected_parent") or {})
    if champion != expected_parent:
        raise RuntimeError("current champion is not the protected rollback parent")
    state["pre_activation_champion"] = champion
    state["champion"] = activated
    state["accepted_policy_generation"] = 15
    state["resume_shape_authority"] = {
        "schema": "poke_bot.marnie_latent_resume_shape/v1",
        "owner_decision_revision": 118,
        "activation_receipt": str(receipt_path),
        "activated_checkpoint": activated,
        "rollback_parent": expected_parent,
    }
    temporary = state_path.with_name(state_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, state_path)
    print(json.dumps({"status": "finalized", "champion": activated, "rollback": expected_parent}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
