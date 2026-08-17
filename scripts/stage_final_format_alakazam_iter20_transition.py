#!/usr/bin/env python3
"""Create one immutable iter-20 Alakazam terminal-handler registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.specialist_runtime_registry/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_iter20_transition_stage/v1"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace different staged artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stage(
    source: Path,
    output: Path,
    receipt: Path,
    *,
    owner_decision_revision: int = 104,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("source runtime registry schema changed")
    specialists = dict(payload.get("specialists") or {})
    alakazam = dict(specialists.get("alakazam") or {})
    common = dict(payload.get("pass_handler") or {})
    if alakazam.get("status") != "ready":
        raise RuntimeError("Alakazam is not the ready specialist")
    payload["minimum_terminal_iteration"] = 20
    payload["iteration_ceiling"] = 20
    alakazam["minimum_terminal_iteration"] = 20
    alakazam["iteration_ceiling"] = 20
    specialists["alakazam"] = alakazam
    payload["specialists"] = specialists
    common["ceiling_behavior"] = "freeze_submit_and_continue_without_false_pass"
    payload["pass_handler"] = common
    payload["owner_decision_revision"] = int(owner_decision_revision)
    _write_once(output, payload)
    staged = {
        "schema": RECEIPT_SCHEMA,
        "status": "staged_waiting_for_exact_iter_00020_commit",
        "owner_decision_revision": int(owner_decision_revision),
        "source_registry": str(source),
        "source_registry_sha256": _sha256(source),
        "staged_registry": str(output),
        "staged_registry_sha256": _sha256(output),
        "minimum_terminal_iteration": 20,
        "ceiling_completed_iteration": 20,
        "next_collection_allowed": False,
        "ceiling_acceptance_is_measured_gate_pass": False,
        "next_specialist": "marnie-s-grimmsnarl-ex",
    }
    _write_once(receipt, staged)
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--owner-decision-revision", type=int, default=104)
    args = parser.parse_args()
    print(
        json.dumps(
            stage(
                args.source,
                args.output,
                args.receipt,
                owner_decision_revision=args.owner_decision_revision,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
