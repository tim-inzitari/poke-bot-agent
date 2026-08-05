#!/usr/bin/env python3
"""Create the checksum-bound zero-guide Marnie runtime-registry derivative."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"


def digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{value}"


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_idempotent(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing immutable output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_registry.expanduser().resolve()
    output = args.output_registry.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    registry = read(base)
    rows = dict(registry.get("specialists") or {})
    row = dict(rows.get(SPECIALIST_ID) or {})
    if not row or float(row.get("guide_loss_weight", -1.0)) != 0.05:
        raise RuntimeError("expected checksum-registered Marnie guide parent")
    row.update(
        {
            "guide_loss_weight": 0.0,
            "guide_retired": True,
            "guide_retirement_revision": 140,
            "guide_target_generation_required": False,
            "guide_conditioned_losses_enabled": False,
            "guide_action_influence": False,
            "guide_historical_artifacts": "audit_only",
        }
    )
    rows[SPECIALIST_ID] = row
    registry["specialists"] = rows
    registry["marnie_guide_retirement"] = {
        "owner_revision": 140,
        "status": "permanently_retired",
        "weight": 0.0,
    }
    write_idempotent(output, registry)
    proof = {
        "schema": "poke_bot.marnie_guide_retirement/v1",
        "status": "staged_for_postbootstrap_and_all_later_marnie_training",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_revision": 140,
        "specialist_id": SPECIALIST_ID,
        "base_registry": str(base),
        "base_registry_sha256": digest(base),
        "runtime_registry": str(output),
        "runtime_registry_sha256": digest(output),
        "guide_weight": 0.0,
        "guide_target_generation_required": False,
        "guide_conditioned_losses_enabled": False,
        "guide_action_influence": False,
        "all_non_guide_heads_and_matchup_adapters_unchanged": True,
    }
    # The timestamp belongs to the immutable receipt, so idempotence is based
    # on its already-validated content rather than regenerating it.
    if receipt.exists():
        existing = read(receipt)
        if (
            existing.get("schema") != proof["schema"]
            or existing.get("runtime_registry_sha256") != proof["runtime_registry_sha256"]
            or existing.get("owner_revision") != 140
        ):
            raise RuntimeError("existing retirement receipt changed")
    else:
        write_idempotent(receipt, proof)
    print(json.dumps(read(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
