#!/usr/bin/env python3
"""Remove a prematurely registered specialist from future gate inputs.

This is deliberately narrow and fail-closed. It replaces an inactive future
gate/registry with the current authoritative predecessor set, removes one
exact stale opponent from the shared baseline manifest, preserves the stale
files in a checksum-addressed quarantine directory, and writes an idempotent
receipt. It never deletes checkpoint or submission evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "poke_bot.unpassed_specialist_quarantine/v1"
REGISTRY_SCHEMA = "poke_bot.frozen_specialist_registry/v1"
GATE_SCHEMA = "poke_bot.competition_gate_program/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _frozen_gate_rows(gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in ((gate.get("next_gate") or {}).get("roster") or [])
        if isinstance(row, dict) and row.get("frozen_specialist") is True
    ]


def _registry_ids(registry: dict[str, Any]) -> set[str]:
    return {
        str(row.get("opponent_id") or "")
        for row in (registry.get("specialists") or [])
        if isinstance(row, dict)
    }


def _validate_pair(gate: dict[str, Any], registry: dict[str, Any]) -> None:
    if (
        gate.get("schema") != GATE_SCHEMA
        or registry.get("schema") != REGISTRY_SCHEMA
    ):
        raise RuntimeError("gate or frozen-specialist registry schema changed")
    registry_ids = _registry_ids(registry)
    gate_rows = _frozen_gate_rows(gate)
    gate_ids = {str(row.get("opponent_id") or "") for row in gate_rows}
    if (
        "" in registry_ids
        or "" in gate_ids
        or registry_ids != gate_ids
        or len(gate_rows) != len(gate_ids)
    ):
        raise RuntimeError("gate and frozen-specialist registry disagree")
    evaluation = dict((gate.get("next_gate") or {}).get("evaluation") or {})
    roster = list((gate.get("next_gate") or {}).get("roster") or [])
    if (
        int(evaluation.get("games_total", -1)) != 250 * len(roster)
        or int(evaluation.get("games_per_opponent", -1)) != 250
        or int(evaluation.get("seat0_games_per_opponent", -1)) != 125
        or int(evaluation.get("seat1_games_per_opponent", -1)) != 125
        or any(row.get("tier") != "S+" for row in gate_rows)
    ):
        raise RuntimeError("gate allocation or S+ tier contract changed")


def quarantine(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "source_gate": args.source_gate.resolve(),
        "source_registry": args.source_registry.resolve(),
        "target_gate": args.target_gate.resolve(),
        "target_registry": args.target_registry.resolve(),
        "baseline_manifest": args.baseline_manifest.resolve(),
    }
    receipt = args.receipt.resolve()
    if receipt.exists():
        existing = _read(receipt)
        if (
            existing.get("schema") != RECEIPT_SCHEMA
            or existing.get("opponent_id") != args.opponent_id
        ):
            raise RuntimeError("quarantine receipt identity changed")
        for name, expected in (existing.get("post_sha256") or {}).items():
            if _digest(paths[name]) != expected:
                raise RuntimeError(f"quarantined path drifted: {name}")
        return existing

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_gate = _read(paths["source_gate"])
    source_registry = _read(paths["source_registry"])
    target_gate = _read(paths["target_gate"])
    target_registry = _read(paths["target_registry"])
    manifest = _read(paths["baseline_manifest"])
    _validate_pair(source_gate, source_registry)
    _validate_pair(target_gate, target_registry)

    source_ids = _registry_ids(source_registry)
    target_ids = _registry_ids(target_registry)
    if (
        args.opponent_id in source_ids
        or target_ids != source_ids | {args.opponent_id}
    ):
        raise RuntimeError("target is not the exact one-opponent premature extension")
    stale_registry = [
        row
        for row in target_registry["specialists"]
        if row.get("opponent_id") == args.opponent_id
    ]
    stale_gate = [
        row
        for row in _frozen_gate_rows(target_gate)
        if row.get("opponent_id") == args.opponent_id
    ]
    if (
        len(stale_registry) != 1
        or len(stale_gate) != 1
        or stale_registry[0].get("specialist_id") != args.specialist_id
        or stale_registry[0].get("checkpoint_digest") != args.checkpoint_digest
        or stale_gate[0].get("frozen_checkpoint_digest")
        != args.checkpoint_digest
    ):
        raise RuntimeError("premature specialist identity or checksum changed")
    agents = list(manifest.get("agents") or [])
    stale_agents = [row for row in agents if row.get("id") == args.opponent_id]
    if len(stale_agents) != 1:
        raise RuntimeError("baseline manifest does not contain one stale opponent")

    before = {name: _digest(path) for name, path in paths.items()}
    quarantine_root = receipt.parent / (
        f"{args.opponent_id}.{args.checkpoint_digest.removeprefix('sha256:')[:16]}"
    )
    if quarantine_root.exists():
        raise RuntimeError("quarantine directory already exists without receipt")
    quarantine_root.mkdir(parents=True)
    for name in ("target_gate", "target_registry", "baseline_manifest"):
        shutil.copy2(paths[name], quarantine_root / f"{name}.json")

    cleaned_manifest = dict(manifest)
    cleaned_manifest["agents"] = [
        row for row in agents if row.get("id") != args.opponent_id
    ]
    _atomic_write(paths["target_gate"], paths["source_gate"].read_bytes())
    _atomic_write(paths["target_registry"], paths["source_registry"].read_bytes())
    _atomic_json(paths["baseline_manifest"], cleaned_manifest)

    after = {name: _digest(path) for name, path in paths.items()}
    payload = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specialist_id": args.specialist_id,
        "opponent_id": args.opponent_id,
        "checkpoint_digest": args.checkpoint_digest,
        "reason": (
            "specialist had not completed the iteration-25 floor, exact S+ "
            "gate, freeze/registration, and one-copy handoff contract"
        ),
        "preserved_evidence": str(quarantine_root),
        "source_predecessor_opponents": sorted(source_ids),
        "pre_sha256": before,
        "post_sha256": after,
        "checkpoint_artifacts_deleted": False,
        "submission_evidence_deleted": False,
    }
    _atomic_json(receipt, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--target-gate", type=Path, required=True)
    parser.add_argument("--target-registry", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--opponent-id", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(quarantine(_parse_args()), indent=2, sort_keys=True))
