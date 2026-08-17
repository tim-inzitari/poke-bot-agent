#!/usr/bin/env python3
"""Create one immutable V6 runtime derivative of a frozen V5 model family."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
    load_slot_registry,
    registry_digest,
)
from poke_bot.pure_rl.model_registry import (
    freeze_model,
    sha256,
    verify_frozen_model,
)
from scripts.canary_migrate_matchup_v6 import (
    RECEIPT_SCHEMA as CANARY_RECEIPT_SCHEMA,
    run_canary,
)


RECEIPT_SCHEMA = "poke_bot.matchup_adapter_v6_runtime_family/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_v6_registry(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict((payload.get("extra") or {}).get("matchup_adapter_config") or {})
    if config.get("format") != V6_ADAPTER_CHECKPOINT_FORMAT:
        raise RuntimeError("runtime derivative is not a V6 checkpoint")
    registry = dict(config.get("slot_registry") or {})
    if not registry:
        raise RuntimeError("V6 runtime derivative lacks its embedded registry")
    return registry


def validate_runtime_family(
    *,
    source_family: Path,
    target_family: Path,
    registry_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    source = verify_frozen_model(source_family)
    target = verify_frozen_model(target_family)
    registry = load_slot_registry(registry_path)
    receipt = _read(receipt_path)
    checkpoint = Path(str(target["model_path"])).resolve()
    embedded = _checkpoint_v6_registry(checkpoint)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("source_family") != str(source_family.resolve())
        or receipt.get("source_checkpoint_sha256")
        != source["checkpoint_digest"]
        or receipt.get("target_family") != str(target_family.resolve())
        or receipt.get("target_checkpoint_sha256")
        != target["checkpoint_digest"]
        or receipt.get("registry") != str(registry_path.resolve())
        or receipt.get("registry_digest") != registry_digest(registry)
        or embedded != registry
    ):
        raise RuntimeError("V6 runtime-family receipt identity changed")
    return receipt


def materialize_runtime_family(
    *,
    source_family: Path,
    target_family: Path,
    registry_path: Path,
    staging_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    source_family = source_family.expanduser().resolve()
    target_family = target_family.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    staging_root = staging_root.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    source = verify_frozen_model(source_family)
    source_checkpoint = Path(str(source["model_path"])).resolve()
    if target_family.parent != source_family.parent:
        raise RuntimeError("V6 derivative must remain in the protected registry")
    if target_family == source_family:
        raise RuntimeError("V6 derivative may not replace its source family")
    if target_family.is_dir() or receipt_path.is_file():
        if not target_family.is_dir() or not receipt_path.is_file():
            raise RuntimeError("partial V6 runtime-family publication detected")
        return validate_runtime_family(
            source_family=source_family,
            target_family=target_family,
            registry_path=registry_path,
            receipt_path=receipt_path,
        )

    staging_root.mkdir(parents=True, exist_ok=True)
    derivative = staging_root / f"{target_family.name}.model.pt"
    canary_receipt = staging_root / f"{target_family.name}.canary.json"
    if derivative.exists() or canary_receipt.exists():
        raise RuntimeError("unreceipted V6 runtime-family staging artifact exists")
    canary = run_canary(
        source=source_checkpoint,
        registry_path=registry_path,
        output=derivative,
        receipt_path=canary_receipt,
    )
    if (
        canary.get("schema") != CANARY_RECEIPT_SCHEMA
        or canary.get("status") != "passed"
        or canary.get("source_checkpoint_sha256")
        != source["checkpoint_digest"]
    ):
        raise RuntimeError("V6 runtime-family canary did not pass")

    frozen = freeze_model(
        registry_root=target_family.parent,
        family=target_family.name,
        display_name=(
            f"{source.get('display_name') or source_family.name} "
            "(Matchup Adapter V6 runtime derivative)"
        ),
        checkpoint=derivative,
        expected_digest=str(canary["output_checkpoint_sha256"]),
        provenance={
            "kind": "matchup_adapter_v6_runtime_derivative",
            "source_family": str(source_family),
            "source_checkpoint_digest": source["checkpoint_digest"],
            "source_family_manifest_sha256": sha256(
                source_family / "manifest.json"
            ),
            "canary_receipt": str(canary_receipt),
            "canary_receipt_sha256": sha256(canary_receipt),
            "base_policy_and_optimizer_bit_exact": True,
            "source_family_immutable": True,
        },
        evidence={
            "migration_canary": canary,
            "training_evidence_inherited_from_source": True,
        },
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_family": str(source_family),
        "source_checkpoint_sha256": source["checkpoint_digest"],
        "target_family": str(target_family),
        "target_checkpoint_sha256": frozen["checkpoint_digest"],
        "registry": str(registry_path),
        "registry_digest": registry_digest(load_slot_registry(registry_path)),
        "canary_receipt": str(canary_receipt),
        "canary_receipt_sha256": sha256(canary_receipt),
        "source_family_immutable": True,
        "ordinary_roster_edits_change_tensor_shape": False,
    }
    _atomic_json(receipt_path, receipt)
    return validate_runtime_family(
        source_family=source_family,
        target_family=target_family,
        registry_path=registry_path,
        receipt_path=receipt_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-family", type=Path, required=True)
    parser.add_argument("--target-family", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    function = validate_runtime_family if args.validate_only else materialize_runtime_family
    kwargs = {
        "source_family": args.source_family,
        "target_family": args.target_family,
        "registry_path": args.registry,
        "receipt_path": args.receipt,
    }
    if not args.validate_only:
        kwargs["staging_root"] = args.staging_root
    print(json.dumps(function(**kwargs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
