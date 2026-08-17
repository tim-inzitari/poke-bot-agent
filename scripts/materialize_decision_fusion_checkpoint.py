#!/usr/bin/env python3
"""Create an immutable zero-safe decision-fusion child checkpoint.

The child preserves every parent tensor and every existing Adam state entry.
New fusion parameters are appended to the existing optimizer group with no
moments, which is AdamW's exact fresh-parameter state.  Runtime inference stays
disabled; ordinary full-model training uses the fusion path and can therefore
produce the nonzero influence evidence required for a later serving boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import (  # noqa: E402
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


SCHEMA = "poke_bot.causal_decision_fusion_checkpoint_migration/v1"
MIGRATION_SCHEMA = "poke_bot.causal_decision_fusion_migration/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _trainable_names(model: torch.nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _expanded_optimizer_state(
    parent_state: dict[str, Any],
    *,
    old_names: list[str],
    new_names: list[str],
) -> dict[str, Any]:
    state = copy.deepcopy(parent_state)
    groups = list(state.get("param_groups") or [])
    if len(groups) != 1:
        raise RuntimeError(
            "decision-fusion migration currently requires exactly one ordinary "
            "optimizer parameter group"
        )
    old_ids = list(groups[0].get("params") or [])
    if len(old_ids) != len(old_names) or len(set(old_ids)) != len(old_ids):
        raise RuntimeError("parent optimizer parameter order is not canonical")
    if new_names[: len(old_names)] != old_names:
        raise RuntimeError("fusion migration changed existing parameter order")
    added_names = new_names[len(old_names) :]
    if not added_names or any(
        not name.startswith("decision_fusion.") for name in added_names
    ):
        raise RuntimeError("fusion parameters are not one appended optimizer suffix")
    next_id = max((int(value) for value in old_ids), default=-1) + 1
    added_ids = list(range(next_id, next_id + len(added_names)))
    groups[0]["params"] = [*old_ids, *added_ids]
    state["param_groups"] = groups
    return state


def materialize(
    *,
    parent: Path,
    output: Path,
    receipt: Path,
    fusion_width: int = 16,
) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("fusion migration outputs are immutable")
    checkpoint.assert_trusted_policy_checkpoint(parent)
    parent_digest = checkpoint.checkpoint_digest(parent)
    parent_payload = checkpoint.load_checkpoint(parent, map_location="cpu")
    parent_config = dict(parent_payload.get("model_config") or {})
    if parent_config.get("expanded_heads_enabled") is not True:
        raise RuntimeError("decision fusion requires expanded strategic heads")
    if parent_config.get("decision_fusion_enabled") is True:
        raise RuntimeError("parent checkpoint already contains decision fusion")
    if not isinstance(parent_payload.get("optimizer_state_dict"), dict):
        raise RuntimeError("active learner parent lacks optimizer state")

    parent_model = load_model_from_checkpoint(parent, device=torch.device("cpu"))
    old_names = _trainable_names(parent_model)
    old_state = {
        key: value.detach().cpu().clone()
        for key, value in parent_model.state_dict().items()
    }

    migrated_payload = copy.deepcopy(parent_payload)
    migrated_config = dict(parent_config)
    migrated_config.update(
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=False,
        decision_fusion_width=int(fusion_width),
    )
    migrated_payload["model_config"] = migrated_config
    extra = dict(migrated_payload.get("extra") or {})
    extra["decision_fusion_migration"] = {
        "schema": MIGRATION_SCHEMA,
        "target_schema": DECISION_FUSION_SCHEMA,
        "source_checkpoint": str(parent),
        "source_checkpoint_digest": parent_digest,
        "zero_safe_initialization": True,
        "runtime_enabled": False,
        "activation_scope": "active_specialist_training_warmup",
        "serving_eligible": False,
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
    }
    migrated_payload["extra"] = extra

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.migration.", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(migrated_payload, temporary)
        migrated_model = load_model_from_checkpoint(
            temporary, device=torch.device("cpu")
        )
        new_names = _trainable_names(migrated_model)
        migrated_state = migrated_model.state_dict()
        for key, value in old_state.items():
            torch.testing.assert_close(
                value, migrated_state[key], rtol=0, atol=0
            )
        fusion_keys = sorted(
            key for key in migrated_state if key.startswith("decision_fusion.")
        )
        if not fusion_keys:
            raise RuntimeError("migration materialized no fusion tensors")
        residual = migrated_state.get("decision_fusion.residual.2.weight")
        if residual is None or bool(torch.count_nonzero(residual).item()):
            raise RuntimeError("fusion migration is not zero-safe")

        migrated_payload["model_state_dict"] = migrated_state
        migrated_payload["optimizer_state_dict"] = _expanded_optimizer_state(
            dict(parent_payload["optimizer_state_dict"]),
            old_names=old_names,
            new_names=new_names,
        )
        provenance = dict(migrated_payload.get("provenance") or {})
        provenance["decision_fusion"] = migrated_model.decision_fusion_inventory()
        provenance["decision_fusion_migration"] = {
            "schema": SCHEMA,
            "source_checkpoint_digest": parent_digest,
            "legacy_tensors_bit_identical": True,
            "optimizer_existing_state_preserved": True,
            "optimizer_new_parameters_fresh": True,
        }
        migrated_payload["provenance"] = provenance
        checkpoint.immutable_torch_save(migrated_payload, output)
    finally:
        temporary.unlink(missing_ok=True)

    loaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if not (
        loaded.decision_fusion_enabled
        and not loaded.decision_fusion_runtime_enabled
        and loaded.decision_fusion is not None
    ):
        raise RuntimeError("published fusion checkpoint failed reconstruction")
    output_digest = checkpoint.checkpoint_digest(output)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_digest": parent_digest,
        "migrated_checkpoint": str(output),
        "migrated_checkpoint_digest": output_digest,
        "archetype_id": migrated_payload.get("archetype_id"),
        "model_id": migrated_payload.get("model_id"),
        "decision_fusion": {
            "schema": DECISION_FUSION_SCHEMA,
            "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
            "required_head_count": len(DECISION_FUSION_REQUIRED_HEADS),
            "width": int(fusion_width),
            "runtime_enabled": False,
            "training_enabled": True,
            "serving_eligible": False,
            "fusion_parameter_count": sum(
                int(value.numel())
                for key, value in loaded.state_dict().items()
                if key.startswith("decision_fusion.")
            ),
        },
        "proof": {
            "legacy_tensors_bit_identical": True,
            "zero_safe_initialization": True,
            "optimizer_existing_state_preserved": True,
            "optimizer_new_parameters_fresh": True,
            "existing_trainable_parameter_count": len(old_names),
            "added_trainable_parameter_count": len(new_names) - len(old_names),
            "checkpoint_load_verified": True,
        },
    }
    _exclusive_json(receipt, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fusion-width", type=int, default=16)
    args = parser.parse_args()
    if args.fusion_width < 1:
        raise ValueError("--fusion-width must be positive")
    print(json.dumps(materialize(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
