"""Create-only r195-first/r274-additions tensor source-map evidence.

This module inventories immutable checkpoints on CPU.  It does not assemble or
write a checkpoint, initialize new parameters, control services, or authorize
training.  The later frozen-tensor receipt must consume this evidence and the
actual candidate architecture independently.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GOAL_REVISION = 9
GOAL_GATEWAY_SHA256 = "sha256:8908c4e8bcf36a089ba7f230c137e259f024125807bdb04b03d77483f533c223"
CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
R195_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R274_SHA256 = "sha256:d5ad45faa1d05dd3e2e62ffd2b25867b6ce0285a918aefa622891880f4902b05"
SOURCE_MAP_SCHEMA = "poke_bot.alakazam_rule_derivative_composite_tensor_source_map/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_rule_derivative_composite_tensor_source_map_receipt/v1"


class CompositeMapError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _shape(value: Any) -> list[int]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise CompositeMapError("model state contains a non-tensor value")
    return [int(item) for item in raw]


def _dtype(value: Any) -> str:
    raw = getattr(value, "dtype", None)
    if raw is None:
        raise CompositeMapError("model tensor lacks dtype")
    return str(raw)


def build_tensor_source_map(
    *,
    r195_state: Mapping[str, Any],
    r274_state: Mapping[str, Any],
) -> dict[str, Any]:
    if not r195_state or not r274_state:
        raise CompositeMapError("checkpoint model state is empty")
    rows: list[dict[str, Any]] = []
    r195_count = 0
    r274_count = 0
    for name in sorted(r274_state):
        target = r274_state[name]
        if name in r195_state:
            if _shape(r195_state[name]) != _shape(target):
                raise CompositeMapError(f"r195 same-name shape drift: {name}")
            if _dtype(r195_state[name]) != _dtype(target):
                raise CompositeMapError(f"r195 same-name dtype drift: {name}")
            source = "r195_exact_name_shape_dtype"
            r195_count += 1
        else:
            source = "r274_architecture_addition_absent_from_r195"
            r274_count += 1
        rows.append({
            "name": name,
            "shape": _shape(target),
            "dtype": _dtype(target),
            "source": source,
        })
    unused_r195 = sorted(set(r195_state) - set(r274_state))
    if unused_r195:
        raise CompositeMapError("r195 contains names absent from the r274 target architecture")
    return {
        "schema": SOURCE_MAP_SCHEMA,
        "target_architecture": "exact_r274_model_state_dict_before_new_rule_parameters",
        "source_precedence": [
            "r195_exact_name_shape_dtype",
            "r274_architecture_addition_absent_from_r195",
            "genuinely_new_rule_parameters_initialized_separately_not_in_this_map",
        ],
        "tensor_count": len(rows),
        "source_counts": {
            "r195_exact_name_shape_dtype": r195_count,
            "r274_architecture_addition_absent_from_r195": r274_count,
        },
        "same_name_shape_or_dtype_drift_count": 0,
        "unused_r195_tensor_count": 0,
        "tensors": rows,
    }


def _load_checkpoint(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise CompositeMapError(f"missing regular checkpoint: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise CompositeMapError(f"checkpoint digest mismatch: {path}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - host runtime gate
        raise CompositeMapError("Torch is required for checkpoint inventory") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise CompositeMapError("checkpoint lacks model_state_dict")
    return payload["model_state_dict"], {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "internal_rl_iteration": payload.get("rl_iteration"),
    }


def build_composite_map_artifacts(
    *,
    r195_checkpoint_path: Path,
    r274_checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    r195_state, r195 = _load_checkpoint(r195_checkpoint_path, R195_SHA256)
    r274_state, r274 = _load_checkpoint(r274_checkpoint_path, R274_SHA256)
    source_map = build_tensor_source_map(r195_state=r195_state, r274_state=r274_state)
    source_map_sha = "sha256:" + hashlib.sha256(canonical_bytes(source_map)).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "status": "passed_inherited_tensor_source_classification_only",
        "r195_checkpoint": r195,
        "r274_checkpoint": r274,
        "target_architecture": source_map["target_architecture"],
        "tensor_count": source_map["tensor_count"],
        "source_counts": source_map["source_counts"],
        "per_tensor_source_map_logical_filename": "per-tensor-source-map.json",
        "per_tensor_source_map_sha256": source_map_sha,
        "per_tensor_source_map_size_bytes": len(canonical_bytes(source_map)),
        "all_inherited_target_tensors_classified_exactly_once": True,
        "ambiguous_overlap_count": 0,
        "shape_or_dtype_drift_count": 0,
        "inherited_tensors_initially_frozen_required": True,
        "new_rule_parameter_initialization_covered_by_this_receipt": False,
        "frozen_tensor_receipt_issued": False,
        "candidate_checkpoint_created": False,
        "parent_checkpoint_bytes_mutated": False,
        "service_control_performed": False,
        "training_or_activation_authority": False,
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return source_map, receipt


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise CompositeMapError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        body = canonical_bytes(value)
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


__all__ = [
    "CompositeMapError",
    "build_composite_map_artifacts",
    "build_tensor_source_map",
    "write_create_only",
]
