"""Read-only revision-9 handoff readiness assessment.

This module cannot control services, mutate checkpoints, activate staged data,
or start training.  It records the exact current evidence and blockers before
the closed r303 handoff receipt sequence may begin.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


GOAL_REVISION = 9
GOAL_GATEWAY_SHA256 = "sha256:8908c4e8bcf36a089ba7f230c137e259f024125807bdb04b03d77483f533c223"
CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
R195_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R274_ITER1_SHA256 = "sha256:d5ad45faa1d05dd3e2e62ffd2b25867b6ce0285a918aefa622891880f4902b05"
BRANCH_RECEIPT_SHA256 = "sha256:084d068bebfa2a0da15209bda798842c38e59a52637bb49723fad063c487a52e"
CORPUS_MANIFEST_SHA256 = "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
SCHEMA = "poke_bot.alakazam_rule_derivative_revision9_readiness_assessment/v1"


class Revision9ReadinessError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Revision9ReadinessError(f"missing regular artifact: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise Revision9ReadinessError(f"artifact digest mismatch: {path}")
    metadata = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "write_bits_absent": metadata.st_mode & 0o222 == 0,
    }


def _systemctl_show(unit: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "systemctl", "--user", "show", unit,
            "-p", "Id", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID",
            "-p", "ExecMainStatus", "-p", "FragmentPath", "-p", "UnitFileState",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if values.get("Id") != unit:
        raise Revision9ReadinessError(f"systemd returned a foreign unit for {unit}")
    return {
        "unit": unit,
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "main_pid": int(values.get("MainPID", "0")),
        "exec_main_status": int(values.get("ExecMainStatus", "0")),
        "fragment_path": values.get("FragmentPath"),
        "unit_file_state": values.get("UnitFileState"),
    }


def _checkpoint_inventory(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - host runtime gate
        raise Revision9ReadinessError("Torch is required for checkpoint inventory") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise Revision9ReadinessError("r274 checkpoint is not a mapping")
    model = checkpoint.get("model_state_dict")
    optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise Revision9ReadinessError("r274 checkpoint lacks model or optimizer state")
    model_shapes = {name: list(value.shape) for name, value in sorted(model.items())}
    model_shape_sha = "sha256:" + hashlib.sha256(canonical_bytes(model_shapes)).hexdigest()
    return {
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "model_tensor_count": len(model),
        "model_name_shape_inventory_sha256": model_shape_sha,
        "optimizer_state_embedded": True,
        "optimizer_top_level_keys": sorted(str(key) for key in optimizer),
        "rl_iteration": checkpoint.get("rl_iteration"),
        "torch_version_recorded": checkpoint.get("torch_version"),
    }


def _gpu_inventory() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - host runtime gate
        raise Revision9ReadinessError("Torch is required for GPU inventory") from exc
    rows = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        rows.append({
            "torch_ordinal": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        })
    return rows


def build_readiness_assessment(
    *,
    r195_checkpoint_path: Path,
    r274_checkpoint_path: Path,
    corpus_manifest_path: Path,
    branch_receipt_path: Path,
    exact_pause_services: Sequence[str],
) -> dict[str, Any]:
    if tuple(exact_pause_services) != (
        "pokebot-alakazam-r274-rl.service",
        "pokebot-alakazam-r274-rl-submission-boundaries.service",
    ):
        raise Revision9ReadinessError("unexpected predecessor service inventory")
    r195 = _identity(r195_checkpoint_path, R195_SHA256)
    r274 = _identity(r274_checkpoint_path, R274_ITER1_SHA256)
    corpus = _identity(corpus_manifest_path, CORPUS_MANIFEST_SHA256)
    branch = _identity(branch_receipt_path, BRANCH_RECEIPT_SHA256)
    branch_payload = json.loads(branch_receipt_path.read_text())
    if (
        branch_payload.get("eligible_trainable_branches") != ["public_rule_semantic_projection"]
        or branch_payload.get("candidate_training_allowed") is not False
    ):
        raise Revision9ReadinessError("branch adjudication does not match revision-9 support")
    services = [_systemctl_show(unit) for unit in exact_pause_services]
    checkpoint = _checkpoint_inventory(r274_checkpoint_path)
    gpus = _gpu_inventory()
    blackwell = [row for row in gpus if row["name"] == "NVIDIA RTX PRO 5000 Blackwell"]

    blockers: list[str] = []
    if not r195["write_bits_absent"]:
        blockers.append("r195_checkpoint_not_filesystem_read_only")
    if not r274["write_bits_absent"]:
        blockers.append("r274_iter1_checkpoint_not_filesystem_read_only")
    for row in services:
        if row["active_state"] != "inactive":
            blockers.append(f"predecessor_service_not_inactive:{row['unit']}:{row['active_state']}/{row['sub_state']}")
    if checkpoint["rl_iteration"] != 1:
        blockers.append("r274_checkpoint_not_committed_iteration_1")
    if len(blackwell) != 1:
        blockers.append("blackwell_device_inventory_not_unique")
    elif blackwell[0]["torch_ordinal"] != 1:
        blockers.append(
            f"contract_cuda_1_does_not_match_torch_blackwell_ordinal:{blackwell[0]['torch_ordinal']}"
        )
    blockers.append("revision9_handoff_corpus_receipt_compatibility_not_yet_sealed")
    blockers.append("composite_per_tensor_source_map_and_frozen_tensor_receipt_not_yet_sealed")
    blockers.append("blackwell_forward_backward_optimizer_canary_not_yet_run")
    blockers.append("rollback_dry_run_receipt_not_yet_sealed")

    return {
        "schema": SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "assessment_only_no_authority": True,
        "artifacts": {
            "r195_checkpoint": r195,
            "r274_iter1_checkpoint": r274,
            "recent20_corpus_manifest": corpus,
            "branch_adjudication_receipt": branch,
        },
        "r274_checkpoint_inventory": checkpoint,
        "service_observations": services,
        "gpu_inventory": gpus,
        "eligible_trainable_branches": ["public_rule_semantic_projection"],
        "unsupported_branches_remain_exact_zero_and_inert": True,
        "readiness_passed": not blockers,
        "blockers": blockers,
        "service_control_performed": False,
        "training_or_activation_performed": False,
        "assessed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise Revision9ReadinessError("assessment output exists")
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
    "Revision9ReadinessError",
    "build_readiness_assessment",
    "write_create_only",
]
