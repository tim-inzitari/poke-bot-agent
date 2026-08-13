"""Five-epoch r274 refreshes from the immutable r279 GPU-resident pack."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from . import checkpoint
from .pure_rl.expert_cpu_pack import validate_cpu_corpus
from .r279_contiguous_expert_pack import load_pack, sha256_file, validate_r279_pack
from .train import supervised_rehearsal_step


PACK_RECEIPT_SCHEMA = "poke_bot.r279_contiguous_expert_pack_receipt/v1"
REFRESH_RECEIPT_SCHEMA = "poke_bot.r280_gpu_resident_scheduled_refresh/v1"
EXPECTED_GAMES = 26_704
EXPECTED_DECISIONS = 2_040_911
EXPECTED_PACK_BYTES = 5_725_073_070
MIN_FREE_GIB_AFTER_PACK = 24.0


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _semantic_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"r280 evidence is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_bytes(payload)
    if target.exists():
        if not target.is_file() or target.read_bytes() != body:
            raise RuntimeError(f"immutable r280 receipt already differs: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.r280-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"r280 JSON artifact must be an object: {path}")
    return value


def validate_refresh_receipt(
    path: Path,
    *,
    expected_parent_sha256: str,
    expected_pack_sha256: str,
    expected_before_iteration: int,
) -> dict[str, Any]:
    receipt = _load_json(path)
    if (
        receipt.get("schema") != REFRESH_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("receipt_sha256") != _semantic_digest(receipt)
        or int(receipt.get("before_iteration", -1))
        != int(expected_before_iteration)
        or int(receipt.get("epochs", -1)) != 5
        or receipt.get("parent", {}).get("sha256") != expected_parent_sha256
        or receipt.get("pack", {}).get("sha256") != expected_pack_sha256
        or receipt.get("gpu_residency", {}).get("full_numeric_pack_resident")
        is not True
        or receipt.get("gpu_residency", {}).get("device_side_batch_gather")
        is not True
        or receipt.get("gpu_residency", {}).get("host_batch_streaming_used")
        is not False
        or receipt.get("gpu_residency", {}).get("resident_python_objects_used")
        is not False
    ):
        raise RuntimeError("r280 scheduled refresh receipt is invalid")
    if _file_identity(Path(receipt["parent"]["path"])) != receipt["parent"]:
        raise RuntimeError("r280 refresh parent identity drifted")
    if _file_identity(Path(receipt["checkpoint"]["path"])) != receipt["checkpoint"]:
        raise RuntimeError("r280 refresh checkpoint identity drifted")
    if receipt["checkpoint"]["sha256"] == receipt["parent"]["sha256"]:
        raise RuntimeError("r280 refresh did not create a child checkpoint")
    bounds = list(receipt.get("rl_iteration_before_after") or ())
    if len(bounds) != 2 or bounds[0] != bounds[1]:
        raise RuntimeError("r280 refresh advanced the RL iteration")
    return receipt


def run_refresh(
    *,
    pack_path: Path,
    pack_receipt_path: Path,
    parent_path: Path,
    parent_digest: str,
    output_path: Path,
    receipt_path: Path,
    before_iteration: int,
    device_name: str,
    requested_batch_size: int,
    learning_rate: float,
    seed: int,
    loss_weights: Mapping[str, float],
    expanded_head_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    if int(before_iteration) not in {5, 10, 15, 20, 25}:
        raise ValueError("r280 refresh is allowed only at exact five-update boundaries")
    pack_receipt = _load_json(pack_receipt_path)
    pack_identity = _file_identity(pack_path)
    if (
        pack_receipt.get("schema") != PACK_RECEIPT_SCHEMA
        or pack_receipt.get("validated") is not True
        or dict(pack_receipt.get("pack") or {}) != pack_identity
        or int(pack_identity["size_bytes"]) != EXPECTED_PACK_BYTES
    ):
        raise RuntimeError("sealed r279 pack receipt is invalid")
    parent_identity = _file_identity(parent_path)
    if parent_identity["sha256"] != str(parent_digest):
        raise RuntimeError("r280 refresh parent digest mismatch")
    if receipt_path.is_file():
        return validate_refresh_receipt(
            receipt_path,
            expected_parent_sha256=parent_identity["sha256"],
            expected_pack_sha256=pack_identity["sha256"],
            expected_before_iteration=before_iteration,
        )
    if output_path.exists():
        raise RuntimeError("r280 refresh checkpoint exists without its receipt")

    core_cpu, side_cpu, metadata = load_pack(pack_path)
    validate_cpu_corpus(core_cpu)
    counts = validate_r279_pack(
        core_cpu,
        side_cpu,
        expected_games=EXPECTED_GAMES,
        expected_decisions=EXPECTED_DECISIONS,
    )
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("r280 primary refresh path requires CUDA")
    torch.cuda.set_device(device)
    side_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in side_cpu.values()
    )
    tensor_bytes = int(core_cpu.tensor_bytes) + int(side_bytes)
    free_before, total_device = torch.cuda.mem_get_info(device)
    if free_before - tensor_bytes < int(MIN_FREE_GIB_AFTER_PACK * 2**30):
        raise MemoryError("r280 refresh pack would violate GPU safety headroom")
    core_gpu = core_cpu.to_device(
        device,
        min_free_gib=MIN_FREE_GIB_AFTER_PACK + side_bytes / 2**30,
    )
    side_gpu = {
        name: value.to(device=device).contiguous() for name, value in side_cpu.items()
    }
    free_after, _ = torch.cuda.mem_get_info(device)
    if free_after < int(MIN_FREE_GIB_AFTER_PACK * 2**30):
        raise MemoryError("r280 refresh pack left insufficient GPU headroom")
    del core_cpu, side_cpu
    gc.collect()

    parent_payload = checkpoint.load_checkpoint(parent_path, map_location="cpu")
    parent_rl_iteration = int(parent_payload.get("rl_iteration", 0))
    result = supervised_rehearsal_step(
        core_gpu,
        base_ckpt=parent_path,
        output_path=output_path,
        parent_digest=parent_digest,
        rehearsal_iteration=int(before_iteration),
        manifest_identity={
            "schema": PACK_RECEIPT_SCHEMA,
            "path": str(pack_path.resolve()),
            "sha256": pack_identity["sha256"],
            "counts": counts,
            "contract": metadata["contract"],
        },
        epochs=5,
        lr=float(learning_rate),
        requested_batch_size=int(requested_batch_size),
        seed=int(seed),
        corpus_split_seed=int(seed),
        device=device,
        aux_loss_weight=float(loss_weights.get("archetype", 0.0)),
        opp_hand_loss_weight=float(loss_weights.get("opponent_hand", 0.0)),
        opp_remainder_loss_weight=float(
            loss_weights.get("opponent_hidden_remainder", 0.0)
        ),
        lethal_threat_loss_weight=float(loss_weights.get("lethal_threat", 0.0)),
        prize_race_loss_weight=float(loss_weights.get("prize_race", 0.0)),
        alakazam_guide_loss_weight=0.0,
        setup_board_outcome_loss_weight=float(
            loss_weights.get("setup_board_outcome", 0.025)
        ),
        combo_state_loss_weight=float(loss_weights.get("combo_state", 0.025)),
        visible_tutor_completion_loss_weight=0.025,
        terminal_conversion_loss_weight=0.025,
        tactical_sequence_outcome_loss_weight=0.025,
        expanded_head_loss_weights=dict(
            expanded_head_schedule.get("loss_weights") or {}
        ),
        expanded_head_schedule=dict(expanded_head_schedule),
        output_archetype_id="alakazam",
        output_model_id=(
            f"alakazam-new-list-direct-policy-r274.refresh-{before_iteration:05d}"
        ),
        r279_side_tensors=side_gpu,
        extra_updates={
            "r280_gpu_resident_refresh": {
                "schema": REFRESH_RECEIPT_SCHEMA,
                "before_iteration": int(before_iteration),
                "pack_sha256": pack_identity["sha256"],
                "device": str(device),
                "full_numeric_pack_resident": True,
                "device_side_batch_gather": True,
                "host_batch_streaming_used": False,
                "resident_python_objects_used": False,
            }
        },
    )
    del core_gpu, side_gpu
    gc.collect()
    torch.cuda.empty_cache()
    child_payload = checkpoint.load_checkpoint(output_path, map_location="cpu")
    child_rl_iteration = int(child_payload.get("rl_iteration", -1))
    receipt: dict[str, Any] = {
        "schema": REFRESH_RECEIPT_SCHEMA,
        "status": "passed",
        "before_iteration": int(before_iteration),
        "epochs": 5,
        "parent_digest": parent_identity["sha256"],
        "checkpoint_digest": sha256_file(output_path),
        "parent": parent_identity,
        "checkpoint": _file_identity(output_path),
        "pack": pack_identity,
        "pack_receipt": _file_identity(pack_receipt_path),
        "counts": counts,
        "loss_weights": {
            **{str(name): float(value) for name, value in loss_weights.items()},
            "visible_tutor_completion": 0.025,
            "terminal_conversion": 0.025,
            "tactical_sequence_outcome": 0.025,
            "expert_guide": 0.0,
        },
        "rl_iteration_before_after": [parent_rl_iteration, child_rl_iteration],
        "gpu_residency": {
            "device": str(device),
            "pack_tensor_bytes": tensor_bytes,
            "device_total_bytes": int(total_device),
            "free_before_bytes": int(free_before),
            "free_after_pack_bytes": int(free_after),
            "full_numeric_pack_resident": True,
            "device_side_batch_gather": True,
            "host_batch_streaming_used": False,
            "resident_python_objects_used": False,
        },
        "training_result": result,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = _semantic_digest(receipt)
    _write_create_only(receipt_path, receipt)
    return validate_refresh_receipt(
        receipt_path,
        expected_parent_sha256=parent_identity["sha256"],
        expected_pack_sha256=pack_identity["sha256"],
        expected_before_iteration=before_iteration,
    )


__all__ = ["REFRESH_RECEIPT_SCHEMA", "run_refresh", "validate_refresh_receipt"]
