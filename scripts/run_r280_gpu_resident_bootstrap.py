#!/usr/bin/env python3
"""Run the r274 25-epoch bootstrap from the sealed r279 pack in GPU VRAM."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from poke_bot import checkpoint
from poke_bot.matchup_adapters_v6 import registry_digest
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.r279_contiguous_expert_pack import (
    load_pack,
    sha256_file,
    validate_r279_pack,
)
from poke_bot.train import supervised_rehearsal_step


PACK_RECEIPT_SCHEMA = "poke_bot.r279_contiguous_expert_pack_receipt/v1"
ROSTER_RECEIPT_SCHEMA = "poke_bot.ptcgreplay_matchup_roster_candidate_r273/v1"
ROUTER_RECEIPT_SCHEMA = "poke_bot.public_matchup_decision_tree_receipt/v1"
ACTIVATED_PARENT_SCHEMA = "poke_bot.r280_bootstrap_parent_activation/v1"
RESULT_SCHEMA = "poke_bot.r280_gpu_resident_bootstrap_result/v1"
EXPECTED_GAMES = 26_704
EXPECTED_DECISIONS = 2_040_911
EXPECTED_PACK_BYTES = 5_725_073_070
MIN_FREE_GIB_AFTER_PACK = 24.0


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_digest(value: Mapping[str, Any], excluded: str) -> str:
    payload = dict(value)
    payload.pop(excluded, None)
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def torch_save_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _norm_for_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> float:
    total = sum(
        float(value.detach().float().square().sum().item())
        for name, value in state.items()
        if name.startswith(prefix)
    )
    return math.sqrt(total)


def materialize_activated_parent(
    *,
    base_path: Path,
    expected_base_sha256: str,
    roster_path: Path,
    roster_receipt_path: Path,
    router_receipt_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create a state-identical config derivative for the r280 bootstrap."""

    if output_path.exists() or receipt_path.exists():
        if not output_path.is_file() or not receipt_path.is_file():
            raise RuntimeError("partial r280 activated-parent artifact exists")
        receipt = _json(receipt_path)
        if (
            receipt.get("schema") != ACTIVATED_PARENT_SCHEMA
            or receipt.get("activated_parent", {}).get("sha256")
            != sha256_file(output_path)
            or receipt.get("receipt_sha256")
            != semantic_digest(receipt, "receipt_sha256")
        ):
            raise RuntimeError("existing r280 activated-parent receipt is invalid")
        return output_path, receipt

    actual_base_sha256 = sha256_file(base_path)
    if actual_base_sha256 != expected_base_sha256:
        raise RuntimeError(
            f"zero-safe successor digest mismatch: {actual_base_sha256}"
        )
    roster = _json(roster_path)
    roster_receipt = _json(roster_receipt_path)
    router_receipt = _json(router_receipt_path)
    if (
        roster_receipt.get("schema") != ROSTER_RECEIPT_SCHEMA
        or roster_receipt.get("candidate_registry_digest")
        != registry_digest(roster)
        or roster_receipt.get("existing_slots_0_through_19_unchanged") is not True
        or roster_receipt.get("new_slots_exact_zero_dormant_required") is not True
        or roster.get("slot_capacity") != 64
        or len(roster.get("slots") or []) != 64
        or len(roster.get("active_expert_ids") or []) != 40
    ):
        raise RuntimeError("append-only matchup roster receipt is invalid")
    if (
        router_receipt.get("schema") != ROUTER_RECEIPT_SCHEMA
        or router_receipt.get("runtime_enabled") is not False
    ):
        raise RuntimeError("expanded router readiness receipt is invalid")
    calibration = dict(router_receipt.get("runtime_calibration") or {})
    precision_floor = float(calibration.get("precision_floor", -1.0))
    if not math.isclose(precision_floor, 0.93, abs_tol=1e-12):
        raise RuntimeError("expanded router precision floor changed")
    available_rows = [
        dict(row)
        for row in dict(calibration.get("per_archetype") or {}).values()
        if isinstance(row, Mapping) and row.get("available") is True
    ]
    if not available_rows or any(
        float(row.get("precision", -1.0)) < precision_floor
        or int(row.get("accepted_weighted_observations", 0)) <= 0
        for row in available_rows
    ):
        raise RuntimeError("expanded router contains an invalid available route")

    payload = checkpoint.load_checkpoint(base_path, map_location="cpu")
    state = dict(payload.get("model_state_dict") or {})
    config = dict(payload.get("model_config") or {})
    if (
        config.get("combo_state_head_enabled") is not True
        or config.get("combo_state_route_enabled") is not False
        or config.get("tactical_sequence_outcome_route_enabled") is not False
        or config.get("matchup_adapters_enabled") is not False
        or _norm_for_prefix(state, "combo_state_route.") != 0.0
    ):
        raise RuntimeError("zero-safe successor is not at the expected route boundary")
    for slot in range(20, 64):
        if _norm_for_prefix(state, f"matchup_adapter_bank.experts.{slot}.") != 0.0:
            raise RuntimeError(f"new matchup adapter slot {slot} is not exact zero")

    config["combo_state_route_enabled"] = True
    config["matchup_adapters_enabled"] = True
    config["matchup_adapter_registry"] = roster
    payload["model_config"] = config
    extra = dict(payload.get("extra") or {})
    adapter_config = dict(extra.get("matchup_adapter_config") or {})
    adapter_config["slot_registry"] = roster
    adapter_config["slot_registry_digest"] = registry_digest(roster)
    extra["matchup_adapter_config"] = adapter_config
    extra["matchup_adapters_runtime_enabled"] = True
    extra["r280_bootstrap_parent_activation"] = {
        "schema": ACTIVATED_PARENT_SCHEMA,
        "base_sha256": actual_base_sha256,
        "model_state_tensors_unchanged": True,
        "combo_state_route_enabled": True,
        "combo_state_route_zero_safe_at_activation": True,
        "tactical_sequence_route_enabled_during_bootstrap": False,
        "matchup_adapter_bank_enabled": True,
        "matchup_adapter_config_rebound_to_append_only_roster": True,
        "new_adapter_slots_20_through_39_exact_zero_dormant": True,
        "unused_adapter_slots_40_through_63_exact_zero": True,
        "roster_file_sha256": sha256_file(roster_path),
        "roster_registry_digest": registry_digest(roster),
        "roster_receipt_sha256": sha256_file(roster_receipt_path),
        "router_readiness_receipt_sha256": sha256_file(router_receipt_path),
        "router_runtime_activation_deferred": True,
    }
    payload["extra"] = extra
    torch_save_create_only(output_path, payload)
    activated_sha256 = sha256_file(output_path)
    receipt: dict[str, Any] = {
        "schema": ACTIVATED_PARENT_SCHEMA,
        "owner_revision": 280,
        "base": {
            "path": str(base_path),
            "sha256": actual_base_sha256,
        },
        "activated_parent": {
            "path": str(output_path),
            "sha256": activated_sha256,
            "size_bytes": output_path.stat().st_size,
        },
        "model_state_tensors_unchanged": True,
        "config_changes": {
            "combo_state_route_enabled": [False, True],
            "matchup_adapters_enabled": [False, True],
            "matchup_adapter_registry": "append_only_r273_candidate",
        },
        "tactical_sequence_route_enabled_during_bootstrap": False,
        "matchup_readiness": extra["r280_bootstrap_parent_activation"],
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = semantic_digest(receipt, "receipt_sha256")
    write_json_create_only(receipt_path, receipt)
    return output_path, receipt


def move_side_to_device(
    side: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device).contiguous()
        for name, value in side.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--pack-receipt", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--matchup-roster", required=True, type=Path)
    parser.add_argument("--matchup-roster-receipt", required=True, type=Path)
    parser.add_argument("--router-ready-receipt", required=True, type=Path)
    parser.add_argument("--canary-receipt", required=True, type=Path)
    parser.add_argument("--activated-parent", required=True, type=Path)
    parser.add_argument("--activated-parent-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result-receipt", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=274)
    args = parser.parse_args()

    if int(args.epochs) != 25:
        raise ValueError("r280 bootstrap must run exactly 25 epochs")
    if args.output.exists() or args.result_receipt.exists():
        raise FileExistsError("r280 bootstrap output is create-only")
    canary = _json(args.canary_receipt)
    if canary.get("status") != "passed":
        raise RuntimeError("r274 pre-start canary is not passed")
    pack_receipt = _json(args.pack_receipt)
    pack_row = dict(pack_receipt.get("pack") or {})
    if (
        pack_receipt.get("schema") != PACK_RECEIPT_SCHEMA
        or pack_receipt.get("validated") is not True
        or pack_row.get("sha256") != sha256_file(args.pack)
        or int(pack_row.get("size_bytes", -1)) != EXPECTED_PACK_BYTES
        or args.pack.stat().st_size != EXPECTED_PACK_BYTES
    ):
        raise RuntimeError("sealed contiguous-pack receipt is invalid")

    activated_parent, activation_receipt = materialize_activated_parent(
        base_path=args.base_checkpoint,
        expected_base_sha256=str(args.base_sha256),
        roster_path=args.matchup_roster,
        roster_receipt_path=args.matchup_roster_receipt,
        router_receipt_path=args.router_ready_receipt,
        output_path=args.activated_parent,
        receipt_path=args.activated_parent_receipt,
    )

    started = time.time()
    core_cpu, side_cpu, pack_metadata = load_pack(args.pack)
    validate_cpu_corpus(core_cpu)
    counts = validate_r279_pack(
        core_cpu,
        side_cpu,
        expected_games=EXPECTED_GAMES,
        expected_decisions=EXPECTED_DECISIONS,
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("r280 primary training path requires CUDA")
    torch.cuda.set_device(device)
    side_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in side_cpu.values()
    )
    total_pack_tensor_bytes = int(core_cpu.tensor_bytes) + int(side_bytes)
    free_before, total_device = torch.cuda.mem_get_info(device)
    if free_before - total_pack_tensor_bytes < int(MIN_FREE_GIB_AFTER_PACK * 2**30):
        raise MemoryError("full GPU pack would violate the r280 safety headroom")
    core_gpu = core_cpu.to_device(device, min_free_gib=MIN_FREE_GIB_AFTER_PACK + side_bytes / 2**30)
    side_gpu = move_side_to_device(side_cpu, device)
    free_after, _ = torch.cuda.mem_get_info(device)
    if free_after < int(MIN_FREE_GIB_AFTER_PACK * 2**30):
        raise MemoryError("full GPU side pack left insufficient training headroom")
    del core_cpu, side_cpu
    gc.collect()
    print(
        "[r280-gpu-pack] resident "
        f"tensor_gib={total_pack_tensor_bytes / 2**30:.3f} "
        f"free_before_gib={free_before / 2**30:.3f} "
        f"free_after_gib={free_after / 2**30:.3f} "
        f"device_total_gib={total_device / 2**30:.3f}",
        flush=True,
    )

    parent = checkpoint.load_checkpoint(activated_parent, map_location="cpu")
    expanded_training = dict(
        dict(parent.get("extra") or {}).get("expanded_head_training") or {}
    )
    expanded_weights = dict(expanded_training.get("loss_weights") or {})
    schedule = {
        "schema": "poke_bot.expanded_head_schedule/v1",
        "runtime_enabled_heads": [],
        "loss_weights": expanded_weights,
        "schedule_digest": expanded_training.get("schedule_digest"),
        "target_schema": expanded_training.get("target_schema_version"),
        "target_schema_digest": expanded_training.get("target_schema_digest"),
        "stage_index": 0,
        "epoch": 25,
    }
    result = supervised_rehearsal_step(
        core_gpu,
        base_ckpt=activated_parent,
        output_path=args.output,
        parent_digest=checkpoint.checkpoint_digest(activated_parent),
        rehearsal_iteration=0,
        manifest_identity={
            "schema": PACK_RECEIPT_SCHEMA,
            "path": str(args.pack),
            "sha256": pack_row["sha256"],
            "counts": counts,
            "contract": pack_metadata["contract"],
        },
        epochs=25,
        lr=1e-5,
        requested_batch_size=int(args.max_decisions_per_batch),
        seed=int(args.seed),
        corpus_split_seed=int(args.seed),
        device=device,
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        alakazam_guide_loss_weight=0.0,
        setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.025,
        visible_tutor_completion_loss_weight=0.025,
        terminal_conversion_loss_weight=0.025,
        tactical_sequence_outcome_loss_weight=0.025,
        expanded_head_loss_weights=expanded_weights,
        expanded_head_schedule=schedule,
        output_archetype_id="alakazam",
        output_model_id="alakazam-new-list-direct-policy-r274-bootstrap-r280",
        r279_side_tensors=side_gpu,
        extra_updates={
            "r280_gpu_resident_training": {
                "schema": RESULT_SCHEMA,
                "pack_sha256": pack_row["sha256"],
                "pack_tensor_bytes": total_pack_tensor_bytes,
                "device": str(device),
                "free_before_bytes": int(free_before),
                "free_after_pack_bytes": int(free_after),
                "cpu_pack_role": "immutable_restart_cache",
                "epoch_batch_gather": "device_side_only",
                "pinned_cpu_streaming_used": False,
                "resident_python_objects_used": False,
                "activated_parent_sha256": activation_receipt["activated_parent"]["sha256"],
            }
        },
    )
    output_sha256 = sha256_file(args.output)
    result_receipt: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "owner_revision": 280,
        "status": "completed",
        "epochs_completed": 25,
        "pack": pack_row,
        "activated_parent": activation_receipt["activated_parent"],
        "output": {
            "path": str(args.output),
            "sha256": output_sha256,
            "size_bytes": args.output.stat().st_size,
        },
        "counts": counts,
        "gpu_residency": {
            "device": str(device),
            "pack_tensor_bytes": total_pack_tensor_bytes,
            "free_before_bytes": int(free_before),
            "free_after_pack_bytes": int(free_after),
            "full_numeric_pack_resident": True,
            "device_side_batch_gather": True,
            "pinned_cpu_fallback_used": False,
        },
        "training_result": result,
        "elapsed_seconds": time.time() - started,
        "receipt_sha256": None,
    }
    result_receipt["receipt_sha256"] = semantic_digest(
        result_receipt, "receipt_sha256"
    )
    write_json_create_only(args.result_receipt, result_receipt)
    print(json.dumps(result_receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
