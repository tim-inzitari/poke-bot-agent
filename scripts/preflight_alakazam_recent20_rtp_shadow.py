#!/usr/bin/env python3
"""Seal the noninterference and optimizer canary for revision-16 RTP shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from poke_bot.recursive_turn_planner.recent20_overlay import (
    Recent20RTPDataset,
    canonical_bytes,
    sha256_file,
)
from poke_bot.recursive_turn_planner.recent20_shadow import (
    Recent20SemanticAdapter,
    _planner_config,
    _stage_batches,
)
from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
from poke_bot.recursive_turn_planner.training.shadow_train import (
    RTPTrainConfig,
    train_step,
)


SCHEMA = "poke_bot.alakazam_recent20_rtp_shadow_preflight/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--overlay-completion-receipt", type=Path, required=True)
    parser.add_argument("--overlay-completion-receipt-sha256", required=True)
    parser.add_argument("--overlay-validation-receipt", type=Path, required=True)
    parser.add_argument("--overlay-validation-receipt-sha256", required=True)
    parser.add_argument("--base-pack-root", type=Path, required=True)
    parser.add_argument("--base-completion-sha256", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--active-service-state", required=True)
    parser.add_argument("--active-service-main-pid", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    identities = {
        "overlay_manifest_sha256": sha256_file(args.manifest),
        "overlay_completion_receipt_sha256": sha256_file(
            args.overlay_completion_receipt
        ),
        "overlay_validation_receipt_sha256": sha256_file(
            args.overlay_validation_receipt
        ),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "goal_contract_sha256": sha256_file(args.goal_contract),
    }
    expected = {
        "overlay_manifest_sha256": args.manifest_sha256,
        "overlay_completion_receipt_sha256": args.overlay_completion_receipt_sha256,
        "overlay_validation_receipt_sha256": args.overlay_validation_receipt_sha256,
        "parent_checkpoint_sha256": args.parent_checkpoint_sha256,
    }
    for key, digest in expected.items():
        if identities[key] != digest:
            raise RuntimeError(f"preflight identity mismatch: {key}")
    contract = json.loads(args.goal_contract.read_text(encoding="utf-8"))
    revision = contract.get("revision_16_recent20_rtp_shadow_bootstrap") or {}
    if (
        contract.get("goal_revision") != 16
        or revision.get("overlay_manifest_sha256") != args.manifest_sha256
        or revision.get("base_pack_completion_sha256")
        != args.base_completion_sha256
        or revision.get("frozen_parent_checkpoint_sha256")
        != args.parent_checkpoint_sha256
    ):
        raise RuntimeError("revision-16 typed contract binding mismatch")
    if args.active_service_state != "active" or args.active_service_main_pid < 1:
        raise RuntimeError("active Inzi derivative service is not healthy")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    dataset = Recent20RTPDataset(
        args.manifest,
        base_pack_root=args.base_pack_root,
        expected_manifest_sha256=args.manifest_sha256,
        expected_base_completion_sha256=args.base_completion_sha256,
        verify_overlay_shards=True,
    )
    adapter = Recent20SemanticAdapter(d_model=96, hidden_width=128).to(device)
    planner = RecursiveTurnPlanner(_planner_config(96)).to(device)
    cfg = RTPTrainConfig(
        d_model=96,
        profile="pure_rl",
        epochs=1,
        lr=1e-4,
        seed=31816,
        device=str(device),
        dynamics_weight=0.05,
    )
    batch, identity = next(
        _stage_batches(
            dataset,
            "train",
            adapter=adapter,
            device=device,
            max_programs=1,
        )
    )
    parameters = list(planner.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    before = [parameter.detach().clone() for parameter in parameters]
    loss, metrics = train_step(planner, batch, cfg=cfg)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("preflight loss is non-finite")
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )
    if not finite_gradients:
        raise RuntimeError("preflight gradients are non-finite")
    optimizer.step()
    parameter_changed = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, parameters)
    )
    if not parameter_changed:
        raise RuntimeError("preflight optimizer did not update shadow parameters")

    gpu: dict[str, object] = {}
    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
    receipt = {
        "schema": SCHEMA,
        "goal_revision": 16,
        **identities,
        "base_pack_root": str(args.base_pack_root.resolve()),
        "base_pack_completion_sha256": args.base_completion_sha256,
        "device": str(device),
        "gpu": gpu,
        "loader_join_passed": True,
        "program_identity": identity["program_identity"],
        "finite_loss": float(loss.detach().cpu()),
        "finite_gradients": finite_gradients,
        "optimizer_parameter_update_passed": parameter_changed,
        "loss_metrics": metrics,
        "active_inzi_derivative_service_state": args.active_service_state,
        "active_inzi_derivative_service_main_pid": args.active_service_main_pid,
        "active_inzi_derivative_service_control_performed": False,
        "policy_checkpoint_loaded_or_mutated": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "passed": True,
        "validated_at_unix_seconds": time.time(),
    }
    body = canonical_bytes(receipt)
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"sha256-{digest.removeprefix('sha256:')}.rtp-shadow-preflight.json"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
