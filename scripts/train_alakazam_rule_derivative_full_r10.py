#!/usr/bin/env python3
"""Full r195→r274-structure bootstrap plus every-occurrence rule training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from poke_bot import checkpoint
from poke_bot.alakazam_rule_derivative_model_r298 import (
    R298PublicRuleSemanticProjection,
    R298SemanticProjectionConfig,
)
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.r279_contiguous_expert_pack import load_pack, validate_r279_pack
from poke_bot.train import supervised_rehearsal_step


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, canonical_bytes(value)); os.fsync(fd)
    finally:
        os.close(fd)


def save_torch(path: Path, value: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            torch.save(dict(value), stream); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def move_side(side: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device).contiguous() for name, value in side.items()}


def semantic_packs(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in completion["packs"]:
        receipt = json.loads(Path(row["receipt_path"]).read_text())
        files = receipt["files"]
        features = np.memmap(files["features_f32"]["path"], dtype="<f4", mode="r").reshape(-1, 40)
        offsets = np.memmap(files["decision_offsets_u64"]["path"], dtype="<u8", mode="r")
        selected = np.memmap(files["selected_option_u32"]["path"], dtype="<u4", mode="r")
        if len(offsets) != len(selected) + 1 or int(offsets[-1]) != len(features):
            raise RuntimeError("semantic tensor pack shape mismatch")
        result.append({"features": features, "offsets": offsets, "selected": selected, "receipt": receipt})
    return result


def train_semantic(
    *, candidate: Mapping[str, Any], completion_path: Path, device: torch.device,
    epochs: int, block_decisions: int, learning_rate: float, seed: int,
    resume_optimizer_from_candidate: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    completion = json.loads(completion_path.read_text())
    if completion.get("deduplicated") is not False:
        raise RuntimeError("semantic pack must preserve every occurrence")
    packs = semantic_packs(completion)
    config = dict(candidate["public_rule_semantic_projection_config"])
    model = R298PublicRuleSemanticProjection(R298SemanticProjectionConfig(**config)).to(device)
    model.load_state_dict(candidate["public_rule_semantic_projection_state_dict"], strict=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    if resume_optimizer_from_candidate:
        state = candidate.get(
            "public_rule_semantic_projection_optimizer_state_dict"
        )
        if not isinstance(state, dict) or not state:
            raise RuntimeError("semantic refresh lacks its optimizer state")
        optimizer.load_state_dict(state)
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
    blocks = []
    for pack_id, pack in enumerate(packs):
        count = len(pack["selected"])
        blocks.extend((pack_id, start, min(start + block_decisions, count)) for start in range(0, count, block_decisions))
    epoch_rows = []
    for epoch in range(epochs):
        order = list(blocks); random.Random(seed + epoch).shuffle(order)
        loss_sum = correct = decisions = options = 0.0
        for pack_id, start, end in order:
            pack = packs[pack_id]; offsets = pack["offsets"]
            option_start, option_end = int(offsets[start]), int(offsets[end])
            lengths_np = np.diff(np.asarray(offsets[start:end + 1], dtype=np.int64))
            selected_np = np.asarray(pack["selected"][start:end], dtype=np.int64)
            flat_np = np.asarray(pack["features"][option_start:option_end], dtype=np.float32)
            lengths = torch.from_numpy(lengths_np).to(device=device, dtype=torch.long)
            targets = torch.from_numpy(selected_np).to(device=device, dtype=torch.long)
            flat = torch.from_numpy(flat_np).to(device=device)
            batch = end - start; width = int(lengths.max().item())
            padded = torch.zeros((batch, width, 40), device=device)
            mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
            rows = torch.repeat_interleave(torch.arange(batch, device=device), lengths)
            starts = torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(lengths, 0)[:-1]))
            cols = torch.arange(len(flat), device=device) - torch.repeat_interleave(starts, lengths)
            padded[rows, cols] = flat; mask[rows, cols] = True
            optimizer.zero_grad(set_to_none=True)
            hidden = model.semantic_projection(padded)
            logits = model.logit_projection(hidden).squeeze(-1).tanh() * float(model.config.logit_delta_limit)
            logits = logits.masked_fill(~mask, -torch.inf)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("semantic bootstrap loss is nonfinite")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            n = end - start
            loss_sum += float(loss.detach()) * n
            correct += int((logits.argmax(-1) == targets).sum().item())
            decisions += n; options += len(flat)
        epoch_row = {"epoch": epoch + 1, "decisions": int(decisions), "options": int(options), "loss": loss_sum / decisions, "accuracy": correct / decisions}
        epoch_rows.append(epoch_row); print(json.dumps({"semantic_epoch": epoch_row}), flush=True)
    return ({name: value.detach().cpu() for name, value in model.state_dict().items()}, optimizer.state_dict(), {"epochs": epoch_rows, "decision_occurrences_per_epoch": int(completion["decision_occurrence_count"]), "option_occurrences_per_epoch": int(completion["option_occurrence_count"])})


def refresh_semantic_checkpoint(
    *,
    base_checkpoint: Path,
    output: Path,
    completion_path: Path,
    device: torch.device,
    epochs: int,
    block_decisions: int,
    learning_rate: float,
    seed: int,
    before_iteration: int,
) -> dict[str, Any]:
    """Train the derivative semantic projection in the same expert boundary.

    The ordinary rehearsal has already updated every architecture-present
    non-combo base tensor.  This second half starts from that immutable child,
    continues the semantic optimizer over the exact same every-occurrence
    recent-20 pack, and publishes one new composite checkpoint.  The base-only
    child remains recovery evidence and is never used as the refreshed learner.
    """

    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    completion_path = Path(completion_path).expanduser().resolve()
    candidate = checkpoint.load_checkpoint(base_checkpoint, map_location="cpu")
    if candidate.get("schema") != (
        "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
    ):
        raise RuntimeError("semantic refresh parent is not a derivative checkpoint")
    semantic_state, semantic_optimizer, semantic_result = train_semantic(
        candidate=candidate,
        completion_path=completion_path,
        device=device,
        epochs=int(epochs),
        block_decisions=int(block_decisions),
        learning_rate=float(learning_rate),
        seed=int(seed),
        resume_optimizer_from_candidate=True,
    )
    final = dict(candidate)
    final["public_rule_semantic_projection_state_dict"] = semantic_state
    final["public_rule_semantic_projection_optimizer_state_dict"] = (
        semantic_optimizer
    )
    extra = dict(final.get("extra") or {})
    refresh = {
        "schema": "poke_bot.alakazam_rule_derivative_semantic_refresh/v1",
        "before_iteration": int(before_iteration),
        "epochs_completed": int(epochs),
        "learning_rate": float(learning_rate),
        "semantic_pack_completion_path": str(completion_path),
        "semantic_pack_completion_sha256": sha256_file(completion_path),
        "base_only_checkpoint_path": str(base_checkpoint),
        "base_only_checkpoint_sha256": sha256_file(base_checkpoint),
        "training": semantic_result,
    }
    extra["alakazam_rule_derivative_semantic_refresh"] = refresh
    final["extra"] = extra
    save_torch(output, final)
    result = {
        **refresh,
        "checkpoint_path": str(output),
        "checkpoint_sha256": sha256_file(output),
        "checkpoint_size_bytes": output.stat().st_size,
    }
    receipt_path = output.with_suffix(".semantic-refresh.json")
    write_json(receipt_path, result)
    return {
        **result,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-pack", type=Path, required=True)
    parser.add_argument("--semantic-pack-completion", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--goal-gateway-sha256", required=True)
    parser.add_argument("--goal-contract-sha256", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--semantic-block-decisions", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=310)
    args = parser.parse_args()
    if args.epochs != 25:
        raise ValueError("revision-10 bootstrap requires exactly 25 epochs")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    candidate_sha = sha256_file(args.candidate)
    if candidate_sha != "sha256:16373d030de6aedd8407a936bdb590dcd9364068238bbc26acc846a2f9f5e2e2":
        raise RuntimeError("composite r195/r274 candidate identity drifted")
    candidate = checkpoint.load_checkpoint(args.candidate, map_location="cpu")
    device = torch.device(args.device); torch.cuda.set_device(device)
    core_cpu, side_cpu, pack_meta = load_pack(args.base_pack)
    validate_cpu_corpus(core_cpu)
    counts = validate_r279_pack(core_cpu, side_cpu, expected_games=26_704, expected_decisions=2_040_911)
    core_gpu = core_cpu.to_device(device, min_free_gib=16.0)
    side_gpu = move_side(side_cpu, device); del core_cpu, side_cpu; gc.collect()
    expanded = dict(dict(candidate.get("extra") or {}).get("expanded_head_training") or {})
    weights = dict(expanded.get("loss_weights") or {})
    schedule = {"schema": "poke_bot.expanded_head_schedule/v1", "runtime_enabled_heads": [], "loss_weights": weights, "schedule_digest": expanded.get("schedule_digest"), "target_schema": expanded.get("target_schema_version"), "target_schema_digest": expanded.get("target_schema_digest"), "stage_index": 0, "epoch": 25}
    base_output = args.output_root / "base-full-model-25epochs.pt"
    base_result = supervised_rehearsal_step(
        core_gpu, base_ckpt=args.candidate, output_path=base_output,
        parent_digest=checkpoint.checkpoint_digest(args.candidate), rehearsal_iteration=0,
        manifest_identity={"schema": "poke_bot.r279_contiguous_expert_pack/v1", "path": str(args.base_pack), "sha256": sha256_file(args.base_pack), "counts": counts, "contract": pack_meta["contract"]},
        epochs=25, lr=1e-5, requested_batch_size=args.max_decisions_per_batch,
        seed=args.seed, corpus_split_seed=args.seed, device=device,
        aux_loss_weight=0.05, opp_hand_loss_weight=0.05, opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025, prize_race_loss_weight=0.025,
        alakazam_guide_loss_weight=0.0, setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.0, visible_tutor_completion_loss_weight=0.025,
        terminal_conversion_loss_weight=0.025, tactical_sequence_outcome_loss_weight=0.0,
        expanded_head_loss_weights=weights, expanded_head_schedule=schedule,
        output_archetype_id="alakazam", output_model_id="alakazam-rule-derivative-r10-full",
        r279_side_tensors=side_gpu,
        extra_updates={"revision_10_full_model_bootstrap": {"r195_source_sha256": "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a", "all_architecture_present_non_combo_weights_optimizer_eligible": True, "combo_loss": 0.0, "combo_route_enabled": False}},
    )
    del core_gpu, side_gpu; gc.collect(); torch.cuda.empty_cache()
    semantic_state, semantic_optimizer, semantic_result = train_semantic(candidate=candidate, completion_path=args.semantic_pack_completion, device=device, epochs=25, block_decisions=args.semantic_block_decisions, learning_rate=3e-4, seed=args.seed)
    base_child = checkpoint.load_checkpoint(base_output, map_location="cpu")
    final = dict(base_child)
    final["schema"] = "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
    final["goal_revision"] = 10
    final["goal_gateway_sha256"] = args.goal_gateway_sha256
    final["goal_contract_sha256"] = args.goal_contract_sha256
    final["base_model_state_dict"] = final.pop("model_state_dict")
    final["public_rule_semantic_projection_config"] = dict(candidate["public_rule_semantic_projection_config"])
    final["public_rule_semantic_projection_state_dict"] = semantic_state
    final["public_rule_semantic_projection_optimizer_state_dict"] = semantic_optimizer
    final["eligible_trainable_branches"] = ["public_rule_semantic_projection"]
    final["unsupported_zero_inert_branches"] = list(candidate["unsupported_zero_inert_branches"])
    extra = dict(final.get("extra") or {})
    extra["revision_10_full_bootstrap"] = {"base_result": base_result, "semantic_result": semantic_result, "semantic_pack_completion_sha256": sha256_file(args.semantic_pack_completion), "epochs_completed": 25, "optimizer_scope": "all_architecture_present_non_combo_weights_plus_public_rule_semantic_projection", "combo_excluded": True}
    final["extra"] = extra
    output = args.output_root / "alakazam-rule-derivative-r10-full-bootstrap.pt"
    save_torch(output, final)
    receipt = {"schema": "poke_bot.alakazam_rule_derivative_r10_full_bootstrap_receipt/v1", "status": "completed", "candidate_parent_sha256": candidate_sha, "r195_source_sha256": "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a", "epochs_completed": 25, "base_training": base_result, "semantic_training": semantic_result, "output_path": str(output), "output_sha256": sha256_file(output), "output_size_bytes": output.stat().st_size, "elapsed_seconds": time.time() - started, "submission_authorized_after_validation": True}
    write_json(args.output_root / "COMPLETE.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
