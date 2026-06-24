from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from poke_agent.dataset import TrainingTensors


def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def estimate_dataset_bytes(tensors: TrainingTensors) -> int:
    return sum(
        _tensor_bytes(tensor)
        for tensor in (
            tensors.x_seq,
            tensors.y,
            tensors.returns,
            tensors.transition_target,
            tensors.next_x,
            tensors.terminal,
            tensors.seq_mask,
            tensors.card_ids,
            tensors.seq_lengths,
        )
    )


def _static_training_bytes(param_count: int, tensors: TrainingTensors) -> dict[str, int]:
    bytes_per_param = 4
    return {
        "dataset_bytes": estimate_dataset_bytes(tensors),
        "model_weights": param_count * bytes_per_param,
        "gradients": param_count * bytes_per_param,
        "optimizer_state": param_count * bytes_per_param * 2,
    }


def _training_step_peak_bytes(
    model: nn.Module,
    tensors: TrainingTensors,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, int]:
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    objective_cfg = config.get("objective", {})
    batch_games = int(config["training"]["batch_games"])
    seq_ids = torch.arange(min(batch_games, tensors.num_seqs), device=device)

    value_loss_fn = nn.MSELoss(reduction="none")
    policy_loss_fn = nn.CrossEntropyLoss(reduction="none")
    dynamics_loss_fn = nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before_step = torch.cuda.memory_allocated(device)

    model.train()
    xb = tensors.x_seq[seq_ids]
    mask_b = tensors.seq_mask[seq_ids]
    card_b = tensors.card_ids[seq_ids]
    yb = tensors.y[seq_ids]
    returns_b = tensors.returns[seq_ids]
    transition_b = tensors.transition_target[seq_ids]
    next_xb = tensors.next_x[seq_ids]
    terminal_b = tensors.terminal[seq_ids]

    optimizer.zero_grad(set_to_none=True)
    out = model(xb, mask_b, card_ids=card_b)
    valid = mask_b > 0
    valid_count = valid.sum().clamp(min=1.0)
    value_loss = (value_loss_fn(out["value"], yb) * valid).sum() / valid_count
    policy_raw = policy_loss_fn(
        out["policy_logits"].reshape(-1, out["policy_logits"].shape[-1]),
        transition_b.reshape(-1),
    ).reshape_as(transition_b)
    policy_loss = (policy_raw * valid).sum() / valid_count
    nonterminal = (1.0 - terminal_b) * valid
    dynamics_loss = (
        dynamics_loss_fn(out["next_features"], next_xb) * nonterminal.unsqueeze(-1)
    ).sum() / nonterminal.sum().clamp(min=1.0)
    probs = torch.softmax(out["policy_logits"], dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    entropy = (entropy * valid).sum() / valid_count
    uncertainty_loss = (
        (torch.exp(-out["log_variance"]) * (out["value"] - yb).pow(2) + out["log_variance"]) * valid
    ).sum() / valid_count
    loss = (
        loss_cfg["value"] * value_loss
        + loss_cfg["policy"] * policy_loss
        + loss_cfg["dynamics"] * dynamics_loss
        - loss_cfg["entropy"] * entropy
        + loss_cfg["uncertainty"] * uncertainty_loss
    )
    _ = returns_b, objective_cfg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    del optimizer, out, loss, xb, mask_b, card_b, yb, transition_b, next_xb, terminal_b
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return {
        "allocated_before_step": int(allocated_before_step),
        "peak_allocated": int(peak_allocated),
        "step_overhead": int(peak_allocated - allocated_before_step),
    }


def print_vram_estimate(
    *,
    model: nn.Module,
    param_count: int,
    tensors: TrainingTensors,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    static = _static_training_bytes(param_count, tensors)
    batch_games = int(config["training"]["batch_games"])
    avg_steps = float(tensors.seq_mask.sum().item()) / max(1, tensors.num_seqs)

    print("\nVRAM estimate for current config")
    print("-" * 34)
    print(f"device: {device}")
    print(
        f"data: {tensors.num_seqs:,} seat-sequences x window={tensors.window_size} x "
        f"{int(tensors.x_seq.shape[-1])} features "
        f"(batch_games={batch_games}, ~{avg_steps:.0f} valid steps/seq)"
    )
    print(f"parameters: {param_count:,}")
    print(f"dataset tensors: {format_bytes(static['dataset_bytes'])}")
    print(f"model weights:   {format_bytes(static['model_weights'])}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        currently_allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

        print(f"currently in use: {format_bytes(currently_allocated)} allocated, {format_bytes(reserved)} reserved")
        print("probing one real training step...")
        measured = _training_step_peak_bytes(model, tensors, config, device)
        print(f"measured peak:    {format_bytes(measured['peak_allocated'])} (after 1 forward+backward+adam step)")
        print(f"step overhead:    {format_bytes(measured['step_overhead'])} above pre-step allocation")
        print(
            f"gpu present:      {format_bytes(total_bytes)} total, "
            f"{format_bytes(free_bytes)} free right now"
        )
        if measured["peak_allocated"] > total_bytes:
            print("warning: measured peak exceeds total GPU memory")
        elif measured["peak_allocated"] > free_bytes + currently_allocated:
            print("warning: measured peak may exceed available GPU memory")
        return

    theoretical = (
        static["dataset_bytes"]
        + static["model_weights"]
        + static["gradients"]
        + static["optimizer_state"]
    )
    print(f"gradients+adam:  {format_bytes(static['gradients'] + static['optimizer_state'])} (not measured on {device.type})")
    print(f"estimated total: {format_bytes(theoretical)} + one batch of activations")
