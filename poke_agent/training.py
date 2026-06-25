from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from poke_agent.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    training_checkpoint_paths,
)
from poke_agent.dataset import TrainingTensors
from poke_agent.features import CARD_ID_SLOT_COUNT
from poke_agent.models.temporal_transformer import TemporalTransformer
from poke_agent.training_diversity import assert_generic_model_inputs


def _restore_training_state(
    resume_path: Path,
    *,
    model: TemporalTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    device: torch.device,
    use_amp: bool,
) -> tuple[int, float, int, int]:
    """Load model + optimizer + loop counters from a `.latest.pt` checkpoint.

    Returns (start_epoch, best_loss, best_epoch, epochs_without_improvement).
    """
    checkpoint, training_state = load_training_checkpoint(resume_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if not training_state:
        print(f"resume: {resume_path.name} has weights but no training_state; restarting epoch count.")
        return 0, float("inf"), 0, 0

    opt_state = training_state.get("optimizer_state_dict")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    scaler_state = training_state.get("scaler_state_dict")
    if use_amp and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(training_state.get("completed_epochs", 0))
    best_loss = float(training_state.get("best_monitor_loss", float("inf")))
    best_epoch = int(training_state.get("best_epoch", 0))
    without_improvement = int(training_state.get("epochs_without_improvement", 0))
    print(
        f"resume: {resume_path.name} -> start_epoch={start_epoch} "
        f"best={best_loss:.5f}@{best_epoch} patience_used={without_improvement}"
    )
    return start_epoch, best_loss, best_epoch, without_improvement


def build_model(config: dict[str, Any], tensors: TrainingTensors, device: torch.device) -> TemporalTransformer:
    model_cfg = config["model"]
    d_model = model_cfg["d_model"]
    heads = model_cfg["heads"]
    if d_model % heads != 0:
        raise ValueError("MODEL_D_MODEL must be divisible by MODEL_HEADS")

    model = TemporalTransformer(
        tensors.x_seq.shape[-1],
        tensors.transition_classes,
        d_model=d_model,
        nhead=heads,
        num_layers=model_cfg["layers"],
        dim_feedforward=model_cfg["ff"],
        dropout=model_cfg["dropout"],
        window_size=tensors.window_size,
        card_vocab_size=int(config.get("card_vocab_size", 2000)),
        card_embed_dim=int(config.get("card_embed_dim", 32)),
        card_slot_count=CARD_ID_SLOT_COUNT,
    ).to(device)
    if bool(config.get("train_grad_checkpoint", False)):
        model.set_gradient_checkpointing(True)
    print(
        f"model: d_model={d_model} heads={heads} "
        f"layers={model_cfg['layers']} ff={model_cfg['ff']} "
        f"dropout={model_cfg['dropout']} window={tensors.window_size}"
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def _policy_weights(
    returns: torch.Tensor,
    values: torch.Tensor,
    *,
    mode: str,
    beta: float,
    weight_max: float,
) -> torch.Tensor:
    if mode == "none":
        return torch.ones_like(returns)
    advantages = (returns - values.detach()) / max(beta, 1e-6)
    weights = torch.exp(advantages).clamp(max=weight_max)
    return weights / weights.mean().clamp(min=1e-6)


def _optimizer_param_groups(model: TemporalTransformer, config: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = config["model"]
    head_lr = float(model_cfg.get("head_learning_rate", model_cfg["learning_rate"]))
    encoder_lr = float(model_cfg.get("encoder_learning_rate", model_cfg["learning_rate"]))
    head_modules = (
        model.value_head,
        model.policy_head,
        model.next_feature_head,
        model.uncertainty_head,
    )
    head_param_ids = {id(param) for module in head_modules for param in module.parameters()}
    encoder_params = [param for param in model.parameters() if id(param) not in head_param_ids]
    head_params = [param for param in model.parameters() if id(param) in head_param_ids]
    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ]


def _policy_loss_batch(
    policy_logits: torch.Tensor,
    transition_b: torch.Tensor,
    *,
    soft_idx: torch.Tensor | None,
    soft_prob: torch.Tensor | None,
    use_soft_search: bool,
    policy_loss_fn: torch.nn.Module,
) -> torch.Tensor:
    hard_loss = policy_loss_fn(
        policy_logits.reshape(-1, policy_logits.shape[-1]),
        transition_b.reshape(-1),
    ).reshape_as(transition_b)

    if not use_soft_search or soft_idx is None or soft_prob is None:
        return hard_loss

    log_probs = torch.log_softmax(policy_logits, dim=-1)
    gathered = log_probs.gather(-1, soft_idx.clamp(min=0))
    soft_loss = -(gathered * soft_prob).sum(dim=-1)
    has_soft = (soft_prob.sum(dim=-1) > 0).to(dtype=hard_loss.dtype)
    return soft_loss * has_soft + hard_loss * (1.0 - has_soft)


def train_model(
    model: TemporalTransformer,
    tensors: TrainingTensors,
    config: dict[str, Any],
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_every_epochs: int = 0,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["training"]
    objective_cfg = config.get("objective", {})

    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(model, config),
        weight_decay=model_cfg["weight_decay"],
    )
    value_loss_fn = torch.nn.MSELoss(reduction="none")
    policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    dynamics_loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    epochs = train_cfg["epochs"]
    patience = train_cfg["patience"]
    min_delta = train_cfg["min_delta"]
    monitor_metric = str(train_cfg.get("early_stop_metric", "value_loss"))
    batch_games = train_cfg["batch_games"]
    use_amp = bool(config.get("train_use_amp", False)) and device.type == "cuda"
    policy_weighting = str(objective_cfg.get("policy_weighting", "awr"))
    awr_beta = float(objective_cfg.get("policy_awr_beta", 0.5))
    awr_weight_max = float(objective_cfg.get("policy_awr_weight_max", 20.0))
    use_soft_search = bool(objective_cfg.get("use_soft_search_policy", True))
    search_kl_weight = float(objective_cfg.get("search_policy_kl_weight", 1.0))

    assert_generic_model_inputs(model, tensors, config)

    num_seqs = tensors.num_seqs
    num_batches = max(1, (num_seqs + batch_games - 1) // batch_games)
    avg_steps = float(tensors.seq_mask.sum().item()) / max(1, num_seqs)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    last_metrics: dict[str, float] = {}
    stopped_early = False
    completed_epochs = 0
    start_epoch = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_paths = training_checkpoint_paths(checkpoint_path) if checkpoint_path is not None else None
    save_every = max(0, int(checkpoint_every_epochs))

    if resume_path is not None and Path(resume_path).exists():
        start_epoch, best_loss, best_epoch, epochs_without_improvement = _restore_training_state(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        completed_epochs = start_epoch

    def _persist(target: Path, epoch_index: int) -> None:
        if ckpt_paths is None:
            return
        if device.type == "cuda":
            torch.cuda.empty_cache()
        interim_report = {
            "device": str(device),
            "data_path": str(tensors.data_path) if tensors.data_path else None,
            "completed_epochs": epoch_index + 1,
            "requested_epochs": epochs,
            "best_epoch": best_epoch,
            "best_monitor_loss": best_loss,
            "early_stop_metric": monitor_metric,
            "last_metrics": last_metrics,
        }
        training_state = {
            "completed_epochs": epoch_index + 1,
            "requested_epochs": epochs,
            "best_epoch": best_epoch,
            "best_monitor_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stop_metric": monitor_metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
        }
        save_training_checkpoint(
            model=model,
            tensors=tensors,
            config=config,
            training_report=interim_report,
            output_path=target,
            training_state=training_state,
        )

    print(
        f"training batches: seqs={num_seqs} batch_games={batch_games} "
        f"batches={num_batches} (~{avg_steps:.0f} valid steps/seq)"
    )
    print(
        f"early stop: monitor={monitor_metric} patience={patience} "
        f"min_delta={min_delta} (stop when meaningful improvement plateaus)"
    )
    print(
        f"optimizer: encoder_lr={config['model'].get('encoder_learning_rate')} "
        f"head_lr={config['model'].get('head_learning_rate')}"
    )
    print(f"policy weighting: {policy_weighting} amp={use_amp} soft_search={use_soft_search}")
    if ckpt_paths is not None and save_every > 0:
        print(
            f"periodic checkpoints: every {save_every} epoch(s) -> {ckpt_paths['latest'].name} "
            f"(best -> {ckpt_paths['best'].name}); start_epoch={start_epoch}"
        )

    print_every = max(1, int(train_cfg.get("print_every", 1)))
    progress = tqdm(
        range(start_epoch, epochs),
        desc="training",
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
        initial=start_epoch,
        total=epochs,
    )
    for epoch in progress:
        seq_order = torch.randperm(num_seqs, device=tensors.x_seq.device)
        metric_sums = {
            "total_loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dynamics_loss": 0.0,
            "entropy": 0.0,
            "uncertainty_loss": 0.0,
        }
        seen = 0.0

        stream = tensors.x_seq.device != device
        for batch_number, start in enumerate(range(0, num_seqs, batch_games), start=1):
            seq_ids = seq_order[start:start + batch_games]
            # Gather the batch where the dataset lives, then move only that slice to
            # the compute device. When data is already on `device` this is a no-op.
            xb = tensors.x_seq[seq_ids].to(device, non_blocking=stream)
            mask_b = tensors.seq_mask[seq_ids].to(device, non_blocking=stream)
            card_b = tensors.card_ids[seq_ids].to(device, non_blocking=stream)
            yb = tensors.y[seq_ids].to(device, non_blocking=stream)
            returns_b = tensors.returns[seq_ids].to(device, non_blocking=stream)
            transition_b = tensors.transition_target[seq_ids].to(device, non_blocking=stream)
            soft_idx_b = tensors.transition_soft_idx[seq_ids].to(device, non_blocking=stream)
            soft_prob_b = tensors.transition_soft_prob[seq_ids].to(device, non_blocking=stream)
            policy_step_weight_b = tensors.policy_step_weight[seq_ids].to(device, non_blocking=stream)
            next_xb = tensors.next_x[seq_ids].to(device, non_blocking=stream)
            terminal_b = tensors.terminal[seq_ids].to(device, non_blocking=stream)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(xb, mask_b, card_ids=card_b)
                valid = mask_b > 0
                valid_count = valid.sum().clamp(min=1.0)

                value_loss = (value_loss_fn(out["value"], yb) * valid).sum() / valid_count
                policy_raw = _policy_loss_batch(
                    out["policy_logits"],
                    transition_b,
                    soft_idx=soft_idx_b,
                    soft_prob=soft_prob_b,
                    use_soft_search=use_soft_search,
                    policy_loss_fn=policy_loss_fn,
                )
                policy_weights = _policy_weights(
                    returns_b,
                    out["value"],
                    mode=policy_weighting,
                    beta=awr_beta,
                    weight_max=awr_weight_max,
                )
                policy_weights = policy_weights * policy_step_weight_b.pow(search_kl_weight)
                policy_loss = (policy_raw * policy_weights * valid).sum() / valid_count

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

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size_actual = float(valid_count.detach().cpu())
            seen += batch_size_actual
            metric_sums["total_loss"] += float(loss.detach().cpu()) * batch_size_actual
            metric_sums["value_loss"] += float(value_loss.detach().cpu()) * batch_size_actual
            metric_sums["policy_loss"] += float(policy_loss.detach().cpu()) * batch_size_actual
            metric_sums["dynamics_loss"] += float(dynamics_loss.detach().cpu()) * batch_size_actual
            metric_sums["entropy"] += float(entropy.detach().cpu()) * batch_size_actual
            metric_sums["uncertainty_loss"] += float(uncertainty_loss.detach().cpu()) * batch_size_actual
            if batch_number % print_every == 0 or batch_number == num_batches:
                progress.set_postfix({
                    "ep": f"{epoch + 1}/{epochs}",
                    "batch": f"{batch_number}/{num_batches}",
                    "loss": f"{float(loss.detach().cpu()):.5f}",
                    "v": f"{float(value_loss.detach().cpu()):.4f}",
                    "p": f"{float(policy_loss.detach().cpu()):.4f}",
                }, refresh=False)

        loss_value = metric_sums["total_loss"] / max(1.0, seen)
        last_metrics = {name: total / max(1.0, seen) for name, total in metric_sums.items()}
        # Play strength depends on value + policy; track them jointly so early stop
        # does not quit the moment value_loss saturates while policy is still learning.
        last_metrics["objective_loss"] = last_metrics.get("value_loss", 0.0) + last_metrics.get(
            "policy_loss", 0.0
        )
        completed_epochs = epoch + 1
        if monitor_metric == "total_loss":
            epoch_score = loss_value
        else:
            if monitor_metric not in last_metrics:
                raise ValueError(f"unknown early_stop_metric: {monitor_metric}")
            epoch_score = last_metrics[monitor_metric]

        improved = epoch_score < best_loss - min_delta
        if improved:
            best_loss = epoch_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if ckpt_paths is not None:
            # Best weights are the current model right after an improvement.
            if improved:
                _persist(ckpt_paths["best"], epoch)
            if save_every > 0 and ((epoch + 1) % save_every == 0 or epoch == start_epoch):
                _persist(ckpt_paths["latest"], epoch)

        progress.set_postfix({
            "loss": f"{loss_value:.5f}",
            "v": f"{last_metrics['value_loss']:.4f}",
            "p": f"{last_metrics['policy_loss']:.4f}",
            "dyn": f"{last_metrics['dynamics_loss']:.4f}",
            "best": f"{best_loss:.5f}@{best_epoch}",
            "mon": monitor_metric[:3],
            "patience": f"{epochs_without_improvement}/{patience}",
        })

        if epochs_without_improvement >= patience:
            progress.set_postfix({
                "loss": f"{loss_value:.5f}",
                "best": f"{best_loss:.5f}@{best_epoch}",
                "mon": monitor_metric[:3],
                "patience": f"{epochs_without_improvement}/{patience}",
                "stopped": "early",
            })
            print(
                f"early stopping at epoch={epoch + 1}; "
                f"best {monitor_metric}={best_loss:.5f}@{best_epoch} "
                f"(no improvement >= {min_delta} for {patience} epochs)"
            )
            stopped_early = True
            break
    progress.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "completed_epochs": completed_epochs,
        "requested_epochs": epochs,
        "resumed_from_epoch": start_epoch,
        "resume_path": str(resume_path) if resume_path is not None else None,
        "stopped_early": stopped_early,
        "early_stop_metric": monitor_metric,
        "best_total_loss": best_loss if monitor_metric == "total_loss" else last_metrics.get("total_loss", 0.0),
        "best_monitor_loss": best_loss,
        "best_epoch": best_epoch,
        "last_metrics": last_metrics,
        "dataset_rows": tensors.dataset_rows,
        "dataset_games": tensors.num_games,
        "dataset_seqs": tensors.num_seqs,
        "train_game_limit": config.get("training", {}).get("games"),
        "input_dim": int(tensors.x_seq.shape[-1]),
        "window_size": tensors.window_size,
        "batch_games": batch_games,
        "device": str(device),
        "data_path": str(tensors.data_path) if tensors.data_path else None,
        "reward_scheme": config.get("rewards"),
        "objective": config.get("objective"),
        "beam_search": config.get("beam_search"),
        "loss_note": (
            "Loss is the training objective, not winrate. Winrate requires CABT evaluation games "
            "using the model as the action policy."
        ),
    }
