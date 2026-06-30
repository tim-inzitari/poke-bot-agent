from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from poke_agent.dataset import TrainingTensors
from poke_agent.models.temporal_transformer import TemporalTransformer
from poke_agent.training_diversity import assert_generic_model_inputs


def build_model(config: dict[str, Any], tensors: TrainingTensors, device: torch.device) -> TemporalTransformer:
    model_cfg = config["model"]
    d_model = model_cfg["d_model"]
    heads = model_cfg["heads"]
    use_kan = bool(model_cfg.get("use_kan", True))
    if d_model % heads != 0:
        raise ValueError("MODEL_D_MODEL must be divisible by MODEL_HEADS")

    model = TemporalTransformer(
        tensors.x.shape[1],
        tensors.transition_classes,
        d_model=d_model,
        nhead=heads,
        num_layers=model_cfg["layers"],
        dim_feedforward=model_cfg["ff"],
        dropout=model_cfg["dropout"],
        window_size=tensors.window_size,
        kan_grid_size=int(model_cfg.get("kan_grid_size", 8)),
        use_kan=use_kan,
    ).to(device)
    print(
        f"model: {'temporal_kan' if use_kan else 'temporal_transformer'} "
        f"d_model={d_model} heads={heads} "
        f"layers={model_cfg['layers']} ff={model_cfg['ff']} "
        f"dropout={model_cfg['dropout']} window={tensors.window_size} "
        f"kan_grid={model_cfg.get('kan_grid_size', 8)}"
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def train_model(
    model: TemporalTransformer,
    tensors: TrainingTensors,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["training"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )
    value_loss_fn = torch.nn.MSELoss()
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    dynamics_loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    epochs = train_cfg["epochs"]
    patience = train_cfg["patience"]
    min_delta = train_cfg["min_delta"]
    batch_games = train_cfg["batch_games"]

    assert_generic_model_inputs(model, tensors, config)

    num_rows = int(tensors.x.shape[0])
    num_games = tensors.num_games
    num_batches = max(1, (num_games + batch_games - 1) // batch_games)
    avg_steps = num_rows / max(1, num_games)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    last_metrics: dict[str, float] = {}
    stopped_early = False
    completed_epochs = 0

    print(
        f"training batches: games={num_games} batch_games={batch_games} "
        f"batches={num_batches} (~{avg_steps:.0f} steps/game)"
    )

    progress = tqdm(range(epochs), desc="training", unit="epoch")
    for epoch in progress:
        game_order = torch.randperm(num_games, device=device)
        metric_sums = {
            "total_loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dynamics_loss": 0.0,
            "entropy": 0.0,
            "uncertainty_loss": 0.0,
        }
        seen = 0

        batch_progress = tqdm(
            range(0, num_games, batch_games),
            desc=f"epoch {epoch + 1}/{epochs} batches",
            unit="batch",
            total=num_batches,
            leave=False,
        )
        for batch_number, start in enumerate(batch_progress, start=1):
            game_ids = game_order[start:start + batch_games]
            batch_idx = tensors.row_indices_for_games(game_ids)
            xb = tensors.x_padded[tensors.history_index[batch_idx]]
            mask_b = tensors.history_mask[batch_idx]
            yb = tensors.y[batch_idx]
            transition_b = tensors.transition_target[batch_idx]
            next_xb = tensors.next_x[batch_idx]
            terminal_b = tensors.terminal[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            out = model(xb, mask_b)

            value_loss = value_loss_fn(out["value"], yb)
            policy_loss = policy_loss_fn(out["policy_logits"], transition_b)
            nonterminal = (1.0 - terminal_b).unsqueeze(-1)
            dynamics_loss = (dynamics_loss_fn(out["next_features"], next_xb) * nonterminal).sum() / nonterminal.sum().clamp_min(1.0)
            probs = torch.softmax(out["policy_logits"], dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            uncertainty_loss = torch.mean(torch.exp(-out["log_variance"]) * (out["value"] - yb).pow(2) + out["log_variance"])

            loss = (
                loss_cfg["value"] * value_loss
                + loss_cfg["policy"] * policy_loss
                + loss_cfg["dynamics"] * dynamics_loss
                - loss_cfg["entropy"] * entropy
                + loss_cfg["uncertainty"] * uncertainty_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = int(xb.shape[0])
            seen += batch_size_actual
            metric_sums["total_loss"] += float(loss.detach().cpu()) * batch_size_actual
            metric_sums["value_loss"] += float(value_loss.detach().cpu()) * batch_size_actual
            metric_sums["policy_loss"] += float(policy_loss.detach().cpu()) * batch_size_actual
            metric_sums["dynamics_loss"] += float(dynamics_loss.detach().cpu()) * batch_size_actual
            metric_sums["entropy"] += float(entropy.detach().cpu()) * batch_size_actual
            metric_sums["uncertainty_loss"] += float(uncertainty_loss.detach().cpu()) * batch_size_actual
            batch_progress.set_postfix({
                "batch": f"{batch_number}/{num_batches}",
                "games": int(game_ids.shape[0]),
                "steps": batch_size_actual,
                "loss": f"{float(loss.detach().cpu()):.5f}",
                "v": f"{float(value_loss.detach().cpu()):.4f}",
                "p": f"{float(policy_loss.detach().cpu()):.4f}",
                "dyn": f"{float(dynamics_loss.detach().cpu()):.4f}",
            })

        loss_value = metric_sums["total_loss"] / max(1, seen)
        last_metrics = {name: total / max(1, seen) for name, total in metric_sums.items()}
        completed_epochs = epoch + 1
        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress.set_postfix({
            "loss": f"{loss_value:.5f}",
            "v": f"{last_metrics['value_loss']:.4f}",
            "p": f"{last_metrics['policy_loss']:.4f}",
            "dyn": f"{last_metrics['dynamics_loss']:.4f}",
            "best": f"{best_loss:.5f}@{best_epoch}",
            "patience": f"{epochs_without_improvement}/{patience}",
        })

        if epochs_without_improvement >= patience:
            progress.set_postfix({
                "loss": f"{loss_value:.5f}",
                "best": f"{best_loss:.5f}@{best_epoch}",
                "patience": f"{epochs_without_improvement}/{patience}",
                "stopped": "early",
            })
            print(f"early stopping at epoch={epoch + 1}; best={best_loss:.5f}@{best_epoch}")
            stopped_early = True
            break
    progress.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "completed_epochs": completed_epochs,
        "requested_epochs": epochs,
        "stopped_early": stopped_early,
        "best_total_loss": best_loss,
        "best_epoch": best_epoch,
        "last_metrics": last_metrics,
        "dataset_rows": int(tensors.x.shape[0]),
        "dataset_games": tensors.num_games,
        "train_game_limit": config.get("training", {}).get("games"),
        "input_dim": int(tensors.x.shape[1]),
        "window_size": tensors.window_size,
        "batch_games": batch_games,
        "device": str(device),
        "data_path": str(tensors.data_path) if tensors.data_path else None,
        "reward_scheme": config.get("rewards"),
        "beam_search": config.get("beam_search"),
        "loss_note": (
            "Loss is the training objective, not winrate. Winrate requires CABT evaluation games "
            "using the model as the action policy."
        ),
    }
