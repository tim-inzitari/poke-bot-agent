"""Supervised bootstrap / AlphaZero training loops with AMP + checkpoints.

Uses whole-game causal temporal context (``return_all``) and reports live
tqdm progress over batches with loss / policy-acc / value metrics.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import archetypes, checkpoint, config, device as device_mod
from .dataset import BootstrapDataset, GameSequence
from .model import TemporalCabtTransformer, build_model


@dataclass
class TrainConfig:
    """Bootstrap supervised training knobs."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    games_per_batch: int = 4
    max_decisions_per_batch: int = 256
    val_frac: float = 0.1
    early_stop_patience: int = 5
    value_loss_weight: float = 1.0
    aux_loss_weight: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 0
    log_every: int = 1


@dataclass
class BatchMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    aux_loss: float = 0.0
    total_loss: float = 0.0
    policy_acc: float = 0.0
    n_decisions: int = 0
    n_games: int = 0


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    patience_left: int = 5
    history: list[dict[str, Any]] = field(default_factory=list)


def _archetype_label(name: str) -> int:
    ids = list(archetypes.archetype_ids()) + ["unknown"]
    try:
        return ids.index(name if name in ids else "unknown")
    except ValueError:
        return len(ids) - 1


def sequence_losses(
    model: TemporalCabtTransformer,
    seq: GameSequence,
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Whole-game forward + CE/Huber losses for one :class:`GameSequence`."""
    decisions = seq.decisions
    if not decisions:
        zero = torch.zeros((), device=next(model.parameters()).device)
        return zero, BatchMetrics()

    boards = [d.board for d in decisions]
    spatial = model.encode_board(boards)  # [T, 24, D]
    cls = model.pool_cls(spatial).unsqueeze(0)  # [1, T, D]
    states, _ = model.temporal_encode(cls, None, append=False, return_all=True)
    # states: [1, T, D]
    target_value = torch.tensor(
        float(seq.value), device=states.device, dtype=states.dtype
    )

    policy_losses: list[torch.Tensor] = []
    value_losses: list[torch.Tensor] = []
    correct = 0
    n_valid = 0

    for t, d in enumerate(decisions):
        state_t = states[0, t]
        spatial_t = spatial[t : t + 1]
        n_opt = d.options.num_words
        if n_opt <= 0:
            continue
        logits = model.decode_options(
            d.options, spatial_t, state_t, n_options=[n_opt]
        )[
            0, :n_opt
        ]

        # Policy target: one-hot BC action, or soft MCTS visit distribution.
        if (
            seq.policy_targets is not None
            and t < len(seq.policy_targets)
            and seq.policy_targets[t] is not None
        ):
            target = torch.tensor(
                seq.policy_targets[t][:n_opt],
                device=logits.device,
                dtype=logits.dtype,
            )
            if target.numel() != n_opt:
                continue
            target = target / target.sum().clamp_min(1e-8)
            log_p = F.log_softmax(logits, dim=-1)
            policy_losses.append(-(target * log_p).sum())
            pred = int(logits.argmax().item())
            correct += int(pred == int(target.argmax().item()))
        else:
            idx = int(d.action_combo_index)
            if idx < 0 or idx >= n_opt:
                continue
            policy_losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([idx], device=logits.device)))
            pred = int(logits.argmax().item())
            correct += int(pred == idx)

        value_pred = torch.tanh(model.value_head(state_t)).squeeze()
        value_losses.append(F.smooth_l1_loss(value_pred, target_value))
        n_valid += 1

    if n_valid == 0:
        zero = torch.zeros((), device=states.device, requires_grad=True)
        return zero, BatchMetrics(n_games=1)

    p_loss = torch.stack(policy_losses).mean()
    v_loss = torch.stack(value_losses).mean()
    aux_loss = torch.zeros((), device=states.device)
    if aux_weight > 0:
        # Aux on final state: predict opp archetype id.
        aux_logits = model.aux_head(states[0, -1])
        label = torch.tensor(
            [_archetype_label(seq.opp_archetype)],
            device=aux_logits.device,
            dtype=torch.long,
        )
        if label.item() < aux_logits.size(-1):
            aux_loss = F.cross_entropy(aux_logits.unsqueeze(0), label)

    total = p_loss + value_weight * v_loss + aux_weight * aux_loss
    metrics = BatchMetrics(
        policy_loss=float(p_loss.detach().item()),
        value_loss=float(v_loss.detach().item()),
        aux_loss=float(aux_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=correct / max(n_valid, 1),
        n_decisions=n_valid,
        n_games=1,
    )
    return total, metrics


def _merge_metrics(parts: Sequence[BatchMetrics]) -> BatchMetrics:
    if not parts:
        return BatchMetrics()
    nd = sum(p.n_decisions for p in parts)
    ng = sum(p.n_games for p in parts)
    if nd == 0:
        return BatchMetrics(n_games=ng)

    def wavg(attr: str) -> float:
        return sum(getattr(p, attr) * p.n_decisions for p in parts) / nd

    return BatchMetrics(
        policy_loss=wavg("policy_loss"),
        value_loss=wavg("value_loss"),
        aux_loss=wavg("aux_loss"),
        total_loss=wavg("total_loss"),
        policy_acc=wavg("policy_acc"),
        n_decisions=nd,
        n_games=ng,
    )


def split_dataset(
    ds: BootstrapDataset, val_frac: float, seed: int
) -> tuple[list[GameSequence], list[GameSequence]]:
    seqs = list(ds.sequences)
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(len(seqs) * val_frac)) if len(seqs) > 1 else 0
    if n_val == 0:
        return seqs, []
    return seqs[n_val:], seqs[:n_val]


def _iter_game_batches(
    sequences: list[GameSequence],
    games_per_batch: int,
    max_decisions: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[GameSequence]]:
    order = list(range(len(sequences)))
    if shuffle:
        rng = random.Random(seed + epoch * 10007)
        rng.shuffle(order)
    batches: list[list[GameSequence]] = []
    cur: list[GameSequence] = []
    cur_dec = 0
    for i in order:
        seq = sequences[i]
        n = len(seq)
        if cur and (
            len(cur) >= games_per_batch or cur_dec + n > max_decisions
        ):
            batches.append(cur)
            cur, cur_dec = [], 0
        cur.append(seq)
        cur_dec += n
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def evaluate(
    model: TemporalCabtTransformer,
    sequences: list[GameSequence],
    *,
    cfg: TrainConfig,
    desc: str = "val",
) -> BatchMetrics:
    model.eval()
    parts: list[BatchMetrics] = []
    for seq in tqdm(sequences, desc=desc, leave=False, unit="game"):
        _, m = sequence_losses(
            model,
            seq,
            value_weight=cfg.value_loss_weight,
            aux_weight=cfg.aux_loss_weight,
        )
        parts.append(m)
    return _merge_metrics(parts)


def train_bootstrap(
    dataset: BootstrapDataset,
    *,
    run_name: str = "dragapult_bootstrap",
    archetype_id: str = "dragapult",
    train_cfg: Optional[TrainConfig] = None,
    resume: Union[str, bool, None] = "auto",
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Run supervised BC/value training with early stopping + AMP + checkpoints."""
    cfg = train_cfg or TrainConfig()
    device = device or device_mod.training_device(
        prefer_name=config.HARDWARE.train_gpu_name, allow_cpu=False
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    train_seqs, val_seqs = split_dataset(dataset, cfg.val_frac, cfg.seed)
    print(
        f"[train] device={device} games={len(dataset)} "
        f"train={len(train_seqs)} val={len(val_seqs)} decisions={dataset.n_decisions}",
        flush=True,
    )

    model = build_model(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )

    state = TrainState(patience_left=cfg.early_stop_patience)
    mgr = checkpoint.CheckpointManager(run_name)

    resume_path = checkpoint.resolve_resume_path(run_name, resume)
    if resume_path is not None:
        print(f"[train] resuming from {resume_path}", flush=True)
        ckpt = checkpoint.load_checkpoint(resume_path, map_location=device)
        meta = checkpoint.apply_checkpoint(
            ckpt, model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler
        )
        state.step = int(meta["step"])
        state.epoch = int(meta["epoch"])
        if meta.get("best_metric") is not None:
            state.best_metric = float(meta["best_metric"])
        es = meta.get("early_stop_state") or {}
        if "patience_left" in es:
            state.patience_left = int(es["patience_left"])
        state.history = list((meta.get("extra") or {}).get("history") or [])

    def build_ckpt() -> dict[str, Any]:
        return checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            scheduler=scheduler,
            step=state.step,
            epoch=state.epoch,
            best_metric=state.best_metric,
            early_stop_state={
                "patience_left": state.patience_left,
                "best_metric": state.best_metric,
            },
            archetype_id=archetype_id,
            model_id=run_name,
            extra={"history": state.history, "train_cfg": cfg.__dict__},
        )

    mgr.install_signal_flush(build_ckpt)

    try:
        epoch_bar = tqdm(
            range(state.epoch, cfg.epochs),
            desc="epochs",
            initial=state.epoch,
            total=cfg.epochs,
            unit="ep",
        )
        for epoch in epoch_bar:
            state.epoch = epoch
            model.train()
            batches = _iter_game_batches(
                train_seqs,
                cfg.games_per_batch,
                cfg.max_decisions_per_batch,
                shuffle=True,
                seed=cfg.seed,
                epoch=epoch,
            )
            epoch_parts: list[BatchMetrics] = []
            batch_bar = tqdm(batches, desc=f"train ep{epoch}", leave=False, unit="batch")
            for batch in batch_bar:
                optimizer.zero_grad(set_to_none=True)
                losses: list[torch.Tensor] = []
                metrics_parts: list[BatchMetrics] = []
                with torch.amp.autocast("cuda", enabled=use_amp):
                    for seq in batch:
                        loss, m = sequence_losses(
                            model,
                            seq,
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                        )
                        losses.append(loss)
                        metrics_parts.append(m)
                    if not losses:
                        continue
                    total = torch.stack(losses).mean()

                scaler.scale(total).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                state.step += 1
                bm = _merge_metrics(metrics_parts)
                epoch_parts.append(bm)
                batch_bar.set_postfix(
                    loss=f"{bm.total_loss:.3f}",
                    p=f"{bm.policy_loss:.3f}",
                    v=f"{bm.value_loss:.3f}",
                    acc=f"{bm.policy_acc:.2%}",
                    step=state.step,
                )

                saved = mgr.maybe_save(state.step, build_ckpt)
                if saved:
                    tqdm.write(
                        f"[checkpoint] step={state.step} saved → "
                        + ", ".join(f"{k}={v.name}" for k, v in saved.items())
                    )

            train_m = _merge_metrics(epoch_parts)
            if val_seqs:
                val_m = evaluate(model, val_seqs, cfg=cfg, desc=f"val ep{epoch}")
                metric = val_m.total_loss
            else:
                val_m = train_m
                metric = train_m.total_loss

            scheduler.step()
            row = {
                "epoch": epoch,
                "step": state.step,
                "train": train_m.__dict__,
                "val": val_m.__dict__,
                "lr": optimizer.param_groups[0]["lr"],
                "t": time.time(),
            }
            state.history.append(row)

            is_best = metric < state.best_metric - 1e-5
            if is_best:
                state.best_metric = metric
                state.patience_left = cfg.early_stop_patience
                mgr.save(build_ckpt(), is_best=True)
                tqdm.write(
                    f"[checkpoint] NEW BEST epoch={epoch} val_loss={metric:.4f} "
                    f"val_acc={val_m.policy_acc:.2%}"
                )
            else:
                state.patience_left -= 1
                mgr.save(build_ckpt(), is_best=False)
                tqdm.write(
                    f"[train] epoch={epoch} train_loss={train_m.total_loss:.4f} "
                    f"val_loss={metric:.4f} val_acc={val_m.policy_acc:.2%} "
                    f"patience={state.patience_left}"
                )

            epoch_bar.set_postfix(
                val_loss=f"{metric:.4f}",
                val_acc=f"{val_m.policy_acc:.2%}",
                best=f"{state.best_metric:.4f}",
                pat=state.patience_left,
            )

            if state.patience_left <= 0:
                tqdm.write(
                    f"[early-stop] patience exhausted at epoch={epoch} "
                    f"best_val_loss={state.best_metric:.4f}"
                )
                break
    finally:
        mgr.uninstall_signal_flush()
        # Final flush.
        try:
            mgr.save(build_ckpt(), is_best=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] final save failed: {exc}", flush=True)

    best = checkpoint.best_path(run_name)
    latest = checkpoint.latest_path(run_name)
    return {
        "run_name": run_name,
        "best_metric": state.best_metric,
        "step": state.step,
        "epoch": state.epoch,
        "best_path": str(best) if best.is_file() else None,
        "latest_path": str(latest) if latest.is_file() else None,
        "history": state.history,
    }


def load_model_from_checkpoint(
    path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> TemporalCabtTransformer:
    device = device or device_mod.inference_device(allow_cpu=True)
    ckpt = checkpoint.load_checkpoint(path, map_location=device)
    model = build_model(device=device)
    checkpoint.apply_checkpoint(ckpt, model=model, restore_rng=False)
    model.eval()
    return model
