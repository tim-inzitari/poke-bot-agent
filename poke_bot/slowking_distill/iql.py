"""Stage B — IQL-style expectile critic and advantage-weighted policy update."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .authority import RESEARCH_ONLY
from .bc_stage import OptionConditionedClone, _option_features, _pad_options, _state_features
from .heuristic_features import attach_heuristic_features


def expectile_loss(diff: Tensor, expectile: float) -> Tensor:
    """IQL expectile regression loss."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


@dataclass
class IQLConfig:
    d_model: int = 64
    epochs: int = 2
    lr: float = 1e-3
    batch_size: int = 32
    expectile: float = 0.7
    awr_beta: float = 1.0
    awr_weight_max: float = 20.0
    discount: float = 1.0
    seed: int = 0
    require_calibration: bool = True
    min_rank_correlation: float = 0.05


class Critic(nn.Module):
    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(16, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q = nn.Sequential(nn.Linear(16 + 8, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def value(self, state_feat: Tensor) -> Tensor:
        return self.v(state_feat).squeeze(-1)

    def q_value(self, state_feat: Tensor, option_feat: Tensor) -> Tensor:
        # option_feat [B,8] chosen option
        return self.q(torch.cat((state_feat, option_feat), dim=-1)).squeeze(-1)


@dataclass
class IQLResult:
    metrics: dict[str, float]
    checkpoint_path: Optional[str]
    advantage_weighting_enabled: bool


def _chosen_option_feat(row: dict[str, Any]) -> list[float]:
    feats = _option_features(row)
    idx = int(row["selected_index"])
    if not feats:
        return [0.0] * 8
    return feats[max(0, min(idx, len(feats) - 1))]


def spearman_rank_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return float(num / (denx * deny))


def calibrate_critic(
    critic: Critic,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    critic.eval()
    values: list[float] = []
    returns: list[float] = []
    with torch.no_grad():
        for row in rows:
            state = torch.tensor([_state_features(row)], dtype=torch.float32)
            values.append(float(critic.value(state)[0].item()))
            returns.append(float(row.get("value_target") or row.get("reward") or 0.0))
    corr = spearman_rank_corr(values, returns)
    mse = sum((v - r) ** 2 for v, r in zip(values, returns)) / max(1, len(values))
    return {"rank_correlation": corr, "value_mse": mse, "n": float(len(values))}


def run_stage_b_iql(
    rows: list[dict[str, Any]],
    *,
    actor: OptionConditionedClone,
    out_dir: Path,
    config: Optional[IQLConfig] = None,
) -> IQLResult:
    """Fit expectile V/Q then optionally advantage-weight the actor."""
    cfg = config or IQLConfig()
    prepared = [attach_heuristic_features(r) for r in rows if r.get("selected_index") is not None]
    if not prepared:
        raise ValueError("Stage B requires decision rows")

    torch.manual_seed(int(cfg.seed))
    critic = Critic(cfg.d_model)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    def batches():
        for start in range(0, len(prepared), cfg.batch_size):
            yield prepared[start : start + cfg.batch_size]

    for _ in range(int(cfg.epochs)):
        for batch in batches():
            state = torch.tensor([_state_features(r) for r in batch], dtype=torch.float32)
            opt_feat = torch.tensor([_chosen_option_feat(r) for r in batch], dtype=torch.float32)
            returns = torch.tensor(
                [float(r.get("value_target") or r.get("reward") or 0.0) for r in batch],
                dtype=torch.float32,
            )
            v = critic.value(state)
            q = critic.q_value(state, opt_feat)
            # Expectile V toward Q; Q toward return (terminal offline rows).
            loss_v = expectile_loss(q.detach() - v, cfg.expectile)
            loss_q = F.mse_loss(q, returns)
            loss = loss_v + loss_q
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()

    calibration = calibrate_critic(critic, prepared)
    weighting_enabled = (not cfg.require_calibration) or (
        calibration["rank_correlation"] >= cfg.min_rank_correlation
    )

    actor_loss = None
    if weighting_enabled:
        opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
        actor.train()
        for _ in range(int(cfg.epochs)):
            total = 0.0
            n = 0
            for batch in batches():
                state = torch.tensor([_state_features(r) for r in batch], dtype=torch.float32)
                option_feats, legal_mask, _ = _pad_options([_option_features(r) for r in batch])
                targets = torch.tensor([int(r["selected_index"]) for r in batch], dtype=torch.long)
                with torch.no_grad():
                    v = critic.value(state)
                    opt_feat = torch.tensor(
                        [_chosen_option_feat(r) for r in batch], dtype=torch.float32
                    )
                    q = critic.q_value(state, opt_feat)
                    adv = q - v
                    weights = torch.exp(adv / max(1e-6, cfg.awr_beta)).clamp(
                        max=cfg.awr_weight_max
                    )
                logits = actor(state, option_feats).masked_fill(~legal_mask, float("-inf"))
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                loss = (weights * nll).mean()
                opt_a.zero_grad()
                loss.backward()
                opt_a.step()
                total += float(loss.item()) * len(batch)
                n += len(batch)
            actor_loss = total / max(1, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "stage_b_iql.pt"
    torch.save(
        {
            "schema": "poke_bot.slowking_distill.stage_b_iql/v1",
            "research_only": RESEARCH_ONLY,
            "training_authority": False,
            "critic": critic.state_dict(),
            "actor": actor.state_dict(),
            "advantage_weighting_enabled": weighting_enabled,
            "calibration": calibration,
            "config": cfg.__dict__,
        },
        ckpt,
    )
    metrics = {
        **calibration,
        "advantage_weighting_enabled": float(weighting_enabled),
        "actor_awr_loss": float(actor_loss or 0.0),
    }
    (out_dir / "stage_b_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return IQLResult(
        metrics=metrics,
        checkpoint_path=str(ckpt),
        advantage_weighting_enabled=weighting_enabled,
    )
