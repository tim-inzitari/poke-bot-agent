"""Stage A — archetype-wide option-conditioned behavior cloning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .authority import RESEARCH_ONLY, TRAINING_AUTHORITY
from .heuristic_features import attach_heuristic_features, heuristic_mask


@dataclass
class StageAConfig:
    d_model: int = 64
    epochs: int = 2
    lr: float = 1e-3
    batch_size: int = 32
    heuristic_aux_weight: float = 0.1
    zero_heuristic_channel: bool = False
    seed: int = 0


class OptionConditionedClone(nn.Module):
    """Lightweight research actor: scores N legal option embeddings."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.state_enc = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.option_enc = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.score = nn.Linear(2 * d_model, 1)

    def forward(self, state_feat: Tensor, option_feat: Tensor) -> Tensor:
        """state_feat [B,16], option_feat [B,N,8] -> logits [B,N]."""
        h = self.state_enc(state_feat).unsqueeze(1).expand(-1, option_feat.size(1), -1)
        o = self.option_enc(option_feat)
        return self.score(torch.cat((h, o), dim=-1)).squeeze(-1)


def _state_features(row: dict[str, Any]) -> list[float]:
    obs = row.get("observation") or {}
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    feat = [
        float(current.get("turn") or 0) / 20.0,
        float(current.get("yourIndex") == current.get("firstPlayer")),
        float(row.get("legal_action_count") or 0) / 32.0,
        float(select.get("context") or 0) / 16.0,
        float(row.get("reward") or 0),
        float(1.0 if row.get("result") == "win" else 0.0),
        float(1.0 if row.get("turn_order") == "first" else 0.0),
        float(0.0 if row.get("heuristic_abstained") else 1.0),
    ]
    while len(feat) < 16:
        feat.append(0.0)
    return feat[:16]


def _option_features(row: dict[str, Any]) -> list[list[float]]:
    legal = row.get("legal_action_combos") or []
    heuristic = row.get("heuristic") or {}
    scores = list(heuristic.get("scores") or [0.0] * len(legal))
    mask = heuristic_mask(row)
    feats: list[list[float]] = []
    for i, combo in enumerate(legal):
        feats.append(
            [
                float(len(combo)),
                float(combo[0] if combo else -1) / 32.0,
                float(scores[i] if i < len(scores) else 0.0) / 8.0,
                float(mask[i] if i < len(mask) else 0.0),
                float(i == row.get("selected_index")),
                float(i) / max(1, len(legal)),
                1.0,
                0.0,
            ]
        )
    return feats


def _pad_options(batch_feats: list[list[list[float]]]) -> tuple[Tensor, Tensor, Tensor]:
    n_max = max(len(x) for x in batch_feats)
    d = len(batch_feats[0][0]) if batch_feats and batch_feats[0] else 8
    out = torch.zeros(len(batch_feats), n_max, d)
    mask = torch.zeros(len(batch_feats), n_max, dtype=torch.bool)
    for b, feats in enumerate(batch_feats):
        for i, row in enumerate(feats):
            out[b, i] = torch.tensor(row, dtype=torch.float32)
            mask[b, i] = True
    return out, mask, torch.tensor([len(x) for x in batch_feats])


@dataclass
class StageAResult:
    metrics: dict[str, float]
    checkpoint_path: Optional[str]
    inventory: dict[str, Any]


def run_stage_a_bc(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    config: Optional[StageAConfig] = None,
) -> StageAResult:
    """Train the research option-conditioned clone on decision rows."""
    cfg = config or StageAConfig()
    if not TRAINING_AUTHORITY:
        # Research lane may train offline artifacts, but never promotes them.
        pass
    prepared = [
        attach_heuristic_features(row, zero_channel=cfg.zero_heuristic_channel)
        for row in rows
        if int(row.get("legal_action_count") or 0) >= 1
        and row.get("selected_index") is not None
    ]
    if not prepared:
        raise ValueError("Stage A requires non-empty decision rows")

    torch.manual_seed(int(cfg.seed))
    model = OptionConditionedClone(cfg.d_model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    def batches():
        for start in range(0, len(prepared), cfg.batch_size):
            yield prepared[start : start + cfg.batch_size]

    last_loss = 0.0
    last_acc = 0.0
    for _epoch in range(int(cfg.epochs)):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in batches():
            state = torch.tensor([_state_features(r) for r in batch], dtype=torch.float32)
            option_feats, legal_mask, _counts = _pad_options([_option_features(r) for r in batch])
            targets = torch.tensor([int(r["selected_index"]) for r in batch], dtype=torch.long)
            logits = model(state, option_feats)
            logits = logits.masked_fill(~legal_mask, float("-inf"))
            loss = F.cross_entropy(logits, targets)
            if cfg.heuristic_aux_weight > 0 and not cfg.zero_heuristic_channel:
                # Encourage agreement on covered heuristic rows only.
                aux = []
                for i, row in enumerate(batch):
                    h = row.get("heuristic") or {}
                    pref = h.get("preferred_combo_index")
                    if row.get("heuristic_abstained") or pref is None:
                        continue
                    aux.append(F.cross_entropy(logits[i].unsqueeze(0), torch.tensor([int(pref)])))
                if aux:
                    loss = loss + float(cfg.heuristic_aux_weight) * torch.stack(aux).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(batch)
            pred = torch.argmax(logits, dim=-1)
            total_correct += int((pred == targets).sum().item())
            total += len(batch)
        last_loss = total_loss / max(1, total)
        last_acc = total_correct / max(1, total)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "stage_a_option_clone.pt"
    torch.save(
        {
            "schema": "poke_bot.slowking_distill.stage_a/v1",
            "research_only": RESEARCH_ONLY,
            "training_authority": False,
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "metrics": {"loss": last_loss, "action_agreement": last_acc},
        },
        ckpt,
    )
    metrics_path = out_dir / "stage_a_metrics.json"
    metrics = {
        "loss": last_loss,
        "action_agreement": last_acc,
        "n_rows": float(len(prepared)),
        "zero_heuristic_channel": float(cfg.zero_heuristic_channel),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return StageAResult(
        metrics=metrics,
        checkpoint_path=str(ckpt),
        inventory={
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "research_only": RESEARCH_ONLY,
        },
    )


def agreement_by_stage(rows: list[dict[str, Any]], model: OptionConditionedClone) -> dict[str, Any]:
    """Report top-1 agreement overall and on heuristic-covered stages."""
    model.eval()
    buckets: dict[str, list[bool]] = {"all": [], "heuristic_covered": [], "heuristic_abstained": []}
    with torch.no_grad():
        for row in rows:
            row = attach_heuristic_features(row)
            if int(row.get("legal_action_count") or 0) < 1:
                continue
            state = torch.tensor([_state_features(row)], dtype=torch.float32)
            opts, mask, _ = _pad_options([_option_features(row)])
            logits = model(state, opts).masked_fill(~mask, float("-inf"))
            pred = int(torch.argmax(logits, dim=-1).item())
            ok = pred == int(row["selected_index"])
            buckets["all"].append(ok)
            key = "heuristic_covered" if not row.get("heuristic_abstained") else "heuristic_abstained"
            buckets[key].append(ok)
    return {
        key: {
            "n": len(vals),
            "agreement": (sum(vals) / len(vals)) if vals else None,
        }
        for key, vals in buckets.items()
    }
