"""Stage E — distill search visit distributions into the fast actor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .authority import RESEARCH_ONLY
from .bc_stage import OptionConditionedClone, StageAConfig, _option_features, _pad_options, _state_features
from .heuristic_features import attach_heuristic_features


def load_search_receipts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _visit_target(receipt: dict[str, Any], n_legal: int) -> list[float]:
    visits = [float(v) for v in (receipt.get("visit_counts") or [])]
    if not visits:
        return [0.0] * n_legal
    # Align by candidate prefix; remaining mass zero.
    out = [0.0] * n_legal
    total = sum(visits) or 1.0
    for i, v in enumerate(visits):
        if i < n_legal:
            out[i] = v / total
    return out


@dataclass
class DistillResult:
    metrics: dict[str, float]
    checkpoint_path: Optional[str]


def run_stage_e_distill(
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    *,
    actor: OptionConditionedClone,
    out_dir: Path,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 0,
) -> DistillResult:
    """KL / CE distill from search visit targets into the option-conditioned actor."""
    by_key = {
        (str(r.get("game_id")), int(r.get("env_step", -1))): r for r in receipts
    }
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        key = (str(row.get("game_id")), int(row.get("env_step", -1)))
        if key in by_key:
            paired.append((attach_heuristic_features(row), by_key[key]))
    if not paired:
        raise ValueError("Stage E requires overlapping rows and search receipts")

    torch.manual_seed(seed)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    actor.train()
    last_loss = 0.0
    for _ in range(int(epochs)):
        total = 0.0
        n = 0
        for start in range(0, len(paired), batch_size):
            chunk = paired[start : start + batch_size]
            states = torch.tensor(
                [_state_features(r) for r, _ in chunk], dtype=torch.float32
            )
            option_feats, legal_mask, _ = _pad_options(
                [_option_features(r) for r, _ in chunk]
            )
            targets = []
            for row, receipt in chunk:
                n_legal = int(row.get("legal_action_count") or option_feats.size(1))
                dist = _visit_target(receipt, n_legal)
                while len(dist) < option_feats.size(1):
                    dist.append(0.0)
                targets.append(dist[: option_feats.size(1)])
            target_t = torch.tensor(targets, dtype=torch.float32)
            # Renormalize over legal masks.
            target_t = target_t * legal_mask.float()
            target_t = target_t / target_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            logits = actor(states, option_feats).masked_fill(~legal_mask, float("-inf"))
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, target_t, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(chunk)
            n += len(chunk)
        last_loss = total / max(1, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "stage_e_search_distilled_actor.pt"
    torch.save(
        {
            "schema": "poke_bot.slowking_distill.stage_e/v1",
            "research_only": RESEARCH_ONLY,
            "training_authority": False,
            "state_dict": actor.state_dict(),
            "n_paired": len(paired),
            "loss": last_loss,
            "config": StageAConfig().__dict__,
        },
        ckpt,
    )
    metrics = {"kl_loss": last_loss, "n_paired": float(len(paired))}
    (out_dir / "stage_e_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return DistillResult(metrics=metrics, checkpoint_path=str(ckpt))
