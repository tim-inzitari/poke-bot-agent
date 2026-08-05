"""Shadow training loop for PokeRLMModelCore (multi-turn planner attachment)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from poke_bot.poke_rlm.config import PokeRLMConfig, config_for_profile
from poke_bot.poke_rlm.model_core import PokeRLMModelCore
from poke_bot.poke_rlm.router import Route, choose_route
from poke_bot.poke_rlm.training.labels import PlanSupervisionLabels
from poke_bot.poke_rlm.training.losses import compute_poke_rlm_losses
from poke_bot.recursive_turn_planner.training.shadow_train import (
    RTPDecisionBatch,
    heuristic_should_recurse,
)


POKE_RLM_SHADOW_SCHEMA = "poke_bot.poke_rlm.shadow_train/v1"


@dataclass
class PokeRLMTrainConfig:
    profile: str = "pure_rl_96"
    d_model: int = 96
    epochs: int = 2
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    action_weight: float = 1.0
    route_weight: float = 0.25
    recurse_weight: float = 0.25
    dynamics_weight: float = 0.5


@dataclass
class PokeRLMTrainResult:
    checkpoint_path: str
    receipt_path: str
    metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    inventory: dict[str, int] = field(default_factory=dict)


def _labels_from_batch(batch: RTPDecisionBatch, cfg: PokeRLMConfig) -> PlanSupervisionLabels:
    fake_logits = batch.option_hidden @ batch.state
    decision = choose_route(
        cfg,
        n_legal=len(batch.legal_actions),
        policy_logits=fake_logits,
    )
    return PlanSupervisionLabels(
        chosen_action_index=int(batch.chosen_index),
        route_target=decision.route.value,
        should_recurse=decision.route is Route.RECURSIVE
        or heuristic_should_recurse(
            len(batch.legal_actions),
            fake_logits,
            option_threshold=cfg.complexity_option_threshold,
            entropy_threshold=cfg.complexity_entropy_threshold,
        ),
        stop_reason="shadow_train",
    )


def train_poke_rlm_shadow(
    batches: Sequence[RTPDecisionBatch],
    *,
    output_dir: Path | str,
    config: Optional[PokeRLMTrainConfig] = None,
    core: Optional[PokeRLMModelCore] = None,
) -> PokeRLMTrainResult:
    cfg = config or PokeRLMTrainConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.seed))

    if core is None:
        try:
            poke_cfg = config_for_profile(cfg.profile, d_model=int(cfg.d_model))
        except Exception:
            poke_cfg = PokeRLMConfig(
                d_model=int(cfg.d_model),
                root_plan_candidates=4 if int(cfg.d_model) == 96 else 6,
                successor_dim=64,
            )
        core = PokeRLMModelCore(poke_cfg)
    else:
        poke_cfg = core.config

    device = torch.device(cfg.device)
    core.to(device)
    core.train()
    opt = torch.optim.AdamW(core.parameters(), lr=float(cfg.lr))

    history: list[dict[str, float]] = []
    last: dict[str, float] = {}
    for epoch in range(int(cfg.epochs)):
        total = 0.0
        n = 0
        for batch in batches:
            opt.zero_grad(set_to_none=True)
            state = batch.state.to(device).unsqueeze(0)
            opts = batch.option_hidden.to(device).unsqueeze(0)
            n_legal = opts.size(1)
            mask = torch.ones(1, n_legal, dtype=torch.bool, device=device)
            heads = core.score_actions(state, opts, legal_mask=mask)
            labels = _labels_from_batch(batch, poke_cfg)
            route_logits = core.route_logits(state)
            recurse_logits = core.recurse_logits(state)
            chosen = max(0, min(int(batch.chosen_index), n_legal - 1))
            action_embed = opts[0, chosen]
            dyn = core.dynamics(state.squeeze(0), action_embed)
            target = batch.next_state.to(device) if batch.next_state is not None else None
            bundle = compute_poke_rlm_losses(
                action_logits=heads.policy_logits[0],
                route_logits=route_logits[0],
                recurse_logits=recurse_logits,
                labels=labels,
                predicted_next_latent=dyn["next_latent"] if target is not None else None,
                target_next_latent=target.unsqueeze(0) if target is not None else None,
                action_weight=cfg.action_weight,
                route_weight=cfg.route_weight,
                recurse_weight=cfg.recurse_weight,
                dynamics_weight=cfg.dynamics_weight,
            )
            if not bool(torch.isfinite(bundle.total)):
                raise RuntimeError("PokeRLM shadow loss is non-finite")
            bundle.total.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            row = {"epoch": float(epoch), **bundle.as_dict()}
            history.append(row)
            total += row["total"]
            n += 1
        last = {"epoch": float(epoch), "mean_loss": total / max(1, n), "n_steps": float(n)}

    ckpt = out / "poke_rlm_shadow_core.pt"
    payload = {
        "schema": POKE_RLM_SHADOW_SCHEMA,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "generated_at_unix": time.time(),
        "config": {
            "d_model": int(poke_cfg.d_model),
            "profile": getattr(poke_cfg.profile, "value", str(poke_cfg.profile)),
            "root_plan_candidates": int(poke_cfg.root_plan_candidates),
            "max_depth": int(poke_cfg.max_depth),
            "q_quantiles": int(poke_cfg.q_quantiles),
            "value_horizons": int(poke_cfg.value_horizons),
            "successor_dim": int(poke_cfg.successor_dim),
        },
        "state_dict": {k: v.detach().cpu() for k, v in core.state_dict().items()},
        "metrics": last,
        "inventory": core.parameter_inventory(),
        "train_config": asdict(cfg),
    }
    torch.save(payload, ckpt)
    receipt = {
        "schema": POKE_RLM_SHADOW_SCHEMA + ".receipt",
        "checkpoint_path": str(ckpt.resolve()),
        "metrics": last,
        "serving_eligible": False,
        "inventory": core.parameter_inventory(),
    }
    receipt_path = ckpt.with_suffix(ckpt.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return PokeRLMTrainResult(
        checkpoint_path=str(ckpt.resolve()),
        receipt_path=str(receipt_path.resolve()),
        metrics=last,
        history=history,
        inventory=core.parameter_inventory(),
    )


def load_poke_rlm_core(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> PokeRLMModelCore:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_raw = dict(payload.get("config") or {})
    poke_cfg = PokeRLMConfig(
        d_model=int(cfg_raw.get("d_model") or 96),
        root_plan_candidates=int(cfg_raw.get("root_plan_candidates") or 6),
        q_quantiles=int(cfg_raw.get("q_quantiles") or 8),
        value_horizons=int(cfg_raw.get("value_horizons") or 3),
        successor_dim=int(cfg_raw.get("successor_dim") or 64),
        max_depth=int(cfg_raw.get("max_depth") or 2),
    )
    core = PokeRLMModelCore(poke_cfg)
    core.load_state_dict(payload["state_dict"], strict=True)
    core.to(torch.device(device))
    return core
