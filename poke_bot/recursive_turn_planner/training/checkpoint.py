"""Sidecar checkpoints for Recursive Turn Planner shadow training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch

from poke_bot.recursive_turn_planner.config import RTPConfig
from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner


RTP_SHADOW_TRAIN_SCHEMA = "poke_bot.recursive_turn_planner.shadow_train/v1"


def save_rtp_checkpoint(
    planner: RecursiveTurnPlanner,
    path: Path | str,
    *,
    metrics: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write an immutable-style planner sidecar (does not rewrite parent CABT)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RTP_SHADOW_TRAIN_SCHEMA,
        "research_only": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "generated_at_unix": time.time(),
        "config": {
            "sizing_profile": planner.config.sizing_profile,
            "d_model": planner.config.d_model,
            "dynamics_width": planner.config.dynamics_width,
            "num_plan_candidates": planner.config.num_plan_candidates,
            "max_recursion_depth": planner.config.max_recursion_depth,
            "max_neural_passes": planner.config.max_neural_passes,
            "max_plan_length": planner.config.max_plan_length,
            "complexity_option_threshold": planner.config.complexity_option_threshold,
            "complexity_entropy_threshold": planner.config.complexity_entropy_threshold,
            "prefer_option_hidden": planner.config.prefer_option_hidden,
            "online_sim_verify_budget": planner.config.online_sim_verify_budget,
        },
        "state_dict": {k: v.detach().cpu() for k, v in planner.state_dict().items()},
        "metrics": dict(metrics or {}),
        "extra": dict(extra or {}),
        "inventory": planner.inventory(),
    }
    torch.save(payload, out)
    receipt = {
        "schema": RTP_SHADOW_TRAIN_SCHEMA + ".receipt",
        "checkpoint_path": str(out.resolve()),
        "d_model": planner.config.d_model,
        "parameters": int(sum(p.numel() for p in planner.parameters())),
        "metrics": dict(metrics or {}),
        "serving_eligible": False,
    }
    receipt_path = out.with_suffix(out.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_rtp_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
    planner: Optional[RecursiveTurnPlanner] = None,
) -> RecursiveTurnPlanner:
    """Load planner weights into a new or existing RecursiveTurnPlanner."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_raw = dict(payload.get("config") or {})
    if planner is None:
        cfg = RTPConfig(
            sizing_profile=str(cfg_raw.get("sizing_profile") or "trained"),
            d_model=int(cfg_raw.get("d_model") or 96),
            dynamics_width=int(cfg_raw.get("dynamics_width") or 192),
            num_plan_candidates=int(cfg_raw.get("num_plan_candidates") or 6),
            max_recursion_depth=int(cfg_raw.get("max_recursion_depth") or 2),
            max_neural_passes=int(cfg_raw.get("max_neural_passes") or 4),
            max_plan_length=int(cfg_raw.get("max_plan_length") or 12),
            complexity_option_threshold=int(
                cfg_raw.get("complexity_option_threshold") or 8
            ),
            complexity_entropy_threshold=float(
                cfg_raw.get("complexity_entropy_threshold") or 1.5
            ),
            prefer_option_hidden=bool(cfg_raw.get("prefer_option_hidden", True)),
            online_sim_verify_budget=int(
                cfg_raw.get("online_sim_verify_budget") or 0
            ),
        )
        planner = RecursiveTurnPlanner(cfg)
    state = payload.get("state_dict") or payload.get("planner_state_dict")
    if state is None:
        raise ValueError(f"RTP checkpoint missing state_dict: {path}")
    planner.load_state_dict(state, strict=False)
    planner.to(torch.device(device))
    return planner
