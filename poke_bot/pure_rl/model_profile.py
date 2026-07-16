"""Small fast Pure-RL model profile (~1–3M params, Kaggle SPS class)."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Optional

import torch

from poke_bot import config
from poke_bot.model import TemporalCabtTransformer, build_model

#: Fail closed above this — leaves headroom over the 3M product target.
PURE_RL_PARAM_FAIL_MAX = int(os.environ.get("PURE_RL_PARAM_FAIL_MAX", "3500000"))
PURE_RL_PARAM_TARGET_MAX = int(os.environ.get("PURE_RL_PARAM_TARGET_MAX", "3000000"))


def pure_rl_model_config(**overrides: Any) -> config.ModelConfig:
    """Lean history policy for high-SPS pure RL (not Hope's d=256 primary).

    Default lands ~2.4M params with the production encoder vocab (~22k).
    Override via ``PURE_RL_D_MODEL`` / layer env knobs or ``overrides``.
    """
    def _i(name: str, default: int) -> int:
        raw = os.environ.get(f"PURE_RL_{name}")
        return int(raw) if raw is not None else default

    def _f(name: str, default: float) -> float:
        raw = os.environ.get(f"PURE_RL_{name}")
        return float(raw) if raw is not None else default

    # Prefer ~1.6M (Abhyuday <2M / high-SPS class); override via PURE_RL_D_MODEL.
    cfg = config.ModelConfig(
        d_model=_i("D_MODEL", 16),
        spatial_layers=_i("SPATIAL_LAYERS", 1),
        temporal_layers=_i("TEMPORAL_LAYERS", 1),
        option_decoder_layers=_i("OPTION_DECODER_LAYERS", 1),
        n_heads=_i("N_HEADS", 4),
        ff_dim=_i("FF_DIM", 32),
        max_context=_i("MAX_CONTEXT", 32),
        temporal_pos=os.environ.get("PURE_RL_TEMPORAL_POS", "rope"),
        decision_context="history",
        kv_cache=True,
        history_action_scale=_f("HISTORY_ACTION_SCALE", 0.1),
        card_embed_dim=_i("CARD_EMBED_DIM", 16),
        attack_embed_dim=_i("ATTACK_EMBED_DIM", 16),
        dropout=_f("DROPOUT", 0.05),
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown ModelConfig field: {key}")
        setattr(cfg, key, value)
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(
            f"pure_rl d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}"
        )
    if (cfg.d_model // cfg.n_heads) % 2 != 0:
        raise ValueError(
            f"pure_rl RoPE head dim must be even; got "
            f"{cfg.d_model // cfg.n_heads}"
        )
    return cfg


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def validate_param_budget(
    n_params: int,
    *,
    fail_max: int = PURE_RL_PARAM_FAIL_MAX,
) -> None:
    if n_params > int(fail_max):
        raise SystemExit(
            f"PURE_RL fail-closed: model has {n_params:,} params "
            f"> {int(fail_max):,} (target ≤{PURE_RL_PARAM_TARGET_MAX:,}). "
            "Use poke_bot.pure_rl.model_profile.pure_rl_model_config()."
        )


def build_pure_rl_model(
    *,
    device: Optional[torch.device] = None,
    cfg: Optional[config.ModelConfig] = None,
    validate: bool = True,
    **build_kwargs: Any,
) -> TemporalCabtTransformer:
    cfg = cfg or pure_rl_model_config()
    model = build_model(cfg, device=device or torch.device("cpu"), **build_kwargs)
    n = count_params(model)
    print(
        f"[pure_rl] model_params={n} ({n / 1e6:.3f}M) "
        f"d_model={cfg.d_model} L={cfg.spatial_layers}/"
        f"{cfg.temporal_layers}/{cfg.option_decoder_layers} "
        f"ff={cfg.ff_dim} ctx={cfg.max_context}",
        flush=True,
    )
    if validate:
        validate_param_budget(n)
    return model


def model_config_dict(cfg: Optional[config.ModelConfig] = None) -> dict[str, Any]:
    cfg = cfg or pure_rl_model_config()
    return asdict(cfg)
