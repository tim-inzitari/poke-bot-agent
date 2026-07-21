"""Small, state-only Pure-RL policy/value profile for high-throughput rollout."""

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
# Sequences never cross a game boundary. Prior corpus measurement showed a
# 320-decision causal suffix covers about 99.15% of games; retaining this cap
# avoids the quadratic cost of padding the rare long game toward 4,000 steps.
PURE_RL_HISTORY_MAX_CONTEXT = int(
    os.environ.get("PURE_RL_HISTORY_MAX_CONTEXT", "320")
)
# Backward-compatible import name for the already-staged warm-start tooling.
# "Full game" describes sequence ownership, not an unbounded attention span.
PURE_RL_FULL_GAME_MAX_CONTEXT = PURE_RL_HISTORY_MAX_CONTEXT


def _dense_card2vec_flag() -> bool:
    """Pure RL default ON. Opt out via ``PURE_RL_DENSE_CARD2VEC=0`` / ``DENSE_CARD2VEC=0``."""
    raw = os.environ.get("PURE_RL_DENSE_CARD2VEC")
    if raw is None:
        raw = os.environ.get("DENSE_CARD2VEC")
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")


def pure_rl_model_config(**overrides: Any) -> config.ModelConfig:
    """Lean state evaluator for high-SPS pure RL (not Hope's d=256 primary).

    The temporal-policy experiment did not improve the formal baseline gate
    and made cross-game leaf batching re-encode up to 320 prior boards. New
    runs therefore default to a state-only 4/0/4 network at ``d_model=96``
    (about 1.5M trainable parameters). Search and belief code can add
    lookahead without putting realized-history attention on every decision.

    Old history checkpoints remain loadable. Override via ``PURE_RL_*``.
    """
    def _i(name: str, default: int) -> int:
        raw = os.environ.get(f"PURE_RL_{name}")
        return int(raw) if raw is not None else default

    def _f(name: str, default: float) -> float:
        raw = os.environ.get(f"PURE_RL_{name}")
        return float(raw) if raw is not None else default

    # Dense card2vec frees ~1.5M flat EmbeddingBag params (Option A compose).
    # Spend that budget on the current board/action evaluator, not a temporal
    # tower whose compute scales with realized history length.
    # Production state default: 4/0/4 @ d96 (~1.5M).
    # Flat-bag / tiny 2/2/2@d16 / 6/6/6 checkpoints are architecture-
    # incompatible — fresh seed required when changing these defaults.
    cfg = config.ModelConfig(
        d_model=_i("D_MODEL", 96),
        spatial_layers=_i("SPATIAL_LAYERS", 4),
        temporal_layers=_i("TEMPORAL_LAYERS", 0),
        option_decoder_layers=_i("OPTION_DECODER_LAYERS", 4),
        n_heads=_i("N_HEADS", 8),
        ff_dim=_i("FF_DIM", 384),
        # Hope ladder evidence: outputs/notes/max_context.md — p99=309,
        # coverage@320=99.15% (n=4985 games); 256 only covered 97.46%.
        max_context=_i("MAX_CONTEXT", 320),
        temporal_pos=os.environ.get("PURE_RL_TEMPORAL_POS", "rope"),
        decision_context=os.environ.get(
            "PURE_RL_DECISION_CONTEXT", "stateless"
        ).strip().lower(),
        kv_cache=False,
        history_action_scale=_f("HISTORY_ACTION_SCALE", 0.1),
        card_embed_dim=_i("CARD_EMBED_DIM", 48),
        attack_embed_dim=_i("ATTACK_EMBED_DIM", 48),
        dense_card2vec=_dense_card2vec_flag(),
        dropout=_f("DROPOUT", 0.05),
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown ModelConfig field: {key}")
        setattr(cfg, key, value)
    if cfg.decision_context not in {"history", "stateless"}:
        raise ValueError(
            "PURE_RL_DECISION_CONTEXT must be 'history' or 'stateless', got "
            f"{cfg.decision_context!r}"
        )
    cfg.kv_cache = bool(cfg.decision_context == "history" and cfg.temporal_layers > 0)
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


def pure_rl_history_model_config(**overrides: Any) -> config.ModelConfig:
    """One-layer causal model over one game-bounded acting-seat sequence.

    This is an explicit experimental profile rather than a changed production
    default. It keeps the lean spatial/option trunk, adds one temporal block,
    retains the newest 320 decisions within that game, and enables the
    incremental KV cache used during play. Training must use game-bounded
    :class:`GameSequence` batches; the stateless GPU-resident flattened-corpus
    path is intentionally incompatible. Longer games drop only their oldest
    prefix; games are never concatenated.
    """
    history = {
        "temporal_layers": 1,
        "decision_context": "history",
        "kv_cache": True,
        "max_context": PURE_RL_HISTORY_MAX_CONTEXT,
    }
    history.update(overrides)
    return pure_rl_model_config(**history)


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
        f"ff={cfg.ff_dim} ctx={cfg.max_context} "
        f"dense_card2vec={bool(cfg.dense_card2vec)}",
        flush=True,
    )
    if validate:
        validate_param_budget(n)
    return model


def model_config_dict(cfg: Optional[config.ModelConfig] = None) -> dict[str, Any]:
    cfg = cfg or pure_rl_model_config()
    return asdict(cfg)
