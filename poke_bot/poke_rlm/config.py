"""PokeRLM configuration. Defaults preserve current policy behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class PokeRLMMode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    EVALUATE = "evaluate"
    ACTIVE = "active"


class PokeRLMProfile(str, Enum):
    PILOT_256 = "pilot_256"
    BASE_384 = "base_384"
    STRONG_512 = "strong_512"
    #: Attach to existing Pure-RL d=96 encoder.
    PURE_RL_96 = "pure_rl_96"
    #: Attach to global d=256 encoder.
    GLOBAL_256 = "global_256"


@dataclass(frozen=True)
class PokeRLMConfig:
    """Feature-flagged PokeRLM runtime/training config.

    ``enabled=False`` / ``mode=disabled`` keeps PolicyAgent on current greedy
    (or explicit MCTS) behavior with zero planner side effects.
    """

    enabled: bool = False
    mode: PokeRLMMode = PokeRLMMode.DISABLED
    profile: PokeRLMProfile = PokeRLMProfile.PURE_RL_96

    d_model: int = 96
    n_heads: int = 8
    ffn_multiplier: int = 4
    planner_layers: int = 2
    dynamics_blocks: int = 2
    dynamics_hidden_multiplier: int = 2
    plan_vocab_size: int = 256
    max_plan_tokens_per_candidate: int = 24
    root_plan_candidates: int = 6
    q_quantiles: int = 8
    value_horizons: int = 3
    successor_dim: int = 64

    max_depth: int = 2
    audit_max_depth: int = 3
    max_subgoals: int = 8
    max_plan_nodes: int = 32
    max_neural_planner_calls_per_turn: int = 4
    max_simulator_calls_per_turn: int = 16
    legacy_hard_cap_simulator_calls: int = 75
    finalist_verification_calls: int = 8
    repair_reserve_calls: int = 4
    repair_budget: int = 1

    complexity_option_threshold: int = 8
    complexity_entropy_threshold: float = 1.5
    compute_cost_weight: float = 0.01
    skip_trivial_decisions: bool = True
    share_planner_weights_across_depth: bool = True
    encode_once_per_turn_when_safe: bool = True
    deterministic_fallback: str = "current_policy"
    validate_schema: bool = True
    validate_visibility: bool = True
    resolve_current_option_index_before_step: bool = True

    # Training additions never mutate RL protocol schedule numbers.
    plan_imitation_enabled: bool = True
    dynamics_loss_enabled: bool = True
    invalid_plan_penalty_enabled: bool = True
    compute_cost_penalty_enabled: bool = True
    privileged_targets_separate: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.root_plan_candidates < 1:
            raise ValueError("root_plan_candidates must be positive")
        if not 0 <= self.max_depth <= 3:
            raise ValueError("max_depth must be in [0, 3]")
        if self.max_neural_planner_calls_per_turn < 1:
            raise ValueError("max_neural_planner_calls_per_turn must be positive")
        if self.max_simulator_calls_per_turn < 0:
            raise ValueError("max_simulator_calls_per_turn cannot be negative")
        if self.max_simulator_calls_per_turn > self.legacy_hard_cap_simulator_calls:
            raise ValueError("simulator budget exceeds legacy hard cap")

    @property
    def selects_actions(self) -> bool:
        return bool(self.enabled) and self.mode in {
            PokeRLMMode.EVALUATE,
            PokeRLMMode.ACTIVE,
        }

    @property
    def shadow_only(self) -> bool:
        return bool(self.enabled) and self.mode is PokeRLMMode.SHADOW

    @property
    def runs_planner(self) -> bool:
        return bool(self.enabled) and self.mode is not PokeRLMMode.DISABLED

    def inventory(self) -> dict[str, Any]:
        return {
            "schema": "poke_bot.poke_rlm.config/v1",
            "enabled": self.enabled,
            "mode": self.mode.value,
            "profile": self.profile.value,
            "d_model": self.d_model,
            "root_plan_candidates": self.root_plan_candidates,
            "max_depth": self.max_depth,
            "max_neural_planner_calls_per_turn": self.max_neural_planner_calls_per_turn,
            "max_simulator_calls_per_turn": self.max_simulator_calls_per_turn,
            "selects_actions": self.selects_actions,
        }


_PROFILE_DIMS: dict[PokeRLMProfile, dict[str, int]] = {
    PokeRLMProfile.PILOT_256: {"d_model": 256, "successor_dim": 64, "root_plan_candidates": 4},
    PokeRLMProfile.BASE_384: {"d_model": 384, "successor_dim": 128, "root_plan_candidates": 6},
    PokeRLMProfile.STRONG_512: {"d_model": 512, "successor_dim": 128, "root_plan_candidates": 8},
    PokeRLMProfile.PURE_RL_96: {"d_model": 96, "successor_dim": 64, "root_plan_candidates": 4},
    PokeRLMProfile.GLOBAL_256: {"d_model": 256, "successor_dim": 64, "root_plan_candidates": 4},
}


def config_for_profile(profile: PokeRLMProfile | str, **overrides: Any) -> PokeRLMConfig:
    prof = PokeRLMProfile(profile)
    dims = dict(_PROFILE_DIMS[prof])
    dims.update(overrides)
    return PokeRLMConfig(profile=prof, **dims)


def load_poke_rlm_config_from_env(
    base: Optional[PokeRLMConfig] = None,
) -> PokeRLMConfig:
    """Resolve runtime flags from environment without mutating protocol YAML."""
    cfg = base or PokeRLMConfig()
    enabled_raw = os.environ.get("POKEBOT_POKE_RLM_ENABLED", "").strip()
    mode_raw = os.environ.get("POKEBOT_POKE_RLM_MODE", "").strip()
    profile_raw = os.environ.get("POKEBOT_POKE_RLM_PROFILE", "").strip()
    enabled = cfg.enabled
    mode = cfg.mode
    profile = cfg.profile
    if enabled_raw:
        enabled = enabled_raw.lower() in {"1", "true", "yes", "on"}
    if mode_raw:
        mode = PokeRLMMode(mode_raw.lower())
    if profile_raw:
        profile = PokeRLMProfile(profile_raw.lower())
    dims = dict(_PROFILE_DIMS[profile])
    return PokeRLMConfig(
        enabled=enabled,
        mode=mode,
        profile=profile,
        d_model=int(dims["d_model"]),
        successor_dim=int(dims["successor_dim"]),
        root_plan_candidates=int(dims["root_plan_candidates"]),
        max_depth=cfg.max_depth,
        max_neural_planner_calls_per_turn=cfg.max_neural_planner_calls_per_turn,
        max_simulator_calls_per_turn=cfg.max_simulator_calls_per_turn,
        complexity_option_threshold=cfg.complexity_option_threshold,
        compute_cost_weight=cfg.compute_cost_weight,
    )


def maybe_load_yaml_overlay(path: str | Path) -> dict[str, Any]:
    """Optional YAML overlay; missing file returns empty dict."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return dict(payload.get("poke_rlm") or payload)
