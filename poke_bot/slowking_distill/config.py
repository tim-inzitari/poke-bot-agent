"""Runtime/feature flags for Slowking distill (default: fully off)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SlowkingDistillMode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class SlowkingDistillRuntimeConfig:
    """Opt-in runtime attachment. Defaults preserve production behavior."""

    enabled: bool = False
    mode: SlowkingDistillMode = SlowkingDistillMode.DISABLED
    actor_checkpoint: Optional[str] = None
    entropy_threshold: float = 1.2
    disagreement_threshold: float = 0.25
    search_budget_sims: int = 32
    search_move_time_s: float = 2.0
    fail_closed_to_actor: bool = True
    use_belief_mcts: bool = True

    @property
    def runs(self) -> bool:
        return bool(self.enabled) and self.mode is not SlowkingDistillMode.DISABLED

    @property
    def selects_actions(self) -> bool:
        return bool(self.enabled) and self.mode is SlowkingDistillMode.ACTIVE

    @property
    def shadow_only(self) -> bool:
        return bool(self.enabled) and self.mode is SlowkingDistillMode.SHADOW

    def inventory(self) -> dict[str, Any]:
        return {
            "schema": "poke_bot.slowking_distill.runtime_config/v1",
            "enabled": self.enabled,
            "mode": self.mode.value,
            "selects_actions": self.selects_actions,
            "actor_checkpoint": self.actor_checkpoint,
            "search_budget_sims": self.search_budget_sims,
            "fail_closed_to_actor": self.fail_closed_to_actor,
            "use_belief_mcts": self.use_belief_mcts,
        }


def load_runtime_config_from_env(
    base: Optional[SlowkingDistillRuntimeConfig] = None,
) -> SlowkingDistillRuntimeConfig:
    cfg = base or SlowkingDistillRuntimeConfig()
    enabled_raw = os.environ.get("POKEBOT_SLOWKING_DISTILL_ENABLED", "").strip()
    mode_raw = os.environ.get("POKEBOT_SLOWKING_DISTILL_MODE", "").strip()
    ckpt = os.environ.get("POKEBOT_SLOWKING_DISTILL_ACTOR_CKPT", "").strip()
    enabled = cfg.enabled
    mode = cfg.mode
    if enabled_raw:
        enabled = enabled_raw.lower() in {"1", "true", "yes", "on"}
    if mode_raw:
        mode = SlowkingDistillMode(mode_raw.lower())
    return SlowkingDistillRuntimeConfig(
        enabled=enabled,
        mode=mode,
        actor_checkpoint=ckpt or cfg.actor_checkpoint,
        entropy_threshold=cfg.entropy_threshold,
        disagreement_threshold=cfg.disagreement_threshold,
        search_budget_sims=int(
            os.environ.get(
                "POKEBOT_SLOWKING_DISTILL_SEARCH_SIMS",
                str(cfg.search_budget_sims),
            )
        ),
        search_move_time_s=float(
            os.environ.get(
                "POKEBOT_SLOWKING_DISTILL_SEARCH_TIME",
                str(cfg.search_move_time_s),
            )
        ),
        fail_closed_to_actor=cfg.fail_closed_to_actor,
        use_belief_mcts=os.environ.get(
            "POKEBOT_SLOWKING_DISTILL_USE_BELIEF_MCTS", "1"
        ).strip().lower()
        not in {"0", "false", "no", "off"},
    )
