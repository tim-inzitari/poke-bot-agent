"""PolicyAgent bridge for Slowking distill runtime (default-off).

Dispatch order when active: MCTS → PokeRLM active → Slowking distill active →
RTP → greedy. Shadow mode records telemetry without changing the selected action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .belief_search_backend import BeliefSearchBundle
from .config import SlowkingDistillRuntimeConfig, load_runtime_config_from_env
from .runtime import DistillRuntimeDecision, SlowkingDistillRuntime, build_runtime_from_env


@dataclass
class SlowkingDistillBridgeDiagnostics:
    mode: str = "disabled"
    source: str = ""
    critical: bool = False
    search_used: bool = False
    search_backend: str = ""
    reason: str = ""
    legal_count: int = 0
    latency_ms: float = 0.0
    fallback_reason: str = ""
    trace: dict[str, Any] = field(default_factory=dict)


class SlowkingDistillAgentBridge:
    """Thin adapter between PolicyAgent observations and SlowkingDistillRuntime."""

    def __init__(
        self,
        *,
        runtime: SlowkingDistillRuntime | None = None,
        config: SlowkingDistillRuntimeConfig | None = None,
        search_bundle: BeliefSearchBundle | None = None,
        device: Any = None,
    ) -> None:
        self.config = config or load_runtime_config_from_env()
        self.runtime = runtime
        if self.runtime is None and self.config.runs and self.config.actor_checkpoint:
            self.runtime = build_runtime_from_env(
                search_bundle=search_bundle,
                device=device,
            )
        elif self.runtime is not None and search_bundle is not None:
            self.runtime.search_bundle = search_bundle
        self.last_diagnostics = SlowkingDistillBridgeDiagnostics(
            mode=self.config.mode.value
        )
        self.last_shadow_decision: DistillRuntimeDecision | None = None

    @property
    def ready(self) -> bool:
        return self.runtime is not None and self.runtime.ready

    def _row_from_obs(
        self,
        obs: dict[str, Any],
        legal: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        return {
            "observation": obs,
            "legal_action_combos": [list(a) for a in legal],
            "legal_action_count": len(legal),
            "action": list(legal[0]) if legal else [],
            "selected_index": 0,
            "game_id": "policy_agent",
            "env_step": -1,
            "heuristic_abstained": True,
        }

    def select(
        self,
        obs: dict[str, Any],
        legal: Sequence[Sequence[int]],
        *,
        policy_logits: Sequence[float] | None = None,
    ) -> list[int] | None:
        """Return an action when active+ready; else None (caller continues)."""
        if self.runtime is None or not self.config.selects_actions:
            self.last_diagnostics = SlowkingDistillBridgeDiagnostics(
                mode=self.config.mode.value,
                reason="not_active",
                legal_count=len(legal),
                fallback_reason="not_active",
            )
            return None
        row = self._row_from_obs(obs, legal)
        decision = self.runtime.decide_row(row, policy_logits=policy_logits)
        self.last_diagnostics = SlowkingDistillBridgeDiagnostics(
            mode=self.config.mode.value,
            source=decision.source,
            critical=decision.critical,
            search_used=decision.search_used,
            search_backend=decision.search_backend,
            reason=decision.reason,
            legal_count=len(legal),
            latency_ms=decision.latency_ms,
            trace={
                "critical_reason": decision.critical_reason,
                "telemetry": decision.telemetry,
                "action": list(decision.action),
            },
        )
        action = list(decision.action)
        legal_lists = [list(a) for a in legal]
        if action not in legal_lists:
            self.last_diagnostics.fallback_reason = "illegal_action"
            return None
        return action

    def shadow(
        self,
        obs: dict[str, Any],
        legal: Sequence[Sequence[int]],
        *,
        committed_action: Sequence[int] | None = None,
        policy_logits: Sequence[float] | None = None,
    ) -> DistillRuntimeDecision | None:
        """Compute a shadow decision for telemetry; never changes gameplay."""
        if self.runtime is None or not self.config.shadow_only:
            return None
        row = self._row_from_obs(obs, legal)
        decision = self.runtime.decide_row(row, policy_logits=policy_logits)
        self.last_shadow_decision = decision
        self.last_diagnostics = SlowkingDistillBridgeDiagnostics(
            mode=self.config.mode.value,
            source=decision.source,
            critical=decision.critical,
            search_used=decision.search_used,
            search_backend=decision.search_backend,
            reason=decision.reason,
            legal_count=len(legal),
            latency_ms=decision.latency_ms,
            trace={
                "shadow": True,
                "would_action": list(decision.action),
                "committed_action": list(committed_action or []),
                "critical_reason": decision.critical_reason,
                "telemetry": decision.telemetry,
            },
        )
        return decision


__all__ = [
    "SlowkingDistillAgentBridge",
    "SlowkingDistillBridgeDiagnostics",
]
