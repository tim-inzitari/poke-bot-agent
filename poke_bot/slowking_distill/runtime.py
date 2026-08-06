"""Gated Slowking distill runtime — fast actor + critical-node search.

Default-off. Never injects heuristic scores into serving logits. Fail-closed to
the fast actor when search is unavailable, errors, or exceeds budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .authority import RESEARCH_ONLY, RUNTIME_AUTHORITY
from .bc_stage import (
    OptionConditionedClone,
    _option_features,
    _pad_options,
    _state_features,
)
from .belief_search_backend import BeliefSearchBundle, resolve_stage_c_search_fn
from .config import (
    SlowkingDistillMode,
    SlowkingDistillRuntimeConfig,
    load_runtime_config_from_env,
)
from .critical_search import SearchFn, is_critical_decision
from .heuristic_features import attach_heuristic_features


@dataclass
class DistillRuntimeDecision:
    action: list[int]
    source: str  # actor | search | fail_closed_actor | disabled | shadow
    critical: bool
    critical_reason: str
    search_used: bool
    search_backend: str
    latency_ms: float
    budget_exceeded: bool = False
    reason: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)


def load_actor_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str | None = None,
) -> tuple[OptionConditionedClone, dict[str, Any]]:
    """Load a Stage A/B/E actor checkpoint."""
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    cfg = payload.get("config") or {}
    d_model = int(cfg.get("d_model") or 0)
    # Stage B/E may store under "actor" or "state_dict"
    state = payload.get("state_dict") or payload.get("actor")
    if state is None:
        raise ValueError(f"checkpoint missing state_dict/actor: {ckpt_path}")
    if d_model <= 0:
        # Infer from first Linear weight in state_enc.0.weight -> [d_model, 16]
        w = state.get("state_enc.0.weight")
        d_model = int(w.shape[0]) if w is not None else 64
    model = OptionConditionedClone(d_model)
    model.load_state_dict(state)
    model.to(torch.device(device or "cpu"))
    model.eval()
    return model, payload


class SlowkingDistillRuntime:
    """Opt-in runtime policy using Stage E actor + Stage C search at critical nodes."""

    def __init__(
        self,
        *,
        config: SlowkingDistillRuntimeConfig | None = None,
        actor: OptionConditionedClone | None = None,
        actor_checkpoint: Path | str | None = None,
        search_bundle: BeliefSearchBundle | None = None,
        search_fn: SearchFn | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or load_runtime_config_from_env()
        self.device = torch.device(device or "cpu")
        self._model = actor
        self._ckpt_meta: dict[str, Any] = {}
        ckpt = actor_checkpoint or self.config.actor_checkpoint
        if self._model is None and ckpt:
            self._model, self._ckpt_meta = load_actor_checkpoint(ckpt, device=self.device)
        self.search_bundle = search_bundle
        self._search_fn = search_fn
        self._search_backend = "unset"
        self._decisions = 0
        self._search_calls = 0

    @property
    def ready(self) -> bool:
        return self._model is not None and self.config.runs

    def _resolve_search(self) -> tuple[SearchFn, str]:
        if self._search_fn is not None:
            return self._search_fn, self._search_backend or "custom"
        fn, backend = resolve_stage_c_search_fn(self.search_bundle)
        self._search_fn = fn
        self._search_backend = backend
        return fn, backend

    def actor_select(self, row: Mapping[str, Any]) -> list[int]:
        """Argmax over legal option embeddings. Heuristic scores are features only."""
        prepared = attach_heuristic_features(dict(row))
        legal = [list(c) for c in (prepared.get("legal_action_combos") or [])]
        if not legal:
            return list(prepared.get("action") or [])
        if self._model is None:
            return list(legal[0])
        state = torch.tensor([_state_features(prepared)], dtype=torch.float32, device=self.device)
        opts, mask, _ = _pad_options([_option_features(prepared)])
        opts = opts.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():
            logits = self._model(state, opts).masked_fill(~mask, float("-inf"))
            idx = int(torch.argmax(logits, dim=-1).item())
        if 0 <= idx < len(legal):
            return list(legal[idx])
        return list(legal[0])

    def decide_row(
        self,
        row: Mapping[str, Any],
        *,
        policy_logits: Sequence[float] | None = None,
        force_search: bool = False,
    ) -> DistillRuntimeDecision:
        t0 = time.perf_counter()
        prepared = attach_heuristic_features(dict(row))
        legal = [list(c) for c in (prepared.get("legal_action_combos") or [])]
        if not legal:
            legal = [list(prepared.get("action") or [])]

        if not self.config.runs:
            return DistillRuntimeDecision(
                action=list(legal[0]) if legal else [],
                source="disabled",
                critical=False,
                critical_reason="runtime_disabled",
                search_used=False,
                search_backend="none",
                latency_ms=0.0,
                reason="runtime_disabled",
                telemetry={"research_only": RESEARCH_ONLY, "runtime_authority": RUNTIME_AUTHORITY},
            )

        critical, crit_reason = is_critical_decision(
            prepared,
            policy_logits=policy_logits,
            entropy_threshold=self.config.entropy_threshold,
            disagreement_threshold=self.config.disagreement_threshold,
        )
        if force_search:
            critical, crit_reason = True, "forced"

        actor_action = self.actor_select(prepared)
        action = list(actor_action)
        source = "actor"
        reason = "actor_default"
        search_used = False
        search_backend = "none"
        budget_exceeded = False

        use_search = critical and self.config.use_belief_mcts
        if use_search:
            try:
                search_fn, search_backend = self._resolve_search()
                # Cap sims via bundle if present
                if self.search_bundle is not None:
                    self.search_bundle.max_sims = int(self.config.search_budget_sims)
                    self.search_bundle.move_time_s = float(self.config.search_move_time_s)
                result = search_fn(
                    prepared.get("observation") or {},
                    prepared.get("legal_action_combos") or legal,
                )
                search_used = True
                self._search_calls += 1
                chosen = list(result.get("chosen_action") or [])
                search_backend = str(result.get("search_backend") or search_backend)
                if chosen and chosen in legal:
                    action = chosen
                    source = "search"
                    reason = crit_reason
                elif self.config.fail_closed_to_actor:
                    action = list(actor_action)
                    source = "fail_closed_actor"
                    reason = "search_action_illegal_or_empty"
                else:
                    action = chosen or list(actor_action)
                    source = "search"
                    reason = "search_unvalidated"
            except Exception as exc:  # noqa: BLE001 — fail closed
                if self.config.fail_closed_to_actor:
                    action = list(actor_action)
                    source = "fail_closed_actor"
                    reason = f"search_error:{type(exc).__name__}"
                else:
                    raise

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._decisions += 1
        decision = DistillRuntimeDecision(
            action=action,
            source=source,
            critical=critical,
            critical_reason=crit_reason,
            search_used=search_used,
            search_backend=search_backend,
            latency_ms=latency_ms,
            budget_exceeded=budget_exceeded,
            reason=reason,
            telemetry={
                "actor_action": list(actor_action),
                "mode": self.config.mode.value,
                "selects_actions": self.config.selects_actions,
                "shadow_only": self.config.shadow_only,
                "decisions": self._decisions,
                "search_calls": self._search_calls,
                "research_only": RESEARCH_ONLY,
                "runtime_authority": RUNTIME_AUTHORITY,
            },
        )
        if self.config.shadow_only:
            decision.telemetry["shadow"] = True
            decision.telemetry["would_action"] = list(action)
        return decision

    def select_action(self, obs: Mapping[str, Any], legal: Sequence[Sequence[int]]) -> list[int]:
        """Convenience for PolicyAgent-shaped inputs."""
        row = {
            "observation": dict(obs),
            "legal_action_combos": [list(a) for a in legal],
            "legal_action_count": len(legal),
            "action": list(legal[0]) if legal else [],
            "selected_index": 0,
            "game_id": "runtime",
            "env_step": -1,
        }
        decision = self.decide_row(row)
        if self.config.shadow_only or not self.config.selects_actions:
            # Shadow / disabled callers get actor suggestion in telemetry only.
            return list(legal[0]) if legal else []
        return list(decision.action)


def build_runtime_from_env(
    *,
    search_bundle: BeliefSearchBundle | None = None,
    device: str | torch.device | None = None,
) -> SlowkingDistillRuntime | None:
    """Return a runtime only when env enables the research lane."""
    cfg = load_runtime_config_from_env()
    if not cfg.runs:
        return None
    if not cfg.actor_checkpoint:
        return None
    path = Path(cfg.actor_checkpoint)
    if not path.is_file():
        return None
    return SlowkingDistillRuntime(
        config=cfg,
        actor_checkpoint=path,
        search_bundle=search_bundle,
        device=device,
    )


__all__ = [
    "DistillRuntimeDecision",
    "SlowkingDistillRuntime",
    "build_runtime_from_env",
    "load_actor_checkpoint",
]
