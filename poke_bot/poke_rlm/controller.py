"""Top-level PokeRLM controller for shadow/active planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from .budget import TurnComputeBudget
from .config import PokeRLMConfig
from .executor import PlanExecutor
from .legal_action import LegalAction, legal_actions_from_select
from .model_core import PokeRLMModelCore
from .observation import DeploymentObservation, build_deployment_observation
from .plan_ir import TurnPlan
from .reasons import ReasonCode
from .recursion import refine_plan, shared_recursive_cell_identity
from .router import Route, choose_route
from .telemetry import ShadowTrace, budget_to_trace


@dataclass
class ControllerResult:
    action: tuple[int, ...]
    reason: ReasonCode
    route: str
    used_for_selection: bool
    plan: Optional[TurnPlan]
    trace: ShadowTrace
    policy_logits: Optional[Tensor] = None


class PokeRLMController:
    """Encode-once planner attachment driven by an external state/options encode."""

    def __init__(
        self,
        config: Optional[PokeRLMConfig] = None,
        *,
        core: Optional[PokeRLMModelCore] = None,
    ) -> None:
        self.config = config or PokeRLMConfig()
        self.core = core or PokeRLMModelCore(self.config)
        self.budget = TurnComputeBudget(
            max_depth=self.config.max_depth,
            max_nodes=self.config.max_plan_nodes,
            max_subgoals=self.config.max_subgoals,
            max_model_calls=self.config.max_neural_planner_calls_per_turn,
            max_simulator_calls=self.config.max_simulator_calls_per_turn,
            legacy_hard_cap_simulator_calls=self.config.legacy_hard_cap_simulator_calls,
            repair_reserve_calls=self.config.repair_reserve_calls,
        )
        self.executor = PlanExecutor(
            budget=self.budget,
            repair_fn=self._repair,
            repair_budget=self.config.repair_budget,
        )
        self.active_turn_key: tuple[int, int] = (-1, -1)
        self.last_trace = ShadowTrace()
        self._state_vec: Optional[Tensor] = None
        self._option_hidden: Optional[Tensor] = None

    def reset_game(self) -> None:
        self.budget = TurnComputeBudget(
            max_depth=self.config.max_depth,
            max_nodes=self.config.max_plan_nodes,
            max_subgoals=self.config.max_subgoals,
            max_model_calls=self.config.max_neural_planner_calls_per_turn,
            max_simulator_calls=self.config.max_simulator_calls_per_turn,
            legacy_hard_cap_simulator_calls=self.config.legacy_hard_cap_simulator_calls,
            repair_reserve_calls=self.config.repair_reserve_calls,
        )
        self.executor = PlanExecutor(
            budget=self.budget,
            repair_fn=self._repair,
            repair_budget=self.config.repair_budget,
        )
        self.active_turn_key = (-1, -1)
        self.last_trace = ShadowTrace()
        self._state_vec = None
        self._option_hidden = None

    def _repair(self, plan: TurnPlan, legal: tuple[LegalAction, ...]) -> TurnPlan:
        if not legal or self._state_vec is None or self._option_hidden is None:
            return plan
        heads = self.core.score_actions(
            self._state_vec,
            self._option_hidden.unsqueeze(0)
            if self._option_hidden.dim() == 2
            else self._option_hidden,
        )
        candidates = self.core.propose_root_plans(
            self._state_vec,
            legal,
            heads,
            observation_hash=plan.provenance.observation_hash,
            encoder_profile=plan.provenance.encoder_profile,
        )
        return candidates[0] if candidates else plan

    def shared_cell_id(self) -> int:
        return shared_recursive_cell_identity(self.core)

    def plan_decision(
        self,
        obs_dict: dict[str, Any],
        *,
        state_vec: Tensor,
        option_hidden: Tensor,
        legal_combos: Optional[Sequence[Sequence[int]]] = None,
        deck: Optional[list[int]] = None,
        matchup_route: int = -1,
        fallback_action: Optional[Sequence[int]] = None,
        observations: Optional[dict[str, Any]] = None,
    ) -> ControllerResult:
        """Run router/planner. Selection follows config.mode."""
        if not self.config.runs_planner:
            action = tuple(int(x) for x in (fallback_action or ()))
            trace = ShadowTrace(
                enabled=False,
                mode=self.config.mode.value,
                reason=ReasonCode.DISABLED,
                selected_action=action,
            )
            self.last_trace = trace
            return ControllerResult(
                action=action,
                reason=ReasonCode.DISABLED,
                route="disabled",
                used_for_selection=False,
                plan=None,
                trace=trace,
            )

        dep, vis_reason = build_deployment_observation(
            obs_dict, deck=deck, matchup_route=matchup_route
        )
        if dep is None:
            action = tuple(int(x) for x in (fallback_action or ()))
            trace = ShadowTrace(
                enabled=True,
                mode=self.config.mode.value,
                reason=vis_reason,
                selected_action=action,
            )
            self.last_trace = trace
            return ControllerResult(
                action=action,
                reason=vis_reason,
                route="fallback",
                used_for_selection=False,
                plan=None,
                trace=trace,
            )

        legal = legal_actions_from_select(obs_dict, complete_combos=legal_combos)
        if option_hidden.dim() == 2:
            option_hidden_b = option_hidden.unsqueeze(0)
        else:
            option_hidden_b = option_hidden
        if state_vec.dim() == 1:
            state_b = state_vec.unsqueeze(0)
        else:
            state_b = state_vec
        if option_hidden_b.size(1) != len(legal) and legal:
            # Align by truncation/pad with zeros for safety.
            n = len(legal)
            d = option_hidden_b.size(-1)
            aligned = option_hidden_b.new_zeros(1, n, d)
            m = min(n, option_hidden_b.size(1))
            aligned[:, :m] = option_hidden_b[:, :m]
            option_hidden_b = aligned

        self._state_vec = state_b
        self._option_hidden = option_hidden_b[0]
        reason = self.budget.consume_model_call(1)
        if reason is not ReasonCode.OK:
            action = tuple(int(x) for x in (fallback_action or (legal[0].option_index_path if legal else ())))
            trace = ShadowTrace(
                enabled=True,
                mode=self.config.mode.value,
                turn_key=dep.turn_key(),
                reason=reason,
                selected_action=action,
                observation_hash=dep.observation_hash,
                budget=budget_to_trace(self.budget),
            )
            self.last_trace = trace
            return ControllerResult(
                action=action,
                reason=reason,
                route="budget",
                used_for_selection=False,
                plan=None,
                trace=trace,
            )

        mask = torch.ones(1, max(len(legal), 1), dtype=torch.bool, device=state_b.device)
        if not legal:
            mask[:] = False
        heads = self.core.score_actions(state_b, option_hidden_b, legal_mask=mask if legal else None)
        route_decision = choose_route(
            self.config,
            n_legal=len(legal),
            policy_logits=heads.policy_logits[0] if legal else None,
        )

        planner_action: tuple[int, ...] = ()
        plan: Optional[TurnPlan] = None
        if legal and route_decision.route is Route.DIRECT:
            idx = int(torch.argmax(heads.policy_logits[0]).item())
            planner_action = legal[idx].option_index_path
        elif legal:
            new_turn = dep.turn_key() != self.active_turn_key
            if new_turn:
                self.active_turn_key = dep.turn_key()
                self.budget.model_calls = max(self.budget.model_calls, 0)
                candidates = self.core.propose_root_plans(
                    state_b,
                    legal,
                    heads,
                    observation_hash=dep.observation_hash,
                    encoder_profile=self.config.profile.value,
                )
                if candidates and route_decision.route is Route.RECURSIVE:
                    plan, _r = refine_plan(
                        self.core,
                        candidates[0],
                        state_vec=state_b,
                        option_hidden=option_hidden_b,
                        legal=legal,
                        budget=self.budget,
                    )
                elif candidates:
                    plan = candidates[0]
                if plan is not None:
                    self.executor.load(plan)
            step = self.executor.next_action(legal, observations=observations)
            if step.action is not None:
                planner_action = step.action
                plan = self.executor.plan
            elif legal:
                idx = int(torch.argmax(heads.policy_logits[0]).item())
                planner_action = legal[idx].option_index_path

        fallback = tuple(int(x) for x in (fallback_action or planner_action or ()))
        used = bool(self.config.selects_actions and planner_action)
        selected = planner_action if used else fallback
        if used and legal and selected not in {a.option_index_path for a in legal}:
            used = False
            selected = fallback
            reason = ReasonCode.ILLEGAL_OPTION_INDEX
        else:
            reason = ReasonCode.OK if planner_action else ReasonCode.FALLBACK_POLICY

        trace = ShadowTrace(
            enabled=True,
            mode=self.config.mode.value,
            turn_key=dep.turn_key(),
            route=route_decision.route.value,
            reason=reason if used or self.config.shadow_only else ReasonCode.SHADOW_ONLY,
            selected_action=selected,
            planner_action=planner_action,
            used_for_selection=used,
            observation_hash=dep.observation_hash,
            plan_id=plan.plan_id if plan is not None else "",
            budget=budget_to_trace(self.budget),
            extras={"route_diagnostics": route_decision.diagnostics},
        )
        self.last_trace = trace
        return ControllerResult(
            action=selected,
            reason=trace.reason,
            route=route_decision.route.value,
            used_for_selection=used,
            plan=plan,
            trace=trace,
            policy_logits=heads.policy_logits.detach(),
        )
