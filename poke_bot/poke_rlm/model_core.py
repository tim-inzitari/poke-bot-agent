"""Batched action decoder heads, latent dynamics, and root plan proposer.

Hard rules:
- one shared state vector for the legal set (no backbone-per-action);
- all legal actions scored in one batched tensor pass;
- no Python loop over legal actions in the deployed forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import PokeRLMConfig
from .legal_action import ActionSelector, LegalAction
from .plan_ir import (
    PLAN_IR_SCHEMA_VERSION,
    ConditionalNode,
    DistributionalValue,
    GoalCode,
    ObjectiveCode,
    ObservationPredicate,
    PlanBudget,
    PlanProvenance,
    PredicateFamily,
    PrimitiveAction,
    SequenceNode,
    StopNode,
    StopReason,
    SubgoalNode,
    TurnPlan,
    validate_plan_static,
)
from .reasons import ReasonCode


@dataclass(frozen=True)
class ActionHeadOutputs:
    policy_logits: Tensor  # [B, N]
    q_quantiles: Tensor  # [B, N, Q]
    horizon_values: Tensor  # [B, N, H]
    successor: Tensor  # [B, N, S]
    uncertainty: Tensor  # [B, N]
    state_value: Tensor  # [B]
    option_hidden: Tensor  # [B, N, D]


class ParallelActionDecoder(nn.Module):
    """Score all legal option embeddings against one state vector."""

    def __init__(self, config: PokeRLMConfig) -> None:
        super().__init__()
        d = int(config.d_model)
        self.d_model = d
        self.q_quantiles = int(config.q_quantiles)
        self.value_horizons = int(config.value_horizons)
        self.successor_dim = int(config.successor_dim)
        self.input_norm = nn.LayerNorm(2 * d)
        hidden = max(d, d * 2)
        self.trunk = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.GELU(),
        )
        # Fused deployment projection: policy + Q + horizons + unc + successor
        fused_out = 1 + self.q_quantiles + self.value_horizons + 1 + self.successor_dim
        self.fused = nn.Linear(d, fused_out)
        self.state_value = nn.Linear(d, 1)
        nn.init.zeros_(self.fused.bias)

    def forward(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        *,
        legal_mask: Optional[Tensor] = None,
    ) -> ActionHeadOutputs:
        """Batched decode.

        state_vec: [B, D] or [D]
        option_hidden: [B, N, D]
        legal_mask: [B, N] True for valid options
        """
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        if option_hidden.dim() != 3:
            raise ValueError("option_hidden must be [B,N,D]")
        if state_vec.size(0) != option_hidden.size(0):
            raise ValueError("batch mismatch")
        if state_vec.size(-1) != self.d_model or option_hidden.size(-1) != self.d_model:
            raise ValueError("d_model mismatch")
        b, n, d = option_hidden.shape
        expanded = state_vec.unsqueeze(1).expand(b, n, d)
        h = self.trunk(self.input_norm(torch.cat((expanded, option_hidden), dim=-1)))
        fused = self.fused(h)
        idx = 0
        policy = fused[..., idx]
        idx += 1
        q = fused[..., idx : idx + self.q_quantiles]
        idx += self.q_quantiles
        horizons = fused[..., idx : idx + self.value_horizons]
        idx += self.value_horizons
        uncertainty = torch.sigmoid(fused[..., idx])
        idx += 1
        successor = fused[..., idx : idx + self.successor_dim]
        if legal_mask is not None:
            if legal_mask.shape != policy.shape:
                raise ValueError("legal_mask shape mismatch")
            policy = policy.masked_fill(~legal_mask, float("-inf"))
            uncertainty = uncertainty.masked_fill(~legal_mask, 1.0)
        state_value = torch.tanh(self.state_value(state_vec)).squeeze(-1)
        return ActionHeadOutputs(
            policy_logits=policy,
            q_quantiles=q,
            horizon_values=horizons,
            successor=successor,
            uncertainty=uncertainty,
            state_value=state_value,
            option_hidden=h,
        )


class LatentDynamics(nn.Module):
    """Short action-conditioned latent dynamics z' = D(z, a)."""

    def __init__(self, config: PokeRLMConfig) -> None:
        super().__init__()
        d = int(config.d_model)
        width = max(32, d * int(config.dynamics_hidden_multiplier))
        self.d_model = d
        blocks = []
        in_dim = 2 * d
        for _ in range(max(1, int(config.dynamics_blocks))):
            blocks.extend([nn.Linear(in_dim, width), nn.GELU()])
            in_dim = width
        self.trunk = nn.Sequential(*blocks)
        self.next_latent = nn.Linear(width, d)
        self.delta_head = nn.Linear(width, 4)  # hand/energy/bench/prize deltas
        self.value_head = nn.Linear(width, 1)
        self.uncertainty_head = nn.Linear(width, 1)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def forward(self, latent: Tensor, action_embed: Tensor) -> dict[str, Tensor]:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if action_embed.dim() == 1:
            action_embed = action_embed.unsqueeze(0)
        if latent.size(0) == 1 and action_embed.size(0) > 1:
            latent = latent.expand(action_embed.size(0), -1)
        h = self.trunk(torch.cat((latent, action_embed), dim=-1))
        return {
            "next_latent": self.next_latent(h),
            "structured_delta": self.delta_head(h),
            "value": torch.tanh(self.value_head(h).squeeze(-1)),
            "uncertainty": torch.sigmoid(self.uncertainty_head(h).squeeze(-1)),
        }


class RootPlanProposer(nn.Module):
    """Finite typed root-plan proposer producing K candidates in one pass."""

    def __init__(self, config: PokeRLMConfig) -> None:
        super().__init__()
        d = int(config.d_model)
        self.k = int(config.root_plan_candidates)
        self.objective_count = len(ObjectiveCode)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, self.k * (1 + self.objective_count)),
        )

    def forward(self, state_vec: Tensor) -> dict[str, Tensor]:
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        raw = self.head(state_vec).view(state_vec.size(0), self.k, 1 + self.objective_count)
        scores = raw[..., 0]
        obj_logits = raw[..., 1:]
        return {"candidate_scores": scores, "objective_logits": obj_logits}


class PokeRLMModelCore(nn.Module):
    """Composable planning attachment for an external shared encoder."""

    def __init__(self, config: PokeRLMConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = ParallelActionDecoder(config)
        self.dynamics = LatentDynamics(config)
        self.root_proposer = RootPlanProposer(config)
        # Shared recursive cell (Phase 5) — weight-shared across depth.
        d = int(config.d_model)
        self.recursive_cell = nn.Sequential(
            nn.LayerNorm(d + len(GoalCode)),
            nn.Linear(d + len(GoalCode), d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.stop_head = nn.Linear(d, 1)
        self.compute_cost_head = nn.Linear(d, 1)
        # Learned complexity router (Phase-7 training); inference may still use
        # the heuristic router until active mode opts into these heads.
        self.route_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, max(d // 2, 1)),
            nn.GELU(),
            nn.Linear(max(d // 2, 1), 3),
        )
        self.recurse_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, max(d // 2, 1)),
            nn.GELU(),
            nn.Linear(max(d // 2, 1), 1),
        )

    def parameter_inventory(self) -> dict[str, int]:
        def _n(module: nn.Module) -> int:
            return int(sum(p.numel() for p in module.parameters()))

        return {
            "decoder": _n(self.decoder),
            "dynamics": _n(self.dynamics),
            "root_proposer": _n(self.root_proposer),
            "recursive_cell": _n(self.recursive_cell),
            "stop_head": _n(self.stop_head),
            "compute_cost_head": _n(self.compute_cost_head),
            "route_head": _n(self.route_head),
            "recurse_head": _n(self.recurse_head),
            "total": _n(self),
        }

    def route_logits(self, state_vec: Tensor) -> Tensor:
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        return self.route_head(state_vec)

    def recurse_logits(self, state_vec: Tensor) -> Tensor:
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        return self.recurse_head(state_vec).squeeze(-1)

    def score_actions(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        *,
        legal_mask: Optional[Tensor] = None,
    ) -> ActionHeadOutputs:
        return self.decoder(state_vec, option_hidden, legal_mask=legal_mask)

    def propose_root_plans(
        self,
        state_vec: Tensor,
        legal: tuple[LegalAction, ...],
        heads: ActionHeadOutputs,
        *,
        observation_hash: str = "",
        encoder_profile: str = "",
    ) -> list[TurnPlan]:
        """Compile K typed candidates from proposer + legal actions."""
        prop = self.root_proposer(state_vec)
        scores = prop["candidate_scores"][0]
        obj_logits = prop["objective_logits"][0]
        if not legal:
            return []
        # Rank legal actions by policy in one tensor op (no python argmax loop body over N for scoring).
        policy = heads.policy_logits[0]
        order = torch.argsort(policy, descending=True)
        order_list = [int(i) for i in order.tolist() if i < len(legal)]
        plans: list[TurnPlan] = []
        objectives = list(ObjectiveCode)
        for k in range(self.config.root_plan_candidates):
            obj_idx = int(torch.argmax(obj_logits[k]).item())
            objective = objectives[obj_idx % len(objectives)]
            primary = legal[order_list[min(k, len(order_list) - 1)]]
            secondary = legal[order_list[min(k + 1, len(order_list) - 1)]]
            goal = list(GoalCode)[k % len(GoalCode)]
            root = SequenceNode(
                children=(
                    PrimitiveAction(
                        selector=ActionSelector(
                            fingerprint=primary.fingerprint,
                            option_type=primary.option_type,
                            card_id=primary.card_id,
                            attack_id=primary.attack_id,
                            planned_option_index_path=primary.option_index_path,
                        ),
                        planned_option_index=primary.option_index_path,
                        on_unresolvable=StopNode(reason=StopReason.FALLBACK),
                    ),
                    SubgoalNode(
                        goal_code=goal,
                        fallback=StopNode(reason=StopReason.FALLBACK),
                    ),
                    ConditionalNode(
                        predicate=ObservationPredicate(
                            PredicateFamily.RESULT_CLASS,
                            "target_found",
                            True,
                        ),
                        if_true=PrimitiveAction(
                            selector=ActionSelector(
                                fingerprint=secondary.fingerprint,
                                planned_option_index_path=secondary.option_index_path,
                            ),
                            planned_option_index=secondary.option_index_path,
                            on_unresolvable=StopNode(reason=StopReason.FALLBACK),
                        ),
                        if_false=StopNode(reason=StopReason.END_TURN),
                    ),
                    StopNode(reason=StopReason.END_TURN),
                )
            )
            plan = TurnPlan(
                plan_id=f"root-{k}",
                schema_version=PLAN_IR_SCHEMA_VERSION,
                objective=objective,
                root=root,
                value=DistributionalValue(mean=float(scores[k].item())),
                uncertainty=float(heads.uncertainty[0, order_list[0]].item()),
                budget=PlanBudget(
                    max_depth=self.config.max_depth,
                    max_nodes=self.config.max_plan_nodes,
                    max_subgoals=self.config.max_subgoals,
                    max_model_calls=self.config.max_neural_planner_calls_per_turn,
                    max_simulator_calls=self.config.max_simulator_calls_per_turn,
                ),
                provenance=PlanProvenance(
                    encoder_profile=encoder_profile,
                    planner_profile=self.config.profile.value,
                    observation_hash=observation_hash,
                ),
            )
            if validate_plan_static(plan) is ReasonCode.OK:
                plans.append(plan)
        return plans
