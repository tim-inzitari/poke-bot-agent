"""Bounded recursive subgoal refinement with shared planner weights."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .budget import TurnComputeBudget
from .legal_action import ActionSelector, LegalAction
from .model_core import PokeRLMModelCore
from .plan_ir import (
    ConditionalNode,
    GoalCode,
    PlanNode,
    PrimitiveAction,
    SequenceNode,
    StopNode,
    StopReason,
    SubgoalNode,
    TurnPlan,
)
from .reasons import ReasonCode


def _goal_one_hot(goal: GoalCode, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    vec = torch.zeros(len(GoalCode), device=device, dtype=dtype)
    vec[list(GoalCode).index(goal)] = 1.0
    return vec


def refine_subgoal(
    core: PokeRLMModelCore,
    *,
    state_vec: Tensor,
    option_hidden: Tensor,
    legal: tuple[LegalAction, ...],
    node: SubgoalNode,
    depth: int,
    budget: TurnComputeBudget,
) -> tuple[PlanNode, ReasonCode]:
    """Refine one subgoal using the shared recursive cell."""
    if depth > core.config.max_depth:
        return StopNode(reason=StopReason.BUDGET), ReasonCode.BUDGET_DEPTH
    reason = budget.note_depth(depth)
    if reason is not ReasonCode.OK:
        return StopNode(reason=StopReason.BUDGET), reason
    reason = budget.consume_model_call(1)
    if reason is not ReasonCode.OK:
        return StopNode(reason=StopReason.BUDGET), reason
    if not legal:
        return node.fallback, ReasonCode.EMPTY_LEGAL

    z = state_vec if state_vec.dim() == 1 else state_vec[0]
    device = z.device
    dtype = z.dtype
    g = _goal_one_hot(node.goal_code, device=device, dtype=dtype)
    hidden = core.recursive_cell(torch.cat((z, g), dim=-1).unsqueeze(0))
    stop_logit = core.stop_head(hidden).squeeze()
    if float(torch.sigmoid(stop_logit).item()) > 0.85 and depth > 0:
        return StopNode(reason=StopReason.OBJECTIVE_MET), ReasonCode.OK

    action_embed = option_hidden[0] if option_hidden.dim() == 3 else option_hidden
    dyn = core.dynamics(hidden, action_embed)
    cost = torch.sigmoid(core.compute_cost_head(hidden)).squeeze()
    scores = dyn["value"] - core.config.compute_cost_weight * (
        dyn["uncertainty"] + cost
    )
    best = int(torch.argmax(scores).item())
    chosen = legal[best]
    budget.subgoals_refined += 1
    primitive = PrimitiveAction(
        selector=ActionSelector(
            fingerprint=chosen.fingerprint,
            option_type=chosen.option_type,
            card_id=chosen.card_id,
            attack_id=chosen.attack_id,
            planned_option_index_path=chosen.option_index_path,
        ),
        planned_option_index=chosen.option_index_path,
        on_unresolvable=node.fallback,
    )
    children: list[PlanNode] = [primitive]
    if (
        depth + 1 <= core.config.max_depth
        and budget.remaining_model_calls() > 0
        and core.config.share_planner_weights_across_depth
    ):
        nested_goal = list(GoalCode)[
            (list(GoalCode).index(node.goal_code) + 1) % len(GoalCode)
        ]
        nested = SubgoalNode(goal_code=nested_goal, fallback=node.fallback)
        next_latent = dyn["next_latent"][best]
        refined_nested, nested_reason = refine_subgoal(
            core,
            state_vec=next_latent,
            option_hidden=option_hidden,
            legal=legal,
            node=nested,
            depth=depth + 1,
            budget=budget,
        )
        if nested_reason is ReasonCode.OK:
            children.append(refined_nested)
    children.append(StopNode(reason=StopReason.END_TURN))
    return SequenceNode(children=tuple(children)), ReasonCode.OK


def refine_plan(
    core: PokeRLMModelCore,
    plan: TurnPlan,
    *,
    state_vec: Tensor,
    option_hidden: Tensor,
    legal: tuple[LegalAction, ...],
    budget: TurnComputeBudget,
    max_depth: Optional[int] = None,
) -> tuple[TurnPlan, ReasonCode]:
    depth_cap = core.config.max_depth if max_depth is None else int(max_depth)

    def walk(node: PlanNode, depth: int) -> PlanNode:
        if isinstance(node, SubgoalNode):
            if depth > depth_cap:
                return node.fallback
            refined, _reason = refine_subgoal(
                core,
                state_vec=state_vec,
                option_hidden=option_hidden,
                legal=legal,
                node=node,
                depth=depth,
                budget=budget,
            )
            return refined
        if isinstance(node, SequenceNode):
            return SequenceNode(
                children=tuple(walk(child, depth + 1) for child in node.children)
            )
        if isinstance(node, ConditionalNode):
            return ConditionalNode(
                predicate=node.predicate,
                if_true=walk(node.if_true, depth + 1),
                if_false=walk(node.if_false, depth + 1),
            )
        return node

    refined_root = walk(plan.root, 0)
    reason = budget.emit_nodes(max(1, plan.node_count()))
    return (
        TurnPlan(
            plan_id=plan.plan_id,
            schema_version=plan.schema_version,
            objective=plan.objective,
            root=refined_root,
            expected_end_turn=dict(plan.expected_end_turn),
            value=plan.value,
            uncertainty=plan.uncertainty,
            budget=plan.budget,
            provenance=plan.provenance,
        ),
        reason if reason is not ReasonCode.OK else ReasonCode.OK,
    )


def shared_recursive_cell_identity(core: PokeRLMModelCore) -> int:
    """Stable id of the shared recursive cell module."""
    return id(core.recursive_cell)
