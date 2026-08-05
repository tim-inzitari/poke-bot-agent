"""Recursive Turn Planner: typed, bounded, latent, batched."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import RTPConfig
from .dynamics import LatentTransitionDynamics
from .legality import TypedLegalityVerifier
from .memory import PersistentTurnMemory
from .types import (
    NodeKind,
    ObservationPredicate,
    PlanNode,
    SubgoalKind,
    TurnProgram,
)


@dataclass(frozen=True)
class PlanProposal:
    program: TurnProgram
    legal: bool
    score: float
    uncertainty: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnDecision:
    """Result of one turn-level planning call."""

    mode: str
    action: tuple[int, ...]
    program: Optional[TurnProgram] = None
    memory: Optional[PersistentTurnMemory] = None
    neural_passes: int = 0
    used_direct_policy: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RecursiveTurnPlanner(nn.Module):
    """Lightweight recursive turn planner.

    Computational unit: approximately one plan (+ occasional repair) per turn,
    not one search tree per atomic decision.
    """

    def __init__(
        self,
        config: Optional[RTPConfig] = None,
        *,
        dynamics: Optional[LatentTransitionDynamics] = None,
        legality: Optional[TypedLegalityVerifier] = None,
    ) -> None:
        super().__init__()
        self.config = config or RTPConfig()
        self.dynamics = dynamics or LatentTransitionDynamics(
            self.config.d_model,
            width=self.config.dynamics_width,
        )
        self.legality = legality or TypedLegalityVerifier()
        d = self.config.d_model
        # Root plan proposal: maps H -> candidate plan embeddings / scores.
        self.root_plan_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, self.config.num_plan_candidates),
        )
        # Complexity controller: whether to recurse or use direct policy.
        self.complexity_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        # Subplanner: refine unresolved subgoals into short chunks.
        self.subgoal_head = nn.Sequential(
            nn.LayerNorm(d + len(SubgoalKind)),
            nn.Linear(d + len(SubgoalKind), d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.subgoal_action_head = nn.Linear(d, 1)
        self._neural_passes = 0

    def reset_pass_counter(self) -> None:
        self._neural_passes = 0

    def _consume_pass(self) -> None:
        self._neural_passes += 1
        if self._neural_passes > self.config.max_neural_passes:
            raise RuntimeError(
                f"RTP neural-pass budget exceeded ({self.config.max_neural_passes})"
            )

    def encode_memory(
        self,
        state_vec: Tensor,
        *,
        legal_actions: Sequence[Sequence[int]],
        spatial_memory: Optional[Tensor] = None,
        value_estimate: float = 0.0,
        remaining_resources: Optional[dict[str, Any]] = None,
        matchup_route: int = -1,
        turn_objective: str = "",
        observations: Optional[dict[str, Any]] = None,
    ) -> PersistentTurnMemory:
        """Cache turn-start encoding once."""
        if state_vec.dim() == 2:
            if state_vec.size(0) != 1:
                raise ValueError("encode_memory expects a single state vector")
            state_vec = state_vec[0]
        if state_vec.dim() != 1 or state_vec.numel() != self.config.d_model:
            raise ValueError("state_vec width must match RTPConfig.d_model")
        legal = tuple(tuple(int(x) for x in action) for action in legal_actions)
        return PersistentTurnMemory(
            H=state_vec.detach().clone(),
            spatial_memory=(
                None if spatial_memory is None else spatial_memory.detach().clone()
            ),
            value_estimate=float(value_estimate),
            legal_actions=legal,
            remaining_resources=dict(remaining_resources or {}),
            matchup_route=int(matchup_route),
            turn_objective=str(turn_objective),
            observations=dict(observations or {}),
        )

    def policy_entropy(self, logits: Tensor) -> float:
        if logits.numel() == 0:
            return 0.0
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        return float((-(probs * log_probs).sum()).item())

    def should_recurse(
        self,
        memory: PersistentTurnMemory,
        *,
        policy_logits: Optional[Tensor] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Complexity gate: simple positions use direct policy."""
        n_options = len(memory.legal_actions)
        entropy = (
            self.policy_entropy(policy_logits)
            if policy_logits is not None
            else 0.0
        )
        self._consume_pass()
        complexity_logit = self.complexity_head(memory.H.unsqueeze(0)).squeeze()
        complexity_score = float(torch.sigmoid(complexity_logit).item())
        by_options = n_options >= self.config.complexity_option_threshold
        by_entropy = entropy >= self.config.complexity_entropy_threshold
        by_head = complexity_score >= 0.5
        recurse = bool(by_options or by_entropy or by_head)
        return recurse, {
            "n_options": n_options,
            "entropy": entropy,
            "complexity_score": complexity_score,
            "by_options": by_options,
            "by_entropy": by_entropy,
            "by_head": by_head,
        }

    def direct_policy_action(
        self,
        policy_logits: Tensor,
        legal_actions: Sequence[Sequence[int]],
    ) -> tuple[int, ...]:
        if not legal_actions:
            raise ValueError("direct policy requires legal actions")
        if policy_logits.numel() != len(legal_actions):
            raise ValueError("policy_logits / legal_actions length mismatch")
        idx = int(torch.argmax(policy_logits).item())
        return tuple(int(x) for x in legal_actions[idx])

    def _subgoal_one_hot(
        self,
        kind: SubgoalKind,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        vec = torch.zeros(len(SubgoalKind), device=device, dtype=dtype)
        vec[list(SubgoalKind).index(kind)] = 1.0
        return vec

    def _candidate_skeleton(
        self,
        memory: PersistentTurnMemory,
        candidate_idx: int,
    ) -> PlanNode:
        """Build a typed candidate program skeleton from legal actions."""
        legal = memory.legal_actions
        if not legal:
            return PlanNode(kind=NodeKind.STOP)
        # Spread candidates across the legal set; later stages refine.
        stride = max(1, len(legal) // self.config.num_plan_candidates)
        primary = legal[min(candidate_idx * stride, len(legal) - 1)]
        secondary = legal[min(candidate_idx * stride + 1, len(legal) - 1)]
        subgoal_names = list(SubgoalKind)
        subgoal = subgoal_names[candidate_idx % len(subgoal_names)]
        # SEQUENCE(action, SUBGOAL(...), BRANCH(...), END_TURN-ish STOP)
        return PlanNode(
            kind=NodeKind.SEQUENCE,
            children=(
                PlanNode(kind=NodeKind.PRIMITIVE, action=primary),
                PlanNode(
                    kind=NodeKind.SUBGOAL,
                    subgoal=subgoal,
                    subgoal_label=subgoal.value,
                    constraints=("same_turn",),
                ),
                PlanNode(
                    kind=NodeKind.BRANCH,
                    predicate=ObservationPredicate(
                        name="target_found",
                        expected=True,
                    ),
                    children=(
                        PlanNode(kind=NodeKind.PRIMITIVE, action=secondary),
                        PlanNode(kind=NodeKind.STOP),
                    ),
                ),
                PlanNode(kind=NodeKind.END_TURN),
            ),
            metadata={"candidate_index": candidate_idx},
        )

    def propose_plans(
        self,
        memory: PersistentTurnMemory,
    ) -> list[TurnProgram]:
        """Batched root proposal of complete turn-program candidates."""
        self._consume_pass()
        logits = self.root_plan_head(memory.H.unsqueeze(0)).squeeze(0)
        probs = F.softmax(logits.float(), dim=-1)
        programs: list[TurnProgram] = []
        for idx in range(self.config.num_plan_candidates):
            root = self._candidate_skeleton(memory, idx)
            programs.append(
                TurnProgram(
                    root=root,
                    score=float(probs[idx].item()),
                    plan_id=f"root-{idx}",
                    diagnostics={"root_prob": float(probs[idx].item())},
                )
            )
        return programs

    def refine_subgoal(
        self,
        memory: PersistentTurnMemory,
        node: PlanNode,
        *,
        depth: int,
        remaining_budget: int,
    ) -> PlanNode:
        """Recursively solve one unresolved subgoal into a short chunk."""
        if node.kind is not NodeKind.SUBGOAL:
            return node
        if depth >= self.config.max_recursion_depth or remaining_budget <= 0:
            return PlanNode(kind=NodeKind.STOP, metadata={"reason": "budget_stop"})
        if not memory.legal_actions:
            return PlanNode(kind=NodeKind.STOP, metadata={"reason": "no_legal"})

        assert node.subgoal is not None
        self._consume_pass()
        device = memory.H.device
        dtype = memory.H.dtype
        subgoal_vec = self._subgoal_one_hot(node.subgoal, device=device, dtype=dtype)
        hidden = self.subgoal_head(
            torch.cat((memory.snapshot_latent(), subgoal_vec), dim=-1).unsqueeze(0)
        )
        # Score legal actions as a batch for this subgoal.
        embeds = self.dynamics.embed_action_ids(
            memory.legal_actions, device=device, dtype=dtype
        )
        # Broadcast subgoal hidden against action embeds via dynamics value.
        dyn = self.dynamics(hidden, embeds)
        scores = dyn["value"] - self.config.compute_cost_penalty * dyn["uncertainty"]
        # Also mix a tiny learned bias from subgoal_action_head on hidden.
        bias = self.subgoal_action_head(hidden).squeeze()
        scores = scores + 0.05 * bias
        best = int(torch.argmax(scores).item())
        action = memory.legal_actions[best]
        # Optionally emit one more nested subgoal if budget remains and depth low.
        children: list[PlanNode] = [
            PlanNode(
                kind=NodeKind.PRIMITIVE,
                action=action,
                metadata={
                    "from_subgoal": node.subgoal.value,
                    "depth": depth,
                },
            )
        ]
        if remaining_budget > 1 and depth + 1 < self.config.max_recursion_depth:
            nested_kind = list(SubgoalKind)[
                (list(SubgoalKind).index(node.subgoal) + 1) % len(SubgoalKind)
            ]
            nested = PlanNode(
                kind=NodeKind.SUBGOAL,
                subgoal=nested_kind,
                subgoal_label=nested_kind.value,
            )
            children.append(
                self.refine_subgoal(
                    memory,
                    nested,
                    depth=depth + 1,
                    remaining_budget=remaining_budget - 1,
                )
            )
        children.append(PlanNode(kind=NodeKind.STOP))
        return PlanNode(kind=NodeKind.SEQUENCE, children=tuple(children))

    def refine_program(
        self,
        memory: PersistentTurnMemory,
        program: TurnProgram,
        *,
        depth: int = 0,
    ) -> TurnProgram:
        """Recursively expand unresolved SUBGOAL nodes (shallow, batched-ready)."""

        def walk(node: PlanNode, depth_now: int) -> PlanNode:
            if node.kind is NodeKind.SUBGOAL:
                return self.refine_subgoal(
                    memory,
                    node,
                    depth=depth_now,
                    remaining_budget=self.config.max_recursion_depth - depth_now,
                )
            if not node.children:
                return node
            return node.replace_children(
                tuple(walk(child, depth_now + 1) for child in node.children)
            )

        refined_root = walk(program.root, depth)
        # Truncate over-long primitive chains.
        primitives = refined_root.flatten_primitives()
        if len(primitives) > self.config.max_plan_length:
            trimmed = primitives[: self.config.max_plan_length]
            refined_root = PlanNode(
                kind=NodeKind.SEQUENCE,
                children=tuple(
                    PlanNode(kind=NodeKind.PRIMITIVE, action=a) for a in trimmed
                )
                + (PlanNode(kind=NodeKind.END_TURN),),
                metadata={"truncated": True},
            )
        return TurnProgram(
            root=refined_root,
            score=program.score,
            uncertainty=program.uncertainty,
            compute_cost=program.compute_cost + float(self._neural_passes),
            plan_id=program.plan_id,
            diagnostics=dict(program.diagnostics),
        )

    def score_program(
        self,
        memory: PersistentTurnMemory,
        program: TurnProgram,
    ) -> PlanProposal:
        report = self.legality.validate_program(program, memory.legal_actions)
        if not report.valid:
            return PlanProposal(
                program=program,
                legal=False,
                score=float("-inf"),
                uncertainty=1.0,
                diagnostics={"illegal_reason": report.reason},
            )
        actions = report.kept_actions
        rollout = self.dynamics.rollout_program_value(
            memory.snapshot_latent(),
            actions,
            max_horizon=min(len(actions), self.config.max_plan_length),
        )
        compute_cost = float(program.compute_cost) + float(rollout["horizon"])
        score = (
            float(rollout["value"])
            + float(program.score)
            - self.config.compute_cost_penalty * compute_cost
            - 0.25 * float(rollout["uncertainty"])
        )
        scored = TurnProgram(
            root=program.root,
            score=score,
            uncertainty=float(rollout["uncertainty"]),
            compute_cost=compute_cost,
            plan_id=program.plan_id,
            diagnostics={
                **program.diagnostics,
                "rollout_value": rollout["value"],
                "rollout_uncertainty": rollout["uncertainty"],
                "rollout_horizon": rollout["horizon"],
            },
        )
        return PlanProposal(
            program=scored,
            legal=True,
            score=score,
            uncertainty=float(rollout["uncertainty"]),
            diagnostics=dict(scored.diagnostics),
        )

    def plan_turn(
        self,
        memory: PersistentTurnMemory,
        *,
        policy_logits: Optional[Tensor] = None,
        force_recurse: Optional[bool] = None,
    ) -> TurnDecision:
        """Main entry: direct policy or recursive conditional plan."""
        self.reset_pass_counter()
        if not memory.legal_actions:
            return TurnDecision(
                mode="empty",
                action=(),
                memory=memory,
                neural_passes=self._neural_passes,
                diagnostics={"reason": "no_legal_actions"},
            )

        if force_recurse is None:
            recurse, gate = self.should_recurse(
                memory, policy_logits=policy_logits
            )
        else:
            recurse, gate = bool(force_recurse), {"forced": force_recurse}

        if not recurse:
            if policy_logits is None:
                # Uniform fallback when caller did not supply leaf logits.
                policy_logits = torch.zeros(
                    len(memory.legal_actions),
                    device=memory.H.device,
                    dtype=memory.H.dtype,
                )
            action = self.direct_policy_action(policy_logits, memory.legal_actions)
            return TurnDecision(
                mode="direct_policy",
                action=action,
                memory=memory,
                neural_passes=self._neural_passes,
                used_direct_policy=True,
                diagnostics={"complexity_gate": gate},
            )

        proposals = self.propose_plans(memory)
        refined = [self.refine_program(memory, p) for p in proposals]
        scored = [self.score_program(memory, p) for p in refined]
        legal_scored = [p for p in scored if p.legal and math.isfinite(p.score)]
        if not legal_scored:
            # Fail closed to direct policy if every plan pruned.
            if policy_logits is None:
                policy_logits = torch.zeros(
                    len(memory.legal_actions),
                    device=memory.H.device,
                    dtype=memory.H.dtype,
                )
            action = self.direct_policy_action(policy_logits, memory.legal_actions)
            return TurnDecision(
                mode="direct_policy_fallback",
                action=action,
                memory=memory,
                neural_passes=self._neural_passes,
                used_direct_policy=True,
                diagnostics={
                    "complexity_gate": gate,
                    "reason": "all_plans_illegal",
                },
            )

        best = max(legal_scored, key=lambda p: p.score)
        action = best.program.first_action()
        if action is None:
            action = memory.legal_actions[0]
        return TurnDecision(
            mode="recursive_plan",
            action=action,
            program=best.program,
            memory=memory,
            neural_passes=self._neural_passes,
            used_direct_policy=False,
            diagnostics={
                "complexity_gate": gate,
                "num_candidates": len(proposals),
                "num_legal_scored": len(legal_scored),
                "best_score": best.score,
                "best_uncertainty": best.uncertainty,
                "online_sim_verify_budget": self.config.online_sim_verify_budget,
            },
        )
