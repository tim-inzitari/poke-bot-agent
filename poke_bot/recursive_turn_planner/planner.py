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


class RTPNeuralPassBudgetExceeded(RuntimeError):
    """Raised when a bounded planner would exceed its serialized pass ceiling."""

    def __init__(self, *, used: int, limit: int) -> None:
        self.used = int(used)
        self.limit = int(limit)
        super().__init__(f"RTP neural-pass budget exceeded ({self.limit})")


def required_recursive_passes(
    config: RTPConfig,
    *,
    force_recurse: bool = False,
) -> int:
    """Return the current skeleton's worst-case pass count for one full plan.

    Every recursive turn proposes all root candidates.  The current candidate
    skeleton has one subgoal at depth one; it can expand at most
    ``max_recursion_depth - 1`` times per candidate.  The normal path also
    spends one pass on the complexity gate, whereas a forced repair/replan
    skips that gate.  Keep this calculation beside the implementation so a
    serving loader can reject a configuration that can never finish a plan.
    """

    subgoal_passes_per_candidate = max(0, int(config.max_recursion_depth) - 1)
    complexity_gate_passes = 0 if force_recurse else 1
    root_proposal_passes = 1
    return (
        complexity_gate_passes
        + root_proposal_passes
        + int(config.num_plan_candidates) * subgoal_passes_per_candidate
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
            prefer_option_hidden=self.config.prefer_option_hidden,
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
            nn.Linear(d, max(d // 2, 1)),
            nn.GELU(),
            nn.Linear(max(d // 2, 1), 1),
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

    def inventory(self) -> dict[str, object]:
        dyn_inv = getattr(self.dynamics, "inventory", None)
        return {
            "schema": self.config.schema,
            "sizing_profile": self.config.sizing_profile,
            "d_model": self.config.d_model,
            "dynamics_width": self.config.dynamics_width,
            "num_plan_candidates": self.config.num_plan_candidates,
            "max_recursion_depth": self.config.max_recursion_depth,
            "max_neural_passes": self.config.max_neural_passes,
            "required_neural_passes_normal": self.required_recursive_passes(),
            "required_neural_passes_forced_replan": self.required_recursive_passes(
                force_recurse=True
            ),
            "max_plan_length": self.config.max_plan_length,
            "complexity_option_threshold": self.config.complexity_option_threshold,
            "skip_trivial_decisions": self.config.skip_trivial_decisions,
            "online_sim_verify_budget": self.config.online_sim_verify_budget,
            "option_batch_hint": self.config.option_batch_hint,
            "prefer_option_hidden": self.config.prefer_option_hidden,
            "policy_aid_cap": self.config.policy_aid_cap,
            "mcts_allowed": False,
            "beam_search_allowed": False,
            "competition_time_simulator_search_allowed": (
                self.config.online_sim_verify_budget > 0
            ),
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "dynamics": dyn_inv() if callable(dyn_inv) else {},
            "reuse_targets": {
                "option_hidden": "TemporalCabtTransformer.decode_options",
                "latent_lookahead": "ActionConditionedLatentLookahead",
                "complex_option_threshold": "SearchConfig.complex_option_threshold",
                "trivial_gate": "submission_budget.forced_or_trivial",
            },
        }

    def reset_pass_counter(self) -> None:
        self._neural_passes = 0

    @property
    def neural_passes(self) -> int:
        """Passes consumed by the current turn's planning attempt."""

        return int(self._neural_passes)

    def required_recursive_passes(self, *, force_recurse: bool = False) -> int:
        """Expose the current planner skeleton's complete-plan budget."""

        return required_recursive_passes(self.config, force_recurse=force_recurse)

    def _consume_pass(self) -> None:
        self._neural_passes += 1
        if self._neural_passes > self.config.max_neural_passes:
            raise RTPNeuralPassBudgetExceeded(
                used=self._neural_passes,
                limit=self.config.max_neural_passes,
            )

    def encode_memory(
        self,
        state_vec: Tensor,
        *,
        legal_actions: Sequence[Sequence[int]],
        spatial_memory: Optional[Tensor] = None,
        option_hidden: Optional[Tensor] = None,
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
        hidden = None
        if option_hidden is not None:
            if option_hidden.dim() == 3:
                if option_hidden.size(0) != 1:
                    raise ValueError("option_hidden batch must be 1 at encode")
                option_hidden = option_hidden[0]
            hidden = option_hidden.detach().clone()
        return PersistentTurnMemory(
            H=state_vec.detach().clone(),
            spatial_memory=(
                None if spatial_memory is None else spatial_memory.detach().clone()
            ),
            option_hidden=hidden,
            value_estimate=float(value_estimate),
            legal_actions=legal,
            remaining_resources=dict(remaining_resources or {}),
            matchup_route=int(matchup_route),
            turn_objective=str(turn_objective),
            observations=dict(observations or {}),
            sizing_profile=self.config.sizing_profile,
        )

    def policy_entropy(self, logits: Tensor) -> float:
        if logits.numel() == 0:
            return 0.0
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        return float((-(probs * log_probs).sum()).item())

    def is_trivial_decision(
        self,
        memory: PersistentTurnMemory,
    ) -> tuple[bool, str]:
        """Mirror submission_budget forced/trivial skips."""
        n_options = len(memory.legal_actions)
        if n_options <= 1:
            return True, "forced_or_trivial"
        return False, ""

    def should_recurse(
        self,
        memory: PersistentTurnMemory,
        *,
        policy_logits: Optional[Tensor] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Complexity gate: simple positions use direct policy."""
        n_options = len(memory.legal_actions)
        if self.config.skip_trivial_decisions:
            trivial, reason = self.is_trivial_decision(memory)
            if trivial:
                return False, {
                    "n_options": n_options,
                    "trivial": True,
                    "reason": reason,
                    "by_options": False,
                    "by_entropy": False,
                    "by_head": False,
                }
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
            "trivial": False,
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
        # Prefer encoder option_hidden; fall back to action-id embeds.
        dyn = self.dynamics.score_actions(
            hidden,
            memory.legal_actions,
            option_hidden=memory.option_hidden,
        )
        value = dyn["value"]
        uncertainty = dyn["uncertainty"]
        assert isinstance(value, Tensor) and isinstance(uncertainty, Tensor)
        scores = value - self.config.compute_cost_penalty * uncertainty
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
        if not actions:
            return PlanProposal(
                program=program,
                legal=False,
                score=float("-inf"),
                uncertainty=1.0,
                diagnostics={"illegal_reason": "no_scorable_actions"},
            )
        rollout = self.dynamics.rollout_program_value(
            memory.snapshot_latent(),
            actions,
            max_horizon=min(len(actions), self.config.max_plan_length),
            option_hidden_by_action=memory.option_hidden_map(),
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
                "option_hidden_fraction": rollout.get("option_hidden_fraction", 0.0),
                "sizing_profile": self.config.sizing_profile,
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
                "sizing_profile": self.config.sizing_profile,
                "used_option_hidden": memory.option_hidden is not None,
            },
        )
