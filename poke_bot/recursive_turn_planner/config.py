"""Configuration for the lightweight Recursive Turn Planner."""

from __future__ import annotations

from dataclasses import dataclass


RTP_SCHEMA = "poke_bot.recursive_turn_planner/v1"


@dataclass(frozen=True)
class RTPConfig:
    """Hard bounds for typed recursive turn planning.

    All recursion and neural-pass budgets are fail-closed ceilings. The planner
    must never exceed them, even when a position looks hard.

    Defaults follow the ``global_transformer`` sizing profile (d_model=256,
    dynamics_width=512) so they match ``TemporalCabtTransformer`` +
    ``ActionConditionedLatentLookahead``. Use
    ``profiles.get_profile("pure_rl").to_config()`` for the lean d=96 parent.
    """

    schema: str = RTP_SCHEMA
    sizing_profile: str = "global_transformer"
    d_model: int = 256
    #: Trunk width for latent dynamics; mirrors latent_lookahead_width=512.
    dynamics_width: int = 512
    num_plan_candidates: int = 4
    max_recursion_depth: int = 2
    max_neural_passes: int = 4
    max_plan_length: int = 12
    #: Shared with SearchConfig.complex_option_threshold / mcts.planned_sims.
    complexity_option_threshold: int = 8
    complexity_entropy_threshold: float = 1.5
    #: Skip recursion for forced/trivial decisions (submission_budget pattern).
    skip_trivial_decisions: bool = True
    #: Online simulator verification calls reserved for finalists / repair.
    #: Lightweight default is zero: latent evaluation only.
    online_sim_verify_budget: int = 0
    repair_budget: int = 1
    compute_cost_penalty: float = 0.01
    #: Hint for option packing width when scoring legal actions in batch.
    option_batch_hint: int = 256
    #: Prefer encoder option_hidden over deterministic action-id embeds.
    prefer_option_hidden: bool = True
    #: Bound used when adapting latent-lookahead policy-aid style residuals.
    policy_aid_cap: float = 0.25
    #: Subgoals initially human-defined; later versions may learn discrete codes.
    default_subgoals: tuple[str, ...] = (
        "establish_attacker",
        "find_resource",
        "maximize_draw",
        "preserve_escape",
        "reach_damage_threshold",
        "setup_next_turn",
    )

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.dynamics_width <= 0:
            raise ValueError("dynamics_width must be positive")
        if self.num_plan_candidates < 1:
            raise ValueError("num_plan_candidates must be positive")
        if self.max_recursion_depth < 0:
            raise ValueError("max_recursion_depth cannot be negative")
        if self.max_neural_passes < 1:
            raise ValueError("max_neural_passes must be positive")
        if self.max_plan_length < 1:
            raise ValueError("max_plan_length must be positive")
        if self.complexity_option_threshold < 1:
            raise ValueError("complexity_option_threshold must be positive")
        if self.complexity_entropy_threshold < 0.0:
            raise ValueError("complexity_entropy_threshold cannot be negative")
        if self.online_sim_verify_budget < 0:
            raise ValueError("online_sim_verify_budget cannot be negative")
        if self.repair_budget < 0:
            raise ValueError("repair_budget cannot be negative")
        if self.compute_cost_penalty < 0.0:
            raise ValueError("compute_cost_penalty cannot be negative")
        if self.option_batch_hint < 1:
            raise ValueError("option_batch_hint must be positive")
        if not 0.0 < float(self.policy_aid_cap) <= 1.0:
            raise ValueError("policy_aid_cap must be in (0, 1]")
