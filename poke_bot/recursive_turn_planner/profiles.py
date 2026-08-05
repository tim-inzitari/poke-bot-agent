"""Sizing profiles that bind RTP to existing in-repo model shapes.

Two production-shaped parents exist today:

* Global / Hope-style ``TemporalCabtTransformer``: ``d_model=256``
* Pure-RL lean evaluator + matchup adapters: ``d_model=96``

RTP must declare which parent it attaches to. Silent mismatch between planner
latents and encoder states is a hard failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import RTPConfig


@dataclass(frozen=True)
class RTPSizingProfile:
    """Named sizing contract for one encoder/runtime parent."""

    name: str
    d_model: int
    dynamics_width: int
    num_plan_candidates: int
    max_recursion_depth: int
    max_neural_passes: int
    max_plan_length: int
    complexity_option_threshold: int
    complexity_entropy_threshold: float
    online_sim_verify_budget: int
    repair_budget: int
    compute_cost_penalty: float
    #: Preferred leaf-option batch pad width for device-local scoring.
    option_batch_hint: int
    #: Whether production action embeds should come from decode_options hidden.
    prefer_option_hidden: bool
    #: Mirror latent-lookahead policy-aid bound when adapting that module.
    policy_aid_cap: float
    notes: str = ""

    def to_config(self, **overrides: object) -> RTPConfig:
        payload = {
            "d_model": self.d_model,
            "dynamics_width": self.dynamics_width,
            "num_plan_candidates": self.num_plan_candidates,
            "max_recursion_depth": self.max_recursion_depth,
            "max_neural_passes": self.max_neural_passes,
            "max_plan_length": self.max_plan_length,
            "complexity_option_threshold": self.complexity_option_threshold,
            "complexity_entropy_threshold": self.complexity_entropy_threshold,
            "online_sim_verify_budget": self.online_sim_verify_budget,
            "repair_budget": self.repair_budget,
            "compute_cost_penalty": self.compute_cost_penalty,
            "sizing_profile": self.name,
            "option_batch_hint": self.option_batch_hint,
            "prefer_option_hidden": self.prefer_option_hidden,
            "policy_aid_cap": self.policy_aid_cap,
        }
        payload.update(overrides)
        return RTPConfig(**payload)  # type: ignore[arg-type]


#: Global TemporalCabtTransformer / Hope-shaped parent (config.ModelConfig).
GLOBAL_TRANSFORMER = RTPSizingProfile(
    name="global_transformer",
    d_model=256,
    # Match ActionConditionedLatentLookahead default width at d=256.
    dynamics_width=512,
    num_plan_candidates=4,
    max_recursion_depth=2,
    # Root + complexity + up to two subgoal passes.
    max_neural_passes=4,
    max_plan_length=12,
    # Shared with mcts.planned_sims / SearchConfig.complex_option_threshold.
    complexity_option_threshold=8,
    complexity_entropy_threshold=1.5,
    online_sim_verify_budget=0,
    repair_budget=1,
    compute_cost_penalty=0.01,
    option_batch_hint=256,
    prefer_option_hidden=True,
    policy_aid_cap=0.25,
    notes=(
        "Attaches to d_model=256 TemporalCabtTransformer; dynamics width "
        "mirrors latent_lookahead_width=512."
    ),
)

#: Pure-RL lean evaluator + V5/V6 matchup-adapter hidden width.
PURE_RL = RTPSizingProfile(
    name="pure_rl",
    d_model=96,
    # Keep the 2x trunk convention used by latent lookahead at d=256→512.
    dynamics_width=192,
    num_plan_candidates=4,
    max_recursion_depth=2,
    max_neural_passes=4,
    max_plan_length=12,
    complexity_option_threshold=8,
    complexity_entropy_threshold=1.5,
    online_sim_verify_budget=0,
    repair_budget=1,
    compute_cost_penalty=0.01,
    # CPU/lean leaf hint; Blackwell can still pack larger externally.
    option_batch_hint=64,
    prefer_option_hidden=True,
    policy_aid_cap=0.25,
    notes=(
        "Attaches to pure_rl model_profile d_model=96 and matchup adapter "
        "hidden dim 96. Prefer archetype adapters for plan scoring later, "
        "not a second full trunk."
    ),
)

#: Tiny deterministic profile for unit tests only.
UNIT_TEST = RTPSizingProfile(
    name="unit_test",
    d_model=16,
    dynamics_width=32,
    num_plan_candidates=3,
    max_recursion_depth=2,
    max_neural_passes=8,
    max_plan_length=8,
    complexity_option_threshold=8,
    complexity_entropy_threshold=1.5,
    online_sim_verify_budget=0,
    repair_budget=1,
    compute_cost_penalty=0.01,
    option_batch_hint=8,
    prefer_option_hidden=False,
    policy_aid_cap=0.25,
    notes="CPU unit-test only; not a deployment profile.",
)

#: Sparse online verification ablation knobs from the architecture note.
VERIFY_ABLATONS: Mapping[str, int] = {
    "latent_only": 0,
    "verify_8": 8,
    "verify_16": 16,
    "verify_32": 32,
    # Existing whole-turn MCTS-style ceiling discussed in the redesign note.
    "legacy_mcts_75": 75,
    # Submission hard cap when search is enabled.
    "submission_max_sims": 50,
}


PROFILES: Mapping[str, RTPSizingProfile] = {
    GLOBAL_TRANSFORMER.name: GLOBAL_TRANSFORMER,
    PURE_RL.name: PURE_RL,
    UNIT_TEST.name: UNIT_TEST,
}


def get_profile(name: str) -> RTPSizingProfile:
    key = str(name).strip().lower()
    if key not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown RTP sizing profile {name!r}; known: {known}")
    return PROFILES[key]


def profile_inventory() -> dict[str, object]:
    return {
        "schema": "poke_bot.recursive_turn_planner.profiles/v1",
        "profiles": {
            name: {
                "d_model": profile.d_model,
                "dynamics_width": profile.dynamics_width,
                "num_plan_candidates": profile.num_plan_candidates,
                "max_neural_passes": profile.max_neural_passes,
                "complexity_option_threshold": profile.complexity_option_threshold,
                "option_batch_hint": profile.option_batch_hint,
                "prefer_option_hidden": profile.prefer_option_hidden,
                "notes": profile.notes,
            }
            for name, profile in PROFILES.items()
        },
        "verify_ablations": dict(VERIFY_ABLATONS),
        "reuse_targets": {
            "latent_lookahead": "poke_bot.action_conditioned_latent_lookahead/v1",
            "complex_option_threshold": "poke_bot.config.SearchConfig",
            "pure_rl_d_model": 96,
            "global_d_model": 256,
            "matchup_adapter_hidden": 96,
            "decision_fusion_route_width": 16,
        },
    }
