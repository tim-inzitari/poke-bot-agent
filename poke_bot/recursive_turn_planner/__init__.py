"""Lightweight Recursive Turn Planner (experimental).

Constructs typed, conditional full-turn programs from a shared state memory,
refines unresolved subgoals with shallow batched recursion, evaluates
candidates through learned latent transitions, and persists a conditional plan
across atomic actions with sparse repair.

Training lives under ``recursive_turn_planner.training`` (shadow sidecar via
``scripts/train_recursive_turn_planner.py``). Load trained weights with
``POKEBOT_RTP_CHECKPOINT``. Does not rewrite owner-contract authority.

Sizing is profile-bound:
- ``global_transformer`` → d_model=256, dynamics_width=512
- ``pure_rl`` → d_model=96, dynamics_width=192
"""

# Keep every public export lazy.  Besides avoiding the historical PokeRLM/RTP
# training import cycle, this lets the isolated revision-202 tree validator be
# imported on a CPU-only worker without importing Torch, the policy bridge, or
# the legacy executor merely because Python initialized this parent package.
_EXPORT_MODULES = {
    "RTPAgentBridge": "agent_bridge",
    "RTPBridgeDiagnostics": "agent_bridge",
    "resolve_rtp_config_for_model": "agent_bridge",
    "turn_key_from_obs": "agent_bridge",
    "RTP_MAX_AUTHORIZED_NEURAL_PASSES": "config",
    "RTPConfig": "config",
    "LatentTransitionDynamics": "dynamics",
    "LookaheadBackedDynamics": "dynamics",
    "PlanExecutor": "executor",
    "PlanStepResult": "executor",
    "TypedLegalityVerifier": "legality",
    "PersistentTurnMemory": "memory",
    "PlanProposal": "planner",
    "RecursiveTurnPlanner": "planner",
    "RTPNeuralPassBudgetExceeded": "planner",
    "TurnDecision": "planner",
    "required_recursive_passes": "planner",
    "GLOBAL_TRANSFORMER": "profiles",
    "PURE_RL": "profiles",
    "PURE_RL_R197": "profiles",
    "PURE_RL_R197_MAX_ACTION_COMBOS": "profiles",
    "UNIT_TEST": "profiles",
    "VERIFY_ABLATONS": "profiles",
    "get_profile": "profiles",
    "profile_inventory": "profiles",
    "NodeKind": "types",
    "ObservationPredicate": "types",
    "PlanNode": "types",
    "SubgoalKind": "types",
    "TurnProgram": "types",
    "ArchetypeRTPJob": "pipeline",
    "ArchetypeRTPResult": "pipeline",
    "example_registry_jobs": "pipeline",
    "load_archetype_registry": "pipeline",
    "run_archetype_rtp_pipeline": "pipeline",
    "run_registry": "pipeline",
}


def __getattr__(name: str) -> object:
    """Resolve public exports without importing heavyweight modules eagerly."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        from importlib import import_module

        value = getattr(import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GLOBAL_TRANSFORMER",
    "PURE_RL",
    "PURE_RL_R197",
    "PURE_RL_R197_MAX_ACTION_COMBOS",
    "RTP_MAX_AUTHORIZED_NEURAL_PASSES",
    "UNIT_TEST",
    "VERIFY_ABLATONS",
    "ArchetypeRTPJob",
    "ArchetypeRTPResult",
    "LatentTransitionDynamics",
    "LookaheadBackedDynamics",
    "NodeKind",
    "ObservationPredicate",
    "PersistentTurnMemory",
    "PlanExecutor",
    "PlanNode",
    "PlanProposal",
    "PlanStepResult",
    "RTPAgentBridge",
    "RTPBridgeDiagnostics",
    "RTPConfig",
    "RTPNeuralPassBudgetExceeded",
    "RecursiveTurnPlanner",
    "SubgoalKind",
    "TurnDecision",
    "TurnProgram",
    "TypedLegalityVerifier",
    "example_registry_jobs",
    "get_profile",
    "load_archetype_registry",
    "profile_inventory",
    "required_recursive_passes",
    "resolve_rtp_config_for_model",
    "run_archetype_rtp_pipeline",
    "run_registry",
    "turn_key_from_obs",
]
