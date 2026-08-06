"""PokeRLM — typed recursive turn planner attachment (experimental).

Defaults preserve current PolicyAgent behavior (``enabled=False`` /
``mode=disabled``). Enable via ``POKEBOT_POKE_RLM_ENABLED`` and
``POKEBOT_POKE_RLM_MODE`` (``shadow`` / ``evaluate`` / ``active``).
"""

from .agent_hooks import PokeRLMAgentBridge, resolve_poke_rlm_config_for_model
from .budget import TurnComputeBudget
from .config import (
    PokeRLMConfig,
    PokeRLMMode,
    PokeRLMProfile,
    config_for_profile,
    load_poke_rlm_config_from_env,
)
from .controller import ControllerResult, PokeRLMController
from .legal_action import LegalAction, resolve_selector
from .model_core import PokeRLMModelCore
from .observation import DeploymentObservation, PrivilegedTargets
from .plan_ir import TurnPlan, round_trip_plan, turn_plan_from_json, turn_plan_to_json
from .reasons import ReasonCode, StopReason
from .specialists import SpecialistAdapterRegistry
from .telemetry import ShadowTrace

__all__ = [
    "ControllerResult",
    "DeploymentObservation",
    "LegalAction",
    "PokeRLMAgentBridge",
    "PokeRLMConfig",
    "PokeRLMController",
    "PokeRLMMode",
    "PokeRLMModelCore",
    "PokeRLMProfile",
    "PrivilegedTargets",
    "ReasonCode",
    "ShadowTrace",
    "SpecialistAdapterRegistry",
    "StopReason",
    "TurnComputeBudget",
    "TurnPlan",
    "config_for_profile",
    "load_poke_rlm_config_from_env",
    "resolve_poke_rlm_config_for_model",
    "resolve_selector",
    "round_trip_plan",
    "turn_plan_from_json",
    "turn_plan_to_json",
]
