"""Structured validation / fallback / repair reason codes."""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    OK = "ok"
    DISABLED = "disabled"
    SHADOW_ONLY = "shadow_only"
    HIDDEN_FIELD_REJECTED = "hidden_field_rejected"
    ILLEGAL_OPTION_INDEX = "illegal_option_index"
    SELECTOR_UNRESOLVED = "selector_unresolved"
    SELECTOR_AMBIGUOUS = "selector_ambiguous"
    STALE_OPTION_INDEX = "stale_option_index"
    BUDGET_MODEL_CALLS = "budget_model_calls"
    BUDGET_SIMULATOR_CALLS = "budget_simulator_calls"
    BUDGET_NODES = "budget_nodes"
    BUDGET_DEPTH = "budget_depth"
    BUDGET_DEADLINE = "budget_deadline"
    INVALID_PLAN_SCHEMA = "invalid_plan_schema"
    INVALID_PLAN_STATIC = "invalid_plan_static"
    BRANCH_PREDICATE_HIDDEN = "branch_predicate_hidden"
    RESOURCE_LEDGER_VIOLATION = "resource_ledger_violation"
    PLAN_DIVERGENCE = "plan_divergence"
    REPAIR_EXHAUSTED = "repair_exhausted"
    FALLBACK_POLICY = "fallback_policy"
    TRIVIAL_DIRECT = "trivial_direct"
    EMPTY_LEGAL = "empty_legal"
    ENCODE_FAILURE = "encode_failure"


class StopReason(str, Enum):
    """Stop reasons aligned with ``schemas/turn_plan.schema.json``."""

    PLAN_COMPLETE = "PLAN_COMPLETE"
    ATTACK_DECLARED = "ATTACK_DECLARED"
    END_TURN = "END_TURN"
    NO_SAFE_CONTINUATION = "NO_SAFE_CONTINUATION"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FALLBACK_REQUIRED = "FALLBACK_REQUIRED"

    # Compatibility aliases used by earlier experiment code.
    OBJECTIVE_MET = "PLAN_COMPLETE"
    BUDGET = "BUDGET_EXHAUSTED"
    FALLBACK = "FALLBACK_REQUIRED"
    UNSAFE = "NO_SAFE_CONTINUATION"
