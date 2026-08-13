"""Public API for the durable-goals protocol resolver."""

from .errors import GoalError, IntegrityError, ResolutionError, ValidationError
from .resolve import Resolution, resolve_gateway
from .writer import (
    activate_amendment,
    chain_goal,
    initialize_goal_package,
    materialize_status,
    record_amendment,
    record_evidence,
)
from .workflow import (
    WorkflowResolution,
    add_dependency,
    add_goal_node,
    claim_next_prompt,
    initialize_workflow,
    next_prompts,
    remove_dependency,
    remove_goal_node,
    release_claim,
    resolve_workflow,
)

__all__ = [
    "GoalError",
    "IntegrityError",
    "Resolution",
    "ResolutionError",
    "ValidationError",
    "WorkflowResolution",
    "add_dependency",
    "add_goal_node",
    "claim_next_prompt",
    "activate_amendment",
    "chain_goal",
    "initialize_goal_package",
    "initialize_workflow",
    "materialize_status",
    "record_amendment",
    "record_evidence",
    "remove_dependency",
    "remove_goal_node",
    "release_claim",
    "resolve_gateway",
    "resolve_workflow",
    "next_prompts",
]

__version__ = "0.1.0"
