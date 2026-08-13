"""Public API for the durable-goals protocol resolver."""

from .errors import GoalError, IntegrityError, ResolutionError, ValidationError
from .resolve import Resolution, resolve_gateway
from .writer import (
    activate_amendment,
    chain_goal,
    initialize_goal_package,
    materialize_status,
    record_amendment,
)

__all__ = [
    "GoalError",
    "IntegrityError",
    "Resolution",
    "ResolutionError",
    "ValidationError",
    "activate_amendment",
    "chain_goal",
    "initialize_goal_package",
    "materialize_status",
    "record_amendment",
    "resolve_gateway",
]

__version__ = "0.1.0"
