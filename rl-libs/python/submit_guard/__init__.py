"""One-shot, fail-closed competition submission authorization."""

from .guard import (
    AUTH_SCHEMA,
    ATTEMPT_SCHEMA,
    AuthorizationError,
    SubmitIdentity,
    consume_and_run,
    validate_authorization,
)

__version__ = "0.2.0"

__all__ = [
    "AUTH_SCHEMA",
    "ATTEMPT_SCHEMA",
    "AuthorizationError",
    "SubmitIdentity",
    "consume_and_run",
    "validate_authorization",
    "__version__",
]
