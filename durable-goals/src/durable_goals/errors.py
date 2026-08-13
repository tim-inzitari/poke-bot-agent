class GoalError(Exception):
    """Base error for deterministic protocol failures."""


class ValidationError(GoalError):
    """A protocol document is malformed or internally inconsistent."""


class IntegrityError(GoalError):
    """A referenced document or evidence payload failed integrity checks."""


class ResolutionError(GoalError):
    """The goal cannot be resolved without guessing."""
