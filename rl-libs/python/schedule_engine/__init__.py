"""Generic staged curriculum schedules with digests."""

from .schedule import (
    EpochPlan,
    schedule_digest,
    validate_schedule,
    epoch_plan,
)

__version__ = "0.2.0"

__all__ = [
    "EpochPlan",
    "epoch_plan",
    "schedule_digest",
    "validate_schedule",
    "__version__",
]
