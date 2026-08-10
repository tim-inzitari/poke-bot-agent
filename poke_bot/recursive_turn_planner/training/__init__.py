"""Training utilities for the Recursive Turn Planner (shadow / offline)."""

from .checkpoint import (
    RTP_PROMOTION_SCHEMA,
    RTP_SHADOW_TRAIN_SCHEMA,
    load_rtp_checkpoint,
    save_rtp_checkpoint,
)
from .losses import RTPLossBundle, compute_rtp_losses
from .shadow_train import (
    RTPTrainConfig,
    RTPTrainResult,
    make_synthetic_batches,
    train_rtp_shadow,
)

__all__ = [
    "RTP_SHADOW_TRAIN_SCHEMA",
    "RTP_PROMOTION_SCHEMA",
    "RTPLossBundle",
    "RTPTrainConfig",
    "RTPTrainResult",
    "compute_rtp_losses",
    "load_rtp_checkpoint",
    "make_synthetic_batches",
    "save_rtp_checkpoint",
    "train_rtp_shadow",
]
