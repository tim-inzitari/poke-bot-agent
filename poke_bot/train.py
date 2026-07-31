"""Supervised bootstrap / policy-value training with realized histories.

Every decision is trained causally on the same acting-seat observation history
that trusted serving consumes incrementally.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from . import archetypes, checkpoint, config, device as device_mod, features
from .aux_label_contract import validated_unique_card_ids
from .blackwell_heads import (
    BLACKWELL_STRATEGY_HEAD_PREFIXES,
    lethal_target_from_aux,
    masked_bce_logit,
    masked_smooth_l1,
    prize_race_target_from_aux,
)
from .dataset import BootstrapDataset, GameSequence, PolicyStage
from .device_corpus import DEFAULT_MIN_FREE_GIB, DeviceResidentBootstrapCorpus
from .combo_state import (
    COMBO_STATE_KEY,
    VECTOR_WIDTH as COMBO_STATE_VECTOR_WIDTH,
    combo_state_loss,
    validate_combo_state_labels,
)
from .matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as MATCHUP_ADAPTER_V5_FORMAT,
)
from .matchup_adapters import HIDDEN_DIM as MATCHUP_ADAPTER_HIDDEN_DIM
from .matchup_adapters import EXPERT_IDS, UNKNOWN_ROUTE
from .matchup_adapter_activation import (
    ActivationReceipt,
    adapter_training_ticket,
    training_routes_for_sequence,
    validate_adapter_training_authorization,
)
from .model import (
    COMBO_STATE_HEAD_NAME,
    DECISION_FUSION_KEY_PREFIX,
    DECISION_FUSION_SCHEMA,
    DECISION_FUSION_V2_SCHEMA,
    EXPANDED_HEAD_KEY_PREFIXES,
    EXPANDED_HEAD_NAMES,
    EXPANDED_HEAD_SCHEMA,
    SETUP_BOARD_OUTCOME_HEAD_NAME,
    TemporalCabtTransformer,
    build_model,
)
from .strategic_losses import (
    GUIDE_OUTCOME_BACKED_HEAD_IDS,
    canonical_expanded_loss_weights,
    expanded_strategic_losses,
    guide_outcome_backed_loss_weights,
    resident_expanded_strategic_losses,
)
from .setup_board_outcome import setup_board_outcome_loss
from .strategic_heads import (
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    TARGET_SCHEMA_DIGEST,
    validate_expanded_strategic_labels,
)
from .strategic_schedule import EXPANDED_HEAD_IDS, EXPANDED_SCHEDULE_SCHEMA

# Distinct named belief + Blackwell strategy heads — warm-start may omit only
# these key prefixes (Scope A particle priors + Scope B Hammer heads).
BELIEF_AUX_HEAD_KEY_PREFIXES: tuple[str, ...] = (
    "opp_hand_head.",
    "opp_remainder_head.",
) + BLACKWELL_STRATEGY_HEAD_PREFIXES

GUIDE_TRAINING_MODE_LEGACY = "legacy_policy_ce_v1"
GUIDE_TRAINING_MODE_STRATEGIC = "strategic_curriculum_v1"
GUIDE_TRAINING_MODES = frozenset(
    {GUIDE_TRAINING_MODE_LEGACY, GUIDE_TRAINING_MODE_STRATEGIC}
)
SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT = 0.025
COMBO_STATE_BASE_LOSS_WEIGHT = 0.025


@dataclass
class TrainConfig:
    """Bootstrap supervised training knobs."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    games_per_batch: int = 4
    max_decisions_per_batch: int = 256
    val_frac: float = 0.1
    #: Keep both acting-seat records from an episode in the same split.
    split_by_episode: bool = False
    early_stop_patience: int = 5
    value_loss_weight: float = 1.0
    aux_loss_weight: float = 0.1
    opp_hand_loss_weight: float = 0.2
    opp_remainder_loss_weight: float = 0.15
    #: Training-only soft strategy guide. This must remain zero for core and
    #: non-Alakazam runs; the specialist launcher owns the explicit opt-in.
    alakazam_guide_loss_weight: float = 0.0
    #: Already-started runs retain direct policy CE. Future runs beginning
    #: with Archaludon use confidence only to scale observed-target losses.
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY
    #: Ordinary observed-target weight for the future setup/bench branch.
    setup_board_outcome_loss_weight: float = (
        SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT
    )
    #: Slowking-only observed causal combo targets. Non-Slowking runs leave
    #: this zero, preserving their forward path and optimizer exactly.
    combo_state_loss_weight: float = 0.0
    #: Checksum-gated future curriculum artifacts, recorded in checkpoints.
    current_deck_guide_curriculum_spec: str = ""
    current_deck_guide_head_role_map: str = ""
    current_deck_guide_curriculum_validation_receipt: str = ""
    #: Scope B (Blackwell Hammer) only — keep 0.0 for core / generic bootstrap.
    lethal_threat_loss_weight: float = 0.0
    prize_race_loss_weight: float = 0.0
    #: Additive V6 auxiliary weights keyed by canonical expanded head id.
    #: Empty/all-zero preserves the exact V5 optimizer and forward path.
    expanded_head_loss_weights: dict[str, float] = field(default_factory=dict)
    #: One clean-boundary, receipt-backed loss rebalance may change inherited
    #: expanded-head multipliers without changing tensors or optimizer state.
    expanded_head_weight_migration_reason: str = ""
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 0
    log_every: int = 1
    #: When True, use AWR on selected actions (never CE to soft behavior π).
    pure_rl: bool = False
    awr_beta: float = 0.5
    awr_weight_max: float = 20.0
    #: Whitening advantages before exp(A/β) (pure_rl only).
    awr_normalize_advantages: bool = True
    #: Precompute V(s) once at the iteration boundary and reuse it for every
    #: AWR update. This makes the documented "stale critic" baseline real
    #: instead of allowing it to drift with batch order.
    awr_freeze_baseline: bool = False
    #: Compute the frozen AWR baseline from V(s) only.  The optimized path
    #: deliberately bypasses policy option decoding, decision fusion, guide
    #: scoring, belief/strategic heads, log-softmax, and AWR diagnostics.
    awr_value_only_baseline: bool = True
    #: Multi-game padded temporal packing remains separately gated because
    #: different attention batch geometry may change the last FP32 bits.  The
    #: value-only path is active independently; packing activates only after an
    #: exact device-specific cache and AWR parity receipt.
    awr_pack_temporal_baseline: bool = False
    #: One-time activation gate: run the reference path on the same immutable
    #: rows and reject the candidate unless every cache key and float matches.
    awr_baseline_exact_parity_check: bool = False
    #: Prepare at most one subsequent baseline batch on a CPU thread while the
    #: current packed temporal batch is evaluated on the learner device.
    awr_baseline_prefetch_batches: int = 1
    #: Subtract from policy loss: ``entropy_bonus * H(π)`` (pure_rl only).
    entropy_bonus: float = 0.01
    #: Shadow-study diagnostic: retain one epoch of scalar AWR weights so the
    #: reported p50/p95 are exact global quantiles, not averaged batch
    #: quantiles. Disabled in production to avoid unnecessary host objects.
    capture_awr_weight_distribution: bool = False
    #: Temporal hot-start only: keep the new history state close to the copied
    #: stateless parent's normalized state during frozen-trunk calibration.
    history_identity_loss_weight: float = 0.0
    #: Explicit oracle-routed bootstrap mode.  This freezes the complete base
    #: model and optimizes only the dormant matchup adapter bank.
    matchup_adapter_training: bool = False
    #: Immutable post-iteration-15 boundary proof.  Adapter optimization is
    #: impossible without a receipt pinned to the exact initialization ckpt.
    matchup_adapter_activation_receipt: str = ""
    #: Optional second, behavior-inert optimizer phase after each ordinary RL
    #: fit. Only oracle-ticketed matchup rows participate; the base learner is
    #: bit-frozen and the serialized/runtime adapter switch remains disabled.
    dormant_matchup_adapter_epochs: int = 0
    dormant_matchup_adapter_lr: float = 1e-4
    dormant_matchup_adapter_activation_receipt: str = ""
    #: Adapter-only option decoding has a different peak-memory curve from the
    #: ordinary learner.  Never inherit a large H10 learner cap blindly.
    dormant_matchup_adapter_max_decisions_per_batch: int = 2048

    @classmethod
    def pure_rl_defaults(cls, **overrides: Any) -> "TrainConfig":
        """Single-model pure-RL knobs: AWR on, all aux/strategy weights off.

        Two measured passes over each fresh replay window are the production
        default. Callers may override this for controlled experiments.
        """
        cfg = cls(
            pure_rl=True,
            aux_loss_weight=0.0,
            opp_hand_loss_weight=0.0,
            opp_remainder_loss_weight=0.0,
            alakazam_guide_loss_weight=0.0,
            lethal_threat_loss_weight=0.0,
            prize_race_loss_weight=0.0,
            early_stop_patience=3,
            epochs=2,
            # Larger than the tiny default=4 so multi-k game shards finish under
            # the collection wall time without inflating optimizer-step count.
            games_per_batch=16,
            max_decisions_per_batch=2048,
            awr_freeze_baseline=True,
            awr_normalize_advantages=bool(
                getattr(config.PURE_RL, "normalize_advantages", True)
            ),
            entropy_bonus=float(getattr(config.PURE_RL, "entropy_bonus", 0.01)),
            awr_beta=float(getattr(config.PURE_RL, "awr_beta", 0.5)),
            awr_weight_max=float(getattr(config.PURE_RL, "awr_weight_max", 20.0)),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown TrainConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass
class BatchMetrics:
    policy_loss: float = 0.0
    teacher_policy_loss: float = 0.0
    value_loss: float = 0.0
    aux_loss: float = 0.0
    opp_hand_loss: float = 0.0
    opp_remainder_loss: float = 0.0
    alakazam_guide_loss: float = 0.0
    guide_strategic_curriculum_loss: float = 0.0
    setup_board_outcome_loss: float = 0.0
    combo_state_loss: float = 0.0
    lethal_threat_loss: float = 0.0
    prize_race_loss: float = 0.0
    history_identity_loss: float = 0.0
    total_loss: float = 0.0
    policy_acc: float = 0.0
    policy_kl: float = 0.0
    target_value_mean: float = 0.0
    value_pred_mean: float = 0.0
    n_decisions: int = 0
    n_games: int = 0
    n_alakazam_guide_rows: int = 0
    n_archetype_rows: int = 0
    n_opp_hand_rows: int = 0
    n_opp_remainder_rows: int = 0
    n_lethal_threat_rows: int = 0
    n_prize_race_rows: int = 0
    n_matchup_adapter_rows: int = 0
    n_teacher_policy_rows: int = 0
    expanded_head_metrics: dict[str, Any] = field(default_factory=dict)
    guide_curriculum_head_metrics: dict[str, Any] = field(default_factory=dict)
    setup_board_outcome_metrics: dict[str, Any] = field(default_factory=dict)
    combo_state_metrics: dict[str, Any] = field(default_factory=dict)
    # Backward-compatible pipeline signal consumed by pure_rl.aborts. It is
    # deliberately the raw mean absolute advantage, not a signed/whitened mean.
    mean_advantage: float = 0.0
    raw_advantage_mean: float = 0.0
    raw_advantage_std: float = 0.0
    raw_advantage_mean_abs: float = 0.0
    raw_advantage_mean_sq: float = 0.0
    normalized_advantage_mean: float = 0.0
    normalized_advantage_std: float = 0.0
    normalized_advantage_mean_abs: float = 0.0
    normalized_advantage_mean_sq: float = 0.0
    awr_weight_mean: float = 0.0
    awr_weight_sum: float = 0.0
    awr_weight_sq_sum: float = 0.0
    awr_weight_p50: float = 0.0
    awr_weight_p95: float = 0.0
    awr_weight_max_observed: float = 0.0
    awr_weight_clip_frac: float = 0.0
    awr_effective_sample_size: float = 0.0
    awr_effective_sample_fraction: float = 0.0
    policy_selected_nll: float = 0.0


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    patience_left: int = 5
    history: list[dict[str, Any]] = field(default_factory=list)


def _archetype_label(name: str) -> Optional[int]:
    """Return a valid auxiliary class, or ``None`` for unknown baseline ids."""
    ids = list(archetypes.archetype_ids())
    return ids.index(name) if name in ids else None


def is_allowed_missing_belief_head_key(key: str) -> bool:
    """True iff ``key`` is an expected new belief-aux head parameter."""
    return any(key.startswith(prefix) for prefix in BELIEF_AUX_HEAD_KEY_PREFIXES)


def belief_head_names_from_state_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Map state-dict keys → distinct head module names (sorted unique)."""
    names: set[str] = set()
    for key in keys:
        if is_allowed_missing_belief_head_key(key):
            names.add(key.split(".", 1)[0])
    return tuple(sorted(names))


def is_allowed_missing_expanded_head_key(key: str) -> bool:
    """Return whether ``key`` belongs to the opt-in V6 strategic head bank."""

    return any(key.startswith(prefix) for prefix in EXPANDED_HEAD_KEY_PREFIXES)


def expanded_head_names_from_state_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Map missing state keys to deterministic expanded-head module names."""

    return tuple(
        sorted(
            {
                key.split(".", 1)[0]
                for key in keys
                if is_allowed_missing_expanded_head_key(key)
            }
        )
    )


def is_allowed_missing_decision_fusion_key(key: str) -> bool:
    """Return whether ``key`` belongs to the opt-in causal fusion module."""

    return key.startswith(DECISION_FUSION_KEY_PREFIX)


def belief_card_vocab_from_state(state: dict[str, Any]) -> int:
    """Resolve the belief-card vocabulary for old or current checkpoints.

    Legacy policy checkpoints legitimately predate the opponent hand and
    hidden-remainder heads.  In that case the same live card vocabulary used
    by :func:`load_model_from_checkpoint` is authoritative.  A partially
    upgraded checkpoint may provide either head, but any present tensor must
    be a valid linear weight and the two output widths must agree.
    """

    widths: dict[str, int] = {}
    for name in ("opp_hand_head", "opp_remainder_head"):
        weight = state.get(f"{name}.weight")
        if weight is None:
            continue
        if getattr(weight, "ndim", 0) != 2:
            raise ValueError(f"checkpoint {name} weight must be rank-2")
        width = int(weight.shape[0])
        if width <= 0:
            raise ValueError(f"checkpoint {name} vocabulary must be positive")
        widths[name] = width
    if len(set(widths.values())) > 1:
        raise ValueError("checkpoint opponent belief head vocabularies disagree")
    return next(iter(widths.values()), int(features.card_vocab_size()))


def expand_aux_head_to_current_registry(model: torch.nn.Module) -> bool:
    """Expand a compatible historical archetype head by stable row identity."""

    target_ids = list(archetypes.archetype_ids())
    target_classes = len(target_ids) + 1
    old = model.aux_head[-1]
    if not isinstance(old, torch.nn.Linear):
        raise TypeError("aux_head final module must be Linear")
    if old.out_features == target_classes:
        return False
    compatible_orders = (
        archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS,
        archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS,
        archetypes.LEGACY_AUX_ARCHETYPE_IDS,
    )
    legacy = next(
        (
            list(order)
            for order in compatible_orders
            if old.out_features == len(order) + 1
        ),
        None,
    )
    if legacy is None:
        raise RuntimeError(
            f"cannot expand unexpected aux head with {old.out_features} classes"
        )
    new = torch.nn.Linear(
        old.in_features,
        target_classes,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        for old_i, name in enumerate(legacy):
            new_i = target_ids.index(name)
            new.weight[new_i].copy_(old.weight[old_i])
            if new.bias is not None and old.bias is not None:
                new.bias[new_i].copy_(old.bias[old_i])
        # Preserve the historical unknown/fallback row at the new final row.
        new.weight[-1].copy_(old.weight[-1])
        if new.bias is not None and old.bias is not None:
            new.bias[-1].copy_(old.bias[-1])
    model.aux_head[-1] = new
    return True


MATCHUP_ADAPTER_PARAMETER_PREFIX = "matchup_adapter_bank."


def matchup_adapter_base_state(
    model: TemporalCabtTransformer,
) -> dict[str, torch.Tensor]:
    """Clone all non-adapter model state for bit-exact isolation checks."""

    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(MATCHUP_ADAPTER_PARAMETER_PREFIX)
    }


def assert_matchup_adapter_parent_identity(
    model: TemporalCabtTransformer,
    *,
    parent_checkpoint: Union[str, Path],
) -> None:
    """Prove every non-adapter tensor still equals the receipt-pinned parent."""

    parent_payload = checkpoint.load_checkpoint(parent_checkpoint, map_location="cpu")
    raw_parent = dict(parent_payload.get("model_state_dict") or {})
    parent = {
        name: value.detach().cpu()
        for name, value in raw_parent.items()
        if not name.startswith(MATCHUP_ADAPTER_PARAMETER_PREFIX)
    }
    current = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith(MATCHUP_ADAPTER_PARAMETER_PREFIX)
    }
    if parent.keys() != current.keys():
        missing = sorted(parent.keys() - current.keys())
        extra = sorted(current.keys() - parent.keys())
        raise AssertionError(
            "adapter checkpoint base keys differ from pinned parent: "
            f"missing={missing[:4]} extra={extra[:4]}"
        )
    changed = [name for name in parent if not torch.equal(parent[name], current[name])]
    if changed:
        raise AssertionError(
            "adapter checkpoint changed receipt-pinned base tensors: "
            f"{changed[:5]}"
        )


def assert_matchup_adapter_training_contract(
    model: TemporalCabtTransformer,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    base_state: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    """Assert frozen-base gradients, optimizer separation, and base identity."""

    adapter_parameters = list(model.matchup_adapter_bank.parameters())
    activation = getattr(model, "_matchup_adapter_activation_receipt", None)
    if not isinstance(activation, ActivationReceipt):
        raise AssertionError(
            "matchup adapter training requires a validated iteration-15 "
            "activation receipt"
        )
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(MATCHUP_ADAPTER_PARAMETER_PREFIX)
    ]
    if any(parameter.requires_grad for parameter in base_parameters):
        raise AssertionError("matchup adapter training requires a fully frozen base")
    if any(parameter.grad is not None for parameter in base_parameters):
        raise AssertionError("frozen base parameter received a gradient")
    if not adapter_parameters or not all(
        parameter.requires_grad for parameter in adapter_parameters
    ):
        raise AssertionError("all matchup adapter parameters must be trainable")

    if optimizer is not None:
        optimized = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimized_ids = {id(parameter) for parameter in optimized}
        if len(optimized) != len(adapter_parameters) or optimized_ids != adapter_ids:
            raise AssertionError(
                "matchup adapter optimizer must contain only the adapter bank"
            )

    if base_state is not None:
        current = {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith(MATCHUP_ADAPTER_PARAMETER_PREFIX)
        }
        if current.keys() != base_state.keys():
            raise AssertionError("frozen base state keys changed during adapter training")
        changed = [
            name
            for name, value in current.items()
            if not torch.equal(value, base_state[name])
        ]
        if changed:
            raise AssertionError(
                f"frozen base state changed during adapter training: {changed[:5]}"
            )


def build_matchup_adapter_optimizer(
    model: TemporalCabtTransformer,
    *,
    lr: float,
    weight_decay: float,
    activation_receipt: ActivationReceipt,
) -> torch.optim.AdamW:
    """Freeze the base and build an AdamW over only the matchup adapters."""

    if int(model.d_model) != MATCHUP_ADAPTER_HIDDEN_DIM:
        raise ValueError(
            "matchup adapter training requires a 96-dimensional temporal state, "
            f"got d_model={model.d_model}"
        )
    if not isinstance(activation_receipt, ActivationReceipt):
        raise ValueError(
            "matchup adapter optimizer requires a validated boundary receipt"
        )
    model._matchup_adapter_activation_receipt = activation_receipt
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    # Training uses an explicit per-call override.  Keep the serialized/runtime
    # activation flag dormant so adapter fitting cannot silently activate it.
    model.matchup_adapter_bank.enabled = False
    model.cfg.matchup_adapters_enabled = False
    model.matchup_adapter_bank.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.matchup_adapter_bank.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    assert_matchup_adapter_training_contract(model, optimizer=optimizer)
    return optimizer


def load_append_only_matchup_adapter_optimizer_state(
    optimizer: torch.optim.Optimizer,
    prior_state: dict[str, Any],
) -> int:
    """Restore Adam state only for the exact current adapter architecture.

    V5 and fixed-capacity V6 both use four parameters per physical slot. Any
    historical or partially migrated optimizer whose parameter-group length
    differs from the instantiated bank fails closed. The explicit V5-to-V6
    migrator expands the group before this loader is allowed to restore it.
    """

    if len(optimizer.param_groups) != 1:
        raise ValueError("matchup adapter optimizer must have one parameter group")
    saved_groups = list(prior_state.get("param_groups") or [])
    if len(saved_groups) != 1:
        raise ValueError("saved matchup adapter optimizer must have one group")
    current = optimizer.state_dict()
    current_params = list(current["param_groups"][0]["params"])
    saved_params = list(saved_groups[0].get("params") or [])
    if (
        not saved_params
        or len(saved_params) != len(current_params)
        or len(saved_params) % 4 != 0
        or len(current_params) % 4 != 0
    ):
        raise ValueError(
            "saved adapter optimizer does not match the current physical bank"
        )
    saved_slots = dict(prior_state.get("state") or {})
    unknown_slots = set(saved_slots) - set(saved_params)
    if unknown_slots:
        raise ValueError("saved adapter optimizer contains unknown parameter slots")
    migrated_group = copy.deepcopy(saved_groups[0])
    migrated_group["params"] = current_params
    migrated_state = {
        current_params[index]: _clone_optimizer_state(saved_slots[saved_id])
        for index, saved_id in enumerate(saved_params)
        if saved_id in saved_slots
    }
    optimizer.load_state_dict(
        {"state": migrated_state, "param_groups": [migrated_group]}
    )
    return len(saved_params) // 4


@dataclass
class MatchupAdapterIsolationGuard:
    """Per-step proof that routes absent from a batch remain untouched."""

    active_routes: frozenset[int]
    inactive_parameters: dict[str, torch.Tensor]
    inactive_optimizer_state: dict[str, dict[str, Any]]


def _clone_optimizer_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_state(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_optimizer_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_optimizer_state_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def prepare_matchup_adapter_isolation_guard(
    model: TemporalCabtTransformer,
    optimizer: torch.optim.Optimizer,
    sequences: Sequence[GameSequence],
) -> MatchupAdapterIsolationGuard:
    """Snapshot absent experts immediately before one adapter-only step."""

    active_routes: set[int] = set()
    for sequence in sequences:
        active_routes.update(training_routes_for_sequence(sequence))
    if not active_routes or any(
        route < 0 or route >= len(model.matchup_adapter_bank.experts)
        for route in active_routes
    ):
        raise RuntimeError("adapter batch has no valid oracle-ticketed route")

    inactive_parameters: dict[str, torch.Tensor] = {}
    inactive_optimizer_state: dict[str, dict[str, Any]] = {}
    for route, expert in enumerate(model.matchup_adapter_bank.experts):
        if route in active_routes:
            continue
        for local_name, parameter in expert.named_parameters():
            name = f"experts.{route}.{local_name}"
            if parameter.grad is not None:
                raise AssertionError(
                    f"inactive adapter parameter retained a gradient before step: {name}"
                )
            inactive_parameters[name] = parameter.detach().clone()
            inactive_optimizer_state[name] = _clone_optimizer_state(
                optimizer.state.get(parameter, {})
            )
    return MatchupAdapterIsolationGuard(
        active_routes=frozenset(active_routes),
        inactive_parameters=inactive_parameters,
        inactive_optimizer_state=inactive_optimizer_state,
    )


def prepare_matchup_adapter_route_isolation_guard(
    model: TemporalCabtTransformer,
    optimizer: torch.optim.Optimizer,
    route: int,
) -> MatchupAdapterIsolationGuard:
    """Snapshot every expert except one constant resident-corpus route."""

    if type(route) is not int or route < 0 or route >= len(EXPERT_IDS):
        raise RuntimeError("resident adapter batch has no valid route")
    inactive_parameters: dict[str, torch.Tensor] = {}
    inactive_optimizer_state: dict[str, dict[str, Any]] = {}
    for candidate, expert in enumerate(model.matchup_adapter_bank.experts):
        if candidate == route:
            continue
        for local_name, parameter in expert.named_parameters():
            name = f"experts.{candidate}.{local_name}"
            if parameter.grad is not None:
                raise AssertionError(
                    f"inactive adapter parameter retained a gradient before step: {name}"
                )
            inactive_parameters[name] = parameter.detach().clone()
            inactive_optimizer_state[name] = _clone_optimizer_state(
                optimizer.state.get(parameter, {})
            )
    return MatchupAdapterIsolationGuard(
        active_routes=frozenset({route}),
        inactive_parameters=inactive_parameters,
        inactive_optimizer_state=inactive_optimizer_state,
    )


def assert_matchup_adapter_isolation_guard(
    model: TemporalCabtTransformer,
    optimizer: torch.optim.Optimizer,
    guard: MatchupAdapterIsolationGuard,
    *,
    after_step: bool,
) -> None:
    """Fail if an absent route receives a gradient, update, or Adam state."""

    current = dict(model.matchup_adapter_bank.named_parameters())
    for name, before in guard.inactive_parameters.items():
        parameter = current[name]
        if parameter.grad is not None:
            raise AssertionError(
                f"inactive adapter route received a gradient: {name}"
            )
        if not after_step:
            continue
        if not torch.equal(parameter.detach(), before):
            raise AssertionError(f"inactive adapter route changed: {name}")
        previous_state = guard.inactive_optimizer_state[name]
        current_state = optimizer.state.get(parameter, {})
        if not _optimizer_state_equal(previous_state, current_state):
            raise AssertionError(
                f"inactive adapter optimizer state changed: {name}"
            )


def masked_belief_card_bce(
    logits: torch.Tensor,
    multilabel: Optional[torch.Tensor],
) -> torch.Tensor:
    """BCE with logits; returns a zero scalar when labels are absent (masked)."""
    if multilabel is None:
        return logits.sum() * 0.0
    if multilabel.shape != logits.shape:
        raise ValueError(
            "belief card multilabel shape mismatch: "
            f"logits={tuple(logits.shape)} labels={tuple(multilabel.shape)}"
        )
    return F.binary_cross_entropy_with_logits(logits, multilabel)


def belief_multihots_from_aux_labels(
    aux_labels: dict[str, Any],
    card_vocab: int,
    *,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build hand / remainder multi-hots from privileged remask aux labels.

    Remainder = hand ∪ deck-order ∪ privileged prize dump when present.
    Either tensor may be ``None`` so the corresponding BCE term masks cleanly.
    """
    hand_ids = validated_unique_card_ids(
        aux_labels.get("opp_hand"), card_vocab, field_name="opp_hand"
    )
    deck_ids = validated_unique_card_ids(
        aux_labels.get("opp_deck_order"),
        card_vocab,
        field_name="opp_deck_order",
    )
    prize_ids = validated_unique_card_ids(
        aux_labels.get("opp_prizes"), card_vocab, field_name="opp_prizes"
    )
    exact_remainder_ids = validated_unique_card_ids(
        aux_labels.get("opp_hidden_remainder"),
        card_vocab,
        field_name="opp_hidden_remainder",
    )
    hand_mh: Optional[torch.Tensor] = None
    if hand_ids is not None:
        hand_mh = torch.zeros(card_vocab, device=device)
        for card_id in hand_ids:
            hand_mh[int(card_id)] = 1.0
    rem_ids: Optional[list[int]] = None
    if exact_remainder_ids is not None:
        rem_ids = list(exact_remainder_ids)
    elif hand_ids is not None or deck_ids is not None or prize_ids is not None:
        rem_ids = []
        for src in (hand_ids, deck_ids, prize_ids):
            if src:
                rem_ids.extend(src)
    rem_mh: Optional[torch.Tensor] = None
    if rem_ids is not None:
        rem_mh = torch.zeros(card_vocab, device=device)
        for card_id in rem_ids:
            rem_mh[int(card_id)] = 1.0
    return hand_mh, rem_mh


def masked_alakazam_guide_ce(
    model_log_probs: torch.Tensor,
    guide_target_indices: Sequence[int],
    guide_confidences: Sequence[float],
    n_options: Sequence[int],
) -> tuple[torch.Tensor, int]:
    """Confidence-weighted CE for collapsed advisory guide targets."""
    if model_log_probs.dim() != 2:
        raise ValueError("guide CE expects [rows, max_options] model log-probs")
    rows = int(model_log_probs.size(0))
    if not (
        len(guide_target_indices) == len(guide_confidences) == len(n_options) == rows
    ):
        raise ValueError("guide rows do not align with policy rows")

    losses: list[torch.Tensor] = []
    for row, (raw_target, raw_confidence, raw_n) in enumerate(
        zip(guide_target_indices, guide_confidences, n_options)
    ):
        n = int(raw_n)
        target = int(raw_target)
        confidence = float(raw_confidence)
        if target < 0 or confidence <= 0.0:
            continue
        if n < 2 or n > int(model_log_probs.size(1)):
            continue
        if target >= n:
            raise ValueError(
                f"guide target is outside row {row}: target={target} options={n}"
            )
        bounded_confidence = min(1.0, max(0.0, confidence))
        losses.append(-model_log_probs[row, target] * bounded_confidence)

    if not losses:
        return model_log_probs.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def canonical_guide_training_mode(value: str) -> str:
    """Normalize the explicit guide semantics without guessing from weights."""

    mode = str(value or "").strip() or GUIDE_TRAINING_MODE_LEGACY
    if mode not in GUIDE_TRAINING_MODES:
        raise ValueError(f"unknown current-deck guide training mode: {mode!r}")
    return mode


def assert_strategic_curriculum_model_contract(
    model: TemporalCabtTransformer,
    *,
    setup_board_outcome_loss_weight: float,
) -> None:
    """Fail before a future curriculum can silently use the legacy model path."""

    if not math.isfinite(float(setup_board_outcome_loss_weight)) or not math.isclose(
        float(setup_board_outcome_loss_weight),
        SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "strategic curriculum requires setup-board base loss weight 0.025"
        )
    if str(getattr(model, "decision_context", "")) != "history":
        raise ValueError(
            "strategic curriculum requires temporal history decision context"
        )
    required_flags = {
        "expanded_heads_enabled": bool(
            getattr(model, "expanded_heads_enabled", False)
        ),
        "setup_board_outcome_head_enabled": bool(
            getattr(model, "setup_board_outcome_head_enabled", False)
        ),
        "decision_fusion_enabled": bool(
            getattr(model, "decision_fusion_enabled", False)
        ),
        "decision_fusion_dedicated_routes_enabled": bool(
            getattr(model, "decision_fusion_dedicated_routes_enabled", False)
        ),
    }
    missing = sorted(name for name, enabled in required_flags.items() if not enabled)
    if missing:
        raise ValueError(
            "strategic curriculum model contract is incomplete: "
            f"missing={missing}"
        )


def assert_strategic_curriculum_receipt_contract(
    *,
    specialist_id: str,
    curriculum_spec: str,
    head_role_map: str,
    validation_receipt: str,
) -> None:
    """Require checksum-linked immutable artifacts at every training entry."""

    paths = {
        "curriculum_spec": Path(curriculum_spec).expanduser().resolve(),
        "head_role_map": Path(head_role_map).expanduser().resolve(),
        "validation_receipt": Path(validation_receipt).expanduser().resolve(),
    }
    if any(not value.is_file() for value in paths.values()):
        missing = sorted(
            name for name, value in paths.items() if not value.is_file()
        )
        raise ValueError(
            "strategic curriculum receipt gate is incomplete: "
            f"missing={missing}"
        )
    try:
        spec = json.loads(paths["curriculum_spec"].read_text(encoding="utf-8"))
        role_map = json.loads(paths["head_role_map"].read_text(encoding="utf-8"))
        receipt = json.loads(
            paths["validation_receipt"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("strategic curriculum receipt gate is unreadable") from exc
    specialist = str(specialist_id or "").strip().casefold()
    if (
        not isinstance(spec, dict)
        or spec.get("schema")
        != "poke_bot.future_specialist_strategic_curriculum/v1"
        or spec.get("training_mode") != GUIDE_TRAINING_MODE_STRATEGIC
        or str(spec.get("specialist_id") or "").strip().casefold()
        != specialist
        or spec.get("direct_policy_cross_entropy_allowed") is not False
        or spec.get("replace_observed_outcome_targets_allowed") is not False
        or not isinstance(role_map, dict)
        or role_map.get("schema")
        != "poke_bot.future_specialist_strategic_head_roles/v1"
        or str(role_map.get("specialist_id") or "").strip().casefold()
        != specialist
        or not isinstance(receipt, dict)
        or receipt.get("schema")
        != "poke_bot.future_specialist_strategic_curriculum_validation/v1"
        or receipt.get("status") != "validated"
        or receipt.get("training_mode") != GUIDE_TRAINING_MODE_STRATEGIC
        or str(receipt.get("specialist_id") or "").strip().casefold()
        != specialist
    ):
        raise ValueError("strategic curriculum receipt gate is invalid")

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if (
        str(receipt.get("curriculum_spec_sha256") or "")
        != digest(paths["curriculum_spec"])
        or str(receipt.get("head_role_map_sha256") or "")
        != digest(paths["head_role_map"])
    ):
        raise ValueError("strategic curriculum receipt digest mismatch")
    checks = receipt.get("checks")
    required_checks = (
        "guide_supervision_terminates_at_strategic_heads",
        "direct_policy_cross_entropy_absent",
        "observed_outcome_targets_not_replaced",
        "all_training_paths_use_the_declared_mode",
    )
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in required_checks
    ):
        raise ValueError("strategic curriculum receipt checks are incomplete")


def _strategic_curriculum_contract_record(
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Serialize the immutable future loss route without guide actions."""

    if (
        canonical_guide_training_mode(cfg.current_deck_guide_training_mode)
        != GUIDE_TRAINING_MODE_STRATEGIC
    ):
        raise ValueError("strategic curriculum record requested for legacy mode")

    def artifact(path_value: str) -> dict[str, str]:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"strategic curriculum checkpoint artifact is missing: {path}"
            )
        return {
            "path": str(path),
            "sha256": "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    return {
        "schema": "poke_bot.strategic_guide_training_contract/v1",
        "mode": GUIDE_TRAINING_MODE_STRATEGIC,
        "guide_multiplier": float(cfg.alakazam_guide_loss_weight),
        "setup_board_outcome_base_loss_weight": float(
            cfg.setup_board_outcome_loss_weight
        ),
        "combo_state_base_loss_weight": float(
            cfg.combo_state_loss_weight
        ),
        "observed_target_heads": [
            *GUIDE_OUTCOME_BACKED_HEAD_IDS,
            "setup_board_outcome",
            *(
                ["combo_state"]
                if float(cfg.combo_state_loss_weight) > 0.0
                else []
            ),
        ],
        "direct_policy_cross_entropy": False,
        "guide_preferred_action_consumed": False,
        "observed_outcome_targets_replaced": False,
        "artifacts": {
            "curriculum_spec": artifact(
                cfg.current_deck_guide_curriculum_spec
            ),
            "head_role_map": artifact(
                cfg.current_deck_guide_head_role_map
            ),
            "validation_receipt": artifact(
                cfg.current_deck_guide_curriculum_validation_receipt
            ),
        },
    }


def _strategic_curriculum_training_record(
    *,
    cfg: TrainConfig,
    train_metrics: BatchMetrics,
    validation_metrics: BatchMetrics,
) -> dict[str, Any]:
    """Serialize future loss telemetry without recording guide actions."""

    def metrics(value: BatchMetrics) -> dict[str, Any]:
        return {
            "guide_conditioned_observed_loss": float(
                value.guide_strategic_curriculum_loss
            ),
            "setup_board_outcome_weighted_loss": float(
                value.setup_board_outcome_loss
            ),
            "combo_state_weighted_loss": float(value.combo_state_loss),
            "guide_rows": int(value.n_alakazam_guide_rows),
            "head_metrics": copy.deepcopy(
                value.guide_curriculum_head_metrics
            ),
            "setup_metrics": copy.deepcopy(
                value.setup_board_outcome_metrics
            ),
            "combo_state_metrics": copy.deepcopy(
                value.combo_state_metrics
            ),
        }

    return {
        "schema": "poke_bot.strategic_guide_training/v1",
        "contract": _strategic_curriculum_contract_record(cfg),
        "train": metrics(train_metrics),
        "validation": metrics(validation_metrics),
    }


def count_usable_strategic_guide_rows(
    sequences: Sequence[GameSequence],
) -> int:
    """Count confidence-bearing rows without reading guide action preferences."""

    usable = 0
    for game in sequences:
        for decision in game.decisions:
            for stage in decision.policy_stages:
                confidence = float(getattr(stage, "guide_confidence", 0.0))
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        "strategic guide confidence must be finite in [0, 1]"
                    )
                if int(stage.options.num_words) >= 1 and confidence > 0.0:
                    usable += 1
    return usable


def _setup_board_targets_from_aux(
    decision_aux: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize only the existing observed resource/outcome labels."""

    resource_targets: list[list[float]] = []
    resource_masks: list[list[bool]] = []
    outcome_targets: list[int] = []
    outcome_masks: list[bool] = []
    for aux in decision_aux:
        raw = dict(aux or {}).get(EXPANDED_STRATEGIC_KEY)
        if raw is None:
            resource_targets.append([0.0] * 6)
            resource_masks.append([False] * 6)
            outcome_targets.append(0)
            outcome_masks.append(False)
            continue
        target = validate_expanded_strategic_labels(raw)
        resource = dict(target["resource_forecast"])
        resource_targets.append(
            [float(value) for value in resource["values"]]
        )
        resource_masks.append([bool(value) for value in resource["mask"]])
        raw_outcome = target.get("outcome_class")
        outcome_targets.append(0 if raw_outcome is None else int(raw_outcome))
        outcome_masks.append(raw_outcome is not None)
    return (
        torch.tensor(resource_targets, device=device, dtype=dtype),
        torch.tensor(resource_masks, device=device, dtype=torch.bool),
        torch.tensor(outcome_targets, device=device, dtype=torch.long),
        torch.tensor(outcome_masks, device=device, dtype=torch.bool),
    )


def _combo_state_targets_from_aux(
    decision_aux: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Materialize strict Slowking targets; absent decisions remain masked."""

    top_targets: list[int] = []
    top_masks: list[bool] = []
    seek_targets: list[int] = []
    seek_masks: list[bool] = []
    vector_targets: list[list[float]] = []
    vector_masks: list[list[bool]] = []
    for aux in decision_aux:
        raw = dict(aux or {}).get(COMBO_STATE_KEY)
        if raw is None:
            top_targets.append(0)
            top_masks.append(False)
            seek_targets.append(0)
            seek_masks.append(False)
            vector_targets.append([0.0] * COMBO_STATE_VECTOR_WIDTH)
            vector_masks.append([False] * COMBO_STATE_VECTOR_WIDTH)
            continue
        clean = validate_combo_state_labels(raw)
        top_targets.append(int(clean["top_deck_target"]))
        top_masks.append(bool(clean["top_deck_mask"]))
        seek_targets.append(int(clean["seek_source_target"]))
        seek_masks.append(bool(clean["seek_source_mask"]))
        vector_targets.append(list(clean["vector_target"]))
        vector_masks.append(list(clean["vector_mask"]))
    return (
        torch.tensor(top_targets, device=device, dtype=torch.long),
        torch.tensor(top_masks, device=device, dtype=torch.bool),
        torch.tensor(seek_targets, device=device, dtype=torch.long),
        torch.tensor(seek_masks, device=device, dtype=torch.bool),
        torch.tensor(vector_targets, device=device, dtype=dtype),
        torch.tensor(vector_masks, device=device, dtype=torch.bool),
    )


def _decision_guide_confidences(
    *,
    row_confidences: Sequence[float],
    decision_keys: Sequence[tuple[int, int]],
    stage_indices: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Put each decision's maximum stage confidence on its stage-zero row."""

    if not (
        len(row_confidences) == len(decision_keys) == len(stage_indices)
    ):
        raise ValueError("guide confidence decision rows do not align")
    maxima: dict[tuple[int, int], float] = {}
    for key, confidence in zip(decision_keys, row_confidences):
        maxima[key] = max(maxima.get(key, 0.0), float(confidence))
    values = [
        maxima[key] if int(stage_index) == 0 else 0.0
        for key, stage_index in zip(decision_keys, stage_indices)
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def _resident_decision_guide_confidences(
    sample_confidences: torch.Tensor,
    sample_state_rows: torch.Tensor,
    *,
    decisions: int,
) -> torch.Tensor:
    """Reduce factorized-stage confidence to one causal decision weight."""

    result = torch.zeros(
        int(decisions),
        device=sample_confidences.device,
        dtype=sample_confidences.dtype,
    )
    result.scatter_reduce_(
        0,
        sample_state_rows.to(dtype=torch.long),
        sample_confidences,
        reduce="amax",
        include_self=True,
    )
    return result


def count_usable_alakazam_guide_rows(
    sequences: Sequence[GameSequence],
) -> int:
    """Count non-flat, comparable guide rows without running the model."""
    usable = 0
    for game in sequences:
        for decision in game.decisions:
            for stage in decision.policy_stages:
                target = int(getattr(stage, "guide_target_index", -1))
                confidence = float(getattr(stage, "guide_confidence", 0.0))
                n = int(stage.options.num_words)
                if target >= n:
                    raise ValueError(
                        "current-deck guide target is not aligned to policy options"
                    )
                if n >= 2 and target >= 0 and confidence > 0.0:
                    usable += 1
    return usable


def sequence_losses(
    model: TemporalCabtTransformer,
    seq: GameSequence,
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    opp_hand_weight: float = 0.2,
    opp_remainder_weight: float = 0.15,
    alakazam_guide_weight: float = 0.0,
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY,
    setup_board_outcome_loss_weight: float = (
        SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT
    ),
    combo_state_loss_weight: float = 0.0,
    lethal_threat_weight: float = 0.0,
    prize_race_weight: float = 0.0,
    expanded_head_weights: Optional[dict[str, float]] = None,
    pure_rl: bool = False,
    awr_beta: float = 0.5,
    awr_weight_max: float = 20.0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Causal history-conditioned losses for one :class:`GameSequence`."""
    return batch_losses(
        model,
        [seq],
        value_weight=value_weight,
        aux_weight=aux_weight,
        opp_hand_weight=opp_hand_weight,
        opp_remainder_weight=opp_remainder_weight,
        alakazam_guide_weight=alakazam_guide_weight,
        current_deck_guide_training_mode=current_deck_guide_training_mode,
        setup_board_outcome_loss_weight=setup_board_outcome_loss_weight,
        combo_state_loss_weight=combo_state_loss_weight,
        lethal_threat_weight=lethal_threat_weight,
        prize_race_weight=prize_race_weight,
        expanded_head_weights=expanded_head_weights,
        pure_rl=pure_rl,
        awr_beta=awr_beta,
        awr_weight_max=awr_weight_max,
    )


def batch_losses(
    model: TemporalCabtTransformer,
    seqs: Sequence[GameSequence],
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    opp_hand_weight: float = 0.2,
    opp_remainder_weight: float = 0.15,
    alakazam_guide_weight: float = 0.0,
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY,
    setup_board_outcome_loss_weight: float = (
        SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT
    ),
    combo_state_loss_weight: float = 0.0,
    lethal_threat_weight: float = 0.0,
    prize_race_weight: float = 0.0,
    expanded_head_weights: Optional[dict[str, float]] = None,
    opp_hand_multihot: Optional[torch.Tensor] = None,
    opp_remainder_multihot: Optional[torch.Tensor] = None,
    pure_rl: bool = False,
    awr_beta: float = 0.5,
    awr_weight_max: float = 20.0,
    awr_normalize_advantages: bool = True,
    entropy_bonus: float = 0.0,
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None,
    awr_capture_baseline: Optional[dict[tuple[int, int, int], float]] = None,
    awr_weight_sink: Optional[list[float]] = None,
    prediction_sink: Optional[list[int]] = None,
    history_identity_weight: float = 0.0,
    matchup_adapter_training: bool = False,
    pack_temporal_games: bool = False,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Causal history forward over all valid decisions.

    Spatial boards are batched, then each game's temporal states are computed
    with a causal mask. The state used for decision ``t`` can see only
    observations ``<= t`` and is parity-tested against incremental KV serving.

    Belief card multilabel losses are attached with zero/masked defaults so
    late head add does not break the training loop when labels are absent.

    Already-started guide runs collapse scores to a unique-best action and use
    masked CE. ``strategic_curriculum_v1`` never reads that action preference:
    confidence scales only losses whose direction comes from observed causal
    strategic/setup targets.

    Scope B (``lethal_threat`` / ``prize_race``) losses are masked when labels
    are absent and default-weight 0 so core / generic trains ignore them.

    When ``pure_rl=True``, soft behavior policy targets are rejected and the
    policy term is advantage-weighted regression (AWR) on the selected action
    index only — never CE toward cloned ``history_policy`` soft targets.
    AWR can consume a value cache keyed by ``(id(game), decision, stage)``.
    ``rl_train_step`` precomputes that cache once per pure-RL iteration, making
    the baseline genuinely stale/frozen. Direct callers without a cache retain
    the legacy detached-online behavior. Optional per-batch whitening and a
    small entropy bonus stabilize high-SPS pure RL.
    """
    if not math.isfinite(float(alakazam_guide_weight)) or float(
        alakazam_guide_weight
    ) < 0.0:
        raise ValueError(
            "current-deck guide loss weight must be finite and nonnegative"
        )
    guide_training_mode = canonical_guide_training_mode(
        current_deck_guide_training_mode
    )
    strategic_curriculum = (
        guide_training_mode == GUIDE_TRAINING_MODE_STRATEGIC
    )
    if strategic_curriculum:
        assert_strategic_curriculum_model_contract(
            model,
            setup_board_outcome_loss_weight=setup_board_outcome_loss_weight,
        )
        if matchup_adapter_training:
            raise ValueError(
                "strategic guide curriculum cannot run in adapter-only fitting"
            )
    expanded_weights = canonical_expanded_loss_weights(expanded_head_weights)
    if not math.isfinite(float(combo_state_loss_weight)) or float(
        combo_state_loss_weight
    ) < 0.0:
        raise ValueError("combo-state loss weight must be finite and nonnegative")
    use_combo_state = float(combo_state_loss_weight) > 0.0
    if use_combo_state and not bool(
        getattr(model, "combo_state_head_enabled", False)
    ):
        raise ValueError(
            "nonzero combo-state loss requires a combo-state-head checkpoint"
        )
    use_expanded_heads = strategic_curriculum or any(
        weight > 0.0 for weight in expanded_weights.values()
    )
    use_option_aux_heads = use_expanded_heads or use_combo_state
    if use_expanded_heads and not bool(
        getattr(model, "expanded_heads_enabled", False)
    ):
        raise ValueError(
            "nonzero expanded strategic loss requires an expanded-head checkpoint"
        )
    if matchup_adapter_training:
        assert_matchup_adapter_training_contract(model)
    device = next(model.parameters()).device
    games = [s for s in seqs if s.decisions]
    if not games:
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()

    all_boards = [d.board for g in games for d in g.decisions]
    spatial_all = model.encode_board(all_boards)
    packed_game_states: list[torch.Tensor] | None = None
    packed_identity_states: list[torch.Tensor] | None = None
    if pack_temporal_games:
        if not matchup_adapter_training:
            raise ValueError(
                "packed temporal games are currently authorized only for "
                "isolated matchup-adapter fitting"
            )
        if model.decision_context != "history":
            raise ValueError("packed temporal games require history context")
        lengths = [len(game.decisions) for game in games]
        previous_actions = [
            action
            for game in games
            for action in (
                [None]
                + [decision.action_token for decision in game.decisions[:-1]]
            )
        ]
        cls_all = model.pool_cls(spatial_all)
        cls_all = cls_all + float(model.cfg.history_action_scale) * (
            model.encode_previous_actions(previous_actions)
        )
        cls_by_game = list(cls_all.split(lengths))
        padded_cls = pad_sequence(cls_by_game, batch_first=True)
        length_tensor = torch.tensor(
            lengths, device=device, dtype=torch.long
        )
        padding_mask = (
            torch.arange(padded_cls.size(1), device=device).unsqueeze(0)
            >= length_tensor.unsqueeze(1)
        )
        packed_states, _ = model.temporal_encode(
            padded_cls,
            append=False,
            return_all=True,
            key_padding_mask=padding_mask,
        )
        packed_game_states = [
            packed_states[row, :length]
            for row, length in enumerate(lengths)
        ]
        identity_all = model.temporal_norm(model.pool_cls(spatial_all)).detach()
        packed_identity_states = list(identity_all.split(lengths))
    valid_spatial: list[torch.Tensor] = []
    valid_states: list[torch.Tensor] = []
    valid_identity_states: list[torch.Tensor] = []
    valid_options = []
    valid_n: list[int] = []
    soft_targets: list[Optional[list[float]]] = []
    hard_idx: list[int] = []
    value_targets: list[float] = []
    awr_baseline_keys: list[tuple[int, int, int]] = []
    aux_rows: list[int] = []
    aux_labels: list[int] = []
    decision_aux: list[dict[str, Any]] = []
    expanded_stage_indices: list[int] = []
    expanded_decision_keys: list[tuple[int, int]] = []
    matchup_routes: list[int] = []
    guide_target_rows: list[int] = []
    guide_confidence_rows: list[float] = []
    setup_select_context_rows: list[int] = []
    setup_selected_is_stop_rows: list[bool] = []
    use_legacy_guide = bool(
        float(alakazam_guide_weight) > 0.0
        and guide_training_mode == GUIDE_TRAINING_MODE_LEGACY
    )
    use_guide_confidence = bool(strategic_curriculum or use_legacy_guide)
    spatial_offset = 0
    for game_index, g in enumerate(games):
        game_matchup_routes = (
            training_routes_for_sequence(g)
            if matchup_adapter_training
            else ()
        )
        val = float(g.value)
        pt = g.policy_targets
        factorized_pt = g.factorized_policy_targets
        length = len(g.decisions)
        game_spatial = spatial_all[spatial_offset : spatial_offset + length]
        spatial_offset += length
        if model.decision_context == "history":
            if packed_game_states is not None and packed_identity_states is not None:
                game_states = packed_game_states[game_index]
                game_identity_states = packed_identity_states[game_index]
            else:
                previous_actions = [None] + [
                    decision.action_token for decision in g.decisions[:-1]
                ]
                cls = model.history_tokens(
                    game_spatial, previous_actions
                ).unsqueeze(0)
                game_states, _ = model.temporal_encode(
                    cls, append=False, return_all=True
                )
                game_states = game_states.squeeze(0)
                game_identity_states = model.temporal_norm(
                    model.pool_cls(game_spatial)
                ).detach()
        else:
            cls = model.pool_cls(game_spatial).unsqueeze(1)
            game_states, _ = model.temporal_encode(
                cls, append=False, return_all=True
            )
            game_states = game_states.squeeze(1)
            game_identity_states = game_states.detach()
        last_valid_row: Optional[int] = None
        for t, d in enumerate(g.decisions):
            matchup_route = (
                game_matchup_routes[t]
                if matchup_adapter_training
                else UNKNOWN_ROUTE
            )
            stages = d.policy_stages or [
                PolicyStage(
                    options=d.options,
                    action_combos=d.action_combos,
                    target_index=d.action_combo_index,
                )
            ]
            target_stages = (
                factorized_pt[t]
                if factorized_pt is not None
                and t < len(factorized_pt)
                and factorized_pt[t] is not None
                else None
            )
            for stage_i, stage in enumerate(stages):
                # Adapter fitting is state-masked.  Unknown/pre-trigger states
                # are absent from the objective rather than merely producing a
                # constant base loss that dilutes relevant gradients.
                if matchup_adapter_training and matchup_route == UNKNOWN_ROUTE:
                    continue
                n_opt = stage.options.num_words
                if n_opt <= 0:
                    continue
                soft = None
                if target_stages is not None and stage_i < len(target_stages):
                    row = dict(target_stages[stage_i] or {})
                    recorded_combos = [
                        list(combo) for combo in (row.get("action_combos") or [])
                    ]
                    if recorded_combos and recorded_combos != stage.action_combos:
                        raise ValueError(
                            "factorized target/action candidate ordering mismatch"
                        )
                    cand = list(row.get("policy") or [])
                    if len(cand) != n_opt or sum(cand) <= 0:
                        raise ValueError("invalid factorized soft policy target")
                    soft = [float(x) for x in cand]
                    idx = int(row.get("selected_index", stage.target_index))
                elif (
                    not d.policy_stages
                    and pt is not None
                    and t < len(pt)
                    and pt[t] is not None
                ):
                    cand = list(pt[t][:n_opt])
                    if len(cand) != n_opt or sum(cand) <= 0:
                        continue
                    soft = cand
                    idx = int(max(range(n_opt), key=lambda j: cand[j]))
                else:
                    idx = int(stage.target_index)
                if idx < 0 or idx >= n_opt:
                    continue
                valid_spatial.append(game_spatial[t])
                valid_states.append(game_states[t])
                valid_identity_states.append(game_identity_states[t])
                valid_options.append(stage.options)
                valid_n.append(n_opt)
                soft_targets.append(soft)
                hard_idx.append(idx)
                value_targets.append(val)
                awr_baseline_keys.append((id(g), t, stage_i))
                decision_aux.append(dict(d.aux_labels or {}))
                expanded_stage_indices.append(stage_i)
                expanded_decision_keys.append((game_index, t))
                matchup_routes.append(matchup_route)
                if use_guide_confidence:
                    guide_confidence_rows.append(
                        float(getattr(stage, "guide_confidence", 0.0))
                    )
                if use_legacy_guide:
                    guide_target_rows.append(
                        int(getattr(stage, "guide_target_index", -1))
                    )
                if strategic_curriculum:
                    setup_select_context_rows.append(
                        int(getattr(stage, "select_context", -1))
                    )
                    setup_selected_is_stop_rows.append(
                        bool(getattr(stage, "selected_is_stop", False))
                    )
                last_valid_row = len(valid_options) - 1
        label = _archetype_label(g.opp_archetype)
        if last_valid_row is not None and label is not None:
            aux_rows.append(last_valid_row)
            aux_labels.append(label)

    if not valid_options:
        return (
            torch.zeros((), device=device, requires_grad=True),
            BatchMetrics(n_games=len(games)),
        )

    state_all = torch.stack(valid_states, dim=0)
    identity_state_all = torch.stack(valid_identity_states, dim=0)
    current_spatial = torch.stack(valid_spatial, dim=0)
    policy_value_state = state_all
    if matchup_adapter_training:
        route_tensor = torch.tensor(
            matchup_routes,
            device=device,
            dtype=torch.long,
        )
        policy_value_state = model.matchup_policy_value_state(
            state_all,
            route_tensor,
            enabled=True,
        )
    decoded = model.decode_options(
        valid_options,
        current_spatial,
        policy_value_state,
        n_options=valid_n,
        return_hidden=use_option_aux_heads,
        decision_fusion_state_vec=state_all,
    )
    if use_option_aux_heads:
        if not isinstance(decoded, tuple):
            raise AssertionError("option auxiliary decoder did not return hidden states")
        logits_all, option_hidden = decoded
        if use_expanded_heads:
            expanded_option_outputs = model.expanded_option_logits(option_hidden)
            expanded_state_outputs = model.expanded_state_logits(state_all)
        else:
            expanded_option_outputs = {}
            expanded_state_outputs = {}
    else:
        if isinstance(decoded, tuple):
            raise AssertionError("legacy option decoder unexpectedly returned a tuple")
        logits_all = decoded
        expanded_option_outputs = {}
        expanded_state_outputs = {}
    value_pred = torch.tanh(model.value_head(policy_value_state)).squeeze(-1)
    need_belief_outputs = any(
        float(weight) > 0.0
        for weight in (
            aux_weight,
            opp_hand_weight,
            opp_remainder_weight,
            lethal_threat_weight,
            prize_race_weight,
        )
    )
    if need_belief_outputs:
        belief = model.belief_aux_logits(state_all)
        aux_logits_all = belief["aux_logits"]
        opp_hand_logits_all = belief["opp_hand_logits"]
        opp_remainder_logits_all = belief["opp_remainder_logits"]
        lethal_logits_all = belief["lethal_threat_logits"]
        prize_race_pred_all = belief["prize_race_pred"]
    else:
        aux_logits_all = None
        opp_hand_logits_all = None
        opp_remainder_logits_all = None
        lethal_logits_all = None
        prize_race_pred_all = None
    k = logits_all.size(0)
    max_n = logits_all.size(1)

    target_idx = torch.tensor(hard_idx, device=device, dtype=torch.long)
    v_target = torch.tensor(value_targets, device=device, dtype=value_pred.dtype)
    log_p = torch.nan_to_num(F.log_softmax(logits_all, dim=-1), neginf=0.0)
    selected_log_p = log_p[torch.arange(k, device=device), target_idx]
    policy_selected_nll = float((-selected_log_p.detach()).mean().item())

    if use_legacy_guide:
        guide_loss, n_guide_rows = masked_alakazam_guide_ce(
            log_p,
            guide_target_rows,
            guide_confidence_rows,
            valid_n,
        )
    else:
        guide_loss, n_guide_rows = log_p.sum() * 0.0, 0

    awr_mean_adv = 0.0
    raw_adv_mean = 0.0
    raw_adv_std = 0.0
    raw_adv_mean_abs = 0.0
    raw_adv_mean_sq = 0.0
    norm_adv_mean = 0.0
    norm_adv_std = 0.0
    norm_adv_mean_abs = 0.0
    norm_adv_mean_sq = 0.0
    awr_w_mean = 0.0
    awr_w_sum = 0.0
    awr_w_sq_sum = 0.0
    awr_w_p50 = 0.0
    awr_w_p95 = 0.0
    awr_w_max = 0.0
    awr_clip_frac = 0.0
    awr_ess = 0.0
    awr_ess_frac = 0.0

    if pure_rl:
        if any(soft is not None for soft in soft_targets):
            raise ValueError(
                "PURE_RL=1 forbids soft factorized_policy_targets as CE/"
                "behavior-clone training targets; store selected_index only"
            )
        if awr_capture_baseline is not None:
            awr_capture_baseline.update(
                {
                    key: float(value)
                    for key, value in zip(
                        awr_baseline_keys,
                        value_pred.detach().float().cpu().tolist(),
                    )
                }
            )
        if awr_baseline_cache is None:
            baseline_pred = value_pred.detach()
        else:
            missing = [key for key in awr_baseline_keys if key not in awr_baseline_cache]
            if missing:
                raise KeyError(
                    "frozen AWR baseline cache is missing "
                    f"{len(missing)} decision-stage row(s)"
                )
            baseline_pred = torch.tensor(
                [awr_baseline_cache[key] for key in awr_baseline_keys],
                device=device,
                dtype=value_pred.dtype,
            )

        # Capture raw diagnostics before whitening. The compatibility
        # ``mean_advantage`` signal intentionally uses mean(abs(A)) so symmetric
        # positive/negative learning signal cannot look dead by cancellation.
        raw_advantages = v_target - baseline_pred.detach()
        raw_stats = raw_advantages.detach().float()
        raw_adv_mean = float(raw_stats.mean().item())
        raw_adv_std = float(raw_stats.std(unbiased=False).item())
        raw_adv_mean_abs = float(raw_stats.abs().mean().item())
        raw_adv_mean_sq = float(raw_stats.square().mean().item())
        awr_mean_adv = raw_adv_mean_abs

        advantages = raw_advantages
        if awr_normalize_advantages and k > 1:
            adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
            advantages = (advantages - advantages.mean()) / adv_std
        norm_stats = advantages.detach().float()
        norm_adv_mean = float(norm_stats.mean().item())
        norm_adv_std = float(norm_stats.std(unbiased=False).item())
        norm_adv_mean_abs = float(norm_stats.abs().mean().item())
        norm_adv_mean_sq = float(norm_stats.square().mean().item())
        beta = max(float(awr_beta), 1e-6)
        wmax = max(float(awr_weight_max), 1e-6)
        raw_w = torch.exp(advantages / beta)
        weights = torch.clamp(raw_w, max=wmax)
        p_loss = -(weights.detach() * selected_log_p).mean()
        if float(entropy_bonus) > 0.0:
            probs = torch.nan_to_num(log_p.exp(), nan=0.0)
            ent = -(probs * log_p).sum(dim=-1).mean()
            p_loss = p_loss - float(entropy_bonus) * ent
        policy_kl = torch.zeros((), device=device)
        awr_w_mean = float(weights.detach().float().mean().item())
        awr_w_sum = float(weights.detach().float().sum().item())
        awr_w_sq_sum = float(weights.detach().float().square().sum().item())
        awr_ess = min(
            float(k),
            (awr_w_sum * awr_w_sum) / max(awr_w_sq_sum, 1e-12),
        )
        awr_ess_frac = awr_ess / max(k, 1)
        sorted_w, _ = torch.sort(weights.detach().float())
        awr_w_p50 = float(sorted_w[int(0.50 * (k - 1))].item())
        awr_w_p95 = float(sorted_w[int(0.95 * (k - 1))].item())
        awr_w_max = float(sorted_w[-1].item())
        if awr_weight_sink is not None:
            awr_weight_sink.extend(sorted_w.cpu().tolist())
        awr_clip_frac = float((raw_w >= wmax).float().mean().item())
    else:
        target_mat = torch.zeros(k, max_n, device=device, dtype=logits_all.dtype)
        target_mat[torch.arange(k, device=device), target_idx] = 1.0
        for r, soft in enumerate(soft_targets):
            if soft is None:
                continue
            row = torch.tensor(soft, device=device, dtype=logits_all.dtype)
            target_mat[r].zero_()
            target_mat[r, : row.numel()] = row / row.sum().clamp_min(1e-8)

        p_loss = -(target_mat * log_p).sum(dim=1).mean()
        target_log = torch.where(
            target_mat > 0,
            target_mat.clamp_min(1e-12).log(),
            torch.zeros_like(target_mat),
        )
        target_entropy = -(target_mat * target_log).sum(dim=1).mean()
        policy_kl = (p_loss - target_entropy).clamp_min(0.0)

    v_loss = F.smooth_l1_loss(value_pred, v_target)
    history_identity_loss = (
        F.mse_loss(state_all, identity_state_all)
        if float(history_identity_weight) > 0.0
        else state_all.sum() * 0.0
    )

    aux_loss = torch.zeros((), device=device)
    if aux_weight > 0 and aux_rows:
        assert aux_logits_all is not None
        row_idx = torch.tensor(aux_rows, device=device, dtype=torch.long)
        aux_logits = aux_logits_all.index_select(0, row_idx)
        labels = torch.tensor(
            aux_labels,
            device=device,
            dtype=torch.long,
        )
        if int(labels.max().item()) >= aux_logits.size(-1):
            raise ValueError(
                "checkpoint auxiliary head is incompatible with registered "
                f"archetype labels ({aux_logits.size(-1)} classes)"
            )
        aux_loss = F.cross_entropy(aux_logits, labels)

    # Masked card-head BCE from privileged aux labels (or explicit override).
    n_hand_rows = 0
    n_remainder_rows = 0
    if float(opp_hand_weight) <= 0.0 and float(opp_remainder_weight) <= 0.0:
        opp_hand_loss = state_all.sum() * 0.0
        opp_remainder_loss = state_all.sum() * 0.0
    elif opp_hand_multihot is None and opp_remainder_multihot is None:
        assert opp_hand_logits_all is not None
        assert opp_remainder_logits_all is not None
        card_vocab = int(
            getattr(model, "belief_card_vocab", opp_hand_logits_all.size(-1))
        )
        hand_rows: list[torch.Tensor] = []
        rem_rows: list[torch.Tensor] = []
        hand_idx: list[int] = []
        rem_idx: list[int] = []
        for i, aux in enumerate(decision_aux):
            hand_mh, rem_mh = belief_multihots_from_aux_labels(
                aux, card_vocab, device=device
            )
            if hand_mh is not None:
                hand_rows.append(hand_mh)
                hand_idx.append(i)
            if rem_mh is not None:
                rem_rows.append(rem_mh)
                rem_idx.append(i)
        if hand_rows:
            n_hand_rows = len(hand_rows)
            opp_hand_loss = F.binary_cross_entropy_with_logits(
                opp_hand_logits_all.index_select(
                    0, torch.tensor(hand_idx, device=device, dtype=torch.long)
                ),
                torch.stack(hand_rows, dim=0),
            )
        else:
            opp_hand_loss = masked_belief_card_bce(opp_hand_logits_all, None)
        if rem_rows:
            n_remainder_rows = len(rem_rows)
            opp_remainder_loss = F.binary_cross_entropy_with_logits(
                opp_remainder_logits_all.index_select(
                    0, torch.tensor(rem_idx, device=device, dtype=torch.long)
                ),
                torch.stack(rem_rows, dim=0),
            )
        else:
            opp_remainder_loss = masked_belief_card_bce(
                opp_remainder_logits_all, None
            )
    else:
        assert opp_hand_logits_all is not None
        assert opp_remainder_logits_all is not None
        hand_labels = opp_hand_multihot
        rem_labels = opp_remainder_multihot
        if hand_labels is not None and hand_labels.dim() == 1:
            hand_labels = hand_labels.unsqueeze(0).expand(k, -1)
        if rem_labels is not None and rem_labels.dim() == 1:
            rem_labels = rem_labels.unsqueeze(0).expand(k, -1)
        opp_hand_loss = masked_belief_card_bce(opp_hand_logits_all, hand_labels)
        opp_remainder_loss = masked_belief_card_bce(
            opp_remainder_logits_all, rem_labels
        )
        n_hand_rows = k if hand_labels is not None else 0
        n_remainder_rows = k if rem_labels is not None else 0

    # Scope B — masked when labels absent; weights default 0 on core/generic.
    lethal_rows: list[float] = []
    lethal_idx: list[int] = []
    race_rows: list[torch.Tensor] = []
    race_idx: list[int] = []
    if float(lethal_threat_weight) <= 0.0 and float(prize_race_weight) <= 0.0:
        lethal_threat_loss = state_all.sum() * 0.0
        prize_race_loss = state_all.sum() * 0.0
    else:
        assert lethal_logits_all is not None
        assert prize_race_pred_all is not None
        for i, aux in enumerate(decision_aux):
            lethal = lethal_target_from_aux(aux)
            if lethal is not None:
                lethal_rows.append(float(lethal))
                lethal_idx.append(i)
            race = prize_race_target_from_aux(aux, device=device)
            if race is not None:
                race_rows.append(race)
                race_idx.append(i)
        if lethal_rows:
            lethal_threat_loss = F.binary_cross_entropy_with_logits(
                lethal_logits_all.index_select(
                    0, torch.tensor(lethal_idx, device=device, dtype=torch.long)
                ),
                torch.tensor(
                    lethal_rows, device=device, dtype=lethal_logits_all.dtype
                ),
            )
        else:
            lethal_threat_loss = masked_bce_logit(lethal_logits_all, None)
        if race_rows:
            prize_race_loss = F.smooth_l1_loss(
                prize_race_pred_all.index_select(
                    0, torch.tensor(race_idx, device=device, dtype=torch.long)
                ),
                torch.stack(race_rows, dim=0).to(
                    dtype=prize_race_pred_all.dtype
                ),
            )
        else:
            prize_race_loss = masked_smooth_l1(prize_race_pred_all, None)

    total = (
        p_loss
        + value_weight * v_loss
        + aux_weight * aux_loss
        + float(alakazam_guide_weight) * guide_loss
        + float(opp_hand_weight) * opp_hand_loss
        + float(opp_remainder_weight) * opp_remainder_loss
        + float(lethal_threat_weight) * lethal_threat_loss
        + float(prize_race_weight) * prize_race_loss
        + float(history_identity_weight) * history_identity_loss
    )
    expanded_loss = total.sum() * 0.0
    expanded_metrics: dict[str, Any] = {}
    guide_strategic_curriculum_loss = total.sum() * 0.0
    guide_curriculum_metrics: dict[str, Any] = {}
    setup_loss = total.sum() * 0.0
    setup_metrics: dict[str, Any] = {}
    combo_loss = total.sum() * 0.0
    combo_metrics: dict[str, Any] = {}
    if use_expanded_heads:
        expanded_loss, expanded_metric_record = expanded_strategic_losses(
            option_outputs=expanded_option_outputs,
            state_outputs=expanded_state_outputs,
            target_indices=target_idx,
            value_targets=v_target,
            option_counts=valid_n,
            stage_indices=expanded_stage_indices,
            decision_aux=decision_aux,
            weights=expanded_weights,
        )
        total = total + expanded_loss
        expanded_metrics = expanded_metric_record.as_dict()
    if strategic_curriculum:
        if not (
            len(guide_confidence_rows)
            == len(setup_select_context_rows)
            == len(setup_selected_is_stop_rows)
            == k
        ):
            raise AssertionError(
                "strategic curriculum metadata does not align with policy rows"
            )
        guide_confidence = torch.tensor(
            guide_confidence_rows,
            device=device,
            dtype=option_hidden.dtype,
        )
        decision_guide_confidence = _decision_guide_confidences(
            row_confidences=guide_confidence_rows,
            decision_keys=expanded_decision_keys,
            stage_indices=expanded_stage_indices,
            device=device,
            dtype=option_hidden.dtype,
        )
        (
            guide_strategic_curriculum_loss,
            guide_curriculum_metric_record,
        ) = expanded_strategic_losses(
            option_outputs=expanded_option_outputs,
            state_outputs=expanded_state_outputs,
            target_indices=target_idx,
            value_targets=v_target,
            option_counts=valid_n,
            stage_indices=expanded_stage_indices,
            decision_aux=decision_aux,
            weights=guide_outcome_backed_loss_weights(),
            row_weights=guide_confidence,
            state_row_weights=decision_guide_confidence,
        )
        resource_target, resource_mask, outcome_target, outcome_mask = (
            _setup_board_targets_from_aux(
                decision_aux,
                device=device,
                dtype=option_hidden.dtype,
            )
        )
        setup_prediction = model.setup_board_outcome_logits(option_hidden)
        setup_loss, setup_metric_record = setup_board_outcome_loss(
            predictions=setup_prediction,
            selected_indices=target_idx,
            option_counts=torch.as_tensor(
                valid_n, device=device, dtype=torch.long
            ),
            select_contexts=torch.tensor(
                setup_select_context_rows,
                device=device,
                dtype=torch.long,
            ),
            selected_is_stop=torch.tensor(
                setup_selected_is_stop_rows,
                device=device,
                dtype=torch.bool,
            ),
            resource_targets=resource_target,
            resource_masks=resource_mask,
            outcome_targets=outcome_target,
            outcome_masks=outcome_mask,
            guide_confidences=guide_confidence,
            base_loss_weight=setup_board_outcome_loss_weight,
            guide_loss_weight=alakazam_guide_weight,
        )
        total = (
            total
            + float(alakazam_guide_weight)
            * guide_strategic_curriculum_loss
            + setup_loss
        )
        guide_loss = guide_strategic_curriculum_loss
        n_guide_rows = int(guide_confidence.gt(0.0).sum().item())
        guide_curriculum_metrics = (
            guide_curriculum_metric_record.as_dict()
        )
        setup_metrics = setup_metric_record.as_dict()
    if use_combo_state:
        (
            combo_top_target,
            combo_top_mask,
            combo_seek_target,
            combo_seek_mask,
            combo_vector_target,
            combo_vector_mask,
        ) = _combo_state_targets_from_aux(
            decision_aux,
            device=device,
            dtype=option_hidden.dtype,
        )
        combo_guide_confidence = torch.tensor(
            guide_confidence_rows
            if len(guide_confidence_rows) == k
            else [0.0] * k,
            device=device,
            dtype=option_hidden.dtype,
        )
        combo_loss, combo_metric_record = combo_state_loss(
            predictions=model.combo_state_logits(option_hidden),
            selected_indices=target_idx,
            option_counts=torch.as_tensor(valid_n, device=device, dtype=torch.long),
            top_deck_targets=combo_top_target,
            top_deck_masks=combo_top_mask,
            seek_source_targets=combo_seek_target,
            seek_source_masks=combo_seek_mask,
            vector_targets=combo_vector_target,
            vector_masks=combo_vector_mask,
            guide_confidences=combo_guide_confidence,
            base_loss_weight=float(combo_state_loss_weight),
            guide_loss_weight=(
                float(alakazam_guide_weight) if strategic_curriculum else 0.0
            ),
        )
        total = total + combo_loss
        combo_metrics = combo_metric_record.as_dict()
    preds = logits_all.argmax(dim=1)
    if prediction_sink is not None:
        prediction_sink.extend(int(x) for x in preds.detach().cpu().tolist())
    correct = int((preds == target_idx).sum().item())
    metrics = BatchMetrics(
        policy_loss=float(p_loss.detach().item()),
        value_loss=float(v_loss.detach().item()),
        aux_loss=float(aux_loss.detach().item()),
        alakazam_guide_loss=float(guide_loss.detach().item()),
        guide_strategic_curriculum_loss=float(
            guide_strategic_curriculum_loss.detach().item()
        ),
        setup_board_outcome_loss=float(setup_loss.detach().item()),
        combo_state_loss=float(combo_loss.detach().item()),
        opp_hand_loss=float(opp_hand_loss.detach().item()),
        opp_remainder_loss=float(opp_remainder_loss.detach().item()),
        lethal_threat_loss=float(lethal_threat_loss.detach().item()),
        prize_race_loss=float(prize_race_loss.detach().item()),
        history_identity_loss=float(history_identity_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=correct / max(k, 1),
        policy_kl=float(policy_kl.detach().item()) if torch.is_tensor(policy_kl) else float(policy_kl),
        target_value_mean=float(v_target.detach().float().mean().item()),
        value_pred_mean=float(value_pred.detach().float().mean().item()),
        n_decisions=k,
        n_games=len(games),
        n_alakazam_guide_rows=int(n_guide_rows),
        n_archetype_rows=len(aux_rows),
        n_opp_hand_rows=int(n_hand_rows),
        n_opp_remainder_rows=int(n_remainder_rows),
        n_lethal_threat_rows=len(lethal_rows),
        n_prize_race_rows=len(race_rows),
        n_matchup_adapter_rows=(
            sum(route != UNKNOWN_ROUTE for route in matchup_routes)
            if matchup_adapter_training
            else 0
        ),
        expanded_head_metrics=expanded_metrics,
        guide_curriculum_head_metrics=guide_curriculum_metrics,
        setup_board_outcome_metrics=setup_metrics,
        combo_state_metrics=combo_metrics,
        mean_advantage=awr_mean_adv,
        raw_advantage_mean=raw_adv_mean,
        raw_advantage_std=raw_adv_std,
        raw_advantage_mean_abs=raw_adv_mean_abs,
        raw_advantage_mean_sq=raw_adv_mean_sq,
        normalized_advantage_mean=norm_adv_mean,
        normalized_advantage_std=norm_adv_std,
        normalized_advantage_mean_abs=norm_adv_mean_abs,
        normalized_advantage_mean_sq=norm_adv_mean_sq,
        awr_weight_mean=awr_w_mean,
        awr_weight_sum=awr_w_sum,
        awr_weight_sq_sum=awr_w_sq_sum,
        awr_weight_p50=awr_w_p50,
        awr_weight_p95=awr_w_p95,
        awr_weight_max_observed=awr_w_max,
        awr_weight_clip_frac=awr_clip_frac,
        awr_effective_sample_size=awr_ess,
        awr_effective_sample_fraction=awr_ess_frac,
        policy_selected_nll=policy_selected_nll,
    )
    return total, metrics


def _resident_hard_target_objective(
    model: TemporalCabtTransformer,
    logits: torch.Tensor,
    state: torch.Tensor,
    target_idx: torch.Tensor,
    v_target: torch.Tensor,
    *,
    value_weight: float,
    n_games: int = 0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Shared supervised objective for stateless and temporal resident paths."""
    k = int(target_idx.numel())
    if k <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()
    value_pred = torch.tanh(model.value_head(state)).squeeze(-1)
    v_target = v_target.to(dtype=value_pred.dtype)
    log_p = torch.nan_to_num(F.log_softmax(logits, dim=-1), neginf=0.0)
    selected_log_p = log_p[
        torch.arange(k, device=logits.device), target_idx
    ]
    p_loss = -selected_log_p.mean()
    v_loss = F.smooth_l1_loss(value_pred, v_target)
    total = p_loss + float(value_weight) * v_loss
    preds = logits.argmax(dim=1)
    correct = int((preds == target_idx).sum().item())
    p_scalar = float(p_loss.detach().item())
    metrics = BatchMetrics(
        policy_loss=p_scalar,
        value_loss=float(v_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=correct / max(k, 1),
        policy_kl=max(p_scalar, 0.0),
        target_value_mean=float(v_target.detach().float().mean().item()),
        value_pred_mean=float(value_pred.detach().float().mean().item()),
        n_decisions=k,
        n_games=int(n_games),
        policy_selected_nll=p_scalar,
    )
    return total, metrics


def device_batch_losses(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    sample_ids: torch.Tensor,
    *,
    value_weight: float = 1.0,
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Hard-target stateless loss with every input already on the device.

    This is deliberately narrow: the latest-ladder hot start has hard selected
    actions and all auxiliary heads disabled.  Keeping that contract explicit
    prevents the fast path from silently discarding a future target type.
    """
    if model.decision_context != "stateless":
        raise ValueError("device-resident bootstrap requires stateless context")
    if (
        canonical_guide_training_mode(current_deck_guide_training_mode)
        == GUIDE_TRAINING_MODE_STRATEGIC
    ):
        raise ValueError(
            "strategic curriculum requires the resident temporal training path"
        )
    board, options, counts, target_idx, v_target = corpus.batch(sample_ids)
    k = int(target_idx.numel())
    if k <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()

    spatial = model.encode_board_packed(board, batch_size=k)
    cls = model.pool_cls(spatial).unsqueeze(1)
    states, _ = model.temporal_encode(cls, append=False, return_all=True)
    state = states.squeeze(1)
    logits = model.decode_options_packed(
        options,
        spatial,
        state,
        n_options=counts,
        batch_size=k,
    )
    return _resident_hard_target_objective(
        model,
        logits,
        state,
        target_idx,
        v_target,
        value_weight=value_weight,
    )


def device_temporal_batch_losses(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    game_ids: torch.Tensor,
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.0,
    opp_hand_weight: float = 0.0,
    opp_remainder_weight: float = 0.0,
    lethal_threat_weight: float = 0.0,
    prize_race_weight: float = 0.0,
    alakazam_guide_weight: float = 0.0,
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY,
    setup_board_outcome_loss_weight: float = (
        SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT
    ),
    combo_state_loss_weight: float = 0.0,
    expanded_head_weights: Optional[dict[str, float]] = None,
    matchup_adapter_route: int | None = None,
    teacher_policy_targets: Optional[torch.Tensor] = None,
    teacher_policy_weight: float = 0.0,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Hard-target full-game loss with every resident temporal target.

    Policy/value behavior is the same as :func:`_resident_hard_target_objective`.
    Optional auxiliary targets are trained only on rows whose packed presence
    masks are valid.  In particular, an absent exact opponent hand is not
    interpreted as an observed empty hand.
    """
    if model.decision_context != "history":
        raise ValueError("temporal resident loss requires history context")
    guide_training_mode = canonical_guide_training_mode(
        current_deck_guide_training_mode
    )
    strategic_curriculum = (
        guide_training_mode == GUIDE_TRAINING_MODE_STRATEGIC
    )
    if strategic_curriculum:
        assert_strategic_curriculum_model_contract(
            model,
            setup_board_outcome_loss_weight=setup_board_outcome_loss_weight,
        )
    expanded_weights = canonical_expanded_loss_weights(expanded_head_weights)
    if not math.isfinite(float(combo_state_loss_weight)) or float(
        combo_state_loss_weight
    ) < 0.0:
        raise ValueError("resident combo-state weight must be finite and nonnegative")
    use_combo_state = float(combo_state_loss_weight) > 0.0
    if use_combo_state:
        if not bool(getattr(model, "combo_state_head_enabled", False)):
            raise ValueError(
                "nonzero resident combo-state loss requires a combo-state head"
            )
        if not corpus.has_combo_state_targets:
            raise ValueError(
                "nonzero resident combo-state loss requires packed combo targets"
            )
    use_expanded_heads = strategic_curriculum or any(
        weight > 0.0 for weight in expanded_weights.values()
    )
    use_option_aux_heads = use_expanded_heads or use_combo_state
    if use_expanded_heads:
        if not bool(getattr(model, "expanded_heads_enabled", False)):
            raise ValueError(
                "nonzero resident strategic loss requires an expanded-head checkpoint"
            )
        if not corpus.has_expanded_strategic_targets:
            raise ValueError(
                "nonzero resident strategic loss requires packed strategic targets"
            )
        if (
            str(corpus.expanded_strategic_schema) != EXPANDED_STRATEGIC_SCHEMA
            or str(corpus.expanded_strategic_schema_digest)
            != TARGET_SCHEMA_DIGEST
        ):
            raise ValueError(
                "resident strategic target schema or digest does not match runtime"
            )
    if matchup_adapter_route is not None:
        if not (
            type(matchup_adapter_route) is int
            and 0 <= matchup_adapter_route < len(EXPERT_IDS)
        ):
            raise ValueError("invalid resident matchup-adapter route")
        assert_matchup_adapter_training_contract(model)
    if float(teacher_policy_weight) < 0.0:
        raise ValueError("teacher policy weight cannot be negative")
    if float(teacher_policy_weight) > 0.0:
        if teacher_policy_targets is None:
            raise ValueError(
                "nonzero teacher policy weight requires resident teacher targets"
            )
        if teacher_policy_targets.device != corpus.device:
            raise ValueError(
                "teacher policy targets must reside with the expert corpus"
            )
        if teacher_policy_targets.ndim != 1 or int(
            teacher_policy_targets.numel()
        ) != int(corpus.total_samples):
            raise ValueError(
                "teacher policy targets must contain one row per resident sample"
            )
    for name, weight in (
        ("aux", aux_weight),
        ("opp_hand", opp_hand_weight),
        ("opp_remainder", opp_remainder_weight),
        ("lethal_threat", lethal_threat_weight),
        ("prize_race", prize_race_weight),
        ("alakazam_guide", alakazam_guide_weight),
    ):
        if float(weight) < 0.0:
            raise ValueError(f"resident temporal {name} weight cannot be negative")
    (
        board,
        previous_actions,
        options,
        counts,
        target_idx,
        v_target,
        game_lengths,
        sample_state_rows,
    ) = corpus.temporal_batch(game_ids)
    decisions = int(game_lengths.sum().item())
    samples = int(target_idx.numel())
    if decisions <= 0 or samples <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()
    if int(game_lengths.max().item()) > int(model.max_context):
        raise ValueError(
            "resident temporal game exceeds checkpoint context: "
            f"game={int(game_lengths.max().item())} max={int(model.max_context)}"
        )

    spatial = model.encode_board_packed(board, batch_size=decisions)
    action_state = model.encode_previous_actions_packed(
        previous_actions, batch_size=decisions
    )
    cls = model.pool_cls(spatial) + float(model.cfg.history_action_scale) * action_state

    # Games of equal length can share one temporal call: the batch dimension is
    # isolated by attention, while avoiding one Python/model launch per game.
    lengths = [int(value) for value in game_lengths.cpu().tolist()]
    starts: list[int] = []
    cursor = 0
    by_length: dict[int, list[int]] = {}
    for length in lengths:
        starts.append(cursor)
        by_length.setdefault(length, []).append(cursor)
        cursor += length
    state_parts: list[torch.Tensor] = []
    row_parts: list[torch.Tensor] = []
    for length, group_starts in by_length.items():
        tokens = torch.stack(
            [cls[start : start + length] for start in group_starts], dim=0
        )
        encoded, _ = model.temporal_encode(
            tokens, append=False, return_all=True
        )
        state_parts.append(encoded.reshape(-1, model.d_model))
        row_parts.append(
            torch.cat(
                [
                    torch.arange(
                        start,
                        start + length,
                        device=corpus.device,
                        dtype=torch.long,
                    )
                    for start in group_starts
                ]
            )
        )
    grouped_states = torch.cat(state_parts, dim=0)
    grouped_rows = torch.cat(row_parts, dim=0)
    state_by_decision = grouped_states.index_select(
        0, torch.argsort(grouped_rows)
    )
    state = state_by_decision.index_select(0, sample_state_rows)
    sample_spatial = spatial.index_select(0, sample_state_rows)
    policy_value_state = state
    if matchup_adapter_route is not None:
        policy_value_state = model.matchup_policy_value_state(
            state,
            torch.full(
                (samples,),
                matchup_adapter_route,
                device=state.device,
                dtype=torch.long,
            ),
            enabled=True,
        )
    decoded = model.decode_options_packed(
        options,
        sample_spatial,
        policy_value_state,
        n_options=counts,
        batch_size=samples,
        return_hidden=use_option_aux_heads,
        decision_fusion_state_vec=state,
    )
    expanded_option_outputs: dict[str, torch.Tensor] = {}
    expanded_state_outputs: dict[str, torch.Tensor] = {}
    if use_option_aux_heads:
        if not isinstance(decoded, tuple):
            raise AssertionError(
                "resident option auxiliary decoder did not return hidden states"
            )
        logits, option_hidden = decoded
        if use_expanded_heads:
            expanded_option_outputs = model.expanded_option_logits(option_hidden)
            # State targets are decision-aligned, so they are evaluated exactly
            # once per real decision rather than once per factorized policy stage.
            expanded_state_outputs = model.expanded_state_logits(state_by_decision)
    else:
        if isinstance(decoded, tuple):
            raise AssertionError(
                "legacy resident option decoder unexpectedly returned a tuple"
            )
        logits = decoded
    total, metrics = _resident_hard_target_objective(
        model,
        logits,
        policy_value_state,
        target_idx,
        v_target,
        value_weight=value_weight,
        n_games=int(game_ids.numel()),
    )
    guide_loss = logits.sum() * 0.0
    guide_strategic_curriculum_loss = logits.sum() * 0.0
    setup_loss = logits.sum() * 0.0
    sample_ids: Optional[torch.Tensor] = None
    if float(teacher_policy_weight) > 0.0:
        assert teacher_policy_targets is not None
        assert corpus.game_sample_offset is not None
        resident_game_ids = game_ids.reshape(-1).to(
            device=corpus.device, dtype=torch.long
        )
        sample_starts = corpus.game_sample_offset.index_select(
            0, resident_game_ids
        ).to(dtype=torch.long)
        sample_ends = corpus.game_sample_offset.index_select(
            0, resident_game_ids + 1
        ).to(dtype=torch.long)
        sample_ids, _sample_lengths = corpus._expand_ranges(  # noqa: SLF001
            sample_starts, sample_ends
        )
        if int(sample_ids.numel()) != samples:
            raise AssertionError(
                "resident teacher target mapping disagrees with policy rows"
            )
        teacher_target = teacher_policy_targets.index_select(
            0, sample_ids
        ).to(dtype=torch.long)
        teacher_mask = teacher_target >= 0
        invalid_teacher = teacher_mask & (teacher_target >= counts)
        if bool(invalid_teacher.any()):
            bad_row = int(
                torch.nonzero(invalid_teacher, as_tuple=False)[0].item()
            )
            raise ValueError(
                "teacher policy target is outside resident option row: "
                f"target={int(teacher_target[bad_row].item())} "
                f"options={int(counts[bad_row].item())}"
            )
        if bool(teacher_mask.any()):
            teacher_rows = torch.nonzero(
                teacher_mask, as_tuple=False
            ).flatten()
            teacher_log_p = torch.nan_to_num(
                F.log_softmax(logits, dim=-1), neginf=0.0
            )
            teacher_loss = -teacher_log_p[
                teacher_rows,
                teacher_target.index_select(0, teacher_rows),
            ].mean()
            total = total + float(teacher_policy_weight) * teacher_loss
            metrics.teacher_policy_loss = float(
                teacher_loss.detach().item()
            )
            metrics.n_teacher_policy_rows = int(teacher_rows.numel())
            metrics.total_loss = float(total.detach().item())
    if matchup_adapter_route is not None:
        metrics.n_matchup_adapter_rows = metrics.n_decisions
    auxiliary_targets_enabled = any(
        float(weight) > 0.0
        for weight in (
            aux_weight,
            opp_hand_weight,
            opp_remainder_weight,
            lethal_threat_weight,
            prize_race_weight,
            alakazam_guide_weight,
        )
    )
    if not auxiliary_targets_enabled and not use_expanded_heads and not use_combo_state:
        # Retain the historical policy/value-only path exactly, including RNG
        # consumption (``aux_head`` contains dropout on nonzero-dropout models).
        return total, metrics

    # Recreate the packed sample ids in the same caller-specified game order
    # used by ``temporal_batch``.  Exact targets are decision-indexed while
    # archetype and guide targets are sample-indexed, so both mappings matter.
    assert corpus.game_sample_offset is not None
    resident_game_ids = game_ids.reshape(-1).to(
        device=corpus.device, dtype=torch.long
    )
    if sample_ids is None:
        sample_starts = corpus.game_sample_offset.index_select(
            0, resident_game_ids
        ).to(dtype=torch.long)
        sample_ends = corpus.game_sample_offset.index_select(
            0, resident_game_ids + 1
        ).to(dtype=torch.long)
        sample_ids, _sample_lengths = corpus._expand_ranges(  # noqa: SLF001
            sample_starts, sample_ends
        )
    if int(sample_ids.numel()) != samples:
        raise AssertionError(
            "resident temporal sample target mapping disagrees with policy rows"
        )
    board_ids = corpus.sample_board.index_select(0, sample_ids).to(
        dtype=torch.long
    )

    assert corpus.game_decision_offset is not None
    decision_starts = corpus.game_decision_offset.index_select(
        0, resident_game_ids
    ).to(dtype=torch.long)
    decision_ends = corpus.game_decision_offset.index_select(
        0, resident_game_ids + 1
    ).to(dtype=torch.long)
    decision_ids, _decision_lengths = corpus._expand_ranges(  # noqa: SLF001
        decision_starts, decision_ends
    )
    if int(decision_ids.numel()) != decisions:
        raise AssertionError(
            "resident temporal decision target mapping disagrees with state rows"
        )

    if use_expanded_heads:
        expanded_loss, expanded_metric_record = (
            resident_expanded_strategic_losses(
                option_outputs=expanded_option_outputs,
                state_outputs=expanded_state_outputs,
                target_indices=target_idx,
                option_counts=counts,
                target_tensors=corpus.tensor_state(),
                sample_ids=sample_ids,
                decision_ids=decision_ids,
                weights=expanded_weights,
            )
        )
        total = total + expanded_loss
        metrics.expanded_head_metrics = expanded_metric_record.as_dict()
        metrics.total_loss = float(total.detach().item())
    if strategic_curriculum:
        if corpus.guide_confidence is None:
            raise ValueError(
                "strategic curriculum requires resident guide confidence rows"
            )
        if corpus.select_context is None or corpus.selected_is_stop is None:
            raise ValueError(
                "strategic curriculum requires resident setup context metadata"
            )
        guide_confidence = corpus.guide_confidence.index_select(
            0, sample_ids
        ).to(dtype=option_hidden.dtype)
        if not bool(torch.isfinite(guide_confidence).all()) or bool(
            ((guide_confidence < 0.0) | (guide_confidence > 1.0)).any()
        ):
            raise ValueError(
                "resident strategic guide confidence is outside [0, 1]"
            )
        decision_guide_confidence = _resident_decision_guide_confidences(
            guide_confidence,
            sample_state_rows,
            decisions=decisions,
        )
        (
            guide_strategic_curriculum_loss,
            guide_curriculum_metric_record,
        ) = resident_expanded_strategic_losses(
            option_outputs=expanded_option_outputs,
            state_outputs=expanded_state_outputs,
            target_indices=target_idx,
            option_counts=counts,
            target_tensors=corpus.tensor_state(),
            sample_ids=sample_ids,
            decision_ids=decision_ids,
            weights=guide_outcome_backed_loss_weights(),
            sample_row_weights=guide_confidence,
            decision_row_weights=decision_guide_confidence,
        )
        setup_required = (
            corpus.strategic_resource_forecast_target,
            corpus.strategic_resource_forecast_mask,
            corpus.strategic_outcome_class_target,
            corpus.strategic_outcome_class_mask,
        )
        if any(value is None for value in setup_required):
            raise ValueError(
                "strategic curriculum lacks packed setup observed targets"
            )
        assert corpus.strategic_resource_forecast_target is not None
        assert corpus.strategic_resource_forecast_mask is not None
        assert corpus.strategic_outcome_class_target is not None
        assert corpus.strategic_outcome_class_mask is not None
        setup_loss, setup_metric_record = setup_board_outcome_loss(
            predictions=model.setup_board_outcome_logits(option_hidden),
            selected_indices=target_idx,
            option_counts=counts,
            select_contexts=corpus.select_context.index_select(
                0, sample_ids
            ),
            selected_is_stop=corpus.selected_is_stop.index_select(
                0, sample_ids
            ),
            resource_targets=(
                corpus.strategic_resource_forecast_target.index_select(
                    0, board_ids
                ).to(dtype=option_hidden.dtype)
            ),
            resource_masks=(
                corpus.strategic_resource_forecast_mask.index_select(
                    0, board_ids
                )
            ),
            outcome_targets=(
                corpus.strategic_outcome_class_target.index_select(
                    0, board_ids
                )
            ),
            outcome_masks=(
                corpus.strategic_outcome_class_mask.index_select(
                    0, board_ids
                )
            ),
            guide_confidences=guide_confidence,
            base_loss_weight=setup_board_outcome_loss_weight,
            guide_loss_weight=alakazam_guide_weight,
        )
        total = (
            total
            + float(alakazam_guide_weight)
            * guide_strategic_curriculum_loss
            + setup_loss
        )
        guide_loss = guide_strategic_curriculum_loss
        metrics.alakazam_guide_loss = float(guide_loss.detach().item())
        metrics.guide_strategic_curriculum_loss = float(
            guide_strategic_curriculum_loss.detach().item()
        )
        metrics.setup_board_outcome_loss = float(
            setup_loss.detach().item()
        )
        metrics.n_alakazam_guide_rows = int(
            guide_confidence.gt(0.0).sum().item()
        )
        metrics.guide_curriculum_head_metrics = (
            guide_curriculum_metric_record.as_dict()
        )
        metrics.setup_board_outcome_metrics = setup_metric_record.as_dict()
        metrics.total_loss = float(total.detach().item())
    if use_combo_state:
        combo_fields = (
            corpus.combo_top_deck_target,
            corpus.combo_top_deck_mask,
            corpus.combo_seek_source_target,
            corpus.combo_seek_source_mask,
            corpus.combo_vector_target,
            corpus.combo_vector_mask,
        )
        if any(value is None for value in combo_fields):
            raise ValueError("resident combo-state target tensors are incomplete")
        assert corpus.combo_top_deck_target is not None
        assert corpus.combo_top_deck_mask is not None
        assert corpus.combo_seek_source_target is not None
        assert corpus.combo_seek_source_mask is not None
        assert corpus.combo_vector_target is not None
        assert corpus.combo_vector_mask is not None
        if corpus.guide_confidence is None:
            combo_guide_confidence = torch.zeros(
                samples, device=corpus.device, dtype=option_hidden.dtype
            )
        else:
            combo_guide_confidence = corpus.guide_confidence.index_select(
                0, sample_ids
            ).to(dtype=option_hidden.dtype)
        combo_loss, combo_metric_record = combo_state_loss(
            predictions=model.combo_state_logits(option_hidden),
            selected_indices=target_idx,
            option_counts=counts,
            top_deck_targets=corpus.combo_top_deck_target.index_select(
                0, sample_ids
            ),
            top_deck_masks=corpus.combo_top_deck_mask.index_select(
                0, sample_ids
            ),
            seek_source_targets=corpus.combo_seek_source_target.index_select(
                0, sample_ids
            ),
            seek_source_masks=corpus.combo_seek_source_mask.index_select(
                0, sample_ids
            ),
            vector_targets=corpus.combo_vector_target.index_select(
                0, sample_ids
            ).to(dtype=option_hidden.dtype),
            vector_masks=corpus.combo_vector_mask.index_select(
                0, sample_ids
            ),
            guide_confidences=combo_guide_confidence,
            base_loss_weight=float(combo_state_loss_weight),
            guide_loss_weight=(
                float(alakazam_guide_weight) if strategic_curriculum else 0.0
            ),
        )
        total = total + combo_loss
        metrics.combo_state_loss = float(combo_loss.detach().item())
        metrics.combo_state_metrics = combo_metric_record.as_dict()
        metrics.total_loss = float(total.detach().item())
    if not auxiliary_targets_enabled:
        return total, metrics

    belief = model.belief_aux_logits(state)
    aux_loss = belief["aux_logits"].sum() * 0.0
    opp_hand_loss = belief["opp_hand_logits"].sum() * 0.0
    opp_remainder_loss = belief["opp_remainder_logits"].sum() * 0.0
    lethal_threat_loss = belief["lethal_threat_logits"].sum() * 0.0
    prize_race_loss = belief["prize_race_pred"].sum() * 0.0
    guide_log_p = torch.nan_to_num(
        F.log_softmax(logits, dim=-1), neginf=0.0
    )

    n_archetype_rows = 0
    if corpus.sample_aux_class is not None:
        aux_target = corpus.sample_aux_class.index_select(0, sample_ids).to(
            dtype=torch.long
        )
        aux_mask = aux_target >= 0
        if bool(aux_mask.any()):
            aux_loss = F.cross_entropy(
                belief["aux_logits"][aux_mask], aux_target[aux_mask]
            )
            n_archetype_rows = int(aux_mask.sum().item())

    n_opp_hand_rows = 0
    hand_fields = (
        corpus.hand_index,
        corpus.hand_offset,
        corpus.hand_present,
    )
    if int(corpus.belief_card_vocab) > 0 and all(
        value is not None for value in hand_fields
    ):
        assert corpus.hand_index is not None
        assert corpus.hand_offset is not None
        assert corpus.hand_present is not None
        hand_target, hand_mask = corpus._card_multihot(  # noqa: SLF001
            board_ids,
            index=corpus.hand_index,
            offset=corpus.hand_offset,
            present=corpus.hand_present,
        )
        if bool(hand_mask.any()):
            opp_hand_loss = F.binary_cross_entropy_with_logits(
                belief["opp_hand_logits"][hand_mask], hand_target[hand_mask]
            )
            n_opp_hand_rows = int(hand_mask.sum().item())

    n_opp_remainder_rows = 0
    remainder_fields = (
        corpus.remainder_index,
        corpus.remainder_offset,
        corpus.remainder_present,
    )
    if int(corpus.belief_card_vocab) > 0 and all(
        value is not None for value in remainder_fields
    ):
        assert corpus.remainder_index is not None
        assert corpus.remainder_offset is not None
        assert corpus.remainder_present is not None
        remainder_target, remainder_mask = corpus._card_multihot(  # noqa: SLF001
            board_ids,
            index=corpus.remainder_index,
            offset=corpus.remainder_offset,
            present=corpus.remainder_present,
        )
        if bool(remainder_mask.any()):
            opp_remainder_loss = F.binary_cross_entropy_with_logits(
                belief["opp_remainder_logits"][remainder_mask],
                remainder_target[remainder_mask],
            )
            n_opp_remainder_rows = int(remainder_mask.sum().item())

    n_lethal_threat_rows = 0
    if corpus.lethal_target is not None:
        lethal_target = corpus.lethal_target.index_select(0, board_ids).to(
            dtype=belief["lethal_threat_logits"].dtype
        )
        lethal_mask = torch.isfinite(lethal_target)
        if bool(lethal_mask.any()):
            lethal_threat_loss = F.binary_cross_entropy_with_logits(
                belief["lethal_threat_logits"][lethal_mask],
                lethal_target[lethal_mask],
            )
            n_lethal_threat_rows = int(lethal_mask.sum().item())

    n_prize_race_rows = 0
    if corpus.prize_race_target is not None:
        prize_race_target = corpus.prize_race_target.index_select(
            0, board_ids
        ).to(dtype=belief["prize_race_pred"].dtype)
        prize_race_mask = torch.isfinite(prize_race_target).all(dim=1)
        if bool(prize_race_mask.any()):
            prize_race_loss = F.smooth_l1_loss(
                belief["prize_race_pred"][prize_race_mask],
                prize_race_target[prize_race_mask],
            )
            n_prize_race_rows = int(prize_race_mask.sum().item())

    n_guide_rows = 0
    if (
        guide_training_mode == GUIDE_TRAINING_MODE_LEGACY
        and
        corpus.guide_target_index is not None
        and corpus.guide_confidence is not None
    ):
        guide_target = corpus.guide_target_index.index_select(
            0, sample_ids
        ).to(dtype=torch.long)
        guide_confidence = corpus.guide_confidence.index_select(
            0, sample_ids
        ).to(dtype=guide_log_p.dtype)
        invalid_guide = (guide_target >= counts) & (guide_target >= 0)
        if bool(invalid_guide.any()):
            bad_row = int(torch.nonzero(invalid_guide, as_tuple=False)[0].item())
            raise ValueError(
                "resident current-deck guide target is outside option row: "
                f"target={int(guide_target[bad_row].item())} "
                f"options={int(counts[bad_row].item())}"
            )
        guide_mask = (
            (guide_target >= 0)
            & (guide_confidence > 0.0)
            & (counts >= 2)
        )
        if bool(guide_mask.any()):
            guide_rows = torch.nonzero(guide_mask, as_tuple=False).flatten()
            bounded_confidence = guide_confidence.index_select(
                0, guide_rows
            ).clamp(0.0, 1.0)
            guide_loss = -(
                guide_log_p[
                    guide_rows,
                    guide_target.index_select(0, guide_rows),
                ]
                * bounded_confidence
            ).mean()
            n_guide_rows = int(guide_rows.numel())

    total = (
        total
        + float(aux_weight) * aux_loss
        + float(opp_hand_weight) * opp_hand_loss
        + float(opp_remainder_weight) * opp_remainder_loss
        + float(lethal_threat_weight) * lethal_threat_loss
        + float(prize_race_weight) * prize_race_loss
        + (
            float(alakazam_guide_weight) * guide_loss
            if guide_training_mode == GUIDE_TRAINING_MODE_LEGACY
            else guide_loss * 0.0
        )
    )
    metrics.aux_loss = float(aux_loss.detach().item())
    metrics.opp_hand_loss = float(opp_hand_loss.detach().item())
    metrics.opp_remainder_loss = float(opp_remainder_loss.detach().item())
    metrics.lethal_threat_loss = float(lethal_threat_loss.detach().item())
    metrics.prize_race_loss = float(prize_race_loss.detach().item())
    metrics.alakazam_guide_loss = float(guide_loss.detach().item())
    metrics.total_loss = float(total.detach().item())
    metrics.n_archetype_rows = n_archetype_rows
    metrics.n_opp_hand_rows = n_opp_hand_rows
    metrics.n_opp_remainder_rows = n_opp_remainder_rows
    metrics.n_lethal_threat_rows = n_lethal_threat_rows
    metrics.n_prize_race_rows = n_prize_race_rows
    if guide_training_mode == GUIDE_TRAINING_MODE_LEGACY:
        metrics.n_alakazam_guide_rows = n_guide_rows
    return total, metrics


def temporal_batches_for_game_ids(
    corpus: DeviceResidentBootstrapCorpus,
    game_ids: torch.Tensor,
    *,
    batch_size: int,
) -> list[torch.Tensor]:
    """Pack an explicit resident game subset under the temporal work budget."""

    if not corpus.has_temporal_layout:
        raise ValueError("explicit temporal batches require game layout")
    assert corpus.game_decision_offset is not None
    assert corpus.game_sample_offset is not None
    ids = game_ids.reshape(-1).to(device=corpus.device, dtype=torch.long)
    if ids.numel() == 0:
        return []
    game_count = int(corpus.train_games + corpus.val_games)
    if bool(((ids < 0) | (ids >= game_count)).any()):
        raise IndexError("explicit temporal game id is outside the corpus")
    decision_lengths = (
        corpus.game_decision_offset.index_select(0, ids + 1)
        - corpus.game_decision_offset.index_select(0, ids)
    ).to(dtype=torch.long)
    sample_lengths = (
        corpus.game_sample_offset.index_select(0, ids + 1)
        - corpus.game_sample_offset.index_select(0, ids)
    ).to(dtype=torch.long)
    work = torch.maximum(decision_lengths, sample_lengths).cpu().tolist()
    ordered_ids = ids.cpu().tolist()
    limit = max(1, int(batch_size))
    chunks: list[torch.Tensor] = []
    current: list[int] = []
    current_work = 0
    for game_id, game_work in zip(ordered_ids, work):
        game_work = int(game_work)
        if game_work <= 0:
            continue
        if current and current_work + game_work > limit:
            chunks.append(
                torch.tensor(
                    current, device=corpus.device, dtype=torch.long
                )
            )
            current = []
            current_work = 0
        current.append(int(game_id))
        current_work += game_work
    if current:
        chunks.append(
            torch.tensor(current, device=corpus.device, dtype=torch.long)
        )
    return chunks


@torch.no_grad()
def device_temporal_greedy_policy_targets(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    game_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global sample IDs and causal greedy actions for resident games."""

    if model.decision_context != "history":
        raise ValueError("teacher policy targets require temporal history")
    (
        board,
        previous_actions,
        options,
        counts,
        _target_idx,
        _value_target,
        game_lengths,
        sample_state_rows,
    ) = corpus.temporal_batch(game_ids)
    decisions = int(game_lengths.sum().item())
    samples = int(sample_state_rows.numel())
    if decisions <= 0 or samples <= 0:
        raise ValueError("teacher policy batch has no causal decisions")
    if int(game_lengths.max().item()) > int(model.max_context):
        raise ValueError("teacher checkpoint context is shorter than corpus")

    was_training = model.training
    model.eval()
    spatial = model.encode_board_packed(board, batch_size=decisions)
    action_state = model.encode_previous_actions_packed(
        previous_actions, batch_size=decisions
    )
    cls = (
        model.pool_cls(spatial)
        + float(model.cfg.history_action_scale) * action_state
    )
    lengths = [int(value) for value in game_lengths.cpu().tolist()]
    starts: list[int] = []
    cursor = 0
    by_length: dict[int, list[int]] = {}
    for length in lengths:
        starts.append(cursor)
        by_length.setdefault(length, []).append(cursor)
        cursor += length
    state_parts: list[torch.Tensor] = []
    row_parts: list[torch.Tensor] = []
    for length, group_starts in by_length.items():
        tokens = torch.stack(
            [cls[start : start + length] for start in group_starts],
            dim=0,
        )
        encoded, _ = model.temporal_encode(
            tokens, append=False, return_all=True
        )
        state_parts.append(encoded.reshape(-1, model.d_model))
        row_parts.append(
            torch.cat(
                [
                    torch.arange(
                        start,
                        start + length,
                        device=corpus.device,
                        dtype=torch.long,
                    )
                    for start in group_starts
                ]
            )
        )
    grouped_states = torch.cat(state_parts, dim=0)
    grouped_rows = torch.cat(row_parts, dim=0)
    state_by_decision = grouped_states.index_select(
        0, torch.argsort(grouped_rows)
    )
    state = state_by_decision.index_select(0, sample_state_rows)
    sample_spatial = spatial.index_select(0, sample_state_rows)
    decoded = model.decode_options_packed(
        options,
        sample_spatial,
        state,
        n_options=counts,
        batch_size=samples,
        decision_fusion_state_vec=state,
    )
    if isinstance(decoded, tuple):
        raise AssertionError("teacher policy decoder unexpectedly returned hidden")
    greedy = decoded.argmax(dim=1).to(dtype=torch.long)

    assert corpus.game_sample_offset is not None
    resident_game_ids = game_ids.reshape(-1).to(
        device=corpus.device, dtype=torch.long
    )
    sample_starts = corpus.game_sample_offset.index_select(
        0, resident_game_ids
    ).to(dtype=torch.long)
    sample_ends = corpus.game_sample_offset.index_select(
        0, resident_game_ids + 1
    ).to(dtype=torch.long)
    sample_ids, _sample_lengths = corpus._expand_ranges(  # noqa: SLF001
        sample_starts, sample_ends
    )
    if int(sample_ids.numel()) != samples:
        raise AssertionError("teacher policy sample mapping changed")
    if was_training:
        model.train()
    return sample_ids, greedy


def device_exact_value_predictions(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    sample_ids: torch.Tensor,
) -> torch.Tensor:
    """Value predictions for an exact resident batch without option gathers."""
    if model.decision_context != "stateless":
        raise ValueError("device-resident exact replay requires stateless context")
    board, _board_ids = corpus.board_batch(sample_ids)
    k = int(sample_ids.numel())
    spatial = model.encode_board_packed(board, batch_size=k)
    cls = model.pool_cls(spatial).unsqueeze(1)
    states, _ = model.temporal_encode(cls, append=False, return_all=True)
    state = states.squeeze(1)
    return torch.tanh(model.value_head(state)).squeeze(-1)


def device_exact_batch_losses(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    sample_ids: torch.Tensor,
    *,
    baseline_pred: Optional[torch.Tensor] = None,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    opp_hand_weight: float = 0.4,
    opp_remainder_weight: float = 0.3,
    lethal_threat_weight: float = 0.1,
    prize_race_weight: float = 0.1,
    alakazam_guide_weight: float = 0.0,
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY,
    awr_beta: float = 0.5,
    awr_weight_max: float = 20.0,
    awr_normalize_advantages: bool = True,
    entropy_bonus: float = 0.01,
) -> tuple[torch.Tensor, BatchMetrics]:
    """Full AWR + exact-hidden losses with all source tensors on-device."""
    if (
        canonical_guide_training_mode(current_deck_guide_training_mode)
        == GUIDE_TRAINING_MODE_STRATEGIC
    ):
        raise ValueError(
            "strategic curriculum requires the resident temporal training path"
        )
    if model.decision_context != "stateless":
        raise ValueError("device-resident exact replay requires stateless context")
    if not corpus.has_exact_targets:
        raise ValueError("resident corpus lacks exact-hidden targets")
    ids = sample_ids.reshape(-1).to(device=corpus.device, dtype=torch.long)
    board, options, counts, target_idx, v_target = corpus.batch(ids)
    k = int(target_idx.numel())
    if k <= 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, requires_grad=True), BatchMetrics()

    spatial = model.encode_board_packed(board, batch_size=k)
    cls = model.pool_cls(spatial).unsqueeze(1)
    states, _ = model.temporal_encode(cls, append=False, return_all=True)
    state = states.squeeze(1)
    logits = model.decode_options_packed(
        options,
        spatial,
        state,
        n_options=counts,
        batch_size=k,
    )
    value_pred = torch.tanh(model.value_head(state)).squeeze(-1)
    v_target = v_target.to(dtype=value_pred.dtype)
    log_p = torch.nan_to_num(F.log_softmax(logits, dim=-1), neginf=0.0)
    selected_log_p = log_p[
        torch.arange(k, device=logits.device), target_idx
    ]

    guide_loss = log_p.sum() * 0.0
    n_guide_rows = 0
    if float(alakazam_guide_weight) > 0.0:
        if not corpus.has_guide_targets:
            raise ValueError(
                "nonzero current-deck guide weight requires resident guide targets"
            )
        assert corpus.guide_target_index is not None
        assert corpus.guide_confidence is not None
        guide_target = corpus.guide_target_index.index_select(0, ids).to(
            dtype=torch.long
        )
        guide_confidence = corpus.guide_confidence.index_select(0, ids).to(
            dtype=log_p.dtype
        )
        invalid = (guide_target >= counts) & (guide_target >= 0)
        if bool(invalid.any()):
            bad_row = int(torch.nonzero(invalid, as_tuple=False)[0].item())
            raise ValueError(
                "resident current-deck guide target is outside option row: "
                f"target={int(guide_target[bad_row].item())} "
                f"options={int(counts[bad_row].item())}"
            )
        guide_mask = (
            (guide_target >= 0)
            & (guide_confidence > 0.0)
            & (counts >= 2)
        )
        if bool(guide_mask.any()):
            guide_rows = torch.nonzero(guide_mask, as_tuple=False).flatten()
            bounded_confidence = guide_confidence.index_select(
                0, guide_rows
            ).clamp(0.0, 1.0)
            guide_loss = -(
                log_p[
                    guide_rows,
                    guide_target.index_select(0, guide_rows),
                ]
                * bounded_confidence
            ).mean()
            n_guide_rows = int(guide_rows.numel())

    baseline = value_pred.detach() if baseline_pred is None else baseline_pred.detach()
    baseline = baseline.to(device=value_pred.device, dtype=value_pred.dtype).reshape(-1)
    if baseline.numel() != k:
        raise ValueError(f"AWR baseline rows {baseline.numel()} != batch rows {k}")
    raw_advantages = v_target - baseline
    raw_stats = raw_advantages.detach().float()
    advantages = raw_advantages
    if awr_normalize_advantages and k > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1e-6)
    beta = max(float(awr_beta), 1e-6)
    weight_max = max(float(awr_weight_max), 1e-6)
    raw_weights = torch.exp(advantages / beta)
    weights = torch.clamp(raw_weights, max=weight_max)
    policy_loss = -(weights.detach() * selected_log_p).mean()
    if float(entropy_bonus) > 0.0:
        probabilities = torch.nan_to_num(log_p.exp(), nan=0.0)
        entropy = -(probabilities * log_p).sum(dim=-1).mean()
        policy_loss = policy_loss - float(entropy_bonus) * entropy
    value_loss = F.smooth_l1_loss(value_pred, v_target)

    belief = model.belief_aux_logits(state)
    targets = corpus.exact_targets(ids)
    aux_labels = targets["aux_class"]
    aux_mask = aux_labels >= 0
    if bool(aux_mask.any()):
        aux_loss = F.cross_entropy(
            belief["aux_logits"][aux_mask], aux_labels[aux_mask]
        )
    else:
        aux_loss = belief["aux_logits"].sum() * 0.0
    hand_mask = targets["hand_mask"]
    if bool(hand_mask.any()):
        opp_hand_loss = F.binary_cross_entropy_with_logits(
            belief["opp_hand_logits"][hand_mask], targets["hand"][hand_mask]
        )
    else:
        opp_hand_loss = belief["opp_hand_logits"].sum() * 0.0
    remainder_mask = targets["remainder_mask"]
    if bool(remainder_mask.any()):
        opp_remainder_loss = F.binary_cross_entropy_with_logits(
            belief["opp_remainder_logits"][remainder_mask],
            targets["remainder"][remainder_mask],
        )
    else:
        opp_remainder_loss = belief["opp_remainder_logits"].sum() * 0.0
    lethal_mask = targets["lethal_mask"]
    if bool(lethal_mask.any()):
        lethal_threat_loss = F.binary_cross_entropy_with_logits(
            belief["lethal_threat_logits"][lethal_mask],
            targets["lethal"][lethal_mask].to(
                dtype=belief["lethal_threat_logits"].dtype
            ),
        )
    else:
        lethal_threat_loss = belief["lethal_threat_logits"].sum() * 0.0
    prize_mask = targets["prize_race_mask"]
    if bool(prize_mask.any()):
        prize_race_loss = F.smooth_l1_loss(
            belief["prize_race_pred"][prize_mask],
            targets["prize_race"][prize_mask].to(
                dtype=belief["prize_race_pred"].dtype
            ),
        )
    else:
        prize_race_loss = belief["prize_race_pred"].sum() * 0.0

    total = (
        policy_loss
        + float(value_weight) * value_loss
        + float(aux_weight) * aux_loss
        + float(opp_hand_weight) * opp_hand_loss
        + float(opp_remainder_weight) * opp_remainder_loss
        + float(lethal_threat_weight) * lethal_threat_loss
        + float(prize_race_weight) * prize_race_loss
        + float(alakazam_guide_weight) * guide_loss
    )
    predictions = logits.argmax(dim=1)
    weight_stats = weights.detach().float()
    weight_sum = float(weight_stats.sum().item())
    weight_sq_sum = float(weight_stats.square().sum().item())
    effective_samples = min(
        float(k), (weight_sum * weight_sum) / max(weight_sq_sum, 1e-12)
    )
    sorted_weights, _ = torch.sort(weight_stats)
    normalized_stats = advantages.detach().float()
    metrics = BatchMetrics(
        policy_loss=float(policy_loss.detach().item()),
        value_loss=float(value_loss.detach().item()),
        aux_loss=float(aux_loss.detach().item()),
        alakazam_guide_loss=float(guide_loss.detach().item()),
        opp_hand_loss=float(opp_hand_loss.detach().item()),
        opp_remainder_loss=float(opp_remainder_loss.detach().item()),
        lethal_threat_loss=float(lethal_threat_loss.detach().item()),
        prize_race_loss=float(prize_race_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=float((predictions == target_idx).float().mean().item()),
        target_value_mean=float(v_target.detach().float().mean().item()),
        value_pred_mean=float(value_pred.detach().float().mean().item()),
        n_decisions=k,
        n_alakazam_guide_rows=n_guide_rows,
        n_archetype_rows=int(aux_mask.sum().item()),
        n_opp_hand_rows=int(hand_mask.sum().item()),
        n_opp_remainder_rows=int(remainder_mask.sum().item()),
        n_lethal_threat_rows=int(lethal_mask.sum().item()),
        n_prize_race_rows=int(prize_mask.sum().item()),
        mean_advantage=float(raw_stats.abs().mean().item()),
        raw_advantage_mean=float(raw_stats.mean().item()),
        raw_advantage_std=float(raw_stats.std(unbiased=False).item()),
        raw_advantage_mean_abs=float(raw_stats.abs().mean().item()),
        raw_advantage_mean_sq=float(raw_stats.square().mean().item()),
        normalized_advantage_mean=float(normalized_stats.mean().item()),
        normalized_advantage_std=float(normalized_stats.std(unbiased=False).item()),
        normalized_advantage_mean_abs=float(normalized_stats.abs().mean().item()),
        normalized_advantage_mean_sq=float(normalized_stats.square().mean().item()),
        awr_weight_mean=float(weight_stats.mean().item()),
        awr_weight_sum=weight_sum,
        awr_weight_sq_sum=weight_sq_sum,
        awr_weight_p50=float(sorted_weights[int(0.50 * (k - 1))].item()),
        awr_weight_p95=float(sorted_weights[int(0.95 * (k - 1))].item()),
        awr_weight_max_observed=float(sorted_weights[-1].item()),
        awr_weight_clip_frac=float((raw_weights >= weight_max).float().mean().item()),
        awr_effective_sample_size=effective_samples,
        awr_effective_sample_fraction=effective_samples / max(k, 1),
        policy_selected_nll=float((-selected_log_p.detach()).mean().item()),
    )
    return total, metrics


def _merge_metrics(parts: Sequence[BatchMetrics]) -> BatchMetrics:
    if not parts:
        return BatchMetrics()
    nd = sum(p.n_decisions for p in parts)
    ng = sum(p.n_games for p in parts)
    if nd == 0:
        return BatchMetrics(n_games=ng)

    def wavg(attr: str) -> float:
        return sum(getattr(p, attr) * p.n_decisions for p in parts) / nd

    raw_mean = wavg("raw_advantage_mean")
    raw_mean_sq = wavg("raw_advantage_mean_sq")
    normalized_mean = wavg("normalized_advantage_mean")
    normalized_mean_sq = wavg("normalized_advantage_mean_sq")
    weight_sum = sum(p.awr_weight_sum for p in parts)
    weight_sq_sum = sum(p.awr_weight_sq_sum for p in parts)
    ess = min(float(nd), (weight_sum * weight_sum) / max(weight_sq_sum, 1e-12))
    # Non-AWR batches carry zero weight sums; avoid reporting a fictitious ESS.
    if weight_sum <= 0.0:
        ess = 0.0
    guide_rows = sum(int(p.n_alakazam_guide_rows) for p in parts)
    archetype_rows = sum(int(p.n_archetype_rows) for p in parts)
    hand_rows = sum(int(p.n_opp_hand_rows) for p in parts)
    remainder_rows = sum(int(p.n_opp_remainder_rows) for p in parts)
    lethal_rows = sum(int(p.n_lethal_threat_rows) for p in parts)
    prize_rows = sum(int(p.n_prize_race_rows) for p in parts)
    matchup_adapter_rows = sum(int(p.n_matchup_adapter_rows) for p in parts)
    teacher_policy_rows = sum(int(p.n_teacher_policy_rows) for p in parts)
    guide_loss = (
        sum(
            float(p.alakazam_guide_loss) * int(p.n_alakazam_guide_rows)
            for p in parts
        )
        / guide_rows
        if guide_rows > 0
        else 0.0
    )
    teacher_policy_loss = (
        sum(
            float(p.teacher_policy_loss) * int(p.n_teacher_policy_rows)
            for p in parts
        )
        / teacher_policy_rows
        if teacher_policy_rows > 0
        else 0.0
    )
    expanded_parts = [
        dict(part.expanded_head_metrics)
        for part in parts
        if part.expanded_head_metrics
    ]
    expanded_metrics: dict[str, Any] = {}
    if expanded_parts:
        head_ids = sorted(
            {
                str(name)
                for record in expanded_parts
                for name in dict(record.get("total") or {})
            }
        )
        labeled = {
            name: sum(
                int(dict(record.get("labeled") or {}).get(name, 0))
                for record in expanded_parts
            )
            for name in head_ids
        }
        total_rows = {
            name: sum(
                int(dict(record.get("total") or {}).get(name, 0))
                for record in expanded_parts
            )
            for name in head_ids
        }
        masked = {
            name: max(0, int(total_rows[name]) - int(labeled[name]))
            for name in head_ids
        }
        losses = {
            name: (
                sum(
                    float(dict(record.get("losses") or {}).get(name, 0.0))
                    * int(dict(record.get("labeled") or {}).get(name, 0))
                    for record in expanded_parts
                )
                / int(labeled[name])
                if int(labeled[name]) > 0
                else 0.0
            )
            for name in head_ids
        }
        outcome_rows = int(labeled.get("outcome_distribution", 0))

        def calibration_average(field_name: str) -> float | None:
            if outcome_rows <= 0:
                return None
            numerator = 0.0
            denominator = 0
            for record in expanded_parts:
                value = dict(record.get("calibration") or {}).get(field_name)
                rows = int(
                    dict(record.get("labeled") or {}).get(
                        "outcome_distribution", 0
                    )
                )
                if value is None or rows <= 0:
                    continue
                numerator += float(value) * rows
                denominator += rows
            return numerator / denominator if denominator else None

        expanded_metrics = {
            "losses": losses,
            "labeled": labeled,
            "masked": masked,
            "total": total_rows,
            "coverage": {
                name: float(labeled[name]) / max(int(total_rows[name]), 1)
                for name in head_ids
            },
            "calibration": {
                "outcome_brier": calibration_average("outcome_brier"),
                "outcome_ece": calibration_average("outcome_ece"),
                "outcome_entropy": calibration_average("outcome_entropy"),
            },
        }
    guide_curriculum_parts = [
        dict(part.guide_curriculum_head_metrics)
        for part in parts
        if part.guide_curriculum_head_metrics
    ]
    guide_curriculum_metrics: dict[str, Any] = {}
    if guide_curriculum_parts:
        head_ids = sorted(
            {
                str(name)
                for record in guide_curriculum_parts
                for name in dict(record.get("total") or {})
            }
        )
        curriculum_labeled = {
            name: sum(
                int(dict(record.get("labeled") or {}).get(name, 0))
                for record in guide_curriculum_parts
            )
            for name in head_ids
        }
        curriculum_total = {
            name: sum(
                int(dict(record.get("total") or {}).get(name, 0))
                for record in guide_curriculum_parts
            )
            for name in head_ids
        }
        guide_curriculum_metrics = {
            "losses": {
                name: (
                    sum(
                        float(
                            dict(record.get("losses") or {}).get(name, 0.0)
                        )
                        * int(
                            dict(record.get("labeled") or {}).get(name, 0)
                        )
                        for record in guide_curriculum_parts
                    )
                    / curriculum_labeled[name]
                    if curriculum_labeled[name] > 0
                    else 0.0
                )
                for name in head_ids
            },
            "labeled": curriculum_labeled,
            "total": curriculum_total,
            "masked": {
                name: max(
                    0, curriculum_total[name] - curriculum_labeled[name]
                )
                for name in head_ids
            },
            "coverage": {
                name: curriculum_labeled[name]
                / max(curriculum_total[name], 1)
                for name in head_ids
            },
        }
    setup_parts = [
        dict(part.setup_board_outcome_metrics)
        for part in parts
        if part.setup_board_outcome_metrics
    ]
    setup_metrics: dict[str, Any] = {}
    if setup_parts:
        setup_metrics = {
            "total_rows": sum(
                int(record.get("total_rows", 0)) for record in setup_parts
            ),
            "eligible_rows": sum(
                int(record.get("eligible_rows", 0)) for record in setup_parts
            ),
            "guide_rows": sum(
                int(record.get("guide_rows", 0)) for record in setup_parts
            ),
            "stop_rows": sum(
                int(record.get("stop_rows", 0)) for record in setup_parts
            ),
            "non_stop_rows": sum(
                int(record.get("non_stop_rows", 0)) for record in setup_parts
            ),
            "context_rows": {
                name: sum(
                    int(dict(record.get("context_rows") or {}).get(name, 0))
                    for record in setup_parts
                )
                for name in ("setup_active", "setup_bench")
            },
        }
    combo_parts = [
        dict(part.combo_state_metrics)
        for part in parts
        if part.combo_state_metrics
    ]
    combo_metrics: dict[str, Any] = {}
    if combo_parts:
        combo_metrics = {
            "total_rows": sum(int(row.get("total_rows", 0)) for row in combo_parts),
            "eligible_rows": sum(int(row.get("eligible_rows", 0)) for row in combo_parts),
            "top_deck_labels": sum(int(row.get("top_deck_labels", 0)) for row in combo_parts),
            "seek_source_labels": sum(int(row.get("seek_source_labels", 0)) for row in combo_parts),
            "guide_rows": sum(int(row.get("guide_rows", 0)) for row in combo_parts),
            "vector_labels": {
                name: sum(
                    int(dict(row.get("vector_labels") or {}).get(name, 0))
                    for row in combo_parts
                )
                for name in (
                    "copied_attack_legality",
                    "visible_combo_piece_availability",
                    "energy_route_readiness",
                    "bench_continuity",
                )
            },
        }

    return BatchMetrics(
        policy_loss=wavg("policy_loss"),
        teacher_policy_loss=teacher_policy_loss,
        value_loss=wavg("value_loss"),
        aux_loss=wavg("aux_loss"),
        alakazam_guide_loss=guide_loss,
        guide_strategic_curriculum_loss=wavg(
            "guide_strategic_curriculum_loss"
        ),
        setup_board_outcome_loss=wavg("setup_board_outcome_loss"),
        combo_state_loss=wavg("combo_state_loss"),
        opp_hand_loss=wavg("opp_hand_loss"),
        opp_remainder_loss=wavg("opp_remainder_loss"),
        lethal_threat_loss=wavg("lethal_threat_loss"),
        prize_race_loss=wavg("prize_race_loss"),
        history_identity_loss=wavg("history_identity_loss"),
        total_loss=wavg("total_loss"),
        policy_acc=wavg("policy_acc"),
        policy_kl=wavg("policy_kl"),
        target_value_mean=wavg("target_value_mean"),
        value_pred_mean=wavg("value_pred_mean"),
        n_decisions=nd,
        n_games=ng,
        n_alakazam_guide_rows=guide_rows,
        n_archetype_rows=archetype_rows,
        n_opp_hand_rows=hand_rows,
        n_opp_remainder_rows=remainder_rows,
        n_lethal_threat_rows=lethal_rows,
        n_prize_race_rows=prize_rows,
        n_matchup_adapter_rows=matchup_adapter_rows,
        n_teacher_policy_rows=teacher_policy_rows,
        expanded_head_metrics=expanded_metrics,
        guide_curriculum_head_metrics=guide_curriculum_metrics,
        setup_board_outcome_metrics=setup_metrics,
        combo_state_metrics=combo_metrics,
        mean_advantage=wavg("mean_advantage"),
        raw_advantage_mean=raw_mean,
        raw_advantage_std=math.sqrt(max(raw_mean_sq - raw_mean * raw_mean, 0.0)),
        raw_advantage_mean_abs=wavg("raw_advantage_mean_abs"),
        raw_advantage_mean_sq=raw_mean_sq,
        normalized_advantage_mean=normalized_mean,
        normalized_advantage_std=math.sqrt(
            max(normalized_mean_sq - normalized_mean * normalized_mean, 0.0)
        ),
        normalized_advantage_mean_abs=wavg("normalized_advantage_mean_abs"),
        normalized_advantage_mean_sq=normalized_mean_sq,
        awr_weight_mean=wavg("awr_weight_mean"),
        awr_weight_sum=weight_sum,
        awr_weight_sq_sum=weight_sq_sum,
        awr_weight_p50=wavg("awr_weight_p50"),
        awr_weight_p95=wavg("awr_weight_p95"),
        awr_weight_max_observed=max(
            float(p.awr_weight_max_observed) for p in parts
        ),
        awr_weight_clip_frac=wavg("awr_weight_clip_frac"),
        awr_effective_sample_size=ess,
        awr_effective_sample_fraction=ess / nd,
        policy_selected_nll=wavg("policy_selected_nll"),
    )


def _set_exact_awr_weight_quantiles(
    metrics: BatchMetrics, values: Sequence[float]
) -> BatchMetrics:
    """Replace batch-aggregated AWR quantiles with exact epoch quantiles."""
    if not values:
        return metrics
    ordered = sorted(float(value) for value in values)
    last = len(ordered) - 1
    metrics.awr_weight_p50 = ordered[int(0.50 * last)]
    metrics.awr_weight_p95 = ordered[int(0.95 * last)]
    metrics.awr_weight_max_observed = ordered[-1]
    return metrics


def split_dataset(
    ds: BootstrapDataset,
    val_frac: float,
    seed: int,
    *,
    group_by_episode: bool = False,
) -> tuple[list[GameSequence], list[GameSequence]]:
    seqs = list(ds.sequences)
    rng = random.Random(seed)
    if group_by_episode and val_frac > 0 and len(seqs) > 1:
        grouped: dict[str, list[GameSequence]] = {}
        sequence_group: dict[int, str] = {}
        for index, seq in enumerate(seqs):
            episode_id = str(seq.episode_id or f"__missing_episode_{index}")
            grouped.setdefault(episode_id, []).append(seq)
            sequence_group[id(seq)] = episode_id
        group_ids = list(grouped)
        rng.shuffle(group_ids)
        if len(group_ids) <= 1:
            return seqs, []
        target = max(1, int(len(seqs) * val_frac))
        val_ids: set[str] = set()
        selected = 0
        # Always leave at least one whole episode in training.
        for episode_id in group_ids[:-1]:
            val_ids.add(episode_id)
            selected += len(grouped[episode_id])
            if selected >= target:
                break
        train = [seq for seq in seqs if sequence_group[id(seq)] not in val_ids]
        val = [seq for seq in seqs if sequence_group[id(seq)] in val_ids]
        return train, val
    rng.shuffle(seqs)
    if val_frac <= 0 or len(seqs) <= 1:
        return seqs, []
    n_val = max(1, int(len(seqs) * val_frac)) if len(seqs) > 1 else 0
    if n_val == 0:
        return seqs, []
    return seqs[n_val:], seqs[:n_val]


def cap_game_sequence_context(
    sequence: GameSequence,
    max_context: int,
) -> tuple[GameSequence, bool]:
    """Keep a causal game prefix within the temporal attention cap.

    The sequence remains owned by exactly one acting seat and one episode.
    A rare game longer than the measured 320-step window contributes its
    correctly conditioned prefix; no reset chunk or cross-game packing is
    introduced merely to retain the tail.
    """
    cap = int(max_context)
    if cap <= 0:
        raise ValueError("history max_context must be positive")
    if len(sequence.decisions) <= cap:
        return sequence, False

    def _prefix(value):
        return value[:cap] if value is not None else None

    return (
        replace(
            sequence,
            decisions=list(sequence.decisions[:cap]),
            policy_targets=_prefix(sequence.policy_targets),
            factorized_policy_targets=_prefix(
                sequence.factorized_policy_targets
            ),
        ),
        True,
    )


def cap_history_sequences(
    sequences: Sequence[GameSequence],
    max_context: int,
) -> tuple[list[GameSequence], int]:
    capped: list[GameSequence] = []
    truncated = 0
    for sequence in sequences:
        row, changed = cap_game_sequence_context(sequence, max_context)
        capped.append(row)
        truncated += int(changed)
    return capped, truncated


def _iter_game_batches(
    sequences: list[GameSequence],
    games_per_batch: int,
    max_decisions: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[GameSequence]]:
    order = list(range(len(sequences)))
    if shuffle:
        rng = random.Random(seed + epoch * 10007)
        rng.shuffle(order)
    batches: list[list[GameSequence]] = []
    cur: list[GameSequence] = []
    cur_dec = 0
    for i in order:
        seq = sequences[i]
        n = len(seq)
        if cur and (
            len(cur) >= games_per_batch or cur_dec + n > max_decisions
        ):
            batches.append(cur)
            cur, cur_dec = [], 0
        cur.append(seq)
        cur_dec += n
    if cur:
        batches.append(cur)
    return batches


def split_matchup_adapter_sequences(
    sequences: Sequence[GameSequence],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[GameSequence], list[GameSequence]]:
    """Episode-disjoint, route-stratified split for all oracle partitions."""

    by_route: dict[int, dict[str, list[GameSequence]]] = {
        route: {} for route in range(len(EXPERT_IDS))
    }
    episode_route: dict[str, int] = {}
    for sequence in sequences:
        if not sequence.decisions:
            continue
        sequence_routes = training_routes_for_sequence(sequence)
        route = sequence_routes[0]
        if route == UNKNOWN_ROUTE:
            raise RuntimeError("adapter split received an unknown oracle route")
        if any(candidate != route for candidate in sequence_routes):
            raise RuntimeError("one adapter sequence contains multiple oracle routes")
        episode_id = str(sequence.episode_id)
        prior = episode_route.setdefault(episode_id, route)
        if prior != route:
            raise RuntimeError("one episode appears in multiple adapter routes")
        by_route[route].setdefault(episode_id, []).append(sequence)

    train_rows: list[GameSequence] = []
    val_rows: list[GameSequence] = []
    for route, archetype_id in enumerate(EXPERT_IDS):
        groups = list(by_route[route].items())
        if len(groups) < 2:
            raise RuntimeError(
                f"adapter route {archetype_id} needs at least two episodes"
            )
        random.Random(int(seed) + route * 100_003).shuffle(groups)
        n_val = max(1, min(len(groups) - 1, round(len(groups) * float(val_frac))))
        val_ids = {episode_id for episode_id, _rows in groups[:n_val]}
        for episode_id, rows in groups:
            (val_rows if episode_id in val_ids else train_rows).extend(rows)
    return train_rows, val_rows


def matchup_adapter_split_contract(
    train_sequences: Sequence[GameSequence],
    val_sequences: Sequence[GameSequence],
) -> dict[str, Any]:
    """Immutable per-route coverage and membership identity for checkpointing."""

    corpus_digests: set[str] = set()
    gate_digests: set[str] = set()
    membership: list[dict[str, Any]] = []
    route_rows: dict[str, dict[str, int]] = {
        archetype_id: {
            "train_sequences": 0,
            "train_decisions": 0,
            "val_sequences": 0,
            "val_decisions": 0,
        }
        for archetype_id in EXPERT_IDS
    }
    split_episode_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    for split, sequences in (("train", train_sequences), ("val", val_sequences)):
        for sequence in sequences:
            ticket = adapter_training_ticket(sequence)
            corpus_digests.add(ticket.corpus_manifest_digest)
            gate_digests.add(ticket.gate_contract_digest)
            route_rows[ticket.archetype_id][f"{split}_sequences"] += 1
            route_rows[ticket.archetype_id][f"{split}_decisions"] += len(
                sequence.decisions
            )
            split_episode_ids[split].add(str(sequence.episode_id))
            membership.append(
                {
                    "split": split,
                    "route": int(ticket.route),
                    "archetype_id": ticket.archetype_id,
                    "package_digest": ticket.package_digest,
                    "episode_id": str(sequence.episode_id),
                    "seat": int(sequence.seat),
                    "decisions": len(sequence.decisions),
                }
            )
    if split_episode_ids["train"] & split_episode_ids["val"]:
        raise RuntimeError("adapter train/validation episode membership overlaps")
    if len(corpus_digests) != 1 or len(gate_digests) != 1:
        raise RuntimeError("adapter split combines different corpus/gate contracts")
    missing = [
        archetype_id
        for archetype_id, row in route_rows.items()
        if any(int(row[field]) <= 0 for field in row)
    ]
    if missing:
        raise RuntimeError(
            f"adapter split lacks train/validation coverage for {missing}"
        )
    membership.sort(
        key=lambda row: (
            row["split"],
            row["route"],
            row["episode_id"],
            row["seat"],
        )
    )
    membership_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            membership,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "poke_bot.matchup_adapter_training_split/v1",
        "routing": "offline-oracle-package-and-full-deck-audited",
        "runtime_router_separate": True,
        "corpus_manifest_digest": next(iter(corpus_digests)),
        "active_gate_contract_digest": next(iter(gate_digests)),
        "membership_digest": membership_digest,
        "per_route": route_rows,
    }


def build_matchup_adapter_training_contract(
    train_sequences: Sequence[GameSequence],
    val_sequences: Sequence[GameSequence],
    *,
    input_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Bind exact rows, source artifacts, routing order, and loss isolation.

    Resume and dormant merge compare this mapping exactly.  A corpus rewrite,
    active-gate change, route reorder, or train/validation membership change
    therefore cannot silently continue an older optimizer.
    """

    inputs = copy.deepcopy(dict(input_provenance or {}))
    if inputs.get("schema") != "poke_bot.matchup_adapter_input_provenance/v1":
        raise ValueError("adapter training lacks exact input provenance")
    required_digests = (
        "source_jsonl_digest",
        "corpus_manifest_file_digest",
        "active_gate_contract_file_digest",
        "implementation_digest",
    )
    for field_name in required_digests:
        value = str(inputs.get(field_name) or "")
        if not value.startswith("sha256:") or len(value) != 71:
            raise ValueError(f"adapter input provenance lacks {field_name}")
    split = matchup_adapter_split_contract(train_sequences, val_sequences)
    return {
        "schema": "poke_bot.matchup_adapter_training_contract/v1",
        "routing": "offline-oracle-package-and-full-deck-audited",
        "runtime_router_separate": True,
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "loss_scope": ["policy", "value"],
        "expert_ids": list(EXPERT_IDS),
        "adapter_config": model_matchup_adapter_config(),
        "corpus_manifest_digest": split["corpus_manifest_digest"],
        "active_gate_contract_digest": split["active_gate_contract_digest"],
        "split": split,
        "inputs": inputs,
    }


def model_matchup_adapter_config() -> dict[str, Any]:
    """Late import-free accessor used by immutable training metadata."""

    from .matchup_adapters import MatchupAdapterBank

    return MatchupAdapterBank.config_dict()


@torch.no_grad()
def evaluate(
    model: TemporalCabtTransformer,
    sequences: list[GameSequence],
    *,
    cfg: TrainConfig,
    desc: str = "val",
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None,
) -> BatchMetrics:
    model.eval()
    parts: list[BatchMetrics] = []
    exact_awr_weights: Optional[list[float]] = (
        [] if cfg.capture_awr_weight_distribution else None
    )
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    use_amp = device_mod.cuda_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    for batch in tqdm(batches, desc=desc, leave=False, unit="batch"):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _, m = batch_losses(
                model,
                batch,
                value_weight=cfg.value_loss_weight,
                aux_weight=cfg.aux_loss_weight,
                opp_hand_weight=cfg.opp_hand_loss_weight,
                opp_remainder_weight=cfg.opp_remainder_loss_weight,
                lethal_threat_weight=cfg.lethal_threat_loss_weight,
                prize_race_weight=cfg.prize_race_loss_weight,
                alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                current_deck_guide_training_mode=(
                    cfg.current_deck_guide_training_mode
                ),
                setup_board_outcome_loss_weight=(
                    cfg.setup_board_outcome_loss_weight
                ),
                combo_state_loss_weight=cfg.combo_state_loss_weight,
                expanded_head_weights=cfg.expanded_head_loss_weights,
                pure_rl=bool(cfg.pure_rl),
                awr_beta=float(cfg.awr_beta),
                awr_weight_max=float(cfg.awr_weight_max),
                awr_normalize_advantages=bool(cfg.awr_normalize_advantages),
                entropy_bonus=float(cfg.entropy_bonus),
                awr_baseline_cache=awr_baseline_cache,
                awr_weight_sink=exact_awr_weights,
                history_identity_weight=float(
                    cfg.history_identity_loss_weight
                ),
                matchup_adapter_training=bool(cfg.matchup_adapter_training),
            )
        parts.append(m)
    return _set_exact_awr_weight_quantiles(
        _merge_metrics(parts), exact_awr_weights or ()
    )


@torch.no_grad()
def evaluate_device_corpus(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    desc: str = "val",
    teacher_policy_targets: Optional[torch.Tensor] = None,
    teacher_policy_weight: float = 0.0,
) -> BatchMetrics:
    """Evaluate the validation partition without moving inputs off device."""
    model.eval()
    parts: list[BatchMetrics] = []
    temporal = model.decision_context == "history"
    batches = (
        corpus.temporal_batches(
            train=False,
            batch_size=batch_size,
            shuffle=False,
            seed=cfg.seed,
            epoch=0,
        )
        if temporal
        else corpus.batches(
            train=False,
            batch_size=batch_size,
            shuffle=False,
            seed=cfg.seed,
            epoch=0,
        )
    )
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    for batch_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            if temporal:
                _, metrics = device_temporal_batch_losses(
                    model,
                    corpus,
                    batch_ids,
                    value_weight=cfg.value_loss_weight,
                    aux_weight=cfg.aux_loss_weight,
                    opp_hand_weight=cfg.opp_hand_loss_weight,
                    opp_remainder_weight=cfg.opp_remainder_loss_weight,
                    lethal_threat_weight=cfg.lethal_threat_loss_weight,
                    prize_race_weight=cfg.prize_race_loss_weight,
                    alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                    current_deck_guide_training_mode=(
                        cfg.current_deck_guide_training_mode
                    ),
                    setup_board_outcome_loss_weight=(
                        cfg.setup_board_outcome_loss_weight
                    ),
                    combo_state_loss_weight=cfg.combo_state_loss_weight,
                    expanded_head_weights=cfg.expanded_head_loss_weights,
                    teacher_policy_targets=teacher_policy_targets,
                    teacher_policy_weight=teacher_policy_weight,
                )
            else:
                _, metrics = device_batch_losses(
                    model,
                    corpus,
                    batch_ids,
                    value_weight=cfg.value_loss_weight,
                    current_deck_guide_training_mode=(
                        cfg.current_deck_guide_training_mode
                    ),
                )
        parts.append(metrics)
    return _merge_metrics(parts)


@torch.no_grad()
def _device_exact_value_cache(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    """Freeze V(s) for every resident train/validation sample on-device."""
    was_training = model.training
    model.eval()
    values = torch.empty(
        corpus.total_samples,
        device=corpus.device,
        dtype=torch.float32,
    )
    ids = torch.arange(
        corpus.total_samples, device=corpus.device, dtype=torch.long
    )
    batches = list(ids.split(max(1, int(batch_size))))
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    try:
        for sample_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                prediction = device_exact_value_predictions(
                    model, corpus, sample_ids
                )
            values.index_copy_(0, sample_ids, prediction.float())
    finally:
        model.train(was_training)
    return values


@torch.no_grad()
def _device_exact_policy_predictions(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    """Stable argmax policy rows for resident parent/candidate agreement."""
    was_training = model.training
    model.eval()
    predictions = torch.empty(
        corpus.total_samples,
        device=corpus.device,
        dtype=torch.int32,
    )
    ids = torch.arange(
        corpus.total_samples, device=corpus.device, dtype=torch.long
    )
    batches = list(ids.split(max(1, int(batch_size))))
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    try:
        for sample_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                board, options, counts, _target, _value = corpus.batch(
                    sample_ids
                )
                k = int(sample_ids.numel())
                spatial = model.encode_board_packed(board, batch_size=k)
                cls = model.pool_cls(spatial).unsqueeze(1)
                states, _ = model.temporal_encode(
                    cls, append=False, return_all=True
                )
                logits = model.decode_options_packed(
                    options,
                    spatial,
                    states.squeeze(1),
                    n_options=counts,
                    batch_size=k,
                )
            predictions.index_copy_(
                0, sample_ids, logits.argmax(dim=1).to(dtype=torch.int32)
            )
    finally:
        model.train(was_training)
    return predictions


@torch.no_grad()
def _evaluate_device_exact_corpus(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    cfg: TrainConfig,
    batch_size: int,
    baseline: torch.Tensor,
    desc: str,
) -> BatchMetrics:
    """Evaluate exact resident validation rows with the frozen AWR baseline."""
    was_training = model.training
    model.eval()
    parts: list[BatchMetrics] = []
    batches = corpus.batches(
        train=False,
        batch_size=batch_size,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    use_amp = bool(cfg.amp and corpus.device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    try:
        for sample_ids in tqdm(batches, desc=desc, leave=False, unit="batch"):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                _, metrics = device_exact_batch_losses(
                    model,
                    corpus,
                    sample_ids,
                    baseline_pred=baseline.index_select(0, sample_ids),
                    value_weight=cfg.value_loss_weight,
                    aux_weight=cfg.aux_loss_weight,
                    opp_hand_weight=cfg.opp_hand_loss_weight,
                    opp_remainder_weight=cfg.opp_remainder_loss_weight,
                    lethal_threat_weight=cfg.lethal_threat_loss_weight,
                    prize_race_weight=cfg.prize_race_loss_weight,
                    alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                    current_deck_guide_training_mode=(
                        cfg.current_deck_guide_training_mode
                    ),
                    awr_beta=cfg.awr_beta,
                    awr_weight_max=cfg.awr_weight_max,
                    awr_normalize_advantages=cfg.awr_normalize_advantages,
                    entropy_bonus=cfg.entropy_bonus,
                )
            parts.append(metrics)
    finally:
        model.train(was_training)
    return _merge_metrics(parts)


def _fit_device_batch_size(
    model: TemporalCabtTransformer,
    corpus: DeviceResidentBootstrapCorpus,
    *,
    requested: int,
    cfg: TrainConfig,
    use_amp: bool,
    amp_dtype: torch.dtype,
    min_step_free_gib: float = 2.0,
) -> int:
    """Prove corpus + forward/backward fit, reducing only after a real OOM."""
    size = min(max(1, int(requested)), corpus.train_samples)
    if size <= 0:
        raise ValueError("device corpus has no training samples")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(corpus.device) if corpus.device.type == "cuda" else None
    try:
        while True:
            # Include the globally widest training row in the probe.  This
            # exercises the decoder's rectangular padding at its actual worst
            # width instead of approving an unusually easy first batch.
            temporal = model.decision_context == "history"
            if temporal:
                batch_ids = corpus.temporal_probe_batch(size)
            else:
                batch_ids = torch.arange(
                    size, device=corpus.device, dtype=torch.long
                )
                widest = torch.argmax(
                    corpus.n_options[: corpus.train_samples]
                ).to(dtype=torch.long)
                batch_ids[-1] = widest
            if corpus.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(corpus.device)
            model.train()
            model.zero_grad(set_to_none=True)
            oom = False
            try:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    if temporal:
                        total, _ = device_temporal_batch_losses(
                            model,
                            corpus,
                            batch_ids,
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                            opp_hand_weight=cfg.opp_hand_loss_weight,
                            opp_remainder_weight=cfg.opp_remainder_loss_weight,
                            lethal_threat_weight=cfg.lethal_threat_loss_weight,
                            prize_race_weight=cfg.prize_race_loss_weight,
                            alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                            current_deck_guide_training_mode=(
                                cfg.current_deck_guide_training_mode
                            ),
                            setup_board_outcome_loss_weight=(
                                cfg.setup_board_outcome_loss_weight
                            ),
                            combo_state_loss_weight=cfg.combo_state_loss_weight,
                            expanded_head_weights=(
                                cfg.expanded_head_loss_weights
                            ),
                        )
                    elif cfg.pure_rl and corpus.has_exact_targets:
                        total, _ = device_exact_batch_losses(
                            model,
                            corpus,
                            batch_ids,
                            baseline_pred=torch.zeros(
                                size,
                                device=corpus.device,
                                dtype=torch.float32,
                            ),
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                            opp_hand_weight=cfg.opp_hand_loss_weight,
                            opp_remainder_weight=cfg.opp_remainder_loss_weight,
                            lethal_threat_weight=cfg.lethal_threat_loss_weight,
                            prize_race_weight=cfg.prize_race_loss_weight,
                            alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                            current_deck_guide_training_mode=(
                                cfg.current_deck_guide_training_mode
                            ),
                            awr_beta=cfg.awr_beta,
                            awr_weight_max=cfg.awr_weight_max,
                            awr_normalize_advantages=(
                                cfg.awr_normalize_advantages
                            ),
                            entropy_bonus=cfg.entropy_bonus,
                        )
                    else:
                        total, _ = device_batch_losses(
                            model,
                            corpus,
                            batch_ids,
                            value_weight=cfg.value_loss_weight,
                            current_deck_guide_training_mode=(
                                cfg.current_deck_guide_training_mode
                            ),
                        )
                total.backward()
                if corpus.device.type == "cuda":
                    torch.cuda.synchronize(corpus.device)
            except RuntimeError as exc:
                if not config.is_cuda_oom(exc):
                    raise
                oom = True
            if not oom:
                model.zero_grad(set_to_none=True)
                if corpus.device.type == "cuda":
                    peak = torch.cuda.max_memory_allocated(corpus.device)
                    free, total_memory = torch.cuda.mem_get_info(corpus.device)
                    if free < int(float(min_step_free_gib) * 2**30):
                        oom = True
                        print(
                            f"[device-corpus] fit-test batch={size} rejected: "
                            f"post-step CUDA-free={free / 2**30:.2f} GiB < "
                            f"required {float(min_step_free_gib):.2f} GiB",
                            flush=True,
                        )
                if not oom:
                    if corpus.device.type == "cuda":
                        free, total_memory = torch.cuda.mem_get_info(corpus.device)
                        peak = torch.cuda.max_memory_allocated(corpus.device)
                        print(
                            f"[device-corpus] fit-test batch={size} passed "
                            f"peak-allocated={peak / 2**30:.2f} GiB "
                            f"CUDA-free={free / 2**30:.2f}/"
                            f"{total_memory / 2**30:.2f} GiB",
                            flush=True,
                        )
                    return size

            model.zero_grad(set_to_none=True)
            del batch_ids
            gc.collect()
            if corpus.device.type == "cuda":
                torch.cuda.empty_cache()
            reduced = size // 2
            if reduced < 128:
                raise MemoryError(
                    "device-resident corpus fits, but even a 128-sample "
                    "training batch exhausts activation memory"
                )
            print(
                f"[device-corpus] fit-test OOM at batch={size}; retrying "
                f"batch={reduced}",
                flush=True,
            )
            size = reduced
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, corpus.device)
        model.zero_grad(set_to_none=True)
        if corpus.device.type == "cuda":
            torch.cuda.empty_cache()


def train_bootstrap(
    dataset: BootstrapDataset,
    *,
    run_name: str = "dragapult_bootstrap",
    archetype_id: str = "dragapult",
    train_cfg: Optional[TrainConfig] = None,
    resume: Union[str, bool, None] = "auto",
    device: Optional[torch.device] = None,
    model_cfg: Optional[config.ModelConfig] = None,
    init_checkpoint: Optional[Union[str, Path]] = None,
    checkpoint_extra: Optional[dict[str, Any]] = None,
    device_resident: bool = False,
    trainable_parameter_prefixes: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Run supervised BC/value training with early stopping + AMP + checkpoints."""
    cfg = train_cfg or TrainConfig()
    device = device or device_mod.training_device(
        prefer_name=config.HARDWARE.train_gpu_name, allow_cpu=False
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.matchup_adapter_training:
        train_seqs, val_seqs = split_matchup_adapter_sequences(
            dataset.sequences,
            val_frac=cfg.val_frac,
            seed=cfg.seed,
        )
    else:
        train_seqs, val_seqs = split_dataset(
            dataset,
            cfg.val_frac,
            cfg.seed,
            group_by_episode=bool(cfg.split_by_episode),
        )
    print(
        f"[train] device={device} games={len(dataset)} "
        f"train={len(train_seqs)} val={len(val_seqs)} decisions={dataset.n_decisions}",
        flush=True,
    )

    resume_path = checkpoint.resolve_resume_path(run_name, resume)
    init_path = Path(init_checkpoint).expanduser().resolve() if init_checkpoint else None
    if (
        resume_path is not None
        and init_path is not None
        and not cfg.matchup_adapter_training
    ):
        raise ValueError("init_checkpoint cannot be combined with a resumed run")
    if init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {init_path}")
    source_path = resume_path or init_path
    model = (
        load_model_from_checkpoint(source_path, device=device)
        if source_path is not None
        else build_model(model_cfg or config.MODEL, device=device)
    )
    cfg.current_deck_guide_training_mode = canonical_guide_training_mode(
        cfg.current_deck_guide_training_mode
    )
    if (
        cfg.current_deck_guide_training_mode
        == GUIDE_TRAINING_MODE_STRATEGIC
    ):
        assert_strategic_curriculum_model_contract(
            model,
            setup_board_outcome_loss_weight=(
                cfg.setup_board_outcome_loss_weight
            ),
        )
        assert_strategic_curriculum_receipt_contract(
            specialist_id=archetype_id,
            curriculum_spec=cfg.current_deck_guide_curriculum_spec,
            head_role_map=cfg.current_deck_guide_head_role_map,
            validation_receipt=(
                cfg.current_deck_guide_curriculum_validation_receipt
            ),
        )
        if device_resident:
            raise ValueError(
                "strategic curriculum bootstrap requires the temporal host path"
            )
        if (
            float(cfg.alakazam_guide_loss_weight) > 0.0
            and count_usable_strategic_guide_rows(
                [*train_seqs, *val_seqs]
            )
            <= 0
        ):
            raise ValueError(
                "nonzero strategic guide multiplier has no confidence-bearing "
                "bootstrap rows"
            )
    if model_cfg is not None and getattr(model, "cfg", None) != model_cfg:
        raise ValueError("initial checkpoint model profile does not match model_cfg")
    if model.decision_context == "history":
        train_seqs, train_truncated = cap_history_sequences(
            train_seqs, model.max_context
        )
        val_seqs, val_truncated = cap_history_sequences(
            val_seqs, model.max_context
        )
        print(
            f"[train] game-bounded temporal context={model.max_context} "
            f"truncated_sequences={train_truncated + val_truncated}",
            flush=True,
        )
    adapter_training_contract: Optional[dict[str, Any]] = None
    if cfg.matchup_adapter_training:
        input_provenance = dict(
            (checkpoint_extra or {}).get("matchup_adapter_input_provenance") or {}
        )
        adapter_training_contract = build_matchup_adapter_training_contract(
            train_seqs,
            val_seqs,
            input_provenance=input_provenance,
        )
    trainable_prefixes = tuple(
        str(value) for value in (trainable_parameter_prefixes or ())
    )
    frozen_base_snapshot: Optional[dict[str, torch.Tensor]] = None
    adapter_activation: Optional[ActivationReceipt] = None
    adapter_parent_path: Optional[Path] = None
    if cfg.matchup_adapter_training:
        if trainable_prefixes:
            raise ValueError(
                "matchup adapter training cannot be combined with generic "
                "trainable_parameter_prefixes"
            )
        if device_resident:
            raise ValueError(
                "matchup adapter training requires host GameSequence matchup labels"
            )
        if cfg.pure_rl:
            raise ValueError("matchup adapter training is bootstrap-only")
        nonzero_auxiliary = {
            name: float(value)
            for name, value in {
                "aux": cfg.aux_loss_weight,
                "opp_hand": cfg.opp_hand_loss_weight,
                "opp_remainder": cfg.opp_remainder_loss_weight,
                "guide": cfg.alakazam_guide_loss_weight,
                "lethal": cfg.lethal_threat_loss_weight,
                "prize": cfg.prize_race_loss_weight,
                "history_identity": cfg.history_identity_loss_weight,
            }.items()
            if float(value) != 0.0
        }
        if nonzero_auxiliary:
            raise ValueError(
                "matchup adapter fitting permits policy/value losses only; "
                f"nonzero auxiliary weights={nonzero_auxiliary}"
            )
        if source_path is None:
            raise ValueError(
                "matchup adapter fitting requires the pinned iteration-15 parent"
            )
        if resume_path is not None:
            resume_payload = checkpoint.load_checkpoint(
                resume_path, map_location="cpu"
            )
            resume_extra = dict(resume_payload.get("extra") or {})
            raw_parent = str(
                resume_extra.get("matchup_adapter_parent_checkpoint") or ""
            )
            if not raw_parent:
                raise ValueError(
                    "adapter resume checkpoint lacks its frozen parent identity"
                )
            adapter_parent_path = Path(raw_parent).expanduser().resolve()
            if init_path is not None and init_path != adapter_parent_path:
                raise ValueError(
                    "adapter resume --init-checkpoint differs from its pinned parent"
                )
        else:
            assert init_path is not None
            adapter_parent_path = init_path
        if not str(cfg.matchup_adapter_activation_receipt).strip():
            raise ValueError(
                "matchup adapter fitting requires an activation receipt"
            )
        adapter_activation = validate_adapter_training_authorization(
            cfg.matchup_adapter_activation_receipt,
            parent_checkpoint=adapter_parent_path,
        )
        optimizer = build_matchup_adapter_optimizer(
            model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            activation_receipt=adapter_activation,
        )
        assert adapter_parent_path is not None
        assert_matchup_adapter_parent_identity(
            model,
            parent_checkpoint=adapter_parent_path,
        )
        trainable_parameters = list(model.matchup_adapter_bank.parameters())
        frozen_base_snapshot = matchup_adapter_base_state(model)
        print(
            "[train] ground-truth matchup adapter mode base=frozen "
            f"adapter_params={sum(p.numel() for p in trainable_parameters)}",
            flush=True,
        )
    else:
        if trainable_prefixes:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(
                    any(name.startswith(prefix) for prefix in trainable_prefixes)
                )
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("training freeze policy left no trainable parameters")
        if trainable_prefixes:
            print(
                "[train] frozen-copy calibration trainable_prefixes="
                f"{list(trainable_prefixes)} params="
                f"{sum(parameter.numel() for parameter in trainable_parameters)}",
                flush=True,
            )
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    # Blackwell throughput: allow TF32 matmuls and prefer bf16 autocast.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    # bf16 carries fp32 dynamic range → no GradScaler needed; only fp16 needs it.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    if use_amp:
        print(f"[train] AMP dtype={amp_dtype} scaler={scaler.is_enabled()}", flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )

    state = TrainState(patience_left=cfg.early_stop_patience)
    mgr = checkpoint.CheckpointManager(run_name)

    inherited_extra: dict[str, Any] = {}
    if resume_path is not None:
        print(f"[train] resuming from {resume_path}", flush=True)
        ckpt = checkpoint.load_checkpoint(resume_path, map_location=device)
        inherited_extra = dict(ckpt.get("extra") or {})
        if cfg.matchup_adapter_training:
            assert adapter_training_contract is not None
            saved_contract = inherited_extra.get(
                "matchup_adapter_training_contract"
            )
            if saved_contract != adapter_training_contract:
                raise ValueError(
                    "adapter resume corpus/gate/split/implementation contract drift"
                )
            assert adapter_activation is not None
            if (
                str(
                    inherited_extra.get(
                        "matchup_adapter_activation_receipt_digest"
                    )
                    or ""
                )
                != checkpoint.checkpoint_digest(adapter_activation.path)
            ):
                raise ValueError("adapter resume activation receipt identity drift")
        meta = checkpoint.apply_checkpoint(
            ckpt, model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler
        )
        state.step = int(meta["step"])
        state.epoch = int(meta["epoch"])
        if meta.get("best_metric") is not None:
            state.best_metric = float(meta["best_metric"])
        es = meta.get("early_stop_state") or {}
        if "patience_left" in es:
            state.patience_left = int(es["patience_left"])
        state.history = list((meta.get("extra") or {}).get("history") or [])
    elif init_path is not None:
        print(f"[train] initializing weights from {init_path}", flush=True)
        seed_ckpt = checkpoint.load_checkpoint(init_path, map_location="cpu")
        seed_extra = dict(seed_ckpt.get("extra") or {})
        for key in ("pure_rl", "smoke", "model_profile"):
            if key in seed_extra:
                inherited_extra[key] = seed_extra[key]
        inherited_extra["initialized_from"] = str(init_path)
        inherited_extra["initialized_from_digest"] = checkpoint.checkpoint_digest(
            init_path
        )

    resident_corpus: Optional[DeviceResidentBootstrapCorpus] = None
    resident_batch_size = int(cfg.max_decisions_per_batch)
    if device_resident:
        if device.type != "cuda":
            raise ValueError("device-resident bootstrap requires a CUDA device")
        if model.decision_context != "stateless":
            raise ValueError(
                "device-resident bootstrap is valid only for a stateless model"
            )
        weighted_aux = {
            "aux": cfg.aux_loss_weight,
            "opp_hand": cfg.opp_hand_loss_weight,
            "opp_remainder": cfg.opp_remainder_loss_weight,
            "lethal_threat": cfg.lethal_threat_loss_weight,
            "prize_race": cfg.prize_race_loss_weight,
            "alakazam_guide": cfg.alakazam_guide_loss_weight,
        }
        nonzero_aux = {
            name: float(weight)
            for name, weight in weighted_aux.items()
            if float(weight) != 0.0
        }
        if nonzero_aux:
            raise ValueError(
                "device-resident bootstrap requires all auxiliary weights zero: "
                f"{nonzero_aux}"
            )
        if cfg.pure_rl:
            raise ValueError(
                "device-resident bootstrap is supervised hard-target training, "
                "not the AWR pure-RL update path"
            )
        resident_corpus = DeviceResidentBootstrapCorpus.from_splits(
            train_seqs,
            val_seqs,
            device=device,
        )
        # The GPU corpus is now authoritative.  Release every host-side
        # GameSequence reference so the training process cannot accumulate both
        # representations and repeat the earlier host-memory failure.
        dataset.sequences.clear()
        train_seqs.clear()
        val_seqs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "[device-corpus] host GameSequence corpus released; all bootstrap "
            "features/targets remain on Blackwell",
            flush=True,
        )
        resident_batch_size = _fit_device_batch_size(
            model,
            resident_corpus,
            requested=cfg.max_decisions_per_batch,
            cfg=cfg,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

    latest_adapter_validation: dict[str, Any] = (
        copy.deepcopy(
            dict(
                inherited_extra.get("matchup_adapter_per_route_validation") or {}
            )
        )
        if cfg.matchup_adapter_training
        else {}
    )

    def build_ckpt() -> dict[str, Any]:
        if cfg.matchup_adapter_training:
            assert_matchup_adapter_training_contract(
                model,
                optimizer=optimizer,
                base_state=frozen_base_snapshot,
            )
        extra = dict(inherited_extra)
        extra.update(checkpoint_extra or {})
        extra.update(
            {
                "history": state.history,
                "train_cfg": cfg.__dict__,
                "device_resident_bootstrap": bool(resident_corpus is not None),
                "device_resident_batch_size": resident_batch_size,
                "trainable_parameter_prefixes": list(trainable_prefixes),
                "trainable_parameter_count": int(
                    sum(parameter.numel() for parameter in trainable_parameters)
                ),
                "matchup_adapter_training": bool(
                    cfg.matchup_adapter_training
                ),
                "matchup_adapter_routing": (
                    "offline-oracle-package-and-full-deck-audited"
                    if cfg.matchup_adapter_training
                    else None
                ),
                "matchup_adapters_runtime_enabled": bool(
                    model.matchup_adapter_bank.enabled
                ),
            }
        )
        if (
            cfg.current_deck_guide_training_mode
            == GUIDE_TRAINING_MODE_STRATEGIC
        ):
            extra["current_deck_guide_training_contract"] = (
                _strategic_curriculum_contract_record(cfg)
            )
        if cfg.matchup_adapter_training:
            extra["matchup_adapter_config"] = (
                model.matchup_adapter_bank.config_dict()
            )
            assert adapter_activation is not None
            assert adapter_parent_path is not None
            extra["matchup_adapter_activation_receipt"] = str(
                adapter_activation.path
            )
            extra["matchup_adapter_activation_receipt_digest"] = (
                checkpoint.checkpoint_digest(adapter_activation.path)
            )
            extra["matchup_adapter_parent_checkpoint"] = str(
                adapter_parent_path
            )
            extra["matchup_adapter_parent_checkpoint_digest"] = (
                adapter_activation.parent_checkpoint_digest
            )
            assert adapter_training_contract is not None
            extra["matchup_adapter_training_contract"] = copy.deepcopy(
                adapter_training_contract
            )
            extra["matchup_adapter_per_route_validation"] = copy.deepcopy(
                latest_adapter_validation
            )
        return checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            scheduler=scheduler,
            step=state.step,
            epoch=state.epoch,
            best_metric=state.best_metric,
            early_stop_state={
                "patience_left": state.patience_left,
                "best_metric": state.best_metric,
            },
            archetype_id=archetype_id,
            model_id=run_name,
            model_config=model.cfg,
            extra=extra,
        )

    mgr.install_signal_flush(build_ckpt)

    try:
        epoch_bar = tqdm(
            range(state.epoch, cfg.epochs),
            desc="epochs",
            initial=state.epoch,
            total=cfg.epochs,
            unit="ep",
        )
        for epoch in epoch_bar:
            state.epoch = epoch
            if cfg.matchup_adapter_training:
                # The frozen parent must emit the same deterministic states as
                # serving.  Only the adapter bank is conceptually in training
                # mode (it currently has no dropout/batch statistics).
                model.eval()
                model.matchup_adapter_bank.train()
            else:
                model.train()
            if resident_corpus is not None:
                batches = resident_corpus.batches(
                    train=True,
                    batch_size=resident_batch_size,
                    shuffle=True,
                    seed=cfg.seed,
                    epoch=epoch,
                )
            else:
                batches = _iter_game_batches(
                    train_seqs,
                    cfg.games_per_batch,
                    cfg.max_decisions_per_batch,
                    shuffle=True,
                    seed=cfg.seed,
                    epoch=epoch,
                )
            epoch_parts: list[BatchMetrics] = []
            batch_bar = tqdm(batches, desc=f"train ep{epoch}", leave=False, unit="batch")
            for batch in batch_bar:
                optimizer.zero_grad(set_to_none=True)
                adapter_isolation_guard = (
                    prepare_matchup_adapter_isolation_guard(
                        model,
                        optimizer,
                        batch,
                    )
                    if cfg.matchup_adapter_training
                    else None
                )
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    if resident_corpus is not None:
                        total, bm = device_batch_losses(
                            model,
                            resident_corpus,
                            batch,
                            value_weight=cfg.value_loss_weight,
                            current_deck_guide_training_mode=(
                                cfg.current_deck_guide_training_mode
                            ),
                        )
                    else:
                        total, bm = batch_losses(
                            model,
                            batch,
                            value_weight=cfg.value_loss_weight,
                            aux_weight=cfg.aux_loss_weight,
                            opp_hand_weight=cfg.opp_hand_loss_weight,
                            opp_remainder_weight=cfg.opp_remainder_loss_weight,
                            lethal_threat_weight=cfg.lethal_threat_loss_weight,
                            prize_race_weight=cfg.prize_race_loss_weight,
                            alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                            current_deck_guide_training_mode=(
                                cfg.current_deck_guide_training_mode
                            ),
                            setup_board_outcome_loss_weight=(
                                cfg.setup_board_outcome_loss_weight
                            ),
                            combo_state_loss_weight=cfg.combo_state_loss_weight,
                            expanded_head_weights=(
                                cfg.expanded_head_loss_weights
                            ),
                            history_identity_weight=float(
                                cfg.history_identity_loss_weight
                            ),
                            matchup_adapter_training=bool(
                                cfg.matchup_adapter_training
                            ),
                        )
                if bm.n_decisions == 0:
                    continue
                if cfg.matchup_adapter_training and bm.n_matchup_adapter_rows == 0:
                    if total.requires_grad:
                        raise AssertionError(
                            "unsupported matchup rows must not touch adapters"
                        )
                    epoch_parts.append(bm)
                    continue

                scaler.scale(total).backward()
                if cfg.matchup_adapter_training:
                    assert_matchup_adapter_training_contract(
                        model,
                        optimizer=optimizer,
                    )
                    assert adapter_isolation_guard is not None
                    assert_matchup_adapter_isolation_guard(
                        model,
                        optimizer,
                        adapter_isolation_guard,
                        after_step=False,
                    )
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, cfg.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                if cfg.matchup_adapter_training:
                    assert_matchup_adapter_training_contract(
                        model,
                        optimizer=optimizer,
                        base_state=frozen_base_snapshot,
                    )
                    assert adapter_isolation_guard is not None
                    assert_matchup_adapter_isolation_guard(
                        model,
                        optimizer,
                        adapter_isolation_guard,
                        after_step=True,
                    )

                state.step += 1
                epoch_parts.append(bm)
                batch_bar.set_postfix(
                    loss=f"{bm.total_loss:.3f}",
                    p=f"{bm.policy_loss:.3f}",
                    v=f"{bm.value_loss:.3f}",
                    aux=(
                        "off"
                        if cfg.aux_loss_weight == 0
                        else f"{bm.aux_loss:.3f}/{bm.n_archetype_rows}"
                    ),
                    hand=(
                        "off"
                        if cfg.opp_hand_loss_weight == 0
                        else f"{bm.opp_hand_loss:.3f}/{bm.n_opp_hand_rows}"
                    ),
                    rem=(
                        "off"
                        if cfg.opp_remainder_loss_weight == 0
                        else f"{bm.opp_remainder_loss:.3f}/"
                        f"{bm.n_opp_remainder_rows}"
                    ),
                    lethal=(
                        "off"
                        if cfg.lethal_threat_loss_weight == 0
                        else f"{bm.lethal_threat_loss:.3f}/"
                        f"{bm.n_lethal_threat_rows}"
                    ),
                    prize=(
                        "off"
                        if cfg.prize_race_loss_weight == 0
                        else f"{bm.prize_race_loss:.3f}/"
                        f"{bm.n_prize_race_rows}"
                    ),
                    guide=(
                        "off"
                        if cfg.alakazam_guide_loss_weight == 0
                        else f"{bm.alakazam_guide_loss:.3f}/"
                        f"{bm.n_alakazam_guide_rows}"
                    ),
                    acc=f"{bm.policy_acc:.2%}",
                    step=state.step,
                )

                saved = mgr.maybe_save(state.step, build_ckpt)
                if saved:
                    tqdm.write(
                        f"[checkpoint] step={state.step} saved → "
                        + ", ".join(f"{k}={v.name}" for k, v in saved.items())
                    )

            train_m = _merge_metrics(epoch_parts)
            if resident_corpus is not None and resident_corpus.val_samples:
                val_m = evaluate_device_corpus(
                    model,
                    resident_corpus,
                    cfg=cfg,
                    batch_size=resident_batch_size,
                    desc=f"val ep{epoch}",
                )
                metric = val_m.total_loss
            elif val_seqs:
                val_m = evaluate(model, val_seqs, cfg=cfg, desc=f"val ep{epoch}")
                metric = val_m.total_loss
            else:
                val_m = train_m
                metric = train_m.total_loss

            if cfg.matchup_adapter_training:
                latest_adapter_validation.clear()
                for route, archetype_id in enumerate(EXPERT_IDS):
                    route_val = [
                        sequence
                        for sequence in val_seqs
                        if int(adapter_training_ticket(sequence).route) == route
                    ]
                    if not route_val:
                        raise RuntimeError(
                            f"adapter route {archetype_id} lost validation coverage"
                        )
                    route_metrics = evaluate(
                        model,
                        route_val,
                        cfg=cfg,
                        desc=f"val {archetype_id} ep{epoch}",
                    )
                    if route_metrics.n_decisions <= 0:
                        raise RuntimeError(
                            f"adapter route {archetype_id} has no validation decisions"
                        )
                    latest_adapter_validation[archetype_id] = {
                        "route": route,
                        "n_games": int(route_metrics.n_games),
                        "n_decisions": int(route_metrics.n_decisions),
                        "total_loss": float(route_metrics.total_loss),
                        "policy_loss": float(route_metrics.policy_loss),
                        "value_loss": float(route_metrics.value_loss),
                        "policy_acc": float(route_metrics.policy_acc),
                    }

            scheduler.step()
            row = {
                "epoch": epoch,
                "step": state.step,
                "train": train_m.__dict__,
                "val": val_m.__dict__,
                "lr": optimizer.param_groups[0]["lr"],
                "t": time.time(),
            }
            if cfg.matchup_adapter_training:
                row["matchup_adapter_per_route_validation"] = copy.deepcopy(
                    latest_adapter_validation
                )
            state.history.append(row)

            is_best = metric < state.best_metric - 1e-5
            if is_best:
                state.best_metric = metric
                state.patience_left = cfg.early_stop_patience
                mgr.save(build_ckpt(), is_best=True)
                tqdm.write(
                    f"[checkpoint] NEW BEST epoch={epoch} val_loss={metric:.4f} "
                    f"val_acc={val_m.policy_acc:.2%}"
                )
            else:
                state.patience_left -= 1
                mgr.save(build_ckpt(), is_best=False)
                tqdm.write(
                    f"[train] epoch={epoch} train_loss={train_m.total_loss:.4f} "
                    f"val_loss={metric:.4f} val_acc={val_m.policy_acc:.2%} "
                    f"patience={state.patience_left}"
                )

            epoch_bar.set_postfix(
                val_loss=f"{metric:.4f}",
                val_acc=f"{val_m.policy_acc:.2%}",
                best=f"{state.best_metric:.4f}",
                pat=state.patience_left,
            )

            if state.patience_left <= 0:
                tqdm.write(
                    f"[early-stop] patience exhausted at epoch={epoch} "
                    f"best_val_loss={state.best_metric:.4f}"
                )
                break
    finally:
        mgr.uninstall_signal_flush()
        if cfg.matchup_adapter_training:
            assert_matchup_adapter_training_contract(
                model,
                optimizer=optimizer,
                base_state=frozen_base_snapshot,
            )
        # Final flush.
        try:
            mgr.save(build_ckpt(), is_best=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] final save failed: {exc}", flush=True)

    best = checkpoint.best_path(run_name)
    latest = checkpoint.latest_path(run_name)
    return {
        "run_name": run_name,
        "best_metric": state.best_metric,
        "step": state.step,
        "epoch": state.epoch,
        "best_path": str(best) if best.is_file() else None,
        "latest_path": str(latest) if latest.is_file() else None,
        "history": state.history,
    }


def supervised_rehearsal_step(
    corpus: DeviceResidentBootstrapCorpus,
    *,
    base_ckpt: Union[str, Path],
    output_path: Union[str, Path],
    parent_digest: str,
    rehearsal_iteration: int,
    manifest_identity: dict[str, Any],
    epochs: int = 1,
    lr: float = 2e-5,
    requested_batch_size: int = 8192,
    seed: int = 0,
    corpus_split_seed: int = 0,
    device: Optional[torch.device] = None,
    aux_loss_weight: float = 0.0,
    opp_hand_loss_weight: float = 0.0,
    opp_remainder_loss_weight: float = 0.0,
    lethal_threat_loss_weight: float = 0.0,
    prize_race_loss_weight: float = 0.0,
    alakazam_guide_loss_weight: float = 0.0,
    current_deck_guide_training_mode: str = GUIDE_TRAINING_MODE_LEGACY,
    setup_board_outcome_loss_weight: float = (
        SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT
    ),
    combo_state_loss_weight: float = 0.0,
    current_deck_guide_curriculum_spec: str = "",
    current_deck_guide_head_role_map: str = "",
    current_deck_guide_curriculum_validation_receipt: str = "",
    expanded_head_loss_weights: Optional[dict[str, float]] = None,
    expanded_head_schedule: Optional[dict[str, Any]] = None,
    output_archetype_id: Optional[str] = None,
    output_model_id: Optional[str] = None,
    extra_updates: Optional[dict[str, Any]] = None,
    teacher_policy_targets: Optional[torch.Tensor] = None,
    teacher_policy_weight: float = 0.0,
    teacher_policy_target_digest: str = "",
    teacher_policy_checkpoint_digests: Sequence[str] = (),
    training_seat_split_receipt: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run a bounded, resumable expert-policy rehearsal on a resident corpus.

    The returned checkpoint is immutable and keeps the RL iteration number
    unchanged.  It restores the learner's optimizer moments when available,
    applies a deliberately small learning rate for the supervised pass, and
    saves the resulting optimizer state so the next AWR update can continue
    from the exact trained learner rather than restarting AdamW.
    """
    base_path = Path(base_ckpt).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    if epochs <= 0:
        raise ValueError("rehearsal epochs must be positive")
    if lr <= 0.0:
        raise ValueError("rehearsal learning rate must be positive")
    for name, weight in (
        ("aux", aux_loss_weight),
        ("opp_hand", opp_hand_loss_weight),
        ("opp_remainder", opp_remainder_loss_weight),
        ("lethal_threat", lethal_threat_loss_weight),
        ("prize_race", prize_race_loss_weight),
        ("alakazam_guide", alakazam_guide_loss_weight),
        ("combo_state", combo_state_loss_weight),
    ):
        if float(weight) < 0.0:
            raise ValueError(f"rehearsal {name} loss weight cannot be negative")
    if float(teacher_policy_weight) < 0.0:
        raise ValueError("rehearsal teacher policy weight cannot be negative")
    teacher_policy_enabled = float(teacher_policy_weight) > 0.0
    if teacher_policy_enabled:
        if teacher_policy_targets is None:
            raise ValueError(
                "teacher policy distillation requires resident targets"
            )
        if teacher_policy_targets.device != corpus.device:
            raise ValueError(
                "teacher policy distillation targets must be device resident"
            )
        if teacher_policy_targets.ndim != 1 or int(
            teacher_policy_targets.numel()
        ) != int(corpus.total_samples):
            raise ValueError(
                "teacher policy targets do not align to the resident corpus"
            )
        if (
            not str(teacher_policy_target_digest).startswith("sha256:")
            or len(str(teacher_policy_target_digest)) != 71
        ):
            raise ValueError("teacher policy target digest is invalid")
        if not teacher_policy_checkpoint_digests or any(
            not str(value).startswith("sha256:") or len(str(value)) != 71
            for value in teacher_policy_checkpoint_digests
        ):
            raise ValueError("teacher checkpoint digests are invalid")
    canonical_expanded_weights = canonical_expanded_loss_weights(
        expanded_head_loss_weights
    )
    expanded_enabled = any(
        weight > 0.0 for weight in canonical_expanded_weights.values()
    )
    schedule_record = dict(expanded_head_schedule or {})
    if expanded_enabled:
        if schedule_record.get("schema") != "poke_bot.expanded_head_schedule/v1":
            raise ValueError(
                "expanded-head rehearsal requires the canonical schedule record"
            )
        if schedule_record.get("runtime_enabled_heads", []) != []:
            raise ValueError("expanded strategic bootstrap must remain shadow-only")
        if dict(schedule_record.get("loss_weights") or {}) != (
            canonical_expanded_weights
        ):
            raise ValueError(
                "expanded strategic schedule/loss-weight contract mismatch"
            )
        for field_name in (
            "schedule_digest",
            "target_schema",
            "target_schema_digest",
        ):
            value = str(schedule_record.get(field_name) or "")
            if not value:
                raise ValueError(
                    f"expanded strategic schedule lacks {field_name}"
                )
    actual_parent_digest = checkpoint.checkpoint_digest(base_path)
    if str(parent_digest) != actual_parent_digest:
        raise ValueError(
            "parent_digest does not match rehearsal base checkpoint: "
            f"expected={parent_digest!r} actual={actual_parent_digest!r}"
        )

    device = device or corpus.device
    if corpus.device != device:
        raise ValueError(
            f"resident corpus device {corpus.device} != rehearsal device {device}"
        )
    cfg = TrainConfig(
        lr=float(lr),
        epochs=int(epochs),
        max_decisions_per_batch=int(requested_batch_size),
        val_frac=0.10,
        split_by_episode=True,
        early_stop_patience=max(1, int(epochs)),
        aux_loss_weight=float(aux_loss_weight),
        opp_hand_loss_weight=float(opp_hand_loss_weight),
        opp_remainder_loss_weight=float(opp_remainder_loss_weight),
        lethal_threat_loss_weight=float(lethal_threat_loss_weight),
        prize_race_loss_weight=float(prize_race_loss_weight),
        alakazam_guide_loss_weight=float(alakazam_guide_loss_weight),
        current_deck_guide_training_mode=canonical_guide_training_mode(
            current_deck_guide_training_mode
        ),
        setup_board_outcome_loss_weight=float(
            setup_board_outcome_loss_weight
        ),
        combo_state_loss_weight=float(combo_state_loss_weight),
        current_deck_guide_curriculum_spec=str(
            current_deck_guide_curriculum_spec
        ),
        current_deck_guide_head_role_map=str(
            current_deck_guide_head_role_map
        ),
        current_deck_guide_curriculum_validation_receipt=str(
            current_deck_guide_curriculum_validation_receipt
        ),
        expanded_head_loss_weights=canonical_expanded_weights,
        amp=device.type == "cuda",
        seed=int(seed),
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = load_model_from_checkpoint(base_path, device=device)
    if (
        cfg.current_deck_guide_training_mode
        == GUIDE_TRAINING_MODE_STRATEGIC
    ):
        assert_strategic_curriculum_model_contract(
            model,
            setup_board_outcome_loss_weight=(
                cfg.setup_board_outcome_loss_weight
            ),
        )
        assert_strategic_curriculum_receipt_contract(
            specialist_id=(
                str(output_archetype_id)
                if output_archetype_id is not None
                else str(
                    checkpoint.load_checkpoint(
                        base_path, map_location="cpu"
                    ).get("archetype_id")
                    or ""
                )
            ),
            curriculum_spec=cfg.current_deck_guide_curriculum_spec,
            head_role_map=cfg.current_deck_guide_head_role_map,
            validation_receipt=(
                cfg.current_deck_guide_curriculum_validation_receipt
            ),
        )
        if model.decision_context != "history":
            raise ValueError(
                "strategic curriculum rehearsal requires temporal history"
            )
        if float(cfg.alakazam_guide_loss_weight) > 0.0:
            if corpus.guide_confidence is None or not bool(
                corpus.guide_confidence.gt(0.0).any()
            ):
                raise ValueError(
                    "nonzero strategic guide multiplier has no "
                    "confidence-bearing rehearsal rows"
                )
    if expanded_enabled and not bool(
        getattr(model, "expanded_heads_enabled", False)
    ):
        raise ValueError(
            "expanded strategic schedule cannot train a V5 architecture"
        )
    if expanded_enabled and model.decision_context != "history":
        raise ValueError(
            "expanded strategic bootstrap requires the resident temporal layout"
        )
    aux_head_expanded = expand_aux_head_to_current_registry(model)
    warm_started_heads_before = tuple(
        getattr(model, "warm_started_belief_heads", ()) or ()
    )
    warm_started_expanded_before = tuple(
        getattr(model, "warm_started_expanded_heads", ()) or ()
    )
    warm_started_fusion_before = bool(
        getattr(model, "warm_started_decision_fusion", False)
    )
    # Dormant matchup adapters are architecture-present but must stay outside
    # the ordinary learner optimizer.  This also keeps legacy/base optimizer
    # param-group cardinality stable when a dormant bank is staged.
    ordinary_trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        ordinary_trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    base_payload = checkpoint.load_checkpoint(base_path, map_location="cpu")
    if (
        warm_started_heads_before
        or warm_started_expanded_before
        or warm_started_fusion_before
        or aux_head_expanded
    ):
        # ``load_model_from_checkpoint`` already restored every pre-existing
        # tensor and deterministically initialized only allowed new heads.  An
        # optimizer snapshot from the legacy architecture cannot contain those
        # parameters, so restoring its param groups would either fail or attach
        # moments to the wrong layout.  Preserve counters/RNG while starting a
        # fresh AdamW state for this one warm-start rehearsal.
        meta = checkpoint.apply_checkpoint(
            base_payload,
            restore_rng=True,
        )
        optimizer_state_restored = False
    else:
        meta = checkpoint.apply_checkpoint(
            base_payload,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            restore_rng=True,
        )
        optimizer_state_restored = "optimizer_state_dict" in base_payload
    # Optimizer restore also restores its old param-group LR.  Rehearsal uses
    # the explicitly recorded conservative LR while preserving the moments.
    for group in optimizer.param_groups:
        group["lr"] = cfg.lr

    batch_size = _fit_device_batch_size(
        model,
        corpus,
        requested=cfg.max_decisions_per_batch,
        cfg=cfg,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    step = int(meta.get("step", 0))
    epoch0 = int(meta.get("epoch", 0))
    rl_iteration = int(meta.get("rl_iteration", 0))
    train_metrics = BatchMetrics()
    started = time.time()
    temporal_rehearsal = model.decision_context == "history"
    if temporal_rehearsal and not corpus.has_temporal_layout:
        raise ValueError(
            "history checkpoint requires a resident full-game rehearsal layout"
        )
    for epoch_offset in range(int(epochs)):
        model.train()
        parts: list[BatchMetrics] = []
        batches = (
            corpus.temporal_batches(
                train=True,
                batch_size=batch_size,
                shuffle=True,
                seed=cfg.seed,
                epoch=epoch_offset,
            )
            if temporal_rehearsal
            else corpus.batches(
                train=True,
                batch_size=batch_size,
                shuffle=True,
                seed=cfg.seed,
                epoch=epoch_offset,
            )
        )
        bar = tqdm(
            batches,
            desc=(
                f"expert rehearsal before iter{int(rehearsal_iteration)} "
                f"ep{epoch_offset + 1}/{int(epochs)}"
            ),
            unit="batch",
        )
        for batch_ids in bar:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                if temporal_rehearsal:
                    total, metrics = device_temporal_batch_losses(
                        model,
                        corpus,
                        batch_ids,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        current_deck_guide_training_mode=(
                            cfg.current_deck_guide_training_mode
                        ),
                        setup_board_outcome_loss_weight=(
                            cfg.setup_board_outcome_loss_weight
                        ),
                        combo_state_loss_weight=cfg.combo_state_loss_weight,
                        expanded_head_weights=cfg.expanded_head_loss_weights,
                        teacher_policy_targets=teacher_policy_targets,
                        teacher_policy_weight=float(teacher_policy_weight),
                    )
                else:
                    total, metrics = device_batch_losses(
                        model,
                        corpus,
                        batch_ids,
                        value_weight=cfg.value_loss_weight,
                        current_deck_guide_training_mode=(
                            cfg.current_deck_guide_training_mode
                        ),
                    )
            scaler.scale(total).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            parts.append(metrics)
            bar.set_postfix(
                loss=f"{metrics.total_loss:.3f}",
                policy=f"{metrics.policy_loss:.3f}",
                teacher=(
                    "off"
                    if not teacher_policy_enabled
                    else (
                        f"{metrics.teacher_policy_loss:.3f}/"
                        f"{metrics.n_teacher_policy_rows}"
                    )
                ),
                value=f"{metrics.value_loss:.3f}",
                aux=(
                    "off"
                    if cfg.aux_loss_weight == 0
                    else f"{metrics.aux_loss:.3f}/{metrics.n_archetype_rows}"
                ),
                hand=(
                    "off"
                    if cfg.opp_hand_loss_weight == 0
                    else f"{metrics.opp_hand_loss:.3f}/{metrics.n_opp_hand_rows}"
                ),
                rem=(
                    "off"
                    if cfg.opp_remainder_loss_weight == 0
                    else f"{metrics.opp_remainder_loss:.3f}/"
                    f"{metrics.n_opp_remainder_rows}"
                ),
                lethal=(
                    "off"
                    if cfg.lethal_threat_loss_weight == 0
                    else f"{metrics.lethal_threat_loss:.3f}/"
                    f"{metrics.n_lethal_threat_rows}"
                ),
                prize=(
                    "off"
                    if cfg.prize_race_loss_weight == 0
                    else f"{metrics.prize_race_loss:.3f}/"
                    f"{metrics.n_prize_race_rows}"
                ),
                guide=(
                    "off"
                    if cfg.alakazam_guide_loss_weight == 0
                    else f"{metrics.alakazam_guide_loss:.3f}/"
                    f"{metrics.n_alakazam_guide_rows}"
                ),
                strategic=(
                    "off"
                    if not expanded_enabled
                    else (
                        f"{sum(int(value) for value in dict((metrics.expanded_head_metrics or {}).get('labeled') or {}).values())}"
                        f"/{len([value for value in canonical_expanded_weights.values() if value > 0.0])}h"
                    )
                ),
                acc=f"{metrics.policy_acc:.2%}",
                step=step,
            )
        train_metrics = _merge_metrics(parts)

    val_metrics = (
        evaluate_device_corpus(
            model,
            corpus,
            cfg=cfg,
            batch_size=batch_size,
            desc=f"expert validation before iter{int(rehearsal_iteration)}",
            teacher_policy_targets=teacher_policy_targets,
            teacher_policy_weight=float(teacher_policy_weight),
        )
        if corpus.val_samples
        else train_metrics
    )
    warm_head_loss_weights = {
        "opp_hand_head": float(cfg.opp_hand_loss_weight),
        "opp_remainder_head": float(cfg.opp_remainder_loss_weight),
        "lethal_threat_head": float(cfg.lethal_threat_loss_weight),
        "prize_race_head": float(cfg.prize_race_loss_weight),
    }
    warm_started_heads_remaining = tuple(
        name
        for name in warm_started_heads_before
        if warm_head_loss_weights.get(name, 0.0) <= 0.0
    )
    # Fully covered positive-weight targets have now trained these heads, so
    # serving may consume them instead of retaining the uniform warm fallback.
    model.warm_started_belief_heads = warm_started_heads_remaining
    inherited_extra = dict(base_payload.get("extra") or {})
    prior_expanded_contract = dict(
        inherited_extra.get("expanded_head_training") or {}
    )
    prior_trained_expanded = {
        str(name) for name in prior_expanded_contract.get("trained_heads") or ()
    }
    expanded_training_contract: dict[str, Any] = {}
    warm_started_expanded_remaining = warm_started_expanded_before
    if expanded_enabled:
        train_expanded = dict(train_metrics.expanded_head_metrics or {})
        validation_expanded = dict(val_metrics.expanded_head_metrics or {})
        train_labeled = dict(train_expanded.get("labeled") or {})
        validation_labeled = dict(validation_expanded.get("labeled") or {})
        gradient_enabled = [
            name
            for name in EXPANDED_HEAD_IDS
            if float(canonical_expanded_weights[name]) > 0.0
        ]
        trained_this_epoch = [
            name
            for name in gradient_enabled
            if int(train_labeled.get(name, 0)) > 0
        ]
        trained_expanded = [
            name
            for name in EXPANDED_HEAD_IDS
            if name in prior_trained_expanded or name in trained_this_epoch
        ]
        warm_started_expanded_remaining = tuple(
            module
            for module in warm_started_expanded_before
            if module.removesuffix("_head") not in trained_expanded
        )
        model.warm_started_expanded_heads = warm_started_expanded_remaining

        train_losses = dict(train_expanded.get("losses") or {})
        validation_losses = dict(validation_expanded.get("losses") or {})
        train_masked = dict(train_expanded.get("masked") or {})
        validation_masked = dict(validation_expanded.get("masked") or {})
        train_total = dict(train_expanded.get("total") or {})
        validation_total = dict(validation_expanded.get("total") or {})
        head_rows: dict[str, dict[str, Any]] = {}
        coverage_rows: dict[str, dict[str, Any]] = {}
        for name in EXPANDED_HEAD_IDS:
            labeled_rows = int(train_labeled.get(name, 0)) + int(
                validation_labeled.get(name, 0)
            )
            masked_rows = int(train_masked.get(name, 0)) + int(
                validation_masked.get(name, 0)
            )
            total_rows = int(train_total.get(name, 0)) + int(
                validation_total.get(name, 0)
            )
            coverage = (
                float(labeled_rows) / total_rows if total_rows > 0 else 0.0
            )
            coverage_rows[name] = {
                "labeled_rows": labeled_rows,
                "masked_rows": masked_rows,
                "total_rows": total_rows,
                "coverage": coverage,
            }
            head_rows[name] = {
                "present": True,
                "trained": name in trained_expanded,
                "trained_this_epoch": name in trained_this_epoch,
                "gradient_enabled": name in gradient_enabled,
                "runtime_enabled": False,
                "loss_weight": float(canonical_expanded_weights[name]),
                "train_loss": (
                    float(train_losses[name]) if name in train_losses else None
                ),
                "validation_loss": (
                    float(validation_losses[name])
                    if name in validation_losses
                    else None
                ),
                "train_labeled_rows": int(train_labeled.get(name, 0)),
                "validation_labeled_rows": int(
                    validation_labeled.get(name, 0)
                ),
                **coverage_rows[name],
            }
        fused_action_path = bool(
            getattr(model, "decision_fusion_enabled", False)
            and getattr(model, "decision_fusion_runtime_enabled", False)
        )
        expanded_training_contract = {
            "schema": "poke_bot.expanded_head_training/v1",
            "target_schema_version": str(schedule_record["target_schema"]),
            "target_schema_digest": str(
                schedule_record["target_schema_digest"]
            ),
            "schedule_version": str(schedule_record["schema"]),
            "schedule_digest": str(schedule_record["schedule_digest"]),
            "stage": int(schedule_record.get("stage_index", 0)),
            "epoch": int(schedule_record.get("epoch", rehearsal_iteration)),
            "epochs_total": 25,
            "architecture_present_heads": list(EXPANDED_HEAD_IDS),
            "trained_heads": trained_expanded,
            "trained_this_epoch": trained_this_epoch,
            "gradient_enabled_heads": gradient_enabled,
            "runtime_enabled_heads": [],
            "loss_weights": dict(canonical_expanded_weights),
            "train_metrics": {
                name: {"loss": train_losses.get(name)}
                for name in EXPANDED_HEAD_IDS
            },
            "validation_metrics": {
                name: {"loss": validation_losses.get(name)}
                for name in EXPANDED_HEAD_IDS
            },
            "coverage": coverage_rows,
            "heads": head_rows,
            "calibration": {
                "train": dict(train_expanded.get("calibration") or {}),
                "validation": dict(
                    validation_expanded.get("calibration") or {}
                ),
            },
            "warm_started_heads_before": list(
                warm_started_expanded_before
            ),
            "warm_started_heads_remaining": list(
                warm_started_expanded_remaining
            ),
            "shadow_only": not fused_action_path,
            "flat_policy_authoritative": not fused_action_path,
            "authoritative_action_path": (
                "fused_policy" if fused_action_path else "flat_policy"
            ),
        }
    strategic_training_record = (
        _strategic_curriculum_training_record(
            cfg=cfg,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
        )
        if cfg.current_deck_guide_training_mode
        == GUIDE_TRAINING_MODE_STRATEGIC
        else {}
    )
    rehearsal_record = {
        "schema": 1,
        "before_iteration": int(rehearsal_iteration),
        "parent_digest": actual_parent_digest,
        "manifest": dict(manifest_identity),
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "batch_size": int(batch_size),
        "requested_batch_size": int(requested_batch_size),
        "train_metrics": train_metrics.__dict__,
        "validation_metrics": val_metrics.__dict__,
        "elapsed_sec": float(time.time() - started),
        "optimizer_state_restored": bool(optimizer_state_restored),
        "aux_head_expanded_to_current_registry": bool(aux_head_expanded),
        "warm_started_belief_heads_before": list(warm_started_heads_before),
        "warm_started_belief_heads_remaining": list(
            warm_started_heads_remaining
        ),
        "warm_started_expanded_heads_before": list(
            warm_started_expanded_before
        ),
        "warm_started_expanded_heads_remaining": list(
            warm_started_expanded_remaining
        ),
        "decision_context": str(model.decision_context),
        "resident_temporal_layout": bool(corpus.has_temporal_layout),
        "corpus_split_seed": int(corpus_split_seed),
        **(
            {"training_seat_split_receipt": dict(training_seat_split_receipt)}
            if training_seat_split_receipt
            else {}
        ),
        "loss_weights": {
            "value": float(cfg.value_loss_weight),
            "archetype": float(cfg.aux_loss_weight),
            "opponent_hand": float(cfg.opp_hand_loss_weight),
            "opponent_hidden_remainder": float(cfg.opp_remainder_loss_weight),
            "lethal_threat": float(cfg.lethal_threat_loss_weight),
            "prize_race": float(cfg.prize_race_loss_weight),
            "alakazam_guide": float(cfg.alakazam_guide_loss_weight),
            "combo_state": float(cfg.combo_state_loss_weight),
            "expanded_strategic": dict(canonical_expanded_weights),
        },
        **(
            {"current_deck_guide_training": strategic_training_record}
            if strategic_training_record
            else {}
        ),
        "teacher_behavior_distillation": {
            "enabled": teacher_policy_enabled,
            "target_digest": (
                str(teacher_policy_target_digest)
                if teacher_policy_enabled
                else None
            ),
            "teacher_checkpoint_digests": (
                [str(value) for value in teacher_policy_checkpoint_digests]
                if teacher_policy_enabled
                else []
            ),
            "loss_weight": float(teacher_policy_weight),
            "train_rows": int(train_metrics.n_teacher_policy_rows),
            "validation_rows": int(val_metrics.n_teacher_policy_rows),
            "train_loss": float(train_metrics.teacher_policy_loss),
            "validation_loss": float(val_metrics.teacher_policy_loss),
            "causal_inputs_only": True,
        },
        **(
            {"expanded_head_training": expanded_training_contract}
            if expanded_training_contract
            else {}
        ),
    }
    inherited_extra.update(
        {
            "pure_rl": True,
            "expert_rehearsal": rehearsal_record,
            "parent_digest": actual_parent_digest,
        }
    )
    if extra_updates:
        inherited_extra.update(dict(extra_updates))
    if expanded_training_contract:
        # The exact checkpoint-producing step owns this record. Callers may
        # add bootstrap provenance but cannot override tensor-bound telemetry.
        inherited_extra["expanded_head_training"] = expanded_training_contract
    payload = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler if use_amp else None,
        step=step,
        epoch=epoch0 + int(epochs),
        rl_iteration=rl_iteration,
        best_metric=float(val_metrics.total_loss),
        early_stop_state={
            "patience_left": max(1, int(epochs)),
            "best_metric": float(val_metrics.total_loss),
        },
        archetype_id=(
            str(output_archetype_id)
            if output_archetype_id is not None
            else str(base_payload.get("archetype_id") or "core")
        ),
        model_id=(
            str(output_model_id)
            if output_model_id is not None
            else str(base_payload.get("model_id") or "pure_rl") + ".expert"
        ),
        model_config=model.cfg,
        extra=inherited_extra,
    )
    saved = checkpoint.immutable_torch_save(payload, out_path)
    digest = checkpoint.checkpoint_digest(saved)
    return {
        "candidate_path": str(saved),
        "candidate_digest": digest,
        "parent_digest": actual_parent_digest,
        "step": step,
        "rl_iteration": rl_iteration,
        "optimizer_state_restored": bool(optimizer_state_restored),
        "batch_size": int(batch_size),
        "train_metrics": train_metrics.__dict__,
        "validation_metrics": val_metrics.__dict__,
        "expanded_head_training": expanded_training_contract,
        "rehearsal": rehearsal_record,
        "output_archetype_id": str(payload.get("archetype_id") or ""),
        "output_model_id": str(payload.get("model_id") or ""),
    }


def process_with_oom_splitting(
    items: Sequence[Any],
    process: Callable[[list[Any]], Any],
    *,
    is_oom: Callable[[BaseException], bool] = config.is_cuda_oom,
    on_split: Optional[Callable[[], None]] = None,
) -> list[Any]:
    """Process every item, recursively splitting an OOMing batch.

    Both halves are queued in original order. A single-item OOM is re-raised,
    because silently dropping it would corrupt the effective training set.
    """
    pending: list[list[Any]] = [list(items)]
    completed: list[Any] = []
    while pending:
        work = pending.pop(0)
        if not work:
            continue
        try:
            completed.append(process(work))
        except BaseException as exc:  # noqa: BLE001 - preserve CUDA exception type
            if not is_oom(exc) or len(work) <= 1:
                raise
            if on_split is not None:
                on_split()
            mid = len(work) // 2
            pending[0:0] = [work[:mid], work[mid:]]
    return completed


@dataclass(frozen=True)
class _AwrValueGamePlan:
    """CPU-prepared row identity for one value-only temporal sequence."""

    sequence: GameSequence
    row_keys_by_decision: tuple[tuple[tuple[int, int, int], ...], ...]


def _plan_awr_value_batch(
    sequences: Sequence[GameSequence],
) -> list[_AwrValueGamePlan]:
    """Validate pure-RL stages and prepare exact baseline cache keys on CPU."""

    plans: list[_AwrValueGamePlan] = []
    for game in sequences:
        if not game.decisions:
            continue
        policy_targets = game.policy_targets
        factorized_targets = game.factorized_policy_targets
        rows_by_decision: list[tuple[tuple[int, int, int], ...]] = []
        for decision_index, decision in enumerate(game.decisions):
            stages = decision.policy_stages or [
                PolicyStage(
                    options=decision.options,
                    action_combos=decision.action_combos,
                    target_index=decision.action_combo_index,
                )
            ]
            stage_targets = (
                factorized_targets[decision_index]
                if factorized_targets is not None
                and decision_index < len(factorized_targets)
                and factorized_targets[decision_index] is not None
                else None
            )
            keys: list[tuple[int, int, int]] = []
            for stage_index, stage in enumerate(stages):
                option_count = stage.options.num_words
                if option_count <= 0:
                    continue
                soft_target = False
                if stage_targets is not None and stage_index < len(stage_targets):
                    target = dict(stage_targets[stage_index] or {})
                    recorded_combos = [
                        list(combo) for combo in (target.get("action_combos") or [])
                    ]
                    if recorded_combos and recorded_combos != stage.action_combos:
                        raise ValueError(
                            "factorized target/action candidate ordering mismatch"
                        )
                    policy = list(target.get("policy") or [])
                    if len(policy) != option_count or sum(policy) <= 0:
                        raise ValueError("invalid factorized soft policy target")
                    selected_index = int(
                        target.get("selected_index", stage.target_index)
                    )
                    soft_target = True
                elif (
                    not decision.policy_stages
                    and policy_targets is not None
                    and decision_index < len(policy_targets)
                    and policy_targets[decision_index] is not None
                ):
                    policy = list(policy_targets[decision_index][:option_count])
                    if len(policy) != option_count or sum(policy) <= 0:
                        continue
                    selected_index = int(
                        max(range(option_count), key=lambda index: policy[index])
                    )
                    soft_target = True
                else:
                    selected_index = int(stage.target_index)
                if selected_index < 0 or selected_index >= option_count:
                    continue
                if soft_target:
                    raise ValueError(
                        "PURE_RL=1 forbids soft factorized_policy_targets as CE/"
                        "behavior-clone training targets; store selected_index only"
                    )
                keys.append((id(game), decision_index, stage_index))
            rows_by_decision.append(tuple(keys))
        plans.append(
            _AwrValueGamePlan(
                sequence=game,
                row_keys_by_decision=tuple(rows_by_decision),
            )
        )
    return plans


def _length_bucket(length: int) -> int:
    """Return a power-of-two temporal bucket without changing sequence order."""

    return 1 << max(0, int(length - 1).bit_length())


@torch.no_grad()
def _capture_value_only_awr_plans(
    model: TemporalCabtTransformer,
    plans: Sequence[_AwrValueGamePlan],
    *,
    pack_temporal_games: bool,
) -> dict[tuple[int, int, int], float]:
    """Evaluate only V(s) for length-bucketed, padded temporal game batches."""

    device = next(model.parameters()).device
    all_boards = [
        decision.board
        for plan in plans
        for decision in plan.sequence.decisions
    ]
    spatial_all = model.encode_board(all_boards)
    all_lengths = [len(plan.sequence.decisions) for plan in plans]
    spatial_by_plan = {
        id(plan): spatial
        for plan, spatial in zip(plans, spatial_all.split(all_lengths))
    }
    buckets: dict[int, list[_AwrValueGamePlan]] = {}
    if pack_temporal_games:
        for plan in plans:
            buckets.setdefault(
                _length_bucket(len(plan.sequence.decisions)), []
            ).append(plan)
    else:
        # One stable bucket per game preserves the reference temporal kernel
        # and therefore the exact frozen values while still bypassing every
        # policy/guide/auxiliary computation.
        buckets = {index: [plan] for index, plan in enumerate(plans)}

    states_by_plan: dict[int, torch.Tensor] = {}
    for bucket_plans in buckets.values():
        games = [plan.sequence for plan in bucket_plans]
        lengths = [len(game.decisions) for game in games]
        bucket_spatial = [spatial_by_plan[id(plan)] for plan in bucket_plans]
        spatial_all = torch.cat(bucket_spatial, dim=0)

        if model.decision_context == "history":
            previous_actions = [
                action
                for game in games
                for action in (
                    [None]
                    + [
                        decision.action_token
                        for decision in game.decisions[:-1]
                    ]
                )
            ]
            cls_all = model.history_tokens(spatial_all, previous_actions)
            cls_by_game = list(cls_all.split(lengths))
            padded_cls = pad_sequence(cls_by_game, batch_first=True)
            length_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
            padding_mask = (
                torch.arange(padded_cls.size(1), device=device).unsqueeze(0)
                >= length_tensor.unsqueeze(1)
            )
            packed_states, _ = model.temporal_encode(
                padded_cls,
                append=False,
                return_all=True,
                key_padding_mask=padding_mask,
            )
            for row, (plan, length) in enumerate(zip(bucket_plans, lengths)):
                states_by_plan[id(plan)] = packed_states[row, :length]
        else:
            cls = model.pool_cls(spatial_all).unsqueeze(1)
            states, _ = model.temporal_encode(
                cls, append=False, return_all=True
            )
            for plan, game_states in zip(
                bucket_plans, states.squeeze(1).split(lengths)
            ):
                states_by_plan[id(plan)] = game_states

    ordered_keys: list[tuple[int, int, int]] = []
    ordered_states: list[torch.Tensor] = []
    for plan in plans:
        game_states = states_by_plan[id(plan)]
        if game_states.size(0) != len(plan.row_keys_by_decision):
            raise AssertionError("value-only temporal row count drift")
        for state, decision_keys in zip(game_states, plan.row_keys_by_decision):
            for key in decision_keys:
                ordered_keys.append(key)
                ordered_states.append(state)
    if not ordered_states:
        return {}
    state_all = torch.stack(ordered_states, dim=0)
    value_pred = (
        torch.tanh(model.value_head(state_all))
        .squeeze(-1)
        .detach()
        .float()
        .cpu()
        .tolist()
    )
    return {
        key: float(value) for key, value in zip(ordered_keys, value_pred)
    }


@torch.no_grad()
def _precompute_awr_baseline_cache_value_only(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    desc: Optional[str] = None,
) -> dict[tuple[int, int, int], float]:
    """Snapshot frozen V(s) with packed temporal inference and bounded prefetch."""

    cache: dict[tuple[int, int, int], float] = {}
    if not sequences:
        return cache
    if int(cfg.awr_baseline_prefetch_batches) not in {0, 1}:
        raise ValueError("AWR baseline prefetch must be bounded to zero or one batch")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )

    def _evaluate(work: list[_AwrValueGamePlan]) -> dict[tuple[int, int, int], float]:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            return _capture_value_only_awr_plans(
                model,
                work,
                pack_temporal_games=bool(cfg.awr_pack_temporal_baseline),
            )

    try:
        visible = tqdm(total=len(batches), desc=desc, leave=False, unit="batch") if desc else None
        if int(cfg.awr_baseline_prefetch_batches) == 1 and batches:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="awr-prefetch") as pool:
                future: Future[list[_AwrValueGamePlan]] = pool.submit(
                    _plan_awr_value_batch, batches[0]
                )
                for index in range(len(batches)):
                    plans = future.result()
                    if index + 1 < len(batches):
                        future = pool.submit(_plan_awr_value_batch, batches[index + 1])
                    parts = process_with_oom_splitting(
                        plans,
                        _evaluate,
                        on_split=(
                            torch.cuda.empty_cache if device.type == "cuda" else None
                        ),
                    )
                    for part in parts:
                        cache.update(part)
                    if visible is not None:
                        visible.update(1)
        else:
            for batch in batches:
                plans = _plan_awr_value_batch(batch)
                parts = process_with_oom_splitting(
                    plans,
                    _evaluate,
                    on_split=(
                        torch.cuda.empty_cache if device.type == "cuda" else None
                    ),
                )
                for part in parts:
                    cache.update(part)
                if visible is not None:
                    visible.update(1)
        if visible is not None:
            visible.close()
    finally:
        model.train(was_training)
    return cache


@torch.no_grad()
def _precompute_awr_baseline_cache_reference(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    desc: Optional[str] = None,
) -> dict[tuple[int, int, int], float]:
    """Snapshot V(s) for every selected decision-stage before an RL update.

    Keys use the in-memory ``GameSequence`` identity plus decision/stage index;
    split/shuffled batches retain those objects, so the cache remains stable for
    the lifetime of one :func:`rl_train_step` without mutating replay records.
    """
    cache: dict[tuple[int, int, int], float] = {}
    if not sequences:
        return cache

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    try:
        visible_batches = (
            tqdm(batches, desc=desc, leave=False, unit="batch")
            if desc
            else batches
        )
        for batch in visible_batches:
            def _capture(work: list[GameSequence]) -> BatchMetrics:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    _, metrics = batch_losses(
                        model,
                        work,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        current_deck_guide_training_mode=(
                            cfg.current_deck_guide_training_mode
                        ),
                        setup_board_outcome_loss_weight=(
                            cfg.setup_board_outcome_loss_weight
                        ),
                        combo_state_loss_weight=cfg.combo_state_loss_weight,
                        expanded_head_weights=cfg.expanded_head_loss_weights,
                        pure_rl=True,
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_capture_baseline=cache,
                    )
                return metrics

            process_with_oom_splitting(
                batch,
                _capture,
                on_split=(
                    torch.cuda.empty_cache if device.type == "cuda" else None
                ),
            )
    finally:
        model.train(was_training)
    return cache


def _precompute_awr_baseline_cache(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    desc: Optional[str] = None,
) -> dict[tuple[int, int, int], float]:
    """Dispatch to the optimized value-only cache or the parity reference."""

    if bool(cfg.awr_value_only_baseline):
        optimized_started = time.perf_counter()
        optimized = _precompute_awr_baseline_cache_value_only(
            model, sequences, cfg=cfg, desc=desc
        )
        optimized_seconds = time.perf_counter() - optimized_started
        if bool(cfg.awr_baseline_exact_parity_check):
            reference_started = time.perf_counter()
            reference = _precompute_awr_baseline_cache_reference(
                model, sequences, cfg=cfg, desc=f"{desc or 'rl-prep baseline'} parity"
            )
            reference_seconds = time.perf_counter() - reference_started
            if optimized.keys() != reference.keys():
                raise RuntimeError(
                    "value-only AWR baseline parity key set differs from reference"
                )
            mismatches = [
                key for key in reference if optimized[key] != reference[key]
            ]
            if mismatches:
                max_abs = max(
                    abs(optimized[key] - reference[key]) for key in mismatches
                )
                raise RuntimeError(
                    "value-only AWR baseline exact parity failed: "
                    f"mismatches={len(mismatches)} max_abs={max_abs:.9g}"
                )
            print(
                "[rl-prep] value-only exact parity passed "
                f"rows={len(optimized)} optimized_s={optimized_seconds:.3f} "
                f"reference_s={reference_seconds:.3f} "
                f"speedup={reference_seconds / max(optimized_seconds, 1e-9):.3f}x",
                flush=True,
            )
        return optimized
    return _precompute_awr_baseline_cache_reference(
        model, sequences, cfg=cfg, desc=desc
    )


@torch.no_grad()
def _policy_argmax_predictions(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None,
    desc: Optional[str] = None,
) -> list[int]:
    """Return deterministic policy argmaxes in stable decision-stage order."""
    if not sequences:
        return []

    predictions: list[int] = []
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    batches = _iter_game_batches(
        list(sequences),
        cfg.games_per_batch,
        cfg.max_decisions_per_batch,
        shuffle=False,
        seed=cfg.seed,
        epoch=0,
    )
    try:
        visible_batches = (
            tqdm(batches, desc=desc, leave=False, unit="batch")
            if desc
            else batches
        )
        for batch in visible_batches:
            def _predict(work: list[GameSequence]) -> BatchMetrics:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    _, metrics = batch_losses(
                        model,
                        work,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        current_deck_guide_training_mode=(
                            cfg.current_deck_guide_training_mode
                        ),
                        setup_board_outcome_loss_weight=(
                            cfg.setup_board_outcome_loss_weight
                        ),
                        combo_state_loss_weight=cfg.combo_state_loss_weight,
                        expanded_head_weights=cfg.expanded_head_loss_weights,
                        pure_rl=bool(cfg.pure_rl),
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_baseline_cache=awr_baseline_cache,
                        prediction_sink=predictions,
                    )
                return metrics

            process_with_oom_splitting(
                batch,
                _predict,
                on_split=(
                    torch.cuda.empty_cache if device.type == "cuda" else None
                ),
            )
    finally:
        model.train(was_training)
    return predictions


def _train_dormant_matchup_adapter_phase(
    model: TemporalCabtTransformer,
    sequences: Sequence[GameSequence],
    *,
    cfg: TrainConfig,
    base_rl_iteration: int,
    target_rl_iteration: Optional[int] = None,
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]],
    seed: int,
    prior_optimizer_state: Optional[dict[str, Any]] = None,
    prior_fit: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit ticketed matchup residuals without changing ordinary RL behavior.

    An immutable committed-boundary receipt authorizes the append-only feature.
    The current learner must match that authorization. Runtime remains off
    during fitting, and a per-batch guard proves absent experts receive neither
    gradients nor Adam state/weight-decay updates.
    """

    epochs = int(cfg.dormant_matchup_adapter_epochs)
    if epochs <= 0:
        return {}, {}
    if not cfg.pure_rl:
        raise ValueError("dormant matchup adapter phase requires pure RL")
    receipt_path = Path(
        str(cfg.dormant_matchup_adapter_activation_receipt)
    ).expanduser().resolve()
    if not str(cfg.dormant_matchup_adapter_activation_receipt).strip():
        raise ValueError("dormant matchup adapter phase requires an activation receipt")
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_parent = Path(
        str(receipt_payload.get("parent_checkpoint") or "")
    ).expanduser().resolve()
    activation = validate_adapter_training_authorization(
        receipt_path,
        parent_checkpoint=receipt_parent,
        permit_post_boundary_use=True,
    )
    effective_iteration = (
        int(target_rl_iteration)
        if target_rl_iteration is not None
        else int(base_rl_iteration) + 1
    )
    if effective_iteration < int(activation.first_eligible_iteration):
        raise ValueError(
            "dormant matchup adapter phase precedes its authorized boundary"
        )

    from .matchup_adapter_routes import resolve_matchup_adapter_route_contract

    route_contract = resolve_matchup_adapter_route_contract(
        model.matchup_adapter_bank.config_dict()
    )
    route_ids = tuple(route_contract.target_ids)
    route_sequences = {expert_id: 0 for expert_id in route_ids}
    route_decisions = {expert_id: 0 for expert_id in route_ids}
    routed: list[GameSequence] = []
    for sequence in sequences:
        if not sequence.decisions or not sequence.matchup_adapter_training_ticket:
            continue
        ticket = adapter_training_ticket(sequence)
        # Validate every decision now, before optimizer construction, so a
        # malformed row cannot partially update an otherwise valid route.
        for route in training_routes_for_sequence(sequence):
            if route != int(ticket.route):
                raise RuntimeError("adapter ticket route changed within one sequence")
        routed.append(sequence)
        route_sequences[ticket.archetype_id] += 1
        route_decisions[ticket.archetype_id] += len(sequence.decisions)
    if not routed:
        raise ValueError("dormant matchup adapter phase has no oracle-ticketed sequences")

    base_state = matchup_adapter_base_state(model)
    original_requires_grad = {
        name: bool(parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    adapter_optimizer = build_matchup_adapter_optimizer(
        model,
        lr=float(cfg.dormant_matchup_adapter_lr),
        weight_decay=float(cfg.weight_decay),
        activation_receipt=activation,
    )
    optimizer_restored = False
    if prior_optimizer_state:
        try:
            load_append_only_matchup_adapter_optimizer_state(
                adapter_optimizer, copy.deepcopy(prior_optimizer_state)
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "dormant matchup adapter optimizer state is incompatible"
            ) from exc
        for group in adapter_optimizer.param_groups:
            group["lr"] = float(cfg.dormant_matchup_adapter_lr)
            group["weight_decay"] = float(cfg.weight_decay)
        optimizer_restored = True

    step_count = 0
    row_count = 0
    last_metrics = BatchMetrics()
    try:
        model.train()
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.synchronize(next(model.parameters()).device)
            reserved_before = torch.cuda.memory_reserved(
                next(model.parameters()).device
            )
            gc.collect()
            torch.cuda.empty_cache()
            reserved_after = torch.cuda.memory_reserved(
                next(model.parameters()).device
            )
            print(
                "[rl-adapters] released CUDA cache before isolated fit "
                f"reserved_before={reserved_before} "
                f"reserved_after={reserved_after} "
                "decisions_cap="
                f"{int(cfg.dormant_matchup_adapter_max_decisions_per_batch)}",
                flush=True,
            )
        for epoch in range(epochs):
            batches = _iter_game_batches(
                routed,
                max(1, int(cfg.games_per_batch)),
                min(
                    max(1, int(cfg.max_decisions_per_batch)),
                    max(
                        1,
                        int(
                            cfg.dormant_matchup_adapter_max_decisions_per_batch
                        ),
                    ),
                ),
                shuffle=True,
                seed=int(seed) + 104729,
                epoch=epoch,
            )
            bar = tqdm(
                batches,
                desc=f"rl-adapters ep{epoch}",
                leave=False,
                unit="batch",
            )
            for batch in bar:
                def _fit_chunk(work: list[GameSequence]) -> BatchMetrics:
                    adapter_optimizer.zero_grad(set_to_none=True)
                    guard = prepare_matchup_adapter_isolation_guard(
                        model, adapter_optimizer, work
                    )
                    total, metrics = batch_losses(
                        model,
                        work,
                        value_weight=float(cfg.value_loss_weight),
                        aux_weight=0.0,
                        opp_hand_weight=0.0,
                        opp_remainder_weight=0.0,
                        alakazam_guide_weight=0.0,
                        # Dormant adapters are an isolated residual fit. They
                        # intentionally consume neither legacy guide imitation nor
                        # the future strategic-head curriculum.
                        current_deck_guide_training_mode=(
                            GUIDE_TRAINING_MODE_LEGACY
                        ),
                        lethal_threat_weight=0.0,
                        prize_race_weight=0.0,
                        pure_rl=True,
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_baseline_cache=awr_baseline_cache,
                        matchup_adapter_training=True,
                    )
                    if metrics.n_matchup_adapter_rows <= 0:
                        raise RuntimeError(
                            "ticketed adapter batch produced zero routed rows"
                        )
                    if not torch.isfinite(total):
                        raise FloatingPointError(
                            "non-finite dormant adapter loss"
                        )
                    total.backward()
                    assert_matchup_adapter_isolation_guard(
                        model, adapter_optimizer, guard, after_step=False
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.matchup_adapter_bank.parameters(),
                        float(cfg.grad_clip),
                    )
                    adapter_optimizer.step()
                    assert_matchup_adapter_isolation_guard(
                        model, adapter_optimizer, guard, after_step=True
                    )
                    assert_matchup_adapter_training_contract(
                        model, optimizer=adapter_optimizer, base_state=base_state
                    )
                    return metrics

                def _clear_adapter_oom() -> None:
                    adapter_optimizer.zero_grad(set_to_none=True)
                    if next(model.parameters()).device.type == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()

                completed = process_with_oom_splitting(
                    batch,
                    _fit_chunk,
                    on_split=_clear_adapter_oom,
                )
                for metrics in completed:
                    step_count += 1
                    row_count += int(metrics.n_matchup_adapter_rows)
                    last_metrics = metrics
                bar.set_postfix(
                    loss=f"{last_metrics.total_loss:.3f}",
                    rows=row_count,
                )
    finally:
        # Persist trained weights as dormant state only. Restore the ordinary
        # base parameter flags so its Adam optimizer remains a valid continuation
        # checkpoint, while the adapter bank is explicitly frozen and disabled.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
            parameter.grad = None
        model.matchup_adapter_bank.requires_grad_(False)
        model.matchup_adapter_bank.enabled = False
        model.cfg.matchup_adapters_enabled = False
    assert matchup_adapter_base_state(model).keys() == base_state.keys()
    changed_base = [
        name
        for name, value in matchup_adapter_base_state(model).items()
        if not torch.equal(value, base_state[name])
    ]
    if changed_base:
        raise AssertionError(f"dormant adapter phase changed base tensors: {changed_base[:5]}")

    prior_fit = dict(prior_fit or {})
    if prior_fit and prior_fit.get("schema") != "poke_bot.dormant_matchup_adapter_fit/v1":
        raise ValueError("prior dormant matchup adapter fit has an invalid schema")
    prior_route_sequences = {
        expert_id: int((prior_fit.get("route_sequences") or {}).get(expert_id, 0))
        for expert_id in route_ids
    }
    prior_route_decisions = {
        expert_id: int((prior_fit.get("route_decisions") or {}).get(expert_id, 0))
        for expert_id in route_ids
    }
    cumulative_route_sequences = {
        expert_id: prior_route_sequences[expert_id] + route_sequences[expert_id]
        for expert_id in route_ids
    }
    cumulative_route_decisions = {
        expert_id: prior_route_decisions[expert_id] + route_decisions[expert_id]
        for expert_id in route_ids
    }
    trained_archetype_ids = [
        expert_id
        for expert_id in route_ids
        if cumulative_route_decisions[expert_id] > 0
    ]
    dormant_archetype_ids = [
        expert_id
        for expert_id in route_ids
        if cumulative_route_decisions[expert_id] == 0
    ]
    fit = {
        "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
        "runtime_enabled": False,
        "base_frozen": True,
        "optimizer_scope": "matchup_adapter_bank_only",
        "activation_receipt": str(receipt_path),
        "activation_receipt_digest": checkpoint.checkpoint_digest(receipt_path),
        # These totals are cumulative because runtime activation must retain
        # proof for routes learned in earlier rehearsal/RL phases.  The
        # phase_* fields preserve an exact audit of this optimizer pass.
        "epochs": int(prior_fit.get("epochs") or 0) + epochs,
        "steps": int(prior_fit.get("steps") or 0) + step_count,
        "rows": int(prior_fit.get("rows") or 0) + row_count,
        "phase_epochs": epochs,
        "phase_steps": step_count,
        "phase_rows": row_count,
        "optimizer_state_restored": optimizer_restored,
        "route_sequences": cumulative_route_sequences,
        "route_decisions": cumulative_route_decisions,
        "phase_route_sequences": route_sequences,
        "phase_route_decisions": route_decisions,
        "trained_archetype_ids": trained_archetype_ids,
        "dormant_no_example_archetype_ids": dormant_archetype_ids,
        "zero_example_routes_remain_dormant": True,
        "last_metrics": dict(last_metrics.__dict__),
    }
    return fit, copy.deepcopy(adapter_optimizer.state_dict())


def rl_train_step(
    dataset: BootstrapDataset,
    *,
    base_ckpt: Union[str, Path],
    out_run_name: str,
    archetype_id: str,
    epochs: int,
    device: Optional[torch.device] = None,
    cfg: Optional[TrainConfig] = None,
    seed: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    parent_digest: Optional[str] = None,
    training_provenance: Optional[dict[str, Any]] = None,
    replace_existing: bool = False,
    device_resident: bool = False,
    device_resident_min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    """One immutable history-policy candidate fit for the RL loop.

    Unlike :func:`train_bootstrap`, each call is one RL iteration over a fresh
    replay window. Model, Adam/scaler state and global learner counters resume
    from ``base_ckpt``; early-stop selection remains local to the new window.
    Pure RL uses AWR on selected actions plus terminal value regression.  A
    stateless CUDA learner can pack the complete exact replay window once and
    keep the frozen baseline, shuffle order, features, and targets on-device;
    this removes repeated Python packing/H2D gaps between optimizer batches.

    Returns ``{"latest_path", "metrics"}``. Deterministic-ish via ``seed``.
    """
    actual_parent_digest = checkpoint.checkpoint_digest(base_ckpt)
    if parent_digest is not None and str(parent_digest) != actual_parent_digest:
        raise ValueError(
            "pure-RL parent_digest does not match base checkpoint bytes: "
            f"supplied={parent_digest!r} actual={actual_parent_digest!r} "
            f"path={Path(base_ckpt).expanduser().resolve()}"
        )
    # Always persist the verified identity, including for callers which did
    # not supply one.  A nullable lineage field is too easy to misinterpret.
    parent_digest = actual_parent_digest
    cfg = cfg or TrainConfig()
    cfg.current_deck_guide_training_mode = canonical_guide_training_mode(
        cfg.current_deck_guide_training_mode
    )
    if int(cfg.dormant_matchup_adapter_epochs) < 0:
        raise ValueError("dormant matchup adapter epochs cannot be negative")
    if float(cfg.dormant_matchup_adapter_lr) <= 0.0:
        raise ValueError("dormant matchup adapter learning rate must be positive")
    if int(cfg.dormant_matchup_adapter_max_decisions_per_batch) <= 0:
        raise ValueError(
            "dormant matchup adapter decision cap must be positive"
        )
    if device_resident and int(cfg.dormant_matchup_adapter_epochs) > 0:
        raise ValueError(
            "dormant matchup adapter training requires retained ticketed game "
            "sequences and is not compatible with the stateless resident corpus"
        )
    if device_resident and cfg.capture_awr_weight_distribution:
        raise ValueError(
            "exact shadow AWR weight quantiles currently require host-batched "
            "temporal training"
        )
    device = device or device_mod.training_device(
        prefer_name=config.HARDWARE.train_gpu_name, allow_cpu=False
    )
    torch.manual_seed(seed)
    random.seed(seed)

    model = load_model_from_checkpoint(base_ckpt, device=device)
    if (
        cfg.current_deck_guide_training_mode
        == GUIDE_TRAINING_MODE_STRATEGIC
    ):
        assert_strategic_curriculum_model_contract(
            model,
            setup_board_outcome_loss_weight=(
                cfg.setup_board_outcome_loss_weight
            ),
        )
        assert_strategic_curriculum_receipt_contract(
            specialist_id=archetype_id,
            curriculum_spec=cfg.current_deck_guide_curriculum_spec,
            head_role_map=cfg.current_deck_guide_head_role_map,
            validation_receipt=(
                cfg.current_deck_guide_curriculum_validation_receipt
            ),
        )
        if device_resident:
            raise ValueError(
                "strategic curriculum RL requires temporal host batching"
            )
    if model.decision_context not in {"history", "stateless"}:
        raise ValueError(
            "trusted RL training requires a history or stateless checkpoint"
        )
    initial_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    model.train()
    # Dormant matchup adapters are present in the architecture but frozen by
    # default. Excluding them preserves the legacy learner optimizer's exact
    # parameter-group cardinality/order and guarantees ordinary RL cannot train
    # or weight-decay the staged bank.
    ordinary_trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        ordinary_trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    base_step = 0
    base_epoch = 0
    base_rl_iteration = 0
    optimizer_state_restored = False
    learner_ckpt: Optional[dict[str, Any]] = None
    if cfg.pure_rl:
        learner_ckpt = checkpoint.load_checkpoint(base_ckpt, map_location=device)
        base_step = int(learner_ckpt.get("step", 0))
        base_epoch = int(learner_ckpt.get("epoch", 0))
        base_rl_iteration = int(learner_ckpt.get("rl_iteration", 0))
        if "optimizer_state_dict" in learner_ckpt:
            try:
                checkpoint.apply_checkpoint(
                    learner_ckpt,
                    optimizer=optimizer,
                    scaler=scaler if use_amp else None,
                    restore_rng=False,
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    "pure-RL parent optimizer state is incompatible with the "
                    "loaded model; refusing a silent Adam reset"
                ) from exc
            optimizer_state_restored = True
            # The checkpoint supplies moments/counters; the current iteration's
            # explicit config remains authoritative for tunable hyperparameters.
            for group in optimizer.param_groups:
                group["lr"] = float(cfg.lr)
                group["weight_decay"] = float(cfg.weight_decay)

    parent_payload = (
        learner_ckpt
        if learner_ckpt is not None
        else checkpoint.load_checkpoint(base_ckpt, map_location="cpu")
    )
    parent_expanded_contract = dict(
        (parent_payload.get("extra") or {}).get("expanded_head_training") or {}
    )
    requested_expanded_weights = canonical_expanded_loss_weights(
        cfg.expanded_head_loss_weights
    )
    if bool(getattr(model, "expanded_heads_enabled", False)):
        if not parent_expanded_contract:
            raise ValueError(
                "expanded-head checkpoint lacks tensor-bound training metadata"
            )
        inherited_expanded_weights = canonical_expanded_loss_weights(
            dict(parent_expanded_contract.get("loss_weights") or {})
        )
        if not any(requested_expanded_weights.values()):
            cfg.expanded_head_loss_weights = inherited_expanded_weights
        elif requested_expanded_weights != inherited_expanded_weights:
            changed = {
                name
                for name in requested_expanded_weights
                if requested_expanded_weights[name]
                != inherited_expanded_weights[name]
            }
            safe_teal_rebalance = bool(
                cfg.expanded_head_weight_migration_reason
                == "receipt_backed_teal_auxiliary_head_rebalance_v1"
                and changed == {"tactical_outcome"}
                and inherited_expanded_weights["tactical_outcome"] == 0.05
                and requested_expanded_weights["tactical_outcome"] == 0.01
            )
            if not safe_teal_rebalance:
                raise ValueError(
                    "RL expanded-head weights drift from the parent checkpoint"
                )
            cfg.expanded_head_loss_weights = requested_expanded_weights
    elif any(requested_expanded_weights.values()):
        raise ValueError("V5 RL checkpoint cannot receive expanded-head losses")

    # Keep only decision-bearing games, then reuse bootstrap's val split / ES.
    usable_sequences = [s for s in dataset.sequences if s.decisions]
    if model.decision_context == "history":
        usable_sequences, truncated_sequences = cap_history_sequences(
            usable_sequences, model.max_context
        )
        print(
            f"[rl-train] game-bounded temporal context={model.max_context} "
            f"truncated_sequences={truncated_sequences}",
            flush=True,
        )
    usable = BootstrapDataset(sequences=usable_sequences)
    train_seqs, val_seqs = split_dataset(usable, cfg.val_frac, seed)
    if not train_seqs:
        train_seqs, val_seqs = list(usable.sequences), []
    if not train_seqs:
        raise ValueError("RL training dataset has no usable decision sequences")
    if float(cfg.alakazam_guide_loss_weight) > 0.0:
        guide_rows = (
            count_usable_strategic_guide_rows([*train_seqs, *val_seqs])
            if cfg.current_deck_guide_training_mode
            == GUIDE_TRAINING_MODE_STRATEGIC
            else count_usable_alakazam_guide_rows([*train_seqs, *val_seqs])
        )
        if guide_rows <= 0:
            raise ValueError(
                "nonzero current-deck guide loss has no usable guide rows; "
                "verify the selected current-deck guide target switch and "
                "scorer coverage"
            )
        print(
            f"[rl-train] current-deck guide rows={guide_rows} "
            f"weight={float(cfg.alakazam_guide_loss_weight):.4f}",
            flush=True,
        )
    resident_corpus: Optional[DeviceResidentBootstrapCorpus] = None
    resident_baseline: Optional[torch.Tensor] = None
    resident_batch_size = int(cfg.max_decisions_per_batch)
    if device_resident:
        if device.type != "cuda":
            raise ValueError("device-resident RL training requires CUDA")
        if model.decision_context != "stateless":
            raise ValueError(
                "device-resident RL training requires a stateless checkpoint"
            )
        if not cfg.pure_rl or not cfg.awr_freeze_baseline:
            raise ValueError(
                "device-resident RL requires pure RL with a frozen AWR baseline"
            )
        try:
            resident_corpus = DeviceResidentBootstrapCorpus.from_splits(
                train_seqs,
                val_seqs,
                device=device,
                min_free_gib=float(device_resident_min_free_gib),
                exact_card_vocab=int(model.belief_card_vocab),
            )
            resident_batch_size = _fit_device_batch_size(
                model,
                resident_corpus,
                requested=cfg.max_decisions_per_batch,
                cfg=cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                min_step_free_gib=max(
                    2.0, min(4.0, float(device_resident_min_free_gib) / 2.0)
                ),
            )
        except (MemoryError, torch.OutOfMemoryError) as exc:
            resident_corpus = None
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "[device-corpus-rl] capacity fallback to bounded CPU batches: "
                f"{exc}",
                flush=True,
            )
        except RuntimeError as exc:
            # Some CUDA/PyTorch combinations surface allocation failure as a
            # plain RuntimeError instead of torch.OutOfMemoryError.  Preserve
            # fail-closed behavior for every other runtime/data error.
            if not config.is_cuda_oom(exc):
                raise
            resident_corpus = None
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "[device-corpus-rl] CUDA capacity fallback to bounded CPU "
                f"batches: {exc}",
                flush=True,
            )
        if resident_corpus is not None:
            # The GPU corpus is now authoritative. Keep no second copy of the
            # 25+ GiB replay window in Python objects while optimization runs.
            dataset.sequences.clear()
            usable.sequences.clear()
            train_seqs.clear()
            val_seqs.clear()
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "[device-corpus-rl] ALL replay features/targets resident "
                f"bytes={resident_corpus.input_bytes} "
                f"samples={resident_corpus.total_samples} "
                f"batch={resident_batch_size}; host GameSequence corpus released",
                flush=True,
            )
    awr_baseline_cache: Optional[dict[tuple[int, int, int], float]] = None
    if resident_corpus is not None:
        resident_baseline = _device_exact_value_cache(
            model,
            resident_corpus,
            cfg=cfg,
            batch_size=resident_batch_size,
            desc="rl-prep baseline",
        )
    elif cfg.pure_rl and cfg.awr_freeze_baseline:
        awr_baseline_cache = _precompute_awr_baseline_cache(
            model,
            [*train_seqs, *val_seqs],
            cfg=cfg,
            desc="rl-prep baseline",
        )
    agreement_sequences = [*train_seqs, *val_seqs]
    if resident_corpus is not None:
        parent_predictions: Union[list[int], torch.Tensor] = (
            _device_exact_policy_predictions(
                model,
                resident_corpus,
                cfg=cfg,
                batch_size=resident_batch_size,
                desc="rl-agreement parent",
            )
        )
    else:
        parent_predictions = _policy_argmax_predictions(
            model,
            agreement_sequences,
            cfg=cfg,
            awr_baseline_cache=awr_baseline_cache,
            desc="rl-agreement parent",
        )
    patience = max(0, int(cfg.early_stop_patience))
    patience_left = patience
    best_metric = float("inf")
    best_state: Optional[dict[str, Any]] = None
    best_optimizer_state: Optional[dict[str, Any]] = None
    best_scaler_state: Optional[dict[str, Any]] = None
    best_step = base_step
    best_epoch_offset = 0
    best_train_m = BatchMetrics()
    best_val_m = BatchMetrics()
    best_validation_source = (
        "heldout"
        if (
            resident_corpus.val_samples
            if resident_corpus is not None
            else val_seqs
        )
        else "current_data"
    )
    last: BatchMetrics = BatchMetrics()
    last_epoch_optimizer_sps = 0.0
    stepped_epochs = 0
    step = base_step
    epoch_bar = tqdm(range(max(1, epochs)), desc="rl-epochs", leave=False, unit="ep")
    for epoch in epoch_bar:
        model.train()
        epoch_started = time.monotonic()
        epoch_samples = 0
        exact_epoch_awr_weights: Optional[list[float]] = (
            [] if cfg.capture_awr_weight_distribution else None
        )
        if resident_corpus is not None:
            batches: Sequence[Any] = resident_corpus.batches(
                train=True,
                batch_size=resident_batch_size,
                shuffle=True,
                seed=seed,
                epoch=epoch,
            )
        else:
            batches = _iter_game_batches(
                train_seqs,
                cfg.games_per_batch,
                cfg.max_decisions_per_batch,
                shuffle=True,
                seed=seed,
                epoch=epoch,
            )
        parts: list[BatchMetrics] = []
        batch_bar = tqdm(
            batches,
            desc=f"rl-train ep{epoch}",
            leave=False,
            unit="batch",
        )
        for batch in batch_bar:
            def _apply_update(total: torch.Tensor, bm: BatchMetrics) -> BatchMetrics:
                nonlocal step
                if bm.n_decisions == 0:
                    return bm
                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite training loss: {total}")
                scaler.scale(total).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    ordinary_trainable_parameters, cfg.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                step += 1
                return bm

            def _train_chunk(work: list[GameSequence]) -> BatchMetrics:
                optimizer.zero_grad(set_to_none=True)
                chunk_awr_weights: Optional[list[float]] = (
                    [] if exact_epoch_awr_weights is not None else None
                )
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    total, bm = batch_losses(
                        model,
                        work,
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        current_deck_guide_training_mode=(
                            cfg.current_deck_guide_training_mode
                        ),
                        setup_board_outcome_loss_weight=(
                            cfg.setup_board_outcome_loss_weight
                        ),
                        combo_state_loss_weight=cfg.combo_state_loss_weight,
                        expanded_head_weights=cfg.expanded_head_loss_weights,
                        pure_rl=bool(cfg.pure_rl),
                        awr_beta=float(cfg.awr_beta),
                        awr_weight_max=float(cfg.awr_weight_max),
                        awr_normalize_advantages=bool(cfg.awr_normalize_advantages),
                        entropy_bonus=float(cfg.entropy_bonus),
                        awr_baseline_cache=awr_baseline_cache,
                        awr_weight_sink=chunk_awr_weights,
                    )
                applied = _apply_update(total, bm)
                if (
                    exact_epoch_awr_weights is not None
                    and chunk_awr_weights is not None
                ):
                    exact_epoch_awr_weights.extend(chunk_awr_weights)
                return applied

            def _clear_oom() -> None:
                optimizer.zero_grad(set_to_none=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if resident_corpus is not None:
                assert resident_baseline is not None
                sample_ids = batch
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                ):
                    total, resident_metrics = device_exact_batch_losses(
                        model,
                        resident_corpus,
                        sample_ids,
                        baseline_pred=resident_baseline.index_select(
                            0, sample_ids
                        ),
                        value_weight=cfg.value_loss_weight,
                        aux_weight=cfg.aux_loss_weight,
                        opp_hand_weight=cfg.opp_hand_loss_weight,
                        opp_remainder_weight=cfg.opp_remainder_loss_weight,
                        lethal_threat_weight=cfg.lethal_threat_loss_weight,
                        prize_race_weight=cfg.prize_race_loss_weight,
                        alakazam_guide_weight=cfg.alakazam_guide_loss_weight,
                        current_deck_guide_training_mode=(
                            cfg.current_deck_guide_training_mode
                        ),
                        awr_beta=cfg.awr_beta,
                        awr_weight_max=cfg.awr_weight_max,
                        awr_normalize_advantages=(
                            cfg.awr_normalize_advantages
                        ),
                        entropy_bonus=cfg.entropy_bonus,
                    )
                completed_parts = [_apply_update(total, resident_metrics)]
            else:
                completed_parts = process_with_oom_splitting(
                    batch,
                    _train_chunk,
                    on_split=_clear_oom,
                )
            for bm in completed_parts:
                if bm.n_decisions > 0:
                    parts.append(bm)
                    epoch_samples += int(bm.n_decisions)
            visible = next(
                (bm for bm in reversed(completed_parts) if bm.n_decisions > 0),
                None,
            )
            if visible is not None:
                batch_bar.set_postfix(
                    loss=f"{visible.total_loss:.3f}",
                    p=f"{visible.policy_loss:.3f}",
                    v=f"{visible.value_loss:.3f}",
                    aux=(
                        "off"
                        if cfg.aux_loss_weight == 0
                        else f"{visible.aux_loss:.3f}/"
                        f"{visible.n_archetype_rows}"
                    ),
                    hand=(
                        "off"
                        if cfg.opp_hand_loss_weight == 0
                        else f"{visible.opp_hand_loss:.3f}/"
                        f"{visible.n_opp_hand_rows}"
                    ),
                    rem=(
                        "off"
                        if cfg.opp_remainder_loss_weight == 0
                        else f"{visible.opp_remainder_loss:.3f}/"
                        f"{visible.n_opp_remainder_rows}"
                    ),
                    lethal=(
                        "off"
                        if cfg.lethal_threat_loss_weight == 0
                        else f"{visible.lethal_threat_loss:.3f}/"
                        f"{visible.n_lethal_threat_rows}"
                    ),
                    prize=(
                        "off"
                        if cfg.prize_race_loss_weight == 0
                        else f"{visible.prize_race_loss:.3f}/"
                        f"{visible.n_prize_race_rows}"
                    ),
                    guide=(
                        "off"
                        if cfg.alakazam_guide_loss_weight == 0
                        else f"{visible.alakazam_guide_loss:.3f}/"
                        f"{visible.n_alakazam_guide_rows}"
                    ),
                    acc=f"{visible.policy_acc:.2%}",
                    sps=f"{epoch_samples / max(time.monotonic() - epoch_started, 1e-6):.0f}",
                )
        last = _set_exact_awr_weight_quantiles(
            _merge_metrics(parts), exact_epoch_awr_weights or ()
        )
        last_epoch_optimizer_sps = epoch_samples / max(
            time.monotonic() - epoch_started, 1e-6
        )
        if resident_corpus is not None and resident_corpus.val_samples:
            assert resident_baseline is not None
            val_m = _evaluate_device_exact_corpus(
                model,
                resident_corpus,
                cfg=cfg,
                batch_size=resident_batch_size,
                baseline=resident_baseline,
                desc=f"rl-val ep{epoch}",
            )
            metric = val_m.total_loss
        elif val_seqs:
            val_m = evaluate(
                model,
                val_seqs,
                cfg=cfg,
                desc=f"rl-val ep{epoch}",
                awr_baseline_cache=awr_baseline_cache,
            )
            metric = val_m.total_loss
        else:
            val_m = last
            metric = last.total_loss
        stepped_epochs = epoch + 1

        is_best = metric < best_metric - 1e-5
        if is_best:
            best_metric = metric
            best_train_m = last
            best_val_m = val_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scaler_state = copy.deepcopy(scaler.state_dict()) if use_amp else None
            best_step = step
            best_epoch_offset = epoch + 1
            if patience > 0:
                patience_left = patience
            tqdm.write(
                f"[rl-train] NEW BEST epoch={epoch} "
                f"{'val' if val_seqs else 'train'}_loss={metric:.4f} "
                f"acc={val_m.policy_acc:.2%}"
            )
        elif patience > 0:
            patience_left -= 1
            tqdm.write(
                f"[rl-train] epoch={epoch} train_loss={last.total_loss:.4f} "
                f"{'val' if val_seqs else 'train'}_loss={metric:.4f} "
                f"acc={val_m.policy_acc:.2%} patience={patience_left}"
            )

        epoch_bar.set_postfix(
            loss=f"{last.total_loss:.3f}",
            p=f"{last.policy_loss:.3f}",
            v=f"{last.value_loss:.3f}",
            aux=(
                "off"
                if cfg.aux_loss_weight == 0
                else f"{last.aux_loss:.3f}/{last.n_archetype_rows}"
            ),
            hand=(
                "off"
                if cfg.opp_hand_loss_weight == 0
                else f"{last.opp_hand_loss:.3f}/{last.n_opp_hand_rows}"
            ),
            rem=(
                "off"
                if cfg.opp_remainder_loss_weight == 0
                else f"{last.opp_remainder_loss:.3f}/"
                f"{last.n_opp_remainder_rows}"
            ),
            lethal=(
                "off"
                if cfg.lethal_threat_loss_weight == 0
                else f"{last.lethal_threat_loss:.3f}/"
                f"{last.n_lethal_threat_rows}"
            ),
            prize=(
                "off"
                if cfg.prize_race_loss_weight == 0
                else f"{last.prize_race_loss:.3f}/"
                f"{last.n_prize_race_rows}"
            ),
            guide=(
                "off"
                if cfg.alakazam_guide_loss_weight == 0
                else f"{last.alakazam_guide_loss:.3f}/"
                f"{last.n_alakazam_guide_rows}"
            ),
            acc=f"{last.policy_acc:.2%}",
            best=f"{best_metric:.3f}",
            pat=patience_left if patience > 0 else "-",
        )

        if patience > 0 and patience_left <= 0:
            tqdm.write(
                f"[rl-early-stop] patience exhausted at epoch={epoch} "
                f"best_loss={best_metric:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if best_optimizer_state is not None:
            optimizer.load_state_dict(best_optimizer_state)
        if use_amp and best_scaler_state is not None:
            scaler.load_state_dict(best_scaler_state)
        step = best_step
        last = best_train_m
        validation_metrics = best_val_m
    else:
        validation_metrics = last

    if resident_corpus is not None:
        candidate_predictions: Union[list[int], torch.Tensor] = (
            _device_exact_policy_predictions(
                model,
                resident_corpus,
                cfg=cfg,
                batch_size=resident_batch_size,
                desc="rl-agreement candidate",
            )
        )
    else:
        candidate_predictions = _policy_argmax_predictions(
            model,
            agreement_sequences,
            cfg=cfg,
            awr_baseline_cache=awr_baseline_cache,
            desc="rl-agreement candidate",
        )
    if isinstance(parent_predictions, torch.Tensor):
        if not isinstance(candidate_predictions, torch.Tensor):
            raise RuntimeError("resident candidate agreement type mismatch")
        parent_rows = int(parent_predictions.numel())
        candidate_rows = int(candidate_predictions.numel())
        if parent_rows <= 0 or parent_rows != candidate_rows:
            raise RuntimeError(
                "cannot measure resident parent/candidate policy agreement: "
                f"parent_rows={parent_rows} candidate_rows={candidate_rows}"
            )
        policy_prev_agreement = float(
            (parent_predictions == candidate_predictions)
            .float()
            .mean()
            .item()
        )
    else:
        if isinstance(candidate_predictions, torch.Tensor):
            raise RuntimeError("host candidate agreement type mismatch")
        parent_rows = len(parent_predictions)
        candidate_rows = len(candidate_predictions)
        if parent_rows <= 0 or parent_rows != candidate_rows:
            raise RuntimeError(
                "cannot measure parent/candidate policy agreement: "
                f"parent_rows={parent_rows} candidate_rows={candidate_rows}"
            )
        policy_prev_agreement = sum(
            int(parent == candidate)
            for parent, candidate in zip(
                parent_predictions, candidate_predictions
            )
        ) / float(parent_rows)
    prior_adapter_optimizer_state = None
    prior_adapter_fit = None
    if learner_ckpt is not None:
        prior_adapter_optimizer_state = dict(
            (learner_ckpt.get("extra") or {}).get(
                "dormant_matchup_adapter_optimizer_state"
            )
            or {}
        )
        prior_adapter_fit = dict(
            (learner_ckpt.get("extra") or {}).get(
                "dormant_matchup_adapter_fit"
            )
            or {}
        )
    dormant_adapter_fit, dormant_adapter_optimizer_state = (
        _train_dormant_matchup_adapter_phase(
            model,
            agreement_sequences,
            cfg=cfg,
            base_rl_iteration=base_rl_iteration,
            target_rl_iteration=(
                int(training_provenance["iteration"])
                if training_provenance is not None
                and training_provenance.get("iteration") is not None
                else int(base_rl_iteration) + 1
            ),
            awr_baseline_cache=awr_baseline_cache,
            seed=seed,
            prior_optimizer_state=prior_adapter_optimizer_state,
            prior_fit=prior_adapter_fit,
        )
        if int(cfg.dormant_matchup_adapter_epochs) > 0
        else ({}, {})
    )
    rl_metrics = dict(last.__dict__)
    rl_metrics["policy_prev_agreement"] = policy_prev_agreement
    rl_metrics["optimizer_samples_per_second"] = last_epoch_optimizer_sps
    rl_metrics["awr_weight_quantiles_exact"] = bool(
        cfg.capture_awr_weight_distribution
    )
    rl_expanded_contract: dict[str, Any] = {}
    if bool(getattr(model, "expanded_heads_enabled", False)):
        train_expanded = dict(last.expanded_head_metrics or {})
        validation_expanded = dict(
            validation_metrics.expanded_head_metrics or {}
        )
        train_labeled = dict(train_expanded.get("labeled") or {})
        validation_labeled = dict(validation_expanded.get("labeled") or {})
        train_masked = dict(train_expanded.get("masked") or {})
        validation_masked = dict(validation_expanded.get("masked") or {})
        train_total = dict(train_expanded.get("total") or {})
        validation_total = dict(validation_expanded.get("total") or {})
        train_losses = dict(train_expanded.get("losses") or {})
        validation_losses = dict(validation_expanded.get("losses") or {})
        gradient_enabled = [
            name
            for name in EXPANDED_HEAD_IDS
            if float(cfg.expanded_head_loss_weights.get(name, 0.0)) > 0.0
        ]
        prior_trained = {
            str(name)
            for name in parent_expanded_contract.get("trained_heads") or ()
        }
        trained_this_iteration = [
            name
            for name in gradient_enabled
            if int(train_labeled.get(name, 0)) > 0
            and int(validation_labeled.get(name, 0)) > 0
        ]
        trained = [
            name
            for name in EXPANDED_HEAD_IDS
            if name in prior_trained or name in trained_this_iteration
        ]
        remaining_warm = tuple(
            module
            for module in (
                getattr(model, "warm_started_expanded_heads", ()) or ()
            )
            if module.removesuffix("_head") not in trained
        )
        model.warm_started_expanded_heads = remaining_warm
        heads: dict[str, dict[str, Any]] = {}
        coverage: dict[str, dict[str, Any]] = {}
        for name in EXPANDED_HEAD_IDS:
            labeled_rows = int(train_labeled.get(name, 0)) + int(
                validation_labeled.get(name, 0)
            )
            masked_rows = int(train_masked.get(name, 0)) + int(
                validation_masked.get(name, 0)
            )
            total_rows = int(train_total.get(name, 0)) + int(
                validation_total.get(name, 0)
            )
            coverage[name] = {
                "labeled_rows": labeled_rows,
                "masked_rows": masked_rows,
                "total_rows": total_rows,
                "coverage": (
                    float(labeled_rows) / total_rows
                    if total_rows > 0
                    else 0.0
                ),
            }
            heads[name] = {
                "present": True,
                "trained": name in trained,
                "trained_this_iteration": name in trained_this_iteration,
                "gradient_enabled": name in gradient_enabled,
                "runtime_enabled": False,
                "loss_weight": float(
                    cfg.expanded_head_loss_weights.get(name, 0.0)
                ),
                "train_loss": train_losses.get(name),
                "validation_loss": validation_losses.get(name),
                "train_labeled_rows": int(train_labeled.get(name, 0)),
                "validation_labeled_rows": int(
                    validation_labeled.get(name, 0)
                ),
                **coverage[name],
            }
        fused_action_path = bool(
            getattr(model, "decision_fusion_enabled", False)
            and getattr(model, "decision_fusion_runtime_enabled", False)
        )
        rl_expanded_contract = {
            **parent_expanded_contract,
            "schema": "poke_bot.expanded_head_training/v1",
            "stage": "rl",
            "epoch": int(base_rl_iteration + 1),
            "architecture_present_heads": list(EXPANDED_HEAD_IDS),
            "trained_heads": trained,
            "trained_this_iteration": trained_this_iteration,
            "gradient_enabled_heads": gradient_enabled,
            "runtime_enabled_heads": [],
            "loss_weights": dict(cfg.expanded_head_loss_weights),
            "loss_weight_migration": (
                {
                    "reason": cfg.expanded_head_weight_migration_reason,
                    "parent_loss_weights": dict(inherited_expanded_weights),
                    "current_loss_weights": dict(
                        cfg.expanded_head_loss_weights
                    ),
                    "tensor_values_changed_by_migration": False,
                    "optimizer_state_preserved": optimizer_state_restored,
                }
                if requested_expanded_weights != inherited_expanded_weights
                else parent_expanded_contract.get("loss_weight_migration")
            ),
            "train_metrics": {
                name: {"loss": train_losses.get(name)}
                for name in EXPANDED_HEAD_IDS
            },
            "validation_metrics": {
                name: {"loss": validation_losses.get(name)}
                for name in EXPANDED_HEAD_IDS
            },
            "coverage": coverage,
            "heads": heads,
            "calibration": {
                "train": dict(train_expanded.get("calibration") or {}),
                "validation": dict(
                    validation_expanded.get("calibration") or {}
                ),
            },
            "warm_started_heads_remaining": list(remaining_warm),
            "shadow_only": not fused_action_path,
            "flat_policy_authoritative": not fused_action_path,
            "authoritative_action_path": (
                "fused_policy" if fused_action_path else "flat_policy"
            ),
        }
    strategic_training_record = (
        _strategic_curriculum_training_record(
            cfg=cfg,
            train_metrics=last,
            validation_metrics=validation_metrics,
        )
        if cfg.current_deck_guide_training_mode
        == GUIDE_TRAINING_MODE_STRATEGIC
        else {}
    )

    delta_sq = 0.0
    base_sq = 0.0
    for name, value in model.state_dict().items():
        current = value.detach().cpu().float()
        base = initial_state[name].float()
        delta_sq += float(torch.sum((current - base) ** 2).item())
        base_sq += float(torch.sum(base ** 2).item())
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"non-finite candidate parameter: {name}")
    update_norm = math.sqrt(delta_sq)
    relative_update_norm = update_norm / max(math.sqrt(base_sq), 1e-12)

    ckpt = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler if use_amp else None,
        step=step,
        epoch=int(base_epoch + (best_epoch_offset or stepped_epochs)),
        rl_iteration=int(base_rl_iteration + 1),
        best_metric=best_metric if best_state is not None else last.total_loss,
        early_stop_state={
            "patience_left": patience_left,
            "best_metric": best_metric if best_state is not None else last.total_loss,
        },
        archetype_id=archetype_id,
        model_id=out_run_name,
        model_config=model.cfg,
        extra={
            "pure_rl": bool(cfg.pure_rl),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "rl_metrics": rl_metrics,
            "validation_metrics": validation_metrics.__dict__,
            "validation_source": best_validation_source,
            "rl_epochs_ran": stepped_epochs,
            "rl_epochs_cap": int(epochs),
            "rl_selected_epoch_offset": int(best_epoch_offset or stepped_epochs),
            "global_step": int(step),
            "optimizer_parent_step": int(base_step),
            "optimizer_state_restored": optimizer_state_restored,
            "awr_baseline_mode": (
                "frozen_device_resident"
                if resident_baseline is not None
                else "frozen_precomputed"
                if awr_baseline_cache is not None
                else "detached_online"
            ),
            "awr_baseline_rows": (
                int(resident_baseline.numel())
                if resident_baseline is not None
                else len(awr_baseline_cache or {})
            ),
            "awr_baseline_implementation": (
                "device_resident"
                if resident_baseline is not None
                else (
                    "value_only_length_bucketed_padded_prefetch_v1"
                    if bool(cfg.awr_pack_temporal_baseline)
                    else "value_only_exact_temporal_prefetch_v1"
                )
                if awr_baseline_cache is not None
                and bool(cfg.awr_value_only_baseline)
                else "full_policy_reference"
                if awr_baseline_cache is not None
                else "detached_online"
            ),
            "device_resident_rl": resident_corpus is not None,
            "device_resident_bytes": (
                int(resident_corpus.input_bytes)
                if resident_corpus is not None
                else 0
            ),
            "device_resident_batch_size": (
                int(resident_batch_size)
                if resident_corpus is not None
                else None
            ),
            "device_resident_build_seconds": (
                float(resident_corpus.build_seconds)
                if resident_corpus is not None
                else None
            ),
            "device_resident_samples": (
                int(resident_corpus.total_samples)
                if resident_corpus is not None
                else 0
            ),
            "training_contract": "causal_realized_history",
            "parent_digest": parent_digest,
            "training_provenance": dict(training_provenance or {}),
            "update_norm_l2": update_norm,
            "relative_update_norm_l2": relative_update_norm,
            "policy_prev_agreement": policy_prev_agreement,
            "policy_prev_agreement_rows": parent_rows,
            "dormant_matchup_adapter_fit": dormant_adapter_fit,
            "dormant_matchup_adapter_optimizer_state": (
                dormant_adapter_optimizer_state
            ),
            **(
                {"expanded_head_training": rl_expanded_contract}
                if rl_expanded_contract
                else {}
            ),
            **(
                {
                    "current_deck_guide_training": (
                        strategic_training_record
                    )
                }
                if strategic_training_record
                else {}
            ),
        },
    )
    if output_path is not None:
        # Iteration candidates are lineage evidence. Only an explicit caller
        # override may replace one; pure-RL provenance is never implicit write
        # permission.
        if replace_existing:
            saved_path = checkpoint.atomic_torch_save(ckpt, output_path)
        else:
            saved_path = checkpoint.immutable_torch_save(ckpt, output_path)
    else:
        mgr = checkpoint.CheckpointManager(out_run_name)
        saved = mgr.save(ckpt, is_best=False)
        saved_path = saved.get("latest", checkpoint.latest_path(out_run_name))
    digest = checkpoint.checkpoint_digest(saved_path)
    return {
        "latest_path": str(saved_path),
        "candidate_path": str(saved_path),
        "candidate_digest": digest,
        "parent_digest": parent_digest,
        "metrics": rl_metrics,
        "validation_metrics": validation_metrics.__dict__,
        "validation_source": best_validation_source,
        **(
            {"current_deck_guide_training": strategic_training_record}
            if strategic_training_record
            else {}
        ),
        "step": step,
        "parent_step": base_step,
        "optimizer_state_restored": optimizer_state_restored,
        "rl_iteration": base_rl_iteration + 1,
        "awr_baseline_mode": (
            "frozen_device_resident"
            if resident_baseline is not None
            else "frozen_precomputed"
            if awr_baseline_cache is not None
            else "detached_online"
        ),
        "awr_baseline_implementation": (
            "device_resident"
            if resident_baseline is not None
            else (
                "value_only_length_bucketed_padded_prefetch_v1"
                if bool(cfg.awr_pack_temporal_baseline)
                else "value_only_exact_temporal_prefetch_v1"
            )
            if awr_baseline_cache is not None
            and bool(cfg.awr_value_only_baseline)
            else "full_policy_reference"
            if awr_baseline_cache is not None
            else "detached_online"
        ),
        "device_resident_rl": resident_corpus is not None,
        "device_resident_bytes": (
            int(resident_corpus.input_bytes)
            if resident_corpus is not None
            else 0
        ),
        "device_resident_batch_size": (
            int(resident_batch_size) if resident_corpus is not None else None
        ),
        "device_resident_build_seconds": (
            float(resident_corpus.build_seconds)
            if resident_corpus is not None
            else None
        ),
        "device_resident_samples": (
            int(resident_corpus.total_samples)
            if resident_corpus is not None
            else 0
        ),
        "epochs_ran": stepped_epochs,
        "best_metric": best_metric if best_state is not None else last.total_loss,
        "update_norm_l2": update_norm,
        "relative_update_norm_l2": relative_update_norm,
        "policy_prev_agreement": policy_prev_agreement,
        "policy_prev_agreement_rows": parent_rows,
        "dormant_matchup_adapter_fit": dormant_adapter_fit,
    }


def load_model_from_checkpoint(
    path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> TemporalCabtTransformer:
    """Reconstruct architecture and load weights with belief-head warm-start.

    Trunk + existing heads load strictly. Only expected new belief/strategy
    head keys (``opp_hand_head.*``, ``opp_remainder_head.*``,
    ``lethal_threat_head.*``, ``prize_race_head.*``) may be missing; those
    heads stay randomly initialized and are recorded on
    ``model.warm_started_belief_heads`` (Scope A card heads trigger uniform
    particle fallback; Scope B strategy heads are root-gated separately).
    """
    device = device or device_mod.inference_device(allow_cpu=True)
    ckpt = checkpoint.load_checkpoint(path, map_location=device)
    snap = ckpt.get("model_config")
    if snap is None:
        cfg = config.MODEL
    elif not isinstance(snap, dict):
        raise ValueError(
            f"checkpoint {path} has invalid model_config type {type(snap).__name__}"
        )
    else:
        # The dormant bank was added after the first pure-RL checkpoints.  A
        # missing flag is the legacy spelling of explicit ``False``; never let
        # an ambient environment default turn a bankless checkpoint on.
        snap = dict(snap)
        snap.setdefault("matchup_adapters_enabled", False)
        # Router Format predates the serialized field. Historical checkpoints
        # that omit it are immutable Router Format 5 artifacts. Never inherit
        # the active process's Format 6 environment: frozen opponents load in
        # the same worker as a current Format 6 candidate, and that inheritance
        # would incorrectly require a registry the historical checkpoint never
        # carried.
        snap.setdefault("matchup_adapter_format", MATCHUP_ADAPTER_V5_FORMAT)
        snap.setdefault("matchup_adapter_registry", None)
        # Expanded strategic heads are an explicit V6 migration.  An ambient
        # environment variable must never materialize them while loading an
        # immutable V5 checkpoint.
        snap.setdefault("expanded_heads_enabled", False)
        snap.setdefault("decision_fusion_enabled", False)
        snap.setdefault("decision_fusion_runtime_enabled", False)
        snap.setdefault("decision_fusion_width", 16)
        # Setup-outcome supervision and per-head fusion routes are future-only
        # architecture additions. Missing fields in an immutable historical
        # checkpoint mean explicit ``False`` even when this process was
        # launched with future-specialist environment defaults.
        snap.setdefault("setup_board_outcome_head_enabled", False)
        snap.setdefault("combo_state_head_enabled", False)
        snap.setdefault("decision_fusion_dedicated_routes_enabled", False)
        snap.setdefault(
            "decision_fusion_dedicated_routes_runtime_enabled",
            False,
        )
        snap.setdefault("h10_capacity_enabled", False)
        snap.setdefault("h10_head_residual_width", 512)
        known = set(config.ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(snap) - known)
        if unknown:
            raise ValueError(
                f"checkpoint {path} has unsupported model_config fields: {unknown}"
            )
        try:
            cfg = config.ModelConfig(**snap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint {path} has incompatible model_config: {exc}"
            ) from exc

    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} is missing model_state_dict")
    dense = bool(getattr(cfg, "dense_card2vec", False))
    try:
        aux_classes = int(state["aux_head.3.weight"].shape[0])
        if dense or "card2vec.card_emb.weight" in state:
            # Factorized path: vocab sizes live on card2vec buffers / tables.
            if "card2vec.board_kind" in state:
                encoder_vocab = int(state["card2vec.board_kind"].shape[0])
            else:
                encoder_vocab = int(features.encoder_vocab_size())
            if "card2vec.option_kind" in state:
                decoder_vocab = int(state["card2vec.option_kind"].shape[0])
            else:
                decoder_vocab = int(features.decoder_vocab_size())
            # Ensure cfg matches dense modules even if snap omitted the flag.
            if not dense:
                cfg.dense_card2vec = True
                dense = True
        else:
            encoder_vocab = int(state["board_bag.weight"].shape[0])
            decoder_vocab = int(state["option_bag.weight"].shape[0])
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            f"checkpoint {path} lacks architecture-defining tensor shapes"
        ) from exc

    belief_card_vocab = belief_card_vocab_from_state(state)

    model = build_model(
        cfg,
        device=device,
        aux_archetype_classes=aux_classes,
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        belief_card_vocab=belief_card_vocab,
    )
    checkpoint.validate_matchup_adapter_contract(
        ckpt,
        model=model,
        source=path,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            f"checkpoint architecture/state incompatibility for {path}: "
            f"unexpected keys {unexpected}"
        )
    extra = dict(ckpt.get("extra") or {})
    expanded_migration = dict(extra.get("expanded_head_migration") or {})
    explicit_expanded_migration = bool(
        getattr(cfg, "expanded_heads_enabled", False)
        and expanded_migration.get("schema")
        == "poke_bot.expanded_head_migration/v1"
        and expanded_migration.get("target_architecture_schema")
        == EXPANDED_HEAD_SCHEMA
        and expanded_migration.get("target_schema")
        == EXPANDED_STRATEGIC_SCHEMA
        and expanded_migration.get("target_schema_digest")
        == TARGET_SCHEMA_DIGEST
        and expanded_migration.get("schedule_schema")
        == EXPANDED_SCHEDULE_SCHEMA
        and str(expanded_migration.get("schedule_digest") or "").startswith(
            "sha256:"
        )
        and len(str(expanded_migration.get("schedule_digest") or "")) == 71
        and expanded_migration.get("runtime_enabled_heads") == []
    )
    expanded_missing = [
        key for key in missing if is_allowed_missing_expanded_head_key(key)
    ]
    if expanded_missing and not explicit_expanded_migration:
        raise RuntimeError(
            f"checkpoint architecture/state incompatibility for {path}: "
            "expanded strategic tensors are missing without an explicit "
            "V6 migration contract"
        )
    if expanded_missing:
        expected_by_head: dict[str, set[str]] = {
            name: {
                key
                for key in model.state_dict()
                if key.startswith(f"{name}.")
            }
            for name in EXPANDED_HEAD_NAMES
        }
        missing_set = set(expanded_missing)
        partial = {
            name: sorted(keys & missing_set)
            for name, keys in expected_by_head.items()
            if keys & missing_set and not keys <= missing_set
        }
        if partial:
            raise RuntimeError(
                f"checkpoint {path} has partially missing expanded head tensors: "
                f"{partial}"
            )
    fusion_migration = dict(extra.get("decision_fusion_migration") or {})
    explicit_fusion_migration = bool(
        getattr(cfg, "decision_fusion_enabled", False)
        and fusion_migration.get("schema")
        == "poke_bot.causal_decision_fusion_migration/v1"
        and fusion_migration.get("target_schema")
        in {DECISION_FUSION_SCHEMA, DECISION_FUSION_V2_SCHEMA}
        and fusion_migration.get("zero_safe_initialization") is True
        and (
            fusion_migration.get("runtime_enabled") is False
            or (
                fusion_migration.get("runtime_enabled") is True
                and fusion_migration.get("activation_scope")
                == "isolated_specialist_bootstrap"
                and fusion_migration.get("serving_eligible") is False
            )
        )
    )
    explicit_fusion_v2_additive_migration = bool(
        getattr(cfg, "decision_fusion_enabled", False)
        and getattr(cfg, "decision_fusion_dedicated_routes_enabled", False)
        and fusion_migration.get("schema")
        == "poke_bot.causal_decision_fusion_v2_migration/v1"
        and fusion_migration.get("source_schema") == DECISION_FUSION_SCHEMA
        and fusion_migration.get("target_schema") == DECISION_FUSION_V2_SCHEMA
        and fusion_migration.get("zero_safe_initialization") is True
        and fusion_migration.get("runtime_enabled") is True
        and fusion_migration.get("activation_scope")
        == "isolated_specialist_bootstrap"
        and fusion_migration.get("serving_eligible") is False
        and fusion_migration.get("all_inherited_tensors_preserved") is True
    )
    fusion_missing = [
        key for key in missing if is_allowed_missing_decision_fusion_key(key)
    ]
    if fusion_missing and not (
        explicit_fusion_migration
        or explicit_fusion_v2_additive_migration
    ):
        raise RuntimeError(
            f"checkpoint architecture/state incompatibility for {path}: "
            "decision-fusion tensors are missing without an explicit "
            "zero-safe migration contract"
        )
    if fusion_missing:
        all_fusion = {
            key
            for key in model.state_dict()
            if is_allowed_missing_decision_fusion_key(key)
        }
        dedicated_fusion = {
            key
            for key in all_fusion
            if key.startswith("decision_fusion.dedicated_routes.")
        }
        expected_fusion = (
            dedicated_fusion
            if explicit_fusion_v2_additive_migration
            else all_fusion
        )
        if set(fusion_missing) != expected_fusion:
            raise RuntimeError(
                f"checkpoint {path} has partially missing decision-fusion "
                f"tensors: {sorted(set(fusion_missing) ^ expected_fusion)}"
            )
        if explicit_fusion_v2_additive_migration:
            expected_routes = sorted(
                {
                    key.split(".", 3)[2]
                    for key in dedicated_fusion
                }
            )
            if (
                fusion_migration.get("new_dedicated_route_names")
                != expected_routes
            ):
                raise RuntimeError(
                    f"checkpoint {path} has an invalid fusion-v2 route "
                    "migration inventory"
                )
            inherited_fusion = sorted(all_fusion - dedicated_fusion)
            if (
                fusion_migration.get("inherited_fusion_tensor_keys")
                != inherited_fusion
            ):
                raise RuntimeError(
                    f"checkpoint {path} has an invalid inherited fusion-v1 "
                    "tensor inventory"
                )
    migrated_auxiliary_prefixes: tuple[str, ...] = ()
    if explicit_fusion_migration or explicit_fusion_v2_additive_migration:
        expected_auxiliary_names = []
        if getattr(cfg, "setup_board_outcome_head_enabled", False):
            expected_auxiliary_names.append(SETUP_BOARD_OUTCOME_HEAD_NAME)
        if getattr(cfg, "combo_state_head_enabled", False):
            expected_auxiliary_names.append(COMBO_STATE_HEAD_NAME)
        if fusion_migration.get("new_auxiliary_head_names", []) != (
            expected_auxiliary_names
        ):
            raise RuntimeError(
                f"checkpoint {path} has an invalid fusion-v2 auxiliary-head "
                "migration inventory"
            )
        migrated_auxiliary_prefixes = tuple(
            f"{name}." for name in expected_auxiliary_names
        )
        for prefix in migrated_auxiliary_prefixes:
            expected_keys = {
                key for key in model.state_dict() if key.startswith(prefix)
            }
            missing_keys = {key for key in missing if key.startswith(prefix)}
            if missing_keys not in (set(), expected_keys):
                raise RuntimeError(
                    f"checkpoint {path} has partially missing migrated "
                    f"auxiliary-head tensors for {prefix[:-1]}"
                )
    disallowed_missing = [
        key
        for key in missing
        if not is_allowed_missing_belief_head_key(key)
        and not (
            explicit_expanded_migration
            and is_allowed_missing_expanded_head_key(key)
        )
        and not (
            (
                explicit_fusion_migration
                or explicit_fusion_v2_additive_migration
            )
            and is_allowed_missing_decision_fusion_key(key)
        )
        and not any(
            key.startswith(prefix) for prefix in migrated_auxiliary_prefixes
        )
    ]
    if disallowed_missing:
        raise RuntimeError(
            f"checkpoint architecture/state incompatibility for {path}: "
            f"missing non-belief-head keys {disallowed_missing}"
        )
    warm = belief_head_names_from_state_keys(missing)
    model.warm_started_belief_heads = warm
    expanded_warm = expanded_head_names_from_state_keys(missing)
    model.warm_started_expanded_heads = expanded_warm
    model.warm_started_decision_fusion = bool(fusion_missing)
    dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
    has_adapter_state = any(
        name.startswith("matchup_adapter_bank.") for name in state
    )
    if dormant:
        model.matchup_adapter_bank.dormant_provenance = dormant
    elif not has_adapter_state:
        # Legacy bankless checkpoints dynamically receive the constructor's
        # frozen zero bank. Persist the immutable source identity in the next
        # checkpoint without rewriting the parent commit or loop ledger.
        dormant_provenance = {
            "materialization": "legacy_bankless_dynamic_zero_init",
            "parent_checkpoint": str(Path(path).expanduser().resolve()),
            "parent_checkpoint_digest": checkpoint.checkpoint_digest(path),
        }
        raw_activation_receipt = os.environ.get(
            "POKEBOT_MATCHUP_ADAPTER_BOUNDARY_RECEIPT", ""
        ).strip()
        if raw_activation_receipt:
            # Startup also reconstructs the protected champion and held-out
            # anchor, which can legitimately predate the boundary parent.  The
            # receipt must still validate against its immutable learner, but a
            # zero-output bank may be attached to any legacy bankless model for
            # architecture compatibility.  Only the exact boundary parent is
            # eligible to carry ``activation_parent_match=True`` into training.
            activation_payload = json.loads(
                Path(raw_activation_receipt).expanduser().read_text(
                    encoding="utf-8"
                )
            )
            activation_parent = Path(
                str(activation_payload.get("parent_checkpoint") or "")
            ).expanduser()
            activation = validate_adapter_training_authorization(
                raw_activation_receipt,
                parent_checkpoint=activation_parent,
            )
            source_path = Path(path).expanduser().resolve()
            dormant_provenance.update(
                activation_receipt=str(activation.path),
                activation_receipt_digest=checkpoint.checkpoint_digest(
                    activation.path
                ),
                activation_completed_iteration=activation.completed_iteration,
                activation_first_eligible_iteration=activation.first_eligible_iteration,
                activation_parent_checkpoint=str(activation.parent_checkpoint),
                activation_parent_checkpoint_digest=(
                    activation.parent_checkpoint_digest
                ),
                activation_parent_match=(source_path == activation.parent_checkpoint),
            )
        model.matchup_adapter_bank.dormant_provenance = dormant_provenance
    if warm or expanded_warm or fusion_missing:
        extra["warm_start"] = True
        if warm:
            extra["warm_started_belief_heads"] = list(warm)
        if expanded_warm:
            extra["warm_started_expanded_heads"] = list(expanded_warm)
        if fusion_missing:
            extra["warm_started_decision_fusion"] = True
        extra["aux_heads_present"] = list(model.aux_heads_present)
        ckpt["extra"] = extra
    model.eval()
    return model
