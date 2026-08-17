"""Standalone public Prize-plan-v2 critic sidecar.

This module is intentionally isolated from the policy model and play loop.  It
learns only from already-sealed, *chosen* complete actions.  State heads see a
complete first-stage legal menu; Q heads additionally see the sealed selected
factorized action and its bounded structural representation.  The separation is
deliberate: selected-action fields have no computational path to ``V_plan``.

The sidecar is FP32-only on CPU and MPS.  It has its own strict checkpoint
format and may never carry policy weights, a policy optimizer, or serving
state.
"""

from __future__ import annotations

import copy
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

try:  # Keep schema inspection importable on non-training hosts.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover - trainer hosts provide torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


PRIZE_PLAN_V2_SIDECAR_SCHEMA: Final = "poke_bot.alakazam_prize_plan_v2_sidecar/v1"
PRIZE_PLAN_V2_SIDECAR_CHECKPOINT_SCHEMA: Final = (
    "poke_bot.alakazam_prize_plan_v2_sidecar_checkpoint/v1"
)
PRIZE_PLAN_V2_HORIZONS: Final[tuple[int, ...]] = (1, 3, 6, 12)
# This order is part of the independently checkpointed ABI.  Do not infer it
# from a mapping construction order in a trainer.
PRIZE_PLAN_V2_OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "V_plan_1",
    "Q_plan_1",
    "V_plan_3",
    "Q_plan_3",
    "V_plan_6",
    "Q_plan_6",
    "V_plan_12",
    "Q_plan_12",
)
PRIZE_PLAN_V2_OUTPUT_COUNT: Final = len(PRIZE_PLAN_V2_OUTPUT_NAMES)
_FORBIDDEN_POLICY_STATE_KEYS: Final = frozenset(
    {
        "model_state_dict",
        "policy_state_dict",
        "policy_model_state_dict",
        "policy_optimizer_state_dict",
        "base_model_state_dict",
        "parent_model_state_dict",
        "runtime_state_dict",
    }
)


class PrizePlanV2SidecarError(ValueError):
    """A Prize-plan sidecar input, target, or checkpoint violates its ABI."""


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("Prize-plan-v2 sidecar requires torch on a training host")


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrizePlanV2SidecarError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrizePlanV2SidecarError(f"{field} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class PrizePlanV2SidecarConfig:
    """Architecture-only configuration for the separate public sidecar."""

    feature_dim: int = 40
    state_hidden_dim: int = 128
    action_hidden_dim: int = 128
    q_hidden_dim: int = 128
    # Fixed bounds are the sealed complete-action structural ABI, not engine
    # action-space limits.  The trainer rejects larger source rows.
    max_action_stages: int = 32
    max_legal_options: int = 64
    max_action_program_tokens: int = 32
    max_action_token_value: int = 63

    def __post_init__(self) -> None:
        for field in (
            "feature_dim",
            "state_hidden_dim",
            "action_hidden_dim",
            "q_hidden_dim",
            "max_action_stages",
            "max_legal_options",
            "max_action_program_tokens",
        ):
            _positive_int(getattr(self, field), field=field)
        _nonnegative_int(self.max_action_token_value, field="max_action_token_value")


@dataclass(frozen=True)
class PrizePlanV2Predictions:
    """Named tensor view over the fixed eight-output public plan ABI."""

    values: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            name: self.values[:, index]
            for index, name in enumerate(PRIZE_PLAN_V2_OUTPUT_NAMES)
        }

    @property
    def v_plan_1(self) -> Tensor:
        return self.values[:, 0]

    @property
    def q_plan_1(self) -> Tensor:
        return self.values[:, 1]

    @property
    def v_plan_3(self) -> Tensor:
        return self.values[:, 2]

    @property
    def q_plan_3(self) -> Tensor:
        return self.values[:, 3]

    @property
    def v_plan_6(self) -> Tensor:
        return self.values[:, 4]

    @property
    def q_plan_6(self) -> Tensor:
        return self.values[:, 5]

    @property
    def v_plan_12(self) -> Tensor:
        return self.values[:, 6]

    @property
    def q_plan_12(self) -> Tensor:
        return self.values[:, 7]


@dataclass(frozen=True)
class PrizePlanV2Loss:
    """Masked loss and support counts for the four public plan horizons."""

    total: Tensor
    by_output: Mapping[str, Tensor]
    valid_by_horizon: Mapping[int, int]

    def metrics(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "total": float(self.total.detach().cpu().item()),
        }
        result.update(
            {
                name: float(value.detach().cpu().item())
                for name, value in self.by_output.items()
            }
        )
        result.update(
            {f"h{horizon}_valid": count for horizon, count in self.valid_by_horizon.items()}
        )
        return result


@dataclass(frozen=True)
class LoadedPrizePlanV2Checkpoint:
    """Strictly validated standalone sidecar checkpoint payload."""

    model: "PrizePlanV2Sidecar"
    optimizer_state_dict: Mapping[str, Any] | None
    training_state: Mapping[str, Any]
    metadata: Mapping[str, Any]


_ModuleBase = object if nn is None else nn.Module


class PrizePlanV2Sidecar(_ModuleBase):
    """Eight bounded public values for state and complete chosen actions.

    Args use the same sealed feature/action representation as the action critic:

    ``first_stage_legal_features`` / mask: complete legal menu, ``[B,L,40]``.
    ``selected_stage_features`` / mask: complete selected factorized action,
    ``[B,S,40]``.  The selected index, legal-count, program-token and token
    mask tensors are Q-only public structural fields.  All masks are strict
    right-padded prefixes.
    """

    def __init__(self, config: PrizePlanV2SidecarConfig) -> None:
        _require_torch()
        if not isinstance(config, PrizePlanV2SidecarConfig):
            raise PrizePlanV2SidecarError("config must be PrizePlanV2SidecarConfig")
        super().__init__()
        self.config = config
        self.first_stage_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden_dim),
        )
        self.state_trunk = nn.Sequential(
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden_dim),
        )
        # Index/count/ordinal/program length and a padded token/value+presence
        # representation make Q distinguish structurally different actions
        # even when their 40-wide selected feature vectors collide.
        self.action_structure_width = 4 + 2 * config.max_action_program_tokens
        self.selected_stage_encoder = nn.Sequential(
            nn.Linear(
                config.feature_dim + self.action_structure_width,
                config.action_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.action_hidden_dim),
        )
        self.selected_stage_sequence = nn.GRU(
            input_size=config.action_hidden_dim,
            hidden_size=config.action_hidden_dim,
            batch_first=True,
        )
        self.q_trunk = nn.Sequential(
            nn.Linear(config.state_hidden_dim + config.action_hidden_dim, config.q_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.q_hidden_dim),
        )
        self.v_heads = nn.ModuleList(
            [nn.Linear(config.state_hidden_dim, 1) for _ in PRIZE_PLAN_V2_HORIZONS]
        )
        self.q_heads = nn.ModuleList(
            [nn.Linear(config.q_hidden_dim, 1) for _ in PRIZE_PLAN_V2_HORIZONS]
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return PRIZE_PLAN_V2_OUTPUT_NAMES

    def _parameter_device(self) -> Any:
        return next(self.parameters()).device

    def _assert_fp32_parameters(self) -> None:
        wrong = [name for name, value in self.named_parameters() if value.dtype != torch.float32]
        if wrong:
            raise PrizePlanV2SidecarError(
                "Prize-plan-v2 sidecar is FP32-only; non-FP32 parameters: "
                + ", ".join(wrong[:3])
            )

    @staticmethod
    def _tensor(value: Any, *, field: str) -> Tensor:
        if not isinstance(value, torch.Tensor):
            raise PrizePlanV2SidecarError(f"{field} must be a torch.Tensor")
        return value

    @staticmethod
    def _finite(value: Tensor, *, field: str) -> None:
        if not bool(torch.isfinite(value).all().detach().cpu().item()):
            raise PrizePlanV2SidecarError(f"{field} must contain only finite values")

    @staticmethod
    def _right_padded(mask: Tensor, *, field: str, require_one: bool) -> None:
        if mask.ndim < 2:
            raise PrizePlanV2SidecarError(f"{field} must be at least two-dimensional")
        if mask.shape[-1] > 1 and bool(
            (mask[..., 1:] & ~mask[..., :-1]).any().detach().cpu().item()
        ):
            raise PrizePlanV2SidecarError(f"{field} must be a right-padded True* False* mask")
        if require_one and bool((mask.sum(dim=-1) <= 0).any().detach().cpu().item()):
            raise PrizePlanV2SidecarError(f"{field} requires at least one valid entry per row")

    def _validate_inputs(
        self,
        first_stage_legal_features: Any,
        first_stage_legal_mask: Any,
        selected_stage_features: Any,
        selected_stage_mask: Any,
        selected_option_indices: Any,
        selected_legal_counts: Any,
        selected_action_program_tokens: Any,
        selected_action_program_mask: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._assert_fp32_parameters()
        state = self._tensor(first_stage_legal_features, field="first_stage_legal_features")
        state_mask = self._tensor(first_stage_legal_mask, field="first_stage_legal_mask")
        action = self._tensor(selected_stage_features, field="selected_stage_features")
        action_mask = self._tensor(selected_stage_mask, field="selected_stage_mask")
        indices = self._tensor(selected_option_indices, field="selected_option_indices")
        counts = self._tensor(selected_legal_counts, field="selected_legal_counts")
        tokens = self._tensor(
            selected_action_program_tokens, field="selected_action_program_tokens"
        )
        token_mask = self._tensor(
            selected_action_program_mask, field="selected_action_program_mask"
        )
        for field, tensor in (
            ("first_stage_legal_features", state),
            ("selected_stage_features", action),
        ):
            if tensor.dtype != torch.float32 or tensor.ndim != 3:
                raise PrizePlanV2SidecarError(f"{field} must be FP32 [batch,slots,feature]")
            if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] != self.config.feature_dim:
                raise PrizePlanV2SidecarError(f"{field} shape does not match the fixed feature ABI")
            self._finite(tensor, field=field)
        if state.shape[0] != action.shape[0] or action.shape[1] > self.config.max_action_stages:
            raise PrizePlanV2SidecarError("state/action batch or action-stage bound drifted")
        for field, mask, features in (
            ("first_stage_legal_mask", state_mask, state),
            ("selected_stage_mask", action_mask, action),
        ):
            if mask.dtype != torch.bool or mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
                raise PrizePlanV2SidecarError(f"{field} must be a bool mask aligned to its feature tensor")
            self._right_padded(mask, field=field, require_one=True)
        for field, tensor in (("selected_option_indices", indices), ("selected_legal_counts", counts)):
            if tensor.dtype != torch.int64 or tensor.ndim != 2 or tuple(tensor.shape) != tuple(action.shape[:2]):
                raise PrizePlanV2SidecarError(f"{field} must be int64 [batch,selected_stages]")
        if (
            tokens.dtype != torch.int64
            or tokens.ndim != 3
            or tuple(tokens.shape[:2]) != tuple(action.shape[:2])
            or not 1 <= tokens.shape[2] <= self.config.max_action_program_tokens
        ):
            raise PrizePlanV2SidecarError("selected_action_program_tokens shape or token bound drifted")
        if token_mask.dtype != torch.bool or tuple(token_mask.shape) != tuple(tokens.shape):
            raise PrizePlanV2SidecarError("selected_action_program_mask must align to action-program tokens")
        self._right_padded(token_mask, field="selected_action_program_mask", require_one=False)
        if bool((token_mask & ~action_mask.unsqueeze(-1)).any().detach().cpu().item()):
            raise PrizePlanV2SidecarError("token mask cannot mark a padded selected stage")
        valid = action_mask
        padded = ~valid
        if bool((valid & ((counts < 1) | (counts > self.config.max_legal_options))).any().detach().cpu().item()):
            raise PrizePlanV2SidecarError("selected legal count exceeds fixed public ABI")
        if bool((valid & ((indices < 0) | (indices >= counts))).any().detach().cpu().item()):
            raise PrizePlanV2SidecarError("selected option index is not within its sealed legal menu")
        if bool((padded & ((indices != 0) | (counts != 0))).any().detach().cpu().item()):
            raise PrizePlanV2SidecarError("padded selected stages must be structural zero padding")
        if bool(((~token_mask) & (tokens != 0)).any().detach().cpu().item()):
            raise PrizePlanV2SidecarError("masked action-program tokens must be zero padded")
        if bool(
            (token_mask & ((tokens < 0) | (tokens > self.config.max_action_token_value)))
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise PrizePlanV2SidecarError("action-program tokens exceed fixed public bounds")
        expected_device = self._parameter_device()
        for field, tensor in (
            ("first_stage_legal_features", state),
            ("first_stage_legal_mask", state_mask),
            ("selected_stage_features", action),
            ("selected_stage_mask", action_mask),
            ("selected_option_indices", indices),
            ("selected_legal_counts", counts),
            ("selected_action_program_tokens", tokens),
            ("selected_action_program_mask", token_mask),
        ):
            if tensor.device != expected_device:
                raise PrizePlanV2SidecarError(f"{field} device does not match the sidecar device")
        return state, state_mask, action, action_mask, indices, counts, tokens, token_mask

    def _action_structure(
        self,
        indices: Tensor,
        counts: Tensor,
        tokens: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        batch, stages = indices.shape
        ordinal = torch.arange(stages, dtype=torch.float32, device=indices.device).view(1, stages, 1)
        ordinal = ordinal.expand(batch, -1, -1) / float(max(self.config.max_action_stages - 1, 1))
        index = indices.to(dtype=torch.float32).unsqueeze(-1) / float(max(self.config.max_legal_options - 1, 1))
        count = counts.to(dtype=torch.float32).unsqueeze(-1) / float(self.config.max_legal_options)
        length = token_mask.sum(dim=2, dtype=torch.float32).unsqueeze(-1)
        length = length / float(self.config.max_action_program_tokens)
        values = tokens.to(dtype=torch.float32) / float(max(self.config.max_action_token_value, 1))
        present = token_mask.to(dtype=torch.float32)
        missing = self.config.max_action_program_tokens - tokens.shape[2]
        if missing:
            values = F.pad(values, (0, missing))
            present = F.pad(present, (0, missing))
        structure = torch.cat((index, count, ordinal, length, values, present), dim=-1)
        if structure.shape[-1] != self.action_structure_width:
            raise PrizePlanV2SidecarError("internal selected-action structure width drifted")
        return structure

    def forward(
        self,
        first_stage_legal_features: Tensor,
        first_stage_legal_mask: Tensor,
        selected_stage_features: Tensor,
        selected_stage_mask: Tensor,
        selected_option_indices: Tensor,
        selected_legal_counts: Tensor,
        selected_action_program_tokens: Tensor,
        selected_action_program_mask: Tensor,
    ) -> Tensor:
        """Emit `[B,8]` bounded values in ``PRIZE_PLAN_V2_OUTPUT_NAMES`` order."""

        state, state_mask, action, action_mask, indices, counts, tokens, token_mask = self._validate_inputs(
            first_stage_legal_features,
            first_stage_legal_mask,
            selected_stage_features,
            selected_stage_mask,
            selected_option_indices,
            selected_legal_counts,
            selected_action_program_tokens,
            selected_action_program_mask,
        )
        # V path is intentionally complete-menu only.
        state_options = self.first_stage_encoder(state)
        weights = state_mask.unsqueeze(-1).to(dtype=torch.float32)
        state_context = self.state_trunk((state_options * weights).sum(dim=1) / weights.sum(dim=1))
        # Q path is the selected complete structured action plus the public
        # state summary.  No Q-only argument is used above.
        structure = self._action_structure(indices, counts, tokens, token_mask)
        selected = self.selected_stage_encoder(torch.cat((action, structure), dim=-1))
        selected = selected * action_mask.unsqueeze(-1).to(dtype=torch.float32)
        sequence, _unused = self.selected_stage_sequence(selected)
        final_index = action_mask.sum(dim=1, dtype=torch.long) - 1
        action_context = sequence.gather(
            1,
            final_index.view(-1, 1, 1).expand(-1, 1, sequence.shape[-1]),
        ).squeeze(1)
        q_context = self.q_trunk(torch.cat((state_context, action_context), dim=-1))
        outputs: list[Tensor] = []
        for v_head, q_head in zip(self.v_heads, self.q_heads, strict=True):
            outputs.append(torch.tanh(v_head(state_context)))
            outputs.append(torch.tanh(q_head(q_context)))
        values = torch.cat(outputs, dim=-1)
        return validate_prize_plan_v2_predictions(values)

    def predict(
        self,
        first_stage_legal_features: Tensor,
        first_stage_legal_mask: Tensor,
        selected_stage_features: Tensor,
        selected_stage_mask: Tensor,
        selected_option_indices: Tensor,
        selected_legal_counts: Tensor,
        selected_action_program_tokens: Tensor,
        selected_action_program_mask: Tensor,
    ) -> PrizePlanV2Predictions:
        return PrizePlanV2Predictions(
            self(
                first_stage_legal_features,
                first_stage_legal_mask,
                selected_stage_features,
                selected_stage_mask,
                selected_option_indices,
                selected_legal_counts,
                selected_action_program_tokens,
                selected_action_program_mask,
            )
        )


def validate_prize_plan_v2_predictions(values: Any) -> Tensor:
    _require_torch()
    if not isinstance(values, torch.Tensor):
        raise PrizePlanV2SidecarError("predictions must be a torch.Tensor")
    if values.dtype != torch.float32 or values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != PRIZE_PLAN_V2_OUTPUT_COUNT:
        raise PrizePlanV2SidecarError("predictions must be FP32 [batch,8]")
    if not bool(torch.isfinite(values).all().detach().cpu().item()):
        raise PrizePlanV2SidecarError("predictions must be finite")
    if bool(((values < -1.0) | (values > 1.0)).any().detach().cpu().item()):
        raise PrizePlanV2SidecarError("Prize-plan predictions must be tanh-bounded to [-1,+1]")
    return values


def split_prize_plan_v2_predictions(values: Tensor) -> PrizePlanV2Predictions:
    return PrizePlanV2Predictions(validate_prize_plan_v2_predictions(values))


def masked_prize_plan_v2_loss(
    predictions: Tensor,
    targets: Tensor,
    masks: Tensor,
    *,
    smooth_l1_beta: float = 1.0,
) -> PrizePlanV2Loss:
    """Compute chosen-action-only masked losses for all state/Q plan heads.

    ``targets`` and ``masks`` are `[B,4]` in H1/H3/H6/H12 order.  Each
    labelled recorded action supervises both the state value observed from that
    state and the value of its chosen complete action.  A masked horizon is
    neither fabricated as zero nor included in a denominator.
    """

    _require_torch()
    predictions = validate_prize_plan_v2_predictions(predictions)
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.float32:
        raise PrizePlanV2SidecarError("targets must be FP32 [batch,4]")
    if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool:
        raise PrizePlanV2SidecarError("masks must be bool [batch,4]")
    if targets.ndim != 2 or masks.ndim != 2 or tuple(targets.shape) != tuple(masks.shape) or targets.shape != (predictions.shape[0], len(PRIZE_PLAN_V2_HORIZONS)):
        raise PrizePlanV2SidecarError("targets/masks must align to batch and four plan horizons")
    if targets.device != predictions.device or masks.device != predictions.device:
        raise PrizePlanV2SidecarError("targets/masks must use the prediction device")
    if not math.isfinite(float(smooth_l1_beta)) or float(smooth_l1_beta) <= 0.0:
        raise PrizePlanV2SidecarError("smooth_l1_beta must be positive and finite")
    if bool((masks & ~torch.isfinite(targets)).any().detach().cpu().item()):
        raise PrizePlanV2SidecarError("unmasked plan targets must be finite")
    if bool((masks & ((targets < -1.0) | (targets > 1.0))).any().detach().cpu().item()):
        raise PrizePlanV2SidecarError("scaled plan targets must be bounded to [-1,+1]")
    components: dict[str, Tensor] = {}
    valid: dict[int, int] = {}
    per_output: list[Tensor] = []
    for horizon_index, horizon in enumerate(PRIZE_PLAN_V2_HORIZONS):
        mask = masks[:, horizon_index]
        count = int(mask.sum().detach().cpu().item())
        valid[horizon] = count
        target = targets[:, horizon_index]
        for offset, prefix in ((0, "V_plan"), (1, "Q_plan")):
            name = f"{prefix}_{horizon}"
            prediction = predictions[:, 2 * horizon_index + offset]
            if count:
                component = F.smooth_l1_loss(
                    prediction[mask], target[mask], beta=float(smooth_l1_beta), reduction="mean"
                )
            else:
                # Connected exact zero preserves finite gradient accounting.
                component = prediction.sum() * 0.0
            components[name] = component
            per_output.append(component)
    total = torch.stack(per_output).mean()
    return PrizePlanV2Loss(total=total, by_output=components, valid_by_horizon=valid)


def _validate_json_metadata(value: Any, *, field: str) -> Any:
    """Deep-copy receipt metadata while excluding policy/runtime state.

    A ``json.dumps`` round trip alone is not enough here: it can validate
    serializability but cannot make the independently checkpointed sidecar
    boundary explicit.  Every nested key is examined before it is copied, so
    a policy-state dictionary cannot be hidden under an innocuous receipt key.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise PrizePlanV2SidecarError(f"{field} cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise PrizePlanV2SidecarError(f"{field} mapping keys must be strings")
            if key.casefold() in _FORBIDDEN_POLICY_STATE_KEYS:
                raise PrizePlanV2SidecarError(
                    f"{field} may not contain policy state key {key!r}"
                )
            copied[key] = _validate_json_metadata(child, field=f"{field}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_validate_json_metadata(child, field=f"{field}[]") for child in value]
    raise PrizePlanV2SidecarError(
        f"{field} must contain only JSON-like receipt metadata, never tensors or policy state"
    )


def _validate_checkpoint_tensors(value: Any, *, field: str) -> None:
    """Reject nested policy keys and non-finite tensors in checkpoint state.

    Optimizer state dictionaries legitimately use integer parameter IDs, so
    unlike receipt metadata this validator permits integer mapping keys.  It
    still rejects policy/runtime names at every nesting level and verifies all
    tensors before an optimizer is ever restored.
    """

    _require_torch()
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().detach().cpu().item()):
            raise PrizePlanV2SidecarError(f"{field} contains a non-finite tensor")
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise PrizePlanV2SidecarError(f"{field} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_POLICY_STATE_KEYS:
                raise PrizePlanV2SidecarError(
                    f"{field} contains forbidden policy state key {key!r}"
                )
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise PrizePlanV2SidecarError(f"{field} has an unsupported mapping key")
            _validate_checkpoint_tensors(child, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_checkpoint_tensors(child, field=f"{field}[{index}]")
        return
    raise PrizePlanV2SidecarError(f"{field} contains an unsupported checkpoint value")


def _validate_sidecar_optimizer(model: PrizePlanV2Sidecar, optimizer: Any) -> None:
    """Require every optimizer parameter to belong to this sidecar only."""

    _require_torch()
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise PrizePlanV2SidecarError("optimizer must be a torch.optim.Optimizer")
    sidecar_parameter_ids = {id(parameter) for parameter in model.parameters()}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group.get("params", ()):  # pragma: no branch - optimizer ABI.
            if id(parameter) not in sidecar_parameter_ids:
                raise PrizePlanV2SidecarError(
                    "optimizer contains a non-sidecar parameter in group "
                    f"{group_index}"
                )


def _validate_optimizer_payload(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PrizePlanV2SidecarError("optimizer_state_dict must be an object or null")
    # A standard PyTorch state dict is deliberately opaque to the sidecar, but
    # finite tensors and the no-policy-state boundary are not optional.
    _validate_checkpoint_tensors(value, field="optimizer_state_dict")
    return copy.deepcopy(dict(value))


def build_prize_plan_v2_checkpoint(
    model: PrizePlanV2Sidecar,
    *,
    optimizer: Any | None = None,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a strict sidecar-only checkpoint payload on CPU tensors."""

    _require_torch()
    if not isinstance(model, PrizePlanV2Sidecar):
        raise PrizePlanV2SidecarError("model must be PrizePlanV2Sidecar")
    model._assert_fp32_parameters()
    if optimizer is not None:
        _validate_sidecar_optimizer(model, optimizer)
    clean_training_state = _validate_json_metadata(
        {} if training_state is None else training_state, field="training_state"
    )
    clean_metadata = _validate_json_metadata(
        {} if metadata is None else metadata, field="metadata"
    )
    sidecar_state = {
        key: value.detach().to(device="cpu", dtype=torch.float32).clone()
        for key, value in model.state_dict().items()
    }
    _validate_checkpoint_tensors(sidecar_state, field="sidecar_state_dict")
    optimizer_state = (
        None if optimizer is None else _validate_optimizer_payload(optimizer.state_dict())
    )
    return {
        "schema": PRIZE_PLAN_V2_SIDECAR_CHECKPOINT_SCHEMA,
        "sidecar_schema": PRIZE_PLAN_V2_SIDECAR_SCHEMA,
        "config": asdict(model.config),
        "sidecar_state_dict": sidecar_state,
        "optimizer_state_dict": optimizer_state,
        "training_state": clean_training_state,
        "metadata": clean_metadata,
    }


def _validate_checkpoint_payload(
    payload: Any,
) -> tuple[PrizePlanV2SidecarConfig, Mapping[str, Tensor], Mapping[str, Any] | None, Mapping[str, Any], Mapping[str, Any]]:
    _require_torch()
    if not isinstance(payload, Mapping):
        raise PrizePlanV2SidecarError("checkpoint payload must be an object")
    _validate_checkpoint_tensors(payload, field="checkpoint")
    expected = {
        "schema",
        "sidecar_schema",
        "config",
        "sidecar_state_dict",
        "optimizer_state_dict",
        "training_state",
        "metadata",
    }
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise PrizePlanV2SidecarError(
            "checkpoint field inventory drifted: "
            + (f"missing={sorted(missing)} " if missing else "")
            + (f"unknown={sorted(unknown)}" if unknown else "")
        )
    if payload.get("schema") != PRIZE_PLAN_V2_SIDECAR_CHECKPOINT_SCHEMA or payload.get("sidecar_schema") != PRIZE_PLAN_V2_SIDECAR_SCHEMA:
        raise PrizePlanV2SidecarError("checkpoint schema is not Prize-plan-v2")
    config_raw = payload.get("config")
    if not isinstance(config_raw, Mapping) or set(config_raw) != set(PrizePlanV2SidecarConfig.__dataclass_fields__):
        raise PrizePlanV2SidecarError("checkpoint config field inventory drifted")
    try:
        config = PrizePlanV2SidecarConfig(**dict(config_raw))
    except (TypeError, ValueError) as exc:
        raise PrizePlanV2SidecarError("checkpoint config is invalid") from exc
    state = payload.get("sidecar_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise PrizePlanV2SidecarError("checkpoint sidecar_state_dict is invalid")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise PrizePlanV2SidecarError("checkpoint sidecar state must contain named FP32 tensors")
        if not bool(torch.isfinite(value).all().detach().cpu().item()):
            raise PrizePlanV2SidecarError("checkpoint sidecar state contains non-finite tensors")
    optimizer = _validate_optimizer_payload(payload.get("optimizer_state_dict"))
    training = _validate_json_metadata(payload.get("training_state"), field="training_state")
    metadata = _validate_json_metadata(payload.get("metadata"), field="metadata")
    if not isinstance(training, Mapping) or not isinstance(metadata, Mapping):
        raise PrizePlanV2SidecarError("training_state and metadata must be objects")
    return config, state, optimizer, training, metadata


def load_prize_plan_v2_checkpoint(
    source: Path | str | Mapping[str, Any], *, device: Any = "cpu"
) -> LoadedPrizePlanV2Checkpoint:
    """Load only a strict Prize-plan-v2 sidecar checkpoint."""

    _require_torch()
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise PrizePlanV2SidecarError("checkpoint path must be a regular non-symlink file")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Older torch has no weights_only argument.
            payload = torch.load(path, map_location="cpu")
    else:
        payload = source
    config, state, optimizer, training, metadata = _validate_checkpoint_payload(payload)
    model = PrizePlanV2Sidecar(config).to(device=device, dtype=torch.float32)
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise PrizePlanV2SidecarError("checkpoint state does not match the Prize-plan-v2 ABI") from exc
    model._assert_fp32_parameters()
    return LoadedPrizePlanV2Checkpoint(
        model=model,
        optimizer_state_dict=copy.deepcopy(optimizer),
        training_state=copy.deepcopy(dict(training)),
        metadata=copy.deepcopy(dict(metadata)),
    )


def save_prize_plan_v2_checkpoint(
    path: Path | str,
    model: PrizePlanV2Sidecar,
    *,
    optimizer: Any | None = None,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically replace one explicit standalone sidecar checkpoint path."""

    _require_torch()
    destination = Path(path).expanduser().resolve()
    if destination.name in {"", ".", ".."} or destination.is_dir() or destination.is_symlink():
        raise PrizePlanV2SidecarError("checkpoint path must name a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_prize_plan_v2_checkpoint(
        model, optimizer=optimizer, training_state=training_state, metadata=metadata
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def restore_prize_plan_v2_checkpoint(
    model: PrizePlanV2Sidecar,
    source: Path | str | Mapping[str, Any],
    *,
    optimizer: Any | None = None,
) -> LoadedPrizePlanV2Checkpoint:
    """Restore a compatible standalone sidecar and optional AdamW state."""

    _require_torch()
    if not isinstance(model, PrizePlanV2Sidecar):
        raise PrizePlanV2SidecarError("model must be PrizePlanV2Sidecar")
    loaded = load_prize_plan_v2_checkpoint(source, device=model._parameter_device())
    if loaded.model.config != model.config:
        raise PrizePlanV2SidecarError("checkpoint config does not match destination sidecar")
    if optimizer is not None:
        _validate_sidecar_optimizer(model, optimizer)
        if loaded.optimizer_state_dict is None:
            raise PrizePlanV2SidecarError("checkpoint has no optimizer state to restore")
    model.load_state_dict(loaded.model.state_dict(), strict=True)
    model._assert_fp32_parameters()
    if optimizer is not None:
        optimizer.load_state_dict(dict(loaded.optimizer_state_dict))
    return LoadedPrizePlanV2Checkpoint(
        model=model,
        optimizer_state_dict=loaded.optimizer_state_dict,
        training_state=loaded.training_state,
        metadata=loaded.metadata,
    )
