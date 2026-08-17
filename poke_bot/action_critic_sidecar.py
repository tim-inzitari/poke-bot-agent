"""Trainer-only action-conditioned critic for recorded complete actions.

This module intentionally has no dependency on the policy model.  In
particular, it does not import the agent, the play loop, a runtime selector, or
any search component.  The sidecar consumes two public, already-materialized
views of a decision:

* the *entire* first-stage legal-option menu for state values; and
* the ordered sequence of selected options across the complete factorized
  action, including sealed selected index/count/program structure, for action
  values.

The two views are kept deliberately separate: changing selected-stage features
cannot change a ``V`` output.  ``Q`` outputs are conditioned on both the state
summary and the selected-stage sequence.  The model is FP32-only so a sealed
checkpoint has the same numerical contract on CPU and MPS training hosts.
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
except ModuleNotFoundError:  # pragma: no cover - a trainer always supplies torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


ACTION_CRITIC_SIDECAR_SCHEMA: Final = "poke_bot.action_critic_sidecar/v1"
ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA: Final = (
    "poke_bot.action_critic_sidecar_checkpoint/v1"
)

# This order is an ABI.  Keep it explicit rather than relying on a dict's
# construction order at a call site that could accidentally drift.
ACTION_CRITIC_OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "V_win",
    "Q_win",
    "V_prize_1",
    "Q_prize_1",
    "V_prize_2",
    "Q_prize_2",
    "V_prize_3",
    "Q_prize_3",
)
ACTION_CRITIC_OUTPUT_COUNT: Final = len(ACTION_CRITIC_OUTPUT_NAMES)

_V_OUTPUT_INDICES: Final = (0, 2, 4, 6)
_Q_OUTPUT_INDICES: Final = (1, 3, 5, 7)
_PRIZE_OUTPUT_INDICES: Final = ((2, 3), (4, 5), (6, 7))
_FORBIDDEN_POLICY_STATE_KEYS: Final = frozenset(
    {
        "model_state_dict",
        "policy_state_dict",
        "policy_model_state_dict",
        "policy_optimizer_state_dict",
        "base_model_state_dict",
        "parent_model_state_dict",
    }
)


class ActionCriticSidecarError(ValueError):
    """A critic input, target, or checkpoint violates the sidecar contract."""


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("action critic sidecar requires torch on a training host")


def _as_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ActionCriticSidecarError(f"{field} must be bool")
    return value


def _as_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActionCriticSidecarError(f"{field} must be a positive integer")
    return int(value)


def _as_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActionCriticSidecarError(f"{field} must be a nonnegative integer")
    return int(value)


def _as_nonnegative_finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionCriticSidecarError(f"{field} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ActionCriticSidecarError(f"{field} must be a finite nonnegative number")
    return result


@dataclass(frozen=True)
class ActionCriticSidecarConfig:
    """Architecture-only configuration for the separately checkpointed critic."""

    feature_dim: int = 40
    state_hidden_dim: int = 128
    action_hidden_dim: int = 128
    q_hidden_dim: int = 128
    # The sealed recent-20 overlay audit observed maxima 28 / 55 / 52 for
    # factorized stages / legal options / raw action tokens respectively.
    # Round each public input bound up once in the standalone checkpoint ABI;
    # no dynamic expansion or runtime action enumeration is allowed.
    max_action_stages: int = 32
    # These caps bound a public, current-decision action representation.  They
    # do not define an engine action space or permit runtime enumeration.
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
            _as_positive_int(getattr(self, field), field=field)
        _as_nonnegative_int(
            self.max_action_token_value, field="max_action_token_value"
        )


@dataclass(frozen=True)
class ActionCriticPredictions:
    """Named, tensor-backed view of the eight-output ABI.

    ``V_win`` and ``Q_win`` remain logits for BCE training.  ``V_prize_h`` and
    ``Q_prize_h`` are already ``tanh``-bounded to ``[-1, +1]``.
    """

    values: Tensor

    @property
    def v_win_logits(self) -> Tensor:
        return self.values[:, 0]

    @property
    def q_win_logits(self) -> Tensor:
        return self.values[:, 1]

    @property
    def v_prize_1(self) -> Tensor:
        return self.values[:, 2]

    @property
    def q_prize_1(self) -> Tensor:
        return self.values[:, 3]

    @property
    def v_prize_2(self) -> Tensor:
        return self.values[:, 4]

    @property
    def q_prize_2(self) -> Tensor:
        return self.values[:, 5]

    @property
    def v_prize_3(self) -> Tensor:
        return self.values[:, 6]

    @property
    def q_prize_3(self) -> Tensor:
        return self.values[:, 7]

    @property
    def v_win_probability(self) -> Tensor:
        return torch.sigmoid(self.v_win_logits)

    @property
    def q_win_probability(self) -> Tensor:
        return torch.sigmoid(self.q_win_logits)

    def as_dict(self) -> dict[str, Tensor]:
        return {
            name: self.values[:, index]
            for index, name in enumerate(ACTION_CRITIC_OUTPUT_NAMES)
        }


@dataclass(frozen=True)
class ActionCriticLoss:
    """Loss components and per-horizon target coverage for one batch."""

    total: Tensor
    win: Tensor
    prize: Tensor
    v_win: Tensor
    q_win: Tensor
    prize_by_horizon: tuple[Tensor, Tensor, Tensor]
    valid_prize_count_by_horizon: tuple[int, int, int]

    def metrics(self) -> dict[str, float | int]:
        """Return detached scalar metrics suitable for a receipt or log."""

        return {
            "total": float(self.total.detach().cpu().item()),
            "win": float(self.win.detach().cpu().item()),
            "prize": float(self.prize.detach().cpu().item()),
            "v_win": float(self.v_win.detach().cpu().item()),
            "q_win": float(self.q_win.detach().cpu().item()),
            "prize_1_valid": self.valid_prize_count_by_horizon[0],
            "prize_2_valid": self.valid_prize_count_by_horizon[1],
            "prize_3_valid": self.valid_prize_count_by_horizon[2],
        }


@dataclass(frozen=True)
class LoadedActionCriticCheckpoint:
    """A validated standalone sidecar checkpoint, never a policy checkpoint."""

    model: "ActionCriticSidecar"
    optimizer_state_dict: Mapping[str, Any] | None
    training_state: Mapping[str, Any]
    metadata: Mapping[str, Any]


_ModuleBase = object if nn is None else nn.Module


class ActionCriticSidecar(_ModuleBase):
    """Eight-output critic over a legal menu and one complete chosen action.

    ``first_stage_legal_features`` is ``[batch, legal_option_slots, feature]``
    and must contain the whole legal first-stage menu.  Its boolean mask may
    right-pad a variable number of legal options, but may not have holes.

    ``selected_stage_features`` is ``[batch, selected_stage_slots, feature]``
    and contains each chosen factorized stage in order.  The associated sealed
    ``selected_option_indices``, ``selected_legal_counts``, and bounded
    ``selected_action_program_tokens`` describe which legal program was
    selected.  They are all current pre-action public fields from the complete
    action overlay; no successor state or target-only value is accepted.

    The final valid GRU output is the complete-action representation.  State
    heads have no path from any selected-action argument; Q heads concatenate
    the complete chosen-action representation with the state representation.
    """

    def __init__(self, config: ActionCriticSidecarConfig) -> None:
        _require_torch()
        if not isinstance(config, ActionCriticSidecarConfig):
            raise ActionCriticSidecarError("config must be ActionCriticSidecarConfig")
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
        # Per-stage Q input contains the selected 40-D vector plus a bounded,
        # injective-for-the-declared-domain structural view: selected index,
        # legal count, stage ordinal, action-program length, raw action tokens,
        # and explicit token-presence bits.  Presence bits distinguish a real
        # zero token from right padding.
        self.action_structure_width = 4 + 2 * config.max_action_program_tokens
        self.selected_stage_encoder = nn.Sequential(
            nn.Linear(
                config.feature_dim + self.action_structure_width,
                config.action_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.action_hidden_dim),
        )
        # The selected stages are a factorized action, not independent
        # transitions.  Use their ordered, complete selected prefix only.
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

        self.v_win_head = nn.Linear(config.state_hidden_dim, 1)
        self.q_win_head = nn.Linear(config.q_hidden_dim, 1)
        self.v_prize_heads = nn.ModuleList(
            [nn.Linear(config.state_hidden_dim, 1) for _ in range(3)]
        )
        self.q_prize_heads = nn.ModuleList(
            [nn.Linear(config.q_hidden_dim, 1) for _ in range(3)]
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return ACTION_CRITIC_OUTPUT_NAMES

    def _parameter_device(self) -> Any:
        return next(self.parameters()).device

    def _assert_fp32_parameters(self) -> None:
        wrong = [name for name, parameter in self.named_parameters() if parameter.dtype != torch.float32]
        if wrong:
            raise ActionCriticSidecarError(
                "action critic sidecar is FP32-only; non-FP32 parameters: "
                + ", ".join(wrong[:3])
            )

    @staticmethod
    def _require_tensor(value: Any, *, field: str) -> Tensor:
        if not isinstance(value, torch.Tensor):
            raise ActionCriticSidecarError(f"{field} must be a torch.Tensor")
        return value

    @staticmethod
    def _require_finite(value: Tensor, *, field: str) -> None:
        if not bool(torch.isfinite(value).all().detach().cpu().item()):
            raise ActionCriticSidecarError(f"{field} must contain only finite values")

    @staticmethod
    def _require_right_padded_mask(mask: Tensor, *, field: str) -> None:
        # A valid row is True* False*.  Holes would make a sequence ambiguous.
        if mask.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any().detach().cpu().item()):
            raise ActionCriticSidecarError(f"{field} must be a right-padded True* False* mask")
        if bool((mask.sum(dim=1) <= 0).any().detach().cpu().item()):
            raise ActionCriticSidecarError(f"{field} requires at least one valid entry per row")

    @staticmethod
    def _require_right_padded_program_mask(mask: Tensor, *, field: str) -> None:
        """Validate ``[batch, stage, token]`` True*False* masks.

        A selected stage may represent an empty legal program, so unlike the
        stage/menu masks this deliberately permits all-false rows.
        """

        if mask.shape[2] > 1 and bool(
            (mask[:, :, 1:] & ~mask[:, :, :-1]).any().detach().cpu().item()
        ):
            raise ActionCriticSidecarError(
                f"{field} must be a right-padded True* False* mask"
            )

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
        state_features = self._require_tensor(
            first_stage_legal_features, field="first_stage_legal_features"
        )
        state_mask = self._require_tensor(
            first_stage_legal_mask, field="first_stage_legal_mask"
        )
        action_features = self._require_tensor(
            selected_stage_features, field="selected_stage_features"
        )
        action_mask = self._require_tensor(selected_stage_mask, field="selected_stage_mask")
        option_indices = self._require_tensor(
            selected_option_indices, field="selected_option_indices"
        )
        legal_counts = self._require_tensor(
            selected_legal_counts, field="selected_legal_counts"
        )
        action_program_tokens = self._require_tensor(
            selected_action_program_tokens, field="selected_action_program_tokens"
        )
        action_program_mask = self._require_tensor(
            selected_action_program_mask, field="selected_action_program_mask"
        )
        for field, tensor in (
            ("first_stage_legal_features", state_features),
            ("selected_stage_features", action_features),
        ):
            if tensor.dtype != torch.float32:
                raise ActionCriticSidecarError(f"{field} must be FP32")
            if tensor.ndim != 3:
                raise ActionCriticSidecarError(f"{field} must have shape [batch, slots, feature]")
            if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
                raise ActionCriticSidecarError(f"{field} batch and slot dimensions must be positive")
            if tensor.shape[2] != self.config.feature_dim:
                raise ActionCriticSidecarError(
                    f"{field} feature dimension must equal config.feature_dim={self.config.feature_dim}"
                )
            self._require_finite(tensor, field=field)
        if action_features.shape[1] > self.config.max_action_stages:
            raise ActionCriticSidecarError(
                "selected_stage_features exceeds config.max_action_stages="
                f"{self.config.max_action_stages}"
            )
        if state_features.shape[0] != action_features.shape[0]:
            raise ActionCriticSidecarError("state and selected-action batches must align")
        for field, mask, features in (
            ("first_stage_legal_mask", state_mask, state_features),
            ("selected_stage_mask", action_mask, action_features),
        ):
            if mask.dtype != torch.bool:
                raise ActionCriticSidecarError(f"{field} must be bool")
            if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
                raise ActionCriticSidecarError(
                    f"{field} must have shape [batch, slots] aligned to its features"
                )
            self._require_right_padded_mask(mask, field=field)
        for field, tensor in (
            ("selected_option_indices", option_indices),
            ("selected_legal_counts", legal_counts),
        ):
            if tensor.dtype != torch.int64:
                raise ActionCriticSidecarError(f"{field} must be int64")
            if tensor.ndim != 2 or tuple(tensor.shape) != tuple(action_features.shape[:2]):
                raise ActionCriticSidecarError(
                    f"{field} must have shape [batch, selected_stage_slots]"
                )
        if action_program_tokens.dtype != torch.int64:
            raise ActionCriticSidecarError("selected_action_program_tokens must be int64")
        if action_program_tokens.ndim != 3 or tuple(action_program_tokens.shape[:2]) != tuple(
            action_features.shape[:2]
        ):
            raise ActionCriticSidecarError(
                "selected_action_program_tokens must have shape "
                "[batch, selected_stage_slots, token_slots]"
            )
        if (
            action_program_tokens.shape[2] <= 0
            or action_program_tokens.shape[2] > self.config.max_action_program_tokens
        ):
            raise ActionCriticSidecarError(
                "selected_action_program_tokens token slots must be in [1, "
                f"{self.config.max_action_program_tokens}]"
            )
        if action_program_mask.dtype != torch.bool or tuple(action_program_mask.shape) != tuple(
            action_program_tokens.shape
        ):
            raise ActionCriticSidecarError(
                "selected_action_program_mask must be bool and align to "
                "selected_action_program_tokens"
            )
        self._require_right_padded_program_mask(
            action_program_mask, field="selected_action_program_mask"
        )
        if bool((action_program_mask & ~action_mask.unsqueeze(-1)).any().detach().cpu().item()):
            raise ActionCriticSidecarError(
                "selected_action_program_mask cannot mark a padded selected stage"
            )
        valid_stages = action_mask
        padded_stages = ~valid_stages
        if bool(
            (
                valid_stages
                & (
                    (legal_counts < 1)
                    | (legal_counts > self.config.max_legal_options)
                )
            )
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise ActionCriticSidecarError(
                "selected_legal_counts for selected stages must be within the configured bound"
            )
        if bool(
            (
                valid_stages
                & ((option_indices < 0) | (option_indices >= legal_counts))
            )
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise ActionCriticSidecarError(
                "selected_option_indices must index the sealed selected-stage legal menu"
            )
        if bool(
            (padded_stages & ((option_indices != 0) | (legal_counts != 0)))
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise ActionCriticSidecarError(
                "padded selected stages must have zero option index and legal count"
            )
        if bool(
            ((~action_program_mask) & (action_program_tokens != 0))
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise ActionCriticSidecarError(
                "masked selected_action_program_tokens must be zero padded"
            )
        if bool(
            (
                action_program_mask
                & (
                    (action_program_tokens < 0)
                    | (action_program_tokens > self.config.max_action_token_value)
                )
            )
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise ActionCriticSidecarError(
                "selected_action_program_tokens exceed the configured public token bound"
            )
        expected_device = self._parameter_device()
        for field, tensor in (
            ("first_stage_legal_features", state_features),
            ("first_stage_legal_mask", state_mask),
            ("selected_stage_features", action_features),
            ("selected_stage_mask", action_mask),
            ("selected_option_indices", option_indices),
            ("selected_legal_counts", legal_counts),
            ("selected_action_program_tokens", action_program_tokens),
            ("selected_action_program_mask", action_program_mask),
        ):
            if tensor.device != expected_device:
                raise ActionCriticSidecarError(
                    f"{field} device {tensor.device} does not match sidecar device {expected_device}"
                )
        return (
            state_features,
            state_mask,
            action_features,
            action_mask,
            option_indices,
            legal_counts,
            action_program_tokens,
            action_program_mask,
        )

    def _chosen_action_structure(
        self,
        option_indices: Tensor,
        legal_counts: Tensor,
        action_program_tokens: Tensor,
        action_program_mask: Tensor,
    ) -> Tensor:
        """Encode sealed current-action structure for Q, not for V.

        The normalization is one-to-one for the validated finite integer
        ranges.  In particular, it prevents a model input alias when two legal
        options share the same 40-D feature vector but differ in selected
        position, legal menu cardinality, stage ordinal, or raw action program.
        """

        batch_size, stage_slots = option_indices.shape
        device = option_indices.device
        index_denominator = float(max(self.config.max_legal_options - 1, 1))
        token_denominator = float(max(self.config.max_action_token_value, 1))
        stage_denominator = float(max(self.config.max_action_stages - 1, 1))
        program_token_slots = action_program_tokens.shape[2]
        stage_ordinal = torch.arange(
            stage_slots, dtype=torch.float32, device=device
        ).view(1, stage_slots, 1) / stage_denominator
        stage_ordinal = stage_ordinal.expand(batch_size, -1, -1)
        program_length = action_program_mask.sum(dim=2, dtype=torch.float32).unsqueeze(-1)
        program_length = program_length / float(self.config.max_action_program_tokens)
        token_values = action_program_tokens.to(dtype=torch.float32) / token_denominator
        token_presence = action_program_mask.to(dtype=torch.float32)
        # The model input fixes token values before presence bits. Pad both
        # blocks independently so a shorter complete action cannot shift its
        # presence bit into another token's value slot.
        missing_token_slots = self.config.max_action_program_tokens - program_token_slots
        if missing_token_slots:
            token_values = F.pad(token_values, (0, missing_token_slots))
            token_presence = F.pad(token_presence, (0, missing_token_slots))
        structure = torch.cat(
            (
                option_indices.to(dtype=torch.float32).unsqueeze(-1) / index_denominator,
                legal_counts.to(dtype=torch.float32).unsqueeze(-1)
                / float(self.config.max_legal_options),
                stage_ordinal,
                program_length,
                token_values,
                token_presence,
            ),
            dim=-1,
        )
        if structure.shape[-1] != self.action_structure_width:
            raise ActionCriticSidecarError("internal chosen-action structure width drifted")
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
        """Return FP32 values in ``ACTION_CRITIC_OUTPUT_NAMES`` order.

        Columns zero and one are win *logits* for BCE.  The six prize columns
        are bounded with ``tanh`` before they leave the sidecar.
        """

        (
            state_features,
            state_mask,
            action_features,
            action_mask,
            option_indices,
            legal_counts,
            action_program_tokens,
            action_program_mask,
        ) = self._validate_inputs(
            first_stage_legal_features,
            first_stage_legal_mask,
            selected_stage_features,
            selected_stage_mask,
            selected_option_indices,
            selected_legal_counts,
            selected_action_program_tokens,
            selected_action_program_mask,
        )

        # V path: the complete first-stage legal menu, and no selected action.
        state_options = self.first_stage_encoder(state_features)
        state_weights = state_mask.unsqueeze(-1).to(dtype=torch.float32)
        state_summary = (state_options * state_weights).sum(dim=1) / state_weights.sum(dim=1)
        state_context = self.state_trunk(state_summary)

        # Q path: an ordered selected-stage prefix plus exact selected-action
        # structure and the state context.  The structure is not reachable
        # from the V heads above.
        action_structure = self._chosen_action_structure(
            option_indices,
            legal_counts,
            action_program_tokens,
            action_program_mask,
        )
        selected_stages = self.selected_stage_encoder(
            torch.cat((action_features, action_structure), dim=-1)
        )
        selected_stages = selected_stages * action_mask.unsqueeze(-1).to(dtype=torch.float32)
        sequence_outputs, _unused_hidden = self.selected_stage_sequence(selected_stages)
        final_stage_index = action_mask.sum(dim=1, dtype=torch.long) - 1
        action_context = sequence_outputs.gather(
            dim=1,
            index=final_stage_index.view(-1, 1, 1).expand(-1, 1, sequence_outputs.shape[-1]),
        ).squeeze(1)
        q_context = self.q_trunk(torch.cat((state_context, action_context), dim=-1))

        values = torch.cat(
            (
                self.v_win_head(state_context),
                self.q_win_head(q_context),
                torch.tanh(self.v_prize_heads[0](state_context)),
                torch.tanh(self.q_prize_heads[0](q_context)),
                torch.tanh(self.v_prize_heads[1](state_context)),
                torch.tanh(self.q_prize_heads[1](q_context)),
                torch.tanh(self.v_prize_heads[2](state_context)),
                torch.tanh(self.q_prize_heads[2](q_context)),
            ),
            dim=-1,
        )
        _validate_action_critic_predictions(values)
        return values

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
    ) -> ActionCriticPredictions:
        """Return a named view over :meth:`forward`'s fixed tensor ABI."""

        return ActionCriticPredictions(
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


def _validate_action_critic_predictions(values: Any) -> Tensor:
    _require_torch()
    if not isinstance(values, torch.Tensor):
        raise ActionCriticSidecarError("predictions must be a torch.Tensor")
    if values.dtype != torch.float32:
        raise ActionCriticSidecarError("predictions must be FP32")
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != ACTION_CRITIC_OUTPUT_COUNT:
        raise ActionCriticSidecarError(
            "predictions must have shape [batch, 8] in ACTION_CRITIC_OUTPUT_NAMES order"
        )
    if not bool(torch.isfinite(values).all().detach().cpu().item()):
        raise ActionCriticSidecarError("predictions must contain only finite values")
    prizes = values[:, 2:]
    if bool((prizes < -1.0).any().detach().cpu().item()) or bool(
        (prizes > 1.0).any().detach().cpu().item()
    ):
        raise ActionCriticSidecarError("prize predictions must be tanh-bounded to [-1, +1]")
    return values


def split_action_critic_predictions(values: Tensor) -> ActionCriticPredictions:
    """Validate and name a tensor returned by :class:`ActionCriticSidecar`."""

    return ActionCriticPredictions(_validate_action_critic_predictions(values))


def _validate_loss_targets(
    predictions: Tensor,
    win_targets: Any,
    prize_targets: Any,
    prize_mask: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(win_targets, torch.Tensor):
        raise ActionCriticSidecarError("win_targets must be a torch.Tensor")
    if not isinstance(prize_targets, torch.Tensor):
        raise ActionCriticSidecarError("prize_targets must be a torch.Tensor")
    if not isinstance(prize_mask, torch.Tensor):
        raise ActionCriticSidecarError("prize_mask must be a torch.Tensor")
    batch_size = predictions.shape[0]
    if win_targets.ndim != 1 or win_targets.shape[0] != batch_size:
        raise ActionCriticSidecarError("win_targets must have shape [batch]")
    if prize_targets.ndim != 2 or tuple(prize_targets.shape) != (batch_size, 3):
        raise ActionCriticSidecarError("prize_targets must have shape [batch, 3]")
    if prize_mask.dtype != torch.bool or prize_mask.ndim != 2 or tuple(prize_mask.shape) != (batch_size, 3):
        raise ActionCriticSidecarError("prize_mask must be bool with shape [batch, 3]")
    for field, value in (
        ("win_targets", win_targets),
        ("prize_targets", prize_targets),
        ("prize_mask", prize_mask),
    ):
        if value.device != predictions.device:
            raise ActionCriticSidecarError(
                f"{field} device {value.device} does not match predictions device {predictions.device}"
            )
    if win_targets.dtype == torch.bool:
        normalized_win_targets = win_targets.to(dtype=torch.float32)
    elif win_targets.dtype == torch.float32:
        normalized_win_targets = win_targets
    else:
        raise ActionCriticSidecarError("win_targets must be bool or FP32")
    if prize_targets.dtype != torch.float32:
        raise ActionCriticSidecarError("prize_targets must be FP32")
    if not bool(torch.isfinite(normalized_win_targets).all().detach().cpu().item()):
        raise ActionCriticSidecarError("win_targets must contain only finite values")
    if not bool(torch.isfinite(prize_targets).all().detach().cpu().item()):
        raise ActionCriticSidecarError("prize_targets must contain only finite values")
    if bool(
        ((normalized_win_targets != 0.0) & (normalized_win_targets != 1.0)).any()
        .detach()
        .cpu()
        .item()
    ):
        raise ActionCriticSidecarError("win_targets must be exactly binary (draw and loss are both 0)")
    if bool((prize_targets < -1.0).any().detach().cpu().item()) or bool(
        (prize_targets > 1.0).any().detach().cpu().item()
    ):
        raise ActionCriticSidecarError("prize_targets must be clipped to [-1, +1]")
    # Numeric values at a false mask are validated only for storage integrity;
    # they are excluded from loss and are never interpreted as a zero label.
    return normalized_win_targets, prize_targets, prize_mask


def action_critic_loss(
    predictions: Tensor,
    *,
    win_targets: Tensor,
    prize_targets: Tensor,
    prize_mask: Tensor,
    prize_loss_weight: float = 1.0,
) -> ActionCriticLoss:
    """Compute BCE win losses and mask-correct SmoothL1 Prize losses.

    Both state and chosen-action heads observe the same completed-action win
    label.  A masked Prize interval contributes no gradient—not a fabricated
    target of zero.  A batch with no labels for one horizon is valid and
    reports a zero loss for that horizon; coverage gates belong to the trainer
    receipt rather than silently changing batch sampling.
    """

    values = _validate_action_critic_predictions(predictions)
    weight = _as_nonnegative_finite_float(prize_loss_weight, field="prize_loss_weight")
    win, prize, mask = _validate_loss_targets(values, win_targets, prize_targets, prize_mask)

    v_win_loss = F.binary_cross_entropy_with_logits(values[:, 0], win, reduction="mean")
    q_win_loss = F.binary_cross_entropy_with_logits(values[:, 1], win, reduction="mean")
    win_loss = (v_win_loss + q_win_loss) * 0.5

    per_horizon_losses: list[Tensor] = []
    valid_counts: list[int] = []
    zero = values[:, 2:].sum() * 0.0
    for horizon, (v_index, q_index) in enumerate(_PRIZE_OUTPUT_INDICES):
        valid = mask[:, horizon]
        valid_count = int(valid.sum().detach().cpu().item())
        valid_counts.append(valid_count)
        if valid_count == 0:
            per_horizon_losses.append(zero)
            continue
        target = prize[:, horizon][valid]
        v_loss = F.smooth_l1_loss(values[:, v_index][valid], target, reduction="mean")
        q_loss = F.smooth_l1_loss(values[:, q_index][valid], target, reduction="mean")
        per_horizon_losses.append((v_loss + q_loss) * 0.5)
    active_losses = [
        loss for loss, count in zip(per_horizon_losses, valid_counts, strict=True) if count > 0
    ]
    prize_loss = sum(active_losses) / len(active_losses) if active_losses else zero
    total = win_loss + prize_loss * weight
    if not bool(torch.isfinite(total).detach().cpu().item()):
        raise ActionCriticSidecarError("critic loss became non-finite")
    return ActionCriticLoss(
        total=total,
        win=win_loss,
        prize=prize_loss,
        v_win=v_win_loss,
        q_win=q_win_loss,
        prize_by_horizon=(per_horizon_losses[0], per_horizon_losses[1], per_horizon_losses[2]),
        valid_prize_count_by_horizon=(valid_counts[0], valid_counts[1], valid_counts[2]),
    )


def _validate_json_metadata(value: Any, *, field: str) -> Any:
    """Deep-copy primitive receipt metadata and reject hidden policy state."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ActionCriticSidecarError(f"{field} cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ActionCriticSidecarError(f"{field} mapping keys must be strings")
            if key.casefold() in _FORBIDDEN_POLICY_STATE_KEYS:
                raise ActionCriticSidecarError(
                    f"{field} may not contain policy state key {key!r}"
                )
            copied[key] = _validate_json_metadata(child, field=f"{field}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_validate_json_metadata(child, field=f"{field}[]") for child in value]
    raise ActionCriticSidecarError(
        f"{field} must contain only JSON-like receipt metadata, never tensors or policy state"
    )


def _validate_sidecar_optimizer(model: ActionCriticSidecar, optimizer: Any) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ActionCriticSidecarError("optimizer must be a torch.optim.Optimizer")
    sidecar_parameter_ids = {id(parameter) for parameter in model.parameters()}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group.get("params", ()):  # pragma: no branch - optimizer ABI.
            if id(parameter) not in sidecar_parameter_ids:
                raise ActionCriticSidecarError(
                    "optimizer contains a non-sidecar parameter in group " f"{group_index}"
                )


def build_action_critic_checkpoint(
    model: ActionCriticSidecar,
    *,
    optimizer: Any | None = None,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standalone checkpoint payload containing no policy state.

    An optional optimizer is accepted only when each of its parameters belongs
    to ``model``.  Receipt metadata is deliberately JSON-like, so a caller
    cannot tuck a policy model state dict into a sidecar checkpoint.
    """

    _require_torch()
    if not isinstance(model, ActionCriticSidecar):
        raise ActionCriticSidecarError("model must be ActionCriticSidecar")
    model._assert_fp32_parameters()
    if optimizer is not None:
        _validate_sidecar_optimizer(model, optimizer)
    clean_training_state = _validate_json_metadata(
        {} if training_state is None else training_state, field="training_state"
    )
    clean_metadata = _validate_json_metadata({} if metadata is None else metadata, field="metadata")
    return {
        "schema": ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA,
        "config": asdict(model.config),
        "sidecar_state_dict": {
            name: tensor.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        "training_state": clean_training_state,
        "metadata": clean_metadata,
    }


def _checkpoint_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionCriticSidecarError(f"checkpoint {field} must be an object")
    return value


def _validate_checkpoint_payload(payload: Any) -> tuple[ActionCriticSidecarConfig, Mapping[str, Tensor], Mapping[str, Any] | None, Mapping[str, Any], Mapping[str, Any]]:
    checkpoint = _checkpoint_mapping(payload, field="payload")
    if checkpoint.get("schema") != ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA:
        raise ActionCriticSidecarError("checkpoint schema is not an action critic sidecar checkpoint")
    forbidden = sorted(key for key in checkpoint if str(key).casefold() in _FORBIDDEN_POLICY_STATE_KEYS)
    if forbidden:
        raise ActionCriticSidecarError(
            "checkpoint may not contain policy state keys: " + ", ".join(forbidden)
        )
    expected_keys = {
        "schema",
        "config",
        "sidecar_state_dict",
        "optimizer_state_dict",
        "training_state",
        "metadata",
    }
    unexpected = sorted(str(key) for key in checkpoint if key not in expected_keys)
    missing = sorted(expected_keys - set(checkpoint))
    if unexpected or missing:
        detail = []
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        if missing:
            detail.append("missing " + ", ".join(missing))
        raise ActionCriticSidecarError("checkpoint fields invalid: " + "; ".join(detail))
    config_payload = _checkpoint_mapping(checkpoint["config"], field="config")
    expected_config = set(ActionCriticSidecarConfig.__dataclass_fields__)
    if set(config_payload) != expected_config:
        raise ActionCriticSidecarError("checkpoint config fields do not match ActionCriticSidecarConfig")
    try:
        config = ActionCriticSidecarConfig(**dict(config_payload))
    except (TypeError, ActionCriticSidecarError) as exc:
        raise ActionCriticSidecarError("checkpoint config is invalid") from exc
    state_dict = _checkpoint_mapping(checkpoint["sidecar_state_dict"], field="sidecar_state_dict")
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ActionCriticSidecarError("sidecar_state_dict must map strings to tensors")
        if value.dtype != torch.float32:
            raise ActionCriticSidecarError("sidecar_state_dict tensors must be FP32")
        if not bool(torch.isfinite(value).all().detach().cpu().item()):
            raise ActionCriticSidecarError("sidecar_state_dict tensors must be finite")
    optimizer_payload = checkpoint["optimizer_state_dict"]
    if optimizer_payload is not None and not isinstance(optimizer_payload, Mapping):
        raise ActionCriticSidecarError("optimizer_state_dict must be an object or null")
    training_state = _validate_json_metadata(checkpoint["training_state"], field="training_state")
    metadata = _validate_json_metadata(checkpoint["metadata"], field="metadata")
    if not isinstance(training_state, Mapping) or not isinstance(metadata, Mapping):
        raise ActionCriticSidecarError("training_state and metadata must be objects")
    return config, state_dict, optimizer_payload, training_state, metadata


def _load_payload_from_path(path: Path, *, map_location: Any) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActionCriticSidecarError("checkpoint path must be a regular non-symlink file")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch before the weights_only keyword.
        payload = torch.load(path, map_location=map_location)
    return _checkpoint_mapping(payload, field="payload")


def load_action_critic_checkpoint(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    device: Any = "cpu",
) -> LoadedActionCriticCheckpoint:
    """Load a strict standalone sidecar checkpoint onto CPU or MPS.

    This helper never loads a policy checkpoint: its schema, exact top-level
    keys, config fields, state keys, dtypes, and finite tensors are validated
    before the returned model is exposed.
    """

    _require_torch()
    payload = _load_payload_from_path(Path(source), map_location="cpu") if isinstance(source, (str, os.PathLike)) else source
    config, state_dict, optimizer_state, training_state, metadata = _validate_checkpoint_payload(payload)
    model = ActionCriticSidecar(config)
    expected_state_keys = set(model.state_dict())
    if set(state_dict) != expected_state_keys:
        raise ActionCriticSidecarError("sidecar_state_dict keys are incompatible with the critic architecture")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ActionCriticSidecarError("sidecar_state_dict is incompatible with the critic architecture") from exc
    model.to(device=device, dtype=torch.float32)
    model._assert_fp32_parameters()
    return LoadedActionCriticCheckpoint(
        model=model,
        optimizer_state_dict=copy.deepcopy(optimizer_state) if optimizer_state is not None else None,
        training_state=copy.deepcopy(dict(training_state)),
        metadata=copy.deepcopy(dict(metadata)),
    )


def save_action_critic_checkpoint(
    path: str | os.PathLike[str],
    model: ActionCriticSidecar,
    *,
    optimizer: Any | None = None,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a standalone sidecar payload at an explicit path."""

    _require_torch()
    destination = Path(path)
    if destination.name in {"", ".", ".."}:
        raise ActionCriticSidecarError("checkpoint path must name a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_action_critic_checkpoint(
        model,
        optimizer=optimizer,
        training_state=training_state,
        metadata=metadata,
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def restore_action_critic_checkpoint(
    model: ActionCriticSidecar,
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    optimizer: Any | None = None,
) -> LoadedActionCriticCheckpoint:
    """Strictly restore a sidecar and an optional sidecar-only optimizer.

    The caller owns the model object; a config mismatch fails before mutation.
    No policy model, policy state, or policy optimizer can enter this path.
    """

    _require_torch()
    if not isinstance(model, ActionCriticSidecar):
        raise ActionCriticSidecarError("model must be ActionCriticSidecar")
    loaded = load_action_critic_checkpoint(source, device=model._parameter_device())
    if loaded.model.config != model.config:
        raise ActionCriticSidecarError("checkpoint config does not match destination sidecar")
    if optimizer is not None:
        _validate_sidecar_optimizer(model, optimizer)
        if loaded.optimizer_state_dict is None:
            raise ActionCriticSidecarError("checkpoint has no optimizer_state_dict to restore")
    model.load_state_dict(loaded.model.state_dict(), strict=True)
    model._assert_fp32_parameters()
    if optimizer is not None:
        optimizer.load_state_dict(dict(loaded.optimizer_state_dict))
    return LoadedActionCriticCheckpoint(
        model=model,
        optimizer_state_dict=loaded.optimizer_state_dict,
        training_state=loaded.training_state,
        metadata=loaded.metadata,
    )
