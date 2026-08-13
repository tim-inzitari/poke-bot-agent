"""Unwired, zero-gated auxiliary heads for the isolated Alakazam r298 study.

This module is intentionally *not* imported by ``model.py``,
``strategic_heads.py``, Fusion, OwnDeck, matchup adapters, or a runtime
selector.  It is the small derivative counterpart to
``alakazam_simulator_rule_targets_r298``: an explicit, receipt-bound
materializer may eventually feed frozen backbone hidden states into these
heads and train only this module.  Elmo is the pre-handoff implementation and
re-featurization environment; any candidate optimization remains unavailable
until the revision-5 Inzi Blackwell handoff receipt has been verified.

The policy residual is exactly bypassed before inspecting candidate tensors
when it is disabled or its gate is zero.  Consequently its default state is
bit-identical to the baseline's logits, including signed zero / NaN payload
behaviour.  Supervised target heads are deliberately separate from that route;
their labels can be trained and audited without granting policy authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alakazam_simulator_rule_targets_r298 import (
    ACTION_UTILITY_LAYOUT,
    ATTACK_READINESS_LAYOUT,
    GAME_PHASE_CLASSES,
    LETHAL_THREAT_LAYOUT,
    PRIZE_RACE_LAYOUT,
    R298_CANONICAL_CONTRACT_SHA256,
    R298_CANONICAL_GOAL_REVISION,
    R298_CANONICAL_GOAL_SHA256,
    R298_PRODUCTION_TYPED_SOURCE_PATH,
    R298_PRODUCTION_TYPED_SOURCE_SHA256,
    R298_PREDECESSOR_CONTRACT_SHA256,
    R298_PREDECESSOR_GOAL_REVISION,
    R298_PREDECESSOR_GOAL_SHA256,
    R298_REVISION,
    R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
    R298_R5_TRAINING_HOST,
    R298_ROOT_OWNER_REVISION,
    R298_RULE_TARGET_SCHEMA,
    R298_RULE_TARGET_SCHEMA_DIGEST,
    r298_rule_target_schema_manifest,
    TERMINAL_CONVERSION_LAYOUT,
    TURN_RESOURCE_LAYOUT,
)

try:  # Keep corpus/schema inspection usable on a CPU-only host.
    import torch
    import torch.nn as nn
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover - exercised on non-training hosts.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


R298_AUX_HEADS_SCHEMA = "poke_bot.alakazam_rule_aux_heads/v1"
R298_AUX_HEADS_SCHEMA_VERSION = 1
R298_AUX_HEADS_CONFIG_SCHEMA = "poke_bot.alakazam_rule_aux_heads_config/v1"
DEFAULT_R298_AUX_HEADS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "auxiliary_heads"
    / "alakazam-simulator-rule-targets-r298.json"
)

_AUX_SCHEMA_DEFINITION = {
    "schema": R298_AUX_HEADS_SCHEMA,
    "version": R298_AUX_HEADS_SCHEMA_VERSION,
    "revision": R298_REVISION,
    "target_schema": R298_RULE_TARGET_SCHEMA,
    "target_schema_digest": R298_RULE_TARGET_SCHEMA_DIGEST,
    "state_heads": {
        "lethal_threat": len(LETHAL_THREAT_LAYOUT),
        "prize_race": len(PRIZE_RACE_LAYOUT),
        "game_phase": len(GAME_PHASE_CLASSES),
        "opponent_hand_belief": "belief_card_vocab",
        "opponent_remainder_belief": "belief_card_vocab",
    },
    "option_heads": {
        "action_utility": len(ACTION_UTILITY_LAYOUT),
        "terminal_conversion": len(TERMINAL_CONVERSION_LAYOUT),
        "turn_resources": len(TURN_RESOURCE_LAYOUT),
        "attack_readiness": len(ATTACK_READINESS_LAYOUT),
    },
    "option_target_binding": "pool_exact_selected_action_rows_only_no_unchosen_option_labels",
    "all_new_prediction_projections_zero_initialized": True,
    "policy_route_final_projection_zero_initialized": True,
    "policy_gate_default": 0.0,
    "runtime_wired": False,
    "disabled_path_complete_bypass": True,
    "legal_option_set_change": False,
    "revision_5_training_boundary": {
        "schema_freeze_and_handoff_activation_required": True,
        "candidate_training_host_after_handoff": R298_R5_TRAINING_HOST,
        "production_serving_selector_authority": False,
        "revision_4_predecessor_evidence_only": True,
        "blind_revision_4_substitution_allowed": False,
    },
}
R298_AUX_HEADS_SCHEMA_DIGEST = "sha256:" + hashlib.sha256(
    json.dumps(
        _AUX_SCHEMA_DEFINITION,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


class RuleAuxHeadsError(ValueError):
    """The isolated r298 head contract was used with invalid input."""


def r298_aux_heads_schema_manifest() -> dict[str, Any]:
    """Receipt-bindable companion manifest for the zero-inert head module."""

    target_manifest = r298_rule_target_schema_manifest()
    return {
        "schema": R298_AUX_HEADS_SCHEMA,
        "version": R298_AUX_HEADS_SCHEMA_VERSION,
        "revision": R298_REVISION,
        "schema_digest": R298_AUX_HEADS_SCHEMA_DIGEST,
        "target_schema": R298_RULE_TARGET_SCHEMA,
        "target_schema_digest": R298_RULE_TARGET_SCHEMA_DIGEST,
        "target_schema_manifest": target_manifest,
        "canonical_goal": target_manifest["canonical_goal"],
        "canonical_contract": target_manifest["canonical_contract"],
        "root_handoff": target_manifest["root_handoff"],
        "revision_4_predecessor": target_manifest["revision_4_predecessor"],
        "heads": {
            "state": {
                "lethal_threat": len(LETHAL_THREAT_LAYOUT),
                "prize_race": len(PRIZE_RACE_LAYOUT),
                "game_phase": len(GAME_PHASE_CLASSES),
                "opponent_hand_belief": "belief_card_vocab",
                "opponent_remainder_belief": "belief_card_vocab",
            },
            "option": {
                "action_utility": len(ACTION_UTILITY_LAYOUT),
                "terminal_conversion": len(TERMINAL_CONVERSION_LAYOUT),
                "turn_resources": len(TURN_RESOURCE_LAYOUT),
                "attack_readiness": len(ATTACK_READINESS_LAYOUT),
            },
        },
        "all_new_prediction_projections_zero_initialized": True,
        "policy_route_final_projection_zero_initialized": True,
        "policy_gate_default": 0.0,
        "runtime_wired": False,
        "disabled_path_complete_bypass": True,
        "baseline_logits_bit_identical_when_off": True,
        "legal_option_set_change": False,
        "candidate_training_before_30_day_census": False,
        "revision_5_training_boundary": {
            "schema_freeze_and_handoff_activation_required": True,
            "handoff_activation_receipt_schema": R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
            "candidate_training_host_after_handoff": R298_R5_TRAINING_HOST,
            "production_serving_selector_authority": False,
            "revision_4_predecessor_evidence_only": True,
            "blind_revision_4_substitution_allowed": False,
        },
    }


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "r298 auxiliary heads require torch; the target compiler remains torch-free"
        )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise RuleAuxHeadsError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuleAuxHeadsError(f"{field} must be a positive integer") from exc
    if result <= 0 or value != result:
        raise RuleAuxHeadsError(f"{field} must be a positive integer")
    return result


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise RuleAuxHeadsError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuleAuxHeadsError(f"{field} must be finite") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise RuleAuxHeadsError(f"{field} is outside its allowed range")
    return result


@dataclass(frozen=True)
class R298RuleAuxHeadsConfig:
    """Shape and safety settings for the additive derivative module.

    ``runtime_enabled_default`` and ``policy_gate_default`` intentionally
    default to off.  The config is a constructor contract only: it does not
    wire this object into an existing checkpoint or production route.
    """

    d_model: int
    route_width: int = 64
    belief_card_vocab: int = 2048
    policy_delta_limit: float = 0.25
    runtime_enabled_default: bool = False
    policy_gate_default: float = 0.0

    def __post_init__(self) -> None:
        _positive_int(self.d_model, field="d_model")
        _positive_int(self.route_width, field="route_width")
        _positive_int(self.belief_card_vocab, field="belief_card_vocab")
        _finite_float(self.policy_delta_limit, field="policy_delta_limit", minimum=0.0)
        _finite_float(self.policy_gate_default, field="policy_gate_default")
        if not isinstance(self.runtime_enabled_default, bool):
            raise RuleAuxHeadsError("runtime_enabled_default must be bool")


_ModuleBase = object if nn is None else nn.Module


class R298RuleAuxiliaryHeads(_ModuleBase):
    """Separate target heads and a bounded, default-off logit residual.

    The state heads consume ``[batch, d_model]``.  Option heads consume
    ``[batch, option_count, d_model]`` and preserve simulator option order;
    this module neither manufactures nor masks legal options.  The residual
    returns a tensor shaped exactly like ``base_logits`` and has no route to
    legality, search, or selection while unwired.
    """

    def __init__(self, config: R298RuleAuxHeadsConfig) -> None:
        _require_torch()
        if not isinstance(config, R298RuleAuxHeadsConfig):
            raise RuleAuxHeadsError("config must be R298RuleAuxHeadsConfig")
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.belief_card_vocab = int(config.belief_card_vocab)
        self.policy_delta_limit = float(config.policy_delta_limit)
        self.runtime_enabled_default = bool(config.runtime_enabled_default)

        # These are target-only supervised projections.  They are not added to
        # the baseline policy unless a caller explicitly arms the separate
        # residual route below.
        self.lethal_threat_head = nn.Linear(self.d_model, len(LETHAL_THREAT_LAYOUT))
        self.prize_race_head = nn.Linear(self.d_model, len(PRIZE_RACE_LAYOUT))
        self.game_phase_head = nn.Linear(self.d_model, len(GAME_PHASE_CLASSES))
        self.opponent_hand_belief_head = nn.Linear(self.d_model, self.belief_card_vocab)
        self.opponent_remainder_belief_head = nn.Linear(
            self.d_model, self.belief_card_vocab
        )

        self.action_utility_head = nn.Linear(self.d_model, len(ACTION_UTILITY_LAYOUT))
        self.terminal_conversion_head = nn.Linear(
            self.d_model, len(TERMINAL_CONVERSION_LAYOUT)
        )
        self.turn_resources_head = nn.Linear(self.d_model, len(TURN_RESOURCE_LAYOUT))
        self.attack_readiness_head = nn.Linear(self.d_model, len(ATTACK_READINESS_LAYOUT))

        # Every newly introduced prediction projection starts neutral.  These
        # target-only logits are not on the live policy path, and zero init
        # makes their initial behaviour explicit/auditable while retaining
        # first-step gradients to each output row from frozen hidden states.
        for projection in (
            self.lethal_threat_head,
            self.prize_race_head,
            self.game_phase_head,
            self.opponent_hand_belief_head,
            self.opponent_remainder_belief_head,
            self.action_utility_head,
            self.terminal_conversion_head,
            self.turn_resources_head,
            self.attack_readiness_head,
        ):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        # Only this path can alter logits.  Its gate and final projection are
        # initialized exactly at zero, while its upstream route remains normal
        # so an armed first optimization step has a real gradient to the final
        # projection.  This mirrors the public metadata residual's isolation.
        route_input_width = self.d_model * 2 + len(ACTION_UTILITY_LAYOUT) + len(
            TERMINAL_CONVERSION_LAYOUT
        )
        self.policy_route = nn.Sequential(
            nn.Linear(route_input_width, int(config.route_width)),
            nn.GELU(),
            nn.Linear(int(config.route_width), 1),
        )
        nn.init.zeros_(self.policy_route[-1].weight)
        nn.init.zeros_(self.policy_route[-1].bias)
        self.policy_gate = nn.Parameter(torch.tensor(float(config.policy_gate_default)))

    @property
    def target_schema(self) -> dict[str, Any]:
        """Manifest needed by a future corpus/materialization adapter."""

        return {
            "schema": R298_AUX_HEADS_SCHEMA,
            "version": R298_AUX_HEADS_SCHEMA_VERSION,
            "revision": R298_REVISION,
            "target_schema": R298_RULE_TARGET_SCHEMA,
            "target_schema_digest": R298_RULE_TARGET_SCHEMA_DIGEST,
            "layouts": {
                "lethal_threat": list(LETHAL_THREAT_LAYOUT),
                "prize_race": list(PRIZE_RACE_LAYOUT),
                "action_utility": list(ACTION_UTILITY_LAYOUT),
                "game_phase": list(GAME_PHASE_CLASSES),
                "terminal_conversion": list(TERMINAL_CONVERSION_LAYOUT),
                "turn_resources": list(TURN_RESOURCE_LAYOUT),
                "attack_readiness": list(ATTACK_READINESS_LAYOUT),
                "opponent_belief": {
                    "hand_vocab": self.belief_card_vocab,
                    "remainder_vocab": self.belief_card_vocab,
                    "target_only": True,
                },
            },
            "runtime_wired": False,
            "policy_feature_eligible": False,
            "revision_5_training_boundary": {
                "schema_freeze_and_handoff_activation_required": True,
                "handoff_activation_receipt_schema": R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
                "candidate_training_host_after_handoff": R298_R5_TRAINING_HOST,
                "production_serving_selector_authority": False,
                "revision_4_predecessor_evidence_only": True,
                "blind_revision_4_substitution_allowed": False,
            },
        }

    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Expose the only parameters a frozen-backbone trainer may optimize."""

        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def _validate_state_hidden(self, state_hidden: Tensor) -> None:
        if not isinstance(state_hidden, torch.Tensor):
            raise RuleAuxHeadsError("state_hidden must be a torch tensor")
        if state_hidden.ndim != 2 or state_hidden.size(-1) != self.d_model:
            raise RuleAuxHeadsError(
                "state_hidden must have shape [batch, d_model] matching config"
            )
        if not torch.is_floating_point(state_hidden):
            raise RuleAuxHeadsError("state_hidden must be floating point")

    def _validate_option_hidden(self, option_hidden: Tensor, *, batch_size: int) -> None:
        if not isinstance(option_hidden, torch.Tensor):
            raise RuleAuxHeadsError("option_hidden must be a torch tensor")
        if (
            option_hidden.ndim != 3
            or option_hidden.size(0) != batch_size
            or option_hidden.size(-1) != self.d_model
        ):
            raise RuleAuxHeadsError(
                "option_hidden must have shape [batch, option_count, d_model]"
            )
        if not torch.is_floating_point(option_hidden):
            raise RuleAuxHeadsError("option_hidden must be floating point")

    def forward_heads(self, state_hidden: Tensor, option_hidden: Tensor) -> dict[str, Tensor]:
        """Produce supervised logits/regressions without modifying policy logits."""

        self._validate_state_hidden(state_hidden)
        self._validate_option_hidden(option_hidden, batch_size=int(state_hidden.size(0)))
        return {
            "lethal_threat": self.lethal_threat_head(state_hidden),
            "prize_race": self.prize_race_head(state_hidden),
            "game_phase": self.game_phase_head(state_hidden),
            "opponent_hand_belief": self.opponent_hand_belief_head(state_hidden),
            "opponent_remainder_belief": self.opponent_remainder_belief_head(state_hidden),
            "action_utility": self.action_utility_head(option_hidden),
            "terminal_conversion": self.terminal_conversion_head(option_hidden),
            "turn_resources": self.turn_resources_head(option_hidden),
            "attack_readiness": self.attack_readiness_head(option_hidden),
        }

    def policy_delta(self, state_hidden: Tensor, option_hidden: Tensor) -> Tensor:
        """Return a bounded residual only; caller controls whether it is armed."""

        self._validate_state_hidden(state_hidden)
        self._validate_option_hidden(option_hidden, batch_size=int(state_hidden.size(0)))
        batch, option_count, _width = option_hidden.shape
        state_rows = state_hidden.unsqueeze(1).expand(batch, option_count, self.d_model)
        action_utility = self.action_utility_head(option_hidden)
        terminal_conversion = self.terminal_conversion_head(option_hidden)
        route_input = torch.cat(
            (option_hidden, state_rows, action_utility, terminal_conversion), dim=-1
        )
        delta = self.policy_route(route_input).squeeze(-1)
        return torch.tanh(delta) * self.policy_delta_limit

    def apply_to_policy(
        self,
        base_logits: Tensor,
        state_hidden: Tensor | None = None,
        option_hidden: Tensor | None = None,
        *,
        runtime_enabled: bool | None = None,
        gate: float | Tensor | None = None,
    ) -> Tensor:
        """Apply the residual iff explicitly armed; otherwise return ``base_logits``.

        The early return intentionally happens *before* tensor validation or
        projection.  It is therefore safe to call during a baseline-parity
        test with malformed/poisoned derivative inputs and still receive the
        identical base tensor object.
        """

        enabled = self.runtime_enabled_default if runtime_enabled is None else runtime_enabled
        if enabled is not True:
            return base_logits
        applied_gate: float | Tensor = self.policy_gate if gate is None else gate
        if isinstance(applied_gate, torch.Tensor):
            if applied_gate.numel() != 1 or not torch.isfinite(applied_gate).all():
                raise RuleAuxHeadsError("policy gate must be one finite scalar")
            if float(applied_gate.detach().cpu()) == 0.0:
                return base_logits
        else:
            gate_float = _finite_float(applied_gate, field="policy gate")
            if gate_float == 0.0:
                return base_logits
            applied_gate = gate_float
        if not isinstance(base_logits, torch.Tensor):
            raise RuleAuxHeadsError("base_logits must be a torch tensor")
        if base_logits.ndim != 2 or not torch.is_floating_point(base_logits):
            raise RuleAuxHeadsError("base_logits must have shape [batch, option_count]")
        if state_hidden is None or option_hidden is None:
            raise RuleAuxHeadsError("armed residual requires state_hidden and option_hidden")
        delta = self.policy_delta(state_hidden, option_hidden)
        if tuple(delta.shape) != tuple(base_logits.shape):
            raise RuleAuxHeadsError("base logits do not align with simulator option rows")
        if not torch.isfinite(delta).all():
            raise RuleAuxHeadsError("policy residual is non-finite")
        return base_logits + applied_gate * delta


def _target_tensor(
    values: Any,
    *,
    prediction: Tensor,
    field: str,
) -> Tensor:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuleAuxHeadsError(f"{field}.values must be a sequence")
    try:
        result = torch.as_tensor(values, dtype=prediction.dtype, device=prediction.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuleAuxHeadsError(f"{field}.values cannot become a tensor") from exc
    if result.ndim == 1:
        result = result.unsqueeze(0).expand(prediction.size(0), -1)
    if tuple(result.shape) != tuple(prediction.shape):
        raise RuleAuxHeadsError(f"{field}.values do not align with head output")
    return result


def _mask_tensor(mask: Any, *, prediction: Tensor, field: str) -> Tensor:
    if not isinstance(mask, Sequence) or isinstance(mask, (str, bytes)):
        raise RuleAuxHeadsError(f"{field}.mask must be a sequence")
    try:
        result = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuleAuxHeadsError(f"{field}.mask cannot become a tensor") from exc
    if result.ndim == 1:
        result = result.unsqueeze(0).expand(prediction.size(0), -1)
    if tuple(result.shape) != tuple(prediction.shape):
        raise RuleAuxHeadsError(f"{field}.mask does not align with head output")
    return result


def _masked_squared_error(prediction: Tensor, target: Mapping[str, Any], *, field: str) -> Tensor:
    values = _target_tensor(target.get("values"), prediction=prediction, field=field)
    mask = _mask_tensor(target.get("mask"), prediction=prediction, field=field)
    if not bool(mask.any()):
        # A graph-preserving exact zero permits batches whose only available
        # labels are other head families.
        return prediction.sum() * 0.0
    error = (prediction - values).square()
    return error.masked_select(mask).mean()


def _selected_action_prediction(
    prediction: Tensor,
    target: Mapping[str, Any],
    *,
    field: str,
) -> Tensor | None:
    """Pool only rows that form the recorded selected legal action.

    Phase-C labels are observations of one whole selected action.  They cannot
    supervise unchosen options.  A one-element index list selects one row; a
    multi-option action mean-pools its exact selected rows into a joint-action
    prediction.  Materializers may supply one common index list or one list per
    batch row.  An empty legal selection intentionally returns ``None`` so its
    option-conditioned labels stay graph-zero rather than being attached to an
    invented candidate.
    """

    if prediction.ndim != 3:
        raise RuleAuxHeadsError(f"{field} prediction must be [batch, option_count, width]")
    raw_indices = target.get("selected_option_indices")
    if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
        raise RuleAuxHeadsError(f"{field}.selected_option_indices must be a sequence")
    batch_size, option_count, _width = prediction.shape
    rows: list[list[Any]]
    if not raw_indices:
        return None
    if all(not isinstance(item, (list, tuple)) for item in raw_indices):
        rows = [list(raw_indices) for _ in range(batch_size)]
    else:
        if len(raw_indices) != batch_size or not all(
            isinstance(item, (list, tuple)) for item in raw_indices
        ):
            raise RuleAuxHeadsError(
                f"{field}.selected_option_indices must be one list or one list per batch row"
            )
        rows = [list(item) for item in raw_indices]
    indices: list[list[int]] = []
    for batch_index, row in enumerate(rows):
        if not row:
            # Mixed batches with empty actions would require a per-example loss
            # mask.  Materializers must bucket/drop those rows instead of
            # silently borrowing an unchosen option from another example.
            raise RuleAuxHeadsError(
                f"{field}.selected_option_indices[{batch_index}] is empty in a batched option loss"
            )
        parsed: list[int] = []
        for position, value in enumerate(row):
            if isinstance(value, bool):
                raise RuleAuxHeadsError(f"{field}.selected_option_indices must be integers")
            try:
                index = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuleAuxHeadsError(
                    f"{field}.selected_option_indices must be integers"
                ) from exc
            if value != index or not 0 <= index < option_count:
                raise RuleAuxHeadsError(
                    f"{field}.selected_option_indices[{batch_index}][{position}] is outside legal option rows"
                )
            if index in parsed:
                raise RuleAuxHeadsError(f"{field}.selected_option_indices repeats a legal option")
            parsed.append(index)
        indices.append(parsed)
    # Ragged selected actions are deliberately not padded/guessed.  Target
    # materialization should bucket by selection cardinality; a future explicit
    # ragged mask can extend this contract without changing its semantics.
    cardinalities = {len(row) for row in indices}
    if len(cardinalities) != 1:
        raise RuleAuxHeadsError(
            f"{field} batch mixes selected-action cardinalities; bucket it explicitly"
        )
    index_tensor = torch.tensor(indices, dtype=torch.long, device=prediction.device)
    batch_tensor = torch.arange(batch_size, device=prediction.device).unsqueeze(1)
    return prediction[batch_tensor, index_tensor].mean(dim=1)


def _belief_squared_error(prediction: Tensor, target: Mapping[str, Any], *, field: str) -> Tensor:
    if target.get("mask") is not True:
        return prediction.sum() * 0.0
    pairs = target.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise RuleAuxHeadsError(f"{field}.pairs must be a sequence")
    expected = torch.zeros_like(prediction)
    for row in pairs:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise RuleAuxHeadsError(f"{field}.pairs contains an invalid row")
        card_id, count = row
        if isinstance(card_id, bool) or isinstance(count, bool):
            raise RuleAuxHeadsError(f"{field}.pairs contains a boolean")
        try:
            card = int(card_id)
            multiplicity = float(count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuleAuxHeadsError(f"{field}.pairs contains non-numeric data") from exc
        if card < 0 or card >= prediction.size(-1) or not math.isfinite(multiplicity):
            raise RuleAuxHeadsError(f"{field}.pairs is outside the configured vocabulary")
        expected[:, card] = multiplicity
    return (prediction - expected).square().mean()


def masked_rule_auxiliary_loss(
    predictions: Mapping[str, Tensor],
    target_vectors: Mapping[str, Mapping[str, Any]],
) -> Tensor:
    """Compute a graph-preserving masked loss for r298 target-only labels.

    This deliberately performs no policy/logit update itself.  A future frozen
    materializer can choose weights/schedules externally; the unweighted sum
    here is an isolated correctness harness and a compact reference adapter.
    """

    _require_torch()
    required = {
        "lethal_threat",
        "prize_race",
        "action_utility",
        "game_phase",
        "terminal_conversion",
        "turn_resources",
        "attack_readiness",
        "opponent_hand_belief",
        "opponent_remainder_belief",
    }
    missing = sorted(name for name in required if name not in predictions)
    if missing:
        raise RuleAuxHeadsError(f"prediction mapping lacks {', '.join(missing)}")
    loss = predictions["lethal_threat"].sum() * 0.0
    for name in (
        "lethal_threat",
        "prize_race",
        "game_phase",
    ):
        target = target_vectors.get(name)
        if not isinstance(target, Mapping):
            raise RuleAuxHeadsError(f"target mapping lacks {name}")
        loss = loss + _masked_squared_error(predictions[name], target, field=name)
    for name in (
        "action_utility",
        "terminal_conversion",
        "turn_resources",
        "attack_readiness",
    ):
        target = target_vectors.get(name)
        if not isinstance(target, Mapping):
            raise RuleAuxHeadsError(f"target mapping lacks {name}")
        selected_prediction = _selected_action_prediction(
            predictions[name], target, field=name
        )
        if selected_prediction is None:
            # Empty selection contains no selected option to supervise.  It is
            # a valid no-label row, not an excuse to synthesize an END option.
            loss = loss + predictions[name].sum() * 0.0
        else:
            loss = loss + _masked_squared_error(
                selected_prediction, target, field=name
            )
    belief = target_vectors.get("opponent_belief")
    if not isinstance(belief, Mapping):
        raise RuleAuxHeadsError("target mapping lacks opponent_belief")
    hand = belief.get("hand_count_distribution")
    remainder = belief.get("remainder_count_distribution")
    if not isinstance(hand, Mapping) or not isinstance(remainder, Mapping):
        raise RuleAuxHeadsError("opponent belief target has no count distributions")
    loss = loss + _belief_squared_error(
        predictions["opponent_hand_belief"], hand, field="opponent_hand_belief"
    )
    loss = loss + _belief_squared_error(
        predictions["opponent_remainder_belief"],
        remainder,
        field="opponent_remainder_belief",
    )
    return loss


def load_r298_aux_heads_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the staged-only config without enabling or wiring anything."""

    source = DEFAULT_R298_AUX_HEADS_CONFIG_PATH if path is None else Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleAuxHeadsError(f"cannot read r298 auxiliary-head config: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuleAuxHeadsError("r298 auxiliary-head config must be an object")
    result = dict(payload)
    if result.get("schema") != R298_AUX_HEADS_CONFIG_SCHEMA:
        raise RuleAuxHeadsError("r298 auxiliary-head config schema mismatch")
    if result.get("revision") != R298_REVISION:
        raise RuleAuxHeadsError("r298 auxiliary-head config revision mismatch")
    target_contract = result.get("target_contract")
    if not isinstance(target_contract, Mapping):
        raise RuleAuxHeadsError("r298 auxiliary-head config lacks target contract")
    if (
        target_contract.get("schema") != R298_RULE_TARGET_SCHEMA
        or target_contract.get("schema_digest") != R298_RULE_TARGET_SCHEMA_DIGEST
        or target_contract.get("policy_feature_eligible") is not False
        or target_contract.get(
            "revision_5_schema_freeze_receipt_required_before_materialization"
        )
        is not True
        or target_contract.get(
            "revision_5_handoff_activation_required_before_trainable_masks"
        )
        is not True
    ):
        raise RuleAuxHeadsError("r298 auxiliary-head config target contract mismatch")
    frozen_manifest = result.get("frozen_schema_manifest")
    if not isinstance(frozen_manifest, Mapping):
        raise RuleAuxHeadsError("r298 auxiliary-head config lacks frozen schema manifest")
    manifest_path = frozen_manifest.get("path")
    manifest_digest = frozen_manifest.get("sha256")
    if not isinstance(manifest_path, str) or not isinstance(manifest_digest, str):
        raise RuleAuxHeadsError("r298 frozen schema manifest binding is malformed")
    manifest_file = Path(__file__).resolve().parents[1] / manifest_path
    if not manifest_file.is_file():
        raise RuleAuxHeadsError("r298 frozen schema manifest is absent")
    actual_manifest_digest = "sha256:" + hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    if actual_manifest_digest != manifest_digest:
        raise RuleAuxHeadsError("r298 frozen schema manifest digest is stale")
    try:
        manifest_payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuleAuxHeadsError("r298 frozen schema manifest is invalid JSON") from exc
    manifest_target = (
        manifest_payload.get("target_schema")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    if (
        not isinstance(manifest_target, Mapping)
        or manifest_target.get("schema") != R298_RULE_TARGET_SCHEMA
        or manifest_target.get("schema_digest") != R298_RULE_TARGET_SCHEMA_DIGEST
    ):
        raise RuleAuxHeadsError("r298 frozen schema manifest target binding mismatch")
    manifest_auxiliary = (
        manifest_payload.get("auxiliary_heads")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    if (
        not isinstance(manifest_auxiliary, Mapping)
        or manifest_auxiliary.get("schema") != R298_AUX_HEADS_SCHEMA
        or manifest_auxiliary.get("schema_digest") != R298_AUX_HEADS_SCHEMA_DIGEST
    ):
        raise RuleAuxHeadsError("r298 frozen schema manifest auxiliary-head binding mismatch")
    authority = result.get("canonical_authority")
    if not isinstance(authority, Mapping):
        raise RuleAuxHeadsError("r298 auxiliary-head config lacks canonical authority")
    expected_authority = r298_rule_target_schema_manifest()
    project_root = Path(__file__).resolve().parents[1]
    for file_key, digest_key, manifest_key in (
        ("goal_path", "goal_sha256", "canonical_goal"),
        ("contract_path", "contract_sha256", "canonical_contract"),
    ):
        relative = authority.get(file_key)
        expected = authority.get(digest_key)
        bound = expected_authority[manifest_key]
        if relative != bound["path"] or expected != bound["sha256"]:
            raise RuleAuxHeadsError("r298 auxiliary-head config authority binding mismatch")
        candidate = project_root / str(relative)
        if not candidate.is_file():
            raise RuleAuxHeadsError("r298 auxiliary-head authority file is absent")
        actual = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise RuleAuxHeadsError("r298 auxiliary-head authority digest is stale")
    expected_handoff = expected_authority["root_handoff"]
    if (
        authority.get("goal_revision") != R298_CANONICAL_GOAL_REVISION
        or authority.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
        or authority.get("production_typed_source")
        != expected_handoff["production_typed_source"]
        or authority.get("production_typed_source_sha256")
        != expected_handoff["production_typed_source_sha256"]
    ):
        raise RuleAuxHeadsError("r298 auxiliary-head revision-5 handoff binding mismatch")
    production_source = project_root / str(authority.get("production_typed_source"))
    if not production_source.is_file():
        raise RuleAuxHeadsError("r298 auxiliary-head production typed source is absent")
    production_source_digest = "sha256:" + hashlib.sha256(
        production_source.read_bytes()
    ).hexdigest()
    if production_source_digest != authority.get("production_typed_source_sha256"):
        raise RuleAuxHeadsError("r298 auxiliary-head production typed source digest is stale")
    predecessor = result.get("revision_4_predecessor")
    expected_predecessor = expected_authority["revision_4_predecessor"]
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("goal_revision")
        != expected_predecessor["goal_revision"]
        or predecessor.get("goal_sha256") != expected_predecessor["goal_sha256"]
        or predecessor.get("contract_sha256")
        != expected_predecessor["contract_sha256"]
        or predecessor.get("historical_evidence_only") is not True
        or predecessor.get("blind_hash_substitution_allowed") is not False
        or predecessor.get("satisfies_revision_5_schema_freeze_alone") is not False
    ):
        raise RuleAuxHeadsError("r298 auxiliary-head revision-4 predecessor boundary mismatch")
    manifest_authority = (
        manifest_payload.get("canonical_authority")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    manifest_handoff = (
        manifest_payload.get("root_handoff")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    manifest_predecessor = (
        manifest_payload.get("revision_4_predecessor")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    if (
        not isinstance(manifest_authority, Mapping)
        or manifest_authority.get("goal_sha256") != R298_CANONICAL_GOAL_SHA256
        or manifest_authority.get("goal_revision") != R298_CANONICAL_GOAL_REVISION
        or manifest_authority.get("contract_sha256") != R298_CANONICAL_CONTRACT_SHA256
        or not isinstance(manifest_handoff, Mapping)
        or manifest_handoff.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
        or manifest_handoff.get("production_typed_source")
        != R298_PRODUCTION_TYPED_SOURCE_PATH
        or manifest_handoff.get("production_typed_source_sha256")
        != R298_PRODUCTION_TYPED_SOURCE_SHA256
        or not isinstance(manifest_predecessor, Mapping)
        or manifest_predecessor.get("goal_revision")
        != R298_PREDECESSOR_GOAL_REVISION
        or manifest_predecessor.get("goal_sha256") != R298_PREDECESSOR_GOAL_SHA256
        or manifest_predecessor.get("contract_sha256")
        != R298_PREDECESSOR_CONTRACT_SHA256
        or manifest_predecessor.get("historical_evidence_only") is not True
        or manifest_predecessor.get("blind_hash_substitution_allowed") is not False
    ):
        raise RuleAuxHeadsError("r298 frozen schema manifest revision-5 binding mismatch")
    materialization_gate = (
        manifest_payload.get("materialization_gate")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    if (
        not isinstance(materialization_gate, Mapping)
        or materialization_gate.get("revision_5_schema_freeze_receipt_required")
        is not True
        or materialization_gate.get("revision_5_handoff_activation_required")
        is not True
        or materialization_gate.get("production_serving_selector_authority")
        is not False
    ):
        raise RuleAuxHeadsError("r298 frozen schema manifest handoff gate mismatch")
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuleAuxHeadsError("r298 auxiliary-head config lacks runtime section")
    if runtime.get("runtime_wired") is not False or runtime.get("enabled_default") is not False:
        raise RuleAuxHeadsError("r298 auxiliary heads must remain unwired and default-off")
    if _finite_float(runtime.get("policy_gate"), field="runtime.policy_gate") != 0.0:
        raise RuleAuxHeadsError("r298 auxiliary policy gate must be exactly zero")
    if (
        runtime.get("immediate_inzi_execution_authority") is not False
        or runtime.get("production_authority") is not False
        or runtime.get("serving_selector_authority") is not False
    ):
        raise RuleAuxHeadsError("r298 auxiliary heads must not claim runtime authority")
    training_boundary = result.get("training_boundary")
    if (
        not isinstance(training_boundary, Mapping)
        or training_boundary.get("pre_handoff_elmo_experiment_only") is not True
        or training_boundary.get("candidate_training_enabled_now") is not False
        or training_boundary.get("candidate_training_host_after_handoff")
        != R298_R5_TRAINING_HOST
        or training_boundary.get("revision_5_handoff_activation_receipt_required")
        is not True
    ):
        raise RuleAuxHeadsError("r298 auxiliary-head training boundary mismatch")
    return result


__all__ = [
    "DEFAULT_R298_AUX_HEADS_CONFIG_PATH",
    "R298_AUX_HEADS_CONFIG_SCHEMA",
    "R298_AUX_HEADS_SCHEMA",
    "R298_AUX_HEADS_SCHEMA_DIGEST",
    "R298_AUX_HEADS_SCHEMA_VERSION",
    "R298RuleAuxHeadsConfig",
    "R298RuleAuxiliaryHeads",
    "RuleAuxHeadsError",
    "load_r298_aux_heads_config",
    "masked_rule_auxiliary_loss",
    "r298_aux_heads_schema_manifest",
]
