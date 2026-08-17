"""Frozen-backbone, additive public-rule residual for the isolated r298 study.

This is intentionally not imported by ``model.py``, the policy agent, Fusion,
OwnDeck, matchup routing, or any selector.  It is a small sidecar module for a
future Elmo-only derivative trainer.  The baseline's Card2Vec path—including
its text-derived inputs, learned tensors, feature ABI, and behavior—remains
outside this module and frozen.  This module consumes only the separately
materialized public-rule representation and receipt-sealed structured catalog
sidecar information.

When disabled or passed an exact zero gate, every policy-facing method returns
the original tensor object before examining semantic inputs.  That is stronger
than numerical equality: signed zeroes, NaNs, storage identity, and baseline
logits remain untouched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final


R298_SEMANTIC_PROJECTION_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_semantic_projection/v1"
)
R298_SEMANTIC_PROJECTION_FEATURE_DIM: Final = 40

try:  # Permit schema/materialization inspection on hosts without torch.
    import torch
    import torch.nn as nn
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover - training host supplies torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


class R298SemanticProjectionError(ValueError):
    """The isolated semantic residual received malformed public features."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R298SemanticProjectionError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise R298SemanticProjectionError(f"{field} must be a list")
    return list(value)


def _finite(value: Any, *, maximum: float = 65535.0) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(-maximum, min(maximum, numeric)) / maximum


def _bool(value: Any) -> float:
    return 1.0 if value is True else 0.0


def _enum(value: Any, allowed: Sequence[str]) -> list[float]:
    normalized = str(value).casefold() if isinstance(value, str) else ""
    return [1.0 if normalized == token else 0.0 for token in allowed]


def _option_semantic_features(
    representation: Mapping[str, Any], option: Mapping[str, Any]
) -> list[float]:
    """Encode fixed public semantics without text hashes, serials, or ordinal.

    Card/attack identity itself is not numerically embedded here.  Sealed
    structured Card2Vec-adjacent catalog vectors are consumed by the separate
    public metadata residual, preserving a clean receipt-bound mechanics path.
    """

    state = _mapping(representation.get("state"), field="representation state")
    selection = _mapping(representation.get("selection"), field="representation selection")
    semantic = _mapping(option.get("semantic"), field="representation option semantic")
    payload = _mapping(semantic.get("option"), field="representation option payload")
    players = _mapping(state.get("players"), field="representation players")
    acting = _mapping(players.get("acting"), field="representation acting player")
    opponent = _mapping(players.get("opponent"), field="representation opponent player")
    option_kind = _enum(
        payload.get("option_type"),
        ("attack", "skill", "number", "card", "pass", "retreat", "target", "unknown"),
    )
    selection_kind = _enum(
        selection.get("selection_type"),
        ("card", "pokemon", "energy", "target", "number", "unknown"),
    )
    # The 40 columns are intentionally simple, public, and documented.  No
    # feature carries raw global serials, a candidate ordinal, card names,
    # prose, or a legacy text hash.
    values = [
        *option_kind,  # 8
        *selection_kind,  # 6 = 14
        _finite(payload.get("number")),
        _finite(payload.get("count")),
        _finite(selection.get("min_count")),
        _finite(selection.get("max_count")),
        _finite(selection.get("remain_damage_counter")),
        _finite(selection.get("remain_energy_cost")),  # 20
        # The sealed recent-20 rollout rows contain the public-observation hash
        # but not these scalar values. Keep their reserved ABI columns exact
        # zero so training and runtime cannot disagree or learn from invented
        # reconstructions; the base r274 policy already consumes full state.
        *([0.0] * 16),  # 36
        _bool(payload.get("source")),
        _bool(payload.get("target")),
        _bool(payload.get("stable_simulator_discriminator")),
        _finite(len(_rows(option.get("referenced_card_ids", ()), field="option card refs"))),
    ]
    if len(values) != R298_SEMANTIC_PROJECTION_FEATURE_DIM:
        raise AssertionError("r298 semantic projection feature layout drifted")
    return values


def semantic_option_feature_rows(representation: Mapping[str, Any]) -> list[list[float]]:
    """Return option-aligned public numeric rows from a materialized record.

    The only retained sequence order is the simulator's option alignment.  No
    column encodes that position, so reordering legal options only reorders
    feature rows unless libcg supplied a real stable discriminator.
    """

    row = _mapping(representation, field="public rule representation")
    if row.get("schema") != "poke_bot.alakazam_public_rule_representation/v1":
        raise R298SemanticProjectionError("public rule representation schema drifted")
    options = _rows(row.get("options"), field="public rule representation options")
    if not options:
        raise R298SemanticProjectionError("public rule representation has no legal options")
    return [_option_semantic_features(row, _mapping(option, field="public option")) for option in options]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("r298 semantic projection requires torch on a training host")


@dataclass(frozen=True)
class R298SemanticProjectionConfig:
    """Small sidecar dimensions; no baseline architecture fields are owned."""

    d_model: int
    hidden_width: int = 64
    logit_delta_limit: float = 0.25
    runtime_enabled_default: bool = False
    policy_gate_default: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int) or self.d_model <= 0:
            raise R298SemanticProjectionError("d_model must be a positive integer")
        if isinstance(self.hidden_width, bool) or not isinstance(self.hidden_width, int) or self.hidden_width <= 0:
            raise R298SemanticProjectionError("hidden_width must be a positive integer")
        if not isinstance(self.runtime_enabled_default, bool):
            raise R298SemanticProjectionError("runtime_enabled_default must be bool")
        for name, value in (
            ("logit_delta_limit", self.logit_delta_limit),
            ("policy_gate_default", self.policy_gate_default),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise R298SemanticProjectionError(f"{name} must be finite")
        if self.logit_delta_limit < 0.0:
            raise R298SemanticProjectionError("logit_delta_limit must be nonnegative")


_ModuleBase = object if nn is None else nn.Module


class R298PublicRuleSemanticProjection(_ModuleBase):
    """Trainable additive sidecar, never a replacement Card2Vec encoder."""

    def __init__(self, config: R298SemanticProjectionConfig) -> None:
        _require_torch()
        if not isinstance(config, R298SemanticProjectionConfig):
            raise R298SemanticProjectionError("config must be R298SemanticProjectionConfig")
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.semantic_projection = nn.Sequential(
            nn.Linear(R298_SEMANTIC_PROJECTION_FEATURE_DIM, int(config.hidden_width)),
            nn.GELU(),
            nn.Linear(int(config.hidden_width), self.d_model),
        )
        self.logit_projection = nn.Sequential(
            nn.Linear(self.d_model, int(config.hidden_width)),
            nn.GELU(),
            nn.Linear(int(config.hidden_width), 1),
        )
        # Only the new sidecar's final additions start exact-zero.  We never
        # mutate baseline tensors, Card2Vec, or feature inputs.
        nn.init.zeros_(self.semantic_projection[-1].weight)
        nn.init.zeros_(self.semantic_projection[-1].bias)
        nn.init.zeros_(self.logit_projection[-1].weight)
        nn.init.zeros_(self.logit_projection[-1].bias)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _parameter in self.named_parameters())

    @staticmethod
    def _gate_is_exact_zero(gate: Any) -> bool:
        if isinstance(gate, (int, float)) and not isinstance(gate, bool):
            return float(gate) == 0.0
        if torch is not None and isinstance(gate, torch.Tensor):
            return gate.numel() == 1 and float(gate.detach().cpu().item()) == 0.0
        return False

    def _features_tensor(self, representations: Sequence[Mapping[str, Any]], *, device: Any, dtype: Any) -> Tensor:
        rows = [semantic_option_feature_rows(representation) for representation in representations]
        widths = {len(row) for row in rows}
        if not rows or len(widths) != 1:
            raise R298SemanticProjectionError("batch must contain nonempty equal-width legal stages")
        return torch.as_tensor(rows, dtype=dtype, device=device)

    def augment_option_hidden(
        self,
        base_option_hidden: Tensor,
        representations: Sequence[Mapping[str, Any]] | None,
        *,
        runtime_enabled: bool = False,
        gate: Any = 0.0,
    ) -> Tensor:
        """Add a semantic sidecar after frozen Card2Vec only when armed."""

        if not runtime_enabled or self._gate_is_exact_zero(gate):
            return base_option_hidden
        if not isinstance(base_option_hidden, torch.Tensor) or base_option_hidden.ndim != 3:
            raise R298SemanticProjectionError("base_option_hidden must be [batch, options, d_model]")
        if base_option_hidden.shape[-1] != self.d_model:
            raise R298SemanticProjectionError("base option hidden dimension drifted")
        if representations is None or len(representations) != base_option_hidden.shape[0]:
            raise R298SemanticProjectionError("semantic representation batch does not align")
        features = self._features_tensor(
            representations,
            device=base_option_hidden.device,
            dtype=base_option_hidden.dtype,
        )
        if features.shape[:2] != base_option_hidden.shape[:2]:
            raise R298SemanticProjectionError("semantic option width does not align with base hidden")
        return base_option_hidden + self.semantic_projection(features) * gate

    def apply_to_logits(
        self,
        base_logits: Tensor,
        base_option_hidden: Tensor | None,
        representations: Sequence[Mapping[str, Any]] | None,
        *,
        runtime_enabled: bool = False,
        gate: Any = 0.0,
    ) -> Tensor:
        """Return base logits unchanged by object identity while default-off."""

        if not runtime_enabled or self._gate_is_exact_zero(gate):
            return base_logits
        if not isinstance(base_logits, torch.Tensor) or base_logits.ndim != 2:
            raise R298SemanticProjectionError("base_logits must be [batch, options]")
        if base_option_hidden is None:
            raise R298SemanticProjectionError("armed semantic residual needs frozen option hidden states")
        augmented = self.augment_option_hidden(
            base_option_hidden,
            representations,
            runtime_enabled=True,
            gate=gate,
        )
        if augmented.shape[:2] != base_logits.shape:
            raise R298SemanticProjectionError("semantic residual option width does not align with logits")
        residual = self.logit_projection(augmented).squeeze(-1)
        limit = float(self.config.logit_delta_limit)
        if limit > 0.0:
            residual = residual.tanh() * limit
        else:
            residual = residual * 0.0
        return base_logits + residual * gate


def semantic_projection_schema_manifest() -> dict[str, Any]:
    """Static schema for freezing this sidecar before corpus materialization."""

    return {
        "schema": R298_SEMANTIC_PROJECTION_SCHEMA,
        "version": 1,
        "feature_dim": R298_SEMANTIC_PROJECTION_FEATURE_DIM,
        "feature_contract": {
            "public_rule_representation_schema": "poke_bot.alakazam_public_rule_representation/v1",
            "card2vec_replaced": False,
            "card2vec_inputs_mutated": False,
            "structured_residual_additive_after_or_alongside_card2vec": True,
            "text_hash_exact_mechanics_authority": False,
            "candidate_ordinal_encoded": False,
            "raw_global_serial_encoded": False,
        },
        "default_zero_and_inert": True,
        "runtime_wired": False,
    }


__all__ = [
    "R298_SEMANTIC_PROJECTION_FEATURE_DIM",
    "R298_SEMANTIC_PROJECTION_SCHEMA",
    "R298PublicRuleSemanticProjection",
    "R298SemanticProjectionConfig",
    "R298SemanticProjectionError",
    "semantic_option_feature_rows",
    "semantic_projection_schema_manifest",
]
