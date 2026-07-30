"""CABT transformer with a history-consistent train/deploy contract.

Design choice (board tokens)
----------------------------
``features.build_board_tokens`` / ``build_option_tokens`` return
:class:`~poke_bot.features.SparseVector` bags sized for ``nn.EmbeddingBag``.
This module therefore consumes those sparse bags directly via EmbeddingBag
encoders (same contract as the official RL+MCTS sample), then runs a dense
spatial Transformer over the resulting 24 board tokens. Dense projected card
embeddings are *not* used for the board path so we stay 1:1 with the foundation
feature builders; card-id density is already inside the bag features.

Architecture
------------
1. Spatial EmbeddingBag + explicit slot ids → 24 tokens → pre-norm
   TransformerEncoder.
2. Mean-pool spatial memory → per-timestep ``[CLS]`` for the temporal tower.
3. Causal temporal tower over each acting seat's deployment-visible realized
   observation history. Offline whole-sequence and incremental KV-cache paths
   implement the same computation.
4. Option EmbeddingBag → cross-attention decoder over spatial memory + state
   → per-option policy logits (variable legal set).
5. Value head → tanh scalar (info-set only).
6. Aux head stub → opponent-hand / archetype logits (info-set *input*; privileged
   labels are supplied separately at train time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config, features
from .features import SparseVector
from .matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as MATCHUP_ADAPTER_V5_FORMAT,
)
from .matchup_adapters import MatchupAdapterBank
from .matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as MATCHUP_ADAPTER_V6_FORMAT,
)
from .matchup_adapters_v6 import MatchupAdapterBankV6

Tensor = torch.Tensor

# Additive V6 auxiliary-head contract.  These modules are absent unless the
# serialized ModelConfig explicitly opts in, so a V5 checkpoint keeps its
# original architecture and serving cost.
EXPANDED_HEAD_SCHEMA = "poke_bot.expanded_strategic_heads/v1"
EXPANDED_HEAD_SCHEMA_VERSION = 1
EXPANDED_HEAD_INIT_SEED = 0x56_2026_07
EXPANDED_HEAD_SPECS: tuple[tuple[str, str, int], ...] = (
    ("action_q_head", "option", 1),
    ("action_type_head", "option", 1),
    ("action_target_head", "option", 1),
    ("action_resource_head", "option", 1),
    ("action_utility_head", "option", 6),
    ("tactical_outcome_head", "state", 3 * 6),
    ("opponent_response_head", "state", 7),
    ("resource_forecast_head", "state", 6),
    ("game_phase_head", "state", 5),
    ("outcome_distribution_head", "state", 3),
    ("remaining_turns_head", "state", 1),
)
EXPANDED_HEAD_NAMES: tuple[str, ...] = tuple(
    name for name, _source, _outputs in EXPANDED_HEAD_SPECS
)
EXPANDED_HEAD_KEY_PREFIXES: tuple[str, ...] = tuple(
    f"{name}." for name in EXPANDED_HEAD_NAMES
)
SETUP_BOARD_OUTCOME_HEAD_SCHEMA = "poke_bot.setup_board_outcome_head/v1"
SETUP_BOARD_OUTCOME_HEAD_NAME = "setup_board_outcome_head"
SETUP_BOARD_OUTCOME_HEAD_OUTPUTS = 9
SETUP_BOARD_OUTCOME_HEAD_HIDDEN = 512
SETUP_BOARD_OUTCOME_HEAD_KEY_PREFIX = f"{SETUP_BOARD_OUTCOME_HEAD_NAME}."
DECISION_FUSION_SCHEMA = "poke_bot.causal_decision_fusion/v1"
DECISION_FUSION_V2_SCHEMA = "poke_bot.causal_decision_fusion/v2"
DECISION_FUSION_V2_ROUTE_SCHEMA = "option_conditioned_per_head/v2"
DECISION_FUSION_V2_ROUTE_WIDTH = 16
DECISION_FUSION_V2_TOTAL_DELTA_CAP = 1.0
DECISION_FUSION_REQUIRED_HEADS: tuple[str, ...] = (
    "value",
    "archetype",
    "opponent_hand",
    "opponent_remainder",
    "lethal_threat",
    "prize_race",
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcomes",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
)
DECISION_FUSION_V2_OPTIONAL_HEADS: tuple[str, ...] = (
    "setup_board_outcome",
)
DECISION_FUSION_KEY_PREFIX = "decision_fusion."


class SetupBoardOutcomeHead(nn.Module):
    """Future-only independent setup-board outcome prediction branch."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(d_model), SETUP_BOARD_OUTCOME_HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(
                SETUP_BOARD_OUTCOME_HEAD_HIDDEN,
                SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
            ),
        )

    def forward(self, option_hidden: Tensor) -> Tensor:
        return self.network(option_hidden)

    def inventory(self) -> dict[str, object]:
        return {
            "schema": SETUP_BOARD_OUTCOME_HEAD_SCHEMA,
            "enabled": True,
            "input": "option_hidden",
            "outputs": SETUP_BOARD_OUTCOME_HEAD_OUTPUTS,
            "hidden_width": SETUP_BOARD_OUTCOME_HEAD_HIDDEN,
            "computation_role": "independent_head",
            "fusion_role": "fused_input",
            "action_influence": "bounded_option_conditioned_route",
            "causal_input": "board_state_cross_attended_option_hidden",
            "action_route_input": (
                "typed_option_output_plus_board_state_cross_attended_legal_option"
            ),
            "direct_action_selection_authority": False,
            "runtime_activation": "receipt_gated",
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "numel": int(tensor.numel()),
                }
                for name, tensor in self.state_dict().items()
            },
        }


class OptionConditionedHeadRoute(nn.Module):
    """One nonlinear, zero-safe action route for one typed prediction head."""

    def __init__(self, *, d_model: int, head_dim: int, width: int) -> None:
        super().__init__()
        if d_model <= 0 or head_dim <= 0 or width <= 0:
            raise ValueError("dedicated fusion-route dimensions must be positive")
        self.d_model = int(d_model)
        self.head_dim = int(head_dim)
        self.width = int(width)
        self.network = nn.Sequential(
            nn.Linear(self.d_model + self.head_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, 1),
        )
        # A fusion-v2 migration must reproduce the learned v1 policy exactly.
        # The hidden projection may be initialized normally, but no route can
        # alter a logit until ordinary training updates this final projection.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, option_hidden: Tensor, typed_output: Tensor) -> Tensor:
        if option_hidden.dim() != 3:
            raise ValueError("option hidden must be [batch, options, d_model]")
        if typed_output.dim() != 3:
            raise ValueError("typed route output must be [batch, options, width]")
        if option_hidden.shape[:2] != typed_output.shape[:2]:
            raise ValueError("typed route output does not match option dimensions")
        if option_hidden.size(-1) != self.d_model:
            raise ValueError("dedicated route option-hidden width mismatch")
        if typed_output.size(-1) != self.head_dim:
            raise ValueError("dedicated route typed-output width mismatch")
        bounded_input = torch.cat(
            [
                torch.tanh(option_hidden.float()).to(dtype=option_hidden.dtype),
                torch.tanh(typed_output.float()).to(
                    device=option_hidden.device,
                    dtype=option_hidden.dtype,
                ),
            ],
            dim=-1,
        )
        return torch.tanh(self.network(bounded_input).squeeze(-1))


class CausalDecisionFusion(nn.Module):
    """Learned bounded residual over the flat policy.

    Every required head has a distinct projection or option feature, so no
    source can be silently omitted. The final layer starts at exact zero:
    architecture migration is behavior-preserving until joint training gives
    the residual evidence-backed weight.
    """

    _STATE_DIMS = {
        "value": 1,
        "lethal_threat": 1,
        "prize_race": 2,
        "tactical_outcomes": 18,
        "opponent_response": 7,
        "resource_forecast": 6,
        "game_phase": 5,
        "outcome_distribution": 3,
        "remaining_turns": 1,
    }
    _OPTION_DIMS = {
        "action_q": 1,
        "action_type": 1,
        "action_target": 1,
        "action_resource": 1,
        "action_utility": 6,
    }

    def __init__(
        self,
        *,
        d_model: int,
        width: int,
        archetype_classes: int,
        belief_card_vocab: int,
        dedicated_routes_enabled: bool = False,
        dedicated_route_width: int = DECISION_FUSION_V2_ROUTE_WIDTH,
        dedicated_route_total_delta_cap: float = (
            DECISION_FUSION_V2_TOTAL_DELTA_CAP
        ),
        setup_board_outcome_outputs: int = 0,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("decision fusion width must be positive")
        state_dims = {
            **self._STATE_DIMS,
            "archetype": int(archetype_classes),
            "opponent_hand": int(belief_card_vocab),
            "opponent_remainder": int(belief_card_vocab),
        }
        self.state_projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, width, bias=False)
                for name, input_dim in state_dims.items()
            }
        )
        self.state_gates = nn.ParameterDict(
            {name: nn.Parameter(torch.ones(())) for name in state_dims}
        )
        self.state_norm = nn.LayerNorm(width)
        option_width = sum(self._OPTION_DIMS.values())
        hidden = max(width, d_model // 2)
        self.residual = nn.Sequential(
            nn.Linear(d_model + width + option_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.dedicated_routes_enabled = bool(dedicated_routes_enabled)
        self.dedicated_route_total_delta_cap = float(
            dedicated_route_total_delta_cap
        )
        if (
            not torch.isfinite(
                torch.tensor(self.dedicated_route_total_delta_cap)
            )
            or self.dedicated_route_total_delta_cap <= 0.0
        ):
            raise ValueError(
                "dedicated fusion-route total delta cap must be finite and positive"
            )
        available_route_dims = {**state_dims, **self._OPTION_DIMS}
        route_dims = {
            name: available_route_dims[name]
            for name in DECISION_FUSION_REQUIRED_HEADS
        }
        if int(setup_board_outcome_outputs) > 0:
            if not self.dedicated_routes_enabled:
                raise ValueError(
                    "setup board outcome fusion requires dedicated routes"
                )
            route_dims["setup_board_outcome"] = int(
                setup_board_outcome_outputs
            )
        self.dedicated_route_dims = (
            dict(route_dims) if self.dedicated_routes_enabled else {}
        )
        self.dedicated_routes = nn.ModuleDict(
            {
                name: OptionConditionedHeadRoute(
                    d_model=d_model,
                    head_dim=head_dim,
                    width=dedicated_route_width,
                )
                for name, head_dim in self.dedicated_route_dims.items()
            }
        )

    @property
    def required_heads(self) -> tuple[str, ...]:
        optional = tuple(
            name
            for name in DECISION_FUSION_V2_OPTIONAL_HEADS
            if name in self.dedicated_route_dims
        )
        return (*DECISION_FUSION_REQUIRED_HEADS, *optional)

    @staticmethod
    def _bounded(value: Tensor) -> Tensor:
        return torch.tanh(value.float()).to(dtype=value.dtype)

    def forward(
        self,
        option_hidden: Tensor,
        base_logits: Tensor,
        *,
        state_sources: dict[str, Tensor],
        option_sources: dict[str, Tensor],
        dedicated_routes_active: bool = False,
    ) -> Tensor:
        missing = [
            name
            for name in self.required_heads
            if name not in state_sources and name not in option_sources
        ]
        if missing:
            raise RuntimeError(
                f"causal decision fusion missing required heads: {missing}"
            )
        state_terms: list[Tensor] = []
        for name, projection in self.state_projections.items():
            value = state_sources[name]
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            value = value.reshape(value.size(0), -1)
            state_terms.append(
                self.state_gates[name] * self._bounded(projection(self._bounded(value)))
            )
        context = self.state_norm(torch.stack(state_terms, dim=0).mean(dim=0))
        context = context.unsqueeze(1).expand(-1, option_hidden.size(1), -1)
        option_terms: list[Tensor] = []
        for name in self._OPTION_DIMS:
            value = option_sources[name]
            if value.dim() == 2:
                value = value.unsqueeze(-1)
            option_terms.append(self._bounded(value))
        option_features = torch.cat(option_terms, dim=-1)
        residual = self.residual(
            torch.cat([option_hidden, context, option_features], dim=-1)
        ).squeeze(-1)
        fused_logits = base_logits + residual
        if dedicated_routes_active:
            fused_logits = fused_logits + self.dedicated_action_delta(
                option_hidden,
                state_sources=state_sources,
                option_sources=option_sources,
            )
        return fused_logits

    @staticmethod
    def _option_conditioned_source(
        source: Tensor,
        *,
        batch_size: int,
        option_count: int,
        already_option_conditioned: bool,
    ) -> Tensor:
        if not already_option_conditioned:
            if source.size(0) != batch_size:
                raise ValueError("state fusion source batch mismatch")
            return source.reshape(batch_size, 1, -1).expand(
                -1, option_count, -1
            )
        if source.dim() == 1:
            source = source.unsqueeze(-1)
        if source.dim() == 2:
            source = source.unsqueeze(-1)
        if source.dim() == 3:
            if source.shape[:2] != (batch_size, option_count):
                raise ValueError("option fusion source dimensions mismatch")
            return source
        raise ValueError("unsupported typed fusion-source shape")

    def dedicated_route_deltas(
        self,
        option_hidden: Tensor,
        *,
        state_sources: dict[str, Tensor],
        option_sources: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Return independently ablatable, unit-bounded per-head route deltas."""

        if not self.dedicated_routes_enabled:
            raise RuntimeError("dedicated option-conditioned routes are disabled")
        batch_size, option_count, _ = option_hidden.shape
        deltas: dict[str, Tensor] = {}
        for name, route in self.dedicated_routes.items():
            already_option_conditioned = name in option_sources
            source = (
                option_sources[name]
                if already_option_conditioned
                else state_sources[name]
            )
            typed_output = self._option_conditioned_source(
                source,
                batch_size=batch_size,
                option_count=option_count,
                already_option_conditioned=already_option_conditioned,
            )
            deltas[name] = route(option_hidden, typed_output)
        return deltas

    def dedicated_action_delta(
        self,
        option_hidden: Tensor,
        *,
        state_sources: dict[str, Tensor],
        option_sources: dict[str, Tensor],
    ) -> Tensor:
        """Return the fixed-mean all-head residual, capped at one logit."""

        deltas = self.dedicated_route_deltas(
            option_hidden,
            state_sources=state_sources,
            option_sources=option_sources,
        )
        if not deltas:
            return torch.zeros(
                option_hidden.shape[:2],
                device=option_hidden.device,
                dtype=option_hidden.dtype,
            )
        mean_delta = torch.stack(tuple(deltas.values()), dim=0).mean(dim=0)
        return self.dedicated_route_total_delta_cap * mean_delta

    def inventory(
        self,
        *,
        runtime_enabled: bool,
        dedicated_routes_runtime_enabled: bool = False,
    ) -> dict[str, object]:
        route_count = len(self.dedicated_routes)
        return {
            "schema": (
                DECISION_FUSION_V2_SCHEMA
                if self.dedicated_routes_enabled
                else DECISION_FUSION_SCHEMA
            ),
            "enabled": True,
            "runtime_enabled": bool(runtime_enabled),
            "required_heads": list(self.required_heads),
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "zero_safe_initialization": True,
            "guide_excluded": True,
            "dedicated_routes": {
                "schema": DECISION_FUSION_V2_ROUTE_SCHEMA,
                "enabled": self.dedicated_routes_enabled,
                "runtime_enabled": bool(dedicated_routes_runtime_enabled),
                "route_count": route_count,
                "route_names": list(self.dedicated_routes),
                "aggregation": "fixed_mean",
                "total_delta_cap": self.dedicated_route_total_delta_cap,
                "zero_safe_final_projection": True,
                "computation_role": "independent_head",
                "fusion_role": "fused_input",
                "action_influence": "bounded_option_conditioned_route",
                "causal_input_authority": (
                    "current_board_state_and_legal_options_only"
                ),
                "state_head_action_conditioning": (
                    "typed_output_plus_board_state_cross_attended_legal_option"
                ),
                "option_head_action_conditioning": (
                    "typed_option_output_plus_board_state_cross_attended_legal_option"
                ),
                "direct_action_selection_authority": False,
                "runtime_activation": "receipt_gated",
            },
        }


# ---------------------------------------------------------------------------
# SparseVector packing (EmbeddingBag batch inputs)
# ---------------------------------------------------------------------------

@dataclass
class PackedSparse:
    """Batched EmbeddingBag inputs for one or more SparseVectors."""

    index: Tensor  # int64 [nnz]
    value: Tensor  # float [nnz]
    offset: Tensor  # int64 [n_words + 1]  (EmbeddingBag wants len=N+1)


def pack_sparse_vectors(svs: Sequence[SparseVector], device: torch.device) -> PackedSparse:
    """Pack one or more SparseVectors into a single EmbeddingBag batch.

    Offsets are concatenated across words and finished with a trailing total-nnz
    sentinel as required by ``nn.EmbeddingBag``.
    """
    index: list[int] = []
    value: list[float] = []
    offset: list[int] = [0]
    for sv in svs:
        base = len(index)
        for w, start in enumerate(sv.offset):
            end = sv.offset[w + 1] if w + 1 < len(sv.offset) else len(sv.index)
            for j in range(start, end):
                index.append(sv.index[j])
                value.append(sv.value[j])
            offset.append(len(index))
        # If a SparseVector somehow has zero words, still advance cleanly.
        if not sv.offset and base == len(index):
            pass
    return PackedSparse(
        index=torch.tensor(index, dtype=torch.long, device=device),
        value=torch.tensor(value, dtype=torch.float32, device=device),
        offset=torch.tensor(offset, dtype=torch.long, device=device),
    )


def pack_sparse_batch(
    batch: Sequence[SparseVector],
    words_per: int,
    device: torch.device,
) -> PackedSparse:
    """Pack a batch of SparseVectors that each have exactly ``words_per`` words."""
    for sv in batch:
        if sv.num_words != words_per:
            raise ValueError(
                f"expected {words_per} words, got {sv.num_words}"
            )
    return pack_sparse_vectors(list(batch), device)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to q/k with shape [B, H, T, D]."""
    # cos/sin: [T, D] or [1, 1, T, D]
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    """Standard RoPE cache for the temporal axis."""

    def __init__(self, dim: int, max_seq: int = 320, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq = max_seq
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq = seq_len

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        end = offset + seq_len
        if end > self.max_seq:
            # Incremental inference advances one token at a time.  Grow
            # geometrically so positions beyond a full context window do not
            # rebuild the complete trigonometric cache on every decision.
            self._build_cache(max(end, 2 * self.max_seq))
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        return cos, sin


# ---------------------------------------------------------------------------
# Temporal layer with KV cache
# ---------------------------------------------------------------------------

@dataclass
class TemporalKVCache:
    """Per-layer cached (K, V) for incremental whole-game encode.

    Each tensor is ``[B, H, T, head_dim]``. ``length`` is the retained T;
    ``next_position`` is the monotonic absolute RoPE position.  The two values
    intentionally diverge once the rolling cache reaches ``max_context``.
    ``input_tokens`` retains the small raw CLS window so a full cache can be
    recomputed exactly when its oldest token is evicted.
    """

    layers: list[tuple[Tensor, Tensor]]
    length: int = 0
    next_position: Optional[int] = None
    input_tokens: Optional[Tensor] = None

    def resolved_next_position(self) -> int:
        """Absolute append position, with compatibility for in-memory v4 caches."""
        if self.next_position is None:
            return int(self.length)
        return int(self.next_position)

    def clone(self) -> "TemporalKVCache":
        return TemporalKVCache(
            layers=[(k.clone(), v.clone()) for k, v in self.layers],
            length=self.length,
            next_position=self.next_position,
            input_tokens=(
                None if self.input_tokens is None else self.input_tokens.clone()
            ),
        )

    def truncate(self, length: int) -> "TemporalKVCache":
        if length < 0:
            raise ValueError(f"cache length must be non-negative, got {length}")
        if length >= self.length:
            return self.clone()
        removed = self.length - length
        return TemporalKVCache(
            layers=[(k[..., :length, :], v[..., :length, :]) for k, v in self.layers],
            length=length,
            next_position=max(0, self.resolved_next_position() - removed),
            input_tokens=(
                None
                if self.input_tokens is None
                else self.input_tokens[:, :length, :]
            ),
        )


class TemporalSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE + optional KV cache append."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        rope: Optional[RotaryEmbedding],
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
        cache_offset: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(out, new_k, new_v)`` where new_k/v include prior cache."""
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope and rope is not None:
            cos, sin = rope(t, offset=cache_offset)
            q, k = apply_rope(q, k, cos, sin)

        if cache_k is not None and cache_v is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        drop = self.dropout.p if self.training else 0.0
        if cache_k is not None and key_padding_mask is not None:
            raise ValueError("key padding is only supported for offline temporal encode")
        if cache_k is None and key_padding_mask is None:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=True
            )
        elif cache_k is None:
            padding = key_padding_mask.to(device=q.device, dtype=torch.bool)
            if tuple(padding.shape) != (b, t):
                raise ValueError(
                    "temporal key padding mask shape mismatch: "
                    f"expected={(b, t)} actual={tuple(padding.shape)}"
                )
            causal = torch.ones(t, t, device=q.device, dtype=torch.bool).tril_()
            allow = causal.view(1, 1, t, t) & (~padding).view(b, 1, 1, t)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allow,
                dropout_p=drop,
                is_causal=False,
            )
        else:
            # Append path: q_len may be << k_len; build an additive causal mask.
            q_len, k_len = q.size(2), k.size(2)
            q_idx = torch.arange(q_len, device=q.device) + cache_offset
            cached_len = int(cache_k.size(2))
            cached_start = cache_offset - cached_len
            if cached_start < 0:
                raise ValueError(
                    "KV cache contains more keys than its absolute position "
                    f"allows: keys={cached_len}, next_position={cache_offset}"
                )
            k_idx = torch.arange(k_len, device=q.device) + cached_start
            allow = k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)
            float_mask = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)
            float_mask = float_mask.masked_fill(~allow, float("-inf"))
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=float_mask, dropout_p=drop, is_causal=False
            )

        out = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out(out), k, v


class TemporalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        use_rope: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TemporalSelfAttention(d_model, n_heads, dropout, use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        *,
        rope: Optional[RotaryEmbedding],
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
        cache_offset: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        h = self.norm1(x)
        attn_out, new_k, new_v = self.attn(
            h,
            rope=rope,
            cache_k=cache_k,
            cache_v=cache_v,
            cache_offset=cache_offset,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_k, new_v


class OptionCrossDecoderLayer(nn.Module):
    """Option queries cross-attend to spatial memory (+ optional state token)."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        options: Tensor,
        memory: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.norm_q(options)
        kv = self.norm_kv(memory)
        y, _ = self.cross(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        options = options + y
        options = options + self.ff(self.norm_ff(options))
        return options


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TemporalCabtTransformer(nn.Module):
    """Spatial + realized-history temporal CABT policy/value network.

    Forward contract::

        encode(obs_t, kv_cache) -> state_vec, spatial_memory, kv_cache'
        value = value_head(state_vec)
        policy_logits = option_decoder(options, spatial_memory, state_vec)

    Inputs are SparseVectors (EmbeddingBag) from :mod:`poke_bot.features`.
    """

    def __init__(
        self,
        cfg: Optional[config.ModelConfig] = None,
        *,
        encoder_vocab: Optional[int] = None,
        decoder_vocab: Optional[int] = None,
        num_board_tokens: int = features.NUM_BOARD_TOKENS,
        aux_archetype_classes: int = 16,
        belief_card_vocab: Optional[int] = None,
    ):
        super().__init__()
        cfg = cfg or config.MODEL
        decision_context = str(cfg.decision_context).lower()
        if decision_context not in {"history", "stateless"}:
            raise ValueError(
                "decision_context must be 'history' or legacy 'stateless', got "
                f"{cfg.decision_context!r}"
            )
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.num_board_tokens = num_board_tokens
        self.max_context = cfg.max_context
        self.use_rope = cfg.temporal_pos.lower() == "rope"
        self.decision_context = decision_context
        self.kv_cache_enabled = bool(cfg.kv_cache and decision_context == "history")
        self.dense_card2vec = bool(getattr(cfg, "dense_card2vec", False))
        self.register_buffer(
            "_feature_schema_version",
            torch.tensor(features.FEATURE_SCHEMA_VERSION, dtype=torch.int32),
            persistent=True,
        )

        enc_vocab = encoder_vocab if encoder_vocab is not None else features.encoder_vocab_size()
        dec_vocab = decoder_vocab if decoder_vocab is not None else features.decoder_vocab_size()
        self.encoder_vocab = enc_vocab
        self.decoder_vocab = dec_vocab
        self.decoder_binding_offset = features.decoder_binding_offset()
        card_vocab = (
            int(belief_card_vocab)
            if belief_card_vocab is not None
            else int(features.card_vocab_size())
        )
        if card_vocab <= 0:
            raise ValueError(f"belief_card_vocab must be positive, got {card_vocab}")
        self.belief_card_vocab = card_vocab
        attack_vocab = int(features.attack_vocab_size())

        # Spatial board tokens: flat EmbeddingBag OR factorized card2vec (Option A).
        # PackedSparse always includes a trailing nnz sentinel → include_last_offset.
        if self.dense_card2vec:
            from .card2vec import FactorizedCard2Vec

            d_card = int(getattr(cfg, "card_embed_dim", cfg.d_model) or cfg.d_model)
            self.card2vec = FactorizedCard2Vec(
                card_vocab=card_vocab,
                attack_vocab=attack_vocab,
                encoder_vocab=enc_vocab,
                decoder_vocab=dec_vocab,
                d_card=d_card,
                d_model=cfg.d_model,
            )
            self.board_bag = None
            self.option_bag = None
        else:
            self.card2vec = None
            self.board_bag = nn.EmbeddingBag(
                enc_vocab, cfg.d_model, mode="sum", include_last_offset=True
            )
            self.option_bag = nn.EmbeddingBag(
                dec_vocab, cfg.d_model, mode="sum", include_last_offset=True
            )
        # Explicit token identity is required because bench card-content spans
        # are shared by design.  Without this embedding, the spatial encoder is
        # permutation-equivariant and cannot tell bench slot 0 from slot 7.
        self.spatial_slot_embedding = nn.Embedding(num_board_tokens, cfg.d_model)

        # Factorized card2vec maps non-card feature rows to a shared null role.
        # Give every composite binding tuple an exact low-rank row, then project
        # it to model width.  This preserves tuple association without adding a
        # 10k × d_model table to the lean dense-card model.
        if self.dense_card2vec:
            binding_dim = min(16, cfg.d_model)
            self.option_binding_embedding = nn.Embedding(
                features.DECODER_BINDING_VOCAB_SIZE, binding_dim
            )
            self.option_binding_projection = nn.Linear(
                binding_dim, cfg.d_model, bias=False
            )
        else:
            # Flat option_bag already owns one independent d_model row per
            # composite tuple; adding the low-rank residual would duplicate it.
            self.option_binding_embedding = None
            self.option_binding_projection = None
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=cfg.spatial_layers, enable_nested_tensor=False
        )
        self.spatial_norm = nn.LayerNorm(cfg.d_model)
        self.cls_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Temporal tower
        self.rope = (
            RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq=cfg.max_context)
            if self.use_rope
            else None
        )
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalBlock(
                    cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout, self.use_rope
                )
                for _ in range(cfg.temporal_layers)
            ]
        )
        self.temporal_norm = nn.LayerNorm(cfg.d_model)
        if not self.use_rope:
            self.learned_pos = nn.Embedding(cfg.max_context, cfg.d_model)
        else:
            self.learned_pos = None

        # Option decoder layers (bag / card2vec already constructed above).
        if not self.dense_card2vec and self.option_bag is None:
            self.option_bag = nn.EmbeddingBag(
                dec_vocab, cfg.d_model, mode="sum", include_last_offset=True
            )
        self.option_decoder = nn.ModuleList(
            [
                OptionCrossDecoderLayer(
                    cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout
                )
                for _ in range(cfg.option_decoder_layers)
            ]
        )
        self.policy_head = nn.Linear(cfg.d_model, 1)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        # Aux: distinct named heads (never grow aux_head into a kitchen sink).
        # Scope A (core): aux_head archetype CE; opp_* multilabel belief priors.
        # Scope B (Blackwell Hammer): lethal_threat + prize_race strategy heads.
        # Strategy heads are architecture-present for warm-start, but training
        # weights stay 0 on core_kernel (see poke_bot.blackwell_heads).
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, aux_archetype_classes),
        )
        self.opp_hand_head = nn.Linear(cfg.d_model, self.belief_card_vocab)
        self.opp_remainder_head = nn.Linear(cfg.d_model, self.belief_card_vocab)
        # Scope B: P(take prize soon) + [own/6, opp/6] prize-race scaffold.
        self.lethal_threat_head = nn.Linear(cfg.d_model, 1)
        self.prize_race_head = nn.Linear(cfg.d_model, 2)
        self.aux_heads_present: tuple[str, ...] = (
            "aux_head",
            "opp_hand_head",
            "opp_remainder_head",
            "lethal_threat_head",
            "prize_race_head",
        )
        # Populated by warm-start load when new head keys were missing in ckpt.
        self.warm_started_belief_heads: tuple[str, ...] = ()

        # Dormant policy/value-only residuals.  Construct this after every base
        # module so adding the bank cannot perturb fresh base initialization via
        # RNG consumption.  Its parameters are opt-in trainable; the dedicated
        # bootstrap path below the model explicitly re-enables only this bank.
        adapter_format = str(
            getattr(cfg, "matchup_adapter_format", MATCHUP_ADAPTER_V5_FORMAT)
        )
        if adapter_format == MATCHUP_ADAPTER_V5_FORMAT:
            adapter_bank_type = MatchupAdapterBank
            adapter_bank_kwargs = {}
        elif adapter_format == MATCHUP_ADAPTER_V6_FORMAT:
            adapter_bank_type = MatchupAdapterBankV6
            serialized_registry = getattr(cfg, "matchup_adapter_registry", None)
            if not isinstance(serialized_registry, dict):
                raise ValueError(
                    "V6 model config requires an immutable matchup adapter registry"
                )
            adapter_bank_kwargs = {"registry": serialized_registry}
        else:
            raise ValueError(
                f"unsupported matchup adapter checkpoint format: {adapter_format}"
            )
        self.matchup_adapter_bank = adapter_bank_type(
            enabled=bool(getattr(cfg, "matchup_adapters_enabled", False)),
            **adapter_bank_kwargs,
        )
        self.matchup_adapter_bank.requires_grad_(False)

        # Expanded strategic heads are constructed last and under a forked,
        # fixed RNG.  Consequently:
        #   * enabling them cannot perturb initialization of any V5 tensor;
        #   * missing-head warm starts are reproducible across processes; and
        #   * construction does not advance the caller's global RNG stream.
        self.expanded_heads_enabled = bool(
            getattr(cfg, "expanded_heads_enabled", False)
        )
        self.expanded_head_schema_version = (
            EXPANDED_HEAD_SCHEMA_VERSION if self.expanded_heads_enabled else 0
        )
        if self.expanded_heads_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(EXPANDED_HEAD_INIT_SEED)
                for name, _source, outputs in EXPANDED_HEAD_SPECS:
                    setattr(self, name, nn.Linear(cfg.d_model, outputs))
            self.aux_heads_present = (
                *self.aux_heads_present,
                *EXPANDED_HEAD_NAMES,
            )
        else:
            # Explicit None attributes make feature detection predictable
            # without registering parameters in an immutable V5 state dict.
            for name in EXPANDED_HEAD_NAMES:
                setattr(self, name, None)
        self.warm_started_expanded_heads: tuple[str, ...] = ()
        self.setup_board_outcome_head_enabled = bool(
            getattr(cfg, "setup_board_outcome_head_enabled", False)
        )
        self.decision_fusion_dedicated_routes_enabled = bool(
            getattr(cfg, "decision_fusion_dedicated_routes_enabled", False)
        )
        self.decision_fusion_dedicated_routes_runtime_enabled = bool(
            getattr(
                cfg,
                "decision_fusion_dedicated_routes_runtime_enabled",
                False,
            )
        )
        if (
            self.setup_board_outcome_head_enabled
            and not self.expanded_heads_enabled
        ):
            raise ValueError(
                "setup board outcome head requires expanded strategic heads"
            )
        if (
            self.setup_board_outcome_head_enabled
            and not self.decision_fusion_dedicated_routes_enabled
        ):
            raise ValueError(
                "setup board outcome head requires dedicated decision-fusion routes"
            )
        if self.setup_board_outcome_head_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(EXPANDED_HEAD_INIT_SEED + 2)
                self.setup_board_outcome_head = SetupBoardOutcomeHead(
                    cfg.d_model
                )
            self.aux_heads_present = (
                *self.aux_heads_present,
                SETUP_BOARD_OUTCOME_HEAD_NAME,
            )
        else:
            self.setup_board_outcome_head = None
        self.decision_fusion_enabled = bool(
            getattr(cfg, "decision_fusion_enabled", False)
        )
        self.decision_fusion_runtime_enabled = bool(
            getattr(cfg, "decision_fusion_runtime_enabled", False)
        )
        if self.decision_fusion_enabled and not self.expanded_heads_enabled:
            raise ValueError(
                "causal decision fusion requires expanded strategic heads"
            )
        if (
            self.decision_fusion_runtime_enabled
            and not self.decision_fusion_enabled
        ):
            raise ValueError(
                "decision_fusion_runtime_enabled requires decision_fusion_enabled"
            )
        if (
            self.decision_fusion_dedicated_routes_enabled
            and not self.decision_fusion_enabled
        ):
            raise ValueError(
                "dedicated decision-fusion routes require decision_fusion_enabled"
            )
        if (
            self.decision_fusion_dedicated_routes_runtime_enabled
            and not self.decision_fusion_dedicated_routes_enabled
        ):
            raise ValueError(
                "dedicated route runtime activation requires route tensors"
            )
        if (
            self.decision_fusion_dedicated_routes_runtime_enabled
            and not self.decision_fusion_runtime_enabled
        ):
            raise ValueError(
                "dedicated route runtime activation requires decision fusion runtime"
            )
        if self.decision_fusion_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(EXPANDED_HEAD_INIT_SEED + 1)
                self.decision_fusion = CausalDecisionFusion(
                    d_model=cfg.d_model,
                    width=int(getattr(cfg, "decision_fusion_width", 16)),
                    archetype_classes=aux_archetype_classes,
                    belief_card_vocab=self.belief_card_vocab,
                    dedicated_routes_enabled=(
                        self.decision_fusion_dedicated_routes_enabled
                    ),
                    setup_board_outcome_outputs=(
                        SETUP_BOARD_OUTCOME_HEAD_OUTPUTS
                        if self.setup_board_outcome_head_enabled
                        else 0
                    ),
                )
        else:
            self.decision_fusion = None
        self.warm_started_decision_fusion = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Reject feature-incompatible checkpoints even under ``strict=False``."""
        schema_key = prefix + "_feature_schema_version"
        saved_schema = state_dict.get(schema_key)
        expected = int(features.FEATURE_SCHEMA_VERSION)
        if saved_schema is None:
            error_msgs.append(
                "checkpoint predates explicit slot/option-binding features "
                f"(required feature schema v{expected}); retrain or migrate it"
            )
        else:
            try:
                actual = int(saved_schema.detach().cpu().item())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                actual = -1
            if actual != expected:
                error_msgs.append(
                    "checkpoint feature schema mismatch: "
                    f"saved v{actual}, runtime requires v{expected}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def matchup_policy_value_state(
        self,
        state_vec: Tensor,
        routes: Optional[Union[Tensor, Sequence[int]]] = None,
        *,
        enabled: Optional[bool] = None,
    ) -> Tensor:
        """Return policy/value state after optional exact-match adaptation.

        ``routes=None`` is an unconditional exact bypass.  The archetype and
        other auxiliary heads intentionally never call this helper.
        """

        if routes is None:
            return state_vec
        return self.matchup_adapter_bank(state_vec, routes, enabled=enabled)

    def belief_aux_logits(self, state_vec: Tensor) -> dict[str, Tensor]:
        """Info-set belief / strategy logits from ``state_vec`` (root-only).

        Scope A belief priors always returned. Scope B strategy logits are also
        returned from the modules; callers must gate *usage* via
        ``blackwell_heads.blackwell_strategy_heads_enabled`` (core must not
        depend on them). Never write any of these into board bags.
        """
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        return {
            "aux_logits": self.aux_head(state_vec),
            "opp_hand_logits": self.opp_hand_head(state_vec),
            "opp_remainder_logits": self.opp_remainder_head(state_vec),
            "lethal_threat_logits": self.lethal_threat_head(state_vec).squeeze(-1),
            "prize_race_pred": self.prize_race_head(state_vec),
        }

    def expanded_head_inventory(self) -> dict[str, object]:
        """Return the checkpoint-safe expanded-head tensor contract.

        The inventory describes architecture only. Training receipts own loss
        weights, target coverage, and whether a head has received an optimizer
        step; the model must not infer those facts merely from non-zero bytes.
        """
        modules: dict[str, object] = {}
        if self.expanded_heads_enabled:
            for name, source, outputs in EXPANDED_HEAD_SPECS:
                module = getattr(self, name)
                if not isinstance(module, nn.Linear):
                    raise RuntimeError(f"expanded head {name} is not Linear")
                tensors = {
                    tensor_name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "numel": int(tensor.numel()),
                    }
                    for tensor_name, tensor in module.state_dict().items()
                }
                modules[name] = {
                    "input": source,
                    "outputs": int(outputs),
                    "fusion_role": "fused_input",
                    **(
                        {
                            "computation_role": "independent_head",
                            "action_influence": (
                                "bounded_option_conditioned_route"
                            ),
                            "action_route_causal_input": (
                                "current_board_state_and_legal_options_only"
                            ),
                            "direct_action_selection_authority": False,
                            "runtime_activation": "receipt_gated",
                        }
                        if self.decision_fusion_dedicated_routes_enabled
                        else {}
                    ),
                    "parameters": int(
                        sum(parameter.numel() for parameter in module.parameters())
                    ),
                    "tensors": tensors,
                }
        if isinstance(self.setup_board_outcome_head, SetupBoardOutcomeHead):
            modules[SETUP_BOARD_OUTCOME_HEAD_NAME] = (
                self.setup_board_outcome_head.inventory()
            )
        runtime_enabled_heads = (
            list(modules)
            if self.decision_fusion_dedicated_routes_runtime_enabled
            else []
        )
        return {
            "schema": EXPANDED_HEAD_SCHEMA,
            "version": int(self.expanded_head_schema_version),
            "enabled": bool(self.expanded_heads_enabled),
            "runtime_enabled_heads": runtime_enabled_heads,
            "modules": modules,
            "fusion_roles": {
                **{name: "fused_input" for name in modules},
            },
        }

    def decision_fusion_inventory(self) -> dict[str, object]:
        if not isinstance(self.decision_fusion, CausalDecisionFusion):
            return {
                "schema": DECISION_FUSION_SCHEMA,
                "enabled": False,
                "runtime_enabled": False,
                "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
                "parameters": 0,
            }
        return self.decision_fusion.inventory(
            runtime_enabled=self.decision_fusion_runtime_enabled,
            dedicated_routes_runtime_enabled=(
                self.decision_fusion_dedicated_routes_runtime_enabled
            ),
        )

    def fused_policy_logits(
        self,
        option_hidden: Tensor,
        state_vec: Tensor,
        base_logits: Tensor,
    ) -> Tensor:
        """Apply the all-head residual in training or after runtime activation.

        A boundary-migrated active learner first receives zero-safe fusion
        tensors with serving disabled.  Its next ordinary full-model update
        must nevertheless train those tensors; otherwise no nonzero influence
        receipt could exist before serving activation.  Evaluation/inference
        remains an exact flat-policy bypass until the separately receipted
        runtime flag is enabled.
        """

        fusion_training = bool(self.training and self.decision_fusion_enabled)
        if not self.decision_fusion_runtime_enabled and not fusion_training:
            return base_logits
        if not isinstance(self.decision_fusion, CausalDecisionFusion):
            raise RuntimeError(
                "decision fusion policy path is enabled without fusion tensors"
            )
        belief = self.belief_aux_logits(state_vec)
        expanded_state = self.expanded_state_logits(state_vec)
        expanded_option = self.expanded_option_logits(option_hidden)
        if self.setup_board_outcome_head_enabled:
            expanded_option["setup_board_outcome"] = (
                self.setup_board_outcome_logits(option_hidden)
            )
        state_sources = {
            "value": torch.tanh(self.value_head(state_vec)),
            "archetype": belief["aux_logits"],
            "opponent_hand": belief["opp_hand_logits"],
            "opponent_remainder": belief["opp_remainder_logits"],
            "lethal_threat": belief["lethal_threat_logits"],
            "prize_race": belief["prize_race_pred"],
            "tactical_outcomes": expanded_state["tactical_outcome"],
            "opponent_response": expanded_state["opponent_response"],
            "resource_forecast": expanded_state["resource_forecast"],
            "game_phase": expanded_state["game_phase"],
            "outcome_distribution": expanded_state["outcome_distribution"],
            "remaining_turns": expanded_state["remaining_turns"],
        }
        return self.decision_fusion(
            option_hidden,
            base_logits,
            state_sources=state_sources,
            option_sources=expanded_option,
            dedicated_routes_active=(
                self.decision_fusion_dedicated_routes_enabled
                and (
                    self.training
                    or self.decision_fusion_dedicated_routes_runtime_enabled
                )
            ),
        )

    def expanded_option_logits(self, option_hidden: Tensor) -> dict[str, Tensor]:
        """Evaluate all option-conditioned V6 heads on shared decoder states.

        ``option_hidden`` is the second value returned by
        :meth:`decode_options` / :meth:`decode_options_packed` when
        ``return_hidden=True``. Padding is intentionally not transformed here;
        the caller already owns the exact ``n_options`` vector and must apply
        target-specific masks.
        """
        if not self.expanded_heads_enabled:
            raise RuntimeError("expanded strategic heads are disabled")
        if option_hidden.size(-1) != self.d_model:
            raise ValueError(
                "option hidden width mismatch: "
                f"got={option_hidden.size(-1)} expected={self.d_model}"
            )
        outputs: dict[str, Tensor] = {}
        for key, name in (
            ("action_q", "action_q_head"),
            ("action_type", "action_type_head"),
            ("action_target", "action_target_head"),
            ("action_resource", "action_resource_head"),
            ("action_utility", "action_utility_head"),
        ):
            module = getattr(self, name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"expanded head {name} is unavailable")
            value = module(option_hidden)
            outputs[key] = value.squeeze(-1) if value.size(-1) == 1 else value
        return outputs

    def setup_board_outcome_logits(self, option_hidden: Tensor) -> Tensor:
        """Evaluate the future-only independent setup prediction branch."""

        module = self.setup_board_outcome_head
        if not isinstance(module, SetupBoardOutcomeHead):
            raise RuntimeError("setup board outcome head is disabled")
        if option_hidden.size(-1) != self.d_model:
            raise ValueError(
                "setup board option hidden width mismatch: "
                f"got={option_hidden.size(-1)} expected={self.d_model}"
            )
        return module(option_hidden)

    def expanded_state_logits(self, state_vec: Tensor) -> dict[str, Tensor]:
        """Evaluate state-conditioned V6 strategic auxiliary heads.

        Tactical output is grouped as three same-seat horizons × six targets.
        All values are raw logits/regressions: the typed loss layer decides
        which columns receive BCE, Smooth-L1, CE, or a positive transform.
        """
        if not self.expanded_heads_enabled:
            raise RuntimeError("expanded strategic heads are disabled")
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        if state_vec.size(-1) != self.d_model:
            raise ValueError(
                "state hidden width mismatch: "
                f"got={state_vec.size(-1)} expected={self.d_model}"
            )

        def run(name: str) -> Tensor:
            module = getattr(self, name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"expanded head {name} is unavailable")
            return module(state_vec)

        tactical = run("tactical_outcome_head")
        return {
            "tactical_outcome": tactical.reshape(
                *tactical.shape[:-1], 3, 6
            ),
            "opponent_response": run("opponent_response_head"),
            "resource_forecast": run("resource_forecast_head"),
            "game_phase": run("game_phase_head"),
            "outcome_distribution": run("outcome_distribution_head"),
            "remaining_turns": run("remaining_turns_head"),
        }

    # ----- encode primitives -----

    def _embed_board_bag(
        self, indices: Tensor, offsets: Tensor, values: Tensor
    ) -> Tensor:
        if self.dense_card2vec:
            assert self.card2vec is not None
            return self.card2vec.embed_board(indices, offsets, values)
        assert self.board_bag is not None
        return self.board_bag(indices, offsets, values)

    def _embed_option_bag(
        self, indices: Tensor, offsets: Tensor, values: Tensor
    ) -> Tensor:
        if self.dense_card2vec:
            assert self.card2vec is not None
            embedded = self.card2vec.embed_option(indices, offsets, values)
        else:
            assert self.option_bag is not None
            return self.option_bag(indices, offsets, values)

        # Binding rows are appended after the v4 layout.  Card2vec intentionally
        # factorizes card identity, so supplement it with an exact row embedding
        # (the flat path receives the same structured residual).
        relative = indices - int(self.decoder_binding_offset)
        binding_mask = (
            (relative >= 0)
            & (relative < features.DECODER_BINDING_VOCAB_SIZE)
        )
        counts = offsets[1:] - offsets[:-1]
        bag_ids = torch.repeat_interleave(
            torch.arange(
                counts.numel(), device=indices.device, dtype=torch.long
            ),
            counts,
        )
        assert self.option_binding_embedding is not None
        assert self.option_binding_projection is not None
        binding_vecs = self.option_binding_projection(
            self.option_binding_embedding(relative[binding_mask])
        )
        binding_vecs = binding_vecs * values[binding_mask].unsqueeze(-1).to(
            dtype=binding_vecs.dtype
        )
        binding_vecs = binding_vecs.to(dtype=embedded.dtype)
        binding_sum = embedded.new_zeros(embedded.shape)
        binding_sum.index_add_(0, bag_ids[binding_mask], binding_vecs)
        return embedded + binding_sum

    @staticmethod
    def _validate_sparse_indices(indices: Tensor, vocab: int, label: str) -> None:
        """Fail closed instead of aliasing out-of-schema rows via ``clamp``."""
        if indices.numel() == 0:
            return
        lo = int(indices.min().item())
        hi = int(indices.max().item())
        if lo < 0 or hi >= int(vocab):
            raise ValueError(
                f"{label} feature index range [{lo}, {hi}] exceeds checkpoint "
                f"vocab [0, {int(vocab) - 1}] under feature schema "
                f"v{features.FEATURE_SCHEMA_VERSION}; use a compatible checkpoint"
            )

    def encode_board(self, board_svs: Union[SparseVector, Sequence[SparseVector]]) -> Tensor:
        """Encode board SparseVector(s) → spatial memory ``[B, 24, D]``."""
        if isinstance(board_svs, SparseVector):
            board_svs = [board_svs]
        device = next(self.parameters()).device
        packed = pack_sparse_batch(board_svs, self.num_board_tokens, device)
        return self.encode_board_packed(packed, batch_size=len(board_svs))

    def encode_board_packed(
        self,
        packed: PackedSparse,
        *,
        batch_size: int,
    ) -> Tensor:
        """Encode a device-resident packed board batch without host repacking."""
        self._validate_sparse_indices(packed.index, self.encoder_vocab, "board")
        tokens = self._embed_board_bag(
            packed.index, packed.offset, packed.value
        )
        b = int(batch_size)
        if packed.offset.numel() != b * self.num_board_tokens + 1:
            raise ValueError("packed board offset count does not match batch size")
        tokens = tokens.view(b, self.num_board_tokens, self.d_model)
        device = next(self.parameters()).device
        slot_ids = torch.arange(self.num_board_tokens, device=device)
        tokens = tokens + self.spatial_slot_embedding(slot_ids).unsqueeze(0)
        return self.spatial_norm(self.spatial_encoder(tokens))

    def pool_cls(self, spatial_memory: Tensor) -> Tensor:
        """Pool spatial memory to a per-timestep CLS ``[B, D]``."""
        return self.cls_proj(spatial_memory.mean(dim=1))

    def encode_previous_actions(
        self,
        actions: Sequence[Optional[SparseVector]],
    ) -> Tensor:
        """Encode shifted, already-taken actions for temporal history tokens."""
        device = next(self.parameters()).device
        present = [(i, action) for i, action in enumerate(actions) if action is not None]
        if not present:
            return torch.zeros(
                len(actions), self.d_model, device=device, dtype=self._activation_dtype(device)
            )
        packed = pack_sparse_batch([action for _, action in present], 1, device)
        self._validate_sparse_indices(
            packed.index, self.decoder_vocab, "previous-action"
        )
        encoded = self._embed_option_bag(
            packed.index,
            packed.offset,
            packed.value,
        )
        # Under CUDA autocast, embeddings/Linear emit bf16 while a default
        # float32 zeros buffer makes index_copy_ raise — leaf servers then
        # FAIL-CLOSED to random legal (SPS collapse to ~tens).
        out = torch.zeros(
            len(actions), self.d_model, device=device, dtype=encoded.dtype
        )
        rows = torch.tensor([i for i, _ in present], dtype=torch.long, device=device)
        return out.index_copy(0, rows, encoded)

    def encode_previous_actions_packed(
        self,
        packed: PackedSparse,
        *,
        batch_size: int,
    ) -> Tensor:
        """Encode one already-shifted resident action row per timestep.

        Empty CSR rows are the causal ``None`` token used for the first
        decision in each game.  Keeping the shift in the corpus gather avoids
        reconstructing millions of :class:`SparseVector` objects on the host.
        """
        count = int(batch_size)
        if count < 0 or packed.offset.numel() != count + 1:
            raise ValueError(
                "packed previous-action offsets do not match batch size"
            )
        device = next(self.parameters()).device
        if packed.index.numel() == 0:
            return torch.zeros(
                count,
                self.d_model,
                device=device,
                dtype=self._activation_dtype(device),
            )
        self._validate_sparse_indices(
            packed.index, self.decoder_vocab, "previous-action"
        )
        return self._embed_option_bag(
            packed.index,
            packed.offset,
            packed.value,
        )

    def _activation_dtype(self, device: torch.device) -> torch.dtype:
        """Match autocast activation dtype so zero buffers don't fight bf16."""
        if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return next(self.parameters()).dtype

    def history_tokens(
        self,
        spatial_memory: Tensor,
        previous_actions: Optional[Sequence[Optional[SparseVector]]] = None,
    ) -> Tensor:
        """Fuse current observable board with the previous realized own action."""
        cls = self.pool_cls(spatial_memory)
        if previous_actions is None:
            return cls
        if len(previous_actions) != cls.size(0):
            raise ValueError("previous-action history length mismatch")
        return cls + float(self.cfg.history_action_scale) * self.encode_previous_actions(
            previous_actions
        )

    def temporal_encode(
        self,
        cls_tokens: Tensor,
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append: bool = True,
        return_all: bool = False,
        position_offset: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[TemporalKVCache]]:
        """Causal temporal encode.

        ``cls_tokens``: ``[B, T, D]`` (T=1 for incremental append).
        When ``kv_cache`` is provided and ``append`` is True, new tokens are
        appended to the cache. Returns ``(state_vec, new_cache)`` where
        ``state_vec`` is ``[B, D]`` (last token) or ``[B, T, D]`` if
        ``return_all`` (used by whole-game supervised training).

        ``position_offset`` is the absolute index of the first offline token.
        Incremental callers get this from ``TemporalKVCache.next_position``;
        it must remain zero when a cache is supplied.
        """
        b, t, _ = cls_tokens.shape
        if t <= 0:
            raise ValueError("temporal_encode requires at least one token")
        if key_padding_mask is not None:
            if kv_cache is not None:
                raise ValueError("temporal key padding cannot be combined with a KV cache")
            if tuple(key_padding_mask.shape) != (b, t):
                raise ValueError(
                    "temporal key padding mask shape mismatch: "
                    f"expected={(b, t)} actual={tuple(key_padding_mask.shape)}"
                )
        position_offset = int(position_offset)
        if position_offset < 0:
            raise ValueError(
                f"position_offset must be non-negative, got {position_offset}"
            )
        if kv_cache is not None and position_offset != 0:
            raise ValueError(
                "position_offset is derived from kv_cache during incremental encode"
            )
        if kv_cache is not None and t > self.max_context:
            raise ValueError(
                "incremental append exceeds max_context: "
                f"T={t}, max_context={self.max_context}"
            )
        if kv_cache is None and t > self.max_context:
            if return_all:
                raise ValueError(
                    "return_all temporal training input exceeds max_context: "
                    f"T={t}, max_context={self.max_context}; truncate boards, "
                    "actions, and targets together before encoding"
                )
            dropped = t - self.max_context
            cls_tokens = cls_tokens[:, dropped:, :]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, dropped:]
            t = self.max_context
            position_offset += dropped
        raw_cls_tokens = cls_tokens
        if kv_cache is not None:
            if len(kv_cache.layers) != len(self.temporal_blocks):
                raise ValueError(
                    "KV cache layer count mismatch: "
                    f"cache={len(kv_cache.layers)}, model={len(self.temporal_blocks)}"
                )
            for layer, (ck, cv) in enumerate(kv_cache.layers):
                if ck.size(2) != kv_cache.length or cv.size(2) != kv_cache.length:
                    raise ValueError(
                        f"KV cache layer {layer} length does not match metadata"
                    )
            cache_offset = kv_cache.resolved_next_position()
        else:
            cache_offset = position_offset

        # Higher-layer cached K/V retain information from tokens they attended
        # to earlier.  Once a full rolling window evicts its oldest token,
        # simply slicing K/V would therefore diverge from offline training's
        # freshly encoded suffix.  Raw CLS storage is tiny; recompute the exact
        # retained window only at/after rollover (rare for max_context=320).
        if (
            kv_cache is not None
            and kv_cache.length >= self.max_context
            and kv_cache.input_tokens is not None
            and kv_cache.input_tokens.size(1) == kv_cache.length
        ):
            prior_count = max(0, self.max_context - t)
            if prior_count == 0:
                window = raw_cls_tokens
            else:
                window = torch.cat(
                    [kv_cache.input_tokens[:, -prior_count:, :], raw_cls_tokens],
                    dim=1,
                )
            window_start = cache_offset - prior_count
            window_state, new_cache = self.temporal_encode(
                window,
                kv_cache=None,
                append=append,
                return_all=return_all,
                position_offset=window_start,
            )
            if return_all:
                window_state = window_state[:, -t:, :]
            return window_state, new_cache

        if not self.use_rope and self.learned_pos is not None:
            # Learned tables describe positions inside the retained window;
            # unlike RoPE they cannot represent an unbounded absolute offset.
            learned_offset = cache_offset if kv_cache is not None else 0
            positions = torch.arange(
                learned_offset, learned_offset + t, device=cls_tokens.device
            ).clamp(max=self.max_context - 1)
            cls_tokens = cls_tokens + self.learned_pos(positions).unsqueeze(0)

        new_layers: list[tuple[Tensor, Tensor]] = []
        x = cls_tokens
        for i, block in enumerate(self.temporal_blocks):
            ck = cv = None
            if kv_cache is not None:
                ck, cv = kv_cache.layers[i]
                # The current tokens count toward max_context.  Drop excess
                # prior keys *before* attention; trimming only the returned
                # cache would let a full cache expose max_context + T keys.
                max_prior = max(0, self.max_context - t)
                if ck.size(2) > max_prior:
                    if max_prior == 0:
                        ck = ck[..., :0, :]
                        cv = cv[..., :0, :]
                    else:
                        ck = ck[..., -max_prior:, :]
                        cv = cv[..., -max_prior:, :]
            x, nk, nv = block(
                x,
                rope=self.rope,
                cache_k=ck,
                cache_v=cv,
                cache_offset=cache_offset,
                key_padding_mask=key_padding_mask,
            )
            # Defensive cap for non-incremental/future multi-token callers.
            if nk.size(2) > self.max_context:
                nk = nk[:, :, -self.max_context :, :]
                nv = nv[:, :, -self.max_context :, :]
            new_layers.append((nk, nv))

        x = self.temporal_norm(x)
        state_vec = x if return_all else x[:, -1, :]

        new_cache = None
        if self.kv_cache_enabled and append:
            new_len = new_layers[0][0].size(2)
            if kv_cache is None:
                input_tokens = raw_cls_tokens
            elif (
                kv_cache.input_tokens is not None
                and kv_cache.input_tokens.size(1) == kv_cache.length
            ):
                input_tokens = torch.cat(
                    [kv_cache.input_tokens, raw_cls_tokens], dim=1
                )
            else:
                input_tokens = None
            if input_tokens is not None and input_tokens.size(1) > new_len:
                input_tokens = input_tokens[:, -new_len:, :]
            new_cache = TemporalKVCache(
                layers=new_layers,
                length=new_len,
                next_position=cache_offset + t,
                input_tokens=input_tokens,
            )
        return state_vec, new_cache

    def encode_history(
        self,
        board_history: Sequence[SparseVector],
        *,
        return_all: bool = False,
        previous_actions: Optional[Sequence[Optional[SparseVector]]] = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode one acting-seat history with no hidden/opponent-private input.

        Returns ``(state, spatial)``. ``state`` is ``[1, D]`` for the latest
        decision or ``[T, D]`` when ``return_all``; ``spatial`` is
        ``[T, board_tokens, D]``.
        """
        boards = list(board_history)
        if not boards:
            raise ValueError("history must contain at least one observation")
        position_offset = max(0, len(boards) - self.max_context)
        if len(boards) > self.max_context:
            boards = boards[-self.max_context :]
            if previous_actions is not None:
                previous_actions = list(previous_actions)[-self.max_context :]
        spatial = self.encode_board(boards)
        cls = self.history_tokens(spatial, previous_actions).unsqueeze(0)
        temporal, _ = self.temporal_encode(
            cls,
            kv_cache=None,
            append=False,
            return_all=return_all,
            position_offset=position_offset,
        )
        if return_all:
            return temporal.squeeze(0), spatial
        return temporal, spatial

    def forward_history_batch(
        self,
        board_histories: Sequence[Sequence[SparseVector]],
        options: Union[SparseVector, Sequence[SparseVector]],
        *,
        n_options: Optional[Sequence[int]] = None,
        previous_action_histories: Optional[
            Sequence[Sequence[Optional[SparseVector]]]
        ] = None,
        matchup_routes: Optional[Union[Tensor, Sequence[int]]] = None,
    ) -> dict[str, Union[Tensor, Optional[TemporalKVCache]]]:
        """Evaluate variable-length realized histories.

        Spatial encoding and option decoding are batched across games. Temporal
        encoding is grouped logically per game so padding can never become
        observable history.
        """
        raw_histories = [list(h) for h in board_histories]
        position_offsets = [
            max(0, len(history) - self.max_context)
            for history in raw_histories
        ]
        histories = [history[-self.max_context :] for history in raw_histories]
        if not histories or any(not h for h in histories):
            raise ValueError("every history must contain at least one observation")
        if isinstance(options, SparseVector):
            options = [options]
        options = list(options)
        if len(options) != len(histories):
            raise ValueError("history/options batch size mismatch")
        if n_options is None:
            n_options = [sv.num_words for sv in options]

        lengths = [len(h) for h in histories]
        if previous_action_histories is None:
            action_histories: list[list[Optional[SparseVector]]] = [
                [None] * length for length in lengths
            ]
        else:
            action_histories = [
                list(actions)[-self.max_context :]
                for actions in previous_action_histories
            ]
            if [len(actions) for actions in action_histories] != lengths:
                raise ValueError("board/action history lengths do not match")
        flat_boards = [board for history in histories for board in history]
        flat_spatial = self.encode_board(flat_boards)
        states: list[Tensor] = []
        current_spatial: list[Tensor] = []
        start = 0
        for length, position_offset, previous_actions in zip(
            lengths, position_offsets, action_histories
        ):
            spatial = flat_spatial[start : start + length]
            cls = self.history_tokens(spatial, previous_actions).unsqueeze(0)
            state, _ = self.temporal_encode(
                cls, append=False, position_offset=position_offset
            )
            states.append(state.squeeze(0))
            current_spatial.append(spatial[-1])
            start += length
        state_vec = torch.stack(states, dim=0)
        spatial_memory = torch.stack(current_spatial, dim=0)
        policy_value_state = self.matchup_policy_value_state(
            state_vec, matchup_routes
        )
        logits = self.decode_options(
            options,
            spatial_memory,
            policy_value_state,
            n_options=n_options,
            decision_fusion_state_vec=state_vec,
        )
        out = {
            "policy_logits": logits,
            "value": torch.tanh(self.value_head(policy_value_state)).squeeze(-1),
            "state_vec": state_vec,
            "spatial_memory": spatial_memory,
            "kv_cache": None,
        }
        out.update(self.belief_aux_logits(state_vec))
        return out

    def decode_options(
        self,
        option_svs: Union[SparseVector, Sequence[SparseVector]],
        spatial_memory: Tensor,
        state_vec: Tensor,
        *,
        n_options: Optional[Sequence[int]] = None,
        return_hidden: bool = False,
        decision_fusion_state_vec: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Score option vectors, optionally returning shared decoder states.

        The default remains the historical logits-only Tensor API. With
        ``return_hidden=True`` the result is ``(policy_logits, option_hidden)``,
        where ``option_hidden`` is ``[B, max_N, d_model]`` and can feed the
        expanded action heads without a second option-decoder pass.
        """
        if isinstance(option_svs, SparseVector):
            option_svs = [option_svs]
        device = next(self.parameters()).device
        if n_options is None:
            n_options = [sv.num_words for sv in option_svs]
        max_n = max(n_options) if n_options else 1
        max_n = max(max_n, 1)

        # Pack with empty-word padding so every sample has max_n option tokens.
        index: list[int] = []
        value: list[float] = []
        offset: list[int] = [0]
        for sv, n in zip(option_svs, n_options):
            for w in range(n):
                start = sv.offset[w]
                end = sv.offset[w + 1] if w + 1 < len(sv.offset) else len(sv.index)
                for j in range(start, end):
                    index.append(sv.index[j])
                    value.append(sv.value[j])
                offset.append(len(index))
            for _ in range(max_n - n):
                offset.append(len(index))  # empty pad word

        b = len(option_svs)
        if not index:
            opt_tokens = torch.zeros(
                b, max_n, self.d_model, device=device, dtype=self._activation_dtype(device)
            )
        else:
            idx_t = torch.tensor(index, dtype=torch.long, device=device)
            self._validate_sparse_indices(idx_t, self.decoder_vocab, "option")
            val_t = torch.tensor(value, dtype=torch.float32, device=device)
            off_t = torch.tensor(offset, dtype=torch.long, device=device)
            opt_tokens = self._embed_option_bag(idx_t, off_t, val_t).view(
                b, max_n, self.d_model
            )

        return self._decode_option_tokens(
            opt_tokens,
            spatial_memory,
            state_vec,
            n_options=n_options,
            return_hidden=return_hidden,
            decision_fusion_state_vec=decision_fusion_state_vec,
        )

    def decode_options_packed(
        self,
        packed: PackedSparse,
        spatial_memory: Tensor,
        state_vec: Tensor,
        *,
        n_options: Union[Sequence[int], Tensor],
        batch_size: int,
        return_hidden: bool = False,
        decision_fusion_state_vec: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Decode a packed option batch, optionally returning decoder states."""
        b = int(batch_size)
        if b <= 0:
            raise ValueError("packed option batch must be non-empty")
        words = int(packed.offset.numel()) - 1
        if words < 0 or words % b:
            raise ValueError("packed option offsets do not form a rectangular batch")
        max_n = max(1, words // b)
        if packed.index.numel() == 0:
            opt_tokens = torch.zeros(
                b,
                max_n,
                self.d_model,
                device=spatial_memory.device,
                dtype=self._activation_dtype(spatial_memory.device),
            )
        else:
            self._validate_sparse_indices(
                packed.index, self.decoder_vocab, "option"
            )
            opt_tokens = self._embed_option_bag(
                packed.index,
                packed.offset,
                packed.value,
            ).view(b, max_n, self.d_model)
        return self._decode_option_tokens(
            opt_tokens,
            spatial_memory,
            state_vec,
            n_options=n_options,
            return_hidden=return_hidden,
            decision_fusion_state_vec=decision_fusion_state_vec,
        )

    def _decode_option_tokens(
        self,
        opt_tokens: Tensor,
        spatial_memory: Tensor,
        state_vec: Tensor,
        *,
        n_options: Union[Sequence[int], Tensor],
        return_hidden: bool = False,
        decision_fusion_state_vec: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Shared option decoder after sparse bags have already been embedded."""
        device = opt_tokens.device
        b, max_n, _ = opt_tokens.shape

        # Memory = spatial tokens + state as an extra key.
        # Normalize shapes so both the batched inference route (state_vec is
        # [B, d_model], spatial_memory is [B, T, d_model]) and the per-decision
        # training route (state_vec is [d_model], spatial_memory is [1, T,
        # d_model]) yield a 3-D state token [B, 1, d_model] that concatenates
        # with spatial memory along the token axis.
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)  # [d_model] -> [1, d_model]
        if spatial_memory.dim() == 2:
            spatial_memory = spatial_memory.unsqueeze(0)  # [T, D] -> [1, T, D]
        state_tok = state_vec.unsqueeze(1)  # [B, d_model] -> [B, 1, d_model]
        assert spatial_memory.dim() == 3, (
            "decode_options: spatial_memory must be [B, T, d_model], got "
            f"{tuple(spatial_memory.shape)}"
        )
        assert state_tok.dim() == 3 and state_tok.size(-1) == spatial_memory.size(-1), (
            "decode_options: state token {} incompatible with spatial memory {} "
            "for cat(dim=1)".format(tuple(state_tok.shape), tuple(spatial_memory.shape))
        )
        assert state_tok.size(0) == spatial_memory.size(0) == opt_tokens.size(0), (
            "decode_options: batch mismatch — options={}, spatial={}, state={}".format(
                opt_tokens.size(0), spatial_memory.size(0), state_tok.size(0)
            )
        )
        memory = torch.cat([spatial_memory, state_tok], dim=1)  # [B, T+1, d_model]

        h = opt_tokens
        for layer in self.option_decoder:
            h = layer(h, memory)
        logits = self.policy_head(h).squeeze(-1)  # [B, max_N]
        fusion_state = (
            state_vec
            if decision_fusion_state_vec is None
            else decision_fusion_state_vec
        )
        logits = self.fused_policy_logits(h, fusion_state, logits)

        # Mask padded options.
        counts = torch.as_tensor(n_options, device=device, dtype=torch.long).reshape(-1)
        if counts.numel() != b:
            raise ValueError("option count vector does not match batch size")
        padding = torch.arange(max_n, device=device).unsqueeze(0) >= counts.unsqueeze(1)
        policy_logits = logits.masked_fill(padding, float("-inf"))
        if return_hidden:
            return policy_logits, h
        return policy_logits

    # ----- high-level API -----

    def forward(
        self,
        board: Union[SparseVector, Sequence[SparseVector]],
        options: Union[SparseVector, Sequence[SparseVector]],
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append_cache: bool = False,
        n_options: Optional[Sequence[int]] = None,
        previous_action: Optional[SparseVector] = None,
        matchup_routes: Optional[Union[Tensor, Sequence[int]]] = None,
    ) -> dict[str, Union[Tensor, Optional[TemporalKVCache]]]:
        """Evaluate one decision per batch row, optionally appending KV history.

        Returns dict with ``policy_logits``, ``value``, ``aux_logits``,
        ``opp_hand_logits``, ``opp_remainder_logits``,
        ``lethal_threat_logits``, ``prize_race_pred``, ``state_vec``,
        ``spatial_memory``, ``kv_cache``.
        """
        if self.decision_context == "stateless" and (kv_cache is not None or append_cache):
            raise ValueError(
                "legacy stateless decision contract forbids KV cache input/append; "
                "pass kv_cache=None and append_cache=False"
            )
        if isinstance(board, SparseVector):
            board = [board]
        if isinstance(options, SparseVector):
            options = [options]
        if kv_cache is not None and len(board) != 1:
            raise ValueError("incremental KV inference supports batch size one")
        if n_options is None:
            n_options = [sv.num_words for sv in options]

        spatial = self.encode_board(board)
        if previous_action is not None and len(board) != 1:
            raise ValueError("previous_action incremental input requires batch size one")
        cls = self.history_tokens(
            spatial,
            [previous_action] if previous_action is not None else None,
        ).unsqueeze(1)
        state_vec, new_cache = self.temporal_encode(
            cls, kv_cache, append=append_cache
        )
        policy_value_state = self.matchup_policy_value_state(
            state_vec, matchup_routes
        )
        logits = self.decode_options(
            options,
            spatial,
            policy_value_state,
            n_options=n_options,
            decision_fusion_state_vec=state_vec,
        )
        value = torch.tanh(self.value_head(policy_value_state)).squeeze(-1)
        out: dict[str, Union[Tensor, Optional[TemporalKVCache]]] = {
            "policy_logits": logits,
            "value": value,
            "state_vec": state_vec,
            "spatial_memory": spatial,
            "kv_cache": new_cache,
        }
        out.update(self.belief_aux_logits(state_vec))
        return out

    def forward_from_obs(
        self,
        obs,
        your_deck: list[int],
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append_cache: bool = False,
        assert_info: bool = True,
        previous_action: Optional[SparseVector] = None,
    ) -> dict[str, Union[Tensor, Optional[TemporalKVCache], list]]:
        """Featurize one observation and run incremental inference (info-set only)."""
        if assert_info:
            features.assert_info_set(obs)
        board = features.build_board_tokens(obs, your_deck)
        combos = features.enumerate_action_combos(obs)
        opt = features.build_option_tokens(obs, combos)
        out = self.forward(
            board,
            opt,
            kv_cache,
            append_cache=append_cache,
            previous_action=previous_action,
        )
        out["action_combos"] = combos
        return out

    def empty_cache(self) -> TemporalKVCache:
        """Return an empty KV cache (no layers filled yet — pass None instead).

        Callers should pass ``kv_cache=None`` for the first timestep; this helper
        exists for API symmetry.
        """
        return TemporalKVCache(layers=[], length=0, next_position=0)


def card_prior_logits_or_uniform(
    logits: Optional[Tensor],
    card_vocab: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Particle-prior logits; zeros ⇒ uniform softmax when head logits unavailable."""
    if logits is not None:
        return logits
    if card_vocab <= 0:
        raise ValueError(f"card_vocab must be positive, got {card_vocab}")
    return torch.zeros(int(card_vocab), device=device, dtype=dtype or torch.float32)


def build_model(
    cfg: Optional[config.ModelConfig] = None,
    *,
    device: Optional[torch.device] = None,
    aux_archetype_classes: Optional[int] = None,
    encoder_vocab: Optional[int] = None,
    decoder_vocab: Optional[int] = None,
    belief_card_vocab: Optional[int] = None,
) -> TemporalCabtTransformer:
    """Construct a :class:`TemporalCabtTransformer` on ``device`` (default CPU)."""
    from . import archetypes

    if aux_archetype_classes is None:
        # registered archetypes + unknown
        aux_archetype_classes = len(archetypes.archetype_ids()) + 1
    model = TemporalCabtTransformer(
        cfg,
        aux_archetype_classes=aux_archetype_classes,
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        belief_card_vocab=belief_card_vocab,
    )
    if device is not None:
        model = model.to(device)
    return model
