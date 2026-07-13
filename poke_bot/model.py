"""Temporal CABT transformer: spatial board + whole-game causal temporal + option decoder.

Design choice (board tokens)
----------------------------
``features.build_board_tokens`` / ``build_option_tokens`` return
:class:`~poke_bot.features.SparseVector` bags sized for ``nn.EmbeddingBag``.
This module therefore consumes those sparse bags directly via EmbeddingBag
encoders (same contract as the official RL+MCTS sample), then runs a dense
spatial Transformer over the resulting 24 board tokens. Dense projected card
embeddings are *not* used for the board path so we stay 1:1 with the foundation
feature builders; card-id density is already inside the bag features.

Architecture (v1)
-----------------
1. Spatial EmbeddingBag → 24 tokens → pre-norm TransformerEncoder.
2. Mean-pool spatial memory → per-timestep ``[CLS]`` for the temporal tower.
3. Causal temporal tower over the full game (``MAX_CONTEXT=320``), RoPE, optional
   incremental KV cache (append one CLS per realized decision).
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

Tensor = torch.Tensor


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
            self._build_cache(end)
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        return cos, sin


# ---------------------------------------------------------------------------
# Temporal layer with KV cache
# ---------------------------------------------------------------------------

@dataclass
class TemporalKVCache:
    """Per-layer cached (K, V) for incremental whole-game encode.

    Each tensor is ``[B, H, T, head_dim]``. ``length`` is the current T.
    """

    layers: list[tuple[Tensor, Tensor]]
    length: int = 0

    def clone(self) -> "TemporalKVCache":
        return TemporalKVCache(
            layers=[(k.clone(), v.clone()) for k, v in self.layers],
            length=self.length,
        )

    def truncate(self, length: int) -> "TemporalKVCache":
        if length >= self.length:
            return self.clone()
        return TemporalKVCache(
            layers=[(k[..., :length, :], v[..., :length, :]) for k, v in self.layers],
            length=length,
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
        if cache_k is None:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=True
            )
        else:
            # Append path: q_len may be << k_len; build an additive causal mask.
            q_len, k_len = q.size(2), k.size(2)
            q_idx = torch.arange(q_len, device=q.device) + cache_offset
            k_idx = torch.arange(k_len, device=q.device)
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        h = self.norm1(x)
        attn_out, new_k, new_v = self.attn(
            h, rope=rope, cache_k=cache_k, cache_v=cache_v, cache_offset=cache_offset
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
    """Spatial + whole-game temporal CABT policy/value network.

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
    ):
        super().__init__()
        cfg = cfg or config.MODEL
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.num_board_tokens = num_board_tokens
        self.max_context = cfg.max_context
        self.use_rope = cfg.temporal_pos.lower() == "rope"
        self.kv_cache_enabled = bool(cfg.kv_cache)

        enc_vocab = encoder_vocab if encoder_vocab is not None else features.encoder_vocab_size()
        dec_vocab = decoder_vocab if decoder_vocab is not None else features.decoder_vocab_size()
        self.encoder_vocab = enc_vocab
        self.decoder_vocab = dec_vocab

        # Spatial EmbeddingBag → board tokens
        # PackedSparse always includes a trailing nnz sentinel → include_last_offset.
        self.board_bag = nn.EmbeddingBag(
            enc_vocab, cfg.d_model, mode="sum", include_last_offset=True
        )
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
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq=cfg.max_context) if self.use_rope else None
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

        # Option decoder
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

        # Aux stub: info-set → opponent-hand/archetype prediction (labels separate).
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, aux_archetype_classes),
        )

    # ----- encode primitives -----

    def encode_board(self, board_svs: Union[SparseVector, Sequence[SparseVector]]) -> Tensor:
        """Encode board SparseVector(s) → spatial memory ``[B, 24, D]``."""
        if isinstance(board_svs, SparseVector):
            board_svs = [board_svs]
        device = next(self.parameters()).device
        packed = pack_sparse_batch(board_svs, self.num_board_tokens, device)
        # Clamp indices into vocab (safety for headroom mismatch).
        idx = packed.index.clamp(0, self.encoder_vocab - 1)
        tokens = self.board_bag(idx, packed.offset, packed.value)
        b = len(board_svs)
        tokens = tokens.view(b, self.num_board_tokens, self.d_model)
        return self.spatial_norm(self.spatial_encoder(tokens))

    def pool_cls(self, spatial_memory: Tensor) -> Tensor:
        """Pool spatial memory to a per-timestep CLS ``[B, D]``."""
        return self.cls_proj(spatial_memory.mean(dim=1))

    def temporal_encode(
        self,
        cls_tokens: Tensor,
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append: bool = True,
        return_all: bool = False,
    ) -> tuple[Tensor, Optional[TemporalKVCache]]:
        """Causal temporal encode.

        ``cls_tokens``: ``[B, T, D]`` (T=1 for incremental append).
        When ``kv_cache`` is provided and ``append`` is True, new tokens are
        appended to the cache. Returns ``(state_vec, new_cache)`` where
        ``state_vec`` is ``[B, D]`` (last token) or ``[B, T, D]`` if
        ``return_all`` (used by whole-game supervised training).
        """
        b, t, _ = cls_tokens.shape
        cache_offset = 0 if kv_cache is None else kv_cache.length

        if not self.use_rope and self.learned_pos is not None:
            positions = torch.arange(
                cache_offset, cache_offset + t, device=cls_tokens.device
            ).clamp(max=self.max_context - 1)
            cls_tokens = cls_tokens + self.learned_pos(positions).unsqueeze(0)

        new_layers: list[tuple[Tensor, Tensor]] = []
        x = cls_tokens
        for i, block in enumerate(self.temporal_blocks):
            ck = cv = None
            if kv_cache is not None:
                ck, cv = kv_cache.layers[i]
            x, nk, nv = block(
                x, rope=self.rope, cache_k=ck, cache_v=cv, cache_offset=cache_offset
            )
            # Truncate to max_context (drop oldest) if needed.
            if nk.size(2) > self.max_context:
                nk = nk[:, :, -self.max_context :, :]
                nv = nv[:, :, -self.max_context :, :]
            new_layers.append((nk, nv))

        x = self.temporal_norm(x)
        state_vec = x if return_all else x[:, -1, :]

        new_cache = None
        if self.kv_cache_enabled and append:
            new_len = new_layers[0][0].size(2)
            new_cache = TemporalKVCache(layers=new_layers, length=new_len)
        return state_vec, new_cache

    def decode_options(
        self,
        option_svs: Union[SparseVector, Sequence[SparseVector]],
        spatial_memory: Tensor,
        state_vec: Tensor,
        *,
        n_options: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Score option SparseVector(s) → logits ``[B, max_N]`` (pad with -inf)."""
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
            opt_tokens = torch.zeros(b, max_n, self.d_model, device=device)
        else:
            idx_t = torch.tensor(index, dtype=torch.long, device=device).clamp(
                0, self.decoder_vocab - 1
            )
            val_t = torch.tensor(value, dtype=torch.float32, device=device)
            off_t = torch.tensor(offset, dtype=torch.long, device=device)
            opt_tokens = self.option_bag(idx_t, off_t, val_t).view(
                b, max_n, self.d_model
            )

        # Memory = spatial tokens + state as an extra key.
        state_tok = state_vec.unsqueeze(1)
        memory = torch.cat([spatial_memory, state_tok], dim=1)

        h = opt_tokens
        for layer in self.option_decoder:
            h = layer(h, memory)
        logits = self.policy_head(h).squeeze(-1)  # [B, max_N]

        # Mask padded options.
        for i, n in enumerate(n_options):
            if n < max_n:
                logits[i, n:] = float("-inf")
        return logits

    # ----- high-level API -----

    def forward(
        self,
        board: Union[SparseVector, Sequence[SparseVector]],
        options: Union[SparseVector, Sequence[SparseVector]],
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append_cache: bool = True,
        n_options: Optional[Sequence[int]] = None,
    ) -> dict[str, Union[Tensor, Optional[TemporalKVCache]]]:
        """Full forward: board + options (+ optional temporal cache).

        Returns dict with ``policy_logits``, ``value``, ``aux_logits``,
        ``state_vec``, ``spatial_memory``, ``kv_cache``.
        """
        if isinstance(board, SparseVector):
            board = [board]
        if isinstance(options, SparseVector):
            options = [options]
        if n_options is None:
            n_options = [sv.num_words for sv in options]

        spatial = self.encode_board(board)
        cls = self.pool_cls(spatial).unsqueeze(1)  # [B, 1, D]
        state_vec, new_cache = self.temporal_encode(
            cls, kv_cache, append=append_cache
        )
        logits = self.decode_options(
            options, spatial, state_vec, n_options=n_options
        )
        value = torch.tanh(self.value_head(state_vec)).squeeze(-1)
        aux = self.aux_head(state_vec)
        return {
            "policy_logits": logits,
            "value": value,
            "aux_logits": aux,
            "state_vec": state_vec,
            "spatial_memory": spatial,
            "kv_cache": new_cache,
        }

    def forward_from_obs(
        self,
        obs,
        your_deck: list[int],
        kv_cache: Optional[TemporalKVCache] = None,
        *,
        append_cache: bool = True,
        assert_info: bool = True,
    ) -> dict[str, Union[Tensor, Optional[TemporalKVCache], list]]:
        """Featurize an observation and run forward (info-set only)."""
        if assert_info:
            features.assert_info_set(obs)
        board = features.build_board_tokens(obs, your_deck)
        combos = features.enumerate_action_combos(obs)
        opt = features.build_option_tokens(obs, combos)
        out = self.forward(board, opt, kv_cache, append_cache=append_cache)
        out["action_combos"] = combos
        return out

    def empty_cache(self) -> TemporalKVCache:
        """Return an empty KV cache (no layers filled yet — pass None instead).

        Callers should pass ``kv_cache=None`` for the first timestep; this helper
        exists for API symmetry.
        """
        return TemporalKVCache(layers=[], length=0)


def build_model(
    cfg: Optional[config.ModelConfig] = None,
    *,
    device: Optional[torch.device] = None,
    aux_archetype_classes: Optional[int] = None,
) -> TemporalCabtTransformer:
    """Construct a :class:`TemporalCabtTransformer` on ``device`` (default CPU)."""
    from . import archetypes

    if aux_archetype_classes is None:
        # registered archetypes + unknown
        aux_archetype_classes = len(archetypes.archetype_ids()) + 1
    model = TemporalCabtTransformer(
        cfg, aux_archetype_classes=aux_archetype_classes
    )
    if device is not None:
        model = model.to(device)
    return model
