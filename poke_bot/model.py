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
1. Spatial EmbeddingBag → 24 tokens → pre-norm TransformerEncoder.
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

        enc_vocab = encoder_vocab if encoder_vocab is not None else features.encoder_vocab_size()
        dec_vocab = decoder_vocab if decoder_vocab is not None else features.decoder_vocab_size()
        self.encoder_vocab = enc_vocab
        self.decoder_vocab = dec_vocab
        card_vocab = (
            int(belief_card_vocab)
            if belief_card_vocab is not None
            else int(features.card_vocab_size())
        )
        if card_vocab <= 0:
            raise ValueError(f"belief_card_vocab must be positive, got {card_vocab}")
        self.belief_card_vocab = card_vocab

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

    def encode_previous_actions(
        self,
        actions: Sequence[Optional[SparseVector]],
    ) -> Tensor:
        """Encode shifted, already-taken actions for temporal history tokens."""
        device = next(self.parameters()).device
        out = torch.zeros(len(actions), self.d_model, device=device)
        present = [(i, action) for i, action in enumerate(actions) if action is not None]
        if not present:
            return out
        packed = pack_sparse_batch([action for _, action in present], 1, device)
        encoded = self.option_bag(
            packed.index.clamp(0, self.decoder_vocab - 1),
            packed.offset,
            packed.value,
        )
        rows = torch.tensor([i for i, _ in present], dtype=torch.long, device=device)
        return out.index_copy(0, rows, encoded)

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
    ) -> dict[str, Union[Tensor, Optional[TemporalKVCache]]]:
        """Evaluate variable-length realized histories.

        Spatial encoding and option decoding are batched across games. Temporal
        encoding is grouped logically per game so padding can never become
        observable history.
        """
        histories = [list(h)[-self.max_context :] for h in board_histories]
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
        for length, previous_actions in zip(lengths, action_histories):
            spatial = flat_spatial[start : start + length]
            cls = self.history_tokens(spatial, previous_actions).unsqueeze(0)
            state, _ = self.temporal_encode(cls, append=False)
            states.append(state.squeeze(0))
            current_spatial.append(spatial[-1])
            start += length
        state_vec = torch.stack(states, dim=0)
        spatial_memory = torch.stack(current_spatial, dim=0)
        logits = self.decode_options(
            options, spatial_memory, state_vec, n_options=n_options
        )
        out = {
            "policy_logits": logits,
            "value": torch.tanh(self.value_head(state_vec)).squeeze(-1),
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
        append_cache: bool = False,
        n_options: Optional[Sequence[int]] = None,
        previous_action: Optional[SparseVector] = None,
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
        logits = self.decode_options(
            options, spatial, state_vec, n_options=n_options
        )
        value = torch.tanh(self.value_head(state_vec)).squeeze(-1)
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
        return TemporalKVCache(layers=[], length=0)


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
