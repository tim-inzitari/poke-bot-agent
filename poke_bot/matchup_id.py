"""Opponent / matchup identification from visible board signals only.

The ID net never sees the opponent's hidden hand. It consumes opponent-VISIBLE
tokens (active, bench, player summary / discard bag, stadium) built from the
acting seat's information set, plus optional shared card EmbeddingBag weights
from :class:`~poke_bot.model.TemporalCabtTransformer`.

Output: softmax posterior over registered archetype ids + ``unknown``.
Phase 5 appends this posterior as a matchup feature token; Phase 6+ may route
to specialist nets when confidence is high.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import archetypes, cg_env, config, features
from .features import SparseVector
from .model import pack_sparse_batch

# Board-token indices in features.build_board_tokens (self-relative layout):
#   0..7   self bench, 8..15 opp bench
#   16     self active, 17 opp active
#   18     self player, 19 opp player
#   20     self hand, 21 self deck, 22 stadium, 23 global
_OPP_BENCH = range(8, 16)
_OPP_ACTIVE = 17
_OPP_PLAYER = 19
_STADIUM = 22
NUM_OPP_VISIBLE_TOKENS = 8 + 1 + 1 + 1  # bench + active + player + stadium


def matchup_class_names() -> list[str]:
    """Ordered class labels: registered archetypes then ``unknown``."""
    return list(archetypes.archetype_ids()) + [archetypes.UNKNOWN]


def num_matchup_classes() -> int:
    return len(matchup_class_names())


def _slice_words(sv: SparseVector, word_indices: Sequence[int]) -> SparseVector:
    """Extract a subset of EmbeddingBag words from a SparseVector."""
    out = SparseVector()
    for w in word_indices:
        out.word_start()
        if w < 0 or w >= sv.num_words:
            continue
        start = sv.offset[w]
        end = sv.offset[w + 1] if w + 1 < len(sv.offset) else len(sv.index)
        # Re-base absolute feature indices into a fresh bag by copying them as-is
        # (EmbeddingBag indices are global feature ids, not word-local).
        for j in range(start, end):
            out.index.append(sv.index[j])
            out.value.append(sv.value[j])
    return out


def build_opponent_visible_tokens(obs, your_deck: Optional[list[int]] = None) -> SparseVector:
    """Build opponent-VISIBLE board tokens only (no self hand / deck).

    Uses the same bag vocabulary as :func:`features.build_board_tokens` so the
    ID net can share EmbeddingBag weights with the policy model. ``your_deck``
    is accepted for API symmetry but ignored (deck pool is not an opp signal).
    """
    del your_deck  # not used; opp-visible only
    features.assert_info_set(obs)
    # Build full board then slice opp-visible words — ensures identical feature
    # index layout / vocab as the policy spatial encoder.
    # A dummy deck is required by build_board_tokens; contents unused for slice.
    if isinstance(obs, dict):
        obs_obj = cg_env.to_observation(obs)
    else:
        obs_obj = obs
    dummy_deck = [0] * 60
    full = features.build_board_tokens(obs_obj, dummy_deck)
    words = list(_OPP_BENCH) + [_OPP_ACTIVE, _OPP_PLAYER, _STADIUM]
    return _slice_words(full, words)


class MatchupIDNet(nn.Module):
    """Small encoder over opponent-visible tokens → archetype posterior.

    Can share ``board_bag`` EmbeddingBag weights with the main transformer by
    passing ``shared_board_bag`` (typically stop-grad during early ID training,
    or jointly trained later).
    """

    def __init__(
        self,
        *,
        d_model: Optional[int] = None,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        encoder_vocab: Optional[int] = None,
        num_classes: Optional[int] = None,
        shared_board_bag: Optional[nn.EmbeddingBag] = None,
        freeze_shared_bag: bool = False,
    ):
        super().__init__()
        cfg = config.MODEL
        d_model = d_model or min(cfg.d_model, 128)
        ff_dim = ff_dim or (d_model * 4)
        enc_vocab = encoder_vocab or features.encoder_vocab_size()
        self.num_classes = num_classes or num_matchup_classes()
        self.d_model = d_model
        self.num_tokens = NUM_OPP_VISIBLE_TOKENS
        self.class_names = matchup_class_names()

        if shared_board_bag is not None:
            self.board_bag = shared_board_bag
            if freeze_shared_bag:
                for p in self.board_bag.parameters():
                    p.requires_grad = False
            # Project shared d_model → local width if they differ.
            shared_dim = shared_board_bag.embedding_dim
            self.bag_proj = (
                nn.Identity()
                if shared_dim == d_model
                else nn.Linear(shared_dim, d_model)
            )
        else:
            self.board_bag = nn.EmbeddingBag(
                enc_vocab, d_model, mode="sum", include_last_offset=True
            )
            self.bag_proj = nn.Identity()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.num_classes),
        )

    def encode_tokens(
        self, opp_svs: Union[SparseVector, Sequence[SparseVector]]
    ) -> torch.Tensor:
        if isinstance(opp_svs, SparseVector):
            opp_svs = [opp_svs]
        device = next(self.parameters()).device
        packed = pack_sparse_batch(opp_svs, self.num_tokens, device)
        vocab = self.board_bag.num_embeddings
        idx = packed.index.clamp(0, vocab - 1)
        tokens = self.board_bag(idx, packed.offset, packed.value)
        b = len(opp_svs)
        tokens = tokens.view(b, self.num_tokens, -1)
        return self.bag_proj(tokens)

    def forward(
        self, opp_svs: Union[SparseVector, Sequence[SparseVector]]
    ) -> dict[str, torch.Tensor]:
        tokens = self.encode_tokens(opp_svs)
        h = self.norm(self.encoder(tokens))
        pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "pooled": pooled,
        }

    def forward_from_obs(self, obs) -> dict[str, torch.Tensor]:
        sv = build_opponent_visible_tokens(obs)
        return self.forward(sv)

    def predict_label(self, obs) -> tuple[str, float]:
        """Return ``(class_name, confidence)`` for a single observation."""
        out = self.forward_from_obs(obs)
        probs = out["probs"][0]
        idx = int(probs.argmax().item())
        return self.class_names[idx], float(probs[idx].item())


def build_matchup_id_net(
    *,
    device: Optional[torch.device] = None,
    shared_board_bag: Optional[nn.EmbeddingBag] = None,
    freeze_shared_bag: bool = False,
) -> MatchupIDNet:
    net = MatchupIDNet(
        shared_board_bag=shared_board_bag,
        freeze_shared_bag=freeze_shared_bag,
    )
    if device is not None:
        net = net.to(device)
    return net
