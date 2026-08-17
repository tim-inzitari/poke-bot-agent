"""Dense / factorized card embeddings (Option A: shared card+attack+role compose).

Flat ``nn.EmbeddingBag`` tables (``board_bag`` / ``option_bag``) dominate Pure RL
params because feature indices tile **cardId × role**. This module replaces both
bags with shared card/attack tables plus typed role embeddings, composed into
``d_model`` tokens via a small Linear. Feature builders stay unchanged; flat
sparse indices are decoded into (entity, role) lookups.

``ModelConfig.dense_card2vec`` / ``DENSE_CARD2VEC`` enables the path. Pure RL
defaults it **ON** via :func:`poke_bot.pure_rl.model_profile.pure_rl_model_config`.
The model wrapper adds a compact exact embedding for schema-v5 composite
option-binding rows; this module continues to factorize card/attack rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# Mirrors features.decoder_vocab_size layout (keep in sync if schema changes).
_DECODER_TYPED_FLAGS = 14
_DECODER_MAIN_FEATURES = 8
_DECODER_SELECT_CONTEXTS = 49  # SelectContext.RECOVER_SPECIAL_CONDITION + 1
_DECODER_BINDING_ROWS = 4 * 3 * 13 * 65  # role × owner × area × exact index
_ENCODER_HEADROOM = 512

# Entity kinds for factorized lookup tables.
_KIND_NULL = 0
_KIND_CARD = 1
_KIND_ATTACK = 2


@dataclass(frozen=True)
class BagVocabEstimate:
    card_vocab: int
    attack_vocab: int
    encoder_vocab: int
    decoder_vocab: int

    def bag_params(self, d_model: int) -> int:
        return (self.encoder_vocab + self.decoder_vocab) * int(d_model)


@dataclass(frozen=True)
class FactorizedEmbedEstimate:
    """Rough param count for shared card/attack + role composition (Option A)."""

    card_vocab: int
    attack_vocab: int
    d_card: int
    d_model: int
    n_roles: int = 80  # typed flags + main slots + select contexts + slack
    n_board_scalars: int = 64

    @property
    def card_table(self) -> int:
        return self.card_vocab * self.d_card

    @property
    def attack_table(self) -> int:
        return self.attack_vocab * self.d_card

    @property
    def role_table(self) -> int:
        return self.n_roles * self.d_card

    @property
    def board_scalar_proj(self) -> int:
        # Linear(n_board_scalars → d_model) + bias
        return self.n_board_scalars * self.d_model + self.d_model

    @property
    def compose_mlp(self) -> int:
        # concat(card, role) → d_model (one Linear + bias); attack reuses path
        inn = 2 * self.d_card
        return inn * self.d_model + self.d_model

    @property
    def total_embed_params(self) -> int:
        return (
            self.card_table
            + self.attack_table
            + self.role_table
            + self.board_scalar_proj
            + self.compose_mlp
        )


def estimate_bag_vocabs(
    card_vocab: int = 1268,
    attack_vocab: int = 1557,
    *,
    headroom: int = _ENCODER_HEADROOM,
    n_select_contexts: int = _DECODER_SELECT_CONTEXTS,
) -> BagVocabEstimate:
    """Closed-form vocab sizes matching :mod:`poke_bot.features` formulas."""
    cc = int(card_vocab)
    ac = int(attack_vocab)
    poke = 2 + 3 * cc
    player_width = 7 + 5 + cc
    encoder = (
        2 * poke  # bench (shared span per player)
        + 2 * poke  # active
        + 2 * player_width
        + cc  # hand
        + cc  # deck
        + cc  # stadium
        + 5  # global
        + int(headroom)
    )
    card_offset = _DECODER_TYPED_FLAGS + ac
    card_blocks = 1 + _DECODER_MAIN_FEATURES + int(n_select_contexts)
    decoder = card_offset + card_blocks * cc + _DECODER_BINDING_ROWS
    return BagVocabEstimate(cc, ac, encoder, decoder)


def estimate_factorized(
    card_vocab: int = 1268,
    attack_vocab: int = 1557,
    *,
    d_card: int = 16,
    d_model: int = 16,
) -> FactorizedEmbedEstimate:
    return FactorizedEmbedEstimate(
        card_vocab=int(card_vocab),
        attack_vocab=int(attack_vocab),
        d_card=int(d_card),
        d_model=int(d_model),
    )


def param_delta_summary(
    *,
    card_vocab: int = 1268,
    attack_vocab: int = 1557,
    d_model: int = 16,
    d_card: Optional[int] = None,
) -> dict[str, int]:
    """Compare flat EmbeddingBag bags vs factorized shared tables."""
    d_card = int(d_card if d_card is not None else d_model)
    bags = estimate_bag_vocabs(card_vocab, attack_vocab)
    fac = estimate_factorized(
        card_vocab, attack_vocab, d_card=d_card, d_model=d_model
    )
    bag_n = bags.bag_params(d_model)
    fac_n = fac.total_embed_params
    return {
        "encoder_vocab": bags.encoder_vocab,
        "decoder_vocab": bags.decoder_vocab,
        "flat_bag_params": bag_n,
        "factorized_embed_params": fac_n,
        "params_freed": bag_n - fac_n,
    }


# ---------------------------------------------------------------------------
# Role ids (shared board + option tables; keep < FactorizedEmbedEstimate.n_roles)
# ---------------------------------------------------------------------------

# Board poke regions: (empty, hp, card, tool, energy) × 4 regions.
_ROLE_BENCH_SELF = (1, 2, 3, 4, 5)
_ROLE_BENCH_OPP = (6, 7, 8, 9, 10)
_ROLE_ACTIVE_SELF = (11, 12, 13, 14, 15)
_ROLE_ACTIVE_OPP = (16, 17, 18, 19, 20)
_ROLE_SELF_PLAYER_SCALAR0 = 21  # ..27
_ROLE_SELF_STATUS0 = 28  # ..32
_ROLE_SELF_DISCARD = 33
_ROLE_OPP_PLAYER_SCALAR0 = 34  # ..40
_ROLE_OPP_STATUS0 = 41  # ..45
_ROLE_OPP_DISCARD = 46
_ROLE_HAND = 47
_ROLE_DECK = 48
_ROLE_STADIUM = 49
_ROLE_GLOBAL0 = 50  # ..54

# Option / decoder roles.
_ROLE_TYPED0 = 0  # 0..13 typed flags (overlap pad role 0 is fine for flag 0)
_ROLE_ATTACK = 55
_ROLE_MAIN0 = 56  # ..63  (8 main features)
_ROLE_CTX0 = 64  # ..64+48  (49 select contexts) → roles 64..112
# n_roles default 128 covers ctx slack; estimate used 80 — bump module default.


def _fill_poke(
    kind: Tensor,
    eid: Tensor,
    role: Tensor,
    base: int,
    cc: int,
    roles: tuple[int, int, int, int, int],
) -> None:
    empty_r, hp_r, card_r, tool_r, energy_r = roles
    role[base] = empty_r
    role[base + 1] = hp_r
    for i in range(cc):
        kind[base + 2 + i] = _KIND_CARD
        eid[base + 2 + i] = i
        role[base + 2 + i] = card_r
        kind[base + 2 + cc + i] = _KIND_CARD
        eid[base + 2 + cc + i] = i
        role[base + 2 + cc + i] = tool_r
        kind[base + 2 + 2 * cc + i] = _KIND_CARD
        eid[base + 2 + 2 * cc + i] = i
        role[base + 2 + 2 * cc + i] = energy_r


def build_encoder_factor_tables(
    card_vocab: int,
    encoder_vocab: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map flat board feature index → (kind, entity_id, role_id)."""
    cc = int(card_vocab)
    n = int(encoder_vocab)
    kind = torch.zeros(n, dtype=torch.long)
    eid = torch.zeros(n, dtype=torch.long)
    role = torch.zeros(n, dtype=torch.long)
    poke = 2 + 3 * cc
    pos = 0
    for roles in (
        _ROLE_BENCH_SELF,
        _ROLE_BENCH_OPP,
        _ROLE_ACTIVE_SELF,
        _ROLE_ACTIVE_OPP,
    ):
        if pos + poke > n:
            break
        _fill_poke(kind, eid, role, pos, cc, roles)
        pos += poke

    player_width = 7 + 5 + cc
    for scalar0, status0, discard_r in (
        (_ROLE_SELF_PLAYER_SCALAR0, _ROLE_SELF_STATUS0, _ROLE_SELF_DISCARD),
        (_ROLE_OPP_PLAYER_SCALAR0, _ROLE_OPP_STATUS0, _ROLE_OPP_DISCARD),
    ):
        if pos + player_width > n:
            break
        for s in range(7):
            role[pos + s] = scalar0 + s
        for s in range(5):
            role[pos + 7 + s] = status0 + s
        for i in range(cc):
            kind[pos + 12 + i] = _KIND_CARD
            eid[pos + 12 + i] = i
            role[pos + 12 + i] = discard_r
        pos += player_width

    for bag_role in (_ROLE_HAND, _ROLE_DECK, _ROLE_STADIUM):
        if pos + cc > n:
            break
        for i in range(cc):
            kind[pos + i] = _KIND_CARD
            eid[pos + i] = i
            role[pos + i] = bag_role
        pos += cc

    for g in range(5):
        if pos + g < n:
            role[pos + g] = _ROLE_GLOBAL0 + g
    return kind, eid, role


def build_decoder_factor_tables(
    card_vocab: int,
    attack_vocab: int,
    decoder_vocab: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map flat option feature index → (kind, entity_id, role_id)."""
    cc = int(card_vocab)
    ac = int(attack_vocab)
    n = int(decoder_vocab)
    kind = torch.zeros(n, dtype=torch.long)
    eid = torch.zeros(n, dtype=torch.long)
    role = torch.zeros(n, dtype=torch.long)

    for i in range(min(_DECODER_TYPED_FLAGS, n)):
        role[i] = _ROLE_TYPED0 + i

    for aid in range(ac):
        idx = _DECODER_TYPED_FLAGS + aid
        if idx >= n:
            break
        kind[idx] = _KIND_ATTACK
        eid[idx] = aid
        role[idx] = _ROLE_ATTACK

    card_offset = _DECODER_TYPED_FLAGS + ac
    # Layout: (1 + 8 main + 49 contexts) card blocks. Builders use block
    # 0..7 for main features and (8 + SelectContext) for CARD/SKILL/etc.
    n_blocks = 1 + _DECODER_MAIN_FEATURES + _DECODER_SELECT_CONTEXTS
    for block in range(n_blocks):
        if block < _DECODER_MAIN_FEATURES:
            role_id = _ROLE_MAIN0 + block
        else:
            # blocks 8..56 → contexts 0..48; block 57 (extra +1) → spare ctx
            role_id = _ROLE_CTX0 + (block - _DECODER_MAIN_FEATURES)
        base = card_offset + block * cc
        for cid in range(cc):
            idx = base + cid
            if idx >= n:
                return kind, eid, role
            kind[idx] = _KIND_CARD
            eid[idx] = cid
            role[idx] = role_id
    return kind, eid, role


def _bag_sum(vecs: Tensor, offsets: Tensor) -> Tensor:
    """Sum ``vecs[nnz, D]`` into bags defined by ``offsets[n_bags+1]``."""
    n_bags = int(offsets.numel()) - 1
    if n_bags <= 0:
        return vecs.new_zeros(0, vecs.size(-1))
    counts = offsets[1:] - offsets[:-1]
    if int(vecs.size(0)) == 0:
        return vecs.new_zeros(n_bags, vecs.size(-1))
    bag_ids = torch.repeat_interleave(
        torch.arange(n_bags, device=vecs.device, dtype=torch.long),
        counts,
    )
    out = vecs.new_zeros(n_bags, vecs.size(-1))
    return out.index_add_(0, bag_ids, vecs)


class FactorizedCard2Vec(nn.Module):
    """Shared card/attack/role tables composing both board and option bags."""

    def __init__(
        self,
        *,
        card_vocab: int,
        attack_vocab: int,
        encoder_vocab: int,
        decoder_vocab: int,
        d_card: int,
        d_model: int,
        n_roles: int = 128,
    ):
        super().__init__()
        self.card_vocab = int(card_vocab)
        self.attack_vocab = int(attack_vocab)
        self.encoder_vocab = int(encoder_vocab)
        self.decoder_vocab = int(decoder_vocab)
        self.d_card = int(d_card)
        self.d_model = int(d_model)
        self.n_roles = int(n_roles)

        # +1 null row for kind=NULL features (scalars / typed flags).
        self.card_emb = nn.Embedding(self.card_vocab + 1, self.d_card)
        self.attack_emb = nn.Embedding(self.attack_vocab + 1, self.d_card)
        self.role_emb = nn.Embedding(self.n_roles, self.d_card)
        self.compose = nn.Linear(2 * self.d_card, self.d_model)
        # Scalar / typed-flag features use this instead of a card/attack row.
        self.null_card = nn.Parameter(torch.zeros(self.d_card))

        b_kind, b_eid, b_role = build_encoder_factor_tables(
            self.card_vocab, self.encoder_vocab
        )
        o_kind, o_eid, o_role = build_decoder_factor_tables(
            self.card_vocab, self.attack_vocab, self.decoder_vocab
        )
        self.register_buffer("board_kind", b_kind, persistent=True)
        self.register_buffer("board_eid", b_eid, persistent=True)
        self.register_buffer("board_role", b_role, persistent=True)
        self.register_buffer("option_kind", o_kind, persistent=True)
        self.register_buffer("option_eid", o_eid, persistent=True)
        self.register_buffer("option_role", o_role, persistent=True)

        # Clamp role ids into table (ctx roles can exceed n_roles if misconfigured).
        if (
            int(self.board_role.max()) >= self.n_roles
            or int(self.option_role.max()) >= self.n_roles
        ):
            need = int(max(int(self.board_role.max()), int(self.option_role.max()))) + 1
            raise ValueError(
                f"FactorizedCard2Vec n_roles={self.n_roles} too small; need >={need}"
            )

    def _embed_indices(
        self,
        indices: Tensor,
        values: Tensor,
        kind_table: Tensor,
        eid_table: Tensor,
        role_table: Tensor,
    ) -> Tensor:
        idx = indices.clamp(0, kind_table.numel() - 1)
        kind = kind_table[idx]
        eid = eid_table[idx]
        role_id = role_table[idx].clamp(0, self.n_roles - 1)

        card_id = eid.clamp(0, self.card_vocab - 1)
        atk_id = eid.clamp(0, self.attack_vocab - 1)
        # Null row = last index of each table.
        null_card = torch.full_like(card_id, self.card_vocab)
        null_atk = torch.full_like(atk_id, self.attack_vocab)

        card_vec = self.card_emb(torch.where(kind == _KIND_CARD, card_id, null_card))
        atk_vec = self.attack_emb(torch.where(kind == _KIND_ATTACK, atk_id, null_atk))
        # Prefer card, else attack, else learned null_card parameter.
        entity = torch.where(
            (kind == _KIND_CARD).unsqueeze(-1),
            card_vec,
            torch.where(
                (kind == _KIND_ATTACK).unsqueeze(-1),
                atk_vec,
                self.null_card.expand_as(card_vec),
            ),
        )
        role_vec = self.role_emb(role_id)
        composed = self.compose(torch.cat([entity, role_vec], dim=-1))
        return composed * values.unsqueeze(-1).to(dtype=composed.dtype)

    def embed_board(
        self, indices: Tensor, offsets: Tensor, values: Tensor
    ) -> Tensor:
        vecs = self._embed_indices(
            indices, values, self.board_kind, self.board_eid, self.board_role
        )
        return _bag_sum(vecs, offsets)

    def embed_option(
        self, indices: Tensor, offsets: Tensor, values: Tensor
    ) -> Tensor:
        vecs = self._embed_indices(
            indices, values, self.option_kind, self.option_eid, self.option_role
        )
        return _bag_sum(vecs, offsets)
