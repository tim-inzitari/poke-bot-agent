from __future__ import annotations

import torch


class TemporalTransformer(torch.nn.Module):
    """Causal temporal transformer with card embeddings and per-timestep heads."""

    def __init__(
        self,
        input_dim: int,
        policy_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        window_size: int,
        *,
        card_vocab_size: int = 2000,
        card_embed_dim: int = 32,
        card_slot_count: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.card_slot_count = card_slot_count
        self.token_proj = torch.nn.Linear(input_dim, d_model)
        self.card_embed = torch.nn.Embedding(card_vocab_size, card_embed_dim)
        self.card_proj = torch.nn.Linear(card_embed_dim, d_model)
        self.position_embed = torch.nn.Embedding(window_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, policy_dim),
        )
        self.next_feature_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, input_dim),
        )
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, 1),
        )
        self._grad_checkpoint = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self._grad_checkpoint = bool(enabled)

    @staticmethod
    def _causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _card_token_bias(self, card_ids: torch.Tensor | None) -> torch.Tensor | None:
        if card_ids is None:
            return None
        embedded = self.card_embed(card_ids.clamp(min=0))
        pooled = embedded.sum(dim=-2)
        return self.card_proj(pooled)

    def encode_sequence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        card_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        tokens = self.token_proj(x) + self.position_embed(positions).unsqueeze(0)
        card_bias = self._card_token_bias(card_ids)
        if card_bias is not None:
            tokens = tokens + card_bias

        padding_mask = mask <= 0
        attn_mask = self._causal_attention_mask(seq_len, x.device)
        if self._grad_checkpoint and self.training:
            encoded = torch.utils.checkpoint.checkpoint(
                self.encoder,
                tokens,
                attn_mask,
                padding_mask,
                use_reentrant=False,
            )
        else:
            encoded = self.encoder(tokens, mask=attn_mask, src_key_padding_mask=padding_mask)
        return self.norm(encoded)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        card_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_sequence(x, mask, card_ids=card_ids)
        return {
            "encoded": encoded,
            "value": self.value_head(encoded).squeeze(-1),
            "policy_logits": self.policy_head(encoded),
            "next_features": self.next_feature_head(encoded),
            "log_variance": self.uncertainty_head(encoded).squeeze(-1).clamp(-5.0, 5.0),
        }

    def forward_last(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        card_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(x, mask, card_ids=card_ids)
        batch_size = x.shape[0]
        last_indices = mask.sum(dim=1).long().clamp(min=1) - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        pooled = out["encoded"][batch_indices, last_indices]
        return {
            "value": self.value_head(pooled).squeeze(-1),
            "policy_logits": self.policy_head(pooled),
            "next_features": self.next_feature_head(pooled),
            "log_variance": self.uncertainty_head(pooled).squeeze(-1).clamp(-5.0, 5.0),
        }


# Checkpoint/submission alias — older code and saved weights use this name.
TransformerRLModel = TemporalTransformer
