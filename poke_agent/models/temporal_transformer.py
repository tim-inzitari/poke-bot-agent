from __future__ import annotations

import torch


class TemporalTransformer(torch.nn.Module):
    """Temporal transformer RL model with value, policy, dynamics, and uncertainty heads."""

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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.token_proj = torch.nn.Linear(input_dim, d_model)
        self.position_embed = torch.nn.Embedding(window_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
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

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        tokens = self.token_proj(x) + self.position_embed(positions).unsqueeze(0)
        padding_mask = mask <= 0
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.norm(encoded[:, -1])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.encode(x, mask)
        return {
            "value": self.value_head(pooled).squeeze(-1),
            "policy_logits": self.policy_head(pooled),
            "next_features": self.next_feature_head(pooled),
            "log_variance": self.uncertainty_head(pooled).squeeze(-1).clamp(-5.0, 5.0),
        }


# Checkpoint/submission alias — older code and saved weights use this name.
TransformerRLModel = TemporalTransformer
