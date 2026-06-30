from __future__ import annotations

import torch


class KANLinear(torch.nn.Module):
    """Small KAN-style layer using learned radial basis expansions.

    This is intentionally self-contained so the Kaggle submission tarball does
    not need another dependency. It combines a regular linear path with learned
    nonlinear basis weights per input dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        base_activation: torch.nn.Module | None = None,
    ):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.base = torch.nn.Linear(in_features, out_features)
        self.base_activation = base_activation or torch.nn.SiLU()
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        spacing = float((grid_max - grid_min) / max(1, grid_size - 1))
        self.gamma = 1.0 / max(spacing * spacing, 1e-6)
        self.spline_weight = torch.nn.Parameter(torch.empty(in_features, grid_size, out_features))
        self.spline_bias = torch.nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.base.weight)
        torch.nn.init.zeros_(self.base.bias)
        torch.nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(self.base_activation(x))
        basis = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.grid) ** 2)
        spline = torch.einsum("...ig,igo->...o", basis, self.spline_weight)
        return base + spline + self.spline_bias


class KANFeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float,
        grid_size: int,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            KANLinear(input_dim, hidden_dim, grid_size=grid_size),
            torch.nn.Dropout(dropout),
            KANLinear(hidden_dim, output_dim, grid_size=grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalKANEncoderLayer(torch.nn.Module):
    """Transformer-style temporal block with KAN feed-forward sublayer."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        kan_grid_size: int,
    ):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = KANFeedForward(
            d_model,
            dim_feedforward,
            d_model,
            dropout=dropout,
            grid_size=kan_grid_size,
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attended))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class TemporalTransformer(torch.nn.Module):
    """Temporal RL model with KAN feed-forward/value heads."""

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
        kan_grid_size: int = 8,
        use_kan: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.kan_grid_size = kan_grid_size
        self.use_kan = use_kan
        self.token_proj = (
            KANLinear(input_dim, d_model, grid_size=kan_grid_size)
            if use_kan
            else torch.nn.Linear(input_dim, d_model)
        )
        self.position_embed = torch.nn.Embedding(window_size, d_model)
        if use_kan:
            self.encoder = torch.nn.ModuleList([
                TemporalKANEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    kan_grid_size=kan_grid_size,
                )
                for _ in range(num_layers)
            ])
        else:
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
        head = KANFeedForward if use_kan else LinearFeedForward
        self.value_head = head(d_model, d_model, 1, dropout=dropout, grid_size=kan_grid_size)
        self.policy_head = head(d_model, d_model, policy_dim, dropout=dropout, grid_size=kan_grid_size)
        self.next_feature_head = head(d_model, d_model, input_dim, dropout=dropout, grid_size=kan_grid_size)
        self.uncertainty_head = head(d_model, d_model, 1, dropout=dropout, grid_size=kan_grid_size)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        tokens = self.token_proj(x) + self.position_embed(positions).unsqueeze(0)
        padding_mask = mask <= 0
        if self.use_kan:
            encoded = tokens
            for layer in self.encoder:
                encoded = layer(encoded, padding_mask)
        else:
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


class LinearFeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float,
        grid_size: int,
    ):
        del grid_size
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Checkpoint/submission alias — older code and saved weights use this name.
TransformerRLModel = TemporalTransformer
