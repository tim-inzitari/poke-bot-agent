import torch

from poke_agent.models.temporal_transformer import KANLinear, TemporalTransformer


def test_kan_linear_preserves_batch_shape():
    layer = KANLinear(5, 3, grid_size=4)
    x = torch.randn(2, 7, 5)
    y = layer(x)
    assert y.shape == (2, 7, 3)


def test_temporal_kan_forward_shapes():
    model = TemporalTransformer(
        input_dim=11,
        policy_dim=8,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        window_size=6,
        kan_grid_size=4,
        use_kan=True,
    )
    x = torch.randn(3, 6, 11)
    mask = torch.ones(3, 6)
    out = model(x, mask)
    assert out["value"].shape == (3,)
    assert out["policy_logits"].shape == (3, 8)
    assert out["next_features"].shape == (3, 11)
    assert out["log_variance"].shape == (3,)


def test_temporal_transformer_legacy_forward_shapes():
    model = TemporalTransformer(
        input_dim=11,
        policy_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        window_size=6,
        use_kan=False,
    )
    x = torch.randn(2, 6, 11)
    mask = torch.ones(2, 6)
    out = model(x, mask)
    assert out["value"].shape == (2,)
    assert out["policy_logits"].shape == (2, 8)
