from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint, config
from poke_bot.model import build_model
from poke_bot.train import load_model_from_checkpoint


def _config() -> config.ModelConfig:
    return config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
    )


def test_checkpoint_reconstructs_saved_model_config(tmp_path: Path) -> None:
    cfg = _config()
    model = build_model(
        cfg,
        aux_archetype_classes=3,
        encoder_vocab=37,
        decoder_vocab=41,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    path = checkpoint.atomic_torch_save(payload, tmp_path / "small.pt")

    loaded = load_model_from_checkpoint(path, device=torch.device("cpu"))
    assert loaded.d_model == 16
    assert loaded.encoder_vocab == 37
    assert loaded.decoder_vocab == 41
    assert loaded.aux_head[-1].out_features == 3
    assert loaded.cfg.decision_context == "history"
    assert checkpoint.assert_trusted_policy_checkpoint(path)["decision_context"] == "history"


def test_stateless_policy_checkpoint_is_trusted_without_oracle_state(
    tmp_path: Path,
) -> None:
    cfg = _config()
    cfg.decision_context = "stateless"
    cfg.temporal_layers = 0
    cfg.kv_cache = False
    model = build_model(
        cfg,
        aux_archetype_classes=3,
        encoder_vocab=37,
        decoder_vocab=41,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    path = checkpoint.atomic_torch_save(payload, tmp_path / "state-only.pt")

    trusted = checkpoint.assert_trusted_policy_checkpoint(path)
    assert trusted["decision_context"] == "stateless"
    assert trusted["provenance"]["trusted_policy_path"] is True


def test_checkpoint_rejects_incompatible_model_config(tmp_path: Path) -> None:
    cfg = _config()
    model = build_model(
        cfg,
        aux_archetype_classes=2,
        encoder_vocab=16,
        decoder_vocab=16,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    payload["model_config"]["future_unsupported_field"] = 1
    path = checkpoint.atomic_torch_save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="unsupported model_config"):
        load_model_from_checkpoint(path, device=torch.device("cpu"))
