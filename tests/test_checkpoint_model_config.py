from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint, config
from poke_bot.matchup_adapters_v6 import load_slot_registry
from poke_bot.model import build_model
from poke_bot.train import load_model_from_checkpoint
from poke_bot.worker_pool import WorkerPool


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


def _spawn_load_checkpoint(path: str) -> dict[str, object]:
    """Exercise the same clean-interpreter loader used by gameplay gates."""

    from poke_bot import config as worker_config
    from poke_bot.train import load_model_from_checkpoint as worker_load

    loaded = worker_load(path, device=torch.device("cpu"))
    return {
        "config_path": str(Path(worker_config.__file__).resolve()),
        "expanded_heads_enabled": bool(loaded.cfg.expanded_heads_enabled),
        "setup_board_outcome_head_enabled": bool(
            loaded.cfg.setup_board_outcome_head_enabled
        ),
        "decision_fusion_dedicated_routes_enabled": bool(
            loaded.cfg.decision_fusion_dedicated_routes_enabled
        ),
        "decision_fusion_dedicated_routes_runtime_enabled": bool(
            loaded.cfg.decision_fusion_dedicated_routes_runtime_enabled
        ),
        "matchup_adapter_format": str(loaded.cfg.matchup_adapter_format),
        "matchup_adapter_registry": loaded.cfg.matchup_adapter_registry,
    }


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


def test_v6_checkpoint_loads_in_gameplay_gate_spawn_worker(
    tmp_path: Path,
) -> None:
    cfg = _config()
    cfg.expanded_heads_enabled = True
    cfg.matchup_adapter_format = "poke-bot-matchup-adapter-bank-v6"
    cfg.matchup_adapter_registry = load_slot_registry()
    model = build_model(
        cfg,
        aux_archetype_classes=2,
        encoder_vocab=16,
        decoder_vocab=16,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    path = checkpoint.atomic_torch_save(payload, tmp_path / "v6.pt")

    with WorkerPool(num_workers=1, recycle_games=1) as pool:
        result = next(pool.imap_unordered(_spawn_load_checkpoint, [str(path)]))

    assert result["config_path"] == str(Path(config.__file__).resolve())
    assert result["expanded_heads_enabled"] is True
    assert result["matchup_adapter_format"] == "poke-bot-matchup-adapter-bank-v6"
    assert result["matchup_adapter_registry"] == cfg.matchup_adapter_registry


def test_legacy_checkpoint_ignores_ambient_v6_router_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    model = build_model(
        cfg,
        aux_archetype_classes=2,
        encoder_vocab=16,
        decoder_vocab=16,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    payload["model_config"].pop("matchup_adapter_format")
    payload["model_config"].pop("matchup_adapter_registry")
    path = checkpoint.atomic_torch_save(payload, tmp_path / "legacy-v5.pt")

    # A fresh gameplay worker imports ModelConfig under the active candidate's
    # Router Format 6 environment. The checkpoint's omitted historical field
    # must still resolve to immutable Router Format 5.
    monkeypatch.setenv(
        "POKEBOT_MATCHUP_ADAPTER_FORMAT",
        "poke-bot-matchup-adapter-bank-v6",
    )
    with WorkerPool(num_workers=1, recycle_games=1) as pool:
        result = next(pool.imap_unordered(_spawn_load_checkpoint, [str(path)]))

    assert result["matchup_adapter_format"] == (
        "poke-bot-matchup-adapter-bank-v5-roster18"
    )
    assert result["matchup_adapter_registry"] is None


def test_historical_checkpoint_ignores_ambient_future_head_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    model = build_model(
        cfg,
        aux_archetype_classes=2,
        encoder_vocab=16,
        decoder_vocab=16,
    )
    payload = checkpoint.build_checkpoint(model=model, model_config=cfg)
    for field in (
        "setup_board_outcome_head_enabled",
        "decision_fusion_dedicated_routes_enabled",
        "decision_fusion_dedicated_routes_runtime_enabled",
    ):
        payload["model_config"].pop(field)
    path = checkpoint.atomic_torch_save(
        payload, tmp_path / "historical-before-fusion-v2.pt"
    )

    monkeypatch.setenv("POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED", "1")
    monkeypatch.setenv(
        "POKEBOT_DECISION_FUSION_DEDICATED_ROUTES_ENABLED", "1"
    )
    monkeypatch.setenv(
        "POKEBOT_DECISION_FUSION_DEDICATED_ROUTES_RUNTIME_ENABLED", "1"
    )
    with WorkerPool(num_workers=1, recycle_games=1) as pool:
        result = next(pool.imap_unordered(_spawn_load_checkpoint, [str(path)]))

    assert result["setup_board_outcome_head_enabled"] is False
    assert result["decision_fusion_dedicated_routes_enabled"] is False
    assert (
        result["decision_fusion_dedicated_routes_runtime_enabled"] is False
    )
