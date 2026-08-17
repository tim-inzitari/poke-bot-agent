from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from poke_bot import checkpoint, features
from poke_bot.matchup_adapters import (
    MatchupAdapterBank,
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
)
from poke_bot.matchup_adapters_v6 import load_slot_registry
from poke_bot.pure_rl.model_profile import model_config_dict, pure_rl_model_config
from poke_bot.train import _restore_streaming_derivative_full_model_trainability
from scripts import train_pure_rl


def test_missing_false_flag_and_provenanced_disabled_bank_share_pure_rl_contract(
    tmp_path: Path, monkeypatch,
) -> None:
    profile = model_config_dict(pure_rl_model_config())
    profile.pop("matchup_adapters_enabled")
    bank = MatchupAdapterBank(enabled=False)
    state = {
        "base.weight": torch.zeros(2, 3),
        "_feature_schema_version": torch.tensor(features.FEATURE_SCHEMA_VERSION),
        **{
            f"matchup_adapter_bank.{name}": value
            for name, value in bank.state_dict().items()
        },
    }
    payload = {
        "model_config": profile,
        "model_state_dict": state,
        "provenance": {"feature_schema": features.FEATURE_SCHEMA_VERSION},
        "extra": {
            "pure_rl": True,
            "smoke": False,
            "matchup_adapter_config": bank.config_dict(),
            "matchup_adapters_runtime_enabled": False,
            "matchup_adapter_training_enabled": False,
            "matchup_adapter_optimizer_included": False,
            "dormant_matchup_adapter_bank": {
                "schema": ZERO_DORMANT_CHECKPOINT_SCHEMA,
                "runtime_enabled": False,
                "training_enabled": False,
                "optimizer_imported": False,
                "optimizer_included": False,
                "frozen": True,
                "zero_output": True,
                "parameter_count": sum(
                    parameter.numel() for parameter in bank.parameters()
                ),
                "adapter_config": bank.config_dict(),
            },
        },
        "rl_iteration": 15,
    }
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"mocked by checkpoint.load_checkpoint")
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        checkpoint,
        "assert_trusted_policy_checkpoint",
        lambda *_a, **_k: {"trusted": True},
    )
    monkeypatch.setattr(
        checkpoint,
        "checkpoint_digest",
        lambda *_a, **_k: "sha256:" + "a" * 64,
    )

    contract = train_pure_rl._checkpoint_contract(path, smoke=False)

    assert contract["model_profile"]["matchup_adapters_enabled"] is False
    assert contract["trainable_parameters"] == 6


def test_pure_rl_profile_forces_dormant_runtime_off(monkeypatch) -> None:
    monkeypatch.setenv("MATCHUP_ADAPTERS_ENABLED", "1")
    assert pure_rl_model_config().matchup_adapters_enabled is False


def test_streaming_full_model_bootstrap_keeps_nonzero_adapters_in_optimizer(
    monkeypatch,
) -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(3, 3)
            self.combo_state = nn.Linear(3, 3)
            self.matchup_adapter_bank = MatchupAdapterBank(enabled=False)

    model = _Model()
    for parameter in model.matchup_adapter_bank.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        for name, parameter in model.matchup_adapter_bank.named_parameters():
            if name.endswith("up.weight") or name.endswith("up.bias"):
                parameter.fill_(0.125)

    monkeypatch.setenv("POKEBOT_DERIVATIVE_FULL_MODEL_BOOTSTRAP", "1")
    assert _restore_streaming_derivative_full_model_trainability(model) is True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    payload = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        model_config={"decision_context": "history"},
        extra={"pure_rl": True},
    )

    assert all(
        parameter.requires_grad
        for parameter in model.matchup_adapter_bank.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.combo_state.parameters()
    )
    assert payload["extra"]["matchup_adapter_training_enabled"] is True
    assert payload["extra"]["matchup_adapter_optimizer_included"] is True
    assert payload["extra"]["matchup_adapters_runtime_enabled"] is False


def test_router_format_6_profile_binds_immutable_registry(
    monkeypatch,
) -> None:
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "state/matchup_adapter_roster.json"
    )
    monkeypatch.setenv(
        "POKEBOT_MATCHUP_ADAPTER_FORMAT",
        "poke-bot-matchup-adapter-bank-v6",
    )
    monkeypatch.setenv(
        "POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH",
        str(registry_path),
    )

    profile = pure_rl_model_config()

    assert profile.matchup_adapter_registry == load_slot_registry(
        registry_path
    )
