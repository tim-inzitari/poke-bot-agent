from __future__ import annotations

from pathlib import Path

import torch

from poke_bot import checkpoint, features
from poke_bot.matchup_adapters import (
    MatchupAdapterBank,
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
)
from poke_bot.pure_rl.model_profile import model_config_dict, pure_rl_model_config
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
