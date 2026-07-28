from __future__ import annotations

from pathlib import Path

import torch

from poke_bot import checkpoint
from poke_bot.matchup_adapters import MatchupAdapterBank
from scripts.run_passed_alakazam_core_distillation import (
    TRANSFER_INITIALIZATION_SCHEMA,
    _materialize_transfer_initialization,
)


def _specialist_source(path: Path) -> tuple[Path, str, dict[str, torch.Tensor]]:
    torch.manual_seed(91)
    bank = MatchupAdapterBank(enabled=False)
    with torch.no_grad():
        bank.experts[0].down.weight.fill_(0.75)
        bank.experts[0].up.bias.fill_(0.5)
    base = {
        "encoder.weight": torch.randn(3, 4),
        "policy_head.weight": torch.randn(1, 4),
    }
    state = {
        **base,
        **{
            f"matchup_adapter_bank.{name}": value.detach().clone()
            for name, value in bank.state_dict().items()
        },
    }
    payload = {
        "model_state_dict": state,
        "model_config": {
            "d_model": 96,
            "temporal_layers": 1,
            "decision_context": "history",
            "matchup_adapters_enabled": False,
        },
        "archetype_id": "alakazam",
        "model_id": "passed-alakazam",
        "step": 999,
        "epoch": 30,
        "rl_iteration": 30,
        "best_metric": 0.1,
        "early_stop_state": {"best": 0.1},
        "optimizer_state_dict": {"state": {1: {"step": 7}}},
        "scaler_state_dict": {"scale": 8.0},
        "scheduler_state_dict": {"last_epoch": 30},
        "rng_state": {"torch": torch.random.get_rng_state()},
        "provenance": {"feature_schema": 9},
        "extra": {"alakazam_only": True},
    }
    checkpoint.immutable_torch_save(payload, path)
    return path, checkpoint.checkpoint_digest(path), base


def test_transfer_initialization_preserves_core_and_resets_specialist_state(
    tmp_path: Path,
) -> None:
    source, digest, base = _specialist_source(tmp_path / "passed.pt")
    output = tmp_path / "transfer.pt"

    result = _materialize_transfer_initialization(
        source_checkpoint=source,
        source_digest=digest,
        output_path=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")

    assert result["path"] == str(output.resolve())
    assert result["adapter_expert_count"] == len(MatchupAdapterBank.expert_ids)
    assert saved["archetype_id"] == "unknown"
    assert saved["model_id"] == "deck_agnostic_core.transfer_initialization"
    assert saved["step"] == saved["epoch"] == saved["rl_iteration"] == 0
    for name, expected in base.items():
        assert torch.equal(saved["model_state_dict"][name], expected)
    for forbidden in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
    ):
        assert forbidden not in saved
    extra = saved["extra"]
    assert "alakazam_only" not in extra
    assert extra["core_transfer_initialization"]["schema"] == (
        TRANSFER_INITIALIZATION_SCHEMA
    )
    assert extra["matchup_adapters_runtime_enabled"] is False
    assert extra["matchup_adapter_training_enabled"] is False
    adapter_state = {
        name: value
        for name, value in saved["model_state_dict"].items()
        if name.startswith("matchup_adapter_bank.")
    }
    assert adapter_state
    assert all(
        int(value.count_nonzero().item()) == 0
        for name, value in adapter_state.items()
        if name.endswith(".up.weight") or name.endswith(".up.bias")
    )

    # Recovery is idempotent only for the exact immutable source identity.
    assert _materialize_transfer_initialization(
        source_checkpoint=source,
        source_digest=digest,
        output_path=output,
    ) == result


def test_transfer_initialization_rejects_wrong_source_digest(tmp_path: Path) -> None:
    source, _digest, _base = _specialist_source(tmp_path / "passed.pt")
    try:
        _materialize_transfer_initialization(
            source_checkpoint=source,
            source_digest="sha256:" + "0" * 64,
            output_path=tmp_path / "transfer.pt",
        )
    except RuntimeError as exc:
        assert "source identity changed" in str(exc)
    else:  # pragma: no cover - explicit failure message is easier to diagnose
        raise AssertionError("wrong passed-checkpoint digest was accepted")


def test_core_distillation_materializes_all_head_targets() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_passed_alakazam_core_distillation.py"
    ).read_text(encoding="utf-8")
    assert "belief_card_vocab_from_state" in source
    assert "belief_card_vocab=belief_card_vocab" in source
    assert "if not corpus.has_exact_targets" in source
