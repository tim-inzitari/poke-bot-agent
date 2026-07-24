from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint
from poke_bot.dormant_adapter_compat import (
    FLEET_ROLES,
    LOADER_RUNTIME_FILES,
    build_fleet_compatibility_receipt,
    validate_zero_dormant_checkpoint,
)
from poke_bot.matchup_adapter_activation import ZERO_DORMANT_CHECKPOINT_SCHEMA
from poke_bot.matchup_adapters import MatchupAdapterBank


ROOT = Path(__file__).resolve().parents[1]


def _zero_checkpoint(tmp_path: Path) -> Path:
    bank = MatchupAdapterBank(enabled=False)
    state = {
        "base.weight": torch.arange(6).reshape(2, 3),
        **{
            f"matchup_adapter_bank.{name}": value
            for name, value in bank.state_dict().items()
        },
    }
    payload = {
        "model_config": {"matchup_adapters_enabled": False},
        "model_state_dict": state,
        "optimizer_state_dict": {"param_groups": [{"params": [0]}]},
        "extra": {
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
                "parameter_count": 16_400,
                "adapter_config": bank.config_dict(),
            },
        },
    }
    return checkpoint.atomic_torch_save(payload, tmp_path / "zero.pt")


def _loader_root(tmp_path: Path, role: str, model: Path) -> Path:
    root = tmp_path / role
    for relative in LOADER_RUNTIME_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if role == "submission":
        shutil.copy2(ROOT / "submission" / "main.py", root / "main.py")
        shutil.copy2(model, root / "model.pt")
    return root


def test_fleet_receipt_requires_exact_loader_on_all_devices_and_submission(
    tmp_path: Path,
) -> None:
    model = _zero_checkpoint(tmp_path)
    roots = {role: _loader_root(tmp_path, role, model) for role in FLEET_ROLES}
    receipt = tmp_path / "fleet.json"
    build_fleet_compatibility_receipt(
        checkpoint_path=model,
        reference_root=ROOT,
        roots=roots,
        output_path=receipt,
    )
    payload = json.loads(receipt.read_text())
    assert payload["all_roles_validated"] is True
    assert payload["runtime_enabled"] is False
    assert payload["training_enabled"] is False
    assert set(payload["devices"]) == set(FLEET_ROLES)
    assert payload["devices"]["submission"]["bundled_checkpoint_digest"] == (
        payload["checkpoint"]["digest"]
    )
    with pytest.raises(FileExistsError):
        build_fleet_compatibility_receipt(
            checkpoint_path=model,
            reference_root=ROOT,
            roots=roots,
            output_path=receipt,
        )


def test_fleet_receipt_fails_closed_on_stale_bert_loader(tmp_path: Path) -> None:
    model = _zero_checkpoint(tmp_path)
    roots = {role: _loader_root(tmp_path, role, model) for role in FLEET_ROLES}
    stale = roots["bert"] / "poke_bot" / "model.py"
    stale.write_text(stale.read_text() + "\n# stale\n")
    with pytest.raises(RuntimeError, match="bert loader file digest mismatch"):
        build_fleet_compatibility_receipt(
            checkpoint_path=model,
            reference_root=ROOT,
            roots=roots,
            output_path=tmp_path / "fleet.json",
        )


def test_zero_checkpoint_validation_rejects_nonzero_or_runtime_active_bank(
    tmp_path: Path,
) -> None:
    model = _zero_checkpoint(tmp_path)
    assert validate_zero_dormant_checkpoint(model)["parameter_count"] == 16_400
    payload = checkpoint.load_checkpoint(model)
    nonzero = copy.deepcopy(payload)
    nonzero["model_state_dict"][
        "matchup_adapter_bank.experts.0.up.bias"
    ].fill_(1.0)
    nonzero_path = checkpoint.atomic_torch_save(nonzero, tmp_path / "nonzero.pt")
    with pytest.raises(RuntimeError, match="output is non-zero"):
        validate_zero_dormant_checkpoint(nonzero_path)

    active = copy.deepcopy(payload)
    active["model_config"]["matchup_adapters_enabled"] = True
    active_path = checkpoint.atomic_torch_save(active, tmp_path / "active.pt")
    with pytest.raises(RuntimeError, match="not an explicit frozen"):
        validate_zero_dormant_checkpoint(active_path)


def test_fleet_receipt_requires_all_four_roles(tmp_path: Path) -> None:
    model = _zero_checkpoint(tmp_path)
    roots = {
        role: _loader_root(tmp_path, role, model)
        for role in FLEET_ROLES
        if role != "elmo"
    }
    with pytest.raises(RuntimeError, match=r"missing=\['elmo'\]"):
        build_fleet_compatibility_receipt(
            checkpoint_path=model,
            reference_root=ROOT,
            roots=roots,
            output_path=tmp_path / "fleet.json",
        )
