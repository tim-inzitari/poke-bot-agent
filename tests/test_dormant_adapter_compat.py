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
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    LEGACY_EXPERT_IDS_V4,
    RETIRED_EXPERT_IDS_V5,
    MatchupAdapterBank,
)


ROOT = Path(__file__).resolve().parents[1]


def _zero_checkpoint(tmp_path: Path) -> Path:
    bank = MatchupAdapterBank(enabled=False)
    parameter_count = sum(value.numel() for value in bank.state_dict().values())
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
                "parameter_count": parameter_count,
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
    expected_parameter_count = sum(
        value.numel() for value in MatchupAdapterBank(enabled=False).state_dict().values()
    )
    assert (
        validate_zero_dormant_checkpoint(model)["parameter_count"]
        == expected_parameter_count
    )
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


def test_trained_roster_migration_accepts_audited_optimizer_reset_only(
    tmp_path: Path,
) -> None:
    model = _zero_checkpoint(tmp_path)
    payload = checkpoint.load_checkpoint(model)
    bank = MatchupAdapterBank(enabled=False)
    parameter_count = sum(value.numel() for value in bank.state_dict().values())
    payload["model_state_dict"]["matchup_adapter_bank.experts.0.up.bias"].fill_(1.0)
    payload["extra"]["dormant_matchup_adapter_bank"].update(
        schema="poke_bot.trained_dormant_matchup_adapter/v1",
        zero_output=False,
        parameter_count=parameter_count,
    )
    payload["extra"]["dormant_matchup_adapter_fit"] = {
        "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
        "runtime_enabled": False,
        "base_frozen": True,
        "optimizer_scope": "matchup_adapter_bank_only",
        "steps": 3,
        "rows": 17,
        "optimizer_state_restored": False,
        "roster_migration": "v4_22_to_canonical_v5_18",
    }
    payload["extra"]["roster_migration"] = {
        "schema": "poke_bot.matchup_adapter_roster_migration/v1",
        "source_checkpoint_digest": "sha256:" + "a" * 64,
        "source_expert_ids": list(LEGACY_EXPERT_IDS_V4),
        "target_expert_ids": list(EXPERT_IDS),
        "removed_expert_ids": sorted(RETIRED_EXPERT_IDS_V5),
        "renamed_expert_ids": {"festival-lead": "thwackey"},
        "zero_initialized_expert_ids": ["team-rockets-spidops"],
        "retained_rows_byte_identical": True,
    }
    payload["extra"].pop("dormant_matchup_adapter_optimizer_state", None)
    migrated = checkpoint.atomic_torch_save(payload, tmp_path / "migrated.pt")

    validated = validate_zero_dormant_checkpoint(migrated, allow_trained=True)
    assert validated["trained"] is True
    assert validated["parameter_count"] == parameter_count

    for field, bad_value in (
        ("retained_rows_byte_identical", False),
        ("source_checkpoint_digest", "sha256:not-a-checksum"),
        ("target_expert_ids", list(EXPERT_IDS[:-1])),
    ):
        tampered = copy.deepcopy(payload)
        tampered["extra"]["roster_migration"][field] = bad_value
        path = checkpoint.atomic_torch_save(
            tampered, tmp_path / f"tampered-{field}.pt"
        )
        with pytest.raises(RuntimeError, match="not an explicit frozen"):
            validate_zero_dormant_checkpoint(path, allow_trained=True)


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
