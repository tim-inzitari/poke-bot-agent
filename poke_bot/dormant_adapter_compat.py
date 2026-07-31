"""Fail-closed loader compatibility for zero-output matchup-adapter banks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from . import checkpoint
from .matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as V5_ADAPTER_CHECKPOINT_FORMAT,
    EXPERT_IDS,
    LEGACY_EXPERT_IDS_V4,
    RETIRED_EXPERT_IDS_V5,
    MatchupAdapterBank,
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
)
from .matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
)
from .matchup_adapters_v6 import MatchupAdapterBankV6


FLEET_ROLES = ("inzi", "elmo", "bert", "submission")

# These files are deployed as one identical loader overlay on each device.
# The surrounding worker package remains the already validated device-specific
# implementation; each role additionally runs an actual checkpoint-load smoke
# before the compatibility receipt is accepted.
LOADER_RUNTIME_FILES = (
    "poke_bot/config.py",
    "poke_bot/matchup_adapters.py",
    "poke_bot/matchup_adapters_v6.py",
    "poke_bot/matchup_adapter_routes.py",
    "poke_bot/public_matchup_router.py",
    "poke_bot/matchup_adapter_activation.py",
    "poke_bot/model.py",
    "poke_bot/checkpoint.py",
    "poke_bot/train.py",
    "poke_bot/combo_state.py",
    "poke_bot/combo_state_contract.py",
    "poke_bot/setup_board_outcome.py",
    # Checkpoint staging emits the digest-bound runtime marker. Keep it in the
    # same fleet overlay so Router Format 6 slot bindings cannot be dropped by
    # an older controller while the workers run the newer validator.
    "poke_bot/remote_jobs.py",
    "poke_bot/strategic_heads.py",
    "poke_bot/strategic_losses.py",
    "poke_bot/strategic_schedule.py",
    "poke_bot/dormant_adapter_compat.py",
    # Remote simulation dispatch imports this module dynamically to construct
    # training records.  Keep its keyword/schema contract in the same
    # checksum-bound overlay as the package-side dispatch code.
    "scripts/train_round_robin.py",
)
COMPATIBILITY_SCHEMA = "poke_bot.dormant_adapter_loader_compatibility/v1"
ROSTER_MIGRATION_SCHEMA = "poke_bot.matchup_adapter_roster_migration/v1"


def _valid_v4_to_v5_optimizer_reset_migration(
    extra: Mapping[str, Any],
    *,
    saved_config: Any,
    fit: Mapping[str, Any],
) -> bool:
    """Accept the one audited migration that intentionally resets moments.

    Deleting and renaming adapter rows makes the v4 optimizer slots unsafe to
    import.  The migration is therefore valid only for the exact current
    architecture and exact historical source roster, with an explicit source
    checksum and identity-preservation receipt.  All other trained banks keep
    requiring their continuation optimizer state.
    """

    migration = dict(extra.get("roster_migration") or {})
    source_digest = str(migration.get("source_checkpoint_digest") or "")
    return bool(
        saved_config == MatchupAdapterBank(enabled=False).config_dict()
        and migration.get("schema") == ROSTER_MIGRATION_SCHEMA
        and tuple(migration.get("source_expert_ids") or ())
        == LEGACY_EXPERT_IDS_V4
        and tuple(migration.get("target_expert_ids") or ()) == EXPERT_IDS
        and set(migration.get("removed_expert_ids") or ())
        == set(RETIRED_EXPERT_IDS_V5)
        and migration.get("renamed_expert_ids")
        == {"festival-lead": "thwackey"}
        and migration.get("zero_initialized_expert_ids")
        == ["team-rockets-spidops"]
        and migration.get("retained_rows_byte_identical") is True
        and source_digest.startswith("sha256:")
        and len(source_digest) == len("sha256:") + 64
        and all(value in "0123456789abcdef" for value in source_digest[7:])
        and fit.get("roster_migration") == "v4_22_to_canonical_v5_18"
        and fit.get("optimizer_state_restored") is False
        and not extra.get("dormant_matchup_adapter_optimizer_state")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_zero_dormant_checkpoint(
    path: Path | str,
    *,
    allow_trained: bool = False,
) -> dict[str, Any]:
    """Validate a complete frozen dormant bank.

    Compatibility receipts remain zero-only by default. The ordinary RL loader
    opts into ``allow_trained`` for isolated, receipt-backed adapter weights;
    runtime and main-optimizer activation are still forbidden.
    """

    resolved = Path(path).expanduser().resolve()
    payload = checkpoint.load_checkpoint(resolved, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    extra = dict(payload.get("extra") or {})
    dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
    saved_config = extra.get("matchup_adapter_config")
    saved_format = str(dict(saved_config or {}).get("format") or "")
    if saved_format == V6_ADAPTER_CHECKPOINT_FORMAT:
        embedded_registry = dict(dict(saved_config).get("slot_registry") or {})
        expected_bank = MatchupAdapterBankV6(
            enabled=False,
            registry=embedded_registry,
        )
        legacy_configs: tuple[dict[str, Any], ...] = ()
        if (
            model_config.get("matchup_adapter_format")
            != V6_ADAPTER_CHECKPOINT_FORMAT
        ):
            raise RuntimeError("V6 checkpoint lacks its serialized model selector")
    else:
        expected_bank = MatchupAdapterBank(enabled=False)
        legacy_configs = (
            expected_bank.legacy_config_dict_v1(),
            expected_bank.legacy_config_dict_v2(),
            expected_bank.legacy_config_dict_v3(),
        )
        if saved_format not in {
            V5_ADAPTER_CHECKPOINT_FORMAT,
            *(str(config.get("format") or "") for config in legacy_configs),
        }:
            raise RuntimeError("unsupported dormant adapter checkpoint format")
    expected_config = expected_bank.config_dict()
    saved_expert_count = len(dict(saved_config or {}).get("expert_ids") or [])
    expected_state = expected_bank.state_dict()
    if saved_config in legacy_configs:
        expected_state = {
            name: value
            for name, value in expected_state.items()
            if int(name.split(".")[1]) < saved_expert_count
        }
    expected_parameter_count = sum(value.numel() for value in expected_state.values())
    trained = dormant.get("schema") == "poke_bot.trained_dormant_matchup_adapter/v1"
    fit = dict(extra.get("dormant_matchup_adapter_fit") or {})
    continuation_optimizer_present = bool(
        extra.get("dormant_matchup_adapter_optimizer_state")
    )
    audited_optimizer_reset = _valid_v4_to_v5_optimizer_reset_migration(
        extra,
        saved_config=saved_config,
        fit=fit,
    )
    trained_contract_ok = bool(
        allow_trained
        and trained
        and saved_config in (expected_config, *legacy_configs)
        and dormant.get("zero_output") is False
        and fit.get("schema") == "poke_bot.dormant_matchup_adapter_fit/v1"
        and fit.get("runtime_enabled") is False
        and fit.get("base_frozen") is True
        and fit.get("optimizer_scope") == "matchup_adapter_bank_only"
        and int(fit.get("steps", 0)) > 0
        and int(fit.get("rows", 0)) > 0
        and (continuation_optimizer_present or audited_optimizer_reset)
    )
    zero_contract_ok = bool(
        dormant.get("schema") == ZERO_DORMANT_CHECKPOINT_SCHEMA
        and dormant.get("zero_output") is True
    )
    if (
        bool(model_config.get("matchup_adapters_enabled", False))
        or extra.get("matchup_adapters_runtime_enabled") is not False
        or extra.get("matchup_adapter_training_enabled") is not False
        or extra.get("matchup_adapter_optimizer_included") is not False
        or not (zero_contract_ok or trained_contract_ok)
        or dormant.get("runtime_enabled") is not False
        or dormant.get("training_enabled") is not False
        or dormant.get("optimizer_imported") is not False
        or dormant.get("optimizer_included") is not False
        or dormant.get("frozen") is not True
        or int(dormant.get("parameter_count", -1)) != expected_parameter_count
    ):
        raise RuntimeError("checkpoint is not an explicit frozen dormant bank")
    if (
        saved_config not in (expected_config, *legacy_configs)
        or dormant.get("adapter_config") != saved_config
    ):
        raise RuntimeError("checkpoint adapter architecture/routing contract differs")
    state = dict(payload.get("model_state_dict") or {})
    actual = {
        name.removeprefix("matchup_adapter_bank."): value
        for name, value in state.items()
        if name.startswith("matchup_adapter_bank.")
    }
    expected = expected_state
    if actual.keys() != expected.keys():
        raise RuntimeError("checkpoint has incomplete or unknown adapter tensors")
    for name, value in actual.items():
        reference = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError(f"adapter tensor contract failed: {name}")
        if (
            not trained
            and (name.endswith("up.weight") or name.endswith("up.bias"))
            and int(value.count_nonzero().item())
        ):
            raise RuntimeError(f"dormant adapter output is non-zero: {name}")
    output_nonzero = any(
        int(value.count_nonzero().item()) > 0
        for name, value in actual.items()
        if name.endswith("up.weight") or name.endswith("up.bias")
    )
    if trained and not output_nonzero:
        raise RuntimeError("trained dormant adapter receipt has zero output tensors")
    return {
        "path": str(resolved),
        "digest": sha256_file(resolved),
        "adapter_config": expected_config,
        "runtime_enabled": False,
        "training_enabled": False,
        "trained": trained,
        "parameter_count": sum(value.numel() for value in actual.values()),
    }


def loader_source_contract(reference_root: Path) -> dict[str, str]:
    root = Path(reference_root).expanduser().resolve()
    contract: dict[str, str] = {}
    for relative in LOADER_RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"reference loader file is missing: {path}")
        contract[relative] = sha256_file(path)
    return contract


def validate_loader_root(
    root: Path,
    *,
    role: str,
    source_contract: Mapping[str, str],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Require one device/submission bundle to carry the exact tested loader."""

    role = str(role).strip().lower()
    if role not in FLEET_ROLES:
        raise ValueError(f"unknown loader role {role!r}; expected {FLEET_ROLES}")
    root = Path(root).expanduser().resolve()
    observed: dict[str, str] = {}
    for relative, expected_digest in source_contract.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"{role} loader file is missing: {path}")
        digest = sha256_file(path)
        if digest != expected_digest:
            raise RuntimeError(
                f"{role} loader file digest mismatch for {relative}: "
                f"expected={expected_digest} actual={digest}"
            )
        observed[relative] = digest
    result: dict[str, Any] = {
        "role": role,
        "root": str(root),
        "source_digests": observed,
    }
    if role == "submission":
        main = root / "main.py"
        bundled_checkpoint = root / "model.pt"
        if not main.is_file() or not bundled_checkpoint.is_file():
            raise RuntimeError("submission stage lacks main.py or model.pt")
        main_text = main.read_text(encoding="utf-8")
        if "load_model_from_checkpoint" not in main_text:
            raise RuntimeError("submission entry point does not use the tested loader")
        expected_checkpoint_digest = sha256_file(checkpoint_path)
        actual_checkpoint_digest = sha256_file(bundled_checkpoint)
        if actual_checkpoint_digest != expected_checkpoint_digest:
            raise RuntimeError("submission model.pt is not the validated checkpoint")
        result.update(
            main_digest=sha256_file(main),
            bundled_checkpoint_digest=actual_checkpoint_digest,
        )
    return result


def build_fleet_compatibility_receipt(
    *,
    checkpoint_path: Path | str,
    reference_root: Path | str,
    roots: Mapping[str, Path | str],
    output_path: Path | str,
) -> Path:
    """Publish proof that every production loader can read the dormant bank."""

    missing = sorted(set(FLEET_ROLES) - {str(role).lower() for role in roots})
    extra = sorted({str(role).lower() for role in roots} - set(FLEET_ROLES))
    if missing or extra:
        raise RuntimeError(
            f"fleet loader roles must be exactly {FLEET_ROLES}: "
            f"missing={missing} extra={extra}"
        )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_contract = validate_zero_dormant_checkpoint(checkpoint_path)
    source_contract = loader_source_contract(Path(reference_root))
    devices = {
        role: validate_loader_root(
            Path(roots[role]),
            role=role,
            source_contract=source_contract,
            checkpoint_path=checkpoint_path,
        )
        for role in FLEET_ROLES
    }
    output = Path(output_path).expanduser().resolve()
    _immutable_json(
        output,
        {
            "schema": COMPATIBILITY_SCHEMA,
            "checkpoint": checkpoint_contract,
            "reference_root": str(Path(reference_root).expanduser().resolve()),
            "loader_source_contract": source_contract,
            "devices": devices,
            "runtime_enabled": False,
            "training_enabled": False,
            "all_roles_validated": True,
        },
    )
    return output
