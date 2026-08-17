#!/usr/bin/env python3
"""Create and verify a staging-only V5-to-V6 matchup adapter canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poke_bot.matchup_adapters_v6 import (
    LEGACY_V5_PREFIX_LENGTH,
    PARAMETERS_PER_SLOT,
    SLOT_CAPACITY,
    MatchupAdapterBankV6,
    load_slot_registry,
    migrate_v5_checkpoint_payload,
    registry_digest,
)
from poke_bot.matchup_adapters import MatchupAdapterBank as MatchupAdapterBankV5

RECEIPT_SCHEMA = "poke_bot.matchup_adapter_v6_canary_receipt/v1"
ADAPTER_PREFIX = "matchup_adapter_bank."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        return (
            isinstance(left, np.generic)
            and isinstance(right, np.generic)
            and left.dtype == right.dtype
            and bool(left == right)
        )
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(raw_path)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(raw_path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_canary(
    *,
    source: Path,
    registry_path: Path,
    output: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    output = output.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    if output == source:
        raise ValueError("V6 canary output must not replace the source checkpoint")

    source_stat = source.stat()
    source_digest = _sha256(source)
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    registry = load_slot_registry(registry_path)
    migrated_payload = migrate_v5_checkpoint_payload(
        source_payload,
        registry=registry,
    )
    _atomic_torch_save(migrated_payload, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)

    source_state = dict(source_payload.get("model_state_dict") or {})
    target_state = dict(loaded.get("model_state_dict") or {})
    source_base = {
        key: value
        for key, value in source_state.items()
        if not key.startswith(ADAPTER_PREFIX)
    }
    target_base = {
        key: value
        for key, value in target_state.items()
        if not key.startswith(ADAPTER_PREFIX)
    }
    if not _equal(source_base, target_base):
        raise RuntimeError("V6 canary changed a non-adapter model tensor")

    retained_tensors = 0
    zero_tensors = 0
    bank_state: dict[str, torch.Tensor] = {}
    for key, value in target_state.items():
        if not key.startswith(ADAPTER_PREFIX):
            continue
        local_name = key.removeprefix(ADAPTER_PREFIX)
        bank_state[local_name] = value
        route = int(local_name.split(".")[1])
        if route < LEGACY_V5_PREFIX_LENGTH:
            if not torch.equal(value, source_state[key]):
                raise RuntimeError(f"V6 canary changed retained adapter tensor: {key}")
            retained_tensors += 1
        else:
            if int(value.count_nonzero().item()) != 0:
                raise RuntimeError(f"V6 canary created a nonzero unused slot: {key}")
            zero_tensors += 1
    bank = MatchupAdapterBankV6(enabled=False, registry=registry)
    bank.load_state_dict(bank_state, strict=True)
    if any(parameter.requires_grad for parameter in bank.parameters()):
        raise RuntimeError("V6 canary bank is not frozen by default")
    source_bank_state = {
        key.removeprefix(ADAPTER_PREFIX): value
        for key, value in source_state.items()
        if key.startswith(ADAPTER_PREFIX)
    }
    source_bank = MatchupAdapterBankV5(enabled=True)
    source_bank.load_state_dict(source_bank_state, strict=True)
    bank.enabled = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260724)
    parity_states = torch.randn(
        LEGACY_V5_PREFIX_LENGTH + 1,
        bank.hidden_dim,
        generator=generator,
    )
    parity_routes = torch.tensor(
        [-1, *range(LEGACY_V5_PREFIX_LENGTH)],
        dtype=torch.long,
    )
    with torch.no_grad():
        source_outputs = source_bank(parity_states, parity_routes)
        target_outputs = bank(parity_states, parity_routes)
    if not torch.equal(source_outputs, target_outputs):
        raise RuntimeError("V6 canary changed retained-route forward outputs")
    bank.enabled = False

    for key in source_payload:
        if key in {"model_state_dict", "extra", "model_config"}:
            continue
        if not _equal(source_payload[key], loaded[key]):
            raise RuntimeError(f"V6 canary changed top-level checkpoint field: {key}")
    source_model_config = dict(source_payload.get("model_config") or {})
    target_model_config = dict(loaded.get("model_config") or {})
    source_adapter_format = source_model_config.pop(
        "matchup_adapter_format",
        None,
    )
    source_adapter_registry = source_model_config.pop(
        "matchup_adapter_registry",
        None,
    )
    target_adapter_format = target_model_config.pop("matchup_adapter_format", None)
    target_adapter_registry = target_model_config.pop(
        "matchup_adapter_registry",
        None,
    )
    if (
        source_adapter_format != "poke-bot-matchup-adapter-bank-v5-roster18"
        or source_adapter_registry is not None
        or target_adapter_format != "poke-bot-matchup-adapter-bank-v6"
        or target_adapter_registry != registry
        or target_model_config != source_model_config
    ):
        raise RuntimeError(
            "V6 canary model config changed beyond its format and registry binding"
        )
    if not _equal(
        source_payload.get("optimizer_state_dict"),
        loaded.get("optimizer_state_dict"),
    ):
        raise RuntimeError("V6 canary changed the main optimizer")
    if not _equal(source_payload.get("rng_state"), loaded.get("rng_state")):
        raise RuntimeError("V6 canary changed RNG state")

    source_extra = dict(source_payload.get("extra") or {})
    target_extra = dict(loaded.get("extra") or {})
    source_adapter_optimizer = dict(
        source_extra.get("dormant_matchup_adapter_optimizer_state") or {}
    )
    target_adapter_optimizer = dict(
        target_extra.get("dormant_matchup_adapter_optimizer_state") or {}
    )
    source_moments = dict(source_adapter_optimizer.get("state") or {})
    target_moments = dict(target_adapter_optimizer.get("state") or {})
    if not _equal(source_moments, target_moments):
        raise RuntimeError("V6 canary changed retained dormant optimizer moments")
    target_groups = list(target_adapter_optimizer.get("param_groups") or [])
    if source_adapter_optimizer and (
        len(target_groups) != 1
        or target_groups[0].get("params")
        != list(range(SLOT_CAPACITY * PARAMETERS_PER_SLOT))
        or set(target_moments) - set(range(LEGACY_V5_PREFIX_LENGTH * PARAMETERS_PER_SLOT))
    ):
        raise RuntimeError("V6 canary dormant optimizer expansion is invalid")

    if (
        _sha256(source) != source_digest
        or source.stat().st_size != source_stat.st_size
        or source.stat().st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise RuntimeError("source V5 checkpoint changed during the canary")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_digest,
        "source_checkpoint_size": source_stat.st_size,
        "source_checkpoint_mtime_ns": source_stat.st_mtime_ns,
        "source_immutable": True,
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": _sha256(output),
        "slot_registry": str(registry_path),
        "slot_registry_digest": registry_digest(registry),
        "legacy_slots_retained": LEGACY_V5_PREFIX_LENGTH,
        "slot_capacity": SLOT_CAPACITY,
        "retained_adapter_tensors_bit_exact": retained_tensors,
        "unused_adapter_tensors_exact_zero": zero_tensors,
        "non_adapter_model_tensors_bit_exact": len(source_base),
        "model_config_only_adapter_contract_changed": True,
        "model_config_registry_bound": True,
        "main_optimizer_bit_exact": True,
        "rng_state_bit_exact": True,
        "dormant_optimizer_moments_bit_exact": True,
        "dormant_optimizer_source_parameter_count": (
            LEGACY_V5_PREFIX_LENGTH * PARAMETERS_PER_SLOT
            if source_adapter_optimizer
            else 0
        ),
        "dormant_optimizer_target_parameter_count": (
            SLOT_CAPACITY * PARAMETERS_PER_SLOT
            if source_adapter_optimizer
            else 0
        ),
        "v6_bank_load_strict": True,
        "v6_bank_frozen_by_default": True,
        "v5_v6_retained_route_forward_bit_exact": True,
        "live_selector_changed": False,
    }
    _atomic_json(receipt, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_canary(
        source=args.source,
        registry_path=args.registry,
        output=args.output,
        receipt_path=args.receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
