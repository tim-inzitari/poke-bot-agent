#!/usr/bin/env python3
"""Register the completed Marnie H10 bootstrap for managed specialist RL."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import (  # noqa: E402
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_V3_MAX_RELIABILITY,
    DECISION_FUSION_V3_MIN_RELIABILITY,
    DECISION_FUSION_V3_ROUTE_SCHEMA,
    DECISION_FUSION_V3_SCHEMA,
)
from poke_bot.matchup_adapters_v6 import (  # noqa: E402
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
    LEGACY_V5_PREFIX_LENGTH,
    PARAMETERS_PER_SLOT,
    SLOT_CAPACITY,
    load_slot_registry,
    migrate_v5_checkpoint_payload,
    registry_digest,
)
from poke_bot.pure_rl.model_registry import freeze_model, verify_frozen_model  # noqa: E402
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS  # noqa: E402
from poke_bot.train import (  # noqa: E402
    GUIDE_TRAINING_MODE_DIRECTIONAL,
    assert_strategic_curriculum_receipt_contract,
    load_model_from_checkpoint,
)
from scripts.register_next_specialist_runtime import (  # noqa: E402
    _atomic_json,
    _atomic_selector,
    _guide_weight_policy,
)


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
READY_SCHEMA = "poke_bot.final_format_marnie_h10_bootstrap_ready/v1"
RECEIPT_SCHEMA = "poke_bot.final_format_marnie_h10_runtime_registration/v1"
SELECTOR_MIGRATION_SCHEMA = "poke_bot.marnie_selector_env_migration/v1"
DEFAULT_SELECTOR_MIGRATION_RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final-format-marnie-r104-selector-prefetch-r125.json"
)
DEFAULT_SELF_PLAY_SCHEDULER_DROPIN = Path(
    "/home/inzi/.config/systemd/user/"
    "pokebot-final-format-marnie-r104-h10-rl.service.d/"
    "zzzz-self-play-elmo-tail-r119.conf"
)
AUTH_SCHEMA = "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1"
ROUTER_V6_MIGRATION_SCHEMA = (
    "poke_bot.final_format_marnie_h10_router_v6_migration/v1"
)
REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
MARNIE_GATE_ID = (
    "specialist-strong-public-roster-sw80-at-iter5-v1+frozen-specialists-r14-r109"
)
RUNTIME_ASSET_SHA256 = {
    "submission/main.py": (
        "sha256:30deadd1d8dc7e3d885dd107600468e5d0dcd6e9227c86fba52504e85fe0b70f"
    ),
    "submission/search_config.json": (
        "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
    ),
    "data/training_mixes/top_ladder.v1.json": (
        "sha256:de9f2f5f65794ed8f0ffd3fff41b04aafd68f7ffc9d562ab2cfc774ea5cac79d"
    ),
    "data/training_mixes/top_ladder_representatives.v1.json": (
        "sha256:ee42f146fd746ed3dd953515a974b22eeb264ea8eb9513c6e69a41a524454002"
    ),
    "data/training_mixes/specialist_representatives.v1.json": (
        "sha256:47f624f35ab5f497055b3059cdf6dfd7d7e06312ccf3e87b01910d6b142c3a35"
    ),
}
REQUIRED_HEADS = (*DECISION_FUSION_REQUIRED_HEADS, "setup_board_outcome", "combo_state")
REQUIRED_TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable JSON projection, or verify the existing bytes."""

    if path.exists():
        existing = _read(path)
        if existing != payload:
            raise RuntimeError(f"immutable JSON projection changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != _sha256(source):
            raise RuntimeError(f"immutable source-family checkpoint changed: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.link(temporary, target)
        temporary.unlink(missing_ok=True)


def _materialize_adapter_authorization(
    *,
    state_root: Path,
    router_checkpoint: Path,
    router_checkpoint_digest: str,
    expert_checkpoint: Path,
    expert_checkpoint_digest: str,
) -> Path:
    """Bind Router-6 to the inherited 25-epoch Marnie training evidence."""

    root = state_root / "final-format-marnie-r104-h10-adapter-authorization-v2"
    source_family = root / "ordinary_source"
    source_model = source_family / "model.pt"
    _link_or_copy(expert_checkpoint, source_model)

    source_manifest = source_family / "manifest.json"
    _write_json_once(
        source_manifest,
        {
            "schema": "poke_bot.final_format_marnie_h10_training_manifest/v1",
            "checkpoint_digest": expert_checkpoint_digest,
            "model_path": str(source_model.resolve(strict=False)),
            "provenance": {
                "acting_seat_archetype": SPECIALIST_ID,
                "epochs_max": 25,
                "trained_target_coverage": list(REQUIRED_TARGETS),
                "final_format_same_archetype_refresh": True,
            },
            "evidence": {
                "epochs_completed": 25,
                "expert_checkpoint_sha256": expert_checkpoint_digest,
                "expert_parent_preserved": True,
            },
        },
    )
    source_manifest_digest = _sha256(source_manifest)

    derivative_manifest = root / "router_v6_derivative_manifest.json"
    _write_json_once(
        derivative_manifest,
        {
            "schema": "poke_bot.final_format_marnie_h10_router_v6_training_manifest/v1",
            "checkpoint_digest": router_checkpoint_digest,
            "model_path": str(router_checkpoint.resolve()),
            "provenance": {
                "kind": "matchup_adapter_v6_runtime_derivative",
                "source_family": str(source_family.resolve()),
                "source_family_immutable": True,
                "source_family_manifest_sha256": source_manifest_digest,
                "source_checkpoint_digest": expert_checkpoint_digest,
            },
            "evidence": {
                "training_evidence_inherited_from_source": True,
                "router_v6_new_slots_zero_safe_pending_rl_training": True,
                "matchup_adapter_bank_dormant_at_launch": True,
            },
        },
    )
    derivative_manifest_digest = _sha256(derivative_manifest)

    authorization = root / "authorization.json"
    _write_json_once(
        authorization,
        {
            "schema": AUTH_SCHEMA,
            "specialist_id": SPECIALIST_ID,
            "completed_iteration": -1,
            "first_eligible_iteration": 0,
            "parent_checkpoint": str(router_checkpoint.resolve()),
            "parent_checkpoint_digest": router_checkpoint_digest,
            "protected_manifest": str(derivative_manifest.resolve()),
            "protected_manifest_digest": derivative_manifest_digest,
            "runtime_enabled": False,
            "optimizer_scope": "matchup_adapter_bank_only",
            "parent_untouched": True,
            "purpose": "final-format-marnie-h10-router6-adapter-fitting",
            "required_target_coverage": list(REQUIRED_TARGETS),
        },
    )
    return authorization


def _validate_runtime_assets(deployment: Path) -> dict[str, str]:
    assets: dict[str, str] = {}
    for relative, expected in RUNTIME_ASSET_SHA256.items():
        path = deployment / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(
                "Marnie deployment lacks a checksum-bound runtime training-mix asset: "
                + relative
            )
        assets[relative] = expected
    return assets


def _selector_values(path: Path) -> dict[str, str]:
    """Parse a selector while rejecting duplicate active assignments."""

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate selector assignment: {key}")
        values[key] = value
    return values


def _unit_environment_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row.startswith("Environment=") or "=" not in row.removeprefix(
            "Environment="
        ):
            continue
        key, value = row.removeprefix("Environment=").split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate service environment assignment: {key}")
        values[key] = value
    return values


def _selector_env_is_authorized(
    *,
    registration_receipt: Path,
    selector_path: Path,
    expected_digest: str,
    deployment: Path,
    migration_receipt: Path | None = None,
    scheduler_dropin: Path | None = None,
) -> bool:
    """Accept the base selector or the one checksum-bound revision-125 repair."""

    actual_digest = _sha256(selector_path)
    if actual_digest == expected_digest:
        return True
    migration_path = Path(
        migration_receipt
        or os.environ.get(
            "POKEBOT_MARNIE_SELECTOR_MIGRATION_RECEIPT",
            str(DEFAULT_SELECTOR_MIGRATION_RECEIPT),
        )
    ).expanduser().resolve()
    if not migration_path.is_file():
        return False
    migration = _read(migration_path)
    exact_environment = {
        "POKEBOT_ACTIVE_SPECIALIST": SPECIALIST_ID,
        "POKEBOT_SPECIALIST_RUNTIME_ROOT": str(deployment),
        "PYTHONPATH": str(deployment),
        "POKEBOT_REMOTE_SOCKET_PREFETCH": "1",
        "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX": "1",
    }
    exact_service_environment = {
        "POKEBOT_REMOTE_SOCKET_PREFETCH": "1",
        "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX": "1",
        "POKEBOT_SELF_PLAY_ELMO_TAIL_ONLY": "1",
        "POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_GAMES": "20",
    }
    selector_values = _selector_values(selector_path)
    dropin_path = Path(
        scheduler_dropin or DEFAULT_SELF_PLAY_SCHEDULER_DROPIN
    ).expanduser().resolve()
    if not dropin_path.is_file():
        return False
    service_values = _unit_environment_values(dropin_path)
    return bool(
        migration.get("schema") == SELECTOR_MIGRATION_SCHEMA
        and migration.get("status") == "activated_at_stopped_uncommitted_boundary"
        and int(migration.get("goal_revision") or 0) == 125
        and migration.get("specialist_id") == SPECIALIST_ID
        and migration.get("run_name")
        == "final_format_marnie_r104_h10_i_v6_8k"
        and Path(str(migration.get("base_registration_receipt") or "")).resolve()
        == registration_receipt.resolve()
        and migration.get("base_registration_sha256")
        == _sha256(registration_receipt)
        and migration.get("base_selector_env_sha256") == expected_digest
        and Path(str(migration.get("selector_env") or "")).resolve()
        == selector_path.resolve()
        and migration.get("selector_env_sha256") == actual_digest
        and migration.get("exact_environment") == exact_environment
        and all(
            selector_values.get(key) == value
            for key, value in exact_environment.items()
        )
        and Path(str(migration.get("scheduler_dropin") or "")).resolve()
        == dropin_path
        and migration.get("scheduler_dropin_sha256") == _sha256(dropin_path)
        and migration.get("exact_service_environment")
        == exact_service_environment
        and all(
            service_values.get(key) == value
            for key, value in exact_service_environment.items()
        )
        and migration.get("prior_attempt_rejected") is True
        and int(migration.get("observed_remote_sockets") or 0) == 104
        and int(migration.get("maximum_remote_sockets") or 0) == 52
    )


def _validate_selected_bootstrap_training(payload: dict[str, Any]) -> None:
    """Require the selected bootstrap child to carry full-head epoch evidence."""

    extra = dict(payload.get("extra") or {})
    bootstrap = dict(extra.get("final_format_marnie_h10_bootstrap") or {})
    expanded = dict(extra.get("expanded_head_training") or {})
    heads = dict(expanded.get("heads") or {})
    epoch = int(bootstrap.get("epoch") or 0)
    if (
        bootstrap.get("schema")
        != "poke_bot.final_format_marnie_h10_bootstrap_epoch/v1"
        or not 16 <= epoch <= 25
        or bootstrap.get("guide_mode") != GUIDE_TRAINING_MODE_DIRECTIONAL
        # This validator is also used by the immutable Router-6 registration
        # receipt, whose checksum-pinned bootstrap predates guide retirement
        # and therefore records the historical 0.05 training weight.  That
        # metadata is audit-only: the active r142 registry and post-upload
        # activation receipt independently require runtime/training authority
        # to be zero.  Accept both exact historical and retired encodings here.
        or float(bootstrap.get("guide_weight", -1.0)) not in {0.0, 0.05}
        or bootstrap.get("decision_fusion_schema") != DECISION_FUSION_V3_SCHEMA
        or expanded.get("schema") != "poke_bot.expanded_head_training/v1"
        or int(expanded.get("epoch") or 0) != epoch
        or int(expanded.get("epochs_total") or 0) != 25
        or set(expanded.get("gradient_enabled_heads") or [])
        != set(EXPANDED_HEAD_IDS)
        or set(expanded.get("trained_this_epoch") or [])
        != set(EXPANDED_HEAD_IDS)
        or set(heads) != set(EXPANDED_HEAD_IDS)
    ):
        raise RuntimeError(
            "Marnie selected bootstrap checkpoint lacks exact full-head epoch evidence"
        )
    for name in EXPANDED_HEAD_IDS:
        row = dict(heads[name] or {})
        if (
            row.get("present") is not True
            or row.get("gradient_enabled") is not True
            or row.get("trained") is not True
            or row.get("trained_this_epoch") is not True
            or int(row.get("train_labeled_rows") or 0) <= 0
            or int(row.get("validation_labeled_rows") or 0) <= 0
        ):
            raise RuntimeError(
                f"Marnie selected bootstrap head lacks training evidence: {name}"
            )


def _route_reliability_telemetry(payload: dict[str, Any]) -> dict[str, float]:
    state = dict(payload.get("model_state_dict") or {})
    prefix = "decision_fusion.dedicated_route_log_reliability."
    raw = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if set(raw) != set(REQUIRED_HEADS):
        raise RuntimeError("Marnie checkpoint route-reliability roster changed")
    effective: dict[str, float] = {}
    for name in REQUIRED_HEADS:
        value = raw[name]
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise RuntimeError(
                f"Marnie route reliability is not a scalar tensor: {name}"
            )
        log_reliability = float(value.detach().cpu().item())
        if not math.isfinite(log_reliability):
            raise RuntimeError(f"Marnie route reliability is not finite: {name}")
        reliability = math.exp(
            min(
                math.log(DECISION_FUSION_V3_MAX_RELIABILITY),
                max(math.log(DECISION_FUSION_V3_MIN_RELIABILITY), log_reliability),
            )
        )
        if name == "action_type":
            reliability = min(reliability, 0.25)
        effective[name] = reliability
    return effective


def _non_adapter_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in dict(payload.get("model_state_dict") or {}).items()
        if not name.startswith("matchup_adapter_bank.")
    }


def _validate_router_v6_checkpoint(
    path: Path,
    expected_digest: str,
    registry_path: Path,
) -> dict[str, Any]:
    if checkpoint.checkpoint_digest(path) != expected_digest:
        raise RuntimeError("Marnie Router-6 checkpoint digest changed")
    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    registry = load_slot_registry(registry_path)
    config = dict((payload.get("extra") or {}).get("matchup_adapter_config") or {})
    state = dict(payload.get("model_state_dict") or {})
    physical_slots = {
        int(name.split(".")[2])
        for name in state
        if name.startswith("matchup_adapter_bank.experts.")
        and len(name.split(".")) >= 4
    }
    if (
        config.get("format") != V6_ADAPTER_CHECKPOINT_FORMAT
        or int(config.get("slot_capacity") or 0) != SLOT_CAPACITY
        or config.get("slot_registry") != registry
        or config.get("slot_registry_digest") != registry_digest(registry)
        or physical_slots != set(range(SLOT_CAPACITY))
        or dict(payload.get("model_config") or {}).get("matchup_adapter_format")
        != V6_ADAPTER_CHECKPOINT_FORMAT
    ):
        raise RuntimeError("Marnie checkpoint is not exact 64-slot Router Format 6")
    return payload


def _materialize_router_v6_derivative(
    *,
    source_checkpoint: Path,
    source_digest: str,
    registry_path: Path,
    family_dir: Path,
    receipt_path: Path,
) -> tuple[Path, str, dict[str, Any]]:
    if checkpoint.checkpoint_digest(source_checkpoint) != source_digest:
        raise RuntimeError("Marnie Router-6 migration source digest changed")
    source_payload = checkpoint.load_checkpoint(source_checkpoint, map_location="cpu")
    registry = load_slot_registry(registry_path)
    migrated = migrate_v5_checkpoint_payload(source_payload, registry=registry)
    source_state = dict(source_payload.get("model_state_dict") or {})
    target_state = dict(migrated.get("model_state_dict") or {})
    source_non_adapter = _non_adapter_state(source_payload)
    target_non_adapter = _non_adapter_state(migrated)
    if source_non_adapter.keys() != target_non_adapter.keys() or any(
        not torch.equal(source_non_adapter[name], target_non_adapter[name])
        for name in source_non_adapter
    ):
        raise RuntimeError("Marnie Router-6 migration changed a non-adapter tensor")
    retained_tensors = 0
    appended_tensors = 0
    for name, value in target_state.items():
        if not name.startswith("matchup_adapter_bank.experts."):
            continue
        slot = int(name.split(".")[2])
        if slot < LEGACY_V5_PREFIX_LENGTH:
            if name not in source_state or not torch.equal(value, source_state[name]):
                raise RuntimeError("Marnie Router-6 migration changed a retained slot")
            retained_tensors += 1
        else:
            if int(value.count_nonzero().item()) != 0:
                raise RuntimeError("Marnie Router-6 migration appended nonzero state")
            appended_tensors += 1
    expected_appended = (
        SLOT_CAPACITY - LEGACY_V5_PREFIX_LENGTH
    ) * PARAMETERS_PER_SLOT
    if (
        retained_tensors != LEGACY_V5_PREFIX_LENGTH * PARAMETERS_PER_SLOT
        or appended_tensors != expected_appended
    ):
        raise RuntimeError("Marnie Router-6 migration tensor inventory changed")

    with tempfile.TemporaryDirectory(prefix="marnie-h10-router-v6-") as raw:
        candidate = Path(raw) / "model.pt"
        torch.save(migrated, candidate)
        candidate_digest = checkpoint.checkpoint_digest(candidate)
        frozen = freeze_model(
            registry_root=family_dir.parent,
            family=family_dir.name,
            display_name="Marnie's Grimmsnarl ex H10 Router Format 6 Bootstrap",
            checkpoint=candidate,
            expected_digest=candidate_digest,
            provenance={
                "specialist_id": SPECIALIST_ID,
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": source_digest,
                "adapter_format": V6_ADAPTER_CHECKPOINT_FORMAT,
                "slot_capacity": SLOT_CAPACITY,
                "slot_registry": str(registry_path),
                "slot_registry_digest": registry_digest(registry),
                "runtime_authority": "none_until_managed_registration",
            },
            evidence={
                "kind": "router_v6_function_preserving_derivative",
                "retained_v5_adapter_tensors_bit_exact": retained_tensors,
                "appended_adapter_tensors_exact_zero": appended_tensors,
                "non_adapter_model_tensors_bit_exact": len(target_non_adapter),
            },
            harden_permissions=True,
        )
    target = Path(str(frozen["model_path"])).resolve()
    target_digest = str(frozen["checkpoint_digest"])
    _validate_router_v6_checkpoint(target, target_digest, registry_path)
    migration = {
        "schema": ROUTER_V6_MIGRATION_SCHEMA,
        "status": "passed_ready_for_managed_registration",
        "specialist_id": SPECIALIST_ID,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_digest,
        "target_checkpoint": str(target),
        "target_checkpoint_sha256": target_digest,
        "adapter_format": V6_ADAPTER_CHECKPOINT_FORMAT,
        "slot_capacity": SLOT_CAPACITY,
        "slot_registry": str(registry_path),
        "slot_registry_digest": registry_digest(registry),
        "retained_v5_adapter_tensors_bit_exact": retained_tensors,
        "appended_adapter_tensors_exact_zero": appended_tensors,
        "non_adapter_model_tensors_bit_exact": len(target_non_adapter),
        "step_zero_policy_behavior_preserved": True,
        "runtime_authority": False,
        "selector_authority": False,
    }
    _atomic_json(receipt_path, migration)
    return target, target_digest, migration


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _replace_arg(values: list[str], flag: str, replacement: str) -> None:
    try:
        index = values.index(flag)
    except ValueError as exc:
        raise RuntimeError(f"runtime template lacks {flag}") from exc
    if index + 1 >= len(values):
        raise RuntimeError(f"runtime template has no value for {flag}")
    values[index + 1] = replacement


def _materialize_marnie_gate(deployment: Path) -> tuple[Path, dict[str, Any]]:
    """Bind the ordinary LC50 criteria to the complete frozen-r14 roster."""

    ordinary_path = deployment / "ops/alakazam_gate_program_v1.json"
    roster_path = deployment / "ops/final_format_alakazam_gate_r94_v4.json"
    frozen_path = (
        deployment
        / "ops/frozen_specialist_registry_marnie_r108_h10_alakazam.json"
    )
    ordinary = _read(ordinary_path)
    roster_contract = _read(roster_path)
    frozen_registry = _read(frozen_path)
    ordinary_gate = copy.deepcopy(dict(ordinary.get("next_gate") or {}))
    roster_gate = dict(roster_contract.get("next_gate") or {})
    frozen = [
        dict(row)
        for row in frozen_registry.get("specialists") or []
        if row.get("frozen") is True and row.get("public_mix_eligible") is True
    ]
    roster = [dict(row) for row in roster_gate.get("roster") or []]
    scoped_alakazam = [
        row for row in frozen if str(row.get("specialist_id") or "") == "alakazam"
    ]
    historical_alakazam_gate = [
        row
        for row in roster
        if row.get("frozen_specialist") is True
        and str(row.get("archetype_id") or "") == "alakazam"
    ]
    if len(scoped_alakazam) != 1 or len(historical_alakazam_gate) != 1:
        raise RuntimeError("Marnie Alakazam roster replacement is ambiguous")
    replacement = copy.deepcopy(historical_alakazam_gate[0])
    replacement.update(
        {
            "opponent_id": scoped_alakazam[0]["opponent_id"],
            "archetype_id": scoped_alakazam[0]["archetype_id"],
            "archetype_label": scoped_alakazam[0]["archetype_label"],
            "source": scoped_alakazam[0]["source"],
            "frozen_checkpoint_digest": scoped_alakazam[0]["checkpoint_digest"],
            "content_digest": scoped_alakazam[0]["content_digest"],
        }
    )
    replacement.pop("kaggle_rating_anchor", None)
    replacement.pop("kaggle_rating_anchor_source", None)
    roster = [
        row
        for row in roster
        if not (
            row.get("frozen_specialist") is True
            and str(row.get("archetype_id") or "") == "alakazam"
        )
    ] + [replacement]
    gate_frozen = {
        str(row.get("opponent_id") or ""): row
        for row in roster
        if row.get("frozen_specialist") is True
    }
    frozen_by_id = {
        str(row.get("opponent_id") or ""): row for row in frozen
    }
    if (
        ordinary.get("schema") != "poke_bot.competition_gate_program/v1"
        or roster_contract.get("schema")
        != "poke_bot.competition_gate_program/v1"
        or frozen_registry.get("schema")
        != "poke_bot.frozen_specialist_registry/v1"
        or len(frozen) != 14
        or len(roster) != 17
        or set(gate_frozen) != set(frozen_by_id)
        or any(
            gate_frozen[opponent_id].get("frozen_checkpoint_digest")
            != row.get("checkpoint_digest")
            for opponent_id, row in frozen_by_id.items()
        )
    ):
        raise RuntimeError("Marnie frozen-r14 gate source contract changed")
    ordinary_gate["id"] = MARNIE_GATE_ID
    ordinary_gate["label"] = (
        "Strong public-agent gate + frozen S+ specialists "
        "(50% lower-confidence bound from iteration 5)"
    )
    ordinary_gate["roster"] = roster
    ordinary_gate["evaluation"] = copy.deepcopy(roster_gate["evaluation"])
    ordinary_gate["exact_result_pointer"] = (
        "/home/inzi/poke-bot-agent/outputs/state/"
        "final-format-marnie-r104-h10-active-gate-result.json"
    )
    criteria = dict(ordinary_gate.get("pass_criteria") or {})
    criteria["skill_weighted_confidence_lower"] = 0.50
    criteria["skill_weighted_win_rate"] = 0.80
    ordinary_gate["pass_criteria"] = criteria
    derived = copy.deepcopy(ordinary)
    derived["active_gate_id"] = MARNIE_GATE_ID
    derived["next_gate"] = ordinary_gate
    derived["active_gate_semantics"] = copy.deepcopy(
        roster_contract.get("active_gate_semantics") or {}
    )
    derived.pop("fallback_transition", None)
    # Revision 108 owns the scoped H10 Alakazam roster. Revision 109 owns the
    # terminal-strength criterion and the requirement that candidate rejection
    # must not block the iteration-1 -> iteration-2 continuation boundary.
    derived["owner_decision_revision"] = 109
    derived["derivation"] = {
        "schema": "poke_bot.final_format_marnie_gate_derivation/v1",
        "ordinary_lc50_source": str(ordinary_path),
        "ordinary_lc50_source_sha256": _sha256(ordinary_path),
        "frozen_r14_roster_source": str(roster_path),
        "frozen_r14_roster_source_sha256": _sha256(roster_path),
        "frozen_registry": str(frozen_path),
        "frozen_registry_sha256": _sha256(frozen_path),
        "criteria_preserved_except_active_lc50_materialization": True,
    }
    output = deployment / "runtime/final_format_marnie_gate_r108_h10_alakazam.json"
    _atomic_json(output, derived)
    return output, derived


def _validate_checkpoint(
    path: Path,
    expected_digest: str,
    *,
    require_bootstrap_training: bool = True,
) -> dict[str, float]:
    if checkpoint.checkpoint_digest(path) != expected_digest:
        raise RuntimeError("Marnie bootstrap checkpoint digest changed")
    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    try:
        _validate_selected_bootstrap_training(payload)
    except RuntimeError as exc:
        if not require_bootstrap_training:
            # The terminal RL learner is validated for architecture/runtime
            # below; bootstrap epoch provenance belongs only to the separate
            # expert-bootstrap checkpoint.
            pass
        else:
        # The immutable H10 freeze intentionally removes optimizer/training
        # extras from the serving payload. Preserve the exact full-head epoch
        # proof by validating the checksum-bound source epoch named by the
        # protected package manifest; never accept a free-standing sidecar or
        # mutable dashboard claim as a substitute.
            manifest_path = path.parent / "manifest.json"
            manifest = _read(manifest_path) if manifest_path.is_file() else {}
            provenance = dict(manifest.get("provenance") or {})
            source_path = Path(str(provenance.get("source_checkpoint") or "")).expanduser()
            source_path = source_path.resolve() if source_path else Path()
            if not source_path.is_file():
                raise exc
            source_payload = checkpoint.load_checkpoint(source_path, map_location="cpu")
            _validate_selected_bootstrap_training(source_payload)
            source_digest = checkpoint.checkpoint_digest(source_path)
            source_manifest_path = Path(str(provenance.get("source_checkpoint") or ""))
            if source_manifest_path.resolve() != source_path or source_digest == expected_digest:
                raise RuntimeError("Marnie source epoch evidence is not distinct and checksum-bound")
    route_reliabilities = _route_reliability_telemetry(payload)
    model = load_model_from_checkpoint(path, device=torch.device("cpu"))
    cfg = model.cfg
    inventory = model.decision_fusion_inventory()
    routes = dict(inventory.get("dedicated_routes") or {})
    if (
        (cfg.spatial_layers, cfg.temporal_layers, cfg.option_decoder_layers)
        != (7, 3, 7)
        or cfg.ff_dim != 2496
        or cfg.h10_head_residual_width != 512
        or inventory.get("schema") != DECISION_FUSION_V3_SCHEMA
        or inventory.get("runtime_enabled") is not True
        or inventory.get("guide_excluded") is not True
        or list(inventory.get("required_heads") or []) != list(REQUIRED_HEADS)
        or routes.get("schema") != DECISION_FUSION_V3_ROUTE_SCHEMA
        or routes.get("enabled") is not True
        or routes.get("runtime_enabled") is not True
        or int(routes.get("route_count") or 0) != len(REQUIRED_HEADS)
        or list(routes.get("route_names") or []) != list(REQUIRED_HEADS)
        or routes.get("reliability_bounds")
        != [DECISION_FUSION_V3_MIN_RELIABILITY, DECISION_FUSION_V3_MAX_RELIABILITY]
        or float(routes.get("action_type_reliability_cap") or -1.0) != 0.25
    ):
        raise RuntimeError("Marnie bootstrap checkpoint is not exact H10/Fusion-v3")
    return route_reliabilities


def register(args: argparse.Namespace) -> dict[str, Any]:
    ready_path = args.bootstrap_ready.expanduser().resolve()
    family = args.bootstrap_family.expanduser().resolve()
    deployment = args.deployment_root.expanduser().resolve()
    runtime_assets = _validate_runtime_assets(deployment)
    template_path = args.template_registry.expanduser().resolve()
    output_registry = args.output_registry.expanduser().resolve()
    template_selector = args.template_selector_env.expanduser().resolve()
    selector = args.selector_env.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    ready = _read(ready_path)
    frozen = verify_frozen_model(family)
    source_checkpoint_path = Path(str(frozen["model_path"])).resolve()
    source_checkpoint_digest = str(frozen["checkpoint_digest"])
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("status") != "ready_for_managed_rl_registration"
        or ready.get("specialist_id") != SPECIALIST_ID
        or int(ready.get("epochs_completed") or 0) != 25
        or ready.get("capacity_profile") != "H10-I/v1"
        or ready.get("decision_fusion_schema") != DECISION_FUSION_V3_SCHEMA
        or int(ready.get("learned_head_count") or 0) != 19
        or int(ready.get("learned_route_count") or 0) != 19
        or ready.get("guide_mode") != GUIDE_TRAINING_MODE_DIRECTIONAL
        or float(ready.get("guide_weight", -1.0)) != 0.0
        or ready.get("guide_enabled") is not False
        or ready.get("training_authority") is not False
        or ready.get("selector_authority") is not False
        or Path(str(ready.get("checkpoint") or "")).resolve()
        != source_checkpoint_path
        or ready.get("checkpoint_sha256") != source_checkpoint_digest
    ):
        raise RuntimeError("Marnie bootstrap-ready receipt is not authoritative")
    _validate_checkpoint(source_checkpoint_path, source_checkpoint_digest)
    checkpoint_path, checkpoint_digest, router_v6_migration = (
        _materialize_router_v6_derivative(
            source_checkpoint=source_checkpoint_path,
            source_digest=source_checkpoint_digest,
            registry_path=args.router_v6_registry.expanduser().resolve(),
            family_dir=args.router_v6_family.expanduser().resolve(),
            receipt_path=args.router_v6_receipt.expanduser().resolve(),
        )
    )
    route_reliabilities = _validate_checkpoint(checkpoint_path, checkpoint_digest)

    guide = args.guide.expanduser().resolve()
    expert = args.expert_manifest.expanduser().resolve()
    tree = args.matchup_tree.expanduser().resolve()
    curriculum = args.curriculum_spec.expanduser().resolve()
    roles = args.head_role_map.expanduser().resolve()
    validation = args.curriculum_validation.expanduser().resolve()
    for path in (guide, expert, tree, curriculum, roles, validation):
        if not path.is_file():
            raise RuntimeError(f"Marnie runtime input is missing: {path}")
    assert_strategic_curriculum_receipt_contract(
        specialist_id=SPECIALIST_ID,
        expected_training_mode=GUIDE_TRAINING_MODE_DIRECTIONAL,
        curriculum_spec=str(curriculum),
        head_role_map=str(roles),
        validation_receipt=str(validation),
    )
    role_payload = _read(roles)
    if (
        role_payload.get("decision_fusion_schema") != DECISION_FUSION_V3_SCHEMA
        or role_payload.get("canonical_learned_decision_sources") != list(REQUIRED_HEADS)
        or set(dict(role_payload.get("heads") or {})) != set(REQUIRED_HEADS)
    ):
        raise RuntimeError("Marnie role map is not the exact 19-route H10 contract")

    registry = copy.deepcopy(_read(template_path))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("runtime template schema changed")
    template_row = dict(dict(registry.get("specialists") or {}).get("alakazam") or {})
    if template_row.get("decision_fusion", {}).get("schema") != DECISION_FUSION_V3_SCHEMA:
        raise RuntimeError("runtime template is not the accepted H10/Fusion-v3 registry")
    trainer_args = list(registry.get("common_trainer_args") or [])
    _replace_arg(trainer_args, "--games-per-iter", "8192")
    _replace_arg(trainer_args, "--train-epochs", "5")
    _replace_arg(trainer_args, "--expert-rehearsal-epochs", "5")
    active_gate_path, active_gate = _materialize_marnie_gate(deployment)
    registry.update(
        {
            "runtime_root": str(deployment),
            "owner_decision_revision": 113,
            "active_gate_contract": str(active_gate_path.relative_to(deployment)),
            "frozen_specialist_registry": (
                "ops/frozen_specialist_registry_marnie_r108_h10_alakazam.json"
            ),
            "common_trainer_args": trainer_args,
            "minimum_terminal_iteration": 5,
            "iteration_ceiling": 20,
            "terminal_active_gate_id": MARNIE_GATE_ID,
        }
    )
    isolated = dict(registry.get("isolated_refresh_contract") or {})
    isolated.update(
        {
            "schema": "poke_bot.final_format_marnie_h10_isolated_runtime/v1",
            "specialist_id": SPECIALIST_ID,
            "checkpoint_sha256": checkpoint_digest,
            "games_per_iteration": 8192,
            "learner_epochs_per_iteration": 5,
            "minimum_terminal_iteration": 5,
            "maximum_iterations": 21,
            "maximum_training_games": 172032,
            "production_selector_write_authority": True,
        }
    )
    registry["isolated_refresh_contract"] = isolated
    top_handler = dict(registry.get("pass_handler") or {})
    top_handler.update(
        {
            "training_service": "pokebot-final-format-marnie-r104-h10-rl.service",
            "default_turn_order_preference": "first_if_allowed",
            "submission_count": 1,
        }
    )
    registry["pass_handler"] = top_handler

    expert_checkpoint = family / "model.pt"
    auth_path = _materialize_adapter_authorization(
        state_root=receipt_path.parent,
        router_checkpoint=checkpoint_path,
        router_checkpoint_digest=checkpoint_digest,
        expert_checkpoint=expert_checkpoint,
        expert_checkpoint_digest=checkpoint.checkpoint_digest(expert_checkpoint),
    )
    guide_policy = _guide_weight_policy(deployment)
    guide_policy.update(
        {
            "decision_fusion_schema": DECISION_FUSION_V3_SCHEMA,
            "canonical_learned_decision_sources": list(REQUIRED_HEADS),
            "route_input": "option_hidden_plus_typed_output_center",
            "route_reduction": "learned_positive_reliability",
        }
    )
    row = copy.deepcopy(template_row)
    row.update(
        {
            "status": "ready",
            "reason": None,
            "run_name": "final_format_marnie_r104_h10_i_v6_8k",
            "log": "/home/inzi/poke-bot-agent/outputs/final_format_marnie_r104/logs/h10_rl.log",
            "initial_checkpoint": str(checkpoint_path),
            "initial_checkpoint_sha256": checkpoint_digest.removeprefix("sha256:"),
            "expert_manifest": str(expert),
            "expert_manifest_sha256": _sha256(expert).removeprefix("sha256:"),
            "expert_minimum_decisions": 100000,
            "expert_required_target_coverage": list(REQUIRED_TARGETS),
            "matchup_runtime_tree": str(tree),
            "matchup_runtime_tree_sha256": _sha256(tree).removeprefix("sha256:"),
            "matchup_adapter_authorization": str(auth_path),
            "matchup_adapter_authorization_sha256": _sha256(auth_path).removeprefix("sha256:"),
            "measurement_decks": SPECIALIST_ID,
            "guide_id": SPECIALIST_ID,
            "guide_loss_weight": 0.0,
            "guide_retired": True,
            "guide_retirement_revision": 140,
            "guide_target_generation_required": False,
            "guide_conditioned_losses_enabled": False,
            "guide_action_influence": False,
            "guide_historical_artifacts": "audit_only",
            "guide_contract": str(guide),
            "guide_contract_sha256": _sha256(guide).removeprefix("sha256:"),
            "guide_version": "marnie-grimmsnarl-north-star-v1",
            "guide_training_mode": GUIDE_TRAINING_MODE_DIRECTIONAL,
            "guide_weight_policy": guide_policy,
            "setup_board_outcome_loss_weight": 0.025,
            "combo_state_loss_weight": 0.025,
            "minimum_terminal_iteration": 5,
            "iteration_ceiling": 20,
            "terminal_gate_marker": "SPECIALIST_GATE_PASSED.marnie-final-h10-r104",
            "decision_fusion": {
                "schema": DECISION_FUSION_V3_SCHEMA,
                "required": True,
                "runtime_enabled": True,
                "required_heads": list(REQUIRED_HEADS),
                "typed_output_centered_routes": True,
                "route_reliability_bounds": [0.25, 4.0],
                "action_type_reliability_cap": 0.25,
            },
            "strategic_curriculum": {
                "schema": "poke_bot.specialist_guide_training_contract/v1",
                "training_mode": GUIDE_TRAINING_MODE_DIRECTIONAL,
                "guide_curriculum_revision": 104,
                "strategic_branch_scope_revision": 56,
                "action_influence_revision": 104,
                "curriculum_spec": str(curriculum),
                "curriculum_spec_sha256": _sha256(curriculum).removeprefix("sha256:"),
                "head_role_map": str(roles),
                "head_role_map_sha256": _sha256(roles).removeprefix("sha256:"),
                "validation_receipt": str(validation),
                "validation_receipt_sha256": _sha256(validation).removeprefix("sha256:"),
                "decision_fusion_schema": DECISION_FUSION_V3_SCHEMA,
                "typed_output_centered_routes": True,
                "route_reliability_bounds": [0.25, 4.0],
                "action_type_reliability_cap": 0.25,
                "require_all_registered_learned_sources": True,
                "final_policy_logits_are_guide_targets": False,
            },
            "pass_handler": {
                "family": "final-format-marnie-r104-h10-refresh-v1",
                "display_name": "Marnie's Grimmsnarl ex Final-Format H10 Refresh Champion",
                "submission_root": "/home/inzi/poke-bot-agent/outputs/submissions/final-format-marnie-r104-h10-refresh-v1",
                "state": "/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-passed-gate-handler-v1.json",
                "lock": "/home/inzi/.local/state/pokebot/final-format-marnie-r104-h10-passed-gate-handler-v1.lock",
                "handoff_service": "pokebot-final-format-marnie-r104-completion.service",
            },
        }
    )
    registry["specialists"] = {SPECIALIST_ID: row}
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_registry, registry)
    if not template_selector.is_file():
        raise RuntimeError(f"runtime selector template is missing: {template_selector}")
    selector.parent.mkdir(parents=True, exist_ok=True)
    if not selector.exists():
        shutil.copyfile(template_selector, selector)
    _atomic_selector(selector, SPECIALIST_ID, deployment, REQUIRED_HEADS)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "registered_ready_for_managed_rl",
        "specialist_id": SPECIALIST_ID,
        "bootstrap_ready": str(ready_path),
        "bootstrap_ready_sha256": _sha256(ready_path),
        "bootstrap_checkpoint": str(source_checkpoint_path),
        "bootstrap_checkpoint_sha256": source_checkpoint_digest,
        "router_v6_migration_receipt": str(
            args.router_v6_receipt.expanduser().resolve()
        ),
        "router_v6_migration_receipt_sha256": _sha256(
            args.router_v6_receipt.expanduser().resolve()
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "runtime_registry": str(output_registry),
        "runtime_registry_sha256": _sha256(output_registry),
        "active_gate": str(active_gate_path),
        "active_gate_sha256": _sha256(active_gate_path),
        "active_gate_id": active_gate["active_gate_id"],
        "runtime_asset_sha256": runtime_assets,
        "selector_env": str(selector),
        "selector_env_sha256": _sha256(selector),
        "managed_service": args.managed_service,
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": DECISION_FUSION_V3_SCHEMA,
        "learned_head_count": 19,
        "learned_route_count": 19,
        "matchup_adapter_format": router_v6_migration["adapter_format"],
        "matchup_adapter_slot_capacity": router_v6_migration["slot_capacity"],
        "matchup_adapter_slot_registry_digest": router_v6_migration[
            "slot_registry_digest"
        ],
        "effective_route_reliabilities": route_reliabilities,
        "training_authority": True,
        "selector_authority": True,
        "activation_boundary": "post_alakazam_completion_and_exact_marnie_h10_bootstrap",
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def check(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.receipt.expanduser().resolve()
    receipt = _read(receipt_path)
    registry_path = Path(str(receipt.get("runtime_registry") or "")).resolve()
    gate_path = Path(str(receipt.get("active_gate") or "")).resolve()
    selector_path = Path(str(receipt.get("selector_env") or "")).resolve()
    checkpoint_path = Path(str(receipt.get("checkpoint") or "")).resolve()
    deployment = args.deployment_root.expanduser().resolve()
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "registered_ready_for_managed_rl"
        or receipt.get("specialist_id") != SPECIALIST_ID
        or receipt.get("managed_service") != args.managed_service
        or receipt.get("active_gate_id") != MARNIE_GATE_ID
        or receipt.get("runtime_asset_sha256")
        != _validate_runtime_assets(deployment)
        or _sha256(gate_path) != receipt.get("active_gate_sha256")
        or receipt.get("matchup_adapter_format") != V6_ADAPTER_CHECKPOINT_FORMAT
        or int(receipt.get("matchup_adapter_slot_capacity") or 0) != SLOT_CAPACITY
        or _sha256(args.router_v6_receipt.expanduser().resolve())
        != receipt.get("router_v6_migration_receipt_sha256")
        or _sha256(registry_path) != receipt.get("runtime_registry_sha256")
        or not _selector_env_is_authorized(
            registration_receipt=receipt_path,
            selector_path=selector_path,
            expected_digest=str(receipt.get("selector_env_sha256") or ""),
            deployment=deployment,
        )
        or checkpoint.checkpoint_digest(checkpoint_path) != receipt.get("checkpoint_sha256")
    ):
        raise RuntimeError("Marnie managed-RL registration receipt changed")
    route_reliabilities = _validate_checkpoint(
        checkpoint_path, str(receipt["checkpoint_sha256"])
    )
    _validate_router_v6_checkpoint(
        checkpoint_path,
        str(receipt["checkpoint_sha256"]),
        args.router_v6_registry.expanduser().resolve(),
    )
    if receipt.get("effective_route_reliabilities") != route_reliabilities:
        raise RuntimeError("Marnie route-reliability telemetry changed")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-ready", type=Path, required=True)
    parser.add_argument("--bootstrap-family", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--template-registry", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--curriculum-spec", type=Path, required=True)
    parser.add_argument("--head-role-map", type=Path, required=True)
    parser.add_argument("--curriculum-validation", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--template-selector-env", type=Path, required=True)
    parser.add_argument("--selector-env", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--router-v6-family", type=Path, required=True)
    parser.add_argument("--router-v6-registry", type=Path, required=True)
    parser.add_argument("--router-v6-receipt", type=Path, required=True)
    parser.add_argument(
        "--managed-service",
        default="pokebot-final-format-marnie-r104-h10-rl.service",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check(args) if args.check else register(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
