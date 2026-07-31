#!/usr/bin/env python3
"""Build a protected deck-agnostic core from all frozen specialist teachers.

The selected teacher checkpoints contribute equally to a neutral, parameter-space
student initialization. Specialist matchup adapters are never averaged: a
fresh canonical zero-output bank replaces them before any supervised update.
The balanced expert corpus then trains every shared policy/value/auxiliary
head in immutable one-epoch steps with episode-disjoint validation and early
stopping.

This script only builds and freezes the candidate core. A separate handoff
coordinator must run the established gameplay regression gates against every
teacher before the core can initialize another specialist.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes, checkpoint, device as device_mod  # noqa: E402
from poke_bot.model import (  # noqa: E402
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
from poke_bot.dormant_adapter_compat import (  # noqa: E402
    validate_zero_dormant_checkpoint,
)
from poke_bot.matchup_adapters import ZERO_DORMANT_CHECKPOINT_SCHEMA  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.expert_feature_stream import (  # noqa: E402
    EpisodeGroupedFeatureManifest,
)
from poke_bot.pure_rl.model_registry import (  # noqa: E402
    freeze_model,
    verify_frozen_model,
)
from poke_bot.strategic_schedule import (  # noqa: E402
    EXPANDED_HEAD_IDS,
    expanded_head_epoch_plan,
)
from poke_bot.train import (  # noqa: E402
    belief_card_vocab_from_state,
    device_temporal_greedy_policy_targets,
    load_model_from_checkpoint,
    supervised_rehearsal_step,
    temporal_batches_for_game_ids,
)
from scripts.run_passed_alakazam_core_distillation import (  # noqa: E402
    TARGETS,
    _architecture,
    _fresh_dormant_bank_state,
    _recover_or_train_epoch,
    _validate_balanced_manifest,
    atomic_json,
)
from scripts.run_starmie_expert_bootstrap import (  # noqa: E402
    DEFAULT_RL_PROTOCOL,
    _manifest_expanded_targets,
    _specialist_hot_start_from_core,
    decision_fusion_handoff_contract,
    load_expanded_head_contract,
    validate_expanded_epoch_checkpoint,
)


DEFAULT_FAMILY = (
    "deck_agnostic_core_distilled_from_hops_trevenant_and_starmie_v2"
)
INITIALIZATION_SCHEMA = "poke_bot.multi_teacher_core_initialization/v1"
STATE_SCHEMA = "poke_bot.multi_teacher_core_refresh_state/v1"
READY_SCHEMA = "poke_bot.multi_teacher_core_ready/v1"
FUSION_TEACHER_INITIALIZATION_SCHEMA = (
    "poke_bot.decision_fusion_teacher_initialization/v1"
)
TEACHER_BEHAVIOR_TARGET_SCHEMA = "poke_bot.teacher_behavior_targets/v1"
FROZEN_V5_DERIVATIVE_SCHEMA = (
    "poke_bot.frozen_specialist_roster_v5_derivative/v1"
)
_ADDITIVE_HEAD_PREFIXES = (
    "action_q_head.",
    "action_type_head.",
    "action_target_head.",
    "action_resource_head.",
    "action_utility_head.",
    "tactical_outcome_head.",
    "opponent_response_head.",
    "resource_forecast_head.",
    "game_phase_head.",
    "outcome_distribution_head.",
    "remaining_turns_head.",
    "setup_board_outcome_head.",
    "combo_state_head.",
    "decision_fusion.",
)


def _teacher_identity(family: Path) -> dict[str, Any]:
    frozen = verify_frozen_model(family.expanduser().resolve())
    model_path = Path(str(frozen["model_path"])).resolve()
    _architecture(model_path)
    payload = checkpoint.load_checkpoint(model_path, map_location="cpu")
    archetype_id = str(payload.get("archetype_id") or "")
    if not archetype_id or archetype_id == "unknown":
        raise RuntimeError(
            f"frozen teacher lacks a specialist archetype: {model_path}"
        )
    return {
        "family": str(frozen["family"]),
        "family_path": str(model_path.parent),
        "checkpoint": str(model_path),
        "checkpoint_digest": str(frozen["checkpoint_digest"]),
        "archetype_id": archetype_id,
        "manifest": str(model_path.parent / "manifest.json"),
    }


def _teacher_behavior_inference_identity(
    teacher: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a checksum-bound, current-loader-compatible teacher artifact.

    Some immutable passing checkpoints retain a historical matchup-bank tensor
    layout.  They remain the source teacher identity, but the frozen registry
    also carries a separately checksummed V5 inference derivative whose shared
    tensors are byte-identical and whose only purpose is loading/serving that
    historical specialist under the current runtime.  Behavior distillation
    bypasses matchup adapters, so this derivative preserves the exact policy
    path being queried here.

    Never guess or silently migrate.  A derivative is usable only when its
    immutable manifest binds the exact source path, source checksum,
    specialist identity, derived path, and derived checksum.  Otherwise the
    original passing checkpoint remains the inference artifact and the normal
    strict checkpoint loader decides whether it is compatible.
    """

    source = Path(str(teacher["checkpoint"])).expanduser().resolve()
    source_digest = str(teacher["checkpoint_digest"])
    specialist_id = str(teacher["archetype_id"])
    if (
        not source.is_file()
        or checkpoint.checkpoint_digest(source) != source_digest
    ):
        raise RuntimeError("frozen teacher source identity changed")

    derivative_family = source.parent.with_name(
        source.parent.name + "-roster18-v5"
    )
    derivative_manifest = derivative_family / "manifest.json"
    if not derivative_manifest.is_file():
        return {
            "source_checkpoint": str(source),
            "source_checkpoint_digest": source_digest,
            "inference_checkpoint": str(source),
            "inference_checkpoint_digest": source_digest,
            "derivative_manifest": None,
            "derivative_manifest_digest": None,
            "derivative_schema": None,
        }

    manifest = json.loads(derivative_manifest.read_text(encoding="utf-8"))
    derived = Path(
        str(manifest.get("derived_checkpoint") or "")
    ).expanduser().resolve()
    derived_digest = str(manifest.get("derived_checkpoint_digest") or "")
    manifest_source = Path(
        str(manifest.get("source_passing_checkpoint") or "")
    ).expanduser().resolve()
    if (
        manifest.get("schema") != FROZEN_V5_DERIVATIVE_SCHEMA
        or str(manifest.get("specialist_id") or "") != specialist_id
        or manifest_source != source
        or manifest.get("source_passing_checkpoint_digest") != source_digest
        or manifest.get("retained_rows_byte_identical") is not True
        or manifest.get("inference_only") is not True
        or not derived.is_file()
        or derived.parent != derivative_family
        or not derived_digest.startswith("sha256:")
        or checkpoint.checkpoint_digest(derived) != derived_digest
    ):
        raise RuntimeError(
            "frozen teacher inference derivative identity is invalid: "
            f"{derivative_manifest}"
        )
    return {
        "source_checkpoint": str(source),
        "source_checkpoint_digest": source_digest,
        "inference_checkpoint": str(derived),
        "inference_checkpoint_digest": derived_digest,
        "derivative_manifest": str(derivative_manifest),
        "derivative_manifest_digest": (
            "sha256:"
            + hashlib.sha256(derivative_manifest.read_bytes()).hexdigest()
        ),
        "derivative_schema": FROZEN_V5_DERIVATIVE_SCHEMA,
    }


def _base_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in dict(payload.get("model_state_dict") or {}).items()
        if not name.startswith("matchup_adapter_bank.")
    }


def _aux_archetype_ids(
    payload: dict[str, Any],
    *,
    classes: int,
) -> list[str]:
    """Recover the semantic row order of an immutable checkpoint aux head."""

    if classes <= 0:
        raise RuntimeError("auxiliary head must contain an unknown row")
    extra = dict(payload.get("extra") or {})
    expansion = dict(
        extra.get("specialist_aux_archetype_head_expansion") or {}
    )
    recorded = expansion.get("target_archetype_ids")
    if isinstance(recorded, list) and len(recorded) + 1 == classes:
        ids = [str(value) for value in recorded]
    else:
        compatible_orders = (
            archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS,
            archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS,
            archetypes.LEGACY_AUX_ARCHETYPE_IDS,
            archetypes.archetype_ids(),
        )
        matches = [
            list(order)
            for order in compatible_orders
            if len(order) + 1 == classes
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "checkpoint auxiliary row identity is unavailable or ambiguous: "
                f"classes={classes}"
            )
        ids = matches[0]
    if len(ids) != len(set(ids)):
        raise RuntimeError("checkpoint auxiliary row identity contains duplicates")
    current = set(archetypes.archetype_ids())
    if any(value not in current for value in ids):
        raise RuntimeError("checkpoint auxiliary row identity is not registered")
    return ids


def _align_teacher_state(
    payload: dict[str, Any],
    *,
    target_state: dict[str, Any],
    target_aux_ids: list[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align a frozen legacy auxiliary head to the canonical row registry.

    Shared tensors remain exact frozen-teacher tensors. A legacy auxiliary
    head contributes its named historical rows and unknown row; rows for
    archetypes that did not exist in that checkpoint retain the accepted
    initialization-core values instead of receiving random or fabricated
    teacher values.
    """

    import torch

    source = _base_state(payload)
    missing = sorted(set(target_state) - set(source))
    extra = sorted(set(source) - set(target_state))
    invalid = [
        name
        for name in (*missing, *extra)
        if not name.startswith(_ADDITIVE_HEAD_PREFIXES)
    ]
    if invalid:
        raise RuntimeError(
            "teacher reusable tensor keys differ outside additive heads: "
            + ", ".join(invalid)
        )
    aligned: dict[str, Any] = {}
    adaptations: list[dict[str, Any]] = []
    for name, target in target_state.items():
        if name not in source:
            if not isinstance(target, torch.Tensor):
                raise RuntimeError(f"initialization state is not tensor-valued: {name}")
            aligned[name] = target.detach().cpu().clone()
            adaptations.append(
                {
                    "tensor": name,
                    "behavior": "retain_initialization_for_absent_additive_head",
                }
            )
            continue
        value = source[name]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise RuntimeError(f"teacher state is not tensor-valued: {name}")
        if name in {"aux_head.3.weight", "aux_head.3.bias"}:
            if target_aux_ids is None:
                raise RuntimeError("target auxiliary row identity is unavailable")
            if value.shape[1:] != target.shape[1:]:
                raise RuntimeError(f"teacher tensor contract differs: {name}")
            source_ids = _aux_archetype_ids(
                payload, classes=int(value.shape[0])
            )
            if int(target.shape[0]) != len(target_aux_ids) + 1:
                raise RuntimeError("target auxiliary row identity differs")
            expanded = target.detach().cpu().clone()
            copied = [
                archetype_id
                for archetype_id in target_aux_ids
                if archetype_id in source_ids
            ]
            for archetype_id in copied:
                expanded[target_aux_ids.index(archetype_id)].copy_(
                    value[source_ids.index(archetype_id)]
                )
            expanded[-1].copy_(value[-1])
            aligned[name] = expanded
            if source_ids != target_aux_ids:
                adaptations.append(
                    {
                        "tensor": name,
                        "source_rows": len(source_ids) + 1,
                        "target_rows": len(target_aux_ids) + 1,
                        "source_archetype_ids": source_ids,
                        "target_archetype_ids": target_aux_ids,
                        "mapped_rows": copied + ["unknown"],
                        "retained_initialization_rows": [
                            archetype_id
                            for archetype_id in target_aux_ids
                            if archetype_id not in source_ids
                        ],
                        "omitted_source_rows": [
                            archetype_id
                            for archetype_id in source_ids
                            if archetype_id not in target_aux_ids
                        ],
                    }
                )
            continue
        if value.shape == target.shape:
            aligned[name] = value
            continue
        raise RuntimeError(f"teacher tensor contract differs: {name}")
    for name in extra:
        adaptations.append(
            {
                "tensor": name,
                "behavior": "omit_additive_head_absent_from_initialization_architecture",
            }
        )
    return aligned, adaptations


def _validate_initialization(
    path: Path,
    *,
    initialization_digest: str,
    teachers: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    extra = dict(payload.get("extra") or {})
    record = dict(extra.get("multi_teacher_core_initialization") or {})
    expected_teachers = [
        {
            "family": row["family"],
            "checkpoint": row["checkpoint"],
            "checkpoint_digest": row["checkpoint_digest"],
            "weight": 1.0 / len(teachers),
        }
        for row in teachers
    ]
    if (
        record.get("schema") != INITIALIZATION_SCHEMA
        or record.get("initialization_core_digest") != initialization_digest
        or record.get("teachers") != expected_teachers
        or record.get("averaging") != "equal_weight_parameter_space_mean"
        or str(payload.get("archetype_id") or "") != "unknown"
        or any(
            key in payload
            for key in (
                "optimizer_state_dict",
                "scaler_state_dict",
                "scheduler_state_dict",
                "rng_state",
            )
        )
    ):
        raise RuntimeError("multi-teacher initialization metadata drifted")

    initialization_path = Path(
        str(record.get("initialization_core_checkpoint") or "")
    ).expanduser().resolve()
    if (
        not initialization_path.is_file()
        or checkpoint.checkpoint_digest(initialization_path)
        != initialization_digest
    ):
        raise RuntimeError("initialization core identity changed")
    target_state = _base_state(
        checkpoint.load_checkpoint(initialization_path, map_location="cpu")
    )
    target_aux_weight = target_state.get("aux_head.3.weight")
    target_aux_ids = (
        _aux_archetype_ids(
            checkpoint.load_checkpoint(initialization_path, map_location="cpu"),
            classes=int(target_aux_weight.shape[0]),
        )
        if target_aux_weight is not None
        else None
    )
    teacher_payloads = [
        checkpoint.load_checkpoint(row["checkpoint"], map_location="cpu")
        for row in teachers
    ]
    aligned = [
        _align_teacher_state(
            value,
            target_state=target_state,
            target_aux_ids=target_aux_ids,
        )
        for value in teacher_payloads
    ]
    states = [value[0] for value in aligned]
    expected_adaptations = [
        {
            "checkpoint_digest": teachers[index]["checkpoint_digest"],
            "tensors": adaptations,
        }
        for index, (_state, adaptations) in enumerate(aligned)
        if adaptations
    ]
    if record.get("architecture_alignment") != expected_adaptations:
        raise RuntimeError("multi-teacher architecture alignment metadata drifted")
    actual = _base_state(payload)
    if not states or any(state.keys() != states[0].keys() for state in states):
        raise RuntimeError("teacher reusable tensor keys differ")
    if actual.keys() != states[0].keys():
        raise RuntimeError("student reusable tensor keys differ from teachers")
    for name in actual:
        values = [state[name] for state in states]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError(f"teacher state is not tensor-valued: {name}")
        if any(value.shape != values[0].shape for value in values):
            raise RuntimeError(f"teacher tensor shape differs: {name}")
        if values[0].is_floating_point() or values[0].is_complex():
            accumulator_dtype = (
                torch.complex128 if values[0].is_complex() else torch.float64
            )
            expected = torch.stack(
                [value.to(accumulator_dtype) for value in values]
            ).mean(dim=0).to(values[0].dtype)
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError(
                    f"non-floating teacher state differs and cannot be averaged: {name}"
                )
            expected = values[0]
        if not torch.equal(actual[name], expected):
            raise RuntimeError(f"student is not the exact teacher mean: {name}")
    validate_zero_dormant_checkpoint(path)
    return {
        "path": str(path),
        "digest": checkpoint.checkpoint_digest(path),
        "reusable_tensor_count": len(actual),
        "teacher_count": len(teachers),
    }


def materialize_initialization(
    *,
    initialization_family: Path,
    teacher_families: list[Path],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create or validate the exact neutral equal-teacher initialization."""

    import torch

    initialization = verify_frozen_model(initialization_family.expanduser().resolve())
    initialization_path = Path(str(initialization["model_path"])).resolve()
    initialization_digest = str(initialization["checkpoint_digest"])
    _architecture(initialization_path)
    teachers = [_teacher_identity(path) for path in teacher_families]
    if len(teachers) < 2:
        raise ValueError("multi-teacher refresh requires at least two teachers")
    if len({row["checkpoint_digest"] for row in teachers}) != len(teachers):
        raise ValueError("multi-teacher refresh requires distinct teacher checkpoints")
    output = output.expanduser().resolve()
    if output.is_file():
        return (
            _validate_initialization(
                output,
                initialization_digest=initialization_digest,
                teachers=teachers,
            ),
            teachers,
        )

    template = checkpoint.load_checkpoint(initialization_path, map_location="cpu")
    teacher_payloads = [
        checkpoint.load_checkpoint(row["checkpoint"], map_location="cpu")
        for row in teachers
    ]
    target_state = _base_state(template)
    target_aux_weight = target_state.get("aux_head.3.weight")
    target_aux_ids = (
        _aux_archetype_ids(
            template, classes=int(target_aux_weight.shape[0])
        )
        if target_aux_weight is not None
        else None
    )
    aligned = [
        _align_teacher_state(
            value,
            target_state=target_state,
            target_aux_ids=target_aux_ids,
        )
        for value in teacher_payloads
    ]
    states = [value[0] for value in aligned]
    averaged: dict[str, Any] = {}
    for name in states[0]:
        values = [state[name] for state in states]
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != values[0].shape
            for value in values
        ):
            raise RuntimeError(f"teacher tensor contract differs: {name}")
        if values[0].is_floating_point() or values[0].is_complex():
            accumulator_dtype = (
                torch.complex128 if values[0].is_complex() else torch.float64
            )
            averaged[name] = (
                torch.stack([value.to(accumulator_dtype) for value in values])
                .mean(dim=0)
                .to(values[0].dtype)
            )
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError(
                    f"non-floating teacher state differs and cannot be averaged: {name}"
                )
            averaged[name] = values[0].detach().cpu().clone()

    bank, adapter_state = _fresh_dormant_bank_state()
    averaged.update(adapter_state)
    teacher_rows = [
        {
            "family": row["family"],
            "checkpoint": row["checkpoint"],
            "checkpoint_digest": row["checkpoint_digest"],
            "weight": 1.0 / len(teachers),
        }
        for row in teachers
    ]
    record = {
        "schema": INITIALIZATION_SCHEMA,
        "initialization_core_family": str(initialization["family"]),
        "initialization_core_checkpoint": str(initialization_path),
        "initialization_core_digest": initialization_digest,
        "teachers": teacher_rows,
        "averaging": "equal_weight_parameter_space_mean",
        "architecture_alignment": [
            {
                "checkpoint_digest": teachers[index]["checkpoint_digest"],
                "tensors": adaptations,
            }
            for index, (_state, adaptations) in enumerate(aligned)
            if adaptations
        ],
        "teacher_gradients_allowed": False,
        "teacher_optimizer_steps_allowed": False,
        "specialist_matchup_adapters_reset": True,
        "runtime_enabled": False,
    }
    payload = copy.deepcopy(template)
    payload["model_state_dict"] = averaged
    payload["archetype_id"] = "unknown"
    payload["model_id"] = "deck_agnostic_core.multi_teacher_initialization"
    payload["step"] = 0
    payload["epoch"] = 0
    payload["rl_iteration"] = 0
    payload["best_metric"] = None
    payload["early_stop_state"] = None
    model_config = dict(payload.get("model_config") or {})
    model_config["matchup_adapters_enabled"] = False
    payload["model_config"] = model_config
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
    ):
        payload.pop(key, None)
    payload["provenance"] = {
        "multi_teacher_core_initialization": copy.deepcopy(record)
    }
    payload["extra"] = {
        "pure_rl": True,
        "multi_teacher_core_initialization": record,
        "matchup_adapter_config": bank.config_dict(),
        "matchup_adapters_runtime_enabled": False,
        "matchup_adapter_training_enabled": False,
        "matchup_adapter_optimizer_included": False,
        "dormant_matchup_adapter_bank": {
            "schema": ZERO_DORMANT_CHECKPOINT_SCHEMA,
            "materialization": "multi_teacher_core_refresh_reset",
            "runtime_enabled": False,
            "training_enabled": False,
            "optimizer_imported": False,
            "optimizer_included": False,
            "frozen": True,
            "zero_output": True,
            "parameter_count": sum(value.numel() for value in adapter_state.values()),
            "adapter_config": bank.config_dict(),
            "teacher_checkpoint_digests": [
                row["checkpoint_digest"] for row in teachers
            ],
        },
    }
    checkpoint.immutable_torch_save(payload, output)
    return (
        _validate_initialization(
            output,
            initialization_digest=initialization_digest,
            teachers=teachers,
        ),
        teachers,
    )


def materialize_fusion_teacher_initialization(
    *,
    source_parent: Path,
    teachers: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    """Seed a new fused core from every architecture-compatible fused teacher.

    Legacy cumulative cores predate decision fusion, so their neutral
    parameter mean cannot contain fusion tensors.  Once frozen specialists
    have learned the canonical fused path, discarding those tensors and
    starting every cumulative core from zero-safe random initialization causes
    localized teacher forgetting.  Average only the exact compatible fusion
    tensors here; the ordinary seven-teacher mean remains authoritative for
    every pre-existing shared tensor.
    """

    source_parent = source_parent.expanduser().resolve()
    output = output.expanduser().resolve()
    source_digest = checkpoint.checkpoint_digest(source_parent)
    source_payload = checkpoint.load_checkpoint(source_parent, map_location="cpu")
    source_model = load_model_from_checkpoint(source_parent, device=torch.device("cpu"))
    target_state = {
        name: value.detach().cpu().clone()
        for name, value in source_model.state_dict().items()
    }
    fusion_keys = sorted(
        name for name in target_state if name.startswith("decision_fusion.")
    )
    if not fusion_keys:
        raise RuntimeError("fused core parent did not materialize fusion tensors")

    contributors: list[dict[str, Any]] = []
    contributor_states: list[dict[str, torch.Tensor]] = []
    excluded: list[dict[str, Any]] = []
    for teacher in teachers:
        payload = checkpoint.load_checkpoint(
            Path(str(teacher["checkpoint"])), map_location="cpu"
        )
        config = dict(payload.get("model_config") or {})
        state = dict(payload.get("model_state_dict") or {})
        reason: str | None = None
        if (
            config.get("decision_fusion_enabled") is not True
            or config.get("decision_fusion_runtime_enabled") is not True
        ):
            reason = "decision_fusion_not_enabled"
        elif any(name not in state for name in fusion_keys):
            reason = "incomplete_decision_fusion_state"
        elif any(
            not isinstance(state[name], torch.Tensor)
            or state[name].shape != target_state[name].shape
            or state[name].dtype != target_state[name].dtype
            for name in fusion_keys
        ):
            reason = "incompatible_decision_fusion_state"
        if reason is not None:
            excluded.append(
                {
                    "family": teacher["family"],
                    "checkpoint_digest": teacher["checkpoint_digest"],
                    "reason": reason,
                }
            )
            continue
        contributors.append(
            {
                "family": teacher["family"],
                "checkpoint": teacher["checkpoint"],
                "checkpoint_digest": teacher["checkpoint_digest"],
            }
        )
        contributor_states.append(
            {
                name: state[name].detach().cpu().clone()
                for name in fusion_keys
            }
        )
    if not contributors:
        raise RuntimeError(
            "decision-fusion core requires at least one compatible fused teacher"
        )

    averaged: dict[str, torch.Tensor] = {}
    for name in fusion_keys:
        values = [state[name] for state in contributor_states]
        if values[0].is_floating_point() or values[0].is_complex():
            accumulator_dtype = (
                torch.complex128 if values[0].is_complex() else torch.float64
            )
            averaged[name] = (
                torch.stack([value.to(accumulator_dtype) for value in values])
                .mean(dim=0)
                .to(values[0].dtype)
            )
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError(
                    "non-floating decision-fusion teacher state differs: "
                    f"{name}"
                )
            averaged[name] = values[0].clone()

    record = {
        "schema": FUSION_TEACHER_INITIALIZATION_SCHEMA,
        "source_parent": str(source_parent),
        "source_parent_digest": source_digest,
        "averaging": "equal_weight_compatible_fused_teacher_mean",
        "contributors": [
            {**row, "weight": 1.0 / len(contributors)}
            for row in contributors
        ],
        "excluded_teachers": excluded,
        "tensor_count": len(fusion_keys),
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
        "legacy_shared_tensor_initialization_unchanged": True,
    }
    if output.is_file():
        existing = checkpoint.load_checkpoint(output, map_location="cpu")
        existing_record = dict(
            (existing.get("extra") or {}).get(
                "decision_fusion_teacher_initialization"
            )
            or {}
        )
        existing_state = dict(existing.get("model_state_dict") or {})
        if (
            existing_record != record
            or any(
                name not in existing_state
                or not torch.equal(existing_state[name], averaged[name])
                for name in fusion_keys
            )
        ):
            raise RuntimeError(
                "existing decision-fusion teacher initialization changed"
            )
    else:
        payload = copy.deepcopy(source_payload)
        payload["model_state_dict"] = target_state
        payload["model_state_dict"].update(averaged)
        payload["model_id"] = (
            "deck_agnostic_core.compatible_fused_teacher_initialization"
        )
        extra = dict(payload.get("extra") or {})
        extra["decision_fusion_teacher_initialization"] = copy.deepcopy(record)
        payload["extra"] = extra
        provenance = dict(payload.get("provenance") or {})
        provenance["decision_fusion_teacher_initialization"] = copy.deepcopy(
            record
        )
        payload["provenance"] = provenance
        checkpoint.immutable_torch_save(payload, output)

    load_model_from_checkpoint(output, device=torch.device("cpu"))
    actual = checkpoint.load_checkpoint(output, map_location="cpu")
    actual_state = dict(actual.get("model_state_dict") or {})
    if any(
        name not in actual_state
        or not torch.equal(actual_state[name], averaged[name])
        for name in fusion_keys
    ):
        raise RuntimeError("materialized decision-fusion teacher mean changed")
    return {
        **record,
        "path": str(output),
        "digest": checkpoint.checkpoint_digest(output),
    }


def materialize_teacher_behavior_targets(
    *,
    corpus: Any,
    manifest_path: Path,
    manifest_digest: str,
    split_seed: int,
    max_context: int,
    teachers: list[dict[str, Any]],
    requested_batch_size: int,
    output: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bind each matching replay game to its frozen teacher's greedy policy.

    Neural parameter averaging is not behavior preserving when independently
    trained hidden units have different permutations.  These targets instead
    evaluate the exact frozen teacher on the exact causal board/action history
    and legal-option row from its own archetype.  Non-teacher archetypes retain
    their ordinary expert action targets and are marked ``-1`` here.
    """

    output = output.expanduser().resolve()
    inference_identities = [
        _teacher_behavior_inference_identity(row) for row in teachers
    ]
    teacher_digests = [str(row["checkpoint_digest"]) for row in teachers]
    teacher_archetypes = [str(row["archetype_id"]) for row in teachers]
    if len(teacher_archetypes) != len(set(teacher_archetypes)):
        raise RuntimeError("teacher behavior archetypes are not unique")
    expected_identity = {
        "schema": TEACHER_BEHAVIOR_TARGET_SCHEMA,
        "manifest_digest": str(manifest_digest),
        "split_seed": int(split_seed),
        "max_context": int(max_context),
        "teacher_checkpoint_digests": teacher_digests,
        "teacher_inference_identities": inference_identities,
        "teacher_archetypes": teacher_archetypes,
        "target_semantics": "matching_archetype_frozen_teacher_greedy_action",
        "causal_inputs_only": True,
        "matchup_adapters_enabled": False,
    }

    def validate_payload(payload: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        record = dict(payload.get("record") or {})
        targets = payload.get("targets")
        if not isinstance(targets, torch.Tensor):
            raise RuntimeError("teacher behavior artifact has no target tensor")
        targets = targets.detach().cpu().to(dtype=torch.int32).contiguous()
        digest = "sha256:" + hashlib.sha256(
            targets.numpy().tobytes()
        ).hexdigest()
        if (
            any(record.get(key) != value for key, value in expected_identity.items())
            or int(targets.numel()) != int(corpus.total_samples)
            or record.get("target_digest") != digest
            or int(record.get("target_rows", 0))
            != int((targets >= 0).sum().item())
        ):
            raise RuntimeError("teacher behavior target identity changed")
        return targets.to(device=device, dtype=torch.long), record

    if output.is_file():
        loaded = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise RuntimeError("teacher behavior artifact is not a mapping")
        return validate_payload(loaded)

    plan = EpisodeGroupedFeatureManifest.open(
        manifest_path,
        expected_manifest_digest=str(manifest_digest),
        val_frac=0.10,
        seed=int(split_seed),
        max_context=int(max_context),
        expected_compact_mode="temporal-expert-v1",
        workers=max(1, min(8, os.cpu_count() or 1)),
    )
    train_archetypes, validation_archetypes = plan.partition_archetypes()
    game_archetypes = train_archetypes + validation_archetypes
    game_count = int(corpus.train_games + corpus.val_games)
    if len(game_archetypes) != game_count:
        raise RuntimeError(
            "teacher behavior game metadata does not match resident corpus: "
            f"metadata={len(game_archetypes)} corpus={game_count}"
        )
    teacher_by_archetype = {
        str(row["archetype_id"]): index
        for index, row in enumerate(teachers)
    }
    game_routes = torch.tensor(
        [teacher_by_archetype.get(value, -1) for value in game_archetypes],
        device=device,
        dtype=torch.long,
    )
    targets = torch.full(
        (int(corpus.total_samples),),
        -1,
        device=device,
        dtype=torch.long,
    )
    route_records: list[dict[str, Any]] = []
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    for route, teacher in enumerate(teachers):
        inference_identity = inference_identities[route]
        route_games = torch.nonzero(
            game_routes == route, as_tuple=False
        ).flatten()
        if int(route_games.numel()) <= 0:
            raise RuntimeError(
                "balanced core corpus has no games for frozen teacher "
                f"{teacher['archetype_id']}"
            )
        model = load_model_from_checkpoint(
            Path(str(inference_identity["inference_checkpoint"])),
            device=device,
        )
        model.eval()
        model.requires_grad_(False)
        target_rows = 0
        for batch in temporal_batches_for_game_ids(
            corpus,
            route_games,
            batch_size=int(requested_batch_size),
        ):
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                sample_ids, greedy = device_temporal_greedy_policy_targets(
                    model, corpus, batch
                )
            if bool((targets.index_select(0, sample_ids) >= 0).any()):
                raise RuntimeError("teacher behavior target rows overlap")
            targets.index_copy_(0, sample_ids, greedy)
            target_rows += int(sample_ids.numel())
        route_records.append(
            {
                "archetype_id": str(teacher["archetype_id"]),
                "checkpoint": str(teacher["checkpoint"]),
                "checkpoint_digest": str(teacher["checkpoint_digest"]),
                "inference_checkpoint": str(
                    inference_identity["inference_checkpoint"]
                ),
                "inference_checkpoint_digest": str(
                    inference_identity["inference_checkpoint_digest"]
                ),
                "derivative_manifest": inference_identity[
                    "derivative_manifest"
                ],
                "derivative_manifest_digest": inference_identity[
                    "derivative_manifest_digest"
                ],
                "games": int(route_games.numel()),
                "target_rows": target_rows,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cpu_targets = targets.detach().cpu().to(dtype=torch.int32).contiguous()
    target_digest = "sha256:" + hashlib.sha256(
        cpu_targets.numpy().tobytes()
    ).hexdigest()
    record = {
        **expected_identity,
        "target_digest": target_digest,
        "target_rows": int((cpu_targets >= 0).sum().item()),
        "masked_non_teacher_rows": int((cpu_targets < 0).sum().item()),
        "total_samples": int(cpu_targets.numel()),
        "routes": route_records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint.immutable_torch_save(
        {"record": record, "targets": cpu_targets}, output
    )
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    return validate_payload(loaded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-corpus", type=Path, required=True)
    parser.add_argument("--initialization-family", type=Path, required=True)
    parser.add_argument(
        "--teacher-family", type=Path, action="append", required=True
    )
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output-family", default=DEFAULT_FAMILY)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-decisions", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=12288)
    parser.add_argument("--split-seed", type=int, default=20260723)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--expanded-heads", action="store_true")
    parser.add_argument("--decision-fusion", action="store_true")
    parser.add_argument(
        "--teacher-behavior-distillation", action="store_true"
    )
    parser.add_argument("--teacher-policy-weight", type=float, default=0.5)
    parser.add_argument("--rl-protocol", type=Path, default=DEFAULT_RL_PROTOCOL)
    parser.add_argument("--expected-expanded-schedule-digest", default="")
    parser.add_argument("--expected-expanded-target-digest", default="")
    args = parser.parse_args()
    if int(args.epochs) <= 0 or int(args.patience) <= 0:
        raise ValueError("epochs/patience must be positive")
    expanded_raw: dict[str, Any] | None = None
    expanded_identity: dict[str, Any] | None = None
    if args.expanded_heads:
        expanded_raw, expanded_identity = load_expanded_head_contract(
            args.rl_protocol
        )
        if int(args.epochs) != int(
            expanded_identity["schedule"]["total_epochs"]
        ):
            raise ValueError(
                "expanded cumulative core is locked to exactly 25 epochs"
            )
        if (
            str(args.expected_expanded_schedule_digest or "")
            != expanded_identity["schedule_digest"]
            or str(args.expected_expanded_target_digest or "")
            != expanded_identity["target_schema_digest"]
        ):
            raise ValueError(
                "generated handoff expanded-head digests do not match canon"
            )
    elif (
        args.expected_expanded_schedule_digest
        or args.expected_expanded_target_digest
    ):
        raise ValueError("expanded-head digests require --expanded-heads")
    if args.decision_fusion and not args.expanded_heads:
        raise ValueError("--decision-fusion requires --expanded-heads")
    if args.teacher_behavior_distillation and not args.decision_fusion:
        raise ValueError(
            "teacher behavior distillation requires decision fusion"
        )
    if not math.isfinite(float(args.teacher_policy_weight)) or float(
        args.teacher_policy_weight
    ) <= 0.0:
        raise ValueError("teacher policy weight must be finite and positive")
    fusion_identity = (
        decision_fusion_handoff_contract(args.rl_protocol)
        if args.decision_fusion
        else None
    )
    family_name = str(args.output_family).strip()
    if not family_name or "/" in family_name or family_name in {".", ".."}:
        raise ValueError("output family must be one safe registry basename")

    core_pointer = args.core_corpus.expanduser().resolve()
    core_payload = _validate_balanced_manifest(
        core_pointer, min_decisions=int(args.min_decisions)
    )
    core_pointer_payload = json.loads(core_pointer.read_text(encoding="utf-8"))
    core_manifest_digest = str(
        core_pointer_payload.get("manifest_sha256") or ""
    )
    core_manifest_raw = Path(
        str(core_pointer_payload.get("manifest") or "")
    )
    core_manifest_path = (
        core_manifest_raw.resolve()
        if core_manifest_raw.is_absolute()
        else (core_pointer.parent / core_manifest_raw).resolve()
    )
    expanded_targets = (
        _manifest_expanded_targets(
            core_manifest_path,
            decisions=int(
                (core_pointer_payload.get("totals") or {}).get(
                    "decisions_kept", 0
                )
            ),
        )
        if expanded_identity is not None
        else None
    )
    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    family_dir = args.registry_root.expanduser().resolve() / family_name
    if args.ready.is_file() and family_dir.is_dir():
        # A completed refresh is an immutable, versioned artifact.  Resume it
        # from its checksums before consulting the current archetype registry:
        # later staged registry additions must not reinterpret or rebuild an
        # already-frozen core under a different auxiliary-head shape.
        ready = json.loads(args.ready.read_text(encoding="utf-8"))
        frozen = verify_frozen_model(family_dir)
        teachers = [_teacher_identity(path) for path in args.teacher_family]
        initialization_path = Path(
            str(ready.get("initialization") or "")
        ).expanduser().resolve()
        training_parent_path = Path(
            str(ready.get("training_parent") or "")
        ).expanduser().resolve()
        status = str(ready.get("status") or "")
        resumable_status = (
            status == "candidate_ready_for_gameplay_regression"
            or (
                status == "ready"
                and ready.get("gameplay_regression_passed") is True
            )
        )
        if (
            ready.get("schema") == READY_SCHEMA
            and resumable_status
            and ready.get("family") == family_name
            and ready.get("run_name") == args.run_name
            and ready.get("checkpoint_digest") == frozen.get("checkpoint_digest")
            and ready.get("checkpoint") == frozen.get("model_path")
            and initialization_path.is_file()
            and ready.get("initialization_digest")
            == checkpoint.checkpoint_digest(initialization_path)
            and ready.get("teacher_checkpoint_digests")
            == [row["checkpoint_digest"] for row in teachers]
            and ready.get("core_manifest_sha256") == core_manifest_digest
            and int(ready.get("split_seed") or 0) == int(args.split_seed)
            and (
                fusion_identity is None
                or (
                    ready.get("decision_fusion") == fusion_identity
                    and ready.get("flat_policy_remains_authoritative")
                    is False
                )
            )
            and int(ready.get("epochs_max") or 0) == int(args.epochs)
            and ready.get("teacher_behavior_distillation_enabled")
            is bool(args.teacher_behavior_distillation)
            and (
                not args.teacher_behavior_distillation
                or float(ready.get("teacher_policy_weight") or 0.0)
                == float(args.teacher_policy_weight)
            )
            and int(ready.get("early_stop_patience") or 0)
            == int(args.patience)
            and int(ready.get("decisions") or 0) >= int(args.min_decisions)
            and ready.get("specialist_matchup_adapters_reset") is True
            and (
                expanded_identity is None
                or (
                    training_parent_path.is_file()
                    and ready.get("training_parent_digest")
                    == checkpoint.checkpoint_digest(
                        training_parent_path
                    )
                    and ready.get("expanded_target_schema_digest")
                    == expanded_identity["target_schema_digest"]
                    and ready.get("expanded_schedule_digest")
                    == expanded_identity["schedule_digest"]
                    and set(ready.get("expanded_heads_trained") or ())
                    == set(EXPANDED_HEAD_IDS)
                    and ready.get("runtime_enabled_heads") == []
                    and int(ready.get("epochs_completed") or 0)
                    == int(
                        expanded_identity["schedule"]["total_epochs"]
                    )
                )
            )
        ):
            print(json.dumps(ready, indent=2), flush=True)
            return 0
        raise RuntimeError("existing refreshed-core readiness identity changed")

    initialization, teachers = materialize_initialization(
        initialization_family=args.initialization_family,
        teacher_families=list(args.teacher_family),
        output=run_dir / "multi_teacher_initialization.pt",
    )
    initialization_path = Path(initialization["path"]).resolve()
    initialization_digest = str(initialization["digest"])
    expanded_hot_start: dict[str, Any] | None = None
    fusion_teacher_initialization: dict[str, Any] | None = None
    if expanded_identity is not None:
        (
            source_parent,
            source_digest,
            expanded_hot_start,
        ) = _specialist_hot_start_from_core(
            initialization_path,
            run_dir=run_dir,
            archetype="deck-agnostic-core",
            enable_expanded_heads=True,
            expanded_identity=expanded_identity,
            enable_decision_fusion=bool(args.decision_fusion),
        )
        if args.decision_fusion:
            fusion_teacher_initialization = (
                materialize_fusion_teacher_initialization(
                    source_parent=source_parent,
                    teachers=teachers,
                    output=(
                        run_dir
                        / "shared_core_hot_start."
                        "expanded-v6-fused-teacher-mean-v1.pt"
                    ),
                )
            )
            source_parent = Path(
                str(fusion_teacher_initialization["path"])
            ).resolve()
            source_digest = str(fusion_teacher_initialization["digest"])
            if expanded_hot_start is not None:
                expanded_hot_start = {
                    **expanded_hot_start,
                    "decision_fusion_teacher_initialization": (
                        fusion_teacher_initialization
                    ),
                }
    else:
        source_parent = initialization_path
        source_digest = initialization_digest
    model_config = _architecture(source_parent)
    manifest_requirements: dict[str, Any] = {
        "min_decisions": int(args.min_decisions),
        "require_protected": True,
        "required_compact_mode": "temporal-expert-v1",
        "required_max_context": int(model_config["max_context"]),
        "required_target_coverage": TARGETS,
    }
    if expanded_identity is not None:
        manifest_requirements.update(
            {
                "required_expanded_target_schema": expanded_identity[
                    "target_schema"
                ],
                "required_expanded_target_digest": expanded_identity[
                    "target_schema_digest"
                ],
                "required_expanded_heads": tuple(EXPANDED_HEAD_IDS),
            }
        )
    manifest_identity = resolve_expert_manifest(
        core_pointer, **manifest_requirements
    )
    transfer_payload = checkpoint.load_checkpoint(source_parent, map_location="cpu")
    belief_card_vocab = belief_card_vocab_from_state(
        dict(transfer_payload.get("model_state_dict") or {})
    )

    state_path = run_dir / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    teacher_digests = [row["checkpoint_digest"] for row in teachers]
    if state and (
        state.get("initialization_digest") != initialization_digest
        or state.get(
            "training_parent_digest", state.get("initialization_digest")
        )
        != source_digest
        or state.get("teacher_checkpoint_digests") != teacher_digests
        or state.get("manifest_digest") != manifest_identity.digest
        or int(state.get("split_seed") or 0) != int(args.split_seed)
        or state.get("decision_fusion") != fusion_identity
        or state.get("teacher_behavior_distillation_enabled")
        is not bool(args.teacher_behavior_distillation)
        or (
            args.teacher_behavior_distillation
            and float(state.get("teacher_policy_weight") or 0.0)
            != float(args.teacher_policy_weight)
        )
        or (
            expanded_identity is not None
            and (
                state.get("expanded_target_schema_digest")
                != expanded_identity["target_schema_digest"]
                or state.get("expanded_schedule_digest")
                != expanded_identity["schedule_digest"]
            )
        )
    ):
        raise RuntimeError("multi-teacher core refresh state identity changed")

    device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
    cache = ResidentExpertCorpusCache(cpu_pack_root=args.cpu_pack_root)
    print(
        "[core-refresh] loading protected balanced corpus "
        f"records={manifest_identity.records} decisions={manifest_identity.decisions} "
        f"archetypes={len(core_payload['totals']['records_per_archetype'])} "
        f"teachers={len(teachers)} device={device}",
        flush=True,
    )
    corpus = cache.prepare(
        manifest_identity,
        device=device,
        seed=int(args.split_seed),
        max_context=int(model_config["max_context"]),
        belief_card_vocab=belief_card_vocab,
    )
    if not corpus.has_exact_targets:
        raise RuntimeError("core refresh corpus lost required all-head targets")
    teacher_policy_targets: torch.Tensor | None = None
    teacher_behavior_record: dict[str, Any] | None = None
    if args.teacher_behavior_distillation:
        teacher_policy_targets, teacher_behavior_record = (
            materialize_teacher_behavior_targets(
                corpus=corpus,
                manifest_path=core_manifest_path,
                manifest_digest=manifest_identity.digest,
                split_seed=int(args.split_seed),
                max_context=int(model_config["max_context"]),
                teachers=teachers,
                requested_batch_size=int(args.batch_size),
                output=run_dir / "teacher_behavior_targets.pt",
                device=device,
            )
        )
        print(
            "[core-refresh] teacher behavior targets "
            f"rows={teacher_behavior_record['target_rows']} "
            f"digest={teacher_behavior_record['target_digest']}",
            flush=True,
        )
    try:
        history = list(state.get("history") or [])
        best_metric = float(state.get("best_metric", math.inf))
        best_path = str(state.get("best_path") or "")
        best_digest = str(state.get("best_digest") or "")
        bad_epochs = int(state.get("bad_epochs", 0))
        parent = source_parent
        parent_digest = source_digest
        start_epoch = 1
        if history:
            last = history[-1]
            parent = Path(str(last["checkpoint"])).resolve()
            parent_digest = str(last["checkpoint_digest"])
            if checkpoint.checkpoint_digest(parent) != parent_digest:
                raise RuntimeError("core refresh resume parent digest mismatch")
            start_epoch = int(last["epoch"]) + 1

        for epoch in range(start_epoch, int(args.epochs) + 1):
            plan = (
                expanded_head_epoch_plan(expanded_raw, epoch)
                if expanded_raw is not None
                else None
            )
            output = checkpoints / f"epoch_{epoch:02d}.pt"
            if plan is None:
                result = _recover_or_train_epoch(
                    epoch=epoch,
                    parent=parent,
                    parent_digest=parent_digest,
                    output=output,
                    corpus=corpus,
                    manifest_identity=manifest_identity,
                    device=device,
                    batch_size=int(args.batch_size),
                    split_seed=int(args.split_seed),
                    source_digest=source_digest,
                    run_name=args.run_name,
                )
            elif output.is_file():
                saved = checkpoint.load_checkpoint(output, map_location="cpu")
                epoch_record = dict(
                    (saved.get("extra") or {}).get(
                        "multi_teacher_core_bootstrap"
                    )
                    or {}
                )
                rehearsal = dict(
                    (saved.get("extra") or {}).get("expert_rehearsal") or {}
                )
                if (
                    epoch_record.get("schema")
                    != "poke_bot.multi_teacher_expanded_core_epoch/v1"
                    or int(epoch_record.get("epoch", -1)) != epoch
                    or epoch_record.get("training_parent_digest")
                    != source_digest
                    or epoch_record.get("parent_digest") != parent_digest
                    or epoch_record.get("manifest_digest")
                    != manifest_identity.digest
                    or epoch_record.get("expanded_target_schema_digest")
                    != expanded_identity["target_schema_digest"]
                    or epoch_record.get("expanded_schedule_digest")
                    != expanded_identity["schedule_digest"]
                    or epoch_record.get("expanded_epoch_plan")
                    != plan.as_dict()
                    or epoch_record.get("decision_fusion")
                    != fusion_identity
                    or epoch_record.get("teacher_behavior_distillation")
                    != teacher_behavior_record
                ):
                    raise RuntimeError(
                        f"existing expanded core epoch identity drift: {output}"
                    )
                result = {
                    "candidate_digest": checkpoint.checkpoint_digest(output),
                    "train_metrics": rehearsal.get("train_metrics"),
                    "validation_metrics": rehearsal.get(
                        "validation_metrics"
                    ),
                    "reused": True,
                }
            else:
                result = supervised_rehearsal_step(
                    corpus,
                    base_ckpt=parent,
                    output_path=output,
                    parent_digest=parent_digest,
                    rehearsal_iteration=epoch,
                    manifest_identity={
                        **manifest_identity.as_dict(),
                        "expanded_strategic_targets": expanded_targets,
                    },
                    epochs=1,
                    lr=5e-5,
                    requested_batch_size=int(args.batch_size),
                    seed=20260723 + epoch,
                    corpus_split_seed=int(args.split_seed),
                    device=device,
                    aux_loss_weight=0.05,
                    opp_hand_loss_weight=0.05,
                    opp_remainder_loss_weight=0.05,
                    lethal_threat_loss_weight=0.025,
                    prize_race_loss_weight=0.025,
                    alakazam_guide_loss_weight=0.0,
                    expanded_head_loss_weights=dict(plan.loss_weights),
                    expanded_head_schedule=plan.as_dict(),
                    output_archetype_id="unknown",
                    output_model_id=(
                        f"{args.run_name}.expanded-core.epoch{epoch:02d}"
                    ),
                    teacher_policy_targets=teacher_policy_targets,
                    teacher_policy_weight=(
                        float(args.teacher_policy_weight)
                        if args.teacher_behavior_distillation
                        else 0.0
                    ),
                    teacher_policy_target_digest=(
                        str(teacher_behavior_record["target_digest"])
                        if teacher_behavior_record is not None
                        else ""
                    ),
                    teacher_policy_checkpoint_digests=(
                        tuple(teacher_digests)
                        if teacher_behavior_record is not None
                        else ()
                    ),
                    extra_updates={
                        "multi_teacher_core_bootstrap": {
                            "schema": (
                                "poke_bot.multi_teacher_expanded_core_epoch/v1"
                            ),
                            "epoch": epoch,
                            "initialization_digest": initialization_digest,
                            "training_parent_digest": source_digest,
                            "parent_digest": parent_digest,
                            "manifest_digest": manifest_identity.digest,
                            "expanded_target_schema_digest": (
                                expanded_identity[
                                    "target_schema_digest"
                                ]
                            ),
                            "expanded_schedule_digest": (
                                expanded_identity["schedule_digest"]
                            ),
                            "expanded_epoch_plan": plan.as_dict(),
                            "expanded_manifest_targets": expanded_targets,
                            "runtime_enabled_heads": [],
                            "decision_fusion": fusion_identity,
                            "teacher_behavior_distillation": (
                                teacher_behavior_record
                            ),
                        }
                    },
                )
            expanded_checkpoint_contract = (
                validate_expanded_epoch_checkpoint(
                    output,
                    plan=plan,
                    identity=expanded_identity,
                    train_metrics=result.get("train_metrics"),
                    validation_metrics=result.get("validation_metrics"),
                )
                if plan is not None
                else None
            )
            metric = float(
                (result.get("validation_metrics") or {}).get(
                    "total_loss", math.inf
                )
            )
            if not math.isfinite(metric):
                raise RuntimeError("core refresh validation metric is not finite")
            row = {
                "epoch": epoch,
                "checkpoint": str(output),
                "checkpoint_digest": str(result["candidate_digest"]),
                "parent_digest": parent_digest,
                "validation_loss": metric,
                "validation_accuracy": float(
                    (result.get("validation_metrics") or {}).get(
                        "policy_acc", 0.0
                    )
                ),
                "train_metrics": result.get("train_metrics"),
                "validation_metrics": result.get("validation_metrics"),
                "reused": bool(result.get("reused")),
                **(
                    {
                        "expanded_epoch_plan": plan.as_dict(),
                        "expanded_head_training": (
                            expanded_checkpoint_contract
                        ),
                    }
                    if plan is not None
                    else {}
                ),
            }
            history.append(row)
            selection_eligible = (
                plan is None
                or set(plan.enabled_heads) == set(EXPANDED_HEAD_IDS)
            )
            if selection_eligible:
                if metric < best_metric - float(args.min_delta):
                    best_metric = metric
                    best_path = str(output)
                    best_digest = str(result["candidate_digest"])
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            parent = output
            parent_digest = str(result["candidate_digest"])
            state = {
                "schema": STATE_SCHEMA,
                "status": "training",
                "initialization": str(initialization_path),
                "initialization_digest": initialization_digest,
                "training_parent": str(source_parent),
                "training_parent_digest": source_digest,
                "teacher_checkpoint_digests": teacher_digests,
                "manifest": manifest_identity.as_dict(),
                "manifest_digest": manifest_identity.digest,
                "split_seed": int(args.split_seed),
                "decision_fusion": fusion_identity,
                "teacher_behavior_distillation_enabled": bool(
                    args.teacher_behavior_distillation
                ),
                "teacher_policy_weight": (
                    float(args.teacher_policy_weight)
                    if args.teacher_behavior_distillation
                    else 0.0
                ),
                "teacher_behavior_distillation": teacher_behavior_record,
                "history": history,
                "best_path": best_path,
                "best_digest": best_digest,
                "best_metric": best_metric,
                "bad_epochs": bad_epochs,
                "patience": int(args.patience),
                "epochs_max": int(args.epochs),
                **(
                    {
                        "expanded_target_schema_digest": (
                            expanded_identity["target_schema_digest"]
                        ),
                        "expanded_schedule_digest": (
                            expanded_identity["schedule_digest"]
                        ),
                        "expanded_schedule": expanded_identity["schedule"],
                        "expanded_manifest_targets": expanded_targets,
                        "expanded_hot_start": expanded_hot_start,
                        "runtime_enabled_heads": [],
                    }
                    if expanded_identity is not None
                    else {}
                ),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(state_path, state)
            print(
                f"[core-refresh] epoch={epoch} val_loss={metric:.6f} "
                f"best={best_metric:.6f} patience={bad_epochs}/{int(args.patience)}",
                flush=True,
            )
            if (
                expanded_identity is None
                and bad_epochs >= int(args.patience)
            ):
                break
    finally:
        cache.release()

    if expanded_identity is not None and [
        int(row.get("epoch", -1)) for row in history
    ] != list(
        range(
            1,
            int(expanded_identity["schedule"]["total_epochs"]) + 1,
        )
    ):
        raise RuntimeError(
            "expanded cumulative core did not complete exact epochs 1..25"
        )
    best = Path(best_path).resolve()
    if not best.is_file() or checkpoint.checkpoint_digest(best) != best_digest:
        raise RuntimeError("selected refreshed core identity is invalid")
    validate_zero_dormant_checkpoint(best)
    load_model_from_checkpoint(best, device="cpu")
    best_payload = checkpoint.load_checkpoint(best, map_location="cpu")
    best_expanded_training = (
        dict(
            (best_payload.get("extra") or {}).get(
                "expanded_head_training"
            )
            or {}
        )
        if expanded_identity is not None
        else None
    )
    if expanded_identity is not None and (
        best_expanded_training.get("schema")
        != "poke_bot.expanded_head_training/v1"
        or set(best_expanded_training.get("trained_heads") or ())
        != set(EXPANDED_HEAD_IDS)
        or set(
            best_expanded_training.get("gradient_enabled_heads") or ()
        )
        != set(EXPANDED_HEAD_IDS)
        or best_expanded_training.get("runtime_enabled_heads") != []
        or best_expanded_training.get("target_schema_digest")
        != expanded_identity["target_schema_digest"]
        or best_expanded_training.get("schedule_digest")
        != expanded_identity["schedule_digest"]
    ):
        raise RuntimeError(
            "selected cumulative core has incomplete expanded-head training"
        )
    if fusion_identity is not None:
        best_config = dict(best_payload.get("model_config") or {})
        best_provenance = dict(best_payload.get("provenance") or {})
        fusion_inventory = dict(best_provenance.get("decision_fusion") or {})
        fusion_state = dict(best_payload.get("model_state_dict") or {})
        output_weight = fusion_state.get("decision_fusion.residual.2.weight")
        parent_payload = checkpoint.load_checkpoint(
            source_parent, map_location="cpu"
        )
        parent_fusion_initialization = dict(
            (parent_payload.get("extra") or {}).get(
                "decision_fusion_teacher_initialization"
            )
            or {}
        )
        if (
            best_config.get("decision_fusion_enabled") is not True
            or best_config.get("decision_fusion_runtime_enabled") is not True
            or fusion_inventory.get("schema") != DECISION_FUSION_SCHEMA
            or fusion_inventory.get("runtime_enabled") is not True
            or fusion_inventory.get("required_heads")
            != list(DECISION_FUSION_REQUIRED_HEADS)
            or not isinstance(output_weight, torch.Tensor)
            or not bool(torch.count_nonzero(output_weight).item())
            or parent_fusion_initialization.get("schema")
            != FUSION_TEACHER_INITIALIZATION_SCHEMA
            or not parent_fusion_initialization.get("contributors")
        ):
            raise RuntimeError(
                "selected cumulative core lacks trained causal decision fusion"
            )
    initialized = checkpoint.load_checkpoint(source_parent, map_location="cpu")
    selected = best_payload
    initialized_adapters = {
        name: value
        for name, value in dict(initialized.get("model_state_dict") or {}).items()
        if name.startswith("matchup_adapter_bank.")
    }
    selected_adapters = {
        name: value
        for name, value in dict(selected.get("model_state_dict") or {}).items()
        if name.startswith("matchup_adapter_bank.")
    }
    if initialized_adapters.keys() != selected_adapters.keys() or any(
        not torch.equal(initialized_adapters[name], selected_adapters[name])
        for name in initialized_adapters
    ):
        raise RuntimeError("core refresh modified specialist matchup adapters")
    frozen = freeze_model(
        registry_root=args.registry_root,
        family=family_name,
        display_name=(
            f"Deck-Agnostic Cumulative Core from {len(teachers)} Frozen Specialists"
        ),
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "initialization_checkpoint": str(initialization_path),
            "initialization_digest": initialization_digest,
            "training_parent_checkpoint": str(source_parent),
            "training_parent_digest": source_digest,
            "teachers": teachers,
            "teacher_contribution": "equal_weight_parameter_space_mean",
            "teacher_behavior_distillation": teacher_behavior_record,
            "teacher_policy_weight": (
                float(args.teacher_policy_weight)
                if args.teacher_behavior_distillation
                else 0.0
            ),
            "teacher_gradients_allowed": False,
            "teacher_optimizer_steps_allowed": False,
            "specialist_matchup_adapters_reset": True,
            "core_manifest": manifest_identity.as_dict(),
            "objective": "multi_teacher_balanced_deck_agnostic_core_refresh",
            "all_auxiliary_heads_trained": True,
            "epochs_max": int(args.epochs),
            "early_stop_patience": int(args.patience),
            "history": history,
            **(
                {
                    "expanded_head_migration": (
                        expanded_hot_start.get(
                            "expanded_head_migration"
                        )
                        if expanded_hot_start is not None
                        else None
                    ),
                    "expanded_target_contract": expanded_targets,
                    "expanded_target_schema_digest": expanded_identity[
                        "target_schema_digest"
                    ],
                    "expanded_schedule_schema": expanded_identity[
                        "schedule_schema"
                    ],
                    "expanded_schedule_digest": expanded_identity[
                        "schedule_digest"
                    ],
                    "expanded_schedule": expanded_identity["schedule"],
                    "expanded_head_training": best_expanded_training,
                    "runtime_enabled_heads": [],
                    "decision_fusion": fusion_identity,
                    "decision_fusion_teacher_initialization": (
                        fusion_teacher_initialization
                    ),
                    "flat_policy_remains_authoritative": (
                        fusion_identity is None
                    ),
                }
                if expanded_identity is not None
                else {}
            ),
        },
        evidence={
            "kind": "episode_disjoint_multideck_expert_validation",
            "best_metric": best_metric,
            "epochs_completed": len(history),
            "archetypes": len(core_payload["totals"]["records_per_archetype"]),
            "gameplay_regression_status": "required_before_specialist_handoff",
        },
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen["model_path"]).parent)
    ready = {
        "schema": READY_SCHEMA,
        "status": "candidate_ready_for_gameplay_regression",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "family": family_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_digest": frozen["checkpoint_digest"],
        "initialization": str(initialization_path),
        "initialization_digest": initialization_digest,
        "training_parent": str(source_parent),
        "training_parent_digest": source_digest,
        "teacher_checkpoint_digests": teacher_digests,
        "teacher_contribution": "equal_weight_parameter_space_mean",
        "teacher_behavior_distillation_enabled": bool(
            args.teacher_behavior_distillation
        ),
        "teacher_policy_weight": (
            float(args.teacher_policy_weight)
            if args.teacher_behavior_distillation
            else 0.0
        ),
        "teacher_behavior_distillation": teacher_behavior_record,
        "specialist_matchup_adapters_reset": True,
        "core_manifest": manifest_identity.path,
        "core_manifest_sha256": manifest_identity.digest,
        "records": manifest_identity.records,
        "decisions": manifest_identity.decisions,
        "archetypes": len(core_payload["totals"]["records_per_archetype"]),
        "epochs_completed": len(history),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "split_seed": int(args.split_seed),
        "best_metric": best_metric,
        "gameplay_regression_required": True,
        "gameplay_regression_passed": False,
        **(
            {
                "expanded_target_contract": expanded_targets,
                "expanded_target_schema": expanded_identity[
                    "target_schema"
                ],
                "expanded_target_schema_digest": expanded_identity[
                    "target_schema_digest"
                ],
                "expanded_schedule_schema": expanded_identity[
                    "schedule_schema"
                ],
                "expanded_schedule_digest": expanded_identity[
                    "schedule_digest"
                ],
                "expanded_schedule": expanded_identity["schedule"],
                "expanded_head_training": best_expanded_training,
                "expanded_heads_trained": list(EXPANDED_HEAD_IDS),
                "runtime_enabled_heads": [],
                "decision_fusion": fusion_identity,
                "decision_fusion_teacher_initialization": (
                    fusion_teacher_initialization
                ),
                "flat_policy_remains_authoritative": (
                    fusion_identity is None
                ),
            }
            if expanded_identity is not None
            else {}
        ),
    }
    atomic_json(args.ready, ready)
    atomic_json(state_path, {**state, "status": "candidate_ready", "ready": ready})
    print(json.dumps(ready, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
