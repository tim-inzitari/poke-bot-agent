#!/usr/bin/env python3
"""Validate the staged Alakazam/Marnie S++ r192 candidate.

This controller intentionally has two modes:

* the default is a read-only preflight of the staged checksum-bound artifacts;
* ``--apply`` accepts only a checksum-verified receipt proving the managed
  trainer is already inactive at an immutable committed boundary.  A finite
  pause marker alone can never authorize a stop, drop-in publication, or
  restart.

There is no generic iter-1 pause in r175: the trainer only emits its 30 second
gate-boundary pause once ``completed_iteration >= minimum_terminal_iteration``.
The r175 registry pins that minimum to five.  Activation therefore remains
unarmed until either the managed trainer is already inactive at a proven
receipt boundary or a later source implements a trainer-owned fence.  This
file never sends a signal to a process directly and never deploys files to
another host.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STAGE_SCHEMA = "poke_bot.alakazam_marnie_splusplus_r192_stage/v1"
RUNTIME_REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
FROZEN_REGISTRY_SCHEMA = "poke_bot.frozen_specialist_registry/v1"
GATE_SCHEMA = "poke_bot.competition_gate_program/v1"
PIN_FLOORS_SCHEMA = "poke_bot.owner_public_mix_pin_floors/v1"

OWNER_REVISION = 192
PARENT_REVISION = 175
SPECIALIST_ID = "alakazam"
OPPONENT_ID = "specialist-marnie-final-format-h10-f20efb20f5c3"
HISTORICAL_MARNIE_ID = "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"
CHECKPOINT_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
CONTENT_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
TIER = "S++"
WEIGHT = 4.0
FLOOR_GAMES_PER_SET = 1024
STRONG_PUBLIC_PRACTICE_GAMES = 4586
DIVERSE_PUBLIC_GAMES = 2586
GATE_GAMES_PER_OPPONENT = 250
GATE_SEAT_GAMES = 125
GATE_ROSTER_SIZE = 18
GATE_GAMES_TOTAL = 4500
BOUNDARY_PAUSE_SECONDS = 30.0
FIRST_GUARANTEED_BOUNDARY_ITERATION = 5
STRONG_PUBLIC_PRACTICE_FLOOR_GAMES = 1024
REQUIRED_DEPLOYMENT_INPUT_LABELS = frozenset(
    {
        "launch_pure_rl",
        "train_pure_rl",
        "launch_active_specialist",
        "activation_controller",
        "dropin_template",
        "boundary_service_template",
        "stop_budget_template",
        "strong_public_gate",
        "public_multi_env_safety",
        "r182_transport_contract",
        "baseline_manifest",
        "h10_marnie_model_provenance",
    }
)
ACTIVATION_ARTIFACT_DEPLOYMENT_LABELS = {
    "controller": "activation_controller",
    "dropin_template": "dropin_template",
    "boundary_service_template": "boundary_service_template",
    "stop_budget_template": "stop_budget_template",
}
STOP_BUDGET_GUARD = {
    "target_filename": "61-marnie-splusplus-r192-stop-budget.conf",
    "timeout_stop_seconds": 8,
    "timeout_stop_usec": 8_000_000,
    "kill_mode": "control-group",
    "send_sigkill": True,
}
R192_MIGRATION_FLAG = "--allow-clean-boundary-design-migration"
R192_MIGRATION_REASON_FLAG = "--boundary-design-migration-reason"
R192_MIGRATION_REASON = (
    "owner_r192_marnie_splusplus_post_iteration5_receipt_backed_migration"
)
UNARMED_APPLY_REASON = (
    "r192 automatic activation is unarmed: current r175 lacks a trainer-owned "
    "handoff fence; require a receipt-proven inactive boundary or a later "
    "fence-enabled source"
)


@dataclass(frozen=True)
class Artifact:
    """A file whose bytes are bound into the r192 staging receipt."""

    key: str
    path: Path
    digest: str


@dataclass(frozen=True)
class BoundaryObservation:
    """A conservative lower-bound deadline for the visible hard pause."""

    commit_path: Path
    commit: dict[str, Any]
    commit_sha256: str
    # The pause cannot have begun before this point because the immediately
    # preceding log read did not contain its marker.  Therefore this is the
    # earliest possible pause deadline and is safe to use after rehashing.
    safe_pause_deadline_monotonic: float


class CandidatePlanPending(RuntimeError):
    """A plan exists but its sibling provenance receipt has not landed yet."""


class CandidateDispatchMayHaveStarted(RuntimeError):
    """Do not roll back once a candidate collection plan may have dispatched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(f"{label} must be a non-empty path")
    return Path(raw).expanduser().resolve()


def _require_bound_file(artifact: Artifact) -> None:
    if not artifact.digest.startswith("sha256:"):
        raise RuntimeError(f"{artifact.key} lacks a sha256 digest")
    if not artifact.path.is_file():
        raise RuntimeError(f"{artifact.key} is missing: {artifact.path}")
    actual = _sha256(artifact.path)
    if actual != artifact.digest:
        raise RuntimeError(
            f"{artifact.key} checksum mismatch: expected={artifact.digest} actual={actual}"
        )


def _artifact(stage: dict[str, Any], key: str) -> Artifact:
    artifacts = stage.get("staged_artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("stage receipt lacks staged_artifacts")
    raw = artifacts.get(key)
    if not isinstance(raw, dict):
        raise RuntimeError(f"stage receipt lacks staged artifact: {key}")
    path = _path(raw.get("path"), label=f"staged_artifacts.{key}.path")
    digest = str(raw.get("sha256") or raw.get("digest") or "")
    artifact = Artifact(key=key, path=path, digest=digest)
    _require_bound_file(artifact)
    return artifact


def _source_artifact(stage: dict[str, Any]) -> Artifact:
    raw = stage.get("source_runtime_registry")
    if not isinstance(raw, dict):
        raise RuntimeError("stage receipt lacks source_runtime_registry")
    artifact = Artifact(
        key="source_runtime_registry",
        path=_path(raw.get("path"), label="source_runtime_registry.path"),
        digest=str(raw.get("sha256") or raw.get("digest") or ""),
    )
    _require_bound_file(artifact)
    return artifact


def _top_level_artifact(stage: dict[str, Any], key: str) -> Artifact:
    raw = stage.get(key)
    if not isinstance(raw, dict):
        raise RuntimeError(f"stage receipt lacks {key}")
    artifact = Artifact(
        key=key,
        path=_path(raw.get("path"), label=f"{key}.path"),
        digest=str(raw.get("sha256") or raw.get("digest") or ""),
    )
    _require_bound_file(artifact)
    return artifact


def _source_child_artifact(stage: dict[str, Any], key: str) -> Artifact:
    artifacts = stage.get("source_artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("stage receipt lacks checksum-bound source_artifacts")
    raw = artifacts.get(key)
    if not isinstance(raw, dict):
        raise RuntimeError(f"stage receipt lacks source artifact: {key}")
    artifact = Artifact(
        key=f"source_{key}",
        path=_path(raw.get("path"), label=f"source_artifacts.{key}.path"),
        digest=str(raw.get("sha256") or raw.get("digest") or ""),
    )
    _require_bound_file(artifact)
    return artifact


def _value_option(values: Iterable[Any], option: str) -> str:
    rows = [str(value) for value in values]
    found: list[str] = []
    index = 0
    while index < len(rows):
        if rows[index] == option:
            if index + 1 >= len(rows):
                raise RuntimeError(f"{option} lacks a value")
            found.append(rows[index + 1])
            index += 2
            continue
        index += 1
    if len(found) != 1:
        raise RuntimeError(f"{option} must occur exactly once, found {len(found)}")
    return found[0]


def _flag_option_once(values: Iterable[Any], option: str) -> None:
    rows = [str(value) for value in values]
    negated = f"--no-{option.removeprefix('--')}"
    aliases = [
        value
        for value in rows
        if value == option
        or value == negated
        or value.startswith(option + "=")
        or value.startswith(negated + "=")
    ]
    if aliases != [option]:
        raise RuntimeError(
            f"{option} must occur exactly once without an alternate spelling"
        )


def _resolve_runtime_child(
    registry: dict[str, Any],
    field: str,
    expected: Path,
) -> None:
    root = _path(registry.get("runtime_root"), label="runtime_registry.runtime_root")
    relative = registry.get(field)
    if not isinstance(relative, str) or not relative.strip():
        raise RuntimeError(f"runtime registry lacks {field}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"runtime registry {field} escapes runtime_root") from exc
    if candidate != expected.resolve():
        raise RuntimeError(
            f"runtime registry {field} does not bind the staged {expected.name}"
        )


def _owner_requirements(owner: dict[str, Any]) -> dict[str, Any]:
    opponent = owner.get("opponent")
    historical = owner.get("historical_marnie")
    collection = owner.get("collection_contract")
    transport = owner.get("transport")
    activation = owner.get("activation")
    if (
        owner.get("schema")
        != "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1"
        or int(owner.get("owner_decision_revision") or -1) != OWNER_REVISION
        or not isinstance(opponent, dict)
        or not isinstance(historical, dict)
        or not isinstance(collection, dict)
        or not isinstance(transport, dict)
        or not isinstance(activation, dict)
    ):
        raise RuntimeError("r192 owner contract has the wrong schema or shape")
    exact = {
        "opponent_id": OPPONENT_ID,
        "checkpoint_sha256": CHECKPOINT_DIGEST,
        "content_digest": CONTENT_DIGEST,
        "tier": TIER,
        "floor_games_per_set": FLOOR_GAMES_PER_SET,
        "distinct_additional_specialist_row": True,
        "duplicate_alias_row_allowed": False,
    }
    for key, expected in exact.items():
        if opponent.get(key) != expected:
            raise RuntimeError(f"r192 owner contract mismatch: opponent.{key}")
    if float(opponent.get("weight") or -1.0) != WEIGHT:
        raise RuntimeError("r192 owner contract has the wrong S++ weight")
    if (
        historical.get("opponent_id") != HISTORICAL_MARNIE_ID
        or historical.get("must_remain_distinct") is not True
        or historical.get("may_be_collapsed_or_substituted") is not False
    ):
        raise RuntimeError("r192 owner contract does not preserve historical Marnie")
    if (
        int(collection.get("games_per_iteration") or -1) != 8196
        or int(collection.get("self_play_mirrors") or -1) != 1024
        or int(collection.get("public_mix_games") or -1) != 7172
        or int(collection.get("strong_public_practice_games") or -1)
        != STRONG_PUBLIC_PRACTICE_GAMES
        or int(collection.get("diverse_public_games") or -1)
        != DIVERSE_PUBLIC_GAMES
        or (
            int(collection.get("strong_public_practice_games") or -1)
            + int(collection.get("diverse_public_games") or -1)
            != int(collection.get("public_mix_games") or -1)
        )
        or float(collection.get("ordinary_strong_public_minimum_share") or -1.0)
        != 0.04
        or collection.get("h10_floor_enforcement")
        != "exact_active_gate_strong_public_practice_floor"
        or collection.get("legacy_diverse_public_h10_pin_removed_on_activation")
        is not True
        or collection.get("exact_total_unchanged") is not True
        or int(collection.get("public_replacement_lanes") or -1) != 32
    ):
        raise RuntimeError("r192 owner collection contract changed")
    if transport != {
        "r182_default_deny_unchanged": True,
        "other_r182_pairs_unchanged": True,
        "prior_pack4_eligible_group": "diverse_public",
        "activation_training_group": "strong_public_practice",
        "dispatch_mode": "singleton_remote_play",
        "pack4_attested_for_activation_group": False,
        "separate_exact_group_retention_attestation_required_for_pack4": True,
    }:
        raise RuntimeError("r192 owner transport contract changed")
    if (
        activation.get("boundary")
        != (
            "receipt_backed_inactive_boundary_or_trainer_owned_fence_enabled_"
            "clean_pause_after_completed_iteration5"
        )
        or int(
            activation.get(
                "first_guaranteed_activation_boundary_completed_iteration"
            )
            or -1
        )
        != FIRST_GUARANTEED_BOUNDARY_ITERATION
        or float(activation.get("boundary_pause_seconds") or -1.0)
        != BOUNDARY_PAUSE_SECONDS
        or activation.get("allow_clean_boundary_design_migration") is not True
        or activation.get("boundary_design_migration_reason")
        != R192_MIGRATION_REASON
        or activation.get("requires_checksum_exact_roster_binding") is not True
        or activation.get("requires_runtime_registry_binding") is not True
        or activation.get("requires_dispatch_provenance_binding") is not True
        or activation.get("requires_focused_exact_retention_tests") is not True
        or activation.get(
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        )
        is not False
        or activation.get("automatic_managed_restart_armed") is not False
        or activation.get("trainer_owned_handoff_fence_required") is not True
        or activation.get("current_r175_source_has_trainer_owned_handoff_fence")
        is not False
        or activation.get("proven_inactive_receipt_boundary_alternative_required")
        is not True
        or activation.get("training_restart_before_validation_allowed") is not False
        or activation.get("interrupt_active_collection_allowed") is not False
    ):
        raise RuntimeError("r192 owner activation guard is incomplete")
    return opponent


def _single_row(rows: Iterable[Any], opponent_id: str, *, label: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if isinstance(row, dict) and row.get("opponent_id") == opponent_id]
    if len(matches) != 1:
        raise RuntimeError(f"{label} must contain exactly one {opponent_id}, found {len(matches)}")
    return matches[0]


def _verify_gate(gate: dict[str, Any]) -> str:
    if gate.get("schema") != GATE_SCHEMA:
        raise RuntimeError("staged active gate has the wrong schema")
    next_gate = gate.get("next_gate")
    if not isinstance(next_gate, dict):
        raise RuntimeError("staged active gate lacks next_gate")
    gate_id = str(next_gate.get("id") or "")
    roster = next_gate.get("roster")
    evaluation = next_gate.get("evaluation")
    if (
        not gate_id
        or gate.get("active_gate_id") != gate_id
        or not isinstance(roster, list)
        or not isinstance(evaluation, dict)
        or len(roster) != GATE_ROSTER_SIZE
        or int(evaluation.get("games_per_opponent") or -1)
        != GATE_GAMES_PER_OPPONENT
        or int(evaluation.get("seat0_games_per_opponent") or -1)
        != GATE_SEAT_GAMES
        or int(evaluation.get("seat1_games_per_opponent") or -1)
        != GATE_SEAT_GAMES
        or int(evaluation.get("games_total") or -1) != GATE_GAMES_TOTAL
    ):
        raise RuntimeError("staged r192 gate is not the exact 18×250/125/125 contract")
    ids = [str(row.get("opponent_id") or "") for row in roster if isinstance(row, dict)]
    digests = [str(row.get("content_digest") or "") for row in roster if isinstance(row, dict)]
    if len(ids) != len(roster) or len(set(ids)) != len(ids) or not all(ids):
        raise RuntimeError("staged r192 gate has ambiguous opponent IDs")
    if len(digests) != len(roster) or len(set(digests)) != len(digests) or not all(digests):
        raise RuntimeError("staged r192 gate has duplicate or missing package digests")
    candidate = _single_row(roster, OPPONENT_ID, label="staged r192 gate")
    historical = _single_row(roster, HISTORICAL_MARNIE_ID, label="staged r192 gate")
    if (
        candidate.get("tier") != TIER
        or float(candidate.get("weight") or -1.0) != WEIGHT
        or candidate.get("content_digest") != CONTENT_DIGEST
        or candidate.get("frozen_checkpoint_digest") != CHECKPOINT_DIGEST
        or candidate.get("frozen_specialist") is not True
        or int(candidate.get("strong_public_practice_floor_games") or -1)
        != STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
        or historical.get("content_digest") == CONTENT_DIGEST
    ):
        raise RuntimeError("staged r192 gate does not preserve the distinct S++ Marnie row")
    semantics = gate.get("active_gate_semantics")
    exact_semantics = {
        "opponent_id": OPPONENT_ID,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "content_digest": CONTENT_DIGEST,
        "tier": TIER,
        "weight": WEIGHT,
        "strong_public_practice_floor_games": STRONG_PUBLIC_PRACTICE_FLOOR_GAMES,
    }
    if (
        not isinstance(semantics, dict)
        or semantics.get("exact_additional_splusplus_specialist") != exact_semantics
    ):
        raise RuntimeError("staged r192 gate lacks its exact S++ semantic binding")
    return gate_id


def _tier_and_weight(row: dict[str, Any]) -> tuple[Any, float]:
    tier = row.get("premium_holdout_tier", row.get("tier"))
    raw_weight = row.get("premium_holdout_weight", row.get("weight"))
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frozen S++ row lacks numeric weight metadata") from exc
    return tier, weight


def _verify_frozen_registry(registry: dict[str, Any]) -> None:
    rows = registry.get("specialists")
    if (
        registry.get("schema") != FROZEN_REGISTRY_SCHEMA
        or int(registry.get("version") or 0) < 15
        or not isinstance(rows, list)
        or len(rows) != 15
    ):
        raise RuntimeError("staged r192 frozen registry has the wrong shape")
    candidate = _single_row(rows, OPPONENT_ID, label="staged r192 frozen registry")
    historical = _single_row(rows, HISTORICAL_MARNIE_ID, label="staged r192 frozen registry")
    tier, weight = _tier_and_weight(candidate)
    if (
        candidate.get("checkpoint_digest") != CHECKPOINT_DIGEST
        or candidate.get("content_digest") != CONTENT_DIGEST
        or candidate.get("frozen") is not True
        or candidate.get("public_mix_eligible") is not True
        or candidate.get("research_eligible") is not False
        or tier != TIER
        or weight != WEIGHT
        or historical.get("content_digest") == CONTENT_DIGEST
    ):
        raise RuntimeError("staged r192 frozen registry does not bind distinct Marnie S++")


def _verify_pin_floors(pin_floors: dict[str, Any]) -> None:
    pins = pin_floors.get("pins")
    if pin_floors.get("schema") != PIN_FLOORS_SCHEMA or not isinstance(pins, list):
        raise RuntimeError("staged r192 public-mix pin sidecar has the wrong schema")
    marnie_pins = [
        dict(row)
        for row in pins
        if isinstance(row, dict) and row.get("package_id") == OPPONENT_ID
    ]
    if marnie_pins:
        raise RuntimeError(
            "staged r192 sidecar must not reinsert H10 Marnie into diverse_public"
        )
    matches = [
        dict(row)
        for row in pins
        if isinstance(row, dict)
        and row.get("package_id")
        == "specialist-crustle-final-format-h10-7efd8d4113e7"
    ]
    if len(matches) != 1:
        raise RuntimeError("staged r192 sidecar must preserve exactly one Crustle pin")
    candidate = matches[0]
    if (
        candidate.get("checkpoint_sha256")
        != "sha256:7efd8d4113e736d28576bdbfa1c9d1c3f3a7cf1a31a0b3cfadd1e7f82cf08955"
        or candidate.get("content_digest")
        != "sha256:359e3b4fed00502e58be4631576501b6f63523226ec92f2d75446df085b19afa"
        or int(candidate.get("floor_games_per_set") or -1) != 512
    ):
        raise RuntimeError("staged r192 sidecar does not preserve the exact Crustle pin")


def _verify_runtime_registry(
    registry: dict[str, Any],
    *,
    owner_contract: Artifact,
    gate_artifact: Artifact,
    frozen_artifact: Artifact,
    pin_artifact: Artifact,
) -> None:
    if (
        registry.get("schema") != RUNTIME_REGISTRY_SCHEMA
        or int(registry.get("owner_decision_revision") or -1) != OWNER_REVISION
    ):
        raise RuntimeError("staged r192 runtime registry has the wrong identity")
    _resolve_runtime_child(
        registry, "active_gate_contract", gate_artifact.path
    )
    _resolve_runtime_child(
        registry, "frozen_specialist_registry", frozen_artifact.path
    )
    isolated = registry.get("isolated_refresh_contract")
    specialists = registry.get("specialists")
    alakazam = specialists.get(SPECIALIST_ID) if isinstance(specialists, dict) else None
    if not isinstance(isolated, dict) or not isinstance(alakazam, dict):
        raise RuntimeError("staged r192 runtime registry lacks Alakazam owner metadata")
    pin = alakazam.get("owner_grimmsnarl_pin")
    if not isinstance(pin, dict):
        raise RuntimeError("staged r192 runtime registry lacks the Marnie pin")
    for source, label in ((isolated, "isolated runtime"), (pin, "owner Marnie pin")):
        package = source.get("grimmsnarl_package_id", source.get("package_id"))
        checkpoint = source.get("grimmsnarl_checkpoint_sha256", source.get("checkpoint_sha256"))
        content = source.get("grimmsnarl_content_digest", source.get("content_digest"))
        floor = source.get("grimmsnarl_floor_per_set", source.get("floor_games_per_set"))
        tier = source.get("grimmsnarl_tier", source.get("tier"))
        weight = source.get("grimmsnarl_weight", source.get("weight"))
        if (
            package != OPPONENT_ID
            or checkpoint != CHECKPOINT_DIGEST
            or content != CONTENT_DIGEST
            or int(floor or -1) != FLOOR_GAMES_PER_SET
            or tier != TIER
            or float(weight or -1.0) != WEIGHT
        ):
            raise RuntimeError(f"staged r192 {label} is not checksum/tier/floor exact")
    if (
        pin.get("enforcement_source")
        != "exact_active_gate_strong_public_practice_floor"
        or pin.get("legacy_diverse_public_sidecar_pin_active") is not False
        or pin.get("superseded_by_exact_strong_gate_floor") is not True
    ):
        raise RuntimeError("staged r192 owner Marnie pin is not explicitly non-executing")
    if int(alakazam.get("minimum_terminal_iteration") or -1) != FIRST_GUARANTEED_BOUNDARY_ITERATION:
        raise RuntimeError("staged r192 registry changed Alakazam's minimum terminal iteration")
    if (
        float(
            _value_option(
                registry.get("common_trainer_args") or [],
                "--gate-boundary-pause-seconds",
            )
        )
        != BOUNDARY_PAUSE_SECONDS
    ):
        raise RuntimeError("staged r192 registry does not retain the exact 30s gate pause")
    if (
        float(
            _value_option(
                registry.get("common_trainer_args") or [],
                "--official-adaptive-min-share",
            )
        )
        != 0.04
    ):
        raise RuntimeError("staged r192 registry does not retain adaptive min share 0.04")
    trainer_args = registry.get("common_trainer_args") or []
    _flag_option_once(trainer_args, R192_MIGRATION_FLAG)
    if any(
        str(value).startswith(R192_MIGRATION_REASON_FLAG + "=")
        for value in trainer_args
    ):
        raise RuntimeError("staged r192 registry uses an alternate migration-reason spelling")
    if (
        _value_option(trainer_args, R192_MIGRATION_REASON_FLAG)
        != R192_MIGRATION_REASON
    ):
        raise RuntimeError("staged r192 registry has the wrong clean-boundary migration reason")
    stage = registry.get("r192_stage")
    bindings = stage.get("artifact_bindings") if isinstance(stage, dict) else None
    if not isinstance(bindings, dict):
        raise RuntimeError("staged r192 runtime registry lacks r192_stage.artifact_bindings")
    if stage.get("boundary_design_migration") != {
        "allow_clean_boundary_design_migration": True,
        "reason": R192_MIGRATION_REASON,
    }:
        raise RuntimeError("staged r192 runtime registry lacks exact migration provenance")
    if stage.get("stop_budget_guard") != STOP_BUDGET_GUARD:
        raise RuntimeError("staged r192 runtime registry lacks exact stop-budget guard")
    root = _path(registry.get("runtime_root"), label="runtime_registry.runtime_root")
    for key, artifact in (
        ("owner_contract", owner_contract),
        ("active_gate_contract", gate_artifact),
        ("frozen_specialist_registry", frozen_artifact),
        ("public_mix_pin_floors", pin_artifact),
    ):
        binding = bindings.get(key)
        if not isinstance(binding, dict):
            raise RuntimeError(f"staged r192 artifact_bindings lacks {key}")
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"staged r192 artifact binding path missing: {key}")
        bound_path = Path(raw_path).expanduser()
        if not bound_path.is_absolute():
            bound_path = root / bound_path
        if (
            bound_path.resolve() != artifact.path.resolve()
            or str(binding.get("sha256") or binding.get("digest") or "")
            != artifact.digest
        ):
            raise RuntimeError(f"staged r192 artifact binding mismatch: {key}")


def _verify_source_runtime_registry(
    source: dict[str, Any],
    *,
    source_gate_artifact: Artifact,
    source_frozen_artifact: Artifact,
) -> tuple[Path, str]:
    if (
        source.get("schema") != RUNTIME_REGISTRY_SCHEMA
        or int(source.get("owner_decision_revision") or -1) != PARENT_REVISION
    ):
        raise RuntimeError("staged parent is not the r175 runtime registry")
    specialists = source.get("specialists")
    alakazam = specialists.get(SPECIALIST_ID) if isinstance(specialists, dict) else None
    if not isinstance(alakazam, dict):
        raise RuntimeError("r175 runtime registry lacks Alakazam")
    if int(alakazam.get("minimum_terminal_iteration") or -1) != FIRST_GUARANTEED_BOUNDARY_ITERATION:
        raise RuntimeError("r175 does not expose the expected first safe gate boundary")
    if float(_value_option(source.get("common_trainer_args") or [], "--gate-boundary-pause-seconds")) != BOUNDARY_PAUSE_SECONDS:
        raise RuntimeError("r175 source registry does not expose the exact 30s gate pause")
    _resolve_runtime_child(source, "active_gate_contract", source_gate_artifact.path)
    _resolve_runtime_child(
        source, "frozen_specialist_registry", source_frozen_artifact.path
    )
    log = _path(alakazam.get("log"), label="r175 Alakazam log")
    run_name = str(alakazam.get("run_name") or "")
    if not run_name:
        raise RuntimeError("r175 runtime registry lacks Alakazam run_name")
    return log, run_name


def _verify_deployment_inputs(stage: dict[str, Any]) -> list[dict[str, str]]:
    raw = stage.get("deployment_inputs")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("stage receipt lacks checksum-bound deployment_inputs")
    values: list[dict[str, str]] = []
    labels: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise RuntimeError("deployment input must be an object")
        label = str(entry.get("label") or "")
        if not label or label in labels:
            raise RuntimeError("deployment input labels must be unique")
        artifact = Artifact(
            key=f"deployment_input:{label}",
            path=_path(entry.get("path"), label=f"deployment_inputs.{label}.path"),
            digest=str(entry.get("sha256") or entry.get("digest") or ""),
        )
        _require_bound_file(artifact)
        labels.add(label)
        values.append({"label": label, "path": str(artifact.path), "sha256": artifact.digest})
    missing = sorted(REQUIRED_DEPLOYMENT_INPUT_LABELS - labels)
    if missing:
        raise RuntimeError(f"stage deployment inputs omit required launch code: {missing}")
    return values


def _verify_stop_budget_template(path: Path) -> None:
    """Check the receipt-bound temporary guard before a boundary can arm."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"r192 stop-budget template is unreadable: {path}") from exc
    settings: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings.setdefault(key, []).append(value)
    expected = {
        "TimeoutStopSec": "8s",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
    }
    if any(settings.get(key) != [value] for key, value in expected.items()):
        raise RuntimeError(
            "r192 stop-budget template must contain exactly one 8s/control-group/SIGKILL guard"
        )


def _verify_activation_artifacts(
    stage: dict[str, Any], deployment_inputs: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Verify the receipt's activation aliases match the hashed input list."""

    raw = stage.get("activation_artifacts")
    if not isinstance(raw, dict):
        raise RuntimeError("stage receipt lacks checksum-bound activation_artifacts")
    by_label = {entry["label"]: entry for entry in deployment_inputs}
    result: dict[str, dict[str, str]] = {}
    for key, label in ACTIVATION_ARTIFACT_DEPLOYMENT_LABELS.items():
        value = raw.get(key)
        if not isinstance(value, dict):
            raise RuntimeError(f"stage activation_artifacts lacks {key}")
        artifact = Artifact(
            key=f"activation_artifacts.{key}",
            path=_path(value.get("path"), label=f"activation_artifacts.{key}.path"),
            digest=str(value.get("sha256") or value.get("digest") or ""),
        )
        _require_bound_file(artifact)
        deployment = by_label.get(label)
        if (
            not isinstance(deployment, dict)
            or artifact.path != Path(deployment["path"]).resolve()
            or artifact.digest != deployment["sha256"]
        ):
            raise RuntimeError(
                f"stage activation artifact {key} does not match deployment input {label}"
            )
        if key == "stop_budget_template":
            _verify_stop_budget_template(artifact.path)
        result[key] = {"path": str(artifact.path), "sha256": artifact.digest}
    return result


def preflight(
    *,
    stage_path: Path,
    owner_contract_path: Path,
) -> dict[str, Any]:
    """Validate all r192 staging inputs without changing local or live state."""

    stage_path = stage_path.expanduser().resolve()
    owner_contract_path = owner_contract_path.expanduser().resolve()
    if not stage_path.is_file() or not owner_contract_path.is_file():
        raise RuntimeError("stage receipt or owner contract is missing")
    stage = _read_json(stage_path)
    owner = _read_json(owner_contract_path)
    owner_artifact = Artifact(
        key="owner_contract",
        path=owner_contract_path,
        digest=_sha256(owner_contract_path),
    )
    if (
        stage.get("schema") != STAGE_SCHEMA
        or int(stage.get("owner_decision_revision") or -1) != OWNER_REVISION
        or stage.get("status") != "staged_non_active"
        or stage.get("active_before_receipt_backed_activation") is not False
        or stage.get("training_interrupted") is not False
        or stage.get("interrupt_active_collection_allowed") is not False
        or stage.get("selector_or_service_changed") is not False
        or stage.get("remote_deployment_performed") is not False
        or stage.get(
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        )
        is not False
        or stage.get("automatic_managed_restart_armed") is not False
        or stage.get("trainer_owned_handoff_fence_required") is not True
        or stage.get("current_r175_source_has_trainer_owned_handoff_fence")
        is not False
    ):
        raise RuntimeError("r192 stage receipt is not an inactive r192 stage")
    if int(stage.get("first_guaranteed_activation_boundary_completed_iteration") or -1) != FIRST_GUARANTEED_BOUNDARY_ITERATION:
        raise RuntimeError("r192 stage does not target the first guaranteed gate-pause boundary")
    if float(stage.get("boundary_pause_seconds") or -1.0) != BOUNDARY_PAUSE_SECONDS:
        raise RuntimeError("r192 stage does not bind the 30 second gate pause")
    if stage.get("boundary_design_migration") != {
        "allow_clean_boundary_design_migration": True,
        "reason": R192_MIGRATION_REASON,
    }:
        raise RuntimeError("r192 stage does not bind its exact migration authority")
    if stage.get("stop_budget_guard") != STOP_BUDGET_GUARD:
        raise RuntimeError("r192 stage does not bind the exact 8-second stop-budget guard")
    if stage.get("stop_budget_guard") != STOP_BUDGET_GUARD:
        raise RuntimeError("r192 stage does not bind the exact temporary stop budget")
    stage_collection = stage.get("collection_contract")
    if not isinstance(stage_collection, dict) or stage_collection != {
        "games_per_iteration": 8196,
        "self_play_mirrors": 1024,
        "public_mix_games": 7172,
        "strong_public_practice_games": STRONG_PUBLIC_PRACTICE_GAMES,
        "diverse_public_games": DIVERSE_PUBLIC_GAMES,
        "ordinary_strong_public_minimum_share": 0.04,
        "h10_executable_floor_owner": (
            "exact_active_gate_strong_public_practice_floor"
        ),
        "legacy_diverse_public_h10_pin_removed_on_activation": True,
    }:
        raise RuntimeError("r192 stage collection contract changed")
    if stage.get("transport") != {
        "r182_default_deny_unchanged": True,
        "other_r182_pairs_unchanged": True,
        "prior_pack4_eligible_group": "diverse_public",
        "activation_training_group": "strong_public_practice",
        "dispatch_mode": "singleton_remote_play",
        "pack4_attested_for_activation_group": False,
        "separate_exact_group_retention_attestation_required_for_pack4": True,
    }:
        raise RuntimeError("r192 stage transport contract changed")
    staged_owner_artifact = _top_level_artifact(stage, "owner_contract")
    if (
        staged_owner_artifact.path != owner_artifact.path
        or staged_owner_artifact.digest != owner_artifact.digest
    ):
        raise RuntimeError("r192 stage owner contract is not the invoked typed owner source")
    _owner_requirements(owner)
    source_artifact = _source_artifact(stage)
    source_gate_artifact = _source_child_artifact(stage, "active_gate_contract")
    source_frozen_artifact = _source_child_artifact(stage, "frozen_specialist_registry")
    runtime_artifact = _artifact(stage, "runtime_registry")
    gate_artifact = _artifact(stage, "active_gate_contract")
    frozen_artifact = _artifact(stage, "frozen_specialist_registry")
    pin_artifact = _artifact(stage, "public_mix_pin_floors")
    source = _read_json(source_artifact.path)
    runtime_registry = _read_json(runtime_artifact.path)
    gate = _read_json(gate_artifact.path)
    frozen = _read_json(frozen_artifact.path)
    pin_floors = _read_json(pin_artifact.path)
    log_path, run_name = _verify_source_runtime_registry(
        source,
        source_gate_artifact=source_gate_artifact,
        source_frozen_artifact=source_frozen_artifact,
    )
    gate_id = _verify_gate(gate)
    _verify_frozen_registry(frozen)
    _verify_pin_floors(pin_floors)
    _verify_runtime_registry(
        runtime_registry,
        owner_contract=owner_artifact,
        gate_artifact=gate_artifact,
        frozen_artifact=frozen_artifact,
        pin_artifact=pin_artifact,
    )
    if runtime_registry.get("terminal_active_gate_id") != gate_id:
        raise RuntimeError("staged runtime registry does not bind the r192 terminal gate")
    deployment_inputs = _verify_deployment_inputs(stage)
    activation_artifacts = _verify_activation_artifacts(stage, deployment_inputs)
    return {
        "stage_receipt": str(stage_path),
        "stage_receipt_sha256": _sha256(stage_path),
        "owner_contract": str(owner_contract_path),
        "owner_contract_sha256": _sha256(owner_contract_path),
        "source_runtime_registry": {
            "path": str(source_artifact.path),
            "sha256": source_artifact.digest,
        },
        "source_artifacts": {
            artifact.key.removeprefix("source_"): {
                "path": str(artifact.path),
                "sha256": artifact.digest,
            }
            for artifact in (source_gate_artifact, source_frozen_artifact)
        },
        "staged_artifacts": {
            artifact.key: {"path": str(artifact.path), "sha256": artifact.digest}
            for artifact in (runtime_artifact, gate_artifact, frozen_artifact, pin_artifact)
        },
        "deployment_inputs": deployment_inputs,
        "activation_artifacts": activation_artifacts,
        "gate_id": gate_id,
        "run_name": run_name,
        "log_path": str(log_path),
        "expected_after_iteration": FIRST_GUARANTEED_BOUNDARY_ITERATION,
        "boundary_pause_seconds": BOUNDARY_PAUSE_SECONDS,
        "stop_budget_guard": dict(STOP_BUDGET_GUARD),
        "no_remote_deployment_performed": True,
        "required_remote_receipts_are_stage_inputs": True,
    }


def _read_log_delta(path: Path, offset: int) -> tuple[str, int]:
    """Read only bytes appended since the previous rolling boundary scan."""

    try:
        size = path.stat().st_size
        if size < offset:
            raise RuntimeError("trainer log rotated while awaiting r192 boundary")
        with path.open("rb") as stream:
            stream.seek(offset)
            text = stream.read().decode("utf-8", errors="replace")
        return text, size
    except OSError as exc:
        raise RuntimeError(f"cannot read trainer boundary log: {path}") from exc


def _read_new_log(path: Path, offset: int) -> str:
    """Compatibility wrapper for small unit tests and one-shot readers."""

    return _read_log_delta(path, offset)[0]


def _boundary_commit(run_dir: Path, expected_iteration: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    loop_path = run_dir / "loop_state.json"
    commit_path = run_dir / "commits" / f"iter_{expected_iteration:05d}.json"
    loop = _read_json(loop_path)
    if (
        int(loop.get("last_completed_iteration") or -1) != expected_iteration
        or int(loop.get("next_iteration") or -1) != expected_iteration + 1
        or not commit_path.is_file()
    ):
        raise RuntimeError("requested r192 boundary is not an immutable committed transition")
    commit = _read_json(commit_path)
    if (
        int(commit.get("last_completed_iteration") or -1) != expected_iteration
        or int(commit.get("next_iteration") or -1) != expected_iteration + 1
    ):
        raise RuntimeError("r192 boundary commit does not advance exactly once")
    rows = [
        dict(row)
        for row in (commit.get("history") or [])
        if isinstance(row, dict) and int(row.get("iteration") or -1) == expected_iteration
    ]
    if len(rows) != 1 or rows[0].get("completed") is not True:
        raise RuntimeError("r192 boundary commit lacks one completed iteration row")
    return commit_path, commit, loop


def _pause_started(
    text: str,
    *,
    expected_iteration: int,
    pause_seconds: float,
) -> bool:
    seconds = re.escape(f"{pause_seconds:.1f}")
    start = re.compile(
        rf"GATE_BOUNDARY_HARD_PAUSE iteration={expected_iteration} seconds={seconds} .*next_collection_blocked=true"
    )
    complete = re.compile(
        rf"GATE_BOUNDARY_HARD_PAUSE_COMPLETE iteration={expected_iteration} .*next_collection_blocked=false"
    )
    return bool(start.search(text)) and not bool(complete.search(text))


def _pause_completed(text: str, *, expected_iteration: int) -> bool:
    complete = re.compile(
        rf"GATE_BOUNDARY_HARD_PAUSE_COMPLETE iteration={expected_iteration} .*next_collection_blocked=false"
    )
    return bool(complete.search(text))


def _wait_for_visible_gate_pause(
    *,
    run_dir: Path,
    log_path: Path,
    expected_iteration: int,
    pause_seconds: float,
    poll_seconds: float,
    minimum_remaining_seconds: float,
    wait_timeout_seconds: float,
) -> BoundaryObservation:
    """Wait only when armed before the boundary; reject a late generic poll."""

    initial_loop = _read_json(run_dir / "loop_state.json")
    initial_completed = int(initial_loop.get("last_completed_iteration") or -1)
    if initial_completed >= expected_iteration:
        raise RuntimeError(
            "r192 boundary controller must be armed before its target boundary; "
            "a late poll cannot prove the remaining 30 second pause"
        )
    if not log_path.is_file():
        raise RuntimeError(f"r175 trainer log is missing: {log_path}")
    log_scan_offset = log_path.stat().st_size
    log_tail = ""
    deadline = time.monotonic() + wait_timeout_seconds
    # The initial offset proves the target marker was not present when the
    # watcher armed.  This timestamp is replaced after *every* negative log
    # read, including the days before completed_iteration reaches five.
    last_absent_marker_monotonic = time.monotonic()
    while time.monotonic() < deadline:
        loop = _read_json(run_dir / "loop_state.json")
        completed = int(loop.get("last_completed_iteration") or -1)
        if completed > expected_iteration:
            raise RuntimeError("r192 target boundary already passed")
        log_delta, log_scan_offset = _read_log_delta(log_path, log_scan_offset)
        # Keep only a bounded overlap so a marker split across two writes is
        # still recognized without rereading days of appended trainer output.
        log_tail = (log_tail + log_delta)[-8192:]
        if _pause_completed(log_tail, expected_iteration=expected_iteration):
            raise RuntimeError("r192 gate pause completed before it could be verified")
        if _pause_started(
            log_tail,
            expected_iteration=expected_iteration,
            pause_seconds=pause_seconds,
        ):
            if completed != expected_iteration:
                raise RuntimeError("r192 pause marker is not paired with its target commit")
            commit_path, commit, _ = _boundary_commit(run_dir, expected_iteration)
            # The marker appeared after the previous negative read.  Its
            # actual start might be immediately after that read, so use that
            # *earliest* possible deadline, not the log mtime.
            safe_deadline = last_absent_marker_monotonic + pause_seconds
            if time.monotonic() > safe_deadline - minimum_remaining_seconds:
                raise RuntimeError("r192 observed gate pause too late to restart safely")
            return BoundaryObservation(
                commit_path=commit_path,
                commit=commit,
                commit_sha256=_sha256(commit_path),
                safe_pause_deadline_monotonic=safe_deadline,
            )
        last_absent_marker_monotonic = time.monotonic()
        if completed == expected_iteration:
            # Commit existence is required before a matching future marker can
            # authorize the stop, even while the watcher is still waiting.
            _boundary_commit(run_dir, expected_iteration)
        time.sleep(max(0.05, poll_seconds))
    raise TimeoutError("timed out waiting for the r192 gate-boundary pause")


def _validate_candidate_collection_plan(
    *,
    plan_path: Path,
    pin_receipt_path: Path,
    expected_iteration: int,
    gate_id: str,
) -> dict[str, Any]:
    """Prove iteration 6 has the r192 split before its receipt is published."""

    plan = _read_json(plan_path)
    groups = plan.get("group_games_per_iteration")
    per_opponent = plan.get("per_opponent")
    minimums = plan.get("minimum_games_by_opponent")
    expected_groups = {
        "self_play": 1024,
        "strong_public_practice": STRONG_PUBLIC_PRACTICE_GAMES,
        "diverse_public": DIVERSE_PUBLIC_GAMES,
    }
    if (
        plan.get("schema") != "poke_bot.strong_public_practice_plan/v1"
        or int(plan.get("iteration") or -1) != expected_iteration
        or plan.get("active_gate_id") != gate_id
        or groups != expected_groups
        or int(plan.get("games") or -1) != STRONG_PUBLIC_PRACTICE_GAMES
        or not isinstance(per_opponent, dict)
        or not isinstance(minimums, dict)
    ):
        raise CandidateDispatchMayHaveStarted(
            "candidate collection plan is not the exact r192 iteration-6 split"
        )
    marnie = per_opponent.get(OPPONENT_ID)
    if (
        not isinstance(marnie, dict)
        or int(marnie.get("games") or -1) < STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
        or int(marnie.get("minimum_games") or -1)
        != STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
        or int(minimums.get(OPPONENT_ID) or -1)
        != STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
    ):
        raise CandidateDispatchMayHaveStarted(
            "candidate collection plan does not schedule H10 Marnie >=1024"
        )
    if not pin_receipt_path.is_file():
        raise CandidatePlanPending(
            "candidate collection plan exists but its cleaned pin receipt is missing"
        )
    pin_receipt = _read_json(pin_receipt_path)
    pins = pin_receipt.get("pins")
    if (
        pin_receipt.get("schema")
        != "poke_bot.owner_public_mix_pin_floor_receipt/v1"
        or not isinstance(pins, list)
        or any(
            isinstance(row, dict) and row.get("package_id") == OPPONENT_ID
            for row in pins
        )
    ):
        raise CandidateDispatchMayHaveStarted(
            "candidate plan proves H10 Marnie was incorrectly handled as a legacy sidecar pin"
        )
    # The producer receipt deliberately contains scheduling facts only (not
    # baseline checkpoint metadata): package_id, floor_games_per_set,
    # scheduled_games, converted_from_diverse, and met.  The retained Crustle
    # floor is a real diverse-public conversion, so its count must be a
    # positive integer rather than a truthy placeholder.
    crustle = [dict(row) for row in pins if isinstance(row, dict)]
    if (
        len(crustle) != 1
        or crustle[0].get("package_id")
        != "specialist-crustle-final-format-h10-7efd8d4113e7"
        or isinstance(crustle[0].get("floor_games_per_set"), bool)
        or not isinstance(crustle[0].get("floor_games_per_set"), int)
        or crustle[0]["floor_games_per_set"] != 512
        or isinstance(crustle[0].get("scheduled_games"), bool)
        or not isinstance(crustle[0].get("scheduled_games"), int)
        or crustle[0]["scheduled_games"] < 512
        or isinstance(crustle[0].get("converted_from_diverse"), bool)
        or not isinstance(crustle[0].get("converted_from_diverse"), int)
        or crustle[0]["converted_from_diverse"] <= 0
        or crustle[0]["converted_from_diverse"] > crustle[0]["scheduled_games"]
        or crustle[0].get("met") is not True
    ):
        raise CandidateDispatchMayHaveStarted(
            "candidate cleaned pin receipt does not prove retained Crustle-512 scheduling"
        )
    return plan


def _wait_for_candidate_collection_plan(
    *,
    run_dir: Path,
    expected_iteration: int,
    gate_id: str,
    timeout_seconds: float,
) -> tuple[Path, Path, dict[str, Any]]:
    """Wait for plan materialization; never rollback once it may dispatch work."""

    plan_path = run_dir / "collection_plans" / f"iter_{expected_iteration:05d}.json"
    pin_receipt_path = (
        run_dir
        / "collection_plans"
        / f"iter_{expected_iteration:05d}.owner_public_mix_pin_floors.json"
    )
    deadline = time.monotonic() + timeout_seconds
    plan_seen = False
    while time.monotonic() < deadline:
        if plan_path.is_file():
            plan_seen = True
            try:
                return (
                    plan_path,
                    pin_receipt_path,
                    _validate_candidate_collection_plan(
                        plan_path=plan_path,
                        pin_receipt_path=pin_receipt_path,
                        expected_iteration=expected_iteration,
                        gate_id=gate_id,
                    ),
                )
            except CandidatePlanPending:
                # The plan can race the sibling receipt writer.  Keep waiting
                # for that proof, but do not permit a later rollback: merely
                # materializing a plan may allow worker dispatch.
                pass
            except CandidateDispatchMayHaveStarted:
                raise
            except RuntimeError as exc:
                # A plan exists, so conservatively treat dispatch as possible.
                raise CandidateDispatchMayHaveStarted(str(exc)) from exc
        time.sleep(0.10)
    if plan_seen:
        raise CandidateDispatchMayHaveStarted(
            "candidate iteration-6 collection plan exists but its cleaned pin receipt did not materialize"
        )
    raise RuntimeError("candidate iteration-6 collection plan did not materialize")


def _next_iteration_artifacts(
    *,
    run_dir: Path,
    next_iteration: int,
) -> list[Path]:
    """Return every receipt/shard name proving the next iteration may exist."""

    if not run_dir.is_dir():
        raise RuntimeError(f"r192 run directory is missing: {run_dir}")
    token = f"iter_{next_iteration:05d}"
    # The bounded run directory is the immutable receipts root.  Scan names,
    # rather than assuming only today's plan/commit layout, so a newly added
    # iter-6 artifact cannot be hidden in a sibling receipt directory.
    found: list[Path] = []
    for root, directories, files in os.walk(run_dir):
        for name in (*directories, *files):
            if token in name:
                found.append(Path(root) / name)
    runtime_state = run_dir / "iteration_runtime.json"
    if runtime_state.is_file():
        try:
            runtime = _read_json(runtime_state)
            runtime_iteration = int(runtime.get("iteration") or -1)
        except RuntimeError:
            # A malformed mutable runtime state means the controller can no
            # longer prove that candidate dispatch has not begun.
            found.append(runtime_state)
        else:
            if runtime_iteration == next_iteration:
                found.append(runtime_state)
    return found


def _assert_no_next_iteration_artifacts(
    *,
    run_dir: Path,
    next_iteration: int,
) -> None:
    """Reject a stale or already-started candidate before touching systemd."""

    found = _next_iteration_artifacts(
        run_dir=run_dir,
        next_iteration=next_iteration,
    )
    if found:
        raise RuntimeError(
            "next iteration artifact already exists; the verified pause is no longer clean: "
            + str(found[0])
        )


def _assert_boundary_commit_still_stable(
    *,
    run_dir: Path,
    observation: BoundaryObservation,
    expected_iteration: int,
) -> None:
    """Reprove immutable iter-5 state immediately before stopping the parent."""

    commit_path, _, _ = _boundary_commit(run_dir, expected_iteration)
    if commit_path.resolve() != observation.commit_path.resolve():
        raise RuntimeError("r192 boundary commit path drifted during the hard pause")
    if _sha256(commit_path) != observation.commit_sha256:
        raise RuntimeError("r192 boundary commit bytes drifted during the hard pause")


def _systemctl(*args: str, timeout: float = 90.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _service_value(unit: str, key: str) -> str:
    result = _systemctl("show", unit, "-p", key, "--value", timeout=15.0)
    if result.returncode:
        raise RuntimeError(
            f"cannot read systemd {key} for {unit}: {result.stdout.strip()}"
        )
    return result.stdout.strip()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inactive_boundary_proof(
    *,
    receipt_path: Path,
    run_dir: Path,
    service: str,
    expected_iteration: int,
) -> dict[str, Any]:
    """Verify the alternative r192 authority: a stopped, committed boundary."""

    receipt_path = receipt_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    receipt = _read_json(receipt_path)
    next_iteration = expected_iteration + 1
    if (
        receipt.get("schema") != "poke_bot.committed_iteration_pause/v1"
        or receipt.get("status") not in {"paused", "paused_successor_started"}
        or receipt.get("unit") != service
        or int(receipt.get("completed_iteration") or -1) != expected_iteration
        or int(receipt.get("next_iteration") or -1) != next_iteration
        or receipt.get("uncommitted_next_iteration_started") is not False
        or receipt.get("recovery_required") is not False
        or receipt.get("service_active_state") not in {"inactive", "failed"}
    ):
        raise RuntimeError("r192 inactive-boundary receipt is not an exact clean stop")
    if _service_value(service, "ActiveState") not in {"inactive", "failed"}:
        raise RuntimeError("r192 inactive-boundary service is still active")
    if int(_service_value(service, "MainPID") or 0) != 0:
        raise RuntimeError("r192 inactive-boundary service still has a MainPID")
    commit_path, commit, loop = _boundary_commit(run_dir, expected_iteration)
    if loop != commit:
        raise RuntimeError("r192 inactive boundary loop state differs from commit")
    if (
        Path(str(receipt.get("commit") or "")).expanduser().resolve()
        != commit_path.resolve()
        or str(receipt.get("commit_digest") or "") != _sha256(commit_path)
    ):
        raise RuntimeError("r192 inactive-boundary commit identity changed")
    learner = dict(commit.get("learner") or {})
    checkpoint = Path(str(learner.get("path") or "")).expanduser().resolve()
    digest = str(learner.get("digest") or "")
    if (
        not checkpoint.is_file()
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        or _sha256(checkpoint) != digest
        or Path(str(receipt.get("checkpoint") or "")).expanduser().resolve()
        != checkpoint
        or str(receipt.get("checkpoint_digest") or "") != digest
    ):
        raise RuntimeError("r192 inactive-boundary learner identity changed")
    _assert_no_next_iteration_artifacts(
        run_dir=run_dir,
        next_iteration=next_iteration,
    )
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "commit": str(commit_path.resolve()),
        "commit_sha256": _sha256(commit_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "completed_iteration": expected_iteration,
        "next_iteration": next_iteration,
    }


def _deployment_input(plan: dict[str, Any], label: str) -> Path:
    matches = [
        Path(str(row["path"])).expanduser().resolve()
        for row in plan.get("deployment_inputs") or []
        if isinstance(row, dict) and row.get("label") == label
    ]
    if len(matches) != 1:
        raise RuntimeError(f"r192 plan lacks exactly one deployment input {label}")
    return matches[0]


def _render_dropin(
    *,
    template: Path,
    runtime_root: Path,
    runtime_registry: Path,
    pin_floors: Path,
    training_arm_file: Path,
    launch_active_specialist: Path,
) -> str:
    text = template.read_text(encoding="utf-8")
    replacements = {
        "@RUNTIME_ROOT@": (str(runtime_root), 1),
        "@RUNTIME_REGISTRY@": (str(runtime_registry), 4),
        "@PIN_FLOORS@": (str(pin_floors), 2),
        "@TRAINING_ARM_FILE@": (str(training_arm_file.expanduser().resolve()), 1),
        "@PYTHON@": (sys.executable, 2),
        "@LAUNCH_ACTIVE_SPECIALIST@": (str(launch_active_specialist), 2),
    }
    for token, (value, expected_count) in replacements.items():
        if text.count(token) != expected_count:
            raise RuntimeError(f"r192 drop-in template token count changed: {token}")
        text = text.replace(token, value)
    if "@" in text:
        raise RuntimeError("r192 rendered drop-in retains an unresolved token")
    return text


def apply(
    *,
    plan: dict[str, Any],
    inactive_boundary_receipt: Path,
    run_dir: Path,
    service: str,
    expected_iteration: int,
    dropin_target: Path,
    activation_receipt: Path,
    training_arm_file: Path,
    startup_timeout_seconds: float,
    candidate_plan_timeout_seconds: float,
) -> dict[str, Any]:
    """Activate only from the receipt-proven already-inactive alternative."""

    boundary = _inactive_boundary_proof(
        receipt_path=inactive_boundary_receipt,
        run_dir=run_dir,
        service=service,
        expected_iteration=expected_iteration,
    )
    staged = dict(plan.get("staged_artifacts") or {})
    runtime_registry = Path(str(dict(staged.get("runtime_registry") or {}).get("path") or "")).resolve()
    pin_floors = Path(str(dict(staged.get("public_mix_pin_floors") or {}).get("path") or "")).resolve()
    runtime_payload = _read_json(runtime_registry)
    runtime_root = Path(str(runtime_payload.get("runtime_root") or "")).expanduser().resolve()
    if not runtime_root.is_dir():
        raise RuntimeError("r192 runtime registry has no readable runtime_root")
    launch = _deployment_input(plan, "launch_active_specialist")
    template = Path(str(dict(plan["activation_artifacts"]["dropin_template"])["path"])).resolve()
    rendered = _render_dropin(
        template=template,
        runtime_root=runtime_root,
        runtime_registry=runtime_registry,
        pin_floors=pin_floors,
        training_arm_file=training_arm_file,
        launch_active_specialist=launch,
    )
    check_env = os.environ.copy()
    inherited_pythonpath = check_env.get("PYTHONPATH", "")
    check_env["PYTHONPATH"] = str(runtime_root) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )
    check = subprocess.run(
        [sys.executable, "-u", str(launch), "--registry", str(runtime_registry), "--check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=check_env,
        timeout=120.0,
    )
    if check.returncode:
        raise RuntimeError(f"r192 candidate launcher preflight failed: {check.stdout.strip()}")
    dropin_target = dropin_target.expanduser().resolve()
    if dropin_target.exists():
        if dropin_target.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("r192 drop-in target already exists with different bytes")
    else:
        _atomic_text(dropin_target, rendered)
    reload_result = _systemctl("daemon-reload")
    if reload_result.returncode:
        raise RuntimeError(f"r192 daemon-reload failed: {reload_result.stdout.strip()}")
    reset = _systemctl("reset-failed", service)
    if reset.returncode:
        raise RuntimeError(f"r192 reset-failed failed: {reset.stdout.strip()}")
    started = _systemctl("start", "--no-block", service)
    if started.returncode:
        raise RuntimeError(f"r192 managed start failed: {started.stdout.strip()}")
    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if _service_value(service, "ActiveState") == "active" and int(
            _service_value(service, "MainPID") or 0
        ) > 0:
            break
        time.sleep(0.10)
    else:
        raise RuntimeError("r192 candidate service did not become active")
    next_iteration = expected_iteration + 1
    try:
        plan_path, pin_receipt_path, collection_plan = _wait_for_candidate_collection_plan(
            run_dir=run_dir,
            expected_iteration=next_iteration,
            gate_id=str(plan["gate_id"]),
            timeout_seconds=candidate_plan_timeout_seconds,
        )
    except Exception:
        _systemctl("stop", service)
        raise
    payload = {
        "schema": "poke_bot.alakazam_marnie_splusplus_r192_activation/v1",
        "status": "activated_from_receipt_proven_inactive_boundary",
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_decision_revision": OWNER_REVISION,
        "service": service,
        "service_main_pid": int(_service_value(service, "MainPID") or 0),
        "boundary": boundary,
        "stage_receipt": plan["stage_receipt"],
        "stage_receipt_sha256": plan["stage_receipt_sha256"],
        "runtime_registry": str(runtime_registry),
        "runtime_registry_sha256": _sha256(runtime_registry),
        "dropin": str(dropin_target),
        "dropin_sha256": _sha256(dropin_target),
        "collection_plan": str(plan_path.resolve()),
        "collection_plan_sha256": _sha256(plan_path),
        "pin_floor_receipt": str(pin_receipt_path.resolve()),
        "pin_floor_receipt_sha256": _sha256(pin_receipt_path),
        "collection_plan_iteration": next_iteration,
        "collection_plan_contract": collection_plan,
    }
    activation_receipt = activation_receipt.expanduser().resolve()
    if activation_receipt.exists():
        if _read_json(activation_receipt) != payload:
            raise RuntimeError("r192 activation receipt already exists with different bytes")
    else:
        _atomic_text(activation_receipt, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

    del plan, _unused
    raise RuntimeError(UNARMED_APPLY_REASON)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument(
        "--owner-contract",
        type=Path,
        default=ROOT / "state/alakazam-marnie-splusplus-opponent-r192.json",
    )
    parser.add_argument(
        "--service",
        default="pokebot-final-format-alakazam-rtp-r175-rl.service",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--expected-after-iteration",
        type=int,
        default=FIRST_GUARANTEED_BOUNDARY_ITERATION,
    )
    parser.add_argument(
        "--dropin-template",
        type=Path,
        default=(
            ROOT
            / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-rl.service.d"
            / "62-marnie-splusplus-r192.conf.in"
        ),
    )
    parser.add_argument(
        "--dropin-target",
        type=Path,
        default=Path(
            "/home/pokebot/.config/systemd/user/"
            "pokebot-final-format-alakazam-rtp-r175-rl.service.d/"
            "62-marnie-splusplus-r192.conf"
        ),
    )
    parser.add_argument(
        "--stop-budget-template",
        type=Path,
        default=(
            ROOT
            / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-rl.service.d"
            / "61-marnie-splusplus-r192-stop-budget.conf.in"
        ),
    )
    parser.add_argument("--stop-budget-target", type=Path)
    parser.add_argument(
        "--activation-receipt",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent/outputs/state/"
            "alakazam-marnie-splusplus-r192-activation.json"
        ),
    )
    parser.add_argument(
        "--inactive-boundary-receipt",
        type=Path,
        help=(
            "Required with --apply; exact committed-iteration pause receipt "
            "proving the managed trainer is already stopped"
        ),
    )
    parser.add_argument("--phase-journal", type=Path)
    parser.add_argument(
        "--launch-active-specialist",
        type=Path,
        default=Path("/home/pokebot/poke-bot-agent/scripts/launch_active_specialist.py"),
    )
    parser.add_argument(
        "--training-arm-file",
        type=Path,
        default=Path("/home/pokebot/poke-bot-agent/outputs/state/TRAINING_ARMED"),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--minimum-remaining-pause-seconds", type=float, default=20.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--boundary-wait-timeout-seconds",
        type=float,
        default=7 * 24 * 60 * 60,
    )
    parser.add_argument("--candidate-plan-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="activate only from --inactive-boundary-receipt",
    )
    args = parser.parse_args()

    plan = preflight(
        stage_path=args.stage_receipt,
        owner_contract_path=args.owner_contract,
    )
    if not args.apply:
        print(json.dumps({"status": "validated_not_activated", **plan}, indent=2, sort_keys=True))
        return 0
    if args.inactive_boundary_receipt is None:
        raise RuntimeError(UNARMED_APPLY_REASON)
    if args.run_dir is None:
        raise RuntimeError("--run-dir is required with --apply")
    payload = apply(
        plan=plan,
        inactive_boundary_receipt=args.inactive_boundary_receipt,
        run_dir=args.run_dir,
        service=args.service,
        expected_iteration=args.expected_after_iteration,
        dropin_target=args.dropin_target,
        activation_receipt=args.activation_receipt,
        training_arm_file=args.training_arm_file,
        startup_timeout_seconds=args.startup_timeout_seconds,
        candidate_plan_timeout_seconds=args.candidate_plan_timeout_seconds,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
