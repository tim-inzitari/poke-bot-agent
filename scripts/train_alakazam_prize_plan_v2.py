#!/usr/bin/env python3
"""Train the isolated, offline Alakazam Prize-plan-v2 action critic.

This trainer is intentionally not a play-loop component.  It joins the sealed
recent-20 40-wide feature view to the portable Prize-plan-v2 *target-only*
set on the exact seven-field complete-action identity, and trains only the
separate ``PrizePlanV2Sidecar`` on recorded chosen actions.  It never imports
policy, runtime, RTP, MCTS, search, or simulator code.

The only learning labels accepted here are the target rows'
``model_target_value`` fields.  ``raw_return_value`` is read solely to verify
the sealed analytic relation ``raw / (1 + gamma**h)``; it is not a training
target, is never clipped, and no terminal return is admitted to this lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These are data/sidecar-only imports.  In particular this program must not
# acquire a policy model, a game engine, or a live play-loop dependency.
from poke_bot.prize_plan_v2_sidecar import (  # noqa: E402
    PRIZE_PLAN_V2_HORIZONS,
    PRIZE_PLAN_V2_OUTPUT_NAMES,
    PRIZE_PLAN_V2_SIDECAR_SCHEMA,
    PrizePlanV2Sidecar,
    PrizePlanV2SidecarConfig,
    PrizePlanV2SidecarError,
    load_prize_plan_v2_checkpoint,
    masked_prize_plan_v2_loss,
    restore_prize_plan_v2_checkpoint,
    save_prize_plan_v2_checkpoint,
)
from poke_bot.recursive_turn_planner.recent20_overlay import (  # noqa: E402
    Recent20OverlayError,
    Recent20RTPDataset,
    canonical_bytes,
    sha256_file,
)


TRAINER_SCHEMA = "poke_bot.alakazam_prize_plan_v2_trainer/v1"
TRAINING_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_v2_training_receipt/v1"
VALIDATION_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_v2_validation_receipt/v1"
H3_SCALE_SUPPORT_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_scale_support/v1"
TRAINING_VIEW_SCHEMA = "poke_bot.alakazam_action_critic_training_view/v1"
TRAINING_VIEW_COMPLETION_SCHEMA = (
    "poke_bot.alakazam_action_critic_training_view_completion/v1"
)
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_target_set_manifest/v2"
TARGET_SET_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_target_set_materialization_receipt/v2"
)
TARGET_DAY_MANIFEST_SCHEMA = "poke_bot.alakazam_prize_plan_target_day_manifest/v2"
TARGET_DAY_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_target_day_materialization_receipt/v2"
)
TARGET_ROW_SCHEMA = "poke_bot.alakazam_prize_plan_target_overlay/v2"
TARGET_VALUE_TRANSFORM_SCHEMA = "poke_bot.alakazam_prize_plan_target_value_transform/v2"
TARGET_VIEW_SCHEMA = "poke_bot.alakazam_prize_plan_v2_target_view/v1"
TARGET_VIEW_COMPLETION_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_target_transfer_completion/v1"
)
TARGET_TRANSFER_PLAN_SCHEMA = "poke_bot.alakazam_prize_plan_v2_target_transfer_plan/v1"
TARGET_TRANSFER_FILE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_target_file_transfer_receipt/v1"
)
PRIZE_AUTHORITY_KEY = "revision_23_prize_plan_v2_h3_actor_canary"
SEMANTIC_OWNER_REVISION = 23
FEATURE_WIDTH = 40
MAX_STAGES = 32
MAX_LEGAL = 64
MAX_TOKENS = 32
MAX_TOKEN_VALUE = 63
MIN_PRODUCTION_DISK_FLOOR_BYTES = 20 * 1024**3
MIN_PRODUCTION_METADATA_RESERVE_BYTES = 4 * 1024**2
EXPECTED_TRAIN_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 6)]
)
EXPECTED_VALIDATION_DAYS = ("2026-08-06", "2026-08-07", "2026-08-08")
EXPECTED_EVALUATION_DAYS = ("2026-08-09", "2026-08-10", "2026-08-11")
EXPECTED_SPLITS = {
    "train": EXPECTED_TRAIN_DAYS,
    "validation": EXPECTED_VALIDATION_DAYS,
    "evaluation": EXPECTED_EVALUATION_DAYS,
}
IDENTITY_FIELDS = (
    "utc_day",
    "source_archive_sha256",
    "source_member",
    "episode_id",
    "acting_seat",
    "env_step",
    "program_identity",
)
TARGET_TRANSFER_ROLES = frozenset(
    {
        "goal_contract",
        "complete_action_overlay_manifest",
        "phi_fit_input_manifest",
        "phi_table",
        "phi_fit_manifest",
        "phi_fit_receipt",
        "phi_fit_artifact",
        "target_value_transform",
        "target_schema",
        "target_shard",
        "target_day_manifest",
        "target_day_receipt",
        "target_set_manifest",
        "target_set_receipt",
    }
)
EPSILON = 1.0e-8


class PrizePlanV2TrainingError(RuntimeError):
    """A sealed input, target, action, or sidecar invariant failed."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrizePlanV2TrainingError(f"{label} must be a JSON object")
    return value


def _regular_file(value: Path | str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PrizePlanV2TrainingError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _regular_directory(value: Path | str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise PrizePlanV2TrainingError(f"{label} must be a real non-symlink directory: {path}")
    return path


def _read_json(path: Path | str, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrizePlanV2TrainingError(f"invalid {label} JSON: {source}") from exc
    if not isinstance(value, dict):
        raise PrizePlanV2TrainingError(f"{label} must contain a JSON object")
    return value


def _sha(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    raw = text.removeprefix("sha256:")
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise PrizePlanV2TrainingError(f"{label} must be a full lowercase SHA-256")
    return "sha256:" + raw


def _sha_match(path: Path | str, expected: Any, *, label: str) -> str:
    actual = sha256_file(_regular_file(path, label=label))
    if actual != _sha(expected, label=f"{label} SHA-256"):
        raise PrizePlanV2TrainingError(f"{label} SHA-256 mismatch")
    return actual


def _exact_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrizePlanV2TrainingError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise PrizePlanV2TrainingError(f"{label} must be at least {minimum}")
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrizePlanV2TrainingError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PrizePlanV2TrainingError(f"{label} must be finite")
    return result


def _relative_member(root: Path, value: Any, *, label: str) -> Path:
    text = str(value or "")
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or any(
        item in {"", "."} for item in pure.parts
    ):
        raise PrizePlanV2TrainingError(f"{label} must be a safe portable relative path")
    path = root.joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise PrizePlanV2TrainingError(f"{label} escapes its sealed root") from exc
    return _regular_file(path, label=label)


def _relative_directory(root: Path, value: Any, *, label: str, allow_dot: bool = False) -> Path:
    """Resolve a portable directory declaration without allowing an escape."""

    text = str(value or "")
    if allow_dot and text == ".":
        return _regular_directory(root, label=label)
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or any(
        item in {"", "."} for item in pure.parts
    ):
        raise PrizePlanV2TrainingError(f"{label} must be a safe portable relative directory")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise PrizePlanV2TrainingError(f"{label} escapes its sealed root") from exc
    return _regular_directory(path, label=label)


def _json_sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    """Atomically replace an explicitly owned trainer-local artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{random.randrange(1 << 32):08x}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    body = canonical_bytes(dict(value))
    _atomic_write_bytes(path, body)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _validate_contract_authority(contract: Mapping[str, Any]) -> int:
    """Accept a newer wrapper only when the embedded r23 authority is exact."""

    current = _exact_int(contract.get("goal_revision"), label="goal contract revision")
    if current < SEMANTIC_OWNER_REVISION:
        raise PrizePlanV2TrainingError("goal contract predates Prize-plan-v2 authority")
    authority = _mapping(contract.get(PRIZE_AUTHORITY_KEY), label="r23 Prize-plan authority")
    if authority.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION:
        raise PrizePlanV2TrainingError("r23 Prize-plan authority owner revision drifted")
    sidecar = _mapping(authority.get("sidecar_strategy"), label="r23 sidecar strategy")
    target = _mapping(authority.get("public_prize_plan_target"), label="r23 public target")
    actor = _mapping(authority.get("actor_advantage"), label="r23 actor advantage")
    activation = _mapping(authority.get("activation_and_failure_boundary"), label="r23 activation")
    if (
        sidecar.get("default_safe_implementation") != "separately_versioned_prize_plan_v2_sidecar"
        or sidecar.get("sidecar_schema") != PRIZE_PLAN_V2_SIDECAR_SCHEMA
        or sidecar.get("plan_horizons_to_train_and_receipt") != list(PRIZE_PLAN_V2_HORIZONS)
        or sidecar.get("training_host") != "bert"
        or target.get("row_schema") != TARGET_ROW_SCHEMA
        or target.get("manifest_schema") != TARGET_SET_MANIFEST_SCHEMA
        or target.get("row_join_identity") != list(IDENTITY_FIELDS)
        or target.get("terminal_observed_z_is_direct_plan_reward_or_actor_term") is not False
        or target.get("hidden_prize_identity_or_other_hidden_information_allowed") is not False
        or actor.get("selected_nonzero_cumulative_prize_horizon") != 3
        or actor.get("simultaneous_or_additive_H1_H3_H6_H12_actor_terms_allowed") is not False
        or actor.get("enabled_formula")
        != "(z-V_existing(s))+0.025*m3*c3*(Q_plan_3(s,a)-V_plan_3(s))"
        or activation.get("current_activation_allowed") is not False
    ):
        raise PrizePlanV2TrainingError("embedded r23 Prize-plan-v2 semantics drifted")
    return current


def load_bound_contract(
    path: Path | str, *, expected_sha256: str, production: bool = True
) -> tuple[Path, str, int]:
    contract_path = _regular_file(path, label="goal contract")
    actual = sha256_file(contract_path)
    if production and not expected_sha256:
        raise PrizePlanV2TrainingError("production training requires --contract-sha256")
    if expected_sha256 and actual != _sha(expected_sha256, label="contract SHA-256"):
        raise PrizePlanV2TrainingError("goal contract SHA-256 mismatch")
    document = _read_json(contract_path, label="goal contract")
    return contract_path, actual, _validate_contract_authority(document) if production else _exact_int(
        document.get("goal_revision", SEMANTIC_OWNER_REVISION), label="test contract revision"
    )


def _training_view_root(pointer: Path) -> Path:
    if pointer.parent.name != "training-view" or pointer.parent.parent.name != "transfer":
        raise PrizePlanV2TrainingError("training-view pointer must live under transfer/training-view")
    return _regular_directory(pointer.parent.parent.parent, label="training-view root")


def _content_addressed_receipt(path: Path, *, label: str) -> str:
    actual = sha256_file(_regular_file(path, label=label))
    if not path.name.startswith("sha256-" + actual.removeprefix("sha256:")):
        raise PrizePlanV2TrainingError(f"{label} filename is not content-addressed")
    return actual


def resolve_training_view(
    *, pointer: Path | str, expected_sha256: str, production: bool = True
) -> dict[str, Any]:
    """Resolve the local portable base/overlay view and its completion chain."""

    pointer_path = _regular_file(pointer, label="training-view pointer")
    pointer_sha = _content_addressed_receipt(pointer_path, label="training-view pointer")
    if production and not expected_sha256:
        raise PrizePlanV2TrainingError("production training requires --training-view-sha256")
    if expected_sha256 and pointer_sha != _sha(expected_sha256, label="training-view SHA-256"):
        raise PrizePlanV2TrainingError("training-view pointer SHA-256 mismatch")
    view = _read_json(pointer_path, label="training-view pointer")
    if (
        view.get("schema") != TRAINING_VIEW_SCHEMA
        or view.get("canonical_manifest_remains_byte_identical") is not True
        or view.get("base_completion_path_override_required") is not True
        or view.get("runtime_or_training_activation_authority") is not False
    ):
        raise PrizePlanV2TrainingError("training-view portability boundary drifted")
    root = _training_view_root(pointer_path)
    overlay = _mapping(view.get("canonical_overlay_manifest"), label="training-view overlay")
    base = _mapping(view.get("canonical_base_completion"), label="training-view base completion")
    overlay_path = _relative_member(root, overlay.get("relative_path"), label="training-view overlay")
    completion_path = _relative_member(root, base.get("relative_path"), label="training-view base completion")
    # ``base_pack_root_relative`` names a directory, so resolve it separately
    # rather than treating it as a member file.
    base_relative = PurePosixPath(str(view.get("base_pack_root_relative") or ""))
    if not str(base_relative) or base_relative.is_absolute() or ".." in base_relative.parts:
        raise PrizePlanV2TrainingError("training-view base-pack root is not portable")
    base_root = root.joinpath(*base_relative.parts)
    if base_root.is_symlink() or not base_root.is_dir():
        raise PrizePlanV2TrainingError("training-view base-pack root is not a real directory")
    overlay_root_relative = PurePosixPath(str(view.get("overlay_root_relative") or ""))
    if not str(overlay_root_relative) or overlay_root_relative.is_absolute() or ".." in overlay_root_relative.parts:
        raise PrizePlanV2TrainingError("training-view overlay root is not portable")
    overlay_root = root.joinpath(*overlay_root_relative.parts)
    if overlay_root.is_symlink() or not overlay_root.is_dir():
        raise PrizePlanV2TrainingError("training-view overlay root is not a real directory")
    overlay_sha = _sha_match(overlay_path, overlay.get("sha256"), label="training-view overlay")
    completion_sha = _sha_match(completion_path, base.get("sha256"), label="training-view base completion")
    plan_sha = _sha(view.get("transfer_plan_sha256"), label="training-view transfer plan SHA-256")
    completion_root = root / "transfer" / "completion"
    if completion_root.is_symlink() or not completion_root.is_dir():
        raise PrizePlanV2TrainingError("training-view completion directory is missing")
    matches: list[tuple[Path, str, Mapping[str, Any]]] = []
    for candidate in sorted(completion_root.glob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        digest = _content_addressed_receipt(candidate, label="training-view completion receipt")
        receipt = _read_json(candidate, label="training-view completion receipt")
        if (
            receipt.get("schema") == TRAINING_VIEW_COMPLETION_SCHEMA
            and receipt.get("training_view_sha256") == pointer_sha
            and receipt.get("training_view_path") == str(pointer_path.relative_to(root))
        ):
            matches.append((candidate, digest, receipt))
    if len(matches) != 1:
        raise PrizePlanV2TrainingError("training-view requires exactly one matching completion receipt")
    receipt_path, receipt_sha, receipt = matches[0]
    source = _mapping(receipt.get("source"), label="training-view completion source")
    if (
        receipt.get("transfer_plan_sha256") != plan_sha
        or source.get("base_completion_sha256") != completion_sha
        or source.get("overlay_manifest_sha256") != overlay_sha
        or receipt.get("all_source_destination_sha256_size_verified") is not True
        or receipt.get("private_partials_not_training_eligible") is not True
        or receipt.get("canonical_overlay_manifest_rewritten") is not False
        or receipt.get("runtime_or_trainer_activation_performed") is not False
    ):
        raise PrizePlanV2TrainingError("training-view completion receipt binding drifted")
    return {
        "pointer_path": str(pointer_path),
        "pointer_sha256": pointer_sha,
        "completion_path": str(receipt_path),
        "completion_sha256": receipt_sha,
        "overlay_manifest": str(overlay_path),
        "overlay_manifest_sha256": overlay_sha,
        "base_pack_root": str(base_root),
        "base_completion_path": str(completion_path),
        "base_completion_sha256": completion_sha,
    }


def _target_view_root(pointer: Path) -> Path:
    if pointer.parent.name != "target-view" or pointer.parent.parent.name != "transfer":
        raise PrizePlanV2TrainingError("v2 target-view pointer must live under transfer/target-view")
    return _regular_directory(pointer.parent.parent.parent, label="v2 target-view root")


def _safe_relative_text(value: Any, *, label: str) -> str:
    text = str(value or "")
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or any(
        item in {"", "."} for item in pure.parts
    ):
        raise PrizePlanV2TrainingError(f"{label} must be a safe portable relative path")
    return "/".join(pure.parts)


def _validate_transfer_target_days(value: Any, *, label: str) -> list[dict[str, Any]]:
    """Validate the compact 20-day transfer graph without opening raw replay."""

    if not isinstance(value, list) or len(value) != 20:
        raise PrizePlanV2TrainingError(f"{label} must contain the exact 20 target days")
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        item = dict(_mapping(raw, label=f"{label} item"))
        day = item.get("utc_day")
        split = item.get("split")
        if not isinstance(day, str) or day in seen or split != _expected_day_split(day):
            raise PrizePlanV2TrainingError(f"{label} day/split inventory drifted")
        seen.add(day)
        for field in (
            "day_artifact_root_relative",
            "day_manifest_relative",
            "day_receipt_relative",
            "target_schema_relative",
            "target_shard_relative",
        ):
            _safe_relative_text(item.get(field), label=f"{label} {day} {field}")
        for field in (
            "day_manifest_sha256",
            "day_receipt_sha256",
            "target_schema_sha256",
            "target_shard_sha256",
            "raw_episode_zip_sha256",
            "complete_action_overlay_sha256",
        ):
            _sha(item.get(field), label=f"{label} {day} {field}")
        _exact_int(item.get("target_shard_size_bytes"), label=f"{label} {day} target size", minimum=1)
        _exact_int(item.get("target_row_count"), label=f"{label} {day} target rows", minimum=1)
        _exact_int(item.get("raw_episode_zip_size_bytes"), label=f"{label} {day} raw provenance size", minimum=1)
        observed.append(item)
    expected = tuple(sum((list(days) for days in EXPECTED_SPLITS.values()), []))
    if tuple(item["utc_day"] for item in observed) != expected:
        raise PrizePlanV2TrainingError(f"{label} day order is not the sealed recent-20 window")
    return observed


def _validate_target_transfer_plan(
    plan: Mapping[str, Any],
    *,
    root: Path,
    expected_sha256: str,
    contract_sha256: str,
    contract_goal_revision: int,
    overlay_manifest_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any], list[dict[str, Any]]]:
    """Independently validate the target transfer plan's immutable object graph."""

    if (
        plan.get("schema") != TARGET_TRANSFER_PLAN_SCHEMA
        or plan.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
        or plan.get("required_authority") != PRIZE_AUTHORITY_KEY
        or plan.get("parallel_lanes_exact") != 4
        or plan.get("target_only") is not True
        or plan.get("raw_episode_zip_objects_transferred") is not False
        or plan.get("feature_pack_objects_transferred") is not False
        or plan.get("complete_action_overlay_payload_objects_transferred") is not False
        or plan.get("canonical_target_set_manifest_remains_byte_identical") is not True
        or plan.get("bert_side_c3_or_advantage_scaling_produced_or_transferred") is not False
    ):
        raise PrizePlanV2TrainingError("v2 target transfer plan authority/information boundary drifted")
    destination = Path(str(plan.get("destination_root") or "")).expanduser().resolve(strict=False)
    if destination != root:
        raise PrizePlanV2TrainingError("v2 target transfer plan destination root does not match pointer root")
    source = _mapping(plan.get("source"), label="v2 target transfer plan source")
    local_contract = _mapping(plan.get("local_contract"), label="v2 target transfer local contract")
    if (
        source.get("read_only") is not True
        or source.get("goal_contract_sha256") != contract_sha256
        or source.get("goal_contract_goal_revision") != contract_goal_revision
        or source.get("embedded_semantic_owner_goal_revision") != SEMANTIC_OWNER_REVISION
        or source.get("complete_action_overlay_manifest_sha256") != overlay_manifest_sha256
        or local_contract.get("sha256") != contract_sha256
        or local_contract.get("goal_revision") != contract_goal_revision
        or local_contract.get("embedded_semantic_owner_goal_revision") != SEMANTIC_OWNER_REVISION
    ):
        raise PrizePlanV2TrainingError("v2 target transfer plan contract/overlay binding drifted")
    for field in (
        "target_set_manifest_relative",
        "target_set_receipt_relative",
        "goal_contract_relative",
        "complete_action_overlay_manifest_relative",
    ):
        _safe_relative_text(source.get(field), label=f"v2 target transfer source {field}")
    for field in ("target_set_manifest_sha256", "target_set_receipt_sha256"):
        _sha(source.get(field), label=f"v2 target transfer source {field}")
    phi = _mapping(source.get("phi_fit"), label="v2 target transfer Phi binding")
    transform = _mapping(source.get("target_value_transform"), label="v2 target transfer transform binding")
    if (
        phi.get("fit_manifest_sha256") is None
        or phi.get("fit_receipt_sha256") is None
        or phi.get("frozen_phi_table_sha256") is None
        or transform.get("formula") != "model_target_value=raw_return_value/(1+gamma**h)"
        or transform.get("gamma") != 1.0
        or transform.get("data_dependent_train_fit") is not False
    ):
        raise PrizePlanV2TrainingError("v2 target transfer Phi/transform binding drifted")
    for field in ("fit_manifest_sha256", "fit_receipt_sha256", "frozen_phi_table_sha256"):
        _sha(phi.get(field), label=f"v2 target transfer Phi {field}")
    _sha(transform.get("sha256"), label="v2 target transfer transform SHA-256")
    _safe_relative_text(transform.get("relative_path"), label="v2 target transfer transform path")
    raw_identities = source.get("raw_episode_zip_identities")
    if not isinstance(raw_identities, list) or len(raw_identities) != 20:
        raise PrizePlanV2TrainingError("v2 target transfer raw provenance inventory drifted")
    if tuple(item.get("utc_day") for item in raw_identities) != tuple(
        sum((list(days) for days in EXPECTED_SPLITS.values()), [])
    ):
        raise PrizePlanV2TrainingError("v2 target transfer raw provenance days drifted")
    for item in raw_identities:
        _sha(_mapping(item, label="v2 target raw provenance").get("sha256"), label="v2 target raw provenance SHA-256")
        _exact_int(_mapping(item, label="v2 target raw provenance").get("size_bytes"), label="v2 target raw provenance size", minimum=1)
    target_days = _validate_transfer_target_days(plan.get("target_days"), label="v2 target transfer target days")
    entries_raw = plan.get("entries")
    lanes_raw = plan.get("lanes")
    if not isinstance(entries_raw, list) or not isinstance(lanes_raw, list) or len(lanes_raw) != 4:
        raise PrizePlanV2TrainingError("v2 target transfer plan entries/lanes are malformed")
    lane_counts = {lane: 0 for lane in range(4)}
    lane_sizes = {lane: 0 for lane in range(4)}
    entries: dict[str, Mapping[str, Any]] = {}
    source_paths: set[str] = set()
    for raw in entries_raw:
        entry = _mapping(raw, label="v2 target transfer entry")
        destination_relative = _safe_relative_text(entry.get("destination_relative"), label="v2 transfer destination")
        source_path = str(entry.get("source_path") or "")
        if not source_path.startswith("/") or source_path in source_paths or destination_relative in entries:
            raise PrizePlanV2TrainingError("v2 target transfer plan has duplicate/invalid object paths")
        source_paths.add(source_path)
        role = str(entry.get("role") or "")
        if role not in TARGET_TRANSFER_ROLES or destination_relative.lower().endswith((".zip", ".part", ".partial")):
            raise PrizePlanV2TrainingError("v2 target transfer plan includes an ineligible object")
        _sha(entry.get("sha256"), label="v2 transfer entry SHA-256")
        size = _exact_int(entry.get("size_bytes"), label="v2 transfer entry size", minimum=1)
        lane = entry.get("lane_id")
        if lane not in lane_counts:
            raise PrizePlanV2TrainingError("v2 target transfer entry lane is invalid")
        lane_counts[int(lane)] += 1
        lane_sizes[int(lane)] += size
        if role in {"target_schema", "target_shard", "target_day_manifest", "target_day_receipt"}:
            day = entry.get("utc_day")
            if not isinstance(day, str) or entry.get("split") != _expected_day_split(day):
                raise PrizePlanV2TrainingError("v2 target transfer day entry split drifted")
        entries[destination_relative] = entry
    if not all(lane_counts.values()):
        raise PrizePlanV2TrainingError("v2 target transfer plan must use all four nonempty lanes")
    observed_lanes: set[int] = set()
    for raw in lanes_raw:
        lane = _mapping(raw, label="v2 target transfer lane")
        identifier = lane.get("lane_id")
        if identifier not in lane_counts or int(identifier) in observed_lanes:
            raise PrizePlanV2TrainingError("v2 target transfer lane inventory drifted")
        observed_lanes.add(int(identifier))
        if (
            lane.get("entry_count") != lane_counts[int(identifier)]
            or lane.get("total_size_bytes") != lane_sizes[int(identifier)]
            or set(lane.get("source_paths") or ())
            != {str(item["source_path"]) for item in entries.values() if item["lane_id"] == identifier}
        ):
            raise PrizePlanV2TrainingError("v2 target transfer lane summary drifted")
    if observed_lanes != set(range(4)) or plan.get("total_size_bytes") != sum(lane_sizes.values()):
        raise PrizePlanV2TrainingError("v2 target transfer total/lane accounting drifted")
    if _json_sha(dict(plan)) != expected_sha256:
        raise PrizePlanV2TrainingError("v2 target transfer plan SHA-256 mismatch")
    return entries, source, target_days


def resolve_target_view(
    *,
    pointer: Path | str,
    expected_sha256: str,
    contract_sha256: str,
    contract_goal_revision: int,
    overlay_manifest_sha256: str,
    production: bool = True,
) -> dict[str, Any]:
    """Resolve the mandatory v2 transfer pointer and verify every copied object.

    A direct target-set manifest is deliberately not a production input: the
    pointer, its matching completion receipt, the exact four-lane plan, and
    one per-object transfer receipt together prove that local Bert bytes are
    complete, immutable, target-only copies rather than reachable Elmo paths.
    """

    pointer_path = _regular_file(pointer, label="v2 target-view pointer")
    pointer_sha = _content_addressed_receipt(pointer_path, label="v2 target-view pointer")
    if production and not expected_sha256:
        raise PrizePlanV2TrainingError("production training requires --target-view-sha256")
    if expected_sha256 and pointer_sha != _sha(expected_sha256, label="v2 target-view SHA-256"):
        raise PrizePlanV2TrainingError("v2 target-view pointer SHA-256 mismatch")
    root = _target_view_root(pointer_path)
    view = _read_json(pointer_path, label="v2 target-view pointer")
    if (
        view.get("schema") != TARGET_VIEW_SCHEMA
        or view.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
        or view.get("required_authority") != PRIZE_AUTHORITY_KEY
        or view.get("status") != "verified_target_only_offline_bert_input"
        or view.get("target_set_root_relative") != "."
        or view.get("parallel_lanes_exact") != 4
        or view.get("raw_zip_feature_or_complete_action_payload_transferred") is not False
        or view.get("bert_side_c3_or_advantage_scaling_produced_or_transferred") is not False
        or view.get("runtime_or_training_started") is not False
        or view.get("private_partials_not_training_eligible") is not True
        or view.get("create_only") is not True
    ):
        raise PrizePlanV2TrainingError("v2 target-view authority/information boundary drifted")
    target_root = _relative_directory(root, view.get("target_set_root_relative"), label="v2 target-set root", allow_dot=True)
    manifest_binding = _mapping(view.get("canonical_target_set_manifest"), label="v2 target-view manifest")
    receipt_binding = _mapping(view.get("canonical_target_set_receipt"), label="v2 target-view receipt")
    if manifest_binding.get("remains_byte_identical") is not True:
        raise PrizePlanV2TrainingError("v2 target-view rewrites the canonical target-set manifest")
    manifest_path = _relative_member(root, manifest_binding.get("relative_path"), label="v2 target-set manifest")
    receipt_path = _relative_member(root, receipt_binding.get("relative_path"), label="v2 target-set receipt")
    try:
        manifest_path.relative_to(target_root)
        receipt_path.relative_to(target_root)
    except ValueError as exc:
        raise PrizePlanV2TrainingError("v2 target-set manifest/receipt escaped the target root") from exc
    manifest_sha = _sha_match(manifest_path, manifest_binding.get("sha256"), label="v2 target-set manifest")
    receipt_sha = _sha_match(receipt_path, receipt_binding.get("sha256"), label="v2 target-set receipt")
    plan_relative = _safe_relative_text(view.get("plan_relative"), label="v2 target-view plan path")
    plan_path = _relative_member(root, plan_relative, label="v2 target transfer plan")
    plan_sha = _content_addressed_receipt(plan_path, label="v2 target transfer plan")
    if plan_sha != _sha(view.get("plan_sha256"), label="v2 target-view plan SHA-256"):
        raise PrizePlanV2TrainingError("v2 target-view plan binding drifted")
    plan = _read_json(plan_path, label="v2 target transfer plan")
    entries, source, target_days = _validate_target_transfer_plan(
        plan,
        root=root,
        expected_sha256=plan_sha,
        contract_sha256=_sha(contract_sha256, label="contract SHA-256"),
        contract_goal_revision=contract_goal_revision,
        overlay_manifest_sha256=_sha(overlay_manifest_sha256, label="overlay manifest SHA-256"),
    )
    if production and (
        source.get("host") != "elmo"
        or plan.get("test_only_non_elmo_source") is not False
        or _exact_int(
            plan.get("bert_disk_free_floor_bytes"),
            label="v2 target transfer Bert disk floor",
            minimum=1,
        )
        < MIN_PRODUCTION_DISK_FLOOR_BYTES
        or _exact_int(
            plan.get("metadata_reserve_bytes"),
            label="v2 target transfer metadata reserve",
            minimum=1,
        )
        < MIN_PRODUCTION_METADATA_RESERVE_BYTES
    ):
        raise PrizePlanV2TrainingError(
            "production v2 target transfer must be an Elmo source plan with "
            "test_only_non_elmo_source=false and the receipt-bound 20 GiB/4 MiB floors"
        )
    if (
        source.get("target_set_manifest_relative") != manifest_binding.get("relative_path")
        or source.get("target_set_manifest_sha256") != manifest_sha
        or source.get("target_set_receipt_relative") != receipt_binding.get("relative_path")
        or source.get("target_set_receipt_sha256") != receipt_sha
        or view.get("target_days") != target_days
    ):
        raise PrizePlanV2TrainingError("v2 target-view target-set/day binding drifted")
    graph = _mapping(view.get("source_identity_graph"), label="v2 target-view source graph")
    expected_graph = {
        "goal_contract_relative": source.get("goal_contract_relative"),
        "goal_contract_sha256": source.get("goal_contract_sha256"),
        "goal_contract_goal_revision": source.get("goal_contract_goal_revision"),
        "embedded_semantic_owner_goal_revision": SEMANTIC_OWNER_REVISION,
        "complete_action_overlay_manifest_relative": source.get("complete_action_overlay_manifest_relative"),
        "complete_action_overlay_manifest_sha256": source.get("complete_action_overlay_manifest_sha256"),
        "phi_fit": source.get("phi_fit"),
        "target_value_transform": source.get("target_value_transform"),
        "raw_episode_zip_identities": source.get("raw_episode_zip_identities"),
    }
    if dict(graph) != expected_graph:
        raise PrizePlanV2TrainingError("v2 target-view source identity graph drifted")
    raw_receipts = view.get("file_receipts")
    if not isinstance(raw_receipts, list) or len(raw_receipts) != len(entries):
        raise PrizePlanV2TrainingError("v2 target-view file-receipt inventory drifted")
    receipt_inventory: list[dict[str, str]] = []
    seen_destinations: set[str] = set()
    for raw in raw_receipts:
        item = _mapping(raw, label="v2 target-view file receipt reference")
        destination = _safe_relative_text(item.get("destination_relative"), label="v2 file receipt destination")
        entry = entries.get(destination)
        if entry is None or destination in seen_destinations:
            raise PrizePlanV2TrainingError("v2 target-view file receipt does not map one-to-one to plan entries")
        seen_destinations.add(destination)
        receipt_relative = _safe_relative_text(item.get("receipt_relative"), label="v2 file receipt path")
        receipt_file = _relative_member(root, receipt_relative, label="v2 per-object transfer receipt")
        # The per-object receipt filename is deterministically keyed by the
        # plan entry; its own SHA is carried by the target-view reference.
        # Do not conflate those two content-address domains.
        receipt_file_sha = sha256_file(receipt_file)
        if receipt_file_sha != _sha(item.get("receipt_sha256"), label="v2 per-object receipt SHA-256"):
            raise PrizePlanV2TrainingError("v2 per-object receipt SHA-256 drifted")
        document = _read_json(receipt_file, label="v2 per-object transfer receipt")
        source_item = _mapping(document.get("source"), label="v2 per-object receipt source")
        destination_item = _mapping(document.get("destination"), label="v2 per-object receipt destination")
        if (
            document.get("schema") != TARGET_TRANSFER_FILE_RECEIPT_SCHEMA
            or document.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
            or document.get("required_authority") != PRIZE_AUTHORITY_KEY
            or document.get("plan_sha256") != plan_sha
            or source_item.get("path") != entry.get("source_path")
            or source_item.get("sha256") != entry.get("sha256")
            or source_item.get("size_bytes") != entry.get("size_bytes")
            or destination_item.get("relative_path") != destination
            or destination_item.get("sha256") != entry.get("sha256")
            or destination_item.get("size_bytes") != entry.get("size_bytes")
            or destination_item.get("regular_non_symlink") is not True
            or document.get("role") != entry.get("role")
            or document.get("utc_day") != entry.get("utc_day")
            or document.get("split") != entry.get("split")
            or document.get("disposition") != "verified_exact"
            or document.get("source_destination_sha256_size_match") is not True
            or document.get("raw_zip_feature_or_complete_action_payload_transferred") is not False
            or document.get("private_partials_not_training_eligible") is not True
            or document.get("create_only") is not True
        ):
            raise PrizePlanV2TrainingError("v2 per-object transfer receipt binding drifted")
        object_file = _relative_member(root, destination, label="v2 transferred target object")
        if object_file.stat().st_size != entry.get("size_bytes") or sha256_file(object_file) != entry.get("sha256"):
            raise PrizePlanV2TrainingError("v2 transferred target object identity drifted")
        receipt_inventory.append({"relative_path": receipt_relative, "sha256": receipt_file_sha})
    if seen_destinations != set(entries):
        raise PrizePlanV2TrainingError("v2 target-view is missing a per-object receipt")
    completion_root = root / "transfer" / "completion"
    if completion_root.is_symlink() or not completion_root.is_dir():
        raise PrizePlanV2TrainingError("v2 target-view completion directory is absent")
    pointer_relative = str(pointer_path.relative_to(root))
    completions: list[tuple[Path, str, Mapping[str, Any]]] = []
    for candidate in sorted(completion_root.glob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        candidate_sha = _content_addressed_receipt(candidate, label="v2 target transfer completion")
        completion = _read_json(candidate, label="v2 target transfer completion")
        if (
            completion.get("schema") == TARGET_VIEW_COMPLETION_SCHEMA
            and completion.get("target_view_relative") == pointer_relative
            and completion.get("target_view_sha256") == pointer_sha
        ):
            completions.append((candidate, candidate_sha, completion))
    if len(completions) != 1:
        raise PrizePlanV2TrainingError("v2 target-view must have exactly one matching completion receipt")
    completion_path, completion_sha, completion = completions[0]
    phi = _mapping(source.get("phi_fit"), label="v2 target transfer Phi source")
    transform = _mapping(source.get("target_value_transform"), label="v2 target transfer transform source")
    if (
        completion.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
        or completion.get("required_authority") != PRIZE_AUTHORITY_KEY
        or completion.get("status") != "complete_verified_target_only_transfer"
        or completion.get("plan_relative") != plan_relative
        or completion.get("plan_sha256") != plan_sha
        or completion.get("target_set_manifest_sha256") != manifest_sha
        or completion.get("target_set_receipt_sha256") != receipt_sha
        or completion.get("goal_contract_sha256") != contract_sha256
        or completion.get("phi_fit_manifest_sha256") != phi.get("fit_manifest_sha256")
        or completion.get("phi_fit_receipt_sha256") != phi.get("fit_receipt_sha256")
        or completion.get("frozen_phi_table_sha256") != phi.get("frozen_phi_table_sha256")
        or completion.get("target_value_transform_sha256") != transform.get("sha256")
        or completion.get("entry_count") != len(entries)
        or completion.get("parallel_lanes_exact") != 4
        or completion.get("source_destination_sha256_size_verified") is not True
        or completion.get("raw_zip_feature_or_complete_action_payload_transferred") is not False
        or completion.get("bert_side_c3_or_advantage_scaling_produced_or_transferred") is not False
        or completion.get("runtime_or_training_started") is not False
        or completion.get("private_partials_not_training_eligible") is not True
        or completion.get("create_only") is not True
    ):
        raise PrizePlanV2TrainingError("v2 target transfer completion binding drifted")
    return {
        "pointer_path": str(pointer_path),
        "pointer_sha256": pointer_sha,
        "root": str(root),
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "entry_count": len(entries),
        "parallel_lanes_exact": 4,
        "source_host": source.get("host"),
        "test_only_non_elmo_source": plan.get("test_only_non_elmo_source"),
        "private_partials_not_training_eligible": True,
        "per_object_receipts": sorted(receipt_inventory, key=lambda item: item["relative_path"]),
    }


def _identity_from_row(row: Mapping[str, Any], *, label: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if field in {"acting_seat", "env_step"}:
            value = _exact_int(value, label=f"{label}.{field}", minimum=0)
        elif not isinstance(value, str) or not value:
            raise PrizePlanV2TrainingError(f"{label}.{field} is absent")
        values.append(value)
    if values[4] not in (0, 1):
        raise PrizePlanV2TrainingError(f"{label}.acting_seat must be 0 or 1")
    _sha(values[1], label=f"{label}.source_archive_sha256")
    return tuple(values)


def _expected_day_split(day: str) -> str:
    for split, days in EXPECTED_SPLITS.items():
        if day in days:
            return split
    raise PrizePlanV2TrainingError(f"day outside sealed recent-20 window: {day}")


@dataclass(frozen=True)
class TargetDay:
    utc_day: str
    split: str
    shard_path: Path
    shard_sha256: str
    row_count: int
    day_manifest_sha256: str
    day_receipt_sha256: str


class PrizePlanTargetSet:
    """Strict portable v2 target-set reader, retaining no target rows in RAM."""

    def __init__(
        self,
        *,
        root: Path | str,
        manifest_path: Path | str,
        manifest_sha256: str,
        receipt_path: Path | str,
        receipt_sha256: str,
        contract_sha256: str,
        contract_goal_revision: int,
        overlay_manifest_sha256: str,
        production: bool = True,
    ) -> None:
        self.root = _regular_directory(root, label="portable v2 target-set root")
        self.manifest_path = _regular_file(manifest_path, label="v2 target-set manifest")
        self.receipt_path = _regular_file(receipt_path, label="v2 target-set receipt")
        try:
            self.manifest_path.relative_to(self.root)
            self.receipt_path.relative_to(self.root)
        except ValueError as exc:
            raise PrizePlanV2TrainingError("target-set manifest or receipt escapes target root") from exc
        self.manifest_sha256 = _sha_match(self.manifest_path, manifest_sha256, label="v2 target-set manifest")
        self.receipt_sha256 = _sha_match(self.receipt_path, receipt_sha256, label="v2 target-set receipt")
        self.manifest = _read_json(self.manifest_path, label="v2 target-set manifest")
        self.receipt = _read_json(self.receipt_path, label="v2 target-set receipt")
        self.contract_sha256 = _sha(contract_sha256, label="contract SHA-256")
        self.contract_goal_revision = int(contract_goal_revision)
        self.overlay_manifest_sha256 = _sha(overlay_manifest_sha256, label="overlay manifest SHA-256")
        self.production = bool(production)
        self.gamma, self.transform_sha256 = self._validate_aggregate()
        self.days = self._load_days()

    def _member(self, value: Any, *, label: str) -> Path:
        return _relative_member(self.root, value, label=label)

    def _validate_aggregate(self) -> tuple[float, str]:
        manifest = self.manifest
        receipt = self.receipt
        if (
            manifest.get("schema") != TARGET_SET_MANIFEST_SCHEMA
            or receipt.get("schema") != TARGET_SET_RECEIPT_SCHEMA
            or manifest.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
            or receipt.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
            or manifest.get("required_authority") != PRIZE_AUTHORITY_KEY
            or receipt.get("required_authority") != PRIZE_AUTHORITY_KEY
            or manifest.get("goal_contract_goal_revision") != self.contract_goal_revision
            or receipt.get("goal_contract_goal_revision") != self.contract_goal_revision
            or receipt.get("goal_contract_sha256") != self.contract_sha256
            or receipt.get("target_set_manifest_sha256") != self.manifest_sha256
            or receipt.get("day_count") != 20
            or receipt.get("whole_day_episode_and_group_split_disjoint") is not True
            or manifest.get("whole_day_episode_and_group_split_disjoint") is not True
        ):
            raise PrizePlanV2TrainingError("v2 target-set aggregate receipt binding drifted")
        goal = _mapping(manifest.get("goal_contract"), label="v2 target-set goal contract")
        if (
            goal.get("sha256") != self.contract_sha256
            or goal.get("goal_revision") != self.contract_goal_revision
            or goal.get("required_authority") != PRIZE_AUTHORITY_KEY
            or goal.get("semantic_owner_goal_revision") != SEMANTIC_OWNER_REVISION
        ):
            raise PrizePlanV2TrainingError("v2 target-set goal contract binding drifted")
        copied_contract = self._member(goal.get("path"), label="portable target-set contract")
        _sha_match(copied_contract, self.contract_sha256, label="portable target-set contract")
        overlay = _mapping(manifest.get("complete_action_overlay_manifest"), label="v2 target-set overlay")
        if overlay.get("sha256") != self.overlay_manifest_sha256:
            raise PrizePlanV2TrainingError("v2 target-set overlay binding drifted")
        transform = _mapping(manifest.get("target_value_transform"), label="v2 target value transform")
        transform_sha = _sha(transform.get("sha256"), label="v2 target value transform SHA-256")
        transform_path = self._member(transform.get("path"), label="v2 target value transform")
        _sha_match(transform_path, transform_sha, label="v2 target value transform")
        transform_doc = _read_json(transform_path, label="v2 target value transform")
        gamma = _finite(transform_doc.get("gamma"), label="v2 target gamma")
        if (
            transform.get("schema") != TARGET_VALUE_TRANSFORM_SCHEMA
            or transform_doc.get("schema") != TARGET_VALUE_TRANSFORM_SCHEMA
            or transform_doc.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
            or transform_doc.get("formula") != "model_target_value=raw_return_value/(1+gamma**h)"
            or transform_doc.get("horizons") != list(PRIZE_PLAN_V2_HORIZONS)
            or transform_doc.get("clipping") is not False
            or transform_doc.get("data_dependent_train_fit") is not False
            or transform_doc.get("expected_model_target_range") != [-1.0, 1.0]
            or receipt.get("target_value_transform_sha256") != transform_sha
        ):
            raise PrizePlanV2TrainingError("v2 analytic target transform drifted")
        phi = _mapping(manifest.get("phi_fit"), label="v2 target-set Phi binding")
        if (
            phi.get("fit_scope") != "sealed_train_split_only"
            or not isinstance(phi.get("fit_manifest"), Mapping)
            or not isinstance(phi.get("fit_receipt"), Mapping)
            or receipt.get("phi_fit_manifest_sha256") != phi["fit_manifest"].get("sha256")
            or receipt.get("phi_fit_receipt_sha256") != phi["fit_receipt"].get("sha256")
        ):
            raise PrizePlanV2TrainingError("v2 Phi fit binding drifted")
        for key in ("fit_manifest", "fit_receipt", "frozen_phi_table"):
            item = _mapping(phi.get(key), label=f"v2 Phi {key}")
            _sha_match(self._member(item.get("path"), label=f"v2 Phi {key}"), item.get("sha256"), label=f"v2 Phi {key}")
        information = _mapping(manifest.get("information_boundary"), label="v2 target information boundary")
        if (
            information.get("raw_zip_or_feature_or_complete_action_overlay_payload_copied") is not False
            or information.get("hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed") is not False
            or information.get("terminal_z_is_direct_plan_target_or_actor_term") is not False
        ):
            raise PrizePlanV2TrainingError("v2 target information boundary drifted")
        return gamma, transform_sha

    def _load_days(self) -> dict[str, TargetDay]:
        raw_days = self.manifest.get("target_days")
        if not isinstance(raw_days, list) or len(raw_days) != 20:
            raise PrizePlanV2TrainingError("v2 target set must contain exactly twenty day descriptors")
        split_days = _mapping(self.manifest.get("split_days"), label="v2 target split days")
        observed: dict[str, TargetDay] = {}
        for item_raw in raw_days:
            item = _mapping(item_raw, label="v2 target day descriptor")
            day = item.get("utc_day")
            split = item.get("split")
            if not isinstance(day, str) or not isinstance(split, str) or split != _expected_day_split(day) or day in observed:
                raise PrizePlanV2TrainingError("v2 target day/split inventory drifted")
            day_manifest = _mapping(item.get("day_manifest"), label=f"v2 day {day} manifest")
            day_receipt = _mapping(item.get("day_receipt"), label=f"v2 day {day} receipt")
            shard = _mapping(item.get("target_shard"), label=f"v2 day {day} target shard")
            manifest_path = self._member(day_manifest.get("path"), label=f"v2 day {day} manifest")
            receipt_path = self._member(day_receipt.get("path"), label=f"v2 day {day} receipt")
            manifest_sha = _sha_match(manifest_path, day_manifest.get("sha256"), label=f"v2 day {day} manifest")
            receipt_sha = _sha_match(receipt_path, day_receipt.get("sha256"), label=f"v2 day {day} receipt")
            day_doc = _read_json(manifest_path, label=f"v2 day {day} manifest")
            receipt_doc = _read_json(receipt_path, label=f"v2 day {day} receipt")
            target_path = self._member(shard.get("path"), label=f"v2 day {day} target shard")
            target_sha = _sha_match(target_path, shard.get("sha256"), label=f"v2 day {day} target shard")
            count = _exact_int(shard.get("row_count"), label=f"v2 day {day} target row count", minimum=1)
            if target_path.stat().st_size != _exact_int(shard.get("size_bytes"), label=f"v2 day {day} target size", minimum=1):
                raise PrizePlanV2TrainingError(f"v2 day {day} target size drifted")
            expected_transform = {
                "formula": "model_target_value=raw_return_value/(1+gamma**h)",
                "gamma": self.gamma,
                "data_dependent_train_fit": False,
                "clipping": False,
                "expected_model_target_range": [-1.0, 1.0],
                "actor_advantage_scaling": "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform",
            }
            if (
                day_doc.get("schema") != TARGET_DAY_MANIFEST_SCHEMA
                or day_doc.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
                or day_doc.get("utc_day") != day
                or day_doc.get("split") != split
                or day_doc.get("gamma") != self.gamma
                or day_doc.get("target_value_transform") != expected_transform
                or receipt_doc.get("schema") != TARGET_DAY_RECEIPT_SCHEMA
                or receipt_doc.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
                or receipt_doc.get("day_manifest_sha256") != manifest_sha
                or receipt_doc.get("goal_contract_sha256") != self.contract_sha256
                or receipt_doc.get("goal_contract_goal_revision") != self.contract_goal_revision
                or receipt_doc.get("gamma") != self.gamma
                or receipt_doc.get("target_value_transform") != expected_transform
                or receipt_doc.get("target_shard_sha256") != target_sha
                or receipt_doc.get("target_row_count") != count
                or receipt_doc.get("terminal_z_used_as_direct_plan_reward_or_target") is not False
                or receipt_doc.get("hidden_information_output_fields_present") is not False
            ):
                raise PrizePlanV2TrainingError(f"v2 day {day} manifest/receipt binding drifted")
            overlay = _mapping(day_doc.get("complete_action_overlay"), label=f"v2 day {day} overlay")
            aggregate_overlay = _mapping(item.get("complete_action_overlay"), label=f"v2 day {day} aggregate overlay")
            if (
                overlay.get("sha256") != aggregate_overlay.get("sha256")
                or overlay.get("size_bytes") != aggregate_overlay.get("size_bytes")
                or receipt_doc.get("complete_action_overlay_sha256") != overlay.get("sha256")
            ):
                raise PrizePlanV2TrainingError(f"v2 day {day} overlay identity drifted")
            observed[day] = TargetDay(day, split, target_path, target_sha, count, manifest_sha, receipt_sha)
        if set(observed) != set(sum((list(days) for days in EXPECTED_SPLITS.values()), [])):
            raise PrizePlanV2TrainingError("v2 target days are not the exact sealed recent-20 window")
        if any(tuple(split_days.get(split) or ()) != days for split, days in EXPECTED_SPLITS.items()):
            raise PrizePlanV2TrainingError("v2 target split-day lists drifted")
        return observed

    def iter_rows(self, day: str) -> Iterator[dict[str, Any]]:
        descriptor = self.days.get(day)
        if descriptor is None:
            raise PrizePlanV2TrainingError(f"v2 target set lacks day {day}")
        count = 0
        with descriptor.shard_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
            for number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PrizePlanV2TrainingError(
                        f"invalid v2 target row {descriptor.shard_path}:{number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise PrizePlanV2TrainingError("v2 target row must be an object")
                self.validate_row(row, day=descriptor.utc_day, split=descriptor.split)
                count += 1
                yield row
        if count != descriptor.row_count:
            raise PrizePlanV2TrainingError(f"v2 target day {day} row count drifted")

    def validate_row(self, row: Mapping[str, Any], *, day: str, split: str) -> None:
        if (
            row.get("schema") != TARGET_ROW_SCHEMA
            or row.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION
            or row.get("goal_contract_goal_revision") != self.contract_goal_revision
            or row.get("utc_day") != day
            or row.get("split") != split
            or row.get("target_only") is not True
            or row.get("hidden_information_fields_present") is not False
            or row.get("terminal_z_present") is not False
        ):
            raise PrizePlanV2TrainingError("v2 target row core authority/information fields drifted")
        _identity_from_row(row, label="v2 target row")
        forbidden = {"z", "terminal_z", "win_target", "reward", "outcome", "terminal_return"}
        if forbidden.intersection(row):
            raise PrizePlanV2TrainingError("v2 target row carries a forbidden terminal label")
        returns = _mapping(row.get("prize_plan_returns"), label="v2 target row returns")
        if set(returns) != {f"h{horizon}" for horizon in PRIZE_PLAN_V2_HORIZONS}:
            raise PrizePlanV2TrainingError("v2 target row horizon inventory drifted")
        for horizon in PRIZE_PLAN_V2_HORIZONS:
            target = _mapping(returns.get(f"h{horizon}"), label=f"v2 target H{horizon}")
            if (
                target.get("h") != horizon
                or target.get("gamma") != self.gamma
                or target.get("required_segment_count") != horizon
                or target.get("mask") not in (True, False)
            ):
                raise PrizePlanV2TrainingError("v2 target horizon identity/mask drifted")
            mask = bool(target["mask"])
            raw = target.get("raw_return_value")
            model = target.get("model_target_value")
            if not mask:
                if raw is not None or model is not None or not isinstance(target.get("unavailable_reason"), str):
                    raise PrizePlanV2TrainingError("unavailable v2 target horizon fabricated a value")
                continue
            raw_value = _finite(raw, label=f"v2 target H{horizon} raw return")
            model_value = _finite(model, label=f"v2 target H{horizon} model target")
            bound = 1.0 + self.gamma**horizon
            if not -bound - 1e-10 <= raw_value <= bound + 1e-10:
                raise PrizePlanV2TrainingError("v2 raw target exceeds analytic bound")
            if not -1.0 - 1e-10 <= model_value <= 1.0 + 1e-10:
                raise PrizePlanV2TrainingError("v2 model target exceeds analytic bound")
            if not math.isclose(model_value, raw_value / bound, rel_tol=0.0, abs_tol=1e-10):
                raise PrizePlanV2TrainingError("v2 model target is not raw/(1+gamma**h)")
            if target.get("unavailable_reason") is not None or target.get("segment_count") != horizon:
                raise PrizePlanV2TrainingError("available v2 target horizon is incomplete")


@dataclass(frozen=True)
class CompleteActionExample:
    """One sealed chosen complete action, with only v2 model target values."""

    identity: tuple[Any, ...]
    utc_day: str
    stage_count: int
    first_stage_menu: tuple[tuple[float, ...], ...]
    selected_stage_features: tuple[tuple[float, ...], ...]
    selected_option_indices: tuple[int, ...]
    selected_legal_counts: tuple[int, ...]
    selected_action_programs: tuple[tuple[int, ...], ...]
    plan_targets: tuple[float, float, float, float]
    plan_masks: tuple[bool, bool, bool, bool]


def _feature_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != FEATURE_WIDTH:
        raise PrizePlanV2TrainingError(f"{label} must be a {FEATURE_WIDTH}-wide feature vector")
    result = tuple(_finite(item, label=label) for item in value)
    return result


def _action_tokens(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > MAX_TOKENS:
        raise PrizePlanV2TrainingError(f"{label} must be an action-program list within the ABI bound")
    result: list[int] = []
    for item in value:
        token = _exact_int(item, label=label, minimum=0)
        if token > MAX_TOKEN_VALUE:
            raise PrizePlanV2TrainingError(f"{label} contains token above fixed v2 ABI")
        result.append(token)
    return tuple(result)


def example_from_join(sample: Mapping[str, Any], target: Mapping[str, Any]) -> CompleteActionExample:
    """Validate the exact seven-field join and seal a complete-action batch row."""

    if sample.get("public_information_only") is not True or sample.get("base_feature_width") != FEATURE_WIDTH:
        raise PrizePlanV2TrainingError("feature sample crossed the public/40-wide boundary")
    program = _mapping(sample.get("program"), label="complete-action program")
    program_identity = _identity_from_row(program, label="complete-action program")
    target_identity = _identity_from_row(target, label="v2 target row")
    if program_identity != target_identity:
        raise PrizePlanV2TrainingError("complete-action / v2 target exact seven-field join mismatch")
    stages = program.get("stages")
    feature_stages = sample.get("base_option_features_by_stage")
    if not isinstance(stages, list) or not isinstance(feature_stages, list) or not stages:
        raise PrizePlanV2TrainingError("complete-action stages or base feature alignment is absent")
    if len(stages) != len(feature_stages) or len(stages) > MAX_STAGES:
        raise PrizePlanV2TrainingError("complete-action stage count exceeds the sealed v2 ABI")
    first_menu: tuple[tuple[float, ...], ...] | None = None
    selected_features: list[tuple[float, ...]] = []
    selected_indices: list[int] = []
    selected_legal_counts: list[int] = []
    selected_programs: list[tuple[int, ...]] = []
    for index, (stage_raw, rows_raw) in enumerate(zip(stages, feature_stages, strict=True)):
        stage = _mapping(stage_raw, label=f"complete-action stage {index}")
        if stage.get("factorized_stage") != index:
            raise PrizePlanV2TrainingError("factorized stages are not contiguous")
        if not isinstance(rows_raw, list) or not rows_raw or len(rows_raw) > MAX_LEGAL:
            raise PrizePlanV2TrainingError("stage has invalid legal option feature rows")
        rows = tuple(_feature_vector(row, label=f"stage {index} option") for row in rows_raw)
        legal = stage.get("ordered_legal_action_programs")
        valid = stage.get("valid_option_mask")
        selected = stage.get("selected_option_index")
        action = stage.get("selected_action_program")
        if (
            not isinstance(legal, list)
            or not isinstance(valid, list)
            or len(legal) != len(rows)
            or valid != [True] * len(rows)
            or isinstance(selected, bool)
            or not isinstance(selected, int)
            or not 0 <= selected < len(rows)
            or legal[selected] != action
        ):
            raise PrizePlanV2TrainingError("sealed legal order/selected complete-action alignment drifted")
        if index == 0:
            first_menu = rows
        selected_features.append(rows[selected])
        selected_indices.append(selected)
        selected_legal_counts.append(len(rows))
        selected_programs.append(_action_tokens(action, label=f"stage {index} selected action"))
    assert first_menu is not None
    returns = _mapping(target.get("prize_plan_returns"), label="v2 target returns")
    labels: list[float] = []
    masks: list[bool] = []
    for horizon in PRIZE_PLAN_V2_HORIZONS:
        plan = _mapping(returns.get(f"h{horizon}"), label=f"v2 target h{horizon}")
        available = bool(plan.get("mask"))
        masks.append(available)
        # This is intentionally the sole label field that leaves a target row.
        labels.append(_finite(plan["model_target_value"], label=f"v2 model target h{horizon}") if available else 0.0)
    return CompleteActionExample(
        identity=program_identity,
        utc_day=str(program_identity[0]),
        stage_count=len(selected_features),
        first_stage_menu=first_menu,
        selected_stage_features=tuple(selected_features),
        selected_option_indices=tuple(selected_indices),
        selected_legal_counts=tuple(selected_legal_counts),
        selected_action_programs=tuple(selected_programs),
        plan_targets=tuple(labels),  # type: ignore[arg-type]
        plan_masks=tuple(masks),  # type: ignore[arg-type]
    )


def _split_days_from_dataset(dataset: Recent20RTPDataset, split: str) -> tuple[str, ...]:
    days = tuple(str(item.get("utc_day") or "") for item in dataset.shards if item.get("split") == split)
    if not days or any(not day for day in days) or len(days) != len(set(days)):
        raise PrizePlanV2TrainingError(f"base overlay {split} day inventory is malformed")
    return days


def assert_split_contract(
    dataset: Recent20RTPDataset,
    targets: PrizePlanTargetSet,
    *,
    allow_noncanonical: bool = False,
) -> dict[str, list[str]]:
    result = {split: list(_split_days_from_dataset(dataset, split)) for split in EXPECTED_SPLITS}
    if not allow_noncanonical:
        if any(tuple(result[split]) != days for split, days in EXPECTED_SPLITS.items()):
            raise PrizePlanV2TrainingError("base overlay split day inventory is not sealed 14/3/3")
        if any(tuple(day.utc_day for day in targets.days.values() if day.split == split) != days for split, days in EXPECTED_SPLITS.items()):
            # Target-day mappings are insertion ordered from their manifest;
            # require its portable order to remain source-day order as well.
            raise PrizePlanV2TrainingError("v2 target split day inventory is not sealed 14/3/3")
    for split, days in result.items():
        for day in days:
            descriptor = targets.days.get(day)
            if descriptor is None or descriptor.split != split:
                raise PrizePlanV2TrainingError("base/target split day inventory mismatch")
    return result


def iter_complete_action_examples(
    dataset: Recent20RTPDataset,
    targets: PrizePlanTargetSet,
    *,
    split: str,
    max_programs: int = 0,
) -> Iterator[CompleteActionExample]:
    """Stream one exact target row beside each base program, never materializing all rows."""

    if max_programs < 0:
        raise PrizePlanV2TrainingError("program limit cannot be negative")
    target_rows: Iterator[dict[str, Any]] | None = None
    target_day = ""
    yielded = 0
    for sample in dataset.iter_samples(split):
        program = _mapping(sample.get("program"), label="complete-action program")
        day = str(program.get("utc_day") or "")
        if day != target_day:
            if target_rows is not None:
                try:
                    extra = next(target_rows)
                except StopIteration:
                    pass
                else:
                    raise PrizePlanV2TrainingError(
                        f"v2 target shard {target_day} has an unmatched row {extra.get('program_identity')!r}"
                    )
            target_day = day
            target_rows = targets.iter_rows(day)
        assert target_rows is not None
        try:
            target = next(target_rows)
        except StopIteration as exc:
            raise PrizePlanV2TrainingError(f"v2 target shard {day} ended before base overlay") from exc
        yield example_from_join(sample, target)
        yielded += 1
        if max_programs and yielded >= max_programs:
            # A smoke limit intentionally does not assert full shard closure;
            # its receipt records the non-full consumed subset.
            return
    if target_rows is not None:
        try:
            extra = next(target_rows)
        except StopIteration:
            pass
        else:
            raise PrizePlanV2TrainingError(
                f"v2 target shard {target_day} has unmatched trailing row {extra.get('program_identity')!r}"
            )
    if not yielded:
        raise PrizePlanV2TrainingError(f"sealed {split} split produced no complete actions")


def bounded_shuffle(
    rows: Iterable[CompleteActionExample], *, buffer_size: int, seed: int
) -> Iterator[CompleteActionExample]:
    """Fixed-memory stream shuffle; it changes no replay membership or weights."""

    if buffer_size < 1:
        raise PrizePlanV2TrainingError("shuffle buffer must be positive")
    randomizer = random.Random(seed)
    buffer: list[CompleteActionExample] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = randomizer.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    randomizer.shuffle(buffer)
    yield from buffer


def batched(rows: Iterable[CompleteActionExample], *, batch_size: int) -> Iterator[list[CompleteActionExample]]:
    if batch_size < 1:
        raise PrizePlanV2TrainingError("batch size must be positive")
    batch: list[CompleteActionExample] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def padded_complete_action_inputs(
    rows: Sequence[CompleteActionExample], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise PrizePlanV2TrainingError("cannot collate an empty complete-action batch")
    for row in rows:
        stages = row.stage_count
        if (
            not 1 <= stages <= MAX_STAGES
            or len(row.first_stage_menu) < 1
            or len(row.first_stage_menu) > MAX_LEGAL
            or len(row.selected_stage_features) != stages
            or len(row.selected_option_indices) != stages
            or len(row.selected_legal_counts) != stages
            or len(row.selected_action_programs) != stages
        ):
            raise PrizePlanV2TrainingError("complete-action structure is outside sealed v2 ABI")
    maximum_menu = max(len(row.first_stage_menu) for row in rows)
    maximum_stages = max(row.stage_count for row in rows)
    maximum_tokens = max(1, max(len(tokens) for row in rows for tokens in row.selected_action_programs))
    menu = torch.zeros((len(rows), maximum_menu, FEATURE_WIDTH), dtype=torch.float32, device=device)
    menu_mask = torch.zeros((len(rows), maximum_menu), dtype=torch.bool, device=device)
    selected = torch.zeros((len(rows), maximum_stages, FEATURE_WIDTH), dtype=torch.float32, device=device)
    selected_mask = torch.zeros((len(rows), maximum_stages), dtype=torch.bool, device=device)
    indices = torch.zeros((len(rows), maximum_stages), dtype=torch.int64, device=device)
    counts = torch.zeros((len(rows), maximum_stages), dtype=torch.int64, device=device)
    tokens = torch.zeros((len(rows), maximum_stages, maximum_tokens), dtype=torch.int64, device=device)
    token_mask = torch.zeros((len(rows), maximum_stages, maximum_tokens), dtype=torch.bool, device=device)
    for batch_index, row in enumerate(rows):
        menu[batch_index, : len(row.first_stage_menu)] = torch.tensor(row.first_stage_menu, dtype=torch.float32, device=device)
        menu_mask[batch_index, : len(row.first_stage_menu)] = True
        selected[batch_index, : row.stage_count] = torch.tensor(row.selected_stage_features, dtype=torch.float32, device=device)
        selected_mask[batch_index, : row.stage_count] = True
        for stage, (chosen, legal_count, action_tokens) in enumerate(
            zip(row.selected_option_indices, row.selected_legal_counts, row.selected_action_programs, strict=True)
        ):
            if not 1 <= legal_count <= MAX_LEGAL or not 0 <= chosen < legal_count:
                raise PrizePlanV2TrainingError("selected option index / legal count drifted")
            if len(action_tokens) > MAX_TOKENS or any(token < 0 or token > MAX_TOKEN_VALUE for token in action_tokens):
                raise PrizePlanV2TrainingError("selected action program drifted outside sealed v2 ABI")
            indices[batch_index, stage] = chosen
            counts[batch_index, stage] = legal_count
            if action_tokens:
                tokens[batch_index, stage, : len(action_tokens)] = torch.tensor(action_tokens, dtype=torch.int64, device=device)
                token_mask[batch_index, stage, : len(action_tokens)] = True
    return menu, menu_mask, selected, selected_mask, indices, counts, tokens, token_mask


def target_batch(rows: Sequence[CompleteActionExample], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.tensor([row.plan_targets for row in rows], dtype=torch.float32, device=device)
    masks = torch.tensor([row.plan_masks for row in rows], dtype=torch.bool, device=device)
    if targets.shape != (len(rows), len(PRIZE_PLAN_V2_HORIZONS)) or masks.shape != targets.shape:
        raise PrizePlanV2TrainingError("v2 target batch shape drifted")
    if not bool(torch.isfinite(targets[masks]).all().detach().cpu().item()):
        raise PrizePlanV2TrainingError("v2 labelled model targets are non-finite")
    if bool(((targets[masks] < -1.0) | (targets[masks] > 1.0)).any().detach().cpu().item()):
        raise PrizePlanV2TrainingError("v2 labelled model targets are outside analytic bound")
    return targets, masks


@dataclass
class _RegressionMetric:
    count: int = 0
    target_sum: float = 0.0
    target_sq_sum: float = 0.0
    squared_error_sum: float = 0.0

    def add(self, prediction: float, target: float) -> None:
        self.count += 1
        self.target_sum += target
        self.target_sq_sum += target * target
        self.squared_error_sum += (prediction - target) ** 2

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {"available": False, "count": 0}
        mean = self.target_sum / self.count
        mse = self.squared_error_sum / self.count
        zero = self.target_sq_sum / self.count
        empirical = max(0.0, zero - mean * mean)
        return {
            "available": True,
            "count": self.count,
            "mse": mse,
            "zero_baseline_mse": zero,
            "empirical_mean_baseline_mse": empirical,
            "empirical_target_mean": mean,
            "better_than_zero_baseline": mse < zero,
            "better_than_empirical_mean_baseline": mse < empirical,
        }


class Metrics:
    """Bounded validation/training metrics without storing predictions."""

    def __init__(self) -> None:
        self.actions = 0
        self.stages = 0
        self.days: set[str] = set()
        self.masked: Counter[str] = Counter()
        self.loss_sum = 0.0
        self.loss_count = 0
        self.outputs = {name: _RegressionMetric() for name in PRIZE_PLAN_V2_OUTPUT_NAMES}

    def update(
        self,
        rows: Sequence[CompleteActionExample],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        values = predictions.detach().cpu().tolist()
        labels = targets.detach().cpu().tolist()
        availability = masks.detach().cpu().tolist()
        self.loss_sum += float(loss.detach().cpu().item())
        self.loss_count += 1
        for row_index, row in enumerate(rows):
            self.actions += 1
            self.stages += row.stage_count
            self.days.add(row.utc_day)
            for horizon_index, horizon in enumerate(PRIZE_PLAN_V2_HORIZONS):
                if not availability[row_index][horizon_index]:
                    self.masked[f"h{horizon}"] += 1
                    continue
                target = float(labels[row_index][horizon_index])
                self.outputs[f"V_plan_{horizon}"].add(float(values[row_index][2 * horizon_index]), target)
                self.outputs[f"Q_plan_{horizon}"].add(float(values[row_index][2 * horizon_index + 1]), target)

    def summary(self) -> dict[str, Any]:
        coverage: dict[str, Any] = {}
        for index, horizon in enumerate(PRIZE_PLAN_V2_HORIZONS):
            available = self.outputs[f"V_plan_{horizon}"].count
            coverage[str(horizon)] = {
                "available": available,
                "masked": int(self.masked[f"h{horizon}"]),
                "coverage": available / self.actions if self.actions else 0.0,
            }
        return {
            "complete_actions": self.actions,
            "factorized_stages": self.stages,
            "days": sorted(self.days),
            "mean_loss": self.loss_sum / self.loss_count if self.loss_count else None,
            "loss_batches": self.loss_count,
            "coverage": {"plan_horizons": coverage},
            "baselines": {name: metric.summary() for name, metric in self.outputs.items()},
            "finite": True,
        }


def critic_predictions(
    model: PrizePlanV2Sidecar, rows: Sequence[CompleteActionExample], *, device: torch.device
) -> torch.Tensor:
    try:
        predictions = model(*padded_complete_action_inputs(rows, device=device))
    except PrizePlanV2SidecarError as exc:
        raise PrizePlanV2TrainingError("Prize-plan-v2 sidecar rejected sealed action ABI") from exc
    if predictions.dtype != torch.float32 or tuple(predictions.shape) != (len(rows), 8):
        raise PrizePlanV2TrainingError("Prize-plan-v2 sidecar output shape/dtype drifted")
    if not bool(torch.isfinite(predictions).all().detach().cpu().item()):
        raise PrizePlanV2TrainingError("Prize-plan-v2 sidecar emitted non-finite predictions")
    return predictions


def run_batches(
    model: PrizePlanV2Sidecar,
    rows: Iterable[CompleteActionExample],
    *,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> tuple[Metrics, int]:
    metrics = Metrics()
    steps = 0
    training = optimizer is not None
    model.train(training)
    for batch in batched(rows, batch_size=batch_size):
        targets, masks = target_batch(batch, device=device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            predictions = critic_predictions(model, batch, device=device)
            loss = masked_prize_plan_v2_loss(predictions, targets, masks).total
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise PrizePlanV2TrainingError("Prize-plan-v2 critic loss is non-finite")
            loss.backward()
            if not all(
                bool(torch.isfinite(parameter.grad).all().detach().cpu().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise PrizePlanV2TrainingError("Prize-plan-v2 critic gradient is non-finite")
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            steps += 1
        else:
            with torch.no_grad():
                predictions = critic_predictions(model, batch, device=device)
                loss = masked_prize_plan_v2_loss(predictions, targets, masks).total
        metrics.update(batch, predictions, targets, masks, loss)
    if not metrics.actions:
        raise PrizePlanV2TrainingError("critic run had zero complete actions")
    return metrics, steps


def _public_action_signature(row: CompleteActionExample) -> str:
    """Stable, current-decision-only support fingerprint; never actor input."""

    value = {
        "selected_option_indices": list(row.selected_option_indices),
        "selected_legal_counts": list(row.selected_legal_counts),
        "selected_action_programs": [list(item) for item in row.selected_action_programs],
    }
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def fit_h3_scale_support(
    model: PrizePlanV2Sidecar,
    rows: Iterable[CompleteActionExample],
    *,
    device: torch.device,
    batch_size: int,
    source_binding: Mapping[str, Any],
    sidecar_checkpoint: Mapping[str, Any],
    full_train_split_consumed: bool = True,
) -> dict[str, Any]:
    """Fit a frozen train-only scale from the sidecar's actual H3 ``Q-V``.

    This is deliberately a post-training, no-grad diagnostic pass.  Fitting a
    target-label standard deviation would not be an ``A_plan_3`` scale: the
    later formula is defined in terms of the frozen sidecar's *predicted*
    chosen-action advantage.  The pass reads only recorded train actions and
    their H3 availability bit; it neither trains the critic nor makes a
    runtime/actor call.  Its output remains actor-ineligible until a separate
    owner-authorized c3 and safe-boundary receipt exist.
    """

    if batch_size < 1:
        raise PrizePlanV2TrainingError("H3 scale/support batch size must be positive")
    checkpoint_path = str(sidecar_checkpoint.get("path") or "")
    checkpoint_sha = _sha(
        sidecar_checkpoint.get("sha256"), label="H3 scale/support sidecar checkpoint SHA-256"
    )
    if not checkpoint_path:
        raise PrizePlanV2TrainingError("H3 scale/support sidecar checkpoint path is absent")
    h3_index = PRIZE_PLAN_V2_HORIZONS.index(3)
    v_index = 2 * h3_index
    q_index = v_index + 1
    count = 0
    advantage_sum = 0.0
    advantage_sq_sum = 0.0
    advantage_min = math.inf
    advantage_max = -math.inf
    structural_counts: Counter[str] = Counter()
    legal_counts: Counter[int] = Counter()
    stages: Counter[int] = Counter()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batched(rows, batch_size=batch_size):
                predictions = critic_predictions(model, batch, device=device)
                advantages = (predictions[:, q_index] - predictions[:, v_index]).detach().cpu().tolist()
                for row, advantage in zip(batch, advantages, strict=True):
                    if not row.plan_masks[h3_index]:
                        continue
                    value = float(advantage)
                    if not math.isfinite(value):
                        raise PrizePlanV2TrainingError("H3 predicted action advantage is non-finite")
                    count += 1
                    advantage_sum += value
                    advantage_sq_sum += value * value
                    advantage_min = min(advantage_min, value)
                    advantage_max = max(advantage_max, value)
                    structural_counts[_public_action_signature(row)] += 1
                    legal_counts[row.selected_legal_counts[0]] += 1
                    stages[row.stage_count] += 1
    finally:
        model.train(was_training)
    if not count:
        raise PrizePlanV2TrainingError("train split has no H3 labels for scale/support diagnostic")
    mean = advantage_sum / count
    variance = max(0.0, advantage_sq_sum / count - mean * mean)
    stddev = math.sqrt(variance)
    rms = math.sqrt(advantage_sq_sum / count)
    # Never clip or re-center the action advantage.  This frozen positive
    # denominator is only a later multiplicative scale for the exact
    # ``Q_plan_3 - V_plan_3`` numerator; its small floor avoids division by
    # zero without changing the critic or target labels.
    denominator = max(rms, 1e-3)
    frequencies = list(structural_counts.values())
    support_summary = {
        "h3_labeled_complete_actions": count,
        "unique_public_selected_action_signatures": len(structural_counts),
        "minimum_signature_count": min(frequencies),
        "maximum_signature_count": max(frequencies),
        "mean_signature_count": sum(frequencies) / len(frequencies),
        "first_stage_legal_count_histogram": {str(key): legal_counts[key] for key in sorted(legal_counts)},
        "factorized_stage_count_histogram": {str(key): stages[key] for key in sorted(stages)},
    }
    artifact = {
        "schema": H3_SCALE_SUPPORT_SCHEMA,
        "owner_goal_revision": SEMANTIC_OWNER_REVISION,
        "required_authority": PRIZE_AUTHORITY_KEY,
        "sidecar_schema": PRIZE_PLAN_V2_SIDECAR_SCHEMA,
        "fit_split": "train",
        "fit_scope": "train_split_only" if full_train_split_consumed else "train_split_prefix_smoke_only",
        "full_train_split_consumed": bool(full_train_split_consumed),
        "validation_or_evaluation_examples_opened_for_fit": False,
        "fit_input": "frozen H3 Q_plan_3(s,a)-V_plan_3(s) predictions on sealed recorded chosen train actions with H3 availability only",
        "frozen_sidecar_checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
            "output_order": list(PRIZE_PLAN_V2_OUTPUT_NAMES),
        },
        "raw_return_value_used_as_fit_target": False,
        "model_target_value_used_as_scale_target": False,
        "terminal_z_or_win_target_used": False,
        "hidden_information_used": False,
        "action_advantage_definition": "A_plan_3=(Q_plan_3(s,a)-V_plan_3(s))/frozen_train_h3_predicted_advantage_rms_floor",
        "unclipped": True,
        "mean_predicted_advantage": mean,
        "stddev_predicted_advantage": stddev,
        "rms_predicted_advantage": rms,
        "minimum_predicted_advantage": advantage_min,
        "maximum_predicted_advantage": advantage_max,
        "frozen_train_h3_predicted_advantage_rms_floor": denominator,
        "support_diagnostic": support_summary,
        "actor_activation": False,
        "actor_integration_required_for_critic_loss": False,
        # A full train pass makes this a candidate *scale* artifact only.  It
        # is not an actor-integration receipt: c3, calibration, ESS/clip,
        # noninterference, rollback, and the clean-boundary gate remain
        # separately mandatory.
        "future_actor_scale_candidate_eligible": bool(full_train_split_consumed),
        "future_actor_integration_eligible": False,
        "future_c3_definition": "not activated; requires separately receipt-bound public confidence/support integration",
        "source_binding": dict(source_binding),
        "source_binding_sha256": _json_sha(dict(source_binding)),
    }
    artifact["artifact_sha256"] = _json_sha(artifact)
    return artifact


def resolve_device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise PrizePlanV2TrainingError("MPS was requested but is unavailable")
    if requested not in {"cpu", "mps"}:
        raise PrizePlanV2TrainingError("Prize-plan-v2 trainer is FP32 CPU/MPS only")
    return torch.device(requested)


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(value)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def open_sealed_inputs(args: argparse.Namespace) -> tuple[Recent20RTPDataset, PrizePlanTargetSet, dict[str, list[str]], dict[str, Any]]:
    production = not bool(getattr(args, "test_mode", False))
    contract_path, contract_sha, revision = load_bound_contract(
        args.contract, expected_sha256=args.contract_sha256, production=production
    )
    training_view = resolve_training_view(
        pointer=args.training_view,
        expected_sha256=args.training_view_sha256,
        production=production,
    )
    dataset = Recent20RTPDataset(
        training_view["overlay_manifest"],
        base_pack_root=training_view["base_pack_root"],
        base_completion_path=training_view["base_completion_path"],
        expected_manifest_sha256=training_view["overlay_manifest_sha256"],
        expected_base_completion_sha256=training_view["base_completion_sha256"],
        verify_overlay_shards=True,
        verify_base_shards=not getattr(args, "test_skip_input_shard_sha256", False),
    )
    target_view_path = getattr(args, "target_view", None)
    direct_target_values = (
        getattr(args, "target_set_root", None),
        getattr(args, "target_manifest", None),
        getattr(args, "target_manifest_sha256", None),
        getattr(args, "target_set_receipt", None),
        getattr(args, "target_set_receipt_sha256", None),
    )
    if target_view_path:
        if production and any(value is not None for value in direct_target_values):
            raise PrizePlanV2TrainingError(
                "production training derives target-set paths only from --target-view; direct target flags are test-only"
            )
        target_view = resolve_target_view(
            pointer=target_view_path,
            expected_sha256=getattr(args, "target_view_sha256", ""),
            contract_sha256=contract_sha,
            contract_goal_revision=revision,
            overlay_manifest_sha256=training_view["overlay_manifest_sha256"],
            production=production,
        )
        target_root = Path(target_view["root"])
        target_manifest = Path(target_view["manifest_path"])
        target_manifest_sha = str(target_view["manifest_sha256"])
        target_receipt = Path(target_view["receipt_path"])
        target_receipt_sha = str(target_view["receipt_sha256"])
    else:
        if production:
            raise PrizePlanV2TrainingError(
                "production training requires --target-view and --target-view-sha256"
            )
        if not all(direct_target_values[1:]):
            raise PrizePlanV2TrainingError("test-only direct target inputs are incomplete")
        target_root = (
            Path(direct_target_values[0]).expanduser().resolve()
            if direct_target_values[0]
            else Path(direct_target_values[1]).expanduser().resolve().parent.parent
        )
        target_manifest = Path(direct_target_values[1])
        target_manifest_sha = str(direct_target_values[2])
        target_receipt = Path(direct_target_values[3])
        target_receipt_sha = str(direct_target_values[4])
        target_view = {
            "test_only_direct_target_input": True,
            "root": str(target_root),
            "manifest_path": str(target_manifest),
            "manifest_sha256": target_manifest_sha,
            "receipt_path": str(target_receipt),
            "receipt_sha256": target_receipt_sha,
        }
    targets = PrizePlanTargetSet(
        root=target_root,
        manifest_path=target_manifest,
        manifest_sha256=target_manifest_sha,
        receipt_path=target_receipt,
        receipt_sha256=target_receipt_sha,
        contract_sha256=contract_sha,
        contract_goal_revision=revision,
        overlay_manifest_sha256=training_view["overlay_manifest_sha256"],
        production=production,
    )
    split_days = assert_split_contract(
        dataset, targets, allow_noncanonical=getattr(args, "test_allow_noncanonical_split", False)
    )
    binding = {
        "contract": {"path": str(contract_path), "sha256": contract_sha, "goal_revision": revision, "semantic_owner_goal_revision": SEMANTIC_OWNER_REVISION, "required_authority": PRIZE_AUTHORITY_KEY},
        "training_view": training_view,
        "target_view": target_view,
        "target_set": {
            "root": str(targets.root),
            "manifest_path": str(targets.manifest_path),
            "manifest_sha256": targets.manifest_sha256,
            "receipt_path": str(targets.receipt_path),
            "receipt_sha256": targets.receipt_sha256,
            "target_value_transform_sha256": targets.transform_sha256,
            "gamma": targets.gamma,
        },
        "sidecar": {"schema": PRIZE_PLAN_V2_SIDECAR_SCHEMA, "horizons": list(PRIZE_PLAN_V2_HORIZONS)},
        "exact_join_identity": list(IDENTITY_FIELDS),
        "target_consumption": "model_target_value_only_raw_return_verified_not_used_as_label_no_clipping",
        "runtime_or_actor_activation": False,
    }
    return dataset, targets, split_days, binding


def _build_model(*, hidden_width: int) -> tuple[PrizePlanV2Sidecar, dict[str, Any]]:
    if hidden_width < 1:
        raise PrizePlanV2TrainingError("hidden width must be positive")
    config = PrizePlanV2SidecarConfig(
        feature_dim=FEATURE_WIDTH,
        state_hidden_dim=hidden_width,
        action_hidden_dim=hidden_width,
        q_hidden_dim=hidden_width,
        max_action_stages=MAX_STAGES,
        max_legal_options=MAX_LEGAL,
        max_action_program_tokens=MAX_TOKENS,
        max_action_token_value=MAX_TOKEN_VALUE,
    )
    return PrizePlanV2Sidecar(config), asdict(config)


def _checkpoint_training_state(
    *,
    epoch_completed: int,
    optimizer_steps: int,
    trainer_config: Mapping[str, Any],
    split_days: Mapping[str, Sequence[str]],
    epoch_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "trainer_schema": TRAINER_SCHEMA,
        "saved_at_unix_seconds": time.time(),
        "epoch_completed": int(epoch_completed),
        "optimizer_steps": int(optimizer_steps),
        "trainer_config": dict(trainer_config),
        "split_days": {key: list(value) for key, value in split_days.items()},
        "epoch_history": [dict(item) for item in epoch_history],
        # The H3 A_plan scale is fit only after this frozen sidecar state has
        # been checkpointed.  Do not bind an input-label proxy into an
        # in-progress checkpoint.
        "h3_scale_support_post_training_only": True,
        "evaluation_split_consumed": False,
        "actor_activation": False,
    }


def _checkpoint_metadata(source_binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_binding": dict(source_binding),
        "source_binding_sha256": _json_sha(dict(source_binding)),
        "runtime_or_policy_attachment": False,
        "policy_model_state_dict_changed": False,
        "runtime_critic_calls": False,
        "search_rtp_mcts_or_simulator_branching_used": False,
        "replay_weights_changed": False,
        "actor_activation": False,
    }


def load_resume_checkpoint(
    path: Path | str,
    *,
    model: PrizePlanV2Sidecar,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    source_binding: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Restore sidecar and optimizer only after strict source/config equality."""

    checkpoint = _regular_file(path, label="Prize-plan-v2 resume checkpoint")
    try:
        restored = restore_prize_plan_v2_checkpoint(model, checkpoint, optimizer=optimizer)
    except (PrizePlanV2SidecarError, OSError, RuntimeError, ValueError) as exc:
        raise PrizePlanV2TrainingError("Prize-plan-v2 resume checkpoint is incompatible") from exc
    metadata = _mapping(restored.metadata, label="resume checkpoint metadata")
    state = _mapping(restored.training_state, label="resume checkpoint training state")
    if (
        dict(metadata.get("source_binding") or {}) != dict(source_binding)
        or metadata.get("source_binding_sha256") != _json_sha(dict(source_binding))
        or metadata.get("actor_activation") is not False
        or state.get("h3_scale_support_post_training_only") is not True
        or state.get("actor_activation") is not False
    ):
        raise PrizePlanV2TrainingError("resume checkpoint source/activation binding mismatch")
    saved_config = _mapping(state.get("trainer_config"), label="resume trainer config")
    for key in (
        "batch_size",
        "shuffle_buffer",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "seed",
        "max_train_programs",
        "max_validation_programs",
    ):
        if saved_config.get(key) != trainer_config.get(key):
            raise PrizePlanV2TrainingError(f"resume trainer config mismatch: {key}")
    history = state.get("epoch_history")
    if not isinstance(history, list) or not all(isinstance(item, Mapping) for item in history):
        raise PrizePlanV2TrainingError("resume checkpoint epoch history is malformed")
    completed = _exact_int(state.get("epoch_completed"), label="resume epoch", minimum=0)
    steps = _exact_int(state.get("optimizer_steps"), label="resume optimizer steps", minimum=0)
    model.to(device=device, dtype=torch.float32)
    _move_optimizer_state(optimizer, device)
    return completed, steps, [dict(item) for item in history]


def _prepare_output_dir(path: Path | str, *, resume: Path | str | None) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise PrizePlanV2TrainingError("output directory must be a real directory")
        if not resume and any(output.iterdir()):
            # A nonempty run root without an explicit resume path is too easy
            # to mistake for a fresh production run.
            raise PrizePlanV2TrainingError("nonempty output directory requires --resume")
        return output
    output.mkdir(parents=True, mode=0o755)
    return _regular_directory(output, label="output directory")


def _fit_and_write_h3_scale_support(
    *,
    output: Path,
    dataset: Recent20RTPDataset,
    targets: PrizePlanTargetSet,
    model: PrizePlanV2Sidecar,
    device: torch.device,
    batch_size: int,
    sidecar_checkpoint: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    max_train_programs: int,
) -> tuple[Path, str, dict[str, Any]]:
    """Seal the post-training H3 Q-minus-V diagnostic for this exact sidecar."""

    path = output / "h3-scale-support.json"
    full_train_split = max_train_programs == 0
    artifact = fit_h3_scale_support(
        model,
        iter_complete_action_examples(
            dataset, targets, split="train", max_programs=max_train_programs
        ),
        device=device,
        batch_size=batch_size,
        source_binding=source_binding,
        sidecar_checkpoint=sidecar_checkpoint,
        full_train_split_consumed=full_train_split,
    )
    digest = atomic_write_json(path, artifact)
    return path, digest, artifact


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run offline sidecar training; it cannot activate an actor or runtime."""

    started = time.time()
    if args.epochs < 1 or args.batch_size < 1 or args.shuffle_buffer < 1:
        raise PrizePlanV2TrainingError("epochs, batch size, and shuffle buffer must be positive")
    if args.max_train_programs < 0 or args.max_validation_programs < 0:
        raise PrizePlanV2TrainingError("program limits cannot be negative")
    for name in ("learning_rate", "weight_decay", "grad_clip"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise PrizePlanV2TrainingError(f"{name} is invalid")
    dataset, targets, split_days, binding = open_sealed_inputs(args)
    output = _prepare_output_dir(args.output_dir, resume=args.resume)
    device = resolve_device(args.device)
    _seed(int(args.seed))
    trainer_config = {
        "schema": TRAINER_SCHEMA,
        "device": str(device),
        "fp32": True,
        "batch_size": int(args.batch_size),
        "shuffle_buffer": int(args.shuffle_buffer),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "max_train_programs": int(args.max_train_programs),
        "max_validation_programs": int(args.max_validation_programs),
        "complete_recorded_chosen_action_only": True,
        "bounded_stream_shuffle_only": True,
        "replay_weights_changed": False,
        "actor_activation": False,
        "evaluation_split_consumed": False,
    }
    resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume_path is not None:
        loaded = load_prize_plan_v2_checkpoint(resume_path, device="cpu")
        model = PrizePlanV2Sidecar(loaded.model.config)
        model_config = asdict(loaded.model.config)
    else:
        model, model_config = _build_model(hidden_width=int(args.hidden_width))
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    completed = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    if resume_path is not None:
        completed, optimizer_steps, history = load_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            source_binding=binding,
            trainer_config=trainer_config,
        )
    if completed > args.epochs:
        raise PrizePlanV2TrainingError("resume checkpoint epoch exceeds requested epochs")
    for epoch in range(completed + 1, int(args.epochs) + 1):
        train_rows = bounded_shuffle(
            iter_complete_action_examples(
                dataset, targets, split="train", max_programs=int(args.max_train_programs)
            ),
            buffer_size=int(args.shuffle_buffer),
            seed=int(args.seed) + epoch,
        )
        train_metrics, steps = run_batches(
            model,
            train_rows,
            device=device,
            batch_size=int(args.batch_size),
            optimizer=optimizer,
            grad_clip=float(args.grad_clip),
        )
        optimizer_steps += steps
        validation_metrics, _ = run_batches(
            model,
            iter_complete_action_examples(
                dataset,
                targets,
                split="validation",
                max_programs=int(args.max_validation_programs),
            ),
            device=device,
            batch_size=int(args.batch_size),
            optimizer=None,
            grad_clip=0.0,
        )
        summary = {
            "epoch": epoch,
            "optimizer_steps_this_epoch": steps,
            "optimizer_steps_total": optimizer_steps,
            "train": train_metrics.summary(),
            "validation": validation_metrics.summary(),
        }
        history.append(summary)
        state = _checkpoint_training_state(
            epoch_completed=epoch,
            optimizer_steps=optimizer_steps,
            trainer_config=trainer_config,
            split_days=split_days,
            epoch_history=history,
        )
        save_prize_plan_v2_checkpoint(
            output / "latest.pt", model, optimizer=optimizer, training_state=state, metadata=_checkpoint_metadata(binding)
        )
        atomic_write_json(
            output / "latest.metrics.json",
            {
                "schema": TRAINING_RECEIPT_SCHEMA,
                "status": "in_progress_offline_sidecar_only",
                "source_binding": binding,
                "source_binding_sha256": _json_sha(binding),
                "model_config": model_config,
                "h3_scale_support": "post_training_frozen_sidecar_diagnostic_pending",
                "summary": summary,
                "actor_activation": False,
            },
        )
        print(json.dumps({"phase": "epoch", **summary}, sort_keys=True), flush=True)
    state = _checkpoint_training_state(
        epoch_completed=int(args.epochs),
        optimizer_steps=optimizer_steps,
        trainer_config=trainer_config,
        split_days=split_days,
        epoch_history=history,
    )
    final_checkpoint = output / "prize-plan-v2-sidecar.pt"
    save_prize_plan_v2_checkpoint(
        final_checkpoint, model, optimizer=optimizer, training_state=state, metadata=_checkpoint_metadata(binding)
    )
    checkpoint_sha = sha256_file(final_checkpoint)
    checkpoint_size = final_checkpoint.stat().st_size
    h3_path, h3_sha, h3_artifact = _fit_and_write_h3_scale_support(
        output=output,
        dataset=dataset,
        targets=targets,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        sidecar_checkpoint={"path": str(final_checkpoint), "sha256": checkpoint_sha},
        source_binding=binding,
        max_train_programs=int(args.max_train_programs),
    )
    validation = history[-1]["validation"] if history else None
    receipt = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "status": "completed_offline_sidecar_only_not_actor_activated",
        "source_binding": binding,
        "source_binding_sha256": _json_sha(binding),
        "split_days": split_days,
        "model_config": model_config,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainer_config": trainer_config,
        "epochs": int(args.epochs),
        "optimizer_steps": optimizer_steps,
        "epoch_history": history,
        "validation": validation,
        "h3_scale_support": {
            "path": str(h3_path),
            "sha256": h3_sha,
            "artifact_sha256": h3_artifact["artifact_sha256"],
            "fit_split": "train",
            "full_train_split_consumed": h3_artifact["full_train_split_consumed"],
            "future_actor_integration_eligible": h3_artifact["future_actor_integration_eligible"],
        },
        "checkpoint_path": str(final_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_size,
        "latest_checkpoint_path": str(output / "latest.pt"),
        "checkpoint_contains_policy_state_dict": False,
        "runtime_or_policy_attachment": False,
        "existing_value_head_changed": False,
        "existing_action_q_head_changed": False,
        "runtime_critic_calls": False,
        "search_rtp_mcts_or_simulator_branching_used": False,
        "replay_weights_changed": False,
        "evaluation_split_consumed": False,
        "actor_activation": False,
        "activation_eligible": False,
        "elapsed_seconds": time.time() - started,
        "completed_at_unix_seconds": time.time(),
    }
    receipt_path = output / "training-receipt.json"
    receipt_sha = atomic_write_json(receipt_path, receipt)
    result = {
        "checkpoint_path": str(final_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "h3_scale_support_path": str(h3_path),
        "h3_scale_support_sha256": h3_sha,
        "optimizer_steps": optimizer_steps,
        "validation": validation,
        "actor_activation": False,
    }
    print(json.dumps({"phase": "complete", **result}, sort_keys=True), flush=True)
    return result


def validate_checkpoint(
    args: argparse.Namespace,
    *,
    output_receipt: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate one strict sidecar checkpoint on validation only (no tuning)."""

    dataset, targets, split_days, binding = open_sealed_inputs(args)
    checkpoint = _regular_file(args.checkpoint, label="Prize-plan-v2 checkpoint")
    try:
        loaded = load_prize_plan_v2_checkpoint(checkpoint, device=resolve_device(args.device))
    except (PrizePlanV2SidecarError, OSError, RuntimeError, ValueError) as exc:
        raise PrizePlanV2TrainingError("cannot load strict Prize-plan-v2 sidecar checkpoint") from exc
    metadata = _mapping(loaded.metadata, label="checkpoint metadata")
    if (
        dict(metadata.get("source_binding") or {}) != binding
        or metadata.get("source_binding_sha256") != _json_sha(binding)
        or metadata.get("actor_activation") is not False
    ):
        raise PrizePlanV2TrainingError("checkpoint source/activation binding mismatch")
    device = resolve_device(args.device)
    model = loaded.model.to(device=device, dtype=torch.float32)
    metrics, _ = run_batches(
        model,
        iter_complete_action_examples(
            dataset, targets, split="validation", max_programs=int(args.max_validation_programs)
        ),
        device=device,
        batch_size=int(args.batch_size),
        optimizer=None,
        grad_clip=0.0,
    )
    result = {
        "schema": VALIDATION_RECEIPT_SCHEMA,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_binding": binding,
        "source_binding_sha256": _json_sha(binding),
        "split_days": split_days,
        "validation": metrics.summary(),
        "max_validation_programs": int(args.max_validation_programs),
        "full_validation_split_consumed": not bool(args.max_validation_programs),
        "evaluation_split_consumed": False,
        "actor_activation": False,
        "activation_eligible": False,
    }
    if output_receipt is not None:
        receipt = Path(output_receipt).expanduser().resolve()
        result["receipt_path"] = str(receipt)
        result["receipt_sha256"] = atomic_write_json(receipt, result)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--training-view-sha256", required=True)
    parser.add_argument("--target-view", type=Path, default=None)
    parser.add_argument("--target-view-sha256", default="")
    # Direct target artifacts are intentionally hidden test-fixture escapes.
    # Production input resolution derives all of these from the target-view.
    parser.add_argument("--target-set-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-manifest", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-manifest-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-set-receipt", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-set-receipt-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--contract", type=Path, default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json")
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23023)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-train-programs", type=int, default=0)
    parser.add_argument("--max-validation-programs", type=int, default=0)
    # Hermetic tests only; production invocation must not use either escape.
    parser.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-allow-noncanonical-split", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-skip-input-shard-sha256", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        train(args)
    except (PrizePlanV2TrainingError, Recent20OverlayError, PrizePlanV2SidecarError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
