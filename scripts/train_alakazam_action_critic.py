#!/usr/bin/env python3
"""Train the isolated Alakazam complete-action critic sidecar.

This is deliberately an offline, trainer-only program.  It streams a sealed
40-wide base pack, the sealed complete-action overlay, and a separate sealed
target overlay one day at a time.  It never imports the policy runtime, calls
search, creates counterfactual labels, or changes replay sampling weights.

The critic is trained on *complete recorded chosen actions*.  Factorized
stages are inputs to one complete-action example, never independent TD
transitions.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.recent20_overlay import (  # noqa: E402
    Recent20OverlayError,
    Recent20RTPDataset,
    canonical_bytes,
    sha256_file,
)


# These imports intentionally remain isolated from any policy/runtime module.
# The target builder and sidecar land independently, so the loader below gives
# a useful error if a caller runs this script before both artifacts are present.
try:  # noqa: E402
    import poke_bot.action_critic_sidecar as action_critic_sidecar
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI users
    if exc.name != "poke_bot.action_critic_sidecar":
        raise
    action_critic_sidecar = None  # type: ignore[assignment]

try:  # noqa: E402
    import poke_bot.alakazam_action_critic_targets as action_critic_targets
except ModuleNotFoundError as exc:  # pragma: no cover - alternate module name
    if exc.name != "poke_bot.alakazam_action_critic_targets":
        raise
    action_critic_targets = None  # type: ignore[assignment]


TRAINER_SCHEMA = "poke_bot.alakazam_action_critic_trainer/v1"
CHECKPOINT_SCHEMA = "poke_bot.alakazam_action_critic_checkpoint/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_training_receipt/v1"
VALIDATION_SCHEMA = "poke_bot.alakazam_action_critic_validation_receipt/v1"
TRAINING_VIEW_SCHEMA = "poke_bot.alakazam_action_critic_training_view/v1"
TRAINING_VIEW_COMPLETION_SCHEMA = (
    "poke_bot.alakazam_action_critic_training_view_completion/v1"
)
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_set_manifest/v1"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_set_receipt/v1"
TARGET_DAY_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_day_manifest/v1"
TARGET_DAY_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_day_receipt/v1"
TARGET_OVERLAY_SCHEMA = "poke_bot.alakazam_action_critic_target_overlay/v1"
TARGET_VIEW_SCHEMA = "poke_bot.alakazam_action_critic_target_view/v1"
TARGET_VIEW_COMPLETION_SCHEMA = (
    "poke_bot.alakazam_action_critic_target_transfer_completion/v1"
)
TARGET_VIEW_FILE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_action_critic_target_file_transfer_receipt/v1"
)
CRITIC_SEMANTIC_OWNER_REVISION = 21
CRITIC_AUTHORITY_KEY = "revision_21_draw_safe_critic_actor_canary"
FEATURE_WIDTH = 40
# These are the rounded sealed recent-20 maxima.  The overlay audit observed
# at most 28 factorized stages, 55 legal options, and raw action tokens <=52.
# The critic has one fixed 32/64/32/63 ABI instead of dynamic shape growth.
MAX_SELECTED_ACTION_STAGES = 32
MAX_SELECTED_LEGAL_OPTIONS = 64
MAX_SELECTED_ACTION_PROGRAM_TOKENS = 32
MAX_SELECTED_ACTION_TOKEN_VALUE = 63
OUTPUT_NAMES = (
    "V_win",
    "Q_win",
    "V_prize^1",
    "Q_prize^1",
    "V_prize^2",
    "Q_prize^2",
    "V_prize^3",
    "Q_prize^3",
)
OUTPUT_INDEX = {name: index for index, name in enumerate(OUTPUT_NAMES)}
EXPECTED_TRAIN_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 6)]
)
EXPECTED_VALIDATION_DAYS = ("2026-08-06", "2026-08-07", "2026-08-08")
EXPECTED_EVALUATION_DAYS = ("2026-08-09", "2026-08-10", "2026-08-11")
EPSILON = 1.0e-6


def _expected_split_for_day(day: str) -> str:
    if day in EXPECTED_TRAIN_DAYS:
        return "train"
    if day in EXPECTED_VALIDATION_DAYS:
        return "validation"
    if day in EXPECTED_EVALUATION_DAYS:
        return "evaluation"
    raise ActionCriticTrainingError("target day is outside the sealed recent-20 window")


class ActionCriticTrainingError(RuntimeError):
    """A sealed-input, target, alignment, or sidecar invariant failed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return canonical_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: str, *, label: str) -> str:
    text = str(value)
    raw = text.removeprefix("sha256:")
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise ActionCriticTrainingError(f"{label} must be a lowercase SHA-256")
    return "sha256:" + raw


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionCriticTrainingError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ActionCriticTrainingError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_write_bytes(path: Path, body: bytes, *, mode: int = 0o644) -> None:
    """Atomically replace one local trainer artifact and fsync its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        written = 0
        while written < len(body):
            written += os.write(fd, body[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    body = _canonical_json_bytes(dict(value))
    _atomic_write_bytes(path, body)
    return _sha256_bytes(body)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _optional_target_schema_module() -> Any | None:
    """Load the target schema module without binding to an implementation name.

    The first name is the canonical sidecar target module.  The aliases are
    intentionally compatibility-only so a sealed target manifest remains the
    authority, not a trainer-side guess about a module filename.
    """
    global action_critic_targets
    if action_critic_targets is not None:
        return action_critic_targets
    for name in (
        "poke_bot.alakazam_action_critic_targets",
        "poke_bot.action_critic_targets",
        "poke_bot.alakazam_action_critic_target_overlay",
    ):
        try:
            action_critic_targets = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            continue
        return action_critic_targets
    return None


def _require_sidecar_module() -> Any:
    global action_critic_sidecar
    if action_critic_sidecar is None:
        try:
            action_critic_sidecar = importlib.import_module(
                "poke_bot.action_critic_sidecar"
            )
        except ModuleNotFoundError as exc:
            if exc.name == "poke_bot.action_critic_sidecar":
                raise ActionCriticTrainingError(
                    "poke_bot.action_critic_sidecar is required before critic training"
                ) from exc
            raise
    current = ("ActionCriticSidecarConfig", "ActionCriticSidecar")
    compatibility = ("ActionCriticConfig", "CompleteActionCritic")
    if not all(hasattr(action_critic_sidecar, name) for name in current) and not all(
        hasattr(action_critic_sidecar, name) for name in compatibility
    ):
        raise ActionCriticTrainingError(
            "action critic sidecar lacks the complete-action critic interface"
        )
    return action_critic_sidecar


def _module_schema_values(module: Any, *, kind: str) -> set[str]:
    if module is None:
        return set()
    names = (
        (
            "TARGET_SET_MANIFEST_SCHEMA",
            "TARGET_MANIFEST_SCHEMA",
            "TARGET_DAY_MANIFEST_SCHEMA",
            "MANIFEST_SCHEMA",
            "ACTION_CRITIC_TARGET_MANIFEST_SCHEMA",
        )
        if kind == "manifest"
        else (
            "TARGET_OVERLAY_SCHEMA",
            "TARGET_ROW_SCHEMA",
            "ROW_SCHEMA",
            "ACTION_CRITIC_TARGET_SCHEMA",
        )
    )
    values: set[str] = set()
    for name in names:
        value = getattr(module, name, "")
        if isinstance(value, str) and value:
            values.add(value)
    return values


def _call_schema_validator(module: Any | None, row: Mapping[str, Any]) -> None:
    if module is None:
        return
    for name in (
        "validate_target_row",
        "validate_action_critic_target_row",
        "validate_target_overlay_row",
    ):
        validator = getattr(module, name, None)
        if not callable(validator):
            continue
        result = validator(dict(row))
        if result is False:
            raise ActionCriticTrainingError("target schema validator rejected row")
        return


def _resolve_object_path(manifest_path: Path, declared: str) -> Path:
    path = Path(str(declared))
    if path.is_absolute():
        return path
    root = (
        manifest_path.parent.parent
        if manifest_path.parent.name == "manifests"
        else manifest_path.parent
    )
    return (root / path).resolve()


def _bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ActionCriticTrainingError(f"{label} must be boolean")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionCriticTrainingError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ActionCriticTrainingError(f"{label} must be finite")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionCriticTrainingError(f"{label} must be an object")
    return value


def _field(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ActionCriticTrainingError(f"{label} must be a regular file")
    return raw.resolve()


def _regular_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ActionCriticTrainingError(f"{label} must be a real directory")
    return raw.resolve()


def _relative_member(root: Path, declared: Any, *, label: str) -> Path:
    """Resolve a required relative path without accepting links or escapes."""

    if not isinstance(declared, str) or not declared:
        raise ActionCriticTrainingError(f"{label} path is absent")
    relative = Path(declared)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ActionCriticTrainingError(f"{label} must be relative to its sealed root")
    root = _regular_directory(root, label=f"{label} root")
    candidate = root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        candidate = candidate / part
        if candidate.is_symlink():
            raise ActionCriticTrainingError(f"{label} path contains a symbolic link")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ActionCriticTrainingError(f"{label} escaped its sealed root") from exc
    return resolved


def _manifest_root(path: Path) -> Path:
    """Return the sealed artifact root for a manifest in ``manifests/``."""

    return path.parent.parent if path.parent.name == "manifests" else path.parent


def _sha_matches(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(_regular_file(path, label=label))
    if actual != _require_sha256(expected, label=f"{label} SHA-256"):
        raise ActionCriticTrainingError(f"{label} digest mismatch")
    return actual


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionCriticTrainingError(f"{label} must be an integer")
    return int(value)


def _validate_embedded_critic_authority(contract: Mapping[str, Any]) -> int:
    """Bind the r21 critic ABI without pinning unrelated contract revisions.

    The top-level derivative contract can legitimately advance (for example,
    for fleet staging).  This trainer owns only the embedded r21 critic
    semantics, so it records both identities and rejects a wrapper that has
    modified the critic ABI underneath a newer top-level revision.
    """

    current_revision = _exact_int(
        contract.get("goal_revision"), label="goal contract revision"
    )
    if current_revision < CRITIC_SEMANTIC_OWNER_REVISION:
        raise ActionCriticTrainingError("goal contract predates the critic semantics")
    authority = _mapping(
        contract.get(CRITIC_AUTHORITY_KEY), label="revision-21 critic authority"
    )
    if authority.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION:
        raise ActionCriticTrainingError("embedded critic semantic owner revision drifted")
    target_overlay = _mapping(
        authority.get("target_overlay"), label="revision-21 target-overlay authority"
    )
    if (
        target_overlay.get("schema") != TARGET_OVERLAY_SCHEMA
        or target_overlay.get("manifest_schema") != TARGET_SET_MANIFEST_SCHEMA
        or target_overlay.get("row_join_identity")
        != [
            "utc_day",
            "source_archive_sha256",
            "source_member",
            "episode_id",
            "acting_seat",
            "env_step",
            "program_identity",
        ]
        or target_overlay.get("required_terminal_fields")
        != [
            "z",
            "z_mask",
            "win_target_one_only_for_z_plus1",
            "win_target_mask",
        ]
        or target_overlay.get("hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed")
        is not False
    ):
        raise ActionCriticTrainingError("embedded revision-21 target-overlay semantics drifted")
    return current_revision


def _load_bound_contract(
    path: Path | str,
    *,
    expected_sha256: str,
    production: bool,
) -> tuple[Path, str, int]:
    contract = _regular_file(path, label="goal contract")
    actual_sha = sha256_file(contract)
    if production and not expected_sha256:
        raise ActionCriticTrainingError("production critic training requires --contract-sha256")
    if expected_sha256 and actual_sha != _require_sha256(
        expected_sha256, label="contract SHA-256"
    ):
        raise ActionCriticTrainingError("contract digest mismatch")
    document = _read_json(contract, label="goal contract")
    if production:
        return contract, actual_sha, _validate_embedded_critic_authority(document)
    # Hermetic fixtures exercise the data-flow mechanics without pretending to
    # be a production authority document.  They must still be explicitly
    # marked test-only by the caller.
    revision = document.get("goal_revision", CRITIC_SEMANTIC_OWNER_REVISION)
    return contract, actual_sha, _exact_int(revision, label="test contract revision")


def _training_view_root(pointer_path: Path) -> Path:
    """Return the exact transferred-view root enclosing a receipt pointer."""

    if (
        pointer_path.parent.name != "training-view"
        or pointer_path.parent.parent.name != "transfer"
    ):
        raise ActionCriticTrainingError(
            "training-view pointer must live under transfer/training-view"
        )
    return _regular_directory(
        pointer_path.parent.parent.parent, label="transferred training-view root"
    )


def _receipt_named_digest(path: Path, *, label: str) -> str:
    """Hash an immutable receipt and require its content-addressed filename."""

    digest = sha256_file(_regular_file(path, label=label))
    expected_prefix = "sha256-" + digest.removeprefix("sha256:")
    if not path.name.startswith(expected_prefix):
        raise ActionCriticTrainingError(f"{label} filename is not content-addressed")
    return digest


def _resolve_transferred_training_view(
    *,
    pointer: Path | str,
    expected_sha256: str,
    production: bool,
) -> dict[str, Any]:
    """Resolve the portable base/overlay paths from a transfer receipt chain."""

    pointer_path = _regular_file(pointer, label="training-view pointer")
    pointer_sha = _receipt_named_digest(pointer_path, label="training-view pointer")
    if production and not expected_sha256:
        raise ActionCriticTrainingError("production critic training requires --training-view-sha256")
    if expected_sha256 and pointer_sha != _require_sha256(
        expected_sha256, label="training-view SHA-256"
    ):
        raise ActionCriticTrainingError("training-view pointer digest mismatch")
    view = _read_json(pointer_path, label="training-view pointer")
    if view.get("schema") != TRAINING_VIEW_SCHEMA:
        raise ActionCriticTrainingError("training-view pointer schema drifted")
    if (
        view.get("canonical_manifest_remains_byte_identical") is not True
        or view.get("base_completion_path_override_required") is not True
        or view.get("runtime_or_training_activation_authority") is not False
    ):
        raise ActionCriticTrainingError("training-view portability boundary drifted")
    root = _training_view_root(pointer_path)
    overlay = _mapping(view.get("canonical_overlay_manifest"), label="training-view overlay")
    base = _mapping(view.get("canonical_base_completion"), label="training-view base completion")
    overlay_path = _relative_member(
        root, overlay.get("relative_path"), label="training-view overlay manifest"
    )
    base_completion_path = _relative_member(
        root, base.get("relative_path"), label="training-view base completion"
    )
    base_root = _relative_member(
        root, view.get("base_pack_root_relative"), label="training-view base root"
    )
    overlay_root = _relative_member(
        root, view.get("overlay_root_relative"), label="training-view overlay root"
    )
    if not base_root.is_dir() or not overlay_root.is_dir():
        raise ActionCriticTrainingError("training-view base or overlay root is not a directory")
    overlay_sha = _sha_matches(
        overlay_path, str(overlay.get("sha256") or ""), label="training-view overlay manifest"
    )
    base_completion_sha = _sha_matches(
        base_completion_path,
        str(base.get("sha256") or ""),
        label="training-view base completion",
    )
    plan_sha = _require_sha256(
        str(view.get("transfer_plan_sha256") or ""), label="training-view transfer plan SHA-256"
    )
    completion_root = _relative_member(
        root, "transfer/completion", label="training-view completion directory"
    )
    if not completion_root.is_dir():
        raise ActionCriticTrainingError("training-view completion directory is absent")
    matches: list[tuple[Path, str, Mapping[str, Any]]] = []
    for candidate in sorted(completion_root.iterdir()):
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".json":
            continue
        completion_sha = _receipt_named_digest(candidate, label="training-view completion receipt")
        completion = _read_json(candidate, label="training-view completion receipt")
        if (
            completion.get("schema") == TRAINING_VIEW_COMPLETION_SCHEMA
            and completion.get("training_view_sha256") == pointer_sha
            and completion.get("training_view_path")
            == str(pointer_path.relative_to(root))
        ):
            matches.append((candidate, completion_sha, completion))
    if len(matches) != 1:
        raise ActionCriticTrainingError(
            "training-view pointer must have exactly one matching completion receipt"
        )
    completion_path, completion_sha, completion = matches[0]
    source = _mapping(completion.get("source"), label="training-view completion source")
    if (
        completion.get("transfer_plan_sha256") != plan_sha
        or source.get("base_completion_sha256") != base_completion_sha
        or source.get("overlay_manifest_sha256") != overlay_sha
        or completion.get("all_source_destination_sha256_size_verified") is not True
        or completion.get("private_partials_not_training_eligible") is not True
        or completion.get("canonical_overlay_manifest_rewritten") is not False
        or completion.get("runtime_or_trainer_activation_performed") is not False
    ):
        raise ActionCriticTrainingError("training-view completion receipt binding drifted")
    return {
        "pointer_path": pointer_path,
        "pointer_sha256": pointer_sha,
        "completion_path": completion_path,
        "completion_sha256": completion_sha,
        "root": root,
        "overlay_manifest": overlay_path,
        "overlay_manifest_sha256": overlay_sha,
        "base_pack_root": base_root,
        "base_completion_path": base_completion_path,
        "base_completion_sha256": base_completion_sha,
    }


def _target_view_root(pointer_path: Path) -> Path:
    if (
        pointer_path.parent.name != "target-view"
        or pointer_path.parent.parent.name != "transfer"
    ):
        raise ActionCriticTrainingError(
            "target-view pointer must live under transfer/target-view"
        )
    return _regular_directory(
        pointer_path.parent.parent.parent, label="transferred target-view root"
    )


def _resolve_transferred_target_view(
    *,
    pointer: Path | str,
    expected_sha256: str,
    expected_contract_sha256: str,
    expected_base_completion_sha256: str,
    expected_overlay_manifest_sha256: str,
    production: bool,
) -> dict[str, Any]:
    """Resolve and verify a portable target-only transfer view on Bert.

    The copied target-set manifest remains byte-identical.  The view pointer
    is the relocation layer, and its completion/per-object receipts prove that
    every active target artifact is a local full-SHA copy rather than an Elmo
    absolute path that happened to exist on the trainer host.
    """

    pointer_path = _regular_file(pointer, label="target-view pointer")
    pointer_sha = _receipt_named_digest(pointer_path, label="target-view pointer")
    if production and not expected_sha256:
        raise ActionCriticTrainingError("production target input requires --target-view-sha256")
    if expected_sha256 and pointer_sha != _require_sha256(
        expected_sha256, label="target-view SHA-256"
    ):
        raise ActionCriticTrainingError("target-view pointer digest mismatch")
    view = _read_json(pointer_path, label="target-view pointer")
    if (
        view.get("schema") != TARGET_VIEW_SCHEMA
        or view.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION
        or view.get("status") != "verified_target_only_offline_input"
        or view.get("raw_episode_zip_transferred") is not False
        or view.get("runtime_or_training_started") is not False
        or view.get("create_only") is not True
    ):
        raise ActionCriticTrainingError("target-view authority or information boundary drifted")
    root = _target_view_root(pointer_path)
    target_set_root = _relative_member(
        root, view.get("target_set_root_relative"), label="target-view target-set root"
    )
    if not target_set_root.is_dir():
        raise ActionCriticTrainingError("target-view target-set root is not a directory")
    manifest = _mapping(
        view.get("canonical_target_set_manifest"), label="target-view target-set manifest"
    )
    aggregate_receipt = _mapping(
        view.get("canonical_target_set_receipt"), label="target-view target-set receipt"
    )
    if manifest.get("remains_byte_identical") is not True:
        raise ActionCriticTrainingError("target-view rewrites the canonical target-set manifest")
    manifest_path = _relative_member(
        root, manifest.get("relative_path"), label="target-view target-set manifest"
    )
    receipt_path = _relative_member(
        root, aggregate_receipt.get("relative_path"), label="target-view target-set receipt"
    )
    manifest_sha = _sha_matches(
        manifest_path, str(manifest.get("sha256") or ""), label="target-view target-set manifest"
    )
    receipt_sha = _sha_matches(
        receipt_path,
        str(aggregate_receipt.get("sha256") or ""),
        label="target-view target-set receipt",
    )
    try:
        manifest_relative_to_set = str(manifest_path.relative_to(target_set_root))
        receipt_relative_to_set = str(receipt_path.relative_to(target_set_root))
    except ValueError as exc:
        raise ActionCriticTrainingError("target-view canonical objects escaped target-set root") from exc
    aggregate = _read_json(receipt_path, label="target-view target-set receipt")
    if (
        aggregate.get("schema") != TARGET_SET_RECEIPT_SCHEMA
        or aggregate.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION
        or aggregate.get("target_set_manifest_path") != manifest_relative_to_set
        or aggregate.get("target_set_manifest_sha256") != manifest_sha
        or aggregate.get("goal_contract_sha256") != expected_contract_sha256
        or aggregate.get("base_pack_completion_sha256") != expected_base_completion_sha256
        or aggregate.get("complete_action_overlay_manifest_sha256")
        != expected_overlay_manifest_sha256
        or aggregate.get("day_count") != 20
        or aggregate.get("critic_semantic_owner_goal_revision")
        != CRITIC_SEMANTIC_OWNER_REVISION
        or aggregate.get("required_critic_authority") != CRITIC_AUTHORITY_KEY
        or aggregate.get("episode_and_seat_group_split_disjoint") is not True
    ):
        raise ActionCriticTrainingError("target-view target-set receipt binding drifted")
    source = _mapping(view.get("source_binding"), label="target-view source binding")
    if (
        source.get("goal_contract_sha256") != expected_contract_sha256
        or source.get("base_pack_completion_sha256") != expected_base_completion_sha256
        or source.get("complete_action_overlay_manifest_sha256")
        != expected_overlay_manifest_sha256
    ):
        raise ActionCriticTrainingError("target-view source identities drifted")
    raw_inventory = source.get("all_20_raw_episode_zip_sha256s")
    if not isinstance(raw_inventory, list) or len(raw_inventory) != 20:
        raise ActionCriticTrainingError("target-view raw ZIP identity inventory is incomplete")

    plan_path = _relative_member(root, view.get("plan_relative"), label="target-view plan")
    plan_sha = _receipt_named_digest(plan_path, label="target-view plan")
    if plan_sha != _require_sha256(str(view.get("plan_sha256") or ""), label="target-view plan SHA-256"):
        raise ActionCriticTrainingError("target-view plan binding drifted")
    completion_root = _relative_member(
        root, "transfer/completion", label="target-view completion directory"
    )
    if not completion_root.is_dir():
        raise ActionCriticTrainingError("target-view completion directory is absent")
    matching_completions: list[tuple[Path, str, Mapping[str, Any]]] = []
    pointer_relative = str(pointer_path.relative_to(root))
    for candidate in sorted(completion_root.iterdir()):
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".json":
            continue
        completion_sha = _receipt_named_digest(candidate, label="target-view completion receipt")
        completion = _read_json(candidate, label="target-view completion receipt")
        if (
            completion.get("schema") == TARGET_VIEW_COMPLETION_SCHEMA
            and completion.get("target_view_relative") == pointer_relative
            and completion.get("target_view_sha256") == pointer_sha
        ):
            matching_completions.append((candidate, completion_sha, completion))
    if len(matching_completions) != 1:
        raise ActionCriticTrainingError(
            "target-view pointer must have exactly one matching completion receipt"
        )
    completion_path, completion_sha, completion = matching_completions[0]
    if (
        completion.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION
        or completion.get("plan_relative") != str(plan_path.relative_to(root))
        or completion.get("plan_sha256") != plan_sha
        or completion.get("parallel_lanes_exact") != 4
        or completion.get("source_destination_sha256_size_verified") is not True
        or completion.get("raw_episode_zip_transferred") is not False
        or completion.get("runtime_or_training_started") is not False
        or completion.get("create_only") is not True
    ):
        raise ActionCriticTrainingError("target-view completion receipt binding drifted")

    raw_receipts = view.get("file_receipts")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise ActionCriticTrainingError("target-view lacks per-object transfer receipts")
    seen_destinations: set[str] = set()
    for raw in raw_receipts:
        item = _mapping(raw, label="target-view file receipt")
        destination_relative = item.get("destination_relative")
        destination = _relative_member(
            root, destination_relative, label="target-view transferred object"
        )
        receipt = _relative_member(
            root, item.get("receipt_relative"), label="target-view file receipt"
        )
        if str(destination_relative) in seen_destinations:
            raise ActionCriticTrainingError("target-view has duplicate object receipt destinations")
        seen_destinations.add(str(destination_relative))
        receipt_sha_expected = _require_sha256(
            str(item.get("receipt_sha256") or ""), label="target-view file receipt SHA-256"
        )
        if sha256_file(_regular_file(receipt, label="target-view file receipt")) != receipt_sha_expected:
            raise ActionCriticTrainingError("target-view file receipt digest mismatch")
        document = _read_json(receipt, label="target-view file receipt")
        destination_binding = _mapping(
            document.get("destination"), label="target-view receipt destination"
        )
        if (
            document.get("schema") != TARGET_VIEW_FILE_RECEIPT_SCHEMA
            or document.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION
            or document.get("plan_sha256") != plan_sha
            or destination_binding.get("relative_path") != destination_relative
            or destination_binding.get("regular_non_symlink") is not True
            or document.get("source_destination_identity_match") is not True
            or document.get("raw_episode_zip_transferred") is not False
            or document.get("create_only") is not True
        ):
            raise ActionCriticTrainingError("target-view per-object receipt binding drifted")
        if (
            destination.stat().st_size != destination_binding.get("size_bytes")
            or sha256_file(destination) != destination_binding.get("sha256")
        ):
            raise ActionCriticTrainingError("target-view transferred object identity drifted")
    required_objects = {
        str(manifest_path.relative_to(root)),
        str(receipt_path.relative_to(root)),
    }
    if not required_objects.issubset(seen_destinations):
        raise ActionCriticTrainingError("target-view lacks aggregate object transfer receipts")
    return {
        "pointer_path": pointer_path,
        "pointer_sha256": pointer_sha,
        "completion_path": completion_path,
        "completion_sha256": completion_sha,
        "target_set_root": target_set_root,
        "target_manifest": manifest_path,
        "target_manifest_sha256": manifest_sha,
        "target_set_receipt_path": receipt_path,
        "target_set_receipt_sha256": receipt_sha,
    }


@dataclass(frozen=True)
class TargetDescriptor:
    path: Path
    sha256: str
    size_bytes: int
    utc_day: str
    split: str
    row_schema: str


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActionCriticTrainingError(f"{label} must be a nonempty string")
    return value


def _canonical_terminal_targets(
    row: Mapping[str, Any], *, strict: bool
) -> tuple[float | None, bool, float | None, bool]:
    """Parse the terminal labels without erasing draw coverage.

    Revision 21 stores the signed terminal return and its BCE projection as
    top-level target-only fields.  Some early local fixtures used a nested
    ``terminal_win`` draft; that compatibility path is explicitly unavailable
    for a production typed target set.
    """

    canonical_keys = {"z", "z_mask", "win_target_mask"}
    has_canonical = bool(canonical_keys.intersection(row)) or any(
        key in row for key in ("win_target", "win_target_one_only_for_z_plus1")
    )
    if strict and not canonical_keys.issubset(row):
        raise ActionCriticTrainingError("canonical target row lacks z/z_mask/win_target_mask")
    if strict and not any(
        key in row for key in ("win_target", "win_target_one_only_for_z_plus1")
    ):
        raise ActionCriticTrainingError("canonical target row lacks win_target")
    if has_canonical:
        z_mask = _bool(row.get("z_mask"), label="terminal z mask")
        win_mask = _bool(row.get("win_target_mask"), label="win target mask")
        if z_mask != win_mask:
            raise ActionCriticTrainingError("terminal z and win target masks disagree")
        raw_z = row.get("z")
        raw_win_primary = row.get("win_target")
        raw_win_named = row.get("win_target_one_only_for_z_plus1")
        if raw_win_primary is not None and raw_win_named is not None:
            if _finite_number(raw_win_primary, label="win target") != _finite_number(
                raw_win_named, label="win target one-only projection"
            ):
                raise ActionCriticTrainingError("canonical win target aliases disagree")
        raw_win = (
            raw_win_primary
            if raw_win_primary is not None
            else raw_win_named
        )
        if z_mask:
            z = _finite_number(raw_z, label="terminal z")
            if z not in {-1.0, 0.0, 1.0}:
                raise ActionCriticTrainingError("terminal z must be -1, 0, or +1")
            win = _finite_number(raw_win, label="win target")
            expected_win = 1.0 if z == 1.0 else 0.0
            if win != expected_win:
                raise ActionCriticTrainingError("win target does not equal the terminal z projection")
        else:
            if raw_z is not None or raw_win is not None:
                raise ActionCriticTrainingError("masked terminal target carries a fabricated value")
            z = None
            win = None
        nested = row.get("terminal_win")
        if isinstance(nested, Mapping):
            nested_mask = _bool(
                _field(nested, "mask", "win_target_mask", default=None),
                label="nested win target mask",
            )
            nested_value = _field(nested, "value", "win_target", "win", default=None)
            if nested_mask != win_mask:
                raise ActionCriticTrainingError("nested and canonical win masks disagree")
            if nested_mask:
                if _finite_number(nested_value, label="nested win target") != win:
                    raise ActionCriticTrainingError("nested and canonical win targets disagree")
            elif nested_value is not None:
                raise ActionCriticTrainingError("masked nested win target carries a value")
        return z, z_mask, win, win_mask

    if strict:
        raise ActionCriticTrainingError("production target row lacks canonical terminal labels")
    nested = row.get("terminal_win")
    if isinstance(nested, Mapping):
        win_mask = _bool(
            _field(nested, "mask", "win_target_mask", default=None),
            label="win target mask",
        )
        raw_win = _field(nested, "value", "win_target", "win", default=None)
        if win_mask:
            win = _finite_number(raw_win, label="win target")
            if win not in {0.0, 1.0}:
                raise ActionCriticTrainingError("win target must be 0 or 1")
        else:
            if raw_win is not None:
                _finite_number(raw_win, label="masked win target")
            win = None
        return None, False, win, win_mask
    terminal = _mapping(row.get("terminal"), label="target terminal")
    z_mask = _bool(
        _field(terminal, "z_mask", "terminal_mask", default=None),
        label="terminal z mask",
    )
    win_mask = _bool(
        _field(terminal, "win_target_mask", "win_mask", default=None),
        label="win target mask",
    )
    if z_mask != win_mask:
        raise ActionCriticTrainingError("terminal z and win target masks disagree")
    raw_z = _field(terminal, "z", "terminal_return", default=None)
    raw_win = _field(terminal, "win_target", "win", default=None)
    if z_mask:
        z = _finite_number(raw_z, label="terminal z")
        if z not in {-1.0, 0.0, 1.0}:
            raise ActionCriticTrainingError("terminal z must be -1, 0, or +1")
        win = _finite_number(raw_win, label="win target")
        if win != (1.0 if z == 1.0 else 0.0):
            raise ActionCriticTrainingError("win target does not equal the terminal z projection")
        return z, True, win, True
    if raw_z is not None or raw_win is not None:
        raise ActionCriticTrainingError("masked terminal target carries a fabricated value")
    return None, False, None, False


def _validate_canonical_target_row(
    row: Mapping[str, Any], *, descriptor: TargetDescriptor, contract_goal_revision: int
) -> None:
    """Validate the production row ABI before it can join a base action."""

    if row.get("schema") != TARGET_OVERLAY_SCHEMA:
        raise ActionCriticTrainingError("target row is not the revision-21 target-overlay schema")
    if row.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION:
        raise ActionCriticTrainingError("target row critic semantic owner revision drifted")
    if row.get("goal_contract_goal_revision") != contract_goal_revision:
        raise ActionCriticTrainingError("target row current goal-contract revision drifted")
    if row.get("target_only") is not True or row.get("hidden_information_fields_present") is not False:
        raise ActionCriticTrainingError("target row crossed the target-only information boundary")
    if row.get("utc_day") != descriptor.utc_day or row.get("split") != descriptor.split:
        raise ActionCriticTrainingError("target row day or split drifted from its sealed shard")
    _nonempty_string(row.get("source_archive_sha256"), label="target source archive SHA-256")
    _nonempty_string(row.get("source_member"), label="target source member")
    _nonempty_string(row.get("episode_id"), label="target episode ID")
    _nonempty_string(row.get("program_identity"), label="target program identity")
    seat = row.get("acting_seat")
    step = row.get("env_step")
    if isinstance(seat, bool) or seat not in {0, 1} or isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ActionCriticTrainingError("target acting seat or env step is invalid")
    _canonical_terminal_targets(row, strict=True)
    horizons = _mapping(row.get("prize_differential"), label="target prize differential")
    if set(horizons) != {"h1", "h2", "h3"}:
        raise ActionCriticTrainingError("target row prize-horizon inventory drifted")
    required = {
        "h",
        "mask",
        "unavailable_reason",
        "future_program_identity",
        "future_env_step",
        "own_remaining_before",
        "own_remaining_after",
        "opponent_remaining_before",
        "opponent_remaining_after",
        "own_taken",
        "opponent_taken",
        "differential",
    }
    for horizon in (1, 2, 3):
        entry = _mapping(horizons[f"h{horizon}"], label=f"target prize h{horizon}")
        if entry.get("h") != horizon or not required.issubset(entry):
            raise ActionCriticTrainingError("target prize horizon ABI drifted")
        mask = _bool(entry.get("mask"), label="target prize mask")
        differential = entry.get("differential")
        if mask:
            value = _finite_number(differential, label="target prize differential")
            if not -1.0 <= value <= 1.0 or entry.get("unavailable_reason") is not None:
                raise ActionCriticTrainingError("labeled target prize horizon is malformed")
        elif differential is not None or not isinstance(entry.get("unavailable_reason"), str):
            raise ActionCriticTrainingError("masked target prize horizon is malformed")


class TargetOverlay:
    """Verified, one-shard-at-a-time target overlay reader."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        expected_sha256: str,
        expected_overlay_manifest_sha256: str,
        expected_base_completion_sha256: str,
        expected_contract_sha256: str,
        expected_contract_goal_revision: int | None = None,
        expected_overlay_day_sha256s: Mapping[str, str] | None = None,
        allow_test_fixture: bool = False,
        verify_shards: bool = True,
    ) -> None:
        self.manifest_path = _regular_file(manifest_path, label="target manifest")
        self.manifest_root = _regular_directory(
            _manifest_root(self.manifest_path), label="target-set artifact root"
        )
        self.allow_test_fixture = bool(allow_test_fixture)
        self.production = not self.allow_test_fixture
        # Full target-shard hashing is non-negotiable for production.  The
        # switch exists only for deliberately tiny in-process test fixtures.
        self.verify_shards = True if self.production else bool(verify_shards)
        self.expected_contract_goal_revision = expected_contract_goal_revision
        self.manifest_sha256 = _sha_matches(
            self.manifest_path, expected_sha256, label="target manifest"
        )
        self.manifest = _read_json(self.manifest_path, label="target manifest")
        manifest_schema = str(self.manifest.get("schema") or "")
        if manifest_schema != TARGET_SET_MANIFEST_SCHEMA:
            raise ActionCriticTrainingError("target manifest schema drifted")
        self.schema = manifest_schema
        self._validate_binding(
            expected_overlay_manifest_sha256=expected_overlay_manifest_sha256,
            expected_base_completion_sha256=expected_base_completion_sha256,
            expected_contract_sha256=expected_contract_sha256,
        )
        raw_descriptors = self._raw_descriptors(
            expected_overlay_day_sha256s=expected_overlay_day_sha256s or {}
        )
        if not raw_descriptors:
            raise ActionCriticTrainingError("target manifest has no target shards")
        descriptors: list[TargetDescriptor] = []
        for raw in raw_descriptors:
            item = _mapping(raw, label="target shard")
            descriptor = TargetDescriptor(
                path=_regular_file(item.get("path") or "", label="target shard"),
                sha256=_require_sha256(str(item.get("sha256") or ""), label="target shard SHA-256"),
                size_bytes=int(item.get("size_bytes", -1)),
                utc_day=str(item.get("utc_day") or ""),
                split=str(item.get("split") or ""),
                row_schema=str(item.get("row_schema") or TARGET_OVERLAY_SCHEMA),
            )
            if (
                descriptor.size_bytes < 1
                or not descriptor.utc_day
                or descriptor.split not in {"train", "validation", "evaluation"}
            ):
                raise ActionCriticTrainingError("target shard descriptor is malformed")
            if descriptor.path.stat().st_size != descriptor.size_bytes:
                raise ActionCriticTrainingError(
                    f"target shard size mismatch: {descriptor.path}"
                )
            if self.verify_shards and sha256_file(descriptor.path) != descriptor.sha256:
                raise ActionCriticTrainingError(
                    f"target shard digest mismatch: {descriptor.path}"
                )
            descriptors.append(descriptor)
        self.descriptors = tuple(descriptors)
        if self.production:
            if len(self.descriptors) != len(EXPECTED_TRAIN_DAYS) + len(
                EXPECTED_VALIDATION_DAYS
            ) + len(EXPECTED_EVALUATION_DAYS):
                raise ActionCriticTrainingError(
                    "production target set does not contain exactly 20 target shards"
                )

    def _raw_descriptors(
        self, *, expected_overlay_day_sha256s: Mapping[str, str]
    ) -> list[Mapping[str, Any]]:
        """Read the typed revision-21 20-day target set.

        A target set is its own sealed artifact.  Direct shard lists and
        source-host absolute roots are deliberately unavailable outside the
        explicit hermetic-fixture escape hatch, because neither carries the
        required per-day receipt or relocation proof.
        """

        members = self.manifest.get("target_days")
        if not isinstance(members, list) or not members:
            if not self.allow_test_fixture:
                raise ActionCriticTrainingError(
                    "target input requires a typed target-set manifest with sealed day members"
                )
            direct = self.manifest.get("target_shards")
            if not isinstance(direct, list) or not direct:
                raise ActionCriticTrainingError("test target fixture has no direct target shards")
            # Direct lists were an early fixture-only convenience.  They are
            # never valid in production because they lack day receipts and a
            # portable artifact-root binding.
            return [
                {
                    **dict(_mapping(item, label="test target shard")),
                    "path": str(
                        _relative_member(
                            self.manifest_root,
                            _mapping(item, label="test target shard").get("path"),
                            label="test target shard",
                        )
                    ),
                    "row_schema": str(
                        _mapping(item, label="test target shard").get("row_schema")
                        or TARGET_OVERLAY_SCHEMA
                    ),
                }
                for item in direct
            ]
        if self.production:
            expected_days = (
                list(EXPECTED_TRAIN_DAYS)
                + list(EXPECTED_VALIDATION_DAYS)
                + list(EXPECTED_EVALUATION_DAYS)
            )
            if len(members) != 20:
                raise ActionCriticTrainingError("target set must contain exactly 20 day members")
            if [str(_mapping(item, label="target-set member").get("utc_day") or "") for item in members] != expected_days:
                raise ActionCriticTrainingError("target-set day order is not the sealed recent-20 order")

        result: list[Mapping[str, Any]] = []
        seen_days: set[str] = set()
        seen_paths: set[Path] = set()
        for member in members:
            item = _mapping(member, label="target-set member")
            day = str(item.get("utc_day") or "")
            split = str(item.get("split") or "")
            if not day or day in seen_days or split not in {"train", "validation", "evaluation"}:
                raise ActionCriticTrainingError("target-set day member is malformed")
            seen_days.add(day)
            if self.production and split != _expected_split_for_day(day):
                raise ActionCriticTrainingError("target-set day split drifts from the sealed split")
            day_root = _relative_member(
                self.manifest_root, item.get("day_artifact_root"), label="target day artifact"
            )
            if not day_root.is_dir() or day_root.is_symlink():
                raise ActionCriticTrainingError("target day artifact root is not a real directory")
            manifest_path = _relative_member(
                self.manifest_root, item.get("day_manifest_path"), label="target day manifest"
            )
            receipt_path = _relative_member(
                self.manifest_root, item.get("day_receipt_path"), label="target day receipt"
            )
            _sha_matches(
                manifest_path,
                str(item.get("day_manifest_sha256") or ""),
                label="target day manifest",
            )
            _sha_matches(
                receipt_path,
                str(item.get("day_receipt_sha256") or ""),
                label="target day receipt",
            )
            day_manifest = _read_json(manifest_path, label="target day manifest")
            day_receipt = _read_json(receipt_path, label="target day receipt")
            self._validate_day_documents(
                item=item,
                day=day,
                split=split,
                day_root=day_root,
                day_manifest=day_manifest,
                day_manifest_sha256=str(item.get("day_manifest_sha256") or ""),
                day_receipt=day_receipt,
                expected_overlay_day_sha256s=expected_overlay_day_sha256s,
            )
            shard = _mapping(item.get("target_shard"), label="target day shard")
            manifest_shard = _mapping(
                day_manifest.get("target_shard"), label="day manifest target shard"
            )
            if dict(shard) != dict(manifest_shard):
                raise ActionCriticTrainingError("target-set/day-manifest target shard binding drifted")
            shard_path = _relative_member(
                day_root, shard.get("path"), label="target day shard"
            )
            if shard_path in seen_paths:
                raise ActionCriticTrainingError("target-set reuses a target shard path")
            seen_paths.add(shard_path)
            result.append(
                {
                    **dict(shard),
                    "path": str(shard_path),
                    "utc_day": day,
                    "split": split,
                    "row_schema": TARGET_OVERLAY_SCHEMA,
                }
            )
        self._validate_aggregate_inventory(result)
        return result

    def _validate_day_documents(
        self,
        *,
        item: Mapping[str, Any],
        day: str,
        split: str,
        day_root: Path,
        day_manifest: Mapping[str, Any],
        day_manifest_sha256: str,
        day_receipt: Mapping[str, Any],
        expected_overlay_day_sha256s: Mapping[str, str],
    ) -> None:
        if day_manifest.get("schema") != TARGET_DAY_MANIFEST_SCHEMA:
            raise ActionCriticTrainingError("target day manifest schema drifted")
        if day_receipt.get("schema") != TARGET_DAY_RECEIPT_SCHEMA:
            raise ActionCriticTrainingError("target day receipt schema drifted")
        if day_manifest.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION:
            raise ActionCriticTrainingError("target day manifest critic semantic owner drifted")
        if day_receipt.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION:
            raise ActionCriticTrainingError("target day receipt critic semantic owner drifted")
        if day_manifest.get("utc_day") != day or day_manifest.get("split") != split:
            raise ActionCriticTrainingError("target day manifest day or split drifted")
        if day_receipt.get("manifest_sha256") != day_manifest_sha256:
            raise ActionCriticTrainingError("target day receipt does not bind its manifest")
        declared_manifest_path = day_receipt.get("manifest_path")
        if not isinstance(declared_manifest_path, str) or not declared_manifest_path:
            raise ActionCriticTrainingError("target day receipt lacks its manifest path")
        manifest_relative = _relative_member(
            day_root, declared_manifest_path, label="target day receipt manifest"
        )
        expected_manifest_path = _relative_member(
            self.manifest_root, item.get("day_manifest_path"), label="target day manifest"
        )
        if manifest_relative != expected_manifest_path:
            raise ActionCriticTrainingError("target day receipt manifest path drifted")
        contract = _mapping(day_manifest.get("goal_contract"), label="target day goal contract")
        if contract.get("sha256") != self.expected_contract_sha256:
            raise ActionCriticTrainingError("target day manifest contract binding drifted")
        if day_receipt.get("goal_contract_sha256") != self.expected_contract_sha256:
            raise ActionCriticTrainingError("target day receipt contract binding drifted")
        if self.production:
            if (
                contract.get("goal_revision") != self.expected_contract_goal_revision
                or contract.get("critic_semantic_owner_goal_revision")
                != CRITIC_SEMANTIC_OWNER_REVISION
                or contract.get("required_authority") != CRITIC_AUTHORITY_KEY
                or day_receipt.get("goal_contract_goal_revision")
                != self.expected_contract_goal_revision
                or day_receipt.get("critic_semantic_owner_goal_revision")
                != CRITIC_SEMANTIC_OWNER_REVISION
            ):
                raise ActionCriticTrainingError("target day current/critic contract binding drifted")
        target = _mapping(item.get("target_shard"), label="target day shard")
        receipt_target_fields = {
            "sha256": "target_shard_sha256",
            "size_bytes": "target_shard_size_bytes",
            "row_count": "target_row_count",
        }
        for field, receipt_field in receipt_target_fields.items():
            if day_receipt.get(receipt_field) != target.get(field):
                raise ActionCriticTrainingError("target day receipt target-shard binding drifted")
        raw_zip = _mapping(day_manifest.get("raw_episode_zip"), label="target day raw ZIP")
        aggregate_raw = _mapping(item.get("raw_episode_zip"), label="target-set raw ZIP")
        raw_identity_fields = (
            "sha256",
            "size_bytes",
            "source_archive_sha256_verified",
        )
        if any(raw_zip.get(field) != aggregate_raw.get(field) for field in raw_identity_fields):
            raise ActionCriticTrainingError("target-set/day-manifest raw ZIP binding drifted")
        if raw_zip.get("source_archive_sha256_verified") is not True:
            raise ActionCriticTrainingError("target day raw ZIP/source archive identity is unverified")
        if day_receipt.get("raw_episode_zip_sha256") != raw_zip.get("sha256"):
            raise ActionCriticTrainingError("target day receipt raw ZIP binding drifted")
        overlay = _mapping(
            day_manifest.get("complete_action_overlay"), label="target day complete action overlay"
        )
        aggregate_overlay = _mapping(
            item.get("complete_action_overlay"), label="target-set complete action overlay"
        )
        overlay_identity_fields = ("schema", "sha256", "size_bytes", "split")
        if any(
            field in overlay or field in aggregate_overlay
            for field in overlay_identity_fields
        ) and any(
            overlay.get(field) != aggregate_overlay.get(field)
            for field in overlay_identity_fields
            if field in overlay or field in aggregate_overlay
        ):
            raise ActionCriticTrainingError("target-set/day-manifest overlay binding drifted")
        expected_day_overlay = expected_overlay_day_sha256s.get(day)
        if expected_day_overlay and overlay.get("sha256") != expected_day_overlay:
            raise ActionCriticTrainingError("target day overlay does not match sealed overlay shard")
        if day_receipt.get("complete_action_overlay_sha256") != overlay.get("sha256"):
            raise ActionCriticTrainingError("target day receipt overlay binding drifted")

    def _validate_aggregate_inventory(
        self, descriptors: Sequence[Mapping[str, Any]]
    ) -> None:
        expected_days = (
            list(EXPECTED_TRAIN_DAYS)
            + list(EXPECTED_VALIDATION_DAYS)
            + list(EXPECTED_EVALUATION_DAYS)
        )
        observed_days = [str(item.get("utc_day") or "") for item in descriptors]
        if self.production and observed_days != expected_days:
            raise ActionCriticTrainingError("target set day inventory is not exactly recent-20")
        declared_source_days = self.manifest.get("source_days")
        if self.production and declared_source_days != expected_days:
            raise ActionCriticTrainingError("target set source day inventory drifted")
        expected_splits = {
            "train": list(EXPECTED_TRAIN_DAYS),
            "validation": list(EXPECTED_VALIDATION_DAYS),
            "evaluation": list(EXPECTED_EVALUATION_DAYS),
        }
        if self.production and self.manifest.get("split_days") != expected_splits:
            raise ActionCriticTrainingError("target set split day lists drifted")
        if self.production and self.manifest.get("episode_and_seat_group_split_disjoint") is not True:
            raise ActionCriticTrainingError("target set lacks group split-disjointness proof")
        if self.production:
            boundary = _mapping(
                self.manifest.get("information_boundary"), label="target-set information boundary"
            )
            if boundary.get(
                "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed"
            ) is not False:
                raise ActionCriticTrainingError("target set information boundary drifted")
        declared_targets = self.manifest.get("all_20_target_shards")
        declared_raw = self.manifest.get("all_20_raw_episode_zip_sha256s")
        if self.production:
            if not isinstance(declared_targets, list) or not isinstance(declared_raw, list):
                raise ActionCriticTrainingError("target set lacks 20-day object inventories")
            target_by_day = {
                str(_mapping(item, label="target inventory item").get("utc_day") or ""): item
                for item in declared_targets
            }
            raw_by_day = {
                str(_mapping(item, label="raw ZIP inventory item").get("utc_day") or ""): item
                for item in declared_raw
            }
            if set(target_by_day) != set(expected_days) or set(raw_by_day) != set(expected_days):
                raise ActionCriticTrainingError("target set 20-day object inventory is incomplete")
            members = _mapping({str(item.get("utc_day")): item for item in self.manifest["target_days"]}, label="target days")
            for descriptor in descriptors:
                day = str(descriptor["utc_day"])
                member = _mapping(members[day], label="target-set member")
                target = _mapping(member.get("target_shard"), label="target day shard")
                raw = _mapping(member.get("raw_episode_zip"), label="target day raw ZIP")
                for field in ("sha256", "size_bytes", "row_count", "split"):
                    if target_by_day[day].get(field) != (descriptor.get(field) if field != "row_count" else target.get(field)):
                        raise ActionCriticTrainingError("target set target-shard inventory drifted")
                for field in ("sha256", "size_bytes"):
                    if raw_by_day[day].get(field) != raw.get(field):
                        raise ActionCriticTrainingError("target set raw ZIP inventory drifted")

    def _validate_binding(
        self,
        *,
        expected_overlay_manifest_sha256: str,
        expected_base_completion_sha256: str,
        expected_contract_sha256: str,
    ) -> None:
        self.expected_contract_sha256 = _require_sha256(
            expected_contract_sha256, label="goal contract SHA-256"
        )
        self.expected_overlay_manifest_sha256 = _require_sha256(
            expected_overlay_manifest_sha256, label="overlay manifest SHA-256"
        )
        self.expected_base_completion_sha256 = _require_sha256(
            expected_base_completion_sha256, label="base completion SHA-256"
        )
        base = self.manifest.get("base_pack_completion")
        base_sha = (
            str(base.get("sha256") or "")
            if isinstance(base, Mapping)
            else str(self.manifest.get("base_pack_completion_sha256") or "")
        )
        if base_sha != self.expected_base_completion_sha256:
            raise ActionCriticTrainingError("target-set base completion binding mismatch")
        overlay = self.manifest.get("complete_action_overlay_manifest")
        overlay_sha = (
            str(overlay.get("sha256") or "")
            if isinstance(overlay, Mapping)
            else str(self.manifest.get("complete_action_overlay_manifest_sha256") or "")
        )
        if overlay_sha != self.expected_overlay_manifest_sha256:
            raise ActionCriticTrainingError("target-set complete-action overlay binding mismatch")
        contract = self.manifest.get("goal_contract")
        contract_sha = (
            str(contract.get("sha256") or "") if isinstance(contract, Mapping) else ""
        )
        if self.production and not isinstance(contract, Mapping):
            raise ActionCriticTrainingError("target-set goal contract binding is absent")
        if isinstance(contract, Mapping) and contract_sha != self.expected_contract_sha256:
            raise ActionCriticTrainingError("target-set goal contract binding mismatch")
        if self.production and contract_sha != self.expected_contract_sha256:
            raise ActionCriticTrainingError("target-set goal contract binding mismatch")
        if self.production and self.manifest.get("owner_goal_revision") != CRITIC_SEMANTIC_OWNER_REVISION:
            raise ActionCriticTrainingError("target-set critic semantic owner revision drifted")
        if self.production and (
            self.expected_contract_goal_revision is None
            or self.manifest.get("goal_contract_goal_revision")
            != self.expected_contract_goal_revision
            or self.manifest.get("critic_semantic_owner_goal_revision")
            != CRITIC_SEMANTIC_OWNER_REVISION
            or self.manifest.get("required_critic_authority") != CRITIC_AUTHORITY_KEY
        ):
            raise ActionCriticTrainingError("target-set current/critic contract binding drifted")

    def descriptors_for_split(self, split: str) -> tuple[TargetDescriptor, ...]:
        result = tuple(item for item in self.descriptors if item.split == split)
        if not result:
            raise ActionCriticTrainingError(f"target overlay has no {split} shards")
        return result

    def split_days(self, split: str) -> tuple[str, ...]:
        days = tuple(item.utc_day for item in self.descriptors_for_split(split))
        if len(set(days)) != len(days):
            raise ActionCriticTrainingError(f"target overlay duplicates {split} UTC days")
        return days

    def iter_rows(self, split: str) -> Iterator[tuple[TargetDescriptor, dict[str, Any]]]:
        schema_module = _optional_target_schema_module()
        allowed_rows = _module_schema_values(schema_module, kind="row")
        for descriptor in self.descriptors_for_split(split):
            with descriptor.path.open(
                "r", encoding="utf-8", buffering=8 * 1024 * 1024
            ) as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ActionCriticTrainingError(
                            f"invalid target row {descriptor.path}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ActionCriticTrainingError("target row must be an object")
                    row_schema = str(row.get("schema") or "")
                    if not row_schema:
                        raise ActionCriticTrainingError("target row has no schema")
                    if descriptor.row_schema and row_schema != descriptor.row_schema:
                        raise ActionCriticTrainingError("target row schema descriptor mismatch")
                    if allowed_rows and row_schema not in allowed_rows:
                        raise ActionCriticTrainingError("target row schema drifted")
                    if self.production:
                        assert self.expected_contract_goal_revision is not None
                        _validate_canonical_target_row(
                            row,
                            descriptor=descriptor,
                            contract_goal_revision=self.expected_contract_goal_revision,
                        )
                    if str(row.get("split") or "") != split:
                        raise ActionCriticTrainingError("target row split drifted")
                    _call_schema_validator(schema_module, row)
                    yield descriptor, row


@dataclass(frozen=True)
class CompleteActionExample:
    """One selected complete action and only its recorded chosen-action labels."""

    program_identity: str
    utc_day: str
    episode_id: str
    acting_seat: int
    env_step: int
    stage_count: int
    first_stage_menu: tuple[tuple[float, ...], ...]
    selected_stage_features: tuple[tuple[float, ...], ...]
    # These current-decision public fields prevent action-value aliases when
    # two legal options happen to share the same sealed 40-D feature vector.
    # They are never passed to state-value heads.
    selected_option_indices: tuple[int, ...]
    selected_legal_counts: tuple[int, ...]
    selected_action_programs: tuple[tuple[int, ...], ...]
    terminal_z: float | None
    terminal_z_mask: bool
    win_target: float | None
    win_target_mask: bool
    prize_targets: tuple[float | None, float | None, float | None]
    prize_masks: tuple[bool, bool, bool]


def _normalize_horizons(
    row: Mapping[str, Any], *, strict_canonical: bool
) -> tuple[
    tuple[float | None, float | None, float | None], tuple[bool, bool, bool]
]:
    raw_targets = _mapping(
        _field(row, "prize_differential", "prize_targets", default=None),
        label="target prize differential",
    )
    raw_horizons = _field(raw_targets, "horizons", "values", default=None)
    found: dict[int, tuple[float | None, bool]] = {}
    if isinstance(raw_horizons, list):
        if strict_canonical:
            raise ActionCriticTrainingError("production target horizons must use canonical h1/h2/h3 members")
        iterable = raw_horizons
    else:
        # The canonical revision-21 target overlay uses h1/h2/h3 members.
        iterable = [
            {"h": horizon, **dict(_mapping(raw_targets.get(f"h{horizon}"), label=f"prize h{horizon}"))}
            for horizon in (1, 2, 3)
        ]
    for item in iterable:
        horizon = _mapping(item, label="target prize horizon")
        h_raw = _field(horizon, "horizon", "h", default=None)
        if isinstance(h_raw, bool) or not isinstance(h_raw, int) or h_raw not in {1, 2, 3}:
            raise ActionCriticTrainingError("target prize horizon must be 1, 2, or 3")
        if h_raw in found:
            raise ActionCriticTrainingError("duplicate target prize horizon")
        mask = _bool(_field(horizon, "mask", "target_mask", default=None), label="prize mask")
        if strict_canonical:
            required = {
                "h",
                "mask",
                "unavailable_reason",
                "future_program_identity",
                "future_env_step",
                "own_remaining_before",
                "own_remaining_after",
                "opponent_remaining_before",
                "opponent_remaining_after",
                "own_taken",
                "opponent_taken",
                "differential",
            }
            if not required.issubset(horizon):
                raise ActionCriticTrainingError("canonical target prize horizon is incomplete")
            raw_value = horizon.get("differential")
        else:
            raw_value = _field(horizon, "differential", "target", "value", default=None)
        if mask:
            value = _finite_number(raw_value, label="prize differential")
            if not -1.0 <= value <= 1.0:
                raise ActionCriticTrainingError("prize differential outside [-1,+1]")
        else:
            # An unavailable interval must remain unavailable.  A numeric
            # value is tolerated only because it is ignored by the mask; it
            # cannot become a fabricated zero training target.
            value = None
            if raw_value is not None:
                _finite_number(raw_value, label="masked prize differential")
            if strict_canonical and not isinstance(horizon.get("unavailable_reason"), str):
                raise ActionCriticTrainingError("masked canonical prize horizon lacks an unavailable reason")
        found[h_raw] = (value, mask)
    if set(found) != {1, 2, 3}:
        raise ActionCriticTrainingError("target overlay lacks one or more prize horizons")
    values = tuple(found[horizon][0] for horizon in (1, 2, 3))
    masks = tuple(found[horizon][1] for horizon in (1, 2, 3))
    return values, masks  # type: ignore[return-value]


def _normalize_target(
    row: Mapping[str, Any],
    descriptor: TargetDescriptor,
    *,
    expected_split: str,
    strict_canonical: bool,
) -> dict[str, Any]:
    # Revision-21's sealed rows are intentionally flat.  Retain support for
    # the earlier nested draft only so a locally sealed test corpus remains
    # readable; both forms are reduced to the same strict join identity here.
    source = row.get("source") if isinstance(row.get("source"), Mapping) else row
    source = _mapping(source, label="target source")
    if str(_field(source, "utc_day", "day", default="")) != descriptor.utc_day:
        raise ActionCriticTrainingError("target source day differs from target shard")
    program_identity = str(
        _field(source, "program_identity", default=_field(row, "program_identity", default=""))
        or ""
    )
    if not program_identity:
        raise ActionCriticTrainingError("target row lacks program identity")
    z, z_mask, win, win_mask = _canonical_terminal_targets(
        row, strict=strict_canonical
    )
    prizes, masks = _normalize_horizons(row, strict_canonical=strict_canonical)
    source_member = _nonempty_string(
        _field(source, "source_member", default=_field(row, "source_member", default=None)),
        label="target source member",
    )
    return {
        "program_identity": program_identity,
        "utc_day": descriptor.utc_day,
        "episode_id": str(_field(source, "episode_id", default="") or ""),
        "acting_seat": _field(source, "acting_seat", "seat", default=None),
        "env_step": _field(source, "env_step", default=None),
        "archive_sha256": str(
            _field(source, "source_archive_sha256", "archive_sha256", default="") or ""
        ),
        "source_member": source_member,
        "terminal_z": z,
        "terminal_z_mask": z_mask,
        "win_target": win,
        "win_target_mask": win_mask,
        "prize_targets": prizes,
        "prize_masks": masks,
        "split": expected_split,
    }


def _feature_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ActionCriticTrainingError(f"{label} must be a feature vector")
    if len(value) != FEATURE_WIDTH:
        raise ActionCriticTrainingError(f"{label} must have exactly {FEATURE_WIDTH} values")
    result = tuple(_finite_number(item, label=label) for item in value)
    return result


def _selected_action_program(value: Any, *, label: str) -> tuple[int, ...]:
    """Validate one bounded, already-sealed current action-program prefix."""

    if not isinstance(value, list):
        raise ActionCriticTrainingError(f"{label} must be a list of action tokens")
    if len(value) > MAX_SELECTED_ACTION_PROGRAM_TOKENS:
        raise ActionCriticTrainingError(
            f"{label} exceeds the sealed action-program token bound"
        )
    tokens: list[int] = []
    for token in value:
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token <= MAX_SELECTED_ACTION_TOKEN_VALUE
        ):
            raise ActionCriticTrainingError(
                f"{label} must contain bounded nonnegative integer action tokens"
            )
        tokens.append(token)
    return tuple(tokens)


def _example_from_join(
    sample: Mapping[str, Any], target: Mapping[str, Any], *, split: str
) -> CompleteActionExample:
    if sample.get("public_information_only") is not True:
        raise ActionCriticTrainingError("base sample crossed public information boundary")
    if int(sample.get("base_feature_width", -1)) != FEATURE_WIDTH:
        raise ActionCriticTrainingError("base feature width drifted from 40")
    program = _mapping(sample.get("program"), label="complete-action program")
    program_identity = str(program.get("program_identity") or "")
    if not program_identity or program_identity != target["program_identity"]:
        raise ActionCriticTrainingError("target/base program identity mismatch")
    day = str(program.get("utc_day") or "")
    if day != target["utc_day"]:
        raise ActionCriticTrainingError("target/base UTC day mismatch")
    fields = (
        (
            "source_member",
            str(program.get("source_member") or ""),
            target["source_member"],
        ),
        ("episode_id", str(program.get("episode_id") or ""), target["episode_id"]),
        ("acting_seat", program.get("acting_seat"), target["acting_seat"]),
        ("env_step", program.get("env_step"), target["env_step"]),
        (
            "source_archive_sha256",
            str(program.get("source_archive_sha256") or ""),
            target["archive_sha256"],
        ),
    )
    for label, observed, expected in fields:
        if expected is None or expected == "":
            raise ActionCriticTrainingError(f"target source lacks {label}")
        if str(observed) != str(expected):
            raise ActionCriticTrainingError(f"target/base {label} mismatch")
    stages = program.get("stages")
    feature_stages = sample.get("base_option_features_by_stage")
    if not isinstance(stages, list) or not isinstance(feature_stages, list):
        raise ActionCriticTrainingError("program stage/base feature alignment is absent")
    if not stages or len(stages) != len(feature_stages):
        raise ActionCriticTrainingError("program stage/base feature alignment drifted")
    if len(stages) > MAX_SELECTED_ACTION_STAGES:
        raise ActionCriticTrainingError("program exceeds the sealed factorized-stage bound")
    selected_features: list[tuple[float, ...]] = []
    selected_indices: list[int] = []
    selected_legal_counts: list[int] = []
    selected_action_programs: list[tuple[int, ...]] = []
    first_menu: tuple[tuple[float, ...], ...] | None = None
    for index, (stage_raw, feature_rows) in enumerate(zip(stages, feature_stages)):
        stage = _mapping(stage_raw, label="program stage")
        if int(stage.get("factorized_stage", -1)) != index:
            raise ActionCriticTrainingError("factorized stages are not contiguous")
        if not isinstance(feature_rows, list) or not feature_rows:
            raise ActionCriticTrainingError("stage contains no base option features")
        if len(feature_rows) > MAX_SELECTED_LEGAL_OPTIONS:
            raise ActionCriticTrainingError("stage exceeds the sealed legal-option bound")
        rows = tuple(
            _feature_vector(value, label=f"stage {index} option feature")
            for value in feature_rows
        )
        legal = stage.get("ordered_legal_action_programs")
        valid = stage.get("valid_option_mask")
        chosen = stage.get("selected_option_index")
        selected_action = stage.get("selected_action_program")
        if (
            not isinstance(legal, list)
            or not isinstance(valid, list)
            or len(legal) != len(rows)
            or valid != [True] * len(rows)
            or isinstance(chosen, bool)
            or not isinstance(chosen, int)
            or not 0 <= chosen < len(rows)
            or legal[chosen] != selected_action
        ):
            raise ActionCriticTrainingError("legal order/selected action alignment drifted")
        action_program = _selected_action_program(
            selected_action, label=f"stage {index} selected action program"
        )
        if index == 0:
            first_menu = rows
        selected_features.append(rows[chosen])
        selected_indices.append(chosen)
        selected_legal_counts.append(len(rows))
        selected_action_programs.append(action_program)
    assert first_menu is not None
    return CompleteActionExample(
        program_identity=program_identity,
        utc_day=day,
        episode_id=str(program["episode_id"]),
        acting_seat=int(program["acting_seat"]),
        env_step=int(program["env_step"]),
        stage_count=len(selected_features),
        first_stage_menu=first_menu,
        selected_stage_features=tuple(selected_features),
        selected_option_indices=tuple(selected_indices),
        selected_legal_counts=tuple(selected_legal_counts),
        selected_action_programs=tuple(selected_action_programs),
        terminal_z=target["terminal_z"],
        terminal_z_mask=bool(target["terminal_z_mask"]),
        win_target=target["win_target"],
        win_target_mask=bool(target["win_target_mask"]),
        prize_targets=target["prize_targets"],
        prize_masks=target["prize_masks"],
    )


def split_days_from_base(dataset: Recent20RTPDataset, split: str) -> tuple[str, ...]:
    days = tuple(
        str(row.get("utc_day") or "")
        for row in dataset.shards
        if str(row.get("split") or "") == split
    )
    if not days or any(not day for day in days) or len(set(days)) != len(days):
        raise ActionCriticTrainingError(f"base overlay {split} day inventory is malformed")
    return days


def assert_exact_split_contract(
    dataset: Recent20RTPDataset,
    targets: TargetOverlay,
    *,
    test_allow_noncanonical_split: bool = False,
) -> dict[str, list[str]]:
    observed = {
        split: list(split_days_from_base(dataset, split))
        for split in ("train", "validation", "evaluation")
    }
    target_days = {
        split: list(targets.split_days(split))
        for split in ("train", "validation", "evaluation")
    }
    for split in observed:
        if observed[split] != target_days[split]:
            raise ActionCriticTrainingError(
                f"target/base {split} day ordering or inventory mismatch"
            )
    all_days = [day for values in observed.values() for day in values]
    if len(set(all_days)) != len(all_days):
        raise ActionCriticTrainingError("day split overlap")
    if not test_allow_noncanonical_split:
        expected = {
            "train": list(EXPECTED_TRAIN_DAYS),
            "validation": list(EXPECTED_VALIDATION_DAYS),
            "evaluation": list(EXPECTED_EVALUATION_DAYS),
        }
        if observed != expected:
            raise ActionCriticTrainingError(
                "recent-20 train/validation/evaluation days do not match the sealed split"
            )
    return observed


def iter_complete_action_examples(
    dataset: Recent20RTPDataset,
    targets: TargetOverlay,
    *,
    split: str,
    max_programs: int = 0,
) -> Iterator[CompleteActionExample]:
    """Merge the two sealed streams in order without caching a day or corpus."""
    target_rows = iter(targets.iter_rows(split))
    emitted = 0
    for sample in dataset.iter_samples(split):
        try:
            descriptor, raw_target = next(target_rows)
        except StopIteration as exc:
            raise ActionCriticTrainingError("target overlay ended before base overlay") from exc
        target = _normalize_target(
            raw_target,
            descriptor,
            expected_split=split,
            strict_canonical=targets.production,
        )
        yield _example_from_join(sample, target, split=split)
        emitted += 1
        if max_programs and emitted >= max_programs:
            return
    try:
        next(target_rows)
    except StopIteration:
        return
    raise ActionCriticTrainingError("target overlay has rows absent from base overlay")


def bounded_shuffle(
    rows: Iterable[CompleteActionExample], *, buffer_size: int, seed: int
) -> Iterator[CompleteActionExample]:
    """A deterministic bounded reservoir shuffle; it never caches the corpus."""
    if buffer_size < 1:
        raise ValueError("shuffle buffer must be positive")
    rng = random.Random(int(seed))
    buffer: list[CompleteActionExample] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def batched(
    rows: Iterable[CompleteActionExample], *, batch_size: int
) -> Iterator[list[CompleteActionExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch: list[CompleteActionExample] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class TargetBatch:
    win_target: torch.Tensor
    win_mask: torch.Tensor
    prize_target: torch.Tensor
    prize_mask: torch.Tensor
    z_target: torch.Tensor
    z_mask: torch.Tensor


def _target_batch(rows: Sequence[CompleteActionExample], device: torch.device) -> TargetBatch:
    win = [0.0 if row.win_target is None else row.win_target for row in rows]
    prize = [
        [0.0 if target is None else target for target in row.prize_targets]
        for row in rows
    ]
    z = [0.0 if row.terminal_z is None else row.terminal_z for row in rows]
    return TargetBatch(
        win_target=torch.tensor(win, dtype=torch.float32, device=device),
        win_mask=torch.tensor(
            [row.win_target_mask for row in rows], dtype=torch.bool, device=device
        ),
        prize_target=torch.tensor(prize, dtype=torch.float32, device=device),
        prize_mask=torch.tensor(
            [row.prize_masks for row in rows], dtype=torch.bool, device=device
        ),
        z_target=torch.tensor(z, dtype=torch.float32, device=device),
        z_mask=torch.tensor(
            [row.terminal_z_mask for row in rows], dtype=torch.bool, device=device
        ),
    )


def _candidate_config_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    hidden_width = int(config.get("hidden_width", config.get("hidden_dim", 128)))
    dropout = float(config.get("dropout", 0.0))
    return {
        "feature_width": FEATURE_WIDTH,
        "feature_dim": FEATURE_WIDTH,
        "input_width": FEATURE_WIDTH,
        "semantic_feature_width": FEATURE_WIDTH,
        "first_stage_feature_width": FEATURE_WIDTH,
        "selected_stage_feature_width": FEATURE_WIDTH,
        "hidden_width": hidden_width,
        "hidden_dim": hidden_width,
        "d_model": hidden_width,
        "state_hidden_dim": hidden_width,
        "action_hidden_dim": hidden_width,
        "q_hidden_dim": hidden_width,
        "max_action_stages": int(config.get("max_action_stages", 32)),
        "max_legal_options": int(config.get("max_legal_options", 64)),
        "max_action_program_tokens": int(
            config.get("max_action_program_tokens", 32)
        ),
        "max_action_token_value": int(
            config.get("max_action_token_value", 63)
        ),
        "dropout": dropout,
    }


def _constructor_kwargs(constructor: Any, supplied: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return dict(supplied)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(supplied)
    return {
        key: value
        for key, value in supplied.items()
        if key in signature.parameters
    }


def _config_to_dict(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        return dict(dataclasses.asdict(config))
    try:
        value = vars(config)
    except TypeError:
        return {"repr": repr(config)}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool, type(None), list, tuple, dict))
    }


def build_critic(
    *, hidden_width: int = 128, dropout: float = 0.0, saved_config: Mapping[str, Any] | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    sidecar = _require_sidecar_module()
    desired = dict(saved_config or {})
    if not desired:
        desired = {"hidden_width": int(hidden_width), "dropout": float(dropout)}
    config_class = getattr(
        sidecar, "ActionCriticSidecarConfig", getattr(sidecar, "ActionCriticConfig", None)
    )
    if config_class is None:
        raise ActionCriticTrainingError("action critic sidecar has no config class")
    config_kwargs = _constructor_kwargs(config_class, _candidate_config_kwargs(desired) | desired)
    try:
        critic_config = config_class(**config_kwargs)
    except TypeError as exc:
        raise ActionCriticTrainingError(
            "could not construct ActionCriticConfig from the isolated trainer config"
        ) from exc
    model_class = getattr(
        sidecar, "ActionCriticSidecar", getattr(sidecar, "CompleteActionCritic", None)
    )
    if model_class is None:
        raise ActionCriticTrainingError("action critic sidecar has no model class")
    attempts = (
        ((critic_config,), {}),
        ((), {"config": critic_config}),
        ((), {"critic_config": critic_config}),
    )
    model: torch.nn.Module | None = None
    failures: list[str] = []
    for positional, keywords in attempts:
        try:
            candidate = model_class(*positional, **keywords)
        except TypeError as exc:
            failures.append(str(exc))
            continue
        if not isinstance(candidate, torch.nn.Module):
            raise ActionCriticTrainingError("CompleteActionCritic is not a torch module")
        model = candidate
        break
    if model is None:
        raise ActionCriticTrainingError(
            "could not construct CompleteActionCritic: " + " | ".join(failures)
        )
    return model, _config_to_dict(critic_config)


def _as_output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    elif hasattr(value, "as_tensor") and callable(value.as_tensor):
        result = value.as_tensor()
    else:
        raise ActionCriticTrainingError("critic forward result is not tensor-like")
    if result.numel() != len(OUTPUT_NAMES):
        raise ActionCriticTrainingError("critic must emit exactly eight outputs")
    return result.reshape(len(OUTPUT_NAMES))


def _mapping_output_tensor(value: Mapping[str, Any]) -> torch.Tensor:
    aliases = {
        "V_win": ("V_win", "v_win", "state_win_probability"),
        "Q_win": ("Q_win", "q_win", "chosen_action_win_probability"),
        "V_prize^1": ("V_prize^1", "v_prize_1", "V_prize_1"),
        "Q_prize^1": ("Q_prize^1", "q_prize_1", "Q_prize_1"),
        "V_prize^2": ("V_prize^2", "v_prize_2", "V_prize_2"),
        "Q_prize^2": ("Q_prize^2", "q_prize_2", "Q_prize_2"),
        "V_prize^3": ("V_prize^3", "v_prize_3", "V_prize_3"),
        "Q_prize^3": ("Q_prize^3", "q_prize_3", "Q_prize_3"),
    }
    values: list[torch.Tensor] = []
    for output_name in OUTPUT_NAMES:
        raw = None
        for alias in aliases[output_name]:
            if alias in value:
                raw = value[alias]
                break
        if raw is None:
            raise ActionCriticTrainingError(f"critic mapping misses {output_name}")
        if not isinstance(raw, torch.Tensor) or raw.numel() != 1:
            raise ActionCriticTrainingError(f"critic output {output_name} must be scalar tensor")
        values.append(raw.reshape(()))
    return torch.stack(values)


def _one_critic_prediction(
    model: torch.nn.Module, row: CompleteActionExample, device: torch.device
) -> torch.Tensor:
    menu = torch.tensor(row.first_stage_menu, dtype=torch.float32, device=device)
    selected = tuple(
        torch.tensor(value, dtype=torch.float32, device=device)
        for value in row.selected_stage_features
    )
    attempts = (
        ((), {"first_stage_menu": menu, "selected_stage_features": selected}),
        ((menu, selected), {}),
        ((), {"menu": menu, "selected_stage_features": selected}),
        ((), {"first_stage_features": menu, "selected_stage_features": selected}),
    )
    failures: list[str] = []
    output: Any = None
    for positional, keywords in attempts:
        try:
            output = model(*positional, **keywords)
        except TypeError as exc:
            failures.append(str(exc))
            continue
        break
    else:
        raise ActionCriticTrainingError(
            "CompleteActionCritic forward interface mismatch: " + " | ".join(failures)
        )
    if isinstance(output, Mapping):
        result = _mapping_output_tensor(output)
    else:
        result = _as_output_tensor(output)
    return result


def _padded_complete_action_inputs(
    rows: Sequence[CompleteActionExample], *, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Collate current-state and sealed complete-action inputs.

    The final four tensors are Q-only structure: selected legal position,
    legal count, and the explicitly recorded action-program prefix.  They
    come from the same pre-action overlay row as the selected feature vector;
    no successor or target-only field enters this batch.
    """
    if not rows:
        raise ActionCriticTrainingError("empty complete-action batch")
    # Reject structural abuse before using any declared token length to size a
    # tensor.  Sealed overlay rows normally pass the same checks in
    # ``_example_from_join``; this also protects direct test/validation calls.
    for row in rows:
        stage_count = len(row.selected_stage_features)
        if (
            row.stage_count != stage_count
            or stage_count > MAX_SELECTED_ACTION_STAGES
            or len(row.selected_option_indices) != stage_count
            or len(row.selected_legal_counts) != stage_count
            or len(row.selected_action_programs) != stage_count
        ):
            raise ActionCriticTrainingError("complete-action structural fields are misaligned")
        for selected_index, legal_count, action_program in zip(
            row.selected_option_indices,
            row.selected_legal_counts,
            row.selected_action_programs,
            strict=True,
        ):
            if (
                isinstance(selected_index, bool)
                or not isinstance(selected_index, int)
                or isinstance(legal_count, bool)
                or not isinstance(legal_count, int)
                or not 1 <= legal_count <= MAX_SELECTED_LEGAL_OPTIONS
                or not 0 <= selected_index < legal_count
                or len(action_program) > MAX_SELECTED_ACTION_PROGRAM_TOKENS
                or any(
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or not 0 <= token <= MAX_SELECTED_ACTION_TOKEN_VALUE
                    for token in action_program
                )
            ):
                raise ActionCriticTrainingError("selected action structure is invalid")
    max_menu = max(len(row.first_stage_menu) for row in rows)
    max_selected = max(len(row.selected_stage_features) for row in rows)
    max_program_tokens = max(
        1,
        max(
            len(program)
            for row in rows
            for program in row.selected_action_programs
        ),
    )
    menu = torch.zeros(
        (len(rows), max_menu, FEATURE_WIDTH), dtype=torch.float32, device=device
    )
    menu_mask = torch.zeros((len(rows), max_menu), dtype=torch.bool, device=device)
    selected = torch.zeros(
        (len(rows), max_selected, FEATURE_WIDTH), dtype=torch.float32, device=device
    )
    selected_mask = torch.zeros(
        (len(rows), max_selected), dtype=torch.bool, device=device
    )
    selected_option_indices = torch.zeros(
        (len(rows), max_selected), dtype=torch.int64, device=device
    )
    selected_legal_counts = torch.zeros(
        (len(rows), max_selected), dtype=torch.int64, device=device
    )
    selected_action_program_tokens = torch.zeros(
        (len(rows), max_selected, max_program_tokens),
        dtype=torch.int64,
        device=device,
    )
    selected_action_program_mask = torch.zeros(
        (len(rows), max_selected, max_program_tokens),
        dtype=torch.bool,
        device=device,
    )
    for index, row in enumerate(rows):
        stage_count = len(row.selected_stage_features)
        if (
            row.stage_count != stage_count
            or stage_count > MAX_SELECTED_ACTION_STAGES
            or len(row.selected_option_indices) != stage_count
            or len(row.selected_legal_counts) != stage_count
            or len(row.selected_action_programs) != stage_count
        ):
            raise ActionCriticTrainingError("complete-action structural fields are misaligned")
        menu[index, : len(row.first_stage_menu)] = torch.tensor(
            row.first_stage_menu, dtype=torch.float32, device=device
        )
        menu_mask[index, : len(row.first_stage_menu)] = True
        selected[index, :stage_count] = torch.tensor(
            row.selected_stage_features, dtype=torch.float32, device=device
        )
        selected_mask[index, :stage_count] = True
        for stage, (selected_index, legal_count, action_program) in enumerate(
            zip(
                row.selected_option_indices,
                row.selected_legal_counts,
                row.selected_action_programs,
                strict=True,
            )
        ):
            if (
                isinstance(selected_index, bool)
                or not isinstance(selected_index, int)
                or isinstance(legal_count, bool)
                or not isinstance(legal_count, int)
                or legal_count < 1
                or legal_count > MAX_SELECTED_LEGAL_OPTIONS
                or not 0 <= selected_index < legal_count
            ):
                raise ActionCriticTrainingError(
                    "selected action structural index/count is invalid"
                )
            if any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not 0 <= token <= MAX_SELECTED_ACTION_TOKEN_VALUE
                for token in action_program
            ):
                raise ActionCriticTrainingError(
                    "selected action program contains an invalid token"
                )
            if len(action_program) > MAX_SELECTED_ACTION_PROGRAM_TOKENS:
                raise ActionCriticTrainingError(
                    "selected action program exceeds the sealed token bound"
                )
            selected_option_indices[index, stage] = selected_index
            selected_legal_counts[index, stage] = legal_count
            if action_program:
                length = len(action_program)
                selected_action_program_tokens[index, stage, :length] = torch.tensor(
                    action_program, dtype=torch.int64, device=device
                )
                selected_action_program_mask[index, stage, :length] = True
    return (
        menu,
        menu_mask,
        selected,
        selected_mask,
        selected_option_indices,
        selected_legal_counts,
        selected_action_program_tokens,
        selected_action_program_mask,
    )


def _batch_tensor_output(value: Any, *, batch_size: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    elif hasattr(value, "values") and isinstance(value.values, torch.Tensor):
        result = value.values
    else:
        raise ActionCriticTrainingError("batched critic forward result is not tensor-like")
    if result.ndim != 2 or tuple(result.shape) != (batch_size, len(OUTPUT_NAMES)):
        raise ActionCriticTrainingError("critic must emit [complete_action_batch,8]")
    return result


def _win_outputs_are_logits() -> bool:
    sidecar = _require_sidecar_module()
    return hasattr(sidecar, "ActionCriticSidecar") and hasattr(
        sidecar, "ACTION_CRITIC_OUTPUT_NAMES"
    )


def _strict_sidecar_checkpoint_available() -> bool:
    sidecar = _require_sidecar_module()
    return all(
        callable(getattr(sidecar, name, None))
        for name in (
            "build_action_critic_checkpoint",
            "restore_action_critic_checkpoint",
        )
    ) and isinstance(
        getattr(sidecar, "ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA", None), str
    )


def _requires_sealed_action_structure() -> bool:
    """Whether this is the production sidecar rather than a test adapter."""

    sidecar = _require_sidecar_module()
    return isinstance(
        getattr(sidecar, "ACTION_CRITIC_SIDECAR_SCHEMA", None), str
    )


def _win_probabilities(predictions: torch.Tensor) -> torch.Tensor:
    wins = predictions[:, :2]
    return torch.sigmoid(wins) if _win_outputs_are_logits() else wins


def critic_predictions(
    model: torch.nn.Module,
    rows: Sequence[CompleteActionExample],
    *,
    device: torch.device,
) -> torch.Tensor:
    # The production sidecar accepts padded variable-length complete-action
    # batches.  Retain a one-program compatibility fallback while the sidecar
    # ABI is being staged; both paths keep optimizer units as complete actions.
    inputs = _padded_complete_action_inputs(rows, device=device)
    try:
        output = model(*inputs)
    except TypeError as structured_error:
        if _requires_sealed_action_structure():
            raise ActionCriticTrainingError(
                "production ActionCriticSidecar rejected the sealed "
                "chosen-index/legal-count/action-program ABI"
            ) from structured_error
        # Compatibility is only for a test/staging sidecar that predates the
        # sealed action-structure ABI.  The production sidecar must accept all
        # eight tensors above; it never silently drops action identity.
        try:
            output = model(*inputs[:4])
        except TypeError:
            try:
                predictions = [_one_critic_prediction(model, row, device) for row in rows]
            except TypeError as exc:
                raise ActionCriticTrainingError(
                    "CompleteActionCritic forward interface mismatch; expected sealed "
                    "action-structure ABI"
                ) from structured_error
            result = torch.stack(predictions, dim=0)
        else:
            result = _batch_tensor_output(output, batch_size=len(rows))
    else:
        result = _batch_tensor_output(output, batch_size=len(rows))
    if not bool(torch.isfinite(result).all()):
        raise ActionCriticTrainingError("critic emitted non-finite prediction")
    prizes = result[:, 2:]
    if bool((prizes < -1.0 - EPSILON).any()) or bool((prizes > 1.0 + EPSILON).any()):
        raise ActionCriticTrainingError("critic prize outputs must be bounded in [-1,+1]")
    return result


def critic_loss(
    predictions: torch.Tensor, targets: TargetBatch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked BCE/MSE loss for all eight sidecar outputs.

    Masks are availability masks, not replay or importance weights.  The two
    win outputs share the observed recorded win label; state and chosen-action
    prize heads share each realized same-seat interval target for their horizon.
    """
    if predictions.ndim != 2 or predictions.shape[1] != len(OUTPUT_NAMES):
        raise ActionCriticTrainingError("critic prediction shape is not [batch,8]")
    sidecar = _require_sidecar_module()
    helper = getattr(sidecar, "action_critic_loss", None)
    # The current sidecar helper is exact for fully observed terminal batches.
    # If a raw episode masks a terminal result, use the same masked BCE and
    # SmoothL1 semantics below rather than silently treating it as a loss.
    if (
        _win_outputs_are_logits()
        and callable(helper)
        and bool(targets.win_mask.all())
    ):
        detailed = helper(
            predictions,
            win_targets=targets.win_target,
            prize_targets=targets.prize_target,
            prize_mask=targets.prize_mask,
        )
        components = {
            "V_win_bce": detailed.v_win,
            "Q_win_bce": detailed.q_win,
            "win": detailed.win,
            "prize": detailed.prize,
            "VQ_prize^1": detailed.prize_by_horizon[0],
            "VQ_prize^2": detailed.prize_by_horizon[1],
            "VQ_prize^3": detailed.prize_by_horizon[2],
            "sidecar_loss_helper_used": detailed.total.detach() * 0.0,
        }
        return detailed.total, components
    components: dict[str, torch.Tensor] = {}
    wins_are_logits = _win_outputs_are_logits()
    win_losses: list[torch.Tensor] = []
    if bool(targets.win_mask.any()):
        target = targets.win_target[targets.win_mask]
        for name in ("V_win", "Q_win"):
            values = predictions[:, OUTPUT_INDEX[name]][targets.win_mask]
            loss = (
                F.binary_cross_entropy_with_logits(values, target)
                if wins_are_logits
                else F.binary_cross_entropy(values.clamp(EPSILON, 1.0 - EPSILON), target)
            )
            components[f"{name}_bce"] = loss
            win_losses.append(loss)
    prize_losses: list[torch.Tensor] = []
    for horizon in (1, 2, 3):
        mask = targets.prize_mask[:, horizon - 1]
        if not bool(mask.any()):
            continue
        target = targets.prize_target[:, horizon - 1][mask]
        for prefix in ("V", "Q"):
            name = f"{prefix}_prize^{horizon}"
            components[f"{name}_smooth_l1"] = F.smooth_l1_loss(
                predictions[:, OUTPUT_INDEX[name]][mask], target
            )
        prize_losses.append(
            (components[f"V_prize^{horizon}_smooth_l1"] + components[f"Q_prize^{horizon}_smooth_l1"])
            * 0.5
        )
    if not win_losses and not prize_losses:
        raise ActionCriticTrainingError("complete-action batch has no available targets")
    win_total = torch.stack(win_losses).mean() if win_losses else predictions[:, :2].sum() * 0.0
    prize_total = torch.stack(prize_losses).mean() if prize_losses else predictions[:, 2:].sum() * 0.0
    components["win"] = win_total
    components["prize"] = prize_total
    components["sidecar_loss_helper_used"] = predictions.sum() * 0.0
    total = win_total + prize_total
    if not bool(torch.isfinite(total)):
        raise ActionCriticTrainingError("critic loss is non-finite")
    return total, components


@dataclass
class _RunningTargetMetric:
    count: int = 0
    sum_target: float = 0.0
    sum_target_sq: float = 0.0
    sum_squared_error: float = 0.0

    def add(self, prediction: float, target: float) -> None:
        self.count += 1
        self.sum_target += target
        self.sum_target_sq += target * target
        self.sum_squared_error += (prediction - target) * (prediction - target)

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {"available": False, "count": 0}
        mean = self.sum_target / self.count
        mse = self.sum_squared_error / self.count
        zero = self.sum_target_sq / self.count
        empirical_mean = max(0.0, self.sum_target_sq / self.count - mean * mean)
        return {
            "available": True,
            "count": self.count,
            "mse": mse,
            "zero_baseline_mse": zero,
            "empirical_mean_baseline_mse": empirical_mean,
            "empirical_target_mean": mean,
            "better_than_zero_baseline": mse < zero,
            "better_than_empirical_mean_baseline": mse < empirical_mean,
        }


@dataclass
class _CalibrationMetric:
    count: int = 0
    brier_sum: float = 0.0
    nll_sum: float = 0.0
    bins: list[list[float]] = dataclasses.field(
        default_factory=lambda: [[0.0, 0.0, 0.0] for _ in range(10)]
    )

    def add(self, prediction: float, target: float) -> None:
        prediction = min(1.0 - EPSILON, max(EPSILON, prediction))
        self.count += 1
        self.brier_sum += (prediction - target) ** 2
        self.nll_sum += -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))
        index = min(9, int(prediction * 10.0))
        self.bins[index][0] += 1.0
        self.bins[index][1] += prediction
        self.bins[index][2] += target

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {"available": False, "count": 0}
        ece = 0.0
        rows: list[dict[str, float]] = []
        for index, (count, pred_sum, target_sum) in enumerate(self.bins):
            if not count:
                continue
            mean_prediction = pred_sum / count
            mean_target = target_sum / count
            ece += (count / self.count) * abs(mean_prediction - mean_target)
            rows.append(
                {
                    "bin": index,
                    "count": int(count),
                    "mean_prediction": mean_prediction,
                    "empirical_win_rate": mean_target,
                }
            )
        return {
            "available": True,
            "count": self.count,
            "brier": self.brier_sum / self.count,
            "negative_log_likelihood": self.nll_sum / self.count,
            "expected_calibration_error": ece,
            "bins": rows,
        }


class MetricAccumulator:
    """Streaming validation/training metrics; no prediction corpus is retained."""

    def __init__(self) -> None:
        self.actions = 0
        self.stages = 0
        self.days: set[str] = set()
        self.terminal_available = 0
        self.win_available = 0
        self.prize_available = [0, 0, 0]
        self.component_sum: defaultdict[str, float] = defaultdict(float)
        self.component_count: defaultdict[str, int] = defaultdict(int)
        self.metrics = {name: _RunningTargetMetric() for name in OUTPUT_NAMES}
        self.calibration = {"V_win": _CalibrationMetric(), "Q_win": _CalibrationMetric()}

    def update(
        self,
        rows: Sequence[CompleteActionExample],
        predictions: torch.Tensor,
        targets: TargetBatch,
        components: Mapping[str, torch.Tensor],
    ) -> None:
        values = predictions.detach().cpu().tolist()
        win_values = _win_probabilities(predictions).detach().cpu().tolist()
        wins = targets.win_target.detach().cpu().tolist()
        win_masks = targets.win_mask.detach().cpu().tolist()
        prizes = targets.prize_target.detach().cpu().tolist()
        prize_masks = targets.prize_mask.detach().cpu().tolist()
        z_masks = targets.z_mask.detach().cpu().tolist()
        for index, row in enumerate(rows):
            self.actions += 1
            self.stages += row.stage_count
            self.days.add(row.utc_day)
            self.terminal_available += int(bool(z_masks[index]))
            if bool(win_masks[index]):
                self.win_available += 1
                target = float(wins[index])
                for name in ("V_win", "Q_win"):
                    prediction = float(win_values[index][OUTPUT_INDEX[name]])
                    self.metrics[name].add(prediction, target)
                    self.calibration[name].add(prediction, target)
            for horizon in (1, 2, 3):
                if bool(prize_masks[index][horizon - 1]):
                    self.prize_available[horizon - 1] += 1
                    target = float(prizes[index][horizon - 1])
                    for prefix in ("V", "Q"):
                        name = f"{prefix}_prize^{horizon}"
                        self.metrics[name].add(
                            float(values[index][OUTPUT_INDEX[name]]), target
                        )
        for name, loss in components.items():
            self.component_sum[name] += float(loss.detach().cpu())
            self.component_count[name] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "complete_actions": self.actions,
            "factorized_stages": self.stages,
            "days": sorted(self.days),
            "coverage": {
                "terminal_z": {
                    "available": self.terminal_available,
                    "total": self.actions,
                    "rate": self.terminal_available / max(1, self.actions),
                },
                "win": {
                    "available": self.win_available,
                    "total": self.actions,
                    "rate": self.win_available / max(1, self.actions),
                },
                "prize_horizons": {
                    str(horizon): {
                        "available": self.prize_available[horizon - 1],
                        "total": self.actions,
                        "rate": self.prize_available[horizon - 1] / max(1, self.actions),
                    }
                    for horizon in (1, 2, 3)
                },
            },
            "loss_components": {
                name: self.component_sum[name] / self.component_count[name]
                for name in sorted(self.component_sum)
                if self.component_count[name]
            },
            "baselines": {name: metric.summary() for name, metric in self.metrics.items()},
            "calibration": {
                name: metric.summary() for name, metric in self.calibration.items()
            },
        }


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    if value not in {"cpu", "mps"}:
        raise ActionCriticTrainingError("isolated critic trainer permits only CPU or MPS")
    if value == "mps" and not torch.backends.mps.is_available():
        raise ActionCriticTrainingError("MPS was requested but is unavailable")
    return torch.device(value)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def source_binding(
    *,
    overlay_manifest: Path,
    overlay_manifest_sha256: str,
    base_pack_root: Path,
    base_completion_sha256: str,
    target_manifest: Path,
    target_manifest_sha256: str,
    contract: Path,
    contract_sha256: str,
    contract_goal_revision: int,
    training_view: Mapping[str, Any] | None = None,
    target_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = {
        "complete_action_overlay_manifest_path": str(overlay_manifest.resolve()),
        "complete_action_overlay_manifest_sha256": overlay_manifest_sha256,
        "base_pack_root": str(base_pack_root.resolve()),
        "base_pack_completion_sha256": base_completion_sha256,
        "target_overlay_manifest_path": str(target_manifest.resolve()),
        "target_overlay_manifest_sha256": target_manifest_sha256,
        "contract_path": str(contract.resolve()),
        "contract_sha256": contract_sha256,
        "current_contract_goal_revision": int(contract_goal_revision),
        "critic_semantic_owner_goal_revision": CRITIC_SEMANTIC_OWNER_REVISION,
    }
    if training_view is not None:
        binding["transferred_training_view"] = {
            "pointer_path": str(training_view["pointer_path"]),
            "pointer_sha256": str(training_view["pointer_sha256"]),
            "completion_path": str(training_view["completion_path"]),
            "completion_sha256": str(training_view["completion_sha256"]),
            "base_completion_path": str(training_view["base_completion_path"]),
        }
    if target_view is not None:
        binding["transferred_target_view"] = {
            "pointer_path": str(target_view["pointer_path"]),
            "pointer_sha256": str(target_view["pointer_sha256"]),
            "completion_path": str(target_view["completion_path"]),
            "completion_sha256": str(target_view["completion_sha256"]),
            "target_set_receipt_path": str(target_view["target_set_receipt_path"]),
            "target_set_receipt_sha256": str(target_view["target_set_receipt_sha256"]),
        }
    return binding


def open_sealed_inputs(args: argparse.Namespace) -> tuple[Recent20RTPDataset, TargetOverlay, dict[str, list[str]], dict[str, Any]]:
    allow_test_fixture = bool(getattr(args, "test_allow_noncanonical_split", False))
    production = not allow_test_fixture
    if bool(getattr(args, "skip_input_shard_sha256", False)):
        raise ActionCriticTrainingError(
            "--skip-input-shard-sha256 is no longer a production input option"
        )
    test_skip_hashes = bool(getattr(args, "test_skip_input_shard_sha256", False))
    if test_skip_hashes and not allow_test_fixture:
        raise ActionCriticTrainingError("input SHA skipping is limited to explicit test fixtures")
    training_view_arg = getattr(args, "training_view", None)
    training_view: dict[str, Any] | None = None
    if training_view_arg is not None:
        training_view = _resolve_transferred_training_view(
            pointer=training_view_arg,
            expected_sha256=str(getattr(args, "training_view_sha256", "") or ""),
            production=production,
        )
        overlay_manifest = Path(training_view["overlay_manifest"])
        base_root = Path(training_view["base_pack_root"])
        base_completion_path: Path | None = Path(training_view["base_completion_path"])
    else:
        if production:
            raise ActionCriticTrainingError(
                "production critic training requires a receipt-bound --training-view"
            )
        overlay_manifest = _regular_file(args.overlay_manifest, label="test overlay manifest")
        base_root = _regular_directory(args.base_pack_root, label="test base-pack root")
        base_completion_path = None
    overlay_sha = _require_sha256(args.overlay_manifest_sha256, label="overlay manifest SHA-256")
    base_completion_sha = _require_sha256(args.base_completion_sha256, label="base completion SHA-256")
    target_sha = _require_sha256(args.target_manifest_sha256, label="target manifest SHA-256")
    if training_view is not None and (
        overlay_sha != training_view["overlay_manifest_sha256"]
        or base_completion_sha != training_view["base_completion_sha256"]
    ):
        raise ActionCriticTrainingError("CLI input identities disagree with the transferred training view")
    if sha256_file(overlay_manifest) != overlay_sha:
        raise ActionCriticTrainingError("complete-action overlay manifest digest mismatch")
    contract, contract_sha, contract_goal_revision = _load_bound_contract(
        args.contract,
        expected_sha256=str(getattr(args, "contract_sha256", "") or ""),
        production=production,
    )
    target_view_arg = getattr(args, "target_view", None)
    target_view: dict[str, Any] | None = None
    if target_view_arg is not None:
        target_view = _resolve_transferred_target_view(
            pointer=target_view_arg,
            expected_sha256=str(getattr(args, "target_view_sha256", "") or ""),
            expected_contract_sha256=contract_sha,
            expected_base_completion_sha256=base_completion_sha,
            expected_overlay_manifest_sha256=overlay_sha,
            production=production,
        )
        target_manifest = Path(target_view["target_manifest"])
        if target_sha != target_view["target_manifest_sha256"]:
            raise ActionCriticTrainingError("CLI target manifest SHA disagrees with target view")
    else:
        target_manifest = _regular_file(args.target_manifest, label="target manifest")
    verify_inputs = not test_skip_hashes
    dataset = Recent20RTPDataset(
        overlay_manifest,
        base_pack_root=base_root,
        base_completion_path=base_completion_path,
        expected_manifest_sha256=overlay_sha,
        expected_base_completion_sha256=base_completion_sha,
        verify_overlay_shards=verify_inputs,
        verify_base_shards=verify_inputs,
    )
    overlay_day_sha256s: dict[str, str] = {}
    for shard in dataset.shards:
        day = str(shard.get("utc_day") or "")
        digest = str(shard.get("sha256") or "")
        if not day or day in overlay_day_sha256s:
            raise ActionCriticTrainingError("base overlay day inventory is malformed")
        overlay_day_sha256s[day] = _require_sha256(
            digest, label="complete-action overlay day SHA-256"
        )
    targets = TargetOverlay(
        target_manifest,
        expected_sha256=target_sha,
        expected_overlay_manifest_sha256=overlay_sha,
        expected_base_completion_sha256=base_completion_sha,
        expected_contract_sha256=contract_sha,
        expected_contract_goal_revision=contract_goal_revision,
        expected_overlay_day_sha256s=overlay_day_sha256s,
        allow_test_fixture=allow_test_fixture,
        verify_shards=verify_inputs,
    )
    split_days = assert_exact_split_contract(
        dataset,
        targets,
        test_allow_noncanonical_split=allow_test_fixture,
    )
    binding = source_binding(
        overlay_manifest=overlay_manifest,
        overlay_manifest_sha256=overlay_sha,
        base_pack_root=base_root,
        base_completion_sha256=base_completion_sha,
        target_manifest=target_manifest,
        target_manifest_sha256=target_sha,
        contract=contract,
        contract_sha256=contract_sha,
        contract_goal_revision=contract_goal_revision,
        training_view=training_view,
        target_view=target_view,
    )
    return dataset, targets, split_days, binding


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_completed: int,
    optimizer_steps: int,
    model_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    source: Mapping[str, Any],
    split_days: Mapping[str, Any],
    epoch_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if _strict_sidecar_checkpoint_available():
        sidecar = _require_sidecar_module()
        # The sidecar owns the strict standalone tensor payload.  Trainer
        # progress and source identities stay JSON-only metadata, which makes
        # it impossible to smuggle a policy state dict into this checkpoint.
        return dict(
            sidecar.build_action_critic_checkpoint(
                model,
                optimizer=optimizer,
                training_state={
                    "trainer_checkpoint_schema": CHECKPOINT_SCHEMA,
                    "saved_at_unix_seconds": time.time(),
                    "epoch_completed": int(epoch_completed),
                    "optimizer_steps": int(optimizer_steps),
                    "trainer_config": dict(trainer_config),
                    "split_days": {str(key): list(value) for key, value in split_days.items()},
                    "epoch_history": [dict(item) for item in epoch_history],
                    "evaluation_split_consumed": False,
                },
                metadata={
                    "source_binding": dict(source),
                    "runtime_or_policy_attachment": False,
                    "search_rtp_or_mcts_used": False,
                    "replay_weights_changed": False,
                    "policy_model_state_dict_changed": False,
                },
            )
        )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "saved_at_unix_seconds": time.time(),
        "epoch_completed": int(epoch_completed),
        "optimizer_steps": int(optimizer_steps),
        "model_config": dict(model_config),
        "trainer_config": dict(trainer_config),
        "source_binding": dict(source),
        "split_days": {str(key): list(value) for key, value in split_days.items()},
        "epoch_history": [dict(item) for item in epoch_history],
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "runtime_or_policy_attachment": False,
        "search_rtp_or_mcts_used": False,
        "replay_weights_changed": False,
        "evaluation_split_consumed": False,
    }


def load_resume_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    source: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ActionCriticTrainingError(f"cannot load critic resume checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ActionCriticTrainingError("resume checkpoint is not an object")
    sidecar = _require_sidecar_module()
    strict_schema = str(getattr(sidecar, "ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA", ""))
    if _strict_sidecar_checkpoint_available() and payload.get("schema") == strict_schema:
        try:
            restored = sidecar.restore_action_critic_checkpoint(
                model, payload, optimizer=optimizer
            )
        except Exception as exc:  # Sidecar owns exact checkpoint validation.
            raise ActionCriticTrainingError("resume sidecar checkpoint is incompatible") from exc
        training_state = _mapping(restored.training_state, label="resume training state")
        metadata = _mapping(restored.metadata, label="resume metadata")
        if dict(metadata.get("source_binding") or {}) != dict(source):
            raise ActionCriticTrainingError("resume checkpoint source binding mismatch")
        saved_config = dict(training_state.get("trainer_config") or {})
        history = training_state.get("epoch_history") or []
        epoch_completed = int(training_state.get("epoch_completed", -1))
        optimizer_steps = int(training_state.get("optimizer_steps", -1))
    else:
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ActionCriticTrainingError("resume checkpoint schema drifted")
        if dict(payload.get("source_binding") or {}) != dict(source):
            raise ActionCriticTrainingError("resume checkpoint source binding mismatch")
        saved_config = dict(payload.get("trainer_config") or {})
        try:
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ActionCriticTrainingError("resume checkpoint state is incompatible") from exc
        history = payload.get("epoch_history") or []
        epoch_completed = int(payload.get("epoch_completed", -1))
        optimizer_steps = int(payload.get("optimizer_steps", -1))
    for key in ("batch_size", "learning_rate", "weight_decay", "seed"):
        if saved_config.get(key) != trainer_config.get(key):
            raise ActionCriticTrainingError(f"resume trainer config mismatch: {key}")
    model.to(device=device, dtype=torch.float32)
    _move_optimizer_state(optimizer, device)
    if not isinstance(history, list) or not all(isinstance(row, Mapping) for row in history):
        raise ActionCriticTrainingError("resume checkpoint history is malformed")
    if epoch_completed < 0 or optimizer_steps < 0:
        raise ActionCriticTrainingError("resume checkpoint counters are malformed")
    return epoch_completed, optimizer_steps, [dict(row) for row in history]


def _run_batches(
    model: torch.nn.Module,
    rows: Iterable[CompleteActionExample],
    *,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> tuple[MetricAccumulator, int]:
    accumulator = MetricAccumulator()
    optimizer_steps = 0
    training = optimizer is not None
    model.train(training)
    for batch in batched(rows, batch_size=batch_size):
        targets = _target_batch(batch, device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            predictions = critic_predictions(model, batch, device=device)
            loss, components = critic_loss(predictions, targets)
            loss.backward()
            if not bool(
                all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            ):
                raise ActionCriticTrainingError("critic gradient is non-finite")
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer_steps += 1
        else:
            with torch.no_grad():
                predictions = critic_predictions(model, batch, device=device)
                loss, components = critic_loss(predictions, targets)
        accumulator.update(batch, predictions, targets, {"total": loss, **components})
    if accumulator.actions == 0:
        raise ActionCriticTrainingError("sealed split produced no complete action examples")
    return accumulator, optimizer_steps


def _activation_metric_summary(validation: Mapping[str, Any], *, full_validation: bool) -> dict[str, Any]:
    baselines = _mapping(validation.get("baselines"), label="validation baselines")
    calibration = _mapping(validation.get("calibration"), label="validation calibration")
    coverage = _mapping(validation.get("coverage"), label="validation coverage")
    h1 = _mapping(_mapping(coverage.get("prize_horizons"), label="prize coverage").get("1"), label="h1 coverage")
    h1_baselines = (
        _mapping(baselines.get("V_prize^1"), label="V prize h1 baseline"),
        _mapping(baselines.get("Q_prize^1"), label="Q prize h1 baseline"),
    )
    win_calibration = tuple(
        _mapping(calibration.get(name), label=f"{name} calibration")
        for name in ("V_win", "Q_win")
    )
    finite_and_covered = (
        int(validation.get("complete_actions", 0)) > 0
        and int(h1.get("available", 0)) > 0
        and all(bool(metric.get("available")) for metric in win_calibration)
    )
    return {
        "full_validation_split_consumed": bool(full_validation),
        "finite_and_covered": finite_and_covered,
        "win_calibration_reported": all(bool(metric.get("available")) for metric in win_calibration),
        "masked_h1_better_than_zero_baseline": all(
            bool(metric.get("better_than_zero_baseline")) for metric in h1_baselines
        ),
        "heldout_prediction_better_than_empirical_mean_baseline": all(
            bool(
                _mapping(baselines.get(name), label=f"{name} baseline").get(
                    "better_than_empirical_mean_baseline"
                )
            )
            for name in OUTPUT_NAMES
            if bool(_mapping(baselines.get(name), label=f"{name} baseline").get("available"))
        ),
        # Paired seat/opponent and AWR checks require the later isolated
        # canary materialization.  This trainer reports evidence only and
        # deliberately cannot self-approve activation.
        "paired_seat_and_opponent_noninferiority_available": False,
        "awr_ess_and_clip_rate_available": False,
        "activation_eligible": False,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.epochs < 1 or args.batch_size < 1 or args.shuffle_buffer < 1:
        raise ActionCriticTrainingError("epochs, batch size, and shuffle buffer must be positive")
    if args.hidden_width < 1 or args.max_train_programs < 0 or args.max_validation_programs < 0:
        raise ActionCriticTrainingError("hidden width and program limits are invalid")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ActionCriticTrainingError("learning rate must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ActionCriticTrainingError("weight decay must be finite and nonnegative")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ActionCriticTrainingError("gradient clip must be finite and nonnegative")
    dataset, targets, split_days, binding = open_sealed_inputs(args)
    device = resolve_device(args.device)
    _seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_config = {
        "schema": TRAINER_SCHEMA,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "shuffle_buffer": int(args.shuffle_buffer),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "max_train_programs": int(args.max_train_programs),
        "max_validation_programs": int(args.max_validation_programs),
        "fp32": True,
        "complete_action_batches": True,
        "bounded_shuffle_only": True,
        "evaluation_split_consumed": False,
    }
    resume_path = Path(args.resume).expanduser().resolve() if args.resume else output_dir / "latest.pt"
    resume_payload: Mapping[str, Any] | None = None
    if resume_path.exists():
        try:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ActionCriticTrainingError("cannot inspect resume checkpoint") from exc
        if not isinstance(resume_payload, Mapping):
            raise ActionCriticTrainingError("resume checkpoint is not an object")
        sidecar = _require_sidecar_module()
        strict_schema = str(
            getattr(sidecar, "ACTION_CRITIC_SIDECAR_CHECKPOINT_SCHEMA", "")
        )
        config_key = "config" if resume_payload.get("schema") == strict_schema else "model_config"
        saved_model_config = _mapping(
            resume_payload.get(config_key), label="resume model config"
        )
        model, model_config = build_critic(saved_config=saved_model_config)
    else:
        model, model_config = build_critic(
            hidden_width=int(args.hidden_width), dropout=float(args.dropout)
        )
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    epoch_completed = 0
    optimizer_steps = 0
    epoch_history: list[dict[str, Any]] = []
    if resume_payload is not None:
        epoch_completed, optimizer_steps, epoch_history = load_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            source=binding,
            trainer_config=trainer_config,
        )
    for epoch in range(epoch_completed + 1, int(args.epochs) + 1):
        rows = bounded_shuffle(
            iter_complete_action_examples(
                dataset,
                targets,
                split="train",
                max_programs=int(args.max_train_programs),
            ),
            buffer_size=int(args.shuffle_buffer),
            seed=int(args.seed) + epoch,
        )
        train_metrics, steps = _run_batches(
            model,
            rows,
            device=device,
            batch_size=int(args.batch_size),
            optimizer=optimizer,
            grad_clip=float(args.grad_clip),
        )
        optimizer_steps += steps
        validation_metrics, _ = _run_batches(
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
        epoch_history.append(summary)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch_completed=epoch,
            optimizer_steps=optimizer_steps,
            model_config=model_config,
            trainer_config=trainer_config,
            source=binding,
            split_days=split_days,
            epoch_history=epoch_history,
        )
        _atomic_torch_save(output_dir / "latest.pt", payload)
        atomic_write_json(
            output_dir / "latest.metrics.json",
            {
                "schema": RECEIPT_SCHEMA,
                "status": "in_progress",
                "source_binding": binding,
                "model_config": model_config,
                "trainer_config": trainer_config,
                "epoch": epoch,
                "summary": summary,
            },
        )
        print(json.dumps({"phase": "epoch", **summary}, sort_keys=True), flush=True)
    final_checkpoint = output_dir / "action_critic.pt"
    final_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch_completed=int(args.epochs),
        optimizer_steps=optimizer_steps,
        model_config=model_config,
        trainer_config=trainer_config,
        source=binding,
        split_days=split_days,
        epoch_history=epoch_history,
    )
    _atomic_torch_save(final_checkpoint, final_payload)
    final_checkpoint_sha = sha256_file(final_checkpoint)
    validation = epoch_history[-1]["validation"]
    completion = {
        "schema": RECEIPT_SCHEMA,
        "status": "completed_shadow_only",
        "source_binding": binding,
        "source_binding_sha256": _sha256_json(binding),
        "split_days": split_days,
        "model_config": model_config,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainer_config": trainer_config,
        "epochs": int(args.epochs),
        "optimizer_steps": optimizer_steps,
        "epoch_history": epoch_history,
        "validation": validation,
        "validation_gates": _activation_metric_summary(
            validation,
            full_validation=not bool(args.max_validation_programs),
        ),
        "checkpoint_path": str(final_checkpoint),
        "checkpoint_sha256": final_checkpoint_sha,
        "checkpoint_size_bytes": final_checkpoint.stat().st_size,
        "resume_checkpoint_path": str(output_dir / "latest.pt"),
        "runtime_or_policy_attachment": False,
        "policy_model_state_dict_changed": False,
        "existing_value_head_changed": False,
        "existing_action_q_head_changed": False,
        "replay_weights_changed": False,
        "search_rtp_mcts_or_simulator_branching_used": False,
        "evaluation_split_consumed": False,
        "activation_eligible": False,
        "elapsed_seconds": time.time() - started,
        "completed_at_unix_seconds": time.time(),
    }
    receipt_path = output_dir / "training-receipt.json"
    receipt_sha = atomic_write_json(receipt_path, completion)
    result = {
        "checkpoint_path": str(final_checkpoint),
        "checkpoint_sha256": final_checkpoint_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "validation": validation,
        "optimizer_steps": optimizer_steps,
    }
    print(json.dumps({"phase": "complete", **result}, sort_keys=True), flush=True)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--base-pack-root", type=Path, required=True)
    parser.add_argument("--base-completion-sha256", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest-sha256", required=True)
    parser.add_argument("--target-view", type=Path, default=None)
    parser.add_argument("--target-view-sha256", default="")
    parser.add_argument("--contract", type=Path, default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json")
    parser.add_argument("--contract-sha256", default="")
    parser.add_argument("--training-view", type=Path, default=None)
    parser.add_argument("--training-view-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=32220)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-train-programs", type=int, default=0)
    parser.add_argument("--max-validation-programs", type=int, default=0)
    # This is solely for tiny hermetic unit fixtures.  Production runs retain
    # the exact sealed 14/3/3 day inventory and record this false in receipts.
    parser.add_argument("--test-allow-noncanonical-split", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-skip-input-shard-sha256", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        train(args)
    except (ActionCriticTrainingError, Recent20OverlayError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
