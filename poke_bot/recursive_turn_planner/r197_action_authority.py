"""Fail-closed action boundary for the sealed r198 r197 evaluator.

The r197 sidecar remains shadow-only.  A bridge may actively select with it
only after either the normal serving-promotion path has validated it, or the
sealed three-arm evaluator supplies a narrowly scoped
``evaluation_action_execution`` context.  The latter is intentionally *not*
serving/action authority: it is a one-child evaluation exception bound to the
immutable manifest, authority, candidate snapshot, arm fence, cell, nonce,
and live process identity.

This module has no engine, selector, training, or promotion side effects.  It
is deliberately standard-library-only so both the policy bridge and focused
tests can use the same verification code.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EVALUATION_ACTION_EXECUTION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluation_action_execution/v1"
)
EVALUATION_ACTION_FENCE_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluator_arm_runtime_contract/v1"
)
EVALUATION_AUTHORITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_authorization/v1"
)
EVALUATION_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_manifest/v2"
)
CANDIDATE_RUNTIME_CONTRACT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluation_candidate_snapshot/v1"
)

R198_CANDIDATE_ID = (
    "r197-bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_PARENT_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R198_SIDECAR_SHA256 = (
    "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31"
)
R198_PROFILE = "pure_rl_r197"
R198_MAX_NEURAL_PASSES = 256
R198_MAX_ACTION_COMBOS = 1024
R198_DIRECT_ARM = "direct_bridge_recursive_disabled"
R198_RECURSIVE_ARM = "recursive_rtp"

RUNTIME_CONTRACT_ENV = "POKEBOT_R198_EVAL_RUNTIME_CONTRACT"
RUNTIME_CONTRACT_SHA256_ENV = "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256"
ACTION_FENCE_ENV = "POKEBOT_R198_EVAL_ACTION_FENCE"
ACTION_FENCE_SHA256_ENV = "POKEBOT_R198_EVAL_ACTION_FENCE_SHA256"
LAUNCH_NONCE_ENV = "POKEBOT_R198_EVAL_LAUNCH_NONCE"
PROCESS_ID_ENV = "POKEBOT_R198_EVAL_PROCESS_ID"
PROCESS_START_TICKS_ENV = "POKEBOT_R198_EVAL_PROCESS_START_TICKS"


class R197ActionAuthorityError(RuntimeError):
    """Raised when an r197 sidecar tries to select without narrow authority."""


def _error(detail: str) -> R197ActionAuthorityError:
    return R197ActionAuthorityError(
        "pure_rl_r197 active selection requires serving-qualified promotion "
        "or sealed evaluation_action_execution: " + detail
    )


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error("evaluation_action_execution is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _copy_json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} is not canonical JSON") from exc
    if not isinstance(copied, dict):  # Defensive after a successful mapping check.
        raise _error(f"{label} must be an object")
    return copied


def _physical_regular_file(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise _error(f"{label}.path is required")
    raw = Path(path_value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise _error(f"{label}.path must be an absolute physical path")
    path = Path(os.path.abspath(os.fspath(raw)))
    current = Path(path.anchor)
    for index, component in enumerate(path.parts[1:]):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise _error(f"{label} does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise _error(f"{label} traverses a symbolic link")
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise _error(f"{label} has a non-directory ancestor")
    try:
        final = os.lstat(path)
    except OSError as exc:
        raise _error(f"{label} does not exist") from exc
    if not stat.S_ISREG(final.st_mode):
        raise _error(f"{label} is not a regular file")
    return path


def _read_immutable_bytes(path_value: Any, label: str) -> tuple[Path, bytes, os.stat_result]:
    """Open one physical 0444 file and retain the exact bytes that were hashed.

    Do not verify a pathname and then reopen it to parse JSON: a replacement
    between those operations would create a hash-to-read race.  ``O_NOFOLLOW``
    closes the final-component link race where supported, while the lstat/fstat
    inode comparison keeps the fallback fail-closed on platforms without it.
    """

    path = _physical_regular_file(path_value, label)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise _error(f"{label} platform cannot make a no-follow physical open")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        # Walk from the root through directory file descriptors rather than
        # reopening the checked pathname.  This keeps every ancestor physical
        # even if a concurrent actor swaps a directory after ``lstat``.
        directory_fd = os.open(path.anchor, directory_flags)
        for component in path.parts[1:-1]:
            next_directory_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_directory_fd
        before_path = os.lstat(path)
        descriptor = os.open(path.name, file_flags, dir_fd=directory_fd)
    except OSError as exc:
        raise _error(f"cannot open {label} without following links") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != before_path.st_dev
            or before.st_ino != before_path.st_ino
        ):
            raise _error(f"{label} changed while opening")
        if stat.S_IMODE(before.st_mode) != 0o444:
            raise _error(f"{label} must be immutable mode 0444")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _error(f"cannot read {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or stat.S_IMODE(before.st_mode) != stat.S_IMODE(after.st_mode)
        or len(content) != before.st_size
    ):
        raise _error(f"{label} changed while being read")
    return path, content, before


def _immutable_identity(value: Any, label: str) -> dict[str, Any]:
    raw = _copy_json_mapping(value, label)
    path, content, metadata = _read_immutable_bytes(raw.get("path"), label)
    expected = raw.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 71
        or not expected.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in expected[7:])
    ):
        raise _error(f"{label}.sha256 is invalid")
    observed = "sha256:" + hashlib.sha256(content).hexdigest()
    if observed != expected:
        raise _error(f"{label} checksum changed")
    size = len(content)
    if "bytes" in raw and (
        isinstance(raw["bytes"], bool) or not isinstance(raw["bytes"], int) or raw["bytes"] != size
    ):
        raise _error(f"{label} byte count changed")
    mode = stat.S_IMODE(metadata.st_mode)
    if "mode" in raw and raw["mode"] != mode:
        raise _error(f"{label} mode changed")
    return {
        "path": str(path),
        "sha256": observed,
        "bytes": size,
        "mode": mode,
        "_content": content,
    }


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "sha256", "bytes"))


def _json_payload(identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    content = identity.get("_content")
    if not isinstance(content, bytes):
        raise _error(f"{label} did not retain verified bytes")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(f"cannot read {label}") from exc
    if not isinstance(payload, dict):
        raise _error(f"{label} must contain an object")
    return payload


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{label} is required")
    return value


def _require_bool_fields(
    value: Mapping[str, Any],
    label: str,
    *,
    include_runtime_flags: bool = False,
) -> None:
    expected: dict[str, bool] = {
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }
    if include_runtime_flags:
        expected.update(
            {
                "serving_eligible": False,
                "action_authority_enabled": False,
                "submission_eligible": False,
                "promotion_eligible": False,
            }
        )
    for key, expected_value in expected.items():
        if value.get(key) is not expected_value:
            raise _error(f"{label}.{key} is not {expected_value}")


def _current_process_identity() -> dict[str, str]:
    boot_id = "unavailable"
    start = "unavailable"
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    if boot_path.is_file():
        try:
            boot_id = boot_path.read_text(encoding="utf-8").strip() or "unavailable"
        except OSError:
            pass
    stat_path = Path("/proc/self/stat")
    if stat_path.is_file():
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
            if len(fields) > 21:
                start = fields[21]
        except OSError:
            pass
    return {
        "process_id": str(os.getpid()),
        "boot_id": boot_id,
        "process_start_ticks": start,
    }


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _verify_r198_config(config: Any, max_action_combos: Any) -> None:
    expected: dict[str, Any] = {
        "schema": "poke_bot.recursive_turn_planner/v1",
        "sizing_profile": R198_PROFILE,
        "d_model": 96,
        "dynamics_width": 192,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_plan_length": 12,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "skip_trivial_decisions": True,
        "online_sim_verify_budget": 0,
        "repair_budget": 1,
        "compute_cost_penalty": 0.01,
        "option_batch_hint": 64,
        "prefer_option_hidden": True,
        "policy_aid_cap": 0.25,
        "default_subgoals": (
            "establish_attacker",
            "find_resource",
            "maximize_draw",
            "preserve_escape",
            "reach_damage_threshold",
            "setup_next_turn",
        ),
    }
    for key, expected_value in expected.items():
        observed = _config_value(config, key)
        if key == "default_subgoals" and isinstance(observed, list):
            observed = tuple(observed)
        if observed != expected_value:
            raise _error(f"planner config differs at {key}")
    if isinstance(max_action_combos, bool) or max_action_combos != R198_MAX_ACTION_COMBOS:
        raise _error("planner max_action_combos is not the exact r198 1024")


def _matching_cell(
    manifest: Mapping[str, Any],
    *,
    cell_id: str,
    evaluation_case_id: str,
    opponent_id: str,
    candidate_seat: Any,
) -> None:
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list):
        raise _error("evaluation manifest lacks a schedule")
    matches = [
        row
        for row in schedule
        if isinstance(row, Mapping) and row.get("cell_id") == cell_id
    ]
    if len(matches) != 1:
        raise _error("evaluation action cell is absent or ambiguous in manifest")
    row = matches[0]
    for key, expected in (
        ("evaluation_case_id", evaluation_case_id),
        ("opponent_id", opponent_id),
        ("candidate_seat", candidate_seat),
    ):
        if row.get(key) != expected:
            raise _error(f"evaluation action cell differs at {key}")


def _require_env_identity(
    environment: Mapping[str, str],
    *,
    path_key: str,
    digest_key: str,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if environment.get(path_key) != expected["path"]:
        raise _error(f"{label} environment path differs")
    if environment.get(digest_key) != expected["sha256"]:
        raise _error(f"{label} environment checksum differs")


def validate_evaluation_action_execution(
    execution: Any,
    *,
    config: Any,
    max_action_combos: Any,
    expected_parent_digest: Any,
    checkpoint_path: str | Path | None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the only evaluator exception to r197 serving promotion.

    The returned mapping is a JSON-deep-copied snapshot.  Callers should retain
    that copy and call this function again immediately before every r197
    select, which rehashes every immutable evidence file and rechecks the live
    process identity.
    """

    value = _copy_json_mapping(execution, "evaluation_action_execution")
    if value.get("schema") != EVALUATION_ACTION_EXECUTION_SCHEMA:
        raise _error("evaluation_action_execution schema is invalid")
    if value.get("status") != "authorized_evaluation_only":
        raise _error("evaluation_action_execution status is invalid")
    if value.get("execution_kind") != "evaluation_action_execution":
        raise _error("evaluation_action_execution kind is invalid")
    _require_bool_fields(value, "evaluation_action_execution", include_runtime_flags=True)

    arm = _require_text(value.get("arm"), "evaluation_action_execution.arm")
    if arm not in {R198_DIRECT_ARM, R198_RECURSIVE_ARM}:
        raise _error("evaluation_action_execution arm is not an r197 action arm")
    cell_id = _require_text(value.get("cell_id"), "evaluation_action_execution.cell_id")
    case_id = _require_text(
        value.get("evaluation_case_id"), "evaluation_action_execution.evaluation_case_id"
    )
    opponent_id = _require_text(
        value.get("opponent_id"), "evaluation_action_execution.opponent_id"
    )
    candidate_seat = value.get("candidate_seat")
    if candidate_seat not in {0, 1}:
        raise _error("evaluation_action_execution.candidate_seat is invalid")
    nonce = _require_text(value.get("launch_nonce"), "evaluation_action_execution.launch_nonce")
    if len(nonce) != 48 or any(character not in "0123456789abcdef" for character in nonce):
        raise _error("evaluation_action_execution.launch_nonce is invalid")
    process = _copy_json_mapping(value.get("process"), "evaluation_action_execution.process")
    current_process = _current_process_identity()
    if process != current_process:
        raise _error("evaluation_action_execution process identity differs")

    manifest_identity = _immutable_identity(value.get("manifest"), "evaluation manifest")
    authority_identity = _immutable_identity(
        value.get("evaluation_authority"), "evaluation authority"
    )
    runtime_identity = _immutable_identity(
        value.get("runtime_contract"), "candidate runtime contract"
    )
    fence_identity = _immutable_identity(value.get("action_fence"), "evaluation action fence")
    env = os.environ if environment is None else environment
    _require_env_identity(
        env,
        path_key=RUNTIME_CONTRACT_ENV,
        digest_key=RUNTIME_CONTRACT_SHA256_ENV,
        expected=runtime_identity,
        label="candidate runtime contract",
    )
    _require_env_identity(
        env,
        path_key=ACTION_FENCE_ENV,
        digest_key=ACTION_FENCE_SHA256_ENV,
        expected=fence_identity,
        label="evaluation action fence",
    )
    if env.get(LAUNCH_NONCE_ENV) != nonce:
        raise _error("evaluation action fence launch nonce environment differs")
    if env.get(PROCESS_ID_ENV) != current_process["process_id"]:
        raise _error("evaluation action fence process ID environment differs")
    if env.get(PROCESS_START_TICKS_ENV) != current_process["process_start_ticks"]:
        raise _error("evaluation action fence process-start environment differs")

    manifest = _json_payload(manifest_identity, "evaluation manifest")
    authority = _json_payload(authority_identity, "evaluation authority")
    runtime = _json_payload(runtime_identity, "candidate runtime contract")
    fence = _json_payload(fence_identity, "evaluation action fence")
    if manifest.get("schema") != EVALUATION_MANIFEST_SCHEMA:
        raise _error("evaluation manifest schema is invalid")
    if authority.get("schema") != EVALUATION_AUTHORITY_SCHEMA or authority.get(
        "status"
    ) != "authorized_evaluation_only":
        raise _error("evaluation authority schema/status is invalid")
    _require_bool_fields(authority, "evaluation authority")
    if authority.get("manifest_sha256") != manifest_identity["sha256"]:
        raise _error("evaluation authority is not bound to manifest")
    if runtime.get("schema") != CANDIDATE_RUNTIME_CONTRACT_SCHEMA or runtime.get(
        "status"
    ) != "sealed":
        raise _error("candidate runtime contract schema/status is invalid")
    if runtime.get("no_symlinks") is not True or runtime.get("all_paths_read_only") is not True:
        raise _error("candidate runtime contract is not physically sealed")
    if runtime.get("candidate_id") != R198_CANDIDATE_ID or runtime.get(
        "candidate_contract_sha256"
    ) != R198_CANDIDATE_CONTRACT_SHA256:
        raise _error("candidate runtime contract identity is invalid")
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise _error("candidate runtime contract lacks artifacts")
    parent_identity = _immutable_identity(artifacts.get("parent_checkpoint"), "candidate parent")
    sidecar_identity = _immutable_identity(artifacts.get("sidecar"), "candidate sidecar")
    if parent_identity["sha256"] != R198_PARENT_CHECKPOINT_SHA256:
        raise _error("candidate runtime contract parent differs")
    if sidecar_identity["sha256"] != R198_SIDECAR_SHA256:
        raise _error("candidate runtime contract sidecar differs")
    if expected_parent_digest != R198_PARENT_CHECKPOINT_SHA256:
        raise _error("loaded sidecar parent binding differs")
    if checkpoint_path is None:
        raise _error("evaluation action selection has no checkpoint-loaded sidecar")
    actual_sidecar, actual_sidecar_bytes, _actual_sidecar_stat = _read_immutable_bytes(
        str(checkpoint_path), "loaded sidecar"
    )
    if str(actual_sidecar) != sidecar_identity["path"] or (
        "sha256:" + hashlib.sha256(actual_sidecar_bytes).hexdigest()
    ) != sidecar_identity["sha256"]:
        raise _error("loaded sidecar differs from candidate runtime contract")
    _verify_r198_config(config, max_action_combos)

    binding = manifest.get("candidate_evaluation_binding")
    if not isinstance(binding, Mapping):
        raise _error("evaluation manifest lacks candidate binding")
    expected_binding = {
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "parent_checkpoint_sha256": R198_PARENT_CHECKPOINT_SHA256,
        "sidecar_sha256": R198_SIDECAR_SHA256,
        "sizing_profile": R198_PROFILE,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_action_combos": R198_MAX_ACTION_COMBOS,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise _error(f"evaluation manifest candidate binding differs at {key}")
    shared = manifest.get("shared_artifacts")
    arms = manifest.get("arms")
    if not isinstance(shared, Mapping) or not isinstance(arms, Mapping):
        raise _error("evaluation manifest lacks runtime artifacts")
    manifest_parent = _immutable_identity(shared.get("parent_checkpoint"), "manifest parent")
    arm_spec = arms.get(arm)
    if not isinstance(arm_spec, Mapping):
        raise _error("evaluation manifest lacks requested arm")
    manifest_sidecar = _immutable_identity(arm_spec.get("rtp_sidecar"), "manifest sidecar")
    if not _same_identity(manifest_parent, parent_identity) or not _same_identity(
        manifest_sidecar, sidecar_identity
    ):
        raise _error("evaluation manifest runtime artifacts differ")
    profile = arm_spec.get("profile")
    if not isinstance(profile, Mapping):
        raise _error("evaluation manifest arm lacks profile")
    for key, expected in (
        ("sizing_profile", R198_PROFILE),
        ("max_neural_passes", R198_MAX_NEURAL_PASSES),
        ("max_action_combos", R198_MAX_ACTION_COMBOS),
        ("serving_eligible", False),
        ("action_authority_enabled", False),
    ):
        if profile.get(key) != expected:
            raise _error(f"evaluation manifest arm profile differs at {key}")
    _matching_cell(
        manifest,
        cell_id=cell_id,
        evaluation_case_id=case_id,
        opponent_id=opponent_id,
        candidate_seat=candidate_seat,
    )

    if fence.get("schema") != EVALUATION_ACTION_FENCE_SCHEMA or fence.get(
        "status"
    ) != "authorized_evaluation_only":
        raise _error("evaluation action fence schema/status is invalid")
    material = {
        key: item for key, item in fence.items() if key != "runtime_contract_sha256"
    }
    if fence.get("runtime_contract_sha256") != _canonical_digest(material):
        raise _error("evaluation action fence payload digest changed")
    fence_authority = _immutable_identity(
        fence.get("evaluation_authority"), "action fence authority"
    )
    if not _same_identity(fence_authority, authority_identity):
        raise _error("evaluation action fence authority differs")
    expected_fence = {
        "manifest_sha256": manifest_identity["sha256"],
        "cell_id": cell_id,
        "evaluation_case_id": case_id,
        "opponent_id": opponent_id,
        "candidate_seat": candidate_seat,
        "arm": arm,
        "launch_nonce": nonce,
        "candidate_parent_checkpoint_sha256": R198_PARENT_CHECKPOINT_SHA256,
        "action_attached_rtp_sidecar_sha256": R198_SIDECAR_SHA256,
        "complexity_probe_sidecar_sha256": R198_SIDECAR_SHA256,
        "rtp_action_attachment_enabled": True,
    }
    for key, expected in expected_fence.items():
        if fence.get(key) != expected:
            raise _error(f"evaluation action fence differs at {key}")
    _require_bool_fields(fence, "evaluation action fence")
    return value


def assert_r197_action_selection_authorized(
    *,
    serving_qualified: bool,
    serving_promotion_validated: bool,
    evaluation_action_execution: Any,
    config: Any,
    max_action_combos: Any,
    expected_parent_digest: Any,
    checkpoint_path: str | Path | None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the narrow authority mode or raise before an r197 selection."""

    if serving_qualified:
        if not serving_promotion_validated:
            raise _error("serving promotion was not validated at sidecar load")
        return {"mode": "serving_qualified_promotion"}
    if evaluation_action_execution is None:
        raise _error("no evaluation_action_execution context was supplied")
    validated = validate_evaluation_action_execution(
        evaluation_action_execution,
        config=config,
        max_action_combos=max_action_combos,
        expected_parent_digest=expected_parent_digest,
        checkpoint_path=checkpoint_path,
        environment=environment,
    )
    return {
        "mode": "evaluation_action_execution",
        "cell_id": validated["cell_id"],
        "arm": validated["arm"],
        "launch_nonce": validated["launch_nonce"],
    }


__all__ = [
    "ACTION_FENCE_ENV",
    "ACTION_FENCE_SHA256_ENV",
    "CANDIDATE_RUNTIME_CONTRACT_SCHEMA",
    "EVALUATION_ACTION_EXECUTION_SCHEMA",
    "EVALUATION_ACTION_FENCE_SCHEMA",
    "LAUNCH_NONCE_ENV",
    "PROCESS_ID_ENV",
    "PROCESS_START_TICKS_ENV",
    "R197ActionAuthorityError",
    "R198_DIRECT_ARM",
    "R198_MAX_ACTION_COMBOS",
    "R198_MAX_NEURAL_PASSES",
    "R198_RECURSIVE_ARM",
    "RUNTIME_CONTRACT_ENV",
    "RUNTIME_CONTRACT_SHA256_ENV",
    "assert_r197_action_selection_authorized",
    "validate_evaluation_action_execution",
]
