#!/usr/bin/env python3
"""Read-only contract checks shared by the r244 Kaggle replay harnesses.

These helpers deliberately inspect the staged package as an immutable object.
They never import its entrypoint, start a simulator, or alter a package member.
The two executable harnesses use them before and after their own isolated
evaluation work so an accidental ``__pycache__`` or other staged-tree write is
treated as a failed preflight rather than hidden evidence drift.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
MANIFEST_FILENAME = "r238_two_lane_bounded_mcts_manifest.json"
R225_TYPED_CONTRACT_MEMBER = (
    "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
)
COMPLETE_ACTION_CAP = 65_536
SIMULATOR_LANES = 2
HIGH_CONFIDENCE_THRESHOLD = 0.80
CHILD_SEARCH_SECONDS = 2.0
PARENT_ACTION_SECONDS = 4.0
MIN_BACKUPS = 8
STABLE_LEADER_OBSERVATIONS = 3
MAX_BACKUPS = 32
MAX_CONTINUATION_DEPTH = 8
PROVEN_TERMINAL_WIN_REVISION = 246
PROVEN_TERMINAL_WIN_STOP_REASON = "proven_deterministic_terminal_win_this_turn"
PROVEN_TERMINAL_WIN_PROOF_KIND = "exact_deterministic_simulator_terminal_win_this_turn"
R246_R225_TYPED_CONTRACT_SHA256 = (
    "sha256:3225b07997bc58cc5e89239491533628cae654b48c092dec76ce56a6b8205eb3"
)

# These are the compact identity projections consumed by the immutable r235
# binder.  They intentionally do not say whether CUDA is available: the
# resource envelope reports only storage/RAM/vCPU limits, while the package
# records an observation immediately before search.
PREFLIGHT_RECEIPT_SCHEMA = "poke_bot.r235_r236_local_preflight_receipt/v1"
SAVED_EPISODE_RECEIPT_NAME = (
    "saved_episode_91766923_seat_0_step_58_two_choice_callback_"
    "legal_hard_deadline_regression_receipt"
)
FULL_GAME_RECEIPT_NAME = "exact_repaired_package_full_local_game_receipt"
PHASE1_SUBMISSION_ENVIRONMENT: dict[str, object] = {
    "hdd_space_gib": 11.8,
    "ram_gib": 12.2,
    "vcpus": 2,
    "submission_archive_limit_mib": 197.7,
}
R242_BINDING_SCHEDULER: dict[str, object] = {
    "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
    "all_selected_stages_finite": True,
    "immediate_no_child": True,
    "no_mcts_select_search_model_or_simulator_calls": True,
    "history_only_existing_child_journal_count_range": [0, 1],
    "high_confidence_degraded": False,
    "child_search_seconds": CHILD_SEARCH_SECONDS,
    "parent_action_deadline_seconds": PARENT_ACTION_SECONDS,
    "minimum_backups_before_stability": MIN_BACKUPS,
    "stable_root_leader_observations": STABLE_LEADER_OBSERVATIONS,
    "maximum_backups_per_decision": MAX_BACKUPS,
    "early_stop_requires_both_lanes_progressed": True,
    "stop_reason_required": True,
}
BINDING_DETERMINISTIC_CONTINUATION: dict[str, object] = {
    "max_depth": MAX_CONTINUATION_DEPTH,
    "exact_observation_fingerprint_required": True,
    "both_lanes_same_fingerprint_and_backed_action_required": True,
    "same_root_actor_required": True,
    "chance_or_boundary_forbidden": True,
    "no_new_search_on_valid_match": True,
    "mismatch_clears_entire_plan": True,
}
R246_TERMINAL_WIN_PROOF_FIELDS = (
    "proof_kind",
    "root_observation_fingerprint",
    "root_legal_order_fingerprint",
    "root_actor_seat",
    "root_action",
    "selected_action",
    "terminal_result",
    "terminal_winner_seat",
    "terminal_leaf_reached",
    "proof_path_action_count",
    "discovering_lane_id",
    "path_actor_seats",
    "path_no_chance_boundary",
    "path_no_actor_change_boundary",
    "path_no_opponent_boundary_crossing",
    "path_no_unresolved_randomness",
    "proof_is_deterministic",
)
R246_TERMINAL_WIN_PROOF_FIELD_SET = frozenset(R246_TERMINAL_WIN_PROOF_FIELDS)
CUDA_RUNTIME_OBSERVATION_SCHEMA = "poke_bot.r238_cuda_runtime_observation/v1"
CUDA_RUNTIME_OBSERVATION_PHASE = "before_search"
BINDING_COMMON_IDENTITY_FIELDS = (
    "candidate_archive_sha256",
    "member_manifest_sha256",
    "entrypoint_sha256",
    "r225_contract_sha256",
    "canonical_libcg_contract_sha256",
    "linux_x86_64_libcg_sha256",
    "linux_x86_64_libcg_size_bytes",
    "complete_ordered_action_cap",
    "simulator_search_lane_count",
    "phase1_submission_environment",
    "r240_hybrid_scheduler",
    "deterministic_continuation",
)

DECISION_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_DECISION "
FULL_GAMEPLAY_SUCCESS_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_FULL_GAMEPLAY_SUCCESS "
DEGRADED_FALLBACK_PREFIX = "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED "
HARD_FAILURE_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_HARD_FAILURE "

R236_NATIVE_MEMBERS: dict[str, dict[str, Any]] = {
    "cg/libcg.so": {
        "platform": "linux_x86_64",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
    },
    "cg/libcg-arm64.so": {
        "platform": "linux_aarch64",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
    },
    "cg/libcg.dylib": {
        "platform": "macos_arm64",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
    },
    "cg/cg.dll": {
        "platform": "windows_x86_64",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
    },
}

REQUIRED_STAGE_MEMBERS = (
    "main.py",
    "r195_direct_main.py",
    "model.pt",
    "matchup_tree.json",
    "search_config.json",
    "deck.csv",
    "poke_bot/r225_stock_native_lane.py",
    "poke_bot/r228_async_shared_tree_queue.py",
    "poke_bot/r228_kaggle_async_runtime.py",
    "poke_bot/r228_kaggle_broker.py",
    R225_TYPED_CONTRACT_MEMBER,
    MANIFEST_FILENAME,
    *tuple(R236_NATIVE_MEMBERS),
)


class HarnessContractError(RuntimeError):
    """A staged package or receipt does not meet the r244 contract."""


def require(value: bool, message: str) -> None:
    if not value:
        raise HarnessContractError(message)


def prepare_exact_stage_import(stage: Path) -> Path:
    """Make the immutable stage the sole source for package-owned imports.

    The executable harnesses are intentionally run outside the extracted
    submission directory.  A host bootstrap can therefore have already loaded
    the repository's ``poke_bot`` package (or an earlier candidate) into
    ``sys.modules``.  ``sys.path`` precedence alone cannot replace that cached
    package, so clear the complete package subtree before importing staged
    ``main.py``.  The harness process is disposable and owns these package
    names; preserving an unverified host copy would defeat exact-stage
    evaluation.
    """

    resolved = stage.resolve()
    require(
        resolved.is_dir() and not resolved.is_symlink(),
        "exact stage is not a physical directory",
    )
    package_dir = resolved / "poke_bot"
    require(
        package_dir.is_dir() and not package_dir.is_symlink(),
        "exact stage lacks a physical poke_bot package",
    )

    def _belongs_to_stage_path(entry: object) -> bool:
        if not isinstance(entry, str):
            return False
        try:
            return Path(entry or os.curdir).resolve() == resolved
        except OSError:
            return False

    # ``main`` and the direct-policy module are top-level package members in
    # the submission archive.  Clear their historic aliases together with all
    # of ``poke_bot`` so an earlier candidate cannot supply one hybrid member.
    for name in tuple(sys.modules):
        if (
            name in {"main", "r195_direct_main", "r228_r195_direct"}
            or name == "poke_bot"
            or name.startswith("poke_bot.")
        ):
            sys.modules.pop(name, None)

    stage_text = str(resolved)
    # Always put the resolved stage first, even when a semantically equivalent
    # path was already present later in ``sys.path``.  Remove equivalents so
    # the invariant is easy to inspect and test.
    sys.path[:] = [
        stage_text,
        *(entry for entry in sys.path if not _belongs_to_stage_path(entry)),
    ]
    importlib.invalidate_caches()
    return resolved


def require_module_from_exact_stage(
    module: object, *, module_name: str, stage: Path
) -> None:
    """Fail closed unless one imported module physically belongs to ``stage``."""

    source = getattr(module, "__file__", None)
    require(
        isinstance(source, str) and bool(source),
        f"staged module {module_name} has no physical source path",
    )
    try:
        Path(source).resolve().relative_to(stage.resolve())
    except (OSError, ValueError) as exc:
        raise HarnessContractError(
            f"staged module {module_name} resolved outside the exact stage: {source}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def json_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessContractError("receipt payload is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def stage_snapshot(stage: Path) -> dict[str, Any]:
    """Return a complete, symlink-free snapshot of the sealed package tree."""

    stage = stage.resolve()
    require(stage.is_dir() and not stage.is_symlink(), "stage is not a physical directory")
    members: dict[str, str] = {}
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise HarnessContractError(
                f"stage contains a symlink: {path.relative_to(stage).as_posix()}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        members[relative] = sha256_file(path)
    return {
        "file_count": len(members),
        "members": members,
        "tree_sha256": json_digest(members),
    }


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"{label} is not a regular file")
    return resolved


def _safe_archive_member_name(name: str) -> str:
    """Normalize one regular tar member without accepting traversal aliases."""

    candidate = name
    while candidate.startswith("./"):
        candidate = candidate[2:]
    parsed = PurePosixPath(candidate)
    require(
        bool(candidate)
        and not parsed.is_absolute()
        and ".." not in parsed.parts
        and str(parsed) not in {"", "."},
        "candidate archive contains an unsafe member path",
    )
    return parsed.as_posix()


def archive_snapshot(archive: Path) -> dict[str, Any]:
    """Hash the exact regular tar members without extracting or mutating them."""

    archive = _regular_file(archive, label="candidate archive")
    members: dict[str, str] = {}
    sizes: dict[str, int] = {}
    try:
        source = tarfile.open(archive, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise HarnessContractError("candidate archive is not a readable tar archive") from exc
    with source:
        for member in source.getmembers():
            relative = _safe_archive_member_name(member.name)
            require(member.isfile(), f"candidate archive member is not regular: {relative}")
            require(relative not in members, f"candidate archive repeats member: {relative}")
            stream = source.extractfile(member)
            require(stream is not None, f"candidate archive member is unreadable: {relative}")
            digest = hashlib.sha256()
            observed_size = 0
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    observed_size += len(block)
            require(
                observed_size == member.size,
                f"candidate archive member is truncated: {relative}",
            )
            members[relative] = "sha256:" + digest.hexdigest()
            sizes[relative] = observed_size
    require(bool(members), "candidate archive has no regular members")
    return {
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "file_count": len(members),
        "members": members,
        "sizes": sizes,
        "tree_sha256": json_digest(members),
    }


def load_binding_identity(
    *,
    stage: Path,
    candidate_archive: Path,
    member_manifest: Path,
    r225_contract: Path,
    r236_contract: Path,
) -> dict[str, Any]:
    """Prove a read-only stage is byte-identical to the binder candidate.

    The binder accepts gate receipts only if their common identity describes
    the exact archive, external/embedded r238 manifest, entrypoint, contracts,
    native member, and compact r242 scheduler.  A staged directory alone is
    insufficient evidence, so these harnesses compare every file digest with
    the archive before any simulator/import work begins.
    """

    stage = stage.expanduser().resolve()
    stage_contract = load_stage_contract(stage)
    stage_members = dict(stage_contract["stage_snapshot"]["members"])
    archive = archive_snapshot(candidate_archive)
    archive_members = dict(archive["members"])
    require(
        archive_members == stage_members,
        "stage is not byte-identical to the candidate archive members",
    )
    external_manifest = _regular_file(member_manifest, label="external member manifest")
    staged_manifest = stage / MANIFEST_FILENAME
    require(
        sha256_file(external_manifest) == sha256_file(staged_manifest),
        "external member manifest does not byte-match staged r238 manifest",
    )
    require(
        archive_members.get(MANIFEST_FILENAME) == sha256_file(external_manifest),
        "candidate archive does not bind the external r238 manifest",
    )
    require(
        archive["sizes"].get(MANIFEST_FILENAME) == external_manifest.stat().st_size,
        "candidate archive r238 manifest size drift",
    )
    canonical_r225 = _regular_file(r225_contract, label="canonical r225 contract")
    canonical_r236 = _regular_file(r236_contract, label="canonical r236 contract")
    staged_r225 = stage / R225_TYPED_CONTRACT_MEMBER
    require(
        sha256_file(canonical_r225) == R246_R225_TYPED_CONTRACT_SHA256,
        "canonical r225 contract is not the owner-pinned r246 source",
    )
    require(
        sha256_file(canonical_r225) == sha256_file(staged_r225),
        "stage r225 member does not byte-match canonical r225 contract",
    )
    entrypoint = stage / "main.py"
    linux = R236_NATIVE_MEMBERS["cg/libcg.so"]
    common_identity: dict[str, object] = {
        "candidate_archive_sha256": archive["archive_sha256"],
        "candidate_archive_size_bytes": archive["archive_size_bytes"],
        "member_manifest_sha256": sha256_file(external_manifest),
        "entrypoint_sha256": sha256_file(entrypoint),
        "r225_contract_sha256": sha256_file(canonical_r225),
        "canonical_libcg_contract_sha256": sha256_file(canonical_r236),
        "linux_x86_64_libcg_sha256": linux["sha256"],
        "linux_x86_64_libcg_size_bytes": linux["size_bytes"],
        "complete_ordered_action_cap": COMPLETE_ACTION_CAP,
        "simulator_search_lane_count": SIMULATOR_LANES,
        "phase1_submission_environment": dict(PHASE1_SUBMISSION_ENVIRONMENT),
        "r240_hybrid_scheduler": dict(R242_BINDING_SCHEDULER),
        "deterministic_continuation": dict(BINDING_DETERMINISTIC_CONTINUATION),
    }
    return {
        "common_identity": common_identity,
        "exact_package": {
            "stage": str(stage),
            "stage_tree_sha256": stage_contract["stage_snapshot"]["tree_sha256"],
            "archive": archive["archive"],
            "archive_member_tree_sha256": archive["tree_sha256"],
            "archive_member_count": archive["file_count"],
            "stage_archive_members_byte_identical": True,
            "member_manifest": str(external_manifest),
            "member_manifest_member": MANIFEST_FILENAME,
            "entrypoint_member": "main.py",
            "r225_contract": str(canonical_r225),
            "r236_contract": str(canonical_r236),
        },
        "stage_contract": stage_contract,
    }


def passed_preflight_receipt(
    *, receipt_name: str, common_identity: Mapping[str, Any], harness_schema: str
) -> dict[str, Any]:
    """Start an immutable binder-compatible passed gate receipt."""

    for field in BINDING_COMMON_IDENTITY_FIELDS:
        require(field in common_identity, f"binding identity lacks {field}")
    return {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "receipt_name": receipt_name,
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        **{field: common_identity[field] for field in BINDING_COMMON_IDENTITY_FIELDS},
        "harness_schema": harness_schema,
    }


def _as_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessContractError(f"{field} must be an object")
    return value


def _as_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessContractError(f"{field} must be an integer")
    return int(value)


def _as_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise HarnessContractError(f"{field} must be finite")
    return result


def validate_cuda_runtime_observation(value: object, *, field: str) -> dict[str, Any]:
    """Validate a pre-search observation without inferring accelerator policy.

    CUDA can be absent or present on the submitted host.  This checks only
    that the package recorded a complete, internally coherent observation at
    the required boundary; it never turns a CPU/GPU observation into an action
    or resource-envelope assertion.
    """

    observation = _as_mapping(value, field=field)
    expected_keys = {
        "schema",
        "phase",
        "torch_imported",
        "cuda_available",
        "cuda_initialized",
        "device_count",
        "devices",
        "model_device",
        "telemetry_complete",
        "error_types",
    }
    require(set(observation) == expected_keys, f"{field} shape drift")
    require(
        observation.get("schema") == CUDA_RUNTIME_OBSERVATION_SCHEMA,
        f"{field} schema drift",
    )
    require(
        observation.get("phase") == CUDA_RUNTIME_OBSERVATION_PHASE,
        f"{field} phase drift",
    )
    booleans: dict[str, bool] = {}
    for key in (
        "torch_imported",
        "cuda_available",
        "cuda_initialized",
        "telemetry_complete",
    ):
        raw = observation.get(key)
        require(isinstance(raw, bool), f"{field}.{key} must be boolean")
        booleans[key] = raw
    device_count = _as_int(observation.get("device_count"), field=f"{field}.device_count")
    require(device_count >= 0, f"{field}.device_count is negative")
    model_device = observation.get("model_device")
    require(isinstance(model_device, str) and bool(model_device), f"{field}.model_device is absent")
    errors = observation.get("error_types")
    require(
        isinstance(errors, list)
        and all(isinstance(error, str) and bool(error) for error in errors),
        f"{field}.error_types is malformed",
    )
    devices_raw = observation.get("devices")
    require(isinstance(devices_raw, list), f"{field}.devices is not a list")
    if not booleans["cuda_available"]:
        require(device_count == 0, f"{field} reports unavailable CUDA with devices")
        require(not devices_raw, f"{field} reports unavailable CUDA device rows")
        require(not booleans["cuda_initialized"], f"{field} initialized unavailable CUDA")
    else:
        require(device_count >= 1, f"{field} reports available CUDA without a device")
        require(len(devices_raw) <= device_count, f"{field} over-reports CUDA device rows")
    normalized_devices: list[dict[str, Any]] = []
    for index, device_raw in enumerate(devices_raw):
        device = _as_mapping(device_raw, field=f"{field}.devices[{index}]")
        require(
            set(device)
            == {
                "device_index",
                "device_name",
                "total_memory_bytes",
                "free_memory_bytes",
            },
            f"{field}.devices[{index}] shape drift",
        )
        require(
            _as_int(device.get("device_index"), field=f"{field}.devices[{index}].device_index")
            == index,
            f"{field}.devices are not ordered by device index",
        )
        name = device.get("device_name")
        require(isinstance(name, str) and bool(name), f"{field}.devices[{index}] name missing")
        total = _as_int(
            device.get("total_memory_bytes"),
            field=f"{field}.devices[{index}].total_memory_bytes",
        )
        free = _as_int(
            device.get("free_memory_bytes"),
            field=f"{field}.devices[{index}].free_memory_bytes",
        )
        require(total > 0 and 0 <= free <= total, f"{field}.devices[{index}] memory drift")
        normalized_devices.append(dict(device))
    if booleans["telemetry_complete"]:
        require(not errors, f"{field} is complete but retains errors")
        require(
            len(normalized_devices) == device_count,
            f"{field} is complete but lacks device observations",
        )
    return {
        "cuda_available": booleans["cuda_available"],
        "cuda_initialized": booleans["cuda_initialized"],
        "device_count": device_count,
        "model_device": model_device,
        "telemetry_complete": booleans["telemetry_complete"],
        "devices": normalized_devices,
        "error_types": list(errors),
    }


def _validate_optional_child_cuda_observation(marker: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate child telemetry when a real child reached its ready boundary."""

    candidates: list[object] = []
    broker = marker.get("broker")
    if isinstance(broker, Mapping):
        identity = broker.get("child_identity")
        if isinstance(identity, Mapping):
            candidates.append(identity.get("cuda_runtime_before_search"))
    identity = marker.get("child_identity")
    if isinstance(identity, Mapping):
        candidates.append(identity.get("cuda_runtime_before_search"))
    fault = marker.get("child_fault")
    if isinstance(fault, Mapping):
        identity = fault.get("child_identity")
        if isinstance(identity, Mapping):
            candidates.append(identity.get("cuda_runtime_before_search"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return validate_cuda_runtime_observation(
                candidate, field="child_cuda_runtime_before_search"
            )
    return None


def _as_handle(value: object, *, field: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise HarnessContractError(f"{field} must be an opaque integer/string handle")
    if isinstance(value, str) and not value:
        raise HarnessContractError(f"{field} must not be empty")
    return value


def as_action(value: object, *, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise HarnessContractError(f"{field} must be a list of integers")
    action: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise HarnessContractError(f"{field} contains a non-integer")
        action.append(int(item))
    return action


def canonical_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    """Mirror the sealed runtime's exact JSON-native root fingerprint."""

    try:
        encoded = json.dumps(
            dict(observation),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessContractError("root observation is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def legal_order_fingerprint(legal_actions: Sequence[Sequence[int]]) -> str:
    try:
        encoded = json.dumps(
            [[int(item) for item in action] for action in legal_actions],
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessContractError("complete legal order is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_r246_terminal_win_proof(
    marker: Mapping[str, Any],
    *,
    legal_actions: Sequence[Sequence[int]],
    selected_action: Sequence[int],
    lane_receipt: Mapping[str, Any],
    completed_backups: int,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Accept only a stock-terminal, current-root, root-actor-only proof."""

    require(
        _as_int(
            marker.get("owner_proven_deterministic_terminal_win_this_turn_revision"),
            field="owner_proven_deterministic_terminal_win_this_turn_revision",
        )
        == PROVEN_TERMINAL_WIN_REVISION,
        "terminal-win proof owner revision drift",
    )
    require(
        marker.get("proven_deterministic_terminal_win_this_turn") is True,
        "terminal-win stop lacks public proof classification",
    )
    proof = _as_mapping(marker.get("terminal_win_proof"), field="terminal-win proof")
    require(
        set(proof) == R246_TERMINAL_WIN_PROOF_FIELD_SET,
        "terminal-win proof has an invalid exact schema",
    )
    root_observation_fingerprint = marker.get("root_observation_fingerprint")
    root_legal_order_fingerprint = marker.get("root_legal_order_fingerprint")
    require(
        isinstance(root_observation_fingerprint, str) and bool(root_observation_fingerprint),
        "terminal-win receipt lacks root observation fingerprint",
    )
    require(
        isinstance(root_legal_order_fingerprint, str) and bool(root_legal_order_fingerprint),
        "terminal-win receipt lacks root legal-order fingerprint",
    )
    require(
        proof.get("root_observation_fingerprint") == root_observation_fingerprint,
        "terminal-win proof is stale for the receipt observation",
    )
    require(
        proof.get("root_legal_order_fingerprint") == root_legal_order_fingerprint,
        "terminal-win proof is stale for the receipt legal order",
    )
    require(
        root_legal_order_fingerprint == legal_order_fingerprint(legal_actions),
        "terminal-win proof does not bind the current complete legal order",
    )
    marker_legal = marker.get("legal_order_fingerprint")
    require(
        marker_legal == root_legal_order_fingerprint,
        "terminal-win receipt parent/legal fingerprint drift",
    )
    if observation is not None:
        require(
            root_observation_fingerprint == canonical_observation_fingerprint(observation),
            "terminal-win proof is stale for the current root observation",
        )
    root_actor = _as_int(marker.get("root_actor_seat"), field="root_actor_seat")
    require(root_actor in (0, 1), "terminal-win root actor is invalid")
    require(
        _as_int(proof.get("root_actor_seat"), field="proof root_actor_seat")
        == root_actor,
        "terminal-win proof root actor drift",
    )
    root_action = as_action(proof.get("root_action"), field="proof root_action")
    proof_selected = as_action(proof.get("selected_action"), field="proof selected_action")
    normalized_selected = as_action(selected_action, field="selected_action")
    normalized_legal = [as_action(action, field="legal action") for action in legal_actions]
    require(
        root_action == proof_selected == normalized_selected
        and proof_selected in normalized_legal,
        "terminal-win proof action is not the current selected legal root action",
    )
    require(
        proof.get("proof_kind") == PROVEN_TERMINAL_WIN_PROOF_KIND,
        "terminal-win proof is not an exact simulator terminal proof",
    )
    require(proof.get("terminal_result") == "win", "terminal-win proof result is not win")
    require(
        _as_int(proof.get("terminal_winner_seat"), field="terminal_winner_seat")
        == root_actor,
        "terminal-win proof winner differs from the root actor",
    )
    for field in (
        "terminal_leaf_reached",
        "path_no_actor_change_boundary",
        "path_no_opponent_boundary_crossing",
        "path_no_chance_boundary",
        "path_no_unresolved_randomness",
        "proof_is_deterministic",
    ):
        require(proof.get(field) is True, f"terminal-win proof does not prove {field}")
    path_count = _as_int(proof.get("proof_path_action_count"), field="proof_path_action_count")
    require(
        1 <= path_count <= completed_backups,
        "terminal-win proof path is not backed into the shared root",
    )
    path_actors = proof.get("path_actor_seats")
    require(
        isinstance(path_actors, list)
        and len(path_actors) == path_count
        and all(_as_int(actor, field="path_actor_seat") == root_actor for actor in path_actors),
        "terminal-win proof crossed an actor/opponent boundary",
    )
    discovering_lane = _as_int(proof.get("discovering_lane_id"), field="discovering_lane_id")
    require(0 <= discovering_lane < SIMULATOR_LANES, "terminal-win discovering lane is invalid")
    depths = lane_receipt["depths"]
    require(
        _as_int(depths[discovering_lane], field="discovering lane depth") >= path_count,
        "terminal-win proof exceeds its discovering lane depth",
    )
    principal_variation = marker.get("principal_variation")
    require(
        principal_variation in (None, [], ()),
        "terminal-win override retained a continuation plan",
    )
    return dict(proof)


def marker_rows(text: str, *, prefix: str, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line[len(prefix) :])
        except json.JSONDecodeError as exc:
            raise HarnessContractError(f"{label} marker is malformed JSON") from exc
        if not isinstance(value, dict):
            raise HarnessContractError(f"{label} marker is not an object")
        rows.append(value)
    return rows


def collect_markers(text: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "decisions": marker_rows(text, prefix=DECISION_PREFIX, label="decision"),
        "degraded_fallbacks": marker_rows(
            text, prefix=DEGRADED_FALLBACK_PREFIX, label="degraded fallback"
        ),
        "hard_failures": marker_rows(
            text, prefix=HARD_FAILURE_PREFIX, label="hard failure"
        ),
        "full_gameplay_successes": marker_rows(
            text, prefix=FULL_GAMEPLAY_SUCCESS_PREFIX, label="full-game success"
        ),
    }


def _validate_manifest_native_members(stage: Path, manifest: Mapping[str, Any]) -> None:
    native_members = _as_mapping(
        manifest.get("canonical_native_members"), field="canonical_native_members"
    )
    require(
        set(native_members) == set(R236_NATIVE_MEMBERS),
        "manifest does not bind the complete four-member r236 native set",
    )
    for relative, expected in R236_NATIVE_MEMBERS.items():
        path = stage / relative
        require(path.is_file() and not path.is_symlink(), f"stage lacks {relative}")
        observed = _as_mapping(native_members.get(relative), field=relative)
        require(
            observed.get("platform") == expected["platform"],
            f"manifest platform drift for {relative}",
        )
        require(
            observed.get("sha256") == expected["sha256"],
            f"manifest digest drift for {relative}",
        )
        require(
            observed.get("size_bytes") == expected["size_bytes"],
            f"manifest size drift for {relative}",
        )
        require(path.stat().st_size == expected["size_bytes"], f"native size drift for {relative}")
        require(sha256_file(path) == expected["sha256"], f"native digest drift for {relative}")


def load_stage_contract(stage: Path) -> dict[str, Any]:
    """Validate the immutable staged r238/r242/r244 package identity."""

    stage = stage.resolve()
    require(stage.is_dir() and not stage.is_symlink(), "stage is not a physical directory")
    for relative in REQUIRED_STAGE_MEMBERS:
        path = stage / relative
        require(path.is_file() and not path.is_symlink(), f"stage lacks {relative}")
    manifest_path = stage / MANIFEST_FILENAME
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessContractError("stage manifest is unreadable") from exc
    manifest = _as_mapping(manifest_value, field="stage manifest")
    require(manifest.get("schema") == SCHEMA, "stage manifest schema drift")
    require(
        manifest.get("role") == "isolated_r238_two_lane_bounded_mcts_fallback_diagnostic",
        "stage manifest role drift",
    )
    require(manifest.get("complete_action_cap") == COMPLETE_ACTION_CAP, "stage cap is not 65536")
    require(manifest.get("lane_count") == SIMULATOR_LANES, "stage lane count is not two")
    require(
        manifest.get("branching_decision_marker") == DECISION_PREFIX.rstrip(),
        "stage decision marker drift",
    )
    require(
        manifest.get("full_gameplay_success_marker") == FULL_GAMEPLAY_SUCCESS_PREFIX.rstrip(),
        "stage full-game success marker drift",
    )
    require(
        manifest.get("hard_failure_marker") == HARD_FAILURE_PREFIX.rstrip(),
        "stage hard-failure marker drift",
    )
    require(
        manifest.get("degraded_fallback_marker") == DEGRADED_FALLBACK_PREFIX.rstrip(),
        "stage degraded marker drift",
    )

    scheduler = _as_mapping(manifest.get("r240_hybrid_scheduler"), field="r242 scheduler")
    require(
        scheduler.get("selected_factorized_stage_probability_threshold")
        == HIGH_CONFIDENCE_THRESHOLD,
        "stage does not bind the inclusive 0.80 threshold",
    )
    require(
        scheduler.get("historical_r240_0_90_threshold_draft_and_preflight_are_ineligible")
        is True,
        "stage does not reject historical 0.90 evidence",
    )
    for field, expected in {
        "ambiguous_mcts_exact_simulator_search_lane_count": SIMULATOR_LANES,
        "child_search_hard_seconds": CHILD_SEARCH_SECONDS,
        "parent_action_hard_seconds": PARENT_ACTION_SECONDS,
        "adaptive_early_stop_min_completed_backups": MIN_BACKUPS,
        "adaptive_early_stop_stable_deterministic_root_leader_observations": STABLE_LEADER_OBSERVATIONS,
        "hard_completed_backup_stop": MAX_BACKUPS,
    }.items():
        require(scheduler.get(field) == expected, f"stage scheduler drift at {field}")
    require(
        scheduler.get("mcts_opponent_action_selection_or_planning_allowed") is False,
        "stage permits opponent planning",
    )
    stop_reasons = scheduler.get("stop_reason_fields")
    require(
        isinstance(stop_reasons, list)
        and PROVEN_TERMINAL_WIN_STOP_REASON in stop_reasons,
        "stage does not bind the r246 terminal-win stop reason",
    )
    r246 = _as_mapping(
        scheduler.get("r246_proven_deterministic_terminal_win_this_turn"),
        field="r246 terminal-win override",
    )
    for field, expected in {
        "owner_decision_revision": PROVEN_TERMINAL_WIN_REVISION,
        "r242_high_confidence_frozen_direct_before_child_is_unchanged": True,
        "requires_exact_two_lane_topology_initialized_before_override": True,
        "one_valid_terminal_win_proof_from_either_lane_is_sufficient": True,
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        "terminal_leaf_must_be_returned_by_exact_stock_simulator": True,
        "terminal_leaf_must_be_backed_up_into_shared_root_tree": True,
        "minimum_completed_backups_for_valid_proof": 1,
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
        "root_action_has_absolute_selection_and_early_stop_authority_over_visits_priors_and_nonterminal_actions": True,
        "proof_kind_required_literal": PROVEN_TERMINAL_WIN_PROOF_KIND,
        "terminal_result_required_literal": "win",
        "terminal_winner_seat_must_equal_root_actor_seat": True,
        "root_action_must_be_currently_legal_and_equal_selected_action": True,
        "proof_path_actor_seats_must_all_equal_root_actor_seat": True,
        "proof_path_actor_change_or_opponent_boundary_allowed": False,
        "proof_path_chance_or_unresolved_randomness_allowed": False,
        "model_value_policy_confidence_or_heuristic_may_substitute_for_terminal_simulator_result": False,
        "loss_draw_nonterminal_stale_or_malformed_claim_has_terminal_win_override_authority": False,
        "stale_or_malformed_claim_marked_as_terminal_win_is_contained_child_protocol_fault": True,
        "all_owned_lane_resources_reservations_and_child_cleanup_required_before_parent_return": True,
        "stop_reason": PROVEN_TERMINAL_WIN_STOP_REASON,
    }.items():
        require(r246.get(field) == expected, f"stage r246 contract drift at {field}")
    require(
        r246.get("required_receipt_fields") == list(R246_TERMINAL_WIN_PROOF_FIELDS),
        "stage r246 terminal proof fields drift",
    )
    continuation = _as_mapping(
        manifest.get("deterministic_continuation"), field="deterministic continuation"
    )
    require(
        continuation.get("maximum_depth") == MAX_CONTINUATION_DEPTH,
        "stage continuation depth drift",
    )
    require(
        continuation.get("valid_plan_starts_or_calls_new_mcts_search") is False,
        "stage continuation may start search",
    )

    r244 = _as_mapping(
        manifest.get("r244_handle_scoped_search_identity"),
        field="r244 handle-scoped identity",
    )
    expected_r244 = {
        "simulator_lane_count": SIMULATOR_LANES,
        "search_id_namespace": "agent_start_handle_local",
        "per_lane_handle_identities_required": True,
        "per_lane_search_id_chains_required": True,
        "composite_state_fields": ["lane_id", "handle_identity", "first_search_id"],
        "distinct_composite_count_required": SIMULATOR_LANES,
        "globally_distinct_raw_first_search_ids_required": False,
        "duplicate_raw_first_search_ids_allowed_when_handles_differ": True,
        "gate_receipt": "official_libcg_handle_scoped_search_id_identity_regression_receipt",
    }
    require(dict(r244) == expected_r244, "stage r244 handle-scoped identity drift")

    r225 = _as_mapping(manifest.get("r225_typed_contract"), field="r225 typed contract")
    require(
        r225.get("path") == R225_TYPED_CONTRACT_MEMBER,
        "stage r225 contract path drift",
    )
    r225_path = stage / R225_TYPED_CONTRACT_MEMBER
    require(
        r225.get("sha256") == sha256_file(r225_path),
        "stage r225 contract digest does not bind packaged member",
    )
    _validate_manifest_native_members(stage, manifest)
    snapshot = stage_snapshot(stage)
    return {
        "manifest": dict(manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "r225_typed_contract": dict(r225),
        "native_members": {
            relative: dict(expected) for relative, expected in R236_NATIVE_MEMBERS.items()
        },
        "stage_snapshot": snapshot,
    }


def _validate_common_decision(
    marker: Mapping[str, Any], *, legal_actions: Sequence[Sequence[int]]
) -> tuple[list[int], list[int]]:
    selected = as_action(marker.get("selected_action"), field="selected_action")
    direct = as_action(marker.get("direct_action"), field="direct_action")
    normalized_legal = [as_action(row, field="legal action") for row in legal_actions]
    require(selected in normalized_legal, "decision selected action is outside complete legal order")
    require(direct in normalized_legal, "decision direct action is outside complete legal order")
    require(
        _as_int(marker.get("complete_action_cap"), field="complete_action_cap")
        == COMPLETE_ACTION_CAP,
        "decision cap drift",
    )
    require(
        _as_int(
            marker.get("configured_simulator_lane_count"),
            field="configured_simulator_lane_count",
        )
        == SIMULATOR_LANES,
        "decision configured lane count drift",
    )
    require(
        _as_int(marker.get("legal_action_count"), field="legal_action_count")
        == len(normalized_legal),
        "decision legal-action count drift",
    )
    validate_cuda_runtime_observation(
        marker.get("parent_cuda_runtime_before_search"),
        field="parent_cuda_runtime_before_search",
    )
    return selected, direct


def _validate_two_lane_composites(
    marker: Mapping[str, Any], *, allow_zero_depth: bool
) -> dict[str, Any]:
    exact_fields = (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "search_end_calls",
        "distinct_search_begin_composite_count",
    )
    for field in exact_fields:
        require(
            _as_int(marker.get(field), field=field) == SIMULATOR_LANES,
            f"{field} is not two",
        )
    require(
        _as_int(marker.get("search_release_calls"), field="search_release_calls")
        >= SIMULATOR_LANES,
        "search releases do not cover both lanes",
    )
    handles_raw = marker.get("per_lane_handle_identities")
    chains_raw = marker.get("per_lane_search_id_chains")
    first_raw = marker.get("per_lane_first_search_ids")
    depths_raw = marker.get("per_lane_depth")
    require(isinstance(handles_raw, (list, tuple)) and len(handles_raw) == SIMULATOR_LANES, "receipt lacks both lane handles")
    require(isinstance(chains_raw, (list, tuple)) and len(chains_raw) == SIMULATOR_LANES, "receipt lacks both SearchId chains")
    require(isinstance(first_raw, (list, tuple)) and len(first_raw) == SIMULATOR_LANES, "receipt lacks both first SearchIds")
    require(isinstance(depths_raw, (list, tuple)) and len(depths_raw) == SIMULATOR_LANES, "receipt lacks both lane depths")
    handles = [_as_handle(value, field="per_lane_handle_identities") for value in handles_raw]
    require(len(set(handles)) == SIMULATOR_LANES, "two lanes do not have distinct handles")
    first_ids: list[int] = []
    depths: list[int] = []
    for lane_id, (chain, first, depth) in enumerate(zip(chains_raw, first_raw, depths_raw)):
        require(isinstance(chain, (list, tuple)) and bool(chain), f"lane {lane_id} has no SearchId chain")
        chain_first = _as_int(chain[0], field=f"lane {lane_id} first SearchId")
        declared_first = _as_int(first, field=f"lane {lane_id} declared first SearchId")
        require(chain_first == declared_first, f"lane {lane_id} first SearchId drift")
        first_ids.append(chain_first)
        normalized_depth = _as_int(depth, field=f"lane {lane_id} depth")
        require(normalized_depth >= (0 if allow_zero_depth else 1), f"lane {lane_id} did not progress")
        depths.append(normalized_depth)
    composites = marker.get("handle_scoped_first_search_id_composite_states")
    require(isinstance(composites, (list, tuple)) and len(composites) == SIMULATOR_LANES, "receipt lacks two r244 composite states")
    seen: set[tuple[int | str, int]] = set()
    normalized_composites: list[dict[str, Any]] = []
    for lane_id, state_raw in enumerate(composites):
        state = _as_mapping(state_raw, field=f"composite state {lane_id}")
        require(set(state) == {"lane_id", "handle_identity", "first_search_id"}, "r244 composite shape drift")
        require(_as_int(state.get("lane_id"), field="composite lane_id") == lane_id, "r244 composite lane order drift")
        handle = _as_handle(state.get("handle_identity"), field="composite handle")
        first = _as_int(state.get("first_search_id"), field="composite first SearchId")
        require(handle == handles[lane_id] and first == first_ids[lane_id], "r244 composite disagrees with lane vectors")
        seen.add((handle, first))
        normalized_composites.append(
            {"lane_id": lane_id, "handle_identity": handle, "first_search_id": first}
        )
    require(len(seen) == SIMULATOR_LANES, "r244 handle/SearchId composites are not distinct")
    require(
        _as_int(marker.get("outstanding_virtual_loss"), field="outstanding_virtual_loss")
        == 0,
        "decision leaked virtual loss",
    )
    return {
        "handles": handles,
        "first_search_ids": first_ids,
        "composites": normalized_composites,
        "depths": depths,
    }


def validate_decision_marker(
    marker: Mapping[str, Any],
    *,
    legal_actions: Sequence[Sequence[int]],
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one parent decision receipt under r242/r244 semantics."""

    selected, direct = _validate_common_decision(marker, legal_actions=legal_actions)
    mode = marker.get("mode")
    require(isinstance(mode, str), "decision mode is absent")
    if mode == "high_confidence_frozen_direct":
        require(selected == direct, "high-confidence action differs from direct action")
        require(marker.get("mcts_action_authority") is False, "high-confidence action claimed MCTS authority")
        require(marker.get("mcts_child_started_for_this_decision") is False, "high-confidence action started a child")
        require(_as_int(marker.get("mcts_select_call_count"), field="mcts_select_call_count") == 0, "high-confidence action called select")
        require(marker.get("degraded") is False, "high-confidence action is degraded")
        require(
            _as_number(
                marker.get("selected_factorized_stage_probability_threshold"),
                field="selected_factorized_stage_probability_threshold",
            )
            == HIGH_CONFIDENCE_THRESHOLD,
            "high-confidence threshold drift",
        )
        probabilities = marker.get("selected_factorized_stage_probabilities")
        require(isinstance(probabilities, list) and bool(probabilities), "high-confidence action lacks stage probabilities")
        require(
            all(_as_number(value, field="selected stage probability") >= HIGH_CONFIDENCE_THRESHOLD for value in probabilities),
            "high-confidence action includes a below-threshold stage",
        )
        require(marker.get("all_selected_factorized_stages_meet_threshold") is True, "high-confidence threshold result is false")
        history_count = _as_int(marker.get("history_only_existing_child_journal_count"), field="history_only_existing_child_journal_count")
        require(0 <= history_count <= 1, "high-confidence history-only journal count drift")
        return {"mode": mode, "selected_action": selected, "direct_action": direct, "degraded": False}

    if mode == "deterministic_mcts_continuation":
        require(marker.get("mcts_action_authority") is False, "continuation claimed fresh MCTS authority")
        require(marker.get("mcts_child_started_for_this_decision") is False, "continuation started a child")
        require(_as_int(marker.get("mcts_select_call_count"), field="mcts_select_call_count") == 0, "continuation called select")
        require(marker.get("degraded") is False, "continuation is degraded")
        require(marker.get("history_rewritten_to_actual_action") is True, "continuation did not rewrite history")
        require(marker.get("continuation_both_lanes_same_fingerprint") is True, "continuation lacks both-lane fingerprint agreement")
        require(marker.get("continuation_backed_leader_agreement") is True, "continuation lacks backed-leader agreement")
        require(marker.get("continuation_plan_no_chance_boundary_or_opponent_transition") is True, "continuation crossed a boundary")
        require(
            0 <= _as_int(marker.get("continuation_plan_depth_remaining"), field="continuation_plan_depth_remaining") < MAX_CONTINUATION_DEPTH,
            "continuation depth is outside r242 bound",
        )
        history_count = _as_int(marker.get("history_only_existing_child_journal_count"), field="history_only_existing_child_journal_count")
        require(0 <= history_count <= 1, "continuation history-only journal count drift")
        return {"mode": mode, "selected_action": selected, "direct_action": direct, "degraded": False}

    if mode == "shared_tree_mcts":
        require(marker.get("mcts_action_authority") is True, "MCTS decision lacks MCTS authority")
        require(marker.get("confidence_classification") == "ambiguous_mcts", "MCTS decision did not originate from ambiguity")
        require(marker.get("mcts_child_started") is True, "MCTS decision did not start its child")
        require(_as_int(marker.get("mcts_child_call_count"), field="mcts_child_call_count") == 1, "MCTS child call count drift")
        require(_as_number(marker.get("child_search_hard_seconds"), field="child_search_hard_seconds") == CHILD_SEARCH_SECONDS, "child search deadline drift")
        require(_as_number(marker.get("parent_action_hard_seconds"), field="parent_action_hard_seconds") == PARENT_ACTION_SECONDS, "parent action deadline drift")
        require(marker.get("all_selected_factorized_stages_meet_threshold") is False, "ambiguous MCTS was labeled high confidence")
        child_cuda = _validate_optional_child_cuda_observation(marker)
        require(
            child_cuda is not None,
            "MCTS decision lacks child CUDA observation before search",
        )
        stop_reason = marker.get("stop_reason")
        terminal_win_stop = stop_reason == PROVEN_TERMINAL_WIN_STOP_REASON
        lane_receipt = _validate_two_lane_composites(
            marker, allow_zero_depth=terminal_win_stop
        )
        backups = _as_int(marker.get("completed_backups"), field="completed_backups")
        minimum_backups = 1 if terminal_win_stop else SIMULATOR_LANES
        require(
            minimum_backups <= backups <= MAX_BACKUPS,
            "MCTS completed backups violate the r242/r246 bound",
        )
        require(
            _as_int(marker.get("search_step_calls"), field="search_step_calls")
            >= SIMULATOR_LANES,
            "MCTS did not step both initialized native lanes",
        )
        require(
            sum(lane_receipt["depths"]) == backups,
            "MCTS per-lane depth does not account for every backup",
        )
        require(
            _as_int(marker.get("root_visits"), field="root_visits") == backups,
            "MCTS root visit count does not match backups",
        )
        if not terminal_win_stop:
            require(marker.get("both_lanes_progressed") is True, "MCTS receipt lacks both-lane progress")
        in_flight = _as_int(
            marker.get("max_simulator_calls_in_flight"),
            field="max_simulator_calls_in_flight",
        )
        require(1 <= in_flight <= SIMULATOR_LANES, "MCTS in-flight lane count drift")
        microbatches = marker.get("microbatch_sizes")
        require(isinstance(microbatches, (list, tuple)) and bool(microbatches), "MCTS lacks frozen-evaluator microbatches")
        normalized_batches = [
            _as_int(value, field="microbatch_sizes") for value in microbatches
        ]
        require(
            all(1 <= value <= SIMULATOR_LANES for value in normalized_batches),
            "MCTS microbatch exceeds two lanes",
        )
        require(sum(normalized_batches) == backups, "MCTS microbatch/backup receipt drift")
        require(_as_int(marker.get("selected_action_visits"), field="selected_action_visits") >= 1, "MCTS selected edge is unbacked")
        require(_as_int(marker.get("minimum_backups_before_stability"), field="minimum_backups_before_stability") == MIN_BACKUPS, "MCTS min-backup contract drift")
        require(_as_int(marker.get("stable_root_leader_observations_required"), field="stable_root_leader_observations_required") == STABLE_LEADER_OBSERVATIONS, "MCTS stability contract drift")
        require(_as_int(marker.get("maximum_backups_per_decision"), field="maximum_backups_per_decision") == MAX_BACKUPS, "MCTS max-backup contract drift")
        require(
            stop_reason
            in {
                PROVEN_TERMINAL_WIN_STOP_REASON,
                "stable_root_leader",
                "maximum_backups",
                "decision_deadline",
                "tree_exhausted",
            },
            "MCTS stop reason drift",
        )
        leader = _as_int(marker.get("deterministic_root_leader_observations"), field="deterministic_root_leader_observations")
        if stop_reason == "stable_root_leader":
            require(backups >= MIN_BACKUPS and leader >= STABLE_LEADER_OBSERVATIONS, "stable stop lacks qualified backups/leader")
        if stop_reason == "maximum_backups":
            require(backups == MAX_BACKUPS, "hard backup stop did not reach cap")
        counts = {
            field: _as_int(marker.get(field), field=field)
            for field in (
                "actor_change_boundary_leaf_count",
                "chance_boundary_leaf_count",
                "boundary_leaf_count",
            )
        }
        require(all(value >= 0 for value in counts.values()), "negative boundary-leaf count")
        require(
            max(counts["actor_change_boundary_leaf_count"], counts["chance_boundary_leaf_count"])
            <= counts["boundary_leaf_count"]
            <= counts["actor_change_boundary_leaf_count"] + counts["chance_boundary_leaf_count"],
            "boundary leaf total does not reflect chance/actor boundaries",
        )
        if terminal_win_stop:
            require(
                _as_int(marker.get("search_step_calls"), field="search_step_calls")
                >= SIMULATOR_LANES,
                "terminal-win receipt did not initialize both native lanes",
            )
            terminal_win_proof = _validate_r246_terminal_win_proof(
                marker,
                legal_actions=legal_actions,
                selected_action=selected,
                lane_receipt=lane_receipt,
                completed_backups=backups,
                observation=observation,
            )
        else:
            require(
                marker.get("terminal_win_proof") is None,
                "ordinary MCTS stop carries a terminal-win override proof",
            )
            require(
                marker.get("proven_deterministic_terminal_win_this_turn") in (None, False),
                "ordinary MCTS stop claims terminal-win override authority",
            )
            terminal_win_proof = None
        return {
            "mode": mode,
            "selected_action": selected,
            "direct_action": direct,
            "degraded": False,
            "lane_receipt": lane_receipt,
            "completed_backups": backups,
            "boundary_leaf_counts": counts,
            "child_cuda_runtime_before_search": child_cuda,
            "terminal_win_proof": terminal_win_proof,
        }

    if mode == "zero_backup_precomputed_direct_fallback":
        require(selected == direct, "clean-zero fallback differs from parent direct action")
        require(marker.get("mcts_action_authority") is False, "clean-zero fallback claimed MCTS authority")
        require(marker.get("zero_backup_precomputed_direct_fallback") is True, "clean-zero marker is missing its explicit mode")
        require(marker.get("clean_deadline_cleanup_complete") is True, "clean-zero marker lacks cleanup completion")
        require(_as_int(marker.get("completed_backups"), field="completed_backups") == 0, "clean-zero marker has backups")
        require(marker.get("stop_reason") == "decision_deadline", "clean-zero marker lacks deadline stop")
        require(
            _as_int(marker.get("search_step_calls"), field="search_step_calls")
            == SIMULATOR_LANES,
            "clean-zero marker did not step both lanes",
        )
        require(
            _as_int(
                marker.get("max_simulator_calls_in_flight"),
                field="max_simulator_calls_in_flight",
            )
            == SIMULATOR_LANES,
            "clean-zero marker did not reserve both lanes",
        )
        microbatches = marker.get("microbatch_sizes")
        require(
            isinstance(microbatches, (list, tuple)) and not microbatches,
            "clean-zero marker admits a frozen-evaluator batch",
        )
        lane_receipt = _validate_two_lane_composites(marker, allow_zero_depth=True)
        child_cuda = _validate_optional_child_cuda_observation(marker)
        require(
            child_cuda is not None,
            "clean-zero decision lacks child CUDA observation before search",
        )
        cleanup = _as_mapping(marker.get("exact_child_cleanup_and_reap"), field="clean-zero child cleanup")
        reap = _as_mapping(cleanup.get("reap"), field="clean-zero child reap")
        require(reap.get("reaped") is True, "clean-zero exact child was not reaped")
        return {
            "mode": mode,
            "selected_action": selected,
            "direct_action": direct,
            "degraded": False,
            "lane_receipt": lane_receipt,
            "clean_zero_reap": dict(reap),
            "child_cuda_runtime_before_search": child_cuda,
        }

    raise HarnessContractError(f"unsupported r244 decision mode: {mode!r}")


def validate_degraded_marker(
    marker: Mapping[str, Any], *, legal_actions: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Require exact-child containment evidence for a degraded direct action."""

    selected = as_action(marker.get("selected_action"), field="degraded selected_action")
    direct = as_action(marker.get("direct_action"), field="degraded direct_action")
    normalized_legal = [as_action(row, field="legal action") for row in legal_actions]
    require(selected == direct and selected in normalized_legal, "degraded fallback is not the precomputed legal direct action")
    require(marker.get("mcts_action_authority") is False, "degraded fallback claimed MCTS authority")
    require(marker.get("action_authority") == "precomputed_frozen_r195_direct_action", "degraded fallback authority drift")
    require(_as_int(marker.get("complete_action_cap"), field="complete_action_cap") == COMPLETE_ACTION_CAP, "degraded cap drift")
    require(_as_int(marker.get("configured_simulator_lane_count"), field="configured_simulator_lane_count") == SIMULATOR_LANES, "degraded lane count drift")
    validate_cuda_runtime_observation(
        marker.get("parent_cuda_runtime_before_search"),
        field="parent_cuda_runtime_before_search",
    )
    fault = _as_mapping(marker.get("child_fault"), field="degraded child_fault")
    require(fault.get("configured_simulator_lane_count") == SIMULATOR_LANES, "degraded fault lane count drift")
    reap = _as_mapping(fault.get("child_reap"), field="degraded child reap")
    require(reap.get("reaped") is True, "degraded exact child was not reaped")
    progress = marker.get("per_lane_progress")
    require(isinstance(progress, Mapping), "degraded lane progress is not an object")
    for raw_lane in progress:
        lane = int(raw_lane)
        require(0 <= lane < SIMULATOR_LANES, "degraded progress includes a non-r238 lane")
    return {
        "selected_action": selected,
        "child_reap": dict(reap),
        "fault_code": fault.get("code"),
        "child_cuda_runtime_before_search": _validate_optional_child_cuda_observation(marker),
    }


def validate_full_game_success(marker: Mapping[str, Any], *, mcts_decision_count: int) -> None:
    require(
        _as_int(marker.get("configured_simulator_lane_count"), field="configured_simulator_lane_count")
        == SIMULATOR_LANES,
        "full-game success lane count drift",
    )
    require(
        _as_int(marker.get("complete_action_cap"), field="complete_action_cap")
        == COMPLETE_ACTION_CAP,
        "full-game success cap drift",
    )
    require(
        _as_int(marker.get("degraded_fault_count"), field="degraded_fault_count") == 0,
        "full-game success marker claims a degraded game",
    )
    require(
        _as_int(marker.get("mcts_branching_decisions"), field="mcts_branching_decisions")
        == mcts_decision_count,
        "full-game MCTS decision count drift",
    )
    validate_cuda_runtime_observation(
        marker.get("parent_cuda_runtime_before_search"),
        field="parent_cuda_runtime_before_search",
    )
