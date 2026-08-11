"""Fail-closed contract for r241's isolated Elmo collection endpoint.

The historical Elmo (:8765) and Bert (:8766) workers are deliberately not
participants in r241.  This module only prepares the separately named Elmo
endpoint at ``192.168.1.143:8767``.  It does not open a socket, start a worker,
load policy weights, invoke native simulator functions, or run a game.

The optional ``serve`` wrapper in
``scripts/launch_r241_elmo_official_r236_worker.py`` calls these checks before
it delegates to the generic remote-worker process.  Keeping the identity and
receipt work here lets the r241 trainer validate an explicit endpoint without
reusing the legacy fleet defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import paths as runtime_paths
from . import r241_baseline_payload_snapshot as baseline_payload_snapshot
from .r241_checkpoint_receipts import (
    BASELINE_ADAPTER_ROSTER_SHA256,
    IMMUTABLE_ADAPTER_SLOT_PREFIX,
    R241CheckpointReceiptError,
    authenticated_source_snapshot_provenance,
    validate_r241_h10_adapter_source_binding,
)
from .r241_direct_policy_runtime import (
    FORBIDDEN_LIBCG_OVERRIDE_KEYS,
    R241_DIRECT_POLICY_ONLY_ENV,
    R241_DIRECT_POLICY_RECEIPT_ENV,
    R241_H10_ADAPTER_RECEIPT_BASENAME,
    R241_H10_CONTENT_SHA256,
    R241_H10_DIR_NAME,
    R241_H10_MODEL_SHA256,
    R241_H10_OPPONENT_ID,
    R241_OFFICIAL_LIBCG_RECEIPT_FILENAME,
    R241_OFFICIAL_LINUX_LIBCG_SHA256,
    R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME,
    R241_REVISION,
    R241DirectPolicyRuntimeError,
    assert_direct_policy_environment,
    read_json_object,
    sha256_file,
    validate_sealed_official_libcg,
)
from .r241_marnie_direct_policy_adapter import (
    R241_H10_MATCHUP_TREE_SHA256,
    R241_MARNIE_ADAPTER_RECEIPT_SCHEMA,
    validate_r241_marnie_direct_policy_adapter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

R241_ELMO_REMOTE_COLLECTION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_collection/v1"
)
R241_ELMO_REMOTE_HOST_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_host/v1"
)
R241_ELMO_REMOTE_RUNTIME_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_runtime/v1"
)
R241_ELMO_REMOTE_GAMEPLAY_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_gameplay/v1"
)
R241_ELMO_REMOTE_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_manifest/v1"
)
R241_ELMO_REMOTE_ENDPOINT_REGISTRY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_remote_endpoint_registry/v1"
)
R241_SOURCE_SNAPSHOT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1"
)
R241_SOURCE_SNAPSHOT_FILENAME = "r241-source-snapshot-manifest.json"
R241_ADAPTER_SLOT_MIGRATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_adapter_slot_migration/v1"
)
R241_BASELINE_PAYLOAD_REGISTRY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_baseline_payload_registry/v1"
)
R241_CHECKPOINT_TRANSPORT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport/v1"
)
R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport_staging/v1"
)
R241_ACTIVATION_OVERLAY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_activation_overlay/v1"
)
R241_ELMO_WORKER_IMAGE_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_worker_image/v1"
)
R241_SOURCE_SNAPSHOT_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_source_snapshot_staging/v1"
)
R241_OWNER_START_AUTHORIZATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization/v1"
)
R241_OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization_generator/v1"
)

ELMO_R241_ENDPOINT_ID = "elmo-r241-official-r236-direct-policy-8767"
ELMO_R241_ENDPOINT_HOST = "192.168.1.143"
ELMO_R241_ENDPOINT_PORT = 8767
ELMO_R241_ENDPOINT = f"{ELMO_R241_ENDPOINT_HOST}:{ELMO_R241_ENDPOINT_PORT}"
ELMO_R241_COLLECTION_CAPABILITY = "r241_direct_policy_collection_v1"

# These literals make an accidental inherited fleet/default conspicuous.  The
# only endpoint that may appear in an r241 remote collection contract is the
# explicit address above.
LEGACY_OR_DEFAULT_ENDPOINTS = (
    "192.168.1.143:8765",
    "192.168.1.158:8766",
    "bert.local:8766",
    "elmo:8765",
    "bert:8766",
)

COLLECTION_JOB_KINDS = (
    "play",
    "self_play",
    "self_play_multi",
    "runtime_probe",
)

DEFAULT_ELMO_RECEIPT_DIR = Path(
    "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/"
    "alakazam_new_list_direct_policy_r241/runtime/elmo-8767"
)
ELMO_REMOTE_ARM_FILE = DEFAULT_ELMO_RECEIPT_DIR / "REMOTE_WORKER_ARMED"
ELMO_REMOTE_PLANNED_ROTATION_EXIT_CODE = "75"
ELMO_REMOTE_MAX_SERVICE_JOBS = "0"
ELMO_REMOTE_WORKER_SAFETY_VERSION = "20260717"
ELMO_OUTPUTS_ROOT = Path("/mnt/Main/main/poke-bot-agent/outputs")
ELMO_R241_RUN_ROOT = ELMO_OUTPUTS_ROOT / "pure_rl/alakazam_new_list_direct_policy_r241"
ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT = Path("/workspace/checkpoint")
ELMO_R241_CHECKPOINT_TRANSPORT_ENV = "POKEBOT_REMOTE_CHECKPOINT_ROOT"
ELMO_R241_CHECKPOINT_VERIFY_PORT_ENV = "POKEBOT_ELMO_CHECKPOINT_VERIFY_PORT"
ELMO_R241_WORKER_IMAGE_ID_ENV = "R241_ELMO_OFFICIAL_R236_IMAGE_ID"
ELMO_R241_CHECKPOINT_TRANSPORT_FILENAME_SCHEME = (
    "poke_bot.remote_jobs.digest_addressed_basename/v1"
)
HOST_RECEIPT_FILENAME = "host-preflight.json"
RUNTIME_RECEIPT_FILENAME = "runtime-preflight.json"
GAMEPLAY_RECEIPT_FILENAME = "gameplay-preflight.json"
MANIFEST_FILENAME = "preflight-manifest.json"

R195_LEARNER_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
# Exact typed owner clarification currently admitted by this worker.  The
# digest remains dynamic and is verified from the immutable source manifest;
# this literal prevents silently accepting a future selector-scope change.
R241_LATEST_OWNER_CLARIFICATION_REVISION = 251

FORBIDDEN_POLICY_ENVIRONMENT_PREFIXES = (
    "POKEBOT_MCTS_",
    "POKEBOT_RTP_",
    "POKEBOT_BELIEF_",
    "POKEBOT_POKE_RLM_",
    "POKEBOT_SLOWKING_DISTILL_",
    "POKEBOT_GUIDE2VEC_",
)
FORBIDDEN_POLICY_ENVIRONMENT_EXACT = {
    *FORBIDDEN_LIBCG_OVERRIDE_KEYS,
    "POKEBOT_ALLOW_ORACLE_DECK",
    # A receipt for CG_LIB_PATH does not constrain a dynamic-loader override.
    # Keep the sealed r236 parent as the only native-library selection path.
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
}
FORBIDDEN_POLICY_ENVIRONMENT_TOKENS = (
    "MCTS",
    "RTP",
    "RECURSIVE_TURN",
    "BELIEF",
    "GUIDE2VEC",
)
# This is a direct-policy safety selector, not an enabled planner.  The
# sealed Compose contract and the direct-policy runtime both require this
# exact zero value.  Any other recursive-turn selector or value remains
# forbidden below.
SAFE_FORBIDDEN_TOKEN_ENVIRONMENT_VALUES = {
    "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
}
ALLOWED_SEARCH_ENVIRONMENT_KEYS = {
    "POKEBOT_SEARCH_MODE",
    "POKEBOT_SUBMISSION_SEARCH_DISABLE",
}

REQUIRED_SOURCE_SNAPSHOT_FILES = (
    "state/alakazam-new-list-direct-policy-r241.json",
    "state/alakazam-new-list-direct-r241-runtime-registry.json",
    "state/matchup_adapter_roster.json",
    "config/rl_protocol.yaml",
    "poke_bot/r241_direct_policy_runtime.py",
    "poke_bot/r241_baseline_payload_snapshot.py",
    "poke_bot/paths.py",
    "poke_bot/r241_marnie_direct_policy_adapter.py",
    "poke_bot/r241_elmo_official_r236_remote_worker.py",
    "poke_bot/r241_checkpoint_receipts.py",
    # The exact digest-addressed checkpoint path ABI used by the isolated
    # :8767 endpoint lives here.  It is part of the executable snapshot
    # closure, rather than an untracked mutable trainer dependency.
    "poke_bot/remote_jobs.py",
    "scripts/launch_r241_elmo_official_r236_worker.py",
    "scripts/run_remote_worker.py",
    "deploy/elmo/docker-compose.r241-elmo-official-r236-remote-worker.yml.template",
    "deploy/elmo/r241-elmo-official-r236-remote-worker.env.template",
    "deploy/systemd/pokebot-r241-elmo-official-r236-remote-worker.service.template",
)


class R241ElmoRemoteWorkerError(RuntimeError):
    """An r241 Elmo collection endpoint input or receipt is unsafe."""


@dataclass(frozen=True)
class R241ElmoBaselinePayload:
    """One receipt-bound external baseline library mounted on Elmo."""

    root: Path
    manifest: Path
    manifest_sha256: str
    baseline_tree_sha256: str
    file_inventory_sha256: str
    staging_receipt: Path
    staging_receipt_sha256: str
    canonical_roster_receipt: Path
    canonical_roster_receipt_sha256: str
    baseline_manifest_sha256: str
    baseline_roster_sha256: str


@dataclass(frozen=True)
class R241ElmoCheckpointTransport:
    """One external content-addressed checkpoint mount for the :8767 worker.

    Source and baseline receipts retain their Elmo-host absolute paths.  This
    deliberately narrower mount is the sole container-path exception because
    the generic remote protocol has always named reloadable checkpoints beneath
    ``/workspace/checkpoint``.
    """

    host_root: Path
    container_root: Path
    staging_receipt: Path
    staging_receipt_sha256: str
    initial_checkpoint: Path
    initial_checkpoint_sha256: str


@dataclass(frozen=True)
class R241ElmoPreflight:
    """Verified inputs used to construct one deterministic receipt set."""

    repo_root: Path
    source_snapshot_manifest: Path
    checkpoint: Path
    cg_lib_path: Path
    adapter_receipt: Path
    learner_matchup_tree: Path
    matchup_runtime_marker: Path
    h10_matchup_tree: Path
    baseline_payload: R241ElmoBaselinePayload
    checkpoint_transport: R241ElmoCheckpointTransport
    environment: dict[str, str]
    sources: dict[str, dict[str, Any]]
    matchup_runtime: dict[str, Any]


def _regular_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241ElmoRemoteWorkerError(f"{label} must be a real directory: {raw}")
    return raw.resolve()


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241ElmoRemoteWorkerError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    try:
        info = raw.stat()
    except OSError as exc:
        raise R241ElmoRemoteWorkerError(f"{label} is unreadable: {raw}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise R241ElmoRemoteWorkerError(f"{label} is not a regular file: {raw}")
    return raw.resolve()


def _require_elmo_absolute_path(path: Path | str, *, label: str) -> Path:
    """Reject container remaps for evidence whose receipt binds host paths.

    The sealed-r236 and H10 receipts name absolute paths.  Mounting them under
    ``/workspace`` would make a host receipt falsely appear to authorize a
    different runtime root, so the :8767 worker deliberately uses identical
    Elmo host paths inside its container.
    """

    raw = str(path).strip()
    if not raw:
        raise R241ElmoRemoteWorkerError(f"{label} must be a non-empty Elmo host path")
    resolved = Path(raw).expanduser().resolve()
    rendered = str(resolved)
    if not resolved.is_absolute() or rendered == "/workspace" or rendered.startswith(
        "/workspace/"
    ):
        raise R241ElmoRemoteWorkerError(
            f"{label} must retain its Elmo host absolute path, not a container remap: {resolved}"
        )
    return resolved


def _checkpoint_transport_member(
    path: Path | str,
    *,
    root: Path,
    label: str,
) -> Path:
    """Return a regular member of the one permitted container checkpoint root."""

    candidate = _regular_file(path, label=label)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise R241ElmoRemoteWorkerError(
            f"{label} must remain beneath the r241 checkpoint transport root {root}"
        ) from exc
    if candidate.parent != root:
        raise R241ElmoRemoteWorkerError(
            f"{label} must use the flat content-addressed checkpoint transport root"
        )
    return candidate


def _checkpoint_transport_receipt_path(
    path: object,
    *,
    root: object,
    label: str,
) -> str:
    """Validate a serialized flat member of the remote checkpoint ABI.

    Manifest validation may run on a trainer that does not mount Elmo's
    container filesystem, so unlike :func:`_checkpoint_transport_member` this
    only proves the exact serialized path relationship.  The offline Elmo
    preflight has already verified the file itself and its digest.
    """

    rendered_root = str(root or "").strip()
    rendered_path = str(path or "").strip()
    if rendered_root != str(ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT):
        raise R241ElmoRemoteWorkerError(
            f"{label} has an unexpected checkpoint transport root: {rendered_root!r}"
        )
    candidate = Path(rendered_path)
    if not candidate.is_absolute() or candidate.parent != Path(rendered_root):
        raise R241ElmoRemoteWorkerError(
            f"{label} must be a flat member of {rendered_root}"
        )
    return str(candidate)


def _sha256(path: Path | str) -> str:
    return sha256_file(_regular_file(path, label="receipt-bound file"))


def _json_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    try:
        return read_json_object(path, label=label)
    except R241DirectPolicyRuntimeError as exc:
        raise R241ElmoRemoteWorkerError(str(exc)) from exc


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241ElmoRemoteWorkerError(f"{label} must be an exact integer")
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _json_digest(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def assert_r241_elmo_endpoint(endpoint: str) -> str:
    """Accept only the new literal Elmo endpoint; never infer a fleet default."""

    candidate = str(endpoint or "").strip()
    if candidate in LEGACY_OR_DEFAULT_ENDPOINTS:
        raise R241ElmoRemoteWorkerError(
            f"legacy/default endpoint is forbidden for r241: {candidate}"
        )
    if candidate != ELMO_R241_ENDPOINT:
        raise R241ElmoRemoteWorkerError(
            "r241 requires the one explicit eligible endpoint "
            f"{ELMO_R241_ENDPOINT!r}, got {candidate!r}"
        )
    return candidate


def _assert_clean_inherited_environment(
    environment: Mapping[str, str],
    *,
    cg_lib_path: Path,
    adapter_receipt: Path,
    learner_matchup_tree: Path,
    baseline_payload: R241ElmoBaselinePayload,
    checkpoint_transport: R241ElmoCheckpointTransport,
    source_execution_root: Path | None = None,
) -> None:
    """Reject, rather than silently scrub, inherited planning/runtime controls."""

    env = {str(key): str(value) for key, value in environment.items()}
    _require_sha256(
        env.get(ELMO_R241_WORKER_IMAGE_ID_ENV),
        label="r241 Elmo worker image ID environment",
    )
    forbidden = sorted(
        key
        for key in env
        if key in FORBIDDEN_POLICY_ENVIRONMENT_EXACT
        or key.startswith(FORBIDDEN_POLICY_ENVIRONMENT_PREFIXES)
        or (
            any(token in key.upper() for token in FORBIDDEN_POLICY_ENVIRONMENT_TOKENS)
            and SAFE_FORBIDDEN_TOKEN_ENVIRONMENT_VALUES.get(key) != env[key]
        )
        or ("LIBCG" in key.upper() and key != "CG_LIB_PATH")
        or ("SEARCH" in key.upper() and key not in ALLOWED_SEARCH_ENVIRONMENT_KEYS)
    )
    if forbidden:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo worker refuses inherited MCTS/RTP/search/private-CG "
            "selectors: "
            + ", ".join(forbidden)
        )

    required_if_present = {
        ELMO_R241_WORKER_IMAGE_ID_ENV: env[ELMO_R241_WORKER_IMAGE_ID_ENV],
        "CG_LIB_PATH": str(cg_lib_path),
        R241_DIRECT_POLICY_ONLY_ENV: "1",
        R241_DIRECT_POLICY_RECEIPT_ENV: str(adapter_receipt),
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_SEARCH_MODE": "policy",
        "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
        "POKEBOT_PUBLIC_MATCHUP_TREE_PATH": str(learner_matchup_tree),
        "POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE": "runtime",
        "PURE_RL_PUBLIC_MIX_LOCAL_ONLY": "0",
        "POKEBOT_REMOTE_ALLOWED_JOB_KINDS": ",".join(COLLECTION_JOB_KINDS),
        "POKEBOT_REMOTE_WORKER_CAPABILITY_TAGS": ELMO_R241_COLLECTION_CAPABILITY,
        "POKEBOT_REMOTE_WORKER_ARM_FILE": str(ELMO_REMOTE_ARM_FILE),
        "POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE": ELMO_REMOTE_PLANNED_ROTATION_EXIT_CODE,
        "POKEBOT_REMOTE_MAX_SERVICE_JOBS": ELMO_REMOTE_MAX_SERVICE_JOBS,
        "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": ELMO_REMOTE_WORKER_SAFETY_VERSION,
        # The code-only source snapshot intentionally contains no baselines.
        # If a parent supplied one, it must be this independently attested
        # external library rather than a checkout-relative fallback.
        "POKEBOT_BASELINES_DIR": str(baseline_payload.root),
        "POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST": str(baseline_payload.manifest),
        "POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST_SHA256": baseline_payload.manifest_sha256,
        "POKEBOT_R241_BASELINE_PAYLOAD_TREE_SHA256": baseline_payload.baseline_tree_sha256,
        ELMO_R241_CHECKPOINT_TRANSPORT_ENV: str(checkpoint_transport.container_root),
    }
    if source_execution_root is not None:
        required_if_present["PYTHONPATH"] = str(source_execution_root)
    mismatched = [
        f"{key}={env[key]!r}"
        for key, expected in required_if_present.items()
        if key in env and env[key] != expected
    ]
    if mismatched:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo worker refuses a conflicting inherited control: "
            + ", ".join(mismatched)
        )


def build_r241_elmo_collection_environment(
    *,
    cg_lib_path: Path | str,
    adapter_receipt: Path | str,
    learner_matchup_tree: Path | str,
    baseline_payload: R241ElmoBaselinePayload,
    checkpoint_transport: R241ElmoCheckpointTransport,
    adapter_format: str,
    source_execution_root: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the one sealed environment accepted by the isolated worker.

    This does not mutate ``os.environ``.  The caller must use this mapping for
    the worker lifetime so a parent shell cannot reintroduce a planner or a
    private libcg path after preflight.
    """

    cg_root = _regular_directory(cg_lib_path, label="CG_LIB_PATH")
    receipt = _regular_file(adapter_receipt, label="H10 adapter receipt")
    learner_tree = _regular_file(learner_matchup_tree, label="r195 learner tree")
    baseline_root = _regular_directory(
        baseline_payload.root, label="r241 external baseline payload root"
    )
    baseline_manifest = _regular_file(
        baseline_payload.manifest, label="r241 external baseline payload manifest"
    )
    if baseline_manifest.parent != baseline_root:
        raise R241ElmoRemoteWorkerError(
            "r241 external baseline manifest must be rooted in POKEBOT_BASELINES_DIR"
        )
    source_root = (
        _regular_directory(source_execution_root, label="r241 source execution root")
        if source_execution_root is not None
        else None
    )
    source = dict(os.environ if environment is None else environment)
    _assert_clean_inherited_environment(
        source,
        cg_lib_path=cg_root,
        adapter_receipt=receipt,
        learner_matchup_tree=learner_tree,
        baseline_payload=baseline_payload,
        checkpoint_transport=checkpoint_transport,
        source_execution_root=source_root,
    )
    format_name = str(adapter_format or "").strip()
    if not format_name:
        raise R241ElmoRemoteWorkerError("Matchup Adapter format is required")

    sealed = {str(key): str(value) for key, value in source.items()}
    sealed.update(
        {
            ELMO_R241_WORKER_IMAGE_ID_ENV: source[ELMO_R241_WORKER_IMAGE_ID_ENV],
            "CG_LIB_PATH": str(cg_root),
            R241_DIRECT_POLICY_ONLY_ENV: "1",
            R241_DIRECT_POLICY_RECEIPT_ENV: str(receipt),
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
            "POKEBOT_SEARCH_MODE": "policy",
            "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
            "POKEBOT_PUBLIC_MATCHUP_TREE_PATH": str(learner_tree),
            "POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE": "runtime",
            "POKEBOT_MATCHUP_ADAPTER_FORMAT": format_name,
            # r241 retains the established diverse public mix.  A default
            # local-only collection topology would silently change that mix.
            "PURE_RL_PUBLIC_MIX_LOCAL_ONLY": "0",
            "POKEBOT_REMOTE_ALLOWED_JOB_KINDS": ",".join(COLLECTION_JOB_KINDS),
            "POKEBOT_REMOTE_WORKER_CAPABILITY_TAGS": ELMO_R241_COLLECTION_CAPABILITY,
            "POKEBOT_REMOTE_WORKER_ARM_FILE": str(ELMO_REMOTE_ARM_FILE),
            "POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE": ELMO_REMOTE_PLANNED_ROTATION_EXIT_CODE,
            "POKEBOT_REMOTE_MAX_SERVICE_JOBS": ELMO_REMOTE_MAX_SERVICE_JOBS,
            "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": ELMO_REMOTE_WORKER_SAFETY_VERSION,
            "POKEBOT_BASELINES_DIR": str(baseline_root),
            "POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST": str(baseline_manifest),
            "POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST_SHA256": baseline_payload.manifest_sha256,
            "POKEBOT_R241_BASELINE_PAYLOAD_TREE_SHA256": baseline_payload.baseline_tree_sha256,
            ELMO_R241_CHECKPOINT_TRANSPORT_ENV: str(
                checkpoint_transport.container_root
            ),
        }
    )
    if source_root is not None:
        sealed["PYTHONPATH"] = str(source_root)
    try:
        assert_direct_policy_environment(sealed)
    except R241DirectPolicyRuntimeError as exc:
        raise R241ElmoRemoteWorkerError(str(exc)) from exc
    return sealed


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise R241ElmoRemoteWorkerError(f"{label} must be an object")
    return dict(value)


def _require_sha256(value: object, *, label: str) -> str:
    try:
        return baseline_payload_snapshot.valid_sha256(value, label=label)
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241ElmoRemoteWorkerError(str(exc)) from exc


def _require_pending_snapshot_projection(
    registry: Mapping[str, Any], *, owner_contract_sha256: str
) -> dict[str, Any]:
    """Require the source-copy registry to remain a non-runnable base layer."""

    source = _require_mapping(
        registry.get("source_snapshot"), label="r241 base source snapshot registry"
    )
    if (
        source.get("schema") != R241_SOURCE_SNAPSHOT_SCHEMA
        or source.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or source.get("owner_contract_sha256") != owner_contract_sha256
        or source.get("status") != "pending_immutable_source_snapshot"
        or str(source.get("manifest_sha256") or "")
        or str(source.get("source_tree_sha256") or "")
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 source snapshot must retain a pending, identity-free base registry"
        )
    hosts = _require_mapping(source.get("hosts"), label="r241 base source snapshot hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241ElmoRemoteWorkerError(
            "r241 base source snapshot registry must name exactly Inzi and Elmo"
        )
    for host_name in ("inzi", "elmo"):
        host = _require_mapping(hosts.get(host_name), label=f"r241 base {host_name} snapshot")
        if (
            str(host.get("root") or "")
            or str(host.get("manifest") or "")
            or not str(host.get("outputs_root") or "")
            or not Path(str(host.get("outputs_root") or "")).is_absolute()
        ):
            raise R241ElmoRemoteWorkerError(
                f"r241 base {host_name} source snapshot names mutable execution paths"
            )
    if str(hosts["elmo"].get("outputs_root") or "") != str(ELMO_OUTPUTS_ROOT):
        raise R241ElmoRemoteWorkerError("r241 base Elmo outputs root drifted")
    return source


def _immutable_external_receipt(
    path: Path | str,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    """Read only a create-only attestation under Elmo's external output root."""

    receipt = _regular_file(
        _require_elmo_absolute_path(path, label=label), label=label
    )
    if receipt.stat().st_mode & 0o222:
        raise R241ElmoRemoteWorkerError(f"{label} must be immutable")
    if ELMO_OUTPUTS_ROOT not in receipt.parents:
        raise R241ElmoRemoteWorkerError(f"{label} must remain under Elmo external outputs")
    if _sha256(receipt) != _require_sha256(expected_sha256, label=f"{label} sha256"):
        raise R241ElmoRemoteWorkerError(f"{label} checksum drifted")
    return receipt


def _validate_elmo_checkpoint_transport(
    *,
    host_root: Path | str,
    staging_receipt: Path | str,
    staging_receipt_sha256: str,
    checkpoint: Path | str,
    source_root: Path,
    baselines_root: Path,
    owner_contract_sha256: str,
) -> R241ElmoCheckpointTransport:
    """Validate the one explicit content-addressed checkpoint mount.

    The generic remote protocol maps every Elmo checkpoint to
    ``/workspace/checkpoint/<digest-addressed-name>``.  That path is a narrow
    transport ABI, not a source-tree remap.  The immutable staging receipt
    binds the host-side publisher root to the container target; the live
    preflight then verifies the initial checkpoint bytes at that target.
    """

    receipt_path = _immutable_external_receipt(
        staging_receipt,
        expected_sha256=staging_receipt_sha256,
        label="r241 Elmo checkpoint transport staging receipt",
    )
    _receipt_path, staged = _json_object(
        receipt_path, label="r241 Elmo checkpoint transport staging receipt"
    )
    transport = _require_mapping(
        staged.get("checkpoint_transport"),
        label="r241 Elmo checkpoint transport binding",
    )
    declared_host_root = str(transport.get("host_root") or "").strip()
    declared_container_root = str(transport.get("container_root") or "").strip()
    expected_owner_contract = _require_sha256(
        owner_contract_sha256, label="r241 checkpoint transport owner contract sha256"
    )
    if (
        staged.get("schema") != R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA
        or staged.get("revision") != R241_REVISION
        or staged.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or staged.get("status") != "passed"
        or staged.get("passed") is not True
        or staged.get("owner_contract_sha256") != expected_owner_contract
        or transport.get("schema") != R241_CHECKPOINT_TRANSPORT_SCHEMA
        or transport.get("endpoint_id") != ELMO_R241_ENDPOINT_ID
        or transport.get("host_role") != "elmo"
        or transport.get("verification_endpoint") != ELMO_R241_ENDPOINT
        or transport.get("verification_port") != ELMO_R241_ENDPOINT_PORT
        or transport.get("environment_key") != ELMO_R241_CHECKPOINT_TRANSPORT_ENV
        or transport.get("container_root")
        != str(ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT)
        or transport.get("remote_path_prefix")
        != f"{ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT}/"
        or transport.get("read_only_container_mount") is not True
        or transport.get("same_absolute_source_and_baseline_paths_preserved")
        is not True
    ):
        raise R241ElmoRemoteWorkerError("r241 checkpoint transport staging receipt drifted")
    content_addressing = _require_mapping(
        transport.get("content_addressing"),
        label="r241 checkpoint transport content addressing",
    )
    if (
        content_addressing.get("algorithm") != "sha256"
        or content_addressing.get("filename_scheme")
        != ELMO_R241_CHECKPOINT_TRANSPORT_FILENAME_SCHEME
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 checkpoint transport must use the generic digest-addressed filename ABI"
        )
    requested_host_root = str(host_root or "").strip()
    if (
        not declared_host_root
        or not Path(declared_host_root).is_absolute()
        or declared_host_root.startswith("/workspace/")
        or declared_host_root == "/workspace"
        or requested_host_root != declared_host_root
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 checkpoint transport host root must exactly match its external staging receipt"
        )
    for forbidden_root, label in (
        (source_root, "source snapshot"),
        (baselines_root, "baseline payload"),
    ):
        if Path(declared_host_root) == forbidden_root:
            raise R241ElmoRemoteWorkerError(
                f"r241 checkpoint transport must not reuse the {label} root"
            )
    container_root = _regular_directory(
        ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT,
        label="r241 container checkpoint transport root",
    )
    if str(container_root) != declared_container_root:
        raise R241ElmoRemoteWorkerError(
            "r241 checkpoint transport container root is not the generic remote ABI path"
        )
    initial = _require_mapping(
        staged.get("initial_checkpoint"), label="r241 checkpoint transport initial checkpoint"
    )
    checkpoint_path = _checkpoint_transport_member(
        checkpoint,
        root=container_root,
        label="r241 collection checkpoint",
    )
    initial_sha256 = _require_sha256(
        initial.get("sha256"), label="r241 checkpoint transport initial checkpoint sha256"
    )
    if (
        initial.get("container_path") != str(checkpoint_path)
        or _sha256(checkpoint_path) != initial_sha256
        or not checkpoint_path.name.endswith(
            f".{initial_sha256.removeprefix('sha256:')[:16]}{checkpoint_path.suffix}"
        )
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 initial checkpoint is not the receipt-bound digest-addressed transport member"
        )
    return R241ElmoCheckpointTransport(
        host_root=Path(declared_host_root),
        container_root=container_root,
        staging_receipt=receipt_path,
        staging_receipt_sha256=_sha256(receipt_path),
        initial_checkpoint=checkpoint_path,
        initial_checkpoint_sha256=initial_sha256,
    )


def _baseline_payload_contract_for_elmo(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the typed external-baseline projection without loading code.

    The source snapshot is deliberately code-only.  This verifies that its
    registry names a separately receipted baseline library and returns the
    Elmo row for the host-local inventory validation below.  ``pending`` is
    structurally valid in a staged source tree, but preflight itself rejects it
    before a worker can be served.
    """

    contract = _require_mapping(
        registry.get("baseline_payloads"), label="r241 baseline payload registry"
    )
    if (
        contract.get("schema") != R241_BASELINE_PAYLOAD_REGISTRY_SCHEMA
        or contract.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or contract.get("separately_mounted_and_receipted") is not True
        or contract.get("source_snapshot_fallback_allowed") is not False
        or contract.get("status")
        not in {"pending_external_baseline_payload_snapshot", "ready"}
    ):
        raise R241ElmoRemoteWorkerError("r241 baseline payload registry contract drifted")
    hosts = _require_mapping(contract.get("hosts"), label="r241 baseline payload hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241ElmoRemoteWorkerError(
            "r241 baseline payload registry must name exactly Inzi and Elmo"
        )
    fields = (
        "root",
        "manifest",
        "manifest_sha256",
        "baseline_tree_sha256",
        "staging_receipt",
        "staging_receipt_sha256",
    )
    canonical_fields = (
        "canonical_roster_receipt",
        "canonical_roster_receipt_sha256",
        "canonical_baseline_manifest_sha256",
        "canonical_baseline_roster_sha256",
    )
    pending = contract.get("status") == "pending_external_baseline_payload_snapshot"
    if pending:
        if any(str(contract.get(field) or "").strip() for field in canonical_fields):
            raise R241ElmoRemoteWorkerError(
                "pending r241 baseline payload names a canonical roster receipt"
            )
    else:
        canonical_receipt = str(contract.get("canonical_roster_receipt") or "").strip()
        if not canonical_receipt or not Path(canonical_receipt).is_absolute():
            raise R241ElmoRemoteWorkerError(
                "ready r241 baseline payload lacks an external canonical roster receipt"
            )
        for field in canonical_fields[1:]:
            try:
                baseline_payload_snapshot.valid_sha256(
                    contract.get(field), label=f"r241 canonical baseline {field}"
                )
            except baseline_payload_snapshot.R241BaselinePayloadError as exc:
                raise R241ElmoRemoteWorkerError(str(exc)) from exc
    for host_name in ("inzi", "elmo"):
        host = _require_mapping(hosts.get(host_name), label=f"r241 {host_name} baseline payload")
        if pending:
            if any(str(host.get(field) or "").strip() for field in fields):
                raise R241ElmoRemoteWorkerError(
                    f"pending r241 {host_name} baseline payload names a fallback"
                )
            continue
        for field in ("root", "manifest", "staging_receipt"):
            raw = str(host.get(field) or "").strip()
            if not raw or not Path(raw).is_absolute():
                raise R241ElmoRemoteWorkerError(
                    f"ready r241 {host_name} baseline payload lacks an absolute {field}"
                )
        for field in (
            "manifest_sha256",
            "baseline_tree_sha256",
            "staging_receipt_sha256",
        ):
            try:
                baseline_payload_snapshot.valid_sha256(
                    host.get(field), label=f"r241 {host_name} baseline payload {field}"
                )
            except baseline_payload_snapshot.R241BaselinePayloadError as exc:
                raise R241ElmoRemoteWorkerError(str(exc)) from exc
    return contract


def _source_snapshot_from_staging_receipt(
    *,
    source_snapshot: Mapping[str, Any],
    owner_contract_sha256: str,
    staging_receipt: Path | str,
    staging_receipt_sha256: str,
) -> dict[str, Any]:
    """Revalidate Elmo's create-only source staging receipt before preflight."""

    receipt_path = _immutable_external_receipt(
        staging_receipt,
        expected_sha256=staging_receipt_sha256,
        label="r241 Elmo source staging receipt",
    )
    _path, staged = _json_object(receipt_path, label="r241 Elmo source staging receipt")
    binding = _require_mapping(
        staged.get("source_snapshot"), label="r241 Elmo source staging binding"
    )
    expected = {
        "schema": R241_SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": source_snapshot.get("root"),
        "source_execution_root": source_snapshot.get("root"),
        "manifest": source_snapshot.get("manifest"),
        "manifest_sha256": source_snapshot.get("manifest_sha256"),
        "source_tree_sha256": source_snapshot.get("source_tree_sha256"),
        "file_inventory_sha256": source_snapshot.get("file_inventory_sha256"),
        "owner_contract_sha256": _require_sha256(
            owner_contract_sha256, label="r241 source snapshot owner contract sha256"
        ),
        "outputs_root": str(ELMO_OUTPUTS_ROOT),
    }
    if (
        staged.get("schema") != R241_SOURCE_SNAPSHOT_STAGING_SCHEMA
        or staged.get("revision") != R241_REVISION
        or staged.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or staged.get("status") != "passed"
        or staged.get("passed") is not True
        or staged.get("operation")
        not in {
            "deterministic_stage_or_verify",
            "verify_published_immutable_source_snapshot",
        }
        or any(binding.get(key) != value for key, value in expected.items())
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo source staging receipt binding drifted")
    closure = _require_mapping(
        staged.get("closure"), label="r241 Elmo source staging closure"
    )
    if closure.get("baseline_payloads_separate_and_receipted") is not True or (
        staged.get("operation") == "verify_published_immutable_source_snapshot"
        and closure.get("verified_from_published_immutable_root") is not True
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo source staging receipt binding drifted")
    return {
        **expected,
        "staging_receipt": str(receipt_path),
        "staging_receipt_sha256": _sha256(receipt_path),
    }


def _baseline_payload_contract_from_staging_receipt(
    *,
    owner_contract_sha256: str,
    staging_receipt: Path | str,
    staging_receipt_sha256: str,
    canonical_roster_receipt: Path | str,
    canonical_roster_receipt_sha256: str,
) -> dict[str, Any]:
    """Derive Elmo's ready baseline binding from immutable staging evidence."""

    receipt_path = _immutable_external_receipt(
        staging_receipt,
        expected_sha256=staging_receipt_sha256,
        label="r241 Elmo baseline staging receipt",
    )
    _path, staged = _json_object(receipt_path, label="r241 Elmo baseline staging receipt")
    snapshot = _require_mapping(
        staged.get("baseline_payload_snapshot"),
        label="r241 Elmo baseline staging payload",
    )
    canonical = _require_mapping(
        staged.get("canonical_roster_receipt"),
        label="r241 Elmo baseline staging canonical roster",
    )
    canonical_path = _immutable_external_receipt(
        canonical_roster_receipt,
        expected_sha256=canonical_roster_receipt_sha256,
        label="r241 canonical baseline roster receipt",
    )
    expected_snapshot = {
        "schema": baseline_payload_snapshot.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": "elmo",
        "owner_contract_sha256": _require_sha256(
            owner_contract_sha256, label="r241 baseline payload owner contract sha256"
        ),
    }
    if (
        staged.get("schema") != baseline_payload_snapshot.BASELINE_PAYLOAD_STAGING_SCHEMA
        or staged.get("revision") != R241_REVISION
        or staged.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or staged.get("status") != "passed"
        or staged.get("passed") is not True
        or staged.get("operation") != "deterministic_stage_or_verify"
        or staged.get("receipt_outside_source_and_baseline_snapshot") is not True
        or any(snapshot.get(key) != value for key, value in expected_snapshot.items())
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo baseline staging receipt drifted")
    for binding, field in (
        (snapshot, "root"),
        (snapshot, "manifest"),
        (canonical, "path"),
    ):
        if not str(binding.get(field) or ""):
            raise R241ElmoRemoteWorkerError(
                f"r241 Elmo baseline staging receipt omits {field}"
            )
    for field in (
        "manifest_sha256",
        "baseline_tree_sha256",
        "file_inventory_sha256",
        "baseline_manifest_sha256",
        "baseline_roster_sha256",
    ):
        _require_sha256(snapshot.get(field), label=f"r241 baseline staging {field}")
    for field in ("sha256", "baseline_manifest_sha256", "baseline_roster_sha256"):
        _require_sha256(canonical.get(field), label=f"r241 baseline canonical {field}")
    if (
        canonical.get("baseline_manifest_sha256")
        != snapshot.get("baseline_manifest_sha256")
        or canonical.get("baseline_roster_sha256")
        != snapshot.get("baseline_roster_sha256")
        or canonical.get("sha256") != snapshot.get("canonical_roster_receipt_sha256")
        or canonical.get("path") != snapshot.get("canonical_roster_receipt")
        or canonical.get("path") != str(canonical_path)
        or canonical.get("sha256") != _sha256(canonical_path)
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo baseline staging canonical roster binding drifted"
        )
    return {
        "status": "ready",
        "hosts": {
            "elmo": {
                "root": snapshot["root"],
                "manifest": snapshot["manifest"],
                "manifest_sha256": snapshot["manifest_sha256"],
                "baseline_tree_sha256": snapshot["baseline_tree_sha256"],
                "staging_receipt": str(receipt_path),
                "staging_receipt_sha256": _sha256(receipt_path),
            }
        },
        "canonical_roster_receipt": str(canonical_path),
        "canonical_roster_receipt_sha256": _sha256(canonical_path),
        "canonical_baseline_manifest_sha256": canonical["baseline_manifest_sha256"],
        "canonical_baseline_roster_sha256": canonical["baseline_roster_sha256"],
    }


def _baseline_payload_identity(payload: R241ElmoBaselinePayload) -> dict[str, Any]:
    """Return the receipt-safe identity of a verified external baseline mount."""

    return {
        "schema": baseline_payload_snapshot.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": str(payload.root),
        "manifest": str(payload.manifest),
        "manifest_sha256": payload.manifest_sha256,
        "baseline_tree_sha256": payload.baseline_tree_sha256,
        "file_inventory_sha256": payload.file_inventory_sha256,
        "staging_receipt": str(payload.staging_receipt),
        "staging_receipt_sha256": payload.staging_receipt_sha256,
        "canonical_roster_receipt": str(payload.canonical_roster_receipt),
        "canonical_roster_receipt_sha256": payload.canonical_roster_receipt_sha256,
        "baseline_manifest_sha256": payload.baseline_manifest_sha256,
        "baseline_roster_sha256": payload.baseline_roster_sha256,
    }


def _checkpoint_transport_identity(
    transport: R241ElmoCheckpointTransport,
) -> dict[str, Any]:
    """Return the immutable receipt-safe checkpoint transport binding."""

    return {
        "schema": R241_CHECKPOINT_TRANSPORT_SCHEMA,
        "endpoint_id": ELMO_R241_ENDPOINT_ID,
        "host_role": "elmo",
        "verification_endpoint": ELMO_R241_ENDPOINT,
        "verification_port": ELMO_R241_ENDPOINT_PORT,
        "host_root": str(transport.host_root),
        "container_root": str(transport.container_root),
        "environment_key": ELMO_R241_CHECKPOINT_TRANSPORT_ENV,
        "remote_path_prefix": f"{transport.container_root}/",
        "content_addressing": {
            "algorithm": "sha256",
            "filename_scheme": ELMO_R241_CHECKPOINT_TRANSPORT_FILENAME_SCHEME,
        },
        "read_only_container_mount": True,
        "same_absolute_source_and_baseline_paths_preserved": True,
        "staging_receipt": str(transport.staging_receipt),
        "staging_receipt_sha256": transport.staging_receipt_sha256,
        "initial_checkpoint": {
            "container_path": str(transport.initial_checkpoint),
            "sha256": transport.initial_checkpoint_sha256,
        },
    }


def _validate_elmo_baseline_payload(
    *,
    contract: Mapping[str, Any],
    baselines_root: Path | str,
    source_root: Path,
    owner_contract_sha256: str,
) -> R241ElmoBaselinePayload:
    """Admit only the generic, immutable baseline mount named by Elmo's row."""

    if contract.get("status") != "ready":
        raise R241ElmoRemoteWorkerError(
            "r241 :8767 preflight requires a ready separately attested baseline payload"
        )
    host = _require_mapping(
        _require_mapping(contract.get("hosts"), label="r241 baseline payload hosts").get(
            "elmo"
        ),
        label="r241 Elmo baseline payload",
    )
    declared_root = _require_elmo_absolute_path(
        str(host.get("root") or ""), label="r241 Elmo baseline payload root"
    )
    requested_root = _regular_directory(
        baselines_root, label="r241 POKEBOT_BASELINES_DIR"
    )
    _require_elmo_absolute_path(requested_root, label="r241 POKEBOT_BASELINES_DIR")
    if requested_root != declared_root:
        raise R241ElmoRemoteWorkerError(
            "POKEBOT_BASELINES_DIR must equal the registry-bound Elmo baseline payload root"
        )
    declared_manifest = _require_elmo_absolute_path(
        str(host.get("manifest") or ""), label="r241 Elmo baseline payload manifest"
    )
    declared_staging = _regular_file(
        _require_elmo_absolute_path(
            str(host.get("staging_receipt") or ""),
            label="r241 Elmo baseline payload staging receipt",
        ),
        label="r241 Elmo baseline payload staging receipt",
    )
    if declared_staging.stat().st_mode & 0o222:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo baseline payload staging receipt must be immutable"
        )
    if ELMO_OUTPUTS_ROOT not in declared_staging.parents:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo baseline payload staging receipt must remain under external outputs"
        )
    canonical_roster_path = _regular_file(
        _require_elmo_absolute_path(
            str(contract.get("canonical_roster_receipt") or ""),
            label="r241 canonical baseline roster receipt",
        ),
        label="r241 canonical baseline roster receipt",
    )
    if (
        canonical_roster_path.stat().st_mode & 0o222
        or ELMO_OUTPUTS_ROOT not in canonical_roster_path.parents
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 canonical baseline roster receipt must be immutable external output evidence"
        )
    try:
        canonical_roster_path, canonical_roster = (
            baseline_payload_snapshot.validate_canonical_roster_receipt(
                canonical_roster_path,
                expected_sha256=str(contract.get("canonical_roster_receipt_sha256") or ""),
                owner_contract_sha256=_require_sha256(
                    owner_contract_sha256,
                    label="r241 canonical baseline owner contract sha256",
                ),
            )
        )
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241ElmoRemoteWorkerError(
            f"r241 canonical baseline roster receipt failed validation: {exc}"
        ) from exc
    if (
        requested_root == source_root
        or requested_root in source_root.parents
        or source_root in requested_root.parents
        or requested_root == ELMO_OUTPUTS_ROOT
        or requested_root in ELMO_OUTPUTS_ROOT.parents
        or ELMO_OUTPUTS_ROOT in requested_root.parents
        or requested_root in declared_staging.parents
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 baseline payload must be a separate external library, not source/output/receipt state"
        )
    try:
        identity = baseline_payload_snapshot.validate_snapshot(
            root=requested_root,
            manifest_path=declared_manifest,
            manifest_sha256=str(host.get("manifest_sha256") or ""),
            baseline_tree_sha256=str(host.get("baseline_tree_sha256") or ""),
            owner_contract_sha256=_require_sha256(
                owner_contract_sha256,
                label="r241 baseline snapshot owner contract sha256",
            ),
        )
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241ElmoRemoteWorkerError(
            f"r241 Elmo external baseline payload failed validation: {exc}"
        ) from exc
    if _sha256(declared_staging) != str(host.get("staging_receipt_sha256") or ""):
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo baseline payload staging receipt checksum drifted"
        )
    _staging_path, staging = _json_object(
        declared_staging, label="r241 Elmo baseline payload staging receipt"
    )
    binding = _require_mapping(
        staging.get("baseline_payload_snapshot"),
        label="r241 Elmo baseline payload staging binding",
    )
    expected = {
        "schema": baseline_payload_snapshot.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": str(identity["root"]),
        "manifest": str(identity["manifest"]),
        "manifest_sha256": identity["manifest_sha256"],
        "baseline_tree_sha256": identity["baseline_tree_sha256"],
        "file_inventory_sha256": identity["file_inventory_sha256"],
        "owner_contract_sha256": _require_sha256(
            owner_contract_sha256, label="r241 mounted baseline owner contract sha256"
        ),
    }
    canonical_roster_rows = baseline_payload_snapshot.normalized_roster(
        list(canonical_roster.get("baseline_roster") or [])
    )
    if (
        canonical_roster.get("baseline_manifest_sha256")
        != contract.get("canonical_baseline_manifest_sha256")
        or canonical_roster.get("baseline_roster_sha256")
        != contract.get("canonical_baseline_roster_sha256")
        or identity.get("baseline_manifest_sha256")
        != contract.get("canonical_baseline_manifest_sha256")
        or identity.get("baseline_roster_sha256")
        != contract.get("canonical_baseline_roster_sha256")
        or identity.get("baseline_roster") != canonical_roster_rows
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 mounted baseline payload is not the canonical public-baseline roster"
        )
    canonical_binding = _require_mapping(
        staging.get("canonical_roster_receipt"),
        label="r241 Elmo baseline staging canonical roster binding",
    )
    if (
        staging.get("schema") != baseline_payload_snapshot.BASELINE_PAYLOAD_STAGING_SCHEMA
        or staging.get("revision") != R241_REVISION
        or staging.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or staging.get("status") != "passed"
        or staging.get("passed") is not True
        or staging.get("receipt_outside_source_and_baseline_snapshot") is not True
        or any(binding.get(key) != value for key, value in expected.items())
        or canonical_binding.get("path") != str(canonical_roster_path)
        or canonical_binding.get("sha256")
        != contract.get("canonical_roster_receipt_sha256")
        or canonical_binding.get("baseline_manifest_sha256")
        != contract.get("canonical_baseline_manifest_sha256")
        or canonical_binding.get("baseline_roster_sha256")
        != contract.get("canonical_baseline_roster_sha256")
        or canonical_binding.get("public_contract_sha256s")
        != canonical_roster.get("public_contract_sha256s")
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo baseline payload staging receipt binding drifted"
        )
    return R241ElmoBaselinePayload(
        root=Path(str(identity["root"])),
        manifest=Path(str(identity["manifest"])),
        manifest_sha256=str(identity["manifest_sha256"]),
        baseline_tree_sha256=str(identity["baseline_tree_sha256"]),
        file_inventory_sha256=str(identity["file_inventory_sha256"]),
        staging_receipt=declared_staging,
        staging_receipt_sha256=str(host["staging_receipt_sha256"]),
        canonical_roster_receipt=canonical_roster_path,
        canonical_roster_receipt_sha256=str(contract["canonical_roster_receipt_sha256"]),
        baseline_manifest_sha256=str(contract["canonical_baseline_manifest_sha256"]),
        baseline_roster_sha256=str(contract["canonical_baseline_roster_sha256"]),
    )


def _assert_runtime_baseline_import_binding(payload: R241ElmoBaselinePayload) -> None:
    """Ensure already-imported runtime modules cannot retain a fallback root.

    ``paths`` is imported transitively while this launcher module is imported.
    It therefore must have observed the sealed external root *before* Python
    loaded any baseline-capable module; merely setting an environment variable
    after preflight would leave a code-only snapshot silently pointing at its
    absent checkout-relative ``baselines/`` directory.
    """

    if (
        runtime_paths.BASELINES_DIR.resolve() != payload.root
        or runtime_paths.BASELINES_MANIFEST.resolve() != payload.root / "manifest.json"
    ):
        raise R241ElmoRemoteWorkerError(
            "POKEBOT_BASELINES_DIR must be the receipt-bound payload root before "
            "the r241 worker imports runtime modules"
        )


def _validate_deferred_matchup_archetype_refresh(
    *,
    policy: Mapping[str, Any],
    registry: Mapping[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    """Prove that the deferred r247 work cannot alter this r241 cycle.

    The owner deferred the authenticated PTCGReplay refresh.  It is therefore
    deliberately *not* a preflight input.  What r241 must prove instead is
    that it continues to use the immutable 0..19 peak-r195 roster and that no
    future slot/migration pin has silently become active.
    """

    # Revision 248 removes PTCGReplay metadata from r241's launch gates.
    # In particular, a future/absent/renamed r247 projection must not block
    # this worker.  The only executable evidence is the immutable baseline
    # roster itself plus the resulting no-slot-change audit below.
    del policy, registry
    roster_path = _regular_file(
        source_root / "state/matchup_adapter_roster.json",
        label="r241 preserved Matchup Adapter roster",
    )
    if _sha256(roster_path) != BASELINE_ADAPTER_ROSTER_SHA256:
        raise R241ElmoRemoteWorkerError(
            "r241 source snapshot does not retain the immutable peak-r195 adapter roster"
        )
    _roster_file, roster = _json_object(
        roster_path, label="r241 preserved Matchup Adapter roster"
    )
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or roster.get("slot_schema") != "poke_bot.matchup_adapter_slot_registry/v1"
        or roster.get("checkpoint_format") != "poke-bot-matchup-adapter-bank-v6"
        or _exact_int(roster.get("slot_capacity"), label="r241 adapter slot capacity")
        != 64
        or _exact_int(
            roster.get("legacy_v5_prefix_length"), label="r241 adapter V5 prefix"
        )
        != 18
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 preserved Matchup Adapter roster is not a Router Format 6 registry"
        )
    slots = list(roster.get("slots") or ())
    seen_ids: set[str] = set()
    for slot, row in enumerate(slots):
        if not isinstance(row, Mapping) or _exact_int(
            row.get("slot"), label="r241 adapter slot"
        ) != slot:
            raise R241ElmoRemoteWorkerError("r241 adapter roster slot order drifted")
        status = str(row.get("status") or "")
        archetype_id = row.get("archetype_id")
        if status not in {"active", "dormant", "retired", "unused"}:
            raise R241ElmoRemoteWorkerError("r241 adapter roster has an unknown slot status")
        if status == "unused":
            if archetype_id is not None:
                raise R241ElmoRemoteWorkerError(
                    "r241 unused adapter slot owns an identity"
                )
        elif (
            not isinstance(archetype_id, str)
            or not archetype_id
            or archetype_id in seen_ids
        ):
            raise R241ElmoRemoteWorkerError(
                "r241 allocated adapter slot identity is invalid"
            )
        elif isinstance(archetype_id, str):
            seen_ids.add(archetype_id)
    if (
        len(slots) != 64
        or any(
            not isinstance(row, Mapping)
            or row.get("slot") != slot
            or row.get("status") == "unused"
            or not str(row.get("archetype_id") or "")
            for slot, row in enumerate(slots[:IMMUTABLE_ADAPTER_SLOT_PREFIX])
        )
        or any(
            not isinstance(row, Mapping)
            or row.get("slot") != slot
            or row.get("status") != "unused"
            or row.get("archetype_id") is not None
            for slot, row in enumerate(
                slots[IMMUTABLE_ADAPTER_SLOT_PREFIX:],
                start=IMMUTABLE_ADAPTER_SLOT_PREFIX,
            )
        )
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 Matchup Adapter roster must retain only the current 0..19 routes"
        )

    return {
        "status": "deferred_not_required",
        "required_for_r241_activation": False,
        "slot_migration_status": "no_slot_change",
        "baseline_slot_registry": str(roster_path),
        "baseline_slot_registry_sha256": BASELINE_ADAPTER_ROSTER_SHA256,
        "immutable_slot_prefix": IMMUTABLE_ADAPTER_SLOT_PREFIX,
        "active_slot_count": IMMUTABLE_ADAPTER_SLOT_PREFIX,
        "new_slots": [],
    }


def validate_current_r241_sources(
    repo_root: Path | str = REPO_ROOT,
    *,
    source_snapshot_manifest: Path | str | None = None,
    source_staging_receipt: Path | str | None = None,
    source_staging_receipt_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Bind preflight to a verified immutable r241 source snapshot.

    ``repo_root`` is the snapshot execution root, not the mutable Elmo checkout.
    The source publisher owns its content-addressed tree algorithm; this
    consumer re-derives its manifest identity through the shared checkpoint
    receipt helper and then verifies the exact files used by :8767.
    """

    root = _regular_directory(repo_root, label="r241 source snapshot root")
    manifest = _regular_file(
        source_snapshot_manifest
        if source_snapshot_manifest is not None
        else root / R241_SOURCE_SNAPSHOT_FILENAME,
        label="r241 source snapshot manifest",
    )
    try:
        snapshot = authenticated_source_snapshot_provenance(
            source_root=root,
            manifest_path=manifest,
            outputs_root=ELMO_OUTPUTS_ROOT,
        )
    except R241CheckpointReceiptError as exc:
        raise R241ElmoRemoteWorkerError(
            f"r241 source snapshot is not authenticated/immutable: {exc}"
        ) from exc
    if (
        snapshot.get("schema") != R241_SOURCE_SNAPSHOT_SCHEMA
        or snapshot.get("root") != str(root)
        or snapshot.get("source_execution_root") != str(root)
        or snapshot.get("manifest") != str(manifest)
        or snapshot.get("outputs_root") != str(ELMO_OUTPUTS_ROOT)
    ):
        raise R241ElmoRemoteWorkerError("r241 source snapshot binding drifted")
    _, snapshot_manifest = _json_object(manifest, label="r241 source snapshot manifest")
    inventory = snapshot_manifest.get("files")
    if not isinstance(inventory, list):
        raise R241ElmoRemoteWorkerError("r241 source snapshot has no file inventory")
    inventory_by_path: dict[str, dict[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, dict):
            raise R241ElmoRemoteWorkerError("r241 source snapshot inventory is malformed")
        relative = str(row.get("path") or "")
        if relative in inventory_by_path:
            raise R241ElmoRemoteWorkerError("r241 source snapshot repeats a file path")
        inventory_by_path[relative] = dict(row)
    missing_runtime_source = [
        relative
        for relative in REQUIRED_SOURCE_SNAPSHOT_FILES
        if relative not in inventory_by_path
    ]
    if missing_runtime_source:
        raise R241ElmoRemoteWorkerError(
            "r241 source snapshot omits :8767 runtime files: "
            + ", ".join(missing_runtime_source)
        )
    policy_path, policy = _json_object(
        root / "state/alakazam-new-list-direct-policy-r241.json",
        label="r241 owner contract",
    )
    registry_path, registry = _json_object(
        root / "state/alakazam-new-list-direct-r241-runtime-registry.json",
        label="r241 runtime registry",
    )
    projection_path = _regular_file(
        root / "config/rl_protocol.yaml", label="r241 protocol projection"
    )
    policy_digest = _sha256(policy_path)
    latest_owner_revision = _exact_int(
        policy.get("latest_owner_clarification_revision"),
        label="r241 owner clarification revision",
    )
    if (
        policy.get("schema") != "poke_bot.alakazam_new_list_direct_policy_r241/v1"
        or _exact_int(policy.get("owner_decision_revision"), label="r241 owner revision")
        != R241_REVISION
        or latest_owner_revision != R241_LATEST_OWNER_CLARIFICATION_REVISION
        or policy.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
    ):
        raise R241ElmoRemoteWorkerError("r241 owner contract identity drifted")
    if snapshot_manifest.get("owner_contract_sha256") != policy_digest:
        raise R241ElmoRemoteWorkerError(
            "r241 source snapshot does not checksum-bind its owner contract"
        )
    preservation = _require_mapping(
        policy.get("peak_r195_behavior_preservation"), label="r195 preservation"
    )
    if (
        _exact_int(
            preservation.get("learned_head_count_present"), label="r195 head count"
        )
        != 19
        or preservation.get("every_architecture_present_non_combo_head_trainable")
        is not True
        or preservation.get("every_architecture_present_non_combo_fusion_route_enabled")
        is not True
        or preservation.get("combo_state_head_remains_present") is not True
        or preservation.get("combo_state_loss_weight") != 0.0
        or preservation.get("combo_state_fusion_route_enabled") is not False
        or preservation.get("matchup_adapter_bank_preserved") is not True
        or preservation.get("matchup_adapter_training_enabled") is not True
        or preservation.get("matchup_adapter_runtime_enabled") is not True
        or preservation.get("learner_public_matchup_tree_sha256")
        != R195_LEARNER_MATCHUP_TREE_SHA256
        or preservation.get("marnie_public_matchup_tree_sha256")
        != R241_H10_MATCHUP_TREE_SHA256
    ):
        raise R241ElmoRemoteWorkerError("r241 peak-r195 preservation contract drifted")
    cycle = _require_mapping(policy.get("training_cycle"), label="r241 training cycle")
    if (
        _exact_int(cycle.get("games_per_update"), label="r241 games per update")
        != 8196
        or _exact_int(cycle.get("self_play_games_exact"), label="r241 self-play games")
        != 1024
        or _exact_int(cycle.get("public_mix_games_exact"), label="r241 public mix games")
        != 7172
        or _exact_int(
            cycle.get("marnie_h10_games_minimum"), label="r241 H10 minimum games"
        )
        != 1024
        or cycle.get("established_diverse_public_mix_preserved") is not True
        or cycle.get("marnie_h10_is_minimum_not_exclusive_public_opponent")
        is not True
        or cycle.get("established_research_control_phase_preserved") is not True
    ):
        raise R241ElmoRemoteWorkerError("r241 diverse public/research contract drifted")
    search = _require_mapping(
        policy.get("search_and_planning_exclusion"), label="r241 search exclusion"
    )
    scoped_direct_roles = {
        "learner": "direct_policy_only",
        "pinned_h10_marnie_opponent": "direct_policy_only",
        "target_generation": "direct_policy_only",
        "terminal_package_and_submission": "direct_policy_only",
        "frozen_non_h10_diverse_public_opponent_packages_and_selectors": (
            "preserve_unchanged_per_r245"
        ),
    }
    if (
        _require_mapping(
            search.get("scope"), label="r251 direct-policy scope"
        )
        != scoped_direct_roles
        or any(
            search.get(name) != "forbidden_for_scoped_direct_roles"
            for name in (
                "mcts",
                "recursive_turn_planner",
                "search_target_generation",
            )
        )
        or any(
            search.get(name) != "direct_policy_only"
            for name in (
                "training_collector_action_selector",
                "marnie_action_selector",
                "submission_action_selector",
            )
        )
        or search.get("public_opponent_selector_change") != "forbidden"
        or search.get("public_search_firewall") != "not_introduced"
        or search.get("search_config_or_belief_deck_assets_required") is not False
        or search.get("concurrent_mcts_work_may_be_interrupted_or_reconfigured")
        is not False
    ):
        raise R241ElmoRemoteWorkerError(
            "r251 direct-policy scope must preserve frozen non-H10 public selectors"
        )

    registry_owner_revision = _exact_int(
        registry.get("owner_clarification_revision"),
        label="r241 registry clarification revision",
    )
    if (
        registry.get("schema")
        != "poke_bot.alakazam_new_list_direct_policy_r241_runtime_registry/v1"
        or _exact_int(registry.get("revision"), label="r241 registry revision")
        != R241_REVISION
        or registry_owner_revision != latest_owner_revision
    ):
        raise R241ElmoRemoteWorkerError("r241 runtime registry identity drifted")
    owner_reference = _require_mapping(
        registry.get("owner_contract"), label="r241 registry owner reference"
    )
    if (
        owner_reference.get("path")
        != "state/alakazam-new-list-direct-policy-r241.json"
        or owner_reference.get("sha256") != policy_digest
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 runtime registry does not checksum-bind the current owner contract"
        )
    _require_pending_snapshot_projection(
        registry, owner_contract_sha256=policy_digest
    )
    base_baseline_payload_contract = _baseline_payload_contract_for_elmo(registry)
    if base_baseline_payload_contract.get("status") != "pending_external_baseline_payload_snapshot":
        raise R241ElmoRemoteWorkerError(
            "r241 source snapshot must retain a pending external baseline payload registry"
        )
    if source_staging_receipt is None or source_staging_receipt_sha256 is None:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo source staging receipt and SHA are required for offline preflight"
        )
    source_staging = _source_snapshot_from_staging_receipt(
        source_snapshot=snapshot,
        owner_contract_sha256=policy_digest,
        staging_receipt=source_staging_receipt,
        staging_receipt_sha256=source_staging_receipt_sha256,
    )
    official = _require_mapping(registry.get("official_libcg"), label="r241 official libcg")
    if (
        official.get("linux_member") != "cg/libcg.so"
        or official.get("linux_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
        or _exact_int(official.get("linux_size_bytes"), label="r241 r236 size")
        != 1_342_400
        or official.get("receipt_filename") != R241_OFFICIAL_LIBCG_RECEIPT_FILENAME
    ):
        raise R241ElmoRemoteWorkerError("r241 r236 D162 registry binding drifted")
    official_hosts = _require_mapping(
        official.get("hosts"), label="r241 official libcg hosts"
    )
    elmo_official = _require_mapping(
        official_hosts.get("elmo"), label="r241 Elmo official libcg host"
    )
    expected_elmo_cg_root = str(elmo_official.get("runtime_root") or "")
    if expected_elmo_cg_root != str(ELMO_R241_RUN_ROOT / "runtime/cg-r236"):
        raise R241ElmoRemoteWorkerError("r241 Elmo official r236 path drifted")
    registry_preservation = _require_mapping(
        registry.get("peak_r195_preservation"), label="r241 registry preservation"
    )
    if (
        _exact_int(
            registry_preservation.get("architecture_present_head_count"),
            label="registry head count",
        )
        != 19
        or registry_preservation.get("every_non_combo_head_trainable") is not True
        or registry_preservation.get("every_non_combo_fusion_route_enabled")
        is not True
        or registry_preservation.get("combo_state_head_present") is not True
        or registry_preservation.get("combo_state_loss_weight") != 0.0
        or registry_preservation.get("combo_state_fusion_route_enabled") is not False
        or registry_preservation.get("matchup_adapter_bank_preserved") is not True
        or registry_preservation.get("matchup_adapter_training_enabled") is not True
        or registry_preservation.get("matchup_adapter_runtime_enabled") is not True
        or registry_preservation.get("matchup_adapter_checkpoint_runtime_enabled")
        is not False
        or registry_preservation.get("matchup_adapter_checkpoint_training_enabled")
        is not False
        or registry_preservation.get("matchup_adapter_checkpoint_main_optimizer_included")
        is not False
        or registry_preservation.get("matchup_adapter_isolated_bank_only_optimizer")
        is not True
        or registry_preservation.get("matchup_adapter_isolated_fit_continuation_required")
        is not True
        or registry_preservation.get("matchup_adapter_external_collection_runtime_enabled")
        is not True
        or registry_preservation.get("matchup_adapter_external_terminal_runtime_enabled")
        is not True
        or registry_preservation.get("learner_matchup_tree_sha256")
        != R195_LEARNER_MATCHUP_TREE_SHA256
        or registry_preservation.get("established_diverse_public_mix_preserved")
        is not True
        or registry_preservation.get("research_control_phase_preserved") is not True
    ):
        raise R241ElmoRemoteWorkerError("r241 registry preservation binding drifted")
    if str(registry_preservation.get("receipt_elmo") or "") != str(
        ELMO_R241_RUN_ROOT
        / "runtime"
        / R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo peak-r195 preservation receipt path drifted"
        )
    direct = _require_mapping(registry.get("direct_policy"), label="r241 direct policy")
    if (
        direct.get("action_selector") != "direct_policy_only"
        or direct.get("mcts") != "forbidden_for_scoped_direct_roles"
        or direct.get("recursive_turn_planner")
        != "forbidden_for_scoped_direct_roles"
        or direct.get("search_target_generation")
        != "forbidden_for_scoped_direct_roles"
        or direct.get("public_opponent_selector_change") != "forbidden"
        or direct.get("public_search_firewall") != "not_introduced"
        or direct.get("matchup_adapters_allowed_and_required") is not True
        or direct.get("adapter_receipt_schema")
        != R241_MARNIE_ADAPTER_RECEIPT_SCHEMA
    ):
        raise R241ElmoRemoteWorkerError("r241 direct H10 adapter contract drifted")
    expected_elmo_adapter_receipt = str(direct.get("adapter_receipt_elmo") or "")
    if expected_elmo_adapter_receipt != str(
        ELMO_R241_RUN_ROOT / "runtime" / R241_H10_ADAPTER_RECEIPT_BASENAME
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo H10 adapter receipt path drifted")

    deferred_archetype_refresh = _validate_deferred_matchup_archetype_refresh(
        policy=policy,
        registry=registry,
        source_root=root,
    )

    projection_text = projection_path.read_text(encoding="utf-8")
    required_projection_lines = (
        "alakazam_new_list_direct_policy_r241:",
        f"values_owned_by_sha256: {policy_digest}",
        f"latest_owner_clarification_revision: {R241_LATEST_OWNER_CLARIFICATION_REVISION}",
        "public_mix_games_exact: 7172",
        "marnie_h10_is_minimum_not_exclusive_public_opponent: true",
        "every_non_combo_head_and_fusion_route_live_and_trainable: true",
        "combo_state_loss_and_fusion_route_enabled: false",
        "matchup_adapters_enabled: true",
        "adapter_slot_migration_required_status: no_slot_change",
        "mcts_rtp_guide2vec_and_search_targets_allowed: false",
        "direct_policy_scope:",
        "learner: direct_policy_only",
        "pinned_h10_marnie_opponent: direct_policy_only",
        "target_generation: direct_policy_only",
        "terminal_package_and_submission: direct_policy_only",
        "frozen_non_h10_diverse_public_opponent_packages_and_selectors: preserve_unchanged_per_r245",
        "public_opponent_selector_change: forbidden",
        "public_search_firewall: not_introduced",
        "logical_activation_overlay_cardinality: one",
        "overlay_host_publication_identity: byte_identical_with_one_shared_sha256",
        "external_activation_overlay_required: true",
    )
    missing_projection = [
        line for line in required_projection_lines if line not in projection_text
    ]
    if missing_projection:
        raise R241ElmoRemoteWorkerError(
            "r241 protocol projection is missing required controls: "
            + ", ".join(missing_projection)
        )

    return {
        "source_snapshot": dict(snapshot),
        "source_staging_receipt": source_staging,
        "owner_contract": {
            "path": "state/alakazam-new-list-direct-policy-r241.json",
            "sha256": policy_digest,
            "schema": str(policy["schema"]),
        },
        "runtime_registry": {
            "path": "state/alakazam-new-list-direct-r241-runtime-registry.json",
            "sha256": _sha256(registry_path),
            "schema": str(registry["schema"]),
        },
        "protocol_projection": {
            "path": "config/rl_protocol.yaml",
            "sha256": _sha256(projection_path),
            "r241_projection_present": True,
        },
        "runtime_module": {
            "path": "poke_bot/r241_direct_policy_runtime.py",
            "sha256": _sha256(root / "poke_bot/r241_direct_policy_runtime.py"),
        },
        "h10_adapter_module": {
            "path": "poke_bot/r241_marnie_direct_policy_adapter.py",
            "sha256": _sha256(root / "poke_bot/r241_marnie_direct_policy_adapter.py"),
        },
        "baseline_payload_contract": base_baseline_payload_contract,
        "matchup_archetype_refresh": deferred_archetype_refresh,
        "elmo_host_paths": {
            "source_execution_root": str(root),
            "source_snapshot_manifest": str(manifest),
            "outputs_root": str(ELMO_OUTPUTS_ROOT),
            "r241_run_root": str(ELMO_R241_RUN_ROOT),
            "official_cg_root": expected_elmo_cg_root,
            "adapter_receipt": expected_elmo_adapter_receipt,
            "identical_host_paths_required_inside_container": True,
        },
    }


def _adapter_spec_from_receipt(adapter_receipt: Path) -> SimpleNamespace:
    """Build only the minimal data descriptor accepted by the H10 validator."""

    _, payload = _json_object(adapter_receipt, label="r241 H10 adapter receipt")
    package = _require_mapping(payload.get("package"), label="H10 adapter package")
    root = _regular_directory(package.get("root_path") or "", label="H10 package root")
    _require_elmo_absolute_path(root, label="H10 package root")
    return SimpleNamespace(
        id=R241_H10_OPPONENT_ID,
        dir_name=R241_H10_DIR_NAME,
        path=root,
    )


def _validate_h10_adapter(
    adapter_receipt: Path,
    *,
    environment: Mapping[str, str],
    source_snapshot: Mapping[str, object],
) -> tuple[Path, dict[str, Any]]:
    """Use the existing data-only H10 checker; it never imports package main."""

    try:
        validate_r241_h10_adapter_source_binding(
            adapter_receipt,
            source_snapshot=source_snapshot,
        )
    except R241CheckpointReceiptError as exc:
        raise R241ElmoRemoteWorkerError(
            "r241 H10 adapter receipt does not bind the active source snapshot"
        ) from exc
    _, receipt = _json_object(adapter_receipt, label="r241 H10 adapter receipt")
    if (
        receipt.get("schema") != R241_MARNIE_ADAPTER_RECEIPT_SCHEMA
        or _exact_int(receipt.get("revision"), label="H10 adapter revision")
        != R241_REVISION
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("direct_policy_only") is not True
        or receipt.get("action_selector") != "direct_policy_only"
    ):
        raise R241ElmoRemoteWorkerError("r241 H10 adapter receipt identity drifted")
    runtime = _require_mapping(receipt.get("runtime"), label="H10 adapter runtime")
    required_runtime = {
        "package_main_imported": False,
        "package_search_invoked": False,
        "embedded_cg_loaded": False,
        "matchup_adapter_runtime": True,
        "matchup_adapter_tree_loaded": True,
        "mcts_calls": 0,
        "rtp_calls": 0,
        "search_calls": 0,
    }
    if any(runtime.get(key) != expected for key, expected in required_runtime.items()):
        raise R241ElmoRemoteWorkerError("r241 H10 adapter no-search receipt drifted")
    package = _require_mapping(receipt.get("package"), label="H10 adapter package")
    if (
        package.get("opponent_id") != R241_H10_OPPONENT_ID
        or package.get("content_sha256") != R241_H10_CONTENT_SHA256
        or _require_mapping(package.get("model"), label="H10 model").get("sha256")
        != R241_H10_MODEL_SHA256
        or _require_mapping(
            package.get("matchup_tree"), label="H10 matchup tree"
        ).get("sha256")
        != R241_H10_MATCHUP_TREE_SHA256
    ):
        raise R241ElmoRemoteWorkerError("r241 H10 data package identity drifted")
    sealed = _require_mapping(receipt.get("sealed_runtime"), label="H10 sealed runtime")
    if sealed.get("linux_x86_64_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256:
        raise R241ElmoRemoteWorkerError("H10 adapter is not bound to official r236 D162")

    spec = _adapter_spec_from_receipt(adapter_receipt)
    try:
        _model, _deck, h10_tree, receipt_path = validate_r241_marnie_direct_policy_adapter(
            spec, environment=environment
        )
    except R241DirectPolicyRuntimeError as exc:
        raise R241ElmoRemoteWorkerError(
            f"r241 H10 data-only adapter validation failed: {exc}"
        ) from exc
    if receipt_path != adapter_receipt.resolve() or _sha256(h10_tree) != R241_H10_MATCHUP_TREE_SHA256:
        raise R241ElmoRemoteWorkerError("r241 H10 bound matchup tree identity drifted")
    return h10_tree.resolve(), receipt


def _validate_learner_matchup_runtime(
    *,
    checkpoint: Path,
    learner_matchup_tree: Path,
) -> tuple[Path, dict[str, Any]]:
    """Require the live r195 route marker, not merely a tree-shaped JSON file."""

    marker = _regular_file(
        checkpoint.parent / "matchup-runtime-activation.json",
        label="r195 Matchup Adapter activation marker",
    )
    if _sha256(learner_matchup_tree) != R195_LEARNER_MATCHUP_TREE_SHA256:
        raise R241ElmoRemoteWorkerError("learner Matchup Adapter tree is not r195 E60")
    try:
        # This path validates the checkpoint model/config/route binding and
        # does not start a worker, invoke libcg, or run a game.
        from scripts.run_remote_worker import _activate_matchup_runtime_from_marker

        runtime = _activate_matchup_runtime_from_marker(
            checkpoint, apply_environment=False
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise R241ElmoRemoteWorkerError(
            f"r195 Matchup Adapter runtime proof failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(runtime, dict):
        raise R241ElmoRemoteWorkerError(
            "r241 requires an active r195 Matchup Adapter marker beside the checkpoint"
        )
    if (
        Path(str(runtime.get("marker") or "")).resolve() != marker
        or Path(str(runtime.get("tree") or "")).resolve() != learner_matchup_tree
        or runtime.get("tree_digest") != R195_LEARNER_MATCHUP_TREE_SHA256
        or runtime.get("continuous_reevaluation") is not True
        or runtime.get("one_route_per_decision") is not True
        or not str(runtime.get("adapter_format") or "")
        or not list(runtime.get("route_target_ids") or ())
    ):
        raise R241ElmoRemoteWorkerError("r195 Matchup Adapter runtime binding drifted")
    physical_slots = list(runtime.get("route_physical_slots") or ())
    if (
        not physical_slots
        or len(physical_slots) != len(runtime.get("route_target_ids") or ())
        or len(set(physical_slots)) != len(physical_slots)
        or any(
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot < 0
            or slot >= IMMUTABLE_ADAPTER_SLOT_PREFIX
            for slot in physical_slots
        )
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 requires the current no-slot-change peak-r195 Matchup Adapter routes"
        )
    return marker, dict(runtime)


def preflight_r241_elmo_remote_collection(
    *,
    endpoint: str,
    repo_root: Path | str,
    source_snapshot_manifest: Path | str,
    checkpoint: Path | str,
    cg_lib_path: Path | str,
    adapter_receipt: Path | str,
    learner_matchup_tree: Path | str,
    baselines_root: Path | str,
    checkpoint_transport_host_root: Path | str,
    checkpoint_transport_staging_receipt: Path | str,
    checkpoint_transport_staging_receipt_sha256: str,
    source_staging_receipt: Path | str,
    source_staging_receipt_sha256: str,
    baseline_staging_receipt: Path | str,
    baseline_staging_receipt_sha256: str,
    canonical_roster_receipt: Path | str,
    canonical_roster_receipt_sha256: str,
    environment: Mapping[str, str] | None = None,
) -> R241ElmoPreflight:
    """Validate every local input needed before an r241 :8767 worker may run."""

    assert_r241_elmo_endpoint(endpoint)
    root = _regular_directory(repo_root, label="r241 source snapshot root")
    _require_elmo_absolute_path(root, label="r241 source snapshot root")
    source_manifest = _regular_file(
        source_snapshot_manifest, label="r241 source snapshot manifest"
    )
    _require_elmo_absolute_path(source_manifest, label="r241 source snapshot manifest")
    source_identities = validate_current_r241_sources(
        root,
        source_snapshot_manifest=source_manifest,
        source_staging_receipt=source_staging_receipt,
        source_staging_receipt_sha256=source_staging_receipt_sha256,
    )
    cg_root = _regular_directory(cg_lib_path, label="CG_LIB_PATH")
    adapter = _regular_file(adapter_receipt, label="H10 adapter receipt")
    for value, label in (
        (cg_root, "CG_LIB_PATH"),
        (adapter, "H10 adapter receipt"),
    ):
        _require_elmo_absolute_path(value, label=label)
    expected_paths = dict(source_identities["elmo_host_paths"])
    if cg_root != Path(expected_paths["official_cg_root"]):
        raise R241ElmoRemoteWorkerError(
            "CG_LIB_PATH must be the exact registry-bound Elmo r236 root"
        )
    if adapter != Path(expected_paths["adapter_receipt"]):
        raise R241ElmoRemoteWorkerError(
            "H10 adapter receipt must be the exact registry-bound Elmo receipt"
        )
    owner_contract_sha256 = _require_sha256(
        _require_mapping(
            source_identities.get("owner_contract"), label="r241 preflight owner contract"
        ).get("sha256"),
        label="r241 preflight owner contract sha256",
    )
    staged_baseline_contract = _baseline_payload_contract_from_staging_receipt(
        owner_contract_sha256=owner_contract_sha256,
        staging_receipt=baseline_staging_receipt,
        staging_receipt_sha256=baseline_staging_receipt_sha256,
        canonical_roster_receipt=canonical_roster_receipt,
        canonical_roster_receipt_sha256=canonical_roster_receipt_sha256,
    )
    baseline_payload = _validate_elmo_baseline_payload(
        contract=staged_baseline_contract,
        baselines_root=baselines_root,
        source_root=root,
        owner_contract_sha256=owner_contract_sha256,
    )
    checkpoint_transport = _validate_elmo_checkpoint_transport(
        host_root=checkpoint_transport_host_root,
        staging_receipt=checkpoint_transport_staging_receipt,
        staging_receipt_sha256=checkpoint_transport_staging_receipt_sha256,
        checkpoint=checkpoint,
        source_root=root,
        baselines_root=baseline_payload.root,
        owner_contract_sha256=owner_contract_sha256,
    )
    checkpoint_path = checkpoint_transport.initial_checkpoint
    learner_tree = _checkpoint_transport_member(
        learner_matchup_tree,
        root=checkpoint_transport.container_root,
        label="r195 learner Matchup Adapter tree",
    )
    _assert_runtime_baseline_import_binding(baseline_payload)

    # Validate source environment before carrying any inherited values into a
    # child process.  The tree's real adapter format is read next from the
    # checkpoint-backed marker, then the final mapping is constructed once.
    inherited = dict(os.environ if environment is None else environment)
    _assert_clean_inherited_environment(
        inherited,
        cg_lib_path=cg_root,
        adapter_receipt=adapter,
        learner_matchup_tree=learner_tree,
        baseline_payload=baseline_payload,
        checkpoint_transport=checkpoint_transport,
        source_execution_root=root,
    )
    marker, matchup_runtime = _validate_learner_matchup_runtime(
        checkpoint=checkpoint_path,
        learner_matchup_tree=learner_tree,
    )
    sealed_environment = build_r241_elmo_collection_environment(
        cg_lib_path=cg_root,
        adapter_receipt=adapter,
        learner_matchup_tree=learner_tree,
        baseline_payload=baseline_payload,
        checkpoint_transport=checkpoint_transport,
        adapter_format=str(matchup_runtime["adapter_format"]),
        source_execution_root=root,
        environment=inherited,
    )
    try:
        actual_cg_root = validate_sealed_official_libcg(
            cg_root, environment=sealed_environment
        )
    except R241DirectPolicyRuntimeError as exc:
        raise R241ElmoRemoteWorkerError(
            f"sealed official r236 D162 validation failed: {exc}"
        ) from exc
    if actual_cg_root != cg_root:
        raise R241ElmoRemoteWorkerError("sealed r236 root changed during preflight")
    h10_tree, _adapter_payload = _validate_h10_adapter(
        adapter,
        environment=sealed_environment,
        source_snapshot=_require_mapping(
            source_identities.get("source_snapshot"),
            label="r241 active source snapshot",
        ),
    )
    h10_spec = _adapter_spec_from_receipt(adapter)
    expected_h10_root = baseline_payload.root / "specialists" / R241_H10_DIR_NAME
    if (
        h10_spec.path != expected_h10_root
        or h10_tree != expected_h10_root / "matchup_tree.json"
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 H10 adapter package is not bound to the generic baseline payload"
        )
    source_identities = dict(source_identities)
    source_identities["baseline_payload_contract"] = staged_baseline_contract
    source_identities["baseline_payload"] = _baseline_payload_identity(baseline_payload)
    source_identities["checkpoint_transport"] = _checkpoint_transport_identity(
        checkpoint_transport
    )
    return R241ElmoPreflight(
        repo_root=root,
        source_snapshot_manifest=source_manifest,
        checkpoint=checkpoint_path,
        cg_lib_path=cg_root,
        adapter_receipt=adapter,
        learner_matchup_tree=learner_tree,
        matchup_runtime_marker=marker,
        h10_matchup_tree=h10_tree,
        baseline_payload=baseline_payload,
        checkpoint_transport=checkpoint_transport,
        environment=sealed_environment,
        sources=source_identities,
        matchup_runtime=matchup_runtime,
    )


def receipt_paths(receipt_dir: Path | str) -> dict[str, Path]:
    root = Path(receipt_dir).expanduser()
    if root.is_symlink():
        raise R241ElmoRemoteWorkerError(f"receipt directory must not be a symlink: {root}")
    return {
        "host": root / HOST_RECEIPT_FILENAME,
        "runtime": root / RUNTIME_RECEIPT_FILENAME,
        "gameplay": root / GAMEPLAY_RECEIPT_FILENAME,
        "manifest": root / MANIFEST_FILENAME,
    }


def build_r241_elmo_preflight_receipts(
    preflight: R241ElmoPreflight,
    *,
    receipt_dir: Path | str,
) -> dict[str, dict[str, Any]]:
    """Build deterministic host/runtime/gameplay/manifest receipts in memory."""

    paths = receipt_paths(receipt_dir)
    source_snapshot = dict(preflight.sources.get("source_snapshot") or {})
    if not source_snapshot:
        raise R241ElmoRemoteWorkerError("preflight omitted authenticated source snapshot")
    source_snapshot["host"] = "elmo"
    source_identities = dict(preflight.sources)
    source_identities["source_snapshot"] = source_snapshot
    baseline_payload_identity = _baseline_payload_identity(preflight.baseline_payload)
    existing_baseline_identity = preflight.sources.get("baseline_payload")
    if (
        existing_baseline_identity is not None
        and _require_mapping(
            existing_baseline_identity, label="preflight baseline payload identity"
        )
        != baseline_payload_identity
    ):
        raise R241ElmoRemoteWorkerError(
            "preflight baseline payload identity does not match the sealed mount"
        )
    source_identities["baseline_payload"] = baseline_payload_identity
    checkpoint_transport_identity = _checkpoint_transport_identity(
        preflight.checkpoint_transport
    )
    existing_checkpoint_transport = preflight.sources.get("checkpoint_transport")
    if (
        existing_checkpoint_transport is not None
        and _require_mapping(
            existing_checkpoint_transport, label="preflight checkpoint transport identity"
        )
        != checkpoint_transport_identity
    ):
        raise R241ElmoRemoteWorkerError(
            "preflight checkpoint transport identity does not match the sealed mount"
        )
    source_identities["checkpoint_transport"] = checkpoint_transport_identity
    deferred_archetype_refresh = _require_mapping(
        preflight.sources.get("matchup_archetype_refresh"),
        label="preflight deferred matchup archetype refresh",
    )
    if (
        deferred_archetype_refresh.get("required_for_r241_activation") is not False
        or deferred_archetype_refresh.get("slot_migration_status") != "no_slot_change"
        or deferred_archetype_refresh.get("new_slots") != []
        or deferred_archetype_refresh.get("baseline_slot_registry_sha256")
        != BASELINE_ADAPTER_ROSTER_SHA256
        or deferred_archetype_refresh.get("immutable_slot_prefix")
        != IMMUTABLE_ADAPTER_SLOT_PREFIX
    ):
        raise R241ElmoRemoteWorkerError(
            "preflight does not retain the current no-slot-change peak-r195 adapter roster"
        )
    endpoint = {
        "id": ELMO_R241_ENDPOINT_ID,
        "host_role": "elmo",
        "address": ELMO_R241_ENDPOINT_HOST,
        "port": ELMO_R241_ENDPOINT_PORT,
        "literal": ELMO_R241_ENDPOINT,
        "explicit_eligible_endpoint_required": True,
        "legacy_or_default_endpoint_fallback_allowed": False,
        "legacy_or_default_endpoints_rejected": list(LEGACY_OR_DEFAULT_ENDPOINTS),
        "capability": ELMO_R241_COLLECTION_CAPABILITY,
    }
    host = {
        "schema": R241_ELMO_REMOTE_HOST_RECEIPT_SCHEMA,
        "revision": R241_REVISION,
        "status": "passed",
        "passed": True,
        "deployment_action": "not_started",
        "endpoint": endpoint,
        "host": {
            "declared_role": "elmo",
            "observed_hostname": socket.gethostname(),
            "collection_worker_port": ELMO_R241_ENDPOINT_PORT,
            "legacy_8765_reused": False,
            "bert_8766_reused": False,
        },
        "source_snapshot": source_snapshot,
        "baseline_payload": baseline_payload_identity,
        "checkpoint_transport": checkpoint_transport_identity,
        "source_identities": source_identities,
    }
    runtime = {
        "schema": R241_ELMO_REMOTE_RUNTIME_RECEIPT_SCHEMA,
        "revision": R241_REVISION,
        "status": "passed",
        "passed": True,
        "deployment_action": "not_started",
        "endpoint": endpoint,
        "sealed_official_libcg": {
            "cg_lib_path": str(preflight.cg_lib_path),
            "receipt": str(
                preflight.cg_lib_path / R241_OFFICIAL_LIBCG_RECEIPT_FILENAME
            ),
            "receipt_sha256": _sha256(
                preflight.cg_lib_path / R241_OFFICIAL_LIBCG_RECEIPT_FILENAME
            ),
            "linux_member": "cg/libcg.so",
            "linux_x86_64_sha256": R241_OFFICIAL_LINUX_LIBCG_SHA256,
            "linux_x86_64_size_bytes": 1_342_400,
        },
        "path_binding": {
            "identical_elmo_host_paths_required_inside_container": True,
            "container_path_remapping_allowed": False,
            "repo_root": str(preflight.repo_root),
            "source_snapshot_manifest": str(preflight.source_snapshot_manifest),
            "checkpoint": str(preflight.checkpoint),
            "cg_lib_path": str(preflight.cg_lib_path),
            "adapter_receipt": str(preflight.adapter_receipt),
            "learner_matchup_tree": str(preflight.learner_matchup_tree),
            "h10_matchup_tree": str(preflight.h10_matchup_tree),
            "baseline_payload_root": str(preflight.baseline_payload.root),
            "baseline_payload_manifest": str(preflight.baseline_payload.manifest),
            "baseline_payload_staging_receipt": str(
                preflight.baseline_payload.staging_receipt
            ),
            "checkpoint_transport_host_root": str(
                preflight.checkpoint_transport.host_root
            ),
            "checkpoint_transport_container_root": str(
                preflight.checkpoint_transport.container_root
            ),
            "checkpoint_transport_staging_receipt": str(
                preflight.checkpoint_transport.staging_receipt
            ),
        },
        "source_snapshot": source_snapshot,
        "baseline_payload": baseline_payload_identity,
        "checkpoint_transport": checkpoint_transport_identity,
        "direct_policy": {
            "action_selector": "direct_policy_only",
            "direct_policy_only": True,
            # This local preflight deliberately does not load a public package
            # or start a game.  It cannot make call-count claims for frozen
            # non-H10 public opponents, whose selectors are preserved as-is.
            "observation_scope": "preflight_no_jobs_started",
            "runtime_call_counters_available": False,
            "runtime_call_counters": "not_measured_preflight_no_jobs_started",
            "learner_and_h10_direct_policy_only": True,
            "frozen_non_h10_public_opponent_selectors": "preserved_external_public_opponents",
            "forbidden_private_cg_overrides": list(FORBIDDEN_LIBCG_OVERRIDE_KEYS),
            "forbidden_planning_prefixes": list(FORBIDDEN_POLICY_ENVIRONMENT_PREFIXES),
        },
        "matchup_adapter": {
            "runtime_enabled": True,
            "matchup_adapter_bank_preserved": True,
            "checkpoint_runtime_enabled": False,
            "checkpoint_training_enabled": False,
            "checkpoint_main_optimizer_included": False,
            "isolated_bank_only_optimizer": True,
            "isolated_fit_continuation_required": True,
            "external_collection_runtime_enabled": True,
            "matchup_adapter_external_terminal_runtime_enabled": True,
            "current_peak_r195_roster_retained": True,
            "slot_migration_status": "no_slot_change",
            "immutable_slot_prefix": IMMUTABLE_ADAPTER_SLOT_PREFIX,
            "new_slots": [],
            "deferred_archetype_refresh": deferred_archetype_refresh,
            "learner_tree": str(preflight.learner_matchup_tree),
            "learner_tree_sha256": R195_LEARNER_MATCHUP_TREE_SHA256,
            "runtime_marker": str(preflight.matchup_runtime_marker),
            "runtime_marker_sha256": _sha256(preflight.matchup_runtime_marker),
            "adapter_format": str(preflight.matchup_runtime["adapter_format"]),
            "route_target_ids": list(preflight.matchup_runtime["route_target_ids"]),
            "route_physical_slots": list(
                preflight.matchup_runtime.get("route_physical_slots") or ()
            ),
            "h10_adapter_receipt": str(preflight.adapter_receipt),
            "h10_adapter_receipt_sha256": _sha256(preflight.adapter_receipt),
            "h10_opponent_id": R241_H10_OPPONENT_ID,
            "h10_matchup_tree": str(preflight.h10_matchup_tree),
            "h10_matchup_tree_sha256": R241_H10_MATCHUP_TREE_SHA256,
            "h10_data_only": True,
        },
        "environment": {
            "PYTHONPATH": str(preflight.repo_root),
            "CG_LIB_PATH": str(preflight.cg_lib_path),
            R241_DIRECT_POLICY_ONLY_ENV: "1",
            R241_DIRECT_POLICY_RECEIPT_ENV: str(preflight.adapter_receipt),
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
            "POKEBOT_SEARCH_MODE": "policy",
            "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
            "POKEBOT_PUBLIC_MATCHUP_TREE_PATH": str(preflight.learner_matchup_tree),
            "POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE": "runtime",
            "POKEBOT_MATCHUP_ADAPTER_FORMAT": str(
                preflight.matchup_runtime["adapter_format"]
            ),
            "PURE_RL_PUBLIC_MIX_LOCAL_ONLY": "0",
            "POKEBOT_REMOTE_ALLOWED_JOB_KINDS": ",".join(COLLECTION_JOB_KINDS),
            "POKEBOT_REMOTE_WORKER_CAPABILITY_TAGS": ELMO_R241_COLLECTION_CAPABILITY,
            "POKEBOT_REMOTE_WORKER_ARM_FILE": str(ELMO_REMOTE_ARM_FILE),
            "POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE": ELMO_REMOTE_PLANNED_ROTATION_EXIT_CODE,
            "POKEBOT_REMOTE_MAX_SERVICE_JOBS": ELMO_REMOTE_MAX_SERVICE_JOBS,
            "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": ELMO_REMOTE_WORKER_SAFETY_VERSION,
            "POKEBOT_BASELINES_DIR": str(preflight.baseline_payload.root),
            "POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST": str(
                preflight.baseline_payload.manifest
            ),
            "POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST_SHA256": preflight.baseline_payload.manifest_sha256,
            "POKEBOT_R241_BASELINE_PAYLOAD_TREE_SHA256": preflight.baseline_payload.baseline_tree_sha256,
            ELMO_R241_CHECKPOINT_TRANSPORT_ENV: str(
                preflight.checkpoint_transport.container_root
            ),
        },
        "source_identities": source_identities,
    }
    gameplay = {
        "schema": R241_ELMO_REMOTE_GAMEPLAY_RECEIPT_SCHEMA,
        "revision": R241_REVISION,
        "status": "ready_no_games_started",
        "passed": True,
        "deployment_action": "not_started",
        "endpoint": endpoint,
        "observation_scope": "preflight_no_games_started",
        "runtime_call_counters_available": False,
        "runtime_call_counters": "not_measured_preflight_no_games_started",
        "allowed_job_kinds": list(COLLECTION_JOB_KINDS),
        "promotion_jobs_allowed": False,
        "collection_contract": {
            "games_per_update": 8196,
            "self_play_games_exact": 1024,
            "public_mix_games_exact": 7172,
            "marnie_h10_games_minimum": 1024,
            "marnie_h10_is_minimum_not_exclusive_public_opponent": True,
            "established_diverse_public_mix_preserved": True,
            "research_control_phase_preserved": True,
            "public_mix_local_only": False,
            "pure_rl_public_mix_local_only": "0",
            "baseline_payload": baseline_payload_identity,
            "checkpoint_transport": checkpoint_transport_identity,
        },
        "adapter_behavior": {
            "learner_matchup_adapter_runtime": True,
            "checkpoint_adapter_runtime": False,
            "checkpoint_adapter_training": False,
            "checkpoint_adapter_main_optimizer_included": False,
            "isolated_adapter_fit_continuation": True,
            "current_peak_r195_roster_retained": True,
            "slot_migration_status": "no_slot_change",
            "new_slots": [],
            "h10_direct_data_only_adapter": True,
            "non_h10_diverse_public_packages_use_normal_loader": True,
            "non_h10_public_opponent_selectors": "preserved_external_public_opponents",
            "combo_state_head_present": True,
            "combo_state_loss_weight": 0.0,
            "combo_state_fusion_route_enabled": False,
        },
    }
    manifest = {
        "schema": R241_ELMO_REMOTE_MANIFEST_SCHEMA,
        "revision": R241_REVISION,
        "status": "passed",
        "passed": True,
        "deployment_action": "not_started",
        "endpoint": endpoint,
        "receipts": {
            name: {
                "path": str(paths[name]),
                "sha256": _json_digest(payload),
                "schema": str(payload["schema"]),
            }
            for name, payload in (
                ("host", host),
                ("runtime", runtime),
                ("gameplay", gameplay),
            )
        },
        "required_runtime": {
            "r236_d162": R241_OFFICIAL_LINUX_LIBCG_SHA256,
            "r195_learner_matchup_tree": R195_LEARNER_MATCHUP_TREE_SHA256,
            "h10_matchup_tree": R241_H10_MATCHUP_TREE_SHA256,
            "capability": ELMO_R241_COLLECTION_CAPABILITY,
            "pure_rl_public_mix_local_only": "0",
            "explicit_endpoint_only": True,
            "current_peak_r195_roster_only": True,
            "external_baseline_payload_manifest": preflight.baseline_payload.manifest_sha256,
            "external_baseline_payload_tree": preflight.baseline_payload.baseline_tree_sha256,
            "checkpoint_transport_root": str(
                preflight.checkpoint_transport.container_root
            ),
            "checkpoint_transport_staging_receipt_sha256": preflight.checkpoint_transport.staging_receipt_sha256,
        },
    }
    return {"host": host, "runtime": runtime, "gameplay": gameplay, "manifest": manifest}


def write_r241_elmo_preflight_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    receipt_dir: Path | str,
) -> dict[str, Path]:
    """Write a receipt set once, accepting only byte-identical reruns.

    Every target is checked before any missing receipt is created.  A retry can
    therefore resume an interrupted preflight, while an input change requires
    a new receipt directory instead of overwriting evidence.
    """

    expected = {"host", "runtime", "gameplay", "manifest"}
    if set(receipts) != expected:
        raise R241ElmoRemoteWorkerError(
            f"preflight receipt set must contain exactly {sorted(expected)}"
        )
    paths = receipt_paths(receipt_dir)
    root = Path(receipt_dir).expanduser()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise R241ElmoRemoteWorkerError(f"receipt path is not a real directory: {root}")
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise R241ElmoRemoteWorkerError(f"receipt path changed while creating it: {root}")

    encoded = {name: _canonical_json_bytes(dict(payload)) for name, payload in receipts.items()}
    for name, path in paths.items():
        if path.exists() or path.is_symlink():
            existing = _regular_file(path, label=f"existing {name} receipt")
            if existing.read_bytes() != encoded[name]:
                raise R241ElmoRemoteWorkerError(
                    f"existing {name} receipt differs; use a new receipt directory: {existing}"
                )

    for name, path in paths.items():
        if path.exists():
            continue
        temporary_name: str | None = None
        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=str(root), text=False
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded[name])
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                existing = _regular_file(path, label=f"concurrent {name} receipt")
                if existing.read_bytes() != encoded[name]:
                    raise R241ElmoRemoteWorkerError(
                        f"concurrent {name} receipt differs: {existing}"
                    )
            else:
                os.unlink(temporary_name)
                temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
    return {name: _regular_file(path, label=f"written {name} receipt") for name, path in paths.items()}


def validate_r241_elmo_preflight_manifest(path: Path | str) -> dict[str, Any]:
    """Validate the manifest a trainer uses to admit the explicit :8767 endpoint."""

    manifest_path, manifest = _json_object(path, label="r241 Elmo preflight manifest")
    if (
        manifest.get("schema") != R241_ELMO_REMOTE_MANIFEST_SCHEMA
        or _exact_int(manifest.get("revision"), label="Elmo manifest revision")
        != R241_REVISION
        or manifest.get("status") != "passed"
        or manifest.get("passed") is not True
        or manifest.get("deployment_action") != "not_started"
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo preflight manifest is invalid")
    endpoint = _require_mapping(manifest.get("endpoint"), label="Elmo manifest endpoint")
    if (
        endpoint.get("id") != ELMO_R241_ENDPOINT_ID
        or endpoint.get("capability") != ELMO_R241_COLLECTION_CAPABILITY
        or endpoint.get("explicit_eligible_endpoint_required") is not True
        or endpoint.get("legacy_or_default_endpoint_fallback_allowed") is not False
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo manifest endpoint policy drifted")
    assert_r241_elmo_endpoint(str(endpoint.get("literal") or ""))
    receipts = _require_mapping(manifest.get("receipts"), label="Elmo manifest receipts")
    expected_schemas = {
        "host": R241_ELMO_REMOTE_HOST_RECEIPT_SCHEMA,
        "runtime": R241_ELMO_REMOTE_RUNTIME_RECEIPT_SCHEMA,
        "gameplay": R241_ELMO_REMOTE_GAMEPLAY_RECEIPT_SCHEMA,
    }
    if set(receipts) != set(expected_schemas):
        raise R241ElmoRemoteWorkerError("r241 Elmo manifest receipt roster drifted")
    loaded: dict[str, dict[str, Any]] = {}
    for name, schema in expected_schemas.items():
        reference = _require_mapping(receipts.get(name), label=f"{name} receipt reference")
        receipt_path, payload = _json_object(reference.get("path") or "", label=f"{name} receipt")
        if (
            payload.get("schema") != schema
            or payload.get("revision") != R241_REVISION
            or payload.get("passed") is not True
            or _sha256(receipt_path) != reference.get("sha256")
            or reference.get("schema") != schema
        ):
            raise R241ElmoRemoteWorkerError(f"r241 Elmo {name} receipt drifted")
        loaded[name] = payload
    required_runtime = _require_mapping(
        manifest.get("required_runtime"), label="Elmo manifest required runtime"
    )
    runtime = loaded["runtime"]
    official = _require_mapping(
        runtime.get("sealed_official_libcg"), label="Elmo r236 receipt"
    )
    adapter = _require_mapping(
        runtime.get("matchup_adapter"), label="Elmo adapter receipt"
    )
    path_binding = _require_mapping(
        runtime.get("path_binding"), label="Elmo path binding"
    )
    environment = _require_mapping(
        runtime.get("environment"), label="Elmo runtime environment"
    )
    baseline_payload = _require_mapping(
        runtime.get("baseline_payload"), label="Elmo external baseline payload"
    )
    host_baseline_payload = _require_mapping(
        loaded["host"].get("baseline_payload"),
        label="Elmo host external baseline payload",
    )
    checkpoint_transport = _require_mapping(
        runtime.get("checkpoint_transport"),
        label="Elmo checkpoint transport",
    )
    host_checkpoint_transport = _require_mapping(
        loaded["host"].get("checkpoint_transport"),
        label="Elmo host checkpoint transport",
    )
    expected_baseline_payload = {
        "schema": baseline_payload_snapshot.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": path_binding.get("baseline_payload_root"),
        "manifest": path_binding.get("baseline_payload_manifest"),
        "staging_receipt": path_binding.get("baseline_payload_staging_receipt"),
    }
    if (
        any(baseline_payload.get(key) != value for key, value in expected_baseline_payload.items())
        or host_baseline_payload != baseline_payload
        or not str(baseline_payload.get("manifest_sha256") or "").startswith("sha256:")
        or not str(baseline_payload.get("baseline_tree_sha256") or "").startswith("sha256:")
        or not str(baseline_payload.get("file_inventory_sha256") or "").startswith("sha256:")
        or not str(baseline_payload.get("staging_receipt_sha256") or "").startswith("sha256:")
        or required_runtime.get("external_baseline_payload_manifest")
        != baseline_payload.get("manifest_sha256")
        or required_runtime.get("external_baseline_payload_tree")
        != baseline_payload.get("baseline_tree_sha256")
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo external baseline payload receipt drifted")
    transport_initial = _require_mapping(
        checkpoint_transport.get("initial_checkpoint"),
        label="Elmo checkpoint transport initial checkpoint",
    )
    checkpoint_root = str(
        path_binding.get("checkpoint_transport_container_root") or ""
    ).strip()
    checkpoint_path = _checkpoint_transport_receipt_path(
        path_binding.get("checkpoint"),
        root=checkpoint_root,
        label="receipt checkpoint",
    )
    learner_tree_path = _checkpoint_transport_receipt_path(
        path_binding.get("learner_matchup_tree"),
        root=checkpoint_root,
        label="receipt learner Matchup Adapter tree",
    )
    expected_checkpoint_transport = {
        "schema": R241_CHECKPOINT_TRANSPORT_SCHEMA,
        "endpoint_id": ELMO_R241_ENDPOINT_ID,
        "host_role": "elmo",
        "verification_endpoint": ELMO_R241_ENDPOINT,
        "verification_port": ELMO_R241_ENDPOINT_PORT,
        "host_root": path_binding.get("checkpoint_transport_host_root"),
        "container_root": checkpoint_root,
        "environment_key": ELMO_R241_CHECKPOINT_TRANSPORT_ENV,
        "remote_path_prefix": f"{ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT}/",
        "content_addressing": {
            "algorithm": "sha256",
            "filename_scheme": ELMO_R241_CHECKPOINT_TRANSPORT_FILENAME_SCHEME,
        },
        "read_only_container_mount": True,
        "same_absolute_source_and_baseline_paths_preserved": True,
        "staging_receipt": path_binding.get("checkpoint_transport_staging_receipt"),
        "staging_receipt_sha256": required_runtime.get(
            "checkpoint_transport_staging_receipt_sha256"
        ),
        "initial_checkpoint": {
            "container_path": checkpoint_path,
            "sha256": transport_initial.get("sha256"),
        },
    }
    if (
        checkpoint_transport != expected_checkpoint_transport
        or host_checkpoint_transport != checkpoint_transport
        or not str(checkpoint_transport.get("host_root") or "").strip()
        or not Path(str(checkpoint_transport["host_root"])).is_absolute()
        or str(checkpoint_transport["host_root"]).startswith("/workspace/")
        or str(checkpoint_transport["host_root"]) == "/workspace"
        or not str(checkpoint_transport.get("staging_receipt") or "").strip()
        or not Path(str(checkpoint_transport["staging_receipt"])).is_absolute()
        or _require_sha256(
            checkpoint_transport.get("staging_receipt_sha256"),
            label="receipt checkpoint transport staging sha256",
        )
        != checkpoint_transport.get("staging_receipt_sha256")
        or _require_sha256(
            transport_initial.get("sha256"),
            label="receipt checkpoint transport initial checkpoint sha256",
        )
        != transport_initial.get("sha256")
        or transport_initial.get("container_path") != checkpoint_path
        or checkpoint_path == learner_tree_path
        or required_runtime.get("checkpoint_transport_root") != checkpoint_root
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo checkpoint transport receipt drifted")
    if (
        environment.get("CG_LIB_PATH") != str(
            _regular_directory(environment.get("CG_LIB_PATH") or "", label="receipt CG_LIB_PATH")
        )
        or environment.get("PYTHONPATH") != str(
            _require_elmo_absolute_path(
                str(path_binding.get("repo_root") or ""),
                label="receipt PYTHONPATH",
            )
        )
        or environment.get(R241_DIRECT_POLICY_ONLY_ENV) != "1"
        or environment.get(R241_DIRECT_POLICY_RECEIPT_ENV)
        != path_binding.get("adapter_receipt")
        or environment.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER") != "0"
        or environment.get("POKEBOT_SEARCH_MODE") != "policy"
        or environment.get("POKEBOT_SUBMISSION_SEARCH_DISABLE") != "1"
        or environment.get("POKEBOT_MATCHUP_ADAPTER_RUNTIME") != "1"
        or environment.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH")
        != path_binding.get("learner_matchup_tree")
        or environment.get("POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE") != "runtime"
        or environment.get("POKEBOT_MATCHUP_ADAPTER_FORMAT")
        != adapter.get("adapter_format")
        or environment.get("PURE_RL_PUBLIC_MIX_LOCAL_ONLY") != "0"
        or environment.get("POKEBOT_REMOTE_ALLOWED_JOB_KINDS")
        != ",".join(COLLECTION_JOB_KINDS)
        or environment.get("POKEBOT_REMOTE_WORKER_CAPABILITY_TAGS")
        != ELMO_R241_COLLECTION_CAPABILITY
        or environment.get("POKEBOT_REMOTE_WORKER_ARM_FILE")
        != str(ELMO_REMOTE_ARM_FILE)
        or environment.get("POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE")
        != ELMO_REMOTE_PLANNED_ROTATION_EXIT_CODE
        or environment.get("POKEBOT_REMOTE_MAX_SERVICE_JOBS")
        != ELMO_REMOTE_MAX_SERVICE_JOBS
        or environment.get("POKEBOT_REMOTE_WORKER_SAFETY_VERSION")
        != ELMO_REMOTE_WORKER_SAFETY_VERSION
        or environment.get("POKEBOT_BASELINES_DIR") != baseline_payload.get("root")
        or environment.get("POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST")
        != baseline_payload.get("manifest")
        or environment.get("POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST_SHA256")
        != baseline_payload.get("manifest_sha256")
        or environment.get("POKEBOT_R241_BASELINE_PAYLOAD_TREE_SHA256")
        != baseline_payload.get("baseline_tree_sha256")
        or environment.get(ELMO_R241_CHECKPOINT_TRANSPORT_ENV) != checkpoint_root
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo runtime environment receipt drifted")
    direct_policy = _require_mapping(
        runtime.get("direct_policy"), label="Elmo direct-policy receipt"
    )
    expected_snapshot = _require_mapping(
        runtime.get("source_snapshot"), label="Elmo runtime source snapshot"
    )
    host_snapshot = _require_mapping(
        loaded["host"].get("source_snapshot"), label="Elmo host source snapshot"
    )
    if (
        official.get("linux_x86_64_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
        or direct_policy.get("action_selector") != "direct_policy_only"
        or direct_policy.get("direct_policy_only") is not True
        or direct_policy.get("observation_scope") != "preflight_no_jobs_started"
        or direct_policy.get("runtime_call_counters_available") is not False
        or direct_policy.get("runtime_call_counters")
        != "not_measured_preflight_no_jobs_started"
        or direct_policy.get("learner_and_h10_direct_policy_only") is not True
        or direct_policy.get("frozen_non_h10_public_opponent_selectors")
        != "preserved_external_public_opponents"
        or adapter.get("runtime_enabled") is not True
        or adapter.get("matchup_adapter_bank_preserved") is not True
        or adapter.get("checkpoint_runtime_enabled") is not False
        or adapter.get("checkpoint_training_enabled") is not False
        or adapter.get("checkpoint_main_optimizer_included") is not False
        or adapter.get("isolated_bank_only_optimizer") is not True
        or adapter.get("isolated_fit_continuation_required") is not True
        or adapter.get("external_collection_runtime_enabled") is not True
        or adapter.get("matchup_adapter_external_terminal_runtime_enabled")
        is not True
        or adapter.get("current_peak_r195_roster_retained") is not True
        or adapter.get("slot_migration_status") != "no_slot_change"
        or adapter.get("immutable_slot_prefix") != IMMUTABLE_ADAPTER_SLOT_PREFIX
        or adapter.get("new_slots") != []
        or adapter.get("learner_tree_sha256") != R195_LEARNER_MATCHUP_TREE_SHA256
        or adapter.get("h10_matchup_tree_sha256") != R241_H10_MATCHUP_TREE_SHA256
        or adapter.get("h10_data_only") is not True
        or path_binding.get("identical_elmo_host_paths_required_inside_container")
        is not True
        or path_binding.get("container_path_remapping_allowed") is not False
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo r236/Matchup Adapter receipt drifted")
    for name in (
        "repo_root",
        "source_snapshot_manifest",
        "cg_lib_path",
        "adapter_receipt",
        "h10_matchup_tree",
        "baseline_payload_root",
        "baseline_payload_manifest",
        "baseline_payload_staging_receipt",
        "checkpoint_transport_host_root",
        "checkpoint_transport_staging_receipt",
    ):
        raw_value = str(path_binding.get(name) or "").strip()
        if not raw_value:
            raise R241ElmoRemoteWorkerError(f"receipt {name} is missing")
        _require_elmo_absolute_path(
            raw_value, label=f"receipt {name}"
        )
    snapshot_required = {
        "schema": R241_SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": str(path_binding.get("repo_root") or ""),
        "source_execution_root": str(path_binding.get("repo_root") or ""),
        "manifest": str(path_binding.get("source_snapshot_manifest") or ""),
        "outputs_root": str(ELMO_OUTPUTS_ROOT),
    }
    if (
        any(expected_snapshot.get(key) != value for key, value in snapshot_required.items())
        or host_snapshot != expected_snapshot
        or not str(expected_snapshot.get("manifest_sha256") or "").startswith("sha256:")
        or not str(expected_snapshot.get("source_tree_sha256") or "").startswith("sha256:")
        or not str(expected_snapshot.get("file_inventory_sha256") or "").startswith("sha256:")
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo host/runtime receipts do not bind the source snapshot"
        )
    for name in ("host", "runtime"):
        identities = _require_mapping(
            loaded[name].get("source_identities"),
            label=f"Elmo {name} source identities",
        )
        if _require_mapping(
            identities.get("source_snapshot"),
            label=f"Elmo {name} source-identity snapshot",
        ) != expected_snapshot:
            raise R241ElmoRemoteWorkerError(
                "r241 Elmo source identities do not bind the source snapshot"
            )
        if _require_mapping(
            identities.get("baseline_payload"),
            label=f"Elmo {name} source-identity baseline payload",
        ) != baseline_payload:
            raise R241ElmoRemoteWorkerError(
                "r241 Elmo source identities do not bind the external baseline payload"
            )
        if _require_mapping(
            identities.get("checkpoint_transport"),
            label=f"Elmo {name} source-identity checkpoint transport",
        ) != checkpoint_transport:
            raise R241ElmoRemoteWorkerError(
                "r241 Elmo source identities do not bind the checkpoint transport"
            )
    gameplay = loaded["gameplay"]
    collection = _require_mapping(gameplay.get("collection_contract"), label="Elmo collection contract")
    adapter_behavior = _require_mapping(
        gameplay.get("adapter_behavior"), label="Elmo gameplay adapter behavior"
    )
    deferred_refresh = _require_mapping(
        adapter.get("deferred_archetype_refresh"),
        label="Elmo deferred archetype refresh",
    )
    if (
        gameplay.get("status") != "ready_no_games_started"
        or gameplay.get("observation_scope") != "preflight_no_games_started"
        or gameplay.get("runtime_call_counters_available") is not False
        or gameplay.get("runtime_call_counters")
        != "not_measured_preflight_no_games_started"
        or gameplay.get("promotion_jobs_allowed") is not False
        or gameplay.get("allowed_job_kinds") != list(COLLECTION_JOB_KINDS)
        or collection.get("public_mix_games_exact") != 7172
        or collection.get("marnie_h10_is_minimum_not_exclusive_public_opponent")
        is not True
        or collection.get("public_mix_local_only") is not False
        or _require_mapping(
            collection.get("baseline_payload"),
            label="Elmo gameplay external baseline payload",
        ) != baseline_payload
        or _require_mapping(
            collection.get("checkpoint_transport"),
            label="Elmo gameplay checkpoint transport",
        ) != checkpoint_transport
        or adapter_behavior.get("learner_matchup_adapter_runtime") is not True
        or adapter_behavior.get("checkpoint_adapter_runtime") is not False
        or adapter_behavior.get("checkpoint_adapter_training") is not False
        or adapter_behavior.get("checkpoint_adapter_main_optimizer_included")
        is not False
        or adapter_behavior.get("isolated_adapter_fit_continuation") is not True
        or adapter_behavior.get("current_peak_r195_roster_retained") is not True
        or adapter_behavior.get("slot_migration_status") != "no_slot_change"
        or adapter_behavior.get("new_slots") != []
        or adapter_behavior.get("non_h10_diverse_public_packages_use_normal_loader")
        is not True
        or adapter_behavior.get("non_h10_public_opponent_selectors")
        != "preserved_external_public_opponents"
        or deferred_refresh.get("required_for_r241_activation") is not False
        or deferred_refresh.get("slot_migration_status") != "no_slot_change"
        or deferred_refresh.get("new_slots") != []
        or deferred_refresh.get("baseline_slot_registry_sha256")
        != BASELINE_ADAPTER_ROSTER_SHA256
    ):
        raise R241ElmoRemoteWorkerError("r241 Elmo gameplay receipt drifted")
    # Retain an absolute identity for callers that need to compare a registry
    # pin to the bytes on the mounted Elmo receipt share.
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "payload": manifest,
        "host_receipt": loaded["host"],
        "runtime_receipt": runtime,
        "gameplay_receipt": gameplay,
    }


def _validate_worker_image_overlay(
    binding: Mapping[str, Any],
    *,
    owner_contract_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the overlay's image ID and receipt to bind this source tree."""

    image = _require_mapping(binding, label="r241 activation-overlay worker image")
    allowed = {
        "schema",
        "image_id_sha256",
        "receipt",
        "source_snapshot",
        "tag",
    }
    if set(image) not in (allowed, allowed - {"tag"}):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay worker image has an unsupported shape"
        )
    image_id = _require_sha256(
        image.get("image_id_sha256"), label="r241 activation overlay image ID"
    )
    source = _require_mapping(
        image.get("source_snapshot"), label="r241 activation-overlay image source"
    )
    expected_source = {
        "owner_contract_sha256": owner_contract_sha256,
        "manifest_sha256": source_snapshot.get("manifest_sha256"),
        "source_tree_sha256": source_snapshot.get("source_tree_sha256"),
    }
    if (
        image.get("schema") != R241_ELMO_WORKER_IMAGE_SCHEMA
        or source != expected_source
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay worker image does not bind this source snapshot"
        )
    receipt = _require_mapping(
        image.get("receipt"), label="r241 activation-overlay worker image receipt"
    )
    if set(receipt) != {"path", "sha256"}:
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay worker image receipt has an unsupported shape"
        )
    receipt_file = _immutable_external_receipt(
        receipt.get("path") or "",
        expected_sha256=_require_sha256(
            receipt.get("sha256"), label="r241 activation overlay image receipt sha256"
        ),
        label="r241 Elmo worker image receipt",
    )
    _receipt_path, receipt_payload = _json_object(
        receipt_file, label="r241 Elmo worker image receipt"
    )
    receipt_image = _require_mapping(
        receipt_payload.get("image"), label="r241 Elmo worker image receipt image"
    )
    receipt_source = _require_mapping(
        receipt_payload.get("source_snapshot"),
        label="r241 Elmo worker image receipt source snapshot",
    )
    authority = _require_mapping(
        receipt_payload.get("activation_authority"),
        label="r241 Elmo worker image receipt authority",
    )
    smoke = _require_mapping(
        receipt_payload.get("noncanonical_network_disabled_one_shot_smoke"),
        label="r241 Elmo worker image receipt no-network smoke",
    )
    if (
        receipt_payload.get("schema") != R241_ELMO_WORKER_IMAGE_SCHEMA
        or receipt_payload.get("candidate_id")
        != "alakazam-new-list-direct-policy-r241"
        or receipt_payload.get("status")
        != "sealed_noncanonical_no_network_smoke_passed"
        or receipt_payload.get("create_only") is not True
        or receipt_image.get("image_id_sha256") != image_id
        or any(receipt_source.get(key) != value for key, value in expected_source.items())
        or authority.get("external_activation_overlay_created") is not False
        or authority.get("managed_service_start_authorized") is not False
        or authority.get("listener_started") is not False
        or authority.get("training_started") is not False
        or smoke.get("validated_external_d162") is not True
        or smoke.get("simulator_battles_started") != 0
        or smoke.get("native_function_calls") != 0
        or smoke.get("search_calls") != 0
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay worker image receipt identity drifted"
        )
    if "tag" in image and str(image.get("tag") or "").strip() != str(
        receipt_image.get("tag") or ""
    ).strip():
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay worker image tag disagrees with its receipt"
        )
    return {
        "image_id_sha256": image_id,
        "receipt_path": str(receipt_file),
        "receipt_sha256": _sha256(receipt_file),
    }


def validate_r241_elmo_activation_overlay(
    *,
    overlay_path: Path | str,
    overlay_sha256: str,
    preflight: R241ElmoPreflight,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit the local mirror of r241's one shared activation overlay.

    Offline preflight intentionally knows only immutable source/baseline and
    checkpoint-transport evidence.  It cannot authorize a listener or jobs.
    A later create-only overlay binds the already-written four Elmo receipts
    and the separately issued, host-mirrored owner-start authorization.  This
    verifier consumes *only* Elmo's byte-identical local copy, whose supplied
    checksum is also the common logical overlay identity shared with Inzi.
    """

    overlay_file = _immutable_external_receipt(
        overlay_path,
        expected_sha256=overlay_sha256,
        label="r241 Elmo activation overlay",
    )
    _overlay_path, overlay = _json_object(
        overlay_file, label="r241 Elmo activation overlay"
    )
    owner = _require_mapping(
        preflight.sources.get("owner_contract"),
        label="r241 preflight owner contract",
    )
    owner_contract_sha256 = _require_sha256(
        owner.get("sha256"), label="r241 preflight owner contract sha256"
    )
    source_snapshot = _require_mapping(
        preflight.sources.get("source_snapshot"),
        label="r241 preflight source snapshot",
    )
    source_staging = _require_mapping(
        preflight.sources.get("source_staging_receipt"),
        label="r241 preflight source staging receipt",
    )
    registry = _require_mapping(
        preflight.sources.get("runtime_registry"),
        label="r241 preflight pending runtime registry",
    )
    manifest_payload = _require_mapping(
        manifest.get("payload"), label="r241 Elmo preflight manifest payload"
    )
    manifest_receipts = _require_mapping(
        manifest_payload.get("receipts"), label="r241 Elmo manifest receipts"
    )
    if (
        manifest.get("sha256") != _require_sha256(
            manifest.get("sha256"), label="r241 Elmo preflight manifest sha256"
        )
        or manifest_payload.get("schema") != R241_ELMO_REMOTE_MANIFEST_SCHEMA
        or manifest_payload.get("revision") != R241_REVISION
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay must bind a validated local Elmo preflight manifest"
        )
    allowed_overlay_fields = {
        "schema",
        "revision",
        "candidate_id",
        "status",
        "passed",
        "owner_contract_sha256",
        "base_registry",
        "source_snapshot",
        "baseline_payloads",
        "peak_r195_preservation",
        "remote_collection",
        "worker_image",
        "mirrors",
        "owner_start_authorization",
    }
    if (
        set(overlay) != allowed_overlay_fields
        or overlay.get("schema") != R241_ACTIVATION_OVERLAY_SCHEMA
        or overlay.get("revision") != R241_REVISION
        or overlay.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or overlay.get("status") != "ready"
        or overlay.get("passed") is not True
        or overlay.get("owner_contract_sha256") != owner_contract_sha256
    ):
        raise R241ElmoRemoteWorkerError("r241 activation overlay identity drifted")
    base_registry = _require_mapping(
        overlay.get("base_registry"), label="r241 activation-overlay base registry"
    )
    if base_registry != {
        "path": "state/alakazam-new-list-direct-r241-runtime-registry.json",
        "sha256": registry.get("sha256"),
    }:
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay does not bind the immutable pending registry"
        )
    mirrors = _require_mapping(overlay.get("mirrors"), label="r241 overlay mirrors")
    if mirrors != {
        "schema": "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v1",
        "hosts": ["inzi", "elmo"],
        "byte_identical_required": True,
    }:
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay does not declare one byte-identical Inzi/Elmo mirror"
        )

    source_overlay = _require_mapping(
        overlay.get("source_snapshot"), label="r241 activation-overlay source snapshot"
    )
    source_hosts = _require_mapping(
        source_overlay.get("hosts"), label="r241 activation-overlay source hosts"
    )
    expected_source_host = {
        "root": str(preflight.repo_root),
        "manifest": str(preflight.source_snapshot_manifest),
        "outputs_root": str(ELMO_OUTPUTS_ROOT),
        "staging_receipt": source_staging.get("staging_receipt"),
        "staging_receipt_sha256": source_staging.get("staging_receipt_sha256"),
    }
    if (
        source_overlay.get("schema") != R241_SOURCE_SNAPSHOT_SCHEMA
        or source_overlay.get("candidate_id")
        != "alakazam-new-list-direct-policy-r241"
        or source_overlay.get("owner_contract_sha256") != owner_contract_sha256
        or source_overlay.get("status") != "ready"
        or source_overlay.get("manifest_sha256")
        != source_snapshot.get("manifest_sha256")
        or source_overlay.get("source_tree_sha256")
        != source_snapshot.get("source_tree_sha256")
        or set(source_hosts) != {"inzi", "elmo"}
        or _require_mapping(
            source_hosts.get("elmo"), label="r241 activation-overlay Elmo source host"
        )
        != expected_source_host
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay source snapshot does not bind this Elmo preflight"
        )

    worker_image = _validate_worker_image_overlay(
        _require_mapping(
            overlay.get("worker_image"), label="r241 activation-overlay worker image"
        ),
        owner_contract_sha256=owner_contract_sha256,
        source_snapshot=source_snapshot,
    )
    container_image_id = _require_sha256(
        os.environ.get(ELMO_R241_WORKER_IMAGE_ID_ENV),
        label="r241 Elmo container image ID environment",
    )
    if container_image_id != worker_image["image_id_sha256"]:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo container image ID does not match activation overlay"
        )

    baseline_overlay = _require_mapping(
        overlay.get("baseline_payloads"),
        label="r241 activation-overlay baseline payload",
    )
    baseline_hosts = _require_mapping(
        baseline_overlay.get("hosts"), label="r241 activation-overlay baseline hosts"
    )
    baseline_identity = _baseline_payload_identity(preflight.baseline_payload)
    expected_baseline_host = {
        key: baseline_identity[key]
        for key in (
            "root",
            "manifest",
            "manifest_sha256",
            "baseline_tree_sha256",
            "staging_receipt",
            "staging_receipt_sha256",
            "canonical_roster_receipt",
        )
    }
    if (
        baseline_overlay.get("schema") != R241_BASELINE_PAYLOAD_REGISTRY_SCHEMA
        or baseline_overlay.get("candidate_id")
        != "alakazam-new-list-direct-policy-r241"
        or baseline_overlay.get("status") != "ready"
        or baseline_overlay.get("separately_mounted_and_receipted") is not True
        or baseline_overlay.get("source_snapshot_fallback_allowed") is not False
        or "canonical_roster_receipt" in baseline_overlay
        or baseline_overlay.get("canonical_roster_receipt_sha256")
        != baseline_identity["canonical_roster_receipt_sha256"]
        or baseline_overlay.get("canonical_baseline_manifest_sha256")
        != baseline_identity["baseline_manifest_sha256"]
        or baseline_overlay.get("canonical_baseline_roster_sha256")
        != baseline_identity["baseline_roster_sha256"]
        or set(baseline_hosts) != {"inzi", "elmo"}
        or _require_mapping(
            baseline_hosts.get("elmo"), label="r241 activation-overlay Elmo baseline host"
        )
        != expected_baseline_host
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay baseline payload does not bind this Elmo mount"
        )

    preservation = _require_mapping(
        overlay.get("peak_r195_preservation"),
        label="r241 activation-overlay peak-r195 preservation",
    )
    if set(preservation) != {"receipt_sha256_inzi", "receipt_sha256_elmo"}:
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay may only bind the two peak-r195 preservation receipts"
        )
    for name, digest in preservation.items():
        _require_sha256(digest, label=f"r241 activation overlay preservation {name}")

    remote = _require_mapping(
        overlay.get("remote_collection"), label="r241 activation-overlay remote collection"
    )
    overlay_transport = _require_mapping(
        remote.get("checkpoint_transport"),
        label="r241 activation-overlay checkpoint transport",
    )
    trainer_visible_root = str(overlay_transport.get("trainer_visible_root") or "").strip()
    # This path belongs to Inzi's publisher/launcher rather than the Elmo
    # container, so it is deliberately not dereferenced here.  It must still
    # be an external host path: accepting a container path would silently
    # revive the historical /workspace remap instead of the receipt-bound
    # content-addressed transport.
    if (
        not trainer_visible_root
        or not Path(trainer_visible_root).is_absolute()
        or trainer_visible_root == "/workspace"
        or trainer_visible_root.startswith("/workspace/")
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay checkpoint transport lacks an external trainer-visible root"
        )
    expected_transport = {
        "status": "ready",
        **_checkpoint_transport_identity(preflight.checkpoint_transport),
        "trainer_visible_root": trainer_visible_root,
    }
    expected_remote = {
        "endpoint_id": ELMO_R241_ENDPOINT_ID,
        "manifest_sha256": manifest.get("sha256"),
        "host_receipt_sha256": _require_mapping(
            manifest_receipts.get("host"), label="r241 Elmo manifest host receipt"
        ).get("sha256"),
        "runtime_receipt_sha256": _require_mapping(
            manifest_receipts.get("runtime"), label="r241 Elmo manifest runtime receipt"
        ).get("sha256"),
        "gameplay_receipt_sha256": _require_mapping(
            manifest_receipts.get("gameplay"), label="r241 Elmo manifest gameplay receipt"
        ).get("sha256"),
        "checkpoint_transport": expected_transport,
    }
    if remote != expected_remote:
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay does not bind the already-written Elmo preflight receipts"
        )

    authorization = _require_mapping(
        overlay.get("owner_start_authorization"),
        label="r241 activation-overlay owner-start authorization",
    )
    authorization_hosts = _require_mapping(
        authorization.get("hosts"),
        label="r241 activation-overlay owner-start authorization hosts",
    )
    if (
        set(authorization)
        != {"schema", "sha256", "byte_identical_mirrors_required", "hosts"}
        or authorization.get("schema") != R241_OWNER_START_AUTHORIZATION_SCHEMA
        or authorization.get("byte_identical_mirrors_required") is not True
        or set(authorization_hosts) != {"inzi", "elmo"}
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay owner-start authorization mirror contract drifted"
        )
    elmo_authorization = _require_mapping(
        authorization_hosts.get("elmo"),
        label="r241 activation-overlay Elmo owner-start authorization",
    )
    if set(elmo_authorization) != {"path"}:
        raise R241ElmoRemoteWorkerError(
            "r241 activation overlay Elmo owner-start authorization must contain only path"
        )
    authorization_file = _immutable_external_receipt(
        elmo_authorization.get("path") or "",
        expected_sha256=_require_sha256(
            authorization.get("sha256"),
            label="r241 activation overlay owner-start authorization sha256",
        ),
        label="r241 Elmo owner-start authorization receipt",
    )
    _authorization_path, authorization_payload = _json_object(
        authorization_file, label="r241 Elmo owner-start authorization receipt"
    )
    expected_authorization = {
        "schema": R241_OWNER_START_AUTHORIZATION_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authorized",
        "authorized": True,
        "owner_contract_sha256": owner_contract_sha256,
        "allowed_actions": ["managed_r241_training_start"],
        "source_snapshot_manifest_sha256": source_snapshot.get("manifest_sha256"),
        "source_tree_sha256": source_snapshot.get("source_tree_sha256"),
        "canonical_baseline_manifest_sha256": baseline_identity[
            "baseline_manifest_sha256"
        ],
        "canonical_baseline_roster_sha256": baseline_identity[
            "baseline_roster_sha256"
        ],
        "submission_boundary": {
            "exact_count": 1,
            "checkpoint_source": "expert_before_iter_00010.pt",
            "intermediate_iteration_5_submission_allowed": False,
            "retry_copy_or_duplicate_allowed": False,
        },
    }
    if any(
        authorization_payload.get(name) != value
        for name, value in expected_authorization.items()
    ):
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo owner-start authorization does not bind the current overlay identities"
        )
    authorization_provenance = _require_mapping(
        authorization_payload.get("authorization_provenance"),
        label="r241 Elmo owner-start authorization provenance",
    )
    if authorization_provenance != {
        "schema": R241_OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA,
        "create_only": True,
        "explicit_operator_intent": "authorize_managed_r241_training_start",
    }:
        raise R241ElmoRemoteWorkerError(
            "r241 Elmo owner-start authorization was not emitted by the explicit create-only generator"
        )
    return {
        "path": str(overlay_file),
        "sha256": _sha256(overlay_file),
        "payload": overlay,
        "owner_start_authorization_path": str(authorization_file),
        "owner_start_authorization_sha256": _sha256(authorization_file),
        "worker_image": worker_image,
    }


def r241_elmo_endpoint_registry_template() -> dict[str, Any]:
    """Return the exact registry subobject for the r241 launcher owner.

    Receipt hashes intentionally remain blank until a local Elmo preflight has
    produced them.  A launcher must reject an empty pin at activation time.
    """

    paths = receipt_paths(DEFAULT_ELMO_RECEIPT_DIR)
    return {
        "schema": R241_ELMO_REMOTE_ENDPOINT_REGISTRY_SCHEMA,
        "revision": R241_REVISION,
        "require_explicit_eligible_endpoints": True,
        "legacy_or_default_endpoint_fallback_allowed": False,
        "pure_rl_public_mix_local_only": "0",
        "eligible_endpoints": [
            {
                "id": ELMO_R241_ENDPOINT_ID,
                "endpoint": ELMO_R241_ENDPOINT,
                "host_role": "elmo",
                "capability": ELMO_R241_COLLECTION_CAPABILITY,
                "manifest_path": str(paths["manifest"]),
                "manifest_sha256": "",
                "host_receipt_path": str(paths["host"]),
                "host_receipt_sha256": "",
                "runtime_receipt_path": str(paths["runtime"]),
                "runtime_receipt_sha256": "",
                "gameplay_receipt_path": str(paths["gameplay"]),
                "gameplay_receipt_sha256": "",
                "allowed_job_kinds": list(COLLECTION_JOB_KINDS),
                "r236_linux_sha256": R241_OFFICIAL_LINUX_LIBCG_SHA256,
                "learner_matchup_tree_sha256": R195_LEARNER_MATCHUP_TREE_SHA256,
                "h10_matchup_tree_sha256": R241_H10_MATCHUP_TREE_SHA256,
            }
        ],
    }


__all__ = [
    "COLLECTION_JOB_KINDS",
    "DEFAULT_ELMO_RECEIPT_DIR",
    "ELMO_R241_COLLECTION_CAPABILITY",
    "ELMO_R241_ENDPOINT",
    "ELMO_R241_ENDPOINT_ID",
    "ELMO_R241_ENDPOINT_PORT",
    "GAMEPLAY_RECEIPT_FILENAME",
    "HOST_RECEIPT_FILENAME",
    "LEGACY_OR_DEFAULT_ENDPOINTS",
    "MANIFEST_FILENAME",
    "R195_LEARNER_MATCHUP_TREE_SHA256",
    "R241_ELMO_REMOTE_ENDPOINT_REGISTRY_SCHEMA",
    "R241_ELMO_REMOTE_MANIFEST_SCHEMA",
    "R241_ELMO_WORKER_IMAGE_SCHEMA",
    "R241_CHECKPOINT_TRANSPORT_SCHEMA",
    "R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA",
    "R241_ACTIVATION_OVERLAY_SCHEMA",
    "ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT",
    "ELMO_R241_CHECKPOINT_TRANSPORT_ENV",
    "ELMO_R241_CHECKPOINT_VERIFY_PORT_ENV",
    "ELMO_R241_WORKER_IMAGE_ID_ENV",
    "R241_LATEST_OWNER_CLARIFICATION_REVISION",
    "R241_OFFICIAL_LINUX_LIBCG_SHA256",
    "RUNTIME_RECEIPT_FILENAME",
    "R241ElmoBaselinePayload",
    "R241ElmoCheckpointTransport",
    "R241ElmoPreflight",
    "R241ElmoRemoteWorkerError",
    "assert_r241_elmo_endpoint",
    "build_r241_elmo_collection_environment",
    "build_r241_elmo_preflight_receipts",
    "preflight_r241_elmo_remote_collection",
    "r241_elmo_endpoint_registry_template",
    "receipt_paths",
    "validate_current_r241_sources",
    "validate_r241_elmo_activation_overlay",
    "validate_r241_elmo_preflight_manifest",
    "write_r241_elmo_preflight_receipts",
]
