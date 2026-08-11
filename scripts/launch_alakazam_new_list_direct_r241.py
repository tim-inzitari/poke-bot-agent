#!/usr/bin/env python3
"""Receipt-gated launcher for the isolated Alakazam r241 training lineage.

This wrapper is intentionally a *preflight and command builder* by default.
It never starts a worker or submits anything unless an operator explicitly
passes ``--execute`` after every canonical activation gate and every host-local
receipt has passed.  The implementation deliberately does not reuse a
historical r175/r195 service unit: r241 has a different deck, guide, fixed
ten-update horizon, and sealed r236 ``CG_LIB_PATH``.  It does, however,
preserve the established 7,172-game diverse public mix, all non-combo r195
heads/routes, and activated Matchup Adapter runtime.

The H10 Marnie package is admitted through ``baselines_runtime``'s r241
data-only adapter.  The launcher only supplies its checksum-bound receipt; it
does not import or run the package's legacy ``main.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


# Static source-snapshot validation imports this launcher from an immutable
# deployment root.  Do not let a caller missing the usual environment flag
# create an unbound ``__pycache__`` in that tree before validation can reject
# it.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r241_direct_policy_runtime import (  # noqa: E402
    R241_DIRECT_POLICY_ONLY_ENV,
    R241_DIRECT_POLICY_RECEIPT_ENV,
    R241_H10_ADAPTER_RECEIPT_BASENAME,
    R241_H10_CONTENT_SHA256,
    R241_H10_MODEL_SHA256,
    R241_H10_OPPONENT_ID,
    R241_OFFICIAL_LINUX_LIBCG_SHA256,
    R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME,
    R241_REVISION,
    R241DirectPolicyRuntimeError,
    assert_direct_policy_environment,
    sha256_file,
    validate_sealed_official_libcg,
)
from poke_bot import r241_checkpoint_receipts as checkpoint_receipts  # noqa: E402
from poke_bot import r241_baseline_payload_snapshot as baseline_payload_snapshot  # noqa: E402


REGISTRY_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241_runtime_registry/v1"
PRESERVATION_SCHEMA = checkpoint_receipts.PEAK_R195_PRESERVATION_SCHEMA
SOURCE_SNAPSHOT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1"
)
REMOTE_ENDPOINT_REGISTRY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_remote_endpoint_registry/v1"
)
BASELINE_PAYLOAD_REGISTRY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_baseline_payload_registry/v1"
)
ACTIVATION_OVERLAY_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_activation_overlay/v1"
)
ACTIVATION_OVERLAY_MIRROR_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirror/v1"
)
ELMO_WORKER_IMAGE_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_elmo_worker_image/v1"
OWNER_START_AUTHORIZATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization/v1"
)
OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_owner_start_authorization_generator/v1"
)
EXPERT_WINDOW_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_expert_window_staging/v1"
)
EXPERT_RECEIPT_SCHEMA = "poke_bot.expert_latest20_receipt/v1"
DEFAULT_REGISTRY = ROOT / "state/alakazam-new-list-direct-r241-runtime-registry.json"
EXACT_WINDOW_START = "2026-07-22"
EXACT_WINDOW_END = "2026-08-10"
EXACT_WINDOW_DAYS = 20
EXACT_SELF_PLAY_FRACTION = "0.12493899463152758"
EXACT_GAMES_PER_UPDATE = 8_196
EXACT_SELF_GAMES = 1_024
EXACT_PUBLIC_GAMES = 7_172
EXACT_FIRST_SEAT_GAMES = 4_098
EXACT_SECOND_SEAT_GAMES = 4_098
EXACT_H10_MINIMUM = 1_024
R241_ELMO_ENDPOINT_ID = "elmo-r241-official-r236-direct-policy-8767"
R241_ELMO_ENDPOINT = "192.168.1.143:8767"
R241_ELMO_COLLECTION_CAPABILITY = "r241_direct_policy_collection_v1"
R241_CHECKPOINT_TRANSPORT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport/v1"
)
R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport_staging/v1"
)
R241_ELMO_CHECKPOINT_TRANSPORT_CONTAINER_ROOT = "/workspace/checkpoint"
R241_ELMO_CHECKPOINT_TRANSPORT_ENV = "POKEBOT_REMOTE_CHECKPOINT_ROOT"
R241_ELMO_CHECKPOINT_TRANSPORT_FILENAME_SCHEME = (
    "poke_bot.remote_jobs.digest_addressed_basename/v1"
)
R241_ELMO_REMOTE_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_elmo_official_r236_remote_manifest/v1"
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
R241_ELMO_LEGACY_ENDPOINTS = frozenset(
    {
        "192.168.1.143:8765",
        "192.168.1.158:8766",
        "bert.local:8766",
        "elmo:8765",
        "bert:8766",
    }
)
R241_REMOTE_ALLOWED_JOB_KINDS = (
    "play",
    "self_play",
    "self_play_multi",
    "runtime_probe",
)

# The r241 source snapshot is a closure, rather than a mutable checkout with a
# handful of launch scripts copied out of it.  These are the direct training,
# collection, adapter, packaging, and source-of-truth members that must be
# present in its content-addressed inventory.  Baseline payloads are mounted
# separately and admitted through their own receipts; they must never become
# an untracked source-tree fallback.
_REQUIRED_SOURCE_SNAPSHOT_FILES = frozenset(
    {
        "scripts/launch_alakazam_new_list_direct_r241.py",
        "scripts/launch_pure_rl.py",
        "scripts/train_pure_rl.py",
        "scripts/train_round_robin.py",
        # ``launch_pure_rl.py`` starts these helpers in a normal r241 run.
        # They must be part of the sealed source closure rather than silently
        # disappearing when the launcher executes from a code-only snapshot.
        "scripts/canary_game_accuracy.py",
        "scripts/resource_watcher.py",
        "scripts/unattended_monitor.py",
        "scripts/run_remote_worker.py",
        "scripts/launch_r241_elmo_official_r236_worker.py",
        "deploy/elmo/docker-compose.r241-elmo-official-r236-remote-worker.yml.template",
        "deploy/elmo/r241-elmo-official-r236-remote-worker.env.template",
        "deploy/systemd/pokebot-r241-elmo-official-r236-remote-worker.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-finalize.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-submission-queue.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-upload.service.template",
        "scripts/preflight_alakazam_new_list_direct_r241_service_chain.py",
        "scripts/finalize_alakazam_new_list_direct_r241.py",
        "scripts/process_alakazam_new_list_direct_r241_submission_queue.py",
        "scripts/upload_alakazam_new_list_direct_r241_submission_queue.py",
        "scripts/process_kaggle_submission_queue.py",
        "scripts/stage_r241_official_libcg_direct_policy.py",
        "scripts/stage_r241_elmo_checkpoint_transport.py",
        "scripts/stage_r241_source_snapshot.py",
        "scripts/stage_r241_baseline_payload_snapshot.py",
        "scripts/publish_r241_activation_overlay.py",
        "scripts/generate_r241_owner_start_authorization.py",
        "scripts/generate_r241_marnie_direct_policy_adapter_receipt.py",
        "scripts/install_r241_activation_overlay_mirror.py",
        "scripts/generate_r241_canonical_baseline_roster.py",
        "scripts/generate_r241_checkpoint_receipts.py",
        "scripts/transfer_r241_exact20_alakazam_corpus.py",
        "poke_bot/paths.py",
        "poke_bot/config.py",
        "poke_bot/model.py",
        "poke_bot/train.py",
        "poke_bot/checkpoint.py",
        "poke_bot/dormant_adapter_compat.py",
        "poke_bot/baselines_runtime.py",
        "poke_bot/remote_jobs.py",
        "poke_bot/remote_sim_jobs.py",
        "poke_bot/worker_pool.py",
        "poke_bot/batched_infer.py",
        "poke_bot/matchup_adapter_routes.py",
        "poke_bot/public_matchup_router.py",
        "poke_bot/public_multi_env_safety.py",
        "poke_bot/r241_checkpoint_receipts.py",
        "poke_bot/r241_baseline_payload_snapshot.py",
        "poke_bot/r241_canonical_baseline_roster.py",
        "poke_bot/r241_direct_policy_runtime.py",
        "poke_bot/r241_elmo_official_r236_remote_worker.py",
        "poke_bot/r241_marnie_direct_policy_adapter.py",
        "poke_bot/pure_rl/matchup_adapter_trainer.py",
        "poke_bot/pure_rl/model_profile.py",
        "poke_bot/pure_rl/model_registry.py",
        "config/rl_protocol.yaml",
        "state/alakazam-new-list-direct-policy-r241.json",
        "state/alakazam-new-list-direct-r241-runtime-registry.json",
        "state/alakazam-new-list-direct-r241-official-libcg-staging.json",
        "state/alakazam-new-list-direct-r241-expert-window-staging.json",
        "state/alakazam-new-list-direct-r241-strategic-curriculum.json",
        "state/alakazam-new-list-direct-r241-strategic-head-roles.json",
        "state/alakazam-new-list-direct-r241-strategic-curriculum-validation.json",
        # ``poke_bot.own_deck_successor`` resolves this canonical typed source
        # relative to the execution root.  Keep the data dependency explicit:
        # recursive Python-import discovery cannot discover JSON path literals.
        "state/alakazam-own-deck-ledger-successor-r258.json",
        "state/canonical-libcg-r236.json",
        "state/matchup_adapter_roster.json",
        "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
        "decks/archetype-samples/alakazam-new-list-direct-r241.csv",
        "config/deck_guides/alakazam-new-list-direct-r241.yaml",
    }
)

_FORBIDDEN_SNAPSHOT_COMPONENTS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "baselines",
        "cg",
        "outputs",
        "runtime",
        "overlay",
        "overlays",
    }
)


class R241LaunchError(RuntimeError):
    """The r241 line cannot safely form a production launch command."""


@dataclass(frozen=True)
class HostContext:
    name: str
    runtime_root: Path
    official_cg_root: Path
    adapter_receipt: Path
    preservation_receipt: Path
    expert_archive_receipt: Path
    expert_manifest_pointer: Path


@dataclass(frozen=True)
class SourceSnapshotContext:
    """One immutable code tree and its separate durable artifact root."""

    root: Path
    manifest: Path
    manifest_sha256: str
    source_tree_sha256: str
    outputs_root: Path


@dataclass(frozen=True)
class BaselinePayloadContext:
    """One separately mounted, immutable public-baseline library."""

    root: Path
    manifest: Path
    manifest_sha256: str
    baseline_tree_sha256: str
    canonical_roster_receipt: Path
    canonical_roster_receipt_sha256: str
    baseline_manifest_sha256: str
    baseline_roster_sha256: str
    baseline_roster: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ActivationOverlayContext:
    """The external, create-only activation transaction for one snapshot."""

    path: Path
    sha256: str
    authorization_receipt: Path
    authorization_receipt_sha256: str
    mirror_receipt: Path
    mirror_receipt_sha256: str


@dataclass(frozen=True)
class PreservationContext:
    active_gate_contract: Path
    frozen_specialist_registry: Path
    research_control_registry: Path
    learner_matchup_tree: Path
    adapter_activation_receipt: Path
    expert_manifest_pointer: Path
    preservation_receipt_sha256: str
    adapter_receipt_sha256: str
    official_collect_fraction: float
    research_control_games: int
    matchup_adapter_epochs_per_update: int
    trainer_args: tuple[str, ...]
    source_snapshot: SourceSnapshotContext
    baseline_payload: BaselinePayloadContext | None = None
    activation_overlay: ActivationOverlayContext | None = None


def _json_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241LaunchError(f"{label} must be a regular non-symlink file: {raw}")
    resolved = raw.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241LaunchError(f"{label} is not readable JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise R241LaunchError(f"{label} must contain an object: {resolved}")
    return resolved, payload


def _sha256(path: Path | str) -> str:
    return sha256_file(Path(path))


def _expect_sha256(path: Path | str, expected: object, *, label: str) -> Path:
    expected_text = str(expected or "")
    if not expected_text.startswith("sha256:") or len(expected_text) != 71:
        raise R241LaunchError(f"{label} is missing a canonical SHA-256")
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241LaunchError(f"{label} is not a regular file: {raw}")
    resolved = raw.resolve()
    actual = _sha256(resolved)
    if actual != expected_text:
        raise R241LaunchError(
            f"{label} checksum mismatch: expected={expected_text} actual={actual}"
        )
    return resolved


def _valid_sha256(value: object, *, label: str) -> str:
    digest = str(value or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise R241LaunchError(f"{label} is missing a canonical SHA-256")
    try:
        int(digest.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise R241LaunchError(f"{label} is not a hexadecimal SHA-256") from exc
    return digest


def _real_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241LaunchError(f"{label} must be a real non-symlink directory: {raw}")
    return raw.resolve()


def _snapshot_member(root: Path, relative: str, *, label: str) -> Path:
    member = Path(relative)
    if (
        not relative
        or member.is_absolute()
        or ".." in member.parts
        or "." in member.parts
    ):
        raise R241LaunchError(f"{label} has an unsafe snapshot path: {relative!r}")
    raw = root.joinpath(*member.parts)
    if raw.is_symlink() or not raw.is_file():
        raise R241LaunchError(f"{label} is not a regular snapshot file: {raw}")
    resolved = raw.resolve()
    if root not in resolved.parents:
        raise R241LaunchError(f"{label} escapes the source snapshot: {raw}")
    return resolved


def _bound_source_path(root: Path, relative: object, *, label: str) -> Path:
    """Return a relative source member without permitting checkout escape.

    Unlike :func:`_snapshot_member`, this helper is also used while building
    the no-I/O command in unit tests, so it deliberately does not stat the
    target.  Activation verifies the member against the snapshot manifest
    before this value can be executed.
    """

    rendered = str(relative or "").strip()
    candidate = Path(rendered)
    if (
        not rendered
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "." in candidate.parts
    ):
        raise R241LaunchError(f"{label} must be a safe relative source path")
    return root.joinpath(*candidate.parts)


def _source_tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the location-independent source-tree digest used by r241.

    The manifest deliberately records only relative paths and file identities,
    so the same sealed source can be mounted under different Inzi/Elmo roots.
    """

    canonical = [
        {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": _exact_int(row["size_bytes"], label="snapshot file size"),
        }
        for row in sorted(rows, key=lambda item: str(item["path"]))
    ]
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_snapshot_tree_shape(
    root: Path,
    *,
    manifest: Path,
    inventory_paths: set[str],
) -> None:
    """Ensure the manifest is the entire immutable executable closure.

    A hidden mutable overlay is as unsafe as an unpinned source file.  The
    snapshot therefore admits no symlinks, bytecode/cache trees, embedded
    baseline/native payloads, or unbound environment files; every regular
    source member other than the self-digesting manifest must appear exactly
    once in the checksum inventory.
    """

    observed: set[str] = set()
    for directory_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory = Path(directory_text)
        for name in list(directory_names):
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise R241LaunchError(
                    f"r241 source snapshot contains a symlink directory: {relative}"
                )
            if child.stat().st_mode & 0o222:
                raise R241LaunchError(
                    f"r241 source snapshot directory must be read-only: {relative}"
                )
            if name in _FORBIDDEN_SNAPSHOT_COMPONENTS:
                raise R241LaunchError(
                    f"r241 source snapshot contains forbidden component: {relative}"
                )
            if name.startswith(".env") or name.endswith(".env"):
                raise R241LaunchError(
                    f"r241 source snapshot contains an unbound environment directory: {relative}"
                )
        for name in file_names:
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink() or not child.is_file():
                raise R241LaunchError(
                    f"r241 source snapshot contains a non-regular file: {relative}"
                )
            if child.resolve() == manifest:
                continue
            if any(part in _FORBIDDEN_SNAPSHOT_COMPONENTS for part in Path(relative).parts):
                raise R241LaunchError(
                    f"r241 source snapshot contains forbidden component: {relative}"
                )
            if name.startswith(".env") or name.endswith(".env"):
                raise R241LaunchError(
                    f"r241 source snapshot contains an unbound environment file: {relative}"
                )
            observed.add(relative)
    if observed != inventory_paths:
        unbound = sorted(observed - inventory_paths)
        absent = sorted(inventory_paths - observed)
        details: list[str] = []
        if unbound:
            details.append("unbound=" + ", ".join(unbound))
        if absent:
            details.append("missing=" + ", ".join(absent))
        raise R241LaunchError(
            "r241 source snapshot inventory is not the complete code closure: "
            + "; ".join(details)
        )


def _require(mapping: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in mapping:
        raise R241LaunchError(f"{label} is missing {key!r}")
    return mapping[key]


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241LaunchError(f"{label} must be an exact integer")
    return value


def _as_path(value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise R241LaunchError(f"{label} is missing")
    return Path(text).expanduser().resolve()


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> tuple[Path, dict[str, Any]]:
    registry_path, registry = _json_object(path, label="r241 runtime registry")
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or _exact_int(registry.get("revision"), label="registry revision")
        != R241_REVISION
    ):
        raise R241LaunchError("unsupported r241 runtime registry schema/revision")
    return registry_path, registry


def _activation_overlay_base_projection_is_pending(
    registry: Mapping[str, Any],
) -> None:
    """Require the snapshot-local registry to remain an inert intent projection.

    The source snapshot is sealed before host receipts exist.  Letting that
    source file evolve from ``pending`` to ``ready`` would either create a
    manifest cycle or reintroduce a mutable checkout as launch authority.  The
    only supported transition is the external create-only activation overlay.
    """

    source = dict(_require(registry, "source_snapshot", label="registry"))
    baseline = dict(_require(registry, "baseline_payloads", label="registry"))
    preservation = dict(
        _require(registry, "peak_r195_preservation", label="registry")
    )
    remote = _validate_remote_collection_contract(registry)
    checkpoint_transport = _remote_checkpoint_transport(registry)
    run = dict(_require(registry, "run", label="registry"))
    if (
        source.get("status") != "pending_immutable_source_snapshot"
        or str(source.get("manifest_sha256") or "")
        or str(source.get("source_tree_sha256") or "")
        or baseline.get("status") != "pending_external_baseline_payload_snapshot"
        or any(
            str(baseline.get(key) or "").strip()
            for key in (
                "canonical_roster_receipt",
                "canonical_roster_receipt_sha256",
                "canonical_baseline_manifest_sha256",
                "canonical_baseline_roster_sha256",
            )
        )
        or any(
            str(preservation.get(key) or "").strip()
            for key in ("receipt_sha256_inzi", "receipt_sha256_elmo")
        )
        or any(
            str(remote.get(key) or "").strip()
            for key in (
                "manifest_sha256",
                "host_receipt_sha256",
                "runtime_receipt_sha256",
                "gameplay_receipt_sha256",
            )
        )
        or checkpoint_transport.get("status")
        != "pending_external_checkpoint_transport"
        or run.get("external_activation_overlay_required") is not True
        or run.get("activation_overlay_schema") != ACTIVATION_OVERLAY_SCHEMA
        or run.get("managed_service_start_authorized") is not False
        or run.get("submission_authorized") is not False
    ):
        raise R241LaunchError(
            "r241 source snapshot registry must remain a pending immutable intent; "
            "activation requires a separate external overlay"
        )
    canonical_receipts: dict[str, str] = {}
    for host in ("inzi", "elmo"):
        source_host = dict(
            _require(
                dict(_require(source, "hosts", label="source snapshot")),
                host,
                label="source snapshot hosts",
            )
        )
        baseline_host = dict(
            _require(
                dict(_require(baseline, "hosts", label="baseline payloads")),
                host,
                label="baseline payload hosts",
            )
        )
        if (
            str(source_host.get("root") or "").strip()
            or str(source_host.get("manifest") or "").strip()
            or any(str(value or "").strip() for value in baseline_host.values())
        ):
            raise R241LaunchError(
                "r241 source snapshot registry contains an unsealed host activation binding"
            )


def _overlay_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R241LaunchError(f"activation overlay {label} must be an object")
    return dict(value)


def _readonly_external_file(path: Path, *, outputs_root: Path, label: str) -> Path:
    """Require a regular, immutable external receipt rather than a snapshot file."""

    resolved = path.expanduser()
    if resolved.is_symlink() or not resolved.is_file():
        raise R241LaunchError(f"{label} must be a regular non-symlink file: {resolved}")
    resolved = resolved.resolve()
    if outputs_root not in resolved.parents:
        raise R241LaunchError(f"{label} must live under the external outputs root")
    if resolved.stat().st_mode & 0o222:
        raise R241LaunchError(f"{label} must be read-only")
    return resolved


def _validate_overlay_source_staging(
    source: Mapping[str, Any], *, owner_contract_sha256: str, host: str
) -> None:
    """Check the local staging receipt without opening another host's paths.

    The canonical publisher has already checked locally copied evidence for
    both hosts.  A service host repeats structural validation of the shared
    overlay but may dereference only its own external receipt path.
    """

    if source.get("status") != "ready":
        raise R241LaunchError("activation overlay source snapshot is not ready")
    expected_manifest = _valid_sha256(
        source.get("manifest_sha256"), label="overlay source manifest"
    )
    expected_tree = _valid_sha256(
        source.get("source_tree_sha256"), label="overlay source tree"
    )
    if source.get("owner_contract_sha256") != owner_contract_sha256:
        raise R241LaunchError("activation overlay source snapshot binds another owner contract")
    hosts = _overlay_mapping(source.get("hosts"), label="source snapshot hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241LaunchError("activation overlay source snapshot must bind Inzi and Elmo")
    if host not in {"inzi", "elmo"}:
        raise R241LaunchError("r241 overlay source staging host is invalid")
    for name in ("inzi", "elmo"):
        row = _overlay_mapping(hosts.get(name), label=f"source snapshot {name}")
        root = str(row.get("root") or "").strip()
        manifest = str(row.get("manifest") or "").strip()
        outputs = str(row.get("outputs_root") or "").strip()
        staging_path = str(row.get("staging_receipt") or "").strip()
        if (
            not root
            or not manifest
            or not outputs
            or not staging_path
            or not Path(root).is_absolute()
            or not Path(manifest).is_absolute()
            or not Path(outputs).is_absolute()
            or not Path(staging_path).is_absolute()
        ):
            raise R241LaunchError(
                f"activation overlay source snapshot {name} lacks absolute receipt bindings"
            )
        _valid_sha256(
            row.get("staging_receipt_sha256"),
            label=f"overlay {name} source staging receipt",
        )
        if name != host:
            continue
        outputs_root = _real_directory(outputs, label=f"overlay {name} outputs root")
        staging = _readonly_external_file(
            _expect_sha256(
                staging_path,
                row.get("staging_receipt_sha256"),
                label=f"overlay {name} source staging receipt",
            ),
            outputs_root=outputs_root,
            label=f"overlay {name} source staging receipt",
        )
        _, receipt = _json_object(staging, label=f"overlay {name} source staging receipt")
        binding = _overlay_mapping(receipt.get("source_snapshot"), label="source staging")
        expected = {
            "schema": SOURCE_SNAPSHOT_SCHEMA,
            "status": "authenticated_immutable_source_snapshot",
            "authenticated": True,
            "host": name,
            "root": root,
            "source_execution_root": root,
            "manifest": manifest,
            "manifest_sha256": expected_manifest,
            "source_tree_sha256": expected_tree,
            "owner_contract_sha256": owner_contract_sha256,
            "outputs_root": str(outputs_root),
        }
        if (
            receipt.get("schema")
            != "poke_bot.alakazam_new_list_direct_r241_source_snapshot_staging/v1"
            or receipt.get("revision") != R241_REVISION
            or receipt.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
            or receipt.get("status") != "passed"
            or receipt.get("passed") is not True
            or any(binding.get(key) != value for key, value in expected.items())
        ):
            raise R241LaunchError(
                f"activation overlay {name} source staging receipt binding drifted"
            )


def _validate_overlay_worker_image(
    image: Mapping[str, Any],
    *,
    owner_contract_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> None:
    """Validate the shared Elmo image identity without dereferencing Elmo paths."""

    binding = _overlay_mapping(image, label="worker image")
    allowed = {"schema", "image_id_sha256", "receipt", "source_snapshot", "tag"}
    if set(binding) not in (allowed, allowed - {"tag"}):
        raise R241LaunchError("activation overlay worker image has an unsupported shape")
    if binding.get("schema") != ELMO_WORKER_IMAGE_SCHEMA:
        raise R241LaunchError("activation overlay worker image schema drifted")
    _valid_sha256(binding.get("image_id_sha256"), label="activation overlay worker image ID")
    receipt = _overlay_mapping(binding.get("receipt"), label="worker image receipt")
    if set(receipt) != {"path", "sha256"}:
        raise R241LaunchError(
            "activation overlay worker image receipt has an unsupported shape"
        )
    receipt_path = str(receipt.get("path") or "").strip()
    if not receipt_path or not Path(receipt_path).is_absolute():
        raise R241LaunchError(
            "activation overlay worker image receipt must name an absolute Elmo path"
        )
    _valid_sha256(
        receipt.get("sha256"), label="activation overlay worker image receipt"
    )
    expected_source = {
        "owner_contract_sha256": owner_contract_sha256,
        "manifest_sha256": source_snapshot.get("manifest_sha256"),
        "source_tree_sha256": source_snapshot.get("source_tree_sha256"),
    }
    if _overlay_mapping(
        binding.get("source_snapshot"), label="worker image source snapshot"
    ) != expected_source:
        raise R241LaunchError(
            "activation overlay worker image does not bind the source snapshot"
        )
    if "tag" in binding and not str(binding.get("tag") or "").strip():
        raise R241LaunchError("activation overlay worker image informational tag is empty")


def _validate_overlay_baseline_staging(
    baseline: Mapping[str, Any], *, owner_contract_sha256: str, host: str
) -> None:
    """Validate the selected host's mount without assuming cross-host paths.

    The canonical overlay names both hosts, but an Inzi launcher must not try
    to open Elmo-only paths (and vice versa).  The canonical publisher checked
    both copied receipts before building the shared overlay; each host now
    repeats the full byte check for its own independently installed mount.
    """

    if baseline.get("status") != "ready":
        raise R241LaunchError("activation overlay baseline payload is not ready")
    hosts = _overlay_mapping(baseline.get("hosts"), label="baseline payload hosts")
    if set(hosts) != {"inzi", "elmo"}:
        raise R241LaunchError("activation overlay baseline payload must bind Inzi and Elmo")
    for name in ("inzi", "elmo"):
        row = _overlay_mapping(hosts.get(name), label=f"baseline payload {name}")
        required = (
            "root",
            "manifest",
            "manifest_sha256",
            "baseline_tree_sha256",
            "staging_receipt",
            "staging_receipt_sha256",
            "canonical_roster_receipt",
        )
        if any(not str(row.get(key) or "").strip() for key in required) or any(
            not Path(str(row[key])).is_absolute()
            for key in ("root", "manifest", "staging_receipt", "canonical_roster_receipt")
        ):
            raise R241LaunchError(
                f"activation overlay baseline payload {name} lacks absolute bindings"
            )
        for key in ("manifest_sha256", "baseline_tree_sha256", "staging_receipt_sha256"):
            _valid_sha256(row.get(key), label=f"overlay {name} baseline {key}")
    _valid_sha256(
        baseline.get("canonical_roster_receipt_sha256"),
        label="overlay canonical baseline roster receipt",
    )
    _valid_sha256(
        baseline.get("canonical_baseline_manifest_sha256"),
        label="overlay canonical baseline manifest",
    )
    _valid_sha256(
        baseline.get("canonical_baseline_roster_sha256"),
        label="overlay canonical baseline roster",
    )
    row = _overlay_mapping(hosts.get(host), label=f"baseline payload {host}")
    canonical_path = _expect_sha256(
        Path(str(row["canonical_roster_receipt"])),
        baseline.get("canonical_roster_receipt_sha256"),
        label="overlay canonical baseline roster receipt",
    )
    try:
        canonical_path, canonical = baseline_payload_snapshot.validate_canonical_roster_receipt(
            canonical_path,
            expected_sha256=str(baseline.get("canonical_roster_receipt_sha256") or ""),
            owner_contract_sha256=owner_contract_sha256,
        )
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241LaunchError(
            f"activation overlay canonical baseline roster is invalid: {exc}"
        ) from exc
    if (
        canonical.get("baseline_manifest_sha256")
        != baseline.get("canonical_baseline_manifest_sha256")
        or canonical.get("baseline_roster_sha256")
        != baseline.get("canonical_baseline_roster_sha256")
    ):
        raise R241LaunchError("activation overlay canonical baseline identities drifted")
    staging = _expect_sha256(
        str(row["staging_receipt"]),
        row.get("staging_receipt_sha256"),
        label=f"overlay {host} baseline staging receipt",
    )
    _, receipt = _json_object(staging, label=f"overlay {host} baseline staging receipt")
    binding = _overlay_mapping(
        receipt.get("baseline_payload_snapshot"), label="baseline staging"
    )
    expected = {
        "schema": baseline_payload_snapshot.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": host,
        "root": row["root"],
        "manifest": row["manifest"],
        "manifest_sha256": row["manifest_sha256"],
        "baseline_tree_sha256": row["baseline_tree_sha256"],
        "baseline_manifest_sha256": canonical.get("baseline_manifest_sha256"),
        "baseline_roster_sha256": canonical.get("baseline_roster_sha256"),
        "baseline_roster": canonical.get("baseline_roster"),
        "owner_contract_sha256": owner_contract_sha256,
    }
    canonical_binding = _overlay_mapping(
        receipt.get("canonical_roster_receipt"), label="baseline staging canonical roster"
    )
    if (
        receipt.get("schema") != baseline_payload_snapshot.BASELINE_PAYLOAD_STAGING_SCHEMA
        or receipt.get("revision") != R241_REVISION
        or receipt.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("receipt_outside_source_and_baseline_snapshot") is not True
        or any(binding.get(key) != value for key, value in expected.items())
        or canonical_binding.get("path") != str(canonical_path)
        or canonical_binding.get("sha256")
        != baseline.get("canonical_roster_receipt_sha256")
        or canonical_binding.get("baseline_manifest_sha256")
        != canonical.get("baseline_manifest_sha256")
        or canonical_binding.get("baseline_roster_sha256")
        != canonical.get("baseline_roster_sha256")
        or canonical_binding.get("public_contract_sha256s")
        != canonical.get("public_contract_sha256s")
    ):
        raise R241LaunchError(
            f"activation overlay {host} baseline staging receipt binding drifted"
        )


def _validate_owner_start_authorization(
    overlay: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    host: str,
) -> tuple[Path, str]:
    """Verify the separately issued, single-purpose service-start authority."""

    binding = _overlay_mapping(
        overlay.get("owner_start_authorization"), label="owner start authorization"
    )
    if set(binding) != {
        "schema",
        "sha256",
        "byte_identical_mirrors_required",
        "hosts",
    }:
        raise R241LaunchError(
            "activation overlay owner-start authorization has an unsupported shape"
        )
    if (
        binding.get("schema") != OWNER_START_AUTHORIZATION_SCHEMA
        or binding.get("byte_identical_mirrors_required") is not True
    ):
        raise R241LaunchError(
            "activation overlay owner-start authorization lacks its shared-mirror contract"
        )
    authorization_sha256 = _valid_sha256(
        binding.get("sha256"), label="activation overlay owner-start authorization"
    )
    authorization_hosts = _overlay_mapping(
        binding.get("hosts"), label="owner-start authorization hosts"
    )
    if set(authorization_hosts) != {"inzi", "elmo"}:
        raise R241LaunchError(
            "activation overlay owner-start authorization must bind Inzi and Elmo"
        )
    host_binding = _overlay_mapping(
        authorization_hosts.get(host), label=f"owner-start authorization {host}"
    )
    if set(host_binding) != {"path"}:
        raise R241LaunchError(
            "activation overlay owner-start authorization host binding must contain only path"
        )
    source = _overlay_mapping(registry.get("source_snapshot"), label="source snapshot")
    host_row = _overlay_mapping(
        _overlay_mapping(source.get("hosts"), label="source snapshot hosts").get(host),
        label=f"source snapshot {host}",
    )
    outputs_root = _real_directory(
        str(host_row.get("outputs_root") or ""),
        label="r241 external outputs root",
    )
    receipt_path = _readonly_external_file(
        _expect_sha256(
            Path(str(host_binding.get("path") or "")),
            authorization_sha256,
            label="r241 owner-start authorization receipt",
        ),
        outputs_root=outputs_root,
        label="r241 owner-start authorization receipt",
    )
    _, receipt = _json_object(receipt_path, label="r241 owner-start authorization receipt")
    baseline = _overlay_mapping(registry.get("baseline_payloads"), label="baseline payloads")
    expected = {
        "schema": OWNER_START_AUTHORIZATION_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authorized",
        "authorized": True,
        "owner_contract_sha256": _overlay_mapping(
            registry.get("owner_contract"), label="owner contract"
        ).get("sha256"),
        "allowed_actions": ["managed_r241_training_start"],
        "source_snapshot_manifest_sha256": source.get("manifest_sha256"),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "canonical_baseline_manifest_sha256": baseline.get(
            "canonical_baseline_manifest_sha256"
        ),
        "canonical_baseline_roster_sha256": baseline.get(
            "canonical_baseline_roster_sha256"
        ),
        "submission_boundary": {
            "exact_count": 1,
            "checkpoint_source": "expert_before_iter_00010.pt",
            "intermediate_iteration_5_submission_allowed": False,
            "retry_copy_or_duplicate_allowed": False,
        },
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise R241LaunchError("r241 owner-start authorization receipt binding drifted")
    provenance = _overlay_mapping(
        receipt.get("authorization_provenance"),
        label="r241 owner-start authorization provenance",
    )
    if provenance != {
        "schema": OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA,
        "create_only": True,
        "explicit_operator_intent": "authorize_managed_r241_training_start",
    }:
        raise R241LaunchError(
            "r241 owner-start authorization was not emitted by the explicit create-only generator"
        )
    return receipt_path, authorization_sha256


def _validate_activation_overlay_mirror_receipt(
    *,
    receipt_path: Path | str,
    receipt_sha256: str,
    overlay_path: Path,
    overlay_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    host: str,
    outputs_root: Path,
) -> tuple[Path, str]:
    """Prove that this host installed the one logical overlay byte-for-byte.

    The controller builds the canonical overlay from local copies of host
    receipts.  It must not try to open a different host's native path at
    launch time.  Instead, each host runs the small mirror installer locally;
    this receipt binds the installed bytes, the local authorization mirror,
    and the shared overlay digest before any service command is formed.
    """

    mirror_sha256 = _valid_sha256(
        receipt_sha256, label="r241 activation-overlay mirror receipt"
    )
    mirror_path = _readonly_external_file(
        _expect_sha256(
            receipt_path,
            mirror_sha256,
            label="r241 activation-overlay mirror receipt",
        ),
        outputs_root=outputs_root,
        label="r241 activation-overlay mirror receipt",
    )
    _, receipt = _json_object(
        mirror_path, label="r241 activation-overlay mirror receipt"
    )
    expected = {
        "schema": ACTIVATION_OVERLAY_MIRROR_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "passed",
        "passed": True,
        "host": host,
        "logical_overlay": {
            "path": str(overlay_path),
            "sha256": _valid_sha256(
                overlay_sha256, label="r241 activation overlay"
            ),
        },
        "owner_start_authorization": {
            "path": str(authorization_path),
            "sha256": _valid_sha256(
                authorization_sha256,
                label="r241 owner-start authorization",
            ),
        },
        "outputs_root": str(outputs_root),
        "byte_identical_copy_verified": True,
    }
    if receipt != expected:
        raise R241LaunchError(
            "r241 activation-overlay mirror receipt does not bind this host-local byte-identical install"
        )
    return mirror_path, mirror_sha256


def apply_activation_overlay(
    registry: Mapping[str, Any],
    *,
    registry_path: Path,
    overlay_path: Path | str,
    overlay_sha256: str,
    overlay_mirror_receipt: Path | str,
    overlay_mirror_receipt_sha256: str,
    host: str,
) -> tuple[dict[str, Any], ActivationOverlayContext]:
    """Merge the only authorized dynamic cells into a pending snapshot registry.

    The overlay is not a second registry.  It is a create-only receipt-backed
    transaction whose base checksum pins the exact pending file inside the
    immutable source tree.  Any attempt to replace unrelated source settings,
    including deck, schedule, selector, or public roster, fails before a
    command is built.
    """

    expected_registry = _bound_source_path(
        ROOT,
        "state/alakazam-new-list-direct-r241-runtime-registry.json",
        label="r241 snapshot runtime registry",
    ).resolve()
    if registry_path.resolve() != expected_registry:
        raise R241LaunchError(
            "r241 activation must use the pending runtime registry inside the verified source snapshot"
        )
    _activation_overlay_base_projection_is_pending(registry)
    overlay_file = _expect_sha256(
        overlay_path, overlay_sha256, label="r241 external activation overlay"
    )
    _, overlay = _json_object(overlay_file, label="r241 external activation overlay")
    owner = _overlay_mapping(registry.get("owner_contract"), label="owner contract")
    base = _overlay_mapping(overlay.get("base_registry"), label="base registry")
    if (
        overlay.get("schema") != ACTIVATION_OVERLAY_SCHEMA
        or overlay.get("revision") != R241_REVISION
        or overlay.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or overlay.get("status") != "ready"
        or overlay.get("passed") is not True
        or overlay.get("owner_contract_sha256") != owner.get("sha256")
        or base.get("path")
        != "state/alakazam-new-list-direct-r241-runtime-registry.json"
        or base.get("sha256") != _sha256(registry_path)
    ):
        raise R241LaunchError("r241 activation overlay does not bind this pending source registry")
    mirrors = _overlay_mapping(overlay.get("mirrors"), label="overlay mirrors")
    if mirrors != {
        "schema": "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v1",
        "hosts": ["inzi", "elmo"],
        "byte_identical_required": True,
    }:
        raise R241LaunchError(
            "r241 activation overlay is not the required shared Inzi/Elmo logical mirror"
        )
    source_overlay = _overlay_mapping(overlay.get("source_snapshot"), label="source snapshot")
    baseline_overlay = _overlay_mapping(
        overlay.get("baseline_payloads"), label="baseline payloads"
    )
    _validate_overlay_source_staging(
        source_overlay, owner_contract_sha256=str(owner["sha256"]), host=host
    )
    _validate_overlay_worker_image(
        _overlay_mapping(overlay.get("worker_image"), label="worker image"),
        owner_contract_sha256=str(owner["sha256"]),
        source_snapshot=source_overlay,
    )
    _validate_overlay_baseline_staging(
        baseline_overlay, owner_contract_sha256=str(owner["sha256"]), host=host
    )
    preservation_overlay = _overlay_mapping(
        overlay.get("peak_r195_preservation"), label="peak-r195 preservation"
    )
    if set(preservation_overlay) != {"receipt_sha256_inzi", "receipt_sha256_elmo"}:
        raise R241LaunchError("activation overlay may only replace preservation receipt hashes")
    for name, digest in preservation_overlay.items():
        _valid_sha256(digest, label=f"activation overlay preservation {name}")
    remote_overlay = _overlay_mapping(
        overlay.get("remote_collection"), label="remote collection"
    )
    if set(remote_overlay) != {
        "endpoint_id",
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
        "checkpoint_transport",
    }:
        raise R241LaunchError("activation overlay may only replace remote receipt hashes")
    if remote_overlay.get("endpoint_id") != R241_ELMO_ENDPOINT_ID:
        raise R241LaunchError("activation overlay names an ineligible remote endpoint")
    for name in (
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
    ):
        _valid_sha256(remote_overlay.get(name), label=f"activation overlay remote {name}")
    transport_overlay = _overlay_mapping(
        remote_overlay.get("checkpoint_transport"), label="remote checkpoint transport"
    )
    if transport_overlay.get("status") != "ready":
        raise R241LaunchError(
            "activation overlay must contain a receipt-bound ready checkpoint transport"
        )
    _validate_remote_checkpoint_transport_contract(
        {"checkpoint_transport": transport_overlay}
    )

    merged = json.loads(json.dumps(registry))
    merged["source_snapshot"] = source_overlay
    merged["baseline_payloads"] = baseline_overlay
    merged_preservation = _overlay_mapping(
        merged.get("peak_r195_preservation"), label="merged preservation"
    )
    merged_preservation.update(preservation_overlay)
    merged["peak_r195_preservation"] = merged_preservation
    merged_remote = _overlay_mapping(merged.get("remote_collection"), label="merged remote")
    endpoints = list(merged_remote.get("eligible_endpoints") or [])
    if len(endpoints) != 1 or not isinstance(endpoints[0], Mapping):
        raise R241LaunchError("pending r241 source registry remote endpoint drifted")
    endpoint = dict(endpoints[0])
    for name in (
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
    ):
        endpoint[name] = remote_overlay[name]
    merged_remote["eligible_endpoints"] = [endpoint]
    merged_remote["checkpoint_transport"] = transport_overlay
    merged["remote_collection"] = merged_remote

    # Static validation runs after the controlled merge.  It proves that the
    # overlay did not alter deck, guide, policy, cycle, source owner, or any
    # other immutable source intent.
    validate_static_registry(merged)
    authorization_path, authorization_sha256 = _validate_owner_start_authorization(
        overlay, registry=merged, host=host
    )
    source_host = _overlay_mapping(
        _overlay_mapping(source_overlay.get("hosts"), label="source snapshot hosts").get(host),
        label=f"source snapshot {host}",
    )
    outputs_root = _real_directory(
        str(source_host.get("outputs_root") or ""), label="r241 external outputs root"
    )
    checked_overlay = _readonly_external_file(
        overlay_file,
        outputs_root=outputs_root,
        label="r241 external activation overlay",
    )
    mirror_receipt, mirror_receipt_sha256 = _validate_activation_overlay_mirror_receipt(
        receipt_path=overlay_mirror_receipt,
        receipt_sha256=overlay_mirror_receipt_sha256,
        overlay_path=checked_overlay,
        overlay_sha256=overlay_sha256,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        host=host,
        outputs_root=outputs_root,
    )
    return merged, ActivationOverlayContext(
        path=checked_overlay,
        sha256=_valid_sha256(overlay_sha256, label="r241 activation overlay"),
        authorization_receipt=authorization_path,
        authorization_receipt_sha256=authorization_sha256,
        mirror_receipt=mirror_receipt,
        mirror_receipt_sha256=mirror_receipt_sha256,
    )


def planned_collection_group_counts(
    *,
    games_per_iteration: int,
    self_play_fraction: float,
    strong_public_fraction_of_public: float,
    research_control_games: int,
) -> dict[str, int]:
    """Mirror ``train_pure_rl._planned_collection_group_counts`` exactly.

    Keeping this dependency-light mirror in the launcher lets a production
    preflight assert the number that the trainer will actually schedule,
    without importing Torch or attempting a collection.
    """

    total = int(games_per_iteration)
    if total <= 0:
        raise R241LaunchError("games per iteration must be positive")
    self_fraction = min(1.0, max(0.0, float(self_play_fraction)))
    practice_fraction = min(
        1.0, max(0.0, float(strong_public_fraction_of_public))
    )
    self_games = int(round(total * self_fraction))
    if self_fraction > 0.0 and self_games == 0:
        self_games = 1
    if self_fraction < 1.0 and self_games == total:
        self_games = max(0, total - 1)
    public_games = total - self_games
    base_practice = int(round(public_games * practice_fraction))
    reclaimed = int(research_control_games)
    if reclaimed < 0 or reclaimed > public_games - base_practice:
        raise R241LaunchError(
            "research-control reclaim does not fit the fixed training budget: "
            f"total={total} self_play={self_games} "
            f"base_strong_public={base_practice} reclaimed={reclaimed}"
        )
    strong_public = base_practice + reclaimed
    result = {
        "self_play": self_games,
        "strong_public_practice": strong_public,
        "diverse_public": public_games - strong_public,
    }
    if sum(result.values()) != total:
        raise R241LaunchError("collection-group plan does not conserve games")
    return result


def _validate_exact_deck(registry: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    deck = dict(_require(registry, "deck", label="registry"))
    source_deck = dict(_require(source, "exact_deck", label="owner contract"))
    path = ROOT / str(_require(deck, "path", label="registry deck"))
    if deck.get("path") != source_deck.get("path"):
        raise R241LaunchError("registry deck path drifted from owner contract")
    if deck.get("sha256") != source_deck.get("file_sha256"):
        raise R241LaunchError("registry deck checksum drifted from owner contract")
    if deck.get("multiset_sha256") != source_deck.get("canonical_multiset_sha256"):
        raise R241LaunchError("registry deck multiset drifted from owner contract")
    if _exact_int(deck.get("card_count"), label="registry deck card_count") != 60:
        raise R241LaunchError("r241 deck must contain exactly 60 cards")
    _expect_sha256(path, deck["sha256"], label="r241 exact deck")
    try:
        cards = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError) as exc:
        raise R241LaunchError("r241 deck is not an integer card-id CSV") from exc
    if len(cards) != 60:
        raise R241LaunchError(f"r241 deck has {len(cards)} cards instead of 60")
    multiset = hashlib.sha256(
        json.dumps(sorted(cards), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if f"sha256:{multiset}" != deck["multiset_sha256"]:
        raise R241LaunchError("r241 deck multiset checksum mismatch")


def _validate_guide(registry: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    guide = dict(_require(registry, "guide", label="registry"))
    owner = dict(_require(source, "owner_guide", label="owner contract"))
    bindings = (
        ("contract", "guide_contract", "contract_sha256", "guide_contract_sha256"),
        ("curriculum_spec", None, "curriculum_spec_sha256", None),
        ("head_role_map", None, "head_role_map_sha256", None),
        ("validation_receipt", None, "validation_receipt_sha256", None),
    )
    for registry_path_key, owner_path_key, registry_hash_key, owner_hash_key in bindings:
        local_path = ROOT / str(_require(guide, registry_path_key, label="registry guide"))
        _expect_sha256(
            local_path,
            _require(guide, registry_hash_key, label="registry guide"),
            label=f"r241 guide {registry_path_key}",
        )
        if owner_path_key is not None:
            if guide.get(registry_path_key) != owner.get(owner_path_key):
                raise R241LaunchError(
                    f"registry guide {registry_path_key} drifted from owner contract"
                )
            if guide.get(registry_hash_key) != owner.get(owner_hash_key):
                raise R241LaunchError(
                    f"registry guide {registry_hash_key} drifted from owner contract"
                )
    if (
        guide.get("selector") != "alakazam"
        or guide.get("version") != owner.get("guide_version")
        or float(guide.get("ordinary_rl_loss_weight", -1.0)) != 0.05
        or float(guide.get("expert_refresh_loss_weight", -1.0)) != 0.0
        or guide.get("training_mode") != "strategic_directional_v2"
    ):
        raise R241LaunchError("r241 guide training-only contract drifted")


def _validate_expert_window_staging(registry: Mapping[str, Any]) -> None:
    window = dict(_require(registry, "expert_window", label="registry"))
    staging = ROOT / str(_require(window, "staging_receipt", label="expert window"))
    _expect_sha256(
        staging,
        _require(window, "staging_receipt_sha256", label="expert window"),
        label="r241 exact expert-window staging receipt",
    )
    _, staged = _json_object(staging, label="r241 exact expert-window staging receipt")
    if (
        staged.get("schema") != EXPERT_WINDOW_STAGING_SCHEMA
        or staged.get("status") != "ready"
    ):
        raise R241LaunchError("exact expert-window staging receipt is not ready")
    staged_window = dict(staged.get("window") or {})
    if (
        staged_window.get("start") != EXACT_WINDOW_START
        or staged_window.get("end") != EXACT_WINDOW_END
        or _exact_int(staged_window.get("days"), label="staged expert days")
        != EXACT_WINDOW_DAYS
        or _exact_int(staged_window.get("validated_episodes"), label="staged expert episodes")
        != 91_253
    ):
        raise R241LaunchError("staged expert receipt is not the exact 2026-07-22..08-10 window")
    canonical = dict(staged.get("canonical_receipt") or {})
    declared = dict(window.get("canonical_receipt") or {})
    if (
        canonical.get("sha256") != declared.get("sha256")
        or canonical.get("schema") != EXPERT_RECEIPT_SCHEMA
        or canonical.get("status") != "ready"
        or canonical.get("inzi_path") != declared.get("inzi_path")
        or canonical.get("elmo_path") != declared.get("elmo_path")
    ):
        raise R241LaunchError("expert-window canonical receipt binding drifted")
    latest = dict(staged.get("latest_day") or {})
    if (
        latest.get("date") != EXACT_WINDOW_END
        or latest.get("archive_sha256")
        != "sha256:2167f90b2e2c769dec3c94f251ff704aaef55a08956fc8c37e596b3926aa5f59"
        or _exact_int(latest.get("validated_episode_count"), label="Aug-10 episode count")
        != 4_603
    ):
        raise R241LaunchError("exact August 10 expert source binding is invalid")


def _validate_official_libcg_staging(registry: Mapping[str, Any]) -> None:
    declared = dict(_require(registry, "official_libcg", label="registry"))
    staging = ROOT / str(_require(declared, "staging_receipt", label="official libcg"))
    _expect_sha256(
        staging,
        _require(declared, "staging_receipt_sha256", label="official libcg"),
        label="r241 official-libcg staging receipt",
    )
    _, staged = _json_object(staging, label="r241 official-libcg staging receipt")
    if staged.get("schema") != "poke_bot.alakazam_new_list_direct_r241_official_libcg_staging/v1":
        raise R241LaunchError("official libcg staging receipt schema changed")
    required = dict(staged.get("required_runtime") or {})
    if (
        required.get("environment") != "CG_LIB_PATH"
        or required.get("member") != "cg/libcg.so"
        or required.get("member_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
        or _exact_int(required.get("member_size_bytes"), label="official libcg size")
        != 1_342_400
        or required.get("forbidden_environment_absent")
        != ["POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"]
    ):
        raise R241LaunchError("official r236 libcg staging identity drifted")
    declared_hosts = dict(_require(declared, "hosts", label="official libcg"))
    staged_hosts = dict(staged.get("hosts") or {})
    for host in ("inzi", "elmo"):
        expected = dict(_require(declared_hosts, host, label="official libcg hosts"))
        actual = dict(_require(staged_hosts, host, label="official staging hosts"))
        if (
            actual.get("runtime_root") != expected.get("runtime_root")
            or actual.get("receipt_sha256") != expected.get("receipt_sha256")
            or actual.get("loaded_member_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
            or actual.get("passed") is not True
        ):
            raise R241LaunchError(f"official r236 libcg host binding drifted: {host}")


def _validate_source_snapshot_contract(registry: Mapping[str, Any]) -> None:
    """Check the staged source-snapshot authority without touching hosts.

    A pending registry may intentionally leave its content address blank while
    other owner gates are false.  It is never launchable in that state: the
    activation validator below requires a ready exact address plus a local
    manifest.  This keeps staging side-effect-free without admitting the
    mutable checkout as an implicit fallback.
    """

    source = dict(_require(registry, "source_snapshot", label="registry"))
    owner = dict(_require(registry, "owner_contract", label="registry"))
    if (
        source.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or source.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or source.get("owner_contract_sha256") != owner.get("sha256")
        or source.get("status")
        not in {"pending_immutable_source_snapshot", "ready"}
    ):
        raise R241LaunchError("r241 source-snapshot registry contract is invalid")
    hosts = dict(_require(source, "hosts", label="source snapshot"))
    if set(hosts) != {"inzi", "elmo"}:
        raise R241LaunchError("source snapshot must bind exactly Inzi and Elmo")
    pending = source.get("status") == "pending_immutable_source_snapshot"
    for host in ("inzi", "elmo"):
        row = dict(_require(hosts, host, label="source snapshot hosts"))
        outputs_root = str(row.get("outputs_root") or "").strip()
        if not outputs_root or not Path(outputs_root).is_absolute():
            raise R241LaunchError(
                f"r241 source snapshot {host} lacks an external outputs root"
            )
        root = str(row.get("root") or "").strip()
        manifest = str(row.get("manifest") or "").strip()
        if pending:
            if root or manifest:
                raise R241LaunchError(
                    f"pending r241 source snapshot {host} may not name mutable code"
                )
            continue
        if (
            not root
            or not manifest
            or not Path(root).is_absolute()
            or not Path(manifest).is_absolute()
        ):
            raise R241LaunchError(
                f"ready r241 source snapshot {host} lacks root/manifest paths"
            )
    if pending:
        if str(source.get("manifest_sha256") or "") or str(
            source.get("source_tree_sha256") or ""
        ):
            raise R241LaunchError(
                "pending r241 source snapshot cannot mix empty and bound identities"
            )
    else:
        _valid_sha256(source.get("manifest_sha256"), label="source manifest")
        _valid_sha256(source.get("source_tree_sha256"), label="source tree")


def _validate_baseline_payload_contract(registry: Mapping[str, Any]) -> None:
    """Validate the separate immutable baseline mount projection.

    The code snapshot intentionally excludes ``baselines/``.  This contract
    therefore cannot fall back to a checkout-relative library: activation must
    later provide a receipt-bound, host-absolute mount through the external
    overlay.  Pending static state remains snapshot-safe and contains no path
    that could be mistaken for a mutable fallback.
    """

    payload = dict(_require(registry, "baseline_payloads", label="registry"))
    if (
        payload.get("schema") != BASELINE_PAYLOAD_REGISTRY_SCHEMA
        or payload.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or payload.get("separately_mounted_and_receipted") is not True
        or payload.get("source_snapshot_fallback_allowed") is not False
        or payload.get("status")
        not in {"pending_external_baseline_payload_snapshot", "ready"}
    ):
        raise R241LaunchError("r241 separate baseline payload contract is invalid")
    hosts = dict(_require(payload, "hosts", label="baseline payloads"))
    if set(hosts) != {"inzi", "elmo"}:
        raise R241LaunchError("r241 baseline payloads must bind exactly Inzi and Elmo")
    pending = payload.get("status") == "pending_external_baseline_payload_snapshot"
    canonical_identity_keys = (
        "canonical_roster_receipt_sha256",
        "canonical_baseline_manifest_sha256",
        "canonical_baseline_roster_sha256",
    )
    if pending:
        if (
            str(payload.get("canonical_roster_receipt") or "").strip()
            or any(str(payload.get(key) or "").strip() for key in canonical_identity_keys)
        ):
            raise R241LaunchError(
                "pending r241 baseline payload may not pre-bind a local canonical roster"
            )
    else:
        # A shared overlay cannot truthfully carry one unscoped path because
        # Inzi and Elmo mount their canonical receipts at different locations.
        # The exact paths live in the corresponding host rows below.
        if "canonical_roster_receipt" in payload:
            raise R241LaunchError(
                "ready r241 baseline payload may not carry an unscoped canonical-roster path"
            )
        _valid_sha256(
            payload.get("canonical_roster_receipt_sha256"),
            label="canonical baseline roster receipt",
        )
        _valid_sha256(
            payload.get("canonical_baseline_manifest_sha256"),
            label="canonical baseline manifest",
        )
        _valid_sha256(
            payload.get("canonical_baseline_roster_sha256"),
            label="canonical baseline roster",
        )
    canonical_receipts: dict[str, str] = {}
    for host in ("inzi", "elmo"):
        row = dict(_require(hosts, host, label="baseline payload hosts"))
        keys = (
            "root",
            "manifest",
            "manifest_sha256",
            "baseline_tree_sha256",
            "staging_receipt",
            "staging_receipt_sha256",
        )
        if pending:
            if any(str(row.get(key) or "").strip() for key in keys):
                raise R241LaunchError(
                    f"pending r241 baseline payload {host} may not name a fallback"
                )
            continue
        root = str(row.get("root") or "").strip()
        manifest = str(row.get("manifest") or "").strip()
        receipt = str(row.get("staging_receipt") or "").strip()
        canonical_receipt = str(row.get("canonical_roster_receipt") or "").strip()
        if (
            not root
            or not manifest
            or not receipt
            or not canonical_receipt
            or not Path(root).is_absolute()
            or not Path(manifest).is_absolute()
            or not Path(receipt).is_absolute()
            or not Path(canonical_receipt).is_absolute()
        ):
            raise R241LaunchError(
                f"ready r241 baseline payload {host} lacks host-absolute bindings"
            )
        _valid_sha256(row.get("manifest_sha256"), label=f"{host} baseline manifest")
        _valid_sha256(row.get("baseline_tree_sha256"), label=f"{host} baseline tree")
        _valid_sha256(
            row.get("staging_receipt_sha256"), label=f"{host} baseline staging receipt"
        )
        canonical_receipts[host] = canonical_receipt
    if not pending and canonical_receipts["inzi"] == canonical_receipts["elmo"]:
        raise R241LaunchError(
            "ready r241 baseline payload canonical-roster paths must remain host-scoped"
        )


def _validate_remote_checkpoint_transport_contract(
    remote: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one digest-addressed checkpoint route for Elmo :8767.

    The snapshot-local registry intentionally has no host path or receipt yet.
    The external overlay promotes this row atomically with the matching Elmo
    receipt.  Keeping it registry-level makes every generic ``RemoteJobClient``
    use the same transport rather than silently recovering the historical
    :8765/SMB defaults.
    """

    transport = dict(
        _require(remote, "checkpoint_transport", label="remote collection")
    )
    expected = {
        "schema": R241_CHECKPOINT_TRANSPORT_SCHEMA,
        "endpoint_id": R241_ELMO_ENDPOINT_ID,
        "host_role": "elmo",
        "verification_endpoint": R241_ELMO_ENDPOINT,
        "verification_port": 8767,
        "container_root": R241_ELMO_CHECKPOINT_TRANSPORT_CONTAINER_ROOT,
        "environment_key": R241_ELMO_CHECKPOINT_TRANSPORT_ENV,
        "remote_path_prefix": f"{R241_ELMO_CHECKPOINT_TRANSPORT_CONTAINER_ROOT}/",
        "content_addressing": {
            "algorithm": "sha256",
            "filename_scheme": R241_ELMO_CHECKPOINT_TRANSPORT_FILENAME_SCHEME,
        },
        "read_only_container_mount": True,
        "same_absolute_source_and_baseline_paths_preserved": True,
    }
    if any(transport.get(key) != value for key, value in expected.items()):
        raise R241LaunchError("r241 Elmo checkpoint transport identity drifted")
    initial = _overlay_mapping(
        transport.get("initial_checkpoint"), label="checkpoint transport initial checkpoint"
    )
    if set(initial) != {"container_path", "sha256"}:
        raise R241LaunchError(
            "r241 checkpoint transport initial checkpoint has an unsupported shape"
        )
    status = transport.get("status")
    dynamic_fields = (
        "host_root",
        "trainer_visible_root",
        "staging_receipt",
        "staging_receipt_sha256",
    )
    if status == "pending_external_checkpoint_transport":
        if any(str(transport.get(key) or "").strip() for key in dynamic_fields) or any(
            str(initial.get(key) or "").strip() for key in initial
        ):
            raise R241LaunchError(
                "pending r241 checkpoint transport may not name a host path or receipt"
            )
        return transport
    if status != "ready":
        raise R241LaunchError("r241 checkpoint transport is neither pending nor ready")
    host_root = str(transport.get("host_root") or "").strip()
    trainer_root = str(transport.get("trainer_visible_root") or "").strip()
    staging_receipt = str(transport.get("staging_receipt") or "").strip()
    if (
        not host_root
        or not trainer_root
        or not staging_receipt
        or not Path(host_root).is_absolute()
        or not Path(trainer_root).is_absolute()
        or not Path(staging_receipt).is_absolute()
        or host_root == "/workspace"
        or host_root.startswith("/workspace/")
    ):
        raise R241LaunchError(
            "ready r241 checkpoint transport lacks sealed Elmo/trainer host roots"
        )
    _valid_sha256(
        transport.get("staging_receipt_sha256"),
        label="r241 checkpoint transport staging receipt",
    )
    checkpoint_path = str(initial.get("container_path") or "").strip()
    candidate = Path(checkpoint_path)
    if (
        not candidate.is_absolute()
        or candidate.parent != Path(R241_ELMO_CHECKPOINT_TRANSPORT_CONTAINER_ROOT)
    ):
        raise R241LaunchError(
            "r241 checkpoint transport initial checkpoint is outside /workspace/checkpoint"
        )
    _valid_sha256(initial.get("sha256"), label="r241 checkpoint transport initial checkpoint")
    return transport


def _remote_checkpoint_transport(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated registry-level Elmo checkpoint transport."""

    remote = dict(_require(registry, "remote_collection", label="registry"))
    _validate_remote_collection_contract(registry)
    return _validate_remote_checkpoint_transport_contract(remote)


def _validate_remote_collection_contract(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the one staged remote endpoint without connecting to it.

    r241 deliberately cannot inherit the generic fleet default, because that
    would re-admit Elmo :8765 or Bert :8766.  Empty receipt hashes are allowed
    only while the registry is staged; activation requires all four exact
    immutable receipt identities.
    """

    remote = dict(_require(registry, "remote_collection", label="registry"))
    if (
        remote.get("schema") != REMOTE_ENDPOINT_REGISTRY_SCHEMA
        or _exact_int(remote.get("revision"), label="remote registry revision")
        != R241_REVISION
        or remote.get("require_explicit_eligible_endpoints") is not True
        or remote.get("legacy_or_default_endpoint_fallback_allowed") is not False
        or remote.get("pure_rl_public_mix_local_only") != "0"
    ):
        raise R241LaunchError("r241 remote collection registry contract drifted")
    endpoints = remote.get("eligible_endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 1:
        raise R241LaunchError("r241 must bind exactly one eligible remote endpoint")
    endpoint = dict(endpoints[0]) if isinstance(endpoints[0], Mapping) else {}
    if (
        endpoint.get("id") != R241_ELMO_ENDPOINT_ID
        or endpoint.get("endpoint") != R241_ELMO_ENDPOINT
        or endpoint.get("host_role") != "elmo"
        or endpoint.get("capability") != R241_ELMO_COLLECTION_CAPABILITY
        or endpoint.get("allowed_job_kinds") != list(R241_REMOTE_ALLOWED_JOB_KINDS)
        or endpoint.get("r236_linux_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
        or endpoint.get("learner_matchup_tree_sha256")
        != "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
        or endpoint.get("h10_matchup_tree_sha256")
        != "sha256:da223c4903dd37511e5cb7656fe405bc0baac085be4f131faef136b7056c4588"
        or endpoint.get("endpoint") in R241_ELMO_LEGACY_ENDPOINTS
    ):
        raise R241LaunchError("r241 remote endpoint identity is not the isolated Elmo :8767 worker")
    expected_paths = {
        "manifest_path": "preflight-manifest.json",
        "host_receipt_path": "host-preflight.json",
        "runtime_receipt_path": "runtime-preflight.json",
        "gameplay_receipt_path": "gameplay-preflight.json",
    }
    endpoint_parent: Path | None = None
    for key, filename in expected_paths.items():
        rendered = str(endpoint.get(key) or "").strip()
        candidate = Path(rendered)
        if not rendered or not candidate.is_absolute() or candidate.name != filename:
            raise R241LaunchError(f"r241 remote endpoint has an invalid {key}")
        if endpoint_parent is None:
            endpoint_parent = candidate.parent
        elif candidate.parent != endpoint_parent:
            raise R241LaunchError("r241 remote receipts must share one isolated directory")
    expected_remote_dir = Path(
        "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/elmo-8767"
    )
    if endpoint_parent != expected_remote_dir:
        raise R241LaunchError("r241 remote endpoint receipt directory drifted")
    for key in (
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
    ):
        rendered = str(endpoint.get(key) or "")
        if rendered:
            _valid_sha256(rendered, label=f"r241 remote {key}")
    _validate_remote_checkpoint_transport_contract(remote)
    return endpoint


def _validate_no_slot_change_contract(
    registry: Mapping[str, Any], source: Mapping[str, Any],
) -> None:
    """Bind the preserved r195 adapter roster without a PTCG launch gate.

    r248 deliberately defers all external archetype refresh work.  r241 only
    needs to prove that its existing E60/slots 0..19 bank remains unchanged;
    it must not require, name, or interpret any external snapshot, migration
    file, or future-allocation policy.  Optional metadata can
    remain in a registry for later work, but it has no authority here.
    """

    preservation = dict(_require(registry, "peak_r195_preservation", label="registry"))
    roster_relative = str(
        _require(preservation, "baseline_adapter_roster", label="r195 preservation")
    )
    roster_sha256 = _require(
        preservation, "baseline_adapter_roster_sha256", label="r195 preservation"
    )
    roster_path = _bound_source_path(
        ROOT,
        roster_relative,
        label="r241 baseline adapter roster source path",
    )
    _expect_sha256(
        roster_path,
        roster_sha256,
        label="r241 immutable 0..19 adapter roster",
    )
    if _exact_int(
        preservation.get("immutable_adapter_slot_prefix"),
        label="r241 immutable adapter-slot prefix",
    ) != 20:
        raise R241LaunchError("r241 must preserve exactly the existing 0..19 adapter bank")

    # The typed source owns the current-cycle no-slot-change decision.  Do not
    # read its external-source labels or activation-gate aliases: they are
    # intentionally inert for r241.
    owner_adapter = source.get("matchup_adapter_archetype_refresh")
    if isinstance(owner_adapter, Mapping) and (
        owner_adapter.get("current_cycle_required_slot_migration_status")
        != "no_slot_change"
        or owner_adapter.get("new_archetype_slots") != []
        or owner_adapter.get("baseline_slot_registry") != roster_relative
        or owner_adapter.get("baseline_slot_registry_sha256") != roster_sha256
    ):
        raise R241LaunchError("r248 owner contract did not preserve no_slot_change")

    # This projection is optional and deliberately shallow.  It can state the
    # current no-slot-change fact, but any PTCG or future-policy labels are
    # metadata only and never become an r241 launch prerequisite.
    raw = registry.get("matchup_archetype_refresh")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise R241LaunchError("r241 optional adapter-slot metadata must be an object")
    refresh = dict(raw)
    if refresh.get("required_for_r241_activation") not in (None, False):
        raise R241LaunchError("optional adapter-slot metadata may not add an r241 activation gate")
    if refresh.get("new_slots") not in (None, []):
        raise R241LaunchError("r241 adapter slot metadata may not allocate new slots")
    if refresh.get("slot_change_status") not in (None, "no_slot_change"):
        raise R241LaunchError("r241 adapter slot metadata must retain no_slot_change")
    if refresh.get("baseline_slot_registry") not in (None, roster_relative):
        raise R241LaunchError("r241 adapter slot metadata changed the preserved roster")
    if refresh.get("baseline_slot_registry_sha256") not in (None, roster_sha256):
        raise R241LaunchError("r241 adapter slot metadata changed the roster checksum")


def _validate_scoped_direct_policy_contract(
    source: Mapping[str, Any], direct: Mapping[str, Any]
) -> None:
    """Keep direct-only restrictions scoped without mutating frozen opponents.

    r251 intentionally narrows the r241 prohibition to the learner, pinned H10
    Marnie, target generation, and terminal package/submission.  The frozen
    non-H10 public packages remain their historical opponents, so this helper
    rejects both a widened public selector firewall and any attempt to relax a
    direct role into a search-capable one.
    """

    source_exclusion = dict(
        _require(source, "search_and_planning_exclusion", label="owner contract")
    )
    exclusion_scope = dict(
        _require(source_exclusion, "scope", label="owner search scope")
    )
    expected_scope = {
        "learner": "direct_policy_only",
        "pinned_h10_marnie_opponent": "direct_policy_only",
        "target_generation": "direct_policy_only",
        "terminal_package_and_submission": "direct_policy_only",
        "frozen_non_h10_diverse_public_opponent_packages_and_selectors": (
            "preserve_unchanged_per_r245"
        ),
    }
    if (
        exclusion_scope != expected_scope
        or source_exclusion.get("mcts") != "forbidden_for_scoped_direct_roles"
        or source_exclusion.get("recursive_turn_planner")
        != "forbidden_for_scoped_direct_roles"
        or source_exclusion.get("search_target_generation")
        != "forbidden_for_scoped_direct_roles"
        or source_exclusion.get("public_opponent_selector_change") != "forbidden"
        or source_exclusion.get("public_search_firewall") != "not_introduced"
    ):
        raise R241LaunchError(
            "r251 direct-policy scope must preserve frozen non-H10 public selectors"
        )
    if (
        direct.get("action_selector") != "direct_policy_only"
        or direct.get("mcts") != "forbidden_for_scoped_direct_roles"
        or direct.get("recursive_turn_planner")
        != "forbidden_for_scoped_direct_roles"
        or direct.get("search_target_generation")
        != "forbidden_for_scoped_direct_roles"
        or direct.get("public_opponent_selector_change") != "forbidden"
        or direct.get("public_search_firewall") != "not_introduced"
        or direct.get("package_main_import_allowed") is not False
        or direct.get("embedded_cg_runtime_allowed") is not False
        or direct.get("matchup_adapters_allowed_and_required") is not True
    ):
        raise R241LaunchError(
            "r241 scoped direct-policy / Matchup Adapter contract drifted"
        )


def validate_static_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate repository-owned, no-host-I/O r241 invariants."""

    owner = dict(_require(registry, "owner_contract", label="registry"))
    source_path = ROOT / str(_require(owner, "path", label="owner contract"))
    _expect_sha256(
        source_path,
        _require(owner, "sha256", label="owner contract"),
        label="canonical r241 owner contract",
    )
    _, source = _json_object(source_path, label="canonical r241 owner contract")
    if (
        source.get("schema") != "poke_bot.alakazam_new_list_direct_policy_r241/v1"
        or _exact_int(source.get("owner_decision_revision"), label="owner revision")
        != R241_REVISION
        or _exact_int(registry.get("owner_clarification_revision"), label="registry owner clarification")
        != _exact_int(source.get("latest_owner_clarification_revision"), label="owner clarification")
    ):
        raise R241LaunchError("registry does not bind the current r241 owner source")
    source_parent = dict(_require(source, "parent", label="owner contract"))
    r195_source = ROOT / str(_require(source_parent, "typed_source", label="owner parent"))
    _expect_sha256(
        r195_source,
        _require(source_parent, "typed_source_sha256", label="owner parent"),
        label="immutable r195 typed source",
    )
    registry_parent = dict(_require(registry, "parent", label="registry"))
    if (
        registry_parent.get("checkpoint") != source_parent.get("checkpoint")
        or registry_parent.get("sha256") != source_parent.get("checkpoint_sha256")
        or _exact_int(registry_parent.get("size_bytes"), label="registry parent size")
        != _exact_int(source_parent.get("checkpoint_bytes"), label="owner parent size")
        or source_parent.get("immutable") is not True
    ):
        raise R241LaunchError("registry no longer pins the immutable r195 parent")
    _validate_exact_deck(registry, source)
    _validate_guide(registry, source)
    _validate_expert_window_staging(registry)
    _validate_official_libcg_staging(registry)
    _validate_source_snapshot_contract(registry)
    _validate_baseline_payload_contract(registry)
    _validate_remote_collection_contract(registry)
    _validate_no_slot_change_contract(registry, source)

    cycle = dict(_require(registry, "training_cycle", label="registry"))
    source_cycle = dict(_require(source, "training_cycle", label="owner contract"))
    expected_cycle = {
        "updates_exact": 10,
        "games_per_update": EXACT_GAMES_PER_UPDATE,
        "self_play_games_exact": EXACT_SELF_GAMES,
        "public_mix_games_exact": EXACT_PUBLIC_GAMES,
        "marnie_h10_games_minimum": EXACT_H10_MINIMUM,
    }
    for key, value in expected_cycle.items():
        if _exact_int(cycle.get(key), label=f"r241 cycle {key}") != value:
            raise R241LaunchError(f"r241 cycle changed {key}: {cycle.get(key)!r}")
    if (
        cycle.get("commits_exact") != list(range(10))
        or cycle.get("iteration_10_collection_allowed") is not False
        or cycle.get("self_play_fraction") != EXACT_SELF_PLAY_FRACTION
        or cycle.get("expert_refresh_boundaries") != [5, 10]
        or _exact_int(cycle.get("expert_refresh_epochs"), label="expert refresh epochs")
        != 5
        or cycle.get("continue_after_gate_argument_allowed") is not False
        or cycle.get("minimum_terminal_iteration_argument_allowed") is not False
        or cycle.get("population_mode_allowed") is not False
    ):
        raise R241LaunchError("r241 fixed ten-update / refresh contract drifted")
    seats = dict(cycle.get("training_seats") or {})
    if seats != {"first": EXACT_FIRST_SEAT_GAMES, "second": EXACT_SECOND_SEAT_GAMES}:
        raise R241LaunchError("r241 must retain an exact 4098/4098 training split")
    if (
        source_cycle.get("rl_updates_exact") != 10
        or source_cycle.get("games_per_update") != EXACT_GAMES_PER_UPDATE
        or source_cycle.get("self_play_games_exact") != EXACT_SELF_GAMES
        or source_cycle.get("public_mix_games_exact") != EXACT_PUBLIC_GAMES
        or source_cycle.get("marnie_h10_games_minimum") != EXACT_H10_MINIMUM
        or source_cycle.get("marnie_h10_is_minimum_not_exclusive_public_opponent")
        is not True
        or source_cycle.get("established_diverse_public_mix_preserved") is not True
        or source_cycle.get("established_research_control_phase_preserved") is not True
    ):
        raise R241LaunchError("current owner source no longer preserves the public mix")
    static_plan = planned_collection_group_counts(
        games_per_iteration=EXACT_GAMES_PER_UPDATE,
        self_play_fraction=float(EXACT_SELF_PLAY_FRACTION),
        strong_public_fraction_of_public=0.50,
        research_control_games=1_000,
    )
    if static_plan != {
        "self_play": EXACT_SELF_GAMES,
        "strong_public_practice": 4_586,
        "diverse_public": 2_586,
    }:
        raise R241LaunchError(
            f"r241 established public-mix plan changed: {static_plan}"
        )

    preservation = dict(_require(registry, "peak_r195_preservation", label="registry"))
    if preservation.get("receipt_schema") != PRESERVATION_SCHEMA:
        raise R241LaunchError("r195 preservation receipt schema is not checkpoint-derived v2")
    required_preservation = {
        "architecture_present_head_count": 19,
        "every_non_combo_head_trainable": True,
        "every_non_combo_fusion_route_enabled": True,
        "combo_state_head_present": True,
        "combo_state_loss_weight": 0.0,
        "combo_state_fusion_route_enabled": False,
        "matchup_adapter_bank_preserved": True,
        "matchup_adapter_training_enabled": True,
        "matchup_adapter_runtime_enabled": True,
        "matchup_adapter_checkpoint_runtime_enabled": False,
        "matchup_adapter_checkpoint_training_enabled": False,
        "matchup_adapter_checkpoint_main_optimizer_included": False,
        "matchup_adapter_isolated_bank_only_optimizer": True,
        "matchup_adapter_isolated_fit_continuation_required": True,
        "matchup_adapter_external_collection_runtime_enabled": True,
        "matchup_adapter_external_terminal_runtime_enabled": True,
        "learner_matchup_tree_sha256": "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049",
        "baseline_adapter_roster": "state/matchup_adapter_roster.json",
        "baseline_adapter_roster_sha256": "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc",
        "immutable_adapter_slot_prefix": 20,
        "established_diverse_public_mix_preserved": True,
        "research_control_phase_preserved": True,
    }
    for key, expected in required_preservation.items():
        if preservation.get(key) != expected:
            raise R241LaunchError(f"r195 preservation field drifted: {key}")
    _expect_sha256(
        _bound_source_path(
            ROOT,
            preservation["baseline_adapter_roster"],
            label="r195 immutable adapter-roster source path",
        ),
        preservation["baseline_adapter_roster_sha256"],
        label="r195 immutable 0..19 adapter roster",
    )
    direct = dict(_require(registry, "direct_policy", label="registry"))
    _validate_scoped_direct_policy_contract(source, direct)
    run = dict(_require(registry, "run", label="registry"))
    for host in ("inzi", "elmo"):
        run_root = _as_path(
            _require(run, f"{host}_root", label="registry run"),
            label=f"{host} run root",
        )
        expected_adapter_receipt = (
            run_root / "runtime" / R241_H10_ADAPTER_RECEIPT_BASENAME
        )
        observed_adapter_receipt = _as_path(
            _require(
                direct,
                f"adapter_receipt_{host}",
                label="direct-policy adapter",
            ),
            label=f"{host} H10 adapter receipt",
        )
        if observed_adapter_receipt != expected_adapter_receipt:
            raise R241LaunchError(
                "r241 H10 adapter receipt does not use the predeclared "
                f"successor path for {host}"
            )
        expected_preservation_receipt = (
            run_root / "runtime" / R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME
        )
        observed_preservation_receipt = _as_path(
            _require(
                preservation,
                f"receipt_{host}",
                label="r195 preservation receipt",
            ),
            label=f"{host} r195 preservation receipt",
        )
        if observed_preservation_receipt != expected_preservation_receipt:
            raise R241LaunchError(
                "r241 peak-r195 preservation receipt does not use the "
                f"predeclared successor path for {host}"
            )
    marnie = dict(_require(registry, "marnie_h10", label="registry"))
    if (
        marnie.get("opponent_id") != R241_H10_OPPONENT_ID
        or marnie.get("model_sha256") != R241_H10_MODEL_SHA256
        or marnie.get("content_sha256") != R241_H10_CONTENT_SHA256
        or marnie.get("matchup_tree_sha256")
        != "sha256:da223c4903dd37511e5cb7656fe405bc0baac085be4f131faef136b7056c4588"
        or _exact_int(marnie.get("matchup_tree_size_bytes"), label="H10 tree size")
        != 2_509_756
    ):
        raise R241LaunchError("r241 H10 Marnie data-only identity drifted")
    return source


def _host_context(registry: Mapping[str, Any], host: str) -> HostContext:
    host = str(host).strip().lower()
    if host not in {"inzi", "elmo"}:
        raise R241LaunchError("--host must be inzi or elmo")
    run = dict(_require(registry, "run", label="registry"))
    run_root = _as_path(_require(run, f"{host}_root", label="registry run"), label=f"{host} run root")
    libcg = dict(_require(registry, "official_libcg", label="registry"))
    libcg_host = dict(_require(dict(libcg.get("hosts") or {}), host, label="official libcg hosts"))
    direct = dict(_require(registry, "direct_policy", label="registry"))
    preservation = dict(_require(registry, "peak_r195_preservation", label="registry"))
    expert = dict(_require(registry, "expert_window", label="registry"))
    canonical = dict(_require(expert, "canonical_receipt", label="expert window"))
    return HostContext(
        name=host,
        runtime_root=run_root / "runtime",
        official_cg_root=_as_path(
            _require(libcg_host, "runtime_root", label="official libcg host"),
            label=f"{host} official CG root",
        ),
        adapter_receipt=_as_path(
            _require(direct, f"adapter_receipt_{host}", label="direct-policy adapter"),
            label=f"{host} H10 adapter receipt",
        ),
        preservation_receipt=_as_path(
            _require(preservation, f"receipt_{host}", label="r195 preservation"),
            label=f"{host} r195 preservation receipt",
        ),
        expert_archive_receipt=_as_path(
            _require(canonical, f"{host}_path", label="expert canonical receipt"),
            label=f"{host} exact expert archive receipt",
        ),
        expert_manifest_pointer=_as_path(
            _require(expert, f"protected_feature_pointer_{host}", label="expert window"),
            label=f"{host} protected expert manifest pointer",
        ),
    )


def _validate_host_expert_window(
    registry: Mapping[str, Any], context: HostContext
) -> None:
    expert = dict(_require(registry, "expert_window", label="registry"))
    canonical = dict(_require(expert, "canonical_receipt", label="expert window"))
    receipt = _expect_sha256(
        context.expert_archive_receipt,
        _require(canonical, "sha256", label="expert canonical receipt"),
        label="host exact expert archive receipt",
    )
    _, payload = _json_object(receipt, label="host exact expert archive receipt")
    dates = [str(row.get("date") or "") for row in payload.get("archives") or ()]
    if (
        payload.get("schema") != EXPERT_RECEIPT_SCHEMA
        or payload.get("status") != "ready"
        or payload.get("window_policy") != "exact_20_consecutive_calendar_days"
        or payload.get("window_start") != EXACT_WINDOW_START
        or payload.get("window_end") != EXACT_WINDOW_END
        or _exact_int(payload.get("days"), label="host expert days") != EXACT_WINDOW_DAYS
        or payload.get("all_dates_represented") is not True
        or len(dates) != EXACT_WINDOW_DAYS
        or dates != sorted(dates)
        or dates[0] != EXACT_WINDOW_START
        or dates[-1] != EXACT_WINDOW_END
        or _exact_int(payload.get("total_episodes"), label="host expert episodes")
        != 91_253
    ):
        raise R241LaunchError("host exact expert archive receipt is incomplete")


def _validate_protected_expert_pointer(
    pointer: Path,
    *,
    expected_archive_receipt: Path,
) -> None:
    pointer_file, payload = _json_object(pointer, label="r241 protected expert pointer")
    if (
        payload.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or payload.get("protected") is not True
        or not str(payload.get("manifest") or "").strip()
        or not str(payload.get("manifest_sha256") or "").startswith("sha256:")
    ):
        raise R241LaunchError("r241 expert manifest must be a protected pointer")
    raw_manifest = Path(str(payload["manifest"])).expanduser()
    manifest = raw_manifest.resolve() if raw_manifest.is_absolute() else (pointer_file.parent / raw_manifest).resolve()
    _expect_sha256(manifest, payload["manifest_sha256"], label="r241 protected expert manifest")
    _, manifest_payload = _json_object(manifest, label="r241 protected expert manifest")
    dates = list(manifest_payload.get("dates") or ())
    selection = dict(manifest_payload.get("selection") or {})
    if (
        manifest_payload.get("format") != "pokebot-bootstrap-feature-manifest"
        or dates != sorted(set(dates))
        or len(dates) != EXACT_WINDOW_DAYS
        or dates[0] != EXACT_WINDOW_START
        or dates[-1] != EXACT_WINDOW_END
        or str(selection.get("value") or "").casefold() != "alakazam"
        or selection.get("seat_semantics") != "acting_seat_only"
    ):
        raise R241LaunchError("protected expert manifest does not pin the exact Alakazam window")
    # A preservation receipt must bind this pointer to the source archive
    # receipt.  Do not infer provenance merely from matching dates.
    provenance = dict(payload.get("r241_archive_binding") or {})
    if (
        _as_path(provenance.get("archive_receipt_path"), label="expert archive binding path")
        != expected_archive_receipt
        or not str(provenance.get("archive_receipt_sha256") or "").startswith("sha256:")
    ):
        raise R241LaunchError("protected expert pointer lacks its exact archive binding")
    # The handoff must bind the archive checksum to the actual host-local
    # receipt, not merely declare something SHA-shaped.  The shared validator
    # also checks the preserved raw Elmo pointer/READY/manifest identities and
    # the immutable transfer receipt without rehashing multi-gigabyte shards
    # on every launch (that complete hash pass happens during transfer).
    try:
        checkpoint_receipts.validate_r241_protected_expert_pointer(
            pointer_file,
            archive_receipt_path=expected_archive_receipt,
        )
    except checkpoint_receipts.R241CheckpointReceiptError as exc:
        raise R241LaunchError(
            "protected expert pointer fails exact20 transfer provenance validation"
        ) from exc


def _path_binding(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    path = _as_path(_require(row, "path", label=label), label=f"{label} path")
    expected = str(_require(row, "sha256", label=label))
    if expected_sha256 is not None and expected != expected_sha256:
        raise R241LaunchError(f"{label} checksum differs from its r241 pin")
    return _expect_sha256(path, expected, label=label)


def _h10_tree_from_preservation_receipt(receipt: Mapping[str, Any]) -> Path:
    """Resolve the separately pinned H10 tree before checkpoint audit replay."""

    return _path_binding(
        dict(receipt.get("h10_marnie_matchup_tree") or {}),
        label="H10 Marnie Matchup Adapter tree",
        expected_sha256=checkpoint_receipts.H10_DIRECT_MATCHUP_TREE_SHA256,
    )


def _validate_gate_has_h10_floor(gate_path: Path) -> None:
    _, gate = _json_object(gate_path, label="r241 active gate contract")
    next_gate = dict(gate.get("next_gate") or {})
    roster = list(next_gate.get("roster") or ())
    rows = [
        dict(row)
        for row in roster
        if isinstance(row, dict)
        and row.get("opponent_id") == R241_H10_OPPONENT_ID
    ]
    if len(rows) != 1:
        raise R241LaunchError("active public gate must contain exactly one H10 Marnie row")
    h10 = rows[0]
    if (
        h10.get("content_digest") != R241_H10_CONTENT_SHA256
        or h10.get("frozen_checkpoint_digest") != R241_H10_MODEL_SHA256
        or h10.get("frozen_specialist") is not True
        or h10.get("tier") != "S++"
        or float(h10.get("weight") or 0.0) != 4.0
        or _exact_int(
            h10.get("strong_public_practice_floor_games"),
            label="H10 strong-public practice floor",
        )
        != EXACT_H10_MINIMUM
    ):
        raise R241LaunchError("active public gate no longer pins H10 Marnie at >=1024 games")


def _validate_marnie_adapter_receipt(
    registry: Mapping[str, Any],
    context: HostContext,
    preservation: PreservationContext,
    environment: Mapping[str, str],
) -> None:
    direct = dict(_require(registry, "direct_policy", label="registry"))
    marnie = dict(_require(registry, "marnie_h10", label="registry"))
    adapter = _expect_sha256(
        context.adapter_receipt,
        preservation.adapter_receipt_sha256,
        label="H10 adapter receipt",
    )
    _, payload = _json_object(adapter, label="H10 adapter receipt")
    try:
        checkpoint_receipts.validate_r241_h10_adapter_source_binding(
            adapter,
            source_snapshot={
                "root": str(preservation.source_snapshot.root),
                "manifest": str(preservation.source_snapshot.manifest),
                "manifest_sha256": preservation.source_snapshot.manifest_sha256,
            },
        )
    except checkpoint_receipts.R241CheckpointReceiptError as exc:
        raise R241LaunchError(
            "H10 adapter receipt does not bind the active source snapshot"
        ) from exc
    runtime = dict(payload.get("runtime") or {})
    package = dict(payload.get("package") or {})
    tree = dict(package.get("matchup_tree") or {})
    if (
        payload.get("schema") != direct.get("adapter_receipt_schema")
        or _exact_int(payload.get("revision"), label="H10 adapter revision") != R241_REVISION
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("direct_policy_only") is not True
        or payload.get("action_selector") != "direct_policy_only"
        or runtime.get("package_main_imported") is not False
        or runtime.get("package_search_invoked") is not False
        or runtime.get("embedded_cg_loaded") is not False
        or runtime.get("matchup_adapter_runtime") is not True
        or runtime.get("matchup_adapter_tree_loaded") is not True
        or _exact_int(runtime.get("mcts_calls"), label="H10 adapter MCTS calls") != 0
        or _exact_int(runtime.get("rtp_calls"), label="H10 adapter RTP calls") != 0
        or _exact_int(runtime.get("search_calls"), label="H10 adapter search calls") != 0
        or package.get("opponent_id") != R241_H10_OPPONENT_ID
        or package.get("content_sha256") != R241_H10_CONTENT_SHA256
        or dict(package.get("model") or {}).get("sha256") != R241_H10_MODEL_SHA256
        or tree.get("sha256") != marnie.get("matchup_tree_sha256")
        or _exact_int(tree.get("size_bytes"), label="H10 adapter tree size")
        != _exact_int(marnie.get("matchup_tree_size_bytes"), label="registry H10 tree size")
    ):
        raise R241LaunchError("H10 adapter receipt does not preserve direct policy + Matchup Adapter runtime")
    if preservation.baseline_payload is None:
        raise R241LaunchError("H10 adapter cannot run without the sealed external baseline payload")
    expected_package_root = (
        preservation.baseline_payload.root
        / "specialists"
        / str(marnie.get("baseline_dir") or "")
    )
    if (
        not str(marnie.get("baseline_dir") or "")
        or _as_path(package.get("root_path"), label="H10 adapter package root")
        != expected_package_root.resolve()
    ):
        raise R241LaunchError(
            "H10 adapter receipt does not bind the separately attested baseline payload"
        )
    sealed = dict(payload.get("sealed_runtime") or {})
    if (
        _as_path(sealed.get("cg_lib_path"), label="H10 sealed CG root")
        != context.official_cg_root
        or sealed.get("linux_x86_64_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
    ):
        raise R241LaunchError("H10 adapter receipt binds a different CG runtime")
    # Explicit environment check makes package old-cg / private overrides fail
    # before we ever hand this receipt to ``baselines_runtime``.
    validate_sealed_official_libcg(context.official_cg_root, environment=environment)


def _validate_source_snapshot(
    registry: Mapping[str, Any],
    context: HostContext,
    receipt: Mapping[str, Any],
) -> SourceSnapshotContext:
    """Verify the immutable code closure and external output boundary.

    ``run_root`` remains the owner-pinned durable output location.  The code
    that invokes it must instead come from an immutable content-addressed
    snapshot; deriving a checkout from ``run_root.parents`` is expressly not
    an allowed fallback.
    """

    contract = dict(_require(registry, "source_snapshot", label="registry"))
    if contract.get("status") != "ready":
        raise R241LaunchError("r241 immutable source snapshot has not been published")
    expected_hosts = dict(_require(contract, "hosts", label="source snapshot"))
    expected_host = dict(_require(expected_hosts, context.name, label="source snapshot hosts"))
    expected_manifest_sha = _valid_sha256(
        contract.get("manifest_sha256"), label="source snapshot manifest"
    )
    expected_tree_sha = _valid_sha256(
        contract.get("source_tree_sha256"), label="source snapshot tree"
    )
    root = _real_directory(
        _require(expected_host, "root", label="source snapshot host"),
        label="r241 source snapshot root",
    )
    outputs_root = _real_directory(
        _require(expected_host, "outputs_root", label="source snapshot host"),
        label="r241 external outputs root",
    )
    manifest_raw = Path(
        _require(expected_host, "manifest", label="source snapshot host")
    ).expanduser()
    if manifest_raw.is_symlink() or not manifest_raw.is_file():
        raise R241LaunchError("r241 source snapshot manifest must be a regular file")
    manifest = manifest_raw.resolve()
    if root not in manifest.parents or manifest.name != "r241-source-snapshot-manifest.json":
        raise R241LaunchError("r241 source snapshot manifest escapes or changes its root name")
    _expect_sha256(
        manifest, expected_manifest_sha, label="r241 source snapshot manifest"
    )
    if root.stat().st_mode & 0o222:
        raise R241LaunchError("r241 source snapshot root must be read-only")
    if manifest.stat().st_mode & 0o222:
        raise R241LaunchError("r241 source snapshot manifest must be read-only")
    if not root.name.startswith("alakazam-new-list-direct-r241-src-"):
        raise R241LaunchError("r241 source root is not content-addressed")
    if root == outputs_root or root in outputs_root.parents or outputs_root in root.parents:
        raise R241LaunchError("r241 source and external outputs roots must be disjoint")
    # The old mutable checkout is the parent of the durable outputs root.  Do
    # not accept it even if someone supplies a valid-looking manifest there.
    if root == outputs_root.parent:
        raise R241LaunchError("r241 refuses the mutable checkout as source execution root")
    run = dict(_require(registry, "run", label="registry"))
    expected_run_root = outputs_root / "pure_rl" / str(run.get("name") or "")
    actual_run_root = context.runtime_root.parent
    if actual_run_root != expected_run_root:
        raise R241LaunchError(
            "r241 run root must remain the bound external outputs/pure_rl path"
        )

    _, manifest_payload = _json_object(manifest, label="r241 source snapshot manifest")
    if (
        manifest_payload.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or manifest_payload.get("candidate_id")
        != "alakazam-new-list-direct-policy-r241"
        or manifest_payload.get("owner_contract_sha256")
        != contract.get("owner_contract_sha256")
        or manifest_payload.get("source_tree_sha256") != expected_tree_sha
        or manifest_payload.get("external_outputs_required") is not True
        or manifest_payload.get("baseline_payloads_separate_and_receipted") is not True
        or manifest_payload.get("authenticated") is not True
        or manifest_payload.get("status") != "authenticated_immutable_source_snapshot"
    ):
        raise R241LaunchError("r241 source snapshot manifest identity is invalid")
    rows_raw = manifest_payload.get("files")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise R241LaunchError("r241 source snapshot manifest lacks a file inventory")
    normalized_rows: list[dict[str, Any]] = []
    paths_seen: set[str] = set()
    for raw in rows_raw:
        if not isinstance(raw, Mapping):
            raise R241LaunchError("r241 source snapshot inventory item is not an object")
        relative = str(raw.get("path") or "")
        if relative == manifest.name:
            raise R241LaunchError(
                "r241 source snapshot manifest must be excluded from its file inventory"
            )
        if relative in paths_seen:
            raise R241LaunchError("r241 source snapshot inventory has duplicate paths")
        paths_seen.add(relative)
        expected_file_sha = _valid_sha256(
            raw.get("sha256"), label=f"snapshot file {relative or '?'}"
        )
        expected_size = _exact_int(
            raw.get("size_bytes"), label=f"snapshot file size {relative or '?'}"
        )
        if expected_size < 0:
            raise R241LaunchError("r241 source snapshot file size cannot be negative")
        member = _snapshot_member(root, relative, label="r241 source snapshot member")
        if member.stat().st_mode & 0o222:
            raise R241LaunchError(
                f"r241 source snapshot member must be read-only: {relative}"
            )
        if member.stat().st_size != expected_size or _sha256(member) != expected_file_sha:
            raise R241LaunchError(
                f"r241 source snapshot member identity drifted: {relative}"
            )
        normalized_rows.append(
            {
                "path": relative,
                "sha256": expected_file_sha,
                "size_bytes": expected_size,
            }
        )
    if _source_tree_digest(normalized_rows) != expected_tree_sha:
        raise R241LaunchError("r241 source snapshot tree digest does not match inventory")
    _validate_snapshot_tree_shape(
        root,
        manifest=manifest,
        inventory_paths=paths_seen,
    )
    missing_closure = sorted(_REQUIRED_SOURCE_SNAPSHOT_FILES - paths_seen)
    if missing_closure:
        raise R241LaunchError(
            "r241 source snapshot omits required execution closure: "
            + ", ".join(missing_closure)
        )

    binding = dict(receipt.get("source_snapshot") or {})
    if (
        binding.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or binding.get("host") != context.name
        or _as_path(binding.get("root"), label="receipt source root") != root
        or _as_path(binding.get("source_execution_root"), label="receipt execution root")
        != root
        or _as_path(binding.get("manifest"), label="receipt source manifest")
        != manifest
        or binding.get("manifest_sha256") != expected_manifest_sha
        or binding.get("source_tree_sha256") != expected_tree_sha
        or _as_path(binding.get("outputs_root"), label="receipt external outputs root")
        != outputs_root
    ):
        raise R241LaunchError("r241 preservation receipt source snapshot binding drifted")
    return SourceSnapshotContext(
        root=root,
        manifest=manifest,
        manifest_sha256=expected_manifest_sha,
        source_tree_sha256=expected_tree_sha,
        outputs_root=outputs_root,
    )


def _validate_baseline_payload_snapshot(
    registry: Mapping[str, Any],
    context: HostContext,
    source_snapshot: SourceSnapshotContext,
) -> BaselinePayloadContext:
    """Verify the external baseline library before any public package loads.

    This check is intentionally independent of the source tree: the mounted
    baseline root has its own complete receipt/inventory and cannot be a
    snapshot-local or checkout-relative fallback.  The direct H10 adapter then
    performs its package-member checks against this same root.
    """

    contract = dict(_require(registry, "baseline_payloads", label="registry"))
    if contract.get("status") != "ready":
        raise R241LaunchError("r241 separately attested baseline payload is not ready")
    host = dict(
        _require(
            dict(_require(contract, "hosts", label="baseline payloads")),
            context.name,
            label="baseline payload hosts",
        )
    )
    canonical_receipt_path = _expect_sha256(
        Path(
            str(
                _require(
                    host,
                    "canonical_roster_receipt",
                    label="baseline payload host",
                )
            )
        ),
        _require(
            contract,
            "canonical_roster_receipt_sha256",
            label="baseline payloads",
        ),
        label="r241 externally derived canonical baseline roster receipt",
    )
    try:
        canonical_receipt_path, canonical_roster_receipt = (
            baseline_payload_snapshot.validate_canonical_roster_receipt(
                canonical_receipt_path,
                expected_sha256=str(contract["canonical_roster_receipt_sha256"]),
                owner_contract_sha256=str(
                    _require(registry, "owner_contract", label="registry")["sha256"]
                ),
            )
        )
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241LaunchError(
            f"r241 canonical baseline roster receipt failed validation: {exc}"
        ) from exc
    try:
        identity = baseline_payload_snapshot.validate_snapshot(
            root=_require(host, "root", label="baseline payload host"),
            manifest_path=_require(host, "manifest", label="baseline payload host"),
            manifest_sha256=_require(host, "manifest_sha256", label="baseline payload host"),
            baseline_tree_sha256=_require(
                host, "baseline_tree_sha256", label="baseline payload host"
            ),
            owner_contract_sha256=str(
                _require(registry, "owner_contract", label="registry")["sha256"]
            ),
        )
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241LaunchError(
            f"r241 separately attested baseline payload failed validation: {exc}"
        ) from exc
    root = Path(str(identity["root"])).resolve()
    manifest = Path(str(identity["manifest"])).resolve()
    canonical_roster = baseline_payload_snapshot.normalized_roster(
        list(canonical_roster_receipt.get("baseline_roster") or [])
    )
    if (
        str(canonical_roster_receipt.get("baseline_manifest_sha256") or "")
        != str(contract.get("canonical_baseline_manifest_sha256") or "")
        or str(canonical_roster_receipt.get("baseline_roster_sha256") or "")
        != str(contract.get("canonical_baseline_roster_sha256") or "")
        or str(identity.get("baseline_manifest_sha256") or "")
        != str(contract.get("canonical_baseline_manifest_sha256") or "")
        or str(identity.get("baseline_roster_sha256") or "")
        != str(contract.get("canonical_baseline_roster_sha256") or "")
        or identity.get("baseline_roster") != canonical_roster
    ):
        raise R241LaunchError(
            "r241 mounted baseline payload is not the externally derived canonical roster"
        )
    if (
        root == source_snapshot.root
        or root in source_snapshot.root.parents
        or source_snapshot.root in root.parents
        or root == source_snapshot.outputs_root
        or root in source_snapshot.outputs_root.parents
        or source_snapshot.outputs_root in root.parents
        or root == ROOT / "baselines"
    ):
        raise R241LaunchError(
            "r241 baseline payload must be a separately mounted external library"
        )
    staging_path = _expect_sha256(
        Path(str(_require(host, "staging_receipt", label="baseline payload host"))),
        _require(host, "staging_receipt_sha256", label="baseline payload host"),
        label="r241 baseline payload staging receipt",
    )
    _, staging = _json_object(staging_path, label="r241 baseline payload staging receipt")
    snapshot_binding = dict(staging.get("baseline_payload_snapshot") or {})
    canonical_binding = dict(staging.get("canonical_roster_receipt") or {})
    if (
        staging.get("schema") != baseline_payload_snapshot.BASELINE_PAYLOAD_STAGING_SCHEMA
        or staging.get("revision") != R241_REVISION
        or staging.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or staging.get("status") != "passed"
        or staging.get("passed") is not True
        or snapshot_binding.get("host") != context.name
        or snapshot_binding.get("root") != str(root)
        or snapshot_binding.get("manifest") != str(manifest)
        or snapshot_binding.get("manifest_sha256") != identity["manifest_sha256"]
        or snapshot_binding.get("baseline_tree_sha256")
        != identity["baseline_tree_sha256"]
        or snapshot_binding.get("owner_contract_sha256")
        != _require(registry, "owner_contract", label="registry")["sha256"]
        or _as_path(
            canonical_binding.get("path"),
            label="baseline staging canonical-roster receipt",
        )
        != canonical_receipt_path
        or canonical_binding.get("sha256")
        != contract.get("canonical_roster_receipt_sha256")
        or canonical_binding.get("baseline_manifest_sha256")
        != contract.get("canonical_baseline_manifest_sha256")
        or canonical_binding.get("baseline_roster_sha256")
        != contract.get("canonical_baseline_roster_sha256")
        or canonical_binding.get("public_contract_sha256s")
        != canonical_roster_receipt.get("public_contract_sha256s")
    ):
        raise R241LaunchError("r241 baseline payload staging receipt binding drifted")
    return BaselinePayloadContext(
        root=root,
        manifest=manifest,
        manifest_sha256=str(identity["manifest_sha256"]),
        baseline_tree_sha256=str(identity["baseline_tree_sha256"]),
        canonical_roster_receipt=canonical_receipt_path,
        canonical_roster_receipt_sha256=str(
            contract["canonical_roster_receipt_sha256"]
        ),
        baseline_manifest_sha256=str(contract["canonical_baseline_manifest_sha256"]),
        baseline_roster_sha256=str(contract["canonical_baseline_roster_sha256"]),
        baseline_roster=tuple(canonical_roster),
    )


def _assert_launcher_executes_from_snapshot(snapshot: SourceSnapshotContext) -> None:
    """Reject a mutable launcher importing checkpoint/runtime code by accident."""

    if ROOT.resolve() != snapshot.root:
        raise R241LaunchError(
            "r241 activation must execute the launcher from its verified source snapshot, "
            f"not {ROOT}"
        )


def _same_declared_path(value: object, expected: object) -> bool:
    """Compare receipt paths without resolving an off-host Elmo source root."""

    actual_text = str(value or "").strip()
    expected_text = str(expected or "").strip()
    if not actual_text or not expected_text:
        return False
    actual = Path(actual_text).expanduser()
    wanted = Path(expected_text).expanduser()
    return actual.is_absolute() and wanted.is_absolute() and actual == wanted


def _validate_remote_source_snapshot_binding(
    binding: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    label: str,
) -> None:
    """Require the Elmo remote receipt to run the same sealed code tree.

    The Inzi process need not resolve Elmo's source root locally.  Instead it
    checks the literal host binding and the location-independent manifest/tree
    identities that the Elmo worker independently verified before emitting its
    receipt.
    """

    contract = dict(_require(registry, "source_snapshot", label="registry"))
    elmo = dict(_require(dict(contract.get("hosts") or {}), "elmo", label="source snapshot hosts"))
    if (
        binding.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or binding.get("host") != "elmo"
        or not _same_declared_path(binding.get("root"), elmo.get("root"))
        or not _same_declared_path(
            binding.get("source_execution_root"), elmo.get("root")
        )
        or not _same_declared_path(binding.get("manifest"), elmo.get("manifest"))
        or not _same_declared_path(
            binding.get("outputs_root"), elmo.get("outputs_root")
        )
        or binding.get("manifest_sha256") != contract.get("manifest_sha256")
        or binding.get("source_tree_sha256") != contract.get("source_tree_sha256")
    ):
        raise R241LaunchError(
            f"{label} does not bind the staged immutable Elmo source snapshot"
        )


def _validate_remote_collection(
    registry: Mapping[str, Any],
) -> None:
    """Require the sealed :8767 receipt identities without opening Elmo paths.

    The canonical overlay publisher inspects checksum-pinned local copies of
    the four Elmo preflight receipts.  The Elmo worker independently repeats
    the full receipt and runtime validation before serving.  An Inzi launcher
    therefore has no authority to dereference `/mnt/Main/...` paths: doing so
    would make a controller topology assumption part of the activation gate.
    This narrow boundary still rejects blank, legacy, or transport-drifting
    remote bindings before generic ``RemoteJobClient`` code can run.
    """

    endpoint = _validate_remote_collection_contract(registry)
    for key in (
        "manifest_sha256",
        "host_receipt_sha256",
        "runtime_receipt_sha256",
        "gameplay_receipt_sha256",
    ):
        _valid_sha256(endpoint.get(key), label=f"r241 remote {key}")
    transport = _remote_checkpoint_transport(registry)
    if transport.get("status") != "ready":
        raise R241LaunchError(
            "r241 remote collection requires the overlay-bound ready checkpoint transport"
        )


def _validate_preservation_receipt(
    registry: Mapping[str, Any], context: HostContext, environment: Mapping[str, str]
) -> PreservationContext:
    preservation = dict(_require(registry, "peak_r195_preservation", label="registry"))
    receipt_path, receipt = _json_object(
        context.preservation_receipt, label="r241 peak-r195 preservation receipt"
    )
    expected_receipt_sha256 = str(
        preservation.get(f"receipt_sha256_{context.name}") or ""
    )
    _expect_sha256(
        receipt_path,
        expected_receipt_sha256,
        label="r241 peak-r195 preservation receipt",
    )
    if (
        receipt.get("schema") != preservation.get("receipt_schema")
        or receipt.get("schema") != PRESERVATION_SCHEMA
        or _exact_int(receipt.get("revision"), label="preservation revision")
        != R241_REVISION
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("derived_not_self_asserted") is not True
    ):
        raise R241LaunchError("peak-r195 preservation receipt is not passed")
    source_snapshot = _validate_source_snapshot(registry, context, receipt)
    baseline_payload = _validate_baseline_payload_snapshot(
        registry, context, source_snapshot
    )
    raw_parent = receipt.get("parent")
    if not isinstance(raw_parent, Mapping):
        raise R241LaunchError(
            "preservation parent must be an exact typed FileIdentity"
        )
    parent = dict(raw_parent)
    registry_parent = dict(_require(registry, "parent", label="registry"))
    if set(parent) != {"path", "sha256", "size_bytes"}:
        raise R241LaunchError(
            "preservation parent must be an exact typed FileIdentity"
        )
    _as_path(registry_parent.get("checkpoint"), label="registry parent checkpoint")
    registry_parent_sha256 = str(registry_parent.get("sha256") or "")
    registry_parent_size = _exact_int(
        registry_parent.get("size_bytes"), label="registry parent size"
    )
    parent_sha256 = str(parent.get("sha256") or "")
    parent_size = _exact_int(
        parent.get("size_bytes"), label="preservation parent size"
    )
    if (
        parent_sha256 != registry_parent_sha256
        or parent_size != registry_parent_size
    ):
        raise R241LaunchError("preservation receipt parent does not pin immutable r195")
    try:
        parent_identity = checkpoint_receipts.file_identity(
            _as_path(parent.get("path"), label="preservation parent path"),
            label="immutable r195 parent checkpoint",
            expected_sha256=parent_sha256,
            expected_size_bytes=parent_size,
        )
    except checkpoint_receipts.R241CheckpointReceiptError as exc:
        raise R241LaunchError(
            f"immutable r195 parent checkpoint identity drifted: {exc}"
        ) from exc
    parent_file = parent_identity.path
    adapter = dict(receipt.get("matchup_adapter") or {})
    learner_tree = _path_binding(
        adapter,
        label="r195 learner Matchup Adapter tree",
        expected_sha256=preservation["learner_matchup_tree_sha256"],
    )
    h10_tree = _h10_tree_from_preservation_receipt(receipt)
    try:
        checkpoint_receipts.validate_peak_r195_preservation_receipt(
            receipt_path=receipt_path,
            parent_checkpoint=parent_file,
            learner_matchup_tree=learner_tree,
            h10_matchup_tree=h10_tree,
            official_cg_root=context.official_cg_root,
            environment=environment,
        )
    except checkpoint_receipts.R241CheckpointReceiptError as exc:
        raise R241LaunchError(
            f"peak-r195 preservation checkpoint audit failed: {exc}"
        ) from exc
    baseline_roster = _path_binding(
        dict(receipt.get("baseline_adapter_roster") or {}),
        label="r195 immutable baseline adapter roster",
        expected_sha256=preservation["baseline_adapter_roster_sha256"],
    )
    expected_baseline_roster = _bound_source_path(
        source_snapshot.root,
        preservation["baseline_adapter_roster"],
        label="r195 immutable baseline adapter roster source path",
    )
    if baseline_roster != expected_baseline_roster:
        raise R241LaunchError(
            "peak-r195 preservation receipt loaded the adapter roster outside the source snapshot"
        )
    slot_migration = dict(receipt.get("adapter_slot_migration") or {})
    if (
        slot_migration.get("schema")
        != "poke_bot.alakazam_new_list_direct_policy_r241_adapter_slot_migration/v1"
        or slot_migration.get("status") != "no_slot_change"
        or _exact_int(
            slot_migration.get("retained_slot_count"),
            label="r195 retained adapter-slot count",
        )
        != _exact_int(
            preservation.get("immutable_adapter_slot_prefix"),
            label="r195 immutable adapter-slot prefix",
        )
        or slot_migration.get("existing_slots_byte_immutable") is not True
        or slot_migration.get("new_slots") != []
        or slot_migration.get("new_slot_proofs") != []
    ):
        raise R241LaunchError(
            "r241 preservation receipt must retain the exact current adapter slots"
        )
    heads = dict(receipt.get("heads") or {})
    combo = dict(heads.get("combo_state") or {})
    expected_heads = {
        "architecture_present_head_count": 19,
        "non_combo_head_count": 18,
        "non_combo_route_count": 18,
        "every_non_combo_head_trainable": True,
        "every_non_combo_fusion_route_enabled": True,
    }
    for key, expected in expected_heads.items():
        if heads.get(key) != expected:
            raise R241LaunchError(f"peak-r195 head/route preservation failed: {key}")
    if (
        combo.get("head_present") is not True
        or combo.get("physical_route_present") is not True
        or combo.get("loss_weight") != 0.0
        or combo.get("route_enabled") is not False
    ):
        raise R241LaunchError("peak-r195 combo-state preservation is invalid")
    activation = _path_binding(
        dict(adapter.get("training_activation") or {}),
        label="r195 Matchup Adapter training activation",
    )
    if (
        adapter.get("bank_preserved") is not True
        or adapter.get("training_enabled") is not True
        or adapter.get("runtime_enabled") is not True
        or adapter.get("checkpoint_runtime_enabled") is not False
        or adapter.get("checkpoint_training_enabled") is not False
        or adapter.get("runtime_package_activation_required") is not True
        or _exact_int(adapter.get("epochs_per_rl_update"), label="adapter epochs per update") <= 0
    ):
        raise R241LaunchError("r195 Matchup Adapter bank/training/runtime is not preserved")
    audit_adapter = dict(
        dict(receipt.get("checkpoint_audit") or {}).get("matchup_adapter") or {}
    )
    dormant = dict(audit_adapter.get("checkpoint_dormant_state") or {})
    fit = dict(audit_adapter.get("fit") or {})
    isolated = dict(audit_adapter.get("isolated_optimizer") or {})
    if (
        dormant.get("runtime_enabled") is not False
        or dormant.get("training_enabled") is not False
        or dormant.get("ordinary_optimizer_included") is not False
        or fit.get("optimizer_scope") != "matchup_adapter_bank_only"
        or _exact_int(
            isolated.get("parameter_count"), label="isolated adapter optimizer parameter count"
        ) != 256
        or _exact_int(
            isolated.get("state_count"), label="isolated adapter optimizer state count"
        ) <= 0
    ):
        raise R241LaunchError(
            "r241 checkpoint must retain the isolated dormant Matchup Adapter optimizer semantics"
        )
    public = dict(receipt.get("public_mix") or {})
    active_gate = _path_binding(dict(public.get("active_gate_contract") or {}), label="r241 active gate contract")
    frozen = _path_binding(dict(public.get("frozen_specialist_registry") or {}), label="r241 frozen specialist registry")
    research = _path_binding(dict(public.get("research_control_registry") or {}), label="r241 research-control registry")
    fraction = float(public.get("official_collect_fraction") or -1.0)
    research_games = _exact_int(public.get("research_control_games_per_iter"), label="research-control games")
    if (
        public.get("established_diverse_public_mix_preserved") is not True
        or public.get("research_control_phase_preserved") is not True
        or fraction != 0.50
        or research_games != 1_000
    ):
        raise R241LaunchError("preservation receipt changed the established public/research mix")
    if baseline_payload is None:
        raise R241LaunchError("r241 cannot bind public contracts without its canonical baseline roster")
    try:
        _canonical_path, canonical_roster_receipt = (
            baseline_payload_snapshot.validate_canonical_roster_receipt(
                baseline_payload.canonical_roster_receipt,
                expected_sha256=baseline_payload.canonical_roster_receipt_sha256,
                owner_contract_sha256=str(
                    _require(registry, "owner_contract", label="registry")["sha256"]
                ),
            )
        )
        baseline_payload_snapshot.validate_canonical_roster_contract_bindings(
            canonical_roster_receipt,
            active_gate_contract=active_gate,
            frozen_specialist_registry=frozen,
            research_control_registry=research,
        )
    except baseline_payload_snapshot.R241BaselinePayloadError as exc:
        raise R241LaunchError(
            f"r241 canonical baseline roster/public-contract binding drifted: {exc}"
        ) from exc
    _validate_gate_has_h10_floor(active_gate)
    plan = planned_collection_group_counts(
        games_per_iteration=EXACT_GAMES_PER_UPDATE,
        self_play_fraction=float(EXACT_SELF_PLAY_FRACTION),
        strong_public_fraction_of_public=fraction,
        research_control_games=research_games,
    )
    if plan != {
        "self_play": EXACT_SELF_GAMES,
        "strong_public_practice": 4_586,
        "diverse_public": 2_586,
    }:
        raise R241LaunchError(
            f"planned r241 collection is not 1024 self + 7172 public mix: {plan}"
        )
    expert = dict(receipt.get("expert_window") or {})
    expert_pointer = _path_binding(dict(expert.get("protected_pointer") or {}), label="r241 protected expert pointer")
    if expert_pointer != context.expert_manifest_pointer:
        raise R241LaunchError("preservation receipt changed the host expert pointer")
    if (
        expert.get("archive_receipt_sha256")
        != dict(_require(registry, "expert_window", label="registry")).get("canonical_receipt", {}).get("sha256")
    ):
        raise R241LaunchError("preservation receipt changed exact expert archive provenance")
    adapter_binding = dict(receipt.get("h10_adapter_receipt") or {})
    adapter_path = _path_binding(adapter_binding, label="H10 direct-policy adapter receipt")
    if adapter_path != context.adapter_receipt:
        raise R241LaunchError("preservation receipt changed the H10 adapter receipt path")
    trainer = dict(receipt.get("trainer") or {})
    args = tuple(str(value) for value in trainer.get("r195_non_combo_arguments") or ())
    _validate_preserved_trainer_args(args)
    return PreservationContext(
        active_gate_contract=active_gate,
        frozen_specialist_registry=frozen,
        research_control_registry=research,
        learner_matchup_tree=learner_tree,
        adapter_activation_receipt=activation,
        expert_manifest_pointer=expert_pointer,
        preservation_receipt_sha256=expected_receipt_sha256,
        adapter_receipt_sha256=str(adapter_binding["sha256"]),
        official_collect_fraction=fraction,
        research_control_games=research_games,
        matchup_adapter_epochs_per_update=_exact_int(
            adapter.get("epochs_per_rl_update"), label="adapter epochs per update"
        ),
        trainer_args=args,
        source_snapshot=source_snapshot,
        baseline_payload=baseline_payload,
    )


_FORBIDDEN_TRAINER_ARGUMENTS = frozenset(
    {
        "--continue-after-gate",
        "--minimum-terminal-iteration",
        "--terminal-active-gate-id",
        "--terminal-gate-marker-name",
        "--population-own-models-only",
        "--population-opponent-registry",
        "--expert-rehearsal-before-first",
        "--expert-rehearsal-force-before",
        "--expert-rehearsal-one-time-before",
        "--expert-rehearsal-one-time-epochs",
        "--smoke",
        "--smoke-games",
        "--no-remote-workers",
        "--allow-single-gpu",
        "--no-require-exact-training-seat-split",
        "--resume",
        "--start-iteration",
        "--allow-clean-boundary-design-migration",
        "--boundary-design-migration-reason",
        "--base-checkpoint",
        "--initial-learner-checkpoint",
        "--iterations",
        "--fixed-cycle-updates",
        "--r241-peak-r195-preservation-receipt",
        "--r241-peak-r195-preservation-receipt-sha256",
        "--games-per-iter",
        "--official-collect-frac",
        "--research-control-games-per-iter",
        "--active-gate-contract",
        "--frozen-specialist-registry",
        "--research-control-registry",
        "--expert-manifest",
        "--terminal-expert-rehearsal",
        "--no-terminal-expert-rehearsal",
        "--expert-rehearsal-every",
        "--expert-rehearsal-epochs",
        "--combo-state-loss-weight",
        "--current-deck-guide-loss-weight",
        "--current-deck-guide-training-mode",
        "--current-deck-guide-curriculum-spec",
        "--current-deck-guide-head-role-map",
        "--current-deck-guide-curriculum-validation-receipt",
        "--dormant-matchup-adapter-epochs",
        "--dormant-matchup-adapter-activation-receipt",
    }
)


def _validate_preserved_trainer_args(args: Sequence[str]) -> None:
    """Allow r195 tuning, but not a second authority over r241 invariants."""

    if not args:
        raise R241LaunchError("preservation receipt omitted r195 non-combo trainer args")
    for token in args:
        if token in _FORBIDDEN_TRAINER_ARGUMENTS or token.startswith(
            tuple(f"{flag}=" for flag in _FORBIDDEN_TRAINER_ARGUMENTS)
        ):
            raise R241LaunchError(
                f"preservation receipt attempts to override r241 invariant: {token}"
            )
    # These are the r195 non-combo auxiliary-loss weights.  Require the exact
    # value pairs rather than assuming parser defaults have remained stable.
    required_pairs = {
        ("--archetype-aux-loss-weight", "0.05"),
        ("--opp-hand-loss-weight", "0.05"),
        ("--opp-remainder-loss-weight", "0.05"),
        ("--lethal-threat-loss-weight", "0.025"),
        ("--prize-race-loss-weight", "0.025"),
        ("--setup-board-outcome-loss-weight", "0.025"),
    }
    pairs = set(zip(args, args[1:]))
    missing = sorted(pair for pair in required_pairs if pair not in pairs)
    if missing:
        raise R241LaunchError(
            f"preservation receipt omitted r195 non-combo trainer settings: {missing}"
        )


def validate_activation(
    registry: Mapping[str, Any],
    *,
    host: str,
    environment: Mapping[str, str] | None = None,
    activation_overlay: ActivationOverlayContext | None = None,
) -> tuple[HostContext, PreservationContext]:
    """Validate every mutable host artifact immediately before command build."""

    if activation_overlay is None:
        raise R241LaunchError(
            "r241 activation requires an explicit checksum-bound external activation overlay"
        )
    validate_static_registry(registry)
    context = _host_context(registry, host)
    env = build_environment(registry, context, environment=environment)
    try:
        assert_direct_policy_environment(env)
        validate_sealed_official_libcg(context.official_cg_root, environment=env)
    except R241DirectPolicyRuntimeError as exc:
        raise R241LaunchError(f"sealed r241 direct-policy runtime failed: {exc}") from exc
    _validate_host_expert_window(registry, context)
    preservation = _validate_preservation_receipt(registry, context, env)
    _assert_launcher_executes_from_snapshot(preservation.source_snapshot)
    _validate_protected_expert_pointer(
        preservation.expert_manifest_pointer,
        expected_archive_receipt=context.expert_archive_receipt,
    )
    final_env = build_environment(
        registry, context, preservation=preservation, environment=environment
    )
    _validate_marnie_adapter_receipt(
        registry, context, preservation, final_env
    )
    _validate_remote_collection(registry)
    return context, preservation


def build_environment(
    registry: Mapping[str, Any],
    context: HostContext,
    preservation: PreservationContext | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Seal the direct-policy selector without removing Matchup Adapters."""

    env = dict(os.environ if environment is None else environment)
    for key in tuple(env):
        if key in {
            "POKEBOT_LIBCG_PATH",
            "POKEBOT_BATCH_LIBCG",
            "POKEBOT_ALLOW_ORACLE_DECK",
            "POKEBOT_BASELINES_DIR",
            "POKEBOT_REMOTE_CHECKPOINT_ROOT",
            "POKEBOT_ELMO_SSH_STAGE",
            "POKEBOT_ELMO_CHECKPOINT_HOST_DIR",
            "POKEBOT_ELMO_CHECKPOINT_VERIFY_PORT",
            "POKEBOT_TRUENAS_CHECKPOINT_SMB",
            "POKEBOT_R241_ACTIVATION_OVERLAY",
            "POKEBOT_R241_ACTIVATION_OVERLAY_SHA256",
            "POKEBOT_R241_ACTIVATION_OVERLAY_MIRROR_RECEIPT",
            "POKEBOT_R241_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256",
            "POKEBOT_R241_OWNER_START_AUTHORIZATION",
            "POKEBOT_R241_OWNER_START_AUTHORIZATION_SHA256",
        }:
            env.pop(key, None)
        elif key.startswith(
            (
                "POKEBOT_MCTS_",
                "POKEBOT_RTP_",
                "POKEBOT_BELIEF_",
                "POKEBOT_POKE_RLM_",
                "POKEBOT_SLOWKING_DISTILL_",
                "POKEBOT_GUIDE2VEC_",
            )
        ):
            env.pop(key, None)
        elif key.startswith("POKEBOT_SEARCH_") and key != "POKEBOT_SEARCH_MODE":
            env.pop(key, None)
    # These parser defaults would otherwise let an inherited shell alter the
    # exact fixed-cycle cadence before the trainer gets a chance to reject it.
    # The command supplies the authorized values explicitly below.
    for key in (
        "PURE_RL_POPULATION_OWN_MODELS_ONLY",
        "PURE_RL_POPULATION_OPPONENT_REGISTRY",
        "PURE_RL_EXPERT_REHEARSAL_FORCE_BEFORE",
        "PURE_RL_EXPERT_REHEARSAL_ONE_TIME_BEFORE",
        "PURE_RL_EXPERT_REHEARSAL_ONE_TIME_EPOCHS",
        "PURE_RL_CONTINUE_AFTER_GATE",
        "PURE_RL_MINIMUM_TERMINAL_ITERATION",
        "PURE_RL_TERMINAL_GATE_MARKER_NAME",
    ):
        env.pop(key, None)
    deck = dict(_require(registry, "deck", label="registry"))
    guide = dict(_require(registry, "guide", label="registry"))
    cycle = dict(_require(registry, "training_cycle", label="registry"))
    parent = dict(_require(registry, "parent", label="registry"))
    remote_endpoint = _validate_remote_collection_contract(registry)
    checkpoint_transport = _remote_checkpoint_transport(registry)
    execution_root = preservation.source_snapshot.root if preservation else ROOT
    env.update(
        {
            "CG_LIB_PATH": str(context.official_cg_root),
            R241_DIRECT_POLICY_ONLY_ENV: "1",
            R241_DIRECT_POLICY_RECEIPT_ENV: str(context.adapter_receipt),
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
            "POKEBOT_SEARCH_MODE": "policy",
            "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
            "POKEBOT_SPECIALIST_DECK_PATH": str(
                _bound_source_path(
                    execution_root,
                    deck["path"],
                    label="r241 exact deck source path",
                )
            ),
            "POKEBOT_SPECIALIST_DECK_SHA256": str(deck["sha256"]),
            "POKEBOT_SPECIALIST_DECK_MULTISET_SHA256": str(deck["multiset_sha256"]),
            "POKEBOT_CURRENT_DECK_GUIDE": "alakazam",
            "POKEBOT_CURRENT_DECK_GUIDE_TARGETS": "1",
            "POKEBOT_CURRENT_DECK_GUIDE_VERSION": str(guide["version"]),
            "POKEBOT_ACTIVE_SPECIALIST": "alakazam",
            "POKEBOT_COMBO_STATE_ROUTE_ENABLED": "0",
            "POKEBOT_COMBO_STATE_ROUTE_SPECIALIST": "alakazam",
            "POKEBOT_COMBO_STATE_ROUTE_CHECKPOINT_DIGEST": str(parent["sha256"]),
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
            "PURE_RL_SELF_PLAY_FRAC": str(cycle["self_play_fraction"]),
            "PURE_RL_FIXED_CYCLE_UPDATES": "10",
            "PURE_RL_EXPERT_REHEARSAL_EVERY": "5",
            "PURE_RL_EXPERT_REHEARSAL_EPOCHS": "5",
            "PURE_RL_TERMINAL_EXPERT_REHEARSAL": "1",
            "PURE_RL_REQUIRE_EXACT_TRAINING_SEAT_SPLIT": "1",
            # r241's preserved public mix must be allowed to use the sealed
            # r236 remote collector once its endpoint receipt passes.  The
            # trainer defaults this setting to local-only, which would silently
            # erase the established remote public-mix topology.
            "PURE_RL_PUBLIC_MIX_LOCAL_ONLY": "0",
            # Do not allow the generic launcher to infer its historical
            # :8765/:8766 default.  The outer command repeats this literal so
            # it remains authoritative even if a future launcher changes env
            # precedence.
            "PURE_RL_REMOTE_WORKER_ENDPOINTS": str(remote_endpoint["endpoint"]),
            "POKEBOT_REMOTE_WORKER_ENDPOINTS": str(remote_endpoint["endpoint"]),
            "POKEBOT_R241_REMOTE_ENDPOINT_ID": str(remote_endpoint["id"]),
            "POKEBOT_R241_REMOTE_COLLECTION_CAPABILITY": str(
                remote_endpoint["capability"]
            ),
            "POKEBOT_R241_REMOTE_PREFLIGHT_MANIFEST": str(
                remote_endpoint["manifest_path"]
            ),
        }
    )
    if preservation is not None:
        # All executable code comes from the checksum-bound source snapshot;
        # generated checkpoints, logs, and state remain in the separate
        # owner-pinned external output root.
        env["POKEBOT_OUTPUTS_DIR"] = str(preservation.source_snapshot.outputs_root)
        env["POKEBOT_R241_SOURCE_EXECUTION_ROOT"] = str(
            preservation.source_snapshot.root
        )
        env["POKEBOT_R241_SOURCE_SNAPSHOT_MANIFEST"] = str(
            preservation.source_snapshot.manifest
        )
        env["POKEBOT_R241_SOURCE_SNAPSHOT_MANIFEST_SHA256"] = (
            preservation.source_snapshot.manifest_sha256
        )
        env["POKEBOT_R241_SOURCE_TREE_SHA256"] = (
            preservation.source_snapshot.source_tree_sha256
        )
        if preservation.baseline_payload is None:
            raise R241LaunchError(
                "r241 source snapshot activation requires a sealed external baseline payload"
            )
        env["POKEBOT_BASELINES_DIR"] = str(preservation.baseline_payload.root)
        env["POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST"] = str(
            preservation.baseline_payload.manifest
        )
        env["POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST_SHA256"] = (
            preservation.baseline_payload.manifest_sha256
        )
        env["POKEBOT_R241_BASELINE_PAYLOAD_TREE_SHA256"] = (
            preservation.baseline_payload.baseline_tree_sha256
        )
        if preservation.activation_overlay is None:
            raise R241LaunchError(
                "r241 source-snapshot environment requires an external activation overlay"
            )
        if checkpoint_transport.get("status") != "ready":
            raise R241LaunchError(
                "r241 source-snapshot environment requires a receipt-bound Elmo checkpoint transport"
            )
        env["POKEBOT_R241_ACTIVATION_OVERLAY"] = str(
            preservation.activation_overlay.path
        )
        env["POKEBOT_R241_ACTIVATION_OVERLAY_SHA256"] = (
            preservation.activation_overlay.sha256
        )
        env["POKEBOT_R241_ACTIVATION_OVERLAY_MIRROR_RECEIPT"] = str(
            preservation.activation_overlay.mirror_receipt
        )
        env["POKEBOT_R241_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256"] = (
            preservation.activation_overlay.mirror_receipt_sha256
        )
        env["POKEBOT_R241_OWNER_START_AUTHORIZATION"] = str(
            preservation.activation_overlay.authorization_receipt
        )
        env["POKEBOT_R241_OWNER_START_AUTHORIZATION_SHA256"] = (
            preservation.activation_overlay.authorization_receipt_sha256
        )
        # Generic remote jobs historically infer a legacy endpoint/SMB route
        # unless these exact :8767 transport bindings are present.  They are
        # supplied only after the overlay has validated the content-addressed
        # transport receipt; no source-snapshot default can provide them.
        env["POKEBOT_ELMO_SSH_STAGE"] = "1"
        env["POKEBOT_ELMO_CHECKPOINT_HOST_DIR"] = str(
            checkpoint_transport["host_root"]
        )
        env["POKEBOT_ELMO_CHECKPOINT_VERIFY_PORT"] = str(
            checkpoint_transport["verification_port"]
        )
        env["POKEBOT_TRUENAS_CHECKPOINT_SMB"] = str(
            checkpoint_transport["trainer_visible_root"]
        )
        env["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(
            preservation.learner_matchup_tree
        )
        env["PURE_RL_OFFICIAL_COLLECT_FRAC"] = str(
            preservation.official_collect_fraction
        )
        env["PURE_RL_RESEARCH_CONTROL_GAMES_PER_ITER"] = str(
            preservation.research_control_games
        )
    return env


def build_command(
    registry: Mapping[str, Any],
    context: HostContext,
    preservation: PreservationContext,
    *,
    python: str,
) -> list[str]:
    """Return the one exact r241 command; this function performs no I/O."""

    guide = dict(_require(registry, "guide", label="registry"))
    parent = dict(_require(registry, "parent", label="registry"))
    cycle = dict(_require(registry, "training_cycle", label="registry"))
    remote_endpoint = _validate_remote_collection_contract(registry)
    execution_root = preservation.source_snapshot.root
    # Use the normal safety/monitor wrapper.  It will require its separate
    # operator arm before actual service work; this wrapper never bypasses it.
    command = [
        str(python),
        "-u",
        str(
            _bound_source_path(
                execution_root,
                "scripts/launch_pure_rl.py",
                label="r241 pure-RL launcher source path",
            )
        ),
        "--run-name",
        str(dict(registry["run"])["name"]),
        "--mode",
        "specialist",
        "--python",
        str(python),
        "--preflight-profile",
        "none",
        "--log",
        str(
            preservation.source_snapshot.outputs_root
            / "logs"
            / "alakazam_new_list_direct_policy_r241.log"
        ),
        "--remote-worker-endpoints",
        str(remote_endpoint["endpoint"]),
        "--",
        "--specialist-archetype",
        "alakazam",
        "--base-checkpoint",
        str(parent["checkpoint"]),
        "--iterations",
        "10",
        "--fixed-cycle-updates",
        "10",
        "--r241-peak-r195-preservation-receipt",
        str(context.preservation_receipt),
        "--r241-peak-r195-preservation-receipt-sha256",
        str(preservation.preservation_receipt_sha256),
        "--games-per-iter",
        str(EXACT_GAMES_PER_UPDATE),
        "--require-exact-training-seat-split",
        "--official-collect-frac",
        str(preservation.official_collect_fraction),
        "--research-control-games-per-iter",
        str(preservation.research_control_games),
        "--active-gate-contract",
        str(preservation.active_gate_contract),
        "--frozen-specialist-registry",
        str(preservation.frozen_specialist_registry),
        "--research-control-registry",
        str(preservation.research_control_registry),
        "--measurement-decks",
        "alakazam",
        "--expert-manifest",
        str(preservation.expert_manifest_pointer),
        "--expert-rehearsal-every",
        "5",
        "--expert-rehearsal-epochs",
        "5",
        "--terminal-expert-rehearsal",
        "--expert-rehearsal-guide-loss-weight",
        str(guide["expert_refresh_loss_weight"]),
        "--current-deck-guide-loss-weight",
        str(guide["ordinary_rl_loss_weight"]),
        "--current-deck-guide-training-mode",
        str(guide["training_mode"]),
        "--current-deck-guide-curriculum-spec",
        str(
            _bound_source_path(
                execution_root,
                guide["curriculum_spec"],
                label="r241 curriculum source path",
            )
        ),
        "--current-deck-guide-head-role-map",
        str(
            _bound_source_path(
                execution_root,
                guide["head_role_map"],
                label="r241 head-role-map source path",
            )
        ),
        "--current-deck-guide-curriculum-validation-receipt",
        str(
            _bound_source_path(
                execution_root,
                guide["validation_receipt"],
                label="r241 curriculum-validation source path",
            )
        ),
        "--combo-state-loss-weight",
        "0",
        "--dormant-matchup-adapter-epochs",
        str(preservation.matchup_adapter_epochs_per_update),
        "--dormant-matchup-adapter-activation-receipt",
        str(preservation.adapter_activation_receipt),
        *preservation.trainer_args,
    ]
    forbidden = set(_FORBIDDEN_TRAINER_ARGUMENTS) - {
        "--base-checkpoint",
        "--iterations",
        "--fixed-cycle-updates",
        "--r241-peak-r195-preservation-receipt",
        "--r241-peak-r195-preservation-receipt-sha256",
        "--games-per-iter",
        "--official-collect-frac",
        "--research-control-games-per-iter",
        "--active-gate-contract",
        "--frozen-specialist-registry",
        "--research-control-registry",
        "--expert-manifest",
        "--terminal-expert-rehearsal",
        "--expert-rehearsal-every",
        "--expert-rehearsal-epochs",
        "--combo-state-loss-weight",
        "--current-deck-guide-loss-weight",
        "--current-deck-guide-training-mode",
        "--current-deck-guide-curriculum-spec",
        "--current-deck-guide-head-role-map",
        "--current-deck-guide-curriculum-validation-receipt",
        "--dormant-matchup-adapter-epochs",
        "--dormant-matchup-adapter-activation-receipt",
    }
    if any(token in forbidden for token in command):
        raise R241LaunchError("r241 command contains a forbidden control")
    if int(cycle["updates_exact"]) != 10:
        raise R241LaunchError("registry no longer has exactly ten updates")
    return command


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--host", choices=("inzi", "elmo"), default="inzi")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--static-check",
        action="store_true",
        help="Validate only repository-owned inputs; never touches host receipts.",
    )
    parser.add_argument(
        "--activation-overlay",
        type=Path,
        help=(
            "external create-only activation overlay; required for every "
            "non-static r241 source-snapshot operation"
        ),
    )
    parser.add_argument(
        "--activation-overlay-sha256",
        help="checksum of --activation-overlay",
    )
    parser.add_argument(
        "--activation-overlay-mirror-receipt",
        type=Path,
        help=(
            "host-local create-only receipt proving the canonical overlay and "
            "owner authorization were installed byte-identically"
        ),
    )
    parser.add_argument(
        "--activation-overlay-mirror-receipt-sha256",
        help="checksum of --activation-overlay-mirror-receipt",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all activation receipts and print the exact command without starting it.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly execute the already-validated normal pure-RL launcher.",
    )
    args = parser.parse_args(argv)
    if args.static_check and (args.check or args.execute):
        parser.error("--static-check cannot be combined with --check or --execute")
    if args.check and args.execute:
        parser.error("--check and --execute are mutually exclusive")
    if bool(args.activation_overlay) != bool(args.activation_overlay_sha256):
        parser.error(
            "--activation-overlay and --activation-overlay-sha256 must be provided together"
        )
    if bool(args.activation_overlay_mirror_receipt) != bool(
        args.activation_overlay_mirror_receipt_sha256
    ):
        parser.error(
            "--activation-overlay-mirror-receipt and "
            "--activation-overlay-mirror-receipt-sha256 must be provided together"
        )
    if bool(args.activation_overlay) != bool(args.activation_overlay_mirror_receipt):
        parser.error(
            "r241 activation requires the canonical overlay and its host-local mirror receipt together"
        )
    if args.static_check and (
        args.activation_overlay
        or args.activation_overlay_sha256
        or args.activation_overlay_mirror_receipt
        or args.activation_overlay_mirror_receipt_sha256
    ):
        parser.error("--static-check cannot consume an activation overlay")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry_path, registry = load_registry(args.registry)
    validate_static_registry(registry)
    if args.static_check:
        print("r241 static registry: passed")
        return 0
    if (
        args.activation_overlay is None
        or args.activation_overlay_sha256 is None
        or args.activation_overlay_mirror_receipt is None
        or args.activation_overlay_mirror_receipt_sha256 is None
    ):
        raise R241LaunchError(
            "r241 non-static launch requires --activation-overlay and "
            "--activation-overlay-sha256 plus the host-local mirror receipt; "
            "the snapshot-local registry is intentionally pending"
        )
    registry, overlay = apply_activation_overlay(
        registry,
        registry_path=registry_path,
        overlay_path=args.activation_overlay,
        overlay_sha256=args.activation_overlay_sha256,
        overlay_mirror_receipt=args.activation_overlay_mirror_receipt,
        overlay_mirror_receipt_sha256=args.activation_overlay_mirror_receipt_sha256,
        host=args.host,
    )
    context, preservation = validate_activation(
        registry, host=args.host, activation_overlay=overlay
    )
    command = build_command(registry, context, preservation, python=str(args.python))
    if args.check:
        print(shlex.join(command))
        return 0
    if not args.execute:
        # A no-flag invocation remains side-effect-free even after activation;
        # an operator must expressly request the managed launcher process.
        print(shlex.join(command))
        return 0
    completed = subprocess.run(
        command,
        cwd=preservation.source_snapshot.root,
        env=build_environment(registry, context, preservation=preservation),
    )
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R241LaunchError as exc:
        print(f"r241 launch preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(78)
