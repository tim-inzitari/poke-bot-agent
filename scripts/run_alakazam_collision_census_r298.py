#!/usr/bin/env python3
"""Create-only r298 Phase-A raw-corpus inventory and collision census.

There are deliberately two explicit offline phases:

``manifest``
    validates the owner-selected 30 raw daily ZIPs (2026-07-13 through
    2026-08-11), records every physical/source identity and emits a new
    immutable r298 raw-corpus manifest/receipt.

``census``
    consumes *only* that r298 manifest/receipt, streams every actor-visible
    factorized selection frame through the exact r274 feature ABI, and writes
    bounded bucketed records.  It has no BattleStart, SearchBegin, simulator
    mutation, checkpoint loading, or training path.

The default command does nothing.  A full census requires an explicit ack and
cannot substitute a 20-day receipt, a partial shard, an adjacent window, or a
recollection.  The full pass holds one global Elmo lease in its parent and
assigns the 30 days to exactly 24 private process lanes.  Workers cannot
publish artifacts; the lease-holding parent validates their bounded spools
and performs one deterministic merge into content-addressed output shards.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import resource
import fcntl
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poke_bot.alakazam_collision_census_r298 import (  # noqa: E402
    CANONICAL_R236_LIBCG_SHA256,
    CANONICAL_R236_LIBCG_SIZE_BYTES,
    CollisionCensusError,
    EXACT_NEW_LIST_MULTISET_SHA256,
    MECHANICS_ATTACHMENT_SHA256,
    OWNER_GOAL_SHA256,
    R274_EXACT_FEATURES_SOURCE_SHA256,
    RAW_CORPUS_END_UTC,
    RAW_CORPUS_EXACT_DAYS,
    RAW_CORPUS_START_UTC,
    RAW_CORPUS_SOURCE_RECEIPT_SHA256S,
    R298_COLLISION_RECEIPT_SCHEMA,
    R298_ENGINE_EVIDENCE_SCHEMA,
    R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
    R298_RAW_CORPUS_MANIFEST_SCHEMA,
    R298_RAW_CORPUS_RECEIPT_SCHEMA,
    R298_REFEATURED_RECORD_MANIFEST_SCHEMA,
    R298_REFEATURED_RECORD_SHARD_SCHEMA,
    R298_OWNER_REVISION,
    R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
    PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE,
    PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
    REVISION_4_CONTRACT_SHA256,
    REVISION_4_GATEWAY_SHA256,
    RULE_DERIVATIVE_CONTRACT_SHA256,
    RULE_DERIVATIVE_GATEWAY_SHA256,
    REVISION_5_GOAL_REVISION,
    REVISION_5_ROOT_HANDOFF_REVISION,
    STATUS_FAILED_COLLISION,
    action_key_sha256,
    analyze_collision_records,
    build_raw_corpus_manifest,
    build_stage_option_records,
    canonical_json_bytes,
    canonical_public_observation_hash,
    canonical_sha256,
    frozen_schema_gate_contract,
    inventory_raw_observations,
    make_raw_corpus_receipt,
    make_receipt,
    make_revision_5_census_validation_receipt,
    recorded_episode_frame_coverage,
    raw_observations_from_recorded_episode,
    require_sha256,
    sha256_file,
    stage_descriptors_from_recorded_episode,
    validate_frozen_schema_gate,
    validate_phase_a_inventory,
    validate_raw_corpus_manifest,
    revision_5_predecessor_classification,
    validate_revision_5_census_validation_receipt,
)
from poke_bot.alakazam_rule_derivative_predecessor_compat_rev7 import (  # noqa: E402
    Rev7PredecessorCompatibilityError,
    load_revision_7_contract,
    revision_7_parallel_execution_plan,
    validate_revision_5_census_predecessors_under_revision_7,
    validate_revision_5_census_completion_under_revision_7,
)


CONFIG_SCHEMA = "poke_bot.alakazam_collision_census_r298_config/v3"
RUN_SCHEMA = "poke_bot.alakazam_collision_census_r298_run/v3"
RAW_MANIFEST_RUN_SCHEMA = "poke_bot.alakazam_collision_census_r298_manifest_run/v1"
HARD_EXPERIMENT_MEMORY_BYTES = 96 * 1024**3
DEFAULT_BUCKET_COUNT = 4096
PHASE_A_VALIDATION_WORKERS = 24
MAX_TRANSFER_SHARD_BYTES = 1 * 1024**3
TARGET_TRANSFER_SHARD_BYTES = 1 * 1024**3
# The parent owns one fixed 24 GiB logical RAM-spool envelope.  Each lane is
# allowed one 1 GiB slice shared by its audit and materialized streams; once a
# lane exhausts that slice it transparently spills to its single ZFS lane file.
RAM_SPOOL_TOTAL_BYTES = 24 * 1024**3
RAM_SPOOL_PER_LANE_BYTES = RAM_SPOOL_TOTAL_BYTES // PHASE_A_VALIDATION_WORKERS
RAM_SPOOL_FLUSH_BYTES = 8 * 1024**2
MAX_OPEN_REFEATURE_SHARD_STREAMS = 64
RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE = "all_actor_visible_collision_audit"
RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743 = "acting_seat_card_743_materialized_rows"
RECORD_SCOPES = {
    RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
    RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
}
GOAL_GATEWAY_PATH = REPO_ROOT / "goals/alakazam-elmo-rule-derivative/GOAL.md"
GOAL_CONTRACT_PATH = REPO_ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
DEFAULT_EXPERIMENT_LEASE_ROOT = Path(
    "/mnt/Main/main/poke-bot-agent/outputs/quarantine/alakazam-elmo-rule-derivative/r298-experiment-lease"
)
ELMO_EXECUTION_ROLE = "elmo"
ELMO_CANONICAL_HOSTNAME = "truenas"


class RunnerError(RuntimeError):
    """The offline create-only runner cannot establish a trustworthy input."""


def _verified_elmo_execution_identity() -> dict[str, Any]:
    """Return the only host identity authorized to materialize r298 artifacts.

    The runner does not accept a host/role flag or environment override.  It
    must run on the TrueNAS host whose canonical hostname is exactly
    ``truenas``.  A container is deliberately not an alternate Elmo identity:
    invoke the wrapper on the canonical host instead.  This avoids a container
    with a convenient mount of the archive namespace manufacturing receipts
    that claim Elmo provenance.
    """

    observed_hostname = socket.gethostname().strip().casefold()
    if observed_hostname != ELMO_CANONICAL_HOSTNAME:
        raise RunnerError(
            "r298 --execute is restricted to canonical Elmo host truenas; "
            f"observed hostname is {observed_hostname or '<empty>'}"
        )
    # These are direct, inexpensive detector checks.  Absence/ambiguity is
    # intentionally not treated as permission: the contract names the host,
    # not an arbitrary container that happens to expose its archive mount.
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        raise RunnerError("r298 --execute rejects container execution on canonical Elmo")
    try:
        probe = subprocess.run(
            ["/usr/bin/systemd-detect-virt", "--container"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerError("r298 cannot verify non-container canonical Elmo execution") from exc
    detected_container = probe.stdout.strip().casefold()
    # systemd-detect-virt returns exit 1 and prints "none" on the TrueNAS
    # host.  Any non-none value, a successful detection, or unexpected output
    # is a fail-closed non-host context.
    if probe.returncode != 1 or detected_container not in {"", "none"}:
        raise RunnerError("r298 --execute rejects detected or ambiguous container execution")
    observed_fqdn = socket.getfqdn().strip().casefold()
    return {
        "execution_host_role": ELMO_EXECUTION_ROLE,
        "canonical_execution_hostname": ELMO_CANONICAL_HOSTNAME,
        "execution_hostname": observed_hostname,
        "execution_fqdn": observed_fqdn,
        "host_verification": "exact_socket_hostname_and_systemd_detect_virt_non_container",
        "container_execution_permitted": False,
    }


def _validate_elmo_execution_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a receipt's host binding before a later phase consumes it."""

    if not isinstance(identity, Mapping):
        raise RunnerError("receipt lacks an execution identity")
    expected = {
        "execution_host_role": ELMO_EXECUTION_ROLE,
        "canonical_execution_hostname": ELMO_CANONICAL_HOSTNAME,
        "execution_hostname": ELMO_CANONICAL_HOSTNAME,
        "host_verification": "exact_socket_hostname_and_systemd_detect_virt_non_container",
        "container_execution_permitted": False,
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise RunnerError(f"receipt execution identity {field} is not canonical Elmo")
    fqdn = identity.get("execution_fqdn")
    if not isinstance(fqdn, str):
        raise RunnerError("receipt execution identity FQDN is malformed")
    return {str(key): value for key, value in identity.items()}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RunnerError(f"{field} must be an exact integer >= {minimum}")
    return int(value)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise RunnerError(f"JSON root is not an object: {path}")
    return value


def _read_stable_regular_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    """Read one immutable evidence object without following a symlink.

    A post-census bridge is a receipt consumer.  It must not accept a path
    that changes while the object is parsed, nor a symlink that can later be
    redirected to a different receipt.  The returned digest is the physical
    file SHA-256; canonical object identities remain owned by the typed
    compatibility validator.
    """

    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"{label} must be a regular non-symlink JSON file")
    try:
        before = path.stat()
        digest = sha256_file(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        after = path.stat()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise RunnerError(f"{label} root is not an object: {path}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(path) != digest:
        raise RunnerError(f"{label} changed while inspected: {path}")
    return value, digest


def _read_protected_source_json(path: Path) -> Mapping[str, Any]:
    """Read an immutable source receipt, using ``sudo -n`` only if required.

    The raw ZIPs are world-readable on Elmo; the two owner-supplied source
    receipts are intentionally protected.  This fallback performs a single
    read-only command with no shell interpolation, no directory traversal,
    and no write/service authority.  It keeps the experiment process and its
    create-only output user-owned rather than broadly running the whole pass
    as root.
    """

    try:
        return _read_json(path)
    except RunnerError as initial_error:
        probe = subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/bin/python3",
                "-c",
                "import pathlib, sys; sys.stdout.write(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode != 0:
            raise initial_error
        try:
            value = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"protected source receipt is not valid JSON: {path}") from exc
        if not isinstance(value, Mapping):
            raise RunnerError(f"protected source receipt root is not an object: {path}")
        return value


def _readonly_sha256_file(path: Path) -> str:
    """Hash a source path directly, with the same narrow protected-read fallback."""

    try:
        return sha256_file(path)
    except CollisionCensusError as initial_error:
        probe = subprocess.run(
            ["sudo", "-n", "/usr/bin/sha256sum", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode != 0:
            raise initial_error
        fields = probe.stdout.strip().split()
        if not fields or len(fields[0]) != 64 or any(char not in "0123456789abcdef" for char in fields[0]):
            raise RunnerError(f"protected source digest output is malformed: {path}")
        return "sha256:" + fields[0]


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise RunnerError(f"create-only output already exists: {path}") from exc
    except OSError as exc:
        raise RunnerError(f"cannot create output: {path}") from exc


def _load_config(path: Path) -> Mapping[str, Any]:
    _verify_canonical_goal_bindings()
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise RunnerError("collision census config schema drifted")
    checks: tuple[tuple[str, Any], ...] = (
        ("owner_revision", R298_OWNER_REVISION),
        ("goal_revision", REVISION_5_GOAL_REVISION),
        ("root_handoff_revision", REVISION_5_ROOT_HANDOFF_REVISION),
        ("status", "revision_5_migrated_create_only_elmo_only_off"),
        ("enabled", False),
        ("runtime_active", False),
        ("host_scope", "elmo_only"),
        ("execution_host_role", ELMO_EXECUTION_ROLE),
        ("canonical_execution_hostname", ELMO_CANONICAL_HOSTNAME),
        ("container_execution_permitted", False),
        ("production_authority", False),
        ("inzi_mutation", False),
        ("training_eligible", False),
        ("owner_goal_sha256", OWNER_GOAL_SHA256),
        ("rule_derivative_contract_sha256", RULE_DERIVATIVE_CONTRACT_SHA256),
        ("rule_derivative_goal_sha256", RULE_DERIVATIVE_GATEWAY_SHA256),
        ("mechanics_attachment_sha256", MECHANICS_ATTACHMENT_SHA256),
        ("r274_features_source_sha256", R274_EXACT_FEATURES_SOURCE_SHA256),
        ("canonical_libcg_sha256", CANONICAL_R236_LIBCG_SHA256),
    )
    for field, expected in checks:
        if config.get(field) != expected:
            raise RunnerError(f"collision census config {field} drifted")
    try:
        if config.get("revision_5_predecessor_classification") != revision_5_predecessor_classification():
            raise RunnerError("collision census config predecessor classification drifted")
    except CollisionCensusError as exc:  # pragma: no cover - constant-only builder
        raise RunnerError("collision census config predecessor classification is invalid") from exc
    raw = config.get("raw_expert_corpus")
    if not isinstance(raw, Mapping) or (
        raw.get("window_start_utc"),
        raw.get("window_end_utc"),
        raw.get("distinct_utc_day_count"),
        raw.get("twenty_day_fallback_permitted"),
        raw.get("adjacent_window_fallback_permitted"),
        raw.get("recollection_authorized"),
    ) != (
        RAW_CORPUS_START_UTC,
        RAW_CORPUS_END_UTC,
        RAW_CORPUS_EXACT_DAYS,
        False,
        False,
        False,
    ):
        raise RunnerError("collision census config does not bind the exact 30-day raw corpus")
    if raw.get("episode_deduplication_required") is not True or set(raw.get("source_receipts", ())) != set(
        RAW_CORPUS_SOURCE_RECEIPT_SHA256S
    ):
        raise RunnerError("collision census config does not bind the exact raw source receipts/dedup proof")
    phase_a = config.get("phase_a_schema_inventory")
    if not isinstance(phase_a, Mapping) or (
        phase_a.get("raw_replay_inventory_scope"),
        phase_a.get("actor_visible_selection_inventory_scope"),
        phase_a.get("all_raw_two_seat_outer_observations_required"),
        phase_a.get("all_actor_visible_and_forced_selection_frames_required"),
        phase_a.get("raw_values_persisted"),
        phase_a.get("rejected_raw_observations_permitted"),
    ) != (
        PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
        PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE,
        True,
        True,
        False,
        False,
    ):
        raise RunnerError("collision census config Phase A raw/schema inventory contract drifted")
    resources = config.get("resource_envelope")
    if not isinstance(resources, Mapping) or resources.get("hard_memory_bytes") != HARD_EXPERIMENT_MEMORY_BYTES:
        raise RunnerError("collision census resource envelope drifted")
    if resources.get("one_memory_heavy_phase_at_a_time") is not True:
        raise RunnerError("collision census must serialize memory-heavy phases")
    if resources.get("zfs_arc_l2arc_tuning_authorized") is not False:
        raise RunnerError("collision census must not tune ZFS/ARC/L2ARC")
    limits = config.get("limits")
    if not isinstance(limits, Mapping) or limits.get("bucket_count") != DEFAULT_BUCKET_COUNT:
        raise RunnerError("collision census config must bind the r298 bounded-shard bucket fanout")
    transfer = config.get("transfer_staging")
    if not isinstance(transfer, Mapping) or (
        transfer.get("this_runner_performs_inzi_transfer"),
        transfer.get("completed_shards_are_quarantined_staging_eligible_only"),
        transfer.get("artifact_shard_maximum_bytes"),
        transfer.get("content_addressed_filename_required"),
        transfer.get("later_transfer_parallel_lane_maximum"),
        transfer.get("runtime_or_training_use_authorized"),
    ) != (False, True, MAX_TRANSFER_SHARD_BYTES, True, 4, False):
        raise RunnerError("collision census transfer-safe artifact contract drifted")
    collision_contract = config.get("collision_contract")
    if not isinstance(collision_contract, Mapping) or (
        collision_contract.get("record_raw_option_payload"),
        collision_contract.get("raw_option_payload_audit_only"),
        collision_contract.get("semantic_identity_uses_normalized_r298_slot_bindings"),
    ) != (True, True, True):
        raise RunnerError("collision census config permits raw payloads in semantic identity")
    return config


def _verify_canonical_goal_bindings() -> None:
    """Authenticate current rev7 authority without retagging r5 schemas.

    The census record/receipt schemas were frozen under revision 5.  Revision
    7 explicitly reuses those bytes through a consumer-side compatibility
    bridge, rather than pretending the historic r5 identities changed.  The
    bridge therefore validates the current gateway/contract while this runner
    continues to validate the frozen r5 config and artifacts separately.
    """

    try:
        load_revision_7_contract()
    except Rev7PredecessorCompatibilityError as exc:
        raise RunnerError("current revision-7 goal/contract binding is unavailable") from exc


def _validate_revision_7_census_predecessor_inputs(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    frozen_schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
) -> dict[str, str]:
    """Check only evidence that predates a revision-5 census pass.

    This is intentionally separate from the stricter materialization
    predecessor validator: the historical 24-lane census did not consume a
    materialization preflight, so manufacturing one later would falsely alter
    its launch boundary.  The returned view is in-memory only and is never
    inserted into a frozen r5 census receipt.
    """

    try:
        load_revision_7_contract()
        return validate_revision_5_census_predecessors_under_revision_7(
            raw_manifest=raw_manifest,
            raw_receipt=raw_receipt,
            schema_manifest=frozen_schema_manifest,
            zero_bypass_receipt=zero_bypass_receipt,
        )
    except Rev7PredecessorCompatibilityError as exc:
        raise RunnerError("revision-7 census predecessor validation failed") from exc


class _ResourceTelemetry:
    """Bounded per-process resource accounting; no host tuning or services."""

    def __init__(self, *, probe_gpu: bool = True) -> None:
        # Day workers are CPU/IO-only.  Probing ``nvidia-smi`` from all 24
        # children would add needless subprocess contention and would not
        # improve the parent-held experiment resource proof.  The parent
        # retains the GPU probe; workers retain independent RSS/CPU/IO facts.
        self._probe_gpu = probe_gpu
        self._start_wall = time.perf_counter()
        start = resource.getrusage(resource.RUSAGE_SELF)
        self._start_cpu = start.ru_utime + start.ru_stime
        self._start_io = self._proc_io()
        self._peak_rss_bytes = self._rss_bytes()
        self._peak_gpu_bytes = 0
        self._peak_gpu_bytes_by_device: dict[str, int] = {}
        self._peak_gpu_utilization_percent_by_device: dict[str, int] = {}
        self._gpu_probe_available = False
        self._gpu_device_probe_available = False
        self.sample()

    @staticmethod
    def _proc_io() -> dict[str, int] | None:
        path = Path("/proc/self/io")
        if not path.is_file():
            return None
        values: dict[str, int] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                if key in {"read_bytes", "write_bytes"}:
                    values[key] = int(raw.strip())
        except (OSError, ValueError):
            return None
        return values

    @staticmethod
    def _rss_bytes() -> int:
        # Linux ru_maxrss is KiB.  Elmo is Linux; the fallback remains explicit
        # rather than silently claiming a cross-platform byte unit.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    @staticmethod
    def _own_gpu_telemetry() -> tuple[bool, dict[str, int], dict[str, int]]:
        """Read own GPU memory and device utilization without changing GPU state."""

        try:
            app_probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            device_probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return False, {}, {}
        if app_probe.returncode != 0 or device_probe.returncode != 0:
            return False, {}, {}
        bytes_by_device: dict[str, int] = {}
        current_pid = os.getpid()
        for line in app_probe.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                if int(fields[0]) == current_pid:
                    uuid = fields[1]
                    if uuid:
                        bytes_by_device[uuid] = bytes_by_device.get(uuid, 0) + int(fields[2]) * 1024 * 1024
            except ValueError:
                continue
        utilization_by_device: dict[str, int] = {}
        for line in device_probe.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 2 or not fields[0]:
                continue
            try:
                utilization_by_device[fields[0]] = int(fields[1])
            except ValueError:
                continue
        return True, bytes_by_device, utilization_by_device

    def sample(self) -> None:
        self._peak_rss_bytes = max(self._peak_rss_bytes, self._rss_bytes())
        available, bytes_by_device, utilization_by_device = (
            self._own_gpu_telemetry() if self._probe_gpu else (False, {}, {})
        )
        self._gpu_probe_available = self._gpu_probe_available or available
        self._gpu_device_probe_available = self._gpu_device_probe_available or available
        self._peak_gpu_bytes = max(self._peak_gpu_bytes, sum(bytes_by_device.values()))
        for device, used in bytes_by_device.items():
            self._peak_gpu_bytes_by_device[device] = max(
                self._peak_gpu_bytes_by_device.get(device, 0), used
            )
        for device, utilization in utilization_by_device.items():
            self._peak_gpu_utilization_percent_by_device[device] = max(
                self._peak_gpu_utilization_percent_by_device.get(device, 0), utilization
            )
        if self._peak_rss_bytes > HARD_EXPERIMENT_MEMORY_BYTES:
            raise RunnerError("r298 process exceeded the hard 96 GiB experiment memory ceiling")

    def final(self) -> dict[str, Any]:
        self.sample()
        now = resource.getrusage(resource.RUSAGE_SELF)
        io_now = self._proc_io()
        io_delta: dict[str, int | None] = {}
        for key in ("read_bytes", "write_bytes"):
            io_delta[key] = (
                int(io_now.get(key, 0)) - int(self._start_io.get(key, 0))
                if io_now is not None and self._start_io is not None
                else None
            )
        return {
            "hard_memory_ceiling_bytes": HARD_EXPERIMENT_MEMORY_BYTES,
            "actual_peak_rss_bytes": self._peak_rss_bytes,
            "actual_peak_gpu_bytes": self._peak_gpu_bytes,
            "actual_peak_gpu_bytes_by_device": dict(sorted(self._peak_gpu_bytes_by_device.items())),
            "actual_peak_gpu_utilization_percent_by_device": dict(
                sorted(self._peak_gpu_utilization_percent_by_device.items())
            ),
            "gpu_process_telemetry_available": self._gpu_probe_available,
            "gpu_device_telemetry_available": self._gpu_device_probe_available,
            "actual_cpu_seconds": (now.ru_utime + now.ru_stime) - self._start_cpu,
            "actual_wall_seconds": time.perf_counter() - self._start_wall,
            "actual_io_bytes": io_delta,
            "managed_service_change": False,
            "zfs_arc_l2arc_tuning": False,
        }


class _ExperimentLease:
    """An advisory Elmo-only exclusive lease for one memory-heavy r298 pass.

    The file is an explicit experiment-control artifact in one fixed r298
    Elmo namespace, independent of a caller-selected output root.  It is not
    a service or system resource setting.  A second r298 phase cannot enter
    while the lock is held.  The parent may supervise private day workers, but
    they never acquire a second lease or publish artifacts; the parent binds
    their conservative aggregate RAM charge to the one-heavy-phase contract.
    """

    def __init__(self, root: Path) -> None:
        self.path = root / ".r298-memory-heavy-phase.lock"
        self._stream: Any | None = None

    def __enter__(self) -> "_ExperimentLease":
        # The lease namespace is one fixed Elmo experiment directory, never a
        # per-run output path.  A caller may create it once if absent, but a
        # symlink or non-directory is rejected rather than followed.  We do
        # not truncate/replace an existing lock path before acquiring flock;
        # the lock's content is diagnostic only and cannot be used to steal
        # another process's lease.
        parent = self.path.parent
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise RunnerError("r298 global experiment lease root is not a real directory")
        else:
            parent.mkdir(parents=True, exist_ok=False)
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_file()):
            raise RunnerError("r298 global experiment lease path is not a real file")
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            raise RunnerError("another r298 memory-heavy phase already holds the experiment lease") from exc
        self._stream.write(json.dumps({"pid": os.getpid(), "acquired_at_utc": _utc()}) + "\n")
        self._stream.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


def _experiment_lease_root() -> Path:
    """Resolve the single global r298 Elmo memory-phase lease namespace."""

    if not DEFAULT_EXPERIMENT_LEASE_ROOT.is_absolute():
        raise RunnerError("r298 global experiment lease root must be absolute")
    return DEFAULT_EXPERIMENT_LEASE_ROOT


def _verify_cg_runtime(runtime_root: Path) -> None:
    root = runtime_root.resolve()
    library = root / "cg" / "libcg.so"
    if not library.is_file():
        raise RunnerError("CG runtime root lacks cg/libcg.so")
    if library.stat().st_size != CANONICAL_R236_LIBCG_SIZE_BYTES:
        raise RunnerError("CG runtime library size differs from canonical r236")
    if sha256_file(library) != CANONICAL_R236_LIBCG_SHA256:
        raise RunnerError("CG runtime library digest differs from canonical r236")
    os.environ["CG_LIB_PATH"] = str(root)


def _token_abi_builder() -> Any:
    from poke_bot import features

    source_path = Path(features.__file__).resolve()
    if sha256_file(source_path) != R274_EXACT_FEATURES_SOURCE_SHA256:
        raise RunnerError("imported features.py does not match the exact r274 feature ABI")
    return features.build_option_tokens


def _expected_dates() -> list[str]:
    return [f"2026-07-{day:02d}" for day in range(13, 32)] + [
        f"2026-08-{day:02d}" for day in range(1, 12)
    ]


def _reference_archive_rows(paths: Sequence[Path]) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    """Merge the two immutable source receipts without treating richer rows as conflicts.

    July 23 appears in both published windows.  Its immutable ZIP identity
    must agree, while the newer receipt legitimately adds validation and
    discrepancy fields.  The merged row retains *both* receipt identities so
    later consumers can prove per-day source coverage instead of merely
    knowing that two receipts were present somewhere in a manifest.
    """

    by_date: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    seen_receipts: set[str] = set()
    immutable_identity_fields = (
        "date",
        "dataset_slug",
        "path",
        "sha256",
        "bytes",
        "episode_count",
        "validated",
    )
    for path in paths:
        receipt = _read_protected_source_json(path)
        archives = receipt.get("archives")
        if not isinstance(archives, list):
            raise RunnerError(f"reference receipt lacks archives: {path}")
        digest = _readonly_sha256_file(path)
        if digest not in RAW_CORPUS_SOURCE_RECEIPT_SHA256S or digest in seen_receipts:
            raise RunnerError("reference receipts must be the exact two owner-selected immutable sources")
        if receipt.get("status") != "ready":
            raise RunnerError(f"reference receipt is not ready: {path}")
        seen_receipts.add(digest)
        receipt_dates: list[str] = []
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "schema": receipt.get("schema"),
                "status": receipt.get("status"),
                "archive_dates": receipt_dates,
            }
        )
        for archive in archives:
            if not isinstance(archive, Mapping) or not isinstance(archive.get("date"), str):
                raise RunnerError(f"reference receipt has malformed archive: {path}")
            date = archive["date"]
            receipt_dates.append(date)
            for field in immutable_identity_fields:
                if field not in archive:
                    raise RunnerError(f"reference receipt has incomplete archive identity: {path}:{date}")
            previous = by_date.get(date)
            if previous is None:
                merged = dict(archive)
                merged["source_receipt_sha256s"] = [digest]
                by_date[date] = merged
                continue
            for field in immutable_identity_fields:
                if previous.get(field) != archive.get(field):
                    raise RunnerError(f"reference receipts conflict for raw day {date}:{field}")
            merged = dict(previous)
            for field, value in archive.items():
                if field not in merged or merged[field] is None:
                    merged[field] = value
                elif value is not None and merged[field] != value:
                    # Optional fields, when present in both receipts, are also
                    # immutable facts and cannot be silently chosen by age.
                    raise RunnerError(f"reference receipts conflict for raw day {date}:{field}")
            merged["source_receipt_sha256s"] = sorted(
                set(previous["source_receipt_sha256s"]) | {digest}
            )
            by_date[date] = merged
    if seen_receipts != set(RAW_CORPUS_SOURCE_RECEIPT_SHA256S):
        raise RunnerError("reference receipt set is incomplete or has an unauthorized source")
    return by_date, provenance


def _validate_raw_zip(path: Path, *, telemetry: _ResourceTelemetry | None) -> tuple[str, int, int]:
    if not path.is_file():
        raise RunnerError(f"raw daily ZIP is missing: {path}")
    digest = sha256_file(path)
    byte_count = path.stat().st_size
    try:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise RunnerError(f"raw ZIP has duplicate member names: {path.name}")
            json_members = [
                name for name in names
                if name.endswith(".json") and not name.endswith("/")
            ]
            auxiliary_members = sorted(set(names) - set(json_members))
            # The canonical daily exporter includes one value-free member
            # index alongside the episode JSON files.  It is not an episode
            # and is never fed to re-featurization; any other extra member is
            # still rejected.
            if not json_members or auxiliary_members not in ([], ["manifest.csv"]):
                raise RunnerError(f"raw ZIP has non-episode or missing JSON members: {path.name}")
            bad = bundle.testzip()
            if bad is not None:
                raise RunnerError(f"raw ZIP CRC validation failed at {path.name}:{bad}")
    except zipfile.BadZipFile as exc:
        raise RunnerError(f"raw daily source is not a ZIP: {path}") from exc
    if telemetry is not None:
        telemetry.sample()
    return digest, byte_count, len(json_members)


def _raw_archive_rows(
    archive_root: Path,
    *,
    reference_rows: Mapping[str, Mapping[str, Any]],
    telemetry: _ResourceTelemetry,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    member_counts: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    dates = _expected_dates()

    for date in dates:
        name = f"pokemon-tcg-ai-battle-episodes-{date}.zip"
        path = archive_root / name
        reference = reference_rows.get(date)
        if reference is None:
            raise RunnerError(f"both canonical source receipts fail to cover required raw day {date}")
        if not path.is_file():
            raise RunnerError(f"raw daily ZIP is missing: {path}")
        digest = require_sha256(reference.get("sha256"), field=f"reference archive digest {date}")
        byte_count = _exact_int(
            reference.get("bytes"), field=f"reference archive bytes {date}", minimum=1
        )
        if reference.get("validated") is not True:
            raise RunnerError(f"raw ZIP source receipt did not validate day {date}")
        source_discrepancy = reference.get("source_discrepancy")
        index_count = reference.get("index_episode_count", reference.get("episode_count"))
        expected_validated_count = reference.get(
            "validated_episode_count",
            reference.get("episode_count"),
        )
        member_count = _exact_int(
            expected_validated_count,
            field=f"source validated member count {date}",
            minimum=1,
        )
        dataset_slug = reference.get("dataset_slug")
        if not isinstance(dataset_slug, str):
            raise RunnerError(f"reference source lacks dataset slug for {date}")
        rows.append(
            {
                "date": date,
                "dataset_slug": dataset_slug,
                "path": str(path.resolve()),
                "sha256": digest,
                "bytes": byte_count,
                "validated": True,
                "validated_episode_count": member_count,
                "index_episode_count": _exact_int(index_count, field=f"source index count {date}", minimum=1),
                "source_discrepancy": source_discrepancy,
                "source_receipt_sha256s": list(reference.get("source_receipt_sha256s", ())),
            }
        )
        member_counts[digest] = member_count
        byte_counts[digest] = byte_count
    return rows, member_counts, byte_counts


def _scan_episode_identities(row: Mapping[str, Any]) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Decode one day independently for the bounded parallel dedup pass."""

    path = Path(str(row["path"]))
    values: list[tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(path) as bundle:
            members = sorted(
                name for name in bundle.namelist()
                if name.endswith(".json") and not name.endswith("/")
            )
            for member in members:
                try:
                    payload = json.loads(bundle.read(member))
                except (KeyError, OSError, json.JSONDecodeError) as exc:
                    raise RunnerError(f"cannot decode raw episode for dedup: {path.name}:{member}") from exc
                if not isinstance(payload, Mapping):
                    raise RunnerError(f"raw episode is not an object for dedup: {path.name}:{member}")
                episode_id = payload.get("id")
                if type(episode_id) is not int and not isinstance(episode_id, str):
                    raise RunnerError(f"raw episode has no stable payload.id: {path.name}:{member}")
                values.append((member, str(episode_id), canonical_sha256(payload)))
    except zipfile.BadZipFile as exc:
        raise RunnerError(f"raw ZIP invalid during dedup: {path}") from exc
    return str(row["date"]), str(row["sha256"]), values


def _episode_deduplication_proof(
    rows: Sequence[Mapping[str, Any]],
    *,
    telemetry: _ResourceTelemetry,
) -> dict[str, Any]:
    """Prove identity/content uniqueness across all thirty raw ZIPs.

    The inventory stores only digests and source coordinates, never raw episode
    bodies.  A repeated ``payload.id`` with changed contents is a hard failure;
    a byte-identical repeated identity is also a hard failure for this exact
    corpus instead of a silent drop.  That makes the manifest unambiguous for
    later source-disjoint splits.
    """

    seen_identity: dict[tuple[str, str], tuple[str, str]] = {}
    seen_id: dict[str, tuple[str, str]] = {}
    inventory_digest = hashlib.sha256()
    duplicate_identity_count = 0
    duplicate_id_changed_count = 0
    observed_member_count = 0
    # Episode JSON decoding and canonical hashing are CPU-heavy Python work.
    # Threads leave most Elmo cores idle because of the GIL, so use bounded
    # worker processes here.  Each worker owns only one day's decoded rows at
    # a time and the experiment-wide 96 GiB envelope remains authoritative.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(PHASE_A_VALIDATION_WORKERS, len(rows)),
    ) as pool:
        scans = list(pool.map(_scan_episode_identities, rows))
    for date, archive_sha, values in scans:
        for member, episode_id_text, content_sha in values:
                    observed_member_count += 1
                    identity = (episode_id_text, content_sha)
                    coordinate = (date, member)
                    if identity in seen_identity:
                        duplicate_identity_count += 1
                    prior_id = seen_id.get(episode_id_text)
                    if prior_id is not None and prior_id[0] != content_sha:
                        duplicate_id_changed_count += 1
                    seen_identity[identity] = coordinate
                    seen_id[episode_id_text] = (content_sha, f"{date}:{member}")
                    inventory_digest.update(
                        canonical_json_bytes(
                            {
                                "date": date,
                                "archive_sha256": archive_sha,
                                "member": member,
                                "payload_id": episode_id_text,
                                "canonical_content_sha256": content_sha,
                            }
                        )
                    )
        telemetry.sample()
    if duplicate_identity_count or duplicate_id_changed_count:
        raise RunnerError("raw 30-day corpus has duplicate episode identities or mutated repeated IDs")
    expected_member_count = sum(
        _exact_int(row.get("validated_episode_count"), field="raw archive validated count", minimum=1)
        for row in rows
    )
    if observed_member_count != expected_member_count:
        raise RunnerError("episode-level deduplication did not scan every validated ZIP member")
    return {
        "episode_identity_algorithm": "payload.id_plus_canonical_content_sha256",
        "unique_episode_identity_count": len(seen_identity),
        "unique_episode_id_count": len(seen_id),
        "duplicate_episode_identity_count": duplicate_identity_count,
        "duplicate_episode_id_with_distinct_content_count": duplicate_id_changed_count,
        "excluded_duplicate_mapping": [],
        "raw_zip_member_count_observed": observed_member_count,
        "episode_identity_inventory_sha256": "sha256:" + inventory_digest.hexdigest(),
    }


def _trusted_source_receipt_episode_proof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use existing daily validation receipts without a redundant payload scan."""

    total = sum(
        _exact_int(row.get("validated_episode_count"), field="trusted validated episode count", minimum=1)
        for row in rows
    )
    identity = canonical_sha256(
        [
            {
                "date": row.get("date"),
                "archive_sha256": row.get("sha256"),
                "validated_episode_count": row.get("validated_episode_count"),
                "source_receipt_sha256s": row.get("source_receipt_sha256s"),
            }
            for row in rows
        ]
    )
    return {
        "episode_identity_algorithm": "trusted_existing_daily_validation_receipts_owner_authorized",
        "unique_episode_identity_count": total,
        "unique_episode_id_count": total,
        "duplicate_episode_identity_count": 0,
        "duplicate_episode_id_with_distinct_content_count": 0,
        "excluded_duplicate_mapping": [],
        "raw_zip_member_count_observed": total,
        "episode_identity_inventory_sha256": identity,
        "fresh_episode_payload_scan_performed": False,
        "existing_daily_validation_receipts_trusted": True,
    }


def _validate_raw_corpus_binding(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    try:
        validate_raw_corpus_manifest(manifest)
    except CollisionCensusError as exc:
        raise RunnerError("census must consume a strict r298 raw corpus manifest") from exc
    if receipt.get("schema") != R298_RAW_CORPUS_RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise RunnerError("census must consume a passed r298 raw corpus receipt")
    if receipt.get("raw_expert_corpus_manifest_sha256") != canonical_sha256(manifest):
        raise RunnerError("raw corpus receipt does not bind the supplied manifest")
    receipt_identity = _validate_elmo_execution_identity(receipt.get("execution_identity"))
    resource_observation = receipt.get("resource_observation")
    if not isinstance(resource_observation, Mapping):
        raise RunnerError("raw corpus receipt lacks resource observation")
    if _validate_elmo_execution_identity(
        resource_observation.get("execution_identity")
    ) != receipt_identity:
        raise RunnerError("raw corpus receipt/resource execution identity disagrees")
    episode_dedup = manifest.get("episode_deduplication")
    assert isinstance(episode_dedup, Mapping)
    source_provenance = manifest.get("source_manifest_provenance")
    assert isinstance(source_provenance, Sequence)
    source_coverage = manifest.get("source_receipt_day_coverage")
    assert isinstance(source_coverage, Sequence)
    if receipt.get("source_manifest_provenance_sha256") != canonical_sha256(source_provenance):
        raise RunnerError("raw corpus receipt does not bind source receipt provenance")
    if receipt.get("source_receipt_day_coverage_sha256") != canonical_sha256(source_coverage):
        raise RunnerError("raw corpus receipt does not bind exact per-day source receipt coverage")
    if receipt.get("episode_deduplication_sha256") != canonical_sha256(episode_dedup):
        raise RunnerError("raw corpus receipt does not bind episode-level dedup proof")
    if (
        receipt.get("completed_raw_zip_member_count"),
        receipt.get("completed_validated_episode_count"),
    ) != (
        manifest.get("total_raw_zip_json_members"),
        manifest.get("total_validated_episodes"),
    ):
        raise RunnerError("raw corpus receipt counts do not bind the complete manifest")
    disjoint = receipt.get("source_disjointness")
    if not isinstance(disjoint, Mapping) or any(
        disjoint.get(name) is not expected
        for name, expected in {
            "archive_date_source_sha256_unique": True,
            "episode_identity_unique": True,
            "episode_id_content_unique": True,
            "source_window_blending_permitted": False,
            "training_eligible": False,
        }.items()
    ):
        raise RunnerError("raw corpus receipt source-disjointness proof drifted")
    for field, expected in {
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "recollection_authorized": False,
    }.items():
        if receipt.get(field) != expected:
            raise RunnerError(f"raw corpus receipt {field} drifted")
    if receipt.get("revision_5_predecessor_classification") != revision_5_predecessor_classification():
        raise RunnerError("raw corpus receipt predecessor classification drifted")
    _verify_manifest_source_receipts(manifest)


def _verify_manifest_source_receipts(manifest: Mapping[str, Any]) -> None:
    """Re-open immutable source receipts and prove every manifest day against them."""

    provenance = manifest.get("source_manifest_provenance")
    coverage = manifest.get("source_receipt_day_coverage")
    archives = manifest.get("archives")
    if not isinstance(provenance, Sequence) or not isinstance(coverage, Sequence) or not isinstance(archives, Sequence):
        raise RunnerError("raw corpus manifest has malformed source receipt bindings")
    archives_by_date = {
        row.get("date"): row for row in archives if isinstance(row, Mapping) and isinstance(row.get("date"), str)
    }
    sources: dict[str, Mapping[str, Any]] = {}
    for raw in provenance:
        if not isinstance(raw, Mapping):
            raise RunnerError("raw corpus source provenance row is malformed")
        digest = require_sha256(raw.get("sha256"), field="raw corpus source receipt digest")
        path = Path(str(raw.get("path", "")))
        if not path.is_file() or _readonly_sha256_file(path) != digest:
            raise RunnerError("raw corpus source receipt physical identity drifted")
        receipt = _read_protected_source_json(path)
        if receipt.get("schema") != raw.get("schema") or receipt.get("status") != "ready":
            raise RunnerError("raw corpus source receipt schema/status drifted")
        rows = receipt.get("archives")
        if not isinstance(rows, Sequence):
            raise RunnerError("raw corpus source receipt lacks archive rows")
        sources[digest] = {
            row.get("date"): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("date"), str)
        }
    if set(sources) != set(RAW_CORPUS_SOURCE_RECEIPT_SHA256S):
        raise RunnerError("raw corpus source receipt physical set drifted")
    for row in coverage:
        if not isinstance(row, Mapping):
            raise RunnerError("raw corpus day source coverage row is malformed")
        date = row.get("date")
        archive = archives_by_date.get(date)
        if not isinstance(date, str) or not isinstance(archive, Mapping):
            raise RunnerError("raw corpus day source coverage lacks manifest archive")
        values = row.get("source_receipt_sha256s")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise RunnerError("raw corpus day source coverage lacks source identities")
        for raw_digest in values:
            digest = require_sha256(raw_digest, field="raw corpus covered source receipt")
            source_row = sources.get(digest, {}).get(date)
            if not isinstance(source_row, Mapping):
                raise RunnerError("raw corpus source receipt does not actually cover its declared day")
            # Manifest rows are emitted only after validated=True was checked;
            # the normalized row deliberately does not duplicate that source
            # receipt field.
            for field in ("date", "dataset_slug", "path", "sha256", "bytes"):
                if source_row.get(field) != archive.get(field):
                    raise RunnerError(f"raw corpus source receipt/archive mismatch for {date}:{field}")


def _manifest_archives(manifest: Mapping[str, Any], *, archive_root: Path | None) -> list[tuple[Mapping[str, Any], Path]]:
    archives = manifest.get("archives")
    assert isinstance(archives, list)
    result: list[tuple[Mapping[str, Any], Path]] = []
    for archive in archives:
        if not isinstance(archive, Mapping):
            raise RunnerError("raw manifest archive row is malformed")
        raw_path = archive.get("path")
        if not isinstance(raw_path, str):
            raise RunnerError("raw manifest archive path is malformed")
        path = Path(raw_path)
        if archive_root is not None:
            path = archive_root / path.name
        result.append((archive, path))
    return result


def _validate_revision_7_parallel_day_plan(
    plan: Mapping[str, Any],
    archives: Sequence[tuple[Mapping[str, Any], Path]],
) -> None:
    """Bind the parent-owned lanes to the exact rev7 thirty-day schedule.

    The compatibility module supplies the authority-owned schedule.  This
    runner still derives the schedule from the consumed manifest and requires
    byte-for-byte agreement, so a stale/foreign manifest cannot be processed
    by a superficially valid 24-worker launch.
    """

    expected_dates = _expected_dates()
    archive_dates: list[str] = []
    for archive, _path in archives:
        date = archive.get("date")
        if not isinstance(date, str):
            raise RunnerError("parallel manifest archive lacks a UTC date")
        archive_dates.append(date)
    if archive_dates != expected_dates:
        raise RunnerError("parallel manifest archive dates do not match the exact 30-day window")
    expected_lanes = [
        expected_dates[lane_index::PHASE_A_VALIDATION_WORKERS]
        for lane_index in range(PHASE_A_VALIDATION_WORKERS)
    ]
    if (
        plan.get("worker_count"),
        plan.get("utc_partition_count"),
        plan.get("window_start_utc"),
        plan.get("window_end_utc"),
        plan.get("day_lanes"),
        plan.get("one_memory_heavy_phase_at_a_time"),
        plan.get("experiment_ram_ceiling_bytes"),
    ) != (
        PHASE_A_VALIDATION_WORKERS,
        len(expected_dates),
        RAW_CORPUS_START_UTC,
        RAW_CORPUS_END_UTC,
        expected_lanes,
        True,
        HARD_EXPERIMENT_MEMORY_BYTES,
    ):
        raise RunnerError("revision-7 24-worker UTC-day plan drifted from the supplied corpus")


def _verify_manifest_archive(archive: Mapping[str, Any], path: Path, *, telemetry: _ResourceTelemetry) -> None:
    # The owner explicitly accepts the existing daily validation receipts as
    # archive-integrity authority.  Avoid a second full ZIP hash/CRC pass here;
    # the re-featurizer will naturally decode every selected JSON member.
    require_sha256(archive.get("sha256"), field=f"manifest archive digest {path.name}")
    if not path.is_file():
        raise RunnerError(f"raw manifest archive is missing: {path}")
    expected_bytes = _exact_int(archive.get("bytes"), field=f"manifest archive bytes {path.name}", minimum=1)
    if path.stat().st_size != expected_bytes:
        raise RunnerError(f"raw manifest archive byte count drifted: {path}")
    telemetry.sample()


def _iter_episode_payloads(
    archives: Sequence[tuple[Mapping[str, Any], Path]],
    *,
    day_shard_index: int,
    day_shard_count: int,
    max_episodes: int | None,
    telemetry: _ResourceTelemetry,
) -> Iterable[tuple[Mapping[str, Any], str, Mapping[str, Any]]]:
    emitted = 0
    inner_episode_sharding = day_shard_count > len(archives)
    for day_index, (archive, path) in enumerate(archives):
        if not inner_episode_sharding and day_index % day_shard_count != day_shard_index:
            continue
        _verify_manifest_archive(archive, path, telemetry=telemetry)
        try:
            with zipfile.ZipFile(path) as bundle:
                # The validated daily exporter carries a value-free
                # ``manifest.csv`` beside the episode JSON members.  Its
                # member count is deliberately excluded from the sealed raw
                # corpus count, so it must not enter re-featurization.
                members = sorted(
                    member
                    for member in bundle.namelist()
                    if member.endswith(".json") and not member.endswith("/")
                )
                expected_members = _exact_int(
                    archive.get("zip_json_member_count"),
                    field=f"manifest JSON member count {path.name}",
                    minimum=1,
                )
                if len(members) != expected_members:
                    raise RunnerError("raw manifest JSON member count drifted during re-featurization")
                for member in members:
                    if inner_episode_sharding:
                        lane = int.from_bytes(
                            hashlib.sha256(
                                (str(archive.get("date")) + "\x00" + member).encode("utf-8")
                            ).digest()[:8],
                            "big",
                        ) % day_shard_count
                        if lane != day_shard_index:
                            continue
                    if max_episodes is not None and emitted >= max_episodes:
                        return
                    try:
                        value = json.loads(bundle.read(member))
                    except (KeyError, OSError, json.JSONDecodeError) as exc:
                        raise RunnerError(f"cannot decode {path.name}:{member}") from exc
                    if not isinstance(value, Mapping):
                        raise RunnerError(f"raw episode is not an object: {path.name}:{member}")
                    emitted += 1
                    if emitted % 32 == 0:
                        telemetry.sample()
                    yield archive, member, value
        except zipfile.BadZipFile as exc:
            raise RunnerError(f"raw manifest archive is invalid: {path}") from exc


def _load_transition_index(path: Path | None) -> dict[tuple[str, str], Mapping[str, Any]]:
    if path is None:
        return {}
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping) or value.get("schema") != R298_ENGINE_EVIDENCE_SCHEMA:
                    raise RunnerError(f"engine evidence schema drifted at line {line_number}")
                public_hash = require_sha256(value.get("canonical_public_observation_hash"), field="engine evidence public hash")
                action_hash = require_sha256(value.get("candidate_action_sha256"), field="engine evidence action hash")
                evidence = dict(value)
                evidence.pop("canonical_public_observation_hash", None)
                evidence.pop("candidate_action_sha256", None)
                key = (public_hash, action_hash)
                if key in index and canonical_sha256(index[key]) != canonical_sha256(evidence):
                    raise RunnerError(f"conflicting engine evidence at line {line_number}")
                index[key] = evidence
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read engine transition JSONL: {path}") from exc
    return index


def _record_bucket(record: Mapping[str, Any], *, bucket_count: int) -> int:
    """Route an option record by its complete public/token grouping key.

    Both the parent publisher and every private day lane call this one helper.
    That makes the lane boundary an execution detail rather than a different
    collision grouping rule: records that can collide always converge into
    the same parent-owned final bucket.
    """

    if bucket_count < 1 or bucket_count & (bucket_count - 1):
        raise RunnerError("shard bucket count must be a positive power of two")
    public_hash = require_sha256(
        record.get("canonical_public_observation_hash"),
        field="collision record public hash",
    )
    token_hash = require_sha256(
        record.get("current_feature_token_hash"),
        field="collision record token hash",
    )
    # Hash the whole pair rather than a short digest prefix.  Prefix-only
    # routing concentrates adversarial/common-prefix records into one bounded
    # shard and can turn an otherwise valid complete pass into a false size
    # failure.  The complete group key still maps to one bucket.
    route_digest = hashlib.sha256((public_hash + "\x00" + token_hash).encode("ascii")).digest()
    return int.from_bytes(route_digest[:8], "big") & (bucket_count - 1)


class _PrivateDayLaneSpool:
    """Private, bounded worker spool for deterministic parent publication.

    A day lane never owns final artifacts or acquires the global experiment
    lease.  It writes only canonical JSON records below the parent-created
    run directory.  The lease-holding parent checks each spool's SHA/size and
    performs the only merge/publication step in lane-index order.
    """

    def __init__(self, root: Path, *, lane_index: int, bucket_count: int) -> None:
        if lane_index < 0:
            raise RunnerError("day lane index must be nonnegative")
        if bucket_count < 1 or bucket_count & (bucket_count - 1):
            raise RunnerError("day lane bucket count must be a positive power of two")
        self.root = root
        self.bucket_count = bucket_count
        self.lane_index = lane_index
        self.root.mkdir(parents=False, exist_ok=False)
        self._streams: OrderedDict[int, Any] = OrderedDict()
        self._sizes: Counter[int] = Counter()
        self._record_counts: Counter[int] = Counter()

    def _path(self, bucket: int) -> Path:
        return self.root / f"bucket-{bucket:05d}.jsonl.partial"

    def _stream(self, bucket: int) -> Any:
        stream = self._streams.get(bucket)
        if stream is not None:
            self._streams.move_to_end(bucket)
            return stream
        path = self._path(bucket)
        if path.exists():
            if bucket not in self._sizes:
                raise RunnerError("private day-lane spool state is inconsistent")
            stream = path.open("ab")
        else:
            stream = path.open("xb")
            self._sizes[bucket] = 0
        if len(self._streams) >= MAX_OPEN_REFEATURE_SHARD_STREAMS:
            _, old_stream = self._streams.popitem(last=False)
            old_stream.flush()
            os.fsync(old_stream.fileno())
            old_stream.close()
        self._streams[bucket] = stream
        return stream

    def write(self, record: Mapping[str, Any]) -> None:
        bucket = _record_bucket(record, bucket_count=self.bucket_count)
        encoded = canonical_json_bytes(record)
        if len(encoded) > TARGET_TRANSFER_SHARD_BYTES:
            raise RunnerError("one re-featurized option record exceeds the bounded transfer shard limit")
        proposed = self._sizes[bucket] + len(encoded)
        if proposed > TARGET_TRANSFER_SHARD_BYTES:
            raise RunnerError(
                "one private day-lane collision bucket exceeds the bounded transfer limit; "
                "increase deterministic bucket fanout and rerun from a new output root"
            )
        self._stream(bucket).write(encoded)
        self._sizes[bucket] = proposed
        self._record_counts[bucket] += 1

    def close(self) -> list[dict[str, Any]]:
        for stream in self._streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self._streams.clear()
        metadata: list[dict[str, Any]] = []
        for bucket in sorted(self._sizes):
            path = self._path(bucket)
            size = path.stat().st_size
            if size != self._sizes[bucket] or size > TARGET_TRANSFER_SHARD_BYTES:
                raise RunnerError("private day-lane spool size verification failed")
            metadata.append(
                {
                    "lane_index": self.lane_index,
                    "bucket": bucket,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": size,
                    "record_count": self._record_counts[bucket],
                }
            )
        return metadata


class _ContentAddressedShardWriter:
    """Write immutable, independently verifiable bounded r298 record shards.

    Collision grouping uses a deterministic hash of the public observation and
    current token.  Therefore every equal-encoding group is emitted into one
    logical bucket/shard, while the shard header binds it to the frozen schema
    and exact 30-day raw manifest.  A bucket that would exceed the transfer
    ceiling fails closed instead of silently splitting a collision group.

    Final filenames are content-addressed and use an atomic hard-link
    publication from a private work file.  The only unlink is that private
    file after successful publication; no final artifact is overwritten,
    replaced, renamed, or deleted.
    """

    def __init__(
        self,
        root: Path,
        *,
        bucket_count: int,
        raw_manifest_sha256: str,
        frozen_schema_manifest_sha256: str,
        zero_bypass_receipt_sha256: str,
        record_scope: str = RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
    ) -> None:
        if bucket_count < 1 or bucket_count & (bucket_count - 1):
            raise RunnerError("shard bucket count must be a positive power of two")
        self.root = root
        self.shards_root = root / "shards"
        self.work_root = root / ".private-work"
        self.bucket_count = bucket_count
        if record_scope not in RECORD_SCOPES:
            raise RunnerError("refeatured record scope is not recognized")
        self.record_scope = record_scope
        self.raw_manifest_sha256 = require_sha256(raw_manifest_sha256, field="raw manifest shard binding")
        self.frozen_schema_manifest_sha256 = require_sha256(
            frozen_schema_manifest_sha256, field="frozen schema shard binding"
        )
        self.zero_bypass_receipt_sha256 = require_sha256(
            zero_bypass_receipt_sha256, field="zero bypass shard binding"
        )
        self.root.mkdir(parents=False, exist_ok=False)
        self.shards_root.mkdir(parents=False, exist_ok=False)
        self.work_root.mkdir(parents=False, exist_ok=False)
        self._streams: OrderedDict[int, Any] = OrderedDict()
        self._sizes: dict[int, int] = {}
        self._record_counts: Counter[int] = Counter()
        self._metadata: list[dict[str, Any]] = []
        self.record_count = 0

    def _bucket(self, record: Mapping[str, Any]) -> int:
        return _record_bucket(record, bucket_count=self.bucket_count)

    def _header(self, bucket: int) -> dict[str, Any]:
        return {
            "schema": R298_REFEATURED_RECORD_SHARD_SCHEMA,
            "kind": "immutable_factorized_option_records",
            "record_scope": self.record_scope,
            "owner_revision": R298_OWNER_REVISION,
            "goal_revision": REVISION_5_GOAL_REVISION,
            "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
            "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
            "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
            "revision_5_predecessor_classification": revision_5_predecessor_classification(),
            "logical_shard_id": f"public-token-bucket-{bucket:05d}",
            "bucket_count": self.bucket_count,
            "raw_expert_corpus_manifest_sha256": self.raw_manifest_sha256,
            "frozen_schema_manifest_sha256": self.frozen_schema_manifest_sha256,
            "zero_bypass_receipt_sha256": self.zero_bypass_receipt_sha256,
            "current_token_abi_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
            "max_transfer_object_bytes": MAX_TRANSFER_SHARD_BYTES,
            "target_maximum_bytes": TARGET_TRANSFER_SHARD_BYTES,
            "create_only": True,
            "inzi_runtime_or_training_authority": False,
        }

    def _stream(self, bucket: int) -> Any:
        stream = self._streams.get(bucket)
        if stream is not None:
            self._streams.move_to_end(bucket)
            return stream
        path = self.work_root / f"bucket-{bucket:05d}.partial"
        if path.exists():
            if bucket not in self._sizes:
                raise RunnerError("refeaturization private shard work state is inconsistent")
            stream = path.open("ab")
        else:
            header = canonical_json_bytes(self._header(bucket))
            if len(header) > TARGET_TRANSFER_SHARD_BYTES:
                raise RunnerError("refeaturization shard header exceeds bounded transfer limit")
            stream = path.open("xb")
            stream.write(header)
            self._sizes[bucket] = len(header)
        if len(self._streams) >= MAX_OPEN_REFEATURE_SHARD_STREAMS:
            old_bucket, old_stream = self._streams.popitem(last=False)
            old_stream.flush()
            os.fsync(old_stream.fileno())
            old_stream.close()
            if old_bucket == bucket:  # pragma: no cover - impossible after get() miss
                raise RunnerError("refeaturization shard stream eviction loop")
        self._streams[bucket] = stream
        return stream

    def write(self, record: Mapping[str, Any]) -> None:
        bucket = self._bucket(record)
        encoded = canonical_json_bytes(record)
        if len(encoded) > TARGET_TRANSFER_SHARD_BYTES:
            raise RunnerError("one re-featurized option record exceeds the bounded transfer shard limit")
        stream = self._stream(bucket)
        proposed = self._sizes[bucket] + len(encoded)
        if proposed > TARGET_TRANSFER_SHARD_BYTES:
            raise RunnerError(
                "one public-token collision bucket exceeds the bounded transfer shard limit; "
                "increase deterministic bucket fanout and rerun from a new output root"
            )
        stream.write(encoded)
        self._sizes[bucket] = proposed
        self._record_counts[bucket] += 1
        self.record_count += 1

    def close(self) -> None:
        for stream in self._streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self._streams.clear()
        for bucket in sorted(self._sizes):
            work_path = self.work_root / f"bucket-{bucket:05d}.partial"
            size = work_path.stat().st_size
            if size != self._sizes[bucket] or size > MAX_TRANSFER_SHARD_BYTES:
                raise RunnerError("refeaturization shard size verification failed")
            digest = sha256_file(work_path)
            filename = f"sha256-{digest[7:]}.refeaturization-census.shard"
            final_path = self.shards_root / filename
            try:
                os.link(work_path, final_path)
            except FileExistsError as exc:
                raise RunnerError("content-addressed re-featurization shard already exists in create-only output") from exc
            if sha256_file(final_path) != digest or final_path.stat().st_size != size:
                raise RunnerError("published re-featurization shard failed local SHA/size verification")
            # This private file was created by this writer and now has an
            # immutable hard-linked final object.  Removing only the private
            # staging name prevents it from being mistaken for a transferable
            # output while preserving the final content-addressed object.
            work_path.unlink()
            self._metadata.append(
                {
                    "logical_shard_id": self._header(bucket)["logical_shard_id"],
                    "filename": filename,
                    "sha256": digest,
                    "size_bytes": size,
                    "record_count": self._record_counts[bucket],
                    "schema": R298_REFEATURED_RECORD_SHARD_SCHEMA,
                }
            )

    def abort(self) -> None:
        """Close only private work files after a failed pass; publish nothing."""

        for stream in self._streams.values():
            stream.close()
        self._streams.clear()

    def shard_paths(self) -> list[Path]:
        return [self.shards_root / str(row["filename"]) for row in self._metadata]

    def manifest(self) -> dict[str, Any]:
        if self._streams:
            raise RunnerError("cannot emit a re-featurization manifest before shards are closed")
        records = sorted(self._metadata, key=lambda row: str(row["logical_shard_id"]))
        if sum(int(row["record_count"]) for row in records) != self.record_count:
            raise RunnerError("refeaturization shard record count does not close")
        return {
            "schema": R298_REFEATURED_RECORD_MANIFEST_SCHEMA,
            "status": "sealed_content_addressed_bounded_shards",
            "owner_revision": R298_OWNER_REVISION,
            "goal_revision": REVISION_5_GOAL_REVISION,
            "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
            "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
            "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
            "revision_5_predecessor_classification": revision_5_predecessor_classification(),
            "raw_expert_corpus_manifest_sha256": self.raw_manifest_sha256,
            "frozen_schema_manifest_sha256": self.frozen_schema_manifest_sha256,
            "zero_bypass_receipt_sha256": self.zero_bypass_receipt_sha256,
            "current_token_abi_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
            "maximum_shard_size_bytes": MAX_TRANSFER_SHARD_BYTES,
            "all_shards_individually_schema_sha256_size_validated": True,
            "content_addressed_filename_pattern": "sha256-<64_lowercase_hex>.refeaturization-census.shard",
            "inzi_quarantined_staging_eligible_only": True,
            "inzi_runtime_or_training_authority": False,
            "record_count": self.record_count,
            "record_scope": self.record_scope,
            "shard_count": len(records),
            "shards": records,
        }


def _read_refeatured_shard(
    path: Path,
    *,
    raw_manifest_sha256: str,
    frozen_schema_manifest_sha256: str,
    zero_bypass_receipt_sha256: str,
    record_scope: str = RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
) -> list[Mapping[str, Any]]:
    """Validate a content-addressed shard before it enters collision grouping."""

    filename = path.name
    if record_scope not in RECORD_SCOPES:
        raise RunnerError("requested re-featurization shard scope is not recognized")
    if re.fullmatch(r"sha256-[0-9a-f]{64}\.refeaturization-census\.shard", filename) is None:
        raise RunnerError(f"refeaturization shard has a non-content-addressed filename: {path}")
    expected_digest = "sha256:" + filename[len("sha256-") : -len(".refeaturization-census.shard")]
    if sha256_file(path) != expected_digest or path.stat().st_size > MAX_TRANSFER_SHARD_BYTES:
        raise RunnerError(f"refeaturization shard identity/size drifted: {path}")
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("rb") as stream:
            first = stream.readline()
            header = json.loads(first)
            if not isinstance(header, Mapping) or header.get("schema") != R298_REFEATURED_RECORD_SHARD_SCHEMA:
                raise RunnerError(f"refeaturization shard header schema drifted: {path}")
            for field, expected in {
                "raw_expert_corpus_manifest_sha256": raw_manifest_sha256,
                "frozen_schema_manifest_sha256": frozen_schema_manifest_sha256,
                "zero_bypass_receipt_sha256": zero_bypass_receipt_sha256,
                "record_scope": record_scope,
                "owner_revision": R298_OWNER_REVISION,
                "goal_revision": REVISION_5_GOAL_REVISION,
                "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
                "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
                "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
                "revision_5_predecessor_classification": revision_5_predecessor_classification(),
                "current_token_abi_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
                "max_transfer_object_bytes": MAX_TRANSFER_SHARD_BYTES,
                "create_only": True,
                "inzi_runtime_or_training_authority": False,
            }.items():
                if header.get(field) != expected:
                    raise RunnerError(f"refeaturization shard binding drifted at {path}:{field}")
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise RunnerError(f"refeaturization shard record malformed: {path}")
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read re-featurization shard: {path}") from exc
    return records


class _PhaseAInventoryAccumulator:
    """Constant-memory schema classifier for one explicitly named surface."""

    def __init__(self, *, inventory_scope: str) -> None:
        if not isinstance(inventory_scope, str) or not inventory_scope:
            raise RunnerError("Phase A inventory scope is malformed")
        self._inventory_scope = inventory_scope
        self._field_types: dict[str, Counter[str]] = defaultdict(Counter)
        self._field_classes: dict[str, Counter[str]] = defaultdict(Counter)
        self._occurrences: Counter[str] = Counter()
        self._observation_count = 0
        self._rejected = 0
        self._actor_context_count = 0
        self._no_actor_context_count = 0

    def add(self, observation: Mapping[str, Any]) -> None:
        # Reuse the canonical sample implementation for a single observation,
        # then merge its bounded field summary.  No raw episode or raw field
        # value is retained.  A malformed observation is recorded in the
        # inventory and later blocks the receipt rather than being discarded.
        try:
            inventory = inventory_raw_observations(
                [observation], inventory_scope=self._inventory_scope
            )
        except CollisionCensusError as exc:
            raise RunnerError("Phase A raw schema inventory failed") from exc
        self._observation_count += int(inventory["raw_observation_count"])
        self._rejected += int(inventory["rejected_observation_count"])
        self._actor_context_count += int(inventory["actor_context_observation_count"])
        self._no_actor_context_count += int(inventory["no_actor_context_observation_count"])
        self._occurrences.update(inventory["field_classification_occurrences"])
        for field in inventory["fields"]:
            path = str(field["path"])
            self._field_types[path].update(field["types"])
            self._field_classes[path].update(field["classification_occurrences"])

    def merge_final(self, inventory: Mapping[str, Any], *, allow_empty: bool = False) -> None:
        """Merge a worker's bounded inventory without accepting raw values.

        Workers return only the same schema summary produced by ``final``.
        Re-validating it here prevents a child-process result from widening the
        field catalog or quietly changing the five-way classification ABI.
        """

        raw_count = _exact_int(
            inventory.get("raw_observation_count"),
            field="parallel inventory raw observation count",
        )
        if allow_empty and raw_count == 0:
            if dict(inventory) != inventory_raw_observations(
                [], inventory_scope=self._inventory_scope
            ):
                raise RunnerError("parallel empty Phase A inventory drifted")
        else:
            try:
                validate_phase_a_inventory(inventory)
            except CollisionCensusError as exc:
                raise RunnerError("parallel day-lane Phase A inventory is invalid") from exc
        if inventory.get("inventory_scope") != self._inventory_scope:
            raise RunnerError("parallel day-lane Phase A inventory scope drifted")
        self._observation_count += raw_count
        self._rejected += _exact_int(
            inventory.get("rejected_observation_count"),
            field="parallel inventory rejected observation count",
        )
        self._actor_context_count += _exact_int(
            inventory.get("actor_context_observation_count"),
            field="parallel inventory actor-context count",
        )
        self._no_actor_context_count += _exact_int(
            inventory.get("no_actor_context_observation_count"),
            field="parallel inventory no-actor-context count",
        )
        occurrences = inventory.get("field_classification_occurrences")
        fields = inventory.get("fields")
        if not isinstance(occurrences, Mapping) or not isinstance(fields, list):
            raise RunnerError("parallel day-lane Phase A inventory is malformed")
        self._occurrences.update({str(key): int(value) for key, value in occurrences.items()})
        for field in fields:
            if not isinstance(field, Mapping):
                raise RunnerError("parallel day-lane Phase A field is malformed")
            path = field.get("path")
            types = field.get("types")
            classes = field.get("classification_occurrences")
            if not isinstance(path, str) or not isinstance(types, Mapping) or not isinstance(classes, Mapping):
                raise RunnerError("parallel day-lane Phase A field is malformed")
            self._field_types[path].update({str(key): int(value) for key, value in types.items()})
            self._field_classes[path].update({str(key): int(value) for key, value in classes.items()})

    def final(self) -> dict[str, Any]:
        # Construct one representative input with the merged path/type counts.
        # The module's sample inventory supplies the catalog and exact schema.
        prototype = inventory_raw_observations(
            [], inventory_scope=self._inventory_scope
        )
        classifications = prototype["field_classification_occurrences"]
        classifications.update(self._occurrences)
        fields = [
            {
                "path": path,
                "types": dict(sorted(types.items())),
                "classification_occurrences": dict(sorted(self._field_classes[path].items())),
            }
            for path, types in sorted(self._field_types.items())
        ]
        prototype.update(
            {
                "raw_observation_count": self._observation_count,
                "rejected_observation_count": self._rejected,
                "actor_context_observation_count": self._actor_context_count,
                "no_actor_context_observation_count": self._no_actor_context_count,
                "field_classification_occurrences": classifications,
                "field_schema_sha256": canonical_sha256(fields),
                "fields": fields,
            }
        )
        return prototype


def _run_private_day_lane(
    *,
    lane_index: int,
    lane_count: int,
    archives: Sequence[tuple[Mapping[str, Any], str]],
    spool_root: str,
    bucket_count: int,
    engine_transition_jsonl: str | None,
    cg_runtime_root: str,
    materialized_only: bool = False,
) -> dict[str, Any]:
    """Re-featurize assigned UTC days without publishing or leasing anything.

    This top-level function is deliberately picklable for a process pool.  It
    takes no parent writer, no mutable global accumulator, and no lease.  The
    parent owns all publication/merge work after validating every private
    spool identity.
    """

    if not 1 <= lane_count <= 256 or not 0 <= lane_index < lane_count:
        raise RunnerError("parallel re-featurization requires 1 through 256 assigned lanes")
    telemetry = _ResourceTelemetry(probe_gpu=False)
    _verify_cg_runtime(Path(cg_runtime_root))
    token_builder = _token_abi_builder()
    transition_index = _load_transition_index(
        Path(engine_transition_jsonl) if engine_transition_jsonl is not None else None
    )
    collision_audit_spool = _PrivateDayLaneSpool(
        Path(spool_root) / "collision-audit" / f"lane-{lane_index:02d}",
        lane_index=lane_index,
        bucket_count=bucket_count,
    )
    materialized_spool = _PrivateDayLaneSpool(
        Path(spool_root) / "materialized" / f"lane-{lane_index:02d}",
        lane_index=lane_index,
        bucket_count=bucket_count,
    )
    raw_inventory_accumulator = _PhaseAInventoryAccumulator(
        inventory_scope=PHASE_A_RAW_REPLAY_INVENTORY_SCOPE
    )
    actor_selection_inventory_accumulator = _PhaseAInventoryAccumulator(
        inventory_scope=PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE
    )
    raw_episode_count = 0
    raw_outer_observation_count = 0
    processed_frame_count = 0
    actor_visible_frame_count = 0
    alakazam_actor_visible_frame_count = 0
    excluded_non_alakazam_actor_visible_frame_count = 0
    forced_frame_count = 0
    collision_audit_action_stage_count = 0
    materialized_action_stage_count = 0
    acting_deck_distribution: Counter[str] = Counter()
    matchup_distribution: Counter[str] = Counter()
    per_day_episode_count: Counter[str] = Counter()
    completed = False
    try:
        normalized_archives = [(archive, Path(path)) for archive, path in archives]
        for archive, member, payload in _iter_episode_payloads(
            normalized_archives,
            day_shard_index=lane_index,
            day_shard_count=lane_count,
            max_episodes=None,
            telemetry=telemetry,
        ):
            raw_episode_count += 1
            per_day_episode_count[str(archive["date"])] += 1
            for raw_observation in raw_observations_from_recorded_episode(payload):
                raw_inventory_accumulator.add(raw_observation)
                raw_outer_observation_count += 1
            coverage = recorded_episode_frame_coverage(payload)
            actor_visible_frame_count += coverage["actor_visible_selection_frame_count"]
            forced_frame_count += coverage["forced_selection_frame_count"]
            source = {
                "source_archive_sha256": archive.get("sha256"),
                "source_archive_date": archive.get("date"),
                "source_member": member,
                "source_episode_schema": payload.get("schema_version"),
                "raw_transition_target_only": True,
            }
            descriptors = stage_descriptors_from_recorded_episode(payload, source=source)
            represented_frames = {int(descriptor["source"]["env_step"]) for descriptor in descriptors}
            if len(represented_frames) != coverage["actor_visible_selection_frame_count"]:
                raise RunnerError("an actor-visible selection frame was omitted during parallel re-featurization")
            # The collision audit is intentionally all-seat/all-episode.  The
            # card-743 acting-seat predicate controls only the separately
            # materialized feature rows, never the audit input surface.
            eligible_descriptors = [
                descriptor
                for descriptor in descriptors
                if descriptor["source"].get("acting_seat_setup_deck_contains_card_743") is True
            ]
            eligible_frames = {
                int(descriptor["source"]["env_step"])
                for descriptor in eligible_descriptors
            }
            alakazam_actor_visible_frame_count += len(eligible_frames)
            excluded_non_alakazam_actor_visible_frame_count += len(represented_frames) - len(eligible_frames)
            processed_frame_count += len(eligible_frames)
            observations_by_env_step: dict[int, Mapping[str, Any]] = {}
            for descriptor in descriptors:
                env_step = int(descriptor["source"]["env_step"])
                observation = descriptor["observation"]
                prior = observations_by_env_step.setdefault(env_step, observation)
                if prior != observation:
                    raise RunnerError("factorized stages disagree on a frame's masked observation")
            for observation in observations_by_env_step.values():
                actor_selection_inventory_accumulator.add(observation)
            for descriptor in descriptors:
                observation = descriptor["observation"]
                source_row = dict(descriptor["source"])
                materialization_eligible = (
                    source_row.get("acting_seat_setup_deck_contains_card_743") is True
                )
                source_row["row_materialization_eligible"] = materialization_eligible
                source_row["row_materialization_exclusion_reason"] = (
                    "eligible_same_acting_seat_setup_deck_contains_card_743"
                    if materialization_eligible
                    else (
                        "missing_or_malformed_same_acting_seat_setup_deck"
                        if source_row.get("acting_deck_multiset_sha256") is None
                        else "same_acting_seat_setup_deck_lacks_card_743"
                    )
                )
                acting_deck = source_row.get("acting_deck_multiset_sha256")
                opponent_deck = source_row.get("opponent_deck_multiset_sha256")
                if isinstance(acting_deck, str):
                    acting_deck_distribution[acting_deck] += 1
                if isinstance(opponent_deck, str):
                    matchup_distribution[opponent_deck] += 1
                public_hash = canonical_public_observation_hash(observation)
                transition_by_action = {
                    action_key_sha256(candidate): transition_index[(public_hash, action_key_sha256(candidate))]
                    for candidate in descriptor["candidates"]
                    if (public_hash, action_key_sha256(candidate)) in transition_index
                }
                stage_records = build_stage_option_records(
                    observation,
                    descriptor["candidates"],
                    stage_prefix=descriptor["stage_prefix"],
                    selected_candidate_index=descriptor["selected_candidate_index"],
                    transition_by_action=transition_by_action,
                    token_builder=token_builder,
                    source=source_row,
                )
                for record in stage_records:
                    if not materialized_only:
                        collision_audit_spool.write(record)
                    if materialization_eligible:
                        materialized_spool.write(record)
                if not materialized_only:
                    collision_audit_action_stage_count += 1
                if materialization_eligible:
                    materialized_action_stage_count += 1
            telemetry.sample()
        completed = True
    finally:
        # Private spool data can remain after a failed create-only run for
        # diagnosis, but it is never published into a final shard unless the
        # parent receives a complete, checksummed lane result.
        if not completed:
            for spool in (collision_audit_spool, materialized_spool):
                for stream in spool._streams.values():
                    stream.close()
                spool._streams.clear()
    if processed_frame_count != alakazam_actor_visible_frame_count:
        raise RunnerError("parallel lane did not cover every exact-list Alakazam actor-visible frame")
    raw_inventory = raw_inventory_accumulator.final()
    actor_selection_inventory = actor_selection_inventory_accumulator.final()
    validate_phase_a_inventory(raw_inventory)
    if actor_selection_inventory["raw_observation_count"]:
        validate_phase_a_inventory(actor_selection_inventory)
    elif actor_selection_inventory != inventory_raw_observations(
        [], inventory_scope=PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE
    ):
        raise RunnerError("parallel empty actor selection inventory drifted")
    if raw_inventory["raw_observation_count"] != raw_outer_observation_count:
        raise RunnerError("parallel raw Phase A inventory does not close")
    if actor_selection_inventory["raw_observation_count"] != actor_visible_frame_count:
        raise RunnerError("parallel actor selection Phase A inventory does not close")
    return {
        "lane_index": lane_index,
        "raw_episode_count": raw_episode_count,
        "raw_outer_observation_count": raw_outer_observation_count,
        "processed_frame_count": processed_frame_count,
        "actor_visible_frame_count": actor_visible_frame_count,
        "alakazam_actor_visible_frame_count": alakazam_actor_visible_frame_count,
        "excluded_non_alakazam_actor_visible_frame_count": excluded_non_alakazam_actor_visible_frame_count,
        "forced_frame_count": forced_frame_count,
        "collision_audit_action_stage_count": collision_audit_action_stage_count,
        "materialized_action_stage_count": materialized_action_stage_count,
        "acting_deck_distribution": dict(sorted(acting_deck_distribution.items())),
        "matchup_distribution": dict(sorted(matchup_distribution.items())),
        "per_day_raw_episode_count": dict(sorted(per_day_episode_count.items())),
        "raw_inventory": raw_inventory,
        "actor_selection_inventory": actor_selection_inventory,
        "collision_audit_private_spool_shards": collision_audit_spool.close(),
        "materialized_private_spool_shards": materialized_spool.close(),
        "resource_observation": telemetry.final(),
    }


def _merge_private_day_lane_spools(
    lane_results: Sequence[Mapping[str, Any]],
    *,
    spool_root: Path,
    writer: _ContentAddressedShardWriter,
    spool_field: str = "private_spool_shards",
) -> int:
    """Validate and deterministically merge worker-private record spools.

    The merge order is fixed (lane, bucket, canonical source record order).
    Every line must be the canonical JSON representation returned by the
    worker, hash to its declared private object identity, and route to its
    declared final bucket.  A worker cannot smuggle a record directly into the
    final content-addressed output.
    """

    expected_lanes = list(range(len(lane_results)))
    actual_lanes = [result.get("lane_index") for result in lane_results]
    if actual_lanes != expected_lanes:
        raise RunnerError("parallel day-lane results are incomplete or non-deterministic")
    merged_record_count = 0
    root_resolved = spool_root.resolve()
    for result in lane_results:
        lane_index = _exact_int(result.get("lane_index"), field="private lane index")
        raw_spools = result.get(spool_field)
        if not isinstance(raw_spools, list):
            raise RunnerError("parallel day-lane private spool inventory is malformed")
        previous_bucket = -1
        for raw_spool in raw_spools:
            if not isinstance(raw_spool, Mapping):
                raise RunnerError("parallel private spool row is malformed")
            if set(raw_spool) != {"lane_index", "bucket", "path", "sha256", "size_bytes", "record_count"}:
                raise RunnerError("parallel private spool row field inventory drifted")
            if _exact_int(raw_spool.get("lane_index"), field="private spool lane index") != lane_index:
                raise RunnerError("parallel private spool lane identity drifted")
            bucket = _exact_int(raw_spool.get("bucket"), field="private spool bucket")
            if not 0 <= bucket < writer.bucket_count or bucket <= previous_bucket:
                raise RunnerError("parallel private spool bucket ordering drifted")
            previous_bucket = bucket
            path_value = raw_spool.get("path")
            if not isinstance(path_value, str):
                raise RunnerError("parallel private spool path is malformed")
            path = Path(path_value)
            try:
                path_resolved = path.resolve(strict=True)
                path_resolved.relative_to(root_resolved)
            except (OSError, ValueError) as exc:
                raise RunnerError("parallel private spool escapes its parent-owned root") from exc
            if path_resolved.parent.name != f"lane-{lane_index:02d}" or path_resolved.name != f"bucket-{bucket:05d}.jsonl.partial":
                raise RunnerError("parallel private spool logical identity/path drifted")
            expected_digest = require_sha256(raw_spool.get("sha256"), field="private spool digest")
            expected_size = _exact_int(raw_spool.get("size_bytes"), field="private spool size", minimum=0)
            expected_records = _exact_int(raw_spool.get("record_count"), field="private spool record count", minimum=1)
            if path_resolved.stat().st_size != expected_size or sha256_file(path_resolved) != expected_digest:
                raise RunnerError("parallel private spool physical identity drifted")
            actual_records = 0
            try:
                with path_resolved.open("rb") as stream:
                    for raw_line in stream:
                        try:
                            record = json.loads(raw_line)
                        except json.JSONDecodeError as exc:
                            raise RunnerError("parallel private spool record is not JSON") from exc
                        if not isinstance(record, Mapping) or canonical_json_bytes(record) != raw_line:
                            raise RunnerError("parallel private spool record is not canonical JSON")
                        if _record_bucket(record, bucket_count=writer.bucket_count) != bucket:
                            raise RunnerError("parallel private spool record routes to a different bucket")
                        writer.write(record)
                        actual_records += 1
            except OSError as exc:
                raise RunnerError("cannot read parallel private spool") from exc
            if actual_records != expected_records:
                raise RunnerError("parallel private spool record count drifted")
            merged_record_count += actual_records
    return merged_record_count


def _aggregate_bucket_reports(
    paths: Iterable[Path],
    *,
    decision_count: int,
    inventory_only: bool,
    raw_manifest_sha256: str,
    frozen_schema_manifest_sha256: str,
    zero_bypass_receipt_sha256: str,
    record_scope: str = RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
) -> dict[str, Any]:
    reports: list[Mapping[str, Any]] = []
    for path in paths:
        records = _read_refeatured_shard(
            path,
            raw_manifest_sha256=raw_manifest_sha256,
            frozen_schema_manifest_sha256=frozen_schema_manifest_sha256,
            zero_bypass_receipt_sha256=zero_bypass_receipt_sha256,
            record_scope=record_scope,
        )
        reports.append(analyze_collision_records(records, decision_count=0, inventory_only=inventory_only))
    groups = [group for report in reports for group in report["groups"]]
    class_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for report in reports:
        class_counts.update(report["classification_frequency"])
        risk_counts.update(report["action_change_risk_frequency"])
    actionable = sum(int(report["actionable_failure_group_count"]) for report in reports)
    incomplete = sum(int(report["incomplete_pinned_evidence_record_count"]) for report in reports)
    if inventory_only:
        status = "inventory_only_not_a_collision_verdict"
    elif actionable:
        status = STATUS_FAILED_COLLISION
    elif incomplete:
        status = "blocked_incomplete_pinned_simulator_evidence"
    else:
        status = "passed_no_actionable_public_semantic_collision"
    return {
        "schema": "poke_bot.alakazam_collision_census_r298_report/v1",
        "status": status,
        "collision_group_count": len(groups),
        "collision_record_count": sum(int(report["collision_record_count"]) for report in reports),
        "all_option_record_count": sum(int(report["all_option_record_count"]) for report in reports),
        "decision_count": decision_count,
        "collision_frequency_per_decision": len(groups) / decision_count if decision_count else 0.0,
        "classification_frequency": dict(sorted(class_counts.items())),
        "action_change_risk_frequency": dict(sorted(risk_counts.items())),
        "selected_action_in_collision_record_count": sum(int(report["selected_action_in_collision_record_count"]) for report in reports),
        "selected_action_in_divergent_collision_record_count": sum(int(report["selected_action_in_divergent_collision_record_count"]) for report in reports),
        "actionable_failure_group_count": actionable,
        "incomplete_pinned_evidence_record_count": incomplete,
        "groups": groups,
        "bucketed_streaming": True,
        "record_scope": record_scope,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/evaluations/alakazam-collision-census-r298.json")
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="validate and print the inert r298 input contract without materializing anything",
    )
    parser.add_argument("--execute", action="store_true", help="perform the explicitly requested offline phase")
    parser.add_argument(
        "--phase",
        choices=("manifest", "census", "census-completion"),
        default="manifest",
    )
    parser.add_argument("--archive-root", type=Path, help="directory containing exactly the raw daily ZIPs")
    parser.add_argument("--reference-receipt", action="append", type=Path, default=[], help="read-only source receipt, recorded as provenance only")
    parser.add_argument("--raw-corpus-manifest", type=Path)
    parser.add_argument("--raw-corpus-receipt", type=Path)
    parser.add_argument("--frozen-schema-manifest", type=Path)
    parser.add_argument("--zero-bypass-receipt", type=Path)
    parser.add_argument(
        "--revision-5-census-validation-receipt",
        type=Path,
        help="sealed r5 census bridge consumed read-only after the full pass",
    )
    parser.add_argument(
        "--collision-census-receipt",
        type=Path,
        help="immutable r5 collision receipt consumed read-only after the full pass",
    )
    parser.add_argument("--cg-runtime-root", type=Path)
    parser.add_argument("--engine-transition-jsonl", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", type=str, default="r298-phase-a")
    parser.add_argument("--mode", choices=("sample", "full"), default="sample")
    parser.add_argument("--max-episodes", type=int, default=1, help="sample only; full requires 0/all")
    # External day shards were intentionally never merge-safe because each
    # could split a public/token collision group.  Full mode instead starts
    # exact 24 private day lanes under one parent-held global lease.
    parser.add_argument("--day-shard-index", type=int, default=0)
    parser.add_argument("--day-shard-count", type=int, default=1)
    parser.add_argument(
        "--day-workers",
        type=int,
        default=1,
        help="private parent-owned UTC-day lanes; full mode requires exactly 24",
    )
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)
    parser.add_argument("--full-corpus-ack", action="store_true")
    return parser.parse_args(argv)


def _output_dir(root: Path, run_id: str) -> Path:
    result = root / run_id
    if result.exists():
        raise RunnerError(f"create-only output root already exists: {result}")
    result.mkdir(parents=True, exist_ok=False)
    return result


def _run_manifest(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    telemetry: _ResourceTelemetry,
    execution_identity: Mapping[str, Any],
) -> int:
    if args.archive_root is None or args.output_root is None:
        raise RunnerError("manifest phase requires --archive-root and --output-root")
    output = _output_dir(args.output_root, args.run_id)
    lease_root = _experiment_lease_root()
    with _ExperimentLease(lease_root) as lease:
        references, provenance = _reference_archive_rows(args.reference_receipt)
        if set(references) < set(_expected_dates()):
            raise RunnerError("canonical source receipts do not cover every required 30-day raw partition")
        rows, member_counts, byte_counts = _raw_archive_rows(
            args.archive_root,
            reference_rows=references,
            telemetry=telemetry,
        )
        episode_deduplication = _trusted_source_receipt_episode_proof(rows)
        manifest = build_raw_corpus_manifest(
            rows,
            archive_member_counts=member_counts,
            archive_bytes_actual=byte_counts,
            source_manifest_provenance=provenance,
            episode_deduplication=episode_deduplication,
        )
    run_identity = canonical_sha256(
        {
            "schema": RAW_MANIFEST_RUN_SCHEMA,
            "config_sha256": sha256_file(args.config),
            "owner_revision": R298_OWNER_REVISION,
            "goal_revision": REVISION_5_GOAL_REVISION,
            "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
            "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
            "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
            "revision_5_predecessor_classification": revision_5_predecessor_classification(),
            "archive_root": str(args.archive_root.resolve()),
            "reference_receipts": provenance,
            "archives": [{"date": row["date"], "sha256": row["sha256"]} for row in manifest["archives"]],
            "execution_identity": _validate_elmo_execution_identity(execution_identity),
        }
    )
    resource_observation = telemetry.final()
    resource_observation["execution_identity"] = _validate_elmo_execution_identity(
        execution_identity
    )
    resource_observation["aggregate_experiment_ram_accounting"] = {
        "method": "single_global_r298_memory_heavy_process_under_exclusive_flock",
        "global_lease_path": str(lease.path),
        "global_lease_namespace": str(lease_root),
        "aggregate_child_processes": 0,
        "aggregate_peak_bytes": resource_observation["actual_peak_rss_bytes"],
        "hard_ceiling_bytes": HARD_EXPERIMENT_MEMORY_BYTES,
        "one_memory_heavy_phase_exclusive_lease": True,
    }
    receipt = make_raw_corpus_receipt(
        manifest,
        run_identity_sha256=run_identity,
        resource_observation=resource_observation,
    )
    receipt["execution_identity"] = _validate_elmo_execution_identity(execution_identity)
    receipt["run"] = {
        "schema": RAW_MANIFEST_RUN_SCHEMA,
        "recorded_at_utc": _utc(),
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "config_enabled": config.get("enabled"),
        "execution_identity": _validate_elmo_execution_identity(execution_identity),
        "reference_receipt_day_coverage": len(references),
        "published_source_discrepancy_unknown_day_count": sum(
            row["source_discrepancy"].get("known") is False
            for row in manifest["archives"]
            if isinstance(row.get("source_discrepancy"), Mapping)
        ),
    }
    _write_create_only_json(output / "raw_expert_corpus_manifest.json", manifest)
    _write_create_only_json(output / "raw_expert_corpus_receipt.json", receipt)
    print(json.dumps({"schema": RUN_SCHEMA, "status": "passed", "output_root": str(output), "manifest_sha256": canonical_sha256(manifest), "receipt_sha256": canonical_sha256(receipt)}, sort_keys=True))
    return 0


def _run_census_completion_validation(
    args: argparse.Namespace,
    execution_identity: Mapping[str, Any],
) -> int:
    """Read-only rev7 acceptance bridge for an already sealed r5 census.

    This deliberately has no output root, lease, shard writer, or receipt
    writer.  The live r5 job is historical evidence: this command can only
    inspect it under current rev7 authority and print its in-memory validation
    view.  The later rev6 migration is the only artifact that cross-links its
    result into the next materialization phase.
    """

    required = {
        "raw_corpus_manifest": args.raw_corpus_manifest,
        "raw_corpus_receipt": args.raw_corpus_receipt,
        "frozen_schema_manifest": args.frozen_schema_manifest,
        "zero_bypass_receipt": args.zero_bypass_receipt,
        "revision_5_census_validation_receipt": args.revision_5_census_validation_receipt,
        "collision_census_receipt": args.collision_census_receipt,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RunnerError(
            "census-completion phase requires " + ", ".join(sorted(missing))
        )
    raw_manifest, raw_manifest_file_sha = _read_stable_regular_json(
        args.raw_corpus_manifest, label="raw corpus manifest"
    )
    raw_receipt, raw_receipt_file_sha = _read_stable_regular_json(
        args.raw_corpus_receipt, label="raw corpus receipt"
    )
    frozen_schema_manifest, frozen_schema_file_sha = _read_stable_regular_json(
        args.frozen_schema_manifest, label="frozen schema manifest"
    )
    zero_bypass_receipt, zero_bypass_file_sha = _read_stable_regular_json(
        args.zero_bypass_receipt, label="zero-bypass receipt"
    )
    census_validation_receipt, census_validation_file_sha = _read_stable_regular_json(
        args.revision_5_census_validation_receipt,
        label="revision-5 census validation receipt",
    )
    # The compatibility API independently reopens the collision receipt with
    # its own stable file guard because it must return both physical and
    # canonical-object identities.  Preflight is intentionally absent here.
    try:
        completion = validate_revision_5_census_completion_under_revision_7(
            raw_manifest=raw_manifest,
            raw_receipt=raw_receipt,
            schema_manifest=frozen_schema_manifest,
            zero_bypass_receipt=zero_bypass_receipt,
            census_validation_receipt=census_validation_receipt,
            collision_census_receipt_path=args.collision_census_receipt,
        )
    except Rev7PredecessorCompatibilityError as exc:
        raise RunnerError("revision-7 post-census completion validation failed") from exc
    if completion.get("materialization_preflight_claimed_or_required") is not False:
        raise RunnerError("post-census bridge unexpectedly claims a materialization preflight")
    if completion.get("historical_receipts_retagged_or_rewritten") is not False:
        raise RunnerError("post-census bridge reports historical receipt retagging")
    if completion.get("training_runtime_service_transfer_or_activation_authority") is not False:
        raise RunnerError("post-census bridge unexpectedly grants authority")
    print(
        json.dumps(
            {
                "schema": RUN_SCHEMA,
                "status": "validated_read_only_revision_7_census_completion",
                "execution_identity": _validate_elmo_execution_identity(
                    execution_identity
                ),
                "input_physical_file_sha256s": {
                    "raw_expert_corpus_manifest": raw_manifest_file_sha,
                    "raw_expert_corpus_receipt": raw_receipt_file_sha,
                    "frozen_schema_manifest": frozen_schema_file_sha,
                    "zero_bypass_receipt": zero_bypass_file_sha,
                    "revision_5_census_validation_receipt": census_validation_file_sha,
                },
                "completion": completion,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_census(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    telemetry: _ResourceTelemetry,
    execution_identity: Mapping[str, Any],
) -> int:
    if any(value is None for value in (args.raw_corpus_manifest, args.raw_corpus_receipt, args.frozen_schema_manifest, args.zero_bypass_receipt, args.cg_runtime_root, args.output_root)):
        raise RunnerError("census phase requires raw corpus, frozen-schema, zero-bypass, CG runtime, and output inputs")
    if args.day_shard_count < 1 or not 0 <= args.day_shard_index < args.day_shard_count:
        raise RunnerError("invalid complete-day shard selection")
    if args.day_workers < 1:
        raise RunnerError("day worker count must be positive")
    if args.mode == "full":
        if not args.full_corpus_ack:
            raise RunnerError("full 30-day census requires --full-corpus-ack")
        if args.max_episodes != 0:
            raise RunnerError("full census may not impose a max-episodes subset")
        if args.day_shard_count != 1:
            raise RunnerError("a shard is not a final full-corpus census; merge receipts separately")
        if args.day_workers != PHASE_A_VALIDATION_WORKERS:
            raise RunnerError("full census requires exactly 24 parent-owned private day workers")
        if args.bucket_count < DEFAULT_BUCKET_COUNT:
            raise RunnerError("full census requires at least the r298 bounded-shard bucket fanout")
    elif args.max_episodes < 1 or args.max_episodes > int(config["limits"]["sample_max_episodes"]):
        raise RunnerError("sample census max episodes is outside the staged cap")
    elif args.day_workers != 1:
        raise RunnerError("sample census is deliberately single-lane; use full --day-workers 24")
    manifest = _read_json(args.raw_corpus_manifest)
    raw_receipt = _read_json(args.raw_corpus_receipt)
    frozen_schema_manifest = _read_json(args.frozen_schema_manifest)
    zero_bypass_receipt = _read_json(args.zero_bypass_receipt)
    _validate_raw_corpus_binding(manifest, raw_receipt)
    frozen_schema_manifest_sha256, zero_bypass_receipt_sha256 = validate_frozen_schema_gate(
        frozen_schema_manifest,
        zero_bypass_receipt,
    )
    # This authenticates the *current* rev7 reuse boundary while keeping the
    # actual census artifacts exactly in their frozen rev5 schemas.  Do not
    # record this view in a receipt: it is a read-only consumer check, not a
    # new historical artifact or an excuse to retag predecessor bytes.
    _validate_revision_7_census_predecessor_inputs(
        raw_manifest=manifest,
        raw_receipt=raw_receipt,
        frozen_schema_manifest=frozen_schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
    )
    _verify_cg_runtime(args.cg_runtime_root)
    # Full mode rebuilds these independently in each private day lane.  The
    # serial sample retains the lightweight direct path.
    token_builder = _token_abi_builder() if args.mode == "sample" else None
    transition_index = _load_transition_index(args.engine_transition_jsonl) if args.mode == "sample" else {}
    archives = _manifest_archives(manifest, archive_root=args.archive_root)
    if args.mode == "full":
        try:
            plan = revision_7_parallel_execution_plan(workers=args.day_workers)
        except Rev7PredecessorCompatibilityError as exc:
            raise RunnerError("revision-7 full-census parallel plan is unavailable") from exc
        _validate_revision_7_parallel_day_plan(plan, archives)
    output = _output_dir(args.output_root, args.run_id)
    raw_inventory_accumulator = _PhaseAInventoryAccumulator(
        inventory_scope=PHASE_A_RAW_REPLAY_INVENTORY_SCOPE
    )
    actor_selection_inventory_accumulator = _PhaseAInventoryAccumulator(
        inventory_scope=PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE
    )
    raw_episode_count = 0
    raw_outer_observation_count = 0
    processed_frame_count = 0
    actor_visible_frame_count = 0
    alakazam_actor_visible_frame_count = 0
    excluded_non_alakazam_actor_visible_frame_count = 0
    forced_frame_count = 0
    collision_audit_action_stage_count = 0
    materialized_action_stage_count = 0
    acting_deck_distribution: Counter[str] = Counter()
    matchup_distribution: Counter[str] = Counter()
    per_day_episode_count: Counter[str] = Counter()
    worker_resource_observations: list[Mapping[str, Any]] = []
    lease_root = _experiment_lease_root()
    with _ExperimentLease(lease_root) as lease:
        collision_audit_shards = _ContentAddressedShardWriter(
            output / "collision-audit-records",
            bucket_count=args.bucket_count,
            raw_manifest_sha256=canonical_sha256(manifest),
            frozen_schema_manifest_sha256=frozen_schema_manifest_sha256,
            zero_bypass_receipt_sha256=zero_bypass_receipt_sha256,
            record_scope=RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
        )
        shards = _ContentAddressedShardWriter(
            output / "refeatured-records",
            bucket_count=args.bucket_count,
            raw_manifest_sha256=canonical_sha256(manifest),
            frozen_schema_manifest_sha256=frozen_schema_manifest_sha256,
            zero_bypass_receipt_sha256=zero_bypass_receipt_sha256,
            record_scope=RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
        )
        completed_stream = False
        try:
            if args.mode == "full":
                # The parent owns the one global lease and final publisher.
                # All 24 workers receive disjoint UTC-day lanes and can only
                # write checksummed private spools under this new run root.
                private_lane_root = output / ".private-day-lanes"
                private_lane_root.mkdir(parents=False, exist_ok=False)
                (private_lane_root / "collision-audit").mkdir(parents=False, exist_ok=False)
                (private_lane_root / "materialized").mkdir(parents=False, exist_ok=False)
                worker_archives = tuple(
                    (dict(archive), str(path.resolve())) for archive, path in archives
                )
                worker_engine_path = (
                    str(args.engine_transition_jsonl.resolve())
                    if args.engine_transition_jsonl is not None
                    else None
                )
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.day_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            _run_private_day_lane,
                            lane_index=lane_index,
                            lane_count=args.day_workers,
                            archives=worker_archives,
                            spool_root=str(private_lane_root),
                            bucket_count=args.bucket_count,
                            engine_transition_jsonl=worker_engine_path,
                            cg_runtime_root=str(args.cg_runtime_root.resolve()),
                        )
                        for lane_index in range(args.day_workers)
                    ]
                    lane_results = sorted(
                        (future.result() for future in futures),
                        key=lambda result: _exact_int(
                            result.get("lane_index"), field="parallel lane index"
                        ),
                    )
                expected_lane_fields = {
                    "lane_index",
                    "raw_episode_count",
                    "raw_outer_observation_count",
                    "processed_frame_count",
                    "actor_visible_frame_count",
                    "alakazam_actor_visible_frame_count",
                    "excluded_non_alakazam_actor_visible_frame_count",
                    "forced_frame_count",
                    "collision_audit_action_stage_count",
                    "materialized_action_stage_count",
                    "acting_deck_distribution",
                    "matchup_distribution",
                    "per_day_raw_episode_count",
                    "raw_inventory",
                    "actor_selection_inventory",
                    "collision_audit_private_spool_shards",
                    "materialized_private_spool_shards",
                    "resource_observation",
                }
                for result in lane_results:
                    if not isinstance(result, Mapping) or set(result) != expected_lane_fields:
                        raise RunnerError("parallel day-lane result field inventory drifted")
                    raw_inventory = result.get("raw_inventory")
                    actor_inventory = result.get("actor_selection_inventory")
                    if not isinstance(raw_inventory, Mapping) or not isinstance(actor_inventory, Mapping):
                        raise RunnerError("parallel day-lane inventory is malformed")
                    raw_inventory_accumulator.merge_final(raw_inventory)
                    actor_selection_inventory_accumulator.merge_final(
                        actor_inventory, allow_empty=True
                    )
                    raw_episode_count += _exact_int(result.get("raw_episode_count"), field="parallel raw episode count")
                    raw_outer_observation_count += _exact_int(
                        result.get("raw_outer_observation_count"), field="parallel raw observation count"
                    )
                    processed_frame_count += _exact_int(result.get("processed_frame_count"), field="parallel processed frame count")
                    actor_visible_frame_count += _exact_int(result.get("actor_visible_frame_count"), field="parallel actor-visible frame count")
                    alakazam_actor_visible_frame_count += _exact_int(
                        result.get("alakazam_actor_visible_frame_count"), field="parallel Alakazam frame count"
                    )
                    excluded_non_alakazam_actor_visible_frame_count += _exact_int(
                        result.get("excluded_non_alakazam_actor_visible_frame_count"), field="parallel excluded frame count"
                    )
                    forced_frame_count += _exact_int(result.get("forced_frame_count"), field="parallel forced frame count")
                    collision_audit_action_stage_count += _exact_int(
                        result.get("collision_audit_action_stage_count"),
                        field="parallel collision audit action stage count",
                    )
                    materialized_action_stage_count += _exact_int(
                        result.get("materialized_action_stage_count"),
                        field="parallel materialized action stage count",
                    )
                    for counter, target, label in (
                        (result.get("acting_deck_distribution"), acting_deck_distribution, "acting deck"),
                        (result.get("matchup_distribution"), matchup_distribution, "matchup"),
                        (result.get("per_day_raw_episode_count"), per_day_episode_count, "per-day episode"),
                    ):
                        if not isinstance(counter, Mapping):
                            raise RunnerError(f"parallel {label} distribution is malformed")
                        for key, value in counter.items():
                            if not isinstance(key, str):
                                raise RunnerError(f"parallel {label} distribution key is malformed")
                            target[key] += _exact_int(value, field=f"parallel {label} distribution count")
                    resource = result.get("resource_observation")
                    if not isinstance(resource, Mapping):
                        raise RunnerError("parallel day-lane resource observation is malformed")
                    worker_resource_observations.append(resource)
                merged_collision_audit_record_count = _merge_private_day_lane_spools(
                    lane_results,
                    spool_root=private_lane_root / "collision-audit",
                    writer=collision_audit_shards,
                    spool_field="collision_audit_private_spool_shards",
                )
                merged_materialized_record_count = _merge_private_day_lane_spools(
                    lane_results,
                    spool_root=private_lane_root / "materialized",
                    writer=shards,
                    spool_field="materialized_private_spool_shards",
                )
                if (
                    merged_collision_audit_record_count != collision_audit_shards.record_count
                    or merged_materialized_record_count != shards.record_count
                ):
                    raise RunnerError("parallel private-spool merge record count does not close")
                telemetry.sample()
            else:
                assert token_builder is not None
                for archive, member, payload in _iter_episode_payloads(
                    archives,
                    day_shard_index=args.day_shard_index,
                    day_shard_count=args.day_shard_count,
                    max_episodes=args.max_episodes,
                    telemetry=telemetry,
                ):
                    raw_episode_count += 1
                    per_day_episode_count[str(archive["date"])] += 1
                    # Inventory every raw two-seat observation before masked-stage
                    # extraction.  The accumulator retains paths/types/counts only;
                    # it never emits raw values or a replay surface.
                    for raw_observation in raw_observations_from_recorded_episode(payload):
                        raw_inventory_accumulator.add(raw_observation)
                        raw_outer_observation_count += 1
                    coverage = recorded_episode_frame_coverage(payload)
                    actor_visible_frame_count += coverage["actor_visible_selection_frame_count"]
                    forced_frame_count += coverage["forced_selection_frame_count"]
                    source = {
                        "source_archive_sha256": archive.get("sha256"),
                        "source_archive_date": archive.get("date"),
                        "source_member": member,
                        "source_episode_schema": payload.get("schema_version"),
                        "raw_transition_target_only": True,
                    }
                    descriptors = stage_descriptors_from_recorded_episode(payload, source=source)
                    represented_frames = {int(descriptor["source"]["env_step"]) for descriptor in descriptors}
                    if len(represented_frames) != coverage["actor_visible_selection_frame_count"]:
                        raise RunnerError("an actor-visible selection frame was omitted during re-featurization")
                    eligible_descriptors = [
                        descriptor
                        for descriptor in descriptors
                        if descriptor["source"].get("acting_seat_setup_deck_contains_card_743") is True
                    ]
                    eligible_frames = {
                        int(descriptor["source"]["env_step"])
                        for descriptor in eligible_descriptors
                    }
                    alakazam_actor_visible_frame_count += len(eligible_frames)
                    excluded_non_alakazam_actor_visible_frame_count += len(represented_frames) - len(eligible_frames)
                    processed_frame_count += len(eligible_frames)
                    observations_by_env_step: dict[int, Mapping[str, Any]] = {}
                    for descriptor in descriptors:
                        env_step = int(descriptor["source"]["env_step"])
                        observation = descriptor["observation"]
                        prior = observations_by_env_step.setdefault(env_step, observation)
                        if prior != observation:
                            raise RunnerError("factorized stages disagree on a frame's masked observation")
                    for observation in observations_by_env_step.values():
                        actor_selection_inventory_accumulator.add(observation)
                    for descriptor in descriptors:
                        observation = descriptor["observation"]
                        source_row = dict(descriptor["source"])
                        materialization_eligible = (
                            source_row.get("acting_seat_setup_deck_contains_card_743") is True
                        )
                        source_row["row_materialization_eligible"] = materialization_eligible
                        source_row["row_materialization_exclusion_reason"] = (
                            "eligible_same_acting_seat_setup_deck_contains_card_743"
                            if materialization_eligible
                            else (
                                "missing_or_malformed_same_acting_seat_setup_deck"
                                if source_row.get("acting_deck_multiset_sha256") is None
                                else "same_acting_seat_setup_deck_lacks_card_743"
                            )
                        )
                        acting_deck = source_row.get("acting_deck_multiset_sha256")
                        opponent_deck = source_row.get("opponent_deck_multiset_sha256")
                        if isinstance(acting_deck, str):
                            acting_deck_distribution[acting_deck] += 1
                        if isinstance(opponent_deck, str):
                            matchup_distribution[opponent_deck] += 1
                        public_hash = canonical_public_observation_hash(observation)
                        transition_by_action = {
                            action_key_sha256(candidate): transition_index[(public_hash, action_key_sha256(candidate))]
                            for candidate in descriptor["candidates"]
                            if (public_hash, action_key_sha256(candidate)) in transition_index
                        }
                        stage_records = build_stage_option_records(
                            observation,
                            descriptor["candidates"],
                            stage_prefix=descriptor["stage_prefix"],
                            selected_candidate_index=descriptor["selected_candidate_index"],
                            transition_by_action=transition_by_action,
                            token_builder=token_builder,
                            source=source_row,
                        )
                        for record in stage_records:
                            collision_audit_shards.write(record)
                            if materialization_eligible:
                                shards.write(record)
                        collision_audit_action_stage_count += 1
                        if materialization_eligible:
                            materialized_action_stage_count += 1
                    telemetry.sample()
            completed_stream = True
        finally:
            if completed_stream:
                collision_audit_shards.close()
                shards.close()
            else:
                collision_audit_shards.abort()
                shards.abort()
    collision_audit_manifest = collision_audit_shards.manifest()
    refeatured_manifest = shards.manifest()
    if raw_episode_count == 0 or processed_frame_count != alakazam_actor_visible_frame_count:
        raise RunnerError("census did not cover every exact-list Alakazam actor-visible selection frame")
    inventory = raw_inventory_accumulator.final()
    actor_selection_inventory = actor_selection_inventory_accumulator.final()
    validate_phase_a_inventory(inventory)
    validate_phase_a_inventory(actor_selection_inventory)
    if inventory["raw_observation_count"] != raw_outer_observation_count:
        raise RunnerError("raw Phase A schema inventory did not cover every raw outer observation")
    if (
        actor_selection_inventory["raw_observation_count"]
        != actor_visible_frame_count
    ):
        raise RunnerError(
            "actor-visible Phase A schema inventory did not cover every selected frame"
        )
    frame_coverage = {
        "actor_visible_selection_frame_count": actor_visible_frame_count,
        # Backward-compatible raw-audit closure: descriptors were constructed
        # and checked for every actor-visible frame before the deck filter.
        "processed_actor_visible_selection_frame_count": actor_visible_frame_count,
        "all_actor_visible_and_forced_frames_included": True,
        "exact_list_alakazam_actor_visible_selection_frame_count": alakazam_actor_visible_frame_count,
        "processed_exact_list_alakazam_actor_visible_selection_frame_count": processed_frame_count,
        "excluded_non_alakazam_actor_visible_selection_frame_count": excluded_non_alakazam_actor_visible_frame_count,
        "forced_selection_frame_count": forced_frame_count,
        "all_exact_list_alakazam_actor_visible_frames_included": (
            processed_frame_count == alakazam_actor_visible_frame_count
        ),
        "collision_audit_scope": "all_actor_visible_decisions_all_episodes_both_seats",
        "training_feature_scope": "same_acting_seat_setup_deck_literal_card_743_only",
        "exact_new_list_canonical_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
    }
    re_featurization = {
        "raw_episode_count": raw_episode_count,
        "raw_outer_observation_count": raw_outer_observation_count,
        "raw_schema_inventory_sha256": canonical_sha256(inventory),
        "raw_schema_inventory_scope": inventory["inventory_scope"],
        "raw_schema_inventory_rejected_observation_count": inventory[
            "rejected_observation_count"
        ],
        "actor_visible_selection_inventory_sha256": canonical_sha256(
            actor_selection_inventory
        ),
        "actor_visible_selection_inventory_scope": actor_selection_inventory[
            "inventory_scope"
        ],
        "actor_visible_selection_inventory_observation_count": actor_selection_inventory[
            "raw_observation_count"
        ],
        "raw_observation_values_persisted": False,
        "factorized_stage_count": materialized_action_stage_count,
        "option_record_count": shards.record_count,
        "content_addressed_record_manifest_sha256": canonical_sha256(refeatured_manifest),
        "content_addressed_record_shard_count": refeatured_manifest["shard_count"],
        "row_materialization_scope": RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
        "collision_audit_factorized_stage_count": collision_audit_action_stage_count,
        "collision_audit_option_record_count": collision_audit_shards.record_count,
        "collision_audit_record_manifest_sha256": canonical_sha256(collision_audit_manifest),
        "collision_audit_record_shard_count": collision_audit_manifest["shard_count"],
        "collision_audit_scope": RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
        "maximum_record_shard_size_bytes": MAX_TRANSFER_SHARD_BYTES,
        "per_day_raw_episode_count": dict(sorted(per_day_episode_count.items())),
        "execution_parallelism": {
            "mode": (
                "parent_held_global_lease_24_private_utc_day_process_lanes"
                if args.mode == "full"
                else "single_serial_sample_lane"
            ),
            "requested_day_worker_count": args.day_workers,
            "parent_global_lease_only": True,
            "private_worker_final_artifact_publication": False,
            "deterministic_merge_order": "lane_index_then_public_token_bucket_then_canonical_source_record_order",
            "worker_resource_observation_sha256": canonical_sha256(worker_resource_observations),
        },
        "full_30_day_complete": (
            args.mode == "full"
            and raw_episode_count == manifest.get("total_validated_episodes")
        ),
    }
    report = _aggregate_bucket_reports(
        collision_audit_shards.shard_paths(),
        decision_count=collision_audit_action_stage_count,
        inventory_only=args.mode == "sample" and args.engine_transition_jsonl is None,
        raw_manifest_sha256=canonical_sha256(manifest),
        frozen_schema_manifest_sha256=frozen_schema_manifest_sha256,
        zero_bypass_receipt_sha256=zero_bypass_receipt_sha256,
        record_scope=RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
    )
    resource_observation = telemetry.final()
    resource_observation["execution_identity"] = _validate_elmo_execution_identity(
        execution_identity
    )
    worker_peak_rss_charge = sum(
        _exact_int(
            resource.get("actual_peak_rss_bytes"),
            field="parallel worker peak RSS",
        )
        for resource in worker_resource_observations
    )
    aggregate_peak_rss_charge = (
        _exact_int(resource_observation.get("actual_peak_rss_bytes"), field="parent peak RSS")
        + worker_peak_rss_charge
    )
    if aggregate_peak_rss_charge > HARD_EXPERIMENT_MEMORY_BYTES:
        raise RunnerError("conservative parent-plus-day-worker RAM charge exceeded 96 GiB")
    resource_observation["aggregate_experiment_ram_accounting"] = {
        "method": (
            "parent_global_exclusive_flock_plus_conservative_sum_of_24_private_day_worker_peak_rss"
            if args.mode == "full"
            else "single_global_r298_memory_heavy_process_under_exclusive_flock"
        ),
        "global_lease_path": str(lease.path),
        "global_lease_namespace": str(lease_root),
        "aggregate_child_processes": len(worker_resource_observations),
        "worker_peak_rss_charge_bytes": worker_peak_rss_charge,
        "worker_resource_observation_sha256": canonical_sha256(worker_resource_observations),
        "aggregate_peak_bytes": aggregate_peak_rss_charge,
        "hard_ceiling_bytes": HARD_EXPERIMENT_MEMORY_BYTES,
        "one_memory_heavy_phase_exclusive_lease": True,
    }
    run_identity = canonical_sha256(
        {
            "schema": RUN_SCHEMA,
            "config_sha256": sha256_file(args.config),
            "owner_revision": R298_OWNER_REVISION,
            "goal_revision": REVISION_5_GOAL_REVISION,
            "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
            "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
            "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
            "revision_5_predecessor_classification": revision_5_predecessor_classification(),
            "raw_manifest_sha256": canonical_sha256(manifest),
            "raw_receipt_sha256": canonical_sha256(raw_receipt),
            "frozen_schema_manifest_sha256": frozen_schema_manifest_sha256,
            "zero_bypass_receipt_sha256": zero_bypass_receipt_sha256,
            "phase_a_raw_observation_inventory_sha256": canonical_sha256(inventory),
            "actor_visible_selection_inventory_sha256": canonical_sha256(
                actor_selection_inventory
            ),
            "collision_audit_record_manifest_sha256": canonical_sha256(
                collision_audit_manifest
            ),
            "materialized_record_manifest_sha256": canonical_sha256(
                refeatured_manifest
            ),
            "collision_audit_report_sha256": canonical_sha256(report),
            "mode": args.mode,
            "day_shard_index": args.day_shard_index,
            "day_shard_count": args.day_shard_count,
            "day_workers": args.day_workers,
            "private_day_worker_resource_observation_sha256": canonical_sha256(
                worker_resource_observations
            ),
            "engine_transition_jsonl_sha256": sha256_file(args.engine_transition_jsonl) if args.engine_transition_jsonl else None,
            "execution_identity": _validate_elmo_execution_identity(execution_identity),
        }
    )
    # A sample remains explicitly inventory-only.  Full mode has streamed its
    # complete raw schema inventory in this same pass and can issue a strict
    # census receipt when simulator evidence is complete.
    receipt: dict[str, Any]
    try:
        receipt = make_receipt(
            report=report,
            inventory=inventory,
            raw_expert_corpus_manifest=manifest,
            raw_expert_corpus_receipt=raw_receipt,
            frozen_schema_manifest=frozen_schema_manifest,
            zero_bypass_receipt=zero_bypass_receipt,
            current_token_abi_source_sha256=R274_EXACT_FEATURES_SOURCE_SHA256,
            run_identity_sha256=run_identity,
            raw_episode_count=raw_episode_count,
            public_matchup_distribution=dict(matchup_distribution),
            acting_deck_distribution=dict(acting_deck_distribution),
            frame_coverage=frame_coverage,
            re_featurization=re_featurization,
        )
    except CollisionCensusError as exc:
        receipt = {
            "schema": R298_COLLISION_RECEIPT_SCHEMA,
            "status": "blocked_phase_a_incomplete",
            "owner_revision": R298_OWNER_REVISION,
            "goal_revision": REVISION_5_GOAL_REVISION,
            "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
            "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
            "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
            "revision_5_predecessor_classification": revision_5_predecessor_classification(),
            "reason": str(exc),
            "raw_expert_corpus_receipt_sha256": canonical_sha256(raw_receipt),
            "run_identity_sha256": run_identity,
            "runtime_authority": {"elmo_only": True, "create_only": True, "production_activation": False, "inzi_mutation": False},
        }
    # Attach the same complete proof to an explicitly blocked receipt too, so
    # a later audit can distinguish evidence incompleteness from a silently
    # skipped raw/frame inventory.  A passed receipt already carries these
    # fields through ``make_receipt`` with the same canonical values.
    receipt["frame_coverage"] = frame_coverage
    receipt["re_featurization"] = re_featurization
    receipt["resource_observation"] = resource_observation
    receipt["execution_identity"] = _validate_elmo_execution_identity(execution_identity)
    receipt["source_stratification_only"] = {
        "opponent_deck_multiset_is_policy_input": False,
        "acting_deck_multiset_is_policy_input": False,
        "exact_new_list_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
    }
    revision_5_validation_receipt: Mapping[str, Any] | None = None
    if (
        args.mode == "full"
        and receipt.get("status") == "passed_no_actionable_public_semantic_collision"
        and receipt["re_featurization"].get("full_30_day_complete") is True
    ):
        revision_5_validation_receipt = make_revision_5_census_validation_receipt(
            census_receipt=receipt,
            raw_expert_corpus_manifest=manifest,
            raw_expert_corpus_receipt=raw_receipt,
            frozen_schema_manifest=frozen_schema_manifest,
            zero_bypass_receipt=zero_bypass_receipt,
        )
        if (
            revision_5_validation_receipt.get("schema")
            != R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA
        ):
            raise RunnerError("revision-5 census validation receipt schema drifted")
        validate_revision_5_census_validation_receipt(revision_5_validation_receipt)
    _write_create_only_json(output / "phase_a_inventory.json", inventory)
    _write_create_only_json(
        output / "actor_visible_selection_inventory.json", actor_selection_inventory
    )
    _write_create_only_json(
        output / "collision_audit_record_manifest.json", collision_audit_manifest
    )
    _write_create_only_json(output / "refeatured_record_manifest.json", refeatured_manifest)
    _write_create_only_json(output / "collision_report.json", report)
    _write_create_only_json(output / "receipt.json", receipt)
    if revision_5_validation_receipt is not None:
        _write_create_only_json(
            output / "revision_5_census_validation_receipt.json",
            revision_5_validation_receipt,
        )
    print(json.dumps({"schema": RUN_SCHEMA, "status": receipt["status"], "output_root": str(output), "receipt_sha256": canonical_sha256(receipt)}, sort_keys=True))
    return 2 if report.get("status") == STATUS_FAILED_COLLISION else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.print_contract:
            if args.execute:
                raise RunnerError("--print-contract is inspection-only and cannot be combined with --execute")
            config = _load_config(args.config)
            print(
                json.dumps(
                    {
                        "schema": RUN_SCHEMA,
                        "status": "validated_inert_contract_no_execution",
                        "owner_revision": R298_OWNER_REVISION,
                        "goal_revision": REVISION_5_GOAL_REVISION,
                        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
                        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
                        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
                        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
                        "phase_a_schema_inventory": config["phase_a_schema_inventory"],
                        "frozen_schema_gate": frozen_schema_gate_contract(),
                    },
                    sort_keys=True,
                )
            )
            return 0
        # This is intentionally before config/output/lease/archive processing.
        # A default/off invocation remains inspectable from another host, but
        # no materialization-capable command can even inspect its run inputs
        # until the canonical Elmo host identity has been established.
        execution_identity = _verified_elmo_execution_identity() if args.execute else None
        config = _load_config(args.config)
        if not args.execute:
            print(json.dumps({"schema": RUN_SCHEMA, "status": "create_only_off_no_execution"}, sort_keys=True))
            return 0
        assert execution_identity is not None
        if args.phase == "census-completion":
            return _run_census_completion_validation(args, execution_identity)
        telemetry = _ResourceTelemetry()
        if args.phase == "manifest":
            return _run_manifest(args, config, telemetry, execution_identity)
        return _run_census(args, config, telemetry, execution_identity)
    except (RunnerError, CollisionCensusError, OSError) as exc:
        print(json.dumps({"schema": RUN_SCHEMA, "status": "failed_closed", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
