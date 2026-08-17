"""Isolated executor for the r197/r198 RTP three-arm evaluation.

This module is intentionally *not* a serving path.  It consumes a frozen
three-arm manifest and an evaluation-only authority, reuses one already sealed
true-engine snapshot for each scheduled cell, and launches a fresh ``exec``
child for each cell/arm.  The child has exactly one game to play and exits.  This matters for
the official controls: several of them keep mutable module globals, so merely
resetting a callable in a worker pool is not a valid paired comparison.

The concrete production factory is deliberately outside this module.  A
factory named by ``module:attribute`` supplies the sealed decks, fresh arm
agents, and the private snapshot engine.  Its public contract is documented by
``RtpThreeArmEvaluationFactory`` below.  Keeping the generic controller here
lets it fail closed before any engine mutation when an immutable identity,
source-isolation proof, or snapshot capability is absent.

Neither this module nor its CLI changes a selector, launches a managed
trainer, publishes a checkpoint, or submits to Kaggle.
"""

from __future__ import annotations

import argparse
import codecs
import concurrent.futures
import contextlib
import hashlib
import importlib
import json
import math
import os
import pickle
import random
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .rtp_three_arm_evaluation import (
    ARMS,
    MANIFEST_SCHEMA,
    PAIRING_CAPABILITY_SCHEMA,
    RTPThreeArmEvaluationError,
    canonical_digest,
    file_digest,
    verify_manifest_frozen_artifacts,
)
from .engine_rebuild.rtp_pairing_snapshot import (
    CAPABILITY_SCHEMA as ENGINE_PAIRING_CAPABILITY_SCHEMA,
    PROBE_SCHEMA as ENGINE_PAIRING_PROBE_SCHEMA,
    SNAPSHOT_SEAL_SCHEMA,
    PairingArtifactSet,
    RTPPairingSnapshotError,
    snapshot_abi_contract,
    snapshot_abi_sha256,
    verify_build_receipt,
)


RUNNER_SCHEMA = "poke_bot.recursive_turn_planner.three_arm_evaluation_runner/v1"
WORKER_REQUEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_worker_request/v1"
)
WORKER_RESPONSE_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_worker_response/v1"
)
AUTHORITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_authorization/v1"
)
EXECUTION_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_execution_receipt/v1"
)
RESULTS_SCHEMA = "poke_bot.recursive_turn_planner.three_arm_evaluation_results/v1"
TRANSCRIPT_SCHEMA = "poke_bot.recursive_turn_planner.three_arm_transcript/v1"
FAILED_WORKER_EVIDENCE_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_failed_worker_evidence/v1"
)
EXECUTION_ATTEMPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_execution_attempt/v1"
)
COHORT_SCHEMA = "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1"
SOURCE_EXCLUSION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_evaluation_only_source_exclusion/v1"
)
PLANNER_PREFLIGHT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_planner_pass_preflight/v1"
)
PAIRING_CASE_BINDING_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_pairing_case_binding/v1"
)
EVALUATION_CG_CLOSURE_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1"
)
EVALUATION_CG_SOURCE_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1"
)
EVALUATION_CG_CLOSURE_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure_manifest/v1"
)
EVALUATION_CG_METADATA_PARITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1"
)
R198_SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_rtp_r198_eval_source_snapshot/v1"
R198_SOURCE_SNAPSHOT_MANIFEST = "r198-eval-source-snapshot-manifest.json"

CANONICAL_DIRECT_ARM = "direct_bridge_recursive_disabled"
LEGACY_DIRECT_ARM = "direct_bridge"
FORCED_TURN_ORDER_CONTROL = "forced_go_first_contract"
# This is the r198 complete-ordered-action materialization cap, not a
# factorized-policy support limit.  A live selection above it is legal only as
# the explicitly audited, planner-ineligible stratum below; it is never an
# excuse to raise the cap or to truncate a complete-action list.
R198_COMPLETE_ORDERED_ACTION_CAP = 1024
OVER_CAP_FACTORIZED_FALLBACK_MODE = "over_cap_factorized_fallback"
OVER_CAP_FACTORIZED_FALLBACK_REASON = (
    "complete_ordered_action_space_over_cap"
)
_ARM_CONTROL_ENV = frozenset(
    {
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER",
        "POKEBOT_RTP_CHECKPOINT",
        "POKEBOT_RTP_SIZING_PROFILE",
        "POKEBOT_RTP_MAX_ACTION_COMBOS",
        "POKEBOT_RTP_SERVING_QUALIFIED",
        "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256",
        "POKEBOT_RTP_PROMOTION_RECEIPT",
        "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256",
        "POKEBOT_RTP_FORCE_DIRECT_BRIDGE_ONLY",
    }
)
# Public read-only view used by the concrete factory and focused tests.  The
# underscore-prefixed value remains the implementation's allowlist source.
ARM_CONTROL_ENV = _ARM_CONTROL_ENV
R198_BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
_FACTORY_ENV_EXACT_KEYS = frozenset(
    {
        "CG_LIB_PATH",
        "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT",
        "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256",
        # The sealed r198 candidate snapshot, supplied by the factory.  This
        # is distinct from the per-arm evaluator action fence below.
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT",
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256",
    }
)

# These values are written by the parent immediately before ``exec`` rather
# than supplied by a factory.  They form the short-lived, evaluation-only
# action fence consumed by the policy/bridge layer.  They never appear in a
# submission or ordinary training environment.
_RUNNER_ACTION_FENCE_ENV = frozenset(
    {
        "POKEBOT_R198_EVAL_ACTION_FENCE",
        "POKEBOT_R198_EVAL_ACTION_FENCE_SHA256",
        "POKEBOT_R198_EVAL_LAUNCH_NONCE",
        "POKEBOT_R198_EVAL_PROCESS_ID",
        "POKEBOT_R198_EVAL_PROCESS_START_TICKS",
    }
)
ACTION_FENCE_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluator_arm_runtime_contract/v1"
)
EVALUATION_ACTION_EXECUTION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluation_action_execution/v1"
)
R198_CANDIDATE_SNAPSHOT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluation_candidate_snapshot/v1"
)

_FAILED_WORKER_CAPTURE_BYTES = 8 * 1024
# The response channel is deliberately bounded independently of the small
# diagnostic sample.  A real 4,000-decision evaluator response can be large,
# but no child is allowed to make the parent retain arbitrary JSON in memory.
_WORKER_RESPONSE_PARSE_MAX_BYTES = 16 * 1024 * 1024
_WORKER_STREAM_READ_BYTES = 64 * 1024
_FAILED_WORKER_AUTHORITY_DENIALS = {
    "training_eligible": False,
    "replay_eligible": False,
    "serving_eligible": False,
    "serving_change_authorized": False,
    "selector_change_authorized": False,
    "selector_authority": False,
    "action_authority_authorized": False,
    "action_authority_enabled": False,
    "kaggle_submission_authorized": False,
    "submission_eligible": False,
    "submission_publication_authorized": False,
    "promotion_eligible": False,
    "promotion_authorized": False,
    "promotion_authority": False,
    "self_promotion_allowed": False,
    "self_promotion_performed": False,
    "evaluator_authority_authorized": False,
    "evaluation_result_authorized": False,
    "result_row_authorized": False,
    "execution_receipt_authorized": False,
    "transcript_authorized": False,
}


class RTPThreeArmRunnerError(RuntimeError):
    """Raised before a result can be treated as evaluation evidence."""


class _ParentBoundWorkerResponse(dict[str, Any]):
    """A normal worker-response mapping with parent-only failure provenance.

    The metadata remains an object attribute so it cannot flow into the
    child-shaped response, a row, transcript, receipt, or final result.
    """

    __slots__ = ("failed_worker_context",)

    def __init__(
        self, response: Mapping[str, Any], *, failed_worker_context: Mapping[str, Any]
    ) -> None:
        super().__init__(response)
        self.failed_worker_context = dict(failed_worker_context)


@dataclass
class _WorkerStreamCapture:
    """Bounded raw-byte capture for one child pipe.

    The byte counter and digest always cover the complete pipe stream.  Only
    a fixed diagnostic sample is retained, and stdout alone retains a bounded
    complete copy for the one JSON response parser.  Strict UTF-8 is checked
    incrementally so malformed output cannot escape the evidence boundary as
    a decoder exception.
    """

    parse_limit_bytes: int | None = None
    capture_limit_bytes: int = _FAILED_WORKER_CAPTURE_BYTES
    total_bytes: int = 0
    parse_limit_exceeded: bool = False
    utf8_error: bool = False
    _digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)
    _decoder: Any = field(init=False, repr=False)
    _full_sample: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _head: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _tail: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _parse: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _head_limit: int = field(init=False, repr=False)
    _tail_limit: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capture_limit_bytes < 128:
            raise RTPThreeArmRunnerError("failed-worker output capture limit is too small")
        if self.parse_limit_bytes is not None and self.parse_limit_bytes < 1:
            raise RTPThreeArmRunnerError("worker response parse limit must be positive")
        marker_bytes = len(b"\n...[truncated; full output bound by sha256]...\n")
        self._head_limit = (self.capture_limit_bytes - marker_bytes) // 2
        self._tail_limit = self.capture_limit_bytes - marker_bytes - self._head_limit
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def feed(self, raw: bytes) -> None:
        if not isinstance(raw, bytes):
            raise RTPThreeArmRunnerError("worker pipe yielded non-bytes output")
        if not raw:
            return
        self.total_bytes += len(raw)
        self._digest.update(raw)
        if len(self._full_sample) < self.capture_limit_bytes:
            remaining = self.capture_limit_bytes - len(self._full_sample)
            self._full_sample.extend(raw[:remaining])
        if len(self._head) < self._head_limit:
            remaining = self._head_limit - len(self._head)
            self._head.extend(raw[:remaining])
        if len(raw) >= self._tail_limit:
            self._tail[:] = raw[-self._tail_limit :]
        else:
            self._tail.extend(raw)
            if len(self._tail) > self._tail_limit:
                del self._tail[: len(self._tail) - self._tail_limit]
        if self.parse_limit_bytes is not None and not self.parse_limit_exceeded:
            remaining = self.parse_limit_bytes - len(self._parse)
            if remaining > 0:
                self._parse.extend(raw[:remaining])
            if len(raw) > remaining:
                self.parse_limit_exceeded = True
        if not self.utf8_error:
            try:
                self._decoder.decode(raw, final=False)
            except UnicodeDecodeError:
                self.utf8_error = True

    def finish(self) -> None:
        if not self.utf8_error:
            try:
                self._decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                self.utf8_error = True

    @property
    def sha256(self) -> str:
        return "sha256:" + self._digest.hexdigest()

    def parse_text(self) -> str:
        if self.parse_limit_bytes is None:
            raise RTPThreeArmRunnerError("stderr cannot be parsed as a worker response")
        if self.parse_limit_exceeded:
            raise RTPThreeArmRunnerError("worker stdout exceeds the response parse limit")
        if self.utf8_error:
            raise RTPThreeArmRunnerError("worker output is not strict UTF-8")
        # The incremental decoder already validated this bounded byte slice;
        # decode again only to hand json.loads a normal str.
        return bytes(self._parse).decode("utf-8", errors="strict")

    @staticmethod
    def _control_safe_text(raw: bytes) -> str:
        decoded = raw.decode("utf-8", errors="replace")
        safe_parts: list[str] = []
        for character in decoded:
            codepoint = ord(character)
            if character in {"\n", "\r", "\t"}:
                safe_parts.append(character)
            elif unicodedata.category(character) == "Cc":
                if codepoint <= 0xFFFF:
                    safe_parts.append(f"\\u{codepoint:04x}")
                else:
                    safe_parts.append(f"\\U{codepoint:08x}")
            else:
                safe_parts.append(character)
        return "".join(safe_parts)

    def evidence(self) -> dict[str, Any]:
        marker = b"\n...[truncated; full output bound by sha256]...\n"
        truncated = self.total_bytes > self.capture_limit_bytes
        if truncated:
            captured = bytes(self._head) + marker + bytes(self._tail)
            capture_mode = "head_and_tail"
            captured_raw_bytes = len(self._head) + len(self._tail)
        else:
            captured = bytes(self._full_sample)
            capture_mode = "full"
            captured_raw_bytes = len(captured)
        return {
            "sha256": self.sha256,
            "bytes": self.total_bytes,
            "raw_bytes": self.total_bytes,
            "capture_limit_bytes": self.capture_limit_bytes,
            "captured_text": self._control_safe_text(captured),
            "captured_bytes": captured_raw_bytes,
            "truncated": truncated,
            "capture_mode": capture_mode,
            "strict_utf8": not self.utf8_error,
            "response_parse_limit_bytes": self.parse_limit_bytes,
            "response_parse_limit_exceeded": self.parse_limit_exceeded,
        }


@dataclass(frozen=True)
class _CapturedWorkerProcess:
    returncode: int
    child_pid: int | None
    stdout: _WorkerStreamCapture
    stderr: _WorkerStreamCapture


class _WorkerSubprocessBoundaryError(RuntimeError):
    """Carry bounded pipe state out of a failed Popen/capture boundary."""

    def __init__(
        self,
        *,
        stage: str,
        cause: BaseException,
        stdout: _WorkerStreamCapture,
        stderr: _WorkerStreamCapture,
        returncode: int | None,
        child_pid: int | None,
    ) -> None:
        super().__init__(f"{stage}: {type(cause).__name__}")
        self.stage = stage
        self.cause = cause
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.child_pid = child_pid


def _canonical_arm(arm: str) -> str:
    value = str(arm).strip()
    if value == LEGACY_DIRECT_ARM:
        return CANONICAL_DIRECT_ARM
    return value


def _manifest_arms(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return canonical arms while accepting old prepared manifests as input.

    The compiler owns the final canonical vocabulary.  The runner accepts a
    legacy prepared input only so it can emit a useful fail-closed error rather
    than silently interpreting its middle arm differently.
    """

    raw = manifest.get("arm_order", list(ARMS))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPThreeArmRunnerError("manifest arm_order must be a sequence")
    normalized = tuple(_canonical_arm(str(value)) for value in raw)
    required = ("no_rtp", CANONICAL_DIRECT_ARM, "recursive_rtp")
    if set(normalized) != set(required) or len(normalized) != 3:
        raise RTPThreeArmRunnerError(
            "manifest must contain exactly no_rtp/direct_bridge_recursive_disabled/"
            "recursive_rtp"
        )
    return required


def _lexical_absolute_path(path: str | Path, label: str) -> Path:
    """Return an absolute evidence path without silently following links.

    Evaluation evidence is deliberately physical: a verifier must not hash one
    symlink target while a later arm child imports another.  In particular,
    reject ``..`` rather than normalising it through a possibly linked parent.
    """

    raw = Path(str(path)).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if ".." in raw.parts:
        raise RTPThreeArmRunnerError(f"{label} may not contain '..'")
    return Path(os.path.abspath(os.fspath(raw)))


def _physical_path(
    path: str | Path,
    label: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    """Require an existing lexical path with no symbolic-link component."""

    source = _lexical_absolute_path(path, label)
    current = Path(source.anchor)
    for index, component in enumerate(source.parts[1:]):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise RTPThreeArmRunnerError(f"{label} does not exist: {source}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RTPThreeArmRunnerError(
                f"{label} must not traverse a symbolic link: {current}"
            )
        if index != len(source.parts[1:]) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise RTPThreeArmRunnerError(
                f"{label} has a non-directory ancestor: {current}"
            )
    try:
        final = os.lstat(source)
    except OSError as exc:
        raise RTPThreeArmRunnerError(f"{label} does not exist: {source}") from exc
    if require_file and not stat.S_ISREG(final.st_mode):
        raise RTPThreeArmRunnerError(f"{label} is not a regular file: {source}")
    if require_directory and not stat.S_ISDIR(final.st_mode):
        raise RTPThreeArmRunnerError(f"{label} is not a directory: {source}")
    return source


def _ensure_physical_directory(path: str | Path, label: str) -> Path:
    """Create a directory tree one physical component at a time."""

    directory = _lexical_absolute_path(path, label)
    current = Path(directory.anchor)
    for component in directory.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            except OSError as exc:
                raise RTPThreeArmRunnerError(
                    f"cannot create {label} directory: {current}"
                ) from exc
            try:
                metadata = os.lstat(current)
            except OSError as exc:
                raise RTPThreeArmRunnerError(
                    f"cannot inspect created {label} directory: {current}"
                ) from exc
        except OSError as exc:
            raise RTPThreeArmRunnerError(
                f"cannot inspect {label} directory: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RTPThreeArmRunnerError(
                f"{label} contains a symlink or non-directory component: {current}"
            )
    return directory


def _json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = _physical_path(path, label, require_file=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTPThreeArmRunnerError(f"cannot read {label}: {source}") from exc
    if not isinstance(value, Mapping):
        raise RTPThreeArmRunnerError(f"{label} must be a JSON object")
    return dict(value)


def _sha256_text(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _identity(path: str | Path, label: str) -> dict[str, Any]:
    source = _physical_path(path, label, require_file=True)
    return {
        "path": str(source),
        "sha256": file_digest(source),
        "bytes": source.stat().st_size,
    }


def _require_identity(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RTPThreeArmRunnerError(f"{label} must be a file identity")
    path = raw.get("path")
    digest = raw.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise RTPThreeArmRunnerError(f"{label} requires path and sha256")
    observed = _identity(path, label)
    if observed["sha256"] != digest:
        raise RTPThreeArmRunnerError(
            f"{label} checksum mismatch: expected {digest}, got {observed['sha256']}"
        )
    if "bytes" in raw and int(raw["bytes"]) != observed["bytes"]:
        raise RTPThreeArmRunnerError(f"{label} byte count mismatch")
    return observed


def _require_immutable_identity(raw: Any, label: str) -> dict[str, Any]:
    """Verify a checksum-bound file is physical and no longer writable."""

    identity = _require_identity(raw, label)
    mode = stat.S_IMODE(os.lstat(identity["path"]).st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RTPThreeArmRunnerError(f"{label} must be read-only immutable evidence")
    return {**identity, "mode": mode}


def _sealed_path_identity(path: str | Path, label: str) -> dict[str, Any]:
    """Derive and independently reverify strict immutable evidence from a path.

    Public CLI and environment boundaries intentionally carry paths, while
    manifests and worker requests carry caller-supplied checksum identities.
    Do not send a bare path through ``_require_immutable_identity``: first
    derive its physical identity, then make the mapping verifier re-read and
    rehash that exact path before accepting its strict 0444 seal.
    """

    identity = _require_immutable_identity(_identity(path, label), label)
    if identity["mode"] != 0o444:
        raise RTPThreeArmRunnerError(f"{label} must use immutable mode 0444")
    return identity


def _immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> Path:
    """Atomically create a content-addressed audit file without overwriting."""

    target = _lexical_absolute_path(path, "immutable evaluation artifact")
    _ensure_physical_directory(target.parent, "immutable evaluation artifact parent")
    wanted = payload.get(digest_key)
    if not isinstance(wanted, str) or not wanted.startswith("sha256:"):
        raise RTPThreeArmRunnerError(f"immutable payload lacks {digest_key}")
    if target.exists():
        _physical_path(target, "existing immutable evaluation artifact", require_file=True)
        existing = _json_object(target, "existing immutable evaluation artifact")
        if existing.get(digest_key) == wanted:
            return target
        raise RTPThreeArmRunnerError(
            f"immutable evaluation artifact already exists with different {digest_key}: "
            f"{target}"
        )
    encoded = json.dumps(dict(payload), sort_keys=True, indent=2) + "\n"
    temp = target.parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, target)
        except FileExistsError:
            _physical_path(target, "racing immutable evaluation artifact", require_file=True)
            existing = _json_object(target, "racing immutable evaluation artifact")
            if existing.get(digest_key) != wanted:
                raise RTPThreeArmRunnerError(
                    f"immutable evaluation artifact appeared with a different "
                    f"{digest_key}: {target}"
                )
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temp.unlink(missing_ok=True)
    return target


def _read_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity(path, "evaluation manifest")
    manifest = _json_object(identity["path"], "evaluation manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RTPThreeArmRunnerError("not an RTP three-arm evaluation manifest")
    # The harness is the single owner of broad frozen-artifact validation.
    try:
        verify_manifest_frozen_artifacts(manifest)
    except (RTPThreeArmEvaluationError, ValueError) as exc:
        raise RTPThreeArmRunnerError(f"manifest failed frozen-artifact validation: {exc}") from exc
    _manifest_arms(manifest)
    return manifest, identity


def _direct_arm_key(manifest: Mapping[str, Any]) -> str:
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping):
        raise RTPThreeArmRunnerError("manifest arms must be an object")
    if CANONICAL_DIRECT_ARM in arms:
        return CANONICAL_DIRECT_ARM
    if LEGACY_DIRECT_ARM in arms:
        return LEGACY_DIRECT_ARM
    raise RTPThreeArmRunnerError("manifest lacks the direct-bridge arm")


def _canonical_runtime_arm(arm: str, manifest: Mapping[str, Any]) -> str:
    # Preserve the key selected by the manifest for profile lookup while every
    # result and receipt uses the corrected canonical public arm ID.
    if arm == CANONICAL_DIRECT_ARM:
        return _direct_arm_key(manifest)
    return arm


def _verify_pairing_capability(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the v2 private ABI/probe chain before any arm is launched.

    The capability record is not a Boolean permission slip.  It must bind the
    exact engine, source, patch, build receipt, ABI contract, and a previously
    completed cross-process probe.  The evaluation itself never supplies that
    probe: using its own rows to bootstrap the capability would be circular.
    """

    raw = manifest.get("pairing_capability")
    if not isinstance(raw, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks pairing_capability")
    receipt = _require_immutable_identity(
        raw.get("receipt"), "pairing capability receipt"
    )
    payload = _json_object(receipt["path"], "pairing capability receipt")
    if (
        payload.get("schema") != PAIRING_CAPABILITY_SCHEMA
        or payload.get("schema") != ENGINE_PAIRING_CAPABILITY_SCHEMA
    ):
        raise RTPThreeArmRunnerError("pairing capability receipt schema is invalid")
    if payload.get("status") != "available":
        raise RTPThreeArmRunnerError("pairing capability receipt is unavailable")
    if payload.get("true_rng_pairing_available") is not True:
        raise RTPThreeArmRunnerError("true RNG pairing capability is unavailable")
    kinds = payload.get("supported_rng_kinds")
    if not isinstance(kinds, Sequence) or isinstance(kinds, (str, bytes)):
        raise RTPThreeArmRunnerError("pairing capability has no supported RNG kinds")
    if "snapshot" not in {str(kind) for kind in kinds}:
        raise RTPThreeArmRunnerError("evaluation requires a true restorable snapshot ABI")
    if sorted(str(kind) for kind in kinds) != ["snapshot"]:
        raise RTPThreeArmRunnerError("pairing capability must authorize snapshot only")
    try:
        artifacts = PairingArtifactSet(
            engine_artifact=_require_immutable_identity(
                payload.get("engine_artifact"), "pairing capability engine artifact"
            ),
            source_artifact=_require_immutable_identity(
                payload.get("source_artifact"), "pairing capability source artifact"
            ),
            patch_artifact=_require_immutable_identity(
                payload.get("patch_artifact"), "pairing capability patch artifact"
            ),
            build_artifact=_require_immutable_identity(
                payload.get("build_artifact"), "pairing capability build artifact"
            ),
        )
        # Rehash all private build inputs and cross-bind the build receipt.
        checked_artifacts = verify_build_receipt(artifacts)
    except (RTPPairingSnapshotError, TypeError, ValueError) as exc:
        raise RTPThreeArmRunnerError(
            f"pairing capability build evidence is invalid: {exc}"
        ) from exc
    abi = payload.get("abi")
    if not isinstance(abi, Mapping):
        raise RTPThreeArmRunnerError("pairing capability lacks ABI evidence")
    expected_abi = {**snapshot_abi_contract(), "canonical_abi_sha256": snapshot_abi_sha256()}
    if dict(abi) != expected_abi:
        raise RTPThreeArmRunnerError("pairing capability ABI differs from v2 contract")
    probe_identity = _require_immutable_identity(
        payload.get("probe"), "pairing capability probe"
    )
    probe = _json_object(probe_identity["path"], "pairing capability probe")
    if probe.get("schema") != ENGINE_PAIRING_PROBE_SCHEMA or probe.get("status") != "passed":
        raise RTPThreeArmRunnerError("pairing capability probe is not a passed v2 probe")
    for key, identity in (
        ("engine_artifact_sha256", checked_artifacts.engine_artifact),
        ("source_artifact_sha256", checked_artifacts.source_artifact),
        ("patch_artifact_sha256", checked_artifacts.patch_artifact),
        ("build_artifact_sha256", checked_artifacts.build_artifact),
    ):
        if probe.get(key) != identity["sha256"]:
            raise RTPThreeArmRunnerError(f"pairing capability probe differs at {key}")
    if probe.get("canonical_abi_sha256") != snapshot_abi_sha256():
        raise RTPThreeArmRunnerError("pairing capability probe ABI differs from v2")
    if probe.get("verified_rng_kinds") != ["snapshot"]:
        raise RTPThreeArmRunnerError("pairing capability probe did not verify snapshot RNG")
    for key in (
        "device_rand_false_verified",
        "requested_seed_only_rejected",
        "duplicate_restore_independent_handles",
        "all_arms_restored_or_replayed",
        "divergent_policy_true_pairing_passed",
        "delayed_restore_transcript_passed",
        "cross_process_restore_passed",
    ):
        if probe.get(key) is not True:
            raise RTPThreeArmRunnerError(f"snapshot pairing capability probe failed at {key}")
    deterministic = probe.get("deterministic_restore_probe")
    if not isinstance(deterministic, Mapping) or deterministic.get("passed") is not True:
        raise RTPThreeArmRunnerError("pairing capability lacks a passing deterministic restore probe")
    for key in (
        "initial_snapshot_fingerprint_sha256",
        "deterministic_transcript_sha256",
    ):
        value = deterministic.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise RTPThreeArmRunnerError(f"pairing deterministic probe lacks {key}")
    if int(deterministic.get("initial_snapshot_fingerprint_bytes", 0) or 0) < 1:
        raise RTPThreeArmRunnerError("pairing deterministic probe lacks snapshot bytes")
    if int(deterministic.get("transcript_steps", 0) or 0) < 1:
        raise RTPThreeArmRunnerError("pairing deterministic probe lacks transcript steps")
    return {
        "receipt": receipt,
        "payload": payload,
        "engine_artifact": dict(checked_artifacts.engine_artifact),
        "source_artifact": dict(checked_artifacts.source_artifact),
        "patch_artifact": dict(checked_artifacts.patch_artifact),
        "build_artifact": dict(checked_artifacts.build_artifact),
        "abi": dict(expected_abi),
        "probe": probe_identity,
        "probe_payload": probe,
    }


def _verify_evaluation_cg_closure(
    manifest: Mapping[str, Any], capability: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the sealed private-ABI ``cg`` closure before an arm is launched.

    A regular competition ``cg`` package calls public ``GameInitialize`` and
    therefore cannot be substituted for the pairing wrapper.  The closure
    receipt proves that the evaluated package contains the exact capability
    engine and that its curated ``sim.py`` uses the private initializer.
    """

    raw = manifest.get("evaluation_cg_closure")
    if not isinstance(raw, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks evaluation_cg_closure")
    receipt = _sealed_case_identity(raw.get("receipt"), "evaluation CG closure receipt")
    # The closure receipt describes its immutable private build artifact.
    # ``runtime_library`` is intentionally a separate identity: it is the
    # physical, relocated libcg.so inside the sealed source snapshot that the
    # child actually imports alongside ``cg.sim``.  Equal bytes alone do not
    # permit substituting another DSO path, because that would split native
    # static state between the wrapper and the imported CG package.
    runtime_library = _sealed_case_identity(
        raw.get("runtime_library"), "evaluation CG closure runtime library"
    )
    payload = _json_object(receipt["path"], "evaluation CG closure receipt")
    if (
        payload.get("schema") != EVALUATION_CG_CLOSURE_SCHEMA
        or payload.get("status") != "sealed"
    ):
        raise RTPThreeArmRunnerError("evaluation CG closure receipt schema/status is invalid")
    if payload.get("canonical_abi_sha256") != capability.get("abi", {}).get(
        "canonical_abi_sha256"
    ):
        raise RTPThreeArmRunnerError("evaluation CG closure ABI differs from pairing capability")
    if payload.get("sim_initializer_symbol") != "RtpPairingSnapshotInitialize":
        raise RTPThreeArmRunnerError("evaluation CG closure does not use private initialization")
    engine = _sealed_case_identity(
        payload.get("engine_artifact"), "evaluation CG closure engine artifact"
    )
    expected_engine = capability.get("engine_artifact")
    if not isinstance(expected_engine, Mapping) or any(
        engine.get(key) != expected_engine.get(key) for key in ("sha256", "bytes")
    ):
        raise RTPThreeArmRunnerError(
            "evaluation CG closure engine differs from pairing capability"
        )
    if payload.get("pairing_engine_artifact_sha256") != expected_engine.get("sha256"):
        raise RTPThreeArmRunnerError("evaluation CG closure misses pairing engine binding")
    if any(
        runtime_library.get(key) != expected_engine.get(key)
        for key in ("sha256", "bytes")
    ) or any(
        runtime_library.get(key) != engine.get(key) for key in ("sha256", "bytes")
    ):
        raise RTPThreeArmRunnerError(
            "evaluation CG runtime library differs from closure/capability engine bytes"
        )
    build = _sealed_case_identity(
        payload.get("pairing_build_artifact"), "evaluation CG closure build artifact"
    )
    expected_build = capability.get("build_artifact")
    if not isinstance(expected_build, Mapping) or not _identity_equal(build, expected_build):
        raise RTPThreeArmRunnerError("evaluation CG closure differs from pairing build receipt")
    if payload.get("pairing_source_artifact_sha256") != capability.get("source_artifact", {}).get(
        "sha256"
    ) or payload.get("pairing_patch_artifact_sha256") != capability.get(
        "patch_artifact", {}
    ).get("sha256"):
        raise RTPThreeArmRunnerError("evaluation CG closure source/patch binding differs")
    evidence_specs = (
        ("cg_source_manifest", EVALUATION_CG_SOURCE_MANIFEST_SCHEMA, None),
        ("closure_manifest", EVALUATION_CG_CLOSURE_MANIFEST_SCHEMA, None),
        ("metadata_parity", EVALUATION_CG_METADATA_PARITY_SCHEMA, "passed"),
    )
    evidence: dict[str, dict[str, Any]] = {}
    for field, schema, required_status in evidence_specs:
        identity = _sealed_case_identity(
            payload.get(field), f"evaluation CG closure {field}"
        )
        artifact = _json_object(identity["path"], f"evaluation CG closure {field}")
        if artifact.get("schema") != schema:
            raise RTPThreeArmRunnerError(f"evaluation CG closure {field} schema is invalid")
        if required_status is not None and artifact.get("status") != required_status:
            raise RTPThreeArmRunnerError(f"evaluation CG closure {field} is not passed")
        evidence[field] = {"identity": identity, "payload": artifact}
    parity_engine = evidence["metadata_parity"]["payload"].get("pairing_engine")
    if not isinstance(parity_engine, Mapping) or parity_engine.get("sha256") != expected_engine.get(
        "sha256"
    ):
        raise RTPThreeArmRunnerError("evaluation CG metadata parity has another engine")
    # ``closure_package_path`` in the build receipt names its private builder
    # output.  The production source snapshot intentionally relocates the
    # curated ``cg`` tree, so do not load or require that stale path here.  The
    # child verifies its snapshot-local ``CG_LIB_PATH/cg/libcg.so`` against
    # this receipt and the capability immediately before imports/restoration.
    return {
        "receipt": receipt,
        "payload": payload,
        "engine_artifact": engine,
        "runtime_library": runtime_library,
        "evidence": evidence,
    }


def _verify_engine_identity(
    engine: Any,
    capability: Mapping[str, Any],
    closure: Mapping[str, Any],
    *,
    loaded_closure_engine: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the loaded private ABI to one closure-owned DSO handle.

    ``engine.identity`` intentionally remains the capability artifact identity
    even when the wrapper loaded the byte-identical copy inside the sealed
    evaluation CG closure.  The distinct ``library_identity`` is therefore
    the authoritative proof of the actual loaded path.  Requiring it to be
    the closure's ``cg/libcg.so`` prevents equal-SHA dual-DSO loading and the
    split native globals that would invalidate snapshot pairing.
    """

    raw = getattr(engine, "identity", None)
    raw = raw() if callable(raw) else raw
    if not isinstance(raw, Mapping):
        raise RTPThreeArmRunnerError("snapshot engine does not expose an identity")
    expected = capability.get("engine_artifact")
    if not isinstance(expected, Mapping):
        raise RTPThreeArmRunnerError("pairing capability has no engine artifact identity")
    for key in ("sha256", "bytes", "mode"):
        if raw.get(key) != expected.get(key):
            raise RTPThreeArmRunnerError(f"loaded snapshot engine differs at {key}")
    # The receipt identity itself must still be the exact capability identity,
    # not merely an equal-byte arbitrary file.
    if raw.get("path") != expected.get("path"):
        raise RTPThreeArmRunnerError("snapshot engine capability identity path differs")
    loaded = getattr(engine, "library_identity", None)
    loaded = loaded() if callable(loaded) else loaded
    if not isinstance(loaded, Mapping):
        raise RTPThreeArmRunnerError("snapshot engine does not expose loaded library identity")
    # The closure receipt may originate at a private build path and later be
    # copied into the sealed source snapshot.  The worker's independently
    # verified snapshot-local CG identity, not that original build path, is
    # the only acceptable loaded DSO location.
    expected_loaded = _require_immutable_identity(
        loaded_closure_engine, "loaded evaluation CG closure engine"
    )
    for key in ("path", "sha256", "bytes", "mode"):
        if loaded.get(key) != expected_loaded.get(key):
            raise RTPThreeArmRunnerError(
                "snapshot engine did not load the evaluation CG closure libcg.so"
            )
    if loaded.get("sha256") != expected.get("sha256") or loaded.get("bytes") != expected.get("bytes"):
        raise RTPThreeArmRunnerError("loaded snapshot engine bytes differ from capability")
    abi = getattr(engine, "abi", None)
    abi = abi() if callable(abi) else abi
    if not isinstance(abi, Mapping) or dict(abi) != capability.get("abi"):
        raise RTPThreeArmRunnerError("loaded snapshot engine ABI differs from capability")
    return {**dict(raw), "loaded_path": loaded["path"]}


def _verify_evaluation_authority(
    authority_path: str | Path,
    *,
    manifest_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _sealed_path_identity(
        authority_path, "evaluation-only authority"
    )
    authority = _json_object(identity["path"], "evaluation-only authority")
    if authority.get("schema") != AUTHORITY_SCHEMA:
        raise RTPThreeArmRunnerError("evaluation authority schema is invalid")
    if authority.get("status") != "authorized_evaluation_only":
        raise RTPThreeArmRunnerError("evaluation authority is not authorized")
    for key, expected in (
        ("evaluation_only", True),
        ("training_eligible", False),
        ("replay_eligible", False),
        ("serving_change_authorized", False),
        ("selector_change_authorized", False),
        ("action_authority_authorized", False),
        ("kaggle_submission_authorized", False),
    ):
        if authority.get(key) != expected:
            raise RTPThreeArmRunnerError(
                f"evaluation authority violates isolation at {key}"
            )
    if authority.get("manifest_sha256") != manifest_identity["sha256"]:
        raise RTPThreeArmRunnerError("evaluation authority is not bound to this manifest")
    return {"identity": identity, "payload": authority}


def _write_action_fence(
    *,
    scratch: Path,
    manifest_identity: Mapping[str, Any],
    authority: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm: str,
    launch_nonce: str,
) -> dict[str, Any]:
    """Publish one immutable evaluator-only action fence for an exec child.

    This is deliberately an input to the child, not a receipt created after it
    has acted.  The bridge can therefore refuse an action if its parent
    authority, cell, arm, candidate artifacts, or one-shot launch nonce do not
    match the isolated evaluator.  The fence conveys no serving, selector,
    training, replay, or Kaggle authority.
    """

    authority_identity = authority.get("identity")
    if not isinstance(authority_identity, Mapping):
        raise RTPThreeArmRunnerError("evaluation authority has no immutable identity")
    shared = manifest.get("shared_artifacts")
    arms = manifest.get("arms")
    if not isinstance(shared, Mapping) or not isinstance(arms, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks action-fence runtime artifacts")
    runtime_key = _canonical_runtime_arm(arm, manifest)
    arm_spec = arms.get(runtime_key)
    if not isinstance(arm_spec, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks action-fence arm specification")
    parent = _require_identity(shared.get("parent_checkpoint"), "action-fence parent")
    action_sidecar = arm_spec.get("rtp_sidecar")
    direct_spec = arms.get(_direct_arm_key(manifest))
    if not isinstance(direct_spec, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks action-fence probe sidecar")
    probe_sidecar = _require_identity(
        direct_spec.get("rtp_sidecar"), "action-fence probe sidecar"
    )
    payload = {
        "schema": ACTION_FENCE_SCHEMA,
        "status": "authorized_evaluation_only",
        "manifest_sha256": manifest_identity["sha256"],
        "evaluation_authority": _identity_triplet(
            authority_identity, "action-fence evaluation authority"
        ),
        "cell_id": cell.get("cell_id"),
        "evaluation_case_id": cell.get("evaluation_case_id"),
        "opponent_id": cell.get("opponent_id"),
        "candidate_seat": cell.get("candidate_seat"),
        "arm": arm,
        "launch_nonce": launch_nonce,
        "candidate_parent_checkpoint_sha256": parent["sha256"],
        "action_attached_rtp_sidecar_sha256": (
            None
            if action_sidecar is None
            else _require_identity(
                action_sidecar, "action-fence action-attached sidecar"
            )["sha256"]
        ),
        "complexity_probe_sidecar_sha256": probe_sidecar["sha256"],
        "rtp_action_attachment_enabled": arm != "no_rtp",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "submission_eligible": False,
        "promotion_eligible": False,
    }
    material = {key: value for key, value in payload.items() if key != "runtime_contract_sha256"}
    payload["runtime_contract_sha256"] = canonical_digest(material)
    path = _immutable_json(
        scratch / "r198-evaluator-arm-runtime-contract.json",
        payload,
        digest_key="runtime_contract_sha256",
    )
    # ``_immutable_json`` returns a physical path while the sealed-identity
    # verifier intentionally accepts only a checksum-bound mapping.  Rehash
    # the just-created 0444 file before handing its identity to the child;
    # this closes the construction/consumption boundary without accepting a
    # bare path as evidence.
    identity = _sealed_case_identity(
        _identity(path, "evaluator arm runtime contract"),
        "evaluator arm runtime contract",
    )
    return {"identity": identity, "payload": payload}


def _verify_action_fence(
    raw: Any,
    *,
    manifest_identity: Mapping[str, Any],
    authority: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm: str,
    launch_nonce: str,
) -> dict[str, Any]:
    """Re-check the parent-created action fence inside the fresh child."""

    identity = _sealed_case_identity(raw, "worker evaluator arm runtime contract")
    payload = _json_object(identity["path"], "worker evaluator arm runtime contract")
    if payload.get("schema") != ACTION_FENCE_SCHEMA or payload.get("status") != "authorized_evaluation_only":
        raise RTPThreeArmRunnerError("worker evaluator action fence schema/status is invalid")
    material = {key: value for key, value in payload.items() if key != "runtime_contract_sha256"}
    if payload.get("runtime_contract_sha256") != canonical_digest(material):
        raise RTPThreeArmRunnerError("worker evaluator action fence digest differs")
    authority_identity = authority.get("identity")
    if not isinstance(authority_identity, Mapping):
        raise RTPThreeArmRunnerError("worker evaluator authority identity is invalid")
    expected = {
        "manifest_sha256": manifest_identity["sha256"],
        "evaluation_authority": _identity_triplet(
            authority_identity, "worker action-fence evaluation authority"
        ),
        "cell_id": cell.get("cell_id"),
        "evaluation_case_id": cell.get("evaluation_case_id"),
        "opponent_id": cell.get("opponent_id"),
        "candidate_seat": cell.get("candidate_seat"),
        "arm": arm,
        "launch_nonce": launch_nonce,
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "submission_eligible": False,
        "promotion_eligible": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise RTPThreeArmRunnerError(f"worker evaluator action fence differs at {key}")
    return {"identity": identity, "payload": payload}


def _evaluation_action_execution_context(
    *,
    manifest_identity: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    action_fence: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm: str,
    launch_nonce: str,
) -> dict[str, Any]:
    """Create the typed, process-bound *evaluation* action exception.

    This is intentionally not a general action-authority grant.  It exists
    only inside a verified fresh B/C worker, after the immutable schedule,
    authorization, candidate snapshot and cell-bound seal have all been
    checked.  The no-RTP child receives no such context.
    """

    if arm == "no_rtp":
        raise RTPThreeArmRunnerError("no-RTP arm must not receive action execution context")
    authority_identity = authority.get("identity")
    if not isinstance(authority_identity, Mapping):
        raise RTPThreeArmRunnerError("action execution has no authority identity")
    fence_identity = action_fence.get("identity")
    if not isinstance(fence_identity, Mapping):
        raise RTPThreeArmRunnerError("action execution has no action-fence identity")
    process = _process_identity()
    return {
        "schema": EVALUATION_ACTION_EXECUTION_SCHEMA,
        "status": "authorized_evaluation_only",
        "execution_kind": "evaluation_action_execution",
        "manifest": _identity_triplet(manifest_identity, "evaluation action manifest"),
        "evaluation_authority": _identity_triplet(
            authority_identity, "evaluation action authority"
        ),
        "runtime_contract": _identity_triplet(
            runtime_contract, "evaluation action runtime contract"
        ),
        "action_fence": _identity_triplet(
            fence_identity, "evaluation action fence"
        ),
        "cell_id": cell.get("cell_id"),
        "evaluation_case_id": cell.get("evaluation_case_id"),
        "opponent_id": cell.get("opponent_id"),
        "candidate_seat": cell.get("candidate_seat"),
        "arm": arm,
        "launch_nonce": launch_nonce,
        "process": process,
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "submission_eligible": False,
        "promotion_eligible": False,
    }


def _verify_cohort_and_source_exclusion(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the runner to the normalized v2 source-exclusion mapping only."""

    binding = manifest.get("r197_source_exclusion_binding")
    if not isinstance(binding, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks r197_source_exclusion_binding")
    cohort_identity = _require_immutable_identity(
        binding.get("evaluation_only_cohort"), "evaluation-only cohort"
    )
    cohort = _json_object(cohort_identity["path"], "evaluation-only cohort")
    required_cohort = {
        "schema": COHORT_SCHEMA,
        "status": "frozen",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
    }
    for key, expected in required_cohort.items():
        if cohort.get(key) != expected:
            raise RTPThreeArmRunnerError(f"evaluation-only cohort fails {key}")
    source_identity = cohort.get("source_identity_sha256")
    if not isinstance(source_identity, str) or not source_identity.startswith("sha256:"):
        raise RTPThreeArmRunnerError("evaluation-only cohort lacks source identity")
    opponent_rows = cohort.get("registry_rows")
    if not isinstance(opponent_rows, Sequence) or isinstance(opponent_rows, (str, bytes)):
        raise RTPThreeArmRunnerError("evaluation-only cohort lacks registry rows")
    manifest_opponents = manifest.get("opponents")
    if not isinstance(manifest_opponents, Sequence) or isinstance(manifest_opponents, (str, bytes)):
        raise RTPThreeArmRunnerError("manifest opponents are invalid")
    expected_ids = {str(row.get("id")) for row in manifest_opponents if isinstance(row, Mapping)}
    cohort_ids = {str(row.get("id")) for row in opponent_rows if isinstance(row, Mapping)}
    if len(expected_ids) != 4 or cohort_ids != expected_ids:
        raise RTPThreeArmRunnerError("evaluation cohort must bind the exact four official opponents")
    if any(not isinstance(row, Mapping) or row.get("training_eligible") is not False for row in opponent_rows):
        raise RTPThreeArmRunnerError("evaluation-only cohort has a training-eligible row")

    proof_identity = _require_immutable_identity(
        binding.get("source_exclusion_proof"), "source exclusion proof"
    )
    proof = _json_object(proof_identity["path"], "source exclusion proof")
    required_proof = {
        "schema": SOURCE_EXCLUSION_SCHEMA,
        "status": "verified",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "source_identity_overlap_count": 0,
        "all_registry_rows_training_eligible": False,
        "evaluation_only_cohort_sha256": cohort_identity["sha256"],
        "evaluation_only_cohort_bytes": cohort_identity["bytes"],
        "source_identity_sha256": source_identity,
    }
    for key, expected in required_proof.items():
        if proof.get(key) != expected:
            raise RTPThreeArmRunnerError(f"source exclusion proof fails {key}")
    if proof.get("registry_rows") != list(opponent_rows):
        raise RTPThreeArmRunnerError("source exclusion proof registry rows differ from cohort")
    case_bindings_sha256 = binding.get("evaluation_case_bindings_sha256")
    if not isinstance(case_bindings_sha256, str) or not case_bindings_sha256.startswith("sha256:"):
        raise RTPThreeArmRunnerError("source exclusion binding lacks evaluation-case checksum")
    if proof.get("evaluation_case_bindings_sha256") != case_bindings_sha256:
        raise RTPThreeArmRunnerError("source exclusion proof case checksum differs from binding")
    cases_raw = cohort.get("cases")
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, (str, bytes)):
        raise RTPThreeArmRunnerError("evaluation-only cohort lacks frozen cases")
    cases: dict[str, dict[str, Any]] = {}
    for raw_case in cases_raw:
        if not isinstance(raw_case, Mapping):
            raise RTPThreeArmRunnerError("evaluation-only cohort case is invalid")
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise RTPThreeArmRunnerError("evaluation-only cohort case ID is invalid")
        if raw_case.get("evaluation_only") is not True or raw_case.get("training_eligible") is not False or raw_case.get("replay_eligible") is not False:
            raise RTPThreeArmRunnerError("evaluation-only cohort case has unsafe authority")
        cases[case_id] = dict(raw_case)
    bindings = binding.get("evaluation_case_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise RTPThreeArmRunnerError("source exclusion binding lacks evaluation cases")
    if canonical_digest(list(bindings)) != case_bindings_sha256:
        raise RTPThreeArmRunnerError("source exclusion binding case checksum changed")
    return {
        "cohort": cohort_identity,
        "cohort_payload": cohort,
        "source_exclusion_proof": proof_identity,
        "source_exclusion_payload": proof,
        "evaluation_case_bindings_sha256": case_bindings_sha256,
        "cases": cases,
    }


def _verify_planner_preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("planner_pass_preflight")
    if not isinstance(raw, Mapping):
        raise RTPThreeArmRunnerError("manifest lacks v2 planner_pass_preflight")
    identity = _require_immutable_identity(
        raw.get("receipt"), "planner preflight receipt"
    )
    payload = _json_object(identity["path"], "planner preflight receipt")
    if payload.get("schema") != PLANNER_PREFLIGHT_SCHEMA or payload.get("status") != "passed":
        raise RTPThreeArmRunnerError("planner preflight is not passed")
    direct_key = _direct_arm_key(manifest)
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping):
        raise RTPThreeArmRunnerError("manifest arms are invalid")
    direct = arms.get(direct_key)
    recursive = arms.get("recursive_rtp")
    if not isinstance(direct, Mapping) or not isinstance(recursive, Mapping):
        raise RTPThreeArmRunnerError("manifest bridge arms are invalid")
    direct_profile = _require_identity(
        direct.get("runtime_profile"), "direct runtime profile"
    )
    recursive_profile = _require_identity(
        recursive.get("runtime_profile"), "recursive runtime profile"
    )
    sidecar = _require_identity(direct.get("rtp_sidecar"), "direct RTP sidecar")
    expected = {
        "sidecar_sha256": sidecar["sha256"],
        "direct_runtime_profile_sha256": direct_profile["sha256"],
        "recursive_runtime_profile_sha256": recursive_profile["sha256"],
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "normal_probe_completed": True,
        "normal_probe_observed_neural_passes": 6,
        "forced_replan_probe_completed": True,
        "forced_replan_probe_observed_neural_passes": 5,
        "neural_budget_failures": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RTPThreeArmRunnerError(f"planner preflight fails {key}")
    if raw.get("sidecar_sha256") != sidecar["sha256"]:
        raise RTPThreeArmRunnerError("planner preflight binding sidecar differs")
    return {"identity": identity, "payload": payload}


def _verify_snapshot_package(
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    cohort: Mapping[str, Any],
) -> None:
    """Validate the explicit frozen evaluation case mapping for one cell."""

    case_id = cell.get("evaluation_case_id")
    if not isinstance(case_id, str) or not case_id:
        raise RTPThreeArmRunnerError("scheduled cell lacks evaluation_case_id")
    cases = cohort.get("cases")
    if not isinstance(cases, Mapping) or case_id not in cases:
        raise RTPThreeArmRunnerError("scheduled cell has no frozen evaluation case")
    case = cases[case_id]
    if not isinstance(case, Mapping):
        raise RTPThreeArmRunnerError("scheduled evaluation case is invalid")
    for key in ("opponent_id", "candidate_seat", "replicate"):
        if case.get(key) != cell.get(key):
            raise RTPThreeArmRunnerError(f"scheduled cell differs from case at {key}")
    if cell.get("evaluation_case_content_digest") != case.get("content_digest"):
        raise RTPThreeArmRunnerError("scheduled cell case content digest differs")
    if cell.get("evaluation_case_bindings_sha256") != cohort.get(
        "evaluation_case_bindings_sha256"
    ):
        raise RTPThreeArmRunnerError("scheduled cell case binding checksum differs")
    if cell.get("requested_seed_is_pairing_proof") is not False:
        raise RTPThreeArmRunnerError("requested seed must never be pairing proof")


def _require_physical_readonly_tree(opponent: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _require_immutable_identity(
        opponent.get("artifact"), "opponent package manifest"
    )
    payload = _json_object(artifact["path"], "opponent package manifest")
    if payload.get("schema") != "poke_bot.recursive_turn_planner.evaluation_package_tree_snapshot/v1":
        raise RTPThreeArmRunnerError("opponent package manifest schema is invalid")
    if payload.get("status") != "sealed" or payload.get("no_symlinks") is not True or payload.get("all_paths_read_only") is not True:
        raise RTPThreeArmRunnerError("opponent package snapshot is not sealed read-only")
    root_raw = payload.get("package_root")
    if not isinstance(root_raw, str) or not os.path.isabs(root_raw):
        raise RTPThreeArmRunnerError("opponent package root must be absolute")
    root = _physical_path(root_raw, "opponent package root", require_directory=True)
    if stat.S_IMODE(os.lstat(root).st_mode) & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise RTPThreeArmRunnerError("opponent package root must be read-only")
    entries = payload.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise RTPThreeArmRunnerError("opponent package snapshot lacks entries")
    declared_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RTPThreeArmRunnerError("opponent package entry is invalid")
        relative = entry.get("path")
        relative_path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "." in relative_path.parts
        ):
            raise RTPThreeArmRunnerError("opponent package entry path is invalid")
        source = _physical_path(
            root / relative_path, "opponent package entry", require_file=True
        )
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RTPThreeArmRunnerError(
                "opponent package entry escapes the package root"
            ) from exc
        metadata = os.lstat(source)
        if stat.S_IMODE(metadata.st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RTPThreeArmRunnerError("opponent package entry is writable")
        observed = {
            "path": relative_path.as_posix(),
            "sha256": file_digest(source),
            "bytes": metadata.st_size,
        }
        if dict(entry) != observed:
            raise RTPThreeArmRunnerError("opponent package entry identity changed")
        declared_entries.append(observed)
    if len({entry["path"] for entry in declared_entries}) != len(declared_entries):
        raise RTPThreeArmRunnerError("opponent package snapshot repeats an entry")

    # Do not merely verify the listed files: a mutable, unlisted module could
    # otherwise be imported by ``main.py``.  Inventory the full physical tree
    # and require exact equality with the sealed manifest.
    observed_entries: list[dict[str, Any]] = []
    for current_raw, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = _physical_path(current_raw, "opponent package directory", require_directory=True)
        current_mode = os.lstat(current).st_mode
        if stat.S_IMODE(current_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RTPThreeArmRunnerError("opponent package directory is writable")
        directories.sort()
        files.sort()
        for directory in directories:
            child = current / directory
            metadata = os.lstat(child)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RTPThreeArmRunnerError("opponent package contains a nonphysical directory")
        for name in files:
            source = _physical_path(current / name, "opponent package file", require_file=True)
            metadata = os.lstat(source)
            if stat.S_IMODE(metadata.st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise RTPThreeArmRunnerError("opponent package contains a writable file")
            observed_entries.append(
                {
                    "path": source.relative_to(root).as_posix(),
                    "sha256": file_digest(source),
                    "bytes": metadata.st_size,
                }
            )
    if sorted(observed_entries, key=lambda row: row["path"]) != sorted(
        declared_entries, key=lambda row: row["path"]
    ):
        raise RTPThreeArmRunnerError("opponent package tree differs from its sealed manifest")
    if canonical_digest(sorted(observed_entries, key=lambda row: row["path"])) != payload.get("tree_entries_sha256"):
        raise RTPThreeArmRunnerError("opponent package tree digest changed")
    content_digest = opponent.get("content_digest")
    if not isinstance(content_digest, str) or payload.get("content_digest") != content_digest:
        raise RTPThreeArmRunnerError("opponent package content digest is not bound")
    if payload.get("opponent_id") != opponent.get("id"):
        raise RTPThreeArmRunnerError("opponent package ID mismatch")
    return {
        "manifest": artifact,
        "payload": payload,
        "package_root": str(root),
        "content_digest": content_digest,
    }


def _sealed_opponents(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("opponents")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPThreeArmRunnerError("manifest opponents must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise RTPThreeArmRunnerError("manifest opponent is invalid")
        opponent_id = str(entry["id"])
        if opponent_id in result:
            raise RTPThreeArmRunnerError("manifest repeats an opponent")
        result[opponent_id] = _require_physical_readonly_tree(entry)
    return result


@dataclass(frozen=True)
class ArmRuntime:
    """Fresh objects returned by a factory inside one arm child process.

    ``candidate`` and ``opponent`` must be fresh callables.  ``runtime_identity``
    is the frozen candidate runtime identity plus optional supplemental audit
    fields.  ``isolation`` supplies factory-specific proof such as its baseline
    module name and frozen candidate model identity.
    """

    candidate: Callable[[dict[str, Any]], list[int]]
    opponent: Callable[[dict[str, Any]], list[int]]
    runtime_identity: Mapping[str, Any]
    isolation: Mapping[str, Any]
    #: A side-effect-free, arm-independent complexity-gate probe.  It runs
    #: before each candidate action so the no-RTP arm cannot pretend every
    #: decision was simple merely because its bridge was disabled.
    complexity_intent: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    #: Factory-owned digest of the candidate state which complexity probing is
    #: forbidden to mutate.  It covers the state the factory's implementation
    #: considers causal (history/cache/bridge/executor/router), not model
    #: parameters or transient diagnostic output.
    complexity_probe_state_digest: Callable[[], str] | None = None
    #: A stable fingerprint of the candidate's *logical policy input* before
    #: an action.  It intentionally excludes bridge/executor implementation
    #: state so the compiler can compare actions only when A/B/C actually saw
    #: the same observation/history/cache/router input, rather than assuming
    #: cells remain lockstep after their distinct bridge paths diverge.
    candidate_policy_input_fingerprint: Callable[[Mapping[str, Any]], str] | None = None


class RtpThreeArmEvaluationFactory(Protocol):
    """Factory contract implemented by the sealed production evaluation tree.

    Every method runs in a fresh worker interpreter and must construct a new
    candidate model/agent and a new baseline module from the sealed package.
    The parent never invokes a factory to create a battle: it passes the exact
    already sealed ``rng_materials`` snapshot bytes to all three children.
    """

    def worker_environment(
        self,
        *,
        manifest: Mapping[str, Any],
        cell: Mapping[str, Any],
        arm: str,
        scratch_dir: str,
    ) -> Mapping[str, str]: ...

    def create_arm_engine(
        self, *, manifest: Mapping[str, Any], cell: Mapping[str, Any], arm: str
    ) -> Any: ...

    def create_arm_runtime(
        self, *, manifest: Mapping[str, Any], cell: Mapping[str, Any], arm: str
    ) -> ArmRuntime | Mapping[str, Any]: ...


def _load_factory(reference: str) -> Any:
    module_name, separator, attribute = str(reference).partition(":")
    if not separator or not module_name or not attribute:
        raise RTPThreeArmRunnerError("factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute, None)
    if value is None:
        raise RTPThreeArmRunnerError(f"factory attribute not found: {reference}")
    return value() if isinstance(value, type) else value


def _as_arm_runtime(value: Any) -> ArmRuntime:
    if isinstance(value, ArmRuntime):
        return value
    if not isinstance(value, Mapping):
        raise RTPThreeArmRunnerError("factory arm runtime must be ArmRuntime or mapping")
    candidate = value.get("candidate")
    opponent = value.get("opponent")
    runtime_identity = value.get("runtime_identity")
    isolation = value.get("isolation")
    if not callable(candidate) or not callable(opponent):
        raise RTPThreeArmRunnerError("factory arm runtime needs candidate/opponent callables")
    if not isinstance(runtime_identity, Mapping) or not isinstance(isolation, Mapping):
        raise RTPThreeArmRunnerError("factory arm runtime lacks identity/isolation")
    complexity_intent = value.get("complexity_intent")
    if complexity_intent is not None and not callable(complexity_intent):
        raise RTPThreeArmRunnerError("factory complexity_intent must be callable")
    complexity_probe_state_digest = value.get("complexity_probe_state_digest")
    if complexity_probe_state_digest is not None and not callable(
        complexity_probe_state_digest
    ):
        raise RTPThreeArmRunnerError(
            "factory complexity_probe_state_digest must be callable"
        )
    candidate_policy_input_fingerprint = value.get("candidate_policy_input_fingerprint")
    if candidate_policy_input_fingerprint is not None and not callable(
        candidate_policy_input_fingerprint
    ):
        raise RTPThreeArmRunnerError(
            "factory candidate_policy_input_fingerprint must be callable"
        )
    return ArmRuntime(
        candidate=candidate,
        opponent=opponent,
        runtime_identity=dict(runtime_identity),
        isolation=dict(isolation),
        complexity_intent=complexity_intent,
        complexity_probe_state_digest=complexity_probe_state_digest,
        candidate_policy_input_fingerprint=candidate_policy_input_fingerprint,
    )


def _sealed_snapshot_materials(
    manifest: Mapping[str, Any], capability: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Read exact pre-sealed snapshot *seals* addressed by the manifest.

    A requested start seed is debugging metadata only.  Main evaluation never
    captures, reconstructs, or transports raw native snapshot bytes.  Each
    ``rng_materials[*]`` carries two immutable identities: the opaque native
    ``snapshot_artifact`` and its cell-bound ``seal`` JSON.  Every fresh A/B/C
    child restores the same seal through the engine's sealed-manifest endpoint.
    """

    raw = manifest.get("rng_materials")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPThreeArmRunnerError("manifest rng_materials is invalid")
    materials: dict[str, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise RTPThreeArmRunnerError("manifest RNG material is invalid")
        material_id = entry.get("id")
        if not isinstance(material_id, str) or not material_id:
            raise RTPThreeArmRunnerError("manifest RNG material lacks ID")
        if entry.get("kind") != "snapshot":
            raise RTPThreeArmRunnerError("r197 evaluation requires sealed snapshot material")
        snapshot_identity = _require_immutable_identity(
            entry.get("snapshot_artifact"), "sealed RNG snapshot artifact"
        )
        seal_identity = _require_immutable_identity(
            entry.get("seal"), "sealed RNG snapshot seal"
        )
        if seal_identity["mode"] != 0o444:
            raise RTPThreeArmRunnerError(
                "sealed RNG snapshot seal must use immutable mode 0444"
            )
        seal = _json_object(seal_identity["path"], "sealed RNG snapshot seal")
        if seal.get("schema") != SNAPSHOT_SEAL_SCHEMA or seal.get("status") != "sealed":
            raise RTPThreeArmRunnerError("RNG material is not a sealed snapshot manifest")
        for key, identity in (
            ("engine_artifact_sha256", capability.get("engine_artifact")),
            ("source_artifact_sha256", capability.get("source_artifact")),
            ("patch_artifact_sha256", capability.get("patch_artifact")),
            ("build_artifact_sha256", capability.get("build_artifact")),
        ):
            if not isinstance(identity, Mapping) or seal.get(key) != identity.get("sha256"):
                raise RTPThreeArmRunnerError(f"sealed RNG snapshot does not bind {key}")
        if seal.get("canonical_abi_sha256") != snapshot_abi_sha256():
            raise RTPThreeArmRunnerError("sealed RNG snapshot ABI differs from capability")
        if seal.get("capture_boundary") != snapshot_abi_contract()["capture_boundary"]:
            raise RTPThreeArmRunnerError("sealed RNG snapshot capture boundary is invalid")
        if seal.get("boundary_tag") != snapshot_abi_contract()["boundary_tag"]:
            raise RTPThreeArmRunnerError("sealed RNG snapshot boundary tag is invalid")
        if snapshot_identity["mode"] != 0o444:
            raise RTPThreeArmRunnerError("sealed RNG snapshot artifact must use immutable mode 0444")
        if seal.get("snapshot_artifact_sha256") != snapshot_identity["sha256"] or seal.get(
            "snapshot_artifact_bytes"
        ) != snapshot_identity["bytes"]:
            raise RTPThreeArmRunnerError("sealed RNG snapshot seal does not bind its artifact")
        requested_seed = entry.get("requested_seed_audit_only")
        if isinstance(requested_seed, bool) or not isinstance(requested_seed, int):
            raise RTPThreeArmRunnerError(
                "sealed RNG material lacks requested_seed_audit_only"
            )
        if seal.get("requested_seed_audit_only") != requested_seed:
            raise RTPThreeArmRunnerError(
                "sealed RNG material audit seed differs from its seal"
            )
        if seal.get("requested_seed_is_pairing_proof") is not False:
            raise RTPThreeArmRunnerError(
                "sealed RNG material incorrectly treats an audit seed as pairing proof"
            )
        if material_id in materials:
            raise RTPThreeArmRunnerError("manifest repeats a sealed RNG snapshot ID")
        materials[material_id] = {
            "id": material_id,
            "kind": "snapshot",
            # Schedule RNG identity binds opaque snapshot bytes and separately
            # records the seal digest/capture boundary that authorized restore.
            "sha256": snapshot_identity["sha256"],
            "bytes": snapshot_identity["bytes"],
            "seal_sha256": seal_identity["sha256"],
            "capture_boundary": snapshot_abi_contract()["capture_boundary"],
            "boundary_tag": snapshot_abi_contract()["boundary_tag"],
            "seal": seal_identity,
            "snapshot_artifact": snapshot_identity,
            "requested_seed_audit_only": requested_seed,
        }
    return materials


def _opponent_deck_digests(opponent: Mapping[str, Any]) -> tuple[str, str]:
    payload = opponent.get("payload")
    if not isinstance(payload, Mapping):
        raise RTPThreeArmRunnerError("sealed opponent lacks package manifest payload")
    deck_sha256 = payload.get("deck_sha256")
    deck_order_sha256 = payload.get("deck_order_sha256")
    if not isinstance(deck_sha256, str) or not isinstance(deck_order_sha256, str):
        raise RTPThreeArmRunnerError("sealed opponent package lacks exact deck identities")
    return deck_sha256, deck_order_sha256


def _opponent_package_entry_digest(opponent: Mapping[str, Any], path: str) -> str:
    payload = opponent.get("payload")
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise RTPThreeArmRunnerError("sealed opponent package lacks entries")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("path") == path
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise RTPThreeArmRunnerError(f"sealed opponent package lacks {path} identity")
    return str(matches[0]["sha256"])


def _verify_snapshot_for_cell(
    *,
    snapshot: Mapping[str, Any],
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
    opponent: Mapping[str, Any],
    cohort: Mapping[str, Any],
) -> None:
    """Verify a seal binds this exact scheduled deck/seat/case before restore."""

    seal_identity = snapshot.get("seal")
    nested_snapshot = snapshot.get("snapshot_artifact")
    if not isinstance(seal_identity, Mapping) or not isinstance(nested_snapshot, Mapping):
        raise RTPThreeArmRunnerError("sealed snapshot material lacks sealed identities")
    seal = _json_object(str(seal_identity.get("path", "")), "sealed RNG snapshot seal")
    shared = manifest.get("shared_artifacts")
    if not isinstance(shared, Mapping):
        raise RTPThreeArmRunnerError("manifest shared artifacts are invalid")
    candidate_deck = _require_identity(shared.get("deck"), "candidate deck")
    opponent_deck, opponent_deck_order = _opponent_deck_digests(opponent)
    candidate_seat = cell.get("candidate_seat")
    if candidate_seat not in {0, 1}:
        raise RTPThreeArmRunnerError("scheduled snapshot cell has invalid candidate seat")
    expected = {
        "snapshot_id": snapshot.get("id"),
        "evaluation_case_id": cell.get("evaluation_case_id"),
        "evaluation_only_cohort_sha256": _require_identity(
            shared.get("evaluation_only_cohort"), "evaluation-only cohort"
        )["sha256"],
        "opponent_id": cell.get("opponent_id"),
        "opponent_content_digest": opponent.get("content_digest"),
        "candidate_seat": candidate_seat,
        "replicate": cell.get("replicate"),
        "candidate_deck_sha256": candidate_deck["sha256"],
        "candidate_deck_order_sha256": candidate_deck["sha256"],
        "opponent_deck_sha256": opponent_deck,
        "opponent_deck_order_sha256": opponent_deck_order,
        "snapshot_artifact_sha256": nested_snapshot.get("sha256"),
        "snapshot_artifact_bytes": nested_snapshot.get("bytes"),
        "capture_boundary": snapshot_abi_contract()["capture_boundary"],
        "boundary_tag": snapshot_abi_contract()["boundary_tag"],
        "requested_seed_is_pairing_proof": False,
    }
    for key, expected_value in expected.items():
        if seal.get(key) != expected_value:
            raise RTPThreeArmRunnerError(
                f"sealed snapshot does not bind scheduled cell at {key}"
            )
    _verify_snapshot_case_binding(
        seal=seal,
        snapshot=snapshot,
        cell=cell,
        candidate_deck=candidate_deck,
        opponent=opponent,
        opponent_deck_sha256=opponent_deck,
        cohort=cohort,
    )


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare a file identity without accepting a mutable mode alias."""

    return all(left.get(key) == right.get(key) for key in ("path", "sha256", "bytes"))


def _identity_triplet(identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Return the portable immutable identity shape used in JSON contracts."""

    observed = _require_identity(identity, label)
    return {key: observed[key] for key in ("path", "sha256", "bytes")}


def _sealed_case_identity(raw: Any, label: str) -> dict[str, Any]:
    identity = _require_immutable_identity(raw, label)
    if identity["mode"] != 0o444:
        raise RTPThreeArmRunnerError(f"{label} must use immutable mode 0444")
    return identity


def _verify_snapshot_case_binding(
    *,
    seal: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    cell: Mapping[str, Any],
    candidate_deck: Mapping[str, Any],
    opponent: Mapping[str, Any],
    opponent_deck_sha256: str,
    cohort: Mapping[str, Any],
) -> None:
    """Bind a snapshot seal to the concrete scheduled evaluation cell.

    The native wrapper verifies that a case-binding record is syntactically
    sealed before restore.  The evaluator additionally owns the semantic
    comparison to its schedule, frozen cohort, deck identities, and source
    exclusion proof.  Keeping that comparison here prevents a valid snapshot
    from being transplanted into another cell merely because its engine ABI is
    compatible.
    """

    requested_seed = seal.get("requested_seed_audit_only")
    if (
        isinstance(requested_seed, bool)
        or not isinstance(requested_seed, int)
        or not 0 <= requested_seed <= 0xFFFFFFFF
    ):
        raise RTPThreeArmRunnerError("snapshot seal lacks a valid audit-only seed")
    if snapshot.get("requested_seed_audit_only") != requested_seed:
        raise RTPThreeArmRunnerError("snapshot material and seal audit seed differ")
    case_identity = _sealed_case_identity(
        seal.get("case_binding_artifact"), "snapshot case binding"
    )
    if seal.get("case_binding_artifact_sha256") != case_identity["sha256"]:
        raise RTPThreeArmRunnerError("snapshot seal case-binding checksum differs")
    binding = _json_object(case_identity["path"], "snapshot case binding")
    if (
        binding.get("schema") != PAIRING_CASE_BINDING_SCHEMA
        or binding.get("status") != "sealed"
    ):
        raise RTPThreeArmRunnerError("snapshot case binding schema/status is invalid")
    expected = {
        "cell_id": cell.get("cell_id"),
        "case_id": cell.get("evaluation_case_id"),
        "opponent_id": cell.get("opponent_id"),
        "opponent_content_digest": opponent.get("content_digest"),
        "seat": cell.get("candidate_seat"),
        "candidate_seat": cell.get("candidate_seat"),
        "replicate": cell.get("replicate"),
        "debug_seed": requested_seed,
        "evaluation_case_bindings_sha256": cell.get(
            "evaluation_case_bindings_sha256"
        ),
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "promotion_eligible": False,
        "kaggle_submission_authorized": False,
    }
    for key, expected_value in expected.items():
        if binding.get(key) != expected_value:
            raise RTPThreeArmRunnerError(
                f"snapshot case binding differs from its scheduled cell at {key}"
            )
    candidate_identity = binding.get("candidate_deck_identity")
    opponent_identity = binding.get("opponent_deck_identity")
    if not isinstance(candidate_identity, Mapping) or not isinstance(opponent_identity, Mapping):
        raise RTPThreeArmRunnerError("snapshot case binding lacks deck identities")
    if candidate_identity.get("sha256") != candidate_deck.get("sha256"):
        raise RTPThreeArmRunnerError("snapshot case binding has another candidate deck")
    if opponent_identity.get("sha256") != opponent_deck_sha256:
        raise RTPThreeArmRunnerError("snapshot case binding has another opponent deck")
    ordered = binding.get("ordered_deck_identities")
    if not isinstance(ordered, Sequence) or isinstance(ordered, (str, bytes)) or len(ordered) != 2:
        raise RTPThreeArmRunnerError("snapshot case binding lacks two ordered deck identities")
    if not all(isinstance(identity, Mapping) for identity in ordered):
        raise RTPThreeArmRunnerError("snapshot case binding ordered deck identity is invalid")
    candidate_seat = int(cell["candidate_seat"])
    expected_ordered_sha256 = (
        (candidate_deck["sha256"], opponent_deck_sha256)
        if candidate_seat == 0
        else (opponent_deck_sha256, candidate_deck["sha256"])
    )
    if tuple(identity.get("sha256") for identity in ordered) != expected_ordered_sha256:
        raise RTPThreeArmRunnerError("snapshot case binding ordered decks differ from seat mapping")
    cohort_identity = cohort.get("cohort")
    source_proof_identity = cohort.get("source_exclusion_proof")
    if not isinstance(cohort_identity, Mapping) or not isinstance(source_proof_identity, Mapping):
        raise RTPThreeArmRunnerError("source-exclusion verification omitted immutable identities")
    binding_cohort = binding.get("cohort_identity")
    binding_proof = binding.get("source_exclusion_identity")
    if not isinstance(binding_cohort, Mapping) or not _identity_equal(binding_cohort, cohort_identity):
        raise RTPThreeArmRunnerError("snapshot case binding differs from frozen cohort")
    if not isinstance(binding_proof, Mapping) or not _identity_equal(binding_proof, source_proof_identity):
        raise RTPThreeArmRunnerError("snapshot case binding differs from source-exclusion proof")
    if seal.get("source_exclusion_proof_sha256") != source_proof_identity.get("sha256"):
        raise RTPThreeArmRunnerError("snapshot seal differs from source-exclusion proof")


def _seed_for_cell(manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> int:
    material = {
        "manifest_input_sha256": manifest.get("manifest_input_sha256"),
        "cell_id": cell.get("cell_id"),
        "domain": "rtp-three-arm-policy-rng-v1",
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _state_digest(value: Any) -> str:
    return _sha256_bytes(pickle.dumps(value, protocol=5))


def _rng_state_snapshot(candidate: Any, opponent: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_rng_initial_state_sha256": _state_digest(random.getstate()),
        "candidate_rng_initial_state_sha256": None,
        "opponent_rng_initial_state_sha256": None,
        "opponent_rng_deterministic_or_no_rng": False,
        "numpy_rng_initial_state_sha256": "unavailable",
        "torch_cpu_rng_initial_state_sha256": "unavailable",
        "torch_cuda_rng_initial_state_sha256": "unavailable",
    }
    candidate_rng = getattr(candidate, "rng", None)
    if isinstance(candidate_rng, random.Random):
        result["candidate_rng_initial_state_sha256"] = _state_digest(candidate_rng.getstate())
    else:
        # The factory may return an agent without an exposed PRNG only when it
        # has a deterministic/no-RNG implementation.  It must attest that fact.
        result["candidate_rng_initial_state_sha256"] = "deterministic_or_no_rng"
    opponent_rng = getattr(opponent, "rng", None)
    if isinstance(opponent_rng, random.Random):
        result["opponent_rng_initial_state_sha256"] = _state_digest(opponent_rng.getstate())
    else:
        result["opponent_rng_deterministic_or_no_rng"] = True
    try:
        import numpy as np

        result["numpy_rng_initial_state_sha256"] = _state_digest(np.random.get_state())
    except Exception:
        pass
    terminal_finished = False
    terminal_winner = 2
    try:
        import torch

        result["torch_cpu_rng_initial_state_sha256"] = _sha256_bytes(
            bytes(torch.get_rng_state().tolist())
        )
        if torch.cuda.is_available():
            state = b"".join(bytes(value.tolist()) for value in torch.cuda.get_rng_state_all())
            result["torch_cuda_rng_initial_state_sha256"] = _sha256_bytes(state)
    except Exception:
        pass
    return result


def _seed_fresh_process(seed: int, candidate: Any, opponent: Any) -> None:
    random.seed(seed)
    for actor in (candidate, opponent):
        rng = getattr(actor, "rng", None)
        if isinstance(rng, random.Random):
            rng.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_grad_enabled(False)
    except Exception:
        pass


def _diagnostic_snapshot(candidate: Any) -> dict[str, Any] | None:
    getter = getattr(candidate, "rtp_diagnostic_snapshot", None)
    if callable(getter):
        value = getter()
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise RTPThreeArmRunnerError("candidate RTP diagnostic snapshot is invalid")
        return dict(value)
    value = getattr(candidate, "last_rtp_diagnostics", None)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RTPThreeArmRunnerError("candidate RTP diagnostics are invalid")
    return dict(value)


_SUCCESSFUL_RECURSIVE_MODES = frozenset(
    {"recursive_plan", "continue_plan", "replan_with_program"}
)
_RECURSIVE_FALLBACK_MODES = frozenset({"direct_policy_fallback", "replan_direct"})
_RECURSIVE_MODE_COUNT_KEYS = tuple(
    sorted(_SUCCESSFUL_RECURSIVE_MODES | _RECURSIVE_FALLBACK_MODES)
)


def _complexity_intent_for_decision(
    runtime: ArmRuntime, observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Obtain the common pre-forcing gate trace without changing behavior."""

    probe = runtime.complexity_intent
    if not callable(probe):
        raise RTPThreeArmRunnerError(
            "arm runtime lacks the required common complexity_intent probe"
        )
    value = probe(dict(observation))
    if not isinstance(value, Mapping):
        raise RTPThreeArmRunnerError("complexity_intent probe must return an object")
    intended = value.get("intended_complex")
    reason = value.get("planner_reason")
    if not isinstance(intended, bool) or not isinstance(reason, str) or not reason.strip():
        raise RTPThreeArmRunnerError(
            "complexity_intent requires boolean intended_complex and planner_reason"
        )
    return {"intended_complex": intended, "planner_reason": reason.strip()}


def _require_sha256_text(value: Any, label: str) -> str:
    """Require a canonical SHA-256 string at a trace boundary."""

    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise RTPThreeArmRunnerError(f"{label} must be a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value[7:]):
        raise RTPThreeArmRunnerError(f"{label} must be a SHA-256 digest")
    return value


def _over_cap_factorized_selection_context(
    runtime: ArmRuntime,
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Classify a non-materializable selection before the common probe.

    The common complexity probe intentionally uses complete action encoding.
    It therefore cannot be called first for a decision whose exact ordered
    cardinality is above 1,024.  This helper computes only the closed-form
    permutation sum, then captures the logical pre-action fingerprint that
    later limits cross-arm parity checks to genuinely equal policy inputs.
    """

    from . import features

    action_space = features.complete_ordered_action_space_summary(
        dict(observation), max_combos=R198_COMPLETE_ORDERED_ACTION_CAP
    )
    if action_space.get("over_cap") is not True:
        return None
    fingerprint = runtime.candidate_policy_input_fingerprint
    if not callable(fingerprint):
        raise RTPThreeArmRunnerError(
            "over-cap selection lacks a candidate policy-input fingerprint"
        )
    observation_sha256 = canonical_digest(dict(observation))
    action_space_sha256 = canonical_digest(action_space)
    candidate_policy_input_sha256 = _require_sha256_text(
        fingerprint(dict(observation)), "candidate policy-input fingerprint"
    )
    return {
        "action_space": action_space,
        "action_space_sha256": action_space_sha256,
        "observation_sha256": observation_sha256,
        "candidate_policy_input_sha256": candidate_policy_input_sha256,
        "logical_pre_action_sha256": canonical_digest(
            {
                "observation_sha256": observation_sha256,
                "action_space_sha256": action_space_sha256,
                "candidate_policy_input_sha256": candidate_policy_input_sha256,
            }
        ),
    }


def _timed_candidate_selection(
    runtime: ArmRuntime,
    candidate: Callable[[dict[str, Any]], list[int]],
    observation: Mapping[str, Any],
) -> tuple[Any, float, dict[str, Any]]:
    """Probe first, assert it is non-mutating, then time selection only.

    The common probe is intentionally excluded from a decision's end-to-end
    latency.  It is shared instrumentation used to define the complexity
    denominator, while the measured value is the candidate's actual action
    selection.  A factory must expose a state digest so a probe cannot quietly
    advance bridge history/cache/RNG and bias the next selected action.
    """

    state_digest = runtime.complexity_probe_state_digest
    if not callable(state_digest):
        raise RTPThreeArmRunnerError(
            "arm runtime lacks required complexity_probe_state_digest"
        )
    before = state_digest()
    if not isinstance(before, str) or not before:
        raise RTPThreeArmRunnerError("complexity probe state digest is invalid")
    intent = _complexity_intent_for_decision(runtime, observation)
    after = state_digest()
    if not isinstance(after, str) or not after:
        raise RTPThreeArmRunnerError("complexity probe state digest is invalid")
    if after != before:
        raise RTPThreeArmRunnerError("complexity intent probe mutated candidate state")
    started = time.perf_counter()
    action = candidate(dict(observation))
    latency = time.perf_counter() - started
    return action, latency, intent


def _forced_turn_order_control(
    observation: Mapping[str, Any],
) -> tuple[list[int], int | str, str] | None:
    """Return the exact external turn-order control, when one is present.

    Choosing ``Yes`` for an explicit ``IsFirst`` prompt is a fixed competition
    control, not a policy or planner decision.  Delegate recognition to the
    same helper the policy uses, then retain a small canonical representation
    of the observed prompt for the immutable evaluator trace.  The live r198
    snapshots use numeric context ``41``; named enum encodings remain accepted
    because the policy helper explicitly supports them too.
    """

    from .features import forced_go_first_action

    try:
        expected = forced_go_first_action(dict(observation))
    except Exception as exc:
        raise RTPThreeArmRunnerError(
            "forced turn-order control is malformed"
        ) from exc
    if expected is None:
        return None
    if (
        not isinstance(expected, Sequence)
        or isinstance(expected, (str, bytes))
        or len(expected) != 1
        or type(expected[0]) is not int
        or expected[0] < 0
    ):
        raise RTPThreeArmRunnerError("forced turn-order action is invalid")
    select = observation.get("select")
    if not isinstance(select, Mapping):  # defensive; helper already checked it
        raise RTPThreeArmRunnerError("forced turn-order prompt is invalid")
    raw_context = select.get("context")
    if type(raw_context) is int and raw_context == 41:
        return [expected[0]], 41, "numeric_41"
    enum_value = getattr(raw_context, "name", raw_context)
    normalized = "".join(character for character in str(enum_value).lower() if character.isalnum())
    if normalized == "isfirst":
        return [expected[0]], "IsFirst", "enum_is_first"
    raise RTPThreeArmRunnerError(
        "forced turn-order helper accepted an unrecognized prompt context"
    )


def _select_candidate_action(
    runtime: ArmRuntime,
    candidate: Callable[[dict[str, Any]], list[int]],
    observation: Mapping[str, Any],
) -> tuple[
    Any,
    float | None,
    dict[str, Any] | None,
    tuple[list[int], int | str, str] | None,
    dict[str, Any] | None,
]:
    """Select one action while excluding external turn-order controls.

    The exact forced control deliberately bypasses the common complexity probe
    and measured selection timer.  Every other action uses the existing probe
    and returns a normal complexity trace, so a missing B/C diagnostic remains
    a hard failure in :meth:`DecisionTelemetry.observe`.
    """

    forced_control = _forced_turn_order_control(observation)
    if forced_control is not None:
        return candidate(dict(observation)), None, None, forced_control, None
    over_cap = _over_cap_factorized_selection_context(runtime, observation)
    if over_cap is not None:
        # Selection itself is measured exactly like every other candidate
        # action.  Only the common probe and planner-derived denominators are
        # excluded, because no complete action set was materialized.
        started = time.perf_counter()
        action = candidate(dict(observation))
        latency = time.perf_counter() - started
        return action, latency, None, None, over_cap
    action, latency, intent = _timed_candidate_selection(runtime, candidate, observation)
    return action, latency, intent, None, None


def _record_forced_turn_order_control(
    telemetry: "DecisionTelemetry",
    candidate: Any,
    *,
    expected_action: Sequence[int],
    returned_action: Sequence[int],
    prompt_context: int | str,
    prompt_context_encoding: str,
) -> None:
    """Verify and record a bridge-free external turn-order action."""

    if list(returned_action) != list(expected_action):
        raise RTPThreeArmRunnerError("forced_turn_order_action_mismatch")
    if _diagnostic_snapshot(candidate) is not None:
        raise RTPThreeArmRunnerError("forced_turn_order_emitted_rtp_diagnostics")
    telemetry.observe_forced_turn_order_control(
        expected_action=expected_action,
        returned_action=returned_action,
        prompt_context=prompt_context,
        prompt_context_encoding=prompt_context_encoding,
    )


def _planner_mode_from_diagnostic(
    diagnostic: Mapping[str, Any], *, arm: str
) -> str:
    """Normalize bridge diagnostics to the evaluator's exact mode vocabulary."""

    raw_mode = str(diagnostic.get("mode", ""))
    decision_mode = str(diagnostic.get("decision_mode", ""))
    if raw_mode == "replan":
        return "replan_with_program" if decision_mode in {"recursive_plan", ""} else "replan_direct"
    if raw_mode == "fallback":
        # A generic bridge failure is only scoreable as a recursive fallback
        # when it occurred after a complexity-intended recursive attempt.  The
        # caller enforces that condition below; preserve replan-direct when
        # the bridge has already recorded it.
        if decision_mode in _RECURSIVE_FALLBACK_MODES:
            return decision_mode
        return "direct_policy_fallback"
    if raw_mode in _SUCCESSFUL_RECURSIVE_MODES | _RECURSIVE_FALLBACK_MODES | {
        "direct_bridge",
        "direct_policy",
    }:
        return raw_mode
    if arm == CANONICAL_DIRECT_ARM and raw_mode in {"", "direct"}:
        return "direct_bridge"
    raise RTPThreeArmRunnerError(
        f"RTP diagnostic has no canonical planner mode: {raw_mode!r}"
    )


def _fallback_classification(
    diagnostic: Mapping[str, Any], planner_mode: str
) -> str:
    """Classify only planner-expected direct fallbacks as expected."""

    decision_details = diagnostic.get("decision_diagnostics")
    details = dict(decision_details) if isinstance(decision_details, Mapping) else {}
    code = str(diagnostic.get("fallback_code", ""))
    reason = str(diagnostic.get("fallback_reason", ""))
    expected_reasons = {"all_plans_illegal", "no_legal_actions", "executor_no_program"}
    if planner_mode == "direct_policy_fallback" and (
        details.get("reason") in expected_reasons
        or code in expected_reasons
        or reason in expected_reasons
    ):
        return "expected"
    if planner_mode == "replan_direct" and (
        details.get("reason") in expected_reasons
        or code in expected_reasons
        or reason in expected_reasons
    ):
        return "expected"
    return "unexpected"


_OVER_CAP_ACTION_SPACE_FIELDS = frozenset(
    {
        "n_options",
        "min_count",
        "max_count",
        "counts",
        "complete_ordered_action_cardinality",
        "complete_ordered_action_cap",
        "over_cap",
        "complete_ordered_actions_materialized",
        "complete_ordered_action_truncated",
    }
)


def _strict_over_cap_action_space(raw: Any) -> dict[str, Any]:
    """Validate the non-materializing r198 selection summary locally.

    The compiler and promotion validator repeat these checks independently.
    Keeping them at the live-worker boundary prevents a bad bridge diagnostic
    from being written as an otherwise well-formed evaluation row.
    """

    if not isinstance(raw, Mapping) or set(raw) != _OVER_CAP_ACTION_SPACE_FIELDS:
        raise RTPThreeArmRunnerError("over-cap action-space summary has an invalid field set")
    value = dict(raw)
    for name in (
        "n_options",
        "min_count",
        "max_count",
        "complete_ordered_action_cardinality",
        "complete_ordered_action_cap",
    ):
        if type(value.get(name)) is not int:
            raise RTPThreeArmRunnerError(
                f"over-cap action-space {name} must be an exact integer"
            )
    n_options = value["n_options"]
    min_count = value["min_count"]
    max_count = value["max_count"]
    cap = value["complete_ordered_action_cap"]
    if cap != R198_COMPLETE_ORDERED_ACTION_CAP:
        raise RTPThreeArmRunnerError("over-cap action-space cap differs from r198 1024")
    if not (0 <= min_count <= max_count <= n_options):
        raise RTPThreeArmRunnerError("over-cap action-space bounds are invalid")
    counts = value.get("counts")
    expected_counts = list(range(min_count, max_count + 1))
    if (
        not isinstance(counts, Sequence)
        or isinstance(counts, (str, bytes))
        or any(type(item) is not int for item in counts)
        or list(counts) != expected_counts
    ):
        raise RTPThreeArmRunnerError("over-cap action-space counts are not exact")
    expected_cardinality = sum(math.perm(n_options, count) for count in expected_counts)
    if value["complete_ordered_action_cardinality"] != expected_cardinality:
        raise RTPThreeArmRunnerError("over-cap action-space cardinality does not recompute")
    if value.get("over_cap") is not True or expected_cardinality <= cap:
        raise RTPThreeArmRunnerError("over-cap action-space does not exceed the cap")
    if value.get("complete_ordered_actions_materialized") is not False:
        raise RTPThreeArmRunnerError("over-cap action-space materialized complete actions")
    if value.get("complete_ordered_action_truncated") is not False:
        raise RTPThreeArmRunnerError("over-cap action-space attests truncation")
    return value


def _factorized_stage_count_for_over_cap_action(
    action_space: Mapping[str, Any], action: Sequence[Any]
) -> int:
    """Validate selected factorized legality without complete enumeration."""

    from . import features

    if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
        raise RTPThreeArmRunnerError("over-cap returned action is not a sequence")
    selected = list(action)
    if any(type(item) is not int for item in selected):
        raise RTPThreeArmRunnerError("over-cap returned action has a non-exact index")
    observation = {
        "select": {
            "option": [{} for _ in range(int(action_space["n_options"]))],
            "minCount": int(action_space["min_count"]),
            "maxCount": int(action_space["max_count"]),
        }
    }
    try:
        stages = features.factorized_teacher_forcing_stages(observation, selected)
    except ValueError as exc:
        raise RTPThreeArmRunnerError(
            "over-cap factorized teacher-forcing legality failed"
        ) from exc
    return len(stages)


def _over_cap_bridge_diagnostic(
    diagnostic: Mapping[str, Any] | None,
    *,
    arm: str,
    action_space: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require the pre-existing B/C factorized fallback, with no planner credit."""

    if arm == "no_rtp":
        if diagnostic is not None:
            raise RTPThreeArmRunnerError("over-cap no-RTP decision emitted RTP diagnostics")
        return None
    if not isinstance(diagnostic, Mapping):
        raise RTPThreeArmRunnerError("over-cap RTP decision omitted bridge diagnostics")
    extras_raw = diagnostic.get("extras")
    extras = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}
    # The special path is reached before a common complexity probe, direct
    # bridge branch, executor continuation, or repair.  Reject their raw
    # diagnostic footprints rather than letting a future control-flow reorder
    # reset counters and relabel hidden planner work as an over-cap audit row.
    planner_work_extras = {
        "pre_forcing_complexity_intent",
        "force_direct_bridge_only",
        "repaired",
    }
    if planner_work_extras.intersection(extras):
        raise RTPThreeArmRunnerError(
            "over-cap bridge diagnostic attests planner work before special fallback"
        )
    raw_decision = diagnostic.get("decision_diagnostics")
    if raw_decision is not None and raw_decision != {}:
        raise RTPThreeArmRunnerError(
            "over-cap bridge diagnostic carries a planner decision payload"
        )
    bridge_trace = extras.get(OVER_CAP_FACTORIZED_FALLBACK_MODE)
    if not isinstance(bridge_trace, Mapping):
        raise RTPThreeArmRunnerError("over-cap bridge lacks its factorized-fallback attestation")
    if set(bridge_trace) != {
        "classification",
        "action_space",
        "factorized_greedy_fallback",
    }:
        raise RTPThreeArmRunnerError("over-cap bridge attestation has an invalid field set")
    if bridge_trace.get("classification") != OVER_CAP_FACTORIZED_FALLBACK_REASON:
        raise RTPThreeArmRunnerError("over-cap bridge has an invalid classification")
    if bridge_trace.get("factorized_greedy_fallback") is not True:
        raise RTPThreeArmRunnerError("over-cap bridge did not attest factorized greedy fallback")
    bridge_action_space = _strict_over_cap_action_space(bridge_trace.get("action_space"))
    if canonical_digest(bridge_action_space) != canonical_digest(dict(action_space)):
        raise RTPThreeArmRunnerError("over-cap bridge action-space differs from runner classification")
    required = {
        "mode": "fallback",
        "fallback_code": "action_space_too_large",
        "neural_passes": 0,
        "required_neural_passes": 0,
        "legal_count": 0,
        "decision_mode": "",
    }
    for field, expected in required.items():
        actual = diagnostic.get(field)
        if isinstance(expected, int):
            if type(actual) is not int or actual != expected:
                raise RTPThreeArmRunnerError(
                    f"over-cap bridge diagnostic differs at {field}"
                )
        elif actual != expected:
            raise RTPThreeArmRunnerError(
                f"over-cap bridge diagnostic differs at {field}"
            )
    return dict(required)


@dataclass
class DecisionTelemetry:
    arm: str
    candidate_decisions: int = 0
    #: Candidate decisions whose complete ordered action space was at or below
    #: the fixed r198 cap and therefore could enter the common planner probe.
    #: The special over-cap stratum remains a real candidate decision/latency
    #: sample but never dilutes planner, fallback, or pass denominators.
    planner_eligible_candidate_decisions: int = 0
    over_cap_factorized_fallback_decisions: int = 0
    #: Explicit turn-order prompts are engine controls, rather than policy
    #: decisions.  Keep their independently verified trace separate so they
    #: cannot dilute planner-use, fallback, or latency denominators.
    forced_turn_order_controls: int = 0
    intended_complex_decisions: int = 0
    recursive_intended_complex_decisions: int = 0
    direct_bridge_decisions: int = 0
    recursive_decisions: int = 0
    fallback_decisions: int = 0
    unexpected_recursive_fallback_decisions: int = 0
    neural_budget_exceeded: int = 0
    neural_budget_failures: int = 0
    illegal_action_count: int = 0
    candidate_forfeit_count: int = 0
    latency_seconds: float = 0.0
    normal_recursive_plan_passes: list[int] | None = None
    forced_replan_passes: list[int] | None = None
    decision_latency_trace: list[dict[str, Any]] | None = None
    decision_diagnostics: list[dict[str, Any]] | None = None
    forced_turn_order_control_trace: list[dict[str, Any]] | None = None
    over_cap_factorized_fallback_trace: list[dict[str, Any]] | None = None
    expected_recursive_fallback_decisions: int = 0
    recursive_mode_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.normal_recursive_plan_passes = []
        self.forced_replan_passes = []
        self.decision_latency_trace = []
        self.decision_diagnostics = []
        self.forced_turn_order_control_trace = []
        self.over_cap_factorized_fallback_trace = []
        self.recursive_mode_counts = {mode: 0 for mode in _RECURSIVE_MODE_COUNT_KEYS}

    def observe_forced_turn_order_control(
        self,
        *,
        expected_action: Sequence[int],
        returned_action: Sequence[int],
        prompt_context: int | str,
        prompt_context_encoding: str,
    ) -> None:
        """Record the one externally forced turn-order action per game.

        This path is intentionally outside ``observe``: it performs neither a
        policy comparison nor an RTP action, and is excluded from all decision
        latency and planner-use measurements.
        """

        if self.forced_turn_order_controls >= 1:
            raise RTPThreeArmRunnerError(
                "a game emitted more than one forced turn-order control"
            )
        expected = list(expected_action)
        returned = list(returned_action)
        if (
            len(expected) != 1
            or len(returned) != 1
            or any(type(value) is not int or value < 0 for value in [*expected, *returned])
            or expected != returned
        ):
            raise RTPThreeArmRunnerError(
                "forced turn-order control action does not exactly match Yes"
            )
        if prompt_context_encoding == "numeric_41":
            valid_context = type(prompt_context) is int and prompt_context == 41
        elif prompt_context_encoding == "enum_is_first":
            valid_context = type(prompt_context) is str and prompt_context == "IsFirst"
        else:
            valid_context = False
        if not valid_context:
            raise RTPThreeArmRunnerError(
                "forced turn-order control has an invalid prompt context"
            )
        assert self.forced_turn_order_control_trace is not None
        self.forced_turn_order_control_trace.append(
            {
                "control_index": self.forced_turn_order_controls,
                "control": FORCED_TURN_ORDER_CONTROL,
                "prompt_context": prompt_context,
                "prompt_context_encoding": prompt_context_encoding,
                "expected_action": expected,
                "returned_action": returned,
                "verified_observation_action_contract": True,
                "rtp_diagnostics_absent": True,
                "complexity_probe_not_invoked": True,
                "excluded_from_candidate_decisions": True,
                "excluded_from_intended_complex_denominator": True,
                "excluded_from_latency": True,
            }
        )
        self.forced_turn_order_controls += 1

    def observe(
        self,
        diagnostic: Mapping[str, Any] | None,
        latency: float,
        *,
        complexity_intent: Mapping[str, Any] | None,
        returned_action: Sequence[Any] | None = None,
        over_cap_factorized_fallback: Mapping[str, Any] | None = None,
    ) -> None:
        if over_cap_factorized_fallback is not None:
            if complexity_intent is not None:
                # The special stratum is valid only when the runner classified
                # it before the common complete-action complexity probe.  Do
                # not silently discard a supplied probe result and relabel a
                # planner-eligible decision after the fact.
                raise RTPThreeArmRunnerError(
                    "over-cap factorized fallback cannot carry complexity intent"
                )
            self._observe_over_cap_factorized_fallback(
                diagnostic,
                latency,
                returned_action=returned_action,
                selection=over_cap_factorized_fallback,
            )
            return
        if not isinstance(complexity_intent, Mapping):
            raise RTPThreeArmRunnerError("normal candidate decision lacks complexity intent")
        self.candidate_decisions += 1
        self.planner_eligible_candidate_decisions += 1
        self.latency_seconds += float(latency)
        intended_complex = complexity_intent.get("intended_complex")
        common_reason = complexity_intent.get("planner_reason")
        if not isinstance(intended_complex, bool) or not isinstance(common_reason, str) or not common_reason:
            raise RTPThreeArmRunnerError("invalid common complexity-intent trace")
        if intended_complex:
            self.intended_complex_decisions += 1
        if self.arm == "no_rtp":
            if diagnostic is not None:
                raise RTPThreeArmRunnerError("no-RTP arm emitted RTP diagnostics")
            trace_mode = "no_rtp"
            planner_mode = "no_rtp"
            planner_reason = common_reason
            fallback_classification: str | None = None
        else:
            if diagnostic is None:
                raise RTPThreeArmRunnerError("RTP arm omitted per-decision diagnostics")
            planner_mode = _planner_mode_from_diagnostic(diagnostic, arm=self.arm)
            extras = diagnostic.get("extras")
            extras = dict(extras) if isinstance(extras, Mapping) else {}
            raw_intent = extras.get("pre_forcing_complexity_intent")
            bridge_intent = dict(raw_intent) if isinstance(raw_intent, Mapping) else None
            if (
                bridge_intent is not None
                and not bridge_intent.get("inherited", False)
                and bridge_intent.get("new_turn") is True
            ):
                bridge_would_recurse = bridge_intent.get("would_recurse")
                if not isinstance(bridge_would_recurse, bool):
                    raise RTPThreeArmRunnerError("bridge complexity trace has no would_recurse")
                if bridge_would_recurse != intended_complex:
                    raise RTPThreeArmRunnerError(
                        "common complexity intent differs from bridge pre-forcing gate"
                    )
            successful_recursive = planner_mode in _SUCCESSFUL_RECURSIVE_MODES
            fallback = planner_mode in _RECURSIVE_FALLBACK_MODES
            fallback_classification: str | None = None
            if self.arm != "recursive_rtp" and (successful_recursive or fallback):
                raise RTPThreeArmRunnerError(
                    "non-recursive arm emitted a recursive planner or fallback mode"
                )
            if intended_complex and successful_recursive:
                self.recursive_intended_complex_decisions += 1
            if successful_recursive:
                self.recursive_decisions += 1
                assert self.recursive_mode_counts is not None
                self.recursive_mode_counts[planner_mode] += 1
                trace_mode = "recursive_rtp"
            elif fallback:
                self.fallback_decisions += 1
                assert self.recursive_mode_counts is not None
                self.recursive_mode_counts[planner_mode] += 1
                trace_mode = "fallback"
                if not intended_complex:
                    raise RTPThreeArmRunnerError(
                        "recursive fallback was not complexity-intended"
                    )
                fallback_classification = _fallback_classification(
                    diagnostic, planner_mode
                )
                if fallback_classification == "unexpected":
                    self.unexpected_recursive_fallback_decisions += 1
                else:
                    self.expected_recursive_fallback_decisions += 1
            else:
                trace_mode = CANONICAL_DIRECT_ARM
                fallback_classification = None
                if planner_mode not in {"direct_bridge", "direct_policy"}:
                    raise RTPThreeArmRunnerError("RTP bridge emitted an unsupported direct mode")
                if self.arm == CANONICAL_DIRECT_ARM:
                    if planner_mode != "direct_bridge" or extras.get(
                        "force_direct_bridge_only"
                    ) is not True:
                        raise RTPThreeArmRunnerError(
                            "direct arm did not attest bridge-only behavior"
                        )
                elif planner_mode != "direct_policy":
                    raise RTPThreeArmRunnerError(
                        "recursive arm may use only direct_policy for non-complex decisions"
                    )
                self.direct_bridge_decisions += 1
            fallback_code = str(diagnostic.get("fallback_code", ""))
            if fallback_code == "neural_pass_budget_exceeded":
                self.neural_budget_exceeded += 1
                self.neural_budget_failures += 1
            neural_passes = int(diagnostic.get("neural_passes", 0) or 0)
            if planner_mode == "recursive_plan":
                self.normal_recursive_plan_passes.append(neural_passes)
            elif planner_mode == "replan_with_program":
                self.forced_replan_passes.append(neural_passes)
            details_raw = diagnostic.get("decision_diagnostics")
            details = dict(details_raw) if isinstance(details_raw, Mapping) else {}
            planner_reason = str(
                diagnostic.get("fallback_reason")
                or details.get("reason")
                or ("forced_direct_bridge_only" if planner_mode == "direct_bridge" else "")
                or ("complexity_gate_direct_policy" if planner_mode == "direct_policy" else "")
                or planner_mode
            )
            self.decision_diagnostics.append(
                {
                    "planner_mode": planner_mode,
                    "planner_reason": planner_reason,
                    "complexity_intent": dict(complexity_intent),
                    "fallback_code": fallback_code,
                    "neural_passes": neural_passes,
                    "required_neural_passes": int(
                        diagnostic.get("required_neural_passes", 0) or 0
                    ),
                }
            )
        if self.arm == "no_rtp":
            fallback_code = ""
            neural_passes = 0
            planner_reason = common_reason
            fallback_classification = None
            self.decision_diagnostics.append(
                {
                    "planner_mode": planner_mode,
                    "planner_reason": planner_reason,
                    "complexity_intent": dict(complexity_intent),
                    "fallback_code": fallback_code,
                    "neural_passes": neural_passes,
                    "required_neural_passes": 0,
                }
            )
        self.decision_latency_trace.append(
            {
                "decision_index": self.candidate_decisions - 1,
                "mode": trace_mode,
                "planner_mode": planner_mode,
                "planner_reason": planner_reason,
                "intended_complex": intended_complex,
                "fallback_classification": fallback_classification,
                "latency_seconds": float(latency),
            }
        )

    def _observe_over_cap_factorized_fallback(
        self,
        diagnostic: Mapping[str, Any] | None,
        latency: float,
        *,
        returned_action: Sequence[Any] | None,
        selection: Mapping[str, Any],
    ) -> None:
        """Record the legal factorized path above the complete-action cap."""

        if not isinstance(selection, Mapping):
            raise RTPThreeArmRunnerError("over-cap selection context is invalid")
        expected_selection_keys = {
            "action_space",
            "action_space_sha256",
            "observation_sha256",
            "candidate_policy_input_sha256",
            "logical_pre_action_sha256",
        }
        if set(selection) != expected_selection_keys:
            raise RTPThreeArmRunnerError("over-cap selection context has an invalid field set")
        action_space = _strict_over_cap_action_space(selection.get("action_space"))
        action_space_sha256 = _require_sha256_text(
            selection.get("action_space_sha256"), "over-cap action-space digest"
        )
        if action_space_sha256 != canonical_digest(action_space):
            raise RTPThreeArmRunnerError("over-cap action-space digest does not bind its summary")
        observation_sha256 = _require_sha256_text(
            selection.get("observation_sha256"), "over-cap observation digest"
        )
        candidate_policy_input_sha256 = _require_sha256_text(
            selection.get("candidate_policy_input_sha256"),
            "over-cap candidate policy-input digest",
        )
        logical_pre_action_sha256 = _require_sha256_text(
            selection.get("logical_pre_action_sha256"),
            "over-cap logical pre-action digest",
        )
        if logical_pre_action_sha256 != canonical_digest(
            {
                "observation_sha256": observation_sha256,
                "action_space_sha256": action_space_sha256,
                "candidate_policy_input_sha256": candidate_policy_input_sha256,
            }
        ):
            raise RTPThreeArmRunnerError(
                "over-cap logical pre-action digest does not bind its inputs"
            )
        stage_count = _factorized_stage_count_for_over_cap_action(
            action_space, returned_action if returned_action is not None else ()
        )
        returned = list(returned_action or ())
        bridge_diagnostic = _over_cap_bridge_diagnostic(
            diagnostic, arm=self.arm, action_space=action_space
        )
        decision_index = self.candidate_decisions
        trace_index = len(self.over_cap_factorized_fallback_trace or ())
        self.candidate_decisions += 1
        self.over_cap_factorized_fallback_decisions += 1
        self.latency_seconds += float(latency)
        fallback_code = (
            "" if bridge_diagnostic is None else bridge_diagnostic["fallback_code"]
        )
        neural_passes = 0 if bridge_diagnostic is None else bridge_diagnostic["neural_passes"]
        required_neural_passes = (
            0 if bridge_diagnostic is None else bridge_diagnostic["required_neural_passes"]
        )
        assert self.decision_diagnostics is not None
        self.decision_diagnostics.append(
            {
                "planner_mode": OVER_CAP_FACTORIZED_FALLBACK_MODE,
                "planner_reason": OVER_CAP_FACTORIZED_FALLBACK_REASON,
                "complexity_intent": None,
                "fallback_code": fallback_code,
                "neural_passes": neural_passes,
                "required_neural_passes": required_neural_passes,
            }
        )
        assert self.decision_latency_trace is not None
        self.decision_latency_trace.append(
            {
                "decision_index": decision_index,
                "mode": OVER_CAP_FACTORIZED_FALLBACK_MODE,
                "planner_mode": OVER_CAP_FACTORIZED_FALLBACK_MODE,
                "planner_reason": OVER_CAP_FACTORIZED_FALLBACK_REASON,
                # The common complexity predicate was deliberately not run,
                # so this is unassessed rather than an ordinary direct/false
                # complexity result.
                "intended_complex": None,
                "fallback_classification": None,
                "latency_seconds": float(latency),
                "over_cap_trace_index": trace_index,
            }
        )
        assert self.over_cap_factorized_fallback_trace is not None
        self.over_cap_factorized_fallback_trace.append(
            {
                "decision_index": decision_index,
                "arm": self.arm,
                "mode": OVER_CAP_FACTORIZED_FALLBACK_MODE,
                "classification": OVER_CAP_FACTORIZED_FALLBACK_REASON,
                "action_space": action_space,
                "action_space_sha256": action_space_sha256,
                "observation_sha256": observation_sha256,
                "candidate_policy_input_sha256": candidate_policy_input_sha256,
                "logical_pre_action_sha256": logical_pre_action_sha256,
                "returned_action": returned,
                "factorized_teacher_forcing_legal": True,
                "factorized_teacher_forcing_stage_count": stage_count,
                "complexity_probe_not_invoked": True,
                "neural_passes": neural_passes,
                "required_neural_passes": required_neural_passes,
                "neural_budget_failure": False,
                "rtp_diagnostic": bridge_diagnostic,
                "included_in_candidate_decisions": True,
                "included_in_candidate_latency": True,
                "excluded_from_planner_eligible_candidate_decisions": True,
                "excluded_from_intended_complex_denominator": True,
                "excluded_from_direct_bridge_metrics": True,
                "excluded_from_recursive_metrics": True,
                "excluded_from_fallback_metrics": True,
                "excluded_from_neural_pass_metrics": True,
                "excluded_from_recursive_latency": True,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        over_cap_trace = list(self.over_cap_factorized_fallback_trace or ())
        return {
            "candidate_decisions": self.candidate_decisions,
            "planner_eligible_candidate_decisions": self.planner_eligible_candidate_decisions,
            "over_cap_factorized_fallback_decisions": self.over_cap_factorized_fallback_decisions,
            "forced_turn_order_controls": self.forced_turn_order_controls,
            "forced_turn_order_control_trace": list(
                self.forced_turn_order_control_trace or ()
            ),
            "intended_complex_decisions": self.intended_complex_decisions,
            "intended_complex_decision_scope": "new_turn_complexity_gate_only",
            "recursive_intended_complex_decisions": self.recursive_intended_complex_decisions,
            "successful_recursive_intended_complex_decisions": self.recursive_intended_complex_decisions,
            "direct_bridge_decisions": self.direct_bridge_decisions,
            "recursive_decisions": self.recursive_decisions,
            "fallback_decisions": self.fallback_decisions,
            "unexpected_recursive_fallback_decisions": self.unexpected_recursive_fallback_decisions,
            "expected_recursive_fallback_decisions": self.expected_recursive_fallback_decisions,
            "neural_budget_exceeded": self.neural_budget_exceeded,
            "neural_budget_failures": self.neural_budget_failures,
            "illegal_action_count": self.illegal_action_count,
            "candidate_forfeit_count": self.candidate_forfeit_count,
            "new_forfeit_count": 0,
            "latency_seconds": self.latency_seconds,
            "normal_recursive_plan_passes": list(self.normal_recursive_plan_passes or ()),
            "forced_replan_passes": list(self.forced_replan_passes or ()),
            "decision_latency_trace": list(self.decision_latency_trace or ()),
            "decision_diagnostics": list(self.decision_diagnostics or ()),
            "over_cap_factorized_fallback_trace": over_cap_trace,
            "over_cap_factorized_fallback_trace_sha256": canonical_digest(over_cap_trace),
            "recursive_mode_counts": dict(self.recursive_mode_counts or {}),
        }


def _current_seat(observation: Mapping[str, Any]) -> int:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise RTPThreeArmRunnerError("engine observation has no current state")
    seat = current.get("yourIndex")
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in {0, 1}:
        raise RTPThreeArmRunnerError("engine observation has invalid acting seat")
    return int(seat)


def _battle_finished(battle: Any) -> bool:
    value = getattr(battle, "finished", None)
    return bool(value() if callable(value) else value)


def _battle_winner(battle: Any) -> int:
    value = getattr(battle, "winner", None)
    value = value() if callable(value) else value
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2}:
        raise RTPThreeArmRunnerError("snapshot engine returned an invalid terminal result")
    return int(value)


def _battle_step(battle: Any, action: list[int]) -> None:
    outcome = battle.step(action)
    if outcome is False:
        raise ValueError("snapshot engine rejected action")
    if isinstance(outcome, Mapping) and outcome.get("accepted") is False:
        raise ValueError("snapshot engine rejected action")


def _runtime_profile_contract(
    manifest: Mapping[str, Any], arm: str, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping):
        raise RTPThreeArmRunnerError("manifest arms are invalid")
    source_arm = _canonical_runtime_arm(arm, manifest)
    spec = arms.get(source_arm)
    if not isinstance(spec, Mapping):
        raise RTPThreeArmRunnerError("manifest arm specification is missing")
    expected: dict[str, Any] = {
        "arm": arm,
        "runtime_artifact_sha256": _require_identity(
            spec.get("runtime_artifact"), "runtime artifact"
        )["sha256"],
        "runtime_profile_sha256": _require_identity(
            spec.get("runtime_profile"), "runtime profile"
        )["sha256"],
    }
    # The three-arm comparison distinguishes the sidecar attached to an
    # acting bridge from the distinct instrumentation-only instance used for
    # the arm-independent complexity probe.  A generic ``rtp_sidecar_sha256``
    # was ambiguous (and let no-RTP look action-attached), so deliberately do
    # not emit or accept it in the evaluated runtime identity.
    direct_key = _direct_arm_key(manifest)
    direct_spec = arms.get(direct_key)
    recursive_spec = arms.get("recursive_rtp")
    if not isinstance(direct_spec, Mapping) or not isinstance(recursive_spec, Mapping):
        raise RTPThreeArmRunnerError("manifest bridge arms are invalid")
    direct_sidecar = _require_identity(
        direct_spec.get("rtp_sidecar"), "direct bridge RTP sidecar"
    )
    recursive_sidecar = _require_identity(
        recursive_spec.get("rtp_sidecar"), "recursive RTP sidecar"
    )
    if not _identity_equal(direct_sidecar, recursive_sidecar):
        raise RTPThreeArmRunnerError(
            "direct and recursive arms must bind the same complexity-probe sidecar"
        )
    action_sidecar = spec.get("rtp_sidecar")
    expected.update(
        {
            "action_attached_rtp_sidecar_sha256": (
                None
                if action_sidecar is None
                else _require_identity(action_sidecar, "action-attached RTP sidecar")[
                    "sha256"
                ]
            ),
            "complexity_probe_sidecar_sha256": direct_sidecar["sha256"],
            "complexity_probe_sidecar_instrumentation_only": True,
            "complexity_probe_latency_excluded": True,
            "rtp_action_attachment_enabled": arm != "no_rtp",
            # All r198 arms remain shadow-only.  Attaching the sidecar to B/C
            # exercises its policy behavior but is never serving authority.
            "rtp_action_authority_enabled": False,
        }
    )
    shared = manifest.get("shared_artifacts")
    if not isinstance(shared, Mapping):
        raise RTPThreeArmRunnerError("manifest shared artifacts are invalid")
    for name, value in shared.items():
        if isinstance(value, Mapping) and "path" in value and "sha256" in value:
            expected[f"{name}_sha256"] = _require_identity(value, f"shared {name}")["sha256"]
    profile = _json_object(
        _require_identity(spec.get("runtime_profile"), "runtime profile")["path"],
        "runtime profile",
    )
    profile = profile.get("rtp", profile)
    if not isinstance(profile, Mapping):
        raise RTPThreeArmRunnerError("runtime profile RTP section is invalid")
    for key in (
        "recursive_turn_planner_enabled",
        "direct_bridge_enabled",
        "force_direct_bridge_only",
        "max_neural_passes",
        "max_action_combos",
    ):
        expected[key] = profile.get(key)
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise RTPThreeArmRunnerError(f"worker runtime identity mismatch at {key}")
    if "rtp_sidecar_sha256" in runtime:
        raise RTPThreeArmRunnerError(
            "runtime identity must use explicit probe/action sidecar fields, not rtp_sidecar_sha256"
        )
    return {
        **expected,
        **{key: value for key, value in runtime.items() if key not in expected},
    }


def _verify_agent_mode(candidate: Any, arm: str) -> None:
    recursive_enabled = bool(getattr(candidate, "use_recursive_turn_planner", False))
    direct_only = bool(getattr(candidate, "force_direct_bridge_only", False))
    bridge = getattr(candidate, "_rtp_bridge", None)
    if arm == "no_rtp":
        if recursive_enabled or direct_only or bridge is not None:
            raise RTPThreeArmRunnerError("no-RTP child has an RTP-enabled candidate")
    elif arm == CANONICAL_DIRECT_ARM:
        if not recursive_enabled or not direct_only or bridge is None:
            raise RTPThreeArmRunnerError("direct arm does not force bridge-only behavior")
    elif arm == "recursive_rtp":
        if not recursive_enabled or direct_only or bridge is None:
            raise RTPThreeArmRunnerError("recursive arm has incorrect bridge mode")
    else:
        raise RTPThreeArmRunnerError("unknown arm")
    if getattr(candidate, "strict_runtime", None) is not True:
        raise RTPThreeArmRunnerError("candidate strict_runtime must be true")
    if bool(getattr(candidate, "sample_actions", False)):
        raise RTPThreeArmRunnerError("candidate sampling is forbidden in evaluation")
    if bool(getattr(candidate, "use_mcts", False)):
        raise RTPThreeArmRunnerError("candidate MCTS is forbidden in evaluation")
    if getattr(candidate, "leaf_backend", None) is not None:
        raise RTPThreeArmRunnerError("candidate remote leaf backend is forbidden")
    if bool(getattr(candidate, "collect_targets", False)):
        raise RTPThreeArmRunnerError("evaluation candidate must not collect targets")


def _process_identity() -> dict[str, str]:
    pid = str(os.getpid())
    boot_id = "unavailable"
    start = "unavailable"
    for candidate in (Path("/proc/sys/kernel/random/boot_id"),):
        if candidate.is_file():
            boot_id = candidate.read_text(encoding="utf-8").strip() or "unavailable"
    stat_path = Path("/proc/self/stat")
    if stat_path.is_file():
        pieces = stat_path.read_text(encoding="utf-8").split()
        if len(pieces) > 21:
            start = pieces[21]
    return {"process_id": pid, "boot_id": boot_id, "process_start_ticks": start}


def _verify_worker_environment_bindings(
    environment: Mapping[str, str],
    *,
    capability: Mapping[str, Any],
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the child can import only its sealed source/CG closure.

    This is deliberately repeated inside the fresh child.  Parent-side
    sanitizing prevents accidental inheritance, while the child-side check
    closes the gap between ``exec`` and its first baseline/model import.
    """

    for key in (
        "CG_LIB_PATH",
        "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT",
        "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256",
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT",
        "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONHASHSEED",
    ):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise RTPThreeArmRunnerError(f"worker environment lacks {key}")
    if environment["CUDA_VISIBLE_DEVICES"] != R198_BLACKWELL_UUID:
        raise RTPThreeArmRunnerError("worker does not pin the required Blackwell UUID")
    if environment["PYTHONDONTWRITEBYTECODE"] != "1" or environment[
        "PYTHONNOUSERSITE"
    ] != "1" or environment["PYTHONHASHSEED"] != "0":
        raise RTPThreeArmRunnerError("worker Python environment is not deterministic and sealed")
    runtime_contract = _sealed_path_identity(
        environment["POKEBOT_R198_EVAL_RUNTIME_CONTRACT"],
        "worker r198 candidate runtime contract environment",
    )
    if environment["POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256"] != runtime_contract[
        "sha256"
    ]:
        raise RTPThreeArmRunnerError(
            "worker r198 candidate runtime contract environment checksum differs"
        )
    runtime_contract_payload = _json_object(
        runtime_contract["path"], "worker r198 candidate runtime contract"
    )
    if (
        runtime_contract_payload.get("schema") != R198_CANDIDATE_SNAPSHOT_SCHEMA
        or runtime_contract_payload.get("status") != "sealed"
    ):
        raise RTPThreeArmRunnerError(
            "worker r198 runtime contract is not the sealed candidate snapshot"
        )
    for key, expected in (
        ("no_symlinks", True),
        ("all_paths_read_only", True),
    ):
        if runtime_contract_payload.get(key) is not expected:
            raise RTPThreeArmRunnerError(
                f"worker r198 candidate runtime contract fails {key}"
            )
    forbidden = [
        key
        for key in environment
        if key.startswith(("LD_", "DYLD_"))
        or (
            key.startswith(("POKEBOT_", "CG_"))
            and key
            not in _FACTORY_ENV_EXACT_KEYS
            and key not in _ARM_CONTROL_ENV
            and key not in _RUNNER_ACTION_FENCE_ENV
        )
    ]
    if forbidden:
        raise RTPThreeArmRunnerError(
            "worker environment contains undeclared runtime controls: "
            + ", ".join(sorted(forbidden))
        )
    source_root = _physical_path(
        environment["POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT"],
        "worker evaluation source snapshot root",
        require_directory=True,
    )
    running_source_root = _physical_path(
        Path(__file__).parent.parent, "runner source root", require_directory=True
    )
    if source_root != running_source_root:
        raise RTPThreeArmRunnerError(
            "worker source snapshot root differs from the executing source tree"
        )
    snapshot_manifest = _sealed_path_identity(
        source_root / R198_SOURCE_SNAPSHOT_MANIFEST,
        "worker source snapshot manifest",
    )
    source_payload = _json_object(snapshot_manifest["path"], "worker source snapshot manifest")
    if source_payload.get("schema") != R198_SOURCE_SNAPSHOT_SCHEMA:
        raise RTPThreeArmRunnerError("worker source snapshot manifest schema is invalid")
    if source_payload.get("source_tree_sha256") != environment[
        "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256"
    ]:
        raise RTPThreeArmRunnerError("worker source tree checksum differs from sealed manifest")
    cg_runtime = _physical_path(
        environment["CG_LIB_PATH"], "worker CG_LIB_PATH", require_directory=True
    )
    cg_package = _physical_path(cg_runtime / "cg", "worker CG package", require_directory=True)
    cg_engine = _physical_path(cg_package / "libcg.so", "worker CG engine", require_file=True)
    for path, label in (
        (cg_runtime, "worker CG runtime"),
        (cg_package, "worker CG package"),
        (cg_engine, "worker CG engine"),
    ):
        if stat.S_IMODE(os.lstat(path).st_mode) & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise RTPThreeArmRunnerError(f"{label} is writable")
    engine_identity = _identity(cg_engine, "worker CG engine")
    snapshot_closure = source_payload.get("eval_cg_closure")
    if not isinstance(snapshot_closure, Mapping):
        raise RTPThreeArmRunnerError("worker source snapshot lacks its CG closure binding")
    library = _require_immutable_identity(
        snapshot_closure.get("library"), "worker source snapshot CG engine"
    )
    if not _identity_equal(library, engine_identity):
        raise RTPThreeArmRunnerError("worker CG_LIB_PATH differs from source snapshot closure")
    if snapshot_closure.get("runtime_root") != str(cg_runtime) or snapshot_closure.get(
        "runtime_path"
    ) != str(cg_package):
        raise RTPThreeArmRunnerError("worker CG runtime path differs from source snapshot closure")
    if snapshot_closure.get("physical_read_only_copy") is not True or snapshot_closure.get(
        "library_mode"
    ) != 0o444:
        raise RTPThreeArmRunnerError("worker source snapshot CG closure is not immutable")
    source_closure_record = _require_immutable_identity(
        snapshot_closure.get("closure_manifest"), "worker source snapshot CG closure record"
    )
    if _json_object(source_closure_record["path"], "worker source snapshot CG closure record").get(
        "schema"
    ) != EVALUATION_CG_CLOSURE_SCHEMA:
        raise RTPThreeArmRunnerError("worker source snapshot CG closure record schema is invalid")
    expected_engine = capability.get("engine_artifact")
    closure_engine = closure.get("engine_artifact")
    runtime_library = closure.get("runtime_library")
    if (
        not isinstance(expected_engine, Mapping)
        or not isinstance(closure_engine, Mapping)
        or not isinstance(runtime_library, Mapping)
    ):
        raise RTPThreeArmRunnerError("worker has no bound CG engine identity")
    for expected in (expected_engine, closure_engine, runtime_library):
        if any(engine_identity.get(key) != expected.get(key) for key in ("sha256", "bytes")):
            raise RTPThreeArmRunnerError("worker CG engine differs from sealed closure/capability")
    if not _identity_equal(engine_identity, runtime_library):
        raise RTPThreeArmRunnerError(
            "worker CG engine path differs from manifest runtime-library identity"
        )
    evidence = closure.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RTPThreeArmRunnerError("worker has no sealed CG closure evidence")
    try:
        closure_manifest = evidence["closure_manifest"]["identity"]
        metadata_parity = evidence["metadata_parity"]["identity"]
    except (KeyError, TypeError) as exc:
        raise RTPThreeArmRunnerError("worker CG closure evidence is incomplete") from exc
    if not isinstance(closure_manifest, Mapping) or not isinstance(metadata_parity, Mapping):
        raise RTPThreeArmRunnerError("worker CG closure identities are invalid")
    return {
        "source_snapshot_manifest_sha256": snapshot_manifest["sha256"],
        "source_tree_sha256": str(source_payload["source_tree_sha256"]),
        "evaluation_cg_closure_receipt_sha256": str(closure["receipt"]["sha256"]),
        "evaluation_cg_engine_sha256": str(engine_identity["sha256"]),
        "evaluation_cg_closure_manifest_sha256": str(closure_manifest["sha256"]),
        "evaluation_cg_metadata_parity_sha256": str(metadata_parity["sha256"]),
        "evaluation_cg_engine_path": str(engine_identity["path"]),
        "evaluation_cg_engine_bytes": int(engine_identity["bytes"]),
        "evaluation_cg_engine_mode": int(stat.S_IMODE(os.lstat(cg_engine).st_mode)),
        "candidate_runtime_contract_sha256": str(runtime_contract["sha256"]),
        "candidate_runtime_contract_path": str(runtime_contract["path"]),
    }


def _sanitize_environment(
    raw: Mapping[str, str],
    scratch: Path,
    *,
    capability: Mapping[str, Any],
    closure: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()):
        raise RTPThreeArmRunnerError("factory worker_environment must be string:string")
    if set(raw).intersection(_RUNNER_ACTION_FENCE_ENV):
        raise RTPThreeArmRunnerError(
            "factory must not supply evaluator action-fence environment variables"
        )
    unknown = sorted(set(raw) - _FACTORY_ENV_EXACT_KEYS - _ARM_CONTROL_ENV)
    if unknown:
        raise RTPThreeArmRunnerError(
            "factory worker_environment has non-allowlisted variables: "
            + ", ".join(unknown)
        )
    protected = sorted(
        key
        for key in raw
        if key in {"PYTHONPATH", "PATH", "CUDA_VISIBLE_DEVICES"}
        or key.startswith(("LD_", "DYLD_", "PYTHON"))
    )
    if protected:
        raise RTPThreeArmRunnerError(
            "factory may not override protected worker environment variables: "
            + ", ".join(protected)
        )
    inherited_forbidden = [
        key
        for key in os.environ
        if key.startswith(("POKEBOT_", "CG_", "LD_", "PYTHON"))
    ]
    # Do not pass through inherited controller state.  ``inherited_forbidden``
    # is retained only as a child-audit fact; the child gets explicit values.
    del inherited_forbidden
    scratch = _physical_path(scratch, "worker scratch", require_directory=True)
    source_root = _physical_path(
        Path(__file__).parent.parent, "runner source root", require_directory=True
    )
    required_factory_environment = (
        "CG_LIB_PATH",
        "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT",
        "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256",
    )
    missing = [key for key in required_factory_environment if not raw.get(key)]
    if missing:
        raise RTPThreeArmRunnerError(
            "factory worker environment lacks required sealed runtime bindings: "
            + ", ".join(missing)
        )
    declared_source_root = _physical_path(
        raw["POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT"],
        "factory evaluation source snapshot root",
        require_directory=True,
    )
    if declared_source_root != source_root:
        raise RTPThreeArmRunnerError(
            "factory evaluation source snapshot root differs from the running source tree"
        )
    source_tree_sha256 = raw["POKEBOT_R198_EVAL_SOURCE_TREE_SHA256"]
    if (
        not source_tree_sha256.startswith("sha256:")
        or len(source_tree_sha256) != 71
    ):
        raise RTPThreeArmRunnerError(
            "factory evaluation source tree identity must be a SHA-256 digest"
        )
    common: dict[str, str] = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(source_root),
        "PATH": os.environ.get("PATH", ""),
        "CUDA_VISIBLE_DEVICES": R198_BLACKWELL_UUID,
        "TMPDIR": str(scratch),
        "XDG_CACHE_HOME": str(scratch / "xdg-cache"),
        "TORCH_HOME": str(scratch / "torch-home"),
        "TORCHINDUCTOR_CACHE_DIR": str(scratch / "torchinductor-cache"),
        "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
    }
    full = {**common, **dict(raw)}
    for path_key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        value = _lexical_absolute_path(full[path_key], f"worker {path_key}")
        if scratch not in value.parents and value != scratch:
            raise RTPThreeArmRunnerError("factory worker environment escapes arm scratch directory")
        _ensure_physical_directory(value, f"worker {path_key}")
        _physical_path(value, f"worker {path_key}", require_directory=True)
    _verify_worker_environment_bindings(
        full, capability=capability, closure=closure
    )
    common_digest_input = {
        key: value
        for key, value in full.items()
        if key not in _ARM_CONTROL_ENV
        and key not in _RUNNER_ACTION_FENCE_ENV
        and key not in {"TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"}
    }
    return full, {
        "common_sanitized_environment_sha256": canonical_digest(common_digest_input),
        "arm_environment_sha256": canonical_digest(full),
    }


def _factory_call(factory: Any, method: str, **kwargs: Any) -> Any:
    target = getattr(factory, method, None)
    if not callable(target):
        raise RTPThreeArmRunnerError(f"evaluation factory lacks {method}()")
    return target(**kwargs)


def _worker_response(
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = request.get("manifest_path")
    factory_ref = request.get("factory")
    arm = _canonical_arm(str(request.get("arm", "")))
    cell = request.get("cell")
    snapshot_seal = request.get("snapshot_seal")
    requested_authority = request.get("evaluation_authority")
    requested_action_fence = request.get("action_fence")
    launch_nonce = request.get("launch_nonce")
    if not isinstance(manifest_path, str) or not isinstance(factory_ref, str):
        raise RTPThreeArmRunnerError("worker request lacks manifest/factory")
    if (
        not isinstance(cell, Mapping)
        or not isinstance(snapshot_seal, Mapping)
        or not isinstance(requested_authority, Mapping)
        or not isinstance(launch_nonce, str)
        or not launch_nonce
    ):
        raise RTPThreeArmRunnerError("worker request is malformed")
    if arm == "no_rtp":
        if requested_action_fence is not None:
            raise RTPThreeArmRunnerError("no-RTP worker request carries an action fence")
    elif not isinstance(requested_action_fence, Mapping):
        raise RTPThreeArmRunnerError("RTP worker request lacks an action fence")
    manifest, manifest_identity = _read_manifest(manifest_path)
    if manifest_identity["sha256"] != request.get("manifest_sha256"):
        raise RTPThreeArmRunnerError("worker manifest changed after parent verification")
    request_authority_identity = _require_immutable_identity(
        requested_authority, "worker evaluation-only authority"
    )
    authority = _verify_evaluation_authority(
        request_authority_identity["path"], manifest_identity=manifest_identity
    )
    if not _identity_equal(request_authority_identity, authority["identity"]):
        raise RTPThreeArmRunnerError("worker authority identity differs from its request")
    _manifest_arms(manifest)
    capability = _verify_pairing_capability(manifest)
    closure = _verify_evaluation_cg_closure(manifest, capability)
    environment_evidence = _verify_worker_environment_bindings(
        os.environ, capability=capability, closure=closure
    )
    runtime_contract_identity = _sealed_path_identity(
        os.environ["POKEBOT_R198_EVAL_RUNTIME_CONTRACT"],
        "worker r198 candidate runtime contract",
    )
    if runtime_contract_identity["sha256"] != environment_evidence.get(
        "candidate_runtime_contract_sha256"
    ):
        raise RTPThreeArmRunnerError(
            "worker candidate runtime contract differs from environment evidence"
        )
    cohort = _verify_cohort_and_source_exclusion(manifest)
    _verify_snapshot_package(manifest, cell, cohort)
    expected_rng = cell.get("rng_identity")
    if not isinstance(expected_rng, Mapping) or expected_rng.get("kind") != "snapshot":
        raise RTPThreeArmRunnerError("worker requires a snapshot RNG identity")
    sealed = _sealed_snapshot_materials(manifest, capability).get(
        str(expected_rng.get("id", ""))
    )
    if sealed is None:
        raise RTPThreeArmRunnerError("worker cell references no sealed snapshot material")
    for key in (
        "id",
        "kind",
        "sha256",
        "bytes",
        "seal_sha256",
        "capture_boundary",
        "boundary_tag",
    ):
        if sealed.get(key) != expected_rng.get(key):
            raise RTPThreeArmRunnerError("worker snapshot seal differs from scheduled identity")
    request_seal = _require_immutable_identity(
        snapshot_seal, "worker snapshot seal"
    )
    if request_seal != sealed["seal"]:
        raise RTPThreeArmRunnerError("worker snapshot seal differs from manifest artifact")
    worker_opponents = _sealed_opponents(manifest)
    opponent_id = cell.get("opponent_id")
    if not isinstance(opponent_id, str) or opponent_id not in worker_opponents:
        raise RTPThreeArmRunnerError("worker cell names an unknown sealed opponent")
    _verify_snapshot_for_cell(
        snapshot=sealed,
        cell=cell,
        manifest=manifest,
        opponent=worker_opponents[opponent_id],
        cohort=cohort,
    )

    action_fence: dict[str, Any] | None = None
    if arm != "no_rtp":
        action_fence = _verify_action_fence(
            requested_action_fence,
            manifest_identity=manifest_identity,
            authority=authority,
            cell=cell,
            arm=arm,
            launch_nonce=launch_nonce,
        )
    evaluation_action_execution: dict[str, Any] | None = None
    if arm == "no_rtp":
        # No-RTP neither selects through RTP nor receives the narrowly scoped
        # evaluator action exception.  Its common probe is instrumentation.
        if any(key in os.environ for key in _RUNNER_ACTION_FENCE_ENV):
            raise RTPThreeArmRunnerError("no-RTP worker inherited an evaluator action fence")
    else:
        assert action_fence is not None
        evaluation_action_execution = _evaluation_action_execution_context(
            manifest_identity=manifest_identity,
            authority=authority,
            runtime_contract=runtime_contract_identity,
            action_fence=action_fence,
            cell=cell,
            arm=arm,
            launch_nonce=launch_nonce,
        )
        process = evaluation_action_execution["process"]
        # Inject only after the child independently proved its manifest,
        # authority, cell/seal, and candidate-runtime identities.  This is a
        # scoped evaluation context, never a serving/action authority flag.
        os.environ.update(
            {
                "POKEBOT_R198_EVAL_ACTION_FENCE": str(action_fence["identity"]["path"]),
                "POKEBOT_R198_EVAL_ACTION_FENCE_SHA256": str(
                    action_fence["identity"]["sha256"]
                ),
                "POKEBOT_R198_EVAL_LAUNCH_NONCE": launch_nonce,
                "POKEBOT_R198_EVAL_PROCESS_ID": str(process["process_id"]),
                "POKEBOT_R198_EVAL_PROCESS_START_TICKS": str(
                    process["process_start_ticks"]
                ),
            }
        )

    factory = _load_factory(factory_ref)
    # Factory loads model/baseline metadata first.  Snapshot restoration is the
    # final battle-state operation before the first select below.
    runtime_kwargs: dict[str, Any] = {"manifest": manifest, "cell": cell, "arm": arm}
    if evaluation_action_execution is not None:
        runtime_kwargs["evaluation_action_execution"] = evaluation_action_execution
    runtime = _as_arm_runtime(_factory_call(factory, "create_arm_runtime", **runtime_kwargs))
    candidate = runtime.candidate
    opponent = runtime.opponent
    if runtime.runtime_identity.get("candidate_snapshot_sha256") != runtime_contract_identity[
        "sha256"
    ]:
        raise RTPThreeArmRunnerError(
            "factory runtime identity does not bind the worker candidate runtime contract"
        )
    _verify_agent_mode(candidate, arm)
    reset = getattr(candidate, "reset_game", None)
    if not callable(reset):
        raise RTPThreeArmRunnerError("candidate lacks reset_game for isolated evaluation")
    reset()
    _seed_fresh_process(_seed_for_cell(manifest, cell), candidate, opponent)
    rng_identity = _rng_state_snapshot(candidate, opponent)
    engine = _factory_call(factory, "create_arm_engine", manifest=manifest, cell=cell, arm=arm)
    loaded_engine_identity = _verify_engine_identity(
        engine,
        capability,
        closure,
        loaded_closure_engine={
            "path": environment_evidence["evaluation_cg_engine_path"],
            "sha256": environment_evidence["evaluation_cg_engine_sha256"],
            "bytes": int(environment_evidence["evaluation_cg_engine_bytes"]),
            "mode": int(environment_evidence["evaluation_cg_engine_mode"]),
        },
    )
    restore = getattr(engine, "restore_sealed_snapshot_manifest", None)
    if not callable(restore):
        raise RTPThreeArmRunnerError("arm engine lacks sealed snapshot restore")
    battle = restore(str(request_seal["path"]))
    if battle is None:
        raise RTPThreeArmRunnerError("snapshot restore returned no battle")

    telemetry = DecisionTelemetry(arm=arm)
    candidate_seat = int(cell.get("candidate_seat", -1))
    if candidate_seat not in {0, 1}:
        raise RTPThreeArmRunnerError("worker cell candidate seat is invalid")
    max_steps = int(request.get("max_steps", 4000))
    steps = 0
    candidate_error: str | None = None
    opponent_error: str | None = None
    engine_error: str | None = None
    failed_seat: int | None = None
    try:
        while not _battle_finished(battle):
            if steps >= max_steps:
                engine_error = "max_steps"
                break
            observation = battle.observation()
            if not isinstance(observation, Mapping):
                raise RTPThreeArmRunnerError("snapshot engine observation is invalid")
            seat = _current_seat(observation)
            actor = candidate if seat == candidate_seat else opponent
            complexity_intent: dict[str, Any] | None = None
            forced_turn_order: tuple[list[int], int | str, str] | None = None
            over_cap_factorized_fallback: dict[str, Any] | None = None
            try:
                if seat == candidate_seat:
                    # Exact IsFirst/Yes controls are external engine rules,
                    # not candidate policy decisions.  They bypass both the
                    # complexity instrumentation and action-latency timer.
                    (
                        action,
                        elapsed,
                        complexity_intent,
                        forced_turn_order,
                        over_cap_factorized_fallback,
                    ) = _select_candidate_action(runtime, candidate, observation)
                else:
                    started = time.perf_counter()
                    action = actor(dict(observation))
                    elapsed = time.perf_counter() - started
            except BaseException as exc:  # audited as a failed game, never a win
                failed_seat = seat
                if seat == candidate_seat:
                    candidate_error = f"{type(exc).__name__}: {exc}"
                else:
                    opponent_error = f"{type(exc).__name__}: {exc}"
                break
            if not isinstance(action, Sequence) or isinstance(action, (str, bytes)) or not all(isinstance(value, int) and not isinstance(value, bool) for value in action):
                failed_seat = seat
                if seat == candidate_seat:
                    telemetry.illegal_action_count += 1
                    candidate_error = "invalid_action_shape"
                else:
                    opponent_error = "invalid_action_shape"
                break
            if seat == candidate_seat:
                if forced_turn_order is not None:
                    expected, prompt_context, prompt_context_encoding = forced_turn_order
                    try:
                        _record_forced_turn_order_control(
                            telemetry,
                            candidate,
                            expected_action=expected,
                            returned_action=list(action),
                            prompt_context=prompt_context,
                            prompt_context_encoding=prompt_context_encoding,
                        )
                    except RTPThreeArmRunnerError as exc:
                        if str(exc) == "forced_turn_order_action_mismatch":
                            telemetry.illegal_action_count += 1
                        candidate_error = str(exc)
                        failed_seat = seat
                        break
                else:
                    assert elapsed is not None
                    telemetry.observe(
                        _diagnostic_snapshot(candidate),
                        elapsed,
                        complexity_intent=complexity_intent,
                        returned_action=action,
                        over_cap_factorized_fallback=over_cap_factorized_fallback,
                    )
            try:
                _battle_step(battle, [int(value) for value in action])
            except BaseException as exc:  # engine rejection is a failed/illegal game
                failed_seat = seat
                if seat == candidate_seat:
                    telemetry.illegal_action_count += 1
                    candidate_error = f"{type(exc).__name__}: {exc}"
                else:
                    opponent_error = f"{type(exc).__name__}: {exc}"
                break
            steps += 1
    except BaseException as exc:  # never score an infrastructure failure
        engine_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            terminal_finished = _battle_finished(battle)
            terminal_winner = _battle_winner(battle) if terminal_finished else 2
        except Exception as exc:
            engine_error = engine_error or f"terminal:{type(exc).__name__}: {exc}"
        try:
            native_events = list(getattr(battle, "transcript_events", ()))
        except Exception:
            native_events = []
        try:
            battle.close()
        except Exception as exc:
            engine_error = engine_error or f"close:{type(exc).__name__}: {exc}"

    completed = engine_error is None and failed_seat is None and terminal_finished
    winner = terminal_winner if completed else 2
    if failed_seat == candidate_seat:
        telemetry.candidate_forfeit_count = 1
    if winner == candidate_seat:
        score = 1.0
        winner_text = "candidate"
    elif winner == 1 - candidate_seat:
        score = 0.0
        winner_text = "opponent"
    else:
        score = 0.5
        winner_text = "draw"
    terminal = {
        "winner": winner_text,
        "engine_result_code": winner,
        "candidate_forfeit": failed_seat == candidate_seat,
        "termination": "completed" if completed else "failed",
        "failed_seat": None if completed else failed_seat,
        "engine_error": None if completed else engine_error,
        "candidate_error": None if completed else candidate_error,
        "opponent_error": None if completed else opponent_error,
    }
    environment = request.get("environment_identity")
    if not isinstance(environment, Mapping):
        raise RTPThreeArmRunnerError("worker has no environment identity")
    isolation = dict(runtime.isolation)
    response = {
        "schema": WORKER_RESPONSE_SCHEMA,
        "status": "completed" if completed else "invalid",
        "cell_id": cell.get("cell_id"),
        "arm": arm,
        "opponent_id": cell.get("opponent_id"),
        "candidate_seat": candidate_seat,
        "candidate_score": score,
        "terminal_outcome": terminal,
        "telemetry": telemetry.as_dict(),
        "runtime_identity": _runtime_profile_contract(manifest, arm, runtime.runtime_identity),
        "native_transcript_events": native_events,
        "isolation": {
            **isolation,
            **rng_identity,
            **dict(environment),
            **environment_evidence,
            **_process_identity(),
            "launch_mode": "subprocess_exec",
            "fresh_process_per_arm": True,
            "one_cell_one_arm": True,
            "pool_reuse": False,
            "forked_from_evaluator": False,
            "process_model_load": True,
            "fresh_candidate_agent": True,
            "candidate_reset_called": True,
            "fresh_opponent_module": True,
            "engine_restore_before_first_select": True,
            "engine_restore_count": 1,
            "battle_start_after_restore_count": 0,
            "no_remote_leaf_sampling_mcts": True,
            # Complexity instrumentation is intentionally outside the action
            # timer, so recursive p95 uses only actual candidate selection.
            "complexity_probe_latency_excluded": True,
            "candidate_runtime_contract_sha256": runtime_contract_identity["sha256"],
            "action_fence_sha256": (
                None if action_fence is None else action_fence["identity"]["sha256"]
            ),
            "evaluation_action_execution_sha256": (
                None
                if evaluation_action_execution is None
                else canonical_digest(evaluation_action_execution)
            ),
            "engine_loaded_path": loaded_engine_identity["loaded_path"],
            "launch_nonce": launch_nonce,
        },
        "steps": steps,
    }
    return response


def _validate_worker_response(
    response: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm: str,
    opponent: Mapping[str, Any],
) -> dict[str, Any]:
    if response.get("schema") != WORKER_RESPONSE_SCHEMA:
        raise RTPThreeArmRunnerError("worker response schema is invalid")
    if response.get("status") != "completed":
        raise RTPThreeArmRunnerError("worker arm did not complete; it cannot be scored")
    expected_simple = {
        "cell_id": cell.get("cell_id"),
        "arm": arm,
        "opponent_id": cell.get("opponent_id"),
        "candidate_seat": cell.get("candidate_seat"),
    }
    for key, expected in expected_simple.items():
        if response.get(key) != expected:
            raise RTPThreeArmRunnerError(f"worker response mismatch at {key}")
    terminal = response.get("terminal_outcome")
    if not isinstance(terminal, Mapping):
        raise RTPThreeArmRunnerError("worker response lacks terminal outcome")
    if terminal.get("termination") != "completed" or terminal.get("failed_seat") is not None:
        raise RTPThreeArmRunnerError("noncompleted terminal outcome cannot be scored")
    if any(terminal.get(key) is not None for key in ("engine_error", "candidate_error", "opponent_error")):
        raise RTPThreeArmRunnerError("terminal outcome contains an execution error")
    candidate_seat = int(cell["candidate_seat"])
    code = terminal.get("engine_result_code")
    winner = terminal.get("winner")
    expected_score = {
        candidate_seat: ("candidate", 1.0),
        1 - candidate_seat: ("opponent", 0.0),
        2: ("draw", 0.5),
    }.get(code)
    if expected_score is None or (winner, response.get("candidate_score")) != expected_score:
        raise RTPThreeArmRunnerError("terminal outcome and candidate score disagree")
    telemetry = response.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise RTPThreeArmRunnerError("worker response telemetry is invalid")
    if int(telemetry.get("candidate_forfeit_count", -1)) != int(bool(terminal.get("candidate_forfeit"))):
        raise RTPThreeArmRunnerError("candidate forfeit telemetry disagrees with terminal outcome")
    if int(telemetry.get("illegal_action_count", -1)) != 0:
        raise RTPThreeArmRunnerError("completed worker row records an illegal candidate action")
    isolation = response.get("isolation")
    if not isinstance(isolation, Mapping):
        raise RTPThreeArmRunnerError("worker response isolation is invalid")
    required = {
        "launch_mode": "subprocess_exec",
        "fresh_process_per_arm": True,
        "one_cell_one_arm": True,
        "pool_reuse": False,
        "forked_from_evaluator": False,
        "process_model_load": True,
        "fresh_candidate_agent": True,
        "candidate_reset_called": True,
        "fresh_opponent_module": True,
        "engine_restore_before_first_select": True,
        "engine_restore_count": 1,
        "battle_start_after_restore_count": 0,
        "no_remote_leaf_sampling_mcts": True,
        "complexity_probe_latency_excluded": True,
        "baseline_content_digest": opponent["content_digest"],
        "package_snapshot_verified_before_import": True,
        "baseline_package_root": opponent["package_root"],
        "baseline_tree_entries_sha256": opponent["payload"]["tree_entries_sha256"],
        "baseline_package_manifest_sha256": opponent["manifest"]["sha256"],
        "baseline_main_py_sha256": _opponent_package_entry_digest(opponent, "main.py"),
        "baseline_deck_sha256": _opponent_package_entry_digest(opponent, "deck.csv"),
        "candidate_factory_calls": 1,
        "opponent_factory_calls": 1,
    }
    for key, expected in required.items():
        if isolation.get(key) != expected:
            raise RTPThreeArmRunnerError(f"worker isolation fails {key}")
    for key in (
        "process_id",
        "launch_nonce",
        "candidate_rng_initial_state_sha256",
        "common_sanitized_environment_sha256",
        "arm_environment_sha256",
        "baseline_module_name",
        "source_snapshot_manifest_sha256",
        "source_tree_sha256",
        "evaluation_cg_closure_receipt_sha256",
        "evaluation_cg_engine_sha256",
        "evaluation_cg_engine_path",
        "evaluation_cg_closure_manifest_sha256",
        "evaluation_cg_metadata_parity_sha256",
        "candidate_runtime_contract_sha256",
        "engine_loaded_path",
    ):
        if not isinstance(isolation.get(key), str) or not isolation[key]:
            raise RTPThreeArmRunnerError(f"worker isolation lacks {key}")
    if not (
        isinstance(isolation.get("opponent_rng_initial_state_sha256"), str)
        or isolation.get("opponent_rng_deterministic_or_no_rng") is True
    ):
        raise RTPThreeArmRunnerError("worker isolation lacks opponent RNG identity")
    if isolation.get("engine_loaded_path") != isolation.get("evaluation_cg_engine_path"):
        raise RTPThreeArmRunnerError(
            "worker engine loaded path differs from the snapshot-local CG closure"
        )
    context_digest = isolation.get("evaluation_action_execution_sha256")
    if arm == "no_rtp":
        if context_digest is not None:
            raise RTPThreeArmRunnerError("no-RTP worker emitted evaluator action execution context")
        if isolation.get("action_fence_sha256") is not None:
            raise RTPThreeArmRunnerError("no-RTP worker emitted an evaluator action fence")
    elif not isinstance(context_digest, str) or not context_digest.startswith("sha256:"):
        raise RTPThreeArmRunnerError("RTP worker lacks evaluator action execution context")
    elif not isinstance(isolation.get("action_fence_sha256"), str) or not str(
        isolation["action_fence_sha256"]
    ).startswith("sha256:"):
        raise RTPThreeArmRunnerError("RTP worker lacks an evaluator action fence")
    return dict(response)


def _safe_component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    rendered = "".join(ch if ch in allowed else "_" for ch in value)
    return rendered[:160] or "item"


def _bounded_worker_output(
    value: str | bytes | _WorkerStreamCapture,
    *,
    limit_bytes: int = _FAILED_WORKER_CAPTURE_BYTES,
) -> dict[str, Any]:
    """Return bounded evidence for either a stream or a small local value."""

    if isinstance(value, _WorkerStreamCapture):
        return value.evidence()
    if isinstance(value, str):
        raw = value.encode("utf-8", errors="replace")
    elif isinstance(value, bytes):
        raw = value
    else:
        # Error text and test doubles are never trusted enough to retain an
        # arbitrary object representation.  The representation is local
        # parent text, so this bounded conversion remains diagnostic-only.
        raw = repr(value).encode("utf-8", errors="replace")
    capture = _WorkerStreamCapture(capture_limit_bytes=limit_bytes)
    capture.feed(raw)
    capture.finish()
    return capture.evidence()


def _worker_output_sha256(value: str | bytes | _WorkerStreamCapture) -> str:
    if isinstance(value, _WorkerStreamCapture):
        return value.sha256
    return _sha256_text(value)


def _worker_subprocess_argv(_request: Mapping[str, Any]) -> list[str]:
    """The only production executable used for an isolated arm response."""

    return [sys.executable, "-m", "poke_bot.rtp_three_arm_evaluation_runner", "--worker"]


def _capture_worker_process(
    *,
    request: Mapping[str, Any],
    environment: Mapping[str, str],
) -> _CapturedWorkerProcess:
    """Exec one child while concurrently draining raw stdout and stderr.

    ``Popen`` is intentionally used directly rather than a high-level capture
    helper: those helpers buffer complete streams before the parent can apply
    any cap.  The two pipe readers retain a fixed sample and hash/count all
    raw bytes as they arrive.  No output is spooled to temporary files.
    """

    stdout = _WorkerStreamCapture(parse_limit_bytes=_WORKER_RESPONSE_PARSE_MAX_BYTES)
    stderr = _WorkerStreamCapture()
    process: Any = None
    child_pid: int | None = None
    returncode: int | None = None
    try:
        request_bytes = json.dumps(dict(request), sort_keys=True).encode("utf-8")
        argv = _worker_subprocess_argv(request)
        cwd = str(
            _physical_path(
                Path(__file__).parent.parent,
                "runner source root",
                require_directory=True,
            )
        )
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env=dict(environment),
            cwd=cwd,
        )
        raw_pid = getattr(process, "pid", None)
        child_pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) else None
    except Exception as exc:
        stdout.finish()
        stderr.finish()
        raise _WorkerSubprocessBoundaryError(
            stage="worker_subprocess_launch_failed",
            cause=exc,
            stdout=stdout,
            stderr=stderr,
            returncode=None,
            child_pid=child_pid,
        ) from exc

    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def note_error(exc: BaseException) -> None:
        with error_lock:
            # A bounded collection prevents a malicious stream shim from
            # turning a diagnostic failure into another memory sink.
            if len(errors) < 3:
                errors.append(exc)

    def drain(stream: Any, capture: _WorkerStreamCapture) -> None:
        try:
            while True:
                block = stream.read(_WORKER_STREAM_READ_BYTES)
                if not block:
                    break
                capture.feed(block)
        except Exception as exc:
            note_error(exc)
        finally:
            try:
                stream.close()
            except Exception as exc:
                note_error(exc)

    def write_request(stream: Any) -> None:
        try:
            remaining = memoryview(request_bytes)
            while remaining:
                written = stream.write(remaining)
                if isinstance(written, bool) or not isinstance(written, int) or written < 1:
                    raise OSError("worker stdin accepted no request bytes")
                remaining = remaining[written:]
            stream.close()
        except Exception as exc:
            note_error(exc)
            try:
                stream.close()
            except Exception as close_exc:
                note_error(close_exc)

    stdout_thread = threading.Thread(
        target=drain, args=(process.stdout, stdout), daemon=True, name="rtp-worker-stdout"
    )
    stderr_thread = threading.Thread(
        target=drain, args=(process.stderr, stderr), daemon=True, name="rtp-worker-stderr"
    )
    stdin_thread = threading.Thread(
        target=write_request, args=(process.stdin,), daemon=True, name="rtp-worker-stdin"
    )

    def terminate_owned_worker_after_io_fault() -> None:
        """Break a pipe-error deadlock for this exact parent-owned child only."""

        try:
            if process.poll() is None:
                # This is the one Popen child created above, never an
                # interactive/user session.  Without this cleanup a failed
                # reader could leave the child blocked on a full other pipe.
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            # The bounded failed-worker record below retains the original I/O
            # cause; no broader or stronger process-tree action is attempted.
            return

    try:
        # Start readers first: an eager child cannot fill an output pipe while
        # the request writer is still handing it its bounded JSON request.
        stdout_thread.start()
        stderr_thread.start()
        stdin_thread.start()
        while True:
            waited = process.poll()
            if waited is not None:
                if isinstance(waited, int) and not isinstance(waited, bool):
                    returncode = waited
                    break
                raise RTPThreeArmRunnerError("worker process returned a non-integer status")
            with error_lock:
                first_io_error = errors[0] if errors else None
            if first_io_error is not None:
                terminate_owned_worker_after_io_fault()
                raise first_io_error
            time.sleep(0.005)
        stdin_thread.join()
        stdout_thread.join()
        stderr_thread.join()
        stdout.finish()
        stderr.finish()
        if errors:
            raise errors[0]
    except Exception as exc:
        # The caller receives a sealed diagnostic.  The only cleanup above is
        # a bounded termination of this exact parent-owned worker when a pipe
        # reader/writer fails; no interactive or unrelated process is touched.
        terminate_owned_worker_after_io_fault()
        for thread in (stdin_thread, stdout_thread, stderr_thread):
            if thread.is_alive():
                thread.join(timeout=0.1)
        stdout.finish()
        stderr.finish()
        observed_returncode = returncode
        if observed_returncode is None:
            raw_returncode = getattr(process, "returncode", None)
            if isinstance(raw_returncode, int) and not isinstance(raw_returncode, bool):
                observed_returncode = raw_returncode
        raise _WorkerSubprocessBoundaryError(
            stage="worker_subprocess_io_failed",
            cause=exc,
            stdout=stdout,
            stderr=stderr,
            returncode=observed_returncode,
            child_pid=child_pid,
        ) from exc
    assert returncode is not None
    return _CapturedWorkerProcess(
        returncode=returncode,
        child_pid=child_pid,
        stdout=stdout,
        stderr=stderr,
    )


def _claimed_failed_worker_file_identity(
    path: Any,
    sha256: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Record a configured identity without turning a failed child into trust.

    A malformed runtime/source path is itself useful failure evidence.  The
    parent therefore records the exact configured claim and independently
    records whether it could observe the claimed physical file.  It never
    substitutes an observed identity for the claim.
    """

    claim = {
        "path": path if isinstance(path, str) else None,
        "sha256": sha256 if isinstance(sha256, str) else None,
    }
    if claim["path"] is None:
        return {**claim, "observation": "not_configured", "observed": None}
    try:
        observed = _identity(claim["path"], label)
    except Exception:
        return {**claim, "observation": "unavailable", "observed": None}
    observation = (
        "observed_without_claimed_digest"
        if claim["sha256"] is None
        else "matched" if claim["sha256"] == observed["sha256"] else "mismatched"
    )
    return {
        **claim,
        "observation": observation,
        "observed": observed,
    }


def _finalize_failed_worker_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Bind immutable parent launch facts before they enter an error path."""

    material = {
        key: value
        for key, value in dict(context).items()
        if key != "failed_worker_context_sha256"
    }
    return {
        **material,
        "failed_worker_context_sha256": canonical_digest(material),
    }


def _failed_worker_evaluation(cell: Mapping[str, Any], arm: str) -> dict[str, Any]:
    return {
        "cell_id": cell.get("cell_id"),
        "evaluation_case_id": cell.get("evaluation_case_id"),
        "evaluation_case_bindings_sha256": cell.get(
            "evaluation_case_bindings_sha256"
        ),
        "opponent_id": cell.get("opponent_id"),
        "candidate_seat": cell.get("candidate_seat"),
        "arm": arm,
    }


def _pre_exec_failed_worker_context(
    *,
    manifest_identity: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    snapshot: Mapping[str, Any],
    authority: Mapping[str, Any],
    capability: Mapping[str, Any],
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Record only facts known before scratch/factory setup begins."""

    snapshot_seal = snapshot.get("seal")
    authority_identity = authority.get("identity")
    return _finalize_failed_worker_context(
        {
            "context_origin": "parent_before_worker_pre_exec_setup",
            "child_launched": False,
            "child_pid": None,
            "manifest_identity": dict(manifest_identity),
            "evaluation_authority_identity": (
                dict(authority_identity)
                if isinstance(authority_identity, Mapping)
                else None
            ),
            "snapshot_seal_identity": (
                dict(snapshot_seal) if isinstance(snapshot_seal, Mapping) else None
            ),
            "action_fence_identity": None,
            "evaluation": _failed_worker_evaluation(cell, arm),
            "frozen_cell": dict(cell),
            "factory": factory_ref,
            "scratch_directory": None,
            "worker_nonce": None,
            "worker_request": None,
            "worker_request_sha256": None,
            "runtime_identity": None,
            "source_snapshot_identity": None,
            "pairing_capability_identity": (
                dict(capability["receipt"])
                if isinstance(capability.get("receipt"), Mapping)
                else None
            ),
            "evaluation_cg_closure_identity": (
                dict(closure["receipt"])
                if isinstance(closure.get("receipt"), Mapping)
                else None
            ),
            "unavailable_fields": [
                "scratch_directory",
                "action_fence_identity",
                "worker_nonce",
                "worker_request",
                "worker_request_sha256",
                "runtime_identity",
                "source_snapshot_identity",
            ],
        }
    )


def _with_failed_worker_context_fields(
    context: Mapping[str, Any], **updates: Any
) -> dict[str, Any]:
    """Apply parent-observed state while keeping the context digest current."""

    updated = {**dict(context), **updates}
    unavailable = updated.get("unavailable_fields")
    if isinstance(unavailable, Sequence) and not isinstance(unavailable, (str, bytes)):
        known = {
            key
            for key, value in updates.items()
            if value is not None and key in {str(item) for item in unavailable}
        }
        if known:
            updated["unavailable_fields"] = [
                item for item in unavailable if item not in known
            ]
    return _finalize_failed_worker_context(updated)


def _failed_worker_context(
    *,
    manifest_identity: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    request: Mapping[str, Any],
    environment: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    capability: Mapping[str, Any],
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind parent-known launch identities for a possible failed child.

    This context is generated before ``Popen``.  A child cannot
    choose or alter it, which is important when the child emits malformed JSON
    or a forged response.
    """

    source_root = environment.get("POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT")
    source_root_value = source_root if isinstance(source_root, str) else None
    source_manifest_path = (
        None
        if source_root_value is None
        else str(Path(source_root_value) / R198_SOURCE_SNAPSHOT_MANIFEST)
    )
    source_manifest = _claimed_failed_worker_file_identity(
        source_manifest_path,
        None,
        label="failed-worker source snapshot manifest",
    )
    runtime_contract = _claimed_failed_worker_file_identity(
        environment.get("POKEBOT_R198_EVAL_RUNTIME_CONTRACT"),
        environment.get("POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256"),
        label="failed-worker candidate runtime contract",
    )
    request_payload = dict(request)
    return _finalize_failed_worker_context({
        "context_origin": "parent_before_subprocess_exec",
        "child_launched": False,
        "child_pid": None,
        "manifest_identity": dict(manifest_identity),
        "evaluation_authority_identity": (
            dict(request["evaluation_authority"])
            if isinstance(request.get("evaluation_authority"), Mapping)
            else None
        ),
        "snapshot_seal_identity": (
            dict(request["snapshot_seal"])
            if isinstance(request.get("snapshot_seal"), Mapping)
            else None
        ),
        "action_fence_identity": (
            dict(request["action_fence"])
            if isinstance(request.get("action_fence"), Mapping)
            else None
        ),
        "evaluation": _failed_worker_evaluation(cell, arm),
        "frozen_cell": dict(cell),
        "factory": factory_ref,
        "worker_nonce": request_payload.get("launch_nonce"),
        "worker_request": request_payload,
        "worker_request_sha256": canonical_digest(request_payload),
        "runtime_identity": {
            "environment_identity": dict(environment_identity),
            "candidate_runtime_contract": runtime_contract,
        },
        "source_snapshot_identity": {
            "root": source_root_value,
            "source_tree_sha256": environment.get("POKEBOT_R198_EVAL_SOURCE_TREE_SHA256"),
            "manifest": source_manifest,
        },
        "pairing_capability_identity": (
            dict(capability["receipt"])
            if isinstance(capability.get("receipt"), Mapping)
            else None
        ),
        "evaluation_cg_closure_identity": (
            dict(closure["receipt"])
            if isinstance(closure.get("receipt"), Mapping)
            else None
        ),
        "unavailable_fields": [],
    })


def _fallback_failed_worker_context(
    *,
    manifest_identity: Mapping[str, Any],
    authority: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve a validation failure from a legacy/injected worker callable.

    Production ``_spawn_worker`` always supplies the parent-generated context
    above.  This explicit fallback keeps focused tests and alternate callers
    diagnosable without pretending unavailable runtime/fence/source identities
    were verified.
    """

    request = {
        "schema": WORKER_REQUEST_SCHEMA,
        "manifest_path": manifest_identity.get("path"),
        "manifest_sha256": manifest_identity.get("sha256"),
        "factory": factory_ref,
        "cell": dict(cell),
        "arm": arm,
        "snapshot_seal": (
            dict(snapshot["seal"]) if isinstance(snapshot.get("seal"), Mapping) else None
        ),
        "evaluation_authority": (
            dict(authority["identity"])
            if isinstance(authority.get("identity"), Mapping)
            else None
        ),
        "action_fence": None,
        "launch_nonce": None,
        "max_steps": None,
        "environment_identity": None,
    }
    context = _failed_worker_context(
        manifest_identity=manifest_identity,
        factory_ref=factory_ref,
        cell=cell,
        arm=arm,
        request=request,
        environment={},
        environment_identity={},
        capability={},
        closure={},
    )
    return _finalize_failed_worker_context({
        **context,
        "context_origin": "parent_fallback_for_injected_or_legacy_worker_result",
        "action_fence_identity": None,
        "runtime_identity": None,
        "source_snapshot_identity": None,
        "unavailable_fields": [
            "action_fence_identity",
            "runtime_identity",
            "source_snapshot_identity",
        ],
    })


def _immutable_json_no_clobber(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    digest_key: str,
    label: str,
) -> Path:
    """Create a 0444 JSON artifact once, rejecting every target reuse.

    ``_immutable_json`` supports idempotent content-addressed receipts.  A
    failed worker diagnostic is different: reusing a path could erase the fact
    that an independently launched child failed, so this helper is strictly
    O_EXCL/no-clobber even when the bytes would happen to match.
    """

    target = _lexical_absolute_path(path, label)
    _ensure_physical_directory(target.parent, f"{label} parent")
    wanted = payload.get(digest_key)
    if not isinstance(wanted, str) or not wanted.startswith("sha256:"):
        raise RTPThreeArmRunnerError(f"{label} lacks {digest_key}")
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RTPThreeArmRunnerError(f"cannot inspect {label}: {target}") from exc
    else:
        # Rehash regular files and reject links/nonfiles through the physical
        # helper before reporting the forbidden reuse.
        _physical_path(target, f"existing {label}", require_file=True)
        raise RTPThreeArmRunnerError(f"{label} already exists; reuse is forbidden: {target}")

    encoded = json.dumps(dict(payload), sort_keys=True, indent=2) + "\n"
    temporary = target.parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    created_temporary = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        created_temporary = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            _physical_path(target, f"racing {label}", require_file=True)
            raise RTPThreeArmRunnerError(
                f"{label} appeared during create; reuse is forbidden: {target}"
            ) from exc
        created = _physical_path(target, label, require_file=True)
        if stat.S_IMODE(os.lstat(created).st_mode) != 0o444:
            raise RTPThreeArmRunnerError(f"{label} did not seal with mode 0444")
        return created
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)


def _verify_precreated_evaluation_input_contract(
    path: Path,
    *,
    manifest_identity: Mapping[str, Any],
    authority_identity: Mapping[str, Any],
) -> None:
    """Allow only the stage's sealed input contract in a fresh output root."""

    contract_path = _physical_path(
        path, "pre-created evaluation input contract", require_file=True
    )
    if stat.S_IMODE(os.lstat(contract_path).st_mode) != 0o444:
        raise RTPThreeArmRunnerError(
            "pre-created evaluation input contract must use immutable mode 0444"
        )
    contract = _json_object(contract_path, "pre-created evaluation input contract")
    if contract.get("schema") != "poke_bot.alakazam_rtp_r198_three_arm_evaluation_stage/v1":
        raise RTPThreeArmRunnerError("pre-created evaluation input contract schema is invalid")
    if contract.get("stage_kind") != "three_arm_true_rng_evaluation":
        raise RTPThreeArmRunnerError("pre-created evaluation input contract kind is invalid")
    inputs = contract.get("evaluation_inputs")
    if not isinstance(inputs, Mapping):
        raise RTPThreeArmRunnerError("pre-created evaluation input contract lacks inputs")
    if not _same_bound_identity(
        inputs.get("prepared_evaluator_manifest"), manifest_identity
    ):
        raise RTPThreeArmRunnerError("pre-created evaluation input contract manifest differs")
    if not _same_bound_identity(
        inputs.get("evaluation_only_authority"), authority_identity
    ):
        raise RTPThreeArmRunnerError("pre-created evaluation input contract authority differs")
    contract_authority = contract.get("authority")
    if not isinstance(contract_authority, Mapping):
        raise RTPThreeArmRunnerError("pre-created evaluation input contract lacks authority denial")
    for key in (
        "training_eligible",
        "replay_eligible",
        "serving_eligible",
        "action_authority_enabled",
        "selector_authority",
        "kaggle_submission_authorized",
        "promotion_authority",
        "self_promotion_allowed",
    ):
        if contract_authority.get(key) is not False:
            raise RTPThreeArmRunnerError(
                f"pre-created evaluation input contract authority is not denied at {key}"
            )


def _write_execution_attempt_sentinel(
    *,
    output: Path,
    manifest_identity: Mapping[str, Any],
    authority_identity: Mapping[str, Any],
    factory_ref: str,
    cells: Sequence[Mapping[str, Any]],
    max_workers: int,
    max_steps: int,
) -> Path:
    """Seal the one execution attempt before any scheduled arm can launch."""

    output_root = _ensure_physical_directory(output.parent, "evaluation output root")
    output_root = _physical_path(
        output_root, "evaluation output root", require_directory=True
    )
    try:
        entries = sorted(output_root.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        raise RTPThreeArmRunnerError("cannot enumerate evaluation output root") from exc
    for entry in entries:
        try:
            metadata = os.lstat(entry)
        except OSError as exc:
            raise RTPThreeArmRunnerError("cannot inspect evaluation output-root entry") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RTPThreeArmRunnerError("evaluation output root contains a symbolic link")
        if entry.name == "evaluation-input-contract.json":
            _verify_precreated_evaluation_input_contract(
                entry,
                manifest_identity=manifest_identity,
                authority_identity=authority_identity,
            )
            continue
        # A prior attempt sentinel, result, transcript, scratch directory,
        # receipt, or failed-worker evidence all reject the root here.  Do not
        # infer that a partial tree belongs to this invocation.
        raise RTPThreeArmRunnerError(
            "evaluation output root already contains an execution artifact: "
            f"{entry.name}"
        )
    jobs = len(cells) * 3
    payload = {
        "schema": EXECUTION_ATTEMPT_SCHEMA,
        "status": "started_evaluation_only_not_a_result",
        "evaluation_only": True,
        "not_an_evaluation_result": True,
        "not_a_result_row": True,
        "not_an_execution_receipt": True,
        "not_a_transcript": True,
        **_FAILED_WORKER_AUTHORITY_DENIALS,
        "manifest_identity": dict(manifest_identity),
        "evaluation_authority_identity": dict(authority_identity),
        "evaluation": {
            "manifest_sha256": manifest_identity.get("sha256"),
            "evaluation_authority_sha256": authority_identity.get("sha256"),
            "cell_count": len(cells),
        },
        "output": {"path": str(output), "root": str(output_root)},
        "factory": factory_ref,
        "schedule": {
            "cells_sha256": canonical_digest([dict(cell) for cell in cells]),
            "cell_count": len(cells),
            "arms": ["no_rtp", CANONICAL_DIRECT_ARM, "recursive_rtp"],
            "job_count": jobs,
            "max_workers": max_workers,
            "max_steps": max_steps,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    material = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "execution_attempt_input_sha256"}
    }
    payload["execution_attempt_input_sha256"] = canonical_digest(material)
    return _immutable_json_no_clobber(
        output_root / "execution-attempt.json",
        payload,
        digest_key="execution_attempt_input_sha256",
        label="execution attempt sentinel",
    )


def _same_bound_identity(actual: Any, expected: Any) -> bool:
    """Compare test/lightweight identities without weakening real triplets."""

    if actual is None or expected is None:
        return actual is expected
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    required = ("path", "sha256", "bytes")
    if all(key in expected for key in required):
        return _identity_equal(actual, expected)
    return dict(actual) == dict(expected)


def _validate_failed_worker_context(
    *,
    failure_context: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    authority_identity: Mapping[str, Any] | None,
    snapshot_seal_identity: Mapping[str, Any] | None,
    failure_stage: str,
    expected_worker_request: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reject mutable/tampered launch provenance before it can be named."""

    context = dict(failure_context)
    expected_context_sha256 = canonical_digest(
        {
            key: value
            for key, value in context.items()
            if key != "failed_worker_context_sha256"
        }
    )
    if context.get("failed_worker_context_sha256") != expected_context_sha256:
        raise RTPThreeArmRunnerError("failed-worker context digest differs")
    if not _same_bound_identity(context.get("manifest_identity"), manifest_identity):
        raise RTPThreeArmRunnerError("failed-worker context manifest identity differs")
    if context.get("evaluation") != _failed_worker_evaluation(cell, arm):
        raise RTPThreeArmRunnerError("failed-worker context evaluation/cell/arm differs")
    if context.get("frozen_cell") != dict(cell):
        raise RTPThreeArmRunnerError("failed-worker context frozen cell differs")
    if context.get("factory") != factory_ref:
        raise RTPThreeArmRunnerError("failed-worker context factory differs")
    if not _same_bound_identity(
        context.get("evaluation_authority_identity"), authority_identity
    ):
        raise RTPThreeArmRunnerError("failed-worker context authority identity differs")
    if not _same_bound_identity(
        context.get("snapshot_seal_identity"), snapshot_seal_identity
    ):
        raise RTPThreeArmRunnerError("failed-worker context snapshot identity differs")
    if not isinstance(context.get("child_launched"), bool):
        raise RTPThreeArmRunnerError("failed-worker context child launch state is invalid")
    child_pid = context.get("child_pid")
    if child_pid is not None and (
        isinstance(child_pid, bool) or not isinstance(child_pid, int)
    ):
        raise RTPThreeArmRunnerError("failed-worker context child PID is invalid")

    request = context.get("worker_request")
    if request is None:
        if failure_stage != "worker_pre_exec_setup_failed":
            raise RTPThreeArmRunnerError("failed-worker evidence lacks its worker request")
        if context.get("child_launched") is not False:
            raise RTPThreeArmRunnerError("pre-exec failed-worker context claims a child launch")
        if context.get("worker_request_sha256") is not None:
            raise RTPThreeArmRunnerError("pre-exec failed-worker request digest is fabricated")
        return None
    if not isinstance(request, Mapping):
        raise RTPThreeArmRunnerError("failed-worker evidence worker request is invalid")
    request_payload = dict(request)
    if expected_worker_request is not None and request_payload != dict(
        expected_worker_request
    ):
        raise RTPThreeArmRunnerError("failed-worker request differs from parent launch request")
    expected_request_sha256 = canonical_digest(request_payload)
    if context.get("worker_request_sha256") != expected_request_sha256:
        raise RTPThreeArmRunnerError("failed-worker evidence request digest differs")
    if request_payload.get("schema") != WORKER_REQUEST_SCHEMA:
        raise RTPThreeArmRunnerError("failed-worker request schema differs")
    if request_payload.get("manifest_path") != manifest_identity.get("path"):
        raise RTPThreeArmRunnerError("failed-worker request manifest path differs")
    if request_payload.get("manifest_sha256") != manifest_identity.get("sha256"):
        raise RTPThreeArmRunnerError("failed-worker request manifest digest differs")
    if request_payload.get("factory") != factory_ref:
        raise RTPThreeArmRunnerError("failed-worker request factory differs")
    if request_payload.get("cell") != dict(cell) or request_payload.get("arm") != arm:
        raise RTPThreeArmRunnerError("failed-worker request cell or arm differs")
    if not _same_bound_identity(
        request_payload.get("snapshot_seal"), context.get("snapshot_seal_identity")
    ):
        raise RTPThreeArmRunnerError("failed-worker request snapshot identity differs")
    if not _same_bound_identity(
        request_payload.get("evaluation_authority"),
        context.get("evaluation_authority_identity"),
    ):
        raise RTPThreeArmRunnerError("failed-worker request authority identity differs")
    if request_payload.get("action_fence") != context.get("action_fence_identity"):
        raise RTPThreeArmRunnerError("failed-worker request action fence differs")
    if request_payload.get("launch_nonce") != context.get("worker_nonce"):
        raise RTPThreeArmRunnerError("failed-worker request nonce differs")
    runtime = context.get("runtime_identity")
    if runtime is None:
        if request_payload.get("environment_identity") is not None:
            raise RTPThreeArmRunnerError("failed-worker fallback runtime identity differs")
    elif not isinstance(runtime, Mapping) or request_payload.get(
        "environment_identity"
    ) != runtime.get("environment_identity"):
        raise RTPThreeArmRunnerError("failed-worker request runtime identity differs")
    return request_payload


def _write_failed_worker_evidence(
    *,
    output_root: Path,
    manifest_identity: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    authority_identity: Mapping[str, Any] | None,
    snapshot_seal_identity: Mapping[str, Any] | None,
    failure_context: Mapping[str, Any],
    failure_stage: str,
    returncode: int | None,
    stdout: str | bytes | _WorkerStreamCapture,
    stderr: str | bytes | _WorkerStreamCapture,
    error: BaseException,
    expected_worker_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal non-result diagnostics before propagating a worker failure."""

    if returncode is not None and (
        isinstance(returncode, bool) or not isinstance(returncode, int)
    ):
        raise RTPThreeArmRunnerError("failed-worker evidence returncode must be an integer or null")
    request_payload = _validate_failed_worker_context(
        failure_context=failure_context,
        manifest_identity=manifest_identity,
        factory_ref=factory_ref,
        cell=cell,
        arm=arm,
        authority_identity=authority_identity,
        snapshot_seal_identity=snapshot_seal_identity,
        failure_stage=failure_stage,
        expected_worker_request=expected_worker_request,
    )
    request_sha256 = (
        None if request_payload is None else canonical_digest(request_payload)
    )
    launch_nonce = None if request_payload is None else request_payload.get("launch_nonce")
    nonce_component = (
        _safe_component(launch_nonce)
        if isinstance(launch_nonce, str) and launch_nonce
        else f"unavailable-{secrets.token_hex(16)}"
    )
    cell_component = _safe_component(str(cell.get("cell_id", "unknown-cell")))
    arm_component = _safe_component(arm)
    unavailable = failure_context.get("unavailable_fields")
    if not isinstance(unavailable, Sequence) or isinstance(unavailable, (str, bytes)):
        raise RTPThreeArmRunnerError("failed-worker context unavailable-field list is invalid")
    evidence = {
        "schema": FAILED_WORKER_EVIDENCE_SCHEMA,
        "status": "failed_closed_not_an_evaluation_result",
        "evaluation_only": True,
        "not_an_evaluation_result": True,
        "not_a_result_row": True,
        "not_an_execution_receipt": True,
        "not_a_transcript": True,
        **_FAILED_WORKER_AUTHORITY_DENIALS,
        "manifest_identity": dict(manifest_identity),
        "evaluation_authority_identity": failure_context.get(
            "evaluation_authority_identity"
        ),
        "snapshot_seal_identity": failure_context.get("snapshot_seal_identity"),
        "action_fence_identity": failure_context.get("action_fence_identity"),
        "evaluation": _failed_worker_evaluation(cell, arm),
        "frozen_cell": dict(cell),
        "factory": factory_ref,
        "scratch_directory": failure_context.get("scratch_directory"),
        "worker_nonce": failure_context.get("worker_nonce"),
        "worker_request": request_payload,
        "worker_request_sha256": request_sha256,
        "runtime_identity": failure_context.get("runtime_identity"),
        "source_snapshot_identity": failure_context.get("source_snapshot_identity"),
        "pairing_capability_identity": failure_context.get(
            "pairing_capability_identity"
        ),
        "evaluation_cg_closure_identity": failure_context.get(
            "evaluation_cg_closure_identity"
        ),
        "context_origin": failure_context.get("context_origin"),
        "failed_worker_context_sha256": failure_context.get(
            "failed_worker_context_sha256"
        ),
        "child_launched": failure_context.get("child_launched"),
        "child_pid": failure_context.get("child_pid"),
        "child": {
            "launched": failure_context.get("child_launched"),
            "pid": failure_context.get("child_pid"),
        },
        "unavailable_fields": list(unavailable),
        "returncode": returncode,
        "failure": {
            "stage": failure_stage,
            "exception_type": type(error).__name__,
            "message": _bounded_worker_output(str(error), limit_bytes=1024)[
                "captured_text"
            ],
        },
        "stdout": _bounded_worker_output(stdout),
        "stderr": _bounded_worker_output(stderr),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    material = {
        key: value
        for key, value in evidence.items()
        if key not in {"created_at_utc", "failed_worker_evidence_input_sha256"}
    }
    evidence["failed_worker_evidence_input_sha256"] = canonical_digest(material)
    path = _immutable_json_no_clobber(
        output_root
        / "failed-worker-evidence"
        / f"{cell_component}-{arm_component}-{nonce_component}.json",
        evidence,
        digest_key="failed_worker_evidence_input_sha256",
        label="failed-worker evidence",
    )
    identity = _identity(path, "failed-worker evidence")
    if stat.S_IMODE(os.lstat(path).st_mode) != 0o444:
        raise RTPThreeArmRunnerError("failed-worker evidence is not immutable mode 0444")
    return identity


def _failed_worker_error(
    *,
    message: str,
    cause: BaseException,
    output_root: Path,
    manifest_identity: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    authority_identity: Mapping[str, Any] | None,
    snapshot_seal_identity: Mapping[str, Any] | None,
    failure_context: Mapping[str, Any],
    failure_stage: str,
    returncode: int | None,
    stdout: str | bytes | _WorkerStreamCapture,
    stderr: str | bytes | _WorkerStreamCapture,
    expected_worker_request: Mapping[str, Any] | None = None,
) -> RTPThreeArmRunnerError:
    """Create an error only after its immutable failed-worker evidence exists."""

    try:
        identity = _write_failed_worker_evidence(
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=authority_identity,
            snapshot_seal_identity=snapshot_seal_identity,
            failure_context=failure_context,
            failure_stage=failure_stage,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            error=cause,
            expected_worker_request=expected_worker_request,
        )
    except Exception as evidence_error:
        raise RTPThreeArmRunnerError(
            f"{message}; unable to seal failed-worker evidence"
        ) from evidence_error
    safe_message = _bounded_worker_output(message, limit_bytes=1024)["captured_text"]
    error = RTPThreeArmRunnerError(
        f"{safe_message}; immutable failed-worker evidence: {identity['path']}"
    )
    # The scheduler uses this marker to avoid manufacturing a second evidence
    # file when a child failure has already been sealed inside _spawn_worker.
    setattr(error, "_failed_worker_evidence_path", identity["path"])
    return error


def _write_row_evidence(
    *,
    output_root: Path,
    manifest_identity: Mapping[str, Any],
    row: Mapping[str, Any],
    stdout: str | bytes | _WorkerStreamCapture,
    stderr: str | bytes | _WorkerStreamCapture,
    cohort_identity: Mapping[str, Any],
) -> dict[str, Any]:
    cell_id = _safe_component(str(row["cell_id"]))
    arm = _safe_component(str(row["arm"]))
    tag = f"{cell_id}-{arm}"
    transcript = {
        "schema": TRANSCRIPT_SCHEMA,
        "manifest_sha256": manifest_identity["sha256"],
        "cell_id": row["cell_id"],
        "arm": row["arm"],
        "native_events": row.get("native_transcript_events", []),
        "decision_diagnostics": row["telemetry"].get("decision_diagnostics", []),
        "decision_latency_trace": row["telemetry"].get("decision_latency_trace", []),
        # Keep the non-materializing over-cap evidence in the immutable
        # transcript as well as the execution receipt's telemetry digest.
        # A local promotion consumer can then reconcile the strict special
        # trace without trusting a summary reconstructed from worker output.
        "over_cap_factorized_fallback_trace": row["telemetry"].get(
            "over_cap_factorized_fallback_trace", []
        ),
        "worker_stdout_sha256": _worker_output_sha256(stdout),
        "worker_stderr_sha256": _worker_output_sha256(stderr),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    transcript_material = {key: value for key, value in transcript.items() if key not in {"created_at_utc", "transcript_input_sha256"}}
    transcript["transcript_input_sha256"] = canonical_digest(transcript_material)
    transcript_path = _immutable_json(
        output_root / "transcripts" / f"{tag}.json",
        transcript,
        digest_key="transcript_input_sha256",
    )
    transcript_identity = _identity(transcript_path, "execution transcript")
    terminal = dict(row["terminal_outcome"])
    execution = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "status": "completed",
        "manifest_sha256": manifest_identity["sha256"],
        "cell_id": row["cell_id"],
        "arm": row["arm"],
        "opponent_id": row["opponent_id"],
        "candidate_seat": row["candidate_seat"],
        "evaluation_case_id": row["evaluation_case_id"],
        "evaluation_case_bindings_sha256": row["evaluation_case_bindings_sha256"],
        "evaluation_corpus_sha256": cohort_identity["sha256"],
        "transcript_sha256": transcript_identity["sha256"],
        "runtime_identity_sha256": canonical_digest(row["runtime_identity"]),
        "rng_identity_sha256": canonical_digest(row["rng_identity"]),
        "telemetry_sha256": canonical_digest(row["telemetry"]),
        "terminal_outcome_sha256": canonical_digest(terminal),
        "candidate_score": row["candidate_score"],
        "termination": terminal["termination"],
        "failed_seat": terminal["failed_seat"],
        "engine_error": terminal["engine_error"],
        "candidate_error": terminal["candidate_error"],
        "opponent_error": terminal["opponent_error"],
        "complexity_probe_latency_excluded": True,
        **dict(row["isolation"]),
        "isolation": row["isolation"],
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    material = {key: value for key, value in execution.items() if key not in {"created_at_utc", "execution_input_sha256"}}
    execution["execution_input_sha256"] = canonical_digest(material)
    receipt_path = _immutable_json(
        output_root / "execution-receipts" / f"{tag}.json",
        execution,
        digest_key="execution_input_sha256",
    )
    receipt_identity = _identity(receipt_path, "execution receipt")
    return {
        **dict(row),
        "evaluation_corpus_sha256": cohort_identity["sha256"],
        "evaluation_case_bindings_sha256": row["evaluation_case_bindings_sha256"],
        "transcript": transcript_identity,
        "execution_receipt": receipt_identity,
    }


def _worker_scratch(output_root: Path, cell_id: str, arm: str) -> Path:
    base = output_root / "worker-scratch"
    _ensure_physical_directory(base, "worker scratch base")
    base = _physical_path(base, "worker scratch base", require_directory=True)
    prefix = _safe_component(f"{cell_id}-{arm}-")
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    return _physical_path(path, "worker scratch directory", require_directory=True)


def _spawn_worker(
    *,
    manifest_path: str,
    manifest_identity: Mapping[str, Any],
    factory_ref: str,
    cell: Mapping[str, Any],
    arm: str,
    snapshot: Mapping[str, Any],
    capability: Mapping[str, Any],
    closure: Mapping[str, Any],
    authority: Mapping[str, Any],
    output_root: Path,
    max_steps: int,
) -> tuple[dict[str, Any], _WorkerStreamCapture, _WorkerStreamCapture]:
    """Prepare and execute one fresh child, sealing every post-attempt fault."""

    failure_context = _pre_exec_failed_worker_context(
        manifest_identity=manifest_identity,
        factory_ref=factory_ref,
        cell=cell,
        arm=arm,
        snapshot=snapshot,
        authority=authority,
        capability=capability,
        closure=closure,
    )
    authority_identity = authority.get("identity")
    snapshot_seal_identity = snapshot.get("seal")
    if not isinstance(authority_identity, Mapping) or not isinstance(
        snapshot_seal_identity, Mapping
    ):
        # This is a parent setup failure, not a child result.  Preserve the
        # minimal context rather than fabricating a request around bad inputs.
        cause = RTPThreeArmRunnerError("worker authority or snapshot seal identity is invalid")
        empty_stdout = _WorkerStreamCapture(
            parse_limit_bytes=_WORKER_RESPONSE_PARSE_MAX_BYTES
        )
        empty_stderr = _WorkerStreamCapture()
        raise _failed_worker_error(
            message=str(cause),
            cause=cause,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=(
                dict(authority_identity) if isinstance(authority_identity, Mapping) else None
            ),
            snapshot_seal_identity=(
                dict(snapshot_seal_identity)
                if isinstance(snapshot_seal_identity, Mapping)
                else None
            ),
            failure_context=failure_context,
            failure_stage="worker_pre_exec_setup_failed",
            returncode=None,
            stdout=empty_stdout,
            stderr=empty_stderr,
        ) from cause

    request: dict[str, Any] | None = None
    try:
        scratch = _worker_scratch(output_root, str(cell["cell_id"]), arm)
        failure_context = _with_failed_worker_context_fields(
            failure_context, scratch_directory=str(scratch)
        )
        launch_nonce = secrets.token_hex(24)
        failure_context = _with_failed_worker_context_fields(
            failure_context, worker_nonce=launch_nonce
        )
        action_fence = (
            None
            if arm == "no_rtp"
            else _write_action_fence(
                scratch=scratch,
                manifest_identity=manifest_identity,
                authority=authority,
                manifest=_json_object(manifest_path, "evaluation manifest"),
                cell=cell,
                arm=arm,
                launch_nonce=launch_nonce,
            )
        )
        if action_fence is not None:
            fence_identity = action_fence.get("identity")
            if not isinstance(fence_identity, Mapping):
                raise RTPThreeArmRunnerError("worker action fence lacks an identity")
            failure_context = _with_failed_worker_context_fields(
                failure_context, action_fence_identity=dict(fence_identity)
            )
        factory = _load_factory(factory_ref)
        explicit_env = _factory_call(
            factory,
            "worker_environment",
            manifest=_json_object(manifest_path, "evaluation manifest"),
            cell=cell,
            arm=arm,
            scratch_dir=str(scratch),
        )
        if not isinstance(explicit_env, Mapping):
            raise RTPThreeArmRunnerError("factory worker_environment is invalid")
        environment, environment_identity = _sanitize_environment(
            explicit_env,
            scratch,
            capability=capability,
            closure=closure,
        )
        request = {
            "schema": WORKER_REQUEST_SCHEMA,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_identity["sha256"],
            "factory": factory_ref,
            "cell": dict(cell),
            "arm": arm,
            # A child receives the exact immutable seal identity, never a seed
            # or raw deserialization blob.
            "snapshot_seal": dict(snapshot_seal_identity),
            "evaluation_authority": dict(authority_identity),
            "action_fence": (
                None if action_fence is None else dict(action_fence["identity"])
            ),
            "launch_nonce": launch_nonce,
            "max_steps": max_steps,
            "environment_identity": environment_identity,
        }
        failure_context = _failed_worker_context(
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            request=request,
            environment=environment,
            environment_identity=environment_identity,
            capability=capability,
            closure=closure,
        )
        failure_context = _with_failed_worker_context_fields(
            failure_context, scratch_directory=str(scratch)
        )
    except Exception as exc:
        empty_stdout = _WorkerStreamCapture(
            parse_limit_bytes=_WORKER_RESPONSE_PARSE_MAX_BYTES
        )
        empty_stderr = _WorkerStreamCapture()
        raise _failed_worker_error(
            message="worker pre-exec setup failed",
            cause=exc,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage="worker_pre_exec_setup_failed",
            returncode=None,
            stdout=empty_stdout,
            stderr=empty_stderr,
            expected_worker_request=request,
        ) from exc

    assert request is not None
    try:
        completed = _capture_worker_process(
            request=request,
            environment=environment,
        )
    except _WorkerSubprocessBoundaryError as exc:
        failure_context = _with_failed_worker_context_fields(
            failure_context,
            child_launched=exc.child_pid is not None,
            child_pid=exc.child_pid,
        )
        raise _failed_worker_error(
            message="isolated worker subprocess launch/capture failed",
            cause=exc.cause,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage=exc.stage,
            returncode=exc.returncode,
            stdout=exc.stdout,
            stderr=exc.stderr,
            expected_worker_request=request,
        ) from exc

    failure_context = _with_failed_worker_context_fields(
        failure_context,
        child_launched=True,
        child_pid=completed.child_pid,
        worker_returncode=completed.returncode,
    )
    if completed.stdout.utf8_error or completed.stderr.utf8_error:
        cause = RTPThreeArmRunnerError("isolated worker emitted non-UTF-8 output")
        raise _failed_worker_error(
            message="isolated worker output is not strict UTF-8",
            cause=cause,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage="worker_output_decode_failed",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected_worker_request=request,
        ) from cause
    if completed.returncode != 0:
        stderr_summary = _bounded_worker_output(
            completed.stderr, limit_bytes=2048
        )["captured_text"]
        cause = RTPThreeArmRunnerError(
            f"isolated worker failed for {cell['cell_id']}/{arm}: {stderr_summary}"
        )
        raise _failed_worker_error(
            message=str(cause),
            cause=cause,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage="worker_exit_nonzero",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected_worker_request=request,
        ) from cause
    if completed.stdout.parse_limit_exceeded:
        cause = RTPThreeArmRunnerError("isolated worker stdout exceeds the response parse limit")
        raise _failed_worker_error(
            message=str(cause),
            cause=cause,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage="worker_stdout_parse_limit_exceeded",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected_worker_request=request,
        ) from cause
    try:
        response = json.loads(completed.stdout.parse_text())
    except Exception as exc:
        message = "isolated worker did not emit one JSON response"
        raise _failed_worker_error(
            message=message,
            cause=exc,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage="worker_stdout_invalid_json",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected_worker_request=request,
        ) from exc
    if not isinstance(response, Mapping):
        cause = RTPThreeArmRunnerError("isolated worker response is not an object")
        raise _failed_worker_error(
            message=str(cause),
            cause=cause,
            output_root=output_root,
            manifest_identity=manifest_identity,
            factory_ref=factory_ref,
            cell=cell,
            arm=arm,
            authority_identity=dict(authority_identity),
            snapshot_seal_identity=dict(snapshot_seal_identity),
            failure_context=failure_context,
            failure_stage="worker_response_not_object",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected_worker_request=request,
        ) from cause
    response_with_context = _ParentBoundWorkerResponse(
        response,
        failed_worker_context=failure_context,
    )
    return response_with_context, completed.stdout, completed.stderr


def _cross_arm_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    seen_processes: set[tuple[str, str, str]] = set()
    nonces: set[str] = set()
    for row in rows:
        isolation = row["isolation"]
        process_key = (
            str(isolation.get("boot_id", "")),
            str(isolation["process_id"]),
            str(isolation.get("process_start_ticks", "")),
        )
        if process_key in seen_processes:
            raise RTPThreeArmRunnerError("an isolated worker process was reused")
        seen_processes.add(process_key)
        nonce = str(isolation["launch_nonce"])
        if nonce in nonces:
            raise RTPThreeArmRunnerError("an isolated worker launch nonce was reused")
        nonces.add(nonce)
        by_cell.setdefault(str(row["cell_id"]), []).append(row)
    for cell_id, items in by_cell.items():
        if len(items) != 3 or {str(item["arm"]) for item in items} != {
            "no_rtp", CANONICAL_DIRECT_ARM, "recursive_rtp"
        }:
            raise RTPThreeArmRunnerError(f"cell {cell_id} does not have exactly three arms")
        for key in (
            "candidate_rng_initial_state_sha256",
            "python_rng_initial_state_sha256",
            "numpy_rng_initial_state_sha256",
            "torch_cpu_rng_initial_state_sha256",
            "torch_cuda_rng_initial_state_sha256",
            "common_sanitized_environment_sha256",
        ):
            values = {str(item["isolation"].get(key)) for item in items}
            if len(values) != 1:
                raise RTPThreeArmRunnerError(
                    f"cell {cell_id} does not reset identical {key} across arms"
                )
        opponent_values = {
            (
                item["isolation"].get("opponent_rng_initial_state_sha256"),
                item["isolation"].get("opponent_rng_deterministic_or_no_rng"),
            )
            for item in items
        }
        if len(opponent_values) != 1:
            raise RTPThreeArmRunnerError(
                f"cell {cell_id} opponent RNG state differs across arms"
            )


def run_three_arm_evaluation(
    *,
    manifest_path: str | Path,
    evaluation_authority_path: str | Path,
    factory: str,
    output_path: str | Path,
    max_workers: int = 1,
    max_steps: int = 4000,
) -> Path:
    """Run isolated arm children and emit immutable rows, not a promotion.

    The function intentionally returns only a result-file path.  A separate
    harness compiler may later turn that immutable file into a review/hold
    receipt; this function has no promotion, selector, or action authority.
    """

    if int(max_workers) < 1 or int(max_steps) < 1:
        raise RTPThreeArmRunnerError("max_workers and max_steps must be positive")
    manifest, manifest_identity = _read_manifest(manifest_path)
    capability = _verify_pairing_capability(manifest)
    closure = _verify_evaluation_cg_closure(manifest, capability)
    authority = _verify_evaluation_authority(
        evaluation_authority_path, manifest_identity=manifest_identity
    )
    cohort = _verify_cohort_and_source_exclusion(manifest)
    _verify_planner_preflight(manifest)
    opponents = _sealed_opponents(manifest)
    schedule = manifest.get("schedule")
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)) or not schedule:
        raise RTPThreeArmRunnerError("manifest schedule is empty")
    cells = [dict(cell) for cell in schedule if isinstance(cell, Mapping)]
    if len(cells) != len(schedule):
        raise RTPThreeArmRunnerError("manifest schedule contains an invalid cell")
    for cell in cells:
        _verify_snapshot_package(manifest, cell, cohort)
        if cell.get("opponent_id") not in opponents:
            raise RTPThreeArmRunnerError("scheduled cell names an unknown opponent")
        _require_physical_readonly_tree(
            next(entry for entry in manifest["opponents"] if entry["id"] == cell["opponent_id"])
        )
    output = _lexical_absolute_path(output_path, "evaluation result output")
    if output.exists():
        # This is a fresh, one-shot execution path.  Returning a file merely
        # because it has the expected schema would let an artifact created in
        # the interval before launch bypass all scheduled A/B/C children.
        # Reuse, if ever authorized, belongs to a separately verified sealed
        # stage receipt rather than this executor.
        _physical_path(output, "existing evaluation result output", require_file=True)
        raise RTPThreeArmRunnerError(
            "evaluation result output already exists; a fresh no-clobber path is required"
        )

    sealed_snapshots = _sealed_snapshot_materials(manifest, capability)
    snapshots: dict[str, dict[str, Any]] = {}
    for cell in cells:
        expected_rng = cell.get("rng_identity")
        if not isinstance(expected_rng, Mapping):
            raise RTPThreeArmRunnerError("scheduled cell lacks RNG identity")
        snapshot = sealed_snapshots.get(str(expected_rng.get("id", "")))
        if snapshot is None:
            raise RTPThreeArmRunnerError("scheduled cell references no sealed snapshot")
        for key in (
            "id",
            "kind",
            "sha256",
            "bytes",
            "seal_sha256",
            "capture_boundary",
            "boundary_tag",
        ):
            if snapshot.get(key) != expected_rng.get(key):
                raise RTPThreeArmRunnerError("scheduled cell snapshot identity differs from sealed artifact")
        _verify_snapshot_for_cell(
            snapshot=snapshot,
            cell=cell,
            manifest=manifest,
            opponent=opponents[str(cell["opponent_id"])],
            cohort=cohort,
        )
        snapshots[str(cell["cell_id"])] = snapshot

    authority_identity = authority.get("identity")
    if not isinstance(authority_identity, Mapping):
        raise RTPThreeArmRunnerError("evaluation authority lacks an immutable identity")
    # This is deliberately after every read-only manifest/cell/snapshot check
    # and before constructing/submitting even the first worker job.
    attempt = _write_execution_attempt_sentinel(
        output=output,
        manifest_identity=manifest_identity,
        authority_identity=authority_identity,
        factory_ref=factory,
        cells=cells,
        max_workers=int(max_workers),
        max_steps=int(max_steps),
    )
    output_root = attempt.parent

    jobs: list[tuple[dict[str, Any], str]] = [
        (cell, arm)
        for cell in cells
        for arm in ("no_rtp", CANONICAL_DIRECT_ARM, "recursive_rtp")
    ]
    complete_rows: list[dict[str, Any]] = []
    worker_limit = int(max_workers)
    job_iterator = iter(jobs)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit)
    futures: dict[
        concurrent.futures.Future[
            tuple[dict[str, Any], _WorkerStreamCapture, _WorkerStreamCapture]
        ],
        tuple[dict[str, Any], str],
    ] = {}

    def submit_next() -> bool:
        try:
            cell, arm = next(job_iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _spawn_worker,
            manifest_path=str(manifest_identity["path"]),
            manifest_identity=manifest_identity,
            factory_ref=factory,
            cell=cell,
            arm=arm,
            snapshot=snapshots[str(cell["cell_id"])],
            capability=capability,
            closure=closure,
            authority=authority,
            output_root=output_root,
            max_steps=int(max_steps),
        )
        futures[future] = (cell, arm)
        return True

    try:
        for _ in range(min(worker_limit, len(jobs))):
            submit_next()
        while futures:
            done, _pending = concurrent.futures.wait(
                tuple(futures),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            verified_batch: list[
                tuple[
                    dict[str, Any],
                    str,
                    dict[str, Any],
                    _WorkerStreamCapture,
                    _WorkerStreamCapture,
                ]
            ] = []
            batch_failure: BaseException | None = None
            for future in done:
                cell, arm = futures.pop(future)
                try:
                    response, stdout, stderr = future.result()
                except Exception as exc:
                    # Child exit/invalid-JSON paths seal their evidence inside
                    # _spawn_worker.  Keep the first failure for fail-fast
                    # propagation while still inspecting every already-done
                    # future in this bounded wave.
                    if batch_failure is None:
                        batch_failure = exc
                    continue
                failure_context: Mapping[str, Any] | None = None
                if isinstance(response, Mapping):
                    raw_context = getattr(response, "failed_worker_context", None)
                    response = dict(response)
                    if isinstance(raw_context, Mapping):
                        failure_context = dict(raw_context)
                if failure_context is None:
                    failure_context = _fallback_failed_worker_context(
                        manifest_identity=manifest_identity,
                        authority=authority,
                        factory_ref=factory,
                        cell=cell,
                        arm=arm,
                        snapshot=snapshots[str(cell["cell_id"])],
                    )
                worker_returncode = failure_context.get("worker_returncode", 0)
                if isinstance(worker_returncode, bool) or not isinstance(
                    worker_returncode, int
                ):
                    worker_returncode = 0
                try:
                    verified = _validate_worker_response(
                        response,
                        manifest=manifest,
                        cell=cell,
                        arm=arm,
                        opponent=opponents[str(cell["opponent_id"])],
                    )
                except Exception as exc:
                    stage = (
                        "worker_response_schema_invalid"
                        if isinstance(response, Mapping)
                        and response.get("schema") != WORKER_RESPONSE_SCHEMA
                        else "worker_response_validation_failed"
                    )
                    failed = _failed_worker_error(
                        message=str(exc),
                        cause=exc,
                        output_root=output_root,
                        manifest_identity=manifest_identity,
                        factory_ref=factory,
                        cell=cell,
                        arm=arm,
                        authority_identity=authority_identity,
                        snapshot_seal_identity=snapshots[str(cell["cell_id"])]["seal"],
                        failure_context=failure_context,
                        failure_stage=stage,
                        returncode=worker_returncode,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    if batch_failure is None:
                        batch_failure = failed
                    continue
                verified_batch.append((cell, arm, verified, stdout, stderr))
            if batch_failure is not None:
                raise batch_failure
            for cell, arm, verified, stdout, stderr in verified_batch:
                row = {
                    "cell_id": verified["cell_id"],
                    "arm": verified["arm"],
                    "opponent_id": verified["opponent_id"],
                    "candidate_seat": verified["candidate_seat"],
                    "evaluation_case_id": cell["evaluation_case_id"],
                    "evaluation_case_bindings_sha256": cell[
                        "evaluation_case_bindings_sha256"
                    ],
                    "completed": True,
                    "invalid": False,
                    "error": None,
                    "candidate_score": verified["candidate_score"],
                    "terminal_outcome": verified["terminal_outcome"],
                    "runtime_identity": verified["runtime_identity"],
                    "rng_identity": {
                        "id": snapshots[str(cell["cell_id"])]["id"],
                        "kind": "snapshot",
                        "sha256": snapshots[str(cell["cell_id"])]["sha256"],
                        "bytes": snapshots[str(cell["cell_id"])]["bytes"],
                        "seal_sha256": snapshots[str(cell["cell_id"])]["seal_sha256"],
                        "capture_boundary": snapshots[str(cell["cell_id"])][
                            "capture_boundary"
                        ],
                        "boundary_tag": snapshots[str(cell["cell_id"])]["boundary_tag"],
                        "restored_or_replayed": True,
                    },
                    "telemetry": verified["telemetry"],
                    "isolation": verified["isolation"],
                }
                complete_rows.append(
                    _write_row_evidence(
                        output_root=output_root,
                        manifest_identity=manifest_identity,
                        row=row,
                        stdout=stdout,
                        stderr=stderr,
                        cohort_identity=cohort["cohort"],
                    )
                )
            for _ in done:
                if not submit_next():
                    break
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=True)
    _cross_arm_isolation(complete_rows)
    complete_rows.sort(key=lambda row: (str(row["cell_id"]), str(row["arm"])))
    result = {
        "schema": RESULTS_SCHEMA,
        "status": "completed_evaluation_only",
        "manifest": manifest_identity,
        "evaluation_only_cohort": cohort["cohort"],
        "source_exclusion_proof": cohort["source_exclusion_proof"],
        "rows": complete_rows,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "self_promotion_performed": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    material = {key: value for key, value in result.items() if key not in {"created_at_utc", "results_input_sha256"}}
    result["results_input_sha256"] = canonical_digest(material)
    return _immutable_json(output, result, digest_key="results_input_sha256")


def _worker_main() -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
        if not isinstance(request, Mapping) or request.get("schema") != WORKER_REQUEST_SCHEMA:
            raise RTPThreeArmRunnerError("invalid worker request")
        # Agent/baseline diagnostics must not corrupt the single JSON response
        # channel.  Their text stays on stderr and is checksum-bound into the
        # parent-written transcript.
        with contextlib.redirect_stdout(sys.stderr):
            response = _worker_response(request=request)
        sys.stdout.write(json.dumps(response, sort_keys=True))
        return 0
    except BaseException as exc:  # parent treats any nonzero worker as invalid
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="run exactly one isolated arm request")
    parser.add_argument("--manifest")
    parser.add_argument("--evaluation-authority")
    parser.add_argument("--factory")
    parser.add_argument("--output")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main()
    missing = [
        name
        for name, value in (
            ("--manifest", args.manifest),
            ("--evaluation-authority", args.evaluation_authority),
            ("--factory", args.factory),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        parser.error("missing " + ", ".join(missing))
    try:
        output = run_three_arm_evaluation(
            manifest_path=args.manifest,
            evaluation_authority_path=args.evaluation_authority,
            factory=args.factory,
            output_path=args.output,
            max_workers=args.max_workers,
            max_steps=args.max_steps,
        )
    except RTPThreeArmRunnerError as exc:
        sys.stderr.write(f"RTP three-arm evaluation failed closed: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(_identity(output, "evaluation results"), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_CONTROL_ENV",
    "AUTHORITY_SCHEMA",
    "ArmRuntime",
    "CANONICAL_DIRECT_ARM",
    "EXECUTION_RECEIPT_SCHEMA",
    "FAILED_WORKER_EVIDENCE_SCHEMA",
    "RESULTS_SCHEMA",
    "RTPThreeArmEvaluationFactory",
    "RTPThreeArmRunnerError",
    "WORKER_REQUEST_SCHEMA",
    "WORKER_RESPONSE_SCHEMA",
    "run_three_arm_evaluation",
]
