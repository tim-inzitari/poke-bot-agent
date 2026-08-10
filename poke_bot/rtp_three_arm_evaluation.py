"""Fail-closed, receipt-producing RTP three-arm evaluation harness.

This module deliberately does not launch an engine, change a selector, or
promote a checkpoint.  It prepares an immutable evaluation schedule and turns
completed engine rows into an auditable receipt.  The three arms separate:

``no_rtp``
    Existing factorized-greedy behavior with the RTP bridge absent.
``direct_bridge_recursive_disabled``
    The bridge/full-combination policy path, with recursive planning disabled.
``recursive_rtp``
    The same bridge and sidecar, with recursive planning allowed.

Requested seeds are retained only as debugging metadata.  A cell is paired
only when every arm records the same checksum-bound *actual* RNG tape or
restorable simulator snapshot and attests that it was replayed/restored.

The r198 comparison is intentionally narrower still: both bridge arms must use
the exact ``pure_rl_r197`` profile with exactly 256 neural passes and an action
combination cap of 1,024.  Normal recursive plans must still consume six
passes and forced replans five.  Any future profile/budget study needs a
distinct authorization and receipt; the absolute ceiling remains 256 passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "poke_bot.recursive_turn_planner.three_arm_evaluation_manifest/v2"
RECEIPT_SCHEMA = "poke_bot.recursive_turn_planner.three_arm_evaluation_receipt/v2"
EXECUTION_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_execution_receipt/v1"
)
EVALUATION_ONLY_COHORT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1"
)
EVALUATION_PACKAGE_TREE_SCHEMA = (
    "poke_bot.recursive_turn_planner.evaluation_package_tree_snapshot/v1"
)
PAIRING_SNAPSHOT_SEAL_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_snapshot_seal/v1"
)
PAIRING_CASE_BINDING_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_pairing_case_binding/v1"
)
LATENCY_SLO_SCHEMA = "poke_bot.recursive_turn_planner.r198_recursive_latency_slo/v1"
CANDIDATE_EVALUATION_BINDING_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_candidate_evaluation_binding/v1"
)
# The typed r197 contract names the middle arm in full.  ``direct_bridge`` is
# accepted only as a legacy input alias; manifests and receipts always publish
# the canonical name below so downstream promotion reviewers cannot compare a
# differently named fourth arm by mistake.
DIRECT_BRIDGE_ARM = "direct_bridge_recursive_disabled"
LEGACY_DIRECT_BRIDGE_ARM = "direct_bridge"
ARMS = ("no_rtp", DIRECT_BRIDGE_ARM, "recursive_rtp")
ARM_ALIASES = {LEGACY_DIRECT_BRIDGE_ARM: DIRECT_BRIDGE_ARM}
# r198 is deliberately narrower than the legacy harness: every cell must be
# restored from a native, sealed snapshot.  A replay tape (or a requested
# seed) cannot make the required full-state pairing claim.
RNG_KINDS = frozenset({"snapshot"})
REQUIRED_SHARED_ARTIFACTS = (
    "parent_checkpoint",
    "deck",
    "matchup_tree",
    # The efficacy cohort is separate from the candidate's supervised heldout
    # batches.  Its source-exclusion proof binds it back to the candidate's
    # completed r197 selection plan without relabeling calibration data as an
    # A/B/C gameplay result.
    "evaluation_only_cohort",
    "r197_completion_receipt",
    "planner_preflight_receipt",
    "research_control_registry",
)
R198_SIZING_PROFILE = "pure_rl_r197"
R198_MAX_NEURAL_PASSES = 256
R198_MAX_ACTION_COMBOS = 1024
R198_NORMAL_RECURSIVE_PLAN_PASSES = 6
R198_FORCED_REPLAN_PASSES = 5
ABSOLUTE_MAX_NEURAL_PASSES = 256
PAIRING_CAPABILITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_capability/v2"
)
PAIRING_PROBE_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_probe/v1"
PAIRING_EVAL_CG_CLOSURE_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1"
)
R197_COMPLETION_SCHEMA = "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
R197_SELECTION_SCHEMA = "poke_bot.recursive_turn_planner.r197_training_selection_plan/v1"
R197_WHOLE_EPISODE_SELECTION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_whole_episode_selection/v1"
)
R197_EVALUATION_ONLY_SOURCE_EXCLUSION_SCHEMA = (
    "poke_bot.recursive_turn_planner.r197_evaluation_only_source_exclusion/v1"
)
PLANNER_PASS_PREFLIGHT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_planner_pass_preflight/v1"
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
R198_SIDECAR_CONFIG_SHA256 = (
    "sha256:7fb0658f0358c93636524a40ddd52f9f76199de261963a85dbf5946901a9f676"
)
R198_DECK_FILE_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R198_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R198_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R198_MATCHUP_ADAPTER_REGISTRY_RELATIVE = Path("state/matchup_adapter_roster.json")
R198_MATCHUP_ADAPTER_REGISTRY_SHA256 = (
    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
)
R198_MATCHUP_ADAPTER_REGISTRY_BYTES = 11_899
R198_MATCHUP_ADAPTER_REGISTRY_MODE = 0o444
R198_MATCHUP_ADAPTER_SLOT_REGISTRY_DIGEST = (
    "sha256:444c42c1235c19d3d95b10e80a12a84f35c9fb803967096736446eac1a5e225a"
)
MATCHUP_ADAPTER_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_roster/v1"
MATCHUP_ADAPTER_SLOT_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_slot_registry/v1"
OFFICIAL_CONTROL_OPPONENT_COUNT = 4
OFFICIAL_CONTROL_SEATS = 2
OFFICIAL_CONTROL_REPLICATES = 125
OFFICIAL_CONTROL_PAIRED_CELLS = (
    OFFICIAL_CONTROL_OPPONENT_COUNT
    * OFFICIAL_CONTROL_SEATS
    * OFFICIAL_CONTROL_REPLICATES
)
MINIMUM_RECURSIVE_DECISIONS = 100
MINIMUM_RECURSIVE_INTENDED_COMPLEX_SHARE = 0.05
MAXIMUM_UNEXPECTED_RECURSIVE_FALLBACK_RATE = 0.01
MINIMUM_CONFIDENCE_LEVEL = 0.90
INTENDED_COMPLEX_DECISION_SCOPE = "new_turn_complexity_gate_only"
FORCED_TURN_ORDER_CONTROL = "forced_go_first_contract"
SUCCESSFUL_RECURSIVE_MODES = frozenset(
    {"recursive_plan", "continue_plan", "replan_with_program"}
)
RECURSIVE_FALLBACK_MODES = frozenset({"direct_policy_fallback", "replan_direct"})
NONRECURSIVE_DIRECT_POLICY_MODE = "direct_policy"
OVER_CAP_FACTORIZED_FALLBACK_MODE = "over_cap_factorized_fallback"
OVER_CAP_FACTORIZED_FALLBACK_REASON = "complete_ordered_action_space_over_cap"
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
_OVER_CAP_TRACE_FIELDS = frozenset(
    {
        "decision_index",
        "arm",
        "mode",
        "classification",
        "action_space",
        "action_space_sha256",
        "observation_sha256",
        "candidate_policy_input_sha256",
        "logical_pre_action_sha256",
        "returned_action",
        "factorized_teacher_forcing_legal",
        "factorized_teacher_forcing_stage_count",
        "complexity_probe_not_invoked",
        "neural_passes",
        "required_neural_passes",
        "neural_budget_failure",
        "rtp_diagnostic",
        "included_in_candidate_decisions",
        "included_in_candidate_latency",
        "excluded_from_planner_eligible_candidate_decisions",
        "excluded_from_intended_complex_denominator",
        "excluded_from_direct_bridge_metrics",
        "excluded_from_recursive_metrics",
        "excluded_from_fallback_metrics",
        "excluded_from_neural_pass_metrics",
        "excluded_from_recursive_latency",
    }
)
R198_RESEARCH_CONTROL_REGISTRY_SHA256 = (
    "sha256:78fd8e52df1464db94e74a49247a67ced41b5d164dc86fafec3229f2c1e47edc"
)
R198_RESEARCH_CONTROL_REGISTRY_BYTES = 2117
R198_OFFICIAL_CONTROL_OPPONENTS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}
# This is part of the snapshot-seal contract, not a presentation preference.
# The input materializer captures cells in frozen registry order; changing to
# lexical map order after capture would attach a valid seal to the wrong cell.
R198_OFFICIAL_CONTROL_ORDER = tuple(R198_OFFICIAL_CONTROL_OPPONENTS)

DEFAULT_GATES: dict[str, Any] = {
    # A production-sized panel is four opponents × two seats × 125 tapes.
    "minimum_paired_cells": OFFICIAL_CONTROL_PAIRED_CELLS,
    "minimum_pairs_per_opponent_seat": OFFICIAL_CONTROL_REPLICATES,
    "minimum_direct_bridge_decisions": 1,
    "minimum_recursive_decisions": MINIMUM_RECURSIVE_DECISIONS,
    "minimum_recursive_share_of_intended_complex_decisions": MINIMUM_RECURSIVE_INTENDED_COMPLEX_SHARE,
    "maximum_unexpected_recursive_fallback_rate": MAXIMUM_UNEXPECTED_RECURSIVE_FALLBACK_RATE,
    "minimum_recursive_delta_lower_bound": 0.0,
    "confidence_level": MINIMUM_CONFIDENCE_LEVEL,
    # This has no default on purpose: a production run must state the SLO it
    # is claiming to satisfy.  An omitted value is not a latency gate.
    "maximum_p95_latency_seconds": None,
}


class RTPThreeArmEvaluationError(ValueError):
    """Raised when evidence cannot truthfully support a three-arm receipt."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lexical_absolute_path(value: str | Path, label: str) -> Path:
    """Return an absolute path without resolving or traversing symlinks.

    Evidence identities must name the physical artifact that was hashed.  A
    normal ``Path.resolve`` silently dereferences a mutable symlink, allowing a
    verifier and a later child process to disagree about what was evaluated.
    Reject ``..`` as well: resolving it can cross a symlink component before
    normalising the path.
    """

    raw = Path(_nonempty_text(value, label)).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if ".." in raw.parts:
        raise RTPThreeArmEvaluationError(f"{label} may not contain '..'")
    return Path(os.path.abspath(os.fspath(raw)))


def _physical_existing_path(
    value: str | Path,
    label: str,
    *,
    require_directory: bool = False,
    require_file: bool = False,
) -> Path:
    """Require an existing physical path with no symlink component."""

    path = _lexical_absolute_path(value, label)
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise RTPThreeArmEvaluationError(f"{label} does not exist: {path}") from exc
        if stat.S_ISLNK(mode):
            raise RTPThreeArmEvaluationError(
                f"{label} must not contain a symbolic-link component: {current}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise RTPThreeArmEvaluationError(
                f"{label} has a non-directory path component: {current}"
            )
    final_mode = os.lstat(path).st_mode
    if require_directory and not stat.S_ISDIR(final_mode):
        raise RTPThreeArmEvaluationError(f"{label} is not a directory: {path}")
    if require_file and not stat.S_ISREG(final_mode):
        raise RTPThreeArmEvaluationError(f"{label} is not a regular file: {path}")
    return path


def _ensure_physical_directory(path: str | Path, label: str) -> Path:
    """Create an output directory component-by-component without symlinks."""

    directory = _lexical_absolute_path(path, label)
    current = Path(directory.anchor)
    for part in directory.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            except OSError as exc:
                raise RTPThreeArmEvaluationError(
                    f"cannot create physical output directory {current}"
                ) from exc
            try:
                mode = os.lstat(current).st_mode
            except OSError as exc:
                raise RTPThreeArmEvaluationError(
                    f"cannot inspect physical output directory {current}"
                ) from exc
        except OSError as exc:
            raise RTPThreeArmEvaluationError(
                f"cannot inspect physical output directory {current}"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RTPThreeArmEvaluationError(
                f"{label} contains a symlink or non-directory component: {current}"
            )
    return directory


def file_digest(path: str | Path) -> str:
    source = _physical_existing_path(path, "digest source", require_file=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RTPThreeArmEvaluationError(f"{label} must be a JSON object")
    return dict(value)


def _nonempty_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RTPThreeArmEvaluationError(f"{label} is required")
    return text


def _canonical_arm_name(value: Any, label: str) -> str:
    """Resolve the one compatibility alias without emitting it downstream."""

    arm = _nonempty_text(value, label)
    return ARM_ALIASES.get(arm, arm)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RTPThreeArmEvaluationError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RTPThreeArmEvaluationError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise RTPThreeArmEvaluationError(f"{label} must be at least {minimum}")
    return parsed


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise RTPThreeArmEvaluationError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RTPThreeArmEvaluationError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise RTPThreeArmEvaluationError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise RTPThreeArmEvaluationError(f"{label} must be at least {minimum}")
    return parsed


def _frozen_identity(raw: Any, label: str) -> dict[str, Any]:
    """Resolve and verify a caller-supplied immutable file identity."""

    value = _mapping(raw, label)
    path_text = _nonempty_text(value.get("path"), f"{label}.path")
    expected = _nonempty_text(value.get("sha256"), f"{label}.sha256")
    if not expected.startswith("sha256:") or len(expected) != 71:
        raise RTPThreeArmEvaluationError(f"{label}.sha256 is not a SHA-256 digest")
    path = _physical_existing_path(path_text, label, require_file=True)
    actual = file_digest(path)
    if actual != expected:
        raise RTPThreeArmEvaluationError(
            f"{label} checksum mismatch: expected {expected}, got {actual}"
        )
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _verify_frozen_identity(raw: Any, label: str) -> dict[str, Any]:
    """Verify a normalized identity again immediately before compilation."""

    return _frozen_identity(raw, label)


def _verify_immutable_frozen_identity(raw: Any, label: str) -> dict[str, Any]:
    """Verify a frozen input is no longer writable by any file class.

    Result files, per-cell execution receipts, and transcripts are evidence,
    not scratch buffers.  A digest alone would be re-checkable, but a writable
    input could still be replaced between a worker's result and a review.
    """

    identity = _verify_frozen_identity(raw, label)
    mode = Path(identity["path"]).stat().st_mode
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RTPThreeArmEvaluationError(f"{label} must be immutable (not writable)")
    return identity


def _verify_mode_0444(identity: Mapping[str, Any], label: str) -> None:
    """The native restore wrapper only accepts exactly world-readable 0444 seals."""

    path = _physical_existing_path(identity.get("path"), label, require_file=True)
    if stat.S_IMODE(os.lstat(path).st_mode) != 0o444:
        raise RTPThreeArmEvaluationError(f"{label} must have immutable mode 0444")


def _canonical_matchup_adapter_registry_digest(payload: Mapping[str, Any]) -> str:
    """Match the V6 digest serialized into the r195 adapter-bank contract."""

    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _snapshot_local_matchup_adapter_registry(
    production_factory: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash the exact roster that the production factory will load.

    The raw roster bytes and the canonical slot-registry digest are separate
    bindings.  Preserving only the latter would lose the source-snapshot path
    and mode guarantee; preserving only the former would not cross-bind the
    parent model's physical adapter-slot layout.
    """

    factory = _mapping(production_factory, "production_factory")
    source_root = _physical_existing_path(
        factory.get("source_snapshot_root"),
        "production_factory.source_snapshot_root",
        require_directory=True,
    )
    raw = _mapping(
        factory.get("matchup_adapter_registry"),
        "production_factory.matchup_adapter_registry",
    )
    expected_keys = {"path", "sha256", "bytes", "mode"}
    if set(raw) != expected_keys:
        raise RTPThreeArmEvaluationError(
            "production_factory.matchup_adapter_registry must contain exactly "
            "path, sha256, bytes, and mode"
        )
    declared_mode = _integer(
        raw.get("mode"), "production_factory.matchup_adapter_registry.mode", minimum=0
    )
    if declared_mode != R198_MATCHUP_ADAPTER_REGISTRY_MODE:
        raise RTPThreeArmEvaluationError(
            "production_factory.matchup_adapter_registry must declare mode 0444"
        )
    observed = _verify_immutable_frozen_identity(
        raw, "production_factory.matchup_adapter_registry"
    )
    _verify_mode_0444(observed, "production_factory.matchup_adapter_registry")
    registry_path = Path(observed["path"])
    expected_path = source_root / R198_MATCHUP_ADAPTER_REGISTRY_RELATIVE
    if registry_path != expected_path:
        raise RTPThreeArmEvaluationError(
            "production_factory.matchup_adapter_registry is not the exact "
            "snapshot-local state/matchup_adapter_roster.json"
        )
    if (
        observed["sha256"] != R198_MATCHUP_ADAPTER_REGISTRY_SHA256
        or observed["bytes"] != R198_MATCHUP_ADAPTER_REGISTRY_BYTES
    ):
        raise RTPThreeArmEvaluationError(
            "production_factory.matchup_adapter_registry does not bind the exact r198 roster bytes"
        )
    payload = _read_json_object(registry_path, "snapshot-local matchup adapter registry")
    if (
        payload.get("schema") != MATCHUP_ADAPTER_REGISTRY_SCHEMA
        or payload.get("slot_schema") != MATCHUP_ADAPTER_SLOT_REGISTRY_SCHEMA
        or _integer(payload.get("slot_capacity"), "snapshot-local roster slot_capacity")
        != 64
    ):
        raise RTPThreeArmEvaluationError(
            "snapshot-local matchup adapter registry is not the exact V6 slot registry"
        )
    if _canonical_matchup_adapter_registry_digest(payload) != (
        R198_MATCHUP_ADAPTER_SLOT_REGISTRY_DIGEST
    ):
        raise RTPThreeArmEvaluationError(
            "snapshot-local matchup adapter registry canonical slot digest changed"
        )
    return {**observed, "mode": R198_MATCHUP_ADAPTER_REGISTRY_MODE}


def _observed_file_identity(path: str | Path, label: str) -> dict[str, Any]:
    source = _physical_existing_path(path, label, require_file=True)
    return {
        "path": str(source),
        "sha256": file_digest(source),
        "bytes": source.stat().st_size,
    }


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = _physical_existing_path(path, label, require_file=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTPThreeArmEvaluationError(f"cannot read {label}: {source}") from exc
    return _mapping(value, label)


def _immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    existing_digest_key: str,
) -> Path:
    """Create a receipt once, or safely reuse the same logical receipt."""

    target = _lexical_absolute_path(path, "immutable receipt output")
    _ensure_physical_directory(target.parent, "immutable receipt output parent")
    if os.path.lexists(target):
        _physical_existing_path(target, "immutable receipt output", require_file=True)
    wanted = _nonempty_text(payload.get(existing_digest_key), existing_digest_key)
    if target.exists():
        existing = _read_json_object(target, "existing immutable receipt")
        if existing.get(existing_digest_key) == wanted:
            return target
        raise RTPThreeArmEvaluationError(
            f"immutable receipt already exists with a different {existing_digest_key}: "
            f"{target}"
        )
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _read_json_object(target, "existing immutable receipt")
            if existing.get(existing_digest_key) != wanted:
                raise RTPThreeArmEvaluationError(
                    f"immutable receipt appeared with a different "
                    f"{existing_digest_key}: {target}"
                )
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _normalize_shared_artifacts(raw: Any) -> dict[str, dict[str, Any]]:
    value = _mapping(raw, "shared_artifacts")
    missing = sorted(set(REQUIRED_SHARED_ARTIFACTS) - set(value))
    if missing:
        raise RTPThreeArmEvaluationError(
            "shared_artifacts missing required identities: " + ", ".join(missing)
        )
    return {
        name: _frozen_identity(value[name], f"shared_artifacts.{name}")
        for name in sorted(value)
    }


def _sha256_value(value: Any, label: str) -> str:
    digest = _nonempty_text(value, label).lower()
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise RTPThreeArmEvaluationError(f"{label} is not a SHA-256 digest")
    try:
        int(digest.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise RTPThreeArmEvaluationError(f"{label} is not a SHA-256 digest") from exc
    return digest


def _r197_selection_side(
    selection: Mapping[str, Any],
    side: str,
) -> tuple[tuple[str, ...], str]:
    row = _mapping(selection.get(side), f"r197 selection.{side}")
    candidate = _mapping(
        row.get("candidate_selection"), f"r197 selection.{side}.candidate_selection"
    )
    if candidate.get("schema") != R197_WHOLE_EPISODE_SELECTION_SCHEMA:
        raise RTPThreeArmEvaluationError(
            f"r197 selection.{side} has an unrecognized candidate selection schema"
        )
    if candidate.get("split") != side:
        raise RTPThreeArmEvaluationError(
            f"r197 selection.{side} is not bound to the {side} split"
        )
    capped = _mapping(
        row.get("batch_cap_selection"), f"r197 selection.{side}.batch_cap_selection"
    )
    retained = row.get("retained_episode_ids")
    if not isinstance(retained, Sequence) or isinstance(retained, (str, bytes)):
        raise RTPThreeArmEvaluationError(
            f"r197 selection.{side}.retained_episode_ids must be a list"
        )
    identities = tuple(
        _nonempty_text(item, f"r197 {side} retained episode") for item in retained
    )
    if not identities or len(set(identities)) != len(identities):
        raise RTPThreeArmEvaluationError(
            f"r197 selection.{side} must retain unique non-empty episodes"
        )
    if _integer(
        capped.get("retained_episode_count"),
        f"r197 selection.{side}.retained_episode_count",
        minimum=1,
    ) != len(identities):
        raise RTPThreeArmEvaluationError(
            f"r197 selection.{side} retained episode count differs from its IDs"
        )
    if capped.get("row_level_sampling") is not False or capped.get(
        "cross_window_dynamics_target"
    ) is not False:
        raise RTPThreeArmEvaluationError(
            f"r197 selection.{side} permits prohibited sampling or targets"
        )
    # Preserve receipt order.  The source-exclusion computation hashes the
    # original r197 selection lists with its own canonical-JSON convention;
    # sorting here would let a different retained-list receipt masquerade as
    # the completed candidate's selection.
    return identities, _sha256_value(
        capped.get("retained_episode_ids_sha256"),
        f"r197 selection.{side}.retained_episode_ids_sha256",
    )


def _r197_episode_ids_digest(value: Sequence[str]) -> str:
    """Match the completed r197 selection planner's canonical JSON digest."""

    encoded = (
        json.dumps(list(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalize_source_exclusion_computation(
    raw: Any,
    *,
    train_episode_ids: Sequence[str],
    heldout_episode_ids: Sequence[str],
    evaluation_case_bindings: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Recompute, rather than trust, the evaluation-only source exclusion.

    The r198 cohort uses its own synthetic source-ID namespace.  A mere
    ``overlap_count: 0`` statement does not establish that those IDs were ever
    compared against the exact completed r197 selections, so this validates
    the materializer's deterministic set-intersection record end to end.
    """

    value = _mapping(raw, f"{label}.source_exclusion_computation")
    evaluation_case_source_ids = sorted(
        "r198-evaluation-only-case/v1/"
        + canonical_digest(
            {
                "case_id": _nonempty_text(case.get("case_id"), "evaluation case case_id"),
                "opponent_id": _nonempty_text(
                    case.get("opponent_id"), "evaluation case opponent_id"
                ),
                "content_digest": _sha256_value(
                    case.get("content_digest"), "evaluation case content_digest"
                ),
                "candidate_seat": _integer(
                    case.get("candidate_seat"), "evaluation case candidate_seat"
                ),
                "replicate": _integer(case.get("replicate"), "evaluation case replicate"),
                "domain": "r198-official-control-evaluation-only",
            }
        )[7:]
        for case in evaluation_case_bindings
    )
    if (
        len(evaluation_case_source_ids) != OFFICIAL_CONTROL_PAIRED_CELLS
        or len(set(evaluation_case_source_ids)) != OFFICIAL_CONTROL_PAIRED_CELLS
    ):
        raise RTPThreeArmEvaluationError(
            "evaluation cohort does not yield exactly one synthetic source identity per cell"
        )
    union_ids = sorted(set(train_episode_ids).union(heldout_episode_ids))
    intersection_ids = sorted(set(union_ids).intersection(evaluation_case_source_ids))
    expected = {
        "method": "exact_source_id_set_intersection",
        "evaluation_case_source_kind": "r198_official_control_synthetic_case_identity_v1",
        "r197_train_episode_ids_sha256": _r197_episode_ids_digest(train_episode_ids),
        "r197_train_episode_count": len(train_episode_ids),
        "r197_heldout_episode_ids_sha256": _r197_episode_ids_digest(heldout_episode_ids),
        "r197_heldout_episode_count": len(heldout_episode_ids),
        "r197_union_episode_ids_sha256": canonical_digest(union_ids),
        "r197_union_episode_count": len(union_ids),
        "evaluation_case_source_ids_sha256": canonical_digest(evaluation_case_source_ids),
        "evaluation_case_source_count": len(evaluation_case_source_ids),
        "intersection_episode_ids_sha256": canonical_digest(intersection_ids),
        "intersection_episode_count": len(intersection_ids),
    }
    if set(value) != set(expected):
        raise RTPThreeArmEvaluationError(
            f"{label}.source_exclusion_computation has unexpected or missing fields"
        )
    for key, wanted in expected.items():
        observed = value.get(key)
        if key.endswith("_sha256"):
            observed = _sha256_value(observed, f"{label}.source_exclusion_computation.{key}")
        elif key.endswith("_count"):
            observed = _integer(
                observed, f"{label}.source_exclusion_computation.{key}", minimum=0
            )
        elif key in {"method", "evaluation_case_source_kind"}:
            observed = _nonempty_text(
                observed, f"{label}.source_exclusion_computation.{key}"
            )
        if observed != wanted:
            raise RTPThreeArmEvaluationError(
                f"{label}.source_exclusion_computation mismatch at {key}"
            )
    if expected["intersection_episode_count"] != 0:
        raise RTPThreeArmEvaluationError(
            f"{label}.source_exclusion_computation proves source overlap"
        )
    return expected


def _normalize_evaluation_cohort_cases(
    payload: Mapping[str, Any],
    *,
    expected_panel: Mapping[str, str],
    label: str,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[dict[str, Any]], str]:
    """Bind every scheduled cell to a separately frozen evaluation case.

    The r197 supervised heldout selection is deliberately *not* reused here.
    These cases are an independently frozen, evaluation-only cohort.  Their
    source-exclusion proof may cite r197 provenance, but no scheduler may turn
    a bare cohort hash into evidence that a particular game came from it.
    """

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise RTPThreeArmEvaluationError(f"{label}.cases must be a list")
    expected_keys = {
        (opponent_id, candidate_seat, replicate)
        for opponent_id in expected_panel
        for candidate_seat in range(OFFICIAL_CONTROL_SEATS)
        for replicate in range(OFFICIAL_CONTROL_REPLICATES)
    }
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _mapping(raw_case, f"{label}.cases[{index}]")
        case_id = _nonempty_text(case.get("case_id"), f"{label}.cases[{index}].case_id")
        if case_id in case_ids:
            raise RTPThreeArmEvaluationError(f"{label}.cases repeats case_id {case_id!r}")
        case_ids.add(case_id)
        opponent_id = _nonempty_text(
            case.get("opponent_id"), f"{label}.cases[{index}].opponent_id"
        )
        content_digest = _sha256_value(
            case.get("content_digest"), f"{label}.cases[{index}].content_digest"
        )
        if expected_panel.get(opponent_id) != content_digest:
            raise RTPThreeArmEvaluationError(
                f"{label}.cases[{index}] is not an exact official control case"
            )
        candidate_seat = _integer(
            case.get("candidate_seat"), f"{label}.cases[{index}].candidate_seat"
        )
        if candidate_seat not in {0, 1}:
            raise RTPThreeArmEvaluationError(f"{label}.cases[{index}] has invalid candidate_seat")
        replicate = _integer(case.get("replicate"), f"{label}.cases[{index}].replicate")
        if replicate >= OFFICIAL_CONTROL_REPLICATES:
            raise RTPThreeArmEvaluationError(
                f"{label}.cases[{index}] has invalid official-control replicate"
            )
        for key in ("evaluation_only",):
            if case.get(key) is not True:
                raise RTPThreeArmEvaluationError(f"{label}.cases[{index}] does not pass {key}")
        for key in ("training_eligible", "replay_eligible"):
            if case.get(key) is not False:
                raise RTPThreeArmEvaluationError(
                    f"{label}.cases[{index}] unexpectedly permits {key}"
                )
        key = (opponent_id, candidate_seat, replicate)
        if key in by_key:
            raise RTPThreeArmEvaluationError(f"{label}.cases repeats scheduled cell {key}")
        by_key[key] = {
            "case_id": case_id,
            "opponent_id": opponent_id,
            "content_digest": content_digest,
            "candidate_seat": candidate_seat,
            "replicate": replicate,
        }
    if set(by_key) != expected_keys or len(by_key) != OFFICIAL_CONTROL_PAIRED_CELLS:
        raise RTPThreeArmEvaluationError(
            f"{label}.cases must provide the exact 4×2×125 official-control panel"
        )
    bindings = sorted(by_key.values(), key=lambda row: row["case_id"])
    digest = canonical_digest(bindings)
    if _sha256_value(payload.get("case_bindings_sha256"), f"{label}.case_bindings_sha256") != digest:
        raise RTPThreeArmEvaluationError(f"{label}.case_bindings_sha256 mismatch")
    return by_key, bindings, digest


def _normalize_r197_source_exclusion_binding(
    shared: Mapping[str, Mapping[str, Any]],
    raw_source_exclusion_proof: Any,
    *,
    opponent_content_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Bind a separate official-control cohort to r197 provenance.

    r197's 7,996 supervised heldout batches are calibration evidence, not
    recursive-versus-direct gameplay efficacy.  This requires a separate
    evaluation-only cohort and a proof that it has no source overlap with the
    candidate's retained train or heldout selections.
    """

    completion = _verify_frozen_identity(
        shared.get("r197_completion_receipt"),
        "shared_artifacts.r197_completion_receipt",
    )
    cohort = _verify_immutable_frozen_identity(
        shared.get("evaluation_only_cohort"),
        "shared_artifacts.evaluation_only_cohort",
    )
    receipt = _read_json_object(completion["path"], "r197 completion receipt")
    if receipt.get("schema") != R197_COMPLETION_SCHEMA:
        raise RTPThreeArmEvaluationError("r197 completion receipt schema is invalid")
    if receipt.get("status") != "completed_shadow_only":
        raise RTPThreeArmEvaluationError("r197 completion receipt is not completed shadow-only")
    if _sha256_value(
        receipt.get("candidate_contract_sha256"), "r197 candidate_contract_sha256"
    ) != R198_CANDIDATE_CONTRACT_SHA256:
        raise RTPThreeArmEvaluationError(
            "r197 completion receipt is not the exact r198 candidate contract"
        )
    authority = _mapping(receipt.get("authority"), "r197 completion authority")
    for key in (
        "serving_eligible",
        "action_authority_enabled",
        "selector_authority",
        "live_checkpoint_publication",
        "submission_eligible",
    ):
        if authority.get(key) is not False:
            raise RTPThreeArmEvaluationError(
                f"r197 completion receipt unexpectedly grants {key}"
            )
    if authority.get("shadow_only") is not True:
        raise RTPThreeArmEvaluationError("r197 completion receipt is not shadow-only")

    contract = _mapping(receipt.get("contract"), "r197 completion contract")
    corpus = _mapping(
        contract.get("complete_action_corpus"), "r197 completion complete_action_corpus"
    )
    if corpus.get("schema") != "poke_bot.rtp_complete_action_shadow_corpus/v1":
        raise RTPThreeArmEvaluationError("r197 completion corpus schema is invalid")
    split = _mapping(corpus.get("split"), "r197 completion corpus split")
    if (
        split.get("source_disjoint") is not True
        or split.get("unit") != "episode_id"
        or _integer(split.get("seed"), "r197 completion split seed", minimum=0)
        != 5_000_000
    ):
        raise RTPThreeArmEvaluationError(
            "r197 completion receipt does not prove the required source-disjoint split"
        )
    selection = _mapping(corpus.get("selection"), "r197 completion selection")
    if selection.get("schema") != R197_SELECTION_SCHEMA:
        raise RTPThreeArmEvaluationError("r197 completion selection schema is invalid")
    if selection.get("row_level_sampling") is not False or selection.get(
        "cross_window_dynamics_target"
    ) is not False:
        raise RTPThreeArmEvaluationError(
            "r197 completion selection permits prohibited sampling or targets"
        )
    train_episode_ids, train_digest = _r197_selection_side(selection, "train")
    heldout_episode_ids, heldout_digest = _r197_selection_side(selection, "heldout")
    train_ids = set(train_episode_ids)
    heldout_ids = set(heldout_episode_ids)
    if train_ids.intersection(heldout_ids):
        raise RTPThreeArmEvaluationError(
            "r197 completion selection leaks heldout episodes into training"
        )
    if _sha256_value(
        selection.get("train_selection_sha256"), "r197 train_selection_sha256"
    ) != train_digest or _sha256_value(
        selection.get("heldout_selection_sha256"), "r197 heldout_selection_sha256"
    ) != heldout_digest:
        raise RTPThreeArmEvaluationError(
            "r197 completion selection digest does not bind its retained episode IDs"
        )
    training = _mapping(receipt.get("training"), "r197 completion training")
    if training.get("heldout_is_source_excluded") is not True:
        raise RTPThreeArmEvaluationError(
            "r197 completion receipt does not attest source-excluded heldout data"
        )
    target_wiring = _mapping(
        training.get("candidate_target_wiring"), "r197 completion candidate_target_wiring"
    )
    if target_wiring.get("status") != "masked_absent_no_fabrication":
        raise RTPThreeArmEvaluationError(
            "r197 completion candidate target status is not the recorded masked/absent status"
        )
    for key, expected in (
        ("latent_lookahead_targets", "not_wired_future_input"),
        ("unobserved_action_returns", "not_fabricated"),
        ("value_of_planning_target", "not_heuristic_labeled"),
    ):
        if target_wiring.get(key) != expected:
            raise RTPThreeArmEvaluationError(
                f"r197 completion candidate target wiring changed at {key}"
            )
    heldout_metrics = _mapping(
        _mapping(training.get("metrics"), "r197 completion metrics").get("rtp_heldout"),
        "r197 completion heldout metrics",
    )
    for key in (
        "mean_candidate_calibration_target_count",
        "mean_candidate_ranking_pair_count",
        "mean_candidate_return_target_count",
    ):
        if _number(heldout_metrics.get(key), f"r197 completion {key}", minimum=0.0) != 0.0:
            raise RTPThreeArmEvaluationError(
                f"r197 completion unexpectedly reports trusted counterfactual targets at {key}"
            )
    cohort_payload = _read_json_object(cohort["path"], "evaluation-only cohort")
    if cohort_payload.get("schema") != EVALUATION_ONLY_COHORT_SCHEMA or cohort_payload.get(
        "status"
    ) != "frozen":
        raise RTPThreeArmEvaluationError("evaluation-only cohort schema/status is invalid")
    for key in ("evaluation_only",):
        if cohort_payload.get(key) is not True:
            raise RTPThreeArmEvaluationError(f"evaluation-only cohort does not pass {key}")
    for key in ("training_eligible", "replay_eligible"):
        if cohort_payload.get(key) is not False:
            raise RTPThreeArmEvaluationError(
                f"evaluation-only cohort unexpectedly permits {key}"
            )
    cohort_source_identity = _sha256_value(
        cohort_payload.get("source_identity_sha256"),
        "evaluation-only cohort source_identity_sha256",
    )
    cohort_rows = cohort_payload.get("registry_rows")
    if not isinstance(cohort_rows, Sequence) or isinstance(cohort_rows, (str, bytes)):
        raise RTPThreeArmEvaluationError("evaluation-only cohort registry_rows must be a list")
    cohort_rows_by_id: dict[str, str] = {}
    for index, raw_row in enumerate(cohort_rows):
        row = _mapping(raw_row, f"evaluation-only cohort.registry_rows[{index}]")
        row_id = _nonempty_text(row.get("id"), f"evaluation-only cohort.registry_rows[{index}].id")
        if row.get("training_eligible") is not False:
            raise RTPThreeArmEvaluationError(
                "evaluation-only cohort contains a training-eligible registry row"
            )
        content_digest = _sha256_value(
            row.get("content_digest"),
            f"evaluation-only cohort.registry_rows[{index}].content_digest",
        )
        if row_id in cohort_rows_by_id:
            raise RTPThreeArmEvaluationError("evaluation-only cohort repeats a registry row")
        cohort_rows_by_id[row_id] = content_digest
    if cohort_rows_by_id != dict(opponent_content_digests):
        raise RTPThreeArmEvaluationError(
            "evaluation-only cohort must bind exactly the four scheduled registry rows/digests"
        )
    _, evaluation_case_bindings, evaluation_case_bindings_sha256 = (
        _normalize_evaluation_cohort_cases(
            cohort_payload,
            expected_panel=opponent_content_digests,
            label="evaluation-only cohort",
        )
    )
    cohort_source_exclusion_computation = _normalize_source_exclusion_computation(
        cohort_payload.get("source_exclusion_computation"),
        train_episode_ids=train_episode_ids,
        heldout_episode_ids=heldout_episode_ids,
        evaluation_case_bindings=evaluation_case_bindings,
        label="evaluation-only cohort",
    )

    proof_input = _mapping(raw_source_exclusion_proof, "source_exclusion_proof")
    proof = _verify_immutable_frozen_identity(
        proof_input.get("receipt"), "source_exclusion_proof.receipt"
    )
    proof_payload = _read_json_object(proof["path"], "source exclusion proof")
    if proof_payload.get("schema") != R197_EVALUATION_ONLY_SOURCE_EXCLUSION_SCHEMA or proof_payload.get(
        "status"
    ) != "verified":
        raise RTPThreeArmEvaluationError("source exclusion proof schema/status is invalid")
    if proof_payload.get("evaluation_only") is not True:
        raise RTPThreeArmEvaluationError("source exclusion proof is not evaluation-only")
    for key in ("training_eligible", "replay_eligible", "all_registry_rows_training_eligible"):
        if proof_payload.get(key) is not False:
            raise RTPThreeArmEvaluationError(
                f"source exclusion proof unexpectedly permits {key}"
            )
    expected_proof_digests = {
        "r197_completion_receipt_sha256": completion["sha256"],
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "r197_corpus_manifest_sha256": _sha256_value(
            corpus.get("manifest_sha256"), "r197 corpus manifest_sha256"
        ),
        "r197_corpus_receipt_sha256": _sha256_value(
            corpus.get("receipt_sha256"), "r197 corpus receipt_sha256"
        ),
        "r197_selection_plan_sha256": _sha256_value(
            selection.get("selection_plan_sha256"), "r197 selection_plan_sha256"
        ),
        "r197_train_selection_sha256": train_digest,
        "r197_heldout_selection_sha256": heldout_digest,
        "evaluation_only_cohort_sha256": cohort["sha256"],
        "source_identity_sha256": cohort_source_identity,
        "evaluation_case_bindings_sha256": evaluation_case_bindings_sha256,
    }
    for key, expected in expected_proof_digests.items():
        if _sha256_value(proof_payload.get(key), f"source exclusion proof {key}") != expected:
            raise RTPThreeArmEvaluationError(
                f"source exclusion proof mismatch at {key}"
            )
    if _integer(
        proof_payload.get("evaluation_only_cohort_bytes"),
        "source exclusion proof evaluation_only_cohort_bytes",
        minimum=1,
    ) != cohort["bytes"]:
        raise RTPThreeArmEvaluationError("source exclusion proof cohort byte count mismatch")
    if _integer(
        proof_payload.get("source_identity_overlap_count"),
        "source exclusion proof source_identity_overlap_count",
        minimum=0,
    ) != 0:
        raise RTPThreeArmEvaluationError(
            "source exclusion proof reports overlap with r197 candidate sources"
        )
    proof_rows = proof_payload.get("registry_rows")
    if not isinstance(proof_rows, Sequence) or isinstance(proof_rows, (str, bytes)):
        raise RTPThreeArmEvaluationError("source exclusion proof registry_rows must be a list")
    proof_rows_by_id: dict[str, str] = {}
    for index, raw_row in enumerate(proof_rows):
        row = _mapping(raw_row, f"source exclusion proof.registry_rows[{index}]")
        row_id = _nonempty_text(row.get("id"), f"source exclusion proof.registry_rows[{index}].id")
        if row.get("training_eligible") is not False:
            raise RTPThreeArmEvaluationError(
                "source exclusion proof contains a training-eligible registry row"
            )
        content_digest = _sha256_value(
            row.get("content_digest"),
            f"source exclusion proof.registry_rows[{index}].content_digest",
        )
        if row_id in proof_rows_by_id:
            raise RTPThreeArmEvaluationError("source exclusion proof repeats a registry row")
        proof_rows_by_id[row_id] = content_digest
    if proof_rows_by_id != dict(opponent_content_digests):
        raise RTPThreeArmEvaluationError(
            "source exclusion proof does not bind exactly the four scheduled rows/digests"
        )
    if proof_payload.get("r197_supervised_heldout_calibration_only") is not True:
        raise RTPThreeArmEvaluationError(
            "source exclusion proof must state that r197 supervised heldout is calibration only"
        )
    proof_source_exclusion_computation = _normalize_source_exclusion_computation(
        proof_payload.get("source_exclusion_computation"),
        train_episode_ids=train_episode_ids,
        heldout_episode_ids=heldout_episode_ids,
        evaluation_case_bindings=evaluation_case_bindings,
        label="source exclusion proof",
    )
    if canonical_digest(proof_source_exclusion_computation) != canonical_digest(
        cohort_source_exclusion_computation
    ):
        raise RTPThreeArmEvaluationError(
            "source exclusion proof and frozen cohort disagree on source-intersection computation"
        )
    return {
        "completion_receipt": completion,
        "evaluation_only_cohort": cohort,
        "source_exclusion_proof": proof,
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "corpus_manifest_sha256": _sha256_value(
            corpus.get("manifest_sha256"), "r197 corpus manifest_sha256"
        ),
        "corpus_receipt_sha256": _sha256_value(
            corpus.get("receipt_sha256"), "r197 corpus receipt_sha256"
        ),
        "selection_plan_sha256": _sha256_value(
            selection.get("selection_plan_sha256"), "r197 selection_plan_sha256"
        ),
        "train_selection_sha256": train_digest,
        "heldout_selection_sha256": heldout_digest,
        "r197_source_disjoint": True,
        "r197_heldout_episode_count": len(heldout_episode_ids),
        "evaluation_only": True,
        "source_identity_overlap_count": 0,
        "evaluation_cohort_source_identity_sha256": cohort_source_identity,
        "evaluation_case_bindings": evaluation_case_bindings,
        "evaluation_case_bindings_sha256": evaluation_case_bindings_sha256,
        "source_exclusion_computation": proof_source_exclusion_computation,
        "candidate_target_status": "masked_absent_no_fabrication",
        "trusted_counterfactual_candidate_targets_available": False,
    }


def _case_bindings_with_cell_ids(
    bindings: Sequence[Mapping[str, Any]],
    opponents: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Assign the one canonical cell ID before its snapshot is captured.

    Native snapshot seals need to name their cell.  The assignment must happen
    before snapshot creation, not after the runner has seen an unbound blob.
    It mirrors the manifest's exact official-panel order.
    """

    raw_by_key = {
        (
            _nonempty_text(row.get("opponent_id"), "evaluation case opponent_id"),
            _integer(row.get("candidate_seat"), "evaluation case candidate_seat"),
            _integer(row.get("replicate"), "evaluation case replicate"),
        ): dict(row)
        for row in bindings
    }
    opponents_by_id = {
        _nonempty_text(opponent.get("id"), "evaluation opponent id"): opponent
        for opponent in opponents
    }
    if set(opponents_by_id) != set(R198_OFFICIAL_CONTROL_ORDER):
        raise RTPThreeArmEvaluationError(
            "evaluation opponents do not match the canonical official-control order"
        )
    expected: dict[tuple[str, int, int], dict[str, Any]] = {}
    for opponent_id in R198_OFFICIAL_CONTROL_ORDER:
        for candidate_seat in (0, 1):
            for replicate in range(OFFICIAL_CONTROL_REPLICATES):
                key = (opponent_id, candidate_seat, replicate)
                row = raw_by_key.get(key)
                if row is None:
                    raise RTPThreeArmEvaluationError(
                        "evaluation cases do not cover the exact official cell order"
                    )
                expected[key] = {**row, "cell_id": f"cell-{len(expected):06d}"}
    if set(raw_by_key) != set(expected) or len(expected) != OFFICIAL_CONTROL_PAIRED_CELLS:
        raise RTPThreeArmEvaluationError("evaluation case bindings are not the exact official panel")
    return expected


def _profile_config(profile_identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    _verify_frozen_identity(profile_identity, label)
    profile = _read_json_object(profile_identity["path"], label)
    # The harness owns this small JSON contract.  A surrounding runtime
    # descriptor may keep the fields below under an explicit ``rtp`` object.
    candidate = profile.get("rtp", profile)
    return _mapping(candidate, f"{label}.rtp")


def _profile_bool(profile: Mapping[str, Any], key: str, label: str) -> bool:
    if key not in profile or not isinstance(profile[key], bool):
        raise RTPThreeArmEvaluationError(f"{label}.{key} must be a boolean")
    return bool(profile[key])


def _normalize_arm(
    arm: str,
    raw: Any,
) -> dict[str, Any]:
    value = _mapping(raw, f"arms.{arm}")
    runtime_artifact = _frozen_identity(
        value.get("runtime_artifact"), f"arms.{arm}.runtime_artifact"
    )
    runtime_profile = _frozen_identity(
        value.get("runtime_profile"), f"arms.{arm}.runtime_profile"
    )
    profile = _profile_config(runtime_profile, f"arms.{arm}.runtime_profile")
    declared_arm = _canonical_arm_name(
        profile.get("evaluation_arm"), "profile.evaluation_arm"
    )
    if declared_arm != arm:
        raise RTPThreeArmEvaluationError(
            f"arms.{arm} runtime profile declares {declared_arm!r}, not {arm!r}"
        )
    enabled = _profile_bool(profile, "recursive_turn_planner_enabled", "profile")
    bridge = _profile_bool(profile, "direct_bridge_enabled", "profile")
    forced_direct = _profile_bool(profile, "force_direct_bridge_only", "profile")
    passes = _integer(profile.get("max_neural_passes"), "profile.max_neural_passes")
    action_cap = _integer(profile.get("max_action_combos"), "profile.max_action_combos")
    sizing_profile = _nonempty_text(profile.get("sizing_profile"), "profile.sizing_profile")
    sidecar_raw = value.get("rtp_sidecar")
    sidecar: dict[str, Any] | None
    if sidecar_raw is None:
        sidecar = None
    else:
        sidecar = _frozen_identity(sidecar_raw, f"arms.{arm}.rtp_sidecar")

    # Keep the parsed profile in canonical form even where a frozen legacy
    # profile still calls the middle arm ``direct_bridge``.
    profile["evaluation_arm"] = arm
    expected = {
        "no_rtp": (False, False, False),
        DIRECT_BRIDGE_ARM: (True, True, True),
        "recursive_rtp": (True, True, False),
    }[arm]
    if (enabled, bridge, forced_direct) != expected:
        raise RTPThreeArmEvaluationError(
            f"arms.{arm} runtime profile does not implement the required arm contract"
        )
    if arm == "no_rtp" and sidecar is not None:
        raise RTPThreeArmEvaluationError("no_rtp must not package an RTP sidecar")
    if arm != "no_rtp" and sidecar is None:
        raise RTPThreeArmEvaluationError(f"{arm} must bind the RTP sidecar")
    if passes > ABSOLUTE_MAX_NEURAL_PASSES:
        raise RTPThreeArmEvaluationError(
            f"{arm} max_neural_passes={passes} exceeds absolute bound "
            f"{ABSOLUTE_MAX_NEURAL_PASSES}"
        )
    if arm != "no_rtp" and sizing_profile != R198_SIZING_PROFILE:
        raise RTPThreeArmEvaluationError(
            f"{arm} must use sizing_profile={R198_SIZING_PROFILE!r} "
            "for the r198 evaluation"
        )
    if arm != "no_rtp" and passes != R198_MAX_NEURAL_PASSES:
        raise RTPThreeArmEvaluationError(
            f"{arm} must use exact max_neural_passes="
            f"{R198_MAX_NEURAL_PASSES} for the r198 evaluation"
        )
    if arm != "no_rtp" and action_cap != R198_MAX_ACTION_COMBOS:
        raise RTPThreeArmEvaluationError(
            f"{arm} must use exact max_action_combos={R198_MAX_ACTION_COMBOS} "
            "for the r198 evaluation"
        )
    return {
        "runtime_artifact": runtime_artifact,
        "runtime_profile": runtime_profile,
        "rtp_sidecar": sidecar,
        "profile": profile,
    }


def _normalize_planner_pass_preflight(
    shared: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
    matchup_adapter_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify deterministic 6/5 probes outside the gameplay cohort.

    A forced replan is not guaranteed to occur organically.  The exact normal
    and forced pass counts are therefore proved by a separate deterministic
    preflight; rows still reject any observed count that drifts from 6/5.
    """

    receipt = _verify_immutable_frozen_identity(
        shared.get("planner_preflight_receipt"),
        "shared_artifacts.planner_preflight_receipt",
    )
    payload = _read_json_object(receipt["path"], "planner pass preflight receipt")
    if payload.get("schema") != PLANNER_PASS_PREFLIGHT_SCHEMA or payload.get(
        "status"
    ) != "passed":
        raise RTPThreeArmEvaluationError("planner pass preflight schema/status is invalid")
    direct = _mapping(arms.get(DIRECT_BRIDGE_ARM), f"arms.{DIRECT_BRIDGE_ARM}")
    recursive = _mapping(arms.get("recursive_rtp"), "arms.recursive_rtp")
    direct_sidecar = _mapping(direct.get("rtp_sidecar"), "direct sidecar")
    recursive_sidecar = _mapping(recursive.get("rtp_sidecar"), "recursive sidecar")
    if direct_sidecar.get("sha256") != recursive_sidecar.get("sha256"):
        raise RTPThreeArmEvaluationError("planner pass preflight arms do not share a sidecar")
    registry = _mapping(
        matchup_adapter_registry, "production_factory.matchup_adapter_registry"
    )
    if (
        registry.get("sha256") != R198_MATCHUP_ADAPTER_REGISTRY_SHA256
        or registry.get("bytes") != R198_MATCHUP_ADAPTER_REGISTRY_BYTES
        or registry.get("mode") != R198_MATCHUP_ADAPTER_REGISTRY_MODE
    ):
        raise RTPThreeArmEvaluationError(
            "planner pass preflight lacks the exact snapshot-local matchup adapter registry"
        )
    expected = {
        "sidecar_sha256": direct_sidecar["sha256"],
        "direct_runtime_profile_sha256": _mapping(
            direct.get("runtime_profile"), "direct runtime profile"
        )["sha256"],
        "recursive_runtime_profile_sha256": _mapping(
            recursive.get("runtime_profile"), "recursive runtime profile"
        )["sha256"],
        "matchup_adapter_registry_sha256": registry["sha256"],
        "matchup_adapter_slot_registry_digest": (
            R198_MATCHUP_ADAPTER_SLOT_REGISTRY_DIGEST
        ),
    }
    for key, value in expected.items():
        if _sha256_value(payload.get(key), f"planner preflight {key}") != value:
            raise RTPThreeArmEvaluationError(f"planner pass preflight mismatch at {key}")
    if _integer(
        payload.get("max_neural_passes"), "planner preflight max_neural_passes", minimum=1
    ) != R198_MAX_NEURAL_PASSES or _integer(
        payload.get("max_action_combos"), "planner preflight max_action_combos", minimum=1
    ) != R198_MAX_ACTION_COMBOS:
        raise RTPThreeArmEvaluationError("planner pass preflight profile budget is invalid")
    for field, expected_passes in (
        ("normal_probe_observed_neural_passes", R198_NORMAL_RECURSIVE_PLAN_PASSES),
        ("forced_replan_probe_observed_neural_passes", R198_FORCED_REPLAN_PASSES),
    ):
        if _integer(payload.get(field), f"planner preflight {field}", minimum=0) != expected_passes:
            raise RTPThreeArmEvaluationError(f"planner pass preflight mismatch at {field}")
    for field in ("normal_probe_completed", "forced_replan_probe_completed"):
        if payload.get(field) is not True:
            raise RTPThreeArmEvaluationError(f"planner pass preflight does not pass {field}")
    if _integer(
        payload.get("neural_budget_failures"),
        "planner preflight neural_budget_failures",
        minimum=0,
    ) != 0:
        raise RTPThreeArmEvaluationError("planner pass preflight recorded budget failures")
    return {
        "receipt": receipt,
        "sidecar_sha256": direct_sidecar["sha256"],
        "direct_runtime_profile_sha256": expected["direct_runtime_profile_sha256"],
        "recursive_runtime_profile_sha256": expected["recursive_runtime_profile_sha256"],
        "matchup_adapter_registry_sha256": expected[
            "matchup_adapter_registry_sha256"
        ],
        "matchup_adapter_slot_registry_digest": expected[
            "matchup_adapter_slot_registry_digest"
        ],
        "normal_probe_observed_neural_passes": R198_NORMAL_RECURSIVE_PLAN_PASSES,
        "forced_replan_probe_observed_neural_passes": R198_FORCED_REPLAN_PASSES,
        "neural_budget_failures": 0,
    }


def _normalize_gates(raw: Any | None) -> dict[str, Any]:
    overrides = {} if raw is None else _mapping(raw, "promotion_gates")
    unknown = sorted(set(overrides) - set(DEFAULT_GATES))
    if unknown:
        raise RTPThreeArmEvaluationError(
            "unknown promotion gate(s): " + ", ".join(unknown)
        )
    gates = {**DEFAULT_GATES, **overrides}
    gates["minimum_paired_cells"] = _integer(
        gates["minimum_paired_cells"], "minimum_paired_cells", minimum=1
    )
    if gates["minimum_paired_cells"] != OFFICIAL_CONTROL_PAIRED_CELLS:
        raise RTPThreeArmEvaluationError(
            "r198 requires exactly 1000 paired official-control cells"
        )
    gates["minimum_pairs_per_opponent_seat"] = _integer(
        gates["minimum_pairs_per_opponent_seat"],
        "minimum_pairs_per_opponent_seat",
        minimum=1,
    )
    if gates["minimum_pairs_per_opponent_seat"] != OFFICIAL_CONTROL_REPLICATES:
        raise RTPThreeArmEvaluationError(
            "r198 requires exactly 125 pairs per opponent/seat stratum"
        )
    gates["minimum_direct_bridge_decisions"] = _integer(
        gates["minimum_direct_bridge_decisions"],
        "minimum_direct_bridge_decisions",
        minimum=1,
    )
    gates["minimum_recursive_decisions"] = _integer(
        gates["minimum_recursive_decisions"],
        "minimum_recursive_decisions",
        minimum=1,
    )
    if gates["minimum_recursive_decisions"] < MINIMUM_RECURSIVE_DECISIONS:
        raise RTPThreeArmEvaluationError(
            "minimum_recursive_decisions cannot weaken the r198 floor of 100"
        )
    gates["minimum_recursive_share_of_intended_complex_decisions"] = _number(
        gates["minimum_recursive_share_of_intended_complex_decisions"],
        "minimum_recursive_share_of_intended_complex_decisions",
        minimum=0.0,
    )
    if gates["minimum_recursive_share_of_intended_complex_decisions"] > 1.0:
        raise RTPThreeArmEvaluationError(
            "minimum_recursive_share_of_intended_complex_decisions must be <= 1"
        )
    if (
        gates["minimum_recursive_share_of_intended_complex_decisions"]
        < MINIMUM_RECURSIVE_INTENDED_COMPLEX_SHARE
    ):
        raise RTPThreeArmEvaluationError(
            "minimum_recursive_share_of_intended_complex_decisions cannot weaken the r198 floor of 0.05"
        )
    gates["maximum_unexpected_recursive_fallback_rate"] = _number(
        gates["maximum_unexpected_recursive_fallback_rate"],
        "maximum_unexpected_recursive_fallback_rate",
        minimum=0.0,
    )
    if gates["maximum_unexpected_recursive_fallback_rate"] > 1.0:
        raise RTPThreeArmEvaluationError(
            "maximum_unexpected_recursive_fallback_rate must be <= 1"
        )
    if (
        gates["maximum_unexpected_recursive_fallback_rate"]
        > MAXIMUM_UNEXPECTED_RECURSIVE_FALLBACK_RATE
    ):
        raise RTPThreeArmEvaluationError(
            "maximum_unexpected_recursive_fallback_rate cannot exceed the r198 maximum of 0.01"
        )
    gates["minimum_recursive_delta_lower_bound"] = _number(
        gates["minimum_recursive_delta_lower_bound"],
        "minimum_recursive_delta_lower_bound",
        minimum=0.0,
    )
    gates["confidence_level"] = _number(
        gates["confidence_level"], "confidence_level"
    )
    if not 0.0 < gates["confidence_level"] < 1.0:
        raise RTPThreeArmEvaluationError("confidence_level must be in (0, 1)")
    if gates["confidence_level"] < MINIMUM_CONFIDENCE_LEVEL:
        raise RTPThreeArmEvaluationError(
            "confidence_level cannot weaken the r198 0.90 minimum"
        )
    maximum_latency = gates["maximum_p95_latency_seconds"]
    if maximum_latency is not None:
        gates["maximum_p95_latency_seconds"] = _number(
            maximum_latency, "maximum_p95_latency_seconds", minimum=0.0
        )
    return gates


def _normalize_latency_slo(
    raw: Any | None,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a numeric latency threshold to a separate owner-owned receipt.

    This evaluator owns latency evidence, not the threshold.  A number typed
    into an evaluation spec therefore never becomes promotion authority by
    itself; without an approved canonical owner record the manifest is still
    valid for measurement but its receipt remains held.
    """

    configured_maximum = gates.get("maximum_p95_latency_seconds")
    if raw is None:
        if configured_maximum is not None:
            raise RTPThreeArmEvaluationError(
                "maximum_p95_latency_seconds requires a canonical owner latency_slo receipt"
            )
        return {
            "status": "owner_threshold_absent",
            "metric": "recursive_decision_p95_seconds",
            "owner_authorized": False,
            "maximum_p95_latency_seconds": None,
        }
    value = _mapping(raw, "latency_slo")
    if value.get("status") == "owner_threshold_absent":
        if configured_maximum is not None or value != {
            "status": "owner_threshold_absent",
            "metric": "recursive_decision_p95_seconds",
            "owner_authorized": False,
            "maximum_p95_latency_seconds": None,
        }:
            raise RTPThreeArmEvaluationError("latency_slo absent-threshold marker is invalid")
        return dict(value)
    receipt = _verify_immutable_frozen_identity(
        value.get("receipt"), "latency_slo.receipt"
    )
    payload = _read_json_object(receipt["path"], "latency_slo.receipt")
    if payload.get("schema") != LATENCY_SLO_SCHEMA or payload.get("status") != "approved":
        raise RTPThreeArmEvaluationError("latency_slo receipt schema/status is invalid")
    if payload.get("owner_authorized") is not True:
        raise RTPThreeArmEvaluationError("latency_slo receipt is not owner-authorized")
    if payload.get("metric") != "recursive_decision_p95_seconds":
        raise RTPThreeArmEvaluationError("latency_slo receipt has an unsupported metric")
    if _sha256_value(
        payload.get("candidate_contract_sha256"), "latency_slo candidate_contract_sha256"
    ) != R198_CANDIDATE_CONTRACT_SHA256:
        raise RTPThreeArmEvaluationError("latency_slo receipt does not bind the r198 candidate")
    _nonempty_text(payload.get("canonical_owner_source"), "latency_slo canonical_owner_source")
    maximum = _number(
        payload.get("maximum_p95_latency_seconds"),
        "latency_slo maximum_p95_latency_seconds",
        minimum=0.0,
    )
    if configured_maximum is not None and float(configured_maximum) != maximum:
        raise RTPThreeArmEvaluationError(
            "promotion gate latency threshold differs from its owner latency_slo receipt"
        )
    return {
        "receipt": receipt,
        "status": "approved",
        "metric": "recursive_decision_p95_seconds",
        "owner_authorized": True,
        "canonical_owner_source": _nonempty_text(
            payload.get("canonical_owner_source"), "latency_slo canonical_owner_source"
        ),
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "maximum_p95_latency_seconds": maximum,
    }


def _readonly_mode(mode: int) -> bool:
    return not bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _normalize_package_tree_entries(
    raw_entries: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise RTPThreeArmEvaluationError(f"{label}.entries must be a list")
    entries: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"{label}.entries[{index}]")
        relative = _nonempty_text(entry.get("path"), f"{label}.entries[{index}].path")
        candidate = Path(relative)
        if candidate.is_absolute() or "." in candidate.parts or ".." in candidate.parts:
            raise RTPThreeArmEvaluationError(
                f"{label}.entries[{index}].path must be a normalized relative file path"
            )
        normalized = candidate.as_posix()
        if normalized in paths:
            raise RTPThreeArmEvaluationError(f"{label}.entries repeats {normalized!r}")
        paths.add(normalized)
        entries.append(
            {
                "path": normalized,
                "sha256": _sha256_value(
                    entry.get("sha256"), f"{label}.entries[{index}].sha256"
                ),
                "bytes": _integer(
                    entry.get("bytes"), f"{label}.entries[{index}].bytes", minimum=0
                ),
            }
        )
    if not entries:
        raise RTPThreeArmEvaluationError(f"{label}.entries must not be empty")
    return sorted(entries, key=lambda entry: entry["path"])


def _physical_readonly_tree_entries(root: Path, label: str) -> list[dict[str, Any]]:
    """Return a sealed regular-file tree or fail on links, specials, or writes."""

    root_mode = os.lstat(root).st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise RTPThreeArmEvaluationError(f"{label} must be a physical directory")
    if not _readonly_mode(root_mode):
        raise RTPThreeArmEvaluationError(f"{label} is writable")
    entries: list[dict[str, Any]] = []
    for current_text, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_text)
        current_mode = os.lstat(current).st_mode
        if stat.S_ISLNK(current_mode) or not stat.S_ISDIR(current_mode):
            raise RTPThreeArmEvaluationError(f"{label} contains an invalid directory")
        if not _readonly_mode(current_mode):
            raise RTPThreeArmEvaluationError(f"{label} contains a writable directory: {current}")
        for directory in list(directories):
            candidate = current / directory
            candidate_mode = os.lstat(candidate).st_mode
            if stat.S_ISLNK(candidate_mode) or not stat.S_ISDIR(candidate_mode):
                raise RTPThreeArmEvaluationError(
                    f"{label} contains a symbolic-link or non-directory child: {candidate}"
                )
            if not _readonly_mode(candidate_mode):
                raise RTPThreeArmEvaluationError(
                    f"{label} contains a writable directory: {candidate}"
                )
        for filename in filenames:
            candidate = current / filename
            candidate_mode = os.lstat(candidate).st_mode
            if stat.S_ISLNK(candidate_mode) or not stat.S_ISREG(candidate_mode):
                raise RTPThreeArmEvaluationError(
                    f"{label} contains a symbolic-link or non-regular file: {candidate}"
                )
            if not _readonly_mode(candidate_mode):
                raise RTPThreeArmEvaluationError(
                    f"{label} contains a writable file: {candidate}"
                )
            entries.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "sha256": file_digest(candidate),
                    "bytes": candidate.stat().st_size,
                }
            )
    if not entries:
        raise RTPThreeArmEvaluationError(f"{label} contains no package files")
    return sorted(entries, key=lambda entry: entry["path"])


def _normalize_evaluation_package_snapshot(
    raw: Mapping[str, Any],
    *,
    opponent_id: str,
    content_digest: str,
    label: str,
) -> dict[str, Any]:
    """Validate the immutable tree manifest and its physical sealed package."""

    artifact = _verify_immutable_frozen_identity(raw.get("artifact"), f"{label}.artifact")
    payload = _read_json_object(artifact["path"], f"{label}.artifact")
    if payload.get("schema") != EVALUATION_PACKAGE_TREE_SCHEMA or payload.get("status") != "sealed":
        raise RTPThreeArmEvaluationError(f"{label}.artifact is not a sealed package tree manifest")
    if _nonempty_text(payload.get("opponent_id"), f"{label}.artifact.opponent_id") != opponent_id:
        raise RTPThreeArmEvaluationError(f"{label}.artifact opponent id mismatch")
    if _sha256_value(payload.get("content_digest"), f"{label}.artifact.content_digest") != content_digest:
        raise RTPThreeArmEvaluationError(f"{label}.artifact content digest mismatch")
    if payload.get("no_symlinks") is not True or payload.get("all_paths_read_only") is not True:
        raise RTPThreeArmEvaluationError(f"{label}.artifact does not attest a sealed physical tree")
    deck_sha256 = _sha256_value(
        payload.get("deck_sha256"), f"{label}.artifact.deck_sha256"
    )
    deck_order_sha256 = _sha256_value(
        payload.get("deck_order_sha256"), f"{label}.artifact.deck_order_sha256"
    )
    root = _physical_existing_path(
        _nonempty_text(raw.get("package_root"), f"{label}.package_root"),
        f"{label}.package_root",
        require_directory=True,
    )
    declared_root = _physical_existing_path(
        _nonempty_text(payload.get("package_root"), f"{label}.artifact.package_root"),
        f"{label}.artifact.package_root",
        require_directory=True,
    )
    if root != declared_root:
        raise RTPThreeArmEvaluationError(
            f"{label}.package_root differs from its sealed package tree manifest"
        )
    declared_entries = _normalize_package_tree_entries(payload.get("entries"), label=f"{label}.artifact")
    actual_entries = _physical_readonly_tree_entries(root, f"{label}.package_root")
    if actual_entries != declared_entries:
        raise RTPThreeArmEvaluationError(
            f"{label}.package_root differs from its sealed package tree manifest"
        )
    tree_entries_sha256 = _sha256_value(
        payload.get("tree_entries_sha256"), f"{label}.artifact.tree_entries_sha256"
    )
    if tree_entries_sha256 != canonical_digest(actual_entries):
        raise RTPThreeArmEvaluationError(f"{label}.artifact tree entry digest mismatch")
    return {
        "artifact": artifact,
        "package_root": str(root),
        "content_digest": content_digest,
        "deck_sha256": deck_sha256,
        "deck_order_sha256": deck_order_sha256,
        "tree_entries_sha256": tree_entries_sha256,
    }


def _normalize_opponents(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPThreeArmEvaluationError("opponents must be a non-empty list")
    opponents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        value = _mapping(entry, f"opponents[{index}]")
        opponent_id = _nonempty_text(value.get("id"), f"opponents[{index}].id")
        if opponent_id in seen:
            raise RTPThreeArmEvaluationError(f"duplicate opponent id: {opponent_id}")
        seen.add(opponent_id)
        content_digest = _sha256_value(
            value.get("content_digest"), f"opponents[{index}].content_digest"
        )
        snapshot = _normalize_evaluation_package_snapshot(
            value,
            opponent_id=opponent_id,
            content_digest=content_digest,
            label=f"opponents[{index}]",
        )
        opponents.append({"id": opponent_id, **snapshot})
    if len(opponents) != OFFICIAL_CONTROL_OPPONENT_COUNT:
        raise RTPThreeArmEvaluationError(
            "r198 requires exactly four frozen official-control opponents"
        )
    return sorted(opponents, key=lambda row: row["id"])


def _validate_official_control_panel(
    shared: Mapping[str, Mapping[str, Any]],
    opponents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind r198 to the exact four non-training research controls."""

    registry = _verify_immutable_frozen_identity(
        shared.get("research_control_registry"),
        "shared_artifacts.research_control_registry",
    )
    if (
        registry["sha256"] != R198_RESEARCH_CONTROL_REGISTRY_SHA256
        or registry["bytes"] != R198_RESEARCH_CONTROL_REGISTRY_BYTES
    ):
        raise RTPThreeArmEvaluationError(
            "research control registry is not the exact frozen r198 registry"
        )
    payload = _read_json_object(registry["path"], "research control registry")
    if payload.get("schema") != "poke_bot.research_control_registry/v1" or payload.get(
        "registry_id"
    ) != "alakazam-research-controls" or _integer(
        payload.get("version"), "research control registry version", minimum=1
    ) != 1:
        raise RTPThreeArmEvaluationError("research control registry schema/identity is invalid")
    controls_raw = payload.get("controls")
    if not isinstance(controls_raw, Sequence) or isinstance(controls_raw, (str, bytes)):
        raise RTPThreeArmEvaluationError("research control registry controls must be a list")
    controls: dict[str, str] = {}
    for index, raw_control in enumerate(controls_raw):
        control = _mapping(raw_control, f"research control registry.controls[{index}]")
        opponent_id = _nonempty_text(
            control.get("opponent_id"), f"research control registry.controls[{index}].opponent_id"
        )
        digest = _sha256_value(
            control.get("content_digest"),
            f"research control registry.controls[{index}].content_digest",
        )
        if (
            control.get("training_eligible") is not False
            or control.get("formal_eval") is not False
            or control.get("included_in_gate_pass") is not False
        ):
            raise RTPThreeArmEvaluationError(
                f"research control registry row {opponent_id} is not evaluation-only"
            )
        if opponent_id in controls:
            raise RTPThreeArmEvaluationError("research control registry repeats an opponent")
        controls[opponent_id] = digest
    if controls != R198_OFFICIAL_CONTROL_OPPONENTS:
        raise RTPThreeArmEvaluationError(
            "research control registry does not contain the exact r198 control panel"
        )
    actual_opponents = {
        _nonempty_text(row.get("id"), "opponent.id"): _sha256_value(
            row.get("content_digest"), "opponent.content_digest"
        )
        for row in opponents
    }
    if actual_opponents != R198_OFFICIAL_CONTROL_OPPONENTS:
        raise RTPThreeArmEvaluationError(
            "scheduled opponents do not match the exact r198 research-control panel"
        )
    return {"registry": registry, "opponents": dict(sorted(controls.items()))}


def _normalize_rng_materials(
    raw: Any,
    *,
    expected_cases: Mapping[tuple[str, int, int], Mapping[str, Any]],
    pairing_capability: Mapping[str, Any],
    candidate_deck_sha256: str,
    opponents: Mapping[str, Mapping[str, Any]],
    evaluation_cohort_sha256: str,
    source_exclusion_proof_sha256: str,
    evaluation_case_bindings_sha256: str,
) -> list[dict[str, Any]]:
    """Require one immutable, cell-bound native snapshot seal per case."""

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPThreeArmEvaluationError("rng_materials must be a list")
    if len(raw) != len(expected_cases):
        raise RTPThreeArmEvaluationError(
            f"rng_materials must provide exactly {len(expected_cases)} sealed snapshots"
        )
    materials: list[dict[str, Any]] = []
    seen_id: set[str] = set()
    seen_digest: set[str] = set()
    seen_cell_keys: set[tuple[str, int, int]] = set()
    for index, entry in enumerate(raw):
        value = _mapping(entry, f"rng_materials[{index}]")
        for legacy_key in ("artifact", "requested_seed", "seed_only_pairing_claim"):
            if legacy_key in value:
                raise RTPThreeArmEvaluationError(
                    f"rng_materials[{index}] uses retired field {legacy_key}"
                )
        material_id = _nonempty_text(value.get("id"), f"rng_materials[{index}].id")
        kind = _nonempty_text(value.get("kind"), f"rng_materials[{index}].kind")
        if kind != "snapshot":
            raise RTPThreeArmEvaluationError(f"rng_materials[{index}].kind must be snapshot")
        if material_id in seen_id:
            raise RTPThreeArmEvaluationError(f"duplicate RNG material id: {material_id}")
        snapshot = _verify_immutable_frozen_identity(
            value.get("snapshot_artifact"), f"rng_materials[{index}].snapshot_artifact"
        )
        _verify_mode_0444(snapshot, f"rng_materials[{index}].snapshot_artifact")
        if snapshot["sha256"] in seen_digest:
            raise RTPThreeArmEvaluationError(
                "each scheduled cell requires a distinct sealed RNG snapshot"
            )
        opponent_id = _nonempty_text(
            value.get("opponent_id"), f"rng_materials[{index}].opponent_id"
        )
        candidate_seat = _integer(
            value.get("candidate_seat"), f"rng_materials[{index}].candidate_seat"
        )
        replicate = _integer(value.get("replicate"), f"rng_materials[{index}].replicate")
        cell_key = (opponent_id, candidate_seat, replicate)
        expected_case = expected_cases.get(cell_key)
        if expected_case is None or cell_key in seen_cell_keys:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] does not bind one unique frozen evaluation case"
            )
        if _nonempty_text(
            value.get("evaluation_case_id"), f"rng_materials[{index}].evaluation_case_id"
        ) != expected_case["case_id"]:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] evaluation case identity mismatch"
            )
        seal = _verify_immutable_frozen_identity(
            value.get("seal"), f"rng_materials[{index}].seal"
        )
        _verify_mode_0444(seal, f"rng_materials[{index}].seal")
        seal_payload = _read_json_object(seal["path"], f"rng_materials[{index}].seal")
        if seal_payload.get("schema") != PAIRING_SNAPSHOT_SEAL_SCHEMA or seal_payload.get(
            "status"
        ) != "sealed":
            raise RTPThreeArmEvaluationError(f"rng_materials[{index}] snapshot seal schema/status is invalid")
        for legacy_key in ("requested_seed", "seed_only_pairing_claim"):
            if legacy_key in seal_payload:
                raise RTPThreeArmEvaluationError(
                    f"rng_materials[{index}] snapshot seal uses retired field {legacy_key}"
                )
        expected_fields: dict[str, Any] = {
            "snapshot_id": material_id,
            "snapshot_artifact_sha256": snapshot["sha256"],
            "snapshot_artifact_bytes": snapshot["bytes"],
            "engine_artifact_sha256": pairing_capability["engine_artifact"]["sha256"],
            "source_artifact_sha256": pairing_capability["source_artifact"]["sha256"],
            "patch_artifact_sha256": pairing_capability["patch_artifact"]["sha256"],
            "build_artifact_sha256": pairing_capability["build_artifact"]["sha256"],
            "canonical_abi_sha256": pairing_capability["canonical_abi_sha256"],
            "candidate_deck_sha256": candidate_deck_sha256,
            "candidate_deck_order_sha256": candidate_deck_sha256,
            "opponent_id": opponent_id,
            "opponent_content_digest": expected_case["content_digest"],
            "candidate_seat": candidate_seat,
            "replicate": replicate,
            "evaluation_case_id": expected_case["case_id"],
            "evaluation_only_cohort_sha256": evaluation_cohort_sha256,
            "capture_boundary": pairing_capability["abi"]["capture_boundary"],
            "boundary_tag": pairing_capability["abi"]["boundary_tag"],
        }
        expected_fields["opponent_deck_sha256"] = opponents[opponent_id]["deck_sha256"]
        expected_fields["opponent_deck_order_sha256"] = opponents[opponent_id][
            "deck_order_sha256"
        ]
        for key, expected in expected_fields.items():
            observed = seal_payload.get(key)
            if key.endswith("_sha256"):
                observed = _sha256_value(observed, f"rng_materials[{index}].seal.{key}")
            elif key.endswith("_bytes") or key in {"candidate_seat", "replicate", "boundary_tag"}:
                observed = _integer(observed, f"rng_materials[{index}].seal.{key}")
            if observed != expected:
                raise RTPThreeArmEvaluationError(
                    f"rng_materials[{index}] snapshot seal mismatch at {key}"
                )
        if seal_payload.get("requested_seed_is_pairing_proof") is not False:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] snapshot seal treats its audit seed as pairing proof"
            )
        requested_seed = _integer(
            seal_payload.get("requested_seed_audit_only"),
            f"rng_materials[{index}].seal.requested_seed_audit_only",
        )
        if _integer(
            value.get("requested_seed_audit_only"),
            f"rng_materials[{index}].requested_seed_audit_only",
        ) != requested_seed:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] audit seed differs from its sealed debug value"
            )
        case_binding = _verify_immutable_frozen_identity(
            seal_payload.get("case_binding_artifact"),
            f"rng_materials[{index}].seal.case_binding_artifact",
        )
        _verify_mode_0444(
            case_binding, f"rng_materials[{index}].seal.case_binding_artifact"
        )
        if _sha256_value(
            seal_payload.get("case_binding_artifact_sha256"),
            f"rng_materials[{index}].seal.case_binding_artifact_sha256",
        ) != case_binding["sha256"]:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] seal does not bind its case binding artifact"
            )
        case_payload = _read_json_object(
            case_binding["path"], f"rng_materials[{index}].case_binding_artifact"
        )
        if (
            case_payload.get("schema") != PAIRING_CASE_BINDING_SCHEMA
            or case_payload.get("status") != "sealed"
        ):
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] case binding schema/status is invalid"
            )
        expected_case_fields: dict[str, Any] = {
            "cell_id": expected_case["cell_id"],
            "case_id": expected_case["case_id"],
            "opponent_id": opponent_id,
            "seat": candidate_seat,
            "replicate": replicate,
            "debug_seed": requested_seed,
            "evaluation_case_bindings_sha256": evaluation_case_bindings_sha256,
        }
        for key, expected in expected_case_fields.items():
            observed = case_payload.get(key)
            if key.endswith("_sha256"):
                observed = _sha256_value(
                    observed, f"rng_materials[{index}].case_binding_artifact.{key}"
                )
            elif key in {"seat", "replicate", "debug_seed"}:
                observed = _integer(
                    observed, f"rng_materials[{index}].case_binding_artifact.{key}"
                )
            elif key in {"cell_id", "case_id", "opponent_id"}:
                observed = _nonempty_text(
                    observed, f"rng_materials[{index}].case_binding_artifact.{key}"
                )
            if observed != expected:
                raise RTPThreeArmEvaluationError(
                    f"rng_materials[{index}] case binding mismatch at {key}"
                )
        for key, expected in (
            ("cohort_identity", evaluation_cohort_sha256),
            ("source_exclusion_identity", source_exclusion_proof_sha256),
        ):
            identity = _mapping(
                case_payload.get(key), f"rng_materials[{index}].case_binding_artifact.{key}"
            )
            if _sha256_value(
                identity.get("sha256"),
                f"rng_materials[{index}].case_binding_artifact.{key}.sha256",
            ) != expected:
                raise RTPThreeArmEvaluationError(
                    f"rng_materials[{index}] case binding mismatch at {key}"
                )
        ordered_decks = case_payload.get("ordered_deck_identities")
        if not isinstance(ordered_decks, Sequence) or isinstance(ordered_decks, (str, bytes)) or len(ordered_decks) != 2:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] case binding lacks two ordered deck identities"
            )
        candidate_deck = candidate_deck_sha256
        opponent_deck = opponents[opponent_id]["deck_sha256"]
        expected_decks = (
            (candidate_deck, opponent_deck)
            if candidate_seat == 0
            else (opponent_deck, candidate_deck)
        )
        observed_decks = tuple(
            _sha256_value(
                _mapping(
                    deck, f"rng_materials[{index}].case_binding_artifact.ordered_deck_identities"
                ).get("sha256"),
                f"rng_materials[{index}].case_binding_artifact.ordered_deck_identities.sha256",
            )
            for deck in ordered_decks
        )
        if observed_decks != expected_decks:
            raise RTPThreeArmEvaluationError(
                f"rng_materials[{index}] case binding ordered decks do not match the seat"
            )
        seen_id.add(material_id)
        seen_digest.add(snapshot["sha256"])
        seen_cell_keys.add(cell_key)
        materials.append(
            {
            "id": material_id,
            "kind": kind,
                "snapshot_artifact": snapshot,
                "seal": seal,
                "opponent_id": opponent_id,
                "candidate_seat": candidate_seat,
                "replicate": replicate,
                "evaluation_case_id": expected_case["case_id"],
                "requested_seed_audit_only": requested_seed,
                "case_binding_artifact": case_binding,
                "case_binding_artifact_sha256": case_binding["sha256"],
            }
        )
    if seen_cell_keys != set(expected_cases):
        raise RTPThreeArmEvaluationError("rng snapshot seals do not cover every official case")
    return sorted(
        materials,
        key=lambda row: (row["opponent_id"], row["candidate_seat"], row["replicate"]),
    )


def _normalize_pairing_capability(raw: Any) -> dict[str, Any]:
    """Verify a real, probe-backed RNG pairing capability.

    A Boolean capability claim is not enough: the engine binary, source-tree
    manifest, applied patch, build receipt, ABI contract, and a divergent-arm
    probe must all be checksum-bound to one another.  This remains a local
    verifier rather than a signing authority, but it makes a stale or
    hand-authored seed-only assertion fail before scheduling a comparison.
    """

    value = _mapping(raw, "pairing_capability")
    receipt = _verify_immutable_frozen_identity(
        value.get("receipt"), "pairing_capability.receipt"
    )
    payload = _read_json_object(receipt["path"], "pairing_capability.receipt")
    if payload.get("schema") != PAIRING_CAPABILITY_SCHEMA:
        raise RTPThreeArmEvaluationError("pairing capability receipt schema is invalid")
    if payload.get("status") != "available":
        raise RTPThreeArmEvaluationError("pairing capability receipt is not available")
    if payload.get("true_rng_pairing_available") is not True:
        raise RTPThreeArmEvaluationError(
            "true RNG pairing capability is unavailable; requested seeds are not pairs"
        )
    kinds = payload.get("supported_rng_kinds")
    if not isinstance(kinds, Sequence) or isinstance(kinds, (str, bytes)):
        raise RTPThreeArmEvaluationError("pairing capability receipt lacks RNG kinds")
    supported = sorted({_nonempty_text(kind, "supported_rng_kind") for kind in kinds})
    if supported != ["snapshot"] or len(supported) != len(kinds):
        raise RTPThreeArmEvaluationError("pairing capability receipt has invalid RNG kinds")

    engine = _verify_immutable_frozen_identity(
        payload.get("engine_artifact"), "pairing capability engine_artifact"
    )
    source = _verify_immutable_frozen_identity(
        payload.get("source_artifact"), "pairing capability source_artifact"
    )
    patch = _verify_immutable_frozen_identity(
        payload.get("patch_artifact"), "pairing capability patch_artifact"
    )
    build = _verify_immutable_frozen_identity(
        payload.get("build_artifact"), "pairing capability build_artifact"
    )
    abi_with_digest = _mapping(payload.get("abi"), "pairing capability abi")
    abi = dict(abi_with_digest)
    abi_digest = _sha256_value(
        abi.pop("canonical_abi_sha256", None),
        "pairing capability abi.canonical_abi_sha256",
    )
    if _nonempty_text(abi.get("name"), "pairing capability abi.name") != (
        "poke_bot.rtp_pairing_snapshot_abi"
    ):
        raise RTPThreeArmEvaluationError("pairing capability ABI name is unrecognized")
    if _integer(abi.get("version"), "pairing capability abi.version", minimum=1) != 2:
        raise RTPThreeArmEvaluationError("pairing capability ABI version is unrecognized")
    for field in (
        "requires_device_rand_false",
        "requires_time_limit_zero",
        "requires_pristine_process_initialization",
        "full_state_game_rng_config_counters",
    ):
        if abi.get(field) is not True:
            raise RTPThreeArmEvaluationError(
                f"pairing capability ABI does not attest {field}"
            )
    if abi.get("serialization_compatibility") != "exact_engine_artifact_only":
        raise RTPThreeArmEvaluationError(
            "pairing capability ABI does not enforce exact-engine serialization"
        )
    if _nonempty_text(
        abi.get("capture_boundary"), "pairing capability abi.capture_boundary"
    ) != "post_battle_start_first_external_selection" or _integer(
        abi.get("boundary_tag"), "pairing capability abi.boundary_tag", minimum=1
    ) != 1:
        raise RTPThreeArmEvaluationError("pairing capability ABI capture boundary is invalid")
    expected_symbols = {
        "start_symbol": "RtpPairingBattleStartSeededOut",
        "initialize_symbol": "RtpPairingSnapshotInitialize",
        "restore_serialized_symbol": "RtpPairingSnapshotRestoreSerialized",
        "observation_symbol": "RtpPairingSnapshotGetBattleJsonOut",
    }
    for field, expected_symbol in expected_symbols.items():
        if _nonempty_text(abi.get(field), f"pairing capability abi.{field}") != expected_symbol:
            raise RTPThreeArmEvaluationError(
                f"pairing capability ABI has an unexpected {field}"
            )
    if "snapshot" in supported:
        for field in (
            "capture_symbol",
            "restore_symbol",
            "release_symbol",
            "serialized_size_symbol",
            "serialize_symbol",
            "fingerprint_size_symbol",
            "fingerprint_symbol",
            "last_error_symbol",
        ):
            _nonempty_text(abi.get(field), f"pairing capability abi.{field}")
    if "tape" in supported:
        for field in ("tape_capture_symbol", "tape_replay_symbol"):
            _nonempty_text(abi.get(field), f"pairing capability abi.{field}")
    if canonical_digest(abi) != abi_digest:
        raise RTPThreeArmEvaluationError("pairing capability ABI digest mismatch")

    build_payload = _read_json_object(build["path"], "pairing capability build receipt")
    if build_payload.get("schema") != (
        "poke_bot.recursive_turn_planner.true_rng_pairing_build/v1"
    ) or build_payload.get("status") != "success":
        raise RTPThreeArmEvaluationError("pairing capability build receipt is not successful")
    for key, identity in (
        ("engine_artifact_sha256", engine),
        ("source_artifact_sha256", source),
        ("patch_artifact_sha256", patch),
    ):
        if _sha256_value(build_payload.get(key), f"pairing build {key}") != identity[
            "sha256"
        ]:
            raise RTPThreeArmEvaluationError(
                f"pairing capability build receipt mismatch at {key}"
            )
    if _sha256_value(
        build_payload.get("canonical_abi_sha256"), "pairing build canonical_abi_sha256"
    ) != abi_digest:
        raise RTPThreeArmEvaluationError("pairing capability build ABI digest mismatch")

    probe = _verify_immutable_frozen_identity(
        payload.get("probe"), "pairing capability probe"
    )
    probe_payload = _read_json_object(probe["path"], "pairing capability probe")
    if probe_payload.get("schema") != PAIRING_PROBE_SCHEMA or probe_payload.get(
        "status"
    ) != "passed":
        raise RTPThreeArmEvaluationError("pairing capability probe did not pass")
    for key, identity in (
        ("engine_artifact_sha256", engine),
        ("source_artifact_sha256", source),
        ("patch_artifact_sha256", patch),
        ("build_artifact_sha256", build),
    ):
        if _sha256_value(probe_payload.get(key), f"pairing probe {key}") != identity[
            "sha256"
        ]:
            raise RTPThreeArmEvaluationError(
                f"pairing capability probe mismatch at {key}"
            )
    if _sha256_value(
        probe_payload.get("canonical_abi_sha256"), "pairing probe canonical_abi_sha256"
    ) != abi_digest:
        raise RTPThreeArmEvaluationError("pairing capability probe ABI digest mismatch")
    verified_kinds = probe_payload.get("verified_rng_kinds")
    if not isinstance(verified_kinds, Sequence) or isinstance(verified_kinds, (str, bytes)):
        raise RTPThreeArmEvaluationError("pairing capability probe lacks verified RNG kinds")
    if sorted({_nonempty_text(kind, "pairing probe verified_rng_kind") for kind in verified_kinds}) != supported:
        raise RTPThreeArmEvaluationError(
            "pairing capability probe kinds do not match the capability receipt"
        )
    for field in (
        "device_rand_false_verified",
        "requested_seed_only_rejected",
        "duplicate_restore_independent_handles",
        "all_arms_restored_or_replayed",
        "divergent_policy_true_pairing_passed",
        "delayed_restore_transcript_passed",
        "cross_process_restore_passed",
    ):
        if probe_payload.get(field) is not True:
            raise RTPThreeArmEvaluationError(f"pairing capability probe does not pass {field}")
    deterministic_restore_probe = _mapping(
        probe_payload.get("deterministic_restore_probe"),
        "pairing probe deterministic_restore_probe",
    )
    if deterministic_restore_probe.get("passed") is not True:
        raise RTPThreeArmEvaluationError("pairing deterministic restore probe did not pass")
    _sha256_value(
        deterministic_restore_probe.get("deterministic_transcript_sha256"),
        "pairing probe deterministic_restore_probe.deterministic_transcript_sha256",
    )
    _integer(
        deterministic_restore_probe.get("transcript_steps"),
        "pairing probe deterministic_restore_probe.transcript_steps",
        minimum=1,
    )
    _sha256_value(
        deterministic_restore_probe.get("initial_snapshot_fingerprint_sha256"),
        "pairing probe deterministic_restore_probe.initial_snapshot_fingerprint_sha256",
    )
    _integer(
        deterministic_restore_probe.get("initial_snapshot_fingerprint_bytes"),
        "pairing probe deterministic_restore_probe.initial_snapshot_fingerprint_bytes",
        minimum=1,
    )
    return {
        "receipt": receipt,
        "engine_artifact": engine,
        "source_artifact": source,
        "patch_artifact": patch,
        "build_artifact": build,
        "abi": {**abi, "canonical_abi_sha256": abi_digest},
        "canonical_abi_sha256": abi_digest,
        "probe": probe,
        "supported_rng_kinds": supported,
        "true_rng_pairing_available": True,
    }


def _normalize_evaluation_cg_closure(
    raw: Any, pairing_capability: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the private evaluation CG closure to the exact pairing engine.

    A native snapshot ABI is insufficient if the simulator used for its
    observations can load a different CG/metadata closure.  The closure record
    is independently sealed and must name the same engine/build/ABI that the
    capability probe verified.
    """

    value = _mapping(raw, "evaluation_cg_closure")
    if set(value) != {"receipt", "runtime_library"}:
        raise RTPThreeArmEvaluationError(
            "evaluation_cg_closure must contain exactly its receipt and snapshot-local runtime library"
        )
    receipt = _verify_immutable_frozen_identity(
        value.get("receipt"), "evaluation_cg_closure.receipt"
    )
    _verify_mode_0444(receipt, "evaluation_cg_closure.receipt")
    # The closure receipt's ``engine_artifact`` is the private build artifact
    # (0555).  Evaluation children deliberately load a byte-identical,
    # snapshot-local 0444 copy under CG_LIB_PATH instead.  Keep both identities
    # explicit: comparing their paths would either reject the intended sealed
    # deployment or tempt a child to load the private builder output.
    runtime_library = _verify_immutable_frozen_identity(
        value.get("runtime_library"), "evaluation_cg_closure.runtime_library"
    )
    _verify_mode_0444(runtime_library, "evaluation_cg_closure.runtime_library")
    if Path(runtime_library["path"]).name != "libcg.so":
        raise RTPThreeArmEvaluationError(
            "evaluation_cg_closure.runtime_library must name snapshot-local libcg.so"
        )
    payload = _read_json_object(receipt["path"], "evaluation_cg_closure.receipt")
    if (
        payload.get("schema") != PAIRING_EVAL_CG_CLOSURE_SCHEMA
        or payload.get("status") != "sealed"
        or _integer(payload.get("snapshot_abi_version"), "evaluation CG closure ABI version") != 2
        or _nonempty_text(
            payload.get("sim_initializer_symbol"), "evaluation CG closure initializer"
        )
        != "RtpPairingSnapshotInitialize"
    ):
        raise RTPThreeArmEvaluationError("evaluation CG closure schema/ABI is invalid")
    expected_identities = {
        "engine_artifact": pairing_capability["engine_artifact"],
        "pairing_build_artifact": pairing_capability["build_artifact"],
    }
    identities: dict[str, dict[str, Any]] = {}
    for name in (
        "engine_artifact",
        "pairing_build_artifact",
        "cg_source_manifest",
        "closure_manifest",
        "metadata_parity",
    ):
        identity = _verify_immutable_frozen_identity(
            payload.get(name), f"evaluation CG closure {name}"
        )
        if name in expected_identities and (
            identity["sha256"] != expected_identities[name]["sha256"]
            or identity["bytes"] != expected_identities[name]["bytes"]
        ):
            raise RTPThreeArmEvaluationError(
                f"evaluation CG closure {name} differs from pairing capability"
            )
        identities[name] = identity
    if _sha256_value(
        payload.get("canonical_abi_sha256"), "evaluation CG closure canonical_abi_sha256"
    ) != pairing_capability["canonical_abi_sha256"]:
        raise RTPThreeArmEvaluationError("evaluation CG closure ABI differs from pairing capability")
    expected_tree_paths = ("__init__.py", "api.py", "game.py", "libcg.so", "sim.py", "utils.py")
    tree_digests: dict[str, str] = {}
    for name, schema in (
        (
            "cg_source_manifest",
            "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1",
        ),
        (
            "closure_manifest",
            "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure_manifest/v1",
        ),
    ):
        tree = _read_json_object(identities[name]["path"], f"evaluation CG closure {name}")
        if tree.get("schema") != schema or _integer(
            tree.get("file_count"), f"evaluation CG closure {name} file_count", minimum=1
        ) != len(expected_tree_paths):
            raise RTPThreeArmEvaluationError(f"evaluation CG closure {name} schema/count is invalid")
        raw_files = tree.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise RTPThreeArmEvaluationError(f"evaluation CG closure {name} files are invalid")
        files: list[dict[str, Any]] = []
        for index, raw_file in enumerate(raw_files):
            file_row = _mapping(raw_file, f"evaluation CG closure {name}.files[{index}]")
            files.append(
                {
                    "relative_path": _nonempty_text(
                        file_row.get("relative_path"),
                        f"evaluation CG closure {name}.files[{index}].relative_path",
                    ),
                    "sha256": _sha256_value(
                        file_row.get("sha256"),
                        f"evaluation CG closure {name}.files[{index}].sha256",
                    ),
                    "bytes": _integer(
                        file_row.get("bytes"),
                        f"evaluation CG closure {name}.files[{index}].bytes",
                        # These two sealed manifests describe regular files,
                        # not nonempty payloads.  The curated package's
                        # ``__init__.py`` is intentionally a zero-byte package
                        # marker in both the source and relocated closure
                        # trees.  Its path, SHA-256, tree digest, and the
                        # enclosing immutable evidence identity remain exact;
                        # negative, missing, or non-integral byte counts are
                        # invalid.
                        minimum=0,
                    ),
                }
            )
        if tuple(item["relative_path"] for item in files) != expected_tree_paths:
            raise RTPThreeArmEvaluationError(
                f"evaluation CG closure {name} does not have the exact curated tree"
            )
        material = {"schema": schema, "file_count": len(files), "files": files}
        tree_digest = _sha256_value(
            tree.get("tree_sha256"), f"evaluation CG closure {name} tree_sha256"
        )
        if canonical_digest(material) != tree_digest:
            raise RTPThreeArmEvaluationError(
                f"evaluation CG closure {name} tree digest mismatch"
            )
        tree_digests[name] = tree_digest
        if name == "closure_manifest":
            libcg = next(item for item in files if item["relative_path"] == "libcg.so")
            if (
                libcg["sha256"] != identities["engine_artifact"]["sha256"]
                or libcg["bytes"] != identities["engine_artifact"]["bytes"]
            ):
                raise RTPThreeArmEvaluationError(
                    "evaluation CG closure libcg.so differs from the pairing engine artifact"
                )
    if (
        runtime_library["sha256"] != identities["engine_artifact"]["sha256"]
        or runtime_library["bytes"] != identities["engine_artifact"]["bytes"]
    ):
        raise RTPThreeArmEvaluationError(
            "snapshot-local evaluation CG libcg.so differs from the pairing engine artifact"
        )
    metadata = _read_json_object(
        identities["metadata_parity"]["path"], "evaluation CG closure metadata_parity"
    )
    if (
        metadata.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1"
        or metadata.get("status") != "passed"
        or metadata.get("independent_processes") is not True
    ):
        raise RTPThreeArmEvaluationError("evaluation CG metadata-parity record is invalid")
    pairing_engine = _verify_immutable_frozen_identity(
        metadata.get("pairing_engine"), "evaluation CG metadata pairing engine"
    )
    _verify_immutable_frozen_identity(
        metadata.get("public_cg_engine"), "evaluation CG metadata public engine"
    )
    if (
        pairing_engine["sha256"] != identities["engine_artifact"]["sha256"]
        or pairing_engine["bytes"] != identities["engine_artifact"]["bytes"]
    ):
        raise RTPThreeArmEvaluationError("metadata parity does not use the pairing engine")
    for key in (
        "all_card_canonical_sha256",
        "all_attack_canonical_sha256",
        "public_all_card_raw_sha256",
        "pairing_all_card_raw_sha256",
        "public_all_attack_raw_sha256",
        "pairing_all_attack_raw_sha256",
    ):
        _sha256_value(metadata.get(key), f"evaluation CG metadata {key}")
    if (
        metadata["public_all_card_raw_sha256"] != metadata["pairing_all_card_raw_sha256"]
        or metadata["public_all_attack_raw_sha256"] != metadata["pairing_all_attack_raw_sha256"]
    ):
        raise RTPThreeArmEvaluationError("evaluation CG metadata raw values are not at parity")
    for field in (
        "public_initialized_before_pairing",
        "pairing_private_initialize_after_public_passed",
        "distinct_dso_handles",
    ):
        if metadata.get(field) is not True:
            raise RTPThreeArmEvaluationError(
                f"evaluation CG metadata does not prove {field}"
            )
    return {
        "receipt": receipt,
        "runtime_library": runtime_library,
        **identities,
        "canonical_abi_sha256": pairing_capability["canonical_abi_sha256"],
        "sim_initializer_symbol": "RtpPairingSnapshotInitialize",
        "snapshot_abi_version": 2,
        "cg_source_tree_sha256": tree_digests["cg_source_manifest"],
        "closure_tree_sha256": tree_digests["closure_manifest"],
        "all_card_canonical_sha256": _sha256_value(
            metadata.get("all_card_canonical_sha256"), "evaluation CG metadata all-card digest"
        ),
        "all_attack_canonical_sha256": _sha256_value(
            metadata.get("all_attack_canonical_sha256"), "evaluation CG metadata all-attack digest"
        ),
    }


def _validate_production_factory_runtime_library(
    production_factory: Mapping[str, Any],
    evaluation_cg_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Tie the evaluator's DSO and V6 roster to the sealed factory snapshot.

    The generic harness cannot inspect the factory's source-snapshot contract
    itself without importing the production implementation.  It can and must
    nevertheless bind the exact snapshot-local library and matchup roster
    identities that the factory later consumes.  That prevents a base spec
    from pointing the runner and compiler at byte-identical-but-different
    loader locations or at an ambient V6 registry.
    """

    factory = _mapping(production_factory, "production_factory")
    factory_cg = _mapping(
        factory.get("evaluation_cg"), "production_factory.evaluation_cg"
    )
    factory_library = _verify_immutable_frozen_identity(
        factory_cg.get("library"), "production_factory.evaluation_cg.library"
    )
    _verify_mode_0444(
        factory_library, "production_factory.evaluation_cg.library"
    )
    runtime_library = _mapping(
        evaluation_cg_closure.get("runtime_library"),
        "evaluation_cg_closure.runtime_library",
    )
    if factory_library != runtime_library:
        raise RTPThreeArmEvaluationError(
            "production factory CG library differs from the evaluator snapshot-local runtime library"
        )
    return _snapshot_local_matchup_adapter_registry(factory)


def _manifest_material(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The immutable logical payload, excluding only the creation timestamp."""

    keys = (
        "schema",
        "status",
        "arm_order",
        "production_factory",
        "shared_artifacts",
        "arms",
        "candidate_evaluation_binding",
        "opponents",
        "rng_materials",
        "schedule",
        "pairing_capability",
        "evaluation_cg_closure",
        "official_control_panel",
        "r197_source_exclusion_binding",
        "planner_pass_preflight",
        "promotion_gates",
        "latency_slo",
        "r198_profile_contract",
        "evaluation_isolation",
    )
    return {key: manifest.get(key) for key in keys}


def _normalize_arm_input(raw: Any) -> dict[str, Any]:
    """Accept the old key once, but construct only canonical arm keys."""

    value = _mapping(raw, "arms")
    normalized: dict[str, Any] = {}
    for supplied_arm, arm_spec in value.items():
        canonical = _canonical_arm_name(supplied_arm, "arms key")
        if canonical not in ARMS:
            raise RTPThreeArmEvaluationError(f"unknown evaluation arm: {supplied_arm}")
        if canonical in normalized:
            raise RTPThreeArmEvaluationError(
                f"duplicate canonical evaluation arm: {canonical}"
            )
        normalized[canonical] = arm_spec
    if set(normalized) != set(ARMS):
        missing = sorted(set(ARMS) - set(normalized))
        extra = sorted(set(normalized) - set(ARMS))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unknown=" + ",".join(extra))
        raise RTPThreeArmEvaluationError(
            "arms must be exactly no_rtp/direct_bridge_recursive_disabled/recursive_rtp ("
            + "; ".join(details)
            + ")"
        )
    return normalized


def _normalize_candidate_evaluation_binding(
    raw: Any,
    *,
    shared: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Make semantic serving identities explicit beside raw file identities.

    The deck CSV file checksum and canonical ordered-60-card checksum are
    intentionally different.  This record prevents a consumer from treating
    one as a substitute for the other, and closes the same gap for the sidecar
    config digest (canonical JSON) versus the sidecar checkpoint bytes.
    """

    value = _mapping(raw, "candidate_evaluation_binding")
    expected = {
        "schema": CANDIDATE_EVALUATION_BINDING_SCHEMA,
        "status": "bound",
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "parent_checkpoint_sha256": R198_PARENT_CHECKPOINT_SHA256,
        "sidecar_sha256": R198_SIDECAR_SHA256,
        "sidecar_config_sha256": R198_SIDECAR_CONFIG_SHA256,
        "deck_file_sha256": R198_DECK_FILE_SHA256,
        "deck_cards_sha256": R198_DECK_CARDS_SHA256,
        "matchup_tree_sha256": R198_MATCHUP_TREE_SHA256,
        "sizing_profile": R198_SIZING_PROFILE,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_action_combos": R198_MAX_ACTION_COMBOS,
        "required_neural_passes": {
            "normal": R198_NORMAL_RECURSIVE_PLAN_PASSES,
            "forced_replan": R198_FORCED_REPLAN_PASSES,
        },
    }
    if set(value) != set(expected):
        raise RTPThreeArmEvaluationError(
            "candidate_evaluation_binding must contain exactly the r198 semantic identities"
        )
    for key, wanted in expected.items():
        observed = value.get(key)
        if key.endswith("_sha256"):
            observed = _sha256_value(observed, f"candidate_evaluation_binding.{key}")
        elif key in {"max_neural_passes", "max_action_combos"}:
            observed = _integer(observed, f"candidate_evaluation_binding.{key}")
        elif key == "required_neural_passes":
            observed = _mapping(observed, f"candidate_evaluation_binding.{key}")
        if observed != wanted:
            raise RTPThreeArmEvaluationError(
                f"candidate_evaluation_binding mismatch at {key}"
            )
    if (
        shared["parent_checkpoint"]["sha256"] != expected["parent_checkpoint_sha256"]
        or shared["deck"]["sha256"] != expected["deck_file_sha256"]
        or shared["matchup_tree"]["sha256"] != expected["matchup_tree_sha256"]
    ):
        raise RTPThreeArmEvaluationError(
            "candidate_evaluation_binding does not match frozen parent/deck/tree files"
        )
    for arm in (DIRECT_BRIDGE_ARM, "recursive_rtp"):
        sidecar = arms[arm].get("rtp_sidecar")
        if sidecar is None or sidecar["sha256"] != expected["sidecar_sha256"]:
            raise RTPThreeArmEvaluationError(
                "candidate_evaluation_binding does not match the frozen RTP sidecar"
            )
        profile = _mapping(arms[arm].get("profile"), f"arms.{arm}.profile")
        for key in ("sizing_profile", "max_neural_passes", "max_action_combos"):
            if profile.get(key) != expected[key]:
                raise RTPThreeArmEvaluationError(
                    f"candidate_evaluation_binding differs from {arm} profile at {key}"
                )
    return expected


def prepare_three_arm_manifest(
    *,
    output_path: str | Path,
    production_factory: Mapping[str, Any],
    shared_artifacts: Mapping[str, Any],
    arms: Mapping[str, Any],
    candidate_evaluation_binding: Mapping[str, Any],
    opponents: Sequence[Mapping[str, Any]],
    rng_materials: Sequence[Mapping[str, Any]],
    pairing_capability: Mapping[str, Any],
    evaluation_cg_closure: Mapping[str, Any],
    source_exclusion_proof: Mapping[str, Any],
    replicates_per_seat: int,
    promotion_gates: Mapping[str, Any] | None = None,
    latency_slo: Mapping[str, Any] | None = None,
) -> Path:
    """Create an immutable, true-RNG-bound three-arm evaluation manifest.

    ``rng_materials`` must already contain one checksum-addressed tape or
    restorable snapshot for every opponent/seat/replicate cell, and
    ``pairing_capability`` must carry a verified engine capability receipt.
    A requested seed alone is intentionally insufficient.
    """

    # The factory owns the source-snapshot and private-engine runtime binding.
    # It is intentionally preserved verbatim in the evaluator manifest so each
    # fresh runner child can re-verify it; dropping it would leave the concrete
    # production factory unable to establish its sealed environment.
    normalized_production_factory = _mapping(
        production_factory, "production_factory"
    )
    if not normalized_production_factory:
        raise RTPThreeArmEvaluationError("production_factory must not be empty")
    gates = _normalize_gates(promotion_gates)
    normalized_latency_slo = _normalize_latency_slo(latency_slo, gates)
    normalized_shared = _normalize_shared_artifacts(shared_artifacts)
    arm_input = _normalize_arm_input(arms)
    normalized_arms = {
        arm: _normalize_arm(arm, arm_input[arm]) for arm in ARMS
    }
    normalized_candidate_evaluation_binding = _normalize_candidate_evaluation_binding(
        candidate_evaluation_binding,
        shared=normalized_shared,
        arms=normalized_arms,
    )
    direct_sidecar = normalized_arms[DIRECT_BRIDGE_ARM]["rtp_sidecar"]
    recursive_sidecar = normalized_arms["recursive_rtp"]["rtp_sidecar"]
    if direct_sidecar is None or recursive_sidecar is None:  # defensive typing
        raise RTPThreeArmEvaluationError("bridge and recursive arms require a sidecar")
    if direct_sidecar["sha256"] != recursive_sidecar["sha256"]:
        raise RTPThreeArmEvaluationError(
            "direct_bridge_recursive_disabled and recursive_rtp must use the exact same RTP sidecar"
        )
    for key in (
        "max_neural_passes",
        "max_action_combos",
        "num_plan_candidates",
        "max_recursion_depth",
        "max_plan_length",
        "d_model",
        "dynamics_width",
        "complexity_option_threshold",
        "complexity_entropy_threshold",
        "repair_budget",
    ):
        direct_value = normalized_arms[DIRECT_BRIDGE_ARM]["profile"].get(key)
        recursive_value = normalized_arms["recursive_rtp"]["profile"].get(key)
        if direct_value != recursive_value:
            raise RTPThreeArmEvaluationError(
                f"direct_bridge_recursive_disabled and recursive_rtp differ at profile.{key}"
            )

    normalized_opponents = _normalize_opponents(opponents)
    official_control_panel = _validate_official_control_panel(
        normalized_shared, normalized_opponents
    )
    r197_source_exclusion_binding = _normalize_r197_source_exclusion_binding(
        normalized_shared,
        source_exclusion_proof,
        opponent_content_digests={
            str(row["id"]): str(row["content_digest"])
            for row in normalized_opponents
        },
    )
    evaluation_cases = _case_bindings_with_cell_ids(
        r197_source_exclusion_binding["evaluation_case_bindings"], normalized_opponents
    )
    repeats = _integer(replicates_per_seat, "replicates_per_seat", minimum=1)
    expected_cells = len(normalized_opponents) * 2 * repeats
    if expected_cells != OFFICIAL_CONTROL_PAIRED_CELLS:
        raise RTPThreeArmEvaluationError(
            f"r198 schedule must be 4 opponents × 2 seats × 125 replicates = "
            f"{OFFICIAL_CONTROL_PAIRED_CELLS} cells, got {expected_cells}"
        )
    if repeats != OFFICIAL_CONTROL_REPLICATES:
        raise RTPThreeArmEvaluationError(
            "r198 requires exactly 125 replicates per opponent/seat"
        )
    normalized_pairing_capability = _normalize_pairing_capability(pairing_capability)
    normalized_evaluation_cg_closure = _normalize_evaluation_cg_closure(
        evaluation_cg_closure, normalized_pairing_capability
    )
    matchup_adapter_registry = _validate_production_factory_runtime_library(
        normalized_production_factory, normalized_evaluation_cg_closure
    )
    normalized_production_factory["matchup_adapter_registry"] = (
        matchup_adapter_registry
    )
    planner_pass_preflight = _normalize_planner_pass_preflight(
        normalized_shared,
        normalized_arms,
        matchup_adapter_registry,
    )
    normalized_rng = _normalize_rng_materials(
        rng_materials,
        expected_cases=evaluation_cases,
        pairing_capability=normalized_pairing_capability,
        candidate_deck_sha256=normalized_shared["deck"]["sha256"],
        opponents={str(row["id"]): row for row in normalized_opponents},
        evaluation_cohort_sha256=r197_source_exclusion_binding["evaluation_only_cohort"][
            "sha256"
        ],
        source_exclusion_proof_sha256=r197_source_exclusion_binding[
            "source_exclusion_proof"
        ]["sha256"],
        evaluation_case_bindings_sha256=r197_source_exclusion_binding[
            "evaluation_case_bindings_sha256"
        ],
    )
    rng_by_case = {
        (str(row["opponent_id"]), int(row["candidate_seat"]), int(row["replicate"])): row
        for row in normalized_rng
    }
    schedule: list[dict[str, Any]] = []
    normalized_opponents_by_id = {
        str(opponent["id"]): opponent for opponent in normalized_opponents
    }
    for opponent_id in R198_OFFICIAL_CONTROL_ORDER:
        opponent = normalized_opponents_by_id[opponent_id]
        for candidate_seat in (0, 1):
            for replicate in range(repeats):
                rng = rng_by_case[(str(opponent["id"]), candidate_seat, replicate)]
                cell: dict[str, Any] = {
                    "cell_id": evaluation_cases[
                        (str(opponent["id"]), candidate_seat, replicate)
                    ]["cell_id"],
                    "opponent_id": opponent["id"],
                    "candidate_seat": candidate_seat,
                    "replicate": replicate,
                    "evaluation_case_id": evaluation_cases[
                        (str(opponent["id"]), candidate_seat, replicate)
                    ]["case_id"],
                    "evaluation_case_content_digest": evaluation_cases[
                        (str(opponent["id"]), candidate_seat, replicate)
                    ]["content_digest"],
                    "evaluation_case_bindings_sha256": r197_source_exclusion_binding[
                        "evaluation_case_bindings_sha256"
                    ],
                    "rng_identity": {
                        "id": rng["id"],
                        "kind": rng["kind"],
                        "sha256": rng["snapshot_artifact"]["sha256"],
                        "bytes": rng["snapshot_artifact"]["bytes"],
                        "seal_sha256": rng["seal"]["sha256"],
                        "capture_boundary": normalized_pairing_capability["abi"][
                            "capture_boundary"
                        ],
                        "boundary_tag": normalized_pairing_capability["abi"][
                            "boundary_tag"
                        ],
                    },
                    # This may help external workers pick a debug trace, but
                    # it is never accepted as evidence of pairing.
                    "requested_seed_audit_only": rng["requested_seed_audit_only"],
                    "requested_seed_is_pairing_proof": False,
                }
                schedule.append(cell)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared_true_rng_pairing_required",
        "arm_order": list(ARMS),
        "production_factory": normalized_production_factory,
        "shared_artifacts": normalized_shared,
        "arms": normalized_arms,
        "candidate_evaluation_binding": normalized_candidate_evaluation_binding,
        "opponents": normalized_opponents,
        "rng_materials": normalized_rng,
        "schedule": schedule,
        "pairing_capability": normalized_pairing_capability,
        "evaluation_cg_closure": normalized_evaluation_cg_closure,
        "official_control_panel": official_control_panel,
        "r197_source_exclusion_binding": r197_source_exclusion_binding,
        "planner_pass_preflight": planner_pass_preflight,
        "promotion_gates": gates,
        "latency_slo": normalized_latency_slo,
        "r198_profile_contract": {
            "bridge_and_recursive_sizing_profile": R198_SIZING_PROFILE,
            "bridge_and_recursive_max_neural_passes": R198_MAX_NEURAL_PASSES,
            "bridge_and_recursive_max_action_combos": R198_MAX_ACTION_COMBOS,
            "normal_recursive_plan_observed_passes": R198_NORMAL_RECURSIVE_PLAN_PASSES,
            "forced_replan_observed_passes": R198_FORCED_REPLAN_PASSES,
            "absolute_future_max_neural_passes": ABSOLUTE_MAX_NEURAL_PASSES,
            "future_change_requires_distinct_authorized_profile_and_receipt": True,
        },
        "evaluation_isolation": {
            "training_eligible": False,
            "replay_eligible": False,
            "formal_gate": False,
            "serving_change_authorized": False,
            "self_promotion_allowed": False,
            "requested_seed_is_pairing_proof": False,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["manifest_input_sha256"] = canonical_digest(_manifest_material(manifest))
    return _immutable_json(
        output_path,
        manifest,
        existing_digest_key="manifest_input_sha256",
    )


def prepare_three_arm_manifest_from_spec(
    spec: Mapping[str, Any], *, output_path: str | Path
) -> Path:
    """Convenience adapter for the JSON shape consumed by the CLI."""

    value = _mapping(spec, "evaluation spec")
    return prepare_three_arm_manifest(
        output_path=output_path,
        production_factory=_mapping(
            value.get("production_factory"), "production_factory"
        ),
        shared_artifacts=_mapping(value.get("shared_artifacts"), "shared_artifacts"),
        arms=_mapping(value.get("arms"), "arms"),
        candidate_evaluation_binding=_mapping(
            value.get("candidate_evaluation_binding"), "candidate_evaluation_binding"
        ),
        opponents=value.get("opponents"),
        rng_materials=value.get("rng_materials"),
        pairing_capability=_mapping(
            value.get("pairing_capability"), "pairing_capability"
        ),
        evaluation_cg_closure=_mapping(
            value.get("evaluation_cg_closure"), "evaluation_cg_closure"
        ),
        source_exclusion_proof=_mapping(
            value.get("source_exclusion_proof"), "source_exclusion_proof"
        ),
        replicates_per_seat=value.get("replicates_per_seat"),
        promotion_gates=(
            None
            if value.get("promotion_gates") is None
            else _mapping(value.get("promotion_gates"), "promotion_gates")
        ),
        latency_slo=(
            None
            if value.get("latency_slo") is None
            else _mapping(value.get("latency_slo"), "latency_slo")
        ),
    )


def _load_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _observed_file_identity(path, "evaluation manifest")
    manifest = _read_json_object(source["path"], "evaluation manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RTPThreeArmEvaluationError("not an RTP three-arm evaluation manifest")
    if manifest.get("status") != "prepared_true_rng_pairing_required":
        raise RTPThreeArmEvaluationError("evaluation manifest is not in prepared state")
    expected = _nonempty_text(
        manifest.get("manifest_input_sha256"), "manifest_input_sha256"
    )
    if canonical_digest(_manifest_material(manifest)) != expected:
        raise RTPThreeArmEvaluationError("evaluation manifest logical digest changed")
    return manifest, source


def verify_manifest_frozen_artifacts(manifest: Mapping[str, Any]) -> None:
    """Re-hash every frozen input before a receipt is allowed to exist."""

    value = _mapping(manifest, "evaluation manifest")
    production_factory = _mapping(
        value.get("production_factory"), "production_factory"
    )
    if not production_factory:
        raise RTPThreeArmEvaluationError("manifest lacks the sealed production factory binding")
    for name, identity in _mapping(value.get("shared_artifacts"), "shared_artifacts").items():
        _verify_frozen_identity(identity, f"shared_artifacts.{name}")
    arms = _mapping(value.get("arms"), "arms")
    if set(arms) != set(ARMS):
        raise RTPThreeArmEvaluationError("manifest arms are not the required three-arm set")
    normalized_arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        spec = _mapping(arms.get(arm), f"arms.{arm}")
        normalized = _normalize_arm(arm, spec)
        if canonical_digest(spec) != canonical_digest(normalized):
            raise RTPThreeArmEvaluationError(
                f"manifest arm contract differs from frozen {arm} profile"
            )
        normalized_arms[arm] = normalized
    normalized_candidate_evaluation_binding = _normalize_candidate_evaluation_binding(
        value.get("candidate_evaluation_binding"),
        shared=_mapping(value.get("shared_artifacts"), "shared_artifacts"),
        arms=normalized_arms,
    )
    if canonical_digest(value.get("candidate_evaluation_binding")) != canonical_digest(
        normalized_candidate_evaluation_binding
    ):
        raise RTPThreeArmEvaluationError("manifest candidate semantic binding changed")
    direct_sidecar = normalized_arms[DIRECT_BRIDGE_ARM]["rtp_sidecar"]
    recursive_sidecar = normalized_arms["recursive_rtp"]["rtp_sidecar"]
    if direct_sidecar is None or recursive_sidecar is None or (
        direct_sidecar["sha256"] != recursive_sidecar["sha256"]
    ):
        raise RTPThreeArmEvaluationError(
            "manifest direct-bridge and recursive sidecars are not checksum-identical"
        )
    expected_profile_contract = {
        "bridge_and_recursive_sizing_profile": R198_SIZING_PROFILE,
        "bridge_and_recursive_max_neural_passes": R198_MAX_NEURAL_PASSES,
        "bridge_and_recursive_max_action_combos": R198_MAX_ACTION_COMBOS,
        "normal_recursive_plan_observed_passes": R198_NORMAL_RECURSIVE_PLAN_PASSES,
        "forced_replan_observed_passes": R198_FORCED_REPLAN_PASSES,
        "absolute_future_max_neural_passes": ABSOLUTE_MAX_NEURAL_PASSES,
        "future_change_requires_distinct_authorized_profile_and_receipt": True,
    }
    if value.get("r198_profile_contract") != expected_profile_contract:
        raise RTPThreeArmEvaluationError("manifest r198 profile contract is not exact")
    normalized_gates = _normalize_gates(value.get("promotion_gates"))
    latency_slo = _normalize_latency_slo(value.get("latency_slo"), normalized_gates)
    if canonical_digest(value.get("latency_slo")) != canonical_digest(latency_slo):
        raise RTPThreeArmEvaluationError("manifest latency SLO binding changed")
    normalized_opponents = _normalize_opponents(value.get("opponents"))
    if canonical_digest(value.get("opponents")) != canonical_digest(normalized_opponents):
        raise RTPThreeArmEvaluationError("manifest opponent package snapshots are not frozen")
    official_control_panel = _validate_official_control_panel(
        _mapping(value.get("shared_artifacts"), "shared_artifacts"), normalized_opponents
    )
    if canonical_digest(value.get("official_control_panel")) != canonical_digest(
        official_control_panel
    ):
        raise RTPThreeArmEvaluationError("manifest official-control panel binding changed")
    source_exclusion_binding = _normalize_r197_source_exclusion_binding(
        _mapping(value.get("shared_artifacts"), "shared_artifacts"),
        {
            "receipt": _mapping(
                value.get("r197_source_exclusion_binding"),
                "r197 source exclusion binding",
            ).get("source_exclusion_proof")
        },
        opponent_content_digests={
            str(row["id"]): str(row["content_digest"])
            for row in normalized_opponents
        },
    )
    if canonical_digest(value.get("r197_source_exclusion_binding")) != canonical_digest(
        source_exclusion_binding
    ):
        raise RTPThreeArmEvaluationError("manifest r197 source-exclusion binding changed")
    pairing_capability = _normalize_pairing_capability(
        {
            "receipt": _mapping(
                value.get("pairing_capability"), "pairing_capability"
            ).get("receipt")
        }
    )
    if canonical_digest(value.get("pairing_capability")) != canonical_digest(
        pairing_capability
    ):
        raise RTPThreeArmEvaluationError("manifest pairing capability is not frozen")
    evaluation_cg_closure = _normalize_evaluation_cg_closure(
        {
            "receipt": _mapping(
                value.get("evaluation_cg_closure"), "evaluation_cg_closure"
            ).get("receipt"),
            "runtime_library": _mapping(
                value.get("evaluation_cg_closure"), "evaluation_cg_closure"
            ).get("runtime_library"),
        },
        pairing_capability,
    )
    if canonical_digest(value.get("evaluation_cg_closure")) != canonical_digest(
        evaluation_cg_closure
    ):
        raise RTPThreeArmEvaluationError("manifest evaluation CG closure is not frozen")
    matchup_adapter_registry = _validate_production_factory_runtime_library(
        production_factory, evaluation_cg_closure
    )
    if canonical_digest(
        _mapping(
            production_factory.get("matchup_adapter_registry"),
            "production_factory.matchup_adapter_registry",
        )
    ) != canonical_digest(matchup_adapter_registry):
        raise RTPThreeArmEvaluationError(
            "manifest snapshot-local matchup adapter registry changed"
        )
    planner_pass_preflight = _normalize_planner_pass_preflight(
        _mapping(value.get("shared_artifacts"), "shared_artifacts"),
        normalized_arms,
        matchup_adapter_registry,
    )
    if canonical_digest(value.get("planner_pass_preflight")) != canonical_digest(
        planner_pass_preflight
    ):
        raise RTPThreeArmEvaluationError("manifest planner pass preflight changed")
    opponent_ids = {str(row["id"]) for row in normalized_opponents}
    evaluation_cases = _case_bindings_with_cell_ids(
        [
            _mapping(row, "evaluation case binding")
            for row in source_exclusion_binding["evaluation_case_bindings"]
        ],
        normalized_opponents,
    )
    if len(evaluation_cases) != OFFICIAL_CONTROL_PAIRED_CELLS:
        raise RTPThreeArmEvaluationError("manifest evaluation cohort lacks the official case panel")
    normalized_rng = _normalize_rng_materials(
        value.get("rng_materials"),
        expected_cases=evaluation_cases,
        pairing_capability=pairing_capability,
        candidate_deck_sha256=_mapping(
            value.get("shared_artifacts"), "shared_artifacts"
        )["deck"]["sha256"],
        opponents={str(row["id"]): row for row in normalized_opponents},
        evaluation_cohort_sha256=source_exclusion_binding["evaluation_only_cohort"][
            "sha256"
        ],
        source_exclusion_proof_sha256=source_exclusion_binding[
            "source_exclusion_proof"
        ]["sha256"],
        evaluation_case_bindings_sha256=source_exclusion_binding[
            "evaluation_case_bindings_sha256"
        ],
    )
    if canonical_digest(value.get("rng_materials")) != canonical_digest(normalized_rng):
        raise RTPThreeArmEvaluationError("manifest snapshot seals are not frozen")
    rng_by_id = {str(row["id"]): row for row in normalized_rng}
    schedule = list(value.get("schedule") or ())
    if len(schedule) != len(rng_by_id):
        raise RTPThreeArmEvaluationError("manifest schedule/RNG material count differs")
    used_rng: set[str] = set()
    used_cases: set[str] = set()
    for index, raw_cell in enumerate(schedule):
        cell = _mapping(raw_cell, f"schedule[{index}]")
        if cell.get("opponent_id") not in opponent_ids:
            raise RTPThreeArmEvaluationError("manifest schedule names an unknown opponent")
        if _integer(cell.get("candidate_seat"), f"schedule[{index}].candidate_seat") not in {0, 1}:
            raise RTPThreeArmEvaluationError("manifest schedule has an invalid candidate seat")
        replicate = _integer(cell.get("replicate"), f"schedule[{index}].replicate")
        evaluation_case = evaluation_cases.get(
            (str(cell["opponent_id"]), int(cell["candidate_seat"]), replicate)
        )
        if evaluation_case is None:
            raise RTPThreeArmEvaluationError("manifest schedule has no frozen evaluation case")
        case_id = _nonempty_text(
            cell.get("evaluation_case_id"), f"schedule[{index}].evaluation_case_id"
        )
        if (
            case_id != evaluation_case["case_id"]
            or cell.get("cell_id") != evaluation_case["cell_id"]
            or case_id in used_cases
        ):
            raise RTPThreeArmEvaluationError("manifest evaluation case identity is invalid or reused")
        if _sha256_value(
            cell.get("evaluation_case_content_digest"),
            f"schedule[{index}].evaluation_case_content_digest",
        ) != evaluation_case["content_digest"]:
            raise RTPThreeArmEvaluationError("manifest evaluation case content digest mismatch")
        if _sha256_value(
            cell.get("evaluation_case_bindings_sha256"),
            f"schedule[{index}].evaluation_case_bindings_sha256",
        ) != source_exclusion_binding["evaluation_case_bindings_sha256"]:
            raise RTPThreeArmEvaluationError("manifest evaluation case binding digest mismatch")
        used_cases.add(case_id)
        rng = _mapping(cell.get("rng_identity"), f"schedule[{index}].rng_identity")
        material = rng_by_id.get(str(rng.get("id") or ""))
        if material is None:
            raise RTPThreeArmEvaluationError("manifest schedule names an unknown RNG material")
        expected_rng = {
            "id": material["id"],
            "kind": material["kind"],
            "sha256": material["snapshot_artifact"]["sha256"],
            "bytes": material["snapshot_artifact"]["bytes"],
            "seal_sha256": material["seal"]["sha256"],
            "capture_boundary": pairing_capability["abi"]["capture_boundary"],
            "boundary_tag": pairing_capability["abi"]["boundary_tag"],
        }
        if rng != expected_rng:
            raise RTPThreeArmEvaluationError("manifest schedule RNG identity is not frozen")
        if expected_rng["id"] in used_rng:
            raise RTPThreeArmEvaluationError("manifest reuses an RNG tape/snapshot")
        used_rng.add(expected_rng["id"])
    if len(schedule) != OFFICIAL_CONTROL_PAIRED_CELLS or len(used_cases) != len(evaluation_cases):
        raise RTPThreeArmEvaluationError("manifest does not schedule every official evaluation case")


def _score(value: Any, label: str) -> float:
    parsed = _number(value, label)
    if parsed not in {0.0, 0.5, 1.0}:
        raise RTPThreeArmEvaluationError(f"{label} must be one of 0, 0.5, 1")
    return parsed


def _terminal_candidate_score(
    raw: Any, label: str, *, candidate_seat: int
) -> tuple[dict[str, Any], float]:
    """Derive expected score from a terminal engine result, never a label."""

    value = _mapping(raw, label)
    value = _mapping(value.get("terminal_outcome"), f"{label}.terminal_outcome")
    winner = _nonempty_text(value.get("winner"), f"{label}.terminal_outcome.winner")
    code = _integer(
        value.get("engine_result_code"),
        f"{label}.terminal_outcome.engine_result_code",
        minimum=0,
    )
    if candidate_seat not in {0, 1}:
        raise RTPThreeArmEvaluationError(f"{label} candidate_seat must be 0 or 1")
    mapping = {
        "candidate": (candidate_seat, 1.0),
        "opponent": (1 - candidate_seat, 0.0),
        "draw": (2, 0.5),
    }
    if winner not in mapping or mapping[winner][0] != code:
        raise RTPThreeArmEvaluationError(
            f"{label} terminal outcome must map candidate/opponent/draw to 1/0/2"
        )
    if value.get("termination") != "completed" or value.get("failed_seat") is not None:
        raise RTPThreeArmEvaluationError(
            f"{label} terminal outcome is not a completed non-failed game"
        )
    for field in ("engine_error", "candidate_error", "opponent_error"):
        if value.get(field) not in {None, ""}:
            raise RTPThreeArmEvaluationError(
                f"{label} terminal outcome recorded {field}"
            )
    if not isinstance(value.get("candidate_forfeit"), bool):
        raise RTPThreeArmEvaluationError(f"{label}.terminal_outcome.candidate_forfeit must be boolean")
    return {
        "winner": winner,
        "engine_result_code": code,
        "candidate_forfeit": bool(value["candidate_forfeit"]),
        "candidate_seat": candidate_seat,
        "termination": "completed",
        "failed_seat": None,
    }, mapping[winner][1]


def _forced_turn_order_control_trace(raw: Any, *, label: str) -> list[dict[str, Any]]:
    """Validate external IsFirst/Yes controls outside policy telemetry.

    These are engine-mandated controls.  They must retain their own immutable
    observation/action attestation, but cannot appear in candidate decision,
    complexity, bridge, recursive, fallback, or latency accounting.
    """

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPThreeArmEvaluationError(
            f"{label}.forced_turn_order_control_trace must be a list"
        )
    required_fields = {
        "control_index",
        "control",
        "prompt_context",
        "prompt_context_encoding",
        "expected_action",
        "returned_action",
        "verified_observation_action_contract",
        "rtp_diagnostics_absent",
        "complexity_probe_not_invoked",
        "excluded_from_candidate_decisions",
        "excluded_from_intended_complex_denominator",
        "excluded_from_latency",
    }
    normalized: list[dict[str, Any]] = []
    for index, raw_trace in enumerate(raw):
        trace = _mapping(
            raw_trace, f"{label}.forced_turn_order_control_trace[{index}]"
        )
        if set(trace) != required_fields:
            raise RTPThreeArmEvaluationError(
                f"{label} forced turn-order trace must have the exact canonical field set"
            )
        if type(trace.get("control_index")) is not int or trace["control_index"] != index:
            raise RTPThreeArmEvaluationError(
                f"{label} forced turn-order control indexes must be monotonic from zero"
            )
        if trace.get("control") != FORCED_TURN_ORDER_CONTROL:
            raise RTPThreeArmEvaluationError(
                f"{label} forced turn-order trace has an unknown control"
            )
        encoding = _nonempty_text(
            trace.get("prompt_context_encoding"),
            f"{label}.forced_turn_order_control_trace[{index}].prompt_context_encoding",
        )
        context = trace.get("prompt_context")
        if encoding == "numeric_41":
            if (
                type(context) is not int
                or context != 41
            ):
                raise RTPThreeArmEvaluationError(
                    f"{label} forced turn-order trace has an invalid prompt context"
                )
            context = 41
        elif (
            encoding == "enum_is_first"
            and type(context) is str
            and context == "IsFirst"
        ):
            context = "IsFirst"
        else:
            raise RTPThreeArmEvaluationError(
                f"{label} forced turn-order trace has an invalid prompt context"
            )

        def _action(field: str) -> list[int]:
            value = trace.get(field)
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 1
                or type(value[0]) is not int
                or value[0] < 0
            ):
                raise RTPThreeArmEvaluationError(
                    f"{label}.forced_turn_order_control_trace[{index}].{field} "
                    "must contain exactly one action index"
                )
            return [value[0]]

        expected_action = _action("expected_action")
        returned_action = _action("returned_action")
        if expected_action != returned_action:
            raise RTPThreeArmEvaluationError(
                f"{label} forced turn-order action does not equal its expected Yes action"
            )
        for field in (
            "verified_observation_action_contract",
            "rtp_diagnostics_absent",
            "complexity_probe_not_invoked",
            "excluded_from_candidate_decisions",
            "excluded_from_intended_complex_denominator",
            "excluded_from_latency",
        ):
            if trace.get(field) is not True:
                raise RTPThreeArmEvaluationError(
                    f"{label} forced turn-order trace must attest {field}"
                )
        normalized.append(
            {
                "control_index": index,
                "control": FORCED_TURN_ORDER_CONTROL,
                "prompt_context": context,
                "prompt_context_encoding": encoding,
                "expected_action": expected_action,
                "returned_action": returned_action,
                "verified_observation_action_contract": True,
                "rtp_diagnostics_absent": True,
                "complexity_probe_not_invoked": True,
                "excluded_from_candidate_decisions": True,
                "excluded_from_intended_complex_denominator": True,
                "excluded_from_latency": True,
            }
        )
    if len(normalized) > 1:
        raise RTPThreeArmEvaluationError(
            f"{label} has more than one forced turn-order control"
        )
    return normalized


def _validate_r198_forced_turn_order_contract(
    telemetry: Mapping[str, Any], *, candidate_seat: int, label: str
) -> None:
    """Bind the external control to the frozen snapshot's physical seat ABI.

    The sealed r198 snapshot is captured exactly at physical player 0's
    ``IsFirst`` prompt, where numeric context 41 offers ``Yes`` at option 0.
    Candidate seat 0 therefore has exactly one external control; candidate
    seat 1 has none because the opponent consumes it.  This is intentionally
    stronger than cross-arm equality, which alone would allow all-zero rows.
    """

    if candidate_seat not in {0, 1}:
        raise RTPThreeArmEvaluationError(f"{label} has an invalid candidate seat")
    expected_count = 1 if candidate_seat == 0 else 0
    observed_count = _integer(
        telemetry.get("forced_turn_order_controls"),
        f"{label}.forced_turn_order_controls",
    )
    trace = telemetry.get("forced_turn_order_control_trace")
    if observed_count != expected_count:
        raise RTPThreeArmEvaluationError(
            f"{label} forced turn-order control count does not match the frozen seat ABI"
        )
    if candidate_seat == 1:
        if trace != []:
            raise RTPThreeArmEvaluationError(
                f"{label} candidate seat 1 must not record the physical-seat-0 control"
            )
        return
    expected_trace = [
        {
            "control_index": 0,
            "control": FORCED_TURN_ORDER_CONTROL,
            "prompt_context": 41,
            "prompt_context_encoding": "numeric_41",
            "expected_action": [0],
            "returned_action": [0],
            "verified_observation_action_contract": True,
            "rtp_diagnostics_absent": True,
            "complexity_probe_not_invoked": True,
            "excluded_from_candidate_decisions": True,
            "excluded_from_intended_complex_denominator": True,
            "excluded_from_latency": True,
        }
    ]
    if trace != expected_trace:
        raise RTPThreeArmEvaluationError(
            f"{label} candidate seat 0 lacks the exact frozen IsFirst/Yes control"
        )


def _exact_nonbool_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    """Reject coercible JSON scalars in the over-cap evidence contract."""

    if type(value) is not int or value < minimum:
        raise RTPThreeArmEvaluationError(f"{label} must be an exact non-bool integer")
    return value


def _over_cap_action_space(raw: Any, *, label: str) -> dict[str, Any]:
    """Independently recompute the non-materializing action-space summary."""

    value = _mapping(raw, label)
    if set(value) != _OVER_CAP_ACTION_SPACE_FIELDS:
        raise RTPThreeArmEvaluationError(f"{label} has an invalid exact field set")
    n_options = _exact_nonbool_integer(value.get("n_options"), f"{label}.n_options")
    min_count = _exact_nonbool_integer(value.get("min_count"), f"{label}.min_count")
    max_count = _exact_nonbool_integer(value.get("max_count"), f"{label}.max_count")
    cap = _exact_nonbool_integer(
        value.get("complete_ordered_action_cap"),
        f"{label}.complete_ordered_action_cap",
    )
    cardinality = _exact_nonbool_integer(
        value.get("complete_ordered_action_cardinality"),
        f"{label}.complete_ordered_action_cardinality",
    )
    if cap != R198_MAX_ACTION_COMBOS:
        raise RTPThreeArmEvaluationError(f"{label} cap differs from exact r198 1024")
    if not (0 <= min_count <= max_count <= n_options):
        raise RTPThreeArmEvaluationError(f"{label} has invalid exact selection bounds")
    counts_raw = value.get("counts")
    expected_counts = list(range(min_count, max_count + 1))
    if (
        not isinstance(counts_raw, Sequence)
        or isinstance(counts_raw, (str, bytes))
        or any(type(item) is not int for item in counts_raw)
        or list(counts_raw) != expected_counts
    ):
        raise RTPThreeArmEvaluationError(f"{label} counts are not exact")
    expected_cardinality = sum(math.perm(n_options, count) for count in expected_counts)
    if cardinality != expected_cardinality:
        raise RTPThreeArmEvaluationError(f"{label} cardinality does not recompute")
    if value.get("over_cap") is not True or cardinality <= cap:
        raise RTPThreeArmEvaluationError(f"{label} does not attest a true over-cap space")
    if value.get("complete_ordered_actions_materialized") is not False:
        raise RTPThreeArmEvaluationError(f"{label} attests complete action materialization")
    if value.get("complete_ordered_action_truncated") is not False:
        raise RTPThreeArmEvaluationError(f"{label} attests truncation")
    return {
        "n_options": n_options,
        "min_count": min_count,
        "max_count": max_count,
        "counts": expected_counts,
        "complete_ordered_action_cardinality": cardinality,
        "complete_ordered_action_cap": cap,
        "over_cap": True,
        "complete_ordered_actions_materialized": False,
        "complete_ordered_action_truncated": False,
    }


def _over_cap_factorized_action_stage_count(
    action: Any, action_space: Mapping[str, Any], *, label: str
) -> tuple[list[int], int]:
    """Recompute factorized legality from exact bounds, without enumeration."""

    if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
        raise RTPThreeArmEvaluationError(f"{label} returned_action must be a sequence")
    selected = list(action)
    if any(type(item) is not int for item in selected):
        raise RTPThreeArmEvaluationError(f"{label} returned_action has a non-exact index")
    n_options = int(action_space["n_options"])
    min_count = int(action_space["min_count"])
    max_count = int(action_space["max_count"])
    if (
        len(selected) < min_count
        or len(selected) > max_count
        or len(selected) != len(set(selected))
        or any(item < 0 or item >= n_options for item in selected)
    ):
        raise RTPThreeArmEvaluationError(f"{label} returned_action is not factorized-legal")
    # `factorized_teacher_forcing_stages` emits one selected-prefix stage per
    # option plus STOP when maxCount has not yet been reached; an empty legal
    # action still has its one explicit STOP stage.
    stages = len(selected) + int(len(selected) < max_count)
    return selected, max(stages, 1)


def _over_cap_bridge_diagnostic(raw: Any, *, arm: str, label: str) -> dict[str, Any] | None:
    if arm == "no_rtp":
        if raw is not None:
            raise RTPThreeArmEvaluationError(f"{label} no-RTP over-cap trace has diagnostics")
        return None
    value = _mapping(raw, f"{label}.rtp_diagnostic")
    expected = {
        "mode": "fallback",
        "fallback_code": "action_space_too_large",
        "neural_passes": 0,
        "required_neural_passes": 0,
        "legal_count": 0,
        "decision_mode": "",
    }
    if set(value) != set(expected):
        raise RTPThreeArmEvaluationError(f"{label} RTP diagnostic has an invalid field set")
    for key, expected_value in expected.items():
        actual = value.get(key)
        if isinstance(expected_value, int):
            if type(actual) is not int or actual != expected_value:
                raise RTPThreeArmEvaluationError(f"{label} RTP diagnostic differs at {key}")
        elif actual != expected_value:
            raise RTPThreeArmEvaluationError(f"{label} RTP diagnostic differs at {key}")
    return expected


def _over_cap_factorized_fallback_trace(
    raw_trace: Any,
    declared_digest: Any,
    *,
    arm: str,
    label: str,
) -> list[dict[str, Any]]:
    """Normalize dedicated over-cap trace rows and bind their digest."""

    if not isinstance(raw_trace, Sequence) or isinstance(raw_trace, (str, bytes)):
        raise RTPThreeArmEvaluationError(f"{label} lacks an over-cap factorized trace")
    supplied = list(raw_trace)
    if _sha256_value(declared_digest, f"{label}.over_cap_factorized_fallback_trace_sha256") != canonical_digest(supplied):
        raise RTPThreeArmEvaluationError(f"{label} over-cap factorized trace digest differs")
    normalized: list[dict[str, Any]] = []
    decision_indexes: set[int] = set()
    exact_true_fields = (
        "factorized_teacher_forcing_legal",
        "complexity_probe_not_invoked",
        "included_in_candidate_decisions",
        "included_in_candidate_latency",
        "excluded_from_planner_eligible_candidate_decisions",
        "excluded_from_intended_complex_denominator",
        "excluded_from_direct_bridge_metrics",
        "excluded_from_recursive_metrics",
        "excluded_from_fallback_metrics",
        "excluded_from_neural_pass_metrics",
        "excluded_from_recursive_latency",
    )
    for position, raw in enumerate(supplied):
        value = _mapping(raw, f"{label}.over_cap_factorized_fallback_trace[{position}]")
        if set(value) != _OVER_CAP_TRACE_FIELDS:
            raise RTPThreeArmEvaluationError(
                f"{label} over-cap factorized trace has an invalid exact field set"
            )
        decision_index = _exact_nonbool_integer(
            value.get("decision_index"),
            f"{label}.over_cap_factorized_fallback_trace[{position}].decision_index",
        )
        if decision_index in decision_indexes:
            raise RTPThreeArmEvaluationError(f"{label} repeats an over-cap decision index")
        decision_indexes.add(decision_index)
        if value.get("arm") != arm or value.get("mode") != OVER_CAP_FACTORIZED_FALLBACK_MODE:
            raise RTPThreeArmEvaluationError(f"{label} over-cap trace has an incompatible arm/mode")
        if value.get("classification") != OVER_CAP_FACTORIZED_FALLBACK_REASON:
            raise RTPThreeArmEvaluationError(f"{label} over-cap trace has an invalid classification")
        action_space = _over_cap_action_space(
            value.get("action_space"),
            label=f"{label}.over_cap_factorized_fallback_trace[{position}].action_space",
        )
        action_space_sha256 = _sha256_value(
            value.get("action_space_sha256"),
            f"{label}.over_cap_factorized_fallback_trace[{position}].action_space_sha256",
        )
        if action_space_sha256 != canonical_digest(action_space):
            raise RTPThreeArmEvaluationError(f"{label} over-cap action-space digest does not recompute")
        observation_sha256 = _sha256_value(
            value.get("observation_sha256"),
            f"{label}.over_cap_factorized_fallback_trace[{position}].observation_sha256",
        )
        policy_input_sha256 = _sha256_value(
            value.get("candidate_policy_input_sha256"),
            f"{label}.over_cap_factorized_fallback_trace[{position}].candidate_policy_input_sha256",
        )
        logical_pre_action_sha256 = _sha256_value(
            value.get("logical_pre_action_sha256"),
            f"{label}.over_cap_factorized_fallback_trace[{position}].logical_pre_action_sha256",
        )
        if logical_pre_action_sha256 != canonical_digest(
            {
                "observation_sha256": observation_sha256,
                "action_space_sha256": action_space_sha256,
                "candidate_policy_input_sha256": policy_input_sha256,
            }
        ):
            raise RTPThreeArmEvaluationError(f"{label} over-cap logical input digest does not recompute")
        selected, stage_count = _over_cap_factorized_action_stage_count(
            value.get("returned_action"), action_space, label=f"{label}.over-cap trace"
        )
        if (
            value.get("factorized_teacher_forcing_legal") is not True
            or _exact_nonbool_integer(
                value.get("factorized_teacher_forcing_stage_count"),
                f"{label}.over-cap stage count",
            ) != stage_count
        ):
            raise RTPThreeArmEvaluationError(f"{label} over-cap factorized legality attestation differs")
        for field in exact_true_fields:
            if value.get(field) is not True:
                raise RTPThreeArmEvaluationError(f"{label} over-cap trace lacks {field}")
        if value.get("neural_budget_failure") is not False:
            raise RTPThreeArmEvaluationError(f"{label} over-cap trace has a neural budget failure")
        for field in ("neural_passes", "required_neural_passes"):
            if _exact_nonbool_integer(value.get(field), f"{label}.over-cap {field}") != 0:
                raise RTPThreeArmEvaluationError(f"{label} over-cap trace spent neural passes")
        diagnostic = _over_cap_bridge_diagnostic(value.get("rtp_diagnostic"), arm=arm, label=label)
        normalized.append(
            {
                "decision_index": decision_index,
                "arm": arm,
                "mode": OVER_CAP_FACTORIZED_FALLBACK_MODE,
                "classification": OVER_CAP_FACTORIZED_FALLBACK_REASON,
                "action_space": action_space,
                "action_space_sha256": action_space_sha256,
                "observation_sha256": observation_sha256,
                "candidate_policy_input_sha256": policy_input_sha256,
                "logical_pre_action_sha256": logical_pre_action_sha256,
                "returned_action": selected,
                "factorized_teacher_forcing_legal": True,
                "factorized_teacher_forcing_stage_count": stage_count,
                "complexity_probe_not_invoked": True,
                "neural_passes": 0,
                "required_neural_passes": 0,
                "neural_budget_failure": False,
                "rtp_diagnostic": diagnostic,
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
    return normalized


def _result_telemetry(raw: Any, *, arm: str, label: str) -> dict[str, Any]:
    value = _mapping(raw, f"{label}.telemetry")
    fields = (
        "candidate_decisions",
        "planner_eligible_candidate_decisions",
        "over_cap_factorized_fallback_decisions",
        "forced_turn_order_controls",
        "intended_complex_decisions",
        "recursive_intended_complex_decisions",
        "successful_recursive_intended_complex_decisions",
        "direct_bridge_decisions",
        "recursive_decisions",
        "fallback_decisions",
        "unexpected_recursive_fallback_decisions",
        "expected_recursive_fallback_decisions",
        "neural_budget_exceeded",
        "neural_budget_failures",
        "illegal_action_count",
        "candidate_forfeit_count",
    )
    metrics = {key: _integer(value.get(key), f"{label}.telemetry.{key}") for key in fields}
    for key in (
        "candidate_decisions",
        "planner_eligible_candidate_decisions",
        "over_cap_factorized_fallback_decisions",
    ):
        metrics[key] = _exact_nonbool_integer(
            value.get(key), f"{label}.telemetry.{key}"
        )
    if (
        metrics["candidate_decisions"]
        != metrics["planner_eligible_candidate_decisions"]
        + metrics["over_cap_factorized_fallback_decisions"]
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} candidate decisions must equal planner-eligible plus over-cap decisions"
        )
    over_cap_trace = _over_cap_factorized_fallback_trace(
        value.get("over_cap_factorized_fallback_trace"),
        value.get("over_cap_factorized_fallback_trace_sha256"),
        arm=arm,
        label=f"{label}.telemetry",
    )
    if len(over_cap_trace) != metrics["over_cap_factorized_fallback_decisions"]:
        raise RTPThreeArmEvaluationError(
            f"{label} over-cap counter does not match its dedicated trace"
        )
    forced_turn_order_trace = _forced_turn_order_control_trace(
        value.get("forced_turn_order_control_trace"), label=f"{label}.telemetry"
    )
    if metrics["forced_turn_order_controls"] != len(forced_turn_order_trace):
        raise RTPThreeArmEvaluationError(
            f"{label} forced turn-order control count does not match its trace"
        )
    if _nonempty_text(
        value.get("intended_complex_decision_scope"),
        f"{label}.telemetry.intended_complex_decision_scope",
    ) != INTENDED_COMPLEX_DECISION_SCOPE:
        raise RTPThreeArmEvaluationError(
            f"{label} must use intended-complex scope "
            f"{INTENDED_COMPLEX_DECISION_SCOPE!r}"
        )
    if (
        metrics["successful_recursive_intended_complex_decisions"]
        != metrics["recursive_intended_complex_decisions"]
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} successful recursive intended-complex count must equal the canonical count"
        )
    recursive_mode_counts_raw = _mapping(
        value.get("recursive_mode_counts"), f"{label}.telemetry.recursive_mode_counts"
    )
    required_planner_modes = tuple(sorted(SUCCESSFUL_RECURSIVE_MODES | RECURSIVE_FALLBACK_MODES))
    if set(recursive_mode_counts_raw) != set(required_planner_modes):
        raise RTPThreeArmEvaluationError(
            f"{label}.telemetry.recursive_mode_counts must name every canonical planner mode"
        )
    recursive_mode_counts = {
        mode: _integer(
            recursive_mode_counts_raw[mode],
            f"{label}.telemetry.recursive_mode_counts.{mode}",
        )
        for mode in required_planner_modes
    }
    for key in (
        "intended_complex_decisions",
        "direct_bridge_decisions",
        "recursive_decisions",
        "fallback_decisions",
        "illegal_action_count",
    ):
        if metrics[key] > metrics["candidate_decisions"]:
            raise RTPThreeArmEvaluationError(
                f"{label} {key} exceeds total candidate decisions"
            )
    if metrics["candidate_forfeit_count"] > 1:
        raise RTPThreeArmEvaluationError(
            f"{label} candidate_forfeit_count must be zero or one per game"
        )
    if metrics["recursive_intended_complex_decisions"] > metrics[
        "intended_complex_decisions"
    ] or metrics["recursive_intended_complex_decisions"] > metrics[
        "recursive_decisions"
    ]:
        raise RTPThreeArmEvaluationError(
            f"{label} recursive intended-complex decisions are inconsistent"
        )
    if metrics["unexpected_recursive_fallback_decisions"] > metrics[
        "intended_complex_decisions"
    ]:
        raise RTPThreeArmEvaluationError(
            f"{label} unexpected recursive fallbacks exceed their intended-complex denominator"
        )
    if (
        metrics["unexpected_recursive_fallback_decisions"]
        + metrics["expected_recursive_fallback_decisions"]
        > metrics["fallback_decisions"]
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} classified recursive fallbacks exceed total fallbacks"
        )
    if metrics["neural_budget_exceeded"] != metrics["neural_budget_failures"]:
        raise RTPThreeArmEvaluationError(
            f"{label} neural budget failure counters disagree"
        )
    metrics["latency_seconds"] = _number(
        value.get("latency_seconds"), f"{label}.telemetry.latency_seconds", minimum=0.0
    )
    latency_trace_raw = value.get("decision_latency_trace")
    if not isinstance(latency_trace_raw, Sequence) or isinstance(
        latency_trace_raw, (str, bytes)
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} must record a mode-labeled decision_latency_trace"
        )
    if len(latency_trace_raw) != metrics["candidate_decisions"]:
        raise RTPThreeArmEvaluationError(
            f"{label} latency trace must contain one entry per candidate decision"
        )
    allowed_latency_modes = {
        "no_rtp",
        DIRECT_BRIDGE_ARM,
        "recursive_rtp",
        "fallback",
        OVER_CAP_FACTORIZED_FALLBACK_MODE,
    }
    coarse_mode_by_planner_mode = {
        "no_rtp": "no_rtp",
        "direct_bridge": DIRECT_BRIDGE_ARM,
        NONRECURSIVE_DIRECT_POLICY_MODE: DIRECT_BRIDGE_ARM,
        "recursive_plan": "recursive_rtp",
        "continue_plan": "recursive_rtp",
        "replan_with_program": "recursive_rtp",
        "direct_policy_fallback": "fallback",
        "replan_direct": "fallback",
        OVER_CAP_FACTORIZED_FALLBACK_MODE: OVER_CAP_FACTORIZED_FALLBACK_MODE,
    }
    latency_trace: list[dict[str, Any]] = []
    recursive_latencies: list[float] = []
    observed_mode_counts = {mode: 0 for mode in required_planner_modes}
    observed_intended_complex = 0
    observed_recursive_intended_complex = 0
    observed_fallback_classification = {"expected": 0, "unexpected": 0}
    observed_direct_bridge = 0
    observed_over_cap_indexes: set[int] = set()
    over_cap_by_decision = {item["decision_index"]: item for item in over_cap_trace}
    for expected_index, raw_trace in enumerate(latency_trace_raw):
        trace = _mapping(raw_trace, f"{label}.telemetry.decision_latency_trace[{expected_index}]")
        if _integer(
            trace.get("decision_index"),
            f"{label}.telemetry.decision_latency_trace[{expected_index}].decision_index",
            minimum=0,
        ) != expected_index:
            raise RTPThreeArmEvaluationError(
                f"{label} latency trace decision indexes must be monotonic from zero"
            )
        mode = _canonical_arm_name(
            trace.get("mode"),
            f"{label}.telemetry.decision_latency_trace[{expected_index}].mode",
        )
        if mode not in allowed_latency_modes:
            raise RTPThreeArmEvaluationError(f"{label} latency trace has an invalid mode")
        planner_mode = _nonempty_text(
            trace.get("planner_mode"),
            f"{label}.telemetry.decision_latency_trace[{expected_index}].planner_mode",
        )
        if planner_mode not in coarse_mode_by_planner_mode:
            raise RTPThreeArmEvaluationError(
                f"{label} latency trace has an invalid planner_mode"
            )
        if mode != coarse_mode_by_planner_mode[planner_mode]:
            raise RTPThreeArmEvaluationError(
                f"{label} latency trace coarse mode disagrees with planner_mode"
            )
        planner_reason = _nonempty_text(
            trace.get("planner_reason"),
            f"{label}.telemetry.decision_latency_trace[{expected_index}].planner_reason",
        )
        intended_complex = trace.get("intended_complex")
        fallback_classification = trace.get("fallback_classification")
        if planner_mode == OVER_CAP_FACTORIZED_FALLBACK_MODE:
            if set(trace) != {
                "decision_index",
                "mode",
                "planner_mode",
                "planner_reason",
                "intended_complex",
                "fallback_classification",
                "latency_seconds",
                "over_cap_trace_index",
            }:
                raise RTPThreeArmEvaluationError(
                    f"{label} over-cap latency trace has an invalid exact field set"
                )
            trace_index = _exact_nonbool_integer(
                trace.get("over_cap_trace_index"),
                f"{label}.telemetry.decision_latency_trace[{expected_index}].over_cap_trace_index",
            )
            if trace_index >= len(over_cap_trace):
                raise RTPThreeArmEvaluationError(f"{label} over-cap latency trace index is out of range")
            special = over_cap_trace[trace_index]
            if special["decision_index"] != expected_index:
                raise RTPThreeArmEvaluationError(
                    f"{label} over-cap latency trace does not bind its special trace"
                )
            if planner_reason != OVER_CAP_FACTORIZED_FALLBACK_REASON:
                raise RTPThreeArmEvaluationError(f"{label} over-cap latency trace has an invalid reason")
            # No common complexity predicate ran for this stratum.  It is
            # therefore unassessed, not an ordinary direct/``False`` result.
            if intended_complex is not None or fallback_classification is not None:
                raise RTPThreeArmEvaluationError(
                    f"{label} over-cap latency trace entered planner/fallback accounting"
                )
            observed_over_cap_indexes.add(expected_index)
        else:
            if not isinstance(intended_complex, bool):
                raise RTPThreeArmEvaluationError(
                    f"{label} latency trace intended_complex must be boolean"
                )
            if "over_cap_trace_index" in trace:
                raise RTPThreeArmEvaluationError(
                    f"{label} ordinary latency trace carries an over-cap trace index"
                )
        if planner_mode in RECURSIVE_FALLBACK_MODES:
            if fallback_classification not in {"expected", "unexpected"}:
                raise RTPThreeArmEvaluationError(
                    f"{label} fallback latency trace must classify expected/unexpected"
                )
            if not intended_complex:
                raise RTPThreeArmEvaluationError(
                    f"{label} recursive fallback must arise from an intended-complex decision"
                )
            observed_fallback_classification[str(fallback_classification)] += 1
        elif fallback_classification not in {None, ""}:
            raise RTPThreeArmEvaluationError(
                f"{label} non-fallback latency trace may not declare a fallback classification"
            )
        latency = _number(
            trace.get("latency_seconds"),
            f"{label}.telemetry.decision_latency_trace[{expected_index}].latency_seconds",
            minimum=0.0,
        )
        normalized_trace = {
            "decision_index": expected_index,
            "mode": mode,
            "planner_mode": planner_mode,
            "planner_reason": planner_reason,
            "intended_complex": intended_complex,
            "fallback_classification": fallback_classification,
            "latency_seconds": latency,
        }
        if planner_mode == OVER_CAP_FACTORIZED_FALLBACK_MODE:
            normalized_trace["over_cap_trace_index"] = _exact_nonbool_integer(
                trace.get("over_cap_trace_index"),
                f"{label}.telemetry.decision_latency_trace[{expected_index}].over_cap_trace_index",
            )
        latency_trace.append(normalized_trace)
        if planner_mode in SUCCESSFUL_RECURSIVE_MODES:
            recursive_latencies.append(latency)
            observed_mode_counts[planner_mode] += 1
            if intended_complex:
                observed_recursive_intended_complex += 1
        elif planner_mode in RECURSIVE_FALLBACK_MODES:
            observed_mode_counts[planner_mode] += 1
        elif planner_mode in {"direct_bridge", NONRECURSIVE_DIRECT_POLICY_MODE}:
            observed_direct_bridge += 1
        if intended_complex:
            observed_intended_complex += 1
    if len(recursive_latencies) != metrics["recursive_decisions"] or sum(
        observed_mode_counts[mode] for mode in SUCCESSFUL_RECURSIVE_MODES
    ) != metrics["recursive_decisions"]:
        raise RTPThreeArmEvaluationError(
            f"{label} recursive decision count does not match its latency trace"
        )
    if observed_mode_counts != recursive_mode_counts:
        raise RTPThreeArmEvaluationError(
            f"{label} recursive_mode_counts do not match its decision trace"
        )
    if observed_intended_complex != metrics["intended_complex_decisions"]:
        raise RTPThreeArmEvaluationError(
            f"{label} intended-complex decisions do not match their decision trace"
        )
    if observed_recursive_intended_complex != metrics["recursive_intended_complex_decisions"]:
        raise RTPThreeArmEvaluationError(
            f"{label} successful recursive intended-complex decisions do not match their trace"
        )
    if sum(observed_mode_counts[mode] for mode in RECURSIVE_FALLBACK_MODES) != metrics[
        "fallback_decisions"
    ]:
        raise RTPThreeArmEvaluationError(f"{label} fallback count does not match its trace")
    if observed_fallback_classification["unexpected"] != metrics[
        "unexpected_recursive_fallback_decisions"
    ] or observed_fallback_classification["expected"] != metrics[
        "expected_recursive_fallback_decisions"
    ]:
        raise RTPThreeArmEvaluationError(
            f"{label} recursive fallback classifications do not match their trace"
        )
    if observed_direct_bridge != metrics["direct_bridge_decisions"]:
        raise RTPThreeArmEvaluationError(
            f"{label} direct-bridge decision count does not match its trace"
        )
    if observed_over_cap_indexes != set(over_cap_by_decision):
        raise RTPThreeArmEvaluationError(
            f"{label} over-cap dedicated trace does not match its latency entries"
        )
    if (
        metrics["planner_eligible_candidate_decisions"]
        != metrics["candidate_decisions"] - len(observed_over_cap_indexes)
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} planner-eligible counter does not exclude exactly the over-cap decisions"
        )
    allowed_planner_modes_by_arm = {
        "no_rtp": {"no_rtp", OVER_CAP_FACTORIZED_FALLBACK_MODE},
        DIRECT_BRIDGE_ARM: {"direct_bridge", OVER_CAP_FACTORIZED_FALLBACK_MODE},
        "recursive_rtp": (
            SUCCESSFUL_RECURSIVE_MODES
            | RECURSIVE_FALLBACK_MODES
            | {NONRECURSIVE_DIRECT_POLICY_MODE, OVER_CAP_FACTORIZED_FALLBACK_MODE}
        ),
    }
    if any(
        trace["planner_mode"] not in allowed_planner_modes_by_arm[arm]
        for trace in latency_trace
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} latency trace planner mode is incompatible with {arm}"
        )
    if not math.isclose(
        metrics["latency_seconds"],
        sum(trace["latency_seconds"] for trace in latency_trace),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} latency_seconds must equal the candidate decision trace total"
        )
    metrics["decision_latency_trace"] = latency_trace
    metrics["recursive_decision_latency_seconds"] = recursive_latencies
    normal_raw = value.get("normal_recursive_plan_passes")
    forced_raw = value.get("forced_replan_passes")
    if (
        not isinstance(normal_raw, Sequence)
        or isinstance(normal_raw, (str, bytes))
        or not isinstance(forced_raw, Sequence)
        or isinstance(forced_raw, (str, bytes))
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} must record normal_recursive_plan_passes and forced_replan_passes"
        )
    normal_passes = [
        _integer(pass_count, f"{label}.telemetry.normal_recursive_plan_passes")
        for pass_count in normal_raw
    ]
    forced_passes = [
        _integer(pass_count, f"{label}.telemetry.forced_replan_passes")
        for pass_count in forced_raw
    ]
    metrics["normal_recursive_plans"] = len(normal_passes)
    metrics["forced_replans"] = len(forced_passes)
    if arm == "recursive_rtp":
        if any(pass_count != R198_NORMAL_RECURSIVE_PLAN_PASSES for pass_count in normal_passes):
            raise RTPThreeArmEvaluationError(
                f"{label} normal recursive plans must consume exactly "
                f"{R198_NORMAL_RECURSIVE_PLAN_PASSES} passes"
            )
        if any(pass_count != R198_FORCED_REPLAN_PASSES for pass_count in forced_passes):
            raise RTPThreeArmEvaluationError(
                f"{label} forced replans must consume exactly "
                f"{R198_FORCED_REPLAN_PASSES} passes"
            )
        if (
            metrics["unexpected_recursive_fallback_decisions"]
            + metrics["expected_recursive_fallback_decisions"]
            != metrics["fallback_decisions"]
        ):
            raise RTPThreeArmEvaluationError(
                f"{label} must classify every recursive fallback as expected or unexpected"
            )
    elif normal_passes or forced_passes:
        raise RTPThreeArmEvaluationError(
            f"{label} recorded recursive planner passes outside recursive_rtp"
        )
    if arm == "no_rtp" and (
        metrics["direct_bridge_decisions"]
        or metrics["recursive_decisions"]
        or metrics["recursive_intended_complex_decisions"]
        or metrics["neural_budget_exceeded"]
        or metrics["neural_budget_failures"]
    ):
        raise RTPThreeArmEvaluationError("no_rtp row recorded RTP activity")
    if arm == DIRECT_BRIDGE_ARM and (
        metrics["recursive_decisions"]
        or metrics["recursive_intended_complex_decisions"]
        or metrics["neural_budget_exceeded"]
        or metrics["neural_budget_failures"]
    ):
        raise RTPThreeArmEvaluationError(
            "direct_bridge_recursive_disabled row recorded recursive activity"
        )
    if arm != "recursive_rtp" and (
        metrics["unexpected_recursive_fallback_decisions"]
        or metrics["expected_recursive_fallback_decisions"]
    ):
        raise RTPThreeArmEvaluationError(
            f"{label} classified a recursive fallback outside recursive_rtp"
        )
    metrics["intended_complex_decision_scope"] = INTENDED_COMPLEX_DECISION_SCOPE
    metrics["recursive_mode_counts"] = recursive_mode_counts
    metrics["forced_turn_order_control_trace"] = forced_turn_order_trace
    metrics["over_cap_factorized_fallback_trace"] = over_cap_trace
    metrics["over_cap_factorized_fallback_trace_sha256"] = canonical_digest(
        over_cap_trace
    )
    return metrics


def _validate_runtime_identity(
    raw: Any,
    *,
    arm: str,
    arm_spec: Mapping[str, Any],
    shared: Mapping[str, Any],
    complexity_probe_sidecar_sha256: str,
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, f"{label}.runtime_identity")
    if "rtp_sidecar_sha256" in value:
        raise RTPThreeArmEvaluationError(
            f"{label} runtime identity uses retired ambiguous rtp_sidecar_sha256"
        )
    if value.get("arm") != arm:
        raise RTPThreeArmEvaluationError(f"{label} runtime arm mismatch")
    expected = {
        "runtime_artifact_sha256": arm_spec["runtime_artifact"]["sha256"],
        "runtime_profile_sha256": arm_spec["runtime_profile"]["sha256"],
    }
    sidecar = arm_spec.get("rtp_sidecar")
    expected["action_attached_rtp_sidecar_sha256"] = (
        None if sidecar is None else sidecar["sha256"]
    )
    expected["complexity_probe_sidecar_sha256"] = complexity_probe_sidecar_sha256
    expected["complexity_probe_sidecar_instrumentation_only"] = True
    expected["complexity_probe_latency_excluded"] = True
    expected["rtp_action_attachment_enabled"] = arm != "no_rtp"
    expected["rtp_action_authority_enabled"] = False
    for name, identity in shared.items():
        expected[f"{name}_sha256"] = identity["sha256"]
    for key, expected_value in expected.items():
        if key not in value or value.get(key) != expected_value:
            raise RTPThreeArmEvaluationError(f"{label} runtime identity mismatch at {key}")
    profile = _mapping(arm_spec["profile"], f"arms.{arm}.profile")
    for key in (
        "recursive_turn_planner_enabled",
        "direct_bridge_enabled",
        "force_direct_bridge_only",
        "max_neural_passes",
        "max_action_combos",
    ):
        if key not in value or value.get(key) != profile.get(key):
            raise RTPThreeArmEvaluationError(f"{label} runtime profile mismatch at {key}")
    return {
        key: value[key]
        for key in sorted(expected)
        + [
            "arm",
            "recursive_turn_planner_enabled",
            "direct_bridge_enabled",
            "force_direct_bridge_only",
            "max_neural_passes",
            "max_action_combos",
            "action_attached_rtp_sidecar_sha256",
            "complexity_probe_sidecar_sha256",
            "complexity_probe_sidecar_instrumentation_only",
            "complexity_probe_latency_excluded",
            "rtp_action_attachment_enabled",
            "rtp_action_authority_enabled",
        ]
    }


def _validate_rng_identity(raw: Any, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = _mapping(raw, f"{label}.rng_identity")
    # A requested seed is intentionally ignored: it does not prove that an
    # engine restored the same RNG state or consumed the same tape.
    for key in (
        "id",
        "kind",
        "sha256",
        "bytes",
        "seal_sha256",
        "capture_boundary",
        "boundary_tag",
    ):
        if value.get(key) != expected.get(key):
            raise RTPThreeArmEvaluationError(f"{label} true RNG identity mismatch at {key}")
    if value.get("kind") != "snapshot":
        raise RTPThreeArmEvaluationError(f"{label} has unsupported RNG identity kind")
    if value.get("restored_or_replayed") is not True:
        raise RTPThreeArmEvaluationError(
            f"{label} lacks restored/replayed true RNG attestation; requested seeds are not pairs"
        )
    return {**{key: value[key] for key in expected}, "restored_or_replayed": True}


def _validate_execution_evidence(
    raw_execution_receipt: Any,
    raw_transcript: Any,
    *,
    cell_id: str,
    arm: str,
    opponent_id: str,
    expected_baseline_content_digest: str,
    expected_package_root: str,
    expected_package_tree_entries_sha256: str,
    expected_package_manifest_sha256: str,
    expected_opponent_deck_sha256: str,
    expected_evaluation_cg_closure: Mapping[str, Any],
    candidate_seat: int,
    evaluation_corpus_sha256: str,
    evaluation_case_id: str,
    evaluation_case_bindings_sha256: str,
    runtime_identity: Any,
    rng_identity: Any,
    telemetry: Any,
    terminal_outcome: Any,
    candidate_score: float,
    label: str,
) -> dict[str, Any]:
    """Bind each summary row to immutable execution and transcript evidence."""

    execution = _verify_immutable_frozen_identity(
        raw_execution_receipt, f"{label}.execution_receipt"
    )
    transcript = _verify_immutable_frozen_identity(raw_transcript, f"{label}.transcript")
    payload = _read_json_object(execution["path"], f"{label}.execution_receipt")
    if payload.get("schema") != EXECUTION_RECEIPT_SCHEMA:
        raise RTPThreeArmEvaluationError(f"{label} execution receipt schema is invalid")
    if payload.get("status") != "completed":
        raise RTPThreeArmEvaluationError(f"{label} execution receipt is not completed")
    if payload.get("termination") != "completed" or payload.get("failed_seat") is not None:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt is not a completed non-failed game"
        )
    for field in ("engine_error", "candidate_error", "opponent_error"):
        if payload.get(field) not in {None, ""}:
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt recorded {field}"
            )
    if payload.get("cell_id") != cell_id or _canonical_arm_name(
        payload.get("arm"), f"{label} execution receipt arm"
    ) != arm:
        raise RTPThreeArmEvaluationError(f"{label} execution receipt cell/arm mismatch")
    if payload.get("opponent_id") != opponent_id or _integer(
        payload.get("candidate_seat"), f"{label} execution receipt candidate_seat"
    ) != candidate_seat:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt opponent/seat mismatch"
        )
    if _nonempty_text(
        payload.get("evaluation_case_id"), f"{label} execution receipt evaluation_case_id"
    ) != evaluation_case_id:
        raise RTPThreeArmEvaluationError(f"{label} execution receipt evaluation case mismatch")
    expected_digests = {
        "transcript_sha256": transcript["sha256"],
        "runtime_identity_sha256": canonical_digest(runtime_identity),
        "rng_identity_sha256": canonical_digest(rng_identity),
        "telemetry_sha256": canonical_digest(telemetry),
        "terminal_outcome_sha256": canonical_digest(terminal_outcome),
        "evaluation_corpus_sha256": evaluation_corpus_sha256,
        "evaluation_case_bindings_sha256": evaluation_case_bindings_sha256,
    }
    for key, expected in expected_digests.items():
        if _sha256_value(payload.get(key), f"{label} execution receipt {key}") != expected:
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt mismatch at {key}"
            )
    if _score(payload.get("candidate_score"), f"{label} execution receipt candidate_score") != candidate_score:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt candidate score mismatch"
        )
    for field in (
        "fresh_process_per_arm",
        "process_model_load",
        "fresh_candidate_agent",
        "fresh_opponent_module",
        "candidate_reset_called",
        "engine_restore_before_first_select",
        "no_remote_leaf_sampling_mcts",
        "package_snapshot_verified_before_import",
        "complexity_probe_latency_excluded",
    ):
        if payload.get(field) is not True:
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt does not pass {field}"
            )
    if payload.get("launch_mode") != "subprocess_exec":
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt launch_mode must be subprocess_exec"
        )
    if "rtp_sidecar_sha256" in payload:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt uses retired ambiguous rtp_sidecar_sha256"
        )
    process_id = _nonempty_text(payload.get("process_id"), f"{label} execution receipt process_id")
    launch_nonce = _nonempty_text(
        payload.get("launch_nonce"), f"{label} execution receipt launch_nonce"
    )
    if _sha256_value(
        payload.get("baseline_content_digest"),
        f"{label} execution receipt baseline_content_digest",
    ) != expected_baseline_content_digest:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt baseline content digest mismatch"
        )
    observed_package_root = _physical_existing_path(
        _nonempty_text(
            payload.get("baseline_package_root"),
            f"{label} execution receipt baseline_package_root",
        ),
        f"{label} execution receipt baseline_package_root",
        require_directory=True,
    )
    if str(observed_package_root) != expected_package_root:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt baseline package root mismatch"
        )
    for key, expected in (
        ("baseline_tree_entries_sha256", expected_package_tree_entries_sha256),
        ("baseline_package_manifest_sha256", expected_package_manifest_sha256),
    ):
        if _sha256_value(payload.get(key), f"{label} execution receipt {key}") != expected:
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt baseline package snapshot mismatch at {key}"
            )
    if _sha256_value(
        payload.get("baseline_deck_sha256"),
        f"{label} execution receipt baseline_deck_sha256",
    ) != expected_opponent_deck_sha256:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt baseline deck differs from sealed opponent package"
        )
    closure = _mapping(expected_evaluation_cg_closure, "evaluation_cg_closure")
    expected_closure_evidence = {
        "evaluation_cg_closure_receipt_sha256": _sha256_value(
            _mapping(closure.get("receipt"), "evaluation_cg_closure.receipt").get("sha256"),
            "evaluation_cg_closure.receipt.sha256",
        ),
        "evaluation_cg_engine_sha256": _sha256_value(
            _mapping(closure.get("runtime_library"), "evaluation_cg_closure.runtime_library").get(
                "sha256"
            ),
            "evaluation_cg_closure.runtime_library.sha256",
        ),
        "evaluation_cg_closure_manifest_sha256": _sha256_value(
            _mapping(closure.get("closure_manifest"), "evaluation_cg_closure.closure_manifest").get(
                "sha256"
            ),
            "evaluation_cg_closure.closure_manifest.sha256",
        ),
        "evaluation_cg_metadata_parity_sha256": _sha256_value(
            _mapping(closure.get("metadata_parity"), "evaluation_cg_closure.metadata_parity").get(
                "sha256"
            ),
            "evaluation_cg_closure.metadata_parity.sha256",
        ),
    }
    for key, expected in expected_closure_evidence.items():
        if _sha256_value(payload.get(key), f"{label} execution receipt {key}") != expected:
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt differs from sealed evaluation CG closure at {key}"
            )
    runtime_library = _mapping(
        closure.get("runtime_library"), "evaluation_cg_closure.runtime_library"
    )
    expected_engine_path = _nonempty_text(
        runtime_library.get("path"), "evaluation_cg_closure.runtime_library.path"
    )
    expected_engine_bytes = _integer(
        runtime_library.get("bytes"), "evaluation_cg_closure.runtime_library.bytes", minimum=1
    )
    for key in ("evaluation_cg_engine_path", "engine_loaded_path"):
        if _nonempty_text(payload.get(key), f"{label} execution receipt {key}") != expected_engine_path:
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt did not load the snapshot-local sealed evaluation CG engine"
            )
    if _integer(
        payload.get("evaluation_cg_engine_bytes"),
        f"{label} execution receipt evaluation_cg_engine_bytes",
        minimum=1,
    ) != expected_engine_bytes:
        raise RTPThreeArmEvaluationError(
            f"{label} execution receipt snapshot-local CG engine bytes differ from the sealed runtime library"
        )
    _sha256_value(
        payload.get("candidate_runtime_contract_sha256"),
        f"{label} execution receipt candidate_runtime_contract_sha256",
    )
    action_fence = payload.get("action_fence_sha256")
    action_context = payload.get("evaluation_action_execution_sha256")
    if arm == "no_rtp":
        if action_fence is not None:
            raise RTPThreeArmEvaluationError(
                f"{label} no_rtp execution receipt unexpectedly has an evaluator action fence"
            )
        if action_context is not None:
            raise RTPThreeArmEvaluationError(
                f"{label} no_rtp execution receipt unexpectedly has evaluator action context"
            )
    else:
        _sha256_value(
            action_fence, f"{label} execution receipt action_fence_sha256"
        )
        _sha256_value(
            action_context, f"{label} execution receipt evaluation_action_execution_sha256"
        )
    nested_isolation = _mapping(payload.get("isolation"), f"{label} execution receipt isolation")
    for key in (
        "launch_mode",
        "fresh_process_per_arm",
        "process_model_load",
        "fresh_candidate_agent",
        "candidate_reset_called",
        "fresh_opponent_module",
        "engine_restore_before_first_select",
        "no_remote_leaf_sampling_mcts",
        "package_snapshot_verified_before_import",
        "complexity_probe_latency_excluded",
        "candidate_runtime_contract_sha256",
        "action_fence_sha256",
        "evaluation_action_execution_sha256",
        "evaluation_cg_closure_receipt_sha256",
        "evaluation_cg_engine_sha256",
        "evaluation_cg_engine_path",
        "evaluation_cg_engine_bytes",
        "evaluation_cg_closure_manifest_sha256",
        "evaluation_cg_metadata_parity_sha256",
    ):
        if nested_isolation.get(key) != payload.get(key):
            raise RTPThreeArmEvaluationError(
                f"{label} execution receipt nested isolation mismatch at {key}"
            )
    candidate_rng_initial_state_sha256 = _sha256_value(
        payload.get("candidate_rng_initial_state_sha256"),
        f"{label} execution receipt candidate_rng_initial_state_sha256",
    )
    opponent_rng_deterministic_or_no_rng = payload.get(
        "opponent_rng_deterministic_or_no_rng"
    ) is True
    opponent_rng_raw = payload.get("opponent_rng_initial_state_sha256")
    if opponent_rng_deterministic_or_no_rng:
        opponent_rng_initial_state_sha256: str | None = None
    else:
        opponent_rng_initial_state_sha256 = _sha256_value(
            opponent_rng_raw,
            f"{label} execution receipt opponent_rng_initial_state_sha256",
        )
    common_sanitized_environment_sha256 = _sha256_value(
        payload.get("common_sanitized_environment_sha256"),
        f"{label} execution receipt common_sanitized_environment_sha256",
    )
    arm_environment_sha256 = _sha256_value(
        payload.get("arm_environment_sha256"),
        f"{label} execution receipt arm_environment_sha256",
    )
    return {
        "execution_receipt": execution,
        "transcript": transcript,
        "process_id": process_id,
        "launch_nonce": launch_nonce,
        "candidate_rng_initial_state_sha256": candidate_rng_initial_state_sha256,
        "opponent_rng_initial_state_sha256": opponent_rng_initial_state_sha256,
        "opponent_rng_deterministic_or_no_rng": opponent_rng_deterministic_or_no_rng,
        "common_sanitized_environment_sha256": common_sanitized_environment_sha256,
        "arm_environment_sha256": arm_environment_sha256,
    }


def _result_rows(
    manifest: Mapping[str, Any], results: Any
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    if isinstance(results, (str, Path)):
        source = _verify_immutable_frozen_identity(
            _observed_file_identity(results, "evaluation results"),
            "evaluation results",
        )
        payload = _read_json_object(source["path"], "evaluation results")
        source_identity: dict[str, Any] = source
        file_backed = True
    else:
        payload = _mapping(results, "evaluation results")
        source_identity = {"canonical_sha256": canonical_digest(payload), "in_memory": True}
        file_backed = False
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RTPThreeArmEvaluationError("evaluation results.rows must be a list")
    schedule = {
        str(cell["cell_id"]): _mapping(cell, "schedule cell")
        for cell in list(manifest.get("schedule") or ())
    }
    arms = _mapping(manifest.get("arms"), "arms")
    shared = _mapping(manifest.get("shared_artifacts"), "shared_artifacts")
    opponents_by_id = {
        _nonempty_text(row.get("id"), "opponent.id"): _mapping(row, "opponent")
        for row in manifest.get("opponents") or ()
    }
    source_exclusion_binding = _mapping(
        manifest.get("r197_source_exclusion_binding"), "r197_source_exclusion_binding"
    )
    expected_evaluation_corpus_sha256 = _sha256_value(
        _mapping(
            source_exclusion_binding.get("evaluation_only_cohort"),
            "r197_source_exclusion_binding.evaluation_only_cohort",
        ).get("sha256"),
        "r197_source_exclusion_binding.evaluation_only_cohort.sha256",
    )
    expected_count = len(schedule) * len(ARMS)
    if len(rows) != expected_count:
        raise RTPThreeArmEvaluationError(
            f"evaluation results have {len(rows)} rows; expected exactly {expected_count}"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        label = f"results.rows[{index}]"
        value = _mapping(raw, label)
        cell_id = _nonempty_text(value.get("cell_id"), f"{label}.cell_id")
        arm = _canonical_arm_name(value.get("arm"), f"{label}.arm")
        if arm not in ARMS:
            raise RTPThreeArmEvaluationError(f"{label}.arm is not a supported arm")
        if cell_id not in schedule:
            raise RTPThreeArmEvaluationError(f"{label} refers to an unknown schedule cell")
        key = (cell_id, arm)
        if key in seen:
            raise RTPThreeArmEvaluationError(f"duplicate result for {cell_id}/{arm}")
        seen.add(key)
        cell = schedule[cell_id]
        if value.get("opponent_id") != cell["opponent_id"]:
            raise RTPThreeArmEvaluationError(f"{label} opponent identity differs from schedule")
        if _integer(value.get("candidate_seat"), f"{label}.candidate_seat") != int(
            cell["candidate_seat"]
        ):
            raise RTPThreeArmEvaluationError(f"{label} candidate seat differs from schedule")
        if _nonempty_text(value.get("evaluation_case_id"), f"{label}.evaluation_case_id") != _nonempty_text(
            cell.get("evaluation_case_id"), f"schedule {cell_id}.evaluation_case_id"
        ):
            raise RTPThreeArmEvaluationError(
                f"{label} evaluation case differs from the frozen schedule"
            )
        if _sha256_value(
            value.get("evaluation_case_bindings_sha256"),
            f"{label}.evaluation_case_bindings_sha256",
        ) != _sha256_value(
            cell.get("evaluation_case_bindings_sha256"),
            f"schedule {cell_id}.evaluation_case_bindings_sha256",
        ):
            raise RTPThreeArmEvaluationError(
                f"{label} evaluation case binding differs from the frozen schedule"
            )
        if value.get("completed") is not True or value.get("invalid") not in {False, None}:
            raise RTPThreeArmEvaluationError(f"{label} is incomplete or invalid")
        if value.get("error") not in {None, ""}:
            raise RTPThreeArmEvaluationError(f"{label} recorded an execution error")
        evaluation_corpus_sha256 = _sha256_value(
            value.get("evaluation_corpus_sha256"), f"{label}.evaluation_corpus_sha256"
        )
        if evaluation_corpus_sha256 != expected_evaluation_corpus_sha256:
            raise RTPThreeArmEvaluationError(
                f"{label} is not bound to the separate source-excluded evaluation cohort"
            )
        runtime = _validate_runtime_identity(
            value.get("runtime_identity"),
            arm=arm,
            arm_spec=_mapping(arms[arm], f"arms.{arm}"),
            shared=shared,
            complexity_probe_sidecar_sha256=_mapping(
                arms[DIRECT_BRIDGE_ARM], f"arms.{DIRECT_BRIDGE_ARM}"
            )["rtp_sidecar"]["sha256"],
            label=label,
        )
        rng = _validate_rng_identity(value.get("rng_identity"), cell["rng_identity"], label)
        terminal_outcome, terminal_score = _terminal_candidate_score(
            value, label, candidate_seat=int(cell["candidate_seat"])
        )
        telemetry = _result_telemetry(value.get("telemetry"), arm=arm, label=label)
        _validate_r198_forced_turn_order_contract(
            telemetry,
            candidate_seat=int(cell["candidate_seat"]),
            label=label,
        )
        candidate_score = _score(value.get("candidate_score"), f"{label}.candidate_score")
        if candidate_score != terminal_score:
            raise RTPThreeArmEvaluationError(
                f"{label} candidate_score does not match its terminal engine outcome"
            )
        if telemetry["candidate_forfeit_count"] != int(
            terminal_outcome["candidate_forfeit"]
        ):
            raise RTPThreeArmEvaluationError(
                f"{label} candidate forfeit telemetry does not match terminal outcome"
            )
        execution_evidence = _validate_execution_evidence(
            value.get("execution_receipt"),
            value.get("transcript"),
            cell_id=cell_id,
            arm=arm,
            opponent_id=str(cell["opponent_id"]),
            expected_baseline_content_digest=_sha256_value(
                opponents_by_id[str(cell["opponent_id"])].get("content_digest"),
                "opponent.content_digest",
            ),
            expected_package_root=_nonempty_text(
                opponents_by_id[str(cell["opponent_id"])].get("package_root"),
                "opponent.package_root",
            ),
            expected_package_tree_entries_sha256=_sha256_value(
                opponents_by_id[str(cell["opponent_id"])].get("tree_entries_sha256"),
                "opponent.tree_entries_sha256",
            ),
            expected_package_manifest_sha256=_sha256_value(
                _mapping(
                    opponents_by_id[str(cell["opponent_id"])].get("artifact"),
                    "opponent.artifact",
                ).get("sha256"),
                "opponent.artifact.sha256",
            ),
            expected_opponent_deck_sha256=_sha256_value(
                opponents_by_id[str(cell["opponent_id"])].get("deck_sha256"),
                "opponent.deck_sha256",
            ),
            expected_evaluation_cg_closure=_mapping(
                manifest.get("evaluation_cg_closure"), "evaluation_cg_closure"
            ),
            candidate_seat=int(cell["candidate_seat"]),
            evaluation_corpus_sha256=evaluation_corpus_sha256,
            evaluation_case_id=str(cell["evaluation_case_id"]),
            evaluation_case_bindings_sha256=str(cell["evaluation_case_bindings_sha256"]),
            runtime_identity=value.get("runtime_identity"),
            rng_identity=value.get("rng_identity"),
            telemetry=value.get("telemetry"),
            terminal_outcome=value.get("terminal_outcome"),
            candidate_score=candidate_score,
            label=label,
        )
        normalized.append(
            {
                "cell_id": cell_id,
                "arm": arm,
                "opponent_id": str(cell["opponent_id"]),
                "candidate_seat": int(cell["candidate_seat"]),
                "rng_identity": rng,
                "candidate_score": candidate_score,
                "terminal_outcome": terminal_outcome,
                "runtime_identity": runtime,
                "telemetry": telemetry,
                "evaluation_corpus_sha256": evaluation_corpus_sha256,
                "evaluation_case_id": str(cell["evaluation_case_id"]),
                "evaluation_case_bindings_sha256": _sha256_value(
                    cell["evaluation_case_bindings_sha256"],
                    f"schedule {cell_id}.evaluation_case_bindings_sha256",
                ),
                "execution_input_sha256": {
                    "runtime_identity": canonical_digest(value.get("runtime_identity")),
                    "rng_identity": canonical_digest(value.get("rng_identity")),
                    "telemetry": canonical_digest(value.get("telemetry")),
                    "terminal_outcome": canonical_digest(value.get("terminal_outcome")),
                },
                "execution_evidence": execution_evidence,
            }
        )
    if len(seen) != expected_count:
        raise RTPThreeArmEvaluationError("evaluation result schedule is incomplete")
    process_ids: set[str] = set()
    launch_nonces: set[str] = set()
    execution_receipt_digests: set[str] = set()
    transcript_digests: set[str] = set()
    by_cell_evidence: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized:
        evidence = _mapping(row["execution_evidence"], "execution evidence")
        process_id = _nonempty_text(evidence.get("process_id"), "execution process_id")
        launch_nonce = _nonempty_text(evidence.get("launch_nonce"), "execution launch_nonce")
        if process_id in process_ids or launch_nonce in launch_nonces:
            raise RTPThreeArmEvaluationError(
                "evaluation execution evidence reused a process_id or launch_nonce"
            )
        process_ids.add(process_id)
        launch_nonces.add(launch_nonce)
        execution_digest = _sha256_value(
            _mapping(evidence.get("execution_receipt"), "execution receipt").get("sha256"),
            "execution receipt.sha256",
        )
        transcript_digest = _sha256_value(
            _mapping(evidence.get("transcript"), "transcript").get("sha256"),
            "transcript.sha256",
        )
        if execution_digest in execution_receipt_digests or transcript_digest in transcript_digests:
            raise RTPThreeArmEvaluationError(
                "evaluation execution evidence reuses a per-cell/arm receipt or transcript"
            )
        execution_receipt_digests.add(execution_digest)
        transcript_digests.add(transcript_digest)
        by_cell_evidence[row["cell_id"]].append(row)
    for cell_id, cell_rows in by_cell_evidence.items():
        if len(cell_rows) != len(ARMS):
            raise RTPThreeArmEvaluationError(f"execution evidence is incomplete for {cell_id}")
        evidence = [
            _mapping(row["execution_evidence"], "execution evidence") for row in cell_rows
        ]
        candidate_rngs = {
            _sha256_value(
                item.get("candidate_rng_initial_state_sha256"),
                "candidate_rng_initial_state_sha256",
            )
            for item in evidence
        }
        common_environments = {
            _sha256_value(
                item.get("common_sanitized_environment_sha256"),
                "common_sanitized_environment_sha256",
            )
            for item in evidence
        }
        if len(candidate_rngs) != 1 or len(common_environments) != 1:
            raise RTPThreeArmEvaluationError(
                f"execution evidence does not share candidate RNG/common environment for {cell_id}"
            )
        deterministic_opponent = {
            bool(item.get("opponent_rng_deterministic_or_no_rng")) for item in evidence
        }
        opponent_rngs = {item.get("opponent_rng_initial_state_sha256") for item in evidence}
        if len(deterministic_opponent) != 1 or (
            deterministic_opponent == {False} and len(opponent_rngs) != 1
        ):
            raise RTPThreeArmEvaluationError(
                f"execution evidence does not share opponent RNG mode for {cell_id}"
            )
    return (
        sorted(normalized, key=lambda row: (row["cell_id"], ARMS.index(row["arm"]))),
        source_identity,
        file_backed,
    )


def _percentile95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _one_sided_hoeffding_lower(deltas: Sequence[float], confidence: float) -> float:
    if not deltas:
        raise RTPThreeArmEvaluationError("cannot summarize zero paired cells")
    # Score deltas live in [-1, 1], whose range is two.
    mean = sum(deltas) / len(deltas)
    alpha = 1.0 - confidence
    radius = math.sqrt(2.0 * math.log(1.0 / alpha) / len(deltas))
    return max(-1.0, mean - radius)


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RTPThreeArmEvaluationError("cannot summarize an empty arm")
    scores = [float(row["candidate_score"]) for row in rows]
    telemetry_keys = (
        "candidate_decisions",
        "planner_eligible_candidate_decisions",
        "over_cap_factorized_fallback_decisions",
        "forced_turn_order_controls",
        "intended_complex_decisions",
        "recursive_intended_complex_decisions",
        "direct_bridge_decisions",
        "recursive_decisions",
        "fallback_decisions",
        "unexpected_recursive_fallback_decisions",
        "expected_recursive_fallback_decisions",
        "neural_budget_exceeded",
        "neural_budget_failures",
        "illegal_action_count",
        "candidate_forfeit_count",
        "normal_recursive_plans",
        "forced_replans",
    )
    totals = {
        key: sum(int(_mapping(row["telemetry"], "telemetry")[key]) for row in rows)
        for key in telemetry_keys
    }
    decisions = totals["candidate_decisions"]
    planner_eligible_decisions = totals["planner_eligible_candidate_decisions"]
    intended_complex = totals["intended_complex_decisions"]
    decision_latency_trace: list[dict[str, Any]] = []
    over_cap_factorized_fallback_trace: list[dict[str, Any]] = []
    forced_turn_order_control_trace: list[dict[str, Any]] = []
    recursive_decision_latencies: list[float] = []
    recursive_mode_totals = {
        mode: 0
        for mode in sorted(SUCCESSFUL_RECURSIVE_MODES | RECURSIVE_FALLBACK_MODES)
    }
    by_opponent: dict[str, int] = defaultdict(int)
    seat_counts = {"0": 0, "1": 0}
    for row in rows:
        by_opponent[str(row["opponent_id"])] += 1
        seat_counts[str(int(row["candidate_seat"]))] += 1
        telemetry = _mapping(row["telemetry"], "telemetry")
        if telemetry.get("intended_complex_decision_scope") != INTENDED_COMPLEX_DECISION_SCOPE:
            raise RTPThreeArmEvaluationError("arm telemetry uses an inconsistent intended-complex scope")
        mode_counts = _mapping(telemetry.get("recursive_mode_counts"), "recursive_mode_counts")
        for mode in recursive_mode_totals:
            recursive_mode_totals[mode] += _integer(
                mode_counts.get(mode), f"recursive_mode_counts.{mode}"
            )
        decision_latency_trace.extend(
            dict(value) for value in telemetry["decision_latency_trace"]
        )
        over_cap_factorized_fallback_trace.extend(
            dict(value) for value in telemetry["over_cap_factorized_fallback_trace"]
        )
        forced_turn_order_control_trace.extend(
            dict(value) for value in telemetry["forced_turn_order_control_trace"]
        )
        recursive_decision_latencies.extend(
            float(value) for value in telemetry["recursive_decision_latency_seconds"]
        )
    if len(decision_latency_trace) != decisions:
        raise RTPThreeArmEvaluationError(
            "decision latency distribution count does not equal candidate decisions"
        )
    if len(over_cap_factorized_fallback_trace) != totals[
        "over_cap_factorized_fallback_decisions"
    ]:
        raise RTPThreeArmEvaluationError(
            "over-cap factorized trace count does not equal its aggregate"
        )
    if len(forced_turn_order_control_trace) != totals["forced_turn_order_controls"]:
        raise RTPThreeArmEvaluationError(
            "forced turn-order control trace count does not equal its aggregate"
        )
    if totals["forced_turn_order_controls"] != OFFICIAL_CONTROL_PAIRED_CELLS // 2:
        raise RTPThreeArmEvaluationError(
            "arm forced turn-order control count does not match the frozen seat ABI"
        )
    return {
        "games": len(rows),
        "mean_candidate_score": sum(scores) / len(scores),
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "candidate_seat_counts": seat_counts,
        "by_opponent": dict(sorted(by_opponent.items())),
        "telemetry": {
            **totals,
            "intended_complex_decision_scope": INTENDED_COMPLEX_DECISION_SCOPE,
            "recursive_mode_counts": recursive_mode_totals,
            "fallback_rate": (
                None
                if planner_eligible_decisions == 0
                else totals["fallback_decisions"] / planner_eligible_decisions
            ),
            "over_cap_factorized_fallback_rate": (
                None
                if decisions == 0
                else totals["over_cap_factorized_fallback_decisions"] / decisions
            ),
            "recursive_share_of_intended_complex_decisions": (
                None
                if intended_complex == 0
                else totals["recursive_intended_complex_decisions"] / intended_complex
            ),
            "unexpected_recursive_fallback_rate": (
                None
                if intended_complex == 0
                else totals["unexpected_recursive_fallback_decisions"] / intended_complex
            ),
            "decision_latency_trace": {
                "count": len(decision_latency_trace),
                "sha256": canonical_digest(decision_latency_trace),
            },
            "forced_turn_order_control_trace": {
                "count": len(forced_turn_order_control_trace),
                "sha256": canonical_digest(forced_turn_order_control_trace),
            },
            "over_cap_factorized_fallback_trace": {
                "count": len(over_cap_factorized_fallback_trace),
                "sha256": canonical_digest(over_cap_factorized_fallback_trace),
            },
            "recursive_decision_latency_distribution": {
                "count": len(recursive_decision_latencies),
                "sha256": canonical_digest(recursive_decision_latencies),
                "p50_seconds": _percentile(recursive_decision_latencies, 0.50),
                "p95_seconds": _percentile95(recursive_decision_latencies),
                "max_seconds": max(recursive_decision_latencies, default=None),
            },
        },
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0.0 < percentile <= 1.0:
        raise RTPThreeArmEvaluationError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _paired_summary(
    left_rows: Mapping[str, Mapping[str, Any]],
    right_rows: Mapping[str, Mapping[str, Any]],
    *,
    confidence: float,
    name: str,
) -> dict[str, Any]:
    if set(left_rows) != set(right_rows):
        raise RTPThreeArmEvaluationError(f"{name} lacks exact cell pairing")
    deltas = [
        float(right_rows[cell_id]["candidate_score"])
        - float(left_rows[cell_id]["candidate_score"])
        for cell_id in sorted(left_rows)
    ]
    return {
        "endpoint": "paired_expected_score_delta",
        "pairs": len(deltas),
        "mean_score_delta": sum(deltas) / len(deltas),
        "one_sided_lower_confidence_bound": _one_sided_hoeffding_lower(
            deltas, confidence
        ),
        "confidence_level": confidence,
        "confidence_method": "one_sided_hoeffding_for_true_rng_paired_score_deltas",
    }


def _stratified_paired_summary(
    left_rows: Mapping[str, Mapping[str, Any]],
    right_rows: Mapping[str, Mapping[str, Any]],
    *,
    confidence: float,
    name: str,
) -> dict[str, dict[str, Any]]:
    """Report recursive efficacy separately for every opponent × candidate seat."""

    if set(left_rows) != set(right_rows):
        raise RTPThreeArmEvaluationError(f"{name} lacks exact cell pairing")
    groups: dict[str, tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]] = {}
    for cell_id in sorted(left_rows):
        left = left_rows[cell_id]
        right = right_rows[cell_id]
        if (
            left["opponent_id"] != right["opponent_id"]
            or left["candidate_seat"] != right["candidate_seat"]
        ):
            raise RTPThreeArmEvaluationError(f"{name} changed opponent or seat within a pair")
        stratum = f"{left['opponent_id']}|seat{left['candidate_seat']}"
        if stratum not in groups:
            groups[stratum] = ({}, {})
        groups[stratum][0][cell_id] = left
        groups[stratum][1][cell_id] = right
    return {
        stratum: _paired_summary(
            group_left,
            group_right,
            confidence=confidence,
            name=f"{name}:{stratum}",
        )
        for stratum, (group_left, group_right) in sorted(groups.items())
    }


def _paired_regression_summary(
    baseline_rows: Mapping[str, Mapping[str, Any]],
    recursive_rows: Mapping[str, Mapping[str, Any]],
    *,
    telemetry_key: str,
    name: str,
) -> dict[str, Any]:
    if set(baseline_rows) != set(recursive_rows):
        raise RTPThreeArmEvaluationError(f"{name} lacks exact cell pairing")
    increases: list[int] = []
    for cell_id in sorted(baseline_rows):
        baseline = int(_mapping(baseline_rows[cell_id]["telemetry"], "telemetry")[telemetry_key])
        recursive = int(_mapping(recursive_rows[cell_id]["telemetry"], "telemetry")[telemetry_key])
        increases.append(max(0, recursive - baseline))
    return {
        "new_count": sum(increases),
        "cells_with_regression": sum(value > 0 for value in increases),
        "maximum_per_cell_increase": max(increases, default=0),
    }


def _gate(passed: bool, *, observed: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "required": required}


def _verify_forced_turn_order_pairing(
    by_cell: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    """Require exact external-control evidence across all matched arms."""

    for cell_id, arms in by_cell.items():
        if set(arms) != set(ARMS):
            raise RTPThreeArmEvaluationError(
                f"cell {cell_id} does not have every arm for forced-control comparison"
            )
        reference = _mapping(
            arms["no_rtp"].get("telemetry"), f"cell {cell_id} no_rtp telemetry"
        )
        expected_count = _integer(
            reference.get("forced_turn_order_controls"),
            f"cell {cell_id} no_rtp forced_turn_order_controls",
        )
        expected_trace = reference.get("forced_turn_order_control_trace")
        for arm in ARMS:
            telemetry = _mapping(
                arms[arm].get("telemetry"), f"cell {cell_id} {arm} telemetry"
            )
            count = _integer(
                telemetry.get("forced_turn_order_controls"),
                f"cell {cell_id} {arm} forced_turn_order_controls",
            )
            if count != expected_count:
                raise RTPThreeArmEvaluationError(
                    f"cell {cell_id} forced turn-order control count differs across arms"
                )
            if canonical_digest(telemetry.get("forced_turn_order_control_trace")) != canonical_digest(
                expected_trace
            ):
                raise RTPThreeArmEvaluationError(
                    f"cell {cell_id} forced turn-order control trace differs across arms"
                )


def _validate_conditional_over_cap_action_parity(
    by_cell: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, int]:
    """Compare actions only for identical logical policy-input fingerprints.

    A/B/C use distinct bridge paths and can legitimately diverge in
    history/cache state.  The special trace binds a canonical factorized
    policy-input digest, so action equality is checked only among traces that
    share that digest in one paired cell.  It is deliberately not a global
    cross-arm event-count or action-equality rule.
    """

    groups: dict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    total = 0
    for cell_id, arms in by_cell.items():
        for arm, row in arms.items():
            telemetry = _mapping(row.get("telemetry"), "row telemetry")
            traces = telemetry.get("over_cap_factorized_fallback_trace")
            if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
                raise RTPThreeArmEvaluationError("row lacks over-cap factorized trace")
            for trace in traces:
                value = _mapping(trace, "over-cap factorized trace")
                fingerprint = _sha256_value(
                    value.get("logical_pre_action_sha256"),
                    "over-cap logical pre-action digest",
                )
                groups[(cell_id, fingerprint)].append((arm, value))
                total += 1
    comparable_groups = 0
    comparable_arm_rows = 0
    for entries in groups.values():
        if len({arm for arm, _trace in entries}) < 2:
            continue
        comparable_groups += 1
        comparable_arm_rows += len(entries)
        actions: set[tuple[int, ...]] = set()
        for _arm, trace in entries:
            raw_action = trace.get("returned_action")
            if not isinstance(raw_action, Sequence) or isinstance(raw_action, (str, bytes)):
                raise RTPThreeArmEvaluationError("over-cap returned action is invalid")
            if any(type(item) is not int for item in raw_action):
                raise RTPThreeArmEvaluationError(
                    "over-cap returned action has a non-exact index"
                )
            actions.add(tuple(raw_action))
        if len(actions) != 1:
            raise RTPThreeArmEvaluationError(
                "over-cap factorized actions differ despite identical logical pre-action inputs"
            )
    return {
        "over_cap_trace_rows": total,
        "logical_pre_action_groups": len(groups),
        "cross_arm_comparable_groups": comparable_groups,
        "cross_arm_comparable_arm_rows": comparable_arm_rows,
    }


def compile_three_arm_receipt(
    *,
    manifest_path: str | Path,
    results: Mapping[str, Any] | str | Path,
    output_path: str | Path,
) -> Path:
    """Validate completed rows and issue a hold/review receipt.

    The function never changes serving state.  Even a fully passing receipt is
    only eligible for a separate promotion review; all failed gates produce a
    durable ``hold`` result.
    """

    manifest, manifest_identity = _load_manifest(manifest_path)
    # A mutable worktree, sidecar, package, opponent, or RNG artifact means
    # the study has stopped being a frozen comparison.  Do not write a result.
    verify_manifest_frozen_artifacts(manifest)
    rows, results_identity, results_file_backed = _result_rows(manifest, results)
    gates = _normalize_gates(manifest.get("promotion_gates"))
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        per_arm[row["arm"]].append(row)
        by_cell[row["cell_id"]][row["arm"]] = row
    if any(set(arms) != set(ARMS) for arms in by_cell.values()):
        raise RTPThreeArmEvaluationError("not every cell has all three arms")
    _verify_forced_turn_order_pairing(by_cell)
    over_cap_parity = _validate_conditional_over_cap_action_parity(by_cell)
    summaries = {arm: _arm_summary(per_arm[arm]) for arm in ARMS}
    if sum(
        int(_mapping(summaries[arm]["telemetry"], "arm summary telemetry")[
            "forced_turn_order_controls"
        ])
        for arm in ARMS
    ) != len(ARMS) * (OFFICIAL_CONTROL_PAIRED_CELLS // 2):
        raise RTPThreeArmEvaluationError(
            "three-arm forced turn-order control total does not match the frozen seat ABI"
        )
    no_rtp_rows = {row["cell_id"]: row for row in per_arm["no_rtp"]}
    bridge_rows = {row["cell_id"]: row for row in per_arm[DIRECT_BRIDGE_ARM]}
    recursive_rows = {row["cell_id"]: row for row in per_arm["recursive_rtp"]}
    bridge_minus_no_rtp_name = f"{DIRECT_BRIDGE_ARM}_minus_no_rtp"
    recursive_minus_bridge_name = f"recursive_rtp_minus_{DIRECT_BRIDGE_ARM}"
    comparisons = {
        bridge_minus_no_rtp_name: _paired_summary(
            no_rtp_rows,
            bridge_rows,
            confidence=gates["confidence_level"],
            name=bridge_minus_no_rtp_name,
        ),
        recursive_minus_bridge_name: _paired_summary(
            bridge_rows,
            recursive_rows,
            confidence=gates["confidence_level"],
            name=recursive_minus_bridge_name,
        ),
        "recursive_rtp_minus_no_rtp": _paired_summary(
            no_rtp_rows,
            recursive_rows,
            confidence=gates["confidence_level"],
            name="recursive_rtp_minus_no_rtp",
        ),
    }
    support: dict[str, int] = defaultdict(int)
    for row in per_arm["no_rtp"]:
        support[f"{row['opponent_id']}|seat{row['candidate_seat']}"] += 1
    stratified_recursive_effect = _stratified_paired_summary(
        bridge_rows,
        recursive_rows,
        confidence=gates["confidence_level"],
        name=recursive_minus_bridge_name,
    )
    comparisons[recursive_minus_bridge_name]["opponent_seat_stratified"] = (
        stratified_recursive_effect
    )
    recursive_telemetry = summaries["recursive_rtp"]["telemetry"]
    bridge_telemetry = summaries[DIRECT_BRIDGE_ARM]["telemetry"]
    recursive_fallback_rate = recursive_telemetry[
        "unexpected_recursive_fallback_rate"
    ]
    budget_failures = {
        arm: summaries[arm]["telemetry"]["neural_budget_failures"] for arm in ARMS
    }
    illegal_regressions = {
        f"recursive_rtp_vs_{DIRECT_BRIDGE_ARM}": _paired_regression_summary(
            bridge_rows,
            recursive_rows,
            telemetry_key="illegal_action_count",
            name=f"recursive illegal-action regression vs {DIRECT_BRIDGE_ARM}",
        ),
        "recursive_rtp_vs_no_rtp": _paired_regression_summary(
            no_rtp_rows,
            recursive_rows,
            telemetry_key="illegal_action_count",
            name="recursive illegal-action regression vs no_rtp",
        ),
    }
    forfeit_regressions = {
        f"recursive_rtp_vs_{DIRECT_BRIDGE_ARM}": _paired_regression_summary(
            bridge_rows,
            recursive_rows,
            telemetry_key="candidate_forfeit_count",
            name=f"recursive forfeit regression vs {DIRECT_BRIDGE_ARM}",
        ),
        "recursive_rtp_vs_no_rtp": _paired_regression_summary(
            no_rtp_rows,
            recursive_rows,
            telemetry_key="candidate_forfeit_count",
            name="recursive forfeit regression vs no_rtp",
        ),
    }
    source_exclusion_binding = _mapping(
        manifest.get("r197_source_exclusion_binding"),
        "r197_source_exclusion_binding",
    )
    promotion_gates = {
        "frozen_artifact_identity": _gate(True, observed=True, required=True),
        "true_rng_tape_or_snapshot_pairing": _gate(True, observed=True, required=True),
        "immutable_file_backed_execution_evidence": _gate(
            results_file_backed,
            observed={
                "results_file_backed": results_file_backed,
                "per_cell_arm_execution_receipts": len(rows),
                "per_cell_arm_transcripts": len(rows),
            },
            required="immutable file-backed results plus one receipt and transcript per cell/arm",
        ),
        "separate_source_excluded_evaluation_only_cohort": _gate(
            source_exclusion_binding.get("r197_source_disjoint") is True
            and source_exclusion_binding.get("evaluation_only") is True
            and source_exclusion_binding.get("source_identity_overlap_count") == 0,
            observed={
                "candidate_contract_sha256": source_exclusion_binding.get(
                    "candidate_contract_sha256"
                ),
                "evaluation_only_cohort_sha256": _mapping(
                    source_exclusion_binding.get("evaluation_only_cohort"),
                    "evaluation_only_cohort",
                ).get("sha256"),
                "selection_plan_sha256": source_exclusion_binding.get(
                    "selection_plan_sha256"
                ),
                "heldout_selection_sha256": source_exclusion_binding.get(
                    "heldout_selection_sha256"
                ),
                "r197_source_disjoint": source_exclusion_binding.get(
                    "r197_source_disjoint"
                ),
                "source_identity_overlap_count": source_exclusion_binding.get(
                    "source_identity_overlap_count"
                ),
            },
            required=True,
        ),
        "trusted_counterfactual_candidate_targets": _gate(
            source_exclusion_binding.get("trusted_counterfactual_candidate_targets_available")
            is True,
            observed={
                "candidate_target_status": source_exclusion_binding.get(
                    "candidate_target_status"
                ),
                "trusted_counterfactual_candidate_targets_available": source_exclusion_binding.get(
                    "trusted_counterfactual_candidate_targets_available"
                ),
            },
            required=True,
        ),
        "complete_three_arm_schedule": _gate(
            len(by_cell) >= gates["minimum_paired_cells"],
            observed=len(by_cell),
            required=gates["minimum_paired_cells"],
        ),
        "opponent_seat_support": _gate(
            bool(support)
            and min(support.values()) >= gates["minimum_pairs_per_opponent_seat"],
            observed=dict(sorted(support.items())),
            required=gates["minimum_pairs_per_opponent_seat"],
        ),
        "opponent_seat_stratified_recursive_efficacy": _gate(
            set(stratified_recursive_effect) == set(support)
            and bool(stratified_recursive_effect)
            and min(
                summary["pairs"] for summary in stratified_recursive_effect.values()
            )
            >= gates["minimum_pairs_per_opponent_seat"],
            observed=stratified_recursive_effect,
            required={
                "all_opponent_seat_strata_reported": True,
                "minimum_pairs_per_stratum": gates["minimum_pairs_per_opponent_seat"],
            },
        ),
        "direct_bridge_path_exercised": _gate(
            bridge_telemetry["direct_bridge_decisions"]
            >= gates["minimum_direct_bridge_decisions"],
            observed=bridge_telemetry["direct_bridge_decisions"],
            required=gates["minimum_direct_bridge_decisions"],
        ),
        "recursive_path_exercised": _gate(
            recursive_telemetry["recursive_decisions"]
            >= gates["minimum_recursive_decisions"],
            observed=recursive_telemetry["recursive_decisions"],
            required=gates["minimum_recursive_decisions"],
        ),
        "recursive_share_of_intended_complex_decisions": _gate(
            recursive_telemetry["recursive_share_of_intended_complex_decisions"]
            is not None
            and recursive_telemetry["recursive_share_of_intended_complex_decisions"]
            >= gates["minimum_recursive_share_of_intended_complex_decisions"],
            observed={
                "recursive_intended_complex_decisions": recursive_telemetry[
                    "recursive_intended_complex_decisions"
                ],
                "intended_complex_decisions": recursive_telemetry[
                    "intended_complex_decisions"
                ],
                "share": recursive_telemetry[
                    "recursive_share_of_intended_complex_decisions"
                ],
            },
            required={
                "minimum": gates[
                    "minimum_recursive_share_of_intended_complex_decisions"
                ]
            },
        ),
        "deterministic_normal_recursive_six_pass_preflight": _gate(
            _mapping(manifest.get("planner_pass_preflight"), "planner_pass_preflight").get(
                "normal_probe_observed_neural_passes"
            )
            == R198_NORMAL_RECURSIVE_PLAN_PASSES,
            observed={
                "preflight": manifest["planner_pass_preflight"],
                "organic_normal_recursive_plans_observed": recursive_telemetry[
                    "normal_recursive_plans"
                ],
            },
            required={
                "deterministic_probe_passes": R198_NORMAL_RECURSIVE_PLAN_PASSES,
                "any_observed_gameplay_passes_must_match": R198_NORMAL_RECURSIVE_PLAN_PASSES,
            },
        ),
        "deterministic_forced_replan_five_pass_preflight": _gate(
            _mapping(manifest.get("planner_pass_preflight"), "planner_pass_preflight").get(
                "forced_replan_probe_observed_neural_passes"
            )
            == R198_FORCED_REPLAN_PASSES,
            observed={
                "preflight": manifest["planner_pass_preflight"],
                "organic_forced_replans_observed": recursive_telemetry[
                    "forced_replans"
                ],
            },
            required={
                "deterministic_probe_passes": R198_FORCED_REPLAN_PASSES,
                "any_observed_gameplay_passes_must_match": R198_FORCED_REPLAN_PASSES,
            },
        ),
        "no_neural_pass_budget_failures": _gate(
            sum(budget_failures.values()) == 0,
            observed=budget_failures,
            required=0,
        ),
        "unexpected_recursive_fallback_rate": _gate(
            recursive_fallback_rate is not None
            and recursive_fallback_rate
            <= gates["maximum_unexpected_recursive_fallback_rate"],
            observed={
                "unexpected_recursive_fallback_decisions": recursive_telemetry[
                    "unexpected_recursive_fallback_decisions"
                ],
                "intended_complex_decisions": recursive_telemetry[
                    "intended_complex_decisions"
                ],
                "rate": recursive_fallback_rate,
            },
            required={
                "maximum": gates["maximum_unexpected_recursive_fallback_rate"]
            },
        ),
        "no_illegal_action_regression": _gate(
            all(summary["new_count"] == 0 for summary in illegal_regressions.values()),
            observed=illegal_regressions,
            required={"new_illegal_action_count": 0},
        ),
        "no_new_forfeit_regression": _gate(
            all(summary["new_count"] == 0 for summary in forfeit_regressions.values()),
            observed=forfeit_regressions,
            required={"new_forfeit_count": 0},
        ),
        "recursive_effect_lower_bound": _gate(
            comparisons[recursive_minus_bridge_name][
                "one_sided_lower_confidence_bound"
            ]
            > 0.0
            and comparisons[recursive_minus_bridge_name][
                "one_sided_lower_confidence_bound"
            ]
            >= gates["minimum_recursive_delta_lower_bound"],
            observed=comparisons[recursive_minus_bridge_name][
                "one_sided_lower_confidence_bound"
            ],
            required={"minimum": gates["minimum_recursive_delta_lower_bound"]},
        ),
    }
    latency_slo = _mapping(manifest.get("latency_slo"), "latency_slo")
    maximum_latency = latency_slo.get("maximum_p95_latency_seconds")
    owner_latency_slo_authorized = latency_slo.get("owner_authorized") is True
    recursive_latency_distribution = recursive_telemetry[
        "recursive_decision_latency_distribution"
    ]
    recursive_p95_latency = recursive_latency_distribution["p95_seconds"]
    promotion_gates["recursive_p95_decision_latency_slo"] = _gate(
        owner_latency_slo_authorized
        and maximum_latency is not None
        and recursive_latency_distribution["count"] >= MINIMUM_RECURSIVE_DECISIONS
        and recursive_p95_latency is not None
        and recursive_p95_latency <= maximum_latency,
        observed={
            "distribution": recursive_latency_distribution,
            "owner_slo_seconds": maximum_latency,
            "latency_slo_binding": latency_slo,
        },
        required=(
            {
                "minimum_recursive_latency_samples": MINIMUM_RECURSIVE_DECISIONS,
                "maximum_p95_seconds": maximum_latency,
            }
            if maximum_latency is not None
            else "an explicit canonical owner p95 recursive-decision latency SLO"
        ),
    )
    all_passed = all(bool(gate["passed"]) for gate in promotion_gates.values())
    failed = sorted(name for name, gate in promotion_gates.items() if not gate["passed"])
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "ready_for_separate_promotion_review" if all_passed else "hold",
        "manifest": manifest_identity,
        "manifest_input_sha256": manifest["manifest_input_sha256"],
        "schedule_sha256": canonical_digest(manifest["schedule"]),
        "results": results_identity,
        "result_rows_sha256": canonical_digest(rows),
        "frozen_artifacts": {
            "shared_artifacts": manifest["shared_artifacts"],
            "arms": manifest["arms"],
            "opponents": manifest["opponents"],
        },
        "candidate_evaluation_binding": manifest["candidate_evaluation_binding"],
        "r197_source_exclusion_binding": manifest["r197_source_exclusion_binding"],
        "official_control_panel": manifest["official_control_panel"],
        "r198_profile_contract": manifest["r198_profile_contract"],
        "latency_slo": latency_slo,
        "paired_rng_contract": {
            "kind": "checksum_bound_tape_or_restorable_snapshot",
            "capability": manifest["pairing_capability"],
            "evaluation_cg_closure": manifest["evaluation_cg_closure"],
            "requested_seed_is_pairing_proof": False,
            "all_rows_attested_restored_or_replayed": True,
        },
        "arm_summaries": summaries,
        "over_cap_factorized_fallback": {
            "classification": OVER_CAP_FACTORIZED_FALLBACK_REASON,
            "complete_ordered_action_cap": R198_MAX_ACTION_COMBOS,
            "conditional_cross_arm_action_parity": over_cap_parity,
        },
        "comparisons": comparisons,
        "reliability_regressions": {
            "illegal_action": illegal_regressions,
            "new_forfeit": forfeit_regressions,
        },
        "promotion_gates": promotion_gates,
        "promotion_decision": {
            "eligible_for_separate_promotion_review": all_passed,
            "failed_gates": failed,
            "self_promotion_performed": False,
            "serving_change_authorized": False,
            "next_step": (
                "request an explicit receipt-backed promotion review"
                if all_passed
                else "hold RTP serving authority and repair/re-run the failed gates"
            ),
        },
        "evaluation_isolation": {
            "training_eligible": False,
            "replay_eligible": False,
            "formal_gate": False,
            "serving_change_authorized": False,
            "self_promotion_allowed": False,
        },
        "validated_rows": rows,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_material = {
        key: value
        for key, value in receipt.items()
        if key not in {"created_at_utc", "receipt_input_sha256"}
    }
    receipt["receipt_input_sha256"] = canonical_digest(receipt_material)
    return _immutable_json(
        output_path,
        receipt,
        existing_digest_key="receipt_input_sha256",
    )


__all__ = [
    "ARMS",
    "ABSOLUTE_MAX_NEURAL_PASSES",
    "DEFAULT_GATES",
    "MANIFEST_SCHEMA",
    "PAIRING_CAPABILITY_SCHEMA",
    "R198_FORCED_REPLAN_PASSES",
    "R198_MAX_ACTION_COMBOS",
    "R198_MAX_NEURAL_PASSES",
    "R198_NORMAL_RECURSIVE_PLAN_PASSES",
    "R198_SIZING_PROFILE",
    "RECEIPT_SCHEMA",
    "RTPThreeArmEvaluationError",
    "canonical_digest",
    "compile_three_arm_receipt",
    "file_digest",
    "prepare_three_arm_manifest",
    "prepare_three_arm_manifest_from_spec",
    "verify_manifest_frozen_artifacts",
]
