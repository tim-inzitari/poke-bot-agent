"""Public, causal Prize-plan targets for the revision-23 critic sidecar.

This is deliberately a new target family.  It does not alter the revision-21
one-frame Prize-differential overlay or its eight-output sidecar ABI.  The
labels here are suitable for a separately versioned ``prize-plan-v2`` sidecar
only after its own model, calibration, support, and safe-boundary receipts.

The only raw facts consumed are:

* the chosen complete action recorded at ``env_step + 1`` (alignment proof);
* the acting seat's public pre-action remaining Prize counts; and
* exact terminal ``z`` values from train trajectories while fitting ``Phi``.

No terminal after-state, Prize identity, hidden state, simulator branch,
search, RTP, MCTS, or unchosen action is read or emitted.  If a causal
same-seat segment cannot be proved from recorded complete actions, it is
masked.  It is never silently substituted with zero.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMPLETE_ACTION_OVERLAY_SCHEMA = (
    "poke_bot.alakazam_recent20_rtp_complete_action_overlay/v1"
)
PRIZE_PLAN_TARGET_OVERLAY_SCHEMA = "poke_bot.alakazam_prize_plan_target_overlay/v2"
PRIZE_PLAN_TARGET_SCHEMA = "poke_bot.alakazam_prize_plan_target_schema/v2"
PRIZE_PLAN_DAY_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_prize_plan_target_day_manifest/v2"
)
PRIZE_PLAN_DAY_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_target_day_materialization_receipt/v2"
)
PRIZE_PLAN_POTENTIAL_SCHEMA = "poke_bot.alakazam_prize_plan_phi_table/v2"
PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_prize_plan_phi_fit_manifest/v2"
)
PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_phi_fit_receipt/v2"
)
PRIZE_PLAN_TARGET_SET_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_prize_plan_target_set_manifest/v2"
)
PRIZE_PLAN_TARGET_SET_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_target_set_materialization_receipt/v2"
)

OWNER_GOAL_REVISION = 23
PRIZE_PLAN_AUTHORITY_KEY = "revision_23_prize_plan_v2_h3_actor_canary"
HORIZONS = (1, 3, 6, 12)
PRIZE_COUNTS = tuple(range(1, 7))
WINDOW_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)
SPLIT_BY_DAY = {
    **{day: "train" for day in WINDOW_DAYS[:14]},
    **{day: "validation" for day in WINDOW_DAYS[14:17]},
    **{day: "evaluation" for day in WINDOW_DAYS[17:]},
}
_SHA256_PREFIX = "sha256:"
_FORBIDDEN_KEYS = frozenset(
    {
        "opponent_hand_identities",
        "opponent_deck_order",
        "opponent_deck_multiset_sha256",
        "unrevealed_prize_identities",
        "privateState",
        "private_state",
        "simulatorState",
        "simulator_state",
        "transition_after",
        "hidden_state",
    }
)
_TERMINAL_AGENT_STATUSES = frozenset({"DONE", "ERROR", "INVALID", "TIMEOUT"})


class PrizePlanTargetError(RuntimeError):
    """A public-target, identity, split, or publication invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used for identities."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return _SHA256_PREFIX + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path | str, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _sha_hex(value: str) -> str:
    raw = str(value).removeprefix(_SHA256_PREFIX)
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise PrizePlanTargetError(f"invalid SHA-256 identity: {value!r}")
    return raw


def _exact_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _read_only_regular_file(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise PrizePlanTargetError(f"{label} is not a regular file: {resolved}")
    return resolved


def _read_only_directory(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise PrizePlanTargetError(f"{label} is not a regular directory: {resolved}")
    return resolved


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        overlap = {str(key) for key in value}.intersection(_FORBIDDEN_KEYS)
        if overlap:
            raise PrizePlanTargetError(
                "input or target contains forbidden hidden-state keys: "
                f"{sorted(overlap)}"
            )
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def _require_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise PrizePlanTargetError(f"row has invalid {field}")
    return value


def _require_source_member(row: Mapping[str, Any]) -> str:
    member = _require_string(row, "source_member")
    candidate = Path(member)
    if (
        candidate.is_absolute()
        or member in {".", ".."}
        or ".." in candidate.parts
        or "\\" in member
        or member.endswith(("/", "\\"))
    ):
        raise PrizePlanTargetError("source_member is not a safe raw ZIP-member identity")
    return member


def _safe_output_root(path: Path | str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.exists() or result.is_symlink():
        raise PrizePlanTargetError(f"output root already exists: {result}")
    if not result.parent.is_dir() or result.parent.is_symlink():
        raise PrizePlanTargetError(f"output parent is unavailable: {result.parent}")
    return result


def _write_file_create_only(path: Path, body: bytes, *, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        written = 0
        while written < len(body):
            written += os.write(descriptor, body[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _SHA256_PREFIX + hashlib.sha256(body).hexdigest()


def _fsync_directory(path: Path | str) -> None:
    directory = Path(path)
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise PrizePlanTargetError(f"cannot durability-sync directory: {directory}") from exc
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for current, directories, _files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise PrizePlanTargetError("publication tree contains a symlink")
        _fsync_directory(current_path)


def _atomic_publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one sibling root without an overwrite fallback."""

    source_path = Path(source).resolve()
    destination_input = Path(destination).expanduser()
    destination_parent = destination_input.parent.resolve()
    destination_path = destination_parent / destination_input.name
    if source_path.parent != destination_parent or destination_input.is_symlink():
        raise PrizePlanTargetError("atomic publication requires sibling regular directories")
    _fsync_tree(source_path)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PrizePlanTargetError("atomic no-replace publication is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source_path), -100, os.fsencode(destination_path), 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise PrizePlanTargetError("atomic no-replace publication is unavailable")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source_path), os.fsencode(destination_path), 0x00000004)
    else:
        raise PrizePlanTargetError("atomic no-replace publication is unsupported")
    if result != 0:
        failure = ctypes.get_errno()
        if failure in {errno.EEXIST, errno.ENOTEMPTY}:
            raise PrizePlanTargetError(f"output root already exists: {destination_path}")
        raise PrizePlanTargetError(f"atomic no-replace publication failed: errno {failure}")
    _fsync_directory(destination_parent)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise PrizePlanTargetError("artifact path escaped its root") from exc


def _publish_temp_object(temp: Path, directory: Path, suffix: str) -> tuple[Path, str, int]:
    digest = sha256_file(temp)
    final = directory / f"sha256-{_sha_hex(digest)}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temp, final)
    except FileExistsError as exc:
        raise PrizePlanTargetError(f"duplicate content-addressed object: {final}") from exc
    os.chmod(final, 0o444)
    size = final.stat().st_size
    temp.unlink()
    return final, digest, size


def _read_json_object(path: Path | str, *, label: str) -> tuple[Path, Mapping[str, Any], str]:
    resolved = _read_only_regular_file(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrizePlanTargetError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise PrizePlanTargetError(f"{label} is not an object")
    return resolved, payload, sha256_file(resolved)


@dataclass(frozen=True)
class _Program:
    utc_day: str
    source_archive_sha256: str
    source_member: str
    episode_id: str
    acting_seat: int
    env_step: int
    program_identity: str
    selected_action_program: tuple[int, ...]
    successor_program_identity: str | None


@dataclass(frozen=True)
class _PrizeCounts:
    own_remaining: int | None
    opponent_remaining: int | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class _DayInput:
    utc_day: str
    split: str
    complete_action_overlay_path: Path
    complete_action_overlay_sha256: str
    raw_episode_zip_path: Path
    raw_episode_zip_sha256: str


def _expected_authority_semantics() -> dict[str, Any]:
    """The closed r23 public-target contract expected by this pipeline.

    It intentionally names only target-pipeline facts.  Actor activation gates
    and the frozen advantage-provider implementation are separate consumers of
    the same canonical authority and must not be loosened by this reader.
    """

    return {
        "row_schema": PRIZE_PLAN_TARGET_OVERLAY_SCHEMA,
        "manifest_schema": PRIZE_PLAN_TARGET_SET_MANIFEST_SCHEMA,
        "day_manifest_schema": PRIZE_PLAN_DAY_MANIFEST_SCHEMA,
        "day_materialization_receipt_schema": PRIZE_PLAN_DAY_RECEIPT_SCHEMA,
        "target_set_materialization_receipt_schema": PRIZE_PLAN_TARGET_SET_RECEIPT_SCHEMA,
        "row_unit": "one_complete_recorded_chosen_action_program",
        "row_join_identity": [
            "utc_day",
            "source_archive_sha256",
            "source_member",
            "episode_id",
            "acting_seat",
            "env_step",
            "program_identity",
        ],
        "horizon_definition": "next_complete_same_seat_actions_with_all_intervening_opponent_activity_included",
        "segment_start": "public_pre_action_state_for_complete_same_seat_action_i_plus_k",
        "segment_end": "public_pre_action_state_for_complete_same_seat_action_i_plus_k_plus_1",
        "public_evidence_only": [
            "public_remaining_prize_counts",
            "exact_public_transition_and_event_evidence",
            "sealed_complete_action_and_public_observation_alignment",
        ],
        "hidden_prize_identity_or_other_hidden_information_allowed": False,
        "terminal_after_state_inference_allowed": False,
        "terminal_observed_z_is_direct_plan_reward_or_actor_term": False,
        "segment_shaping_reward": "rP_t=gamma*Phi(s_t_plus_1)-Phi(s_t)",
        "horizon_return": "sum_{k=0}^{h-1}gamma^k*rP_{t+k}_over_exact_complete_same_seat_segments",
        "H3_return_requires_exact_segment_count": 3,
        "missing_ambiguous_nonmonotone_or_terminal_censored_evidence_behavior": "mask_target_and_interval_never_assign_zero",
        "m3_requires_all_h3_segments_available": True,
        "closest_valid_diagnostic_target_only_if_exact_target_is_impossible": True,
        "materialization_failure_behavior": "record_measured_schema_or_evidence_blocker_keep_legacy_active_never_fabricate_labels",
    }


def _validate_authority_semantics(authority: Mapping[str, Any]) -> None:
    """Reject semantic drift instead of treating a later wrapper as approval."""

    target = authority.get("public_prize_plan_target")
    if not isinstance(target, Mapping):
        raise PrizePlanTargetError("revision-23 authority lacks prize-plan target semantics")
    expected = _expected_authority_semantics()
    for key, expected_value in expected.items():
        actual = target.get(key)
        if actual != expected_value:
            raise PrizePlanTargetError(
                f"revision-23 prize-plan target semantics drifted at {key}"
            )
    potential = target.get("prize_race_potential")
    if not isinstance(potential, Mapping) or potential != {
        "fit_manifest_schema": PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA,
        "fit_receipt_schema": PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA,
        "frozen_table_schema": PRIZE_PLAN_POTENTIAL_SCHEMA,
        "definition": "Phi(our_remaining,opponent_remaining)=2*P_iso(win|counts)-1",
        "fit_scope": "sealed_train_split_only",
        "fit_examples": "causally_available_public_count_pairs_with_observed_completed_trajectory_win_indicator",
        "smoothing_required": True,
        "monotone_constraints": {
            "Phi_when_our_remaining_count_falls": "must_not_decrease",
            "Phi_when_opponent_remaining_count_falls": "must_not_increase",
        },
        "fit_input_manifest_sha256_bound": True,
        "fit_configuration_sha256_bound": True,
        "frozen_table_sha256_bound": True,
        "validation_evaluation_or_runtime_refit_allowed": False,
    }:
        raise PrizePlanTargetError("revision-23 Prize-potential semantics drifted")
    gamma = target.get("gamma")
    if not isinstance(gamma, Mapping) or gamma != {
        "must_be_explicit_fixed_and_receipt_bound_before_materialization_or_actor_use": True,
        "may_silently_default": False,
        "fit_or_tune_on_validation_evaluation_or_runtime": False,
    }:
        raise PrizePlanTargetError("revision-23 gamma semantics drifted")
    sidecar = authority.get("sidecar_strategy")
    actor = authority.get("actor_advantage")
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("default_safe_implementation") != "separately_versioned_prize_plan_v2_sidecar"
        or sidecar.get("sidecar_schema") != "poke_bot.alakazam_prize_plan_v2_sidecar/v1"
        or sidecar.get("plan_horizons_to_train_and_receipt") != list(HORIZONS)
        or not isinstance(actor, Mapping)
        or actor.get("enabled_formula")
        != "(z-V_existing(s))+0.025*m3*c3*(Q_plan_3(s,a)-V_plan_3(s))"
        or actor.get("selected_nonzero_cumulative_prize_horizon") != 3
        or actor.get("simultaneous_or_additive_H1_H3_H6_H12_actor_terms_allowed")
        is not False
    ):
        raise PrizePlanTargetError("revision-23 sidecar or H3 actor semantics drifted")


def _load_goal_contract(
    path: Path | str, *, expected_sha256: str
) -> tuple[Path, Mapping[str, Any], str]:
    """Bind the current wrapper and its embedded r23 semantic owner.

    A later unrelated top-level goal revision is allowed.  The pipeline refuses
    to infer r23 semantics from that wrapper: the immutable embedded authority
    must still be present, correctly owned, and exact.
    """

    _sha_hex(expected_sha256)
    contract_path = _read_only_regular_file(path, label="goal contract")
    actual_sha = sha256_file(contract_path)
    if actual_sha != expected_sha256:
        raise PrizePlanTargetError("goal contract SHA-256 mismatch")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrizePlanTargetError("goal contract is invalid JSON") from exc
    if not isinstance(contract, Mapping):
        raise PrizePlanTargetError("goal contract is not an object")
    top_revision = _exact_int(contract.get("goal_revision"))
    if top_revision is None or top_revision < OWNER_GOAL_REVISION:
        raise PrizePlanTargetError("goal contract predates the r23 Prize-plan authority")
    authority = contract.get(PRIZE_PLAN_AUTHORITY_KEY)
    if not isinstance(authority, Mapping):
        raise PrizePlanTargetError("goal contract lacks r23 Prize-plan authority")
    if authority.get("owner_goal_revision") != OWNER_GOAL_REVISION:
        raise PrizePlanTargetError("r23 Prize-plan authority owner revision drifted")
    _validate_authority_semantics(authority)
    return contract_path, contract, actual_sha


def prize_plan_target_schema_document() -> dict[str, Any]:
    """The independently versioned, public-only output ABI."""

    return {
        "schema": PRIZE_PLAN_TARGET_SCHEMA,
        "version": 2,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "row_unit": "recorded_complete_chosen_action_program",
        "join_identity": _expected_authority_semantics()["row_join_identity"],
        "plan_horizons": list(HORIZONS),
        "target": {
            "potential": "Phi(own_remaining,opponent_remaining)=2*P_iso(win|counts)-1",
            "segment_reward": "gamma*Phi(next_pre_action_public_counts)-Phi(current_pre_action_public_counts)",
            "return": "sum(k=0..h-1, gamma^k * segment_reward_k)",
            "model_target": "raw_return_value/(1+gamma**h), analytic bound transform with no clipping",
            "actor_advantage_scaling": "separate future train-split-only frozen actor/sidecar receipt; never a target-row transform",
            "horizon_behavior": "exactly_h_complete_same_seat_segments_required; unavailable endpoint masks",
            "terminal_z": "absent_from_target_rows; fit-only provenance is in the sealed Phi artifact",
        },
        "raw_read_scope": [
            "payload.id",
            "payload.statuses and payload.rewards only while fitting Phi from train",
            "steps[env_step][acting_seat].observation.current.yourIndex",
            "steps[env_step][acting_seat].observation.current.players[*].public_prize_count",
            "steps[env_step+1][acting_seat].action only for selected-action alignment",
        ],
        "forbidden_output_fields": sorted(_FORBIDDEN_KEYS),
        "copied_feature_tensors": False,
        "copied_actions": False,
        "terminal_after_state_used": False,
        "shared_segment_proof": "one causal_segments prefix of at most twelve segments per row; each horizon references its exact prefix",
        "search_or_planner_calls": False,
        "counterfactual_or_unchosen_targets": False,
    }


def _parse_program(row: Mapping[str, Any], *, expected_day: str) -> _Program:
    _assert_no_forbidden_keys(row)
    if row.get("schema") != COMPLETE_ACTION_OVERLAY_SCHEMA:
        raise PrizePlanTargetError("foreign complete-action overlay row")
    if row.get("hidden_information_fields_present") is not False:
        raise PrizePlanTargetError("complete-action row is not marked public-only")
    if row.get("complete_action_program_reconstructed") is not True:
        raise PrizePlanTargetError("complete-action row is not a reconstructed complete action")
    if not isinstance(row.get("stages"), list) or not row["stages"]:
        raise PrizePlanTargetError("complete-action row has no factorized stages")
    utc_day = _require_string(row, "utc_day")
    if utc_day != expected_day:
        raise PrizePlanTargetError(
            f"complete-action row day mismatch: expected {expected_day}, got {utc_day}"
        )
    source_archive_sha256 = _require_string(row, "source_archive_sha256")
    _sha_hex(source_archive_sha256)
    seat = _exact_int(row.get("acting_seat"))
    env_step = _exact_int(row.get("env_step"))
    if seat not in (0, 1) or env_step is None or env_step < 0:
        raise PrizePlanTargetError("complete-action row has invalid acting seat or env_step")
    selected = row.get("selected_action_program")
    if not isinstance(selected, list):
        raise PrizePlanTargetError("complete-action row lacks selected action program")
    selected_action: list[int] = []
    for item in selected:
        value = _exact_int(item)
        if value is None or value < 0:
            raise PrizePlanTargetError("selected action program is malformed")
        selected_action.append(value)
    successor = row.get("recorded_successor_program_identity")
    if successor is not None and (not isinstance(successor, str) or not successor):
        raise PrizePlanTargetError("recorded successor identity is malformed")
    return _Program(
        utc_day=utc_day,
        source_archive_sha256=source_archive_sha256,
        source_member=_require_source_member(row),
        episode_id=_require_string(row, "episode_id"),
        acting_seat=seat,
        env_step=env_step,
        program_identity=_require_string(row, "program_identity"),
        selected_action_program=tuple(selected_action),
        successor_program_identity=successor,
    )


def _iter_program_groups(
    overlay_path: Path, *, expected_day: str
) -> Iterator[list[_Program]]:
    """Yield contiguous exact archive/episode/seat groups in source order."""

    current_key: tuple[str, str, str, int] | None = None
    current: list[_Program] = []
    with overlay_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise PrizePlanTargetError(f"blank overlay row at {overlay_path}:{line_number}")
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrizePlanTargetError(
                    f"invalid overlay JSON at {overlay_path}:{line_number}"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise PrizePlanTargetError("complete-action overlay row is not an object")
            program = _parse_program(decoded, expected_day=expected_day)
            key = (
                program.source_archive_sha256,
                program.source_member,
                program.episode_id,
                program.acting_seat,
            )
            if current_key is not None and key != current_key:
                yield current
                current = []
            current_key = key
            current.append(program)
    if current:
        yield current


def _validate_group(programs: Sequence[_Program]) -> list[_Program]:
    if not programs:
        raise PrizePlanTargetError("complete-action group is empty")
    first = programs[0]
    expected = (
        first.source_archive_sha256,
        first.source_member,
        first.episode_id,
        first.acting_seat,
    )
    ids: set[str] = set()
    steps: set[int] = set()
    previous_step: int | None = None
    for program in programs:
        identity = (
            program.source_archive_sha256,
            program.source_member,
            program.episode_id,
            program.acting_seat,
        )
        if identity != expected:
            raise PrizePlanTargetError("complete-action group identity drifted")
        if program.program_identity in ids or program.env_step in steps:
            raise PrizePlanTargetError("duplicate complete action program identity or env_step")
        if previous_step is not None and program.env_step <= previous_step:
            raise PrizePlanTargetError("same-seat complete actions are not strictly ordered")
        ids.add(program.program_identity)
        steps.add(program.env_step)
        previous_step = program.env_step
    return list(programs)


def _unique_zip_member_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.filename in result:
            raise PrizePlanTargetError(
                f"raw episode ZIP has duplicate physical member name: {info.filename}"
            )
        result[info.filename] = info
    return result


def _open_exact_episode(
    archive: zipfile.ZipFile,
    member_index: Mapping[str, zipfile.ZipInfo],
    program: _Program,
) -> Mapping[str, Any]:
    info = member_index.get(program.source_member)
    if info is None:
        raise PrizePlanTargetError(f"raw archive member is absent: {program.source_member}")
    try:
        # Use the resolved ZipInfo rather than a name lookup so duplicate-name
        # handling cannot silently pick a different physical entry.
        with archive.open(info) as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PrizePlanTargetError(
            f"raw archive member is unreadable: {program.source_member}"
        ) from exc
    if not isinstance(payload, Mapping) or str(payload.get("id") or "") != program.episode_id:
        raise PrizePlanTargetError("raw archive member episode identity mismatch")
    return payload


def _agent_at(payload: Mapping[str, Any], *, env_step: int, seat: int) -> Mapping[str, Any] | None:
    steps = payload.get("steps")
    if not isinstance(steps, list) or not 0 <= env_step < len(steps):
        return None
    pair = steps[env_step]
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    agent = pair[seat]
    return agent if isinstance(agent, Mapping) else None


def _public_prize_count(player: Any) -> int | None:
    """Read only a public count or the length of the public Prize-zone list."""

    if not isinstance(player, Mapping):
        return None
    observed: list[int] = []
    for key in ("prize_count", "prizeCount", "remainingPrizes"):
        if key not in player:
            continue
        value = _exact_int(player.get(key))
        if value is None or value < 0:
            return None
        observed.append(value)
    prize_zone = player.get("prize")
    if prize_zone is not None:
        if not isinstance(prize_zone, (list, tuple)):
            return None
        # ``len`` is intentionally the only operation on this zone.
        observed.append(len(prize_zone))
    if not observed or any(value != observed[0] for value in observed[1:]):
        return None
    return observed[0]


def _pre_action_public_prize_counts(
    payload: Mapping[str, Any], program: _Program
) -> _PrizeCounts:
    agent = _agent_at(payload, env_step=program.env_step, seat=program.acting_seat)
    if agent is None:
        return _PrizeCounts(None, None, "pre_action_frame_absent_or_malformed")
    observation = agent.get("observation")
    current = observation.get("current") if isinstance(observation, Mapping) else None
    if not isinstance(current, Mapping):
        return _PrizeCounts(None, None, "pre_action_public_current_absent")
    your_index = _exact_int(current.get("yourIndex"))
    players = current.get("players")
    if your_index not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        return _PrizeCounts(None, None, "pre_action_public_player_mapping_absent")
    # Do not repair a perspective mismatch from a global/private field.
    if your_index != program.acting_seat:
        return _PrizeCounts(None, None, "pre_action_public_actor_index_mismatch")
    own = _public_prize_count(players[your_index])
    opponent = _public_prize_count(players[1 - your_index])
    if own is None or opponent is None:
        return _PrizeCounts(None, None, "pre_action_public_prize_count_absent_or_ambiguous")
    if own not in PRIZE_COUNTS or opponent not in PRIZE_COUNTS:
        return _PrizeCounts(
            own,
            opponent,
            "pre_action_public_prize_count_outside_valid_1_to_6_range",
        )
    return _PrizeCounts(own, opponent, None)


def _assert_raw_action_alignment(payload: Mapping[str, Any], programs: Sequence[_Program]) -> None:
    """Prove every target row still belongs to its exact recorded action."""

    for program in programs:
        agent = _agent_at(
            payload, env_step=program.env_step + 1, seat=program.acting_seat
        )
        if agent is None:
            raise PrizePlanTargetError("raw selected-action frame is absent or malformed")
        raw_action = agent.get("action")
        if not isinstance(raw_action, list):
            raise PrizePlanTargetError("raw selected complete action is absent")
        parsed: list[int] = []
        for item in raw_action:
            value = _exact_int(item)
            if value is None or value < 0:
                raise PrizePlanTargetError("raw selected complete action is malformed")
            parsed.append(value)
        if tuple(parsed) != program.selected_action_program:
            raise PrizePlanTargetError(
                "raw selected complete action disagrees with sealed complete-action overlay"
            )


def _terminal_z_for_fit(payload: Mapping[str, Any], *, seat: int) -> tuple[float | None, str | None]:
    """Read terminal outcome only for the train-only Phi fit.

    This function is never used by segment shaping and its output is never
    copied into a target row.  A draw remains a completed non-win for the
    requested P(win | counts) potential; it does not enter the actor terminal
    branch or become a shaped endpoint.
    """

    statuses = payload.get("statuses")
    if not isinstance(statuses, list) or statuses != ["DONE", "DONE"]:
        return None, "terminal_statuses_not_exact_done_pair"
    rewards = payload.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 2:
        return None, "terminal_rewards_absent_or_malformed"
    left = _finite_number(rewards[0])
    right = _finite_number(rewards[1])
    if left not in {-1.0, 0.0, 1.0} or right not in {-1.0, 0.0, 1.0}:
        return None, "terminal_reward_absent_or_invalid"
    if left != -right:
        return None, "terminal_rewards_not_zero_sum"
    return (left, right)[seat], None


def _is_terminal_program_frame(payload: Mapping[str, Any], program: _Program) -> bool:
    """Detect only explicit recorded terminal status; absent status is neutral.

    The presence of a later aligned complete action already proves a recorded
    nonterminal endpoint for ordinary archives.  This guard catches a malformed
    archive that labels such a frame terminal rather than guessing an after
    state from it.
    """

    agent = _agent_at(payload, env_step=program.env_step, seat=program.acting_seat)
    if agent is None:
        return True
    status = agent.get("status")
    return isinstance(status, str) and status.upper() in _TERMINAL_AGENT_STATUSES


def _parse_day_input(value: Mapping[str, Any]) -> _DayInput:
    day = value.get("utc_day")
    split = value.get("split")
    if not isinstance(day, str) or day not in WINDOW_DAYS:
        raise PrizePlanTargetError("day input has an out-of-window utc_day")
    if split != SPLIT_BY_DAY[day]:
        raise PrizePlanTargetError("day input split does not match exact recent-20 split")
    overlay = _read_only_regular_file(
        value.get("complete_action_overlay_path", ""), label="complete-action overlay"
    )
    raw_zip = _read_only_regular_file(
        value.get("raw_episode_zip_path", ""), label="raw episode ZIP"
    )
    overlay_sha = value.get("complete_action_overlay_sha256")
    raw_sha = value.get("raw_episode_zip_sha256")
    if not isinstance(overlay_sha, str) or not isinstance(raw_sha, str):
        raise PrizePlanTargetError("day input lacks input SHA-256 identities")
    _sha_hex(overlay_sha)
    _sha_hex(raw_sha)
    if sha256_file(overlay) != overlay_sha:
        raise PrizePlanTargetError(f"complete-action overlay SHA-256 mismatch for {day}")
    if sha256_file(raw_zip) != raw_sha:
        raise PrizePlanTargetError(f"raw episode ZIP SHA-256 mismatch for {day}")
    return _DayInput(day, str(split), overlay, overlay_sha, raw_zip, raw_sha)


def _validate_day_inputs(inputs: Iterable[Mapping[str, Any] | _DayInput]) -> list[_DayInput]:
    parsed: list[_DayInput] = []
    for value in inputs:
        if isinstance(value, _DayInput):
            # Rehash even a programmatic caller's object; no caller can make an
            # input training-eligible by claiming it was verified earlier.
            raw = {
                "utc_day": value.utc_day,
                "split": value.split,
                "complete_action_overlay_path": str(value.complete_action_overlay_path),
                "complete_action_overlay_sha256": value.complete_action_overlay_sha256,
                "raw_episode_zip_path": str(value.raw_episode_zip_path),
                "raw_episode_zip_sha256": value.raw_episode_zip_sha256,
            }
            parsed.append(_parse_day_input(raw))
        elif isinstance(value, Mapping):
            parsed.append(_parse_day_input(value))
        else:
            raise PrizePlanTargetError("day input is not an object")
    if len(parsed) != len(WINDOW_DAYS):
        raise PrizePlanTargetError("Prize-plan fit requires exactly 20 day inputs")
    days = [item.utc_day for item in parsed]
    if len(set(days)) != len(days) or set(days) != set(WINDOW_DAYS):
        raise PrizePlanTargetError("day input inventory is not the exact recent-20 window")
    parsed.sort(key=lambda item: item.utc_day)
    return parsed


def _day_input_identity(item: _DayInput) -> dict[str, Any]:
    return {
        "utc_day": item.utc_day,
        "split": item.split,
        "complete_action_overlay": {
            "sha256": item.complete_action_overlay_sha256,
            "size_bytes": item.complete_action_overlay_path.stat().st_size,
            "schema": COMPLETE_ACTION_OVERLAY_SCHEMA,
        },
        "raw_episode_zip": {
            "sha256": item.raw_episode_zip_sha256,
            "size_bytes": item.raw_episode_zip_path.stat().st_size,
        },
    }


def _parse_fit_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fully explicit, train-only Phi-fitting configuration."""

    expected_keys = {
        "algorithm",
        "smoothing_prior_strength",
        "max_iterations",
        "convergence_tolerance",
    }
    if set(value) != expected_keys:
        raise PrizePlanTargetError(
            "Phi fit configuration must contain exactly algorithm, smoothing_prior_strength, "
            "max_iterations, and convergence_tolerance"
        )
    if value.get("algorithm") != "alternating_weighted_2d_isotonic_pava/v1":
        raise PrizePlanTargetError("unsupported or unbound Phi fit algorithm")
    prior = _finite_number(value.get("smoothing_prior_strength"))
    iterations = _exact_int(value.get("max_iterations"))
    tolerance = _finite_number(value.get("convergence_tolerance"))
    if prior is None or not 0.0 < prior <= 1_000_000.0:
        raise PrizePlanTargetError("Phi smoothing_prior_strength must be finite and positive")
    if iterations is None or not 1 <= iterations <= 1_000_000:
        raise PrizePlanTargetError("Phi max_iterations is outside its finite bound")
    if tolerance is None or not 0.0 < tolerance <= 1e-3:
        raise PrizePlanTargetError("Phi convergence_tolerance is outside its finite bound")
    return {
        "algorithm": str(value["algorithm"]),
        "smoothing_prior_strength": float(prior),
        "max_iterations": int(iterations),
        "convergence_tolerance": float(tolerance),
    }


def _weighted_pava(
    values: Sequence[float], weights: Sequence[float], *, increasing: bool
) -> list[float]:
    """Deterministic weighted pool-adjacent-violators projection.

    It is dependency-free and always receives strictly positive smoothed cell
    weights.  A decreasing fit is implemented through a sign flip, preserving
    the exact same deterministic block-merging order.
    """

    if len(values) != len(weights) or not values:
        raise PrizePlanTargetError("PAVA values and weights are malformed")
    signed = [float(item) if increasing else -float(item) for item in values]
    blocks: list[dict[str, float | int]] = []
    for position, (value, weight) in enumerate(zip(signed, weights, strict=True)):
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0.0:
            raise PrizePlanTargetError("PAVA received non-finite or nonpositive weight")
        blocks.append(
            {
                "start": position,
                "end": position,
                "weight": float(weight),
                "sum": float(value * weight),
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["sum"]) / float(left["weight"])
            right_mean = float(right["sum"]) / float(right["weight"])
            if left_mean <= right_mean:
                break
            merged = {
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "sum": float(left["sum"]) + float(right["sum"]),
            }
            blocks[-2:] = [merged]
    result = [0.0] * len(values)
    for block in blocks:
        mean = float(block["sum"]) / float(block["weight"])
        for position in range(int(block["start"]), int(block["end"]) + 1):
            result[position] = mean if increasing else -mean
    return result


def _fit_monotone_potential_table(
    counts: Mapping[tuple[int, int], Mapping[str, int]],
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the frozen 6×6 public Prize-race potential from train labels only.

    Alternating weighted row/column isotonic projections enforce the two
    partial orders.  The initial values use a train-derived global win prior
    with explicit positive pseudo-count strength, so unsupported cells are
    smoothed instead of becoming invented hard values or NaNs.
    """

    parsed = _parse_fit_configuration(config)
    total = sum(int(cell.get("labeled_count", 0)) for cell in counts.values())
    wins = sum(int(cell.get("win_count", 0)) for cell in counts.values())
    if total <= 0:
        raise PrizePlanTargetError("Phi fit has no exact completed train outcomes")
    global_win_probability = wins / total
    prior_strength = float(parsed["smoothing_prior_strength"])
    values: list[list[float]] = []
    weights: list[list[float]] = []
    cell_rows: list[dict[str, Any]] = []
    for own in PRIZE_COUNTS:
        value_row: list[float] = []
        weight_row: list[float] = []
        for opponent in PRIZE_COUNTS:
            raw = counts.get((own, opponent), {})
            labeled = int(raw.get("labeled_count", 0))
            win_count = int(raw.get("win_count", 0))
            draw_count = int(raw.get("draw_count", 0))
            loss_count = int(raw.get("loss_count", 0))
            if min(labeled, win_count, draw_count, loss_count) < 0:
                raise PrizePlanTargetError("Phi raw count table contains a negative count")
            if win_count + draw_count + loss_count != labeled:
                raise PrizePlanTargetError("Phi raw count table does not reconcile")
            weight = labeled + prior_strength
            smoothed = (win_count + prior_strength * global_win_probability) / weight
            value_row.append(smoothed)
            weight_row.append(weight)
            cell_rows.append(
                {
                    "our_remaining": own,
                    "opponent_remaining": opponent,
                    "labeled_count": labeled,
                    "win_count": win_count,
                    "draw_count": draw_count,
                    "loss_count": loss_count,
                    "smoothed_win_probability_before_isotonic": smoothed,
                    "isotonic_weight": weight,
                }
            )
        values.append(value_row)
        weights.append(weight_row)
    max_delta = math.inf
    iterations = 0
    for iterations in range(1, int(parsed["max_iterations"]) + 1):
        before = [row[:] for row in values]
        # At fixed own remaining, larger opponent remaining is better for us:
        # P(win) is nondecreasing across opponent count 1..6.
        for own_index in range(len(PRIZE_COUNTS)):
            values[own_index] = _weighted_pava(
                values[own_index], weights[own_index], increasing=True
            )
        # At fixed opponent remaining, larger own remaining is worse for us:
        # P(win) is nonincreasing across own count 1..6.
        for opponent_index in range(len(PRIZE_COUNTS)):
            column = [values[row][opponent_index] for row in range(len(PRIZE_COUNTS))]
            column_weights = [
                weights[row][opponent_index] for row in range(len(PRIZE_COUNTS))
            ]
            projected = _weighted_pava(column, column_weights, increasing=False)
            for row, projected_value in enumerate(projected):
                values[row][opponent_index] = projected_value
        max_delta = max(
            abs(values[row][column] - before[row][column])
            for row in range(len(PRIZE_COUNTS))
            for column in range(len(PRIZE_COUNTS))
        )
        if max_delta <= float(parsed["convergence_tolerance"]):
            break
    else:
        raise PrizePlanTargetError("weighted 2D isotonic Phi fit did not converge")
    tolerance = float(parsed["convergence_tolerance"])
    for own_index in range(len(PRIZE_COUNTS)):
        for opponent_index in range(len(PRIZE_COUNTS) - 1):
            if values[own_index][opponent_index] > values[own_index][opponent_index + 1] + tolerance:
                raise PrizePlanTargetError("Phi table violates opponent-count monotonicity")
    for own_index in range(len(PRIZE_COUNTS) - 1):
        for opponent_index in range(len(PRIZE_COUNTS)):
            if values[own_index][opponent_index] + tolerance < values[own_index + 1][opponent_index]:
                raise PrizePlanTargetError("Phi table violates own-count monotonicity")
    by_pair = {(row["our_remaining"], row["opponent_remaining"]): row for row in cell_rows}
    for own_index, own in enumerate(PRIZE_COUNTS):
        for opponent_index, opponent in enumerate(PRIZE_COUNTS):
            row = by_pair[(own, opponent)]
            probability = values[own_index][opponent_index]
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise PrizePlanTargetError("Phi fit yielded a nonfinite or unbounded probability")
            row["isotonic_win_probability"] = probability
            row["phi"] = 2.0 * probability - 1.0
    diagnostics = {
        "algorithm": parsed["algorithm"],
        "iterations": iterations,
        "final_max_abs_delta": max_delta,
        "converged": True,
        "monotonicity_validated": True,
        "global_train_win_probability": global_win_probability,
        "train_terminal_labeled_count": total,
        "train_terminal_win_count": wins,
    }
    table = {
        "schema": PRIZE_PLAN_POTENTIAL_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "table_shape": [6, 6],
        "count_axes": {
            "our_remaining": list(PRIZE_COUNTS),
            "opponent_remaining": list(PRIZE_COUNTS),
        },
        "definition": "Phi(our_remaining,opponent_remaining)=2*P_iso(win|counts)-1",
        "fit_scope": "sealed_train_split_only",
        "smoothing": {
            "prior_kind": "global_train_win_probability_pseudocount",
            "prior_strength": prior_strength,
        },
        "fit_configuration": parsed,
        "diagnostics": diagnostics,
        "cells": cell_rows,
        "terminal_z_role": "train_fit_outcome_only_never_segment_shaping_endpoint",
        "public_only": True,
    }
    return table, diagnostics


def _phi_lookup(table: Mapping[tuple[int, int], float], counts: _PrizeCounts) -> float:
    if counts.unavailable_reason is not None:
        raise PrizePlanTargetError("attempted Phi lookup for unavailable public Prize counts")
    assert counts.own_remaining is not None and counts.opponent_remaining is not None
    try:
        value = table[(counts.own_remaining, counts.opponent_remaining)]
    except KeyError as exc:
        raise PrizePlanTargetError("Phi table does not cover one valid public Prize pair") from exc
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise PrizePlanTargetError("Phi table value is nonfinite or out of range")
    return value


def fit_prize_plan_potential_v2(
    *,
    day_inputs: Iterable[Mapping[str, Any] | _DayInput],
    output_root: Path | str,
    goal_contract_path: Path | str,
    expected_goal_contract_sha256: str,
    fit_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the train-only monotone public Prize-race potential.

    All 20 immutable input identities are validated and recorded; raw terminal
    outcomes are opened only for the fourteen train days.  Validation and
    evaluation episode payloads are not decoded by this fit.
    """

    inputs = _validate_day_inputs(day_inputs)
    contract_path, contract, contract_sha = _load_goal_contract(
        goal_contract_path, expected_sha256=expected_goal_contract_sha256
    )
    contract_revision = _exact_int(contract.get("goal_revision"))
    if contract_revision is None:
        raise PrizePlanTargetError("goal contract revision is invalid")
    parsed_config = _parse_fit_configuration(fit_configuration)
    final_root = _safe_output_root(output_root)
    private_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.private-", dir=final_root.parent)
    )
    objects = private_root / "objects"
    manifests = private_root / "manifests"
    receipts = private_root / "receipts"
    schemas = private_root / "schemas"
    for directory in (objects, manifests, receipts, schemas):
        directory.mkdir()
    started = time.time()
    raw_counts: dict[tuple[int, int], Counter[str]] = {
        (own, opponent): Counter()
        for own in PRIZE_COUNTS
        for opponent in PRIZE_COUNTS
    }
    unavailable: Counter[str] = Counter()
    train_counters: Counter[str] = Counter()
    # Fitting itself also refuses a repeated group across the train split;
    # otherwise one completed outcome could be counted twice under duplicate
    # source day evidence.
    closed_groups: set[tuple[str, str, str, int]] = set()
    seen_members: dict[tuple[str, str], str] = {}
    for day_input in inputs:
        if day_input.split != "train":
            continue
        source_archive_sha: str | None = None
        with zipfile.ZipFile(day_input.raw_episode_zip_path) as archive:
            member_index = _unique_zip_member_index(archive)
            for raw_group in _iter_program_groups(
                day_input.complete_action_overlay_path, expected_day=day_input.utc_day
            ):
                programs = _validate_group(raw_group)
                first = programs[0]
                group_key = (
                    first.source_archive_sha256,
                    first.source_member,
                    first.episode_id,
                    first.acting_seat,
                )
                if group_key in closed_groups:
                    raise PrizePlanTargetError("complete-action group is non-contiguous")
                closed_groups.add(group_key)
                if source_archive_sha is None:
                    source_archive_sha = first.source_archive_sha256
                elif source_archive_sha != first.source_archive_sha256:
                    raise PrizePlanTargetError("one overlay day references multiple raw archive hashes")
                if first.source_archive_sha256 != day_input.raw_episode_zip_sha256:
                    raise PrizePlanTargetError("overlay raw archive SHA does not match day ZIP")
                episode_key = (first.source_archive_sha256, first.episode_id)
                previous_member = seen_members.get(episode_key)
                if previous_member is None:
                    seen_members[episode_key] = first.source_member
                elif previous_member != first.source_member:
                    raise PrizePlanTargetError("one raw archive episode has conflicting source members")
                payload = _open_exact_episode(archive, member_index, first)
                _assert_raw_action_alignment(payload, programs)
                z, terminal_reason = _terminal_z_for_fit(payload, seat=first.acting_seat)
                counts = [_pre_action_public_prize_counts(payload, item) for item in programs]
                for program, public_counts in zip(programs, counts, strict=True):
                    train_counters["complete_action_programs_seen"] += 1
                    if _is_terminal_program_frame(payload, program):
                        unavailable["terminal_censored_or_terminal_program_frame"] += 1
                        train_counters["terminal_fit_masked"] += 1
                        continue
                    if terminal_reason is not None or z is None:
                        unavailable[str(terminal_reason)] += 1
                        train_counters["terminal_fit_masked"] += 1
                        continue
                    if public_counts.unavailable_reason is not None:
                        unavailable[public_counts.unavailable_reason] += 1
                        train_counters["terminal_fit_masked"] += 1
                        continue
                    assert public_counts.own_remaining is not None
                    assert public_counts.opponent_remaining is not None
                    cell = raw_counts[(public_counts.own_remaining, public_counts.opponent_remaining)]
                    cell["labeled_count"] += 1
                    cell["win_count" if z == 1.0 else "draw_count" if z == 0.0 else "loss_count"] += 1
                    train_counters["terminal_fit_labeled"] += 1
        if source_archive_sha is None:
            raise PrizePlanTargetError(f"complete-action overlay is empty for train day {day_input.utc_day}")
    count_document = {
        "schema": "poke_bot.alakazam_prize_plan_phi_fit_inputs/v2",
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "fit_scope": "sealed_train_split_only",
        "input_days": [_day_input_identity(item) for item in inputs],
        "fit_days": [item.utc_day for item in inputs if item.split == "train"],
        "raw_terminal_count_cells": [
            {
                "our_remaining": own,
                "opponent_remaining": opponent,
                "labeled_count": int(raw_counts[(own, opponent)]["labeled_count"]),
                "win_count": int(raw_counts[(own, opponent)]["win_count"]),
                "draw_count": int(raw_counts[(own, opponent)]["draw_count"]),
                "loss_count": int(raw_counts[(own, opponent)]["loss_count"]),
            }
            for own in PRIZE_COUNTS
            for opponent in PRIZE_COUNTS
        ],
        "coverage": {
            "counts": {key: int(value) for key, value in sorted(train_counters.items())},
            "masked_unavailable_reasons": {
                key: int(value) for key, value in sorted(unavailable.items())
            },
        },
        "terminal_z_role": "completed_train_outcome_for_Phi_fit_only_not_target_row_or_shaping_endpoint",
        "validation_evaluation_payloads_opened_for_fit": False,
        "hidden_information_read": False,
        "search_simulator_rtp_mcts_called": False,
    }
    count_body = canonical_bytes(count_document)
    count_sha = _SHA256_PREFIX + hashlib.sha256(count_body).hexdigest()
    count_path = objects / f"sha256-{_sha_hex(count_sha)}.phi-fit-inputs.json"
    _write_file_create_only(count_path, count_body)
    table, diagnostics = _fit_monotone_potential_table(raw_counts, config=parsed_config)
    table["fit_input_manifest_sha256"] = count_sha
    table["fit_configuration_sha256"] = canonical_sha256(parsed_config)
    table_body = canonical_bytes(table)
    table_sha = _SHA256_PREFIX + hashlib.sha256(table_body).hexdigest()
    table_path = objects / f"sha256-{_sha_hex(table_sha)}.phi-table.json"
    _write_file_create_only(table_path, table_body)
    schema = prize_plan_target_schema_document()
    schema_sha = canonical_sha256(schema)
    schema_path = schemas / f"sha256-{_sha_hex(schema_sha)}.target-schema.json"
    _write_file_create_only(schema_path, canonical_bytes(schema))
    fit_manifest = {
        "schema": PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract": {
            "sha256": contract_sha,
            "goal_revision": contract_revision,
            "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
            "semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        },
        "fit_scope": "sealed_train_split_only",
        "all_input_days": [_day_input_identity(item) for item in inputs],
        "fit_input_manifest": {
            "path": _relative(count_path, private_root),
            "sha256": count_sha,
            "size_bytes": count_path.stat().st_size,
        },
        "fit_configuration": parsed_config,
        "fit_configuration_sha256": canonical_sha256(parsed_config),
        "frozen_phi_table": {
            "path": _relative(table_path, private_root),
            "sha256": table_sha,
            "size_bytes": table_path.stat().st_size,
            "schema": PRIZE_PLAN_POTENTIAL_SCHEMA,
        },
        "target_schema": {
            "path": _relative(schema_path, private_root),
            "sha256": schema_sha,
        },
        "diagnostics": diagnostics,
        "split_isolation": {
            "fit_days": [item.utc_day for item in inputs if item.split == "train"],
            "validation_days_not_opened_for_fit": [
                item.utc_day for item in inputs if item.split == "validation"
            ],
            "evaluation_days_not_opened_for_fit": [
                item.utc_day for item in inputs if item.split == "evaluation"
            ],
        },
        "publication": {
            "create_only": True,
            "atomic_root_no_replace": True,
            "input_paths_read_only": True,
        },
    }
    manifest_body = canonical_bytes(fit_manifest)
    manifest_sha = _SHA256_PREFIX + hashlib.sha256(manifest_body).hexdigest()
    manifest_path = manifests / f"sha256-{_sha_hex(manifest_sha)}.phi-fit-manifest.json"
    _write_file_create_only(manifest_path, manifest_body)
    receipt = {
        "schema": PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_sha256": contract_sha,
        "goal_contract_goal_revision": contract_revision,
        "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
        "phi_fit_manifest_path": _relative(manifest_path, private_root),
        "phi_fit_manifest_sha256": manifest_sha,
        "fit_input_manifest_sha256": count_sha,
        "fit_configuration_sha256": canonical_sha256(parsed_config),
        "frozen_phi_table_sha256": table_sha,
        "fit_scope": "sealed_train_split_only",
        "validation_evaluation_or_runtime_refit": False,
        "hidden_information_read": False,
        "terminal_z_used_only_for_train_phi_fit": True,
        "search_simulator_rtp_mcts_called": False,
        "recollection_or_training_performed": False,
        "sealed_at_unix_seconds": time.time(),
        "elapsed_seconds": max(0.0, time.time() - started),
    }
    receipt_body = canonical_bytes(receipt)
    receipt_sha = _SHA256_PREFIX + hashlib.sha256(receipt_body).hexdigest()
    receipt_path = receipts / f"sha256-{_sha_hex(receipt_sha)}.phi-fit-receipt.json"
    _write_file_create_only(receipt_path, receipt_body)
    _atomic_publish_directory_noreplace(private_root, final_root)
    return {
        "output_root": str(final_root),
        "fit_manifest_path": str(final_root / _relative(manifest_path, private_root)),
        "fit_manifest_sha256": manifest_sha,
        "fit_receipt_path": str(final_root / _relative(receipt_path, private_root)),
        "fit_receipt_sha256": receipt_sha,
        "phi_table_path": str(final_root / _relative(table_path, private_root)),
        "phi_table_sha256": table_sha,
        "fit_input_manifest_sha256": count_sha,
        "fit_configuration_sha256": canonical_sha256(parsed_config),
    }


def _artifact_member(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PrizePlanTargetError(f"{label} has no portable relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PrizePlanTargetError(f"{label} escaped its artifact root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PrizePlanTargetError(f"{label} escaped its artifact root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PrizePlanTargetError(f"{label} is not a regular artifact member")
    return resolved


def _load_phi_artifact(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_contract_sha256: str,
) -> tuple[Path, Mapping[str, Any], str, Mapping[tuple[int, int], float]]:
    manifest_path, manifest, manifest_sha = _read_json_object(path, label="Phi fit manifest")
    _sha_hex(expected_sha256)
    if manifest_sha != expected_sha256:
        raise PrizePlanTargetError("Phi fit manifest SHA-256 mismatch")
    if manifest.get("schema") != PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA:
        raise PrizePlanTargetError("foreign Phi fit manifest")
    if manifest.get("owner_goal_revision") != OWNER_GOAL_REVISION:
        raise PrizePlanTargetError("Phi fit manifest owner revision drifted")
    goal = manifest.get("goal_contract")
    if (
        not isinstance(goal, Mapping)
        or goal.get("sha256") != expected_contract_sha256
        or goal.get("required_authority") != PRIZE_PLAN_AUTHORITY_KEY
        or goal.get("semantic_owner_goal_revision") != OWNER_GOAL_REVISION
    ):
        raise PrizePlanTargetError("Phi fit manifest goal-contract binding drifted")
    artifact_root = manifest_path.parent.parent.resolve()
    table_info = manifest.get("frozen_phi_table")
    if not isinstance(table_info, Mapping):
        raise PrizePlanTargetError("Phi fit manifest lacks frozen table")
    table_path = _artifact_member(artifact_root, table_info.get("path"), label="Phi table")
    table_sha = table_info.get("sha256")
    if not isinstance(table_sha, str) or sha256_file(table_path) != table_sha:
        raise PrizePlanTargetError("Phi table SHA-256 mismatch")
    if table_path.stat().st_size != table_info.get("size_bytes"):
        raise PrizePlanTargetError("Phi table size mismatch")
    try:
        table = json.loads(table_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrizePlanTargetError("Phi table is invalid JSON") from exc
    if not isinstance(table, Mapping) or table.get("schema") != PRIZE_PLAN_POTENTIAL_SCHEMA:
        raise PrizePlanTargetError("foreign Phi table")
    if table.get("owner_goal_revision") != OWNER_GOAL_REVISION:
        raise PrizePlanTargetError("Phi table owner revision drifted")
    if table.get("fit_configuration") != manifest.get("fit_configuration"):
        raise PrizePlanTargetError("Phi table fit configuration drifted")
    if table.get("fit_configuration_sha256") != manifest.get("fit_configuration_sha256"):
        raise PrizePlanTargetError("Phi table configuration SHA binding drifted")
    if table.get("fit_input_manifest_sha256") != (manifest.get("fit_input_manifest") or {}).get("sha256"):
        raise PrizePlanTargetError("Phi table fit-input binding drifted")
    cells = table.get("cells")
    if not isinstance(cells, list) or len(cells) != 36:
        raise PrizePlanTargetError("Phi table does not have the complete 6x6 cell inventory")
    lookup: dict[tuple[int, int], float] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise PrizePlanTargetError("Phi table cell is malformed")
        own = _exact_int(cell.get("our_remaining"))
        opponent = _exact_int(cell.get("opponent_remaining"))
        value = _finite_number(cell.get("phi"))
        probability = _finite_number(cell.get("isotonic_win_probability"))
        if own not in PRIZE_COUNTS or opponent not in PRIZE_COUNTS or value is None or probability is None:
            raise PrizePlanTargetError("Phi table cell has invalid public count or value")
        if not -1.0 <= value <= 1.0 or not 0.0 <= probability <= 1.0:
            raise PrizePlanTargetError("Phi table cell is out of range")
        if not math.isclose(value, 2.0 * probability - 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise PrizePlanTargetError("Phi table cell violates its stated definition")
        if (own, opponent) in lookup:
            raise PrizePlanTargetError("Phi table duplicates a public count cell")
        lookup[(own, opponent)] = value
    if set(lookup) != {(own, opponent) for own in PRIZE_COUNTS for opponent in PRIZE_COUNTS}:
        raise PrizePlanTargetError("Phi table does not cover each public count pair exactly once")
    tolerance = _finite_number((manifest.get("fit_configuration") or {}).get("convergence_tolerance"))
    if tolerance is None:
        raise PrizePlanTargetError("Phi fit manifest lacks convergence tolerance")
    for own in PRIZE_COUNTS:
        for opponent in range(1, 6):
            if lookup[(own, opponent)] > lookup[(own, opponent + 1)] + tolerance * 2.0:
                raise PrizePlanTargetError("Phi table opponent-count monotonicity drifted")
    for own in range(1, 6):
        for opponent in PRIZE_COUNTS:
            if lookup[(own, opponent)] + tolerance * 2.0 < lookup[(own + 1, opponent)]:
                raise PrizePlanTargetError("Phi table own-count monotonicity drifted")
    return artifact_root, manifest, manifest_sha, lookup


def _parse_gamma(value: Any) -> float:
    gamma = _finite_number(value)
    if gamma is None or not 0.0 < gamma <= 1.0:
        raise PrizePlanTargetError("gamma must be explicit, finite, and in (0, 1]")
    return float(gamma)


def _segment_unavailable_reason(
    programs: Sequence[_Program],
    counts: Sequence[_PrizeCounts],
    payload: Mapping[str, Any],
    *,
    index: int,
) -> str | None:
    endpoint = index + 1
    if endpoint >= len(programs):
        return "no_later_complete_same_seat_action_terminal_censored_or_incomplete"
    current = programs[index]
    following = programs[endpoint]
    if following.env_step <= current.env_step:
        return "same_seat_env_steps_not_strictly_increasing"
    if current.successor_program_identity != following.program_identity:
        return "same_seat_successor_program_link_incomplete_or_ambiguous"
    if _is_terminal_program_frame(payload, current) or _is_terminal_program_frame(payload, following):
        return "terminal_censored_complete_same_seat_segment"
    before = counts[index]
    after = counts[endpoint]
    if before.unavailable_reason is not None:
        return before.unavailable_reason
    if after.unavailable_reason is not None:
        return after.unavailable_reason
    assert before.own_remaining is not None and before.opponent_remaining is not None
    assert after.own_remaining is not None and after.opponent_remaining is not None
    if after.own_remaining > before.own_remaining or after.opponent_remaining > before.opponent_remaining:
        return "non_monotone_public_prize_count"
    return None


def _masked_plan_target(
    *,
    horizon: int,
    gamma: float,
    reason: str,
    programs: Sequence[_Program],
    counts: Sequence[_PrizeCounts],
    index: int,
    completed_segment_count: int,
) -> dict[str, Any]:
    endpoint = index + horizon
    future = programs[endpoint] if endpoint < len(programs) else None
    current = counts[index]
    final = counts[endpoint] if endpoint < len(counts) else None
    return {
        "h": horizon,
        "mask": False,
        "unavailable_reason": reason,
        "gamma": gamma,
        "segment_count": completed_segment_count,
        "required_segment_count": horizon,
        "first_future_program_identity": (
            programs[index + 1].program_identity if index + 1 < len(programs) else None
        ),
        "first_future_env_step": programs[index + 1].env_step if index + 1 < len(programs) else None,
        "final_future_program_identity": None if future is None else future.program_identity,
        "final_future_env_step": None if future is None else future.env_step,
        "own_remaining_before": current.own_remaining,
        "opponent_remaining_before": current.opponent_remaining,
        "own_remaining_after": None if final is None else final.own_remaining,
        "opponent_remaining_after": None if final is None else final.opponent_remaining,
        "raw_return_value": None,
        "model_target_value": None,
    }


def _plan_target_for_horizon(
    programs: Sequence[_Program],
    counts: Sequence[_PrizeCounts],
    payload: Mapping[str, Any],
    phi: Mapping[tuple[int, int], float],
    *,
    index: int,
    horizon: int,
    gamma: float,
) -> dict[str, Any]:
    """Return an exact-h causal plan return or a reasoned masked target."""

    # Row-level ``causal_segments`` owns the shared proof material.  Each
    # horizon stores only its prefix length/endpoints/return so H1/H3/H6/H12 do
    # not multiply up to 22 copies of the same segment evidence.
    completed: list[dict[str, Any]] = []
    return_value = 0.0
    for offset in range(horizon):
        current_index = index + offset
        reason = _segment_unavailable_reason(
            programs, counts, payload, index=current_index
        )
        if reason is not None:
            return _masked_plan_target(
                horizon=horizon,
                gamma=gamma,
                reason=reason,
                programs=programs,
                counts=counts,
                index=index,
                completed_segment_count=len(completed),
            )
        start = counts[current_index]
        end = counts[current_index + 1]
        start_phi = _phi_lookup(phi, start)
        end_phi = _phi_lookup(phi, end)
        reward = gamma * end_phi - start_phi
        weighted_reward = (gamma**offset) * reward
        if not math.isfinite(reward) or not math.isfinite(weighted_reward):
            raise PrizePlanTargetError("Prize-plan segment return is non-finite")
        current = programs[current_index]
        following = programs[current_index + 1]
        completed.append(
            {
                "segment_index": offset,
                "start_program_identity": current.program_identity,
                "start_env_step": current.env_step,
                "end_program_identity": following.program_identity,
                "end_env_step": following.env_step,
                "intervening_env_step_count": following.env_step - current.env_step - 1,
                "intervening_opponent_activity_included": True,
                "own_remaining_before": start.own_remaining,
                "opponent_remaining_before": start.opponent_remaining,
                "own_remaining_after": end.own_remaining,
                "opponent_remaining_after": end.opponent_remaining,
                "phi_before": start_phi,
                "phi_after": end_phi,
                "segment_shaping_reward": reward,
                "discount_multiplier": gamma**offset,
                "discounted_segment_shaping_reward": weighted_reward,
            }
        )
        return_value += weighted_reward
    # Algebraically this is gamma**h * Phi(end) - Phi(start); keeping the
    # segment sum in the receipt avoids hiding an accidental off-by-one.
    model_bound = 1.0 + gamma**horizon
    if not math.isfinite(return_value) or not -model_bound - 1e-12 <= return_value <= model_bound + 1e-12:
        raise PrizePlanTargetError("Prize-plan return is nonfinite or outside its theoretical bound")
    first = programs[index + 1]
    final = programs[index + horizon]
    return {
        "h": horizon,
        "mask": True,
        "unavailable_reason": None,
        "gamma": gamma,
        "segment_count": horizon,
        "required_segment_count": horizon,
        "first_future_program_identity": first.program_identity,
        "first_future_env_step": first.env_step,
        "final_future_program_identity": final.program_identity,
        "final_future_env_step": final.env_step,
        "own_remaining_before": counts[index].own_remaining,
        "opponent_remaining_before": counts[index].opponent_remaining,
        "own_remaining_after": counts[index + horizon].own_remaining,
        "opponent_remaining_after": counts[index + horizon].opponent_remaining,
        "raw_return_value": return_value,
        # The potential telescopes exactly: G_h = gamma**h Phi(end)-Phi(start),
        # hence its known finite bound is 1 + gamma**h.  This is an analytic
        # target representation transform, not data-dependent fitting or
        # clipping, and is identical on train/validation/evaluation.
        "model_target_value": return_value / model_bound,
    }


def _causal_segments_prefix(
    programs: Sequence[_Program],
    counts: Sequence[_PrizeCounts],
    payload: Mapping[str, Any],
    phi: Mapping[tuple[int, int], float],
    *,
    index: int,
    gamma: float,
) -> list[dict[str, Any]]:
    """Materialize one shared causal prefix (at most H12) for a row.

    It contains no target-value duplication.  Horizon entries reference an
    exact prefix through their segment count and exact endpoint identities.
    A prefix ends as soon as causal evidence ends; its horizon is masked by
    ``_plan_target_for_horizon`` with the same reason.
    """

    segments: list[dict[str, Any]] = []
    for offset in range(max(HORIZONS)):
        current_index = index + offset
        if _segment_unavailable_reason(programs, counts, payload, index=current_index) is not None:
            break
        start = counts[current_index]
        end = counts[current_index + 1]
        start_phi = _phi_lookup(phi, start)
        end_phi = _phi_lookup(phi, end)
        reward = gamma * end_phi - start_phi
        current = programs[current_index]
        following = programs[current_index + 1]
        segments.append(
            {
                "segment_index": offset,
                "start_program_identity": current.program_identity,
                "start_env_step": current.env_step,
                "end_program_identity": following.program_identity,
                "end_env_step": following.env_step,
                "intervening_env_step_count": following.env_step - current.env_step - 1,
                "intervening_opponent_activity_included": True,
                "own_remaining_before": start.own_remaining,
                "opponent_remaining_before": start.opponent_remaining,
                "own_remaining_after": end.own_remaining,
                "opponent_remaining_after": end.opponent_remaining,
                "phi_before": start_phi,
                "phi_after": end_phi,
                "segment_shaping_reward": reward,
                "discount_multiplier": gamma**offset,
                "discounted_segment_shaping_reward": (gamma**offset) * reward,
            }
        )
    return segments


def _coverage_with_required_keys(counters: Mapping[str, int]) -> dict[str, int]:
    required = {"complete_action_programs"}
    for horizon in HORIZONS:
        required.add(f"plan_h{horizon}_labeled")
        required.add(f"plan_h{horizon}_masked")
    required.update(
        {
            "non_monotone_mask_count",
            "terminal_censored_mask_count",
            "missing_or_ambiguous_segment_mask_count",
        }
    )
    result = {str(key): int(value) for key, value in counters.items()}
    for key in required:
        result.setdefault(key, 0)
    return {key: result[key] for key in sorted(result)}


def _assert_phi_input_contains_day(phi_manifest: Mapping[str, Any], day: _DayInput) -> None:
    all_inputs = phi_manifest.get("all_input_days")
    if not isinstance(all_inputs, list):
        raise PrizePlanTargetError("Phi fit manifest lacks all-input inventory")
    matches = [item for item in all_inputs if isinstance(item, Mapping) and item.get("utc_day") == day.utc_day]
    if len(matches) != 1:
        raise PrizePlanTargetError("Phi fit manifest does not bind this target day exactly once")
    expected = _day_input_identity(day)
    if matches[0] != expected:
        raise PrizePlanTargetError("Phi fit manifest input identity drifted for target day")


def build_prize_plan_target_overlay_day(
    *,
    complete_action_overlay_path: Path | str,
    raw_episode_zip_path: Path | str,
    output_root: Path | str,
    utc_day: str,
    split: str,
    goal_contract_path: Path | str,
    expected_goal_contract_sha256: str,
    phi_fit_manifest_path: Path | str,
    expected_phi_fit_manifest_sha256: str,
    gamma: float,
    expected_complete_action_overlay_sha256: str,
    expected_raw_episode_zip_sha256: str,
) -> dict[str, Any]:
    """Materialize one sealed day of raw public Prize-plan labels.

    ``gamma`` has no default by design.  It becomes part of every row and the
    day receipt before a future sidecar can consume these target values.
    """

    source = _parse_day_input(
        {
            "utc_day": utc_day,
            "split": split,
            "complete_action_overlay_path": str(complete_action_overlay_path),
            "complete_action_overlay_sha256": expected_complete_action_overlay_sha256,
            "raw_episode_zip_path": str(raw_episode_zip_path),
            "raw_episode_zip_sha256": expected_raw_episode_zip_sha256,
        }
    )
    gamma_value = _parse_gamma(gamma)
    contract_path, contract, contract_sha = _load_goal_contract(
        goal_contract_path, expected_sha256=expected_goal_contract_sha256
    )
    contract_revision = _exact_int(contract.get("goal_revision"))
    if contract_revision is None:
        raise PrizePlanTargetError("goal contract revision is invalid")
    _phi_root, phi_manifest, phi_manifest_sha, phi_lookup = _load_phi_artifact(
        phi_fit_manifest_path,
        expected_sha256=expected_phi_fit_manifest_sha256,
        expected_contract_sha256=contract_sha,
    )
    _assert_phi_input_contains_day(phi_manifest, source)
    final_root = _safe_output_root(output_root)
    private_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.private-", dir=final_root.parent)
    )
    objects = private_root / "objects"
    manifests = private_root / "manifests"
    receipts = private_root / "receipts"
    schemas = private_root / "schemas"
    for directory in (objects, manifests, receipts, schemas):
        directory.mkdir()
    started = time.time()
    schema = prize_plan_target_schema_document()
    schema_sha = canonical_sha256(schema)
    schema_path = schemas / f"sha256-{_sha_hex(schema_sha)}.target-schema.json"
    _write_file_create_only(schema_path, canonical_bytes(schema))
    temporary_rows = private_root / ".prize-plan-targets.jsonl.partial"
    counters: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    closed_groups: set[tuple[str, str, str, int]] = set()
    seen_members: dict[tuple[str, str], str] = {}
    group_count = 0
    source_archive_sha: str | None = None
    with (
        zipfile.ZipFile(source.raw_episode_zip_path) as archive,
        temporary_rows.open("xb", buffering=8 * 1024 * 1024) as output,
    ):
        member_index = _unique_zip_member_index(archive)
        for raw_group in _iter_program_groups(
            source.complete_action_overlay_path, expected_day=source.utc_day
        ):
            programs = _validate_group(raw_group)
            first = programs[0]
            group_key = (
                first.source_archive_sha256,
                first.source_member,
                first.episode_id,
                first.acting_seat,
            )
            if group_key in closed_groups:
                raise PrizePlanTargetError("complete-action group is non-contiguous")
            closed_groups.add(group_key)
            group_count += 1
            if source_archive_sha is None:
                source_archive_sha = first.source_archive_sha256
            elif source_archive_sha != first.source_archive_sha256:
                raise PrizePlanTargetError("one overlay day references multiple raw archive hashes")
            if first.source_archive_sha256 != source.raw_episode_zip_sha256:
                raise PrizePlanTargetError("overlay raw archive SHA does not match exact day ZIP")
            episode_key = (first.source_archive_sha256, first.episode_id)
            existing_member = seen_members.get(episode_key)
            if existing_member is None:
                seen_members[episode_key] = first.source_member
            elif existing_member != first.source_member:
                raise PrizePlanTargetError("one raw archive episode has conflicting source members")
            payload = _open_exact_episode(archive, member_index, first)
            _assert_raw_action_alignment(payload, programs)
            counts = [_pre_action_public_prize_counts(payload, program) for program in programs]
            for index, program in enumerate(programs):
                causal_segments = _causal_segments_prefix(
                    programs,
                    counts,
                    payload,
                    phi_lookup,
                    index=index,
                    gamma=gamma_value,
                )
                targets = {
                    f"h{horizon}": _plan_target_for_horizon(
                        programs,
                        counts,
                        payload,
                        phi_lookup,
                        index=index,
                        horizon=horizon,
                        gamma=gamma_value,
                    )
                    for horizon in HORIZONS
                }
                row = {
                    "schema": PRIZE_PLAN_TARGET_OVERLAY_SCHEMA,
                    "owner_goal_revision": OWNER_GOAL_REVISION,
                    "goal_contract_goal_revision": contract_revision,
                    "utc_day": program.utc_day,
                    "split": source.split,
                    "source_archive_sha256": program.source_archive_sha256,
                    "source_member": program.source_member,
                    "episode_id": program.episode_id,
                    "acting_seat": program.acting_seat,
                    "env_step": program.env_step,
                    "program_identity": program.program_identity,
                    "causal_segments": causal_segments,
                    "prize_plan_returns": targets,
                    "target_only": True,
                    "hidden_information_fields_present": False,
                    "terminal_z_present": False,
                    "raw_target_provenance": {
                        "pre_action_public_prize_counts_only": True,
                        "selected_complete_action_alignment_verified": True,
                        "all_intervening_same_seat_segment_activity_included": True,
                        "terminal_after_state_inferred": False,
                        "hidden_prize_identity_read": False,
                    },
                }
                _assert_no_forbidden_keys(row)
                output.write(canonical_bytes(row))
                counters["complete_action_programs"] += 1
                for horizon in HORIZONS:
                    target = targets[f"h{horizon}"]
                    name = f"plan_h{horizon}"
                    counters[f"{name}_labeled" if target["mask"] else f"{name}_masked"] += 1
                    if not target["mask"]:
                        reason = str(target["unavailable_reason"])
                        unavailable[reason] += 1
                        if reason == "non_monotone_public_prize_count":
                            counters["non_monotone_mask_count"] += 1
                        if "terminal_censored" in reason:
                            counters["terminal_censored_mask_count"] += 1
                        if (
                            "no_later_complete_same_seat_action" in reason
                            or "ambiguous" in reason
                            or "incomplete" in reason
                        ):
                            counters["missing_or_ambiguous_segment_mask_count"] += 1
        output.flush()
        os.fsync(output.fileno())
    if source_archive_sha is None:
        raise PrizePlanTargetError("complete-action overlay is empty")
    target_path, target_sha, target_size = _publish_temp_object(
        temporary_rows, objects, ".prize-plan-targets.jsonl"
    )
    table_info = phi_manifest.get("frozen_phi_table")
    if not isinstance(table_info, Mapping):
        raise PrizePlanTargetError("Phi fit manifest lost frozen table binding")
    manifest = {
        "schema": PRIZE_PLAN_DAY_MANIFEST_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract": {
            "sha256": contract_sha,
            "goal_revision": contract_revision,
            "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
            "semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        },
        "target_schema": {
            "path": _relative(schema_path, private_root),
            "sha256": schema_sha,
        },
        "utc_day": source.utc_day,
        "split": source.split,
        "gamma": gamma_value,
        "target_value_transform": {
            "formula": "model_target_value=raw_return_value/(1+gamma**h)",
            "gamma": gamma_value,
            "data_dependent_train_fit": False,
            "clipping": False,
            "expected_model_target_range": [-1.0, 1.0],
            "actor_advantage_scaling": "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform",
        },
        "phi_fit_manifest": {
            "sha256": phi_manifest_sha,
            "frozen_phi_table_sha256": table_info.get("sha256"),
            "fit_input_manifest_sha256": (phi_manifest.get("fit_input_manifest") or {}).get("sha256"),
            "fit_configuration_sha256": phi_manifest.get("fit_configuration_sha256"),
        },
        "complete_action_overlay": {
            "sha256": source.complete_action_overlay_sha256,
            "size_bytes": source.complete_action_overlay_path.stat().st_size,
            "schema": COMPLETE_ACTION_OVERLAY_SCHEMA,
        },
        "raw_episode_zip": {
            "sha256": source.raw_episode_zip_sha256,
            "size_bytes": source.raw_episode_zip_path.stat().st_size,
            "source_archive_sha256_verified": source_archive_sha == source.raw_episode_zip_sha256,
        },
        "target_shard": {
            "path": _relative(target_path, private_root),
            "sha256": target_sha,
            "size_bytes": target_size,
            "row_count": int(counters["complete_action_programs"]),
        },
        "coverage": {
            "program_group_count": group_count,
            "counts": _coverage_with_required_keys(counters),
            "masked_unavailable_reasons": {
                key: int(value) for key, value in sorted(unavailable.items())
            },
        },
        "information_boundary": {
            "public_prize_counts_only": True,
            "exact_recorded_chosen_action_alignment_verified": True,
            "terminal_z_absent_from_target_rows": True,
            "terminal_after_state_inferred": False,
            "hidden_information_read_or_output": False,
            "search_simulator_rtp_mcts_called": False,
            "counterfactual_or_unchosen_targets_present": False,
        },
        "publication": {
            "create_only": True,
            "atomic_root_no_replace": True,
            "input_paths_read_only": True,
        },
    }
    manifest_body = canonical_bytes(manifest)
    manifest_sha = _SHA256_PREFIX + hashlib.sha256(manifest_body).hexdigest()
    manifest_path = manifests / f"sha256-{_sha_hex(manifest_sha)}.prize-plan-target-day-manifest.json"
    _write_file_create_only(manifest_path, manifest_body)
    receipt = {
        "schema": PRIZE_PLAN_DAY_RECEIPT_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_sha256": contract_sha,
        "goal_contract_goal_revision": contract_revision,
        "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
        "day_manifest_path": _relative(manifest_path, private_root),
        "day_manifest_sha256": manifest_sha,
        "target_schema_sha256": schema_sha,
        "phi_fit_manifest_sha256": phi_manifest_sha,
        "frozen_phi_table_sha256": table_info.get("sha256"),
        "gamma": gamma_value,
        "target_value_transform": manifest["target_value_transform"],
        "complete_action_overlay_sha256": source.complete_action_overlay_sha256,
        "raw_episode_zip_sha256": source.raw_episode_zip_sha256,
        "target_shard_sha256": target_sha,
        "target_shard_size_bytes": target_size,
        "target_row_count": int(counters["complete_action_programs"]),
        "coverage": manifest["coverage"],
        "input_identity_verified": True,
        "terminal_z_used_as_direct_plan_reward_or_target": False,
        "terminal_after_state_inferred": False,
        "hidden_information_output_fields_present": False,
        "search_simulator_or_planner_called": False,
        "recollection_or_training_performed": False,
        "atomic_root_no_replace": True,
        "sealed_at_unix_seconds": time.time(),
        "elapsed_seconds": max(0.0, time.time() - started),
    }
    receipt_body = canonical_bytes(receipt)
    receipt_sha = _SHA256_PREFIX + hashlib.sha256(receipt_body).hexdigest()
    receipt_path = receipts / f"sha256-{_sha_hex(receipt_sha)}.prize-plan-target-day-receipt.json"
    _write_file_create_only(receipt_path, receipt_body)
    _atomic_publish_directory_noreplace(private_root, final_root)
    return {
        "output_root": str(final_root),
        "day_manifest_path": str(final_root / _relative(manifest_path, private_root)),
        "day_manifest_sha256": manifest_sha,
        "day_receipt_path": str(final_root / _relative(receipt_path, private_root)),
        "day_receipt_sha256": receipt_sha,
        "target_shard_path": str(final_root / _relative(target_path, private_root)),
        "target_shard_sha256": target_sha,
        "target_schema_sha256": schema_sha,
        "coverage": manifest["coverage"],
    }


def _find_one_document(root: Path, *, directory: str, suffix: str, label: str) -> tuple[Path, Mapping[str, Any], str]:
    matches = sorted((root / directory).glob(suffix))
    if len(matches) != 1:
        raise PrizePlanTargetError(f"{label} must have exactly one document")
    return _read_json_object(matches[0], label=label)


def _day_artifact_documents(
    root: Path,
) -> tuple[Path, Mapping[str, Any], str, Path, Mapping[str, Any], str]:
    manifest_path, manifest, manifest_sha = _find_one_document(
        root,
        directory="manifests",
        suffix="*.prize-plan-target-day-manifest.json",
        label="Prize-plan day manifest",
    )
    receipt_path, receipt, receipt_sha = _find_one_document(
        root,
        directory="receipts",
        suffix="*.prize-plan-target-day-receipt.json",
        label="Prize-plan day receipt",
    )
    if manifest.get("schema") != PRIZE_PLAN_DAY_MANIFEST_SCHEMA:
        raise PrizePlanTargetError("foreign Prize-plan day manifest")
    if receipt.get("schema") != PRIZE_PLAN_DAY_RECEIPT_SCHEMA:
        raise PrizePlanTargetError("foreign Prize-plan day receipt")
    return manifest_path, manifest, manifest_sha, receipt_path, receipt, receipt_sha


def _copy_regular_file_create_only(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise PrizePlanTargetError(f"portable copy source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with source.open("rb") as input_stream:
            while True:
                block = input_stream.read(8 * 1024 * 1024)
                if not block:
                    break
                offset = 0
                while offset < len(block):
                    offset += os.write(descriptor, block[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if source.stat().st_size != destination.stat().st_size or sha256_file(source) != sha256_file(destination):
        raise PrizePlanTargetError("portable copy SHA-256/size verification failed")


def _copy_tree_create_only(source: Path, destination: Path) -> None:
    """Copy a closed artifact root while refusing symlinks and overwrite paths."""

    if source.is_symlink() or not source.is_dir() or destination.exists():
        raise PrizePlanTargetError("portable tree copy source/destination is invalid")
    destination.mkdir(parents=True)
    for current, directories, filenames in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        relative = _relative(current_path, source)
        target_directory = destination if relative == "." else destination / relative
        for name in directories:
            source_directory = current_path / name
            if source_directory.is_symlink():
                raise PrizePlanTargetError("portable tree source contains a symlink")
            (target_directory / name).mkdir()
        for name in filenames:
            _copy_regular_file_create_only(current_path / name, target_directory / name)
    _fsync_tree(destination)


def _portable_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory every portable input object, excluding aggregate self-docs."""

    rows: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise PrizePlanTargetError("portable aggregate tree contains a symlink")
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise PrizePlanTargetError("portable aggregate member is not a regular file")
            relative = _relative(path, root)
            if relative.startswith("manifests/") or relative.startswith("receipts/"):
                # Aggregate manifest/receipt are deliberately excluded from the
                # non-self inventory.  Day documents sit under ``days/``.
                continue
            parts = Path(relative).parts
            role = "portable_binding"
            day: str | None = None
            split: str | None = None
            if len(parts) >= 3 and parts[0] == "days" and parts[1] in WINDOW_DAYS:
                day = parts[1]
                role = "day_artifact"
                if ".prize-plan-targets.jsonl" in name:
                    role = "target_shard"
                elif ".target-schema.json" in name:
                    role = "target_schema"
                elif name.endswith(".prize-plan-target-day-manifest.json"):
                    role = "target_day_manifest"
                elif name.endswith(".prize-plan-target-day-receipt.json"):
                    role = "target_day_receipt"
                split = SPLIT_BY_DAY[day]
            elif relative.startswith("bindings/phi_fit/"):
                if ".phi-fit-inputs.json" in name:
                    role = "phi_fit_input_manifest"
                elif ".phi-table.json" in name:
                    role = "phi_table"
                elif ".phi-fit-manifest.json" in name:
                    role = "phi_fit_manifest"
                elif ".phi-fit-receipt.json" in name:
                    role = "phi_fit_receipt"
                else:
                    role = "phi_fit_artifact"
            elif relative.startswith("bindings/goal_contract/"):
                role = "goal_contract"
            elif relative.startswith("bindings/complete_action_overlay_manifest/"):
                role = "complete_action_overlay_manifest"
            elif relative.startswith("bindings/") and "target-value-transform" in name:
                role = "target_value_transform"
            item: dict[str, Any] = {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role": role,
            }
            if day is not None:
                item["utc_day"] = day
                item["split"] = split
            rows.append(item)
    return sorted(rows, key=lambda item: item["relative_path"])


def _validate_phi_receipt(
    phi_root: Path,
    *,
    manifest_sha: str,
    expected_contract_sha: str,
) -> tuple[Path, Mapping[str, Any], str]:
    receipt_path, receipt, receipt_sha = _find_one_document(
        phi_root,
        directory="receipts",
        suffix="*.phi-fit-receipt.json",
        label="Phi fit receipt",
    )
    if (
        receipt.get("schema") != PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA
        or receipt.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or receipt.get("goal_contract_sha256") != expected_contract_sha
        or receipt.get("required_authority") != PRIZE_PLAN_AUTHORITY_KEY
        or receipt.get("phi_fit_manifest_sha256") != manifest_sha
        or receipt.get("fit_scope") != "sealed_train_split_only"
        or receipt.get("validation_evaluation_or_runtime_refit") is not False
        or receipt.get("terminal_z_used_only_for_train_phi_fit") is not True
    ):
        raise PrizePlanTargetError("Phi fit receipt binding drifted")
    return receipt_path, receipt, receipt_sha


def _load_complete_action_overlay_manifest(
    path: Path | str,
    *,
    expected_sha256: str,
    day_identities: Sequence[Mapping[str, Any]],
) -> tuple[Path, Mapping[str, Any], str]:
    manifest_path, manifest, manifest_sha = _read_json_object(
        path, label="complete-action overlay manifest"
    )
    _sha_hex(expected_sha256)
    if manifest_sha != expected_sha256:
        raise PrizePlanTargetError("complete-action overlay manifest SHA-256 mismatch")
    if manifest.get("schema") != "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1":
        raise PrizePlanTargetError("foreign complete-action overlay manifest")
    rows = manifest.get("overlay_shards")
    if not isinstance(rows, list) or len(rows) != len(WINDOW_DAYS):
        raise PrizePlanTargetError("complete-action overlay manifest lacks exact 20-day inventory")
    by_day: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PrizePlanTargetError("complete-action overlay manifest shard is malformed")
        day = row.get("utc_day")
        if not isinstance(day, str) or day not in WINDOW_DAYS or day in by_day:
            raise PrizePlanTargetError("complete-action overlay manifest days drifted")
        if row.get("split") != SPLIT_BY_DAY[day]:
            raise PrizePlanTargetError("complete-action overlay manifest split drifted")
        digest = row.get("sha256")
        if not isinstance(digest, str):
            raise PrizePlanTargetError("complete-action overlay manifest shard lacks SHA")
        _sha_hex(digest)
        by_day[day] = row
    for expected in day_identities:
        day = expected.get("utc_day")
        if not isinstance(day, str):
            raise PrizePlanTargetError("target overlay identity lacks utc_day")
        source = by_day.get(day)
        if (
            source is None
            or source.get("sha256") != expected.get("sha256")
            or source.get("size_bytes") != expected.get("size_bytes")
            or source.get("split") != expected.get("split")
        ):
            raise PrizePlanTargetError(
                "complete-action overlay manifest does not bind target input day"
            )
    return manifest_path, manifest, manifest_sha


def _read_portable_day_rows(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    expected_day: str,
    expected_split: str,
    expected_phi_manifest_sha256: str,
    expected_gamma: float,
) -> Iterator[Mapping[str, Any]]:
    target = manifest.get("target_shard")
    if not isinstance(target, Mapping):
        raise PrizePlanTargetError("day manifest lacks target shard")
    target_path = _artifact_member(root, target.get("path"), label="Prize-plan target shard")
    if (
        target_path.stat().st_size != target.get("size_bytes")
        or sha256_file(target_path) != target.get("sha256")
    ):
        raise PrizePlanTargetError("Prize-plan target shard identity drifted")
    with target_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrizePlanTargetError(
                    f"invalid Prize-plan target row {target_path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise PrizePlanTargetError("Prize-plan target row is not an object")
            if (
                row.get("schema") != PRIZE_PLAN_TARGET_OVERLAY_SCHEMA
                or row.get("owner_goal_revision") != OWNER_GOAL_REVISION
                or row.get("utc_day") != expected_day
                or row.get("split") != expected_split
                or row.get("target_only") is not True
                or row.get("hidden_information_fields_present") is not False
                or row.get("terminal_z_present") is not False
            ):
                raise PrizePlanTargetError("Prize-plan target row core identity drifted")
            _assert_no_forbidden_keys(row)
            yield row


def _validate_row_identity_and_targets(
    row: Mapping[str, Any],
    *,
    expected_day: str,
    expected_split: str,
    expected_gamma: float,
) -> tuple[tuple[str, str, int], str, int, str, Counter[str], list[tuple[int, float]]]:
    source_sha = _require_string(row, "source_archive_sha256")
    _sha_hex(source_sha)
    source_member = _require_source_member(row)
    episode_id = _require_string(row, "episode_id")
    seat = _exact_int(row.get("acting_seat"))
    env_step = _exact_int(row.get("env_step"))
    program_id = _require_string(row, "program_identity")
    if seat not in (0, 1) or env_step is None or env_step < 0:
        raise PrizePlanTargetError("Prize-plan target row has invalid seat/env_step")
    if row.get("utc_day") != expected_day or row.get("split") != expected_split:
        raise PrizePlanTargetError("Prize-plan target row day/split drifted")
    segments = row.get("causal_segments")
    if not isinstance(segments, list) or len(segments) > max(HORIZONS):
        raise PrizePlanTargetError("Prize-plan row causal-segment prefix is malformed")
    previous_end_step: int | None = None
    previous_end_program: str | None = None
    for offset, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or segment.get("segment_index") != offset:
            raise PrizePlanTargetError("Prize-plan causal segments are not contiguous")
        start_step = _exact_int(segment.get("start_env_step"))
        end_step = _exact_int(segment.get("end_env_step"))
        start_program = segment.get("start_program_identity")
        end_program = segment.get("end_program_identity")
        if (
            start_step is None
            or end_step is None
            or end_step <= start_step
            or not isinstance(start_program, str)
            or not isinstance(end_program, str)
            or segment.get("intervening_opponent_activity_included") is not True
        ):
            raise PrizePlanTargetError("Prize-plan causal segment is malformed")
        if offset == 0:
            if start_step != env_step or start_program != program_id:
                raise PrizePlanTargetError("Prize-plan first segment does not begin at row action")
        elif start_step != previous_end_step or start_program != previous_end_program:
            raise PrizePlanTargetError("Prize-plan segment chain is discontinuous")
        previous_end_step, previous_end_program = end_step, end_program
    targets = row.get("prize_plan_returns")
    if not isinstance(targets, Mapping) or set(targets) != {f"h{item}" for item in HORIZONS}:
        raise PrizePlanTargetError("Prize-plan row horizon inventory drifted")
    counters: Counter[str] = Counter({"complete_action_programs": 1})
    labeled: list[tuple[int, float]] = []
    for horizon in HORIZONS:
        target = targets[f"h{horizon}"]
        if not isinstance(target, Mapping):
            raise PrizePlanTargetError("Prize-plan horizon target is malformed")
        if (
            target.get("h") != horizon
            or target.get("gamma") != expected_gamma
            or target.get("required_segment_count") != horizon
        ):
            raise PrizePlanTargetError("Prize-plan horizon identity/gamma drifted")
        masked = target.get("mask") is False
        if target.get("mask") not in {True, False}:
            raise PrizePlanTargetError("Prize-plan horizon mask is malformed")
        segment_count = _exact_int(target.get("segment_count"))
        if segment_count is None or not 0 <= segment_count <= horizon or segment_count > len(segments):
            raise PrizePlanTargetError("Prize-plan horizon segment count is malformed")
        raw = target.get("raw_return_value")
        model_target = target.get("model_target_value")
        if masked:
            if (
                raw is not None
                or model_target is not None
                or not isinstance(target.get("unavailable_reason"), str)
            ):
                raise PrizePlanTargetError("masked Prize-plan horizon fabricated a target")
            counters[f"plan_h{horizon}_masked"] += 1
            reason = str(target["unavailable_reason"])
            if reason == "non_monotone_public_prize_count":
                counters["non_monotone_mask_count"] += 1
            if "terminal_censored" in reason:
                counters["terminal_censored_mask_count"] += 1
            if "no_later_complete_same_seat_action" in reason or "ambiguous" in reason or "incomplete" in reason:
                counters["missing_or_ambiguous_segment_mask_count"] += 1
        else:
            value = _finite_number(raw)
            normalized = _finite_number(model_target)
            bound = 1.0 + expected_gamma**horizon
            if value is None or not -bound - 1e-12 <= value <= bound + 1e-12:
                raise PrizePlanTargetError("labeled Prize-plan raw return is nonfinite/out of bound")
            if normalized is None or not -1.0 - 1e-12 <= normalized <= 1.0 + 1e-12:
                raise PrizePlanTargetError("Prize-plan model target is nonfinite/out of bound")
            if not math.isclose(normalized, value / bound, rel_tol=0.0, abs_tol=1e-12):
                raise PrizePlanTargetError("Prize-plan model target is not its analytic raw-return transform")
            if target.get("unavailable_reason") is not None or segment_count != horizon:
                raise PrizePlanTargetError("labeled Prize-plan horizon is incomplete")
            if horizon > len(segments):
                raise PrizePlanTargetError("labeled Prize-plan horizon lacks shared segment proof")
            summed = sum(
                _finite_number(segment.get("discounted_segment_shaping_reward"))
                for segment in segments[:horizon]
            )
            if not math.isfinite(summed) or not math.isclose(value, summed, rel_tol=0.0, abs_tol=1e-10):
                raise PrizePlanTargetError("Prize-plan raw return disagrees with shared segment proof")
            final = segments[horizon - 1]
            if (
                target.get("first_future_program_identity") != segments[0].get("end_program_identity")
                or target.get("first_future_env_step") != segments[0].get("end_env_step")
                or target.get("final_future_program_identity") != final.get("end_program_identity")
                or target.get("final_future_env_step") != final.get("end_env_step")
            ):
                raise PrizePlanTargetError("Prize-plan target endpoint identity drifted")
            counters[f"plan_h{horizon}_labeled"] += 1
            labeled.append((horizon, value))
    return (source_sha, episode_id, seat), source_member, env_step, program_id, counters, labeled


def finalize_prize_plan_target_set(
    *,
    day_artifact_roots: Iterable[Path | str],
    output_root: Path | str,
    goal_contract_path: Path | str,
    expected_goal_contract_sha256: str,
    phi_fit_manifest_path: Path | str,
    expected_phi_fit_manifest_sha256: str,
    complete_action_overlay_manifest_path: Path | str,
    expected_complete_action_overlay_manifest_sha256: str,
    gamma: float,
) -> dict[str, Any]:
    """Validate exactly twenty day roots and seal one portable target set.

    The aggregate has no raw ZIP, feature, or complete-action JSONL payload.
    It carries only compact target labels plus copied immutable schemas, Phi
    fit/table records, contract, and overlay-manifest identity.  Its inventory
    lets the four-stream transfer verify every eligible target byte before
    local Bert training begins.
    """

    roots = [_read_only_directory(item, label="Prize-plan target day root") for item in day_artifact_roots]
    if len(roots) != len(WINDOW_DAYS) or len(set(roots)) != len(roots):
        raise PrizePlanTargetError("target-set finalizer requires exactly 20 unique day roots")
    gamma_value = _parse_gamma(gamma)
    contract_path, contract, contract_sha = _load_goal_contract(
        goal_contract_path, expected_sha256=expected_goal_contract_sha256
    )
    contract_revision = _exact_int(contract.get("goal_revision"))
    if contract_revision is None:
        raise PrizePlanTargetError("goal contract revision is invalid")
    phi_root, phi_manifest, phi_manifest_sha, _phi = _load_phi_artifact(
        phi_fit_manifest_path,
        expected_sha256=expected_phi_fit_manifest_sha256,
        expected_contract_sha256=contract_sha,
    )
    phi_receipt_path, phi_receipt, phi_receipt_sha = _validate_phi_receipt(
        phi_root, manifest_sha=phi_manifest_sha, expected_contract_sha=contract_sha
    )
    final_root = _safe_output_root(output_root)
    private_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.private-", dir=final_root.parent)
    )
    (private_root / "manifests").mkdir()
    (private_root / "receipts").mkdir()
    (private_root / "bindings").mkdir()
    (private_root / "days").mkdir()
    database_path = private_root / ".split-validation.sqlite3"
    aggregate: Counter[str] = Counter()
    observed_days: set[str] = set()
    observed_splits: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "evaluation": set(),
    }
    day_rows: list[dict[str, Any]] = []
    day_overlay_identities: list[dict[str, Any]] = []
    source_target_shards: list[dict[str, Any]] = []
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE groups (archive_sha TEXT, episode_id TEXT, seat INTEGER, split TEXT, PRIMARY KEY (archive_sha, episode_id, seat))"
        )
        connection.execute(
            "CREATE TABLE episode_members (archive_sha TEXT, episode_id TEXT, source_member TEXT, PRIMARY KEY (archive_sha, episode_id))"
        )
        connection.execute(
            "CREATE TABLE programs (program_identity TEXT PRIMARY KEY, archive_sha TEXT, episode_id TEXT, seat INTEGER, env_step INTEGER, UNIQUE (archive_sha, episode_id, seat, env_step))"
        )
        connection.execute(
            "CREATE TABLE closed_groups (archive_sha TEXT, episode_id TEXT, seat INTEGER, PRIMARY KEY (archive_sha, episode_id, seat))"
        )
        for root in roots:
            (
                day_manifest_path,
                day_manifest,
                day_manifest_sha,
                day_receipt_path,
                day_receipt,
                day_receipt_sha,
            ) = _day_artifact_documents(root)
            day = day_manifest.get("utc_day")
            split = day_manifest.get("split")
            if (
                not isinstance(day, str)
                or day not in WINDOW_DAYS
                or day in observed_days
                or split != SPLIT_BY_DAY[day]
            ):
                raise PrizePlanTargetError("target day or split inventory drifted")
            observed_days.add(day)
            observed_splits[str(split)].add(day)
            goal = day_manifest.get("goal_contract")
            if (
                day_manifest.get("owner_goal_revision") != OWNER_GOAL_REVISION
                or not isinstance(goal, Mapping)
                or goal.get("sha256") != contract_sha
                or goal.get("goal_revision") != contract_revision
                or goal.get("required_authority") != PRIZE_PLAN_AUTHORITY_KEY
                or goal.get("semantic_owner_goal_revision") != OWNER_GOAL_REVISION
            ):
                raise PrizePlanTargetError("day target goal-contract binding drifted")
            transform = day_manifest.get("target_value_transform")
            expected_transform = {
                "formula": "model_target_value=raw_return_value/(1+gamma**h)",
                "gamma": gamma_value,
                "data_dependent_train_fit": False,
                "clipping": False,
                "expected_model_target_range": [-1.0, 1.0],
                "actor_advantage_scaling": "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform",
            }
            if day_manifest.get("gamma") != gamma_value or transform != expected_transform:
                raise PrizePlanTargetError("day target gamma/analytic transform drifted")
            phi_reference = day_manifest.get("phi_fit_manifest")
            if (
                not isinstance(phi_reference, Mapping)
                or phi_reference.get("sha256") != phi_manifest_sha
                or phi_reference.get("frozen_phi_table_sha256")
                != (phi_manifest.get("frozen_phi_table") or {}).get("sha256")
                or phi_reference.get("fit_input_manifest_sha256")
                != (phi_manifest.get("fit_input_manifest") or {}).get("sha256")
                or phi_reference.get("fit_configuration_sha256")
                != phi_manifest.get("fit_configuration_sha256")
            ):
                raise PrizePlanTargetError("day target Phi fit binding drifted")
            if (
                day_receipt.get("owner_goal_revision") != OWNER_GOAL_REVISION
                or day_receipt.get("schema") != PRIZE_PLAN_DAY_RECEIPT_SCHEMA
                or day_receipt.get("goal_contract_sha256") != contract_sha
                or day_receipt.get("goal_contract_goal_revision") != contract_revision
                or day_receipt.get("required_authority") != PRIZE_PLAN_AUTHORITY_KEY
                or day_receipt.get("day_manifest_sha256") != day_manifest_sha
                or day_receipt.get("phi_fit_manifest_sha256") != phi_manifest_sha
                or day_receipt.get("gamma") != gamma_value
                or day_receipt.get("target_value_transform") != expected_transform
                or day_receipt.get("coverage") != day_manifest.get("coverage")
            ):
                raise PrizePlanTargetError("day target receipt binding drifted")
            overlay = day_manifest.get("complete_action_overlay")
            raw_zip = day_manifest.get("raw_episode_zip")
            target_shard = day_manifest.get("target_shard")
            target_schema = day_manifest.get("target_schema")
            if not isinstance(overlay, Mapping) or not isinstance(raw_zip, Mapping) or not isinstance(target_shard, Mapping) or not isinstance(target_schema, Mapping):
                raise PrizePlanTargetError("day target manifest is incomplete")
            if raw_zip.get("source_archive_sha256_verified") is not True:
                raise PrizePlanTargetError("day target source archive was not verified")
            target_path = _artifact_member(root, target_shard.get("path"), label="day target shard")
            if (
                target_path.stat().st_size != target_shard.get("size_bytes")
                or sha256_file(target_path) != target_shard.get("sha256")
            ):
                raise PrizePlanTargetError("day target shard identity drifted")
            schema_path = _artifact_member(root, target_schema.get("path"), label="day target schema")
            if sha256_file(schema_path) != target_schema.get("sha256"):
                raise PrizePlanTargetError("day target schema identity drifted")
            previous_group: tuple[str, str, int] | None = None
            previous_step: int | None = None
            day_counts: Counter[str] = Counter()
            row_count = 0
            for row in _read_portable_day_rows(
                root,
                manifest=day_manifest,
                expected_day=day,
                expected_split=str(split),
                expected_phi_manifest_sha256=phi_manifest_sha,
                expected_gamma=gamma_value,
            ):
                (
                    group,
                    source_member,
                    env_step,
                    program_id,
                    row_counts,
                    _labeled,
                ) = _validate_row_identity_and_targets(
                    row,
                    expected_day=day,
                    expected_split=str(split),
                    expected_gamma=gamma_value,
                )
                if row.get("goal_contract_goal_revision") != contract_revision:
                    raise PrizePlanTargetError("target row contract revision drifted")
                if group[0] != raw_zip.get("sha256"):
                    raise PrizePlanTargetError("target row raw archive identity drifted")
                existing_member = connection.execute(
                    "SELECT source_member FROM episode_members WHERE archive_sha=? AND episode_id=?",
                    group[:2],
                ).fetchone()
                if existing_member is None:
                    connection.execute(
                        "INSERT INTO episode_members VALUES (?, ?, ?)", (*group[:2], source_member)
                    )
                elif existing_member[0] != source_member:
                    raise PrizePlanTargetError("target row source_member conflicts within raw episode")
                if previous_group == group:
                    if previous_step is None or env_step <= previous_step:
                        raise PrizePlanTargetError("target group env_steps are not strictly increasing")
                else:
                    if previous_group is not None:
                        connection.execute("INSERT INTO closed_groups VALUES (?, ?, ?)", previous_group)
                    if connection.execute(
                        "SELECT 1 FROM closed_groups WHERE archive_sha=? AND episode_id=? AND seat=?", group
                    ).fetchone() is not None:
                        raise PrizePlanTargetError("target group is non-contiguous")
                try:
                    connection.execute(
                        "INSERT INTO programs VALUES (?, ?, ?, ?, ?)",
                        (program_id, *group, env_step),
                    )
                except sqlite3.IntegrityError as exc:
                    raise PrizePlanTargetError("target rows duplicate a program identity or group env_step") from exc
                old_split = connection.execute(
                    "SELECT split FROM groups WHERE archive_sha=? AND episode_id=? AND seat=?", group
                ).fetchone()
                if old_split is None:
                    connection.execute("INSERT INTO groups VALUES (?, ?, ?, ?)", (*group, split))
                elif old_split[0] != split:
                    raise PrizePlanTargetError("whole episode/seat group crosses splits")
                day_counts.update(row_counts)
                row_count += 1
                previous_group = group
                previous_step = env_step
            if previous_group is not None:
                connection.execute("INSERT INTO closed_groups VALUES (?, ?, ?)", previous_group)
            if (
                row_count != target_shard.get("row_count")
                or row_count != day_receipt.get("target_row_count")
                or day_counts != Counter(day_manifest.get("coverage", {}).get("counts") or {})
            ):
                raise PrizePlanTargetError("day target row count or coverage receipt drifted")
            aggregate.update(day_counts)
            overlay_sha = str(overlay.get("sha256") or "")
            raw_sha = str(raw_zip.get("sha256") or "")
            _sha_hex(overlay_sha)
            _sha_hex(raw_sha)
            day_overlay_identities.append(
                {
                    "utc_day": day,
                    "split": split,
                    "sha256": overlay_sha,
                    "size_bytes": overlay.get("size_bytes"),
                }
            )
            source_target_shards.append(
                {
                    "utc_day": day,
                    "split": split,
                    "sha256": target_shard.get("sha256"),
                    "size_bytes": target_shard.get("size_bytes"),
                    "row_count": target_shard.get("row_count"),
                }
            )
            portable_day_root = private_root / "days" / day
            _copy_tree_create_only(root, portable_day_root)
            portable_manifest_path = _relative(
                portable_day_root / _relative(day_manifest_path, root), private_root
            )
            portable_receipt_path = _relative(
                portable_day_root / _relative(day_receipt_path, root), private_root
            )
            portable_target_path = _relative(
                portable_day_root / _relative(target_path, root), private_root
            )
            portable_schema_path = _relative(
                portable_day_root / _relative(schema_path, root), private_root
            )
            day_rows.append(
                {
                    "utc_day": day,
                    "split": split,
                    "day_artifact_root": _relative(portable_day_root, private_root),
                    "day_manifest": {"path": portable_manifest_path, "sha256": day_manifest_sha},
                    "day_receipt": {"path": portable_receipt_path, "sha256": day_receipt_sha},
                    "target_schema": {
                        "path": portable_schema_path,
                        "sha256": target_schema.get("sha256"),
                    },
                    "target_shard": {
                        "path": portable_target_path,
                        "sha256": target_shard.get("sha256"),
                        "size_bytes": target_shard.get("size_bytes"),
                        "row_count": target_shard.get("row_count"),
                    },
                    "raw_episode_zip": {
                        "sha256": raw_zip.get("sha256"),
                        "size_bytes": raw_zip.get("size_bytes"),
                    },
                    "complete_action_overlay": {
                        "sha256": overlay.get("sha256"),
                        "size_bytes": overlay.get("size_bytes"),
                    },
                    "coverage": day_manifest.get("coverage"),
                }
            )
        connection.commit()
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    if observed_days != set(WINDOW_DAYS) or any(
        observed_splits[name] != {day for day in WINDOW_DAYS if SPLIT_BY_DAY[day] == name}
        for name in observed_splits
    ):
        raise PrizePlanTargetError("target set split day lists are incomplete")
    day_overlay_identities.sort(key=lambda item: str(item["utc_day"]))
    overlay_manifest_path, overlay_manifest, overlay_manifest_sha = _load_complete_action_overlay_manifest(
        complete_action_overlay_manifest_path,
        expected_sha256=expected_complete_action_overlay_manifest_sha256,
        day_identities=day_overlay_identities,
    )
    day_rows.sort(key=lambda item: item["utc_day"])
    source_target_shards.sort(key=lambda item: item["utc_day"])
    transform = {
        "schema": "poke_bot.alakazam_prize_plan_target_value_transform/v2",
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "formula": "model_target_value=raw_return_value/(1+gamma**h)",
        "gamma": gamma_value,
        "horizons": list(HORIZONS),
        "analytic_raw_return_bound": "abs(raw_return_value)<=1+gamma**h",
        "expected_model_target_range": [-1.0, 1.0],
        "clipping": False,
        "data_dependent_train_fit": False,
        "actor_advantage_scaling": "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform",
        "source_target_shards": source_target_shards,
        "coverage": _coverage_with_required_keys(aggregate),
    }
    transform_body = canonical_bytes(transform)
    transform_sha = _SHA256_PREFIX + hashlib.sha256(transform_body).hexdigest()
    transform_path = private_root / "bindings" / f"sha256-{_sha_hex(transform_sha)}.target-value-transform.json"
    _write_file_create_only(transform_path, transform_body)
    portable_phi_root = private_root / "bindings" / "phi_fit"
    _copy_tree_create_only(phi_root, portable_phi_root)
    portable_contract_dir = private_root / "bindings" / "goal_contract"
    portable_contract_dir.mkdir()
    portable_contract_path = portable_contract_dir / contract_path.name
    _copy_regular_file_create_only(contract_path, portable_contract_path)
    portable_overlay_dir = private_root / "bindings" / "complete_action_overlay_manifest"
    portable_overlay_dir.mkdir()
    portable_overlay_path = portable_overlay_dir / overlay_manifest_path.name
    _copy_regular_file_create_only(overlay_manifest_path, portable_overlay_path)
    portable_phi_manifest_path = _relative(
        portable_phi_root / _relative(Path(phi_fit_manifest_path).resolve(), phi_root), private_root
    )
    portable_phi_receipt_path = _relative(
        portable_phi_root / _relative(phi_receipt_path, phi_root), private_root
    )
    table_info = phi_manifest.get("frozen_phi_table")
    if not isinstance(table_info, Mapping):
        raise PrizePlanTargetError("Phi manifest lacks table after validation")
    portable_phi_table_path = _relative(
        portable_phi_root / str(table_info.get("path")), private_root
    )
    # Re-open copied immutable bindings to ensure no source path escaped into
    # the portable inventory before signing the aggregate manifest.
    copied_manifest = private_root / portable_phi_manifest_path
    if sha256_file(copied_manifest) != phi_manifest_sha:
        raise PrizePlanTargetError("portable Phi manifest copy verification failed")
    copied_receipt = private_root / portable_phi_receipt_path
    if sha256_file(copied_receipt) != phi_receipt_sha:
        raise PrizePlanTargetError("portable Phi receipt copy verification failed")
    copied_table = private_root / portable_phi_table_path
    if sha256_file(copied_table) != table_info.get("sha256"):
        raise PrizePlanTargetError("portable Phi table copy verification failed")
    inventory = _portable_inventory(private_root)
    target_set = {
        "schema": PRIZE_PLAN_TARGET_SET_MANIFEST_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_goal_revision": contract_revision,
        "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
        "goal_contract": {
            "path": _relative(portable_contract_path, private_root),
            "sha256": contract_sha,
            "goal_revision": contract_revision,
            "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
            "semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        },
        "complete_action_overlay_manifest": {
            "path": _relative(portable_overlay_path, private_root),
            "sha256": overlay_manifest_sha,
            "schema": overlay_manifest.get("schema"),
        },
        "phi_fit": {
            "portable_root": _relative(portable_phi_root, private_root),
            "fit_manifest": {"path": portable_phi_manifest_path, "sha256": phi_manifest_sha},
            "fit_receipt": {"path": portable_phi_receipt_path, "sha256": phi_receipt_sha},
            "frozen_phi_table": {
                "path": portable_phi_table_path,
                "sha256": table_info.get("sha256"),
                "schema": PRIZE_PLAN_POTENTIAL_SCHEMA,
            },
            "fit_input_manifest_sha256": (phi_manifest.get("fit_input_manifest") or {}).get("sha256"),
            "fit_configuration_sha256": phi_manifest.get("fit_configuration_sha256"),
            "fit_scope": "sealed_train_split_only",
        },
        "target_value_transform": {
            "path": _relative(transform_path, private_root),
            "sha256": transform_sha,
            "schema": transform["schema"],
        },
        "source_days": list(WINDOW_DAYS),
        "split_days": {name: sorted(days) for name, days in observed_splits.items()},
        "target_days": day_rows,
        "all_20_raw_episode_zip_sha256s": [
            {
                "utc_day": row["utc_day"],
                "sha256": row["raw_episode_zip"]["sha256"],
                "size_bytes": row["raw_episode_zip"]["size_bytes"],
            }
            for row in day_rows
        ],
        "all_20_complete_action_overlay_sha256s": [
            {
                "utc_day": row["utc_day"],
                "sha256": row["complete_action_overlay"]["sha256"],
                "size_bytes": row["complete_action_overlay"]["size_bytes"],
                "split": row["split"],
            }
            for row in day_rows
        ],
        "all_20_target_shards": source_target_shards,
        "coverage": {"counts": _coverage_with_required_keys(aggregate)},
        "whole_day_episode_and_group_split_disjoint": True,
        "portable_objects": inventory,
        "information_boundary": {
            "raw_zip_or_feature_or_complete_action_overlay_payload_copied": False,
            "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed": False,
            "terminal_z_is_direct_plan_target_or_actor_term": False,
        },
        "publication": {
            "create_only": True,
            "atomic_root_no_replace": True,
            "portable_relative_paths_only": True,
        },
    }
    manifest_body = canonical_bytes(target_set)
    manifest_sha = _SHA256_PREFIX + hashlib.sha256(manifest_body).hexdigest()
    manifest_path = private_root / "manifests" / f"sha256-{_sha_hex(manifest_sha)}.prize-plan-target-set-manifest.json"
    _write_file_create_only(manifest_path, manifest_body)
    receipt = {
        "schema": PRIZE_PLAN_TARGET_SET_RECEIPT_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_sha256": contract_sha,
        "goal_contract_goal_revision": contract_revision,
        "required_authority": PRIZE_PLAN_AUTHORITY_KEY,
        "target_set_manifest_path": _relative(manifest_path, private_root),
        "target_set_manifest_sha256": manifest_sha,
        "phi_fit_manifest_sha256": phi_manifest_sha,
        "phi_fit_receipt_sha256": phi_receipt_sha,
        "frozen_phi_table_sha256": table_info.get("sha256"),
        "target_value_transform_sha256": transform_sha,
        "complete_action_overlay_manifest_sha256": overlay_manifest_sha,
        "day_count": len(day_rows),
        "coverage": target_set["coverage"],
        "whole_day_episode_and_group_split_disjoint": True,
        "portable_object_count": len(inventory),
        "raw_zip_or_feature_or_complete_action_overlay_payload_copied": False,
        "terminal_z_used_as_direct_plan_target_or_actor_term": False,
        "atomic_root_no_replace": True,
        "sealed_at_unix_seconds": time.time(),
    }
    receipt_body = canonical_bytes(receipt)
    receipt_sha = _SHA256_PREFIX + hashlib.sha256(receipt_body).hexdigest()
    receipt_path = private_root / "receipts" / f"sha256-{_sha_hex(receipt_sha)}.prize-plan-target-set-receipt.json"
    _write_file_create_only(receipt_path, receipt_body)
    _atomic_publish_directory_noreplace(private_root, final_root)
    return {
        "output_root": str(final_root),
        "target_set_manifest_path": str(final_root / _relative(manifest_path, private_root)),
        "target_set_manifest_sha256": manifest_sha,
        "target_set_receipt_path": str(final_root / _relative(receipt_path, private_root)),
        "target_set_receipt_sha256": receipt_sha,
        "phi_fit_manifest_sha256": phi_manifest_sha,
        "target_value_transform_sha256": transform_sha,
        "portable_object_count": len(inventory),
        "coverage": target_set["coverage"],
    }


__all__ = [
    "COMPLETE_ACTION_OVERLAY_SCHEMA",
    "HORIZONS",
    "OWNER_GOAL_REVISION",
    "PRIZE_PLAN_AUTHORITY_KEY",
    "PRIZE_PLAN_DAY_MANIFEST_SCHEMA",
    "PRIZE_PLAN_DAY_RECEIPT_SCHEMA",
    "PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA",
    "PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA",
    "PRIZE_PLAN_POTENTIAL_SCHEMA",
    "PRIZE_PLAN_TARGET_OVERLAY_SCHEMA",
    "PRIZE_PLAN_TARGET_SCHEMA",
    "PRIZE_PLAN_TARGET_SET_MANIFEST_SCHEMA",
    "PRIZE_PLAN_TARGET_SET_RECEIPT_SCHEMA",
    "SPLIT_BY_DAY",
    "WINDOW_DAYS",
    "PrizePlanTargetError",
    "build_prize_plan_target_overlay_day",
    "canonical_bytes",
    "canonical_sha256",
    "finalize_prize_plan_target_set",
    "fit_prize_plan_potential_v2",
    "prize_plan_target_schema_document",
    "sha256_file",
]
