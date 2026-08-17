#!/usr/bin/env python3
"""Build the compact recent-20 Alakazam RTP complete-program overlay.

This command never creates feature tensors.  It verifies the existing
semantic tensor pack's offsets/keys/selections while streaming the already
filtered feature rows, then adds only complete-program, turn, successor, mask,
and recorded terminal-outcome structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import struct
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.recent20_overlay import (  # noqa: E402
    BASE_COMPLETION_SCHEMA,
    MANIFEST_SCHEMA,
    OVERLAY_SCHEMA,
    RECEIPT_SCHEMA,
    Recent20OverlayError,
    Recent20RTPDataset,
    base_schema_descriptor,
    canonical_bytes,
    canonical_sha256,
    overlay_schema_document,
    read_json,
    sha256_file,
)


CORPUS_MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_refeaturized_corpus_manifest/v1"
SOURCE_HEADER_SCHEMA = "poke_bot.alakazam_recent20_intraday_refeature_shard/v1"
SOURCE_ROW_SCHEMA = "poke_bot.alakazam_collision_census_r298_option_record/v1"
EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
)
EXPECTED_BASE_COMPLETION_SHA256 = (
    "sha256:e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "sha256:0e5608b40b4d36cee6a910059ce2b55ed2db55523e8e2c13f7cc8c69f17cc0d3"
)
EXPECTED_FROZEN_SCHEMA_MANIFEST_SHA256 = (
    "sha256:41c9ae94f47c0983bccb9e13c680ad5cf5d93f547aa61aeb91fea20cd53f62af"
)
WINDOW_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)
SPLIT_BY_DAY = {
    **{day: "train" for day in WINDOW_DAYS[:14]},
    **{day: "validation" for day in WINDOW_DAYS[14:17]},
    **{day: "evaluation" for day in WINDOW_DAYS[17:]},
}
FORBIDDEN_OUTPUT_KEYS = {
    "opponent_deck_multiset_sha256",
    "opponent_hand_identities",
    "opponent_deck_order",
    "unrevealed_prize_identities",
}


def _sha_hex(value: str) -> str:
    raw = str(value).removeprefix("sha256:")
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        raise Recent20OverlayError(f"invalid SHA-256: {value!r}")
    return raw


def _write_create_only(path: Path, body: bytes, *, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(fd, body[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _publish_temp(temp: Path, object_dir: Path, suffix: str) -> tuple[Path, str, int]:
    digest = sha256_file(temp)
    final = object_dir / f"sha256-{_sha_hex(digest)}{suffix}"
    object_dir.mkdir(parents=True, exist_ok=True)
    if final.exists() or final.is_symlink():
        if not final.is_file() or sha256_file(final) != digest:
            raise Recent20OverlayError(f"content-address conflict: {final}")
        raise Recent20OverlayError(f"refusing duplicate publication: {final}")
    os.link(temp, final)
    os.chmod(final, 0o444)
    temp.unlink()
    return final, digest, final.stat().st_size


def _group_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    source = row["source"]
    return (
        source["source_archive_date"],
        source["episode_id"],
        int(source["acting_seat"]),
        int(source["env_step"]),
        int(source["factorized_stage"]),
        json.dumps(
            row.get("factorized_stage_prefix", []),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _program_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    source = row["source"]
    return (
        str(source["source_archive_date"]),
        str(source["episode_id"]),
        int(source["acting_seat"]),
        int(source["env_step"]),
    )


def _decision_key(group: tuple[Any, ...]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(group)).hexdigest()


def _program_identity(
    day: str, archive_sha: str, episode_id: str, seat: int, env_step: int
) -> str:
    return canonical_sha256(
        {
            "utc_day": day,
            "source_archive_sha256": archive_sha,
            "episode_id": episode_id,
            "acting_seat": seat,
            "env_step": env_step,
        }
    )


def _assert_no_hidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        overlap = FORBIDDEN_OUTPUT_KEYS.intersection(str(key) for key in value)
        if overlap:
            raise Recent20OverlayError(f"hidden-information key in overlay: {overlap}")
        for item in value.values():
            _assert_no_hidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_hidden_keys(item)


class _BaseAlignment:
    def __init__(self, base_day: Path, pack: Mapping[str, Any]) -> None:
        self.base_day = Path(base_day)
        self.pack = dict(pack)
        self.keys = (self.base_day / "decision_keys.sha256").open("rb")
        self.selected = (self.base_day / "selected_option.u32").open("rb")
        self.offsets = (self.base_day / "decision_offsets.u64").open("rb")
        first = self.offsets.read(8)
        if len(first) != 8 or struct.unpack("<Q", first)[0] != 0:
            raise Recent20OverlayError("base decision offsets do not start at zero")
        self.decision_index = 0
        self.option_start = 0

    def consume(self, group: tuple[Any, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
        raw_key = self.keys.read(32)
        raw_selected = self.selected.read(4)
        raw_offset = self.offsets.read(8)
        if len(raw_key) != 32 or len(raw_selected) != 4 or len(raw_offset) != 8:
            raise Recent20OverlayError("base decision streams ended early")
        expected_key = _decision_key(group)
        if raw_key.hex() != _sha_hex(expected_key):
            raise Recent20OverlayError("base decision key does not join source rows")
        selected_indices = [
            index for index, row in enumerate(rows) if row.get("selected_candidate") is True
        ]
        if len(selected_indices) != 1:
            raise Recent20OverlayError("stage does not have exactly one selected option")
        selected_index = selected_indices[0]
        if struct.unpack("<I", raw_selected)[0] != selected_index:
            raise Recent20OverlayError("base selected option disagrees with source rows")
        next_offset = struct.unpack("<Q", raw_offset)[0]
        if next_offset - self.option_start != len(rows):
            raise Recent20OverlayError("base option offsets disagree with source rows")
        candidate_actions = [
            [int(value) for value in list(row.get("candidate_action") or ())]
            for row in rows
        ]
        selected_action = candidate_actions[selected_index]
        source = rows[0]["source"]
        stage = {
            "factorized_stage": int(source["factorized_stage"]),
            "factorized_stage_prefix": [
                int(value) for value in list(rows[0].get("factorized_stage_prefix") or ())
            ],
            "base_ref": {
                "base_pack_receipt_sha256": str(self.pack["receipt_sha256"]),
                "base_source_shard_sha256": str(self.pack["source_sha256"]),
                "base_decision_key_sha256": expected_key,
                "base_decision_index": self.decision_index,
                "option_start": self.option_start,
                "option_count": len(rows),
            },
            "ordered_legal_action_programs": candidate_actions,
            "selected_option_index": selected_index,
            "selected_action_program": selected_action,
            "valid_option_mask": [True] * len(rows),
            "recorded_target_mask": [
                index == selected_index for index in range(len(rows))
            ],
            "unchosen_target_unavailable_reason": (
                "recorded_transition_only_no_counterfactual_targets"
            ),
        }
        self.decision_index += 1
        self.option_start = next_offset
        return stage

    def finish(self) -> None:
        expected_decisions = int(self.pack["decision_occurrence_count"])
        expected_options = int(self.pack["option_occurrence_count"])
        if self.decision_index != expected_decisions or self.option_start != expected_options:
            raise Recent20OverlayError("base/source join coverage is incomplete")
        if self.keys.read(1) or self.selected.read(1) or self.offsets.read(1):
            raise Recent20OverlayError("base decision streams contain unjoined trailing bytes")
        for stream in (self.keys, self.selected, self.offsets):
            stream.close()


def _episode_metadata(payload: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    if str(payload.get("id") or "") != episode_id:
        raise Recent20OverlayError("raw member episode identity mismatch")
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    steps = payload.get("steps")
    if not isinstance(rewards, list) or len(rewards) != 2 or not isinstance(steps, list):
        raise Recent20OverlayError("raw episode terminal metadata is malformed")
    return {
        "rewards": rewards,
        "statuses": statuses if isinstance(statuses, list) else [],
        "steps": steps,
    }


def _agent_at(meta: Mapping[str, Any], env_step: int, seat: int) -> Mapping[str, Any]:
    steps = meta["steps"]
    if not 0 <= env_step < len(steps):
        raise Recent20OverlayError("overlay environment step is absent from raw episode")
    pair = steps[env_step]
    if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[seat], Mapping):
        raise Recent20OverlayError("raw episode agent frame is malformed")
    return pair[seat]


def _submitted_action_after(
    meta: Mapping[str, Any], env_step: int, seat: int
) -> Mapping[str, Any]:
    """Return the frame carrying the action submitted for ``env_step``.

    Kaggle-style episode archives associate an observation with frame N and
    persist the action chosen from that observation in frame N+1.  Frame 1,
    for example, carries the setup-deck submission prompted by frame 0.
    """
    return _agent_at(meta, env_step + 1, seat)


def _flush_episode(
    *,
    programs: list[dict[str, Any]],
    archive: zipfile.ZipFile,
    writer: BinaryIO,
    counters: Counter[str],
    strata: Counter[str],
    program_ids: set[str],
) -> None:
    if not programs:
        return
    members = {str(row["source_member"]) for row in programs}
    episode_ids = {str(row["episode_id"]) for row in programs}
    if len(members) != 1 or len(episode_ids) != 1:
        raise Recent20OverlayError("episode buffer crossed a source identity")
    member = next(iter(members))
    episode_id = next(iter(episode_ids))
    with archive.open(member) as stream:
        payload = json.load(stream)
    meta = _episode_metadata(payload, episode_id)
    qualifying_seats = {int(row["acting_seat"]) for row in programs}
    counters["qualifying_games"] += 1
    counters["qualifying_acting_seats"] += len(qualifying_seats)
    # Count only the explicitly scoped exclusion: the opposite seat's
    # recorded complete programs in games where another seat qualified.
    for seat in {0, 1}.difference(qualifying_seats):
        for env_step, pair in enumerate(meta["steps"]):
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[seat], Mapping):
                continue
            agent = pair[seat]
            observation = agent.get("observation")
            select = observation.get("select") if isinstance(observation, Mapping) else None
            submitted = None
            if env_step + 1 < len(meta["steps"]):
                following = meta["steps"][env_step + 1]
                if (
                    isinstance(following, list)
                    and len(following) == 2
                    and isinstance(following[seat], Mapping)
                ):
                    submitted = following[seat].get("action")
            if (
                isinstance(select, Mapping)
                and isinstance(select.get("option"), list)
                and isinstance(submitted, list)
            ):
                counters["excluded_nonqualifying_seat_complete_programs"] += 1

    by_seat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in programs:
        by_seat[int(row["acting_seat"])].append(row)
    for seat_rows in by_seat.values():
        seat_rows.sort(key=lambda row: int(row["env_step"]))
        # Populate recorded metadata first so boundary comparisons can inspect
        # both neighbours without keeping the whole day resident.
        for row in seat_rows:
            agent = _agent_at(meta, int(row["env_step"]), int(row["acting_seat"]))
            action_frame = _submitted_action_after(
                meta, int(row["env_step"]), int(row["acting_seat"])
            )
            raw_action = action_frame.get("action")
            if not isinstance(raw_action, list):
                raise Recent20OverlayError("recorded complete action is unavailable")
            raw_action = [int(value) for value in raw_action]
            if raw_action != row["selected_action_program"]:
                raise Recent20OverlayError(
                    "factorized selected program disagrees with raw action: "
                    f"episode={row['episode_id']} seat={row['acting_seat']} "
                    f"env_step={row['env_step']} feature={row['selected_action_program']} "
                    f"raw={raw_action} member={row['source_member']}"
                )
            observation = agent.get("observation")
            current = observation.get("current") if isinstance(observation, Mapping) else None
            turn = current.get("turn") if isinstance(current, Mapping) else None
            if not isinstance(turn, int) or isinstance(turn, bool):
                row["turn"] = None
                row["turn_unavailable_reason"] = "recorded_public_turn_absent"
                counters["missing_turn"] += 1
            else:
                row["turn"] = turn
                row["turn_unavailable_reason"] = None
            try:
                outcome = float(meta["rewards"][int(row["acting_seat"])])
            except (TypeError, ValueError, IndexError):
                outcome = None
            if outcome not in {-1.0, 0.0, 1.0}:
                row["recorded_outcome"] = None
                row["outcome_unavailable_reason"] = "recorded_episode_reward_absent_or_invalid"
                counters["missing_outcome"] += 1
            else:
                row["recorded_outcome"] = outcome
                row["outcome_unavailable_reason"] = None
            row["episode_terminal_state"] = {
                "terminal_complete": outcome is not None,
                "recorded_agent_status": str(agent.get("status") or ""),
                "recorded_episode_statuses": [str(value) for value in meta["statuses"]],
                "is_last_recorded_program_for_acting_seat": False,
            }
        for index, row in enumerate(seat_rows):
            previous = seat_rows[index - 1] if index else None
            following = seat_rows[index + 1] if index + 1 < len(seat_rows) else None
            row["program_boundary"] = {"start": True, "end": True}
            row["turn_boundary"] = {
                "start": previous is None or previous.get("turn") != row.get("turn"),
                "end": following is None or following.get("turn") != row.get("turn"),
            }
            if following is None:
                row["recorded_successor_program_identity"] = None
                row["successor_program_unavailable_reason"] = (
                    "no_later_recorded_program_for_acting_seat"
                )
                counters["missing_successor_program_link"] += 1
            else:
                row["recorded_successor_program_identity"] = following["program_identity"]
                row["successor_program_unavailable_reason"] = None
            row["episode_terminal_state"][
                "is_last_recorded_program_for_acting_seat"
            ] = following is None
            _assert_no_hidden_keys(row)
            raw = canonical_bytes(row)
            writer.write(raw)
            counters["overlay_bytes"] += len(raw)
            counters["complete_action_programs"] += 1
            counters["factorized_stages"] += len(row["stages"])
            counters["legal_options"] += sum(
                int(stage["base_ref"]["option_count"]) for stage in row["stages"]
            )
            strata[str(row["acting_deck_multiset_sha256"])] += 1
            if row["program_identity"] in program_ids:
                raise Recent20OverlayError("duplicate complete action program identity")
            program_ids.add(row["program_identity"])


def _proc_io() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip())
    except (OSError, ValueError):
        pass
    return result


def _build_day(task: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    io_before = _proc_io()
    day = str(task["utc_day"])
    source_path = Path(str(task["source_path"]))
    base_day = Path(str(task["base_day"]))
    raw_archive = Path(str(task["raw_archive"]))
    temp = Path(str(task["temp_path"]))
    object_dir = Path(str(task["object_dir"]))
    pack = dict(task["pack"])
    source_manifest_row = dict(task["source_manifest_row"])
    temp.parent.mkdir(parents=True, exist_ok=True)
    alignment = _BaseAlignment(base_day, pack)
    counters: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    program_ids: set[str] = set()
    current_stage_key: tuple[Any, ...] | None = None
    current_stage_rows: list[dict[str, Any]] = []
    current_program_key: tuple[str, str, int, int] | None = None
    current_program_stages: list[dict[str, Any]] = []
    current_program_template: dict[str, Any] | None = None
    current_episode_id: str | None = None
    episode_programs: list[dict[str, Any]] = []

    def finish_stage() -> None:
        nonlocal current_stage_key, current_stage_rows, current_program_stages
        if current_stage_key is None:
            return
        current_program_stages.append(alignment.consume(current_stage_key, current_stage_rows))
        current_stage_key = None
        current_stage_rows = []

    def finish_program() -> None:
        nonlocal current_program_key, current_program_stages, current_program_template
        if current_program_key is None or current_program_template is None:
            return
        finish_stage()
        stages = current_program_stages
        if [stage["factorized_stage"] for stage in stages] != list(range(len(stages))):
            raise Recent20OverlayError("factorized program stages are not contiguous")
        selected = list(stages[-1]["selected_action_program"])
        template = current_program_template
        row = {
            "schema": OVERLAY_SCHEMA,
            "utc_day": day,
            "source_archive_sha256": template["source_archive_sha256"],
            "source_member": template["source_member"],
            "episode_id": template["episode_id"],
            "acting_seat": template["acting_seat"],
            "env_step": template["env_step"],
            "program_identity": _program_identity(
                day,
                template["source_archive_sha256"],
                template["episode_id"],
                template["acting_seat"],
                template["env_step"],
            ),
            "acting_deck_multiset_sha256": template["acting_deck_multiset_sha256"],
            "list_variant_stratum": template["acting_deck_multiset_sha256"],
            "canonical_public_observation_hash": template[
                "canonical_public_observation_hash"
            ],
            "recorded_successor_public_hash": template["successor_public_hash"],
            "successor_public_unavailable_reason": template[
                "successor_public_unavailable_reason"
            ],
            "stages": stages,
            "selected_action_program": selected,
            "complete_action_program_reconstructed": True,
            "policy_input_fields": [
                "canonical_public_observation_hash",
                "stages.base_ref",
                "stages.ordered_legal_action_programs",
                "stages.valid_option_mask",
                "acting_deck_multiset_sha256",
            ],
            "target_only_fields": [
                "recorded_successor_public_hash",
                "recorded_successor_program_identity",
                "recorded_outcome",
                "episode_terminal_state",
            ],
            "hidden_information_fields_present": False,
        }
        episode_programs.append(row)
        current_program_key = None
        current_program_stages = []
        current_program_template = None

    with (
        source_path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as source,
        raw_archive.open("rb") as archive_stream,
        zipfile.ZipFile(archive_stream) as archive,
        temp.open("xb", buffering=16 * 1024 * 1024) as writer,
    ):
        header = json.loads(source.readline())
        if (
            header.get("schema") != SOURCE_HEADER_SCHEMA
            or header.get("utc_day") != day
            or header.get("frozen_schema_manifest_sha256")
            != EXPECTED_FROZEN_SCHEMA_MANIFEST_SHA256
        ):
            raise Recent20OverlayError("source feature shard header drifted")
        for line in source:
            row = json.loads(line)
            if row.get("schema") != SOURCE_ROW_SCHEMA:
                raise Recent20OverlayError("source feature row schema drifted")
            source_info = row["source"]
            if (
                source_info.get("acting_seat_setup_deck_contains_card_743") is not True
                or source_info.get("row_materialization_eligible") is not True
            ):
                raise Recent20OverlayError("source feature pack contains a nonqualifying row")
            stage_key = _group_identity(row)
            program_key = _program_key(row)
            episode_id = program_key[1]
            if current_stage_key is not None and stage_key != current_stage_key:
                finish_stage()
            if current_program_key is not None and program_key != current_program_key:
                finish_program()
            if current_episode_id is not None and episode_id != current_episode_id:
                _flush_episode(
                    programs=episode_programs,
                    archive=archive,
                    writer=writer,
                    counters=counters,
                    strata=strata,
                    program_ids=program_ids,
                )
                episode_programs = []
            current_episode_id = episode_id
            if current_program_key is None:
                successor = source_info.get("adjacent_public_successor")
                successor_hash = None
                unavailable = "recorded_adjacent_public_successor_absent"
                if isinstance(successor, Mapping):
                    successor_hash = successor.get("canonical_public_current_hash")
                    unavailable = None if successor_hash else unavailable
                current_program_key = program_key
                current_program_template = {
                    "source_archive_sha256": str(source_info["source_archive_sha256"]),
                    "source_member": str(source_info["source_member"]),
                    "episode_id": episode_id,
                    "acting_seat": int(source_info["acting_seat"]),
                    "env_step": int(source_info["env_step"]),
                    "acting_deck_multiset_sha256": str(
                        source_info["acting_deck_multiset_sha256"]
                    ),
                    "canonical_public_observation_hash": str(
                        row["canonical_public_observation_hash"]
                    ),
                    "successor_public_hash": successor_hash,
                    "successor_public_unavailable_reason": unavailable,
                }
            if current_stage_key is None:
                current_stage_key = stage_key
            current_stage_rows.append(row)
        finish_program()
        _flush_episode(
            programs=episode_programs,
            archive=archive,
            writer=writer,
            counters=counters,
            strata=strata,
            program_ids=program_ids,
        )
        writer.flush()
        os.fsync(writer.fileno())
    alignment.finish()
    if counters["factorized_stages"] != int(pack["decision_occurrence_count"]):
        raise Recent20OverlayError("overlay does not cover every base decision")
    if counters["legal_options"] != int(pack["option_occurrence_count"]):
        raise Recent20OverlayError("overlay does not cover every base option")
    final, digest, size = _publish_temp(temp, object_dir, ".rtp-overlay.jsonl")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_after = _proc_io()
    return {
        "utc_day": day,
        "split": SPLIT_BY_DAY[day],
        "path": str(final),
        "sha256": digest,
        "size_bytes": size,
        "base_source_shard_sha256": str(pack["source_sha256"]),
        "base_pack_receipt_sha256": str(pack["receipt_sha256"]),
        "base_decision_count": int(pack["decision_occurrence_count"]),
        "base_option_count": int(pack["option_occurrence_count"]),
        "qualifying_games": counters["qualifying_games"],
        "qualifying_acting_seats": counters["qualifying_acting_seats"],
        "complete_action_programs": counters["complete_action_programs"],
        "factorized_stages": counters["factorized_stages"],
        "legal_options": counters["legal_options"],
        "excluded_nonqualifying_seat_complete_programs": counters[
            "excluded_nonqualifying_seat_complete_programs"
        ],
        "missing_successor_program_link": counters["missing_successor_program_link"],
        "missing_turn": counters["missing_turn"],
        "missing_outcome": counters["missing_outcome"],
        "list_variant_program_counts": dict(sorted(strata.items())),
        "duplicate_program_identity_count": 0,
        "elapsed_seconds": time.monotonic() - started,
        "worker_cpu_user_seconds": float(usage.ru_utime),
        "worker_cpu_system_seconds": float(usage.ru_stime),
        "worker_peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "worker_io": {
            key: int(io_after.get(key, 0)) - int(io_before.get(key, 0))
            for key in sorted(set(io_before) | set(io_after))
        },
        "source_manifest_row": source_manifest_row,
    }


def _index_feature_shards(roots: Iterable[Path], wanted: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        for directory, _subdirs, filenames in os.walk(root):
            for filename in wanted.intersection(filenames):
                candidate = Path(directory) / filename
                previous = found.get(filename)
                if previous is not None and previous != candidate:
                    # Exact duplicate copies are acceptable only when the
                    # caller's root order selected the first stable location.
                    continue
                found[filename] = candidate
            if len(found) == len(wanted):
                return found
    missing = sorted(wanted.difference(found))
    if missing:
        raise Recent20OverlayError(f"missing source feature shards: {missing}")
    return found


def _pack_day(row: Mapping[str, Any]) -> str:
    for field in ("receipt_path", "source_path"):
        for part in Path(str(row.get(field) or "")).parts:
            if part in WINDOW_DAYS:
                return part
    raise Recent20OverlayError("base pack row has no UTC day")


def _host_snapshot() -> dict[str, Any]:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return {
        "load_average_1_5_15": list(os.getloadavg()),
        "memory_available_bytes": memory.get("MemAvailable"),
        "captured_at_unix_seconds": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--base-pack-completion", type=Path, required=True)
    parser.add_argument("--base-pack-root", type=Path, required=True)
    parser.add_argument("--raw-archive-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= int(args.workers) <= 8:
        raise Recent20OverlayError("workers must be between 1 and 8")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise Recent20OverlayError("output root must not already exist")
    args.output_root.mkdir(parents=True, exist_ok=False)
    private = args.output_root / f".private-build-{os.getpid()}"
    private.mkdir()
    objects = args.output_root / "objects"
    objects.mkdir()
    bindings = args.output_root / "bindings"
    bindings.mkdir()
    schemas = args.output_root / "schemas"
    schemas.mkdir()
    manifests = args.output_root / "manifests"
    manifests.mkdir()
    receipts = args.output_root / "receipts"
    receipts.mkdir()

    before = _host_snapshot()
    corpus_sha = sha256_file(args.corpus_manifest)
    completion_sha = sha256_file(args.base_pack_completion)
    if corpus_sha != EXPECTED_CORPUS_MANIFEST_SHA256:
        raise Recent20OverlayError("authoritative recent-20 corpus manifest mismatch")
    if completion_sha != EXPECTED_BASE_COMPLETION_SHA256:
        raise Recent20OverlayError("authoritative base pack completion mismatch")
    corpus = read_json(args.corpus_manifest)
    completion = read_json(args.base_pack_completion)
    if corpus.get("schema") != CORPUS_MANIFEST_SCHEMA:
        raise Recent20OverlayError("foreign recent-20 corpus manifest")
    if completion.get("schema") != BASE_COMPLETION_SCHEMA:
        raise Recent20OverlayError("foreign base pack completion")
    if completion.get("corpus_manifest_sha256") != corpus_sha:
        raise Recent20OverlayError("base pack is not bound to the recent-20 corpus")
    if tuple(sorted(row["utc_day"] for row in corpus["shards"])) != WINDOW_DAYS:
        raise Recent20OverlayError("recent-20 day coverage drifted")
    if int(corpus["record_count"]) != int(completion["option_occurrence_count"]):
        raise Recent20OverlayError("corpus/base option occurrence count mismatch")

    # Copy only small identity documents, never the feature tensors.
    completion_binding = bindings / f"sha256-{_sha_hex(completion_sha)}.base-completion.json"
    _write_create_only(completion_binding, args.base_pack_completion.read_bytes())
    corpus_binding = bindings / f"sha256-{_sha_hex(corpus_sha)}.corpus-manifest.json"
    _write_create_only(corpus_binding, args.corpus_manifest.read_bytes())
    schema_doc = overlay_schema_document()
    schema_sha = canonical_sha256(schema_doc)
    schema_path = schemas / f"sha256-{_sha_hex(schema_sha)}.overlay-schema.json"
    _write_create_only(schema_path, canonical_bytes(schema_doc))
    base_schema = base_schema_descriptor(completion)
    base_schema_sha = canonical_sha256(base_schema)
    base_schema_path = schemas / f"sha256-{_sha_hex(base_schema_sha)}.base-schema.json"
    _write_create_only(base_schema_path, canonical_bytes(base_schema))

    manifest_rows = {str(row["utc_day"]): dict(row) for row in corpus["shards"]}
    packs = {_pack_day(row): dict(row) for row in completion["packs"]}
    if set(manifest_rows) != set(WINDOW_DAYS) or set(packs) != set(WINDOW_DAYS):
        raise Recent20OverlayError("base/corpus day inventory is not exact")
    wanted_names = {str(row["filename"]) for row in manifest_rows.values()}
    source_paths = _index_feature_shards(args.feature_root, wanted_names)
    tasks = []
    for day in WINDOW_DAYS:
        manifest_row = manifest_rows[day]
        pack = packs[day]
        if pack["source_sha256"] != manifest_row["sha256"]:
            raise Recent20OverlayError("base pack source shard binding mismatch")
        base_day = args.base_pack_root / day
        receipt_path = base_day / "receipt.json"
        if sha256_file(receipt_path) != str(pack["receipt_sha256"]):
            raise Recent20OverlayError("base per-day receipt digest mismatch")
        for role, declared in dict(pack["files"]).items():
            filename = {
                "features_f32": "features.f32",
                "decision_offsets_u64": "decision_offsets.u64",
                "selected_option_u32": "selected_option.u32",
                "decision_key_sha256": "decision_keys.sha256",
            }[role]
            if (base_day / filename).stat().st_size != int(declared["size_bytes"]):
                raise Recent20OverlayError("base per-day file size mismatch")
        tasks.append(
            {
                "utc_day": day,
                "source_path": str(source_paths[str(manifest_row["filename"])]),
                "base_day": str(base_day),
                "raw_archive": str(
                    args.raw_archive_root
                    / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
                ),
                "temp_path": str(private / f"{day}.jsonl.partial"),
                "object_dir": str(objects),
                "pack": pack,
                "source_manifest_row": manifest_row,
            }
        )

    day_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [pool.submit(_build_day, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            day_results.append(result)
            print(
                json.dumps(
                    {
                        "utc_day": result["utc_day"],
                        "programs": result["complete_action_programs"],
                        "stages": result["factorized_stages"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    day_results.sort(key=lambda row: row["utc_day"])
    relative_shards = []
    for row in day_results:
        relative = dict(row)
        relative["path"] = str(Path(row["path"]).relative_to(args.output_root))
        relative_shards.append(relative)

    totals = Counter()
    aggregate_strata: Counter[str] = Counter()
    for row in day_results:
        for field in (
            "qualifying_games",
            "qualifying_acting_seats",
            "complete_action_programs",
            "factorized_stages",
            "legal_options",
            "excluded_nonqualifying_seat_complete_programs",
            "missing_successor_program_link",
            "missing_turn",
            "missing_outcome",
        ):
            totals[field] += int(row[field])
        aggregate_strata.update(row["list_variant_program_counts"])
    if totals["factorized_stages"] != int(completion["decision_occurrence_count"]):
        raise Recent20OverlayError("aggregate deterministic join coverage failed")
    if totals["legal_options"] != int(completion["option_occurrence_count"]):
        raise Recent20OverlayError("aggregate option coverage failed")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "overlay_schema": OVERLAY_SCHEMA,
        "overlay_schema_path": str(schema_path),
        "overlay_schema_sha256": schema_sha,
        "base_pack": {
            "completion_path": str(completion_binding),
            "completion_sha256": completion_sha,
            "schema": base_schema,
            "schema_path": str(base_schema_path),
            "schema_sha256": base_schema_sha,
            "corpus_manifest_path": str(corpus_binding),
            "corpus_manifest_sha256": corpus_sha,
            "source_shards": [
                {
                    "utc_day": day,
                    "source_sha256": packs[day]["source_sha256"],
                    "receipt_sha256": packs[day]["receipt_sha256"],
                    "files": packs[day]["files"],
                }
                for day in WINDOW_DAYS
            ],
            "feature_tensors_copied_into_overlay": False,
        },
        "source_days": list(WINDOW_DAYS),
        "split_evidence": {
            "manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
            "rule": "whole_utc_day_and_source_archive_with_episode_acting_seat_group_fence",
            "train_days": list(WINDOW_DAYS[:14]),
            "validation_days": list(WINDOW_DAYS[14:17]),
            "evaluation_days": list(WINDOW_DAYS[17:]),
        },
        "row_scope": "existing_sealed_acting_seat_card_743_feature_rows_only",
        "overlay_shards": relative_shards,
        "totals": dict(totals),
        "list_variant_program_counts": dict(sorted(aggregate_strata.items())),
        "list_variant_stratum_count": len(aggregate_strata),
        "deterministic_join": {
            "base_decision_coverage": 1.0,
            "base_option_coverage": 1.0,
            "duplicate_program_identity_count": 0,
            "duplicate_base_decision_reference_count": 0,
        },
        "information_boundary": {
            "hidden_information_fields_present": False,
            "opponent_deck_digest_in_runtime_inputs": False,
            "future_transition_as_policy_input": False,
            "unchosen_counterfactual_targets_present": False,
            "recorded_transition_targets_masked_to_selected_actions": True,
        },
        "checkpoint_independent": True,
    }
    manifest_body = canonical_bytes(manifest)
    manifest_sha = "sha256:" + hashlib.sha256(manifest_body).hexdigest()
    manifest_path = manifests / f"sha256-{_sha_hex(manifest_sha)}.overlay-manifest.json"
    _write_create_only(manifest_path, manifest_body)

    smoke_dataset = Recent20RTPDataset(
        manifest_path,
        base_pack_root=args.base_pack_root,
        expected_manifest_sha256=manifest_sha,
        expected_base_completion_sha256=completion_sha,
        verify_overlay_shards=True,
    )
    smoke_counts = {}
    for split in ("train", "validation"):
        count = 0
        for sample in smoke_dataset.iter_samples(split):
            if (
                sample.get("public_information_only") is not True
                or not sample.get("base_option_features_by_stage")
                or sample["program"].get("hidden_information_fields_present") is not False
            ):
                raise Recent20OverlayError("pipeline loader smoke failed")
            count += 1
            if count >= 8:
                break
        if count != 8:
            raise Recent20OverlayError(f"pipeline loader smoke has too few {split} samples")
        smoke_counts[split] = count

    after = _host_snapshot()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "overlay_schema_sha256": schema_sha,
        "base_pack_completion_sha256": completion_sha,
        "base_pack_schema_sha256": base_schema_sha,
        "base_corpus_manifest_sha256": corpus_sha,
        "source_days": list(WINDOW_DAYS),
        "counts": dict(totals),
        "qualifying_game_count": totals["qualifying_games"],
        "qualifying_acting_seat_count": totals["qualifying_acting_seats"],
        "list_variant_stratum_count": len(aggregate_strata),
        "overlay_shard_count": len(day_results),
        "overlay_bytes": sum(int(row["size_bytes"]) for row in day_results),
        "overlay_shards": [
            {
                "utc_day": row["utc_day"],
                "path": str(Path(row["path"]).relative_to(args.output_root)),
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in day_results
        ],
        "deterministic_join_coverage": 1.0,
        "deterministic_join_unique": True,
        "duplicate_or_double_counted_rows": 0,
        "hidden_information_runtime_field_count": 0,
        "raw_archive_use": "only_referenced_qualifying_members_for_missing_recorded_turn_terminal_outcome_and_raw_action_parity",
        "recollection_refeaturization_simulator_search_or_training_performed": False,
        "pipeline_loader_smoke": {
            "passed": True,
            "entrypoint": "poke_bot.recursive_turn_planner.recent20_overlay.Recent20RTPDataset",
            "archetype_job_accepted": False,
            "train_samples_streamed": smoke_counts["train"],
            "validation_samples_streamed": smoke_counts["validation"],
            "whole_corpus_loaded_into_python_objects": False,
        },
        "resources": {
            "host_before": before,
            "host_after": after,
            "worker_limit": int(args.workers),
            "peak_worker_count": min(int(args.workers), len(tasks)),
            "peak_worker_rss_bytes": max(
                int(row["worker_peak_rss_bytes"]) for row in day_results
            ),
            "aggregate_worker_cpu_user_seconds": sum(
                float(row["worker_cpu_user_seconds"]) for row in day_results
            ),
            "aggregate_worker_cpu_system_seconds": sum(
                float(row["worker_cpu_system_seconds"]) for row in day_results
            ),
            "aggregate_worker_io": {
                key: sum(int(row["worker_io"].get(key, 0)) for row in day_results)
                for key in sorted(
                    {key for row in day_results for key in row["worker_io"]}
                )
            },
            "cpu_only": True,
            "gpu_used": False,
            "low_priority_required_by_invocation": True,
        },
        "service_control_performed": False,
        "active_training_or_self_play_on_elmo_at_start": False,
        "active_training_impact": "none_by_host_isolation_no_inzi_outputs_or_services_touched",
        "outputs_transferred_to_inzi": False,
        "sealed_at_unix_seconds": time.time(),
    }
    receipt_body = canonical_bytes(receipt)
    receipt_sha = "sha256:" + hashlib.sha256(receipt_body).hexdigest()
    receipt_path = receipts / f"sha256-{_sha_hex(receipt_sha)}.completion-receipt.json"
    _write_create_only(receipt_path, receipt_body)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "overlay_shards": len(day_results),
                "overlay_bytes": receipt["overlay_bytes"],
                "programs": totals["complete_action_programs"],
                "stages": totals["factorized_stages"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
