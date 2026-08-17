"""Deterministic target-only overlays for the revision-21 action critic.

This module deliberately has no dependency on a model, simulator, search, or
planner.  It joins a sealed complete-action JSONL overlay to the exact raw
episode ZIP for one UTC day and emits only target labels keyed by the existing
complete-action program identity.

The raw reader is intentionally narrow: for an action at ``env_step`` it reads
only the acting agent's *pre-action* public ``current.players`` Prize counts,
plus the terminal reward for that acting seat.  It never copies cards from a
Prize zone (or any other hidden state) into the output.
"""

from __future__ import annotations

import hashlib
import ctypes
import errno
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
TARGET_OVERLAY_SCHEMA = "poke_bot.alakazam_action_critic_target_overlay/v1"
TARGET_DAY_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_day_manifest/v1"
TARGET_DAY_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_day_receipt/v1"
TARGET_SET_MANIFEST_SCHEMA = "poke_bot.alakazam_action_critic_target_set_manifest/v1"
TARGET_SET_RECEIPT_SCHEMA = "poke_bot.alakazam_action_critic_target_set_receipt/v1"
# The revision-21 contract's unqualified manifest/receipt are the aggregate
# 20-day target set.  Per-day documents carry the explicit ``DAY`` names.
TARGET_MANIFEST_SCHEMA = TARGET_SET_MANIFEST_SCHEMA
TARGET_RECEIPT_SCHEMA = TARGET_SET_RECEIPT_SCHEMA
TARGET_SCHEMA = "poke_bot.alakazam_action_critic_target_schema/v1"
OWNER_GOAL_REVISION = 21
CRITIC_AUTHORITY_KEY = "revision_21_draw_safe_critic_actor_canary"
HORIZONS = (1, 2, 3)
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
_FORBIDDEN_INPUT_KEYS = frozenset(
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
    }
)


class ActionCriticTargetError(RuntimeError):
    """A source identity or target-only information boundary was violated."""


def canonical_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used for all output identities."""

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
        raise ActionCriticTargetError(f"invalid SHA-256 identity: {value!r}")
    return raw


def _exact_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_only_regular_file(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ActionCriticTargetError(f"{label} is not a regular file: {resolved}")
    return resolved


def _load_goal_contract(
    path: Path | str, *, expected_sha256: str
) -> tuple[Path, Mapping[str, Any], str]:
    """Bind the current contract while pinning the revision-21 critic semantics.

    The top-level goal contract can advance for unrelated work.  The target
    ABI remains owned by the embedded revision-21 authority, so both identities
    are recorded and checked separately rather than treating a later wrapper
    revision as a semantic change to this artifact.
    """

    if not expected_sha256:
        raise ActionCriticTargetError("goal contract SHA-256 is required")
    _sha_hex(expected_sha256)
    contract_path = _read_only_regular_file(path, label="goal contract")
    actual_sha = sha256_file(contract_path)
    if actual_sha != expected_sha256:
        raise ActionCriticTargetError("goal contract SHA-256 mismatch")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionCriticTargetError("goal contract is invalid JSON") from exc
    if not isinstance(contract, Mapping):
        raise ActionCriticTargetError("goal contract is not an object")
    top_revision = _exact_int(contract.get("goal_revision"))
    if top_revision is None or top_revision < OWNER_GOAL_REVISION:
        raise ActionCriticTargetError("goal contract revision predates the revision-21 critic")
    authority = contract.get(CRITIC_AUTHORITY_KEY)
    if not isinstance(authority, Mapping):
        raise ActionCriticTargetError("goal contract lacks revision-21 critic authority")
    if authority.get("owner_goal_revision") != OWNER_GOAL_REVISION:
        raise ActionCriticTargetError("revision-21 critic authority owner revision drifted")
    _validate_revision_21_critic_semantics(authority)
    return contract_path, contract, actual_sha


def _validate_revision_21_critic_semantics(authority: Mapping[str, Any]) -> None:
    """Reject a wrapper contract whose embedded critic semantics have drifted."""

    target_overlay = authority.get("target_overlay")
    actor_advantage = authority.get("actor_advantage")
    if not isinstance(target_overlay, Mapping) or not isinstance(actor_advantage, Mapping):
        raise ActionCriticTargetError("revision-21 critic authority is incomplete")
    expected_join_identity = [
        "utc_day",
        "source_archive_sha256",
        "source_member",
        "episode_id",
        "acting_seat",
        "env_step",
        "program_identity",
    ]
    expected_horizon_fields = [
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
    ]
    expected_manifest_bindings = [
        "goal_contract_sha256",
        "base_pack_completion_sha256",
        "complete_action_overlay_manifest_sha256",
        "all_20_raw_episode_zip_sha256s",
        "all_20_target_shard_sha256s_sizes_rows_and_split",
        "train_validation_evaluation_day_lists",
        "episode_and_seat_group_split_disjointness",
        "terminal_and_each_horizon_mask_coverage",
        "zero_prize_setup_mask_count",
        "non_monotone_mask_count",
    ]
    if (
        target_overlay.get("schema") != TARGET_OVERLAY_SCHEMA
        or target_overlay.get("manifest_schema") != TARGET_SET_MANIFEST_SCHEMA
        or target_overlay.get("row_join_identity") != expected_join_identity
        or target_overlay.get("group_key")
        != ["source_archive_sha256", "episode_id", "acting_seat"]
        or target_overlay.get("group_order")
        != "strictly_increasing_env_step_no_duplicates"
        or target_overlay.get("public_state_endpoint")
        != "steps[env_step][acting_seat].observation.current"
        or target_overlay.get("raw_action_alignment")
        != "selected_complete_action_is_carried_by_steps[env_step+1][acting_seat].action_and_must_already_equal_the_sealed_complete_action_overlay"
        or target_overlay.get("required_terminal_fields")
        != ["z", "z_mask", "win_target_one_only_for_z_plus1", "win_target_mask"]
        or target_overlay.get("required_per_horizon_fields") != expected_horizon_fields
        or target_overlay.get("target_set_manifest_must_bind")
        != expected_manifest_bindings
        or target_overlay.get("hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed")
        is not False
    ):
        raise ActionCriticTargetError("revision-21 target-overlay semantics drifted")
    prize_count = target_overlay.get("prize_count")
    horizons = target_overlay.get("horizon_definition")
    if (
        not isinstance(prize_count, Mapping)
        or prize_count.get("source")
        != "length_only_of_public_current.players[seat].prize_or_exact_public_count_alias"
        or prize_count.get("valid_inclusive_range") != [1, 6]
        or prize_count.get("zero_behavior")
        != "mask_as_setup_or_uninitialized_never_treat_as_real_zero_progress"
        or prize_count.get("card_identities_copied_or_consumed") is not False
        or not isinstance(horizons, Mapping)
        or horizons.get("values") != list(HORIZONS)
        or horizons.get("start")
        != "pre_action_public_prize_counts_at_complete_action_i"
        or horizons.get("end")
        != "pre_action_public_prize_counts_at_complete_action_i_plus_h_for_same_group"
        or horizons.get("own_taken") != "own_remaining_before-own_remaining_after"
        or horizons.get("opponent_taken")
        != "opponent_remaining_before-opponent_remaining_after"
        or horizons.get("target") != "clip((own_taken-opponent_taken)/3,-1,+1)"
        or horizons.get("terminal_ending_interval")
        != "mask_when_no_later_complete_same_seat_action_exists_do_not_infer_a_terminal_after_state"
        or horizons.get("invalid_or_non_monotone_behavior")
        != "mask_that_horizon_never_assign_zero"
        or actor_advantage.get("enabled_formula")
        != "(z-V_existing(s))+0.05*m1*(Q_prize^1(s,a)-V_prize^1(s))"
        or actor_advantage.get("complete_action_value_broadcast_identically_across_selected_factorized_stages")
        is not True
        or actor_advantage.get("actor_gradient_into_sidecar_allowed") is not False
    ):
        raise ActionCriticTargetError("revision-21 critic semantics drifted")


def _assert_no_forbidden_keys(value: Any) -> None:
    """Reject an input overlay carrying a hidden-state field anywhere."""

    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        overlap = keys.intersection(_FORBIDDEN_INPUT_KEYS)
        if overlap:
            raise ActionCriticTargetError(
                "complete-action overlay contains forbidden hidden-state keys: "
                f"{sorted(overlap)}"
            )
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def target_schema_document() -> dict[str, Any]:
    """Describe the revision-21 immutable target-only row ABI.

    The schema is intentionally independent of the complete-action feature
    representation.  A downstream reader joins by ``program_identity``.
    """

    return {
        "schema": TARGET_SCHEMA,
        "version": 1,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "row_unit": "recorded_complete_chosen_action_program",
        "join_identity": [
            "utc_day",
            "source_archive_sha256",
            "source_member",
            "episode_id",
            "acting_seat",
            "env_step",
            "program_identity",
        ],
        "terminal_fields": {
            "z": "observed terminal return in {-1,0,+1}",
            "z_mask": "false unless the raw episode has an exact DONE/DONE zero-sum terminal envelope",
            "win_target_one_only_for_z_plus1": "1 only for z=+1; 0 for z in {-1,0}",
            "win_target_mask": "identical to z_mask",
        },
        "prize_differential_targets": {
            "horizons": list(HORIZONS),
            "target": "clip((own_prizes_taken-opponent_prizes_taken)/3,-1,+1)",
            "interval": "pre-action frame i through pre-action frame i+h for the same acting seat",
            "valid_remaining_prize_range": [1, 6],
            "mask": "false for setup-zero, missing, incomplete, non-causal, ambiguous, or non-monotone intervals; never write zero as a substitute",
            "required_fields": [
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
            ],
        },
        "raw_read_scope": [
            "payload.id",
            "payload.rewards[acting_seat]",
            "steps[env_step][acting_seat].observation.current.yourIndex",
            "steps[env_step][acting_seat].observation.current.players[*].public_prize_count",
        ],
        "forbidden_output_fields": sorted(_FORBIDDEN_INPUT_KEYS),
        "copied_feature_tensors": False,
        "raw_action_alignment_verified": True,
        "copied_actions": False,
        "search_or_planner_calls": False,
        "counterfactual_or_unchosen_targets": False,
    }


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
    recorded_outcome: float | None


@dataclass(frozen=True)
class _PrizeCounts:
    own_remaining: int | None
    opponent_remaining: int | None
    unavailable_reason: str | None


def _require_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ActionCriticTargetError(f"complete-action row has invalid {field}")
    return value


def _require_source_member(row: Mapping[str, Any], field: str = "source_member") -> str:
    """Require a stable, safe ZIP-member identity without opening any payload."""

    member = _require_string(row, field)
    candidate = Path(member)
    if (
        candidate.is_absolute()
        or member in {".", ".."}
        or ".." in candidate.parts
        or "\\" in member
        or member.endswith(("/", "\\"))
    ):
        raise ActionCriticTargetError("source_member is not a safe raw ZIP member identity")
    return member


def _optional_recorded_outcome(value: Any) -> float | None:
    if value is None:
        return None
    result = _finite_number(value)
    if result not in {-1.0, 0.0, 1.0}:
        raise ActionCriticTargetError("complete-action row has invalid recorded_outcome")
    return result


def _parse_program(row: Mapping[str, Any], *, expected_day: str) -> _Program:
    _assert_no_forbidden_keys(row)
    if row.get("schema") != COMPLETE_ACTION_OVERLAY_SCHEMA:
        raise ActionCriticTargetError("foreign complete-action overlay row")
    if row.get("hidden_information_fields_present") is not False:
        raise ActionCriticTargetError("complete-action row is not marked public-only")
    if row.get("complete_action_program_reconstructed") is not True:
        raise ActionCriticTargetError("complete-action row is not a reconstructed program")
    if not isinstance(row.get("stages"), list) or not row["stages"]:
        raise ActionCriticTargetError("complete-action row has no factorized stages")
    utc_day = _require_string(row, "utc_day")
    if utc_day != expected_day:
        raise ActionCriticTargetError(
            f"complete-action row day mismatch: expected {expected_day}, got {utc_day}"
        )
    source_archive_sha256 = _require_string(row, "source_archive_sha256")
    _sha_hex(source_archive_sha256)
    seat = _exact_int(row.get("acting_seat"))
    env_step = _exact_int(row.get("env_step"))
    if seat not in (0, 1) or env_step is None or env_step < 0:
        raise ActionCriticTargetError("complete-action row has invalid seat or env_step")
    successor = row.get("recorded_successor_program_identity")
    if successor is not None and (not isinstance(successor, str) or not successor):
        raise ActionCriticTargetError("complete-action row has malformed successor identity")
    raw_action = row.get("selected_action_program")
    if not isinstance(raw_action, list):
        raise ActionCriticTargetError("complete-action row has invalid selected_action_program")
    selected_action: list[int] = []
    for value in raw_action:
        action_value = _exact_int(value)
        if action_value is None or action_value < 0:
            raise ActionCriticTargetError("complete-action row selected action is malformed")
        selected_action.append(action_value)
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
        recorded_outcome=_optional_recorded_outcome(row.get("recorded_outcome")),
    )


def _iter_program_groups(
    overlay_path: Path, *, expected_day: str
) -> Iterator[list[_Program]]:
    """Stream contiguous archive/episode/seat groups from a sealed overlay.

    The complete-action producer emits a given episode and acting seat as one
    contiguous run.  Refusing a non-contiguous row inside a run keeps the
    target builder bounded to one recorded game-seat at a time.
    """

    current_key: tuple[str, str, int] | None = None
    current: list[_Program] = []
    with overlay_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ActionCriticTargetError(
                    f"blank complete-action overlay row at {overlay_path}:{line_number}"
                )
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ActionCriticTargetError(
                    f"invalid complete-action overlay JSON at {overlay_path}:{line_number}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise ActionCriticTargetError(
                    f"complete-action row is not an object at {overlay_path}:{line_number}"
                )
            program = _parse_program(raw, expected_day=expected_day)
            key = (
                program.source_archive_sha256,
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


def _public_prize_count(player: Any) -> int | None:
    """Read a public count without examining a Prize-zone card object."""

    if not isinstance(player, Mapping):
        return None
    observed: list[int] = []
    for name in ("prize_count", "prizeCount", "remainingPrizes"):
        if name not in player:
            continue
        count = _exact_int(player.get(name))
        if count is None or count < 0:
            return None
        observed.append(count)
    prize_zone = player.get("prize")
    if prize_zone is not None:
        if not isinstance(prize_zone, (list, tuple)):
            return None
        # ``len`` is deliberately the only operation applied to this zone.
        observed.append(len(prize_zone))
    if not observed or any(count != observed[0] for count in observed[1:]):
        return None
    return observed[0]


def _pre_action_public_prize_counts(
    payload: Mapping[str, Any], program: _Program
) -> _PrizeCounts:
    steps = payload.get("steps")
    if not isinstance(steps, list) or not 0 <= program.env_step < len(steps):
        return _PrizeCounts(None, None, "pre_action_frame_absent")
    pair = steps[program.env_step]
    if not isinstance(pair, list) or len(pair) != 2:
        return _PrizeCounts(None, None, "pre_action_agent_pair_malformed")
    agent = pair[program.acting_seat]
    if not isinstance(agent, Mapping):
        return _PrizeCounts(None, None, "pre_action_acting_agent_malformed")
    observation = agent.get("observation")
    current = observation.get("current") if isinstance(observation, Mapping) else None
    if not isinstance(current, Mapping):
        return _PrizeCounts(None, None, "pre_action_public_current_absent")
    your_index = _exact_int(current.get("yourIndex"))
    players = current.get("players")
    if your_index not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        return _PrizeCounts(None, None, "pre_action_public_player_mapping_absent")
    # The observed perspective must identify the same real acting seat.  A
    # mismatch is not repaired from a private/global state; its targets mask.
    if your_index != program.acting_seat:
        return _PrizeCounts(None, None, "pre_action_public_actor_index_mismatch")
    own = _public_prize_count(players[your_index])
    opponent = _public_prize_count(players[1 - your_index])
    if own is None or opponent is None:
        return _PrizeCounts(None, None, "pre_action_public_prize_count_absent_or_ambiguous")
    # Preserve a zero in the target-only endpoint fields for audit, but never
    # treat it as an ordinary no-progress count: it is setup/uninitialized or
    # terminal and masks every interval touching it.
    if not 1 <= own <= 6 or not 1 <= opponent <= 6:
        return _PrizeCounts(
            own,
            opponent,
            "pre_action_public_prize_count_outside_valid_1_to_6_range",
        )
    return _PrizeCounts(own, opponent, None)


def _terminal_win_target(
    payload: Mapping[str, Any], *, seat: int
) -> tuple[float | None, bool, str | None, float | None]:
    """Return an observed terminal BCE target only from a DONE envelope.

    Numeric rewards on an interrupted, timed-out, or malformed episode are
    not terminal-outcome provenance.  The compact RTP overlay may retain a
    historical numeric outcome for diagnostics, but this sidecar masks it
    unless the raw envelope proves the terminal two-seat result.
    """

    statuses = payload.get("statuses")
    if not isinstance(statuses, list) or statuses != ["DONE", "DONE"]:
        return None, False, "terminal_statuses_not_exact_done_pair", None
    rewards = payload.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 2:
        return None, False, "terminal_rewards_absent_or_malformed", None
    left = _finite_number(rewards[0])
    right = _finite_number(rewards[1])
    if left not in {-1.0, 0.0, 1.0} or right not in {-1.0, 0.0, 1.0}:
        return None, False, "terminal_reward_absent_or_invalid", None
    if left != -right:
        return None, False, "terminal_rewards_not_zero_sum", None
    reward = (left, right)[seat]
    return (1.0 if reward == 1.0 else 0.0), True, None, reward


def _target(value: float | None, *, reason: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"value": None, "mask": False, "unavailable_reason": reason}
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ActionCriticTargetError("target is non-finite or outside [-1,+1]")
    return {"value": float(value), "mask": True, "unavailable_reason": None}


def _validate_group(programs: Sequence[_Program]) -> list[_Program]:
    if not programs:
        raise ActionCriticTargetError("empty complete-action program group")
    key = (
        programs[0].source_archive_sha256,
        programs[0].episode_id,
        programs[0].acting_seat,
    )
    source_member = programs[0].source_member
    identities: set[str] = set()
    steps: set[int] = set()
    previous_step: int | None = None
    for program in programs:
        candidate = (
            program.source_archive_sha256,
            program.episode_id,
            program.acting_seat,
        )
        if candidate != key or program.source_member != source_member:
            raise ActionCriticTargetError("complete-action program group identity drifted")
        if program.program_identity in identities or program.env_step in steps:
            raise ActionCriticTargetError("duplicate complete-action program identity or env_step")
        if previous_step is not None and program.env_step <= previous_step:
            raise ActionCriticTargetError(
                "same-seat complete-action env_steps are not strictly increasing"
            )
        identities.add(program.program_identity)
        steps.add(program.env_step)
        previous_step = program.env_step
    return list(programs)


def _interval_reason(
    programs: Sequence[_Program],
    counts: Sequence[_PrizeCounts],
    index: int,
    horizon: int,
) -> str | None:
    endpoint = index + horizon
    if endpoint >= len(programs):
        return "no_later_same_seat_recorded_program"
    for offset in range(index, endpoint):
        current = programs[offset]
        following = programs[offset + 1]
        if following.env_step <= current.env_step:
            return "same_seat_program_env_steps_not_strictly_increasing"
        if current.successor_program_identity != following.program_identity:
            return "same_seat_successor_program_link_incomplete_or_ambiguous"
    for offset in range(index, endpoint + 1):
        if counts[offset].unavailable_reason is not None:
            return counts[offset].unavailable_reason
    for offset in range(index, endpoint):
        current = counts[offset]
        following = counts[offset + 1]
        assert current.own_remaining is not None
        assert current.opponent_remaining is not None
        assert following.own_remaining is not None
        assert following.opponent_remaining is not None
        if (
            following.own_remaining > current.own_remaining
            or following.opponent_remaining > current.opponent_remaining
        ):
            return "non_monotone_public_prize_count"
    return None


def _prize_target_for_interval(
    programs: Sequence[_Program],
    counts: Sequence[_PrizeCounts],
    *,
    index: int,
    horizon: int,
) -> dict[str, Any]:
    endpoint = index + horizon
    start = counts[index]
    end = counts[endpoint] if endpoint < len(counts) else None
    reason = _interval_reason(programs, counts, index, horizon)
    future = programs[endpoint] if endpoint < len(programs) else None
    result: dict[str, Any] = {
        "h": horizon,
        "mask": False,
        "unavailable_reason": reason,
        "future_program_identity": None if future is None else future.program_identity,
        "future_env_step": None if future is None else future.env_step,
        "own_remaining_before": start.own_remaining,
        "own_remaining_after": None if end is None else end.own_remaining,
        "opponent_remaining_before": start.opponent_remaining,
        "opponent_remaining_after": None if end is None else end.opponent_remaining,
        "own_taken": None,
        "opponent_taken": None,
        "differential": None,
    }
    if reason is not None:
        return result
    assert end is not None
    assert start.own_remaining is not None
    assert start.opponent_remaining is not None
    assert end.own_remaining is not None
    assert end.opponent_remaining is not None
    own_taken = start.own_remaining - end.own_remaining
    opponent_taken = start.opponent_remaining - end.opponent_remaining
    differential = max(-1.0, min(1.0, (own_taken - opponent_taken) / 3.0))
    result.update(
        {
            "mask": True,
            "unavailable_reason": None,
            "own_taken": own_taken,
            "opponent_taken": opponent_taken,
            "differential": differential,
        }
    )
    return result


def _assert_raw_action_alignment(
    payload: Mapping[str, Any], programs: Sequence[_Program]
) -> None:
    """Verify the sealed selected program against the raw action at N+1."""

    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ActionCriticTargetError("raw episode steps are absent for action alignment")
    for program in programs:
        action_step = program.env_step + 1
        if not 0 <= action_step < len(steps):
            raise ActionCriticTargetError("raw selected action frame is absent")
        pair = steps[action_step]
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(
            pair[program.acting_seat], Mapping
        ):
            raise ActionCriticTargetError("raw selected action agent frame is malformed")
        raw_action = pair[program.acting_seat].get("action")
        if not isinstance(raw_action, list):
            raise ActionCriticTargetError("raw selected complete action is absent")
        parsed: list[int] = []
        for value in raw_action:
            item = _exact_int(value)
            if item is None or item < 0:
                raise ActionCriticTargetError("raw selected complete action is malformed")
            parsed.append(item)
        if tuple(parsed) != program.selected_action_program:
            raise ActionCriticTargetError(
                "raw selected complete action disagrees with sealed complete-action overlay"
            )


def _open_exact_episode(
    archive: zipfile.ZipFile,
    member_index: Mapping[str, zipfile.ZipInfo],
    program: _Program,
) -> Mapping[str, Any]:
    info = member_index.get(program.source_member)
    if info is None:
        raise ActionCriticTargetError(
            f"raw archive member is absent: {program.source_member}"
        )
    try:
        # Opening the exact ZipInfo avoids name-based first/last-entry behavior.
        with archive.open(info) as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionCriticTargetError(
            f"raw archive member is unreadable: {program.source_member}"
        ) from exc
    if not isinstance(payload, Mapping) or str(payload.get("id") or "") != program.episode_id:
        raise ActionCriticTargetError("raw archive member episode identity mismatch")
    return payload


def _unique_zip_member_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Index physical members only when every filename occurs exactly once."""

    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.filename in result:
            raise ActionCriticTargetError(
                f"raw episode ZIP has duplicate physical member name: {info.filename}"
            )
        result[info.filename] = info
    return result


def _check_overlay_outcome(
    programs: Iterable[_Program], raw_outcome: float | None
) -> None:
    for program in programs:
        if (
            program.recorded_outcome is not None
            and raw_outcome is not None
            and program.recorded_outcome != raw_outcome
        ):
            raise ActionCriticTargetError(
                "sealed complete-action recorded outcome disagrees with exact raw episode"
            )


def _write_file_create_only(path: Path, body: bytes, *, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(fd, body[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return _SHA256_PREFIX + hashlib.sha256(body).hexdigest()


def _publish_temp_object(temp: Path, object_dir: Path, *, suffix: str) -> tuple[Path, str, int]:
    digest = sha256_file(temp)
    final = object_dir / f"sha256-{_sha_hex(digest)}{suffix}"
    object_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temp, final)
    except FileExistsError as exc:
        raise ActionCriticTargetError(f"refusing duplicate object publication: {final}") from exc
    os.chmod(final, 0o444)
    size = final.stat().st_size
    temp.unlink()
    return final, digest, size


def _safe_output_root(path: Path | str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.exists() or result.is_symlink():
        raise ActionCriticTargetError(f"output root already exists: {result}")
    if not result.parent.is_dir() or result.parent.is_symlink():
        raise ActionCriticTargetError(f"output parent is unavailable: {result.parent}")
    return result


def _fsync_directory(path: Path | str) -> None:
    directory = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError as exc:
        raise ActionCriticTargetError(
            f"cannot open artifact directory for durability sync: {directory}"
        ) from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise ActionCriticTargetError(
            f"artifact directory durability sync failed: {directory}"
        ) from exc
    finally:
        os.close(fd)


def _fsync_artifact_directories(root: Path) -> None:
    """Durably order all created directory entries before root publication."""

    for current, directories, _files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in directories:
            member = current_path / name
            if member.is_symlink():
                raise ActionCriticTargetError(
                    f"artifact publication tree contains a symlink: {member}"
                )
        _fsync_directory(current_path)


def _atomic_publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish one sibling directory atomically without replacement.

    Linux must provide ``renameat2(..., RENAME_NOREPLACE)``.  Darwin uses its
    atomic ``renamex_np(..., RENAME_EXCL)`` equivalent so focused tests retain
    the same race-free property.  Other platforms, missing libc entry points,
    and unsupported filesystems fail closed.
    """

    source_input = Path(source).expanduser()
    destination_input = Path(destination).expanduser()
    if source_input.is_symlink() or not source_input.is_dir():
        raise ActionCriticTargetError("atomic publication source is not a regular directory")
    if destination_input.is_symlink():
        raise ActionCriticTargetError("output root already exists")
    source_path = source_input.resolve()
    destination_parent = destination_input.parent.resolve()
    destination_path = destination_parent / destination_input.name
    if source_path.parent != destination_parent or not destination_input.name:
        raise ActionCriticTargetError(
            "atomic publication requires source and destination sibling directories"
        )

    _fsync_artifact_directories(source_path)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ActionCriticTargetError(
                "atomic no-replace publication is unavailable: libc.renameat2 absent"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD on Linux
            os.fsencode(source_path),
            -100,
            os.fsencode(destination_path),
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise ActionCriticTargetError(
                "atomic no-replace publication is unavailable: libc.renamex_np absent"
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source_path),
            os.fsencode(destination_path),
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise ActionCriticTargetError(
            f"atomic no-replace directory publication is unsupported on {sys.platform}"
        )
    if result != 0:
        failure = ctypes.get_errno()
        if failure in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ActionCriticTargetError(f"output root already exists: {destination_path}")
        unavailable = {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}
        if hasattr(errno, "EOPNOTSUPP"):
            unavailable.add(errno.EOPNOTSUPP)
        if failure in unavailable:
            raise ActionCriticTargetError(
                "atomic no-replace directory publication is unavailable on this filesystem"
            )
        raise ActionCriticTargetError(
            f"atomic no-replace directory publication failed: errno {failure}"
        )
    _fsync_directory(destination_parent)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise ActionCriticTargetError("output member escaped target overlay root") from exc


def _counter_summary(counters: Mapping[str, int]) -> dict[str, int]:
    required = {
        "complete_action_programs",
        "terminal_win_labeled",
        "terminal_win_masked",
        "prize_h1_labeled",
        "prize_h1_masked",
        "prize_h2_labeled",
        "prize_h2_masked",
        "prize_h3_labeled",
        "prize_h3_masked",
        "zero_prize_setup_mask_count",
        "out_of_range_prize_mask_count",
        "non_monotone_mask_count",
    }
    values = {str(key): int(value) for key, value in counters.items()}
    for key in required:
        values.setdefault(key, 0)
    return {key: values[key] for key in sorted(values)}


def build_action_critic_target_overlay_day(
    *,
    complete_action_overlay_path: Path | str,
    raw_episode_zip_path: Path | str,
    output_root: Path | str,
    utc_day: str,
    split: str,
    goal_contract_path: Path | str,
    expected_goal_contract_sha256: str,
    expected_complete_action_overlay_sha256: str = "",
    expected_raw_episode_zip_sha256: str = "",
) -> dict[str, Any]:
    """Build one atomic, content-addressed target overlay day.

    ``output_root`` must not exist.  The function first builds a private
    sibling directory and publishes it with one directory rename only after
    the object, manifest, and receipt are all durable.  No existing artifact
    is overwritten.
    """

    if utc_day not in WINDOW_DAYS:
        raise ActionCriticTargetError("utc_day is outside the exact recent-20 window")
    if split != SPLIT_BY_DAY[utc_day]:
        raise ActionCriticTargetError("target day split does not match the sealed recent-20 split")
    overlay_path = _read_only_regular_file(
        complete_action_overlay_path, label="complete-action overlay"
    )
    raw_zip_path = _read_only_regular_file(raw_episode_zip_path, label="raw episode ZIP")
    goal_contract_path_resolved, goal_contract, goal_contract_sha = _load_goal_contract(
        goal_contract_path, expected_sha256=expected_goal_contract_sha256
    )
    contract_goal_revision = _exact_int(goal_contract.get("goal_revision"))
    if contract_goal_revision is None:  # _load_goal_contract already establishes this.
        raise ActionCriticTargetError("goal contract revision is invalid")
    final_root = _safe_output_root(output_root)
    overlay_sha = sha256_file(overlay_path)
    raw_zip_sha = sha256_file(raw_zip_path)
    if expected_complete_action_overlay_sha256:
        _sha_hex(expected_complete_action_overlay_sha256)
        if overlay_sha != expected_complete_action_overlay_sha256:
            raise ActionCriticTargetError("complete-action overlay SHA-256 mismatch")
    if expected_raw_episode_zip_sha256:
        _sha_hex(expected_raw_episode_zip_sha256)
        if raw_zip_sha != expected_raw_episode_zip_sha256:
            raise ActionCriticTargetError("raw episode ZIP SHA-256 mismatch")

    started = time.time()
    private_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.private-", dir=final_root.parent)
    )
    objects = private_root / "objects"
    manifests = private_root / "manifests"
    receipts = private_root / "receipts"
    schemas = private_root / "schemas"
    for directory in (objects, manifests, receipts, schemas):
        directory.mkdir()
    schema = target_schema_document()
    schema_sha = canonical_sha256(schema)
    schema_path = schemas / f"sha256-{_sha_hex(schema_sha)}.target-schema.json"
    _write_file_create_only(schema_path, canonical_bytes(schema))

    temp_object = private_root / ".targets.jsonl.partial"
    counters: Counter[str] = Counter()
    unavailable_reasons: Counter[str] = Counter()
    source_archive_sha256: str | None = None
    group_count = 0
    with (
        zipfile.ZipFile(raw_zip_path) as archive,
        temp_object.open("xb", buffering=8 * 1024 * 1024) as output,
    ):
        raw_member_index = _unique_zip_member_index(archive)
        for raw_group in _iter_program_groups(overlay_path, expected_day=utc_day):
            programs = _validate_group(raw_group)
            group_count += 1
            group_archive_sha = programs[0].source_archive_sha256
            if source_archive_sha256 is None:
                source_archive_sha256 = group_archive_sha
            elif source_archive_sha256 != group_archive_sha:
                raise ActionCriticTargetError("one overlay day references multiple raw archive hashes")
            if group_archive_sha != raw_zip_sha:
                raise ActionCriticTargetError(
                    "complete-action row source_archive_sha256 does not match raw episode ZIP"
                )
            payload = _open_exact_episode(archive, raw_member_index, programs[0])
            terminal_value, terminal_mask, terminal_reason, raw_outcome = _terminal_win_target(
                payload, seat=programs[0].acting_seat
            )
            _check_overlay_outcome(programs, raw_outcome)
            _assert_raw_action_alignment(payload, programs)
            counts = [_pre_action_public_prize_counts(payload, program) for program in programs]
            for index, program in enumerate(programs):
                terminal = _target(
                    terminal_value if terminal_mask else None, reason=terminal_reason
                )
                targets = {
                    f"h{horizon}": _prize_target_for_interval(
                        programs, counts, index=index, horizon=horizon
                    )
                    for horizon in HORIZONS
                }
                row = {
                    "schema": TARGET_OVERLAY_SCHEMA,
                    "owner_goal_revision": OWNER_GOAL_REVISION,
                    "goal_contract_goal_revision": contract_goal_revision,
                    "utc_day": program.utc_day,
                    "split": split,
                    "source_archive_sha256": program.source_archive_sha256,
                    "source_member": program.source_member,
                    "episode_id": program.episode_id,
                    "acting_seat": program.acting_seat,
                    "env_step": program.env_step,
                    "program_identity": program.program_identity,
                    "z": raw_outcome if terminal_mask else None,
                    "z_mask": terminal_mask,
                    "win_target_one_only_for_z_plus1": (
                        terminal_value if terminal_mask else None
                    ),
                    "win_target_mask": terminal_mask,
                    "terminal_win": terminal,
                    "prize_differential": targets,
                    "target_only": True,
                    "hidden_information_fields_present": False,
                    "raw_target_provenance": {
                        "terminal_reward_source": "raw_episode.rewards[acting_seat]",
                        "prize_count_source": "pre_action_public_observation.current.players",
                        "prize_zone_card_identities_read": False,
                        "post_action_or_future_frame_used_as_input": False,
                    },
                }
                _assert_no_forbidden_keys(row)
                encoded = canonical_bytes(row)
                output.write(encoded)
                counters["complete_action_programs"] += 1
                counters["terminal_win_labeled" if terminal["mask"] else "terminal_win_masked"] += 1
                if not terminal["mask"]:
                    unavailable_reasons[str(terminal["unavailable_reason"])] += 1
                for horizon in HORIZONS:
                    target = targets[f"h{horizon}"]
                    name = f"prize_h{horizon}"
                    counters[f"{name}_labeled" if target["mask"] else f"{name}_masked"] += 1
                    if not target["mask"]:
                        unavailable_reasons[str(target["unavailable_reason"])] += 1
                        if target["unavailable_reason"] == (
                            "pre_action_public_prize_count_outside_valid_1_to_6_range"
                        ):
                            endpoint_values = (
                                target["own_remaining_before"],
                                target["own_remaining_after"],
                                target["opponent_remaining_before"],
                                target["opponent_remaining_after"],
                            )
                            counter = (
                                "zero_prize_setup_mask_count"
                                if 0 in endpoint_values
                                else "out_of_range_prize_mask_count"
                            )
                            counters[counter] += 1
                        if target["unavailable_reason"] == "non_monotone_public_prize_count":
                            counters["non_monotone_mask_count"] += 1
        output.flush()
        os.fsync(output.fileno())
    if source_archive_sha256 is None:
        raise ActionCriticTargetError("complete-action overlay is empty")
    object_path, object_sha, object_size = _publish_temp_object(
        temp_object, objects, suffix=".action-critic-targets.jsonl"
    )
    manifest = {
        "schema": TARGET_DAY_MANIFEST_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract": {
            "path": str(goal_contract_path_resolved),
            "sha256": goal_contract_sha,
            "goal_revision": contract_goal_revision,
            "critic_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
            "required_authority": CRITIC_AUTHORITY_KEY,
        },
        "target_schema_path": _relative(schema_path, private_root),
        "target_schema_sha256": schema_sha,
        "utc_day": utc_day,
        "split": split,
        "complete_action_overlay": {
            "path": str(overlay_path),
            "sha256": overlay_sha,
            "size_bytes": overlay_path.stat().st_size,
            "schema": COMPLETE_ACTION_OVERLAY_SCHEMA,
        },
        "raw_episode_zip": {
            "path": str(raw_zip_path),
            "sha256": raw_zip_sha,
            "size_bytes": raw_zip_path.stat().st_size,
            "source_archive_sha256_verified": source_archive_sha256 == raw_zip_sha,
        },
        "target_shard": {
            "path": _relative(object_path, private_root),
            "sha256": object_sha,
            "size_bytes": object_size,
            "row_count": int(counters["complete_action_programs"]),
        },
        "coverage": {
            "program_group_count": group_count,
            "counts": _counter_summary(counters),
            "masked_unavailable_reasons": _counter_summary(unavailable_reasons),
        },
        "information_boundary": {
            "raw_read_scope": schema["raw_read_scope"],
            "hidden_information_fields_present": False,
            "prize_zone_card_identities_read": False,
            "feature_tensors_copied": False,
            "actions_copied": False,
            "search_or_planner_called": False,
            "raw_selected_action_alignment_verified": True,
            "counterfactual_or_unchosen_targets_present": False,
        },
        "publication": {
            "create_only": True,
            "atomic_root_rename": True,
            "atomic_root_no_replace": True,
            "artifact_directories_fsynced_before_publication": True,
            "publication_parent_fsynced_after_publication": True,
            "input_paths_read_only": True,
        },
    }
    manifest_body = canonical_bytes(manifest)
    manifest_sha = _SHA256_PREFIX + hashlib.sha256(manifest_body).hexdigest()
    manifest_path = manifests / f"sha256-{_sha_hex(manifest_sha)}.target-manifest.json"
    _write_file_create_only(manifest_path, manifest_body)
    receipt = {
        "schema": TARGET_DAY_RECEIPT_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_goal_revision": contract_goal_revision,
        "critic_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_sha256": goal_contract_sha,
        "manifest_path": _relative(manifest_path, private_root),
        "manifest_sha256": manifest_sha,
        "target_schema_sha256": schema_sha,
        "complete_action_overlay_sha256": overlay_sha,
        "raw_episode_zip_sha256": raw_zip_sha,
        "target_shard_sha256": object_sha,
        "target_shard_size_bytes": object_size,
        "target_row_count": int(counters["complete_action_programs"]),
        "coverage": manifest["coverage"],
        "input_identity_verified": True,
        "raw_reads_limited_to_terminal_reward_and_pre_action_public_prize_counts": True,
        "raw_selected_complete_action_alignment_verified": True,
        "hidden_information_output_fields_present": False,
        "search_simulator_or_planner_called": False,
        "recollection_or_training_performed": False,
        "atomic_root_no_replace": True,
        "artifact_directories_fsynced_before_publication": True,
        "publication_parent_fsynced_after_publication": True,
        "sealed_at_unix_seconds": time.time(),
        "elapsed_seconds": max(0.0, time.time() - started),
    }
    receipt_body = canonical_bytes(receipt)
    receipt_sha = _SHA256_PREFIX + hashlib.sha256(receipt_body).hexdigest()
    receipt_path = receipts / f"sha256-{_sha_hex(receipt_sha)}.target-receipt.json"
    _write_file_create_only(receipt_path, receipt_body)
    # A completed root is not visible until all immutable members exist.
    _atomic_publish_directory_noreplace(private_root, final_root)
    return {
        "output_root": str(final_root),
        "target_shard_path": str(final_root / _relative(object_path, private_root)),
        "target_shard_sha256": object_sha,
        "manifest_path": str(final_root / _relative(manifest_path, private_root)),
        "manifest_sha256": manifest_sha,
        "receipt_path": str(final_root / _relative(receipt_path, private_root)),
        "receipt_sha256": receipt_sha,
        "row_count": int(counters["complete_action_programs"]),
        "coverage": manifest["coverage"],
    }


def _read_json_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = _read_only_regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionCriticTargetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ActionCriticTargetError(f"{label} is not a JSON object")
    return resolved, value, sha256_file(resolved)


def _read_only_directory(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise ActionCriticTargetError(f"{label} is not a directory: {resolved}")
    return resolved


def _artifact_member(root: Path, declared: Any, *, label: str) -> Path:
    if not isinstance(declared, str) or not declared:
        raise ActionCriticTargetError(f"{label} path is absent")
    candidate = Path(declared)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = _read_only_regular_file(candidate, label=label)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ActionCriticTargetError(f"{label} escaped its sealed artifact root") from exc
    return resolved


def _portable_member(
    root: Path, declared: Any, *, label: str, directory: bool = False
) -> Path:
    """Resolve an aggregate pointer that must stay inside a relocated set root."""

    if not isinstance(declared, str) or not declared:
        raise ActionCriticTargetError(f"{label} portable path is absent")
    candidate = Path(declared)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ActionCriticTargetError(f"{label} must be a relative target-set path")
    root = _read_only_directory(root, label="target-set root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ActionCriticTargetError(f"{label} escaped the target-set root") from exc
    if resolved.is_symlink() or (not resolved.is_dir() if directory else not resolved.is_file()):
        kind = "directory" if directory else "regular file"
        raise ActionCriticTargetError(f"{label} is not a sealed {kind}")
    return resolved


def _copy_regular_file_create_only(source: Path, destination: Path) -> None:
    """Copy one immutable regular file without links or overwrite semantics."""

    source = _read_only_regular_file(source, label="portable artifact source")
    if destination.exists() or destination.is_symlink():
        raise ActionCriticTargetError(f"refusing duplicate portable artifact member: {destination}")
    source_size = source.stat().st_size
    source_sha = sha256_file(source)
    input_fd = os.open(source, os.O_RDONLY)
    output_fd: int | None = None
    try:
        output_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        while True:
            block = os.read(input_fd, 8 * 1024 * 1024)
            if not block:
                break
            offset = 0
            while offset < len(block):
                offset += os.write(output_fd, block[offset:])
        os.fsync(output_fd)
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(input_fd)
    os.chmod(destination, 0o444)
    if destination.stat().st_size != source_size or sha256_file(destination) != source_sha:
        raise ActionCriticTargetError("portable artifact copy failed its SHA-256 verification")


def _copy_tree_create_only(source_root: Path, destination_root: Path) -> None:
    """Copy a sealed day root as ordinary files and directories only."""

    source_root = _read_only_directory(source_root, label="portable day artifact source")
    if destination_root.exists() or destination_root.is_symlink():
        raise ActionCriticTargetError(
            f"refusing duplicate portable day artifact root: {destination_root}"
        )
    destination_root.mkdir(mode=0o755, parents=True)

    def copy_directory(source_directory: Path, destination_directory: Path) -> None:
        with os.scandir(source_directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                source = source_directory / entry.name
                destination = destination_directory / entry.name
                if entry.is_symlink():
                    raise ActionCriticTargetError(
                        f"sealed day artifact contains a symlink: {source}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    destination.mkdir(mode=0o755)
                    copy_directory(source, destination)
                elif entry.is_file(follow_symlinks=False):
                    _copy_regular_file_create_only(source, destination)
                else:
                    raise ActionCriticTargetError(
                        f"sealed day artifact contains a non-regular member: {source}"
                    )

    copy_directory(source_root, destination_root)


def _portable_binding(
    private_root: Path, source: Path, *, sha256: str, label: str, suffix: str
) -> dict[str, Any]:
    """Bundle a small active binding document under a deterministic relative path."""

    source = _read_only_regular_file(source, label=label)
    if sha256_file(source) != sha256:
        raise ActionCriticTargetError(f"{label} SHA-256 changed before portable bundling")
    relative = Path("bindings") / f"sha256-{_sha_hex(sha256)}.{suffix}.json"
    destination = private_root / relative
    _copy_regular_file_create_only(source, destination)
    return {
        "path": str(relative),
        "sha256": sha256,
        "size_bytes": destination.stat().st_size,
    }


def resolve_action_critic_target_set_day(
    target_set_root: Path | str, target_day: Mapping[str, Any]
) -> dict[str, Path]:
    """Resolve and verify one aggregate day after arbitrary whole-tree relocation.

    Only aggregate-relative pointers are interpreted.  Historical provenance
    paths inside the immutable copied day manifest are intentionally never
    dereferenced here.
    """

    root = _read_only_directory(target_set_root, label="target-set root")
    if not isinstance(target_day, Mapping):
        raise ActionCriticTargetError("aggregate target-day entry is not an object")
    day_root = _portable_member(
        root, target_day.get("day_artifact_root"), label="day artifact root", directory=True
    )
    manifest_path = _portable_member(
        root, target_day.get("day_manifest_path"), label="day manifest"
    )
    receipt_path = _portable_member(
        root, target_day.get("day_receipt_path"), label="day receipt"
    )
    for member, label in ((manifest_path, "day manifest"), (receipt_path, "day receipt")):
        try:
            member.relative_to(day_root)
        except ValueError as exc:
            raise ActionCriticTargetError(f"{label} is outside its portable day root") from exc
    if sha256_file(manifest_path) != target_day.get("day_manifest_sha256"):
        raise ActionCriticTargetError("portable day manifest SHA-256 mismatch")
    if sha256_file(receipt_path) != target_day.get("day_receipt_sha256"):
        raise ActionCriticTargetError("portable day receipt SHA-256 mismatch")
    _manifest_path, manifest, manifest_sha = _read_json_object(manifest_path, label="day manifest")
    _receipt_path, receipt, receipt_sha = _read_json_object(receipt_path, label="day receipt")
    if manifest_sha != target_day.get("day_manifest_sha256") or receipt_sha != target_day.get(
        "day_receipt_sha256"
    ):
        raise ActionCriticTargetError("portable day document identity drifted")
    if (
        manifest.get("utc_day") != target_day.get("utc_day")
        or manifest.get("split") != target_day.get("split")
        or receipt.get("manifest_sha256") != manifest_sha
        or receipt.get("manifest_path") != _relative(manifest_path, day_root)
    ):
        raise ActionCriticTargetError("portable day manifest or receipt binding drifted")
    target = target_day.get("target_shard")
    if not isinstance(target, Mapping):
        raise ActionCriticTargetError("portable target shard binding is absent")
    target_path = _portable_member(day_root, target.get("path"), label="target shard")
    manifest_target = manifest.get("target_shard")
    if not isinstance(manifest_target, Mapping) or dict(manifest_target) != dict(target):
        raise ActionCriticTargetError("portable target shard binding disagrees with day manifest")
    if (
        target_path.stat().st_size != target.get("size_bytes")
        or sha256_file(target_path) != target.get("sha256")
        or receipt.get("target_shard_sha256") != target.get("sha256")
        or receipt.get("target_shard_size_bytes") != target.get("size_bytes")
        or receipt.get("target_row_count") != target.get("row_count")
    ):
        raise ActionCriticTargetError("portable target shard receipt binding drifted")
    return {
        "day_artifact_root": day_root,
        "day_manifest_path": manifest_path,
        "day_receipt_path": receipt_path,
        "target_shard_path": target_path,
    }


def _only_json_member(directory: Path, *, label: str) -> Path:
    members = sorted(
        path for path in directory.glob("*.json") if path.is_file() and not path.is_symlink()
    )
    if len(members) != 1:
        raise ActionCriticTargetError(f"{label} must contain exactly one JSON document")
    return members[0]


def _validate_target_row(
    row: Mapping[str, Any], *, expected_day: str, expected_split: str
) -> tuple[tuple[str, str, int], str, int, str, Counter[str]]:
    if row.get("schema") != TARGET_OVERLAY_SCHEMA:
        raise ActionCriticTargetError("foreign target shard row schema")
    _assert_no_forbidden_keys(row)
    if (
        row.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or row.get("target_only") is not True
        or row.get("hidden_information_fields_present") is not False
    ):
        raise ActionCriticTargetError("target shard row crosses the information boundary")
    if row.get("utc_day") != expected_day or row.get("split") != expected_split:
        raise ActionCriticTargetError("target shard row day or split drifted")
    source_sha = _require_string(row, "source_archive_sha256")
    _sha_hex(source_sha)
    source_member = _require_source_member(row)
    episode = _require_string(row, "episode_id")
    seat = _exact_int(row.get("acting_seat"))
    env_step = _exact_int(row.get("env_step"))
    program_identity = _require_string(row, "program_identity")
    if seat not in (0, 1) or env_step is None or env_step < 0:
        raise ActionCriticTargetError("target shard row acting seat or env_step is invalid")
    z_mask = row.get("z_mask")
    win_mask = row.get("win_target_mask")
    z = row.get("z")
    win = row.get("win_target_one_only_for_z_plus1")
    if not isinstance(z_mask, bool) or not isinstance(win_mask, bool) or z_mask != win_mask:
        raise ActionCriticTargetError("terminal target masks are malformed")
    if z_mask:
        if _finite_number(z) not in {-1.0, 0.0, 1.0}:
            raise ActionCriticTargetError("terminal z target is invalid")
        expected_win = 1.0 if float(z) == 1.0 else 0.0
        if _finite_number(win) != expected_win:
            raise ActionCriticTargetError("terminal binary win target is invalid")
    elif z is not None or win is not None:
        raise ActionCriticTargetError("masked terminal target carries a fabricated value")
    counters: Counter[str] = Counter()
    counters["complete_action_programs"] += 1
    counters["terminal_win_labeled" if z_mask else "terminal_win_masked"] += 1
    horizons = row.get("prize_differential")
    if not isinstance(horizons, Mapping):
        raise ActionCriticTargetError("target shard lacks prize horizons")
    for horizon in HORIZONS:
        value = horizons.get(f"h{horizon}")
        if not isinstance(value, Mapping) or value.get("h") != horizon:
            raise ActionCriticTargetError("target horizon identity is invalid")
        mask = value.get("mask")
        reason = value.get("unavailable_reason")
        required = (
            "future_program_identity",
            "future_env_step",
            "own_remaining_before",
            "own_remaining_after",
            "opponent_remaining_before",
            "opponent_remaining_after",
            "own_taken",
            "opponent_taken",
            "differential",
        )
        if not isinstance(mask, bool) or any(field not in value for field in required):
            raise ActionCriticTargetError("target horizon required fields are absent")
        name = f"prize_h{horizon}"
        if mask:
            if reason is not None:
                raise ActionCriticTargetError("labeled target horizon has an unavailable reason")
            before_own = _exact_int(value["own_remaining_before"])
            after_own = _exact_int(value["own_remaining_after"])
            before_opp = _exact_int(value["opponent_remaining_before"])
            after_opp = _exact_int(value["opponent_remaining_after"])
            own_taken = _exact_int(value["own_taken"])
            opp_taken = _exact_int(value["opponent_taken"])
            differential = _finite_number(value["differential"])
            if (
                None in (before_own, after_own, before_opp, after_opp, own_taken, opp_taken)
                or differential is None
                or not -1.0 <= differential <= 1.0
                or not all(1 <= count <= 6 for count in (before_own, after_own, before_opp, after_opp))
                or after_own > before_own
                or after_opp > before_opp
                or own_taken != before_own - after_own
                or opp_taken != before_opp - after_opp
                or differential != max(-1.0, min(1.0, (own_taken - opp_taken) / 3.0))
            ):
                raise ActionCriticTargetError("labeled target horizon facts are inconsistent")
            counters[f"{name}_labeled"] += 1
        else:
            if not isinstance(reason, str) or not reason or value.get("differential") is not None:
                raise ActionCriticTargetError("masked target horizon is malformed")
            counters[f"{name}_masked"] += 1
            if reason == "pre_action_public_prize_count_outside_valid_1_to_6_range":
                endpoint_values = (
                    value["own_remaining_before"],
                    value["own_remaining_after"],
                    value["opponent_remaining_before"],
                    value["opponent_remaining_after"],
                )
                counter = (
                    "zero_prize_setup_mask_count"
                    if 0 in endpoint_values
                    else "out_of_range_prize_mask_count"
                )
                counters[counter] += 1
            if reason == "non_monotone_public_prize_count":
                counters["non_monotone_mask_count"] += 1
    return (source_sha, episode, seat), source_member, env_step, program_identity, counters


def _day_artifact_documents(root: Path) -> tuple[Path, dict[str, Any], str, Path, dict[str, Any], str]:
    manifest_path = _only_json_member(root / "manifests", label="day manifests")
    receipt_path = _only_json_member(root / "receipts", label="day receipts")
    manifest_path, manifest, manifest_sha = _read_json_object(manifest_path, label="day manifest")
    receipt_path, receipt, receipt_sha = _read_json_object(receipt_path, label="day receipt")
    if manifest.get("schema") != TARGET_DAY_MANIFEST_SCHEMA:
        raise ActionCriticTargetError("foreign target day manifest")
    if receipt.get("schema") != TARGET_DAY_RECEIPT_SCHEMA:
        raise ActionCriticTargetError("foreign target day receipt")
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ActionCriticTargetError("day receipt does not bind its manifest")
    if receipt.get("manifest_path") != _relative(manifest_path, root):
        raise ActionCriticTargetError("day receipt manifest path drifted")
    return manifest_path, manifest, manifest_sha, receipt_path, receipt, receipt_sha


def finalize_action_critic_target_set(
    *,
    day_artifact_roots: Iterable[Path | str],
    output_root: Path | str,
    goal_contract_path: Path | str,
    expected_goal_contract_sha256: str,
    base_pack_completion_path: Path | str,
    expected_base_pack_completion_sha256: str,
    complete_action_overlay_manifest_path: Path | str,
    expected_complete_action_overlay_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and atomically seal the exact 20-day target-set manifest."""

    roots = [_read_only_directory(item, label="target day artifact root") for item in day_artifact_roots]
    if len(roots) != len(WINDOW_DAYS) or len(set(roots)) != len(roots):
        raise ActionCriticTargetError("target-set finalizer requires exactly 20 unique day artifact roots")
    final_root = _safe_output_root(output_root)
    contract_path, contract, contract_sha = _load_goal_contract(
        goal_contract_path, expected_sha256=expected_goal_contract_sha256
    )
    contract_goal_revision = _exact_int(contract.get("goal_revision"))
    if contract_goal_revision is None:  # _load_goal_contract already establishes this.
        raise ActionCriticTargetError("goal contract revision is invalid")
    base_path, base, base_sha = _read_json_object(base_pack_completion_path, label="base pack completion")
    _sha_hex(expected_base_pack_completion_sha256)
    if base_sha != expected_base_pack_completion_sha256:
        raise ActionCriticTargetError("base pack completion SHA-256 mismatch")
    if base.get("schema") != "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1":
        raise ActionCriticTargetError("foreign base pack completion")
    overlay_manifest_path, overlay_manifest, overlay_manifest_sha = _read_json_object(
        complete_action_overlay_manifest_path, label="complete-action overlay manifest"
    )
    _sha_hex(expected_complete_action_overlay_manifest_sha256)
    if overlay_manifest_sha != expected_complete_action_overlay_manifest_sha256:
        raise ActionCriticTargetError("complete-action overlay manifest SHA-256 mismatch")
    if overlay_manifest.get("schema") != "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1":
        raise ActionCriticTargetError("foreign complete-action overlay manifest")
    overlay_shards = overlay_manifest.get("overlay_shards")
    if not isinstance(overlay_shards, list) or len(overlay_shards) != len(WINDOW_DAYS):
        raise ActionCriticTargetError(
            "complete-action overlay manifest must contain exactly 20 shard entries"
        )
    overlay_by_day: dict[str, dict[str, Any]] = {}
    for item in overlay_shards:
        if not isinstance(item, Mapping):
            raise ActionCriticTargetError("complete-action overlay shard is not an object")
        day = item.get("utc_day")
        if not isinstance(day, str) or day not in WINDOW_DAYS or day in overlay_by_day:
            raise ActionCriticTargetError(
                "complete-action overlay days must be unique and exactly match the recent-20 window"
            )
        if item.get("split") != SPLIT_BY_DAY[day]:
            raise ActionCriticTargetError(
                f"complete-action overlay split drifted for {day}"
            )
        shard_sha = item.get("sha256")
        if not isinstance(shard_sha, str):
            raise ActionCriticTargetError("complete-action overlay shard SHA-256 is absent")
        _sha_hex(shard_sha)
        overlay_by_day[day] = dict(item)
    if set(overlay_by_day) != set(WINDOW_DAYS):
        raise ActionCriticTargetError(
            "complete-action overlay day inventory is not the exact recent-20 window"
        )

    private_root = Path(tempfile.mkdtemp(prefix=f".{final_root.name}.private-", dir=final_root.parent))
    (private_root / "manifests").mkdir()
    (private_root / "receipts").mkdir()
    (private_root / "bindings").mkdir()
    (private_root / "days").mkdir()
    database_path = private_root / ".group-split-validation.sqlite3"
    aggregate: Counter[str] = Counter()
    day_rows: list[dict[str, Any]] = []
    observed_days: set[str] = set()
    observed_splits: dict[str, set[str]] = {"train": set(), "validation": set(), "evaluation": set()}
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE groups (archive_sha TEXT, episode_id TEXT, seat INTEGER, split TEXT, PRIMARY KEY (archive_sha, episode_id, seat))"
        )
        connection.execute(
            "CREATE TABLE closed_groups (archive_sha TEXT, episode_id TEXT, seat INTEGER, PRIMARY KEY (archive_sha, episode_id, seat))"
        )
        connection.execute(
            "CREATE TABLE program_ids (archive_sha TEXT, episode_id TEXT, seat INTEGER, env_step INTEGER, program_identity TEXT, PRIMARY KEY (archive_sha, episode_id, seat, env_step), UNIQUE (program_identity))"
        )
        connection.execute(
            "CREATE TABLE episode_members (archive_sha TEXT, episode_id TEXT, source_member TEXT, PRIMARY KEY (archive_sha, episode_id))"
        )
        for root in roots:
            manifest_path, manifest, manifest_sha, receipt_path, receipt, receipt_sha = _day_artifact_documents(root)
            manifest_contract = manifest.get("goal_contract")
            if not isinstance(manifest_contract, Mapping):
                raise ActionCriticTargetError("day artifact goal contract binding is absent")
            if (
                manifest.get("owner_goal_revision") != OWNER_GOAL_REVISION
                or manifest_contract.get("sha256") != contract_sha
                or manifest_contract.get("goal_revision") != contract_goal_revision
                or manifest_contract.get("critic_semantic_owner_goal_revision")
                != OWNER_GOAL_REVISION
                or manifest_contract.get("required_authority") != CRITIC_AUTHORITY_KEY
            ):
                raise ActionCriticTargetError("day artifact goal contract binding drifted")
            if (
                receipt.get("owner_goal_revision") != OWNER_GOAL_REVISION
                or receipt.get("goal_contract_sha256") != contract_sha
                or receipt.get("goal_contract_goal_revision") != contract_goal_revision
                or receipt.get("critic_semantic_owner_goal_revision") != OWNER_GOAL_REVISION
                or receipt.get("coverage") != manifest.get("coverage")
            ):
                raise ActionCriticTargetError("day receipt contract or coverage binding drifted")
            day = str(manifest.get("utc_day") or "")
            split = str(manifest.get("split") or "")
            if day not in WINDOW_DAYS or split != SPLIT_BY_DAY[day] or day in observed_days:
                raise ActionCriticTargetError("target day or split inventory drifted")
            observed_days.add(day)
            observed_splits[split].add(day)
            expected_overlay = overlay_by_day[day]
            complete = dict(manifest.get("complete_action_overlay") or {})
            if complete.get("sha256") != expected_overlay.get("sha256") or complete.get("split") not in (None, split):
                raise ActionCriticTargetError("target day complete-action overlay binding drifted")
            raw_zip = dict(manifest.get("raw_episode_zip") or {})
            target = dict(manifest.get("target_shard") or {})
            if raw_zip.get("source_archive_sha256_verified") is not True:
                raise ActionCriticTargetError("day raw ZIP/source-archive binding is not verified")
            target_path = _artifact_member(root, target.get("path"), label="target shard")
            if target_path.stat().st_size != target.get("size_bytes") or sha256_file(target_path) != target.get("sha256"):
                raise ActionCriticTargetError("target shard SHA-256 or size mismatch")
            if (
                receipt.get("complete_action_overlay_sha256") != complete.get("sha256")
                or receipt.get("raw_episode_zip_sha256") != raw_zip.get("sha256")
                or receipt.get("target_shard_sha256") != target.get("sha256")
                or receipt.get("target_shard_size_bytes") != target.get("size_bytes")
            ):
                raise ActionCriticTargetError("day receipt source or target binding drifted")
            row_count = 0
            day_counts: Counter[str] = Counter()
            previous_group: tuple[str, str, int] | None = None
            previous_step: int | None = None
            with target_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ActionCriticTargetError(f"invalid target row {target_path}:{line_number}") from exc
                    if not isinstance(value, Mapping):
                        raise ActionCriticTargetError("target row is not an object")
                    group, source_member, env_step, program_identity, row_counts = _validate_target_row(
                        value, expected_day=day, expected_split=split
                    )
                    if value.get("goal_contract_goal_revision") != contract_goal_revision:
                        raise ActionCriticTargetError(
                            "target row current goal-contract revision drifted"
                        )
                    if group[0] != raw_zip.get("sha256"):
                        raise ActionCriticTargetError("target row raw ZIP identity drifted")
                    existing_member = connection.execute(
                        "SELECT source_member FROM episode_members WHERE archive_sha=? AND episode_id=?",
                        group[:2],
                    ).fetchone()
                    if existing_member is None:
                        connection.execute(
                            "INSERT INTO episode_members VALUES (?, ?, ?)",
                            (*group[:2], source_member),
                        )
                    elif existing_member[0] != source_member:
                        raise ActionCriticTargetError(
                            "target row source_member conflicts for one raw archive episode"
                        )
                    if group == previous_group:
                        if previous_step is None or env_step <= previous_step:
                            raise ActionCriticTargetError(
                                "target group env_steps are not strictly increasing"
                            )
                    else:
                        if previous_group is not None:
                            connection.execute(
                                "INSERT INTO closed_groups VALUES (?, ?, ?)", previous_group
                            )
                        if connection.execute(
                            "SELECT 1 FROM closed_groups WHERE archive_sha=? AND episode_id=? AND seat=?", group
                        ).fetchone() is not None:
                            raise ActionCriticTargetError("target group is non-contiguous")
                    try:
                        connection.execute(
                            "INSERT INTO program_ids VALUES (?, ?, ?, ?, ?)",
                            (*group, env_step, program_identity),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ActionCriticTargetError(
                            "target shard has duplicate program identity or same-seat env_step"
                        ) from exc
                    existing = connection.execute(
                        "SELECT split FROM groups WHERE archive_sha=? AND episode_id=? AND seat=?", group
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO groups VALUES (?, ?, ?, ?)", (*group, split)
                        )
                    elif existing[0] != split:
                        raise ActionCriticTargetError("episode/seat group crosses target splits")
                    day_counts.update(row_counts)
                    row_count += 1
                    previous_group = group
                    previous_step = env_step
            if previous_group is not None:
                connection.execute(
                    "INSERT INTO closed_groups VALUES (?, ?, ?)", previous_group
                )
            if row_count != target.get("row_count") or row_count != receipt.get("target_row_count"):
                raise ActionCriticTargetError("target day row-count receipt drifted")
            if day_counts != Counter(dict(manifest.get("coverage", {}).get("counts") or {})):
                raise ActionCriticTargetError("target day coverage receipt drifted")
            aggregate.update(day_counts)
            portable_day_root = private_root / "days" / day
            _copy_tree_create_only(root, portable_day_root)
            portable_manifest_path = _relative(
                portable_day_root / _relative(manifest_path, root), private_root
            )
            portable_receipt_path = _relative(
                portable_day_root / _relative(receipt_path, root), private_root
            )
            portable_day = {
                "utc_day": day,
                "split": split,
                "day_artifact_root": _relative(portable_day_root, private_root),
                "day_manifest_path": portable_manifest_path,
                "day_manifest_sha256": manifest_sha,
                "day_receipt_path": portable_receipt_path,
                "day_receipt_sha256": receipt_sha,
                "raw_episode_zip": {
                    "sha256": raw_zip.get("sha256"),
                    "size_bytes": raw_zip.get("size_bytes"),
                    "source_archive_sha256_verified": True,
                },
                "complete_action_overlay": {
                    key: complete[key]
                    for key in ("sha256", "size_bytes", "schema", "split")
                    if key in complete
                },
                "target_shard": target,
                "coverage": manifest["coverage"],
            }
            resolve_action_critic_target_set_day(private_root, portable_day)
            day_rows.append(portable_day)
        connection.commit()
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
    if observed_days != set(WINDOW_DAYS) or any(
        observed_splits[name] != {day for day in WINDOW_DAYS if SPLIT_BY_DAY[day] == name}
        for name in observed_splits
    ):
        raise ActionCriticTargetError("target-set split day lists are incomplete")
    day_rows.sort(key=lambda item: item["utc_day"])
    normalized_coverage = _counter_summary(aggregate)
    portable_goal_contract = _portable_binding(
        private_root,
        contract_path,
        sha256=contract_sha,
        label="goal contract",
        suffix="goal-contract",
    )
    portable_base_completion = _portable_binding(
        private_root,
        base_path,
        sha256=base_sha,
        label="base pack completion",
        suffix="base-pack-completion",
    )
    portable_overlay_manifest = _portable_binding(
        private_root,
        overlay_manifest_path,
        sha256=overlay_manifest_sha,
        label="complete-action overlay manifest",
        suffix="complete-action-overlay-manifest",
    )
    target_set = {
        "schema": TARGET_SET_MANIFEST_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_goal_revision": contract_goal_revision,
        "critic_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        "required_critic_authority": CRITIC_AUTHORITY_KEY,
        "goal_contract": portable_goal_contract,
        "base_pack_completion": portable_base_completion,
        "complete_action_overlay_manifest": portable_overlay_manifest,
        "source_days": list(WINDOW_DAYS),
        "split_days": {name: sorted(days) for name, days in observed_splits.items()},
        "target_days": day_rows,
        "all_20_raw_episode_zip_sha256s": [
            {"utc_day": row["utc_day"], "sha256": row["raw_episode_zip"]["sha256"], "size_bytes": row["raw_episode_zip"]["size_bytes"]}
            for row in day_rows
        ],
        "all_20_target_shards": [
            {"utc_day": row["utc_day"], "sha256": row["target_shard"]["sha256"], "size_bytes": row["target_shard"]["size_bytes"], "row_count": row["target_shard"]["row_count"], "split": row["split"]}
            for row in day_rows
        ],
        "coverage": {
            "counts": normalized_coverage,
            "zero_prize_setup_mask_count": normalized_coverage["zero_prize_setup_mask_count"],
            "non_monotone_mask_count": normalized_coverage["non_monotone_mask_count"],
        },
        "episode_and_seat_group_split_disjoint": True,
        "information_boundary": {"hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed": False},
        "publication": {
            "create_only": True,
            "atomic_root_rename": True,
            "atomic_root_no_replace": True,
            "artifact_directories_fsynced_before_publication": True,
            "publication_parent_fsynced_after_publication": True,
        },
    }
    body = canonical_bytes(target_set)
    set_sha = _SHA256_PREFIX + hashlib.sha256(body).hexdigest()
    manifest_path = private_root / "manifests" / f"sha256-{_sha_hex(set_sha)}.target-set-manifest.json"
    _write_file_create_only(manifest_path, body)
    receipt = {
        "schema": TARGET_SET_RECEIPT_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "goal_contract_goal_revision": contract_goal_revision,
        "critic_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        "required_critic_authority": CRITIC_AUTHORITY_KEY,
        "target_set_manifest_path": _relative(manifest_path, private_root),
        "target_set_manifest_sha256": set_sha,
        "goal_contract_sha256": contract_sha,
        "base_pack_completion_sha256": base_sha,
        "complete_action_overlay_manifest_sha256": overlay_manifest_sha,
        "day_count": len(day_rows),
        "coverage": target_set["coverage"],
        "episode_and_seat_group_split_disjoint": True,
        "atomic_root_no_replace": True,
        "artifact_directories_fsynced_before_publication": True,
        "publication_parent_fsynced_after_publication": True,
    }
    receipt_body = canonical_bytes(receipt)
    receipt_sha = _SHA256_PREFIX + hashlib.sha256(receipt_body).hexdigest()
    receipt_path = private_root / "receipts" / f"sha256-{_sha_hex(receipt_sha)}.target-set-receipt.json"
    _write_file_create_only(receipt_path, receipt_body)
    _atomic_publish_directory_noreplace(private_root, final_root)
    return {
        "output_root": str(final_root),
        "manifest_path": str(final_root / _relative(manifest_path, private_root)),
        "manifest_sha256": set_sha,
        "receipt_path": str(final_root / _relative(receipt_path, private_root)),
        "receipt_sha256": receipt_sha,
        "day_count": len(day_rows),
        "coverage": target_set["coverage"],
    }


__all__ = [
    "COMPLETE_ACTION_OVERLAY_SCHEMA",
    "CRITIC_AUTHORITY_KEY",
    "HORIZONS",
    "OWNER_GOAL_REVISION",
    "SPLIT_BY_DAY",
    "TARGET_DAY_MANIFEST_SCHEMA",
    "TARGET_DAY_RECEIPT_SCHEMA",
    "TARGET_MANIFEST_SCHEMA",
    "TARGET_OVERLAY_SCHEMA",
    "TARGET_RECEIPT_SCHEMA",
    "TARGET_SCHEMA",
    "TARGET_SET_MANIFEST_SCHEMA",
    "TARGET_SET_RECEIPT_SCHEMA",
    "WINDOW_DAYS",
    "ActionCriticTargetError",
    "build_action_critic_target_overlay_day",
    "canonical_bytes",
    "canonical_sha256",
    "finalize_action_critic_target_set",
    "resolve_action_critic_target_set_day",
    "sha256_file",
    "target_schema_document",
]
