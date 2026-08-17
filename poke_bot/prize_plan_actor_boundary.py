"""Receipt-bound loader for the trainer-only Prize-plan-v2 H3 provider.

The policy loop remains exactly legacy when no provider is supplied.  An
enabled provider is accepted only when an immutable cache, an explicit later
activation receipt, the current learner parent, and the complete in-memory
replay membership all agree.  This module performs no critic inference and is
never imported by serving/runtime action selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .frozen_prize_plan_advantage import (
    ACTIVATION_RECEIPT_SCHEMA,
    H3_COEFFICIENT,
    PortableStageAdvantage,
    bind_portable_stage_advantages,
    canonical_sha256,
)


CACHE_ROW_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_policy_additive_cache/v1"
CACHE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_h3_policy_additive_cache_receipt/v1"
)
REPLAY_MEMBERSHIP_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_h3_replay_membership/v1"
)
PROVIDER_BINDING_SCHEMA = (
    "poke_bot.alakazam_prize_plan_v2_h3_actor_provider_binding/v1"
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class PrizePlanActorBoundaryError(ValueError):
    """The H3 provider is not safe for this actor/optimizer boundary."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PrizePlanActorBoundaryError(
            f"{label} must be a lowercase sha256:<64-hex> digest"
        )
    return value


def _regular_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if Path(path).expanduser().is_symlink() or not resolved.is_file():
        raise PrizePlanActorBoundaryError(f"{label} must be a regular non-symlink file")
    return resolved


def _load_json(path: Path, expected_sha256: str, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, label)
    expected = _digest(expected_sha256, f"{label} SHA-256")
    actual = _sha256_file(resolved)
    if actual != expected:
        raise PrizePlanActorBoundaryError(
            f"{label} digest mismatch: expected={expected} actual={actual}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrizePlanActorBoundaryError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PrizePlanActorBoundaryError(f"{label} must contain one JSON object")
    return resolved, value


def _self_digest(value: Mapping[str, Any], label: str) -> str:
    payload = dict(value)
    observed = payload.pop("artifact_sha256", None)
    expected = canonical_sha256(payload)
    if observed != expected:
        raise PrizePlanActorBoundaryError(f"{label} artifact digest mismatch")
    return expected


def replay_membership_rows(sequences: Sequence[object]) -> list[dict[str, object]]:
    """Return the stable, order-independent complete replay membership."""

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for sequence in sequences:
        episode_id = str(getattr(sequence, "episode_id", "") or "")
        seat = getattr(sequence, "seat", None)
        decisions = getattr(sequence, "decisions", None)
        if not episode_id or seat not in (0, 1) or not isinstance(decisions, list):
            raise PrizePlanActorBoundaryError("replay sequence identity is malformed")
        for decision in decisions:
            env_step = getattr(decision, "env_step", None)
            stages = getattr(decision, "policy_stages", None)
            if (
                isinstance(env_step, bool)
                or not isinstance(env_step, int)
                or env_step < 0
                or not isinstance(stages, list)
                or not stages
            ):
                raise PrizePlanActorBoundaryError("replay decision/stage identity is malformed")
            key = (episode_id, int(seat), env_step)
            if key in seen:
                raise PrizePlanActorBoundaryError("replay membership repeats an action key")
            seen.add(key)
            rows.append(
                {
                    "episode_id": episode_id,
                    "seat": int(seat),
                    "env_step": env_step,
                    "stage_count": len(stages),
                }
            )
    if not rows:
        raise PrizePlanActorBoundaryError("replay membership is empty")
    return sorted(
        rows,
        key=lambda row: (
            str(row["episode_id"]),
            int(row["seat"]),
            int(row["env_step"]),
        ),
    )


def replay_membership_sha256(sequences: Sequence[object]) -> str:
    return canonical_sha256(
        {"schema": REPLAY_MEMBERSHIP_SCHEMA, "actions": replay_membership_rows(sequences)}
    )


def _cache_records(
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> tuple[list[PortableStageAdvantage], str, int]:
    shards = receipt.get("day_shards")
    if not isinstance(shards, list) or not shards:
        raise PrizePlanActorBoundaryError("H3 cache receipt has no immutable shards")
    records: list[PortableStageAdvantage] = []
    action_stage_counts: dict[tuple[str, int, int], int] = {}
    shard_paths: set[str] = set()
    for descriptor in shards:
        if not isinstance(descriptor, Mapping):
            raise PrizePlanActorBoundaryError("H3 cache shard descriptor is malformed")
        relative = descriptor.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).name != relative
            or relative in shard_paths
        ):
            raise PrizePlanActorBoundaryError("H3 cache shard path is not a unique basename")
        shard_paths.add(relative)
        shard = _regular_file(receipt_path.parent / relative, "H3 cache shard")
        if shard.parent != receipt_path.parent:
            raise PrizePlanActorBoundaryError("H3 cache shard escaped its sealed root")
        if _sha256_file(shard) != _digest(descriptor.get("sha256"), "cache shard SHA-256"):
            raise PrizePlanActorBoundaryError("H3 cache shard digest mismatch")
        expected_rows = descriptor.get("rows")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 1:
            raise PrizePlanActorBoundaryError("H3 cache shard row count is invalid")
        observed_rows = 0
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise PrizePlanActorBoundaryError("H3 cache shard contains a blank row")
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PrizePlanActorBoundaryError(
                        f"H3 cache row {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(row, Mapping) or row.get("schema") != CACHE_ROW_SCHEMA:
                    raise PrizePlanActorBoundaryError("H3 cache row schema mismatch")
                episode_id = str(row.get("policy_episode_id") or "")
                seat = row.get("acting_seat")
                env_step = row.get("env_step")
                stage_count = row.get("stage_count")
                addend = row.get("h3_additive_term")
                if (
                    not episode_id
                    or seat not in (0, 1)
                    or isinstance(env_step, bool)
                    or not isinstance(env_step, int)
                    or env_step < 0
                    or isinstance(stage_count, bool)
                    or not isinstance(stage_count, int)
                    or stage_count < 1
                    or stage_count > 32
                    or isinstance(addend, bool)
                    or not isinstance(addend, (int, float))
                    or not math.isfinite(float(addend))
                    or abs(float(addend)) > 2.5
                    or row.get("utc_day") != descriptor.get("utc_day")
                    or row.get("split") != descriptor.get("split")
                ):
                    raise PrizePlanActorBoundaryError("H3 cache row identity/value is malformed")
                action_key = (episode_id, int(seat), env_step)
                if action_key in action_stage_counts:
                    raise PrizePlanActorBoundaryError("H3 cache repeats a complete action")
                action_stage_counts[action_key] = stage_count
                for stage_index in range(stage_count):
                    records.append(
                        PortableStageAdvantage(
                            episode_id=episode_id,
                            seat=int(seat),
                            env_step=env_step,
                            stage_index=stage_index,
                            advantage=float(addend),
                        )
                    )
                observed_rows += 1
        if observed_rows != expected_rows:
            raise PrizePlanActorBoundaryError("H3 cache shard row count mismatch")
    return records, canonical_sha256(
        {
            "schema": REPLAY_MEMBERSHIP_SCHEMA,
            "actions": sorted(
                (
                    {
                        "episode_id": episode_id,
                        "seat": seat,
                        "env_step": env_step,
                        "stage_count": stage_count,
                    }
                    for (episode_id, seat, env_step), stage_count in action_stage_counts.items()
                ),
                key=lambda row: (
                    str(row["episode_id"]),
                    int(row["seat"]),
                    int(row["env_step"]),
                ),
            ),
        }
    ), len(action_stage_counts)


def validate_activation_receipt(
    receipt: Mapping[str, Any],
    *,
    cache_receipt_sha256: str,
    cache_artifact_sha256: str,
    policy_checkpoint_sha256: str,
    replay_membership_digest: str,
) -> dict[str, Any]:
    """Validate the later owner-authorized clean-boundary receipt."""

    immutable_receipt = dict(receipt)
    immutable_receipt.pop("provider_binding", None)
    _self_digest(immutable_receipt, "H3 activation receipt")
    required_true = (
        "activation_eligible",
        "actor_activation",
        "safe_boundary",
        "all_pre_activation_gates_passed",
        "contract_current_activation_allowed",
        "exact_legacy_baseline_computed_in_batch",
        "rollback_preflight_passed",
        "noninterference_passed",
        "no_search_rtp_mcts",
    )
    if receipt.get("schema") != ACTIVATION_RECEIPT_SCHEMA or any(
        receipt.get(field) is not True for field in required_true
    ):
        raise PrizePlanActorBoundaryError("H3 activation receipt is not passing")
    if (
        receipt.get("cache_value_semantics") != "h3_additive_term_only"
        or float(receipt.get("coefficient", float("nan"))) != H3_COEFFICIENT
        or receipt.get("h1_h6_h12_actor_coefficients") != [0.0, 0.0, 0.0]
        or receipt.get("runtime_critic_calls") is not False
        or receipt.get("cache_receipt_sha256") != cache_receipt_sha256
        or receipt.get("cache_artifact_sha256") != cache_artifact_sha256
        or receipt.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
        or receipt.get("replay_membership_sha256") != replay_membership_digest
        or receipt.get("semantic_owner_goal_revision") != 23
    ):
        raise PrizePlanActorBoundaryError("H3 activation receipt identity drifted")
    _digest(receipt.get("contract_sha256"), "activation contract SHA-256")
    return dict(receipt)


def load_h3_actor_provider(
    *,
    sequences: Sequence[object],
    policy_checkpoint_sha256: str,
    cache_receipt_path: Path,
    cache_receipt_sha256: str,
    activation_receipt_path: Path,
    activation_receipt_sha256: str,
) -> tuple[dict[tuple[int, int, int], float], dict[str, Any]]:
    """Load and bind one exact replay-window H3 provider or fail closed."""

    policy_sha = _digest(policy_checkpoint_sha256, "policy checkpoint SHA-256")
    cache_path, cache = _load_json(
        cache_receipt_path, cache_receipt_sha256, "H3 cache receipt"
    )
    if (
        cache.get("schema") != CACHE_RECEIPT_SCHEMA
        or cache.get("cache_value_semantics") != "h3_additive_term_only"
        or cache.get("exact_legacy_baseline_computed_in_batch") is not True
        or float(cache.get("coefficient", float("nan"))) != H3_COEFFICIENT
        or cache.get("h1_h6_h12_actor_coefficients") != [0.0, 0.0, 0.0]
        or cache.get("runtime_critic_calls") is not False
    ):
        raise PrizePlanActorBoundaryError("H3 cache receipt semantics drifted")
    cache_artifact_sha = _self_digest(cache, "H3 cache receipt")
    records, cache_membership_sha, action_count = _cache_records(cache_path, cache)
    dataset_membership_sha = replay_membership_sha256(sequences)
    if cache_membership_sha != dataset_membership_sha:
        raise PrizePlanActorBoundaryError("H3 cache/replay membership digest mismatch")
    activation_path, activation = _load_json(
        activation_receipt_path,
        activation_receipt_sha256,
        "H3 activation receipt",
    )
    validated_activation = validate_activation_receipt(
        activation,
        cache_receipt_sha256=_sha256_file(cache_path),
        cache_artifact_sha256=cache_artifact_sha,
        policy_checkpoint_sha256=policy_sha,
        replay_membership_digest=dataset_membership_sha,
    )
    bound = bind_portable_stage_advantages(sequences, records)
    provider = {
        **validated_activation,
        "provider_binding": {
            "schema": PROVIDER_BINDING_SCHEMA,
            "cache_receipt_path": str(cache_path),
            "cache_receipt_sha256": _sha256_file(cache_path),
            "cache_artifact_sha256": cache_artifact_sha,
            "activation_receipt_path": str(activation_path),
            "activation_receipt_sha256": _sha256_file(activation_path),
            "replay_membership_sha256": dataset_membership_sha,
            "complete_actions": action_count,
            "factorized_stages": len(bound),
            "policy_checkpoint_sha256": policy_sha,
        },
    }
    return bound, provider
