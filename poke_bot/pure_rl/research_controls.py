"""Versioned research-control roster and gate-retirement safety contract.

Research controls are additive greedy diagnostic opponents. They are never
training data, formal-gate evidence, or contributors to an active gate's
pass/fail result. Once an exact active gate is committed and passes every
required check, its immutable opponent packages can be appended here for
measurement by a later training lineage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REGISTRY_SCHEMA = "poke_bot.research_control_registry/v1"
GATE_RESULT_SCHEMA = "poke_bot.public_agent_gate_result/v1"
LEGACY_SEED_CONTROLS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}


def _is_digest(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(ch in "0123456789abcdef" for ch in text[7:])
    )


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an exact integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _durable_atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The atomic rename still holds on filesystems that reject a
            # directory fsync; the file itself was flushed above.
            pass
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def validate_research_control_registry(
    registry: Mapping[str, Any],
    *,
    installed_digests: Mapping[str, str] | None = None,
    active_gate_ids: tuple[str, ...] = (),
    active_gate_digests: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a normalized copy or fail closed on identity/role ambiguity."""
    data = json.loads(json.dumps(dict(registry)))
    if data.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("research-control registry has the wrong schema")
    if (
        not str(data.get("registry_id") or "")
        or _exact_int(data.get("version"), "registry version") < 1
    ):
        raise ValueError("research-control registry identity/version is invalid")
    controls = data.get("controls")
    retirements = data.get("retirements")
    if not isinstance(controls, list) or not controls:
        raise ValueError("research-control registry has no controls")
    if not isinstance(retirements, list):
        raise ValueError("research-control registry retirements are missing")

    ids: list[str] = []
    digests: list[str] = []
    active = set(active_gate_ids)
    active_digests = set(active_gate_digests)
    installed = dict(installed_digests or {})
    for row in controls:
        if not isinstance(row, dict):
            raise ValueError("research-control entry must be an object")
        opponent_id = str(row.get("opponent_id") or "")
        content_digest = str(row.get("content_digest") or "")
        if not opponent_id or not _is_digest(content_digest):
            raise ValueError("research-control identity/content digest is invalid")
        if (
            _finite_float(row.get("gate_weight"), "research-control gate_weight")
            != 0.0
            or row.get("included_in_gate_pass") is not False
            or row.get("formal_eval") is not False
            or row.get("training_eligible") is not False
        ):
            raise ValueError(
                f"research control {opponent_id!r} has unsafe role semantics"
            )
        if opponent_id in active:
            raise ValueError(
                f"active-gate opponent is also a research control: {opponent_id}"
            )
        if content_digest in active_digests:
            raise ValueError(
                f"active-gate package alias is a research control: {opponent_id}"
            )
        if installed and installed.get(opponent_id) != content_digest:
            raise ValueError(
                f"research-control package digest mismatch for {opponent_id}: "
                f"expected={content_digest!r} "
                f"actual={installed.get(opponent_id)!r}"
            )
        ids.append(opponent_id)
        digests.append(content_digest)
    if len(set(ids)) != len(ids):
        raise ValueError("research-control registry repeats an opponent ID")
    if len(set(digests)) != len(digests):
        raise ValueError("research-control registry contains a package alias")

    retired_gate_ids: set[str] = set()
    retired_control_ids: set[str] = set()
    control_ids = set(ids)
    controls_by_id = {
        str(row["opponent_id"]): row for row in controls if isinstance(row, dict)
    }
    for row in retirements:
        if not isinstance(row, dict):
            raise ValueError("research-control retirement must be an object")
        gate_id = str(row.get("gate_id") or "")
        retired_ids = tuple(str(value) for value in row.get("opponent_ids") or ())
        exact_result_digest = str(row.get("exact_result_digest") or "")
        checkpoint_digest = str(row.get("checkpoint_digest") or "")
        if (
            not gate_id
            or gate_id in retired_gate_ids
            or not _is_digest(exact_result_digest)
            or not _is_digest(checkpoint_digest)
            or not retired_ids
            or len(set(retired_ids)) != len(retired_ids)
            or not set(retired_ids).issubset(control_ids)
            or set(retired_ids) & retired_control_ids
            or _exact_int(row.get("iteration"), "retirement iteration") < 0
            or any(
                str(controls_by_id[opponent_id].get("source_gate_id") or "")
                != gate_id
                or str(
                    controls_by_id[opponent_id].get(
                        "retired_exact_result_digest"
                    )
                    or ""
                )
                != exact_result_digest
                or str(
                    controls_by_id[opponent_id].get(
                        "retired_checkpoint_digest"
                    )
                    or ""
                )
                != checkpoint_digest
                for opponent_id in retired_ids
            )
        ):
            raise ValueError("research-control retirement history is invalid")
        retired_gate_ids.add(gate_id)
        retired_control_ids.update(retired_ids)
    legacy_ids = {
        opponent_id
        for opponent_id, row in controls_by_id.items()
        if str(row.get("source_gate_id") or "") == "legacy-original-four"
    }
    if (
        legacy_ids != set(LEGACY_SEED_CONTROLS)
        or any(
            str(controls_by_id[opponent_id].get("content_digest") or "")
            != expected_digest
            for opponent_id, expected_digest in LEGACY_SEED_CONTROLS.items()
        )
        or control_ids != legacy_ids | retired_control_ids
    ):
        raise ValueError(
            "research controls lack an exact seed or committed retirement proof"
        )
    return data


def load_research_control_registry(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_research_control_registry(data)


def pin_research_control_registry_file(
    source_path: Path,
    *,
    snapshot_dir: Path,
) -> Path:
    """Materialize one content-addressed immutable registry for a lineage."""
    source_bytes = Path(source_path).expanduser().resolve().read_bytes()
    registry = validate_research_control_registry(json.loads(source_bytes))
    canonical = (
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = _bytes_digest(canonical)
    destination = (
        Path(snapshot_dir).expanduser().resolve()
        / f"registry_v{int(registry['version']):06d}_{digest[7:23]}.json"
    )
    if destination.is_file():
        if destination.read_bytes() != canonical:
            raise ValueError("content-addressed research registry snapshot conflicts")
        return destination
    _durable_atomic_write(destination, canonical)
    if destination.read_bytes() != canonical:
        raise RuntimeError("research registry snapshot verification failed")
    return destination


def research_control_ids(registry: Mapping[str, Any]) -> tuple[str, ...]:
    valid = validate_research_control_registry(registry)
    return tuple(str(row["opponent_id"]) for row in valid["controls"])


def retire_passed_gate(
    *,
    registry: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
    exact_result: Mapping[str, Any],
    exact_result_digest: str,
    commit_record: Mapping[str, Any],
    commit_digest: str,
    updated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Append one fully committed passed gate as zero-weight diagnostics."""
    current = validate_research_control_registry(registry)
    contract = dict(gate_contract)
    result = dict(exact_result)
    active = contract.get("next_gate")
    if not isinstance(active, dict):
        raise ValueError("gate contract has no active gate")
    gate_id = str(active.get("id") or "")
    if not gate_id or str(contract.get("active_gate_id") or "") != gate_id:
        raise ValueError("gate contract identity is inconsistent")
    if not _is_digest(exact_result_digest):
        raise ValueError("exact gate result digest is invalid")
    if not _is_digest(commit_digest):
        raise ValueError("immutable iteration commit digest is invalid")
    if (
        result.get("schema") != GATE_RESULT_SCHEMA
        or str(result.get("gate_id") or "") != gate_id
        or result.get("passed") is not True
        or result.get("pipeline_gate_passed") is not True
        or result.get("committed") is not True
        or not _is_digest(result.get("checkpoint_digest"))
        or result.get("promotion_passed") is not True
        or result.get("candidate_safety_passed") is not True
        or str(result.get("commit_digest") or "") != commit_digest
    ):
        raise ValueError("only a committed, pipeline-passed exact gate may retire")
    checks = result.get("checks")
    audit = result.get("audit")
    required_checks = {
        "audit",
        "skill_weighted_win_rate",
        "skill_weighted_confidence_lower",
        "s_tier_mean_floor",
        "individual_opponent_floor",
    }
    if "s_plus_individual_floor" in dict(
        (active.get("pass_criteria") or {})
    ):
        required_checks.add("s_plus_matchup_floor_allowance")
    if (
        not isinstance(checks, dict)
        or set(checks) != required_checks
        or not all(checks[key] is True for key in required_checks)
        or not isinstance(audit, dict)
        or audit.get("passed") is not True
        or audit.get("both_seats") is not True
        or audit.get("greedy") is not True
        or audit.get("exact_distribution") is not True
        or audit.get("exact_weights") is not True
        or list(audit.get("duplicate_job_ids") or [])
        or list(audit.get("missing_job_ids") or [])
        or list(audit.get("unexpected_job_ids") or [])
    ):
        raise ValueError("passed gate result lacks exact audit/check evidence")

    roster = active.get("roster")
    evaluation = active.get("evaluation")
    if (
        not isinstance(roster, list)
        or not all(isinstance(row, dict) for row in roster)
        or not isinstance(evaluation, dict)
        or not roster
    ):
        raise ValueError("active gate roster/evaluation is missing")
    roster_ids = tuple(str(row.get("opponent_id") or "") for row in roster)
    roster_digests = tuple(str(row.get("content_digest") or "") for row in roster)
    if (
        any(not value for value in roster_ids)
        or any(not _is_digest(value) for value in roster_digests)
        or len(set(roster_ids)) != len(roster_ids)
        or len(set(roster_digests)) != len(roster_digests)
    ):
        raise ValueError("active gate roster identity is invalid")
    expected_per = _exact_int(
        evaluation.get("games_per_opponent"), "games_per_opponent"
    )
    expected_total = _exact_int(evaluation.get("games_total"), "games_total")
    if expected_per <= 0 or expected_per % 2 or expected_total <= 0:
        raise ValueError("active gate evaluation counts are invalid")
    criteria = active.get("pass_criteria")
    if not isinstance(criteria, dict):
        raise ValueError("active gate pass criteria are missing")
    if (
        _finite_float(result.get("skill_weighted_wr"), "skill_weighted_wr")
        < _finite_float(
            criteria.get("skill_weighted_win_rate"),
            "skill_weighted_win_rate threshold",
        )
        or _finite_float(result.get("confidence_lower"), "confidence_lower")
        < _finite_float(
            criteria.get("skill_weighted_confidence_lower"),
            "skill_weighted_confidence_lower threshold",
        )
        or _finite_float(result.get("s_tier_mean"), "s_tier_mean")
        < _finite_float(criteria.get("s_tier_mean_floor"), "s_tier_mean_floor")
        or _finite_float(
            result.get("minimum_opponent_wr"), "minimum_opponent_wr"
        )
        < _finite_float(
            criteria.get("individual_opponent_floor"),
            "individual_opponent_floor",
        )
        or (
            "s_plus_individual_floor" in criteria
            and _exact_int(
                result.get("s_plus_below_floor_count"),
                "s_plus_below_floor_count",
            )
            > _exact_int(
                criteria.get("s_plus_below_floor_allowance"),
                "s_plus_below_floor_allowance",
            )
        )
    ):
        raise ValueError("passed gate result does not satisfy contract thresholds")
    matchups = result.get("matchups")
    if not isinstance(matchups, list) or not all(
        isinstance(row, dict) for row in matchups
    ):
        raise ValueError("passed gate result has no matchup rows")
    by_id = {str(row.get("opponent_id") or ""): row for row in matchups}
    if (
        set(by_id) != set(roster_ids)
        or len(by_id) != len(matchups)
        or expected_total != expected_per * len(roster_ids)
        or _exact_int(result.get("games"), "result games") != expected_total
        or _exact_int(audit.get("valid_games"), "audit valid_games")
        != expected_total
        or _exact_int(audit.get("requested_games"), "audit requested_games")
        != expected_total
        or any(
            _exact_int(by_id[key].get("games"), "matchup games") != expected_per
            or _exact_int(by_id[key].get("seat0"), "matchup seat0")
            != expected_per // 2
            or _exact_int(by_id[key].get("seat1"), "matchup seat1")
            != expected_per // 2
            for key in roster_ids
        )
    ):
        raise ValueError("passed gate result is not the exact contracted roster")

    commit = json.loads(json.dumps(dict(commit_record)))
    iteration = _exact_int(result.get("iteration"), "result iteration")
    history = commit.get("history")
    history_row = (
        next(
            (
                row
                for row in reversed(history)
                if isinstance(row, dict)
                and row.get("iteration") == iteration
            ),
            None,
        )
        if isinstance(history, list)
        else None
    )
    committed_result = (
        history_row.get("active_gate_result")
        if isinstance(history_row, dict)
        and isinstance(history_row.get("active_gate_result"), dict)
        else None
    )
    result_core = {
        key: value
        for key, value in result.items()
        if key
        not in {"committed", "commit", "commit_digest", "created_at_utc"}
    }
    candidate = history_row.get("candidate") if isinstance(history_row, dict) else {}
    if (
        _exact_int(
            commit.get("last_completed_iteration"),
            "commit last_completed_iteration",
        )
        != iteration
        or _exact_int(commit.get("next_iteration"), "commit next_iteration")
        != iteration + 1
        or not isinstance(history_row, dict)
        or history_row.get("completed") is not True
        or history_row.get("promoted") is not True
        or not isinstance(candidate, dict)
        or str(candidate.get("digest") or "") != str(result["checkpoint_digest"])
        or committed_result != result_core
        or _canonical_digest(commit) != commit_digest
    ):
        raise ValueError("exact result is not bound to its immutable iteration commit")

    aliases = active.get("excluded_aliases") or []
    alias_ids = {str(row.get("opponent_id") or "") for row in aliases}
    if alias_ids & (set(roster_ids) | {row["opponent_id"] for row in current["controls"]}):
        raise ValueError("excluded package alias leaked into a canonical roster")

    prior = {
        str(row.get("gate_id") or ""): row for row in current["retirements"]
    }.get(gate_id)

    controls_by_id = {
        str(row["opponent_id"]): row for row in current["controls"]
    }
    controls_by_digest = {
        str(row["content_digest"]): row for row in current["controls"]
    }
    retired_control_ids = {
        str(opponent_id)
        for retirement in current["retirements"]
        for opponent_id in retirement.get("opponent_ids") or ()
    }
    new_roster: list[dict[str, Any]] = []
    for row in roster:
        opponent_id = str(row["opponent_id"])
        content_digest = str(row["content_digest"])
        id_match = controls_by_id.get(opponent_id)
        digest_match = controls_by_digest.get(content_digest)
        if id_match is None and digest_match is None:
            new_roster.append(row)
            continue
        if (
            id_match is None
            or digest_match is None
            or id_match is not digest_match
            or opponent_id not in retired_control_ids
            or not _is_digest(id_match.get("retired_exact_result_digest"))
            or not _is_digest(id_match.get("retired_checkpoint_digest"))
        ):
            raise ValueError("passed gate overlaps an existing research control/alias")

    if prior is not None:
        prior_ids = tuple(
            str(value) for value in prior.get("opponent_ids") or ()
        )
        if (
            str(prior.get("exact_result_digest") or "") != exact_result_digest
            or str(prior.get("checkpoint_digest") or "")
            != str(result["checkpoint_digest"])
            or _exact_int(prior.get("iteration"), "retirement iteration")
            != iteration
            or not prior_ids
            or not set(prior_ids).issubset(roster_ids)
            or new_roster
        ):
            raise ValueError("retired gate was replayed with a different exact result")
        return current

    # Gate revisions are additive. A later exact gate may contain opponents
    # already retired by an earlier committed gate plus newly admitted
    # opponents (for example, a frozen specialist). The old retirement proof
    # remains immutable; only the newly proven identities are appended under
    # the new gate's exact result.
    if not new_roster:
        return current

    timestamp = str(
        updated_at_utc
        or result.get("created_at_utc")
        or datetime.now(timezone.utc).isoformat()
    )
    controls = list(current["controls"])
    new_roster_ids = tuple(str(row["opponent_id"]) for row in new_roster)
    for row in new_roster:
        controls.append(
            {
                "opponent_id": str(row["opponent_id"]),
                "content_digest": str(row["content_digest"]),
                "source_gate_id": gate_id,
                "source": str(row.get("source") or ""),
                "archetype_id": str(row.get("archetype_id") or ""),
                "archetype_label": str(row.get("archetype_label") or ""),
                "retired_at_utc": timestamp,
                "retired_exact_result_digest": exact_result_digest,
                "retired_checkpoint_digest": str(result["checkpoint_digest"]),
                "gate_weight": 0.0,
                "included_in_gate_pass": False,
                "formal_eval": False,
                "training_eligible": False,
            }
        )
    updated = {
        **current,
        "version": int(current["version"]) + 1,
        "updated_at_utc": timestamp,
        "controls": controls,
        "retirements": [
            *list(current["retirements"]),
            {
                "gate_id": gate_id,
                "retired_at_utc": timestamp,
                "exact_result_digest": exact_result_digest,
                "checkpoint_digest": str(result["checkpoint_digest"]),
                "iteration": int(result.get("iteration", -1)),
                "opponent_ids": list(new_roster_ids),
            },
        ],
    }
    return validate_research_control_registry(updated)


def retire_passed_gate_file(
    *,
    registry_path: Path,
    gate_contract: Mapping[str, Any],
    exact_result_path: Path,
    commit_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically advance the registry from a committed exact-result file."""
    registry_path = Path(registry_path).expanduser().resolve()
    exact_result_path = Path(exact_result_path).expanduser().resolve()
    commit_path = Path(commit_path).expanduser().resolve()
    destination = Path(output_path or registry_path).expanduser().resolve()
    # Bind parsing and hashing to the same inode contents. The shared mutable
    # result pointer may be atomically replaced by a later iteration.
    result_bytes = exact_result_path.read_bytes()
    result = json.loads(result_bytes)
    if Path(str(result.get("commit") or "")).expanduser().resolve() != commit_path:
        raise ValueError("exact gate result references the wrong iteration commit")
    commit_bytes = commit_path.read_bytes()
    commit_record = json.loads(commit_bytes)
    commit_digest = _canonical_digest(commit_record)
    source = load_research_control_registry(registry_path)
    existing = None
    base = source
    if destination.is_file():
        existing = load_research_control_registry(destination)
        source_controls = {
            str(row["opponent_id"]): row for row in source["controls"]
        }
        existing_controls = {
            str(row["opponent_id"]): row for row in existing["controls"]
        }
        source_retirements = {
            str(row["gate_id"]): row for row in source["retirements"]
        }
        existing_retirements = {
            str(row["gate_id"]): row for row in existing["retirements"]
        }
        if (
            str(existing["registry_id"]) != str(source["registry_id"])
            or int(existing["version"]) < int(source["version"])
            or any(
                existing_controls.get(opponent_id) != row
                for opponent_id, row in source_controls.items()
            )
            or any(
                existing_retirements.get(gate_id) != row
                for gate_id, row in source_retirements.items()
            )
        ):
            raise ValueError(
                "research-control destination does not extend its source registry"
            )
        base = existing
    updated = retire_passed_gate(
        registry=base,
        gate_contract=gate_contract,
        exact_result=result,
        exact_result_digest=_bytes_digest(result_bytes),
        commit_record=commit_record,
        commit_digest=commit_digest,
    )
    payload = json.dumps(updated, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        if existing == updated:
            return existing
        if int(existing["version"]) >= int(updated["version"]):
            raise ValueError("research-control registry would roll back or fork")
    _durable_atomic_write(destination, payload.encode("utf-8"))
    return updated
