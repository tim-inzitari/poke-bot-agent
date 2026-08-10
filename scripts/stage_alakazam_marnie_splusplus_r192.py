#!/usr/bin/env python3
"""Derive the non-deploying Alakazam r192 Marnie H10 gate artifacts.

Revision 192 adds the checksum-exact final-format H10 Marnie's Grimmsnarl ex
package as one *additional* strong-public/formal opponent.  The historical
Marnie specialist row is deliberately retained.  This utility only writes
new, derived JSON files supplied on its command line; it never changes a
selector, service, baseline package, or an input artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OWNER_REVISION = 192
DESIGN_SCHEMA = "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1"
FROZEN_SCHEMA = "poke_bot.frozen_specialist_registry/v1"
GATE_SCHEMA = "poke_bot.competition_gate_program/v1"
RUNTIME_REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
R182_SCHEMA = "poke_bot.alakazam_public_multi_env_safety_r182/v1"
PIN_FLOORS_SCHEMA = "poke_bot.owner_public_mix_pin_floors/v1"
STAGE_SCHEMA = "poke_bot.alakazam_marnie_splusplus_r192_stage/v1"
PARENT_OWNER_REVISION = 175

HISTORICAL_MARNIE_OPPONENT_ID = (
    "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"
)
H10_MARNIE_OPPONENT_ID = "specialist-marnie-final-format-h10-f20efb20f5c3"
H10_MARNIE_BASELINE_DIR = "marnie-final-format-h10-f20efb20f5c3"
H10_MARNIE_ARCHETYPE_ID = "marnie-s-grimmsnarl-ex"
H10_MARNIE_CHECKPOINT_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
H10_MARNIE_CONTENT_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
H10_MARNIE_TIER = "S++"
H10_MARNIE_WEIGHT = 4.0
FORMAL_GAMES_PER_OPPONENT = 250
FORMAL_GAMES_PER_SEAT = 125
STRONG_PUBLIC_PRACTICE_FLOOR_GAMES = 1024
STRONG_PUBLIC_PRACTICE_GAMES_TOTAL = 4586
PUBLIC_MIX_GAMES_TOTAL = 7172
DIVERSE_PUBLIC_GAMES_TOTAL = 2586
ORDINARY_STRONG_PUBLIC_MINIMUM_SHARE = 0.04
ORDINARY_STRONG_PUBLIC_MINIMUM_GAMES = int(
    STRONG_PUBLIC_PRACTICE_GAMES_TOTAL * ORDINARY_STRONG_PUBLIC_MINIMUM_SHARE
)
MINIMUM_TERMINAL_ITERATION = 5
BOUNDARY_PAUSE_SECONDS = 30
ADAPTIVE_MIN_SHARE_FLAG = "--official-adaptive-min-share"
ADAPTIVE_MIN_SHARE = "0.04"
CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG = "--allow-clean-boundary-design-migration"
NO_CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG = (
    "--no-allow-clean-boundary-design-migration"
)
BOUNDARY_DESIGN_MIGRATION_REASON_FLAG = "--boundary-design-migration-reason"
R192_BOUNDARY_DESIGN_MIGRATION_REASON = (
    "owner_r192_marnie_splusplus_post_iteration5_receipt_backed_migration"
)
UNARMED_BOUNDARY = (
    "receipt_backed_inactive_boundary_or_trainer_owned_fence_enabled_"
    "clean_pause_after_completed_iteration5"
)
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
R192_TRANSPORT = {
    "r182_default_deny_unchanged": True,
    "other_r182_pairs_unchanged": True,
    "prior_pack4_eligible_group": "diverse_public",
    "activation_training_group": "strong_public_practice",
    "dispatch_mode": "singleton_remote_play",
    "pack4_attested_for_activation_group": False,
    "separate_exact_group_retention_attestation_required_for_pack4": True,
}
CRUSTLE_H10_OPPONENT_ID = "specialist-crustle-final-format-h10-7efd8d4113e7"
CRUSTLE_H10_CHECKPOINT_DIGEST = (
    "sha256:7efd8d4113e736d28576bdbfa1c9d1c3f3a7cf1a31a0b3cfadd1e7f82cf08955"
)
CRUSTLE_H10_CONTENT_DIGEST = (
    "sha256:359e3b4fed00502e58be4631576501b6f63523226ec92f2d75446df085b19afa"
)
CRUSTLE_H10_FLOOR_GAMES = 512
REQUIRED_DEPLOYMENT_INPUT_LABELS = frozenset(
    {
        "launch_pure_rl",
        "train_pure_rl",
        "launch_active_specialist",
        # These remain inactive future-activation inputs.  Binding their
        # candidate-clone bytes does not arm them; the current source still
        # requires a proven inactive boundary or trainer-owned fence.
        "activation_controller",
        "dropin_template",
        "boundary_service_template",
        "stop_budget_template",
        "strong_public_gate",
        "public_multi_env_safety",
        "r182_transport_contract",
        "baseline_manifest",
        "h10_marnie_model_provenance",
    }
)
ACTIVATION_ARTIFACT_DEPLOYMENT_LABELS = {
    "controller": "activation_controller",
    "dropin_template": "dropin_template",
    "boundary_service_template": "boundary_service_template",
    "stop_budget_template": "stop_budget_template",
}
STOP_BUDGET_GUARD = {
    "target_filename": "61-marnie-splusplus-r192-stop-budget.conf",
    "timeout_stop_seconds": 8,
    "timeout_stop_usec": 8_000_000,
    "kill_mode": "control-group",
    "send_sigkill": True,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_json_file_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _json_file_text(value: dict[str, Any]) -> str:
    """Return the exact JSON bytes emitted by ``_atomic_json`` as text."""

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _json_file_digest(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        _json_file_text(value).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _runtime_child_reference(
    candidate_runtime_root: Path,
    *,
    artifact_path: Path,
    requested_reference: str | None,
    label: str,
) -> str:
    """Return a candidate-runtime-root-contained reference for a staged artifact."""

    runtime_root = candidate_runtime_root.expanduser().resolve()
    if not runtime_root.is_absolute() or not runtime_root.is_dir():
        raise RuntimeError("r192 candidate runtime_root must be an existing absolute directory")
    expected = artifact_path.expanduser().resolve()
    try:
        expected.relative_to(runtime_root)
    except ValueError as exc:
        raise RuntimeError(
            f"r192 staged {label} must live under runtime_root: {runtime_root}"
        ) from exc
    reference = requested_reference or str(expected.relative_to(runtime_root))
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = runtime_root / candidate
    try:
        candidate.resolve().relative_to(runtime_root)
    except ValueError as exc:
        raise RuntimeError(
            f"r192 {label} reference escapes runtime_root: {reference}"
        ) from exc
    if candidate.resolve() != expected:
        raise RuntimeError(
            f"r192 {label} reference does not bind its staged output: {reference}"
        )
    return reference


def _resolve_source_runtime_artifacts(
    registry_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Resolve the immutable r175 parent artifacts from its own runtime root."""

    source_path = registry_path.expanduser().resolve()
    registry = _read_json(source_path)
    if registry.get("schema") != RUNTIME_REGISTRY_SCHEMA:
        raise RuntimeError("source runtime registry schema mismatch")
    raw_root = str(registry.get("runtime_root") or "").strip()
    root = Path(raw_root).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError("source runtime registry must declare an existing absolute runtime_root")
    root = root.resolve()

    def resolve(field: str) -> Path:
        raw = str(registry.get(field) or "").strip()
        if not raw:
            raise RuntimeError(f"source runtime registry lacks {field}")
        value = Path(raw).expanduser()
        path = value.resolve() if value.is_absolute() else (root / value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"source runtime registry {field} escapes runtime_root"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"source runtime registry {field} is missing: {path}")
        return path

    return registry, root, resolve("active_gate_contract"), resolve(
        "frozen_specialist_registry"
    )


def _assert_digest(value: object, *, label: str) -> str:
    digest = str(value or "")
    if not DIGEST_RE.fullmatch(digest):
        raise RuntimeError(f"{label} must be a canonical sha256 digest")
    return digest


def _rows(payload: dict[str, Any], *, key: str, label: str) -> list[dict[str, Any]]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise RuntimeError(f"{label} must be an array of JSON objects")
    return [copy.deepcopy(row) for row in raw]


def _set_unique_valued_arg(
    values: object,
    *,
    option: str,
    value: str,
) -> list[str]:
    """Set one repeatable ``--option value`` pair without allowing aliases."""

    if not isinstance(values, list):
        raise RuntimeError("common_trainer_args must be an array")
    result = [str(item) for item in values]
    matches: list[int] = []
    index = 0
    while index < len(result):
        if result[index] == option:
            if index + 1 >= len(result):
                raise RuntimeError(f"{option} lacks a value")
            matches.append(index)
            index += 2
        else:
            index += 1
    if len(matches) > 1:
        raise RuntimeError(f"{option} must occur no more than once")
    if matches:
        result[matches[0] + 1] = value
    else:
        result.extend((option, value))
    return result


def _unique_valued_arg(values: object, *, option: str) -> str:
    if not isinstance(values, list):
        raise RuntimeError("common_trainer_args must be an array")
    rows = [str(item) for item in values]
    found: list[str] = []
    index = 0
    while index < len(rows):
        if rows[index] == option:
            if index + 1 >= len(rows):
                raise RuntimeError(f"{option} lacks a value")
            found.append(rows[index + 1])
            index += 2
        else:
            index += 1
    if len(found) != 1:
        raise RuntimeError(f"{option} must occur exactly once, found {len(found)}")
    return found[0]


def _set_exact_r192_boundary_migration_args(values: object) -> list[str]:
    """Replace any inherited migration authority with r192's one-shot grant.

    A prior registry can legitimately contain a clean-boundary migration
    reason for an unrelated design change.  Carrying it into r192 would make
    the later receipt ambiguous, so strip both Boolean spellings and every
    reason spelling before appending r192's exact flag/reason pair.
    """

    if not isinstance(values, list):
        raise RuntimeError("common_trainer_args must be an array")
    rows = [str(value) for value in values]
    result: list[str] = []
    index = 0
    while index < len(rows):
        value = rows[index]
        if value in {
            CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG,
            NO_CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG,
        } or value.startswith(CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG + "=") or value.startswith(
            NO_CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG + "="
        ):
            index += 1
            continue
        if value == BOUNDARY_DESIGN_MIGRATION_REASON_FLAG:
            if index + 1 >= len(rows):
                raise RuntimeError(
                    f"{BOUNDARY_DESIGN_MIGRATION_REASON_FLAG} lacks a value"
                )
            index += 2
            continue
        if value.startswith(BOUNDARY_DESIGN_MIGRATION_REASON_FLAG + "="):
            index += 1
            continue
        result.append(value)
        index += 1
    result.extend(
        (
            CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG,
            BOUNDARY_DESIGN_MIGRATION_REASON_FLAG,
            R192_BOUNDARY_DESIGN_MIGRATION_REASON,
        )
    )
    _require_exact_r192_boundary_migration_args(result)
    return result


def _require_exact_r192_boundary_migration_args(values: object) -> None:
    """Fail closed unless the candidate has precisely r192's migration grant."""

    if not isinstance(values, list):
        raise RuntimeError("common_trainer_args must be an array")
    rows = [str(value) for value in values]
    flag_aliases = [
        value
        for value in rows
        if value == CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG
        or value == NO_CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG
        or value.startswith(CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG + "=")
        or value.startswith(NO_CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG + "=")
    ]
    if flag_aliases != [CLEAN_BOUNDARY_DESIGN_MIGRATION_FLAG]:
        raise RuntimeError(
            "r192 runtime must contain exactly one --allow-clean-boundary-design-migration"
        )
    if any(
        value.startswith(BOUNDARY_DESIGN_MIGRATION_REASON_FLAG + "=")
        for value in rows
    ):
        raise RuntimeError(
            "r192 runtime must use the separate boundary migration reason argument"
        )
    if (
        _unique_valued_arg(
            rows, option=BOUNDARY_DESIGN_MIGRATION_REASON_FLAG
        )
        != R192_BOUNDARY_DESIGN_MIGRATION_REASON
    ):
        raise RuntimeError("r192 runtime has the wrong boundary migration reason")


def _assert_unique_identities(rows: list[dict[str, Any]], *, label: str) -> None:
    ids: set[str] = set()
    digests: set[str] = set()
    for row in rows:
        opponent_id = str(row.get("opponent_id") or "")
        content_digest = _assert_digest(
            row.get("content_digest"), label=f"{label} content digest"
        )
        if not opponent_id or opponent_id in ids:
            raise RuntimeError(f"{label} has a duplicate or empty opponent id")
        if content_digest in digests:
            raise RuntimeError(
                f"{label} has duplicate package content digest {content_digest}"
            )
        ids.add(opponent_id)
        digests.add(content_digest)


def _row_by_id(rows: list[dict[str, Any]], opponent_id: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("opponent_id") == opponent_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {opponent_id!r} row, found {len(matches)}"
        )
    return matches[0]


def _load_design(path: Path) -> dict[str, Any]:
    design = _read_json(path)
    if design.get("schema") != DESIGN_SCHEMA:
        raise RuntimeError("r192 design schema mismatch")
    if int(design.get("owner_decision_revision") or 0) != OWNER_REVISION:
        raise RuntimeError("r192 owner revision mismatch")
    if design.get("specialist_id") != "alakazam":
        raise RuntimeError("r192 design must be scoped to Alakazam")
    opponent = dict(design.get("opponent") or {})
    expected = {
        "opponent_id": H10_MARNIE_OPPONENT_ID,
        "archetype_id": H10_MARNIE_ARCHETYPE_ID,
        "checkpoint_sha256": H10_MARNIE_CHECKPOINT_DIGEST,
        "content_digest": H10_MARNIE_CONTENT_DIGEST,
        "tier": H10_MARNIE_TIER,
        "weight": H10_MARNIE_WEIGHT,
    }
    for key, expected_value in expected.items():
        if opponent.get(key) != expected_value:
            raise RuntimeError(f"r192 design opponent {key} mismatch")
    if opponent.get("floor_games_per_set") != STRONG_PUBLIC_PRACTICE_FLOOR_GAMES:
        raise RuntimeError("r192 design H10 Marnie floor mismatch")
    historical = dict(design.get("historical_marnie") or {})
    if (
        historical.get("opponent_id") != HISTORICAL_MARNIE_OPPONENT_ID
        or historical.get("must_remain_distinct") is not True
        or historical.get("may_be_collapsed_or_substituted") is not False
    ):
        raise RuntimeError("r192 design must preserve the historical Marnie row")
    collection = dict(design.get("collection_contract") or {})
    expected_collection = {
        "games_per_iteration": 8196,
        "self_play_mirrors": 1024,
        "public_mix_games": PUBLIC_MIX_GAMES_TOTAL,
        "strong_public_practice_games": STRONG_PUBLIC_PRACTICE_GAMES_TOTAL,
        "diverse_public_games": DIVERSE_PUBLIC_GAMES_TOTAL,
        "ordinary_strong_public_minimum_share": (
            ORDINARY_STRONG_PUBLIC_MINIMUM_SHARE
        ),
        "h10_floor_enforcement": "exact_active_gate_strong_public_practice_floor",
        "legacy_diverse_public_h10_pin_removed_on_activation": True,
        "exact_total_unchanged": True,
        "public_replacement_lanes": 32,
    }
    for key, expected_value in expected_collection.items():
        if collection.get(key) != expected_value:
            raise RuntimeError(f"r192 collection contract {key} mismatch")
    if dict(design.get("transport") or {}) != R192_TRANSPORT:
        raise RuntimeError("r192 transport contract changed")
    if (
        STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
        + 17 * ORDINARY_STRONG_PUBLIC_MINIMUM_GAMES
        > STRONG_PUBLIC_PRACTICE_GAMES_TOTAL
    ):
        raise RuntimeError("r192 strong-public floor plan is infeasible")
    activation = dict(design.get("activation") or {})
    expected_activation = {
        "boundary": UNARMED_BOUNDARY,
        "first_guaranteed_activation_boundary_completed_iteration": (
            MINIMUM_TERMINAL_ITERATION
        ),
        "boundary_pause_seconds": BOUNDARY_PAUSE_SECONDS,
        "allow_clean_boundary_design_migration": True,
        "boundary_design_migration_reason": R192_BOUNDARY_DESIGN_MIGRATION_REASON,
        "managed_restart_during_verified_post_iteration5_hard_pause_allowed": False,
        "automatic_managed_restart_armed": False,
        "trainer_owned_handoff_fence_required": True,
        "current_r175_source_has_trainer_owned_handoff_fence": False,
        "proven_inactive_receipt_boundary_alternative_required": True,
        "training_restart_before_validation_allowed": False,
        "interrupt_active_collection_allowed": False,
    }
    for key, expected_value in expected_activation.items():
        if activation.get(key) != expected_value:
            raise RuntimeError(f"r192 activation contract {key} mismatch")
    return design


def _validate_r182_transport_contract(path: Path) -> None:
    contract = _read_json(path.expanduser().resolve())
    if (
        contract.get("schema") != R182_SCHEMA
        or int(contract.get("owner_decision_revision") or -1) != 182
        or contract.get("default_singleton") is not True
    ):
        raise RuntimeError("r182 transport contract identity changed")
    groups = dict(contract.get("safe_public_group_ids") or {})
    diverse = list(groups.get("diverse_public") or ())
    strong = list(groups.get("strong_public_practice") or ())
    rows = _rows(contract, key="safe_public_pairs", label="r182 safe public pairs")
    h10 = [row for row in rows if row.get("opponent_id") == H10_MARNIE_OPPONENT_ID]
    if (
        h10
        != [
            {
                "opponent_id": H10_MARNIE_OPPONENT_ID,
                "content_digest": H10_MARNIE_CONTENT_DIGEST,
            }
        ]
        or diverse.count(H10_MARNIE_OPPONENT_ID) != 1
        or H10_MARNIE_OPPONENT_ID in strong
        or H10_MARNIE_OPPONENT_ID
        in list(contract.get("legacy_singleton_opponent_ids") or ())
    ):
        raise RuntimeError("r182 H10 transport eligibility drifted")


def _new_gate_id(source_id: str) -> str:
    if not source_id:
        raise RuntimeError("source gate has no active id")
    return source_id + "+marnie-h10-splusplus-r192"


def _h10_gate_row() -> dict[str, Any]:
    return {
        "archetype_id": H10_MARNIE_ARCHETYPE_ID,
        "archetype_label": "Frozen final-format H10 Marnie's Grimmsnarl ex refresh",
        "content_digest": H10_MARNIE_CONTENT_DIGEST,
        "frozen_checkpoint_digest": H10_MARNIE_CHECKPOINT_DIGEST,
        "frozen_specialist": True,
        "opponent_id": H10_MARNIE_OPPONENT_ID,
        "owner_decision_revision": OWNER_REVISION,
        "source": (
            "checksum-exact final-format H10 Marnie's Grimmsnarl ex "
            "checkpoint f20efb20f5c3; distinct from historical Marnie"
        ),
        # This is a public-mix strong-practice constraint, not a formal-gate
        # allocation.  Formal evaluation remains exactly 250/125/125.
        "strong_public_practice_floor_games": STRONG_PUBLIC_PRACTICE_FLOOR_GAMES,
        "tier": H10_MARNIE_TIER,
        "weight": H10_MARNIE_WEIGHT,
    }


def _h10_frozen_row() -> dict[str, Any]:
    return {
        "archetype_id": H10_MARNIE_ARCHETYPE_ID,
        "archetype_label": "Frozen final-format H10 Marnie's Grimmsnarl ex refresh",
        "baseline_dir": H10_MARNIE_BASELINE_DIR,
        "baseline_group": "specialists",
        "checkpoint_digest": H10_MARNIE_CHECKPOINT_DIGEST,
        "content_digest": H10_MARNIE_CONTENT_DIGEST,
        "final_format_h10_refresh": True,
        "frozen": True,
        "historical_marnie_row_retained": True,
        "kaggle_submission_eligible": False,
        "matchup_runtime_enabled": True,
        "matchup_runtime_inference_only": True,
        "opponent_id": H10_MARNIE_OPPONENT_ID,
        "owner_decision_revision": OWNER_REVISION,
        "premium_holdout_tier": H10_MARNIE_TIER,
        "premium_holdout_weight": H10_MARNIE_WEIGHT,
        "public_mix_eligible": True,
        "research_eligible": False,
        "source": (
            "owner-designated checksum-exact final-format H10 Marnie's "
            "Grimmsnarl ex package; distinct from historical frozen Marnie"
        ),
        "source_passing_checkpoint_digest": H10_MARNIE_CHECKPOINT_DIGEST,
        "specialist_id": H10_MARNIE_ARCHETYPE_ID,
        "strong_public_practice_floor_games": STRONG_PUBLIC_PRACTICE_FLOOR_GAMES,
        "tier": H10_MARNIE_TIER,
        "weight": H10_MARNIE_WEIGHT,
    }


def build_gate(source: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Return the exact r192 18-opponent formal/strong-practice gate."""

    if source.get("schema") != GATE_SCHEMA:
        raise RuntimeError("source gate schema mismatch")
    source_id = str(source.get("active_gate_id") or "")
    next_gate = dict(source.get("next_gate") or {})
    roster = _rows(next_gate, key="roster", label="source formal roster")
    _assert_unique_identities(roster, label="source formal roster")
    if len(roster) != 17:
        raise RuntimeError(f"r192 requires the 17-opponent parent, got {len(roster)}")
    source_evaluation = dict(next_gate.get("evaluation") or {})
    if int(source_evaluation.get("games_total") or 0) != 4250:
        raise RuntimeError("r192 requires the 4,250-game formal parent")
    historical_before = copy.deepcopy(
        _row_by_id(roster, HISTORICAL_MARNIE_OPPONENT_ID)
    )
    if (
        historical_before.get("tier") != "S+"
        or float(historical_before.get("weight") or 0) != 2.0
    ):
        raise RuntimeError("r192 requires the unretiered historical Marnie S+/2 row")
    if any(row.get("opponent_id") == H10_MARNIE_OPPONENT_ID for row in roster):
        raise RuntimeError("r192 H10 Marnie row already exists in the source gate")
    if any(
        row.get("content_digest") == H10_MARNIE_CONTENT_DIGEST for row in roster
    ):
        raise RuntimeError("r192 H10 Marnie content digest already exists in source gate")

    gate = copy.deepcopy(source)
    next_gate = copy.deepcopy(next_gate)
    roster.append(_h10_gate_row())
    _assert_unique_identities(roster, label="r192 formal roster")
    if len(roster) != 18:
        raise RuntimeError("r192 formal roster must contain exactly 18 opponents")
    if _row_by_id(roster, HISTORICAL_MARNIE_OPPONENT_ID) != historical_before:
        raise RuntimeError("r192 must not rewrite the historical Marnie gate row")

    gate_id = _new_gate_id(source_id)
    evaluation = dict(source_evaluation)
    evaluation.update(
        {
            "games_total": FORMAL_GAMES_PER_OPPONENT * len(roster),
            "games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
            "minimum_games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
            "seat0_games_per_opponent": FORMAL_GAMES_PER_SEAT,
            "seat1_games_per_opponent": FORMAL_GAMES_PER_SEAT,
            "package_digest_deduplicated": True,
        }
    )
    next_gate.update(
        {
            "id": gate_id,
            "label": (
                str(next_gate.get("label") or "Strong public/frozen gate")
                + " + final-format H10 Marnie S++ (revision 192)"
            ),
            "evaluation": evaluation,
            "roster": roster,
        }
    )
    semantics = dict(gate.get("active_gate_semantics") or {})
    # Keep r100's existing S+/2 frozen and S/2, A/1 public semantics verbatim.
    # The H10 package is a separately named exceptional row, not a retiering of
    # either the historical Marnie package or the rest of the existing roster.
    semantics.update(
        {
            "frozen_specialist_agents": 15,
            "gate_roster_size": 18,
            "gate_games_total": 4500,
            "invariant": (
                "The active premium gate contains every non-superseded external "
                "agent plus all 14 historical frozen completed specialists as S+ "
                "and the one checksum-exact additional H10 Marnie's Grimmsnarl ex "
                "specialist as S++/4.0. Each receives exactly 250 greedy games "
                "split 125/125 by seat. The original four baselines remain exact "
                "research controls only."
            ),
            # The launcher compares this block byte-for-byte. Keep formal
            # allocation and historical-row metadata outside it so the only
            # authorized exception is precisely the H10 package binding.
            "exact_additional_splusplus_specialist": {
                "opponent_id": H10_MARNIE_OPPONENT_ID,
                "checkpoint_digest": H10_MARNIE_CHECKPOINT_DIGEST,
                "content_digest": H10_MARNIE_CONTENT_DIGEST,
                "tier": H10_MARNIE_TIER,
                "weight": H10_MARNIE_WEIGHT,
                "strong_public_practice_floor_games": (
                    STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
                ),
            },
        }
    )
    gate.update(
        {
            "active_gate_id": gate_id,
            "active_gate_semantics": semantics,
            "next_gate": next_gate,
            "owner_decision_revision": OWNER_REVISION,
            "r192_stage": {
                "design_schema": design["schema"],
                "status": "staged_non_active",
                "source_gate_id": source_id,
                "historical_marnie_opponent_id": HISTORICAL_MARNIE_OPPONENT_ID,
                "added_opponent_id": H10_MARNIE_OPPONENT_ID,
                "parent_roster_size": 17,
                "new_roster_size": 18,
                "parent_formal_games": 4250,
                "new_formal_games": 4500,
                "content_digest_unique": True,
                "first_guaranteed_activation_boundary_completed_iteration": (
                    MINIMUM_TERMINAL_ITERATION
                ),
                "boundary_pause_seconds": BOUNDARY_PAUSE_SECONDS,
                "pre_minimum_terminal_iteration_activation_allowed": False,
                "interrupt_active_collection_allowed": False,
                "managed_restart_during_verified_post_iteration5_hard_pause_allowed": False,
                "automatic_managed_restart_armed": False,
                "trainer_owned_handoff_fence_required": True,
                "current_r175_source_has_trainer_owned_handoff_fence": False,
            },
        }
    )
    return gate


def build_frozen_registry(source: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Return the additive 15-row frozen registry, without mutating its parent."""

    if source.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError("source frozen registry schema mismatch")
    rows = _rows(source, key="specialists", label="source frozen registry")
    _assert_unique_identities(rows, label="source frozen registry")
    if len(rows) != 14:
        raise RuntimeError(f"r192 requires the 14-row frozen parent, got {len(rows)}")
    historical_before = copy.deepcopy(
        _row_by_id(rows, HISTORICAL_MARNIE_OPPONENT_ID)
    )
    if any(row.get("opponent_id") == H10_MARNIE_OPPONENT_ID for row in rows):
        raise RuntimeError("r192 H10 Marnie row already exists in frozen registry")
    if any(
        row.get("content_digest") == H10_MARNIE_CONTENT_DIGEST for row in rows
    ):
        raise RuntimeError(
            "r192 H10 Marnie content digest already exists in frozen registry"
        )

    result = copy.deepcopy(source)
    rows.append(_h10_frozen_row())
    _assert_unique_identities(rows, label="r192 frozen registry")
    if len(rows) != 15:
        raise RuntimeError("r192 frozen registry must contain exactly 15 rows")
    if _row_by_id(rows, HISTORICAL_MARNIE_OPPONENT_ID) != historical_before:
        raise RuntimeError("r192 must not rewrite the historical Marnie frozen row")
    result.update(
        {
            "specialists": rows,
            "version": int(source.get("version") or 0) + 1,
            "owner_decision_revision": OWNER_REVISION,
            "scope": "alakazam-r192-marnie-h10-splusplus-staged",
            "r192_stage": {
                "design_schema": design["schema"],
                "status": "staged_not_activated",
                "historical_marnie_opponent_id": HISTORICAL_MARNIE_OPPONENT_ID,
                "added_opponent_id": H10_MARNIE_OPPONENT_ID,
                "parent_row_count": 14,
                "new_row_count": 15,
                "content_digest_unique": True,
            },
        }
    )
    return result


def build_pin_sidecar(source: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Remove only r175's legacy H10 diverse-public floor pin.

    The H10 package remains mandatory, but revision 192 moves its executable
    1,024-game floor into the checksum-bound strong-public gate row.  Leaving
    the old sidecar row would schedule the same package through two conflicting
    provenance paths.  All other sidecar rows, notably the retained Crustle
    pin, are copied byte-for-byte at the row level.
    """

    if source.get("schema") != PIN_FLOORS_SCHEMA:
        raise RuntimeError("source public-mix pin sidecar schema mismatch")
    pins = _rows(source, key="pins", label="source public-mix pin sidecar")
    package_ids = [str(row.get("package_id") or "") for row in pins]
    if not all(package_ids) or len(package_ids) != len(set(package_ids)):
        raise RuntimeError("source public-mix pin sidecar has ambiguous package IDs")
    legacy_h10 = _row_by_id(
        [
            {"opponent_id": row.get("package_id"), **row}
            for row in pins
        ],
        H10_MARNIE_OPPONENT_ID,
    )
    if (
        legacy_h10.get("checkpoint_sha256") != H10_MARNIE_CHECKPOINT_DIGEST
        or int(legacy_h10.get("floor_games_per_set") or -1)
        != STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
        or legacy_h10.get("archetype_id") != H10_MARNIE_ARCHETYPE_ID
    ):
        raise RuntimeError("r175 H10 Marnie sidecar pin is not checksum/floor exact")
    crustle = _row_by_id(
        [
            {"opponent_id": row.get("package_id"), **row}
            for row in pins
        ],
        CRUSTLE_H10_OPPONENT_ID,
    )
    if (
        crustle.get("checkpoint_sha256") != CRUSTLE_H10_CHECKPOINT_DIGEST
        or crustle.get("content_digest") != CRUSTLE_H10_CONTENT_DIGEST
        or int(crustle.get("floor_games_per_set") or -1) != CRUSTLE_H10_FLOOR_GAMES
    ):
        raise RuntimeError("r175 Crustle sidecar pin is not checksum/floor exact")

    retained = [
        row for row in pins if row.get("package_id") != H10_MARNIE_OPPONENT_ID
    ]
    if len(retained) != len(pins) - 1:
        raise RuntimeError("r192 must remove exactly one legacy H10 Marnie pin")
    if any(row.get("package_id") == H10_MARNIE_OPPONENT_ID for row in retained):
        raise RuntimeError("r192 retains an executable legacy H10 Marnie pin")
    if _row_by_id(
        [{"opponent_id": row.get("package_id"), **row} for row in retained],
        CRUSTLE_H10_OPPONENT_ID,
    ) != crustle:
        raise RuntimeError("r192 must not rewrite the retained Crustle pin")

    result = copy.deepcopy(source)
    result.update(
        {
            "owner_decision_revision": OWNER_REVISION,
            "pins": retained,
            "r192_stage": {
                "design_schema": design["schema"],
                "status": "staged_not_activated",
                "removed_legacy_diverse_public_package_id": H10_MARNIE_OPPONENT_ID,
                "h10_executable_floor_owner": (
                    "exact_active_gate_strong_public_practice_floor"
                ),
                "retained_crustle_package_id": CRUSTLE_H10_OPPONENT_ID,
            },
        }
    )
    return result


def validate_pin_sidecar(pin_sidecar: dict[str, Any], design: dict[str, Any]) -> None:
    """Validate r192's no-legacy-alias sidecar result."""

    if pin_sidecar.get("schema") != PIN_FLOORS_SCHEMA:
        raise RuntimeError("r192 pin sidecar schema mismatch")
    pins = _rows(pin_sidecar, key="pins", label="r192 public-mix pin sidecar")
    package_ids = [str(row.get("package_id") or "") for row in pins]
    if not all(package_ids) or len(package_ids) != len(set(package_ids)):
        raise RuntimeError("r192 public-mix pin sidecar has ambiguous package IDs")
    if H10_MARNIE_OPPONENT_ID in package_ids:
        raise RuntimeError("r192 pin sidecar must not contain legacy H10 Marnie")
    crustle = _row_by_id(
        [{"opponent_id": row.get("package_id"), **row} for row in pins],
        CRUSTLE_H10_OPPONENT_ID,
    )
    if (
        crustle.get("checkpoint_sha256") != CRUSTLE_H10_CHECKPOINT_DIGEST
        or crustle.get("content_digest") != CRUSTLE_H10_CONTENT_DIGEST
        or int(crustle.get("floor_games_per_set") or -1) != CRUSTLE_H10_FLOOR_GAMES
    ):
        raise RuntimeError("r192 pin sidecar lost the exact Crustle pin")
    stage = dict(pin_sidecar.get("r192_stage") or {})
    if (
        pin_sidecar.get("owner_decision_revision") != OWNER_REVISION
        or stage.get("design_schema") != design.get("schema")
        or stage.get("removed_legacy_diverse_public_package_id")
        != H10_MARNIE_OPPONENT_ID
        or stage.get("h10_executable_floor_owner")
        != "exact_active_gate_strong_public_practice_floor"
    ):
        raise RuntimeError("r192 pin sidecar provenance binding changed")


def build_runtime_registry(
    source: dict[str, Any],
    *,
    gate_reference: str,
    frozen_reference: str,
    gate_id: str,
    gate_sha256: str,
    frozen_sha256: str,
    owner_contract_reference: str | None = None,
    owner_contract_sha256: str | None = None,
    pin_sidecar_reference: str | None = None,
    pin_sidecar_sha256: str | None = None,
    candidate_runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Rebind a *derived copy* of an Alakazam runtime registry to r192 artifacts."""

    if (
        source.get("schema") != RUNTIME_REGISTRY_SCHEMA
        or int(source.get("owner_decision_revision") or -1)
        != PARENT_OWNER_REVISION
    ):
        raise RuntimeError("source runtime registry is not the r175 parent")
    if not gate_reference or not frozen_reference or not gate_id:
        raise RuntimeError("derived runtime registry requires non-empty artifact references")
    _assert_digest(gate_sha256, label="derived gate digest")
    _assert_digest(frozen_sha256, label="derived frozen registry digest")
    full_binding_values = (
        owner_contract_reference,
        owner_contract_sha256,
        pin_sidecar_reference,
        pin_sidecar_sha256,
    )
    if any(value is not None for value in full_binding_values) and not all(
        full_binding_values
    ):
        raise RuntimeError(
            "r192 runtime artifact bindings require owner contract and pin sidecar"
        )
    if owner_contract_sha256 is not None:
        _assert_digest(owner_contract_sha256, label="owner contract digest")
        _assert_digest(pin_sidecar_sha256, label="derived pin sidecar digest")
    if candidate_runtime_root is not None:
        candidate_runtime_root = candidate_runtime_root.expanduser().resolve()
        if not candidate_runtime_root.is_absolute() or not candidate_runtime_root.is_dir():
            raise RuntimeError("candidate runtime_root must be an existing absolute directory")
    result = copy.deepcopy(source)
    isolated = dict(result.get("isolated_refresh_contract") or {})
    specialists = dict(result.get("specialists") or {})
    alakazam = dict(specialists.get("alakazam") or {})
    pin = dict(alakazam.get("owner_grimmsnarl_pin") or {})
    if (
        not isolated
        or not alakazam
        or not pin
        or int(result.get("minimum_terminal_iteration") or -1)
        != MINIMUM_TERMINAL_ITERATION
        or int(alakazam.get("minimum_terminal_iteration") or -1)
        != MINIMUM_TERMINAL_ITERATION
    ):
        raise RuntimeError("source runtime registry is not the exact r175 Alakazam parent")
    common_trainer_args = _set_unique_valued_arg(
        result.get("common_trainer_args"),
        option=ADAPTIVE_MIN_SHARE_FLAG,
        value=ADAPTIVE_MIN_SHARE,
    )
    common_trainer_args = _set_exact_r192_boundary_migration_args(
        common_trainer_args
    )
    _require_exact_r192_boundary_migration_args(common_trainer_args)
    isolated.update(
        {
            "grimmsnarl_package_id": H10_MARNIE_OPPONENT_ID,
            "grimmsnarl_checkpoint_sha256": H10_MARNIE_CHECKPOINT_DIGEST,
            "grimmsnarl_content_digest": H10_MARNIE_CONTENT_DIGEST,
            "grimmsnarl_floor_per_set": STRONG_PUBLIC_PRACTICE_FLOOR_GAMES,
            "grimmsnarl_tier": H10_MARNIE_TIER,
            "grimmsnarl_weight": H10_MARNIE_WEIGHT,
        }
    )
    pin.update(
        {
            "package_id": H10_MARNIE_OPPONENT_ID,
            "checkpoint_sha256": H10_MARNIE_CHECKPOINT_DIGEST,
            "content_digest": H10_MARNIE_CONTENT_DIGEST,
            "floor_games_per_set": STRONG_PUBLIC_PRACTICE_FLOOR_GAMES,
            "tier": H10_MARNIE_TIER,
            "weight": H10_MARNIE_WEIGHT,
            # r175's sidecar used this identity under diverse_public.  r192
            # moves the floor into the checksum-bound active gate; the
            # registry pin remains provenance only and must never recreate a
            # second executable diverse-public floor.
            "enforcement_source": (
                "exact_active_gate_strong_public_practice_floor"
            ),
            "legacy_diverse_public_sidecar_pin_active": False,
            "superseded_by_exact_strong_gate_floor": True,
        }
    )
    alakazam["owner_grimmsnarl_pin"] = pin
    specialists["alakazam"] = alakazam
    result.update(
        {
            "active_gate_contract": gate_reference,
            "frozen_specialist_registry": frozen_reference,
            "terminal_active_gate_id": gate_id,
            "owner_decision_revision": OWNER_REVISION,
            **(
                {"runtime_root": str(candidate_runtime_root)}
                if candidate_runtime_root is not None
                else {}
            ),
            "common_trainer_args": common_trainer_args,
            "isolated_refresh_contract": isolated,
            "specialists": specialists,
            "r192_stage": {
                "status": "staged_not_activated",
                "selector_changed": False,
                "service_changed": False,
                "historical_marnie_opponent_id": HISTORICAL_MARNIE_OPPONENT_ID,
                "added_opponent_id": H10_MARNIE_OPPONENT_ID,
                "ordinary_strong_public_minimum_share": (
                    ORDINARY_STRONG_PUBLIC_MINIMUM_SHARE
                ),
                "h10_executable_floor_owner": (
                    "exact_active_gate_strong_public_practice_floor"
                ),
                "legacy_diverse_public_h10_pin_active": False,
                "first_guaranteed_activation_boundary_completed_iteration": (
                    MINIMUM_TERMINAL_ITERATION
                ),
                "boundary_pause_seconds": BOUNDARY_PAUSE_SECONDS,
                "managed_restart_during_verified_post_iteration5_hard_pause_allowed": False,
                "automatic_managed_restart_armed": False,
                "trainer_owned_handoff_fence_required": True,
                "current_r175_source_has_trainer_owned_handoff_fence": False,
                "interrupt_active_collection_allowed": False,
                "boundary_design_migration": {
                    "allow_clean_boundary_design_migration": True,
                    "reason": R192_BOUNDARY_DESIGN_MIGRATION_REASON,
                },
                "stop_budget_guard": copy.deepcopy(STOP_BUDGET_GUARD),
                "artifact_bindings": {
                    "gate_path": gate_reference,
                    "gate_sha256": gate_sha256,
                    "frozen_registry_path": frozen_reference,
                    "frozen_registry_sha256": frozen_sha256,
                },
            },
        }
    )
    if owner_contract_reference is not None:
        result["r192_stage"]["artifact_bindings"] = {
            "owner_contract": {
                "path": owner_contract_reference,
                "sha256": owner_contract_sha256,
            },
            "active_gate_contract": {
                "path": gate_reference,
                "sha256": gate_sha256,
            },
            "frozen_specialist_registry": {
                "path": frozen_reference,
                "sha256": frozen_sha256,
            },
            "public_mix_pin_floors": {
                "path": pin_sidecar_reference,
                "sha256": pin_sidecar_sha256,
            },
            # The independent identity validator consumes these established
            # flat aliases; the controller consumes the four nested entries.
            "gate_path": gate_reference,
            "gate_sha256": gate_sha256,
            "frozen_registry_path": frozen_reference,
            "frozen_registry_sha256": frozen_sha256,
        }
    return result


def validate_artifacts(
    *, gate: dict[str, Any], frozen: dict[str, Any], design: dict[str, Any]
) -> None:
    """Fail closed unless the exact additive r192 relationship holds."""

    if gate.get("schema") != GATE_SCHEMA or frozen.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError("r192 derived artifact schema mismatch")
    roster = _rows(dict(gate.get("next_gate") or {}), key="roster", label="r192 roster")
    frozen_rows = _rows(frozen, key="specialists", label="r192 frozen registry")
    _assert_unique_identities(roster, label="r192 formal roster")
    _assert_unique_identities(frozen_rows, label="r192 frozen registry")
    if len(roster) != 18 or len(frozen_rows) != 15:
        raise RuntimeError("r192 derived roster cardinality mismatch")
    evaluation = dict((gate.get("next_gate") or {}).get("evaluation") or {})
    expected_evaluation = {
        "games_total": 4500,
        "games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
        "minimum_games_per_opponent": FORMAL_GAMES_PER_OPPONENT,
        "seat0_games_per_opponent": FORMAL_GAMES_PER_SEAT,
        "seat1_games_per_opponent": FORMAL_GAMES_PER_SEAT,
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise RuntimeError(f"r192 formal evaluation {key} mismatch")
    gate_h10 = _row_by_id(roster, H10_MARNIE_OPPONENT_ID)
    frozen_h10 = _row_by_id(frozen_rows, H10_MARNIE_OPPONENT_ID)
    for row, label in ((gate_h10, "gate"), (frozen_h10, "frozen")):
        if row.get("content_digest") != H10_MARNIE_CONTENT_DIGEST:
            raise RuntimeError(f"r192 {label} H10 Marnie content digest mismatch")
    if (
        gate_h10.get("tier") != H10_MARNIE_TIER
        or float(gate_h10.get("weight") or 0) != H10_MARNIE_WEIGHT
        or gate_h10.get("frozen_checkpoint_digest") != H10_MARNIE_CHECKPOINT_DIGEST
    ):
        raise RuntimeError("r192 H10 Marnie gate tier/checkpoint mismatch")
    if (
        frozen_h10.get("checkpoint_digest") != H10_MARNIE_CHECKPOINT_DIGEST
        or frozen_h10.get("public_mix_eligible") is not True
        or frozen_h10.get("research_eligible") is not False
        or frozen_h10.get("final_format_h10_refresh") is not True
        or frozen_h10.get("premium_holdout_tier") != H10_MARNIE_TIER
        or float(frozen_h10.get("premium_holdout_weight") or 0)
        != H10_MARNIE_WEIGHT
        or frozen_h10.get("tier") != H10_MARNIE_TIER
        or float(frozen_h10.get("weight") or 0) != H10_MARNIE_WEIGHT
        or frozen_h10.get("strong_public_practice_floor_games")
        != STRONG_PUBLIC_PRACTICE_FLOOR_GAMES
    ):
        raise RuntimeError("r192 H10 Marnie frozen registry row is invalid")
    expected_semantics = {
        "opponent_id": H10_MARNIE_OPPONENT_ID,
        "checkpoint_digest": H10_MARNIE_CHECKPOINT_DIGEST,
        "content_digest": H10_MARNIE_CONTENT_DIGEST,
        "tier": H10_MARNIE_TIER,
        "weight": H10_MARNIE_WEIGHT,
        "strong_public_practice_floor_games": STRONG_PUBLIC_PRACTICE_FLOOR_GAMES,
    }
    if dict(gate.get("active_gate_semantics") or {}).get(
        "exact_additional_splusplus_specialist"
    ) != expected_semantics:
        raise RuntimeError("r192 gate S++ semantic binding is invalid")
    semantics = dict(gate.get("active_gate_semantics") or {})
    if (
        semantics.get("frozen_specialist_agents") != 15
        or semantics.get("frozen_specialist_tier") != "S+"
        or "14 historical frozen completed specialists as S+" not in str(
            semantics.get("invariant") or ""
        )
        or "S++/4.0" not in str(semantics.get("invariant") or "")
        or "opponent_tier_policy" in semantics
    ):
        raise RuntimeError("r192 gate did not preserve r100-plus-one semantics")
    historical_gate = _row_by_id(roster, HISTORICAL_MARNIE_OPPONENT_ID)
    if (
        historical_gate.get("tier") != "S+"
        or float(historical_gate.get("weight") or 0) != 2.0
    ):
        raise RuntimeError("r192 gate retiered historical Marnie")
    for rows, label in ((roster, "gate"), (frozen_rows, "frozen")):
        historical = _row_by_id(rows, HISTORICAL_MARNIE_OPPONENT_ID)
        if historical.get("content_digest") == H10_MARNIE_CONTENT_DIGEST:
            raise RuntimeError(f"r192 {label} collapsed historical Marnie into H10")
    if gate.get("r192_stage", {}).get("design_schema") != design.get("schema"):
        raise RuntimeError("r192 gate no longer binds the design source")
    if frozen.get("r192_stage", {}).get("design_schema") != design.get("schema"):
        raise RuntimeError("r192 frozen registry no longer binds the design source")


def _receipt_artifact(path: Path, *, label: str) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"r192 {label} artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": _file_digest(resolved)}


def build_stage_receipt(
    *,
    design_path: Path,
    source_runtime_registry: Path,
    source_gate: Path,
    source_frozen: Path,
    source_pin_sidecar: Path,
    runtime_registry: Path,
    gate: Path,
    frozen: Path,
    pin_sidecar: Path,
    deployment_inputs: dict[str, Path],
) -> dict[str, Any]:
    """Build the inactive, checksum-bound handoff consumed by r192 preflight.

    It contains no selector, service, or remote-operation instruction.  The
    later boundary controller only reads this receipt and independently
    re-hashes every bound file before it can consider a managed restart.
    """

    missing_labels = sorted(
        REQUIRED_DEPLOYMENT_INPUT_LABELS - set(deployment_inputs)
    )
    if missing_labels:
        raise RuntimeError(
            "r192 stage receipt deployment inputs missing: "
            + ", ".join(missing_labels)
        )
    if any(not label or not isinstance(path, Path) for label, path in deployment_inputs.items()):
        raise RuntimeError("r192 stage receipt deployment inputs are invalid")
    runtime_payload = _read_json(runtime_registry)
    _require_exact_r192_boundary_migration_args(
        runtime_payload.get("common_trainer_args")
    )
    runtime_stage = dict(runtime_payload.get("r192_stage") or {})
    if runtime_stage.get("boundary_design_migration") != {
        "allow_clean_boundary_design_migration": True,
        "reason": R192_BOUNDARY_DESIGN_MIGRATION_REASON,
    }:
        raise RuntimeError("r192 runtime registry lacks exact migration provenance")
    if runtime_stage.get("stop_budget_guard") != STOP_BUDGET_GUARD:
        raise RuntimeError("r192 runtime registry lacks exact stop-budget guard")
    receipt: dict[str, Any] = {
        "schema": STAGE_SCHEMA,
        # This exact inactive value is consumed by the boundary controller.
        # It describes a derived receipt only; it is not an activation.
        "status": "staged_non_active",
        "owner_decision_revision": OWNER_REVISION,
        "active_before_receipt_backed_activation": False,
        "training_interrupted": False,
        "interrupt_active_collection_allowed": False,
        "managed_restart_during_verified_post_iteration5_hard_pause_allowed": False,
        "automatic_managed_restart_armed": False,
        "trainer_owned_handoff_fence_required": True,
        "current_r175_source_has_trainer_owned_handoff_fence": False,
        "first_guaranteed_activation_boundary_completed_iteration": (
            MINIMUM_TERMINAL_ITERATION
        ),
        "boundary_pause_seconds": BOUNDARY_PAUSE_SECONDS,
        "boundary_design_migration": {
            "allow_clean_boundary_design_migration": True,
            "reason": R192_BOUNDARY_DESIGN_MIGRATION_REASON,
        },
        "stop_budget_guard": copy.deepcopy(STOP_BUDGET_GUARD),
        "owner_contract": _receipt_artifact(design_path, label="owner contract"),
        "source_runtime_registry": _receipt_artifact(
            source_runtime_registry, label="source runtime registry"
        ),
        "source_artifacts": {
            "active_gate_contract": _receipt_artifact(
                source_gate, label="source active gate"
            ),
            "frozen_specialist_registry": _receipt_artifact(
                source_frozen, label="source frozen specialist registry"
            ),
        },
        "source_public_mix_pin_floors": _receipt_artifact(
            source_pin_sidecar, label="source public-mix pin sidecar"
        ),
        "staged_artifacts": {
            "runtime_registry": _receipt_artifact(
                runtime_registry, label="runtime registry"
            ),
            "active_gate_contract": _receipt_artifact(gate, label="active gate"),
            "frozen_specialist_registry": _receipt_artifact(
                frozen, label="frozen specialist registry"
            ),
            "public_mix_pin_floors": _receipt_artifact(
                pin_sidecar, label="public-mix pin sidecar"
            ),
        },
        "activation_artifacts": {
            key: _receipt_artifact(
                deployment_inputs[label], label=f"activation artifact {key}"
            )
            for key, label in ACTIVATION_ARTIFACT_DEPLOYMENT_LABELS.items()
        },
        "deployment_inputs": [
            {
                "label": label,
                **_receipt_artifact(path, label=f"deployment input {label}"),
            }
            for label, path in sorted(deployment_inputs.items())
        ],
        # Exact allocation/ownership metadata is checked by the boundary
        # controller as well as by the typed owner contract.
        "collection_contract": {
            "games_per_iteration": 8196,
            "self_play_mirrors": 1024,
            "public_mix_games": PUBLIC_MIX_GAMES_TOTAL,
            "strong_public_practice_games": STRONG_PUBLIC_PRACTICE_GAMES_TOTAL,
            "diverse_public_games": DIVERSE_PUBLIC_GAMES_TOTAL,
            "ordinary_strong_public_minimum_share": (
                ORDINARY_STRONG_PUBLIC_MINIMUM_SHARE
            ),
            "h10_executable_floor_owner": (
                "exact_active_gate_strong_public_practice_floor"
            ),
            "legacy_diverse_public_h10_pin_removed_on_activation": True,
        },
        "transport": copy.deepcopy(R192_TRANSPORT),
        "selector_or_service_changed": False,
        "remote_deployment_performed": False,
    }
    receipt["receipt_payload_digest"] = _canonical_digest(receipt)
    return receipt


def _same_json(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _canonical_digest(left) == _canonical_digest(right)


def _is_exact_staged_json(path: Path, value: dict[str, Any]) -> bool:
    """Whether ``path`` has the deterministic bytes bound by the registry.

    The candidate runtime registry carries SHA-256 values for these files, so
    accepting merely semantically equivalent JSON would make an existing,
    differently formatted file fail later receipt/controller validation.
    """

    return path.is_file() and _file_digest(path) == _json_file_digest(value)


def _write_or_verify(path: Path, value: dict[str, Any]) -> None:
    """Create a staged artifact once, or accept only identical bytes later."""

    target = path.expanduser().resolve()
    if target.exists():
        if not _is_exact_staged_json(target, value):
            raise RuntimeError(
                f"refusing to replace a non-identical staged artifact: {target}"
            )
        return
    _atomic_json(target, value)


def _candidate_output_path(
    candidate_root: Path,
    raw: Path | None,
    default_relative: str,
    *,
    label: str,
) -> Path:
    candidate_root = candidate_root.expanduser().resolve()
    value = raw.expanduser() if raw is not None else Path(default_relative)
    # Preserve the lexical candidate-root path long enough to reject any
    # symlink component.  Resolving first alone would permit an in-root
    # symlink to redirect a receipt or artifact into a surprising location.
    lexical_path = Path(os.path.abspath(str(value))) if value.is_absolute() else (
        candidate_root / value
    )
    try:
        relative = lexical_path.relative_to(candidate_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must live under candidate runtime_root") from exc
    component = candidate_root
    for part in relative.parts:
        component = component / part
        if component.is_symlink():
            raise RuntimeError(f"{label} may not traverse a symbolic link")
    path = lexical_path.resolve()
    try:
        path.relative_to(candidate_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must live under candidate runtime_root") from exc
    return path


def _source_path_from_root(source_root: Path, raw: Path) -> Path:
    value = raw.expanduser()
    return value.resolve() if value.is_absolute() else (source_root / value).resolve()


def _validate_stop_budget_template(path: Path) -> None:
    """Require the temporary r192 guard's exact bounded-stop semantics."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"r192 stop-budget template is unreadable: {path}") from exc
    settings: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings.setdefault(key, []).append(value)
    expected = {
        "TimeoutStopSec": "8s",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
    }
    if any(settings.get(key) != [value] for key, value in expected.items()):
        raise RuntimeError(
            "r192 stop-budget template must contain exactly one 8s/control-group/SIGKILL guard"
        )


def _deployment_input_paths(
    *,
    candidate_root: Path,
    r182_contract: Path,
    values: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "launch_pure_rl": candidate_root / "scripts" / "launch_pure_rl.py",
        "train_pure_rl": candidate_root / "scripts" / "train_pure_rl.py",
        "launch_active_specialist": candidate_root
        / "scripts"
        / "launch_active_specialist.py",
        "activation_controller": candidate_root
        / "scripts"
        / "activate_alakazam_marnie_splusplus_r192.py",
        "dropin_template": candidate_root
        / "deploy"
        / "systemd"
        / "pokebot-final-format-alakazam-rtp-r175-rl.service.d"
        / "62-marnie-splusplus-r192.conf.in",
        "boundary_service_template": candidate_root
        / "deploy"
        / "systemd"
        / "pokebot-final-format-alakazam-rtp-r175-marnie-splusplus-r192-boundary.service.in",
        "stop_budget_template": candidate_root
        / "deploy"
        / "systemd"
        / "pokebot-final-format-alakazam-rtp-r175-rl.service.d"
        / "61-marnie-splusplus-r192-stop-budget.conf.in",
        "strong_public_gate": candidate_root
        / "poke_bot"
        / "pure_rl"
        / "strong_public_gate.py",
        "public_multi_env_safety": candidate_root
        / "poke_bot"
        / "public_multi_env_safety.py",
        "r182_transport_contract": r182_contract.expanduser().resolve(),
        "baseline_manifest": candidate_root / "baselines" / "manifest.json",
        "h10_marnie_model_provenance": candidate_root
        / "baselines"
        / "specialists"
        / H10_MARNIE_BASELINE_DIR
        / "model.pt",
    }
    for raw in values:
        label, separator, raw_path = str(raw).partition("=")
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", label)
            or not raw_path
        ):
            raise RuntimeError("--deployment-input must be LABEL=PATH")
        value = Path(raw_path).expanduser()
        paths[label] = value.resolve() if value.is_absolute() else (candidate_root / value).resolve()
    missing = sorted(REQUIRED_DEPLOYMENT_INPUT_LABELS - set(paths))
    if missing:
        raise RuntimeError("missing required deployment inputs: " + ", ".join(missing))
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"deployment input {label} is missing: {path}")
    for label in ACTIVATION_ARTIFACT_DEPLOYMENT_LABELS.values():
        path = paths[label].resolve()
        try:
            path.relative_to(candidate_root)
        except ValueError as exc:
            raise RuntimeError(
                f"activation deployment input {label} must live under candidate runtime_root"
            ) from exc
    _validate_stop_budget_template(paths["stop_budget_template"])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT / "state/alakazam-marnie-splusplus-opponent-r192.json",
    )
    parser.add_argument(
        "--r182-contract",
        type=Path,
        default=ROOT / "state/alakazam-public-multi-env-split-r182.json",
    )
    parser.add_argument("--source-runtime-registry", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Existing candidate clone root; it must differ from the live r175 root.",
    )
    parser.add_argument(
        "--source-gate",
        type=Path,
        help="Optional assertion against the gate resolved from source runtime_root.",
    )
    parser.add_argument(
        "--source-frozen",
        type=Path,
        help="Optional assertion against the frozen registry resolved from source runtime_root.",
    )
    parser.add_argument("--source-pin-sidecar", type=Path, required=True)
    parser.add_argument("--out-gate", type=Path)
    parser.add_argument("--out-frozen", type=Path)
    parser.add_argument("--out-runtime-registry", type=Path)
    parser.add_argument("--out-pin-sidecar", type=Path)
    parser.add_argument("--gate-reference")
    parser.add_argument("--frozen-reference")
    parser.add_argument("--pin-sidecar-reference")
    parser.add_argument(
        "--out-stage-receipt",
        type=Path,
    )
    parser.add_argument(
        "--deployment-input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Optional extra or replacement checksum-bound candidate input."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing outputs against the deterministic derivation without writing.",
    )
    args = parser.parse_args(argv)

    design_path = args.design.expanduser().resolve()
    r182_path = args.r182_contract.expanduser().resolve()
    design = _load_design(design_path)
    _validate_r182_transport_contract(r182_path)
    source_runtime, source_root, source_gate_path, source_frozen_path = (
        _resolve_source_runtime_artifacts(args.source_runtime_registry)
    )
    candidate_root = args.runtime_root.expanduser().resolve()
    if not candidate_root.is_absolute() or not candidate_root.is_dir():
        raise RuntimeError("--runtime-root must be an existing absolute candidate clone")
    if candidate_root == source_root:
        raise RuntimeError("r192 candidate runtime_root must not be the live r175 root")

    for raw, expected, label in (
        (args.source_gate, source_gate_path, "--source-gate"),
        (args.source_frozen, source_frozen_path, "--source-frozen"),
    ):
        if raw is not None and _source_path_from_root(source_root, raw) != expected:
            raise RuntimeError(f"{label} differs from the source runtime registry")
    source_pin_path = _source_path_from_root(source_root, args.source_pin_sidecar)
    if not source_pin_path.is_file():
        raise RuntimeError(f"source pin sidecar is missing: {source_pin_path}")

    out_gate_path = _candidate_output_path(
        candidate_root,
        args.out_gate,
        "runtime/final_format_alakazam_gate_r192_marnie_splusplus.json",
        label="--out-gate",
    )
    out_frozen_path = _candidate_output_path(
        candidate_root,
        args.out_frozen,
        "ops/frozen_specialist_registry_alakazam_r192_marnie_splusplus.json",
        label="--out-frozen",
    )
    out_runtime_path = _candidate_output_path(
        candidate_root,
        args.out_runtime_registry,
        "runtime/specialist_runtime_registry_h10_r175_marnie_splusplus_r192.json",
        label="--out-runtime-registry",
    )
    out_pin_path = _candidate_output_path(
        candidate_root,
        args.out_pin_sidecar,
        "runtime/owner_public_mix_pin_floors_r192.json",
        label="--out-pin-sidecar",
    )
    out_receipt_path = _candidate_output_path(
        candidate_root,
        args.out_stage_receipt,
        # ``outputs`` is often a checkout-local symlink.  The persistent
        # receipt must instead live in the candidate runtime clone itself.
        "runtime/alakazam-marnie-splusplus-r192-stage.json",
        label="--out-stage-receipt",
    )
    output_paths = {
        out_gate_path,
        out_frozen_path,
        out_runtime_path,
        out_pin_path,
        out_receipt_path,
    }
    if len(output_paths) != 5:
        raise RuntimeError("r192 candidate output paths must be distinct")

    gate = build_gate(_read_json(source_gate_path), design)
    frozen = build_frozen_registry(_read_json(source_frozen_path), design)
    pin_sidecar = build_pin_sidecar(_read_json(source_pin_path), design)
    validate_artifacts(gate=gate, frozen=frozen, design=design)
    validate_pin_sidecar(pin_sidecar, design)

    gate_reference = _runtime_child_reference(
        candidate_root,
        artifact_path=out_gate_path,
        requested_reference=args.gate_reference,
        label="active gate",
    )
    frozen_reference = _runtime_child_reference(
        candidate_root,
        artifact_path=out_frozen_path,
        requested_reference=args.frozen_reference,
        label="frozen registry",
    )
    pin_reference = _runtime_child_reference(
        candidate_root,
        artifact_path=out_pin_path,
        requested_reference=args.pin_sidecar_reference,
        label="public-mix pin sidecar",
    )
    derived_registry = build_runtime_registry(
        source_runtime,
        gate_reference=gate_reference,
        frozen_reference=frozen_reference,
        gate_id=str(gate["active_gate_id"]),
        gate_sha256=_json_file_digest(gate),
        frozen_sha256=_json_file_digest(frozen),
        owner_contract_reference=str(design_path),
        owner_contract_sha256=_file_digest(design_path),
        pin_sidecar_reference=pin_reference,
        pin_sidecar_sha256=_json_file_digest(pin_sidecar),
        candidate_runtime_root=candidate_root,
    )
    if _unique_valued_arg(
        derived_registry.get("common_trainer_args"),
        option=ADAPTIVE_MIN_SHARE_FLAG,
    ) != ADAPTIVE_MIN_SHARE:
        raise RuntimeError("candidate runtime registry did not set adaptive min share 0.04")

    expected_outputs = (
        (out_gate_path, gate, "gate"),
        (out_frozen_path, frozen, "frozen registry"),
        (out_pin_path, pin_sidecar, "cleaned pin sidecar"),
        (out_runtime_path, derived_registry, "runtime registry"),
    )
    if args.check:
        for path, expected, label in expected_outputs:
            if not _is_exact_staged_json(path, expected):
                raise RuntimeError(f"derived r192 {label} differs from deterministic result")
    else:
        for path, value, _ in expected_outputs:
            _write_or_verify(path, value)

    deployment_inputs = _deployment_input_paths(
        candidate_root=candidate_root,
        r182_contract=r182_path,
        values=list(args.deployment_input),
    )
    stage_receipt = build_stage_receipt(
        design_path=design_path,
        source_runtime_registry=args.source_runtime_registry.expanduser().resolve(),
        source_gate=source_gate_path,
        source_frozen=source_frozen_path,
        source_pin_sidecar=source_pin_path,
        runtime_registry=out_runtime_path,
        gate=out_gate_path,
        frozen=out_frozen_path,
        pin_sidecar=out_pin_path,
        deployment_inputs=deployment_inputs,
    )
    if args.check:
        if not _is_exact_staged_json(out_receipt_path, stage_receipt):
            raise RuntimeError("r192 persisted stage receipt differs from deterministic result")
    else:
        _write_or_verify(out_receipt_path, stage_receipt)

    print(
        json.dumps(
            {
                "status": "validated_non_active_stage" if args.check else "staged_non_active",
                "stage_receipt": _receipt_artifact(
                    out_receipt_path, label="persisted stage receipt"
                ),
                "active_gate_id": gate["active_gate_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
