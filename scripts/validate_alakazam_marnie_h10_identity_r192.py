#!/usr/bin/env python3
"""Read-only provenance check for Alakazam's additional r192 H10 Marnie.

This checker intentionally has no staging, selector, service, installation,
or network behaviour.  It verifies the exact installed package bytes and the
two distinct Marnie identities before a separately prepared r192 candidate can
be promoted at a receipt-backed boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.baselines_runtime import baseline_content_digest  # noqa: E402


DESIGN_SCHEMA = "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1"
FROZEN_SCHEMA = "poke_bot.frozen_specialist_registry/v1"
GATE_SCHEMA = "poke_bot.competition_gate_program/v1"
R182_SCHEMA = "poke_bot.alakazam_public_multi_env_safety_r182/v1"
RUNTIME_REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"
PIN_SIDECAR_SCHEMA = "poke_bot.owner_public_mix_pin_floors/v1"

H10_OPPONENT_ID = "specialist-marnie-final-format-h10-f20efb20f5c3"
H10_BASELINE_DIR = "marnie-final-format-h10-f20efb20f5c3"
H10_ARCHETYPE_ID = "marnie-s-grimmsnarl-ex"
H10_CHECKPOINT_DIGEST = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
H10_CONTENT_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
H10_TIER = "S++"
H10_WEIGHT = 4.0
H10_FLOOR_GAMES = 1024
R192_BOUNDARY_DESIGN_MIGRATION_REASON = (
    "owner_r192_marnie_splusplus_post_iteration5_receipt_backed_migration"
)

STAGED_STATUS = (
    "staged_candidate_not_armed_pending_trainer_owned_fence_or_"
    "proven_inactive_boundary"
)
ACTIVATION_BOUNDARY = (
    "receipt_backed_inactive_boundary_or_trainer_owned_fence_enabled_"
    "clean_pause_after_completed_iteration5"
)
EXPECTED_TRANSPORT = {
    "r182_default_deny_unchanged": True,
    "other_r182_pairs_unchanged": True,
    "prior_pack4_eligible_group": "diverse_public",
    "activation_training_group": "strong_public_practice",
    "dispatch_mode": "singleton_remote_play",
    "pack4_attested_for_activation_group": False,
    "separate_exact_group_retention_attestation_required_for_pack4": True,
}

HISTORICAL_OPPONENT_ID = "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"
HISTORICAL_CHECKPOINT_DIGEST = (
    "sha256:52a5207e4c98dce80b49b6403cbb17f14d6fc4d2ac5b625532020a1a25f233ac"
)
HISTORICAL_CONTENT_DIGEST = (
    "sha256:ae9f3c31e2705a955aa1c51b79fbeffcab0d93dfe65da23572b18b9a52d8e8f6"
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _exactly_one(
    rows: list[dict[str, Any]], opponent_id: str, *, label: str
) -> dict[str, Any]:
    matches = [row for row in rows if row.get("opponent_id") == opponent_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one {opponent_id!r} row; "
            f"found {len(matches)}"
        )
    return dict(matches[0])


def _require_rows(payload: dict[str, Any], *, key: str, label: str) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{label} must contain an object array at {key!r}")
    return [dict(row) for row in rows]


def _validate_design(design: dict[str, Any]) -> None:
    opponent = dict(design.get("opponent") or {})
    expected_opponent = {
        "opponent_id": H10_OPPONENT_ID,
        "archetype_id": H10_ARCHETYPE_ID,
        "checkpoint_sha256": H10_CHECKPOINT_DIGEST,
        "content_digest": H10_CONTENT_DIGEST,
        "tier": H10_TIER,
        "weight": H10_WEIGHT,
        "floor_games_per_set": H10_FLOOR_GAMES,
        "distinct_additional_specialist_row": True,
        "duplicate_alias_row_allowed": False,
    }
    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("owner_decision_revision") != 192
        or design.get("status") != STAGED_STATUS
        or design.get("specialist_id") != "alakazam"
        or any(opponent.get(key) != value for key, value in expected_opponent.items())
    ):
        raise RuntimeError("r192 design does not bind the exact H10 Marnie identity")
    historical = dict(design.get("historical_marnie") or {})
    expected_historical = {
        "opponent_id": HISTORICAL_OPPONENT_ID,
        "must_remain_distinct": True,
        "may_be_collapsed_or_substituted": False,
    }
    if any(historical.get(key) != value for key, value in expected_historical.items()):
        raise RuntimeError("r192 design historical-Marnie retention contract changed")
    collection = dict(design.get("collection_contract") or {})
    if (
        collection.get("games_per_iteration") != 8196
        or collection.get("self_play_mirrors") != 1024
        or collection.get("public_mix_games") != 7172
        or collection.get("strong_public_practice_games") != 4586
        or collection.get("diverse_public_games") != 2586
        or (
            collection.get("strong_public_practice_games")
            + collection.get("diverse_public_games")
            != collection.get("public_mix_games")
        )
        or collection.get("ordinary_strong_public_minimum_share") != 0.04
        or collection.get("exact_total_unchanged") is not True
        or collection.get("public_replacement_lanes") != 32
    ):
        raise RuntimeError("r192 collection contract changed")
    if dict(design.get("transport") or {}) != EXPECTED_TRANSPORT:
        raise RuntimeError("r192 strong-public transport contract changed")
    activation = dict(design.get("activation") or {})
    if (
        activation.get("boundary") != ACTIVATION_BOUNDARY
        or activation.get("first_guaranteed_activation_boundary_completed_iteration")
        != 5
        or activation.get("boundary_pause_seconds") != 30
        or activation.get("allow_clean_boundary_design_migration") is not True
        or activation.get("boundary_design_migration_reason")
        != R192_BOUNDARY_DESIGN_MIGRATION_REASON
        or activation.get(
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        )
        is not False
        or activation.get("automatic_managed_restart_armed") is not False
        or activation.get("trainer_owned_handoff_fence_required") is not True
        or activation.get("current_r175_source_has_trainer_owned_handoff_fence")
        is not False
        or activation.get("proven_inactive_receipt_boundary_alternative_required")
        is not True
        or activation.get("requires_checksum_exact_roster_binding") is not True
        or activation.get("requires_runtime_registry_binding") is not True
        or activation.get("requires_dispatch_provenance_binding") is not True
        or activation.get("requires_focused_exact_retention_tests") is not True
        or activation.get("training_restart_before_validation_allowed") is not False
        or activation.get("interrupt_active_collection_allowed") is not False
    ):
        raise RuntimeError("r192 activation boundary contract changed")


def _validate_installed_package(
    *,
    baseline_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    package = baseline_dir.expanduser().resolve()
    if not package.is_dir() or package.name != H10_BASELINE_DIR:
        raise RuntimeError("H10 baseline directory identity is invalid")
    agents = _require_rows(manifest, key="agents", label="baseline manifest")
    rows = [row for row in agents if row.get("id") == H10_OPPONENT_ID]
    if len(rows) != 1:
        raise RuntimeError("baseline manifest must contain H10 Marnie exactly once")
    manifest_row = dict(rows[0])
    if (
        manifest_row.get("group") != "specialists"
        or manifest_row.get("dir") != H10_BASELINE_DIR
    ):
        raise RuntimeError("baseline manifest H10 Marnie group/directory changed")
    required = ("main.py", "model.pt", "deck.csv", "matchup_tree.json")
    missing = [name for name in required if not (package / name).is_file()]
    if missing:
        raise RuntimeError(f"H10 package is missing required files: {missing}")
    checkpoint_digest = _sha256(package / "model.pt")
    if checkpoint_digest != H10_CHECKPOINT_DIGEST:
        raise RuntimeError(
            "H10 package checkpoint digest mismatch: "
            f"expected={H10_CHECKPOINT_DIGEST} actual={checkpoint_digest}"
        )
    content_digest = baseline_content_digest(package)
    if content_digest != H10_CONTENT_DIGEST:
        raise RuntimeError(
            "H10 package content digest mismatch: "
            f"expected={H10_CONTENT_DIGEST} actual={content_digest}"
        )
    return {
        "path": str(package),
        "manifest_id": H10_OPPONENT_ID,
        "manifest_group": "specialists",
        "manifest_dir": H10_BASELINE_DIR,
        "checkpoint_digest": checkpoint_digest,
        "content_digest": content_digest,
        "required_files": list(required),
    }


def _validate_historical_frozen(frozen: dict[str, Any]) -> dict[str, Any]:
    if frozen.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError("historical frozen registry schema changed")
    rows = _require_rows(frozen, key="specialists", label="historical frozen registry")
    historical = _exactly_one(rows, HISTORICAL_OPPONENT_ID, label="historical frozen registry")
    if (
        historical.get("archetype_id") != H10_ARCHETYPE_ID
        or historical.get("checkpoint_digest") != HISTORICAL_CHECKPOINT_DIGEST
        or historical.get("content_digest") != HISTORICAL_CONTENT_DIGEST
        or historical.get("frozen") is not True
        or historical.get("public_mix_eligible") is not True
        or historical.get("research_eligible") is not False
    ):
        raise RuntimeError("historical Marnie frozen identity changed")
    if (
        HISTORICAL_OPPONENT_ID == H10_OPPONENT_ID
        or HISTORICAL_CHECKPOINT_DIGEST == H10_CHECKPOINT_DIGEST
        or HISTORICAL_CONTENT_DIGEST == H10_CONTENT_DIGEST
    ):
        raise RuntimeError("historical and H10 Marnie identities are not distinct")
    return historical


def _validate_r182(r182: dict[str, Any]) -> dict[str, Any]:
    if (
        r182.get("schema") != R182_SCHEMA
        or r182.get("owner_decision_revision") != 182
    ):
        raise RuntimeError("r182 transport contract changed")
    if (
        r182.get("default_singleton") is not True
        or set(r182.get("permitted_training_groups") or ())
        != {"diverse_public", "strong_public_practice"}
    ):
        raise RuntimeError("r182 singleton/default group contract changed")
    safe_pairs = _require_rows(r182, key="safe_public_pairs", label="r182 safe pairs")
    h10_pairs = [row for row in safe_pairs if row.get("opponent_id") == H10_OPPONENT_ID]
    if h10_pairs != [{"opponent_id": H10_OPPONENT_ID, "content_digest": H10_CONTENT_DIGEST}]:
        raise RuntimeError("r182 H10 Marnie allowlist binding changed")
    groups = dict(r182.get("safe_public_group_ids") or {})
    if set(groups) != {"diverse_public", "strong_public_practice"}:
        raise RuntimeError("r182 public-pack group map changed")
    diverse = list(groups.get("diverse_public") or [])
    strong_practice = list(groups.get("strong_public_practice") or [])
    legacy = list(r182.get("legacy_singleton_opponent_ids") or [])
    if (
        diverse.count(H10_OPPONENT_ID) != 1
        or H10_OPPONENT_ID in strong_practice
        or legacy.count(HISTORICAL_OPPONENT_ID) != 1
        or H10_OPPONENT_ID in legacy
    ):
        raise RuntimeError("r182 H10/historical Marnie transport distinction changed")
    return {
        "h10_transport": "allowlisted_diverse_public_only",
        "h10_strong_public_transport": "singleton_remote_play_pending_attestation",
        "h10_strong_public_practice_pack4_admitted": False,
        "historical_transport": "legacy_singleton",
        "h10_content_digest": H10_CONTENT_DIGEST,
    }


def _validate_candidate_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("schema") != GATE_SCHEMA:
        raise RuntimeError("candidate gate schema changed")
    gate = dict(candidate.get("next_gate") or {})
    rows = _require_rows(gate, key="roster", label="candidate gate")
    evaluation = dict(gate.get("evaluation") or {})
    if (
        len(rows) != 18
        or not str(gate.get("id") or "")
        or candidate.get("active_gate_id") != gate.get("id")
        or evaluation.get("games_total") != 4500
        or evaluation.get("games_per_opponent") != 250
        or evaluation.get("seat0_games_per_opponent") != 125
        or evaluation.get("seat1_games_per_opponent") != 125
    ):
        raise RuntimeError("candidate gate cardinality/evaluation changed")
    h10 = _exactly_one(rows, H10_OPPONENT_ID, label="candidate gate")
    historical = _exactly_one(rows, HISTORICAL_OPPONENT_ID, label="candidate gate")
    if (
        h10.get("archetype_id") != H10_ARCHETYPE_ID
        or h10.get("frozen_specialist") is not True
        or h10.get("frozen_checkpoint_digest") != H10_CHECKPOINT_DIGEST
        or h10.get("content_digest") != H10_CONTENT_DIGEST
        or h10.get("tier") != H10_TIER
        or h10.get("weight") != H10_WEIGHT
        or h10.get("strong_public_practice_floor_games") != H10_FLOOR_GAMES
    ):
        raise RuntimeError("candidate gate H10 Marnie binding changed")
    semantics = dict(candidate.get("active_gate_semantics") or {})
    scoped = dict(semantics.get("exact_additional_splusplus_specialist") or {})
    expected_scoped = {
        "opponent_id": H10_OPPONENT_ID,
        "checkpoint_digest": H10_CHECKPOINT_DIGEST,
        "content_digest": H10_CONTENT_DIGEST,
        "tier": H10_TIER,
        "weight": H10_WEIGHT,
        "strong_public_practice_floor_games": H10_FLOOR_GAMES,
    }
    if scoped != expected_scoped:
        raise RuntimeError("candidate gate H10 Marnie floor/semantics changed")
    stage = dict(candidate.get("r192_stage") or {})
    if (
        stage.get("historical_marnie_opponent_id") != HISTORICAL_OPPONENT_ID
        or stage.get("content_digest_unique") is not True
    ):
        raise RuntimeError("candidate gate does not retain the historical Marnie")
    if (
        historical.get("content_digest") != HISTORICAL_CONTENT_DIGEST
        or historical.get("frozen_checkpoint_digest") != HISTORICAL_CHECKPOINT_DIGEST
        or historical.get("content_digest") == h10.get("content_digest")
    ):
        raise RuntimeError("candidate gate collapses historical Marnie")
    ids = [str(row.get("opponent_id") or "") for row in rows]
    content_digests = [str(row.get("content_digest") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)) or len(content_digests) != len(set(content_digests)):
        raise RuntimeError("candidate gate has duplicate opponent/package identities")
    return {
        "gate_id": str(gate.get("id") or ""),
        "roster_size": len(rows),
        "evaluation": {
            "games_total": 4500,
            "games_per_opponent": 250,
            "seat0_games_per_opponent": 125,
            "seat1_games_per_opponent": 125,
        },
        "h10_strong_public_practice_floor_games": H10_FLOOR_GAMES,
    }


def _validate_candidate_frozen(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError("candidate frozen registry schema changed")
    rows = _require_rows(candidate, key="specialists", label="candidate frozen registry")
    if len(rows) != 15:
        raise RuntimeError("candidate frozen registry cardinality changed")
    h10 = _exactly_one(rows, H10_OPPONENT_ID, label="candidate frozen registry")
    historical = _exactly_one(rows, HISTORICAL_OPPONENT_ID, label="candidate frozen registry")
    if (
        h10.get("archetype_id") != H10_ARCHETYPE_ID
        or h10.get("baseline_dir") != H10_BASELINE_DIR
        or h10.get("baseline_group") != "specialists"
        or h10.get("checkpoint_digest") != H10_CHECKPOINT_DIGEST
        or h10.get("content_digest") != H10_CONTENT_DIGEST
        or h10.get("frozen") is not True
        or h10.get("public_mix_eligible") is not True
        or h10.get("research_eligible") is not False
        or h10.get("final_format_h10_refresh") is not True
        or h10.get("premium_holdout_tier") != H10_TIER
        or h10.get("premium_holdout_weight") != H10_WEIGHT
        or h10.get("tier") != H10_TIER
        or h10.get("weight") != H10_WEIGHT
        or h10.get("strong_public_practice_floor_games") != H10_FLOOR_GAMES
        or historical.get("checkpoint_digest") != HISTORICAL_CHECKPOINT_DIGEST
        or historical.get("content_digest") != HISTORICAL_CONTENT_DIGEST
    ):
        raise RuntimeError("candidate frozen registry Marnie binding changed")
    ids = [str(row.get("opponent_id") or "") for row in rows]
    content_digests = [str(row.get("content_digest") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)) or len(content_digests) != len(set(content_digests)):
        raise RuntimeError("candidate frozen registry has duplicate identities")
    return {"frozen_roster_size": len(rows)}


def _validate_candidate_pin_sidecar(candidate: dict[str, Any]) -> dict[str, Any]:
    """Reject legacy sidecar enforcement for the exact H10 gate floor."""

    if candidate.get("schema") != PIN_SIDECAR_SCHEMA:
        raise RuntimeError("candidate public-mix pin sidecar schema changed")
    pins = _require_rows(candidate, key="pins", label="candidate public-mix pin sidecar")
    h10_matches = [
        row
        for row in pins
        if any(
            row.get(key) == H10_OPPONENT_ID
            for key in ("package_id", "opponent_id", "id")
        )
    ]
    if h10_matches:
        raise RuntimeError(
            "candidate public-mix pin sidecar must exclude H10 Marnie; the exact "
            "strong-public gate row owns its 1024-game floor"
        )
    return {
        "pin_count": len(pins),
        "h10_sidecar_enforcement": False,
        "h10_executable_floor_owner": "exact_strong_public_gate_row",
    }


def _resolve_runtime_reference(
    *,
    registry_path: Path,
    registry: dict[str, Any],
    reference: object,
) -> Path:
    raw = str(reference or "").strip()
    if not raw:
        raise RuntimeError("candidate runtime registry has an empty artifact reference")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    runtime_root = str(registry.get("runtime_root") or "").strip()
    if runtime_root:
        root = Path(runtime_root).expanduser()
        if not root.is_absolute():
            root = registry_path.parent / root
        return (root / path).resolve()
    return (registry_path.parent / path).resolve()


def _validate_candidate_runtime_registry(
    candidate: dict[str, Any],
    *,
    registry_path: Path,
    candidate_gate_path: Path,
    candidate_frozen_path: Path,
    gate_id: str,
) -> dict[str, Any]:
    """Require the optional staged registry to bind both artifacts by path+SHA."""

    if (
        candidate.get("schema") != RUNTIME_REGISTRY_SCHEMA
        or candidate.get("owner_decision_revision") != 192
        or candidate.get("terminal_active_gate_id") != gate_id
    ):
        raise RuntimeError("candidate runtime registry identity changed")
    expected_gate = candidate_gate_path.expanduser().resolve()
    expected_frozen = candidate_frozen_path.expanduser().resolve()
    resolved_gate = _resolve_runtime_reference(
        registry_path=registry_path,
        registry=candidate,
        reference=candidate.get("active_gate_contract"),
    )
    resolved_frozen = _resolve_runtime_reference(
        registry_path=registry_path,
        registry=candidate,
        reference=candidate.get("frozen_specialist_registry"),
    )
    if resolved_gate != expected_gate or resolved_frozen != expected_frozen:
        raise RuntimeError("candidate runtime registry paths do not bind r192 artifacts")
    stage = dict(candidate.get("r192_stage") or {})
    bindings = dict(stage.get("artifact_bindings") or {})
    structured_gate = bindings.get("active_gate_contract")
    structured_frozen = bindings.get("frozen_specialist_registry")
    if isinstance(structured_gate, dict) or isinstance(structured_frozen, dict):
        if not isinstance(structured_gate, dict) or not isinstance(
            structured_frozen, dict
        ):
            raise RuntimeError("candidate runtime registry has partial r192 bindings")
        gate_reference = structured_gate.get("path")
        frozen_reference = structured_frozen.get("path")
        bound_gate_sha256 = structured_gate.get("sha256") or structured_gate.get(
            "digest"
        )
        bound_frozen_sha256 = structured_frozen.get(
            "sha256"
        ) or structured_frozen.get("digest")
    else:
        # Keep validating the minimal local derivation form, while the full
        # receipt-backed stage uses the structured records above.
        gate_reference = bindings.get("gate_path")
        frozen_reference = bindings.get("frozen_registry_path")
        bound_gate_sha256 = bindings.get("gate_sha256")
        bound_frozen_sha256 = bindings.get("frozen_registry_sha256")
    bound_gate = _resolve_runtime_reference(
        registry_path=registry_path,
        registry=candidate,
        reference=gate_reference,
    )
    bound_frozen = _resolve_runtime_reference(
        registry_path=registry_path,
        registry=candidate,
        reference=frozen_reference,
    )
    gate_sha256 = _sha256(expected_gate)
    frozen_sha256 = _sha256(expected_frozen)
    if (
        bound_gate != expected_gate
        or bound_frozen != expected_frozen
        or bound_gate_sha256 != gate_sha256
        or bound_frozen_sha256 != frozen_sha256
        or stage.get("selector_changed") is not False
        or stage.get("service_changed") is not False
    ):
        raise RuntimeError("candidate runtime registry checksum binding changed")
    trainer_args = [str(value) for value in candidate.get("common_trainer_args") or []]
    if (
        trainer_args.count("--allow-clean-boundary-design-migration") != 1
        or "--no-allow-clean-boundary-design-migration" in trainer_args
        or trainer_args.count("--boundary-design-migration-reason") != 1
    ):
        raise RuntimeError("candidate runtime registry lacks exactly one r192 migration grant")
    reason_index = trainer_args.index("--boundary-design-migration-reason")
    if (
        reason_index + 1 >= len(trainer_args)
        or trainer_args[reason_index + 1]
        != R192_BOUNDARY_DESIGN_MIGRATION_REASON
        or stage.get("boundary_design_migration")
        != {
            "allow_clean_boundary_design_migration": True,
            "reason": R192_BOUNDARY_DESIGN_MIGRATION_REASON,
        }
    ):
        raise RuntimeError("candidate runtime registry migration provenance changed")
    isolated = dict(candidate.get("isolated_refresh_contract") or {})
    specialists = dict(candidate.get("specialists") or {})
    alakazam = dict(specialists.get("alakazam") or {})
    pin = dict(alakazam.get("owner_grimmsnarl_pin") or {})
    for source, label in ((isolated, "isolated runtime"), (pin, "owner pin")):
        package = source.get("grimmsnarl_package_id", source.get("package_id"))
        checkpoint = source.get(
            "grimmsnarl_checkpoint_sha256", source.get("checkpoint_sha256")
        )
        content = source.get(
            "grimmsnarl_content_digest", source.get("content_digest")
        )
        floor = source.get(
            "grimmsnarl_floor_per_set", source.get("floor_games_per_set")
        )
        tier = source.get("grimmsnarl_tier", source.get("tier"))
        weight = source.get("grimmsnarl_weight", source.get("weight"))
        if (
            package != H10_OPPONENT_ID
            or checkpoint != H10_CHECKPOINT_DIGEST
            or content != H10_CONTENT_DIGEST
            or floor != H10_FLOOR_GAMES
            or tier != H10_TIER
            or weight != H10_WEIGHT
        ):
            raise RuntimeError(
                f"candidate runtime registry {label} lost its exact S++ binding"
            )
    if (
        pin.get("enforcement_source")
        != "exact_active_gate_strong_public_practice_floor"
        or pin.get("legacy_diverse_public_sidecar_pin_active") is not False
        or pin.get("superseded_by_exact_strong_gate_floor") is not True
    ):
        raise RuntimeError(
            "candidate runtime owner_grimmsnarl_pin must mark the exact "
            "strong-public gate row as the sole executable H10 floor"
        )
    if alakazam.get("minimum_terminal_iteration") != 5:
        raise RuntimeError("candidate runtime registry changed the r175 terminal boundary")
    return {
        "path": str(registry_path.expanduser().resolve()),
        "gate_path": str(expected_gate),
        "gate_sha256": gate_sha256,
        "frozen_registry_path": str(expected_frozen),
        "frozen_registry_sha256": frozen_sha256,
        "owner_grimmsnarl_pin_superseded_by_exact_strong_gate_floor": True,
        "h10_executable_floor_owner": "exact_strong_public_gate_row",
    }


def validate(
    *,
    design_path: Path,
    baseline_manifest_path: Path,
    baseline_dir: Path,
    historical_frozen_path: Path,
    r182_path: Path,
    candidate_gate_path: Path | None = None,
    candidate_frozen_path: Path | None = None,
    candidate_runtime_registry_path: Path | None = None,
    candidate_pin_sidecar_path: Path | None = None,
) -> dict[str, Any]:
    """Validate identity provenance without writing or contacting a service."""

    design = _read_json(design_path, label="r192 design")
    manifest = _read_json(baseline_manifest_path, label="baseline manifest")
    historical_frozen = _read_json(
        historical_frozen_path, label="historical frozen registry"
    )
    r182 = _read_json(r182_path, label="r182 transport contract")
    _validate_design(design)
    package = _validate_installed_package(
        baseline_dir=baseline_dir,
        manifest=manifest,
    )
    historical = _validate_historical_frozen(historical_frozen)
    transport = _validate_r182(r182)
    report: dict[str, Any] = {
        "schema": "poke_bot.alakazam_marnie_h10_identity_validation_r192/v1",
        "read_only": True,
        "owner_decision_revision": 192,
        "passed": True,
        "inputs": {
            "design": {
                "path": str(design_path.expanduser().resolve()),
                "sha256": _sha256(design_path),
            },
            "baseline_manifest": {
                "path": str(baseline_manifest_path.expanduser().resolve()),
                "sha256": _sha256(baseline_manifest_path),
            },
            "historical_frozen_registry": {
                "path": str(historical_frozen_path.expanduser().resolve()),
                "sha256": _sha256(historical_frozen_path),
            },
            "r182_transport_contract": {
                "path": str(r182_path.expanduser().resolve()),
                "sha256": _sha256(r182_path),
            },
        },
        "h10_package": package,
        "historical_marnie": {
            "opponent_id": HISTORICAL_OPPONENT_ID,
            "checkpoint_digest": historical["checkpoint_digest"],
            "content_digest": historical["content_digest"],
            "distinct_from_h10": True,
        },
        "r182_transport": transport,
        "candidate": {},
    }
    candidate_paths = (
        candidate_gate_path,
        candidate_frozen_path,
        candidate_runtime_registry_path,
    )
    if any(path is not None for path in candidate_paths) and not all(
        path is not None for path in candidate_paths
    ):
        raise RuntimeError(
            "candidate gate, frozen registry, and runtime registry must be supplied together"
        )
    if candidate_pin_sidecar_path is not None and not all(
        path is not None for path in candidate_paths
    ):
        raise RuntimeError(
            "candidate public-mix pin sidecar requires the candidate gate, frozen "
            "registry, and runtime registry"
        )
    if candidate_gate_path is not None:
        gate_payload = _read_json(candidate_gate_path, label="candidate gate")
        frozen_payload = _read_json(
            candidate_frozen_path, label="candidate frozen registry"
        )
        runtime_payload = _read_json(
            candidate_runtime_registry_path, label="candidate runtime registry"
        )
        gate_report = _validate_candidate_gate(gate_payload)
        report["candidate"]["gate"] = {
            **gate_report,
            "path": str(candidate_gate_path.expanduser().resolve()),
            "sha256": _sha256(candidate_gate_path),
        }
        report["candidate"]["frozen_registry"] = {
            **_validate_candidate_frozen(frozen_payload),
            "path": str(candidate_frozen_path.expanduser().resolve()),
            "sha256": _sha256(candidate_frozen_path),
        }
        report["candidate"]["runtime_registry"] = _validate_candidate_runtime_registry(
            runtime_payload,
            registry_path=candidate_runtime_registry_path,
            candidate_gate_path=candidate_gate_path,
            candidate_frozen_path=candidate_frozen_path,
            gate_id=str(gate_report["gate_id"]),
        )
        if candidate_pin_sidecar_path is not None:
            pin_payload = _read_json(
                candidate_pin_sidecar_path,
                label="candidate public-mix pin sidecar",
            )
            report["candidate"]["public_mix_pin_sidecar"] = {
                **_validate_candidate_pin_sidecar(pin_payload),
                "path": str(candidate_pin_sidecar_path.expanduser().resolve()),
                "sha256": _sha256(candidate_pin_sidecar_path),
            }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT / "state/alakazam-marnie-splusplus-opponent-r192.json",
    )
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument(
        "--historical-frozen-registry",
        type=Path,
        default=ROOT / "ops/frozen_specialist_registry_v1.json",
    )
    parser.add_argument(
        "--r182-contract",
        type=Path,
        default=ROOT / "state/alakazam-public-multi-env-split-r182.json",
    )
    parser.add_argument("--candidate-gate", type=Path)
    parser.add_argument("--candidate-frozen-registry", type=Path)
    parser.add_argument("--candidate-runtime-registry", type=Path)
    parser.add_argument("--candidate-pin-sidecar", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate(
            design_path=args.design,
            baseline_manifest_path=args.baseline_manifest,
            baseline_dir=args.baseline_dir,
            historical_frozen_path=args.historical_frozen_registry,
            r182_path=args.r182_contract,
            candidate_gate_path=args.candidate_gate,
            candidate_frozen_path=args.candidate_frozen_registry,
            candidate_runtime_registry_path=args.candidate_runtime_registry,
            candidate_pin_sidecar_path=args.candidate_pin_sidecar,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
