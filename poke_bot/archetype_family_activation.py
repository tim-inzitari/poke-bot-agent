"""Receipt-backed, atomic family/loss activation decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TRIGGER_SCHEMA = "poke_bot.marnie_family_iteration9_upload_trigger/v1"
PAUSE_SCHEMA = "poke_bot.marnie_family_boundary_pause/v1"
MIGRATION_SCHEMA = "poke_bot.marnie_family_design_migration/v1"


class FamilyActivationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_bound(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise FamilyActivationError(f"missing or digest-mismatched artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_iteration9_upload_trigger(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Require upload success plus consumed auth and all external exact bindings."""
    if contract.get("schema") != TRIGGER_SCHEMA or int(contract.get("iteration", -1)) != 9:
        raise FamilyActivationError("wrong trigger schema or iteration")
    if contract.get("specialist_id") != "marnie-s-grimmsnarl-ex":
        raise FamilyActivationError("wrong trigger specialist")
    bindings = contract.get("bindings")
    required = {
        "commit",
        "checkpoint",
        "package_bundle",
        "representative_deck",
        "competition",
        "submission_label",
        "uploaded_file",
    }
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise FamilyActivationError("trigger does not contain the exact binding set")
    for name in ("commit", "checkpoint", "package_bundle", "representative_deck", "uploaded_file"):
        row = bindings[name]
        path = Path(str(row.get("path", "")))
        expected = str(row.get("sha256", ""))
        if not expected.startswith("sha256:") or sha256(path) != expected:
            raise FamilyActivationError(f"invalid trigger binding: {name}")
    attempt_row = contract.get("attempt") or {}
    auth_row = contract.get("consumed_authorization") or {}
    attempt_path = Path(str(attempt_row.get("path", "")))
    auth_path = Path(str(auth_row.get("path", "")))
    attempt = _read_bound(attempt_path, str(attempt_row.get("sha256", "")))
    auth = _read_bound(auth_path, str(auth_row.get("sha256", "")))
    if attempt.get("schema") != "poke_bot.kaggle_submission_attempt/v1" or int(attempt.get("returncode", -1)) != 0:
        raise FamilyActivationError("Kaggle attempt is not an immutable success")
    if auth.get("schema") != "poke_bot.kaggle_submission_authorization/v1" or auth.get("consumed_before_upload") is not True or int(auth.get("remaining_uses", -1)) != 0:
        raise FamilyActivationError("one-shot authorization was not consumed before upload")
    if Path(str(attempt.get("authorization_consumed", ""))).resolve() != auth_path.resolve():
        raise FamilyActivationError("attempt does not bind the consumed authorization")
    uploaded = bindings["uploaded_file"]
    identity = attempt.get("identity") or {}
    if identity.get("file_sha256") != uploaded["sha256"] or Path(str(identity.get("file", ""))).resolve() != Path(uploaded["path"]).resolve():
        raise FamilyActivationError("attempt uploaded a different file")
    if auth.get("submission_file_checksum") != uploaded["sha256"] or auth.get("frozen_checkpoint_checksum") != bindings["checkpoint"]["sha256"]:
        raise FamilyActivationError("authorization checksum binding mismatch")
    if identity.get("competition") != bindings["competition"] or identity.get("message") != bindings["submission_label"]:
        raise FamilyActivationError("competition or submission label mismatch")
    return {"valid": True, "attempt": str(attempt_path), "authorization": str(auth_path)}


def boundary_decision(
    *,
    trigger_valid: bool,
    study_passed: bool,
    committed_iteration: int,
    next_collection_started: bool,
    already_paused_for_commit: bool,
) -> dict[str, Any]:
    """Pure idempotent decision used immediately after commit."""
    if not trigger_valid or not study_passed:
        return {"action": "continue_unchanged", "target_iteration": None}
    target = int(committed_iteration) + 1
    if next_collection_started:
        return {"action": "defer", "target_iteration": target + 1}
    if already_paused_for_commit:
        return {"action": "already_paused", "target_iteration": target}
    return {"action": "pause_for_atomic_activation", "target_iteration": target}


def validate_atomic_migration(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    allowed = {
        "deck_family_distribution",
        "family_manifest",
        "variant_provenance",
        "replay_list_sampler",
        "archetype_loss_contract",
        "selected_loss_vector",
    }
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if not changed or not {"family_manifest", "selected_loss_vector"}.issubset(changed):
        raise FamilyActivationError("family sampler and loss vector did not change atomically")
    if changed - allowed:
        raise FamilyActivationError(f"migration changed unrelated fields: {sorted(changed - allowed)}")


__all__ = [
    "FamilyActivationError",
    "MIGRATION_SCHEMA",
    "PAUSE_SCHEMA",
    "TRIGGER_SCHEMA",
    "boundary_decision",
    "sha256",
    "validate_atomic_migration",
    "validate_iteration9_upload_trigger",
]
