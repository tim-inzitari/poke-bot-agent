"""Receipt-backed, atomic family/loss activation decisions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import fcntl

from .archetype_loss_contract import canonical_residual_weights, validate_loss_contract
from .archetype_family_study import (
    FamilyStudyError,
    MONITOR_SCHEMA,
    SELECTED_VECTOR_SCHEMA,
    validate_post_activation_monitor,
    validate_study_receipt,
)
from .specialist_archetype_family import SPECIALIST_ID, multiset_digest, validate_manifest


TRIGGER_SCHEMA = "poke_bot.marnie_family_iteration9_upload_trigger/v1"
PAUSE_SCHEMA = "poke_bot.marnie_family_boundary_pause/v1"
MIGRATION_SCHEMA = "poke_bot.marnie_family_design_migration/v1"
REQUEST_SCHEMA = "poke_bot.marnie_family_activation_request/v1"
OWNER_CEILING_SCHEMA = "poke_bot.marnie_family_owner_ceiling_acceptance/v1"
PACKAGE_SCHEMA = "poke_bot.package_deck_contract/v1"
ROLLBACK_REQUEST_SCHEMA = "poke_bot.marnie_family_rollback_request/v1"
ROLLBACK_RECEIPT_SCHEMA = "poke_bot.marnie_family_design_rollback/v1"


class FamilyActivationError(ValueError):
    pass


def validate_owner_ceiling_acceptance(
    acceptance: Mapping[str, Any], *, study_path: Path,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate explicit owner authority without rewriting a failed study."""
    if (
        acceptance.get("schema") != OWNER_CEILING_SCHEMA
        or int(acceptance.get("owner_revision", -1)) != 139
        or acceptance.get("status")
        != "owner_ceiling_accepted_after_inconclusive_study"
        or acceptance.get("measured_study_passed") is not False
        or acceptance.get("measured_study_relabelled") is not False
        or acceptance.get("runtime_authority_before_atomic_migration") is not False
    ):
        raise FamilyActivationError("invalid family owner-ceiling authority")
    study_row = acceptance.get("study") or {}
    if (
        Path(str(study_row.get("path", ""))).resolve()
        != Path(study_path).resolve()
        or str(study_row.get("sha256", "")) != sha256(study_path)
        or study.get("schema")
        != "poke_bot.marnie_archetype_family_shadow_study/v1"
        or study.get("passed") is not False
        or study.get("status")
        != "failed_closed_inconclusive_after_two_rounds"
        or study.get("training_eligible") is not False
        or study.get("replay_eligible") is not False
    ):
        raise FamilyActivationError("owner ceiling does not bind the inconclusive study")
    rounds = list(study.get("rounds") or ())
    selection = acceptance.get("selection") or {}
    if (
        len(rounds) != 2
        or any(row.get("status") != "inconclusive" for row in rounds)
        or int(selection.get("round", -1)) != 1
        or selection.get("direction") != "plus"
    ):
        raise FamilyActivationError("owner ceiling selection is not exact round-1 plus")
    round_one = rounds[0]
    tested = (((round_one.get("perturbation") or {}).get("candidates") or {}).get("plus") or {})
    selected = selection.get("weights") or {}
    if canonical_residual_weights(selected) != canonical_residual_weights(tested):
        raise FamilyActivationError("owner ceiling weights differ from tested round-1 plus")
    same_parent = study.get("same_parent_validation") or {}
    if same_parent.get("valid") is not True:
        raise FamilyActivationError("owner ceiling study lacks same-parent validation")
    return {
        "valid": True,
        "owner_revision": 139,
        "selection": {
            "round": 1,
            "direction": "plus",
            "weights": canonical_residual_weights(selected),
        },
    }


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


def _verify_bound(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise FamilyActivationError(f"missing or digest-mismatched artifact: {path}")


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
    if not trigger_valid:
        return {"action": "continue_unchanged", "target_iteration": None}
    target = int(committed_iteration) + 1
    if next_collection_started:
        return {"action": "defer", "target_iteration": target + 1}
    if already_paused_for_commit:
        return {"action": "already_paused", "target_iteration": target}
    if not study_passed:
        return {"action": "pause_for_required_evidence", "target_iteration": target}
    return {"action": "pause_for_atomic_activation", "target_iteration": target}


def validate_atomic_migration(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    allowed = {
        "runtime_root",
        "common_trainer_args",
        "deck_family_distribution",
        "family_manifest",
        "variant_provenance",
        "replay_list_sampler",
        "archetype_loss_contract",
        "selected_loss_vector",
        "family_activation_authority",
    }
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if not changed or not {"family_manifest", "selected_loss_vector"}.issubset(changed):
        raise FamilyActivationError("family sampler and loss vector did not change atomically")
    if changed - allowed:
        raise FamilyActivationError(f"migration changed unrelated fields: {sorted(changed - allowed)}")


def validate_activation_request(
    request: Mapping[str, Any], *, expected_learner_digest: str | None = None
) -> dict[str, Any]:
    """Validate every checksum-bound study/manifest/vector activation input."""
    if request.get("schema") != REQUEST_SCHEMA:
        raise FamilyActivationError("wrong activation request schema")
    trigger_row = request.get("trigger") or {}
    trigger_path = Path(str(trigger_row.get("path", "")))
    trigger = _read_bound(trigger_path, str(trigger_row.get("sha256", "")))
    validate_iteration9_upload_trigger(trigger)
    manifest_row = request.get("manifest") or {}
    manifest_path = Path(str(manifest_row.get("path", "")))
    manifest = _read_bound(manifest_path, str(manifest_row.get("sha256", "")))
    validate_manifest(manifest, require_activation_ready=True)
    loss_row = request.get("loss_contract") or {}
    loss_path = Path(str(loss_row.get("path", "")))
    loss = _read_bound(loss_path, str(loss_row.get("sha256", "")))
    validate_loss_contract(loss)
    study_row = request.get("study") or {}
    study_path = Path(str(study_row.get("path", "")))
    study = _read_bound(study_path, str(study_row.get("sha256", "")))
    authority_digest: str | None = None
    authority_row = request.get("activation_authority") or {}
    if authority_row:
        authority_path = Path(str(authority_row.get("path", "")))
        authority = _read_bound(
            authority_path, str(authority_row.get("sha256", ""))
        )
        validate_owner_ceiling_acceptance(
            authority, study_path=study_path, study=study
        )
        authority_digest = sha256(authority_path)
    else:
        try:
            validate_study_receipt(study)
        except FamilyStudyError as exc:
            raise FamilyActivationError(str(exc)) from exc
    bindings = request.get("bindings") or {}
    learner_digest = str(bindings.get("learner_sha256") or "")
    if (
        not learner_digest.startswith("sha256:")
        or (
            expected_learner_digest is not None
            and learner_digest != str(expected_learner_digest)
        )
    ):
        raise FamilyActivationError("activation request targets a different learner")
    verified: dict[str, Path] = {}
    for name in (
        "checkpoint", "registry", "selector", "selected_loss_vector",
        "candidate_registry",
    ):
        row = bindings.get(name) or {}
        path = Path(str(row.get("path", "")))
        _verify_bound(path, str(row.get("sha256", "")))
        verified[name] = path
    if sha256(verified["checkpoint"]) != learner_digest:
        raise FamilyActivationError("activation checkpoint is not the exact learner")
    vector = json.loads(
        verified["selected_loss_vector"].read_text(encoding="utf-8")
    )
    expected_vector_status = (
        "selected_by_owner_ceiling_after_inconclusive_study"
        if authority_digest is not None
        else "selected_by_passing_shadow_study"
    )
    if (
        vector.get("schema") != SELECTED_VECTOR_SCHEMA
        or vector.get("status") != expected_vector_status
        or vector.get("specialist_id") != SPECIALIST_ID
        or vector.get("manifest_sha256") != sha256(manifest_path)
        or vector.get("loss_contract_sha256") != sha256(loss_path)
        or vector.get("study_sha256") != sha256(study_path)
        or vector.get("activates_only_with_family_sampler") is not True
        or (
            authority_digest is not None
            and vector.get("activation_authority_sha256") != authority_digest
        )
        or (
            authority_digest is None
            and "activation_authority_sha256" in vector
        )
    ):
        raise FamilyActivationError("selected loss vector binding is invalid")
    sealed = request.get("sealed_pre_activation") or {}
    required_seals = {
        "registry", "selector", "learner", "checkpoint", "optimizer",
        "scaler", "rng", "design_contract", "manifest", "loss_vector",
    }
    if set(sealed) != required_seals or any(
        not str(value).startswith("sha256:") for value in sealed.values()
    ):
        raise FamilyActivationError("pre-activation state is not completely sealed")
    return {
        "valid": True,
        "learner_sha256": learner_digest,
        "trigger_sha256": sha256(trigger_path),
        "manifest_sha256": sha256(manifest_path),
        "loss_contract_sha256": sha256(loss_path),
        "study_sha256": sha256(study_path),
        "selected_loss_vector_sha256": sha256(
            verified["selected_loss_vector"]
        ),
        "activation_authority": (
            "owner_ceiling_after_inconclusive_study"
            if authority_digest is not None
            else "measured_shadow_study_pass"
        ),
        "activation_authority_sha256": authority_digest,
    }


def materialize_activation_ready_pause(
    *,
    original_pause_path: Path,
    request_path: Path,
    output_path: Path,
    managed_training_active: bool,
) -> dict[str, Any]:
    """Bind a post-study request while the original status-75 pause persists."""
    if bool(managed_training_active):
        raise FamilyActivationError("managed training is active at activation pause")
    original = json.loads(Path(original_pause_path).read_text(encoding="utf-8"))
    if (
        original.get("schema") != PAUSE_SCHEMA
        or original.get("next_collection_started") is not False
        or int(original.get("restart_prevent_status", -1)) != 75
    ):
        raise FamilyActivationError("original pause is not an exact status-75 boundary")
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    validate_activation_request(
        request,
        expected_learner_digest=str(original.get("learner_sha256") or ""),
    )
    trigger = request.get("trigger") or {}
    if (
        Path(str(trigger.get("path", ""))).resolve()
        != Path(str(original.get("trigger_path", ""))).resolve()
        or str(trigger.get("sha256", ""))
        != str(original.get("trigger_sha256", ""))
    ):
        raise FamilyActivationError("study request binds a different upload trigger")
    payload = {
        "schema": PAUSE_SCHEMA,
        "owner_revision": 130,
        "committed_iteration": int(original["committed_iteration"]),
        "target_iteration": int(original["target_iteration"]),
        "learner_sha256": str(original["learner_sha256"]),
        "trigger_path": str(original["trigger_path"]),
        "trigger_sha256": str(original["trigger_sha256"]),
        "request_path": str(Path(request_path).resolve()),
        "request_sha256": sha256(request_path),
        "activation_evidence_complete": True,
        "pause_reason": "ready_for_atomic_activation_after_shadow_study",
        "next_collection_started": False,
        "restart_prevent_status": 75,
        "managed_training_active": False,
        "derived_from_pause": {
            "path": str(Path(original_pause_path).resolve()),
            "sha256": sha256(original_pause_path),
        },
    }
    if Path(output_path).is_file():
        existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
        if existing != payload:
            raise FamilyActivationError("activation-ready pause identity changed")
        return existing
    _write_exclusive(Path(output_path), payload)
    return payload


def validate_migration_receipt(
    receipt: Mapping[str, Any], *, request_path: Path
) -> dict[str, Any]:
    if (
        receipt.get("schema") != MIGRATION_SCHEMA
        or receipt.get("status") != "activated_atomically"
        or Path(str(receipt.get("request", ""))).resolve()
        != Path(request_path).resolve()
        or receipt.get("request_sha256") != sha256(request_path)
    ):
        raise FamilyActivationError("family migration receipt is invalid")
    for path_key, digest_key in (
        ("activated_registry", "activated_registry_sha256"),
        ("environment_drop_in", "environment_drop_in_sha256"),
    ):
        _verify_bound(
            Path(str(receipt.get(path_key, ""))),
            str(receipt.get(digest_key, "")),
        )
    return dict(receipt)


def validate_rollback_request(
    request: Mapping[str, Any],
    *,
    migration_path: Path,
    monitor_path: Path,
) -> dict[str, Any]:
    if request.get("schema") != ROLLBACK_REQUEST_SCHEMA:
        raise FamilyActivationError("wrong family rollback-request schema")
    migration = _read_bound(
        Path(str((request.get("migration") or {}).get("path", ""))),
        str((request.get("migration") or {}).get("sha256", "")),
    )
    if Path(str((request.get("migration") or {}).get("path", ""))).resolve() != Path(migration_path).resolve():
        raise FamilyActivationError("rollback request binds another migration")
    validate_migration_receipt(
        migration,
        request_path=Path(str(migration.get("request", ""))),
    )
    monitor = _read_bound(
        Path(str((request.get("monitor") or {}).get("path", ""))),
        str((request.get("monitor") or {}).get("sha256", "")),
    )
    if Path(str((request.get("monitor") or {}).get("path", ""))).resolve() != Path(monitor_path).resolve():
        raise FamilyActivationError("rollback request binds another monitor")
    try:
        validate_post_activation_monitor(monitor)
    except FamilyStudyError as exc:
        raise FamilyActivationError(str(exc)) from exc
    if monitor.get("rollback_required") is not True:
        raise FamilyActivationError("rollback request lacks a triggering monitor")
    return dict(request)


def _post_activation_boundary(
    *,
    request_path: Path,
    run_dir: Path,
    committed_iteration: int,
    learner_digest: str,
    next_collection_started: bool,
) -> dict[str, Any] | None:
    """Return a post-activation decision, or None before activation."""
    family_root = Path(request_path).resolve().parent
    migration_path = family_root / "migration-receipt.json"
    monitor_path = family_root / "post-activation-monitor.json"
    rollback_request_path = family_root / "rollback-request.json"
    rollback_receipt_path = family_root / "rollback-receipt.json"
    if rollback_receipt_path.is_file():
        rollback = json.loads(rollback_receipt_path.read_text(encoding="utf-8"))
        if (
            rollback.get("schema") != ROLLBACK_RECEIPT_SCHEMA
            or rollback.get("status") != "rolled_back_at_clean_boundary"
            or rollback.get("request_sha256")
            != (sha256(rollback_request_path) if rollback_request_path.is_file() else None)
        ):
            raise FamilyActivationError("family rollback receipt is invalid")
        return {
            "action": "continue_unchanged",
            "target_iteration": None,
            "reason": "family_design_already_rolled_back",
            "rollback_receipt": str(rollback_receipt_path),
        }
    if not migration_path.exists():
        return None
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    validate_migration_receipt(migration, request_path=request_path)
    if next_collection_started:
        return {
            "action": "defer_post_activation_monitor",
            "target_iteration": int(committed_iteration) + 2,
        }
    if monitor_path.is_file():
        monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
        try:
            validate_post_activation_monitor(monitor)
        except FamilyStudyError as exc:
            raise FamilyActivationError(str(exc)) from exc
        monitor_pause = Path(str(monitor.get("pause_receipt", ""))).resolve()
        if (
            not monitor_pause.is_file()
            or monitor.get("pause_receipt_sha256") != sha256(monitor_pause)
        ):
            raise FamilyActivationError(
                "post-activation monitor binds another boundary"
            )
        if monitor.get("rollback_required") is True:
            if not rollback_request_path.is_file():
                raise FamilyActivationError(
                    "required family rollback request is absent"
                )
            validate_rollback_request(
                json.loads(rollback_request_path.read_text(encoding="utf-8")),
                migration_path=migration_path,
                monitor_path=monitor_path,
            )
            return {
                "action": "pause_for_required_rollback",
                "target_iteration": int(committed_iteration) + 1,
                "pause_receipt": str(monitor_pause),
            }
        return {
            "action": "continue_unchanged",
            "target_iteration": None,
            "reason": "post_activation_monitor_passed",
            "monitor_receipt": str(monitor_path),
        }

    monitor_pause = (
        run_dir
        / "family_activation"
        / f"monitor_pause_after_iter_{int(committed_iteration):05d}.json"
    )
    if not monitor_path.is_file():
        payload = {
            "schema": PAUSE_SCHEMA,
            "owner_revision": 133,
            "committed_iteration": int(committed_iteration),
            "target_iteration": int(committed_iteration) + 1,
            "learner_sha256": str(learner_digest),
            "request_path": str(request_path.resolve()),
            "request_sha256": sha256(request_path),
            "migration_path": str(migration_path),
            "migration_sha256": sha256(migration_path),
            "pause_reason": "fresh_post_activation_family_monitor_required",
            "next_collection_started": False,
            "restart_prevent_status": 78,
        }
        if monitor_pause.is_file():
            if json.loads(monitor_pause.read_text(encoding="utf-8")) != payload:
                raise FamilyActivationError("post-activation monitor pause changed")
        else:
            _write_exclusive(monitor_pause, payload)
        return {
            "action": "pause_for_post_activation_monitor",
            "target_iteration": int(committed_iteration) + 1,
            "pause_receipt": str(monitor_pause),
            "pause_receipt_sha256": sha256(monitor_pause),
        }
    raise AssertionError("unreachable post-activation monitor state")


def validate_package_deck_contract(
    contract: Mapping[str, Any], *, legality: Any, classify: Any
) -> dict[str, Any]:
    """Validate a future owner-supplied exact package list; never select one."""
    if contract.get("schema") != PACKAGE_SCHEMA or contract.get("automatic_selection") is not False:
        raise FamilyActivationError("invalid or automatically selected package contract")
    cards = contract.get("authorized_exact_card_ids")
    if cards is None:
        return {"status": "pending_owner_exact_list", "authorized": False}
    if not isinstance(cards, list) or len(cards) != 60 or any(
        isinstance(card, bool) or not isinstance(card, int) or card <= 0 for card in cards
    ):
        raise FamilyActivationError("owner package list is not exactly 60 integer card IDs")
    if not legality(cards) or classify(cards) != SPECIALIST_ID:
        raise FamilyActivationError("owner package list failed legality or family identity")
    evidence = contract.get("validation_evidence") or {}
    required = {"engine_smoke", "exact_formal_evaluation", "paired_old_package_comparison", "clean_boundary"}
    if set(evidence) != required or not all(bool(value) for value in evidence.values()):
        raise FamilyActivationError("future package switch lacks required passing evidence")
    return {"status": "authorized_at_clean_boundary", "authorized": True, "multiset_digest": multiset_digest(cards)}


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def boundary_pause_hook(
    *,
    request_path: Path | None,
    trigger_path: Path | None = None,
    run_dir: Path,
    committed_iteration: int,
    learner_digest: str,
    next_collection_started: bool,
) -> dict[str, Any]:
    """Enforce the immutable last-old-system boundary from owner revision 130.

    Before a successful checksum-exact upload trigger exists, the old recipe
    may continue.  After it exists, every absent, failed or inconclusive
    activation input produces an immutable restart-preventing pause rather
    than allowing another old-system collection.
    """
    request_path = Path(request_path) if request_path is not None else None
    trigger_path = Path(trigger_path) if trigger_path is not None else None
    request: dict[str, Any] | None = None
    if request_path is not None and request_path.is_file():
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            request = None

    trigger_row: Mapping[str, Any] = {}
    if trigger_path is not None and trigger_path.is_file():
        trigger_row = {"path": str(trigger_path), "sha256": sha256(trigger_path)}
    elif request is not None:
        trigger_row = request.get("trigger") or {}
    candidate_trigger = Path(str(trigger_row.get("path", "")))
    try:
        trigger = _read_bound(candidate_trigger, str(trigger_row.get("sha256", "")))
        validate_iteration9_upload_trigger(trigger)
    except (FamilyActivationError, OSError, json.JSONDecodeError):
        # Iteration 9 is the immutable last old-system milestone.  Its Kaggle
        # upload is asynchronous, so the trigger will normally not exist at
        # the instant the commit lands.  Earlier iterations may continue, but
        # iteration 10 must never begin while that exact upload is pending.
        # Persist a restart-preventing receipt and let the managed post-upload
        # chain resume only after it can bind the successful trigger.
        if int(committed_iteration) >= 9 and trigger_path is not None:
            lock_path = run_dir / "family_activation" / "boundary.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                pause_path = (
                    run_dir
                    / "family_activation"
                    / f"await_upload_after_iter_{int(committed_iteration):05d}.json"
                )
                payload = {
                    "schema": PAUSE_SCHEMA,
                    "owner_revision": 134,
                    "committed_iteration": int(committed_iteration),
                    "target_iteration": int(committed_iteration) + 1,
                    "learner_sha256": str(learner_digest),
                    "expected_trigger_path": str(trigger_path),
                    "pause_reason": "awaiting_successful_checksum_exact_iteration9_upload",
                    "next_collection_started": False,
                    "restart_prevent_status": 75,
                }
                if pause_path.is_file():
                    existing = json.loads(pause_path.read_text(encoding="utf-8"))
                    if existing != payload:
                        raise FamilyActivationError(
                            "iteration-9 upload-wait pause identity changed"
                        )
                else:
                    _write_exclusive(pause_path, payload)
                return {
                    "action": "pause_for_required_evidence",
                    "target_iteration": int(committed_iteration) + 1,
                    "reason": "successful_iteration9_upload_pending",
                    "pause_receipt": str(pause_path),
                    "pause_receipt_sha256": sha256(pause_path),
                }
        return {
            "action": "continue_unchanged",
            "target_iteration": None,
            "reason": "successful_iteration9_upload_trigger_absent",
        }

    lock_path = run_dir / "family_activation" / "boundary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if request_path is not None and request is not None:
            try:
                post_activation = _post_activation_boundary(
                    request_path=request_path,
                    run_dir=run_dir,
                    committed_iteration=committed_iteration,
                    learner_digest=learner_digest,
                    next_collection_started=next_collection_started,
                )
            except (FamilyActivationError, OSError, json.JSONDecodeError) as exc:
                failure_path = (
                    run_dir
                    / "family_activation"
                    / f"lifecycle_failure_after_iter_{int(committed_iteration):05d}.json"
                )
                failure = {
                    "schema": PAUSE_SCHEMA,
                    "owner_revision": 133,
                    "committed_iteration": int(committed_iteration),
                    "target_iteration": int(committed_iteration) + 1,
                    "learner_sha256": str(learner_digest),
                    "request_path": str(request_path),
                    "request_sha256": sha256(request_path),
                    "pause_reason": f"post_activation_lifecycle_invalid: {exc}",
                    "next_collection_started": False,
                    "restart_prevent_status": 78,
                }
                if failure_path.is_file():
                    if json.loads(failure_path.read_text(encoding="utf-8")) != failure:
                        raise FamilyActivationError(
                            "post-activation lifecycle failure identity changed"
                        )
                else:
                    _write_exclusive(failure_path, failure)
                return {
                    "action": "pause_for_post_activation_monitor",
                    "target_iteration": int(committed_iteration) + 1,
                    "pause_receipt": str(failure_path),
                    "pause_receipt_sha256": sha256(failure_path),
                }
            if post_activation is not None:
                return post_activation
        pause_path = run_dir / "family_activation" / f"pause_after_iter_{committed_iteration:05d}.json"
        evidence_error = ""
        evidence_complete = False
        if request is None:
            evidence_error = "activation request is absent or unreadable"
        else:
            try:
                if request.get("schema") != REQUEST_SCHEMA:
                    raise FamilyActivationError("wrong activation request schema")
                bound_trigger = request.get("trigger") or {}
                if (
                    Path(str(bound_trigger.get("path", ""))).resolve()
                    != candidate_trigger.resolve()
                    or str(bound_trigger.get("sha256", "")) != sha256(candidate_trigger)
                ):
                    raise FamilyActivationError("activation request binds a different upload trigger")
                manifest_row = request.get("manifest") or {}
                manifest = _read_bound(Path(str(manifest_row.get("path", ""))), str(manifest_row.get("sha256", "")))
                validate_manifest(manifest, require_activation_ready=True)
                loss_row = request.get("loss_contract") or {}
                loss = _read_bound(Path(str(loss_row.get("path", ""))), str(loss_row.get("sha256", "")))
                validate_loss_contract(loss)
                study_row = request.get("study") or {}
                study = _read_bound(Path(str(study_row.get("path", ""))), str(study_row.get("sha256", "")))
                try:
                    validate_study_receipt(study)
                except FamilyStudyError as exc:
                    raise FamilyActivationError(str(exc)) from exc
                bindings = request.get("bindings") or {}
                if bindings.get("learner_sha256") != learner_digest:
                    raise FamilyActivationError("activation request targets a different learner")
                for name in (
                    "checkpoint", "registry", "selector",
                    "selected_loss_vector", "candidate_registry",
                ):
                    row = bindings.get(name) or {}
                    _verify_bound(Path(str(row.get("path", ""))), str(row.get("sha256", "")))
                evidence_complete = True
            except (FamilyActivationError, OSError, json.JSONDecodeError) as exc:
                evidence_error = str(exc)
        decision = boundary_decision(
            trigger_valid=True,
            study_passed=evidence_complete,
            committed_iteration=committed_iteration,
            next_collection_started=next_collection_started,
            already_paused_for_commit=pause_path.is_file(),
        )
        if decision["action"] in {
            "pause_for_atomic_activation",
            "pause_for_required_evidence",
        }:
            payload = {
                "schema": PAUSE_SCHEMA,
                "owner_revision": 130,
                "committed_iteration": int(committed_iteration),
                "target_iteration": decision["target_iteration"],
                "learner_sha256": learner_digest,
                "trigger_path": str(candidate_trigger),
                "trigger_sha256": sha256(candidate_trigger),
                "request_path": str(request_path) if request_path is not None else None,
                "request_sha256": sha256(request_path) if request_path is not None and request_path.is_file() else None,
                "activation_evidence_complete": evidence_complete,
                "pause_reason": "ready_for_atomic_activation" if evidence_complete else evidence_error,
                "next_collection_started": False,
                "restart_prevent_status": 75,
            }
            _write_exclusive(pause_path, payload)
            decision["pause_receipt"] = str(pause_path)
            decision["pause_receipt_sha256"] = sha256(pause_path)
        return decision


__all__ = [
    "FamilyActivationError",
    "MIGRATION_SCHEMA",
    "PAUSE_SCHEMA",
    "PACKAGE_SCHEMA",
    "REQUEST_SCHEMA",
    "OWNER_CEILING_SCHEMA",
    "ROLLBACK_RECEIPT_SCHEMA",
    "ROLLBACK_REQUEST_SCHEMA",
    "TRIGGER_SCHEMA",
    "boundary_decision",
    "boundary_pause_hook",
    "materialize_activation_ready_pause",
    "sha256",
    "validate_atomic_migration",
    "validate_activation_request",
    "validate_owner_ceiling_acceptance",
    "validate_iteration9_upload_trigger",
    "validate_migration_receipt",
    "validate_package_deck_contract",
    "validate_rollback_request",
]
