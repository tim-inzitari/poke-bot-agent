#!/usr/bin/env python3
"""Consume one receipt-gated r241 direct-policy authorization into a queue.

This is deliberately *not* a Kaggle uploader.  It performs no network I/O,
does not import a Kaggle client, and has no retry or upload operation.  Its
only mutation, when ``--enqueue`` is explicit, is one local direct-policy
queue entry after re-validating the terminal finalizer's immutable evidence.

The older generic Kaggle queue processor is intentionally incompatible with
this format because it requires belief/search assets that r241 forbids.  A
future uploader must explicitly support this schema and record an upload
receipt; until then this processor is the concrete, safe handoff boundary.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scripts import finalize_alakazam_new_list_direct_r241 as finalizer


QUEUE_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241_submission_queue/v1"
CONSUMPTION_BINDING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_queue_consumption_binding/v1"
)
CONSUMPTION_BINDING_SUFFIX = ".r241-direct-policy-queue-consumption.json"


class R241QueueProcessorError(RuntimeError):
    """The single authorized direct-policy handoff cannot be queued safely."""


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241QueueProcessorError(f"{label} must be a regular non-symlink file")
    return raw.resolve()


def _directory(path: Path | str, *, label: str, create: bool = False) -> Path:
    raw = Path(path).expanduser()
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    if raw.is_symlink() or not raw.is_dir():
        raise R241QueueProcessorError(f"{label} must be a real directory")
    return raw.resolve()


def _read_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241QueueProcessorError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R241QueueProcessorError(f"{label} must be a JSON object")
    return source, value


def _identity(path: Path, *, label: str) -> dict[str, object]:
    source = _regular_file(path, label=label)
    return {
        "path": str(source),
        "sha256": finalizer.sha256_file(source),
        "size_bytes": int(source.stat().st_size),
    }


def _identity_matches(value: object, *, expected: Mapping[str, object], label: str) -> None:
    if not isinstance(value, Mapping):
        raise R241QueueProcessorError(f"{label} is not an identity")
    try:
        declared_size = int(value.get("size_bytes", expected.get("size_bytes", -1)))
        expected_size = int(expected.get("size_bytes", -2))
    except (TypeError, ValueError) as exc:
        raise R241QueueProcessorError(f"{label} size is invalid") from exc
    if (
        str(value.get("sha256") or value.get("digest") or "")
        != str(expected.get("sha256") or "")
        or declared_size != expected_size
    ):
        raise R241QueueProcessorError(f"{label} identity mismatches terminal evidence")
    declared_path = str(value.get("path") or "").strip()
    if declared_path and Path(declared_path).expanduser().resolve() != Path(
        str(expected["path"])
    ).resolve():
        raise R241QueueProcessorError(f"{label} path mismatches terminal evidence")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return finalizer.sha256_bytes(finalizer.canonical_json(dict(payload)))


def _entry_from_authorization(
    authorization: Mapping[str, Any],
    *,
    authorization_identity: Mapping[str, object],
    finalizer_receipt_identity: Mapping[str, object],
) -> dict[str, object]:
    rows = authorization.get("queue_entries")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise R241QueueProcessorError("r241 authorization must contain exactly one queue entry")
    source = dict(rows[0])
    nonce = str(source.get("single_use_nonce") or "")
    if not finalizer._SHA256.fullmatch(nonce):
        raise R241QueueProcessorError("r241 queue authorization nonce is invalid")
    if (
        source.get("sequence") != 1
        or source.get("queue_status") != "authorized_pending_external_queue"
        or source.get("turn_order_preference") != "first_if_allowed"
        or source.get("maximum_uses") != 1
        or source.get("remaining_uses") != 1
        or source.get("retry_allowed") is not False
        or source.get("duplicate_allowed") is not False
        or source.get("copy_allowed") is not False
        or source.get("direct_policy_only") is not True
    ):
        raise R241QueueProcessorError("r241 authorization entry is not single-use/direct-only")
    return {
        "sequence": 1,
        "single_use_nonce": nonce,
        "queue_status": "pending_explicit_direct_policy_uploader",
        "turn_order_preference": "first_if_allowed",
        "maximum_uses": 1,
        "remaining_uses": 1,
        "retry_allowed": False,
        "duplicate_allowed": False,
        "copy_allowed": False,
        "direct_policy_only": True,
        "upload_performed": False,
        "upload_receipt_present": False,
        "submission_execution": "explicit_direct_policy_uploader_required",
        "authorization": dict(authorization_identity),
        "finalizer_receipt": dict(finalizer_receipt_identity),
        "terminal_checkpoint": dict(authorization["terminal_checkpoint"]),
        "terminal_five_epoch_rehearsal_receipt": dict(
            authorization["terminal_five_epoch_rehearsal_receipt"]
        ),
        "expert_window": dict(authorization["expert_window"]),
        "archive": dict(authorization["archive"]),
    }


def validate_authorized_handoff(
    *,
    authorization_path: Path,
    finalizer_receipt_path: Path,
    contract_path: Path = finalizer.CONTRACT_PATH,
    official_libcg_staging_path: Path = finalizer.OFFICIAL_LIBCG_STAGING_PATH,
) -> dict[str, object]:
    """Return one concrete queue entry only after all immutable gates verify."""

    authorization_file, authorization = _read_object(
        authorization_path, label="r241 queue authorization"
    )
    receipt_file, receipt = _read_object(
        finalizer_receipt_path, label="r241 terminal finalizer receipt"
    )
    authorization_identity = _identity(
        authorization_file, label="r241 queue authorization"
    )
    receipt_identity = _identity(receipt_file, label="r241 terminal finalizer receipt")
    if (
        authorization.get("schema") != finalizer.QUEUE_AUTHORIZATION_SCHEMA
        or int(authorization.get("owner_decision_revision", -1))
        != finalizer.OWNER_REVISION
        or authorization.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or authorization.get("competition") != "pokemon-tcg-ai-battle"
        or authorization.get("authorization_scope") != "one_external_queue_entry_only"
        or authorization.get("submission_count_authorized") != 1
        or authorization.get("submission_count_emitted") != 1
        or authorization.get("turn_order_preference") != "first_if_allowed"
        or authorization.get("direct_policy_only") is not True
        or authorization.get("direct_submission_performed") is not False
        or authorization.get("upload_receipt_present") is not False
        or authorization.get("emitter_network_io_performed") is not False
        or authorization.get("emitter_queue_mutation_performed") is not False
        or authorization.get("submission_execution") != "external_queue_processor_only"
    ):
        raise R241QueueProcessorError("r241 authorization is not an unopened direct-only slot")
    if (
        receipt.get("schema") != finalizer.FINALIZER_RECEIPT_SCHEMA
        or int(receipt.get("owner_decision_revision", -1)) != finalizer.OWNER_REVISION
        or receipt.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or receipt.get("status")
        != "package_built_and_exactly_one_queue_authorization_emitted"
        or receipt.get("queue_authorizations_emitted") != 1
        or receipt.get("submission_count_authorized") != 1
        or receipt.get("submission_count_performed") != 0
        or receipt.get("turn_order_preference") != "first_if_allowed"
        or receipt.get("direct_policy_only") is not True
        or receipt.get("direct_submission_performed") is not False
        or receipt.get("network_io_performed") is not False
        or receipt.get("shared_queue_mutated") is not False
    ):
        raise R241QueueProcessorError("r241 finalizer receipt is not an offline terminal handoff")
    _identity_matches(
        receipt.get("queue_authorization"),
        expected=authorization_identity,
        label="finalizer queue authorization",
    )
    terminal = dict(receipt.get("terminal_evidence") or {})
    package = dict(receipt.get("package") or {})
    _identity_matches(
        terminal.get("terminal_checkpoint"),
        expected=dict(authorization["terminal_checkpoint"]),
        label="finalizer terminal checkpoint",
    )
    _identity_matches(
        terminal.get("terminal_five_epoch_rehearsal_receipt"),
        expected=dict(authorization["terminal_five_epoch_rehearsal_receipt"]),
        label="finalizer terminal rehearsal",
    )
    _identity_matches(
        dict(terminal.get("expert_window") or {}).get("staging_receipt"),
        expected=dict(dict(authorization.get("expert_window") or {}).get("staging_receipt") or {}),
        label="finalizer expert-window staging",
    )
    _identity_matches(
        package.get("archive"),
        expected=dict(authorization["archive"]),
        label="finalizer archive",
    )
    archive_path = _regular_file(
        str(authorization["archive"].get("path") or ""),
        label="r241 authorized archive",
    )
    archive_identity = _identity(archive_path, label="r241 authorized archive")
    _identity_matches(
        authorization.get("archive"), expected=archive_identity, label="authorization archive"
    )
    contract_file, contract = finalizer._read_object(
        contract_path, label="r241 typed contract"
    )
    try:
        if (
            contract_file == finalizer.CONTRACT_PATH.resolve()
            and finalizer.sha256_file(contract_file) != finalizer.CONTRACT_SHA256
        ):
            raise finalizer.R241FinalizerError(
                "canonical r241 typed contract changed without queue-processor revision"
            )
        finalizer._assert_exact_contract(contract)
        staging = finalizer._load_official_cg_staging(
            contract=contract, staging_path=official_libcg_staging_path
        )
        expert_window = finalizer._load_expert_window_evidence(contract)
        audit = finalizer.audit_direct_policy_archive(
            archive_path,
            contract=contract,
            expected_model_sha256=str(
                dict(authorization["terminal_checkpoint"]).get("sha256") or ""
            ),
            official_cg_staging=staging,
        )
    except finalizer.R241FinalizerError as exc:
        raise R241QueueProcessorError("r241 archive revalidation failed") from exc
    if (
        dict(terminal.get("contract") or {}).get("sha256")
        != finalizer.sha256_file(contract_file)
        or str(audit.get("archive_sha256") or "") != archive_identity["sha256"]
        or int(audit.get("archive_size_bytes", -1)) != archive_identity["size_bytes"]
    ):
        raise R241QueueProcessorError("r241 handoff contract or archive provenance drifted")
    runtime_source_audit = dict(package.get("runtime_source_audit") or {})
    local_preflight = dict(runtime_source_audit.get("official_cg_local_preflight") or {})
    if (
        str(local_preflight.get("sha256") or "")
        != str(audit.get("official_r236_local_preflight_sha256") or "")
        or not finalizer._SHA256.fullmatch(str(local_preflight.get("sha256") or ""))
    ):
        raise R241QueueProcessorError("r241 sealed libcg local-preflight provenance drifted")
    authorization_expert_window = dict(authorization.get("expert_window") or {})
    try:
        finalizer._validate_expert_window_binding(
            authorization_expert_window,
            evidence=expert_window,
            label="queue authorization",
        )
    except finalizer.R241FinalizerError as exc:
        raise R241QueueProcessorError("r241 authorization expert window drifted") from exc
    return {
        "authorization": authorization_identity,
        "finalizer_receipt": receipt_identity,
        "entry": _entry_from_authorization(
            authorization,
            authorization_identity=authorization_identity,
            finalizer_receipt_identity=receipt_identity,
        ),
        "archive_audit": audit,
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return finalizer.canonical_json(dict(payload))
    except finalizer.R241FinalizerError as exc:
        raise R241QueueProcessorError("r241 queue payload is not canonical JSON") from exc


def _write_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    parent = _directory(path.parent, label=f"{label} parent", create=True)
    target = parent / path.name
    body = _canonical_json(payload)
    if target.exists() or target.is_symlink():
        existing = _regular_file(target, label=label)
        if existing.read_bytes() != body:
            raise R241QueueProcessorError(f"immutable {label} already exists with different bytes")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.r241-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _regular_file(target, label=label)
            if existing.read_bytes() != body:
                raise R241QueueProcessorError(
                    f"immutable {label} raced with different bytes"
                )
        else:
            os.chmod(target, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _read_queue_or_empty(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    _source, queue = _read_object(path, label="r241 direct-policy submission queue")
    return queue


def _queue_payload(entry: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema": QUEUE_SCHEMA,
        "owner_decision_revision": finalizer.OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "competition": "pokemon-tcg-ai-battle",
        "queue_policy": "exactly_one_receipt_bound_direct_policy_entry",
        "turn_order_preference": "first_if_allowed",
        "submission_count_authorized": 1,
        "submission_count_enqueued": 1,
        "submission_count_performed": 0,
        "direct_policy_only": True,
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
        "queue": [dict(entry)],
    }


def _queue_is_exact(queue: Mapping[str, Any], *, entry: Mapping[str, Any]) -> bool:
    expected = _queue_payload(entry)
    return queue == expected


def _atomic_queue_write(path: Path, payload: Mapping[str, Any]) -> None:
    parent = _directory(path.parent, label="r241 queue parent", create=True)
    target = parent / path.name
    if target.is_symlink():
        raise R241QueueProcessorError("r241 queue path must not be a symlink")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.r241-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise R241QueueProcessorError(
                "r241 target queue appeared during one-shot enqueue"
            ) from exc
        os.chmod(target, 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def enqueue_authorized_handoff(
    *,
    authorization_path: Path,
    finalizer_receipt_path: Path,
    queue_path: Path,
    contract_path: Path = finalizer.CONTRACT_PATH,
    official_libcg_staging_path: Path = finalizer.OFFICIAL_LIBCG_STAGING_PATH,
    enqueue: bool = False,
) -> dict[str, object]:
    """Validate and, only when explicit, enqueue the one authorized package.

    The consumption binding is written before the mutable queue.  If a crash
    occurs between those operations, replaying into another queue is refused;
    that availability loss is intentional because manufacturing a second
    possible submission is worse than requiring a human recovery decision.
    """

    handoff = validate_authorized_handoff(
        authorization_path=authorization_path,
        finalizer_receipt_path=finalizer_receipt_path,
        contract_path=contract_path,
        official_libcg_staging_path=official_libcg_staging_path,
    )
    if not enqueue:
        return {
            "schema": QUEUE_SCHEMA,
            "status": "preflight_passed_no_queue_written",
            "direct_submission_performed": False,
            "network_io_performed": False,
            "handoff": handoff,
        }
    authorization_file = _regular_file(authorization_path, label="r241 queue authorization")
    queue_target = Path(queue_path).expanduser().absolute()
    _directory(queue_target.parent, label="r241 queue parent", create=True)
    marker = authorization_file.with_name(
        authorization_file.name + CONSUMPTION_BINDING_SUFFIX
    )
    entry = dict(handoff["entry"])
    expected_queue = _queue_payload(entry)
    binding = {
        "schema": CONSUMPTION_BINDING_SCHEMA,
        "owner_decision_revision": finalizer.OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "authorization": dict(handoff["authorization"]),
        "finalizer_receipt": dict(handoff["finalizer_receipt"]),
        "queue_path": str(queue_target),
        "queue_entry_sha256": _canonical_digest(entry),
        "submission_count_consumed": 1,
        "direct_submission_performed": False,
        "network_io_performed": False,
    }
    lock_path = marker.with_name(marker.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing_marker = marker.exists() or marker.is_symlink()
            if existing_marker:
                marker_file, marker_payload = _read_object(
                    marker, label="r241 queue consumption binding"
                )
                del marker_file
                if marker_payload != binding:
                    raise R241QueueProcessorError(
                        "r241 authorization was already consumed for a different queue handoff"
                    )
                queue = _read_queue_or_empty(queue_target)
                if queue is None or not _queue_is_exact(queue, entry=entry):
                    raise R241QueueProcessorError(
                        "r241 authorization is already consumed but its exact queue entry is absent"
                    )
                status = "already_enqueued_idempotent"
            else:
                queue = _read_queue_or_empty(queue_target)
                if queue is not None:
                    raise R241QueueProcessorError(
                        "r241 target queue already exists; it must not be merged or overwritten"
                    )
                _write_immutable_json(
                    marker, binding, label="r241 single-use queue consumption binding"
                )
                _atomic_queue_write(queue_target, expected_queue)
                status = "enqueued_pending_explicit_direct_policy_uploader"
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "schema": QUEUE_SCHEMA,
        "status": status,
        "queue_path": str(queue_target),
        "queue_entry_sha256": _canonical_digest(entry),
        "consumption_binding_path": str(marker),
        "submission_count_enqueued": 1,
        "submission_count_performed": 0,
        "direct_submission_performed": False,
        "network_io_performed": False,
        "upload_receipt_present": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--finalizer-receipt", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=finalizer.CONTRACT_PATH)
    parser.add_argument(
        "--official-libcg-staging",
        type=Path,
        default=finalizer.OFFICIAL_LIBCG_STAGING_PATH,
    )
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="materialize the one local handoff; never uploads to Kaggle",
    )
    args = parser.parse_args(argv)
    try:
        result = enqueue_authorized_handoff(
            authorization_path=args.authorization,
            finalizer_receipt_path=args.finalizer_receipt,
            queue_path=args.queue,
            contract_path=args.contract,
            official_libcg_staging_path=args.official_libcg_staging,
            enqueue=args.enqueue,
        )
    except R241QueueProcessorError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
