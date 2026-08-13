from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .errors import ResolutionError, ValidationError
from .evidence import evaluate_predicate, require_evidence_ids
from .io import (
    load_json,
    load_jsonl,
    resolve_local_path,
    verify_reference,
    write_json_lines,
)
from .pointers import apply_operations
from .resolve import Resolution, resolve_gateway
from .validate import (
    validate_activations,
    validate_amendments,
    validate_contract,
    validate_evidence_index,
)

try:  # POSIX and Windows both get a standard-library advisory lock.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _assert_safe_write_path(root: Path, path: Path) -> None:
    root = root.resolve()
    absolute = path.absolute()
    if not absolute.is_relative_to(root):
        raise ResolutionError(f"write target escapes the goal package: {path}")
    current = root
    for part in absolute.relative_to(root).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise ResolutionError(f"write target traverses a symbolic link: {current}")
            if not current.resolve().is_relative_to(root):
                raise ResolutionError(f"write target escapes the goal package: {current}")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(root: Path, path: Path, payload: bytes) -> None:
    _assert_safe_write_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_write_path(root, path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable(root: Path, path: Path, payload: bytes) -> None:
    _assert_safe_write_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_write_path(root, path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ResolutionError(f"immutable history path already has other content: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _assert_safe_write_path(root, path)
            if path.read_bytes() != payload:
                raise ResolutionError(
                    f"immutable history path already has other content: {path}"
                )
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _goal_write_lock(root: Path, *, wait: bool = False) -> Iterator[None]:
    root = root.resolve()
    lock_path = root / ".dgoal" / "write.lock"
    _assert_safe_write_path(root, lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_write_path(root, lock_path)
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            if fcntl is not None:
                operation = fcntl.LOCK_EX
                if not wait:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(handle.fileno(), operation)
            else:  # pragma: no cover - exercised on Windows
                handle.seek(0)
                mode = msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK
                msvcrt.locking(handle.fileno(), mode, 1)
        except (BlockingIOError, OSError) as exc:
            raise ResolutionError("another durable-goal writer holds the package lock") from exc
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _commit_history(
    *,
    root: Path,
    gateway_path: Path,
    gateway: dict[str, Any],
    kind: str,
    revision: int,
    records: list[dict[str, Any]],
) -> None:
    payload = write_json_lines(records).encode("utf-8")
    checksum = _sha256_bytes(payload)
    relative = Path(".dgoal") / "history" / f"{kind}-r{revision:06d}-{checksum[7:19]}.jsonl"
    history_path = root / relative
    _write_immutable(root, history_path, payload)
    updated_gateway = deepcopy(gateway)
    updated_gateway[kind] = {"path": relative.as_posix(), "sha256": checksum}
    if kind == "amendments":
        updated_gateway["current_revision"] = revision
    gateway_payload = (json.dumps(updated_gateway, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(root, gateway_path, gateway_payload)


def _commit_json_reference(
    *,
    root: Path,
    gateway_path: Path,
    gateway: dict[str, Any],
    key: str,
    revision: int,
    value: dict[str, Any],
) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    checksum = _sha256_bytes(payload)
    relative = (
        Path(".dgoal")
        / "history"
        / f"{key}-r{revision:06d}-{checksum[7:19]}.json"
    )
    _write_immutable(root, root / relative, payload)
    updated_gateway = deepcopy(gateway)
    updated_gateway[key] = {"path": relative.as_posix(), "sha256": checksum}
    gateway_payload = (
        json.dumps(updated_gateway, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(root, gateway_path, gateway_payload)


def record_amendment(
    gateway_path: str | Path,
    *,
    operations: list[dict[str, Any]],
    reason: str,
    activation_mode: str = "manual",
    activation_condition: dict[str, Any] | None = None,
    authority: str = "owner",
    recorded_at: str | None = None,
) -> Resolution:
    path = Path(gateway_path).resolve()
    root = path.parent
    with _goal_write_lock(root):
        resolution = resolve_gateway(path)
        gateway = load_json(path)
        if not operations:
            raise ValidationError("an amendment requires at least one operation")
        candidate = apply_operations(resolution.desired_contract, operations)
        revision = resolution.current_revision + 1
        candidate["revision"] = revision
        validate_contract(candidate, goal_id=resolution.goal_id)

        amendments_path = verify_reference(
            root, gateway["amendments"], label="gateway.amendments"
        )
        amendments = load_jsonl(amendments_path)
        amendment = {
            "schema": "durable-goals.amendment/v1",
            "goal_id": resolution.goal_id,
            "revision": revision,
            "recorded_at": recorded_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "authority": authority,
            "reason": reason,
            "activation_mode": activation_mode,
            "operations": operations,
        }
        if activation_condition is not None:
            amendment["activation_condition"] = activation_condition
        candidate_amendments = [*amendments, amendment]
        base_revision = resolution.desired_contract["revision"] - len(amendments)
        validate_amendments(
            candidate_amendments,
            goal_id=resolution.goal_id,
            base_revision=base_revision,
            current_revision=revision,
        )
        _commit_history(
            root=root,
            gateway_path=path,
            gateway=gateway,
            kind="amendments",
            revision=revision,
            records=candidate_amendments,
        )
        return resolve_gateway(path)


def activate_amendment(
    gateway_path: str | Path,
    amendment_revision: int,
    *,
    evidence_id: str | None = None,
    evidence_ids: list[str] | None = None,
    activated_at: str | None = None,
) -> Resolution:
    path = Path(gateway_path).resolve()
    root = path.parent
    with _goal_write_lock(root):
        resolution = resolve_gateway(path)
        if not resolution.pending_activations:
            raise ResolutionError("the goal has no pending amendment to activate")
        expected = resolution.pending_activations[0]["revision"]
        if amendment_revision != expected:
            raise ResolutionError(
                f"only the next pending amendment may activate: expected {expected}, "
                f"got {amendment_revision}"
            )
        pending = resolution.pending_activations[0]
        condition = pending.get("condition")
        if pending["mode"] == "next_safe_boundary":
            if condition is None:
                raise ResolutionError("next_safe_boundary activation lacks a condition")
            result = evaluate_predicate(condition, resolution.evidence)
            if not result.satisfied:
                raise ResolutionError(
                    "activation condition is not satisfied: "
                    + json.dumps(result.explanation, sort_keys=True)
                )

        selected_evidence = set(evidence_ids or [])
        if evidence_id is not None:
            selected_evidence.add(evidence_id)
        if condition is not None:
            selected_evidence.update(require_evidence_ids(condition))
        undeclared = sorted(selected_evidence - set(resolution.evidence))
        if undeclared:
            raise ResolutionError(
                "activation evidence is not declared: " + ", ".join(undeclared)
            )

        gateway = load_json(path)
        amendments_path = verify_reference(
            root, gateway["amendments"], label="gateway.amendments"
        )
        activations_path = verify_reference(
            root, gateway["activations"], label="gateway.activations"
        )
        amendment_revisions = [item["revision"] for item in load_jsonl(amendments_path)]
        activations = load_jsonl(activations_path)
        evidence_index_path = verify_reference(
            root, gateway["evidence_index"], label="gateway.evidence_index"
        )
        evidence_index = load_json(evidence_index_path)
        evidence_checksums = {
            item["id"]: item["sha256"] for item in evidence_index["entries"]
        }
        activation: dict[str, Any] = {
            "schema": "durable-goals.activation/v1",
            "goal_id": resolution.goal_id,
            "amendment_revision": amendment_revision,
            "activated_at": activated_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if selected_evidence:
            activation["evidence"] = [
                {"id": item, "sha256": evidence_checksums[item]}
                for item in sorted(selected_evidence)
            ]
        candidate_activations = [*activations, activation]
        validate_activations(
            candidate_activations,
            goal_id=resolution.goal_id,
            amendment_revisions=amendment_revisions,
        )
        _commit_history(
            root=root,
            gateway_path=path,
            gateway=gateway,
            kind="activations",
            revision=amendment_revision,
            records=candidate_activations,
        )
        return resolve_gateway(path)


def materialize_status(gateway_path: str | Path) -> Path:
    path = Path(gateway_path).resolve()
    root = path.parent
    with _goal_write_lock(root):
        resolution = resolve_gateway(path)
        gateway = load_json(path)
        status_reference = gateway.get("status", {"path": "STATUS.json"})
        status_path_value = status_reference.get("path")
        if not isinstance(status_path_value, str) or not status_path_value:
            raise ValidationError("gateway.status.path must be a non-empty string")
        status_path = resolve_local_path(root, status_path_value)
        payload = (json.dumps(resolution.status, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        _atomic_write(root, status_path, payload)
        return status_path


def record_evidence(
    gateway_path: str | Path,
    evidence_id: str,
    source_path: str | Path,
    *,
    contract_revision: int | None = None,
    recorded_at: str | None = None,
) -> Resolution:
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", evidence_id) is None:
        raise ValidationError(
            "evidence_id must contain lowercase letters, digits, dots, underscores, "
            "or hyphens"
        )
    path = Path(gateway_path).resolve()
    root = path.parent
    source = Path(source_path).resolve()
    load_json(source)
    payload = source.read_bytes()
    checksum = _sha256_bytes(payload)
    with _goal_write_lock(root):
        resolution = resolve_gateway(path)
        gateway = load_json(path)
        index_path = verify_reference(
            root, gateway["evidence_index"], label="gateway.evidence_index"
        )
        index = validate_evidence_index(load_json(index_path), goal_id=resolution.goal_id)
        if any(item["id"] == evidence_id for item in index["entries"]):
            raise ResolutionError(f"evidence id already exists: {evidence_id}")
        revision = (
            resolution.active_revision
            if contract_revision is None
            else contract_revision
        )
        if revision < 1 or revision > resolution.current_revision:
            raise ValidationError(
                f"contract_revision must be between 1 and {resolution.current_revision}"
            )
        relative = (
            Path(".dgoal")
            / "evidence"
            / f"{evidence_id}-{checksum[7:19]}.json"
        )
        _write_immutable(root, root / relative, payload)
        updated_index = deepcopy(index)
        updated_index["entries"].append(
            {
                "id": evidence_id,
                "path": relative.as_posix(),
                "sha256": checksum,
                "contract_revision": revision,
                "recorded_at": recorded_at
                or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
        validate_evidence_index(updated_index, goal_id=resolution.goal_id)
        _commit_json_reference(
            root=root,
            gateway_path=path,
            gateway=gateway,
            key="evidence_index",
            revision=resolution.current_revision,
            value=updated_index,
        )
        return resolve_gateway(path)


def initialize_goal_package(
    directory: str | Path,
    *,
    goal_id: str,
    objective: str,
) -> Path:
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", goal_id) is None:
        raise ValidationError(
            "goal_id must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, dots, underscores, and hyphens"
        )
    if not objective.strip():
        raise ValidationError("objective must be non-empty")
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise ResolutionError(f"goal package directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema": "durable-goals.contract/v1",
        "goal_id": goal_id,
        "revision": 1,
        "objective": objective,
        "invariants": [],
        "completion": {"literal": False},
        "delegations": [],
        "transitions": [],
    }
    validate_contract(contract, goal_id=goal_id)
    contract_payload = (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8")
    amendments_payload = b""
    activations_payload = b""
    evidence_index = {
        "schema": "durable-goals.evidence-index/v1",
        "goal_id": goal_id,
        "entries": [],
    }
    evidence_payload = (
        json.dumps(evidence_index, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    gateway = {
        "schema": "durable-goals.gateway/v1",
        "goal_id": goal_id,
        "current_revision": 1,
        "contract": {
            "path": "contract.json",
            "sha256": _sha256_bytes(contract_payload),
        },
        "amendments": {
            "path": "amendments.jsonl",
            "sha256": _sha256_bytes(amendments_payload),
        },
        "activations": {
            "path": "activations.jsonl",
            "sha256": _sha256_bytes(activations_payload),
        },
        "evidence_index": {
            "path": "evidence-index.json",
            "sha256": _sha256_bytes(evidence_payload),
        },
        "status": {"path": "STATUS.json"},
    }
    goal_markdown = f"""# Goal Gateway

Schema: `durable-goals.goal-gateway/v1`
Status: `authoritative`

Read this file completely before acting, followed by the canonical sources
required for the current action.

## Objective

> {objective}

## Canonical sources

- Machine-verifiable gateway: `gateway.json`
- Typed contract: `contract.json`
- Gateway-selected owner amendment ledger
- Gateway-selected activation ledger
- Evidence index: `evidence-index.json`
- Generated status: `STATUS.json` (non-authoritative)
- Optional multi-goal prompt DAG: repository `workflow.json`

## Source precedence

1. Valid owner amendments determine desired intent.
2. The typed contract owns normalized base semantics.
3. Valid activation records determine adopted intent.
4. Checksum-verified evidence determines factual progress.
5. Generated status and conversation summaries are projections only.

Stop and report contradictions rather than guessing.

When this goal is a workflow node, act on it only when `dgoal workflow next`
emits it. The workflow never assigns or launches a model.
""".encode("utf-8")

    _atomic_write(root, root / "contract.json", contract_payload)
    _atomic_write(root, root / "amendments.jsonl", amendments_payload)
    _atomic_write(root, root / "activations.jsonl", activations_payload)
    _atomic_write(root, root / "evidence-index.json", evidence_payload)
    _atomic_write(
        root,
        root / "gateway.json",
        (json.dumps(gateway, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(root, root / "GOAL.md", goal_markdown)
    materialize_status(root / "gateway.json")
    return root / "gateway.json"


def chain_goal(
    source_gateway: str | Path,
    successor_gateway: str | Path,
    *,
    transition_id: str,
    reason: str,
) -> Resolution:
    source_path = Path(source_gateway).resolve()
    successor_path = Path(successor_gateway).resolve()
    source = resolve_gateway(source_path)
    successor = resolve_gateway(successor_path)
    successor_goal = successor_path.parent / "GOAL.md"
    if not successor_goal.is_file():
        raise ResolutionError(f"successor has no authoritative GOAL.md: {successor_goal}")
    transitions = deepcopy(source.desired_contract.get("transitions", []))
    if any(item["id"] == transition_id for item in transitions):
        raise ResolutionError(f"transition id already exists: {transition_id}")
    goal_gateway = os.path.relpath(successor_goal, source_path.parent)
    transitions.append(
        {
            "id": transition_id,
            "goal_id": successor.goal_id,
            "goal_gateway": Path(goal_gateway).as_posix(),
            "after": "completion",
        }
    )
    return record_amendment(
        source_path,
        operations=[
            {
                "op": "set",
                "path": "/transitions",
                "expect": source.desired_contract.get("transitions", []),
                "value": transitions,
            }
        ],
        reason=reason,
        activation_mode="manual",
    )
