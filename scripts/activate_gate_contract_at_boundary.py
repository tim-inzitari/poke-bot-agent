#!/usr/bin/env python3
"""Atomically activate a staged gate contract at an immutable iteration boundary."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def systemctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def service_pid(service: str) -> int:
    result = systemctl("show", service, "-p", "MainPID", "--value")
    return int(result.stdout.strip() or 0)


def wait_for_commit(
    path: Path,
    *,
    expected_iteration: int,
    poll_seconds: float,
) -> dict[str, Any]:
    while True:
        if path.is_file():
            commit = read_json(path)
            loop_state_path = path.parent.parent / "loop_state.json"
            loop_state = (
                read_json(loop_state_path) if loop_state_path.is_file() else {}
            )
            if (
                int(
                    commit.get(
                        "last_completed_iteration",
                        commit.get("iteration", -1),
                    )
                )
                == int(expected_iteration)
                and int(loop_state.get("last_completed_iteration", -1))
                == int(expected_iteration)
                and int(loop_state.get("next_iteration", -1))
                == int(expected_iteration) + 1
            ):
                return commit
        time.sleep(max(0.1, float(poll_seconds)))


def validate_boundary_commit(
    commit: dict[str, Any],
    *,
    expected_iteration: int,
) -> dict[str, Any]:
    """Prove this is the unique immutable commit for the requested boundary."""

    observed_iteration = int(
        commit.get("last_completed_iteration", commit.get("iteration", -1))
    )
    if observed_iteration != int(expected_iteration):
        raise RuntimeError("boundary commit iteration does not match")
    if int(commit.get("next_iteration", -1)) != int(expected_iteration) + 1:
        raise RuntimeError("boundary commit does not advance exactly one iteration")
    rows = [
        dict(row)
        for row in list(commit.get("history") or [])
        if isinstance(row, dict)
        and int(row.get("iteration", -1)) == int(expected_iteration)
    ]
    if len(rows) != 1 or rows[0].get("completed") is not True:
        raise RuntimeError("boundary commit lacks one completed iteration row")
    result = rows[0].get("active_gate_result")
    if not isinstance(result, dict) or int(result.get("iteration", -1)) != int(
        expected_iteration
    ):
        raise RuntimeError("boundary commit lacks its exact active-gate result")
    for field in ("candidate", "learner_after"):
        identity = rows[0].get(field)
        if (
            not isinstance(identity, dict)
            or not str(identity.get("path") or "")
            or not str(identity.get("digest") or "").startswith("sha256:")
        ):
            raise RuntimeError(f"boundary commit lacks a valid {field} identity")
    return rows[0]


def validate_staged_gate(path: Path, *, activate_after: int) -> dict[str, Any]:
    contract = read_json(path)
    fallback = dict(contract.get("fallback_transition") or {})
    primary = dict(contract.get("next_gate") or {})
    criteria = dict(primary.get("pass_criteria") or {})
    if (
        int(fallback.get("activate_after_completed_iteration", -1))
        != int(activate_after)
        or float(fallback.get("skill_weighted_confidence_lower", -1.0))
        != 0.50
        or fallback.get("only_if_prior_gate_unpassed") is not True
        or str(contract.get("active_gate_id") or "")
        != str(primary.get("id") or "")
        or str(fallback.get("prior_gate_id") or "")
        != str(primary.get("id") or "")
        or float(criteria.get("skill_weighted_confidence_lower", -1.0))
        != float(fallback.get("prior_confidence_lower", -2.0))
    ):
        raise RuntimeError("staged LC50 gate contract is not boundary-compatible")
    derived = copy.deepcopy(contract)
    derived_gate = copy.deepcopy(primary)
    derived_gate["id"] = str(fallback["id"])
    derived_gate["pass_criteria"]["skill_weighted_confidence_lower"] = 0.50
    derived["active_gate_id"] = str(fallback["id"])
    derived["next_gate"] = derived_gate
    if (
        not isinstance(derived, dict)
        or str(derived.get("active_gate_id") or "")
        != str(fallback.get("id") or "")
        or str((derived.get("next_gate") or {}).get("id") or "")
        != str(fallback.get("id") or "")
        or float(
            ((derived.get("next_gate") or {}).get("pass_criteria") or {}).get(
                "skill_weighted_confidence_lower", -1.0
            )
        )
        != 0.50
    ):
        raise RuntimeError("staged contract does not materialize the exact LC50 gate")
    return contract


def activate(
    *,
    service: str,
    commit_path: Path,
    expected_iteration: int,
    source: Path,
    target: Path,
    receipt: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    commit = read_json(commit_path)
    loop_state_path = commit_path.parent.parent / "loop_state.json"
    loop_state = read_json(loop_state_path)
    boundary_row = validate_boundary_commit(
        commit,
        expected_iteration=expected_iteration,
    )
    observed_iteration = int(expected_iteration)
    if (
        int(loop_state.get("last_completed_iteration", -1))
        != int(expected_iteration)
        or int(loop_state.get("next_iteration", -1))
        != int(expected_iteration) + 1
    ):
        raise RuntimeError("loop state has not atomically advanced to the boundary")
    staged_contract = validate_staged_gate(
        source,
        activate_after=expected_iteration,
    )
    fallback = dict(staged_contract["fallback_transition"])
    derived = copy.deepcopy(staged_contract)
    derived["active_gate_id"] = str(fallback["id"])
    derived["next_gate"] = copy.deepcopy(staged_contract["next_gate"])
    derived["next_gate"]["id"] = str(fallback["id"])
    derived["next_gate"]["pass_criteria"][
        "skill_weighted_confidence_lower"
    ] = 0.50
    if systemctl("is-active", service, check=False).stdout.strip() != "active":
        raise RuntimeError("managed trainer is not active")

    old_pid = service_pid(service)
    if old_pid <= 0:
        raise RuntimeError("managed trainer has no MainPID")
    original = target.read_bytes()
    source_bytes = source.read_bytes()
    old_digest = sha256(target)
    new_digest = sha256(source)
    commit_digest = sha256(commit_path)
    drop_in = (
        runtime_root
        / f"{service}.d"
        / "99-canonical-boundary-activation.conf"
    )
    atomic_bytes(drop_in, b"[Unit]\nRefuseManualStop=no\n")
    systemctl("daemon-reload")
    started = False
    try:
        systemctl("stop", service)
        atomic_bytes(target, source_bytes)
        try:
            systemctl("start", service)
            started = True
        except BaseException:
            atomic_bytes(target, original)
            systemctl("start", service)
            started = True
            raise
    finally:
        drop_in.unlink(missing_ok=True)
        try:
            drop_in.parent.rmdir()
        except OSError:
            pass
        systemctl("daemon-reload")
        if not started:
            atomic_bytes(target, original)
            systemctl("start", service, check=False)

    deadline = time.monotonic() + 90.0
    new_pid = 0
    while time.monotonic() < deadline:
        if systemctl("is-active", service, check=False).stdout.strip() == "active":
            new_pid = service_pid(service)
            if new_pid > 0 and new_pid != old_pid:
                break
        time.sleep(1.0)
    if new_pid <= 0 or new_pid == old_pid:
        raise RuntimeError("managed trainer did not restart with a new MainPID")
    time.sleep(5.0)
    if (
        systemctl("is-active", service, check=False).stdout.strip() != "active"
        or service_pid(service) != new_pid
    ):
        raise RuntimeError("managed trainer did not remain stable after restart")
    if sha256(target) != new_digest:
        raise RuntimeError("activated gate contract checksum changed")
    if (
        systemctl("show", service, "-p", "RefuseManualStop", "--value")
        .stdout.strip()
        .lower()
        != "yes"
    ):
        raise RuntimeError("manual-stop protection was not restored")
    value = {
        "schema": "poke_bot.gate_contract_boundary_activation/v1",
        "status": "ready",
        "service": service,
        "commit": str(commit_path),
        "commit_sha256": commit_digest,
        "loop_state": str(loop_state_path),
        "loop_state_sha256": sha256(loop_state_path),
        "completed_iteration": observed_iteration,
        "source": str(source),
        "target": str(target),
        "prior_contract_sha256": old_digest,
        "active_contract_sha256": new_digest,
        "configured_primary_gate_id": str(staged_contract["active_gate_id"]),
        "configured_fallback_gate_id": str(fallback["id"]),
        "effective_gate_from_next_iteration": str(derived["active_gate_id"]),
        "effective_confidence_lower_from_next_iteration": float(
            derived["next_gate"]["pass_criteria"][
                "skill_weighted_confidence_lower"
            ]
        ),
        "boundary_gate_id": str(
            (boundary_row.get("active_gate_result") or {}).get("gate_id") or ""
        ),
        "boundary_gate_passed": bool(
            (boundary_row.get("active_gate_result") or {}).get("passed")
        ),
        "old_main_pid": old_pid,
        "new_main_pid": new_pid,
        "completed_boundary_work_replayed_or_discarded": False,
        "manual_stop_protection_restored": True,
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(receipt, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(f"/run/user/{os.getuid()}/systemd/user"),
    )
    args = parser.parse_args()
    if args.receipt.is_file():
        existing = read_json(args.receipt)
        if existing.get("status") == "ready":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
    wait_for_commit(
        args.commit,
        expected_iteration=args.expected_iteration,
        poll_seconds=args.poll_seconds,
    )
    value = activate(
        service=args.service,
        commit_path=args.commit,
        expected_iteration=args.expected_iteration,
        source=args.source,
        target=args.target,
        receipt=args.receipt,
        runtime_root=args.runtime_root,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
