#!/usr/bin/env python3
"""Fail-closed verifier for real Elmo r298 simulator-engine evidence.

This is intentionally *not* a simulator launcher.  The local test suite uses
an injectable selected-action fixture to cover public target semantics, while
this verifier accepts a result only after an Elmo job has run the pinned
competition simulator and emitted immutable per-case witnesses.  It performs
no training, staging, service control, submission, or policy activation.

Run without ``--verify`` to print the exact manifest that an Elmo test job
must bind.  ``--verify`` fails closed unless it is executed on the canonical
Elmo host and all witness artifacts are physically rehashed below a declared
read-only artifact root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOAL_PATH = PROJECT_ROOT / "goals/alakazam-elmo-rule-derivative/GOAL.md"
CONTRACT_PATH = PROJECT_ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
DEFAULT_ARTIFACT_ROOT = Path(
    "/srv/poke-bot-agent/outputs/experiments/alakazam-elmo-rule-derivative-g1"
)
MANIFEST_SCHEMA = "poke_bot.alakazam_simulator_rules_r298_elmo_engine_manifest/v1"
ENVELOPE_SCHEMA = "poke_bot.alakazam_simulator_rules_r298_elmo_engine_envelope/v1"
CANONICAL_HOSTNAME = "truenas"
HOST_VERIFICATION = "exact_socket_hostname_and_systemd_detect_virt_non_container"


def _load_validation_module() -> Any:
    # Script entry points are often launched outside the repository root.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from poke_bot import alakazam_simulator_rules_r298_validation as validation

    return validation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _current_contract() -> Mapping[str, Any]:
    try:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read canonical derivative contract") from exc
    contract = _mapping(value, label="canonical derivative contract")
    if contract.get("goal_revision") != 5:
        raise ValueError("canonical derivative contract is not the required rev5 contract")
    return contract


def engine_manifest() -> dict[str, Any]:
    """Return the immutable contract an actual Elmo engine job must bind."""

    validation = _load_validation_module()
    contract = _current_contract()
    return {
        "schema": MANIFEST_SCHEMA,
        "goal_gateway_path": str(GOAL_PATH.relative_to(PROJECT_ROOT)),
        "goal_gateway_sha256": _sha256_file(GOAL_PATH),
        "goal_contract_path": str(CONTRACT_PATH.relative_to(PROJECT_ROOT)),
        "goal_contract_sha256": _sha256_file(CONTRACT_PATH),
        "goal_revision": contract["goal_revision"],
        "canonical_simulator": {
            "libcg_sha256": validation.CANONICAL_LIBCG_SHA256,
            "kaggle_environments_version": "1.32.6",
            "legal_option_list_authority": "pinned_competition_simulator",
        },
        "execution_identity_required": {
            "execution_host_role": "elmo",
            "canonical_execution_hostname": CANONICAL_HOSTNAME,
            "execution_hostname": CANONICAL_HOSTNAME,
            "execution_fqdn": "nonempty_string",
            "host_verification": HOST_VERIFICATION,
            "container_execution_permitted": False,
        },
        "case_names": list(validation.ENGINE_CASE_NAMES),
        "fixture_only_is_engine_evidence": False,
        "required_case_witnesses": "read_only_content_addressed_file_identity_per_case",
        "authority": {
            "training": False,
            "runtime": False,
            "service_control": False,
            "submission": False,
            "production": False,
        },
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_readonly_identity(value: object, *, label: str, root: Path) -> dict[str, Any]:
    row = _mapping(value, label=label)
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} identity fields differ")
    raw_path = row.get("path")
    raw_sha = row.get("sha256")
    raw_size = row.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be nonempty")
    if not isinstance(raw_sha, str) or not raw_sha.startswith("sha256:") or len(raw_sha) != 71:
        raise ValueError(f"{label}.sha256 is malformed")
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise ValueError(f"{label}.size_bytes is malformed")
    candidate = Path(raw_path).expanduser()
    # Resolve only after rejecting any symlink at each component.  A witness
    # is evidence, not an arbitrary path handle a caller may redirect.
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular nonsymlink file")
    resolved = candidate.resolve()
    if not _is_within(resolved, root):
        raise ValueError(f"{label} lies outside the declared artifact root")
    cursor = root
    relative = resolved.relative_to(root)
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} has a symlinked path component")
    metadata = resolved.stat()
    if stat.S_ISREG(metadata.st_mode) is False or metadata.st_mode & 0o222:
        raise ValueError(f"{label} must be immutable/read-only")
    if metadata.st_size != raw_size:
        raise ValueError(f"{label} size does not match")
    actual_sha = _sha256_file(resolved)
    if actual_sha != raw_sha:
        raise ValueError(f"{label} digest does not match")
    return {"path": str(resolved), "sha256": actual_sha, "size_bytes": metadata.st_size}


def _verify_live_elmo_identity(identity: Mapping[str, Any]) -> None:
    expected = engine_manifest()["execution_identity_required"]
    if dict(identity) != {
        **expected,
        "execution_fqdn": identity.get("execution_fqdn"),
    }:
        raise ValueError("execution identity fields differ from the canonical Elmo contract")
    fqdn = identity.get("execution_fqdn")
    if not isinstance(fqdn, str) or not fqdn:
        raise ValueError("execution_fqdn must be nonempty")
    if socket.gethostname() != CANONICAL_HOSTNAME:
        raise ValueError("real engine verification is Elmo-only and requires hostname truenas")
    # systemd-detect-virt returns 1 for no detected container.  Any other
    # result is not evidence that the real host process ran bare-metal.
    try:
        probe = subprocess.run(
            ["systemd-detect-virt", "--container"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot prove non-container Elmo execution") from exc
    if probe.returncode != 1:
        raise ValueError("engine verifier refuses container or unknown virtualization state")


def verify_engine_envelope(path: Path | str, *, artifact_root: Path | str) -> dict[str, Any]:
    """Verify a real pinned-engine envelope and every case witness on disk."""

    validation = _load_validation_module()
    root = Path(artifact_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("artifact root must be a real directory")
    envelope_path = Path(path).expanduser()
    if envelope_path.is_symlink() or not envelope_path.is_file():
        raise ValueError("engine envelope must be a regular file")
    try:
        envelope = _mapping(json.loads(envelope_path.read_text(encoding="utf-8")), label="engine envelope")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read engine envelope") from exc
    required = {
        "schema",
        "manifest_sha256",
        "goal_gateway_sha256",
        "goal_contract_sha256",
        "goal_revision",
        "execution_identity",
        "engine_evidence",
        "case_witness_identities",
    }
    if set(envelope) != required or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ValueError("engine envelope schema or exact field inventory differs")
    manifest = engine_manifest()
    if envelope.get("manifest_sha256") != _canonical_json_digest(manifest):
        raise ValueError("engine envelope does not bind the current exact manifest")
    for name in ("goal_gateway_sha256", "goal_contract_sha256", "goal_revision"):
        if envelope.get(name) != manifest[name]:
            raise ValueError(f"engine envelope binds wrong {name}")
    execution_identity = _mapping(envelope.get("execution_identity"), label="execution identity")
    _verify_live_elmo_identity(execution_identity)
    evidence = _mapping(envelope.get("engine_evidence"), label="engine evidence")
    normalized = validation.validate_engine_evidence(evidence)
    identities = _mapping(envelope.get("case_witness_identities"), label="case witness identities")
    case_names = set(validation.ENGINE_CASE_NAMES)
    if set(identities) != case_names:
        raise ValueError("engine envelope witness inventory differs from manifest")
    witnesses = _mapping(evidence.get("case_witnesses"), label="engine evidence witnesses")
    for case_name in validation.ENGINE_CASE_NAMES:
        identity = _strict_readonly_identity(
            identities[case_name],
            label=f"case witness {case_name}",
            root=root,
        )
        if witnesses.get(case_name) != identity["sha256"]:
            raise ValueError(f"case witness {case_name} does not bind its physical artifact")
    return {
        "schema": ENVELOPE_SCHEMA,
        "status": "passed_pinned_engine_verified_elmo_only",
        "manifest_sha256": _canonical_json_digest(manifest),
        "engine_evidence": normalized,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="real Elmo engine evidence envelope to rehash")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="sealed Elmo artifact root containing each immutable case witness",
    )
    arguments = parser.parse_args(argv)
    if arguments.verify is None:
        print(json.dumps({"status": "manifest_only_no_engine_execution", "manifest": engine_manifest()}, sort_keys=True))
        return 2
    try:
        result = verify_engine_envelope(arguments.verify, artifact_root=arguments.artifact_root)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed_closed", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point.
    raise SystemExit(main())
