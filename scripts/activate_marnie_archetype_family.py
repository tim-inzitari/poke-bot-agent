#!/usr/bin/env python3
"""Managed one-shot atomic installer for the staged Marnie family contract.

This program never restarts a service.  A declared service manager may invoke
it only after the trainer's immutable clean-boundary pause receipt exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import (  # noqa: E402
    MIGRATION_SCHEMA,
    PAUSE_SCHEMA,
    REQUEST_SCHEMA,
    sha256,
    validate_atomic_migration,
    validate_activation_request,
)


def _read_exact(path: Path, digest: str) -> dict:
    if not path.is_file() or sha256(path) != digest:
        raise RuntimeError(f"digest mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def activate(
    *, pause: Path, request: Path, before_registry: Path,
    candidate_registry: Path, output_registry: Path,
    environment_drop_in: Path, receipt: Path,
    python_executable: Path, launcher: Path,
) -> dict:
    pause_payload = json.loads(pause.read_text(encoding="utf-8"))
    if pause_payload.get("schema") != PAUSE_SCHEMA or pause_payload.get("next_collection_started") is not False:
        raise RuntimeError("activation lacks an exact clean-boundary pause")
    if pause_payload.get("request_sha256") != sha256(request):
        raise RuntimeError("pause receipt binds a different request")
    request_payload = json.loads(request.read_text(encoding="utf-8"))
    validate_activation_request(
        request_payload,
        expected_learner_digest=str(pause_payload.get("learner_sha256") or ""),
    )
    bindings = request_payload.get("bindings") or {}
    before = _read_exact(before_registry, str((bindings.get("registry") or {}).get("sha256", "")))
    candidate = json.loads(candidate_registry.read_text(encoding="utf-8"))
    validate_atomic_migration(before, candidate)
    sealed = request_payload.get("sealed_pre_activation") or {}
    manifest = request_payload.get("manifest") or {}
    loss_contract = request_payload.get("loss_contract") or {}
    selected_vector = (request_payload.get("bindings") or {}).get(
        "selected_loss_vector"
    ) or {}
    paths = {
        "POKEBOT_ARCHETYPE_FAMILY_MANIFEST": str(
            Path(str(manifest.get("path") or "")).resolve()
        ),
        "POKEBOT_ARCHETYPE_LOSS_CONTRACT": str(
            Path(str(loss_contract.get("path") or "")).resolve()
        ),
        "POKEBOT_ARCHETYPE_LOSS_VECTOR": str(
            Path(str(selected_vector.get("path") or "")).resolve()
        ),
    }
    if any("\n" in value or not Path(value).is_file() for value in paths.values()):
        raise RuntimeError("activation environment contains an invalid artifact path")
    for executable_path in (python_executable, launcher):
        if not executable_path.is_file() or any(
            character in str(executable_path) for character in ("\n", " ")
        ):
            raise RuntimeError("activation launcher identity is invalid")
    environment_bytes = (
        "[Service]\n"
        + "".join(
            f"Environment={name}={value}\n" for name, value in paths.items()
        )
        + "ExecStart=\n"
        + f"ExecStart={python_executable} -u {launcher} --registry {output_registry}\n"
    ).encode("utf-8")
    if receipt.exists():
        existing = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != MIGRATION_SCHEMA
            or existing.get("status") != "activated_atomically"
            or sha256(output_registry)
            != existing.get("activated_registry_sha256")
            or sha256(environment_drop_in)
            != existing.get("environment_drop_in_sha256")
        ):
            raise RuntimeError("existing activation receipt is invalid")
        return existing
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    candidate_bytes = (
        json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if output_registry.exists():
        if output_registry.read_bytes() != candidate_bytes:
            raise RuntimeError("partially installed registry identity changed")
    else:
        temporary = output_registry.with_suffix(output_registry.suffix + ".partial")
        temporary.write_bytes(candidate_bytes)
        os.replace(temporary, output_registry)
    environment_drop_in.parent.mkdir(parents=True, exist_ok=True)
    if environment_drop_in.exists():
        if environment_drop_in.read_bytes() != environment_bytes:
            raise RuntimeError("partially installed family environment changed")
    else:
        environment_temporary = environment_drop_in.with_suffix(
            environment_drop_in.suffix + ".partial"
        )
        environment_temporary.write_bytes(environment_bytes)
        os.replace(environment_temporary, environment_drop_in)
    payload = {
        "schema": MIGRATION_SCHEMA,
        "status": "activated_atomically",
        "pause_receipt": str(pause),
        "pause_receipt_sha256": sha256(pause),
        "request": str(request),
        "request_sha256": sha256(request),
        "before_registry_sha256": sha256(before_registry),
        "activated_registry": str(output_registry),
        "activated_registry_sha256": sha256(output_registry),
        "environment_drop_in": str(environment_drop_in),
        "environment_drop_in_sha256": sha256(environment_drop_in),
        "atomic_environment": paths,
        "managed_exec_start": {
            "python": str(python_executable),
            "launcher": str(launcher),
            "launcher_sha256": sha256(launcher),
            "registry": str(output_registry),
        },
        "sealed_pre_activation": sealed,
        "restart_authority": "declared_service_manager_only",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "pause", "request", "before-registry", "candidate-registry",
        "output-registry", "environment-drop-in", "receipt",
        "python-executable", "launcher",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = activate(
        pause=args.pause.resolve(), request=args.request.resolve(),
        before_registry=args.before_registry.resolve(),
        candidate_registry=args.candidate_registry.resolve(),
        output_registry=args.output_registry.resolve(),
        environment_drop_in=args.environment_drop_in.resolve(),
        receipt=args.receipt.resolve(),
        python_executable=args.python_executable.resolve(),
        launcher=args.launcher.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
