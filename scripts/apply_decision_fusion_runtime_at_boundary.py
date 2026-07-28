#!/usr/bin/env python3
"""Activate an audited fusion checkpoint at one exact committed boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import DECISION_FUSION_REQUIRED_HEADS  # noqa: E402


SCHEMA = "poke_bot.causal_decision_fusion_runtime_boundary/v1"
MATERIALIZATION_SCHEMA = (
    "poke_bot.causal_decision_fusion_runtime_materialization/v1"
)
VALIDATION_SCHEMA = "poke_bot.causal_decision_fusion_activation_validation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _assert_service_stopped(service: str | None) -> None:
    if not service:
        return
    result = subprocess.run(
        [
            "systemctl", "--user", "show", service,
            "-p", "ActiveState", "-p", "MainPID",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if (
        result.returncode
        or values.get("ActiveState") not in {"inactive", "failed"}
        or int(values.get("MainPID") or 0) != 0
    ):
        raise RuntimeError(f"trainer must be stopped at exact boundary: {values}")


def apply_boundary(
    *,
    run_dir: Path,
    trained: Path,
    runtime_checkpoint: Path,
    validation_receipt: Path,
    materialization_receipt: Path,
    activation_receipt: Path,
    expected_last_iteration: int,
    service: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    trained = trained.expanduser().resolve()
    runtime_checkpoint = runtime_checkpoint.expanduser().resolve()
    validation_receipt = validation_receipt.expanduser().resolve()
    materialization_receipt = materialization_receipt.expanduser().resolve()
    activation_receipt = activation_receipt.expanduser().resolve()
    _assert_service_stopped(service)

    loop_path = run_dir / "loop_state.json"
    commit_path = run_dir / "commits" / f"iter_{expected_last_iteration:05d}.json"
    state = _read(loop_path)
    commit = _read(commit_path)
    if state != commit or not (
        int(state.get("last_completed_iteration", -1)) == expected_last_iteration
        and int(state.get("next_iteration", -1)) == expected_last_iteration + 1
    ):
        raise RuntimeError("runtime activation requires one exact clean boundary")

    trained_digest = checkpoint.checkpoint_digest(trained)
    runtime_digest = checkpoint.checkpoint_digest(runtime_checkpoint)
    current = dict(state.get("learner") or {})
    if (
        str(current.get("digest") or "") == runtime_digest
        and activation_receipt.is_file()
    ):
        return _read(activation_receipt)
    if not (
        Path(str(current.get("path") or "")).expanduser().resolve() == trained
        and str(current.get("digest") or "") == trained_digest
    ):
        raise RuntimeError("boundary learner is not the audited fusion checkpoint")

    validation = _read(validation_receipt)
    materialization = _read(materialization_receipt)
    checkpoint.assert_trusted_policy_checkpoint(runtime_checkpoint)
    payload = checkpoint.load_checkpoint(runtime_checkpoint, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    fusion = dict((payload.get("provenance") or {}).get("decision_fusion") or {})
    if not (
        validation.get("schema") == VALIDATION_SCHEMA
        and validation.get("checkpoint_digest") == trained_digest
        and validation.get("passed") is True
        and materialization.get("schema") == MATERIALIZATION_SCHEMA
        and materialization.get("source_checkpoint_digest") == trained_digest
        and materialization.get("runtime_checkpoint_digest") == runtime_digest
        and materialization.get("validation_receipt_digest")
        == _sha256(validation_receipt)
        and materialization.get("model_tensors_bit_identical_to_validated_source")
        is True
        and materialization.get(
            "optimizer_state_bit_identical_to_validated_source"
        )
        is True
        and model_config.get("decision_fusion_enabled") is True
        and model_config.get("decision_fusion_runtime_enabled") is True
        and fusion.get("required_heads") == list(DECISION_FUSION_REQUIRED_HEADS)
    ):
        raise RuntimeError("runtime checkpoint lacks the complete activation proof")

    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "boundary": {
            "last_completed_iteration": expected_last_iteration,
            "next_iteration": expected_last_iteration + 1,
            "commit": str(commit_path),
            "commit_digest": _sha256(commit_path),
        },
        "trained_learner": {"path": str(trained), "digest": trained_digest},
        "runtime_learner": {
            "path": str(runtime_checkpoint),
            "digest": runtime_digest,
        },
        "validation_receipt": {
            "path": str(validation_receipt),
            "digest": _sha256(validation_receipt),
        },
        "materialization_receipt": {
            "path": str(materialization_receipt),
            "digest": _sha256(materialization_receipt),
        },
        "decision_fusion": {
            "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
            "runtime_enabled": True,
            "serving_eligible": True,
            "future_specialists_required": True,
        },
        "champion_replaced": False,
        "heldout_champion_replaced": False,
        "immutable_commit_modified": False,
    }
    if not publish:
        return {**receipt, "validation_only": True}
    _exclusive_json(activation_receipt, receipt)
    updated = copy.deepcopy(state)
    updated["learner"] = {"path": str(runtime_checkpoint), "digest": runtime_digest}
    fit = dict(updated.get("dormant_matchup_adapter_fit") or {})
    if fit:
        fit["checkpoint_path"] = str(runtime_checkpoint)
        fit["checkpoint_digest"] = runtime_digest
        updated["dormant_matchup_adapter_fit"] = fit
    updated["decision_fusion_activation"] = {
        "schema": SCHEMA,
        "phase": "runtime_active",
        "boundary_next_iteration": expected_last_iteration + 1,
        "learner_digest": runtime_digest,
        "runtime_enabled": True,
        "serving_eligible": True,
        "receipt": str(activation_receipt),
        "receipt_digest": _sha256(activation_receipt),
    }
    _atomic_json(loop_path, updated)
    if dict(_read(loop_path).get("learner") or {}) != updated["learner"]:
        raise RuntimeError("runtime fusion learner publication did not verify")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--runtime-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--expected-last-iteration", type=int, required=True)
    parser.add_argument("--service")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["publish"] = not values.pop("validate_only")
    print(json.dumps(apply_boundary(**values), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
