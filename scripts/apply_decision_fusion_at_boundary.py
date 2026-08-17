#!/usr/bin/env python3
"""Register a zero-safe fusion child at one exact committed RL boundary.

This is phase one of activation.  It changes only the mutable learner pointer;
champion, heldout champion, immutable commits, and completed checkpoints remain
unchanged.  Serving stays flat-policy until a later nonzero influence receipt
authorizes phase two.
"""

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
from poke_bot.model import (  # noqa: E402
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)


SCHEMA = "poke_bot.causal_decision_fusion_boundary_warmup/v1"
MATERIALIZATION_SCHEMA = (
    "poke_bot.causal_decision_fusion_checkpoint_migration/v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    if (
        result.returncode
        or values.get("ActiveState") not in {"inactive", "failed"}
        or int(values.get("MainPID") or 0) != 0
    ):
        raise RuntimeError(
            f"managed trainer must be stopped at the exact boundary: {values}"
        )


def apply_boundary(
    *,
    run_dir: Path,
    parent: Path,
    migrated: Path,
    materialization_receipt: Path,
    activation_receipt: Path,
    expected_last_iteration: int,
    service: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    parent = parent.expanduser().resolve()
    migrated = migrated.expanduser().resolve()
    materialization_receipt = materialization_receipt.expanduser().resolve()
    activation_receipt = activation_receipt.expanduser().resolve()
    _assert_service_stopped(service)

    loop_path = run_dir / "loop_state.json"
    commit_path = (
        run_dir / "commits" / f"iter_{expected_last_iteration:05d}.json"
    )
    state = _read_json(loop_path)
    commit = _read_json(commit_path)
    next_iteration = expected_last_iteration + 1
    if state != commit or not (
        int(state.get("last_completed_iteration", -1))
        == expected_last_iteration
        and int(state.get("next_iteration", -1)) == next_iteration
    ):
        raise RuntimeError("fusion warmup requires one exact clean boundary")

    parent_digest = checkpoint.checkpoint_digest(parent)
    migrated_digest = checkpoint.checkpoint_digest(migrated)
    current = dict(state.get("learner") or {})
    if (
        Path(str(current.get("path") or "")).expanduser().resolve() != parent
        or str(current.get("digest") or "") != parent_digest
    ):
        if (
            str(current.get("digest") or "") == migrated_digest
            and activation_receipt.is_file()
        ):
            return _read_json(activation_receipt)
        raise RuntimeError("boundary learner is not the migration parent")

    materialization = _read_json(materialization_receipt)
    checkpoint.assert_trusted_policy_checkpoint(migrated)
    payload = checkpoint.load_checkpoint(migrated, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    fusion = dict((payload.get("provenance") or {}).get("decision_fusion") or {})
    migration = dict((payload.get("extra") or {}).get(
        "decision_fusion_migration"
    ) or {})
    if not (
        materialization.get("schema") == MATERIALIZATION_SCHEMA
        and materialization.get("parent_checkpoint_digest") == parent_digest
        and materialization.get("migrated_checkpoint_digest")
        == migrated_digest
        and dict(materialization.get("proof") or {}).get(
            "legacy_tensors_bit_identical"
        ) is True
        and dict(materialization.get("proof") or {}).get(
            "optimizer_existing_state_preserved"
        ) is True
        and model_config.get("decision_fusion_enabled") is True
        and model_config.get("decision_fusion_runtime_enabled") is False
        and fusion.get("schema") == DECISION_FUSION_SCHEMA
        and fusion.get("enabled") is True
        and fusion.get("runtime_enabled") is False
        and fusion.get("required_heads")
        == list(DECISION_FUSION_REQUIRED_HEADS)
        and migration.get("source_checkpoint_digest") == parent_digest
        and migration.get("zero_safe_initialization") is True
        and migration.get("serving_eligible") is False
    ):
        raise RuntimeError("migrated checkpoint lacks the zero-safe fusion proof")

    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "boundary": {
            "last_completed_iteration": expected_last_iteration,
            "next_iteration": next_iteration,
            "commit": str(commit_path),
            "commit_digest": _sha256(commit_path),
        },
        "parent_learner": {"path": str(parent), "digest": parent_digest},
        "warmup_learner": {"path": str(migrated), "digest": migrated_digest},
        "materialization_receipt": {
            "path": str(materialization_receipt),
            "digest": _sha256(materialization_receipt),
        },
        "decision_fusion": {
            "schema": DECISION_FUSION_SCHEMA,
            "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
            "training_enabled": True,
            "runtime_enabled": False,
            "serving_eligible": False,
            "next_phase": "train_in_ordinary_full_model_rl_then_validate",
        },
        "champion_replaced": False,
        "heldout_champion_replaced": False,
        "immutable_commit_modified": False,
    }
    if not publish:
        return {**receipt, "validation_only": True}
    _exclusive_json(activation_receipt, receipt)

    updated = copy.deepcopy(state)
    updated["learner"] = {"path": str(migrated), "digest": migrated_digest}
    fit = dict(updated.get("dormant_matchup_adapter_fit") or {})
    if fit:
        fit["checkpoint_path"] = str(migrated)
        fit["checkpoint_digest"] = migrated_digest
        updated["dormant_matchup_adapter_fit"] = fit
    updated["decision_fusion_activation"] = {
        "schema": SCHEMA,
        "phase": "training_warmup",
        "boundary_next_iteration": next_iteration,
        "learner_digest": migrated_digest,
        "runtime_enabled": False,
        "serving_eligible": False,
        "receipt": str(activation_receipt),
        "receipt_digest": _sha256(activation_receipt),
    }
    _atomic_json(loop_path, updated)
    if dict(_read_json(loop_path).get("learner") or {}) != updated["learner"]:
        raise RuntimeError("fusion learner pointer publication did not verify")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, required=True)
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
