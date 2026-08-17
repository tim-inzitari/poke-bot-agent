#!/usr/bin/env python3
"""Attempt the normal post-Alakazam core refresh, then release Marnie H10."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from poke_bot import checkpoint
from scripts.run_post_starmie_core_handoff import _resolve_boundary_core
from scripts.run_specialist_cycle_handoff import _publish_latest_core_pointer


BOUNDARY_SCHEMA = "poke_bot.post_alakazam_core_refresh_boundary/v1"
POINTER_SCHEMA = "poke_bot.latest_cumulative_core_pointer/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return checkpoint.checkpoint_digest(path)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def _materialize_contract(
    *, template_path: Path, registry_path: Path, pointer_path: Path, output: Path
) -> dict[str, Any]:
    template = _read(template_path)
    registry = _read(registry_path)
    pointer = _read(pointer_path)
    refreshes = [
        dict(row)
        for row in registry.get("refreshes") or []
        if dict(row).get("specialist_id") == "alakazam"
    ]
    if (
        template.get("schema") != "poke_bot.post_specialist_core_refresh_handoff/v1"
        or registry.get("schema") != "poke_bot.post_fleet_refresh_registry/v1"
        or len(refreshes) != 1
        or pointer.get("schema") != POINTER_SCHEMA
    ):
        raise RuntimeError("post-Alakazam core inputs are not authoritative")
    refreshed = refreshes[0]
    refreshed_checkpoint = Path(str(refreshed.get("checkpoint") or "")).resolve()
    refreshed_digest = str(refreshed.get("checkpoint_checksum") or "")
    if not refreshed_checkpoint.is_file() or _sha256(refreshed_checkpoint) != refreshed_digest:
        raise RuntimeError("registered Alakazam refresh identity changed")

    contract = copy.deepcopy(template)
    core = dict(contract["core_refresh"])
    teachers = [dict(row) for row in core.get("teachers") or []]
    matches = [index for index, row in enumerate(teachers) if row.get("specialist_id") == "alakazam"]
    if len(matches) != 1:
        raise RuntimeError("normal cumulative core does not have one Alakazam teacher")
    teachers[matches[0]] = {
        "specialist_id": "alakazam",
        "mode": "frozen_inference_only",
        "checkpoint": str(refreshed_checkpoint),
        "checksum": refreshed_digest,
    }
    generation = 15
    stem = f"deck-agnostic-core-cumulative-v{generation}-post-alakazam-r104-fused-v1"
    state_root = output.parent
    core.update(
        {
            "version": generation,
            "family": f"/home/pokebot/poke-bot-agent/outputs/pure_rl/_protected/models/{stem.replace('-', '_')}",
            "initialization": {
                "checkpoint": str(Path(str(pointer["family"])) / "model.pt"),
                "checksum": str(pointer["checkpoint_digest"]),
            },
            "teachers": teachers,
            "ready_receipt": str(state_root / f"{stem}-ready.json"),
            "run_name": stem.replace("-", "_"),
            "run_dir": f"/home/pokebot/poke-bot-agent/outputs/bootstrap/{stem}",
            "cpu_pack_root": f"/home/pokebot/poke-bot-agent/outputs/bootstrap/cpu-packs/{stem}",
            "refresh_attempt": 1,
            "split_seed": 20260801,
        }
    )
    contract["core_refresh"] = core
    contract["status"] = "staged_post_alakazam_refresh"
    contract["trigger"] = {
        "specialist_id": "alakazam",
        "refresh_model_version": refreshed.get("refresh_model_version"),
        "completion_authority": refreshed.get("completion_authority"),
        "checkpoint": str(refreshed_checkpoint),
        "checkpoint_sha256": refreshed_digest,
        "requires_training_complete": True,
        "requires_exact_frozen_checkpoint": True,
    }
    contract["acceptance"]["regression_result"] = str(
        state_root / f"{stem}-gameplay-regression.json"
    )
    fallback = dict(contract["core_failure_fallback"])
    fallback.update(
        {
            "version": int(pointer["version"]),
            "family": str(pointer["family"]),
            "checkpoint_digest": str(pointer["checkpoint_digest"]),
            "ready_receipt": str(pointer["ready"]),
        }
    )
    contract["core_failure_fallback"] = fallback
    _write_once(output, contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--refresh-registry", type=Path, required=True)
    parser.add_argument("--latest-core", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--next-service", required=True)
    args = parser.parse_args()
    paths = [
        args.template,
        args.refresh_registry,
        args.latest_core,
        args.contract,
        args.state,
        args.lock,
        args.receipt,
    ]
    template_path, registry_path, pointer_path, contract_path, state_path, lock_path, receipt_path = [
        path.expanduser().resolve() for path in paths
    ]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt_path.is_file():
            receipt = _read(receipt_path)
            if receipt.get("schema") != BOUNDARY_SCHEMA:
                raise RuntimeError("post-Alakazam boundary receipt schema changed")
        else:
            contract = _materialize_contract(
                template_path=template_path,
                registry_path=registry_path,
                pointer_path=pointer_path,
                output=contract_path,
            )
            pointer_before = _read(pointer_path)
            attempted = dict(contract["core_refresh"])
            runtime = {
                "python": "/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python",
                "registry_root": "/home/pokebot/poke-bot-agent/outputs/pure_rl/_protected/models",
            }
            ready, frozen, selected, _resolved = _resolve_boundary_core(
                contract_path=contract_path,
                contract=contract,
                runtime=runtime,
                state_path=state_path,
            )
            attempted_ready_path = Path(str(attempted["ready_receipt"]))
            attempted_ready = _read(attempted_ready_path)
            accepted = (
                attempted_ready.get("status") == "ready"
                and attempted_ready.get("gameplay_regression_passed") is True
            )
            if accepted:
                _publish_latest_core_pointer(
                    pointer_path,
                    family=Path(str(attempted["family"])),
                    ready_path=attempted_ready_path,
                    previous_digest=str(pointer_before["checkpoint_digest"]),
                )
            pointer_after = _read(pointer_path)
            receipt = {
                "schema": BOUNDARY_SCHEMA,
                "status": "selected_core_ready_for_marnie_h10",
                "predecessor_refresh": "alakazam",
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "normal_core_refresh_attempted": True,
                "attempted_core_generation": int(attempted["version"]),
                "attempted_core_contract": str(contract_path),
                "attempted_core_contract_sha256": _sha256(contract_path),
                "attempt_accepted": accepted,
                "attempt_ready_receipt": str(attempted_ready_path),
                "attempt_ready_receipt_sha256": _sha256(attempted_ready_path),
                "rejected_candidate_blocks_production": False,
                "selected_core_generation": int(pointer_after["version"]),
                "selected_core_checkpoint_sha256": str(pointer_after["checkpoint_digest"]),
                "latest_core_pointer": str(pointer_path),
                "latest_core_pointer_sha256": _sha256(pointer_path),
                "resolver_selected_checkpoint_sha256": str(frozen["checkpoint_digest"]),
                "resolver_selected_ready_status": str(ready.get("status") or ""),
                "training_authority": False,
                "selector_authority": False,
            }
            _write_once(receipt_path, receipt)
        started = subprocess.run(
            ["systemctl", "--user", "start", args.next_service],
            check=False,
            capture_output=True,
            text=True,
        )
        if started.returncode:
            raise RuntimeError(f"could not start Marnie H10 materializer: {started.stdout}{started.stderr}")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
