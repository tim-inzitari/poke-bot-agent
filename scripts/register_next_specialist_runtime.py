#!/usr/bin/env python3
"""Atomically register one bootstrapped specialist in the canonical runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from poke_bot.matchup_adapters import EXPERT_IDS


AUTH_SCHEMA = "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1"
REGISTRY_SCHEMA = "poke_bot.specialist_runtime_registry/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_selector(path: Path, specialist_id: str) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()
    key = "POKEBOT_ACTIVE_SPECIALIST="
    replaced = False
    output: list[str] = []
    for row in rows:
        if row.startswith(key):
            if replaced:
                continue
            output.append(key + specialist_id)
            replaced = True
        else:
            output.append(row)
    if not replaced:
        output.append(key + specialist_id)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def register(
    *,
    specialist_id: str,
    family: Path,
    expert: Path,
    runtime_tree: Path,
    runtime_registry: Path,
    selector_env: Path,
    state_root: Path,
    run_name: str,
    handoff_service: str,
    minimum_decisions: int = 20_000,
    required_target_coverage: tuple[str, ...] = (),
) -> dict[str, Any]:
    specialist_id = specialist_id.strip().lower()
    if not specialist_id or not all(
        char.islower() or char.isdigit() or char == "-" for char in specialist_id
    ):
        raise ValueError("unsafe specialist id")
    frozen = verify_frozen_model(family.expanduser().resolve())
    checkpoint_path = Path(str(frozen["model_path"])).resolve()
    checkpoint_digest = str(frozen["checkpoint_digest"])
    expert = expert.expanduser().resolve()
    runtime_tree = runtime_tree.expanduser().resolve()
    runtime_registry = runtime_registry.expanduser().resolve()
    selector_env = selector_env.expanduser().resolve()
    state_root = state_root.expanduser().resolve()
    registry = _read(runtime_registry)
    tree = _read(runtime_tree)
    pointer = _read(expert)
    targets = tuple(str(value) for value in tree.get("targets") or ())
    accepted = {
        str(value)
        for value in (tree.get("runtime_contract") or {}).get(
            "accepted_archetype_ids", ()
        )
    }
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or int((pointer.get("totals") or {}).get("decisions_kept") or 0)
        < int(minimum_decisions)
        or tree.get("runtime_enabled") is not True
        or targets != EXPERT_IDS
        or len(set(targets)) != len(EXPERT_IDS)
        or specialist_id not in targets
        or specialist_id not in accepted
        or (tree.get("runtime_contract") or {}).get("one_route_per_decision")
        is not True
        or (tree.get("runtime_contract") or {}).get("unknown_route_exact_bypass")
        is not True
        or not selector_env.is_file()
        or not handoff_service.startswith("pokebot-")
        or not handoff_service.endswith(".service")
    ):
        raise RuntimeError("next specialist runtime input contract failed")

    manifest = family.resolve() / "manifest.json"
    timestamp = datetime.now(timezone.utc).isoformat()
    authorization_path = (
        state_root / f"{specialist_id}-matchup-adapter-bootstrap-v1.json"
    )
    authorization = {
        "schema": AUTH_SCHEMA,
        "specialist_id": specialist_id,
        "completed_iteration": -1,
        "first_eligible_iteration": 0,
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_digest": checkpoint_digest,
        "protected_manifest": str(manifest),
        "protected_manifest_digest": sha256(manifest),
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "parent_untouched": True,
        "purpose": "specialist-bootstrap-causal-router-aligned-adapter-fitting",
        "required_target_coverage": list(required_target_coverage),
        "created_at_utc": timestamp,
    }
    if authorization_path.is_file():
        if _read(authorization_path) != authorization:
            # The timestamp is evidence, not identity. Permit an idempotent
            # replay only when every substantive field is unchanged.
            existing = _read(authorization_path)
            existing_semantic = {
                key: value
                for key, value in existing.items()
                if key != "created_at_utc"
            }
            expected_semantic = {
                key: value
                for key, value in authorization.items()
                if key != "created_at_utc"
            }
            legacy_expected = dict(expected_semantic)
            legacy_expected.pop("required_target_coverage")
            if existing_semantic == legacy_expected:
                authorization["created_at_utc"] = str(
                    existing.get("created_at_utc") or timestamp
                )
                _atomic_json(authorization_path, authorization)
            elif existing_semantic != expected_semantic:
                raise RuntimeError(
                    "existing adapter authorization identity changed"
                )
            else:
                authorization = existing
    else:
        _atomic_json(authorization_path, authorization)

    row = {
        "status": "ready",
        "reason": None,
        "run_name": run_name,
        "log": f"/home/inzi/poke-bot-agent/outputs/logs/{run_name}.log",
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_sha256": checkpoint_digest.removeprefix("sha256:"),
        "expert_manifest": str(expert),
        "expert_manifest_sha256": sha256(expert).removeprefix("sha256:"),
        "expert_minimum_decisions": int(minimum_decisions),
        "expert_required_target_coverage": list(required_target_coverage),
        "matchup_runtime_tree": str(runtime_tree),
        "matchup_runtime_tree_sha256": sha256(runtime_tree).removeprefix("sha256:"),
        "matchup_adapter_authorization": str(authorization_path),
        "matchup_adapter_authorization_sha256": sha256(
            authorization_path
        ).removeprefix("sha256:"),
        "matchup_adapter_epochs_per_rl_iteration": 1,
        "measurement_decks": specialist_id,
        "guide_loss_weight": 0.0,
        "terminal_gate_marker": f"SPECIALIST_GATE_PASSED.{specialist_id}-splus-v1",
        "pass_handler": {
            "family": f"{specialist_id}-protocol-gate-pass-v1",
            "display_name": f"{specialist_id} Exact Protocol Gate Champion",
            "submission_root": (
                "/home/inzi/poke-bot-agent/outputs/submissions/"
                f"{specialist_id}-protocol-gate-pass-v1"
            ),
            "state": (
                "/home/inzi/poke-bot-agent/outputs/state/"
                f"{specialist_id}-passed-gate-handler-v1.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                f"{specialist_id}-passed-gate-handler-v1.lock"
            ),
            "handoff_service": handoff_service,
        },
    }
    specialists = dict(registry.get("specialists") or {})
    existing_row = specialists.get(specialist_id)
    if existing_row is not None and existing_row != row:
        legacy_row = dict(row)
        legacy_row.pop("expert_minimum_decisions")
        legacy_row.pop("expert_required_target_coverage")
        authorization_migration_row = dict(row)
        authorization_migration_row[
            "matchup_adapter_authorization_sha256"
        ] = str(
            dict(existing_row).get(
                "matchup_adapter_authorization_sha256"
            )
            or ""
        )
        if (
            existing_row != legacy_row
            and existing_row != authorization_migration_row
        ):
            raise RuntimeError(
                "refusing to replace a different specialist runtime row"
            )
    specialists[specialist_id] = row
    registry["specialists"] = specialists
    _atomic_json(runtime_registry, registry)
    _atomic_selector(selector_env, specialist_id)
    receipt = {
        "schema": "poke_bot.specialist_runtime_registration/v1",
        "created_at_utc": timestamp,
        "specialist_id": specialist_id,
        "runtime_row": row,
        "runtime_registry": str(runtime_registry),
        "runtime_registry_sha256": sha256(runtime_registry),
        "selector_env": str(selector_env),
        "selector_env_sha256": sha256(selector_env),
    }
    encoded = json.dumps(
        {key: value for key, value in receipt.items() if key != "created_at_utc"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt["identity_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    receipt_path = state_root / f"{specialist_id}-runtime-registration-v1.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--runtime-tree", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--selector-env", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--handoff-service",
        default="pokebot-specialist-cycle-handoff.service",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            register(
                specialist_id=args.specialist_id,
                family=args.family,
                expert=args.expert,
                runtime_tree=args.runtime_tree,
                runtime_registry=args.runtime_registry,
                selector_env=args.selector_env,
                state_root=args.state_root,
                run_name=args.run_name,
                handoff_service=args.handoff_service,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
