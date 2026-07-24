#!/usr/bin/env python3
"""Register Trevenant as S+ and atomically stage canonical Starmie RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from scripts.materialize_frozen_specialist_gate import (
    _build_baseline_manifest,
    _build_gate,
    _build_registry,
    _materialize_package,
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trevenant-family", type=Path, required=True)
    parser.add_argument("--trevenant-acceptance", type=Path, required=True)
    parser.add_argument("--starmie-family", type=Path, required=True)
    parser.add_argument("--starmie-expert", type=Path, required=True)
    parser.add_argument("--runtime-tree", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    state_root = args.state_root.resolve()
    registry_path = runtime_root / "ops/frozen_specialist_registry_v1.json"
    gate_path = runtime_root / "ops/alakazam_gate_program_v1.json"
    runtime_registry_path = runtime_root / "ops/specialist_runtime_registry_v1.json"
    base_registry = read_json(registry_path)
    base_gate = read_json(gate_path)
    base_manifest = read_json(args.baseline_manifest)
    runtime_registry = read_json(runtime_registry_path)

    acceptance = read_json(args.trevenant_acceptance)
    trevenant = verify_frozen_model(args.trevenant_family)
    trevenant_digest = "sha256:462f201f8de6c07eef07b3e8f58229360972d1d64308db9c155f211d2ce3faf1"
    bundle_digest = sha256(args.bundle)
    if (
        acceptance.get("schema")
        != "poke_bot.operator_accepted_specialist_floor_transition/v1"
        or int(acceptance.get("completed_iteration_index", -1)) != 10
        or int(acceptance.get("completed_iteration_count", -1)) != 11
        or acceptance.get("formal_active_gate_passed") is not True
        or acceptance.get("checkpoint_digest") != trevenant_digest
        or trevenant.get("checkpoint_digest") != trevenant_digest
        or bundle_digest
        != "sha256:6229e3dd9840268e5cd35b18516a56dc97440cd36d3372f8bc108a0536fb9231"
    ):
        raise RuntimeError("Trevenant owner-accepted floor identity changed")

    timestamp = datetime.now(timezone.utc).isoformat()
    baseline_dir = "hops-trevenant-gate-iter10-462f201f8de6"
    opponent_id = f"specialist-{baseline_dir}"
    package, content_digest, tree_digest = _materialize_package(
        bundle_path=args.bundle,
        bundle_digest=bundle_digest,
        checkpoint_digest=trevenant_digest,
        baseline_root=args.baseline_root,
        baseline_dir=baseline_dir,
    )
    specialist_row = {
        "specialist_id": "hops-trevenant",
        "opponent_id": opponent_id,
        "archetype_id": "hops-trevenant",
        "archetype_label": "Frozen Hop's Trevenant specialist",
        "baseline_dir": baseline_dir,
        "baseline_group": "specialists",
        "checkpoint_digest": trevenant_digest,
        "content_digest": content_digest,
        "matchup_tree_checksum": tree_digest,
        "frozen": True,
        "public_mix_eligible": True,
        "research_eligible": False,
        "registered_at_utc": timestamp,
        "source": (
            "explicit owner-accepted formal protocol gate at zero-indexed "
            "iteration 10 (eleven completed iterations)"
        ),
    }
    frozen_registry = _build_registry(
        base=base_registry,
        specialist_row=specialist_row,
        timestamp=timestamp,
    )
    next_gate = _build_gate(
        base=base_gate,
        registry=frozen_registry,
        timestamp=timestamp,
    )
    stale_ids = {
        str(row.get("opponent_id") or "")
        for row in base_registry.get("specialists", [])
        if str(row.get("specialist_id") or "") == "hops-trevenant"
    }
    stale_ids.add(opponent_id)
    baseline_manifest = _build_baseline_manifest(
        current=base_manifest,
        stale_opponent_ids=stale_ids,
        manifest_row={
            "id": opponent_id,
            "name": "Frozen Hop's Trevenant Specialist · Iteration 10",
            "dir": baseline_dir,
            "group": "specialists",
            "source": specialist_row["source"],
        },
    )

    starmie = verify_frozen_model(args.starmie_family)
    starmie_digest = (
        "sha256:5d835ecf524d3ca8c48e84e3392263f816728bfba85de0a4a42f823dd9dfc7de"
    )
    starmie_manifest = args.starmie_family / "manifest.json"
    starmie_expert_digest = sha256(args.starmie_expert)
    runtime_tree_digest = sha256(args.runtime_tree)
    if (
        starmie.get("checkpoint_digest") != starmie_digest
        or starmie_expert_digest
        != "sha256:979f405c2dc1c4b1c6bcbbfcf5a30a0a467d1e04d744b4cc5d4840fbac1ca9ae"
        or runtime_tree_digest
        != "sha256:0bbbd1075c0c2058e07be6723c2f2bb7902193ce3132613e70d354c132f75c3d"
    ):
        raise RuntimeError("Starmie launch input identity changed")
    authorization_path = state_root / "starmie-matchup-adapter-bootstrap-v1.json"
    authorization = {
        "schema": "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1",
        "specialist_id": "starmie",
        "completed_iteration": -1,
        "first_eligible_iteration": 0,
        "parent_checkpoint": starmie["model_path"],
        "parent_checkpoint_digest": starmie_digest,
        "protected_manifest": str(starmie_manifest.resolve()),
        "protected_manifest_digest": sha256(starmie_manifest),
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "parent_untouched": True,
        "purpose": "specialist-bootstrap-causal-router-aligned-adapter-fitting",
        "created_at_utc": timestamp,
    }
    atomic_json(authorization_path, authorization)

    starmie_row = {
        "status": "ready",
        "reason": None,
        "run_name": "pure_rl_starmie_temporal1_8k_v4_20260723",
        "log": (
            "/home/inzi/poke-bot-agent/outputs/logs/"
            "pure_rl_starmie_temporal1_8k_v4_20260723.log"
        ),
        "initial_checkpoint": starmie["model_path"],
        "initial_checkpoint_sha256": starmie_digest.removeprefix("sha256:"),
        "expert_manifest": str(args.starmie_expert.resolve()),
        "expert_manifest_sha256": starmie_expert_digest.removeprefix("sha256:"),
        "matchup_runtime_tree": str(args.runtime_tree.resolve()),
        "matchup_runtime_tree_sha256": runtime_tree_digest.removeprefix("sha256:"),
        "matchup_adapter_authorization": str(authorization_path),
        "matchup_adapter_authorization_sha256": sha256(
            authorization_path
        ).removeprefix("sha256:"),
        "matchup_adapter_epochs_per_rl_iteration": 1,
        "measurement_decks": "starmie",
        "guide_loss_weight": 0.0,
        "terminal_gate_marker": "SPECIALIST_GATE_PASSED.starmie-splus-v1",
        "pass_handler": {
            "family": "starmie-protocol-gate-pass-v1",
            "display_name": "Mega Starmie ex Exact Protocol Gate Champion",
            "submission_root": (
                "/home/inzi/poke-bot-agent/outputs/submissions/"
                "starmie-protocol-gate-pass-v1"
            ),
            "state": (
                "/home/inzi/poke-bot-agent/outputs/state/"
                "starmie-passed-gate-handler-v1.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                "starmie-passed-gate-handler-v1.lock"
            ),
            "handoff_service": "pokebot-post-starmie-next-specialist-handoff.service",
        },
    }
    runtime_registry["specialists"]["starmie"] = starmie_row

    snapshot = {
        "schema": "poke_bot.starmie_activation_preimage/v1",
        "frozen_registry": base_registry,
        "gate": base_gate,
        "baseline_manifest": base_manifest,
        "runtime_registry": read_json(runtime_registry_path),
        "created_at_utc": timestamp,
    }
    atomic_json(state_root / "starmie-activation-preimage-v1.json", snapshot)
    atomic_json(args.baseline_manifest, baseline_manifest)
    atomic_json(registry_path, frozen_registry)
    atomic_json(gate_path, next_gate)
    atomic_json(runtime_registry_path, runtime_registry)

    receipt = {
        "schema": "poke_bot.starmie_canonical_activation/v1",
        "created_at_utc": timestamp,
        "trevenant_checkpoint_digest": trevenant_digest,
        "trevenant_kaggle_submission_id": 54937353,
        "trevenant_baseline_package": str(package),
        "trevenant_baseline_content_digest": content_digest,
        "frozen_specialist_ids": [
            row["specialist_id"] for row in frozen_registry["specialists"]
        ],
        "gate_id": next_gate["next_gate"]["id"],
        "gate_games_total": next_gate["next_gate"]["evaluation"]["games_total"],
        "starmie_checkpoint_digest": starmie_digest,
        "starmie_authorization": str(authorization_path),
        "starmie_authorization_digest": sha256(authorization_path),
        "starmie_run_name": starmie_row["run_name"],
        "runtime_registry_digest": sha256(runtime_registry_path),
        "frozen_registry_digest": sha256(registry_path),
        "gate_digest": sha256(gate_path),
        "baseline_manifest_digest": sha256(args.baseline_manifest),
    }
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["identity_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
