#!/usr/bin/env python3
"""Materialize immutable roster-v5 inference derivatives of frozen specialists.

The original gate-passing checkpoints remain untouched.  Each derivative keeps
all non-adapter tensors and retained adapter rows byte-identical, removes the
five retired v4 routes, renames Festival Lead to Thwackey, and adds an exact-zero
Team Rocket's Spidops route.  This command never submits anything to Kaggle.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.baselines_runtime import baseline_content_digest
from scripts.migrate_active_matchup_roster import migrate_checkpoint, migrate_tree

LOADER_RUNTIME_FILES = (
    "poke_bot/config.py",
    "poke_bot/matchup_adapters.py",
    "poke_bot/matchup_adapter_routes.py",
    "poke_bot/public_matchup_router.py",
    "poke_bot/matchup_adapter_activation.py",
    "poke_bot/model.py",
    "poke_bot/checkpoint.py",
    "poke_bot/train.py",
    "poke_bot/dormant_adapter_compat.py",
)

FAMILIES = {
    "alakazam": (
        "alakazam-owner-accepted-iter39-v1",
        "alakazam-owner-accepted-iter39",
    ),
    "hops-trevenant": (
        "hops-trevenant-protocol-gate-pass-v1",
        "hops-trevenant-gate-iter10-462f201f8de6",
    ),
    "starmie": (
        "starmie-protocol-gate-pass-v1",
        "starmie-gate-iter10-51ed1cc6ffe6",
    ),
    "lucario": (
        "lucario-protocol-gate-pass-v1",
        "lucario-gate-iter10-ffa401242b2c",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_package(root: Path, deployments: Path, directory: str) -> Path:
    direct = root / "baselines" / "specialists" / directory
    if direct.is_dir():
        return direct
    matches = sorted(
        deployments.glob(f"*/baselines/specialists/{directory}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"no source package for {directory}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--deployments-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    runtime = args.runtime_root.expanduser().resolve()
    protected = root / "outputs/pure_rl/_protected/models"
    baseline_root = root / "baselines/specialists"
    manifest_path = root / "baselines/manifest.json"
    registry_path = runtime / "ops/frozen_specialist_registry_v1.json"
    gate_path = runtime / "ops/alakazam_gate_program_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["specialist_id"]: row for row in registry["specialists"]}
    results: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for specialist_id, (source_family, source_dir) in FAMILIES.items():
        source_model = protected / source_family / "model.pt"
        source_package = find_package(root, args.deployments_root, source_dir)
        source_tree = source_package / "matchup_tree.json"
        target_family_name = source_family + "-roster18-v5"
        target_family = protected / target_family_name
        target_dir = source_dir + "-roster18-v5"
        target_package = baseline_root / target_dir
        target_model = target_family / "model.pt"
        target_tree = target_family / "matchup_tree.json"
        target_family.mkdir(parents=True, exist_ok=True)
        if target_model.exists() and target_tree.exists():
            prior = json.loads(
                (target_family / "manifest.json").read_text(encoding="utf-8")
            )
            if prior.get("source_passing_checkpoint_digest") != sha256(source_model):
                raise RuntimeError(f"existing derivative source changed: {target_family}")
            checkpoint_result = {
                "output_digest": sha256(target_model),
                "source_digest": sha256(source_model),
                "retained_rows_byte_identical": bool(
                    prior.get("retained_rows_byte_identical")
                ),
                "spidops_exact_zero": bool(prior.get("spidops_exact_zero")),
            }
            tree_result = {"output_digest": sha256(target_tree)}
        else:
            checkpoint_result = migrate_checkpoint(source_model, target_model)
            tree_result = migrate_tree(
                source_tree,
                target_tree,
                checkpoint=target_model,
                checkpoint_digest=checkpoint_result["output_digest"],
            )
        if not target_package.exists():
            shutil.copytree(source_package, target_package)
        shutil.copy2(target_model, target_package / "model.pt")
        shutil.copy2(target_tree, target_package / "matchup_tree.json")
        for relative in LOADER_RUNTIME_FILES:
            source_runtime_file = runtime / relative
            if not source_runtime_file.is_file():
                raise FileNotFoundError(f"missing loader overlay: {source_runtime_file}")
            destination = target_package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_runtime_file, destination)
        content_digest = baseline_content_digest(target_package)
        source_digest = checkpoint_result["source_digest"]
        receipt = {
            "schema": "poke_bot.frozen_specialist_roster_v5_derivative/v1",
            "created_at_utc": now,
            "specialist_id": specialist_id,
            "source_passing_checkpoint": str(source_model),
            "source_passing_checkpoint_digest": source_digest,
            "derived_checkpoint": str(target_model),
            "derived_checkpoint_digest": checkpoint_result["output_digest"],
            "derived_tree": str(target_tree),
            "derived_tree_digest": tree_result["output_digest"],
            "baseline_package": str(target_package),
            "baseline_content_digest": content_digest,
            "retained_rows_byte_identical": checkpoint_result[
                "retained_rows_byte_identical"
            ],
            "spidops_exact_zero": checkpoint_result["spidops_exact_zero"],
            "kaggle_submission_eligible": False,
            "kaggle_submission_requested": False,
            "inference_only": True,
        }
        atomic_json(target_family / "manifest.json", receipt)
        atomic_json(
            target_family / "PROTECTED_DO_NOT_PRUNE.json",
            {
                "schema": "poke_bot.protected_model_family/v1",
                "family": target_family_name,
                "checkpoint_digest": checkpoint_result["output_digest"],
                "automatic_pruning_allowed": False,
                "manual_removal_requires_explicit_model_registry_override": True,
            },
        )
        row = rows[specialist_id]
        old_opponent_id = "specialist-" + source_dir
        new_opponent_id = old_opponent_id + "-roster18-v5"
        row.update(
            baseline_dir=target_dir,
            opponent_id=new_opponent_id,
            checkpoint_digest=checkpoint_result["output_digest"],
            content_digest=content_digest,
            matchup_tree_checksum=tree_result["output_digest"],
            source_passing_checkpoint_digest=source_digest,
            roster_version=5,
            adapter_route_count=18,
            v5_derivative_receipt=str(target_family / "manifest.json"),
            kaggle_submission_eligible=False,
        )
        for gate_row in gate["next_gate"]["roster"]:
            if gate_row.get("opponent_id") not in (old_opponent_id, new_opponent_id):
                continue
            gate_row.update(
                opponent_id=new_opponent_id,
                content_digest=content_digest,
                frozen_checkpoint_digest=checkpoint_result["output_digest"],
                source_passing_checkpoint_digest=source_digest,
                roster_version=5,
            )
        baseline_manifest["agents"] = [
            item
            for item in baseline_manifest.get("agents", [])
            if item.get("id") not in (old_opponent_id, new_opponent_id)
        ]
        baseline_manifest["agents"].append(
            {
                "dir": target_dir,
                "group": "specialists",
                "id": new_opponent_id,
                "name": row["archetype_label"] + " · roster v5",
                "source": (
                    f"internal inference-only roster-v5 derivative of {source_digest}"
                ),
            }
        )
        results.append(receipt)

    registry["updated_at_utc"] = now
    registry["version"] = int(registry.get("version", 0)) + 1
    registry["canonical_internal_roster_version"] = 5
    gate["updated_at_utc"] = now
    # Package/checkpoint identities change, but the established gate semantics
    # and gate identifier do not.
    gate["active_gate_id"] = str(gate["active_gate_id"]).removesuffix(
        "-roster18-v5"
    )
    gate["next_gate"]["id"] = str(gate["next_gate"]["id"]).removesuffix(
        "-roster18-v5"
    )
    atomic_json(manifest_path, baseline_manifest)
    atomic_json(registry_path, registry)
    atomic_json(gate_path, gate)
    receipt_path = root / "outputs/state/frozen-specialist-roster18-v5-pool.json"
    atomic_json(
        receipt_path,
        {
            "schema": "poke_bot.frozen_specialist_roster_v5_pool/v1",
            "created_at_utc": now,
            "kaggle_submissions_created": 0,
            "specialists": results,
        },
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
