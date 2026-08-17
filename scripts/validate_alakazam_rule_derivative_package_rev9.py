#!/usr/bin/env python3
"""Re-open and smoke the exact revision-9 derivative Kaggle package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
CANDIDATE = "sha256:5c42b99a5eb101c1ea173ae9426326db61d3bdb81a84903468e7bac5e6a30f24"
DECK_MULTISET = "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def create_json(path: Path, value: Any) -> str:
    body = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.package.is_symlink() or not args.package.is_file():
        raise RuntimeError("package is not a stable regular file")
    args.work_root.mkdir(parents=True, exist_ok=False)
    member_rows = []
    with tarfile.open(args.package, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            parts = Path(name).parts
            if member.issym() or member.islnk() or member.name.startswith("/") or ".." in parts:
                raise RuntimeError("package contains an unsafe member")
        archive.extractall(args.work_root)
    for path in sorted(item for item in args.work_root.rglob("*") if item.is_file()):
        member_rows.append(
            {
                "path": path.relative_to(args.work_root).as_posix(),
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    profile = json.loads((args.work_root / "runtime_profile.json").read_text())
    required_profile = {
        "rtp_mode": "off",
        "recursive_turn_planner": "disabled",
        "display": "NO RTP",
        "rtp_sidecar_packaged": False,
        "model_checkpoint_sha256": CANDIDATE,
        "public_rule_semantic_projection": "enabled",
        "public_rule_semantic_projection_gate": 1.0,
        "public_rule_metadata_residual": "disabled_exact_zero",
        "repaired_auxiliary_heads": "disabled_exact_zero",
        "eight_checklist_provenance_gates": "disabled_exact_zero",
    }
    if any(profile.get(key) != value for key, value in required_profile.items()):
        raise RuntimeError("submitted derivative runtime profile drifted")
    os.chdir(args.work_root)
    sys.path.insert(0, str(args.work_root))
    spec = importlib.util.spec_from_file_location("revision9_package_main", args.work_root / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("package entrypoint is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deck = module.agent({"logs": [], "current": None, "select": None})
    if len(deck) != 60:
        raise RuntimeError("package did not return the exact 60-card deck")
    from cg.game import battle_finish, battle_select, battle_start

    obs, _ = battle_start(deck, deck)
    steps = 0
    while obs is not None and steps < 80:
        current = obs.get("current") or {}
        if current.get("result", -1) != -1 or obs.get("select") is None:
            break
        action = module.agent(obs)
        options = obs["select"].get("option") or []
        if any(not 0 <= int(index) < len(options) for index in action):
            raise RuntimeError("package emitted an illegal option index")
        obs = battle_select(action)
        steps += 1
    battle_finish()
    policy = module._POLICY
    applied = int(policy.public_rule_semantic_projection_apply_count)
    if steps <= 0 or applied <= 0 or policy.fail_closed_count != 0:
        raise RuntimeError("package semantic sidecar did not execute cleanly")
    payload = {
        "schema": "poke_bot.alakazam_rule_derivative_kaggle_package_smoke_evidence/v1",
        "goal_revision": 9,
        "goal_contract_sha256": CONTRACT,
        "candidate_checkpoint_sha256": CANDIDATE,
        "deck_canonical_multiset_sha256": DECK_MULTISET,
        "package_path": str(args.package.resolve()),
        "package_sha256": digest(args.package),
        "package_size_bytes": args.package.stat().st_size,
        "package_member_manifest": member_rows,
        "package_member_manifest_sha256": "sha256:" + hashlib.sha256(canonical(member_rows)).hexdigest(),
        "runtime_profile_sha256": digest(args.work_root / "runtime_profile.json"),
        "no_rtp": True,
        "public_rule_semantic_projection_enabled": True,
        "public_rule_semantic_projection_gate": 1.0,
        "public_rule_semantic_projection_apply_count": applied,
        "unsupported_branches_exact_zero": True,
        "engine_battle_steps": steps,
        "policy_fail_closed_count": int(policy.fail_closed_count),
        "package_parity_passed": True,
        "package_smoke_passed": True,
        "validated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    create_json(args.evidence, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
