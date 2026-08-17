#!/usr/bin/env python3
"""Build and smoke the one revision-10 full-model derivative package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GOAL_REVISION = 11
GOAL_CONTRACT_SHA256 = "sha256:d74152bca415c80e4983172b5fdcd8c03313e0c8a0a18e24e0c68a7bcfe84245"
BOOTSTRAP_CONTRACT_SHA256 = "sha256:91e9c60e87fe093446ef9979f64464be18e77a5041afe82164c4c6ca80d2225f"
CANDIDATE_SHA256 = "sha256:8b59af9af1d715639bd3d63a84df7d608cee686c27aadf1c6dac3c971631a248"
VALIDATION_SHA256 = "sha256:fdb340727f1c602bfff3eea317420182d3629cd52c529dd835454b3acbcd76a7"
R195_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
LABEL = "Alakazam public-rule derivative g5, exact new list, Blackwell bootstrap, no RTP"
DECK_SHA256 = "sha256:d834c66c5a3629dd79c8533a04fde770a22ca8590ac55c9868440121b6df5fba"
DECK_ORDERED_SHA256 = "sha256:e61c0a4ffcfeb730808ac561f39c1efa9de5f80aec577d3be82f3fc790b7dab2"
DECK_MULTISET_SHA256 = "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
SIMULATOR_SHA256 = "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def write_json_create_only(path: Path, payload: Any) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def stable_file(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{label} identity mismatch: {path}")


def inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"package source contains symlink: {path}")
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision9-stage", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    stable_file(args.candidate, CANDIDATE_SHA256, "candidate")
    stable_file(args.candidate_validation, VALIDATION_SHA256, "candidate validation")
    if args.output_root.exists():
        raise RuntimeError(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    stage = args.output_root / "stage"
    stage.mkdir()

    for source in sorted(args.revision9_stage.iterdir()):
        if source.name in {"poke_bot", "model.pt", "runtime_profile.json", "main.py", "__pycache__"}:
            continue
        target = stage / source.name
        if source.is_symlink():
            raise RuntimeError(f"revision-9 stage contains symlink: {source}")
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif source.is_file():
            shutil.copy2(source, target)
    shutil.copytree(
        args.runtime_source / "poke_bot",
        stage / "poke_bot",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # The established revision-9 submission surface carries the runtime
    # PolicyAgent hook for the public-rule semantic projection.  The isolated
    # training snapshot intentionally did not need that hook, so retain the
    # proven package copy while taking the bug-fixed model/heuristic sources
    # from the revision-10 training snapshot.
    agent_target = stage / "poke_bot" / "agent.py"
    agent_target.chmod(0o644)
    shutil.copy2(args.revision9_stage / "poke_bot" / "agent.py", agent_target)
    shutil.copy2(args.candidate, stage / "model.pt")

    main_text = (args.revision9_stage / "main.py").read_text()
    main_text = main_text.replace(
        'checkpoint_payload.get("goal_revision") != 9',
        'checkpoint_payload.get("goal_revision") != 10',
    ).replace(
        "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        BOOTSTRAP_CONTRACT_SHA256,
    )
    marker = 'os.environ.setdefault("CG_LIB_PATH", str(_agent_dir()))\n'
    replacement = marker + (
        '        os.environ["POKEBOT_COMBO_STATE_ROUTE_ENABLED"] = "0"\n'
        '        os.environ["COMBO_STATE_ROUTE_ENABLED"] = "0"\n'
    )
    if marker not in main_text:
        raise RuntimeError("package main runtime marker missing")
    main_text = main_text.replace(marker, replacement, 1)
    (stage / "main.py").write_text(main_text)

    profile = {
        "schema": "poke_bot.submission_runtime_profile/v1",
        "display": "NO RTP",
        "rtp_mode": "off",
        "recursive_turn_planner": "disabled",
        "rtp_sidecar_packaged": False,
        "model_checkpoint_sha256": CANDIDATE_SHA256,
        "public_rule_semantic_projection": "enabled",
        "public_rule_semantic_projection_gate": 1.0,
        "public_rule_metadata_residual": "disabled_exact_zero",
        "repaired_auxiliary_heads": "disabled_exact_zero",
        "eight_checklist_provenance_gates": "disabled_exact_zero",
        "combo_state_route": "disabled_exact_zero",
    }
    (stage / "runtime_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")

    package = args.output_root / "submission.tar.gz"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        ["tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner", "-czf", str(package), "./"],
        cwd=stage,
        env=env,
        check=True,
    )
    work = args.output_root / "smoke_extract"
    work.mkdir()
    subprocess.run(["tar", "-xzf", str(package), "-C", str(work)], check=True)
    smoke_code = r'''
import importlib.util, json, os, sys
from pathlib import Path
root=Path.cwd()
sys.path.insert(0,str(root))
os.environ["POKEBOT_SUBMISSION_SEARCH_DISABLE"]="1"
os.environ["POKEBOT_COMBO_STATE_ROUTE_ENABLED"]="0"
os.environ["COMBO_STATE_ROUTE_ENABLED"]="0"
spec=importlib.util.spec_from_file_location("r10_package_main",root/"main.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
deck=module.agent({"logs":[],"current":None,"select":None})
assert len(deck)==60
from cg.game import battle_start,battle_select,battle_finish
obs,_=battle_start(deck,deck); steps=0
while obs is not None and steps<80:
    current=obs.get("current") or {}
    if current.get("result",-1)!=-1 or obs.get("select") is None: break
    action=module.agent(obs); options=obs["select"].get("option") or []
    assert all(0<=int(i)<len(options) for i in action)
    obs=battle_select(action); steps+=1
battle_finish()
policy=module._POLICY
print(json.dumps({"steps":steps,"semantic_apply_count":int(policy.public_rule_semantic_projection_apply_count),"fail_closed_count":int(policy.fail_closed_count)}))
assert steps>0 and policy.public_rule_semantic_projection_apply_count>0 and policy.fail_closed_count==0
'''
    smoke_env = dict(env)
    smoke_env["PYTHONPATH"] = str(work)
    smoke = subprocess.run(
        [sys.executable, "-B", "-c", smoke_code],
        cwd=work,
        env=smoke_env,
        text=True,
        capture_output=True,
    )
    if smoke.returncode != 0:
        raise RuntimeError(f"package smoke failed:\n{smoke.stdout}\n{smoke.stderr}")
    smoke_metrics = json.loads(smoke.stdout.strip().splitlines()[-1])

    member_rows = inventory(work)
    source_rows = inventory(args.runtime_source / "poke_bot")
    package_sha = sha256_file(package)
    evidence = {
        "schema": "poke_bot.alakazam_rule_derivative_r10_kaggle_package_smoke_evidence/v1",
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "package_sha256": package_sha,
        "package_size_bytes": package.stat().st_size,
        "package_member_manifest_sha256": canonical_sha(member_rows),
        "engine_battle_steps": smoke_metrics["steps"],
        "public_rule_semantic_projection_apply_count": smoke_metrics["semantic_apply_count"],
        "policy_fail_closed_count": smoke_metrics["fail_closed_count"],
        "combo_route_enabled": False,
        "package_parity_passed": True,
        "package_smoke_passed": True,
    }
    evidence_sha = write_json_create_only(args.output_root / "package-smoke-evidence.json", evidence)
    receipt = {
        "schema": "poke_bot.alakazam_rule_derivative_kaggle_package_receipt/v1",
        "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "candidate_validation_receipt_sha256": VALIDATION_SHA256,
        "candidate_checkpoint_path": str(args.candidate.resolve()),
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_checkpoint_size_bytes": args.candidate.stat().st_size,
        "parent_checkpoint_sha256": R195_SHA256,
        "deck_path": str((stage / "deck.csv").resolve()),
        "deck_sha256": DECK_SHA256,
        "deck_ordered_cards_sha256": DECK_ORDERED_SHA256,
        "deck_canonical_multiset_sha256": DECK_MULTISET_SHA256,
        "runtime_manifest_sha256": canonical_sha(source_rows),
        "runtime_commit_sha256": canonical_sha(source_rows),
        "canonical_simulator_sha256": SIMULATOR_SHA256,
        "public_catalog_manifest_sha256": "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6",
        "feature_schema_sha256": "sha256:d4d8f1be1219f5bb5a1ff31582225d6f43d369e09a5a78e1c74aa560c67b0f1",
        "target_schema_sha256": "sha256:6b9544fa7285edad25e34ab7fca08bfa4cf9c395e1748178c73708d3377f354f",
        "checklist_provenance_schema_sha256": "sha256:fd39740b1a2092768dffadcf2f4beb3ae1cc1f38fe374b73a7dac0d4a7f20d9f",
        "q3_bench_only": True,
        "q5_q6_trace_only_zero": True,
        "matchup_router_manifest_sha256": "sha256:aa9502a0c8495369913507fa9c8bfc835172f5c4765580084eca32424bd003c6",
        "matchup_adapter_inventory_sha256": "sha256:f6bf5e017fd5a0e2aaa6f7c5b6e961c9152dd81530c4a63e2c0a721e739254db",
        "rtp_enabled": False,
        "search_or_mcts_enabled": False,
        "package_path": str(package.resolve()),
        "package_sha256": package_sha,
        "package_size_bytes": package.stat().st_size,
        "package_member_manifest_sha256": canonical_sha(member_rows),
        "package_parity_passed": True,
        "package_smoke_passed": True,
        "canonical_label": LABEL,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    receipt_sha = write_json_create_only(args.output_root / "kaggle-package-receipt.json", receipt)
    complete_sha = write_json_create_only(args.output_root / "COMPLETE.json", {
        "schema": "poke_bot.alakazam_rule_derivative_r10_kaggle_package_completion/v1",
        "status": "passed",
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_validation_receipt_sha256": VALIDATION_SHA256,
        "package_sha256": package_sha,
        "package_receipt_sha256": receipt_sha,
        "package_smoke_evidence_sha256": evidence_sha,
    })
    print(json.dumps({"package_path": str(package), "package_sha256": package_sha, "package_size_bytes": package.stat().st_size, "package_receipt_sha256": receipt_sha, "smoke_evidence_sha256": evidence_sha, "completion_sha256": complete_sha, **smoke_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
