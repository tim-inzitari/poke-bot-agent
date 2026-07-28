#!/usr/bin/env python3
"""Validate and atomically arm the complete pre-transition Alakazam build."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.alakazam_heuristics import GUIDE_VERSION
from poke_bot.pure_rl.model_registry import sha256
from scripts.check_alakazam_guide_runtime import run_canary
from scripts.run_alakazam_expert_bootstrap import validate_filtered_manifest
from scripts.watch_deck_agnostic_core_transition import (
    SPECIALIST_BUILD_ARTIFACTS,
    SPECIALIST_DEPLOYMENT_ARTIFACTS,
)


UNITS = (
    "pokebot-pure-rl-alakazam-bootstrap.service",
    "pokebot-pure-rl-alakazam.service",
    "pokebot-deck-agnostic-transition.service",
)
TESTS = (
    "tests/test_alakazam_bootstrap_contract.py",
    "tests/test_alakazam_guide_distillation.py",
    "tests/test_alakazam_guide_targets.py",
    "tests/test_specialist_archetype_contract.py",
    "tests/test_curriculum_all_heads.py",
    "tests/test_deck_agnostic_transition_service.py",
    "tests/test_pure_rl_awr.py",
    "tests/test_pure_rl_replay_cache.py",
    "tests/test_training_memory_containment.py",
    "tests/test_pure_rl_recovery_and_scheduling.py",
    "tests/test_pure_rl_deferred_weight_publish.py",
    "tests/test_pure_rl_lineage_retention.py",
    "tests/test_pure_rl_state_profile.py",
    "tests/test_core_transition.py",
    "tests/test_protected_model_registry.py",
    "tests/test_filter_feature_manifest.py",
    "tests/test_incremental_feature_manifest.py",
    "tests/test_dataset_robustness.py",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(argv: list[str], *, timeout: float = 600.0) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"readiness command exited {completed.returncode}: {' '.join(argv)}"
        )
    return completed.stdout


def systemctl_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    return result.stdout.strip()


def endpoint_ready(endpoint: str, *, timeout: float = 3.0) -> dict[str, Any]:
    host, raw_port = endpoint.rsplit(":", 1)
    port = int(raw_port)
    with socket.create_connection((host, port), timeout=timeout) as connection:
        peer = connection.getpeername()
    return {"endpoint": endpoint, "peer": f"{peer[0]}:{peer[1]}", "status": "ready"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-manifest", type=Path, required=True)
    parser.add_argument("--canary-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--installed-unit-dir",
        type=Path,
        default=Path.home() / ".config/systemd/user",
    )
    parser.add_argument(
        "--specialist-deployment-root",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent-deployments/"
            "pure-rl-continuous-rehearsal-v1"
        ),
    )
    parser.add_argument(
        "--training-arm",
        type=Path,
        default=ROOT / "outputs/state/TRAINING_ARMED",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/run/user/1000/gvfs/"
            "smb-share:server=192.168.1.143,share=main,user=inzi/"
            "poke-bot-agent/containers/truenas-worker/checkpoint"
        ),
    )
    parser.add_argument("--min-expert-decisions", type=int, default=100_000)
    parser.add_argument("--min-free-disk-gb", type=float, default=100.0)
    parser.add_argument(
        "--hidden-engine",
        type=Path,
        default=ROOT / "outputs/engines/libcg_hidden_inzi_v1.so",
    )
    parser.add_argument(
        "--hidden-engine-sha256",
        default=(
            "sha256:923cb0705bab303d0d2025da3647233c35fa4ce324e763816038d1e6fda10387"
        ),
    )
    parser.add_argument(
        "--allow-frozen-core-handoff",
        action="store_true",
        help=(
            "Re-arm a digest-stale specialist build after the core has already "
            "been frozen and the immutable bootstrap is running or complete."
        ),
    )
    parser.add_argument(
        "--remote-endpoint",
        action="append",
        default=["192.168.1.143:8765", "bert.local:8766"],
    )
    args = parser.parse_args()

    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    filtered_pointer = args.filtered_manifest.expanduser().resolve()
    filtered = validate_filtered_manifest(
        filtered_pointer,
        min_decisions=int(args.min_expert_decisions),
    )
    totals = dict(filtered.get("totals") or {})

    test_output = command(
        [str(python), "-m", "pytest", "-q", *TESTS],
        timeout=900,
    )
    matches = re.findall(r"(\d+) passed", test_output)
    passed = int(matches[-1]) if matches else 0
    if passed <= 0:
        raise RuntimeError("focused specialist test suite reported no passing tests")

    source_units = [ROOT / "deploy/systemd" / name for name in UNITS]
    command(
        ["systemd-analyze", "--user", "verify", *map(str, source_units)],
        timeout=120,
    )
    installed: dict[str, str] = {}
    for source in source_units:
        target = args.installed_unit_dir.expanduser().resolve() / source.name
        if not target.is_file():
            raise FileNotFoundError(f"installed specialist unit missing: {target}")
        source_digest = sha256(source)
        target_digest = sha256(target)
        if source_digest != target_digest:
            raise RuntimeError(f"installed specialist unit is stale: {source.name}")
        installed[source.name] = target_digest

    os.environ["POKEBOT_ALAKAZAM_GUIDE_TARGETS"] = "1"
    canary = run_canary(
        args.canary_jsonl,
        max_records=32,
        min_guide_rows=25,
    )

    arm = args.training_arm.expanduser().resolve()
    if not arm.is_file() or arm.stat().st_size <= 0:
        raise RuntimeError(f"training arm is absent/empty: {arm}")
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"checkpoint share is unavailable: {checkpoint_dir}")
    free_gb = shutil.disk_usage(ROOT).free / (1024**3)
    if free_gb < float(args.min_free_disk_gb):
        raise RuntimeError(
            f"free disk {free_gb:.1f} GiB < required {float(args.min_free_disk_gb):.1f} GiB"
        )
    gpu_text = command(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        timeout=30,
    )
    if "RTX 3080 Ti" not in gpu_text or "RTX PRO 5000 Blackwell" not in gpu_text:
        raise RuntimeError("specialist build requires both the 3080 Ti and Blackwell")
    hidden_engine = args.hidden_engine.expanduser().resolve()
    if not hidden_engine.is_file():
        raise FileNotFoundError(f"hidden-state training engine missing: {hidden_engine}")
    hidden_engine_digest = sha256(hidden_engine)
    if hidden_engine_digest != str(args.hidden_engine_sha256):
        raise RuntimeError(
            "hidden-state training engine digest mismatch: "
            f"actual={hidden_engine_digest} expected={args.hidden_engine_sha256}"
        )
    states = {unit: systemctl_state(unit) for unit in UNITS}
    if states["pokebot-pure-rl-alakazam.service"] not in {"inactive", "failed"}:
        raise RuntimeError("Alakazam specialist is unexpectedly active during readiness")
    core_state = systemctl_state("pokebot-pure-rl-continuous-rehearsal.service")
    bootstrap_state = states["pokebot-pure-rl-alakazam-bootstrap.service"]
    if args.allow_frozen_core_handoff:
        if bootstrap_state == "failed":
            raise RuntimeError("Alakazam bootstrap failed during handoff re-arm")
        if core_state == "active":
            raise RuntimeError("frozen-core handoff re-arm found core still active")
    else:
        if bootstrap_state not in {"inactive", "failed"}:
            raise RuntimeError(
                "Alakazam bootstrap is unexpectedly active during readiness"
            )
        if core_state != "active":
            raise RuntimeError("Deck Agnostic Core is not active during readiness")

    endpoints = [endpoint_ready(value) for value in dict.fromkeys(args.remote_endpoint)]
    artifacts = {
        relative: sha256(ROOT / relative) for relative in SPECIALIST_BUILD_ARTIFACTS
    }
    deployment_root = args.specialist_deployment_root.expanduser().resolve()
    deployment_artifacts: dict[str, str] = {}
    for relative in SPECIALIST_DEPLOYMENT_ARTIFACTS:
        canonical = ROOT / relative
        deployed = deployment_root / relative
        if not deployed.is_file() or sha256(deployed) != sha256(canonical):
            raise RuntimeError(f"specialist deployment artifact is stale: {relative}")
        deployment_artifacts[relative] = sha256(canonical)
    payload = {
        "schema": "poke_bot.alakazam_specialist_build_ready/v1",
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "guide": {
            "version": GUIDE_VERSION,
            "source_sha256": sha256(ROOT / "poke_bot/alakazam_heuristics.py"),
            "training_only": True,
            "evaluation_overrides": False,
        },
        "artifacts": artifacts,
        "specialist_deployment": {
            "status": "validated",
            "root": str(deployment_root),
            "artifacts": deployment_artifacts,
        },
        "tests": {"status": "passed", "passed": passed, "selectors": list(TESTS)},
        "systemd_verify": {
            "status": "passed",
            "installed_unit_digests": installed,
        },
        "runtime_canary": canary,
        "expert_corpus": {
            "status": "validated",
            "pointer": str(filtered_pointer),
            "pointer_sha256": sha256(filtered_pointer),
            "records": int(totals.get("records_kept", 0)),
            "decisions": int(totals.get("decisions_kept", 0)),
        },
        "runtime_preflight": {
            "status": "passed",
            "core_service": core_state,
            "specialist_services": states,
            "training_arm": str(arm),
            "checkpoint_dir": str(checkpoint_dir),
            "free_disk_gib": free_gb,
            "gpus": [line for line in gpu_text.splitlines() if line.strip()],
            "hidden_engine": {
                "path": str(hidden_engine),
                "sha256": hidden_engine_digest,
                "training_only": True,
            },
            "remote_endpoints": endpoints,
        },
        "handoff_contract": {
            "core_continues_during_bootstrap": False,
            "bootstrap_physical_gpu": "RTX PRO 5000 Blackwell",
            "device_resident_bootstrap": True,
            "stop_core_at_exact_gate": True,
        },
    }
    atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
