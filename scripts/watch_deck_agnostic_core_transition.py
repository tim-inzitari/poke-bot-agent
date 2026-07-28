#!/usr/bin/env python3
"""Freeze Deck Agnostic Core and hand off to Alakazam at an exact boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.core_transition import inherited_anchor, transition_decision
from poke_bot.pure_rl.model_registry import freeze_model, sha256, verify_frozen_model
from poke_bot.alakazam_heuristics import GUIDE_VERSION
from scripts.run_alakazam_expert_bootstrap import validate_filtered_manifest


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish(path: Path, state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    state.update(updates)
    state["schema"] = "poke_bot.deck_agnostic_core_transition/v1"
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, state)
    print(
        f"[core-transition] status={state.get('status')} "
        f"reason={state.get('decision', {}).get('reason', '')}",
        flush=True,
    )
    return state


def command(
    argv: list[str], *, timeout: float = 180.0, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result


def systemctl(*args: str, timeout: float = 180.0, check: bool = True) -> str:
    return command(
        ["systemctl", "--user", *args], timeout=timeout, check=check
    ).stdout.strip()


def active(unit: str) -> bool:
    return systemctl("is-active", unit, timeout=15, check=False) == "active"


def apply_forced_iteration_boundary(
    decision: dict[str, Any], force_after_iteration: int
) -> dict[str, Any]:
    """Force the handoff only after that iteration has exact heldout proof."""
    boundary = int(force_after_iteration)
    if boundary < 0 or int(decision.get("latest_iteration", -1)) < boundary:
        return decision
    forced = json.loads(json.dumps(decision))
    forced["triggered"] = True
    forced["reason"] = f"user_forced_after_exact_iteration_{boundary}"
    forced["forced_after_iteration"] = boundary
    return forced


SPECIALIST_BUILD_ARTIFACTS = (
    "poke_bot/alakazam_heuristics.py",
    "poke_bot/dataset.py",
    "poke_bot/feature_shards.py",
    "poke_bot/pure_rl/dataset_bridge.py",
    "poke_bot/pure_rl/multi_env_self_play.py",
    "poke_bot/pure_rl/shards.py",
    "poke_bot/remote_sim_jobs.py",
    "poke_bot/train.py",
    "scripts/check_alakazam_guide_runtime.py",
    "scripts/launch_pure_rl.py",
    "scripts/prepare_alakazam_specialist_build.py",
    "scripts/run_alakazam_expert_bootstrap.py",
    "scripts/train_pure_rl.py",
    "scripts/watch_deck_agnostic_core_transition.py",
    "deploy/systemd/pokebot-pure-rl-alakazam-bootstrap.service",
    "deploy/systemd/pokebot-pure-rl-alakazam.service",
    "deploy/systemd/pokebot-deck-agnostic-transition.service",
)
SPECIALIST_DEPLOYMENT_ARTIFACTS = (
    "poke_bot/alakazam_heuristics.py",
    "poke_bot/dataset.py",
    "poke_bot/feature_shards.py",
    "poke_bot/pure_rl/dataset_bridge.py",
    "poke_bot/pure_rl/multi_env_self_play.py",
    "poke_bot/pure_rl/shards.py",
    "poke_bot/remote_sim_jobs.py",
    "poke_bot/train.py",
    "scripts/launch_pure_rl.py",
    "scripts/train_pure_rl.py",
)


def validate_handoff_ready(
    path: Path,
    *,
    filtered_manifest: Path | None = None,
    min_expert_decisions: int = 100_000,
    installed_unit_dir: Path | None = None,
    specialist_deployment_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed until the complete tested specialist build is staged.

    This marker covers every pre-transition artifact, the immutable expert
    corpus, systemd validation, the real-runtime guide canary, and the focused
    test suite. It deliberately does *not* claim that the expert warm start
    already exists: once the exact gate passes, the watcher freezes that core,
    releases Blackwell, runs the full device-resident warm start, and launches
    specialist RL from the verified checkpoint.
    """
    payload = read_json(path)
    if payload.get("schema") != "poke_bot.alakazam_specialist_build_ready/v1":
        raise ValueError("Alakazam specialist readiness schema is invalid")
    if payload.get("status") != "ready":
        raise ValueError("Alakazam specialist readiness status is not ready")
    guide = payload.get("guide")
    if not isinstance(guide, dict) or guide.get("version") != GUIDE_VERSION:
        raise ValueError("Alakazam guide readiness version does not match source")
    expected_guide = sha256(ROOT / "poke_bot/alakazam_heuristics.py")
    if guide.get("source_sha256") != expected_guide:
        raise ValueError("Alakazam guide readiness digest is stale")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Alakazam specialist readiness lacks artifact digests")
    for relative in SPECIALIST_BUILD_ARTIFACTS:
        source = ROOT / relative
        if not source.is_file():
            raise ValueError(f"Alakazam specialist artifact is missing: {relative}")
        if artifacts.get(relative) != sha256(source):
            raise ValueError(f"Alakazam specialist artifact digest is stale: {relative}")

    deployment = payload.get("specialist_deployment")
    if not isinstance(deployment, dict) or deployment.get("status") != "validated":
        raise ValueError("Alakazam specialist deployment tree is not validated")
    deployment_root = (
        Path(specialist_deployment_root).expanduser().resolve()
        if specialist_deployment_root is not None
        else Path(str(deployment.get("root") or "")).expanduser().resolve()
    )
    deployment_digests = deployment.get("artifacts")
    if not isinstance(deployment_digests, dict):
        raise ValueError("Alakazam specialist deployment digests are missing")
    for relative in SPECIALIST_DEPLOYMENT_ARTIFACTS:
        canonical = ROOT / relative
        deployed = deployment_root / relative
        expected = sha256(canonical)
        if (
            not deployed.is_file()
            or sha256(deployed) != expected
            or deployment_digests.get(relative) != expected
        ):
            raise ValueError(f"Alakazam specialist deployment is stale: {relative}")

    tests = payload.get("tests")
    if (
        not isinstance(tests, dict)
        or tests.get("status") != "passed"
        or int(tests.get("passed", 0)) <= 0
    ):
        raise ValueError("Alakazam specialist readiness lacks passing tests")
    systemd = payload.get("systemd_verify")
    if not isinstance(systemd, dict) or systemd.get("status") != "passed":
        raise ValueError("Alakazam specialist systemd validation has not passed")
    installed = systemd.get("installed_unit_digests")
    if not isinstance(installed, dict):
        raise ValueError("Alakazam specialist installed-unit identity is missing")
    for unit in (
        "pokebot-pure-rl-alakazam-bootstrap.service",
        "pokebot-pure-rl-alakazam.service",
        "pokebot-deck-agnostic-transition.service",
    ):
        source = ROOT / "deploy/systemd" / unit
        unit_dir = (
            Path(installed_unit_dir).expanduser().resolve()
            if installed_unit_dir is not None
            else Path.home() / ".config/systemd/user"
        )
        target = unit_dir / unit
        expected = sha256(source)
        if (
            not target.is_file()
            or sha256(target) != expected
            or installed.get(unit) != expected
        ):
            raise ValueError(f"Alakazam specialist installed unit is stale: {unit}")
    canary = payload.get("runtime_canary")
    if (
        not isinstance(canary, dict)
        or canary.get("status") != "passed"
        or int(canary.get("guide_rows", 0)) <= 0
        or canary.get("guide_version") != GUIDE_VERSION
        or canary.get("guide_source_sha256") != expected_guide
    ):
        raise ValueError("Alakazam specialist runtime guide canary has not passed")

    runtime = payload.get("runtime_preflight")
    if not isinstance(runtime, dict) or runtime.get("status") != "passed":
        raise ValueError("Alakazam specialist runtime preflight has not passed")
    contract = payload.get("handoff_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("core_continues_during_bootstrap") is not False
        or contract.get("bootstrap_physical_gpu") != "RTX PRO 5000 Blackwell"
        or contract.get("device_resident_bootstrap") is not True
        or contract.get("stop_core_at_exact_gate") is not True
    ):
        raise ValueError("Alakazam specialist handoff contract is incomplete")

    corpus = payload.get("expert_corpus")
    if not isinstance(corpus, dict) or corpus.get("status") != "validated":
        raise ValueError("Alakazam specialist expert corpus is not validated")
    if filtered_manifest is not None:
        filtered_manifest = Path(filtered_manifest).expanduser().resolve()
        validated = validate_filtered_manifest(
            filtered_manifest, min_decisions=int(min_expert_decisions)
        )
        totals = dict(validated.get("totals") or {})
        if corpus.get("pointer_sha256") != sha256(filtered_manifest):
            raise ValueError("Alakazam specialist expert corpus pointer is stale")
        if int(corpus.get("decisions", 0)) != int(totals.get("decisions_kept", 0)):
            raise ValueError("Alakazam specialist expert decision count changed")
        if int(corpus.get("records", 0)) != int(totals.get("records_kept", 0)):
            raise ValueError("Alakazam specialist expert record count changed")
    return payload


def rollback_to_core(
    *,
    source_unit: str,
    bootstrap_unit: str,
    specialist_unit: str,
    state: dict[str, Any],
    status: Path,
    error: BaseException,
) -> None:
    systemctl("stop", bootstrap_unit, timeout=60, check=False)
    systemctl("stop", specialist_unit, timeout=60, check=False)
    systemctl("disable", specialist_unit, timeout=30, check=False)
    systemctl("enable", source_unit, timeout=30, check=False)
    systemctl("reset-failed", source_unit, timeout=15, check=False)
    systemctl("start", source_unit, timeout=90, check=False)
    publish(
        status,
        state,
        status="transition_failed_core_resumed",
        error=f"{type(error).__name__}: {error}",
        core_active=active(source_unit),
        specialist_active=active(specialist_unit),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-unit", required=True)
    parser.add_argument("--bootstrap-unit", required=True)
    parser.add_argument("--specialist-unit", required=True)
    parser.add_argument("--specialist-run-dir", type=Path, required=True)
    parser.add_argument("--specialist-deployment-root", type=Path, required=True)
    parser.add_argument("--filtered-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-ready", type=Path, required=True)
    parser.add_argument("--handoff-ready", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--threshold-wr", type=float, default=0.40)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--force-after-iteration", type=int, default=-1)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--required-heldout-games", type=int, default=1000)
    parser.add_argument("--min-expert-decisions", type=int, default=100_000)
    args = parser.parse_args()

    run_dir = args.source_run_dir.expanduser().resolve()
    status_path = args.status.expanduser().resolve()
    state = read_json(status_path)
    if state.get("status") == "complete":
        # Reboot-idempotent: never revive the core after a completed handoff.
        frozen = verify_frozen_model(args.registry_root / "deck_agnostic_core")
        if not active(args.specialist_unit):
            systemctl("enable", args.specialist_unit, timeout=30, check=False)
            systemctl("start", args.specialist_unit, timeout=90)
        print(json.dumps({"status": "complete", "frozen": frozen}, indent=2))
        return 0
    anchor = state.get("anchor")
    if not isinstance(anchor, dict):
        anchor = inherited_anchor(
            run_dir,
            required_games=int(args.required_heldout_games),
            verify_bytes=True,
        )
        state = publish(
            status_path,
            state,
            status="watching_exact_heldout",
            source_run=str(run_dir),
            start_iteration=int(args.start_iteration),
            threshold_wr=float(args.threshold_wr),
            plateau_patience=int(args.plateau_patience),
            anchor=anchor,
        )

    decision: dict[str, Any] = {}
    while True:
        decision = transition_decision(
            run_dir,
            anchor=anchor,
            start_iteration=int(args.start_iteration),
            threshold_wr=float(args.threshold_wr),
            plateau_patience=int(args.plateau_patience),
            required_games=int(args.required_heldout_games),
            verify_best_bytes=True,
        )
        decision = apply_forced_iteration_boundary(
            decision, int(args.force_after_iteration)
        )
        handoff_ready: dict[str, Any] = {}
        handoff_wait_reason = ""
        if decision["triggered"]:
            try:
                handoff_ready = validate_handoff_ready(
                    args.handoff_ready,
                    filtered_manifest=args.filtered_manifest,
                    min_expert_decisions=int(args.min_expert_decisions),
                    specialist_deployment_root=args.specialist_deployment_root,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                handoff_wait_reason = f"{type(exc).__name__}: {exc}"
        state = publish(
            status_path,
            state,
            status=(
                "triggered_waiting_for_alakazam_specialist_build_ready"
                if decision["triggered"] and not handoff_ready
                else "triggered_waiting_for_expert_filter"
                if decision["triggered"]
                else "watching_exact_heldout"
            ),
            decision=decision,
            handoff_ready=handoff_ready,
            handoff_wait_reason=handoff_wait_reason,
        )
        if decision["triggered"] and handoff_ready:
            break
        time.sleep(max(1.0, float(args.poll_seconds)))

    best = dict(decision["best"])
    source_manifest_path = run_dir / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    handoff = read_json(run_dir / "lineage_handoff.json")
    frozen = freeze_model(
        registry_root=args.registry_root,
        family="deck_agnostic_core",
        display_name="Deck Agnostic Core",
        checkpoint=Path(str(best["checkpoint"])),
        expected_digest=str(best["checkpoint_digest"]),
        provenance={
            "source_run": run_dir.name,
            "source_iteration": int(best["iteration"]),
            "global_iteration_offset": int(handoff.get("global_iteration_offset") or 0),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256(source_manifest_path),
            "design_contract": source_manifest.get("design_contract"),
            "transition_policy": {
                "threshold_wr": float(args.threshold_wr),
                "plateau_patience": int(args.plateau_patience),
                "reason": decision["reason"],
                "exact_iterations_observed": decision["exact_iterations_observed"],
            },
        },
        evidence=best,
        require_exact_heldout=True,
    )
    state = publish(
        status_path,
        state,
        status="deck_agnostic_core_frozen",
        frozen_model=frozen,
    )

    # Do not take Blackwell away from core until the reusable filtered corpus
    # is complete and independently checksummed.
    while True:
        try:
            validate_filtered_manifest(
                args.filtered_manifest,
                min_decisions=int(args.min_expert_decisions),
            )
            break
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state = publish(
                status_path,
                state,
                status="deck_agnostic_core_frozen_waiting_for_expert_filter",
                filter_wait_reason=f"{type(exc).__name__}: {exc}",
            )
            time.sleep(max(5.0, float(args.poll_seconds)))

    stopped_core = False
    try:
        # The complete specialist build is already checksum-armed before the
        # exact gate can break this loop. At that boundary, release Blackwell
        # from core and use the full device-resident corpus for the one-time
        # warm start; then launch specialist RL without another idle gap.
        handoff_ready = validate_handoff_ready(
            args.handoff_ready,
            filtered_manifest=args.filtered_manifest,
            min_expert_decisions=int(args.min_expert_decisions),
            specialist_deployment_root=args.specialist_deployment_root,
        )
        state = publish(
            status_path,
            state,
            status="stopping_core_at_exact_gate_for_blackwell_bootstrap",
            handoff_ready=handoff_ready,
        )
        systemctl("stop", args.source_unit, timeout=90)
        stopped_core = True
        if active(args.source_unit):
            raise RuntimeError("core service remained active after exact gate")

        state = publish(
            status_path,
            state,
            status="training_alakazam_expert_bootstrap_blackwell_device_resident",
            core_active=False,
        )
        systemctl("reset-failed", args.bootstrap_unit, timeout=15, check=False)
        systemctl("start", args.bootstrap_unit, timeout=14400)
        ready = read_json(args.bootstrap_ready)
        if ready.get("status") != "ready":
            raise RuntimeError("Alakazam bootstrap service did not publish readiness")
        bootstrap_frozen = verify_frozen_model(
            args.registry_root / "alakazam_expert_bootstrap"
        )
        if ready.get("checkpoint_digest") != bootstrap_frozen.get("checkpoint_digest"):
            raise RuntimeError("Alakazam bootstrap ready/frozen identity mismatch")

        # Revalidate source/deployment identity after the one-time bootstrap
        # before the specialist service is allowed to consume the checkpoint.
        handoff_ready = validate_handoff_ready(
            args.handoff_ready,
            filtered_manifest=args.filtered_manifest,
            min_expert_decisions=int(args.min_expert_decisions),
            specialist_deployment_root=args.specialist_deployment_root,
        )
        state = publish(
            status_path,
            state,
            status="alakazam_specialist_bootstrap_ready_launching",
            handoff_ready=handoff_ready,
            bootstrap=ready,
            bootstrap_frozen=bootstrap_frozen,
            core_active=False,
        )

        state = publish(status_path, state, status="launching_alakazam_specialist")
        systemctl("disable", args.source_unit, timeout=30, check=False)
        systemctl("enable", args.specialist_unit, timeout=30)
        systemctl("reset-failed", args.specialist_unit, timeout=15, check=False)
        systemctl("start", args.specialist_unit, timeout=90)
        deadline = time.monotonic() + 300.0
        contract_seen = False
        while time.monotonic() < deadline:
            if not active(args.specialist_unit):
                result = systemctl(
                    "show", args.specialist_unit, "-p", "Result", "--value", check=False
                )
                if result == "failed":
                    raise RuntimeError("Alakazam specialist service entered failed state")
            manifest = read_json(args.specialist_run_dir / "manifest.json")
            log_path = ROOT / "outputs/logs/pure_rl_alakazam_public64k_v1_20260720.log"
            log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            if (
                manifest.get("mode") == "specialist"
                and manifest.get("specialist_archetype") == "alakazam"
                and manifest.get("our_decks") == ["alakazam"]
                and "self_play=9216 public_mix=64512" in log_text
                and "official_target=32256 diverse_public=32256" in log_text
            ):
                contract_seen = True
                break
            time.sleep(1.0)
        if not contract_seen:
            raise RuntimeError("Alakazam specialist did not publish its 9,216/64,512 contract")
        publish(
            status_path,
            state,
            status="complete",
            bootstrap=ready,
            specialist_run=str(args.specialist_run_dir),
            specialist_active=active(args.specialist_unit),
            core_enabled=False,
            public_games_per_iteration=64512,
            mirror_games_per_iteration=9216,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - explicit production rollback
        if stopped_core:
            rollback_to_core(
                source_unit=args.source_unit,
                bootstrap_unit=args.bootstrap_unit,
                specialist_unit=args.specialist_unit,
                state=state,
                status=status_path,
                error=exc,
            )
        else:
            # The only pre-stop failures are readiness/freeze failures; keep
            # or heal the current core rather than leaving production idle.
            if not active(args.source_unit):
                systemctl("reset-failed", args.source_unit, timeout=15, check=False)
                systemctl("start", args.source_unit, timeout=90, check=False)
            publish(
                status_path,
                state,
                status="specialist_preparation_failed_core_continues",
                error=f"{type(exc).__name__}: {exc}",
                core_active=active(args.source_unit),
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
