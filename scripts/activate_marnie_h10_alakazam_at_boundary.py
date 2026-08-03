#!/usr/bin/env python3
"""Activate Marnie's scoped H10 Alakazam roster after iteration 1 commits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


SERVICE = "pokebot-final-format-marnie-r104-h10-rl.service"
EXPECTED_CHECKPOINT = (
    "sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92"
)
EXPECTED_OPPONENT = "specialist-alakazam-final-format-h10-02c014ad7c33"
EXPECTED_GATE_ID = (
    "specialist-strong-public-roster-sw80-at-iter5-v1+frozen-specialists-r14-r109"
)
DEPLOYMENT = Path(
    "/home/inzi/poke-bot-agent-deployments/final-format-marnie-h10-r104"
)
RUN = DEPLOYMENT / "outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k"
COMMIT = RUN / "commits/iter_00001.json"
NEXT_PLAN = RUN / "collection_plans/iter_00002.json"
STAGED_SCRIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/staging/"
    "register_final_format_marnie_h10_rl_r108.py"
)
ACTIVE_SCRIPT = DEPLOYMENT / "scripts/register_final_format_marnie_h10_rl.py"
SCOPED_REGISTRY = (
    DEPLOYMENT / "ops/frozen_specialist_registry_marnie_r108_h10_alakazam.json"
)
GATE = DEPLOYMENT / "runtime/final_format_marnie_gate_r108_h10_alakazam.json"
RUNTIME_REGISTRY = Path(
    "/home/inzi/poke-bot-agent/outputs/final_format_marnie_r104/runtime/"
    "specialist_runtime_registry_h10_r104_fusion_v3.json"
)
STAGE_RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final-format-marnie-h10-alakazam-opponent-stage-r108.json"
)
ACTIVATION_RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final-format-marnie-h10-alakazam-opponent-activation-r108.json"
)
FINAL_ACTIVATION_RECEIPT = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final-format-marnie-h10-alakazam-opponent-activation-r109.json"
)
DROPIN = Path.home() / (
    ".config/systemd/user/" + SERVICE + ".d/zz-r108-boundary.conf"
)
RACED_QUARANTINE = RUN / "quarantine/iter_00002/boundary_r108_r109"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def _wait_service(state: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = _run(
            "systemctl", "--user", "show", SERVICE, "-p", "ActiveState", "--value"
        ).stdout.strip()
        main_pid = int(
            _run(
                "systemctl", "--user", "show", SERVICE, "-p", "MainPID", "--value"
            ).stdout.strip()
            or 0
        )
        if current == state or (
            state == "inactive" and current in {"inactive", "failed"} and main_pid == 0
        ):
            return
        time.sleep(1.0)
    raise RuntimeError(f"{SERVICE} did not reach {state}")


def _quarantine_raced_iteration_2() -> list[dict[str, Any]]:
    """Preserve, but never reuse, work opened under the pre-boundary roster."""

    sources = [
        NEXT_PLAN,
        RUN / "shards/iter_00002.jsonl",
        RUN / "checkpoints/iter_00002.pt",
        RUN / "eval/iter_00002.json",
        RUN / "metrics/iter_00002.json",
        RUN / "research_controls/iter_00002.json",
        RUN / "collection_receipts/iter_00002.json",
    ]
    sources.extend(sorted((RUN / "shards").glob(".iter_00002*.tmp")))
    sources.extend(sorted((RUN / "checkpoints").glob("iter_00002.pt.tmp.*")))
    runtime = RUN / "iteration_runtime.json"
    if runtime.is_file():
        try:
            if int(_read(runtime).get("iteration", -1)) == 2:
                sources.append(runtime)
        except (TypeError, ValueError):
            sources.append(runtime)
    latest = RUN / "metrics/latest.json"
    if latest.is_file():
        try:
            if int(_read(latest).get("iteration", -1)) == 2:
                sources.append(latest)
        except (TypeError, ValueError):
            sources.append(latest)

    moved: list[dict[str, Any]] = []
    for source in dict.fromkeys(path for path in sources if path.exists()):
        relative = source.relative_to(RUN)
        destination = RACED_QUARANTINE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = _sha256(source)
        if destination.exists():
            if _sha256(destination) != digest:
                raise RuntimeError(
                    f"raced iteration-2 quarantine collision: {destination}"
                )
            raise RuntimeError(
                f"raced iteration-2 source and quarantine both exist: {source}"
            )
        os.replace(source, destination)
        moved.append(
            {
                "source": str(source),
                "quarantine": str(destination),
                "sha256": digest,
            }
        )
    return moved


def _register_runtime() -> None:
    python = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
    command = [
        python,
        "-u",
        str(ACTIVE_SCRIPT),
        "--bootstrap-ready",
        "/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-bootstrap-ready.json",
        "--bootstrap-family",
        "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/final-format-marnie-r104-h10-expert-bootstrap-v1",
        "--deployment-root",
        str(DEPLOYMENT),
        "--template-registry",
        "/home/inzi/poke-bot-agent/outputs/final_format_alakazam_r79/runtime/specialist_runtime_registry_h10_r104_fusion_v3_directional_iter20_exact.json",
        "--guide",
        str(DEPLOYMENT / "config/deck_guides/marnie-s-grimmsnarl-ex.yaml"),
        "--expert-manifest",
        "/home/inzi/poke-bot-agent/data/bootstrap/expert-latest20-2026-07-04-2026-07-23-roster18-v6-strategic/marnie-s-grimmsnarl-ex/PROTECTED_EXPERT_CORPUS.json",
        "--matchup-tree",
        "/home/inzi/poke-bot-agent/outputs/state/slowking-public-matchup-tree-v33.json",
        "--curriculum-spec",
        str(DEPLOYMENT / "state/final_format_marnie_curriculum_r104_h10_19/marnie-s-grimmsnarl-ex-strategic-curriculum-r104.json"),
        "--head-role-map",
        str(DEPLOYMENT / "state/final_format_marnie_curriculum_r104_h10_19/marnie-s-grimmsnarl-ex-strategic-head-roles-r104.json"),
        "--curriculum-validation",
        str(DEPLOYMENT / "state/final_format_marnie_curriculum_r104_h10_19/marnie-s-grimmsnarl-ex-strategic-curriculum-validation-r104.json"),
        "--output-registry",
        str(RUNTIME_REGISTRY),
        "--template-selector-env",
        str(DEPLOYMENT / "config/specialist_runtime.env"),
        "--selector-env",
        "/home/inzi/poke-bot-agent/outputs/final_format_marnie_r104/runtime/specialist_runtime_h10_r104.env",
        "--receipt",
        "/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-runtime-registration.json",
        "--router-v6-family",
        "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/final-format-marnie-r104-h10-router-v6-bootstrap-v1",
        "--router-v6-registry",
        str(DEPLOYMENT / "state/matchup_adapter_roster.json"),
        "--router-v6-receipt",
        "/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-router-v6-migration.json",
    ]
    result = _run(*command)
    if "registered_ready_for_managed_rl" not in result.stdout:
        raise RuntimeError("revision-109 Marnie runtime registration failed")


def _validate_stage() -> dict[str, Any]:
    stage = _read(STAGE_RECEIPT)
    registry = _read(SCOPED_REGISTRY)
    gate = _read(GATE)
    frozen = [
        row
        for row in (registry.get("specialists") or [])
        if str(row.get("specialist_id") or "") == "alakazam"
    ]
    roster = [
        row
        for row in (dict(gate.get("next_gate") or {}).get("roster") or [])
        if str(row.get("archetype_id") or "") == "alakazam"
    ]
    if (
        stage.get("status") != "staged_for_next_committed_iteration_boundary"
        or stage.get("checkpoint_digest") != EXPECTED_CHECKPOINT
        or len(registry.get("specialists") or []) != 14
        or len(frozen) != 1
        or frozen[0].get("opponent_id") != EXPECTED_OPPONENT
        or frozen[0].get("checkpoint_digest") != EXPECTED_CHECKPOINT
        or len(roster) != 1
        or roster[0].get("opponent_id") != EXPECTED_OPPONENT
        or roster[0].get("frozen_checkpoint_digest") != EXPECTED_CHECKPOINT
        or not STAGED_SCRIPT.is_file()
    ):
        raise RuntimeError("Marnie H10 Alakazam stage validation failed")
    return stage


def _finalize_revision_109_receipt() -> dict[str, Any]:
    """Bind the corrected SW80 gate identity and live iteration-2 plan."""

    if FINAL_ACTIVATION_RECEIPT.exists():
        return _read(FINAL_ACTIVATION_RECEIPT)
    original = _read(ACTIVATION_RECEIPT)
    runtime = _read(RUNTIME_REGISTRY)
    gate = _read(GATE)
    plan = _read(NEXT_PLAN)
    next_gate = dict(gate.get("next_gate") or {})
    criteria = dict(next_gate.get("pass_criteria") or {})
    alakazam = [
        (opponent_id, dict(row or {}))
        for opponent_id, row in dict(plan.get("per_opponent") or {}).items()
        if str(dict(row or {}).get("archetype_id") or "") == "alakazam"
    ]
    if (
        runtime.get("owner_decision_revision") != 109
        or runtime.get("terminal_active_gate_id") != EXPECTED_GATE_ID
        or gate.get("owner_decision_revision") != 109
        or gate.get("active_gate_id") != EXPECTED_GATE_ID
        or next_gate.get("id") != EXPECTED_GATE_ID
        or float(criteria.get("skill_weighted_win_rate", -1.0)) != 0.80
        or float(criteria.get("skill_weighted_confidence_lower", -1.0)) != 0.50
        or int(plan.get("iteration", -1)) != 2
        or plan.get("active_gate_id") != EXPECTED_GATE_ID
        or len(alakazam) != 1
        or alakazam[0][0] != EXPECTED_OPPONENT
    ):
        raise RuntimeError("revision-109 Marnie activation evidence is incomplete")
    main_pid = int(
        _run("systemctl", "--user", "show", SERVICE, "-p", "MainPID", "--value")
        .stdout.strip()
        or 0
    )
    if main_pid <= 0:
        raise RuntimeError("Marnie service is not live for revision-109 finalization")
    receipt = {
        "schema": "poke_bot.marnie_h10_alakazam_sw80_activation/v1",
        "status": "activated_iteration_2_collecting",
        "owner_decision_revision": 109,
        "opponent_roster_owner_decision_revision": 108,
        "activated_at_utc": original.get("activated_at_utc"),
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "superseded_intermediate_activation_receipt": str(ACTIVATION_RECEIPT),
        "superseded_intermediate_activation_receipt_sha256": _sha256(
            ACTIVATION_RECEIPT
        ),
        "boundary_commit": str(COMMIT),
        "boundary_commit_sha256": _sha256(COMMIT),
        "runtime_registry": str(RUNTIME_REGISTRY),
        "runtime_registry_sha256": _sha256(RUNTIME_REGISTRY),
        "active_gate": str(GATE),
        "active_gate_sha256": _sha256(GATE),
        "active_gate_id": EXPECTED_GATE_ID,
        "terminal_skill_weighted_win_rate_required": 0.80,
        "terminal_skill_weighted_confidence_lower_required": 0.50,
        "terminal_strength_gate_blocks_iteration_1_to_2": False,
        "iteration_2_plan": str(NEXT_PLAN),
        "iteration_2_plan_sha256": _sha256(NEXT_PLAN),
        "iteration_2_alakazam_opponent_id": EXPECTED_OPPONENT,
        "iteration_2_alakazam_games": int(alakazam[0][1].get("games") or 0),
        "iteration_2_alakazam_seat0": int(alakazam[0][1].get("seat0") or 0),
        "iteration_2_alakazam_seat1": int(alakazam[0][1].get("seat1") or 0),
        "managed_service": SERVICE,
        "managed_main_pid": main_pid,
        "research_control_changed": False,
        "historical_v5_rewritten": False,
    }
    _atomic_json(FINAL_ACTIVATION_RECEIPT, receipt)
    return receipt


def main() -> int:
    if ACTIVATION_RECEIPT.exists():
        print(json.dumps(_finalize_revision_109_receipt(), sort_keys=True))
        return 0
    stage = _validate_stage()
    while not COMMIT.is_file():
        time.sleep(1.0)
    commit_digest = _sha256(COMMIT)

    # Prevent a deliberate boundary stop from invoking the terminal handler.
    DROPIN.parent.mkdir(parents=True, exist_ok=True)
    DROPIN.write_text("[Unit]\nOnSuccess=\n", encoding="utf-8")
    _run("systemctl", "--user", "daemon-reload")
    _run("systemctl", "--user", "stop", SERVICE)
    _wait_service("inactive")
    raced_iteration_2_artifacts = _quarantine_raced_iteration_2()

    temporary = ACTIVE_SCRIPT.with_name(f".{ACTIVE_SCRIPT.name}.r108.tmp")
    shutil.copy2(STAGED_SCRIPT, temporary)
    os.replace(temporary, ACTIVE_SCRIPT)
    _register_runtime()
    _run("systemctl", "--user", "start", SERVICE)
    _wait_service("active")

    runtime = _read(RUNTIME_REGISTRY)
    if (
        runtime.get("owner_decision_revision") != 109
        or runtime.get("active_gate_contract")
        != "runtime/final_format_marnie_gate_r108_h10_alakazam.json"
        or runtime.get("frozen_specialist_registry")
        != "ops/frozen_specialist_registry_marnie_r108_h10_alakazam.json"
    ):
        raise RuntimeError("managed Marnie runtime did not activate revision 109")

    # Restore the ordinary terminal success chain only after the new process is live.
    DROPIN.unlink(missing_ok=True)
    _run("systemctl", "--user", "daemon-reload")
    main_pid = _run(
        "systemctl", "--user", "show", SERVICE, "-p", "MainPID", "--value"
    ).stdout.strip()
    receipt = {
        "schema": "poke_bot.marnie_h10_alakazam_opponent_activation/v1",
        "status": "activated",
        "owner_decision_revision": 109,
        "opponent_roster_owner_decision_revision": 108,
        "terminal_strength_owner_decision_revision": 109,
        "iteration_1_to_2_continuation_owner_decision_revision": 109,
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_commit": str(COMMIT),
        "boundary_commit_sha256": commit_digest,
        "checkpoint_digest": EXPECTED_CHECKPOINT,
        "opponent_id": EXPECTED_OPPONENT,
        "practice_public_mix_active": True,
        "formal_holdout_active": True,
        "terminal_skill_weighted_win_rate_required": 0.80,
        "terminal_skill_weighted_confidence_lower_required": 0.50,
        "terminal_strength_gate_blocks_iteration_1_to_2": False,
        "raced_pre_boundary_iteration_2_artifacts": raced_iteration_2_artifacts,
        "research_control_changed": False,
        "historical_v5_rewritten": False,
        "scoped_frozen_registry": str(SCOPED_REGISTRY),
        "scoped_frozen_registry_sha256": _sha256(SCOPED_REGISTRY),
        "active_gate": str(GATE),
        "active_gate_sha256": _sha256(GATE),
        "runtime_registry": str(RUNTIME_REGISTRY),
        "runtime_registry_sha256": _sha256(RUNTIME_REGISTRY),
        "managed_service": SERVICE,
        "managed_main_pid": int(main_pid),
        "stage_receipt": str(STAGE_RECEIPT),
        "stage_receipt_sha256": _sha256(STAGE_RECEIPT),
        "stage_payload_digest": stage.get("receipt_payload_digest"),
    }
    _atomic_json(ACTIVATION_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
