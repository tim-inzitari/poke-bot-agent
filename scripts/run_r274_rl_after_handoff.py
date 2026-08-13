#!/usr/bin/env python3
"""Wait for the tactical handoff, then exec the exact r274 25-update trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


DEFAULT_REMOTE_WORKER_ENDPOINTS = "192.168.1.143:8765,192.168.1.158:8766"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _emit(status: str, **fields: object) -> None:
    print(json.dumps({"status": status, **fields}, sort_keys=True), flush=True)


def _path(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)


def _adapter_training_contract(preservation_receipt: Path) -> tuple[int, Path]:
    """Resolve the inherited receipt-backed adapter continuation contract.

    The r195 parent already contains the trained V6 bank.  Ordinary full-model
    rehearsal deliberately freezes that bank, so each r274 RL update must keep
    the established isolated one-epoch adapter phase.  Resolve the immutable
    authorization from the checksum-bound preservation receipt instead of
    duplicating a mutable path in the service unit.
    """

    payload = json.loads(preservation_receipt.read_text(encoding="utf-8"))
    adapter = dict(payload.get("matchup_adapter") or {})
    provenance = dict(adapter.get("activation_provenance") or {})
    training = dict(adapter.get("training_activation") or {})
    epochs = int(adapter.get("epochs_per_rl_update") or 0)
    authorization = Path(str(training.get("path") or "")).expanduser().resolve()
    expected_digest = str(training.get("sha256") or "")
    expected_size = int(training.get("size_bytes") or -1)
    if (
        epochs != 1
        or provenance.get("matchup_adapter_bank_preserved") is not True
        or provenance.get("matchup_adapter_training_enabled") is not True
        or provenance.get("matchup_adapter_isolated_fit_continuation_required")
        is not True
        or provenance.get("matchup_adapter_isolated_bank_only_optimizer")
        is not True
        or not authorization.is_file()
        or authorization.stat().st_size != expected_size
        or sha256(authorization) != expected_digest
    ):
        raise RuntimeError("r274 adapter continuation contract is invalid")
    return epochs, authorization


def _remote_worker_endpoints() -> str:
    """Return the owner-selected r274 simulator farm endpoints.

    Revision 287 explicitly permits an empty value for Inzi-only production.
    The command builder converts that exact state into ``--no-remote-workers``
    so the trainer cannot fall back to its historical default endpoints.
    """

    endpoints = os.environ.get(
        "PURE_RL_REMOTE_WORKER_ENDPOINTS", DEFAULT_REMOTE_WORKER_ENDPOINTS
    ).strip()
    return endpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime_root",
        "python",
        "handoff_receipt",
        "initial_learner",
        "base_checkpoint",
        "preservation_receipt",
        "migration_receipt",
        "canary_activation_receipt",
        "sidecar_binding",
        "inzi_dataset_binding",
        "tactical_overlay",
        "r195_research_baseline_receipt",
        "contiguous_expert_pack",
        "contiguous_expert_pack_receipt",
        "active_gate_contract",
        "formal_holdout_contract",
        "r284_iteration_boundary_receipt",
        "frozen_specialist_registry",
        "research_control_registry",
        "expert_manifest",
        "guide_curriculum",
        "guide_head_roles",
        "guide_validation",
        "submission_boundary_dir",
    ):
        _path(parser, name)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    while not args.handoff_receipt.is_file():
        _emit("waiting_tactical_handoff", receipt=str(args.handoff_receipt))
        time.sleep(max(15.0, float(args.poll_seconds)))

    from poke_bot.r274_bootstrap_handoff import validate_handoff_receipt

    handoff = validate_handoff_receipt(
        args.handoff_receipt,
        expected_initial_checkpoint=args.initial_learner,
    )
    adapter_epochs, adapter_authorization = _adapter_training_contract(
        args.preservation_receipt
    )
    remote_worker_endpoints = _remote_worker_endpoints()
    required_files = (
        args.base_checkpoint,
        args.preservation_receipt,
        args.migration_receipt,
        args.canary_activation_receipt,
        args.sidecar_binding,
        args.inzi_dataset_binding,
        args.tactical_overlay,
        args.r195_research_baseline_receipt,
        args.contiguous_expert_pack,
        args.contiguous_expert_pack_receipt,
        args.active_gate_contract,
        args.formal_holdout_contract,
        args.r284_iteration_boundary_receipt,
        args.frozen_specialist_registry,
        args.research_control_registry,
        args.expert_manifest,
        args.guide_curriculum,
        args.guide_head_roles,
        args.guide_validation,
        adapter_authorization,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"r274 RL launch inputs are missing: {missing}")

    command = [
        str(args.python),
        "-u",
        str(args.runtime_root / "scripts/train_pure_rl.py"),
        "--run-name",
        "alakazam_new_list_direct_policy_r274",
        "--mode",
        "specialist",
        "--specialist-archetype",
        "alakazam",
        "--base-checkpoint",
        str(args.base_checkpoint),
        "--initial-learner-checkpoint",
        str(args.initial_learner),
        "--iterations",
        "20",
        "--fixed-cycle-updates",
        "20",
        "--r241-peak-r195-preservation-receipt",
        str(args.preservation_receipt),
        "--r241-peak-r195-preservation-receipt-sha256",
        sha256(args.preservation_receipt),
        "--r241-own-deck-migration-receipt",
        str(args.migration_receipt),
        "--r241-own-deck-migration-receipt-sha256",
        sha256(args.migration_receipt),
        "--r241-own-deck-canary-activation-receipt",
        str(args.canary_activation_receipt),
        "--r241-own-deck-canary-activation-receipt-sha256",
        sha256(args.canary_activation_receipt),
        "--r241-own-deck-sidecar-binding",
        str(args.sidecar_binding),
        "--r241-own-deck-sidecar-binding-sha256",
        sha256(args.sidecar_binding),
        "--r241-own-deck-inzi-dataset-binding",
        str(args.inzi_dataset_binding),
        "--r241-own-deck-inzi-dataset-binding-sha256",
        sha256(args.inzi_dataset_binding),
        "--r274-expert-tactical-overlay",
        str(args.tactical_overlay),
        "--r274-expert-tactical-overlay-sha256",
        sha256(args.tactical_overlay),
        "--r274-bootstrap-handoff-receipt",
        str(args.handoff_receipt),
        "--r274-bootstrap-handoff-receipt-sha256",
        sha256(args.handoff_receipt),
        "--r274-r195-research-baseline-receipt",
        str(args.r195_research_baseline_receipt),
        "--r274-r195-research-baseline-receipt-sha256",
        sha256(args.r195_research_baseline_receipt),
        "--r280-contiguous-expert-pack",
        str(args.contiguous_expert_pack),
        "--r280-contiguous-expert-pack-receipt",
        str(args.contiguous_expert_pack_receipt),
        "--r280-refresh-device",
        "cuda:1",
        "--r274-submission-boundary-dir",
        str(args.submission_boundary_dir),
        "--games-per-iter",
        "8196",
        "--require-exact-training-seat-split",
        "--official-collect-frac",
        "0.5",
        "--official-adaptive-min-share",
        "0.045",
        "--research-control-games-per-iter",
        "1000",
        "--active-gate-contract",
        str(args.active_gate_contract),
        "--formal-holdout-contract",
        str(args.formal_holdout_contract),
        "--r284-iteration-boundary-receipt",
        str(args.r284_iteration_boundary_receipt),
        "--frozen-specialist-registry",
        str(args.frozen_specialist_registry),
        "--research-control-registry",
        str(args.research_control_registry),
        "--measurement-decks",
        "alakazam",
        "--expert-manifest",
        str(args.expert_manifest),
        "--expert-manifest-workers",
        "32",
        "--expert-rehearsal-every",
        "5",
        "--expert-rehearsal-epochs",
        "5",
        "--terminal-expert-rehearsal",
        "--no-expert-rehearsal-before-first",
        "--expert-rehearsal-one-time-before",
        "-1",
        "--expert-rehearsal-one-time-epochs",
        "0",
        "--expert-rehearsal-batch-size",
        "2048",
        "--expert-rehearsal-guide-loss-weight",
        "0.0",
        "--current-deck-guide-loss-weight",
        "0.05",
        "--current-deck-guide-training-mode",
        "strategic_directional_v2",
        "--current-deck-guide-curriculum-spec",
        str(args.guide_curriculum),
        "--current-deck-guide-head-role-map",
        str(args.guide_head_roles),
        "--current-deck-guide-curriculum-validation-receipt",
        str(args.guide_validation),
        "--archetype-aux-loss-weight",
        "0.05",
        "--opp-hand-loss-weight",
        "0.05",
        "--opp-remainder-loss-weight",
        "0.05",
        "--lethal-threat-loss-weight",
        "0.025",
        "--prize-race-loss-weight",
        "0.025",
        "--setup-board-outcome-loss-weight",
        "0.025",
        "--combo-state-loss-weight",
        "0.0",
        "--dormant-matchup-adapter-epochs",
        str(adapter_epochs),
        "--dormant-matchup-adapter-activation-receipt",
        str(adapter_authorization),
        "--dormant-matchup-adapter-max-decisions-per-batch",
        "2048",
        "--train-max-decisions-per-batch",
        "2048",
        "--multi-env-per-worker",
        "4",
        "--leaf-eval",
        "gpu-server",
        "--resume",
        "auto",
        "--allow-clean-boundary-design-migration",
        "--boundary-design-migration-reason",
        "receipt_backed_completed_collection_resume_v1",
        "--seed",
        "274",
    ]
    if remote_worker_endpoints:
        command[command.index("--resume"):command.index("--resume")] = [
            "--remote-worker-endpoints",
            remote_worker_endpoints,
            "--heldout-remotes",
        ]
    else:
        command[command.index("--resume"):command.index("--resume")] = [
            "--no-remote-workers",
        ]
    _emit(
        "launching_rl_update_0",
        initial_learner=handoff["activated_checkpoint"],
        updates=20,
        games_per_update=8196,
        multi_env_per_worker=4,
        remote_worker_endpoints=remote_worker_endpoints,
        simulator_workers=int(os.environ.get("SIM_WORKERS", "32")),
    )
    os.execv(str(args.python), command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
