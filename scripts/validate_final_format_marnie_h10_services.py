#!/usr/bin/env python3
"""Validate the dormant managed Marnie H10 materialize/bootstrap chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


SERVICES = (
    "pokebot-final-format-marnie-r79-handoff.service",
    "pokebot-post-alakazam-core-refresh-r104.service",
    "pokebot-final-format-marnie-r104-h10-materialize.service",
    "pokebot-final-format-marnie-r104-h10-bootstrap.service",
    "pokebot-final-format-marnie-r104-h10-register.service",
    "pokebot-final-format-marnie-r104-h10-rl.service",
    "pokebot-final-format-marnie-r104-h10-gate-handler.service",
    "pokebot-final-format-marnie-r104-completion.service",
    "pokebot-post-refresh-capacity-boundary-r104.service",
    "pokebot-population-round-robin-handoff.service",
    "pokebot-population-round-robin.service",
)
RUNTIME_ASSET_SHA256 = {
    "submission/main.py": (
        "sha256:30deadd1d8dc7e3d885dd107600468e5d0dcd6e9227c86fba52504e85fe0b70f"
    ),
    "submission/search_config.json": (
        "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
    ),
    "data/training_mixes/top_ladder.v1.json": (
        "sha256:de9f2f5f65794ed8f0ffd3fff41b04aafd68f7ffc9d562ab2cfc774ea5cac79d"
    ),
    "data/training_mixes/top_ladder_representatives.v1.json": (
        "sha256:ee42f146fd746ed3dd953515a974b22eeb264ea8eb9513c6e69a41a524454002"
    ),
    "data/training_mixes/specialist_representatives.v1.json": (
        "sha256:47f624f35ab5f497055b3059cdf6dfd7d7e06312ccf3e87b01910d6b142c3a35"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"Marnie H10 service receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-root", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    unit_root = args.unit_root.expanduser().resolve()
    deployment = args.deployment_root.expanduser().resolve()
    cg_lib_path = Path("/home/pokebot/poke-bot-agent/kaggle/input/cg-lib").resolve()
    unit_paths = {name: unit_root / name for name in SERVICES}
    required_files = (
        deployment / "scripts/materialize_final_format_marnie_h10.py",
        deployment / "scripts/run_post_alakazam_core_refresh.py",
        deployment / "scripts/run_final_format_marnie_h10_bootstrap.py",
        deployment / "scripts/register_final_format_marnie_h10_rl.py",
        deployment / "scripts/launch_active_specialist.py",
        deployment / "scripts/launch_active_specialist_gate_handler.py",
        deployment / "scripts/complete_final_format_marnie_refresh.py",
        deployment / "scripts/finalize_post_refresh_capacity_boundary.py",
        deployment / "scripts/validate_final_format_marnie_h10.py",
        deployment / "scripts/prepare_population_round_robin.py",
        deployment / "scripts/population_round_robin_state.py",
        deployment / "scripts/materialize_population_opponents.py",
        deployment / "scripts/run_population_round_robin.py",
        deployment / "ops/population_round_robin_v1.json",
        deployment / "ops/specialist_cycle_handoff_v1.json",
        deployment / "scripts/migrate_final_format_alakazam_h10.py",
        deployment / "poke_bot/h10_migration.py",
        deployment / "poke_bot/model.py",
        deployment / "poke_bot/train.py",
        deployment / "config/rl_protocol.yaml",
        deployment / "config/final_format_marnie_h10_runtime.env",
        deployment / "baselines/manifest.json",
        deployment / "submission/main.py",
        deployment / "submission/search_config.json",
        deployment / "data/training_mixes/top_ladder.v1.json",
        deployment / "data/training_mixes/top_ladder_representatives.v1.json",
        deployment / "data/training_mixes/specialist_representatives.v1.json",
        deployment / "state/matchup_adapter_roster.json",
        deployment
        / "state/final_format_marnie_curriculum_r104_h10_19"
        / "marnie-s-grimmsnarl-ex-strategic-curriculum-r104.json",
        deployment
        / "state/final_format_marnie_curriculum_r104_h10_19"
        / "marnie-s-grimmsnarl-ex-strategic-head-roles-r104.json",
        deployment
        / "state/final_format_marnie_curriculum_r104_h10_19"
        / "marnie-s-grimmsnarl-ex-strategic-curriculum-validation-r104.json",
    )
    missing = [str(path) for path in (*unit_paths.values(), *required_files) if not path.is_file()]
    if not (cg_lib_path / "cg").is_dir():
        missing.append(str(cg_lib_path / "cg"))
    if missing:
        raise RuntimeError(f"Marnie H10 managed-chain inputs are missing: {missing}")
    changed_assets = [
        relative
        for relative, expected in RUNTIME_ASSET_SHA256.items()
        if _sha256(deployment / relative) != expected
    ]
    if changed_assets:
        raise RuntimeError(
            f"Marnie H10 deployment training-mix digest changed: {changed_assets}"
        )
    handoff = unit_paths[SERVICES[0]].read_text(encoding="utf-8")
    core_refresh = unit_paths[SERVICES[1]].read_text(encoding="utf-8")
    materialize = unit_paths[SERVICES[2]].read_text(encoding="utf-8")
    bootstrap = unit_paths[SERVICES[3]].read_text(encoding="utf-8")
    register = unit_paths[SERVICES[4]].read_text(encoding="utf-8")
    rl = unit_paths[SERVICES[5]].read_text(encoding="utf-8")
    gate_handler = unit_paths[SERVICES[6]].read_text(encoding="utf-8")
    completion = unit_paths[SERVICES[7]].read_text(encoding="utf-8")
    capacity_boundary = unit_paths[SERVICES[8]].read_text(encoding="utf-8")
    population_handoff = unit_paths[SERVICES[9]].read_text(encoding="utf-8")
    population = unit_paths[SERVICES[10]].read_text(encoding="utf-8")
    required_handoff = (
        "After=pokebot-final-format-alakazam-r79-completion.service",
        "OnSuccess=pokebot-post-alakazam-core-refresh-r104.service",
        "prepare_final_format_marnie_refresh.py",
    )
    required_core_refresh = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-alakazam-r79-h10-completion-v1.json",
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r79-prestage-r104-h10-19.json",
        "run_post_alakazam_core_refresh.py",
        "--next-service pokebot-final-format-marnie-r104-h10-materialize.service",
    )
    required_materialize = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r79-prestage-r104-h10-19.json",
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/post-alakazam-core-refresh-boundary-r104.json",
        "OnSuccess=pokebot-final-format-marnie-r104-h10-bootstrap.service",
        "Environment=CG_LIB_PATH=/home/pokebot/poke-bot-agent/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg",
        "materialize_final_format_marnie_h10.py",
    )
    required_bootstrap = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-materialization.json",
        "Environment=CG_LIB_PATH=/home/pokebot/poke-bot-agent/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg",
        "POKEBOT_H10_CAPACITY_ENABLED=1",
        "POKEBOT_DECISION_FUSION_TYPED_OUTPUT_CENTERED_ROUTES_ENABLED=1",
        "OnSuccess=pokebot-final-format-marnie-r104-h10-register.service",
        "--epochs 25",
        "--batch-size 2048",
        "run_final_format_marnie_h10_bootstrap.py",
    )
    if any(value not in handoff for value in required_handoff):
        raise RuntimeError("Marnie pre-stage unit lost its core-refresh handoff")
    if any(value not in core_refresh for value in required_core_refresh):
        raise RuntimeError("post-Alakazam core unit lost a boundary invariant")
    if any(value not in materialize for value in required_materialize):
        raise RuntimeError("Marnie H10 materialize unit lost a boundary invariant")
    if any(value not in bootstrap for value in required_bootstrap):
        raise RuntimeError("Marnie H10 bootstrap unit lost an architecture invariant")
    required_register = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-bootstrap-ready.json",
        "OnSuccess=pokebot-final-format-marnie-r104-h10-rl.service",
        "register_final_format_marnie_h10_rl.py",
        "final-format-marnie-r104-h10-expert-bootstrap-v1",
        "specialist_runtime_registry_h10_r104_fusion_v3_directional_iter20_exact.json",
    )
    required_rl = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-runtime-registration.json",
        "Environment=PURE_RL_SPATIAL_LAYERS=7",
        "Environment=PURE_RL_TEMPORAL_LAYERS=3",
        "Environment=PURE_RL_OPTION_DECODER_LAYERS=7",
        "Environment=PURE_RL_FF_DIM=2496",
        "Environment=POKEBOT_H10_CAPACITY_ENABLED=1",
        "Environment=POKEBOT_H10_HEAD_RESIDUAL_WIDTH=512",
        "Environment=POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=1",
        "Environment=POKEBOT_COMBO_STATE_HEAD_ENABLED=1",
        "Environment=POKEBOT_DECISION_FUSION_TYPED_OUTPUT_CENTERED_ROUTES_ENABLED=1",
        "Environment=POKEBOT_MATCHUP_ADAPTER_FORMAT=poke-bot-matchup-adapter-bank-v6",
        "Environment=POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH=/home/pokebot/poke-bot-agent-deployments/final-format-marnie-h10-r104/state/matchup_adapter_roster.json",
        "Environment=PURE_RL_SIM_WORKERS=96",
        "Environment=PURE_RL_GAMES_IN_FLIGHT=96",
        "Environment=PURE_RL_HELDOUT_LOCAL_WORKERS=96",
        "Environment=PURE_RL_REBALANCE_MIN_WORKERS=96",
        "Environment=PURE_RL_REBALANCE_MAX_WORKERS=96",
        "Environment=POKEBOT_LIVE_POOL_MAX_WORKERS=96",
        "Environment=POKEBOT_BERT_REMOTE_ROOT=/Users/example/workspace/poke-bot-agent-h10-r79-stage",
        "EnvironmentFile=/home/pokebot/poke-bot-agent-deployments/final-format-marnie-h10-r104/config/final_format_marnie_h10_runtime.env",
        "ExecStartPre=/usr/bin/test -s /home/pokebot/poke-bot-agent-deployments/final-format-marnie-h10-r104/baselines/manifest.json",
        "ExecStartPre=/usr/bin/test -d /home/pokebot/poke-bot-agent-deployments/final-format-marnie-h10-r104/decks",
        "ExecStartPre=/usr/bin/test -d /home/pokebot/poke-bot-agent-deployments/final-format-marnie-h10-r104/kaggle",
        "register_final_format_marnie_h10_rl.py",
        "launch_active_specialist.py",
        "launch_active_specialist_gate_handler.py",
        "complete_final_format_marnie_refresh.py --check",
        "OnSuccess=pokebot-final-format-marnie-r104-h10-gate-handler.service",
    )
    required_gate_handler = (
        "EnvironmentFile=/home/pokebot/poke-bot-agent/outputs/final_format_marnie_r104/runtime/specialist_runtime_h10_r104.env",
        "launch_active_specialist_gate_handler.py",
        "specialist_runtime_registry_h10_r104_fusion_v3.json",
    )
    if any(value not in register for value in required_register):
        raise RuntimeError("Marnie H10 registration unit lost a boundary invariant")
    if any(value not in rl for value in required_rl):
        raise RuntimeError("Marnie H10 RL unit lost an architecture invariant")
    if any(value not in gate_handler for value in required_gate_handler):
        raise RuntimeError("Marnie H10 gate-handler unit lost a boundary invariant")
    required_completion = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-passed-gate-handler-v1.json",
        "complete_final_format_marnie_refresh.py --check",
        "complete_final_format_marnie_refresh.py --handler-state",
        "marnie-s-grimmsnarl-ex-protocol-gate-pass-v1/model.pt",
        "final-format-marnie-r104-h10-completion-v1.json",
        "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service",
        "--next-unit /home/pokebot/.config/systemd/user/pokebot-final-format-crustle-r113-h10-bootstrap.service",
    )
    if any(value not in completion for value in required_completion):
        raise RuntimeError("Marnie H10 completion unit lost a terminal invariant")
    required_capacity_boundary = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-alakazam-r79-h10-completion-v1.json",
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-completion-v1.json",
        "--crustle-completion /home/pokebot/poke-bot-agent/outputs/state/final-format-crustle-r113-h10-completion-v1.json",
        "finalize_post_refresh_capacity_boundary.py --check",
        "finalize_post_refresh_capacity_boundary.py --alakazam-completion",
        "post_refresh_sequence_complete_for_capacity_v2.json",
        "--next-service pokebot-population-round-robin-handoff.service",
    )
    if any(value not in capacity_boundary for value in required_capacity_boundary):
        raise RuntimeError("post-refresh capacity boundary lost a release invariant")
    required_population_handoff = (
        "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/post_refresh_sequence_complete_for_capacity_v2.json",
        "prepare_population_round_robin.py",
    )
    required_population = (
        "run_population_round_robin.py --contract ops/population_round_robin_v1.json --check",
        "run_population_round_robin.py --contract ops/population_round_robin_v1.json",
    )
    if any(value not in population_handoff for value in required_population_handoff):
        raise RuntimeError("population handoff lost its capacity boundary")
    if any(value not in population for value in required_population):
        raise RuntimeError("population service lost its controller boundary")
    population_contract = json.loads(
        (deployment / "ops/population_round_robin_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        population_contract.get("schema")
        != "poke_bot.population_round_robin_controller_contract/v1"
        or int((population_contract.get("schedule") or {}).get("members") or 0)
        != 14
        or int(
            (population_contract.get("training") or {}).get(
                "train_max_decisions_per_batch"
            )
            or 0
        )
        != 2048
    ):
        raise RuntimeError("population controller is not the 14-member safe-batch contract")
    service_state: dict[str, dict[str, str]] = {}
    for service in SERVICES:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                service,
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "--value",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"cannot inspect Marnie managed service: {service}")
        values = result.stdout.splitlines()
        if values != ["loaded", "inactive", "dead"]:
            raise RuntimeError(f"Marnie service started before its boundary: {service} {values}")
        service_state[service] = {
            "load_state": values[0],
            "active_state": values[1],
            "sub_state": values[2],
        }
    receipt = {
        "schema": "poke_bot.final_format_marnie_h10_managed_chain_stage/v13",
        "status": "staged_inactive_waiting_for_alakazam_iter20_and_core_boundary",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "deployment_root": str(deployment),
        "capacity_profile": "H10-I/v1",
        "architecture": {
            "spatial_layers": 7,
            "temporal_layers": 3,
            "option_decoder_layers": 7,
            "feed_forward_width": 2496,
            "strategic_head_residual_width": 512,
        },
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "exact_bootstrap_epochs": 25,
        "cg_lib_path": str(cg_lib_path),
        "cg_runtime_verified": True,
        "training_before_h10_migration_allowed": False,
        "unit_sha256": {name: _sha256(path) for name, path in unit_paths.items()},
        "deployment_file_sha256": {
            str(path.relative_to(deployment)): _sha256(path) for path in required_files
        },
        "service_state": service_state,
        "training_authority": False,
        "selector_authority": False,
        "next_boundary": "post_alakazam_core_refresh_boundary_receipt",
        "terminal_managed_rl_service": "pokebot-final-format-marnie-r104-h10-rl.service",
        "terminal_completion_service": "pokebot-final-format-marnie-r104-completion.service",
        "post_refresh_capacity_boundary_service": "pokebot-post-refresh-capacity-boundary-r104.service",
        "registration_requires_exact_h10_bootstrap_receipt": True,
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
