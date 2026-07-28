from __future__ import annotations

import json
from pathlib import Path

from scripts.run_population_round_robin import (
    _member_command,
    validate_contract,
)
from scripts.train_pure_rl import _parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_population_controller_contract_is_exact() -> None:
    contract = json.loads(
        (ROOT / "ops/population_round_robin_v1.json").read_text(
            encoding="utf-8"
        )
    )
    validate_contract(contract)
    assert contract["schedule"] == {
        "members": 22,
        "rl_iterations_per_member_cycle": 5,
        "expert_rehearsal_epochs_per_member_cycle": 5,
        "games_per_rl_iteration": 8192,
        "train_epochs_per_rl_iteration": 1,
        "expert_rehearsal_every": 5,
    }


def test_population_member_command_is_own_models_only() -> None:
    contract = json.loads(
        (ROOT / "ops/population_round_robin_v1.json").read_text(
            encoding="utf-8"
        )
    )
    state = {
        "active_member_index": 0,
        "population_cycle": 3,
        "members": [
            {
                "specialist_id": "starmie",
                "expert_manifest": "/expert/starmie.json",
                "current": {"checkpoint": "/models/starmie-current.pt"},
            }
        ],
    }
    command, run_dir = _member_command(contract, state)
    joined = " ".join(command)
    assert "--population-own-models-only" in command
    assert "--population-opponent-registry" in command
    assert "--official-collect-frac 0" in joined
    assert "--iterations 5" in joined
    assert "--games-per-iter 8192" in joined
    assert "--expert-rehearsal-every 5" in joined
    assert "--expert-rehearsal-epochs 5" in joined
    assert "--resume never" in joined
    assert run_dir.name == "population_starmie_cycle_0003"
    trainer_argv = command[command.index("--") + 1 :]
    parsed = _parse_args(
        [
            "--run-name",
            run_dir.name,
            "--mode",
            "specialist",
            *trainer_argv,
        ]
    )
    assert parsed.population_own_models_only is True
    assert parsed.population_opponent_registry is not None
    assert parsed.iterations == 5
    assert parsed.games_per_iter == 8192


def test_population_service_is_the_declared_terminal_target() -> None:
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert cycle["runtime"]["population_training_service"] == (
        "pokebot-population-round-robin.service"
    )
    unit = (
        ROOT / "deploy/systemd/pokebot-population-round-robin.service"
    ).read_text(encoding="utf-8")
    assert "scripts/run_population_round_robin.py" in unit


def test_population_controller_reuses_immutable_cycle_boundary() -> None:
    source = (
        ROOT / "scripts/run_population_round_robin.py"
    ).read_text(encoding="utf-8")
    assert "if boundary_path.is_file()" in source
    assert source.index("fleet = [") < source.index(
        "state = record_completed_member_cycle"
    )
