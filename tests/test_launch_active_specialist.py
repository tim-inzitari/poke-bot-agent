from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.launch_active_specialist import (
    _build_command,
    _load_registry,
    _resolve,
    _sha256,
)


def _fixture(tmp_path: Path, *, status: str = "ready") -> Path:
    root = tmp_path / "runtime"
    (root / "scripts").mkdir(parents=True)
    (root / "ops").mkdir()
    checkpoint = tmp_path / "model.pt"
    expert = tmp_path / "expert.json"
    routes = ["trevenant", *[f"route-{index}" for index in range(21)]]
    torch.save(
        {
            "model_state_dict": {
                f"matchup_adapter_bank.experts.{index}.up.weight": torch.zeros(1)
                for index in range(22)
            },
            "extra": {"matchup_adapter_config": {"expert_ids": routes}},
        },
        checkpoint,
    )
    expert.write_text("{}", encoding="utf-8")
    runtime_tree = tmp_path / "runtime-tree.json"
    runtime_tree.write_text(
        json.dumps(
            {
                "runtime_enabled": True,
                "targets": routes,
                "runtime_contract": {
                    "accepted_archetype_ids": ["trevenant"],
                    "one_route_per_decision": True,
                    "unknown_route_exact_bypass": True,
                },
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "adapter-authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_adapter_rehearsal_authorization/v1",
                "optimizer_scope": "matchup_adapter_bank_only",
                "runtime_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    for relative in ("scripts/launch_pure_rl.py", "ops/research.json"):
        path = root / relative
        path.write_text("{}", encoding="utf-8")
    digest = "sha256:" + "1" * 64
    (root / "ops/frozen.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "specialist_id": "alakazam",
                        "opponent_id": "specialist-alakazam",
                        "checkpoint_digest": digest,
                        "frozen": True,
                        "public_mix_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "ops/gate.json").write_text(
        json.dumps(
            {
                "next_gate": {
                    "id": "gate-v1+frozen-specialists-r1",
                    "evaluation": {
                        "games_total": 2250,
                        "games_per_opponent": 250,
                        "seat0_games_per_opponent": 125,
                        "seat1_games_per_opponent": 125,
                    },
                    "roster": [
                        *[
                            {"opponent_id": f"public-{index}"}
                            for index in range(8)
                        ],
                        {
                            "opponent_id": "specialist-alakazam",
                            "tier": "S+",
                            "frozen_specialist": True,
                            "frozen_checkpoint_digest": digest,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema": "poke_bot.specialist_runtime_registry/v1",
        "version": 1,
        "selector_environment_variable": "POKEBOT_ACTIVE_SPECIALIST",
        "runtime_root": str(root),
        "python": "/usr/bin/python3",
        "launcher": "scripts/launch_pure_rl.py",
        "active_gate_contract": "ops/gate.json",
        "research_control_registry": "ops/research.json",
        "frozen_specialist_registry": "ops/frozen.json",
        "terminal_active_gate_id": "gate-v1",
        "minimum_terminal_iteration": 5,
        "iteration_ceiling": 15,
        "common_launcher_args": ["--mode", "specialist"],
        "common_trainer_args": [
            "--expert-min-decisions",
            "20000",
            "--continue-after-gate",
            "--resume",
            "auto",
        ],
        "specialists": {
            "trevenant": {
                "status": status,
                "reason": "not bootstrapped",
                "run_name": "run-trevenant",
                "log": str(tmp_path / "run.log"),
                "initial_checkpoint": str(checkpoint),
                "initial_checkpoint_sha256": _sha256(checkpoint),
                "expert_manifest": str(expert),
                "expert_manifest_sha256": _sha256(expert),
                "expert_minimum_decisions": 10_000,
                "expert_required_target_coverage": [
                    "temporal_action_rows",
                    "opponent_remainder_rows",
                ],
                "matchup_runtime_tree": str(runtime_tree),
                "matchup_runtime_tree_sha256": _sha256(runtime_tree),
                "matchup_adapter_authorization": str(authorization),
                "matchup_adapter_authorization_sha256": _sha256(authorization),
                "matchup_adapter_epochs_per_rl_iteration": 1,
                "measurement_decks": "trevenant",
                "guide_loss_weight": 0.0,
                "terminal_gate_marker": "SPECIALIST_GATE_PASSED.trevenant-v1",
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ready_specialist_resolves_one_complete_command(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "trevenant"
    )
    command = _build_command(
        registry,
        "trevenant",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command.count("--specialist-archetype") == 1
    assert command[command.index("--specialist-archetype") + 1] == "trevenant"
    assert command[command.index("--minimum-terminal-iteration") + 1] == "5"
    assert command[command.index("--iterations") + 1] == "16"
    assert command[command.index("--terminal-active-gate-id") + 1] == (
        "gate-v1+frozen-specialists-r1"
    )
    assert command[command.index("--heldout-games") + 1] == "2250"
    assert command.count("--expert-min-decisions") == 1
    assert command[command.index("--expert-min-decisions") + 1] == "10000"
    assert command.count("--expert-required-target") == 2
    assert "--frozen-specialist-registry" in command


def test_future_specialist_uses_registry_floor_and_ceiling(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    registry["minimum_terminal_iteration"] = 5
    registry["iteration_ceiling"] = 15
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "trevenant"
    )
    command = _build_command(
        registry,
        "trevenant",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command[command.index("--minimum-terminal-iteration") + 1] == "5"
    assert command[command.index("--iterations") + 1] == "16"


def test_not_ready_specialist_fails_closed(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path, status="not_ready"))
    with pytest.raises(RuntimeError, match="not ready"):
        _resolve(registry, "trevenant")


def test_registered_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    registry_path = _fixture(tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["specialists"]["trevenant"]["initial_checkpoint_sha256"] = "0" * 64
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = _load_registry(registry_path)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _resolve(registry, "trevenant")


def test_gate_missing_frozen_predecessor_fails_closed(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    gate_path = Path(registry["runtime_root"]) / registry["active_gate_contract"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["next_gate"]["roster"].pop()
    gate["next_gate"]["evaluation"]["games_total"] = 2000
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "trevenant"
    )
    with pytest.raises(RuntimeError, match="S\\+ gate/registry"):
        _build_command(
            registry,
            "trevenant",
            row,
            checkpoint,
            expert,
            runtime_tree,
            authorization,
        )


def test_gate_total_scales_with_every_frozen_predecessor(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    runtime = Path(registry["runtime_root"])
    frozen_path = runtime / registry["frozen_specialist_registry"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    trevenant_digest = "sha256:" + "2" * 64
    frozen["specialists"].append(
        {
            "specialist_id": "hops-trevenant",
            "opponent_id": "specialist-trevenant",
            "checkpoint_digest": trevenant_digest,
            "frozen": True,
            "public_mix_eligible": True,
        }
    )
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    gate_path = runtime / registry["active_gate_contract"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["next_gate"]["id"] = "gate-v1+frozen-specialists-r2"
    gate["next_gate"]["evaluation"]["games_total"] = 2500
    gate["next_gate"]["roster"].append(
        {
            "opponent_id": "specialist-trevenant",
            "tier": "S+",
            "frozen_specialist": True,
            "frozen_checkpoint_digest": trevenant_digest,
        }
    )
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "trevenant"
    )
    command = _build_command(
        registry,
        "trevenant",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command[command.index("--terminal-active-gate-id") + 1] == (
        "gate-v1+frozen-specialists-r2"
    )
    assert command[command.index("--heldout-games") + 1] == "2500"
