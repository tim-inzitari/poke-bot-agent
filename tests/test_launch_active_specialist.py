from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
    SLOT_CAPACITY,
    load_slot_registry,
)
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
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
    routes = list(EXPERT_IDS)
    torch.save(
        {
            "model_state_dict": {
                f"matchup_adapter_bank.experts.{index}.up.weight": torch.zeros(1)
                for index in range(len(EXPERT_IDS))
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
                    "accepted_archetype_ids": ["dragapult-dusknoir"],
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
            "dragapult-dusknoir": {
                "status": status,
                "reason": "not bootstrapped",
                "run_name": "run-dragapult-dusknoir",
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
                "measurement_decks": "dragapult-dusknoir",
                "guide_loss_weight": 0.0,
                "terminal_gate_marker": "SPECIALIST_GATE_PASSED.dragapult-dusknoir-v1",
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ready_specialist_resolves_one_complete_command(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )
    assert command.count("--specialist-archetype") == 1
    assert command[command.index("--specialist-archetype") + 1] == "dragapult-dusknoir"
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


def test_successor_runtime_fails_closed_without_exact_17_head_fusion(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["specialists"]["dragapult-dusknoir"]
    row["decision_fusion"] = {
        "schema": DECISION_FUSION_SCHEMA,
        "required": True,
        "runtime_enabled": True,
        "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mandatory 17-head"):
        _resolve(_load_registry(path), "dragapult-dusknoir")

    checkpoint = Path(row["initial_checkpoint"])
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_payload["model_config"] = {
        "expanded_heads_enabled": True,
        "decision_fusion_enabled": True,
        "decision_fusion_runtime_enabled": True,
    }
    checkpoint_payload["provenance"] = {
        "decision_fusion": {
            "schema": DECISION_FUSION_SCHEMA,
            "enabled": True,
            "runtime_enabled": True,
            "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
        }
    }
    checkpoint_payload["model_state_dict"][
        "decision_fusion.residual.2.weight"
    ] = torch.ones(1)
    torch.save(checkpoint_payload, checkpoint)
    row["initial_checkpoint_sha256"] = _sha256(checkpoint)
    path.write_text(json.dumps(payload), encoding="utf-8")

    _resolve(_load_registry(path), "dragapult-dusknoir")


def test_v6_checkpoint_must_bind_exact_runtime_registry(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["specialists"]["dragapult-dusknoir"]
    checkpoint = Path(row["initial_checkpoint"])
    slot_registry = load_slot_registry()
    torch.save(
        {
            "model_state_dict": {
                f"matchup_adapter_bank.experts.{index}.up.weight": torch.zeros(1)
                for index in range(SLOT_CAPACITY)
            },
            "extra": {
                "matchup_adapter_config": {
                    "format": V6_ADAPTER_CHECKPOINT_FORMAT,
                    "slot_capacity": SLOT_CAPACITY,
                    "slot_registry": slot_registry,
                }
            },
        },
        checkpoint,
    )
    row["initial_checkpoint_sha256"] = _sha256(checkpoint)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _resolve(_load_registry(path), "dragapult-dusknoir")

    mismatched = json.loads(json.dumps(slot_registry))
    mismatched["revision"] = int(mismatched["revision"]) + 1
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_payload["extra"]["matchup_adapter_config"][
        "slot_registry"
    ] = mismatched
    torch.save(checkpoint_payload, checkpoint)
    payload["specialists"]["dragapult-dusknoir"][
        "initial_checkpoint_sha256"
    ] = _sha256(checkpoint)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact runtime registry"):
        _resolve(_load_registry(path), "dragapult-dusknoir")


def test_existing_run_uses_validated_loop_state_not_initial_seed(
    tmp_path: Path,
) -> None:
    registry = _load_registry(_fixture(tmp_path))
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    loop_state = (
        Path(registry["runtime_root"])
        / "outputs"
        / "pure_rl"
        / row["run_name"]
        / "loop_state.json"
    )
    loop_state.parent.mkdir(parents=True)
    loop_state.write_text(
        json.dumps(
            {
                "next_iteration": 0,
                "learner": {
                    "path": str(checkpoint),
                    "digest": f"sha256:{_sha256(checkpoint)}",
                },
            }
        ),
        encoding="utf-8",
    )

    command = _build_command(
        registry,
        "dragapult-dusknoir",
        row,
        checkpoint,
        expert,
        runtime_tree,
        authorization,
    )

    assert "--initial-learner-checkpoint" not in command
    assert command[command.index("--resume") + 1] == "auto"


def test_future_specialist_uses_registry_floor_and_ceiling(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    registry["minimum_terminal_iteration"] = 5
    registry["iteration_ceiling"] = 15
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
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
        _resolve(registry, "dragapult-dusknoir")


def test_registered_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    registry_path = _fixture(tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["specialists"]["dragapult-dusknoir"]["initial_checkpoint_sha256"] = "0" * 64
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = _load_registry(registry_path)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _resolve(registry, "dragapult-dusknoir")


def test_gate_missing_frozen_predecessor_fails_closed(tmp_path: Path) -> None:
    registry = _load_registry(_fixture(tmp_path))
    gate_path = Path(registry["runtime_root"]) / registry["active_gate_contract"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["next_gate"]["roster"].pop()
    gate["next_gate"]["evaluation"]["games_total"] = 2000
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    row, checkpoint, expert, runtime_tree, authorization = _resolve(
        registry, "dragapult-dusknoir"
    )
    with pytest.raises(RuntimeError, match="S\\+ gate/registry"):
        _build_command(
            registry,
            "dragapult-dusknoir",
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
        registry, "dragapult-dusknoir"
    )
    command = _build_command(
        registry,
        "dragapult-dusknoir",
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
