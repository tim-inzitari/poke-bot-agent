from __future__ import annotations

import json
from pathlib import Path

from scripts.launch_active_specialist_gate_handler import build_command


def test_gate_handler_resolves_from_selected_registry_record(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "ops").mkdir(parents=True)
    (runtime / "scripts").mkdir()
    (runtime / "data/training_mixes").mkdir(parents=True)
    for relative in (
        "scripts/handle.py",
        "ops/gate.json",
        "data/training_mixes/representatives.json",
    ):
        (runtime / relative).write_text("{}", encoding="utf-8")
    tree = tmp_path / "tree.json"
    tree.write_text("{}", encoding="utf-8")
    registry = {
        "runtime_root": str(runtime),
        "python": "/usr/bin/python3",
        "active_gate_contract": "ops/gate.json",
        "minimum_terminal_iteration": 5,
        "pass_handler": {
            "launcher": "scripts/handle.py",
            "registry_root": "/models",
            "representatives": "data/training_mixes/representatives.json",
            "competition": "competition",
            "submission_count": 1,
            "submission_mode": "queue_and_continue",
            "submission_queue": "/state/queue.json",
            "kaggle": "/bin/kaggle",
            "authorization": "/config/authorization.json",
            "submission_receipts": "/state/receipts",
            "training_service": "trainer.service",
            "continue_drop_in_source": "ops/gate.json",
            "continue_drop_in_target": "/config/unused.conf",
            "poll_seconds": 15,
            "upload_timeout_seconds": 900,
        },
        "specialists": {
            "example": {
                "status": "ready",
                "run_name": "run-example",
                "terminal_gate_marker": "SPECIALIST_GATE_PASSED.example",
                "matchup_runtime_tree": str(tree),
                "pass_handler": {
                    "family": "example-v1",
                    "display_name": "Example",
                    "submission_root": "/submissions/example",
                    "state": "/state/example.json",
                    "lock": "/state/example.lock",
                    "handoff_service": "next.service",
                },
            }
        },
    }

    command = build_command(registry, "example")

    assert command[command.index("--archetype") + 1] == "example"
    assert command[command.index("--minimum-completed-iteration") + 1] == "5"
    assert command[command.index("--training-service") + 1] == "trainer.service"
    assert command[command.index("--handoff-service") + 1] == "next.service"
    assert command.count("--run-dir") == 1


def test_gate_handler_uses_canonical_floor_without_specialist_override(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "ops").mkdir(parents=True)
    (runtime / "scripts").mkdir()
    (runtime / "data/training_mixes").mkdir(parents=True)
    for relative in (
        "scripts/handle.py",
        "ops/gate.json",
        "data/training_mixes/representatives.json",
    ):
        (runtime / relative).write_text("{}", encoding="utf-8")
    tree = tmp_path / "tree.json"
    tree.write_text("{}", encoding="utf-8")
    registry = {
        "runtime_root": str(runtime),
        "python": "/usr/bin/python3",
        "active_gate_contract": "ops/gate.json",
        "minimum_terminal_iteration": 5,
        "pass_handler": {
            "launcher": "scripts/handle.py",
            "registry_root": "/models",
            "representatives": "data/training_mixes/representatives.json",
            "competition": "competition",
            "submission_count": 1,
            "submission_mode": "queue_and_continue",
            "submission_queue": "/state/queue.json",
            "kaggle": "/bin/kaggle",
            "authorization": "/config/authorization.json",
            "submission_receipts": "/state/receipts",
            "training_service": "trainer.service",
            "continue_drop_in_source": "ops/gate.json",
            "continue_drop_in_target": "/config/unused.conf",
        },
        "specialists": {
            "starmie": {
                "status": "ready",
                "run_name": "run-starmie",
                "terminal_gate_marker": "SPECIALIST_GATE_PASSED.starmie",
                "matchup_runtime_tree": str(tree),
                "pass_handler": {
                    "family": "starmie-v1",
                    "display_name": "Starmie",
                    "submission_root": "/submissions/starmie",
                    "state": "/state/starmie.json",
                    "lock": "/state/starmie.lock",
                    "handoff_service": "next.service",
                },
            }
        },
    }
    command = build_command(registry, "starmie")
    assert command[command.index("--minimum-completed-iteration") + 1] == "5"
