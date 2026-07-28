from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.launch_active_specialist_gate_handler import build_command


def _write_submission_contract_inputs(runtime: Path) -> None:
    for relative in (
        "scripts/build_submission.sh",
        "scripts/build_submission_belief_posterior.py",
        "submission/main.py",
        "submission/search_config.json",
        "poke_bot/submission_budget.py",
        "data/training_mixes/top_ladder_representatives.v1.json",
        "data/training_mixes/specialist_representatives.v1.json",
    ):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def _write_representatives(runtime: Path) -> None:
    example_cards = [1] * 60
    starmie_cards = [2] * 60

    def digest(cards: list[int]) -> str:
        return "sha256:" + hashlib.sha256(
            ",".join(str(card_id) for card_id in cards).encode("ascii")
        ).hexdigest()

    payload = {
        "schema": "poke_bot.specialist_deck_representatives/v1",
        "decks": {
            "example": {
                "card_ids": example_cards,
                "canonical_multiset_sha256": digest(example_cards),
            },
            "starmie": {
                "card_ids": starmie_cards,
                "canonical_multiset_sha256": digest(starmie_cards),
            },
        },
    }
    payload["artifact_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    (runtime / "data/training_mixes/representatives.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _classify_fixture_representatives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.launch_active_specialist_gate_handler.classify_deck",
        lambda cards: "example" if cards and cards[0] == 1 else "starmie",
    )


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
    _write_submission_contract_inputs(runtime)
    _write_representatives(runtime)
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
                    "runtime_exact_gate_receipt": "/state/runtime-gate.json",
                },
            }
        },
    }

    command = build_command(registry, "example")

    assert command[command.index("--archetype") + 1] == "example"
    assert command[command.index("--minimum-completed-iteration") + 1] == "5"
    assert command[command.index("--training-service") + 1] == "trainer.service"
    assert command[command.index("--handoff-service") + 1] == "next.service"
    assert "--require-decision-fusion-runtime" in command
    assert command[command.index("--runtime-exact-gate-receipt") + 1] == (
        "/state/runtime-gate.json"
    )
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
    _write_submission_contract_inputs(runtime)
    _write_representatives(runtime)
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


def test_gate_handler_preflight_requires_selected_exact_representative(
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
    _write_submission_contract_inputs(runtime)
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

    with pytest.raises(RuntimeError, match="exact 60-card"):
        build_command(registry, "example")


def test_gate_handler_fails_before_training_when_submission_assets_are_missing(
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
    _write_representatives(runtime)
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

    try:
        build_command(registry, "example")
    except RuntimeError as exc:
        assert "pass-handler submission input is missing" in str(exc)
    else:
        raise AssertionError("missing submission assets must fail closed")
