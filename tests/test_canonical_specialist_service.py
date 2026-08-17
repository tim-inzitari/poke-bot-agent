from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse_env(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        assert key not in result
        result[key] = value
    return result


def _env() -> dict[str, str]:
    source = ROOT / "config/specialist_runtime.env"
    return _parse_env(source.read_text(encoding="utf-8"))


def test_registry_is_the_only_owner_of_training_arguments() -> None:
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    env = _env()
    unit = (
        ROOT / "ops/systemd/pokebot-pure-rl-specialist.service"
    ).read_text(encoding="utf-8")

    assert registry["minimum_terminal_iteration"] == 5
    assert registry["iteration_ceiling"] == 15
    assert "minimum_terminal_iteration" not in registry["specialists"]["starmie"]
    args = registry["common_trainer_args"]
    assert "--iterations" not in args
    assert "iteration_ceiling" not in registry["specialists"]["starmie"]
    assert "PURE_RL_MINIMUM_TERMINAL_ITERATION" not in env
    assert "PURE_RL_ITERATION_CEILING" not in env
    assert "--minimum-terminal-iteration" not in unit
    assert "--iterations" not in unit


def test_service_has_one_environment_file_and_one_selector_launcher() -> None:
    dummy_env = _parse_env(
        "POKEBOT_ACTIVE_SPECIALIST=dummy-specialist\n"
        "POKEBOT_SPECIALIST_RUNTIME_ROOT=/tmp/runtime\n"
    )
    unit = (
        ROOT / "ops/systemd/pokebot-pure-rl-specialist.service"
    ).read_text(encoding="utf-8")

    assert dummy_env["POKEBOT_ACTIVE_SPECIALIST"] == "dummy-specialist"
    assert unit.count("EnvironmentFile=") == 1
    assert unit.count("\nExecStart=") == 1
    assert unit.count("\nExecStartPre=") == 2
    assert "launch_active_specialist.py" in unit
    assert "launch_active_specialist_gate_handler.py --check" in unit
    assert (
        "OnSuccess=pokebot-specialist-passed-gate-handler.service"
        in unit
    )
    assert ".service.d" not in unit


def test_live_selector_references_a_registry_record() -> None:
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    selected = _env()["POKEBOT_ACTIVE_SPECIALIST"]
    assert selected in registry["specialists"]
    assert _env()["POKEBOT_EXPANDED_HEADS_ENABLED"] == "1"
