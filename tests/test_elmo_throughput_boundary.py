from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_elmo_rotation_at_boundary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("elmo_boundary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_elmo_boundary_compose_command_uses_managed_production_files() -> None:
    module = _load_module()
    command = module._compose_command(
        "/mnt/Main/main/poke-bot-agent/deployments/runtime/worker",
        action="config --quiet",
    )
    assert "docker compose" in command
    assert "-f docker-compose.host.yml" in command
    assert "-f docker-compose.production.yml" in command
    assert command.endswith("config --quiet")


def test_elmo_boundary_unit_is_pinned_and_does_not_restart_local_trainer() -> None:
    unit = (
        ROOT
        / "deploy"
        / "systemd"
        / "pokebot-elmo-rev60-throughput-boundary.service"
    ).read_text(encoding="utf-8")
    assert "--target-next-iteration 15" in unit
    assert "--expected-rotation-jobs 768" in unit
    assert (
        "--expected-staged-sha256 "
        "74f280812aa350df6f320edfc118016ac386daae4e20f714e9250a814ad09a6a"
        in unit
    )
    assert "pokebot-pure-rl-trevenant-staged.service" not in unit
    assert "systemctl" not in unit


def test_elmo_boundary_waits_before_acquiring_shared_maintenance_lock() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    wait = source.index(
        "while _next_iteration(args.loop_state) < args.target_next_iteration:"
    )
    acquire = source.index("fcntl.flock(lock.fileno(), fcntl.LOCK_EX)")
    assert wait < acquire
