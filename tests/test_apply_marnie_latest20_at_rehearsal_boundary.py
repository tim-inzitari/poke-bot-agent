from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.apply_marnie_latest20_at_rehearsal_boundary as activation

from scripts.apply_marnie_latest20_at_rehearsal_boundary import (
    EARLY_POOL_RELEASE_ENV,
    EARLY_POOL_RELEASE_IMPLEMENTATION,
    committed_boundary,
    render_dropins,
    sha256,
    validate_early_pool_release_implementation,
    validate_stage,
    wait_service_pid,
)


def test_activation_restarts_long_lived_gate_watcher_on_new_registry() -> None:
    source = Path(
        "scripts/apply_marnie_latest20_at_rehearsal_boundary.py"
    ).read_text(encoding="utf-8")
    normal_activation = source.index(
        'old_pid = int(service_value(args.rl_unit, "MainPID") or 0)'
    )
    stop_gate = source.index(
        'run(["systemctl", "--user", "stop", args.gate_unit]',
        normal_activation,
    )
    stop_trainer = source.index(
        'run(["systemctl", "--user", "stop", args.rl_unit]',
        normal_activation,
    )
    start_trainer = source.index(
        'run(["systemctl", "--user", "start", args.rl_unit]',
        normal_activation,
    )
    start_gate = source.index(
        '["systemctl", "--user", "start", "--no-block", args.gate_unit]',
        normal_activation,
    )

    assert stop_gate < stop_trainer < start_trainer < start_gate
    assert '"old_gate_pid": old_gate_pid' in source
    assert '"new_gate_pid": new_gate_pid' in source
    assert '{"active", "activating"}' in source
    assert source.count("validate_stage(args.stage_receipt)") == 2
    assert source.count(
        "validate_early_pool_release_implementation("
    ) >= 3
    assert "implementation changed while waiting" in source
    assert "EARLY_POOL_RELEASE_ENV not in process_environment(new_pid)" in source
    assert '"active_process_environment_verified": True' in source
    assert "new_gate_pid = wait_service_pid(" in source
    assert '"trainer_restarted_during_recovery": False' in source
    assert '"recovered_after_gate_pid_observation_race": True' in source


def test_wait_service_pid_does_not_accept_activating_with_pid_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pids = iter(("0", "1234", "1234"))

    def fake_service_value(_unit: str, key: str) -> str:
        if key == "ActiveState":
            return "activating"
        return next(pids)

    monkeypatch.setattr(activation, "service_value", fake_service_value)
    monkeypatch.setattr(activation.time, "sleep", lambda _seconds: None)

    assert wait_service_pid(
        "example.service", 999, 5, stable_seconds=0
    ) == 1234


OLD_REGISTRY = "/state/old-registry.json"
NEW_REGISTRY = "/state/new-registry.json"
OLD_REGISTRATION = "/state/old-registration.json"
NEW_REGISTRATION = "/state/new-registration.json"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_render_dropins_changes_every_runtime_consumer() -> None:
    python = "/venv/bin/python"
    rl = (
        "[Service]\n"
        f"ExecStart={python} -u scripts/launch_active_specialist.py "
        f"--registry {OLD_REGISTRY}\n"
    )
    gate = (
        "[Service]\n"
        f"ExecStartPre={python} -u scripts/launch_active_specialist_gate_handler.py "
        f"--registry {OLD_REGISTRY} --check\n"
        f"ExecStart={python} -u scripts/launch_active_specialist_gate_handler.py "
        f"--registry {OLD_REGISTRY}\n"
    )
    completion = (
        "[Service]\n"
        f"ExecStartPre={python} -u scripts/complete_final_format_marnie_refresh.py "
        f"--check --runtime-registration {OLD_REGISTRATION}\n"
        f"ExecStart={python} -u scripts/complete_final_format_marnie_refresh.py "
        f"--runtime-registration {OLD_REGISTRATION}\n"
    )

    rl_dropin, gate_dropin, completion_dropin, *checks = render_dropins(
        rl_unit_text=rl,
        gate_unit_text=gate,
        completion_unit_text=completion,
        old_registry=OLD_REGISTRY,
        candidate_registry=NEW_REGISTRY,
        old_registration=OLD_REGISTRATION,
        candidate_registration=NEW_REGISTRATION,
    )

    assert OLD_REGISTRY not in rl_dropin + gate_dropin
    assert NEW_REGISTRY in rl_dropin and NEW_REGISTRY in gate_dropin
    assert OLD_REGISTRATION not in completion_dropin
    assert NEW_REGISTRATION in rl_dropin and NEW_REGISTRATION in completion_dropin
    assert all("--check" in command for command in checks)
    assert "ExecStart=\n" in rl_dropin
    assert f"Environment={EARLY_POOL_RELEASE_ENV}\n" in rl_dropin
    assert "ExecStartPre=\n" in gate_dropin
    assert "ExecStartPre=\n" in completion_dropin


def test_validate_stage_and_exact_commit(tmp_path: Path) -> None:
    sync = tmp_path / "sync.json"
    _write(
        sync,
        {
            "schema": "poke_bot.latest20_specialist_sync/v1",
            "status": "ready",
        },
    )
    pointer = tmp_path / "corpus" / "PROTECTED_EXPERT_CORPUS.json"
    _write(pointer, {"schema": "poke_bot.pinned_expert_corpus/v1"})
    registry = tmp_path / "candidate-registry.json"
    _write(
        registry,
        {
            "specialists": {
                "marnie-s-grimmsnarl-ex": {
                    "expert_manifest": str(pointer),
                    "expert_manifest_sha256": sha256(pointer).removeprefix(
                        "sha256:"
                    ),
                }
            }
        },
    )
    registration = tmp_path / "candidate-registration.json"
    _write(
        registration,
        {
            "runtime_registry": str(registry),
            "runtime_registry_sha256": sha256(registry),
        },
    )
    stage = tmp_path / "stage.json"
    _write(
        stage,
        {
            "schema": "poke_bot.marnie_latest20_runtime_migration_stage/v1",
            "status": "ready_for_clean_rehearsal_boundary",
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "window_start": "2026-07-14",
            "window_end": "2026-08-02",
            "active_registry_modified": False,
            "training_interrupted": False,
            "sync_receipt": str(sync),
            "sync_receipt_sha256": sha256(sync),
            "candidate_registry": str(registry),
            "candidate_registry_sha256": sha256(registry),
            "candidate_registration": str(registration),
            "candidate_registration_sha256": sha256(registration),
            "candidate_expert_pointer": str(pointer),
            "candidate_expert_pointer_sha256": sha256(pointer),
        },
    )
    assert validate_stage(stage)["window_end"] == "2026-08-02"

    _write(
        sync,
        {
            "schema": "poke_bot.latest20_specialist_sync/v1",
            "status": "ready",
            "changed": True,
        },
    )
    with pytest.raises(RuntimeError, match="migration stage is invalid"):
        validate_stage(stage)
    _write(
        sync,
        {
            "schema": "poke_bot.latest20_specialist_sync/v1",
            "status": "ready",
        },
    )

    checkpoint = tmp_path / "iter_00004.pt"
    checkpoint.write_bytes(b"checkpoint")
    commit = tmp_path / "iter_00004.json"
    _write(
        commit,
        {
            "last_completed_iteration": 4,
            "next_iteration": 5,
            "learner": {
                "path": str(checkpoint),
                "digest": sha256(checkpoint),
            },
        },
    )
    assert committed_boundary(commit, 4) is not None


def test_early_pool_release_implementation_is_checksum_bound(
    tmp_path: Path,
) -> None:
    for relative, markers in EARLY_POOL_RELEASE_IMPLEMENTATION.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(markers) + "\n", encoding="utf-8")

    evidence = validate_early_pool_release_implementation(tmp_path)

    assert evidence["deployment_root"] == str(tmp_path.resolve())
    assert evidence["enabled_environment"] == EARLY_POOL_RELEASE_ENV
    assert set(evidence["modules"]) == set(EARLY_POOL_RELEASE_IMPLEMENTATION)
    for module in evidence["modules"].values():
        assert module["sha256"] == sha256(Path(module["path"]))

    trainer = tmp_path / "scripts/train_pure_rl.py"
    trainer.write_text("feature removed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="implementation is incomplete"):
        validate_early_pool_release_implementation(tmp_path)


def test_managed_activation_unit_targets_only_rehearsal_boundary() -> None:
    unit = Path(
        "deploy/systemd/pokebot-marnie-latest20-runtime-activate-r109.service"
    ).read_text()

    assert "apply_marnie_latest20_at_rehearsal_boundary.py" in unit
    assert "--completed-iteration 4" in unit
    assert (
        "--deployment-root /home/inzi/poke-bot-agent-deployments/"
        "final-format-marnie-h10-r104"
        in unit
    )
    assert "--poll-seconds 0.1" in unit
    assert (
        "EnvironmentFile=/home/inzi/poke-bot-agent/outputs/"
        "final_format_marnie_r104/runtime/specialist_runtime_h10_r104.env"
        in unit
    )
    assert (
        "EnvironmentFile=/home/inzi/poke-bot-agent-deployments/"
        "final-format-marnie-h10-r104/config/"
        "final_format_marnie_h10_runtime.env"
        in unit
    )
    assert (
        "Environment=PYTHONPATH=/home/inzi/poke-bot-agent-deployments/"
        "final-format-marnie-h10-r104:/home/inzi/poke-bot-agent"
        in unit
    )
    assert "receipt_backed_latest20_rehearsal_boundary_r109" in unit
    assert "kill" not in unit.lower()
