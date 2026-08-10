from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import stage_alakazam_marnie_splusplus_r192 as stage


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_code_tree(candidate: Path) -> None:
    for relative in (
        "scripts/launch_pure_rl.py",
        "scripts/train_pure_rl.py",
        "scripts/launch_active_specialist.py",
        "scripts/activate_alakazam_marnie_splusplus_r192.py",
        "poke_bot/pure_rl/strong_public_gate.py",
        "poke_bot/public_multi_env_safety.py",
        "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-rl.service.d/62-marnie-splusplus-r192.conf.in",
        "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-marnie-splusplus-r192-boundary.service.in",
        "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-rl.service.d/61-marnie-splusplus-r192-stop-budget.conf.in",
    ):
        contents = "# candidate r192 deployment input\n"
        if relative.endswith("61-marnie-splusplus-r192-stop-budget.conf.in"):
            contents = (
                "[Service]\n"
                "TimeoutStopSec=8s\n"
                "KillMode=control-group\n"
                "SendSIGKILL=yes\n"
            )
        _write(candidate / relative, contents)
    _write(candidate / "baselines/manifest.json", {"agents": []})
    _write(
        candidate
        / "baselines/specialists/marnie-final-format-h10-f20efb20f5c3/model.pt",
        "checksum-bound model proof fixture\n",
    )


def _stage_args(source_registry: Path, candidate: Path, source_pin: Path) -> list[str]:
    return [
        "--source-runtime-registry",
        str(source_registry),
        "--runtime-root",
        str(candidate),
        "--source-pin-sidecar",
        str(source_pin),
    ]


@pytest.fixture
def r175_parent_and_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "live-r175"
    candidate = tmp_path / "candidate-r192"
    source_root.mkdir()
    candidate.mkdir()
    _candidate_code_tree(candidate)

    gate = _json(ROOT / "ops/final_format_alakazam_gate_r100_v1.json")
    frozen = _json(ROOT / "ops/frozen_specialist_registry_v1.json")
    _write(source_root / "ops/active-gate.json", gate)
    _write(source_root / "ops/frozen.json", frozen)
    source_pin = source_root / "runtime/owner-pins.json"
    crustle_pin = {
        "package_id": stage.CRUSTLE_H10_OPPONENT_ID,
        "checkpoint_sha256": stage.CRUSTLE_H10_CHECKPOINT_DIGEST,
        "content_digest": stage.CRUSTLE_H10_CONTENT_DIGEST,
        "floor_games_per_set": stage.CRUSTLE_H10_FLOOR_GAMES,
        "archetype_id": "crustle",
        "opaque_retained_field": {"must": "survive"},
    }
    _write(
        source_pin,
        {
            "schema": stage.PIN_FLOORS_SCHEMA,
            "pins": [
                {
                    "package_id": stage.H10_MARNIE_OPPONENT_ID,
                    "checkpoint_sha256": stage.H10_MARNIE_CHECKPOINT_DIGEST,
                    "content_digest": stage.H10_MARNIE_CONTENT_DIGEST,
                    "floor_games_per_set": 1024,
                    "archetype_id": stage.H10_MARNIE_ARCHETYPE_ID,
                    "tier": "S++",
                    "weight": 4.0,
                },
                crustle_pin,
            ],
        },
    )
    source_registry = source_root / "runtime/specialist-runtime-r175.json"
    _write(
        source_registry,
        {
            "schema": stage.RUNTIME_REGISTRY_SCHEMA,
            "owner_decision_revision": 175,
            "runtime_root": str(source_root.resolve()),
            "active_gate_contract": "ops/active-gate.json",
            "frozen_specialist_registry": "ops/frozen.json",
            "minimum_terminal_iteration": 5,
            "common_trainer_args": [
                "--gate-boundary-pause-seconds",
                "30",
                "--games-per-iter",
                "8196",
                "--no-allow-clean-boundary-design-migration",
                "--boundary-design-migration-reason",
                "stale-parent-reason",
            ],
            "isolated_refresh_contract": {
                "grimmsnarl_package_id": stage.H10_MARNIE_OPPONENT_ID,
                "grimmsnarl_checkpoint_sha256": stage.H10_MARNIE_CHECKPOINT_DIGEST,
                "grimmsnarl_floor_per_set": 1024,
            },
            "specialists": {
                "alakazam": {
                    "minimum_terminal_iteration": 5,
                    "owner_grimmsnarl_pin": {
                        "package_id": stage.H10_MARNIE_OPPONENT_ID,
                        "checkpoint_sha256": stage.H10_MARNIE_CHECKPOINT_DIGEST,
                        "floor_games_per_set": 1024,
                    },
                }
            },
        },
    )
    return source_registry, candidate, source_pin


def test_full_stage_is_candidate_root_bound_and_idempotent(
    r175_parent_and_candidate: tuple[Path, Path, Path],
) -> None:
    source_registry, candidate, source_pin = r175_parent_and_candidate
    source_before = _json(source_registry.parent.parent / "ops/active-gate.json")

    assert stage.main(_stage_args(source_registry, candidate, source_pin)) == 0

    gate_path = candidate / "runtime/final_format_alakazam_gate_r192_marnie_splusplus.json"
    frozen_path = candidate / "ops/frozen_specialist_registry_alakazam_r192_marnie_splusplus.json"
    runtime_path = candidate / "runtime/specialist_runtime_registry_h10_r175_marnie_splusplus_r192.json"
    pin_path = candidate / "runtime/owner_public_mix_pin_floors_r192.json"
    receipt_path = candidate / "runtime/alakazam-marnie-splusplus-r192-stage.json"
    gate = _json(gate_path)
    frozen = _json(frozen_path)
    runtime = _json(runtime_path)
    pin_sidecar = _json(pin_path)
    receipt = _json(receipt_path)

    # The materializer only derives checksum-bound artifacts.  It cannot arm
    # a restart, alter the selector/service, or interrupt the r175 parent.
    assert gate["r192_stage"]["status"] == "staged_non_active"
    assert (
        gate["r192_stage"][
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        ]
        is False
    )
    assert gate["r192_stage"]["automatic_managed_restart_armed"] is False
    assert gate["r192_stage"]["trainer_owned_handoff_fence_required"] is True
    assert (
        gate["r192_stage"]["current_r175_source_has_trainer_owned_handoff_fence"]
        is False
    )
    assert len(gate["next_gate"]["roster"]) == 18
    assert gate["next_gate"]["evaluation"]["games_total"] == 4500
    assert gate["active_gate_semantics"]["frozen_specialist_agents"] == 15
    assert gate["active_gate_semantics"]["exact_additional_splusplus_specialist"] == {
        "opponent_id": stage.H10_MARNIE_OPPONENT_ID,
        "checkpoint_digest": stage.H10_MARNIE_CHECKPOINT_DIGEST,
        "content_digest": stage.H10_MARNIE_CONTENT_DIGEST,
        "tier": "S++",
        "weight": 4.0,
        "strong_public_practice_floor_games": 1024,
    }
    assert len(frozen["specialists"]) == 15
    assert all(
        row.get("package_id") != stage.H10_MARNIE_OPPONENT_ID
        for row in pin_sidecar["pins"]
    )
    assert pin_sidecar["pins"][0]["opaque_retained_field"] == {"must": "survive"}

    assert runtime["runtime_root"] == str(candidate.resolve())
    assert runtime["r192_stage"]["status"] == "staged_not_activated"
    assert (
        runtime["r192_stage"][
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        ]
        is False
    )
    assert runtime["r192_stage"]["automatic_managed_restart_armed"] is False
    assert runtime["r192_stage"]["trainer_owned_handoff_fence_required"] is True
    assert (
        runtime["r192_stage"]["current_r175_source_has_trainer_owned_handoff_fence"]
        is False
    )
    assert runtime["active_gate_contract"] == "runtime/final_format_alakazam_gate_r192_marnie_splusplus.json"
    assert runtime["frozen_specialist_registry"] == "ops/frozen_specialist_registry_alakazam_r192_marnie_splusplus.json"
    args = runtime["common_trainer_args"]
    assert args.count("--official-adaptive-min-share") == 1
    assert args[args.index("--official-adaptive-min-share") + 1] == "0.04"
    assert args.count("--allow-clean-boundary-design-migration") == 1
    assert "--no-allow-clean-boundary-design-migration" not in args
    assert args.count("--boundary-design-migration-reason") == 1
    assert (
        args[args.index("--boundary-design-migration-reason") + 1]
        == stage.R192_BOUNDARY_DESIGN_MIGRATION_REASON
    )
    owner_pin = runtime["specialists"]["alakazam"]["owner_grimmsnarl_pin"]
    assert owner_pin["enforcement_source"] == "exact_active_gate_strong_public_practice_floor"
    assert owner_pin["legacy_diverse_public_sidecar_pin_active"] is False
    assert owner_pin["superseded_by_exact_strong_gate_floor"] is True
    bindings = runtime["r192_stage"]["artifact_bindings"]
    assert set(
        ("owner_contract", "active_gate_contract", "frozen_specialist_registry", "public_mix_pin_floors")
    ).issubset(bindings)
    assert bindings["active_gate_contract"]["sha256"] == stage._file_digest(gate_path)
    assert bindings["frozen_specialist_registry"]["sha256"] == stage._file_digest(frozen_path)
    assert bindings["public_mix_pin_floors"]["sha256"] == stage._file_digest(pin_path)
    assert bindings["gate_sha256"] == stage._file_digest(gate_path)
    assert bindings["frozen_registry_sha256"] == stage._file_digest(frozen_path)

    assert receipt["source_artifacts"]["active_gate_contract"] == {
        "path": str((source_registry.parent.parent / "ops/active-gate.json").resolve()),
        "sha256": stage._file_digest(source_registry.parent.parent / "ops/active-gate.json"),
    }
    assert receipt["staged_artifacts"]["runtime_registry"]["path"] == str(
        runtime_path.resolve()
    )
    assert receipt["status"] == "staged_non_active"
    assert receipt["active_before_receipt_backed_activation"] is False
    assert receipt["training_interrupted"] is False
    assert receipt["selector_or_service_changed"] is False
    assert receipt["remote_deployment_performed"] is False
    assert (
        receipt[
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        ]
        is False
    )
    assert receipt["automatic_managed_restart_armed"] is False
    assert receipt["trainer_owned_handoff_fence_required"] is True
    assert receipt["current_r175_source_has_trainer_owned_handoff_fence"] is False
    assert receipt["boundary_design_migration"] == {
        "allow_clean_boundary_design_migration": True,
        "reason": stage.R192_BOUNDARY_DESIGN_MIGRATION_REASON,
    }
    assert receipt["stop_budget_guard"] == stage.STOP_BUDGET_GUARD
    assert set(receipt["activation_artifacts"]) == {
        "controller",
        "dropin_template",
        "boundary_service_template",
        "stop_budget_template",
    }
    assert (
        receipt["activation_artifacts"]["controller"]["path"]
        == str((candidate / "scripts/activate_alakazam_marnie_splusplus_r192.py").resolve())
    )
    labels = {row["label"] for row in receipt["deployment_inputs"]}
    assert stage.REQUIRED_DEPLOYMENT_INPUT_LABELS.issubset(labels)
    assert source_before == _json(source_registry.parent.parent / "ops/active-gate.json")

    assert stage.main(_stage_args(source_registry, candidate, source_pin) + ["--check"]) == 0


def test_runtime_migration_authorization_replaces_stale_values() -> None:
    source = {
        "schema": stage.RUNTIME_REGISTRY_SCHEMA,
        "owner_decision_revision": stage.PARENT_OWNER_REVISION,
        "minimum_terminal_iteration": 5,
        "common_trainer_args": [
            "--allow-clean-boundary-design-migration",
            "--allow-clean-boundary-design-migration",
            "--boundary-design-migration-reason=generic",
            "--boundary-design-migration-reason",
            "another-generic-reason",
        ],
        "isolated_refresh_contract": {"parent": "r175"},
        "specialists": {
            "alakazam": {
                "minimum_terminal_iteration": 5,
                "owner_grimmsnarl_pin": {"package_id": stage.H10_MARNIE_OPPONENT_ID},
            }
        },
    }
    runtime = stage.build_runtime_registry(
        source,
        gate_reference="runtime/gate.json",
        frozen_reference="ops/frozen.json",
        gate_id="r192-gate",
        gate_sha256="sha256:" + "1" * 64,
        frozen_sha256="sha256:" + "2" * 64,
    )
    args = runtime["common_trainer_args"]
    assert args.count("--allow-clean-boundary-design-migration") == 1
    assert args.count("--boundary-design-migration-reason") == 1
    assert args[args.index("--boundary-design-migration-reason") + 1] == (
        stage.R192_BOUNDARY_DESIGN_MIGRATION_REASON
    )


def test_stop_budget_template_requires_exact_bounded_service_settings(
    tmp_path: Path,
) -> None:
    template = tmp_path / "61-marnie-splusplus-r192-stop-budget.conf.in"
    _write(
        template,
        "[Service]\nTimeoutStopSec=8s\nKillMode=control-group\nSendSIGKILL=yes\n",
    )
    stage._validate_stop_budget_template(template)

    _write(
        template,
        "[Service]\nTimeoutStopSec=30s\nKillMode=control-group\nSendSIGKILL=yes\n",
    )
    with pytest.raises(RuntimeError, match="8s/control-group/SIGKILL"):
        stage._validate_stop_budget_template(template)


def test_default_receipt_avoids_outputs_symlink_and_explicit_escape_is_rejected(
    r175_parent_and_candidate: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source_registry, candidate, source_pin = r175_parent_and_candidate
    outside = tmp_path / "outside-outputs"
    outside.mkdir()
    (candidate / "outputs").symlink_to(outside, target_is_directory=True)

    assert stage.main(_stage_args(source_registry, candidate, source_pin)) == 0
    assert (candidate / "runtime/alakazam-marnie-splusplus-r192-stage.json").is_file()
    assert not (outside / "state/alakazam-marnie-splusplus-r192-stage.json").exists()

    with pytest.raises(RuntimeError, match="symbolic link"):
        stage.main(
            _stage_args(source_registry, candidate, source_pin)
            + ["--out-stage-receipt", "outputs/state/escaped-receipt.json"]
        )


def test_source_gate_override_must_match_live_parent(
    r175_parent_and_candidate: tuple[Path, Path, Path],
) -> None:
    source_registry, candidate, source_pin = r175_parent_and_candidate
    wrong = candidate / "wrong.json"
    _write(wrong, copy.deepcopy(_json(ROOT / "ops/final_format_alakazam_gate_r100_v1.json")))

    with pytest.raises(RuntimeError, match="--source-gate differs"):
        stage.main(
            _stage_args(source_registry, candidate, source_pin)
            + ["--source-gate", str(wrong)]
        )
