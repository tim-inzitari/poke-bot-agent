from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "activate_alakazam_marnie_splusplus_r192.py"
SPEC = importlib.util.spec_from_file_location("activate_alakazam_marnie_r192", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)

STAGE_SCRIPT = ROOT / "scripts" / "stage_alakazam_marnie_splusplus_r192.py"
STAGE_SPEC = importlib.util.spec_from_file_location("stage_alakazam_marnie_r192", STAGE_SCRIPT)
assert STAGE_SPEC is not None and STAGE_SPEC.loader is not None
stage = importlib.util.module_from_spec(STAGE_SPEC)
sys.modules[STAGE_SPEC.name] = stage
STAGE_SPEC.loader.exec_module(stage)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(key: str, path: Path) -> controller.Artifact:
    return controller.Artifact(key=key, path=path, digest=controller._sha256(path))


def _exact_crustle_pin() -> dict:
    return {
        "package_id": "specialist-crustle-final-format-h10-7efd8d4113e7",
        "checkpoint_sha256": (
            "sha256:7efd8d4113e736d28576bdbfa1c9d1c3f3a7cf1a31a0b3cfadd1e7f82cf08955"
        ),
        "content_digest": (
            "sha256:359e3b4fed00502e58be4631576501b6f63523226ec92f2d75446df085b19afa"
        ),
        "floor_games_per_set": 512,
    }


def _producer_crustle_pin() -> dict:
    return {
        "package_id": "specialist-crustle-final-format-h10-7efd8d4113e7",
        "floor_games_per_set": 512,
        "scheduled_games": 512,
        # The final receipt schema requires an explicit non-boolean positive
        # conversion count; zero is deliberately not a valid proof.
        "converted_from_diverse": 1,
        "met": True,
    }


def _runtime_registry(tmp_path: Path) -> tuple[dict, dict[str, controller.Artifact]]:
    root = tmp_path / "runtime"
    root.mkdir()
    owner = _write_json(root / "owner.json", {"owner": 192})
    gate = _write_json(root / "gate.json", {"gate": 192})
    frozen = _write_json(root / "frozen.json", {"frozen": 192})
    pins = _write_json(
        root / "pins.json",
        {"schema": controller.PIN_FLOORS_SCHEMA, "pins": [_exact_crustle_pin()]},
    )
    bindings = {
        "owner_contract": _artifact("owner_contract", owner),
        "active_gate_contract": _artifact("active_gate_contract", gate),
        "frozen_specialist_registry": _artifact("frozen_specialist_registry", frozen),
        "public_mix_pin_floors": _artifact("public_mix_pin_floors", pins),
    }
    artifacts = {
        "owner_contract": bindings["owner_contract"],
        "gate_artifact": bindings["active_gate_contract"],
        "frozen_artifact": bindings["frozen_specialist_registry"],
        "pin_artifact": bindings["public_mix_pin_floors"],
    }
    pin = {
        "package_id": controller.OPPONENT_ID,
        "checkpoint_sha256": controller.CHECKPOINT_DIGEST,
        "content_digest": controller.CONTENT_DIGEST,
        "floor_games_per_set": controller.FLOOR_GAMES_PER_SET,
        "tier": controller.TIER,
        "weight": controller.WEIGHT,
        "enforcement_source": "exact_active_gate_strong_public_practice_floor",
        "legacy_diverse_public_sidecar_pin_active": False,
        "superseded_by_exact_strong_gate_floor": True,
    }
    registry = {
        "schema": controller.RUNTIME_REGISTRY_SCHEMA,
        "owner_decision_revision": 192,
        "runtime_root": str(root),
        "active_gate_contract": "gate.json",
        "frozen_specialist_registry": "frozen.json",
        "minimum_terminal_iteration": 5,
        "common_trainer_args": [
            "--gate-boundary-pause-seconds",
            "30",
            "--official-adaptive-min-share",
            "0.04",
            controller.R192_MIGRATION_FLAG,
            controller.R192_MIGRATION_REASON_FLAG,
            controller.R192_MIGRATION_REASON,
        ],
        "isolated_refresh_contract": {
            "grimmsnarl_package_id": controller.OPPONENT_ID,
            "grimmsnarl_checkpoint_sha256": controller.CHECKPOINT_DIGEST,
            "grimmsnarl_content_digest": controller.CONTENT_DIGEST,
            "grimmsnarl_floor_per_set": 1024,
            "grimmsnarl_tier": "S++",
            "grimmsnarl_weight": 4.0,
        },
        "specialists": {
            "alakazam": {
                "minimum_terminal_iteration": 5,
                "owner_grimmsnarl_pin": pin,
            }
        },
        "r192_stage": {
            "boundary_design_migration": {
                "allow_clean_boundary_design_migration": True,
                "reason": controller.R192_MIGRATION_REASON,
            },
            "stop_budget_guard": controller.STOP_BUDGET_GUARD,
            "artifact_bindings": {
                key: {"path": str(artifact.path), "sha256": artifact.digest}
                for key, artifact in bindings.items()
            }
        },
    }
    return registry, artifacts


def test_candidate_sidecar_excludes_h10_but_preserves_crustle(tmp_path: Path) -> None:
    sidecar = {
        "schema": controller.PIN_FLOORS_SCHEMA,
        "pins": [_exact_crustle_pin()],
    }
    controller._verify_pin_floors(sidecar)

    sidecar["pins"].append({"package_id": controller.OPPONENT_ID})
    with pytest.raises(RuntimeError, match="must not reinsert H10 Marnie"):
        controller._verify_pin_floors(sidecar)


def test_candidate_pin_receipt_requires_exact_retained_crustle(
    tmp_path: Path,
) -> None:
    plan_path = _write_json(
        tmp_path / "iter_00006.json",
        {
            "schema": "poke_bot.strong_public_practice_plan/v1",
            "iteration": 6,
            "active_gate_id": "gate-r192",
            "games": 4586,
            "group_games_per_iteration": {
                "self_play": 1024,
                "strong_public_practice": 4586,
                "diverse_public": 2586,
            },
            "minimum_games_by_opponent": {controller.OPPONENT_ID: 1024},
            "per_opponent": {
                controller.OPPONENT_ID: {"games": 1024, "minimum_games": 1024}
            },
        },
    )
    pin_receipt_path = _write_json(
        tmp_path / "iter_00006.owner_public_mix_pin_floors.json",
        {
            "schema": "poke_bot.owner_public_mix_pin_floor_receipt/v1",
            "pins": [_producer_crustle_pin()],
        },
    )
    controller._validate_candidate_collection_plan(
        plan_path=plan_path,
        pin_receipt_path=pin_receipt_path,
        expected_iteration=6,
        gate_id="gate-r192",
    )

    _write_json(
        pin_receipt_path,
        {
            "schema": "poke_bot.owner_public_mix_pin_floor_receipt/v1",
            "pins": [],
        },
    )
    with pytest.raises(
        controller.CandidateDispatchMayHaveStarted,
        match="retained Crustle-512 scheduling",
    ):
        controller._validate_candidate_collection_plan(
            plan_path=plan_path,
            pin_receipt_path=pin_receipt_path,
            expected_iteration=6,
            gate_id="gate-r192",
        )


def test_runtime_requires_r192_stage_bindings_and_adaptive_point04(
    tmp_path: Path,
) -> None:
    registry, artifacts = _runtime_registry(tmp_path)
    controller._verify_runtime_registry(registry, **artifacts)

    args = registry["common_trainer_args"]
    args[args.index("--official-adaptive-min-share") + 1] = "0.03"
    with pytest.raises(RuntimeError, match="adaptive min share 0.04"):
        controller._verify_runtime_registry(registry, **artifacts)


def test_runtime_requires_exact_one_r192_migration_authorization(
    tmp_path: Path,
) -> None:
    registry, artifacts = _runtime_registry(tmp_path)
    registry["common_trainer_args"].extend(
        [
            controller.R192_MIGRATION_FLAG,
            controller.R192_MIGRATION_REASON_FLAG,
            "generic-reason",
        ]
    )
    with pytest.raises(RuntimeError, match="must occur exactly once"):
        controller._verify_runtime_registry(registry, **artifacts)

    other_root = tmp_path / "missing"
    other_root.mkdir()
    registry, artifacts = _runtime_registry(other_root)
    args = registry["common_trainer_args"]
    del args[args.index(controller.R192_MIGRATION_FLAG)]
    with pytest.raises(RuntimeError, match="must occur exactly once"):
        controller._verify_runtime_registry(registry, **artifacts)


def test_canonical_stage_receipt_is_preflight_valid_but_unarmed(
    tmp_path: Path,
) -> None:
    """Exercise the real deterministic materializer without authorizing apply."""

    design_path = ROOT / "state" / "alakazam-marnie-splusplus-opponent-r192.json"
    design = stage._load_design(design_path)
    source_gate = json.loads(
        (ROOT / "ops" / "final_format_alakazam_gate_r100_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source_frozen = json.loads(
        (ROOT / "ops" / "frozen_specialist_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    gate = stage.build_gate(source_gate, design)
    frozen = stage.build_frozen_registry(source_frozen, design)
    source_gate_path = tmp_path / "r175-gate.json"
    source_frozen_path = tmp_path / "r175-frozen.json"
    gate_path = tmp_path / "r192-gate.json"
    frozen_path = tmp_path / "r192-frozen.json"
    for path, value in (
        (source_gate_path, source_gate),
        (source_frozen_path, source_frozen),
        (gate_path, gate),
        (frozen_path, frozen),
    ):
        stage._atomic_json(path, value)
    source_pin = {
        "schema": stage.PIN_FLOORS_SCHEMA,
        "owner_decision_revision": 175,
        "pins": [
            {
                "package_id": stage.CRUSTLE_H10_OPPONENT_ID,
                "checkpoint_sha256": stage.CRUSTLE_H10_CHECKPOINT_DIGEST,
                "content_digest": stage.CRUSTLE_H10_CONTENT_DIGEST,
                "floor_games_per_set": 512,
            },
            {
                "package_id": stage.H10_MARNIE_OPPONENT_ID,
                "archetype_id": stage.H10_MARNIE_ARCHETYPE_ID,
                "checkpoint_sha256": stage.H10_MARNIE_CHECKPOINT_DIGEST,
                "content_digest": stage.H10_MARNIE_CONTENT_DIGEST,
                "floor_games_per_set": 1024,
            },
        ],
    }
    source_pin_path = tmp_path / "r175-pins.json"
    pin_path = tmp_path / "r192-pins.json"
    stage._atomic_json(source_pin_path, source_pin)
    pin = stage.build_pin_sidecar(source_pin, design)
    stage._atomic_json(pin_path, pin)
    log_path = tmp_path / "r175.log"
    log_path.write_text("r175 active\n", encoding="utf-8")
    source_runtime = {
        "schema": stage.RUNTIME_REGISTRY_SCHEMA,
        "owner_decision_revision": 175,
        "runtime_root": str(tmp_path),
        "minimum_terminal_iteration": 5,
        "active_gate_contract": source_gate_path.name,
        "frozen_specialist_registry": source_frozen_path.name,
        "common_trainer_args": [
            "--gate-boundary-pause-seconds",
            "30",
            "--official-adaptive-min-share",
            "0.04",
        ],
        "isolated_refresh_contract": {"schema": "r175"},
        "specialists": {
            "alakazam": {
                "minimum_terminal_iteration": 5,
                "owner_grimmsnarl_pin": {
                    "package_id": stage.H10_MARNIE_OPPONENT_ID,
                },
                "log": str(log_path),
                "run_name": "alakazam-r175",
            }
        },
    }
    source_runtime_path = tmp_path / "r175-runtime.json"
    stage._atomic_json(source_runtime_path, source_runtime)
    runtime = stage.build_runtime_registry(
        source_runtime,
        gate_reference=gate_path.name,
        frozen_reference=frozen_path.name,
        gate_id=str(gate["active_gate_id"]),
        gate_sha256=stage._json_file_digest(gate),
        frozen_sha256=stage._json_file_digest(frozen),
        owner_contract_reference=str(design_path.resolve()),
        owner_contract_sha256=stage._file_digest(design_path),
        pin_sidecar_reference=pin_path.name,
        pin_sidecar_sha256=stage._json_file_digest(pin),
    )
    runtime_path = tmp_path / "r192-runtime.json"
    stage._atomic_json(runtime_path, runtime)
    baseline_manifest_path = tmp_path / "baseline-manifest.json"
    h10_model_path = tmp_path / "h10-marnie-model.pt"
    baseline_manifest_path.write_text('{"agents": []}\n', encoding="utf-8")
    h10_model_path.write_bytes(b"checksum-bound H10 Marnie fixture\n")
    receipt = stage.build_stage_receipt(
        design_path=design_path,
        source_runtime_registry=source_runtime_path,
        source_gate=source_gate_path,
        source_frozen=source_frozen_path,
        source_pin_sidecar=source_pin_path,
        runtime_registry=runtime_path,
        gate=gate_path,
        frozen=frozen_path,
        pin_sidecar=pin_path,
        deployment_inputs={
            "launch_pure_rl": ROOT / "scripts" / "launch_pure_rl.py",
            "train_pure_rl": ROOT / "scripts" / "train_pure_rl.py",
            "launch_active_specialist": ROOT
            / "scripts"
            / "launch_active_specialist.py",
            "activation_controller": ROOT
            / "scripts"
            / "activate_alakazam_marnie_splusplus_r192.py",
            "dropin_template": ROOT
            / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-rl.service.d"
            / "62-marnie-splusplus-r192.conf.in",
            "boundary_service_template": ROOT
            / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-marnie-splusplus-r192-boundary.service.in",
            "stop_budget_template": ROOT
            / "deploy/systemd/pokebot-final-format-alakazam-rtp-r175-rl.service.d"
            / "61-marnie-splusplus-r192-stop-budget.conf.in",
            "strong_public_gate": ROOT / "poke_bot" / "pure_rl" / "strong_public_gate.py",
            "public_multi_env_safety": ROOT
            / "poke_bot"
            / "public_multi_env_safety.py",
            "r182_transport_contract": ROOT
            / "state"
            / "alakazam-public-multi-env-split-r182.json",
            "baseline_manifest": baseline_manifest_path,
            "h10_marnie_model_provenance": h10_model_path,
        },
    )
    receipt_path = tmp_path / "r192-stage.json"
    stage._atomic_json(receipt_path, receipt)

    plan = controller.preflight(
        stage_path=receipt_path,
        owner_contract_path=design_path,
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
        "reason": controller.R192_MIGRATION_REASON,
    }
    assert receipt["stop_budget_guard"] == controller.STOP_BUDGET_GUARD
    assert set(receipt["activation_artifacts"]) == {
        "controller",
        "dropin_template",
        "boundary_service_template",
        "stop_budget_template",
    }
    assert receipt["collection_contract"] == {
        "games_per_iteration": 8196,
        "self_play_mirrors": 1024,
        "public_mix_games": 7172,
        "strong_public_practice_games": 4586,
        "diverse_public_games": 2586,
        "ordinary_strong_public_minimum_share": 0.04,
        "h10_executable_floor_owner": "exact_active_gate_strong_public_practice_floor",
        "legacy_diverse_public_h10_pin_removed_on_activation": True,
    }
    assert plan["gate_id"] == gate["active_gate_id"]
    assert plan["source_artifacts"]["active_gate_contract"]["path"] == str(
        source_gate_path.resolve()
    )
    assert plan["activation_artifacts"] == receipt["activation_artifacts"]


def test_waiter_refuses_late_post_iter5_poll(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "loop_state.json", {"last_completed_iteration": 5, "next_iteration": 6})
    log = tmp_path / "rl.log"
    log.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be armed before"):
        controller._wait_for_visible_gate_pause(
            run_dir=run_dir,
            log_path=log,
            expected_iteration=5,
            pause_seconds=30.0,
            poll_seconds=0.01,
            minimum_remaining_seconds=20.0,
            wait_timeout_seconds=1.0,
        )


def test_next_iteration_runtime_state_blocks_rollback_proof(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    runtime_state = _write_json(
        run_dir / "iteration_runtime.json",
        {"iteration": 6, "phase": "collect"},
    )
    assert controller._next_iteration_artifacts(
        run_dir=run_dir,
        next_iteration=6,
    ) == [runtime_state]
    with pytest.raises(RuntimeError, match="next iteration artifact"):
        controller._assert_no_next_iteration_artifacts(
            run_dir=run_dir,
            next_iteration=6,
        )


def test_waiter_refreshes_negative_marker_time_before_iter5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    log = tmp_path / "rl.log"
    log.write_text("", encoding="utf-8")
    commit = _write_json(
        run_dir / "commits" / "iter_00005.json",
        {
            "last_completed_iteration": 5,
            "next_iteration": 6,
            "history": [{"iteration": 5, "completed": True}],
        },
    )
    calls = {"loop": 0, "log": 0, "time": 0}

    def fake_read_json(path: Path) -> dict:
        if path.name == "loop_state.json":
            calls["loop"] += 1
            if calls["loop"] < 3:
                return {"last_completed_iteration": 4, "next_iteration": 5}
            return {"last_completed_iteration": 5, "next_iteration": 6}
        if path == commit:
            return json.loads(commit.read_text(encoding="utf-8"))
        raise AssertionError(path)

    def fake_read_log(path: Path, offset: int) -> tuple[str, int]:
        calls["log"] += 1
        return (
            (
                ""
                if calls["log"] == 1
                else "[pure_rl] GATE_BOUNDARY_HARD_PAUSE iteration=5 seconds=30.0 stage_gate_passed=true next_collection_blocked=true\n"
            ),
            offset + 1,
        )

    def fake_monotonic() -> float:
        calls["time"] += 1
        return float(calls["time"])

    monkeypatch.setattr(controller, "_read_json", fake_read_json)
    monkeypatch.setattr(controller, "_read_log_delta", fake_read_log)
    monkeypatch.setattr(controller.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    observation = controller._wait_for_visible_gate_pause(
        run_dir=run_dir,
        log_path=log,
        expected_iteration=5,
        pause_seconds=30.0,
        poll_seconds=0.01,
        minimum_remaining_seconds=20.0,
        wait_timeout_seconds=90.0,
    )
    assert observation.commit_path == commit
    # It is based on the immediately preceding negative log read, not the
    # watcher arm time (which would make a days-long resident watcher fail).
    assert observation.safe_pause_deadline_monotonic > 30.0


def test_inactive_boundary_proof_requires_stopped_service_and_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/iter_00016.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"r192 inactive boundary learner")
    digest = controller._sha256(checkpoint)
    state = {
        "last_completed_iteration": 16,
        "next_iteration": 17,
        "learner": {"path": str(checkpoint), "digest": digest},
        "history": [{"iteration": 16, "completed": True}],
    }
    commit = _write_json(run_dir / "commits/iter_00016.json", state)
    _write_json(run_dir / "loop_state.json", state)
    receipt = _write_json(
        tmp_path / "boundary.json",
        {
            "schema": "poke_bot.committed_iteration_pause/v1",
            "status": "paused",
            "unit": "trainer.service",
            "completed_iteration": 16,
            "next_iteration": 17,
            "commit": str(commit.resolve()),
            "commit_digest": controller._sha256(commit),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_digest": digest,
            "uncommitted_next_iteration_started": False,
            "recovery_required": False,
            "service_active_state": "inactive",
        },
    )
    monkeypatch.setattr(
        controller,
        "_service_value",
        lambda _unit, key: "inactive" if key == "ActiveState" else "0",
    )

    proof = controller._inactive_boundary_proof(
        receipt_path=receipt,
        run_dir=run_dir,
        service="trainer.service",
        expected_iteration=16,
    )

    assert proof["commit_sha256"] == controller._sha256(commit)
    assert proof["checkpoint_sha256"] == digest
    assert proof["next_iteration"] == 17

    monkeypatch.setattr(
        controller,
        "_service_value",
        lambda _unit, key: "active" if key == "ActiveState" else "123",
    )
    with pytest.raises(RuntimeError, match="still active"):
        controller._inactive_boundary_proof(
            receipt_path=receipt,
            run_dir=run_dir,
            service="trainer.service",
            expected_iteration=16,
        )


def test_dropin_render_is_exact_and_rejects_token_drift(tmp_path: Path) -> None:
    template = tmp_path / "candidate.conf.in"
    template.write_text(
        "\n".join(
            [
                "@RUNTIME_ROOT@",
                "@RUNTIME_REGISTRY@",
                "@RUNTIME_REGISTRY@",
                "@RUNTIME_REGISTRY@",
                "@RUNTIME_REGISTRY@",
                "@PIN_FLOORS@",
                "@PIN_FLOORS@",
                "@TRAINING_ARM_FILE@",
                "@PYTHON@",
                "@PYTHON@",
                "@LAUNCH_ACTIVE_SPECIALIST@",
                "@LAUNCH_ACTIVE_SPECIALIST@",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = controller._render_dropin(
        template=template,
        runtime_root=tmp_path,
        runtime_registry=tmp_path / "runtime.json",
        pin_floors=tmp_path / "pins.json",
        training_arm_file=tmp_path / "TRAINING_ARMED",
        launch_active_specialist=tmp_path / "launch.py",
    )
    assert "@" not in rendered
    assert str(tmp_path / "runtime.json") in rendered

    template.write_text("@RUNTIME_REGISTRY@\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="token count changed"):
        controller._render_dropin(
            template=template,
            runtime_root=tmp_path,
            runtime_registry=tmp_path / "runtime.json",
            pin_floors=tmp_path / "pins.json",
            training_arm_file=tmp_path / "TRAINING_ARMED",
            launch_active_specialist=tmp_path / "launch.py",
        )


def test_cli_apply_is_fail_closed_without_inactive_boundary_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public --apply entry point cannot use a pause marker as authority."""

    preflight_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        controller,
        "preflight",
        lambda **kwargs: preflight_calls.append(kwargs) or {"gate_id": "r192"},
    )
    stage_path = tmp_path / "stage.json"
    owner_path = tmp_path / "owner.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--stage-receipt",
            str(stage_path),
            "--owner-contract",
            str(owner_path),
            "--apply",
        ],
    )

    with pytest.raises(RuntimeError) as raised:
        controller.main()

    assert str(raised.value) == controller.UNARMED_APPLY_REASON
    assert preflight_calls == [
        {"stage_path": stage_path, "owner_contract_path": owner_path}
    ]
