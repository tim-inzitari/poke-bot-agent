from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot import public_multi_env_safety


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "train_pure_rl.py"
SPEC = importlib.util.spec_from_file_location(
    "train_pure_rl_alakazam_marnie_splusplus_r192", TRAINER_PATH
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)

STAGE_PATH = ROOT / "scripts" / "stage_alakazam_marnie_splusplus_r192.py"
STAGE_SPEC = importlib.util.spec_from_file_location(
    "stage_alakazam_marnie_splusplus_r192", STAGE_PATH
)
assert STAGE_SPEC is not None and STAGE_SPEC.loader is not None
stage = importlib.util.module_from_spec(STAGE_SPEC)
STAGE_SPEC.loader.exec_module(stage)

ACTIVATE_PATH = ROOT / "scripts" / "activate_alakazam_marnie_splusplus_r192.py"
ACTIVATE_SPEC = importlib.util.spec_from_file_location(
    "activate_alakazam_marnie_splusplus_r192", ACTIVATE_PATH
)
assert ACTIVATE_SPEC is not None and ACTIVATE_SPEC.loader is not None
activate = importlib.util.module_from_spec(ACTIVATE_SPEC)
sys.modules[ACTIVATE_SPEC.name] = activate
ACTIVATE_SPEC.loader.exec_module(activate)


H10_MARNIE = "specialist-marnie-final-format-h10-f20efb20f5c3"
H10_MARNIE_DIGEST = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
HISTORICAL_MARNIE = "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98"


def _materialized_active_gate() -> dict:
    roster = [
        {
            "opponent_id": H10_MARNIE,
            "archetype_id": "marnie-s-grimmsnarl-ex",
            "content_digest": H10_MARNIE_DIGEST,
            "tier": "S++",
            "weight": 4.0,
            "frozen_specialist": True,
            "strong_public_practice_floor_games": 1024,
        },
        {
            "opponent_id": HISTORICAL_MARNIE,
            "archetype_id": "marnie-s-grimmsnarl-ex",
            "content_digest": "sha256:" + "a" * 64,
            "tier": "S+",
            "weight": 2.0,
            "frozen_specialist": True,
        },
    ]
    for index in range(16):
        roster.append(
            {
                "opponent_id": f"gate-opponent-{index:02d}",
                "archetype_id": f"archetype-{index:02d}",
                "content_digest": "sha256:" + f"{index + 1:064x}",
                "tier": "S+" if index < 13 else "A",
                "weight": 2.0 if index < 13 else 1.0,
                "frozen_specialist": index < 13,
            }
        )
    return {
        "id": "final-format-alakazam-r175-r192",
        "evaluation": {
            "games_per_opponent": 250,
            "seat0_games_per_opponent": 125,
            "seat1_games_per_opponent": 125,
            "games_total": 4500,
        },
        "roster": roster,
    }


def _content_digest(opponent_id: str) -> str:
    if opponent_id == H10_MARNIE:
        return H10_MARNIE_DIGEST
    return "sha256:" + hashlib.sha256(opponent_id.encode("utf-8")).hexdigest()


def test_r192_stage_derivation_is_checksum_exact_and_additive(
    tmp_path: Path,
) -> None:
    design = stage._load_design(
        ROOT / "state" / "alakazam-marnie-splusplus-opponent-r192.json"
    )
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
    stage.validate_artifacts(gate=gate, frozen=frozen, design=design)

    gate_rows = gate["next_gate"]["roster"]
    source_historical = next(
        row
        for row in source_gate["next_gate"]["roster"]
        if row["opponent_id"] == HISTORICAL_MARNIE
    )
    staged_historical = next(
        row for row in gate_rows if row["opponent_id"] == HISTORICAL_MARNIE
    )
    h10 = next(row for row in gate_rows if row["opponent_id"] == H10_MARNIE)
    assert len(gate_rows) == 18
    assert gate["next_gate"]["evaluation"]["games_total"] == 4500
    assert staged_historical == source_historical
    assert h10 == {
        "archetype_id": "marnie-s-grimmsnarl-ex",
        "archetype_label": "Frozen final-format H10 Marnie's Grimmsnarl ex refresh",
        "content_digest": H10_MARNIE_DIGEST,
        "frozen_checkpoint_digest": (
            "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3"
            "bbb431f9c8b44381"
        ),
        "frozen_specialist": True,
        "opponent_id": H10_MARNIE,
        "owner_decision_revision": 192,
        "source": (
            "checksum-exact final-format H10 Marnie's Grimmsnarl ex "
            "checkpoint f20efb20f5c3; distinct from historical Marnie"
        ),
        "strong_public_practice_floor_games": 1024,
        "tier": "S++",
        "weight": 4.0,
    }
    assert gate["active_gate_semantics"][
        "exact_additional_splusplus_specialist"
    ] == {
        "opponent_id": H10_MARNIE,
        "checkpoint_digest": h10["frozen_checkpoint_digest"],
        "content_digest": H10_MARNIE_DIGEST,
        "tier": "S++",
        "weight": 4.0,
        "strong_public_practice_floor_games": 1024,
    }
    frozen_h10 = next(
        row for row in frozen["specialists"] if row["opponent_id"] == H10_MARNIE
    )
    assert frozen_h10["premium_holdout_tier"] == "S++"
    assert frozen_h10["premium_holdout_weight"] == 4.0
    assert frozen_h10["strong_public_practice_floor_games"] == 1024

    runtime = stage.build_runtime_registry(
        {
            "schema": stage.RUNTIME_REGISTRY_SCHEMA,
            "owner_decision_revision": stage.PARENT_OWNER_REVISION,
            "minimum_terminal_iteration": 5,
            "common_trainer_args": [
                "--official-adaptive-min-share",
                "0.04",
            ],
            "isolated_refresh_contract": {"schema": "r175"},
            "specialists": {
                "alakazam": {
                    "minimum_terminal_iteration": 5,
                    "owner_grimmsnarl_pin": {"package_id": H10_MARNIE},
                }
            },
        },
        gate_reference="/stage/gate.json",
        frozen_reference="/stage/frozen.json",
        gate_id=str(gate["active_gate_id"]),
        gate_sha256=stage._json_file_digest(gate),
        frozen_sha256=stage._json_file_digest(frozen),
    )
    assert runtime["owner_decision_revision"] == 192
    assert runtime["isolated_refresh_contract"]["grimmsnarl_content_digest"] == (
        H10_MARNIE_DIGEST
    )
    assert runtime["isolated_refresh_contract"]["grimmsnarl_tier"] == "S++"
    assert runtime["isolated_refresh_contract"]["grimmsnarl_weight"] == 4.0
    assert runtime["specialists"]["alakazam"]["owner_grimmsnarl_pin"] == {
        "package_id": H10_MARNIE,
        "checkpoint_sha256": (
            "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3"
            "bbb431f9c8b44381"
        ),
        "content_digest": H10_MARNIE_DIGEST,
        "floor_games_per_set": 1024,
        "tier": "S++",
        "weight": 4.0,
        "enforcement_source": "exact_active_gate_strong_public_practice_floor",
        "legacy_diverse_public_sidecar_pin_active": False,
        "superseded_by_exact_strong_gate_floor": True,
    }
    assert runtime["r192_stage"]["artifact_bindings"] == {
        "gate_path": "/stage/gate.json",
        "gate_sha256": stage._json_file_digest(gate),
        "frozen_registry_path": "/stage/frozen.json",
        "frozen_registry_sha256": stage._json_file_digest(frozen),
    }


def test_r192_full_stage_handoff_is_preflight_compatible_but_unarmed(
    tmp_path: Path,
) -> None:
    """The local stage artifact is inert and cannot authorize an r175 restart."""

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
    source_pin = {
        "schema": stage.PIN_FLOORS_SCHEMA,
        "owner_decision_revision": 175,
        "pins": [
            {
                "archetype_id": "crustle",
                "checkpoint_sha256": stage.CRUSTLE_H10_CHECKPOINT_DIGEST,
                "content_digest": stage.CRUSTLE_H10_CONTENT_DIGEST,
                "floor_games_per_set": stage.CRUSTLE_H10_FLOOR_GAMES,
                "package_id": stage.CRUSTLE_H10_OPPONENT_ID,
            },
            {
                "archetype_id": "marnie-s-grimmsnarl-ex",
                "checkpoint_sha256": stage.H10_MARNIE_CHECKPOINT_DIGEST,
                "floor_games_per_set": 1024,
                "package_id": H10_MARNIE,
            },
        ],
    }
    pin = stage.build_pin_sidecar(source_pin, design)
    stage.validate_artifacts(gate=gate, frozen=frozen, design=design)
    stage.validate_pin_sidecar(pin, design)
    assert pin["pins"] == [source_pin["pins"][0]]

    gate_path = tmp_path / "gate.json"
    frozen_path = tmp_path / "frozen.json"
    pin_path = tmp_path / "pin.json"
    source_gate_path = tmp_path / "r175-gate.json"
    source_frozen_path = tmp_path / "r175-frozen.json"
    source_runtime_path = tmp_path / "r175-runtime.json"
    runtime_path = tmp_path / "r192-runtime.json"
    source_pin_path = tmp_path / "r175-pin.json"
    log_path = tmp_path / "r175.log"
    log_path.write_text("r175 remains active\n", encoding="utf-8")
    source_runtime = {
        "schema": stage.RUNTIME_REGISTRY_SCHEMA,
        "owner_decision_revision": 175,
        "runtime_root": str(tmp_path),
        "minimum_terminal_iteration": 5,
        "common_trainer_args": [
            "--gate-boundary-pause-seconds",
            "30",
            "--official-adaptive-min-share",
            "0.04",
        ],
        "active_gate_contract": source_gate_path.name,
        "frozen_specialist_registry": source_frozen_path.name,
        "isolated_refresh_contract": {"schema": "r175"},
        "specialists": {
            "alakazam": {
                "minimum_terminal_iteration": 5,
                "owner_grimmsnarl_pin": {"package_id": H10_MARNIE},
                "log": str(log_path),
                "run_name": "alakazam-r175",
            }
        },
    }
    stage._atomic_json(gate_path, gate)
    stage._atomic_json(frozen_path, frozen)
    stage._atomic_json(pin_path, pin)
    stage._atomic_json(source_gate_path, source_gate)
    stage._atomic_json(source_frozen_path, source_frozen)
    stage._atomic_json(source_runtime_path, source_runtime)
    stage._atomic_json(source_pin_path, source_pin)
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
    stage._atomic_json(runtime_path, runtime)
    assert runtime["r192_stage"]["artifact_bindings"]["public_mix_pin_floors"] == {
        "path": pin_path.name,
        "sha256": stage._json_file_digest(pin),
    }

    receipt_path = tmp_path / "r192-stage.json"
    baseline_manifest_path = tmp_path / "baseline-manifest.json"
    h10_marnie_model_path = tmp_path / "h10-marnie-model.pt"
    baseline_manifest_path.write_text("{\"agents\": []}\n", encoding="utf-8")
    h10_marnie_model_path.write_bytes(b"checksum-bound H10 Marnie fixture\n")
    stage_receipt = stage.build_stage_receipt(
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
            "train_pure_rl": TRAINER_PATH,
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
            "h10_marnie_model_provenance": h10_marnie_model_path,
        },
    )
    stage._atomic_json(receipt_path, stage_receipt)
    plan = activate.preflight(
        stage_path=receipt_path,
        owner_contract_path=design_path,
    )
    assert stage_receipt["status"] == "staged_non_active"
    assert stage_receipt["active_before_receipt_backed_activation"] is False
    assert stage_receipt["training_interrupted"] is False
    assert stage_receipt["selector_or_service_changed"] is False
    assert stage_receipt["remote_deployment_performed"] is False
    assert (
        stage_receipt[
            "managed_restart_during_verified_post_iteration5_hard_pause_allowed"
        ]
        is False
    )
    assert stage_receipt["automatic_managed_restart_armed"] is False
    assert stage_receipt["trainer_owned_handoff_fence_required"] is True
    assert stage_receipt["current_r175_source_has_trainer_owned_handoff_fence"] is False
    assert plan["gate_id"] == gate["active_gate_id"]
    assert plan["expected_after_iteration"] == 5
    assert plan["boundary_pause_seconds"] == 30.0
    assert plan["staged_artifacts"]["public_mix_pin_floors"]["path"] == str(
        pin_path
    )
    assert plan["staged_artifacts"]["runtime_registry"]["path"] == str(
        runtime_path
    )
    assert plan["no_remote_deployment_performed"] is True
    assert plan["required_remote_receipts_are_stage_inputs"] is True


def test_materialized_h10_marnie_row_survives_frozen_gate_augmentation() -> None:
    active_gate = _materialized_active_gate()
    contract = {
        "active_gate_id": active_gate["id"],
        "next_gate": active_gate,
        "active_gate_semantics": {
            "gate_roster_size": 18,
            "gate_games_total": 4500,
        },
    }
    frozen_registry = {
        "version": 192,
        "specialists": [
            {"opponent_id": str(row["opponent_id"])}
            for row in active_gate["roster"]
            if row.get("frozen_specialist") is True
        ],
    }

    effective = trainer._augment_gate_with_frozen_specialists(
        contract, frozen_registry
    )
    h10_rows = [
        row
        for row in effective["next_gate"]["roster"]
        if row["opponent_id"] == H10_MARNIE
    ]

    assert effective == contract
    assert len(h10_rows) == 1
    assert h10_rows[0]["tier"] == "S++"
    assert h10_rows[0]["weight"] == 4.0
    assert h10_rows[0]["strong_public_practice_floor_games"] == 1024
    assert sum(
        row["opponent_id"] == HISTORICAL_MARNIE
        for row in effective["next_gate"]["roster"]
    ) == 1
    assert effective["next_gate"]["evaluation"]["games_total"] == 4500


def test_r192_h10_floor_stays_in_strong_practice_without_expanding_public_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_gate = _materialized_active_gate()
    roster = active_gate["roster"]
    group_plan = trainer._planned_collection_group_counts(
        games_per_iteration=8196,
        self_play_fraction=1024 / 8196,
        strong_public_fraction_of_public=0.50,
        research_control_games=1000,
    )
    assert group_plan == {
        "self_play": 1024,
        trainer.STRONG_PUBLIC_PRACTICE_GROUP: 4586,
        "diverse_public": 2586,
    }
    minimums = trainer._strong_public_practice_minimum_games(
        active_gate=active_gate,
        expected_practice_games=group_plan[trainer.STRONG_PUBLIC_PRACTICE_GROUP],
        minimum_share=0.04,
    )
    assert minimums[H10_MARNIE] == 1024
    assert set(minimums) == {str(row["opponent_id"]) for row in roster}
    assert all(
        minimums[row["opponent_id"]] == 183
        for row in roster
        if row["opponent_id"] != H10_MARNIE
    )
    assert sum(minimums.values()) == 4135
    with pytest.raises(RuntimeError, match="floors exceed the fixed quota"):
        trainer._strong_public_practice_minimum_games(
            active_gate=active_gate,
            expected_practice_games=group_plan[
                trainer.STRONG_PUBLIC_PRACTICE_GROUP
            ],
            minimum_share=0.05,
        )

    monkeypatch.setattr(
        trainer,
        "_spec_payload",
        lambda spec: {
            "id": str(spec.id),
            "contract_schema": "poke_bot.portable_baseline_spec/v1",
            "content_digest": _content_digest(str(spec.id)),
        },
    )
    priority_specs = [
        SimpleNamespace(id=str(row["opponent_id"])) for row in roster
    ]
    # The same public catalog previously exposes H10 Marnie as a diverse row.
    # Gate membership removes it before the two group schedules are built.
    public_catalog = [
        SimpleNamespace(id=H10_MARNIE),
        SimpleNamespace(id="diverse-a"),
        SimpleNamespace(id="diverse-b"),
    ]
    active_gate_ids = {str(row["opponent_id"]) for row in roster}
    diverse_specs = [
        spec for spec in public_catalog if str(spec.id) not in active_gate_ids
    ]
    assert H10_MARNIE not in {str(spec.id) for spec in diverse_specs}
    weights = {
        str(row["opponent_id"]): float(row["weight"])
        for row in roster
    }
    self_jobs, public_jobs = trainer._build_collect_jobs(
        n_games=8196,
        ckpt=Path("/tmp/alakazam-r175.pt"),
        digest="sha256:" + "b" * 64,
        model_generation=2,
        decks=[("alakazam", [1] * 60)],
        specs=diverse_specs,
        seed=192_000,
        game_timeout_s=10,
        mode="specialist",
        self_play_frac=1024 / 8196,
        iteration=2,
        priority_specs=priority_specs,
        priority_frac=4586 / 7172,
        priority_weights=weights,
        priority_minimum_games=minimums,
        priority_group=trainer.STRONG_PUBLIC_PRACTICE_GROUP,
        priority_temperature=0.35,
        priority_archetypes={
            str(row["opponent_id"]): str(row["archetype_id"])
            for row in roster
        },
        priority_context={
            "active_gate_id": active_gate["id"],
            "formal_eval": False,
            "seed_namespace": "train/strong-public-practice-v1",
            "formal_gate_seed_namespace": "eval/strong-public-fixed-manifest-v1",
        },
    )
    groups = Counter(
        job["target_provenance"]["opponent_training_group"]
        for job in public_jobs
    )
    h10_jobs = [job for job in public_jobs if job["opponent_id"] == H10_MARNIE]

    assert len(self_jobs) == 1024
    assert len(public_jobs) == 7172
    assert groups == {
        trainer.STRONG_PUBLIC_PRACTICE_GROUP: 4586,
        "diverse_public": 2586,
    }
    assert len(h10_jobs) >= 1024
    assert all(
        job["target_provenance"]["opponent_training_group"]
        == trainer.STRONG_PUBLIC_PRACTICE_GROUP
        for job in h10_jobs
    )
    assert all(
        job["target_provenance"]["strong_public_practice_floor_games"]
        == 1024
        for job in h10_jobs
    )
    assert H10_MARNIE not in {
        job["opponent_id"]
        for job in public_jobs
        if job["target_provenance"]["opponent_training_group"] == "diverse_public"
    }
    # r182 remains default-deny for this ID after its group changes: it must
    # use singleton public transport rather than gaining a new packed admission.
    assert all(not public_multi_env_safety.public_multi_env_safe_job(job) for job in h10_jobs)

    receipt = trainer._assert_strong_public_practice_jobs(
        all_jobs=[*self_jobs, *public_jobs],
        public_jobs=public_jobs,
        active_gate=active_gate,
        expected_practice_games=4586,
        iteration=2,
        root_seed=0,
        formal_games=4500,
        minimum_share=0.04,
        practice_temperature=0.35,
        minimum_games_by_opponent=minimums,
    )
    assert receipt["games"] == 4586
    assert receipt["minimum_games_by_opponent"][H10_MARNIE] == 1024
    assert receipt["per_opponent"][H10_MARNIE]["games"] >= 1024
    assert receipt["seed_disjoint"] is True
