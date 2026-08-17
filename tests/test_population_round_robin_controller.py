from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_population_round_robin import (
    _population_roster,
    _readiness_identity,
    prepare,
)
from scripts.run_population_round_robin import (
    _member_command,
    validate_contract,
)
from scripts.train_pure_rl import _parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_population_controller_contract_is_exact() -> None:
    contract = json.loads(
        (ROOT / "ops/population_round_robin_v1.json").read_text(
            encoding="utf-8"
        )
    )
    validate_contract(contract)
    assert contract["schedule"] == {
        "members": 15,
        "rl_iterations_per_member_cycle": 5,
        "expert_rehearsal_epochs_per_member_cycle": 5,
        "games_per_rl_iteration": 8192,
        "train_epochs_per_rl_iteration": 1,
        "expert_rehearsal_every": 5,
    }
    assert contract["training"]["train_max_decisions_per_batch"] == 2048
    assert contract["paths"]["runtime_root"].endswith(
        "/final-format-marnie-h10-r104"
    )


def test_population_member_command_is_own_models_only() -> None:
    contract = json.loads(
        (ROOT / "ops/population_round_robin_v1.json").read_text(
            encoding="utf-8"
        )
    )
    state = {
        "active_member_index": 0,
        "population_cycle": 3,
        "members": [
            {
                "specialist_id": "starmie",
                "expert_manifest": "/expert/starmie.json",
                "current": {"checkpoint": "/models/starmie-current.pt"},
            }
        ],
    }
    command, run_dir = _member_command(contract, state)
    joined = " ".join(command)
    assert "--population-own-models-only" in command
    assert "--population-opponent-registry" in command
    assert "--official-collect-frac 0" in joined
    assert "--iterations 5" in joined
    assert "--games-per-iter 8192" in joined
    assert "--expert-rehearsal-every 5" in joined
    assert "--expert-rehearsal-epochs 5" in joined
    assert "--resume never" in joined
    assert run_dir.name == "population_starmie_cycle_0003"
    trainer_argv = command[command.index("--") + 1 :]
    parsed = _parse_args(
        [
            "--run-name",
            run_dir.name,
            "--mode",
            "specialist",
            *trainer_argv,
        ]
    )
    assert parsed.population_own_models_only is True
    assert parsed.population_opponent_registry is not None
    assert parsed.iterations == 5
    assert parsed.games_per_iter == 8192


def test_population_service_is_the_declared_terminal_target() -> None:
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert cycle["runtime"]["population_training_service"] == (
        "pokebot-population-round-robin.service"
    )
    unit = (
        ROOT / "deploy/systemd/pokebot-population-round-robin.service"
    ).read_text(encoding="utf-8")
    assert "scripts/run_population_round_robin.py" in unit
    assert "final-format-marnie-h10-r104" in unit
    handoff = (
        ROOT / "deploy/systemd/pokebot-population-round-robin-handoff.service"
    ).read_text(encoding="utf-8")
    assert "post_refresh_sequence_complete_for_capacity_v2.json" in handoff


def test_direct_population_preparation_requires_post_fleet_refreshes(
    tmp_path: Path,
) -> None:
    contract = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract["selection"]["state"] = str(
        ROOT / "state/specialists.yaml"
    )
    contract["runtime"]["lock"] = str(tmp_path / "cycle.lock")
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="post-fleet.*refresh",
    ):
        prepare(path, launch=False)


def test_population_controller_reuses_immutable_cycle_boundary() -> None:
    source = (
        ROOT / "scripts/run_population_round_robin.py"
    ).read_text(encoding="utf-8")
    assert "if boundary_path.is_file()" in source
    assert source.index("fleet = [") < source.index(
        "state = record_completed_member_cycle"
    )


def test_population_readiness_identity_binds_complete_canonical_payload() -> None:
    left = {
        "schema": "poke_bot.population_round_robin_ready/v1",
        "status": "ready",
        "members": [{"specialist_id": "alakazam", "digest": "sha256:a"}],
        "member_count": 1,
    }
    reordered = {
        "member_count": 1,
        "members": [{"digest": "sha256:a", "specialist_id": "alakazam"}],
        "status": "ready",
        "schema": "poke_bot.population_round_robin_ready/v1",
    }
    changed = json.loads(json.dumps(left))
    changed["members"][0]["digest"] = "sha256:b"

    assert _readiness_identity(left) == _readiness_identity(reordered)
    assert _readiness_identity(left) != _readiness_identity(changed)


def test_population_roster_adds_new_h10_crustle_without_public_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.prepare_population_round_robin as module

    historical_ids = [f"member-{index:02d}" for index in range(11)] + [
        "alakazam",
        "marnie-s-grimmsnarl-ex",
        "starmie",
    ]
    state_path = tmp_path / "specialists.yaml"
    state_path.write_text(
        "specialists:\n"
        + "".join(
            f"  - id: {specialist_id}\n"
            "    status: passed_frozen\n"
            "    frozen: true\n"
            for specialist_id in historical_ids
        ),
        encoding="utf-8",
    )
    baseline_root = tmp_path / "baselines"
    frozen_rows = []
    runtime_rows = {}
    for specialist_id in historical_ids:
        package = baseline_root / "specialists" / specialist_id
        package.mkdir(parents=True)
        (package / "model.pt").write_bytes(b"model")
        expert = tmp_path / f"{specialist_id}-expert.json"
        tree = tmp_path / f"{specialist_id}-tree.json"
        expert.write_text("{}", encoding="utf-8")
        tree.write_text("{}", encoding="utf-8")
        frozen_rows.append(
            {
                "specialist_id": specialist_id,
                "opponent_id": f"specialist-{specialist_id}",
                "baseline_group": "specialists",
                "baseline_dir": specialist_id,
                "checkpoint_digest": "sha256:file",
                "content_digest": "sha256:package",
                "frozen": True,
                "public_mix_eligible": True,
            }
        )
        runtime_rows[specialist_id] = {
            "expert_manifest": str(expert),
            "expert_manifest_sha256": "file",
            "matchup_runtime_tree": str(tree),
            "matchup_runtime_tree_sha256": "file",
        }

    frozen_registry = tmp_path / "frozen.json"
    frozen_registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": frozen_rows,
            }
        ),
        encoding="utf-8",
    )
    runtime_registry = tmp_path / "runtime.json"
    runtime_registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_runtime_registry/v1",
                "specialists": runtime_rows,
            }
        ),
        encoding="utf-8",
    )
    refreshes = []
    for specialist_id in ("alakazam", "marnie-s-grimmsnarl-ex", "crustle"):
        expert = tmp_path / f"{specialist_id}-refresh-expert.json"
        tree = tmp_path / f"{specialist_id}-refresh-tree.json"
        bundle = tmp_path / f"{specialist_id}.tar.gz"
        expert.write_text("{}", encoding="utf-8")
        tree.write_text("{}", encoding="utf-8")
        bundle.write_bytes(b"bundle")
        refreshes.append(
            {
                "specialist_id": specialist_id,
                "checkpoint_checksum": "sha256:file",
                "submission_bundle": str(bundle),
                "submission_bundle_sha256": "sha256:file",
                "expert_manifest": str(expert),
                "expert_manifest_sha256": "sha256:file",
                "matchup_runtime_tree": str(tree),
                "matchup_runtime_tree_sha256": "sha256:file",
            }
        )
    refresh_registry = tmp_path / "refresh.json"
    refresh_registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.post_fleet_refresh_registry/v1",
                "ordered_refresh_ids": [
                    "alakazam",
                    "marnie-s-grimmsnarl-ex",
                    "crustle",
                ],
                "refreshes": refreshes,
            }
        ),
        encoding="utf-8",
    )
    baseline_manifest = tmp_path / "manifest.json"
    baseline_manifest.write_text('{"agents": []}', encoding="utf-8")

    monkeypatch.setattr(module, "sha256", lambda _path: "sha256:file")
    monkeypatch.setattr(
        module, "baseline_content_digest", lambda _path: "sha256:package"
    )

    def fake_materialize(**kwargs: object) -> dict[str, str]:
        specialist_id = str(kwargs["specialist_id"])
        package = baseline_root / "population-refresh" / specialist_id
        package.mkdir(parents=True, exist_ok=True)
        model = package / "model.pt"
        model.write_bytes(b"model")
        return {
            "opponent_id": f"refresh-{specialist_id}",
            "baseline_group": "population-refresh",
            "baseline_dir": specialist_id,
            "baseline_package": str(package),
            "checkpoint": str(model),
            "checkpoint_digest": "sha256:file",
            "content_digest": "sha256:package",
        }

    monkeypatch.setattr(module, "materialize_refresh_bundle", fake_materialize)
    roster = _population_roster(
        state_path=state_path,
        frozen_registry_path=frozen_registry,
        runtime_registry_path=runtime_registry,
        baseline_root=baseline_root,
        baseline_manifest=baseline_manifest,
        refresh_registry_path=refresh_registry,
    )

    assert len(roster) == 15
    crustle = next(row for row in roster if row["specialist_id"] == "crustle")
    assert crustle["current_role"] == "current_post_fleet_refresh"
    assert crustle["selected_history"] == []
    assert crustle["trainable_in_population"] is True
    assert all("public" not in row["opponent_id"] for row in roster)
