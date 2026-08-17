from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from poke_bot.dormant_adapter_compat import LOADER_RUNTIME_FILES
from scripts.run_specialist_cycle_handoff import (
    _append_only_v6_registry_upgrade,
    _compatible_prior_cumulative_contract,
    _start_post_fleet_refresh_handoff,
    _cumulative_core_contract,
    _generated,
    _is_expected_additive_gate_successor,
    _required_specialist_ids,
    _source,
    _validated_post_fleet_refresh_progress,
    population_transition_ready,
)
from scripts.run_post_starmie_core_handoff import (
    _resolve_boundary_core,
    _reusable_core_candidate,
    _reusable_core_regression,
)
from scripts.run_starmie_expert_bootstrap import (
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)
from scripts.run_sequential_specialist_handoff import bootstrap_command


ROOT = Path(__file__).resolve().parents[1]


def test_handoff_service_uses_stable_boundary_runtime_pointer() -> None:
    units = (
        ROOT / "deploy/staging/pokebot-specialist-cycle-handoff-task-graph.service",
        ROOT / "ops/systemd/pokebot-specialist-cycle-handoff.service",
    )
    for unit in units:
        source = unit.read_text(encoding="utf-8")
        assert (
            "/home/pokebot/poke-bot-agent-deployments/specialist-handoff-current"
            in source
        )
        assert "pure-rl-resident-v41-specialist-matchup-runtime" not in source
        assert (
            "ExecStart=/usr/bin/env "
            "PYTHONPATH=/home/pokebot/poke-bot-agent-deployments/"
            "specialist-handoff-current "
        ) in source
        assert (
            "Environment=PYTHONPATH=/home/pokebot/poke-bot-agent-deployments/"
            "specialist-handoff-current"
        ) not in source
        assert "Restart=on-failure" in source
        assert "RestartSec=60" in source


def test_transition_graph_pins_imports_to_its_own_runtime_root() -> None:
    source = (
        ROOT / "scripts/run_specialist_transition_graph.py"
    ).read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source
    assert "implementation.is_relative_to(ROOT)" in source
    assert "specialist handoff implementation escaped" in source


def test_cycle_contract_keeps_exact_protocol_for_later_specialists() -> None:
    contract = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = {
        "id": "lucario",
        "run_dir": "/runs/lucario",
        "training_service": "pokebot-pure-rl-trevenant-staged.service",
        "gate_contract": "/runtime/ops/gate.json",
        "gate_marker_name": "SPECIALIST_GATE_PASSED.lucario-splus-v1",
        "minimum_completed_iteration": 5,
        "matchup_runtime_tree": "/state/tree.json",
        "passed_family": "/models/lucario",
        "handler_state": "/state/lucario-handler.json",
    }
    result = _generated(
        contract=contract,
        source=source,
        selected={
            "specialist_id": "dragapult",
            "pointer": "/corpora/dragapult/PROTECTED_EXPERT_CORPUS.json",
        },
        core_digest="sha256:" + "b" * 64,
    )

    assert result["source_specialist"] == source
    assert result["next_specialist"]["id"] == "dragapult"
    assert result["training"]["supervised_epochs"] == 25
    assert result["training"]["minimum_decisions"] == 20000
    expanded = result["training"]["expanded_heads"]
    assert expanded["schema"] == "poke_bot.expanded_head_training/v1"
    assert expanded["schedule"]["total_epochs"] == 25
    assert expanded["runtime_enabled_heads"] == []
    assert expanded["schedule_digest"].startswith("sha256:")
    assert expanded["target_schema_digest"].startswith("sha256:")
    fusion = result["training"]["decision_fusion"]
    assert fusion == decision_fusion_handoff_contract()
    assert fusion["head_count"] == 17
    command = bootstrap_command(result)
    assert "--decision-fusion" in command
    assert command.index("--decision-fusion") == command.index("--expanded-heads") + 1
    assert result["submission_policy"]["completion_blocks_handoff"] is False
    assert result["runtime_registration"]["handoff_service"] == (
        "pokebot-specialist-cycle-handoff.service"
    )
    assert result["runtime_registration"]["gate_handler_service"] == (
        "pokebot-specialist-passed-gate-handler.service"
    )
    policy = result["runtime_registration"]["future_guide_weight_policy"]
    assert policy["scope"] == "future_specialist_training_runs_only"
    assert policy["prospective_scope_revision"] == 44
    assert policy["learning_semantics_revision"] == 46
    assert policy["prospective_effective_specialist"] == "archaludon-ex"
    assert policy[
        "retroactive_application_to_completed_frozen_or_started_runs"
    ] is False
    assert policy["historical_weight_or_receipt_rewrite_allowed"] is False
    assert policy["files"]["scripts/train_pure_rl.py"].startswith("sha256:")
    fleet = result["runtime_registration"]["matchup_v6"]["fleet"]
    assert fleet["bert"]["expected_workers"] == 16
    assert fleet["bert"]["expected_leaves"] == 4
    assert fleet["elmo"]["expected_workers"] == 36
    assert fleet["elmo"]["expected_leaves"] == 4
    assert fleet["elmo"]["image"] == (
        "poke-bot-truenas-worker:matchup-v41-v6-slowking-runtime-r74-memory-v1"
    )
    assert result["paths"]["state"].endswith(
        "post-lucario-dragapult-handoff-v1.json"
    )


def test_cycle_contract_has_explicit_population_terminal_handoff() -> None:
    contract = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["runtime"]["population_handoff_service"] == (
        "pokebot-population-round-robin-handoff.service"
    )
    assert contract["runtime"]["post_fleet_refresh_handoff_service"] == (
        "pokebot-final-format-alakazam-handoff.service"
    )
    assert contract["runtime"]["population_training_service"] == (
        "pokebot-population-round-robin.service"
    )
    assert contract["selection"]["corpus_root"].endswith(
        "data/bootstrap/current-specialist-latest20-v6-strategic"
    )
    assert contract["selection"]["corpus_source_receipt"].endswith(
        "outputs/state/expert-latest20-v6-strategic-sync.json"
    )
    assert contract["selection"]["core_corpus"].endswith(
        "current-specialist-latest20-v6-strategic/"
        "core-balanced-v6/PROTECTED_CORE_CORPUS.json"
    )
    assert contract["prestage"]["required_expanded_target_schema"] == (
        "poke_bot.expanded_strategic_targets/v2"
    )
    assert contract["prestage"]["required_expanded_target_digest"].startswith(
        "sha256:"
    )
    assert contract["runtime"]["inactive_tree_candidate"].endswith(
        "public-matchup-tree-calibration-roster20-v45.inactive.json"
    )
    assert contract["runtime"]["candidate_audit"].endswith(
        "public-matchup-tree-calibration-roster20-v45.audit.json"
    )
    assert contract["runtime"]["future_assets_receipt"].endswith(
        "slowking_router_promotion_v1.json"
    )
    assert contract["runtime"]["future_assets_scope"] == "router_only"
    assert contract["selection"]["minimum_decisions_by_specialist"] == {
        "dragapult-dusknoir": 10000,
        "dudunsparce": 2000,
        "slowking": 19000,
        "team-rockets-spidops": 20000,
    }
    assert contract["selection"]["minimum_records_by_specialist"] == {
        "team-rockets-spidops": 16639,
    }
    assert contract["selection"]["strict_priority_prefix"] == [
        "dragapult-dusknoir",
        "dudunsparce",
        "marnie-s-grimmsnarl-ex",
        "garchomp",
        "rockets-mewtwo",
        "thwackey",
        "team-rockets-spidops",
        "hammer-pult",
        "teal-mask-ogerpon-ex",
        "archaludon-ex",
    ]


def test_post_fleet_refresh_handoff_starts_only_canonical_alakazam_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        argv: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert check is False
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.subprocess.run",
        fake_run,
    )
    runtime = {
        "post_fleet_refresh_handoff_service": (
            "pokebot-final-format-alakazam-handoff.service"
        )
    }
    _start_post_fleet_refresh_handoff(runtime, specialist_id="alakazam")
    assert calls == [
        [
            "/usr/bin/systemctl",
            "--user",
            "start",
            "pokebot-final-format-alakazam-handoff.service",
        ]
    ]

    with pytest.raises(RuntimeError, match="must begin with alakazam"):
        _start_post_fleet_refresh_handoff(
            runtime,
            specialist_id="marnie-s-grimmsnarl-ex",
        )


def test_exact_additive_gate_successor_is_resume_safe() -> None:
    checkpoint = "sha256:" + "a" * 64
    saved = {
        "gate_id": "strong+frozen-specialists-r5",
        "checkpoint_digest": checkpoint,
        "roster_ids": ["public-a", "specialist-starmie"],
    }
    current = {
        "active_gate_id": "strong+frozen-specialists-r6",
        "next_gate": {
            "id": "strong+frozen-specialists-r6",
            "roster": [
                {"opponent_id": "public-a"},
                {"opponent_id": "specialist-starmie"},
                {
                    "opponent_id": "specialist-dudunsparce",
                    "archetype_id": "dudunsparce",
                    "frozen_specialist": True,
                    "frozen_checkpoint_digest": checkpoint,
                },
            ],
        },
    }
    registry = {
        "specialists": [
            {
                "specialist_id": "dudunsparce",
                "checkpoint_digest": checkpoint,
                "frozen": True,
            }
        ]
    }

    assert _is_expected_additive_gate_successor(
        active_id="dudunsparce",
        saved_gate=saved,
        current_gate=current,
        frozen_registry=registry,
    )
    saved["base_gate_id"] = "alakazam-strong+frozen-specialists-r5"
    saved["gate_id"] = "specialist-lc50+frozen-specialists-r5"
    current["active_gate_id"] = "alakazam-strong+frozen-specialists-r6"
    current["next_gate"]["id"] = current["active_gate_id"]
    assert _is_expected_additive_gate_successor(
        active_id="dudunsparce",
        saved_gate=saved,
        current_gate=current,
        frozen_registry=registry,
    )
    current["next_gate"]["roster"].append({"opponent_id": "unexpected"})
    assert not _is_expected_additive_gate_successor(
        active_id="dudunsparce",
        saved_gate=saved,
        current_gate=current,
        frozen_registry=registry,
    )


def test_ceiling_gate_resume_accepts_exact_materialization_receipt_across_namespaces() -> None:
    checkpoint = "sha256:" + "a" * 64
    gate_digest = "sha256:" + "b" * 64
    saved = {
        "base_gate_id": (
            "specialist-strong-public-roster-lc50-at-iter5-v1"
            "+frozen-specialists-r11"
        ),
        "checkpoint_digest": checkpoint,
        "roster_ids": ["public-a", "specialist-starmie"],
    }
    current = {
        "active_gate_id": (
            "alakazam-strong-public-roster-lc55-v2"
            "+frozen-specialists-r12"
        ),
        "next_gate": {
            "id": (
                "alakazam-strong-public-roster-lc55-v2"
                "+frozen-specialists-r12"
            ),
            "roster": [
                {"opponent_id": "public-a"},
                {"opponent_id": "specialist-starmie"},
                {
                    "opponent_id": "specialist-hammer-pult",
                    "archetype_id": "hammer-pult",
                    "frozen_specialist": True,
                    "frozen_checkpoint_digest": checkpoint,
                },
            ],
        },
    }
    registry = {
        "specialists": [
            {
                "specialist_id": "hammer-pult",
                "checkpoint_digest": checkpoint,
                "frozen": True,
            }
        ]
    }
    receipt = {
        "schema": "poke_bot.frozen_specialist_gate_materialization/v1",
        "specialist_id": "hammer-pult",
        "checkpoint_digest": checkpoint,
        "gate_id": current["active_gate_id"],
        "gate_contract_sha256": gate_digest,
        "opponent_id": "specialist-hammer-pult",
        "frozen_specialist_ids": ["starmie", "hammer-pult"],
    }

    assert _is_expected_additive_gate_successor(
        active_id="hammer-pult",
        saved_gate=saved,
        current_gate=current,
        frozen_registry=registry,
        materialization_receipt=receipt,
        current_gate_sha256=gate_digest,
    )
    receipt["checkpoint_digest"] = "sha256:" + "c" * 64
    assert not _is_expected_additive_gate_successor(
        active_id="hammer-pult",
        saved_gate=saved,
        current_gate=current,
        frozen_registry=registry,
        materialization_receipt=receipt,
        current_gate_sha256=gate_digest,
    )


def test_owner_ceiling_resume_accepts_exact_additive_successor_namespace() -> None:
    checkpoint = "sha256:" + "a" * 64
    saved = {
        "base_gate_id": "specialist-lc50+frozen-specialists-r12",
        "checkpoint_digest": checkpoint,
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "roster_ids": ["public-a"],
    }
    current = {
        "active_gate_id": "alakazam-lc55+frozen-specialists-r13",
        "next_gate": {
            "id": "alakazam-lc55+frozen-specialists-r13",
            "roster": [
                {"opponent_id": "public-a"},
                {
                    "opponent_id": "specialist-teal",
                    "archetype_id": "teal",
                    "frozen_specialist": True,
                    "frozen_checkpoint_digest": checkpoint,
                },
            ],
        },
    }
    registry = {
        "specialists": [
            {
                "specialist_id": "teal",
                "checkpoint_digest": checkpoint,
                "frozen": True,
            }
        ]
    }

    assert _is_expected_additive_gate_successor(
        active_id="teal",
        saved_gate=saved,
        current_gate=current,
        frozen_registry=registry,
    )


def test_normal_gate_resume_uses_frozen_evidence_after_additive_successor(
    tmp_path: Path, monkeypatch
) -> None:
    digest = "sha256:" + "a" * 64
    frozen = tmp_path / "models" / "garchomp-pass"
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text("{}\n", encoding="utf-8")
    current_gate = tmp_path / "gate.json"
    current_gate.write_text('{"active_gate_id":"successor"}\n', encoding="utf-8")
    handler_state = tmp_path / "handler.json"
    frozen_identity = {
        "family": "garchomp-pass",
        "model_path": str(frozen / "model.pt"),
        "checkpoint_digest": digest,
    }
    handler_state.write_text(
        json.dumps(
            {
                "schema": "poke_bot.passed_gate_handler/v1",
                "phase": "complete_handoff_started",
                "submission_mode": "queue_and_continue",
                "gate": {
                    "contract": str(current_gate),
                    "contract_sha256": "sha256:historical",
                    "checkpoint_digest": digest,
                    "commit_boundary": 5,
                    "validation": {"exact_gate": True},
                },
                "frozen_model": frozen_identity,
                "queued_submissions": [
                    {
                        "copy_number": 1,
                        "label": "garchomp",
                        "checkpoint_checksum": digest,
                        "queued_at": "2026-07-27T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_registry = tmp_path / "runtime.json"
    runtime_registry.write_text(
        json.dumps(
            {
                "runtime_root": str(tmp_path),
                "active_gate_contract": "gate.json",
                "minimum_terminal_iteration": 5,
                "specialists": {
                    "garchomp": {
                        "status": "ready",
                        "run_name": "garchomp-run",
                        "terminal_gate_marker": "PASSED",
                        "matchup_runtime_tree": str(tmp_path / "tree.json"),
                        "pass_handler": {
                            "family": frozen.name,
                            "state": str(handler_state),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    frozen_registry = tmp_path / "frozen.json"
    frozen_registry.write_text('{"specialists":[]}\n', encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.verify_frozen_model",
        lambda path: frozen_identity,
    )
    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff._is_expected_additive_gate_successor",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.validate_source",
        lambda contract: (_ for _ in ()).throw(
            AssertionError("historical result must not be reinterpreted")
        ),
    )

    source, evidence = _source(
        contract={
            "runtime": {
                "runtime_registry": str(runtime_registry),
                "frozen_specialist_registry": str(frozen_registry),
                "registry_root": str(frozen.parent),
                "training_service": "trainer.service",
            }
        },
        active_id="garchomp",
    )

    assert source["id"] == "garchomp"
    assert evidence["checkpoint_digest"] == digest
    assert evidence["queued_submission_copies"][0]["copy_number"] == 1


def test_saved_source_accepts_two_explicitly_approved_copies(
    tmp_path: Path, monkeypatch
) -> None:
    frozen = tmp_path / "models" / "source"
    frozen.mkdir(parents=True)
    digest = "sha256:" + "4" * 64
    (frozen / "manifest.json").write_text("{}\n", encoding="utf-8")
    gate_contract = tmp_path / "gate.json"
    gate_contract.write_text("{}\n", encoding="utf-8")
    frozen_identity = {
        "family": "source",
        "model_path": str(frozen / "model.pt"),
        "checkpoint_digest": digest,
    }
    handler_state = tmp_path / "handler.json"
    handler_state.write_text(
        json.dumps(
            {
                "schema": "poke_bot.passed_gate_handler/v1",
                "phase": "complete_handoff_started",
                "submission_mode": "queue_and_continue",
                "approved_submission_count": 2,
                "gate": {
                    "commit_boundary": 14,
                    "checkpoint_digest": digest,
                    "contract": str(gate_contract),
                    "validation": {"committed": True},
                },
                "frozen_model": frozen_identity,
                "queued_submissions": [
                    {
                        "copy_number": copy_number,
                        "label": f"source copy {copy_number}",
                        "checkpoint_checksum": digest,
                        "queued_at": "2026-07-30T00:00:00+00:00",
                    }
                    for copy_number in (1, 2)
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_registry = tmp_path / "runtime.json"
    runtime_registry.write_text(
        json.dumps(
                {
                    "schema": "poke_bot.specialist_runtime_registry/v1",
                    "minimum_terminal_iteration": 14,
                    "specialists": {
                    "source": {
                        "status": "ready",
                        "run_name": "source-run",
                        "minimum_terminal_iteration": 14,
                        "terminal_gate_marker": "PASSED",
                        "matchup_runtime_tree": str(tmp_path / "tree.json"),
                        "pass_handler": {
                            "family": frozen.name,
                            "state": str(handler_state),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    frozen_registry = tmp_path / "frozen.json"
    frozen_registry.write_text('{"specialists":[]}\n', encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.verify_frozen_model",
        lambda path: frozen_identity,
    )
    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff._is_expected_additive_gate_successor",
        lambda **kwargs: True,
    )

    _, evidence = _source(
        contract={
            "runtime": {
                "runtime_registry": str(runtime_registry),
                "frozen_specialist_registry": str(frozen_registry),
                "registry_root": str(frozen.parent),
                "training_service": "trainer.service",
            }
        },
        active_id="source",
    )

    assert [
        row["copy_number"] for row in evidence["queued_submission_copies"]
    ] == [1, 2]


def test_lucario_runtime_row_exposes_exact_threshold_transition() -> None:
    registry = json.loads(
        (ROOT / "ops/specialist_runtime_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    lucario = registry["specialists"]["lucario"]

    assert lucario["status"] == "ready"
    assert lucario["minimum_terminal_iteration"] == 10
    assert lucario["pass_handler"]["threshold_transition_receipt"] == (
        "/home/pokebot/poke-bot-agent/outputs/state/"
        "lucario-gate-floor15-transition-v1.json"
    )


def test_accepted_core_is_reused_without_rematerialization(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "models" / "core-v4"
    family.mkdir(parents=True)
    ready_path = tmp_path / "core-v4-ready.json"
    digest = "sha256:" + "a" * 64
    teacher_digests = ["sha256:" + "b" * 64, "sha256:" + "c" * 64]
    ready_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.multi_teacher_core_ready/v1",
                "status": "ready",
                "gameplay_regression_passed": True,
                "checkpoint_digest": digest,
                "teacher_checkpoint_digests": teacher_digests,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.run_post_starmie_core_handoff.verify_frozen_model",
        lambda path: {
            "family": Path(path).name,
            "model_path": str(Path(path) / "model.pt"),
            "checkpoint_digest": digest,
        },
    )
    result = _reusable_core_candidate(
        contract={
            "core_refresh": {
                "family": str(family),
                "ready_receipt": str(ready_path),
                "teachers": [
                    {"checksum": teacher_digests[0]},
                    {"checksum": teacher_digests[1]},
                ],
            }
        }
    )
    assert result is not None
    assert result[0]["status"] == "ready"
    assert result[1]["checkpoint_digest"] == digest


def test_expanded_core_reuse_requires_complete_shadow_only_schedule(
    tmp_path: Path, monkeypatch
) -> None:
    family = tmp_path / "models" / "core-v6"
    family.mkdir(parents=True)
    ready_path = tmp_path / "core-v6-ready.json"
    digest = "sha256:" + "a" * 64
    teacher_digests = ["sha256:" + "b" * 64, "sha256:" + "c" * 64]
    expanded = expanded_handoff_training_contract()
    expected_heads = expanded["schedule"]["stages"][-1]["enabled_heads"]
    payload = {
        "schema": "poke_bot.multi_teacher_core_ready/v1",
        "status": "candidate_ready_for_gameplay_regression",
        "checkpoint_digest": digest,
        "teacher_checkpoint_digests": teacher_digests,
        "expanded_target_schema_digest": expanded[
            "target_schema_digest"
        ],
        "expanded_schedule_digest": expanded["schedule_digest"],
        "expanded_heads_trained": expected_heads,
        "runtime_enabled_heads": [],
        "epochs_completed": 25,
    }
    ready_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_post_starmie_core_handoff.verify_frozen_model",
        lambda path: {
            "family": Path(path).name,
            "model_path": str(Path(path) / "model.pt"),
            "checkpoint_digest": digest,
        },
    )
    contract = {
        "core_refresh": {
            "family": str(family),
            "ready_receipt": str(ready_path),
            "teachers": [
                {"checksum": teacher_digests[0]},
                {"checksum": teacher_digests[1]},
            ],
            "expanded_heads": expanded,
        }
    }
    assert _reusable_core_candidate(contract=contract) is not None

    payload["runtime_enabled_heads"] = ["action_q"]
    ready_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="readiness identity changed"
    ):
        _reusable_core_candidate(contract=contract)


def test_prior_cumulative_contract_accepts_only_additive_selection_controls() -> None:
    current = {
        "trigger": {
            "specialist_id": "lucario",
            "threshold_transition_receipt": "/state/transition.json",
        },
        "next_specialist": {
            "minimum_decisions": 20_000,
            "minimum_decisions_by_specialist": {
                "dragapult-dusknoir": 10_000
            },
            "minimum_records_by_specialist": {
                "team-rockets-spidops": 16_639
            },
            "strict_priority_prefix": ["dragapult-dusknoir"],
            "current_deck_guide_required": True,
        },
        "runtime": {
            "inactive_tree_candidate": "/state/tree-v37.json",
            "candidate_audit": "/state/tree-v37-audit.json",
            "future_assets_receipt": "/state/rare-route-assets-v38-ready.json",
            "future_assets_scope": "router_only",
        },
    }
    prior = json.loads(json.dumps(current))
    prior["trigger"].pop("threshold_transition_receipt")
    prior["next_specialist"].pop("minimum_decisions_by_specialist")
    prior["next_specialist"].pop("minimum_records_by_specialist")
    prior["next_specialist"].pop("strict_priority_prefix")
    prior["next_specialist"].pop("current_deck_guide_required")
    prior["runtime"]["inactive_tree_candidate"] = None
    prior["runtime"]["candidate_audit"] = None
    prior["runtime"]["future_assets_receipt"] = None
    prior["runtime"].pop("future_assets_scope")
    assert _compatible_prior_cumulative_contract(prior, current)
    prior_with_assets = json.loads(json.dumps(current))
    prior_with_assets["runtime"].pop("future_assets_scope")
    assert _compatible_prior_cumulative_contract(prior_with_assets, current)
    prior_asset_generation = json.loads(json.dumps(current))
    prior_asset_generation["runtime"].update(
        inactive_tree_candidate="/state/tree-v38.json",
        candidate_audit="/state/tree-v38-audit.json",
        future_assets_receipt="/state/rare-route-assets-v38-ready.json",
    )
    assert _compatible_prior_cumulative_contract(
        prior_asset_generation, current
    )
    prior["next_specialist"]["minimum_decisions"] = 9_999
    assert not _compatible_prior_cumulative_contract(prior, current)


def test_prior_cumulative_contract_accepts_only_paired_protocol_checksum_refresh() -> None:
    old_digest = "sha256:" + "1" * 64
    new_digest = "sha256:" + "2" * 64
    current = {
        "trigger": {"specialist_id": "team-rockets-spidops"},
        "next_specialist": {"minimum_decisions": 20_000},
        "runtime": {},
        "core_refresh": {
            "decision_fusion": {
                "canonical_config_sha256": new_digest,
                "head_count": 17,
            },
            "expanded_heads": {
                "canonical_config_sha256": new_digest,
                "head_count": 11,
            },
        },
    }
    prior = json.loads(json.dumps(current))
    prior["core_refresh"]["decision_fusion"][
        "canonical_config_sha256"
    ] = old_digest
    prior["core_refresh"]["expanded_heads"][
        "canonical_config_sha256"
    ] = old_digest

    assert _compatible_prior_cumulative_contract(prior, current)

    prior["core_refresh"]["decision_fusion"]["head_count"] = 16
    assert not _compatible_prior_cumulative_contract(prior, current)

    prior = json.loads(json.dumps(current))
    prior["core_refresh"]["decision_fusion"][
        "canonical_config_sha256"
    ] = old_digest
    assert not _compatible_prior_cumulative_contract(prior, current)


def test_prior_cumulative_contract_accepts_versioned_v6_fleet_receipt(
    tmp_path: Path,
) -> None:
    old_receipt = tmp_path / "matchup-v6-fleet-v1.json"
    old_receipt.write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_adapter_v6_fleet_activation/v1",
                "status": "active",
            }
        ),
        encoding="utf-8",
    )
    new_receipt = tmp_path / "matchup-v6-fleet-v2.json"
    current = {
        "trigger": {"specialist_id": "hammer-pult"},
        "next_specialist": {"minimum_decisions": 20_000},
        "runtime": {
            "matchup_v6": {
                "enabled": True,
                "fleet": {
                    "receipt": str(new_receipt),
                    "elmo": {
                        "endpoint": "elmo:8765",
                        "image": "poke-bot-truenas-worker:v39",
                        "build_context": "/srv/poke-bot",
                        "dockerfile": "/srv/poke-bot/Dockerfile",
                    },
                },
            }
        },
        "core_refresh": {},
    }
    prior = json.loads(json.dumps(current))
    prior["runtime"]["matchup_v6"]["fleet"]["receipt"] = str(old_receipt)
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"]["image"] = (
        "poke-bot-truenas-worker:v38"
    )
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"].pop("build_context")
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"].pop("dockerfile")

    assert _compatible_prior_cumulative_contract(prior, current)
    old_receipt.write_text('{"schema":"wrong","status":"active"}\n')
    assert not _compatible_prior_cumulative_contract(prior, current)
    prior = json.loads(json.dumps(current))
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"]["image"] = (
        "poke-bot-truenas-worker:v38"
    )
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"].pop("build_context")
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"].pop("dockerfile")
    assert not new_receipt.exists()
    assert _compatible_prior_cumulative_contract(prior, current)

    old_root = tmp_path / "loader-v1"
    new_root = tmp_path / "loader-v2"
    old_registry = old_root / "state/matchup_adapter_roster.json"
    new_registry = new_root / "state/matchup_adapter_roster.json"
    old_registry.parent.mkdir(parents=True)
    new_registry.parent.mkdir(parents=True)
    old_registry.write_text('{"slots":[1,2,3]}\n', encoding="utf-8")
    new_registry.write_text(old_registry.read_text(encoding="utf-8"), encoding="utf-8")
    for relative in LOADER_RUNTIME_FILES:
        loader_file = new_root / relative
        loader_file.parent.mkdir(parents=True, exist_ok=True)
        loader_file.write_text(f"# {relative}\n", encoding="utf-8")

    current["runtime"]["matchup_v6"]["fleet"]["source_root"] = str(new_root)
    current["runtime"]["matchup_v6"]["fleet"]["registry"] = str(new_registry)
    current["runtime"]["matchup_v6"]["registry"] = str(new_registry)
    prior = json.loads(json.dumps(current))
    prior["runtime"]["matchup_v6"]["fleet"]["source_root"] = str(old_root)
    prior["runtime"]["matchup_v6"]["fleet"]["registry"] = str(old_registry)
    prior["runtime"]["matchup_v6"]["registry"] = str(old_registry)
    prior["runtime"]["matchup_v6"]["fleet"]["receipt"] = str(old_receipt)
    prior["runtime"]["matchup_v6"]["fleet"]["elmo"]["image"] = (
        "poke-bot-truenas-worker:v38"
    )
    old_receipt.write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_adapter_v6_fleet_activation/v1",
                "status": "active",
            }
        ),
        encoding="utf-8",
    )
    assert _compatible_prior_cumulative_contract(prior, current)

    new_registry.write_text('{"slots":[1,2,4]}\n', encoding="utf-8")
    assert not _compatible_prior_cumulative_contract(prior, current)
    new_registry.write_text(old_registry.read_text(encoding="utf-8"), encoding="utf-8")
    (new_root / LOADER_RUNTIME_FILES[-1]).unlink()
    assert not _compatible_prior_cumulative_contract(prior, current)


def test_append_only_v6_registry_upgrade_rejects_existing_slot_changes(
    tmp_path: Path,
) -> None:
    source = json.loads(
        (ROOT / "state/matchup_adapter_roster.json").read_text(
            encoding="utf-8"
        )
    )
    source["revision"] -= 1
    source["required_specialist_count"] -= 1
    source["active_expert_ids"].remove("slowking")
    source["expert_ids"].remove("slowking")
    source["specialist_priority"].remove("slowking")
    source["canonical_display_names"].pop("slowking")
    source["meta_analysis_source"]["crosswalk"].pop("slowking")
    source["slots"][19] = {
        "slot": 19,
        "archetype_id": None,
        "status": "unused",
        "lineage": None,
    }
    old_registry = tmp_path / "old.json"
    new_registry = tmp_path / "new.json"
    old_registry.write_text(json.dumps(source), encoding="utf-8")
    new_registry.write_text(
        (ROOT / "state/matchup_adapter_roster.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    assert _append_only_v6_registry_upgrade(old_registry, new_registry)

    changed = json.loads(new_registry.read_text(encoding="utf-8"))
    changed["slots"][0]["lineage"] = "rewritten"
    new_registry.write_text(json.dumps(changed), encoding="utf-8")
    assert not _append_only_v6_registry_upgrade(old_registry, new_registry)


def test_accepted_core_regression_survives_controller_only_change(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "d" * 64
    teachers = ["sha256:" + "e" * 64]
    path = tmp_path / "regression.json"
    path.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.multi_teacher_core_gameplay_regression/v1"
                ),
                "passed": True,
                "training_eligible": False,
                "identity": {
                    "contract_digest": "sha256:" + "0" * 64,
                    "candidate": {"digest": digest},
                    "teacher_checkpoint_digests": teachers,
                },
            }
        ),
        encoding="utf-8",
    )
    result = _reusable_core_regression(
        path=path,
        candidate_digest=digest,
        teacher_digests=teachers,
    )
    assert result is not None
    assert result["passed"] is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    rejected = _reusable_core_regression(
        path=path,
        candidate_digest=digest,
        teacher_digests=teachers,
    )
    assert rejected is not None
    assert rejected["passed"] is False


def test_each_future_specialist_gets_a_new_cumulative_core(
    tmp_path: Path, monkeypatch
) -> None:
    template = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cycle["runtime"]["registry_root"] = str(tmp_path / "models")
    cycle["runtime"]["state_root"] = str(tmp_path / "state")
    cycle["selection"]["state"] = str(tmp_path / "specialists.yaml")
    cycle["selection"]["corpus_root"] = str(tmp_path / "corpora")
    core_corpus = tmp_path / "core" / "PROTECTED_CORE_CORPUS.json"
    core_corpus.parent.mkdir()
    core_corpus.write_text("{}\n", encoding="utf-8")
    cycle["selection"]["core_corpus"] = str(core_corpus)
    specialists = {
        specialist_id: {
            "pass_handler": {"family": f"{specialist_id}-pass-v1"}
        }
        for specialist_id in ("alakazam", "hops-trevenant", "starmie", "lucario")
    }
    specialists["alakazam"] = {}

    def frozen(path: Path) -> dict[str, str]:
        family = Path(path).name
        return {
            "family": family,
            "model_path": str(Path(path) / "model.pt"),
            "checkpoint_digest": "sha256:" + family.encode().hex().ljust(64, "0")[:64],
        }

    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.verify_frozen_model", frozen
    )
    result = _cumulative_core_contract(
        template=template,
        cycle=cycle,
        source={
            "id": "lucario",
            "run_dir": "/runs/lucario",
            "training_service": "trainer.service",
            "gate_contract": "/runtime/gate.json",
            "gate_marker_name": "SPECIALIST_GATE_PASSED.lucario",
            "minimum_completed_iteration": 5,
            "passed_family": "/models/lucario-pass-v1",
            "handler_state": "/state/lucario-handler.json",
        },
        completed_ids={"alakazam", "hops-trevenant", "starmie", "lucario"},
        runtime_registry={"specialists": specialists},
        current_core={
            "model_path": "/models/deck-core-v2/model.pt",
            "checkpoint_digest": "sha256:" + "c" * 64,
            "ready": "/state/deck-core-v2-ready.json",
            "version": 2,
        },
    )
    assert result["core_refresh"]["version"] == 4
    assert result["core_refresh"]["refresh_attempt"] == 1
    assert result["core_refresh"]["split_seed"] == 20260723
    assert [
        row["specialist_id"] for row in result["core_refresh"]["teachers"]
    ] == ["alakazam", "hops-trevenant", "lucario", "starmie"]
    assert result["core_refresh"]["direct_checkpoint_tensor_sources_exclude"] == []
    assert result["core_refresh"]["initialization"]["checkpoint"] == (
        "/models/deck-core-v2/model.pt"
    )
    assert result["next_specialist"]["hot_start_core_version"] == 4
    assert result["trigger"]["minimum_completed_iteration"] == 5
    assert result["core_failure_fallback"] == {
        "enabled": True,
        "behavior": "continue_with_latest_accepted_core",
        "continue_refresh_after_each_specialist": True,
        "version": 2,
        "family": "/models/deck-core-v2",
        "checkpoint_digest": "sha256:" + "c" * 64,
        "ready_receipt": "/state/deck-core-v2-ready.json",
        "owner_decision": "GOAL.md#/decision-ledger/revision-19",
    }


def test_failed_core_regression_reuses_boundary_and_falls_back(
    tmp_path: Path, monkeypatch
) -> None:
    template = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cycle = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cycle["runtime"]["registry_root"] = str(tmp_path / "models")
    cycle["runtime"]["state_root"] = str(tmp_path / "state")
    cycle["selection"]["state"] = str(tmp_path / "specialists.yaml")
    cycle["selection"]["corpus_root"] = str(tmp_path / "corpora")
    core_corpus = tmp_path / "core" / "PROTECTED_CORE_CORPUS.json"
    core_corpus.parent.mkdir()
    core_corpus.write_text("{}\n", encoding="utf-8")
    cycle["selection"]["core_corpus"] = str(core_corpus)
    specialists = {
        specialist_id: {
            "pass_handler": {"family": f"{specialist_id}-pass-v1"}
        }
        for specialist_id in ("alakazam", "hops-trevenant")
    }

    def frozen(path: Path) -> dict[str, str]:
        family = Path(path).name
        return {
            "family": family,
            "model_path": str(Path(path) / "model.pt"),
            "checkpoint_digest": (
                "sha256:" + family.encode().hex().ljust(64, "0")[:64]
            ),
        }

    monkeypatch.setattr(
        "scripts.run_specialist_cycle_handoff.verify_frozen_model", frozen
    )
    source = {
        "id": "hops-trevenant",
        "run_dir": "/runs/hops",
        "training_service": "trainer.service",
        "gate_contract": "/runtime/gate.json",
        "gate_marker_name": "SPECIALIST_GATE_PASSED.hops",
        "minimum_completed_iteration": 5,
        "passed_family": "/models/hops-pass-v1",
        "handler_state": "/state/hops-handler.json",
    }
    kwargs = {
        "template": template,
        "cycle": cycle,
        "source": source,
        "completed_ids": {"alakazam", "hops-trevenant"},
        "runtime_registry": {"specialists": specialists},
        "current_core": {
            "model_path": "/models/deck-core-v1/model.pt",
            "checkpoint_digest": "sha256:" + "c" * 64,
            "ready": "/state/deck-core-v1-ready.json",
            "version": 1,
        },
    }
    first = _cumulative_core_contract(**kwargs)
    teacher_digests = [
        row["checksum"] for row in first["core_refresh"]["teachers"]
    ]
    failed = tmp_path / (
        "state/deck-agnostic-core-cumulative-v2-fused-v1-"
        "gameplay-regression.json"
    )
    failed.parent.mkdir(exist_ok=True)
    failed.write_text(
        json.dumps(
            {
                "schema": "poke_bot.multi_teacher_core_gameplay_regression/v1",
                "passed": False,
                "training_eligible": False,
                "identity": {
                    "teacher_checkpoint_digests": teacher_digests,
                },
                "criteria": {"all_reports_valid": True},
            }
        ),
        encoding="utf-8",
    )
    resumed = _cumulative_core_contract(**kwargs)
    assert resumed["core_refresh"]["refresh_attempt"] == 1
    assert resumed["core_refresh"]["split_seed"] == 20260723
    assert resumed["core_refresh"]["family"] == first["core_refresh"]["family"]
    assert (
        resumed["acceptance"]["regression_result"]
        == first["acceptance"]["regression_result"]
    )
    assert resumed["core_failure_fallback"]["enabled"] is True
    assert resumed["core_failure_fallback"]["version"] == 1


def test_pretraining_core_rejection_is_immutable_and_nonblocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback_digest = "sha256:" + "9" * 64
    fallback_family = tmp_path / "models/core-v9"
    fallback_family.mkdir(parents=True)
    fallback_ready = tmp_path / "state/core-v9-ready.json"
    fallback_ready.parent.mkdir()
    fallback_ready.write_text(
        json.dumps(
            {
                "schema": "poke_bot.multi_teacher_core_ready/v1",
                "status": "ready",
                "gameplay_regression_passed": True,
                "checkpoint_digest": fallback_digest,
            }
        ),
        encoding="utf-8",
    )
    attempted_ready = tmp_path / "state/core-v11-ready.json"
    contract_path = tmp_path / "post-spidops-core-v11.json"
    contract = {
        "core_refresh": {
            "version": 11,
            "family": str(tmp_path / "models/core-v11"),
            "ready_receipt": str(attempted_ready),
            "initialization": {
                "checkpoint": str(fallback_family / "model.pt"),
                "checksum": fallback_digest,
            },
            "teachers": [
                {"checksum": "sha256:" + "1" * 64},
                {"checksum": "sha256:" + "2" * 64},
            ],
        },
        "core_failure_fallback": {
            "enabled": True,
            "behavior": "continue_with_latest_accepted_core",
            "continue_refresh_after_each_specialist": True,
            "version": 9,
            "family": str(fallback_family),
            "checkpoint_digest": fallback_digest,
            "ready_receipt": str(fallback_ready),
        },
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    state_path = tmp_path / "state/handoff.json"
    calls = {"command": 0}

    def fail_command(_command: list[str]) -> None:
        calls["command"] += 1
        raise RuntimeError(
            "checkpoint auxiliary row identity is unavailable or ambiguous"
        )

    monkeypatch.setattr(
        "scripts.run_post_starmie_core_handoff._core_refresh_command",
        lambda **_kwargs: ["refresh"],
    )
    monkeypatch.setattr(
        "scripts.run_post_starmie_core_handoff._command",
        fail_command,
    )
    monkeypatch.setattr(
        "scripts.run_post_starmie_core_handoff.verify_frozen_model",
        lambda _path: {
            "checkpoint_digest": fallback_digest,
            "model_path": str(fallback_family / "model.pt"),
        },
    )

    ready, frozen, core, updated = _resolve_boundary_core(
        contract_path=contract_path,
        contract=contract,
        runtime={},
        state_path=state_path,
    )

    rejection = json.loads(attempted_ready.read_text(encoding="utf-8"))
    assert rejection["status"] == "rejected_pretraining_validation"
    assert rejection["training_eligible"] is False
    assert rejection["candidate_checkpoint_created"] is False
    assert rejection["gameplay_regression_run"] is False
    assert ready["checkpoint_digest"] == fallback_digest
    assert frozen["checkpoint_digest"] == fallback_digest
    assert core["version"] == 9
    assert updated["core_refresh"]["version"] == 9
    assert json.loads(state_path.read_text(encoding="utf-8"))[
        "production_continues"
    ] is True

    _resolve_boundary_core(
        contract_path=contract_path,
        contract=contract,
        runtime={},
        state_path=state_path,
    )
    assert calls["command"] == 1


def test_required_specialist_ids_is_exact_canonical_roster() -> None:
    identifiers = _required_specialist_ids(ROOT / "state/specialists.yaml")
    assert len(identifiers) == 15
    assert "starmie" in identifiers
    assert "hops-trevenant" in identifiers
    assert "teal-mask-ogerpon-ex" in identifiers
    assert "dragapult-blaziken" not in identifiers
    assert "dragapult-dudunsparce" not in identifiers
    assert "dragapult" not in identifiers
    assert "walrein" not in identifiers
    assert "crustle" not in identifiers
    assert "slowking" in identifiers


def test_required_specialist_ids_rejects_completed_specialist_in_unfinished_order(
    tmp_path: Path,
) -> None:
    state = yaml.safe_load(
        (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
    )
    completed = next(
        row["id"]
        for row in state["specialists"]
        if row["status"] == "passed_frozen"
    )
    state["training_priority"]["ordered_unfinished_ids_after_active"].insert(
        0, completed
    )
    state["current"]["program_progress"]["remaining_unfinished"] += 1
    state_path = tmp_path / "specialists.yaml"
    state_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    (tmp_path / "matchup_adapter_roster.json").write_text(
        (ROOT / "state/matchup_adapter_roster.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unfinished priority"):
        _required_specialist_ids(state_path)


def test_starmie_pass_still_leaves_unfinished_roster_before_population() -> None:
    required = _required_specialist_ids(ROOT / "state/specialists.yaml")
    completed_after_starmie = {
        "alakazam",
        "hops-trevenant",
        "starmie",
    }
    assert len(required - completed_after_starmie) == 12
    assert not population_transition_ready(
        completed_after_starmie,
        required,
        completed_refresh_ids=[],
        required_refresh_order=[
            "alakazam",
            "marnie-s-grimmsnarl-ex",
        ],
    )


def test_population_requires_every_canonical_specialist_and_refresh() -> None:
    required = _required_specialist_ids(ROOT / "state/specialists.yaml")
    refresh_order = ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]
    assert not population_transition_ready(
        set(sorted(required)[:-1]),
        required,
        completed_refresh_ids=refresh_order,
        required_refresh_order=refresh_order,
    )
    assert not population_transition_ready(
        set(required),
        required,
        completed_refresh_ids=[],
        required_refresh_order=refresh_order,
    )
    assert not population_transition_ready(
        set(required),
        required,
        completed_refresh_ids=["alakazam"],
        required_refresh_order=refresh_order,
    )
    assert population_transition_ready(
        set(required),
        required,
        completed_refresh_ids=refresh_order,
        required_refresh_order=refresh_order,
    )
    with pytest.raises(RuntimeError, match="refresh order"):
        population_transition_ready(
            set(required),
            required,
            completed_refresh_ids=list(reversed(refresh_order)),
            required_refresh_order=refresh_order,
        )
    with pytest.raises(RuntimeError, match="refresh order"):
        population_transition_ready(
            set(required),
            required,
            completed_refresh_ids=["alakazam", "unknown"],
            required_refresh_order=refresh_order,
        )


def test_post_fleet_refresh_progress_is_staged_and_receipt_bound() -> None:
    contract = json.loads(
        (ROOT / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    refresh = contract["post_fleet_refresh"]
    seats = refresh["first_refresh"]["turn_order"]
    gates = refresh["release_gates"]
    fallback = refresh["first_refresh"]["preferred_parent_migration"][
        "failure_fallback"
    ]
    assert seats["training_seat_split"] == {"first": 0.5, "second": 0.5}
    assert seats["deterministic_assignment_required"] is True
    assert seats["seat_count_parity_receipt_required"] is True
    assert seats["seat_count_parity_receipt_schema"] == (
        "poke_bot.alakazam_refresh_seat_split/v1"
    )
    assert seats["seat_count_receipt_required_stages"] == [
        "assigned",
        "actual",
        "consumed",
    ]
    assert seats["equal_first_second_counts_required_at_each_stage"] is True
    assert seats["package_preference"] == "first_if_allowed"
    assert seats["second_focus_1_to_7_allowed"] is False
    assert seats["always_second_arm_allowed"] is False
    assert seats["second_preferring_refresh_copy_allowed"] is False
    assert gates["final_alakazam_model_computation"]["required_receipts"] == [
        "required_specialist_fleet_complete_for_final_alakazam_v1",
        "capacity_research_resource_lease_v1",
    ]
    assert gates["broader_multi_archetype_capacity_program"][
        "required_receipt"
    ] == "post_refresh_sequence_complete_for_capacity_v2"
    assert fallback == {
        "migration_failure_receipt_preserved": True,
        "ordinary_same_archetype_alakazam_refresh_initialized_from": (
            "then_latest_checksum_accepted_core"
        ),
        "expand_only_that_completed_alakazam_derivative_to_final_format": True,
        "latest_core_direct_final_format_tensor_parent_allowed": False,
        "partial_old_alakazam_core_overlay_allowed": False,
    }
    order, completed = _validated_post_fleet_refresh_progress(
        state_path=ROOT / "state/specialists.yaml",
        cycle_contract=contract,
    )
    assert order == ["alakazam", "marnie-s-grimmsnarl-ex", "crustle"]
    assert completed == ["alakazam"]
