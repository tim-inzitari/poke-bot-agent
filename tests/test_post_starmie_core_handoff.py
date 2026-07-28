from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_post_starmie_core_handoff import (
    _compatible_selected_asset_upgrade,
    _generated_contract,
    _upgrade_selected_handoff_contract,
)
from scripts.run_starmie_expert_bootstrap import (
    current_deck_guide_epoch_weight,
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_selected_handoff_allows_only_boundary_router_generation_change() -> None:
    old = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {"gate_contract": "/ops/gate-old.json"},
        "next_specialist": {"id": "dudunsparce"},
        "asset_generation": {"promotion_receipt": "/state/v38.json"},
        "runtime_registration": {
            "run_name": "dudunsparce",
            "inactive_tree_candidate": "/state/tree-v37.json",
            "candidate_audit": "/state/audit-v37.json",
        },
    }
    new = json.loads(json.dumps(old))
    new["source_specialist"]["gate_contract"] = "/ops/gate-saved.json"
    new["asset_generation"] = {"promotion_receipt": "/state/v39.json"}
    new["runtime_registration"]["inactive_tree_candidate"] = (
        "/state/tree-roster18-v39.json"
    )
    new["runtime_registration"]["candidate_audit"] = (
        "/state/audit-roster18-v39.json"
    )
    assert _compatible_selected_asset_upgrade(old, new)
    new["next_specialist"]["id"] = "walrein"
    assert not _compatible_selected_asset_upgrade(old, new)


def test_promoted_router_audit_is_bound_to_resolved_candidate_tree() -> None:
    source = (ROOT / "scripts/run_post_starmie_core_handoff.py").read_text(
        encoding="utf-8"
    )
    assert 'sha256(Path(assets["candidate_tree"]))' in source
    assert 'sha256(_path(runtime, "inactive_tree_candidate"))' not in source


def test_completed_core_regression_is_resumable_without_retraining() -> None:
    source = (ROOT / "scripts/run_post_starmie_core_handoff.py").read_text(
        encoding="utf-8"
    )
    assert 'ready_status == "ready"' in source
    assert 'ready.get("gameplay_regression_passed") is True' in source


def test_validated_core_receipt_is_resumable_after_later_handoff_failure() -> None:
    source = (ROOT / "scripts/run_multi_teacher_core_refresh.py").read_text(
        encoding="utf-8"
    )
    assert 'status == "ready"' in source
    assert 'ready.get("gameplay_regression_passed") is True' in source


def test_selected_specialist_resume_uses_checksum_pinned_generated_contract() -> None:
    source = (ROOT / "scripts/run_post_starmie_core_handoff.py").read_text(
        encoding="utf-8"
    )
    assert "def _resume_selected_handoff" in source
    assert "sha256(generated) != expected_digest" in source
    assert "run_sequential_handoff(generated)" in source


def test_generated_handoff_continues_to_lucario_not_population() -> None:
    contract = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    generated = _generated_contract(
        contract=contract,
        source={"specialist_id": "starmie"},
        selected={
            "specialist_id": "lucario",
            "pointer": (
                "/home/inzi/poke-bot-agent/data/bootstrap/"
                "expert-latest20-20260702-20260721/"
                "specialist-corpora-v1/lucario/PROTECTED_EXPERT_CORPUS.json"
            ),
            "decisions": 173490,
        },
        core_digest="sha256:" + "a" * 64,
    )

    assert generated["source_specialist"]["id"] == "starmie"
    assert generated["source_specialist"]["minimum_completed_iteration"] == 5
    assert generated["next_specialist"]["id"] == "lucario"
    assert generated["training"]["supervised_epochs"] == 25
    assert generated["training"]["minimum_decisions"] == 20000
    expanded = generated["training"]["expanded_heads"]
    assert expanded["schema"] == "poke_bot.expanded_head_training/v1"
    assert expanded["schedule"]["total_epochs"] == 25
    assert expanded["schedule_digest"].startswith("sha256:")
    assert expanded["target_schema_digest"].startswith("sha256:")
    assert expanded["runtime_enabled_heads"] == []
    assert generated["training"]["decision_fusion"] == (
        decision_fusion_handoff_contract()
    )
    assert generated["gate_materialization"]["archetype_label"] == "Starmie"
    assert generated["shared_core"]["checkpoint_checksum"] == (
        "sha256:" + "a" * 64
    )
    assert generated["runtime_registration"]["run_name"].startswith(
        "pure_rl_lucario_"
    )
    assert generated["runtime_registration"]["handoff_service"] == (
        "pokebot-specialist-cycle-handoff.service"
    )
    assert generated["next_specialist"]["training_service"] == (
        "pokebot-pure-rl-trevenant-staged.service"
    )


def test_generated_handoff_propagates_selected_corpus_minimum() -> None:
    contract = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    generated = _generated_contract(
        contract=contract,
        source={"specialist_id": "starmie"},
        selected={
            "specialist_id": "dragapult-dusknoir",
            "pointer": "/corpora/dragapult-dusknoir/PROTECTED_EXPERT_CORPUS.json",
            "decisions": 10_946,
            "minimum_decisions": 10_000,
        },
        core_digest="sha256:" + "b" * 64,
    )

    assert generated["training"]["minimum_decisions"] == 10_000
    assert generated["gate_materialization"]["archetype_label"] == "Starmie"


def test_generated_handoff_reuses_checksum_bound_prestaged_pack(
    tmp_path: Path,
) -> None:
    contract = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = tmp_path / "prestage.json"
    receipt.write_text('{"schema":"poke_bot.next_specialist_prestage/v1"}\n')
    guide_contract = (
        ROOT
        / "config/deck_guides/marnie-s-grimmsnarl-ex.yaml"
    )
    guide_ready = tmp_path / "guide-ready.json"
    guide_ready.write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "guide_version": "marnie-grimmsnarl-north-star-v1",
                "guide_rows": 100,
            }
        ),
        encoding="utf-8",
    )
    contract["next_specialist"]["prestage_receipt"] = str(receipt)
    generated = _generated_contract(
        contract=contract,
        source={"specialist_id": "starmie"},
        selected={
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "pointer": (
                "/corpora/marnie-s-grimmsnarl-ex/"
                "PROTECTED_EXPERT_CORPUS.json"
            ),
            "decisions": 20_000,
            "minimum_decisions": 20_000,
        },
        core_digest="sha256:" + "c" * 64,
        prestage={
            "cpu_pack": {
                "root": "/cache/prestaged/marnie-s-grimmsnarl-ex",
                "status": "ready",
            },
            "current_deck_guide": {
                "status": "ready",
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "implementation_ready": True,
                "corpus_binding_ready": True,
                "targets_ready": True,
                "path": str(guide_contract),
                "sha256": (
                    "sha256:"
                    + hashlib.sha256(guide_contract.read_bytes()).hexdigest()
                ),
                "guide_version": "marnie-grimmsnarl-north-star-v1",
                "corpus_ready_receipt": str(guide_ready),
                "corpus_ready_receipt_sha256": (
                    "sha256:"
                    + hashlib.sha256(guide_ready.read_bytes()).hexdigest()
                ),
            },
        },
    )
    assert (
        generated["next_specialist"]["cpu_pack_root"]
        == "/cache/prestaged/marnie-s-grimmsnarl-ex"
    )
    assert generated["prestage"]["expert_cpu_pack_reused"] is True
    guide = generated["training"]["current_deck_guide"]
    assert guide["specialist_id"] == "marnie-s-grimmsnarl-ex"
    assert guide["runtime_initial_weight"] == 0.05
    assert [
        current_deck_guide_epoch_weight(guide, epoch)
        for epoch in range(1, 7)
    ] == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.05])


def test_prebootstrap_selected_handoff_upgrade_is_narrow_and_recorded() -> None:
    payload = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {"id": "starmie"},
        "next_specialist": {"id": "dragapult-dusknoir"},
        "training": {"minimum_decisions": 20_000, "supervised_epochs": 25},
        "gate_materialization": {"archetype_label": "Mega Starmie ex"},
        "unrelated": {"preserved": True},
    }
    upgraded, changes = _upgrade_selected_handoff_contract(
        payload,
        {
            "specialist_id": "dragapult-dusknoir",
            "decisions": 10_946,
            "minimum_decisions": 10_000,
        },
    )

    assert upgraded["training"]["minimum_decisions"] == 10_000
    assert upgraded["gate_materialization"]["archetype_label"] == "Starmie"
    assert upgraded["training"]["supervised_epochs"] == 25
    assert upgraded["training"]["expanded_heads"]["schedule"]["total_epochs"] == 25
    assert upgraded["training"]["decision_fusion"] == (
        decision_fusion_handoff_contract()
    )
    assert upgraded["unrelated"] == {"preserved": True}
    assert set(changes) == {
        "minimum_decisions",
        "source_archetype_label",
        "expanded_heads",
        "decision_fusion",
    }


def test_expanded_schedule_migration_refuses_started_bootstrap(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {"id": "starmie"},
        "next_specialist": {
            "id": "dragapult-dusknoir",
            "run_dir": str(run_dir),
        },
        "training": {"minimum_decisions": 20_000, "supervised_epochs": 25},
        "gate_materialization": {"archetype_label": "Starmie"},
    }

    with pytest.raises(RuntimeError, match="bootstrap has started"):
        _upgrade_selected_handoff_contract(
            payload,
            {
                "specialist_id": "dragapult-dusknoir",
                "decisions": 20_000,
                "minimum_decisions": 20_000,
            },
        )


def test_prebootstrap_upgrade_binds_required_current_deck_guide() -> None:
    payload = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {"id": "garchomp"},
        "next_specialist": {"id": "rockets-mewtwo"},
        "training": {"minimum_decisions": 20_000, "supervised_epochs": 25},
        "gate_materialization": {"archetype_label": "Garchomp"},
    }
    guide = {
        "schema": "poke_bot.current_deck_guide_handoff/v1",
        "specialist_id": "rockets-mewtwo",
        "guide_version": "rockets-mewtwo-north-star-v1",
    }

    upgraded, changes = _upgrade_selected_handoff_contract(
        payload,
        {
            "specialist_id": "rockets-mewtwo",
            "decisions": 868_246,
            "minimum_decisions": 20_000,
        },
        current_deck_guide=guide,
    )

    assert upgraded["training"]["current_deck_guide"] == guide
    assert changes["current_deck_guide"] == {
        "before": None,
        "after": guide,
    }


def test_current_deck_guide_migration_refuses_started_bootstrap(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {"id": "garchomp"},
        "next_specialist": {
            "id": "rockets-mewtwo",
            "run_dir": str(run_dir),
        },
        "training": {
            "minimum_decisions": 20_000,
            "supervised_epochs": 25,
            "expanded_heads": expanded_handoff_training_contract(),
            "decision_fusion": decision_fusion_handoff_contract(),
        },
        "gate_materialization": {"archetype_label": "Garchomp"},
    }

    with pytest.raises(RuntimeError, match="guide after specialist bootstrap"):
        _upgrade_selected_handoff_contract(
            payload,
            {
                "specialist_id": "rockets-mewtwo",
                "decisions": 868_246,
                "minimum_decisions": 20_000,
            },
            current_deck_guide={
                "schema": "poke_bot.current_deck_guide_handoff/v1",
                "specialist_id": "rockets-mewtwo",
            },
        )


def test_prebootstrap_upgrade_recovers_checksum_bound_transition(
    tmp_path: Path,
) -> None:
    transition = tmp_path / "transition.json"
    transition.write_text('{"passed": true}\\n', encoding="utf-8")
    transition_digest = "sha256:" + hashlib.sha256(
        transition.read_bytes()
    ).hexdigest()
    handler = tmp_path / "handler.json"
    handler.write_text(
        json.dumps(
            {
                "threshold_transition_receipt": str(transition),
                "threshold_transition_receipt_sha256": transition_digest,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {
            "id": "starmie",
            "handler_state": str(handler),
        },
        "next_specialist": {"id": "dragapult-dusknoir"},
        "training": {"minimum_decisions": 20_000},
        "gate_materialization": {"archetype_label": "Mega Starmie ex"},
    }

    upgraded, changes = _upgrade_selected_handoff_contract(
        payload,
        {
            "specialist_id": "dragapult-dusknoir",
            "decisions": 10_946,
            "minimum_decisions": 10_000,
        },
    )

    assert upgraded["source_specialist"][
        "threshold_transition_receipt"
    ] == str(transition)
    assert changes["threshold_transition_receipt"]["before"] is None


def test_core_template_keeps_historical_v2_teachers_without_future_exclusions() -> None:
    contract = json.loads(
        (ROOT / "ops/post_starmie_core_v2_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_ids = [
        row["specialist_id"] for row in contract["core_refresh"]["teachers"]
    ]
    assert teacher_ids == ["hops-trevenant", "starmie"]
    assert contract["core_refresh"]["direct_checkpoint_tensor_sources_exclude"] == []
