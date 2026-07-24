from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_post_starmie_core_handoff import (
    _generated_contract,
    _upgrade_selected_handoff_contract,
)


ROOT = Path(__file__).resolve().parents[1]


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
    contract["next_specialist"]["prestage_receipt"] = str(receipt)
    generated = _generated_contract(
        contract=contract,
        source={"specialist_id": "starmie"},
        selected={
            "specialist_id": "lucario",
            "pointer": "/corpora/lucario/PROTECTED_EXPERT_CORPUS.json",
            "decisions": 20_000,
            "minimum_decisions": 20_000,
        },
        core_digest="sha256:" + "c" * 64,
        prestage={
            "cpu_pack": {
                "root": "/cache/prestaged/lucario",
                "status": "ready",
            }
        },
    )
    assert (
        generated["next_specialist"]["cpu_pack_root"]
        == "/cache/prestaged/lucario"
    )
    assert generated["prestage"]["expert_cpu_pack_reused"] is True


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
    assert upgraded["unrelated"] == {"preserved": True}
    assert set(changes) == {
        "minimum_decisions",
        "source_archetype_label",
    }


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
