from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from scripts.prestage_next_specialist import (
    _deck_guide_contract,
    _nonlinear_decision_support_contract,
)


ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, selected_guide_rows: int) -> tuple[Path, Path]:
    specialist_id = "example-specialist"
    writeup = tmp_path / "docs" / "deck_guides" / "example.txt"
    writeup.parent.mkdir(parents=True)
    writeup.write_text("How to play well.\n", encoding="utf-8")

    contract = {
        "schema_version": "poke_bot.current_deck_guide/v1",
        "specialist_id": specialist_id,
        "guide_version": "example-guide-v1",
        "teacher_module": "poke_bot.example",
        "strategy_sources": [
            {
                "url": "https://example.invalid/guide",
                "reviewed_at_utc": "2026-07-25T00:00:00Z",
            }
        ],
        "expert_writeup": {
            "path": "docs/deck_guides/example.txt",
            "sha256": _sha256(writeup),
            "word_count": 4,
            "maximum_words": 10000,
            "guide_identity": specialist_id,
            "cites_same_strategy_source_set": True,
        },
        "validation": {
            "unit_tests_passed": True,
            "scorer_canary_passed": True,
            "guide_rows_in_filtered_expert_corpus": 7,
        },
    }
    contract_path = (
        tmp_path / "config" / "deck_guides" / f"{specialist_id}.yaml"
    )
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = corpus / "manifest.json"
    _write_json(
        manifest,
        {
            "totals": {
                "decisions_kept": 11,
                "target_coverage": {"guide_rows": selected_guide_rows},
            }
        },
    )
    pointer = corpus / "PROTECTED_EXPERT_CORPUS.json"
    _write_json(
        pointer,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "manifest": "manifest.json",
            "manifest_sha256": _sha256(manifest),
        },
    )
    ready = corpus / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    _write_json(
        ready,
        {
            "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
            "status": "ready",
            "specialist_id": specialist_id,
            "guide_version": "example-guide-v1",
            "manifest_sha256": _sha256(manifest),
            "protected_pointer_sha256": _sha256(pointer),
            "decisions": 11,
            "guide_rows": selected_guide_rows,
        },
    )
    return pointer, manifest


def test_deck_guide_readiness_binds_to_selected_corpus(tmp_path: Path) -> None:
    pointer, manifest = _fixture(tmp_path, selected_guide_rows=7)
    result = _deck_guide_contract(
        tmp_path,
        "example-specialist",
        corpus_pointer=pointer,
        corpus_manifest=manifest,
    )
    assert result["status"] == "ready"
    assert result["corpus_binding_ready"] is True
    assert result["selected_expert_corpus_guide_rows"] == 7


def test_deck_guide_readiness_rejects_unguided_selected_corpus(
    tmp_path: Path,
) -> None:
    pointer, manifest = _fixture(tmp_path, selected_guide_rows=0)
    result = _deck_guide_contract(
        tmp_path,
        "example-specialist",
        corpus_pointer=pointer,
        corpus_manifest=manifest,
    )
    assert result["status"] == "staged"
    assert result["targets_ready"] is False
    assert result["reason"] == "current_deck_guide_corpus_binding_not_ready"


def test_deck_guide_ready_receipt_supplies_post_featurization_count(
    tmp_path: Path,
) -> None:
    pointer, manifest = _fixture(tmp_path, selected_guide_rows=7)
    contract_path = (
        tmp_path
        / "config"
        / "deck_guides"
        / "example-specialist.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    contract["validation"]["guide_rows_in_filtered_expert_corpus"] = None
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")

    result = _deck_guide_contract(
        tmp_path,
        "example-specialist",
        corpus_pointer=pointer,
        corpus_manifest=manifest,
    )

    assert result["status"] == "ready"
    assert result["corpus_binding_ready"] is True
    assert result["filtered_expert_corpus_guide_rows"] == 7
    assert result["declared_filtered_expert_corpus_guide_rows"] is None


def test_hammer_pult_nonlinear_systems_are_checksum_bound() -> None:
    contract = yaml.safe_load(
        (ROOT / "config/deck_guides/hammer-pult.yaml").read_text(
            encoding="utf-8"
        )
    )

    result = _nonlinear_decision_support_contract(
        ROOT,
        "hammer-pult",
        contract,
    )

    assert result["required"] is True
    assert result["ready"] is True
    assert result["authoritative_action_path"] == "fused_policy"
    assert len(result["required_fused_head_inputs"]) == 17
    assert result["required_system_ids"] == [
        "branching_setup_and_pivot",
        "attacker_and_engine_preservation",
        "typed_energy_and_resource_planning",
        "attack_target_and_prize_routing",
        "stochastic_disruption_timing",
        "bench_and_recovery_routing",
        "guide_weight_annealing",
    ]


def test_hammer_pult_missing_nonlinear_systems_fails_closed() -> None:
    result = _nonlinear_decision_support_contract(
        ROOT,
        "hammer-pult",
        {
            "schema_version": "poke_bot.current_deck_guide/v1",
            "specialist_id": "hammer-pult",
        },
    )

    assert result["required"] is True
    assert result["ready"] is False
    assert result["reason"] == "nonlinear_decision_support_not_validated"


def test_teal_mask_ogerpon_nonlinear_systems_are_checksum_bound() -> None:
    contract = yaml.safe_load(
        (
            ROOT / "config/deck_guides/teal-mask-ogerpon-ex.yaml"
        ).read_text(encoding="utf-8")
    )

    result = _nonlinear_decision_support_contract(
        ROOT,
        "teal-mask-ogerpon-ex",
        contract,
    )

    assert result["required"] is True
    assert result["ready"] is True
    assert result["authoritative_action_path"] == "fused_policy"
    assert len(result["required_fused_head_inputs"]) == 17
    assert len(result["required_system_ids"]) == 7
