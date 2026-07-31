from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot import checkpoint
from poke_bot.slowking_candidate_validation import (
    FINAL_SCHEMA,
    REQUIRED_FINAL_CHECKS,
    validate_final_receipt,
    validate_pretraining_authorization,
)
from scripts.run_sequential_specialist_handoff import bootstrap_command
from scripts.run_starmie_expert_bootstrap import (
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = ROOT / "ops/post_trevenant_starmie_handoff_v1.json"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint.checkpoint_digest(path)


def test_pretraining_receipts_authorize_candidate_but_not_freeze(
    tmp_path: Path,
) -> None:
    payloads = {
        "implementation": {
            "schema": "poke_bot.slowking_combo_head_implementation_validation/v1",
            "specialist_id": "slowking",
            "status": "implementation_validated_data_pending",
            "checks": {"implementation_importable": True},
        },
        "corpus": {
            "schema": "poke_bot.slowking_combo_corpus_validation/v1",
            "specialist_id": "slowking",
            "status": "validated",
            "decisions": 12,
            "coverage": {"metadata_combo_rows": 12},
            "checks": {"exact_deck_binding": True},
        },
        "cpu_pack": {
            "schema": "poke_bot.slowking_expert_cpu_pack_validation/v1",
            "specialist_id": "slowking",
            "status": "validated_inactive",
            "decisions": 12,
            "checks": {
                "temporal_layout": True,
                "exact_targets": True,
                "combo_state_targets": True,
                "train_validation_split_nonempty": True,
                "active_archaludon_training_modified": False,
            },
        },
        "parameter_inventory": {
            "schema": "poke_bot.slowking_candidate_parameter_inventory/v1",
            "specialist_id": "slowking",
            "status": "validated_prestage_instantiation",
            "parameter_inventory": {"total_parameters": 1_910_963},
            "budget": {
                "slowking_hard_ceiling": 3_500_000,
                "hard_ceiling_exceeded": False,
            },
        },
    }
    artifacts = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        artifacts[name] = (path, _write(path, payload))

    result = validate_pretraining_authorization(artifacts)

    assert result["status"] == "authorized_to_train_candidate_only"
    assert result["freeze_or_registration_authorized"] is False


def test_final_receipt_must_bind_the_exact_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    receipt = {
        "schema": FINAL_SCHEMA,
        "specialist_id": "slowking",
        "status": "validated",
        "candidate_checkpoint_sha256": checkpoint.checkpoint_digest(candidate),
        "checks": {name: True for name in REQUIRED_FINAL_CHECKS},
    }

    validate_final_receipt(receipt, checkpoint_path=candidate)
    candidate.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="final combo-candidate"):
        validate_final_receipt(receipt, checkpoint_path=candidate)


def test_slowking_bootstrap_command_carries_preflight_not_final_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = json.loads(BASE_CONTRACT.read_text(encoding="utf-8"))
    contract["next_specialist"]["id"] = "slowking"
    contract["next_specialist"]["combo_state_head"] = True
    contract["training"]["expanded_heads"] = expanded_handoff_training_contract()
    contract["training"]["decision_fusion"] = decision_fusion_handoff_contract(
        strategic_curriculum=True,
        combo_state_head=True,
    )
    expected_fusion = contract["training"]["decision_fusion"]
    contract["training"]["combo_state"] = {
        "implementation_receipt": "/state/implementation.json",
        "implementation_receipt_sha256": "sha256:" + "1" * 64,
        "corpus_validation_receipt": "/state/corpus.json",
        "corpus_validation_receipt_sha256": "sha256:" + "2" * 64,
        "cpu_pack_validation_receipt": "/state/pack.json",
        "cpu_pack_validation_receipt_sha256": "sha256:" + "3" * 64,
        "parameter_inventory_receipt": "/state/inventory.json",
        "parameter_inventory_receipt_sha256": "sha256:" + "4" * 64,
        "final_validation_output": "/state/final.json",
    }
    monkeypatch.setattr(
        "scripts.run_sequential_specialist_handoff.expected_decision_fusion",
        lambda _contract: expected_fusion,
    )

    command = bootstrap_command(contract)

    assert "--combo-state-head" in command
    assert "--combo-state-implementation-receipt" in command
    assert "--combo-state-corpus-validation-receipt" in command
    assert "--combo-state-cpu-pack-validation-receipt" in command
    assert "--combo-state-parameter-inventory-receipt" in command
    assert "--combo-state-validation-output" in command
    assert "--combo-state-validation-receipt" not in command
