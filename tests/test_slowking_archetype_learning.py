from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.slowking_archetype_learning import (
    SlowkingArchetypeLearningError,
    build_game_rows,
    load_aggregate,
    load_contract,
    load_reverse_teacher_audit,
    summarize_rows,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/slowking_archetype_learning.v1.json"


def _evidence():
    contract = load_contract(CONTRACT)
    aggregate = load_aggregate(contract, root=ROOT)
    return contract, aggregate


def test_all_768_games_are_archetype_learning_eligible() -> None:
    contract, aggregate = _evidence()
    rows = build_game_rows(contract, aggregate)
    summary = summarize_rows(rows)
    assert summary == {
        "games": 768,
        "policy_learning_eligible": 768,
        "by_split": {"train": 429, "validation": 180, "test": 159},
        "teams": ["ShumpeiNomura", "vibechu"],
        "exact_lists": [
            "sha256:56ac56d0d2bf1ae0c2d18562422492cffbfafc04b7fe8292cf355809a84c1cf7",
            "sha256:d3f092b737c0990149541576444e767d34a7367576d15a8f1b39f7b15460645d",
            "sha256:d44f701afb3101a0c0d5edca8fe81cb3ff68b4f2f1b733b87bc163a2c1a23d09",
        ],
    }
    assert all(row["exact_list_learning_gate"] is False for row in rows)
    assert {row["sample_weight"] for row in rows} == {1.0}


def test_capability_masks_follow_deck_contents_without_filtering_games() -> None:
    contract, aggregate = _evidence()
    rows = build_game_rows(contract, aggregate)
    by_fingerprint = {}
    for row in rows:
        by_fingerprint.setdefault(row["deck_fingerprint"], row)
    old = by_fingerprint[
        "sha256:56ac56d0d2bf1ae0c2d18562422492cffbfafc04b7fe8292cf355809a84c1cf7"
    ]
    intermediate = by_fingerprint[
        "sha256:d44f701afb3101a0c0d5edca8fe81cb3ff68b4f2f1b733b87bc163a2c1a23d09"
    ]
    final = by_fingerprint[
        "sha256:d3f092b737c0990149541576444e767d34a7367576d15a8f1b39f7b15460645d"
    ]
    assert old["capabilities"]["boomerang_energy_route"] is True
    assert old["capabilities"]["secret_box_route"] is True
    assert old["capabilities"]["prime_catcher_route"] is False
    assert intermediate["capabilities"]["spectrier_route"] is True
    assert intermediate["capabilities"]["second_smoochum_composition"] is False
    assert final["capabilities"]["spectrier_route"] is False
    assert final["capabilities"]["second_smoochum_composition"] is True
    assert all(row["policy_learning_eligible"] for row in (old, intermediate, final))


def test_exact_list_gate_is_rejected() -> None:
    payload = json.loads(CONTRACT.read_text())
    payload["archetype"]["exact_list_required_for_learning"] = True
    with pytest.raises(SlowkingArchetypeLearningError, match="exact-list learning gate"):
        validate_contract(payload)


def test_contract_cannot_grant_training_or_runtime_authority() -> None:
    payload = json.loads(CONTRACT.read_text())
    payload["authority"]["may_start_training"] = True
    with pytest.raises(SlowkingArchetypeLearningError, match="forbidden authority"):
        validate_contract(payload)


def test_sparse_teacher_feature_index_is_checksum_bound_and_causal() -> None:
    contract = load_contract(CONTRACT)
    audit = load_reverse_teacher_audit(contract, root=ROOT)
    assert audit["overall"]["games"] == 768
    assert audit["overall"]["covered"] == 430
    assert audit["overall"]["agreed"] == 410
    assert len(audit["covered_decisions"]) == 430
    assert all("option_scores" in row for row in audit["covered_decisions"])
    assert audit["policy"]["runtime_authority"] == "none"
