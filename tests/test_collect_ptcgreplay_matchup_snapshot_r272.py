"""Focused invariants for the r272 PTCGReplay snapshot collector."""

from __future__ import annotations

import json

import pytest

from scripts.collect_ptcgreplay_matchup_snapshot_r272 import (
    SnapshotError,
    _card_multiset,
    decode_facts,
    semantic_digest,
)


def test_decode_facts_requires_exact_bounded_quintuples() -> None:
    base, decks, facts = decode_facts(
        {"base": "2026-07-10", "decks": ["a", "b"], "facts": "0,0,1,0,0"}
    )
    assert base == "2026-07-10"
    assert decks == ["a", "b"]
    assert facts == [(0, 0, 1, 0, 0)]

    with pytest.raises(SnapshotError, match="quintuples"):
        decode_facts({"base": "2026-07-10", "decks": ["a"], "facts": "0,0"})
    with pytest.raises(SnapshotError, match="out-of-domain"):
        decode_facts(
            {"base": "2026-07-10", "decks": ["a"], "facts": "0,0,1,0,0"}
        )


def test_guide_deck_multiset_is_exact_60_and_named() -> None:
    rows = _card_multiset(
        [1] * 4 + [2] * 56,
        card_by_id={
            1: {"name": "Abra", "stage_type": "Pokémon", "type": "Psychic"},
            2: {"name": "Trainer", "stage_type": "Trainer", "type": None},
        },
    )
    assert rows == [
        {
            "card_id": 1,
            "name": "Abra",
            "count": 4,
            "stage_type": "Pokémon",
            "type": "Psychic",
        },
        {
            "card_id": 2,
            "name": "Trainer",
            "count": 56,
            "stage_type": "Trainer",
            "type": None,
        },
    ]
    with pytest.raises(SnapshotError, match="exactly 60"):
        _card_multiset([1] * 59, card_by_id={1: {"name": "Abra"}})


def test_snapshot_digest_is_self_field_independent_and_canonical() -> None:
    payload = {"schema": "x", "rows": [{"name": "Alakazam", "id": 48}]}
    digest = semantic_digest(payload)
    payload["snapshot_sha256"] = digest
    assert semantic_digest(payload) == digest
    assert digest.startswith("sha256:")
    assert len(digest) == 71
    json.dumps(payload, allow_nan=False)
