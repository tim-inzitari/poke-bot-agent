"""Append-only exact-ID roster allocation for the r273 candidate."""

from __future__ import annotations

import copy

import pytest

from scripts.build_ptcgreplay_matchup_roster_r273 import (
    RosterError,
    SNAPSHOT_FILE_SHA256,
    SNAPSHOT_SHA256,
    _allocate,
    _validate_baseline,
)


def _baseline() -> dict:
    slots = [
        {
            "slot": index,
            "archetype_id": f"existing-{index}" if index < 20 else None,
            "status": "active" if index < 18 else "dormant" if index < 20 else "unused",
            "lineage": f"old-{index}" if index < 20 else None,
        }
        for index in range(64)
    ]
    return {
        "schema": "poke_bot.matchup_adapter_roster/v1",
        "slot_schema": "poke_bot.matchup_adapter_slot_registry/v1",
        "slot_capacity": 64,
        "slots": slots,
        "canonical_display_names": {},
        "specialist_priority": [],
        "meta_analysis_source": {"crosswalk": {}},
    }


def test_exact_numeric_ids_allocate_lowest_dormant_slots() -> None:
    candidate = _baseline()
    _validate_baseline(candidate)
    first = _allocate(
        candidate,
        identity="ptcgreplay-source-id-50",
        display_name="Barbaracle",
        source_namespace="ptcgreplay",
        source_id=50,
        source_name="Barbaracle",
        lineage="test",
    )
    second = _allocate(
        candidate,
        identity="ptcgreplay-source-id-167",
        display_name="Teal Mask Ogerpon ex",
        source_namespace="ptcgreplay",
        source_id=167,
        source_name="Teal Mask Ogerpon ex",
        lineage="test",
    )
    assert (first, second) == (20, 21)
    assert candidate["slots"][20]["status"] == "dormant"
    assert candidate["slots"][21]["archetype_id"] == "ptcgreplay-source-id-167"
    assert candidate["meta_analysis_source"]["crosswalk"][
        "ptcgreplay-source-id-167"
    ]["source_id"] == 167


def test_package_identity_uses_separate_namespace() -> None:
    candidate = _baseline()
    slot = _allocate(
        candidate,
        identity="kaggle-submission-55378392",
        display_name="Alakazam r195 NO-RTP submission 55378392",
        source_namespace="kaggle_submission",
        source_id=55378392,
        source_name="Alakazam r195 NO-RTP",
        lineage="test",
    )
    row = candidate["meta_analysis_source"]["crosswalk"][
        "kaggle-submission-55378392"
    ]
    assert slot == 20
    assert row["source_namespace"] == "kaggle_submission"
    assert row["source_id"] == 55378392


def test_baseline_validation_rejects_nonprefix_allocation() -> None:
    baseline = _baseline()
    altered = copy.deepcopy(baseline)
    altered["slots"][20] = {
        "slot": 20,
        "archetype_id": "unexpected",
        "status": "dormant",
        "lineage": "bad",
    }
    with pytest.raises(RosterError, match="unused-slot suffix"):
        _validate_baseline(altered)


def test_snapshot_raw_and_semantic_digests_are_distinct_and_bound() -> None:
    assert SNAPSHOT_FILE_SHA256 == (
        "sha256:3bd08735ba9e8d10e16757d3f83aed572ff53273bbd40b69c9488cb3bcdac52e"
    )
    assert SNAPSHOT_SHA256 == (
        "sha256:5d392e6f593aaaab43fd6738586c93d7c1303ee08d1dd2223977812ea1fac5a5"
    )
    assert SNAPSHOT_FILE_SHA256 != SNAPSHOT_SHA256
