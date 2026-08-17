"""Regression coverage for the isolated, zero-gated r298 checklist boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import poke_bot.alakazam_checklist_provenance_r298 as checklist


def _trace(*, bench_only: bool = True, hidden_marker: object | None = None) -> dict[str, object]:
    """A structurally valid r288-shaped trace with two legal candidates."""

    facts: dict[str, object] = {
        "bench_only": bench_only,
        "classification": "ready",
    }
    if hidden_marker is not None:
        # This must not change strict public mechanics rows.  It is deliberately
        # not part of the trace-evidence digest or any allowed provenance fact.
        facts["opponent_hand_identities"] = hidden_marker
    return {
        "channel_names": list(checklist.CHANNEL_NAMES),
        "channels": [
            {
                "name": name,
                "raw": [0.25, -0.25],
                "normalized": [1.0, -1.0],
                "option_availability": [True, True],
                "available": True,
                "reason": "synthetic_public_diagnostic_only",
            }
            for name in checklist.CHANNEL_NAMES
        ],
        "residuals": [0.0, 0.0],
        "facts": facts,
    }


def _rows(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload["channels"]
    assert isinstance(rows, list)
    return {str(row["name"]): row for row in rows if isinstance(row, dict)}


def test_all_eight_rows_are_exact_zero_when_provenance_is_missing() -> None:
    result = checklist.filter_r288_checklist_trace_r298(
        _trace(), provenance={}
    ).to_dict()
    rows = _rows(result)

    assert list(rows) == list(checklist.CHANNEL_NAMES)
    assert result["residuals"] == [0.0, 0.0]
    assert result["active"] is False
    assert result["runtime_wired"] is False
    for name in checklist.CHANNEL_NAMES:
        assert rows[name]["final_residual"] == 0.0
        assert rows[name]["final_residual_vector"] == [0.0, 0.0]
    for name in checklist.STRICT_CHANNELS:
        assert rows[name]["available"] is False
        assert rows[name]["status"] == "unavailable"
        assert rows[name]["provenance_eligible"] is False
    assert rows["bench_prize_exposure"]["trace_only"] is True
    assert rows["immediate_disruption_outcome"]["trace_only"] is True
    assert rows["bench_prize_exposure"]["applied_gate"] == 0.0
    assert rows["immediate_disruption_outcome"]["applied_gate"] == 0.0
    assert result["guide_support"]["final_residual_vector"] == [0.0, 0.0]


def test_hidden_fact_variation_cannot_change_strict_public_rows() -> None:
    left = checklist.filter_r288_checklist_trace_r298(
        _trace(hidden_marker=[10, 11]), provenance={}
    ).to_dict()
    right = checklist.filter_r288_checklist_trace_r298(
        _trace(hidden_marker=[999, 1000]), provenance={}
    ).to_dict()
    left_rows = _rows(left)
    right_rows = _rows(right)

    for name in checklist.STRICT_CHANNELS:
        for field in (
            "raw",
            "normalized",
            "option_availability",
            "available",
            "status",
            "reason",
            "provenance_eligible",
            "final_residual_vector",
        ):
            assert left_rows[name][field] == right_rows[name][field]


def test_q3_active_only_claim_is_rejected_as_not_bench_only() -> None:
    result = checklist.filter_r288_checklist_trace_r298(
        _trace(bench_only=False), provenance={}
    ).to_dict()
    row = _rows(result)["replacement_alakazam_line"]

    assert row["available"] is False
    assert row["status"] == "unavailable"
    assert row["provenance_eligible"] is False
    assert "bench_only_missing_or_false" in str(row["reason"])
    assert row["final_residual_vector"] == [0.0, 0.0]


def test_forged_catalog_envelope_cannot_become_authority(
    tmp_path: Path,
) -> None:
    """A locally self-claimed receipt fails before its JSON is trusted."""

    forged = tmp_path / "receipt.json"
    forged.write_text(
        '{"schema":"poke_bot.alakazam_public_catalog_r298_receipt/v1",'
        '"status":"passed_elmo_only_nonproduction"}',
        encoding="utf-8",
    )
    with pytest.raises(
        checklist.ChecklistProvenanceError,
        match="not the immutable r298 sealed artifact",
    ):
        checklist.validate_sealed_public_catalog_r298(forged)


def test_catalog_json_by_itself_is_never_authority(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    with pytest.raises(
        checklist.ChecklistProvenanceError,
        match="receipt filename is not canonical",
    ):
        checklist.validate_pinned_public_catalog_r298(catalog)


def test_real_elmo_catalog_sealer_round_trip_when_mounted() -> None:
    """Exercise the actual r4 receipt as predecessor-only evidence."""

    artifact_raw = os.environ.get("R298_PUBLIC_CATALOG_ARTIFACT_DIR")
    if not artifact_raw:
        pytest.skip("real r298 public catalog artifact is not mounted")
    artifact = Path(artifact_raw)
    receipt = artifact / "receipt.json"
    if not receipt.is_file():
        pytest.skip("mounted r298 public catalog artifact has no receipt.json")
    evidence = checklist.validate_sealed_public_catalog_r298(receipt)
    assert evidence.sealer_receipt.sha256 == (
        "sha256:9ca7534281dd08d1804babe223fa0779a84a1f5a7ee6f609497d595dca10b4d1"
    )
    assert evidence.catalog.sha256 == (
        "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6"
    )
    assert evidence.fixed_vectors.sha256 == (
        "sha256:2e1a817dac1d17d0131056ead9669b76803b41f5864d4e9cd82d7f34a9180e09"
    )
    assert evidence.execution_host == "truenas"
    assert evidence.predecessor_only is True
    assert evidence.to_dict()["revision_5_mechanics_authority"] is False
    with pytest.raises(
        checklist.ChecklistProvenanceError,
        match="predecessor evidence only",
    ):
        checklist.validate_pinned_public_catalog_r298(receipt)


def test_r303_manifest_binds_current_goal_contract_and_r241_source() -> None:
    config = checklist.load_checklist_provenance_config_r298()
    manifest = checklist.checklist_provenance_schema_manifest_r298()

    assert config.canonical_goal_sha256 == checklist.CANONICAL_GOAL_SHA256
    assert config.canonical_contract_sha256 == checklist.CANONICAL_CONTRACT_SHA256
    assert config.canonical_r241_typed_source_sha256 == (
        checklist.CANONICAL_R241_TYPED_SOURCE_SHA256
    )
    assert config.revision_5_consumer_migration_anchor["status"] == "pinned_immutable_artifact"
    assert config.revision_5_consumer_migration_anchor["receipt_file_sha256"] == (
        "sha256:d33aa897f244c18203bf3851b70b7269cae928fc6e8e2447e5a363b8817f77de"
    )
    definition = manifest["schema_definition"]
    assert isinstance(definition, dict)
    assert definition["canonical_goal_sha256"] == checklist.CANONICAL_GOAL_SHA256
    assert definition["canonical_contract_sha256"] == checklist.CANONICAL_CONTRACT_SHA256
    assert definition["canonical_r241_typed_source_sha256"] == (
        checklist.CANONICAL_R241_TYPED_SOURCE_SHA256
    )
    assert definition["revision_4_catalog_reuse_without_explicit_revision_5_migration"] is False


def test_self_authored_revision_5_migration_receipt_is_not_authority(tmp_path: Path) -> None:
    forged = tmp_path / "migration.json"
    forged.write_text("{}", encoding="utf-8")

    with pytest.raises(
        checklist.ChecklistProvenanceError,
        match="not the immutable pinned artifact",
    ):
        checklist.validate_revision_5_consumer_migration_r298(forged)
