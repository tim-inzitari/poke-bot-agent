"""Focused offline coverage for the standalone R244 SearchId identity gate."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from poke_bot.r235_kaggle_phase1_preflight import ImmutableReceiptError
from poke_bot.r244_handle_scoped_search_id_preflight import (
    R244HandleScopedSearchIdFailure,
    R244HandleScopedSearchIdInputs,
    WITNESS_SCHEMA,
    run_r244_handle_scoped_search_id_preflight,
)
from scripts.build_r235_r236_immutable_replacement_binding import (
    _validate_handle_scoped_search_id_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
R225 = ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
R236 = ROOT / "state/canonical-libcg-r236.json"


def _witness() -> dict[str, object]:
    """A valid proof that raw SearchId zero is handle-scoped, not global."""

    return {
        "schema": WITNESS_SCHEMA,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_id_numeric_namespace": "per_distinct_agent_start_handle",
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        "per_lane_handle_identities": ["AgentStart#A", "AgentStart#B"],
        "per_lane_search_id_chains": [[0, 1], [0, 1]],
        "per_lane_first_search_ids": [0, 0],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": "AgentStart#A", "first_search_id": 0},
            {"lane_id": 1, "handle_identity": "AgentStart#B", "first_search_id": 0},
        ],
    }


def _inputs(tmp_path: Path, witness: dict[str, object]) -> R244HandleScopedSearchIdInputs:
    witness_path = tmp_path / "r244-witness.json"
    witness_path.write_text(json.dumps(witness, sort_keys=True), encoding="utf-8")
    return R244HandleScopedSearchIdInputs(
        witness_path=witness_path,
        receipt_path=tmp_path / "r244-receipt.json",
        r225_contract_path=R225,
        r236_contract_path=R236,
    )


def test_two_distinct_handles_may_both_start_at_raw_search_id_zero(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, _witness())

    receipt = run_r244_handle_scoped_search_id_preflight(inputs=inputs)

    assert receipt["status"] == "passed"
    assert receipt["passed"] is True
    assert receipt["immutable"] is True
    assert receipt["write_once"] is True
    assert receipt["r225_active_owner_revision"] == 246
    assert receipt["r244_owner_revision"] == 244
    assert receipt["official_libcg_handle_scoped_search_id_identity_regression_passed"] is True
    assert receipt["per_lane_first_search_ids"] == [0, 0]
    assert receipt["handle_scoped_first_search_id_composite_states"] == [
        {"lane_id": 0, "handle_identity": "AgentStart#A", "first_search_id": 0},
        {"lane_id": 1, "handle_identity": "AgentStart#B", "first_search_id": 0},
    ]
    assert receipt["same_raw_first_search_id_on_distinct_handles_accepted"] is True
    assert receipt["duplicate_handle_identity_rejected"] is True
    assert receipt["duplicate_handle_scoped_first_search_id_composite_rejected"] is True
    assert json.loads(inputs.receipt_path.read_text(encoding="utf-8")) == receipt
    assert stat.S_IMODE(inputs.receipt_path.stat().st_mode) == 0o444


def test_duplicate_handle_identity_fails_closed_with_an_immutable_receipt(tmp_path: Path) -> None:
    witness = _witness()
    witness["per_lane_handle_identities"] = ["AgentStart#A", "AgentStart#A"]
    inputs = _inputs(tmp_path, witness)

    with pytest.raises(R244HandleScopedSearchIdFailure) as exc_info:
        run_r244_handle_scoped_search_id_preflight(inputs=inputs)

    assert "distinct raw handles" in str(exc_info.value)
    failure = json.loads(inputs.receipt_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["passed"] is False
    assert failure["immutable"] is True
    assert failure["write_once"] is True


def test_noncanonical_composite_projection_fails_closed(tmp_path: Path) -> None:
    witness = _witness()
    witness["handle_scoped_first_search_id_composite_states"] = [
        {"lane_id": 0, "handle_identity": "AgentStart#A", "first_search_id": 0},
        {"lane_id": 1, "handle_identity": "AgentStart#B", "first_search_id": 1},
    ]
    inputs = _inputs(tmp_path, witness)

    with pytest.raises(R244HandleScopedSearchIdFailure) as exc_info:
        run_r244_handle_scoped_search_id_preflight(inputs=inputs)

    assert "composite states mismatch" in str(exc_info.value)
    assert json.loads(inputs.receipt_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_success_receipt_is_write_once_and_cannot_be_replaced(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, _witness())
    run_r244_handle_scoped_search_id_preflight(inputs=inputs)

    with pytest.raises(ImmutableReceiptError):
        run_r244_handle_scoped_search_id_preflight(inputs=inputs)


def test_success_receipt_satisfies_the_immutable_binding_gate(tmp_path: Path) -> None:
    receipt = run_r244_handle_scoped_search_id_preflight(
        inputs=_inputs(tmp_path, _witness())
    )

    _validate_handle_scoped_search_id_receipt(
        receipt,
        {
            "r225_contract_sha256": receipt["r225_contract_sha256"],
            "canonical_libcg_contract_sha256": receipt[
                "canonical_libcg_contract_sha256"
            ],
        },
    )
