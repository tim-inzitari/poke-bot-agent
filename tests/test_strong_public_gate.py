from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.strong_public_gate import (
    _s_plus_floor_check,
    build_active_gate_result,
    build_strong_public_gate_result,
    load_active_gate_contract,
    materialize_fallback_gate_contract,
    verify_roster_content,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ops/alakazam_gate_program_v1.json"


def _rows(ids: list[str], wr_by_id: dict[str, float], *, digest: str) -> list[dict]:
    rows = []
    for pair in range(125):
        for opponent_index, opponent_id in enumerate(ids):
            for seat in (0, 1):
                job_index = ((pair * len(ids) + opponent_index) * 2) + seat
                win_count = round(wr_by_id[opponent_id] * 250)
                opponent_occurrence = pair * 2 + seat
                rows.append(
                    {
                        "job_index": job_index,
                        "opponent_id": opponent_id,
                        "our_seat": seat,
                        "winner": seat if opponent_occurrence < win_count else 1 - seat,
                        "checkpoint_digest": digest,
                        "action_selection": "greedy",
                        "invalid": False,
                        "baseline_failed": False,
                    }
                )
    return rows


def _audit(ids: list[str], *, digest: str) -> dict:
    return {
        "passed": True,
        "valid_games": len(ids) * 250,
        "checkpoint_digest": digest,
        "exact_distribution": True,
        "exact_weights": True,
        "greedy_required": True,
        "per_opponent": {
            opponent_id: {"games": 250, "seat0": 125, "seat1": 125}
            for opponent_id in ids
        },
    }


def test_active_contract_is_superseded_external_plus_frozen_s_plus_gate() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    gate = contract["next_gate"]
    external = [row for row in gate["roster"] if not row.get("frozen_specialist")]
    frozen = [row for row in gate["roster"] if row.get("frozen_specialist")]
    assert len(external) == 3
    assert len(frozen) == 4
    assert len(gate["roster"]) == 7
    assert gate["evaluation"]["games_total"] == 1750
    assert all(row["tier"] == "S+" for row in frozen)
    assert len(gate["research_measurements"]) == 4


def test_package_digest_validation_fails_closed() -> None:
    gate = load_active_gate_contract(CONTRACT_PATH)["next_gate"]
    installed = {row["opponent_id"]: row["content_digest"] for row in gate["roster"]}
    verify_roster_content(gate, installed)
    installed[gate["roster"][0]["opponent_id"]] = "sha256:wrong"
    with pytest.raises(ValueError, match="package digest mismatch"):
        verify_roster_content(gate, installed)


def test_rating_simulation_is_a_separate_actual_game_gate() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    contract = copy.deepcopy(contract)
    gate = contract["next_gate"]
    anchors = gate["roster"][:2]
    for row, rating in zip(anchors, (850.0, 700.0)):
        row["kaggle_rating_anchor"] = rating
    gate["kaggle_rating_simulation"] = {
        "separate_from_premium_strength_gate": True,
        "training_eligible": False,
        "replay_eligible": False,
        "minimum_anchor_count": 2,
        "confidence_level": 0.90,
        "bootstrap_resamples": 200,
        "projected_rating_lower_bound": 1000.0,
    }
    ids = [row["opponent_id"] for row in gate["roster"]]
    digest = "sha256:candidate"
    strong = build_active_gate_result(
        contract=contract,
        checkpoint="/checkpoint.pt",
        checkpoint_digest=digest,
        iteration=5,
        gate_rows=_rows(ids, {key: 0.90 for key in ids}, digest=digest),
        gate_audit=_audit(ids, digest=digest),
        gate_seed=9_050_000,
        bootstrap_resamples=100,
    )
    simulation = strong["kaggle_rating_simulation"]
    assert simulation["actual_simulated_games"] == 500
    assert simulation["separate_from_skill_weighted_win_rate"] is True
    assert simulation["confidence_lower"] >= 1000.0
    assert strong["checks"]["kaggle_rating_simulation"] is True

    weak = build_active_gate_result(
        contract=contract,
        checkpoint="/checkpoint.pt",
        checkpoint_digest=digest,
        iteration=5,
        gate_rows=_rows(ids, {key: 0.65 for key in ids}, digest=digest),
        gate_audit=_audit(ids, digest=digest),
        gate_seed=9_050_001,
        bootstrap_resamples=100,
    )
    assert weak["checks"]["skill_weighted_win_rate"] is True
    assert weak["checks"]["kaggle_rating_simulation"] is False
    assert weak["passed"] is False


def test_strong_gate_uses_active_roster_plus_specialists_and_zero_weight_research() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    gate = contract["next_gate"]
    ids = [row["opponent_id"] for row in gate["roster"]]
    research_ids = [row["opponent_id"] for row in gate["research_measurements"]]
    digest = "sha256:candidate"
    rates = {opponent_id: 0.60 for opponent_id in ids}
    research_rates = {opponent_id: 0.55 for opponent_id in research_ids}
    result = build_strong_public_gate_result(
        contract=contract,
        checkpoint="/checkpoint.pt",
        checkpoint_digest=digest,
        iteration=3,
        gate_rows=_rows(ids, rates, digest=digest),
        gate_audit=_audit(ids, digest=digest),
        research_rows=_rows(research_ids, research_rates, digest=digest),
        research_audit=_audit(research_ids, digest=digest),
        gate_seed=9_030_000,
        research_seed=10_030_000,
        bootstrap_resamples=200,
    )
    assert result["passed"] is True
    assert result["games"] == gate["evaluation"]["games_total"]
    assert len(result["matchups"]) == len(gate["roster"])
    assert result["research_controls"]["games"] == 1000
    assert result["research_controls"]["gate_weight"] == 0.0
    assert result["research_controls"]["included_in_skill_weighted_wr"] is False
    assert result["skill_weighted_wr"] == pytest.approx(0.60)

    changed = copy.deepcopy(result)
    changed["research_controls"]["pooled_wr"] = 0.99
    assert changed["skill_weighted_wr"] == result["skill_weighted_wr"]


def test_s_plus_frozen_specialist_counts_in_s_tier_floor() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    gate = contract["next_gate"]
    ids = [row["opponent_id"] for row in gate["roster"]]
    rates = {
        row["opponent_id"]: (
            0.0
            if row.get("tier") == "S+"
            else 0.40
            if row.get("tier") == "S"
            else 0.80
        )
        for row in gate["roster"]
    }
    digest = "sha256:candidate"
    result = build_active_gate_result(
        contract=contract,
        checkpoint="/checkpoint.pt",
        checkpoint_digest=digest,
        iteration=3,
        gate_rows=_rows(ids, rates, digest=digest),
        gate_audit=_audit(ids, digest=digest),
        gate_seed=9_030_000,
        bootstrap_resamples=100,
    )
    s_rows = [row for row in gate["roster"] if row.get("tier") in {"S", "S+"}]
    expected_s_tier_mean = sum(
        rates[row["opponent_id"]] * float(row["weight"]) for row in s_rows
    ) / sum(float(row["weight"]) for row in s_rows)
    assert result["s_tier_mean"] == pytest.approx(expected_s_tier_mean)
    assert result["checks"]["s_tier_mean_floor"] is False


def test_s_plus_floor_allows_at_most_two_below_thirty_percent() -> None:
    roster = [
        {"opponent_id": f"specialist-{index}", "tier": "S+"}
        for index in range(4)
    ]
    by_id = {
        "specialist-0": {"wr": 0.29},
        "specialist-1": {"wr": 0.10},
        "specialist-2": {"wr": 0.30},
        "specialist-3": {"wr": 0.75},
    }
    passed, below, floor, allowance = _s_plus_floor_check(
        roster,
        by_id,
        {
            "s_plus_individual_floor": 0.30,
            "s_plus_below_floor_allowance": 2,
        },
    )
    assert passed is True
    assert below == ["specialist-0", "specialist-1"]
    assert floor == 0.30
    assert allowance == 2
    by_id["specialist-2"]["wr"] = 0.29
    assert _s_plus_floor_check(
        roster,
        by_id,
        {
            "s_plus_individual_floor": 0.30,
            "s_plus_below_floor_allowance": 2,
        },
    )[0] is False


def test_partial_or_cross_contaminated_gate_cannot_pass() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    gate = contract["next_gate"]
    ids = [row["opponent_id"] for row in gate["roster"]]
    research_ids = [row["opponent_id"] for row in gate["research_measurements"]]
    digest = "sha256:candidate"
    gate_rows = _rows(ids, {key: 0.60 for key in ids}, digest=digest)
    # Corrupt one matched pair by relabeling it as a different matchup.
    gate_rows[1]["opponent_id"] = ids[1]
    audit = _audit(ids, digest=digest)
    audit["passed"] = False
    result = build_strong_public_gate_result(
        contract=contract,
        checkpoint="/checkpoint.pt",
        checkpoint_digest=digest,
        iteration=3,
        gate_rows=gate_rows,
        gate_audit=audit,
        research_rows=_rows(
            research_ids, {key: 0.55 for key in research_ids}, digest=digest
        ),
        research_audit=_audit(research_ids, digest=digest),
        gate_seed=9_030_000,
        research_seed=10_030_000,
        bootstrap_resamples=200,
    )
    assert result["passed"] is False
    assert result["audit"]["passed"] is False


def test_contract_json_remains_machine_readable() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["active_gate_id"] == contract["next_gate"]["id"]
    assert (
        contract["next_gate"]["pass_criteria"][
            "skill_weighted_confidence_lower"
        ]
        == 0.55
    )


def test_lc50_fallback_stays_dormant_before_iteration_5_floor_or_after_pass() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    assert materialize_fallback_gate_contract(
        contract, completed_iteration=3, prior_gate_passed=False
    ) is None
    assert materialize_fallback_gate_contract(
        contract, completed_iteration=4, prior_gate_passed=True
    ) is None


def test_lc50_fallback_changes_only_identity_label_status_and_lower_bound(
    tmp_path: Path,
) -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    fallback = materialize_fallback_gate_contract(
        contract, completed_iteration=4, prior_gate_passed=False
    )
    assert fallback is not None
    assert fallback["active_gate_id"] == (
        "specialist-strong-public-roster-lc50-at-iter5-v1"
        "+frozen-specialists-r4"
    )
    assert fallback["next_gate"]["pass_criteria"][
        "skill_weighted_confidence_lower"
    ] == 0.50
    assert fallback["next_gate"]["activation"] == {
        "schema": "poke_bot.iteration_gate_fallback_activation/v1",
        "prior_gate_id": (
            "alakazam-strong-public-roster-lc55-v2+frozen-specialists-r4"
        ),
        "activate_after_completed_iteration": 4,
        "observed_completed_iteration": 4,
        "prior_gate_passed": False,
        "only_changed_criterion": "skill_weighted_confidence_lower",
        "prior_confidence_lower": 0.55,
        "active_confidence_lower": 0.50,
    }
    ignored = {"id", "label", "status", "pass_criteria", "activation"}
    assert {
        key: value
        for key, value in fallback["next_gate"].items()
        if key not in ignored
    } == {
        key: value
        for key, value in contract["next_gate"].items()
        if key not in ignored
    }
    primary_criteria = dict(contract["next_gate"]["pass_criteria"])
    fallback_criteria = dict(fallback["next_gate"]["pass_criteria"])
    primary_criteria.pop("skill_weighted_confidence_lower")
    fallback_criteria.pop("skill_weighted_confidence_lower")
    assert fallback_criteria == primary_criteria
    # The fully materialized result must satisfy the same structural gate
    # loader before a boundary watcher can install it.
    path = tmp_path / "lc50-derived.json"
    path.write_text(json.dumps(fallback), encoding="utf-8")
    loaded = load_active_gate_contract(path)
    assert loaded["active_gate_id"] == fallback["active_gate_id"]
