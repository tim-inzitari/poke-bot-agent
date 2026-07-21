from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.strong_public_gate import (
    build_strong_public_gate_result,
    load_active_gate_contract,
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


def test_active_contract_is_exact_eight_agent_gate() -> None:
    contract = load_active_gate_contract(CONTRACT_PATH)
    gate = contract["next_gate"]
    assert len(gate["roster"]) == 8
    assert gate["evaluation"]["games_total"] == 2000
    assert len(gate["research_measurements"]) == 4


def test_package_digest_validation_fails_closed() -> None:
    gate = load_active_gate_contract(CONTRACT_PATH)["next_gate"]
    installed = {row["opponent_id"]: row["content_digest"] for row in gate["roster"]}
    verify_roster_content(gate, installed)
    installed[gate["roster"][0]["opponent_id"]] = "sha256:wrong"
    with pytest.raises(ValueError, match="package digest mismatch"):
        verify_roster_content(gate, installed)


def test_strong_gate_uses_eight_agents_and_research_has_zero_score_weight() -> None:
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
    assert result["games"] == 2000
    assert len(result["matchups"]) == 8
    assert result["research_controls"]["games"] == 1000
    assert result["research_controls"]["gate_weight"] == 0.0
    assert result["research_controls"]["included_in_skill_weighted_wr"] is False
    assert result["skill_weighted_wr"] == pytest.approx(0.60)

    changed = copy.deepcopy(result)
    changed["research_controls"]["pooled_wr"] = 0.99
    assert changed["skill_weighted_wr"] == result["skill_weighted_wr"]


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
    assert json.loads(CONTRACT_PATH.read_text())["active_gate_id"] == (
        "alakazam-strong-public-roster-v1"
    )
