from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_spidops_retrain_is_causal_and_roster_driven() -> None:
    source = (
        ROOT / "scripts/retrain_public_matchup_tree_spidops_v41.py"
    ).read_text(encoding="utf-8")
    assert "visible_opponent_card_ids" in source
    assert "opponent_hand" in source
    assert "future_observations" in source
    assert '"team-rockets-spidops"' in source
    assert 'payload.get("active_expert_ids")' in source
    assert "source-name-to-canonical-roster-v1" in source


def test_spidops_retrain_keeps_candidate_inactive() -> None:
    source = (
        ROOT / "scripts/retrain_public_matchup_tree_spidops_v41.py"
    ).read_text(encoding="utf-8")
    assert source.count('"runtime_enabled": False') >= 2
    assert '"activation_status": "validation_required"' in source
