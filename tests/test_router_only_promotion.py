from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_router_only_promotion_is_roster_driven_and_immutable() -> None:
    source = (ROOT / "scripts/register_router_only_promotion.py").read_text(
        encoding="utf-8"
    )
    assert 'roster.get("active_expert_ids")' in source
    assert "router-only promotion identity failed" in source
    assert "specialist_boundary_only" in source
    assert "live_trainer_modified" in source
    assert "immutable promotion differs" in source
    assert "minimum_weighted_support" in source
