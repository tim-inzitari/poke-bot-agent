import json
from pathlib import Path

from poke_bot.matchup_adapters import (
    ACTIVE_EXPERT_IDS_V5,
    EXPERT_IDS,
    RETIRED_EXPERT_IDS_V5,
    STAGED_EXPERT_IDS_V5,
)


def test_v5_roster_is_the_stable_canonical_physical_roster() -> None:
    roster = json.loads(
        Path("state/matchup_adapter_roster.json").read_text(encoding="utf-8")
    )

    assert tuple(roster["expert_ids"]) == EXPERT_IDS
    assert STAGED_EXPERT_IDS_V5 == EXPERT_IDS
    assert ACTIVE_EXPERT_IDS_V5 == EXPERT_IDS
    assert EXPERT_IDS[-1] == "team-rockets-spidops"
    assert not (set(EXPERT_IDS) & RETIRED_EXPERT_IDS_V5)


def test_v5_removed_routes_cannot_count_as_required_specialists() -> None:
    roster = json.loads(
        Path("state/matchup_adapter_roster.json").read_text(encoding="utf-8")
    )

    assert roster["required_specialist_count"] == 18
    assert roster["physical_checkpoint_rows"] == 18
    assert set(roster["migration_from_v4"]["remove_expert_ids"]) == (
        RETIRED_EXPERT_IDS_V5
    )
    assert roster["migration_from_v4"]["removed_rows_must_be_absent"]
    assert roster["migration_from_v4"]["rename_expert_ids"] == {
        "festival-lead": "thwackey"
    }
