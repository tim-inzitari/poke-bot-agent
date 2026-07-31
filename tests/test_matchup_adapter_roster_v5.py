import json
from pathlib import Path

from poke_bot.matchup_adapters import (
    ACTIVE_EXPERT_IDS_V5,
    EXPERT_IDS,
    LEGACY_EXPERT_IDS_V4,
    RETIRED_EXPERT_IDS_V5,
    STAGED_EXPERT_IDS_V5,
)
from scripts.migrate_active_matchup_roster import migrate_validation


def test_v5_roster_is_the_stable_canonical_physical_roster() -> None:
    roster = json.loads(
        Path("state/matchup_adapter_roster.json").read_text(encoding="utf-8")
    )

    assert tuple(roster["expert_ids"][:18]) == EXPERT_IDS
    assert roster["expert_ids"][18:] == [
        "teal-mask-ogerpon-ex",
        "slowking",
    ]
    assert STAGED_EXPERT_IDS_V5 == EXPERT_IDS
    assert ACTIVE_EXPERT_IDS_V5 == EXPERT_IDS
    assert EXPERT_IDS[-1] == "team-rockets-spidops"
    assert not (set(EXPERT_IDS) & RETIRED_EXPERT_IDS_V5)


def test_v5_physical_rows_remain_stable_when_v6_appends_a_route() -> None:
    roster = json.loads(
        Path("state/matchup_adapter_roster.json").read_text(encoding="utf-8")
    )

    assert roster["required_specialist_count"] == 20
    assert roster["physical_checkpoint_rows"] == 18
    assert roster["legacy_v5_prefix_length"] == 18
    assert set(roster["migration_from_v4"]["remove_expert_ids"]) == (
        RETIRED_EXPERT_IDS_V5
    )
    assert roster["migration_from_v4"]["removed_rows_must_be_absent"]
    assert roster["migration_from_v4"]["rename_expert_ids"] == {
        "festival-lead": "thwackey"
    }


def test_router_validation_migration_preserves_renamed_route_evidence() -> None:
    old_names = (*LEGACY_EXPERT_IDS_V4, "unknown")
    old_index = {name: index for index, name in enumerate(old_names)}
    matrix = [[0.0 for _ in old_names] for _ in old_names]
    matrix[old_index["festival-lead"]][old_index["festival-lead"]] = 68_674
    matrix[old_index["gardevoir"]][old_index["gardevoir"]] = 24_447
    matrix[old_index["unknown"]][old_index["unknown"]] = 10_000

    migrated = migrate_validation(
        {"confusion_matrix": matrix}, LEGACY_EXPERT_IDS_V4
    )

    assert migrated["weighted_observations"] == 103_121
    assert migrated["classes"]["thwackey"]["weighted_support"] == 68_674
    assert migrated["classes"]["thwackey"]["precision"] == 1.0
    assert migrated["classes"]["thwackey"]["recall"] == 1.0
    assert migrated["classes"]["team-rockets-spidops"]["weighted_support"] == 0
    assert migrated["classes"]["unknown"]["weighted_support"] == 34_447
    assert len(migrated["confusion_matrix"]) == len(EXPERT_IDS) + 1
