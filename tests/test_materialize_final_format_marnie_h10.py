from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_final_format_marnie_h10 import (
    BOUNDARY_SCHEMA,
    _validate_authority,
)


def test_marnie_materializer_requires_post_alakazam_core_attempt(
    tmp_path: Path,
) -> None:
    prestage = tmp_path / "prestage.json"
    boundary = tmp_path / "boundary.json"
    pointer = tmp_path / "latest.json"
    prestage.write_text(
        json.dumps(
            {
                "schema": "poke_bot.final_format_marnie_refresh_prestage/v1",
                "status": "authorized_preparation_started",
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "boundary_iteration": 20,
                "final_capacity_profile": "H10-I/v1",
                "final_decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
                "training_authority": False,
                "selector_authority": False,
            }
        )
    )
    pointer.write_text(json.dumps({"schema": "poke_bot.latest_cumulative_core_pointer/v1"}))
    boundary.write_text(
        json.dumps(
            {
                "schema": BOUNDARY_SCHEMA,
                "status": "selected_core_ready_for_marnie_h10",
                "predecessor_refresh": "alakazam",
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "normal_core_refresh_attempted": False,
                "rejected_candidate_blocks_production": False,
                "training_authority": False,
                "selector_authority": False,
                "latest_core_pointer": str(pointer.resolve()),
            }
        )
    )
    with pytest.raises(RuntimeError, match="core boundary"):
        _validate_authority(
            prestage_path=prestage,
            boundary_path=boundary,
            latest_core_path=pointer,
        )
