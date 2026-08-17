from __future__ import annotations

import pytest

from scripts.build_alakazam_prize_plan_v2_c3 import (
    build_c3_artifact,
    public_support_confidence,
)
from scripts.materialize_alakazam_prize_plan_v2_h3_cache import (
    CACHE_SCHEMA,
    H3CacheError,
    _validate_cache_jsonl,
)
from scripts.train_alakazam_prize_plan_v2 import CompleteActionExample


SHA = "sha256:" + "1" * 64


def _row(token: int, *, h3: bool = True) -> CompleteActionExample:
    return CompleteActionExample(
        identity=("2026-07-23", SHA, "e.json", "episode", 0, token, f"p{token}"),
        utc_day="2026-07-23",
        stage_count=1,
        first_stage_menu=((0.0,) * 40,),
        selected_stage_features=((0.0,) * 40,),
        selected_option_indices=(0,),
        selected_legal_counts=(1,),
        selected_action_programs=((token,),),
        plan_targets=(0.0, 0.0, 0.0, 0.0),
        plan_masks=(True, h3, True, True),
    )


def test_support_confidence_is_zero_for_unseen_and_monotone() -> None:
    assert public_support_confidence(0, 9) == 0.0
    assert 0.0 < public_support_confidence(1, 9) < public_support_confidence(10, 9) < 1.0


def test_artifact_counts_only_h3_available_train_actions() -> None:
    rows = [_row(1), _row(1), _row(2), _row(3, h3=False)]
    artifact = build_c3_artifact(
        rows,
        source_binding={"sealed": True},
        historical_contract_sha256=SHA,
        current_contract_sha256=SHA,
        h3_scale_support_sha256=SHA,
        h3_scale_artifact_sha256=SHA,
        critic_checkpoint_sha256=SHA,
        validation_receipt_sha256=SHA,
    )
    assert artifact["train_complete_actions_opened"] == 4
    assert artifact["train_h3_labeled_complete_actions"] == 3
    assert artifact["unique_public_selected_action_signatures"] == 2
    counts = sorted(item["train_h3_chosen_action_count"] for item in artifact["support_table"].values())
    assert counts == [1, 2]
    assert artifact["actor_activation"] is False


def test_cache_jsonl_validator_rejects_blank_records(tmp_path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text('{"schema":"' + CACHE_SCHEMA + '"}\n', encoding="utf-8")
    _validate_cache_jsonl(valid, 1)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"schema":"' + CACHE_SCHEMA + '"}\n\n', encoding="utf-8")
    with pytest.raises(H3CacheError, match="blank JSONL"):
        _validate_cache_jsonl(invalid, 1)
