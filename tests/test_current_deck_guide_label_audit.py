from __future__ import annotations

import pytest

from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from scripts.audit_current_deck_guide_labels import (
    _metadata_guide_rows,
    audit_sequences,
)


def _sequence(*stages: PolicyStage) -> GameSequence:
    decision = DecisionSample(
        board=None,  # type: ignore[arg-type]
        options=None,  # type: ignore[arg-type]
        action=[],
        action_combo_index=0,
        action_combos=[],
        env_step=1,
        policy_stages=list(stages),
    )
    return GameSequence(
        episode_id="1",
        seat=0,
        archetype="dragapult",
        opp_archetype="crustle",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
    )


def _stage(
    *,
    target: int,
    guide: int,
    confidence: float,
) -> PolicyStage:
    return PolicyStage(
        options=None,  # type: ignore[arg-type]
        action_combos=[[0], [1]],
        target_index=target,
        guide_target_index=guide,
        guide_confidence=confidence,
    )


def test_audit_counts_sparse_labels_abstentions_and_agreement() -> None:
    result = audit_sequences(
        [
            _sequence(
                _stage(target=0, guide=0, confidence=0.8),
                _stage(target=1, guide=0, confidence=0.4),
                _stage(target=1, guide=-1, confidence=0.0),
            )
        ]
    )
    assert result["records"] == 1
    assert result["decisions"] == 1
    assert result["policy_stages"] == 3
    assert result["labeled_stages"] == 2
    assert result["abstained_stages"] == 1
    assert result["expert_action_agreements"] == 1
    assert result["expert_action_disagreements"] == 1
    assert result["label_coverage_rate"] == 2 / 3
    assert result["abstention_rate"] == 1 / 3
    assert result["expert_action_agreement_rate"] == 0.5
    assert result["confidence_weighted_expert_action_agreement_rate"] == pytest.approx(
        0.8 / 1.2
    )


def test_audit_rejects_invalid_label_indices_and_confidences() -> None:
    result = audit_sequences(
        [
            _sequence(
                _stage(target=2, guide=0, confidence=1.2),
                _stage(target=0, guide=2, confidence=0.5),
                _stage(target=0, guide=-1, confidence=0.2),
            )
        ]
    )
    assert result["invalid_target_indices"] == 2
    assert result["invalid_confidences"] == 2


def test_metadata_guide_rows_allows_legacy_empty_day_shard() -> None:
    assert (
        _metadata_guide_rows(
            {
                "records_kept": 0,
                "decisions_kept": 0,
                "target_coverage": {},
            }
        )
        == 0
    )


def test_metadata_guide_rows_rejects_missing_count_on_nonempty_shard() -> None:
    assert (
        _metadata_guide_rows(
            {
                "records_kept": 1,
                "decisions_kept": 1,
                "target_coverage": {},
            }
        )
        == -1
    )


def test_metadata_guide_rows_uses_explicit_count() -> None:
    assert (
        _metadata_guide_rows(
            {
                "records_kept": 0,
                "decisions_kept": 0,
                "target_coverage": {"guide_rows": 7},
            }
        )
        == 7
    )
