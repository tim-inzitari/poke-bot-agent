from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.prepare_final_format_marnie_refresh import _execution_plan, _static


def _args(tmp_path: Path, guide: Path, ready: Path) -> argparse.Namespace:
    state = tmp_path / "specialists.yaml"
    expert = tmp_path / "expert.json"
    original = tmp_path / "original.pt"
    for path in (state, expert, original):
        path.write_text("{}", encoding="utf-8")
    learned = [f"head_{index}" for index in range(18)] + ["combo_state"]
    for name, payload in {
        "roles.json": {
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "training_mode": "strategic_directional_v2",
            "canonical_learned_decision_sources": learned,
            "heads": {
                head: {
                    "causal_input": "board_state_and_legal_option",
                    "enters_decision_fusion": True,
                    "action_influence": "bounded_option_conditioned_route",
                }
                for head in learned
            },
        },
        "curriculum.json": {"specialist_id": "marnie-s-grimmsnarl-ex", "training_mode": "strategic_directional_v2"},
        "validation.json": {"specialist_id": "marnie-s-grimmsnarl-ex", "training_mode": "strategic_directional_v2", "status": "validated", "guide_contract_sha256": "placeholder", "guide_ready_receipt_sha256": "placeholder", "measurements": {"guide_labeled_rows": 1}},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return argparse.Namespace(
        check=True,
        specialist_state=state,
        guide=guide,
        guide_ready=ready,
        head_role_map=tmp_path / "roles.json",
        curriculum_spec=tmp_path / "curriculum.json",
        curriculum_validation=tmp_path / "validation.json",
        expert_manifest=expert,
        original_checkpoint=original,
        refresh_registry=tmp_path / "not-yet-registered.json",
    )


def test_marnie_prestage_requires_strategic_training_only_guide(tmp_path: Path) -> None:
    guide = tmp_path / "guide.yaml"
    ready = tmp_path / "ready.json"
    guide.write_text(
        """schema_version: poke_bot.current_deck_guide/v1
specialist_id: marnie-s-grimmsnarl-ex
guide_version: test-v1
policy_target:
  training_mode: strategic_directional_v2
  target_logits: none
  direct_policy_cross_entropy_allowed: false
  guide_preference_index_may_affect_loss_or_gradient: true
  final_policy_logits_are_guide_targets: false
  serving_authority: false
  override_authoritative_policy: false
""",
        encoding="utf-8",
    )
    ready.write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": "marnie-s-grimmsnarl-ex",
                "guide_version": "test-v1",
                "guide_rows": 1,
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, guide, ready)
    validation = json.loads(args.curriculum_validation.read_text())
    import hashlib

    guide_digest = "sha256:" + hashlib.sha256(guide.read_bytes()).hexdigest()
    for path in (args.head_role_map, args.curriculum_spec):
        payload = json.loads(path.read_text())
        payload["guide_contract_sha256"] = guide_digest
        path.write_text(json.dumps(payload), encoding="utf-8")
    validation["guide_contract_sha256"] = guide_digest
    validation["guide_ready_receipt_sha256"] = "sha256:" + hashlib.sha256(ready.read_bytes()).hexdigest()
    args.curriculum_validation.write_text(json.dumps(validation), encoding="utf-8")
    assert "guide_ready" in _static(args)

    guide.write_text(
        guide.read_text(encoding="utf-8").replace(
            "direct_policy_cross_entropy_allowed: false",
            "direct_policy_cross_entropy_allowed: true",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="strategic-ready"):
        _static(_args(tmp_path, guide, ready))


def test_marnie_training_cannot_begin_before_h10_expansion() -> None:
    plan = _execution_plan()
    migration = next(row for row in plan if row["ordinal"] == 2)
    launch = next(row for row in plan if row["stage"] == "migrate_router_and_launch_managed_rl")
    assert migration["stage"] == "hot_start_and_expand_to_final_submission_h10"
    assert migration["training_before_h10_migration_allowed"] is False
    assert migration["bootstrap_runs_entirely_after_h10_migration"] is True
    assert migration["epochs"] == 25
    assert migration["capacity_profile"] == "H10-I/v1"
    assert migration["final_learned_head_count"] == 19
    assert migration["final_distinct_action_route_count"] == 19
    assert migration["decision_fusion_schema"] == "poke_bot.causal_decision_fusion/v3"
    assert migration["guide_training_mode"] == "strategic_directional_v2"
    assert migration["guide_weight"] == 0.05
    assert "combo_state" in migration["guide_pairwise_route_heads"]
    assert launch["ordinal"] > migration["ordinal"]
