from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.features import SparseVector
from poke_bot.frozen_prize_plan_advantage import (
    ACTIVATION_RECEIPT_SCHEMA,
    canonical_sha256,
)
from poke_bot.prize_plan_actor_boundary import (
    CACHE_RECEIPT_SCHEMA,
    CACHE_ROW_SCHEMA,
    PrizePlanActorBoundaryError,
    load_h3_actor_provider,
    replay_membership_sha256,
)


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence() -> GameSequence:
    sparse = SparseVector()
    sparse.offset.extend([0, 0])
    stage = PolicyStage(options=sparse, action_combos=[[0]], target_index=0)
    decision = DecisionSample(
        board=sparse,
        options=sparse,
        action=[0],
        action_combo_index=0,
        action_combos=[[0]],
        env_step=7,
        policy_stages=[stage, stage],
    )
    return GameSequence(
        episode_id="episode-7",
        seat=1,
        archetype="alakazam",
        opp_archetype="other",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _artificed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def _sealed(tmp_path: Path) -> tuple[GameSequence, Path, str, Path, str, str]:
    sequence = _sequence()
    shard = tmp_path / "day.h3-additive.jsonl"
    row = {
        "schema": CACHE_ROW_SCHEMA,
        "utc_day": "2026-08-14",
        "split": "train",
        "policy_episode_id": "episode-7",
        "acting_seat": 1,
        "env_step": 7,
        "stage_count": 2,
        "h3_additive_term": 0.0125,
    }
    _write_json(shard, row)
    cache = _artificed(
        {
            "schema": CACHE_RECEIPT_SCHEMA,
            "cache_value_semantics": "h3_additive_term_only",
            "exact_legacy_baseline_computed_in_batch": True,
            "coefficient": 0.025,
            "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0],
            "runtime_critic_calls": False,
            "day_shards": [
                {
                    "path": shard.name,
                    "sha256": _file_sha(shard),
                    "rows": 1,
                    "utc_day": "2026-08-14",
                    "split": "train",
                }
            ],
        }
    )
    cache_path = tmp_path / "cache-receipt.json"
    _write_json(cache_path, cache)
    policy_sha = "sha256:" + "a" * 64
    activation = _artificed(
        {
            "schema": ACTIVATION_RECEIPT_SCHEMA,
            "activation_eligible": True,
            "actor_activation": True,
            "safe_boundary": True,
            "all_pre_activation_gates_passed": True,
            "contract_current_activation_allowed": True,
            "exact_legacy_baseline_computed_in_batch": True,
            "rollback_preflight_passed": True,
            "noninterference_passed": True,
            "no_search_rtp_mcts": True,
            "cache_value_semantics": "h3_additive_term_only",
            "coefficient": 0.025,
            "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0],
            "runtime_critic_calls": False,
            "cache_receipt_sha256": _file_sha(cache_path),
            "cache_artifact_sha256": cache["artifact_sha256"],
            "policy_checkpoint_sha256": policy_sha,
            "replay_membership_sha256": replay_membership_sha256([sequence]),
            "semantic_owner_goal_revision": 23,
            "contract_sha256": "sha256:" + "b" * 64,
        }
    )
    activation_path = tmp_path / "activation.json"
    _write_json(activation_path, activation)
    return (
        sequence,
        cache_path,
        _file_sha(cache_path),
        activation_path,
        _file_sha(activation_path),
        policy_sha,
    )


def test_boundary_loader_binds_exact_complete_action_to_all_stages(tmp_path: Path) -> None:
    sequence, cache, cache_sha, activation, activation_sha, policy_sha = _sealed(tmp_path)
    bound, provider = load_h3_actor_provider(
        sequences=[sequence],
        policy_checkpoint_sha256=policy_sha,
        cache_receipt_path=cache,
        cache_receipt_sha256=cache_sha,
        activation_receipt_path=activation,
        activation_receipt_sha256=activation_sha,
    )
    assert bound == {(id(sequence), 0, 0): 0.0125, (id(sequence), 0, 1): 0.0125}
    assert provider["actor_activation"] is True
    assert provider["provider_binding"]["complete_actions"] == 1


def test_boundary_loader_rejects_policy_or_replay_membership_drift(tmp_path: Path) -> None:
    sequence, cache, cache_sha, activation, activation_sha, policy_sha = _sealed(tmp_path)
    with pytest.raises(PrizePlanActorBoundaryError, match="identity drifted"):
        load_h3_actor_provider(
            sequences=[sequence],
            policy_checkpoint_sha256="sha256:" + "c" * 64,
            cache_receipt_path=cache,
            cache_receipt_sha256=cache_sha,
            activation_receipt_path=activation,
            activation_receipt_sha256=activation_sha,
        )
    sequence.decisions[0].env_step = 8
    with pytest.raises(PrizePlanActorBoundaryError, match="membership digest"):
        load_h3_actor_provider(
            sequences=[sequence],
            policy_checkpoint_sha256=policy_sha,
            cache_receipt_path=cache,
            cache_receipt_sha256=cache_sha,
            activation_receipt_path=activation,
            activation_receipt_sha256=activation_sha,
        )


def test_boundary_loader_rejects_nonpassing_activation(tmp_path: Path) -> None:
    sequence, cache, cache_sha, activation, _activation_sha, policy_sha = _sealed(tmp_path)
    value = json.loads(activation.read_text())
    value["actor_activation"] = False
    value.pop("artifact_sha256")
    value = _artificed(value)
    _write_json(activation, value)
    with pytest.raises(PrizePlanActorBoundaryError, match="not passing"):
        load_h3_actor_provider(
            sequences=[sequence],
            policy_checkpoint_sha256=policy_sha,
            cache_receipt_path=cache,
            cache_receipt_sha256=cache_sha,
            activation_receipt_path=activation,
            activation_receipt_sha256=_file_sha(activation),
        )
