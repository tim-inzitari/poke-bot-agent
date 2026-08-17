import json
from pathlib import Path

import pytest

from poke_bot.expert_pilot_importance import (
    PILOT_MAP_SCHEMA,
    SNAPSHOT_SCHEMA,
    TARGET_SCHEMA,
    canonical_digest,
    load_aligned_training_weights,
    load_training_weights_for_corpus,
    materialize_importance_index,
    weight_for_support,
)


def _row(index: int) -> dict[str, object]:
    return {
        "source_index": index,
        "episode_id": f"episode-{index}",
        "archetype_id": "marnie-s-grimmsnarl-ex",
        "seat": index % 2,
        "raw_decisions": 3,
        "packed_decisions": 3,
    }


def test_support_tiers_are_higher_but_bounded() -> None:
    assert [weight_for_support(value) for value in (
        0, 1, 31, 32, 127, 128, 511, 512, 1023, 1024, 2047, 2048, 9000
    )] == [
        1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 7.0, 7.0, 10.0, 10.0
    ]


def test_materializer_uses_training_support_and_leaves_validation_unweighted(
    tmp_path: Path,
) -> None:
    train = [_row(index) for index in range(33)]
    validation = [_row(100)]
    targets = {
        "schema": TARGET_SCHEMA,
        "corpus_manifest": "/corpus/manifest.json",
        "corpus_manifest_sha256": "sha256:" + "a" * 64,
        "split_seed": 20260801,
        "validation_fraction": 0.1,
        "max_context": 320,
        "train_rows": train,
        "validation_rows": validation,
    }
    pilot_map = {
        "schema": PILOT_MAP_SCHEMA,
        "rows": [
            {"episode_id": row["episode_id"], "seat": row["seat"], "team_name": "leader"}
            for row in train[:32] + validation
        ],
    }
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "rows": [
            {"rank": rank, "team_name": "leader" if rank == 1 else f"team-{rank}"}
            for rank in range(1, 101)
        ],
    }
    output = materialize_importance_index(
        targets=targets,
        pilot_map=pilot_map,
        leaderboard_snapshot=snapshot,
        targets_digest="sha256:" + "b" * 64,
        pilot_map_digest="sha256:" + "c" * 64,
        leaderboard_snapshot_digest="sha256:" + "d" * 64,
    )
    assert output["per_team_support"][0]["training_games"] == 32
    assert output["per_team_support"][0]["weight"] == 2.0
    assert output["train_game_weights"][:32] == [2.0] * 32
    assert output["train_game_weights"][32] == 1.0
    assert output["validation_unweighted"] is True

    index = tmp_path / "importance.json"
    index.write_text(json.dumps(output, sort_keys=True), encoding="utf-8")
    weights, contract = load_aligned_training_weights(
        index,
        expected_manifest_digest=targets["corpus_manifest_sha256"],
        train_identity_rows=train,
        validation_identity_rows=validation,
    )
    assert weights == output["train_game_weights"]
    assert contract["matched_top_100_train_games"] == 32

    resident_weights, resident_contract = load_training_weights_for_corpus(
        index,
        expected_manifest_digest=targets["corpus_manifest_sha256"],
        split_seed=20260801,
        validation_fraction=0.1,
        max_context=320,
        train_games=len(train),
        validation_games=len(validation),
    )
    assert resident_weights == weights
    assert resident_contract["support_partition"] == "training_only"

    reversed_rows = list(reversed(train))
    with pytest.raises(ValueError, match="misaligned"):
        load_aligned_training_weights(
            index,
            expected_manifest_digest=targets["corpus_manifest_sha256"],
            train_identity_rows=reversed_rows,
            validation_identity_rows=validation,
        )
