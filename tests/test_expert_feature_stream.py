from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import weakref
from pathlib import Path

import pytest
import torch

from poke_bot import archetypes, features
from poke_bot.dataset import (
    DATASET_CACHE_SCHEMA_VERSION,
    BootstrapDataset,
    DecisionSample,
    GameSequence,
    PolicyStage,
)
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    compact_temporal_expert_sequence,
)
from poke_bot.pure_rl import expert_feature_stream
from poke_bot.pure_rl.expert_feature_stream import EpisodeGroupedFeatureManifest
from poke_bot.train import cap_history_sequences, split_dataset


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sparse(words: int, offset: int) -> features.SparseVector:
    value = features.SparseVector()
    for word in range(words):
        value.word_start()
        value.add((offset + word) % 48, 1.0)
    return value


def _game(
    episode_id: str,
    seat: int,
    *,
    decisions: int = 2,
    archetype: str = "alakazam",
) -> GameSequence:
    rows: list[DecisionSample] = []
    for index in range(decisions):
        options = _sparse(2, 3 + index)
        stage = PolicyStage(
            options=options,
            action_combos=[[0], [1]],
            target_index=1,
            guide_target_index=0,
            guide_confidence=0.75,
        )
        rows.append(
            DecisionSample(
                board=_sparse(features.NUM_BOARD_TOKENS, index),
                options=options,
                action=[1],
                action_combo_index=1,
                action_combos=[[0], [1]],
                env_step=index,
                action_token=_sparse(1, 17 + index),
                policy_stages=[stage],
                aux_labels={
                    "opp_hand": [1, 2],
                    "opp_hidden_remainder": [3, 4, 5],
                    "opp_prizes": [5],
                    "lethal_threat": 1.0,
                    "prize_race": [0.25, 0.75],
                },
            )
        )
    return GameSequence(
        episode_id=episode_id,
        seat=seat,
        archetype=archetype,
        opp_archetype="alakazam",
        deck=[1] * 60,
        value=1.0 if seat == 0 else -1.0,
        decisions=rows,
    )


def _write_manifest(
    root: Path,
    shard_sequences: list[list[GameSequence]],
) -> tuple[Path, str]:
    rows = []
    records = 0
    decisions = 0
    for shard_index, sequences in enumerate(shard_sequences):
        path = root / f"expert-{shard_index}.features"
        compacted = [compact_temporal_expert_sequence(row) for row in sequences]
        stats = {
            "records_kept": len(compacted),
            "decisions_kept": sum(len(row.decisions) for row in compacted),
        }
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "format": SHARD_FORMAT,
                    "format_version": SHARD_FORMAT_VERSION,
                    "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                    "feature_schema": features.FEATURE_SCHEMA_VERSION,
                    "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            for sequence in compacted:
                pickle.dump(sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(
                {
                    "format": SHARD_FORMAT + "-footer",
                    "format_version": SHARD_FORMAT_VERSION,
                    "stats": stats,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        rows.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "stats": stats,
            }
        )
        records += stats["records_kept"]
        decisions += stats["decisions_kept"]
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "shards": rows,
                "totals": {
                    "records_kept": records,
                    "decisions_kept": decisions,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, _sha256(manifest)


def _identity(rows: list[GameSequence]) -> list[tuple[str, int, int]]:
    return [(row.episode_id, row.seat, len(row.decisions)) for row in rows]


def test_streamed_group_split_exactly_matches_materialized_reference(
    tmp_path: Path,
) -> None:
    source = [
        _game("a", 0, decisions=4),
        _game("b", 0),
        _game("a", 1, decisions=3),
        _game("", 0),
        _game("c", 0),
        _game("d", 1),
    ]
    manifest, digest = _write_manifest(tmp_path, [source[:2], source[2:]])
    expected_capped, expected_truncated = cap_history_sequences(source, 2)
    expected_train, expected_val = split_dataset(
        BootstrapDataset(expected_capped),
        0.34,
        7,
        group_by_episode=True,
    )

    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.34,
        seed=7,
        max_context=2,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
    )
    train, val = plan.splits()

    assert plan.truncated_sequences == expected_truncated
    assert _identity(list(train)) == _identity(expected_train)
    assert _identity(list(val)) == _identity(expected_val)
    assert {row.episode_id for row in train}.isdisjoint(
        {row.episode_id for row in val}
    )
    assert _identity(list(train)) == _identity(list(train))


def test_exact_seat_view_balances_train_and_validation_without_source_rewrite(
    tmp_path: Path,
) -> None:
    source = []
    for episode in ("a", "b", "c", "d"):
        source.extend(
            [
                _game(episode, 0, decisions=2),
                _game(episode, 1, decisions=3),
                _game(episode, 1, decisions=4),
            ]
        )
    manifest, digest = _write_manifest(tmp_path, [source[:6], source[6:]])
    before = manifest.read_bytes()

    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.25,
        seed=79,
        max_context=3,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        require_exact_seat_split=True,
    )
    train, validation = plan.splits()
    train_rows = list(train)
    validation_rows = list(validation)
    evidence = plan.exact_seat_split_evidence()

    assert manifest.read_bytes() == before
    assert plan.source_sequences == 12
    assert plan.sequences == 8
    assert [row.seat for row in train_rows].count(0) == 3
    assert [row.seat for row in train_rows].count(1) == 3
    assert [row.seat for row in validation_rows].count(0) == 1
    assert [row.seat for row in validation_rows].count(1) == 1
    assert {row.episode_id for row in train_rows}.isdisjoint(
        {row.episode_id for row in validation_rows}
    )
    assert evidence["source"] == {"games": 12, "seat0": 4, "seat1": 8}
    assert evidence["selected_games"] == 8
    assert evidence["partitions"]["train"]["exact_even_split"] is True
    assert evidence["partitions"]["validation"]["exact_even_split"] is True
    assert evidence["deterministic_assignment_manifest_sha256"].startswith(
        "sha256:"
    )


def test_stream_plan_exposes_archetypes_in_exact_packed_split_order(
    tmp_path: Path,
) -> None:
    source = [
        _game("a", 0, archetype="alakazam"),
        _game("b", 0, archetype="starmie"),
        _game("a", 1, archetype="alakazam"),
        _game("c", 0, archetype="dudunsparce"),
    ]
    manifest, digest = _write_manifest(tmp_path, [source[:2], source[2:]])
    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.25,
        seed=9,
        max_context=8,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
    )
    train, validation = plan.splits()
    train_archetypes, validation_archetypes = (
        plan.partition_archetypes()
    )
    assert train_archetypes == tuple(row.archetype for row in train)
    assert validation_archetypes == tuple(
        row.archetype for row in validation
    )


def test_stream_plan_exposes_identity_rows_in_exact_packed_split_order(
    tmp_path: Path,
) -> None:
    source = [
        _game("a", 0, archetype="alakazam", decisions=2),
        _game("b", 1, archetype="starmie", decisions=3),
        _game("a", 1, archetype="alakazam", decisions=4),
        _game("c", 0, archetype="dudunsparce", decisions=5),
    ]
    manifest, digest = _write_manifest(tmp_path, [source[:2], source[2:]])
    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.25,
        seed=9,
        max_context=8,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
    )
    train, validation = plan.splits()
    train_rows, validation_rows = plan.partition_identity_rows()

    assert [
        (row["episode_id"], row["seat"], row["archetype_id"])
        for row in train_rows
    ] == [(row.episode_id, row.seat, row.archetype) for row in train]
    assert [
        (row["episode_id"], row["seat"], row["archetype_id"])
        for row in validation_rows
    ] == [
        (row.episode_id, row.seat, row.archetype)
        for row in validation
    ]
    assert [row["source_index"] for row in train_rows] == sorted(
        row["source_index"] for row in train_rows
    )


def test_stream_plan_is_reiterable_bounded_and_rejects_changed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [[_game(f"episode-{index}", index % 2) for index in range(8)]]
    manifest, digest = _write_manifest(tmp_path, source)
    source.clear()
    gc.collect()

    original = expert_feature_stream.iter_feature_shard
    loaded_refs: list[weakref.ReferenceType[GameSequence]] = []
    passes = 0

    def tracked(path: Path):
        nonlocal passes
        passes += 1
        for sequence in original(path):
            loaded_refs.append(weakref.ref(sequence))
            yield sequence

    monkeypatch.setattr(expert_feature_stream, "iter_feature_shard", tracked)
    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.25,
        seed=11,
        max_context=8,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
    )
    gc.collect()
    assert loaded_refs and all(reference() is None for reference in loaded_refs)
    assert not any(
        isinstance(value, GameSequence) for value in plan.__dict__.values()
    )

    train, _val = plan.splits()
    first = _identity(list(train))
    gc.collect()
    second = _identity(list(train))
    assert first == second
    assert passes == 3  # metadata pass plus two independent train passes

    shard = next(tmp_path.glob("*.features"))
    original_stat = shard.stat()
    changed = bytearray(shard.read_bytes())
    changed[len(changed) // 2] ^= 0x01
    shard.write_bytes(changed)
    os.utime(
        shard,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert shard.stat().st_size == original_stat.st_size
    assert shard.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(ValueError, match="changed after verification"):
        list(train)


def test_streamed_splits_preserve_temporal_and_all_head_pack_layout(
    tmp_path: Path,
) -> None:
    source = [
        _game("episode-a", 0, decisions=3),
        _game("episode-a", 1, decisions=3),
        _game("episode-b", 0, decisions=2),
        _game("episode-c", 1, decisions=2),
    ]
    manifest, digest = _write_manifest(tmp_path, [source[:2], source[2:]])
    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.25,
        seed=17,
        max_context=2,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
    )
    train, val = plan.splits()
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        train,
        val,
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )

    assert corpus.has_temporal_layout
    assert corpus.has_exact_targets
    assert corpus.train_games == len(train)
    assert corpus.val_games == len(val)
    assert corpus.decisions == 8
    assert corpus.game_decision_offset is not None
    assert corpus.game_decision_offset.tolist() == [0, 2, 4, 6, 8]
    assert corpus.action_offset is not None
    assert corpus.action_offset.numel() == corpus.decisions + 1
    assert corpus.hand_present is not None
    assert corpus.hand_present.tolist() == [1] * corpus.decisions
    assert corpus.hand_index is not None
    assert corpus.hand_index.tolist() == [1, 2] * corpus.decisions
    assert corpus.hand_offset is not None
    assert corpus.hand_offset.tolist() == list(
        range(0, 2 * corpus.decisions + 1, 2)
    )
    assert corpus.remainder_present is not None
    assert corpus.remainder_present.tolist() == [1] * corpus.decisions
    assert corpus.remainder_index is not None
    assert corpus.remainder_index.tolist() == [3, 4, 5] * corpus.decisions
    assert corpus.remainder_offset is not None
    assert corpus.remainder_offset.tolist() == list(
        range(0, 3 * corpus.decisions + 1, 3)
    )
    assert corpus.lethal_target is not None
    assert corpus.lethal_target.tolist() == [1.0] * corpus.decisions
    assert corpus.prize_race_target is not None
    assert corpus.prize_race_target.tolist() == [
        [0.25, 0.75]
    ] * corpus.decisions
    assert corpus.sample_aux_class is not None
    alakazam = archetypes.archetype_ids().index("alakazam")
    assert corpus.sample_aux_class.tolist() == [alakazam] * corpus.total_samples
    assert corpus.guide_target_index is not None
    assert corpus.guide_target_index.tolist() == [0] * corpus.total_samples
