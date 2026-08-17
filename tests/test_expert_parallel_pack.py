from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import pytest
import torch

from poke_bot import features
from poke_bot.dataset import (
    DATASET_CACHE_SCHEMA_VERSION,
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
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.pure_rl.expert_cpu_pack import (
    ExpertCpuPackCache,
    ExpertCpuPackKey,
)
from poke_bot.pure_rl.expert_feature_stream import (
    EpisodeGroupedFeatureManifest,
)
from poke_bot.pure_rl import expert_parallel_pack
from poke_bot.pure_rl.expert_parallel_pack import (
    _load_fragment,
    _write_fragment,
    build_parallel_expert_cpu_pack,
)
from poke_bot.pure_rl.expert_rehearsal import (
    ExpertManifestIdentity,
    ResidentExpertCorpusCache,
)
from poke_bot.strategic_heads import (
    ACTION_UTILITY_NAMES,
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    EXPANDED_STRATEGIC_SCHEMA_VERSION,
    OPPONENT_RESPONSE_NAMES,
    RESOURCE_FORECAST_NAMES,
    TACTICAL_HORIZONS,
    TACTICAL_OUTCOME_NAMES,
    masked_expanded_strategic_coverage,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sparse(words: int, offset: int) -> features.SparseVector:
    vector = features.SparseVector()
    for word in range(words):
        vector.word_start()
        vector.add((offset + word) % 48, 1.0 + word / 100.0)
        if word % 2:
            vector.add((offset + word + 7) % 48, 0.25)
    return vector


def _expanded(stage_count: int) -> dict:
    utility = [float(index + 1) for index in range(len(ACTION_UTILITY_NAMES))]
    utility[-1] = 1.0
    tactical = [
        [
            float(10 * horizon + column)
            for column in range(len(TACTICAL_OUTCOME_NAMES))
        ]
        for horizon in range(len(TACTICAL_HORIZONS))
    ]
    for row in tactical:
        row[2] = 1.0
        row[3] = 0.0
    resources = [
        float(index) for index in range(len(RESOURCE_FORECAST_NAMES))
    ]
    resources[4] = 1.0
    resources[5] = 0.0
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "version": EXPANDED_STRATEGIC_SCHEMA_VERSION,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "action_factors": [
            {
                "action_type": index % 2 == 0,
                "target": index % 2 == 1,
                "resource": index == stage_count - 1,
            }
            for index in range(stage_count)
        ],
        "action_utility": {
            "values": utility,
            "mask": [True] * len(ACTION_UTILITY_NAMES),
        },
        "tactical_outcomes": {
            "values": tactical,
            "mask": [
                [True] * len(TACTICAL_OUTCOME_NAMES)
                for _ in TACTICAL_HORIZONS
            ],
        },
        "opponent_response": {
            "values": [
                float(index % 2)
                for index in range(len(OPPONENT_RESPONSE_NAMES))
            ],
            "mask": [True] * len(OPPONENT_RESPONSE_NAMES),
        },
        "resource_forecast": {
            "values": resources,
            "mask": [True] * len(RESOURCE_FORECAST_NAMES),
        },
        "game_phase": 2,
        "outcome_class": 2,
        "remaining_turns_log1p": 1.25,
        "provenance": {
            "trajectory": "full_untruncated_same_seat",
            "transition_after": "explicit_public_post_action",
            "terminal_complete": True,
            "target_only": True,
        },
    }


def _game(
    episode_id: str,
    seat: int,
    *,
    decisions: int,
    strategic: bool,
    zero_sample_decision: bool = False,
    all_zero_sample: bool = False,
    strategic_none: bool = False,
) -> GameSequence:
    rows: list[DecisionSample] = []
    for index in range(decisions):
        if all_zero_sample or (
            zero_sample_decision and index == decisions - 1
        ):
            stages = [
                PolicyStage(
                    options=features.SparseVector(),
                    action_combos=[],
                    target_index=-1,
                )
            ]
        else:
            stages = [
                PolicyStage(
                    options=_sparse(2, 3 + index),
                    action_combos=[[0], [1]],
                    target_index=1,
                    guide_target_index=0,
                    guide_confidence=0.75,
                )
            ]
            if index % 2 == 0:
                stages.append(
                    PolicyStage(
                        options=_sparse(3, 11 + index),
                        action_combos=[[0], [1], [2]],
                        target_index=2,
                        guide_target_index=1,
                        guide_confidence=0.5,
                    )
                )
        options = stages[0].options
        aux = {
            "opp_hand": [] if index % 2 == 0 else [1, 2],
            "opp_hidden_remainder": [3, 4, 5],
            "opp_prizes": [5],
            "lethal_threat": 1.0 if index % 2 == 0 else 0.0,
            "prize_race": [0.25, 0.75],
        }
        if strategic:
            aux[EXPANDED_STRATEGIC_KEY] = _expanded(len(stages))
        elif strategic_none:
            aux[EXPANDED_STRATEGIC_KEY] = None
        rows.append(
            DecisionSample(
                board=_sparse(features.NUM_BOARD_TOKENS, index + seat),
                options=options,
                action=[1],
                action_combo_index=max(0, stages[0].target_index),
                action_combos=stages[0].action_combos,
                env_step=index,
                action_token=_sparse(1, 19 + index),
                policy_stages=stages,
                aux_labels=aux,
            )
        )
    return GameSequence(
        episode_id=episode_id,
        seat=seat,
        archetype="alakazam",
        opp_archetype="alakazam",
        deck=[1] * 60,
        value=1.0,
        decisions=rows,
    )


def _write_manifest(
    root: Path,
    shards: list[list[GameSequence]],
    *,
    expanded: bool,
) -> tuple[Path, str]:
    shard_rows = []
    records = 0
    decisions = 0
    for ordinal, source in enumerate(shards):
        path = root / f"expert-{ordinal}.features"
        compacted = [
            compact_temporal_expert_sequence(sequence) for sequence in source
        ]
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
                pickle.dump(
                    sequence, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            pickle.dump(
                {
                    "format": SHARD_FORMAT + "-footer",
                    "format_version": SHARD_FORMAT_VERSION,
                    "stats": stats,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        shard_rows.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "stats": stats,
            }
        )
        records += stats["records_kept"]
        decisions += stats["decisions_kept"]
    payload = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
        "shards": shard_rows,
        "totals": {
            "records_kept": records,
            "decisions_kept": decisions,
        },
    }
    if not expanded:
        # Modern manifests retain the strategic schema even when every row is
        # masked. Presence of the metadata alone must not materialize targets.
        payload["expanded_strategic_targets"] = (
            masked_expanded_strategic_coverage(decisions)
        )
    # The expanded=True fixture deliberately omits the optional metadata.
    # Actual retained labels, rather than metadata truthiness, must determine
    # the global tensor layout.
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return manifest, _sha256(manifest)


def _fixture(root: Path, *, expanded: bool = True) -> tuple[Path, str]:
    return _write_manifest(
        root,
        [
            [
                _game(
                    "cross-shard",
                    0,
                    decisions=3,
                    strategic=expanded,
                    strategic_none=not expanded,
                ),
                _game("train-a", 0, decisions=2, strategic=False),
                _game("", 0, decisions=2, strategic=False),
            ],
            [
                _game(
                    "middle",
                    0,
                    decisions=3,
                    strategic=False,
                    zero_sample_decision=True,
                ),
                # Exceeds the fixture's max_context=8. The raw manifest count
                # and packed decision count intentionally differ.
                _game("train-b", 1, decisions=10, strategic=False),
            ],
            [
                _game("cross-shard", 1, decisions=2, strategic=False),
                _game("", 1, decisions=1, strategic=False),
                _game("last", 0, decisions=2, strategic=False),
            ],
            [
                _game(
                    "zero-sample-game",
                    0,
                    decisions=1,
                    strategic=False,
                    all_zero_sample=True,
                ),
            ],
        ],
        expanded=expanded,
    )


def _plan(
    manifest: Path,
    digest: str,
    *,
    workers: int,
) -> EpisodeGroupedFeatureManifest:
    return EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=0.34,
        seed=17,
        max_context=8,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        workers=workers,
    )


def _assert_same(
    actual: DeviceResidentBootstrapCorpus,
    expected: DeviceResidentBootstrapCorpus,
) -> None:
    assert set(actual.tensor_state()) == set(expected.tensor_state())
    for name, wanted in expected.tensor_state().items():
        observed = actual.tensor_state()[name]
        assert observed.dtype == wanted.dtype, name
        assert observed.shape == wanted.shape, name
        assert torch.equal(
            observed.contiguous().view(torch.uint8),
            wanted.contiguous().view(torch.uint8),
        ), f"{name} is not bitwise identical"
        if observed.is_floating_point():
            torch.testing.assert_close(
                observed,
                wanted,
                rtol=0,
                atol=0,
                equal_nan=True,
                msg=name,
            )
        else:
            assert torch.equal(observed, wanted), name
    expected_scalars = expected.scalar_state()
    actual_scalars = actual.scalar_state()
    expected_scalars.pop("build_seconds")
    actual_scalars.pop("build_seconds")
    assert actual_scalars == expected_scalars
    validate_cpu_corpus(actual)


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_pack_is_tensor_exact_to_serial(
    tmp_path: Path,
    workers: int,
) -> None:
    manifest, digest = _fixture(tmp_path)
    serial_plan = _plan(manifest, digest, workers=1)
    assert serial_plan.decisions == 26
    assert serial_plan.packed_decisions == 24
    assert serial_plan.truncated_sequences == 1
    train, validation = serial_plan.splits()
    serial = DeviceResidentBootstrapCorpus.from_splits(
        train,
        validation,
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    parallel_plan = _plan(manifest, digest, workers=workers)
    parallel = build_parallel_expert_cpu_pack(
        parallel_plan,
        workers=workers,
        exact_card_vocab=64,
        spool_root=tmp_path / f"spool-{workers}",
        memory_reserve_gib=0,
        disk_reserve_gib=0,
    )
    _assert_same(parallel, serial)
    assert not list((tmp_path / f"spool-{workers}").glob(".parallel-pack-*"))


def test_parallel_pack_without_strategic_labels_stays_absent(
    tmp_path: Path,
) -> None:
    manifest, digest = _fixture(tmp_path, expanded=False)
    serial_plan = _plan(manifest, digest, workers=1)
    serial = DeviceResidentBootstrapCorpus.from_splits(
        *serial_plan.splits(),
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    parallel_plan = _plan(manifest, digest, workers=2)
    parallel = build_parallel_expert_cpu_pack(
        parallel_plan,
        workers=2,
        exact_card_vocab=64,
        spool_root=tmp_path / "spool",
        memory_reserve_gib=0,
        disk_reserve_gib=0,
    )
    assert serial.has_expanded_strategic_targets is False
    assert parallel.has_expanded_strategic_targets is False
    _assert_same(parallel, serial)


def test_parallel_pack_failure_cleans_spool_and_commits_nothing(
    tmp_path: Path,
) -> None:
    manifest, digest = _fixture(tmp_path)
    plan = _plan(manifest, digest, workers=2)
    shard = tmp_path / "expert-1.features"
    original = shard.stat()
    changed = bytearray(shard.read_bytes())
    changed[len(changed) // 2] ^= 1
    shard.write_bytes(changed)
    os.utime(shard, ns=(original.st_atime_ns, original.st_mtime_ns))
    spool = tmp_path / "spool"

    cache = ExpertCpuPackCache(spool)
    key = ExpertCpuPackKey(
        manifest_digest=digest,
        split_seed=17,
        val_frac=0.34,
        max_context=8,
        belief_card_vocab=64,
    )
    with pytest.raises(ValueError, match="changed after verification"):
        cache.load_or_build(
            key,
            lambda: build_parallel_expert_cpu_pack(
                plan,
                workers=2,
                exact_card_vocab=64,
                spool_root=spool,
                memory_reserve_gib=0,
                disk_reserve_gib=0,
            ),
        )

    assert not list(spool.glob(".parallel-pack-*"))
    assert not list(spool.glob("expert-pack-*"))
    assert not (spool / "active.json").exists()


def test_parallel_pack_cache_hit_does_not_reopen_source(
    tmp_path: Path,
) -> None:
    manifest, digest = _fixture(tmp_path)
    identity = ExpertManifestIdentity(
        path=str(manifest),
        digest=digest,
        dates=("2026-07-25",),
        decisions=26,
        records=9,
    )
    root = tmp_path / "cache"
    first = ResidentExpertCorpusCache(cpu_pack_root=root)
    corpus = first.prepare(
        identity,
        device=torch.device("cpu"),
        seed=17,
        val_frac=0.34,
        max_context=8,
        belief_card_vocab=64,
        pack_workers=2,
        pack_memory_reserve_gib=0,
        pack_disk_reserve_gib=0,
    )
    validate_cpu_corpus(corpus)
    assert first.pack_info is not None
    assert first.pack_info["cache_hit"] is False
    first.release()

    manifest.unlink()
    restarted = ResidentExpertCorpusCache(cpu_pack_root=root)
    cached = restarted.prepare(
        identity,
        device=torch.device("cpu"),
        seed=17,
        val_frac=0.34,
        max_context=8,
        belief_card_vocab=64,
        pack_workers=4,
        pack_memory_reserve_gib=0,
        pack_disk_reserve_gib=0,
    )
    validate_cpu_corpus(cached)
    assert restarted.pack_info is not None
    assert restarted.pack_info["cache_hit"] is True


def test_parallel_fragment_checksum_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    game = _game("one", 0, decisions=2, strategic=True)
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [game],
        [],
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    descriptor = _write_fragment(
        tmp_path,
        ordinal=0,
        partition="train",
        corpus=corpus,
    )
    payload = Path(descriptor.payload)
    changed = bytearray(payload.read_bytes())
    changed[len(changed) // 2] ^= 1
    payload.write_bytes(changed)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _load_fragment(descriptor)


def test_parallel_merge_checks_memory_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = _game("one", 0, decisions=2, strategic=True)
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [game],
        [],
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    descriptor = _write_fragment(
        tmp_path,
        ordinal=0,
        partition="train",
        corpus=corpus,
    )
    monkeypatch.setattr(
        expert_parallel_pack,
        "_available_memory_bytes",
        lambda: 1,
    )
    monkeypatch.setattr(
        expert_parallel_pack,
        "_allocate_merged",
        lambda _specs: pytest.fail("allocation preceded memory guard"),
    )

    with pytest.raises(
        MemoryError,
        match="cannot preserve its memory reserve",
    ):
        expert_parallel_pack._merge_fragments(
            [descriptor],
            build_seconds=0.0,
            memory_reserve_gib=0.0,
        )
