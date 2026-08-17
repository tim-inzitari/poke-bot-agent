from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from array import array

import pytest

from poke_bot import features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)
from poke_bot.guide2vec_data import (
    R212_PARTITION_DATES,
    R212_QUARANTINE_IDENTITIES,
    R212_QUARANTINE_IDENTITIES_SHA256,
    R212_SOURCE_DATES,
    build_r212_guide2vec_split,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    result = features.SparseVector()
    for index in range(words):
        result.word_start()
        result.add((index + offset) % 32, 1.0)
    return result


def _sequence(
    episode_id: str,
    seat: int,
    deck_card: int,
    *,
    teacher_compatible: bool = True,
    guide_label: bool = True,
) -> GameSequence:
    options = _sparse(2, offset=4)
    stage = PolicyStage(
        options=options,
        action_combos=[[0], [1]],
        target_index=1,
        guide_target_index=0 if guide_label else -1,
        guide_confidence=0.5 if guide_label else 0.0,
    )
    decision = DecisionSample(
        board=_sparse(3),
        options=options,
        action=[1],
        action_combo_index=1,
        action_combos=[[0], [1]],
        env_step=0,
        policy_stages=[stage],
    )
    return GameSequence(
        episode_id=episode_id,
        seat=seat,
        archetype="alakazam",
        opp_archetype="alakazam",
        # Temporal-expert shards compact the deck into an ``array('I')``.
        deck=array(
            "I",
            (
                [741, 741, 742, 742, 743, 743] + [deck_card] * 54
                if teacher_compatible
                else [deck_card] * 60
            ),
        ),
        value=1.0 if seat == 0 else -1.0,
        decisions=[decision],
    )


def _partition_for(day: str) -> str:
    return next(
        name for name, dates in R212_PARTITION_DATES.items() if day in dates
    )


def _write_source(
    root: Path,
    *,
    episode_for_day=lambda day: f"episode-{day}",
    deck_for_day=lambda day: {"train": 1, "validation": 2, "test": 3}[
        _partition_for(day)
    ],
    extra_july17_sequences: tuple[GameSequence, ...] = (),
) -> Path:
    rows: list[dict] = []
    totals = {"records_kept": 0, "decisions_kept": 0, "guide_rows": 0}
    for index, day in enumerate(R212_SOURCE_DATES):
        sequences = [_sequence(episode_for_day(day), index % 2, deck_for_day(day))]
        # Ensure a same-game seat sibling remains grouped without being treated
        # as a cross-partition duplicate.
        if day == R212_SOURCE_DATES[0]:
            sequences.append(_sequence(episode_for_day(day), 1, deck_for_day(day)))
        if day == "2026-07-17":
            sequences.extend(
                _sequence(
                    episode_id,
                    seat,
                    99,
                    teacher_compatible=False,
                    guide_label=False,
                )
                for _source_date, episode_id, seat in R212_QUARANTINE_IDENTITIES
            )
            sequences.extend(extra_july17_sequences)
        path = root / f"alakazam-{day}.features"
        stats = {
            "records_kept": len(sequences),
            "decisions_kept": len(sequences),
            "target_coverage": {
                "guide_rows": sum(
                    int(stage.guide_target_index >= 0)
                    for sequence in sequences
                    for decision in sequence.decisions
                    for stage in decision.policy_stages
                )
            },
        }
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "format": SHARD_FORMAT,
                    "format_version": SHARD_FORMAT_VERSION,
                    "dataset_schema": 8,
                    "feature_schema": features.FEATURE_SCHEMA_VERSION,
                    "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                    "required_archetype": "alakazam",
                    "source_dates": [day],
                    "max_context": 320,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            for sequence in sequences:
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
                "bytes": path.stat().st_size,
                "stats": stats,
            }
        )
        totals["records_kept"] += stats["records_kept"]
        totals["decisions_kept"] += stats["decisions_kept"]
        totals["guide_rows"] += stats["target_coverage"]["guide_rows"]
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "date_start": R212_SOURCE_DATES[0],
                "date_end": R212_SOURCE_DATES[-1],
                "dates": list(R212_SOURCE_DATES),
                "selection": {
                    "value": "alakazam",
                    "seat_semantics": "acting_seat_only",
                    "opponent_routes_only": False,
                },
                "quality_gates": {
                    "passed": True,
                    "checksummed": True,
                    "acting_seat_archetype_exact": True,
                    "hidden_targets_are_aux_only": True,
                    "temporal_action_tokens_complete": True,
                },
                "shards": rows,
                "totals": {
                    "records_kept": totals["records_kept"],
                    "decisions_kept": totals["decisions_kept"],
                    "target_coverage": {"guide_rows": totals["guide_rows"]},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pointer = root / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": manifest.name,
                "manifest_sha256": _sha256(manifest),
                "totals": {
                    "records_kept": totals["records_kept"],
                    "decisions_kept": totals["decisions_kept"],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return pointer


def test_r212_split_is_fixed_by_day_and_whole_episode(tmp_path: Path) -> None:
    split = build_r212_guide2vec_split(_write_source(tmp_path))

    manifest = split.manifest()
    assert (
        manifest["partition_policy"]["cross_partition_episode_overlap_policy"]
        == "fail_closed"
    )
    assert manifest["partitions"]["train"]["records"] == 17
    assert manifest["partitions"]["validation"]["records"] == 2
    assert manifest["partitions"]["test"]["records"] == 2
    assert manifest["partitions"]["train"]["whole_game_groups"] == 16
    assert manifest["partitions"]["train"]["exact_deck_multisets"] == 1
    assert manifest["partitions"]["train"]["source_records"] == 28
    assert manifest["partitions"]["train"]["quarantined_records"] == 11
    assert manifest["partitions"]["train"]["quarantined_decisions"] == 11
    assert manifest["accounting"]["source"]["records"] == 32
    assert manifest["accounting"]["retained"]["records"] == 21
    assert manifest["accounting"]["quarantined"] == {
        "records": 11,
        "decisions": 11,
        "guide_rows": 0,
    }
    assert manifest["quarantine"]["identities_sha256"] == R212_QUARANTINE_IDENTITIES_SHA256
    assert manifest["quarantine"]["identity_count"] == 11
    assert [
        (row["source_date"], row["episode_id"], row["seat"])
        for row in manifest["quarantine"]["identities"]
    ] == list(R212_QUARANTINE_IDENTITIES)
    assert len(list(split.iter_partition("train"))) == 17
    assert [row.episode_id for row in split.iter_partition("validation")] == [
        "episode-2026-07-20",
        "episode-2026-07-21",
    ]


def test_r212_split_rejects_episode_crossing_partitions(tmp_path: Path) -> None:
    pointer = _write_source(
        tmp_path,
        episode_for_day=lambda day: (
            "same-episode" if day in {"2026-07-04", "2026-07-20"} else f"episode-{day}"
        ),
    )
    with pytest.raises(ValueError, match="episode crosses date partitions"):
        build_r212_guide2vec_split(pointer)


def test_r212_split_records_exact_deck_crossing_partitions(tmp_path: Path) -> None:
    pointer = _write_source(
        tmp_path,
        deck_for_day=lambda day: 1 if day in {"2026-07-04", "2026-07-20"} else {
            "train": 2,
            "validation": 3,
            "test": 4,
        }[_partition_for(day)],
    )
    manifest = build_r212_guide2vec_split(pointer).manifest()

    overlap = manifest["overlap_checks"]
    assert overlap["passed"] is True
    assert overlap["exact_deck_multisets_partition_disjoint"] is False
    assert overlap["cross_partition_deck_fingerprint_policy"] == "record_only"
    assert len(overlap["cross_partition_deck_fingerprints"]) == 1
    fingerprint = overlap["cross_partition_deck_fingerprints"][0]
    assert fingerprint in manifest["partitions"]["train"]["deck_fingerprints"]
    assert fingerprint in manifest["partitions"]["validation"]["deck_fingerprints"]


def test_r212_split_enforces_an_explicit_deck_authorization(tmp_path: Path) -> None:
    pointer = _write_source(tmp_path)
    plan = build_r212_guide2vec_split(pointer)
    accepted = {
        fingerprint
        for partition in plan.manifest()["partitions"].values()
        for fingerprint in partition["deck_fingerprints"]
    }

    assert build_r212_guide2vec_split(
        pointer,
        authorized_deck_fingerprints=accepted,
    ).manifest()["partition_policy"]["authorized_deck_fingerprints"] == sorted(accepted)
    with pytest.raises(ValueError, match="deck fingerprint is not authorized"):
        build_r212_guide2vec_split(
            pointer,
            authorized_deck_fingerprints={"sha256:" + "0" * 64},
        )


def test_r212_split_rejects_unexpected_zero_guide_incompatible_record(
    tmp_path: Path,
) -> None:
    pointer = _write_source(
        tmp_path,
        extra_july17_sequences=(
            _sequence(
                "unexpected-zero-guide-incompatible",
                0,
                77,
                teacher_compatible=False,
                guide_label=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside the r212 quarantine"):
        build_r212_guide2vec_split(pointer)


def test_r212_split_rejects_incompatible_record_with_positive_guide_label(
    tmp_path: Path,
) -> None:
    pointer = _write_source(
        tmp_path,
        extra_july17_sequences=(
            _sequence(
                "unexpected-labelled-incompatible",
                0,
                78,
                teacher_compatible=False,
                guide_label=True,
            ),
        ),
    )

    with pytest.raises(ValueError, match="carries positive guide labels"):
        build_r212_guide2vec_split(pointer)


def test_r212_split_rejects_a_changed_shard_before_unpickling(tmp_path: Path) -> None:
    pointer = _write_source(tmp_path)
    shard = tmp_path / "alakazam-2026-07-04.features"
    changed = bytearray(shard.read_bytes())
    changed[-1] ^= 0x01
    shard.write_bytes(changed)

    with pytest.raises(ValueError, match="feature shard digest mismatch"):
        build_r212_guide2vec_split(pointer)
