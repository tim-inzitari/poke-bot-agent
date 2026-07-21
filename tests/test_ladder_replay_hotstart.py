from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.dataset import BootstrapDataset, GameSequence
from poke_bot.ladder_deck_mix import (
    load_ladder_deck_mix,
    load_ladder_deck_representatives,
)
from poke_bot.ladder_replay import LadderReplayClassifier
from poke_bot.train import split_dataset
from scripts.run_top_ladder_hotstart import _sha256, _validate_dataset
from scripts.split_top_ladder_dataset import main as split_top_ladder_dataset


ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
REPS = ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"


def _sequence(episode_id: str, seat: int) -> GameSequence:
    return GameSequence(
        episode_id=episode_id,
        seat=seat,
        archetype="crustle",
        opp_archetype="alakazam",
        deck=[1] * 60,
        value=1.0,
        decisions=[],
    )


def test_every_pinned_representative_has_exact_family_label() -> None:
    mix = load_ladder_deck_mix(MIX)
    representatives = load_ladder_deck_representatives(REPS)
    classifier = LadderReplayClassifier(mix, representatives)

    assert set(classifier.active_ids) == set(representatives.decks)
    for deck_id, row in representatives.decks.items():
        label = classifier.classify_deck(row["card_ids"])
        assert label.deck_id == deck_id
        assert label.method == "representative_exact"


def test_ace_labeled_families_generalize_beyond_exact_modal_list(tmp_path: Path) -> None:
    card_csv = tmp_path / "cards.csv"
    card_csv.write_text(
        "Card ID,Card Name,HP\n"
        "140,Fezandipiti ex,210\n"
        "190,Archaludon ex,300\n"
        "648,Marnie's Grimmsnarl ex,320\n",
        encoding="utf-8",
    )
    classifier = LadderReplayClassifier.from_paths(
        MIX, REPS, card_csv=card_csv
    )
    representatives = load_ladder_deck_representatives(REPS)
    for deck_id in ("marnie-s-grimmsnarl-ex", "archaludon-ex"):
        cards = list(representatives.decks[deck_id]["card_ids"])
        # Change one trainer/basic card so this is no longer the modal multiset.
        replace_at = next(i for i, card_id in enumerate(cards) if card_id not in {140, 190, 648})
        cards[replace_at] = 999_999
        label = classifier.classify_deck(cards)
        assert label.deck_id == deck_id
        assert label.method == "derived_primary_ace"


def test_episode_grouped_split_never_leaks_other_seat() -> None:
    sequences = [
        _sequence("ep-a", 0),
        _sequence("ep-a", 1),
        _sequence("ep-b", 0),
        _sequence("ep-b", 1),
        _sequence("ep-c", 0),
        _sequence("ep-c", 1),
    ]
    train, val = split_dataset(
        BootstrapDataset(sequences), 0.34, 7, group_by_episode=True
    )
    assert train
    assert val
    assert {seq.episode_id for seq in train}.isdisjoint(
        {seq.episode_id for seq in val}
    )


def _validation_meta(dataset: Path, represented: set[str]) -> dict:
    mix = load_ladder_deck_mix(MIX)
    expected = {entry.deck_id for entry in mix.decks}
    return {
        "output_sha256": _sha256(dataset),
        "classifier": {
            "mix_artifact_sha256": mix.artifact_sha256,
            "active_deck_ids": sorted(expected),
        },
        "stats": {
            "record_archetypes": {name: 1 for name in sorted(represented)}
        },
    }


def test_dataset_gate_allows_unknown_deck_agnostic_seats(tmp_path: Path) -> None:
    dataset = tmp_path / "all.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    expected = {entry.deck_id for entry in load_ladder_deck_mix(MIX).decks}
    _validate_dataset(
        dataset,
        _validation_meta(dataset, expected | {"unknown"}),
        MIX,
    )


def test_followup_shard_can_have_partial_family_coverage(tmp_path: Path) -> None:
    dataset = tmp_path / "followup.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    represented = {"crustle", "unknown"}
    meta = _validation_meta(dataset, represented)
    _validate_dataset(
        dataset,
        meta,
        MIX,
        require_all_families=False,
    )
    with pytest.raises(RuntimeError, match="missing="):
        _validate_dataset(dataset, meta, MIX, require_all_families=True)


def test_top_ladder_split_routes_every_record_and_rebuilds_metadata(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.jsonl"
    records = [
        {
            "source": "pokemon-tcg-ai-battle-episodes-2026-07-03",
            "episode_id": "a",
            "seat": 0,
            "archetype": "crustle",
            "n_decisions": 3,
            "info_set_ok": True,
        },
        {
            "source": "pokemon-tcg-ai-battle-episodes-2026-07-04",
            "episode_id": "b",
            "seat": 1,
            "archetype": "alakazam",
            "n_decisions": 5,
            "info_set_ok": True,
        },
        {
            "source": "pokemon-tcg-ai-battle-episodes-2026-07-05",
            "episode_id": "c",
            "seat": 0,
            "archetype": "crustle",
            "n_decisions": 7,
            "info_set_ok": True,
        },
    ]
    parent.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    mix = load_ladder_deck_mix(MIX)
    parent_meta = {
        "output_sha256": _sha256(parent),
        "sources": [
            {"date": f"2026-07-0{day}", "slug": f"day-{day}"}
            for day in (3, 4, 5)
        ],
        "classifier": {
            "mix_artifact_sha256": mix.artifact_sha256,
            "active_deck_ids": [entry.deck_id for entry in mix.decks],
        },
        "policy_scope": "all_valid_top_ladder_seats",
        "quality_gates": {"passed": True},
        "stats": {"records_written": len(records)},
    }
    parent.with_suffix(".meta.json").write_text(
        json.dumps(parent_meta), encoding="utf-8"
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    assert (
        split_top_ladder_dataset(
            [
                "--dataset",
                str(parent),
                "--shard",
                "2026-07-03",
                "2026-07-04",
                str(first),
                "--shard",
                "2026-07-05",
                "2026-07-05",
                str(second),
                "--min-sequences",
                "1",
            ]
        )
        == 0
    )
    first_rows = [json.loads(line) for line in first.read_text().splitlines()]
    second_rows = [json.loads(line) for line in second.read_text().splitlines()]
    assert [row["episode_id"] for row in first_rows] == ["a", "b"]
    assert [row["episode_id"] for row in second_rows] == ["c"]
    assert len(first_rows) + len(second_rows) == len(records)
    first_meta = json.loads(first.with_suffix(".meta.json").read_text())
    second_meta = json.loads(second.with_suffix(".meta.json").read_text())
    assert first_meta["output_sha256"] == _sha256(first)
    assert second_meta["output_sha256"] == _sha256(second)
    assert first_meta["stats"]["decisions_written"] == 8
    assert second_meta["stats"]["decisions_written"] == 7


def test_top_ladder_split_quality_failure_leaves_no_final_children(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.jsonl"
    record = {
        "source": "pokemon-tcg-ai-battle-episodes-2026-07-03",
        "episode_id": "a",
        "seat": 0,
        "archetype": "unknown",
        "n_decisions": 3,
        "info_set_ok": True,
    }
    parent.write_text(json.dumps(record) + "\n", encoding="utf-8")
    mix = load_ladder_deck_mix(MIX)
    parent.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "output_sha256": _sha256(parent),
                "sources": [{"date": "2026-07-03", "slug": "day-3"}],
                "classifier": {
                    "mix_artifact_sha256": mix.artifact_sha256,
                    "active_deck_ids": [entry.deck_id for entry in mix.decks],
                },
                "quality_gates": {"passed": True},
                "stats": {"records_written": 1},
            }
        ),
        encoding="utf-8",
    )
    child = tmp_path / "child.jsonl"

    with pytest.raises(RuntimeError, match="recognized seat fraction"):
        split_top_ladder_dataset(
            [
                "--dataset",
                str(parent),
                "--shard",
                "2026-07-03",
                "2026-07-03",
                str(child),
            ]
        )

    assert not child.exists()
    assert not child.with_suffix(".meta.json").exists()
