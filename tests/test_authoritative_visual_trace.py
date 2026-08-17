from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from poke_bot.authoritative_visual_trace import (
    TARGET_CONSUMER_CONTRACT,
    VisualTraceError,
    convert_visual_episode,
    materialize_day,
)
from poke_bot.feature_shards import (
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot import features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    TARGET_SCHEMA_DIGEST,
    validate_expanded_strategic_labels,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


@dataclass(frozen=True)
class _Label:
    deck_id: str
    method: str = "representative_exact"


class _Classifier:
    """Small pickle-safe stand-in for ``LadderReplayClassifier``."""

    def __init__(
        self,
        deck_ids: tuple[str, str] = ("alakazam", "crustle"),
    ) -> None:
        self.deck_ids = deck_ids

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "format": "synthetic-ladder-classifier",
            "format_version": 1,
            "active_deck_ids": list(self.deck_ids),
        }

    def classify_episode(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[list[int]], list[_Label]]:
        actions = payload["steps"][1]
        decks = [list(actions[seat]["action"]) for seat in (0, 1)]
        return decks, [_Label(deck_id) for deck_id in self.deck_ids]


def test_schema_7_protected_feature_shards_remain_readable(tmp_path: Path) -> None:
    shard = tmp_path / "protected-schema7.features"
    with shard.open("wb") as handle:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": 7,
                "feature_schema": features.FEATURE_SCHEMA_VERSION,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "stats": {"records_kept": 0},
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    assert list(iter_feature_shard(shard)) == []


def test_schema_6_expert_shard_masks_absent_setup_metadata(
    tmp_path: Path,
) -> None:
    sparse = features.SparseVector()
    sparse.word_start()
    sparse.add(0, 1.0)
    stage = PolicyStage(
        options=sparse,
        action_combos=[[0]],
        target_index=0,
    )
    del stage.select_context
    del stage.selected_is_stop
    decision = DecisionSample(
        board=sparse,
        options=sparse,
        action=[0],
        action_combo_index=0,
        action_combos=[[0]],
        env_step=0,
        policy_stages=[stage],
    )
    sequence = GameSequence(
        episode_id="schema6",
        seat=0,
        archetype="archaludon-ex",
        opp_archetype="baseline",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
    )
    shard = tmp_path / "protected-schema6.features"
    with shard.open("wb") as handle:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": 6,
                "feature_schema": features.FEATURE_SCHEMA_VERSION,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        pickle.dump(sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "stats": {"records_kept": 1},
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    loaded = list(iter_feature_shard(shard))
    loaded_stage = loaded[0].decisions[0].policy_stages[0]
    assert loaded_stage.select_context == -1
    assert loaded_stage.selected_is_stop is False


def _card(card_id: int, seat: int) -> dict[str, int]:
    # The two submitted decks use disjoint ids, so ids are also stable serials.
    return {"id": card_id, "serial": card_id, "playerIndex": seat}


def _player(
    seat: int,
    deck_ids: list[int],
    *,
    hand_ids: list[int],
    prize_ids: list[int],
) -> dict[str, Any]:
    hidden = set(hand_ids) | set(prize_ids)
    remaining = [card_id for card_id in deck_ids if card_id not in hidden]
    assert len(remaining) + len(hand_ids) + len(prize_ids) == 60
    return {
        "active": [],
        "asleep": False,
        "bench": [],
        "benchMax": 5,
        "burned": False,
        "confused": False,
        "deck": [_card(card_id, seat) for card_id in remaining],
        "deckCount": len(remaining),
        "discard": [],
        "hand": [_card(card_id, seat) for card_id in hand_ids],
        "handCount": len(hand_ids),
        "paralyzed": False,
        "poisoned": False,
        "prize": [_card(card_id, seat) for card_id in prize_ids],
    }


def _full_state(
    deck0: list[int],
    deck1: list[int],
    *,
    actor: int,
    turn: int,
    hand0: list[int],
    hand1: list[int],
    prize0: list[int],
    prize1: list[int],
) -> dict[str, Any]:
    return {
        "energyAttached": False,
        "firstPlayer": 0,
        "looking": [],
        "lookingCount": 0,
        "players": [
            _player(0, deck0, hand_ids=hand0, prize_ids=prize0),
            _player(1, deck1, hand_ids=hand1, prize_ids=prize1),
        ],
        "result": -1,
        "retreated": False,
        "stadium": [],
        "stadiumPlayed": False,
        "supporterPlayed": False,
        "turn": turn,
        "turnActionCount": 0,
        "yourIndex": actor,
    }


def _masked_observation(
    state: dict[str, Any],
    *,
    actor: int,
    step: int,
    decision: bool,
) -> dict[str, Any]:
    current = copy.deepcopy(state)
    current["yourIndex"] = actor
    for seat, player in enumerate(current["players"]):
        player.pop("deck", None)
        player["prize"] = [None] * len(player["prize"])
        if seat != actor:
            player["hand"] = None
    select = (
        {
            "context": "IsFirst",
            "contextCard": None,
            "deck": None,
            "effect": None,
            "maxCount": 1,
            "minCount": 1,
            "option": [{"type": "Yes"}, {"type": "No"}],
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "type": "YesNo",
        }
        if decision
        else None
    )
    return {
        "current": current,
        "logs": [],
        "remainingOverageTime": 600,
        "select": select,
        "step": step,
    }


def _entry(
    *,
    action: list[int] | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action": list(action or []),
        "info": {},
        "observation": copy.deepcopy(observation or {}),
        "reward": 0,
        "status": "ACTIVE" if observation else "INACTIVE",
    }


def _synthetic_episode() -> dict[str, Any]:
    """Return a transition-aligned Kaggle episode with two learner decisions.

    ``visual[i].current`` is the full state *after* ``visual[i].action``.
    Therefore the exact private target for the action stored in ``visual[i]``
    is ``visual[i - 1].current``.  State C deliberately changes the opponent's
    hidden hand and deck so an off-by-one/look-ahead implementation is visible.
    """

    deck0 = list(range(1, 61))
    deck1 = list(range(101, 161))
    prize0 = list(range(3, 9))
    prize1 = list(range(103, 109))

    # Setup action produces a complete, not-yet-dealt state.
    state_seed = _full_state(
        deck0,
        deck1,
        actor=0,
        turn=0,
        hand0=[],
        hand1=[],
        prize0=[],
        prize1=[],
    )
    # An automatic transition deals the private zones; seat 1 acts next.
    state_a = _full_state(
        deck0,
        deck1,
        actor=1,
        turn=1,
        hand0=[1, 2],
        hand1=[101, 102],
        prize0=prize0,
        prize1=prize1,
    )
    # The seat-1 decision is present to prove acting-seat-only filtering.
    state_b = _full_state(
        deck0,
        deck1,
        actor=0,
        turn=2,
        hand0=[1, 2],
        hand1=[101, 102],
        prize0=prize0,
        prize1=prize1,
    )
    # The first Alakazam action changes the opponent's hidden zones (Iono-like).
    state_c = _full_state(
        deck0,
        deck1,
        actor=0,
        turn=2,
        hand0=[1, 2],
        hand1=[109, 110],
        prize0=prize0,
        prize1=prize1,
    )
    state_d = copy.deepcopy(state_c)

    setup_obs = _masked_observation(
        state_seed,
        actor=0,
        step=0,
        decision=False,
    )
    automatic_obs = _masked_observation(
        state_seed,
        actor=0,
        step=1,
        decision=False,
    )
    opponent_obs = _masked_observation(
        state_a,
        actor=1,
        step=2,
        decision=True,
    )
    learner_obs_a = _masked_observation(
        state_b,
        actor=0,
        step=3,
        decision=True,
    )
    learner_obs_b = _masked_observation(
        state_c,
        actor=0,
        step=4,
        decision=True,
    )

    visual = [
        {
            "action": [deck0, deck1],
            "current": state_seed,
            "logs": [],
            "obs": setup_obs,
            "select": setup_obs["select"],
            "selected": None,
            "ver": "synthetic-v1",
        },
        {
            "action": [[], []],
            "current": state_a,
            "logs": [],
            "obs": automatic_obs,
            "select": automatic_obs["select"],
            "selected": None,
            "ver": "synthetic-v1",
        },
        {
            "action": [[], [0]],
            "current": state_b,
            "logs": [],
            "obs": opponent_obs,
            "select": opponent_obs["select"],
            "selected": [0],
            "ver": "synthetic-v1",
        },
        {
            "action": [[0], []],
            "current": state_c,
            "logs": [],
            "obs": learner_obs_a,
            "select": learner_obs_a["select"],
            "selected": [0],
            "ver": "synthetic-v1",
        },
        {
            "action": [[1], []],
            "current": state_d,
            "logs": [],
            "obs": learner_obs_b,
            "select": learner_obs_b["select"],
            "selected": [1],
            "ver": "synthetic-v1",
        },
    ]

    steps = [[_entry(), _entry()] for _ in range(len(visual) + 1)]
    observations = [
        (0, setup_obs),
        (0, automatic_obs),
        (1, opponent_obs),
        (0, learner_obs_a),
        (0, learner_obs_b),
    ]
    for index, (actor, observation) in enumerate(observations):
        steps[index][actor] = _entry(observation=observation)
    for index, item in enumerate(visual):
        for seat in (0, 1):
            steps[index + 1][seat]["action"] = list(item["action"][seat])
    steps[0][0]["visualize"] = copy.deepcopy(visual)

    return {
        "id": "synthetic-authoritative-episode",
        "info": {
            "EpisodeId": 42,
            "TeamNames": ["Alakazam Expert", "Crustle Expert"],
        },
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": steps,
    }


def _ids(cards: list[Any]) -> list[int]:
    return [int(card["id"] if isinstance(card, dict) else card) for card in cards]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def test_exact_transition_targets_are_masked_and_alakazam_acting_seat_only() -> None:
    result = convert_visual_episode(
        _synthetic_episode(),
        _Classifier(),
        source="pokemon-tcg-ai-battle-episodes-2026-07-20",
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record["seat"] == 0
    assert record["archetype"] == "alakazam"
    assert record["opp_archetype"] == "crustle"
    assert record["value"] == 1.0
    assert len(record["steps"]) == 2

    first, second = record["steps"]
    assert first["action"] == [0]
    assert second["action"] == [1]

    # The first target is state B, immediately before the action.  State C is
    # the post-action visual state and must only become the next target.
    assert _ids(first["aux_labels"]["opp_hand"]) == [101, 102]
    assert _ids(second["aux_labels"]["opp_hand"]) == [109, 110]
    assert _ids(first["aux_labels"]["opp_prizes"]) == list(range(103, 109))
    assert _ids(second["aux_labels"]["opp_prizes"]) == list(range(103, 109))
    assert 109 in _ids(first["aux_labels"]["opp_deck_order"])
    assert 101 in _ids(second["aux_labels"]["opp_deck_order"])
    for step in record["steps"]:
        aux = step["aux_labels"]
        assert aux["opp_hidden_remainder"] == (
            aux["opp_hand"] + aux["opp_deck_order"] + aux["opp_prizes"]
        )
        assert aux["acting_archetype"] == "alakazam"
        assert _ids(aux["own_prizes"]) == list(range(3, 9))
        assert aux["prize_race"] == pytest.approx([1.0, 1.0])
        assert aux["lethal_threat"] == 0.0
        validate_expanded_strategic_labels(
            aux[EXPANDED_STRATEGIC_KEY]
        )
        assert "transition_after" not in step

    strategic_contract = record["target_provenance"][
        "expanded_strategic_targets"
    ]
    assert strategic_contract["schema"] == EXPANDED_STRATEGIC_SCHEMA
    assert strategic_contract["digest"] == TARGET_SCHEMA_DIGEST
    assert strategic_contract["decisions"] == len(record["steps"])

    assert TARGET_CONSUMER_CONTRACT["stored_without_loss"] == [
        "acting_archetype",
        "own_prizes",
    ]
    assert set(TARGET_CONSUMER_CONTRACT["loss_wired"]) == {
        "opp_archetype",
        "opp_hand",
            "opp_hidden_remainder",
            "lethal_threat",
            "prize_race",
            "current_deck_guide",
            "combo_state",
        }

    # Exact private zones are target-only.  The policy/value observation stays
    # byte-for-byte faithful to the acting seat's masked Kaggle observation.
    for step in record["steps"]:
        opponent = step["observation"]["current"]["players"][1]
        assert opponent["hand"] is None
        assert opponent["prize"] == [None] * 6
        assert "deck" not in opponent

    assert result.stats == {
        "transitions_validated": 5,
        "decisions_validated": 3,
        "exact_target_rows": 3,
        "alakazam_records": 1,
        "required_archetype": "alakazam",
        "selected_records": 1,
        "seat_labels": ["alakazam", "crustle"],
        "label_methods": ["representative_exact", "representative_exact"],
    }


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda payload: payload["steps"][0][0]["visualize"].pop(),
            "length",
        ),
        (
            lambda payload: payload["steps"][4][0].__setitem__("action", [1]),
            "action",
        ),
        (
            lambda payload: payload["steps"][3][0]["observation"][
                "current"
            ].__setitem__("turn", 999),
            "observation",
        ),
    ],
)
def test_trace_alignment_mismatches_fail_closed(
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    payload = _synthetic_episode()
    mutate(payload)
    with pytest.raises(VisualTraceError, match=expected):
        convert_visual_episode(payload, _Classifier(), source="synthetic")


def test_public_mask_and_card_conservation_mismatches_fail_closed() -> None:
    leaked = _synthetic_episode()
    visual = leaked["steps"][0][0]["visualize"]
    leaked_hand = copy.deepcopy(visual[2]["current"]["players"][1]["hand"])
    visual[3]["obs"]["current"]["players"][1]["hand"] = leaked_hand
    leaked["steps"][3][0]["observation"]["current"]["players"][1][
        "hand"
    ] = copy.deepcopy(leaked_hand)
    with pytest.raises(VisualTraceError, match="mask|private|hand"):
        convert_visual_episode(leaked, _Classifier(), source="synthetic")

    impossible = _synthetic_episode()
    visual = impossible["steps"][0][0]["visualize"]
    # Keep the following public count aligned so the failure is specifically
    # the full-state 60-card conservation gate, not a shallow public mismatch.
    visual[2]["current"]["players"][1]["deck"].pop()
    visual[2]["current"]["players"][1]["deckCount"] -= 1
    visual[3]["obs"]["current"]["players"][1]["deckCount"] -= 1
    impossible["steps"][3][0]["observation"]["current"]["players"][1][
        "deckCount"
    ] -= 1
    with pytest.raises(VisualTraceError, match="conservation|60|multiset"):
        convert_visual_episode(impossible, _Classifier(), source="synthetic")


def test_materialized_day_is_atomic_and_checksum_resumable(tmp_path: Path) -> None:
    archive = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-20.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("42.json", json.dumps(_synthetic_episode()))
    output = tmp_path / "alakazam-2026-07-20.features.pkl"
    receipt_path = Path(str(output) + ".receipt.json")

    receipt = materialize_day(
        archive,
        output,
        classifier=_Classifier(),
        source_date="2026-07-20",
        workers=1,
        max_in_flight=1,
        max_context=320,
        min_available_bytes=0,
        resume=True,
    )

    assert output.is_file()
    assert receipt_path.is_file()
    assert receipt["format"] == "pokebot-authoritative-visual-day-receipt"
    assert receipt["format_version"] == 1
    assert receipt["source_date"] == "2026-07-20"
    assert receipt["source_archive"]["sha256"] == _sha256(archive)
    assert receipt["output"]["sha256"] == _sha256(output)
    assert receipt["classifier"]["sha256"].startswith("sha256:")
    assert receipt["schemas"]["compact_mode"] == "temporal-expert-v1"
    assert receipt["schemas"]["expanded_strategic_targets"] == {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": TARGET_SCHEMA_DIGEST,
    }
    assert receipt["target_consumer_contract"] == TARGET_CONSUMER_CONTRACT
    metadata = json.loads(
        Path(receipt["output"]["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["target_consumer_contract"] == TARGET_CONSUMER_CONTRACT
    expanded = metadata["stats"]["expanded_strategic_targets"]
    assert expanded["schema"] == EXPANDED_STRATEGIC_SCHEMA
    assert expanded["digest"] == TARGET_SCHEMA_DIGEST
    assert expanded["decisions"] == metadata["stats"]["decisions_kept"]
    assert set(expanded["head_coverage"]) == set(EXPANDED_HEAD_IDS)
    assert all(
        row["labeled_rows"] + row["masked_rows"]
        == row["total_rows"]
        == expanded["decisions"]
        for row in expanded["head_coverage"].values()
    )
    for head_id in ("action_q", "game_phase", "outcome_distribution"):
        assert expanded["head_coverage"][head_id]["labeled_rows"] > 0
    loaded = list(iter_feature_shard(output))
    assert loaded
    assert all(
        EXPANDED_STRATEGIC_KEY in decision.aux_labels
        for sequence in loaded
        for decision in sequence.decisions
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert not [
        path
        for path in tmp_path.iterdir()
        if ".partial" in path.name or path.name.endswith(".tmp")
    ]

    # A checksum-valid resume must do no packing or final-file replacement.
    frozen_ns = 1_700_000_000_000_000_000
    os.utime(output, ns=(frozen_ns, frozen_ns))
    before = output.read_bytes()
    resumed = materialize_day(
        archive,
        output,
        classifier=_Classifier(),
        source_date="2026-07-20",
        workers=1,
        max_context=320,
        min_available_bytes=0,
        resume=True,
    )
    assert output.read_bytes() == before
    assert output.stat().st_mtime_ns == frozen_ns
    assert resumed["output"]["sha256"] == receipt["output"]["sha256"]

    # An existing receipt never authorizes changed bytes.  Fail closed and keep
    # both pieces for audit instead of silently replacing the corruption.
    output.chmod(0o644)
    with output.open("ab") as handle:
        handle.write(b"tamper")
    tampered = output.read_bytes()
    with pytest.raises(VisualTraceError, match="checksum|digest|changed|mismatch"):
        materialize_day(
            archive,
            output,
            classifier=_Classifier(),
            source_date="2026-07-20",
            workers=1,
            max_context=320,
            min_available_bytes=0,
            resume=True,
        )
    assert output.read_bytes() == tampered
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_materialization_quarantines_bad_member_and_keeps_exact_good_member(
    tmp_path: Path,
) -> None:
    valid = _synthetic_episode()
    invalid = _synthetic_episode()
    invalid["steps"][4][0]["action"] = [1]
    archive = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-20.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # A valid first member proves rollback also works after partial payload
        # bytes have already been packed and fsync has not yet occurred.
        zf.writestr("00-valid.json", json.dumps(valid))
        zf.writestr("01-invalid.json", json.dumps(invalid))
    output = tmp_path / "alakazam-2026-07-20.features.pkl"
    receipt_path = Path(str(output) + ".receipt.json")

    receipt = materialize_day(
        archive,
        output,
        classifier=_Classifier(),
        source_date="2026-07-20",
        workers=1,
        max_context=320,
        min_available_bytes=0,
        resume=True,
    )

    assert output.exists()
    assert receipt_path.exists()
    assert receipt["stats"]["episodes_validated"] == 1
    assert receipt["stats"]["episodes_rejected"] == 1
    assert receipt["stats"]["records_kept"] == 1
    assert receipt["stats"]["records_dropped"] == 1
    assert receipt["stats"]["drop_reasons"] == {
        "action_alignment_mismatch": 1
    }
    example = receipt["stats"]["drop_reason_examples"][
        "action_alignment_mismatch"
    ]
    assert example["member"] == "01-invalid.json"
    assert "action" in example["message"]
    assert not [path for path in tmp_path.iterdir() if ".partial" in path.name]


def test_all_bad_materialization_publishes_neither_shard_nor_receipt(
    tmp_path: Path,
) -> None:
    invalid = _synthetic_episode()
    invalid["steps"][4][0]["action"] = [1]
    archive = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-20.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("invalid.json", json.dumps(invalid))
    output = tmp_path / "alakazam-2026-07-20.features.pkl"
    receipt_path = Path(str(output) + ".receipt.json")

    with pytest.raises(VisualTraceError, match="records 0 < minimum 1"):
        materialize_day(
            archive,
            output,
            classifier=_Classifier(),
            source_date="2026-07-20",
            workers=1,
            max_context=320,
            min_available_bytes=0,
            resume=True,
        )

    assert not output.exists()
    assert not receipt_path.exists()
    assert not [path for path in tmp_path.iterdir() if ".partial" in path.name]
