"""Focused contract coverage for the immutable r259 rollout side store."""

from __future__ import annotations

import fcntl
import hashlib
import json
import runpy
import subprocess
import sys
import threading
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from poke_bot import cg_env, features
from poke_bot import own_deck_rollout_store as store


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _card(card_id: int, serial: int, *, hp: int = 100) -> dict[str, object]:
    return {"id": card_id, "serial": serial, "hp": hp, "energyCards": []}


def _observation(
    *,
    result: int = -1,
    turn: int = 1,
    own_prizes: int = 6,
    opponent_active: bool = True,
    opponent_discard: bool = False,
    select_deck: list[dict[str, object]] | None = None,
    options: list[dict[str, object]] | None = None,
    min_count: int = 0,
    max_count: int = 0,
    own_hand: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    own_hand = [_card(1, 100)] if own_hand is None else own_hand
    opponent = _card(2, 200)
    return {
        "current": {
            "yourIndex": 0,
            "turn": turn,
            "result": result,
            "energyAttached": False,
            "retreated": False,
            "stadium": [],
            "looking": [],
            "players": [
                {
                    "hand": own_hand,
                    "handCount": len(own_hand),
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * own_prizes,
                    "deckCount": 53,
                },
                {
                    "hand": None,
                    "handCount": 1,
                    "active": [opponent] if opponent_active else [],
                    "bench": [],
                    "discard": [opponent] if opponent_discard else [],
                    "prize": [None] * 6,
                    "deckCount": 53,
                },
            ],
        },
        "select": {
            "deck": [] if select_deck is None else select_deck,
            "option": [{"type": "pass"}] if options is None else options,
            "minCount": min_count,
            "maxCount": max_count,
        },
    }


def _record(
    observation: dict[str, object],
    *,
    action: list[int] | None = None,
    transition_after: dict[str, object] | None = None,
    day: str = store.WINDOW_START,
) -> dict[str, object]:
    step: dict[str, object] = {
        "env_step": 0,
        "observation": observation,
        "action": [] if action is None else action,
    }
    if transition_after is not None:
        step["transition_after"] = transition_after
    return {
        "episode_id": "episode-r259-test",
        "source": f"pokemon-tcg-ai-battle-episodes-{day}",
        "seat": 0,
        "archetype": "alakazam",
        "deck": [1] * 60,
        "info_set_ok": True,
        "steps": [step],
    }


def _unknown_outcome_payload(rewards: object) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """One Alakazam seat with a valid ACTIVE decision and public successor."""

    before = _observation(turn=1)
    after = _observation(result=0, turn=2)
    payload: dict[str, object] = {
        "info": {"EpisodeId": "unknown-outcome-r259"},
        "rewards": rewards,
        "steps": [
            [{"action": [1] * 60}, {"action": [2] * 60}],
            [
                {"status": "ACTIVE", "observation": before, "action": []},
                {"status": "INACTIVE", "observation": before, "action": []},
            ],
            [
                {"status": "INACTIVE", "observation": before, "action": []},
                {"status": "ACTIVE", "observation": after, "action": []},
            ],
        ],
    }
    return payload, before, after


def _pinned_unknown_outcome_smoke_window(
    tmp_path: Path,
) -> tuple[store.SourceWindow, store.SourceArchive]:
    payload, _before, _after = _unknown_outcome_payload([None, 1.0])
    payload["info"] = {"EpisodeId": store.SMOKE_UNKNOWN_OUTCOME_EPISODE_ID}
    payload["statuses"] = ["TIMEOUT", "DONE"]
    archive_path = tmp_path / "first-day.zip"
    with zipfile.ZipFile(archive_path, "w") as handle:
        handle.writestr(store.SMOKE_UNKNOWN_OUTCOME_MEMBER, json.dumps(payload, sort_keys=True))
    archive = store.SourceArchive(
        day=store.WINDOW_START,
        path=archive_path,
        sha256=_sha256(archive_path),
        bytes=archive_path.stat().st_size,
        validated_episode_count=1,
        source_slug=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
    )
    return (
        store.SourceWindow(
            manifest_path=tmp_path / "current.lock.json",
            manifest_sha256="sha256:" + "a" * 64,
            original_manifest_path="/protected/current.json",
            original_versioned_receipt_path="/protected/window.json",
            versioned_receipt_path=tmp_path / "window.lock.json",
            versioned_receipt_sha256="sha256:" + "b" * 64,
            archives=(archive,),
        ),
        archive,
    )


def _board_fingerprint(_observation: object, _deck: object) -> str:
    return "sha256:" + "b" * 64


def _raw_next_frame(
    post_observation: dict[str, object],
    *,
    active_seat: int = 1,
    chance: bool = False,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seat in (0, 1):
        row: dict[str, object] = {
            "status": "ACTIVE" if seat == active_seat else "INACTIVE",
            # The source actor's post-frame view is intentionally used for
            # target construction.  Status, not current.yourIndex, proves
            # the next actor.
            "observation": post_observation,
        }
        if chance:
            row["chance"] = True
        rows.append(row)
    return {"steps": [[{}, {}], rows]}


def _archive_fixture(tmp_path: Path) -> tuple[Path, str, str, list[Path]]:
    """Create a checksum-bound synthetic 20-ZIP receipt with exact totals."""

    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    start = date.fromisoformat(store.WINDOW_START)
    archive_rows: list[dict[str, object]] = []
    archives: list[Path] = []
    # 13 * 4563 + 7 * 4562 = 91,253 exactly.
    counts = [4563] * 13 + [4562] * 7
    for index in range(store.WINDOW_DAYS):
        day = (start + timedelta(days=index)).isoformat()
        path = archive_dir / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("placeholder.txt", day)
        archives.append(path)
        archive_rows.append(
            {
                "date": day,
                "dataset_slug": f"pokemon-tcg-ai-battle-episodes-{day}",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "validated": True,
                "validated_episode_count": counts[index],
            }
        )
    versioned = tmp_path / "window.json"
    core = {
        "schema": store.ARCHIVE_RECEIPT_SCHEMA,
        "status": "ready",
        "window_policy": "exact_20_consecutive_calendar_days",
        "window_start": store.WINDOW_START,
        "window_end": store.WINDOW_END,
        "days": store.WINDOW_DAYS,
        "all_dates_represented": True,
        "archives": archive_rows,
        "total_episodes": store.WINDOW_TOTAL_EPISODES,
    }
    versioned.write_text(json.dumps(core, sort_keys=True), encoding="utf-8")
    versioned_sha = _sha256(versioned)
    manifest = tmp_path / "current.json"
    manifest.write_text(
        json.dumps({**core, "versioned_receipt": str(versioned)}, sort_keys=True),
        encoding="utf-8",
    )
    return manifest, _sha256(manifest), versioned_sha, archives


def test_native_attack_end_turn_post_frame_keeps_terminal_ko_prize_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adjacent opposite ACTIVE frame is a causal attack result, not a boundary."""

    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    before = _observation(own_prizes=1)
    after = _observation(
        result=0,
        own_prizes=0,
        opponent_active=False,
        opponent_discard=True,
    )
    # The inactive actor-side row intentionally retains the stale pre-attack
    # state.  The unique ACTIVE row is the only admissible fresh transition.
    transition = store._derive_public_transition_from_episode(
        {
            "statuses": ["DONE", "DONE"],
            "rewards": [1.0, -1.0],
            "steps": [
                [{}, {}],
                [
                    {"status": "INACTIVE", "observation": before},
                    {"status": "ACTIVE", "observation": after},
                ],
            ]
        },
        0,
        0,
    )
    assert transition is not None
    assert transition["next_actor_seat"] == 1
    assert transition["transition_after_immediate"] is True
    assert "boundary" not in transition

    rows, counts = store.materialize_record_sidecar_rows(
        _record(before, transition_after=transition),
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    terminal = rows[0]["supervision"]["terminal_conversion"]
    assert terminal["labels"]["terminal_class"] == {"value": 1, "mask": True}
    assert terminal["labels"]["prize_closeout"] == {"value": 1.0, "mask": True}
    assert terminal["labels"]["opponent_knockout"] == {"value": 1.0, "mask": True}
    assert terminal["mask"] == [True] * 6
    assert counts["terminal_conversion"]["terminal_class"]["own_win"] == 1
    assert counts["terminal_conversion"]["opponent_knockout"] == {"labeled": 1, "positive": 1}


@pytest.mark.parametrize("chance,active_seat", [(True, 1), (False, 2)])
def test_non_immediate_chance_or_ambiguous_frames_mask_targets(
    monkeypatch: pytest.MonkeyPatch,
    chance: bool,
    active_seat: int,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    post = _observation(result=-1)
    if active_seat == 2:
        payload = {"steps": [[{}, {}], [{"status": "ACTIVE", "observation": post}, {"status": "ACTIVE", "observation": post}]]}
    else:
        payload = _raw_next_frame(post, active_seat=active_seat, chance=chance)
    transition = store._derive_public_transition_from_episode(payload, 0, 0)  # type: ignore[arg-type]
    if chance:
        assert transition is not None
        assert transition["transition_after_immediate"] is False
    else:
        assert transition is None
    rows, _counts = store.materialize_record_sidecar_rows(
        _record(_observation(), transition_after=transition),
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    assert rows[0]["supervision"]["terminal_conversion"]["mask"] == [False] * 6


def test_native_filter_drops_stale_inactive_done_echoes_and_materializes_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    inactive = _observation(turn=1)
    active = _observation(turn=2)
    done = _observation(turn=3)
    record = _record(inactive)
    record["steps"] = [
        {"env_step": 0, "observation": inactive, "action": []},
        {"env_step": 1, "observation": active, "action": []},
        {"env_step": 2, "observation": done, "action": []},
    ]
    record["n_decisions"] = 3
    payload = {
        "steps": [
            [
                {"status": "INACTIVE", "observation": inactive, "action": []},
                {},
            ],
            [
                {"status": "ACTIVE", "observation": active, "action": []},
                {},
            ],
            [
                {"status": "DONE", "observation": done, "action": []},
                {},
            ],
        ]
    }
    filtered = store._validate_native_record_decisions(record, payload)  # type: ignore[arg-type]
    assert filtered is not None
    assert [step["env_step"] for step in filtered["steps"]] == [1]
    assert filtered["n_decisions"] == 1
    assert [step["env_step"] for step in record["steps"]] == [0, 1, 2]

    rows, _counts = store.materialize_record_sidecar_rows(
        filtered,
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    assert [row["env_step"] for row in rows] == [1]


def test_native_filter_rejects_active_public_observation_mismatch() -> None:
    raw_observation = _observation(turn=1)
    converted_observation = _observation(turn=2)
    payload = {
        "steps": [
            [
                {"status": "ACTIVE", "observation": raw_observation, "action": []},
                {},
            ]
        ]
    }
    with pytest.raises(store.SourceRecordError, match="public observation drifted"):
        store._validate_native_record_decisions(_record(converted_observation), payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["INACTIVE", "DONE"])
def test_native_filter_skips_record_when_no_active_decision_remains(status: str) -> None:
    observation = _observation()
    payload = {
        "statuses": ["DONE", "DONE"],
        "rewards": [0.0, 0.0],
        "steps": [
            [
                {"status": status, "observation": observation, "action": []},
                {},
            ]
        ]
    }
    assert store._validate_native_record_decisions(_record(observation), payload) is None  # type: ignore[arg-type]


def test_native_iterator_skips_episode_when_only_stale_rows_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale-only generic-converter record never reaches sidecar materialization."""

    observation = _observation()
    payload = {
        "statuses": ["DONE", "DONE"],
        "rewards": [0.0, 0.0],
        "steps": [
            [
                {"status": "DONE", "observation": observation, "action": []},
                {},
            ]
        ]
    }
    archive_path = tmp_path / "one-episode.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("episode.json", json.dumps(payload, sort_keys=True))
    archive = store.SourceArchive(
        day=store.WINDOW_START,
        path=archive_path,
        sha256=_sha256(archive_path),
        bytes=archive_path.stat().st_size,
        validated_episode_count=1,
        source_slug=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
    )

    from poke_bot import replay_import

    monkeypatch.setattr(replay_import, "convert_episode_to_records", lambda *_args, **_kwargs: [_record(observation)])

    class _Label:
        deck_id = "alakazam"

    class _Classifier:
        def classify_episode(self, _payload: object) -> tuple[list[object], list[_Label]]:
            return [], [_Label(), _Label()]

    accounting = store._ArchiveNativeAccounting()
    assert list(store._iter_archive_native_records(archive, _Classifier(), accounting=accounting)) == []
    assert accounting.to_dict()["records_skipped_stale_only"] == 1
    assert accounting.to_dict()["records_emitted"] == 0


def test_invalid_rewards_use_r259_fallback_and_mask_outcome_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    payload, _before, _after = _unknown_outcome_payload([None, -1.0])
    fallback = store._convert_unknown_outcome_episode_to_r259_records(
        payload,
        source=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
        seat_archetypes=("alakazam", "other"),
    )
    assert len(fallback) == 1
    assert not {"winner", "value", "opp_deck", "aux_labels"}.intersection(fallback[0])
    assert all("aux_labels" not in step for step in fallback[0]["steps"])

    projected = store._project_r259_native_record(
        fallback[0],
        outcome_provenance=store._OUTCOME_PROVENANCE_MASKED,
    )
    filtered = store._validate_native_record_decisions(projected, payload)
    assert filtered is not None
    assert [step["env_step"] for step in filtered["steps"]] == [1]
    attached = store._attach_public_transitions(
        filtered,
        payload,
        outcome_verified=False,
    )
    transition = attached["steps"][0]["transition_after"]
    assert transition["result"] is None

    rows, counts = store.materialize_record_sidecar_rows(
        attached,
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    assert len(rows) == 1
    terminal = rows[0]["supervision"]["terminal_conversion"]
    tutor = rows[0]["supervision"]["visible_tutor_completion"]
    assert terminal["mask"] == [False] * 6
    assert tutor["mask"][3:] == [False] * 4
    assert counts["outcome_provenance"] == {
        "verified_reward_rows": 0,
        "masked_invalid_or_missing_reward_rows": 1,
    }
    assert "_r259_outcome_provenance" not in rows[0]


def test_smoke_unknown_outcome_conversion_stops_before_later_full_observations() -> None:
    """The smoke cap must stop raw iteration immediately after an ACTIVE row."""

    payload, before, _after = _unknown_outcome_payload([None, 1.0])
    payload["info"] = {"EpisodeId": "bounded-smoke-r259"}

    class _FailAfterBoundedDecision(list[object]):
        def __iter__(self):  # type: ignore[override]
            yield self[0]
            yield self[1]
            raise AssertionError("smoke converter read beyond its active-decision bound")

    payload["steps"] = _FailAfterBoundedDecision(
        [
            [{"action": [1] * 60}, {"action": [2] * 60}],
            [
                {"status": "ACTIVE", "observation": before, "action": []},
                {"status": "INACTIVE", "observation": before, "action": []},
            ],
        ]
    )
    records = store._convert_unknown_outcome_episode_to_r259_records(
        payload,
        source=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
        seat_archetypes=("alakazam", "other"),
        max_active_decisions_per_record=1,
    )
    assert len(records) == 1
    assert [step["env_step"] for step in records[0]["steps"]] == [1]

    with pytest.raises(store.SourceRecordError, match="positive integer"):
        store._convert_unknown_outcome_episode_to_r259_records(
            payload,
            source=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
            seat_archetypes=("alakazam", "other"),
            max_active_decisions_per_record=0,
        )


@pytest.mark.parametrize(
    ("rewards", "statuses"),
    [([None, 1.0], ["TIMEOUT", "DONE"]), ([1.0, -1.0], ["TIMEOUT", "DONE"])],
)
def test_unknown_outcome_archive_member_bypasses_generic_winner_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rewards: list[object],
    statuses: list[str],
) -> None:
    """Regression for Elmo's 2026-07-22:87394115.json failure shape."""

    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    payload, _before, _after = _unknown_outcome_payload(rewards)
    payload["info"] = {"EpisodeId": "87394115"}
    payload["statuses"] = statuses
    archive_path = tmp_path / "87394115.zip"
    with zipfile.ZipFile(archive_path, "w") as handle:
        handle.writestr("87394115.json", json.dumps(payload, sort_keys=True))
    archive = store.SourceArchive(
        day=store.WINDOW_START,
        path=archive_path,
        sha256=_sha256(archive_path),
        bytes=archive_path.stat().st_size,
        validated_episode_count=1,
        source_slug=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
    )

    from poke_bot import replay_import

    def generic_must_not_run(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("generic winner converter must not run for None rewards")

    monkeypatch.setattr(replay_import, "convert_episode_to_records", generic_must_not_run)

    class _Label:
        deck_id = "alakazam"

    class _Classifier:
        def classify_episode(self, _payload: object) -> tuple[list[object], list[_Label]]:
            return [], [_Label(), _Label()]

    accounting = store._ArchiveNativeAccounting()
    records = list(store._iter_archive_native_records(archive, _Classifier(), accounting=accounting))
    assert len(records) == 1
    assert records[0]["episode_id"] == "87394115"
    assert records[0]["_r259_outcome_provenance"] == store._OUTCOME_PROVENANCE_MASKED
    assert [step["env_step"] for step in records[0]["steps"]] == [1]
    rows, _counts = store.materialize_record_sidecar_rows(
        records[0],
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    assert rows[0]["supervision"]["terminal_conversion"]["mask"] == [False] * 6
    assert rows[0]["supervision"]["visible_tutor_completion"]["mask"][3:] == [False] * 4
    assert accounting.to_dict() == {
        "schema": "poke_bot.own_deck_rollout_archive_native_accounting/v1",
        "episodes_seen": 1,
        "verified_reward_episodes": 0,
        "invalid_or_missing_reward_episodes": 1,
        "invalid_reward_fallback_records": 1,
        "records_emitted": 1,
        "records_skipped_stale_only": 0,
        "invalid_reward_episodes_skipped_unconvertible": 0,
    }


def test_numeric_timeout_rewards_are_not_outcome_provenance() -> None:
    payload, _before, _after = _unknown_outcome_payload([-1.0, 1.0])
    payload["statuses"] = ["TIMEOUT", "TIMEOUT"]
    assert store._verified_episode_rewards(payload) is None
    transition = store._derive_public_transition_from_episode(payload, 1, 0)  # type: ignore[arg-type]
    assert transition is not None
    assert transition["result"] is None


@pytest.mark.parametrize(
    "rewards",
    [None, [None, -1.0], ["1", -1.0], [float("nan"), 0.0], [1.0, 1.0]],
)
def test_invalid_rewards_never_attest_done_or_nonterminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    rewards: object,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    payload, before, after = _unknown_outcome_payload(rewards)
    done_payload = {
        "statuses": ["DONE", "DONE"],
        "rewards": rewards,
        "steps": [
            [{}, {}],
            [
                {"status": "DONE", "observation": after},
                {"status": "DONE", "observation": after},
            ],
        ],
    }
    assert store._derive_public_transition_from_episode(done_payload, 0, 0) is None  # type: ignore[arg-type]

    transition = store._derive_public_transition_from_episode(payload, 1, 0)  # type: ignore[arg-type]
    assert transition is not None
    assert transition["result"] is None
    record = _record(before, transition_after=transition)
    record["_r259_outcome_provenance"] = store._OUTCOME_PROVENANCE_MASKED
    rows, _counts = store.materialize_record_sidecar_rows(
        record,
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    assert rows[0]["supervision"]["terminal_conversion"]["mask"] == [False] * 6


def test_invalid_outcome_mask_count_is_sealed_in_daily_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    manifest, manifest_sha, versioned_sha, _archives = _archive_fixture(tmp_path)
    versioned_original = Path(json.loads(manifest.read_text(encoding="utf-8"))["versioned_receipt"])
    versioned_lock = tmp_path / "window.lock.json"
    versioned_lock.write_bytes(versioned_original.read_bytes())
    record = _record(_observation())
    record["_r259_outcome_provenance"] = store._OUTCOME_PROVENANCE_MASKED
    stream = tmp_path / "protected.jsonl"
    stream.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "sidecar"

    store.build_protected_jsonl_sidecar(
        source_manifest=manifest,
        protected_records=stream,
        output_root=output,
        source_snapshot_path="/snapshot",
        source_snapshot_tree_sha256="sha256:" + "c" * 64,
        expected_manifest_sha256=manifest_sha,
        expected_versioned_receipt_sha256=versioned_sha,
        versioned_receipt_lock=versioned_lock,
        only_days=[store.WINDOW_START],
    )
    meta = store.read_daily_meta(output, store.WINDOW_START)
    assert meta["label_counts"]["outcome_provenance"] == {
        "verified_reward_rows": 0,
        "masked_invalid_or_missing_reward_rows": 1,
    }
    assert meta["label_counts"]["terminal_conversion"]["terminal_class_labeled"] == 0


def test_archive_native_unknown_outcome_accounting_is_sealed_in_daily_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback's skip/mask accounting is part of immutable archive metadata."""

    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    payload, _before, _after = _unknown_outcome_payload([None, 1.0])
    payload["info"] = {"EpisodeId": "native-meta-unknown-outcome"}
    payload["statuses"] = ["TIMEOUT", "DONE"]
    fallback = store._convert_unknown_outcome_episode_to_r259_records(
        payload,
        source=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
        seat_archetypes=("alakazam", "other"),
    )
    assert len(fallback) == 1
    projected = store._project_r259_native_record(
        fallback[0],
        outcome_provenance=store._OUTCOME_PROVENANCE_MASKED,
    )
    filtered = store._validate_native_record_decisions(projected, payload)
    assert filtered is not None
    record = store._attach_public_transitions(filtered, payload, outcome_verified=False)

    archive_path = tmp_path / "native-meta.zip"
    archive_path.write_bytes(b"native-meta")
    archive = store.SourceArchive(
        day=store.WINDOW_START,
        path=archive_path,
        sha256=_sha256(archive_path),
        bytes=archive_path.stat().st_size,
        validated_episode_count=1,
        source_slug=f"pokemon-tcg-ai-battle-episodes-{store.WINDOW_START}",
    )
    window = store.SourceWindow(
        manifest_path=tmp_path / "manifest.lock.json",
        manifest_sha256="sha256:" + "a" * 64,
        original_manifest_path="/protected/current.json",
        original_versioned_receipt_path="/protected/window.json",
        versioned_receipt_path=tmp_path / "window.lock.json",
        versioned_receipt_sha256="sha256:" + "b" * 64,
        archives=(archive,),
    )
    identity = store._BuildIdentity(
        mode="archive_native",
        source_snapshot_path="/sealed/source",
        source_snapshot_tree_sha256="sha256:" + "c" * 64,
        image_tag=store.EXPECTED_IMAGE_TAG,
        image_id=store.EXPECTED_IMAGE_ID,
        code_identities={"own_deck_rollout_store.py": "sha256:" + "d" * 64},
        classifier={"schema": "test"},
        protected_stream_sha256=None,
    )
    accounting = store._ArchiveNativeAccounting(
        episodes_seen=1,
        invalid_or_missing_reward_episodes=1,
        invalid_reward_fallback_records=1,
        records_emitted=1,
    )
    result = store._build_one_day(
        output_root=tmp_path / "sidecar",
        window=window,
        archive=archive,
        identity=identity,
        records=[record],
        archive_native_accounting=accounting,
    )
    meta = store.read_daily_meta(result.directory.parent.parent, store.WINDOW_START)
    assert meta["archive_native_accounting"] == accounting.to_dict()
    assert meta["label_counts"]["outcome_provenance"] == {
        "verified_reward_rows": 0,
        "masked_invalid_or_missing_reward_rows": 1,
    }
    assert meta["label_counts"]["terminal_conversion"]["terminal_class_labeled"] == 0


def test_done_envelope_uses_rewards_not_stale_terminal_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    stale = _observation(result=-1, own_prizes=1)
    payload = {
        "statuses": ["DONE", "DONE"],
        "rewards": [1.0, -1.0],
        "steps": [
            [{}, {}],
            [
                {"status": "DONE", "observation": stale, "action": [0]},
                {"status": "DONE", "observation": stale, "action": [0]},
            ],
        ],
    }
    transition = store._derive_public_transition_from_episode(payload, 0, 0)  # type: ignore[arg-type]
    assert transition is not None
    assert transition["result"] == 0
    assert transition["next_actor_seat"] is None
    assert transition["players"] == []
    rows, _ = store.materialize_record_sidecar_rows(
        _record(_observation(own_prizes=1), transition_after=transition),
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    terminal = rows[0]["supervision"]["terminal_conversion"]
    assert terminal["labels"]["terminal_class"] == {"value": 1, "mask": True}
    assert terminal["labels"]["prize_closeout"] == {"value": 0.0, "mask": False}
    assert terminal["labels"]["opponent_knockout"] == {"value": 0.0, "mask": False}


@pytest.mark.parametrize(
    ("statuses", "rewards"),
    [(["DONE", "DONE"], [2.0, -1.0]), (["DONE", "ACTIVE"], [1.0, -1.0])],
)
def test_done_envelope_requires_authoritative_zero_sum_provenance(
    statuses: list[str], rewards: list[float]
) -> None:
    stale = _observation(result=0)
    payload = {
        "statuses": statuses,
        "rewards": rewards,
        "steps": [
            [{}, {}],
            [
                {"status": "DONE", "observation": stale},
                {"status": "DONE", "observation": stale},
            ],
        ],
    }
    assert store._derive_public_transition_from_episode(payload, 0, 0) is None  # type: ignore[arg-type]


def test_visible_tutor_labels_and_vectors_are_target_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    selected = _card(1, 101)
    before = _observation(
        select_deck=[selected],
        options=[{"area": 1, "index": 0, "playerIndex": 0}],
        min_count=1,
        max_count=1,
    )
    after = _observation(
        own_hand=[_card(1, 100), selected],
        select_deck=[],
        options=[{"type": "pass"}],
    )
    transition = store._derive_public_transition_from_episode(
        _raw_next_frame(after, active_seat=0), 0, 0  # type: ignore[arg-type]
    )
    assert transition is not None
    rows, _counts = store.materialize_record_sidecar_rows(
        _record(before, action=[0], transition_after=transition),
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    row = rows[0]
    tutor = row["supervision"]["visible_tutor_completion"]
    assert tutor["labels"]["selected_card_id"] == {"value": 1, "mask": True}
    assert tutor["labels"]["selected_from_visible_deck"] == {"value": 1.0, "mask": True}
    assert tutor["labels"]["selected_target_observed_after_action"] == {"value": 1.0, "mask": True}
    assert tutor["vector"][:2] == [1.0, 1.0]
    assert tutor["mask"][:2] == [True, True]
    stage = row["policy_stage_option_features"][0]
    assert stage["candidate_count"] == 1
    assert len(stage["ledger_option_features"]) == 1
    assert len(stage["ledger_option_features"][0]) == store.OPTION_FEATURE_DIM
    assert "observation" not in row
    assert "action" not in row
    assert "transition_after" not in row


def test_private_surface_changes_do_not_affect_public_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    left = _observation()
    right = _observation()
    left.update(
        {
            "search_begin_input": {"hidden": [1, 2, 3]},
            "hidden_state": {"deck": [99]},
            "simulatorState": {"prizeOrder": [7]},
        }
    )
    right.update(
        {
            "search_begin_input": {"hidden": [9]},
            "hidden_state": {"deck": [42]},
            "simulatorState": {"prizeOrder": [8]},
        }
    )
    masked_left = store._masked_public_observation(left, expected_seat=0)
    masked_right = store._masked_public_observation(right, expected_seat=0)
    assert masked_left == masked_right
    assert store._public_observation_fingerprint(masked_left) == store._public_observation_fingerprint(masked_right)

    left_row, _ = store.materialize_record_sidecar_rows(
        _record(left),
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    right_row, _ = store.materialize_record_sidecar_rows(
        _record(right),
        source_day=store.WINDOW_START,
        source_manifest_sha256="sha256:" + "a" * 64,
    )
    for name in (
        "observation_fingerprint",
        "ledger_observation_fingerprint",
        "deck_fingerprint",
        "board_feature_fingerprint",
    ):
        assert left_row[0][name] == right_row[0][name]


def test_manifest_zip_rehash_and_immutable_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    manifest, manifest_sha, versioned_sha, archives = _archive_fixture(tmp_path)
    versioned_original = Path(json.loads(manifest.read_text(encoding="utf-8"))["versioned_receipt"])
    versioned_lock = tmp_path / "root-readable-window.lock.json"
    versioned_lock.write_bytes(versioned_original.read_bytes())
    window = store.load_source_window(
        manifest,
        expected_manifest_sha256=manifest_sha,
        expected_versioned_receipt_sha256=versioned_sha,
        versioned_receipt_lock=versioned_lock,
    )
    assert len(window.archives) == 20
    assert sum(row.validated_episode_count for row in window.archives) == 91_253
    assert window.original_versioned_receipt_path == str(versioned_original)
    assert window.versioned_receipt_path == versioned_lock
    with pytest.raises(store.SourceRecordError, match="episode-member count drifted"):
        list(store._iter_archive_native_records(window.archives[0], object()))

    stream = tmp_path / "protected.jsonl"
    stream.write_text(json.dumps(_record(_observation())) + "\n", encoding="utf-8")
    output = tmp_path / "sidecar"
    kwargs = {
        "source_manifest": manifest,
        "protected_records": stream,
        "output_root": output,
        "source_snapshot_path": "/snapshot",
        "source_snapshot_tree_sha256": "sha256:" + "c" * 64,
        "expected_manifest_sha256": manifest_sha,
        "expected_versioned_receipt_sha256": versioned_sha,
        "versioned_receipt_lock": versioned_lock,
        "only_days": [store.WINDOW_START],
    }
    first = store.build_protected_jsonl_sidecar(**kwargs)
    assert first[0].skipped_existing is False
    rows = list(store.iter_daily_sidecar_rows(output, store.WINDOW_START))
    assert len(rows) == 1
    assert rows[0]["training_eligibility"]["active_r241"] is False
    meta = store.read_daily_meta(output, store.WINDOW_START)
    assert meta["status"] == "complete_immutable_sidecar"
    assert meta["shard_sha256"] == meta["shard"]["sha256"]
    second = store.build_protected_jsonl_sidecar(**kwargs)
    assert second[0].skipped_existing is True

    with archives[0].open("ab") as changed:
        changed.write(b"drift")
    with pytest.raises(store.SourceManifestError, match="(byte size|checksum) drifted"):
        store.load_source_window(
            manifest,
            expected_manifest_sha256=manifest_sha,
            expected_versioned_receipt_sha256=versioned_sha,
        )


def test_day_lock_serializes_duplicate_resume(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    day = store.WINDOW_START
    lock_root = output / ".r259-locks"
    lock_root.mkdir()
    lock_path = lock_root / f"{day}.lock"
    lock_path.touch()
    descriptor = lock_path.open("r+")
    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX)
    entered = threading.Event()
    released = threading.Event()

    def contender() -> None:
        with store._exclusive_day_lock(output, day):
            entered.set()
            released.wait(timeout=2)

    worker = threading.Thread(target=contender)
    worker.start()
    time.sleep(0.05)
    assert not entered.is_set()
    fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
    descriptor.close()
    assert entered.wait(timeout=2)
    released.set()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_launcher_preserves_image_cg_runtime_for_container_import_smoke() -> None:
    """The two preflights are isolated and preserve the baked-in CG runtime."""

    launcher = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "elmo"
        / "run_own_deck_rollout_store_r259.sh"
    ).read_text(encoding="utf-8")

    assert 'readonly CONTAINER_SOURCE="/r259-source"' in launcher
    assert '--workdir "${CONTAINER_SOURCE}"' in launcher
    assert 'src=${SOURCE_SNAPSHOT},dst=${CONTAINER_SOURCE},readonly' in launcher
    assert 'src=${SOURCE_SNAPSHOT},dst=/workspace,readonly' not in launcher
    assert '"${CONTAINER_SOURCE}/scripts/update_own_deck_rollout_store.py"' in launcher
    assert '--card-csv "${CONTAINER_SOURCE}/cards/EN_Card_Data.csv"' in launcher
    assert "/workspace/kaggle/input/cg-lib" in launcher

    preflights = launcher.split("exec docker run --rm --init", maxsplit=1)[0]
    smoke_blocks = preflights.split("docker run --rm --init")[1:]
    assert len(smoke_blocks) == 2
    normal, unknown_outcome = smoke_blocks
    assert 'readonly SMOKE_CONTAINER_NAME=' not in launcher
    assert '--name "${NORMAL_SMOKE_CONTAINER_NAME}"' in normal
    assert '--name "${UNKNOWN_OUTCOME_SMOKE_CONTAINER_NAME}"' in unknown_outcome
    assert "--archive-native-smoke" in normal
    assert "--archive-native-unknown-outcome-smoke" not in normal
    assert "--archive-native-unknown-outcome-smoke" in unknown_outcome
    assert "--archive-native-smoke" not in unknown_outcome
    for smoke in (normal, unknown_outcome):
        assert "--network none" in smoke
        assert "--runtime runc" in smoke
        assert "--read-only" in smoke
        assert "--user \"${SHARED_UID}:${SHARED_GID}\"" in smoke
        assert "--cap-drop ALL" in smoke
        assert "--security-opt no-new-privileges:true" in smoke
        assert "NVIDIA_VISIBLE_DEVICES=none" in smoke
        assert "CUDA_VISIBLE_DEVICES=" in smoke
        assert "LEAF_GPU=cpu" in smoke
        assert "src=${OUTPUT},dst=/output" not in smoke
        assert "--output-root" not in smoke
    assert "--pids-limit 128" in normal
    assert "--cpus 0.5" in normal
    assert "--memory 1g" in normal
    assert "--memory-swap 1g" in normal
    assert "--pids-limit 256" in unknown_outcome
    assert "--cpus 1.0" in unknown_outcome
    assert "--memory 2g" in unknown_outcome
    assert "--memory-swap 2g" in unknown_outcome


def test_launcher_host_preflight_rejects_duplicate_current_receipt_keys(tmp_path: Path) -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "elmo"
        / "run_own_deck_rollout_store_r259.sh"
    ).read_text(encoding="utf-8")

    assert "def reject_duplicate_json_keys(pairs):" in launcher
    assert "raise ValueError(f\"duplicate JSON key: {key}\")" in launcher
    assert "json.load(stream, object_pairs_hook=reject_duplicate_json_keys)" in launcher
    start = launcher.index("from __future__ import annotations", launcher.index("VERSIONED_ORIGINAL"))
    end = launcher.index("\nPY\n)", start)
    duplicate = tmp_path / "duplicate-current.json"
    duplicate.write_text(
        '{"versioned_receipt":"/same","versioned_receipt":"/same"}',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-", str(duplicate)],
        input=launcher[start:end],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "duplicate JSON key: versioned_receipt" in completed.stderr


def test_source_stager_publishes_nonroot_readable_immutable_source() -> None:
    stager = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "elmo"
        / "stage_own_deck_rollout_store_r259_source.sh"
    ).read_text(encoding="utf-8")

    assert 'find "${stage}" -xdev -type d -exec chmod 0555 {} +' in stager
    assert 'find "${stage}" -xdev -type f -exec chmod 0444 {} +' in stager
    assert 'stat.S_IMODE(info.st_mode) != 0o555' in stager
    assert 'stat.S_IMODE(info.st_mode) != expected_mode' in stager
    assert 'relative == "ops/elmo/run_own_deck_rollout_store_r259.sh"' in stager


def test_archive_native_smoke_materializes_one_record_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal smoke does not retain or execute the pinned regression."""

    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    monkeypatch.setattr(
        cg_env,
        "ensure_cg_importable",
        lambda: Path(store.EXPECTED_SMOKE_CG_RUNTIME),
    )
    monkeypatch.setattr(features, "card_vocab_size", lambda: 1269)
    window, _archive = _pinned_unknown_outcome_smoke_window(tmp_path)
    monkeypatch.setattr(store, "load_source_window", lambda *_args, **_kwargs: window)

    class _Label:
        deck_id = "alakazam"

    class _Classifier:
        def classify_episode(self, _payload: object) -> tuple[list[object], list[_Label]]:
            return [], [_Label(), _Label()]

    monkeypatch.setattr(store, "_load_classifier", lambda *_args: _Classifier())
    captured: dict[str, object] = {}

    def normal_records(
        _archive: object,
        _classifier: object,
        *,
        excluded_members: object = None,
    ) -> object:
        captured["excluded_members"] = excluded_members
        return iter([_record(_observation())])

    monkeypatch.setattr(
        store,
        "_iter_archive_native_records",
        normal_records,
    )
    monkeypatch.setattr(
        store,
        "_smoke_pinned_unknown_outcome_member",
        lambda **_kwargs: pytest.fail("normal smoke must not invoke pinned-member smoke"),
    )

    result = store.smoke_archive_native_one_record(
        source_manifest=tmp_path / "current.lock.json",
        classifier_mix=tmp_path / "mix.json",
        classifier_representatives=tmp_path / "representatives.json",
        card_csv=tmp_path / "cards.csv",
        original_manifest_path="/protected/current.json",
        versioned_receipt_lock=tmp_path / "window.lock.json",
        expected_manifest_sha256="sha256:" + "a" * 64,
        expected_versioned_receipt_sha256="sha256:" + "b" * 64,
    )

    assert result["status"] == "passed_in_memory"
    assert result["archive"]["day"] == store.WINDOW_START
    assert result["episode_id"] == "episode-r259-test"
    assert result["row_count"] == 1
    assert result["card_vocab_size"] == 1269
    assert result["cg_runtime"] == store.EXPECTED_SMOKE_CG_RUNTIME
    assert result["smoke_kind"] == "normal_archive_native_record"
    assert "pinned_unknown_outcome_member" not in result
    assert captured["excluded_members"] == (store.SMOKE_UNKNOWN_OUTCOME_MEMBER,)


def test_exact_unknown_outcome_smoke_materializes_only_pinned_member_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "board_feature_fingerprint", _board_fingerprint)
    monkeypatch.setattr(
        cg_env,
        "ensure_cg_importable",
        lambda: Path(store.EXPECTED_SMOKE_CG_RUNTIME),
    )
    monkeypatch.setattr(features, "card_vocab_size", lambda: 1269)
    window, _archive = _pinned_unknown_outcome_smoke_window(tmp_path)
    monkeypatch.setattr(store, "load_source_window", lambda *_args, **_kwargs: window)

    class _Label:
        deck_id = "alakazam"

    class _Classifier:
        def classify_episode(self, _payload: object) -> tuple[list[object], list[_Label]]:
            return [], [_Label(), _Label()]

    monkeypatch.setattr(store, "_load_classifier", lambda *_args: _Classifier())
    monkeypatch.setattr(
        store,
        "_iter_archive_native_records",
        lambda *_args, **_kwargs: pytest.fail(
            "exact unknown-outcome smoke must not run the normal iterator"
        ),
    )

    from poke_bot import replay_import

    def generic_must_not_run(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("exact unknown-outcome smoke must not call generic conversion")

    monkeypatch.setattr(replay_import, "convert_episode_to_records", generic_must_not_run)
    result = store.smoke_archive_native_unknown_outcome_member(
        source_manifest=tmp_path / "current.lock.json",
        classifier_mix=tmp_path / "mix.json",
        classifier_representatives=tmp_path / "representatives.json",
        card_csv=tmp_path / "cards.csv",
        original_manifest_path="/protected/current.json",
        versioned_receipt_lock=tmp_path / "window.lock.json",
        expected_manifest_sha256="sha256:" + "a" * 64,
        expected_versioned_receipt_sha256="sha256:" + "b" * 64,
    )

    assert result["schema"] == "poke_bot.own_deck_rollout_unknown_outcome_smoke/v1"
    assert result["smoke_kind"] == "pinned_unknown_outcome_member"
    assert result["status"] == "passed_in_memory"
    assert result["card_vocab_size"] == 1269
    pinned = result["pinned_unknown_outcome_member"]
    assert {
        name: value
        for name, value in pinned.items()
        if name != "label_counts"
    } == {
        "member": store.SMOKE_UNKNOWN_OUTCOME_MEMBER,
        "episode_id": store.SMOKE_UNKNOWN_OUTCOME_EPISODE_ID,
        "fallback_record_count": 1,
        "retained_active_record_count": 1,
        "stale_only_record_count": 0,
        "max_active_decisions_per_record": 2,
        "fallback_active_decision_count": 1,
        "retained_active_decision_count": 1,
        "materialized_record_count": 1,
        "materialized_active_decision_count": 1,
        "sidecar_row_count": 1,
        "terminal_outcome_masked_row_count": 1,
        "outcome_provenance": store._OUTCOME_PROVENANCE_MASKED,
    }
    assert pinned["label_counts"]["outcome_provenance"] == {
        "verified_reward_rows": 0,
        "masked_invalid_or_missing_reward_rows": 1,
    }
    assert pinned["label_counts"]["terminal_conversion"]["terminal_class_labeled"] == 0


def test_normal_iterator_excludes_pinned_member_before_generic_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 1 GiB normal smoke must never hand 87394115 to the generic route."""

    _window, archive = _pinned_unknown_outcome_smoke_window(tmp_path)

    class _Classifier:
        def classify_episode(self, _payload: object) -> tuple[list[object], list[object]]:
            pytest.fail("excluded pinned member must not be classified by normal iterator")

    from poke_bot import replay_import

    monkeypatch.setattr(
        replay_import,
        "convert_episode_to_records",
        lambda *_args, **_kwargs: pytest.fail(
            "excluded pinned member must not reach generic conversion"
        ),
    )
    assert list(
        store._iter_archive_native_records(
            archive,
            _Classifier(),
            excluded_members=(store.SMOKE_UNKNOWN_OUTCOME_MEMBER,),
        )
    ) == []


def test_cli_smoke_flags_are_mutually_exclusive_and_output_free() -> None:
    cli_path = Path(__file__).resolve().parents[1] / "scripts" / "update_own_deck_rollout_store.py"
    arguments = runpy.run_path(str(cli_path))["_arguments"]
    common = [
        "--source-manifest",
        "/input/current.json",
        "--original-manifest-path",
        "/protected/current.json",
        "--versioned-receipt-lock",
        "/input/window.json",
        "--classifier-mix",
        "/input/mix.json",
        "--classifier-representatives",
        "/input/representatives.json",
        "--card-csv",
        "/r259-source/cards/EN_Card_Data.csv",
    ]
    normal = arguments([*common, "--archive-native-smoke"])
    assert normal.archive_native_smoke is True
    assert normal.archive_native_unknown_outcome_smoke is False
    assert normal.output_root is None

    unknown = arguments([*common, "--archive-native-87394115-smoke"])
    assert unknown.archive_native_smoke is False
    assert unknown.archive_native_unknown_outcome_smoke is True
    assert unknown.output_root is None

    with pytest.raises(SystemExit):
        arguments(
            [
                *common,
                "--archive-native-smoke",
                "--archive-native-unknown-outcome-smoke",
            ]
        )


def test_store_json_reader_rejects_same_valued_duplicate_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"source_manifest_sha256":"same","source_manifest_sha256":"same"}')

    with pytest.raises(store.OwnDeckRolloutStoreError, match="not valid JSON"):
        store._read_json_object(duplicate, label="duplicate manifest")


@pytest.mark.parametrize(
    ("statuses", "rewards"),
    [(["DONE", "DONE"], [None, 1.0]), (["TIMEOUT", "DONE"], [None, 0.0])],
)
def test_pinned_smoke_member_requires_exact_audited_unknown_outcome_shape(
    statuses: list[str], rewards: list[object]
) -> None:
    payload, _before, _after = _unknown_outcome_payload(rewards)
    payload["info"] = {"EpisodeId": store.SMOKE_UNKNOWN_OUTCOME_EPISODE_ID}
    payload["statuses"] = statuses
    with pytest.raises(store.SourceRecordError, match="(status|reward) envelope drifted"):
        store._validate_smoke_unknown_outcome_payload(payload)


def test_unit_requires_existing_docker_without_pulling_it_in() -> None:
    unit = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "elmo"
        / "pokebot-own-deck-rollout-store-r259.service"
    ).read_text(encoding="utf-8")

    assert "Wants=docker.service" not in unit
    assert "Requires=docker.service" not in unit
    assert "ExecStartPre=/usr/bin/systemctl is-active --quiet docker.service" in unit
    assert "ExecStartPre=/usr/bin/test -S /var/run/docker.sock" in unit
