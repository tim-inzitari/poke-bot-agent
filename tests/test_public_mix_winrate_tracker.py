from __future__ import annotations

from pathlib import Path

from scripts.track_public_mix_winrate import (
    _atomic_json,
    _consume_available,
    _new_state,
    _payload,
    _read_json,
    _state_for_iteration,
)


def _row(
    value: float,
    collect: str,
    padding: int = 0,
    *,
    opponent: str = "agent-a",
    target_opponent: str = "",
    seat: int = 0,
    checkpoint_digest: str = "sha256:model-a",
) -> bytes:
    return (
        b'{"episode_id":"x","seat":'
        + str(seat).encode()
        + b',"opp_archetype":"'
        + opponent.encode()
        + b'","value":'
        + str(value).encode()
        + b',"target_provenance":{"collect":"'
        + collect.encode()
        + b'","opponent_id":"'
        + (target_opponent or opponent).encode()
        + b'","behavior_checkpoint_digest":"'
        + checkpoint_digest.encode()
        + b'"},"decisions":[{"observation":"'
        + (b"x" * padding)
        + b'"}]}\n'
    )


def test_tracker_counts_only_complete_public_mix_rows(tmp_path: Path) -> None:
    shard = tmp_path / "iter_00005.jsonl"
    shard.write_bytes(
        _row(1.0, "self_play")
        + _row(1.0, "public_mix", padding=1024 * 1024)
        + _row(0.0, "public_mix")
        + _row(-1.0, "public_mix")
        + _row(1.0, "public_mix")[:-1]
    )
    state = _new_state("run", 5, shard)
    assert _consume_available(shard, state) == 3
    assert state["games"] == 3
    assert state["wins"] == 1.5
    assert state["losses"] == 1
    payload = _payload(state, stage="collect:public_mix", active=True)
    assert payload["win_rate"] == 0.5
    assert payload["checkpoint_digest"] == "sha256:model-a"
    assert payload["checkpoint_mixed"] is False
    assert payload["matchups"] == [
        {
            "opponent_id": "agent-a",
            "games": 3,
            "wins": 1.5,
            "draws": 1,
            "losses": 1,
            "seat0": 3,
            "seat1": 0,
            "win_rate": 0.5,
        }
    ]

    with shard.open("ab") as handle:
        handle.write(b"\n")
    assert _consume_available(shard, state) == 1
    assert state["games"] == 4
    assert state["wins"] == 2.5


def test_tracker_breaks_out_opponents_and_seats(tmp_path: Path) -> None:
    shard = tmp_path / "iter_00005.jsonl"
    shard.write_bytes(
        _row(1.0, "public_mix", opponent="hard-a", seat=0)
        + _row(-1.0, "public_mix", opponent="hard-a", seat=1)
        + _row(1.0, "public_mix", opponent="hard-b", seat=1)
    )
    state = _new_state("run", 5, shard)
    assert _consume_available(shard, state) == 3

    rows = {
        row["opponent_id"]: row
        for row in _payload(state, stage="complete", active=False)["matchups"]
    }
    assert rows["hard-a"] == {
        "opponent_id": "hard-a",
        "games": 2,
        "wins": 1.0,
        "draws": 0,
        "losses": 1,
        "seat0": 1,
        "seat1": 1,
        "win_rate": 0.5,
    }
    assert rows["hard-b"]["win_rate"] == 1.0
    assert rows["hard-b"]["seat1"] == 1


def test_tracker_keeps_research_controls_separate_and_non_gate(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter_00007.jsonl"
    shard.write_bytes(
        _row(1.0, "research_controls", opponent="iono", seat=0)
        + _row(-1.0, "research_controls", opponent="iono", seat=1)
        + _row(0.0, "research_controls", opponent="dragapult-ex", seat=0)
        + _row(1.0, "public_mix", opponent="public-a", seat=1)
        + _row(1.0, "self_play", opponent="self", seat=0)
    )
    state = _new_state("run", 7, shard)
    assert _consume_available(shard, state) == 4

    payload = _payload(
        state,
        stage="collect:research_controls",
        active=False,
    )
    assert payload["games"] == 1
    assert payload["win_rate"] == 1.0
    research = payload["research_controls"]
    assert research["active"] is True
    assert research["games"] == 3
    assert research["win_rate"] == 0.5
    assert research["schema"] == "poke_bot.research_controls_live_winrate/v1"
    assert "legacy in-shard" in research["definition"]
    assert "separate additive greedy non-training result artifact" in research[
        "definition"
    ]
    assert {row["opponent_id"] for row in research["matchups"]} == {
        "iono",
        "dragapult-ex",
    }

    completed = _payload(
        state,
        stage="collect:public_mix",
        active=True,
    )["research_controls"]
    assert completed["active"] is False
    assert completed["stage"] == "collect:research_controls:complete"


def test_tracker_counts_strong_public_practice_separately_from_public_mix(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter_00012.jsonl"
    shard.write_bytes(
        _row(
            1.0,
            "strong_public_practice",
            opponent="lucario",
            target_opponent="yaminh",
            seat=0,
        )
        + _row(
            -1.0,
            "strong_public_practice",
            opponent="lucario",
            target_opponent="aman",
            seat=1,
        )
        + _row(
            1.0,
            "strong_public_practice",
            opponent="lucario",
            target_opponent="yaminh",
            seat=1,
        )
        + _row(1.0, "public_mix", opponent="diverse", seat=0)
        + _row(1.0, "self_play", opponent="self", seat=0)
    )
    state = _new_state("run", 12, shard)
    assert _consume_available(shard, state) == 4

    payload = _payload(state, stage="collect:public_mix", active=True)
    assert payload["games"] == 1
    assert payload["matchups"][0]["opponent_id"] == "diverse"
    practice = payload["strong_public_practice"]
    assert practice["available"] is True
    assert practice["active"] is True
    assert practice["games"] == 3
    assert practice["win_rate"] == 2 / 3
    assert practice["checkpoint_mixed"] is False
    assert practice["schema"] == (
        "poke_bot.strong_public_practice_live_winrate/v1"
    )
    assert {row["opponent_id"] for row in practice["matchups"]} == {
        "aman",
        "yaminh",
    }
    assert "never formal gate evidence" in practice["definition"]


def test_tracker_carries_last_strong_public_practice_until_new_results(
    tmp_path: Path,
) -> None:
    previous_shard = tmp_path / "iter_00011.jsonl"
    previous_shard.write_bytes(
        _row(1.0, "strong_public_practice", opponent="pilkwang", seat=0)
    )
    previous = _new_state("run", 11, previous_shard)
    assert _consume_available(previous_shard, previous) == 1

    current_shard = tmp_path / "iter_00012.jsonl"
    current_shard.write_bytes(_row(1.0, "self_play", opponent="self"))
    current = _state_for_iteration(
        previous,
        run_name="run",
        iteration=12,
        shard=current_shard,
    )
    practice = _payload(
        current,
        stage="collect:self_play",
        active=False,
    )["strong_public_practice"]
    assert practice["available"] is True
    assert practice["active"] is False
    assert practice["iteration"] == 11
    assert practice["games"] == 1
    assert practice["stage"] == "collect:strong_public_practice:complete"


def test_tracker_carries_last_completed_research_result_into_next_iteration(
    tmp_path: Path,
) -> None:
    previous_shard = tmp_path / "iter_00007.jsonl"
    previous_shard.write_bytes(
        _row(1.0, "research_controls", opponent="iono", seat=0)
        + _row(-1.0, "research_controls", opponent="iono", seat=1)
    )
    previous = _new_state("run", 7, previous_shard)
    assert _consume_available(previous_shard, previous) == 2
    status = tmp_path / "public_mix_live_wr.json"
    _atomic_json(
        status,
        _payload(previous, stage="collect:research_controls:complete", active=False),
    )
    previous = _read_json(status)
    current_shard = tmp_path / "iter_00008.jsonl"
    current_shard.write_bytes(_row(1.0, "self_play", opponent="self"))
    current = _state_for_iteration(
        previous,
        run_name="run",
        iteration=8,
        shard=current_shard,
    )
    assert _consume_available(current_shard, current) == 0
    research = _payload(
        current,
        stage="collect:self_play",
        active=False,
    )["research_controls"]
    assert research["available"] is True
    assert research["active"] is False
    assert research["iteration"] == 7
    assert research["games"] == 2
    assert research["win_rate"] == 0.5
    assert research["stage"] == "collect:research_controls:complete"


def test_tracker_restart_during_new_iteration_does_not_merge_last_controls(
    tmp_path: Path,
) -> None:
    previous_shard = tmp_path / "iter_00007.jsonl"
    previous_shard.write_bytes(
        _row(1.0, "research_controls", opponent="old-a", seat=0)
        + _row(-1.0, "research_controls", opponent="old-b", seat=1)
    )
    previous = _new_state("run", 7, previous_shard)
    assert _consume_available(previous_shard, previous) == 2

    current_shard = tmp_path / "iter_00008.jsonl"
    current_shard.write_bytes(_row(1.0, "self_play", opponent="self"))
    current = _state_for_iteration(
        previous,
        run_name="run",
        iteration=8,
        shard=current_shard,
    )
    assert _consume_available(current_shard, current) == 0
    before_restart = _payload(current, stage="collect:self_play", active=False)
    assert before_restart["research_controls"]["iteration"] == 7
    assert before_restart["research_controls"]["games"] == 2

    # Exercise the real persistence shape: _payload is atomically serialized,
    # then a fresh process reads that JSON as its internal checkpoint.
    status = tmp_path / "public_mix_live_wr.json"
    _atomic_json(status, before_restart)
    restarted = _read_json(status)
    restarted = _state_for_iteration(
        restarted,
        run_name="run",
        iteration=8,
        shard=current_shard,
    )
    with current_shard.open("ab") as handle:
        handle.write(
            _row(
                0.0,
                "research_controls",
                opponent="new-only",
                seat=0,
                checkpoint_digest="sha256:model-b",
            )
        )
    assert _consume_available(current_shard, restarted) == 1

    after_restart = _payload(
        restarted,
        stage="collect:research_controls",
        active=False,
    )["research_controls"]
    assert after_restart["iteration"] == 8
    assert after_restart["games"] == 1
    assert after_restart["win_rate"] == 0.5
    assert [row["opponent_id"] for row in after_restart["matchups"]] == [
        "new-only"
    ]
    assert after_restart["checkpoint_digest"] == "sha256:model-b"


def test_tracker_restart_preserves_prior_display_until_current_row_arrives(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter_00008.jsonl"
    shard.write_bytes(_row(1.0, "self_play", opponent="self"))
    prior = {
        "games": 3,
        "wins": 2.0,
        "draws": 0,
        "losses": 1,
        "per_opponent": {
            "old-control": {
                "games": 3,
                "wins": 2.0,
                "draws": 0,
                "losses": 1,
                "seat0": 2,
                "seat1": 1,
            }
        },
        "checkpoint_digests": {"sha256:model-a": 3},
        "run": "run",
        "iteration": 7,
        "shard": str(tmp_path / "iter_00007.jsonl"),
    }
    state = _new_state(
        "run", 8, shard, last_completed_research_controls=prior
    )
    assert _consume_available(shard, state) == 0

    first_payload = _payload(state, stage="collect:self_play", active=False)
    status = tmp_path / "public_mix_live_wr.json"
    _atomic_json(status, first_payload)
    restarted = _read_json(status)
    assert _consume_available(shard, restarted) == 0
    second_payload = _payload(
        restarted, stage="collect:self_play", active=False
    )["research_controls"]
    assert second_payload["iteration"] == 7
    assert second_payload["games"] == 3
    assert second_payload["matchups"][0]["opponent_id"] == "old-control"


def test_same_iteration_replaced_inode_discards_partial_research_counter(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter_00008.jsonl"
    shard.write_bytes(
        _row(1.0, "research_controls", opponent="partial-current", seat=0)
    )
    prior = {
        "games": 2,
        "wins": 1.0,
        "draws": 0,
        "losses": 1,
        "per_opponent": {
            "completed-prior": {
                "games": 2,
                "wins": 1.0,
                "draws": 0,
                "losses": 1,
                "seat0": 1,
                "seat1": 1,
            }
        },
        "checkpoint_digests": {"sha256:model-a": 2},
        "run": "run",
        "iteration": 7,
        "shard": str(tmp_path / "iter_00007.jsonl"),
    }
    state = _new_state(
        "run", 8, shard, last_completed_research_controls=prior
    )
    assert _consume_available(shard, state) == 1
    assert state["_research_controls_current"]["games"] == 1

    status = tmp_path / "public_mix_live_wr.json"
    _atomic_json(
        status,
        _payload(state, stage="collect:research_controls", active=False),
    )
    restarted = _read_json(status)
    old_inode = shard.stat().st_ino
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(_row(1.0, "self_play", opponent="self"))
    replacement.replace(shard)
    assert shard.stat().st_ino != old_inode

    assert _consume_available(shard, restarted) == 0
    assert restarted["_research_controls_current"]["games"] == 0
    research = _payload(
        restarted, stage="collect:self_play", active=False
    )["research_controls"]
    assert research["iteration"] == 7
    assert research["games"] == 2
    assert research["stage"] == "collect:research_controls:complete"
    assert [row["opponent_id"] for row in research["matchups"]] == [
        "completed-prior"
    ]


def test_v1_state_is_rebuilt_for_per_opponent_metrics(tmp_path: Path) -> None:
    shard = tmp_path / "iter_00005.jsonl"
    shard.write_bytes(_row(1.0, "public_mix", opponent="hard-a", seat=1))
    state = {
        "schema": "poke_bot.public_mix_live_winrate/v1",
        "run": "run",
        "iteration": 5,
        "shard": str(shard),
        "inode": shard.stat().st_ino,
        "offset": shard.stat().st_size,
        "games": 99,
        "wins": 99.0,
    }

    assert _consume_available(shard, state) == 1
    assert state["games"] == 1
    assert state["per_opponent"]["hard-a"]["seat1"] == 1


def test_tracker_resumes_from_saved_offset(tmp_path: Path) -> None:
    shard = tmp_path / "iter_00000.jsonl"
    shard.write_bytes(_row(1.0, "public_mix"))
    state = _new_state("run", 0, shard)
    assert _consume_available(shard, state) == 1
    offset = state["offset"]
    assert _consume_available(shard, state) == 0
    assert state["offset"] == offset


def test_tracker_flags_mixed_behavior_checkpoint_identity(tmp_path: Path) -> None:
    shard = tmp_path / "iter_00005.jsonl"
    shard.write_bytes(
        _row(1.0, "public_mix", checkpoint_digest="sha256:model-a")
        + _row(1.0, "public_mix", checkpoint_digest="sha256:model-b")
    )
    state = _new_state("run", 5, shard)
    assert _consume_available(shard, state) == 2
    payload = _payload(state, stage="complete", active=False)
    assert payload["checkpoint_digest"] is None
    assert payload["checkpoint_mixed"] is True
