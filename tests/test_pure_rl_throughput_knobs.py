"""Default-off throughput helpers preserve compact-RL semantics exactly."""

from __future__ import annotations

import pytest

from poke_bot import batched_infer, features, remote_sim_jobs, worker_pool
from poke_bot.pure_rl.shards import (
    CompactDecision,
    CompactGame,
    CompactShardWriter,
    compact_shard_write_buffer_bytes,
)


def _game(index: int) -> CompactGame:
    return CompactGame(
        episode_id=f"episode-{index}",
        seat=index % 2,
        archetype="alakazam",
        opp_archetype="alakazam",
        deck=[1] * 60,
        value=1.0,
        decisions=[
            CompactDecision(
                env_step=0,
                selected_index=0,
                n_options=1,
                action=[0],
                observation={"current": {"result": -1}},
            )
        ],
    )


def test_compact_shard_buffer_is_default_off_and_exact_after_flush(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("PURE_RL_SHARD_WRITE_BUFFER_BYTES", raising=False)
    assert compact_shard_write_buffer_bytes() == 0

    legacy = CompactShardWriter(tmp_path / "legacy.jsonl")
    buffered = CompactShardWriter(tmp_path / "buffered.jsonl", buffer_bytes=1 << 20)
    for index in range(8):
        game = _game(index)
        assert legacy.write_game(game) is True
        assert buffered.write_game(game) is False

    assert buffered.pending_bytes > 0
    assert buffered.flush_count == 0
    assert not buffered.path.exists()
    assert buffered.flush() is True
    assert buffered.flush() is False
    assert buffered.flush_count == 1
    assert legacy.flush_count == 8
    assert buffered.path.read_bytes() == legacy.path.read_bytes()
    assert buffered.n_games == legacy.n_games == 8
    assert buffered.n_decisions == legacy.n_decisions == 8


def test_compact_shard_buffer_env_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("PURE_RL_SHARD_WRITE_BUFFER_BYTES", str(128 * 1024 * 1024))
    assert compact_shard_write_buffer_bytes() == 64 * 1024 * 1024
    monkeypatch.setenv("PURE_RL_SHARD_WRITE_BUFFER_BYTES", "not-an-int")
    assert compact_shard_write_buffer_bytes() == 0


def test_immutable_deck_bag_matches_legacy_sparse_word(monkeypatch) -> None:
    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    deck = [1, 2, 2, 7]

    legacy = features.SparseVector()
    legacy.word_start()
    for card_id in deck:
        legacy.add(card_id, 0.25)
    legacy.add_pos(64)

    cached = features.SparseVector()
    cached.word_start()
    features.immutable_deck_bag(deck).append_to(cached)
    assert cached.index == legacy.index
    assert cached.value == legacy.value
    assert cached.offset == legacy.offset
    assert cached.pos == legacy.pos

    with pytest.raises(features.FeatureContractError, match="own deck card id"):
        features.immutable_deck_bag([0])


def test_leaf_deck_cache_and_cpu_affinity_are_both_default_off(monkeypatch) -> None:
    # This is a leaf-worker-local cache test, not a cg runtime integration
    # test. Keeping its tiny vocabulary synthetic makes the default-off
    # contract executable on developer hosts without a native simulator tree.
    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    batched_infer._cached_immutable_deck_bag.cache_clear()
    monkeypatch.delenv("POKEBOT_IMMUTABLE_DECK_ENCODING_CACHE", raising=False)
    assert batched_infer._maybe_cached_immutable_deck_bag([1] * 60) is None

    monkeypatch.setenv("POKEBOT_IMMUTABLE_DECK_ENCODING_CACHE", "1")
    first = batched_infer._maybe_cached_immutable_deck_bag([1] * 60)
    second = batched_infer._maybe_cached_immutable_deck_bag([1] * 60)
    assert first is not None
    assert first is second

    monkeypatch.delenv("POKEBOT_SIM_WORKER_CPU_AFFINITY", raising=False)
    assert worker_pool.apply_sim_worker_cpu_affinity() is None


def test_cpu_affinity_opt_in_uses_only_the_allowed_cpuset(monkeypatch) -> None:
    calls: list[tuple[int, set[int]]] = []

    class _Proc:
        _identity = (2,)

    monkeypatch.setenv("POKEBOT_SIM_WORKER_CPU_AFFINITY", "1")
    monkeypatch.setenv("POKEBOT_SIM_WORKER_CPUSET", "2,4")
    monkeypatch.setattr(
        worker_pool.os,
        "sched_getaffinity",
        lambda _pid: {1, 2, 4},
        raising=False,
    )
    monkeypatch.setattr(
        worker_pool.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((int(pid), set(cpus))),
        raising=False,
    )
    monkeypatch.setattr(worker_pool.mp, "current_process", lambda: _Proc())

    assert worker_pool.apply_sim_worker_cpu_affinity() == 4
    assert calls == [(0, {4})]


def test_remote_gc_interval_is_opt_in_and_receipted(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.delenv("POKEBOT_REMOTE_WORKER_GC_EVERY_JOBS", raising=False)
    monkeypatch.setattr(
        remote_sim_jobs.gc,
        "collect",
        lambda: calls.append(1) or 7,
    )
    default_state: dict[str, object] = {}
    assert remote_sim_jobs._collect_remote_worker_game_cycles(default_state) == 7
    assert remote_sim_jobs._collect_remote_worker_game_cycles(default_state) == 7
    assert calls == [1, 1]

    calls.clear()
    monkeypatch.setenv("POKEBOT_REMOTE_WORKER_GC_EVERY_JOBS", "3")
    state: dict[str, object] = {}
    assert remote_sim_jobs._collect_remote_worker_game_cycles(state) == 0
    assert remote_sim_jobs._collect_remote_worker_game_cycles(state) == 0
    assert remote_sim_jobs._collect_remote_worker_game_cycles(state) == 7
    assert calls == [1]
    lifecycle = state["_remote_worker_memory_lifecycle"]
    assert isinstance(lifecycle, dict)
    assert lifecycle["post_job_gc_checks"] == 3
    assert lifecycle["post_job_gc_runs"] == 1
