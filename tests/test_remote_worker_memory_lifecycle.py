"""Regression coverage for bounded r182 remote-worker residency."""

from __future__ import annotations

import os

from poke_bot import remote_sim_jobs
from poke_bot.worker_pool import WorkerPool, recycle_task_limit


def _job(
    checkpoint: str,
    digest: str,
    *,
    opponent_checkpoint: str | None = None,
) -> dict[str, str]:
    job = {"checkpoint": checkpoint, "checkpoint_digest": digest}
    if opponent_checkpoint is not None:
        job["opponent_checkpoint"] = opponent_checkpoint
    return job


def test_packed_worker_recycle_budget_stays_a_game_budget() -> None:
    """A full four-game packet cannot stretch 256 games to 1,024 tasks."""

    assert recycle_task_limit(recycle_games=256, max_games_per_task=1) == 256
    assert recycle_task_limit(recycle_games=256, max_games_per_task=4) == 64
    # A partial final packet may retire early, never past the game budget.
    assert recycle_task_limit(recycle_games=256, max_games_per_task=3) == 85
    pool = WorkerPool(
        num_workers=1,
        recycle_games=256,
        max_games_per_task=4,
    )
    assert pool.max_tasks_per_child == 64


def test_multi_env_cache_retains_only_current_packet_models() -> None:
    """The current parent plus at-most-four packet opponents is the hard bound."""

    active = "/checkpoints/current.pt"
    first_opp = "/checkpoints/last-one.pt"
    next_opps = [f"/checkpoints/last-{index}.pt" for index in range(2, 6)]
    state: dict[str, object] = {}

    assert remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        [_job(active, "sha256:active", opponent_checkpoint=first_opp)],
        active_cache_family="multi_env",
    )
    state["multi_env_models"] = {
        active: object(),
        first_opp: object(),
        "/checkpoints/stale.pt": object(),
    }

    packet = [
        _job(active, "sha256:active", opponent_checkpoint=opponent)
        for opponent in next_opps
    ]
    assert remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        packet,
        active_cache_family="multi_env",
    )

    cache = state["multi_env_models"]
    assert isinstance(cache, dict)
    assert set(cache) == {active}
    assert len(cache) <= 1 + len(packet)


def test_checkpoint_digest_change_flushes_same_path_model_cache() -> None:
    """An atomic active-path replacement never reuses old in-process weights."""

    checkpoint = "/checkpoint/active.pt"
    state: dict[str, object] = {}
    remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        [_job(checkpoint, "sha256:before")],
        active_cache_family="multi_env",
    )
    state["multi_env_models"] = {checkpoint: object()}
    state["key"] = f"{checkpoint}|cpu|rtp=1"
    state["model"] = object()

    assert remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        [_job(checkpoint, "sha256:after")],
        active_cache_family="multi_env",
    )
    assert state["multi_env_models"] == {}
    assert "key" not in state
    assert "model" not in state


def test_cache_family_transition_drops_duplicate_current_parent() -> None:
    """A child cannot retain the same parent in play and multi-env caches."""

    checkpoint = "/checkpoints/current.pt"
    state: dict[str, object] = {}
    remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        [_job(checkpoint, "sha256:current")],
        active_cache_family="play",
    )
    state["key"] = f"{checkpoint}|cpu|rtp=1"
    state["model"] = object()

    assert remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        [_job(checkpoint, "sha256:current")],
        active_cache_family="multi_env",
    )
    assert "key" not in state
    assert "model" not in state


def test_singleton_self_play_multi_uses_its_actual_cache_family() -> None:
    """A one-game tail cannot retain the preceding packed H10 cache."""

    checkpoint = "/checkpoints/current.pt"
    state: dict[str, object] = {}
    remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        [_job(checkpoint, "sha256:current")],
        active_cache_family="multi_env",
    )
    state["multi_env_models"] = {checkpoint: object()}

    jobs = [_job(checkpoint, "sha256:current")]
    family = remote_sim_jobs._remote_self_play_multi_cache_family(jobs)
    assert family == "self_play"
    assert remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        jobs,
        active_cache_family=family,
    )
    assert state["multi_env_models"] == {}


def test_privileged_single_self_play_uses_multi_env_cache_family() -> None:
    """The direct privileged path cannot leave multi-env models mislabeled."""

    # Keep the test independent from a real training libcg: classification
    # only checks that the explicit private-runtime path is armed.
    previous = os.environ.get("POKEBOT_LIBCG_PATH")
    try:
        os.environ["POKEBOT_LIBCG_PATH"] = "/tmp/libcg-private.so"
        job = _job("/checkpoints/current.pt", "sha256:current")
        job["collect_privileged_belief"] = "1"
        assert (
            remote_sim_jobs._remote_self_play_cache_family(job)
            == "multi_env_self_play"
        )
    finally:
        if previous is None:
            os.environ.pop("POKEBOT_LIBCG_PATH", None)
        else:
            os.environ["POKEBOT_LIBCG_PATH"] = previous


def test_multi_env_transport_transition_releases_rekeyed_parent() -> None:
    """Raw self-play and decorated public keys cannot coexist across a phase."""

    checkpoint = "/checkpoints/current.pt"
    state: dict[str, object] = {}
    jobs = [_job(checkpoint, "sha256:current")]
    remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        jobs,
        active_cache_family="multi_env_self_play",
    )
    state["multi_env_models"] = {checkpoint: object()}

    assert remote_sim_jobs._advance_remote_worker_checkpoint_scope(
        state,
        jobs,
        active_cache_family="multi_env_play",
    )
    assert state["multi_env_models"] == {}


def test_self_play_multi_rejects_a_packet_above_the_memory_bound() -> None:
    """The cache/recycle calculation is valid for every admitted packet."""

    jobs = [{"job_index": index} for index in range(5)]
    try:
        remote_sim_jobs.remote_self_play_multi_job({"jobs": jobs})
    except ValueError as exc:
        assert "1..4" in str(exc)
    else:  # pragma: no cover - documents the fail-closed contract
        raise AssertionError("oversized self-play packet was admitted")
