"""Pickle-safety for remote WorkerPool job entry points."""

from __future__ import annotations

import multiprocessing as mp
import pickle

from poke_bot.remote_sim_jobs import remote_play_job, remote_promotion_job


def test_remote_job_callables_are_importable_module_attrs() -> None:
    """Spawn Pool unpickles by module+name; synthetic RR module is not importable."""
    assert remote_play_job.__module__ == "poke_bot.remote_sim_jobs"
    assert remote_promotion_job.__module__ == "poke_bot.remote_sim_jobs"
    assert remote_play_job.__name__ == "remote_play_job"
    assert remote_promotion_job.__name__ == "remote_promotion_job"


def test_remote_job_callables_survive_pickle_roundtrip() -> None:
    for fn in (remote_play_job, remote_promotion_job):
        restored = pickle.loads(pickle.dumps(fn))
        assert restored is fn or restored.__module__ == fn.__module__
        assert restored.__name__ == fn.__name__


def _identity(x: int) -> int:
    return x


def test_pool_apply_accepts_importable_callable() -> None:
    """Guard the remote dispatch pattern: apply(importable_fn, arg)."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        assert pool.apply(_identity, (7,)) == 7
