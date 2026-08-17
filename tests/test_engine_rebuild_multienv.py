"""Unit tests for engine_rebuild MultiEnv spike (no libcg required)."""

from __future__ import annotations

import pytest

from poke_bot.engine_rebuild import (
    FakeMultiEnv,
    ResetSpec,
    fingerprint_select,
    record_episode,
    transition_hash,
)
from poke_bot.engine_rebuild.parity import assert_parity


def _deck(fill: int = 119) -> list[int]:
    return [fill] * 60


def test_fake_multienv_batch_step_and_protocol():
    env = FakeMultiEnv(4, target=3)
    assert env.num_envs == 4
    specs = [ResetSpec(_deck(1), _deck(2), seed=i) for i in range(4)]
    batch = env.reset(specs)
    assert len(batch) == 4
    assert all(not e.done for e in batch.envs)

    # Step only even slots; odd wait (None).
    actions = [[0], None, [1], None]
    batch2 = env.step_batch(actions)
    assert batch2.envs[0].obs["current"]["count"] != batch.envs[0].obs["current"]["count"]
    assert batch2.envs[1].obs["current"]["count"] == batch.envs[1].obs["current"]["count"]
    env.close()
    with pytest.raises(RuntimeError):
        env.step_batch([[0]] * 4)


def test_fake_multienv_reaches_done():
    env = FakeMultiEnv(1, target=2)
    env.reset([ResetSpec(_deck(), _deck(), seed=0)])
    # seed 0 → count 0; each step +1 or +2 → done quickly
    b = env.step_batch([[0]])
    if not b.envs[0].done:
        b = env.step_batch([[0]])
    assert b.envs[0].done
    assert b.envs[0].obs["select"] is None


def test_parity_hash_stable_and_detects_divergence():
    from dataclasses import replace

    env = FakeMultiEnv(1, target=3)
    obs0 = env.reset([ResetSpec(_deck(), _deck(), seed=1)]).envs[0].obs
    obs1 = env.step_batch([[0]]).envs[0].obs
    obs2 = env.step_batch([[1]]).envs[0].obs
    records = record_episode(seed=1, steps=[(obs0, [0], obs1), (obs1, [1], obs2)])
    h = transition_hash(records)
    assert len(h) == 64
    assert_parity(records, list(records))

    tweaked = list(records)
    tweaked[0] = replace(tweaked[0], action=[1])
    with pytest.raises(AssertionError, match="parity fail"):
        assert_parity(records, tweaked, label="fake-fork")


def test_fingerprint_terminal_vs_select():
    term = fingerprint_select({"current": {"result": 0}, "select": None})
    live = fingerprint_select(
        {
            "select": {
                "type": 1,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"id": 3, "area": 2}],
            }
        }
    )
    assert term.startswith("terminal:")
    assert len(live) == 16
    assert term != live
