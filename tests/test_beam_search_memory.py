from __future__ import annotations

import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from poke_agent.beam_search import BeamSearchConfig, run_beam_search
from poke_agent.policy_agent import PolicyRuntime, PolicySession


def test_self_play_beam_config_is_short_and_sim_mode():
    config = BeamSearchConfig.from_self_play_config({
        "self_play": {
            "beam_width": 8,
            "beam_time_budget_ms": 1500,
            "beam_max_search_steps": 128,
            "beam_rollout_policy_width": 12,
            "search_determinizations": 2,
        },
    })
    assert config.sim_mode is True
    assert config.width == 8
    assert config.time_budget_ms == 1500
    assert config.max_search_steps == 128
    assert config.rollout_policy_width == 12
    assert config.num_determinizations == 2


def test_choose_action_caps_session_history(monkeypatch):
    runtime = PolicyRuntime(Path("unused.pt"), device="cpu")
    runtime._loaded = True
    runtime._window_size = 4
    runtime._feature_mean = np.zeros(8, dtype=np.float32)
    runtime._feature_std = np.ones(8, dtype=np.float32)
    runtime._policy_dim = 8
    runtime._model = MagicMock()

    monkeypatch.setattr(
        runtime,
        "_encode_observation",
        lambda obs, session, our_deck=None: (
            np.arange(8, dtype=np.float32),
            np.zeros(30, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(runtime, "_model_logits", lambda session: np.zeros(8, dtype=np.float32))
    monkeypatch.setattr(runtime, "_choose_from_policy_logits", lambda logits, actions: actions[0])

    session = PolicySession()
    obs = {
        "select": {"option": [0, 1], "minCount": 1, "maxCount": 1},
        "current": {"yourIndex": 0},
    }
    for _ in range(50):
        runtime.choose_action(obs, session, use_beam=False)

    assert len(session.history) == runtime._window_size


def test_run_beam_search_does_not_grow_python_heap_unboundedly(monkeypatch):
    runtime = PolicyRuntime(Path("unused.pt"), device="cpu")
    runtime._loaded = True
    runtime._window_size = 4
    runtime._feature_mean = np.zeros(8, dtype=np.float32)
    runtime._feature_std = np.ones(8, dtype=np.float32)
    runtime._policy_dim = 8
    runtime._state_hash_dim = 4
    runtime._model = MagicMock()

    monkeypatch.setattr(runtime, "_model_logits", lambda session: np.zeros(8, dtype=np.float32))
    monkeypatch.setattr(
        "poke_agent.beam_search.expand_action_in_search",
        lambda *args, **kwargs: 0.25,
    )

    session = PolicySession()
    obs = {
        "search_begin_input": True,
        "select": {"option": [0, 1, 2], "minCount": 1, "maxCount": 1},
        "current": {"yourIndex": 0, "result": -1},
    }
    deck = list(range(60))
    config = BeamSearchConfig(sim_mode=True, width=3, time_budget_ms=150, max_search_steps=8)

    tracemalloc.start()
    try:
        baseline_current, baseline_peak = tracemalloc.get_traced_memory()
        for _ in range(300):
            run_beam_search(runtime, obs, session, deck, [[0], [1], [2]], 0, config)
        end_current, end_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Allow modest growth from session history cap and allocator noise, not unbounded leak.
    assert end_current - baseline_current < 5_000_000
    assert end_peak - baseline_peak < 10_000_000


def _checkpoint_compatible(path: Path) -> bool:
    if not path.exists():
        return False
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") or {}
    return "card_embed.weight" in state


@pytest.mark.skipif(
    not _checkpoint_compatible(Path("outputs/checkpoints/pre_self_train.pt")),
    reason="requires phase-1 checkpoint with card embeddings",
)
def test_beam_choose_action_rss_stable_with_real_checkpoint():
    """RSS should not balloon after many short beam decisions on a real checkpoint."""
    import resource

    checkpoint = Path("outputs/checkpoints/pre_self_train.pt")
    runtime = PolicyRuntime(checkpoint, device="cpu")
    session = PolicySession()
    deck = list(range(60))
    config = BeamSearchConfig.from_self_play_config({"self_play": {}})

    obs = {
        "search_begin_input": True,
        "select": {"option": [0, 1], "minCount": 1, "maxCount": 1},
        "current": {"yourIndex": 0, "result": -1, "players": []},
    }

    def rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    baseline = rss_mb()
    for _ in range(100):
        runtime.choose_action(
            obs,
            session,
            our_deck=deck,
            use_beam=True,
            beam_config=config,
        )
        session.history.clear()

    # Linux ru_maxrss is peak since process start; allow generous headroom vs tiny baseline.
    assert rss_mb() < max(baseline + 512, 2048)
