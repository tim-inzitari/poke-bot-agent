"""Unit tests for pure-RL self-play ↔ GPU leaf wiring plans."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the module file directly so this test does not require torch (pure_rl
# package __init__ imports model_profile → torch).
_PATH = Path(__file__).resolve().parents[1] / "poke_bot" / "pure_rl" / "leaf_self_play.py"
_SPEC = importlib.util.spec_from_file_location("poke_bot_leaf_self_play", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)
expected_leaf_share = _mod.expected_leaf_share
plan_self_play_leaf_wiring = _mod.plan_self_play_leaf_wiring


def test_cpu_local_when_no_leaf_channel() -> None:
    plan = plan_self_play_leaf_wiring(
        us_checkpoint="/ckpt/a.pt",
        them_checkpoint="/ckpt/a.pt",
        leaf_channel_active=False,
    )
    assert plan.mode == "cpu-local"
    assert plan.load_us_local and plan.load_them_local
    assert not plan.use_leaf_for_us
    assert expected_leaf_share(plan) == 0.0


def test_gpu_leaf_both_when_same_checkpoint() -> None:
    plan = plan_self_play_leaf_wiring(
        us_checkpoint="/ckpt/a.pt",
        them_checkpoint="/ckpt/a.pt",
        leaf_channel_active=True,
    )
    assert plan.mode == "gpu-leaf-both"
    assert plan.use_leaf_for_us and plan.use_leaf_for_them
    assert not plan.load_us_local and not plan.load_them_local
    assert expected_leaf_share(plan) == 1.0


def test_gpu_leaf_us_only_when_opponent_pool_differs() -> None:
    plan = plan_self_play_leaf_wiring(
        us_checkpoint="/ckpt/champ.pt",
        them_checkpoint="/ckpt/recent.pt",
        leaf_channel_active=True,
    )
    assert plan.mode == "gpu-leaf-us-only"
    assert plan.use_leaf_for_us and not plan.use_leaf_for_them
    assert not plan.load_us_local and plan.load_them_local
    assert expected_leaf_share(plan) == 0.5
