"""Unit tests for multi-env / leaf coalesce helpers (no torch, no libcg)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "poke_bot" / "pure_rl" / "multi_env_self_play.py"
_SPEC = importlib.util.spec_from_file_location("poke_bot_multi_env_self_play", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

chunk_jobs = _mod.chunk_jobs
process_worker_count = _mod.process_worker_count
pure_rl_leaf_coalesce_ms = _mod.pure_rl_leaf_coalesce_ms
resolve_multi_env_per_worker = _mod.resolve_multi_env_per_worker
terminal_policy_failure_outcome = _mod.terminal_policy_failure_outcome


def test_resolve_multi_env_default_off(monkeypatch) -> None:
    monkeypatch.delenv("POKEBOT_MULTI_ENV", raising=False)
    monkeypatch.delenv("PURE_RL_MULTI_ENV", raising=False)
    monkeypatch.delenv("POKEBOT_MULTI_ENV_PER_WORKER", raising=False)
    monkeypatch.delenv("PURE_RL_MULTI_ENV_PER_WORKER", raising=False)
    assert resolve_multi_env_per_worker(None) == 1


def test_resolve_multi_env_flag_defaults_to_four(monkeypatch) -> None:
    monkeypatch.delenv("POKEBOT_MULTI_ENV_PER_WORKER", raising=False)
    monkeypatch.delenv("PURE_RL_MULTI_ENV_PER_WORKER", raising=False)
    monkeypatch.setenv("POKEBOT_MULTI_ENV", "1")
    assert resolve_multi_env_per_worker(None) == 4


def test_resolve_multi_env_cli_wins(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_MULTI_ENV", "1")
    monkeypatch.setenv("POKEBOT_MULTI_ENV_PER_WORKER", "8")
    assert resolve_multi_env_per_worker(2) == 2


def test_process_worker_count_keeps_games_in_flight() -> None:
    assert process_worker_count(40, 1) == 40
    assert process_worker_count(40, 4) == 10
    assert process_worker_count(40, 3) == 13
    assert process_worker_count(5, 8) == 1


def test_chunk_jobs() -> None:
    jobs = list(range(10))
    chunks = chunk_jobs(jobs, 4)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_pure_rl_leaf_coalesce_default_zero(monkeypatch) -> None:
    monkeypatch.delenv("PURE_RL_LEAF_COALESCE_MS", raising=False)
    monkeypatch.setenv("LEAF_SERVER_COALESCE_MS", "4")  # RR leftover must not win
    assert pure_rl_leaf_coalesce_ms() == 0.0


def test_pure_rl_leaf_coalesce_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PURE_RL_LEAF_COALESCE_MS", "1")
    monkeypatch.setenv("LEAF_SERVER_COALESCE_MS", "4")
    assert pure_rl_leaf_coalesce_ms() == 1.0


def test_terminal_policy_failure_retains_correct_win_loss_value() -> None:
    assert terminal_policy_failure_outcome(failed_seat=0, our_seat=0) == (1, -1.0)
    assert terminal_policy_failure_outcome(failed_seat=1, our_seat=0) == (0, 1.0)
    with __import__("pytest").raises(ValueError):
        terminal_policy_failure_outcome(failed_seat=2, our_seat=0)
