"""Pickle-safe entry points for remote WorkerPool job dispatch.

``scripts/run_remote_worker.py`` loads ``train_round_robin`` via importlib under the
synthetic module name ``train_round_robin_remote``. Multiprocessing (spawn) cannot
unpickle those callables in child workers — they raise
``ModuleNotFoundError: train_round_robin_remote``, die, leak leaf response slots,
and leave the farm stuck at ``jobs_completed=0`` while hello/health still pass.

These wrappers live in an importable package module so ``Pool.apply`` round-trips
cleanly under spawn.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_RR = None
_MODULE_NAME = "train_round_robin_remote"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_round_robin_module():
    """Load ``scripts/train_round_robin.py`` once per process (lazy)."""
    global _RR
    if _RR is not None:
        return _RR
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        _RR = existing
        return _RR
    path = _repo_root() / "scripts" / "train_round_robin.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    _RR = mod
    return _RR


def remote_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Whole-game collection job (importable; safe to ``Pool.apply``)."""
    return load_round_robin_module()._worker_play(job)


def remote_promotion_job(job: dict[str, Any]) -> dict[str, Any]:
    """Promotion evaluation job (importable; safe to ``Pool.apply``)."""
    return load_round_robin_module()._worker_promotion(job)


def remote_self_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Pure self-play: current policy vs recent-self / itself (spawn-safe).

    Primary Stage A collect path (Abhyuday: millions of self-play variations).
    Builds a trusted record from our seat only — no baseline heuristics.
    """
    import json
    import random
    import signal
    import time

    import torch

    from poke_bot import config as _config
    from poke_bot.agent import PolicyAgent, install_quiet_stdout, play_game
    from poke_bot.train import load_model_from_checkpoint

    our_seat = int(job.get("our_seat", 0))
    opp_id = str(job.get("opponent_id") or "self")
    seed = int(job["seed"])
    deck = list(job["our_deck"])
    timeout_s = int(job.get("game_timeout_s", 600))

    def _base(**over):
        r = {
            "job_index": int(job.get("job_index", 0)),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": 2,
            "value": 0.0,
            "steps": 0,
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "training_eligible": True,
            "record_json": None,
            "error": None,
            "self_play": True,
        }
        r.update(over)
        return r

    try:
        install_quiet_stdout(_config.agent_verbose())
        random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(job.get("device", "cpu"))
        us = str(job["checkpoint"])
        them = str(job.get("opponent_checkpoint") or us)

        # Prefer coalesced GPU leaf servers when WorkerPool registered a channel
        # (same pattern as round-robin ``_worker_play``). Official libcg still
        # steps on CPU; only network eval moves to the leaf farm.
        from poke_bot import batched_infer
        from poke_bot.pure_rl.leaf_self_play import plan_self_play_leaf_wiring

        leaf_backend = batched_infer.remote_leaf_backend_from_worker()
        plan = plan_self_play_leaf_wiring(
            us_checkpoint=us,
            them_checkpoint=them,
            leaf_channel_active=leaf_backend is not None,
        )

        state = load_round_robin_module()._WORKER_STATE
        us_model = None
        them_model = None
        if plan.load_us_local or plan.load_them_local:
            cache_key = f"self|{us}|{them}|{device}|{plan.mode}"
            if state.get("self_play_key") != cache_key:
                if plan.load_us_local:
                    state["self_play_us"] = load_model_from_checkpoint(
                        us, device=device
                    )
                else:
                    state["self_play_us"] = None
                if plan.load_them_local:
                    state["self_play_them"] = (
                        state["self_play_us"]
                        if them == us and plan.load_us_local
                        else load_model_from_checkpoint(them, device=device)
                    )
                else:
                    state["self_play_them"] = None
                state["self_play_key"] = cache_key
            us_model = state.get("self_play_us") if plan.load_us_local else None
            them_model = state.get("self_play_them") if plan.load_them_local else None

        us_leaf = leaf_backend if plan.use_leaf_for_us else None
        them_leaf = leaf_backend if plan.use_leaf_for_them else None
        temp = float(job.get("action_temperature", 1.0))
        sample = bool(job.get("sample_actions", True))
        ctx = int(job.get("model_max_context") or _config.MODEL.max_context)
        us_agent = PolicyAgent(
            model=us_model,
            deck=deck,
            use_mcts=False,
            max_sims=0,
            collect_targets=bool(job.get("training_eligible", True)),
            sample_actions=sample,
            action_temperature=temp,
            rng=random.Random(seed ^ 0xA11CE),
            device=device,
            leaf_backend=us_leaf,
            max_context_override=ctx,
            game_time_budget_s=float(timeout_s),
            game_watchdog_reserve_s=min(60.0, max(10.0, 0.1 * float(timeout_s))),
            strict_runtime=True,
        )
        them_agent = PolicyAgent(
            model=them_model,
            deck=list(job.get("opp_deck") or deck),
            use_mcts=False,
            max_sims=0,
            collect_targets=False,
            sample_actions=sample,
            action_temperature=temp,
            rng=random.Random(seed ^ 0xBEEF),
            device=device,
            leaf_backend=them_leaf,
            max_context_override=ctx,
            game_time_budget_s=float(timeout_s),
            game_watchdog_reserve_s=min(60.0, max(10.0, 0.1 * float(timeout_s))),
            strict_runtime=True,
        )
        us_agent.reset_game()
        them_agent.reset_game()

        def _on_timeout(_s, _f):
            raise TimeoutError(f"self-play game exceeded {timeout_s}s")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(timeout_s)
        t0 = time.perf_counter()
        try:
            if our_seat == 0:
                result = play_game(us_agent, them_agent, deck, list(job.get("opp_deck") or deck))
            else:
                result = play_game(them_agent, us_agent, list(job.get("opp_deck") or deck), deck)
        finally:
            if had_alarm:
                signal.alarm(0)
        wall_s = time.perf_counter() - t0
        if result.get("failed_seat") is not None:
            failed = int(result["failed_seat"])
            if failed == our_seat:
                return _base(
                    our_failed=True,
                    winner=1 - our_seat,
                    value=-1.0,
                    steps=int(result.get("steps", 0)),
                    error=str(result.get("error")),
                )
            return _base(
                winner=our_seat,
                value=1.0,
                steps=int(result.get("steps", 0)),
                error=f"opp-failed: {result.get('error')}",
            )
        if result.get("incomplete"):
            return _base(
                resource_error=True,
                game_timeout=True,
                steps=int(result.get("steps", 0)),
                error="incomplete",
            )
        winner = int(result["winner"])
        value = 0.0 if winner == 2 else (1.0 if winner == our_seat else -1.0)
        record = None
        if job.get("training_eligible", True) and us_agent.targets:
            rr = load_round_robin_module()
            record = rr._build_selfplay_record(
                us_agent.targets,
                our_deck=deck,
                our_seat=our_seat,
                value=value,
                opp_id=opp_id,
                archetype=str(job.get("archetype") or "core"),
                seed=seed,
                target_provenance={
                    **dict(job.get("target_provenance") or {}),
                    "pure_rl": True,
                    "self_play": True,
                    "soft_policy_targets": False,
                },
            )
        return {
            "job_index": int(job.get("job_index", 0)),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": winner,
            "value": value,
            "steps": int(result.get("steps", 0)),
            "wall_s": wall_s,
            "n_decisions": len(us_agent.targets),
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "training_eligible": True,
            "self_play": True,
            "leaf_remote": bool(plan.use_leaf_for_us or plan.use_leaf_for_them),
            "leaf_self_play_mode": plan.mode,
            "record_json": (
                json.dumps(record, separators=(",", ":")) if record else None
            ),
            "error": None,
        }
    except BaseException as exc:  # noqa: BLE001
        return _base(our_failed=True, error=f"{type(exc).__name__}: {exc}")
