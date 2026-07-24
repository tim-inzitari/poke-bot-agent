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
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_RR = None
_MODULE_NAME = "train_round_robin_remote"
_CONTROLLER_MATCHUP_RUNTIME_KEY = "_controller_matchup_runtime"


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


def _bind_controller_matchup_runtime(job: dict[str, Any]) -> None:
    """Reassert the controller-owned routing contract before every remote job.

    WorkerPool children are intentionally recycled.  They can also execute
    frozen specialist opponents whose submission packages use the same
    process-global environment variables as the active candidate.  Binding
    from the server-injected contract at each job boundary prevents either
    lifecycle from carrying an opponent's tree into the next candidate game.
    The server overwrites this reserved field, so clients cannot select a tree.
    """

    if _CONTROLLER_MATCHUP_RUNTIME_KEY not in job:
        # Compatibility for direct unit tests and non-server callers.
        return
    runtime = job.get(_CONTROLLER_MATCHUP_RUNTIME_KEY)
    if runtime is None:
        os.environ.pop("POKEBOT_MATCHUP_ADAPTER_RUNTIME", None)
        os.environ.pop("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", None)
        os.environ.pop("POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE", None)
        return
    if not isinstance(runtime, dict):
        raise ValueError("controller matchup runtime contract must be an object")
    tree_path = Path(str(runtime.get("tree") or "")).expanduser().resolve()
    expected_digest = str(runtime.get("tree_digest") or "")
    expected_ids = sorted(
        str(value) for value in runtime.get("accepted_archetype_ids") or ()
    )
    if not tree_path.is_file() or not expected_digest or not expected_ids:
        raise ValueError("controller matchup runtime contract is incomplete")
    raw = tree_path.read_bytes()
    actual_digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if actual_digest != expected_digest:
        raise ValueError(
            "controller matchup runtime tree digest changed: "
            f"expected={expected_digest} actual={actual_digest}"
        )
    payload = json.loads(raw)
    actual_ids = sorted(
        str(value)
        for value in dict(payload.get("runtime_contract") or {}).get(
            "accepted_archetype_ids", ()
        )
    )
    if actual_ids != expected_ids:
        raise ValueError(
            "controller matchup runtime accepted-route roster changed"
        )
    os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(tree_path)
    os.environ["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] = "runtime"


def remote_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Whole-game collection job (importable; safe to ``Pool.apply``)."""
    _bind_controller_matchup_runtime(job)
    return load_round_robin_module()._worker_play(job)


def remote_promotion_job(job: dict[str, Any]) -> dict[str, Any]:
    """Promotion evaluation job (importable; safe to ``Pool.apply``)."""
    _bind_controller_matchup_runtime(job)
    return load_round_robin_module()._worker_promotion(job)


def remote_matchup_runtime_probe(_job: dict[str, Any]) -> dict[str, Any]:
    """Report the routing identity actually inherited by one pool child.

    Controller hello metadata lives in the parent process.  A hot marker
    update can therefore make hello look current while long-lived simulator
    children still use the prior tree.  This probe crosses the process-pool
    boundary so the trainer can fail before collecting a mixed-routing shard.
    """

    _bind_controller_matchup_runtime(_job)

    runtime_enabled = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    tree_path = Path(
        os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "")
    ).expanduser()
    if not runtime_enabled or not tree_path.is_file():
        return {
            "runtime_probe": {
                "runtime_enabled": runtime_enabled,
                "tree": str(tree_path),
                "tree_digest": None,
                "accepted_archetype_ids": [],
            },
            "error": None,
        }
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    accepted = sorted(
        str(value)
        for value in dict(payload.get("runtime_contract") or {}).get(
            "accepted_archetype_ids", ()
        )
    )
    digest = hashlib.sha256(tree_path.read_bytes()).hexdigest()
    return {
        "runtime_probe": {
            "runtime_enabled": True,
            "tree": str(tree_path.resolve()),
            "tree_digest": f"sha256:{digest}",
            "accepted_archetype_ids": accepted,
        },
        "error": None,
    }


def remote_self_play_multi_job(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Batch self-play via in-process ``LibcgMultiEnv`` (spawn-safe).

    ``batch`` is ``{"jobs": [job, ...]}``. Used when pure-RL enables
    ``--multi-env-per-worker`` / ``POKEBOT_MULTI_ENV`` so one OS worker owns
    many official libcg handles. GPU leaf wiring matches
    :func:`remote_self_play_job`.
    """
    from poke_bot.pure_rl.multi_env_self_play import run_self_play_multi

    jobs = list(batch.get("jobs") or [])
    if not jobs:
        return []
    _bind_controller_matchup_runtime(jobs[0])
    if len(jobs) == 1:
        return [remote_self_play_job(jobs[0])]
    return run_self_play_multi(jobs)


def remote_self_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Pure self-play: current policy vs recent-self / itself (spawn-safe).

    Primary Stage A collect path (Abhyuday: millions of self-play variations).
    Builds a trusted record from our seat only — no baseline heuristics.

    When the WorkerPool registered a leaf channel, both seats use coalesced
    GPU leaves for same-checkpoint games (``gpu-leaf-both``); recent-self
    opponents stay CPU-local on the opp seat (``gpu-leaf-us-only``).
    """
    _bind_controller_matchup_runtime(job)
    # Exact hidden-zone targets require the explicit private training engine.
    # Route even a single game through the multi-handle adapter when that ABI
    # is configured; hosts using the official engine continue normally and
    # their unavailable belief labels are masked during training.
    if bool(job.get("collect_privileged_belief", False)) and os.environ.get(
        "POKEBOT_LIBCG_PATH", ""
    ).strip():
        from poke_bot.pure_rl.multi_env_self_play import run_self_play_multi

        return run_self_play_multi([job])[0]

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
            "record_jsons": None,
            "error": None,
            "self_play": True,
            "matchup_runtime_audit": None,
            "opponent_matchup_runtime_audit": None,
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
        collect_both = bool(
            job.get("training_eligible", True)
            and job.get("collect_both_seats", False)
            and them == us
        )

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
            collect_targets=collect_both,
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
        terminal_policy_failure = result.get("failed_seat") is not None
        terminal_failure_error = None
        if terminal_policy_failure:
            failed = int(result["failed_seat"])
            from poke_bot.pure_rl.multi_env_self_play import (
                terminal_policy_failure_outcome,
            )

            winner, value = terminal_policy_failure_outcome(
                failed_seat=failed, our_seat=our_seat
            )
            terminal_failure_error = str(
                result.get("error") or "policy action failure"
            )
        else:
            if result.get("incomplete"):
                return _base(
                    resource_error=True,
                    game_timeout=True,
                    steps=int(result.get("steps", 0)),
                    error="incomplete",
                )
            winner = int(result["winner"])
            value = (
                0.0 if winner == 2 else (1.0 if winner == our_seat else -1.0)
            )
        matchup_runtime_audit = us_agent.matchup_adapter_shadow_snapshot()
        opponent_matchup_runtime_audit = (
            them_agent.matchup_adapter_shadow_snapshot()
        )
        record = None
        records = []
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
                    "terminal_policy_failure": terminal_policy_failure,
                    "terminal_failed_seat": (
                        int(result["failed_seat"])
                        if terminal_policy_failure
                        else None
                    ),
                    "matchup_runtime_audit": matchup_runtime_audit,
                },
                opp_archetype=(
                    str(job.get("opp_archetype") or "") or None
                ),
            )
            if record is not None:
                records.append(record)
        if collect_both and them_agent.targets:
            rr = load_round_robin_module()
            opp_deck = list(job.get("opp_deck") or deck)
            opp_value = 0.0 if winner == 2 else -value
            opp_record = rr._build_selfplay_record(
                them_agent.targets,
                our_deck=opp_deck,
                our_seat=1 - our_seat,
                value=opp_value,
                opp_id=f"self:{Path(us).name}",
                archetype=str(
                    job.get("opp_archetype")
                    or job.get("archetype")
                    or "core"
                ),
                seed=seed ^ 0x51DE,
                target_provenance={
                    **dict(job.get("target_provenance") or {}),
                    "pure_rl": True,
                    "self_play": True,
                    "same_policy_second_seat": True,
                    "soft_policy_targets": False,
                    "terminal_policy_failure": terminal_policy_failure,
                    "terminal_failed_seat": (
                        int(result["failed_seat"])
                        if terminal_policy_failure
                        else None
                    ),
                    "matchup_runtime_audit": opponent_matchup_runtime_audit,
                },
                opp_archetype=(
                    str(job.get("archetype") or "") or None
                ),
            )
            if opp_record is not None:
                records.append(opp_record)
        return {
            "job_index": int(job.get("job_index", 0)),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": winner,
            "value": value,
            "steps": int(result.get("steps", 0)),
            "wall_s": wall_s,
            "n_decisions": sum(len(r.get("steps") or []) for r in records),
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "training_eligible": True,
            "self_play": True,
            "leaf_remote": bool(plan.use_leaf_for_us or plan.use_leaf_for_them),
            "leaf_self_play_mode": plan.mode,
            "matchup_runtime_audit": matchup_runtime_audit,
            "opponent_matchup_runtime_audit": opponent_matchup_runtime_audit,
            "policy_terminal_failure": terminal_policy_failure,
            "failed_seat": (
                int(result["failed_seat"])
                if terminal_policy_failure
                else None
            ),
            "record_json": (
                json.dumps(record, separators=(",", ":")) if record else None
            ),
            "record_jsons": [
                json.dumps(row, separators=(",", ":")) for row in records
            ],
            "error": terminal_failure_error,
        }
    except BaseException as exc:  # noqa: BLE001
        return _base(our_failed=True, error=f"{type(exc).__name__}: {exc}")
