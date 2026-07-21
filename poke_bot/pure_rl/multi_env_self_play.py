"""In-process multi-battle self-play for pure-RL collect (LibcgMultiEnv).

Default collect still uses one OS process per game via ``WorkerPool``. When
``POKEBOT_MULTI_ENV=1`` / ``--multi-env-per-worker N`` is set, each worker
process runs ``N`` concurrent official ``libcg`` handles and interleaves
policy steps — fewer processes for the same games-in-flight.

GPU leaf wiring reuses :func:`plan_self_play_leaf_wiring` (same as the
single-game ``remote_self_play_job`` path).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pure_rl_leaf_coalesce_ms(*, default: float = 0.0) -> float:
    """Coalesce window for pure-RL leaf servers only.

    Tiny ~1.6M policies lose to CPU-local at the global RR default (4 ms).
    Only ``PURE_RL_LEAF_COALESCE_MS`` is read here so a shell still exporting
    ``LEAF_SERVER_COALESCE_MS=4`` for Hope-large RR cannot slow pure-RL.
    ``launch_pure_rl`` setdefaults ``PURE_RL_LEAF_COALESCE_MS=0``.
    """
    raw = os.environ.get("PURE_RL_LEAF_COALESCE_MS")
    if raw is not None and str(raw).strip() != "":
        try:
            return float(raw)
        except ValueError:
            pass
    return float(default)


def resolve_multi_env_per_worker(
    cli_value: Optional[int] = None,
    *,
    default_when_enabled: int = 4,
) -> int:
    """Return envs/process (1 = classic one-battle WorkerPool).

    Priority: CLI ``--multi-env-per-worker`` → ``PURE_RL_MULTI_ENV_PER_WORKER``
    / ``POKEBOT_MULTI_ENV_PER_WORKER`` → if ``POKEBOT_MULTI_ENV``/``PURE_RL_MULTI_ENV``
    truthy then ``default_when_enabled`` → else 1.
    """
    if cli_value is not None:
        return max(1, int(cli_value))
    for key in ("PURE_RL_MULTI_ENV_PER_WORKER", "POKEBOT_MULTI_ENV_PER_WORKER"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return max(1, _env_int(key, 1))
    if _env_truthy("PURE_RL_MULTI_ENV") or _env_truthy("POKEBOT_MULTI_ENV"):
        return max(1, int(default_when_enabled))
    return 1


def process_worker_count(sim_workers: int, multi_env_per_worker: int) -> int:
    """Fewer OS processes so ``procs × multi ≈ sim_workers`` games-in-flight."""
    n = max(1, int(sim_workers))
    m = max(1, int(multi_env_per_worker))
    if m <= 1:
        return n
    return max(1, n // m)


def chunk_jobs(jobs: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    """Split jobs into batches of ``chunk_size`` (last batch may be shorter)."""
    size = max(1, int(chunk_size))
    out: list[list[Any]] = []
    buf: list[Any] = []
    for job in jobs:
        buf.append(job)
        if len(buf) >= size:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


def run_self_play_multi(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Play ``jobs`` concurrently in one process via :class:`LibcgMultiEnv`.

    Requires competition ``cg`` / ``libcg`` on ``PYTHONPATH``. Returns one
    result dict per input job (same shape as ``remote_self_play_job``).
    """
    import json
    import random
    import time

    import torch

    from poke_bot import batched_infer, config as _config
    from poke_bot.agent import PolicyAgent, install_quiet_stdout
    from poke_bot.engine_rebuild.interfaces import Action, ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv
    from poke_bot.pure_rl.leaf_self_play import plan_self_play_leaf_wiring
    from poke_bot.remote_sim_jobs import load_round_robin_module
    from poke_bot.train import load_model_from_checkpoint

    if not jobs:
        return []

    install_quiet_stdout(_config.agent_verbose())
    leaf_backend = batched_infer.remote_leaf_backend_from_worker()
    rr = load_round_robin_module()
    state = rr._WORKER_STATE
    model_cache: dict[str, Any] = state.setdefault("multi_env_models", {})

    n = len(jobs)
    env = LibcgMultiEnv(n)
    max_steps = 4000
    agents0: list[Optional[PolicyAgent]] = [None] * n
    agents1: list[Optional[PolicyAgent]] = [None] * n
    our_seats = [int(job.get("our_seat", 0)) for job in jobs]
    plans = []
    steps = [0] * n
    failed_seat: list[Optional[int]] = [None] * n
    errors: list[Optional[str]] = [None] * n
    finished = [False] * n
    t0 = time.perf_counter()

    def _base(job: dict[str, Any], **over: Any) -> dict[str, Any]:
        r = {
            "job_index": int(job.get("job_index", 0)),
            "opponent_id": str(job.get("opponent_id") or "self"),
            "our_seat": int(job.get("our_seat", 0)),
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
            "multi_env": True,
        }
        r.update(over)
        return r

    try:
        specs: list[ResetSpec] = []
        for i, job in enumerate(jobs):
            seed = int(job["seed"])
            random.seed(seed)
            torch.manual_seed(seed)
            deck = list(job["our_deck"])
            opp_deck = list(job.get("opp_deck") or deck)
            our_seat = our_seats[i]
            if our_seat == 0:
                specs.append(ResetSpec(deck, opp_deck, seed=seed))
            else:
                specs.append(ResetSpec(opp_deck, deck, seed=seed))

            us = str(job["checkpoint"])
            them = str(job.get("opponent_checkpoint") or us)
            collect_both = bool(
                job.get("training_eligible", True)
                and job.get("collect_both_seats", False)
                and them == us
            )
            plan = plan_self_play_leaf_wiring(
                us_checkpoint=us,
                them_checkpoint=them,
                leaf_channel_active=leaf_backend is not None,
            )
            plans.append(plan)
            device = torch.device(job.get("device", "cpu"))
            us_model = None
            them_model = None
            if plan.load_us_local:
                if us not in model_cache:
                    model_cache[us] = load_model_from_checkpoint(us, device=device)
                us_model = model_cache[us]
            if plan.load_them_local:
                if them == us and us_model is not None:
                    them_model = us_model
                else:
                    if them not in model_cache:
                        model_cache[them] = load_model_from_checkpoint(
                            them, device=device
                        )
                    them_model = model_cache[them]

            us_leaf = leaf_backend if plan.use_leaf_for_us else None
            them_leaf = leaf_backend if plan.use_leaf_for_them else None
            temp = float(job.get("action_temperature", 1.0))
            sample = bool(job.get("sample_actions", True))
            ctx = int(job.get("model_max_context") or _config.MODEL.max_context)
            timeout_s = int(job.get("game_timeout_s", 600))
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
                deck=opp_deck,
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
            if our_seat == 0:
                agents0[i], agents1[i] = us_agent, them_agent
            else:
                agents0[i], agents1[i] = them_agent, us_agent

        batch = env.reset(specs)
        obs_list = [e.obs for e in batch.envs]

        while not all(finished):
            actions: list[Optional[Action]] = [None] * n
            any_active = False
            for i in range(n):
                if finished[i]:
                    continue
                obs = obs_list[i]
                if obs is None:
                    finished[i] = True
                    continue
                cur = obs.get("current") or {}
                if int(cur.get("result", -1)) != -1:
                    finished[i] = True
                    continue
                if steps[i] >= max_steps:
                    finished[i] = True
                    continue
                seat = int(cur.get("yourIndex", 0))
                agent = agents0[i] if seat == 0 else agents1[i]
                assert agent is not None
                try:
                    privileged_aux = None
                    target_count_before = len(agent.targets)
                    if bool(
                        jobs[i].get("collect_privileged_belief", False)
                    ) and bool(getattr(agent, "collect_targets", False)):
                        hidden = env.hidden_snapshot(i, 1 - seat)
                        privileged_aux = {
                            "opp_hand": hidden["hand"],
                            "opp_deck_order": hidden["deck"],
                            "opp_prizes": hidden["prize"],
                            "privileged_label_source": (
                                "training_fork_exact_same_state"
                            ),
                        }
                    actions[i] = list(agent(obs))
                    if privileged_aux is not None:
                        if len(agent.targets) != target_count_before + 1:
                            raise RuntimeError(
                                "privileged belief collection expected exactly "
                                "one policy target for the acting state"
                            )
                        agent.targets[-1]["aux_labels"] = privileged_aux
                except BaseException as exc:  # noqa: BLE001
                    failed_seat[i] = seat
                    errors[i] = f"{type(exc).__name__}: {exc}"
                    finished[i] = True
                    continue
                any_active = True
            if not any_active:
                break
            batch = env.step_batch(actions)
            for e in batch.envs:
                i = e.env_id
                if i >= n or finished[i]:
                    continue
                if actions[i] is not None:
                    steps[i] += 1
                obs_list[i] = e.obs
                if e.done:
                    finished[i] = True

        wall_s = time.perf_counter() - t0
        results: list[dict[str, Any]] = []
        for i, job in enumerate(jobs):
            our_seat = our_seats[i]
            plan = plans[i]
            us_agent = agents0[i] if our_seat == 0 else agents1[i]
            them_agent = agents1[i] if our_seat == 0 else agents0[i]
            assert us_agent is not None
            assert them_agent is not None
            if failed_seat[i] is not None:
                failed = int(failed_seat[i])
                if failed == our_seat:
                    results.append(
                        _base(
                            job,
                            our_failed=True,
                            winner=1 - our_seat,
                            value=-1.0,
                            steps=steps[i],
                            error=errors[i],
                            wall_s=wall_s,
                            leaf_remote=bool(
                                plan.use_leaf_for_us or plan.use_leaf_for_them
                            ),
                            leaf_self_play_mode=plan.mode,
                            multi_env_batch=n,
                        )
                    )
                else:
                    results.append(
                        _base(
                            job,
                            winner=our_seat,
                            value=1.0,
                            steps=steps[i],
                            error=f"opp-failed: {errors[i]}",
                            wall_s=wall_s,
                            leaf_remote=bool(
                                plan.use_leaf_for_us or plan.use_leaf_for_them
                            ),
                            leaf_self_play_mode=plan.mode,
                            multi_env_batch=n,
                        )
                    )
                continue

            obs = obs_list[i] or {}
            cur = obs.get("current") or {}
            result_code = cur.get("result", -1)
            incomplete = int(result_code) == -1
            if incomplete:
                results.append(
                    _base(
                        job,
                        resource_error=True,
                        game_timeout=True,
                        steps=steps[i],
                        error="incomplete",
                        wall_s=wall_s,
                        leaf_remote=bool(
                            plan.use_leaf_for_us or plan.use_leaf_for_them
                        ),
                        leaf_self_play_mode=plan.mode,
                        multi_env_batch=n,
                    )
                )
                continue

            winner = int(result_code)
            value = 0.0 if winner == 2 else (1.0 if winner == our_seat else -1.0)
            record = None
            records = []
            if job.get("training_eligible", True) and us_agent.targets:
                record = rr._build_selfplay_record(
                    us_agent.targets,
                    our_deck=list(job["our_deck"]),
                    our_seat=our_seat,
                    value=value,
                    opp_id=str(
                        job.get("opp_archetype")
                        or job.get("opponent_id")
                        or "self"
                    ),
                    archetype=str(job.get("archetype") or "core"),
                    seed=int(job["seed"]),
                    target_provenance={
                        **dict(job.get("target_provenance") or {}),
                        "pure_rl": True,
                        "self_play": True,
                        "soft_policy_targets": False,
                        "multi_env": True,
                    },
                )
                if record is not None:
                    records.append(record)
            us = str(job["checkpoint"])
            them = str(job.get("opponent_checkpoint") or us)
            collect_both = bool(
                job.get("training_eligible", True)
                and job.get("collect_both_seats", False)
                and them == us
            )
            if collect_both and them_agent.targets:
                opp_deck = list(job.get("opp_deck") or job["our_deck"])
                opp_value = 0.0 if winner == 2 else -value
                opp_record = rr._build_selfplay_record(
                    them_agent.targets,
                    our_deck=opp_deck,
                    our_seat=1 - our_seat,
                    value=opp_value,
                    opp_id=str(
                        job.get("archetype") or f"self:{Path(us).name}"
                    ),
                    archetype=str(
                        job.get("opp_archetype")
                        or job.get("archetype")
                        or "core"
                    ),
                    seed=int(job["seed"]) ^ 0x51DE,
                    target_provenance={
                        **dict(job.get("target_provenance") or {}),
                        "pure_rl": True,
                        "self_play": True,
                        "same_policy_second_seat": True,
                        "soft_policy_targets": False,
                        "multi_env": True,
                    },
                )
                if opp_record is not None:
                    records.append(opp_record)
            results.append(
                {
                    "job_index": int(job.get("job_index", 0)),
                    "opponent_id": str(job.get("opponent_id") or "self"),
                    "our_seat": our_seat,
                    "winner": winner,
                    "value": value,
                    "steps": steps[i],
                    "wall_s": wall_s,
                    "n_decisions": len(us_agent.targets),
                    "baseline_failed": False,
                    "our_failed": False,
                    "resource_error": False,
                    "cancelled": False,
                    "training_eligible": True,
                    "self_play": True,
                    "multi_env": True,
                    "multi_env_batch": n,
                    "leaf_remote": bool(
                        plan.use_leaf_for_us or plan.use_leaf_for_them
                    ),
                    "leaf_self_play_mode": plan.mode,
                    "record_json": (
                        json.dumps(record, separators=(",", ":")) if record else None
                    ),
                    "record_jsons": [
                        json.dumps(row, separators=(",", ":")) for row in records
                    ],
                    "error": None,
                }
            )
        return results
    except BaseException as exc:  # noqa: BLE001
        return [
            _base(job, our_failed=True, error=f"{type(exc).__name__}: {exc}")
            for job in jobs
        ]
    finally:
        try:
            env.close()
        except Exception:
            pass
