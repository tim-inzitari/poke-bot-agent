#!/usr/bin/env python3
"""Benchmark PokeRLM planner attachment latency (host / cg-gated).

Cloud pods without the ``cg`` runtime cannot exercise real CABT turns.
This script:

1. prints the PokeRLM parameter inventory (always);
2. micro-benchmarks decoder / dynamics / root / recursive refine on synthetic
   tensors (always);
3. optionally runs whole-turn agent timing when ``--obs-json`` or a live ``cg``
   environment is available.

Examples
--------
python scripts/bench_poke_rlm_turn_latency.py
python scripts/bench_poke_rlm_turn_latency.py --profile pure_rl_96 --repeats 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.poke_rlm import (  # noqa: E402
    PokeRLMConfig,
    PokeRLMController,
    PokeRLMMode,
    PokeRLMModelCore,
    config_for_profile,
)
from poke_bot.poke_rlm.budget import TurnComputeBudget  # noqa: E402
from poke_bot.poke_rlm.legal_action import legal_actions_from_select  # noqa: E402
from poke_bot.poke_rlm.recursion import refine_plan  # noqa: E402


def _synthetic_obs(n_opts: int = 8) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "players": [
                {"hand": [1, 2, 3], "prizeCards": 6},
                {"hand": None, "prizeCards": 6},
            ],
        },
        "select": {
            "type": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14, "cardId": 1000 + i} for i in range(n_opts)],
        },
    }


def _timed(fn, repeats: int) -> list[float]:
    # warmup
    fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _summary(name: str, samples: list[float]) -> dict:
    ordered = sorted(samples)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[max(0, int(0.95 * len(ordered)) - 1)]
    return {
        "name": name,
        "n": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": p50,
        "p95_ms": p95,
        "max_ms": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="pure_rl_96")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--n-options", type=int, default=8)
    parser.add_argument("--obs-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = config_for_profile(
        args.profile,
        enabled=True,
        mode=PokeRLMMode.SHADOW,
    )
    core = PokeRLMModelCore(cfg)
    core.eval()
    inventory = core.parameter_inventory()
    print(json.dumps({"parameter_inventory": inventory, "config": cfg.inventory()}, indent=2))

    n = int(args.n_options)
    d = int(cfg.d_model)
    state = torch.randn(1, d)
    opts = torch.randn(1, n, d)
    mask = torch.ones(1, n, dtype=torch.bool)

    def decode():
        with torch.no_grad():
            return core.score_actions(state, opts, legal_mask=mask)

    def dynamics():
        with torch.no_grad():
            return core.dynamics(state, opts[0])

    results = [
        _summary("parallel_action_decoder", _timed(decode, args.repeats)),
        _summary("latent_dynamics", _timed(dynamics, args.repeats)),
    ]

    obs = (
        json.loads(args.obs_json.read_text(encoding="utf-8"))
        if args.obs_json is not None
        else _synthetic_obs(n)
    )
    legal = legal_actions_from_select(obs)
    if legal:
        heads = decode()

        def root_plans():
            with torch.no_grad():
                return core.propose_root_plans(state, legal, heads)

        results.append(_summary("root_plan_propose", _timed(root_plans, args.repeats)))
        plans = root_plans()
        if plans:

            def recurse():
                budget = TurnComputeBudget(
                    max_depth=cfg.max_depth,
                    max_model_calls=cfg.max_neural_planner_calls_per_turn,
                    max_nodes=cfg.max_plan_nodes,
                    max_subgoals=cfg.max_subgoals,
                    max_simulator_calls=cfg.max_simulator_calls_per_turn,
                )
                with torch.no_grad():
                    return refine_plan(
                        core,
                        plans[0],
                        state_vec=state,
                        option_hidden=opts,
                        legal=legal,
                        budget=budget,
                    )

            results.append(_summary("recursive_refine", _timed(recurse, args.repeats)))

        controller = PokeRLMController(cfg, core=core)

        def controller_shadow():
            with torch.no_grad():
                return controller.plan_decision(
                    obs,
                    state_vec=state[0],
                    option_hidden=opts[0],
                    legal_combos=[a.option_index_path for a in legal],
                    fallback_action=(0,),
                )

        # Reset budget between repeats inside timed fn by reconstructing lightly.
        def controller_shadow_fresh():
            c = PokeRLMController(cfg, core=core)
            with torch.no_grad():
                return c.plan_decision(
                    obs,
                    state_vec=state[0],
                    option_hidden=opts[0],
                    legal_combos=[a.option_index_path for a in legal],
                    fallback_action=(0,),
                )

        results.append(
            _summary("controller_shadow", _timed(controller_shadow_fresh, args.repeats))
        )
        _ = controller  # silence lint

    cg_status = "BLOCKED"
    try:
        from poke_bot import paths  # noqa: F401

        paths.cg_runtime_dir()
        cg_status = "available"
    except Exception as exc:
        cg_status = f"BLOCKED: {type(exc).__name__}: {exc}"

    payload = {
        "profile": args.profile,
        "cg_runtime": cg_status,
        "results_ms": results,
        "notes": [
            "Whole-turn CABT latency requires host cg runtime and real obs traces.",
            "Synthetic microbenchmarks do not replace gate evidence in docs/poke_rlm/07.",
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
