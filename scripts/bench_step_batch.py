#!/usr/bin/env python3
"""Microbench: stock N×Select+GetBattleData vs native StepBatch.

Reports steps/s (and short random-play games/s when enabled) for N=8,32,64.

Requires competition ``cg`` / ``libcg.so`` on PYTHONPATH (or under
``kaggle/input/.../sample_submission``). Builds / loads ``libcg_step_batch.so``
when present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_cg_path() -> Path:
    from poke_bot.paths import COMPETITION_CG_PARENT, cg_runtime_dir

    try:
        parent = cg_runtime_dir()
    except FileNotFoundError:
        parent = COMPETITION_CG_PARENT
    if not (parent / "cg" / "libcg.so").is_file():
        raise SystemExit(
            f"libcg.so missing under {parent}. Run scripts/fetch_cg_runtime_mirror.sh "
            "or scripts/setup_competition_data.sh"
        )
    p = str(parent)
    if p not in sys.path:
        sys.path.insert(0, p)
    os.environ.setdefault("LIBCG_SO", str((parent / "cg" / "libcg.so").resolve()))
    return parent


def _load_deck(parent: Path) -> list[int]:
    for cand in (
        parent / "deck.csv",
        ROOT / "submission" / "deck.csv",
        ROOT / "kaggle" / "cabt-sim" / "deck.csv",
    ):
        if cand.is_file():
            deck = [int(x) for x in cand.read_text().splitlines() if x.strip()]
            if len(deck) == 60:
                return deck
    raise SystemExit("no 60-card deck.csv found")


def _action_from_obs(obs: dict) -> list[int] | None:
    sel = (obs or {}).get("select") or {}
    opts = sel.get("option") or []
    if not opts:
        return None
    max_c = int(sel.get("maxCount") or 1)
    min_c = int(sel.get("minCount") or 0)
    n = max(min_c, min(max_c, len(opts)))
    if n <= 0:
        return None
    return list(range(n))


def _is_done(obs: dict) -> bool:
    cur = (obs or {}).get("current") or {}
    result = cur.get("result")
    return result is not None and int(result) != -1


def _warmup_env(num_envs: int, deck: list[int], *, native: bool):
    from poke_bot.engine_rebuild.interfaces import ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(num_envs, prefer_native_step_batch=native)
    env.reset([ResetSpec(deck, deck, seed=i) for i in range(num_envs)])
    return env


def bench_steps(
    *,
    num_envs: int,
    deck: list[int],
    native: bool,
    steps: int,
    repeats: int,
) -> dict:
    from poke_bot.engine_rebuild.libcg_step_batch import step_batch_native

    env = _warmup_env(num_envs, deck, native=native)
    try:
        # Prime one step so JSON buffers / caches are hot.
        acts = [_action_from_obs(o or {}) for o in env._obs]
        if any(a is None for a in acts):
            return {"error": "no legal actions after reset", "native": native, "n": num_envs}
        env.step_batch(acts)

        times: list[float] = []
        total_steps = 0
        for _ in range(repeats):
            # Re-arm actions each wave from current obs
            t0 = time.perf_counter()
            for _s in range(steps):
                acts = []
                for i in range(num_envs):
                    if env._done[i]:
                        acts.append(None)
                    else:
                        acts.append(_action_from_obs(env._obs[i] or {}))
                if all(a is None for a in acts):
                    break
                if native and env.uses_native_step_batch:
                    # Measure the raw C call path (same as MultiEnv native).
                    env.step_batch(acts)
                else:
                    env._step_batch_python(acts)
                total_steps += sum(1 for a in acts if a is not None)
            times.append(time.perf_counter() - t0)

        elapsed = sum(times)
        return {
            "mode": "native_step_batch" if native else "python_select_get",
            "uses_native": bool(getattr(env, "uses_native_step_batch", False)),
            "n": num_envs,
            "waves": steps,
            "repeats": repeats,
            "env_steps": total_steps,
            "elapsed_s": elapsed,
            "steps_per_s": (total_steps / elapsed) if elapsed > 0 else 0.0,
            "waves_per_s": (steps * repeats / elapsed) if elapsed > 0 else 0.0,
        }
    finally:
        env.close()


def bench_games(
    *,
    num_envs: int,
    deck: list[int],
    native: bool,
    max_steps: int,
    games_target: int,
) -> dict:
    from poke_bot.engine_rebuild.interfaces import ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(num_envs, prefer_native_step_batch=native)
    finished = 0
    steps = 0
    t0 = time.perf_counter()
    try:
        env.reset([ResetSpec(deck, deck, seed=i) for i in range(num_envs)])
        while finished < games_target:
            acts = []
            for i in range(num_envs):
                if env._done[i]:
                    acts.append(None)
                else:
                    acts.append(_action_from_obs(env._obs[i] or {}))
            if all(a is None for a in acts):
                # reset finished slots
                specs = []
                idxs = []
                for i in range(num_envs):
                    if env._done[i] or env._ptrs[i] is None:
                        specs.append(ResetSpec(deck, deck, seed=finished + i))
                        idxs.append(i)
                if not specs:
                    break
                # LibcgMultiEnv.reset only fills prefix; finish+restart manually
                for i in idxs:
                    env._finish(i)
                    cards_deck = deck
                    import ctypes

                    cards = (ctypes.c_int * 120)(*(cards_deck + cards_deck))
                    start = env._lib.BattleStart(cards)
                    ptr = int(start.battlePtr or 0)
                    if not ptr:
                        return {"error": "BattleStart failed mid-bench", "native": native}
                    env._ptrs[i] = ptr
                    obs = env._get_obs(ptr)
                    env._obs[i] = obs
                    env._done[i] = env._is_done(obs)
                continue
            before_done = list(env._done)
            if native:
                env.step_batch(acts)
            else:
                env._step_batch_python(acts)
            steps += sum(1 for a in acts if a is not None)
            for i in range(num_envs):
                if env._done[i] and not before_done[i]:
                    finished += 1
            if steps > max_steps * games_target:
                break
        elapsed = time.perf_counter() - t0
        return {
            "mode": "native_step_batch" if native else "python_select_get",
            "n": num_envs,
            "games_finished": finished,
            "env_steps": steps,
            "elapsed_s": elapsed,
            "games_per_s": (finished / elapsed) if elapsed > 0 else 0.0,
            "steps_per_s": (steps / elapsed) if elapsed > 0 else 0.0,
        }
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="8,32,64", help="comma env counts")
    ap.add_argument("--steps", type=int, default=40, help="waves per repeat")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--games", type=int, default=16, help="0 to skip game bench")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    parent = _ensure_cg_path()
    deck = _load_deck(parent)

    # Build shim if missing
    from poke_bot.engine_rebuild.libcg_step_batch import has_step_batch, load_step_batch_lib

    if not has_step_batch():
        build = ROOT / "poke_bot" / "engine_rebuild" / "native" / "build_step_batch.sh"
        if build.is_file():
            import subprocess

            subprocess.check_call(["bash", str(build)])
            # reset loader cache
            import poke_bot.engine_rebuild.libcg_step_batch as m

            m._STEP_BATCH_TRIED = False
            m._STEP_BATCH_LIB = None

    sb = load_step_batch_lib()
    print("stock_cg:", parent)
    print("step_batch_lib:", "YES" if sb else "NO")
    if sb:
        print("LIBCG_SO:", os.environ.get("LIBCG_SO"))

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    results: dict = {"step_benches": [], "game_benches": []}
    for n in ns:
        py = bench_steps(
            num_envs=n, deck=deck, native=False, steps=args.steps, repeats=args.repeats
        )
        nat = bench_steps(
            num_envs=n, deck=deck, native=True, steps=args.steps, repeats=args.repeats
        )
        speedup = None
        if py.get("steps_per_s") and nat.get("steps_per_s"):
            speedup = nat["steps_per_s"] / py["steps_per_s"]
        row = {"n": n, "python": py, "native": nat, "speedup_steps": speedup}
        results["step_benches"].append(row)
        if speedup is not None:
            print(
                f"N={n}: python={py.get('steps_per_s', 0):.1f} steps/s  "
                f"native={nat.get('steps_per_s', 0):.1f} steps/s  "
                f"speedup={speedup:.2f}x"
            )
        else:
            print(f"N={n}: {json.dumps(row)}")

    if args.games > 0 and sb:
        for n in ns:
            g_py = bench_games(
                num_envs=n,
                deck=deck,
                native=False,
                max_steps=500,
                games_target=args.games,
            )
            g_nat = bench_games(
                num_envs=n,
                deck=deck,
                native=True,
                max_steps=500,
                games_target=args.games,
            )
            results["game_benches"].append({"n": n, "python": g_py, "native": g_nat})
            print(
                f"games N={n}: python={g_py.get('games_per_s', 0):.2f} g/s "
                f"native={g_nat.get('games_per_s', 0):.2f} g/s "
                f"(steps/s {g_py.get('steps_per_s', 0):.1f} vs {g_nat.get('steps_per_s', 0):.1f})"
            )

    text = json.dumps(results, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
        print("wrote", args.json_out)
    else:
        out = ROOT / "outputs" / "bench_step_batch.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
