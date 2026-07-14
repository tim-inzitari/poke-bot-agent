#!/usr/bin/env python3
"""Play one native compatibility game per installed baseline, fail-fast."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import cg_env, paths
from poke_bot.baselines_runtime import (
    BaselineSpec,
    ensure_baselines_installed,
    load_baseline_agent,
    load_manifest,
)
from poke_bot.worker_pool import WorkerPool


def _simulator_digest() -> str:
    digest = hashlib.sha256()
    libraries = sorted(paths.cg_runtime_dir().rglob("libcg.so"))
    if not libraries:
        return "unavailable"
    for library in libraries:
        digest.update(library.read_bytes())
    return "sha256:" + digest.hexdigest()


def _random_legal(observation: dict, rng: random.Random) -> list[int]:
    select = observation.get("select") or {}
    options = list(select.get("option") or [])
    count = len(options)
    if count == 0:
        return []
    minimum = max(0, int(select.get("minCount", 0) or 0))
    maximum = min(count, int(select.get("maxCount", 0) or 0))
    chosen = rng.randint(minimum, maximum) if maximum >= minimum else maximum
    return rng.sample(range(count), chosen) if chosen > 0 else []


def _play_game(
    agent0, agent1, deck0: list[int], deck1: list[int]
) -> dict[str, Any]:
    observation, _start = cg_env.battle_start(deck0, deck1)
    steps = 0
    failed_seat = None
    error = None
    try:
        while (
            observation is not None
            and not cg_env.is_finished(observation)
            and steps < 4000
        ):
            seat = int((observation.get("current") or {}).get("yourIndex", 0))
            try:
                action = (agent0 if seat == 0 else agent1)(observation)
                observation = cg_env.battle_select(action)
            except BaseException as exc:  # noqa: BLE001
                failed_seat = seat
                error = f"{type(exc).__name__}: {exc}"
                break
            steps += 1
        incomplete = (
            failed_seat is None
            and observation is not None
            and not cg_env.is_finished(observation)
        )
        winner = (
            1 - failed_seat
            if failed_seat is not None
            else 2
            if incomplete
            else cg_env.result_winner(observation)
        )
        return {
            "winner": int(2 if winner is None else winner),
            "steps": steps,
            "failed_seat": failed_seat,
            "error": error,
            "incomplete": incomplete,
            "termination": (
                "agent_failure"
                if failed_seat is not None
                else "max_steps"
                if incomplete
                else "completed"
            ),
        }
    finally:
        try:
            cg_env.battle_finish()
        except Exception:
            pass


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-game-timeout", type=int, default=90)
    parser.add_argument("--budget-seconds", type=float, default=360.0)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def _cache_key(specs: list[BaselineSpec]) -> str:
    digest = hashlib.sha256()
    digest.update(paths.BASELINES_MANIFEST.read_bytes())
    digest.update(_simulator_digest().encode())
    for spec in specs:
        digest.update(spec.id.encode())
        digest.update(spec.main_py.read_bytes())
        digest.update(spec.deck_csv.read_bytes())
    return digest.hexdigest()


def _job(spec: BaselineSpec, index: int, timeout_s: int) -> dict[str, Any]:
    return {
        "index": index,
        "timeout_s": timeout_s,
        "spec": {
            "id": spec.id,
            "name": spec.name,
            "dir_name": spec.dir_name,
            "group": spec.group,
            "source": spec.source,
            "path": str(spec.path),
        },
    }


def _play_compat(job: dict[str, Any]) -> dict[str, Any]:
    row = dict(job["spec"])
    row["path"] = Path(row["path"])
    spec = BaselineSpec(**row)
    timeout_s = int(job["timeout_s"])
    baseline_seat = int(job["index"]) % 2
    previous = None

    def timeout(_signum, _frame) -> None:  # noqa: ANN001
        raise TimeoutError(f"baseline game exceeded {timeout_s}s")

    if hasattr(signal, "SIGALRM"):
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(timeout_s)
    started = time.perf_counter()
    try:
        baseline, deck = load_baseline_agent(spec)
        rng = random.Random(73_000 + int(job["index"]))

        def random_legal(observation: dict) -> list[int]:
            return _random_legal(observation, rng)

        agents = (
            (baseline, random_legal)
            if baseline_seat == 0
            else (random_legal, baseline)
        )
        result = _play_game(agents[0], agents[1], deck, deck)
        ok = (
            result.get("termination") == "completed"
            and result.get("failed_seat") is None
        )
        return {
            "baseline_id": spec.id,
            "ok": ok,
            "baseline_seat": baseline_seat,
            "wall_s": time.perf_counter() - started,
            **result,
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "baseline_id": spec.id,
            "ok": False,
            "baseline_seat": baseline_seat,
            "wall_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if previous is not None:
                signal.signal(signal.SIGALRM, previous)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if args.workers < 1 or args.per_game_timeout < 1:
        raise ValueError("workers and timeout must be positive")
    specs = load_manifest()
    if args.only:
        selected = set(args.only)
        specs = [spec for spec in specs if spec.id in selected]
        missing = selected - {spec.id for spec in specs}
        if missing:
            raise ValueError(
                "unknown baseline id(s): " + ", ".join(sorted(missing))
            )
    specs = ensure_baselines_installed(specs)
    key = _cache_key(specs)
    cache = ROOT / "outputs/test-cache" / f"baseline-compat-{key}.json"
    if not args.no_cache and cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if (
            payload.get("ok")
            and payload.get("baseline_ids") == [spec.id for spec in specs]
            and payload.get("simulator_digest") == _simulator_digest()
        ):
            print(
                f"BASELINE_COMPAT cached=true games={len(specs)} "
                f"key={key[:12]}"
            )
            return 0

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    jobs = [_job(spec, i, args.per_game_timeout) for i, spec in enumerate(specs)]
    with WorkerPool(
        num_workers=min(int(args.workers), len(jobs)),
        recycle_games=4,
    ) as pool:
        for row in pool.imap_unordered(_play_compat, jobs):
            rows.append(row)
            if not row.get("ok"):
                failure = row
                pool.request_stop(
                    f"baseline compatibility failed: {row['baseline_id']}"
                )
                break
            if time.perf_counter() - started > float(args.budget_seconds):
                failure = {
                    "baseline_id": "suite",
                    "error": (
                        f"compatibility budget {args.budget_seconds:.0f}s exceeded"
                    ),
                }
                pool.request_stop("baseline compatibility budget exceeded")
                break

    wall_s = time.perf_counter() - started
    if failure is not None:
        print(
            "BASELINE_COMPAT_FAIL "
            f"baseline={failure.get('baseline_id')} "
            f"cause={failure.get('error') or failure.get('termination')} "
            f"wall_s={wall_s:.2f}",
            file=sys.stderr,
        )
        return 1
    if len(rows) != len(specs):
        print(
            f"BASELINE_COMPAT_FAIL completed={len(rows)}/{len(specs)}",
            file=sys.stderr,
        )
        return 1

    payload = {
        "schema": 1,
        "ok": True,
        "cache_key": key,
        "simulator_digest": _simulator_digest(),
        "baseline_ids": [spec.id for spec in specs],
        "games": sorted(rows, key=lambda row: row["baseline_id"]),
        "wall_s": wall_s,
        "created_at": time.time(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, cache)
    print(
        f"BASELINE_COMPAT cached=false games={len(rows)} "
        f"wall_s={wall_s:.2f} key={key[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
