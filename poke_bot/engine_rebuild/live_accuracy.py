"""Live CABT accuracy checks for multi-env / throughput paths.

Uses the official ``libcg`` binary (no fork). Validates that in-process
multi-handle play stays isolated and that ``LibcgMultiEnv`` returns fresh
native observations — so max-throughput collect cannot silently corrupt the
live rules engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from poke_bot.engine_rebuild.parity import fingerprint_select


@dataclass
class AccuracyCheckResult:
    name: str
    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyReport:
    ok: bool
    checks: list[AccuracyCheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(c) for c in self.checks],
        }


def _load_deck(deck_csv: Path) -> list[int]:
    return [int(x) for x in deck_csv.read_text().splitlines() if x.strip()]


def _action_from_obs(obs: dict) -> Optional[list[int]]:
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
    if result is None:
        return False
    return int(result) != -1


def run_live_accuracy_suite(
    *,
    cg_parent: Path,
    deck_csv: Path,
    num_envs: int = 4,
    max_steps: int = 80,
) -> AccuracyReport:
    """Run accuracy suite. ``cg_parent`` is the dir containing the ``cg`` package."""
    import sys

    parent = str(cg_parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from cg import sim  # type: ignore

    lib = sim.lib
    deck = _load_deck(deck_csv)
    if len(deck) != 60:
        return AccuracyReport(
            ok=False,
            checks=[
                AccuracyCheckResult(
                    "deck", False, f"deck must be 60 cards, got {len(deck)}"
                )
            ],
        )

    n = max(2, int(num_envs))
    checks: list[AccuracyCheckResult] = [
        _check_isolation(lib, deck, num_envs=n),
        _check_wrapper_obs_fresh(lib, deck, num_envs=n, max_steps=max_steps),
        _check_valid_playthrough(lib, deck, num_envs=n, max_steps=400),
    ]
    return AccuracyReport(ok=all(c.ok for c in checks), checks=checks)


def _check_isolation(lib: Any, deck: list[int], *, num_envs: int) -> AccuracyCheckResult:
    """Stepping env 0 must not change other envs' select fingerprints."""
    from poke_bot.engine_rebuild.interfaces import ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(num_envs, lib=lib)
    try:
        batch = env.reset([ResetSpec(deck, deck, seed=i) for i in range(num_envs)])
        before = [fingerprint_select(e.obs) for e in batch.envs]
        actions: list[Optional[list[int]]] = [None] * num_envs
        a0 = _action_from_obs(batch.envs[0].obs)
        if a0 is None:
            return AccuracyCheckResult("isolation", False, "env0 has no options")
        actions[0] = a0
        after_batch = env.step_batch(actions)
        after = [fingerprint_select(e.obs) for e in after_batch.envs]
        if after[0] == before[0] and not after_batch.envs[0].done:
            return AccuracyCheckResult(
                "isolation",
                False,
                "env0 fingerprint unchanged after step (engine stuck?)",
                {"before0": before[0], "after0": after[0]},
            )
        leaked = [i for i in range(1, num_envs) if after[i] != before[i]]
        ok = not leaked
        return AccuracyCheckResult(
            "isolation",
            ok,
            "ok" if ok else f"envs changed without step: {leaked}",
            {"before": before, "after": after, "leaked": leaked},
        )
    except Exception as exc:
        return AccuracyCheckResult("isolation", False, f"{type(exc).__name__}: {exc}")
    finally:
        env.close()


def _check_wrapper_obs_fresh(
    lib: Any,
    deck: list[int],
    *,
    num_envs: int,
    max_steps: int,
) -> AccuracyCheckResult:
    """After each ``step_batch``, re-read ``GetBattleData`` on the same ptrs."""
    from poke_bot.engine_rebuild.interfaces import ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(num_envs, lib=lib)
    try:
        batch = env.reset([ResetSpec(deck, deck, seed=i) for i in range(num_envs)])
        mismatches = 0
        steps = 0
        for _ in range(max_steps):
            if all(e.done for e in batch.envs):
                break
            actions: list[Optional[list[int]]] = []
            for e in batch.envs:
                if e.done:
                    actions.append(None)
                else:
                    actions.append(_action_from_obs(e.obs))
            if all(a is None for a in actions):
                break
            batch = env.step_batch(actions)
            steps += 1
            for i, e in enumerate(batch.envs):
                ptr = env._ptrs[i]
                if ptr is None:
                    continue
                sd = lib.GetBattleData(ptr)
                fresh = json.loads(sd.json.decode())
                if fingerprint_select(fresh) != fingerprint_select(e.obs):
                    mismatches += 1
                if _is_done(fresh) != e.done:
                    mismatches += 1
        ok = mismatches == 0 and steps > 0
        return AccuracyCheckResult(
            "wrapper_obs_fresh",
            ok,
            "ok" if ok else f"stale/mismatched obs x{mismatches}",
            {"steps": steps, "mismatches": mismatches},
        )
    except Exception as exc:
        return AccuracyCheckResult(
            "wrapper_obs_fresh", False, f"{type(exc).__name__}: {exc}"
        )
    finally:
        env.close()


def _check_valid_playthrough(
    lib: Any,
    deck: list[int],
    *,
    num_envs: int,
    max_steps: int,
) -> AccuracyCheckResult:
    """Greedy-first multi-env games must terminate without Select errors."""
    from poke_bot.engine_rebuild.interfaces import ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(num_envs, lib=lib)
    try:
        batch = env.reset([ResetSpec(deck, deck, seed=i) for i in range(num_envs)])
        select_errors = 0
        illegal = 0
        for _ in range(max_steps):
            if all(e.done for e in batch.envs):
                break
            actions: list[Optional[list[int]]] = []
            for e in batch.envs:
                if e.done:
                    actions.append(None)
                    continue
                act = _action_from_obs(e.obs)
                opts = (e.obs.get("select") or {}).get("option") or []
                if act is None:
                    illegal += 1
                    actions.append(None)
                    continue
                if any(i < 0 or i >= len(opts) for i in act):
                    illegal += 1
                actions.append(act)
            try:
                batch = env.step_batch(actions)
            except RuntimeError as exc:
                if "Select failed" in str(exc):
                    select_errors += 1
                    break
                raise
        finished = sum(1 for e in batch.envs if e.done)
        ok = select_errors == 0 and illegal == 0 and finished == num_envs
        return AccuracyCheckResult(
            "valid_playthrough",
            ok,
            "ok"
            if ok
            else (
                f"select_errors={select_errors} illegal={illegal} "
                f"finished={finished}/{num_envs}"
            ),
            {
                "select_errors": select_errors,
                "illegal": illegal,
                "finished": finished,
                "num_envs": num_envs,
            },
        )
    except Exception as exc:
        return AccuracyCheckResult(
            "valid_playthrough", False, f"{type(exc).__name__}: {exc}"
        )
    finally:
        env.close()
