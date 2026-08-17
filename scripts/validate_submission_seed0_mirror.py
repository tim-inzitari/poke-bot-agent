#!/usr/bin/env python3
"""Fail closed unless an exact packaged seed-0 mirror reaches a terminal state.

This is intentionally a package test, not a checkpoint test.  Two independent
instances of the submitted ``main.py`` are loaded and the vendored competition
engine drives the submitted deck against itself.  A long period with no
win-relevant public progress is treated as the same class of failure as a wall
timeout: Kaggle's validation game cannot award a useful result to either copy.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Callable


SCHEMA = "poke_bot.submission_exact_package_64_mirror_gate/v1"


class MirrorValidationError(RuntimeError):
    """The exact package mirror is unsafe to upload."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _pokemon_signature(card: Any) -> tuple[Any, ...] | None:
    if not isinstance(card, dict):
        return None
    return (
        card.get("id"),
        card.get("hp"),
        card.get("maxHp"),
        tuple(sorted(str(value) for value in (card.get("energies") or []))),
        tuple(sorted(str(value) for value in (card.get("energyCards") or []))),
    )


def win_progress_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
    """Return only public state that can make a stalled game approach a win.

    Hand, deck, and discard contents are deliberately excluded: the validation
    failure recycled those zones forever while prizes, damage, KOs, and board
    energy never changed.
    """

    current = dict(observation.get("current") or {})
    rows: list[tuple[Any, ...]] = []
    for raw_player in current.get("players") or []:
        player = raw_player if isinstance(raw_player, dict) else {}
        active = tuple(
            value
            for value in (
                _pokemon_signature(card) for card in (player.get("active") or [])
            )
            if value is not None
        )
        bench = tuple(
            value
            for value in (
                _pokemon_signature(card) for card in (player.get("bench") or [])
            )
            if value is not None
        )
        rows.append(
            (
                len(player.get("prize") or []),
                active,
                bench,
                bool(player.get("poisoned")),
                bool(player.get("burned")),
                bool(player.get("asleep")),
                bool(player.get("paralyzed")),
                bool(player.get("confused")),
            )
        )
    return tuple(rows)


def _load_agent(stage: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, stage / "main.py")
    if spec is None or spec.loader is None:
        raise MirrorValidationError("submitted main.py is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_choice(observation: dict[str, Any], choice: Any) -> None:
    selection = observation.get("select")
    if not isinstance(selection, dict):
        raise MirrorValidationError("nonterminal observation has no selection")
    options = selection.get("option") or []
    if not isinstance(choice, list) or not all(
        isinstance(index, int) and 0 <= index < len(options) for index in choice
    ):
        raise MirrorValidationError("package emitted an illegal option index")
    minimum = int(selection.get("minCount", 0) or 0)
    maximum = min(int(selection.get("maxCount", 0) or 0), len(options))
    if not minimum <= len(choice) <= maximum:
        raise MirrorValidationError("package emitted an illegal option count")


def _fail_closed_count(module: Any) -> int:
    policy = getattr(module, "_POLICY", None)
    return int(getattr(policy, "fail_closed_count", 0) or 0)


def run_exact_package_mirror(
    stage: Path,
    *,
    seed: int = 0,
    wall_timeout_s: int = 600,
    max_engine_steps: int = 10_000,
    max_stagnant_turns: int = 768,
    mirror_games: int = 64,
    module_loader: Callable[[Path, str], Any] = _load_agent,
) -> dict[str, Any]:
    stage = stage.expanduser().resolve()
    if not (stage / "main.py").is_file():
        raise MirrorValidationError("submission stage lacks main.py")
    if (
        max_stagnant_turns < 1
        or max_engine_steps < 1
        or wall_timeout_s < 1
        or mirror_games < 1
    ):
        raise MirrorValidationError("mirror limits must be positive")

    random.seed(seed)
    try:
        ctypes.CDLL(None).srand(int(seed))
    except (AttributeError, OSError):
        pass

    previous_cwd = Path.cwd()
    old_path = list(sys.path)
    started = time.monotonic()
    alarm_supported = hasattr(signal, "SIGALRM")

    def _alarm(_signum: int, _frame: Any) -> None:
        raise TimeoutError(
            f"exact package mirror exceeded {wall_timeout_s}s wall timeout"
        )

    try:
        os.chdir(stage)
        sys.path.insert(0, str(stage))
        if alarm_supported:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(wall_timeout_s)
        agents = (
            module_loader(stage, "submission_seed0_mirror_seat0"),
            module_loader(stage, "submission_seed0_mirror_seat1"),
        )
        from cg.game import battle_finish, battle_select, battle_start

        results: list[int] = []
        steps_by_game: list[int] = []
        final_turns: list[int] = []
        action_calls_by_seat = [0, 0]
        action_seconds_by_seat = [0.0, 0.0]
        maximum_action_seconds_by_seat = [0.0, 0.0]
        for mirror_index in range(mirror_games):
            decks = tuple(
                module.agent({"logs": [], "current": None, "select": None})
                for module in agents
            )
            if any(
                not isinstance(deck, list) or len(deck) != 60 for deck in decks
            ):
                raise MirrorValidationError(
                    "package mirror did not return two 60-card decks"
                )
            observation, _ = battle_start(list(decks[0]), list(decks[1]))
            if observation is None:
                raise MirrorValidationError(
                    f"vendored engine failed to start mirror {mirror_index}"
                )
            steps = 0
            last_seen_turn: int | None = None
            progress_turn: int | None = None
            progress_signature: tuple[Any, ...] | None = None
            result = -1
            try:
                while steps < max_engine_steps:
                    current = dict(observation.get("current") or {})
                    result = int(current.get("result", -1))
                    if result != -1:
                        break
                    turn = int(current.get("turn", 0) or 0)
                    if last_seen_turn != turn:
                        signature = win_progress_signature(observation)
                        if progress_signature != signature:
                            progress_signature = signature
                            progress_turn = turn
                        elif (
                            progress_turn is not None
                            and turn - progress_turn >= max_stagnant_turns
                        ):
                            raise MirrorValidationError(
                                f"exact package mirror {mirror_index} made no "
                                "win-relevant progress for "
                                f"{turn - progress_turn} turns"
                            )
                        last_seen_turn = turn
                    seat = int(current.get("yourIndex", -1))
                    if seat not in (0, 1):
                        raise MirrorValidationError(
                            "engine returned an invalid acting seat"
                        )
                    action_started = time.monotonic()
                    choice = agents[seat].agent(observation)
                    action_elapsed = time.monotonic() - action_started
                    action_calls_by_seat[seat] += 1
                    action_seconds_by_seat[seat] += action_elapsed
                    maximum_action_seconds_by_seat[seat] = max(
                        maximum_action_seconds_by_seat[seat], action_elapsed
                    )
                    _validate_choice(observation, choice)
                    observation = battle_select(choice)
                    steps += 1
            finally:
                battle_finish()
            if result == -1:
                raise MirrorValidationError(
                    f"exact package mirror {mirror_index} did not terminate "
                    f"in {steps} steps"
                )
            results.append(result)
            steps_by_game.append(steps)
            final_turns.append(
                int(dict(observation.get("current") or {}).get("turn", -1))
            )
        fail_closed = [_fail_closed_count(module) for module in agents]
        if any(fail_closed):
            raise MirrorValidationError(
                f"exact package mirror used fail-closed actions: {fail_closed}"
            )
        return {
            "schema": SCHEMA,
            "passed": True,
            "requested_framework_seed": int(seed),
            "native_shuffle_seed_controlled": False,
            "native_shuffle_source": "official_libcg_os_entropy",
            "package_instances": 2,
            "same_exact_package_both_seats": True,
            "mirror_games_requested": int(mirror_games),
            "mirror_games_completed": len(results),
            "terminal_result_counts": {
                str(value): results.count(value) for value in sorted(set(results))
            },
            "max_engine_steps_observed": max(steps_by_game),
            "max_final_turn_observed": max(final_turns),
            "wall_seconds": time.monotonic() - started,
            "wall_timeout_seconds": int(wall_timeout_s),
            "max_engine_steps": int(max_engine_steps),
            "max_stagnant_turns": int(max_stagnant_turns),
            "policy_fail_closed_counts": fail_closed,
            "policy_action_timing_by_seat": [
                {
                    "seat": seat,
                    "calls": action_calls_by_seat[seat],
                    "total_seconds": action_seconds_by_seat[seat],
                    "mean_seconds": (
                        action_seconds_by_seat[seat] / action_calls_by_seat[seat]
                        if action_calls_by_seat[seat]
                        else 0.0
                    ),
                    "maximum_seconds": maximum_action_seconds_by_seat[seat],
                }
                for seat in (0, 1)
            ],
        }
    except TimeoutError as exc:
        raise MirrorValidationError(str(exc)) from exc
    finally:
        if alarm_supported:
            signal.alarm(0)
        os.chdir(previous_cwd)
        sys.path[:] = old_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-engine-steps", type=int, default=10_000)
    parser.add_argument("--max-stagnant-turns", type=int, default=768)
    parser.add_argument("--mirror-games", type=int, default=64)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    if package.is_symlink() or not package.is_file():
        raise MirrorValidationError("submission package is not a stable file")
    payload = run_exact_package_mirror(
        args.stage,
        seed=args.seed,
        wall_timeout_s=args.wall_timeout_seconds,
        max_engine_steps=args.max_engine_steps,
        max_stagnant_turns=args.max_stagnant_turns,
        mirror_games=args.mirror_games,
    )
    payload.update(
        {
            "package_path": str(package),
            "package_sha256": _sha256(package),
            "package_size_bytes": package.stat().st_size,
        }
    )
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.evidence, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
