#!/usr/bin/env python3
"""Seat-balanced baseline-policy diagnostic with an explicit deck override.

This is intentionally separate from the formal neural heldout gate.  It answers
one narrow question: can an existing public policy demonstrate stronger play
with the exact deck used by a neural specialist?  The result can justify
teacher distillation, but can never promote a neural checkpoint.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck must contain exactly 60 card IDs: {path} has {len(cards)}")
    return cards


def _read_json_deck(path: Path, dotted_key: str) -> list[int]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    for component in dotted_key.split("."):
        value = value[component]
    cards = [int(card_id) for card_id in value]
    if len(cards) != 60:
        raise ValueError(
            f"JSON deck must contain exactly 60 card IDs: {dotted_key} has {len(cards)}"
        )
    return cards


def _game(task: dict[str, Any]) -> dict[str, Any]:
    from poke_bot.agent import install_quiet_stdout, play_game
    from poke_bot.baselines_runtime import BaselineSpec, load_baseline_agent

    install_quiet_stdout(False)
    a_dir = Path(task["a_dir"])
    b_dir = Path(task["b_dir"])
    a_spec = BaselineSpec("diagnostic-a", "diagnostic-a", a_dir.name, "diagnostic", "local", a_dir)
    b_spec = BaselineSpec("diagnostic-b", "diagnostic-b", b_dir.name, "diagnostic", "local", b_dir)
    try:
        a_fn, original_a_deck = load_baseline_agent(a_spec)
        b_fn, b_deck = load_baseline_agent(b_spec)
        a_deck = list(task.get("a_deck") or original_a_deck)

        def a_agent(observation: dict) -> list[int]:
            if observation.get("select") is None:
                return list(a_deck)
            return a_fn(observation)

        def b_agent(observation: dict) -> list[int]:
            if observation.get("select") is None:
                return list(b_deck)
            return b_fn(observation)

        a_seat = int(task["a_seat"])

        def timeout_handler(_signum: int, _frame: object) -> None:
            raise TimeoutError("diagnostic game timeout")

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(task["timeout_s"]))
        try:
            result = (
                play_game(a_agent, b_agent, a_deck, b_deck)
                if a_seat == 0
                else play_game(b_agent, a_agent, b_deck, a_deck)
            )
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        winner = int(result["winner"])
        score = 0.5 if winner == 2 else float(winner == a_seat)
        return {
            "ok": result.get("failed_seat") is None,
            "a_seat": a_seat,
            "score": score,
            "winner": winner,
            "steps": int(result.get("steps") or 0),
            "error": result.get("error"),
        }
    except BaseException as exc:  # isolate a diagnostic worker failure
        return {
            "ok": False,
            "a_seat": int(task["a_seat"]),
            "score": None,
            "winner": None,
            "steps": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    deck_source = parser.add_mutually_exclusive_group()
    deck_source.add_argument("--a-deck", type=Path)
    deck_source.add_argument("--a-deck-json", type=Path)
    parser.add_argument("--a-deck-key", default="decks.alakazam.card_ids")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=180)
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    a_deck = None
    if args.a_deck is not None:
        a_deck = _read_deck(args.a_deck)
    elif args.a_deck_json is not None:
        a_deck = _read_json_deck(args.a_deck_json, args.a_deck_key)

    common = {
        "a_dir": str(args.a_dir.resolve()),
        "b_dir": str(args.b_dir.resolve()),
        "a_deck": a_deck,
        "timeout_s": int(args.timeout_s),
    }
    tasks = [{**common, "a_seat": index % 2} for index in range(args.games)]
    with ProcessPoolExecutor(max_workers=min(args.workers, args.games)) as pool:
        rows = list(pool.map(_game, tasks, chunksize=1))

    valid = [row for row in rows if row["ok"] and row["score"] is not None]
    by_seat = {}
    for seat in (0, 1):
        selected = [row for row in valid if row["a_seat"] == seat]
        by_seat[f"seat{seat}"] = {
            "games": len(selected),
            "win_rate": (
                sum(float(row["score"]) for row in selected) / len(selected)
                if selected
                else None
            ),
        }
    report = {
        "schema": "poke_bot.deck_override_policy_diagnostic/v1",
        "formal_gate": False,
        "policy_a": str(args.a_dir.resolve()),
        "policy_b": str(args.b_dir.resolve()),
        "a_deck_override": bool(a_deck is not None),
        "requested_games": int(args.games),
        "valid_games": len(valid),
        "failures": len(rows) - len(valid),
        "win_rate": (
            sum(float(row["score"]) for row in valid) / len(valid) if valid else None
        ),
        "mean_steps": (
            sum(int(row["steps"]) for row in valid) / len(valid) if valid else None
        ),
        "by_seat": by_seat,
        "errors": [row["error"] for row in rows if not row["ok"]][:10],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if len(valid) == args.games else 2


if __name__ == "__main__":
    raise SystemExit(main())
