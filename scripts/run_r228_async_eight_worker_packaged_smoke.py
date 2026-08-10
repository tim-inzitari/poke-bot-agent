#!/usr/bin/env python3
"""Run one real packaged branching decision through the r228 async queue."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

PREFIX = "R228_ASYNC_EIGHT_WORKER_DECISION "
FINAL_PREFIX = "R228_ASYNC_EIGHT_WORKER_PACKAGED_SMOKE "
SCHEMA = "poke_bot.r228_async_eight_worker_packaged_smoke/v1"


class SmokeError(RuntimeError):
    pass


class _Tee:
    def __init__(self, *targets: TextIO) -> None:
        self.targets = targets
        self.parts: list[str] = []

    def write(self, value: str) -> int:
        self.parts.append(value)
        for target in self.targets:
            target.write(value)
        return len(value)

    def flush(self) -> None:
        for target in self.targets:
            target.flush()

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _deck(stage: Path) -> list[int]:
    cards: list[int] = []
    for raw in (stage / "deck.csv").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            cards.append(int(line.split(",", 1)[0]))
        if len(cards) == 60:
            break
    if len(cards) != 60:
        raise SmokeError("packaged deck is not 60 cards")
    return cards


def _markers(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if line.startswith(PREFIX):
            value = json.loads(line[len(PREFIX) :])
            if not isinstance(value, dict):
                raise SmokeError("decision marker is not an object")
            rows.append(value)
    return rows


def _require(value: bool, message: str) -> None:
    if not value:
        raise SmokeError(message)


def _run(stage: Path, *, max_actions: int) -> dict[str, Any]:
    stage = stage.resolve()
    _require(stage.is_dir() and not stage.is_symlink(), "stage is not physical")
    for relative in (
        "main.py",
        "r195_direct_main.py",
        "model.pt",
        "matchup_tree.json",
        "cg/libcg.so",
    ):
        _require((stage / relative).is_file(), f"stage lacks {relative}")
    os.environ["CG_LIB_PATH"] = str(stage)
    os.chdir(stage)
    sys.path.insert(0, str(stage))
    main = importlib.import_module("main")
    from poke_bot import cg_env, features

    deck = _deck(stage)
    tee = _Tee(sys.stdout)
    observation = None
    accepted = False
    marker: dict[str, Any] | None = None
    legal_at_trigger: list[list[int]] = []
    action_at_trigger: list[int] = []
    calls = 0
    try:
        observation, started = cg_env.battle_start(deck, deck)
        if observation is None:
            raise SmokeError(f"BattleStart failed: {getattr(started, 'errorType', None)}")
        for _ in range(max_actions):
            if cg_env.is_finished(observation):
                raise SmokeError("game ended before a branching decision")
            legal = [list(map(int, row)) for row in features.enumerate_action_combos(observation)]
            before = len(_markers(tee.text))
            with contextlib.redirect_stdout(tee):
                action = list(main.agent(observation))
            calls += 1
            after_rows = _markers(tee.text)
            next_observation = cg_env.battle_select(action)
            if len(after_rows) > before:
                _require(len(after_rows) == before + 1, "one call emitted multiple decision rows")
                marker = after_rows[-1]
                legal_at_trigger = legal
                action_at_trigger = action
                observation = next_observation
                accepted = True
                break
            observation = next_observation
        if marker is None:
            raise SmokeError("no branching r228 decision marker was emitted")
    finally:
        if observation is not None:
            cg_env.battle_finish()

    _require(accepted, "stock BattleSelect did not accept the r228 action")
    _require(marker.get("mode") == "shared_tree_mcts", "decision was not MCTS-authoritative")
    _require(marker.get("mcts_action_authority") is True, "MCTS action authority is off")
    _require(marker.get("selected_action") == action_at_trigger, "played action differs from marker")
    _require(action_at_trigger in legal_at_trigger, "played action is not legal")
    _require(marker.get("arena_count") == 8, "arena count is not eight")
    _require(marker.get("unique_handle_count") == 8, "raw handle count is not eight")
    _require(marker.get("search_begin_calls") == 8, "SearchBegin count is not eight")
    backups = int(marker.get("completed_backups") or 0)
    steps = int(marker.get("search_step_calls") or 0)
    _require(backups >= 1 and steps == backups, "no complete simulator/model backup")
    _require(int(marker.get("selected_action_visits") or 0) >= 1, "selected edge is unbacked")
    _require(marker.get("max_simulator_calls_in_flight") == 8, "eight simulator calls were not in flight")
    batches = marker.get("microbatch_sizes")
    _require(isinstance(batches, list) and sum(map(int, batches)) == backups, "model microbatch receipt is incomplete")
    _require(marker.get("search_release_calls") == 8 + steps, "native SearchId cleanup count changed")
    _require(marker.get("search_end_calls") == 8, "SearchEnd count is not eight")
    _require(marker.get("outstanding_virtual_loss") == 0, "shared tree leaked virtual loss")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "one_packaged_local_branching_decision",
        "agent_calls_before_branch": calls,
        "stage": str(stage),
        "main_sha256": _sha256(stage / "main.py"),
        "runtime_sha256": _sha256(stage / "poke_bot/r228_kaggle_async_runtime.py"),
        "queue_sha256": _sha256(stage / "poke_bot/r228_async_shared_tree_queue.py"),
        "libcg_sha256": _sha256(stage / "cg/libcg.so"),
        "decision": marker,
        "stock_action_accepted": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=128)
    parser.add_argument("--cuda-visible-devices")
    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    receipt = args.receipt.resolve()
    if receipt.exists() or receipt.is_symlink():
        raise SmokeError(f"receipt already exists: {receipt}")
    payload = _run(args.stage, max_actions=max(1, int(args.max_actions)))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(FINAL_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
