#!/usr/bin/env python3
"""Capture accepted official `IsFirst` control-flow transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv


def slice_state(obs: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    return {
        "turn": int(current.get("turn", -1)),
        "your_index": int(current.get("yourIndex", -1)),
        "first_player": int(current.get("firstPlayer", -1)),
        "result": int(current.get("result", -1)),
        "select_type": int(select.get("type", -1)),
        "select_context": int(select.get("context", -1)),
        "select_min": int(select.get("minCount", 0)),
        "select_max": int(select.get("maxCount", 0)),
        "option_types": [int(option.get("type", -1)) for option in (select.get("option") or [])],
        "option_count": len(select.get("option") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--official-lib", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--fixtures", type=int, default=64)
    args = parser.parse_args()
    deck = [int(value) for value in args.deck.read_text().split()]
    if len(deck) != 60:
        raise ValueError("deck must contain exactly 60 cards")
    rows: list[dict[str, Any]] = []
    env = LibcgMultiEnv(1)
    try:
        for i in range(args.fixtures):
            before_obs = env.reset([ResetSpec(deck, deck, seed=i)]).envs[0].obs
            before = slice_state(before_obs)
            if (
                before["select_context"] != 41
                or before["select_type"] != 9
                or before["select_min"] != 1
                or before["select_max"] != 1
                or before["option_types"] != [1, 2]
            ):
                raise RuntimeError(f"unexpected official IsFirst state: {before}")
            action = i % 2
            after_obs = env.step_batch([[action]]).envs[0].obs
            after = slice_state(after_obs)
            rows.append({
                "fixture": i,
                "selected_option_index": action,
                "selected_option_type": before["option_types"][action],
                "before": before,
                "after": after,
            })
    finally:
        env.close()
    report = {
        "schema": "poke_bot.official_first_player_fixtures/v1",
        "status": "complete",
        "scope": "accepted IsFirst selection control-flow transition",
        "official_lib_sha256": hashlib.sha256(args.official_lib.read_bytes()).hexdigest(),
        "fixture_count": len(rows),
        "selected_yes_count": sum(row["selected_option_type"] == 1 for row in rows),
        "selected_no_count": sum(row["selected_option_type"] == 2 for row in rows),
        "fixtures": rows,
        "completed_at": time.time(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.json_out)
    print(json.dumps({key: report[key] for key in (
        "status", "scope", "official_lib_sha256", "fixture_count",
        "selected_yes_count", "selected_no_count",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
