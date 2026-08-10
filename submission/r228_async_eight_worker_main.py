"""Kaggle entrypoint for the r228 asynchronous eight-worker viability smoke."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

R228_ASYNC_SELECTED_ACTION_AUTHORITY = "receipt.selected_action"
_DIRECT: Any | None = None
_RUNTIME: Any | None = None
_AGENT_DIRS = (Path.cwd(), Path("/kaggle_simulations/agent"))


def _agent_dir() -> Path:
    for candidate in _AGENT_DIRS:
        if (candidate / "r195_direct_main.py").is_file():
            return candidate
    return Path.cwd()


def _direct() -> Any:
    global _DIRECT
    if _DIRECT is not None:
        return _DIRECT
    source = _agent_dir() / "r195_direct_main.py"
    spec = importlib.util.spec_from_file_location("r228_r195_direct", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen r195 direct entrypoint")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DIRECT = module
    return module


def _runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    direct = _direct()
    deck, model, policy = direct._ensure_runtime()
    from poke_bot.r228_kaggle_async_runtime import R228AsyncGameplay

    _RUNTIME = R228AsyncGameplay(
        stage=_agent_dir(), model=model, policy=policy, deck=deck
    )
    return _RUNTIME


def agent(obs_dict: dict) -> list[int]:
    """Use shared-tree MCTS on every branch; preserve direct forced prompts."""

    direct = _direct()
    turn_order = direct._turn_order_choice(obs_dict)
    if turn_order is not None:
        return direct._fail_closed(obs_dict, turn_order)
    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if selection is None:
        if _RUNTIME is not None and int(_RUNTIME.decision_count) > 0:
            print(
                "R228_ASYNC_EIGHT_WORKER_FULL_GAMEPLAY_SUCCESS "
                + json.dumps(
                    {
                        "schema": "poke_bot.r228_async_eight_worker_kaggle_viability/v1",
                        "mcts_branching_decisions": int(_RUNTIME.decision_count),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _RUNTIME.decision_count = 0
        return list(direct.agent(obs_dict))

    direct._ensure_runtime()
    from poke_bot import features

    try:
        legal = features.enumerate_action_combos(obs_dict)
    except features.ActionSpaceTooLarge:
        # Preserve the exact frozen r195 fail-closed behavior when complete
        # ordered enumeration exceeds its packaged safety ceiling.  Search
        # receives no authority and no legal line is silently pruned.
        return list(direct.agent(obs_dict))
    if len(legal) <= 1:
        return list(direct.agent(obs_dict))
    try:
        selected = list(_runtime().select(obs_dict))
    except Exception as exc:
        print(
            "R228_ASYNC_EIGHT_WORKER_HARD_FAILURE "
            + json.dumps(
                {"error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    canonical = [list(map(int, action)) for action in legal]
    if selected not in canonical:
        raise RuntimeError("r228 shared-tree action is outside complete legal order")
    return selected
