"""One-shot r225 stock-libcg shared-tree viability wrapper.

This entrypoint intentionally delegates every real-game decision to the exact
archived r195 NO-RTP entrypoint saved as ``r195_direct_main.py``.  On the first
ordinary real decision only, it runs an isolated capability/throughput probe
*after* obtaining that direct action and before returning it.  A probe success
does not receive action authority; a probe failure also leaves the direct
action unchanged.

The staging tool substitutes this file for ``main.py`` in an isolated
diagnostic tarball.  It must remain free of ``__file__`` because Kaggle imports
submission entrypoints from an isolated archive.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_AGENT_DIR_CANDIDATES = (Path.cwd(), Path("/kaggle_simulations/agent"))
_DIRECT_MODULE: Any | None = None
_DIAGNOSTIC: Any | None = None


def _agent_dir() -> Path:
    for candidate in _AGENT_DIR_CANDIDATES:
        if (candidate / "r195_direct_main.py").is_file():
            return candidate
    return Path.cwd()


def _load_direct_module() -> Any:
    global _DIRECT_MODULE
    if _DIRECT_MODULE is not None:
        return _DIRECT_MODULE
    stage = _agent_dir()
    source = stage / "r195_direct_main.py"
    if not source.is_file():
        raise RuntimeError("r225 diagnostic package lacks r195_direct_main.py")
    # The archived direct entrypoint lazily inserts the agent directory before
    # importing its vendored runtime.  Do not alter that order here.
    spec = importlib.util.spec_from_file_location("r225_r195_direct", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load archived r195 direct entrypoint")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DIRECT_MODULE = module
    return module


def _ordinary_real_decision(obs_dict: dict[str, Any]) -> bool:
    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    current = obs_dict.get("current") if isinstance(obs_dict, dict) else None
    if not isinstance(selection, dict) or not isinstance(current, dict):
        return False
    # Turn-order is a deterministic pre-model short circuit.  The direct
    # entrypoint owns its exact resolver; the diagnostic must not initialize
    # native search just because the competition asks IsFirst.
    context = selection.get("context")
    normalized = "".join(char for char in str(context).lower() if char.isalnum())
    if context == 41 or normalized == "isfirst":
        return False
    return bool(selection.get("option"))


def _maybe_run_diagnostic(obs_dict: dict[str, Any], direct_action: list[int]) -> None:
    global _DIAGNOSTIC
    if _DIAGNOSTIC is None:
        from poke_bot.r225_eight_lane_diagnostic import R225DiagnosticEntrypoint

        _DIAGNOSTIC = R225DiagnosticEntrypoint.from_packaged_files(_agent_dir())
    direct = _load_direct_module()
    _DIAGNOSTIC.maybe_run(
        obs_dict,
        direct_action=direct_action,
        model=getattr(direct, "_MODEL", None),
        policy=getattr(direct, "_POLICY", None),
        deck=getattr(direct, "_DECK", None),
    )


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return the exact r195 direct action, regardless of diagnostic outcome."""

    direct = _load_direct_module()
    action = direct.agent(obs_dict)
    if _ordinary_real_decision(obs_dict):
        try:
            _maybe_run_diagnostic(obs_dict, list(action))
        except Exception as exc:  # pragma: no cover - outcome is log-only safety.
            # Do not give a telemetry probe any real-game action authority.
            print(
                "R225_EIGHT_LANE_DIAGNOSTIC_FAILED "
                + type(exc).__name__
                + ": "
                + str(exc),
                flush=True,
            )
    return list(action)

