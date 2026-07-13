"""Load baseline agents from ``baselines/manifest.json`` for local eval/RL."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import deck_pool, paths

AgentFn = Callable[[dict], list[int]]


@dataclass(frozen=True)
class BaselineSpec:
    id: str
    name: str
    dir_name: str
    group: str
    source: str
    path: Path

    @property
    def main_py(self) -> Path:
        return self.path / "main.py"

    @property
    def deck_csv(self) -> Path:
        return self.path / "deck.csv"


def load_manifest(manifest: Optional[Path] = None) -> list[BaselineSpec]:
    manifest = Path(manifest) if manifest else paths.BASELINES_MANIFEST
    data = json.loads(manifest.read_text(encoding="utf-8"))
    specs: list[BaselineSpec] = []
    for a in data.get("agents", []):
        group = a.get("group", "community")
        parent = paths.BASELINES_DIR / group
        # Manifest uses group dirs: official / community / roster
        path = parent / a["dir"]
        specs.append(
            BaselineSpec(
                id=a["id"],
                name=a.get("name", a["id"]),
                dir_name=a["dir"],
                group=group,
                source=a.get("source", ""),
                path=path,
            )
        )
    return specs


def ensure_baselines_installed(specs: Optional[list[BaselineSpec]] = None) -> list[BaselineSpec]:
    specs = specs if specs is not None else load_manifest()
    missing = [s for s in specs if not (s.main_py.is_file() and s.deck_csv.is_file())]
    if missing:
        ids = ", ".join(s.id for s in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"{len(missing)} baselines missing main.py/deck.csv (e.g. {ids}{more}). "
            "Run: bash scripts/download_baselines.sh"
        )
    return specs


def _load_module(path: Path, module_name: str) -> types.ModuleType:
    """Import ``main.py`` under a unique module name (cwd-independent deck load)."""
    # Many baselines open relative ``deck.csv`` at import — chdir into agent dir.
    import os

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    prev = os.getcwd()
    try:
        os.chdir(str(path.parent))
        # Ensure competition cg is importable as ``cg``.
        from . import cg_env

        cg_env.ensure_cg_importable()
        spec.loader.exec_module(mod)
    finally:
        os.chdir(prev)
    return mod


def load_baseline_agent(spec: BaselineSpec) -> tuple[AgentFn, list[int]]:
    """Return ``(agent_fn, deck)`` for a baseline spec."""
    path = Path(spec.path)
    deck = deck_pool.read_deck(path / "deck.csv")
    mod = _load_module(path / "main.py", f"poke_bot_baseline_{spec.id.replace('-', '_')}")
    if not hasattr(mod, "agent"):
        raise AttributeError(f"{path / 'main.py'} has no agent()")
    agent_fn: AgentFn = getattr(mod, "agent")
    return agent_fn, deck


def load_all_baselines(
    *,
    manifest: Optional[Path] = None,
) -> dict[str, tuple[AgentFn, list[int], BaselineSpec]]:
    specs = ensure_baselines_installed(load_manifest(manifest))
    out: dict[str, tuple[AgentFn, list[int], BaselineSpec]] = {}
    for spec in specs:
        fn, deck = load_baseline_agent(spec)
        out[spec.id] = (fn, deck, spec)
    return out
