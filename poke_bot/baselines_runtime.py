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


def filter_loadable_baselines(
    specs: list[BaselineSpec], *, verbose: bool = True
) -> tuple[list[BaselineSpec], list[tuple[str, str]]]:
    """Return ``(loadable, failed)`` by test-importing each baseline once.

    Some roster agents hardcode Kaggle-only paths (e.g.
    ``/kaggle_simulations/agent/<deck>.csv``) that don't exist locally and
    raise at import. Pre-filtering in the parent process keeps one broken agent
    from crashing the whole worker pool mid-run.
    """
    ok: list[BaselineSpec] = []
    failed: list[tuple[str, str]] = []
    for s in specs:
        try:
            load_baseline_agent(s)
            ok.append(s)
        except Exception as exc:  # noqa: BLE001
            failed.append((s.id, f"{type(exc).__name__}: {exc}"))
    if verbose and failed:
        print(
            f"[baselines] dropped {len(failed)}/{len(specs)} unloadable agents "
            f"(kept {len(ok)}):",
            flush=True,
        )
        for sid, err in failed:
            print(f"    - {sid}: {err[:140]}", flush=True)
    return ok, failed


def _safe_rmtree_under(base: Path, target: Path) -> bool:
    """Delete ``target`` iff it is strictly inside ``base``. No-op if missing.

    Guards against ever removing anything outside the baselines library: the
    resolved target must be a descendant of the resolved base (and not base
    itself). Returns True only if something was actually deleted.
    """
    import shutil

    base_r = base.resolve()
    target_r = target.resolve()
    if target_r == base_r or base_r not in target_r.parents:
        raise ValueError(f"refusing to delete {target_r}: not under {base_r}")
    if not target_r.exists():
        return False
    shutil.rmtree(target_r)
    return True


def delete_baseline_payload(dir_name: str) -> list[str]:
    """Remove an installed baseline's payload dirs from the library on disk.

    Baselines can live under multiple subdirs (``official/community/roster`` for
    the agent, plus ``decks`` / ``kernels`` copies). Deletes every matching
    ``<baselines>/<sub>/<dir_name>`` that exists. Idempotent: already-gone dirs
    are skipped. Returns the repo-relative paths that were actually removed.
    """
    removed: list[str] = []
    for sub in ("official", "community", "roster", "decks", "kernels"):
        target = paths.BASELINES_DIR / sub / dir_name
        try:
            if _safe_rmtree_under(paths.BASELINES_DIR, target):
                removed.append(str(target.relative_to(paths.BASELINES_DIR.parent)))
        except ValueError:
            # Path escaped the baselines dir — never delete; skip defensively.
            continue
    return removed


def remove_from_manifest(agent_id: str, manifest: Optional[Path] = None) -> bool:
    """Drop ``agent_id`` from the tracked manifest so it won't re-download.

    Also appends the id to a persistent ``excluded_broken`` list the download
    script honors (belt-and-suspenders if an entry is ever re-added). Idempotent:
    returns False (no-op) if the agent is neither present nor already excluded.
    Writes atomically and keeps the JSON valid/clean.
    """
    manifest = Path(manifest) if manifest else paths.BASELINES_MANIFEST
    data = json.loads(manifest.read_text(encoding="utf-8"))
    agents = data.get("agents", [])
    kept = [a for a in agents if a.get("id") != agent_id]
    excl = data.get("excluded_broken", [])
    already = agent_id in excl
    changed = (len(kept) != len(agents)) or not already
    if not changed:
        return False
    data["agents"] = kept
    if not already:
        excl.append(agent_id)
    data["excluded_broken"] = sorted(set(excl))
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest)
    return True


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
