"""Regression: sim workers must not install SIGALRM (thread-unsafe)."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _alarm_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Attribute) and func.attr in {"signal", "alarm"}:
            if isinstance(func.value, ast.Name) and func.value.id == "signal":
                name = f"signal.{func.attr}"
        if name == "signal.alarm" or (
            name == "signal.signal"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "SIGALRM"
        ):
            hits.append(f"{path.name}:{node.lineno}:{name}")
    return hits


def test_round_robin_workers_do_not_install_sigalrm():
    path = ROOT / "scripts" / "train_round_robin.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    worker_fns = {
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_worker_play", "_worker_promotion"}
    }
    hits: list[str] = []
    for fn in worker_fns:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "signal"
                and func.attr in {"signal", "alarm"}
            ):
                hits.append(f"{fn.name}:{node.lineno}:signal.{func.attr}")
    assert hits == [], hits


def test_core_bc_worker_does_not_install_sigalrm():
    path = ROOT / "poke_bot" / "core_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_worker_core_bc_game":
            hits = []
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "signal"
                    and func.attr in {"signal", "alarm"}
                ):
                    hits.append(f"{child.lineno}:signal.{func.attr}")
            assert hits == [], hits
            return
    raise AssertionError("_worker_core_bc_game not found")
