#!/usr/bin/env python3
"""Small source-backed status snapshot for the Elmo matchup-routing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
OUTPUT = ROOT / "output"


def container(name: str) -> dict:
    completed = subprocess.run(
        [
            "docker",
            "inspect",
            name,
            "--format",
            "{{json .State}}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode:
        return {"name": name, "exists": False, "running": False}
    state = json.loads(completed.stdout)
    return {
        "name": name,
        "exists": True,
        "running": state.get("Running") is True,
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
    }


def partial_bytes(pattern: str) -> int:
    total = 0
    for path in OUTPUT.glob(pattern):
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(
                child.stat().st_size for child in path.rglob("*") if child.is_file()
            )
    return total


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    tree_path = OUTPUT / "public-matchup-tree-calibrated-v31/public-matchup-tree.json"
    staged_manifest_path = OUTPUT / "alakazam-adapter-staged-all22-v31/manifest.json"
    fit_result_path = OUTPUT / "alakazam-adapter-fit-v31/final.pt"
    tree = read_json(tree_path)
    staged = read_json(staged_manifest_path)
    route_rows = staged.get("routes") if isinstance(staged.get("routes"), list) else []
    coverage = {
        str(row.get("archetype_id") or ""): row
        for row in route_rows
        if isinstance(row, dict) and row.get("archetype_id")
    }
    ready_routes = sorted(
        str(route)
        for route, row in coverage.items()
        if isinstance(row, dict) and int(row.get("train_sequences") or 0) > 0
    )
    dormant_routes = sorted(set(coverage) - set(ready_routes))
    result = {
        "schema": "poke_bot.matchup_pipeline_dashboard/v1",
        "observed_at": time.time(),
        "host": "Elmo",
        "production_blocking": True,
        "tree": {
            **container("pokebot-public-tree-calibrated-v31"),
            "complete": bool(tree),
            "target_routes": len(tree.get("targets") or []) if tree else 22,
            "precision_floor": (
                (tree.get("runtime_calibration") or {}).get("precision_floor")
                if tree
                else 0.93
            ),
            "artifact": str(tree_path),
            "partial_bytes": partial_bytes("public-matchup-tree-calibrated-v31/*"),
        },
        "staging": {
            **container("pokebot-adapter-stage-all22-v31"),
            "complete": bool(staged),
            "target_routes": 22,
            "ready_routes": ready_routes,
            "ready_route_count": len(ready_routes),
            "dormant_no_example_routes": dormant_routes,
            "manifest": str(staged_manifest_path),
            "partial_bytes": partial_bytes(".alakazam-adapter-staged-all22-v31.partial.*"),
        },
        "adapter_fit": {
            **container("pokebot-adapter-fit-all22-v31"),
            "complete": fit_result_path.is_file(),
            "epochs_target": 25,
            "result": str(fit_result_path),
        },
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
