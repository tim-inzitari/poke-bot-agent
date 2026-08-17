from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "activate_alakazam_stall_reward_r333.py"


def _load():
    spec = importlib.util.spec_from_file_location("stall_boundary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_commit_requires_exact_iter5_boundary(tmp_path: Path) -> None:
    module = _load()
    commit = tmp_path / "iter_00005.json"
    commit.write_text(
        json.dumps(
            {
                "run_name": "run",
                "last_completed_iteration": 5,
                "next_iteration": 6,
                "learner": {"digest": "sha256:" + "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    payload = module._validate_commit(commit, "run")
    assert payload["next_iteration"] == 6


def test_drop_in_explicitly_arms_only_64_turn_training_gate() -> None:
    text = (
        ROOT
        / "deploy/systemd/pokebot-alakazam-rule-derivative-g5-rl.service.d/342-stall-reward.conf"
    ).read_text(encoding="utf-8")
    assert "PURE_RL_NO_PROGRESS_MAX_TURNS=64" in text
    assert "POKEBOT_USE_RECURSIVE_TURN_PLANNER" not in text
    assert "POKEBOT_SEARCH_MODE" not in text
