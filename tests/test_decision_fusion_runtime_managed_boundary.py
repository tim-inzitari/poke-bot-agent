from __future__ import annotations

from pathlib import Path

import pytest

from scripts.apply_decision_fusion_runtime_managed_boundary import (
    SCHEMA,
    _pass_marker_supersedes_boundary,
    _runtime_enabled_selector,
)


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_boundary_uses_only_declared_managed_lifecycle() -> None:
    script = (
        ROOT / "scripts/apply_decision_fusion_runtime_managed_boundary.py"
    ).read_text(encoding="utf-8")
    unit = (
        ROOT
        / "deploy/systemd/pokebot-decision-fusion-runtime-boundary.service"
    ).read_text(encoding="utf-8")
    assert SCHEMA == "poke_bot.causal_decision_fusion_managed_runtime_boundary/v1"
    for forbidden in ("pkill", "killall", "os.kill(", "SIGKILL", "SIGTERM"):
        assert forbidden not in script
        assert forbidden not in unit
    assert '["systemctl", "--user", "stop", args.unit]' in script
    assert '["systemctl", "--user", "start", args.unit]' in script
    assert "_assert_warmup_learner(trained)" in script
    assert "_assert_tree_matches(" in script
    assert "_assert_elmo_image_tree_matches(" in script
    assert "ThreadPoolExecutor(max_workers=3)" in script
    assert "rolled_back_to_trained_fusion_warmup" in script
    assert "--maintenance-lock" in unit
    assert "--selector /home/pokebot/.config/pokebot/specialist_runtime.env" in unit
    assert "--terminal-boundary" in unit
    assert "if args.terminal_boundary:" in script
    assert "--after-iteration 15" in unit
    managed_start = script.index(
        '_run(["systemctl", "--user", "start", args.unit], timeout=90)'
    )
    lock_releases = [
        index
        for index in range(len(script))
        if script.startswith(
            "args.maintenance_lock.unlink(missing_ok=True)", index
        )
    ]
    assert len(lock_releases) >= 2
    assert script.index("if args.terminal_boundary:") < lock_releases[0]
    assert managed_start < lock_releases[1]
    assert script.index(
        'raise RuntimeError("runtime-fusion trainer failed stability check")'
    ) < lock_releases[1]


def test_runtime_boundary_enables_fusion_in_exact_canonical_selector() -> None:
    root = Path("/runtime/fusion")
    source = (
        "POKEBOT_ACTIVE_SPECIALIST=dudunsparce\n"
        "POKEBOT_SPECIALIST_RUNTIME_ROOT=/runtime/fusion\n"
        "PYTHONPATH=/runtime/fusion\n"
        "POKEBOT_DECISION_FUSION_ENABLED=1\n"
        "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=0\n"
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE="
        "receipt_backed_decision_fusion_warmup_v1\n"
        "UNRELATED=preserved\n"
    )
    updated = _runtime_enabled_selector(source, root)
    assert "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=1" in updated
    assert (
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE="
        "receipt_backed_decision_fusion_runtime_v1"
    ) in updated
    assert "UNRELATED=preserved" in updated
    assert updated.count("POKEBOT_DECISION_FUSION_ENABLED=") == 1
    assert updated.count("POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=") == 1


def test_runtime_boundary_selector_fails_closed_on_wrong_root_or_duplicates() -> None:
    with pytest.raises(RuntimeError, match="exactly once"):
        _runtime_enabled_selector(
            "POKEBOT_SPECIALIST_RUNTIME_ROOT=/runtime/other\n"
            "PYTHONPATH=/runtime/fusion\n",
            Path("/runtime/fusion"),
        )
    with pytest.raises(RuntimeError, match="duplicate fusion key"):
        _runtime_enabled_selector(
            "POKEBOT_SPECIALIST_RUNTIME_ROOT=/runtime/fusion\n"
            "PYTHONPATH=/runtime/fusion\n"
            "POKEBOT_DECISION_FUSION_ENABLED=1\n"
            "POKEBOT_DECISION_FUSION_ENABLED=1\n",
            Path("/runtime/fusion"),
        )


def test_terminal_runtime_boundary_is_not_superseded_by_parent_gate_marker() -> None:
    assert (
        _pass_marker_supersedes_boundary(
            terminal_boundary=True,
            marker_exists=True,
        )
        is False
    )
    assert (
        _pass_marker_supersedes_boundary(
            terminal_boundary=False,
            marker_exists=True,
        )
        is True
    )
    assert (
        _pass_marker_supersedes_boundary(
            terminal_boundary=False,
            marker_exists=False,
        )
        is False
    )
