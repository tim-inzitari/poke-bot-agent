from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import apply_marnie_opponent_tiers_r111 as activation
from scripts.launch_active_specialist import _gate_runtime
from scripts.stage_marnie_opponent_tiers_r111 import (
    H10_CHECKPOINT,
    H10_OPPONENT_ID,
    NEW_GATE_ID,
    derive_gate,
    derive_registry,
)


def source_gate() -> dict:
    roster = [
        {"opponent_id": f"public-{index}", "content_digest": f"sha256:p{index}", "tier": "S", "weight": 2.0}
        for index in range(3)
    ]
    roster += [
        {
            "opponent_id": f"specialist-{index}",
            "content_digest": f"sha256:s{index}",
            "frozen_specialist": True,
            "frozen_checkpoint_digest": f"sha256:c{index}",
            "tier": "S+",
            "weight": 2.0,
        }
        for index in range(13)
    ]
    roster.append(
        {
            "opponent_id": H10_OPPONENT_ID,
            "content_digest": "sha256:h10",
            "frozen_specialist": True,
            "frozen_checkpoint_digest": H10_CHECKPOINT,
            "tier": "S+",
            "weight": 2.0,
        }
    )
    return {
        "active_gate_id": "old",
        "owner_decision_revision": 109,
        "active_gate_semantics": {
            "base_premium_agents": 3,
            "frozen_specialist_tier": "S+",
        },
        "derivation": {"schema": "old"},
        "next_gate": {
            "id": "old",
            "label": "old",
            "evaluation": {
                "games_per_opponent": 250,
                "games_total": 4250,
                "seat0_games_per_opponent": 125,
                "seat1_games_per_opponent": 125,
            },
            "pass_criteria": {
                "skill_weighted_win_rate": 0.80,
                "skill_weighted_confidence_lower": 0.50,
            },
            "roster": roster,
        },
    }


def test_tier_derivative_changes_only_requested_classification() -> None:
    source = source_gate()
    original = copy.deepcopy(source)
    derived, counts = derive_gate(source)
    assert source == original
    assert derived["active_gate_id"] == NEW_GATE_ID
    assert counts == {"h10_s": 1, "other_frozen_a": 13, "public_a": 3}
    assert [(row["opponent_id"], row["content_digest"]) for row in derived["next_gate"]["roster"]] == [
        (row["opponent_id"], row["content_digest"]) for row in original["next_gate"]["roster"]
    ]
    assert [(row["tier"], row["weight"]) for row in derived["next_gate"]["roster"]] == [
        *(('A', 1.0) for _ in range(16)),
        ('S', 2.0),
    ]
    assert derived["next_gate"]["evaluation"] == original["next_gate"]["evaluation"]
    assert derived["next_gate"]["pass_criteria"] == original["next_gate"]["pass_criteria"]


def test_registry_points_to_new_gate_without_changing_specialists() -> None:
    source = {
        "schema": "poke_bot.specialist_runtime_registry/v1",
        "version": 4,
        "owner_decision_revision": 109,
        "active_gate_contract": "runtime/old.json",
        "terminal_active_gate_id": "old",
        "common_trainer_args": ["--boundary-design-migration-reason", "old"],
        "specialists": {"marnie-s-grimmsnarl-ex": {"status": "ready"}},
    }
    derived = derive_registry(source, gate_relative_path="runtime/new.json")
    assert derived["version"] == 5
    assert derived["active_gate_contract"] == "runtime/new.json"
    assert derived["terminal_active_gate_id"] == NEW_GATE_ID
    assert derived["common_trainer_args"] == [
        "--boundary-design-migration-reason",
        "receipt_backed_opponent_tiers_r111",
    ]
    assert derived["specialists"] == source["specialists"]


def test_launcher_accepts_exact_r111_tiers_and_frozen_checksums(
    tmp_path: Path,
) -> None:
    gate, _ = derive_gate(source_gate())
    gate_path = tmp_path / "gate.json"
    frozen_path = tmp_path / "frozen.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    frozen_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "specialists": [
                    {
                        "opponent_id": row["opponent_id"],
                        "checkpoint_digest": row["frozen_checkpoint_digest"],
                        "frozen": True,
                        "public_mix_eligible": True,
                    }
                    for row in gate["next_gate"]["roster"]
                    if row.get("frozen_specialist") is True
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _gate_runtime(gate_path, frozen_path) == (NEW_GATE_ID, 4250)


def test_boundary_watcher_accepts_stable_long_running_oneshot(
    monkeypatch,
) -> None:
    """A polling gate handler is healthy while systemd says activating."""

    times = iter((0.0, 0.0, 0.0, 6.0, 6.0))
    monkeypatch.setattr(activation.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(activation.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        activation,
        "systemctl",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="activating\n"),
    )
    monkeypatch.setattr(activation, "main_pid", lambda _unit: 42)

    assert activation.wait_stable("gate.service", old_pid=7, timeout=30.0) == 42
