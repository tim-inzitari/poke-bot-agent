from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import apply_matchup_runtime_at_boundary as boundary


def test_boundary_cli_maps_receipt_flag_to_receipt_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    paths = [tmp_path / name for name in ("run", "merged", "parent", "auth", "tree", "receipt")]
    captured = {}

    def fake_apply(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(boundary, "apply_boundary_activation", fake_apply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_matchup_runtime_at_boundary.py",
            "--run-dir", str(paths[0]),
            "--merged-checkpoint", str(paths[1]),
            "--parent-checkpoint", str(paths[2]),
            "--activation-authorization", str(paths[3]),
            "--runtime-tree", str(paths[4]),
            "--receipt", str(paths[5]),
            "--expected-last-iteration", "26",
            "--validate-only",
        ],
    )
    assert boundary.main() == 0
    assert captured["receipt_path"] == paths[5]
    assert "receipt" not in captured
    assert captured["publish"] is False
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_iteration26_rehearsal_authorization_publishes_only_activated_learner(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    (run / "commits").mkdir(parents=True)
    parent = tmp_path / "iter_00026.pt"
    merged = tmp_path / "iter_00026_matchup.pt"
    runtime_tree = tmp_path / "runtime-tree.json"
    authorization = tmp_path / "iter26-rehearsal-authorization.json"
    receipt = tmp_path / "boundary-receipt.json"
    parent.write_bytes(b"parent-checkpoint")
    merged.write_bytes(b"merged-checkpoint")
    authorization.write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_adapter_rehearsal_authorization/v1",
                "completed_iteration": 26,
                "first_eligible_iteration": 27,
            }
        )
        + "\n"
    )
    parent_digest = boundary._sha256(parent)
    merged_digest = boundary._sha256(merged)
    state = {
        "last_completed_iteration": 26,
        "next_iteration": 27,
        "learner": {"path": str(parent), "digest": parent_digest},
        "champion": {"path": "protected-champion.pt", "digest": "champion"},
        "heldout_champion": {
            "path": "protected-heldout.pt",
            "digest": "heldout",
        },
    }
    serialized = json.dumps(state, sort_keys=True) + "\n"
    (run / "loop_state.json").write_text(serialized)
    (run / "commits" / "iter_00026.json").write_text(serialized)
    runtime_tree.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "checkpoint_digest": merged_digest,
                    "one_route_per_decision": True,
                    "unknown_route_exact_bypass": True,
                }
            },
            sort_keys=True,
        )
        + "\n"
    )

    validated = []

    def validate_training_authorization(path, *, parent_checkpoint):
        validated.append((Path(path).resolve(), Path(parent_checkpoint).resolve()))
        assert json.loads(Path(path).read_text())["schema"].endswith(
            "rehearsal_authorization/v1"
        )

    monkeypatch.setattr(
        boundary,
        "validate_adapter_training_authorization",
        validate_training_authorization,
    )
    monkeypatch.setattr(
        boundary.checkpoint,
        "checkpoint_digest",
        lambda path: boundary._sha256(Path(path)),
    )
    monkeypatch.setattr(
        boundary.checkpoint, "assert_trusted_policy_checkpoint", lambda _path: None
    )
    monkeypatch.setattr(
        boundary.checkpoint,
        "load_checkpoint",
        lambda _path, map_location="cpu": {
            "extra": {
                "dormant_matchup_adapter_bank": {
                    "runtime_enabled": False,
                    "optimizer_imported": False,
                    "activation_parent_digest": parent_digest,
                },
                "dormant_matchup_adapter_fit": {
                    "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
                    "runtime_enabled": False,
                    "base_frozen": True,
                    "route_decisions": {"alakazam": 10, "lucario": 0},
                },
            }
        },
    )

    class FakeTree:
        def __init__(self, _payload, *, digest):
            self.digest = digest
            self.runtime_enabled = True
            self.runtime_accepted_archetype_ids = {"alakazam"}
            self.runtime_consecutive_required = 2

    monkeypatch.setattr(boundary, "PublicMatchupDecisionTree", FakeTree)

    published = boundary.apply_boundary_activation(
        run_dir=run,
        merged_checkpoint=merged,
        parent_checkpoint=parent,
        activation_authorization=authorization,
        runtime_tree=runtime_tree,
        receipt_path=receipt,
        expected_last_iteration=26,
    )
    assert published["schema"] == boundary.SCHEMA
    assert validated == [(authorization.resolve(), parent.resolve())]
    updated = json.loads((run / "loop_state.json").read_text())
    assert updated["learner"] == {"path": str(merged), "digest": merged_digest}
    assert updated["champion"] == state["champion"]
    assert updated["heldout_champion"] == state["heldout_champion"]
    assert updated["dormant_matchup_adapter_fit"]["runtime_enabled"] is True
    assert updated["matchup_runtime_activation"]["boundary_next_iteration"] == 27

    # A retry after receipt publication is identity-idempotent and cannot
    # replace either protected champion pointer.
    retried = boundary.apply_boundary_activation(
        run_dir=run,
        merged_checkpoint=merged,
        parent_checkpoint=parent,
        activation_authorization=authorization,
        runtime_tree=runtime_tree,
        receipt_path=receipt,
        expected_last_iteration=26,
    )
    assert retried == published
    assert json.loads((run / "loop_state.json").read_text())["champion"] == state[
        "champion"
    ]
