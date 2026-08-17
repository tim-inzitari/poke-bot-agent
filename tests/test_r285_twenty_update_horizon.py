from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "state" / "alakazam-new-list-direct-policy-r241.json"


def _literal_assignment(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name} in {path}")


def test_r304_supersedes_r285_with_iter1_terminal_submission() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    horizon = contract["twenty_update_horizon_override"]
    cycle = contract["training_cycle"]
    submission = contract["submission"]
    authority = contract["authority"]

    assert horizon["owner_revision"] == 285
    assert horizon["rl_updates_exact"] == 20
    assert horizon["zero_indexed_last_iteration"] == 19
    assert horizon["next_iteration_after_loop"] == 20
    assert horizon["iteration_20_collection_allowed"] is False
    assert horizon["expert_refresh_and_submission_boundaries"] == [5, 10, 15, 20]
    assert horizon["total_submission_count_including_bootstrap"] == 5

    assert horizon["superseded_for_active_r274_by_owner_revision"] == 304
    assert cycle["rl_updates_exact"] == 2
    assert cycle["zero_indexed_iteration_commits"] == [0, 1]
    assert cycle["iteration_2_collection_allowed"] is False
    assert submission["exact_count"] == 2
    assert submission["eligibility_boundaries"] == [
        "after_initial_25_epoch_expert_bootstrap",
        "after_durable_iteration_1_update",
    ]
    assert submission["checkpoint_sources"] == [
        "expert_bootstrap_before_iter_00000.pt",
        "iter_00001.pt",
    ]
    assert "25" not in authority["submission_cardinality"]


def test_r304_submission_workers_add_terminal_boundary_1() -> None:
    expected = (1, 5, 10, 15, 20)
    boundaries = _literal_assignment(
        ROOT / "scripts" / "run_r274_rl_submission_boundaries.py",
        "BOUNDARIES",
    )
    allowed_boundaries = _literal_assignment(
        ROOT / "scripts" / "stage_r274_rl_submission.py",
        "ALLOWED_BOUNDARIES",
    )
    assert boundaries == expected
    assert allowed_boundaries == set(expected)
    assert 25 not in boundaries
    assert 25 not in allowed_boundaries
