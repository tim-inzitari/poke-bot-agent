from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from poke_bot.pure_rl.artifact_retention import apply_artifact_retention
from poke_bot.pure_rl.eval_public import OFFICIAL_BASELINE_IDS
from poke_bot.pure_rl.expert_rehearsal import (
    carry_learner_candidate,
    rehearsal_due,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_train_pure_rl():
    path = ROOT / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_lineage_test", path)
    assert spec is not None and spec.loader is not None
    sys.modules.pop("train_pure_rl_lineage_test", None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rehearsal_cadence_is_before_each_fifth_iteration() -> None:
    assert rehearsal_due(0, 5) is False
    assert rehearsal_due(4, 5) is False
    assert rehearsal_due(5, 5) is True
    assert rehearsal_due(10, 5) is True
    assert rehearsal_due(10, 0) is False


def test_continuous_learner_carries_safe_rejections_and_rolls_back_collapse() -> None:
    assert carry_learner_candidate({"valid": True, "passed": False, "wr": 0.44}, abort=False) == (
        True,
        "continuous_learner",
    )
    assert carry_learner_candidate({"valid": True, "passed": True, "wr": 0.55}, abort=False) == (
        True,
        "promoted",
    )
    assert carry_learner_candidate({"valid": True, "passed": False, "wr": 0.34}, abort=False)[0] is False
    assert carry_learner_candidate({"valid": True, "passed": True, "wr": 0.80}, abort=True)[0] is False


def test_retention_keeps_latest_five_and_all_protected_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for part in ("shards", "checkpoints", "metrics", "artifact_receipts"):
        (run_dir / part).mkdir(parents=True, exist_ok=True)
    for iteration in range(8):
        (run_dir / "shards" / f"iter_{iteration:05d}.jsonl").write_bytes(
            f"shard-{iteration}".encode()
        )
        (run_dir / "checkpoints" / f"iter_{iteration:05d}.pt").write_bytes(
            f"checkpoint-{iteration}".encode()
        )
    expert = run_dir / "checkpoints" / "expert_before_iter_00000.pt"
    expert.write_bytes(b"expert")
    seed = run_dir / "checkpoints" / "seed.pt"
    seed.write_bytes(b"seed")

    def identity(path: Path) -> dict[str, str]:
        return {"path": str(path), "digest": "sha256:test"}

    state = {
        "champion": identity(run_dir / "checkpoints" / "iter_00001.pt"),
        "heldout_champion": identity(run_dir / "checkpoints" / "iter_00002.pt"),
        "learner": identity(run_dir / "checkpoints" / "iter_00004.pt"),
        "lineage_base": identity(seed),
        "opponent_pool": [
            identity(run_dir / "checkpoints" / "iter_00005.pt")
        ],
    }
    report = apply_artifact_retention(
        run_dir,
        state,
        completed_iteration=7,
        replay_window_shards=2,
        history_iterations=5,
    )

    assert report["retired_shards"] == [0, 1, 2]
    assert not (run_dir / "shards" / "iter_00000.jsonl").exists()
    assert (run_dir / "shards" / "iter_00003.jsonl").is_file()
    assert not (run_dir / "checkpoints" / "iter_00000.pt").exists()
    assert (run_dir / "checkpoints" / "iter_00001.pt").is_file()
    assert (run_dir / "checkpoints" / "iter_00002.pt").is_file()
    assert (run_dir / "checkpoints" / "iter_00003.pt").is_file()
    assert report["retired_expert_checkpoints"] == [0]
    assert not expert.exists()
    assert (run_dir / "artifact_receipts" / "shards" / "iter_00000.json").is_file()
    assert (
        run_dir
        / "artifact_receipts"
        / "checkpoints"
        / "expert_before_iter_00000.json"
    ).is_file()


def _heldout_rows(digest: str) -> list[dict]:
    rows: list[dict] = []
    for job_index in range(1000):
        pair = job_index // 2
        rows.append(
            {
                "job_index": job_index,
                "opponent_id": OFFICIAL_BASELINE_IDS[pair % 4],
                "our_seat": job_index % 2,
                "winner": job_index % 2,
                "checkpoint_digest": digest,
                "action_selection": "greedy",
                "invalid": False,
            }
        )
    return rows


def test_heldout_audit_requires_all_1000_exact_digest_greedy_balanced_games() -> None:
    module = _load_train_pure_rl()
    digest = "sha256:" + "a" * 64
    rows = _heldout_rows(digest)
    audit = module._audit_heldout_rows(
        rows,
        n_games=1000,
        checkpoint_digest=digest,
    )
    assert audit["passed"] is True
    assert all(cell["games"] == 250 for cell in audit["per_opponent"].values())
    assert all(cell["seat0"] == 125 for cell in audit["per_opponent"].values())
    assert all(cell["seat1"] == 125 for cell in audit["per_opponent"].values())

    rows[700]["checkpoint_digest"] = "sha256:" + "b" * 64
    assert module._audit_heldout_rows(
        rows,
        n_games=1000,
        checkpoint_digest=digest,
    )["passed"] is False

