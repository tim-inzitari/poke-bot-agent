from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.pure_rl.guide_weight_shadow_pair import (
    OFFICIAL_BASELINE_IDS,
    finalize,
    prepare_manifest,
    run_evaluation,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, specialist: str = "archaludon-ex") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    guide = tmp_path / "guide.yaml"
    guide.write_text("version: arch-guide-v1\n")
    receipts = []
    for iteration in range(1, 6):
        shard = tmp_path / f"iter_{iteration:05d}.jsonl"
        shard.write_text(
            json.dumps(
                {
                    "episode_id": f"episode-{iteration}",
                    "seat": 0,
                    "decisions": [{"action": [0]}],
                }
            )
            + "\n"
        )
        receipt = tmp_path / f"iter_{iteration:05d}.receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "poke_bot.completed_collection/v1",
                    "iteration": iteration,
                    "requested_games": 1,
                    "shard": {
                        "path": str(shard),
                        "sha256": _sha(shard),
                        "size": shard.stat().st_size,
                        "games": 1,
                        "decisions": 1,
                    },
                    "replay_cache": {
                        "covered_bytes": shard.stat().st_size,
                        "records": 1,
                        "sequences": 1,
                        "dropped": 0,
                        "signature": {"max_context": 320},
                    },
                }
            )
        )
        receipts.append(
            {
                "path": str(receipt),
                "sha256": _sha(receipt),
                "bytes": receipt.stat().st_size,
            }
        )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_weight_review_request/v1",
                "status": "ready_for_isolated_shadow_pair",
                "scope": "future_specialist_training_runs_only",
                "prospective_policy_revision": 44,
                "learning_semantics_revision": 46,
                "retroactive_application_allowed": False,
                "specialist_id": specialist,
                "completed_iteration": 5,
                "current_weight": 0.05,
                "consecutive_nonpositive_evaluations": 0,
                "guide_contract": {
                    "path": str(guide),
                    "sha256": _sha(guide),
                    "bytes": guide.stat().st_size,
                },
                "guide_version": "arch-guide-v1",
                "review_window": {
                    "collection_receipts": receipts,
                    "seed_checkpoint": {
                        "path": str(parent),
                        "sha256": _sha(parent),
                        "bytes": parent.stat().st_size,
                    },
                },
            }
        )
    )
    return request


def test_prepare_future_pair_freezes_exact_training_and_eval_schedule(
    tmp_path: Path,
) -> None:
    request = _fixture(tmp_path)
    baseline = tmp_path / "baselines.json"
    baseline.write_text("{}")
    manifest_path = prepare_manifest(
        request_path=request,
        output_dir=tmp_path / "shadow",
        shadow_device="cpu",
        production_device="cuda:1",
        baseline_manifest=baseline,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["specialist_id"] == "archaludon-ex"
    assert manifest["variants"] == [
        {"id": "guide_on", "guide_loss_weight": 0.05},
        {"id": "guide_off", "guide_loss_weight": 0.0},
    ]
    assert len(manifest["evaluation_schedule"]) == 1000
    assert set(
        row["opponent_id"] for row in manifest["evaluation_schedule"]
    ) == set(OFFICIAL_BASELINE_IDS)
    for pair in range(0, 1000, 2):
        first, second = manifest["evaluation_schedule"][pair : pair + 2]
        assert first["opponent_id"] == second["opponent_id"]
        assert (first["candidate_seat"], second["candidate_seat"]) == (0, 1)
    assert manifest["training_eligible"] is False
    assert manifest["serving_allowed"] is False


def test_prepare_refuses_teal_or_production_device(tmp_path: Path) -> None:
    baseline = tmp_path / "baselines.json"
    baseline.write_text("{}")
    with pytest.raises(ValueError, match="authorized future-run"):
        prepare_manifest(
            request_path=_fixture(
                tmp_path / "teal",
                specialist="teal-mask-ogerpon-ex",
            ),
            output_dir=tmp_path / "shadow",
            shadow_device="cpu",
            production_device="cuda:1",
            baseline_manifest=baseline,
        )


def test_finalize_compiles_realized_win_schedule(tmp_path: Path) -> None:
    request = _fixture(tmp_path)
    baseline = tmp_path / "baselines.json"
    baseline.write_text("{}")
    manifest = prepare_manifest(
        request_path=request,
        output_dir=tmp_path / "shadow",
        shadow_device="cpu",
        production_device="cuda:1",
        baseline_manifest=baseline,
    )
    on = tmp_path / "on.pt"
    off = tmp_path / "off.pt"
    on.write_bytes(b"on")
    off.write_bytes(b"off")
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "schema": "poke_bot.future_guide_weight_shadow_pair/v1",
                "status": "training_complete",
                "same_parent_replay_split_batch_order_and_optimizer": True,
                "variants": [
                    {
                        "variant": "guide_on",
                        "checkpoint": {"path": str(on), "sha256": _sha(on)},
                    },
                    {
                        "variant": "guide_off",
                        "checkpoint": {"path": str(off), "sha256": _sha(off)},
                    },
                ],
            }
        )
    )
    rows = []
    schedule = json.loads(manifest.read_text())["evaluation_schedule"]
    for row in schedule:
        for variant, checkpoint, score in (
            ("guide_on", on, 1.0),
            ("guide_off", off, 0.0),
        ):
            rows.append(
                {
                    "variant": variant,
                    **row,
                    "checkpoint_sha256": _sha(checkpoint),
                    "score": score,
                    "training_eligible": False,
                    "replay_eligible": False,
                    "formal_gate": False,
                    "invalid": False,
                    "error": None,
                }
            )
    evaluation = tmp_path / "rows.json"
    evaluation.write_text(json.dumps(rows))
    evidence_path, schedule_path = finalize(manifest, training, evaluation)
    evidence = json.loads(evidence_path.read_text())
    compiled = json.loads(schedule_path.read_text())
    assert len(evidence["rows"]) == 2000
    assert compiled["next_state"]["weight"] == 0.15
    assert compiled["status"] == "ready_for_clean_boundary"


def test_evaluation_runs_exact_same_jobs_for_both_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baselines.json"
    baseline.write_text("{}")
    on = tmp_path / "on.pt"
    off = tmp_path / "off.pt"
    on.write_bytes(b"on")
    off.write_bytes(b"off")
    schedule = [
        {
            "schedule_id": f"pair-{index:04d}",
            "opponent_id": OFFICIAL_BASELINE_IDS[(index // 2) % 4],
            "candidate_seat": index % 2,
            "requested_seed": 1_000_000 + index,
        }
        for index in range(1000)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "poke_bot.future_guide_weight_shadow_pair/v1",
                "specialist_id": "archaludon-ex",
                "baseline_manifest": {
                    "path": str(baseline),
                    "sha256": _sha(baseline),
                    "bytes": baseline.stat().st_size,
                },
                "evaluation_schedule": schedule,
            }
        )
    )
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "schema": "poke_bot.future_guide_weight_shadow_pair/v1",
                "status": "training_complete",
                "same_parent_replay_split_batch_order_and_optimizer": True,
                "variants": [
                    {
                        "variant": "guide_on",
                        "checkpoint": {"path": str(on), "sha256": _sha(on)},
                    },
                    {
                        "variant": "guide_off",
                        "checkpoint": {"path": str(off), "sha256": _sha(off)},
                    },
                ],
            }
        )
    )

    from poke_bot import baselines_runtime
    from poke_bot import worker_pool
    from scripts import train_pure_rl, train_round_robin

    specs = [
        SimpleNamespace(id=opponent_id) for opponent_id in OFFICIAL_BASELINE_IDS
    ]
    monkeypatch.setattr(baselines_runtime, "ensure_baselines_installed", lambda: None)
    monkeypatch.setattr(baselines_runtime, "load_manifest", lambda: {})
    monkeypatch.setattr(
        baselines_runtime,
        "filter_loadable_baselines",
        lambda _manifest: (specs, []),
    )
    monkeypatch.setattr(
        train_pure_rl,
        "_our_decks",
        lambda _mode, specialist: [(specialist, [1] * 60)],
    )

    def _jobs(**kwargs):
        checkpoint = str(kwargs["ckpt"])
        digest = str(kwargs["digest"])
        jobs = []
        for index, expected in enumerate(schedule):
            jobs.append(
                {
                    "job_index": index,
                    "checkpoint": checkpoint,
                    "checkpoint_digest": digest,
                    "our_seat": expected["candidate_seat"],
                    "seed": expected["requested_seed"],
                    "opponent_id": expected["opponent_id"],
                }
            )
        return [], jobs

    monkeypatch.setattr(train_pure_rl, "_build_collect_jobs", _jobs)
    monkeypatch.setattr(
        train_round_robin,
        "_worker_play",
        lambda job: {
            **job,
            "winner": job["our_seat"],
            "error": None,
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
        },
    )

    class _Pool:
        def __init__(self, num_workers):
            self.num_workers = num_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, fn, jobs):
            return [fn(job) for job in reversed(jobs)]

    monkeypatch.setattr(worker_pool, "WorkerPool", _Pool)
    rows_path = run_evaluation(manifest, training, workers=4)
    rows = json.loads(rows_path.read_text())
    assert len(rows) == 2000
    assert {row["variant"] for row in rows} == {"guide_on", "guide_off"}
    for variant in ("guide_on", "guide_off"):
        selected = [row for row in rows if row["variant"] == variant]
        assert [
            (
                row["schedule_id"],
                row["opponent_id"],
                row["candidate_seat"],
                row["requested_seed"],
            )
            for row in selected
        ] == [
            (
                row["schedule_id"],
                row["opponent_id"],
                row["candidate_seat"],
                row["requested_seed"],
            )
            for row in schedule
        ]
