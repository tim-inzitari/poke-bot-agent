from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts/train_pure_rl.py"
SPEC = importlib.util.spec_from_file_location("train_pure_rl_authoritative", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)

GATE_ROWS = (
    ("pilkwang-meta-20260708", "crustle", 2.0, 0.320),
    ("yaminh-ai-challenge", "lucario", 2.0, 0.366),
    ("aman-crustle-fighting", "lucario", 2.0, 0.412),
    ("penguin-public-scores-915", "lucario", 1.0, 0.504),
    ("archaludon-ex", "archaludon-ex", 1.0, 0.576),
    ("yaroslav-lucario-v2-crustle", "lucario", 1.0, 0.456),
    ("makthanithin-1084-5", "lucario", 1.0, 0.504),
    ("lucifer19-battlecore", "archaludon-ex", 1.0, 0.564),
)

TRAINING_GROUP_PLAN = {
    "self_play": 1024,
    trainer.STRONG_PUBLIC_PRACTICE_GROUP: 4584,
    "diverse_public": 2584,
}
EFFECTIVE_PRACTICE_FRACTION = 4584 / 7168


def _gate() -> dict:
    return {
        "id": "alakazam-strong-public-roster-v1",
        "roster": [
            {
                "opponent_id": opponent_id,
                "archetype_id": archetype_id,
                "weight": weight,
            }
            for opponent_id, archetype_id, weight, _win_rate in GATE_ROWS
        ],
    }


def _build(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trainer, "_spec_payload", lambda spec: {"id": spec.id})
    ids = tuple(row[0] for row in GATE_ROWS)
    weights = trainer._adaptive_official_target_weights(
        ids,
        {row[0]: row[3] for row in GATE_ROWS},
        target_win_rate=0.55,
        minimum_share=0.05,
        gap_power=2.0,
        skill_weights={row[0]: row[2] for row in GATE_ROWS},
    )
    self_jobs, public_jobs = trainer._build_collect_jobs(
        n_games=8192,
        ckpt=Path("/tmp/candidate.pt"),
        digest="sha256:candidate",
        model_generation=9,
        decks=[("alakazam", [1] * 60)],
        specs=[SimpleNamespace(id="diverse-a"), SimpleNamespace(id="diverse-b")],
        seed=800_000,
        game_timeout_s=10,
        mode="specialist",
        collect_temperature=1.0,
        self_play_frac=0.125,
        iteration=8,
        priority_specs=[SimpleNamespace(id=opponent_id) for opponent_id in ids],
        priority_frac=EFFECTIVE_PRACTICE_FRACTION,
        priority_weights=weights,
        priority_group=trainer.STRONG_PUBLIC_PRACTICE_GROUP,
        priority_temperature=0.35,
        priority_archetypes={row[0]: row[1] for row in GATE_ROWS},
        priority_context={
            "active_gate_id": "alakazam-strong-public-roster-v1",
            "formal_eval": False,
            "seed_namespace": "train/strong-public-practice-v1",
            "formal_gate_seed_namespace": "eval/strong-public-fixed-manifest-v1",
        },
    )
    return self_jobs, public_jobs


def test_active_gate_practice_is_exact_weighted_balanced_and_seed_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_jobs, public_jobs = _build(monkeypatch)
    receipt = trainer._assert_strong_public_practice_jobs(
        all_jobs=[*self_jobs, *public_jobs],
        public_jobs=public_jobs,
        active_gate=_gate(),
        expected_practice_games=4584,
        iteration=8,
        root_seed=0,
        formal_games=2000,
        minimum_share=0.05,
        practice_temperature=0.35,
    )
    practice = [
        job
        for job in public_jobs
        if job["target_provenance"]["opponent_training_group"]
        == trainer.STRONG_PUBLIC_PRACTICE_GROUP
    ]
    assert len(self_jobs) == 1024
    assert len(public_jobs) == 7168
    assert len(practice) == 4584
    assert Counter(job["opponent_id"] for job in practice) == {
        "pilkwang-meta-20260708": 1525,
        "yaminh-ai-challenge": 1058,
        "aman-crustle-fighting": 696,
        "yaroslav-lucario-v2-crustle": 337,
        "makthanithin-1084-5": 255,
        "penguin-public-scores-915": 255,
        "archaludon-ex": 229,
        "lucifer19-battlecore": 229,
    }
    assert receipt["seed_disjoint"] is True
    assert all(job["training_eligible"] is True for job in practice)
    assert all(job["sample_actions"] is True for job in practice)
    assert all(job["action_temperature"] == pytest.approx(0.35) for job in practice)
    assert all(
        job["target_provenance"]["formal_eval"] is False for job in practice
    )


def test_8192_schedule_contains_all_three_training_groups_without_extra_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research_digests = {
        opponent_id: "sha256:" + f"{index + 1:x}" * 64
        for index, (opponent_id, _archetype) in enumerate(
            (
                ("iono", "iono-bellibolt"),
                ("dragapult-ex", "dragapult"),
                ("mega-abomasnow-ex", "mega-abomasnow-ex"),
                ("mega-lucario-ex", "lucario"),
            )
        )
    }
    monkeypatch.setattr(
        trainer,
        "_spec_payload",
        lambda spec: {
            "id": spec.id,
            **(
                {"content_digest": research_digests[spec.id]}
                if spec.id in research_digests
                else {}
            ),
        },
    )
    gate_ids = tuple(row[0] for row in GATE_ROWS)
    research_rows = (
        ("iono", "iono-bellibolt"),
        ("dragapult-ex", "dragapult"),
        ("mega-abomasnow-ex", "mega-abomasnow-ex"),
        ("mega-lucario-ex", "lucario"),
    )
    weights = trainer._adaptive_official_target_weights(
        gate_ids,
        {row[0]: row[3] for row in GATE_ROWS},
        target_win_rate=0.55,
        minimum_share=0.05,
        gap_power=2.0,
        skill_weights={row[0]: row[2] for row in GATE_ROWS},
    )
    registry = {
        "registry_id": "alakazam-research-controls",
        "version": 1,
        "controls": [
            {
                "opponent_id": opponent_id,
                "archetype_id": archetype_id,
                "content_digest": research_digests[opponent_id],
            }
            for opponent_id, archetype_id in research_rows
        ],
    }
    assert trainer._planned_collection_group_counts(
        games_per_iteration=8192,
        self_play_fraction=0.125,
        strong_public_fraction_of_public=0.50,
        research_control_games=1000,
    ) == TRAINING_GROUP_PLAN
    self_jobs, public_jobs = trainer._build_collect_jobs(
        n_games=8192,
        ckpt=Path("/tmp/candidate.pt"),
        digest="sha256:candidate",
        model_generation=9,
        decks=[("alakazam", [1] * 60)],
        specs=[SimpleNamespace(id="diverse-a"), SimpleNamespace(id="diverse-b")],
        seed=800_000,
        game_timeout_s=10,
        mode="specialist",
        collect_temperature=1.0,
        self_play_frac=0.125,
        iteration=8,
        priority_specs=[SimpleNamespace(id=opponent_id) for opponent_id in gate_ids],
        priority_frac=EFFECTIVE_PRACTICE_FRACTION,
        priority_weights=weights,
        priority_group=trainer.STRONG_PUBLIC_PRACTICE_GROUP,
        priority_temperature=0.35,
        priority_archetypes={row[0]: row[1] for row in GATE_ROWS},
        priority_context={
            "active_gate_id": "alakazam-strong-public-roster-v1",
            "formal_eval": False,
            "seed_namespace": "train/strong-public-practice-v1",
            "formal_gate_seed_namespace": "eval/strong-public-fixed-manifest-v1",
        },
    )
    groups = Counter(
        job["target_provenance"]["opponent_training_group"]
        for job in public_jobs
    )
    assert len(self_jobs) == 1024
    assert groups == {
        trainer.STRONG_PUBLIC_PRACTICE_GROUP: 4584,
        "diverse_public": 2584,
    }
    all_jobs = [*self_jobs, *public_jobs]
    assert len(all_jobs) == 8192
    assert len({job["job_index"] for job in all_jobs}) == 8192
    assert len({job["seed"] for job in all_jobs}) == 8192

    trainer._assert_training_jobs_exclude_research_controls(all_jobs, registry)
    research = trainer._build_research_control_jobs(
        n_games=1000,
        ckpt=Path("/tmp/candidate.pt"),
        digest="sha256:candidate",
        model_generation=9,
        decks=[("alakazam", [1] * 60)],
        specs=[SimpleNamespace(id=opponent_id) for opponent_id, _ in research_rows],
        seed=trainer.RESEARCH_CONTROL_SEED_OFFSET + 8 * trainer.ITERATION_SEED_STRIDE,
        game_timeout_s=10,
        mode="specialist",
        registry=registry,
        iteration=8,
    )
    assert len(research) == 1000
    receipt = trainer._assert_research_control_jobs(
        research,
        expected_games=1000,
        registry=registry,
        iteration=8,
        root_seed=0,
        training_jobs=all_jobs,
        formal_games=2000,
        checkpoint_digest="sha256:candidate",
        active_gate_digests={"sha256:" + "f" * 64},
    )
    assert receipt["games"] == 1000
    assert {
        opponent_id: row["games"]
        for opponent_id, row in receipt["per_opponent"].items()
    } == {opponent_id: 250 for opponent_id, _ in research_rows}
    for opponent_id, _ in research_rows:
        rows = [job for job in research if job["opponent_id"] == opponent_id]
        assert Counter(job["our_seat"] for job in rows) == {0: 125, 1: 125}
        assert all(job["training_eligible"] is False for job in rows)
        assert all(job["target_provenance"]["replay_eligible"] is False for job in rows)
        assert all(job["sample_actions"] is False and job["greedy"] is True for job in rows)
    assert {job["seed"] for job in research}.isdisjoint(
        {job["seed"] for job in all_jobs}
    )

    trainer._assert_strong_public_practice_jobs(
        all_jobs=all_jobs,
        public_jobs=public_jobs,
        active_gate=_gate(),
        expected_practice_games=4584,
        iteration=8,
        root_seed=0,
        formal_games=2000,
        minimum_share=0.05,
        practice_temperature=0.35,
    )


def test_practice_archetype_labels_cannot_cross_contaminate_matchups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_jobs, public_jobs = _build(monkeypatch)
    practice = [
        job
        for job in public_jobs
        if job["target_provenance"]["opponent_training_group"]
        == trainer.STRONG_PUBLIC_PRACTICE_GROUP
    ]
    expected = {row[0]: row[1] for row in GATE_ROWS}
    assert all(job["opp_archetype"] == expected[job["opponent_id"]] for job in practice)

    practice[0]["opp_archetype"] = "lucario"
    with pytest.raises(RuntimeError, match="practice archetype mismatch"):
        trainer._assert_strong_public_practice_jobs(
            all_jobs=[*self_jobs, *public_jobs],
            public_jobs=public_jobs,
            active_gate=_gate(),
            expected_practice_games=4584,
            iteration=8,
            root_seed=0,
            formal_games=2000,
            minimum_share=0.05,
            practice_temperature=0.35,
        )


def test_research_measurement_recovery_is_idempotent_and_never_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = (
        ("iono", "iono-bellibolt", "1"),
        ("dragapult-ex", "dragapult", "2"),
        ("mega-abomasnow-ex", "mega-abomasnow-ex", "3"),
        ("mega-lucario-ex", "lucario", "4"),
    )
    registry = {
        "registry_id": "alakazam-research-controls",
        "version": 1,
        "controls": [
            {
                "opponent_id": opponent_id,
                "archetype_id": archetype_id,
                "content_digest": "sha256:" + nibble * 64,
            }
            for opponent_id, archetype_id, nibble in controls
        ],
    }
    digest_by_id = {
        row["opponent_id"]: row["content_digest"] for row in registry["controls"]
    }
    monkeypatch.setattr(
        trainer,
        "_spec_payload",
        lambda spec: {
            "id": spec.id,
            "content_digest": digest_by_id[spec.id],
        },
    )
    iteration = 8
    specs = [SimpleNamespace(id=row[0]) for row in controls]
    jobs = trainer._build_research_control_jobs(
        n_games=1000,
        ckpt=Path("/tmp/candidate.pt"),
        digest="sha256:candidate",
        model_generation=9,
        decks=[("alakazam", [1] * 60)],
        specs=specs,
        seed=trainer.RESEARCH_CONTROL_SEED_OFFSET
        + iteration * trainer.ITERATION_SEED_STRIDE,
        game_timeout_s=10,
        mode="specialist",
        registry=registry,
        iteration=iteration,
    )
    plan = trainer._assert_research_control_jobs(
        jobs,
        expected_games=1000,
        registry=registry,
        iteration=iteration,
        root_seed=0,
        training_jobs=[
            {"seed": iteration * trainer.ITERATION_SEED_STRIDE + index}
            for index in range(8192)
        ],
        formal_games=2000,
        checkpoint_digest="sha256:candidate",
        active_gate_digests={"sha256:" + "f" * 64},
    )
    result_path = trainer._research_control_result_path(tmp_path, iteration)
    result = {
        "schema": trainer.RESEARCH_CONTROL_RESULT_SCHEMA,
        "iteration": iteration,
        "registry_id": registry["registry_id"],
        "registry_version": registry["version"],
        "checkpoint_digest": "sha256:candidate",
        "schedule_digest": plan["schedule_digest"],
        "seed_namespace": plan["seed_namespace"],
        "training_eligible": False,
        "replay_eligible": False,
        "diagnostic_only": True,
        "included_in_gate_pass": False,
        "gate_weight": 0.0,
        "formal_eval": False,
        "action_selection": "greedy",
        "games": 1000,
        "wins": 500.0,
        "draws": 0,
        "losses": 500,
        "win_rate": 0.5,
        "matchups": [
            {
                "opponent_id": opponent_id,
                "content_digest": digest_by_id[opponent_id],
                "games": 250,
                "wins": 125.0,
                "draws": 0,
                "losses": 125,
                "seat0": 125,
                "seat1": 125,
                "win_rate": 0.5,
            }
            for opponent_id, _archetype, _nibble in controls
        ],
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "seed_disjoint": True,
            "package_disjoint_from_active_gate": True,
            "replay_records_written": 0,
        },
        "result_path": str(result_path.resolve()),
    }
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(
        trainer,
        "_collect_wave",
        lambda **_kwargs: pytest.fail("a valid immutable result must be recovered"),
    )
    kwargs = {
        "run_dir": tmp_path,
        "iteration": iteration,
        "root_seed": 0,
        "n_games": 1000,
        "training_games": 8192,
        "formal_games": 2000,
        "ckpt": Path("/tmp/candidate.pt"),
        "digest": "sha256:candidate",
        "decks": [("alakazam", [1] * 60)],
        "specs": specs,
        "registry": registry,
        "active_gate_digests": {"sha256:" + "f" * 64},
        "game_timeout_s": 10,
        "n_workers": 1,
        "leaf_channel": None,
        "remote_farm": None,
        "worker_play": None,
        "worker_self_play": None,
        "mode": "specialist",
        "allow_remote_play": False,
    }
    assert trainer._research_control_measurement(**kwargs) == result
    assert trainer._research_control_measurement(**kwargs) == result


def test_seed_namespaces_fail_closed_before_iteration_190_collision() -> None:
    safe = trainer._assert_seed_namespace_contract(
        root_seed=0,
        iterations=190,
        games_per_iteration=8192,
        formal_games=2000,
        research_control_games=1000,
    )
    assert safe["disjoint"] is True
    with pytest.raises(RuntimeError, match="seed namespaces overlap"):
        trainer._assert_seed_namespace_contract(
            root_seed=0,
            iterations=191,
            games_per_iteration=8192,
            formal_games=2000,
            research_control_games=1000,
        )


def test_adaptive_weights_read_latest_active_gate_matchup_rows() -> None:
    ids = tuple(row[0] for row in GATE_ROWS)
    stale = {
        opponent_id: {"games": 250, "win_rate": 0.9}
        for opponent_id in ids
    }
    latest = [
        {"opponent_id": opponent_id, "games": 250, "wr": win_rate}
        for opponent_id, _archetype, _weight, win_rate in GATE_ROWS
    ]
    state = {
        "history": [
            {"completed": True, "raw_heldout_gate": {"per_opponent": stale}},
            {"completed": True, "raw_heldout_gate": {"matchups": latest}},
        ],
        "heldout_champion_evidence": {"per_opponent": stale},
    }
    assert trainer._latest_official_heldout_win_rates(state, ids) == {
        row[0]: row[3] for row in GATE_ROWS
    }


def test_formal_gate_builder_remains_training_ineligible_greedy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trainer, "_spec_payload", lambda spec: {"id": spec.id})
    captured: dict[str, list[dict]] = {}

    def fake_collect_wave(**kwargs):
        captured["jobs"] = kwargs["baseline_jobs"]
        return None, [], {}

    monkeypatch.setattr(trainer, "_collect_wave", fake_collect_wave)
    monkeypatch.setattr(trainer.paths, "OUTPUTS_DIR", tmp_path)
    specs = [SimpleNamespace(id=row[0]) for row in GATE_ROWS]
    _rows, _audit = trainer._heldout_eval(
        ckpt=Path("/tmp/candidate.pt"),
        digest="sha256:candidate",
        n_games=2000,
        decks=[("alakazam", [1] * 60)],
        official_specs=specs,
        seed=19_800_000,
        game_timeout_s=10,
        n_workers=1,
        leaf_channel=None,
        remote_farm=None,
        worker_play=None,
        worker_self_play=None,
        mode="specialist",
        opponent_ids=tuple(row[0] for row in GATE_ROWS),
    )
    assert len(captured["jobs"]) == 2000
    assert all(job["training_eligible"] is False for job in captured["jobs"])
    assert all(job["sample_actions"] is False for job in captured["jobs"])
    assert all(job["greedy"] is True for job in captured["jobs"])
