from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot.pure_rl.eval_public import OFFICIAL_BASELINE_IDS, aggregate_heldout_wr
from scripts import train_pure_rl


def _decks() -> list[tuple[str, list[int]]]:
    return [(f"deck-{i}", [i + 1] * 60) for i in range(3)]


def test_head_focus_batch_schedule_uses_8192_for_first_30_iterations() -> None:
    for iteration in (0, 1, 29):
        assert train_pure_rl._scheduled_train_decision_cap(
            iteration,
            steady_cap=32768,
            warmup_cap=8192,
            warmup_iterations=30,
        ) == (8192, "head_focus_warmup")
    assert train_pure_rl._scheduled_train_decision_cap(
        30,
        steady_cap=32768,
        warmup_cap=8192,
        warmup_iterations=30,
    ) == (32768, "steady_state")
    with pytest.raises(ValueError, match="must be paired"):
        train_pure_rl._scheduled_train_decision_cap(
            0,
            steady_cap=32768,
            warmup_cap=8192,
            warmup_iterations=0,
        )


def test_self_play_uses_cross_deck_matchups_and_both_seats() -> None:
    jobs, public = train_pure_rl._build_collect_jobs(
        n_games=18,
        ckpt=Path("/tmp/champion.pt"),
        digest="abc",
        model_generation=1,
        decks=_decks(),
        specs=[],
        seed=7,
        game_timeout_s=10,
        mode="core",
        self_play_frac=1.0,
    )
    assert not public
    assert all(job["our_deck"] != job["opp_deck"] for job in jobs)
    by_arch: dict[str, set[int]] = {}
    for job in jobs:
        by_arch.setdefault(job["archetype"], set()).add(job["our_seat"])
        assert job["target_provenance"]["behavior_checkpoint_digest"] == "abc"
    assert all(seats == {0, 1} for seats in by_arch.values())


def test_formal_eval_pairs_every_opponent_and_deck_across_seats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_pure_rl, "_spec_payload", lambda spec: {"id": spec.id}
    )
    specs = [
        SimpleNamespace(
            id=oid,
            name=oid,
            dir_name=oid,
            group="test",
            source="test",
            path=Path(f"/tmp/{oid}"),
        )
        for oid in ("opp-a", "opp-b")
    ]
    _self, jobs = train_pure_rl._build_collect_jobs(
        n_games=24,
        ckpt=Path("/tmp/champion.pt"),
        digest="abc",
        model_generation=1,
        decks=_decks(),
        specs=specs,
        seed=9,
        game_timeout_s=10,
        mode="core",
        self_play_frac=0.0,
        balanced_eval=True,
    )
    cells: dict[tuple[str, str], set[int]] = {}
    for job in jobs:
        cells.setdefault((job["opponent_id"], job["archetype"]), set()).add(
            job["our_seat"]
        )
    assert len(cells) == len(specs) * len(_decks())
    assert all(seats == {0, 1} for seats in cells.values())


def test_specialist_public_wave_targets_official_policies_without_losing_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_calls: dict[str, int] = {}

    def _payload(spec: SimpleNamespace) -> dict[str, str]:
        payload_calls[spec.id] = payload_calls.get(spec.id, 0) + 1
        return {"id": spec.id}

    monkeypatch.setattr(train_pure_rl, "_spec_payload", _payload)
    def _spec(opponent_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=opponent_id,
            name=opponent_id,
            dir_name=opponent_id,
            group="test",
            source="test",
            path=Path(f"/tmp/{opponent_id}"),
        )

    official = [_spec(opponent_id) for opponent_id in OFFICIAL_BASELINE_IDS]
    diverse = [_spec(f"public-{index}") for index in range(22)]
    schedule = train_pure_rl._interleaved_opponent_schedule(
        64_512,
        priority_specs=official,
        diverse_specs=diverse,
        priority_frac=0.50,
        seed=20260720,
        iteration=0,
    )
    groups: dict[str, int] = {}
    opponents: dict[str, int] = {}
    for spec, group in schedule:
        groups[group] = groups.get(group, 0) + 1
        opponents[spec.id] = opponents.get(spec.id, 0) + 1
    assert groups == {"diverse_public": 32_256, "official_target": 32_256}
    assert all(opponents[opponent_id] == 8_064 for opponent_id in OFFICIAL_BASELINE_IDS)
    diverse_counts = [opponents[spec.id] for spec in diverse]
    assert max(diverse_counts) - min(diverse_counts) <= 1

    # Exercise the exact 64.5k production schedule without materializing
    # 64.5k full jobs/deck copies. Every opponent's own occurrence counter,
    # rather than global group parity, owns seat alternation.
    occurrences: dict[tuple[str, str], int] = {}
    exact_seats: dict[str, dict[int, int]] = {}
    for spec, _group in schedule:
        key = ("alakazam", str(spec.id))
        occurrence = occurrences.get(key, 0)
        seat = train_pure_rl._public_training_seat(
            seed=20260720,
            iteration=0,
            archetype=key[0],
            opponent_id=key[1],
            occurrence=occurrence,
        )
        occurrences[key] = occurrence + 1
        bucket = exact_seats.setdefault(str(spec.id), {0: 0, 1: 0})
        bucket[seat] += 1
    assert all(
        abs(bucket[0] - bucket[1]) <= 1 for bucket in exact_seats.values()
    )
    assert all(
        exact_seats[opponent_id] == {0: 4_032, 1: 4_032}
        for opponent_id in OFFICIAL_BASELINE_IDS
    )

    _self, jobs = train_pure_rl._build_collect_jobs(
        n_games=128,
        ckpt=Path("/tmp/alakazam.pt"),
        digest="sha256:alakazam",
        model_generation=1,
        decks=[("alakazam", [1] * 60)],
        specs=diverse,
        priority_specs=official,
        priority_frac=0.50,
        seed=7,
        game_timeout_s=10,
        mode="specialist",
        self_play_frac=0.0,
    )
    assert len(jobs) == 128
    assert sum(
        job["target_provenance"]["opponent_training_group"]
        == "official_target"
        for job in jobs
    ) == 64
    assert all(job["training_eligible"] is True for job in jobs)
    assert all(job["sample_actions"] is True for job in jobs)
    group_seats: dict[str, set[int]] = {}
    opponent_seats: dict[str, dict[int, int]] = {}
    for job in jobs:
        provenance = job["target_provenance"]
        assert (
            provenance["seat_schedule"]
            == "per_opponent_archetype_alternating_v1"
        )
        group = provenance["opponent_training_group"]
        group_seats.setdefault(group, set()).add(job["our_seat"])
        bucket = opponent_seats.setdefault(job["opponent_id"], {0: 0, 1: 0})
        bucket[job["our_seat"]] += 1
    assert group_seats == {
        "diverse_public": {0, 1},
        "official_target": {0, 1},
    }
    assert all(
        abs(bucket[0] - bucket[1]) <= 1 for bucket in opponent_seats.values()
    )
    assert set(payload_calls) == {
        *(spec.id for spec in official),
        *(spec.id for spec in diverse),
    }
    assert set(payload_calls.values()) == {1}


def test_official_exploit_temperature_is_paired_by_seat_and_target_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_pure_rl, "_spec_payload", lambda spec: {"id": spec.id}
    )

    def _spec(opponent_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=opponent_id,
            name=opponent_id,
            dir_name=opponent_id,
            group="test",
            source="test",
            path=Path(f"/tmp/{opponent_id}"),
        )

    official = [_spec(opponent_id) for opponent_id in OFFICIAL_BASELINE_IDS]
    diverse = [_spec("public-other")]
    _self, jobs = train_pure_rl._build_collect_jobs(
        n_games=128,
        ckpt=Path("/tmp/alakazam.pt"),
        digest="sha256:alakazam",
        model_generation=1,
        decks=[("alakazam", [1] * 60)],
        specs=diverse,
        priority_specs=official,
        priority_frac=1.0,
        seed=17,
        game_timeout_s=10,
        mode="specialist",
        self_play_frac=0.0,
        collect_temperature=1.0,
        official_exploit_opponents=("iono",),
        official_exploit_frac=0.5,
        official_exploit_temperature=0.35,
    )
    iono = [job for job in jobs if job["opponent_id"] == "iono"]
    assert len(iono) == 32
    assert sum(job["action_temperature"] == 0.35 for job in iono) == 16
    for pair_start in range(0, len(iono), 2):
        pair = iono[pair_start : pair_start + 2]
        assert len(pair) == 2
        assert {job["our_seat"] for job in pair} == {0, 1}
        assert len({job["action_temperature"] for job in pair}) == 1
    assert all(
        job["action_temperature"] == 1.0
        for job in jobs
        if job["opponent_id"] != "iono"
    )
    assert all(
        job["target_provenance"]["action_temperature"]
        == job["action_temperature"]
        for job in jobs
    )

    # Even explicitly supplied exploit settings cannot contaminate formal eval.
    _self, heldout = train_pure_rl._build_collect_jobs(
        n_games=8,
        ckpt=Path("/tmp/alakazam.pt"),
        digest="sha256:alakazam",
        model_generation=1,
        decks=[("alakazam", [1] * 60)],
        specs=official,
        seed=99,
        game_timeout_s=10,
        mode="specialist",
        self_play_frac=0.0,
        balanced_eval=True,
        collect_temperature=1.0,
        official_exploit_opponents=("iono",),
        official_exploit_frac=1.0,
        official_exploit_temperature=0.35,
    )
    assert all(job["action_temperature"] == 1.0 for job in heldout)
    assert all(
        job["target_provenance"]["behavior_mode"]
        == "base_sampling_temperature_v1"
        for job in heldout
    )


def test_adaptive_official_targeting_focuses_exact_weakest_without_starvation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win_rates = {
        "dragapult-ex": 0.36,
        "iono": 0.172,
        "mega-abomasnow-ex": 0.36,
        "mega-lucario-ex": 0.568,
    }
    weights = train_pure_rl._adaptive_official_target_weights(
        OFFICIAL_BASELINE_IDS,
        win_rates,
        target_win_rate=0.50,
        minimum_share=0.05,
        gap_power=2.0,
    )
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(value >= 0.05 for value in weights.values())
    assert weights["iono"] > 0.60
    assert weights["dragapult-ex"] == pytest.approx(
        weights["mega-abomasnow-ex"]
    )
    assert weights["mega-lucario-ex"] == pytest.approx(0.05)

    monkeypatch.setattr(
        train_pure_rl, "_spec_payload", lambda spec: {"id": spec.id}
    )
    official = [
        SimpleNamespace(id=opponent_id)
        for opponent_id in OFFICIAL_BASELINE_IDS
    ]
    _self, jobs = train_pure_rl._build_collect_jobs(
        n_games=128,
        ckpt=Path("/tmp/alakazam.pt"),
        digest="sha256:alakazam",
        model_generation=1,
        decks=[("alakazam", [1] * 60)],
        specs=[],
        priority_specs=official,
        priority_frac=1.0,
        priority_weights=weights,
        seed=17,
        game_timeout_s=10,
        mode="specialist",
        self_play_frac=0.0,
    )
    counts = {
        opponent_id: sum(job["opponent_id"] == opponent_id for job in jobs)
        for opponent_id in OFFICIAL_BASELINE_IDS
    }
    assert sum(counts.values()) == 128
    assert counts["iono"] > 80
    assert min(counts.values()) >= 6
    for opponent_id in OFFICIAL_BASELINE_IDS:
        rows = [job for job in jobs if job["opponent_id"] == opponent_id]
        assert abs(
            sum(job["our_seat"] == 0 for job in rows)
            - sum(job["our_seat"] == 1 for job in rows)
        ) <= 1
        assert all(
            job["target_provenance"]["opponent_schedule"]
            == "adaptive_exact_heldout_gap_v1"
            for job in rows
        )


def test_adaptive_official_targeting_uses_latest_complete_exact_evidence() -> None:
    older = {
        opponent_id: {"games": 250, "win_rate": 0.25}
        for opponent_id in OFFICIAL_BASELINE_IDS
    }
    newest = {
        opponent_id: {"games": 250, "win_rate": 0.1 + index / 10}
        for index, opponent_id in enumerate(OFFICIAL_BASELINE_IDS)
    }
    state = {
        "history": [
            {"completed": True, "raw_heldout_gate": {"per_opponent": older}},
            {"completed": True, "raw_heldout_gate": {"per_opponent": newest}},
        ]
    }
    assert train_pure_rl._latest_official_heldout_win_rates(state) == {
        opponent_id: row["win_rate"] for opponent_id, row in newest.items()
    }


def test_commit_record_recovers_atomic_loop_pointer(tmp_path: Path) -> None:
    (tmp_path / "commits").mkdir()
    initial = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "r",
        "mode": "core",
        "next_iteration": 0,
        "last_completed_iteration": -1,
    }
    train_pure_rl._atomic_json(tmp_path / "loop_state.json", initial)
    committed = {
        **initial,
        "next_iteration": 1,
        "last_completed_iteration": 0,
    }
    (tmp_path / "commits" / "iter_00000.json").write_text(
        json.dumps(committed), encoding="utf-8"
    )
    assert train_pure_rl._load_loop_state(tmp_path) == committed
    assert json.loads((tmp_path / "loop_state.json").read_text()) == committed


def test_immutable_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "iter.json"
    train_pure_rl._write_json_exclusive(path, {"iteration": 1})
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        train_pure_rl._write_json_exclusive(path, {"iteration": 2})


def test_heldout_requires_confidence_coverage_floor_and_seat_balance() -> None:
    rows = []
    for opponent in OFFICIAL_BASELINE_IDS:
        for i in range(50):
            seat = i % 2
            rows.append(
                {"opponent_id": opponent, "our_seat": seat, "winner": seat}
            )
    passed = aggregate_heldout_wr(rows, target_wr=0.70, min_games=200)
    assert passed.passed
    assert passed.confidence_lower > 0.70
    assert passed.seat_games == {"seat0": 100, "seat1": 100}

    rows[0]["our_seat"] = 1
    unbalanced = aggregate_heldout_wr(rows, target_wr=0.70, min_games=200)
    assert not unbalanced.passed
    assert unbalanced.reason == "seat_imbalance"


def test_smoke_has_a_separate_artifact_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_pure_rl.paths, "OUTPUTS_DIR", tmp_path)
    production = train_pure_rl._run_dir("same-name")
    smoke = train_pure_rl._run_dir("same-name", smoke=True)
    assert production == tmp_path / "pure_rl" / "same-name"
    assert smoke == tmp_path / "pure_rl_smoke" / "same-name"
    assert production != smoke


def test_interrupted_iteration_is_quarantined_and_can_retry(tmp_path: Path) -> None:
    for name in ("shards", "checkpoints", "metrics", "eval", "commits"):
        (tmp_path / name).mkdir()
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "r",
        "mode": "core",
        "next_iteration": 1,
        "last_completed_iteration": 0,
    }
    partials = [
        tmp_path / "shards" / "iter_00001.jsonl",
        tmp_path / "checkpoints" / "iter_00001.pt",
        tmp_path / "eval" / "iter_00001.json",
        tmp_path / "metrics" / "iter_00001.json",
    ]
    for i, path in enumerate(partials):
        path.write_bytes(f"partial-{i}".encode())
    prior_metrics = {"iteration": 0, "games": 8}
    (tmp_path / "metrics" / "iter_00000.json").write_text(
        json.dumps(prior_metrics), encoding="utf-8"
    )
    (tmp_path / "metrics" / "latest.json").write_text(
        json.dumps({"iteration": 1, "games": 2}), encoding="utf-8"
    )

    failure = train_pure_rl._recover_interrupted_iteration(tmp_path, state)
    assert failure is not None and failure.is_file()
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["reason"] == "interrupted_before_append_only_commit"
    assert len(payload["artifacts"]) == 5
    assert all(not path.exists() for path in partials)
    assert json.loads((tmp_path / "metrics" / "latest.json").read_text()) == prior_metrics

    # Recovery is idempotent; immutable names are now free for a clean retry.
    assert train_pure_rl._recover_interrupted_iteration(tmp_path, state) is None
    for path in partials:
        assert not path.exists()


def test_completed_collection_is_receipted_and_resumed_without_recollection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("shards", "checkpoints", "metrics", "eval", "commits"):
        (tmp_path / name).mkdir()
    checkpoint = tmp_path / "learner.pt"
    checkpoint.write_bytes(b"learner")
    checkpoint_digest = train_pure_rl._sha256_file(checkpoint)
    contract = {
        "games": {"per_iteration": 2, "minimum_usable_fraction": 0.98},
        "learner": {"max_context": 64},
        "source": {"source_tree_sha256": "sha256:test", "git_head": None},
    }
    design_fingerprint = train_pure_rl._design_fingerprint(contract)
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "r",
        "mode": "core",
        "next_iteration": 1,
        "last_completed_iteration": 0,
        "design_fingerprint": design_fingerprint,
        "learner": {"path": str(checkpoint), "digest": checkpoint_digest},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "design_contract": contract,
                "design_fingerprint": design_fingerprint,
            }
        )
    )
    shard = tmp_path / "shards" / "iter_00001.jsonl"
    shard.write_text(
        "".join(
            json.dumps(
                {
                    "episode_id": f"e-{index}",
                    "target_provenance": {
                        "behavior_checkpoint_digest": checkpoint_digest,
                    },
                    "decisions": [
                        {"env_step": 0, "selected_index": 0, "n_options": 1}
                    ],
                },
                separators=(",", ":"),
            )
            + "\n"
            for index in range(2)
        )
    )
    (tmp_path / "iteration_runtime.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "phase": "collect",
                "started_at": shard.stat().st_mtime - 2.0,
                "checkpoint_digest": checkpoint_digest,
            }
        )
    )

    def valid_manifest(path: Path, **_kwargs):
        return {
            "records": 2,
            "sequences": 2,
            "dropped": 0,
            "covered_bytes": path.stat().st_size,
            "manifest_path": str(tmp_path / "cache" / "manifest.json"),
            "signature": {"source": str(path)},
        }

    monkeypatch.setattr(
        train_pure_rl, "validated_replay_cache_manifest", valid_manifest
    )
    receipt = train_pure_rl._ensure_recoverable_completed_collection(
        tmp_path, state, contract
    )
    assert receipt is not None and receipt["recovery_derived"] is True
    assert receipt["shard"]["games"] == 2
    assert receipt["shard"]["decisions"] == 2

    # Recovery preserves exactly the verified shard rather than quarantining it.
    assert train_pure_rl._recover_interrupted_iteration(tmp_path, state) is None
    assert shard.is_file()
    bundle = train_pure_rl._completed_collection_bundle(tmp_path, state, contract)
    assert bundle is not None and bundle["recovered"] is True
    assert bundle["writer"].n_games == 2
    assert bundle["writer"].n_decisions == 2
    behavior = train_pure_rl._collection_behavior_identity(bundle)
    assert behavior.path == str(checkpoint.resolve())
    assert behavior.digest == checkpoint_digest


def test_completed_collection_scan_rejects_duplicate_and_mixed_weights(
    tmp_path: Path,
) -> None:
    expected = "sha256:" + "a" * 64
    shard = tmp_path / "iter_00001.jsonl"

    def row(episode_id: str, digest: str) -> str:
        return json.dumps(
            {
                "episode_id": episode_id,
                "target_provenance": {
                    "behavior_checkpoint_digest": digest,
                },
                "decisions": [
                    {"env_step": 0, "selected_index": 0, "n_options": 1}
                ],
            },
            separators=(",", ":"),
        ) + "\n"

    shard.write_text(row("same", expected) + row("same", expected))
    with pytest.raises(RuntimeError, match="duplicate episode_id"):
        train_pure_rl._scan_completed_compact_shard(
            shard, expected_checkpoint_digest=expected
        )

    shard.write_text(row("first", expected) + row("second", "sha256:stale"))
    with pytest.raises(RuntimeError, match="stale/mixed behavior digest"):
        train_pure_rl._scan_completed_compact_shard(
            shard, expected_checkpoint_digest=expected
        )


def test_tampered_completed_collection_and_receipt_are_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("shards", "checkpoints", "metrics", "eval", "commits"):
        (tmp_path / name).mkdir()
    checkpoint = tmp_path / "learner.pt"
    checkpoint.write_bytes(b"learner")
    checkpoint_digest = train_pure_rl._sha256_file(checkpoint)
    contract = {
        "games": {"per_iteration": 1, "minimum_usable_fraction": 0.98},
        "learner": {"max_context": 64},
        "source": {"source_tree_sha256": "sha256:test", "git_head": None},
    }
    design_fingerprint = train_pure_rl._design_fingerprint(contract)
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "r",
        "mode": "core",
        "next_iteration": 1,
        "last_completed_iteration": 0,
        "design_fingerprint": design_fingerprint,
        "learner": {"path": str(checkpoint), "digest": checkpoint_digest},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "design_contract": contract,
                "design_fingerprint": design_fingerprint,
            }
        )
    )
    shard = tmp_path / "shards" / "iter_00001.jsonl"
    shard.write_text(
        json.dumps(
            {
                "episode_id": "episode-a",
                "target_provenance": {
                    "behavior_checkpoint_digest": checkpoint_digest,
                },
                "decisions": [
                    {"env_step": 0, "selected_index": 0, "n_options": 1}
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    (tmp_path / "iteration_runtime.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "phase": "collect",
                "started_at": shard.stat().st_mtime - 1.0,
                "checkpoint_digest": checkpoint_digest,
            }
        )
    )

    def valid_manifest(path: Path, **_kwargs):
        return {
            "records": 1,
            "sequences": 1,
            "dropped": 0,
            "covered_bytes": path.stat().st_size,
            "manifest_path": str(tmp_path / "cache" / "manifest.json"),
            "signature": {"source": str(path)},
        }

    monkeypatch.setattr(
        train_pure_rl, "validated_replay_cache_manifest", valid_manifest
    )
    receipt = train_pure_rl._ensure_recoverable_completed_collection(
        tmp_path, state, contract
    )
    assert receipt is not None
    receipt_path = Path(receipt["receipt_path"])
    original_stat = shard.stat()
    original = shard.read_bytes()
    replacement = (b"z" if original[:1] != b"z" else b"y") + original[1:]
    shard.write_bytes(replacement)
    os.utime(
        shard,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )

    assert (
        train_pure_rl._verified_completed_collection_receipt(
            tmp_path, state, contract
        )
        is None
    )
    assert (
        train_pure_rl._ensure_recoverable_completed_collection(
            tmp_path, state, contract
        )
        is None
    )
    failure = train_pure_rl._recover_interrupted_iteration(tmp_path, state)
    assert failure is not None and failure.is_file()
    assert not shard.exists()
    assert not receipt_path.exists()
    failure_payload = json.loads(failure.read_text())
    relative_paths = {
        row["relative_path"] for row in failure_payload["artifacts"]
    }
    assert relative_paths == {
        "shards/iter_00001.jsonl",
        "collection_receipts/iter_00001.json",
    }


def test_terminal_gate_marker_is_recreated_and_validated(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "champion.pt"
    checkpoint_path.write_bytes(b"immutable champion")
    from poke_bot.promotion import CheckpointIdentity

    champion = CheckpointIdentity.from_path(checkpoint_path)
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "r",
        "mode": "core",
        "champion": champion.as_dict(),
        "history": [
            {
                "iteration": 3,
                "completed": True,
                "stage_gate": {
                    "passed": True,
                    "win_rate": 0.75,
                    "confidence_lower": 0.71,
                    "games": 200,
                },
            }
        ],
    }
    marker = train_pure_rl._ensure_terminal_gate_marker(tmp_path, state)
    assert marker == tmp_path / "CORE_GATE_PASSED"
    assert train_pure_rl._ensure_terminal_gate_marker(tmp_path, state) == marker
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["checkpoint_digest"] == champion.digest

    marker.write_text(json.dumps({"iteration": 99}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="disagrees with committed ledger"):
        train_pure_rl._ensure_terminal_gate_marker(tmp_path, state)


def test_resume_rejects_design_drift() -> None:
    stored = {"games": {"per_iteration": 256}, "seed": 7}
    digest = train_pure_rl._design_fingerprint(stored)
    state = {"design_fingerprint": digest}
    manifest = {"design_fingerprint": digest, "design_contract": stored}
    assert (
        train_pure_rl._validate_design_fingerprint(
            state=state, manifest=manifest, current=dict(stored)
        )
        == digest
    )
    with pytest.raises(RuntimeError, match="design drift"):
        train_pure_rl._validate_design_fingerprint(
            state=state,
            manifest=manifest,
            current={"games": {"per_iteration": 512}, "seed": 7},
        )


def test_clean_boundary_migration_audits_operational_batch_update(
    tmp_path: Path,
) -> None:
    stored = {
        "learner": {"games_per_batch": 96, "max_decisions_per_batch": 8192},
        "source": {"source_tree_sha256": "sha256:old", "git_head": None},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 1,
        "last_completed_iteration": 0,
    }
    manifest = {
        "design_fingerprint": stored_digest,
        "design_contract": stored,
    }
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits" / "iter_00000.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = {
        # Mirrors the production correction after 20,480-decision batches
        # repeatedly exhausted Blackwell VRAM and poisoned the CUDA context.
        "learner": {"games_per_batch": 240, "max_decisions_per_batch": 12288},
        "source": {"source_tree_sha256": "sha256:new", "git_head": None},
    }

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="test throughput update",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    assert state["design_fingerprint"] == digest
    receipts = list((tmp_path / "design_migrations").glob("migration_*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["changed_paths"] == [
        "learner.games_per_batch",
        "learner.max_decisions_per_batch",
        "source.source_tree_sha256",
    ]
    assert (
        train_pure_rl._validate_or_migrate_design_fingerprint(
            run_dir=tmp_path,
            state=state,
            manifest=manifest,
            current=current,
            allow_clean_boundary_migration=False,
            migration_reason=None,
        )
        == digest
    )

    current_again = json.loads(json.dumps(current))
    current_again["source"]["source_tree_sha256"] = "sha256:newer"
    digest_again = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current_again,
        allow_clean_boundary_migration=True,
        migration_reason="second scheduler fix at same boundary",
    )
    assert digest_again == train_pure_rl._design_fingerprint(current_again)
    assert len(list((tmp_path / "design_migrations").glob("migration_*.json"))) == 2


def test_clean_boundary_migration_allows_explicit_measurement_deck_change(
    tmp_path: Path,
) -> None:
    stored = {
        "learner": {"games_per_batch": 240, "max_decisions_per_batch": 20480},
        "measurement_deck_distribution": {"decks": ["all"]},
        "source": {"source_tree_sha256": "sha256:old", "git_head": None},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 2,
        "last_completed_iteration": 1,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits" / "iter_00001.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = {
        "learner": {"games_per_batch": 240, "max_decisions_per_batch": 20480},
        "measurement_deck_distribution": {
            "decks": ["lucario", "alakazam", "starmie", "crustle"]
        },
        "source": {"source_tree_sha256": "sha256:new", "git_head": None},
    }

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="operator requested four-deck measurement pool",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert receipt["changed_paths"] == [
        "measurement_deck_distribution.decks",
        "source.source_tree_sha256",
    ]


def test_clean_boundary_migration_records_continuous_behavior_policy(
    tmp_path: Path,
) -> None:
    stored = {
        "collection": {},
        "learner": {"games_per_batch": 240, "max_decisions_per_batch": 32768},
        "source": {"source_tree_sha256": "sha256:old", "git_head": None},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 1,
        "last_completed_iteration": 0,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits" / "iter_00000.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = json.loads(json.dumps(stored))
    current["collection"]["behavior_policy"] = (
        "continuous_learner_after_h2h_safety_and_exact_audit_v2"
    )
    current["source"]["source_tree_sha256"] = "sha256:new"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="rollouts follow the safety-approved continuous learner",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert receipt["changed_paths"] == [
        "collection.behavior_policy",
        "source.source_tree_sha256",
    ]


def test_clean_boundary_migration_allows_versioned_expert_pointer() -> None:
    assert (
        "expert_rehearsal.rolling_manifest_pointer"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )


def test_clean_boundary_migration_allows_adaptive_official_targeting(
    tmp_path: Path,
) -> None:
    stored = {
        "collection": {
            "official_targeting": {
                "strategy": "uniform_v1",
                "minimum_share_per_opponent": 0.05,
                "gap_power": 2.0,
                "target_win_rate": 0.5,
                "formal_eval_disjoint": True,
            }
        },
        "learner": {
            "games_per_batch": 240,
            "max_decisions_per_batch": 8192,
        },
        "source": {"source_tree_sha256": "sha256:old", "git_head": None},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 3,
        "last_completed_iteration": 2,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits" / "iter_00002.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = json.loads(json.dumps(stored))
    current["collection"]["official_targeting"]["strategy"] = (
        "latest_exact_heldout_gap_v1"
    )
    current["source"]["source_tree_sha256"] = "sha256:new"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="adaptive official targeting after exact iteration 2",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert receipt["boundary_next_iteration"] == 3
    assert receipt["changed_paths"] == [
        "collection.official_targeting.strategy",
        "source.source_tree_sha256",
    ]


def test_source_snapshot_hashes_untracked_file_bytes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "new_module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = train_pure_rl._source_snapshot(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = train_pure_rl._source_snapshot(tmp_path)
    assert first["untracked_files"][0]["path"] == "new_module.py"
    assert first["untracked_files"][0]["sha256"] != second["untracked_files"][0]["sha256"]
    assert first["source_tree_sha256"] != second["source_tree_sha256"]


def test_checkpoint_contract_requires_pure_rl_and_exact_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poke_bot import checkpoint, features

    path = tmp_path / "seed.pt"
    path.write_bytes(b"checkpoint bytes")
    payload = {
        "model_config": train_pure_rl.model_config_dict(
            train_pure_rl.pure_rl_model_config()
        ),
        "provenance": {"feature_schema": features.FEATURE_SCHEMA_VERSION},
        "extra": {"pure_rl": True, "smoke": False},
    }
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        checkpoint,
        "assert_trusted_policy_checkpoint",
        lambda *_a, **_k: {
            "decision_context": "stateless",
            "provenance": payload["provenance"],
            "model_config": payload["model_config"],
        },
    )
    contract = train_pure_rl._checkpoint_contract(path, smoke=False)
    assert contract["decision_context"] == "stateless"
    assert contract["max_context"] == train_pure_rl.pure_rl_model_config().max_context

    payload["extra"] = {"pure_rl": False, "smoke": False}
    with pytest.raises(RuntimeError, match="not explicitly pure_rl"):
        train_pure_rl._checkpoint_contract(path, smoke=False)


def test_checkpoint_contract_uses_state_dict_when_saved_param_count_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poke_bot import checkpoint, features

    path = tmp_path / "candidate.pt"
    path.write_bytes(b"checkpoint bytes")
    payload = {
        "model_config": train_pure_rl.model_config_dict(
            train_pure_rl.pure_rl_model_config()
        ),
        "model_state_dict": {
            "policy.weight": torch.zeros(3, 4),
            "_feature_schema_version": torch.zeros(()),
        },
        "provenance": {"feature_schema": features.FEATURE_SCHEMA_VERSION},
        "extra": {"pure_rl": True, "smoke": False, "param_count": 7},
    }
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        checkpoint,
        "assert_trusted_policy_checkpoint",
        lambda *_a, **_k: {
            "decision_context": "stateless",
            "provenance": payload["provenance"],
            "model_config": payload["model_config"],
        },
    )

    contract = train_pure_rl._checkpoint_contract(path, smoke=False)

    assert contract["trainable_parameters"] == 12


def test_rl_train_step_rejects_false_parent_digest_before_loading(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent bytes")
    with pytest.raises(ValueError, match="parent_digest does not match"):
        train_pure_rl.rl_train_step(
            SimpleNamespace(),
            base_ckpt=parent,
            out_run_name="r",
            archetype_id="core",
            epochs=1,
            parent_digest="sha256:not-the-parent",
        )


def test_collect_jobs_and_learner_results_bind_both_digests(tmp_path: Path) -> None:
    from poke_bot.promotion import CheckpointIdentity

    champion_path = tmp_path / "champion.pt"
    opponent_path = tmp_path / "opponent.pt"
    candidate_path = tmp_path / "candidate.pt"
    champion_path.write_bytes(b"champion")
    opponent_path.write_bytes(b"opponent")
    candidate_path.write_bytes(b"candidate")
    champion = CheckpointIdentity.from_path(champion_path)
    opponent = CheckpointIdentity.from_path(opponent_path)
    candidate = CheckpointIdentity.from_path(candidate_path)
    jobs, _ = train_pure_rl._build_collect_jobs(
        n_games=2,
        ckpt=champion_path,
        digest=champion.digest,
        model_generation=1,
        decks=_decks(),
        specs=[],
        seed=1,
        game_timeout_s=10,
        mode="core",
        opponent_pool=[opponent],
        self_play_frac=1.0,
    )
    assert all(job["opponent_checkpoint_digest"] == opponent.digest for job in jobs)
    assert all(
        job["target_provenance"]["opponent_checkpoint_digest"] == opponent.digest
        for job in jobs
    )
    train_pure_rl._verify_learner_lineage(
        {
            "candidate_digest": candidate.digest,
            "parent_digest": champion.digest,
        },
        candidate=candidate,
        parent=champion,
    )
    with pytest.raises(RuntimeError, match="candidate digest"):
        train_pure_rl._verify_learner_lineage(
            {"candidate_digest": opponent.digest, "parent_digest": champion.digest},
            candidate=candidate,
            parent=champion,
        )
